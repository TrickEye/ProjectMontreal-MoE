"""
Model utilities for R-SEFT.

Handles:
  - Loading OLMoE models from HuggingFace
  - Freezing/unfreezing parameters by name pattern
  - Registering forward hooks to capture gate/router scores
  - Parameter counting and inspection
"""

import json
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


# ── Model Loading ──────────────────────────────────────────────────────────────

def load_model_and_tokenizer(config):
    """
    Load the OLMoE model and tokenizer.

    Returns:
        model: OLMoEForCausalLM
        tokenizer: AutoTokenizer
    """
    print(f"Loading model: {config.model_name}")

    tokenizer_name = config.tokenizer_name or config.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine torch dtype
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "no": torch.float32,
    }
    torch_dtype = dtype_map.get(config.mixed_precision, torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    print(f"  Model loaded. Total parameters: {count_params(model):,}")
    return model, tokenizer


# ── Parameter Inspection ───────────────────────────────────────────────────────

def count_params(model, trainable_only: bool = False) -> int:
    """Count total or trainable parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_param_summary(model):
    """Print summary of trainable vs frozen parameters."""
    total = count_params(model)
    trainable = count_params(model, trainable_only=True)
    print(f"  Total params: {total:,}")
    print(f"  Trainable:    {trainable:,} ({100 * trainable / total:.2f}%)")
    print(f"  Frozen:       {total - trainable:,}")


# ── Parameter Freezing ─────────────────────────────────────────────────────────

def freeze_all(model):
    """Freeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_by_name_pattern(model, patterns, verbose: bool = True):
    """
    Unfreeze parameters whose names contain any of the given patterns.

    Args:
        model: PyTorch model
        patterns: list of substrings to match against parameter names
        verbose: if True, print how many parameters were unfrozen
    """
    count = 0
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in patterns):
            param.requires_grad = True
            count += param.numel()

    if verbose:
        print(f"  Unfroze {count:,} parameters matching patterns: {patterns}")


def freeze_all_except_gates(model):
    """
    Freeze everything except the router/gate parameters.

    Used in Phase 1 (Router Unmasking).
    """
    print("Freezing all parameters except router gates...")
    freeze_all(model)
    # The gate is a Linear layer in each MoE block
    unfreeze_by_name_pattern(model, [".gate.", "gate."])
    print_param_summary(model)


def freeze_all_except_selected_experts(model, selected_experts, verbose: bool = True):
    """
    Freeze everything except the selected expert parameters.

    selected_experts: set of (layer_idx, expert_idx) tuples.

    Used in Phase 3 (Reasoning-Expert Fine-Tuning).
    """
    if verbose:
        print(f"Freezing all parameters except {len(selected_experts)} selected experts...")

    freeze_all(model)

    # Build a set of parameter name substrings to match
    unfrozen_count = 0
    for name, param in model.named_parameters():
        # Check if this parameter belongs to a selected expert
        for layer_idx, expert_idx in selected_experts:
            # Pattern looks like: model.layers.{layer_idx}.mlp.experts.{expert_idx}.
            layer_str = f".layers.{layer_idx}."
            expert_str = f".experts.{expert_idx}."
            if layer_str in name and expert_str in name:
                param.requires_grad = True
                unfrozen_count += param.numel()
                break

    if verbose:
        print(f"  Unfroze {unfrozen_count:,} parameters for selected experts")
        print_param_summary(model)


# ── MoE Layer Inspection ───────────────────────────────────────────────────────

def get_moe_layer_info(model):
    """
    Inspect the model to find all MoE layers and their expert counts.

    Returns:
        list of dicts: [
            {
                'layer_idx': int,
                'num_experts': int,
                'gate_name': str,       # e.g. 'model.layers.0.mlp.gate'
                'expert_prefix': str,   # e.g. 'model.layers.0.mlp.experts'
            },
            ...
        ]
    """
    moe_layers = []
    for name, module in model.named_modules():
        # Look for gate modules in MoE blocks
        if name.endswith(".gate") and isinstance(module, torch.nn.Linear):
            # Check if this is an MoE gate (has many output features = num_experts)
            if module.out_features >= 8:  # MoE gates typically have 8+ experts
                # Extract layer info
                parts = name.split(".")
                layer_idx = None
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                        except ValueError:
                            pass
                        break

                if layer_idx is not None:
                    expert_prefix = ".".join(parts[: parts.index("gate")]) + ".experts"
                    moe_layers.append(
                        {
                            "layer_idx": layer_idx,
                            "num_experts": module.out_features,
                            "gate_name": name,
                            "expert_prefix": expert_prefix,
                        }
                    )

    print(
        f"  Found {len(moe_layers)} MoE layers, "
        f"each with {moe_layers[0]['num_experts'] if moe_layers else 0} experts"
    )
    return moe_layers


# ── Gate Score Collection via Hooks ────────────────────────────────────────────

def register_gate_hooks(model):
    """
    Register forward hooks on every MoE gate to capture routing weights.

    The hooks capture the softmax-normalized gate scores (before top-k selection)
    for each token position.

    Returns:
        gate_store: dict mapping layer_idx -> list of gate weight tensors
        handles: list of hook handles (call .remove() on each to clean up)
    """
    gate_store = {}  # {layer_idx: tensor of shape (num_tokens, num_experts)}
    handles = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output shape: (batch*seq_len, num_experts) — router logits
            # Convert to softmax probabilities for meaningful gate scores
            gate_weights = F.softmax(output.float(), dim=-1).detach().cpu()
            if layer_idx in gate_store:
                gate_store[layer_idx] = torch.cat(
                    [gate_store[layer_idx], gate_weights], dim=0
                )
            else:
                gate_store[layer_idx] = gate_weights

        return hook_fn

    for name, module in model.named_modules():
        if name.endswith(".gate") and isinstance(module, torch.nn.Linear):
            if module.out_features >= 8:  # MoE gate
                parts = name.split(".")
                layer_idx = None
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                        except ValueError:
                            pass
                        break
                if layer_idx is not None:
                    h = module.register_forward_hook(make_hook(layer_idx))
                    handles.append(h)

    print(f"  Registered gate hooks on {len(handles)} MoE layers")
    return gate_store, handles


def remove_hooks(handles):
    """Remove all registered hooks."""
    for h in handles:
        h.remove()
    handles.clear()


# ── Saving & Loading ───────────────────────────────────────────────────────────

def save_model(model, tokenizer, path):
    """Save model and tokenizer to disk."""
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"  Model saved to {path}")


def save_selected_experts(selected_experts, path):
    """
    Save selected expert indices to a JSON file.

    selected_experts: list of (layer_idx, expert_idx) tuples
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "selected_experts": [
            {"layer": layer, "expert": expert}
            for layer, expert in selected_experts
        ],
        "total_selected": len(selected_experts),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Selected experts saved to {path}")


def load_selected_experts(path):
    """
    Load selected expert indices from a JSON file.

    Returns:
        set of (layer_idx, expert_idx) tuples
    """
    with open(path, "r") as f:
        data = json.load(f)
    selected = set()
    for entry in data["selected_experts"]:
        selected.add((entry["layer"], entry["expert"]))
    print(f"  Loaded {len(selected)} selected experts from {path}")
    return selected
