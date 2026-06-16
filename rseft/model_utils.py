"""
Model utilities for R-SEFT.

Handles:
  - Loading OLMoE models from HuggingFace
  - Freezing/unfreezing parameters using runtime-discovered naming
  - Registering forward hooks to capture gate/router scores
  - Parameter counting and inspection
"""

import json
import os
import torch
import torch.nn.functional as F
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def inspect_param_patterns(model, top_n: int = 30):
    """
    Discover and print the unique parameter name patterns in the model.

    Groups parameter names by their 'category' (last component)
    and prints statistics. Critical for debugging parameter freezing.
    """
    print("\n  ── Parameter Name Inspection ──")
    categories = defaultdict(list)

    for name, param in model.named_parameters():
        # Extract the last meaningful component as "category"
        parts = name.split(".")
        # Find a meaningful category: the last 2-3 parts
        if len(parts) >= 2:
            cat = ".".join(parts[-2:])  # e.g. "gate.weight", "q_proj.weight"
        else:
            cat = parts[-1]

        # Also track the full module path (without weight/bias)
        if "weight" in parts[-1] or "bias" in parts[-1]:
            module_path = ".".join(parts[:-1])
        else:
            module_path = name

        categories[cat].append((name, param.numel()))

    print(f"  Found {len(categories)} unique parameter categories:")
    for cat, entries in sorted(categories.items()):
        total_params = sum(n for _, n in entries)
        example = entries[0][0]
        print(f"    {cat:40s} → {len(entries):3d} params, {total_params:>12,} total  e.g. {example}")

    # Also print all parameter names that might be gate-related
    print("\n  ── Potential Gate/Router Parameters ──")
    gate_candidates = []
    for name, param in model.named_parameters():
        name_lower = name.lower()
        if any(kw in name_lower for kw in ["gate", "router"]):
            gate_candidates.append((name, param.numel()))

    if gate_candidates:
        for name, numel in gate_candidates:
            print(f"    {name}  ({numel:,})")
    else:
        print("    NONE FOUND! This is a problem — no 'gate' or 'router' params detected.")
        print("    Dumping first 20 parameter names:")
        for i, (name, _) in enumerate(model.named_parameters()):
            if i >= 20:
                break
            print(f"      {name}")

    print()
    return categories


# ── Parameter Freezing ─────────────────────────────────────────────────────────

def freeze_all(model):
    """Freeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = False


def find_gate_param_names(model):
    """
    Discover gate/router parameter names at runtime.

    Strategy:
    1. Look for nn.Linear modules whose name ends with '.gate'
    2. If none found, look for modules containing 'gate' that are Linear
    3. If still none, look for 'router' in module names
    4. Finally, substring-match parameter names for 'gate.weight' or 'router.weight'

    Returns:
        list of parameter name strings that belong to gate/routers
    """
    gate_param_names = []

    # Strategy 1: Find nn.Linear modules named "gate" (standard OLMoE convention)
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Check if this module name ends with '.gate'
            if module_name.endswith(".gate"):
                # This is a router gate — collect its parameter names
                for param_name, _ in model.named_parameters():
                    if param_name.startswith(module_name + "."):
                        gate_param_names.append(param_name)

    if gate_param_names:
        print(f"  Strategy 1 ('.gate' Linear modules): found {len(gate_param_names)} gate params")
        return gate_param_names

    # Strategy 2: substring match 'gate' in module name for Linear layers
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "gate" in module_name.lower():
            for param_name, _ in model.named_parameters():
                if param_name.startswith(module_name + "."):
                    gate_param_names.append(param_name)

    if gate_param_names:
        print(f"  Strategy 2 ('gate' in Linear name): found {len(gate_param_names)} gate params")
        return gate_param_names

    # Strategy 3: Look for 'router' in Linear module names
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "router" in module_name.lower():
            for param_name, _ in model.named_parameters():
                if param_name.startswith(module_name + "."):
                    gate_param_names.append(param_name)

    if gate_param_names:
        print(f"  Strategy 3 ('router' Linear): found {len(gate_param_names)} gate params")
        return gate_param_names

    # Strategy 4: substring match in parameter names (last resort)
    for name, _ in model.named_parameters():
        name_lower = name.lower()
        # Match 'gate.weight' or 'router.weight' but NOT 'gate_proj' (expert internals)
        if ".gate.weight" in name_lower and "gate_proj" not in name_lower:
            gate_param_names.append(name)
        elif ".router.weight" in name_lower:
            gate_param_names.append(name)

    if gate_param_names:
        print(f"  Strategy 4 (param name substring): found {len(gate_param_names)} gate params")
        return gate_param_names

    return gate_param_names


def freeze_all_except_gates(model):
    """
    Freeze everything except the router/gate parameters.

    Uses runtime discovery to find gate parameter names.
    Used in Phase 1 (Router Unmasking).

    Raises:
        RuntimeError: if no gate parameters are found
    """
    print("Freezing all parameters except router gates...")

    # First, discover gate parameter names
    gate_param_names = find_gate_param_names(model)

    if not gate_param_names:
        print("\n  ❌ ERROR: Could not find any gate/router parameters!")
        print("  Running parameter inspection for debugging...")
        inspect_param_patterns(model)
        raise RuntimeError(
            "No gate/router parameters found. Cannot run Phase 1 (Router Unmasking). "
            "Check the parameter inspection output above and update find_gate_param_names()."
        )

    print(f"  Found {len(gate_param_names)} gate parameter(s):")
    for name in gate_param_names[:10]:
        print(f"    {name}")
    if len(gate_param_names) > 10:
        print(f"    ... and {len(gate_param_names) - 10} more")

    # Freeze everything
    freeze_all(model)

    # Unfreeze only gate params
    gate_param_set = set(gate_param_names)
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if name in gate_param_set:
            param.requires_grad = True
            unfrozen_count += param.numel()

    if unfrozen_count == 0:
        raise RuntimeError(
            f"Gate param names were found ({gate_param_names[:5]}...) but unfreezing failed. "
            "Named parameter mismatch — gates may be on a different device or shard."
        )

    print(f"  Unfroze {unfrozen_count:,} gate parameters")
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

    # First, discover the expert parameter naming convention
    expert_prefixes_by_layer = defaultdict(list)
    for name, _ in model.named_parameters():
        # Match pattern: ...layers.{N}.mlp.experts.{M}...
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    continue
                # Look for 'experts' after 'layers.{N}'
                for j in range(i + 2, len(parts)):
                    if parts[j] == "experts" and j + 1 < len(parts):
                        try:
                            expert_idx = int(parts[j + 1])
                        except ValueError:
                            continue
                        prefix = ".".join(parts[: j + 2])  # up to ...experts.{M}
                        expert_prefixes_by_layer[(layer_idx, expert_idx)].append(prefix)
                        break
                break  # Only process one 'layers' occurrence per name

    # Unfreeze selected experts
    unfrozen_count = 0
    for layer_idx, expert_idx in selected_experts:
        prefixes = expert_prefixes_by_layer.get((layer_idx, expert_idx), [])
        if not prefixes:
            if verbose:
                print(f"  WARNING: No parameters found for Layer {layer_idx}, Expert {expert_idx}")
            continue

        for name, param in model.named_parameters():
            for prefix in prefixes:
                if name.startswith(prefix):
                    param.requires_grad = True
                    unfrozen_count += param.numel()
                    break

    if verbose:
        print(f"  Unfroze {unfrozen_count:,} parameters for selected experts")
        print_param_summary(model)

    if unfrozen_count == 0:
        raise RuntimeError(
            "No expert parameters were unfrozen! Check the selected_experts set "
            "and the model's parameter naming convention."
        )


# ── MoE Layer Inspection ───────────────────────────────────────────────────────

def get_moe_layer_info(model):
    """
    Inspect the model to find all MoE layers and their expert counts.

    Returns:
        list of dicts: [
            {
                'layer_idx': int,
                'num_experts': int,
                'gate_name': str,
                'expert_prefix': str,
            },
            ...
        ]
    """
    moe_layers = []
    for name, module in model.named_modules():
        if name.endswith(".gate") and isinstance(module, torch.nn.Linear):
            if module.out_features >= 8:  # MoE gates have many outputs
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
                    gate_name = name
                    # Build expert prefix: strip ".gate" and append ".experts"
                    expert_prefix = ".".join(parts[:-1]) + ".experts"
                    moe_layers.append({
                        "layer_idx": layer_idx,
                        "num_experts": module.out_features,
                        "gate_name": gate_name,
                        "expert_prefix": expert_prefix,
                    })

    if moe_layers:
        print(
            f"  Found {len(moe_layers)} MoE layers, "
            f"each with {moe_layers[0]['num_experts']} experts"
        )
    else:
        print("  WARNING: No MoE layers found via .gate Linear module detection!")
        print("  Attempting fallback detection...")
        # Fallback: count unique 'experts.{N}' patterns
        expert_indices = set()
        for name, _ in model.named_modules():
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part == "experts" and i + 1 < len(parts):
                    try:
                        expert_indices.add(int(parts[i + 1]))
                    except ValueError:
                        pass
        if expert_indices:
            print(f"  Fallback: found expert indices {sorted(expert_indices)[:10]}...")

    return moe_layers


# ── Gate Score Collection via Hooks ────────────────────────────────────────────

def register_gate_hooks(model):
    """
    Register forward hooks on every MoE gate to capture routing weights.

    Returns:
        gate_store: dict mapping layer_idx -> tensor of gate weights
        handles: list of hook handles
    """
    gate_store = {}
    handles = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
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
            if module.out_features >= 8:
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

    selected_experts: list/set of (layer_idx, expert_idx) tuples
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "selected_experts": [
            {"layer": int(layer), "expert": int(expert)}
            for layer, expert in sorted(selected_experts)
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
        selected.add((int(entry["layer"]), int(entry["expert"])))
    print(f"  Loaded {len(selected)} selected experts from {path}")
    return selected
