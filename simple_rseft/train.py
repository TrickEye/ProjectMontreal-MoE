"""
Training utilities for R-SEFT.

Two training stages:
  1. Router-only full-parameter FT: tune only the router gates (mlp.gate).
     Uses full fine-tuning (not LoRA) because custom gate modules (e.g.
     DeepSeek's MoEGate) are invisible to the PEFT library's module scanner.
     Router parameters are few enough to train fully without LoRA.
  2. Expert-selective LoRA: tune only the selected reasoning experts' parameters.
     Standard nn.Linear layers — PEFT LoRA works normally here.
"""

import os
import logging
import torch
import torch.nn as nn
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)


# ── Router / gate identification ───────────────────────────────────────────────

def _is_router_module(module_name: str, module: nn.Module,
                      model_type: str = "olmoe_1b_7b_instruct") -> bool:
    """
    Check whether a module is a router (gate) module that selects experts.

    Uses both the module's fully-qualified name and its class name to handle
    custom gate implementations (e.g. DeepSeek's MoEGate).
    """
    last_segment = module_name.rsplit(".", 1)[-1] if "." in module_name else module_name
    name_lower = module_name.lower()
    class_name = module.__class__.__name__

    # --- explicit name matches (most reliable) --------------------------------
    # OLMoE:           model.layers.{i}.mlp.gate
    # DeepSeekMoE:      model.layers.{i}.mlp.gate  or  .mlp.router
    if last_segment in ("gate", "router"):
        return True

    # Also catch nested sub-modules inside a gate/router parent, e.g.
    # model.layers.0.mlp.gate.linear_out  (custom MoEGate children).
    if ".gate." in name_lower or ".router." in name_lower:
        return True

    # --- class-name fallback (handles custom modules like MoEGate) ------------
    # Compare case-insensitively so MoEGate / moe_gate / Gate all match.
    class_lower = class_name.lower()
    if "gate" in class_lower or "router" in class_lower:
        return True

    return False


def _find_router_modules(model: nn.Module,
                         model_type: str = "olmoe_1b_7b_instruct") -> list[str]:
    """
    Return the fully-qualified names of all router/gate sub-modules.
    """
    router_names: list[str] = []
    for name, module in model.named_modules():
        if _is_router_module(name, module, model_type):
            router_names.append(name)
    return router_names


def _ensure_contiguous_params(model: nn.Module):
    """
    Ensure all parameters that require gradients are contiguous.

    Some 4-bit quantized models may have non-contiguous parameter tensors.
    This can cause issues with the optimizer; making them contiguous prevents
    cryptic "view size is not compatible" errors during the backward pass.
    """
    for _, param in model.named_parameters():
        if param.requires_grad and not param.is_contiguous():
            # Replace the parameter data with a contiguous copy while
            # preserving autograd history through the nn.Parameter wrapper.
            param.data = param.data.contiguous()


def prepare_stage1_model(model: nn.Module,
                         model_type: str = "olmoe_1b_7b_instruct") -> nn.Module:
    """
    Freeze every parameter in the model EXCEPT those belonging to router/gate
    modules.  Those are unfrozen for full-parameter fine-tuning.

    Returns the same model with ``requires_grad`` set appropriately.
    """
    # 1. Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # 2. Locate & unfreeze router parameters
    router_names = _find_router_modules(model, model_type)

    if not router_names:
        raise RuntimeError(
            f"No router/gate modules found for model_type={model_type!r}. "
            "Check _is_router_module — the naming convention may differ."
        )

    trainable_param_count = 0
    for rname in router_names:
        router_mod = model.get_submodule(rname)
        for _, param in router_mod.named_parameters():
            param.requires_grad = True
            trainable_param_count += param.numel()

    logger.info(
        "Stage 1 router full-FT: "
        f"{len(router_names)} router modules → {trainable_param_count:,} trainable params"
    )
    for rname in router_names:
        logger.info(f"  ✓ {rname}")

    # 3. Ensure gradients are contiguous (mitigates 4-bit quantisation issues)
    _ensure_contiguous_params(model)

    return model


# ── Expert target module builder (unchanged — still used for Stage 3 LoRA) ─────

def _module_names(model) -> list[str]:
    """Return all module names for runtime target probing."""
    if model is None:
        return []
    return [name for name, _ in model.named_modules()]


def _has_suffix(module_names: list[str], suffix: str) -> bool:
    """Whether any runtime module name contains the given suffix string."""
    return any(suffix in name for name in module_names)


def build_expert_target_modules(selected_experts: list, num_layers: int,
                                model_type: str = "olmoe_1b_7b_instruct",
                                include_shared_experts: bool = False,
                                model=None) -> list[str]:
    """
    Build target module list for specific experts across layers.

    OLMoE expert parameter naming:
      model.layers.{i}.mlp.experts.{e}.{gate_proj, up_proj, down_proj}

    DeepSeekMoE expert parameter naming:
      model.layers.{i}.mlp.routed_experts.{e}.{gate_proj, up_proj, down_proj}
      model.layers.{i}.mlp.shared_experts.{...}

    Args:
        selected_experts: list of length num_layers, each is array-like of expert indices
        num_layers: total MoE layers
        model_type: model family key
        include_shared_experts: whether to include shared_experts modules for DeepSeek
    Returns:
        list of module name substrings for PEFT LoRA targeting
    """
    names = _module_names(model)
    if model_type == "deepseek_moe_16b_chat":
        # Prefer routed_experts when available, fallback to experts for variants.
        if _has_suffix(names, ".mlp.routed_experts."):
            expert_prefix = "layers.{lv}.mlp.routed_experts.{e}"
        elif _has_suffix(names, ".mlp.experts."):
            expert_prefix = "layers.{lv}.mlp.experts.{e}"
        else:
            expert_prefix = "layers.{lv}.mlp.routed_experts.{e}"
    else:
        expert_prefix = "layers.{lv}.mlp.experts.{e}"

    targets = []
    for lv in range(num_layers):
        for e in selected_experts[lv]:
            base = expert_prefix.format(lv=lv, e=int(e))
            targets.append(f"{base}.gate_proj")
            targets.append(f"{base}.up_proj")
            targets.append(f"{base}.down_proj")

    if include_shared_experts and model_type == "deepseek_moe_16b_chat":
        # PEFT substring targeting for shared experts (all layers).
        if _has_suffix(names, "mlp.shared_experts") or not names:
            targets.extend([
                "mlp.shared_experts.gate_proj",
                "mlp.shared_experts.up_proj",
                "mlp.shared_experts.down_proj",
            ])

    return targets


# ── Training execution ─────────────────────────────────────────────────────────

def run_training(model, tokenizer, train_dataset, val_dataset,
                 output_dir: str, num_epochs: int = 3, lr: float = 3e-4,
                 per_device_batch_size: int = 1, grad_accum_steps: int = 4,
                 max_seq_length: int = 512, seed: int = 42,
                 remove_tmp: bool = True,
                 is_peft_model: bool = False) -> str:
    """
    Run fine-tuning with the HuggingFace Trainer.

    Args:
        model:
            For Stage 1 (``is_peft_model=False``): the raw model with
            ``requires_grad`` already set so only router parameters are
            trainable.
            For Stage 3 (``is_peft_model=True``):  a PEFT-wrapped model.
        tokenizer: tokenizer
        train_dataset: HF Dataset with 'input_ids', 'attention_mask', 'labels'
        val_dataset: HF Dataset
        output_dir: where to save the final weights
        num_epochs, lr, per_device_batch_size, grad_accum_steps,
            max_seq_length, seed: training hyper-parameters
        remove_tmp: whether to delete intermediate checkpoints
        is_peft_model: True → save with ``model.save_pretrained()``
                       False → save only trainable params as ``router_weights.pt``

    Returns:
        output_dir path
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="no",          # don't save intermediates for simplicity
        eval_strategy="no",
        seed=seed,
        data_seed=seed,
        fp16=True,
        report_to=None,              # no wandb
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,}")
    trainer.train()

    os.makedirs(output_dir, exist_ok=True)

    if is_peft_model:
        # PEFT adapter — uses its own serialisation
        model.save_pretrained(output_dir)
    else:
        # Full fine-tuning on a subset of parameters — save only those weights.
        router_state = {
            name: param.data.clone().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        ckpt_path = os.path.join(output_dir, "router_weights.pt")
        torch.save(router_state, ckpt_path)
        logger.info(
            "Saved %d trainable parameter tensors → %s",
            len(router_state), ckpt_path,
        )

    tokenizer.save_pretrained(output_dir)

    return output_dir


def load_stage1_router_weights(model: nn.Module, stage1_dir: str) -> nn.Module:
    """
    Load Stage‑1 full-FT router weights into *model* (in place).

    Args:
        model:   base model (same arch as the one that was trained).
        stage1_dir:  path containing ``router_weights.pt`` (written by
                     :func:`run_training` with ``is_peft_model=False``).
    Returns:
        *model* with the stored router weights loaded.
    """
    ckpt_path = os.path.join(stage1_dir, "router_weights.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Router checkpoint not found at {ckpt_path!r}. "
            "Did Stage 1 complete successfully?"
        )

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        logger.warning(
            "load_stage1_router_weights: %d missing keys (expected if "
            "router naming differs), e.g. %s", len(missing), missing[:3],
        )
    if unexpected:
        logger.warning(
            "load_stage1_router_weights: %d unexpected keys, e.g. %s",
            len(unexpected), unexpected[:3],
        )

    logger.info("Stage 1 router weights loaded from %s (%d tensors)",
                ckpt_path, len(state_dict))
    return model


# ── Stage-specific functions ───────────────────────────────────────────────────

def train_stage1_router(model, tokenizer, train_dataset, val_dataset,
                        save_path: str, model_type: str = "olmoe_1b_7b_instruct",
                        **train_kwargs) -> str:
    """
    Stage 1 — Router Unmasking:
    Full-parameter fine-tune ONLY the router gates on the reasoning dataset.

    This removes the pretrained load-balancing constraint and lets the router
    learn to map inputs to the most appropriate experts for reasoning.

    The router (gate) modules contain very few parameters relative to the
    whole model, so full fine-tuning is used instead of LoRA.  This also
    sidesteps the issue that custom gate implementations (e.g. DeepSeek's
    MoEGate) cannot be targeted by the PEFT library.
    """
    logger.info("=" * 60)
    logger.info("Stage 1: Router Unmasking (full FT on router gates only)")
    logger.info("=" * 60)

    model = prepare_stage1_model(model, model_type)

    output_dir = os.path.join(save_path, "stage1_router")
    run_training(model, tokenizer, train_dataset, val_dataset,
                 output_dir=output_dir,
                 is_peft_model=False,
                 **train_kwargs)

    logger.info(f"Stage 1 router weights saved to: {output_dir}")
    return output_dir


def train_stage3_experts(model, tokenizer, train_dataset, val_dataset,
                         selected_experts: list, num_layers: int,
                         save_path: str, model_type: str = "olmoe_1b_7b_instruct",
                         include_shared_experts: bool = False,
                         **train_kwargs) -> str:
    """
    Stage 3 — Reasoning Expert Fine-Tuning:
    LoRA fine-tune ONLY the selected reasoning experts' parameters.

    Args:
        selected_experts: list of per-layer expert indices (from RFAR analysis)
        num_layers: number of MoE layers
    """
    logger.info("=" * 60)
    logger.info("Stage 3: Reasoning Expert Fine-Tuning (LoRA on selected experts)")
    logger.info("=" * 60)

    target_modules = build_expert_target_modules(
        selected_experts,
        num_layers,
        model_type=model_type,
        include_shared_experts=include_shared_experts,
        model=model,
    )
    logger.info(f"Targeting {len(target_modules)} expert parameter groups "
                f"across {num_layers} layers")

    # Show which experts are selected per layer
    for lv in range(num_layers):
        logger.info(f"  Layer {lv}: experts {list(selected_experts[lv])}")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    output_dir = os.path.join(save_path, "stage3_experts")
    run_training(model, tokenizer, train_dataset, val_dataset,
                 output_dir=output_dir,
                 is_peft_model=True,
                 **train_kwargs)

    logger.info(f"Stage 3 adapter saved to: {output_dir}")
    return output_dir
