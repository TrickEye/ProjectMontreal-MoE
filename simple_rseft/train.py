"""
Training utilities for R-SEFT.

Two training stages:
  1. Router-only LoRA: tune only the router gates (mlp.gate).
  2. Expert-selective LoRA: tune only the selected reasoning experts' parameters.
"""

import os
import shutil
import math
import logging
import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from lora_gate_adapter import (
    apply_lora_to_moe_gates,
    save_gate_lora_weights,
)

logger = logging.getLogger(__name__)


def _has_custom_gate_adapters(model) -> bool:
    """Return True when the model contains LoRAGateAdapter modules."""
    return any(module.__class__.__name__ == "LoRAGateAdapter" for module in model.modules())


def _run_custom_gate_training(model, tokenizer, train_dataset,
                              output_dir: str, num_epochs: int = 3,
                              lr: float = 3e-4,
                              per_device_batch_size: int = 1,
                              grad_accum_steps: int = 4,
                              seed: int = 42,
                              skip_save: bool = False) -> str:
    """Train custom LoRAGateAdapter modules without HuggingFace Trainer."""
    del seed  # kept for signature parity with run_training

    def _collate_batch(features):
        batch = {}
        for key in features[0].keys():
            values = [feature[key] for feature in features]
            batch[key] = torch.stack([torch.as_tensor(value) for value in values])
        return batch

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        shuffle=True,
        collate_fn=_collate_batch,
    )
    device = next(model.parameters()).device
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if not trainable_params:
        raise ValueError("No trainable parameters found for custom gate training.")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    total_update_steps = max(1, math.ceil(len(train_loader) / max(1, grad_accum_steps)) * num_epochs)
    warmup_steps = max(0, int(total_update_steps * 0.1))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    use_amp = device.type == "cuda"
    amp_device_type = "cuda" if use_amp else "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    model.train()
    optimizer.zero_grad(set_to_none=True)
    update_step = 0

    logger.info("Using manual optimization loop for custom LoRAGateAdapter training")
    logger.info("Trainable parameters: %s", f"{sum(p.numel() for p in trainable_params):,}")

    for epoch in range(num_epochs):
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }

            with torch.amp.autocast(amp_device_type, enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.detach().float().item() * grad_accum_steps

            should_step = (step % grad_accum_steps == 0) or (step == len(train_loader))
            if should_step:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1

                if update_step % 10 == 0:
                    avg_loss = running_loss / max(1, min(step, len(train_loader)))
                    logger.info(
                        "Epoch %d/%d | update %d/%d | loss=%.4f",
                        epoch + 1,
                        num_epochs,
                        update_step,
                        total_update_steps,
                        avg_loss,
                    )

        logger.info(
            "Finished epoch %d/%d | avg_loss=%.4f",
            epoch + 1,
            num_epochs,
            running_loss / max(1, len(train_loader)),
        )

    if not skip_save:
        os.makedirs(output_dir, exist_ok=True)
        save_gate_lora_weights(model, output_dir)
        tokenizer.save_pretrained(output_dir)

    return output_dir


# ── LoRA target module builders ──────────────────────────────────────────────

def _module_names(model) -> list[str]:
    """Return all module names for runtime target probing."""
    if model is None:
        return []
    return [name for name, _ in model.named_modules()]


def _has_suffix(module_names: list[str], suffix: str) -> bool:
    """Whether any runtime module name contains the given suffix string."""
    return any(suffix in name for name in module_names)


def _find_linear_modules_in_parent(model, parent_name: str) -> list[str]:
    """Find all Linear module full paths within a parent module."""
    target_modules = []
    parent_module = None
    for name, mod in model.named_modules():
        if name == parent_name:
            parent_module = mod
            break

    if parent_module is not None:
        for name, mod in parent_module.named_modules():
            if isinstance(mod, torch.nn.Linear) and name != "":
                full_name = f"{parent_name}.{name}".lstrip(".")
                target_modules.append(full_name)

    return target_modules


def build_router_target_modules(model_type: str = "olmoe_1b_7b_instruct", model=None) -> list[str]:
    """
    Target all router gates for LoRA by model family.
    For custom modules like MoEGate, targets Linear layers within them.
    """
    names = _module_names(model)

    if model_type == "deepseek_moe_16b_chat":
        # Probe for router naming variants across DeepSeek implementations.
        if _has_suffix(names, "mlp.router"):
            linear_mods = _find_linear_modules_in_parent(model, "mlp.router")
            if linear_mods:
                return linear_mods
            return ["mlp.router"]
        if _has_suffix(names, "mlp.gate"):
            linear_mods = _find_linear_modules_in_parent(model, "mlp.gate")
            if linear_mods:
                return linear_mods
            return ["mlp.gate"]
        if _has_suffix(names, "gate"):
            linear_mods = _find_linear_modules_in_parent(model, "gate")
            if linear_mods:
                return linear_mods
            return ["gate"]

    if _has_suffix(names, "mlp.gate"):
        linear_mods = _find_linear_modules_in_parent(model, "mlp.gate")
        if linear_mods:
            return linear_mods
        return ["mlp.gate"]
    return ["mlp.gate"]


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


# ── Training execution ───────────────────────────────────────────────────────

def run_training(model, tokenizer, train_dataset, val_dataset,
                 output_dir: str, num_epochs: int = 3, lr: float = 3e-4,
                 per_device_batch_size: int = 1, grad_accum_steps: int = 4,
                 max_seq_length: int = 512, seed: int = 42,
                 remove_tmp: bool = True,
                 skip_save: bool = False) -> str:
    """
    Run fine-tuning with HuggingFace Trainer.

    Args:
        model: model with LoRA / adapter already applied (PEFT or custom).
        tokenizer: tokenizer
        train_dataset: HF Dataset with 'input_ids', 'attention_mask', 'labels'
        val_dataset: HF Dataset
        output_dir: where to save the final adapter
        num_epochs, lr, per_device_batch_size, grad_accum_steps, max_seq_length, seed
        remove_tmp: whether to delete intermediate checkpoints
        skip_save: if True, skip model.save_pretrained (caller handles saving).

    Returns:
        output_dir path
    """
    if _has_custom_gate_adapters(model):
        return _run_custom_gate_training(
            model,
            tokenizer,
            train_dataset,
            output_dir=output_dir,
            num_epochs=num_epochs,
            lr=lr,
            per_device_batch_size=per_device_batch_size,
            grad_accum_steps=grad_accum_steps,
            seed=seed,
            skip_save=skip_save,
        )

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
        save_strategy="no",  # don't save intermediates for simplicity
        eval_strategy="no",
        seed=seed,
        data_seed=seed,
        fp16=True,
        report_to=None,  # no wandb
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_count:,}")
    trainer.train()

    if not skip_save:
        # Save final adapter (PEFT path)
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    return output_dir


# ── Stage-specific functions ─────────────────────────────────────────────────

def train_stage1_router(model, tokenizer, train_dataset, val_dataset,
                        save_path: str, model_type: str = "olmoe_1b_7b_instruct",
                        **train_kwargs) -> str:
    """
    Stage 1 — Router Unmasking:
    LoRA fine-tune ONLY the router gates on the reasoning dataset.

    For DeepSeekMoE models (custom MoEGate, not nn.Linear), uses a
    LoRAGateAdapter that wraps each gate module — compatible with 4-bit
    quantization since the new lora_A / lora_B are float tensors.

    For OLMoE and other models with standard nn.Linear gates, uses PEFT LoRA.
    """
    logger.info("=" * 60)
    logger.info("Stage 1: Router Unmasking (LoRA on router gates only)")
    logger.info("=" * 60)

    output_dir = os.path.join(save_path, "stage1_router")

    if model_type == "deepseek_moe_16b_chat":
        # ── DeepSeek custom MoEGate path ────────────────────────────────────
        logger.info("DeepSeekMoE detected — using LoRAGateAdapter for router gates")

        # prepare_model_for_kbit_training enables gradient checkpointing,
        # casts LayerNorm to fp32, and (crucially) signals to the HuggingFace
        # Trainer that this quantized model is ready for fine-tuning.
        model = prepare_model_for_kbit_training(model)

        lora_r = train_kwargs.pop("lora_r", 8)
        lora_alpha = train_kwargs.pop("lora_alpha", 32)
        lora_dropout = train_kwargs.pop("lora_dropout", 0.1)

        num_adapted = apply_lora_to_moe_gates(
            model,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        logger.info(f"Adapted {num_adapted} MoEGate(s) with LoRA (r={lora_r})")

        # Train — skip the default save (quantized model can't be saved naively)
        run_training(model, tokenizer, train_dataset, val_dataset,
                     output_dir=output_dir, skip_save=True, **train_kwargs)

        # Save only the LoRA gate adapter weights
        save_gate_lora_weights(model, output_dir)
        tokenizer.save_pretrained(output_dir)

    else:
        # ── Standard PEFT LoRA path (OLMoE, etc.) ───────────────────────────
        target_modules = build_router_target_modules(model_type=model_type, model=model)
        logger.info(f"Target modules: {target_modules}")

        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)

        run_training(model, tokenizer, train_dataset, val_dataset,
                     output_dir=output_dir, **train_kwargs)

    logger.info(f"Stage 1 adapter saved to: {output_dir}")
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
                 output_dir=output_dir, **train_kwargs)

    logger.info(f"Stage 3 adapter saved to: {output_dir}")
    return output_dir
