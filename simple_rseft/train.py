"""
Training utilities for R-SEFT.

Two training stages:
  1. Router-only LoRA: tune only the router gates (mlp.gate).
  2. Expert-selective LoRA: tune only the selected reasoning experts' parameters.
"""

import os
import shutil
import logging
import torch
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


# ── LoRA target module builders ──────────────────────────────────────────────

def _module_names(model) -> list[str]:
    """Return all module names for runtime target probing."""
    if model is None:
        return []
    return [name for name, _ in model.named_modules()]


def _has_suffix(module_names: list[str], suffix: str) -> bool:
    """Whether any runtime module name contains the given suffix string."""
    return any(suffix in name for name in module_names)

def build_router_target_modules(model_type: str = "olmoe_1b_7b_instruct", model=None) -> list[str]:
    """
    Target all router gates for LoRA by model family.
    """
    names = _module_names(model)

    if model_type == "deepseek_moe_16b_chat":
        return ["gate_proj", "up_proj", "down_proj"]

    if _has_suffix(names, "mlp.gate"):
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
                 output_dir: str, num_epochs: int = 3, lr: float = 1e-5,
                 per_device_batch_size: int = 1, grad_accum_steps: int = 4,
                 max_seq_length: int = 512, seed: int = 42,
                 remove_tmp: bool = True) -> str:
    """
    Run LoRA fine-tuning with the given target modules already configured on the model.
    Uses HuggingFace Trainer with standard SFT settings.

    Args:
        model: PEFT-wrapped model (LoRA already applied)
        tokenizer: tokenizer
        train_dataset: HF Dataset with 'input_ids', 'attention_mask', 'labels'
        val_dataset: HF Dataset
        output_dir: where to save the final adapter
        num_epochs, lr, per_device_batch_size, grad_accum_steps, max_seq_length, seed
        remove_tmp: whether to delete intermediate checkpoints

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
        logging_steps=1,
        save_strategy="no",  # don't save intermediates for simplicity
        evaluation_strategy="no",
        seed=seed,
        data_seed=seed,
        bf16=True,
        report_to=None,  # no wandb
        remove_unused_columns=False,
    )

    from transformers import TrainerCallback
    import torch

    class RouterSanityCallback(TrainerCallback):
        """
        每隔 check_every 步，打印一次 router weight 的统计量，
        用来观察 router 在训练过程中是否在坍缩 / 数值爆炸。
        """
        def __init__(self, check_every: int = 10):
            self.check_every = check_every

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if model is None:
                return control
            if state.global_step % self.check_every != 0:
                return control

            for name, param in model.named_parameters():
                if "gate.weight" in name and param.requires_grad:
                    w = param.data
                    msg = (
                        f"[step {state.global_step}] {name}: "
                        f"norm={w.norm().item():.4f}, "
                        f"max={w.max().item():.4f}, "
                        f"min={w.min().item():.4f}, "
                        f"has_nan={torch.isnan(w).any().item()}, "
                        f"has_inf={torch.isinf(w).any().item()}"
                    )
                    print(msg)
                    break  # 只看第一层就够了，省得刷屏

            return control

    for name, param in model.named_parameters():
        if "gate.weight" in name and param.requires_grad:
            w = param.data
            print(f"[训练前] {name}: norm={w.norm().item():.4f}, "
                f"has_nan={torch.isnan(w).any().item()}, "
                f"has_inf={torch.isinf(w).any().item()}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        callbacks=[RouterSanityCallback(check_every=1)]
    )

    if False:
        logger.debug("Debugging")
        batch = next(iter(trainer.get_train_dataloader()))
        batch = {k: v.to(model.device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        print("outputs.loss:", outputs.loss)
        print("是否有 aux_loss 字段:", hasattr(outputs, "aux_loss"))
        if hasattr(outputs, "aux_loss"):
            print("aux_loss:", outputs.aux_loss)

        # 检查 labels 里到底有多少个有效 token（非 -100）
        labels = batch["labels"]
        valid_ratio = (labels != -100).float().mean().item()
        print(f"labels 中参与 loss 计算的 token 占比: {valid_ratio:.4f}")
        # 如果这个比例很低（比如 < 0.1），说明大部分位置被忽略，
        # 剩下的有效 token 很可能被模型迅速学成 trivial pattern

        # 看看真正参与 loss 的 token 都是什么内容
        sample_labels = labels[0]
        visible_ids = sample_labels[sample_labels != -100]
        print("第一条样本参与 loss 的 token 解码:", tokenizer.decode(visible_ids))
        logger.debug("Debugging done")

    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    trainer.train()

    # Save final adapter
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

    This removes the pretrained load-balancing constraint and lets the router
    learn to map inputs to the most appropriate experts for reasoning.
    """
    logger.info("=" * 60)
    logger.info("Stage 1: Router Unmasking (LoRA on router gates only)")
    logger.info("=" * 60)

    target_modules = build_router_target_modules(model_type=model_type, model=model)
    logger.info(f"Target modules: {target_modules}")

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            module.weight.requires_grad = False
        if name.endswith(".gate"):  # only unfreeze MoEGate
            if hasattr(module, "weight") and isinstance(module.weight, torch.nn.Parameter):
                module.weight.requires_grad = True
                logger.info(f"Unfreezing {name}, {module.weight.shape}")

    model.print_trainable_parameters()

    output_dir = os.path.join(save_path, "stage1_router")
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
    logger.info(f"{target_modules=}")

    model = prepare_model_for_kbit_training(model)

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
