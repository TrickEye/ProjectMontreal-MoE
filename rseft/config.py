"""
Configuration for R-SEFT (Reasoning-Specialized Expert Fine-Tuning).

All hyperparameters for the three-phase training pipeline on OLMoE + GSM8K.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RSEFTConfig:
    # ── Model ──────────────────────────────────────────────
    model_name: str = "allenai/OLMoE-1B-7B-0924-Instruct"
    tokenizer_name: Optional[str] = None  # defaults to model_name

    # ── Dataset ────────────────────────────────────────────
    dataset_name: str = "openai/gsm8k"
    max_seq_length: int = 1024

    # ── Phase 1: Router Unmasking ─────────────────────────
    phase1_epochs: int = 2
    phase1_lr: float = 2e-4
    phase1_batch_size: int = 2
    phase1_grad_accum: int = 8
    phase1_warmup_ratio: float = 0.1
    phase1_weight_decay: float = 0.01
    phase1_max_grad_norm: float = 1.0

    # ── Phase 2: RFAR Expert Identification ───────────────
    # Entropy percentile for identifying "forking path" tokens.
    # 0.8 → tokens with entropy in the top 20th percentile.
    entropy_percentile: float = 0.8
    # Number of training examples to process for RFAR computation
    # (subset to keep memory/time manageable).
    rfar_sample_size: int = 1500
    rfar_batch_size: int = 1  # process one example at a time for gate-score collection
    rfar_max_new_tokens: int = 512  # max tokens to generate per example for entropy collection

    # ── Phase 3: Reasoning-Expert Fine-Tuning ─────────────
    # Number of experts to fine-tune (global selection across all layers).
    top_k_experts: int = 64
    phase3_epochs: int = 3
    phase3_lr: float = 1e-4
    phase3_batch_size: int = 2
    phase3_grad_accum: int = 8
    phase3_warmup_ratio: float = 0.1
    phase3_weight_decay: float = 0.01
    phase3_max_grad_norm: float = 1.0

    # ── General ────────────────────────────────────────────
    output_dir: str = "./rseft_output"
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16"
    use_gradient_checkpointing: bool = True
    seed: int = 42
    log_interval: int = 10

    # ── Evaluation ─────────────────────────────────────────
    eval_max_new_tokens: int = 512
    eval_batch_size: int = 4

    # ── WandB (optional) ───────────────────────────────────
    use_wandb: bool = False
    wandb_project: str = "rseft"
    wandb_run_name: Optional[str] = None
