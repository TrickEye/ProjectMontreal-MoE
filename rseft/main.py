#!/usr/bin/env python3
"""
R-SEFT: Reasoning-Specialized Expert Fine-Tuning
================================================

End-to-end pipeline for adapting OLMoE-1B-7B to GSM8K via:
  Phase 1: Router Unmasking
  Phase 2: RFAR-based Expert Identification
  Phase 3: Reasoning-Expert Fine-Tuning

Paper: "Reasoning at the Forking Paths: Identifying and Activating
        Specialized Experts in MoE Models via High-Entropy Token Analysis"

Usage:
    python main.py                          # run with default config
    python main.py --output_dir ./my_run    # custom output directory
    python main.py --top_k_experts 128      # tune more experts
    python main.py --skip_phase1            # skip router unmasking
    python main.py --eval_only ./path       # evaluate saved model only
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import torch

from config import RSEFTConfig
from data_utils import load_gsm8k
from model_utils import (
    load_model_and_tokenizer,
    freeze_all_except_gates,
    freeze_all_except_selected_experts,
    save_model,
    save_selected_experts,
    load_selected_experts,
    count_params,
    get_moe_layer_info,
)
from rfar import compute_rfar_scores, select_top_k_experts
from train import train
from evaluate import evaluate, evaluate_base_model


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_wandb(config, phase: str):
    """Initialize wandb logging if enabled."""
    if not config.use_wandb:
        return
    import wandb
    run_name = config.wandb_run_name or f"rseft-{phase}"
    wandb.init(
        project=config.wandb_project,
        name=run_name,
        config=config.__dict__,
        reinit=True,
    )


def run_phase1(config):
    """Phase 1: Router Unmasking."""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  Phase 1: Router Unmasking".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    setup_wandb(config, "phase1")

    # Load model and data
    model, tokenizer = load_model_and_tokenizer(config)
    train_texts, val_texts, test_dataset = load_gsm8k(config)

    # Inspect MoE structure
    moe_info = get_moe_layer_info(model)

    # Freeze all but router gates
    freeze_all_except_gates(model)

    # Train
    model = train(model, tokenizer, train_texts, val_texts, config, phase="phase1")

    # Save
    save_dir = f"{config.output_dir}/phase1_router_tuned"
    save_model(model, tokenizer, save_dir)

    # Quick eval
    print("\n  Quick evaluation after Phase 1...")
    accuracy, _ = evaluate(model, tokenizer, test_dataset, config, verbose=False)
    print(f"  Phase 1 accuracy: {accuracy:.4f} ({accuracy:.2%})")

    if config.use_wandb:
        import wandb
        wandb.log({"phase1/test_accuracy": accuracy})
        wandb.finish()

    return model, tokenizer, train_texts, val_texts, test_dataset, moe_info


def run_phase2(model, tokenizer, train_texts, config):
    """Phase 2: RFAR-based Expert Identification."""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  Phase 2: Expert Identification via RFAR".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    # Compute RFAR scores
    rfar_scores, entropy_threshold = compute_rfar_scores(
        model, tokenizer, train_texts, config
    )

    # Select top-K experts
    selected_experts = select_top_k_experts(
        rfar_scores, config.top_k_experts
    )

    # Save selection
    save_selected_experts(
        selected_experts, f"{config.output_dir}/selected_experts.json"
    )

    # Also save RFAR scores for analysis
    rfar_data = {
        "entropy_threshold": entropy_threshold,
        "scores": {
            f"layer{layer}_expert{expert}": score
            for (layer, expert), score in rfar_scores.items()
        },
    }
    with open(f"{config.output_dir}/rfar_scores.json", "w") as f:
        json.dump(rfar_data, f, indent=2)

    return selected_experts


def run_phase3(model, tokenizer, train_texts, val_texts, test_dataset, selected_experts, config):
    """Phase 3: Reasoning-Expert Fine-Tuning."""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  Phase 3: Reasoning-Expert Fine-Tuning".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    setup_wandb(config, "phase3")

    # Freeze all but selected experts
    freeze_all_except_selected_experts(model, selected_experts)

    # Train
    model = train(model, tokenizer, train_texts, val_texts, config, phase="phase3")

    # Save
    save_dir = f"{config.output_dir}/phase3_final"
    save_model(model, tokenizer, save_dir)

    # Evaluate
    print("\n  Evaluating Phase 3 result...")
    accuracy, results = evaluate(model, tokenizer, test_dataset, config)

    if config.use_wandb:
        import wandb
        wandb.log({"phase3/test_accuracy": accuracy})
        wandb.finish()

    return model, accuracy


def run_full_pipeline(config):
    """Run the complete R-SEFT pipeline."""
    print("\n" + "█" * 70)
    print("█" + "  R-SEFT: Reasoning-Specialized Expert Fine-Tuning".center(68) + "█")
    print("█" + f"  Model: {config.model_name}".center(68) + "█")
    print("█" + f"  Dataset: GSM8K".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" + f"  Phase 1 epochs: {config.phase1_epochs}".center(68) + "█")
    print("█" + f"  Phase 3 epochs: {config.phase3_epochs}".center(68) + "█")
    print("█" + f"  Top-K experts: {config.top_k_experts}".center(68) + "█")
    print("█" + f"  Entropy percentile: {config.entropy_percentile}".center(68) + "█")
    print("█" + f"  Mixed precision: {config.mixed_precision}".center(68) + "█")
    print("█" * 70)

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # ── Phase 1: Router Unmasking ─────────────────────────────────────────
    model, tokenizer, train_texts, val_texts, test_dataset, moe_info = run_phase1(config)

    # ── Phase 2: Expert Identification ────────────────────────────────────
    selected_experts = run_phase2(model, tokenizer, train_texts, config)

    # ── Phase 3: Expert Fine-Tuning ───────────────────────────────────────
    # Reload the Phase-1 model (so Phase 3 starts from router-tuned weights,
    # not from the model that was modified during RFAR collection)
    print("\n  Reloading Phase 1 model for Phase 3...")
    del model
    torch.cuda.empty_cache()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "no": torch.float32,
    }
    torch_dtype = dtype_map.get(config.mixed_precision, torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        f"{config.output_dir}/phase1_router_tuned",
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(f"{config.output_dir}/phase1_router_tuned")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model, final_accuracy = run_phase3(
        model, tokenizer, train_texts, val_texts, test_dataset, selected_experts, config
    )

    # ── Final Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("R-SEFT Pipeline Complete!")
    print("=" * 70)
    print(f"  Output directory: {config.output_dir}")
    print(f"  Selected experts saved to: {config.output_dir}/selected_experts.json")
    print(f"  Final model saved to: {config.output_dir}/phase3_final")
    print(f"  Final GSM8K accuracy: {final_accuracy:.4f} ({final_accuracy:.2%})")

    return final_accuracy


def main():
    parser = argparse.ArgumentParser(
        description="R-SEFT: Reasoning-Specialized Expert Fine-Tuning"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./rseft_output",
        help="Output directory for checkpoints and results",
    )
    parser.add_argument(
        "--top_k_experts",
        type=int,
        default=None,
        help="Number of experts to fine-tune in Phase 3",
    )
    parser.add_argument(
        "--phase1_epochs",
        type=int,
        default=None,
        help="Number of epochs for Phase 1",
    )
    parser.add_argument(
        "--phase3_epochs",
        type=int,
        default=None,
        help="Number of epochs for Phase 3",
    )
    parser.add_argument(
        "--entropy_percentile",
        type=float,
        default=None,
        help="Percentile threshold for forking path tokens",
    )
    parser.add_argument(
        "--rfar_sample_size",
        type=int,
        default=None,
        help="Number of examples to use for RFAR computation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (applied to both phases)",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode",
    )
    parser.add_argument(
        "--skip_phase1",
        action="store_true",
        help="Skip Phase 1 (use existing router-tuned model)",
    )
    parser.add_argument(
        "--skip_phase2",
        action="store_true",
        help="Skip Phase 2 (use existing expert selection)",
    )
    parser.add_argument(
        "--eval_only",
        type=str,
        default=None,
        help="Only evaluate a saved model at the given path, then exit",
    )
    parser.add_argument(
        "--base_eval",
        action="store_true",
        help="Evaluate the base (untuned) model for reference",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable WandB logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="rseft",
        help="WandB project name",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Override model name",
    )

    args = parser.parse_args()

    # Build config
    config = RSEFTConfig()

    # Override from CLI args
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.top_k_experts is not None:
        config.top_k_experts = args.top_k_experts
    if args.phase1_epochs is not None:
        config.phase1_epochs = args.phase1_epochs
    if args.phase3_epochs is not None:
        config.phase3_epochs = args.phase3_epochs
    if args.entropy_percentile is not None:
        config.entropy_percentile = args.entropy_percentile
    if args.rfar_sample_size is not None:
        config.rfar_sample_size = args.rfar_sample_size
    if args.batch_size is not None:
        config.phase1_batch_size = args.batch_size
        config.phase3_batch_size = args.batch_size
    if args.mixed_precision is not None:
        config.mixed_precision = args.mixed_precision
    if args.use_wandb:
        config.use_wandb = True
    if args.wandb_project:
        config.wandb_project = args.wandb_project
    if args.model_name:
        config.model_name = args.model_name

    # Set seeds
    set_seed(config.seed)

    # Check CUDA
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Running on CPU will be very slow.")
        config.mixed_precision = "no"

    # ── Eval-only mode ────────────────────────────────────────────────────
    if args.eval_only:
        print(f"Evaluation-only mode: loading model from {args.eval_only}")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            args.eval_only,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.eval_only)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        _, _, test_dataset = load_gsm8k(config)
        accuracy, _ = evaluate(model, tokenizer, test_dataset, config)
        print(f"\nAccuracy: {accuracy:.4f} ({accuracy:.2%})")
        return

    # ── Base evaluation mode ──────────────────────────────────────────────
    if args.base_eval:
        accuracy = evaluate_base_model(config)
        print(f"\nBase model accuracy: {accuracy:.4f} ({accuracy:.2%})")
        return

    # ── Skip phases: load existing artifacts ──────────────────────────────
    if args.skip_phase1:
        print("Skipping Phase 1. Loading existing router-tuned model...")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "no": torch.float32,
        }
        torch_dtype = dtype_map.get(config.mixed_precision, torch.float32)

        model = AutoModelForCausalLM.from_pretrained(
            f"{config.output_dir}/phase1_router_tuned",
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            f"{config.output_dir}/phase1_router_tuned"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if config.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        train_texts, val_texts, test_dataset = load_gsm8k(config)
    else:
        # ── Run Phase 1 ───────────────────────────────────────────────────
        model, tokenizer, train_texts, val_texts, test_dataset, moe_info = run_phase1(
            config
        )

    if args.skip_phase2:
        print("Skipping Phase 2. Loading existing expert selection...")
        selected_experts = load_selected_experts(
            f"{config.output_dir}/selected_experts.json"
        )
    else:
        # ── Run Phase 2 ───────────────────────────────────────────────────
        selected_experts = run_phase2(model, tokenizer, train_texts, config)

    # ── Run Phase 3 ───────────────────────────────────────────────────────
    # Reload Phase-1 model for clean Phase 3 start
    print("\n  Reloading Phase 1 model for Phase 3...")
    del model
    torch.cuda.empty_cache()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "no": torch.float32,
    }
    torch_dtype = dtype_map.get(config.mixed_precision, torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        f"{config.output_dir}/phase1_router_tuned",
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(f"{config.output_dir}/phase1_router_tuned")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model, final_accuracy = run_phase3(
        model, tokenizer, train_texts, val_texts, test_dataset, selected_experts, config
    )

    # ── Final Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("R-SEFT Pipeline Complete!")
    print("=" * 70)
    print(f"  Output directory: {config.output_dir}")
    print(f"  Final GSM8K accuracy: {final_accuracy:.4f} ({final_accuracy:.2%})")


if __name__ == "__main__":
    main()
