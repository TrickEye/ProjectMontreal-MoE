#!/usr/bin/env python3
"""
R-SEFT: Reasoning Specialized Expert Fine-Tuning for MoE Models.

Complete pipeline:
  Stage 1 — Router Unmasking: LoRA fine-tune router gates on reasoning data
  Stage 2 — RFAR Analysis: Identify reasoning experts via high-entropy token analysis
  Stage 3 — Expert Fine-Tuning: LoRA only on selected reasoning experts
  Eval   — Compare base, router-tuned, and expert-tuned accuracy

Usage:
  python main.py --model_path allenai/OLMoE-1B-7B-0924-Instruct
  python main.py --model_path <path> --max_samples 200 --skip_stage1
"""

import os
import sys
import json
import logging
import argparse
import torch
import numpy as np

from model_utils import set_seed, load_model_and_tokenizer, infer_model_type
from data_utils import load_gsm8k, format_prompt, format_full_sample, tokenize_batch
from hook_rfar import run_rfar_analysis
from train import train_stage1_router, train_stage3_experts
from evaluate import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("rseft")


def parse_args():
    p = argparse.ArgumentParser(description="R-SEFT: Reasoning Expert Fine-Tuning")

    # Model & data
    p.add_argument("--model_path", type=str, required=True,
                   help="HuggingFace model ID or local path (e.g. allenai/OLMoE-1B-7B-0924-Instruct)")
    p.add_argument("--model_type", type=str, default="auto",
                   choices=["auto", "olmoe_1b_7b_instruct", "deepseek_moe_16b_chat"],
                   help="Model family key. Use auto to infer from model_path")
    p.add_argument("--max_samples", type=int, default=200,
                   help="Max training samples for hook/stages (smaller = faster)")
    p.add_argument("--max_seq_length", type=int, default=512,
                   help="Max token length for training and hook")
    p.add_argument("--eval_samples", type=int, default=50,
                   help="Max evaluation samples")
    p.add_argument("--seed", type=int, default=42)

    # Stage control
    p.add_argument("--skip_stage1", action="store_true", help="Skip router tuning")
    p.add_argument("--skip_stage2", action="store_true", help="Skip RFAR analysis")
    p.add_argument("--skip_stage3", action="store_true", help="Skip expert tuning")
    p.add_argument("--skip_eval", action="store_true", help="Skip final evaluation")

    # RFAR parameters
    p.add_argument("--top_k_experts", type=int, default=2,
                   help="Number of reasoning experts to select per layer")
    p.add_argument("--stage2_use_base", action="store_true",
                   help="Use base model for RFAR instead of router-tuned (default: use stage-1 model)")

    # Training hyperparameters
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--include_shared_experts", action="store_true",
                   help="For DeepSeekMoE only: also LoRA shared experts in stage 3")

    # Output
    p.add_argument("--save_path", type=str, default="./outputs",
                   help="Directory to save adapters and RFAR results")

    return p.parse_args()


def build_datasets(tokenizer, samples: list[dict], max_seq_length: int,
                   model_type: str):
    """Build tokenized train/val datasets from GSM8K samples."""
    # Simple 90/10 split
    split = int(len(samples) * 0.9)
    train_samples = samples[:split]
    val_samples = samples[split:]

    prompts_train = [format_prompt(s["question"], model_type=model_type) for s in train_samples]
    completions_train = []
    for s in train_samples:
        reasoning = s.get("reasoning", "")
        completions_train.append(
            f"{reasoning}\n\nTherefore, the answer is \\boxed{{{s['answer']}}}."
        )

    prompts_val = [format_prompt(s["question"], model_type=model_type) for s in val_samples]
    completions_val = []
    for s in val_samples:
        reasoning = s.get("reasoning", "")
        completions_val.append(
            f"{reasoning}\n\nTherefore, the answer is \\boxed{{{s['answer']}}}."
        )

    train_ds = tokenize_batch(tokenizer, prompts_train, completions_train, max_seq_length)
    val_ds = tokenize_batch(tokenizer, prompts_val, completions_val, max_seq_length)

    return train_ds, val_ds


def main():
    args = parse_args()
    resolved_model_type = infer_model_type(args.model_path, args.model_type)
    set_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    logger.info(f"Using model_type={resolved_model_type}")

    # ── Load data ─────────────────────────────────────────────────────────
    logger.info(f"Loading GSM8K dataset (max_samples={args.max_samples})...")
    train_samples = load_gsm8k("train", max_samples=args.max_samples)
    test_samples = load_gsm8k("test", max_samples=args.eval_samples)
    logger.info(f"  Train: {len(train_samples)} samples, Test: {len(test_samples)} samples")

    # ── Stage 1: Router Unmasking ─────────────────────────────────────────
    stage1_path = os.path.join(args.save_path, "stage1_router")

    if not args.skip_stage1:
        logger.info("=" * 60)
        logger.info("STAGE 1: Router Unmasking")
        logger.info("=" * 60)

        model, tokenizer = load_model_and_tokenizer(args.model_path, resolved_model_type)

        train_ds, val_ds = build_datasets(
            tokenizer,
            train_samples,
            args.max_seq_length,
            resolved_model_type,
        )

        train_stage1_router(
            model, tokenizer, train_ds, val_ds,
            save_path=args.save_path,
            model_type=resolved_model_type,
            num_epochs=args.num_epochs,
            lr=args.lr,
            per_device_batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum,
            max_seq_length=args.max_seq_length,
            seed=args.seed,
        )

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # ── Stage 2: RFAR Analysis ────────────────────────────────────────────
    rfar_path = os.path.join(args.save_path, "rfar_results.json")

    if not args.skip_stage2:
        logger.info("=" * 60)
        logger.info("STAGE 2: RFAR Analysis (identify reasoning experts)")
        logger.info("=" * 60)

        # Load the appropriate model for RFAR analysis
        model, tokenizer = load_model_and_tokenizer(args.model_path, resolved_model_type)

        if not args.stage2_use_base and os.path.exists(stage1_path):
            logger.info("Loading Stage-1 router adapter for RFAR analysis...")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, stage1_path) # load router-tuned model's router
        else:
            logger.info("Using base model for RFAR analysis")

        # Run RFAR on training samples
        rfar_results = run_rfar_analysis(
            model, tokenizer, train_samples,
            top_k=args.top_k_experts,
            device="cuda",
            max_seq_length=args.max_seq_length,
            model_type=resolved_model_type,
        )

        # Save RFAR results
        serializable = {
            "rfar_scores": rfar_results["rfar_scores"].tolist(),
            "selected_experts": [x.tolist() for x in rfar_results["selected_experts"]],
            "avg_fork_ratio": rfar_results["avg_fork_ratio"],
            "num_layers": rfar_results["num_layers"],
            "num_experts": rfar_results["num_experts"],
        }
        with open(rfar_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"RFAR results saved to: {rfar_path}")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # ── Stage 3: Expert Fine-Tuning ───────────────────────────────────────
    stage3_path = os.path.join(args.save_path, "stage3_experts")

    if not args.skip_stage3:
        logger.info("=" * 60)
        logger.info("STAGE 3: Reasoning Expert Fine-Tuning")
        logger.info("=" * 60)

        # Load selected experts from RFAR results
        if not os.path.exists(rfar_path):
            logger.error(f"RFAR results not found at {rfar_path}. Run Stage 2 first.")
            sys.exit(1)

        with open(rfar_path) as f:
            rfar_data = json.load(f)
        selected_experts = [np.array(x) for x in rfar_data["selected_experts"]]
        num_layers = rfar_data["num_layers"]

        # Fresh base model for Stage 3
        model, tokenizer = load_model_and_tokenizer(args.model_path, resolved_model_type)

        train_ds, val_ds = build_datasets(
            tokenizer,
            train_samples,
            args.max_seq_length,
            resolved_model_type,
        )

        train_stage3_experts(
            model, tokenizer, train_ds, val_ds,
            selected_experts=selected_experts,
            num_layers=num_layers,
            save_path=args.save_path,
            model_type=resolved_model_type,
            include_shared_experts=args.include_shared_experts,
            num_epochs=args.num_epochs,
            lr=args.lr,
            per_device_batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum,
            max_seq_length=args.max_seq_length,
            seed=args.seed,
        )

        del model
        torch.cuda.empty_cache()

    # ── Evaluation ────────────────────────────────────────────────────────
    if not args.skip_eval:
        logger.info("=" * 60)
        logger.info("EVALUATION")
        logger.info("=" * 60)

        # 1) Base model
        logger.info("--- Evaluating: Base Model ---")
        model, tokenizer = load_model_and_tokenizer(args.model_path, resolved_model_type)
        base_result = evaluate(model, tokenizer, test_samples,
                       max_seq_length=args.max_seq_length,
                       model_type=resolved_model_type)
        del model
        torch.cuda.empty_cache()

        # 2) Router-tuned model (Stage 1)
        if os.path.exists(stage1_path):
            logger.info("--- Evaluating: Router-Tuned Model ---")
            from peft import PeftModel
            model, tokenizer = load_model_and_tokenizer(args.model_path, resolved_model_type)
            model = PeftModel.from_pretrained(model, stage1_path)
            router_result = evaluate(model, tokenizer, test_samples,
                                     max_seq_length=args.max_seq_length,
                                     model_type=resolved_model_type)
            del model
            torch.cuda.empty_cache()
        else:
            router_result = None

        # 3) Expert-tuned model (Stage 3)
        if os.path.exists(stage3_path):
            logger.info("--- Evaluating: Expert-Tuned Model ---")
            from peft import PeftModel
            model, tokenizer = load_model_and_tokenizer(args.model_path, resolved_model_type)
            model = PeftModel.from_pretrained(model, stage3_path)
            expert_result = evaluate(model, tokenizer, test_samples,
                                     max_seq_length=args.max_seq_length,
                                     model_type=resolved_model_type)
            del model
            torch.cuda.empty_cache()
        else:
            expert_result = None

        # ── Summary ───────────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Base model accuracy:         {base_result['accuracy']:.4f}")
        if router_result:
            logger.info(f"  Router-tuned accuracy:       {router_result['accuracy']:.4f}")
        if expert_result:
            logger.info(f"  Expert-tuned accuracy:       {expert_result['accuracy']:.4f}")

        summary = {
            "base": base_result["accuracy"],
            "router_tuned": router_result["accuracy"] if router_result else None,
            "expert_tuned": expert_result["accuracy"] if expert_result else None,
        }
        summary_path = os.path.join(args.save_path, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nSummary saved to: {summary_path}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
