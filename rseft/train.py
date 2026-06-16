"""
Training loop for R-SEFT Phases 1 and 3.

Provides a clean training loop with:
  - Gradient accumulation
  - Mixed precision (AMP)
  - Gradient clipping
  - Learning rate scheduling (cosine with warmup)
  - Validation logging
  - Checkpointing
"""

import os
import math
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import numpy as np

from data_utils import create_dataloader


def get_optimizer_and_scheduler(model, config, phase: str):
    """
    Create optimizer and learning rate scheduler.

    Args:
        model: the model (only params with requires_grad=True are passed to optimizer)
        phase: "phase1" or "phase3"

    Returns:
        optimizer, scheduler
    """
    lr = config.phase1_lr if phase == "phase1" else config.phase3_lr
    warmup_ratio = (
        config.phase1_warmup_ratio
        if phase == "phase1"
        else config.phase3_warmup_ratio
    )
    weight_decay = (
        config.phase1_weight_decay
        if phase == "phase1"
        else config.phase3_weight_decay
    )
    epochs = config.phase1_epochs if phase == "phase1" else config.phase3_epochs
    batch_size = (
        config.phase1_batch_size if phase == "phase1" else config.phase3_batch_size
    )
    grad_accum = (
        config.phase1_grad_accum if phase == "phase1" else config.phase3_grad_accum
    )

    # Only optimize parameters that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  Optimizer training {sum(p.numel() for p in trainable_params):,} parameters")

    optimizer = AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    # Estimate total training steps
    # We'll get the actual dataloader length later; use a placeholder
    # The scheduler will be set up properly with total_steps

    return optimizer, None  # scheduler created later with total_steps known


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    scaler,
    config,
    phase: str,
    epoch: int,
    device: torch.device,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_tokens = 0

    grad_accum = (
        config.phase1_grad_accum if phase == "phase1" else config.phase3_grad_accum
    )
    max_grad_norm = (
        config.phase1_max_grad_norm
        if phase == "phase1"
        else config.phase3_max_grad_norm
    )

    use_amp = config.mixed_precision in ("fp16", "bf16")

    progress = tqdm(dataloader, desc=f"  Epoch {epoch}")
    optimizer.zero_grad()

    for step, batch in enumerate(progress):
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast(enabled=use_amp, dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16):
            outputs = model(**batch)
            loss = outputs.loss
            # Scale loss for gradient accumulation
            loss = loss / grad_accum

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        loss_item = loss.item() * grad_accum
        total_loss += loss_item * batch["input_ids"].size(0)
        total_tokens += batch["attention_mask"].sum().item()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        # Update progress bar
        progress.set_postfix(
            {
                "loss": f"{loss_item:.3f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
        )

        if (step + 1) % config.log_interval == 0 and config.use_wandb:
            import wandb
            wandb.log(
                {
                    f"{phase}/loss": loss_item,
                    f"{phase}/lr": optimizer.param_groups[0]["lr"],
                }
            )

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, config, device):
    """Run validation and return perplexity."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="  Validating", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast(
            enabled=config.mixed_precision in ("fp16", "bf16"),
            dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16,
        ):
            outputs = model(**batch)
            loss = outputs.loss

        total_loss += loss.item() * batch["input_ids"].size(0)
        total_tokens += batch["attention_mask"].sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    return avg_loss, perplexity


def train(model, tokenizer, train_texts, val_texts, config, phase: str):
    """
    Main training function for a single phase (Phase 1 or Phase 3).

    Args:
        model: the model to train (with appropriate params frozen/unfrozen)
        tokenizer: tokenizer
        train_texts: list of training text strings
        val_texts: list of validation text strings
        config: RSEFTConfig
        phase: "phase1" or "phase3"

    Returns:
        trained model
    """
    epochs = config.phase1_epochs if phase == "phase1" else config.phase3_epochs
    batch_size = (
        config.phase1_batch_size if phase == "phase1" else config.phase3_batch_size
    )
    lr = config.phase1_lr if phase == "phase1" else config.phase3_lr
    warmup_ratio = (
        config.phase1_warmup_ratio
        if phase == "phase1"
        else config.phase3_warmup_ratio
    )
    weight_decay = (
        config.phase1_weight_decay
        if phase == "phase1"
        else config.phase3_weight_decay
    )
    grad_accum = (
        config.phase1_grad_accum if phase == "phase1" else config.phase3_grad_accum
    )
    max_grad_norm = (
        config.phase1_max_grad_norm
        if phase == "phase1"
        else config.phase3_max_grad_norm
    )

    device = next(model.parameters()).device
    use_amp = config.mixed_precision in ("fp16", "bf16")

    print(f"\n{'─' * 50}")
    print(f"{'Phase 1: Router Unmasking' if phase == 'phase1' else 'Phase 3: Expert Fine-Tuning'}")
    print(f"{'─' * 50}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size} × {grad_accum} grad accum = effective {batch_size * grad_accum}")
    print(f"  Learning rate: {lr}")
    print(f"  Mixed precision: {config.mixed_precision}")

    # Create dataloaders
    train_loader = create_dataloader(
        train_texts, tokenizer, config, batch_size=batch_size, shuffle=True
    )
    val_loader = create_dataloader(
        val_texts, tokenizer, config, batch_size=batch_size, shuffle=False
    )

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    # Scheduler: linear warmup + cosine decay
    total_steps = len(train_loader) * epochs // grad_accum
    warmup_steps = int(total_steps * warmup_ratio)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=lr * 0.01,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # Gradient scaler for mixed precision
    scaler = GradScaler(enabled=config.mixed_precision == "fp16")

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        print(f"\n  ── Epoch {epoch}/{epochs} ──")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, config, phase, epoch, device
        )

        # Validate
        val_loss, val_ppl = validate(model, val_loader, config, device)

        print(
            f"  Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val ppl: {val_ppl:.2f}"
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_dir = f"{config.output_dir}/{phase}_best"
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"  ✓ Best model saved to {checkpoint_dir}")

        if config.use_wandb:
            import wandb
            wandb.log(
                {
                    f"{phase}/train_loss": train_loss,
                    f"{phase}/val_loss": val_loss,
                    f"{phase}/val_ppl": val_ppl,
                    f"{phase}/epoch": epoch,
                }
            )

    # Load best checkpoint
    checkpoint_dir = f"{config.output_dir}/{phase}_best"
    if os.path.exists(checkpoint_dir):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            torch_dtype=model.dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    return model
