"""
Phase 2: Reasoning-Fork Activation Ratio (RFAR) computation.

This is the core of the R-SEFT method. It:
1. Runs the router-tuned model over training data to collect:
   - Per-token entropy (from the output distribution)
   - Per-token gate scores for each expert at each MoE layer
2. Identifies "forking path" tokens: those with entropy in the top percentile
3. Computes RFAR for each expert:
   RFAR(E_i) = sum_{t in T_fork} g_i(x_t) / sum_{t in T_total} g_i(x_t)
4. Selects the top-K experts with highest RFAR scores
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from model_utils import register_gate_hooks, remove_hooks


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute Shannon entropy from logits.

    Args:
        logits: tensor of shape (..., vocab_size)

    Returns:
        entropy: tensor of shape (...) in bits (log base 2)
    """
    probs = F.softmax(logits.float(), dim=-1)
    entropy = -(probs * torch.log2(probs + 1e-12)).sum(dim=-1)
    return entropy


def compute_rfar_scores(model, tokenizer, train_texts, config):
    """
    Compute RFAR scores for all experts across all MoE layers.

    Phase 2 of R-SEFT: Run the router-tuned model over a subset of the training
    data, collect per-token entropy and gate scores, then compute RFAR for each expert.

    Args:
        model: router-tuned OLMoE model
        tokenizer: tokenizer
        train_texts: list of formatted CoT training strings
        config: RSEFTConfig

    Returns:
        rfar_scores: dict mapping (layer_idx, expert_idx) -> RFAR score
        entropy_threshold: the entropy value at the configured percentile
    """
    print("\n" + "=" * 70)
    print("Phase 2: Computing RFAR (Reasoning-Fork Activation Ratio)")
    print("=" * 70)

    model.eval()
    device = next(model.parameters()).device

    # Subsample training data for efficiency
    sample_size = min(config.rfar_sample_size, len(train_texts))
    indices = np.random.RandomState(config.seed).choice(
        len(train_texts), size=sample_size, replace=False
    )
    sample_texts = [train_texts[i] for i in indices]
    print(f"  Processing {sample_size} examples for RFAR computation")

    # Register gate hooks
    gate_store, handles = register_gate_hooks(model)

    # Collect per-token entropy and gate scores
    all_entropies = []

    with torch.no_grad():
        for idx, text in enumerate(tqdm(sample_texts, desc="  Collecting gate scores & entropy")):
            # Tokenize a single example
            encoding = tokenizer(
                text,
                truncation=True,
                max_length=config.max_seq_length,
                return_tensors="pt",
            ).to(device)

            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

            # Forward pass (we need logits for entropy)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits  # (1, seq_len, vocab_size)

            # Compute per-token entropy
            # entropy shape: (1, seq_len)
            entropy = compute_entropy(logits).squeeze(0).cpu()  # (seq_len,)
            all_entropies.append(entropy)

            # Gate scores are automatically collected by hooks
            # Each entry in gate_store is a tensor of shape (num_tokens_so_far, num_experts)
            # Hooks fire for each forward pass, appending new tokens

            # Clear GPU cache periodically
            if (idx + 1) % 100 == 0:
                torch.cuda.empty_cache()

    # Remove hooks
    remove_hooks(handles)

    # Concatenate all entropies
    all_entropies = torch.cat(all_entropies, dim=0)  # (total_tokens,)

    # Determine entropy threshold (top percentile = forking path tokens)
    # entropy_percentile = 0.8 means top 20% of tokens by entropy
    entropy_threshold = float(
        np.percentile(all_entropies.numpy(), config.entropy_percentile * 100)
    )
    print(f"\n  Entropy threshold (top {(1 - config.entropy_percentile) * 100:.0f}%): {entropy_threshold:.4f} bits")

    # Identify forking path token indices
    fork_mask = all_entropies >= entropy_threshold  # (total_tokens,)
    total_tokens = len(all_entropies)
    fork_tokens = fork_mask.sum().item()
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Forking path tokens: {fork_tokens:,} ({100 * fork_tokens / total_tokens:.1f}%)")

    # Compute RFAR for each expert at each MoE layer
    rfar_scores = {}
    print(f"\n  Computing RFAR for {len(gate_store)} MoE layers...")

    for layer_idx, gate_weights in gate_store.items():
        # gate_weights shape: (total_tokens, num_experts)
        num_experts = gate_weights.shape[1]

        # Ensure gate_weights length matches entropy length
        # (they should, since hooks fire once per forward pass with the same number of tokens)
        min_len = min(gate_weights.shape[0], len(all_entropies))
        gate_weights = gate_weights[:min_len]
        fork_mask_layer = fork_mask[:min_len]

        for expert_idx in range(num_experts):
            expert_gate_scores = gate_weights[:, expert_idx]  # (total_tokens,)

            # Numerator: sum of gate scores at forking path tokens
            numerator = expert_gate_scores[fork_mask_layer].sum().item()

            # Denominator: sum of gate scores at all tokens
            denominator = expert_gate_scores.sum().item()

            if denominator > 0:
                rfar = numerator / denominator
            else:
                rfar = 0.0

            rfar_scores[(layer_idx, expert_idx)] = rfar

    # Statistics
    scores = list(rfar_scores.values())
    print(f"  RFAR score range: [{min(scores):.4f}, {max(scores):.4f}]")
    print(f"  Mean RFAR: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")

    if gate_store:
        # Check: experts with RFAR=1.0 (only active at fork tokens)
        pure_reasoning = sum(1 for s in scores if s > 0.99)
        print(f"  Experts with RFAR > 0.99 (pure reasoning): {pure_reasoning}")

    torch.cuda.empty_cache()
    return rfar_scores, entropy_threshold


def select_top_k_experts(rfar_scores, top_k: int, moe_layers_info=None):
    """
    Select the top-K experts with highest RFAR scores (global ranking).

    Args:
        rfar_scores: dict mapping (layer_idx, expert_idx) -> RFAR score
        top_k: number of experts to select
        moe_layers_info: optional list of MoE layer info for logging

    Returns:
        selected: set of (layer_idx, expert_idx) tuples
    """
    print(f"\n  Selecting top-{top_k} experts by RFAR (global ranking)...")

    # Sort experts by RFAR score (descending)
    sorted_experts = sorted(rfar_scores.items(), key=lambda x: x[1], reverse=True)

    # Select top-K
    selected = set()
    for (layer_idx, expert_idx), score in sorted_experts[:top_k]:
        selected.add((layer_idx, expert_idx))

    # Log selection statistics
    layer_counts = defaultdict(int)
    for layer_idx, _ in selected:
        layer_counts[layer_idx] += 1

    print(f"  Experts selected per layer:")
    for layer_idx in sorted(layer_counts.keys()):
        bar = "█" * max(1, layer_counts[layer_idx])
        print(f"    Layer {layer_idx:2d}: {layer_counts[layer_idx]:2d} experts {bar}")

    # Print top-10 experts
    print(f"\n  Top-10 experts by RFAR:")
    for rank, ((layer_idx, expert_idx), score) in enumerate(sorted_experts[:10]):
        print(f"    {rank + 1:2d}. Layer {layer_idx:2d}, Expert {expert_idx:2d}  RFAR={score:.4f}")

    return selected
