"""
RFAR (Reasoning Fork Activation Ratio) computation.

This is the core algorithm from the paper:
  1. Forward pass each sample, collecting router logits and prediction logits.
  2. Compute token-level entropy from prediction logits.
  3. Identify "forking tokens" — top 20% highest-entropy tokens.
  4. For each MoE expert, count how often it's activated on forking tokens
     vs. all tokens → RFAR = high_entropy_activations / total_activations.
  5. Select top-K experts per layer by RFAR score.
"""

import logging
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from data_utils import format_full_sample
from model_utils import get_model_info

logger = logging.getLogger(__name__)


# ── Entropy computation ──────────────────────────────────────────────────────

def calculate_token_entropy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Compute per-token prediction entropy H(P_t) = -Σ p(v) log₂ p(v).

    Args:
        logits: [B, S, V] output logits from the model
        temperature: softmax temperature (default 1.0)
    Returns:
        entropy: [B, S] per-token entropy in bits
    """
    logits = logits.float()  # cast to fp32 for numerical stability
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)
    log_probs = F.log_softmax(scaled, dim=-1)
    # entropy in bits (log base 2)
    entropy = -torch.sum(probs * log_probs, dim=-1) / torch.log(torch.tensor(2.0))
    entropy = torch.clamp(entropy, min=0.0)
    return entropy  # [B, S]


def identify_forking_tokens(entropy: torch.Tensor, percentile: float = 80.0) -> torch.Tensor:
    """
    Identify high-entropy "forking" tokens — the top (100-percentile)% by entropy.

    Args:
        entropy: [S] or [B, S] per-token entropy
        percentile: threshold percentile (default 80 → top 20% are forking)
    Returns:
        mask: [S] or [B, S] boolean tensor, True = forking token
    """
    flat = entropy.flatten()
    threshold = torch.quantile(flat, percentile / 100.0)
    return entropy > threshold


# ── Expert activation collection ─────────────────────────────────────────────

def topk_expert_mask(router_logits: torch.Tensor, top_k: int) -> np.ndarray:
    """
    Given router logits for one layer, return binary mask of which experts
    are in the top-k (by logit value).

    Args:
        router_logits: [S, E] raw gate logits for one layer
        top_k: number of active experts
    Returns:
        mask: [S, E] binary numpy array
    """
    top_indices = torch.argsort(router_logits, dim=-1)[:, -top_k:]  # [S, K]
    mask = np.zeros((router_logits.shape[0], router_logits.shape[1]), dtype=bool)
    for t in range(router_logits.shape[0]):
        mask[t, top_indices[t].cpu().numpy()] = True
    return mask


def collect_expert_activations(model, tokenizer, samples: list[dict], device: str = "cuda",
                               max_seq_length: int = 1024,
                               model_type: str = "olmoe_1b_7b_instruct"):
    """
    Forward pass over all samples, collecting per-layer, per-token expert activation
    masks and per-token entropy/forking masks.

    Returns:
        all_router_masks: list of [L, S, E] numpy arrays (one per sample, S varies)
        all_fork_masks:   list of [S] numpy arrays (one per sample)
        all_entropy:      list of [S] numpy arrays
        num_layers, num_experts, num_act_experts: int
    """
    model_info = get_model_info(model_type)
    num_layers = model_info["num_layers"]
    configured_num_experts = model_info["num_experts"]
    num_act_experts = model_info["num_act_experts"]

    all_router_masks = []
    all_fork_masks = []
    all_entropy = []

    model.eval()
    with torch.no_grad():
        for sample in tqdm(samples, desc="Collecting expert activations"):
            # Build full prompt+completion text
            if "reasoning" in sample and sample["reasoning"]:
                full_text = format_full_sample(
                    sample["question"], sample["reasoning"], sample["answer"],
                    model_type=model_type,
                )
            else:
                # If no reasoning available, use a minimal completion
                full_text = format_full_sample(
                    sample["question"], "", sample["answer"],
                    model_type=model_type,
                )

            inputs = tokenizer(
                [full_text],
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
            ).to(device)

            if inputs["input_ids"].shape[1] < 2:
                continue

            outputs = model(**inputs, output_router_logits=True, return_dict=True)

            # Router logits: tuple of [B=1, S, E] tensors, one per MoE layer
            router_logits_list = outputs.router_logits
            if len(router_logits_list) == 0:
                continue

            # Resolve runtime dimensions from router output to avoid stale hardcoding.
            runtime_num_layers = len(router_logits_list)
            runtime_num_experts = int(router_logits_list[0].shape[-1])
            num_layers = runtime_num_layers
            num_experts = runtime_num_experts
            # Prediction logits: [B=1, S, V]
            pred_logits = outputs.logits

            # Entropy on the *prediction* logits (shifted: logits[t] predicts token[t+1])
            # We align: entropy[t] uses logits[t-1] to predict token[t]
            # For simplicity, use logits at each position directly
            entropy = calculate_token_entropy(pred_logits[0])  # [S]

            # Forking mask: top-20% by entropy
            fork_mask = identify_forking_tokens(entropy)  # [S]

            # Build per-layer expert activation mask
            layer_masks = []
            for lv, rl in enumerate(router_logits_list):
                # rl: [1, S, E]
                emask = topk_expert_mask(rl[0].cpu(), min(num_act_experts, rl.shape[-1]))
                layer_masks.append(emask)

            # Stack: [L, S, E]
            sample_mask = np.stack(layer_masks, axis=0)
            all_router_masks.append(sample_mask)
            all_fork_masks.append(fork_mask.cpu().numpy())
            all_entropy.append(entropy.cpu().numpy())

    if not all_router_masks:
        raise RuntimeError("No valid samples produced router activations for RFAR.")

    # Fallback to configured metadata only when runtime is unavailable.
    if "num_experts" not in locals():
        num_experts = configured_num_experts

    return all_router_masks, all_fork_masks, all_entropy, num_layers, num_experts


# ── RFAR computation ─────────────────────────────────────────────────────────

def compute_rfar(all_router_masks: list, all_fork_masks: list,
                 num_layers: int, num_experts: int) -> np.ndarray:
    """
    Compute Reasoning Fork Activation Ratio for each expert.

    RFAR(E_i) = Σ_{t∈T_fork} I[E_i ∈ top-k at t] / Σ_{t∈T_all} I[E_i ∈ top-k at t]

    Args:
        all_router_masks: list of [L, S, E] binary expert activation masks
        all_fork_masks:   list of [S] boolean forking token masks
        num_layers, num_experts: int

    Returns:
        rfar_scores: [L, E] array of RFAR scores
        high_entropy_activations: [L, E]
        total_activations: [L, E]
        avg_fork_ratio: float (fraction of tokens that are forking)
    """
    high_entropy_activations = np.zeros([num_layers, num_experts], dtype=np.float64)
    total_activations = np.zeros([num_layers, num_experts], dtype=np.float64)
    total_tokens = 0
    total_fork_tokens = 0

    for sample_mask, fork_mask in zip(all_router_masks, all_fork_masks):
        # sample_mask: [L, S, E], fork_mask: [S]
        seq_len = sample_mask.shape[1]
        fork_idx = np.where(fork_mask[:seq_len])[0]
        total_tokens += seq_len
        total_fork_tokens += len(fork_idx)

        for lv in range(num_layers):
            layer_mask = sample_mask[lv]  # [S, E]
            total_activations[lv] += layer_mask.sum(axis=0)
            if len(fork_idx) > 0:
                high_entropy_activations[lv] += layer_mask[fork_idx].sum(axis=0)

    # Avoid division by zero
    total_activations = np.maximum(total_activations, 1e-10)
    rfar_scores = high_entropy_activations / total_activations

    avg_fork_ratio = total_fork_tokens / max(total_tokens, 1)

    return rfar_scores, high_entropy_activations, total_activations, avg_fork_ratio


def select_reasoning_experts(rfar_scores: np.ndarray, top_k: int = 2) -> list:
    """
    Select top-K reasoning experts per layer based on RFAR scores.

    Args:
        rfar_scores: [L, E] RFAR scores
        top_k: number of experts to select per layer
    Returns:
        selected: list of length L, each element is np.ndarray of K expert indices
    """
    num_layers = rfar_scores.shape[0]
    selected = []
    for lv in range(num_layers):
        indices = np.argsort(rfar_scores[lv])[-top_k:]
        selected.append(indices)
    return selected


# ── High-level API ───────────────────────────────────────────────────────────

def run_rfar_analysis(model, tokenizer, samples: list[dict], top_k: int = 2,
                      device: str = "cuda", max_seq_length: int = 1024,
                      model_type: str = "olmoe_1b_7b_instruct") -> dict:
    """
    Run complete RFAR analysis pipeline.

    Returns dict with:
        - rfar_scores: [L, E]
        - selected_experts: list of per-layer expert indices
        - high_entropy_activations: [L, E]
        - total_activations: [L, E]
        - avg_fork_ratio: float
    """
    logger.info("Step 1/3: Collecting expert activations...")
    all_router_masks, all_fork_masks, all_entropy, num_layers, num_experts = \
        collect_expert_activations(
            model,
            tokenizer,
            samples,
            device,
            max_seq_length,
            model_type=model_type,
        )

    logger.info("Step 2/3: Computing RFAR scores...")
    rfar_scores, he_act, tot_act, fork_ratio = compute_rfar(
        all_router_masks, all_fork_masks, num_layers, num_experts
    )

    logger.info(f"  Average forking token ratio: {fork_ratio:.3f}")
    logger.info(f"  RFAR score range: [{rfar_scores.min():.4f}, {rfar_scores.max():.4f}]")

    logger.info(f"Step 3/3: Selecting top-{top_k} reasoning experts per layer...")
    selected = select_reasoning_experts(rfar_scores, top_k)

    for lv in range(num_layers):
        sel_ids = selected[lv]
        sel_scores = rfar_scores[lv, sel_ids]
        experts_str = ", ".join(f"E{e}({s:.3f})" for e, s in zip(sel_ids, sel_scores))
        logger.info(f"  Layer {lv:2d}: {experts_str}")

    return {
        "rfar_scores": rfar_scores,
        "selected_experts": selected,
        "high_entropy_activations": he_act,
        "total_activations": tot_act,
        "avg_fork_ratio": fork_ratio,
        "num_layers": num_layers,
        "num_experts": num_experts,
    }
