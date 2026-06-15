"""
LoRA Gate Adapter for DeepSeek MoE Router Fine-Tuning.

Problem: DeepSeek's MoEGate is a custom nn.Module (not nn.Linear), so PEFT/LoRA
cannot target it. Under 4-bit quantization, full-parameter fine-tuning is infeasible.

Solution: LoRAGateAdapter wraps each MoEGate instance, freezes the original
quantized gate.weight, and adds two trainable float parameters lora_A and lora_B:

    W_eff = W_frozen + (alpha / r) * (B @ A)

where A ∈ R^{r × gating_dim}, B ∈ R^{n_routed_experts × r}.

The forward pass replicates MoEGate.forward exactly — including auxiliary
loss computation — using W_eff in place of the original weight. This couples
the adapter to DeepSeek's gate implementation, but keeps the original model
files untouched.

Usage:
  # Apply adapters to a loaded model
  apply_lora_to_moe_gates(model, lora_r=8, lora_alpha=32)

  # Train normally — only lora_A / lora_B receive gradients

  # Save / load
  save_gate_lora_weights(model, "./stage1_router/")
  load_gate_lora_weights(model, "./stage1_router/")
"""

import json
import math
import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── LoRA Gate Adapter ──────────────────────────────────────────────────────────

class LoRAGateAdapter(nn.Module):
    """
    LoRA adapter wrapping a DeepSeek MoEGate module.

    Freezes the original (quantized) gate.weight and exposes trainable
    lora_A / lora_B. The forward() method replicates MoEGate.forward
    using the effective (LoRA-augmented) weight matrix.

    Compatible with:
      - bitsandbytes 4-bit quantization (original weight stays quantized;
        LoRA params are float32)
      - torch.compile / fp16 training (LoRA matmul runs in compute dtype)
      - output_router_logits (DeepseekMoE handles that; adapter is transparent)
    """

    def __init__(
        self,
        original_gate: nn.Module,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ):
        super().__init__()

        # ── Snapshot all gate configuration attributes ──────────────────────
        # These are read from the original MoEGate instance and must match
        # the DeepSeek modeling code (v1 / 16B release).
        self.original_gate = original_gate

        self.top_k: int = original_gate.top_k
        self.n_routed_experts: int = original_gate.n_routed_experts
        self.scoring_func: str = original_gate.scoring_func
        self.alpha: float = original_gate.alpha
        self.seq_aux: bool = original_gate.seq_aux
        self.norm_topk_prob: bool = original_gate.norm_topk_prob
        self.gating_dim: int = original_gate.gating_dim

        # ── Freeze original gate ───────────────────────────────────────────
        for param in original_gate.parameters():
            param.requires_grad = False

        # ── LoRA hyperparameters ───────────────────────────────────────────
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_r
        # NOTE: dropout is stored for config round-tripping but not applied
        # in the weight-space formulation (W_eff = W + s·B·A). To apply
        # dropout you would need the input-space formulation:
        #   logits = W·x + s·B·A·dropout(x)
        self.lora_dropout_rate = lora_dropout

        # ── Trainable LoRA parameters (always float32, never quantized) ────
        # A ∈ R^{r × gating_dim}   — compresses hidden_dim → rank
        # B ∈ R^{n_experts × r}    — expands rank → n_experts
        self.lora_A = nn.Parameter(torch.empty(lora_r, self.gating_dim))
        self.lora_B = nn.Parameter(torch.empty(self.n_routed_experts, lora_r))

        self._reset_lora_parameters()

    def _reset_lora_parameters(self):
        """Kaiming-uniform init for A, zeros for B (standard LoRA convention)."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def _effective_weight(self) -> torch.Tensor:
        """
        Compute W_eff = W_frozen + (alpha / r) * (B @ A).

        The original gate.weight is accessed as an attribute — bitsandbytes
        dequantizes it on access, producing a float16 tensor. The LoRA delta
        is computed in float32 and cast appropriately by PyTorch's type
        promotion rules during the addition.
        """
        orig_weight = self.original_gate.weight  # dequantized, [E, H]
        delta = (self.lora_B @ self.lora_A) * self.scaling  # [E, H]
        return orig_weight + delta

    def forward(self, hidden_states: torch.Tensor):
        """
        Replicate MoEGate.forward with LoRA-augmented weight.

        This is coupled to the DeepSeek v1 MoEGate implementation and must
        be updated if the upstream modeling code changes its gate logic.

        Args:
            hidden_states: [B, S, H] input hidden states from the previous layer.

        Returns:
            topk_idx:    [T, K] expert indices for each token
            topk_weight: [T, K] routing weights for each token
            aux_loss:    scalar auxiliary load-balancing loss, or None
        """
        bsz, seq_len, h = hidden_states.shape

        # ── Flatten batch × sequence ───────────────────────────────────────
        hidden_states_flat = hidden_states.view(-1, h)  # [T, H] where T = B·S

        # ── Step 1: routing logits via effective weight ────────────────────
        weight = self._effective_weight()
        logits = F.linear(hidden_states_flat, weight, None)  # [T, E]

        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)  # token-to-expert affinity
        else:
            raise NotImplementedError(
                f"Unsupported scoring_func='{self.scoring_func}'. "
                f"LoRAGateAdapter currently only supports 'softmax'."
            )

        # ── Step 2: top-k expert selection ─────────────────────────────────
        topk_weight, topk_idx = torch.topk(
            scores, k=self.top_k, dim=-1, sorted=False
        )
        # topk_weight: [T, K], topk_idx: [T, K]

        # ── Step 3: optional top-k weight renormalization ───────────────────
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # ── Step 4: auxiliary load-balancing loss (training only) ───────────
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)  # [B, S·K]

            if self.seq_aux:
                # Sequence-level auxiliary loss
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(
                    bsz, self.n_routed_experts,
                    device=hidden_states.device,
                )
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(
                        bsz, seq_len * aux_topk,
                        device=hidden_states.device,
                    ),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (
                    (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean()
                    * self.alpha
                )
            else:
                # Token-level auxiliary loss
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1),
                    num_classes=self.n_routed_experts,
                )
                ce = mask_ce.float().mean(0)           # f_i  — expert frequency
                Pi = scores_for_aux.mean(0)             # P_i  — mean routing prob
                fi = ce * self.n_routed_experts         # f_i · N
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None

        return topk_idx, topk_weight, aux_loss

    @torch.no_grad()
    def merge_lora_to_original(self):
        """
        Merge LoRA weights into the original gate.weight (in-place on original).

        After merging, lora_A / lora_B are zeroed out, making the adapter
        equivalent to the original gate at inference time. Useful for
        exporting or when you no longer need trainability.

        NOTE: With 4-bit quantization, the merged weight is immediately
        re-quantized on the next .weight access if bitsandbytes manages
        the parameter. Test quantization fidelity before relying on this.
        """
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.original_gate.weight.data = self.original_gate.weight.data + delta
        nn.init.zeros_(self.lora_A)
        nn.init.zeros_(self.lora_B)

    def extra_repr(self) -> str:
        return (
            f"n_routed_experts={self.n_routed_experts}, "
            f"gating_dim={self.gating_dim}, "
            f"top_k={self.top_k}, "
            f"lora_r={self.lora_r}, lora_alpha={self.lora_alpha}"
        )


# ── Model-level operations ─────────────────────────────────────────────────────

def _find_moegate_modules(model: nn.Module) -> list[tuple[nn.Module, str]]:
    """
    Find all MoEGate instances in a model.

    Returns list of (parent_module, child_attribute_name) tuples.
    The search uses class-name matching to avoid importing DeepSeek types.
    """
    replacements: list[tuple[nn.Module, str]] = []

    for parent_name, parent_module in model.named_modules():
        for child_name, child_module in parent_module.named_children():
            if child_module.__class__.__name__ == "MoEGate":
                replacements.append((parent_module, child_name))
                logger.debug(
                    "Found MoEGate: %s.%s%s",
                    parent_name, child_name,
                    "" if parent_name else child_name,
                )

    return replacements


def apply_lora_to_moe_gates(
    model: nn.Module,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
) -> int:
    """
    Replace every MoEGate in *model* (in-place) with a LoRAGateAdapter.

    The original gate is frozen and stored inside the adapter as
    ``adapter.original_gate``.

    Args:
        model: a loaded HuggingFace model (modified in-place).
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor (effective scale = alpha / r).
        lora_dropout: dropout rate on hidden states before LoRA projection.

    Returns:
        Number of MoEGate instances that were adapted.
    """
    pairs = _find_moegate_modules(model)

    if not pairs:
        logger.warning(
            "No MoEGate modules found in model. "
            "If this is a DeepSeekMoE model, the gate class name may differ — "
            "check the model's modeling file."
        )
        return 0

    # ── Validate gate interface before wrapping ─────────────────────────────
    _GATE_REQUIRED_ATTRS = [
        "top_k", "n_routed_experts", "scoring_func", "alpha",
        "seq_aux", "norm_topk_prob", "gating_dim", "weight",
    ]

    for parent_module, child_name in pairs:
        original_gate = getattr(parent_module, child_name)

        # Verify the gate exports the expected DeepSeek v1 interface.
        missing = [a for a in _GATE_REQUIRED_ATTRS if not hasattr(original_gate, a)]
        if missing:
            raise AttributeError(
                f"MoEGate at '{child_name}' is missing required attributes: {missing}. "
                f"The DeepSeek MoE gate implementation may have changed from v1. "
                f"LoRAGateAdapter needs to be updated to match."
            )

        adapter = LoRAGateAdapter(
            original_gate,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        setattr(parent_module, child_name, adapter)

    logger.info(
        "Applied LoRA gate adapters to %d MoEGate instance(s) "
        "(r=%d, alpha=%d, dropout=%.2f)",
        len(pairs), lora_r, lora_alpha, lora_dropout,
    )
    return len(pairs)


def _collect_adapter_state(model: nn.Module) -> dict[str, dict]:
    """Collect lora_A, lora_B and metadata from all LoRAGateAdapter modules."""
    state: dict[str, dict] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRAGateAdapter):
            state[name] = {
                "lora_A": module.lora_A.data.cpu().clone(),
                "lora_B": module.lora_B.data.cpu().clone(),
                "lora_r": module.lora_r,
                "lora_alpha": module.lora_alpha,
                "lora_dropout": module.lora_dropout_rate,
            }
    return state


def save_gate_lora_weights(model: nn.Module, save_dir: str) -> str:
    """
    Save only the LoRA gate adapter weights (not the full model).

    Creates two files under *save_dir*:
      - ``gate_lora_weights.pt``   — state dict of all lora_A / lora_B
      - ``gate_lora_config.json``  — metadata (adapter type + count)

    Args:
        model: model with LoRAGateAdapter modules applied.
        save_dir: directory to save into.

    Returns:
        Path to the saved weights file.
    """
    os.makedirs(save_dir, exist_ok=True)

    state = _collect_adapter_state(model)

    if not state:
        raise RuntimeError(
            "No LoRAGateAdapter found in model — did you call "
            "apply_lora_to_moe_gates() first?"
        )

    weights_path = os.path.join(save_dir, "gate_lora_weights.pt")
    torch.save(state, weights_path)

    config = {
        "adapter_type": "lora_gate_adapter",
        "num_adapted_gates": len(state),
        "lora_r": next(iter(state.values()))["lora_r"],
        "lora_alpha": next(iter(state.values()))["lora_alpha"],
        "lora_dropout": next(iter(state.values()))["lora_dropout"],
    }
    config_path = os.path.join(save_dir, "gate_lora_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(
        "Saved %d LoRA gate adapter(s) to %s", len(state), weights_path,
    )
    return weights_path


def load_gate_lora_weights(model: nn.Module, load_dir: str) -> int:
    """
    Apply LoRAGateAdapter to a model and load saved LoRA weights.

    This is the inverse of saving via :func:`save_gate_lora_weights`.
    It first applies adapters (using hyperparameters from the saved config),
    then loads the trained lora_A / lora_B values.

    Args:
        model: a freshly-loaded base model (modified in-place).
        load_dir: directory containing gate_lora_weights.pt and
                  gate_lora_config.json.

    Returns:
        Number of adapted gates.
    """
    weights_path = os.path.join(load_dir, "gate_lora_weights.pt")
    config_path = os.path.join(load_dir, "gate_lora_config.json")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Gate LoRA weights not found at {weights_path}"
        )

    # Load config for hyperparameters
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        lora_r = config.get("lora_r", 8)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.1)
    else:
        # Fallback: try to infer from saved state
        lora_r = 8
        lora_alpha = 32
        lora_dropout = 0.1

    # Apply adapters first
    num_adapted = apply_lora_to_moe_gates(
        model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    if num_adapted == 0:
        logger.warning(
            "No MoEGate modules found to adapt. "
            "Is this a DeepSeekMoE model?"
        )
        return 0

    # Load saved weights
    saved_state = torch.load(weights_path, map_location="cpu", weights_only=True)

    loaded = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRAGateAdapter) and name in saved_state:
            module.lora_A.data.copy_(
                saved_state[name]["lora_A"].to(module.lora_A.device)
            )
            module.lora_B.data.copy_(
                saved_state[name]["lora_B"].to(module.lora_B.device)
            )
            loaded += 1

    logger.info(
        "Loaded LoRA gate weights for %d/%d adapter(s) from %s",
        loaded, num_adapted, weights_path,
    )
    return loaded


def is_gate_lora_adapter_dir(path: str) -> bool:
    """Check if *path* contains a gate LoRA adapter checkpoint."""
    if not os.path.isdir(path):
        return False
    config_path = os.path.join(path, "gate_lora_config.json")
    weights_path = os.path.join(path, "gate_lora_weights.pt")
    if os.path.exists(config_path) and os.path.exists(weights_path):
        return True
    return False


def get_trainable_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """
    Return (name, param) pairs for all trainable LoRA gate adapter parameters.
    Useful for debugging or custom optimizer setups.
    """
    trainable: list[tuple[str, nn.Parameter]] = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable.append((name, param))
    return trainable
