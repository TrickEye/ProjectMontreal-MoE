"""
Model loading utilities for R-SEFT.
Supports multiple MoE backbones (OLMoE and DeepSeekMoE).
"""

import os
import random
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ── Model metadata ───────────────────────────────────────────────────────────

MODEL_INFO = {
    "olmoe_1b_7b_instruct": {
        "num_layers": 16,
        "num_experts": 64,
        "num_act_experts": 8,  # top-k
        "num_shared_experts": 0,
    },
    "deepseek_moe_16b_chat": {
        # DeepSeekMoE-16B: one dense layer + multiple MoE layers.
        # We keep router-related stats based on actual router_logits at runtime.
        "num_layers": 27,
        "num_experts": 64,
        "num_act_experts": 6,
        "num_shared_experts": 2,
    },
}

MODEL_ALIASES = {
    "OLMoE-1B-7B-0924-Instruct": "olmoe_1b_7b_instruct",
    "deepseek-ai/deepseek-moe-16b-chat": "deepseek_moe_16b_chat",
}

# Chat templates by model family
CHAT_TEMPLATES = {
    "olmoe_1b_7b_instruct": "<|endoftext|><|user|>\n{user}\n<|assistant|>\n{assistant}",
    "deepseek_moe_16b_chat": "User: {user}\n\nAssistant: {assistant}",
}

SEP_TOKENS = {
    "olmoe_1b_7b_instruct": "<|assistant|>\n",
    "deepseek_moe_16b_chat": "Assistant:",
}

# Backward-compatible exports for existing imports.
CHAT_TEMPLATE = CHAT_TEMPLATES["olmoe_1b_7b_instruct"]
SEP_TOKEN = SEP_TOKENS["olmoe_1b_7b_instruct"]

# Math reasoning prompt template
MATH_REASONING_TEMPLATE = """Problem:
{q}

Reasoning this step by step, and put your final answer within boxed {{}}."""


def infer_model_type(model_path: str, model_type: str = "auto") -> str:
    """Infer internal model_type key from CLI config and model path."""
    if model_type and model_type != "auto":
        return model_type

    if model_path in MODEL_ALIASES:
        return MODEL_ALIASES[model_path]

    lowered = model_path.lower()
    if "deepseek" in lowered:
        return "deepseek_moe_16b_chat"
    if "olmoe" in lowered:
        return "olmoe_1b_7b_instruct"

    raise ValueError(
        f"Cannot infer model_type from model_path='{model_path}'. "
        "Please pass --model_type explicitly."
    )


def get_model_info(model_type: str) -> dict:
    """Return model metadata for a normalized model_type key."""
    if model_type not in MODEL_INFO:
        raise KeyError(f"Unsupported model_type: {model_type}")
    return MODEL_INFO[model_type]


def get_chat_template(model_type: str) -> str:
    """Return the chat template string for the given model type."""
    if model_type not in CHAT_TEMPLATES:
        raise KeyError(f"Unsupported model_type for chat template: {model_type}")
    return CHAT_TEMPLATES[model_type]


def format_chat(user: str, assistant: str, model_type: str) -> str:
    """Format a single-turn chat for the selected model family."""
    template = get_chat_template(model_type)
    return template.format(user=user, assistant=assistant)


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_model_and_tokenizer(model_path: str, model_type: str = "auto"):
    """
    Load model with 4-bit quantization and tokenizer.

    Args:
        model_path: HuggingFace model ID or local path
        model_type: normalized model type key or "auto"
    Returns:
        (model, tokenizer)
    """
    resolved_model_type = infer_model_type(model_path, model_type)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Keep tokenizer chat template aligned with selected model family.
    tokenizer.chat_template = get_chat_template(resolved_model_type)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Ensure generation config has pad_token_id set
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.do_sample = False

    return model, tokenizer
