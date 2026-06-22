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
        # torch_dtype=torch.float16,
        # device_map="auto",
        trust_remote_code=True,
    )

    # Ensure generation config has pad_token_id set
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.do_sample = False

    return model, tokenizer

def save_router_weights(model, output_dir: str) -> str:
    """
    把 router（MoEGate.weight）的更新后权重单独保存到 output_dir。
    PEFT 的 save_pretrained 不会保存这些裸 nn.Parameter，必须手动处理。

    保存的 key 去掉 PEFT 注入的 'base_model.model.' 前缀，
    这样加载时不依赖 PEFT wrapper 的具体层级结构，更健壮。
    """
    os.makedirs(output_dir, exist_ok=True)

    router_state = {}
    for name, param in model.named_parameters():
        # 只保存 gate.weight，排除 lora_A / lora_B 等 PEFT 内部参数
        if "gate.weight" in name and "lora_" not in name:
            # 'base_model.model.model.layers.1.mlp.gate.weight'
            #  → 'model.layers.1.mlp.gate.weight'
            clean_name = name.replace("base_model.model.", "", 1)
            router_state[clean_name] = param.data.cpu().clone()

    save_path = os.path.join(output_dir, "router_weights.pt")
    torch.save(router_state, save_path)
    print(f"Saved {len(router_state)} router weight tensors → {save_path}")
    return save_path

def load_router_weights(model, checkpoint_dir: str) -> int:
    """
    把之前保存的 router 权重加载回模型。
    可以在任意时刻调用：加载 PEFT adapter 之后，或者直接在 base model 上。

    Returns:
        成功加载的 tensor 数量
    """
    load_path = os.path.join(checkpoint_dir, "router_weights.pt")
    if not os.path.exists(load_path):
        raise FileNotFoundError(
            f"找不到 router 权重文件: {load_path}\n"
            f"请确认 checkpoint_dir 指向正确的 Stage 1 输出目录。"
        )

    router_state = torch.load(load_path, map_location="cpu")

    loaded = 0
    for name, param in model.named_parameters():
        # 同样去掉前缀后匹配
        clean_name = name.replace("base_model.model.", "", 1)
        if clean_name in router_state:
            param.data.copy_(router_state[clean_name].to(param.device))
            loaded += 1

    if loaded == 0:
        raise RuntimeError(
            "加载了 0 个 router 权重，key 可能不匹配。\n"
            f"checkpoint 里的 key 示例: {list(router_state.keys())[:3]}\n"
            f"模型里的 gate 参数名示例: "
            f"{[n for n, _ in model.named_parameters() if 'gate.weight' in n][:3]}"
        )
    
    return loaded

