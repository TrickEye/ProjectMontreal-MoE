"""
Data loading and preprocessing for GSM8K dataset.
"""

import torch
from datasets import load_dataset, Dataset
from model_utils import MATH_REASONING_TEMPLATE, format_chat


def load_gsm8k(split: str = "train", max_samples: int = -1):
    """
    Load GSM8K from HuggingFace datasets.

    Args:
        split: 'train' or 'test'
        max_samples: cap number of samples (-1 = all)
    Returns:
        list of dicts with keys: question, answer, reasoning
    """
    # GSM8K main split: 'train' has 7473 samples, 'test' has 1319
    hf_split = "train" if split == "train" else "test"
    ds = load_dataset("openai/gsm8k", "main", split=hf_split)

    samples = []
    for item in ds:
        # item['answer'] is "#### 14" format, item['question'] is the problem
        raw_answer = str(item.get("answer", "")).strip()
        if not raw_answer:
            continue
        answer_parts = raw_answer.split("####")
        reasoning = answer_parts[0].strip() if len(answer_parts) > 1 else ""
        answer = answer_parts[-1].strip()
        if not answer:
            continue
        samples.append({
            "question": str(item.get("question", "")).strip(),
            "reasoning": reasoning,
            "answer": answer,
        })

    if max_samples > 0:
        samples = samples[:max_samples]

    return samples


def format_prompt(question: str, model_type: str = "olmoe_1b_7b_instruct") -> str:
    """Build the user prompt for a GSM8K problem."""
    math_prompt = MATH_REASONING_TEMPLATE.format(q=question)
    return format_chat(user=math_prompt, assistant="", model_type=model_type)


def format_full_sample(question: str, reasoning: str, answer: str,
                       model_type: str = "olmoe_1b_7b_instruct") -> str:
    """Build the complete (prompt + completion) for training."""
    math_prompt = MATH_REASONING_TEMPLATE.format(q=question)
    completion = f"{reasoning}\n\nTherefore, the answer is \\boxed{{{answer}}}."
    return format_chat(user=math_prompt, assistant=completion, model_type=model_type)


def tokenize_batch(tokenizer, prompts: list[str], completions: list[str] = None,
                    max_length: int = 1024):
    # SFT 训练阶段强制右侧 padding，避免 left-padding 导致 mask 错位
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    try:
        if completions is not None:
            full_texts = [p + c for p, c in zip(prompts, completions)]
            tokenized = tokenizer(
                full_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            prompt_tokenized = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            labels = tokenized["input_ids"].clone()
            for i in range(len(prompts)):
                prompt_len = prompt_tokenized["attention_mask"][i].sum().item()
                labels[i, :prompt_len] = -100

            # 关键修复：把 padding 位置也设成 -100，不参与 loss 计算
            labels[tokenized["attention_mask"] == 0] = -100

            tokenized["labels"] = labels
        else:
            tokenized = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
    finally:
        tokenizer.padding_side = original_padding_side

    return Dataset.from_dict(tokenized)