"""
GSM8K data loading and preprocessing utilities for R-SEFT.

Handles:
  - Loading the GSM8K dataset from HuggingFace
  - Formatting with chain-of-thought prompt templates
  - Tokenization and batching
"""

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


# ── Prompt Template ────────────────────────────────────────────────────────────
# Consistent with the paper: zero-shot CoT prompting.
COT_PROMPT_TEMPLATE = (
    "Q: {question}\n"
    "A: Let's think step by step.\n"
    "{answer}\n"
)

EVAL_PROMPT_TEMPLATE = (
    "Q: {question}\n"
    "A: Let's think step by step.\n"
)


def format_gsm8k_cot(example: dict) -> str:
    """Format a GSM8K example into a chain-of-thought training string."""
    question = example["question"]
    # GSM8K answers often already contain reasoning.
    # We wrap them in the CoT template.
    answer = example["answer"]
    return COT_PROMPT_TEMPLATE.format(question=question, answer=answer)


def format_gsm8k_prompt(example: dict) -> str:
    """Format a GSM8K example into an evaluation prompt (no answer)."""
    question = example["question"]
    return EVAL_PROMPT_TEMPLATE.format(question=question)


class TokenizedDataset(Dataset):
    """Tokenized dataset that returns input_ids, attention_mask, and labels."""

    def __init__(self, texts, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None,  # list of dicts
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        input_ids = self.encodings["input_ids"][idx]
        attention_mask = self.encodings["attention_mask"][idx]

        # For causal LM, labels = input_ids (shifted internally by the model)
        labels = input_ids.copy()

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch):
    """Pad batch of variable-length sequences."""
    import torch.nn.functional as F

    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = []
    attention_masks = []
    labels = []

    for item in batch:
        pad_len = max_len - item["input_ids"].size(0)
        input_ids.append(F.pad(item["input_ids"], (0, pad_len), value=0))
        attention_masks.append(F.pad(item["attention_mask"], (0, pad_len), value=0))
        # Pad labels with -100 (ignored in cross-entropy loss)
        labels.append(F.pad(item["labels"], (0, pad_len), value=-100))

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels),
    }


def load_gsm8k(config):
    """
    Load GSM8K dataset and return train/val/test splits.

    Returns:
        train_texts: list of formatted CoT strings for training
        val_texts: list of formatted CoT strings for validation
        test_raw: list of raw examples (dict with question/answer) for evaluation
    """
    print("Loading GSM8K dataset...")

    # HuggingFace GSM8K uses 'main' config; splits are 'train' and 'test'
    # The 'test' split has 1319 examples; we'll use it for final eval.
    # We split 'train' into train+val.
    try:
        dataset = load_dataset(config.dataset_name, "main")
    except Exception:
        # Fallback: try without config name
        dataset = load_dataset(config.dataset_name)

    train_raw = dataset["train"]
    test_raw = dataset["test"]

    # Create val split from train (10% of train)
    train_val = train_raw.train_test_split(test_size=0.1, seed=config.seed)
    train_raw = train_val["train"]
    val_raw = train_val["test"]

    # Format training/val data with CoT
    train_texts = [format_gsm8k_cot(ex) for ex in train_raw]
    val_texts = [format_gsm8k_cot(ex) for ex in val_raw]

    print(
        f"  Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_raw)}"
    )

    return train_texts, val_texts, test_raw


def create_dataloader(texts, tokenizer, config, batch_size, shuffle=True):
    """Create a DataLoader from a list of text strings."""
    dataset = TokenizedDataset(texts, tokenizer, max_length=config.max_seq_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,  # set to 0 for simplicity; increase if needed
        pin_memory=True,
    )


def extract_answer(text: str) -> str:
    """
    Extract the final numeric answer from a GSM8K model output.

    GSM8K uses the format: #### <number>
    We look for the last occurrence of '####' and extract the number after it.
    Falls back to looking for the last number in the text.
    """
    import re

    # Pattern 1: Look for #### <number>
    match = re.findall(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if match:
        return match[-1].replace(",", "")

    # Pattern 2: Look for "answer is <number>" or "= <number>" at the end
    match = re.findall(
        r"(?:answer\s+is|=)\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", text, re.IGNORECASE
    )
    if match:
        return match[-1].replace(",", "")

    # Pattern 3: Last standalone number
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""
