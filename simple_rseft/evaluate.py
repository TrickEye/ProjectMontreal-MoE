"""
GSM8K evaluation: answer extraction and accuracy scoring.
"""

import re
import logging
import torch
from tqdm import tqdm

from data_utils import format_prompt

logger = logging.getLogger(__name__)


# ── Answer extraction ────────────────────────────────────────────────────────

def extract_gsm8k_answer(text: str) -> str | None:
    """
    Extract the final numeric answer from model output for GSM8K.

    Strategy (in priority order):
      1. \\boxed{...} pattern
      2. "the answer is ..." / "answer is ..." / "final answer ..."
      3. Last number in the text

    Returns the extracted answer string, or None if nothing found.
    """
    if not text:
        return None

    # Strategy 1: boxed{...}
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        content = boxed_match.group(1).strip()
        num = _extract_number(content)
        if num is not None:
            return num

    # Strategy 1.1: malformed boxed (missing braces)
    boxed_loose = re.search(r'\\boxed\s*([^\n\r.,;]+)', text)
    if boxed_loose:
        num = _extract_number(boxed_loose.group(1))
        if num is not None:
            return num

    # Strategy 2: explicit answer statements
    answer_patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)?\s*\$?\\?boxed\s*\{([^}]+)\}',
        r'(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|：)\s*([\d,.$%]+)',
        r'(?:final\s+)?answer\s*(?:is|:|：)\s*\$?([\d,.$%]+)',
    ]
    for pat in answer_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            num = _extract_number(m.group(1))
            if num is not None:
                return num

    # Strategy 3: last number in the text
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        return _normalize_number(numbers[-1])

    return None


def _extract_number(s: str) -> str | None:
    """Extract and normalize a number from a string."""
    # Remove LaTeX formatting and other wrappers
    s = s.strip().replace("$", "").replace("%", "")
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', s)
    if numbers:
        return _normalize_number(numbers[0])
    return None


def _normalize_number(s: str) -> str:
    """Normalize a number string: remove commas, handle trailing .0."""
    s = s.replace(",", "")
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        return s


# ── Evaluation loop ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, tokenizer, samples: list[dict], max_new_tokens: int = 512,
             max_seq_length: int = 512, batch_size: int = 4,
             device: str = "cuda",
             model_type: str = "olmoe_1b_7b_instruct") -> dict:
    """
    Evaluate a model on GSM8K test samples.

    Args:
        model: HF model (can be base or PEFT-wrapped)
        tokenizer: HF tokenizer
        samples: list of dicts with 'question' and 'answer' keys
        max_new_tokens, max_seq_length, batch_size

    Returns:
        dict with accuracy, correct, total, per_sample results
    """
    model.eval()
    correct = 0
    total = 0
    results = []

    for i in tqdm(range(0, len(samples), batch_size), desc="Evaluating"):
        batch = samples[i:i + batch_size]
        prompts = [format_prompt(s["question"], model_type=model_type) for s in batch]
        golds = [s["answer"] for s in batch]

        inputs = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode only the generated part (after prompt).
        # With left padding, actual prompt length = number of non-pad tokens.
        for j in range(len(batch)):
            actual_prompt_len = inputs["attention_mask"][j].sum().item()
            gen_ids = outputs[j, actual_prompt_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            pred = extract_gsm8k_answer(gen_text)
            gold = _normalize_number(golds[j])
            is_correct = (pred is not None and pred == gold)

            if is_correct:
                correct += 1
            total += 1

            results.append({
                "question": batch[j]["question"][:100],
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "gen_text": gen_text[:300],
            })

    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"Accuracy: {correct}/{total} = {accuracy:.4f}")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }
