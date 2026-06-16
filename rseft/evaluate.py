"""
GSM8K evaluation for R-SEFT.

Evaluates a fine-tuned model on GSM8K using chain-of-thought generation
and exact-match accuracy on the final numeric answer.
"""

import re
import torch
from tqdm import tqdm

from data_utils import format_gsm8k_prompt, extract_answer


def normalize_answer(answer: str) -> str:
    """
    Normalize a numeric answer string for comparison.

    Handles: commas, trailing zeros, percentage signs, dollar signs, fractions.
    """
    answer = answer.strip().lower()

    # Remove commas from numbers
    answer = answer.replace(",", "")

    # Remove trailing zeros after decimal
    if "." in answer:
        answer = answer.rstrip("0").rstrip(".")

    # Remove whitespace
    answer = answer.strip()

    return answer


def answers_match(predicted: str, ground_truth: str) -> bool:
    """Check if predicted and ground truth answers match numerically."""
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    if pred_norm == gt_norm:
        return True

    # Try numeric comparison
    try:
        pred_num = float(pred_norm)
        gt_num = float(gt_norm)
        return abs(pred_num - gt_num) < 1e-6
    except (ValueError, TypeError):
        pass

    return False


def extract_gsm8k_ground_truth(answer_text: str) -> str:
    """
    Extract the final numeric answer from a GSM8K ground truth answer.

    GSM8K answers end with: #### <number>
    """
    # The answer field contains the full CoT + final answer
    # Extract the number after the last ####
    match = re.findall(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", answer_text)
    if match:
        return match[-1].replace(",", "")
    return ""


@torch.no_grad()
def evaluate(model, tokenizer, test_dataset, config, verbose: bool = True):
    """
    Evaluate a model on the GSM8K test set.

    Args:
        model: the model to evaluate
        tokenizer: tokenizer
        test_dataset: raw GSM8K test examples (list of dicts with 'question' and 'answer')
        config: RSEFTConfig
        verbose: if True, print example-level details

    Returns:
        accuracy: float (0.0 to 1.0)
        results: list of dicts with per-example results
    """
    print("\n" + "=" * 70)
    print("Evaluation: GSM8K Test Set")
    print("=" * 70)

    model.eval()
    device = next(model.parameters()).device

    correct = 0
    total = 0
    results = []

    for idx, example in enumerate(
        tqdm(test_dataset, desc="  Evaluating", total=len(test_dataset))
    ):
        question = example["question"]
        ground_truth_full = example["answer"]
        ground_truth = extract_gsm8k_ground_truth(ground_truth_full)

        # Format prompt
        prompt = format_gsm8k_prompt({"question": question})

        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_seq_length,
        ).to(device)

        # Generate
        try:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=config.eval_max_new_tokens,
                do_sample=False,  # greedy decoding (as per paper)
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        except Exception as e:
            print(f"\n  [Error generating for example {idx}]: {e}")
            results.append(
                {
                    "question": question,
                    "predicted": "",
                    "ground_truth": ground_truth,
                    "correct": False,
                    "error": str(e),
                }
            )
            total += 1
            continue

        # Decode the generated part (after the prompt)
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Extract final answer
        predicted_answer = extract_answer(generated_text)
        is_correct = answers_match(predicted_answer, ground_truth)

        if is_correct:
            correct += 1
        total += 1

        results.append(
            {
                "question": question,
                "predicted": generated_text,
                "predicted_answer": predicted_answer,
                "ground_truth": ground_truth,
                "correct": is_correct,
            }
        )

        if verbose and idx < 3:
            print(f"\n  --- Example {idx + 1} ---")
            print(f"  Q: {question[:100]}...")
            print(f"  Generated (truncated): {generated_text[:200]}...")
            print(f"  Predicted answer: {predicted_answer}")
            print(f"  Ground truth: {ground_truth}")
            print(f"  Correct: {is_correct}")

    accuracy = correct / total if total > 0 else 0.0

    print(f"\n  ── Results ──")
    print(f"  Correct: {correct}/{total}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy:.2%})")

    return accuracy, results


def evaluate_base_model(config):
    """
    Quick evaluation of the base (untuned) model for comparison.

    Returns the base accuracy for reference.
    """
    from model_utils import load_model_and_tokenizer

    print("\n" + "=" * 70)
    print("Evaluating Base Model (untuned)")
    print("=" * 70)

    model, tokenizer = load_model_and_tokenizer(config)

    from data_utils import load_gsm8k
    _, _, test_dataset = load_gsm8k(config)

    accuracy, _ = evaluate(model, tokenizer, test_dataset, config, verbose=False)
    print(f"\n  Base model accuracy: {accuracy:.4f} ({accuracy:.2%})")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return accuracy
