
"""Compare original and fine-tuned FLAN-T5 on held-out reviews."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)

BASE_MODEL = "google/flan-t5-small"
TEST_PATH = Path("data/processed/test.jsonl")
OUTPUT_DIR = Path("artifacts/evaluation")

LABELS = [
    "Positive feature:",
    "Primary problem:",
    "Recommendation:",
]


def load_test_records() -> list[dict]:
    """Load only the held-out test records."""
    with TEST_PATH.open(encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if not records:
        raise ValueError("Test dataset is empty")

    for record in records:
        if not record.get("target", "").strip():
            raise ValueError("Test record has an empty target")

    return records


def generate(
    model,
    tokenizer,
    prompt: str,
    device: str,
) -> str:
    """Generate an answer with fixed generation settings."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            num_beams=1,
        )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )


def normalize_tokens(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def token_f1(prediction: str, reference: str) -> float:
    """Calculate lexical overlap between two answers."""
    predicted = Counter(normalize_tokens(prediction))
    expected = Counter(normalize_tokens(reference))

    overlap = sum((predicted & expected).values())

    predicted_count = sum(predicted.values())
    expected_count = sum(expected.values())

    if predicted_count == 0 or expected_count == 0:
        return 0.0

    precision = overlap / predicted_count
    recall = overlap / expected_count

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def format_score(response: str) -> float:
    """Measure how many required section labels are present."""
    return sum(
        label.lower() in response.lower()
        for label in LABELS
    ) / len(LABELS)


def evaluate_response(
    response: str,
    target: str,
) -> dict:
    """Calculate the same metrics for both models."""
    return {
        "token_f1": token_f1(response, target),
        "format_score": format_score(response),
        "exact_match": int(
            response.strip().lower() == target.strip().lower()
        ),
    }


def load_model(model_path: str, device: str):
    """Load a language model and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def average(records: list[dict], key: str) -> float:
    """Calculate the mean of a metric."""
    return sum(record[key] for record in records) / len(records)


def main() -> None:
    """Run a controlled comparison on held-out reviews."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fine-tuned-model",
        required=True,
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default="cpu",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_records = load_test_records()

    LOGGER.info("Loading original model")
    base_model, base_tokenizer = load_model(
        BASE_MODEL, args.device
    )

    LOGGER.info("Loading fine-tuned model")
    tuned_model, tuned_tokenizer = load_model(
        args.fine_tuned_model, args.device
    )

    comparisons = []
    base_metrics = []
    tuned_metrics = []

    for record in test_records:
        # Both models receive exactly the same input.
        prompt = (
            record["instruction"]
            + "\n\n"
            + record["context"]
        )

        base_response = generate(
            base_model,
            base_tokenizer,
            prompt,
            args.device,
        )

        tuned_response = generate(
            tuned_model,
            tuned_tokenizer,
            prompt,
            args.device,
        )

        target = record["target"]

        base_scores = evaluate_response(base_response, target)
        tuned_scores = evaluate_response(tuned_response, target)

        base_metrics.append(base_scores)
        tuned_metrics.append(tuned_scores)

        comparisons.append(
            {
                "review_id": record["review_id"],
                "product": record["product"],
                "age": record.get("age"),
                "rating": record["rating"],
                "review": record["review"],
                "expected_target": target,
                "base_response": base_response,
                "fine_tuned_response": tuned_response,
                "base_metrics": base_scores,
                "fine_tuned_metrics": tuned_scores,
            }
        )

        LOGGER.info("Review: %s", record["review_id"])
        LOGGER.info("Base: %s", base_response)
        LOGGER.info("Fine-tuned: %s", tuned_response)

    output_path = OUTPUT_DIR / "model_comparison.jsonl"

    with output_path.open("w", encoding="utf-8") as file:
        for row in comparisons:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = ["token_f1", "format_score", "exact_match"]

    summary = {
        "base_model": BASE_MODEL,
        "fine_tuned_model": args.fine_tuned_model,
        "test_records": len(test_records),
        "device": args.device,
        "generation_settings": {
            "max_new_tokens": 128,
            "do_sample": False,
            "num_beams": 1,
        },
        "metrics": {},
        "limitations": [
            "Only five held-out test examples",
            "Lexical overlap does not establish factual accuracy",
            "Format adherence does not establish answer correctness",
            "Expected answers require human verification",
        ],
    }

    for metric in metrics:
        base_score = average(base_metrics, metric)
        tuned_score = average(tuned_metrics, metric)

        summary["metrics"][metric] = {
            "original": base_score,
            "fine_tuned": tuned_score,
            "difference": tuned_score - base_score,
        }

        LOGGER.info(
            "%s: original=%.3f fine-tuned=%.3f",
            metric,
            base_score,
            tuned_score,
        )

    summary_path = OUTPUT_DIR / "model_comparison_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    LOGGER.info("Comparison saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
