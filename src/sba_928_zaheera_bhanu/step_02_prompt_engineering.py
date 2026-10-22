from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from sba_928_zaheera_bhanu.prompts import build_prompt_variants

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "google/flan-t5-small"
TEST_FILE = Path("data/processed/test.jsonl")
OUTPUT_DIR = Path("artifacts/prompt_baseline")


def load_records(path: Path, limit: int) -> list[dict]:
    """Read JSONL records and select a limited number."""

    with path.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]

    return records[:limit]


def generate_response(model, tokenizer, prompt: str, device: str) -> str:
    """Generate an answer from the base language model."""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            num_beams=1,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


def main() -> None:
    """Evaluate three prompt variations on the same test records."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    LOGGER.info("Using device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()

    records = load_records(TEST_FILE, args.limit)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions = []
    counts = {"minimal": 0, "structured": 0, "grounded": 0}

    for record in records:
        variants = build_prompt_variants(record["context"])

        for name, prompt in variants.items():
            response = generate_response(model, tokenizer, prompt, device)

            predictions.append(
                {
                    "review_id": record["review_id"],
                    "prompt_type": name,
                    "prompt": prompt,
                    "response": response,
                    "expected_target": record["target"],
                }
            )

            counts[name] += 1
            LOGGER.info(
                "Review %s | Prompt %s | %s",
                record["review_id"],
                name,
                response[:100],
            )

    predictions_file = OUTPUT_DIR / "prompt_predictions.jsonl"

    with predictions_file.open("w", encoding="utf-8") as file:
        for prediction in predictions:
            file.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    summary = {
        "model": MODEL_NAME,
        "device": device,
        "reviews_evaluated": len(records),
        "prompt_counts": counts,
        "total_predictions": len(predictions),
        "generation_settings": {
            "max_new_tokens": 100,
            "do_sample": False,
            "num_beams": 1,
        },
        "note": "Prediction counts are not accuracy scores.",
    }

    with (OUTPUT_DIR / "prompt_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    LOGGER.info("Saved %s predictions", len(predictions))
    LOGGER.info("Results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
