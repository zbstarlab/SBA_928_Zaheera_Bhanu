
"""Analyze a new customer review using the fine-tuned FLAN-T5 model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)


def build_prompt(product: str, rating: int, review: str) -> str:
    """Create an input consistent with the fine-tuning dataset."""
    instruction = (
        "Identify the positive feature, primary problem, "
        "and recommendation."
    )

    context = f"Product: {product} | Review: {review}"

    return f"{instruction}\n\n{context}"


def load_model(model_path: str, device: str):
    """Load the saved model and tokenizer."""

    if not Path(model_path).is_dir():
        raise FileNotFoundError(
            f"Fine-tuned model not found: {model_path}"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    model = model.to(device)
    model.eval()

    return model, tokenizer


def analyze_review(
    model,
    tokenizer,
    prompt: str,
    device: str,
) -> str:
    """Generate a market research analysis."""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            num_beams=1,
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )


def main() -> None:
    """Run inference on a new customer review."""

    parser = argparse.ArgumentParser(
        description="Analyze customer reviews using fine-tuned FLAN-T5"
    )

    parser.add_argument(
        "--model",
        default="artifacts/models/flan-t5-sba-928/final",
    )

    parser.add_argument("--product", required=True)
    parser.add_argument("--rating", type=int, required=True)
    parser.add_argument("--review", required=True)

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default="cpu",
    )

    args = parser.parse_args()

    if not 1 <= args.rating <= 5:
        parser.error("--rating must be between 1 and 5")

    if not args.review.strip():
        parser.error("--review cannot be empty")

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    LOGGER.info("Loading fine-tuned model")

    model, tokenizer = load_model(args.model, args.device)

    prompt = build_prompt(
        args.product,
        args.rating,
        args.review,
    )

    response = analyze_review(
        model,
        tokenizer,
        prompt,
        args.device,
    )

    print("\nCUSTOMER REVIEW")
    print(f"Product: {args.product}")
    print(f"Rating: {args.rating}/5")
    print(f"Review: {args.review}")

    print("\nAI MARKET RESEARCH ANALYSIS")
    print(response)

    print("\nNote: AI-generated insights require human verification.")


if __name__ == "__main__":
    main()
