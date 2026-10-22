
"""Fine-tune FLAN-T5 for market research review analysis."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

LOGGER = logging.getLogger(__name__)


def load_records(path: str) -> list[dict]:
    """Load validated JSONL training or validation records."""
    records = []

    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            if not record.get("target", "").strip():
                raise ValueError(
                    f"Missing target for {record.get('review_id')}"
                )

            records.append(record)

    if not records:
        raise ValueError(f"No records found in {path}")

    return records


def prepare_dataset(records, tokenizer, config):
    """Tokenize model inputs and expected answers separately."""
    examples = []

    for record in records:
        examples.append(
            {
                "input_text": (
                    record["instruction"]
                    + "\n\n"
                    + record["context"]
                ),
                "target": record["target"],
            }
        )

    dataset = Dataset.from_list(examples)

    def tokenize_batch(batch):
        inputs = tokenizer(
            batch["input_text"],
            max_length=config["max_source_length"],
            truncation=True,
        )

        targets = tokenizer(
            text_target=batch["target"],
            max_length=config["max_target_length"],
            truncation=True,
        )

        inputs["labels"] = targets["input_ids"]
        return inputs

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
    )


def train_model(config: dict) -> None:
    """Train FLAN-T5 and save model artifacts and metrics."""
    set_seed(config["seed"])

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading model: %s", config["model_name"])

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"]
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config["model_name"]
    )

    train_records = load_records(config["train_file"])
    validation_records = load_records(config["validation_file"])

    train_dataset = prepare_dataset(
        train_records, tokenizer, config
    )
    validation_dataset = prepare_dataset(
        validation_records, tokenizer, config
    )

    LOGGER.info(
        "Training examples: %s | Validation examples: %s",
        len(train_dataset),
        len(validation_dataset),
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config[
            "per_device_train_batch_size"
        ],
        per_device_eval_batch_size=config[
            "per_device_eval_batch_size"
        ],
        gradient_accumulation_steps=config[
            "gradient_accumulation_steps"
        ],
        weight_decay=config["weight_decay"],
        predict_with_generate=False,
        report_to="none",
        push_to_hub=False,
        seed=config["seed"],
        save_total_limit=1,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )

    LOGGER.info("Starting fine-tuning")

    training_result = trainer.train()
    evaluation_metrics = trainer.evaluate()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    resolved_config = {
        **config,
        "training_examples": len(train_records),
        "validation_examples": len(validation_records),
        "test_examples_used_during_training": 0,
    }

    with (output_dir / "training_config_resolved.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(resolved_config, file, indent=2)

    summary = {
        "model": config["model_name"],
        "training_examples": len(train_records),
        "validation_examples": len(validation_records),
        "training_metrics": training_result.metrics,
        "validation_metrics": evaluation_metrics,
        "training_history": trainer.state.log_history,
        "final_model_path": str(final_dir),
    }

    with (output_dir / "training_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2, default=str)

    LOGGER.info("Fine-tuning completed")
    LOGGER.info("Final model saved to %s", final_dir)
    LOGGER.info("Validation metrics: %s", evaluation_metrics)


def main() -> None:
    """Read configuration and execute the training experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/default.json",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    with Path(args.config).open(encoding="utf-8") as file:
        config = json.load(file)

    train_model(config)


if __name__ == "__main__":
    main()
