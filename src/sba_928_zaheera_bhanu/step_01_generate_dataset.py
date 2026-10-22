
"""Generate reproducible market research training datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
from pathlib import Path

import pandas as pd

from sba_928_zaheera_bhanu.schema import validate_dataset

LOGGER = logging.getLogger(__name__)

RAW_CSV = Path("data/raw/Womens Clothing E-Commerce Reviews.csv")
ANNOTATIONS_CSV = Path("data/raw/review_annotations.csv")
OUTPUT_DIR = Path("data/processed")

SEED = 42
SAMPLE_SIZE = 300


def load_annotations() -> dict[str, dict]:
    """Load and check the reviewed annotations."""
    annotations = {}

    with ANNOTATIONS_CSV.open(
        newline="", encoding="utf-8"
    ) as file:
        for row in csv.DictReader(file):
            review_id = row["review_id"]

            if review_id in annotations:
                raise ValueError(f"Duplicate annotation: {review_id}")

            for field in [
                "positive_feature",
                "primary_problem",
                "recommendation",
            ]:
                if not row[field].strip():
                    raise ValueError(
                        f"Missing {field} for {review_id}"
                    )

            annotations[review_id] = row

    return annotations


def create_target(annotation: dict) -> str:
    """Format the supervised learning target."""
    return (
        f"Positive feature: {annotation['positive_feature']}. "
        f"Primary problem: {annotation['primary_problem']}. "
        f"Recommendation: {annotation['recommendation']}."
    )


def generate_records() -> list[dict]:
    """Recreate the original seeded sample of 300 reviews."""
    df = pd.read_csv(RAW_CSV)

    columns = [
        "Clothing ID",
        "Age",
        "Review Text",
        "Rating",
        "Recommended IND",
        "Department Name",
        "Class Name",
    ]

    df = df[columns].dropna(
        subset=["Review Text", "Rating"]
    )

    df = df.drop_duplicates(subset=["Review Text"])

    df = df.sample(
        n=min(SAMPLE_SIZE, len(df)),
        random_state=SEED,
    )

    records = []

    for i, row in enumerate(df.to_dict("records")):
        review = str(row["Review Text"])
        category = str(row["Class Name"])

        records.append(
            {
                "review_id": f"REV-{i:05d}",
                "product": category,
                "age": (
                    int(row["Age"])
                    if pd.notna(row["Age"])
                    else None
                ),
                "rating": int(row["Rating"]),
                "review": review,
                "instruction": (
                    "Identify the positive feature, primary problem, "
                    "and recommendation."
                ),
                "context": (
                    f"Product: {category} | Review: {review}"
                ),
                "target": "",
            }
        )

    random.Random(SEED).shuffle(records)

    return records


def split_records(records: list[dict]) -> dict[str, list[dict]]:
    """Recreate the original 80/10/10 split."""
    n = len(records)
    train_end = int(n * 0.8)
    validation_end = int(n * 0.9)

    return {
        "train": records[:train_end],
        "validation": records[train_end:validation_end],
        "test": records[validation_end:],
    }


def save_jsonl(path: Path, records: list[dict]) -> None:
    """Save one JSON object per line."""
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def sha256(path: Path) -> str:
    """Calculate a file's SHA-256 checksum."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    """Prepare annotated data and save reproducible outputs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    LOGGER.info("Generating reproducible market research dataset")

    annotations = load_annotations()
    records = generate_records()
    assignments = {}

    with Path("data/raw/split_assignments.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        for row in csv.DictReader(file):
            assignments[row["review_id"]] = row["split"]

    splits = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for record in records:
        review_id = record["review_id"]

        if review_id in assignments:
            split_name = assignments[review_id]
            splits[split_name].append(record)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_labeled = []
    matched_ids = set()
    labeled_splits = {}

    for split_name, split_records_list in splits.items():
        labeled = []

        for record in split_records_list:
            review_id = record["review_id"]

            if review_id not in annotations:
                continue

            record["target"] = create_target(
                annotations[review_id]
            )
            record["split"] = split_name

            labeled.append(record)
            all_labeled.append(record)
            matched_ids.add(review_id)

        labeled_splits[split_name] = labeled

    missing = set(annotations) - matched_ids

    if missing:
        raise ValueError(
            f"Annotations without matching reviews: {sorted(missing)}"
        )

    validate_dataset(all_labeled)

    if any(not rows for rows in labeled_splits.values()):
        raise ValueError("Every split needs labeled examples")

    # Verify that review IDs appear in exactly one split.
    ids = [
        row["review_id"]
        for rows in labeled_splits.values()
        for row in rows
    ]

    if len(ids) != len(set(ids)):
        raise ValueError("Data leakage: duplicate review IDs")

    manifest = {
        "source_dataset": str(RAW_CSV),
        "annotations_file": str(ANNOTATIONS_CSV),
        "random_seed": SEED,
        "sample_size_before_annotation": len(records),
        "total_records": len(all_labeled),
        "split_counts": {},
        "file_hashes": {},
        "annotation_method": "Human-reviewed customer insights",
        "limitations": [
            "Small annotated sample",
            "Some original review text is truncated",
            "No geographic or gender comparison available",
        ],
    }

    for split_name, labeled in labeled_splits.items():
        output_path = OUTPUT_DIR / f"{split_name}.jsonl"

        save_jsonl(output_path, labeled)

        manifest["split_counts"][split_name] = len(labeled)
        manifest["file_hashes"][str(output_path)] = sha256(
            output_path
        )

        LOGGER.info("%s: %d records", split_name, len(labeled))

    csv_path = OUTPUT_DIR / "annotated_market_reviews.csv"
    pd.DataFrame(all_labeled).to_csv(csv_path, index=False)

    manifest["file_hashes"][str(csv_path)] = sha256(csv_path)

    with (OUTPUT_DIR / "manifest.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, indent=2)

    LOGGER.info("Saved %d annotated records", len(all_labeled))
    LOGGER.info("Dataset preparation complete")


if __name__ == "__main__":
    main()
