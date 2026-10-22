
"""Audit model performance across market research subgroups."""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

LOGGER = logging.getLogger(__name__)

INPUT_FILE = Path("artifacts/evaluation/model_comparison.jsonl")
OUTPUT_DIR = Path("artifacts/bias_audit")

METRICS = ("token_f1", "format_score", "exact_match")


def load_comparisons() -> list[dict]:
    """Load original and fine-tuned model comparisons."""
    with INPUT_FILE.open(encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if not records:
        raise ValueError("No comparison records found")

    return records


def age_group(age: int | None) -> str:
    """Place reviewer ages into broad groups."""
    if age is None:
        return "Unknown"
    if age < 35:
        return "Under 35"
    if age < 50:
        return "35-49"
    return "50+"


def rating_group(rating: int) -> str:
    """Separate lower-rated and higher-rated reviews."""
    return "Low rating (1-3)" if rating <= 3 else "High rating (4-5)"


def group_records(records: list[dict]) -> dict:
    """Group evaluations by category, age, and rating."""
    groups = defaultdict(list)

    for record in records:
        categories = {
            "age_group": age_group(record.get("age")),
            "product_category": record["product"],
            "rating_group": rating_group(record["rating"]),
        }

        for dimension, group in categories.items():
            groups[(dimension, group)].append(record)

    return groups


def mean(values: list[float]) -> float:
    """Calculate the arithmetic average."""
    return sum(values) / len(values)


def calculate_subgroup_metrics(records: list[dict]) -> list[dict]:
    """Calculate matching metrics for both models in each subgroup."""
    results = []

    for (dimension, group), members in group_records(records).items():
        row = {
            "dimension": dimension,
            "group": group,
            "sample_size": len(members),
        }

        for metric in METRICS:
            base_values = [
                member["base_metrics"][metric]
                for member in members
            ]
            tuned_values = [
                member["fine_tuned_metrics"][metric]
                for member in members
            ]

            original = mean(base_values)
            fine_tuned = mean(tuned_values)

            row[f"original_{metric}"] = round(original, 4)
            row[f"fine_tuned_{metric}"] = round(fine_tuned, 4)
            row[f"difference_{metric}"] = round(
                fine_tuned - original, 4
            )

        row["interpretation"] = (
            "Insufficient sample for reliable subgroup conclusions"
            if len(members) < 5
            else "Exploratory result; further validation required"
        )

        results.append(row)

    return sorted(
        results,
        key=lambda row: (row["dimension"], row["group"]),
    )


def calculate_gaps(rows: list[dict]) -> list[dict]:
    """Find exploratory performance gaps within each dimension."""
    findings = []

    dimensions = sorted({row["dimension"] for row in rows})

    for dimension in dimensions:
        members = [
            row for row in rows
            if row["dimension"] == dimension
        ]

        if len(members) < 2:
            continue

        for metric in ("token_f1", "format_score"):
            scores = [
                row[f"fine_tuned_{metric}"]
                for row in members
            ]

            gap = max(scores) - min(scores)

            findings.append(
                {
                    "dimension": dimension,
                    "metric": metric,
                    "observed_gap": round(gap, 4),
                    "smallest_group_size": min(
                        row["sample_size"] for row in members
                    ),
                    "reliable_fairness_conclusion": False,
                }
            )

    return findings


def inspect_failures(records: list[dict]) -> list[dict]:
    """Identify examples requiring qualitative human review."""
    failures = []

    for record in records:
        response = record["fine_tuned_response"]
        metrics = record["fine_tuned_metrics"]

        if metrics["format_score"] < 1:
            failures.append(
                {
                    "review_id": record["review_id"],
                    "category": record["product"],
                    "reason": "Incomplete output format",
                    "response": response,
                    "review": record["review"],
                }
            )

    return failures


def main() -> None:
    """Generate subgroup metrics and an exploratory fairness audit."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    records = load_comparisons()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = calculate_subgroup_metrics(records)

    csv_path = OUTPUT_DIR / "subgroup_metrics.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    audit = {
        "test_sample_size": len(records),
        "subgroup_metrics": rows,
        "observed_performance_gaps": calculate_gaps(rows),
        "examples_for_human_review": inspect_failures(records),
        "limitations": [
            "Only five held-out test records are available",
            "Subgroup sample sizes are too small for reliable conclusions",
            "The dataset represents one clothing retail context",
            "Reviewer age is available but gender and region are not",
            "Token overlap and format scores cannot prove factual fairness",
            "Some review text was truncated during data preparation",
        ],
        "fairness_strategies": [
            "Expand and balance the evaluation dataset across age groups",
            "Include more clothing categories and rating distributions",
            "Measure model performance separately for each subgroup",
            "Use human reviewers to check unsupported recommendations",
            "Improve representation of underrepresented review types",
            "Collect regional data only when appropriate and lawful",
            "Avoid claiming demographic fairness without suitable evidence",
            "Repeat evaluations using larger independent test samples",
        ],
    }

    audit_path = OUTPUT_DIR / "bias_audit.json"

    with audit_path.open("w", encoding="utf-8") as file:
        json.dump(audit, file, indent=2, ensure_ascii=False)

    LOGGER.info("Evaluated %s held-out records", len(records))
    LOGGER.info("Saved %s subgroup rows", len(rows))
    LOGGER.info("Subgroup metrics: %s", csv_path)
    LOGGER.info("Bias audit: %s", audit_path)


if __name__ == "__main__":
    main()
