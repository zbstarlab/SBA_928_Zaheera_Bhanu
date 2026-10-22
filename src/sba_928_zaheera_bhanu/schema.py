from __future__ import annotations

REQUIRED_FIELDS = [
    "review_id",
    "product",
    "rating",
    "review",
    "instruction",
    "context",
    "target",
    "split",
]

VALID_SPLITS = {"train", "validation", "test"}


def validate_record(record: dict) -> None:
    """Raise ValueError when a dataset record is invalid."""

    for field in REQUIRED_FIELDS:
        if field not in record:
            raise ValueError(f"Missing required field: {field}")

    for field in [
        "review_id",
        "product",
        "review",
        "instruction",
        "context",
        "target",
    ]:
        value = record[field]

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Invalid or empty field: {field}"
            )

    rating = record["rating"]

    if type(rating) is not int or not 1 <= rating <= 5:
        raise ValueError("Rating must be an integer between 1 and 5")

    if record["split"] not in VALID_SPLITS:
        raise ValueError("Invalid dataset split")


def validate_dataset(records: list[dict]) -> None:
    """Validate records and check for duplicate IDs."""

    seen_ids = set()

    for record in records:
        validate_record(record)

        review_id = record["review_id"]

        if review_id in seen_ids:
            raise ValueError(
                f"Duplicate review ID: {review_id}"
            )

        seen_ids.add(review_id)
