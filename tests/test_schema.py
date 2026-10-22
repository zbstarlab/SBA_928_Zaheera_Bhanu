
"""Automated tests for market research dataset validation."""

from __future__ import annotations

import pytest

from sba_928_zaheera_bhanu.schema import (
    validate_dataset,
    validate_record,
)


@pytest.fixture
def valid_record() -> dict:
    """Create a valid example record for testing."""
    return {
        "review_id": "TEST-001",
        "product": "Dresses",
        "rating": 4,
        "review": "Beautiful dress but slightly tight.",
        "instruction": (
            "Identify the positive feature, primary problem, "
            "and recommendation."
        ),
        "context": (
            "Product: Dresses | Review: "
            "Beautiful dress but slightly tight."
        ),
        "target": (
            "Positive feature: Beautiful appearance. "
            "Primary problem: Tight fit. "
            "Recommendation: Improve sizing."
        ),
        "split": "train",
    }


def test_valid_record_passes(valid_record):
    """A complete record should pass validation."""
    validate_record(valid_record)


def test_missing_target_fails(valid_record):
    """An empty target should raise an error."""
    valid_record["target"] = ""

    with pytest.raises(ValueError, match="target"):
        validate_record(valid_record)


def test_invalid_rating_fails(valid_record):
    """Ratings outside the 1-5 range should fail."""
    valid_record["rating"] = 6

    with pytest.raises(ValueError, match="Rating"):
        validate_record(valid_record)


def test_invalid_split_fails(valid_record):
    """Unknown dataset split names should fail."""
    valid_record["split"] = "unknown"

    with pytest.raises(ValueError, match="split"):
        validate_record(valid_record)


def test_duplicate_ids_fail(valid_record):
    """Duplicate review IDs should not be allowed."""
    duplicate = valid_record.copy()

    with pytest.raises(ValueError, match="Duplicate"):
        validate_dataset([valid_record, duplicate])


def test_valid_dataset_passes(valid_record):
    """Distinct valid records should pass validation."""
    second_record = valid_record.copy()
    second_record["review_id"] = "TEST-002"

    validate_dataset([valid_record, second_record])
