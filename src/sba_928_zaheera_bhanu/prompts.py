
from __future__ import annotations


def build_prompt_variants(context: str) -> dict[str, str]:
    """Create three prompt styles for the same customer review."""

    minimal = (
        "Analyze this customer review and provide useful market research insights.\n\n"
        f"{context}"
    )

    structured = (
        "Analyze the customer review below.\n"
        "Return exactly three fields:\n"
        "Positive feature: what the customer liked.\n"
        "Primary problem: what the customer disliked.\n"
        "Recommendation: what the company should improve.\n\n"
        f"{context}"
    )

    grounded = (
        "You are a market research analyst.\n"
        "Analyze the customer review using ONLY the information provided.\n"
        "Do not invent product features, problems, or customer opinions.\n"
        "If something is not mentioned, write 'Not mentioned'.\n"
        "Use exactly this format:\n"
        "Positive feature: ...\n"
        "Primary problem: ...\n"
        "Recommendation: ...\n\n"
        f"{context}"
    )

    return {
        "minimal": minimal,
        "structured": structured,
        "grounded": grounded,
    }
