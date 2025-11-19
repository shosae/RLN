"""Intent classification helpers."""

from __future__ import annotations

from typing import Literal


PLAN_KEYWORDS = ("하고와", "보고와")


def normalize_question(question: str) -> str:
    """Utility to remove spaces for easier substring matching."""

    return "".join(question.split())


def classify_intent(question: str) -> Literal["conversation", "plan"]:
    """Return 'plan' when the utterance implies plan/execution."""

    normalized = normalize_question(question or "")
    for keyword in PLAN_KEYWORDS:
        if keyword in normalized:
            return "plan"
    return "conversation"
