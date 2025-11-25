"""Intent classification helpers."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from app.services.classify_intent_service import (
    IntentClassifierService,
    IntentPrediction,
)

_CLASSIFIER = IntentClassifierService()


def classify_intent(
    question: str,
    *,
    llm: BaseChatModel | None = None,
) -> IntentPrediction:
    """LLM 기반 intent 분류."""

    return _CLASSIFIER.classify(question, llm=llm)
