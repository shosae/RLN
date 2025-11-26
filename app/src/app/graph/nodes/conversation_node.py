"""Conversation node helpers."""

from __future__ import annotations

from app.services.conversation_service import ConversationService

_SERVICE = ConversationService()


def run_conversation_node(
    question: str,
    llm,
    retriever,
    extra_context: str | None = None,
    history: str | None = None,
) -> str:
    """대화 노드 진입점."""

    return _SERVICE.answer(
        question,
        llm=llm,
        retriever=retriever,
        extra_context=extra_context,
        history=history,
    )
