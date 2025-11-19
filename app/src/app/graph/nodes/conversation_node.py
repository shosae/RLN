"""Conversation node helpers."""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

try:
    from langchain_core.prompts import ChatPromptTemplate
except ModuleNotFoundError:  # pragma: no cover
    from langchain.prompts import ChatPromptTemplate


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """너는 실내 서비스 로봇 시스템의 대화 담당자이다.
            주어진 컨텍스트를 바탕으로 한국어로 답변하되 사실이 없으면 모른다고 말해라.""",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def _summarize_docs(documents: List[Document]) -> str:
    if not documents:
        return "No context retrieved."
    chunks = []
    for idx, doc in enumerate(documents, start=1):
        meta = doc.metadata or {}
        source = meta.get("source") or meta.get("title") or f"doc-{idx}"
        chunks.append(f"{idx}. ({source})\n{doc.page_content.strip()}")
    return "\n\n".join(chunks)


def run_conversation_node(question: str, llm, retriever) -> str:
    """대화 노드 진입점."""

    docs = retriever.invoke(question)
    context_text = _summarize_docs(docs)
    messages = PROMPT.format_messages(context=context_text, question=question)
    response = llm.invoke(messages)
    return getattr(response, "content", response)
