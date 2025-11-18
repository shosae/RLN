from __future__ import annotations

from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
try:
    from langchain_core.prompts import ChatPromptTemplate
except ModuleNotFoundError:  # fallback for older langchain versions
    from langchain.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph


class RAGState(TypedDict, total=False):
    question: str
    context: List[Document]
    answer: str


def _summarize_docs(documents: List[Document]) -> str:
    formatted = []
    for idx, doc in enumerate(documents, start=1):
        meta = doc.metadata or {}
        source = meta.get("source") or meta.get("file_path") or "snippet"
        formatted.append(f"{idx}. ({source})\n{doc.page_content.strip()}")
    return "\n\n".join(formatted) if formatted else "No context retrieved."


def build_rag_graph(retriever: BaseRetriever, llm: BaseChatModel):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an assistant helping RLN experiments. Use the context to answer the question.
Cite the source index numbers whenever possible and explain if the answer is uncertain.""",
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}",
            ),
        ]
    )

    graph = StateGraph(RAGState)

    def retrieve(state: RAGState) -> RAGState:
        question = state["question"]
        docs = retriever.invoke(question)
        return {"question": question, "context": docs}

    def generate(state: RAGState) -> RAGState:
        context = state.get("context", [])
        context_text = _summarize_docs(context)
        question = state["question"]
        messages = prompt.format_messages(context=context_text, question=question)
        response = llm.invoke(messages)
        answer = getattr(response, "content", response)
        return {"question": question, "context": context, "answer": answer}

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
