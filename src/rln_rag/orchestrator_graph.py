from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
try:  # langchain 0.2+
    from langchain_core.prompts import ChatPromptTemplate
except ModuleNotFoundError:  # pragma: no cover - fallback for older versions
    from langchain.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from .planner_graph import generate_plan
from .plan_validator import PlanValidationResult, validate_plan


PLAN_KEYWORDS = [
    "가서",
    "가줘",
    "이동",
    "전달",
    "확인해",
    "확인해줘",
    "확인해줄래",
    "보고 와",
    "보고와",
    "순찰"
]


class OrchestratorState(TypedDict, total=False):
    question: str
    intent: Literal["conversation", "plan"]
    context: List[Document]
    answer: str
    plan: dict
    plan_validation: PlanValidationResult
    execution_index: int
    current_step: Dict[str, Any]
    execution_log: List[Dict[str, Any]]
    next_action: str
    mission_summary: str
    conversation_transcript: List[str]
    last_action_result: str


def _summarize_docs(documents: List[Document]) -> str:
    formatted = []
    for idx, doc in enumerate(documents, start=1):
        meta = doc.metadata or {}
        source = meta.get("source") or meta.get("file_path") or "snippet"
        formatted.append(f"{idx}. ({source})\n{doc.page_content.strip()}")
    return "\n\n".join(formatted) if formatted else "No context retrieved."


def _needs_plan(question: str) -> bool:
    text = question or ""
    lower = text.lower()
    return any(keyword in text or keyword in lower for keyword in PLAN_KEYWORDS)


def build_orchestrator_graph(retriever: BaseRetriever, llm: BaseChatModel):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """너는 실내 서비스 로봇 시스템의 대화 담당자이다.\n"
                "주어진 컨텍스트와 최신 실행 로그를 바탕으로 사용자의 질문에 한국어로 답변하고,\n"
                "사실이 없는 내용은 지어내지 말고 모르면 모른다고 답해라.\n""",
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}",
            ),
        ]
    )

    graph = StateGraph(OrchestratorState)

    def classify(state: OrchestratorState) -> OrchestratorState:
        question = state["question"]
        intent: Literal["conversation", "plan"] = (
            "plan" if _needs_plan(question) else "conversation"
        )
        return {"question": question, "intent": intent}

    def conversation_agent(state: OrchestratorState) -> OrchestratorState:
        question = state["question"]
        docs = retriever.invoke(question)
        context_text = _summarize_docs(docs)
        messages = prompt.format_messages(context=context_text, question=question)
        response = llm.invoke(messages)
        answer = getattr(response, "content", response)
        return {
            "question": question,
            "intent": "conversation",
            "context": docs,
            "answer": answer,
        }

    def planner_agent(state: OrchestratorState) -> OrchestratorState:
        question = state["question"]
        plan_obj, docs = generate_plan(question, retriever, llm)
        validation = validate_plan(plan_obj)
        plan_json = json.dumps(plan_obj, ensure_ascii=False, indent=2)

        summary_lines = ["행동 모드 결과:", plan_json]
        if validation.errors:
            summary_lines.append("\n[검증 오류]\n" + "\n".join(validation.errors))
        if validation.warnings:
            summary_lines.append("\n[검증 경고]\n" + "\n".join(validation.warnings))

        return {
            "question": question,
            "intent": "plan",
            "context": docs,
            "plan": plan_obj,
            "plan_validation": validation,
            "answer": "\n".join(summary_lines),
        }

    graph.add_node("classify_intent", classify)
    graph.add_node("conversation_agent", conversation_agent)
    graph.add_node("planner_agent", planner_agent)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        lambda state: state.get("intent", "conversation"),
        {
            "conversation": "conversation_agent",
            "plan": "planner_agent",
        },
    )
    graph.add_edge("conversation_agent", END)
    graph.add_edge("planner_agent", END)

    return graph.compile()
