"""LangGraph graph builder."""

from __future__ import annotations

from typing import Dict, Any, List, Literal, TypedDict

from langgraph.graph import StateGraph, END

from app.graph.nodes.plan_node import run_plan_node
from app.graph.nodes.execute_node import execute_plan
from app.graph.nodes.conversation_node import run_conversation_node
from app.graph.nodes.intent_node import classify_intent
from app.services.executor_service import ExecutorService


class OrchestratorState(TypedDict, total=False):
    question: str
    mode: Literal["conversation", "plan"]
    answer: str
    plan: Dict[str, Any]
    validation: Any
    execution_logs: List[Dict[str, Any]]


def build_orchestrator_graph(llm, retriever, executor: ExecutorService):
    """간단한 계획/대화 그래프를 구성한다."""

    def entry(state: OrchestratorState) -> OrchestratorState:
        question = state.get("question", "")
        mode = classify_intent(question)
        return {"mode": mode}

    def conversation(state: OrchestratorState) -> OrchestratorState:
        answer = run_conversation_node(state.get("question", ""), llm, retriever)
        return {"answer": answer}

    def plan_and_execute(state: OrchestratorState) -> OrchestratorState:
        result = run_plan_node(state.get("question", ""), llm, [])
        execution_logs = execute_plan(result.plan.get("plan", []), executor)
        return {
            "plan": result.plan,
            "validation": result.validation,
            "execution_logs": execution_logs,
        }

    graph = StateGraph(OrchestratorState)
    graph.add_node("entry", entry)
    graph.add_node("conversation", conversation)
    graph.add_node("plan", plan_and_execute)
    graph.set_entry_point("entry")
    graph.add_conditional_edges(
        "entry",
        lambda s: s.get("mode", "conversation"),
        {"conversation": "conversation", "plan": "plan"},
    )
    graph.add_edge("conversation", END)
    graph.add_edge("plan", END)
    return graph.compile()
