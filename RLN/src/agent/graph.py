"""LangGraph wrapper around the RLN plan-and-execute pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

from rln_rag.llm import build_llm
from rln_rag.plan_executor_graph import build_plan_executor_graph
from rln_rag.settings import get_settings
from rln_rag.vectorstore import load_vectorstore


class Context(TypedDict, total=False):
    """Runtime overrides for the RLN assistant."""

    top_k: int


@dataclass
class State:
    """Instruction-level state for the LangGraph project."""

    instruction: str
    response: str | None = None
    plan: List[Dict[str, Any]] | None = None
    past_steps: List[Tuple[Dict[str, Any], Dict[str, Any]]] | None = None
    validation_errors: List[str] | None = None


@lru_cache(maxsize=8)
def _get_plan_graph(top_k: int | None):
    """Cache compiled plan executor graphs keyed by retriever depth."""

    settings = get_settings()
    if top_k is not None:
        settings.top_k = top_k
    vectorstore = load_vectorstore(settings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.top_k})
    llm = build_llm(settings)
    return build_plan_executor_graph(retriever, llm)


async def run_plan(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Invoke the RLN plan-and-execute graph and surface its results."""

    instruction = (state.instruction or "").strip()
    if not instruction:
        return {"response": "instruction must be provided."}

    ctx = runtime.context or {}
    top_k = ctx.get("top_k")
    try:
        plan_graph = _get_plan_graph(top_k)
    except FileNotFoundError:
        return {
            "instruction": instruction,
            "response": "Vector store missing. Run `rln-rag ingest` in the root project.",
        }
    result = plan_graph.invoke({"input": instruction})

    return {
        "instruction": instruction,
        "response": result.get("response", ""),
        "plan": result.get("plan", []),
        "past_steps": result.get("past_steps", []),
        "validation_errors": result.get("validation_errors", []),
    }


graph = (
    StateGraph(State, context_schema=Context)
    .add_node(run_plan)
    .add_edge("__start__", "run_plan")
    .compile(name="RLN Plan-and-Execute Wrapper")
)
