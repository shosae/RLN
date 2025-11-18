from __future__ import annotations

from typing import Any, Dict, List, Tuple, TypedDict

from langgraph.graph import END, StateGraph

from .plan_tools import build_action_dispatch
from .planner_graph import generate_plan
from .plan_validator import validate_plan


class PlanState(TypedDict, total=False):
    """State passed between Plan-and-Execute nodes."""

    input: str
    plan: List[Dict[str, Any]]
    next_step_index: int
    past_steps: List[Tuple[Dict[str, Any], Dict[str, Any]]]
    response: str
    validation_errors: List[str]


def build_plan_executor_graph(retriever, llm):
    """Compile a LangGraph that plans and then executes sequential actions."""

    action_dispatch = build_action_dispatch(retriever, llm)

    def _planning_node(state: PlanState) -> PlanState:
        question = state["input"]
        plan_dict, _ = generate_plan(question, retriever, llm)
        validation = validate_plan(plan_dict)
        return {
            "plan": plan_dict.get("plan", []),
            "next_step_index": 0,
            "past_steps": [],
            "validation_errors": validation.errors,
        }

    def _execute_step(state: PlanState) -> PlanState:
        plan_steps = state.get("plan") or []
        idx = state.get("next_step_index", 0)
        if idx >= len(plan_steps):
            return {
                "status": "done",
                "response": _summarize_execution(state.get("past_steps", [])),
            }

        step = plan_steps[idx]
        action = step.get("action")
        params = step.get("params") or {}
        handler = action_dispatch.get(action or "")
        if not handler:
            result = {"status": "error", "message": f"Unsupported action '{action}'"}
        else:
            result = handler(params)

        past = list(state.get("past_steps") or [])
        past.append((step, result))
        next_idx = idx + 1
        status = "continue" if next_idx < len(plan_steps) else "done"
        response = (
            _summarize_execution(past) if status == "done" else state.get("response")
        )
        return {
            "past_steps": past,
            "next_step_index": next_idx,
            "status": status,
            "response": response,
        }

    graph = StateGraph(PlanState)
    graph.add_node("planning", _planning_node)
    graph.add_node("execute_step", _execute_step)
    graph.set_entry_point("planning")
    graph.add_edge("planning", "execute_step")
    graph.add_conditional_edges(
        "execute_step",
        lambda s: s.get("status", "done"),
        {
            "continue": "execute_step",
            "done": END,
        },
    )
    return graph.compile()


def _summarize_execution(past_steps: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> str:
    """Simple textual summary of executed steps."""
    if not past_steps:
        return "No actions were executed."
    lines = ["Plan execution summary:"]
    for idx, (step, result) in enumerate(past_steps, start=1):
        action = step.get("action")
        res_status = result.get("status")
        message = result.get("message") or result.get("answer") or result
        lines.append(f"{idx}. [{action}] status={res_status} -> {message}")
    return "\n".join(lines)
