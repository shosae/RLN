from __future__ import annotations

from typing import Any, Callable, Dict

from langchain_core.tools import tool

from app.services.executor_service import ExecutorService
from .rag_graph import build_rag_graph

_executor = ExecutorService()


@tool("navigate")
def navigate_tool(target: str) -> Dict[str, Any]:
    """Simulate navigation by returning a status payload."""
    return _executor.navigate(target)


@tool("summarize_mission")
def summarize_tool(summary: str | None = None) -> Dict[str, Any]:
    """Return a text summary to the user."""
    return _executor.summarize_mission(summary)


@tool("report")
def report_tool(content: str) -> Dict[str, Any]:
    """Send a textual report back to the requesting user."""
    return _executor.report(content)


@tool("wait")
def wait_tool(seconds: int = 1) -> Dict[str, Any]:
    """Simulate a blocking wait."""
    return _executor.wait(seconds)


@tool("observe_scene")
def observe_scene_tool(target: str, query: str | None = None) -> Dict[str, Any]:
    """Simulate observing a physical location."""
    return _executor.observe_scene(target, query)


@tool("deliver_object")
def deliver_object_tool(receiver: str | None = None, item: str | None = None) -> Dict[str, Any]:
    """Simulate delivering an item."""
    return _executor.deliver_object(receiver, item)


def build_action_dispatch(retriever, llm) -> Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]:
    """
    Build runtime action handlers. Some actions are LangChain tools,
    others call into LangGraph/RAG components directly.
    """

    rag_graph = build_rag_graph(retriever, llm)

    def _run_tool(tool_callable, params: Dict[str, Any]) -> Dict[str, Any]:
        return tool_callable.invoke(params)

    def rag_qa_action(params: Dict[str, Any]) -> Dict[str, Any]:
        question = params.get("question")
        if not question:
            return {"status": "error", "message": "rag_qa requires 'question'."}
        result = rag_graph.invoke({"question": question})
        return {
            "status": "completed",
            "answer": result.get("answer"),
            "context_count": len(result.get("context") or []),
        }

    dispatch: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
        "navigate": lambda p: _run_tool(navigate_tool, p),
        "observe_scene": lambda p: _run_tool(observe_scene_tool, p),
        "deliver_object": lambda p: _run_tool(deliver_object_tool, p),
        "summarize_mission": lambda p: _run_tool(summarize_tool, p),
        "report": lambda p: _run_tool(report_tool, p),
        "wait": lambda p: _run_tool(wait_tool, p),
        "rag_qa": rag_qa_action,
        "rag_retrieve": rag_qa_action,
    }

    return dispatch
