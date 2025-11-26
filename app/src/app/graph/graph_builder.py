"""LangGraph graph builder."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Dict, Any, List, Literal, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from app.graph.nodes.plan_node import run_plan_node
from app.graph.nodes.execute_node import build_action_tools
from app.graph.nodes.conversation_node import run_conversation_node
from app.graph.nodes.intent_node import classify_intent as classify
from app.graph.nodes.validator_node import run_validator
from app.services.executor_service import ExecutorService
from app.services.mission_summary_service import MissionSummaryService


_ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "artifacts"
_VALIDATION_LOG = _ARTIFACT_DIR / "plan_validation.log"
_TRACE_LOG = _ARTIFACT_DIR / "orchestrator_trace.log"


def _append_validation_log(
    *,
    question: str,
    plan: Dict[str, Any],
    validation: Any,
    status: str,
) -> None:
    """Persist each validation attempt for later inspection."""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "question": question,
        "plan": plan,
        "errors": list(getattr(validation, "errors", []) or []),
        "warnings": list(getattr(validation, "warnings", []) or []),
        "llm_verdict": getattr(validation, "llm_verdict", None),
        "llm_reason": getattr(validation, "llm_reason", None),
    }
    _VALIDATION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _VALIDATION_LOG.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def _append_trace_entry(
    *,
    question: str,
    phase: str,
    payload: Dict[str, Any],
) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "phase": phase,
        "payload": payload,
    }
    _TRACE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _TRACE_LOG.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _format_retry_feedback(validation: Any) -> str:
    errors = list(getattr(validation, "errors", []) or [])
    warnings = list(getattr(validation, "warnings", []) or [])
    if not errors and not warnings:
        return ""
    lines = ["이전 PLAN 검증에서 발견된 문제:"]
    if errors:
        lines.append("오류:")
        lines.extend(f"- {err}" for err in errors)
    if warnings:
        lines.append("경고:")
        lines.extend(f"- {warn}" for warn in warnings)
    return "\n".join(lines)


def _format_failure_context(
    *,
    question: str,
    logs: List[Dict[str, Any]],
    error_message: str | None,
    failed_step: Dict[str, Any] | None,
) -> str:
    data = {
        "user_request": question,
        "failed_step": {},
        "error_message": (error_message or "").strip(),
        "log_summary": [],
    }

    if failed_step:
        action = failed_step.get("action", "unknown")
        params = failed_step.get("params", {}) or {}
        target = params.get("target") or params.get("receiver")
        data["failed_step"] = {
            "action": action,
            "target": target,
        }

    if logs:
        for entry in logs[:5]:
            if not isinstance(entry, dict):
                continue
            step = entry.get("step") or {}
            result = entry.get("result") or {}
            action = step.get("action", "unknown")
            params = step.get("params", {}) or {}
            target = params.get("target") or params.get("receiver")
            status = result.get("status", "")
            data["log_summary"].append(
                {
                    "action": action,
                    "target": target,
                    "status": status,
                }
            )

    return json.dumps(data, ensure_ascii=False)


def _format_conversation_history(messages: List[Any] | None) -> str:
    if not messages:
        return "대화 기록 없음."
    lines: List[str] = []
    for msg in messages[-10:]:
        role = getattr(msg, "type", None) or getattr(msg, "role", None)
        content = getattr(msg, "content", None)
        if isinstance(msg, tuple):
            role, content = msg
        if content is None:
            continue
        lines.append(f"{role or 'assistant'}: {content}")
    return "\n".join(lines) if lines else "대화 기록 없음."


def _format_plan_history(plan_history: List[Dict[str, Any]] | None) -> str:
    if not plan_history:
        return ""
    lines = ["이전 PLAN 기록:"]
    for entry in plan_history[-3:]:
        question = entry.get("question", "")
        plan = entry.get("plan", {})
        lines.append(f"- 요청: {question}")
        lines.append(json.dumps(plan, ensure_ascii=False))
    return "\n".join(lines)


class OrchestratorState(TypedDict, total=False):
    question: str
    mode: Literal["conversation", "plan"]
    conversation_mode: Literal["normal", "unsupported"]
    conversation_reason: str
    conversation_route: str
    answer: str
    plan: Dict[str, Any]
    validation: Any
    execution_logs: List[Dict[str, Any]]
    plan_queue: List[Dict[str, Any]]
    current_step: Dict[str, Any]
    agent_status: Literal["continue", "done"]
    plan_attempts: int
    plan_status: Literal[
        "pending",
        "ok",
        "retry",
        "failed",
        "unsupported",
    ]
    plan_feedback: str
    mission_status: Literal["pending", "failed", "completed"]
    failure_context: str
    messages: Annotated[list, add_messages]
    plan_history: List[Dict[str, Any]]


def build_orchestrator_graph(llm,retriever,executor: ExecutorService):
    """간단한 계획/대화 그래프를 구성한다."""

    action_tools = build_action_tools(executor)
    summary_service = MissionSummaryService()

    def classify_intent(state: OrchestratorState) -> OrchestratorState:
        question = state.get("question", "")
        prediction = classify(question, llm=llm)
        messages = state.get("messages") or []
        if question:
            messages = add_messages(messages, ("user", question))
        return {
            "mode": prediction.intent,
            "plan_attempts": 0,
            "conversation_mode": prediction.conversation_mode,
            "conversation_reason": prediction.reason or "",
            "plan_feedback": "",
            "execution_logs": [],
            "mission_status": "pending",
            "failure_context": "",
            "messages": messages,
            "plan_history": state.get("plan_history") or [],
        }

    def conversation(state: OrchestratorState) -> OrchestratorState:
        override = state.get("conversation_override")
        if override:
            return {"answer": override, "conversation_override": ""}
        conv_mode = state.get("conversation_mode", "normal")
        if conv_mode == "unsupported":
            base = "아직은 이런 요청을 도와드리기 어려워요. 다른 도움을 드릴 수 있을까요?"
            return {"answer": base, "conversation_reason": ""}
        messages = state.get("messages") or []
        history_text = _format_conversation_history(messages)
        extra_blocks: List[str] = []
        failure_ctx = state.get("failure_context")
        if failure_ctx:
            extra_blocks.append(failure_ctx)
        plan_ctx = _format_plan_history(state.get("plan_history"))
        if plan_ctx:
            extra_blocks.append(plan_ctx)
        extra_context = "\n\n".join(extra_blocks) if extra_blocks else None
        answer = run_conversation_node(
            state.get("question", ""),
            llm,
            retriever,
            extra_context=extra_context,
            history=history_text,
        )
        updated_messages = add_messages(messages, ("assistant", answer))
        return {"answer": answer, "messages": updated_messages}

    def plan_and_execute(state: OrchestratorState) -> OrchestratorState:
        feedback = (state.get("plan_feedback") or "").strip()
        result = run_plan_node(
            state.get("question", ""),
            llm,
            [],
            retry_feedback=feedback if feedback else None,
        )
        plan_queue = list(result.plan.get("plan", []))
        plan_history = list(state.get("plan_history") or [])
        plan_history.append(
            {
                "question": state.get("question", ""),
                "plan": result.plan,
            }
        )
        attempts = int(state.get("plan_attempts") or 0) + 1
        _append_trace_entry(
            question=state.get("question", ""),
            phase="plan",
            payload={
                "attempt": attempts,
                "plan": result.plan,
            },
        )
        return {
            "plan": result.plan,
            "plan_queue": plan_queue,
            "plan_attempts": attempts,
            "plan_status": "pending",
            "plan_feedback": "",
            "plan_history": plan_history,
        }

    MAX_PLAN_RETRIES = 3

    def validate_plan_node(state: OrchestratorState) -> OrchestratorState:
        plan_obj = state.get("plan")
        if not plan_obj:
            return {
                "plan_status": "failed",
                "validation": None,
                "answer": "PLAN 생성 결과가 비어 있어 실행을 종료합니다.",
                "plan_feedback": "",
            }
        validation = run_validator(
            plan_obj,
            question=state.get("question", ""),
            llm=llm,
            extra_context=state.get("extra_context", None), # llm rag 문서 검색 결과 
        )
        question = state.get("question", "")
        llm_verdict = getattr(validation, "llm_verdict", None)
        llm_reason = getattr(validation, "llm_reason", "")

        if llm_verdict == "UNSUPPORTED":
            status = "unsupported"
        else:
            status = "ok" if validation.is_valid else None
            attempts = int(state.get("plan_attempts") or 0)
            if not validation.is_valid:
                status = "retry" if attempts < MAX_PLAN_RETRIES else "failed"

        _append_validation_log(
            question=question,
            plan=plan_obj,
            validation=validation,
            status=status or "unknown",
        )
        _append_trace_entry(
            question=question,
            phase="validation",
            payload={
                "status": status,
                "errors": list(validation.errors),
                "warnings": list(validation.warnings),
                "llm_verdict": getattr(validation, "llm_verdict", None),
                "llm_reason": getattr(validation, "llm_reason", None),
            },
        )

        if llm_verdict == "UNSUPPORTED":
            reason_text = (llm_reason or "").strip()
            base = "죄송하지만 이 요청을 수행할 수 없습니다. 다른 요청을 해주시겠어요?"
            answer = f"{base}\n\n상세: {reason_text}" if reason_text else base
            return {
                "validation": validation,
                "plan_status": "unsupported",
                "plan_queue": [],
                "conversation_override": answer,
                "plan_feedback": "",
            }

        if status == "ok":
            return {"validation": validation, "plan_status": "ok", "plan_feedback": ""}

        if status == "retry":
            feedback = _format_retry_feedback(validation)
            return {
                "validation": validation,
                "plan_status": "retry",
                "plan_queue": [],
                "plan_feedback": feedback,
            }

        error_text = "; ".join(validation.errors)
        return {
            "validation": validation,
            "plan_status": "failed",
            "answer": f"PLAN 검증 실패로 중단합니다: {error_text}",
            "plan_feedback": "",
        }

    def executor(state: OrchestratorState) -> OrchestratorState:
        queue = state.get("plan_queue") or []
        if not queue:
            return {"agent_status": "done", "plan_queue": []}
        next_step = queue[0]
        remaining = queue[1:]
        return {
            "current_step": next_step,
            "plan_queue": remaining,
            "agent_status": "continue",
        }

    def execute_current_step(state: OrchestratorState) -> OrchestratorState:
        step = state.get("current_step") or {}
        action = step.get("action")
        params = step.get("params") or {}
        tool = action_tools.get(action)
        if action == "summarize_mission":
            summary_text = summary_service.build_summary(
                question=state.get("question", ""),
                plan=state.get("plan"),
                execution_logs=state.get("execution_logs") or [],
                llm=llm,
            )
            params = dict(params)
            params["summary"] = summary_text
            tool = action_tools.get(action)

        if tool is None:
            result = {"status": "error", "message": f"Unsupported action {action}"}
        else:
            result = tool.invoke(params)
        remaining = state.get("plan_queue") or []
        logs = (state.get("execution_logs") or []) + [{"step": step, "result": result}]
        _append_trace_entry(
            question=state.get("question", ""),
            phase="execution_step",
            payload={
                "step": step,
                "result": result,
                "remaining_queue": remaining,
            },
        )
        status = (result.get("status") or "").lower()
        if status and status not in {"success", "reported"}:
            failure_text = _format_failure_context(
                question=state.get("question", ""),
                logs=logs,
                error_message=result.get("message"),
                failed_step=step,
            )
            return {
                "execution_logs": logs,
                "current_step": {},
                "plan_queue": [],
                "agent_status": "done",
                "mission_status": "failed",
                "failure_context": failure_text,
            }
        return {"execution_logs": logs, "current_step": {}}

    graph = StateGraph(OrchestratorState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("conversation", conversation)
    graph.add_node("plan", plan_and_execute)
    graph.add_node("validate_plan", validate_plan_node)
    graph.add_node("executor", executor)
    graph.add_node("execute_step", execute_current_step)
    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        lambda s: s.get("mode", "conversation"),
        {"conversation": "conversation", "plan": "plan"},
    )
    graph.add_edge("conversation", END)
    graph.add_edge("plan", "validate_plan")
    graph.add_conditional_edges(
        "validate_plan",
        lambda s: s.get("plan_status", "ok"),
        {
            "ok": "executor",
            "retry": "plan",
            "failed": END,
            "unsupported": "conversation",
        },
    )
    graph.add_conditional_edges(
        "executor",
        lambda s: (
            "failure"
            if s.get("mission_status") == "failed"
            else s.get("agent_status", "done")
        ),
        {"continue": "execute_step", "done": END, "failure": "conversation"},
    )
    graph.add_edge("execute_step", "executor")
    return graph.compile()
