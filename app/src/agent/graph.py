"""LangGraph graph that exposes the RLN orchestrator flow node-by-node."""

from __future__ import annotations

import asyncio
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Tuple

from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

# Ensure LangGraph server sees the repo root resources even when launched from app/.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("DOCS_DIR", str(PROJECT_ROOT / "data" / "seed"))
os.environ.setdefault("VECTORSTORE_DIR", str(PROJECT_ROOT / "artifacts" / "vectorstore"))

try:
    from langchain_core.prompts import ChatPromptTemplate
except ModuleNotFoundError:  # pragma: no cover
    from langchain.prompts import ChatPromptTemplate

from rln_rag.llm import build_llm
from rln_rag.orchestrator_graph import (
    OrchestratorState,
    _needs_plan,
    _summarize_docs,
    generate_plan,
    validate_plan,
)
from rln_rag.plan_validator import PlanValidationResult, ALLOWED_ACTIONS
from rln_rag.settings import Settings, get_settings
from rln_rag.vectorstore import load_vectorstore
from rln_rag.waypoints import load_waypoint_documents


class Context(TypedDict, total=False):
    """Runtime overrides for the RLN pipelines."""

    docs_dir: str
    vectorstore_dir: str
    embedding_model: str
    llm_provider: str
    llm_model: str
    langgraph_api_key: str
    langgraph_base_url: str
    groq_api_key: str
    ollama_base_url: str
    temperature: float
    top_k: int


def _to_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _apply_context_overrides(settings: Settings, ctx: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Apply LangGraph runtime overrides to the Settings object."""

    if not ctx:
        settings.ensure_directories()
        return {}

    if docs_dir := ctx.get("docs_dir"):
        settings.docs_dir = _to_path(docs_dir)
    if vectorstore_dir := ctx.get("vectorstore_dir"):
        settings.vectorstore_dir = _to_path(vectorstore_dir)
    if embedding_model := ctx.get("embedding_model"):
        settings.embedding_model = str(embedding_model)
    if llm_provider := ctx.get("llm_provider"):
        settings.llm_provider = str(llm_provider)
    if llm_model := ctx.get("llm_model"):
        settings.llm_model = str(llm_model)
    if langgraph_api_key := ctx.get("langgraph_api_key"):
        settings.langgraph_api_key = str(langgraph_api_key)
    if langgraph_base_url := ctx.get("langgraph_base_url"):
        settings.langgraph_base_url = str(langgraph_base_url)
    if groq_api_key := ctx.get("groq_api_key"):
        settings.groq_api_key = str(groq_api_key)
    if ollama_base_url := ctx.get("ollama_base_url"):
        settings.ollama_base_url = str(ollama_base_url)

    if (temperature := ctx.get("temperature")) is not None:
        settings.temperature = float(temperature)
    if (top_k := ctx.get("top_k")) is not None:
        settings.top_k = int(top_k)

    settings.ensure_directories()
    return ctx


def _error_state(question: str, message: str) -> OrchestratorState:
    return {
        "question": question,
        "intent": "conversation",
        "context": [],
        "answer": message,
    }


def _normalize_context_key(ctx: Mapping[str, Any] | None) -> Tuple[Tuple[str, Any], ...]:
    if not ctx:
        return ()
    items = []
    for key in sorted(Context.__annotations__.keys()):
        if key in ctx and ctx[key] is not None:
            value = ctx[key]
            if isinstance(value, (str, int, float, bool)):
                items.append((key, value))
            else:
                items.append((key, str(value)))
    return tuple(items)


class ResourceInitError(RuntimeError):
    """Raised when the retriever/LLM resources cannot be prepared."""


@lru_cache(maxsize=8)
def _load_resources(signature: Tuple[Tuple[str, Any], ...]):
    overrides = dict(signature)
    settings = get_settings()
    _apply_context_overrides(settings, overrides)

    try:
        vectorstore = load_vectorstore(settings)
    except FileNotFoundError as exc:
        raise ResourceInitError(
            f"Vector store not found at {settings.vectorstore_dir}. CLI에서 `rln-rag ingest`를 먼저 실행하세요."
        ) from exc

    try:
        llm = build_llm(settings)
    except Exception as exc:  # noqa: BLE001
        raise ResourceInitError(f"LLM 초기화 실패: {exc}") from exc

    waypoint_docs = load_waypoint_documents(settings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.top_k})
    return settings, retriever, llm, waypoint_docs


def _get_resources(runtime: Runtime[Context]):
    signature = _normalize_context_key(runtime.context)
    return _load_resources(signature)


async def _aget_resources(runtime: Runtime[Context]):
    signature = _normalize_context_key(runtime.context)
    return await asyncio.to_thread(_load_resources, signature)


def _get_plan_steps(state: OrchestratorState) -> list[dict[str, Any]]:
    plan = state.get("plan") or {}
    steps = plan.get("plan")
    if isinstance(steps, list):
        return steps
    return []


def _append_log(state: OrchestratorState, entry: dict[str, Any]) -> list[dict[str, Any]]:
    logs = list(state.get("execution_log") or [])
    logs.append(entry)
    return logs


def _log_text(logs: list[dict[str, Any]]) -> str:
    lines = []
    for idx, entry in enumerate(logs, start=1):
        action = entry.get("action")
        result = entry.get("result")
        params = entry.get("params")
        param_text = f" params={params}" if params else ""
        lines.append(f"{idx}. [{action}]{param_text} -> {result}")
    return "\n".join(lines) if lines else "실행된 action이 없습니다."


def _with_updates(state: OrchestratorState, **updates: Any) -> OrchestratorState:
    new_state = dict(state)
    new_state.update(updates)
    return new_state


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """너는 실내 서비스 로봇 시스템의 대화 담당자이다.
            주어진 컨텍스트와 최신 실행 로그를 바탕으로 사용자의 질문에 한국어로 답변하고,
            사실이 없는 내용은 지어내지 말고 모르면 모른다고 답해라.""",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """너는 실내 임무 로봇의 운영자이다.
다음 로그를 바탕으로 어떤 행동이 있었는지 한국어로 간결하게 요약하고,
향후 후속 조치나 주의할 점이 있으면 bullet로 정리해라.""",
        ),
        ("human", "Question: {question}\nLogs:\n{logs}"),
    ]
)


def classify_intent(state: OrchestratorState, runtime: Runtime[Context]) -> OrchestratorState:
    question = (state.get("question") or "").strip()
    if not question:
        return _error_state("", "질문을 입력해 주세요.")
    intent = "plan" if _needs_plan(question) else "conversation"
    return {"question": question, "intent": intent}


async def conversation_agent(state: OrchestratorState, runtime: Runtime[Context]) -> OrchestratorState:
    question = state["question"]
    try:
        _, retriever, llm, _ = await _aget_resources(runtime)
    except ResourceInitError as exc:
        return _error_state(question, str(exc))

    docs = retriever.invoke(question)
    context_text = _summarize_docs(docs)
    messages = PROMPT.format_messages(context=context_text, question=question)
    response = await asyncio.to_thread(llm.invoke, messages)
    answer = getattr(response, "content", response)
    return {
        "question": question,
        "intent": "conversation",
        "context": docs,
        "answer": answer,
    }


async def planner_agent(state: OrchestratorState, runtime: Runtime[Context]) -> OrchestratorState:
    question = state["question"]
    try:
        _, _, llm, waypoint_docs = await _aget_resources(runtime)
    except ResourceInitError as exc:
        return _error_state(question, str(exc))

    plan_obj, docs = await asyncio.to_thread(generate_plan, question, waypoint_docs, llm)
    validation: PlanValidationResult = validate_plan(plan_obj)
    plan_json = json.dumps(plan_obj, ensure_ascii=False, indent=2)
    return {
        "question": question,
        "intent": "plan",
        "context": docs,
        "plan": plan_obj,
        "plan_validation": validation,
        "answer": plan_json,
        "execution_index": 0,
        "execution_log": [],
        "current_step": None,
        "next_action": None,
        "mission_summary": "",
        "last_action_result": "",
        "conversation_transcript": [],
    }


def next_action(state: OrchestratorState, runtime: Runtime[Context]) -> OrchestratorState:
    validation = state.get("plan_validation")
    logs = list(state.get("execution_log") or [])
    if validation and getattr(validation, "errors", None):
        logs.append(
            {
                "action": "plan_validation",
                "params": {},
                "result": "PLAN 오류가 있어 실행을 건너뜁니다.",
            }
        )
        return _with_updates(
            state,
            execution_log=logs,
            next_action="done",
            execution_index=int(state.get("execution_index", 0)),
            current_step=None,
        )

    steps = _get_plan_steps(state)
    idx = int(state.get("execution_index", 0))
    if not steps:
        logs.append(
            {"action": "plan", "params": {}, "result": "실행할 단계가 없습니다."}
        )
        return _with_updates(
            state,
            execution_log=logs,
            next_action="done",
            execution_index=idx,
            current_step=None,
        )

    while idx < len(steps):
        step = steps[idx] if isinstance(steps[idx], dict) else {}
        action = step.get("action")
        if action in ALLOWED_ACTIONS:
            return _with_updates(
                state,
                current_step=step,
                next_action=action,
                execution_index=idx,
            )
        logs.append(
            {
                "action": step.get("action"),
                "params": step.get("params"),
                "result": "지원하지 않는 action이라 건너뜀.",
            }
        )
        idx += 1

    return _with_updates(
        state,
        execution_log=logs,
        next_action="done",
        execution_index=idx,
        current_step=None,
    )


async def execute_navigate(state: OrchestratorState, runtime: Runtime[Context]) -> OrchestratorState:
    step = state.get("current_step") or {}
    params = step.get("params", {}) or {}
    target = params.get("target", "unknown")
    await asyncio.sleep(4)
    result = f"ROS2 브리지에 '{target}' 좌표 navigate_to_pose 명령을 전달합니다."
    logs = _append_log(state, {"action": "navigate", "params": params, "result": result})
    return _with_updates(
        state,
        execution_log=logs,
        execution_index=int(state.get("execution_index", 0)) + 1,
        last_action_result=result,
        current_step=None,
        next_action="loop",
    )


async def execute_deliver_object(state: OrchestratorState, runtime: Runtime[Context]) -> OrchestratorState:
    step = state.get("current_step") or {}
    params = step.get("params", {}) or {}
    target = params.get("target") or "delivery_point"
    payload = params.get("object", "물품")
    await asyncio.sleep(4)
    result = (
        f"'{payload}'을 전달하기 위해 '{target}'까지 이동 후 전달 지시를 브리지에 전송합니다."
    )
    logs = _append_log(
        state, {"action": "deliver_object", "params": params, "result": result}
    )
    return _with_updates(
        state,
        execution_log=logs,
        execution_index=int(state.get("execution_index", 0)) + 1,
        last_action_result=result,
        current_step=None,
        next_action="loop",
    )


async def execute_observe_scene(state: OrchestratorState, runtime: Runtime[Context]) -> OrchestratorState:
    step = state.get("current_step") or {}
    params = step.get("params", {}) or {}
    target = params.get("target", "unknown")
    query = params.get("query", "scene status")
    await asyncio.sleep(4)
    result = (
        f"카메라 피드를 캡처하고 '{query}' 질문으로 비전-언어 모델에게 분석을 요청합니다. "
        f"관측 위치: {target}."
    )
    logs = _append_log(
        state, {"action": "observe_scene", "params": params, "result": result}
    )
    return _with_updates(
        state,
        execution_log=logs,
        execution_index=int(state.get("execution_index", 0)) + 1,
        last_action_result=result,
        current_step=None,
        next_action="loop",
    )


async def execute_talk_to_person(state: OrchestratorState, runtime: Runtime[Context]) -> OrchestratorState:
    step = state.get("current_step") or {}
    params = step.get("params", {}) or {}
    target = params.get("target", "person")
    topic = params.get("topic", "topic")
    await asyncio.sleep(4)
    result = (
        f"TTS로 '{target}'에게 '{topic}'을 전달하고, STT를 통해 응답을 수집하여 로그에 저장합니다."
    )
    transcript = f"{target}에게 질문: {topic} (STT 결과 대기)"
    logs = _append_log(
        state,
        {
            "action": "talk_to_person",
            "params": params,
            "result": result,
            "transcript": transcript,
        },
    )
    transcripts = list(state.get("conversation_transcript") or [])
    transcripts.append(transcript)
    return _with_updates(
        state,
        execution_log=logs,
        conversation_transcript=transcripts,
        execution_index=int(state.get("execution_index", 0)) + 1,
        last_action_result=result,
        current_step=None,
        next_action="loop",
    )


async def execute_summarize_mission(state: OrchestratorState, runtime: Runtime[Context]) -> OrchestratorState:
    logs = list(state.get("execution_log") or [])
    try:
        _, _, llm, _ = await _aget_resources(runtime)
        await asyncio.sleep(4)
        log_text = _log_text(logs)
        messages = SUMMARY_PROMPT.format_messages(
            logs=log_text,
            question=state.get("question", ""),
        )
        response = await asyncio.to_thread(llm.invoke, messages)
        summary = getattr(response, "content", response)
    except ResourceInitError as exc:
        summary = f"LLM 요약 실패: {exc}"
    except Exception as exc:  # noqa: BLE001
        summary = f"요약 중 오류 발생: {exc}"

    logs = _append_log(
        state,
        {"action": "summarize_mission", "params": {}, "result": summary},
    )
    return _with_updates(
        state,
        execution_log=logs,
        mission_summary=summary,
        execution_index=int(state.get("execution_index", 0)) + 1,
        last_action_result=summary,
        current_step=None,
        next_action="loop",
    )


def finalize_execution(state: OrchestratorState, runtime: Runtime[Context]) -> OrchestratorState:
    logs = list(state.get("execution_log") or [])
    summary_lines = [state.get("answer", "")]
    if logs:
        summary_lines.append("\n[실행 로그]\n" + _log_text(logs))
    mission_summary = state.get("mission_summary")
    if mission_summary:
        summary_lines.append("\n[임무 요약]\n" + mission_summary)
    answer = "\n".join(line for line in summary_lines if line).strip()
    return _with_updates(state, answer=answer or state.get("answer", ""))


graph = (
    StateGraph(OrchestratorState, context_schema=Context)
    .add_node("classify_intent", classify_intent)
    .add_node("conversation_agent", conversation_agent)
    .add_node("planner_agent", planner_agent)
    .add_node("next_action", next_action)
    .add_node("execute_navigate", execute_navigate)
    .add_node("execute_deliver_object", execute_deliver_object)
    .add_node("execute_observe_scene", execute_observe_scene)
    .add_node("execute_talk_to_person", execute_talk_to_person)
    .add_node("execute_summarize_mission", execute_summarize_mission)
    .add_node("finalize_execution", finalize_execution)
    .set_entry_point("classify_intent")
    .add_conditional_edges(
        "classify_intent",
        lambda state: state.get("intent", "conversation"),
        {"conversation": "conversation_agent", "plan": "planner_agent"},
    )
    .add_edge("conversation_agent", END)
    .add_edge("planner_agent", "next_action")
    .add_conditional_edges(
        "next_action",
        lambda state: state.get("next_action") or "loop",
        {
            "navigate": "execute_navigate",
            "deliver_object": "execute_deliver_object",
            "observe_scene": "execute_observe_scene",
            "talk_to_person": "execute_talk_to_person",
            "summarize_mission": "execute_summarize_mission",
            "done": "finalize_execution",
            "loop": "next_action",
        },
    )
    .add_edge("execute_navigate", "next_action")
    .add_edge("execute_deliver_object", "next_action")
    .add_edge("execute_observe_scene", "next_action")
    .add_edge("execute_talk_to_person", "next_action")
    .add_edge("execute_summarize_mission", "next_action")
    .add_edge("finalize_execution", END)
    .compile(name="RLN RAG Orchestrator")
)
