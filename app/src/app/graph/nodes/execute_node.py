"""Plan execution node helpers."""

from __future__ import annotations

from typing import Dict, Any, List

from app.services.executor_service import ExecutorService


def execute_step(step: Dict[str, Any], executor: ExecutorService) -> Dict[str, Any]:
    """단일 action을 executor에 위임."""

    action = (step or {}).get("action")
    params = (step or {}).get("params") or {}

    if action == "navigate":
        return executor.navigate(params.get("target", "unknown"))
    if action == "observe_scene":
        return executor.observe_scene(
            params.get("target", "unknown"), params.get("query")
        )
    if action == "deliver_object":
        return executor.deliver_object(
            params.get("receiver"), params.get("item")
        )
    if action == "summarize_mission":
        return executor.summarize_mission(params.get("summary"))
    if action == "report":
        return executor.report(params.get("content", ""))
    if action == "wait":
        return executor.wait(params.get("seconds", 1))

    return {"status": "error", "message": f"Unsupported action {action}"}


def execute_plan(plan_steps: List[Dict[str, Any]], executor: ExecutorService) -> List[Dict[str, Any]]:
    """PLAN 전체를 순차 실행."""

    results: List[Dict[str, Any]] = []
    for step in plan_steps or []:
        result = execute_step(step, executor)
        results.append({"step": step, "result": result})
    return results
