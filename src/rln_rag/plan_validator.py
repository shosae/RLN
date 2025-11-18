from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

ALLOWED_ACTIONS = {"navigate", "deliver_object", "observe_scene", "wait", "summarize_mission"}
CORE_ACTIONS = {"navigate", "deliver_object", "observe_scene", "wait"}


@dataclass(slots=True)
class PlanValidationResult:
    """Stores structural errors and best-effort warnings for a PLAN JSON."""

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


def _ensure_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _find_previous_navigation(steps: Sequence[Dict[str, Any]], current_idx: int, target: str | None = None) -> bool:
    """Return True if there is a matching navigate step before the given index."""
    for idx in range(current_idx - 1, -1, -1):
        step = steps[idx]
        if not isinstance(step, dict):
            continue
        if step.get("action") != "navigate":
            continue
        step_target = step.get("params", {}).get("target")
        if target is None or step_target == target:
            return True
    return False


def validate_plan(plan: Any) -> PlanValidationResult:
    """
    Validate that the PLAN JSON obeys the enforced schema and high-level domain rules.
    This is a prototype validator intentionally limited to structural checks and a few heuristics.
    """
    result = PlanValidationResult()
    if not isinstance(plan, dict):
        result.add_error("PLAN JSON 최상위 객체가 dict 형태가 아닙니다.")
        return result

    steps = plan.get("plan")
    if not isinstance(steps, list):
        result.add_error('"plan" 필드는 리스트(List)여야 합니다.')
        return result
    if not steps:
        result.add_error('"plan" 배열이 비어 있습니다.')
        return result

    core_action_seen = False
    requires_home = False
    returned_home = False
    current_location = None
    for idx, raw_step in enumerate(steps):
        prefix = f"[step {idx}]"
        if not isinstance(raw_step, dict):
            result.add_error(f"{prefix} 각 단계는 dict로 표현되어야 합니다.")
            continue

        action = raw_step.get("action")
        params = raw_step.get("params")

        if action not in ALLOWED_ACTIONS:
            result.add_error(f"{prefix} 허용되지 않은 action '{action}'이(가) 포함되었습니다.")
            continue
        if not isinstance(params, dict):
            result.add_error(f"{prefix} params는 dict여야 합니다.")
            continue

        if action in CORE_ACTIONS:
            core_action_seen = True

        if action == "navigate":
            target = params.get("target")
            if not _ensure_nonempty_str(target):
                result.add_error(f"{prefix} navigate params.target은 비어 있지 않은 문자열이어야 합니다.")
            else:
                current_location = target
                if target == "basecamp":
                    returned_home = True
                    requires_home = False
                else:
                    requires_home = True
                    returned_home = False

        elif action == "deliver_object":
            if params and not all(isinstance(key, str) for key in params.keys()):
                result.add_error(f"{prefix} deliver_object params의 키는 문자열이어야 합니다.")
            if not _find_previous_navigation(steps, idx):
                result.add_warning(f"{prefix} deliver_object 전에 navigate 단계가 없습니다.")
            requires_home = True
            returned_home = False

        elif action == "observe_scene":
            target = params.get("target")
            query = params.get("query")
            if not _ensure_nonempty_str(target):
                result.add_error(f"{prefix} observe_scene params.target이 비어 있습니다.")
            if not _ensure_nonempty_str(query):
                result.add_error(f"{prefix} observe_scene params.query가 비어 있습니다.")
            requires_home = True
            returned_home = False

        elif action == "wait":
            seconds = params.get("seconds", 0)
            if not isinstance(seconds, (int, float)) or seconds < 0:
                result.add_error(f"{prefix} wait params.seconds는 0 이상의 숫자여야 합니다.")
            requires_home = True
            returned_home = False

        elif action == "summarize_mission":
            if params:
                result.add_error(f"{prefix} summarize_mission params는 빈 객체여야 합니다.")
            if not core_action_seen:
                result.add_error(f"{prefix} summarize_mission 전에 핵심 action이 최소 1회 이상 등장해야 합니다.")
            if requires_home and not returned_home:
                result.add_error(
                    f"{prefix} summarize_mission 실행 전에 basecamp로 navigate 단계가 필요합니다."
                )
            requires_home = False

    if not core_action_seen:
        result.add_error("navigate/deliver_object/observe_scene/wait 중 하나 이상의 핵심 action이 필요합니다.")

    return result
