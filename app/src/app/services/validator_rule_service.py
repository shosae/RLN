"""Rule-based PLAN validator service (Seed 기반)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence
import json
from pathlib import Path


ParamValidator = Callable[[Any], bool]
PlanRule = Callable[[Sequence[Dict[str, Any]]], List[str]]


@dataclass(slots=True)
class PlanValidationResult:
    """Structural rule validation result."""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


# --------------------- 기본 Param Validators ---------------------

def _ensure_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


# --------------------- Seed Loading ---------------------

_SEED_DIR = Path(__file__).resolve().parents[3] / "data" / "seed"


def _load_seed_items(filename: str, key: str):
    path = _SEED_DIR / filename
    try:
        with path.open() as f:
            data = json.load(f)
    except FileNotFoundError as exc:  # pragma: no cover - clearer local error
        raise FileNotFoundError(f"Seed file not found: {path}") from exc
    return data[key]


def load_actions() -> List[Dict[str, Any]]:
    return _load_seed_items("actions.json", "actions")


def load_locations() -> List[Dict[str, Any]]:
    return _load_seed_items("locations.json", "locations")


# --------------------- ActionRule 정의 ---------------------

@dataclass(slots=True)
class ActionRule:
    name: str
    required_params: Mapping[str, ParamValidator] = field(default_factory=dict)
    allow_extra_params: bool = True
    counts_as_core: bool = True

    def validate(self, params: Dict[str, Any], idx: int, result: PlanValidationResult):
        prefix = f"[step {idx}]"
        # 필수 파라미터 검사
        for key, validator in self.required_params.items():
            if key not in params:
                result.add_error(f"{prefix} '{self.name}' params.{key} 값이 누락되었습니다.")
                continue
            if not validator(params[key]):
                result.add_error(f"{prefix} '{self.name}' params.{key} 값이 유효하지 않습니다.")


# --------------------- PLAN 전체 규칙 ---------------------

def _summarize_last_rule(steps: Sequence[Dict[str, Any]]) -> List[str]:
    errors = []
    for idx, st in enumerate(steps[:-1]):
        if st.get("action") == "summarize_mission":
            errors.append(f"[step {idx}] summarize_mission은 마지막에만 등장해야 합니다.")
    return errors


def _basecamp_before_summarize(steps):
    errors = []
    if len(steps) < 2:
        return errors

    if steps[-1].get("action") != "summarize_mission":
        errors.append("마지막 action은 summarize_mission이어야 합니다.")
        return errors

    prev = steps[-2]
    if prev.get("action") != "navigate" or prev.get("params", {}).get("target") != "basecamp":
        errors.append("summarize_mission 직전은 navigate(target='basecamp')여야 합니다.")
    return errors


def _navigate_must_be_followed_by_action(steps):
    """navigate 뒤에는 반드시 action이 1개 존재해야 한다."""
    errors = []
    for i in range(len(steps) - 2):
        if steps[i].get("action") == "navigate":
            if steps[i + 1].get("action") == "navigate":
                errors.append(f"[step {i}] navigate 뒤에는 반드시 action 1개가 따라와야 합니다.")
    return errors


# --------------------- Rule-Based Validator ---------------------

class RuleBasedValidator:
    def __init__(self, actions_seed, locations_seed):
        self.allowed_actions = {a["name"] for a in actions_seed}
        self.allowed_locations = {l["id"] for l in locations_seed}
        self.action_rules: MutableMapping[str, ActionRule] = {}

        # 기본 action rule 등록
        for a in actions_seed:
            req_params = {p: _ensure_nonempty_str for p in a.get("required_params", [])}
            self.action_rules[a["name"]] = ActionRule(name=a["name"], required_params=req_params)

        self.plan_rules = [
            _summarize_last_rule,
            _basecamp_before_summarize,
            _navigate_must_be_followed_by_action,
        ]

    # ----------------------------

    def validate(self, plan: Any, user_query: str | None) -> PlanValidationResult:
        result = PlanValidationResult()
        query_text = user_query or ""

        if not isinstance(plan, dict) or "plan" not in plan:
            result.add_error("PLAN 최상위는 dict이며 'plan' 배열이 있어야 합니다.")
            return result

        steps = plan["plan"]
        if not isinstance(steps, list) or not steps:
            result.add_error("'plan' 배열이 비어 있습니다.")
            return result

        # -------- [1] step 검사 --------
        for idx, st in enumerate(steps):
            if not isinstance(st, dict):
                result.add_error(f"[step {idx}] 단계는 dict여야 합니다.")
                continue

            action = st.get("action")
            params = st.get("params", {})

            # action 유효
            if action not in self.allowed_actions:
                result.add_error(f"[step {idx}] 허용되지 않은 action: {action}")
                continue

            # target 유효
            if action == "navigate":
                tgt = params.get("target")
                if tgt not in self.allowed_locations:
                    result.add_error(f"[step {idx}] 잘못된 target 장소: {tgt}")

            # params 텍스트 규칙: substring + escape 금지
            for key, val in params.items():
                if isinstance(val, str):
                    is_location_ref = key == "target" and val in self.allowed_locations
                    # if not is_location_ref and query_text and val not in query_text:
                    #     result.add_error(f"[step {idx}] params.{key}='{val}' 은 사용자 요청에서 substring으로 찾을 수 없습니다.")
                    if "\\u" in val or "/e" in val:
                        result.add_error(f"[step {idx}] params.{key} 에 escape 문자가 포함되어 있습니다.")

            # 필수 파라미터 rule 검증
            rule = self.action_rules.get(action)
            if rule:
                rule.validate(params, idx, result)

        # -------- [2] PLAN 규칙 검증 --------
        for rule in self.plan_rules:
            result.errors.extend(rule(steps))

        return result


# --------------------- Service Wrapper ---------------------

class PlanValidatorService:
    def __init__(self):
        actions_seed = load_actions()
        locations_seed = load_locations()
        self._validator = RuleBasedValidator(actions_seed, locations_seed)

    def validate(self, plan: Any, user_query: str | None = None) -> PlanValidationResult:
        return self._validator.validate(plan, user_query)


_default_service = PlanValidatorService()


def validate_plan(plan: Any, user_query: str | None = None) -> PlanValidationResult:
    return _default_service.validate(plan, user_query)
