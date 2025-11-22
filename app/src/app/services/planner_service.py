"""Planner LLM 호출과 PLAN 생성/검증 로직 (Fully Data-Driven)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field, field_validator, model_validator

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


# ---------- PLAN 모델 ----------


class Step(BaseModel):
    action: str
    params: dict = Field(default_factory=dict)

    @field_validator("action")
    def _ensure_allowed_action(cls, value: str) -> str:
        if value not in _get_allowed_actions():
            allowed = ", ".join(sorted(_get_allowed_actions()))
            raise ValueError(f"허용되지 않은 action '{value}'. 사용 가능: {allowed}")
        return value

    @model_validator(mode="after")
    def _validate_params(cls, values: "Step"):
        action = values.action
        params = values.params or {}
        if action == "navigate":
            target = params.get("target")
            if target not in _get_allowed_locations():
                allowed = ", ".join(sorted(_get_allowed_locations()))
                raise ValueError(f"navigate target '{target}' 가 허용된 장소({allowed})에 없습니다.")
        return values


class Plan(BaseModel):
    plan: List[Step] = Field(default_factory=list)


# ---------- HINTS (메타 규칙만 남김) ----------

DEFAULT_HINTS = r"""
[PLAN 생성 원칙]
1. **순서 준수:** 사용자가 여러 장소나 행동을 언급했다면, 반드시 **발화된 순서대로** PLAN을 구성해야 한다.
2. **데이터 기반:** 오직 제공된 `Location List`와 `Action List`에 정의된 항목만 사용할 수 있다. 없는 장소나 행동은 생성하지 않는다.
3. **이동 우선(Implicit Navigation):** "복도 확인해"처럼 행동의 대상이 **특정 장소**라면, 반드시 그 장소로 **먼저 이동(navigate)한 후** 행동해야 한다. (바로 action 금지)

[파라미터 추출 규칙 (nl_params)]
- Action 정의에 `nl_params`(예: question, instruction)가 있다면, 그 값은 **사용자 원문에서 해당 부분의 구절(Substring)을 그대로 복사**해야 한다.
- 요약하거나 단어를 바꾸지 말고, 문맥상 필요한 부분을 통째로 발췌한다.
- 예시: "(목표)가 있는지 확인해" -> question="(목표)가 있는지" (O)
"""


PLAN_SYSTEM_PROMPT = r"""
너는 이동 로봇을 위한 Task Planner이다.
아래 제공되는 **Action List**와 **Location List** 데이터만을 사용하여, 사용자의 요청을 실행 가능한 JSON PLAN으로 변환하라.

============================================================
[1] Action List (사용 가능한 행동)
============================================================
{% for a in actions -%}
### {{ a.name }}
- Description: {{ a.description }}
- Required Params: {{ a.required_params }}
{% if a.trigger_phrases %}- Triggers: {{ a.trigger_phrases | join(", ") }}{% endif %}
{% if a.nl_params %}- NL Params: {{ a.nl_params | join(", ") }} (원문 발췌 필수){% endif %}
{% endfor %}

============================================================
[2] Location List (사용 가능한 장소)
============================================================
{% for loc in locations -%}
- {{ loc.id }}: {{ loc.description }}
{% endfor %}

============================================================
[3] PLAN 구조 생성 규칙 (절대 준수 - 위반 시 로봇 고장남)
============================================================
**규칙 A. 암묵적 장소 추론 (Implicit Location Inference)**
사용자가 "이동해"라고 명시하지 않았더라도, **행동의 대상이 '특정 장소'라면 그곳으로 이동하는 단계를 자동으로 추가**해야 한다.

- **Case 1 (명시적):** "복도로 가서 확인해" -> `Navigate("corridor")` -> `observe`
- **Case 2 (암묵적):** "복도에 사람이 있는지 확인해" -> **문맥상 '복도'로 가야 함을 스스로 추론** -> `Navigate("corridor")` -> `observe`
- **주의:** 현재 위치와 다른 장소를 언급하며 행동을 지시하면, 반드시 그 장소로 `Navigate` 단계를 먼저 넣어야 한다.

**규칙 B. 작업 루프 (Work Loop)**
사용자가 의도한 장소 순서대로 아래 패턴을 반복한다.
   1) `Navigate` (target="장소ID")
   2) `action` (그 장소에서 할 일)

**규칙 C. 금지 사항**
- ❌ `Navigate` 없이 `action`만 단독으로 출력하지 말 것.
- ❌ `deliver`, `observe` 등 action이 `Navigate`보다 먼저 나오면 안 됨.

============================================================
[4] 파라미터 채우기 규칙
============================================================
- **target**: 반드시 위 [2] Location List에 있는 `id`만 사용해야 한다.
- **NL Params (question, instruction 등)**:
  - Action 정의에 `nl_params`가 있다면, 그 값은 **사용자 원문에서 해당 부분의 구절(Substring)을 그대로 복사**해야 한다.
  - 요약하거나 단어를 바꾸지 말고, 문맥상 필요한 부분을 통째로 발췌한다.

============================================================
[5] 미션 종료 규칙
============================================================
사용자가 요청한 모든 [이동 -> 행동] 루프가 끝난 후,
**절대로 멈추지 말고** 무조건 아래 두 단계를 마지막에 추가하여 복귀하라.

1. navigate(target="basecamp")
2. summarize_mission

============================================================
[6] 출력 형식 예시
============================================================
사용자 요청: "교수님 방에 서류 갖다주고, 복도에 불 났는지 확인하고 와"

{
  "plan": [
    { "action": "navigate", "params": { "target": "professor_office" } },       <-- [1] 첫 번째 장소 이동
    { "action": "deliver_object", "params": { "target": "professor_office" } }, <-- [1] 행동 수행

    { "action": "navigate", "params": { "target": "corridor_center" } },        <-- [2] 두 번째 장소 이동 (중요!)
    { "action": "observe_scene", "params": { "question": "불 났는지 확인해" } }, <-- [2] 행동 수행 (원문 반영)

    { "action": "navigate", "params": { "target": "basecamp" } },               <-- [3] 복귀 (필수)
    { "action": "summarize_mission", "params": {} }                             <-- [3] 보고 (필수)
  ]
}"""

USER_QUESTION_PROMPT = """
사용자의 요청:
{{question}}

[필수 주의사항]
사용자 요청을 모두 수행한 뒤, 반드시!! 아래 두 단계를 추가하여 계획을 끝내라.
1. navigate(target="basecamp")
2. summarize_mission
이것을 어기면 시스템 에러가 발생한다.
"""
USER_HINTS_PROMPT = """
참고 힌트(HINTS):
{{hints}}
"""

_SEED_DIR = Path(__file__).resolve().parents[3] / "data" / "seed"
_ACTIONS_CACHE: List[dict] | None = None
_LOCATIONS_CACHE: List[dict] | None = None
_ALLOWED_ACTIONS: set[str] | None = None
_ALLOWED_LOCATIONS: set[str] | None = None


def _load_seed_items(filename: str, key: str):
    path = _SEED_DIR / filename
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Seed file not found: {path}") from exc
    return data[key]


def _get_actions_seed() -> List[dict]:
    global _ACTIONS_CACHE
    if _ACTIONS_CACHE is None:
        _ACTIONS_CACHE = _load_seed_items("actions.json", "actions")
    return _ACTIONS_CACHE


def _get_locations_seed() -> List[dict]:
    global _LOCATIONS_CACHE
    if _LOCATIONS_CACHE is None:
        _LOCATIONS_CACHE = _load_seed_items("locations.json", "locations")
    return _LOCATIONS_CACHE


def _get_allowed_actions() -> set[str]:
    global _ALLOWED_ACTIONS
    if _ALLOWED_ACTIONS is None:
        _ALLOWED_ACTIONS = {a["name"] for a in _get_actions_seed()}
    return _ALLOWED_ACTIONS


def _get_allowed_locations() -> set[str]:
    global _ALLOWED_LOCATIONS
    if _ALLOWED_LOCATIONS is None:
        _ALLOWED_LOCATIONS = {loc["id"] for loc in _get_locations_seed()}
    return _ALLOWED_LOCATIONS

@dataclass(slots=True)
class PlannerDependencies:
    llm: BaseChatModel

def _build_plan_chain(llm: BaseChatModel):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                PLAN_SYSTEM_PROMPT,
                template_format="jinja2",
            ),
            HumanMessagePromptTemplate.from_template(
                USER_QUESTION_PROMPT,
                template_format="jinja2",
            ),
            HumanMessagePromptTemplate.from_template(
                USER_HINTS_PROMPT,
                template_format="jinja2",
            ),
        ]
    )
    return prompt | llm.with_structured_output(Plan)

def generate_plan(
    question: str,
    llm: BaseChatModel,
    waypoint_docs: List[Document] | None = None,
) -> Tuple[dict, List[Document]]:
    actions = _get_actions_seed()
    locations = _get_locations_seed()
    docs = waypoint_docs or []

    chain = _build_plan_chain(llm)

    # hints에는 이제 특정 액션 이름이 들어가지 않음 (Generic Logic)
    plan_obj = chain.invoke(
        {
            "question": question,
            "hints": DEFAULT_HINTS, 
            "actions": actions,
            "locations": locations,
        }
    )

    plan_dict = plan_obj.model_dump() if isinstance(plan_obj, Plan) else plan_obj
    return plan_dict, docs
