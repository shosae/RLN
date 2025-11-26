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
- Action 정의에 `nl_params`(예: question, instruction, target)가 있다면, 그 값은 **사용자 원문에서 해당 부분의 구절(Substring)을 그대로 복사**해야 한다.
- 요약하거나 단어를 바꾸지 말고, 문맥상 필요한 부분을 통째로 발췌한다.
- 예시: "(목표)가 있는지 확인해" -> question="(목표)가 있는지" (O)
"""


PLAN_SYSTEM_PROMPT = r"""
너는 실내 자율주행 이동 로봇을 위한 Task Planner이다.
사용자의 한국어 요청과 아래 Action/Location 목록을 기반으로, 실행 가능한 PLAN을 JSON으로 생성한다.

============================================================
[0] 출력 형식 (반드시 준수)
============================================================

- 최종 출력은 다음 스키마를 따르는 JSON 객체 하나여야 한다.

{
  "plan": [
    {
      "action": "<Action List 중 하나의 이름>",
      "params": { /* action별 파라미터 */ }
    },
    ...
  ]
}

- JSON 앞뒤에 자연어 설명, 마크다운 코드 블록, 주석, 기타 문자를 절대 추가하지 마라.
- 오직 위 스키마를 따르는 JSON만 출력한다.

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
[3] PLAN 구성 규칙
============================================================

(1) 순서 유지
- 사용자가 여러 장소나 행동을 언급했다면, **발화된 순서대로** PLAN을 구성한다.

(2) 기본 패턴: [이동 → 행동들]
- 새로운 장소에서 행동을 해야 한다면, 항상 다음 패턴으로 작성한다.
  1) { "action": "navigate", "params": { "target": "<장소ID>" } }
  2) 같은 장소에서 수행할 action step 1개 이상

- 하나의 장소에서 여러 행동을 지시했다면,
  - navigate 한 번 뒤에 **같은 target을 가지는 action step들을 연속으로 배치**한다.

(3) 암묵적 이동 (Implicit Navigation)
- 사용자 발화에 "가", "이동해" 등의 표현이 없어도,
  특정 장소에서 수행해야 의미가 통하는 행동이면 먼저 그 장소로 navigate 해야 한다.
  - 예: "복도에 사람 있는지 확인하고 ~.."
    1) { "action": "navigate", "params": { "target": "corridor_center" } }
    2) { "action": "observe_scene", "params": { ... } }
    3) {~..}

(4) 금지 사항
- ❌ navigate 없이 deliver/observe 등 action만 단독으로 두지 말 것.
- ❌ basecamp가 아닌 다른 장소에서 summarize_mission을 호출하지 말 것.
- ❌ Action List / Location List에 없는 이름을 새로 만들지 말 것.

============================================================
[4] 파라미터 채우기 규칙
============================================================

- target:
  - 항상 [2] Location List의 id 값 중 하나를 사용한다.

- nl_params (예: question, instruction):
  - Action 정의에 nl_params가 있다면, 해당 값은 **사용자 원문에서 그 부분을 그대로 복사**해 채운다.
  - 의미를 변경하는 요약/재구성은 금지한다.
  - 예: "불 났는지 확인해" → question="불 났는지 확인해"

============================================================
[5] 미션 종료 규칙
============================================================

사용자가 요청한 모든 [이동 → 행동들]이 끝난 후,
항상 다음 두 step을 마지막에 추가해야 한다.

1) { "action": "navigate", "params": { "target": "basecamp" } }
2) { "action": "summarize_mission", "params": {} }

!!!! ** 매우 중요** !!!!
또한 사용자의 요청이 "와", "돌아와", "하고 와", "갔다와", "해주고 와" 등
복귀를 암시하는 표현으로 끝나는 경우,
해당 표현이 있든 없든 무조건 basecamp로 복귀 후 summarize_mission을 실행해야 한다.
**문장 후반부에서 복귀를 의미하는 표현이 있으면 반드시
{navigate basecamp → summarize_mission}를 마지막에 넣어야 한다.**

============================================================
[6] 예시
============================================================

사용자 요청: "교수님 방에 서류 갖다주고, 복도에 불 났는지 확인하고 와"

{
  "plan": [
    { "action": "navigate", "params": { "target": "professor_office" } },
    { "action": "deliver_object", "params": { "target": "서류" } },

    { "action": "navigate", "params": { "target": "corridor_center" } },
    { "action": "observe_scene", "params": { "question": "불 났는지 확인해" } },

    { "action": "navigate", "params": { "target": "basecamp" } },
    { "action": "summarize_mission", "params": {} }
  ]
}
"""

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
