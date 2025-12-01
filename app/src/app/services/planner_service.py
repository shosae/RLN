"""Planner LLM 호출과 PLAN 생성/검증 로직 (Fully Data-Driven)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError

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
                raise ValueError(
                    f"navigate target '{target}' 가 허용된 장소({allowed})에 없습니다."
                )
        return values


class Plan(BaseModel):
    plan: List[Step] = Field(default_factory=list)


class PlannerUnsupportedError(RuntimeError):
    """Raised when planner output references unsupported actions or locations."""

    pass


def _infer_unsupported_message(error_text: str) -> str:
    lowered = (error_text or "").lower()
    if ("action" in lowered) or ("허용되지 않은 action" in error_text):
        return "해당 행동은 수행할 수 없어요. 다른 요청을 도와드릴까요?"
    if ("target" in lowered) or ("장소" in lowered):
        return "해당 장소로는 갈 수 없어요. 다른 요청을 도와드릴까요?"
    return "요청하신 행동이나 위치는 아직 지원하지 못해요. 다른 요청을 도와드릴까요?"


# ---------- HINTS (메타 규칙만 남김) ----------

DEFAULT_HINTS = r"""
[PLAN 생성 원칙]
1. **순서 준수:** 사용자가 여러 장소나 행동을 언급했다면, 반드시 **발화된 순서대로** PLAN을 구성해야 한다.
2. **데이터 기반:** 오직 제공된 `Location List`와 `Action List`에 정의된 항목만 사용할 수 있다. 없는 장소나 행동은 생성하지 않는다.
3. **이동 우선(Implicit Navigation):** "복도 확인해"처럼 행동의 대상이 **특정 장소**라면, 반드시 그 장소로 **먼저 이동(navigate)한 후** 행동해야 한다. (바로 action 금지)
4. **미션 종료 하드 규칙:** 사용자가 "거기 계속 있어라", "대기해", "머물러 있어", "기다려" 등 **명시적으로 특정 위치에 계속 머무르라고 지시하는 표현**을 사용하지 않은 이상,
   모든 PLAN은 **반드시** basecamp로 복귀한 뒤 summarize_mission으로 끝나야 한다.
   (예외가 아닌데 마지막 두 step이 navigate(basecamp)와 summarize_mission이 아니면 잘못된 PLAN이다.)
5. **순수 이동 미션 처리:** 사용자가 특정 위치만 언급하고
   "갔다 와", "다녀와", "들렀다 와", "갔다가 와" 정도의 표현만 사용하며,
   무엇을 배달하거나(커피, 서류 등) 무엇을 확인하라는 내용이 없다면,
   그 미션은 **단순 방문(순수 이동)** 으로 해석한다.
   - 이 경우 plan에는 해당 위치로의 `navigate`만 넣고,
     추가 action(deliver_object, observe_scene 등)을 새로 만들지 않는다.
   - 복귀와 요약은 전역 규칙에 따라
     `navigate(target="basecamp")` 와 `summarize_mission` 으로 마무리한다.
6. **deliver_object 의미와 제약:**
   - `deliver_object`는 로봇이 **이미 사람이 실어준 물건을 들고** 지정된 위치로 이동한 뒤,
     그 위치에서 사람이 물건을 가져갈 때까지 잠시 대기하고 전달하는 행동이다.
   - 물건을 찾아다니거나, 집어서 다시 다른 곳으로 가져오는 행동이 아니다.
   - 항상 **해당 위치로의 navigate 이후**에 나와야 하며,
     그 navigate와 **같은 장소 컨텍스트**에서 바로 이어지는 step으로 둔다.
   - `navigate` 없이 단독으로 사용하거나, 전혀 다른 장소로 바로 이어지는 `deliver_object`를 만들지 말아야 한다.

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
[0-1] PLAN Tail 하드 규칙 (위반 시 잘못된 PLAN)
============================================================

- **기본 원칙:**
  - 사용자가 "거기 계속 있어라", "그 자리에서 대기해", "그 위치에서 계속 봐줘",
    "머물러 있어"처럼 **명시적으로 특정 위치에 머무르라고 지시하는 표현**을 사용하지 않은 이상,
    모든 미션은 **반드시 basecamp로 복귀한 뒤 summarize_mission으로 끝나야 한다.**

- 따라서 plan 배열의 **마지막 두 step은 항상** 아래와 같아야 한다.

  1) { "action": "navigate", "params": { "target": "basecamp" } }
  2) { "action": "summarize_mission", "params": {} }

- 사용자가 "와", "돌아와", "하고 와", "갔다 와", "해주고 와" 등
  복귀를 암시하는 표현을 쓰든 쓰지 않든,
  **예외적 지시(계속 있어, 대기해 등)가 없다면** 위 두 step을 반드시 마지막에 넣어야 한다.

- 예외:
  - 오직 사용자 발화에
    - "거기 계속 있어", "그 자리에서 대기해", "머물러 있어",
      "그 위치에서 계속 보고 있어"
    와 같이 **특정 위치에서 계속 머무르라고 분명하게 지시하는 표현**이 포함된 경우에만,
    basecamp로 복귀하지 않고 해당 위치에서 미션을 끝낼 수 있다.
  - 이 예외가 아닌데 마지막 두 step이 위 형태가 아니라면,
    그 PLAN은 **잘못된 PLAN**이다.

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

추가 설명 – deliver_object:
- `deliver_object`는 로봇이 **이미 사람이 적재해 준 물건**을 들고,
  지정된 위치로 이동한 뒤 **그 자리에서 사람이 가져갈 때까지 잠시 대기하며 전달하는 행동**이다.
- 물건을 찾아서 집어 들거나, 다른 곳으로 다시 가져오는 픽업/회수 미션이 아니다.
- 항상 해당 위치에 대한 `navigate` 이후에 바로 이어지는 step으로 사용해야 한다.

예:
  1) { "action": "navigate", "params": { "target": "professor_office" } }
  2) { "action": "deliver_object", "params": { ... } }   ← 같은 위치에서 바로 배달

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

- 특히 `deliver_object`는 항상
  - 직전에 해당 위치로의 `navigate`가 나와 있고,
  - 그 `navigate`와 같은 장소 컨텍스트에서 바로 이어져야 한다.
  - 다른 장소로 `navigate`한 뒤에 곧바로 `deliver_object`를 두는 것은 허용되지 않는다.

(3) 암묵적 이동 (Implicit Navigation)
- 사용자 발화에 "가", "이동해" 등의 표현이 없어도,
  특정 장소에서 수행해야 의미가 통하는 행동이면 먼저 그 장소로 navigate 해야 한다.
  - 예: "복도에 사람 있는지 확인하고 ~.."
    1) { "action": "navigate", "params": { "target": "corridor_center" } }
    2) { "action": "observe_scene", "params": { ... } }
    3) {~..}

(4) 금지 사항
- ❌ `navigate` 없이 `deliver_object`, `observe_scene` 등 action만 단독으로 두지 말 것.
- ❌ basecamp가 아닌 다른 장소에서 `summarize_mission`을 호출하지 말 것.
- ❌ Action List / Location List에 없는 이름을 새로 만들지 말 것.
- ❌ `deliver_object`를, 직전에 해당 위치로 이동하지 않은 상태에서 다른 장소로 연결되는 step 사이에 끼워 넣지 말 것.

(5) 순수 이동 미션
- 사용자가 특정 위치만 언급하고
  "갔다 와", "다녀와", "들렀다 와", "갔다가 와"처럼
  **단순히 다녀오는 표현만 있고**,  
  무엇을 배달해 주거나(커피, 서류 등),  
  무엇을 확인·관찰하라는 질문이 없다면,
  이 미션은 **순수 이동(단순 방문)** 으로 처리한다.
- 이 경우 PLAN은 다음과 같이 구성한다.
  1) { "action": "navigate", "params": { "target": "<해당 위치>" } }
  2) 마지막에는 전역 규칙에 따라
     { "action": "navigate", "params": { "target": "basecamp" } },
     { "action": "summarize_mission", "params": {} } 를 붙인다.
- 이때 `deliver_object`, `observe_scene` 등의 추가 action을
  사용자가 명시적으로 요구하지 않았다면 **절대 생성하지 않는다.**

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
[5] 미션 종료 규칙 (요약)
============================================================

- 다시 한 번 강조한다.
  - 사용자가 "거기 계속 있어라", "대기해", "머물러 있어" 등 **특정 위치에 머무르라는 지시**를 하지 않은 경우,
    모든 PLAN은 **반드시** 아래 두 step으로 끝나야 한다.

    1) { "action": "navigate", "params": { "target": "basecamp" } }
    2) { "action": "summarize_mission", "params": {} }

- 이 규칙을 지키지 않은 PLAN은 잘못된 PLAN으로 간주된다.

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

사용자 요청: "화장실 앞에서 사람한테 커피만 전해주고 와"

{
  "plan": [
    { "action": "navigate", "params": { "target": "restroom_front" } },
    { "action": "deliver_object", "params": { "target": "커피" } },

    { "action": "navigate", "params": { "target": "basecamp" } },
    { "action": "summarize_mission", "params": {} }
  ]
}

사용자 요청: "화장실 갔다와"

{
  "plan": [
    { "action": "navigate", "params": { "target": "restroom_front" } },

    { "action": "navigate", "params": { "target": "basecamp" } },
    { "action": "summarize_mission", "params": {} }
  ]
}

"""

USER_QUESTION_PROMPT = """
사용자의 요청:
{{question}}

[필수 주의사항 – 하드 규칙]
- 사용자가 "거기 계속 있어라", "그 자리에서 대기해", "머물러 있어" 등
  **특정 위치에 계속 머무르라고 명시적으로 지시하지 않은 경우**,
  모든 PLAN은 무조건 아래 두 단계를 마지막에 포함해야 한다.
  이를 지키지 않으면 잘못된 PLAN이다.

1. navigate(target="basecamp")
2. summarize_mission
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
    try:
        plan_obj = chain.invoke(
            {
                "question": question,
                "hints": DEFAULT_HINTS,
                "actions": actions,
                "locations": locations,
            }
        )
    except (ValidationError, OutputParserException) as exc:
        raise PlannerUnsupportedError(
            _infer_unsupported_message(str(exc))
        ) from exc

    plan_dict = plan_obj.model_dump() if isinstance(plan_obj, Plan) else plan_obj
    return plan_dict, docs
