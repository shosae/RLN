"""Planner LLM 호출과 PLAN 생성/검증 로직 (Fully Data-Driven)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

try:
    from langchain_core.prompts import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        SystemMessagePromptTemplate,
    )
except ModuleNotFoundError:  # pragma: no cover
    from langchain.prompts import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        SystemMessagePromptTemplate,
    )


# ---------- PLAN 모델 ----------


class Step(BaseModel):
    action: str
    params: dict = Field(default_factory=dict)


class Plan(BaseModel):
    plan: List[Step] = Field(default_factory=list)


# ---------- HINTS (메타 규칙만 남김) ----------

DEFAULT_HINTS = r"""
[PLAN 생성 원칙]
1. **순서 준수:** 사용자가 여러 장소나 행동을 언급했다면, 반드시 **발화된 순서대로** PLAN을 구성해야 한다.
2. **데이터 기반:** 오직 제공된 `Location List`와 `Action List`에 정의된 항목만 사용할 수 있다. 없는 장소나 행동은 생성하지 않는다.

[파라미터 추출 규칙 (nl_params)]
- Action 정의에 `nl_params`(예: question, instruction)가 있다면, 그 값은 **사용자 원문에서 해당 부분의 구절(Substring)을 그대로 복사**해야 한다.
- 요약하거나 단어를 바꾸지 말고, 문맥상 필요한 부분을 통째로 발췌한다.
- 예시: "강아지가 있는지 확인해" -> question="강아지가 있는지" (O)
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
너는 **순간이동을 할 수 없는 로봇**이다.
따라서 어떤 장소에서 Action을 수행하려면, **반드시 먼저 그 장소로 이동(navigate)해야 한다.**

사용자가 언급한 장소들의 순서를 따라 아래 **[이동 -> 행동] 패턴**을 엄격히 지켜라.

**(패턴 정의)**
1. **Navigate Step**: `{"action": "navigate", "params": {"target": "목적지ID"}}`
2. **Action Step**: 그 장소에서 해야 할 Action (예: deliver, observe 등)

**[금지 사항 - Negative Constraints]**
- ❌ 절대로 `Navigate` 없이 `action`만 단독으로 출력하지 말 것.
- ❌ `deliver_object`, `observe_scene` 등이 `Navigate`보다 먼저 나오면 안 됨.
- ❌ 같은 장소에서 여러 행동을 하더라도, 최초 1회는 반드시 `Navigate`가 선행되어야 함.

============================================================
[4] 파라미터 채우기 규칙 (Data Consistency)
============================================================
- **target**: 반드시 위 [2] Location List에 있는 `id`만 사용해야 한다.
- **NL Params (question, instruction 등)**:
  - Action 정의에 `nl_params`가 있다면, 그 값은 **사용자 원문에서 해당 부분의 구절(Substring)을 그대로 복사**해야 한다.
  - 요약하거나 단어를 바꾸지 말고, 문맥상 필요한 부분을 통째로 발췌한다.

============================================================
[5] 미션 종료 규칙 (Closing Sequence)
============================================================
사용자가 요청한 모든 [이동 -> 행동] 루프가 끝난 후,
**절대로 멈추지 말고** 무조건 아래 두 단계를 마지막에 추가하여 복귀하라.

1. navigate(target="basecamp")
2. summarize_mission

============================================================
[6] 출력 형식 예시
============================================================
{
  "plan": [
    { "action": "navigate", "params": { "target": "professor_office" } },  <-- 먼저 이동!
    { "action": "deliver_object", "params": { "target": "professor_office" } }, <-- 행동!
    ... (중간 반복) ...,
    { "action": "navigate", "params": { "target": "basecamp" } }, <-- 복귀 필수!
    { "action": "summarize_mission", "params": {} } <-- 보고 필수!
  ]
}
"""

USER_QUESTION_PROMPT = """
사용자의 요청:
{{question}}
"""

USER_HINTS_PROMPT = """
참고 힌트(HINTS):
{{hints}}
"""

_SEED_DIR = Path(__file__).resolve().parents[3] / "data" / "seed"

def _load_seed_items(filename: str, key: str):
    path = _SEED_DIR / filename
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Seed file not found: {path}") from exc
    return data[key]

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
    actions = _load_seed_items("actions.json", "actions")
    locations = _load_seed_items("locations.json", "locations")
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