"""Planner LLM 호출과 PLAN 생성/검증 로직."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from app.services.validator_service import validate_plan, PlanValidationResult

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


# ---------- 힌트/컨텍스트 ----------

CORRIDOR_KEYWORDS = ["복도", "corridor"]
LOUNGE_KEYWORDS = ["라운지", "휴게실", "휴게 공간", "lounge"]
RESTROOM_KEYWORDS = ["화장실 앞", "화장실입구", "restroom"]
REPORT_KEYWORDS = [
    "다시 알려줘",
    "다시알려줘",
    "나한테 와서",
    "보고 와서",
    "보고와",
    "결과를 알려줘",
    "결과알려줘",
    "나한테 보고",
    "내게 다시",
    "내게다시",
]
PROFESSOR_KEYWORDS = ["교수", "professor"]


def _format_context(docs: List[Document]) -> str:
    if not docs:
        return "RAG 결과가 없습니다."
    entries: List[str] = []
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        source = (
            meta.get("source")
            or meta.get("file_path")
            or meta.get("title")
            or f"doc-{idx}"
        )
        snippet = " ".join(doc.page_content.split())
        entries.append(f"[{idx}] source: {source}\nsummary: {snippet}")
    return "\n\n".join(entries)


def _build_hints(question: str) -> str:
    q = question.strip()
    q_lower = q.lower()
    hints: List[str] = []

    professor = any(k in q for k in PROFESSOR_KEYWORDS)
    corridor = any(k in q for k in CORRIDOR_KEYWORDS)
    lounge = any(k in q for k in LOUNGE_KEYWORDS)
    restroom = any(k in q for k in RESTROOM_KEYWORDS)
    needs_report = any(k in q for k in REPORT_KEYWORDS)
    noisy = "시끄럽" in q or "시끄러" in q or "소음" in q_lower
    lighting = ("불" in q and ("켜" in q or "꺼" in q)) or ("조명" in q)

    if professor:
        corridor = lounge = restroom = False
        hints.append(
            '- 교수님 관련 요청: navigate → target "professor_office" 후 '
            'talk_to_person target "professor"를 적절히 사용할 수 있다.'
        )
        hints.append(
            '- 교수님에게 질문하는 경우 PLAN 순서는 '
            '[navigate professor_office → talk_to_person(target "professor") → '
            'navigate basecamp → summarize_mission]을 따르고, '
            '사용자가 보고를 요구했으면 talk_to_person(target "user")를 마지막에 추가할 것.'
        )
    else:
        hints.append(
            "- 교수님 관련 action(교수님 이동/대화)은 사용하지 말 것."
        )

    if corridor:
        hints.append(
            '- 복도 관련: 기본 위치는 "corridor_center"를 우선 고려하고, '
            'navigate / observe_scene에서 해당 target을 사용할 것.'
        )
    if lounge:
        hints.append(
            '- 라운지/휴게 공간 관련: 기본 위치는 "lounge"를 사용하고, '
            'navigate / observe_scene에서 해당 target을 사용할 것.'
        )
    if restroom:
        hints.append(
            '- 화장실 앞 관련: 기본 위치는 "restroom_front"를 사용하고, '
            'navigate / observe_scene에서 해당 target을 사용할 것.'
        )

    if noisy:
        hints.append(
            "- 소음/시끄러움 확인: observe_scene의 query에 사람 수나 소란 여부를 반드시 포함할 것."
        )
    if lighting:
        hints.append(
            "- 조명/불 확인: observe_scene의 query에 조명 상태(켜짐/꺼짐)를 명시할 것."
        )

    if needs_report:
        hints.append(
            '- 사용자에게 다시 알려주라는 요청이 있으므로, 핵심 작업 후 PLAN 마지막에 '
            'navigate target "basecamp"로 복귀하고 summarize_mission으로 결과를 보고할 것.'
        )

    if not corridor and not lounge and not restroom and not professor:
        hints.append(
            "- 기본적으로 mission_examples 문서의 패턴을 참고해 적절한 navigate / observe_scene 조합을 만들 것."
        )

    hints.append(
        '- PLAN의 기본 순서는 [navigate → 핵심 action → (필요 시 추가 action) → '
        'navigate target "basecamp" → summarize_mission]을 따르게 할 것.'
    )

    return "\n".join(hints)


# ---------- 프롬프트 ----------

PLAN_SYSTEM_PROMPT = """너는 이동 로봇을 위한 Task Planner이다.
사용자의 한국어 요청을 받아, 5가지 액션만을 조합한 PLAN JSON을 생성한다.

[공통 규칙]

0. HINTS 해석
- 사용자 프롬프트에는 [HINTS_START] ... [HINTS_END] 구간이 함께 제공된다.
- 해당 구간에 적힌 지시는 RAG 문서보다 우선하며 반드시 지켜야 한다.

1. 출력 형식
- 출력은 반드시 하나의 유효한 JSON 객체만 포함해야 한다.
- JSON 바깥에 어떤 설명, 코드 블록, 자연어 문장도 쓰지 마라.
- JSON 안에도 주석을 쓰지 마라.

2. JSON 스키마
- JSON 구조는 아래 형식만 사용한다.
  {
    "plan": [
      {
        "action": "<string>",
        "params": { ... }
      },
      ...
    ]
  }
- action 이름은 ["navigate", "deliver_object", "observe_scene", "wait", "summarize_mission"]만 사용할 수 있다.

4. 최소 요구 사항
- "plan" 배열은 비어 있으면 안 된다.
- "summarize_mission"만 단독으로 사용하는 PLAN은 허용되지 않는다.

5. 보고 / 다시 알려줘 관련 규칙
- 현장 작업을 수행했다면 navigate basecamp 후 summarize_mission으로 보고한다.

6. 이동 후 귀환/요약 공통 규칙
- 이동/관찰/전달 등 현장 작업 요청 시 기본 순서는 [navigate → 핵심 action → basecamp navigate → summarize_mission]이다.

7~8. 추가 도메인 규칙
- 교수님/복도/라운지/화장실 키워드에 따른 위치 규칙을 지킨다.
- 사람 수/조명/소음 확인 요청에는 query에 해당 내용을 명시한다.

9. 규칙 위반 시에도 반드시 JSON만 출력한다.
"""

PLAN_USER_PROMPT = """다음은 RAG로 검색된 컨텍스트이다:

[CONTEXT_START]
{context}
[CONTEXT_END]

사용자의 요청은 다음과 같다:
"{question}"

추가 지시(HINTS):
[HINTS_START]
{hints}
[HINTS_END]

위 컨텍스트와 요청을 바탕으로,
위에서 정의한 JSON 형식에 맞는 PLAN만 출력해라.
JSON 이외의 어떤 텍스트도 출력하지 마라."""


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
                PLAN_USER_PROMPT,
                template_format="jinja2",
            ),
        ]
    )
    return prompt | llm.with_structured_output(Plan)


def generate_plan(
    question: str,
    llm: BaseChatModel,
    waypoint_docs: List[Document] | None = None,
) -> Tuple[dict, List[Document], PlanValidationResult]:
    """PLANNER LLM을 호출해 PLAN JSON과 검증 결과를 반환한다."""

    docs = waypoint_docs or []
    context_text = _format_context(docs) if docs else "No external context."
    hints = _build_hints(question)

    chain = _build_plan_chain(llm)
    plan_obj = chain.invoke(
        {"context": context_text, "question": question, "hints": hints}
    )
    plan_dict = plan_obj.model_dump() if isinstance(plan_obj, Plan) else plan_obj
    validation = validate_plan(plan_dict)
    return plan_dict, docs, validation
