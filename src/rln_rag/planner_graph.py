from __future__ import annotations

from typing import List, Tuple
import textwrap

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.retrievers import BaseRetriever

from .planner_models import Plan
from .planner_prompts import PLAN_SYSTEM_PROMPT, PLAN_USER_PROMPT

# 키워드들은 그대로 둔다. _build_hints 에서만 사용한다.
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
    """RAG로 가져온 문서를 LLM이 보기 좋게 요약된 문자열로 변환."""
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
        snippet = textwrap.shorten(snippet, width=500, placeholder="...")
        entries.append(f"[{idx}] source: {source}\nsummary: {snippet}")
    return "\n\n".join(entries)


def _build_hints(question: str) -> str:
    """
    현재 사용자 질문을 기반으로, PLAN LLM에 넘길 추가 지시(HINTS)를 생성한다.
    - 이 함수는 '완성 PLAN'을 만들지 않고, 오직 힌트 텍스트만 만든다.
    - 실제 PLAN 조립은 항상 LLM이 담당한다.
    """
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

    # 교수님 관련 허용/금지
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
        hints.append(
            "- 교수님 미션에서는 복도/라운지/화장실 등 다른 위치를 관찰하려 하지 말고, "
            "오직 교수님과의 대화에 집중할 것."
        )
    else:
        hints.append(
            "- 교수님 관련 action(교수님 이동/대화)은 사용하지 말 것."
        )

    # 위치 관련 힌트
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

    # 관찰 내용 관련 힌트
    if noisy:
        hints.append(
            "- 소음/시끄러움 확인: observe_scene의 query에 사람 수나 소란 여부를 반드시 포함할 것."
        )
    if lighting:
        hints.append(
            "- 조명/불 확인: observe_scene의 query에 조명 상태(켜짐/꺼짐)를 명시할 것."
        )

    # 보고 요청 힌트
    if needs_report:
        hints.append(
            '- 사용자에게 다시 알려주라는 요청이 있으므로, 핵심 작업 후 PLAN 마지막에 '
            'navigate target "basecamp"로 복귀하고 summarize_mission으로 결과를 보고할 것.'
        )

    # 어떤 키워드에도 안 걸리는 일반적인 경우
    if not corridor and not lounge and not restroom and not professor:
        hints.append(
            "- 기본적으로 mission_examples 문서의 패턴을 참고해 "
            "적절한 navigate / observe_scene / talk_to_person 조합을 만들 것."
        )

    hints.append(
        '- PLAN의 기본 순서는 [navigate → 핵심 action → (필요 시 추가 action) → '
        'navigate target "basecamp" → summarize_mission]을 따르게 할 것. '
        '보고 목적이라면 talk_to_person을 추가하지 말고 summarize_mission 결과로 대응할 것.'
    )

    return "\n".join(hints)


def build_plan_chain(llm: BaseChatModel):
    """SYSTEM / HUMAN 프롬프트를 조합해 planner용 체인을 생성."""
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
    retriever: BaseRetriever,
    llm: BaseChatModel,
    use_rag: bool = False,
) -> Tuple[dict, List[Document]]:
    """
    RAG + PLAN LLM을 사용해 항상 PLAN JSON을 생성한다.
    - 휴리스틱은 _build_hints()에서 '힌트 텍스트' 형태로만 사용된다.
    """
    # 1) RAG로 문서 검색 (옵션)
    docs: List[Document] = retriever.invoke(question) if use_rag else []

    # 2) 컨텍스트/힌트 구성
    context_text = _format_context(docs) if docs else "No external context."
    hints = _build_hints(question)

    # 3) PLAN LLM 호출
    chain = build_plan_chain(llm)
    plan_obj = chain.invoke(
        {"context": context_text, "question": question, "hints": hints}
    )
    plan_dict = plan_obj.model_dump() if isinstance(plan_obj, Plan) else plan_obj
    return plan_dict, docs
