"""LLM 기반 Intent 분류 서비스."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel

from app.services.validator_rule_service import load_actions, load_locations

from pydantic import BaseModel, ValidationError
from pydantic import ConfigDict  # pydantic v2 기준


try:
    from langchain_core.prompts import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        SystemMessagePromptTemplate,
    )
except ModuleNotFoundError:  # pragma: no cover - compatibility for langchain<0.2
    from langchain.prompts import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        SystemMessagePromptTemplate,
    )


INTENT_SYSTEM_PROMPT = r"""
너는 실내 자율주행 이동로봇 보조 시스템을 위한 INTENT CLASSIFIER이다.
너의 유일한 역할은 아래 세 필드만 채운 JSON 하나를 반환하는 것이다.

{
  "intent": "PLAN" | "CONVERSATION",
  "conversation_mode": "NORMAL" | "UNSUPPORTED",
  "reason": "<UNSUPPORTED 이유 또는 빈 문자열>"
}

[필드 정의]
- intent:
  - "PLAN"         : 실제로 로봇에게 지금 어떤 행동을 수행하라고 시키는 명령/요청
  - "CONVERSATION" : 설명 요청, 질문, 잡담, 가정/예시 등 로봇이 당장 행동하지 않는 경우
- conversation_mode: intent가 "CONVERSATION"인 경우에만 의미가 있다.
  - "NORMAL"      : 일반 대화/질문/잡담 등 단순 상호작용
  - "UNSUPPORTED" : 실제로 로봇에게 한 명령이지만,
                    능력 밖/지원되지 않는 행동·장소이거나
                    불법·비윤리적인 요청이어서 수행 불가인 경우
- reason:
  - conversation_mode가 "UNSUPPORTED"일 때, 왜 수행 불가능한지 한국어 한 줄로 설명한다.
  - intent가 "PLAN"이거나 conversation_mode가 "NORMAL"이면 빈 문자열("") 또는 null 로 둔다.

너는 Task Planner가 아니다.
ACTION 이름이나 target을 만들지 말고, 위 세 필드만 채워라.

============================================================
[로봇 도메인]
============================================================
- 로봇은 건물 내부를 이동하고 관찰/보고/전달을 수행할 수 있다.
- 사람이 적재/하차를 도와준다는 가정하에 물건 배달이 가능하다.
- 요리, 가전 직접 제어, 해킹, 시간 여행, 공상/판타지 등 비현실적인 일은 불가능하다.
- 도덕적/법적으로 문제 있는 행동도 수행할 수 없다.

============================================================
[안전 / 불법 행위에 대한 규칙]
============================================================
- 다음과 같은 요청은, 로봇이 절대 수행해서는 안 되는 불법/비윤리적 행동이다.
  - 시험문제를 훔쳐오라, 정답/답안을 몰래 가져오라
  - 남의 물건을 훔쳐오라, 파괴하라
  - 사람을 몰래 따라가 촬영/감시하라, 스토킹하라
  - 폭력, 범죄, 자해 등을 돕는 행위
- 이런 요청이 포함되면, 설령 Action/Location 조합으로 PLAN을 만들 수 있어 보이더라도:
  - intent = "CONVERSATION"
  - conversation_mode = "UNSUPPORTED"
  - reason에는 "불법적이거나 비윤리적인 요청이므로 수행할 수 없습니다." 와 같이 한 줄 설명을 쓴다.
- 문장 형태가 "OO에 가서 ~~하고 와"처럼 PLAN 패턴이어도,
  내용에 불법·비윤리적 요소가 포함되어 있으면 위 규칙을 항상 우선한다.

============================================================
[허용 ACTION 목록]
============================================================
로봇은 반드시 아래 ACTIONS로 표현 가능한 일만 수행한다.
여기에 없는 행동만 요구하면 PLAN으로 분류하지 마라.

{% for action in actions %}
{{ loop.index }}. "{{ action.name }}"
   - 설명: {{ action.description }}
   - 필수 파라미터: [{% for p in action.required_params %}"{{ p }}"{% if not loop.last %}, {% endif %}{% endfor %}]
   - 자연어 파라미터 필드: [{% for p in action.nl_params %}"{{ p }}"{% if not loop.last %}, {% endif %}{% endfor %}]
   - 대표 트리거 문장:
     {% if action.trigger_phrases %}
     {% for phrase in action.trigger_phrases %}"{{ phrase }}"{% if not loop.last %}, {% endif %}{% endfor %}
     {% else %}
     (없음)
     {% endif %}

{% endfor %}

============================================================
[알고 있는 LOCATION 목록]
============================================================
다음 LOCATION ID는 로봇이 아는 장소 목록이다.
여기에 없는 장소를 대상으로 한 요청이면 PLAN으로 분류하지 마라.

{% for loc in locations %}
- {{ loc.id }}: {{ loc.description }}
{% endfor %}

============================================================
[핵심 판단 규칙]
============================================================

1. intent = "PLAN" 으로 분류해야 하는 경우
   - 사용자가 로봇에게 **실제 행동**을 수행하도록 명령/요청하는 문장일 때.
   - 요청이 현실적으로 가능하고, [허용 ACTION 목록]과 [LOCATION 목록]으로 표현 가능할 때.
   - 한국어에서 명령형이 생략된 형태도 PLAN 으로 본다.
     - 예: "교수님 방에 가서 강아지가 있는지" / "복도에 사람 있는지" 처럼,
       로봇이 이동+관찰을 해야 의미가 통하는 문장은 모두 PLAN이다.
   - 하나의 발화 안에 PLAN 성격의 명령과 일반 대화가 섞여 있으면,
     → PLAN을 우선한다. (intent = "PLAN")
     (단, PLAN으로 표현 가능한 명령 부분이 전혀 없으면 CONVERSATION)

2. intent = "CONVERSATION" 으로 분류해야 하는 경우
   - 단순 정보 질문, 설명 요청, 코드/설계/프롬프트 질문, 잡담 등.
   - 예: "강아지가 교수님 방에 있을까?" (그냥 궁금한 말, 로봇에게 '가서' 라고 하지 않음)
   - 허용되지 않은 ACTION만 등장하거나, LOCATION 목록에 없는 장소만 등장하는 명령.
   - 가정/예시/아이디어 수준의 설명(예: "로봇에게 교수님 방에 가서 강아지 확인하라고 시키면 PLAN이 되겠지?").
   - 의도가 애매해서 PLAN/CONVERSATION 중 하나로 확실히 정하기 어렵다면,
     기본값으로 intent="CONVERSATION", conversation_mode="NORMAL" 로 둔다.

3. conversation_mode 세분화 (intent == "CONVERSATION" 인 경우만 의미 있음)
   - "NORMAL":
     - 일반 질의/설명/잡담 등, 로봇에게 실제 행동을 시키는 것이 아니라면 모두 NORMAL.
   - "UNSUPPORTED":
     - 사용자가 로봇에게 실제 행동을 시켰지만,
       - 허용되지 않은 ACTION만으로 이루어진 요청이거나
       - LOCATION 목록에 없는 장소로 이동/전달/확인 요청이거나
       - 불법·비윤리적인 행동을 시키는 요청인 경우.
     - reason에는 "요청한 위치는 로봇이 모르는 장소입니다." / 
       "불법적이거나 비윤리적인 요청이므로 수행할 수 없습니다." 등의 한 줄 설명을 쓴다.

[예시]

- 예시 1
  사용자: "교수님 방에 가서 강아지가 있는지"
  → intent = "PLAN"

- 예시 2
  사용자: "교수님 방에 강아지가 있을까?"
  → intent = "CONVERSATION", conversation_mode = "NORMAL"

- 예시 3
  사용자: "집에 가서 TV 꺼져 있는지 확인해줘"
  → intent = "CONVERSATION", conversation_mode = "UNSUPPORTED"
     reason: "집은 로봇이 인식하는 실내 위치 목록에 없는 장소입니다."

- 예시 4
  사용자: "교수님 방에 가서 시험문제 좀 훔치고 와"
  → intent = "CONVERSATION", conversation_mode = "UNSUPPORTED"
     reason: "시험문제를 훔쳐오는 것은 불법적이고 비윤리적인 요청이므로 수행할 수 없습니다."

============================================================
[출력 형식]
============================================================

반드시 아래 JSON 형식 **한 개만** 출력하라.
설명 문장, 코드 블록, 주석, 자연어 문장을 JSON 앞뒤에 붙이지 마라.

{
  "intent": "PLAN" | "CONVERSATION",
  "conversation_mode": "NORMAL" | "UNSUPPORTED",
  "reason": "<UNSUPPORTED 이유 또는 빈 문자열>"
}
"""


INTENT_HUMAN_PROMPT = """
아래는 사용자의 발화이다.
위 지침에 따라 intent와 conversation_mode, reason을 판단하여
반드시 JSON 한 개만 출력하라.

[사용자 메시지]
\"\"\"{{ question }}\"\"\"
"""


class IntentLLMResponse(BaseModel):
    """LLM이 따라야 하는 의도 분류 결과 스키마."""

    intent: Literal["PLAN", "CONVERSATION"]
    conversation_mode: Literal["NORMAL", "UNSUPPORTED"] = "NORMAL"
    reason: str | None = None

    model_config = ConfigDict(extra="ignore")


@dataclass(slots=True)
class IntentPrediction:
    intent: Literal["conversation", "plan"]
    conversation_mode: Literal["normal", "unsupported"] = "normal"
    reason: str | None = None


class IntentClassifierService:
    """LLM 호출을 통한 intent 분류기."""

    def __init__(self) -> None:
        self.actions = load_actions()
        self.locations = load_locations()
        self._prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    INTENT_SYSTEM_PROMPT,
                    template_format="jinja2",
                ),
                HumanMessagePromptTemplate.from_template(
                    INTENT_HUMAN_PROMPT,
                    template_format="jinja2",
                ),
            ]
        )

    def classify(
        self,
        question: str,
        *,
        llm: BaseChatModel | None,
    ) -> IntentPrediction:
        """
        LLM 기반 intent 분류.

        - llm이 없거나, 호출 자체가 실패하면 예외를 발생시킨다.
        - LLM이 이상한 형식으로 답하더라도, 가능한 한 PLAN / CONVERSATION과 세부 모드를 추출한다.
        - 최종적으로 판단 불가하면 보수적으로 CONVERSATION/NORMAL 값을 반환한다.
        """

        if llm is None:
            raise RuntimeError("IntentClassifierService: llm 인스턴스가 필요합니다.")

        try:
            formatted = self._prompt.format_messages(
                question=question or "",
                actions=self.actions,
                locations=self.locations,
            )
            response = llm.invoke(formatted)
        except Exception as e:
            # 여기까지 오면 LLM이 아예 죽은 상황: 상위 레이어에서 처리
            raise RuntimeError("IntentClassifierService: LLM 호출 실패") from e

        content = getattr(response, "content", response)
        return self._parse_intent_payload(content)

    @staticmethod
    def _parse_intent_payload(payload) -> IntentPrediction:
        """LLM 응답에서 의도/모드 정보를 추출한다."""

        text = payload if isinstance(payload, str) else str(payload or "")
        text = text.strip()
        if not text:
            return IntentPrediction(intent="conversation")

        # 1) JSON 파싱 + 스키마 검증
        try:
            result = IntentLLMResponse.model_validate_json(text)
            return IntentClassifierService._to_prediction(result)
        except ValidationError:
            pass

        # 2) 일반 JSON으로 파싱해서 필드를 추출
        try:
            data = json.loads(text)
            candidate = data.get("intent")
            normalized = IntentClassifierService._normalize_label(candidate)
            if normalized:
                mode = IntentClassifierService._normalize_mode(
                    data.get("conversation_mode")
                )
                reason = (data.get("reason") or "").strip() or None
                if mode == "unsupported":
                    reason = reason or "요청이 지원되지 않습니다."
                else:
                    reason = None
                return IntentPrediction(
                    intent=normalized,
                    conversation_mode=mode,
                    reason=reason,
                )
        except Exception:
            pass

        # 3) 그냥 문자열 전체에서 PLAN / CONVERSATION 유추
        normalized = IntentClassifierService._normalize_label(text)
        if normalized:
            return IntentPrediction(intent=normalized)

        # 4) 여기까지 오면 의도를 확실히 못 뽑은 상태 → 보수적으로 conversation
        return IntentPrediction(intent="conversation")

    @staticmethod
    def _normalize_label(value) -> Literal["conversation", "plan"] | None:
        if not value:
            return None
        lowered = str(value).strip().lower()
        if "plan" in lowered:
            return "plan"
        if "conversation" in lowered or "chat" in lowered:
            return "conversation"
        return None

    @staticmethod
    def _normalize_mode(value) -> Literal["normal", "unsupported"]:
        if not value:
            return "normal"
        lowered = str(value).strip().lower()
        if "unsupported" in lowered:
            return "unsupported"
        return "normal"

    @staticmethod
    def _to_prediction(data: IntentLLMResponse) -> IntentPrediction:
        intent = "plan" if data.intent == "PLAN" else "conversation"
        mode = "unsupported" if data.conversation_mode == "UNSUPPORTED" else "normal"
        reason = (data.reason or "").strip()
        if mode == "unsupported":
            reason = reason or "요청이 지원되지 않습니다."
        else:
            reason = None
        return IntentPrediction(
            intent=intent,
            conversation_mode=mode,
            reason=reason,
        )
