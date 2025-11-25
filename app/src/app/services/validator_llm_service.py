"""LLM-based PLAN validator (Logic Reinforced)."""

from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from app.services.validator_rule_service import PlanValidationResult, load_actions, load_locations

try:
    from langchain_core.prompts import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        SystemMessagePromptTemplate,
    )
except ModuleNotFoundError:
    from langchain.prompts import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        SystemMessagePromptTemplate,
    )


# -------------------- SYSTEM PROMPT (Semantic Focus) --------------------

# -------------------- SYSTEM PROMPT --------------------

LLM_VALIDATOR_SYSTEM = r"""
너는 실내 자율주행 이동 로봇을 위한 PLAN Validator이다.
이 Validator는 **이미 Rule-Based Validator를 통과한 PLAN**만 입력으로 받는다.

너의 역할은 오직 다음 두 가지다.
1) PLAN이 사용자의 의도와 의미론적으로 일치하는지 평가한다.
2) 애초에 수행하면 안 되는(불법/비윤리/능력 밖) 요청인지 판정한다.

**[핵심 지침]**
1. 절대 입력된 PLAN JSON을 다시 출력하지 마라. (No Echo)
2. 설명이나 서론 없이 **결과 JSON 하나만** 출력하라.
3. Markdown 코드블록, 헤더("#", "##") 등을 쓰지 마라.
4. JSON 구조/스키마/마지막 단계(basecamp + summarize_mission)는
   이미 시스템과 Rule-Based Validator가 검증했다. 다시 지적하지 마라.
5. 오직 의미론(Semantic) 관점에서만 심사하라.

============================================================
[1] 허용 데이터
============================================================
(Actions) {% for a in actions %}{{ a.name }}{% if not loop.last %}, {% endif %}{% endfor %}
(Locations) {% for l in locations %}{{ l.id }}{% if not loop.last %}, {% endif %}{% endfor %}

============================================================
[2] 판정 값 정의
============================================================

- verdict = "VALID" | "INVALID" | "UNSUPPORTED"

1) VALID
   - PLAN이 사용자의 요청을 충분히 커버하고,
     장소/행동 선택과 순서가 자연스럽게 일치할 때.
   - basecamp로 돌아가는 이동과 summarize_mission은
     시스템 규칙이므로, 사용자가 말하지 않아도 있어도 된다.

2) INVALID
   - 구조적으로는 문제 없지만, 의미론적으로 다음과 같은 문제가 있을 때.
     - 사용자가 명시한 장소/행동 중 일부가 PLAN에서 빠져 있음.
     - 사용자가 말하지 않은 핵심 행동을 PLAN이 자의적으로 추가함
       (단, basecamp + summarize_mission은 예외).
     - 사용자가 원하는 장소와 다른 Location id로 매핑됨.
     - 요청 순서(A 후에 B)를 PLAN이 바꿔서 수행함.
     - 행동 종류가 문맥에 맞지 않음 (예: "확인해줘"인데 전혀 다른 종류의 행동).

3) UNSUPPORTED  (최우선 판정)
   - 사용자의 요청 자체가 로봇 능력/도메인 밖이거나,
     불법·비윤리적이어서 **애초에 수행하면 안 되는 경우**.
   - 예시:
     - Location 목록에 전혀 없는 장소(집, 카페 등)로 가서 무언가 하라고 함.
     - 시험문제 훔쳐오기, 남의 물건 훔치기, 사람을 몰래 감시/촬영하기 등.
     - 현실적으로 로봇이 할 수 없는 행동(해킹, 건물 전기 끄기 등).
   - Planner가 어떻게 PLAN을 만들었든, 요청이 위 조건에 해당하면
     verdict는 항상 "UNSUPPORTED"여야 한다.

============================================================
[3] 의미론적 검증 기준 (Semantic Check List)
============================================================

다음 항목을 중심으로 PLAN을 평가하라.

1) 커버리지
   - 사용자의 발화에 등장한 주요 목표/행동이 PLAN에 모두 반영되어 있는가?
   - "A 하고, B도 하고 와"인데 A만 있거나 B만 있으면 INVALID.

2) 장소 매핑
   - 사용자가 말한 장소 표현이 Location List 중 어떤 id로 매핑되었는지 추론하라.
   - 의미상 다른 장소로 매핑되어 있다면 INVALID 또는 UNSUPPORTED.
     - 예: 사용자는 "화장실"을 말했는데, PLAN은 "corridor_center"로만 이동.

3) 행동 선택
   - 사용자의 표현이 적절한 action으로 매핑되었는지 본다.
     - "사람 있는지 확인해 줘" → observe_scene
     - "서류 갖다 줘" → deliver_object
   - 로봇 능력 밖의 행동(훔치기 등)은 UNSUPPORTED.

4) 순서
   - 사용자가 말한 순서와 PLAN의 순서가 크게 어긋나지 않는지 본다.
   - 단, basecamp로 복귀 후 summarize_mission은 항상 마지막이므로 순서 예외로 허용.

============================================================
[4] 출력 형식 (JSON Only)
============================================================

반드시 아래 형식의 JSON 하나만 출력하라.

{ "verdict": "VALID", "reasons": [] }

또는

{
  "verdict": "INVALID",
  "reasons": ["이유 1", "이유 2", ...]
}

또는

{
  "verdict": "UNSUPPORTED",
  "reasons": ["이유 1", "이유 2", ...]
}

- reasons에는 한국어 한 줄 설명을 1개 이상 넣어라.
- 예시의 문장을 그대로 베끼지 말고, 실제 사용자 요청과 PLAN을 근거로 작성하라.
- verdict가 "VALID"일 때는 reasons를 빈 배열([])로 둔다.
"""

LLM_VALIDATOR_HUMAN = r"""
사용자 요청:
{{question}}

{% if extra_context %}
추가 참고 정보:
{{extra_context}}
{% endif %}

검증 대상 PLAN JSON:
{{plan_json}}

위 PLAN이 사용자의 요청과 의미론적으로 일치하는지,
그리고 수행 불가능한(UNSUPPORTED) 요청은 아닌지 판단하여
[출력 형식]에 맞는 JSON만 반환하라.
"""

# -------------------- Service --------------------

_LLM_LOG_PATH = Path(__file__).resolve().parents[3] / "artifacts" / "plan_validator_llm.log"


def _append_llm_log(*, question: str, plan: Any, extra_context: str | None, raw_output: str):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "plan": plan,
        "extra_context": extra_context,
        "raw_output": raw_output,
    }
    try:
        _LLM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LLM_LOG_PATH.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:
        # 로그 기록 오류는 검증 흐름을 막지 않음
        pass

class LLMValidatorService:
    def __init__(self):
        actions = load_actions()
        locations = load_locations()

        self._prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    LLM_VALIDATOR_SYSTEM,
                    template_format="jinja2",
                ),
                HumanMessagePromptTemplate.from_template(
                    LLM_VALIDATOR_HUMAN,
                    template_format="jinja2",
                ),
            ]
        )

        self.actions = actions
        self.locations = locations

    # ---------------------

    def validate(
            self,
            plan: Any,
            *,
            question: str,
            llm: BaseChatModel,
            extra_context: str | None = None,
        ) -> PlanValidationResult:

            result = PlanValidationResult()

            try:
                # 1. 프롬프트 포맷팅 (실제 코드 복원)
                formatted = self._prompt.format_messages(
                    question=question,
                    extra_context=extra_context,
                    plan_json=json.dumps(plan, ensure_ascii=False),
                    actions=self.actions,
                    locations=self.locations,
                )
                
                # 2. LLM 호출
                response = llm.invoke(formatted)
                content = getattr(response, "content", response)
                
                # 3. 로그 기록 (문자열 변환 안전장치 추가)
                raw_output_str = content if isinstance(content, str) else str(content)
                _append_llm_log(
                    question=question,
                    plan=plan,
                    extra_context=extra_context,
                    raw_output=raw_output_str,
                )
                
                # 4. JSON 파싱
                payload = self._extract_json(raw_output_str)

            except Exception as exc:
                result.add_warning(f"LLM Validator 호출 실패: {exc}")
                return result

            # -------------------------------------------------------
            # 결과 판정 로직
            # -------------------------------------------------------
            verdict = (payload.get("verdict") or "").upper()
            reasons = payload.get("reasons") or []

            # 1. VALID (통과)
            if verdict == "VALID":
                result.llm_verdict = "VALID"
                result.llm_reason = None
                return result

            # 2. UNSUPPORTED (지원하지 않음 -> Conversation Agent로)
            if verdict == "UNSUPPORTED":
                result.llm_verdict = "UNSUPPORTED"
                
                if reasons:
                    result.llm_reason = reasons[0]
                else:
                    result.llm_reason = "요청하신 항목을 지원하지 않습니다."
                
                # errors에 추가하지 않음 (Retry 방지)
                return result

            # 3. INVALID (Planner 실패 -> Retry)
            if not reasons:
                result.add_error("LLM validator가 INVALID를 반환했지만 이유가 없습니다.")
            else:
                for r in reasons:
                    result.add_error(f"[LLM] {r}")

            return result

    # ---------------------

    @staticmethod
    def _extract_json(text: str) -> dict:
        decoder = json.JSONDecoder()

        # JSON 후보군을 찾기 위한 정규식 (코드블럭 또는 중괄호 덩어리)
        candidates = []
        
        # 1. 마크다운 코드블럭 안의 내용 우선 탐색
        code_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        candidates.extend(code_blocks)

        # 2. 전체 텍스트에서 중괄호로 시작하고 끝나는 부분 탐색
        # (단순하게 가장 바깥쪽 중괄호를 찾음)
        try:
            stripped = text.strip()
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start != -1 and end != -1:
                candidates.append(stripped[start : end + 1])
        except Exception:
            pass

        # 3. 후보군 순회하며 'verdict' 키가 있는 유효한 JSON 찾기
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
                if isinstance(payload, dict) and "verdict" in payload:
                    return payload
            except json.JSONDecodeError:
                continue
        
        # 4. 실패 시, 혹시 모르니 그냥 첫 번째 유효한 JSON이라도 리턴 (Fallback)
        for candidate in candidates:
             try:
                return json.loads(candidate)
             except json.JSONDecodeError:
                continue

        raise ValueError(f"LLM 출력에서 유효한 검증 결과 JSON을 찾을 수 없습니다. Output: {text[:100]}...")
