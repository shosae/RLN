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
너는 이동 로봇을 위한 PLAN Validator이다.

**[핵심 지침]**
1. **절대 입력된 PLAN JSON을 다시 출력하지 마라.** (No Echo)
2. 설명이나 서론/본론 없이 **오직 결과 JSON 하나만** 출력하라.
3. Markdown 헤더("## Validator") 등을 쓰지 마라.
4. 기계적인 구조(JSON 문법, 순서, 마지막 단계 등)는 이미 시스템이 검증했다.**
5. 너는 오직 **"사용자의 의도와 PLAN의 내용이 일치하는지"** 의미론적(Semantic) 관점에서만 심사하라.

============================================================
[1] 허용 데이터
============================================================
(Actions) {% for a in actions %}{{ a.name }}, {% endfor %}
(Locations) {% for l in locations %}{{ l.id }}, {% endfor %}

============================================================
[2] 검증 규칙 (의미론적 심사 기준)
============================================================

**Rule 1. UNSUPPORTED (최우선 순위)**
- 사용자가 요청한 **장소(명사)**가 위 [1] Locations 목록에 정확히 없다면 무조건 **UNSUPPORTED**이다.
- **[주의]** Planner가 엉뚱한 장소(`professor_office`)로 매핑했더라도, 원본 요청('집')이 목록에 없으면 **PLAN을 무시하고 UNSUPPORTED로 판정**해야 한다.
- 예: 사용자가 '집'을 가라고 했는데 Location 목록에 없다면 -> **UNSUPPORTED**
- 예: 사용자가 '춤춰'라고 했는데 Action 목록에 없다면 -> **UNSUPPORTED**
- 이유(reasons)에는 "사용자가 요청한 장소 'X'는 지원되지 않습니다."라고 적어라.

**Rule 2. INVALID**
- 장소는 목록에 있지만, 의도와 다르게 매핑된 경우.
- 행동이 문맥상 틀린 경우.
- 예: '화장실' -> '복도')는 **INVALID**이다.

**Rule 3. VALID**
- 문맥, 장소, 행동이 모두 사용자의 의도와 일치하는 경우.
- 문맥상 의미가 통하면, 텍스트가 100% 일치하지 않아도 "VALID"이다.
- 한국어 어미 변화("확인하고"->"확인해")는 허용.

============================================================
[3] 판정 가이드 (Priority)
============================================================
1. **UNSUPPORTED**: 사용자가 요청한 장소/행동이 시스템(목록)에 없을 때. (최우선 순위)
2. **INVALID**: 목록에는 있지만, 의도와 다르게 잘못 연결되었거나 엉뚱한 행동을 할 때.
3. **VALID**: 위 문제없이 의도가 정확히 반영되었을 때.

============================================================
[4] 출력 형식 (JSON Only)
============================================================
반드시 JSON 하나만 출력하라.

**[주의] 아래 예시의 '이유' 텍스트를 그대로 베끼지 마라. 실제 PLAN의 내용을 분석해서 작성하라.**

(Case 1: 성공)
{ "verdict": "VALID", "reasons": [] }

(Case 2: 지원하지 않음)
{
  "verdict": "UNSUPPORTED",
  "reasons": ["사용자가 요청한 장소 'A'는 목록에 없습니다."]
}

(Case 3: 틀림 - 구체적 서술 필수)
{
  "verdict": "INVALID",
  "reasons": ["사용자는 'B장소'를 원했는데, PLAN은 'C장소'로 이동합니다.", "이동(navigate) 없이 행동(action)을 수행하려고 합니다."]
}
"""
LLM_VALIDATOR_HUMAN = r"""
사용자 요청:
{{question}}

{% if extra_context %}
추가 참고 정보:
{{extra_context}}
{% endif %}

PLAN JSON (검증 대상):
{{plan_json}}

위 PLAN을 [핵심 검증 규칙]에 따라 엄격히 심사하여 JSON 결과를 출력하라.
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
