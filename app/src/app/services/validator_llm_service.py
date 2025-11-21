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

LLM_VALIDATOR_SYSTEM = r"""
너는 이동 로봇을 위한 PLAN Validator이다.

**[매우 중요] 기계적인 구조(JSON 문법, 순서, 마지막 단계 확인 등)는 이미 시스템이 검증을 마쳤다.**
너는 오직 **"사용자의 의도와 PLAN의 내용이 일치하는지"** 의미론적(Semantic) 관점에서만 심사하라.

============================================================
[1] 허용 데이터
============================================================
(Actions) {% for a in actions %}{{ a.name }}, {% endfor %}
(Locations) {% for l in locations %}{{ l.id }}, {% endfor %}

============================================================
[2] 검증 규칙 (오직 의미만 심사)
============================================================

**Rule 1. 파라미터 유연성 (Parameter Flexibility)**
- Planner가 JSON을 만들 때, 문맥에 맞게 **한국어 어미를 변경**("확인하고" -> "확인해")했을 수 있다.
- 텍스트가 원문과 100% 똑같지 않아도, **의미가 왜곡되지 않았다면 "VALID"로 판정**하라.
- 단, 아예 없는 말을 지어내거나("커피" -> "콜라"), 엉뚱한 대상을 가리키면 INVALID이다.

**Rule 2. 장소-행동 일치 (Context Match)**
- 사용자가 "화장실"을 언급했으면 `restroom` 관련 ID로 가야 한다.
- 사용자의 의도와 다른 엉뚱한 행동을 하는지 감시하라.

**Rule 3. 요청 가능 여부 (Availability)**
- 사용자가 언급한 장소가 허용 `locations` 목록에 없다면 해당 PLAN은 수행 불가이다.
- 사용자가 요구한 행동이 허용 `actions` 목록에 없다면 PLAN 역시 수행 불가이다.
- 위 조건을 발견하면 `verdict`는 반드시 `"UNSUPPORTED"`여야 하며,
  `reasons` 에는 `[UNSUPPORTED_LOCATION] ...` 또는 `[UNSUPPORTED_ACTION] ...` 형태로 이유를 남겨라.

**[금지 사항]**
- ❌ `summarize_mission`이 마지막에 있는지 검사하지 마라. (이미 통과됨)
- ❌ `navigate` 순서를 검사하지 마라. (이미 통과됨)

============================================================
[3] 출력 형식
============================================================
아래 셋 중 하나만 허용한다.
- {"verdict": "VALID", "reasons": []}
- {"verdict": "INVALID", "reasons": ["..."]}
- {"verdict": "UNSUPPORTED", "reasons": ["[UNSUPPORTED_LOCATION] ..."]} (또는 ACTION 태그)
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
            formatted = self._prompt.format_messages(
                question=question,
                extra_context=extra_context,
                plan_json=json.dumps(plan, ensure_ascii=False),
                actions=self.actions,
                locations=self.locations,
            )
            # LLM 호출
            response = llm.invoke(formatted)
            content = getattr(response, "content", response)
            _append_llm_log(
                question=question,
                plan=plan,
                extra_context=extra_context,
                raw_output=content if isinstance(content, str) else str(content),
            )
            payload = self._extract_json(content)

        except Exception as exc:
            result.add_warning(f"LLM Validator 호출 실패: {exc}")
            return result

        verdict = (payload.get("verdict") or "").upper()
        reasons = payload.get("reasons") or []

        unsupported_kind: str | None = None
        for r in reasons:
            if "[UNSUPPORTED_LOCATION]" in r:
                unsupported_kind = "location"
                break
            if "[UNSUPPORTED_ACTION]" in r:
                unsupported_kind = "action"
                break

        if verdict == "VALID":
            return result

        if not reasons:
            result.add_error("LLM validator가 INVALID를 반환했지만 이유가 없습니다.")
        else:
            for r in reasons:
                result.add_error(f"[LLM] {r}")

        if verdict == "UNSUPPORTED":
            result.metadata["unsupported_request"] = unsupported_kind or "unknown"

        return result

    # ---------------------

    @staticmethod
    def _extract_json(text: str) -> dict:
        decoder = json.JSONDecoder()

        def _decode(candidate: str) -> dict:
            stripped = candidate.lstrip()
            payload, _ = decoder.raw_decode(stripped)
            return payload

        candidates = [text]
        cleaned = text.strip()
        if cleaned:
            candidates.append(cleaned)

        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 2:
                candidates.append("\n".join(lines[1:-1]))

        first_brace = text.find("{")
        if first_brace != -1:
            candidates.append(text[first_brace:])
        first_brace_clean = cleaned.find("{")
        if first_brace_clean != -1:
            candidates.append(cleaned[first_brace_clean:])

        for candidate in candidates:
            try:
                return _decode(candidate)
            except json.JSONDecodeError:
                continue

        raise ValueError(f"LLM 출력에서 JSON을 찾을 수 없습니다. Output: {text[:50]}...")
