"""Composite validator that combines rule-based and LLM-based checks."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from app.services.validator_llm_service import LLMValidatorService
from app.services.validator_rule_service import PlanValidationResult, PlanValidatorService


class CompositeValidatorService:
    """Applies fast rule-based checks before invoking the LLM validator."""

    def __init__(
        self,
        rule_service: PlanValidatorService | None = None,
        llm_service: LLMValidatorService | None = None,
    ) -> None:
        self.rule_service = rule_service or PlanValidatorService()
        self.llm_service = llm_service or LLMValidatorService()

    def validate(
        self,
        plan: Any,
        *,
        question: str | None = None,
        llm: BaseChatModel | None = None,
        extra_context: str | None = None,
    ) -> PlanValidationResult:

        # 1. Rule-based 검사 먼저 수행
        result = self.rule_service.validate(plan, question)
        
        # LLM이 없으면 Rule-based 결과만 반환
        if llm is None:
            return result

        # 2. LLM 검사 수행
        llm_result = self.llm_service.validate(
            plan,
            question=question,
            llm=llm,
            extra_context=extra_context,
        )
                
        result.llm_verdict = llm_result.llm_verdict
        result.llm_reason = llm_result.llm_reason

        if result.llm_verdict == "UNSUPPORTED":
            result.errors.clear()  # 기존 에러 삭제 (Retry 방지)
            # 필요하다면 경고로 남겨둠
            return result

        # (3) 그 외(VALID / INVALID)의 경우: 에러 병합
        result.errors.extend(llm_result.errors)
        result.warnings.extend(llm_result.warnings)
        
        return result

_DEFAULT_VALIDATOR = CompositeValidatorService()


def validate_plan(
    plan: Any,
    *,
    question: str | None = None,
    llm: BaseChatModel | None = None,
    extra_context: str | None = None,
) -> PlanValidationResult:
    """
    Validate the PLAN JSON.

    Parameters
    ----------
    plan:
        PLAN dict to validate.
    question:
        Optional request text to give additional context to the LLM validator.
    llm:
        Optional chat model. When omitted, only the rule-based checks run.
    extra_context:
        Optional string (e.g., HINTS) that should be appended to the LLM validator prompt.
    """

    return _DEFAULT_VALIDATOR.validate(
        plan,
        question=question,
        llm=llm,
        extra_context=extra_context,
    )
