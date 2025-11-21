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
        result = self.rule_service.validate(plan, question)
        if llm is None:
            return result

        llm_result = self.llm_service.validate(
            plan,
            question=question,
            llm=llm,
            extra_context=extra_context,
        )
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
