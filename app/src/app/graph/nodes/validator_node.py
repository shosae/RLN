"""Validator node helpers."""

from __future__ import annotations

from typing import Any

from app.services.validator_service import validate_plan, PlanValidationResult


def run_validator(plan_dict: Any) -> PlanValidationResult:
    """PLAN JSON 구조 검증."""

    return validate_plan(plan_dict)
