"""Planner node helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from langchain_core.documents import Document

from app.services.planner_service import generate_plan
from rln_rag.plan_validator import PlanValidationResult


@dataclass
class PlanNodeResult:
    plan: Dict[str, Any]
    context_docs: List[Document]
    validation: PlanValidationResult


def run_plan_node(question: str, llm, waypoint_docs: List[Document] | None = None) -> PlanNodeResult:
    """Planner node가 호출할 진입점."""

    plan_dict, docs, validation = generate_plan(question, llm, waypoint_docs)
    return PlanNodeResult(plan=plan_dict, context_docs=docs, validation=validation)
