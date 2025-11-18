from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class Step(BaseModel):
    """Single executable step in a PLAN."""

    action: Literal[
        "navigate",
        "deliver_object",
        "observe_scene",
        "talk_to_person",
        "summarize_mission",
        "report",
        "rag_qa",
        "rag_retrieve",
        "wait",
    ]
    params: Dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    """Ordered list of steps to execute."""

    plan: List[Step] = Field(
        default_factory=list,
        description="Robot should execute these steps in order",
    )
