"""Plan 단계별 tool 실행 서비스."""

from __future__ import annotations

from typing import Dict, Any, Optional

from app.services.navigation_service import NavigationBackend


class ExecutorService:
    """LangGraph plan executor가 호출할 action 핸들러."""

    def __init__(self, navigation_backend: Optional[NavigationBackend] = None) -> None:
        self.navigation_backend = navigation_backend

    def navigate(self, target: str) -> Dict[str, Any]:
        if self.navigation_backend:
            return self.navigation_backend.navigate(target)
        return {
            "status": "success",
            "message": f"Navigated to {target}",
            "target": target,
        }

    def observe_scene(self, target: str, query: str | None = None) -> Dict[str, Any]:
        return {
            "status": "success",
            "message": f"Observed {target}: {query or '환경 상태를 관찰'}",
            "target": target,
        }

    def deliver_object(self, receiver: str | None = None, item: str | None = None) -> Dict[str, Any]:
        return {
            "status": "success",
            "message": f"Delivered {item or '물품'} to {receiver or 'recipient'}",
        }

    def summarize_mission(self, summary: str | None = None) -> Dict[str, Any]:
        return {
            "status": "reported",
            "message": summary or "임무 요약이 준비되었습니다.",
        }

    def report(self, content: str) -> Dict[str, Any]:
        return {"status": "reported", "message": content}

    def wait(self, seconds: int = 1) -> Dict[str, Any]:
        return {"status": "waiting", "seconds": seconds}
