"""Plan 단계별 tool 실행 서비스."""

from __future__ import annotations

from typing import Dict, Any, Optional

from app.services.navigation_service import NavigationBackend
from app.services.robot_grpc_client import RobotGrpcClient


class ExecutorService:
    """LangGraph plan executor가 호출할 action 핸들러."""

    def __init__(
        self,
        navigation_backend: Optional[NavigationBackend] = None,
        robot_client: RobotGrpcClient | None = None,
    ) -> None:
        self.navigation_backend = navigation_backend
        self.robot_client = robot_client

    def navigate(self, target: str) -> Dict[str, Any]:
        if self.robot_client:
            try:
                success, message = self.robot_client.navigate(target)
                return {
                    "status": "success" if success else "error",
                    "message": message or "",
                    "target": target,
                }
            except Exception as exc:  # pragma: no cover - network failure path
                return {
                    "status": "error",
                    "message": f"gRPC navigate 실패: {exc}",
                    "target": target,
                }
        if self.navigation_backend:
            return self.navigation_backend.navigate(target)
        return {
            "status": "success",
            "message": f"Navigated to {target}",
            "target": target,
        }

    def observe_scene(self, target: str, query: str | None = None) -> Dict[str, Any]:
        prompt = query or f"{target}의 상황을 설명해줘."
        if self.robot_client:
            try:
                success, description = self.robot_client.describe_scene(prompt)
                base_message = description or "장면 설명이 비었습니다."
                return {
                    "status": "success" if success else "error",
                    "message": base_message,
                    "target": target,
                }
            except Exception as exc:  # pragma: no cover - network failure path
                return {
                    "status": "error",
                    "message": f"gRPC observe 실패: {exc}",
                    "target": target,
                }
        return {
            "status": "success",
            "message": f"Observed {target}: {query or '환경 상태를 관찰'}",
            "target": target,
        }

    def deliver_object(self, receiver: str | None = None, item: str | None = None) -> Dict[str, Any]:
        if self.robot_client and receiver:
            try:
                success, message = self.robot_client.deliver(receiver, item)
                return {
                    "status": "success" if success else "error",
                    "message": message,
                    "target": receiver,
                }
            except Exception as exc:  # pragma: no cover - network failure path
                return {
                    "status": "error",
                    "message": f"gRPC deliver 실패: {exc}",
                    "target": receiver,
                }
        return {
            "status": "success",
            "message": f"Delivered {item or '물품'} to {receiver or 'recipient'}",
        }

    def summarize_mission(self, summary: str | None = None) -> Dict[str, Any]:
        return {
            "status": "reported",
            "message": summary or "임무 요약이 준비되었습니다.",
        }
