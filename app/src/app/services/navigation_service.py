"""Navigation/NAV2 통신 옵션을 정의."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Dict, Any


class NavigationBackend(Protocol):
    def navigate(self, target_name: str) -> Dict[str, Any]: ...


@dataclass
class ROS2Backend:
    action_server: str = "/navigate_to_pose"

    def navigate(self, target_name: str) -> Dict[str, Any]:
        return {
            "status": "pending",
            "message": f"ROS2 action {self.action_server} -> {target_name}",
        }


@dataclass
class GRPCBackend:
    endpoint: str

    def navigate(self, target_name: str) -> Dict[str, Any]:
        return {
            "status": "pending",
            "message": f"gRPC call {self.endpoint} -> {target_name}",
        }


@dataclass
class HTTPBackend:
    base_url: str

    def navigate(self, target_name: str) -> Dict[str, Any]:
        return {
            "status": "pending",
            "message": f"HTTP POST {self.base_url} {target_name}",
        }
