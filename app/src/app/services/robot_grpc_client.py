"""Thin gRPC client used by the executor to reach the robot server."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Tuple

import grpc

# The generated pb2 modules live outside of the installed package tree.
_PB2_DIR = Path(__file__).resolve().parents[3] / "grpc"
if str(_PB2_DIR) not in sys.path:
    sys.path.append(str(_PB2_DIR))

import bero_pb2
import bero_pb2_grpc


class RobotGrpcClient:
    """Lazy gRPC stub wrapper for the remote robot controller."""

    def __init__(self, target: str) -> None:
        self._target = target
        self._channel: grpc.Channel | None = None
        self._stub: bero_pb2_grpc.RobotControllerStub | None = None

    def _get_stub(self) -> bero_pb2_grpc.RobotControllerStub:
        if self._stub is None:
            self._channel = grpc.insecure_channel(self._target)
            self._stub = bero_pb2_grpc.RobotControllerStub(self._channel)
        return self._stub

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def navigate(self, waypoint: str) -> Tuple[bool, str]:
        """Invoke Navigate RPC."""
        request = bero_pb2.NavigateRequest(waypoint=waypoint)
        stub = self._get_stub()
        response = stub.Navigate(request)
        return response.success, response.message

    def describe_scene(self, prompt: str) -> Tuple[bool, str]:
        """Invoke DescribeScene RPC."""
        request = bero_pb2.SceneRequest(prompt=prompt)
        stub = self._get_stub()
        response = stub.DescribeScene(request)
        return response.success, response.description

    def deliver(self, receiver: str, item: str | None = None) -> Tuple[bool, str]:
        """Temporary deliver helper that reuses Navigate RPC for waypoint hopping."""
        message_suffix = f"{item or '물품'} 전달 요청"
        response = self._get_stub().Navigate(
            bero_pb2.NavigateRequest(waypoint=receiver)
        )
        combined_message = f"{message_suffix}: {response.message}"
        return response.success, combined_message
