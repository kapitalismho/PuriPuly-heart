from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class SelfCaptureVadEventRuntime(Protocol):
    async def handle_vad_event(self, event: object) -> None: ...

    async def reject_owned_segment(
        self,
        event: object,
        *,
        reason: str,
        outcome: str,
    ) -> None: ...

    async def fail_owned_segment(self, event: object, *, reason: str) -> None: ...


class PeerCaptureVadEventRuntime(Protocol):
    async def handle_peer_owned_vad_event(self, event: object) -> None: ...


SelfCaptureVadEventRuntimeProvider = Callable[[], SelfCaptureVadEventRuntime | None]
PeerCaptureVadEventRuntimeProvider = Callable[[], PeerCaptureVadEventRuntime | None]


__all__ = [
    "PeerCaptureVadEventRuntime",
    "PeerCaptureVadEventRuntimeProvider",
    "SelfCaptureVadEventRuntime",
    "SelfCaptureVadEventRuntimeProvider",
]
