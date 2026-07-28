from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


class PeerCaptureVadEventRuntime(Protocol):
    async def handle_peer_vad_event(self, event: object) -> None: ...


PeerCaptureVadEventRuntimeProvider = Callable[[], PeerCaptureVadEventRuntime | None]


@dataclass(frozen=True, slots=True)
class PeerCaptureVadSinkAdapter:
    runtime_provider: PeerCaptureVadEventRuntimeProvider

    async def handle_vad_event(self, event: object) -> None:
        runtime = self.runtime_provider()
        if runtime is None:
            raise RuntimeError("Peer VAD sink requires the production hub")
        await runtime.handle_peer_vad_event(event)


__all__ = [
    "PeerCaptureVadEventRuntime",
    "PeerCaptureVadEventRuntimeProvider",
    "PeerCaptureVadSinkAdapter",
]
