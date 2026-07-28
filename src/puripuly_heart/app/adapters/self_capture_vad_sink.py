from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


class SelfCaptureVadEventRuntime(Protocol):
    async def handle_vad_event(self, event: object) -> None: ...


SelfCaptureVadEventRuntimeProvider = Callable[[], SelfCaptureVadEventRuntime | None]


@dataclass(frozen=True, slots=True)
class SelfCaptureVadSinkAdapter:
    runtime_provider: SelfCaptureVadEventRuntimeProvider

    async def handle_vad_event(self, event: object) -> None:
        runtime = self.runtime_provider()
        if runtime is None:
            raise RuntimeError("Self VAD sink requires the production hub")
        await runtime.handle_vad_event(event)


__all__ = [
    "SelfCaptureVadEventRuntime",
    "SelfCaptureVadEventRuntimeProvider",
    "SelfCaptureVadSinkAdapter",
]
