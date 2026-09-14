from __future__ import annotations

from dataclasses import dataclass

from puripuly_heart.app.ports.capture_vad_runtime import (
    SelfCaptureVadEventRuntime,
    SelfCaptureVadEventRuntimeProvider,
)


@dataclass(frozen=True, slots=True)
class SelfCaptureVadSinkAdapter:
    runtime_provider: SelfCaptureVadEventRuntimeProvider

    async def handle_vad_event(self, event: object) -> None:
        runtime = self._require_runtime()
        await runtime.handle_vad_event(event)

    async def reject_owned_segment(
        self,
        event: object,
        *,
        reason: str,
        outcome: str,
    ) -> None:
        runtime = self._require_runtime()
        await runtime.reject_owned_segment(event, reason=reason, outcome=outcome)

    async def fail_owned_segment(self, event: object, *, reason: str) -> None:
        runtime = self._require_runtime()
        await runtime.fail_owned_segment(event, reason=reason)

    def _require_runtime(self) -> SelfCaptureVadEventRuntime:
        runtime = self.runtime_provider()
        if runtime is None:
            raise RuntimeError("Self VAD sink requires the Self translation owner")
        return runtime


__all__ = [
    "SelfCaptureVadEventRuntime",
    "SelfCaptureVadEventRuntimeProvider",
    "SelfCaptureVadSinkAdapter",
]
