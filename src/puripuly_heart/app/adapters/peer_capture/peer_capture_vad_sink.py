from __future__ import annotations

from dataclasses import dataclass

from puripuly_heart.app.ports.capture_vad_runtime import (
    PeerCaptureVadEventRuntime,
    PeerCaptureVadEventRuntimeProvider,
)


@dataclass(frozen=True, slots=True)
class PeerCaptureVadSinkAdapter:
    runtime_provider: PeerCaptureVadEventRuntimeProvider

    async def handle_owned_vad_event(self, event: object) -> None:
        runtime = self.runtime_provider()
        if runtime is None:
            raise RuntimeError("Peer VAD sink requires the Peer translation owner")
        await runtime.handle_peer_owned_vad_event(event)

    async def observe_source_activity(
        self,
        *,
        speech_observed: bool,
        observed_at_monotonic_s: float,
    ) -> None:
        runtime = self.runtime_provider()
        if runtime is None:
            raise RuntimeError("Peer VAD sink requires the Peer translation owner")
        await runtime.observe_source_activity(
            speech_observed=speech_observed,
            observed_at_monotonic_s=observed_at_monotonic_s,
        )

    async def observe_pending_source_work(self, *, pending: bool) -> None:
        runtime = self.runtime_provider()
        if runtime is None:
            raise RuntimeError("Peer VAD sink requires the Peer translation owner")
        observe = getattr(runtime, "observe_pending_source_work", None)
        if callable(observe):
            await observe(pending=pending)


__all__ = [
    "PeerCaptureVadEventRuntime",
    "PeerCaptureVadEventRuntimeProvider",
    "PeerCaptureVadSinkAdapter",
]
