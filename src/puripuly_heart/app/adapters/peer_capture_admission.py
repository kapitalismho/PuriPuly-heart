from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from puripuly_heart.core.peer_capture import (
    PeerCaptureAdmission,
    PeerCaptureAdmissionStatus,
    PeerCaptureSessionConfig,
)

PeerCaptureRuntimeAvailable = Callable[[], bool]
PeerCaptureLocalReadiness = Callable[[], Awaitable[bool]]


@dataclass(frozen=True, slots=True)
class PeerCaptureAdmissionAdapter:
    runtime_available: PeerCaptureRuntimeAvailable
    ensure_local_ready: PeerCaptureLocalReadiness

    async def admit(self, config: PeerCaptureSessionConfig) -> PeerCaptureAdmission:
        if not self.runtime_available():
            return PeerCaptureAdmission(
                PeerCaptureAdmissionStatus.REJECTED,
                reason="runtime_unavailable",
            )
        if config.local_provider and not await self.ensure_local_ready():
            return PeerCaptureAdmission(
                PeerCaptureAdmissionStatus.PENDING,
                reason="provider_unavailable",
                retain_intent=True,
            )
        return PeerCaptureAdmission(PeerCaptureAdmissionStatus.ADMITTED)


__all__ = [
    "PeerCaptureAdmissionAdapter",
    "PeerCaptureLocalReadiness",
    "PeerCaptureRuntimeAvailable",
]
