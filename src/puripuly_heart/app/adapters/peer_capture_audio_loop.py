from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

PeerCaptureAudioLoopRunner = Callable[..., Awaitable[None]]
PeerCaptureAudioLoopDetailedLog = Callable[[str], object]
PeerCaptureAudioLoopDetailedEnabled = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class PeerCaptureAudioLoopAdapter:
    runner: PeerCaptureAudioLoopRunner
    log_detailed: PeerCaptureAudioLoopDetailedLog
    is_detailed_enabled: PeerCaptureAudioLoopDetailedEnabled

    async def __call__(self, **kwargs: object) -> None:
        await self.runner(
            **kwargs,
            channel_label="peer",
            is_detailed_enabled=self.is_detailed_enabled,
            log_detailed=lambda message: self.log_detailed(message),
        )


__all__ = [
    "PeerCaptureAudioLoopAdapter",
    "PeerCaptureAudioLoopDetailedEnabled",
    "PeerCaptureAudioLoopDetailedLog",
    "PeerCaptureAudioLoopRunner",
]
