from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

SelfCaptureAudioLoopRunner = Callable[..., Awaitable[None]]
SelfCaptureAudioGateProvider = Callable[[], object | None]
SelfCaptureAudioLoopDetailedLog = Callable[[str], object]
SelfCaptureAudioLoopBasicLog = Callable[[str], object]
SelfCaptureAudioLoopDetailedEnabled = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class SelfCaptureAudioLoopAdapter:
    runner: SelfCaptureAudioLoopRunner
    audio_gate_provider: SelfCaptureAudioGateProvider
    log_detailed: SelfCaptureAudioLoopDetailedLog
    log_basic: SelfCaptureAudioLoopBasicLog
    is_detailed_enabled: SelfCaptureAudioLoopDetailedEnabled

    async def __call__(self, **kwargs: object) -> None:
        await self.runner(
            **kwargs,
            audio_gate=self.audio_gate_provider(),
            channel_label="self",
            is_detailed_enabled=self.is_detailed_enabled,
            log_detailed=lambda message: self.log_detailed(message),
            log_basic=lambda message: self.log_basic(message),
        )


__all__ = [
    "SelfCaptureAudioLoopBasicLog",
    "SelfCaptureAudioGateProvider",
    "SelfCaptureAudioLoopAdapter",
    "SelfCaptureAudioLoopDetailedEnabled",
    "SelfCaptureAudioLoopDetailedLog",
    "SelfCaptureAudioLoopRunner",
]
