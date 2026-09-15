from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

SelfCaptureAudioLoopRunner = Callable[..., Awaitable[None]]
SelfCaptureAudioGateProvider = Callable[[], object | None]
SelfCaptureAudioLoopBasicLog = Callable[[str], object]


@dataclass(frozen=True, slots=True)
class SelfCaptureAudioLoopAdapter:
    runner: SelfCaptureAudioLoopRunner
    audio_gate_provider: SelfCaptureAudioGateProvider
    log_basic: SelfCaptureAudioLoopBasicLog

    async def __call__(self, **kwargs: object) -> None:
        await self.runner(
            **kwargs,
            audio_gate=self.audio_gate_provider(),
            channel_label="self",
            log_basic=lambda message: self.log_basic(message),
        )


__all__ = [
    "SelfCaptureAudioLoopBasicLog",
    "SelfCaptureAudioGateProvider",
    "SelfCaptureAudioLoopAdapter",
    "SelfCaptureAudioLoopRunner",
]
