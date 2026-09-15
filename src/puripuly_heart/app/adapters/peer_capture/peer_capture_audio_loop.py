from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

PeerCaptureAudioLoopRunner = Callable[..., Awaitable[None]]
PeerCaptureAudioLoopBasicLog = Callable[[str], object]


@dataclass(frozen=True, slots=True)
class PeerCaptureAudioLoopAdapter:
    runner: PeerCaptureAudioLoopRunner
    log_basic: PeerCaptureAudioLoopBasicLog

    async def __call__(self, **kwargs: object) -> None:
        await self.runner(
            **kwargs,
            channel_label="peer",
            log_basic=lambda message: self.log_basic(message),
        )


__all__ = [
    "PeerCaptureAudioLoopBasicLog",
    "PeerCaptureAudioLoopAdapter",
    "PeerCaptureAudioLoopRunner",
]
