from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from puripuly_heart.core.vad.smart_turn import (
    SmartTurnEventSinkFactory,
    SmartTurnExperimentConfig,
)

PeerCaptureAudioLoopRunner = Callable[..., Awaitable[None]]
PeerCaptureAudioLoopDetailedLog = Callable[[str], object]
PeerCaptureAudioLoopDetailedEnabled = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class PeerCaptureAudioLoopAdapter:
    runner: PeerCaptureAudioLoopRunner
    log_detailed: PeerCaptureAudioLoopDetailedLog
    is_detailed_enabled: PeerCaptureAudioLoopDetailedEnabled
    smart_turn_config_provider: Callable[[], SmartTurnExperimentConfig] | None = None
    smart_turn_event_sink_factory: SmartTurnEventSinkFactory | None = None

    async def __call__(self, **kwargs: object) -> None:
        smart_turn_kwargs: dict[str, object] = {}
        if self.smart_turn_config_provider is not None:
            smart_turn_kwargs["smart_turn_config"] = self.smart_turn_config_provider()
        if self.smart_turn_event_sink_factory is not None:
            smart_turn_kwargs["vad_event_sink_factory"] = self.smart_turn_event_sink_factory
        await self.runner(
            **kwargs,
            channel_label="peer",
            is_detailed_enabled=self.is_detailed_enabled,
            log_detailed=lambda message: self.log_detailed(message),
            **smart_turn_kwargs,
        )


__all__ = [
    "PeerCaptureAudioLoopAdapter",
    "PeerCaptureAudioLoopDetailedEnabled",
    "PeerCaptureAudioLoopDetailedLog",
    "PeerCaptureAudioLoopRunner",
]
