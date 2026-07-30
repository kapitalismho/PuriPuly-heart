from __future__ import annotations

from dataclasses import dataclass, field

from puripuly_heart.domain.events import STTSessionState, STTSessionStateEvent
from puripuly_heart.domain.models import ChannelId


@dataclass(slots=True)
class SttSessionStateProjection:
    _states: dict[ChannelId, STTSessionState | None] = field(
        default_factory=lambda: {"self": None, "peer": None},
    )

    def record(self, event: object) -> None:
        if isinstance(event, STTSessionStateEvent):
            self._states[event.channel] = event.state

    def state(self, channel: ChannelId) -> STTSessionState | None:
        return self._states[channel]


__all__ = ["SttSessionStateProjection"]
