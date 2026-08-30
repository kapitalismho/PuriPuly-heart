from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

VRC_OSC_RECEIVER_HOST = "127.0.0.1"
VRC_OSC_RECEIVER_PORT = 9001


@dataclass(slots=True)
class VrcMicState:
    muted: bool | None = None

    def update(self, muted: bool) -> bool:
        if self.muted == muted:
            return False
        self.muted = muted
        return True

    def reset(self) -> None:
        self.muted = None


class OscReceiverPort(Protocol):
    effective_port: int

    async def start(self) -> None: ...

    def stop(self) -> Awaitable[None] | None: ...


class VrcOscReceiverFactory(Protocol):
    def __call__(
        self,
        state: VrcMicState,
        *,
        host: str,
        port: int,
        mute_delay_s: float,
        mute_packet_handler: Callable[[bool], object] | None,
        control_packet_handler: Callable[[str, tuple[object, ...]], object] | None,
        avatar_change_handler: Callable[[tuple[object, ...]], object] | None,
        packet_handler: Callable[[str, tuple[object, ...]], object] | None,
    ) -> OscReceiverPort: ...


__all__ = [
    "OscReceiverPort",
    "VRC_OSC_RECEIVER_HOST",
    "VRC_OSC_RECEIVER_PORT",
    "VrcMicState",
    "VrcOscReceiverFactory",
]
