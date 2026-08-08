from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer

from puripuly_heart.core.osc.control_schema import (
    OSC_AVATAR_CHANGE_ADDRESS,
    OSC_MUTE_SELF_ADDRESS,
    is_puripuly_parameter_address,
)

logger = logging.getLogger(__name__)

VRC_OSC_RECEIVER_HOST = "127.0.0.1"
VRC_OSC_RECEIVER_PORT = 9001
VRC_OSC_MUTE_ADDRESS = OSC_MUTE_SELF_ADDRESS


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


class VrcOscReceiver:
    def __init__(
        self,
        state: VrcMicState,
        *,
        host: str = VRC_OSC_RECEIVER_HOST,
        port: int = VRC_OSC_RECEIVER_PORT,
        mute_delay_s: float = 0.4,
        mute_packet_handler: Callable[[bool], object] | None = None,
        control_packet_handler: Callable[[str, tuple[Any, ...]], object] | None = None,
        avatar_change_handler: Callable[[tuple[Any, ...]], object] | None = None,
        packet_handler: Callable[[str, tuple[Any, ...]], object] | None = None,
    ) -> None:
        self.state = state
        self.host = host
        self.port = port
        self.effective_port = port
        self.mute_delay_s = mute_delay_s
        self._mute_packet_handler = mute_packet_handler
        self._control_packet_handler = control_packet_handler
        self._avatar_change_handler = avatar_change_handler
        self._packet_handler = packet_handler
        self.transport = None
        self._mute_task: asyncio.Task[None] | None = None

    def mute_handler(self, address: str, *args: Any) -> None:
        _ = address
        if not args:
            return
        is_muted = bool(args[0])

        if self._mute_packet_handler is not None:
            self._mute_packet_handler(is_muted)
            return

        if self._mute_task is not None and not self._mute_task.done():
            self._mute_task.cancel()

        loop = asyncio.get_running_loop()
        self._mute_task = loop.create_task(self._apply_mute_state(is_muted))

    def message_handler(self, address: str, *args: Any) -> None:
        if address == VRC_OSC_MUTE_ADDRESS:
            self.mute_handler(address, *args)
            return
        values = tuple(args)
        if self._packet_handler is not None:
            self._packet_handler(address, values)
        if address == OSC_AVATAR_CHANGE_ADDRESS:
            if self._avatar_change_handler is not None:
                self._avatar_change_handler(values)
            return
        if is_puripuly_parameter_address(address) and self._control_packet_handler is not None:
            self._control_packet_handler(address, values)

    async def _apply_mute_state(self, is_muted: bool) -> None:
        try:
            if is_muted:
                await asyncio.sleep(self.mute_delay_s)

            if self.state.update(is_muted):
                logger.info("[OSC Receiver] VRChat mic muted state applied: %s", is_muted)
        except asyncio.CancelledError:
            raise

    async def start(self) -> None:
        if self.transport is not None:
            return

        # A restarted receiver must wait for a fresh VRChat mute edge.
        self.state.reset()

        dispatcher = Dispatcher()
        dispatcher.map(VRC_OSC_MUTE_ADDRESS, self.mute_handler)
        dispatcher.map(OSC_AVATAR_CHANGE_ADDRESS, self.message_handler)
        dispatcher.set_default_handler(self.message_handler)

        loop = asyncio.get_running_loop()
        try:
            server = AsyncIOOSCUDPServer((self.host, self.port), dispatcher, loop)
            self.transport, _ = await server.create_serve_endpoint()
            get_extra_info = getattr(self.transport, "get_extra_info", None)
            if callable(get_extra_info):
                sockname = get_extra_info("sockname")
                if isinstance(sockname, tuple) and len(sockname) >= 2:
                    self.effective_port = int(sockname[1])
        except OSError:
            logger.exception(
                "[OSC Receiver] Failed to start AsyncIOOSCUDPServer on %s:%s",
                self.host,
                self.port,
            )
            raise

        logger.info(
            "[OSC Receiver] Listening on %s:%s for VRChat parameters",
            self.host,
            self.port,
        )

    def stop(self) -> None:
        if self._mute_task is not None and not self._mute_task.done():
            self._mute_task.cancel()
        self._mute_task = None

        if self.transport is not None:
            self.transport.close()
            self.transport = None
            self.effective_port = self.port
            logger.info("[OSC Receiver] Stopped listening.")
