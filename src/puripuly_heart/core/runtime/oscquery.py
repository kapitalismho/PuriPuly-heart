from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

from puripuly_heart.app.ports.oscquery import (
    OscQueryAdvertisement,
    OscQueryServiceInfo,
    OscQueryServicePort,
)

OscConnectionMode = Literal["automatic", "manual", "off"]
OscReceiverStarter = Callable[[], Awaitable[object]]
OscReceiverStopper = Callable[[], Awaitable[object]]
OscEffectivePort = Callable[[], int]
OscDestinationChanged = Callable[[str, int], Awaitable[None] | None]
OscSnapshotPublisher = Callable[[str], Awaitable[None] | None]
OscAvatarInspector = Callable[[Mapping[str, object]], Awaitable[None] | None]


@dataclass(slots=True)
class OscQueryRuntime:
    service: OscQueryServicePort
    receiver_start: OscReceiverStarter
    receiver_stop: OscReceiverStopper
    receiver_effective_port: OscEffectivePort
    sender_destination_changed: OscDestinationChanged | None = None
    snapshot_publisher: OscSnapshotPublisher | None = None
    avatar_inspector: OscAvatarInspector | None = None
    receiver_host: str = "127.0.0.1"
    mode: OscConnectionMode = "off"
    service_info: OscQueryServiceInfo | None = field(init=False, default=None)
    advertised_port: int | None = field(init=False, default=None)
    effective_send_port: int | None = field(init=False, default=None)
    effective_receive_port: int | None = field(init=False, default=None)
    avatar_tree: Mapping[str, object] | None = field(init=False, default=None)
    _started: bool = field(init=False, default=False)
    _service_started: bool = field(init=False, default=False)
    _refresh_requested: bool = field(init=False, default=False)
    _manual_send_port: int = field(init=False, default=9000)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)

    @property
    def started(self) -> bool:
        return self._started

    async def start(
        self,
        mode: OscConnectionMode,
        *,
        manual_send_port: int = 9000,
    ) -> None:
        async with self._lock:
            await self._stop_locked()
            self.mode = mode
            if mode == "off":
                return
            self._manual_send_port = int(manual_send_port)
            if mode == "manual":
                await self.receiver_start()
                self.effective_receive_port = self.receiver_effective_port()
                self.effective_send_port = manual_send_port
                self._started = True
                if self.sender_destination_changed is not None:
                    await _maybe_await(
                        self.sender_destination_changed(self.receiver_host, manual_send_port)
                    )
                if self.snapshot_publisher is not None:
                    await _maybe_await(self.snapshot_publisher("start"))
                return

            try:
                self._service_started = True
                await self.service.start(self._services_changed)
                await self.receiver_start()
                self.effective_receive_port = self.receiver_effective_port()
                self.service_info = await self.service.discover_vrchat()
                await self._apply_service_info(self.service_info, manual_send_port=manual_send_port)
                await self.service.advertise_receiver(
                    OscQueryAdvertisement(
                        host=self.receiver_host,
                        port=self.effective_receive_port,
                        parameters={"/avatar": "PuriPuly_*"},
                    )
                )
                self.advertised_port = self.effective_receive_port
                self._started = True
                if self._refresh_requested:
                    self._refresh_requested = False
                    await self._refresh_locked(publish_snapshot=False)
                if self.snapshot_publisher is not None:
                    await _maybe_await(self.snapshot_publisher("start"))
            except BaseException:
                await self._stop_locked()
                raise

    async def refresh(self, *, publish_snapshot: bool = True) -> None:
        async with self._lock:
            await self._refresh_locked(publish_snapshot=publish_snapshot)

    async def _refresh_locked(self, *, publish_snapshot: bool) -> None:
        if not self._started or self.mode != "automatic":
            return
        self.service_info = await self.service.discover_vrchat()
        await self._apply_service_info(
            self.service_info,
            manual_send_port=self._manual_send_port,
        )
        if self.service_info is not None:
            self.avatar_tree = await self.service.query_avatar(self.service_info)
            if self.avatar_inspector is not None:
                await _maybe_await(self.avatar_inspector(self.avatar_tree))
        if publish_snapshot and self.snapshot_publisher is not None:
            await _maybe_await(self.snapshot_publisher("discovery"))

    async def on_avatar_change(self) -> None:
        await self.refresh(publish_snapshot=False)
        if self.snapshot_publisher is not None:
            await _maybe_await(self.snapshot_publisher("avatar_change"))

    async def stop(self) -> None:
        async with self._lock:
            await self._stop_locked()

    async def _stop_locked(self) -> None:
        if self.advertised_port is not None:
            with contextlib.suppress(Exception):
                await self.service.unadvertise_receiver()
        if self._started or self.effective_receive_port is not None:
            with contextlib.suppress(Exception):
                await self.receiver_stop()
        if self._service_started and self.mode == "automatic":
            with contextlib.suppress(Exception):
                await self.service.stop()
        self._started = False
        self._service_started = False
        self._refresh_requested = False
        self.service_info = None
        self.advertised_port = None
        self.effective_send_port = None
        self.effective_receive_port = None
        self.avatar_tree = None

    async def _services_changed(self, _services: object) -> None:
        if self.mode != "automatic":
            return
        if not self._started:
            self._refresh_requested = True
            return
        await self.refresh()

    async def _apply_service_info(
        self,
        service: OscQueryServiceInfo | None,
        *,
        manual_send_port: int,
    ) -> None:
        destination_port = (
            service.osc_send_port
            if service is not None and service.osc_send_port is not None
            else manual_send_port
        )
        destination_host = service.host if service is not None and service.host else "127.0.0.1"
        self.effective_send_port = destination_port
        if self.sender_destination_changed is not None:
            await _maybe_await(self.sender_destination_changed(destination_host, destination_port))
        if service is not None:
            self.avatar_tree = await self.service.query_avatar(service)
            if self.avatar_inspector is not None:
                await _maybe_await(self.avatar_inspector(self.avatar_tree))


async def _maybe_await(value: object) -> object:
    if isinstance(value, Awaitable):
        return await value
    return value


__all__ = ["OscQueryRuntime"]
