from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Final, Literal

from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task
from puripuly_heart.core.osc.oscquery_contract import (
    OscQueryAdvertisement,
    OscQueryServiceInfo,
    OscQueryServicePort,
)

OscConnectionMode = Literal["automatic", "manual", "off"]
OscReceiverStarter = Callable[[], Awaitable[object]]
OscReceiverStopper = Callable[[], Awaitable[object]]
OscEffectivePort = Callable[[], int]
OscDestinationChanged = Callable[[str, int], Awaitable[None] | None]
OscSnapshotPublisher = Callable[[str, int | None], Awaitable[None] | None]
OscResyncGenerationProvider = Callable[[], int]
OscResyncStarter = Callable[[str, int | None], int | None]
OscAvatarInspector = Callable[[Mapping[str, object]], Awaitable[None] | None]
VRCHAT_OSC_DEFAULT_INPUT_PORT: Final = 9000


@dataclass(slots=True)
class OscQueryRuntime:
    service: OscQueryServicePort
    receiver_start: OscReceiverStarter
    receiver_stop: OscReceiverStopper
    receiver_effective_port: OscEffectivePort
    sender_destination_changed: OscDestinationChanged | None = None
    snapshot_publisher: OscSnapshotPublisher | None = None
    resync_starter: OscResyncStarter | None = None
    resync_generation_provider: OscResyncGenerationProvider | None = None
    avatar_inspector: OscAvatarInspector | None = None
    receiver_host: str = "127.0.0.1"
    discovery_poll_interval_seconds: float = 2.0
    mode: OscConnectionMode = "off"
    service_info: OscQueryServiceInfo | None = field(init=False, default=None)
    advertised_port: int | None = field(init=False, default=None)
    effective_send_port: int | None = field(init=False, default=None)
    effective_receive_port: int | None = field(init=False, default=None)
    avatar_tree: Mapping[str, object] | None = field(init=False, default=None)
    _started: bool = field(init=False, default=False)
    _service_started: bool = field(init=False, default=False)
    _refresh_requested: bool = field(init=False, default=False)
    _fallback_send_port: int = field(init=False, default=VRCHAT_OSC_DEFAULT_INPUT_PORT)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)
    _monitor_scope: LifecycleScope = field(
        init=False,
        default_factory=lambda: LifecycleScope("OscQueryRuntimeMonitor"),
        repr=False,
    )
    _monitor_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)

    @property
    def started(self) -> bool:
        return self._started

    async def start(
        self,
        mode: OscConnectionMode,
        *,
        manual_send_port: int = 9000,
        snapshot_generation: int | None = None,
    ) -> None:
        async with self._lock:
            await self._stop_locked()
            self.mode = mode
            if mode == "off":
                return
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
                    await _maybe_await(self.snapshot_publisher("start", snapshot_generation))
                return

            self._fallback_send_port = VRCHAT_OSC_DEFAULT_INPUT_PORT
            try:
                self._service_started = True
                await self.service.start(self._services_changed)
                await self.receiver_start()
                self.effective_receive_port = self.receiver_effective_port()
                self.service_info = await self.service.discover_vrchat()
                await self._apply_service_info(
                    self.service_info,
                    fallback_send_port=self._fallback_send_port,
                )
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
                    await _maybe_await(self.snapshot_publisher("start", snapshot_generation))
                self._start_discovery_monitor()
            except BaseException:
                await self._stop_locked()
                raise

    async def refresh(self, *, publish_snapshot: bool = True) -> None:
        async with self._lock:
            await self._refresh_locked(publish_snapshot=publish_snapshot)

    async def _refresh_locked(
        self,
        *,
        publish_snapshot: bool,
        force_requery: bool = False,
    ) -> None:
        if not self._started or self.mode != "automatic":
            return
        resync_parent_generation = (
            self.resync_generation_provider()
            if publish_snapshot
            and self.resync_starter is not None
            and self.resync_generation_provider is not None
            else None
        )
        service_info = await self.service.discover_vrchat()
        changed = service_info != self.service_info
        if not changed and not force_requery:
            return
        snapshot_generation = None
        if (
            publish_snapshot
            and service_info is not None
            and self.snapshot_publisher is not None
            and self.resync_starter is not None
        ):
            snapshot_generation = self.resync_starter(
                "discovery",
                resync_parent_generation,
            )
        self.service_info = service_info
        await self._apply_service_info(
            self.service_info,
            fallback_send_port=self._fallback_send_port,
        )
        if (
            publish_snapshot
            and self.service_info is not None
            and self.snapshot_publisher is not None
            and (self.resync_starter is None or snapshot_generation is not None)
        ):
            await _maybe_await(self.snapshot_publisher("discovery", snapshot_generation))

    async def on_avatar_change(self, *, snapshot_generation: int | None = None) -> None:
        async with self._lock:
            await self._refresh_locked(
                publish_snapshot=False,
                force_requery=True,
            )
        if self.snapshot_publisher is not None:
            await _maybe_await(self.snapshot_publisher("avatar_change", snapshot_generation))

    async def stop(self) -> None:
        async with self._lock:
            await self._stop_locked()

    async def _stop_locked(self) -> None:
        await self._monitor_scope.close()
        self._monitor_scope = LifecycleScope("OscQueryRuntimeMonitor")
        self._monitor_task = None
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
        fallback_send_port: int,
    ) -> None:
        destination_port = (
            service.osc_send_port
            if service is not None and service.osc_send_port is not None
            else fallback_send_port
        )
        destination_host = service.host if service is not None and service.host else "127.0.0.1"
        self.effective_send_port = destination_port
        if self.sender_destination_changed is not None:
            await _maybe_await(self.sender_destination_changed(destination_host, destination_port))
        self.avatar_tree = await self.service.query_avatar(service) if service is not None else None
        if self.avatar_inspector is not None:
            await _maybe_await(self.avatar_inspector(self.avatar_tree or {}))

    def _start_discovery_monitor(self) -> None:
        if self._monitor_task is not None and not self._monitor_task.done():
            return
        self._monitor_task = start_lifecycle_task(
            self._monitor_scope,
            self._monitor_discovery(),
            name="discovery",
        )

    async def _monitor_discovery(self) -> None:
        interval = max(0.1, float(self.discovery_poll_interval_seconds))
        while self._started and self.mode == "automatic":
            await asyncio.sleep(interval)
            try:
                await self.refresh()
            except Exception:
                continue


async def _maybe_await(value: object) -> object:
    if isinstance(value, Awaitable):
        return await value
    return value


__all__ = ["OscQueryRuntime", "VRCHAT_OSC_DEFAULT_INPUT_PORT"]
