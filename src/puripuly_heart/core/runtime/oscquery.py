from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Final, Literal

from puripuly_heart.core.diagnostics import emit_failure, exception_diagnostics
from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.osc.oscquery_contract import (
    OscQueryAdvertisement,
    OscQueryServiceInfo,
    OscQueryServicePort,
)
from puripuly_heart.core.runtime.logging import emit_basic_log

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
_MDNS_COALESCE_SECONDS: Final = 0.05
logger = logging.getLogger(__name__)


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
    healthy_probe_interval_seconds: float = 60.0
    discovery_retry_delays_seconds: tuple[float, ...] = (2.0, 5.0, 15.0, 30.0)
    application_current: Callable[[], bool] | None = None
    mode: OscConnectionMode = "off"
    service_info: OscQueryServiceInfo | None = field(init=False, default=None)
    advertised_port: int | None = field(init=False, default=None)
    effective_send_port: int | None = field(init=False, default=None)
    effective_receive_port: int | None = field(init=False, default=None)
    avatar_tree: Mapping[str, object] | None = field(init=False, default=None)
    _started: bool = field(init=False, default=False)
    failures: tuple[ErrorDiagnostics, ...] = field(init=False, default=())
    _service_started: bool = field(init=False, default=False)
    _fallback_send_port: int = field(init=False, default=VRCHAT_OSC_DEFAULT_INPUT_PORT)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)
    _monitor_scope: LifecycleScope = field(
        init=False,
        default_factory=lambda: LifecycleScope("OscQueryRuntimeMonitor"),
        repr=False,
    )
    _monitor_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _discovery_wakeup: asyncio.Event = field(init=False, default_factory=asyncio.Event, repr=False)
    _mdns_refresh_at: float | None = field(init=False, default=None)
    _probe_at: float | None = field(init=False, default=None)
    _retry_step: int = field(init=False, default=0)
    _discovery_warned: bool = field(init=False, default=False)

    @property
    def started(self) -> bool:
        return self._started

    async def start(
        self,
        mode: OscConnectionMode,
        *,
        manual_send_port: int = 9000,
        snapshot_generation: int | None = None,
        preserve_receiver: bool = False,
        publish_snapshot: bool = True,
        is_current: Callable[[], bool] | None = None,
        report_discovery_failure: bool = False,
    ) -> None:
        async with self._lock:
            await self._stop_locked(stop_receiver=not preserve_receiver)
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
                await self._coalesce_startup_changes()
                self._mdns_refresh_at = None
                self.service_info = await self.service.discover_vrchat()
                if not self._application_current() or (is_current is not None and not is_current()):
                    await self._stop_locked(stop_receiver=not preserve_receiver)
                    return
                await self._apply_service_info(
                    self.service_info,
                    fallback_send_port=self._fallback_send_port,
                )
                if not self._application_current() or (is_current is not None and not is_current()):
                    await self._stop_locked(stop_receiver=not preserve_receiver)
                    return
                await self.service.advertise_receiver(
                    OscQueryAdvertisement(
                        host=self.receiver_host,
                        port=self.effective_receive_port,
                        parameters={"/avatar": "PuriPuly_*"},
                    )
                )
                self.advertised_port = self.effective_receive_port
                if not self._application_current() or (is_current is not None and not is_current()):
                    await self._stop_locked(stop_receiver=not preserve_receiver)
                    return
                self._started = True
                if self._mdns_refresh_at is not None:
                    await self._coalesce_startup_changes()
                    await self._refresh_locked(
                        publish_snapshot=False,
                        is_current=is_current,
                        report_discovery_failure=report_discovery_failure,
                    )
                else:
                    self._schedule_after_discovery(self.service_info, report_discovery_failure)
                if publish_snapshot and self.snapshot_publisher is not None:
                    await _maybe_await(self.snapshot_publisher("start", snapshot_generation))
                self._start_discovery_monitor()
            except BaseException:
                await self._stop_locked(stop_receiver=not preserve_receiver)
                raise

    async def refresh(
        self,
        *,
        publish_snapshot: bool = True,
        force_requery: bool = False,
        is_current: Callable[[], bool] | None = None,
        report_discovery_failure: bool = False,
    ) -> bool:
        async with self._lock:
            return await self._refresh_locked(
                publish_snapshot=publish_snapshot,
                force_requery=force_requery,
                is_current=is_current,
                report_discovery_failure=report_discovery_failure,
            )

    async def target_is_applied(self, service: OscQueryServiceInfo | None) -> bool:
        async with self._lock:
            return (
                service is not None
                and service.osc_send_port is not None
                and self._started
                and self.mode == "automatic"
                and self._application_current()
                and self.service_info == service
                and self.effective_send_port == service.osc_send_port
            )

    async def _refresh_locked(
        self,
        *,
        publish_snapshot: bool,
        force_requery: bool = False,
        is_current: Callable[[], bool] | None = None,
        report_discovery_failure: bool = False,
        preferred_service_id: str | None = None,
    ) -> bool:
        if not self._started or self.mode != "automatic" or not self._application_current():
            return False
        resync_parent_generation = (
            self.resync_generation_provider()
            if publish_snapshot
            and self.resync_starter is not None
            and self.resync_generation_provider is not None
            else None
        )
        pending_mdns = self._mdns_refresh_at
        self._mdns_refresh_at = None
        try:
            service_info = await self.service.discover_vrchat(
                preferred_service_id=preferred_service_id
            )
        except Exception, asyncio.CancelledError:
            if pending_mdns is not None and self._mdns_refresh_at is None:
                self._mdns_refresh_at = pending_mdns
                self._discovery_wakeup.set()
            raise
        if not self._application_current() or (is_current is not None and not is_current()):
            return False
        self._schedule_after_discovery(service_info, report_discovery_failure)
        changed = service_info != self.service_info
        if not changed and not force_requery:
            return service_info is not None and service_info.osc_send_port is not None
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
        if not self._application_current() or (is_current is not None and not is_current()):
            return False
        if (
            publish_snapshot
            and self.service_info is not None
            and self.snapshot_publisher is not None
            and (self.resync_starter is None or snapshot_generation is not None)
        ):
            await _maybe_await(self.snapshot_publisher("discovery", snapshot_generation))
        return service_info is not None and service_info.osc_send_port is not None

    async def on_avatar_change(self, *, snapshot_generation: int | None = None) -> None:
        async with self._lock:
            await self._refresh_locked(
                publish_snapshot=False,
                force_requery=True,
            )
        if self._application_current() and self.snapshot_publisher is not None:
            await _maybe_await(self.snapshot_publisher("avatar_change", snapshot_generation))

    async def stop(self) -> None:
        async with self._lock:
            await self._stop_locked()

    async def _stop_locked(self, *, stop_receiver: bool = True) -> None:
        await self._monitor_scope.close()
        self._monitor_scope = LifecycleScope("OscQueryRuntimeMonitor")
        self._monitor_task = None
        if self.advertised_port is not None:
            try:
                await self.service.unadvertise_receiver()
            except Exception as exc:
                self._retain_failure(exc, "unadvertise_receiver")
        if stop_receiver and (self._started or self.effective_receive_port is not None):
            try:
                await self.receiver_stop()
            except Exception as exc:
                self._retain_failure(exc, "receiver_stop")
        if self._service_started and self.mode == "automatic":
            try:
                await self.service.stop()
            except Exception as exc:
                self._retain_failure(exc, "service_stop")
        self._started = False
        self._service_started = False
        self._mdns_refresh_at = None
        self._probe_at = None
        self._retry_step = 0
        self._discovery_warned = False
        self._discovery_wakeup.clear()
        self.service_info = None
        self.advertised_port = None
        self.effective_send_port = None
        self.effective_receive_port = None
        self.avatar_tree = None

    def _services_changed(self, _services: object) -> None:
        if self.mode != "automatic" or not self._service_started:
            return
        if self._mdns_refresh_at is None:
            self._mdns_refresh_at = asyncio.get_running_loop().time() + _MDNS_COALESCE_SECONDS
            self._discovery_wakeup.set()

    async def _coalesce_startup_changes(self) -> None:
        deadline = self._mdns_refresh_at
        if deadline is not None:
            delay = deadline - asyncio.get_running_loop().time()
            if delay > 0:
                await asyncio.sleep(delay)

    def _application_current(self) -> bool:
        return self.application_current is None or self.application_current()

    async def _apply_service_info(
        self,
        service: OscQueryServiceInfo | None,
        *,
        fallback_send_port: int,
    ) -> None:
        if not self._application_current():
            return
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
        if (
            self._application_current()
            and self.avatar_inspector is not None
            and getattr(self.service, "avatar_observation", "available") == "available"
        ):
            await _maybe_await(self.avatar_inspector(self.avatar_tree or {}))

    def _start_discovery_monitor(self) -> None:
        if self._monitor_task is not None and not self._monitor_task.done():
            return
        self._monitor_task = start_lifecycle_task(
            self._monitor_scope,
            self._monitor_discovery(),
            name="discovery",
        )

    def _schedule_after_discovery(
        self, service_info: OscQueryServiceInfo | None, report_discovery_failure: bool
    ) -> None:
        unavailable = getattr(self.service, "discovery_observation", None) == "unavailable"
        usable = service_info is not None and service_info.osc_send_port is not None
        recovery_reported = False
        if report_discovery_failure or (unavailable and not self._discovery_warned):
            failure = getattr(self.service, "discovery_failure", None)
            if isinstance(failure, ErrorDiagnostics):
                emit_failure(
                    logger,
                    failure,
                    severity="warning" if unavailable else "info",
                    disposition="recovered" if usable else "degraded",
                )
                recovery_reported = usable
                if unavailable:
                    self._discovery_warned = True
        if usable:
            if self._discovery_warned and not recovery_reported:
                emit_basic_log(logger, "[OSCQuery] Discovery recovered")
            self._probe_at = asyncio.get_running_loop().time() + self.healthy_probe_interval_seconds
        elif unavailable or service_info is not None:
            self._schedule_retry()
            return
        elif self._mdns_refresh_at is not None:
            return
        else:
            self._probe_at = None
        self._retry_step = 0
        self._discovery_warned = False
        self._discovery_wakeup.set()

    def _schedule_retry(self) -> None:
        delays = self.discovery_retry_delays_seconds
        delay = delays[min(self._retry_step, len(delays) - 1)]
        self._probe_at = asyncio.get_running_loop().time() + delay
        self._retry_step = min(self._retry_step + 1, len(delays))
        self._discovery_wakeup.set()

    def _next_discovery_at(self) -> float | None:
        return self._mdns_refresh_at if self._mdns_refresh_at is not None else self._probe_at

    async def _monitor_discovery(self) -> None:
        loop = asyncio.get_running_loop()
        while self._started and self.mode == "automatic":
            self._discovery_wakeup.clear()
            deadline = self._next_discovery_at()
            if deadline is None or deadline > loop.time():
                try:
                    async with asyncio.timeout_at(deadline):
                        await self._discovery_wakeup.wait()
                except TimeoutError:
                    pass
                continue
            async with self._lock:
                deadline = self._next_discovery_at()
                if deadline is None or deadline > loop.time():
                    continue
                preferred_service_id = (
                    self.service_info.service_id
                    if self._mdns_refresh_at is None and not self._retry_step and self.service_info
                    else None
                )
                self._mdns_refresh_at = None
                self._probe_at = None
                if not self._application_current():
                    continue
                try:
                    await self._refresh_locked(
                        publish_snapshot=True,
                        preferred_service_id=preferred_service_id,
                    )
                except Exception as exc:
                    if self._application_current():
                        if not self._discovery_warned:
                            self._retain_failure(exc, "discovery_monitor")
                            self._discovery_warned = True
                        self._schedule_retry()

    def _retain_failure(self, exc: Exception, operation: str) -> None:
        diagnostics = exception_diagnostics(
            exc, component="osc_control", operation=operation, code="oscquery_unavailable"
        )
        self.failures = (*self.failures[-7:], diagnostics)
        emit_failure(logger, diagnostics, severity="warning", disposition="degraded")


async def _maybe_await(value: object) -> object:
    if isinstance(value, Awaitable):
        return await value
    return value


__all__ = ["OscQueryRuntime", "VRCHAT_OSC_DEFAULT_INPUT_PORT"]
