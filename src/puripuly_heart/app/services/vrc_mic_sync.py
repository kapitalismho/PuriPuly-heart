from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from puripuly_heart.core.runtime.receiver import VrcMicReceiverRuntime


class VrcMicAudioGatePort(Protocol):
    def set_enabled(self, enabled: bool) -> None: ...

    def set_receiver_active(self, active: bool) -> None: ...

    def reset(self) -> None: ...


@dataclass(slots=True)
class VrcMicSyncOwner:
    state_provider: Callable[[], object | None]
    gate_provider: Callable[[], VrcMicAudioGatePort | None]
    receiver_factory: Callable[..., object]
    diagnostics_sink: Callable[[str, Mapping[str, object]], None]
    error_sink: Callable[[str], None]
    host: str
    port: int
    control_packet_handler: Callable[[str, tuple[object, ...]], object] | None = field(
        default=None,
        repr=False,
    )
    avatar_change_handler: Callable[[tuple[object, ...]], object] | None = field(
        default=None,
        repr=False,
    )
    packet_handler: Callable[[str, tuple[object, ...]], object] | None = field(
        default=None,
        repr=False,
    )
    _runtime: VrcMicReceiverRuntime | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _receiver: object | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _lock: asyncio.Lock | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _last_enabled: bool | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _control_active: bool = field(init=False, default=False, repr=False)
    _accepting_ingress: bool = field(init=False, default=True, repr=False)

    @property
    def runtime(self) -> VrcMicReceiverRuntime | None:
        return self._runtime

    @runtime.setter
    def runtime(self, runtime: VrcMicReceiverRuntime | None) -> None:
        if runtime is not None and not self._accepting_ingress:
            return
        self._runtime = runtime

    @property
    def receiver(self) -> object | None:
        return self._receiver

    @receiver.setter
    def receiver(self, receiver: object | None) -> None:
        if receiver is not None and not self._accepting_ingress:
            return
        self._receiver = receiver

    @property
    def last_enabled(self) -> bool | None:
        return self._last_enabled

    @last_enabled.setter
    def last_enabled(self, enabled: bool | None) -> None:
        self._last_enabled = enabled

    @property
    def control_active(self) -> bool:
        return self._control_active

    @property
    def effective_port(self) -> int:
        runtime = self._runtime
        if runtime is None:
            return self.port
        return int(getattr(runtime, "effective_port", self.port))

    @property
    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @property
    def accepting_ingress(self) -> bool:
        return self._accepting_ingress

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": "VrcMicSyncOwner",
            "resource_fields": (
                "_runtime",
                "_receiver",
                "_lock",
                "_control_active",
                "_accepting_ingress",
            ),
            "stop_ingress": "reject runtime creation and configuration",
            "shutdown_policy": "stop ingress, close the runtime and keep the audio gate disabled",
            "late_callback_rule": "closed owners cannot recreate the receiver runtime",
        }

    def stop_ingress(self) -> None:
        self._accepting_ingress = False

    def get_runtime(self) -> VrcMicReceiverRuntime | None:
        if not self._accepting_ingress:
            return None
        state = self.state_provider()
        if state is None:
            return None
        if self._runtime is None:
            self._runtime = VrcMicReceiverRuntime(
                state=state,
                host=self.host,
                port=self.port,
                receiver_factory=self.receiver_factory,
                diagnostics_sink=self.diagnostics_sink,
                state_changed=self.sync_runtime_receiver,
                control_packet_handler=self.control_packet_handler,
                avatar_change_handler=self.avatar_change_handler,
                packet_handler=self.packet_handler,
            )
        return self._runtime

    def set_packet_handlers(
        self,
        *,
        control_packet_handler: Callable[[str, tuple[object, ...]], object] | None = None,
        avatar_change_handler: Callable[[tuple[object, ...]], object] | None = None,
        packet_handler: Callable[[str, tuple[object, ...]], object] | None = None,
    ) -> None:
        if self._runtime is not None or self._receiver is not None:
            raise RuntimeError("receiver packet handlers cannot change while running")
        self.control_packet_handler = control_packet_handler
        self.avatar_change_handler = avatar_change_handler
        self.packet_handler = packet_handler

    async def configure_control(
        self,
        *,
        active: bool,
        host: str,
        port: int,
        force_restart: bool = False,
    ) -> None:
        async with self.lock:
            if not self._accepting_ingress:
                return
            endpoint_changed = self.host != host or self.port != port
            self.host = host
            self.port = int(port)
            self._control_active = bool(active)
            if (endpoint_changed or force_restart) and self._receiver is not None:
                await self._stop_locked()
            runtime = self._runtime
            configure_endpoint = getattr(runtime, "configure_endpoint", None)
            if callable(configure_endpoint) and self._receiver is None:
                configure_endpoint(self.host, self.port)
            await self._reconcile_locked()

    async def ensure_receiver(self) -> object | None:
        async with self.lock:
            if not self._accepting_ingress:
                return None
            if not (self._control_active or bool(self._last_enabled)):
                return None
            return await self._start_locked()

    async def stop_receiver(self) -> None:
        async with self.lock:
            await self._stop_locked()

    def sync_runtime_receiver(self, runtime: object | None = None) -> None:
        owner = runtime or self._runtime
        receiver = getattr(owner, "receiver", None) if owner is not None else None
        self.receiver = receiver

    async def configure(self, *, enabled: bool) -> None:
        async with self.lock:
            if not self._accepting_ingress:
                return
            self._last_enabled = bool(enabled)
            gate = self.gate_provider()
            if gate is not None:
                gate.set_enabled(self._last_enabled)
            await self._reconcile_locked()

    async def stop(self) -> None:
        async with self.lock:
            await self._stop_locked()

    async def _stop_locked(self) -> None:
        runtime = self._runtime
        if runtime is not None:
            with contextlib.suppress(Exception):
                await runtime.stop(strict_runtime_errors=False)
            self.sync_runtime_receiver(runtime)
        elif self._receiver is not None:
            with contextlib.suppress(Exception):
                stop = getattr(self._receiver, "stop", None)
                if callable(stop):
                    stop()
            self._receiver = None
        gate = self.gate_provider()
        if gate is not None:
            gate.set_receiver_active(False)

    async def _start_locked(self) -> object | None:
        if self._receiver is not None:
            return self._receiver
        state = self.state_provider()
        if state is None:
            self._set_receiver_active(False)
            return None
        runtime = self.get_runtime()
        if runtime is None:
            self._set_receiver_active(False)
            return None
        try:
            receiver = await runtime.start()
        except OSError as exc:
            self._set_receiver_active(False)
            self.error_sink(
                f"VRChat mic sync receiver unavailable on {self.host}:{self.port}: {exc}"
            )
            return None
        if not self._accepting_ingress:
            await self._close_locked()
            return None
        self._receiver = receiver
        self._set_receiver_active(True)
        gate = self.gate_provider()
        if gate is not None:
            gate.reset()
        return receiver

    async def _reconcile_locked(self) -> None:
        should_run = self._control_active or bool(self._last_enabled)
        if should_run:
            await self._start_locked()
            return
        await self._stop_locked()

    def _set_receiver_active(self, active: bool) -> None:
        gate = self.gate_provider()
        if gate is not None:
            gate.set_receiver_active(active)

    async def close(self) -> None:
        self.stop_ingress()
        async with self.lock:
            await self._close_locked()

    async def _close_locked(self) -> None:
        self._last_enabled = False
        self._control_active = False
        gate = self.gate_provider()
        if gate is not None:
            gate.set_enabled(False)
            gate.set_receiver_active(False)

        runtime = self._runtime
        if runtime is None:
            receiver = self._receiver
            if receiver is None:
                return
            stop = getattr(receiver, "stop", None)
            if callable(stop):
                stop()
            self._receiver = None
            return

        try:
            await runtime.close()
        except Exception:
            self._receiver = getattr(runtime, "receiver", self._receiver)
            raise

        if self._runtime is runtime:
            self._runtime = None
        self._receiver = None


__all__ = ["VrcMicAudioGatePort", "VrcMicSyncOwner"]
