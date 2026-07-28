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
            )
        return self._runtime

    def sync_runtime_receiver(self, runtime: object | None = None) -> None:
        owner = runtime or self._runtime
        self._receiver = getattr(owner, "receiver", None) if owner is not None else None

    async def configure(self, *, enabled: bool) -> None:
        async with self.lock:
            if not self._accepting_ingress:
                return
            self._last_enabled = enabled
            gate = self.gate_provider()
            if gate is not None:
                gate.set_enabled(enabled)

            if not enabled:
                await self._stop_locked()
                return

            state = self.state_provider()
            if self._receiver is not None or state is None:
                if gate is not None:
                    gate.set_receiver_active(self._receiver is not None)
                return

            runtime = self.get_runtime()
            if runtime is None:
                if gate is not None:
                    gate.set_receiver_active(False)
                return
            try:
                receiver = await runtime.start()
            except OSError as exc:
                if gate is not None:
                    gate.set_receiver_active(False)
                self.error_sink(
                    f"VRChat mic sync receiver unavailable on {self.host}:{self.port}: {exc}"
                )
                return

            self._receiver = receiver
            if gate is not None:
                gate.set_receiver_active(True)
                gate.reset()

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

    async def close(self) -> None:
        async with self.lock:
            self.stop_ingress()
            await self._close_locked()

    async def _close_locked(self) -> None:
        self._last_enabled = False
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
