from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections.abc import Callable, Mapping
from typing import Any, Protocol, TypeVar

from puripuly_heart.core.osc.receiver import (
    VRC_OSC_RECEIVER_HOST,
    VRC_OSC_RECEIVER_PORT,
    VrcMicState,
    VrcOscReceiver,
)

logger = logging.getLogger(__name__)

_ReceiverT = TypeVar("_ReceiverT", bound="OscReceiverProtocol")
ReceiverDiagnosticsSink = Callable[[str, Mapping[str, object]], None]


class OscReceiverProtocol(Protocol):
    async def start(self) -> None: ...

    def stop(self) -> object: ...


OscReceiverFactory = Callable[[], _ReceiverT]
VrcOscReceiverFactory = Callable[..., OscReceiverProtocol]
ReceiverRuntimeStateChanged = Callable[[object], None]


def _raise_cleanup_failures(message: str, failures: list[Exception]) -> None:
    if not failures:
        return
    if len(failures) == 1:
        raise failures[0]
    raise ExceptionGroup(message, failures)


class OscReceiverRuntime:
    """Owns a single OSC receiver socket lifecycle."""

    resource_fields = ("receiver", "_generation")
    stop_ingress = "stop receiver before runtime shutdown"
    shutdown_policy = "close socket and clear receiver ownership"
    late_callback_rule = "late packets dropped after stop"

    def __init__(
        self,
        *,
        receiver_factory: OscReceiverFactory[_ReceiverT],
        diagnostics_sink: ReceiverDiagnosticsSink | None = None,
        state_changed: ReceiverRuntimeStateChanged | None = None,
    ) -> None:
        self._receiver_factory = receiver_factory
        self._diagnostics_sink = diagnostics_sink
        self._state_changed = state_changed
        self._receiver: _ReceiverT | None = None
        self._generation = 0
        self._closing = False
        self._closed = False
        self._lock = asyncio.Lock()

    @property
    def owner_name(self) -> str:
        return "OscReceiverRuntime"

    @property
    def receiver(self) -> _ReceiverT | None:
        return self._receiver

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def is_closing(self) -> bool:
        return self._closing

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "resource_fields": self.resource_fields,
            "stop_ingress": self.stop_ingress,
            "shutdown_policy": self.shutdown_policy,
            "late_callback_rule": self.late_callback_rule,
        }

    async def start(self) -> _ReceiverT:
        async with self._lock:
            if self._closing or self._closed:
                state = "closing" if self._closing else "closed"
                raise RuntimeError(f"{self.owner_name} is {state} to new receiver work")
            if self._receiver is not None:
                return self._receiver

            receiver = self._receiver_factory()
            self._generation += 1
            try:
                await receiver.start()
            except Exception as exc:
                self._emit(
                    "osc_receiver_start_failed",
                    {"error_type": type(exc).__name__},
                )
                raise

            self._receiver = receiver
            self._notify_state_changed()
            return receiver

    async def stop(self, *, strict_runtime_errors: bool = False) -> None:
        async with self._lock:
            self._generation += 1
            receiver = self._receiver
            if receiver is None:
                self._notify_state_changed()
                return
            try:
                await _call_stop(receiver)
            except Exception as exc:
                self._emit(
                    "osc_receiver_stop_failed",
                    {"error_type": type(exc).__name__},
                )
                self._notify_state_changed()
                if strict_runtime_errors:
                    raise
                return
            if self._receiver is receiver:
                self._receiver = None
            self._notify_state_changed()

    async def close(self) -> None:
        if self._closed and self._receiver is None:
            return
        self._closing = True
        self._closed = True
        self._notify_state_changed()
        try:
            await self.stop(strict_runtime_errors=True)
        finally:
            self._closing = False
            self._notify_state_changed()

    def _emit(self, event: str, metadata: Mapping[str, object]) -> None:
        if self._diagnostics_sink is not None:
            with contextlib.suppress(Exception):
                self._diagnostics_sink(event, metadata)

    def _notify_state_changed(self) -> None:
        if self._state_changed is not None:
            with contextlib.suppress(Exception):
                self._state_changed(self)


OscReceiverRuntimeFactory = Callable[..., OscReceiverRuntime]


class VrcMicReceiverRuntime:
    """Owns the VRChat mic OSC receiver socket and delayed mute task."""

    resource_fields = ("receiver", "_mute_task", "_mute_tasks", "_generation")
    stop_ingress = "stop receiver before runtime shutdown"
    shutdown_policy = "close socket, cancel/gather receiver task"
    late_callback_rule = "late packets dropped after stop"

    def __init__(
        self,
        *,
        state: VrcMicState,
        host: str = VRC_OSC_RECEIVER_HOST,
        port: int = VRC_OSC_RECEIVER_PORT,
        mute_delay_s: float = 0.4,
        receiver_factory: VrcOscReceiverFactory | None = None,
        osc_runtime_factory: OscReceiverRuntimeFactory | None = None,
        cancel_timeout_s: float = 2.0,
        diagnostics_sink: ReceiverDiagnosticsSink | None = None,
        state_changed: ReceiverRuntimeStateChanged | None = None,
        control_packet_handler: Callable[[str, tuple[Any, ...]], object] | None = None,
        avatar_change_handler: Callable[[tuple[Any, ...]], object] | None = None,
        packet_handler: Callable[[str, tuple[Any, ...]], object] | None = None,
    ) -> None:
        self._state = state
        self._host = host
        self._port = port
        self._mute_delay_s = mute_delay_s
        self._receiver_factory = receiver_factory or _default_vrc_receiver_factory
        self._cancel_timeout_s = max(0.0, float(cancel_timeout_s))
        self._diagnostics_sink = diagnostics_sink
        self._state_changed = state_changed
        self._control_packet_handler = control_packet_handler
        self._avatar_change_handler = avatar_change_handler
        self._packet_handler = packet_handler
        self._mute_task: asyncio.Task[None] | None = None
        self._mute_tasks: set[asyncio.Task[None]] = set()
        self._generation = 0
        self._active_generation: int | None = None
        self._pending_receiver_generation: int | None = None
        self._closing = False
        self._closed = False
        self._lock = asyncio.Lock()
        self._osc_runtime = (osc_runtime_factory or OscReceiverRuntime)(
            receiver_factory=self._build_receiver_for_osc_runtime,
            diagnostics_sink=None,
            state_changed=self._on_osc_runtime_state_changed,
        )

    @property
    def owner_name(self) -> str:
        return "VrcMicReceiverRuntime"

    @property
    def receiver(self) -> OscReceiverProtocol | None:
        return self._osc_runtime.receiver

    @property
    def effective_port(self) -> int:
        receiver = self.receiver
        value = getattr(receiver, "effective_port", self._port)
        return int(value)

    @property
    def mute_task(self) -> asyncio.Task[None] | None:
        return self._mute_task

    def configure_endpoint(self, host: str, port: int) -> None:
        if self.receiver is not None:
            raise RuntimeError("cannot change OSC receiver endpoint while running")
        if not host:
            raise ValueError("OSC receiver host must be non-empty")
        if not 0 <= int(port) <= 65535:
            raise ValueError("OSC receiver port must be in 0..65535")
        self._host = host
        self._port = int(port)

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def is_closing(self) -> bool:
        return self._closing

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "resource_fields": self.resource_fields,
            "stop_ingress": self.stop_ingress,
            "shutdown_policy": self.shutdown_policy,
            "late_callback_rule": self.late_callback_rule,
        }

    async def start(self) -> OscReceiverProtocol:
        async with self._lock:
            if self._closing or self._closed:
                state = "closing" if self._closing else "closed"
                raise RuntimeError(f"{self.owner_name} is {state} to new receiver work")
            if self.receiver is not None and self._active_generation is not None:
                return self.receiver

            self._generation += 1
            generation = self._generation
            self._active_generation = generation
            self._pending_receiver_generation = generation
            try:
                receiver = await self._osc_runtime.start()
            except Exception as exc:
                self._active_generation = None
                self._emit(
                    "vrc_mic_receiver_start_failed",
                    {
                        "host": self._host,
                        "port": self._port,
                        "error_type": type(exc).__name__,
                    },
                )
                self._notify_state_changed()
                raise
            finally:
                self._pending_receiver_generation = None

            self._notify_state_changed()
            return receiver

    async def stop(self, *, strict_runtime_errors: bool = False) -> None:
        async with self._lock:
            await self._stop_locked(strict_runtime_errors=strict_runtime_errors, terminal=False)

    async def close(self) -> None:
        if self._closed and self.receiver is None and not self._mute_tasks:
            return
        self._closing = True
        self._closed = True
        self._notify_state_changed()
        try:
            async with self._lock:
                await self._stop_locked(strict_runtime_errors=True, terminal=True)
        finally:
            self._closing = False
            self._notify_state_changed()

    def handle_mute_packet(self, is_muted: bool, *, generation: int | None = None) -> bool:
        expected_generation = self._packet_generation(generation)
        if expected_generation is None:
            return False

        previous_task = self._mute_task
        if previous_task is not None and not previous_task.done():
            previous_task.cancel()
        try:
            task = asyncio.create_task(
                self._apply_mute_state(bool(is_muted), expected_generation),
                name=f"{self.owner_name}:mute-state",
            )
        except RuntimeError:
            self._emit(
                "vrc_mic_receiver_task_schedule_failed",
                {"host": self._host, "port": self._port},
            )
            return False
        self._mute_task = task
        self._mute_tasks.add(task)
        task.add_done_callback(self._on_mute_task_done)
        self._notify_state_changed()
        return True

    def handle_control_packet(
        self,
        address: str,
        values: tuple[Any, ...],
        *,
        generation: int | None = None,
    ) -> object:
        if self._packet_generation(generation) is None:
            return False
        handler = self._control_packet_handler
        return handler(address, values) if handler is not None else False

    def handle_avatar_change_packet(
        self,
        values: tuple[Any, ...],
        *,
        generation: int | None = None,
    ) -> object:
        if self._packet_generation(generation) is None:
            return False
        handler = self._avatar_change_handler
        return handler(values) if handler is not None else False

    def handle_packet(
        self,
        address: str,
        values: tuple[Any, ...],
        *,
        generation: int | None = None,
    ) -> object:
        if self._packet_generation(generation) is None:
            return False
        handler = self._packet_handler
        return handler(address, values) if handler is not None else False

    def _packet_generation(self, generation: int | None) -> int | None:
        expected_generation = self._active_generation if generation is None else generation
        if expected_generation is not None and self.is_current_generation(expected_generation):
            return expected_generation
        self._emit(
            "vrc_mic_receiver_late_packet_dropped",
            {"host": self._host, "port": self._port},
        )
        return None

    def is_current_generation(self, generation: int) -> bool:
        return (
            not self._closing
            and not self._closed
            and self._active_generation == generation
            and self._generation == generation
        )

    def _build_receiver(self, generation: int) -> OscReceiverProtocol:
        return self._receiver_factory(
            state=self._state,
            host=self._host,
            port=self._port,
            mute_delay_s=self._mute_delay_s,
            mute_packet_handler=lambda is_muted, *, _generation=generation: self.handle_mute_packet(
                is_muted,
                generation=_generation,
            ),
            control_packet_handler=(
                (
                    lambda address, values, *, _generation=generation: self.handle_control_packet(
                        address,
                        values,
                        generation=_generation,
                    )
                )
                if self._control_packet_handler is not None
                else None
            ),
            avatar_change_handler=(
                (
                    lambda values, *, _generation=generation: self.handle_avatar_change_packet(
                        values,
                        generation=_generation,
                    )
                )
                if self._avatar_change_handler is not None
                else None
            ),
            packet_handler=(
                (
                    lambda address, values, *, _generation=generation: self.handle_packet(
                        address,
                        values,
                        generation=_generation,
                    )
                )
                if self._packet_handler is not None
                else None
            ),
        )

    def _build_receiver_for_osc_runtime(self) -> OscReceiverProtocol:
        generation = self._pending_receiver_generation
        if generation is None:
            generation = self._generation
        return self._build_receiver(generation)

    async def _stop_locked(
        self,
        *,
        strict_runtime_errors: bool,
        terminal: bool,
    ) -> None:
        self._generation += 1
        self._active_generation = None
        mute_tasks = tuple(self._mute_tasks)
        self._mute_task = None
        self._notify_state_changed()

        stop_failure: Exception | None = None
        if self._osc_runtime.receiver is not None or terminal:
            try:
                if terminal:
                    await self._osc_runtime.close()
                else:
                    await self._osc_runtime.stop(strict_runtime_errors=strict_runtime_errors)
            except Exception as exc:
                stop_failure = exc
                self._emit(
                    "vrc_mic_receiver_stop_failed",
                    {
                        "host": self._host,
                        "port": self._port,
                        "error_type": type(exc).__name__,
                    },
                )

        cancel_failures = await self._cancel_tasks_bounded(mute_tasks, terminal=terminal)
        self._notify_state_changed()

        cleanup_failures: list[Exception] = []
        if stop_failure is not None and strict_runtime_errors:
            cleanup_failures.append(stop_failure)
        if strict_runtime_errors:
            cleanup_failures.extend(cancel_failures)
        _raise_cleanup_failures(
            f"{self.owner_name} stop cleanup failed",
            cleanup_failures,
        )

    def _on_osc_runtime_state_changed(self, _runtime: object) -> None:
        self._notify_state_changed()

    async def _apply_mute_state(self, is_muted: bool, generation: int) -> None:
        try:
            if is_muted:
                await asyncio.sleep(self._mute_delay_s)
            if not self.is_current_generation(generation):
                self._emit(
                    "vrc_mic_receiver_late_mute_update_dropped",
                    {"host": self._host, "port": self._port},
                )
                return
            if self._state.update(is_muted):
                logger.info("[OSC Receiver] VRChat mic muted state applied: %s", is_muted)
        except asyncio.CancelledError:
            raise

    async def _cancel_tasks_bounded(
        self,
        tasks: tuple[asyncio.Task[Any], ...],
        *,
        terminal: bool,
    ) -> list[Exception]:
        current_task = asyncio.current_task()
        pending_tasks = tuple(task for task in tasks if task is not current_task)
        if not pending_tasks:
            return []
        for task in pending_tasks:
            if not task.done():
                task.cancel()
        done, pending = await asyncio.wait(pending_tasks, timeout=self._cancel_timeout_s)
        for completed in done:
            _observe_task_exception(completed)
            self._mute_tasks.discard(completed)  # type: ignore[arg-type]
        if pending:
            for task in pending:
                task.cancel()
            self._emit(
                "vrc_mic_receiver_task_cancel_timeout",
                {"host": self._host, "port": self._port},
            )
            if terminal:
                return [
                    TimeoutError(f"{self.owner_name} timed out cancelling mute task during close")
                ]
        return []

    def _on_mute_task_done(self, task: asyncio.Task[None]) -> None:
        self._mute_tasks.discard(task)
        _observe_task_exception(task)
        if self._mute_task is task:
            self._mute_task = None
            self._notify_state_changed()

    def _emit(self, event: str, metadata: Mapping[str, object]) -> None:
        if self._diagnostics_sink is not None:
            with contextlib.suppress(Exception):
                self._diagnostics_sink(event, metadata)

    def _notify_state_changed(self) -> None:
        if self._state_changed is not None:
            with contextlib.suppress(Exception):
                self._state_changed(self)


def _default_vrc_receiver_factory(**kwargs: object) -> VrcOscReceiver:
    return VrcOscReceiver(**kwargs)  # type: ignore[arg-type]


async def _call_stop(receiver: object) -> None:
    stop = getattr(receiver, "stop", None)
    if not callable(stop):
        return
    result = stop()
    if inspect.isawaitable(result):
        await result


def _observe_task_exception(task: asyncio.Task[Any]) -> None:
    if task.cancelled():
        return
    try:
        task.exception()
    except asyncio.CancelledError:
        pass


__all__ = ["OscReceiverRuntime", "VrcMicReceiverRuntime"]
