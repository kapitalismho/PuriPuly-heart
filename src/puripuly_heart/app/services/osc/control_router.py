from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from puripuly_heart.app.ports.osc_control import (
    ASR_IDS,
    LANGUAGE_IDS,
    SECONDARY_LANGUAGE_IDS,
    TRANSLATION_CONNECTION_BY_MODEL_ID,
    TRANSLATION_MODEL_IDS,
    OscControlApplicationPort,
    OscControlCodecError,
    OscControlMessage,
    decode_control_message,
)
from puripuly_heart.app.ports.ui_models import OscControlPresentationName
from puripuly_heart.core.diagnostics import deserialize_diagnostics, emit_failure, exception_diagnostics
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task

logger = logging.getLogger(__name__)

_COALESCED_CONTROLS = frozenset(
    {
        "PuriPuly_SelfSrcLang",
        "PuriPuly_SelfDstLang",
        "PuriPuly_SelfDstLang2",
        "PuriPuly_PeerSrcLang",
        "PuriPuly_PeerDstLang",
        "PuriPuly_SelfASR",
        "PuriPuly_PeerASR",
        "PuriPuly_Translator",
    }
)

RUNTIME_BOOL_CONTROLS = frozenset(
    {"PuriPuly_Talk", "PuriPuly_Listen", "PuriPuly_Trans", "PuriPuly_Captions"}
)


@dataclass(frozen=True, slots=True)
class OscDispatchResult:
    applied: bool
    parameter: str
    superseded: bool = False
    error: str | None = None
    diagnostics: ErrorDiagnostics | None = None
    operation_id: str | None = None
    request_id: str | None = None


@dataclass(slots=True)
class _PendingCommand:
    message: OscControlMessage
    future: asyncio.Future[OscDispatchResult]
    sequence: int
    generation: int


class OscControlRouter:
    def __init__(
        self,
        application: OscControlApplicationPort,
        *,
        language_state_provider: Callable[[], tuple[str, str, str, str]] | None = None,
        echo_suppression_provider: Callable[[OscControlMessage], bool] | None = None,
        canonical_state_republisher: Callable[[], object] | None = None,
        canonical_state_full_republisher: Callable[[], object] | None = None,
        canonical_state_projector: Callable[[OscControlPresentationName], object] | None = None,
        error_sink: Callable[[str], None] | None = None,
        command_observer: Callable[[OscControlMessage, int], None] | None = None,
        settlement_observer: (
            Callable[[OscControlMessage, int, OscDispatchResult | None], None] | None
        ) = None,
        application_orders_runtime_controls: bool = False,
        application_orders_mutations: bool = False,
    ) -> None:
        self._application = application
        self._error_sink = error_sink
        self._language_state_provider = language_state_provider
        self._echo_suppression_provider = echo_suppression_provider
        self._canonical_state_republisher = canonical_state_republisher
        self._canonical_state_full_republisher = canonical_state_full_republisher
        self._canonical_state_projector = canonical_state_projector
        self._command_observer = command_observer
        self._settlement_observer = settlement_observer
        self._application_orders_runtime_controls = application_orders_runtime_controls
        self._application_orders_mutations = application_orders_mutations
        self._language_values = (
            language_state_provider()
            if language_state_provider is not None
            else ("ko", "en", "en", "ko")
        )
        self._serial_lock = asyncio.Lock()
        self._pending_lock = asyncio.Lock()
        self._pending: dict[str, _PendingCommand] = {}
        self._sequence = 0
        self.failures: tuple[ErrorDiagnostics, ...] = ()
        self._worker_task: asyncio.Task[None] | None = None
        self._packet_sequence = 0
        self._scope = LifecycleScope("OscControlRouter")
        self._closed = False
        self._ingress_enabled = True
        self._generation = 0
        self._admission_sequence = 0
        self._unsettled_commands: dict[int, OscControlMessage] = {}
        self._active_invocation_tasks: set[asyncio.Task[Any]] = set()
        self._invocation_messages: dict[asyncio.Task[Any], OscControlMessage] = {}
        self._invocation_generations: dict[asyncio.Task[Any], int] = {}

    def set_ingress_enabled(self, enabled: bool) -> None:
        self._generation += 1
        self._ingress_enabled = bool(enabled)

    async def suspend_ingress(self, *, invalidate: bool = False) -> None:
        if self._application_orders_mutations and not invalidate:
            self._ingress_enabled = False
        elif self._application_orders_mutations:
            self.set_ingress_enabled(False)
        else:
            self.set_ingress_enabled(False)
            await self._wait_for_active_invocations()

    def resume_ingress(self) -> None:
        if self._application_orders_mutations:
            self._ingress_enabled = True
        else:
            self.set_ingress_enabled(True)

    async def dispatch(
        self,
        message: OscControlMessage,
        *,
        admission_generation: int | None = None,
    ) -> OscDispatchResult:
        if self._closed:
            return OscDispatchResult(False, message.name, error="router_closed")
        if not self._ingress_enabled or (
            admission_generation is not None and admission_generation != self._generation
        ):
            return OscDispatchResult(False, message.name, error="router_disabled")
        if self._echo_suppression_provider is not None and self._echo_suppression_provider(message):
            return OscDispatchResult(False, message.name, error="echo_suppressed")
        generation = self._generation
        self._admission_sequence += 1
        admission = self._admission_sequence
        self._unsettled_commands[admission] = message
        if self._command_observer is not None:
            self._command_observer(message, admission)
        result: OscDispatchResult | None = None
        try:
            if message.name in _COALESCED_CONTROLS:
                result = await self._enqueue_coalesced(message, generation)
            else:
                result = await self._apply_serialized(message, generation=generation)
            return result
        finally:
            self._unsettled_commands.pop(admission, None)
            if self._settlement_observer is not None:
                self._settlement_observer(message, admission, result)

    async def dispatch_packet(self, address: str, *args: Any) -> OscDispatchResult:
        message = decode_control_message(address, *args)
        return await self.dispatch(message)

    def handle_packet(self, address: str, *args: Any) -> bool:
        if self._closed or not self._ingress_enabled:
            return False
        try:
            message = decode_control_message(address, *args)
        except OscControlCodecError as exc:
            self._republish_canonical_state()
            self._report_error(str(exc))
            return False
        try:
            self._packet_sequence += 1
            task = start_lifecycle_task(
                self._scope,
                self.dispatch(message, admission_generation=self._generation),
                name=f"packet-{self._packet_sequence}",
            )
        except RuntimeError:
            self._report_error("OSC control packet received without a running event loop")
            return False
        task.add_done_callback(self._observe_task)
        return True

    async def close(self) -> None:
        self._closed = True
        self.set_ingress_enabled(False)
        self._cancel_active_invocations()
        await self._wait_for_active_invocations()
        async with self._pending_lock:
            pending = tuple(self._pending.values())
            self._pending.clear()
        for item in pending:
            if not item.future.done():
                item.future.set_result(
                    OscDispatchResult(False, item.message.name, error="router_closed")
                )
        await self._scope.close()
        self._worker_task = None

    async def _enqueue_coalesced(
        self,
        message: OscControlMessage,
        generation: int,
    ) -> OscDispatchResult:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[OscDispatchResult] = loop.create_future()
        async with self._pending_lock:
            if not self._is_generation_current(generation):
                return OscDispatchResult(False, message.name, error="router_disabled")
            self._sequence += 1
            previous = self._pending.get(message.name)
            if previous is not None and not previous.future.done():
                previous.future.set_result(
                    OscDispatchResult(
                        False,
                        previous.message.name,
                        superseded=True,
                    )
                )
            self._pending[message.name] = _PendingCommand(
                message,
                future,
                self._sequence,
                generation,
            )
            if self._worker_task is None or self._worker_task.done():
                self._worker_task = start_lifecycle_task(
                    self._scope,
                    self._drain_pending(),
                    name=f"coalesced-{self._sequence}",
                )
        return await future

    async def _drain_pending(self) -> None:
        current: _PendingCommand | None = None
        try:
            while True:
                async with self._pending_lock:
                    if not self._pending:
                        self._worker_task = None
                        return
                    name, current = min(
                        self._pending.items(),
                        key=lambda item: item[1].sequence,
                    )
                    del self._pending[name]
                if current.future.cancelled():
                    current = None
                    continue
                result = await self._apply_serialized(
                    current.message,
                    generation=current.generation,
                    cancelled=current.future.cancelled,
                )
                if not current.future.done():
                    current.future.set_result(result)
                current = None
        except asyncio.CancelledError:
            if current is not None and not current.future.done():
                current.future.set_result(
                    OscDispatchResult(False, current.message.name, error="router_closed")
                )
            raise

    async def _apply_serialized(
        self,
        message: OscControlMessage,
        *,
        generation: int,
        cancelled: Callable[[], bool] | None = None,
    ) -> OscDispatchResult:
        if self._application_orders_runtime_controls and message.name in RUNTIME_BOOL_CONTROLS:
            return await self._apply_command(message, generation=generation, cancelled=cancelled)
        async with self._serial_lock:
            return await self._apply_command(message, generation=generation, cancelled=cancelled)

    async def _apply_command(
        self,
        message: OscControlMessage,
        *,
        generation: int,
        cancelled: Callable[[], bool] | None = None,
    ) -> OscDispatchResult:
        if not self._is_generation_current(generation):
            return OscDispatchResult(False, message.name, error=self._generation_error())
        if cancelled is not None and cancelled():
            return OscDispatchResult(False, message.name, error="cancelled")
        invocation_task: asyncio.Task[Any] | None = None
        try:
            invocation_task = asyncio.current_task()
            if invocation_task is not None:
                self._active_invocation_tasks.add(invocation_task)
                self._invocation_messages[invocation_task] = message
                self._invocation_generations[invocation_task] = generation
            try:
                result = await self._invoke(message)
            finally:
                if invocation_task is not None:
                    self._active_invocation_tasks.discard(invocation_task)
                    self._invocation_messages.pop(invocation_task, None)
                    self._invocation_generations.pop(invocation_task, None)
        except Exception as exc:
            diagnostics = exception_diagnostics(
                exc, component="osc_control", operation="control_dispatch", code="osc_dispatch_failed"
            )
            emit_failure(logger, diagnostics)
            self.failures = (*self.failures[-7:], diagnostics)
            self._publish_canonical_delta()
            self._report_error(f"{message.name}: {type(exc).__name__}")
            return OscDispatchResult(
                False, message.name, error=type(exc).__name__, diagnostics=diagnostics
            )
        if self._closed:
            return OscDispatchResult(False, message.name, error="router_closed")
        if (cancelled is not None and cancelled()) or (
            invocation_task is not None and invocation_task.cancelling()
        ):
            return OscDispatchResult(False, message.name, error="cancelled")
        if not self._is_generation_current(generation):
            return OscDispatchResult(False, message.name, error=self._generation_error())
        if _application_result_rejected(result):
            self._publish_canonical_delta()
            if (
                _application_result_changed_canonical_state(result)
                and self._is_generation_current(generation)
                and not (cancelled is not None and cancelled())
            ):
                self._project_canonical_state(cast(OscControlPresentationName, message.name))
            self._report_error(f"{message.name}: application_rejected")
            diagnostics = getattr(result, "failure_diagnostics", None)
            if not isinstance(diagnostics, ErrorDiagnostics):
                value = getattr(result, "diagnostics", None)
                diagnostics = deserialize_diagnostics(value) if isinstance(value, Mapping) else None
            return OscDispatchResult(
                False, message.name, error="application_rejected", diagnostics=diagnostics,
                operation_id=getattr(result, "operation_id", None),
                request_id=getattr(result, "request_id", None),
            )
        self._publish_canonical_delta()
        if (cancelled is not None and cancelled()) or (
            invocation_task is not None and invocation_task.cancelling()
        ):
            return OscDispatchResult(False, message.name, error="cancelled")
        if not self._is_generation_current(generation):
            return OscDispatchResult(False, message.name, error=self._generation_error())
        self._project_canonical_state(cast(OscControlPresentationName, message.name))
        return OscDispatchResult(True, message.name, error=None if result is None else None)

    async def _invoke(self, message: OscControlMessage) -> object:
        app = self._application
        if message.name == "PuriPuly_Talk":
            return await app.set_self_capture(message.value)  # type: ignore[arg-type]
        if message.name == "PuriPuly_Listen":
            return await app.set_peer_capture(message.value)  # type: ignore[arg-type]
        if message.name == "PuriPuly_Trans":
            return await app.set_translation(message.value)  # type: ignore[arg-type]
        if message.name == "PuriPuly_Captions":
            return await app.set_captions(message.value)  # type: ignore[arg-type]
        if message.name == "PuriPuly_PeerAuto":
            return await app.set_peer_auto_detect(message.value)  # type: ignore[arg-type]
        if message.name == "PuriPuly_MuteSync":
            return await app.set_mute_sync(message.value)  # type: ignore[arg-type]
        if message.name == "PuriPuly_ChatboxSource":
            return await app.set_chatbox_source(message.value)  # type: ignore[arg-type]
        if message.name == "PuriPuly_SelfDstLang2":
            return await app.set_secondary_target_language(
                SECONDARY_LANGUAGE_IDS[message.value]  # type: ignore[index]
            )
        if message.name in {
            "PuriPuly_SelfSrcLang",
            "PuriPuly_SelfDstLang",
            "PuriPuly_PeerSrcLang",
            "PuriPuly_PeerDstLang",
        }:
            values = list(
                self._language_state_provider()
                if self._language_state_provider is not None
                else self._language_values
            )
            index = {
                "PuriPuly_SelfSrcLang": 0,
                "PuriPuly_SelfDstLang": 1,
                "PuriPuly_PeerSrcLang": 2,
                "PuriPuly_PeerDstLang": 3,
            }[message.name]
            values[index] = LANGUAGE_IDS[message.value]  # type: ignore[index]
            result = await app.set_languages(
                self_source=values[0],
                self_target=values[1],
                peer_source=values[2],
                peer_target=values[3],
            )
            self._language_values = tuple(values)  # type: ignore[assignment]
            return result
        if message.name == "PuriPuly_SelfASR":
            return await app.set_self_asr(ASR_IDS[message.value])  # type: ignore[index]
        if message.name == "PuriPuly_PeerASR":
            return await app.set_peer_asr(ASR_IDS[message.value])  # type: ignore[index]
        if message.name == "PuriPuly_Translator":
            value = int(message.value)
            return await app.set_translation_model(
                TRANSLATION_MODEL_IDS[value],
                TRANSLATION_CONNECTION_BY_MODEL_ID.get(value),
            )
        raise RuntimeError(f"unhandled PuriPuly OSC parameter: {message.name}")

    def _observe_task(self, task: asyncio.Task[OscDispatchResult]) -> None:
        with contextlib.suppress(asyncio.CancelledError):
            result = task.result()
            if result.error is not None:
                self._report_error(f"{result.parameter}: {result.error}")

    def _report_error(self, message: str) -> None:
        if self._error_sink is not None:
            with contextlib.suppress(Exception):
                self._error_sink(message)
        else:
            with contextlib.suppress(Exception):
                logger.warning("[OSC Control] %s", message)

    def _republish_canonical_state(self) -> None:
        callback = self._canonical_state_full_republisher or self._canonical_state_republisher
        if callback is None:
            return
        try:
            callback()
        except Exception as exc:
            self._retain_projection_failure(exc)

    def _publish_canonical_delta(self) -> None:
        callback = self._canonical_state_republisher
        if callback is None:
            return
        try:
            callback()
        except Exception as exc:
            self._retain_projection_failure(exc)

    def _project_canonical_state(self, control: OscControlPresentationName) -> None:
        callback = self._canonical_state_projector
        if callback is None:
            return
        try:
            callback(control)
        except Exception as exc:
            self._retain_projection_failure(exc)

    def _retain_projection_failure(self, exc: Exception) -> None:
        diagnostics = exception_diagnostics(
            exc, component="osc_control", operation="control_dispatch", code="output_observer_failed"
        )
        self.failures = (*self.failures[-7:], diagnostics)
        emit_failure(logger, diagnostics, severity="warning", disposition="degraded")

    def current_invocation_guard(self) -> Callable[[], bool]:
        task = asyncio.current_task()
        generation = self._invocation_generations.get(task, self._generation)
        return lambda: self._is_generation_current(generation) and (
            task is None or not task.cancelling()
        )

    def _is_generation_current(self, generation: int) -> bool:
        return (
            not self._closed
            and generation == self._generation
            and (self._ingress_enabled or self._application_orders_mutations)
        )

    def _generation_error(self) -> str:
        return "router_closed" if self._closed else "router_disabled"

    @property
    def pending_controls(self) -> tuple[OscControlMessage, ...]:
        return tuple(self._unsettled_commands.values()) + tuple(self._invocation_messages.values())

    def _cancel_active_invocations(self) -> None:
        current = asyncio.current_task()
        for task in tuple(self._active_invocation_tasks):
            if task is not current and not task.done():
                task.cancel()

    async def _wait_for_active_invocations(self) -> None:
        current = asyncio.current_task()
        tasks = tuple(
            task
            for task in self._active_invocation_tasks
            if task is not current and not task.done()
        )
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def _application_result_rejected(result: object) -> bool:
    if result is False:
        return True
    applied = getattr(result, "applied", None)
    if applied is False:
        return True
    status = getattr(result, "status", None)
    status = getattr(status, "value", status)
    if not isinstance(status, str):
        return False
    return "degraded" in status or "failed" in status


def _application_result_changed_canonical_state(result: object) -> bool:
    return getattr(result, "canonical_state_changed", False) is True


__all__ = ["OscControlRouter", "OscDispatchResult"]
