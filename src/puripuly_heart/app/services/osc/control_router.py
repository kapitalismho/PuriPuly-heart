from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from puripuly_heart.app.ports.osc_control import (
    ASR_IDS,
    FALLBACK_IDS,
    LANGUAGE_IDS,
    TRANSLATION_MODEL_IDS,
    OscControlApplicationPort,
    OscControlCodecError,
    OscControlMessage,
    decode_control_message,
)
from puripuly_heart.app.ports.ui_models import OscControlPresentationName
from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task

logger = logging.getLogger(__name__)

_COALESCED_CONTROLS = frozenset(
    {
        "PuriPuly_SelfSrcLang",
        "PuriPuly_SelfDstLang",
        "PuriPuly_PeerSrcLang",
        "PuriPuly_PeerDstLang",
        "PuriPuly_SelfASR",
        "PuriPuly_PeerASR",
        "PuriPuly_Translator",
        "PuriPuly_Fallback",
    }
)


@dataclass(frozen=True, slots=True)
class OscDispatchResult:
    applied: bool
    parameter: str
    superseded: bool = False
    error: str | None = None


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
    ) -> None:
        self._application = application
        self._error_sink = error_sink
        self._language_state_provider = language_state_provider
        self._echo_suppression_provider = echo_suppression_provider
        self._canonical_state_republisher = canonical_state_republisher
        self._canonical_state_full_republisher = canonical_state_full_republisher
        self._canonical_state_projector = canonical_state_projector
        self._language_values = (
            language_state_provider()
            if language_state_provider is not None
            else ("ko", "en", "en", "ko")
        )
        self._serial_lock = asyncio.Lock()
        self._pending_lock = asyncio.Lock()
        self._pending: dict[str, _PendingCommand] = {}
        self._sequence = 0
        self._worker_task: asyncio.Task[None] | None = None
        self._packet_sequence = 0
        self._scope = LifecycleScope("OscControlRouter")
        self._closed = False
        self._ingress_enabled = True
        self._generation = 0
        self._unsettled_by_parameter: dict[str, int] = {}
        self._active_invocation_tasks: set[asyncio.Task[Any]] = set()

    def set_ingress_enabled(self, enabled: bool) -> None:
        self._generation += 1
        self._ingress_enabled = bool(enabled)

    async def suspend_ingress(self) -> None:
        self.set_ingress_enabled(False)
        await self._wait_for_active_invocations()

    async def dispatch(self, message: OscControlMessage) -> OscDispatchResult:
        if self._closed:
            return OscDispatchResult(False, message.name, error="router_closed")
        if not self._ingress_enabled:
            return OscDispatchResult(False, message.name, error="router_disabled")
        if (
            not self._has_unsettled_command(message.name)
            and self._echo_suppression_provider is not None
            and self._echo_suppression_provider(message)
        ):
            return OscDispatchResult(False, message.name, error="echo_suppressed")
        generation = self._generation
        self._mark_unsettled(message.name)
        try:
            if message.name in _COALESCED_CONTROLS:
                return await self._enqueue_coalesced(message, generation)
            return await self._apply_serialized(message, generation=generation)
        finally:
            self._mark_settled(message.name)

    async def dispatch_packet(self, address: str, *args: Any) -> OscDispatchResult:
        message = decode_control_message(address, *args)
        return await self.dispatch(message)

    def handle_packet(self, address: str, *args: Any) -> bool:
        if self._closed:
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
                self.dispatch(message),
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
        if not self._is_generation_current(generation):
            return OscDispatchResult(False, message.name, error=self._generation_error())
        if cancelled is not None and cancelled():
            return OscDispatchResult(False, message.name, error="cancelled")
        invocation_task: asyncio.Task[Any] | None = None
        try:
            async with self._serial_lock:
                if not self._is_generation_current(generation):
                    return OscDispatchResult(False, message.name, error=self._generation_error())
                if cancelled is not None and cancelled():
                    return OscDispatchResult(False, message.name, error="cancelled")
                invocation_task = asyncio.current_task()
                if invocation_task is not None:
                    self._active_invocation_tasks.add(invocation_task)
                try:
                    result = await self._invoke(message)
                finally:
                    if invocation_task is not None:
                        self._active_invocation_tasks.discard(invocation_task)
        except Exception as exc:
            self._republish_canonical_state()
            self._report_error(f"{message.name}: {type(exc).__name__}")
            return OscDispatchResult(False, message.name, error=type(exc).__name__)
        if self._closed:
            return OscDispatchResult(False, message.name, error="router_closed")
        if (cancelled is not None and cancelled()) or (
            invocation_task is not None and invocation_task.cancelling()
        ):
            return OscDispatchResult(False, message.name, error="cancelled")
        if not self._is_generation_current(generation):
            return OscDispatchResult(False, message.name, error=self._generation_error())
        if _application_result_rejected(result):
            self._republish_canonical_state()
            if (
                _application_result_changed_canonical_state(result)
                and self._is_generation_current(generation)
                and not (cancelled is not None and cancelled())
            ):
                self._project_canonical_state(cast(OscControlPresentationName, message.name))
            self._report_error(f"{message.name}: application_rejected")
            return OscDispatchResult(False, message.name, error="application_rejected")
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
            return await app.set_translation_model(TRANSLATION_MODEL_IDS[message.value])  # type: ignore[index]
        if message.name == "PuriPuly_Fallback":
            return await app.set_fallback(FALLBACK_IDS[message.value])  # type: ignore[index]
        raise RuntimeError(f"unhandled PuriPuly OSC parameter: {message.name}")

    def _observe_task(self, task: asyncio.Task[OscDispatchResult]) -> None:
        with contextlib.suppress(asyncio.CancelledError):
            result = task.result()
            if result.error is not None:
                self._report_error(f"{result.parameter}: {result.error}")

    def _report_error(self, message: str) -> None:
        if self._error_sink is not None:
            self._error_sink(message)
        else:
            logger.warning("[OSC Control] %s", message)

    def _republish_canonical_state(self) -> None:
        callback = self._canonical_state_full_republisher or self._canonical_state_republisher
        if callback is None:
            return
        with contextlib.suppress(Exception):
            callback()

    def _publish_canonical_delta(self) -> None:
        callback = self._canonical_state_republisher
        if callback is None:
            return
        with contextlib.suppress(Exception):
            callback()

    def _project_canonical_state(self, control: OscControlPresentationName) -> None:
        callback = self._canonical_state_projector
        if callback is None:
            return
        with contextlib.suppress(Exception):
            callback(control)

    def _is_generation_current(self, generation: int) -> bool:
        return not self._closed and self._ingress_enabled and generation == self._generation

    def _generation_error(self) -> str:
        return "router_closed" if self._closed else "router_disabled"

    def _has_unsettled_command(self, parameter: str) -> bool:
        return self._unsettled_by_parameter.get(parameter, 0) > 0

    def _mark_unsettled(self, parameter: str) -> None:
        self._unsettled_by_parameter[parameter] = self._unsettled_by_parameter.get(parameter, 0) + 1

    def _mark_settled(self, parameter: str) -> None:
        remaining = self._unsettled_by_parameter.get(parameter, 0) - 1
        if remaining > 0:
            self._unsettled_by_parameter[parameter] = remaining
            return
        self._unsettled_by_parameter.pop(parameter, None)

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
