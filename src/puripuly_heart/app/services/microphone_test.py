from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from puripuly_heart.app.ports.microphone_test import (
    MicrophoneTestCapturePort,
    MicrophoneTestCaptureRequest,
)
from puripuly_heart.core.runtime.mic_test import MicTestRuntime

MicrophoneTestMeterCallback = Callable[[float], object]
MicrophoneTestCaptureRequestFactory = Callable[
    [int, MicrophoneTestMeterCallback | None, float],
    MicrophoneTestCaptureRequest,
]
MicrophoneTestDisableSelfCapture = Callable[[], Awaitable[None]]
MicrophoneTestDiagnosticsSink = Callable[
    [str, Mapping[str, object], BaseException | None],
    None,
]
MicrophoneTestLogSink = Callable[[str], None]
MicrophoneTestRuntimeFactory = Callable[[], MicTestRuntime]
MicrophoneTestSelfCaptureSnapshotProvider = Callable[
    [],
    "MicrophoneTestSelfCaptureState",
]


@dataclass(frozen=True, slots=True)
class MicrophoneTestSelfCaptureState:
    stop_required: bool
    source_open: bool
    close_exception: BaseException | None = field(default=None, repr=False)


def _inactive_self_capture_state() -> MicrophoneTestSelfCaptureState:
    return MicrophoneTestSelfCaptureState(
        stop_required=False,
        source_open=False,
    )


async def _disable_inactive_self_capture() -> None:
    return None


def _log_value(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, str):
        return repr(value)
    return str(value)


@dataclass(frozen=True, slots=True)
class MicrophoneTestSessionRequest:
    audio_signature: tuple[object, ...]
    meter_callback: MicrophoneTestMeterCallback | None = field(
        default=None,
        repr=False,
    )
    level_log_interval_s: float = 10.0


@dataclass(slots=True)
class MicrophoneTestSessionOwner:
    capture_port: MicrophoneTestCapturePort
    capture_request_factory: MicrophoneTestCaptureRequestFactory
    self_capture_snapshot: MicrophoneTestSelfCaptureSnapshotProvider = _inactive_self_capture_state
    disable_self_capture: MicrophoneTestDisableSelfCapture = _disable_inactive_self_capture
    log_sink: MicrophoneTestLogSink | None = None
    diagnostics_sink: MicrophoneTestDiagnosticsSink | None = None
    runtime_factory: MicrophoneTestRuntimeFactory = MicTestRuntime
    _runtime: MicTestRuntime | None = field(init=False, default=None, repr=False)
    _lifecycle_lock: asyncio.Lock | None = field(init=False, default=None, repr=False)
    _meter_level: float = field(init=False, default=0.0, repr=False)
    _audio_signature: tuple[object, ...] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _closed: bool = field(init=False, default=False, repr=False)

    @property
    def owner_name(self) -> str:
        return "MicrophoneTestSessionOwner"

    @property
    def runtime_if_created(self) -> MicTestRuntime | None:
        return self._runtime

    @property
    def runtime(self) -> MicTestRuntime:
        if self._runtime is None:
            if self._closed:
                raise RuntimeError("MicrophoneTestSessionOwner is closed")
            self._runtime = self.runtime_factory()
        return self._runtime

    @runtime.setter
    def runtime(self, runtime: MicTestRuntime | None) -> None:
        self._runtime = runtime

    @property
    def meter_level(self) -> float:
        return self._meter_level

    @meter_level.setter
    def meter_level(self, value: float) -> None:
        self._meter_level = self._normalize_meter_level(value)

    @property
    def audio_signature(self) -> tuple[object, ...] | None:
        return self._audio_signature

    @audio_signature.setter
    def audio_signature(self, signature: tuple[object, ...] | None) -> None:
        self._audio_signature = signature

    @property
    def active(self) -> bool:
        runtime = self._runtime
        task = runtime.session_task if runtime is not None else None
        return task is not None and not task.done()

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "resource_fields": ("_runtime", "_lifecycle_lock", "_closed"),
            "stop_ingress": "close marks the owner closed and rejects new sessions",
            "shutdown_policy": "close the subordinate MicTestRuntime and clear meter state",
            "late_callback_rule": "drop meter updates from stale runtime generations",
        }

    async def start(self, request: MicrophoneTestSessionRequest) -> bool:
        if self._closed:
            return False
        if self._audio_signature is None:
            self._audio_signature = request.audio_signature
        async with self._lock():
            if self._closed:
                return False
            runtime = self.runtime
            task = runtime.session_task
            if task is not None:
                if not task.done():
                    return False
                await asyncio.gather(task, return_exceptions=True)

            if not await self._recover_before_start(runtime):
                return False
            if not await self._prepare_self_capture():
                return False

            try:
                runtime.start(
                    lambda generation: self._run_session(
                        generation,
                        request=request,
                    )
                )
            except RuntimeError:
                return False
            return True

    async def stop(self) -> None:
        async with self._lock():
            runtime = self._runtime
            if runtime is not None:
                await runtime.stop()
            self._meter_level = 0.0

    async def close(self) -> None:
        self._closed = True
        async with self._lock():
            runtime = self._runtime
            try:
                if runtime is not None:
                    await runtime.close()
            finally:
                self._meter_level = 0.0

    async def set_meter_level(
        self,
        value: float,
        meter_callback: MicrophoneTestMeterCallback | None,
        *,
        generation: int | None = None,
    ) -> None:
        if self._closed:
            return
        if generation is not None:
            runtime = self._runtime
            if runtime is None or not runtime.is_current_generation(generation):
                return
        level = self._normalize_meter_level(value)
        self._meter_level = level
        if meter_callback is None:
            return
        try:
            outcome = meter_callback(level)
            if inspect.isawaitable(outcome):
                await outcome
        except Exception as exc:
            self._emit(
                "meter_callback_failed",
                {"error_type": type(exc).__name__},
                exc,
            )

    async def _run_session(
        self,
        generation: int,
        *,
        request: MicrophoneTestSessionRequest,
    ) -> None:
        try:
            capture_request = self.capture_request_factory(
                generation,
                request.meter_callback,
                request.level_log_interval_s,
            )
            await self.capture_port.capture(
                capture_request,
                runtime=self.runtime,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._emit(
                "session_failed",
                {"error_type": type(exc).__name__},
                exc,
            )

    async def _recover_before_start(self, runtime: MicTestRuntime) -> bool:
        if runtime.has_active_direct_capture:
            return False
        if runtime.source is None and runtime.pending_frame_task is None:
            return True
        if runtime.pending_frame_task is not None and not runtime.pending_frame_task.done():
            return False
        try:
            await runtime.stop()
        except Exception as exc:
            self._emit(
                "cleanup_retry_failed",
                {"error_type": type(exc).__name__},
                exc,
            )
            return False
        return runtime.source is None and runtime.pending_frame_task is None

    async def _prepare_self_capture(self) -> bool:
        initial = self.self_capture_snapshot()
        requested = initial.stop_required
        if not requested:
            if initial.close_exception is not None:
                self._log_self_capture_auto_off(
                    requested=False,
                    completed=False,
                    exception=initial.close_exception,
                )
                return False
            self._log_self_capture_auto_off(
                requested=False,
                completed=True,
            )
            return True

        try:
            await self.disable_self_capture()
            current = self.self_capture_snapshot()
            if current.close_exception is not None:
                raise current.close_exception
            if current.source_open:
                raise RuntimeError("self microphone source still open after STT auto-off")
        except Exception as exc:
            self._log_self_capture_auto_off(
                requested=True,
                completed=False,
                exception=exc,
            )
            return False

        self._log_self_capture_auto_off(
            requested=True,
            completed=True,
        )
        return True

    def _log_self_capture_auto_off(
        self,
        *,
        requested: bool,
        completed: bool,
        exception: BaseException | None = None,
    ) -> None:
        if self.log_sink is None:
            return
        self.log_sink(
            "[MicTest] stt_auto_off "
            f"requested={requested} "
            f"completed={completed} "
            "exception_class="
            f"{_log_value(type(exception).__name__ if exception else None)} "
            "exception_message="
            f"{_log_value(str(exception) if exception else None)}"
        )

    def _lock(self) -> asyncio.Lock:
        if self._lifecycle_lock is None:
            self._lifecycle_lock = asyncio.Lock()
        return self._lifecycle_lock

    @staticmethod
    def _normalize_meter_level(value: float) -> float:
        level = max(0.0, min(1.0, float(value)))
        return 0.0 if level <= 1e-6 else level

    def _emit(
        self,
        event: str,
        metadata: Mapping[str, object],
        exception: BaseException | None = None,
    ) -> None:
        if self.diagnostics_sink is None:
            return
        try:
            self.diagnostics_sink(event, metadata, exception)
        except Exception:
            return


__all__ = [
    "MicrophoneTestCaptureRequestFactory",
    "MicrophoneTestDisableSelfCapture",
    "MicrophoneTestDiagnosticsSink",
    "MicrophoneTestLogSink",
    "MicrophoneTestMeterCallback",
    "MicrophoneTestRuntimeFactory",
    "MicrophoneTestSelfCaptureSnapshotProvider",
    "MicrophoneTestSelfCaptureState",
    "MicrophoneTestSessionOwner",
    "MicrophoneTestSessionRequest",
]
