from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from puripuly_heart.app.ports.audio_test import (
    AUDIO_TEST_CHANNELS,
    AudioDiagnosticCandidate,
    AudioTestCaptureControlPort,
    AudioTestCapturePort,
    AudioTestCaptureRequest,
    AudioTestChannel,
    AudioTestSourceOpener,
)
from puripuly_heart.app.services.audio_diagnosis import (
    AUDIO_DIAGNOSE_CANDIDATE_TIMEOUT_S,
    AUDIO_DIAGNOSE_MAX_CANDIDATES,
    AudioDiagnosticEvidence,
    audio_diagnosis_receipt,
)
from puripuly_heart.core.audio.audio_test_vad import (
    AudioTestVadProbe,
    AudioTestVadSample,
    AudioTestVadSettings,
)
from puripuly_heart.core.diagnostic_codes import AUDIO_DIAGNOSE_FAILURE_CODES
from puripuly_heart.core.diagnostics import (
    deserialize_diagnostics,
    exception_diagnostics,
    serialize_diagnostics,
)
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.owned_thread import run_owned_thread_call
from puripuly_heart.core.runtime.audio_test import AudioTestCaptureRuntime

DIAGNOSTIC_CLEANUP_TIMEOUT_S = 5.0

AudioTestObservationCallback = Callable[[AudioTestChannel, tuple[AudioTestVadSample, ...]], object]
AudioTestDiagnosticsSink = Callable[[str, Mapping[str, object], BaseException | None], None]
AudioTestLogSink = Callable[[str], None]
AudioTestRuntimeFactory = Callable[[], AudioTestCaptureRuntime]
AudioTestVadSettingsProvider = Callable[[AudioTestChannel], AudioTestVadSettings]


@dataclass(frozen=True, slots=True)
class AudioTestChannelPlan:
    opener: AudioTestSourceOpener
    probe_factory: Callable[[AudioTestVadSettings], AudioTestVadProbe]
    ready_on_open: bool = False
    frame_timeout_s: float | None = None


AudioTestChannelPlanFactory = Callable[[AudioTestChannel], AudioTestChannelPlan]


def _idle_channel_snapshot() -> dict[str, object]:
    return {
        "state": "off",
        "failure_reason": None,
        "failure_type": None,
        "diagnostics": None,
        "secondary_diagnostics": [],
        "recovery_diagnostics": [],
        "device": {},
        "meter_level": 0.0,
        "speech_probability": 0.0,
        "speech_detected": False,
        "speech_threshold": None,
    }


@dataclass(slots=True)
class _ChannelSession:
    runtime: AudioTestCaptureRuntime
    state: str = "off"
    failure_reason: str | None = None
    failure_type: str | None = None
    diagnostics: ErrorDiagnostics | None = None
    secondary_diagnostics: tuple[ErrorDiagnostics, ...] = ()
    secondary_omitted_count: int = 0
    recovery_diagnostics: tuple[ErrorDiagnostics, ...] = ()
    device: dict[str, object] = field(default_factory=dict)
    meter_level: float = 0.0
    speech_probability: float = 0.0
    speech_detected: bool = False
    speech_threshold: float | None = None
    readiness: asyncio.Future[bool] | None = None
    probe: AudioTestVadProbe | None = None

    def reset(self) -> None:
        self.state = "off"
        self.failure_reason = None
        self.failure_type = None
        self.diagnostics = None
        self.secondary_diagnostics = ()
        self.secondary_omitted_count = 0
        self.recovery_diagnostics = ()
        self.device = {}
        self.clear_observation()
        self.probe = None

    def retain_failure(self, diagnostics: ErrorDiagnostics) -> None:
        if self.diagnostics is None:
            self.diagnostics = diagnostics
        elif diagnostics.diagnostic_id != self.diagnostics.diagnostic_id:
            self.retain_secondary(diagnostics)

    def retain_secondary(self, diagnostics: ErrorDiagnostics) -> None:
        if any(item.diagnostic_id == diagnostics.diagnostic_id for item in self.secondary_diagnostics):
            return
        if len(self.secondary_diagnostics) < 4:
            self.secondary_diagnostics += (diagnostics,)
        else:
            self.secondary_omitted_count += 1

    def clear_observation(self) -> None:
        self.meter_level = 0.0
        self.speech_probability = 0.0
        self.speech_detected = False

    def settle(self, ready: bool) -> None:
        if self.readiness is not None and not self.readiness.done():
            self.readiness.set_result(ready)

    def snapshot(self) -> dict[str, object]:
        return {
            "state": self.state,
            "failure_reason": self.failure_reason,
            "failure_type": self.failure_type,
            "diagnostics": (
                serialize_diagnostics(self.diagnostics) if self.diagnostics is not None else None
            ),
            "secondary_diagnostics": [
                serialize_diagnostics(item) for item in self.secondary_diagnostics
            ],
            "secondary_omitted_count": self.secondary_omitted_count,
            "secondary_truncated": self.secondary_omitted_count > 0,
            "device_metadata_availability": self.device.get("metadata_availability", "unknown"),
            "recovery_diagnostics": [
                serialize_diagnostics(item) for item in self.recovery_diagnostics
            ],
            "device": dict(self.device),
            "meter_level": self.meter_level,
            "speech_probability": self.speech_probability,
            "speech_detected": self.speech_detected,
            "speech_threshold": self.speech_threshold,
        }


@dataclass(slots=True)
class AudioTestSessionOwner:
    capture_port: AudioTestCapturePort
    plan_factory: AudioTestChannelPlanFactory
    vad_settings: AudioTestVadSettingsProvider
    capture_control: AudioTestCaptureControlPort
    log_sink: AudioTestLogSink | None = None
    diagnostics_sink: AudioTestDiagnosticsSink | None = None
    runtime_factory: AudioTestRuntimeFactory = AudioTestCaptureRuntime
    channels: tuple[AudioTestChannel, ...] = AUDIO_TEST_CHANNELS
    _sessions: dict[AudioTestChannel, _ChannelSession] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )
    _lifecycle_lock: asyncio.Lock | None = field(init=False, default=None, repr=False)
    _stopping: bool = field(init=False, default=False, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)
    _starting: bool = field(init=False, default=False, repr=False)
    _admission_revision: int = field(init=False, default=0, repr=False)
    _diagnosis_task: asyncio.Task[object] | None = field(init=False, default=None, repr=False)
    _diagnosis_cleanup_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _diagnosis_pause_task: asyncio.Task[bool] | None = field(init=False, default=None, repr=False)
    _diagnosis_pause_aborted: bool = field(init=False, default=False, repr=False)
    _diagnostic_evidence: AudioDiagnosticEvidence | None = field(
        init=False, default=None, repr=False
    )
    _restoration_status: dict[AudioTestChannel, str] = field(
        init=False, default_factory=dict, repr=False
    )
    _capture_intent_revision: dict[AudioTestChannel, int] = field(
        init=False, default_factory=dict, repr=False
    )
    _diagnosis_cancel_requested: bool = field(init=False, default=False, repr=False)
    _restore: dict[AudioTestChannel, object] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )
    _observation_callback: AudioTestObservationCallback | None = field(
        init=False,
        default=None,
        repr=False,
    )

    @property
    def owner_name(self) -> str:
        return "AudioTestSessionOwner"

    @property
    def diagnosing(self) -> bool:
        return self._diagnosis_task is not None

    @property
    def active(self) -> bool:
        return (
            self._starting
            or self._diagnosis_cleanup_task is not None
            or self._diagnosis_pause_task is not None
            or any(session.runtime.has_resources for session in self._sessions.values())
        )

    @property
    def snapshot(self) -> dict[str, object]:
        channels = {
            channel: (
                self._sessions[channel].snapshot()
                if channel in self._sessions
                else _idle_channel_snapshot()
            )
            for channel in self.channels
        }
        states = {item["state"] for item in channels.values()}
        state = self._overall_state(states)
        if self._starting:
            state = "pending"
        return {
            "state": state,
            "desired_active": state in {"pending", "ready", "degraded"},
            "effective_active": any(
                session.runtime.source is not None for session in self._sessions.values()
            ),
            "channels": channels,
            "restore_capture": [channel for channel in self.channels if channel in self._restore],
        }

    def runtime_for(self, channel: AudioTestChannel) -> AudioTestCaptureRuntime | None:
        session = self._sessions.get(channel)
        return session.runtime if session is not None else None

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "resource_fields": (
                "_sessions",
                "_lifecycle_lock",
                "_restore",
                "_closed",
                "_diagnosis_task",
                "_diagnosis_cleanup_task",
                "_diagnosis_pause_task",
            ),
            "stop_ingress": "close marks the owner closed and rejects new sessions",
            "shutdown_policy": (
                "close each channel AudioTestCaptureRuntime without restoring paused capture"
            ),
            "late_callback_rule": "drop observations and readiness from stale channel generations",
        }

    def release_capture_restore(self, channel: AudioTestChannel) -> None:
        self._capture_intent_revision[channel] = self._capture_intent_revision.get(channel, 0) + 1
        if (
            channel in self._restore
            or self._diagnosis_task is not None
            or self._diagnosis_pause_task is not None
            or self._diagnosis_cleanup_task is not None
        ):
            self._restoration_status[channel] = "skipped"
        self._restore.pop(channel, None)

    async def wait_ready(self) -> bool:
        futures = [
            session.readiness
            for session in self._sessions.values()
            if session.readiness is not None
        ]
        if not futures:
            return False
        results = await asyncio.gather(*(asyncio.shield(item) for item in futures))
        return all(results)

    async def start(
        self, *, observation_callback: AudioTestObservationCallback | None = None
    ) -> bool:
        if self._closed:
            return False
        async with self._lock():
            if self._closed:
                return False
            if self._diagnosis_pause_task is not None or self._diagnosis_cleanup_task is not None:
                return False
            if self.active and any(
                session.state in {"pending", "ready"} for session in self._sessions.values()
            ):
                self._observation_callback = observation_callback
                return True
            for channel in self.channels:
                session = self._session(channel)
                if not await self._recover_before_start(session.runtime):
                    self._fail_session(session, "resource_release_incomplete", None)
                    return False
            revision = self._admission_revision
            self._starting = True
            try:
                if not await self._pause_translation_capture():
                    if not self._closed:
                        await self._restore_translation_capture()
                    return False
                if self._closed or revision != self._admission_revision:
                    if not self._closed:
                        await self._restore_translation_capture()
                    return False
                self._observation_callback = observation_callback
                loop = asyncio.get_running_loop()
                started = False
                for channel in self.channels:
                    session = self._session(channel)
                    session.reset()
                    session.state = "pending"
                    session.readiness = loop.create_future()
                    try:
                        plan = self.plan_factory(channel)
                    except Exception as exc:
                        self._fail_session(
                            session, "unavailable", type(exc).__name__,
                            exception_diagnostics(
                                exc, component="audio_test", operation="audio.test",
                                code="source_open_failed",
                            ),
                        )
                        self._emit("plan_failed", {"channel": channel}, exc)
                        continue
                    try:
                        session.runtime.start(
                            lambda generation, channel=channel, plan=plan: self._run_session(
                                channel,
                                generation,
                                plan=plan,
                            )
                        )
                    except RuntimeError as exc:
                        self._fail_session(session, "session_start_failed", type(exc).__name__)
                        continue
                    started = True
                if not started:
                    await self._restore_translation_capture()
                return started
            finally:
                self._starting = False

    async def diagnose(
        self,
        *,
        candidates: tuple[AudioDiagnosticCandidate, ...],
        plan_factory: Callable[[AudioDiagnosticCandidate], AudioTestChannelPlan],
        input_host_api: str,
        input_device: str,
    ) -> dict[str, object]:
        receipt = audio_diagnosis_receipt(input_host_api=input_host_api, input_device=input_device)
        diagnosis = receipt["diagnosis"]
        async with self._lock():
            if self._closed or self.active:
                diagnosis["failure_code"] = "owner_closed" if self._closed else "audio_test_busy"
                return receipt
            self._diagnosis_task = asyncio.current_task()
            self._starting = True
            self._restoration_status = {channel: "not_needed" for channel in self.channels}
            evidence: list[AudioDiagnosticEvidence] = []
            cancelled = False
            self._diagnosis_cancel_requested = False
            self._diagnosis_pause_aborted = False
            cleanup_failed = False
            try:
                if not candidates:
                    diagnosis["failure_code"] = "no_input_candidates"
                elif not await self._pause_diagnostic_capture():
                    diagnosis["failure_code"] = "capture_pause_failed"
                    for channel in self.channels:
                        current = self.capture_control.state(channel)
                        if current.close_exception is not None or current.source_open:
                            self._restoration_status[channel] = "cleanup_pending"
                        session = self._sessions.get(channel)
                        if (
                            session is not None
                            and session.failure_reason == "resource_release_incomplete"
                        ):
                            diagnosis["failure_code"] = "resource_release_incomplete"
                else:
                    diagnosis["state"] = "completed"
                    if not any(candidate.current_route for candidate in candidates):
                        diagnosis["failure_code"] = "input_route_unavailable"
                    for candidate in candidates[:AUDIO_DIAGNOSE_MAX_CANDIDATES]:
                        if self._closed:
                            raise asyncio.CancelledError
                        item = AudioDiagnosticEvidence(candidate)
                        evidence.append(item)
                        self._diagnostic_evidence = item
                        self._emit(
                            "diagnose_started",
                            {"channel": "self", "target_index": candidate.device_index},
                        )
                        session = self._session("self")
                        session.reset()
                        session.state = "pending"
                        session.readiness = asyncio.get_running_loop().create_future()
                        started_at = asyncio.get_running_loop().time()
                        try:
                            plan = plan_factory(candidate)
                            task = session.runtime.start(
                                lambda generation, plan=plan: self._run_session(
                                    "self", generation, plan=plan
                                )
                            )
                            done, _pending = await asyncio.wait(
                                {task}, timeout=AUDIO_DIAGNOSE_CANDIDATE_TIMEOUT_S
                            )
                            if not done and not item.frame_count:
                                item.failure_code = "candidate_timeout"
                        except asyncio.CancelledError:
                            item.failure_code = item.failure_code or "cancelled"
                            raise
                        except Exception as exc:
                            item.failure_code = "session_start_failed"
                            item.diagnostics = exception_diagnostics(
                                exc, component="audio_test", operation="audio.test",
                                code="session_start_failed",
                            )
                            self._emit("diagnose_start_failed", {"channel": "self"}, exc)
                        finally:
                            item.elapsed_ms = int(
                                (asyncio.get_running_loop().time() - started_at) * 1000
                            )
                        if not await self._release_diagnostic_resources():
                            item.cleanup_complete = False
                            cleanup_failed = True
                            diagnosis["failure_code"] = (
                                diagnosis["failure_code"] or "resource_release_incomplete"
                            )
                            break
                        self._diagnostic_evidence = None
                        if self._diagnosis_cancel_requested:
                            raise asyncio.CancelledError
            except asyncio.CancelledError:
                cancelled = True
            finally:
                cleanup_complete = (
                    False
                    if cleanup_failed
                    else await self._release_diagnostic_resources(restore_capture=True)
                )
                self._diagnostic_evidence = None
                if evidence:
                    evidence[-1].cleanup_complete = (
                        self._diagnosis_cleanup_task is None
                        and not any(
                            session.runtime.has_resources for session in self._sessions.values()
                        )
                    )
                diagnosis["cleanup_complete"] = cleanup_complete
                if not cleanup_complete or self._closed:
                    for channel in self._restore:
                        self._restoration_status[channel] = (
                            "skipped" if self._closed else "cleanup_pending"
                        )
                diagnosis["restoration"] = dict(self._restoration_status)
                diagnosis["candidates"] = [item.snapshot() for item in evidence]
                diagnosis["secondary_diagnostics"] = [
                    serialize_diagnostics(item)
                    for session in self._sessions.values()
                    for item in (
                        *((session.diagnostics,) if session.diagnostics is not None else ()),
                        *session.secondary_diagnostics,
                    )
                ]
                diagnosis["secondary_omitted_count"] = sum(
                    session.secondary_omitted_count for session in self._sessions.values()
                )
                diagnosis["secondary_truncated"] = diagnosis["secondary_omitted_count"] > 0
                selected = max(evidence, key=lambda item: item.selection_rank, default=None)
                if selected is not None and selected.selection_rank:
                    diagnosis["selected_candidate_id"] = f"input-{selected.candidate.device_index}"
                    diagnosis["selection_basis"] = {1: "opened", 2: "frames", 3: "speech"}[
                        selected.selection_rank
                    ]
                if not cleanup_complete:
                    diagnosis["failure_code"] = (
                        diagnosis["failure_code"] or "resource_release_incomplete"
                    )
                elif "failed" in self._restoration_status.values():
                    diagnosis["failure_code"] = diagnosis["failure_code"] or "restore_failed"
                cancelled |= self._diagnosis_cancel_requested
                if cancelled:
                    diagnosis["state"] = "cancelled"
                    receipt["status"] = "cancelled"
                elif not cleanup_complete or "failed" in self._restoration_status.values():
                    receipt["status"] = "failed"
                elif any(item.selection_rank >= 2 for item in evidence):
                    receipt["status"] = "applied"
                self._starting = False
                self._diagnosis_task = None
            return receipt

    async def _pause_diagnostic_capture(self) -> bool:
        task = asyncio.create_task(
            self._pause_translation_capture(), name="AudioTestSessionOwner:diagnose-pause"
        )
        self._diagnosis_pause_task = task
        try:
            done, _pending = await asyncio.wait({task}, timeout=DIAGNOSTIC_CLEANUP_TIMEOUT_S)
        except asyncio.CancelledError:
            self._diagnosis_pause_aborted = True
            raise
        if not done:
            self._diagnosis_pause_aborted = True
            return False
        return task.result()

    async def _release_diagnostic_resources(self, *, restore_capture: bool = False) -> bool:
        task = self._diagnosis_cleanup_task
        if task is None:
            task = asyncio.create_task(
                self._finish_diagnostic_cleanup(restore_capture=restore_capture),
                name="AudioTestSessionOwner:diagnose-cleanup",
            )
            self._diagnosis_cleanup_task = task
            task.add_done_callback(self._on_diagnostic_cleanup_done)
        deadline = asyncio.get_running_loop().time() + DIAGNOSTIC_CLEANUP_TIMEOUT_S
        while not task.done():
            try:
                await asyncio.wait(
                    {task}, timeout=max(0.0, deadline - asyncio.get_running_loop().time())
                )
            except asyncio.CancelledError:
                self._diagnosis_cancel_requested = True
                continue
            if not task.done():
                return False
        self._diagnosis_cleanup_task = None
        try:
            task.result()
        except Exception as exc:
            self._emit("diagnose_cleanup_failed", {"channel": "self"}, exc)
            return False
        return not any(session.runtime.has_resources for session in self._sessions.values()) and (
            not restore_capture or "cleanup_pending" not in self._restoration_status.values()
        )

    def _on_diagnostic_cleanup_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        failure = task.exception()
        if (
            failure is None
            and self._diagnosis_task is None
            and self._diagnosis_pause_task is None
            and not any(session.runtime.has_resources for session in self._sessions.values())
            and self._diagnosis_cleanup_task is task
        ):
            self._diagnosis_cleanup_task = None

    async def _finish_diagnostic_cleanup(self, *, restore_capture: bool) -> None:
        pause = self._diagnosis_pause_task
        if pause is not None:
            try:
                await asyncio.shield(pause)
            except asyncio.CancelledError:
                if not pause.cancelled():
                    raise
                self._diagnosis_cancel_requested = True
            finally:
                if pause.done() and self._diagnosis_pause_task is pause:
                    self._diagnosis_pause_task = None
        await self._stop_sessions()
        if not restore_capture:
            return
        if self._closed:
            for channel in self._restore:
                current = self.capture_control.state(channel)
                if not current.source_open and current.close_exception is None:
                    self._restoration_status[channel] = "skipped"
        elif not any(session.runtime.has_resources for session in self._sessions.values()):
            await self._restore_translation_capture()

    def _cancel_diagnosis(self) -> None:
        task = self._diagnosis_task
        if task is not None and task is not asyncio.current_task() and not task.cancelling():
            task.cancel()

    async def stop(self) -> None:
        self._admission_revision += 1
        self._cancel_diagnosis()
        async with self._lock():
            if self._diagnosis_cleanup_task is not None or self._diagnosis_pause_task is not None:
                if not await self._release_diagnostic_resources():
                    raise RuntimeError("Audio test resource release is incomplete")
            await self._stop_sessions()
            if self.active:
                raise RuntimeError("Audio test resource release is incomplete")
            if not self._closed and not self.active:
                await self._restore_translation_capture()

    async def close(self) -> None:
        self._closed = True
        self._admission_revision += 1
        self._cancel_diagnosis()
        async with self._lock():
            if self._diagnosis_cleanup_task is not None or self._diagnosis_pause_task is not None:
                if not await self._release_diagnostic_resources():
                    raise RuntimeError("Audio test resource release is incomplete")
            try:
                await self._stop_sessions(close=True)
                if self.active:
                    raise RuntimeError("Audio test resource release is incomplete")
            finally:
                self._restore.clear()

    def mark_ready(self, channel: AudioTestChannel, generation: int) -> None:
        session = self._current(channel, generation)
        if session is None or session.state != "pending" or session.runtime.source is None:
            return
        session.state = "ready"
        session.settle(True)

    def mark_failure(
        self,
        channel: AudioTestChannel,
        generation: int,
        reason: str,
        error_type: str | None,
        diagnostics: ErrorDiagnostics | None = None,
    ) -> None:
        session = self._current(channel, generation)
        if session is None:
            return
        self._fail_session(session, reason, error_type, diagnostics)
        if self._diagnostic_evidence is not None and channel == "self":
            self._diagnostic_evidence.failure_code = self._diagnostic_evidence.failure_code or (
                reason if reason in AUDIO_DIAGNOSE_FAILURE_CODES else "capture_failed"
            )
            if diagnostics is not None:
                self._diagnostic_evidence.retain_failure(diagnostics)

    def set_device(
        self,
        channel: AudioTestChannel,
        generation: int,
        device: Mapping[str, object],
        recovery_diagnostics: tuple[ErrorDiagnostics, ...] = (),
    ) -> None:
        session = self._current(channel, generation)
        if session is not None:
            session.device = {key: value for key, value in device.items() if key != "diagnostics"}
            session.recovery_diagnostics = recovery_diagnostics
            metadata_diagnostics = device.get("diagnostics")
            if isinstance(metadata_diagnostics, Mapping):
                metadata_diagnostics = deserialize_diagnostics(metadata_diagnostics)
                session.retain_secondary(metadata_diagnostics)
            if self._diagnostic_evidence is not None and channel == "self":
                self._diagnostic_evidence.opened = True
                self._diagnostic_evidence.recovery_diagnostics = recovery_diagnostics
                self._diagnostic_evidence.device_metadata_availability = device.get(
                    "metadata_availability", "unknown"
                )
                if isinstance(metadata_diagnostics, ErrorDiagnostics):
                    self._diagnostic_evidence.retain_secondary(metadata_diagnostics)
                self._emit(
                    "diagnose_opened",
                    {
                        "channel": channel,
                        "target_index": self._diagnostic_evidence.candidate.device_index,
                    },
                )

    def observe_frame(self, channel: AudioTestChannel, generation: int, frame: object) -> None:
        session = self._current(channel, generation)
        if session is None or session.probe is None:
            return
        evidence = self._diagnostic_evidence if channel == "self" else None
        if evidence is not None:
            evidence.frame_count += 1
        session.probe.reconfigure(self.vad_settings(channel))
        samples = session.probe.process(frame)  # type: ignore[arg-type]
        if evidence is not None:
            evidence.observe(samples)
        if not samples:
            return
        latest = samples[-1]
        session.meter_level = max(sample.level for sample in samples)
        session.speech_probability = latest.probability
        session.speech_detected = latest.speech
        session.speech_threshold = latest.threshold
        callback = self._observation_callback
        if callback is None:
            return
        try:
            callback(channel, samples)
        except Exception as exc:
            self._emit("observation_callback_failed", {"channel": channel}, exc)

    async def _run_session(
        self,
        channel: AudioTestChannel,
        generation: int,
        *,
        plan: AudioTestChannelPlan,
    ) -> None:
        session = self._session(channel)
        try:
            settings = self.vad_settings(channel)
            try:
                probe = await run_owned_thread_call(lambda: plan.probe_factory(settings))
            except Exception as exc:
                self.mark_failure(
                    channel, generation, "vad_unavailable", type(exc).__name__,
                    exception_diagnostics(
                        exc, component="audio_test", operation="audio.test", code="vad_unavailable"
                    ),
                )
                self._emit("vad_unavailable", {"channel": channel}, exc)
                return
            if self._current(channel, generation) is None:
                return
            session.probe = probe
            session.speech_threshold = settings.speech_threshold
            await self.capture_port.capture(
                AudioTestCaptureRequest(
                    channel=channel,
                    generation=generation,
                    opener=plan.opener,
                    ready_on_open=plan.ready_on_open,
                    frame_timeout_s=plan.frame_timeout_s,
                    opened_callback=lambda gen, device, recovery: self.set_device(
                        channel, gen, device, recovery
                    ),
                    ready_callback=lambda gen: self.mark_ready(channel, gen),
                    failure_callback=lambda gen, reason, error_type, diagnostics: self.mark_failure(
                        channel,
                        gen,
                        reason,
                        error_type,
                        diagnostics,
                    ),
                    frame_callback=lambda gen, frame: self.observe_frame(channel, gen, frame),
                ),
                runtime=session.runtime,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.mark_failure(
                channel, generation, "capture_failed", type(exc).__name__,
                exception_diagnostics(
                    exc, component="audio_test", operation="audio.test", code="capture_failed"
                ),
            )
            self._emit("session_failed", {"channel": channel}, exc)
        finally:
            self._end_session(channel, generation)

    def _end_session(self, channel: AudioTestChannel, generation: int) -> None:
        session = self._current(channel, generation)
        if session is None:
            return
        if session.state == "pending":
            self._fail_session(session, "no_audio_frames", None)
        elif session.state == "ready":
            session.state = "off"
            session.settle(False)
        session.clear_observation()
        session.probe = None

    async def _stop_sessions(self, *, close: bool = False) -> None:
        self._stopping = True
        failures: list[Exception] = []
        try:
            for session in self._sessions.values():
                if session.state != "off" or session.runtime.has_resources:
                    session.state = "stopping"
                session.settle(False)
            for session in self._sessions.values():
                try:
                    if close:
                        await session.runtime.close()
                    else:
                        await session.runtime.stop()
                except Exception as exc:
                    failures.append(exc)
                    self._fail_session(
                        session, "resource_release_incomplete", type(exc).__name__,
                        exception_diagnostics(
                            exc, component="audio_test", operation="audio.test",
                            code="resource_release_incomplete",
                        ),
                    )
                else:
                    if session.runtime.has_resources:
                        self._fail_session(session, "resource_release_incomplete", None)
                    else:
                        session.reset()
                session.clear_observation()
                session.probe = None
        finally:
            self._stopping = False
            self._observation_callback = None
        if failures:
            raise ExceptionGroup("Audio test cleanup failed", failures)

    async def _pause_translation_capture(self) -> bool:
        for channel in self.channels:
            initial = self.capture_control.state(channel)
            intent_revision = self._capture_intent_revision.get(channel, 0)
            if initial.close_exception is not None:
                self._log(f"[AudioTest] channel={channel} capture cleanup failure is retained")
                self._fail_session(
                    self._session(channel),
                    "resource_release_incomplete",
                    type(initial.close_exception).__name__,
                    exception_diagnostics(
                        initial.close_exception,
                        component="audio_test", operation="audio.test",
                        code="resource_release_incomplete",
                    ),
                )
                return False
            if not initial.busy:
                continue
            if initial.enabled:
                self._restoration_status[channel] = "cleanup_pending"
            try:
                await self.capture_control.set_enabled(channel, False)
                current = self.capture_control.state(channel)
                if current.close_exception is not None:
                    raise current.close_exception
                if current.source_open:
                    raise RuntimeError("translation capture source remains open")
            except Exception as exc:
                self._log(
                    f"[AudioTest] channel={channel} failed to pause translation capture "
                    f"cause={type(exc).__name__}"
                )
                self._fail_session(
                    self._session(channel), "capture_pause_failed", type(exc).__name__,
                    exception_diagnostics(
                        exc, component="audio_test", operation="audio.test",
                        code="capture_pause_failed",
                    ),
                )
                return False
            finally:
                if initial.enabled and intent_revision == self._capture_intent_revision.get(
                    channel, 0
                ):
                    self._restore[channel] = self.capture_control.state(channel).token
                    self._restoration_status[channel] = "cleanup_pending"
            if (
                self._diagnosis_pause_task is asyncio.current_task()
                and self._diagnosis_pause_aborted
            ):
                return False
        return True

    async def _restore_translation_capture(self) -> None:
        for channel, token in tuple(self._restore.items()):
            if self._closed:
                return
            if self._restore.get(channel) != token:
                self._restoration_status[channel] = "skipped"
                continue
            try:
                current = self.capture_control.state(channel)
                if current.token != token:
                    self._restore.pop(channel, None)
                    self._restoration_status[channel] = "skipped"
                    continue
                if current.source_open or current.close_exception is not None:
                    self._restoration_status[channel] = "cleanup_pending"
                    continue
                if current.busy:
                    self._restore.pop(channel, None)
                    self._restoration_status[channel] = "skipped"
                    continue
                await self.capture_control.set_enabled(channel, True)
                if self._restore.get(channel) != token:
                    self._restoration_status[channel] = "skipped"
                    continue
                if not self.capture_control.state(channel).enabled:
                    raise RuntimeError("translation capture was not restored")
                self._restore.pop(channel, None)
                self._restoration_status[channel] = "restored"
            except Exception as exc:
                self._restoration_status[channel] = "failed"
                self._emit("restore_failed", {"channel": channel}, exc)
                self._log(
                    f"[AudioTest] channel={channel} failed to restore translation capture "
                    f"cause={type(exc).__name__}"
                )

    async def _recover_before_start(self, runtime: AudioTestCaptureRuntime) -> bool:
        if any(
            task is not None and not task.done()
            for task in (runtime.session_task, runtime.pending_frame_task)
        ):
            return False
        if not runtime.has_resources:
            return True
        try:
            await runtime.stop()
        except Exception as exc:
            self._emit("cleanup_retry_failed", {"error_type": type(exc).__name__}, exc)
            return False
        return not runtime.has_resources

    def _session(self, channel: AudioTestChannel) -> _ChannelSession:
        session = self._sessions.get(channel)
        if session is None:
            session = _ChannelSession(runtime=self.runtime_factory())
            self._sessions[channel] = session
        return session

    def _current(self, channel: AudioTestChannel, generation: int) -> _ChannelSession | None:
        session = self._sessions.get(channel)
        if session is None or self._closed or self._stopping:
            return None
        if not session.runtime.is_current_generation(generation):
            return None
        return session

    @staticmethod
    def _fail_session(
        session: _ChannelSession,
        reason: str,
        error_type: str | None,
        diagnostics: ErrorDiagnostics | None = None,
    ) -> None:
        session.state = "failed"
        session.failure_reason = reason
        session.failure_type = error_type
        if diagnostics is None:
            diagnostics = ErrorDiagnostics(
                component="audio_test",
                operation="audio.test",
                code=reason if reason in AUDIO_DIAGNOSE_FAILURE_CODES else "capture_failed",
                category="lifecycle",
                visibility="diagnostic_only",
                content_policy="metadata_only",
                status_code=None,
                retry_after_ms=None,
                fields={},
            )
        session.retain_failure(diagnostics)
        session.clear_observation()
        session.settle(False)

    @staticmethod
    def _overall_state(states: set[object]) -> str:
        if "stopping" in states:
            return "stopping"
        if "pending" in states:
            return "pending"
        if states == {"off"}:
            return "off"
        if states == {"ready"}:
            return "ready"
        if "ready" in states:
            return "degraded"
        if "failed" in states:
            return "failed"
        return "off"

    def _lock(self) -> asyncio.Lock:
        if self._lifecycle_lock is None:
            self._lifecycle_lock = asyncio.Lock()
        return self._lifecycle_lock

    def _log(self, message: str) -> None:
        if self.log_sink is not None:
            try:
                self.log_sink(message)
            except Exception:
                pass

    def _emit(
        self,
        event: str,
        metadata: Mapping[str, object],
        exception: BaseException | None = None,
    ) -> None:
        if exception is not None:
            channel = metadata.get("channel")
            session = self._sessions.get(channel)
            if session is not None and (
                session.diagnostics is None
                or event in {"diagnose_cleanup_failed", "cleanup_retry_failed", "restore_failed"}
            ):
                diagnostics = exception_diagnostics(
                    exception,
                    component="audio_test",
                    operation="audio.test",
                    code=event if event in AUDIO_DIAGNOSE_FAILURE_CODES else "capture_failed",
                    fields={"channel": channel},
                )
                session.retain_failure(diagnostics)
                if self._diagnostic_evidence is not None:
                    self._diagnostic_evidence.retain_failure(diagnostics)
        if self.diagnostics_sink is None:
            return
        try:
            self.diagnostics_sink(event, metadata, exception)
        except Exception:
            return


__all__ = [
    "AudioTestChannelPlan",
    "AudioTestChannelPlanFactory",
    "AudioTestDiagnosticsSink",
    "AudioTestLogSink",
    "AudioTestObservationCallback",
    "AudioTestRuntimeFactory",
    "AudioTestSessionOwner",
    "AudioTestVadSettingsProvider",
]
