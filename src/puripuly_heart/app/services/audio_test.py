from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from puripuly_heart.app.ports.audio_test import (
    AUDIO_TEST_CHANNELS,
    AudioTestCaptureControlPort,
    AudioTestCapturePort,
    AudioTestCaptureRequest,
    AudioTestChannel,
    AudioTestSourceOpener,
)
from puripuly_heart.core.audio.audio_test_vad import (
    AudioTestVadProbe,
    AudioTestVadSample,
    AudioTestVadSettings,
)
from puripuly_heart.core.owned_thread import run_owned_thread_call
from puripuly_heart.core.runtime.audio_test import AudioTestCaptureRuntime

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
        self.device = {}
        self.clear_observation()
        self.probe = None

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
    def active(self) -> bool:
        return self._starting or any(
            session.runtime.has_resources for session in self._sessions.values()
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
            "resource_fields": ("_sessions", "_lifecycle_lock", "_restore", "_closed"),
            "stop_ingress": "close marks the owner closed and rejects new sessions",
            "shutdown_policy": (
                "close each channel AudioTestCaptureRuntime without restoring paused capture"
            ),
            "late_callback_rule": "drop observations and readiness from stale channel generations",
        }

    def release_capture_restore(self, channel: AudioTestChannel) -> None:
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
                        self._fail_session(session, "unavailable", type(exc).__name__)
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

    async def stop(self) -> None:
        self._admission_revision += 1
        async with self._lock():
            await self._stop_sessions()
            if self.active:
                raise RuntimeError("Audio test resource release is incomplete")
            if not self._closed and not self.active:
                await self._restore_translation_capture()

    async def close(self) -> None:
        self._closed = True
        self._admission_revision += 1
        async with self._lock():
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
    ) -> None:
        session = self._current(channel, generation)
        if session is None:
            return
        self._fail_session(session, reason, error_type)

    def set_device(
        self,
        channel: AudioTestChannel,
        generation: int,
        device: Mapping[str, object],
    ) -> None:
        session = self._current(channel, generation)
        if session is not None:
            session.device = dict(device)

    def observe_frame(self, channel: AudioTestChannel, generation: int, frame: object) -> None:
        session = self._current(channel, generation)
        if session is None or session.probe is None:
            return
        session.probe.reconfigure(self.vad_settings(channel))
        samples = session.probe.process(frame)  # type: ignore[arg-type]
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
                self.mark_failure(channel, generation, "vad_unavailable", type(exc).__name__)
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
                    opened_callback=lambda gen, device: self.set_device(channel, gen, device),
                    ready_callback=lambda gen: self.mark_ready(channel, gen),
                    failure_callback=lambda gen, reason, error_type: self.mark_failure(
                        channel,
                        gen,
                        reason,
                        error_type,
                    ),
                    frame_callback=lambda gen, frame: self.observe_frame(channel, gen, frame),
                ),
                runtime=session.runtime,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.mark_failure(channel, generation, "capture_failed", type(exc).__name__)
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
                    self._fail_session(session, "resource_release_incomplete", type(exc).__name__)
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
            if initial.close_exception is not None:
                self._log(f"[AudioTest] channel={channel} capture cleanup failure is retained")
                self._fail_session(
                    self._session(channel),
                    "resource_release_incomplete",
                    type(initial.close_exception).__name__,
                )
                return False
            if not initial.busy:
                continue
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
                    self._session(channel), "capture_pause_failed", type(exc).__name__
                )
                return False
            if initial.enabled:
                self._restore[channel] = current.token
        return True

    async def _restore_translation_capture(self) -> None:
        restore = dict(self._restore)
        self._restore.clear()
        for channel, token in restore.items():
            try:
                current = self.capture_control.state(channel)
                if current.token != token or current.busy:
                    continue
                await self.capture_control.set_enabled(channel, True)
            except Exception as exc:
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
    def _fail_session(session: _ChannelSession, reason: str, error_type: str | None) -> None:
        session.state = "failed"
        session.failure_reason = reason
        session.failure_type = error_type
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
            self.log_sink(message)

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
    "AudioTestChannelPlan",
    "AudioTestChannelPlanFactory",
    "AudioTestDiagnosticsSink",
    "AudioTestLogSink",
    "AudioTestObservationCallback",
    "AudioTestRuntimeFactory",
    "AudioTestSessionOwner",
    "AudioTestVadSettingsProvider",
]
