from __future__ import annotations

import asyncio
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

import puripuly_heart.core.stt.controller as stt_controller_module
from puripuly_heart.config.settings import STTProviderName
from puripuly_heart.core import messages
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.runtime_logging import SessionLoggingMode, SessionRuntimeLoggingService
from puripuly_heart.core.speech_boundary import SpeechBoundaryReason
from puripuly_heart.core.stt.backend import STTBackendTranscriptEvent
from puripuly_heart.core.stt.controller import ManagedSTTProvider
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart
from puripuly_heart.domain.events import (
    STTErrorEvent,
    STTFinalEvent,
    STTPartialEvent,
    STTSessionState,
    STTSessionStateEvent,
)
from puripuly_heart.domain.models import FinalLanguageRun
from puripuly_heart.providers.stt.local_qwen_sherpa import LocalQwenSherpaSTTBackend
from tests.helpers.fakes import samples


@dataclass(slots=True)
class _RuntimeLogSinks:
    stream_handler: logging.Handler
    file_handler: logging.Handler
    log_file: object


def _make_runtime_logging_capture() -> tuple[SessionRuntimeLoggingService, io.StringIO]:
    stream = io.StringIO()
    stream_handler = logging.StreamHandler(stream)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))

    root_logger = logging.getLogger(f"test.stt.runtime.root.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False

    session_logger = logging.getLogger(f"test.stt.runtime.session.{uuid4()}")
    session_logger.handlers.clear()
    session_logger.propagate = False

    runtime_logging = SessionRuntimeLoggingService(
        root_logger=root_logger,
        session_logger=session_logger,
        sinks=_RuntimeLogSinks(
            stream_handler=stream_handler,
            file_handler=logging.NullHandler(),
            log_file="runtime.log",
        ),
    )
    return runtime_logging, stream


def _runtime_log_messages(stream: io.StringIO) -> list[str]:
    return [line for line in stream.getvalue().splitlines() if line]


def _raising_stt_fault_profile() -> str:
    raise RuntimeError("fault profile unavailable")


@dataclass(slots=True)
class _RaisingAudioDiagRuntimeLogging:
    fail_marker: str
    mode: SessionLoggingMode = SessionLoggingMode.DETAILED
    detailed_messages: list[str] | None = None
    basic_messages: list[str] | None = None

    def __post_init__(self) -> None:
        self.detailed_messages = []
        self.basic_messages = []

    def emit_basic(self, message: str, *, level: int = logging.INFO) -> None:
        _ = level
        assert self.basic_messages is not None
        self.basic_messages.append(message)

    def emit_detailed(self, message: str, *, level: int = logging.INFO) -> None:
        _ = level
        assert self.detailed_messages is not None
        self.detailed_messages.append(message)
        if self.fail_marker in message:
            raise RuntimeError("diagnostic log sink unavailable")


@dataclass(slots=True)
class FakeSession:
    audio: list[bytes]
    _queue: asyncio.Queue
    calls: list[str]
    _closed: bool = False

    def __init__(self) -> None:
        self.audio = []
        self._queue = asyncio.Queue()
        self.calls = []

    async def send_audio(self, pcm16le: bytes) -> None:
        self.audio.append(pcm16le)
        if len(self.audio) == 1:
            await self._queue.put(STTBackendTranscriptEvent(text="partial", is_final=False))

    async def stop(self) -> None:
        self.calls.append("stop")
        await self._queue.put(STTBackendTranscriptEvent(text="final", is_final=True))
        await self._queue.put(None)  # sentinel

    async def on_speech_end(
        self,
        *,
        trailing_silence_ms: int | None = None,
        reason: SpeechBoundaryReason | None = None,
    ) -> None:
        _ = (trailing_silence_ms, reason)
        self.calls.append("on_speech_end")

    async def close(self) -> None:
        self._closed = True
        self.calls.append("close")

    async def events(self):
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item


@dataclass(slots=True)
class FakeBackend:
    sessions: list[FakeSession]

    def __init__(self) -> None:
        self.sessions = []

    async def open_session(self) -> FakeSession:
        s = FakeSession()
        self.sessions.append(s)
        return s


class ClosableFakeBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


@dataclass(slots=True)
class Float32Session:
    audio_f32: list[np.ndarray]
    audio_bytes: list[bytes]
    _queue: asyncio.Queue
    calls: list[str]
    speech_ends: list[tuple[int | None, SpeechBoundaryReason | None]]
    _closed: bool = False

    def __init__(self) -> None:
        self.audio_f32 = []
        self.audio_bytes = []
        self._queue = asyncio.Queue()
        self.calls = []
        self.speech_ends = []

    async def send_audio(self, pcm16le: bytes) -> None:
        self.audio_bytes.append(pcm16le)

    async def send_audio_f32(self, samples_f32: np.ndarray) -> None:
        self.audio_f32.append(np.asarray(samples_f32, dtype=np.float32).copy())

    async def stop(self) -> None:
        self.calls.append("stop")
        await self._queue.put(None)

    async def on_speech_end(
        self,
        *,
        trailing_silence_ms: int | None = None,
        reason: SpeechBoundaryReason | None = None,
    ) -> None:
        self.calls.append("on_speech_end")
        self.speech_ends.append((trailing_silence_ms, reason))

    async def close(self) -> None:
        self._closed = True
        self.calls.append("close")
        await self._queue.put(None)

    async def events(self):
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item


@dataclass(slots=True)
class Float32Backend:
    sessions: list[Float32Session]

    def __init__(self) -> None:
        self.sessions = []

    async def open_session(self) -> Float32Session:
        session = Float32Session()
        self.sessions.append(session)
        return session


class StopFinalizingSession(Float32Session):
    __slots__ = ("stop_final_text",)

    def __init__(self, *, stop_final_text: str | None = None) -> None:
        super().__init__()
        self.stop_final_text = stop_final_text

    async def stop(self) -> None:
        self.calls.append("stop")
        if self.stop_final_text is not None:
            await self._queue.put(
                STTBackendTranscriptEvent(text=self.stop_final_text, is_final=True)
            )
        await self._queue.put(None)


@dataclass(slots=True)
class StopFinalizingBackend:
    sessions: list[StopFinalizingSession]
    first_stop_final_text: str

    def __init__(self, *, first_stop_final_text: str) -> None:
        self.sessions = []
        self.first_stop_final_text = first_stop_final_text

    async def open_session(self) -> StopFinalizingSession:
        stop_final_text = self.first_stop_final_text if not self.sessions else None
        session = StopFinalizingSession(stop_final_text=stop_final_text)
        self.sessions.append(session)
        return session


@dataclass(slots=True)
class EventOnlySession:
    items: list[object]

    async def send_audio(self, pcm16le: bytes) -> None:
        _ = pcm16le

    async def on_speech_end(
        self,
        *,
        trailing_silence_ms: int | None = None,
        reason: SpeechBoundaryReason | None = None,
    ) -> None:
        _ = (trailing_silence_ms, reason)

    async def stop(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def events(self):
        for item in self.items:
            yield item


@dataclass(slots=True)
class EventOnlyBackend:
    session: object

    async def open_session(self):
        return self.session


@dataclass(slots=True)
class FailingSession:
    error: Exception
    audio: list[bytes]

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.audio = []

    async def send_audio(self, pcm16le: bytes) -> None:
        self.audio.append(pcm16le)

    async def on_speech_end(
        self,
        *,
        trailing_silence_ms: int | None = None,
        reason: SpeechBoundaryReason | None = None,
    ) -> None:
        _ = (trailing_silence_ms, reason)

    async def stop(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def events(self):
        if False:
            yield None
        raise self.error


@dataclass(slots=True)
class FailingBackend:
    error: Exception

    async def open_session(self):
        return FailingSession(self.error)


@dataclass(slots=True)
class FailingOpenBackend:
    error: Exception

    async def open_session(self):
        raise self.error


class ControlledOpenBackend:
    def __init__(self) -> None:
        self.sessions: list[FakeSession] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.open_calls = 0
        self.active_opens = 0
        self.max_active_opens = 0

    async def open_session(self):
        self.open_calls += 1
        self.active_opens += 1
        self.max_active_opens = max(self.max_active_opens, self.active_opens)
        self.started.set()
        try:
            await self.release.wait()
        finally:
            self.active_opens -= 1
        session = FakeSession()
        self.sessions.append(session)
        return session

    def reset_controls(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()


@dataclass(slots=True)
class TerminalFailureSession:
    closed: bool = False
    stopped: bool = False

    async def send_audio(self, pcm16le: bytes) -> None:
        _ = pcm16le

    async def on_speech_end(
        self,
        *,
        trailing_silence_ms: int | None = None,
        reason: SpeechBoundaryReason | None = None,
    ) -> None:
        _ = (trailing_silence_ms, reason)

    async def stop(self) -> None:
        self.stopped = True

    async def close(self) -> None:
        self.closed = True

    async def events(self):
        if False:
            yield STTBackendTranscriptEvent(text="", is_final=False)
        raise RuntimeError("backend closed")


@dataclass(slots=True)
class TerminalFailureBackend:
    sessions: list[TerminalFailureSession]

    def __init__(self) -> None:
        self.sessions = []

    async def open_session(self) -> TerminalFailureSession:
        session = TerminalFailureSession()
        self.sessions.append(session)
        return session


class TerminalThenHealthyBackend:
    def __init__(self) -> None:
        self.sessions: list[object] = []

    async def open_session(self):
        if not self.sessions:
            session = TerminalFailureSession()
        else:
            session = FakeSession()
        self.sessions.append(session)
        return session


async def _next_event(stream, *, timeout_s: float = 0.2):
    return await asyncio.wait_for(stream.__anext__(), timeout=timeout_s)


async def _next_state(stream, state, *, max_events: int = 5):
    for _ in range(max_events):
        event = await _next_event(stream)
        if isinstance(event, STTSessionStateEvent) and event.state == state:
            return event
    raise AssertionError(f"Expected state {state}")


async def _next_typed_event(stream, event_type, *, max_events: int = 10):
    for _ in range(max_events):
        event = await _next_event(stream)
        if isinstance(event, event_type):
            return event
    raise AssertionError(f"Expected event of type {event_type.__name__}")


async def test_stt_controller_connects_on_speech_start():
    clock = FakeClock()
    backend = FakeBackend()
    stt = ManagedSTTProvider(
        backend=backend, sample_rate_hz=16000, clock=clock, reset_deadline_s=90.0
    )

    uid = __import__("uuid").uuid4()
    stream = stt.events()
    await stt.handle_vad_event(SpeechStart(uid, pre_roll=samples(0.0), chunk=samples(1.0)))
    first = await _next_state(stream, STTSessionState.STREAMING)

    assert len(backend.sessions) == 1
    assert isinstance(first, STTSessionStateEvent)
    assert first.state == STTSessionState.STREAMING

    await stt.close()


async def test_stt_provider_close_backend_closes_session_and_backend_once() -> None:
    backend = ClosableFakeBackend()
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        reset_deadline_s=90.0,
        finalize_grace_s=0.0,
    )
    uid = uuid4()
    await stt.handle_vad_event(SpeechStart(uid, pre_roll=samples(0.0), chunk=samples(1.0)))
    assert len(backend.sessions) == 1

    close_backend = getattr(stt, "close_backend", None)

    assert callable(close_backend)
    await close_backend()
    await close_backend()

    assert backend.sessions[0].calls[-1] == "close"
    assert backend.close_calls == 1
    assert stt.state == STTSessionState.DISCONNECTED


async def test_stt_controller_prefers_float32_session_audio_path() -> None:
    backend = Float32Backend()
    stt = ManagedSTTProvider(backend=backend, sample_rate_hz=16000, reset_deadline_s=90.0)

    uid = uuid4()
    chunk = np.array([0.123456, -0.234567, 0.9999], dtype=np.float32)
    stream = stt.events()
    await stt.handle_vad_event(
        SpeechStart(uid, pre_roll=np.zeros(0, dtype=np.float32), chunk=chunk)
    )
    await _next_state(stream, STTSessionState.STREAMING)

    session = backend.sessions[0]
    assert session.audio_bytes == []
    assert len(session.audio_f32) == 1
    np.testing.assert_array_equal(session.audio_f32[0], chunk)

    await stt.close()


async def test_stt_controller_logs_input_diagnostics_on_speech_end() -> None:
    backend = Float32Backend()
    runtime_logging, log_stream = _make_runtime_logging_capture()
    runtime_logging.set_mode(SessionLoggingMode.DETAILED)
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        reset_deadline_s=90.0,
        runtime_logging=runtime_logging,
    )

    try:
        uid = uuid4()
        await stt.handle_vad_event(
            SpeechStart(
                uid,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=np.ones(16000, dtype=np.float32),
            )
        )
        await stt.handle_vad_event(SpeechEnd(uid, trailing_silence_ms=64))

        messages = _runtime_log_messages(log_stream)
        assert any("[AudioDiag][STTInput][self]" in message for message in messages)
        assert any(
            "chunk_count=1" in message and "audio_ms=1000.0" in message for message in messages
        )
    finally:
        await stt.close()
        runtime_logging.close()


async def test_stt_controller_preserves_soft_boundary_reason_and_observed_tail() -> None:
    backend = Float32Backend()
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        channel="peer",
        reset_deadline_s=90.0,
    )
    uid = uuid4()

    try:
        await stt.handle_vad_event(
            SpeechStart(
                uid,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(1.0),
            )
        )
        await stt.handle_vad_event(SpeechEnd(uid, trailing_silence_ms=160, reason="soft_pause"))

        assert backend.sessions[0].speech_ends == [(160, "soft_pause")]
    finally:
        await stt.close()


async def test_stt_input_fault_profile_modifies_audio_after_vad() -> None:
    backend = Float32Backend()
    runtime_logging, log_stream = _make_runtime_logging_capture()
    runtime_logging.set_mode(SessionLoggingMode.DETAILED)
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        reset_deadline_s=90.0,
        runtime_logging=runtime_logging,
        stt_input_fault_profile_provider=lambda: "stt_input_low_snr_vad_pass",
    )

    try:
        uid = uuid4()
        original = np.ones(16000, dtype=np.float32)
        original_before = original.copy()
        await stt.handle_vad_event(
            SpeechStart(uid, pre_roll=np.zeros(0, dtype=np.float32), chunk=original)
        )

        session = backend.sessions[0]
        assert len(session.audio_f32) == 1
        assert float(np.max(np.abs(session.audio_f32[0]))) < 0.05
        np.testing.assert_array_equal(original, original_before)
        messages = _runtime_log_messages(log_stream)
        assert any(
            "[AudioDiag][STTFault][self] profile=stt_input_low_snr_vad_pass" in message
            for message in messages
        )
    finally:
        await stt.close()
        runtime_logging.close()


async def test_stt_input_fault_log_failure_does_not_block_backend_audio() -> None:
    backend = Float32Backend()
    runtime_logging = _RaisingAudioDiagRuntimeLogging("[AudioDiag][STTFault]")
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        reset_deadline_s=90.0,
        runtime_logging=runtime_logging,  # type: ignore[arg-type]
        stt_input_fault_profile_provider=lambda: "stt_input_low_snr_vad_pass",
    )

    try:
        uid = uuid4()
        original = np.ones(16000, dtype=np.float32)
        original_before = original.copy()
        await stt.handle_vad_event(
            SpeechStart(uid, pre_roll=np.zeros(0, dtype=np.float32), chunk=original)
        )

        session = backend.sessions[0]
        assert len(session.audio_f32) == 1
        assert float(np.max(np.abs(session.audio_f32[0]))) < 0.05
        np.testing.assert_array_equal(original, original_before)
        assert runtime_logging.detailed_messages is not None
        assert any(
            "[AudioDiag][STTFault][self] profile=stt_input_low_snr_vad_pass" in message
            for message in runtime_logging.detailed_messages
        )
    finally:
        await stt.close()


async def test_stt_input_diagnostic_log_failure_does_not_skip_speech_end() -> None:
    backend = Float32Backend()
    runtime_logging = _RaisingAudioDiagRuntimeLogging("[AudioDiag][STTInput]")
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        reset_deadline_s=90.0,
        runtime_logging=runtime_logging,  # type: ignore[arg-type]
    )

    try:
        uid = uuid4()
        await stt.handle_vad_event(
            SpeechStart(
                uid,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=np.ones(16000, dtype=np.float32),
            )
        )
        await stt.handle_vad_event(SpeechEnd(uid, trailing_silence_ms=64))

        session = backend.sessions[0]
        assert "on_speech_end" in session.calls
        assert runtime_logging.detailed_messages is not None
        assert any(
            "[AudioDiag][STTInput][self]" in message
            for message in runtime_logging.detailed_messages
        )
    finally:
        await stt.close()


async def test_stt_input_metric_record_failure_does_not_block_backend_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = Float32Backend()
    runtime_logging, _log_stream = _make_runtime_logging_capture()
    runtime_logging.set_mode(SessionLoggingMode.DETAILED)
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        reset_deadline_s=90.0,
        runtime_logging=runtime_logging,
    )

    def fail_sum(*_args, **_kwargs):
        raise RuntimeError("diagnostic sum failed")

    original_sum = stt_controller_module.np.sum
    monkeypatch.setattr(stt_controller_module.np, "sum", fail_sum)

    try:
        uid = uuid4()
        original = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
        original_before = original.copy()
        await stt.handle_vad_event(
            SpeechStart(uid, pre_roll=np.zeros(0, dtype=np.float32), chunk=original)
        )
        monkeypatch.setattr(stt_controller_module.np, "sum", original_sum)

        session = backend.sessions[0]
        assert len(session.audio_f32) == 1
        np.testing.assert_array_equal(session.audio_f32[0], original)
        np.testing.assert_array_equal(original, original_before)
    finally:
        await stt.close()
        runtime_logging.close()


async def test_stt_input_metric_emit_failure_does_not_skip_speech_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = Float32Backend()
    runtime_logging, _log_stream = _make_runtime_logging_capture()
    runtime_logging.set_mode(SessionLoggingMode.DETAILED)
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        reset_deadline_s=90.0,
        runtime_logging=runtime_logging,
    )

    try:
        uid = uuid4()
        await stt.handle_vad_event(
            SpeechStart(
                uid,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=np.ones(16000, dtype=np.float32),
            )
        )

        def fail_sqrt(*_args, **_kwargs):
            raise RuntimeError("diagnostic sqrt failed")

        original_sqrt = stt_controller_module.np.sqrt
        monkeypatch.setattr(stt_controller_module.np, "sqrt", fail_sqrt)
        await stt.handle_vad_event(SpeechEnd(uid, trailing_silence_ms=64))
        monkeypatch.setattr(stt_controller_module.np, "sqrt", original_sqrt)

        session = backend.sessions[0]
        assert "on_speech_end" in session.calls
    finally:
        await stt.close()
        runtime_logging.close()


@pytest.mark.parametrize(
    "profile_provider",
    [
        _raising_stt_fault_profile,
        lambda: "not_a_fault_profile",
    ],
)
async def test_stt_input_fault_profile_resolution_failure_uses_original_audio(
    profile_provider,
) -> None:
    backend = Float32Backend()
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        reset_deadline_s=90.0,
        stt_input_fault_profile_provider=profile_provider,
    )

    try:
        uid = uuid4()
        original = np.linspace(-1.0, 1.0, 16000, dtype=np.float32)
        original_before = original.copy()
        await stt.handle_vad_event(
            SpeechStart(uid, pre_roll=np.zeros(0, dtype=np.float32), chunk=original)
        )

        session = backend.sessions[0]
        assert len(session.audio_f32) == 1
        np.testing.assert_array_equal(session.audio_f32[0], original)
        np.testing.assert_array_equal(original, original_before)
    finally:
        await stt.close()


async def test_stt_controller_resets_with_bridging_during_speech():
    """Timer-based reset triggers bridging when speaking at deadline."""
    backend = FakeBackend()
    runtime_logging, log_stream = _make_runtime_logging_capture()
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        reset_deadline_s=0.1,  # 100ms for fast test
        drain_timeout_s=0.05,
        bridging_ms=64,
        finalize_grace_s=0.0,
        runtime_logging=runtime_logging,
    )

    try:
        uid = __import__("uuid").uuid4()
        stream = stt.events()
        await stt.handle_vad_event(SpeechStart(uid, pre_roll=samples(0.0), chunk=samples(1.0)))
        _ = await _next_event(stream)

        # Wait for timer to fire while still speaking (utterance_id is set)
        await asyncio.sleep(0.15)

        assert len(backend.sessions) == 2
        assert len(backend.sessions[1].audio) >= 1  # bridging audio
        assert "on_speech_end" not in backend.sessions[0].calls

        messages = _runtime_log_messages(log_stream)
        assert "[STT] Session reset while speaking; bridged to a new session" in messages
        assert not any("BRIDGING:" in message for message in messages)
    finally:
        await stt.close()
        runtime_logging.close()


async def test_stt_controller_resets_with_bridging_uses_float32_fast_path() -> None:
    backend = Float32Backend()
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        reset_deadline_s=0.1,
        drain_timeout_s=0.05,
        bridging_ms=64,
        finalize_grace_s=0.0,
    )

    try:
        uid = uuid4()
        chunk = np.array([0.123456, -0.234567, 0.9999], dtype=np.float32)
        stream = stt.events()
        await stt.handle_vad_event(
            SpeechStart(uid, pre_roll=np.zeros(0, dtype=np.float32), chunk=chunk)
        )
        await _next_state(stream, STTSessionState.STREAMING)

        await asyncio.sleep(0.15)

        assert len(backend.sessions) == 2
        assert backend.sessions[1].audio_bytes == []
        assert len(backend.sessions[1].audio_f32) == 1
        np.testing.assert_array_equal(backend.sessions[1].audio_f32[0], chunk)
    finally:
        await stt.close()


async def test_stt_controller_resets_on_silence():
    """Timer-based reset closes session when silent at deadline."""
    backend = FakeBackend()
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        reset_deadline_s=0.1,  # 100ms for fast test
        reconnect_window_s=0.0,  # Disable auto-reconnect -> silence reset
        drain_timeout_s=0.05,
        finalize_grace_s=0.0,
    )

    uid = __import__("uuid").uuid4()
    stream = stt.events()
    await stt.handle_vad_event(SpeechStart(uid, pre_roll=samples(0.0), chunk=samples(1.0)))
    await _next_state(stream, STTSessionState.STREAMING)

    # End speech before timer fires
    await stt.handle_vad_event(SpeechEnd(uid))

    # Wait for timer to fire during silence
    await asyncio.sleep(0.15)

    # Verify: session closed (DISCONNECTED state)
    assert stt.state == STTSessionState.DISCONNECTED
    assert len(backend.sessions) == 1  # No new session created

    await stt.close()


async def test_stt_controller_finalize_on_close_while_speaking():
    clock = FakeClock()
    backend = FakeBackend()
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        clock=clock,
        reset_deadline_s=90.0,
        finalize_grace_s=0.0,
    )

    uid = __import__("uuid").uuid4()
    stream = stt.events()
    await stt.handle_vad_event(SpeechStart(uid, pre_roll=samples(0.0), chunk=samples(1.0)))
    await _next_state(stream, STTSessionState.STREAMING)

    await stt.close()

    calls = backend.sessions[0].calls
    assert "on_speech_end" in calls
    assert "stop" in calls
    assert calls.index("on_speech_end") < calls.index("stop")


async def test_stt_controller_reconnects_when_recent_speech():
    """Timer-based reset reconnects when recent speech at deadline."""
    backend = FakeBackend()
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        reset_deadline_s=0.1,  # 100ms for fast test
        reconnect_window_s=0.5,  # Enable auto-reconnect
        drain_timeout_s=0.05,
        finalize_grace_s=0.0,
    )

    uid = __import__("uuid").uuid4()
    stream = stt.events()

    # 1. Speech start -> session 1 opens
    await stt.handle_vad_event(SpeechStart(uid, pre_roll=samples(0.0), chunk=samples(1.0)))
    await _next_state(stream, STTSessionState.STREAMING)
    assert len(backend.sessions) == 1

    # 2. End speech before timer fires (sets _last_speech_end_time)
    await stt.handle_vad_event(SpeechEnd(uid))

    # 3. Wait for timer to fire while in "recent speech" window
    await asyncio.sleep(0.15)

    # 4. Verify: new session opened via reconnect (not silence reset)
    assert len(backend.sessions) == 2
    assert "on_speech_end" in backend.sessions[0].calls  # allow_finalize=True

    await stt.close()


async def test_stt_controller_disconnects_when_reconnect_disabled():
    """Timer-based reset with reconnect_window_s=0 -> silence reset (DISCONNECTED)"""
    backend = FakeBackend()
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        reset_deadline_s=0.1,  # 100ms for fast test
        reconnect_window_s=0.0,  # Disabled -> always silence reset
        drain_timeout_s=0.05,
        finalize_grace_s=0.0,
    )

    uid = __import__("uuid").uuid4()
    stream = stt.events()

    # 1. Speech start -> session opens
    await stt.handle_vad_event(SpeechStart(uid, pre_roll=samples(0.0), chunk=samples(1.0)))
    await _next_state(stream, STTSessionState.STREAMING)

    # 2. End speech before timer fires
    await stt.handle_vad_event(SpeechEnd(uid))

    # 3. Wait for timer to fire - since reconnect_window_s=0, always silence reset
    await asyncio.sleep(0.15)

    # Verify: DISCONNECTED state, no new session
    assert stt.state == STTSessionState.DISCONNECTED
    assert len(backend.sessions) == 1  # No new session

    await stt.close()


async def test_stt_controller_reconnect_allows_finalize():
    """Timer-based reconnect drains old session with allow_finalize=True"""
    backend = FakeBackend()
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        reset_deadline_s=0.1,  # 100ms for fast test
        reconnect_window_s=0.5,  # Enable auto-reconnect
        drain_timeout_s=0.05,
        finalize_grace_s=0.0,
    )

    uid = __import__("uuid").uuid4()
    stream = stt.events()

    await stt.handle_vad_event(SpeechStart(uid, pre_roll=samples(0.0), chunk=samples(1.0)))
    await _next_state(stream, STTSessionState.STREAMING)

    # End speech, then wait for timer to trigger reconnect
    await stt.handle_vad_event(SpeechEnd(uid))
    await asyncio.sleep(0.15)

    # Verify: old session called on_speech_end (finalize via allow_finalize=True)
    old_session = backend.sessions[0]
    assert "on_speech_end" in old_session.calls
    assert "stop" in old_session.calls

    await stt.close()


async def test_stt_controller_reconnect_no_bridging_audio():
    """Timer-based reconnect should not send bridging audio to new session"""
    backend = FakeBackend()
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        reset_deadline_s=0.1,  # 100ms for fast test
        reconnect_window_s=0.5,  # Enable auto-reconnect
        bridging_ms=64,
        drain_timeout_s=0.05,
        finalize_grace_s=0.0,
    )

    uid = __import__("uuid").uuid4()
    stream = stt.events()

    await stt.handle_vad_event(SpeechStart(uid, pre_roll=samples(0.0), chunk=samples(1.0)))
    await _next_state(stream, STTSessionState.STREAMING)

    # End speech, then wait for timer to trigger reconnect
    await stt.handle_vad_event(SpeechEnd(uid))
    await asyncio.sleep(0.15)

    # Verify: new session has no bridging audio (unlike bridging reset)
    new_session = backend.sessions[1]
    assert len(new_session.audio) == 0

    await stt.close()


async def test_stt_controller_reconnect_fallback_on_failure():
    """Timer-based reconnect failure should fallback to silence reset"""

    class FailingBackend:
        def __init__(self):
            self.sessions = []
            self.call_count = 0

        async def open_session(self):
            self.call_count += 1
            if self.call_count == 1:
                s = FakeSession()
                self.sessions.append(s)
                return s
            raise ConnectionError("Failed to connect")

    backend = FailingBackend()
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        reset_deadline_s=0.1,  # 100ms for fast test
        reconnect_window_s=0.5,  # Enable auto-reconnect
        drain_timeout_s=0.05,
        finalize_grace_s=0.0,
        connect_attempts=1,
    )

    uid = __import__("uuid").uuid4()
    stream = stt.events()

    await stt.handle_vad_event(SpeechStart(uid, pre_roll=samples(0.0), chunk=samples(1.0)))
    await _next_state(stream, STTSessionState.STREAMING)

    # End speech, then wait for timer to trigger reconnect (which will fail)
    await stt.handle_vad_event(SpeechEnd(uid))
    await asyncio.sleep(0.15)

    # Verify: connection failure -> DISCONNECTED state (fallback to silence reset)
    assert stt.state == STTSessionState.DISCONNECTED

    await stt.close()


async def test_stt_controller_reconnect_failure_uses_safe_runtime_log() -> None:
    raw_detail = "socket reconnect failed token=stt-reconnect-secret-456"

    class FailingReconnectBackend:
        def __init__(self) -> None:
            self.sessions = []
            self.call_count = 0

        async def open_session(self):
            self.call_count += 1
            if self.call_count == 1:
                session = FakeSession()
                self.sessions.append(session)
                return session
            raise ConnectionError(raw_detail)

    runtime_logging, log_stream = _make_runtime_logging_capture()
    stt = ManagedSTTProvider(
        backend=FailingReconnectBackend(),
        sample_rate_hz=16000,
        reset_deadline_s=0.1,
        reconnect_window_s=0.5,
        drain_timeout_s=0.05,
        finalize_grace_s=0.0,
        connect_attempts=1,
        stt_provider_name=STTProviderName.SONIOX,
        runtime_logging=runtime_logging,
    )

    try:
        stream = stt.events()
        utterance_id = uuid4()
        await stt.handle_vad_event(
            SpeechStart(utterance_id, pre_roll=samples(0.0), chunk=samples(1.0))
        )
        await _next_state(stream, STTSessionState.STREAMING)
        await stt.handle_vad_event(SpeechEnd(utterance_id))

        await asyncio.sleep(0.15)

        runtime_log = "\n".join(_runtime_log_messages(log_stream))
        assert raw_detail not in runtime_log
        assert "stt-reconnect-secret-456" not in runtime_log
        assert (
            "[STT] Reconnect failed; closing until next speech: category=network code=stt.network"
        ) in runtime_log
    finally:
        await stt.close()
        runtime_logging.close()


async def test_stt_controller_summarizes_retry_connect_in_basic_runtime_logs() -> None:
    class RetryOnceBackend:
        def __init__(self) -> None:
            self.attempts = 0

        async def open_session(self):
            self.attempts += 1
            if self.attempts == 1:
                raise ConnectionError("temporary outage")
            return FakeSession()

    runtime_logging, log_stream = _make_runtime_logging_capture()
    stt = ManagedSTTProvider(
        backend=RetryOnceBackend(),
        sample_rate_hz=16000,
        clock=FakeClock(),
        connect_attempts=2,
        connect_retry_base_s=0.001,
        connect_retry_max_s=0.001,
        runtime_logging=runtime_logging,
    )

    try:
        stream = stt.events()
        await stt.handle_vad_event(SpeechStart(uuid4(), pre_roll=samples(0.0), chunk=samples(1.0)))
        await _next_state(stream, STTSessionState.STREAMING)

        messages = _runtime_log_messages(log_stream)
        assert "[STT] Session connected after 1 retry" in messages
        assert not any("Opening new session" in message for message in messages)
        assert not any("Retrying session in" in message for message in messages)
    finally:
        await stt.close()
        runtime_logging.close()


async def test_managed_stt_provider_open_failure_uses_message_ref_and_safe_runtime_log() -> None:
    raw_detail = "microphone socket failed token=stt-secret-789"
    runtime_logging, log_stream = _make_runtime_logging_capture()
    runtime_logging.set_mode(SessionLoggingMode.DETAILED)
    stt = ManagedSTTProvider(
        backend=FailingOpenBackend(ConnectionError(raw_detail)),
        sample_rate_hz=16000,
        stt_provider_name=STTProviderName.SONIOX,
        connect_attempts=1,
        runtime_logging=runtime_logging,
    )

    try:
        stream = stt.events()
        await stt.handle_vad_event(SpeechStart(uuid4(), pre_roll=samples(0.0), chunk=samples(1.0)))
        error_event = None
        for _ in range(5):
            event = await _next_event(stream)
            if isinstance(event, STTErrorEvent):
                error_event = event
                break
        assert error_event is not None

        error_report_type = getattr(messages, "UserErrorReport", None)
        assert error_report_type is not None, "UserErrorReport DTO is missing"
        assert isinstance(error_event.message, messages.UserMessageRef)
        assert error_event.message.key == "stt.failure"
        assert error_event.diagnostics is not None
        assert error_event.diagnostics.category == messages.DIAGNOSTIC_CATEGORY_NETWORK
        assert error_event.diagnostics.fields["provider"] == "soniox"
        assert raw_detail not in repr(error_event.message)
        assert raw_detail not in repr(error_event.diagnostics)

        runtime_log = "\n".join(_runtime_log_messages(log_stream))
        assert raw_detail not in runtime_log
        assert "category=network" in runtime_log
        assert "code=stt.network" in runtime_log
    finally:
        await stt.close()
        runtime_logging.close()


async def test_stt_controller_serializes_concurrent_session_open() -> None:
    backend = ControlledOpenBackend()
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        reset_deadline_s=90.0,
        connect_attempts=1,
    )

    try:
        first_open = asyncio.create_task(stt.warmup())
        await asyncio.wait_for(backend.started.wait(), timeout=0.2)
        second_open = asyncio.create_task(stt.warmup())

        await asyncio.sleep(0.01)

        assert backend.open_calls == 1
        assert backend.max_active_opens == 1
        assert stt.state == STTSessionState.CONNECTING

        backend.release.set()
        await asyncio.gather(first_open, second_open)

        assert backend.open_calls == 1
        assert backend.max_active_opens == 1
        assert len(backend.sessions) == 1
        assert stt.state == STTSessionState.STREAMING
    finally:
        await stt.close()


async def test_stt_controller_open_cancellation_releases_serialization() -> None:
    backend = ControlledOpenBackend()
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        reset_deadline_s=90.0,
        connect_attempts=1,
    )

    try:
        cancelled_open = asyncio.create_task(stt.warmup())
        await asyncio.wait_for(backend.started.wait(), timeout=0.2)

        cancelled_open.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_open

        assert stt.state == STTSessionState.DISCONNECTED
        assert backend.active_opens == 0

        backend.reset_controls()
        reopened = asyncio.create_task(stt.warmup())
        await asyncio.wait_for(backend.started.wait(), timeout=0.2)
        backend.release.set()
        await reopened

        assert backend.open_calls == 2
        assert len(backend.sessions) == 1
        assert stt.state == STTSessionState.STREAMING
    finally:
        await stt.close()


async def test_stt_controller_without_runtime_logging_stays_basic_only(caplog) -> None:
    class RetryOnceBackend:
        def __init__(self) -> None:
            self.attempts = 0

        async def open_session(self):
            self.attempts += 1
            if self.attempts == 1:
                raise ConnectionError("temporary outage")
            return FakeSession()

    stt = ManagedSTTProvider(
        backend=RetryOnceBackend(),
        sample_rate_hz=16000,
        clock=FakeClock(),
        connect_attempts=2,
        connect_retry_base_s=0.001,
        connect_retry_max_s=0.001,
    )

    try:
        with caplog.at_level(logging.INFO, logger="puripuly_heart.core.stt.controller"):
            await stt.handle_vad_event(
                SpeechStart(uuid4(), pre_roll=samples(0.0), chunk=samples(1.0))
            )

        assert "[STT] Session connected after 1 retry" in caplog.messages
        assert not any("Opening new session" in message for message in caplog.messages)
        assert not any("Retrying session in" in message for message in caplog.messages)
    finally:
        await stt.close()


async def test_managed_stt_provider_final_after_next_speech_start_uses_ended_utterance() -> None:
    backend = Float32Backend()
    stt = ManagedSTTProvider(backend=backend, sample_rate_hz=16000, reset_deadline_s=90.0)

    first_utterance_id = uuid4()
    second_utterance_id = uuid4()
    stream = stt.events()

    try:
        await stt.handle_vad_event(
            SpeechStart(
                first_utterance_id,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(1.0),
            )
        )
        await _next_state(stream, STTSessionState.STREAMING)
        await stt.handle_vad_event(SpeechEnd(first_utterance_id))

        await stt.handle_vad_event(
            SpeechStart(
                second_utterance_id,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(0.5),
            )
        )
        await backend.sessions[0]._queue.put(
            STTBackendTranscriptEvent(text="first final", is_final=True)
        )

        event = await _next_typed_event(stream, STTFinalEvent)

        assert event.utterance_id == first_utterance_id
        assert event.transcript.utterance_id == first_utterance_id
        assert event.transcript.text == "first final"
    finally:
        await stt.close()


async def test_managed_stt_provider_multiple_pending_finals_resolve_fifo() -> None:
    backend = Float32Backend()
    stt = ManagedSTTProvider(backend=backend, sample_rate_hz=16000, reset_deadline_s=90.0)

    first_utterance_id = uuid4()
    second_utterance_id = uuid4()
    stream = stt.events()

    try:
        await stt.handle_vad_event(
            SpeechStart(
                first_utterance_id,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(1.0),
            )
        )
        await _next_state(stream, STTSessionState.STREAMING)
        await stt.handle_vad_event(SpeechEnd(first_utterance_id))
        await stt.handle_vad_event(
            SpeechStart(
                second_utterance_id,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(0.5),
            )
        )
        await stt.handle_vad_event(SpeechEnd(second_utterance_id))

        await backend.sessions[0]._queue.put(
            STTBackendTranscriptEvent(text="first final", is_final=True)
        )
        await backend.sessions[0]._queue.put(
            STTBackendTranscriptEvent(text="second final", is_final=True)
        )

        first_event = await _next_typed_event(stream, STTFinalEvent)
        second_event = await _next_typed_event(stream, STTFinalEvent)

        assert [first_event.utterance_id, second_event.utterance_id] == [
            first_utterance_id,
            second_utterance_id,
        ]
        assert [first_event.transcript.text, second_event.transcript.text] == [
            "first final",
            "second final",
        ]
    finally:
        await stt.close()


async def test_managed_stt_provider_emits_later_session_final_without_earlier_boundary() -> None:
    backend = Float32Backend()
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        reset_deadline_s=0.1,
        reconnect_window_s=0.5,
        drain_timeout_s=0.05,
        finalize_grace_s=0.0,
    )

    first_utterance_id = uuid4()
    second_utterance_id = uuid4()
    stream = stt.events()

    try:
        await stt.handle_vad_event(
            SpeechStart(
                first_utterance_id,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(1.0),
            )
        )
        await _next_state(stream, STTSessionState.STREAMING)
        await stt.handle_vad_event(SpeechEnd(first_utterance_id))

        await asyncio.sleep(0.15)
        assert len(backend.sessions) == 2

        await stt.handle_vad_event(
            SpeechStart(
                second_utterance_id,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(0.5),
            )
        )
        await stt.handle_vad_event(SpeechEnd(second_utterance_id))

        await backend.sessions[1]._queue.put(
            STTBackendTranscriptEvent(text="second final", is_final=True)
        )

        event = await _next_typed_event(stream, STTFinalEvent)
        assert event.transcript.text == "second final"
        assert event.transcript.is_final is True
    finally:
        await stt.close()


async def test_managed_stt_provider_emits_delayed_finalization_lag_diagnostic() -> None:
    backend = Float32Backend()
    clock = FakeClock(10.0)
    runtime_logging, log_stream = _make_runtime_logging_capture()
    runtime_logging.set_mode(SessionLoggingMode.DETAILED)
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        clock=clock,
        reconnect_window_s=0.1,
        reset_deadline_s=90.0,
        stt_provider_name=STTProviderName.SONIOX,
        runtime_logging=runtime_logging,
    )

    utterance_id = uuid4()
    stream = stt.events()

    try:
        await stt.handle_vad_event(
            SpeechStart(
                utterance_id,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(1.0),
            )
        )
        await _next_state(stream, STTSessionState.STREAMING)
        await stt.handle_vad_event(SpeechEnd(utterance_id))
        clock.advance(0.25)

        await backend.sessions[0]._queue.put(
            STTBackendTranscriptEvent(text="safe final text must not appear", is_final=True)
        )
        event = await _next_typed_event(stream, STTFinalEvent)

        messages = _runtime_log_messages(log_stream)
        lag_message = next(message for message in messages if "[STT][FinalizationLag]" in message)

        assert event.utterance_id == utterance_id
        assert "channel=self" in lag_message
        assert "provider=soniox" in lag_message
        assert f"utterance_id={str(utterance_id)[:8]}" in lag_message
        assert "pending_ms=250" in lag_message
        assert "threshold_ms=100" in lag_message
        assert "dominant_stage=stt_finalization_pending" in lag_message
        assert "speech_end_to_stt_final_ms=250" in lag_message
        assert "safe final text" not in lag_message
    finally:
        await stt.close()
        runtime_logging.close()


async def test_managed_stt_provider_does_not_emit_normal_finalization_lag() -> None:
    backend = Float32Backend()
    clock = FakeClock(10.0)
    runtime_logging, log_stream = _make_runtime_logging_capture()
    runtime_logging.set_mode(SessionLoggingMode.DETAILED)
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        clock=clock,
        reconnect_window_s=0.5,
        reset_deadline_s=90.0,
        runtime_logging=runtime_logging,
    )

    utterance_id = uuid4()
    stream = stt.events()

    try:
        await stt.handle_vad_event(
            SpeechStart(
                utterance_id,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(1.0),
            )
        )
        await _next_state(stream, STTSessionState.STREAMING)
        await stt.handle_vad_event(SpeechEnd(utterance_id))
        clock.advance(0.1)

        await backend.sessions[0]._queue.put(
            STTBackendTranscriptEvent(text="normal final", is_final=True)
        )
        event = await _next_typed_event(stream, STTFinalEvent)

        assert event.utterance_id == utterance_id
        assert not any(
            "[STT][FinalizationLag]" in message for message in _runtime_log_messages(log_stream)
        )
    finally:
        await stt.close()
        runtime_logging.close()


async def test_managed_stt_provider_empty_final_boundary_consumes_pending_id_before_next_final() -> (
    None
):
    backend = Float32Backend()
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        reset_deadline_s=90.0,
    )

    empty_utterance_id = uuid4()
    next_utterance_id = uuid4()
    stream = stt.events()

    try:
        await stt.handle_vad_event(
            SpeechStart(
                empty_utterance_id,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(1.0),
            )
        )
        await _next_state(stream, STTSessionState.STREAMING)
        await stt.handle_vad_event(SpeechEnd(empty_utterance_id))
        await backend.sessions[0]._queue.put(STTBackendTranscriptEvent(text="", is_final=True))

        await stt.handle_vad_event(
            SpeechStart(
                next_utterance_id,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(0.5),
            )
        )
        await stt.handle_vad_event(SpeechEnd(next_utterance_id))
        await backend.sessions[0]._queue.put(
            STTBackendTranscriptEvent(text="next final", is_final=True)
        )

        event = await _next_typed_event(stream, STTFinalEvent)

        assert event.utterance_id == next_utterance_id
        assert event.transcript.utterance_id == next_utterance_id
        assert event.transcript.text == "next final"
    finally:
        await stt.close()


async def test_local_qwen_empty_decode_keeps_next_final_on_next_utterance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = iter(["", "next final"])

    async def ensure_recognizer(self) -> object:
        self._recognizer = object()
        return self._recognizer

    async def decode_f32(self, samples_f32: np.ndarray) -> str:
        _ = samples_f32
        return next(results)

    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "_ensure_recognizer", ensure_recognizer)
    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "decode_f32", decode_f32)

    backend = LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen"))
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        stt_provider_name=STTProviderName.LOCAL_QWEN,
        reset_deadline_s=90.0,
    )
    empty_utterance_id = uuid4()
    next_utterance_id = uuid4()
    stream = stt.events()

    try:
        await stt.handle_vad_event(
            SpeechStart(
                empty_utterance_id,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(1.0),
            )
        )
        await _next_state(stream, STTSessionState.STREAMING)
        await stt.handle_vad_event(SpeechEnd(empty_utterance_id))
        await stt.handle_vad_event(
            SpeechStart(
                next_utterance_id,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(0.5),
            )
        )
        await stt.handle_vad_event(SpeechEnd(next_utterance_id))

        event = await _next_typed_event(stream, STTFinalEvent)

        assert event.utterance_id == next_utterance_id
        assert event.transcript.utterance_id == next_utterance_id
        assert event.transcript.text == "next final"
    finally:
        await stt.close()


async def test_local_qwen_bridging_reset_preserves_final_id_fifo_across_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    decode_count = 0

    async def ensure_recognizer(self) -> object:
        self._recognizer = object()
        return self._recognizer

    async def decode_f32(self, samples_f32: np.ndarray) -> str:
        nonlocal decode_count
        _ = samples_f32
        decode_count += 1
        sequence = decode_count
        if sequence == 1:
            first_started.set()
            await release_first.wait()
        return f"final-{sequence}"

    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "_ensure_recognizer", ensure_recognizer)
    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "decode_f32", decode_f32)

    backend = LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen"))
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        stt_provider_name=STTProviderName.LOCAL_QWEN,
        reset_deadline_s=90.0,
        drain_timeout_s=1.0,
        bridging_ms=64,
        finalize_grace_s=0.0,
    )
    utterance_ids = [uuid4(), uuid4(), uuid4()]
    stream = stt.events()

    try:
        await stt.handle_vad_event(
            SpeechStart(
                utterance_ids[0],
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(1.0),
            )
        )
        await _next_state(stream, STTSessionState.STREAMING)
        await stt.handle_vad_event(SpeechEnd(utterance_ids[0]))
        await asyncio.wait_for(first_started.wait(), timeout=0.1)

        await stt.handle_vad_event(
            SpeechStart(
                utterance_ids[1],
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(0.5),
            )
        )
        await stt.handle_vad_event(SpeechEnd(utterance_ids[1]))
        await stt.handle_vad_event(
            SpeechStart(
                utterance_ids[2],
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(0.25),
            )
        )

        await stt._reset_with_bridging()
        await stt.handle_vad_event(SpeechEnd(utterance_ids[2]))
        await asyncio.sleep(0)

        assert decode_count == 1

        release_first.set()
        final_events = [
            await _next_typed_event(stream, STTFinalEvent),
            await _next_typed_event(stream, STTFinalEvent),
            await _next_typed_event(stream, STTFinalEvent),
        ]

        assert [event.utterance_id for event in final_events] == utterance_ids
        assert [event.transcript.text for event in final_events] == [
            "final-1",
            "final-2",
            "final-3",
        ]
    finally:
        release_first.set()
        await stt.close()


async def test_local_qwen_bridging_reset_retires_failed_old_session_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    decode_count = 0

    async def ensure_recognizer(self) -> object:
        self._recognizer = object()
        return self._recognizer

    async def decode_f32(self, samples_f32: np.ndarray) -> str:
        nonlocal decode_count
        _ = samples_f32
        decode_count += 1
        if decode_count == 1:
            first_started.set()
            await release_first.wait()
            raise RuntimeError("old decode failed")
        return "new final"

    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "_ensure_recognizer", ensure_recognizer)
    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "decode_f32", decode_f32)

    backend = LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen"))
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        stt_provider_name=STTProviderName.LOCAL_QWEN,
        reset_deadline_s=90.0,
        drain_timeout_s=1.0,
        bridging_ms=64,
        finalize_grace_s=0.0,
    )
    utterance_ids = [uuid4(), uuid4(), uuid4()]
    stream = stt.events()

    try:
        await stt.handle_vad_event(
            SpeechStart(
                utterance_ids[0],
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(1.0),
            )
        )
        await _next_state(stream, STTSessionState.STREAMING)
        await stt.handle_vad_event(SpeechEnd(utterance_ids[0]))
        await asyncio.wait_for(first_started.wait(), timeout=0.1)

        await stt.handle_vad_event(
            SpeechStart(
                utterance_ids[1],
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(0.5),
            )
        )
        await stt.handle_vad_event(SpeechEnd(utterance_ids[1]))
        await stt.handle_vad_event(
            SpeechStart(
                utterance_ids[2],
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(0.25),
            )
        )

        await stt._reset_with_bridging()
        await stt.handle_vad_event(SpeechEnd(utterance_ids[2]))
        release_first.set()

        event = await _next_typed_event(stream, STTFinalEvent)

        assert decode_count == 2
        assert event.utterance_id == utterance_ids[2]
        assert event.transcript.utterance_id == utterance_ids[2]
        assert event.transcript.text == "new final"
    finally:
        release_first.set()
        await stt.close()


async def test_local_qwen_provider_close_bounds_decode_and_reopen_maps_new_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()
    decode_count = 0

    async def ensure_recognizer(self) -> object:
        self._recognizer = object()
        return self._recognizer

    async def decode_f32(self, samples_f32: np.ndarray) -> str:
        nonlocal decode_count
        _ = samples_f32
        decode_count += 1
        if decode_count == 1:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        return "new final"

    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "_ensure_recognizer", ensure_recognizer)
    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "decode_f32", decode_f32)

    stt = ManagedSTTProvider(
        backend=LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen")),
        sample_rate_hz=16000,
        stt_provider_name=STTProviderName.LOCAL_QWEN,
        reset_deadline_s=90.0,
        drain_timeout_s=0.02,
        finalize_grace_s=0.0,
    )
    canceled_utterance_id = uuid4()
    new_utterance_id = uuid4()
    stream = stt.events()

    await stt.handle_vad_event(
        SpeechStart(
            canceled_utterance_id,
            pre_roll=np.zeros(0, dtype=np.float32),
            chunk=samples(1.0),
        )
    )
    await _next_state(stream, STTSessionState.STREAMING)
    await stt.handle_vad_event(SpeechEnd(canceled_utterance_id))
    await asyncio.wait_for(started.wait(), timeout=0.1)

    await asyncio.wait_for(stt.close(), timeout=0.5)

    assert cancelled.is_set()
    assert stt.state == STTSessionState.DISCONNECTED
    assert stt._active_utterance_id is None
    assert list(stt._pending_final_utterance_ids) == []
    assert stt._pending_final_utterance_times == {}

    try:
        await stt.handle_vad_event(
            SpeechStart(
                new_utterance_id,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(0.5),
            )
        )
        await _next_state(stream, STTSessionState.STREAMING)
        await stt.handle_vad_event(SpeechEnd(new_utterance_id))

        event = await _next_typed_event(stream, STTFinalEvent)

        assert event.utterance_id == new_utterance_id
        assert event.transcript.utterance_id == new_utterance_id
        assert event.transcript.text == "new final"
    finally:
        await stt.close()


async def test_local_qwen_cancelled_close_is_retryable_and_reopens_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()
    decode_count = 0

    async def ensure_recognizer(self) -> object:
        self._recognizer = object()
        return self._recognizer

    async def decode_f32(self, samples_f32: np.ndarray) -> str:
        nonlocal decode_count
        _ = samples_f32
        decode_count += 1
        if decode_count == 1:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        return "new final"

    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "_ensure_recognizer", ensure_recognizer)
    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "decode_f32", decode_f32)

    stt = ManagedSTTProvider(
        backend=LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen")),
        sample_rate_hz=16000,
        stt_provider_name=STTProviderName.LOCAL_QWEN,
        reset_deadline_s=90.0,
        drain_timeout_s=1.0,
        finalize_grace_s=0.0,
    )
    canceled_utterance_id = uuid4()
    new_utterance_id = uuid4()
    stream = stt.events()

    await stt.handle_vad_event(
        SpeechStart(
            canceled_utterance_id,
            pre_roll=np.zeros(0, dtype=np.float32),
            chunk=samples(1.0),
        )
    )
    await _next_state(stream, STTSessionState.STREAMING)
    await stt.handle_vad_event(SpeechEnd(canceled_utterance_id))
    await asyncio.wait_for(started.wait(), timeout=0.1)
    session = stt._active_session
    assert session is not None

    close_task = asyncio.create_task(stt.close())
    for _ in range(100):
        if not session._decode_coordinator.accepting:
            break
        await asyncio.sleep(0)
    assert session._decode_coordinator.accepting is False
    close_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_task

    assert cancelled.is_set()
    assert stt._active_session is None
    assert stt._consumer_task is None
    assert stt._active_utterance_id is None
    assert list(stt._pending_final_utterance_ids) == []
    assert stt._pending_final_utterance_times == {}
    assert stt._closing is False
    assert session._decode_coordinator._worker_task is not None
    assert session._decode_coordinator._worker_task.done()

    await asyncio.wait_for(stt.close(), timeout=0.2)

    try:
        await stt.handle_vad_event(
            SpeechStart(
                new_utterance_id,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(0.5),
            )
        )
        await _next_state(stream, STTSessionState.STREAMING)
        await stt.handle_vad_event(SpeechEnd(new_utterance_id))

        event = await _next_typed_event(stream, STTFinalEvent)

        assert event.utterance_id == new_utterance_id
        assert event.transcript.utterance_id == new_utterance_id
        assert event.transcript.text == "new final"
    finally:
        await stt.close()


async def test_managed_stt_repeated_close_cancellation_completes_owned_cleanup() -> None:
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def slow_draining_cleanup() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup_started.set()
            while not release_cleanup.is_set():
                try:
                    await release_cleanup.wait()
                except asyncio.CancelledError:
                    continue

    stt = ManagedSTTProvider(
        backend=FakeBackend(),
        sample_rate_hz=16000,
        reset_deadline_s=90.0,
    )
    stale_utterance_id = uuid4()
    stt._active_utterance_id = stale_utterance_id
    stt._pending_final_utterance_ids.append(stale_utterance_id)
    stt._pending_final_utterance_times[stale_utterance_id] = 1.0
    draining_task = asyncio.create_task(slow_draining_cleanup())
    stt._draining.add(draining_task)

    close_task = asyncio.create_task(stt.close())
    await asyncio.wait_for(cleanup_started.wait(), timeout=0.1)
    close_task.cancel()
    await asyncio.sleep(0)
    close_task.cancel()
    release_cleanup.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(close_task, timeout=0.5)

    assert draining_task.done()
    assert close_task.cancelling() == 2
    assert stt._draining == set()
    assert stt._active_session is None
    assert stt._consumer_task is None
    assert stt._active_utterance_id is None
    assert list(stt._pending_final_utterance_ids) == []
    assert stt._pending_final_utterance_times == {}
    assert stt._closing is False


async def test_managed_stt_contains_provider_originated_close_cancellation() -> None:
    class CancelCloseSession:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            raise asyncio.CancelledError

    stt = ManagedSTTProvider(
        backend=FakeBackend(),
        sample_rate_hz=16000,
        reset_deadline_s=90.0,
    )
    session = CancelCloseSession()
    stt._active_session = session

    await asyncio.wait_for(stt.close(), timeout=0.2)

    assert session.close_calls == 2
    assert stt._active_session is None
    assert stt._closing is False
    assert stt.state == STTSessionState.DISCONNECTED


async def test_managed_stt_propagates_caller_cancellation_during_session_close() -> None:
    class BlockingCloseSession:
        def __init__(self) -> None:
            self.close_calls = 0
            self.close_started = asyncio.Event()
            self.events_queue: asyncio.Queue[STTBackendTranscriptEvent | None] = asyncio.Queue()

        async def stop(self) -> None:
            await self.events_queue.put(None)

        async def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                self.close_started.set()
                await asyncio.Event().wait()

        async def events(self):
            while True:
                event = await self.events_queue.get()
                if event is None:
                    return
                yield event

    stt = ManagedSTTProvider(
        backend=FakeBackend(),
        sample_rate_hz=16000,
        reset_deadline_s=90.0,
    )
    session = BlockingCloseSession()
    consumer_task = asyncio.create_task(stt._consume_session_events(session))
    stt._active_session = session
    stt._consumer_task = consumer_task

    close_task = asyncio.create_task(stt.close())
    await asyncio.wait_for(session.close_started.wait(), timeout=0.1)
    close_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(close_task, timeout=0.5)

    assert close_task.cancelling() == 1
    assert session.close_calls == 2
    assert consumer_task.done()
    assert stt._active_session is None
    assert stt._consumer_task is None
    assert stt._closing is False
    assert stt.state == STTSessionState.DISCONNECTED


async def test_local_qwen_provider_close_cancels_handoff_waiting_successor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_started = asyncio.Event()
    first_cancelled = asyncio.Event()

    async def ensure_recognizer(self) -> object:
        self._recognizer = object()
        return self._recognizer

    async def decode_f32(self, samples_f32: np.ndarray) -> str:
        _ = samples_f32
        first_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            first_cancelled.set()
        return ""

    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "_ensure_recognizer", ensure_recognizer)
    monkeypatch.setattr(LocalQwenSherpaSTTBackend, "decode_f32", decode_f32)

    backend = LocalQwenSherpaSTTBackend(model_dir=Path("/models/qwen"))
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        stt_provider_name=STTProviderName.LOCAL_QWEN,
        reset_deadline_s=90.0,
        drain_timeout_s=0.02,
        bridging_ms=64,
        finalize_grace_s=0.0,
    )
    old_utterance_id = uuid4()
    new_utterance_id = uuid4()
    stream = stt.events()

    await stt.handle_vad_event(
        SpeechStart(
            old_utterance_id,
            pre_roll=np.zeros(0, dtype=np.float32),
            chunk=samples(1.0),
        )
    )
    await _next_state(stream, STTSessionState.STREAMING)
    await stt.handle_vad_event(SpeechEnd(old_utterance_id))
    await asyncio.wait_for(first_started.wait(), timeout=0.1)
    old_session = stt._active_session
    await stt.handle_vad_event(
        SpeechStart(
            new_utterance_id,
            pre_roll=np.zeros(0, dtype=np.float32),
            chunk=samples(0.5),
        )
    )
    await stt._reset_with_bridging()
    await stt.handle_vad_event(SpeechEnd(new_utterance_id))
    await asyncio.sleep(0)

    active_session = stt._active_session

    await asyncio.wait_for(stt.close(), timeout=0.5)

    assert first_cancelled.is_set()
    assert active_session is not None
    assert active_session._decode_coordinator._worker_task is not None
    assert active_session._decode_coordinator._worker_task.done()
    assert old_session is not None
    assert old_session._decode_coordinator._worker_task is not None
    assert old_session._decode_coordinator._worker_task.done()


async def test_managed_stt_provider_drops_stale_pending_final_before_later_final() -> None:
    backend = Float32Backend()
    clock = FakeClock(10.0)
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        clock=clock,
        reconnect_window_s=20.0,
        reset_deadline_s=90.0,
    )

    stale_utterance_id = uuid4()
    current_utterance_id = uuid4()
    stream = stt.events()

    try:
        await stt.handle_vad_event(
            SpeechStart(
                stale_utterance_id,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(1.0),
            )
        )
        await _next_state(stream, STTSessionState.STREAMING)
        await stt.handle_vad_event(SpeechEnd(stale_utterance_id))

        clock.advance(25.0)

        await stt.handle_vad_event(
            SpeechStart(
                current_utterance_id,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(0.5),
            )
        )
        await stt.handle_vad_event(SpeechEnd(current_utterance_id))

        await backend.sessions[0]._queue.put(
            STTBackendTranscriptEvent(text="current final", is_final=True)
        )

        event = await _next_typed_event(stream, STTFinalEvent)

        assert event.utterance_id == current_utterance_id
        assert event.transcript.utterance_id == current_utterance_id
        assert event.transcript.text == "current final"
    finally:
        await stt.close()


async def test_managed_stt_provider_repeated_forced_boundaries_reuse_session_and_finalize_fifo() -> (
    None
):
    backend = Float32Backend()
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        channel="peer",
        reset_deadline_s=90.0,
    )

    first_utterance_id = uuid4()
    second_utterance_id = uuid4()
    stream = stt.events()

    try:
        await stt.handle_vad_event(
            SpeechStart(
                first_utterance_id,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(1.0),
            )
        )
        await _next_state(stream, STTSessionState.STREAMING)
        await stt.handle_vad_event(
            SpeechEnd(first_utterance_id, trailing_silence_ms=0, reason="max_duration")
        )

        await stt.handle_vad_event(
            SpeechStart(
                second_utterance_id,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(0.5),
            )
        )
        await stt.handle_vad_event(
            SpeechEnd(second_utterance_id, trailing_silence_ms=0, reason="max_duration")
        )

        assert len(backend.sessions) == 1
        session = backend.sessions[0]
        assert session.calls == ["on_speech_end", "on_speech_end"]
        assert session.speech_ends == [
            (0, "max_duration"),
            (0, "max_duration"),
        ]
        assert len(session.audio_f32) == 2

        await session._queue.put(STTBackendTranscriptEvent(text="first forced", is_final=True))
        await session._queue.put(STTBackendTranscriptEvent(text="second forced", is_final=True))

        first_event = await _next_typed_event(stream, STTFinalEvent)
        second_event = await _next_typed_event(stream, STTFinalEvent)

        assert [first_event.utterance_id, second_event.utterance_id] == [
            first_utterance_id,
            second_utterance_id,
        ]
        assert [first_event.transcript.text, second_event.transcript.text] == [
            "first forced",
            "second forced",
        ]
    finally:
        await stt.close()


async def test_managed_stt_provider_partials_do_not_consume_pending_finals() -> None:
    backend = Float32Backend()
    stt = ManagedSTTProvider(backend=backend, sample_rate_hz=16000, reset_deadline_s=90.0)

    ended_utterance_id = uuid4()
    active_utterance_id = uuid4()
    stream = stt.events()

    try:
        await stt.handle_vad_event(
            SpeechStart(
                ended_utterance_id,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(1.0),
            )
        )
        await _next_state(stream, STTSessionState.STREAMING)
        await stt.handle_vad_event(SpeechEnd(ended_utterance_id))
        await stt.handle_vad_event(
            SpeechStart(
                active_utterance_id,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(0.5),
            )
        )

        await backend.sessions[0]._queue.put(
            STTBackendTranscriptEvent(text="active partial", is_final=False)
        )
        await backend.sessions[0]._queue.put(
            STTBackendTranscriptEvent(text="ended final", is_final=True)
        )

        partial_event = await _next_typed_event(stream, STTPartialEvent)
        final_event = await _next_typed_event(stream, STTFinalEvent)

        assert partial_event.utterance_id == active_utterance_id
        assert partial_event.transcript.text == "active partial"
        assert final_event.utterance_id == ended_utterance_id
        assert final_event.transcript.text == "ended final"
    finally:
        await stt.close()


async def test_managed_stt_provider_final_without_pending_uses_active_fallback() -> None:
    backend = Float32Backend()
    stt = ManagedSTTProvider(backend=backend, sample_rate_hz=16000, reset_deadline_s=90.0)

    active_utterance_id = uuid4()
    stream = stt.events()

    try:
        await stt.handle_vad_event(
            SpeechStart(
                active_utterance_id,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(1.0),
            )
        )
        await _next_state(stream, STTSessionState.STREAMING)
        await backend.sessions[0]._queue.put(
            STTBackendTranscriptEvent(text="active final", is_final=True)
        )

        event = await _next_typed_event(stream, STTFinalEvent)

        assert event.utterance_id == active_utterance_id
        assert event.transcript.utterance_id == active_utterance_id
        assert event.transcript.text == "active final"
    finally:
        await stt.close()


async def test_managed_stt_provider_bridging_reset_preserves_pending_final() -> None:
    backend = StopFinalizingBackend(first_stop_final_text="drained final")
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        reset_deadline_s=90.0,
        drain_timeout_s=0.2,
        bridging_ms=64,
        finalize_grace_s=0.0,
    )

    pending_utterance_id = uuid4()
    active_utterance_id = uuid4()
    stream = stt.events()

    try:
        await stt.handle_vad_event(
            SpeechStart(
                pending_utterance_id,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(1.0),
            )
        )
        await _next_state(stream, STTSessionState.STREAMING)
        await stt.handle_vad_event(SpeechEnd(pending_utterance_id))
        await stt.handle_vad_event(
            SpeechStart(
                active_utterance_id,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(0.5),
            )
        )

        await stt._reset_with_bridging()

        event = await _next_typed_event(stream, STTFinalEvent)

        assert len(backend.sessions) == 2
        assert event.utterance_id == pending_utterance_id
        assert event.transcript.text == "drained final"
    finally:
        await stt.close()


async def test_managed_stt_provider_peer_channel_produces_final_event():
    provider = ManagedSTTProvider(
        backend=FakeBackend(),
        sample_rate_hz=16000,
        channel="peer",
    )
    utterance_id = uuid4()
    provider._pending_final_utterance_ids.append(utterance_id)

    await provider._consume_session_events(
        EventOnlySession(
            items=[
                STTBackendTranscriptEvent(
                    text="peer line",
                    is_final=True,
                )
            ]
        ),
    )

    event = await _next_event(provider.events())
    assert isinstance(event, STTFinalEvent)
    assert event.transcript.channel == "peer"
    assert event.transcript.text == "peer line"


async def test_managed_stt_provider_preserves_provider_neutral_final_language_runs():
    provider = ManagedSTTProvider(
        backend=FakeBackend(),
        sample_rate_hz=16000,
        channel="peer",
    )
    utterance_id = uuid4()
    provider._pending_final_utterance_ids.append(utterance_id)
    runs = (
        FinalLanguageRun(text="日本語", language="ja"),
        FinalLanguageRun(text="中文", language="zh"),
    )

    await provider._consume_session_events(
        EventOnlySession(
            items=[
                STTBackendTranscriptEvent(
                    text="日本語中文",
                    is_final=True,
                    final_language_runs=runs,
                )
            ]
        )
    )

    event = await _next_event(provider.events())
    assert isinstance(event, STTFinalEvent)
    assert event.transcript.final_language_runs == runs


@pytest.mark.parametrize(
    ("channel", "text"),
    [("self", "leşme"), ("peer", "acia")],
)
async def test_managed_stt_provider_suppresses_known_local_qwen_final_and_notifies_without_text(
    channel,
    text,
) -> None:
    runtime_logging, log_stream = _make_runtime_logging_capture()
    notifications: list[object] = []
    provider = ManagedSTTProvider(
        backend=FakeBackend(),
        sample_rate_hz=16000,
        channel=channel,
        stt_provider_name=STTProviderName.LOCAL_QWEN,
        runtime_logging=runtime_logging,
        on_final_transcript_suppressed=notifications.append,
    )
    utterance_id = uuid4()
    provider._pending_final_utterance_ids.append(utterance_id)
    provider._pending_final_utterance_times[utterance_id] = 10.0

    await provider._consume_session_events(
        EventOnlySession([STTBackendTranscriptEvent(text=text, is_final=True)])
    )

    assert provider._events.empty()
    assert list(provider._pending_final_utterance_ids) == []
    assert provider._pending_final_utterance_times == {}
    assert len(notifications) == 1
    notification = notifications[0]
    assert getattr(notification, "utterance_id") == utterance_id
    assert getattr(notification, "channel") == channel
    assert getattr(notification, "stt_provider_name") == STTProviderName.LOCAL_QWEN
    assert not hasattr(notification, "text")
    assert not hasattr(notification, "transcript")

    messages = _runtime_log_messages(log_stream)
    assert any(
        f"[STT][local_qwen][{channel}] Known hallucination suppressed" in message
        and f"utterance_id={str(utterance_id)[:8]}" in message
        and "notification=emitted" in message
        for message in messages
    )
    assert not any(text in message for message in messages)
    assert not any("text=" in message for message in messages)


async def test_managed_stt_provider_suppression_log_marks_missing_notification_callback_without_text() -> (
    None
):
    runtime_logging, log_stream = _make_runtime_logging_capture()
    provider = ManagedSTTProvider(
        backend=FakeBackend(),
        sample_rate_hz=16000,
        stt_provider_name=STTProviderName.LOCAL_QWEN,
        runtime_logging=runtime_logging,
    )
    utterance_id = uuid4()
    provider._pending_final_utterance_ids.append(utterance_id)

    await provider._consume_session_events(
        EventOnlySession([STTBackendTranscriptEvent(text="leşme", is_final=True)])
    )

    messages = _runtime_log_messages(log_stream)
    assert provider._events.empty()
    assert any(
        "[STT][local_qwen][self] Known hallucination suppressed" in message
        and f"utterance_id={str(utterance_id)[:8]}" in message
        and "notification=not_configured" in message
        for message in messages
    )
    assert not any("leşme" in message for message in messages)
    assert not any("text=" in message for message in messages)


async def test_managed_stt_provider_suppression_log_marks_notification_failure_without_text() -> (
    None
):
    runtime_logging, log_stream = _make_runtime_logging_capture()

    def fail_notification(_notification: object) -> None:
        raise RuntimeError("counter unavailable")

    provider = ManagedSTTProvider(
        backend=FakeBackend(),
        sample_rate_hz=16000,
        stt_provider_name=STTProviderName.LOCAL_QWEN,
        runtime_logging=runtime_logging,
        on_final_transcript_suppressed=fail_notification,
    )
    utterance_id = uuid4()
    provider._pending_final_utterance_ids.append(utterance_id)

    await provider._consume_session_events(
        EventOnlySession([STTBackendTranscriptEvent(text="acia", is_final=True)])
    )

    messages = _runtime_log_messages(log_stream)
    assert provider._events.empty()
    assert any(
        "[STT][local_qwen][self] Known hallucination suppressed" in message
        and f"utterance_id={str(utterance_id)[:8]}" in message
        and "notification=failed" in message
        for message in messages
    )
    assert not any("acia" in message for message in messages)
    assert not any("text=" in message for message in messages)


@pytest.mark.parametrize(
    "stt_provider_name",
    [STTProviderName.DEEPGRAM, STTProviderName.SONIOX, STTProviderName.QWEN_ASR],
)
async def test_managed_stt_provider_allows_known_text_from_non_local_provider_instances(
    stt_provider_name,
) -> None:
    notifications: list[object] = []
    provider = ManagedSTTProvider(
        backend=FakeBackend(),
        sample_rate_hz=16000,
        stt_provider_name=stt_provider_name,
        on_final_transcript_suppressed=notifications.append,
    )
    utterance_id = uuid4()
    provider._pending_final_utterance_ids.append(utterance_id)

    await provider._consume_session_events(
        EventOnlySession([STTBackendTranscriptEvent(text="leşme", is_final=True)])
    )

    event = await _next_event(provider.events())
    assert isinstance(event, STTFinalEvent)
    assert event.utterance_id == utterance_id
    assert event.transcript.text == "leşme"
    assert notifications == []
    assert list(provider._pending_final_utterance_ids) == []


@pytest.mark.parametrize(
    "text",
    ["的答案", "虚构", "夫", "夫夫", "格力", "Leşme", "xleşmex", "AcIa", "acia."],
)
async def test_managed_stt_provider_allows_non_matching_local_qwen_finals(text: str) -> None:
    notifications: list[object] = []
    provider = ManagedSTTProvider(
        backend=FakeBackend(),
        sample_rate_hz=16000,
        stt_provider_name=STTProviderName.LOCAL_QWEN,
        on_final_transcript_suppressed=notifications.append,
    )
    utterance_id = uuid4()
    provider._pending_final_utterance_ids.append(utterance_id)

    await provider._consume_session_events(
        EventOnlySession([STTBackendTranscriptEvent(text=text, is_final=True)])
    )

    event = await _next_event(provider.events())
    assert isinstance(event, STTFinalEvent)
    assert event.transcript.text == text
    assert notifications == []


async def test_managed_stt_provider_suppression_decision_uses_producer_instance_identity() -> None:
    local_notifications: list[object] = []
    local_provider = ManagedSTTProvider(
        backend=FakeBackend(),
        sample_rate_hz=16000,
        stt_provider_name=STTProviderName.LOCAL_QWEN,
        on_final_transcript_suppressed=local_notifications.append,
    )
    local_id = uuid4()
    local_provider._pending_final_utterance_ids.append(local_id)

    non_local_notifications: list[object] = []
    non_local_provider = ManagedSTTProvider(
        backend=FakeBackend(),
        sample_rate_hz=16000,
        stt_provider_name=STTProviderName.DEEPGRAM,
        on_final_transcript_suppressed=non_local_notifications.append,
    )
    non_local_id = uuid4()
    non_local_provider._pending_final_utterance_ids.append(non_local_id)

    await local_provider._consume_session_events(
        EventOnlySession([STTBackendTranscriptEvent(text="leşme", is_final=True)])
    )
    await non_local_provider._consume_session_events(
        EventOnlySession([STTBackendTranscriptEvent(text="leşme", is_final=True)])
    )

    assert local_provider._events.empty()
    assert getattr(local_notifications[0], "stt_provider_name") == STTProviderName.LOCAL_QWEN
    non_local_event = await _next_event(non_local_provider.events())
    assert isinstance(non_local_event, STTFinalEvent)
    assert non_local_event.utterance_id == non_local_id
    assert non_local_event.transcript.text == "leşme"
    assert non_local_notifications == []


async def test_managed_stt_provider_skips_empty_audio_send() -> None:
    session = FakeSession()
    backend = EventOnlyBackend(session=session)
    stt = ManagedSTTProvider(backend=backend, sample_rate_hz=16000, channel="peer")

    uid = uuid4()
    await stt.handle_vad_event(
        SpeechStart(uid, pre_roll=np.zeros(0, dtype=np.float32), chunk=samples(1.0))
    )

    assert b"" not in session.audio


async def test_managed_stt_provider_invokes_terminal_failure_callback_after_consumer_error() -> (
    None
):
    errors: list[str] = []
    backend = FailingBackend(RuntimeError("closed"))
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        channel="peer",
        connect_attempts=1,
        on_terminal_failure=lambda exc: errors.append(str(exc)),
    )

    uid = uuid4()
    await stt.handle_vad_event(SpeechStart(uid, pre_roll=samples(0.0), chunk=samples(1.0)))
    await asyncio.sleep(0)

    assert stt.state == STTSessionState.DISCONNECTED
    assert stt._active_session is None
    assert errors == ["closed"]


async def test_stt_controller_closes_failed_session_after_consumer_error() -> None:
    backend = TerminalFailureBackend()
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        reset_deadline_s=90.0,
        drain_timeout_s=0.05,
    )

    uid = uuid4()
    stream = stt.events()
    await stt.handle_vad_event(SpeechStart(uid, pre_roll=samples(0.0), chunk=samples(1.0)))
    await _next_state(stream, STTSessionState.STREAMING)

    await asyncio.sleep(0.01)

    assert stt.state == STTSessionState.DISCONNECTED
    assert stt._active_session is None
    assert stt._consumer_task is None
    assert backend.sessions[0].closed is True


async def test_managed_stt_provider_reopens_on_next_speech_after_terminal_failure() -> None:
    backend = TerminalThenHealthyBackend()
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        channel="peer",
        reset_deadline_s=90.0,
        drain_timeout_s=0.05,
    )
    stream = stt.events()

    await stt.handle_vad_event(SpeechStart(uuid4(), pre_roll=samples(0.0), chunk=samples(1.0)))
    await _next_state(stream, STTSessionState.STREAMING)
    await _next_state(stream, STTSessionState.DISCONNECTED, max_events=10)

    await stt.handle_vad_event(SpeechStart(uuid4(), pre_roll=samples(0.0), chunk=samples(1.0)))

    assert len(backend.sessions) == 2
    assert stt.state == STTSessionState.STREAMING
    assert stt._active_session is backend.sessions[1]

    await stt.close()


async def test_stt_suppresses_error_event_during_close() -> None:
    stt = ManagedSTTProvider(
        backend=TerminalFailureBackend(),
        sample_rate_hz=16000,
        channel="peer",
        connect_attempts=1,
    )
    session = TerminalFailureSession()
    stt._closing = True

    await stt._handle_terminal_session_failure(session, RuntimeError("backend closed"))

    assert stt._events.empty()


async def test_stt_emits_error_event_when_not_closing() -> None:
    stt = ManagedSTTProvider(
        backend=TerminalFailureBackend(),
        sample_rate_hz=16000,
        channel="peer",
        connect_attempts=1,
    )
    session = TerminalFailureSession()

    await stt._handle_terminal_session_failure(session, RuntimeError("backend closed"))

    events: list[object] = []
    while not stt._events.empty():
        events.append(stt._events.get_nowait())
    assert any(isinstance(e, STTErrorEvent) for e in events)
