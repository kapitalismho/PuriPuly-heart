from __future__ import annotations

import asyncio
import io
import logging
from dataclasses import dataclass
from uuid import uuid4

import numpy as np
import pytest

import puripuly_heart.core.stt.controller as stt_controller_module
from puripuly_heart.config.settings import STTProviderName
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.runtime_logging import SessionLoggingMode, SessionRuntimeLoggingService
from puripuly_heart.core.stt.backend import STTBackendTranscriptEvent
from puripuly_heart.core.stt.controller import ManagedSTTProvider
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart
from puripuly_heart.domain.events import (
    STTFinalEvent,
    STTPartialEvent,
    STTSessionState,
    STTSessionStateEvent,
)
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

    async def on_speech_end(self, *, trailing_silence_ms: int | None = None) -> None:
        _ = trailing_silence_ms
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


@dataclass(slots=True)
class Float32Session:
    audio_f32: list[np.ndarray]
    audio_bytes: list[bytes]
    _queue: asyncio.Queue
    calls: list[str]
    _closed: bool = False

    def __init__(self) -> None:
        self.audio_f32 = []
        self.audio_bytes = []
        self._queue = asyncio.Queue()
        self.calls = []

    async def send_audio(self, pcm16le: bytes) -> None:
        self.audio_bytes.append(pcm16le)

    async def send_audio_f32(self, samples_f32: np.ndarray) -> None:
        self.audio_f32.append(np.asarray(samples_f32, dtype=np.float32).copy())

    async def stop(self) -> None:
        self.calls.append("stop")
        await self._queue.put(None)

    async def on_speech_end(self, *, trailing_silence_ms: int | None = None) -> None:
        _ = trailing_silence_ms
        self.calls.append("on_speech_end")

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


@dataclass(slots=True)
class SlowOpenBackend:
    sessions: list[Float32Session]
    open_started: asyncio.Event
    release_open: asyncio.Event
    open_calls: int

    def __init__(self) -> None:
        self.sessions = []
        self.open_started = asyncio.Event()
        self.release_open = asyncio.Event()
        self.open_calls = 0

    async def open_session(self) -> Float32Session:
        self.open_calls += 1
        self.open_started.set()
        await self.release_open.wait()
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

    async def on_speech_end(self, *, trailing_silence_ms: int | None = None) -> None:
        _ = trailing_silence_ms

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

    async def on_speech_end(self, *, trailing_silence_ms: int | None = None) -> None:
        _ = trailing_silence_ms

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
class TerminalFailureSession:
    closed: bool = False
    stopped: bool = False

    async def send_audio(self, pcm16le: bytes) -> None:
        _ = pcm16le

    async def on_speech_end(self, *, trailing_silence_ms: int | None = None) -> None:
        _ = trailing_silence_ms

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


async def test_stt_controller_serializes_concurrent_session_open() -> None:
    backend = SlowOpenBackend()
    stt = ManagedSTTProvider(backend=backend, sample_rate_hz=16000, reset_deadline_s=90.0)

    first = asyncio.create_task(stt._ensure_session())
    second: asyncio.Task[bool] | None = None

    try:
        await asyncio.wait_for(backend.open_started.wait(), timeout=0.2)
        second = asyncio.create_task(stt._ensure_session())
        await asyncio.sleep(0)

        assert backend.open_calls == 1

        backend.release_open.set()
        results = await asyncio.gather(first, second)

        assert results == [True, True]
        assert backend.open_calls == 1
        assert len(backend.sessions) == 1
    finally:
        backend.release_open.set()
        tasks = [task for task in (first, second) if task is not None and not task.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await stt.close()


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


async def test_managed_stt_provider_empty_final_ack_pops_pending_without_output() -> None:
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
        await backend.sessions[0]._queue.put(STTBackendTranscriptEvent(text="", is_final=True))
        await asyncio.sleep(0)

        await stt.handle_vad_event(
            SpeechStart(
                second_utterance_id,
                pre_roll=np.zeros(0, dtype=np.float32),
                chunk=samples(0.5),
            )
        )
        await stt.handle_vad_event(SpeechEnd(second_utterance_id))
        await backend.sessions[0]._queue.put(
            STTBackendTranscriptEvent(text="second final", is_final=True)
        )

        event = await _next_typed_event(stream, STTFinalEvent)

        assert event.utterance_id == second_utterance_id
        assert event.transcript.utterance_id == second_utterance_id
        assert event.transcript.text == "second final"
    finally:
        await stt.close()


async def test_managed_stt_provider_logs_delayed_pending_finalization_lag() -> None:
    backend = Float32Backend()
    clock = FakeClock(10.0)
    runtime_logging, log_stream = _make_runtime_logging_capture()
    runtime_logging.set_mode(SessionLoggingMode.DETAILED)
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        clock=clock,
        stt_provider_name=STTProviderName.SONIOX,
        runtime_logging=runtime_logging,
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
        await stt.handle_vad_event(SpeechEnd(first_utterance_id))
        clock.advance(2.0)
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

        event = await _next_typed_event(stream, STTFinalEvent)

        assert event.utterance_id == first_utterance_id
        messages = _runtime_log_messages(log_stream)
        diagnostic = next(message for message in messages if "[STT][FinalizationLag]" in message)
        assert f"utterance_id={str(first_utterance_id)[:8]}" in diagnostic
        assert "pending_age_ms=2000" in diagnostic
        assert "pending_queue_size=2" in diagnostic
        assert "active_utterance_id=none" in diagnostic
        assert "provider=soniox" in diagnostic
    finally:
        runtime_logging.close()
        await stt.close()


async def test_managed_stt_provider_normal_pending_finalization_does_not_log_lag() -> None:
    backend = Float32Backend()
    clock = FakeClock(10.0)
    runtime_logging, log_stream = _make_runtime_logging_capture()
    runtime_logging.set_mode(SessionLoggingMode.DETAILED)
    stt = ManagedSTTProvider(
        backend=backend,
        sample_rate_hz=16000,
        clock=clock,
        stt_provider_name=STTProviderName.SONIOX,
        runtime_logging=runtime_logging,
        reset_deadline_s=90.0,
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
        await backend.sessions[0]._queue.put(STTBackendTranscriptEvent(text="final", is_final=True))

        event = await _next_typed_event(stream, STTFinalEvent)

        assert event.utterance_id == utterance_id
        assert not any(
            "[STT][FinalizationLag]" in message for message in _runtime_log_messages(log_stream)
        )
    finally:
        runtime_logging.close()
        await stt.close()


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
