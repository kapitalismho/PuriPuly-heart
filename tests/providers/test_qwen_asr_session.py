from __future__ import annotations

import asyncio
import logging
import sys
import threading
import types

import pytest

from puripuly_heart.core.stt.backend import STTBackendTranscriptEvent
from puripuly_heart.providers.stt import qwen_asr as qwen_asr_module
from puripuly_heart.providers.stt.qwen_asr import (
    _COMMIT,
    _END_SESSION,
    _STOP,
    QwenASRRealtimeSTTBackend,
    _QwenASRSession,
)
from tests.helpers.fakes import TargetThread


def _make_session() -> _QwenASRSession:
    return _QwenASRSession(
        api_key="k",
        model="m",
        language="en",
        endpoint="wss://example",
        sample_rate_hz=16000,
        connect_timeout_s=5.0,
    )


@pytest.mark.asyncio
async def test_qwen_backend_open_cancellation_closes_started_session(monkeypatch) -> None:
    started = asyncio.Event()
    aborted = asyncio.Event()
    closed = asyncio.Event()
    sessions = []

    class PartialSession:
        def __init__(self, **_kwargs) -> None:
            self.worker_alive = True
            sessions.append(self)

        async def start(self) -> None:
            started.set()
            await asyncio.Event().wait()

        async def abort_for_toggle_off(self) -> None:
            aborted.set()

        async def close(self) -> None:
            self.worker_alive = False
            closed.set()

    monkeypatch.setattr(qwen_asr_module, "_QwenASRSession", PartialSession)
    backend = QwenASRRealtimeSTTBackend(api_key="k", language="en")
    open_task = asyncio.create_task(backend.open_session())
    await started.wait()

    open_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await open_task

    assert aborted.is_set()
    assert closed.is_set()
    assert sessions and sessions[0].worker_alive is False


@pytest.mark.asyncio
async def test_qwen_asr_session_on_speech_end_enqueues_commit(caplog) -> None:
    session = _make_session()

    with caplog.at_level(logging.INFO):
        await session.on_speech_end(trailing_silence_ms=500, reason="silence")
    commit = session._audio_q.get_nowait()
    assert commit is _COMMIT
    assert "observed_tail_ms=500 injected_padding_ms=0" in caplog.text
    assert "boundary_reason=silence" in caplog.text
    assert "boundary_wait_ms=500" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO):
        await session.on_speech_end(trailing_silence_ms=0, reason="max_duration")
    commit = session._audio_q.get_nowait()

    assert commit is _COMMIT
    assert session._audio_q.empty()
    assert "observed_tail_ms=0 injected_padding_ms=0" in caplog.text
    assert "boundary_reason=max_duration" in caplog.text
    assert "declared_trim_ms=0 boundary_wait_ms=0" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO):
        await session.on_speech_end(trailing_silence_ms=160, reason="soft_pause")
    commit = session._audio_q.get_nowait()

    assert commit is _COMMIT
    assert session._audio_q.empty()
    assert "observed_tail_ms=160 injected_padding_ms=0" in caplog.text
    assert "boundary_reason=soft_pause" in caplog.text
    assert "boundary_wait_ms=160" in caplog.text


@pytest.mark.asyncio
async def test_qwen_asr_session_send_audio_and_stop() -> None:
    session = _make_session()

    await session.send_audio(b"abc")
    assert session._audio_q.get_nowait() == b"abc"

    await session.stop()
    assert session._stopped is True
    assert session._audio_q.get_nowait() is _END_SESSION


@pytest.mark.asyncio
async def test_qwen_asr_session_close_joins_worker_off_event_loop(monkeypatch) -> None:
    session = _make_session()
    joined: list[float] = []
    to_thread_calls: list[tuple[object, tuple[object, ...]]] = []

    class FakeThread:
        def join(self, timeout: float) -> None:
            joined.append(timeout)

    async def fake_to_thread(func, *args):
        to_thread_calls.append((func, args))
        return func(*args)

    session._thread = FakeThread()
    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    await session.close()

    assert to_thread_calls
    assert joined == [5.0]
    assert session._thread is None


@pytest.mark.asyncio
async def test_qwen_asr_session_abort_purges_backlog_and_rejects_late_terminal() -> None:
    session = _make_session()
    session._loop = asyncio.get_running_loop()
    pending = session._register_commit()
    assert pending is not None
    session._audio_q.put_nowait(b"audio")
    session._audio_q.put_nowait(_COMMIT)

    await session.abort_for_toggle_off()
    session._handle_provider_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "late-item",
            "transcript": "late",
        }
    )
    await asyncio.sleep(0)

    assert session._audio_q.get_nowait() is _STOP
    assert session._audio_q.empty()
    assert session._events.empty()
    assert not session._pending_commits


@pytest.mark.asyncio
async def test_qwen_asr_session_empty_completed_and_failed_advance_pending_commits() -> None:
    session = _make_session()
    session._loop = asyncio.get_running_loop()
    assert session._register_commit() is not None
    assert session._register_commit() is not None
    session._handle_provider_event(
        {"type": "input_audio_buffer.committed", "event_id": "e1", "item_id": "i1"}
    )
    session._handle_provider_event(
        {"type": "input_audio_buffer.committed", "event_id": "e2", "item_id": "i2"}
    )

    session._handle_provider_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "i1",
            "transcript": "   ",
        }
    )
    session._handle_provider_event(
        {
            "type": "conversation.item.input_audio_transcription.failed",
            "item_id": "i2",
            "error": {"message": "decode failed"},
        }
    )
    await asyncio.sleep(0)

    first = await session._events.get()
    second = await session._events.get()
    assert first == STTBackendTranscriptEvent(text="", is_final=True)
    assert second == STTBackendTranscriptEvent(text="", is_final=True)
    assert not session._pending_commits


@pytest.mark.asyncio
async def test_qwen_asr_session_buffers_out_of_order_items_and_ignores_duplicate_terminal() -> None:
    session = _make_session()
    session._loop = asyncio.get_running_loop()
    assert session._register_commit() is not None
    assert session._register_commit() is not None
    session._handle_provider_event({"type": "input_audio_buffer.committed", "item_id": "i1"})
    session._handle_provider_event({"type": "input_audio_buffer.committed", "item_id": "i2"})

    session._handle_provider_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "i2",
            "transcript": "second",
        }
    )
    await asyncio.sleep(0)
    assert session._events.empty()
    session._handle_provider_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "i1",
            "transcript": "first",
        }
    )
    session._handle_provider_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "i1",
            "transcript": "duplicate",
        }
    )
    await asyncio.sleep(0)

    assert (await session._events.get()).text == "first"
    assert (await session._events.get()).text == "second"
    assert session._events.empty()


@pytest.mark.asyncio
async def test_qwen_asr_session_itemless_duplicate_event_does_not_consume_next_commit() -> None:
    session = _make_session()
    session._loop = asyncio.get_running_loop()
    assert session._register_commit() is not None
    assert session._register_commit() is not None

    first_terminal = {
        "type": "conversation.item.input_audio_transcription.completed",
        "event_id": "terminal-1",
        "transcript": "first",
    }
    session._handle_provider_event(first_terminal)
    session._handle_provider_event(first_terminal)
    session._handle_provider_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "event_id": "terminal-2",
            "transcript": "second",
        }
    )
    await asyncio.sleep(0)

    assert (await session._events.get()).text == "first"
    assert (await session._events.get()).text == "second"
    assert session._events.empty()
    assert not session._pending_commits


@pytest.mark.asyncio
async def test_qwen_asr_session_commit_failure_fences_session_and_rejects_late_terminal() -> None:
    session = _make_session()
    session._loop = asyncio.get_running_loop()

    class FailingConversation:
        def commit(self) -> None:
            raise RuntimeError("commit failed")

    assert session._send_commit(FailingConversation()) is False
    assert session._register_commit() is None
    session._handle_provider_event(
        {
            "type": "input_audio_buffer.committed",
            "item_id": "late-failed-item",
        }
    )
    session._handle_provider_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "late-failed-item",
            "transcript": "late",
        }
    )
    await asyncio.sleep(0)

    first = await session._events.get()
    second = await session._events.get()
    assert first == STTBackendTranscriptEvent(text="", is_final=True)
    assert isinstance(second, RuntimeError)
    assert str(second) == "Qwen ASR commit send failed"
    assert not session._pending_commits
    assert session._events.empty()


@pytest.mark.asyncio
async def test_qwen_asr_session_audio_failure_resolves_pending_and_reports_error() -> None:
    session = _make_session()
    session._loop = asyncio.get_running_loop()
    assert session._register_commit() is not None

    class FailingConversation:
        def append_audio(self, audio_b64: str) -> None:
            _ = audio_b64
            raise RuntimeError("append failed")

    assert session._append_audio(FailingConversation(), b"pcm") is False
    await asyncio.sleep(0)

    first = await session._events.get()
    second = await session._events.get()
    assert first == STTBackendTranscriptEvent(text="", is_final=True)
    assert isinstance(second, RuntimeError)
    assert str(second) == "Qwen ASR audio send failed"
    assert not session._pending_commits
    assert session._events.empty()


@pytest.mark.asyncio
async def test_qwen_asr_session_reports_error(monkeypatch) -> None:
    session = _make_session()
    session._loop = asyncio.get_running_loop()

    err = RuntimeError("boom")
    session._report_error(err)
    await asyncio.sleep(0)

    event = await session._events.get()
    assert event is err
    assert session._error_reported is True
    assert session._connect_error is err
    assert session._connected.is_set() is True


@pytest.mark.asyncio
async def test_qwen_asr_session_events_yield_and_raise() -> None:
    session = _make_session()

    session._events.put_nowait(STTBackendTranscriptEvent(text="hi", is_final=True))
    session._events.put_nowait(None)

    gen = session.events()
    event = await gen.__anext__()
    assert event.text == "hi"
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()

    session._events.put_nowait(RuntimeError("boom"))
    gen = session.events()
    with pytest.raises(RuntimeError, match="boom"):
        await gen.__anext__()


@pytest.mark.asyncio
async def test_qwen_asr_session_start_success(monkeypatch) -> None:
    session = _make_session()

    def fake_run_sync():
        session._connected.set()

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(qwen_asr_module.threading, "Thread", TargetThread)
    monkeypatch.setattr(session, "_run_sync", fake_run_sync)

    await session.start()
    assert session._connected.is_set() is True


@pytest.mark.asyncio
async def test_qwen_asr_session_start_failure(monkeypatch) -> None:
    session = _make_session()

    def fake_run_sync():
        session._connect_error = RuntimeError("fail")
        session._connected.set()

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(qwen_asr_module.threading, "Thread", TargetThread)
    monkeypatch.setattr(session, "_run_sync", fake_run_sync)

    with pytest.raises(RuntimeError, match="fail"):
        await session.start()


@pytest.mark.asyncio
async def test_qwen_asr_session_signal_stop_is_safe() -> None:
    session = _make_session()
    session._signal_stop()
    assert session._audio_q.get_nowait() is _STOP


@pytest.mark.asyncio
async def test_qwen_asr_session_report_error_only_once() -> None:
    session = _make_session()
    session._loop = asyncio.get_running_loop()

    err = RuntimeError("boom")
    session._report_error(err)
    session._report_error(RuntimeError("second"))
    await asyncio.sleep(0)

    assert session._error_reported is True
    assert await session._events.get() is err
    assert session._events.empty()


@pytest.mark.asyncio
async def test_qwen_asr_session_run_sync_processes_audio_commit_and_final_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _make_session()
    session._loop = asyncio.get_running_loop()
    session._connect_started_at = 1.0

    append_calls: list[str] = []
    commit_calls = 0
    end_session_calls: list[float] = []
    closed = False
    latest_dashscope: dict[str, object] = {}

    class FakeOmniRealtimeCallback:
        pass

    class FakeMultiModality:
        TEXT = "text"

    class FakeTranscriptionParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeConversation:
        def __init__(self, model: str, url: str, callback):
            _ = (model, url)
            self.callback = callback

        def connect(self):
            self.callback.on_open()

        def update_session(self, **kwargs):
            _ = kwargs
            self.callback.on_event({"type": "session.created", "session": {"id": "sid"}})

        def append_audio(self, audio_b64: str):
            append_calls.append(audio_b64)
            self.callback.on_event(
                {
                    "type": "conversation.item.input_audio_transcription.text",
                    "text": "t",
                    "stash": "",
                }
            )

        def commit(self):
            nonlocal commit_calls
            commit_calls += 1
            item_id = f"item-{commit_calls}"
            self.callback.on_event({"type": "input_audio_buffer.committed", "item_id": item_id})
            self.callback.on_event(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": item_id,
                    "transcript": "final transcript",
                }
            )

        def end_session(self, timeout):
            end_session_calls.append(timeout)
            self.callback.on_event({"type": "session.finished"})

        def close(self):
            nonlocal closed
            closed = True

    dashscope_pkg = types.ModuleType("dashscope")
    dashscope_pkg.api_key = None
    latest_dashscope["pkg"] = dashscope_pkg
    qwen_omni_pkg = types.ModuleType("dashscope.audio.qwen_omni")
    qwen_omni_pkg.MultiModality = FakeMultiModality
    qwen_omni_pkg.OmniRealtimeCallback = FakeOmniRealtimeCallback
    qwen_omni_pkg.OmniRealtimeConversation = FakeConversation
    omni_rt_pkg = types.ModuleType("dashscope.audio.qwen_omni.omni_realtime")
    omni_rt_pkg.TranscriptionParams = FakeTranscriptionParams

    monkeypatch.setitem(sys.modules, "dashscope", dashscope_pkg)
    monkeypatch.setitem(sys.modules, "dashscope.audio", types.ModuleType("dashscope.audio"))
    monkeypatch.setitem(sys.modules, "dashscope.audio.qwen_omni", qwen_omni_pkg)
    monkeypatch.setitem(sys.modules, "dashscope.audio.qwen_omni.omni_realtime", omni_rt_pkg)

    session._audio_q.put_nowait(b"pcm")
    session._audio_q.put_nowait(_COMMIT)
    session._audio_q.put_nowait(_COMMIT)
    session._audio_q.put_nowait(_END_SESSION)
    session._run_sync()
    await asyncio.sleep(0)

    first = await session._events.get()
    assert isinstance(first, STTBackendTranscriptEvent)
    assert first.text == "final transcript"
    assert latest_dashscope["pkg"].api_key == "k"
    assert append_calls
    assert commit_calls == 2
    assert end_session_calls == [session.finish_timeout_s]
    assert session._connected.is_set() is True
    assert closed is True

    tail: list[object] = []
    while not session._events.empty():
        tail.append(session._events.get_nowait())
    assert None in tail


@pytest.mark.asyncio
async def test_qwen_asr_session_public_stop_after_idle_calls_end_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _make_session()
    session._loop = asyncio.get_running_loop()
    connected = threading.Event()
    end_session_calls: list[float] = []
    closed = threading.Event()

    class FakeOmniRealtimeCallback:
        pass

    class FakeMultiModality:
        TEXT = "text"

    class FakeTranscriptionParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeConversation:
        def __init__(self, model: str, url: str, callback):
            _ = (model, url)
            self.callback = callback

        def connect(self) -> None:
            self.callback.on_open()
            connected.set()

        def update_session(self, **kwargs) -> None:
            _ = kwargs

        def append_audio(self, audio_b64: str) -> None:
            _ = audio_b64

        def commit(self) -> None:
            return None

        def end_session(self, timeout: float) -> None:
            end_session_calls.append(timeout)

        def close(self) -> None:
            closed.set()

    dashscope_pkg = types.ModuleType("dashscope")
    dashscope_pkg.api_key = None
    qwen_omni_pkg = types.ModuleType("dashscope.audio.qwen_omni")
    qwen_omni_pkg.MultiModality = FakeMultiModality
    qwen_omni_pkg.OmniRealtimeCallback = FakeOmniRealtimeCallback
    qwen_omni_pkg.OmniRealtimeConversation = FakeConversation
    omni_rt_pkg = types.ModuleType("dashscope.audio.qwen_omni.omni_realtime")
    omni_rt_pkg.TranscriptionParams = FakeTranscriptionParams

    monkeypatch.setitem(sys.modules, "dashscope", dashscope_pkg)
    monkeypatch.setitem(sys.modules, "dashscope.audio", types.ModuleType("dashscope.audio"))
    monkeypatch.setitem(sys.modules, "dashscope.audio.qwen_omni", qwen_omni_pkg)
    monkeypatch.setitem(sys.modules, "dashscope.audio.qwen_omni.omni_realtime", omni_rt_pkg)

    worker = asyncio.create_task(asyncio.to_thread(session._run_sync))
    assert await asyncio.to_thread(connected.wait, 1.0)
    await asyncio.sleep(0.15)
    await session.stop()
    await asyncio.wait_for(worker, timeout=2.0)
    await asyncio.sleep(0)

    assert end_session_calls == [session.finish_timeout_s]
    assert closed.is_set()


@pytest.mark.asyncio
async def test_qwen_asr_session_end_session_timeout_does_not_report_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _make_session()
    session._loop = asyncio.get_running_loop()
    closed = False

    class FakeOmniRealtimeCallback:
        pass

    class FakeMultiModality:
        TEXT = "text"

    class FakeTranscriptionParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeConversation:
        def __init__(self, model: str, url: str, callback):
            _ = (model, url)
            self.callback = callback

        def connect(self):
            self.callback.on_open()

        def update_session(self, **kwargs):
            _ = kwargs

        def end_session(self, timeout):
            _ = timeout
            raise TimeoutError("session finish timed out")

        def close(self):
            nonlocal closed
            closed = True

    dashscope_pkg = types.ModuleType("dashscope")
    dashscope_pkg.api_key = None
    qwen_omni_pkg = types.ModuleType("dashscope.audio.qwen_omni")
    qwen_omni_pkg.MultiModality = FakeMultiModality
    qwen_omni_pkg.OmniRealtimeCallback = FakeOmniRealtimeCallback
    qwen_omni_pkg.OmniRealtimeConversation = FakeConversation
    omni_rt_pkg = types.ModuleType("dashscope.audio.qwen_omni.omni_realtime")
    omni_rt_pkg.TranscriptionParams = FakeTranscriptionParams

    monkeypatch.setitem(sys.modules, "dashscope", dashscope_pkg)
    monkeypatch.setitem(sys.modules, "dashscope.audio", types.ModuleType("dashscope.audio"))
    monkeypatch.setitem(sys.modules, "dashscope.audio.qwen_omni", qwen_omni_pkg)
    monkeypatch.setitem(sys.modules, "dashscope.audio.qwen_omni.omni_realtime", omni_rt_pkg)

    assert session._register_commit() is not None
    session._audio_q.put_nowait(_END_SESSION)
    session._run_sync()
    await asyncio.sleep(0)

    first = await session._events.get()
    assert first == STTBackendTranscriptEvent(text="", is_final=True)
    tail: list[object] = []
    while not session._events.empty():
        tail.append(session._events.get_nowait())
    assert None in tail
    assert not any(isinstance(item, BaseException) for item in tail)
    assert session._error_reported is False
    assert closed is True
