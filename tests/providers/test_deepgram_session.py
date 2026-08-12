from __future__ import annotations

import asyncio
import logging
import sys
import types

import pytest

from puripuly_heart.core.stt.backend import STTBackendTranscriptEvent
from puripuly_heart.providers.stt import deepgram as deepgram_module
from puripuly_heart.providers.stt.deepgram import _FINALIZE, _STOP, _DeepgramSDKSession
from tests.helpers.fakes import NoopThread, TargetThread


def _make_session(
    *,
    model: str = "nova-3",
    keyterms: list[str] | None = None,
    stream_label: str | None = None,
    finalize_timeout_s: float = 1.0,
) -> _DeepgramSDKSession:
    return _DeepgramSDKSession(
        api_key="k",
        model=model,
        language="en",
        sample_rate_hz=16000,
        connect_timeout_s=5.0,
        keyterms=keyterms or [],
        stream_label=stream_label,
        finalize_timeout_s=finalize_timeout_s,
    )


def _result(
    text: str,
    *,
    is_final: bool = True,
    speech_final: bool = False,
    from_finalize: bool = False,
    start: float | None = None,
    duration: float | None = None,
):
    return types.SimpleNamespace(
        channel=types.SimpleNamespace(alternatives=[types.SimpleNamespace(transcript=text)]),
        is_final=is_final,
        speech_final=speech_final,
        from_finalize=from_finalize,
        start=start,
        duration=duration,
    )


async def _wait_until(predicate, *, timeout_s: float = 1.0) -> None:
    async with asyncio.timeout(timeout_s):
        while not predicate():
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_deepgram_session_on_speech_end_enqueues_finalize(caplog) -> None:
    session = _make_session()

    with caplog.at_level(logging.INFO):
        await session.on_speech_end(trailing_silence_ms=500)
    finalize = session._audio_q.get_nowait()
    assert finalize is _FINALIZE
    assert "observed_tail_ms=500 injected_padding_ms=0" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO):
        await session.on_speech_end(trailing_silence_ms=0)
    finalize = session._audio_q.get_nowait()

    assert finalize is _FINALIZE
    assert session._audio_q.empty()
    assert "observed_tail_ms=0 injected_padding_ms=0" in caplog.text
    assert "declared_trim_ms=0 boundary_wait_ms=unknown" in caplog.text


def test_deepgram_session_aggregates_segments_only_at_finalize_fence() -> None:
    session = _make_session()
    session._outstanding_finalizes = 1
    session._all_finalized.clear()
    session._register_finalize(has_audio=True)

    assert session._build_transcript_events(_result("first")) == []
    assert session._build_transcript_events(_result("second")) == []
    assert session._build_transcript_events(
        _result("third", from_finalize=True)
    ) == []
    pending = session._active_finalize
    assert pending is not None
    events = session._complete_active_finalize(pending)

    assert [event.text for event in events] == ["first second third"]
    assert session._outstanding_finalizes == 0
    assert session._all_finalized.is_set()


def test_deepgram_session_preserves_finalize_order_for_no_audio_request() -> None:
    session = _make_session()
    session._outstanding_finalizes = 2
    session._all_finalized.clear()
    session._register_finalize(has_audio=True)
    session._register_finalize(has_audio=False)

    assert session._build_transcript_events(_result("first", from_finalize=True)) == []
    pending = session._active_finalize
    assert pending is not None
    events = session._complete_active_finalize(pending)

    assert [event.text for event in events] == ["first", ""]
    assert session._outstanding_finalizes == 0


@pytest.mark.asyncio
async def test_deepgram_session_does_not_signal_drain_after_new_boundary_is_accepted() -> None:
    session = _make_session()
    session._loop = asyncio.get_running_loop()
    session._outstanding_finalizes = 1
    session._all_finalized.clear()
    pending = session._register_finalize(has_audio=True)
    session._build_transcript_events(
        _result("first", from_finalize=True, start=0.0, duration=0.5)
    )
    session._complete_active_finalize(pending)

    await session.on_speech_end(trailing_silence_ms=500)
    await asyncio.sleep(0)

    assert session._outstanding_finalizes == 1
    assert not session._all_finalized.is_set()


@pytest.mark.asyncio
async def test_deepgram_session_finalize_timeout_resolves_and_fences_late_results() -> None:
    session = _make_session()
    session._loop = asyncio.get_running_loop()
    session._outstanding_finalizes = 1
    session._all_finalized.clear()
    pending = session._register_finalize(has_audio=True)

    session._fail_finalize_session(
        RuntimeError(f"Deepgram finalize fence timeout sequence={pending.sequence}"),
        status="finalize_timeout",
        sequence=pending.sequence,
    )
    assert session._build_transcript_events(
        _result("late", from_finalize=True)
    ) == []
    await asyncio.sleep(0)

    first = await session._events.get()
    second = await session._events.get()
    assert first == STTBackendTranscriptEvent(text="", is_final=True)
    assert isinstance(second, RuntimeError)
    assert "finalize fence timeout" in str(second)
    assert session._stopped is True
    assert session._accept_results is False


@pytest.mark.asyncio
async def test_deepgram_session_abort_purges_and_rejects_late_results() -> None:
    session = _make_session()
    session._outstanding_finalizes = 1
    session._all_finalized.clear()
    session._register_finalize(has_audio=True)
    session._audio_q.put_nowait(b"audio")

    await session.abort_for_toggle_off()

    assert session._audio_q.get_nowait() is _STOP
    assert session._audio_q.empty()
    assert session._build_transcript_events(
        _result("late", from_finalize=True)
    ) == []
    assert not session._pending_finalizes


@pytest.mark.asyncio
async def test_deepgram_session_send_audio_and_stop() -> None:
    session = _make_session()

    await session.send_audio(b"abc")
    assert session._audio_q.get_nowait() == b"abc"

    await session.stop()
    assert session._stopped is True
    assert session._audio_q.get_nowait() is _STOP


@pytest.mark.asyncio
async def test_deepgram_session_events_yield_and_raise() -> None:
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
async def test_deepgram_session_emits_test_final() -> None:
    session = _make_session()

    await session._emit_test_final(text="hello there")
    event = await session._events.get()

    assert isinstance(event, STTBackendTranscriptEvent)
    assert event.text == "hello there"
    assert event.is_final is True


def test_deepgram_peer_session_acknowledges_empty_final_with_dedicated_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = _make_session(stream_label="peer")
    result = types.SimpleNamespace(
        channel=types.SimpleNamespace(alternatives=[types.SimpleNamespace(transcript="")]),
        is_final=True,
        speech_final=False,
    )

    with caplog.at_level(logging.INFO, logger=deepgram_module.logger.name):
        events = session._build_transcript_events(result)

    assert events == []
    assert session._empty_final_acks == 0


def test_deepgram_peer_session_counts_emitted_finals() -> None:
    session = _make_session(stream_label="peer")
    session._outstanding_finalizes = 1
    session._register_finalize(has_audio=True)
    result = types.SimpleNamespace(
        channel=types.SimpleNamespace(
            alternatives=[types.SimpleNamespace(transcript="hello world")]
        ),
        is_final=True,
        speech_final=False,
        from_finalize=True,
    )

    assert session._build_transcript_events(result) == []
    pending = session._active_finalize
    assert pending is not None
    events = session._complete_active_finalize(pending)

    assert [event.text for event in events] == ["hello world"]
    assert session._emitted_finals == 1


@pytest.mark.asyncio
async def test_deepgram_peer_session_logs_summary_once_on_shutdown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = _make_session(stream_label="peer")
    session._outstanding_finalizes = 2
    session._register_finalize(has_audio=False)
    session._register_finalize(has_audio=True)
    assert session._build_transcript_events(
        types.SimpleNamespace(
            channel=types.SimpleNamespace(alternatives=[types.SimpleNamespace(transcript="hello")]),
            is_final=True,
            speech_final=False,
            from_finalize=True,
        )
    ) == []
    pending = session._active_finalize
    assert pending is not None
    session._complete_active_finalize(pending)

    with caplog.at_level(logging.INFO, logger=deepgram_module.logger.name):
        await session.stop()
        await session.close()

    summary_messages = [
        message for message in caplog.messages if "[STT][peer] Session summary:" in message
    ]
    assert len(summary_messages) == 1
    assert "emitted_finals=1" in summary_messages[0]
    assert "empty_final_acks=1" in summary_messages[0]
    assert "total_finals_seen=2" in summary_messages[0]


@pytest.mark.asyncio
async def test_deepgram_session_start_success(monkeypatch) -> None:
    session = _make_session()

    def fake_run_sync():
        session._connected.set()

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(deepgram_module.threading, "Thread", TargetThread)
    monkeypatch.setattr(session, "_run_sync", fake_run_sync)

    await session.start()
    assert session._connected.is_set() is True


@pytest.mark.asyncio
async def test_deepgram_session_start_timeout(monkeypatch) -> None:
    session = _make_session()

    def fake_run_sync():
        return None

    async def fake_to_thread(*_args, **_kwargs):
        return False

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(deepgram_module.threading, "Thread", TargetThread)
    monkeypatch.setattr(session, "_run_sync", fake_run_sync)

    with pytest.raises(RuntimeError, match="connection timeout"):
        await session.start()


@pytest.mark.asyncio
async def test_deepgram_session_report_error_is_emitted_once() -> None:
    session = _make_session()
    session._loop = asyncio.get_running_loop()

    err = RuntimeError("boom")
    session._report_error(err)
    session._report_error(RuntimeError("second"))
    await asyncio.sleep(0)

    assert session._error_reported is True
    assert await session._events.get() is err
    assert session._events.empty()


@pytest.fixture
def fake_deepgram_modules(monkeypatch: pytest.MonkeyPatch):
    sent_media: list[bytes] = []
    sent_controls: list[str] = []
    connect_kwargs: dict[str, object] = {}
    auto_results: list[object] = []
    real_thread = deepgram_module.threading.Thread

    class FakeEventType:
        OPEN = "open"
        MESSAGE = "message"
        ERROR = "error"
        CLOSE = "close"

    class FakeControlMessage:
        def __init__(self, type: str):
            self.type = type

    class FakeConnection:
        def __init__(self):
            self.message_callback = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def on(self, event_type, callback):
            if event_type == FakeEventType.OPEN:
                callback(object())
            if event_type == FakeEventType.MESSAGE:
                self.message_callback = callback

        def start_listening(self):
            return None

        def send_control(self, message):
            sent_controls.append(message.type)
            if message.type == "Finalize" and auto_results:
                self.emit(auto_results.pop(0))

        def send_media(self, data: bytes):
            sent_media.append(data)

        def emit(self, result: object) -> None:
            assert self.message_callback is not None
            self.message_callback(result)

    connection = FakeConnection()

    class FakeV1:
        def connect(self, **kwargs):
            connect_kwargs.update(kwargs)
            return connection

    class FakeListen:
        v1 = FakeV1()

    class FakeClient:
        def __init__(self, api_key: str):
            _ = api_key
            self.listen = FakeListen()

    deepgram_pkg = types.ModuleType("deepgram")
    deepgram_pkg.DeepgramClient = FakeClient
    deepgram_core = types.ModuleType("deepgram.core")
    deepgram_events = types.ModuleType("deepgram.core.events")
    deepgram_events.EventType = FakeEventType
    deepgram_ext = types.ModuleType("deepgram.extensions")
    deepgram_ext_types = types.ModuleType("deepgram.extensions.types")
    deepgram_sockets = types.ModuleType("deepgram.extensions.types.sockets")
    deepgram_sockets.ListenV1ControlMessage = FakeControlMessage

    monkeypatch.setitem(sys.modules, "deepgram", deepgram_pkg)
    monkeypatch.setitem(sys.modules, "deepgram.core", deepgram_core)
    monkeypatch.setitem(sys.modules, "deepgram.core.events", deepgram_events)
    monkeypatch.setitem(sys.modules, "deepgram.extensions", deepgram_ext)
    monkeypatch.setitem(sys.modules, "deepgram.extensions.types", deepgram_ext_types)
    monkeypatch.setitem(sys.modules, "deepgram.extensions.types.sockets", deepgram_sockets)

    def thread_factory(*args, **kwargs):
        target = kwargs.get("target")
        if target is not None and target.__name__ in {"listening_thread", "keepalive_thread"}:
            return NoopThread(*args, **kwargs)
        return real_thread(*args, **kwargs)

    monkeypatch.setattr(deepgram_module.threading, "Thread", thread_factory)

    return types.SimpleNamespace(
        connect_kwargs=connect_kwargs,
        sent_media=sent_media,
        sent_controls=sent_controls,
        auto_results=auto_results,
        connection=connection,
    )


@pytest.mark.asyncio
async def test_deepgram_session_run_sync_handles_message_finalize_and_stop(
    fake_deepgram_modules,
) -> None:
    session = _make_session(keyterms=["Puripuly", "VRChat"])
    session._loop = asyncio.get_running_loop()
    session._connect_started_at = 1.0
    session._outstanding_finalizes = 2
    session._all_finalized.clear()
    fake_deepgram_modules.auto_results.append(
        _result("hello world", from_finalize=True, start=0.0, duration=0.5)
    )

    session._audio_q.put_nowait(b"pcm")
    session._audio_q.put_nowait(_FINALIZE)
    session._audio_q.put_nowait(_FINALIZE)
    session._audio_q.put_nowait(_STOP)
    session._run_sync()
    await asyncio.sleep(0)

    first = await session._events.get()
    assert isinstance(first, STTBackendTranscriptEvent)
    assert first.text == "hello world"
    second = await session._events.get()
    assert isinstance(second, STTBackendTranscriptEvent)
    assert second.text == ""
    assert fake_deepgram_modules.sent_controls == ["Finalize"]
    assert fake_deepgram_modules.sent_media == [b"pcm"]
    assert session._connected.is_set() is True
    assert "diarize" not in fake_deepgram_modules.connect_kwargs
    assert fake_deepgram_modules.connect_kwargs["keyterm"] == ["Puripuly", "VRChat"]

    # _run_sync posts termination markers in stop path/finally.
    tail: list[object] = []
    while not session._events.empty():
        tail.append(session._events.get_nowait())
    assert None in tail


@pytest.mark.asyncio
async def test_deepgram_session_serializes_two_wire_finalizes_and_drops_stale_duplicate(
    fake_deepgram_modules,
) -> None:
    session = _make_session(finalize_timeout_s=0.5)
    session._loop = asyncio.get_running_loop()
    session._connect_started_at = 1.0

    await session.send_audio(b"first")
    await session.on_speech_end(trailing_silence_ms=500)
    await session.send_audio(b"second")
    await session.on_speech_end(trailing_silence_ms=500)
    run_task = asyncio.create_task(asyncio.to_thread(session._run_sync))

    await _wait_until(lambda: fake_deepgram_modules.sent_controls == ["Finalize"])
    assert fake_deepgram_modules.sent_media == [b"first"]

    fake_deepgram_modules.connection.emit(
        _result("first-a", start=0.0, duration=0.4)
    )
    fake_deepgram_modules.connection.emit(
        _result("first-b", from_finalize=True, start=0.4, duration=0.3)
    )
    await _wait_until(lambda: fake_deepgram_modules.sent_controls == ["Finalize", "Finalize"])
    assert fake_deepgram_modules.sent_media == [b"first", b"second"]

    stop_task = asyncio.create_task(session.stop())
    await asyncio.sleep(0)
    assert not stop_task.done()

    fake_deepgram_modules.connection.emit(
        _result("first-b", from_finalize=True, start=0.4, duration=0.3)
    )
    await asyncio.sleep(0.01)
    assert not stop_task.done()

    fake_deepgram_modules.connection.emit(
        _result("second", from_finalize=True, start=0.7, duration=0.5)
    )
    await stop_task
    await run_task
    await asyncio.sleep(0)

    items: list[object] = []
    while not session._events.empty():
        items.append(session._events.get_nowait())
    transcripts = [item for item in items if isinstance(item, STTBackendTranscriptEvent)]
    errors = [item for item in items if isinstance(item, BaseException)]
    assert [event.text for event in transcripts] == ["first-a first-b", "second"]
    assert errors == []


@pytest.mark.asyncio
async def test_deepgram_session_wire_timeout_resolves_all_queued_boundaries(
    fake_deepgram_modules,
) -> None:
    session = _make_session(finalize_timeout_s=0.02)
    session._loop = asyncio.get_running_loop()
    session._connect_started_at = 1.0

    await session.send_audio(b"first")
    await session.on_speech_end(trailing_silence_ms=500)
    await session.send_audio(b"second")
    await session.on_speech_end(trailing_silence_ms=500)
    await asyncio.to_thread(session._run_sync)
    await asyncio.sleep(0)

    items: list[object] = []
    while not session._events.empty():
        items.append(session._events.get_nowait())
    transcripts = [item for item in items if isinstance(item, STTBackendTranscriptEvent)]
    errors = [item for item in items if isinstance(item, RuntimeError)]
    assert [event.text for event in transcripts] == ["", ""]
    assert len(errors) == 1
    assert "finalize fence timeout sequence=1" in str(errors[0])
    assert fake_deepgram_modules.sent_media == [b"first"]
    assert fake_deepgram_modules.sent_controls == ["Finalize"]
    assert session._outstanding_finalizes == 0
    assert session._accept_results is False

    fake_deepgram_modules.connection.emit(
        _result("late", from_finalize=True, start=0.0, duration=0.5)
    )
    await asyncio.sleep(0)
    assert session._events.empty()


@pytest.mark.asyncio
async def test_deepgram_session_wire_abort_purges_queued_audio_and_late_callback(
    fake_deepgram_modules,
) -> None:
    session = _make_session(finalize_timeout_s=0.5)
    session._loop = asyncio.get_running_loop()
    session._connect_started_at = 1.0

    await session.send_audio(b"first")
    await session.on_speech_end(trailing_silence_ms=500)
    await session.send_audio(b"second")
    await session.on_speech_end(trailing_silence_ms=500)
    run_task = asyncio.create_task(asyncio.to_thread(session._run_sync))
    await _wait_until(lambda: fake_deepgram_modules.sent_controls == ["Finalize"])

    await session.abort_for_toggle_off()
    fake_deepgram_modules.connection.emit(
        _result("late", from_finalize=True, start=0.0, duration=0.5)
    )
    await run_task
    await asyncio.sleep(0)

    items: list[object] = []
    while not session._events.empty():
        items.append(session._events.get_nowait())
    assert not any(isinstance(item, STTBackendTranscriptEvent) for item in items)
    assert not any(isinstance(item, BaseException) for item in items)
    assert fake_deepgram_modules.sent_media == [b"first"]
    assert fake_deepgram_modules.sent_controls == ["Finalize"]
    assert session._outstanding_finalizes == 0
    assert session._accept_results is False


@pytest.mark.asyncio
async def test_deepgram_session_run_sync_omits_keyterm_when_empty(
    fake_deepgram_modules,
) -> None:
    session = _make_session()
    session._loop = asyncio.get_running_loop()
    session._connect_started_at = 1.0

    session._audio_q.put_nowait(_STOP)
    session._run_sync()
    await asyncio.sleep(0)

    assert "keyterm" not in fake_deepgram_modules.connect_kwargs


@pytest.mark.asyncio
async def test_deepgram_session_run_sync_omits_keyterm_for_unsupported_model(
    fake_deepgram_modules,
) -> None:
    session = _make_session(model="nova-2", keyterms=["Puripuly"])
    session._loop = asyncio.get_running_loop()
    session._connect_started_at = 1.0

    session._audio_q.put_nowait(_STOP)
    session._run_sync()
    await asyncio.sleep(0)

    assert "keyterm" not in fake_deepgram_modules.connect_kwargs
