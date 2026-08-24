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
) -> _DeepgramSDKSession:
    return _DeepgramSDKSession(
        api_key="k",
        model=model,
        language="en",
        sample_rate_hz=16000,
        connect_timeout_s=5.0,
        keyterms=keyterms or [],
        stream_label=stream_label,
    )


@pytest.mark.asyncio
async def test_deepgram_session_on_speech_end_enqueues_finalize(caplog):
    session = _make_session()

    with caplog.at_level(logging.INFO):
        await session.on_speech_end(trailing_silence_ms=500, reason="silence")
    finalize = session._audio_q.get_nowait()
    assert finalize is _FINALIZE
    assert session._audio_q.empty()
    assert "boundary_reason=silence" in caplog.text
    assert "observed_tail_ms=500" in caplog.text
    assert "boundary_wait_ms=500" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO):
        await session.on_speech_end(trailing_silence_ms=0, reason="max_duration")
    finalize = session._audio_q.get_nowait()
    assert finalize is _FINALIZE
    assert session._audio_q.empty()
    assert "boundary_reason=max_duration" in caplog.text
    assert "boundary_wait_ms=0" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO):
        await session.on_speech_end(trailing_silence_ms=160, reason="soft_pause")
    finalize = session._audio_q.get_nowait()
    assert finalize is _FINALIZE
    assert session._audio_q.empty()
    assert "boundary_reason=soft_pause" in caplog.text
    assert "boundary_wait_ms=160" in caplog.text


@pytest.mark.asyncio
async def test_deepgram_session_on_speech_end_boundary_unknown(caplog):
    session = _make_session()

    with caplog.at_level(logging.INFO):
        await session.on_speech_end()

    assert "boundary_reason=None" in caplog.text
    assert "boundary_wait_ms=None" in caplog.text


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
        event = session._build_transcript_event(result)

    assert isinstance(event, STTBackendTranscriptEvent)
    assert event.text == ""
    assert event.is_final is True
    assert session._empty_final_acks == 1
    assert any(
        "[STT][peer] Empty final transcript acknowledged" in message for message in caplog.messages
    )


def test_deepgram_peer_session_counts_emitted_finals() -> None:
    session = _make_session(stream_label="peer")
    result = types.SimpleNamespace(
        channel=types.SimpleNamespace(
            alternatives=[types.SimpleNamespace(transcript="hello world")]
        ),
        is_final=True,
        speech_final=False,
    )

    event = session._build_transcript_event(result)

    assert isinstance(event, STTBackendTranscriptEvent)
    assert event.text == "hello world"
    assert session._emitted_finals == 1


@pytest.mark.asyncio
async def test_deepgram_peer_session_logs_summary_once_on_shutdown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = _make_session(stream_label="peer")
    empty_result = types.SimpleNamespace(
        channel=types.SimpleNamespace(alternatives=[types.SimpleNamespace(transcript="")]),
        is_final=True,
        speech_final=False,
    )
    final_result = types.SimpleNamespace(
        channel=types.SimpleNamespace(alternatives=[types.SimpleNamespace(transcript="hello")]),
        is_final=True,
        speech_final=False,
    )

    session._build_transcript_event(empty_result)
    session._build_transcript_event(final_result)

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

    class FakeEventType:
        OPEN = "open"
        MESSAGE = "message"
        ERROR = "error"
        CLOSE = "close"

    class FakeControlMessage:
        def __init__(self, type: str):
            self.type = type

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def on(self, event_type, callback):
            if event_type == FakeEventType.OPEN:
                callback(object())
            if event_type == FakeEventType.MESSAGE:
                alt = types.SimpleNamespace(transcript="hello world")
                result = types.SimpleNamespace(
                    channel=types.SimpleNamespace(alternatives=[alt]),
                    is_final=True,
                    speech_final=False,
                )
                callback(result)

        def start_listening(self):
            return None

        def send_control(self, message):
            sent_controls.append(message.type)

        def send_media(self, data: bytes):
            sent_media.append(data)

    class FakeV1:
        def connect(self, **kwargs):
            connect_kwargs.update(kwargs)
            return FakeConnection()

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
    monkeypatch.setattr(deepgram_module.threading, "Thread", NoopThread)

    return types.SimpleNamespace(
        connect_kwargs=connect_kwargs,
        sent_media=sent_media,
        sent_controls=sent_controls,
    )


@pytest.mark.asyncio
async def test_deepgram_session_run_sync_handles_message_finalize_and_stop(
    fake_deepgram_modules,
) -> None:
    session = _make_session(keyterms=["Puripuly", "VRChat"])
    session._loop = asyncio.get_running_loop()
    session._connect_started_at = 1.0

    session._audio_q.put_nowait(_FINALIZE)
    session._audio_q.put_nowait(_FINALIZE)
    session._audio_q.put_nowait(b"pcm")
    session._audio_q.put_nowait(_STOP)
    session._run_sync()
    await asyncio.sleep(0)

    first = await session._events.get()
    assert isinstance(first, STTBackendTranscriptEvent)
    assert first.text == "hello world"
    assert fake_deepgram_modules.sent_controls == ["Finalize", "Finalize"]
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
