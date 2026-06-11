from __future__ import annotations

import asyncio
import base64
import json
import sys
from types import SimpleNamespace

import pytest

from puripuly_heart.core.stt.backend import STTBackendTranscriptEvent
from puripuly_heart.providers.stt.sixtydb import (
    _FINALIZE,
    _STOP,
    SixtyDBRealtimeSTTBackend,
    _SixtyDBSession,
)


def _make_session(*, context_terms: list[str] | None = None) -> _SixtyDBSession:
    return _SixtyDBSession(
        api_key="k",
        endpoint="wss://example",
        sample_rate_hz=16000,
        language_codes=["en"],
        context_terms=context_terms or [],
        utterance_end_ms=300,
        trailing_silence_ms=400,
        connect_timeout_s=5.0,
    )


@pytest.mark.asyncio
async def test_sixtydb_backend_validates_params() -> None:
    backend = SixtyDBRealtimeSTTBackend(api_key="", language_codes=["en"])
    with pytest.raises(ValueError, match="api_key"):
        await backend.open_session()

    backend = SixtyDBRealtimeSTTBackend(api_key="k", language_codes=["en"], endpoint="")
    with pytest.raises(ValueError, match="endpoint"):
        await backend.open_session()

    backend = SixtyDBRealtimeSTTBackend(
        api_key="k",
        language_codes=["en"],
        utterance_end_ms=100,
    )
    with pytest.raises(ValueError, match="utterance_end_ms"):
        await backend.open_session()

    backend = SixtyDBRealtimeSTTBackend(
        api_key="k",
        language_codes=["en"],
        sample_rate_hz=12345,
    )
    with pytest.raises(ValueError, match="sample_rate_hz"):
        await backend.open_session()


@pytest.mark.asyncio
async def test_sixtydb_session_handles_message_errors() -> None:
    session = _make_session()

    session._handle_message("not-json")
    assert session._events.empty()

    session._handle_message(json.dumps({"type": "error", "error": "bad"}))
    event = session._events.get_nowait()
    assert isinstance(event, RuntimeError)


@pytest.mark.asyncio
async def test_sixtydb_session_emits_canonical_final() -> None:
    session = _make_session()

    # Interim partial — ignored.
    session._handle_message(
        json.dumps({"type": "transcription", "text": "Hel", "is_final": False})
    )
    # Dict-corrected pre-final (speech_final False) — ignored, wait for canonical.
    session._handle_message(
        json.dumps(
            {"type": "transcription", "text": "Hello", "is_final": True, "speech_final": False}
        )
    )
    assert session._events.empty()

    # Canonical final.
    session._handle_message(
        json.dumps(
            {
                "type": "transcription",
                "text": "Hello world",
                "is_final": True,
                "speech_final": True,
            }
        )
    )
    event = session._events.get_nowait()
    assert isinstance(event, STTBackendTranscriptEvent)
    assert event.text == "Hello world"
    assert event.is_final is True


@pytest.mark.asyncio
async def test_sixtydb_session_skips_empty_and_non_transcription() -> None:
    session = _make_session()

    # Empty final = speech-end-no-result.
    session._handle_message(
        json.dumps({"type": "transcription", "text": "", "is_final": True, "speech_final": True})
    )
    # Non transcript-bearing messages.
    session._handle_message(json.dumps({"type": "speech_started", "timestamp": 1.0}))
    session._handle_message(json.dumps({"type": "session_stopped"}))

    assert session._events.empty()


@pytest.mark.asyncio
async def test_sixtydb_session_on_speech_end_tops_up_silence_then_finalize() -> None:
    session = _make_session()

    await session.on_speech_end(trailing_silence_ms=None)

    silence = await session._audio_q.get()
    finalize = await session._audio_q.get()

    assert isinstance(silence, bytes)
    assert len(silence) > 0
    assert finalize is _FINALIZE


@pytest.mark.asyncio
async def test_sixtydb_session_on_speech_end_no_missing_silence() -> None:
    session = _make_session()

    # Trailing silence already exceeds the configured target.
    await session.on_speech_end(trailing_silence_ms=1000)

    finalize = await session._audio_q.get()
    assert finalize is _FINALIZE
    assert session._audio_q.empty()


@pytest.mark.asyncio
async def test_sixtydb_session_send_audio_and_stop() -> None:
    session = _make_session()

    await session.send_audio(b"abc")
    assert await session._audio_q.get() == b"abc"

    await session.stop()
    assert session._stopped is True
    assert await session._audio_q.get() is _STOP


@pytest.mark.asyncio
async def test_sixtydb_session_events_yield_and_raise() -> None:
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
async def test_sixtydb_verify_api_key_handles_timeout(monkeypatch):
    class FakeWebSocket:
        async def recv(self):
            raise asyncio.TimeoutError

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeWebsockets:
        @staticmethod
        def connect(*_args, **_kwargs):
            return FakeWebSocket()

    monkeypatch.setitem(sys.modules, "websockets", FakeWebsockets)

    assert await SixtyDBRealtimeSTTBackend.verify_api_key("secret") is True
    assert await SixtyDBRealtimeSTTBackend.verify_api_key("") is False


@pytest.mark.asyncio
async def test_sixtydb_session_start_send_recv_and_close(monkeypatch) -> None:
    recv_queue: asyncio.Queue[object] = asyncio.Queue()

    class FakeWebSocket:
        def __init__(self):
            self.sent: list[object] = []
            self.closed = False

        async def send(self, payload):
            self.sent.append(payload)

        async def recv(self):
            return await recv_queue.get()

        async def close(self):
            self.closed = True

    ws = FakeWebSocket()

    async def connect(*_args, **_kwargs):
        return ws

    fake_websockets = SimpleNamespace(
        connect=connect,
        exceptions=SimpleNamespace(ConnectionClosedOK=type("ConnectionClosedOK", (Exception,), {})),
    )
    monkeypatch.setitem(sys.modules, "websockets", fake_websockets)

    # Handshake frame consumed by start() before it returns.
    await recv_queue.put(json.dumps({"connection_established": {"service": "stt"}}))

    session = _SixtyDBSession(
        api_key="k",
        endpoint="wss://example",
        sample_rate_hz=16000,
        language_codes=["en"],
        context_terms=["Puripuly", "VRChat"],
        utterance_end_ms=300,
        trailing_silence_ms=50,
        connect_timeout_s=5.0,
    )

    await session.start()

    await session.send_audio(b"abc")
    await session.on_speech_end()

    await recv_queue.put(
        json.dumps(
            {
                "type": "transcription",
                "text": "Hi",
                "is_final": True,
                "speech_final": True,
            }
        )
    )

    event = await session._events.get()
    assert event.text == "Hi"

    # Mirror the managed drain order: stop() (flushes the stop control) then close().
    await session.stop()
    await asyncio.sleep(0.02)
    await recv_queue.put(None)
    await session.close()

    # start message carries config + context.
    start_msg = json.loads(ws.sent[0])
    assert start_msg["type"] == "start"
    assert start_msg["config"]["encoding"] == "linear"
    assert start_msg["config"]["continuous_mode"] is True
    assert start_msg["context"]["terms"] == ["Puripuly", "VRChat"]

    # audio sent as base64 JSON frames.
    audio_msgs = [
        json.loads(p)
        for p in ws.sent
        if isinstance(p, str) and p.strip().startswith("{") and json.loads(p).get("type") == "audio"
    ]
    assert audio_msgs
    assert base64.b64decode(audio_msgs[0]["audio"]) == b"abc"

    # stop control sent on close.
    assert any(
        isinstance(p, str) and p.strip().startswith("{") and json.loads(p).get("type") == "stop"
        for p in ws.sent
    )
    assert ws.closed is True


@pytest.mark.asyncio
async def test_sixtydb_session_start_omits_context_when_no_terms(monkeypatch) -> None:
    recv_queue: asyncio.Queue[object] = asyncio.Queue()

    class FakeWebSocket:
        def __init__(self):
            self.sent: list[object] = []
            self.closed = False

        async def send(self, payload):
            self.sent.append(payload)

        async def recv(self):
            return await recv_queue.get()

        async def close(self):
            self.closed = True

    ws = FakeWebSocket()

    async def connect(*_args, **_kwargs):
        return ws

    fake_websockets = SimpleNamespace(
        connect=connect,
        exceptions=SimpleNamespace(ConnectionClosedOK=type("ConnectionClosedOK", (Exception,), {})),
    )
    monkeypatch.setitem(sys.modules, "websockets", fake_websockets)

    await recv_queue.put(json.dumps({"connection_established": {}}))

    session = _make_session()
    await session.start()
    await recv_queue.put(None)
    await session.close()

    start_msg = json.loads(ws.sent[0])
    assert "context" not in start_msg
    assert start_msg["languages"] == ["en"]
