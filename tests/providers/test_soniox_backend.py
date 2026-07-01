from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from puripuly_heart.core.stt.backend import STTBackendTranscriptEvent
from puripuly_heart.providers.stt.soniox import (
    _STOP,
    SonioxRealtimeSTTBackend,
    _FinalizeRequest,
    _SonioxSession,
)


def _make_session(*, context_terms: list[str] | None = None) -> _SonioxSession:
    return _SonioxSession(
        api_key="k",
        model="m",
        endpoint="wss://example",
        sample_rate_hz=16000,
        language_hints=["en"],
        context_terms=context_terms or [],
        keepalive_interval_s=10.0,
        trailing_silence_ms=100,
        connect_timeout_s=5.0,
    )


@pytest.mark.asyncio
async def test_soniox_backend_validates_params() -> None:
    backend = SonioxRealtimeSTTBackend(api_key="", language_hints=["en"])
    with pytest.raises(ValueError, match="api_key"):
        await backend.open_session()

    backend = SonioxRealtimeSTTBackend(api_key="k", language_hints=["en"], endpoint="")
    with pytest.raises(ValueError, match="endpoint"):
        await backend.open_session()

    backend = SonioxRealtimeSTTBackend(
        api_key="k",
        language_hints=["en"],
        keepalive_interval_s=0.0,
    )
    with pytest.raises(ValueError, match="keepalive_interval_s"):
        await backend.open_session()


@pytest.mark.asyncio
async def test_soniox_session_handles_message_errors() -> None:
    session = _make_session()

    session._handle_message("not-json")
    assert session._events.empty()

    session._handle_message(json.dumps({"error": "bad"}))
    event = session._events.get_nowait()
    assert isinstance(event, RuntimeError)


@pytest.mark.asyncio
async def test_soniox_session_collects_final_tokens() -> None:
    session = _make_session()

    message = {
        "tokens": [
            {"text": "Hello", "is_final": True, "end_ms": 100},
            {"text": " ", "is_final": True, "end_ms": 110},
            {"text": "world", "is_final": True, "end_ms": 120},
            {"text": "<fin>", "is_final": True},
        ]
    }
    session._handle_message(json.dumps(message))
    event = session._events.get_nowait()

    assert isinstance(event, STTBackendTranscriptEvent)
    assert event.text == "Hello world"


@pytest.mark.asyncio
async def test_soniox_session_merges_final_batches_by_end_ms() -> None:
    session = _make_session()

    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "Hello", "is_final": True, "end_ms": 100},
                    {"text": " world", "is_final": True, "end_ms": 200},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )
    event = session._events.get_nowait()
    assert isinstance(event, STTBackendTranscriptEvent)
    assert event.text == "Hello world"

    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": ". ", "is_final": True, "end_ms": 150},
                    {"text": "world", "is_final": True, "end_ms": 200},
                    {"text": "!", "is_final": True, "end_ms": 260},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )
    event = session._events.get_nowait()
    assert isinstance(event, STTBackendTranscriptEvent)
    assert event.text == "Hello. world!"


@pytest.mark.asyncio
async def test_soniox_session_skips_out_of_order_tokens() -> None:
    session = _make_session()

    session._handle_message(
        json.dumps({"tokens": [{"text": "A", "is_final": True, "end_ms": 100}]})
    )
    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "B", "is_final": True, "end_ms": 90},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )
    event = session._events.get_nowait()

    assert isinstance(event, STTBackendTranscriptEvent)
    assert event.text == "A"


@pytest.mark.asyncio
async def test_soniox_session_on_speech_end_enqueues_finalize() -> None:
    session = _make_session()

    await session.on_speech_end(trailing_silence_ms=240)

    finalize = await session._audio_q.get()

    assert isinstance(finalize, _FinalizeRequest)
    assert finalize.trailing_silence_ms == session.trailing_silence_ms


@pytest.mark.asyncio
async def test_soniox_session_on_speech_end_none_injects_configured_trailing_silence() -> None:
    session = _make_session()

    await session.on_speech_end(trailing_silence_ms=None)

    silence = await session._audio_q.get()
    finalize = await session._audio_q.get()

    assert isinstance(silence, bytes)
    assert len(silence) > 0
    assert isinstance(finalize, _FinalizeRequest)
    assert finalize.trailing_silence_ms == session.trailing_silence_ms


@pytest.mark.asyncio
async def test_soniox_session_empty_finalize_boundary_emits_empty_final_ack() -> None:
    session = _make_session()

    await session.on_speech_end(trailing_silence_ms=0)
    finalize = await session._audio_q.get()
    assert isinstance(finalize, _FinalizeRequest)

    session._handle_message(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))

    event = session._events.get_nowait()
    assert isinstance(event, STTBackendTranscriptEvent)
    assert event.is_final is True
    assert event.text == ""


@pytest.mark.asyncio
async def test_soniox_session_repeated_finalize_boundaries_clear_each_final_segment() -> None:
    session = _make_session()

    await session.on_speech_end(trailing_silence_ms=0)
    await session.on_speech_end(trailing_silence_ms=0)

    first_finalize = await session._audio_q.get()
    second_finalize = await session._audio_q.get()
    assert isinstance(first_finalize, _FinalizeRequest)
    assert isinstance(second_finalize, _FinalizeRequest)

    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "First", "is_final": True, "end_ms": 100},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )
    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "Second", "is_final": True, "end_ms": 200},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )

    await session.on_speech_end(trailing_silence_ms=0)
    third_finalize = await session._audio_q.get()
    assert isinstance(third_finalize, _FinalizeRequest)
    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "Third", "is_final": True, "end_ms": 300},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )

    events = [session._events.get_nowait() for _ in range(3)]
    assert [event.text for event in events] == ["First", "Second", "Third"]


@pytest.mark.asyncio
async def test_soniox_session_send_audio_and_stop() -> None:
    session = _make_session()

    await session.send_audio(b"abc")
    assert await session._audio_q.get() == b"abc"

    await session.stop()
    assert session._stopped is True
    assert await session._audio_q.get() is _STOP


@pytest.mark.asyncio
async def test_soniox_session_events_yield_and_raise() -> None:
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
async def test_soniox_verify_api_key_handles_timeout(monkeypatch):
    class FakeWebSocket:
        def __init__(self):
            self.sent = []

        async def send(self, payload):
            self.sent.append(payload)

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

    monkeypatch.setitem(__import__("sys").modules, "websockets", FakeWebsockets)

    assert await SonioxRealtimeSTTBackend.verify_api_key("secret") is True


@pytest.mark.asyncio
async def test_soniox_session_start_send_recv_and_close(monkeypatch) -> None:
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

    session = _SonioxSession(
        api_key="k",
        model="m",
        endpoint="wss://example",
        sample_rate_hz=16000,
        language_hints=["en"],
        context_terms=["Puripuly", "VRChat"],
        keepalive_interval_s=0.01,
        trailing_silence_ms=50,
        connect_timeout_s=5.0,
    )

    await session.start()

    await session.send_audio(b"abc")
    await session.on_speech_end()

    await recv_queue.put(
        json.dumps(
            {"tokens": [{"text": "Hi", "is_final": True}, {"text": "<fin>", "is_final": True}]}
        )
    )

    event = await session._events.get()
    assert event.text == "Hi"

    await asyncio.sleep(0.02)
    await recv_queue.put(None)
    await session.close()

    config = json.loads(ws.sent[0])
    assert config["context"]["terms"] == ["Puripuly", "VRChat"]

    payloads = [
        payload
        for payload in ws.sent
        if isinstance(payload, str) and payload.strip().startswith("{")
    ]
    assert any(json.loads(p).get("type") == "finalize" for p in payloads)
    assert any(json.loads(p).get("type") == "keepalive" for p in payloads)
    assert b"abc" in ws.sent
    assert ws.closed is True


@pytest.mark.asyncio
async def test_soniox_session_start_omits_context_when_no_terms(monkeypatch) -> None:
    class FakeWebSocket:
        def __init__(self):
            self.sent: list[object] = []
            self.closed = False

        async def send(self, payload):
            self.sent.append(payload)

        async def recv(self):
            return None

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

    session = _make_session()
    await session.start()
    await session.close()

    config = json.loads(ws.sent[0])
    assert "context" not in config
