from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from puripuly_heart.core.stt.custom import (
    CUSTOM_STT_VALIDATION_AUTH_FAILURE,
    CUSTOM_STT_VALIDATION_COMPATIBILITY_MISMATCH,
    CUSTOM_STT_VALIDATION_MODEL_UNAVAILABLE,
    CUSTOM_STT_VALIDATION_READY,
    CUSTOM_STT_VALIDATION_TRANSCRIPTION_UNVERIFIED,
    CUSTOM_STT_VALIDATION_UNREACHABLE,
    classify_http_failure,
    sanitize_custom_stt_text,
    sanitize_endpoint_for_display,
)
from puripuly_heart.core.stt.custom_connection import validate_custom_stt_connection
from puripuly_heart.providers.stt.custom import CustomSTTBackend


class _FakeResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = payload if isinstance(payload, str) else json.dumps(payload)

    def json(self) -> object:
        if isinstance(self._payload, str):
            raise ValueError("not json")
        return self._payload


class _FakeAsyncClient:
    def __init__(self, handler) -> None:
        self._handler = handler
        self.closed = False

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        return await self._handler(url, **kwargs)

    async def aclose(self) -> None:
        self.closed = True

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


class _FakeWebSocket:
    def __init__(self, messages: list[object], *, hang: bool = False) -> None:
        self.sent: list[str] = []
        self._messages = list(messages)
        self._hang = hang
        self._closed = asyncio.Event()
        self.closed = False

    async def send(self, payload: str) -> None:
        self.sent.append(payload)

    def __aiter__(self) -> _FakeWebSocket:
        return self

    async def __anext__(self) -> object:
        if self._messages:
            return self._messages.pop(0)
        if self._hang:
            await self._closed.wait()
        raise StopAsyncIteration

    async def recv(self) -> object:
        if not self._messages:
            raise TimeoutError("no message")
        return self._messages.pop(0)

    async def close(self) -> None:
        self.closed = True
        self._closed.set()


def _backend(**kwargs: Any) -> CustomSTTBackend:
    values = {
        "mode": "offline",
        "compatibility": "openai_transcription",
        "endpoint": "http://127.0.0.1:8000",
        "model": "whisper-1",
        "api_key": "sk-secret-value",
        "source_language": "en",
    }
    values.update(kwargs)
    return CustomSTTBackend(**values)


@pytest.mark.asyncio
async def test_offline_session_emits_one_final_transcript() -> None:
    captured: dict[str, Any] = {}

    async def handler(url: str, **kwargs: Any) -> _FakeResponse:
        captured["url"] = url
        captured["headers"] = kwargs.get("headers")
        captured["data"] = kwargs.get("data")
        captured["files"] = kwargs.get("files")
        return _FakeResponse(200, {"text": "hello there"})

    backend = _backend(http_client_factory=lambda **_: _FakeAsyncClient(handler))
    session = await backend.open_session()
    await session.send_audio(b"\x00\x00" * 160)
    await session.on_speech_end()
    event = await anext(session.events())
    await session.close()

    assert event.is_final is True
    assert event.text == "hello there"
    assert captured["url"].endswith("/v1/audio/transcriptions")
    assert captured["data"]["model"] == "whisper-1"
    assert captured["data"]["language"] == "en"
    assert captured["headers"]["Authorization"] == "Bearer sk-secret-value"


@pytest.mark.asyncio
async def test_offline_session_preserves_utterance_order() -> None:
    texts = ["first", "second"]

    async def handler(url: str, **kwargs: Any) -> _FakeResponse:
        _ = url, kwargs
        return _FakeResponse(200, {"text": texts.pop(0)})

    backend = _backend(http_client_factory=lambda **_: _FakeAsyncClient(handler))
    session = await backend.open_session()
    await session.send_audio(b"\x00\x00" * 80)
    await session.on_speech_end()
    await session.send_audio(b"\x00\x00" * 80)
    await session.on_speech_end()
    first = await anext(session.events())
    second = await anext(session.events())
    await session.close()

    assert [first.text, second.text] == ["first", "second"]
    assert first.is_final and second.is_final


@pytest.mark.asyncio
async def test_offline_empty_result_still_finalizes() -> None:
    async def handler(url: str, **kwargs: Any) -> _FakeResponse:
        _ = url, kwargs
        return _FakeResponse(200, {"text": "   "})

    backend = _backend(http_client_factory=lambda **_: _FakeAsyncClient(handler))
    session = await backend.open_session()
    await session.send_audio(b"\x00\x00" * 80)
    await session.on_speech_end()
    event = await anext(session.events())
    await session.close()

    assert event.is_final is True
    assert event.text == ""


@pytest.mark.asyncio
async def test_offline_failure_does_not_leak_secret() -> None:
    async def handler(url: str, **kwargs: Any) -> _FakeResponse:
        _ = url, kwargs
        return _FakeResponse(401, {"error": "invalid sk-secret-value"})

    backend = _backend(http_client_factory=lambda **_: _FakeAsyncClient(handler))
    session = await backend.open_session()
    await session.send_audio(b"\x00\x00" * 80)
    await session.on_speech_end()
    event = await anext(session.events())
    await session.close()
    assert event.is_final is True
    assert event.text == ""


@pytest.mark.asyncio
async def test_offline_session_merges_extra_into_form_and_query() -> None:
    captured: dict[str, Any] = {}

    async def handler(url: str, **kwargs: Any) -> _FakeResponse:
        captured["url"] = url
        captured["data"] = kwargs.get("data")
        return _FakeResponse(200, {"text": "hello there"})

    backend = _backend(
        http_client_factory=lambda **_: _FakeAsyncClient(handler),
        extra={"prompt": "custom hint", "max_tokens": 32},
    )
    session = await backend.open_session()
    await session.send_audio(b"\x00\x00" * 160)
    await session.on_speech_end()
    event = await anext(session.events())
    await session.close()

    assert event.text == "hello there"
    assert captured["data"]["prompt"] == "custom hint"
    assert captured["data"]["max_tokens"] == "32"
    assert "prompt=" in captured["url"]


@pytest.mark.asyncio
async def test_realtime_session_appends_extra_query_params() -> None:
    ws = _FakeWebSocket(
        [
            json.dumps({"type": "session.updated"}),
            json.dumps(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "text": "hello",
                }
            ),
        ],
        hang=True,
    )

    async def connect(url: str, **kwargs: object) -> _FakeWebSocket:
        captured["url"] = url
        _ = kwargs
        return ws

    captured: dict[str, Any] = {}
    backend = CustomSTTBackend(
        mode="realtime",
        compatibility="openai_realtime",
        endpoint="http://127.0.0.1:8000",
        model="gpt-4o-mini-transcribe",
        extra={"model": "speaches-model", "language": "en"},
        websocket_connect=connect,
    )
    session = await backend.open_session()
    await session.send_audio(b"\x00\x00" * 80)
    await session.on_speech_end()
    event = await anext(session.events())
    await session.close()

    assert event.text == "hello"
    assert captured["url"].startswith("ws://127.0.0.1:8000/v1/realtime?")
    assert "model=speaches-model" in captured["url"]
    assert "language=en" in captured["url"]


@pytest.mark.asyncio
async def test_realtime_validation_with_extra_model_query() -> None:
    ws = _FakeWebSocket([json.dumps({"type": "session.updated"})])

    async def connect(url: str, **kwargs: object) -> _FakeWebSocket:
        captured["url"] = url
        _ = kwargs
        return ws

    captured: dict[str, Any] = {}
    result = await validate_custom_stt_connection(
        mode="realtime",
        compatibility="openai_realtime",
        endpoint="http://127.0.0.1:8000",
        model="gpt-4o-mini-transcribe",
        extra={"model": "speaches-model"},
        websocket_connect=connect,
    )
    assert result.status == CUSTOM_STT_VALIDATION_TRANSCRIPTION_UNVERIFIED
    assert "model=speaches-model" in captured["url"]


def test_normalize_custom_stt_extra_rejects_secrets_and_reserved() -> None:
    from puripuly_heart.core.stt.custom import normalize_custom_stt_extra

    assert normalize_custom_stt_extra(None) == {}
    assert normalize_custom_stt_extra({"prompt": "x", "num": 3}) == {"prompt": "x", "num": 3}
    with pytest.raises(Exception):
        normalize_custom_stt_extra({"api_key": "secret"})
    with pytest.raises(Exception):
        normalize_custom_stt_extra({"file": "x"})


@pytest.mark.asyncio
async def test_offline_validation_sends_extra_form_fields() -> None:
    captured: dict[str, Any] = {}

    async def handler(url: str, **kwargs: Any) -> _FakeResponse:
        captured["data"] = kwargs.get("data")
        captured["url"] = url
        return _FakeResponse(200, {"text": ""})

    result = await validate_custom_stt_connection(
        mode="offline",
        compatibility="openai_transcription",
        endpoint="http://127.0.0.1:8000",
        model="whisper-1",
        extra={"prompt": "hello"},
        http_client_factory=lambda **_: _FakeAsyncClient(handler),
    )
    assert result.status == CUSTOM_STT_VALIDATION_READY
    assert captured["data"]["prompt"] == "hello"
    assert "prompt=hello" in captured["url"]


@pytest.mark.asyncio
async def test_streaming_session_discards_partials_and_keeps_finals() -> None:
    ws = _FakeWebSocket(
        [
            json.dumps({"type": "session.updated"}),
            json.dumps(
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "text": "hel",
                }
            ),
            json.dumps(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "text": "hello",
                }
            ),
        ],
        hang=True,
    )

    async def connect(*args: object, **kwargs: object) -> _FakeWebSocket:
        _ = args, kwargs
        return ws

    backend = CustomSTTBackend(
        mode="realtime",
        compatibility="openai_realtime",
        endpoint="http://127.0.0.1:8000",
        model="gpt-4o-mini-transcribe",
        websocket_connect=connect,
    )
    session = await backend.open_session()
    await session.send_audio(b"\x00\x00" * 80)
    await session.on_speech_end()
    event = await anext(session.events())
    await session.close()

    assert event.is_final is True
    assert event.text == "hello"
    sent_types = [json.loads(item)["type"] for item in ws.sent]
    assert "session.update" in sent_types
    assert "input_audio_buffer.append" in sent_types
    assert "input_audio_buffer.commit" in sent_types


@pytest.mark.asyncio
async def test_offline_validation_ready_requires_transcription_shape() -> None:
    async def handler(url: str, **kwargs: Any) -> _FakeResponse:
        _ = url, kwargs
        return _FakeResponse(200, {"text": ""})

    result = await validate_custom_stt_connection(
        mode="offline",
        compatibility="openai_transcription",
        endpoint="http://127.0.0.1:8000",
        model="whisper-1",
        http_client_factory=lambda **_: _FakeAsyncClient(handler),
    )
    assert result.status == CUSTOM_STT_VALIDATION_READY


@pytest.mark.asyncio
async def test_offline_validation_classifies_auth_and_unreachable() -> None:
    async def unauthorized(url: str, **kwargs: Any) -> _FakeResponse:
        _ = url, kwargs
        return _FakeResponse(401, {"error": "nope"})

    auth = await validate_custom_stt_connection(
        mode="offline",
        compatibility="openai_transcription",
        endpoint="http://127.0.0.1:8000",
        model="whisper-1",
        http_client_factory=lambda **_: _FakeAsyncClient(unauthorized),
    )
    assert auth.status == CUSTOM_STT_VALIDATION_AUTH_FAILURE

    def boom(**kwargs: Any) -> _FakeAsyncClient:
        _ = kwargs

        async def handler(url: str, **inner: Any) -> _FakeResponse:
            _ = url, inner
            raise httpx.ConnectError("down")

        return _FakeAsyncClient(handler)

    unreachable = await validate_custom_stt_connection(
        mode="offline",
        compatibility="openai_transcription",
        endpoint="http://127.0.0.1:8000",
        model="whisper-1",
        http_client_factory=boom,
    )
    assert unreachable.status == CUSTOM_STT_VALIDATION_UNREACHABLE


def test_endpoint_and_secret_sanitization() -> None:
    assert (
        sanitize_endpoint_for_display("https://user:pass@example.test:8443/v1?api_key=secret")
        == "https://example.test:8443/v1"
    )
    assert "secret" not in sanitize_endpoint_for_display("//user:secret@example.test/v1")
    assert sanitize_endpoint_for_display("http://example.test:notaport/v1")
    sanitized = sanitize_custom_stt_text(
        "Bearer sk-secret-value failed for https://user:token@host/v1",
        secret="sk-secret-value",
    )
    assert "sk-secret-value" not in sanitized
    assert "token" not in sanitized
    assert "[redacted]" in sanitized


def test_http_404_model_failure_is_model_unavailable() -> None:
    assert (
        classify_http_failure(404, "model_not_found: unknown model")
        == CUSTOM_STT_VALIDATION_MODEL_UNAVAILABLE
    )
    assert (
        classify_http_failure(404, "no such route") == CUSTOM_STT_VALIDATION_COMPATIBILITY_MISMATCH
    )


@pytest.mark.asyncio
async def test_realtime_validation_handshake_is_not_ready() -> None:
    ws = _FakeWebSocket([json.dumps({"type": "session.updated"})])

    async def connect(*args: object, **kwargs: object) -> _FakeWebSocket:
        _ = args, kwargs
        return ws

    result = await validate_custom_stt_connection(
        mode="realtime",
        compatibility="openai_realtime",
        endpoint="http://127.0.0.1:8000",
        model="gpt-4o-mini-transcribe",
        websocket_connect=connect,
    )
    assert result.status == CUSTOM_STT_VALIDATION_TRANSCRIPTION_UNVERIFIED


@pytest.mark.asyncio
async def test_realtime_connect_auth_failure_is_not_unreachable() -> None:
    async def connect(*args: object, **kwargs: object) -> _FakeWebSocket:
        _ = args, kwargs
        raise ConnectionError("401 unauthorized")

    result = await validate_custom_stt_connection(
        mode="realtime",
        compatibility="openai_realtime",
        endpoint="http://127.0.0.1:8000",
        model="gpt-4o-mini-transcribe",
        websocket_connect=connect,
    )
    assert result.status == CUSTOM_STT_VALIDATION_AUTH_FAILURE


@pytest.mark.asyncio
async def test_realtime_ignores_finals_before_speech_end() -> None:
    ws = _FakeWebSocket(
        [
            json.dumps({"type": "session.updated"}),
            json.dumps(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "text": "early",
                }
            ),
        ],
        hang=True,
    )

    async def connect(*args: object, **kwargs: object) -> _FakeWebSocket:
        _ = args, kwargs
        return ws

    backend = CustomSTTBackend(
        mode="realtime",
        compatibility="openai_realtime",
        endpoint="http://127.0.0.1:8000",
        model="gpt-4o-mini-transcribe",
        websocket_connect=connect,
    )
    session = await backend.open_session()
    await session.send_audio(b"\x00\x00" * 80)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(anext(session.events()), timeout=0.05)
    await session.close()


@pytest.mark.asyncio
async def test_realtime_close_does_not_raise_cancelled_error() -> None:
    ws = _FakeWebSocket([json.dumps({"type": "session.updated"})], hang=True)

    async def connect(*args: object, **kwargs: object) -> _FakeWebSocket:
        _ = args, kwargs
        return ws

    backend = CustomSTTBackend(
        mode="realtime",
        compatibility="openai_realtime",
        endpoint="http://127.0.0.1:8000",
        model="gpt-4o-mini-transcribe",
        websocket_connect=connect,
    )
    session = await backend.open_session()
    await session.close()
    assert ws.closed is True
