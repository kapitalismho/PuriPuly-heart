from __future__ import annotations

import asyncio
from urllib.parse import parse_qs, urlsplit

import pytest

from puripuly_heart.providers.stt.elevenlabs_scribe import (
    ElevenLabsScribeSTTBackend,
    scribe_keyterms,
    scribe_language_code,
)
from tests.helpers.scoped_provider_sessions import FakeScribeWebSocket


def _backend(
    connection: FakeScribeWebSocket,
    *,
    language_code: str | None = "ko",
    keyterms: tuple[str, ...] = (),
    keepalive_interval_s: float = 10.0,
) -> tuple[ElevenLabsScribeSTTBackend, list]:
    factories: list = []

    async def factory(url, *, additional_headers):
        assert additional_headers == {"xi-api-key": "key"}
        factories.append(parse_qs(urlsplit(url).query))
        return connection

    backend = ElevenLabsScribeSTTBackend(
        api_key="key",
        language_code=language_code,
        keyterms=keyterms,
        keepalive_interval_s=keepalive_interval_s,
        websocket_factory=factory,
    )
    return backend, factories


def _emit(session, event, data) -> None:
    session._connection.emit(event, data)


@pytest.mark.asyncio
async def test_open_session_rejects_invalid_sample_rate() -> None:
    backend = ElevenLabsScribeSTTBackend(api_key="key", sample_rate_hz=44100)
    with pytest.raises(ValueError):
        await backend.open_session()


@pytest.mark.asyncio
async def test_open_session_rejects_empty_api_key() -> None:
    backend = ElevenLabsScribeSTTBackend(api_key="")
    with pytest.raises(ValueError):
        await backend.open_session()


@pytest.mark.asyncio
async def test_session_uses_manual_commit_without_vad_parameters() -> None:
    connection = FakeScribeWebSocket()
    backend, factories = _backend(connection, language_code="ko")
    session = await backend.open_session()
    try:
        raw = factories[0]
        assert raw["commit_strategy"] == ["manual"]
        assert "vad_threshold" not in raw
        assert "vad_silence_threshold_secs" not in raw
        assert "min_speech_duration_ms" not in raw
        assert "min_silence_duration_ms" not in raw
        assert raw["audio_format"] == ["pcm_16000"]
        assert raw["language_code"] == ["ko"]
        assert raw["model_id"] == ["scribe_v3_live"]
        assert raw["diarize"] == ["false"]
        assert raw["include_timestamps"] == ["false"]
        assert raw["filter_background_audio"] == ["false"]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_session_omits_language_code_for_auto_language() -> None:
    connection = FakeScribeWebSocket()
    backend, factories = _backend(connection, language_code=None)
    session = await backend.open_session()
    try:
        assert "language_code" not in factories[0]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_committed_transcript_is_authoritative_final() -> None:
    connection = FakeScribeWebSocket()
    backend, _ = _backend(connection)
    session = await backend.open_session()
    events_task = asyncio.create_task(_collect(session, 1))
    try:
        await session.on_speech_end()
        _emit(
            session,
            "committed_transcript",
            {"message_type": "committed_transcript", "text": "hello"},
        )
        events = await asyncio.wait_for(events_task, timeout=1)
        assert [event.text for event in events] == ["hello"]
        assert all(event.is_final for event in events)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_partial_and_final_transcripts_are_not_authoritative() -> None:
    connection = FakeScribeWebSocket()
    backend, _ = _backend(connection)
    session = await backend.open_session()
    try:
        _emit(session, "partial_transcript", {"message_type": "partial_transcript", "text": "part"})
        _emit(session, "final_transcript", {"message_type": "final_transcript", "text": "settled"})
        await asyncio.sleep(0)
        assert session._event_projection._legacy_events.empty()
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_keepalive_sends_empty_audio_when_idle() -> None:
    connection = FakeScribeWebSocket()
    backend, _ = _backend(connection, keepalive_interval_s=0.01)
    session = await backend.open_session()
    try:
        task = session._keepalive_task
        assert session.keepalive_interval_s == 0.01
        assert task is not None
        session._last_send_at = 0.0
        deadline = asyncio.get_running_loop().time() + 1.0
        while asyncio.get_running_loop().time() < deadline:
            if connection.sent:
                break
            if task.done():
                task.result()
                break
            await asyncio.sleep(0.01)
        assert connection.sent == [
            {
                "message_type": "input_audio_chunk",
                "audio_base_64": "",
                "sample_rate": 16000,
                "commit": False,
            }
        ]
    finally:
        await session.close()
    assert connection.closed is True


@pytest.mark.asyncio
async def test_speech_end_sends_explicit_commit() -> None:
    connection = FakeScribeWebSocket()
    backend, _ = _backend(connection)
    session = await backend.open_session()
    try:
        await session.send_audio(b"\x00\x00" * 16)
        await session.on_speech_end(trailing_silence_ms=100, reason="silence")
        assert connection.commits == 1
        assert len(connection.sent) == 2
        assert connection.sent[0]["audio_base_64"]
        assert connection.sent[1] == {
            "message_type": "input_audio_chunk",
            "audio_base_64": "",
            "sample_rate": 16000,
            "commit": True,
        }
    finally:
        await session.close()


@pytest.mark.parametrize(
    "event_name",
    (
        "close",
        "session_time_limit_exceeded",
        "insufficient_audio_activity",
    ),
)
@pytest.mark.asyncio
async def test_recoverable_scribe_events_end_stream_without_raising(event_name: str) -> None:
    connection = FakeScribeWebSocket()
    backend, _ = _backend(connection)
    session = await backend.open_session()
    events_task = asyncio.create_task(_collect(session, 1))
    try:
        if event_name == "close":
            connection.emit_close()
        else:
            _emit(session, event_name, {"message_type": event_name})
        events = await asyncio.wait_for(events_task, timeout=1)
        assert events == []
        await session.send_audio(b"\x00\x00" * 16)
        await session.on_speech_end(trailing_silence_ms=100, reason="silence")
        assert connection.sent == []
        assert connection.commits == 0
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_send_failure_ends_scribe_session_without_raising() -> None:
    class RaisingConnection(FakeScribeWebSocket):
        async def send(self, payload: dict) -> None:
            _ = payload
            raise RuntimeError("closed")

    connection = RaisingConnection()
    backend, _ = _backend(connection)
    session = await backend.open_session()
    events_task = asyncio.create_task(_collect(session, 1))
    try:
        await session.send_audio(b"\x00\x00" * 16)
        events = await asyncio.wait_for(events_task, timeout=1)
        assert events == []
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_fatal_error_is_not_hidden_by_later_close_or_send_failure() -> None:
    class RaisingConnection(FakeScribeWebSocket):
        async def send(self, payload: dict) -> None:
            _ = payload
            raise RuntimeError("closed")

    connection = RaisingConnection()
    backend, _ = _backend(connection)
    session = await backend.open_session()
    events_task = asyncio.create_task(_collect_error(session))
    try:
        _emit(session, "quota_exceeded", {"message_type": "quota_exceeded", "text": ""})
        await asyncio.sleep(0)
        await session.send_audio(b"\x00\x00" * 16)
        connection.emit_close()
        error = await asyncio.wait_for(events_task, timeout=1)
        assert "quota_exceeded" in str(error)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_provider_error_events_fail_the_session() -> None:
    connection = FakeScribeWebSocket()
    backend, _ = _backend(connection)
    session = await backend.open_session()
    events_task = asyncio.create_task(_collect_error(session))
    try:
        _emit(session, "quota_exceeded", {"message_type": "quota_exceeded", "text": ""})
        error = await asyncio.wait_for(events_task, timeout=1)
        assert "quota_exceeded" in str(error)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_close_terminates_event_stream() -> None:
    connection = FakeScribeWebSocket()
    backend, _ = _backend(connection)
    session = await backend.open_session()
    try:
        connection.emit_close()
        assert await asyncio.wait_for(_collect(session, 1), timeout=1) == []
        assert session._receive_task.done()
        await session.close()
        assert connection.closed is True
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_verify_api_key_succeeds_on_session_started() -> None:
    connection = FakeScribeWebSocket()
    seen_query = []

    async def factory(url, *, additional_headers):
        seen_query.append(parse_qs(urlsplit(url).query))
        assert additional_headers == {"xi-api-key": "secret"}
        return connection

    assert await ElevenLabsScribeSTTBackend.verify_api_key("secret", websocket_factory=factory)
    assert connection.closed
    assert seen_query[0]["model_id"] == ["scribe_v3_live"]


@pytest.mark.asyncio
async def test_verify_api_key_reports_auth_rejection_as_unauthorized() -> None:
    connection = FakeScribeWebSocket(None)
    connection.push("auth_error")
    with pytest.raises(RuntimeError) as raised:
        await ElevenLabsScribeSTTBackend.verify_api_key(
            "bad-secret", websocket_factory=connection.connect
        )
    assert raised.value.failure_diagnostics.category == "auth"
    assert raised.value.failure_diagnostics.code == "stt.scribe_auth_error"
    assert connection.closed


@pytest.mark.asyncio
async def test_verify_api_key_reports_close_before_session_start() -> None:
    connection = FakeScribeWebSocket(None)
    connection.emit_close()
    with pytest.raises(Exception, match="closed before session start"):
        await ElevenLabsScribeSTTBackend.verify_api_key(
            "secret", websocket_factory=connection.connect
        )
    assert connection.closed


@pytest.mark.asyncio
async def test_verify_api_key_times_out_without_server_signal() -> None:
    connection = FakeScribeWebSocket(None)

    with pytest.raises(TimeoutError, match="timed out waiting for session start"):
        await ElevenLabsScribeSTTBackend.verify_api_key(
            "secret",
            websocket_factory=connection.connect,
            settle_timeout_s=0.05,
        )

    assert connection.closed is True


@pytest.mark.asyncio
async def test_verify_api_key_rejects_empty_key_without_connecting() -> None:
    async def factory(url, *, additional_headers):
        raise AssertionError("factory must not be called for an empty key")

    assert await ElevenLabsScribeSTTBackend.verify_api_key("", websocket_factory=factory) is False


def test_scribe_keyterms_normalization() -> None:
    terms = tuple(f"term-{index:02d}-0123456789" for index in range(60))
    normalized = scribe_keyterms(terms)
    assert len(normalized) == 50
    assert all(len(term) <= 20 for term in normalized)
    assert scribe_keyterms(("", "  ", "dup", "dup")) == ("dup",)


def test_scribe_language_code() -> None:
    assert scribe_language_code("ko") == "ko"
    assert scribe_language_code("auto") is None
    assert scribe_language_code(None) is None
    assert scribe_language_code("") is None


async def _collect(session, count: int):
    events = []
    async for event in session.events():
        events.append(event)
        if len(events) >= count:
            break
    return events


async def _collect_error(session):
    try:
        async for event in session.events():
            if isinstance(event, BaseException):
                return event
    except Exception as exc:
        return exc
    return None
