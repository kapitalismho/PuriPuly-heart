from __future__ import annotations

import asyncio

import pytest

from puripuly_heart.providers.stt.elevenlabs_scribe import (
    ElevenLabsScribeSTTBackend,
    scribe_keyterms,
    scribe_language_code,
)


class _FakeConnection:
    def __init__(self) -> None:
        self.handlers: dict = {}
        self.sent: list[dict] = []
        self.commits = 0
        self.closed = False

    def on(self, event, callback) -> None:
        self.handlers[event] = callback

    async def send(self, payload: dict) -> None:
        self.sent.append(payload)

    async def commit(self) -> None:
        self.commits += 1

    async def close(self) -> None:
        self.closed = True


def _backend(
    connection: _FakeConnection,
    *,
    language_code: str | None = "ko",
    keyterms: tuple[str, ...] = (),
) -> tuple[ElevenLabsScribeSTTBackend, list]:
    factories: list = []

    async def factory(options):
        factories.append(options)
        return connection

    backend = ElevenLabsScribeSTTBackend(
        api_key="key",
        language_code=language_code,
        keyterms=keyterms,
        scribe_connect_factory=factory,
    )
    return backend, factories


def _emit(session, event, data) -> None:
    connection = session._connection
    handler = connection.handlers[event]
    handler(data)


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
    connection = _FakeConnection()
    backend, factories = _backend(connection, language_code="ko")
    session = await backend.open_session()
    try:
        options = factories[0]
        raw = options.model_dump(exclude_none=True) if hasattr(options, "model_dump") else options
        commit_strategy = raw["commit_strategy"]
        commit_strategy = getattr(commit_strategy, "value", commit_strategy)
        assert commit_strategy == "manual"
        assert "vad_threshold" not in raw
        assert "vad_silence_threshold_secs" not in raw
        assert "min_speech_duration_ms" not in raw
        assert "min_silence_duration_ms" not in raw
        assert raw["audio_format"] == "pcm_16000"
        assert raw["sample_rate"] == 16000
        assert raw["language_code"] == "ko"
        assert raw["model_id"] == "scribe_v2_realtime"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_session_omits_language_code_for_auto_language() -> None:
    connection = _FakeConnection()
    backend, factories = _backend(connection, language_code=None)
    session = await backend.open_session()
    try:
        options = factories[0]
        if hasattr(options, "model_dump"):
            assert getattr(options, "language_code", None) is None
        else:
            assert options["language_code"] is None
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_committed_transcript_is_authoritative_final() -> None:
    connection = _FakeConnection()
    backend, _ = _backend(connection)
    session = await backend.open_session()
    events_task = asyncio.create_task(_collect(session, 1))
    try:
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
    connection = _FakeConnection()
    backend, _ = _backend(connection)
    session = await backend.open_session()
    try:
        _emit(session, "partial_transcript", {"message_type": "partial_transcript", "text": "part"})
        _emit(session, "final_transcript", {"message_type": "final_transcript", "text": "settled"})
        await asyncio.sleep(0)
        assert session._events.empty()
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_speech_end_sends_explicit_commit() -> None:
    connection = _FakeConnection()
    backend, _ = _backend(connection)
    session = await backend.open_session()
    try:
        await session.send_audio(b"\x00\x00" * 16)
        await session.on_speech_end(trailing_silence_ms=100, reason="silence")
        assert connection.commits == 1
        assert len(connection.sent) == 1
        assert connection.sent[0]["audio_base_64"]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_provider_error_events_fail_the_session() -> None:
    connection = _FakeConnection()
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
    connection = _FakeConnection()
    backend, _ = _backend(connection)
    session = await backend.open_session()
    try:
        _emit(session, "close", {"message_type": "close"})
        await asyncio.sleep(0.05)
        assert session._queue_task is not None
        await session.close()
        assert connection.closed is True
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_verify_api_key_succeeds_on_session_started() -> None:
    from elevenlabs.realtime import RealtimeEvents

    connection = _FakeConnection()
    seen_options: list = []

    async def factory(options):
        seen_options.append(options)
        return connection

    async def emit_started() -> None:
        for _ in range(200):
            handler = connection.handlers.get(RealtimeEvents.SESSION_STARTED)
            if handler is not None:
                handler({"message_type": "session_started"})
                return
            await asyncio.sleep(0.01)
        raise AssertionError("session_started handler was never registered")

    task = asyncio.create_task(
        ElevenLabsScribeSTTBackend.verify_api_key("secret", scribe_connect_factory=factory)
    )
    emitter = asyncio.create_task(emit_started())
    try:
        assert await asyncio.wait_for(task, timeout=5) is True
    finally:
        emitter.cancel()
        await asyncio.gather(emitter, return_exceptions=True)

    assert connection.closed is True
    assert seen_options and seen_options[0]["model_id"] == "scribe_v2_realtime"


@pytest.mark.asyncio
async def test_verify_api_key_reports_auth_rejection_as_unauthorized() -> None:
    from elevenlabs.realtime import RealtimeEvents

    connection = _FakeConnection()

    async def factory(options):
        _ = options
        return connection

    async def emit_auth_error() -> None:
        for _ in range(200):
            handler = connection.handlers.get(RealtimeEvents.ERROR)
            if handler is not None:
                handler({"message_type": "auth_error"})
                return
            await asyncio.sleep(0.01)
        raise AssertionError("error handler was never registered")

    task = asyncio.create_task(
        ElevenLabsScribeSTTBackend.verify_api_key("bad-secret", scribe_connect_factory=factory)
    )
    emitter = asyncio.create_task(emit_auth_error())
    try:
        with pytest.raises(Exception, match="unauthorized"):
            await asyncio.wait_for(task, timeout=5)
    finally:
        emitter.cancel()
        await asyncio.gather(emitter, return_exceptions=True)

    assert connection.closed is True


@pytest.mark.asyncio
async def test_verify_api_key_reports_close_before_session_start() -> None:
    from elevenlabs.realtime import RealtimeEvents

    connection = _FakeConnection()

    async def factory(options):
        _ = options
        return connection

    async def emit_close() -> None:
        for _ in range(200):
            handler = connection.handlers.get(RealtimeEvents.CLOSE)
            if handler is not None:
                handler({"message_type": "close"})
                return
            await asyncio.sleep(0.01)
        raise AssertionError("close handler was never registered")

    task = asyncio.create_task(
        ElevenLabsScribeSTTBackend.verify_api_key("secret", scribe_connect_factory=factory)
    )
    emitter = asyncio.create_task(emit_close())
    try:
        with pytest.raises(Exception, match="closed before session start"):
            await asyncio.wait_for(task, timeout=5)
    finally:
        emitter.cancel()
        await asyncio.gather(emitter, return_exceptions=True)


@pytest.mark.asyncio
async def test_verify_api_key_times_out_without_server_signal() -> None:
    connection = _FakeConnection()

    async def factory(options):
        _ = options
        return connection

    with pytest.raises(TimeoutError, match="timed out waiting for session start"):
        await ElevenLabsScribeSTTBackend.verify_api_key(
            "secret",
            scribe_connect_factory=factory,
            settle_timeout_s=0.05,
        )

    assert connection.closed is True


@pytest.mark.asyncio
async def test_verify_api_key_rejects_empty_key_without_connecting() -> None:
    async def factory(options):  # pragma: no cover - must not be called
        _ = options
        raise AssertionError("factory must not be called for an empty key")

    assert (
        await ElevenLabsScribeSTTBackend.verify_api_key("", scribe_connect_factory=factory) is False
    )


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
