from __future__ import annotations

import asyncio
import urllib.error

import pytest

from puripuly_heart.providers.stt.gemini_transcribe import (
    GeminiTranscribeSTTBackend,
    gemini_transcribe_language_codes,
)


class _FakeLiveSession:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed = False
        self._queue: asyncio.Queue = asyncio.Queue()
        self._waiters: list[asyncio.Future] = []

    async def send_realtime_input(self, **kwargs) -> None:
        self.sent.append(kwargs)

    async def receive(self):
        while True:
            item = await self._queue.get()
            if item is _DONE:
                return
            yield item

    def push(self, item) -> None:
        self._queue.put_nowait(item)

    async def close(self) -> None:
        self.closed = True


_DONE = object()
_TURN_COMPLETE = object()


class _FakeLiveContext:
    def __init__(self, session: _FakeLiveSession) -> None:
        self._session = session
        self.exited = False

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True
        await self._session.close()
        return False


class _RecordingFactory:
    def __init__(self, session: _FakeLiveSession) -> None:
        self._session = session
        self.calls: list[tuple[str, object]] = []
        self.context: _FakeLiveContext | None = None

    def __call__(self, *, model: str, config: object):
        self.calls.append((model, config))
        self.context = _FakeLiveContext(self._session)
        return self.context


def _backend(
    session: _FakeLiveSession,
    *,
    language_codes: tuple[str, ...] = ("ko-KR",),
    custom_vocabulary: tuple[str, ...] = (),
    finalize_timeout_s: float = 2.0,
) -> tuple[GeminiTranscribeSTTBackend, _RecordingFactory]:
    factory = _RecordingFactory(session)
    backend = GeminiTranscribeSTTBackend(
        api_key="key",
        language_codes=language_codes,
        custom_vocabulary=custom_vocabulary,
        finalize_timeout_s=finalize_timeout_s,
        live_connect_factory=factory,
    )
    return backend, factory


def _config_dict(config: object) -> dict:
    return config.model_dump(exclude_none=True)


@pytest.mark.asyncio
async def test_open_session_rejects_invalid_sample_rate() -> None:
    backend = GeminiTranscribeSTTBackend(
        api_key="key",
        sample_rate_hz=44100,
    )
    with pytest.raises(ValueError):
        await backend.open_session()


@pytest.mark.asyncio
async def test_open_session_rejects_empty_api_key() -> None:
    backend = GeminiTranscribeSTTBackend(api_key="")
    with pytest.raises(ValueError):
        await backend.open_session()


@pytest.mark.asyncio
async def test_session_sends_manual_activity_and_verbatim_config() -> None:
    session = _FakeLiveSession()
    backend, factory = _backend(session)
    stt = await backend.open_session()
    try:
        model, config = factory.calls[0]
        assert model == "gemini-3.5-transcribe-live"
        raw = _config_dict(config)
        assert raw["response_modalities"] == ["TEXT"]
        transcription = raw["input_audio_transcription"]
        assert transcription["mode"] == "VERBATIM"
        assert transcription["language_codes"] == ["ko-KR"]
        assert raw["realtime_input_config"]["automatic_activity_detection"]["disabled"] is True
    finally:
        await stt.close()


@pytest.mark.asyncio
async def test_session_omits_language_codes_for_auto_language() -> None:
    session = _FakeLiveSession()
    backend, factory = _backend(session, language_codes=())
    stt = await backend.open_session()
    try:
        _, config = factory.calls[0]
        raw = _config_dict(config)
        assert "language_codes" not in raw["input_audio_transcription"]
    finally:
        await stt.close()


@pytest.mark.asyncio
async def test_session_maps_custom_vocabulary() -> None:
    session = _FakeLiveSession()
    backend, factory = _backend(session, custom_vocabulary=("PuriPuly", "Gemini"))
    stt = await backend.open_session()
    try:
        await stt.send_audio(b"\x00\x00" * 16)
        await _wait_for_sent(session, "audio")
        first = session.sent[0]
        assert "activity_start" in first
    finally:
        await stt.close()
    _, config = factory.calls[0]
    raw = _config_dict(config)
    assert raw["input_audio_transcription"]["custom_vocabulary"] == ["PuriPuly", "Gemini"]


@pytest.mark.asyncio
async def test_first_audio_sends_activity_start_then_audio() -> None:
    session = _FakeLiveSession()
    backend, _ = _backend(session)
    stt = await backend.open_session()
    try:
        await stt.send_audio(b"\x00\x00" * 16)
        await _wait_for_sent(session, "audio")
        assert session.sent[0].get("activity_start") is not None
        audio = session.sent[1].get("audio")
        assert audio["mime_type"] == "audio/pcm;rate=16000"
        assert audio["data"] == b"\x00\x00" * 16
    finally:
        await stt.close()


@pytest.mark.asyncio
async def test_activity_end_emits_single_final_per_finalize() -> None:
    session = _FakeLiveSession()
    backend, _ = _backend(session)
    stt = await backend.open_session()
    events_task = asyncio.create_task(_collect(stt, 1))
    try:
        await stt.send_audio(b"\x00\x00" * 16)
        await stt.on_speech_end(trailing_silence_ms=100, reason="silence")
        await _wait_for_sent(session, "activity_end")
        session.push(_final("hello"))
        session.push(_final("hello again"))
        assert session.sent[-1].get("activity_end") is not None
        events = await asyncio.wait_for(events_task, timeout=1)
        assert [event.text for event in events] == ["hello"]
    finally:
        await stt.close()


@pytest.mark.asyncio
async def test_empty_final_still_acknowledges_finalize() -> None:
    session = _FakeLiveSession()
    backend, _ = _backend(session)
    stt = await backend.open_session()
    events_task = asyncio.create_task(_collect(stt, 1))
    try:
        await stt.send_audio(b"\x00\x00" * 16)
        await stt.on_speech_end(trailing_silence_ms=100, reason="silence")
        await _wait_for_sent(session, "activity_end")
        session.push(_final(""))
        events = await asyncio.wait_for(events_task, timeout=1)
        assert [event.text for event in events] == [""]
        assert all(event.is_final for event in events)
    finally:
        await stt.close()


@pytest.mark.asyncio
async def test_activity_end_ack_emits_empty_final_without_transcription() -> None:
    session = _FakeLiveSession()
    backend, _ = _backend(session)
    stt = await backend.open_session()
    events_task = asyncio.create_task(_collect(stt, 1))
    try:
        await stt.send_audio(b"\x00\x00" * 16)
        await stt.on_speech_end(trailing_silence_ms=100, reason="silence")
        await _wait_for_sent(session, "activity_end")
        session.push(_activity_end_ack())
        events = await asyncio.wait_for(events_task, timeout=1)
        assert [event.text for event in events] == [""]
        assert all(event.is_final for event in events)
    finally:
        await stt.close()


@pytest.mark.asyncio
async def test_activity_end_ack_promotes_latest_interim_when_final_is_missing() -> None:
    session = _FakeLiveSession()
    backend, _ = _backend(session)
    stt = await backend.open_session()
    events_task = asyncio.create_task(_collect(stt, 1))
    try:
        await stt.send_audio(b"\x00\x00" * 16)
        await _wait_for_sent(session, "audio")
        session.push(_interim("first"))
        session.push(_interim("latest"))
        await stt.on_speech_end(trailing_silence_ms=100, reason="silence")
        await _wait_for_sent(session, "activity_end")
        session.push(_activity_end_ack())
        events = await asyncio.wait_for(events_task, timeout=1)
        assert [event.text for event in events] == ["latest"]
    finally:
        await stt.close()


@pytest.mark.asyncio
async def test_final_then_activity_end_ack_emits_exactly_once() -> None:
    session = _FakeLiveSession()
    backend, _ = _backend(session)
    stt = await backend.open_session()
    try:
        await stt.send_audio(b"\x00\x00" * 16)
        await stt.on_speech_end(trailing_silence_ms=100, reason="silence")
        await _wait_for_sent(session, "activity_end")
        session.push(_final("hello"))
        session.push(_activity_end_ack())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert stt._events.qsize() == 1
        event = stt._events.get_nowait()
        assert event.text == "hello"
    finally:
        await stt.close()


@pytest.mark.asyncio
async def test_empty_turn_does_not_shift_following_transcript() -> None:
    session = _FakeLiveSession()
    backend, _ = _backend(session)
    stt = await backend.open_session()
    events_task = asyncio.create_task(_collect(stt, 2))
    try:
        await stt.send_audio(b"\x00\x00" * 16)
        await stt.on_speech_end(trailing_silence_ms=100, reason="silence")
        await _wait_for_sent(session, "activity_end")
        session.push(_activity_end_ack())

        await stt.send_audio(b"\x00\x00" * 16)
        await stt.on_speech_end(trailing_silence_ms=100, reason="silence")
        await _wait_for_sent(session, "activity_end", count=2)
        session.push(_final("second"))
        session.push(_activity_end_ack())

        events = await asyncio.wait_for(events_task, timeout=1)
        assert [event.text for event in events] == ["", "second"]
    finally:
        await stt.close()


@pytest.mark.asyncio
async def test_next_activity_waits_for_previous_server_activity_end() -> None:
    session = _FakeLiveSession()
    backend, _ = _backend(session)
    stt = await backend.open_session()
    events_task = asyncio.create_task(_collect(stt, 2))
    try:
        await stt.send_audio(b"\x00\x00" * 16)
        await stt.on_speech_end(trailing_silence_ms=100, reason="silence")
        await _wait_for_sent(session, "activity_end")

        await stt.send_audio(b"\x01\x00" * 16)
        await stt.on_speech_end(trailing_silence_ms=100, reason="silence")
        await asyncio.sleep(0)
        assert sum("activity_start" in call for call in session.sent) == 1

        session.push(_activity_end_ack())
        await _wait_for_sent(session, "activity_end", count=2)
        session.push(_final("second"))
        session.push(_activity_end_ack())

        events = await asyncio.wait_for(events_task, timeout=1)
        assert [event.text for event in events] == ["", "second"]
        assert sum("activity_start" in call for call in session.sent) == 2
    finally:
        await stt.close()


@pytest.mark.asyncio
async def test_finalize_timeout_emits_fallback_then_fails_session() -> None:
    session = _FakeLiveSession()
    backend, _ = _backend(session, finalize_timeout_s=0.01)
    stt = await backend.open_session()
    try:
        await stt.send_audio(b"\x00\x00" * 16)
        await _wait_for_sent(session, "audio")
        session.push(_interim("fallback"))
        await stt.on_speech_end(trailing_silence_ms=100, reason="silence")
        await _wait_for_sent(session, "activity_end")
        events = stt.events()
        event = await asyncio.wait_for(anext(events), timeout=1)
        assert event.text == "fallback"
        with pytest.raises(Exception, match="finalize timed out"):
            await asyncio.wait_for(anext(events), timeout=1)
    finally:
        await stt.close()


@pytest.mark.asyncio
async def test_final_without_finalize_request_is_not_authoritative() -> None:
    session = _FakeLiveSession()
    backend, _ = _backend(session)
    stt = await backend.open_session()
    try:
        session.push(_final("spurious"))
        await asyncio.sleep(0)
        assert stt._events.empty()
    finally:
        await stt.close()


@pytest.mark.asyncio
async def test_many_utterances_in_one_session() -> None:
    session = _FakeLiveSession()
    backend, _ = _backend(session)
    stt = await backend.open_session()
    events_task = asyncio.create_task(_collect(stt, 2))
    try:
        await stt.send_audio(b"\x00\x00" * 16)
        await stt.on_speech_end(trailing_silence_ms=100, reason="silence")
        await _wait_for_sent(session, "activity_end")
        session.push(_final("one"))
        session.push(_activity_end_ack())

        await stt.send_audio(b"\x00\x00" * 16)
        await stt.on_speech_end(trailing_silence_ms=100, reason="silence")
        await _wait_for_sent(session, "activity_end", count=2)
        session.push(_final("two"))
        session.push(_activity_end_ack())

        events = await asyncio.wait_for(events_task, timeout=1)
        assert [event.text for event in events] == ["one", "two"]
        activity_starts = [call for call in session.sent if "activity_start" in call]
        activity_ends = [call for call in session.sent if "activity_end" in call]
        assert len(activity_starts) == 2
        assert len(activity_ends) == 2
    finally:
        await stt.close()


@pytest.mark.asyncio
async def test_stop_terminates_events_consumer() -> None:
    session = _FakeLiveSession()
    backend, _ = _backend(session)
    stt = await backend.open_session()
    try:
        await stt.send_audio(b"\x00\x00" * 16)
        await stt.stop()
        collected = []
        async for event in stt.events():
            collected.append(event)
        assert collected == []
    finally:
        await stt.close()


@pytest.mark.asyncio
async def test_close_reports_unresolved_finalizes_and_closes_session() -> None:
    session = _FakeLiveSession()
    backend, factory = _backend(session)
    stt = await backend.open_session()
    try:
        await stt.send_audio(b"\x00\x00" * 16)
        await stt.on_speech_end(trailing_silence_ms=100, reason="silence")
        await _wait_for_sent(session, "activity_end")
    finally:
        await stt.close()
    assert factory.context is not None
    assert factory.context.exited is True
    assert session.closed is True


@pytest.mark.asyncio
async def test_verify_api_key_uses_models_metadata_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_urls: list[str] = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=0):
        seen_urls.append(request.full_url)
        assert request.get_header("X-goog-api-key") == "secret"
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert await GeminiTranscribeSTTBackend.verify_api_key("") is False
    assert await GeminiTranscribeSTTBackend.verify_api_key("secret") is True
    assert seen_urls == ["https://generativelanguage.googleapis.com/v1beta/models"]


@pytest.mark.asyncio
async def test_verify_api_key_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(_request, timeout=0):
        _ = timeout
        raise urllib.error.HTTPError(
            url="https://generativelanguage.googleapis.com/v1beta/models",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(Exception, match="HTTP 401"):
        await GeminiTranscribeSTTBackend.verify_api_key("secret")


class _TurnScopedLiveSession(_FakeLiveSession):
    def __init__(self) -> None:
        super().__init__()
        self.receive_calls = 0

    async def receive(self):
        self.receive_calls += 1
        while True:
            item = await self._queue.get()
            if item is _DONE or item is _TURN_COMPLETE:
                return
            yield item


class _FailingReceiveSession(_FakeLiveSession):
    def __init__(self, exc: BaseException) -> None:
        super().__init__()
        self._exc = exc

    async def receive(self):
        raise self._exc
        yield


class _ApiFailure(Exception):
    def __init__(self, *, code: int, status: str) -> None:
        super().__init__(status)
        self.code = code
        self.status = status


@pytest.mark.asyncio
async def test_receive_loop_continues_after_turn_complete() -> None:
    session = _TurnScopedLiveSession()
    backend, _ = _backend(session)
    stt = await backend.open_session()
    events_task = asyncio.create_task(_collect(stt, 2))
    try:
        await stt.send_audio(b"\x00\x00" * 16)
        await stt.on_speech_end(trailing_silence_ms=100, reason="silence")
        await _wait_for_sent(session, "activity_end")
        session.push(_final("one"))
        session.push(_activity_end_ack())
        session.push(_TURN_COMPLETE)
        await asyncio.sleep(0)

        await stt.send_audio(b"\x00\x00" * 16)
        await stt.on_speech_end(trailing_silence_ms=100, reason="silence")
        await _wait_for_sent(session, "activity_end", count=2)
        session.push(_final("two"))
        session.push(_activity_end_ack())
        session.push(_TURN_COMPLETE)

        events = await asyncio.wait_for(events_task, timeout=1)
        assert [event.text for event in events] == ["one", "two"]
        assert session.receive_calls >= 2
    finally:
        await stt.close()


@pytest.mark.asyncio
async def test_recv_failure_logs_exception_class_and_closes_session(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = _FailingReceiveSession(_ApiFailure(code=400, status="INVALID_ARGUMENT"))
    backend, factory = _backend(session)
    with caplog.at_level("ERROR"):
        stt = await backend.open_session()
        try:
            with pytest.raises(_ApiFailure):
                async for _event in stt.events():
                    pass
        finally:
            await stt.close()
    assert factory.context is not None
    assert factory.context.exited is True
    assert session.closed is True
    combined = "\n".join(record.getMessage() for record in caplog.records)
    assert "exception_class=_ApiFailure" in combined
    assert "api_code=400" in combined
    assert "api_status=INVALID_ARGUMENT" in combined
    assert "message_kind=validation" in combined


def test_gemini_transcribe_language_codes() -> None:
    assert gemini_transcribe_language_codes("ko") == ["ko-KR"]
    assert gemini_transcribe_language_codes("en-US") == ["en-US"]
    assert gemini_transcribe_language_codes("auto") == []
    assert gemini_transcribe_language_codes(" AUTO ") == []
    assert gemini_transcribe_language_codes("Auto") == []
    assert gemini_transcribe_language_codes(None) == []
    assert gemini_transcribe_language_codes("") == []
    assert gemini_transcribe_language_codes("   ") == []
    assert gemini_transcribe_language_codes("xx") == []


def _final(text: str):
    from google.genai import types

    return types.LiveServerMessage(
        server_content=types.LiveServerContent(input_transcription=types.Transcription(text=text))
    )


def _interim(text: str):
    from google.genai import types

    return types.LiveServerMessage(
        server_content=types.LiveServerContent(
            interim_input_transcription=types.Transcription(text=text)
        )
    )


def _activity_end_ack():
    from google.genai import types

    return types.LiveServerMessage(
        voice_activity=types.VoiceActivity(
            voice_activity_type=types.VoiceActivityType.ACTIVITY_END,
        )
    )


async def _wait_for_sent(
    session: _FakeLiveSession,
    key: str,
    *,
    count: int = 1,
) -> None:
    async def ready() -> None:
        while sum(key in call for call in session.sent) < count:
            await asyncio.sleep(0)

    await asyncio.wait_for(ready(), timeout=1)


async def _collect(stt, count: int):
    events = []
    async for event in stt.events():
        events.append(event)
        if len(events) >= count:
            break
    return events
