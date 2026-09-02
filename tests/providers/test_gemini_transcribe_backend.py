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


class _FakeLiveContext:
    def __init__(self, session: _FakeLiveSession) -> None:
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _RecordingFactory:
    def __init__(self, session: _FakeLiveSession) -> None:
        self._session = session
        self.calls: list[tuple[str, object]] = []

    def __call__(self, *, model: str, config: object):
        self.calls.append((model, config))
        return _FakeLiveContext(self._session)


def _backend(
    session: _FakeLiveSession,
    *,
    language_codes: tuple[str, ...] = ("ko",),
    custom_vocabulary: tuple[str, ...] = (),
) -> tuple[GeminiTranscribeSTTBackend, _RecordingFactory]:
    factory = _RecordingFactory(session)
    backend = GeminiTranscribeSTTBackend(
        api_key="key",
        language_codes=language_codes,
        custom_vocabulary=custom_vocabulary,
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
    backend, factory = _backend(session, language_codes=("ko",))
    stt = await backend.open_session()
    try:
        model, config = factory.calls[0]
        assert model == "gemini-3.5-transcribe-live"
        raw = _config_dict(config)
        assert raw["response_modalities"] == ["TEXT"]
        transcription = raw["input_audio_transcription"]
        assert transcription["mode"] == "VERBATIM"
        assert transcription["language_codes"] == ["ko"]
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
        await asyncio.sleep(0)
        session.push(_final("hello"))
        session.push(_final("hello again"))
        await stt.on_speech_end(trailing_silence_ms=100, reason="silence")
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
        session.push(_final(""))
        await stt.on_speech_end(trailing_silence_ms=100, reason="silence")
        events = await asyncio.wait_for(events_task, timeout=1)
        assert [event.text for event in events] == [""]
        assert all(event.is_final for event in events)
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
        session.push(_final("one"))
        await stt.on_speech_end(trailing_silence_ms=100, reason="silence")

        await stt.send_audio(b"\x00\x00" * 16)
        session.push(_final("two"))
        await stt.on_speech_end(trailing_silence_ms=100, reason="silence")

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
    backend, _ = _backend(session)
    stt = await backend.open_session()
    try:
        await stt.send_audio(b"\x00\x00" * 16)
        await stt.on_speech_end(trailing_silence_ms=100, reason="silence")
    finally:
        await stt.close()
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


def test_gemini_transcribe_language_codes() -> None:
    assert gemini_transcribe_language_codes("ko") == ["ko"]
    assert gemini_transcribe_language_codes("en-US") == ["en-US"]
    assert gemini_transcribe_language_codes("auto") == []
    assert gemini_transcribe_language_codes(None) == []
    assert gemini_transcribe_language_codes("") == []


def _final(text: str):
    from google.genai import types

    return types.LiveServerMessage(
        server_content=types.LiveServerContent(input_transcription=types.Transcription(text=text))
    )


async def _collect(stt, count: int):
    events = []
    async for event in stt.events():
        events.append(event)
        if len(events) >= count:
            break
    return events
