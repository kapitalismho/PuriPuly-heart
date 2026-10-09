from __future__ import annotations

import asyncio

import httpx
import pytest

from puripuly_heart.core import network_clients
from puripuly_heart.providers.stt.gemini_transcribe import (
    GeminiTranscribeSTTBackend,
    gemini_transcribe_language_codes,
)


class _StubSyncTransport:
    def __init__(self) -> None:
        self.closed = False
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


class _StubAsyncTransport:
    def __init__(self) -> None:
        self.closed = False
        self.aclose_calls = 0

    async def aclose(self) -> None:
        self.aclose_calls += 1
        self.closed = True


class _StubClientAio:
    def __init__(self) -> None:
        self.aclose_calls = 0

    async def aclose(self) -> None:
        self.aclose_calls += 1


class _StubClient:
    def __init__(self) -> None:
        self.aio = _StubClientAio()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


@pytest.fixture(autouse=True)
def _stub_gemini_setup(monkeypatch: pytest.MonkeyPatch):
    from puripuly_heart.providers.stt import gemini_transcribe as gemini_module

    created: list = []

    def fake_prepare(api_key, language_codes, custom_vocabulary):
        resources = gemini_module._GeminiClientResources(
            client=_StubClient(),
            sync_transport=_StubSyncTransport(),
            async_transport=_StubAsyncTransport(),
            config=gemini_module._build_live_config_sync(language_codes, custom_vocabulary),
        )
        created.append(resources)
        return resources

    monkeypatch.setattr(gemini_module, "_prepare_gemini_resources_sync", fake_prepare)
    return created


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
async def test_session_configures_verbatim_transcription() -> None:
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
        _, config = factory.calls[0]
        assert _config_dict(config)["input_audio_transcription"]["custom_vocabulary"] == [
            "PuriPuly",
            "Gemini",
        ]
    finally:
        await stt.close()


@pytest.mark.asyncio
async def test_legacy_audio_end_and_next_audio_do_not_wait_for_native_final() -> None:
    session = _FakeLiveSession()
    backend, _ = _backend(session)
    stt = await backend.open_session()
    try:
        first = b"\x01\x00" * 16
        second = b"\x02\x00" * 16
        await stt.send_audio(first)
        await stt.on_speech_end(reason="silence")
        await stt.send_audio(second)
        assert session.sent == [
            {"audio": {"data": first, "mime_type": "audio/pcm;rate=16000"}},
            {"audio_stream_end": True},
            {"audio": {"data": second, "mime_type": "audio/pcm;rate=16000"}},
        ]
        session.push(_final("one"))
        session.push(_final("two"))
        assert [event.text for event in await asyncio.wait_for(_collect(stt, 2), 1)] == [
            "one",
            "two",
        ]
    finally:
        await stt.close()


@pytest.mark.asyncio
async def test_legacy_native_final_before_end_and_multiple_after_end_are_distinct() -> None:
    session = _FakeLiveSession()
    backend, _ = _backend(session)
    stt = await backend.open_session()
    try:
        await stt.send_audio(b"\x01\x00" * 16)
        session.push(_final("early"))
        first = await asyncio.wait_for(anext(stt.events()), 1)
        await stt.on_speech_end(reason="silence")
        session.push(_final("same"))
        session.push(_final("same"))
        later = await asyncio.wait_for(_collect(stt, 2), 1)
        assert [(event.text, event.is_final) for event in [first, *later]] == [
            ("early", True),
            ("same", True),
            ("same", True),
        ]
    finally:
        await stt.close()


@pytest.mark.asyncio
async def test_native_activity_end_and_interim_do_not_promote_or_block_audio() -> None:
    session = _FakeLiveSession()
    backend, _ = _backend(session)
    stt = await backend.open_session()
    try:
        await stt.send_audio(b"\x01\x00")
        session.push(_interim("partial"))
        session.push(_activity_end_ack())
        await stt.on_speech_end(reason="silence")
        await stt.send_audio(b"\x02\x00")
        assert [call["audio"]["data"] for call in session.sent if "audio" in call] == [
            b"\x01\x00",
            b"\x02\x00",
        ]
        session.push(_final("actual"))
        assert (await asyncio.wait_for(anext(stt.events()), 1)).text == "actual"
        assert stt._event_projection._legacy_events.empty()
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
        assert [event async for event in stt.events()] == []
    finally:
        await stt.close()


@pytest.mark.asyncio
async def test_close_after_unanswered_audio_end_closes_session() -> None:
    session = _FakeLiveSession()
    backend, factory = _backend(session)
    stt = await backend.open_session()
    await stt.send_audio(b"\x00\x00" * 16)
    await stt.on_speech_end(reason="silence")
    await stt.close()
    assert factory.context is not None
    assert factory.context.exited is True
    assert session.closed is True


@pytest.mark.asyncio
async def test_verify_api_key_uses_models_metadata_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_urls: list[str] = []

    def respond(request):
        seen_urls.append(str(request.url))
        assert request.headers["x-goog-api-key"] == "secret"
        return httpx.Response(200)
    monkeypatch.setattr(
        network_clients, "external_client",
        lambda **kwargs: httpx.Client(transport=httpx.MockTransport(respond)),
    )

    assert await GeminiTranscribeSTTBackend.verify_api_key("") is False
    assert await GeminiTranscribeSTTBackend.verify_api_key("secret") is True
    assert seen_urls == ["https://generativelanguage.googleapis.com/v1beta/models"]


@pytest.mark.asyncio
async def test_verify_api_key_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        network_clients, "external_client",
        lambda **kwargs: httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(401))),
    )

    with pytest.raises(httpx.HTTPStatusError) as caught:
        await GeminiTranscribeSTTBackend.verify_api_key("secret")
    assert caught.value.response.status_code == 401


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
async def test_receive_loop_continues_after_iterator_end() -> None:
    session = _TurnScopedLiveSession()
    backend, _ = _backend(session)
    stt = await backend.open_session()
    events_task = asyncio.create_task(_collect(stt, 2))
    try:
        await stt.send_audio(b"\x00\x00" * 16)
        session.push(_final("one"))
        session.push(_TURN_COMPLETE)
        await stt.on_speech_end(reason="silence")
        await stt.send_audio(b"\x01\x00" * 16)
        session.push(_final("two"))
        session.push(_TURN_COMPLETE)
        events = await asyncio.wait_for(events_task, timeout=1)
        assert [event.text for event in events] == ["one", "two"]
        assert session.receive_calls >= 2
    finally:
        await stt.close()


@pytest.mark.asyncio
async def test_recv_failure_closes_session() -> None:
    session = _FailingReceiveSession(_ApiFailure(code=400, status="INVALID_ARGUMENT"))
    backend, factory = _backend(session)
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


async def _collect(stt, count: int):
    events = []
    async for event in stt.events():
        events.append(event)
        if len(events) >= count:
            break
    return events
