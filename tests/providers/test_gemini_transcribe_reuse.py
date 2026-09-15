from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from puripuly_heart.core.audio.ownership import (
    AudioSegmentIdentity,
    AudioSegmentSettingsSnapshot,
)
from puripuly_heart.core.stt.backend import (
    STTProviderEpochEnded,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTSessionProjection,
)
from puripuly_heart.providers.stt.gemini_transcribe import (
    GeminiTranscribeSTTBackend,
    _GeminiTranscribeLiveSession,
)


class _Live:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.queue: asyncio.Queue[object] = asyncio.Queue()
        self.send_gate = asyncio.Event()
        self.send_gate.set()
        self.closed = False
        self.receive_calls = 0

    async def send_realtime_input(self, **kwargs) -> None:
        self.sent.append(kwargs)
        await self.send_gate.wait()

    async def receive(self):
        self.receive_calls += 1
        while True:
            item = await self.queue.get()
            if item is _ITERATOR_END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def push(self, item: object) -> None:
        self.queue.put_nowait(item)

    async def close(self) -> None:
        self.closed = True


class _LiveContext:
    def __init__(self, live: _Live) -> None:
        self.live = live
        self.exit_calls = 0

    async def __aenter__(self):
        return self.live

    async def __aexit__(self, exc_type, exc, tb):
        self.exit_calls += 1
        await self.live.close()
        return False


class _LiveFactory:
    def __init__(self, live: _Live) -> None:
        self.live = live
        self.calls: list[tuple[str, object]] = []
        self.context: _LiveContext | None = None

    def __call__(self, *, model: str, config: object):
        self.calls.append((model, config))
        self.context = _LiveContext(self.live)
        return self.context


_ITERATOR_END = object()


def _request(order: int) -> STTProviderTurnRequest:
    settings = AudioSegmentSettingsSnapshot(
        provider_id="gemini_transcribe",
        provider_signature=("gemini_transcribe",),
        runtime_signature=("gemini_transcribe",),
        source_mode="desktop",
        source_language="en",
        expected_languages=("en",),
        target_sample_rate_hz=16000,
        vad_speech_threshold=0.4,
        vad_hangover_ms=800,
        vad_pre_roll_ms=500,
    )
    segment = AudioSegmentIdentity(
        activation_generation=1,
        segment_order=order,
        segment_id=uuid4(),
        capture_epoch=1,
    )
    return STTProviderTurnRequest(
        identity=STTProviderTurnIdentity(
            segment=segment,
            provider_epoch_id="epoch",
            provider_turn_id=f"turn-{order}",
        ),
        settings=settings,
    )


async def _session(*, timeout: float = 0.05) -> tuple[_GeminiTranscribeLiveSession, _Live]:
    live = _Live()
    session = _GeminiTranscribeLiveSession(
        api_key="key",
        language_codes=[],
        custom_vocabulary=[],
        model="gemini-3.5-transcribe-live",
        sample_rate_hz=16000,
        connect_timeout_s=1.0,
        finalize_timeout_s=timeout,
        projection=STTSessionProjection(mode="scoped", provider_epoch_id="epoch"),
    )
    session._live_session = live
    session._send_task = asyncio.create_task(session._send_loop())
    session._recv_task = asyncio.create_task(session._recv_loop())
    return session, live


def _message(*, final: str | None = None, include_final: bool = False, ack: bool = False):
    from google.genai import types

    content = None
    if include_final:
        content = types.LiveServerContent(input_transcription=types.Transcription(text=final or ""))
    voice_activity = None
    if ack:
        voice_activity = types.VoiceActivity(
            voice_activity_type=types.VoiceActivityType.ACTIVITY_END
        )
    return types.LiveServerMessage(server_content=content, voice_activity=voice_activity)


def _interim(text: str):
    from google.genai import types

    return types.LiveServerMessage(
        server_content=types.LiveServerContent(
            interim_input_transcription=types.Transcription(text=text)
        )
    )


def _go_away(*, final: str | None = None, ack: bool = False):
    from google.genai import types

    message = _message(final=final, include_final=final is not None, ack=ack)
    message.go_away = types.LiveServerGoAway(time_left="1s")
    return message


async def _wait_sent(live: _Live, key: str, count: int = 1) -> None:
    async def ready() -> None:
        while sum(key in item for item in live.sent) < count:
            await asyncio.sleep(0)

    await asyncio.wait_for(ready(), timeout=1)


async def _next(session: _GeminiTranscribeLiveSession):
    return await asyncio.wait_for(anext(session.turn_events()), timeout=1)


async def _seal(session: _GeminiTranscribeLiveSession, request: STTProviderTurnRequest) -> None:
    await session.seal_turn(
        request.identity,
        sealed_content_ranges=(),
        seal_reason="silence",
        observed_trailing_silence_ms=0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["final_first", "ack_first", "same_message"])
async def test_both_barrier_orders_and_same_message_reuse(order: str) -> None:
    session, live = await _session()
    request = _request(1)
    try:
        await session.begin_turn(request)
        await _seal(session, request)
        if order == "final_first":
            live.push(_message(final="one", include_final=True))
            live.push(_message(final="replacement", include_final=True))
            await asyncio.sleep(0)
            assert isinstance(await _next(session), STTProviderTurnUpdate)
            assert session._event_projection.scoped_event_depth == 0
            live.push(_message(ack=True))
        elif order == "ack_first":
            live.push(_message(ack=True))
            await asyncio.sleep(0)
            assert session._event_projection.scoped_event_depth == 0
            live.push(_message(final="one", include_final=True))
            assert isinstance(await _next(session), STTProviderTurnUpdate)
        else:
            live.push(_message(final="one", include_final=True, ack=True))
            assert isinstance(await _next(session), STTProviderTurnUpdate)
        terminal = await _next(session)
        assert isinstance(terminal, STTProviderTurnTerminal)
        assert (terminal.text, terminal.text_authority, terminal.epoch_disposition) == (
            "one",
            "authoritative",
            "reuse",
        )
        assert session._protocol_failed is False
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_consecutive_final_empty_and_identical_turns_keep_one_session() -> None:
    session, live = await _session()
    send_task = session._send_task
    recv_task = session._recv_task
    try:
        for order, text in enumerate(("same", "", "same"), start=1):
            request = _request(order)
            await session.begin_turn(request)
            await session.send_turn_audio(
                request.identity,
                bytes([order, 0]),
                payload_sequence=1,
                source_ranges=(),
                context_only=False,
            )
            await _seal(session, request)
            live.push(_message(final=text, include_final=True, ack=True))
            update = await _next(session)
            terminal = await _next(session)
            assert isinstance(update, STTProviderTurnUpdate)
            assert isinstance(terminal, STTProviderTurnTerminal)
            assert terminal.identity == request.identity
            assert terminal.text == text
            assert terminal.outcome == ("final" if text else "empty")
            assert terminal.epoch_disposition == "reuse"
            assert not session._pending_turns
            assert session._scoped_turn is None
        assert session._send_task is send_task
        assert session._recv_task is recv_task
        assert live.closed is False
        assert sum("activity_start" in item for item in live.sent) == 3
        assert sum("activity_end" in item for item in live.sent) == 3
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_backend_retains_prepared_sdk_and_http_resources_across_turns() -> None:
    live = _Live()
    factory = _LiveFactory(live)
    backend = GeminiTranscribeSTTBackend(api_key="offline-key", live_connect_factory=factory)
    session = await backend.open_session(
        projection=STTSessionProjection(mode="scoped", provider_epoch_id="epoch")
    )
    resources = session._client_resources
    send_task = session._send_task
    recv_task = session._recv_task
    try:
        for order in (1, 2):
            request = _request(order)
            await session.begin_turn(request)
            await _seal(session, request)
            live.push(_message(final=str(order), include_final=True, ack=True))
            assert isinstance(await _next(session), STTProviderTurnUpdate)
            terminal = await _next(session)
            assert terminal.epoch_disposition == "reuse"
        assert resources is not None
        assert factory.calls == [("gemini-3.5-transcribe-live", resources.config)]
        assert session._client_resources is resources
        assert session._send_task is send_task
        assert session._recv_task is recv_task
        assert resources.sync_transport.is_closed is False
        assert resources.async_transport.is_closed is False
        assert live.closed is False
    finally:
        await session.close()
    assert factory.context is not None
    assert factory.context.exit_calls == 1
    assert live.closed is True
    assert resources is not None
    assert resources.sync_transport.is_closed is True
    assert resources.async_transport.is_closed is True


@pytest.mark.asyncio
async def test_queued_successor_cannot_write_until_barrier_and_prior_send_finishes() -> None:
    session, live = await _session()
    request_a = _request(1)
    request_b = _request(2)
    try:
        await session.begin_turn(request_a)
        live.send_gate.clear()
        seal_a = asyncio.create_task(_seal(session, request_a))
        await _wait_sent(live, "activity_end")
        live.push(_message(final="a", include_final=True, ack=True))
        await _next(session)
        terminal_a = await _next(session)
        assert terminal_a.epoch_disposition == "reuse"
        begin_b = asyncio.create_task(session.begin_turn(request_b))
        await asyncio.sleep(0)
        assert sum("activity_start" in item for item in live.sent) == 1
        assert not begin_b.done()
        live.send_gate.set()
        await seal_a
        await begin_b
        assert sum("activity_start" in item for item in live.sent) == 2
        assert session._capture_turn is not None
        assert session._capture_turn.identity == request_b.identity
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("received", "expected"),
    [
        ("authoritative", ("final", "kept", "authoritative")),
        ("interim", ("degraded", "partial", "degraded")),
        ("none", ("failed", "", "none")),
    ],
)
async def test_missing_half_timeout_always_retires(
    received: str, expected: tuple[str, str, str]
) -> None:
    session, live = await _session(timeout=0.01)
    request = _request(1)
    try:
        await session.begin_turn(request)
        await _seal(session, request)
        if received == "authoritative":
            live.push(_message(final="kept", include_final=True))
            assert isinstance(await _next(session), STTProviderTurnUpdate)
        elif received == "interim":
            live.push(_interim("partial"))
            live.push(_message(ack=True))
            assert isinstance(await _next(session), STTProviderTurnUpdate)
        terminal = await _next(session)
        assert isinstance(terminal, STTProviderTurnTerminal)
        assert (terminal.outcome, terminal.text, terminal.text_authority) == expected
        assert terminal.failure_reason == "gemini_finalize_timeout"
        assert terminal.epoch_disposition == "retire"
        with pytest.raises(RuntimeError, match="closed"):
            await session.begin_turn(_request(2))
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_abort_during_queued_finalize_does_not_leave_turn_state() -> None:
    session, live = await _session()
    request = _request(1)
    try:
        await session.begin_turn(request)
        live.send_gate.clear()
        seal = asyncio.create_task(_seal(session, request))
        await _wait_sent(live, "activity_end")
        await session.abort_turn(request.identity, reason="cancelled")
        terminal = await _next(session)
        assert terminal.outcome == "cancelled"
        assert isinstance(await _next(session), STTProviderEpochEnded)
        live.send_gate.set()
        with pytest.raises(RuntimeError, match="retired during finalize"):
            await seal
        assert not session._pending_turns
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_abort_and_go_away_cannot_be_reused() -> None:
    aborted, _ = await _session()
    request = _request(1)
    try:
        await aborted.begin_turn(request)
        await aborted.abort_turn(request.identity, reason="cancelled")
        terminal = await _next(aborted)
        assert terminal.epoch_disposition == "retire"
        assert isinstance(await _next(aborted), STTProviderEpochEnded)
        with pytest.raises(RuntimeError, match="closed"):
            await aborted.begin_turn(_request(2))
    finally:
        await aborted.close()

    retiring, live = await _session()
    request = _request(1)
    try:
        await retiring.begin_turn(request)
        await _seal(retiring, request)
        live.push(_go_away(final="kept", ack=True))
        assert isinstance(await _next(retiring), STTProviderTurnUpdate)
        terminal = await _next(retiring)
        assert (terminal.text, terminal.failure_reason, terminal.epoch_disposition) == (
            "kept",
            "gemini_go_away",
            "retire",
        )
        with pytest.raises(RuntimeError, match="closed"):
            await retiring.begin_turn(_request(2))
    finally:
        await retiring.close()


@pytest.mark.asyncio
async def test_repeated_receive_iterators_and_idle_contradiction() -> None:
    session, live = await _session()
    request = _request(1)
    try:
        live.push(_ITERATOR_END)
        await session.begin_turn(request)
        await _seal(session, request)
        live.push(_message(final="one", include_final=True, ack=True))
        assert isinstance(await _next(session), STTProviderTurnUpdate)
        terminal = await _next(session)
        assert terminal.epoch_disposition == "reuse"
        assert live.receive_calls >= 2
        while session._streaming_turn is not None:
            await asyncio.sleep(0)
        live.push(_message(final="unsolicited", include_final=True))
        ended = await _next(session)
        assert isinstance(ended, STTProviderEpochEnded)
        assert ended.reason == "gemini_unsolicited_authoritative"
        with pytest.raises(RuntimeError, match="closed"):
            await session.begin_turn(_request(2))
    finally:
        await session.close()
