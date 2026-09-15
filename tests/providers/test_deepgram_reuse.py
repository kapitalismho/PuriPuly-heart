from __future__ import annotations

import asyncio
import types
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
from puripuly_heart.providers.stt.deepgram import (
    _CLOSE_STREAM,
    _FINALIZE,
    _DeepgramSDKSession,
)


def _request(order: int) -> STTProviderTurnRequest:
    identity = STTProviderTurnIdentity(
        segment=AudioSegmentIdentity(
            activation_generation=1,
            segment_order=order,
            segment_id=uuid4(),
            capture_epoch=1,
        ),
        provider_epoch_id="epoch",
        provider_turn_id=f"turn-{order}",
    )
    settings = AudioSegmentSettingsSnapshot(
        provider_id="deepgram",
        provider_signature=("deepgram", "nova-3"),
        runtime_signature=("en",),
        source_mode="microphone",
        source_language="en",
        expected_languages=("en",),
        target_sample_rate_hz=16000,
        vad_speech_threshold=0.4,
        vad_hangover_ms=800,
        vad_pre_roll_ms=500,
    )
    return STTProviderTurnRequest(identity=identity, settings=settings, channel="self")


def _session(*, drain_timeout_s: float = 1.0) -> _DeepgramSDKSession:
    session = _DeepgramSDKSession(
        api_key="key",
        model="nova-3",
        language="en",
        sample_rate_hz=16000,
        connect_timeout_s=1.0,
        keyterms=[],
        drain_timeout_s=drain_timeout_s,
        projection=STTSessionProjection(mode="scoped", provider_epoch_id="epoch"),
    )
    session._loop = asyncio.get_running_loop()
    return session


def _result(
    text: str,
    *,
    is_final: bool = True,
    speech_final: bool = False,
    top_level_ack: bool = False,
    metadata_ack: bool = False,
):
    return types.SimpleNamespace(
        channel=types.SimpleNamespace(alternatives=[types.SimpleNamespace(transcript=text)]),
        is_final=is_final,
        speech_final=speech_final,
        from_finalize=top_level_ack,
        metadata=types.SimpleNamespace(
            request_id="connection-request",
            from_finalize=metadata_ack,
        ),
    )


def _installed_empty_alternatives_result(*, metadata_ack: bool):
    from deepgram.extensions.types.sockets import ListenV1ResultsEvent
    from deepgram.extensions.types.sockets.listen_v1_results_event import (
        ListenV1Channel,
        ListenV1ModelInfo,
        ListenV1ResultsMetadata,
    )

    metadata = ListenV1ResultsMetadata(
        request_id="connection-request",
        model_info=ListenV1ModelInfo(name="nova-3", version="1", arch="nova"),
        model_uuid="model",
        from_finalize=metadata_ack,
    )
    return ListenV1ResultsEvent(
        type="Results",
        channel_index=[0, 1],
        duration=0.1,
        start=0.0,
        is_final=True,
        speech_final=False,
        channel=ListenV1Channel(alternatives=[]),
        metadata=metadata,
        from_finalize=not metadata_ack,
    )


async def _next(session: _DeepgramSDKSession):
    return await asyncio.wait_for(anext(session.turn_events()), timeout=1)


async def _seal(session: _DeepgramSDKSession, request: STTProviderTurnRequest) -> None:
    await session.seal_turn(
        request.identity,
        sealed_content_ranges=(),
        seal_reason="silence",
        observed_trailing_silence_ms=200,
    )


@pytest.mark.asyncio
async def test_acknowledged_turns_reuse_one_session_for_identical_and_empty_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[object] = []

    async def write(_session: _DeepgramSDKSession, payload: object) -> None:
        writes.append(payload)

    monkeypatch.setattr(_DeepgramSDKSession, "_write_thread_payload", write)
    session = _session()

    for order, metadata_ack in ((1, False), (2, True)):
        request = _request(order)
        await session.begin_turn(request)
        await session.send_turn_audio(
            request.identity,
            b"identical",
            payload_sequence=1,
            source_ranges=(),
            context_only=False,
        )
        session._build_transcript_event(_result("identical", speech_final=True))
        await _seal(session, request)
        session._build_transcript_event(
            _result(
                "",
                is_final=False,
                top_level_ack=not metadata_ack,
                metadata_ack=metadata_ack,
            )
        )
        await asyncio.sleep(0)

        update = await _next(session)
        terminal = await _next(session)
        assert isinstance(update, STTProviderTurnUpdate)
        assert update.identity == request.identity
        assert update.text == "identical"
        assert isinstance(terminal, STTProviderTurnTerminal)
        assert terminal.identity == request.identity
        assert (terminal.outcome, terminal.text, terminal.epoch_disposition) == (
            "final",
            "identical",
            "reuse",
        )

    empty_request = _request(3)
    await session.begin_turn(empty_request)
    await _seal(session, empty_request)
    session._build_transcript_event(_result("", top_level_ack=True))
    await asyncio.sleep(0)
    empty_terminal = await _next(session)

    assert isinstance(empty_terminal, STTProviderTurnTerminal)
    assert (empty_terminal.outcome, empty_terminal.text, empty_terminal.epoch_disposition) == (
        "empty",
        "",
        "reuse",
    )
    assert writes == [b"identical", _FINALIZE, b"identical", _FINALIZE, _FINALIZE]
    assert session._scoped_drain_task is None
    assert session._scoped_fragments == []
    assert session._scoped_provenance == []


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_ack", [False, True])
async def test_installed_empty_alternatives_finalize_ack_is_authoritative_empty_reuse(
    monkeypatch: pytest.MonkeyPatch,
    metadata_ack: bool,
) -> None:
    writes: list[object] = []

    async def write(_session: _DeepgramSDKSession, payload: object) -> None:
        writes.append(payload)

    monkeypatch.setattr(_DeepgramSDKSession, "_write_thread_payload", write)
    session = _session(drain_timeout_s=0.001)
    request = _request(1)
    await session.begin_turn(request)
    await _seal(session, request)
    session._build_transcript_event(_installed_empty_alternatives_result(metadata_ack=metadata_ack))
    await asyncio.sleep(0)

    terminal = await _next(session)
    assert isinstance(terminal, STTProviderTurnTerminal)
    assert (terminal.outcome, terminal.text_authority, terminal.epoch_disposition) == (
        "empty",
        "authoritative",
        "reuse",
    )
    await asyncio.sleep(0.01)
    assert writes == [_FINALIZE]
    assert session._scoped_drain_task is None


@pytest.mark.asyncio
async def test_empty_alternatives_without_finalize_ack_remains_incomplete() -> None:
    session = _session()
    request = _request(1)
    await session.begin_turn(request)
    result = types.SimpleNamespace(
        channel=types.SimpleNamespace(alternatives=[]),
        is_final=True,
        speech_final=False,
        from_finalize=False,
        metadata=types.SimpleNamespace(
            request_id="connection-request",
            from_finalize=False,
        ),
    )

    assert session._build_transcript_event(result) is None
    await asyncio.sleep(0)
    assert session._event_projection.active_identity == request.identity
    assert session._event_projection.scoped_event_depth == 0


@pytest.mark.asyncio
async def test_final_fragment_waits_for_ack_and_close_fallback_is_irreversible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[object] = []
    close_started = asyncio.Event()
    release_close = asyncio.Event()

    async def write(_session: _DeepgramSDKSession, payload: object) -> None:
        writes.append(payload)
        if payload is _CLOSE_STREAM:
            close_started.set()
            await release_close.wait()

    monkeypatch.setattr(_DeepgramSDKSession, "_write_thread_payload", write)
    session = _session(drain_timeout_s=0.001)
    request = _request(1)
    await session.begin_turn(request)
    await _seal(session, request)
    session._build_transcript_event(_result("kept", speech_final=True))
    await asyncio.sleep(0)

    update = await _next(session)
    assert isinstance(update, STTProviderTurnUpdate)
    assert session._event_projection.active_identity == request.identity
    await asyncio.wait_for(close_started.wait(), timeout=1)

    session._build_transcript_event(_result("", is_final=False, top_level_ack=True))
    await asyncio.sleep(0)
    terminal = await _next(session)
    release_close.set()
    await asyncio.sleep(0)

    assert isinstance(terminal, STTProviderTurnTerminal)
    assert (terminal.outcome, terminal.text, terminal.epoch_disposition) == (
        "final",
        "kept",
        "retire",
    )
    assert writes.count(_CLOSE_STREAM) == 1
    with pytest.raises(RuntimeError, match="closed"):
        await session.begin_turn(_request(2))


@pytest.mark.asyncio
async def test_close_cancels_blocked_close_stream_drain_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[object] = []
    close_stream_started = asyncio.Event()
    blocked = asyncio.Event()

    async def write(_session: _DeepgramSDKSession, payload: object) -> None:
        writes.append(payload)
        if payload is _CLOSE_STREAM:
            close_stream_started.set()
            await blocked.wait()

    monkeypatch.setattr(_DeepgramSDKSession, "_write_thread_payload", write)
    session = _session(drain_timeout_s=0.001)
    request = _request(1)
    await session.begin_turn(request)
    await _seal(session, request)
    await asyncio.wait_for(close_stream_started.wait(), timeout=1)
    drain_task = session._scoped_drain_task

    assert drain_task is not None
    assert not drain_task.done()
    await asyncio.wait_for(session.close(), timeout=1)

    assert session._scoped_drain_task is None
    assert drain_task.done()
    assert writes.count(_CLOSE_STREAM) == 1
    session._build_transcript_event(_result("late", top_level_ack=True))
    await asyncio.sleep(0)
    assert session._stopped is True
    with pytest.raises(RuntimeError, match="closed"):
        await session.begin_turn(_request(2))


@pytest.mark.asyncio
async def test_idle_finalize_ack_retires_epoch_instead_of_attaching_to_next_turn() -> None:
    session = _session()
    session._build_transcript_event(_result("late", top_level_ack=True))
    await asyncio.sleep(0)

    ended = await _next(session)
    assert isinstance(ended, STTProviderEpochEnded)
    assert ended.reason == "deepgram_idle_result"
    with pytest.raises(RuntimeError, match="retired"):
        await session.begin_turn(_request(1))


@pytest.mark.asyncio
async def test_callback_captured_for_old_turn_is_rejected_after_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def write(_session: _DeepgramSDKSession, _payload: object) -> None:
        return None

    monkeypatch.setattr(_DeepgramSDKSession, "_write_thread_payload", write)
    session = _session()
    first = _request(1)
    await session.begin_turn(first)
    await _seal(session, first)
    session._build_transcript_event(_result("first", top_level_ack=True))
    await asyncio.sleep(0)
    assert isinstance(await _next(session), STTProviderTurnUpdate)
    assert isinstance(await _next(session), STTProviderTurnTerminal)

    second = _request(2)
    await session.begin_turn(second)
    session._handle_scoped_result(
        first.identity,
        "stale",
        True,
        True,
        types.SimpleNamespace(),
    )
    await _seal(session, second)
    session._build_transcript_event(_result("second", top_level_ack=True))
    await asyncio.sleep(0)

    update = await _next(session)
    terminal = await _next(session)
    assert isinstance(update, STTProviderTurnUpdate)
    assert update.text == "second"
    assert isinstance(terminal, STTProviderTurnTerminal)
    assert terminal.text == "second"
    assert terminal.epoch_disposition == "reuse"
