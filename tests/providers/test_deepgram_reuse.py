from __future__ import annotations

import asyncio
import json
import types
from contextlib import asynccontextmanager
from dataclasses import replace
from uuid import uuid4

import pytest

from puripuly_heart.core.audio.ownership import (
    AudioSegmentIdentity,
    AudioSegmentSettingsSnapshot,
)
from puripuly_heart.core.diagnostics import normalize_diagnostic
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

    monkeypatch.setattr(_DeepgramSDKSession, "_write_payload", write)
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

    monkeypatch.setattr(_DeepgramSDKSession, "_write_payload", write)
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

    monkeypatch.setattr(_DeepgramSDKSession, "_write_payload", write)
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

    monkeypatch.setattr(_DeepgramSDKSession, "_write_payload", write)
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

    monkeypatch.setattr(_DeepgramSDKSession, "_write_payload", write)
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


class _InstalledSdkWire:
    def __init__(self) -> None:
        self.messages = asyncio.Queue()
        self.writes = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.messages.get()

    async def send(self, payload):
        self.writes.append(payload)


def _installed_callback_payload(channel, *, finalize=False):
    return json.dumps(
        {
            "type": "Results",
            "channel_index": [0, 1],
            "duration": 0.1,
            "start": 0.0,
            "is_final": True,
            "speech_final": False,
            "channel": channel,
            "from_finalize": finalize,
            "metadata": {
                "request_id": uuid4().hex,
                "model_info": {"name": "nova-3", "version": "1", "arch": "nova"},
                "model_uuid": uuid4().hex,
            },
            "private_detail": "PRIVATE_DEEPGRAM_PAYLOAD",
        }
    )


async def _installed_callback_session(monkeypatch, *, channel="self"):
    from deepgram.listen.v1.socket_client import AsyncV1SocketClient

    from puripuly_heart.providers.stt import sdk_network

    wire = _InstalledSdkWire()
    epoch = uuid4().hex
    session = _DeepgramSDKSession(
        api_key="PRIVATE_NO_REQUEST_KEY",
        model="nova-3",
        language="en",
        sample_rate_hz=16000,
        connect_timeout_s=1.0,
        keyterms=[],
        stream_label=channel,
        drain_timeout_s=30,
        projection=STTSessionProjection(mode="scoped", provider_epoch_id=epoch),
    )
    observations = {"callbacks": [], "causes": [], "processed": asyncio.Event()}
    handle_message = _DeepgramSDKSession._handle_message
    report_error = _DeepgramSDKSession._report_error

    def observe_message(self, result):
        observations["callbacks"].append(type(result).__name__)
        handle_message(self, result)
        observations["processed"].set()

    def observe_error(self, exception):
        observations["causes"].append(exception)
        report_error(self, exception)
        observations["processed"].set()

    @asynccontextmanager
    async def connect(_client, **_kwargs):
        yield AsyncV1SocketClient(websocket=wire)

    monkeypatch.setattr(sdk_network, "deepgram_listen_connect", connect)
    monkeypatch.setattr(_DeepgramSDKSession, "_handle_message", observe_message)
    monkeypatch.setattr(_DeepgramSDKSession, "_report_error", observe_error)
    await session.start()
    initial = _request(1)
    request = replace(
        initial,
        channel=channel,
        identity=replace(
            initial.identity,
            provider_epoch_id=epoch,
            provider_turn_id=uuid4().hex,
        ),
    )
    return session, wire, observations, request


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize(
    "malformed",
    ["missing_channel", "missing_alternatives", "invalid_transcript", "invalid_json"],
)
async def test_installed_sdk_malformed_callback_fails_scoped_turn_and_epoch(
    monkeypatch, malformed, partial
):
    session, wire, observations, request = await _installed_callback_session(
        monkeypatch,
        channel="peer",
    )
    try:
        await session.begin_turn(request)
        if partial:
            wire.messages.put_nowait(
                _installed_callback_payload(
                    {
                        "alternatives": [
                            {
                                "transcript": "PRIVATE_TRANSCRIPT",
                                "confidence": 1.0,
                                "words": [],
                            }
                        ],
                    }
                )
            )
            update = await _next(session)
            assert isinstance(update, STTProviderTurnUpdate)
            assert update.identity == request.identity
        await _seal(session, request)
        observations["processed"].clear()
        payload = {
            "missing_channel": _installed_callback_payload(None),
            "missing_alternatives": _installed_callback_payload({}),
            "invalid_transcript": _installed_callback_payload(
                {
                    "alternatives": [{"transcript": 42, "confidence": 1.0, "words": []}],
                }
            ),
            "invalid_json": '{"type":"Results","private_detail":"PRIVATE_BROKEN_JSON"',
        }[malformed]
        wire.messages.put_nowait(payload)
        await asyncio.wait_for(observations["processed"].wait(), timeout=1)
        terminal = await _next(session)
        ended = await _next(session)
        assert isinstance(terminal, STTProviderTurnTerminal)
        assert terminal.identity == request.identity
        assert terminal.outcome == ("degraded" if partial else "failed")
        assert terminal.text_authority == ("degraded" if partial else "none")
        assert terminal.epoch_disposition == "retire"
        assert terminal.failure_reason == "deepgram_protocol_failed"
        assert isinstance(ended, STTProviderEpochEnded)
        assert not ended.orderly
        assert ended.reason == "deepgram_protocol_failed"
        assert ended.provider_epoch_id == request.identity.provider_epoch_id
        assert ended.provider_turn_id == request.identity.provider_turn_id
        diagnostics = terminal.failure_diagnostics
        assert diagnostics is not None
        assert ended.failure_diagnostics is diagnostics
        assert diagnostics.category == "invalid_response"
        assert diagnostics.code == "stt.deepgram_protocol_failed"
        assert diagnostics.channel == request.channel
        assert diagnostics.turn_id == str(request.identity.segment.segment_id)
        assert diagnostics.generation == request.identity.segment.activation_generation
        assert diagnostics.fields["provider_epoch_id"] == request.identity.provider_epoch_id
        assert diagnostics.fields["provider_turn_id"] == request.identity.provider_turn_id
        assert diagnostics.fields["recognition_cause"] == "deepgram_protocol_failed"
        assert observations["causes"]
        if malformed == "invalid_json":
            assert isinstance(observations["causes"][0], json.JSONDecodeError)
        else:
            assert isinstance(observations["causes"][0], ValueError | TypeError)
            assert observations["callbacks"][-1] == "ListenV1ResultsEvent"
        safe = normalize_diagnostic(
            diagnostics,
            instance_id=uuid4().hex,
            severity="error",
            disposition="failed",
            stage="protocol",
        )
        assert safe["diagnostic_id"] == diagnostics.diagnostic_id
        assert safe["channel"] == request.channel
        assert safe["turn_id"] == str(request.identity.segment.segment_id)
        assert "PRIVATE_" not in json.dumps(safe)
        assert session._scoped_drain_task is None
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["empty_no_ack", "whitespace", "metadata", "empty_ack"])
async def test_installed_sdk_valid_callbacks_preserve_empty_and_cancellation_semantics(
    monkeypatch, control
):
    session, wire, observations, request = await _installed_callback_session(monkeypatch)
    try:
        await session.begin_turn(request)
        if control == "empty_ack":
            await _seal(session, request)
        payload = {
            "empty_no_ack": _installed_callback_payload({"alternatives": []}),
            "whitespace": _installed_callback_payload(
                {
                    "alternatives": [{"transcript": "  ", "confidence": 1.0, "words": []}],
                }
            ),
            "metadata": json.dumps(
                {
                    "type": "Metadata",
                    "request_id": uuid4().hex,
                    "sha256": "0" * 64,
                    "created": "2026-01-01T00:00:00Z",
                    "duration": 0.0,
                    "channels": 1.0,
                }
            ),
            "empty_ack": _installed_callback_payload({"alternatives": []}, finalize=True),
        }[control]
        wire.messages.put_nowait(payload)
        await asyncio.wait_for(observations["processed"].wait(), timeout=1)
        assert observations["callbacks"]
        assert not observations["causes"]
        if control == "empty_ack":
            terminal = await _next(session)
            assert isinstance(terminal, STTProviderTurnTerminal)
            assert terminal.outcome == "empty"
            assert terminal.text_authority == "authoritative"
            assert terminal.epoch_disposition == "reuse"
            assert terminal.failure_diagnostics is None
            request = replace(
                _request(2),
                identity=replace(
                    request.identity,
                    segment=replace(request.identity.segment, segment_order=2, segment_id=uuid4()),
                    provider_turn_id=uuid4().hex,
                ),
            )
            await session.begin_turn(request)
        else:
            assert session._event_projection.active_identity == request.identity
            if control == "whitespace":
                update = await _next(session)
                assert isinstance(update, STTProviderTurnUpdate)
                assert not update.text.strip()
            else:
                assert session._event_projection.scoped_event_depth == 0
        await session.abort_turn(request.identity, reason="cancelled")
        cancelled = await _next(session)
        ended = await _next(session)
        assert cancelled.outcome == "cancelled"
        assert cancelled.failure_reason == "cancelled"
        assert cancelled.failure_diagnostics is None
        assert isinstance(ended, STTProviderEpochEnded)
        assert ended.reason == "cancelled"
        assert ended.failure_diagnostics is None
    finally:
        await session.close()


def _timed_result(text: str, start: float, duration: float = 0.5, *, speaker: int = 1):
    result = _result(text)
    result.start = start
    result.duration = duration
    result.channel_index = [0, 1]
    result.metadata.model_info = types.SimpleNamespace(
        name="general-nova-3", version="2025-04-17.21547", arch="nova"
    )
    result.metadata.model_uuid = "qualified-model"
    result.channel.alternatives[0].words = [
        types.SimpleNamespace(
            word=text.strip(),
            punctuated_word=text.strip(),
            start=start,
            end=None,
            speaker=speaker,
        )
    ]
    return result


@pytest.mark.asyncio
async def test_final_segments_release_exact_prefix_and_keep_distinct_equal_terminal_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from puripuly_heart.core.stt.backend import STTContributionConsumptionLedger
    from puripuly_heart.core.stt.scoped_normalizer import STTScopedTurnNormalizer

    async def write(_session, _payload):
        return None

    monkeypatch.setattr(_DeepgramSDKSession, "_write_payload", write)
    session = _session()
    request = _request(1)
    await session.begin_turn(request)
    await session.send_turn_audio(
        request.identity, b"\0" * 64000, payload_sequence=1, source_ranges=(), context_only=False
    )
    session._build_transcript_event(_result("provisional", is_final=False, speech_final=True))
    assert session._event_projection.scoped_event_depth == 0
    first_result = _timed_result("  No. ", 0.1, speaker=3)
    session._build_transcript_event(first_result)
    session._build_transcript_event(first_result)
    first = await _next(session)
    assert first.text == "  No. "
    assert first.source_release.text_end == len(first.text)
    assert first.provenance.transcription.words == (("No.", "0.1", None),)
    assert first.provenance.transcription.speaker_label is None
    assert dict(first.provenance.native_model_metadata) == {
        "name": "general-nova-3",
        "version": "2025-04-17.21547",
        "arch": "nova",
        "model_uuid": "qualified-model",
    }
    assert first.provenance.audio_clocks[0].origin_sent_sample == 0
    assert first.provenance.sent_until_sample == 32000
    await _seal(session, request)
    tail_result = _timed_result("  No. ", 1.1, speaker=99)
    tail_result.from_finalize = True
    session._build_transcript_event(tail_result)
    tail = await _next(session)
    terminal = await _next(session)
    assert first.contribution.contribution_id != tail.contribution.contribution_id
    assert tail.provenance.control_sent_sample == 32000
    assert tail.provenance.control_kind == "Finalize"
    assert terminal.text == "  No.   No. "
    normalizer = STTScopedTurnNormalizer(request.identity)
    consumption = STTContributionConsumptionLedger()
    normalized_first = normalizer.apply_update(first)
    assert consumption.consume(normalized_first) == "  No. "
    normalizer.apply_update(tail)
    assert consumption.consume(normalizer.apply_terminal(terminal)) == "  No. "
    assert session._scoped_drain_task is None


@pytest.mark.asyncio
async def test_native_clock_discontinuity_preserves_text_without_guessed_correction() -> None:
    session = _session()
    request = _request(1)
    await session.begin_turn(request)
    session._build_transcript_event(_timed_result("first", 2.0))
    session._build_transcript_event(_timed_result("尾", 0.2))
    first, reset = await _next(session), await _next(session)
    assert first.provenance.audio_clocks[0].qualified
    assert reset.text == "尾"
    assert not reset.provenance.audio_clocks[0].qualified
    assert reset.provenance.text_timing[0].raw_start == 0.2
    assert reset.provenance.text_timing[0].text_start == len("first")
    session._scoped_transport_end(False, "test_finished")


@pytest.mark.asyncio
async def test_missing_empty_finalize_ack_never_authorizes_success_or_successor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes = []

    async def write(_session, payload):
        writes.append(payload)

    monkeypatch.setattr(_DeepgramSDKSession, "_write_payload", write)
    session = _session(drain_timeout_s=0.001)
    request = _request(1)
    await session.begin_turn(request)
    await _seal(session, request)
    terminal = await _next(session)
    assert terminal.outcome == "failed"
    assert terminal.text_authority == "none"
    assert terminal.failure_reason == "deepgram_finalize_ack_missing"
    assert writes == [_FINALIZE, _CLOSE_STREAM]
    with pytest.raises(RuntimeError, match="closed"):
        await session.begin_turn(_request(2))
    session._build_transcript_event(_timed_result("late", 0.1))
    ended = await _next(session)
    assert isinstance(ended, STTProviderEpochEnded)


@pytest.mark.asyncio
async def test_failed_pcm_write_does_not_advance_actual_submitted_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def write(_session, _payload):
        raise OSError("failed actual write")

    monkeypatch.setattr(_DeepgramSDKSession, "_write_payload", write)
    session = _session()
    request = _request(1)
    await session.begin_turn(request)
    with pytest.raises(OSError, match="actual write"):
        await session.send_turn_audio(
            request.identity, b"\0" * 32, payload_sequence=1, source_ranges=(), context_only=False
        )
    assert session._sent_samples == 0
    session._scoped_transport_end(False, "test_finished")


@pytest.mark.asyncio
async def test_stop_or_writer_end_is_not_an_empty_finalize_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def write(_session, _payload):
        return None

    monkeypatch.setattr(_DeepgramSDKSession, "_write_payload", write)
    session = _session()
    request = _request(1)
    await session.begin_turn(request)
    await _seal(session, request)
    await session.stop()
    session._scoped_transport_end(True, "deepgram_writer_ended")
    terminal = await _next(session)
    assert terminal.outcome == "failed"
    assert terminal.text_authority == "none"
    assert terminal.epoch_disposition == "retire"


@pytest.mark.asyncio
async def test_lawful_close_stream_drains_late_final_source_without_results_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes = []

    async def write(_session, payload):
        writes.append(payload)

    monkeypatch.setattr(_DeepgramSDKSession, "_write_payload", write)
    session = _session(drain_timeout_s=0.05)
    request = _request(1)
    await session.begin_turn(request)
    await _seal(session, request)
    while not session._scoped_close_written:
        await asyncio.sleep(0)
    session._build_transcript_event(_timed_result(" tail ", 0.1))
    session._scoped_transport_end(True, "deepgram_connection_closed")
    update, terminal = await _next(session), await _next(session)
    assert update.text == terminal.text == " tail "
    assert (terminal.outcome, terminal.text_authority, terminal.epoch_disposition) == (
        "final",
        "authoritative",
        "retire",
    )
    assert writes == [_FINALIZE, _CLOSE_STREAM]
    assert any(item.control_kind == "CloseStream" for item in terminal.provenance)
    assert session._scoped_drain_task is None


@pytest.mark.asyncio
async def test_native_word_languages_are_retained_without_native_speaker_authority() -> None:
    session = _session()
    request = _request(1)
    await session.begin_turn(request)
    result = _timed_result(" Hi  好! ", 0.1)
    result.channel.alternatives[0].languages = ["en", "zh"]
    result.channel.alternatives[0].words = [
        types.SimpleNamespace(
            word="Hi", punctuated_word="Hi", start=0.1, end=0.3, language="en", speaker=0
        ),
        types.SimpleNamespace(
            word="好", punctuated_word="好!", start=0.4, end=0.6, language="zh", speaker=99
        ),
    ]
    session._build_transcript_event(result)
    update = await _next(session)
    assert [(run.text, run.language) for run in update.final_language_runs] == [
        (" Hi", "en"),
        ("  好! ", "zh"),
    ]
    assert "".join(run.text for run in update.final_language_runs) == update.text
    assert update.final_speaker_runs == ()
    assert update.provenance.transcription.language_code is None
    session._scoped_transport_end(False, "test_finished")


@pytest.mark.asyncio
@pytest.mark.parametrize("correction", ["Correction.", "First."])
async def test_reused_interval_records_unresolved_raw_without_duplicate_or_lost_prefix(correction):
    from puripuly_heart.core.stt.backend import STTContributionConsumptionLedger
    from puripuly_heart.core.stt.scoped_normalizer import STTScopedTurnNormalizer

    session = _session()
    request = _request(1)
    await session.begin_turn(request)
    normalizer = STTScopedTurnNormalizer(request.identity)
    consumption = STTContributionConsumptionLedger()
    for text, start in (("First.", 0.1), ("Second.", 2.0)):
        session._build_transcript_event(_timed_result(text, start))
        assert consumption.consume(normalizer.apply_update(await _next(session))) == text
    session._build_transcript_event(_timed_result(correction, 0.1))
    terminal = await _next(session)
    assert isinstance(terminal, STTProviderTurnTerminal)
    assert terminal.outcome == "failed"
    assert terminal.epoch_disposition == "retire"
    assert terminal.failure_reason == "deepgram_final_source_order_unresolved"
    assert terminal.text == "First.Second."
    assert len(terminal.included_contributions) == 2
    assert terminal.provenance[-1].unresolved_source_text == correction
    assert not terminal.provenance[-1].audio_clocks[0].qualified
    assert dict(terminal.provenance[-1].native_model_metadata)["name"] == "general-nova-3"
    normalized = normalizer.apply_terminal(terminal)
    assert normalized.provenance[-1].unresolved_source_text == correction
    assert consumption.consume(normalized) == ""
    assert isinstance(await _next(session), STTProviderEpochEnded)
    assert session._event_projection.active_identity is None
    with pytest.raises(RuntimeError):
        await session.begin_turn(_request(2))
    recovered = _session()
    next_request = _request(2)
    await recovered.begin_turn(next_request)
    recovered._build_transcript_event(_timed_result("Next.", 0.1))
    assert (await _next(recovered)).text == "Next."
    recovered._scoped_transport_end(False, "test_finished")


@pytest.mark.asyncio
async def test_unresolved_raw_payload_obeys_existing_event_buffer_byte_limit():
    from puripuly_heart.core.stt.backend import STTNativeProvenance
    from puripuly_heart.core.stt.scoped_event_buffer import STTProviderEventBuffer

    buffer = STTProviderEventBuffer()
    terminal = STTProviderTurnTerminal(
        identity=_request(1).identity,
        outcome="failed",
        failure_reason="source_order_unresolved",
        epoch_disposition="retire",
        provenance=(
            STTNativeProvenance(
                unresolved_source_text="不" * (buffer.MAX_RECOGNITION_BYTES // 3 + 1)
            ),
        ),
    )
    assert buffer.put(terminal) is False
    ended = await buffer.get()
    assert isinstance(ended, STTProviderEpochEnded)
    assert ended.reason == "provider_event_buffer_overflow"
