from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import PeerAudioSegmentLedger
from puripuly_heart.core.runtime.local_asr_provider_runtime import LocalASRProviderRuntimeOwner
from puripuly_heart.core.stt.backend import (
    STTContributionConsumptionLedger,
    STTNativeProvenance,
    STTProviderEpochEnded,
    STTProviderInputTerminal,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTRecognitionUnit,
    STTSessionProjection,
)
from puripuly_heart.core.stt.scoped_engine import (
    ScopedRecognitionEngine,
    STTRecognitionWatchdogs,
    STTRetentionProfile,
)
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart
from puripuly_heart.providers.stt.deepgram import _CLOSE_STREAM, _FINALIZE, _DeepgramSDKSession
from puripuly_heart.providers.stt.soniox import _SonioxSession
from tests.helpers.scoped_provider_sessions import (
    _deepgram_result,
    _deepgram_session,
    _FakeSonioxWebSocket,
    _gemini_message,
    _gemini_session,
    _owned_deepgram_events,
    _request,
    _scribe_session,
    _soniox_session,
    _wait,
)


def _request_for_epoch(provider: str, order: int, epoch: str = "epoch-1") -> STTProviderTurnRequest:
    request = _request(provider, order)
    return replace(
        request,
        identity=replace(request.identity, provider_epoch_id=epoch),
    )


def _span() -> tuple[AudioCaptureSpan, ...]:
    return (
        AudioCaptureSpan(
            capture_epoch=1,
            callback_sequence=1,
            source_sample_rate_hz=16000,
            source_start_sample=0,
            source_end_sample=2,
            source_start_monotonic_s=0.0,
            source_end_monotonic_s=0.000125,
            normalized_sample_rate_hz=16000,
            normalized_start_sample=0,
            normalized_end_sample=2,
        ),
    )


async def _next(session):
    return await asyncio.wait_for(anext(session.turn_events()), timeout=1)


@pytest.mark.asyncio
async def test_non_soniox_provider_does_not_add_audio_at_listen_hard_boundary(monkeypatch) -> None:
    writes: list[object] = []

    async def write(_session, payload) -> None:
        writes.append(payload)

    monkeypatch.setattr(_DeepgramSDKSession, "_write_payload", write)
    session = _deepgram_session()
    request = _request("deepgram")
    await session.begin_turn(request)

    await session.seal_turn(
        request.identity,
        sealed_content_ranges=_span(),
        seal_reason="delivery_deadline",
        observed_trailing_silence_ms=0,
    )

    assert writes == [_FINALIZE]
    await session.close()


@pytest.mark.asyncio
async def test_deepgram_actual_result_shape_reuses_acknowledged_epoch_and_detects_idle_result(
    monkeypatch,
) -> None:
    writes: list[tuple[_DeepgramSDKSession, object]] = []
    gate = asyncio.Event()
    gate.set()

    async def write(session, payload) -> None:
        writes.append((session, payload))
        await gate.wait()

    monkeypatch.setattr(_DeepgramSDKSession, "_write_payload", write)
    session = _deepgram_session()
    assert session._audio_q.maxsize == 258
    request_a = _request("deepgram")
    await session.begin_turn(request_a)
    gate.clear()
    send = asyncio.create_task(
        session.send_turn_audio(
            request_a.identity,
            b"pcm-a",
            payload_sequence=1,
            source_ranges=_span(),
            context_only=False,
        )
    )
    await _wait(lambda: any(payload == b"pcm-a" for _, payload in writes))
    assert not send.done()
    gate.set()
    await send
    await session.seal_turn(
        request_a.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=224,
    )
    session._build_transcript_event(_deepgram_result("same "))
    session._build_transcript_event(_deepgram_result("text", start=0.1))
    session._build_transcript_event(_deepgram_result("", from_finalize=True))
    await asyncio.sleep(0)
    assert (await _next(session)).text == "same "
    assert (await _next(session)).text == "text"
    terminal_a = await _next(session)
    assert (terminal_a.outcome, terminal_a.text, terminal_a.epoch_disposition) == (
        "final",
        "same text",
        "reuse",
    )

    request_b = _request_for_epoch("deepgram", 2)
    await session.begin_turn(request_b)
    await session.send_turn_audio(
        request_b.identity,
        b"pcm-b",
        payload_sequence=1,
        source_ranges=_span(),
        context_only=False,
    )
    await session.seal_turn(
        request_b.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=0,
    )
    session._build_transcript_event(_deepgram_result("same text", start=0.2, from_finalize=True))
    await asyncio.sleep(0)
    assert (await _next(session)).text == "same text"
    terminal_b = await _next(session)
    assert (terminal_b.text, terminal_b.epoch_disposition) == ("same text", "reuse")
    assert terminal_a.identity != terminal_b.identity
    assert all(item[0] is session for item in writes)
    assert [payload for _, payload in writes].count(_FINALIZE) == 2
    assert _CLOSE_STREAM not in [payload for _, payload in writes]
    await session.close()

    ambiguous = _deepgram_session()
    ambiguous._build_transcript_event(_deepgram_result("unsolicited", from_finalize=True))
    ended = await _next(ambiguous)
    assert isinstance(ended, STTProviderEpochEnded)
    assert ended.reason == "deepgram_idle_result"
    with pytest.raises(RuntimeError):
        await ambiguous.begin_turn(_request("deepgram"))
    await ambiguous.close()


@pytest.mark.asyncio
async def test_deepgram_empty_error_abort_and_two_drain_path(monkeypatch) -> None:
    writes: list[object] = []

    async def write(_session, payload) -> None:
        writes.append(payload)

    monkeypatch.setattr(_DeepgramSDKSession, "_write_payload", write)
    empty = _deepgram_session()
    request = _request("deepgram")
    await empty.begin_turn(request)
    await empty.seal_turn(
        request.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=0,
    )
    empty._build_transcript_event(_deepgram_result("", from_finalize=True))
    await asyncio.sleep(0)
    terminal = await _next(empty)
    assert (terminal.outcome, terminal.text_authority, terminal.epoch_disposition) == (
        "empty",
        "authoritative",
        "reuse",
    )
    await empty.close()

    errored = _deepgram_session()
    error_request = _request("deepgram")
    await errored.begin_turn(error_request)
    errored._build_transcript_event(_deepgram_result("kept"))
    await asyncio.sleep(0)
    await _next(errored)
    errored._scoped_transport_end(False, "deepgram_transport_error")
    degraded = await _next(errored)
    ended = await _next(errored)
    assert (degraded.outcome, degraded.text) == ("degraded", "kept")
    assert isinstance(ended, STTProviderEpochEnded)
    await errored.close()

    missing = _deepgram_session(0.01)
    missing_request = _request("deepgram")
    await missing.begin_turn(missing_request)
    await missing.seal_turn(
        missing_request.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=0,
    )
    missing_terminal = await _next(missing)
    assert missing_terminal.failure_reason == "deepgram_finalize_ack_missing"
    assert writes.count(_CLOSE_STREAM) == 1
    await missing.close()

    aborted = _deepgram_session()
    abort_request = _request("deepgram")
    await aborted.begin_turn(abort_request)
    await aborted.abort_turn(abort_request.identity, reason="cancelled")
    aborted_terminal = await _next(aborted)
    assert (aborted_terminal.outcome, aborted_terminal.failure_reason) == (
        "cancelled",
        "cancelled",
    )
    assert isinstance(await _next(aborted), STTProviderEpochEnded)
    with pytest.raises(RuntimeError):
        await aborted.begin_turn(_request("deepgram", 2))
    await aborted.close()
    assert writes.count(_FINALIZE) == 2


@pytest.mark.asyncio
async def test_gemini_native_results_are_stream_scoped_receipt_units() -> None:
    session, live = await _gemini_session()
    first = _request("gemini_transcribe")
    second = _request_for_epoch("gemini_transcribe", 2)
    try:
        await session.begin_turn(first)
        await session.send_turn_audio(
            first.identity,
            b"\x01\x00\x02\x00",
            payload_sequence=1,
            source_ranges=_span(),
            context_only=False,
        )
        live.push(_gemini_message(final="before seal"))
        early = await _next(session)
        assert isinstance(early, STTRecognitionUnit)
        await session.seal_turn(
            first.identity,
            sealed_content_ranges=_span(),
            seal_reason="silence",
            observed_trailing_silence_ms=0,
        )
        submitted = await _next(session)
        assert isinstance(submitted, STTProviderInputTerminal)
        assert submitted.independent_recognition
        assert (submitted.outcome, submitted.identity) == ("submitted", first.identity)
        await session.begin_turn(second)
        next_span = (
            replace(
                _span()[0],
                source_start_sample=2,
                source_end_sample=4,
                normalized_start_sample=2,
                normalized_end_sample=4,
                source_start_monotonic_s=2 / 16000,
                source_end_monotonic_s=4 / 16000,
            ),
        )
        await session.send_turn_audio(
            second.identity,
            b"\x03\x00\x04\x00",
            payload_sequence=1,
            source_ranges=next_span,
            context_only=False,
        )
        assert session.recognition_source_covers(next_span)
        live.push(_gemini_message(ack=True, interim="not final"))
        live.push(_gemini_message(final="same"))
        live.push(_gemini_message(final="same"))
        units = [await _next(session), await _next(session)]
        assert all(isinstance(unit, STTRecognitionUnit) for unit in units)
        assert [unit.text for unit in units] == ["same", "same"]
        assert [unit.identity.receipt_sequence for unit in [early, *units]] == [1, 2, 3]
        assert len({unit.identity.unit_id for unit in [early, *units]}) == 3
        assert all(unit.identity.stream == early.identity.stream for unit in units)
        assert [next(iter(item)) for item in live.sent] == ["audio", "audio_stream_end", "audio"]
        await session.seal_turn(
            second.identity,
            sealed_content_ranges=next_span,
            seal_reason="silence",
            observed_trailing_silence_ms=0,
        )
        assert (await _next(session)).outcome == "submitted"
        assert live.closed is False
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_gemini_empty_result_transport_error_and_abort_have_distinct_receipts() -> None:
    empty, empty_live = await _gemini_session()
    request = _request("gemini_transcribe")
    try:
        await empty.begin_turn(request)
        empty_live.push(_gemini_message(final=""))
        unit = await _next(empty)
        assert isinstance(unit, STTRecognitionUnit)
        assert unit.text == ""
        await empty.seal_turn(
            request.identity,
            sealed_content_ranges=(),
            seal_reason="silence",
            observed_trailing_silence_ms=0,
        )
        assert (await _next(empty)).outcome == "submitted"
    finally:
        await empty.close()

    errored, error_live = await _gemini_session()
    error_request = _request("gemini_transcribe")
    try:
        await errored.begin_turn(error_request)
        error_live.push(RuntimeError("receive failed"))
        failed = await _next(errored)
        ended = await _next(errored)
        assert isinstance(failed, STTProviderInputTerminal)
        assert failed.independent_recognition
        assert failed.failure_reason == "gemini_receive_failed"
        assert isinstance(ended, STTProviderEpochEnded)
    finally:
        await errored.close()

    aborted, _ = await _gemini_session()
    abort_request = _request("gemini_transcribe")
    try:
        await aborted.begin_turn(abort_request)
        await aborted.abort_turn(abort_request.identity, reason="cancelled")
        aborted_terminal = await _next(aborted)
        assert isinstance(aborted_terminal, STTProviderInputTerminal)
        assert aborted_terminal.independent_recognition
        assert (aborted_terminal.outcome, aborted_terminal.failure_reason) == (
            "cancelled",
            "cancelled",
        )
        assert isinstance(await _next(aborted), STTProviderEpochEnded)
    finally:
        await aborted.close()


@pytest.mark.asyncio
async def test_soniox_unkeyed_tokens_preserve_source_through_reuse_and_idle_stream() -> None:
    session, ws = _soniox_session()
    assert session._audio_q.maxsize == 258
    request_a = _request("soniox")
    await session.begin_turn(request_a)
    ws.send_gate.clear()
    send = asyncio.create_task(
        session.send_turn_audio(
            request_a.identity,
            b"pcm-a",
            payload_sequence=1,
            source_ranges=_span(),
            context_only=False,
        )
    )
    await _wait(lambda: ws.sent == [b"pcm-a"])
    assert not send.done()
    ws.send_gate.set()
    await send
    await session.seal_turn(
        request_a.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=224,
    )
    ws.push(
        json.dumps(
            {
                "tokens": [
                    {"text": "same ", "is_final": True, "end_ms": 100, "language": "en"},
                    {"text": "text", "is_final": True, "end_ms": 200, "language": "en"},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )
    assert (await _next(session)).text == "same "
    assert (await _next(session)).text == "text"
    terminal_a = await _next(session)
    assert (terminal_a.text, terminal_a.epoch_disposition) == ("same text", "reuse")

    request_b = _request_for_epoch("soniox", 2)
    await session.begin_turn(request_b)
    await session.send_turn_audio(
        request_b.identity,
        b"pcm-b",
        payload_sequence=1,
        source_ranges=_span(),
        context_only=False,
    )
    await session.seal_turn(
        request_b.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=0,
    )
    ws.push(
        json.dumps(
            {
                "tokens": [
                    {"text": "same text", "is_final": True, "end_ms": 200, "language": "en"},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )
    source_b = await _next(session)
    assert isinstance(source_b, STTRecognitionUnit)
    assert source_b.text == "same text"
    terminal_b = await _next(session)
    assert (terminal_b.text, terminal_b.outcome, terminal_b.epoch_disposition) == (
        "",
        "empty",
        "reuse",
    )
    assert terminal_a.identity != terminal_b.identity
    assert source_b.identity.stream.provider_epoch_id == request_b.identity.provider_epoch_id
    assert ws.closed is False
    assert b"pcm-a" in ws.sent and b"pcm-b" in ws.sent
    ws.push(
        json.dumps(
            {
                "tokens": [
                    {"text": "<fin>", "is_final": True},
                    {"text": " idle tail ", "is_final": True, "end_ms": 300, "language": "en"},
                    {"text": "<end>", "is_final": True},
                ]
            }
        )
    )
    idle_source = await _next(session)
    assert isinstance(idle_source, STTRecognitionUnit)
    assert idle_source.text == " idle tail "
    assert idle_source.identity.stream == source_b.identity.stream
    assert idle_source.identity.receipt_sequence > source_b.identity.receipt_sequence
    assert session._event_projection.active_identity is None
    assert not session._event_projection.retired
    await session.close()
    assert ws.closed is True


@pytest.mark.asyncio
async def test_soniox_empty_error_and_abort_receipts() -> None:
    empty, ws = _soniox_session()
    request = _request("soniox")
    await empty.begin_turn(request)
    await empty.seal_turn(
        request.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=0,
    )
    ws.push(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))
    terminal = await _next(empty)
    assert (terminal.outcome, terminal.text_authority, terminal.epoch_disposition) == (
        "empty",
        "authoritative",
        "reuse",
    )
    await empty.close()

    errored, error_ws = _soniox_session()
    error_request = _request("soniox")
    await errored.begin_turn(error_request)
    error_ws.push(json.dumps({"error": "bad"}))
    failed = await _next(errored)
    ended = await _next(errored)
    assert failed.failure_reason == "soniox_request_failed"
    assert isinstance(ended, STTProviderEpochEnded)
    await errored.close()

    aborted, _ = _soniox_session()
    abort_request = _request("soniox")
    await aborted.begin_turn(abort_request)
    await aborted.abort_turn(abort_request.identity, reason="cancelled")
    aborted_terminal = await _next(aborted)
    assert (aborted_terminal.outcome, aborted_terminal.failure_reason) == (
        "cancelled",
        "cancelled",
    )
    assert isinstance(await _next(aborted), STTProviderEpochEnded)
    with pytest.raises(RuntimeError):
        await aborted.begin_turn(_request("soniox", 2))
    await aborted.close()


@pytest.mark.asyncio
async def test_scribe_actual_payload_shape_reuses_connection_and_detects_idle_ambiguity() -> None:
    from elevenlabs.types import (
        CommittedTranscriptPayload,
        CommittedTranscriptWithTimestampsPayload,
        PartialTranscriptPayload,
        TranscriptionWord,
    )

    for payload_type in (CommittedTranscriptPayload, CommittedTranscriptWithTimestampsPayload):
        assert not {"id", "transcript_id", "commit_id"}.intersection(payload_type.model_fields)
    session, connection = _scribe_session()
    assert session._connection_events.maxsize == 258
    request_a = _request("elevenlabs_scribe")
    await session.begin_turn(request_a)
    session._on_committed(CommittedTranscriptPayload(text="before-commit"))
    assert session._event_projection.scoped_event_depth == 0
    connection.send_gate.clear()
    send = asyncio.create_task(
        session.send_turn_audio(
            request_a.identity,
            b"pcm-a",
            payload_sequence=1,
            source_ranges=_span(),
            context_only=False,
        )
    )
    await _wait(lambda: bool(connection.sent))
    assert not send.done()
    connection.send_gate.set()
    await send
    session._on_partial(PartialTranscriptPayload(text="same"))
    assert (await _next(session)).text == "same"
    connection.commit_gate.clear()
    seal = asyncio.create_task(
        session.seal_turn(
            request_a.identity,
            sealed_content_ranges=_span(),
            seal_reason="silence",
            observed_trailing_silence_ms=224,
        )
    )
    await _wait(lambda: connection.commits == 1)
    assert not seal.done()
    session._on_committed(CommittedTranscriptPayload(text="different plain completion"))
    session._on_committed_with_timestamps(CommittedTranscriptWithTimestampsPayload(text="same"))
    assert session._event_projection.scoped_event_depth == 0
    connection.commit_gate.set()
    await seal
    terminal_a = await _next(session)
    assert (terminal_a.text, terminal_a.epoch_disposition) == ("same", "reuse")

    request_b = _request_for_epoch("elevenlabs_scribe", 2)
    await session.begin_turn(request_b)
    await session.send_turn_audio(
        request_b.identity,
        b"pcm-b",
        payload_sequence=1,
        source_ranges=_span(),
        context_only=False,
    )
    await session.seal_turn(
        request_b.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=0,
    )
    session._on_committed_with_timestamps(
        CommittedTranscriptWithTimestampsPayload(
            text="same",
            words=[TranscriptionWord(text="same", type="word", start=1.0, end=1.2)],
        )
    )
    terminal_b = await _next(session)
    assert (terminal_b.text, terminal_b.epoch_disposition) == ("same", "reuse")
    assert terminal_a.identity != terminal_b.identity
    assert connection.commits == 2
    assert connection.closed is False
    await session.close()
    assert connection.closed is True

    ambiguous, ambiguous_connection = _scribe_session()
    ambiguous._on_committed(CommittedTranscriptPayload(text="unsolicited plain"))
    assert ambiguous._event_projection.scoped_event_depth == 0
    ambiguous._on_committed_with_timestamps(
        CommittedTranscriptWithTimestampsPayload(text="unowned source")
    )
    ended = await _next(ambiguous)
    assert isinstance(ended, STTProviderEpochEnded)
    assert ended.reason == "scribe_unsolicited_timestamped_transcript"
    with pytest.raises(RuntimeError):
        await ambiguous.begin_turn(_request("elevenlabs_scribe"))
    await ambiguous.close()
    assert ambiguous_connection.closed is True


@pytest.mark.asyncio
async def test_scribe_empty_error_and_abort_receipts() -> None:
    from elevenlabs.types import CommittedTranscriptWithTimestampsPayload

    empty, _ = _scribe_session()
    request = _request("elevenlabs_scribe")
    await empty.begin_turn(request)
    await empty.seal_turn(
        request.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=0,
    )
    empty._on_committed_with_timestamps(CommittedTranscriptWithTimestampsPayload(text=""))
    terminal = await _next(empty)
    assert (terminal.outcome, terminal.text_authority, terminal.epoch_disposition) == (
        "empty",
        "authoritative",
        "reuse",
    )
    await empty.close()

    errored, _ = _scribe_session()
    error_request = _request("elevenlabs_scribe")
    await errored.begin_turn(error_request)
    errored._on_error_event({"message_type": "quota_exceeded"})
    failed = await _next(errored)
    ended = await _next(errored)
    assert failed.failure_reason == "scribe_quota_exceeded"
    assert isinstance(ended, STTProviderEpochEnded)
    await errored.close()

    aborted, _ = _scribe_session()
    abort_request = _request("elevenlabs_scribe")
    await aborted.begin_turn(abort_request)
    await aborted.abort_turn(abort_request.identity, reason="cancelled")
    aborted_terminal = await _next(aborted)
    assert (aborted_terminal.outcome, aborted_terminal.failure_reason) == (
        "cancelled",
        "cancelled",
    )
    assert isinstance(await _next(aborted), STTProviderEpochEnded)
    with pytest.raises(RuntimeError):
        await aborted.begin_turn(_request("elevenlabs_scribe", 2))
    await aborted.close()


class _NoopGpuRuntime:
    state = "idle"
    discovery_state = "idle"
    active_channels = frozenset()
    pending_count = 0
    worker_pid = None
    last_failure_code = None
    configured_device_id = None
    startup_cleanup_snapshot = {
        "pending_startups": 0,
        "close_complete": True,
        "cleanup_failures": [],
    }

    async def close(self) -> None:
        return None


class _UnusedProviderFactory:
    async def create(self, request, *, gpu_runtime, on_terminal_failure=None):
        raise AssertionError((request, gpu_runtime, on_terminal_failure))


def _deepgram_engine(
    channel: str,
    sessions: list[_DeepgramSDKSession],
) -> ScopedRecognitionEngine:
    async def open_session(settings, provider_epoch_id):
        session = _DeepgramSDKSession(
            api_key="k",
            model="nova-3",
            language="en",
            sample_rate_hz=16000,
            connect_timeout_s=5.0,
            keyterms=[],
            drain_timeout_s=0.2,
            projection=STTSessionProjection("scoped", provider_epoch_id),
        )
        sessions.append(session)
        return session

    return ScopedRecognitionEngine(
        channel=channel,
        session_factory=open_session,
        watchdog_resolver=lambda _settings: STTRecognitionWatchdogs(
            write_timeout_s=0.5,
            final_timeout_s=0.5,
            drain_timeout_s=0.2,
        ),
        accepted_settings_scope=("deepgram", ("deepgram",), ("deepgram",)),
        retention_profile_resolver=lambda _settings: STTRetentionProfile(
            max_retained_samples=4,
            max_retained_bytes=8,
            release_after_write=True,
        ),
    )


@pytest.mark.asyncio
async def test_shared_engine_real_owner_and_deepgram_adapter_serve_both_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_gate = asyncio.Event()
    write_gate.set()
    writes: list[tuple[_DeepgramSDKSession, object]] = []

    async def write(session: _DeepgramSDKSession, payload: object) -> None:
        writes.append((session, payload))
        await write_gate.wait()

    monkeypatch.setattr(_DeepgramSDKSession, "_write_payload", write)
    peer_sessions: list[_DeepgramSDKSession] = []
    self_sessions: list[_DeepgramSDKSession] = []
    peer_events: list[object] = []
    self_events: list[object] = []
    consumption = STTContributionConsumptionLedger()
    consumed: list[str] = []

    async def self_consumer(event: object) -> None:
        self_events.append(event)
        if isinstance(event, STTProviderTurnUpdate) and event.source_release is not None:
            text = consumption.consume(event)
            if text:
                consumed.append(text)
        elif isinstance(event, STTProviderTurnTerminal):
            text = consumption.consume(event)
            if text:
                consumed.append(text)

    peer_engine = _deepgram_engine("peer", peer_sessions)
    self_engine = _deepgram_engine("self", self_sessions)
    owner = LocalASRProviderRuntimeOwner(
        provider_factory=_UnusedProviderFactory(),
        gpu_runtime_factory=lambda _sink: _NoopGpuRuntime(),
        provisioning=object(),
        self_event_handler=self_consumer,
        peer_event_handler=lambda event: asyncio.sleep(0, result=peer_events.append(event)),
        prebuilt_providers={"self": self_engine, "peer": peer_engine},
    )
    await owner.start()
    settings = _request("deepgram").settings
    peer_ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings)
    self_ledger = PeerAudioSegmentLedger(activation_generation=2, settings=settings)
    client_events = {
        "peer": _owned_deepgram_events(peer_ledger, start_sample=100),
        "self": _owned_deepgram_events(self_ledger, start_sample=1000),
    }

    for channel in ("peer", "self"):
        start, chunks, end = client_events[channel]
        writes_before = len(writes)
        write_gate.clear()
        start_task = asyncio.create_task(owner.handle_owned_vad_event(channel, start))
        sessions = peer_sessions if channel == "peer" else self_sessions
        await _wait(lambda: len(sessions) == 1 and len(writes) > writes_before)
        engine = peer_engine if channel == "peer" else self_engine
        session = sessions[-1]
        assert engine.retention_snapshot.retained_samples == 4
        assert engine.retention_snapshot.retained_bytes == 8
        assert start_task.done() is False
        write_gate.set()
        await start_task
        assert engine.retention_snapshot.retained_samples == 0
        assert engine.retention_snapshot.retained_bytes == 0
        assert engine.retention_snapshot.high_water_samples == 4
        for chunk in chunks:
            await owner.handle_owned_vad_event(channel, chunk)
        assert engine.retention_snapshot.high_water_samples == 4
        identity = session._event_projection.active_identity
        assert identity is not None
        session._build_transcript_event(_deepgram_result("A"))
        await _wait(
            lambda: any(
                isinstance(event, STTProviderTurnUpdate)
                for event in (peer_events if channel == "peer" else self_events)
            )
        )
        assert session._event_projection.sealed is False
        end_task = asyncio.create_task(owner.handle_owned_vad_event(channel, end))
        await _wait(lambda: session._event_projection.sealed)
        session._build_transcript_event(_deepgram_result("B", start=0.1))
        session._build_transcript_event(_deepgram_result("", from_finalize=True, is_final=False))
        await end_task

    peer_terminal = next(
        event for event in peer_events if isinstance(event, STTProviderTurnTerminal)
    )
    self_terminal = next(
        event for event in self_events if isinstance(event, STTProviderTurnTerminal)
    )
    assert peer_terminal.text == "AB"
    assert [item.text_start for item in peer_terminal.included_contributions] == [0, 1]
    assert self_terminal.text == "AB"
    assert consumed == ["A", "B"]
    assert peer_engine.channel == "peer"
    assert self_engine.channel == "self"
    assert len(writes) == 44
    await owner.close()


@pytest.mark.asyncio
async def test_two_scoped_deepgram_clients_isolate_abort_and_native_late_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def write(_session: _DeepgramSDKSession, _payload: object) -> None:
        return None

    monkeypatch.setattr(_DeepgramSDKSession, "_write_payload", write)
    peer_sessions: list[_DeepgramSDKSession] = []
    self_sessions: list[_DeepgramSDKSession] = []
    peer_events: list[object] = []
    self_events: list[object] = []
    peer_engine = _deepgram_engine("peer", peer_sessions)
    self_engine = _deepgram_engine("self", self_sessions)
    owner = LocalASRProviderRuntimeOwner(
        provider_factory=_UnusedProviderFactory(),
        gpu_runtime_factory=lambda _sink: _NoopGpuRuntime(),
        provisioning=object(),
        self_event_handler=lambda event: asyncio.sleep(0, result=self_events.append(event)),
        peer_event_handler=lambda event: asyncio.sleep(0, result=peer_events.append(event)),
        prebuilt_providers={"self": self_engine, "peer": peer_engine},
    )
    await owner.start()
    settings = _request("deepgram").settings
    peer_start, _peer_chunks, peer_end = _owned_deepgram_events(
        PeerAudioSegmentLedger(activation_generation=1, settings=settings),
        start_sample=2000,
    )
    self_start, _self_chunks, _self_end = _owned_deepgram_events(
        PeerAudioSegmentLedger(activation_generation=2, settings=settings),
        start_sample=3000,
    )
    await owner.handle_owned_vad_event("peer", peer_start)
    await owner.handle_owned_vad_event("self", self_start)
    retired_self_identity = self_sessions[0]._event_projection.active_identity
    await self_engine.abort(reason="self_off")
    self_sessions[0]._handle_scoped_result(
        retired_self_identity,
        "late-self",
        True,
        True,
        STTNativeProvenance(native_event_id="late"),
    )
    peer_end_task = asyncio.create_task(owner.handle_owned_vad_event("peer", peer_end))
    await _wait(lambda: peer_sessions[0]._event_projection.sealed)
    peer_sessions[0]._build_transcript_event(_deepgram_result("peer-ok"))
    peer_sessions[0]._build_transcript_event(
        _deepgram_result("", from_finalize=True, is_final=False)
    )
    await peer_end_task
    await _wait(lambda: any(isinstance(event, STTProviderTurnTerminal) for event in peer_events))

    assert [
        (event.outcome, event.failure_reason)
        for event in self_events
        if isinstance(event, STTProviderTurnTerminal)
    ] == [("cancelled", "self_off")]
    assert [event.text for event in peer_events if isinstance(event, STTProviderTurnTerminal)] == [
        "peer-ok"
    ]
    assert owner.snapshot.channel_for("peer").phase == "running"
    await owner.close()


@pytest.mark.asyncio
async def test_scoped_configuration_handoff_is_channel_local_with_concrete_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def write(_session: _DeepgramSDKSession, _payload: object) -> None:
        return None

    monkeypatch.setattr(_DeepgramSDKSession, "_write_payload", write)
    peer_sessions: list[_DeepgramSDKSession] = []
    old_self_sessions: list[_DeepgramSDKSession] = []
    new_self_sessions: list[_DeepgramSDKSession] = []
    peer_events: list[object] = []
    self_events: list[object] = []
    peer_engine = _deepgram_engine("peer", peer_sessions)
    old_self_engine = _deepgram_engine("self", old_self_sessions)
    new_self_engine = _deepgram_engine("self", new_self_sessions)
    new_self_engine.accepted_settings_scope = (
        "deepgram",
        ("deepgram",),
        ("deepgram-new",),
    )
    owner = LocalASRProviderRuntimeOwner(
        provider_factory=_UnusedProviderFactory(),
        gpu_runtime_factory=lambda _sink: _NoopGpuRuntime(),
        provisioning=object(),
        self_event_handler=lambda event: asyncio.sleep(0, result=self_events.append(event)),
        peer_event_handler=lambda event: asyncio.sleep(0, result=peer_events.append(event)),
        prebuilt_providers={"self": old_self_engine, "peer": peer_engine},
    )
    await owner.start()
    old_settings = _request("deepgram").settings
    peer_start, _peer_chunks, peer_end = _owned_deepgram_events(
        PeerAudioSegmentLedger(activation_generation=1, settings=old_settings),
        start_sample=4000,
    )
    await owner.handle_owned_vad_event("peer", peer_start)
    peer_session = peer_sessions[0]
    peer_identity = peer_session._event_projection.active_identity

    await owner.handoff_prebuilt_provider("self", new_self_engine, start=True)
    new_settings = replace(old_settings, runtime_signature=("deepgram-new",))
    self_start, _self_chunks, self_end = _owned_deepgram_events(
        PeerAudioSegmentLedger(activation_generation=2, settings=new_settings),
        start_sample=5000,
    )
    await owner.handle_owned_vad_event("self", self_start)
    assert old_self_sessions == []
    assert len(new_self_sessions) == 1
    assert peer_sessions == [peer_session]
    assert peer_session._event_projection.active_identity == peer_identity

    self_end_task = asyncio.create_task(owner.handle_owned_vad_event("self", self_end))
    await _wait(lambda: new_self_sessions[0]._event_projection.sealed)
    new_self_sessions[0]._build_transcript_event(_deepgram_result("new-self"))
    new_self_sessions[0]._build_transcript_event(
        _deepgram_result("", from_finalize=True, is_final=False)
    )
    await self_end_task

    peer_end_task = asyncio.create_task(owner.handle_owned_vad_event("peer", peer_end))
    await _wait(lambda: peer_session._event_projection.sealed)
    peer_session._build_transcript_event(_deepgram_result("peer-survived"))
    peer_session._build_transcript_event(_deepgram_result("", from_finalize=True, is_final=False))
    await peer_end_task
    await _wait(lambda: any(isinstance(event, STTProviderTurnTerminal) for event in peer_events))
    assert [event.text for event in self_events if isinstance(event, STTProviderTurnTerminal)] == [
        "new-self"
    ]
    assert [event.text for event in peer_events if isinstance(event, STTProviderTurnTerminal)] == [
        "peer-survived"
    ]
    await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_id",
    ["deepgram", "gemini_transcribe", "soniox", "elevenlabs_scribe"],
)
async def test_each_concrete_streaming_protocol_serves_self_and_peer_concurrently(
    provider_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elevenlabs.types import CommittedTranscriptWithTimestampsPayload

    async def deepgram_write(_session, _payload) -> None:
        return None

    monkeypatch.setattr(_DeepgramSDKSession, "_write_payload", deepgram_write)
    sessions: list[object] = []
    native_boundaries: list[object | None] = []
    for order in (1, 2):
        if provider_id == "deepgram":
            session, boundary = _deepgram_session(order=order), None
        elif provider_id == "gemini_transcribe":
            session, boundary = await _gemini_session(order=order)
        elif provider_id == "soniox":
            session, boundary = _soniox_session(order=order)
        else:
            session, boundary = _scribe_session(order=order)
        sessions.append(session)
        native_boundaries.append(boundary)

    requests = [
        replace(_request(provider_id, order), channel=channel)
        for order, channel in ((1, "self"), (2, "peer"))
    ]
    await asyncio.gather(
        *(session.begin_turn(request) for session, request in zip(sessions, requests, strict=True))
    )
    await asyncio.gather(
        *(
            session.send_turn_audio(
                request.identity,
                b"\x01\x00\x02\x00" if provider_id == "gemini_transcribe" else b"pcm",
                payload_sequence=1,
                source_ranges=_span(),
                context_only=False,
            )
            for session, request in zip(sessions, requests, strict=True)
        )
    )
    await asyncio.gather(
        *(
            session.seal_turn(
                request.identity,
                sealed_content_ranges=_span(),
                seal_reason="silence",
                observed_trailing_silence_ms=0,
            )
            for session, request in zip(sessions, requests, strict=True)
        )
    )

    for index, (session, boundary) in enumerate(
        zip(sessions, native_boundaries, strict=True),
        start=1,
    ):
        text = f"client-{index}"
        if provider_id == "deepgram":
            session._build_transcript_event(_deepgram_result(text, from_finalize=True))
        elif provider_id == "gemini_transcribe":
            boundary.push(_gemini_message(final=text))
        elif provider_id == "soniox":
            boundary.push(
                json.dumps(
                    {
                        "tokens": [
                            {"text": text, "is_final": True, "language": "en"},
                            {"text": "<fin>", "is_final": True},
                        ]
                    }
                )
            )
        else:
            session._on_committed_with_timestamps(
                CommittedTranscriptWithTimestampsPayload(text=text)
            )

    terminals: list[STTProviderTurnTerminal | STTRecognitionUnit] = []
    for session in sessions:
        if provider_id == "gemini_transcribe":
            submitted = await _next(session)
            assert isinstance(submitted, STTProviderInputTerminal)
            assert submitted.independent_recognition
            assert submitted.outcome == "submitted"
            unit = await _next(session)
            assert isinstance(unit, STTRecognitionUnit)
            terminals.append(unit)
        else:
            while True:
                event = await _next(session)
                if isinstance(event, STTProviderTurnTerminal):
                    terminals.append(event)
                    break

    assert [terminal.text for terminal in terminals] == ["client-1", "client-2"]
    assert [request.channel for request in requests] == ["self", "peer"]
    assert terminals[0].identity != terminals[1].identity
    await asyncio.gather(*(session.close() for session in sessions))


@pytest.mark.asyncio
async def test_real_soniox_shared_engine_preserves_trailing_token_separator() -> None:
    sessions: list[_SonioxSession] = []
    sockets: list[_FakeSonioxWebSocket] = []
    emitted: list[object] = []
    consumption = STTContributionConsumptionLedger()

    async def open_session(_settings, provider_epoch_id):
        ws = _FakeSonioxWebSocket()
        session = _SonioxSession(
            api_key="k",
            model="model",
            endpoint="endpoint",
            sample_rate_hz=16000,
            language_hints=["en", "ja"],
            context_terms=[],
            keepalive_interval_s=100.0,
            trailing_silence_ms=100,
            connect_timeout_s=5.0,
            enable_language_identification=True,
            projection=STTSessionProjection("scoped", provider_epoch_id),
        )
        session._ws = ws
        session._send_task = asyncio.create_task(session._send_loop())
        session._recv_task = asyncio.create_task(session._recv_loop())
        sessions.append(session)
        sockets.append(ws)
        return session

    settings = _request("soniox").settings
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings)
    segment_id = uuid4()
    content = AudioCaptureSpan(
        capture_epoch=1,
        callback_sequence=1,
        source_sample_rate_hz=16000,
        source_start_sample=4000,
        source_end_sample=68000,
        source_start_monotonic_s=0.25,
        source_end_monotonic_s=4.25,
        normalized_sample_rate_hz=16000,
        normalized_start_sample=4000,
        normalized_end_sample=68000,
    )
    start = ledger.observe_vad_event(
        SpeechStart(
            segment_id,
            np.empty((0,), dtype=np.float32),
            np.ones(64000, dtype=np.float32),
            chunk_capture=(content,),
        ),
        now_monotonic_s=0.25,
    )
    chunks: list[object] = []
    end = ledger.observe_vad_event(
        SpeechEnd(segment_id, trailing_silence_ms=224, reason="delivery_pause"),
        now_monotonic_s=4.25,
    )
    engine = ScopedRecognitionEngine(
        channel="peer",
        session_factory=open_session,
        event_sink=emitted.append,
        watchdog_resolver=lambda _settings: STTRecognitionWatchdogs(
            write_timeout_s=0.5,
            final_timeout_s=0.5,
            drain_timeout_s=0.2,
        ),
    )

    await engine.handle_owned_vad_event(start)
    for chunk in chunks:
        await engine.handle_owned_vad_event(chunk)
    end_task = asyncio.create_task(engine.handle_owned_vad_event(end))
    await _wait(lambda: sessions[0]._event_projection.sealed)
    await _wait(
        lambda: any(
            isinstance(payload, str) and payload and json.loads(payload).get("type") == "finalize"
            for payload in sockets[0].sent
        )
    )
    finalize_index = next(
        index
        for index, payload in enumerate(sockets[0].sent)
        if isinstance(payload, str) and payload and json.loads(payload).get("type") == "finalize"
    )
    assert sockets[0].sent[finalize_index - 1] == bytes(16000 * 200 // 1000 * 2)
    sockets[0].push(
        json.dumps(
            {
                "tokens": [
                    {"text": "same ", "is_final": True, "end_ms": 100, "language": "en"},
                    {"text": "世界", "is_final": True, "end_ms": 200, "language": "ja"},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )
    await end_task

    updates = [event for event in emitted if isinstance(event, STTProviderTurnUpdate)]
    terminal = next(event for event in emitted if isinstance(event, STTProviderTurnTerminal))
    assert [event.text for event in updates] == ["same ", "same 世界"]
    assert [(item.text_start, item.text_end) for item in terminal.included_contributions] == [
        (0, 5),
        (5, 7),
    ]
    assert len({item.contribution_id for item in terminal.included_contributions}) == 2
    assert consumption.consume(updates[0]) == "same "
    assert consumption.consume(terminal) == "世界"
    assert terminal.text == "same 世界"
    assert "".join(run.text for run in terminal.final_language_runs) == terminal.text
    await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["deepgram", "elevenlabs_scribe"])
@pytest.mark.parametrize("boundary_case", ["clean", "crossing", "unproven", "unconfirmed"])
async def test_actual_native_word_boundaries_partition_source_without_word_jobs(
    provider_id: str,
    boundary_case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepgram.extensions.types.sockets.listen_v1_results_event import ListenV1Word
    from elevenlabs.types import CommittedTranscriptWithTimestampsPayload, TranscriptionWord

    from puripuly_heart.core.audio.ownership import AudioSegmentTerminalReceipt
    from puripuly_heart.core.stt.scoped_normalizer import STTScopedTurnNormalizer
    from puripuly_heart.core.stt.streaming_speaker import StreamingSpeakerSession
    from tests.core.test_source_context_contract import (
        ControlledProvider,
        close,
        finish,
        harness_for,
        source_fixture,
    )
    from tests.core.test_streaming_speaker import evidence, next_text, send, span, temporal_frames

    async def write(_session, _payload):
        return None

    monkeypatch.setattr(_DeepgramSDKSession, "_write_payload", write)
    snapshot, identity = source_fixture()
    request = STTProviderTurnRequest(
        identity, snapshot.settings, display_group_id=snapshot.display_group_id
    )
    if provider_id == "deepgram":
        native = _deepgram_session(epoch=identity.provider_epoch_id)
    else:
        native, _ = _scribe_session(epoch=identity.provider_epoch_id, model="scribe_v2_realtime")
    wrapper = StreamingSpeakerSession(native)
    translation = ControlledProvider()
    harness, trace = harness_for(translation)
    words = [
        ("Alpha,", 0.0, 0.04),
        ("again!", 0.04, 0.08),
    ]
    if boundary_case != "clean":
        words.append(("crossing!", 0.09, 0.17))
    words.extend([("Beta,", 0.20, 0.24), ("later.", 0.24, 0.28)])
    raw_text = (
        "Alpha, again!  " + ("crossing! " if boundary_case != "clean" else "") + "Beta, later."
    )
    try:
        await wrapper.begin_turn(request)
        await send(wrapper, request, 0, 4800, 1)
        speaker_evidence = (
            evidence(0, 1600, 0),
            evidence(1600, 2400, None),
            evidence(2400, 4800, 1),
        )
        await wrapper.observe_speaker_evidence(
            speaker_evidence
            if boundary_case == "unconfirmed"
            else temporal_frames(speaker_evidence)
        )
        if provider_id == "deepgram":
            result = _deepgram_result(raw_text, duration=0.28)
            native_words = [
                ListenV1Word(
                    word=word,
                    punctuated_word=word,
                    start=start,
                    end=end,
                    confidence=1.0,
                    speaker=99,
                )
                for word, start, end in words
            ]
            if boundary_case == "unproven":
                native_words[0] = native_words[0].model_copy(
                    update={
                        "word": "mismatch",
                        "punctuated_word": "mismatch",
                    }
                )
            result = result.model_copy(
                update={
                    "channel": result.channel.model_copy(
                        update={
                            "alternatives": [
                                result.channel.alternatives[0].model_copy(
                                    update={
                                        "words": native_words,
                                    }
                                )
                            ],
                        }
                    ),
                }
            )
            native._build_transcript_event(result)
        else:
            native_words = []
            cursor = 0
            for word, start, end in words:
                offset = raw_text.index(word, cursor)
                if offset > cursor:
                    native_words.append(
                        TranscriptionWord(
                            text=raw_text[cursor:offset],
                            start=start,
                            end=start,
                            type="spacing",
                        )
                    )
                native_words.append(
                    TranscriptionWord(
                        text=word,
                        start=start,
                        end=end,
                        type=None if boundary_case == "unproven" else "word",
                        speaker_id="native-wrong",
                    )
                )
                cursor = offset + len(word)
            native._on_committed_with_timestamps(
                CommittedTranscriptWithTimestampsPayload(
                    text=raw_text,
                    words=native_words,
                    language_code="en",
                )
            )
        stream = wrapper.turn_events()
        outgoing = STTScopedTurnNormalizer(identity).apply_update(await next_text(stream))
        await harness.peer_owner.handle_provider_turn_update(snapshot, outgoing)
        await finish(harness)
        sources = [item.source_child for item in trace.items]
        assert "".join(source.raw_text for source in sources) == raw_text
        if boundary_case in ("unproven", "unconfirmed"):
            assert len(sources) == 1
            assert sources[0].speaker.speaker_id is None
        else:
            expected = ["Alpha, again!  ", "Beta, later."]
            if boundary_case == "crossing":
                expected[0] += "crossing! "
            assert [source.raw_text for source in sources] == expected
            assert [source.speaker.speaker_id for source in sources] == (
                [None, "1"] if boundary_case == "crossing" else ["0", "1"]
            )
            assert len(sources) < len(words)
        assert len(translation.calls) == len(sources)
        await harness.peer_owner.handle_provider_turn_update(snapshot, outgoing)
        await finish(harness)
        assert len(trace.items) == len(sources)
        await wrapper.seal_turn(
            identity,
            sealed_content_ranges=(span(0, 4800),),
            seal_reason="silence",
            observed_trailing_silence_ms=200,
        )
        if provider_id == "deepgram":
            native._build_transcript_event(_deepgram_result("", from_finalize=True))
        else:
            native._on_committed_with_timestamps(CommittedTranscriptWithTimestampsPayload(text=""))
        terminal = await asyncio.wait_for(anext(stream), 1)
        assert isinstance(terminal, STTProviderTurnTerminal)
        await harness.peer_owner.handle_provider_turn_terminal(
            AudioSegmentTerminalReceipt(identity.segment, "final", snapshot, 11), terminal
        )
        await finish(harness)
        assert len(trace.items) == len(sources)
        assert len(translation.calls) == len(sources)
    finally:
        await wrapper.close()
        await close(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["deepgram", "elevenlabs_scribe"])
async def test_sdk_clock_collision_preserves_prior_context_without_publishing_rejected_source(
    provider_id, monkeypatch
):
    from elevenlabs.types import CommittedTranscriptWithTimestampsPayload, TranscriptionWord

    from puripuly_heart.core.audio.ownership import AudioSegmentTerminalReceipt
    from puripuly_heart.core.stt.scoped_normalizer import STTScopedTurnNormalizer
    from puripuly_heart.core.stt.streaming_speaker import StreamingSpeakerSession
    from tests.core.test_source_context_contract import (
        ControlledProvider,
        close,
        finish,
        harness_for,
        source_fixture,
    )
    from tests.core.test_streaming_speaker import evidence, next_text, send, span

    async def write(_session, _payload):
        return None

    monkeypatch.setattr(_DeepgramSDKSession, "_write_payload", write)
    snapshot, identity = source_fixture()
    request = STTProviderTurnRequest(identity, snapshot.settings)
    if provider_id == "deepgram":
        native = _deepgram_session(epoch=identity.provider_epoch_id)
    else:
        native, _ = _scribe_session(epoch=identity.provider_epoch_id, model="scribe_v2_realtime")

    def emit(text, start, end):
        if provider_id == "deepgram":
            native._build_transcript_event(
                _deepgram_result(text, start=start, duration=end - start)
            )
        else:
            native._on_committed_with_timestamps(
                CommittedTranscriptWithTimestampsPayload(
                    text=text,
                    words=[TranscriptionWord(text=text, type="word", start=start, end=end)],
                )
            )

    wrapper = StreamingSpeakerSession(native)
    translation = ControlledProvider()
    harness, trace = harness_for(translation)
    normalizer = STTScopedTurnNormalizer(identity)
    try:
        await wrapper.begin_turn(request)
        await send(wrapper, request, 0, 40000, 1)
        await wrapper.observe_speaker_evidence((evidence(0, 40000, 0),))
        for text, start, end in (("First.", 0.1, 0.3), ("Second.", 2.0, 2.2)):
            emit(text, start, end)
            outgoing = normalizer.apply_update(await next_text(wrapper.turn_events()))
            await harness.peer_owner.handle_provider_turn_update(snapshot, outgoing)
            await finish(harness)
        await wrapper.seal_turn(
            identity,
            sealed_content_ranges=(span(0, 40000),),
            seal_reason="silence",
            observed_trailing_silence_ms=200,
        )
        emit("Correction.", 0.1, 0.3)
        terminal = normalizer.apply_terminal(
            await asyncio.wait_for(anext(wrapper.turn_events()), 1)
        )
        assert terminal.text == "First.Second."
        assert terminal.epoch_disposition == "retire"
        assert terminal.provenance[-1].unresolved_source_text == "Correction."
        assert len(terminal.included_contributions) == 2
        await harness.peer_owner.handle_provider_turn_terminal(
            AudioSegmentTerminalReceipt(identity.segment, "failed", snapshot, 11), terminal
        )
        await finish(harness)
        assert [item.source_child.raw_text for item in trace.items] == ["First.", "Second."]
        assert len(translation.calls) == 2
        assert native._event_projection.active_identity is None
        assert native._event_projection.retired
    finally:
        await wrapper.close()
        await close(harness)
