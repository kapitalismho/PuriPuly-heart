from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import (
    AudioSegmentIdentity,
    AudioSegmentSettingsSnapshot,
    PeerAudioSegmentLedger,
)
from puripuly_heart.core.runtime.local_asr_provider_runtime import LocalASRProviderRuntimeOwner
from puripuly_heart.core.stt.backend import (
    STTContributionConsumptionLedger,
    STTNativeProvenance,
    STTProviderEpochEnded,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTSessionProjection,
)
from puripuly_heart.core.stt.scoped_engine import (
    ScopedRecognitionEngine,
    STTRecognitionWatchdogs,
    STTRetentionProfile,
)
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart
from puripuly_heart.providers.stt.deepgram import _CLOSE_STREAM, _FINALIZE, _DeepgramSDKSession
from puripuly_heart.providers.stt.elevenlabs_scribe import _ElevenLabsScribeSession
from puripuly_heart.providers.stt.gemini_transcribe import _GeminiTranscribeLiveSession
from puripuly_heart.providers.stt.soniox import _SonioxSession


def _request(provider: str, order: int = 1) -> STTProviderTurnRequest:
    settings = AudioSegmentSettingsSnapshot(
        provider_id=provider,
        provider_signature=(provider,),
        runtime_signature=(provider,),
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
            provider_epoch_id=f"epoch-{order}",
            provider_turn_id=f"turn-{order}",
        ),
        settings=settings,
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


async def _wait(predicate) -> None:
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition not reached")


def _deepgram_result(text: str, *, from_finalize: bool = False, is_final: bool = True):
    from deepgram.extensions.types.sockets import ListenV1ResultsEvent
    from deepgram.extensions.types.sockets.listen_v1_results_event import (
        ListenV1Alternative,
        ListenV1Channel,
        ListenV1ModelInfo,
        ListenV1ResultsMetadata,
    )

    return ListenV1ResultsEvent(
        type="Results",
        channel_index=[0, 1],
        duration=0.1,
        start=0.0,
        is_final=is_final,
        speech_final=False,
        channel=ListenV1Channel(
            alternatives=[ListenV1Alternative(transcript=text, confidence=1.0, words=[])]
        ),
        metadata=ListenV1ResultsMetadata(
            request_id="session-request",
            model_info=ListenV1ModelInfo(name="nova-3", version="1", arch="nova"),
            model_uuid="model",
        ),
        from_finalize=from_finalize,
    )


def _deepgram_session(
    drain_timeout_s: float = 10.0,
    *,
    order: int = 1,
) -> _DeepgramSDKSession:
    session = _DeepgramSDKSession(
        api_key="k",
        model="nova-3",
        language="en",
        sample_rate_hz=16000,
        connect_timeout_s=5.0,
        keyterms=[],
        drain_timeout_s=drain_timeout_s,
        projection=STTSessionProjection(
            mode="scoped",
            provider_epoch_id=f"epoch-{order}",
        ),
    )
    session._loop = asyncio.get_running_loop()
    return session


@pytest.mark.asyncio
async def test_deepgram_actual_result_shape_retires_unkeyed_epoch_and_isolates_next(
    monkeypatch,
) -> None:
    writes: list[tuple[_DeepgramSDKSession, object]] = []
    gate = asyncio.Event()
    gate.set()

    async def write(session, payload) -> None:
        writes.append((session, payload))
        await gate.wait()

    monkeypatch.setattr(_DeepgramSDKSession, "_write_thread_payload", write)
    session_a = _deepgram_session()
    assert session_a._audio_q.maxsize == 258
    session_a._build_transcript_event(_deepgram_result("unsolicited", from_finalize=True))
    assert session_a._event_projection.scoped_event_depth == 0
    request_a = _request("deepgram")
    await session_a.begin_turn(request_a)
    gate.clear()
    send = asyncio.create_task(
        session_a.send_turn_audio(
            request_a.identity,
            b"pcm",
            payload_sequence=1,
            source_ranges=_span(),
            context_only=False,
        )
    )
    await _wait(lambda: any(payload == b"pcm" for _, payload in writes))
    assert not send.done()
    gate.set()
    await send
    await session_a.seal_turn(
        request_a.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=224,
    )
    session_a._build_transcript_event(_deepgram_result("same "))
    session_a._build_transcript_event(_deepgram_result("text"))
    session_a._build_transcript_event(_deepgram_result("", from_finalize=True))
    await asyncio.sleep(0)
    assert (await _next(session_a)).text == "same "
    assert (await _next(session_a)).text == "text"
    terminal_a = await _next(session_a)
    assert (terminal_a.outcome, terminal_a.text, terminal_a.epoch_disposition) == (
        "final",
        "same text",
        "retire",
    )
    session_a._build_transcript_event(_deepgram_result("late-a"))
    session_a._build_transcript_event(_deepgram_result("late-a", from_finalize=True))
    await asyncio.sleep(0)
    assert session_a._event_projection.scoped_event_depth == 0
    with pytest.raises(RuntimeError, match="epoch is retired"):
        await session_a.begin_turn(_request("deepgram", 2))

    session_b = _deepgram_session(order=2)
    request_b = _request("deepgram", 2)
    await session_b.begin_turn(request_b)
    await session_b.seal_turn(
        request_b.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=0,
    )
    session_b._build_transcript_event(_deepgram_result("same text", from_finalize=True))
    await asyncio.sleep(0)
    assert (await _next(session_b)).text == "same text"
    terminal_b = await _next(session_b)
    assert (terminal_b.text, terminal_b.epoch_disposition) == ("same text", "retire")
    await session_a.close()
    await session_b.close()


@pytest.mark.asyncio
async def test_deepgram_empty_error_abort_and_two_drain_path(monkeypatch) -> None:
    writes: list[object] = []

    async def write(_session, payload) -> None:
        writes.append(payload)

    monkeypatch.setattr(_DeepgramSDKSession, "_write_thread_payload", write)
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
        "retire",
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
    with pytest.raises(RuntimeError, match="epoch is retired"):
        await aborted.begin_turn(_request("deepgram", 2))
    await aborted.close()
    assert writes.count(_FINALIZE) == 2


class _FakeGeminiLive:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.send_gate = asyncio.Event()
        self.send_gate.set()
        self.queue: asyncio.Queue[object] = asyncio.Queue()
        self.closed = False

    async def send_realtime_input(self, **kwargs) -> None:
        self.sent.append(kwargs)
        await self.send_gate.wait()

    async def receive(self):
        while True:
            item = await self.queue.get()
            if isinstance(item, BaseException):
                raise item
            yield item

    async def close(self) -> None:
        self.closed = True

    def push(self, item: object) -> None:
        self.queue.put_nowait(item)


_MISSING = object()


def _gemini_message(*, final: object = _MISSING, interim: object = _MISSING, ack: bool = False):
    from google.genai import types

    content = None
    if final is not _MISSING or interim is not _MISSING:
        content = types.LiveServerContent(
            interim_input_transcription=(
                types.Transcription(text=str(interim)) if interim is not _MISSING else None
            ),
            input_transcription=(
                types.Transcription(text=str(final)) if final is not _MISSING else None
            ),
        )
    activity = None
    if ack:
        activity = types.VoiceActivity(voice_activity_type=types.VoiceActivityType.ACTIVITY_END)
    return types.LiveServerMessage(server_content=content, voice_activity=activity)


async def _gemini_session(timeout: float = 0.05, *, order: int = 1):
    live = _FakeGeminiLive()
    session = _GeminiTranscribeLiveSession(
        api_key="k",
        language_codes=[],
        custom_vocabulary=[],
        model="model",
        sample_rate_hz=16000,
        connect_timeout_s=10.0,
        finalize_timeout_s=timeout,
        projection=STTSessionProjection(
            mode="scoped",
            provider_epoch_id=f"epoch-{order}",
        ),
    )
    session._live_session = live
    session._send_task = asyncio.create_task(session._send_loop())
    session._recv_task = asyncio.create_task(session._recv_loop())
    return session, live


@pytest.mark.asyncio
async def test_gemini_actual_message_shape_retires_unkeyed_epoch_and_isolates_next() -> None:
    from google.genai import types

    assert "id" not in types.LiveServerMessage.model_fields
    assert "event_id" not in types.LiveServerMessage.model_fields
    session_a, live_a = await _gemini_session()
    assert session_a._send_queue.maxsize == 258
    live_a.push(_gemini_message(final="unsolicited"))
    await asyncio.sleep(0)
    assert session_a._event_projection.scoped_event_depth == 0
    request_a = _request("gemini_transcribe")
    await session_a.begin_turn(request_a)
    live_a.send_gate.clear()
    send = asyncio.create_task(
        session_a.send_turn_audio(
            request_a.identity,
            b"pcm",
            payload_sequence=1,
            source_ranges=_span(),
            context_only=False,
        )
    )
    await _wait(lambda: any("audio" in item for item in live_a.sent))
    assert not send.done()
    live_a.send_gate.set()
    await send
    await session_a.seal_turn(
        request_a.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=224,
    )
    live_a.push(_gemini_message(ack=True))
    live_a.push(_gemini_message(final="same"))
    update_a = await _next(session_a)
    terminal_a = await _next(session_a)
    assert update_a.text == "same"
    assert (terminal_a.text, terminal_a.epoch_disposition) == ("same", "retire")
    live_a.push(_gemini_message(final="late-a"))
    live_a.push(_gemini_message(ack=True))
    await asyncio.sleep(0)
    assert session_a._event_projection.scoped_event_depth == 0
    with pytest.raises(RuntimeError, match="epoch is retired"):
        await session_a.begin_turn(_request("gemini_transcribe", 2))

    session_b, live_b = await _gemini_session(order=2)
    request_b = _request("gemini_transcribe", 2)
    await session_b.begin_turn(request_b)
    await session_b.seal_turn(
        request_b.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=0,
    )
    live_b.push(_gemini_message(final="same"))
    live_b.push(_gemini_message(final="duplicate"))
    live_b.push(_gemini_message(ack=True))
    assert (await _next(session_b)).text == "same"
    terminal_b = await _next(session_b)
    assert (terminal_b.text, terminal_b.epoch_disposition) == ("same", "retire")
    await session_a.close()
    await session_b.close()


@pytest.mark.asyncio
async def test_gemini_empty_timeout_error_and_abort_receipts() -> None:
    empty, empty_live = await _gemini_session()
    request = _request("gemini_transcribe")
    await empty.begin_turn(request)
    await empty.seal_turn(
        request.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=0,
    )
    empty_live.push(_gemini_message(final=""))
    empty_live.push(_gemini_message(ack=True))
    await _next(empty)
    terminal = await _next(empty)
    assert (terminal.outcome, terminal.text_authority, terminal.epoch_disposition) == (
        "empty",
        "authoritative",
        "retire",
    )
    await empty.close()

    timeout, timeout_live = await _gemini_session(0.01)
    timeout_request = _request("gemini_transcribe")
    await timeout.begin_turn(timeout_request)
    await timeout.seal_turn(
        timeout_request.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=0,
    )
    timeout_live.push(_gemini_message(final="kept"))
    await _next(timeout)
    timed = await _next(timeout)
    assert (timed.text, timed.failure_reason, timed.epoch_disposition) == (
        "kept",
        "gemini_finalize_timeout",
        "retire",
    )
    await timeout.close()

    errored, error_live = await _gemini_session()
    error_request = _request("gemini_transcribe")
    await errored.begin_turn(error_request)
    error_live.push(RuntimeError("receive failed"))
    failed = await _next(errored)
    ended = await _next(errored)
    assert failed.failure_reason == "gemini_receive_failed"
    assert isinstance(ended, STTProviderEpochEnded)
    await errored.close()

    aborted, _ = await _gemini_session()
    abort_request = _request("gemini_transcribe")
    await aborted.begin_turn(abort_request)
    await aborted.abort_turn(abort_request.identity, reason="cancelled")
    aborted_terminal = await _next(aborted)
    assert (aborted_terminal.outcome, aborted_terminal.failure_reason) == (
        "cancelled",
        "cancelled",
    )
    assert isinstance(await _next(aborted), STTProviderEpochEnded)
    with pytest.raises(RuntimeError, match="session is closed"):
        await aborted.begin_turn(_request("gemini_transcribe", 2))
    await aborted.close()


class _FakeSonioxWebSocket:
    def __init__(self) -> None:
        self.sent: list[object] = []
        self.send_gate = asyncio.Event()
        self.send_gate.set()
        self.queue: asyncio.Queue[object] = asyncio.Queue()
        self.closed = False

    async def send(self, payload: object) -> None:
        self.sent.append(payload)
        await self.send_gate.wait()

    async def recv(self):
        return await self.queue.get()

    async def close(self) -> None:
        self.closed = True

    def push(self, payload: object) -> None:
        self.queue.put_nowait(payload)


def _soniox_session(*, order: int = 1):
    ws = _FakeSonioxWebSocket()
    session = _SonioxSession(
        api_key="k",
        model="model",
        endpoint="endpoint",
        sample_rate_hz=16000,
        language_hints=["en"],
        context_terms=[],
        keepalive_interval_s=100.0,
        trailing_silence_ms=100,
        connect_timeout_s=5.0,
        enable_language_identification=True,
        projection=STTSessionProjection(
            mode="scoped",
            provider_epoch_id=f"epoch-{order}",
        ),
    )
    session._ws = ws
    session._send_task = asyncio.create_task(session._send_loop())
    session._recv_task = asyncio.create_task(session._recv_loop())
    return session, ws


@pytest.mark.asyncio
async def test_soniox_documented_unkeyed_tokens_retire_epoch_and_isolate_next() -> None:
    session_a, ws_a = _soniox_session()
    assert session_a._audio_q.maxsize == 258
    ws_a.push(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))
    await asyncio.sleep(0)
    assert session_a._event_projection.scoped_event_depth == 0
    request_a = _request("soniox")
    await session_a.begin_turn(request_a)
    ws_a.send_gate.clear()
    send = asyncio.create_task(
        session_a.send_turn_audio(
            request_a.identity,
            b"pcm",
            payload_sequence=1,
            source_ranges=_span(),
            context_only=False,
        )
    )
    await _wait(lambda: ws_a.sent == [b"pcm"])
    assert not send.done()
    ws_a.send_gate.set()
    await send
    await session_a.seal_turn(
        request_a.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=224,
    )
    ws_a.push(
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
    assert (await _next(session_a)).text == "same "
    assert (await _next(session_a)).text == "text"
    terminal_a = await _next(session_a)
    assert (terminal_a.text, terminal_a.epoch_disposition) == ("same text", "retire")
    ws_a.push(json.dumps({"tokens": [{"text": "late-a", "is_final": True}]}))
    ws_a.push(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))
    await asyncio.sleep(0)
    assert session_a._event_projection.scoped_event_depth == 0
    with pytest.raises(RuntimeError, match="epoch is retired"):
        await session_a.begin_turn(_request("soniox", 2))

    session_b, ws_b = _soniox_session(order=2)
    request_b = _request("soniox", 2)
    await session_b.begin_turn(request_b)
    await session_b.seal_turn(
        request_b.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=0,
    )
    ws_b.push(
        json.dumps(
            {
                "tokens": [
                    {"text": "same text", "is_final": True, "end_ms": 200, "language": "en"},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )
    assert (await _next(session_b)).text == "same text"
    terminal_b = await _next(session_b)
    assert (terminal_b.text, terminal_b.epoch_disposition) == ("same text", "retire")
    await session_a.close()
    await session_b.close()


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
        "retire",
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
    with pytest.raises(RuntimeError, match="epoch is retired"):
        await aborted.begin_turn(_request("soniox", 2))
    await aborted.close()


class _FakeScribeConnection:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.send_gate = asyncio.Event()
        self.send_gate.set()
        self.commit_gate = asyncio.Event()
        self.commit_gate.set()
        self.commits = 0
        self.closed = False

    async def send(self, payload: dict) -> None:
        self.sent.append(payload)
        await self.send_gate.wait()

    async def commit(self) -> None:
        self.commits += 1
        await self.commit_gate.wait()

    async def close(self) -> None:
        self.closed = True


def _scribe_session(*, order: int = 1):
    connection = _FakeScribeConnection()
    session = _ElevenLabsScribeSession(
        api_key="k",
        language_code="en",
        keyterms=(),
        model="model",
        sample_rate_hz=16000,
        connect_timeout_s=10.0,
        projection=STTSessionProjection(
            mode="scoped",
            provider_epoch_id=f"epoch-{order}",
        ),
    )
    session._connection = connection
    return session, connection


@pytest.mark.asyncio
async def test_scribe_actual_payload_shape_retires_unkeyed_epoch_and_isolates_next() -> None:
    from elevenlabs.types import CommittedTranscriptPayload, PartialTranscriptPayload

    assert "id" not in CommittedTranscriptPayload.model_fields
    assert "transcript_id" not in CommittedTranscriptPayload.model_fields
    assert "commit_id" not in CommittedTranscriptPayload.model_fields
    session_a, connection_a = _scribe_session()
    assert session_a._connection_events.maxsize == 258
    session_a._on_committed(CommittedTranscriptPayload(text="unsolicited"))
    assert session_a._event_projection.scoped_event_depth == 0
    request_a = _request("elevenlabs_scribe")
    await session_a.begin_turn(request_a)
    for index in range(2_000):
        session_a._on_partial(PartialTranscriptPayload(text=f"partial-{index}"))
        session_a._on_committed(CommittedTranscriptPayload(text=f"early-{index}"))
    assert session_a._connection_events.empty()
    assert session_a._event_projection.scoped_event_depth == 1
    connection_a.send_gate.clear()
    send = asyncio.create_task(
        session_a.send_turn_audio(
            request_a.identity,
            b"pcm",
            payload_sequence=1,
            source_ranges=_span(),
            context_only=False,
        )
    )
    await _wait(lambda: bool(connection_a.sent))
    assert not send.done()
    connection_a.send_gate.set()
    await send
    session_a._on_partial(PartialTranscriptPayload(text="same"))
    assert (await _next(session_a)).text == "same"
    connection_a.commit_gate.clear()
    seal = asyncio.create_task(
        session_a.seal_turn(
            request_a.identity,
            sealed_content_ranges=_span(),
            seal_reason="silence",
            observed_trailing_silence_ms=224,
        )
    )
    await _wait(lambda: connection_a.commits == 1)
    assert not seal.done()
    session_a._on_committed(CommittedTranscriptPayload(text="same"))
    terminal_a = await _next(session_a)
    assert (terminal_a.text, terminal_a.epoch_disposition) == ("same", "retire")
    connection_a.commit_gate.set()
    await seal
    session_a._on_committed(CommittedTranscriptPayload(text="late-a"))
    session_a._on_committed(CommittedTranscriptPayload(text="late-a"))
    assert session_a._event_projection.scoped_event_depth == 0
    with pytest.raises(RuntimeError, match="epoch is retired"):
        await session_a.begin_turn(_request("elevenlabs_scribe", 2))

    session_b, _ = _scribe_session(order=2)
    request_b = _request("elevenlabs_scribe", 2)
    await session_b.begin_turn(request_b)
    await session_b.seal_turn(
        request_b.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=0,
    )
    session_b._on_committed(CommittedTranscriptPayload(text="same"))
    terminal_b = await _next(session_b)
    assert (terminal_b.text, terminal_b.epoch_disposition) == ("same", "retire")
    await session_a.close()
    await session_b.close()


@pytest.mark.asyncio
async def test_scribe_empty_error_and_abort_receipts() -> None:
    from elevenlabs.types import CommittedTranscriptPayload

    empty, _ = _scribe_session()
    request = _request("elevenlabs_scribe")
    await empty.begin_turn(request)
    await empty.seal_turn(
        request.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=0,
    )
    empty._on_committed(CommittedTranscriptPayload(text=""))
    terminal = await _next(empty)
    assert (terminal.outcome, terminal.text_authority, terminal.epoch_disposition) == (
        "empty",
        "authoritative",
        "retire",
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
    with pytest.raises(RuntimeError, match="epoch is retired"):
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

    async def close(self) -> None:
        return None


class _UnusedProviderFactory:
    async def create(self, request, *, gpu_runtime, on_terminal_failure=None):
        raise AssertionError((request, gpu_runtime, on_terminal_failure))


def _owned_deepgram_events(
    ledger: PeerAudioSegmentLedger,
    *,
    start_sample: int,
) -> tuple[object, list[object], object]:
    segment_id = uuid4()

    def capture(sequence: int, start: int, end: int) -> AudioCaptureSpan:
        return AudioCaptureSpan(
            capture_epoch=1,
            callback_sequence=sequence,
            source_sample_rate_hz=16000,
            source_start_sample=start,
            source_end_sample=end,
            source_start_monotonic_s=start / 16000,
            source_end_monotonic_s=end / 16000,
            normalized_sample_rate_hz=16000,
            normalized_start_sample=start,
            normalized_end_sample=end,
        )

    first_range = capture(1, start_sample, start_sample + 4)
    start = ledger.observe_vad_event(
        SpeechStart(
            segment_id,
            np.empty((0,), dtype=np.float32),
            np.ones(4, dtype=np.float32),
            chunk_capture=(first_range,),
        ),
        now_monotonic_s=start_sample / 16000,
    )
    chunks: list[object] = []
    for index in range(1, 21):
        chunk_start = start_sample + index * 4
        chunks.append(
            ledger.observe_vad_event(
                SpeechChunk(
                    segment_id,
                    np.ones(4, dtype=np.float32),
                    chunk_capture=(capture(index + 1, chunk_start, chunk_start + 4),),
                ),
                now_monotonic_s=(chunk_start + 4) / 16000,
            )
        )
    end = ledger.observe_vad_event(
        SpeechEnd(segment_id, trailing_silence_ms=800, reason="silence"),
        now_monotonic_s=(start_sample + 84) / 16000,
    )
    return start, chunks, end


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
        session._loop = asyncio.get_running_loop()
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

    monkeypatch.setattr(_DeepgramSDKSession, "_write_thread_payload", write)
    peer_sessions: list[_DeepgramSDKSession] = []
    self_sessions: list[_DeepgramSDKSession] = []
    peer_events: list[object] = []
    self_events: list[object] = []
    consumption = STTContributionConsumptionLedger()
    consumed: list[str] = []

    async def self_consumer(event: object) -> None:
        self_events.append(event)
        if isinstance(event, STTProviderTurnUpdate) and not consumed:
            consumed.append(consumption.consume(event))
        elif isinstance(event, STTProviderTurnTerminal):
            consumed.append(consumption.consume(event))

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
        session._build_transcript_event(_deepgram_result("B"))
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

    monkeypatch.setattr(_DeepgramSDKSession, "_write_thread_payload", write)
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

    monkeypatch.setattr(_DeepgramSDKSession, "_write_thread_payload", write)
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
    from elevenlabs.types import CommittedTranscriptPayload

    async def deepgram_write(_session, _payload) -> None:
        return None

    monkeypatch.setattr(_DeepgramSDKSession, "_write_thread_payload", deepgram_write)
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
                b"pcm",
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
            boundary.push(_gemini_message(ack=True))
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
            session._on_committed(CommittedTranscriptPayload(text=text))

    terminals: list[STTProviderTurnTerminal] = []
    for session in sessions:
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
    start, chunks, end = _owned_deepgram_events(ledger, start_sample=4000)
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
    assert [event.text for event in updates] == ["same", "same 世界"]
    assert [
        (item.contribution_id, item.text_start, item.text_end)
        for item in terminal.included_contributions
    ] == [
        (f"{terminal.identity.provider_turn_id}:1", 0, 4),
        (f"{terminal.identity.provider_turn_id}:2", 4, 7),
    ]
    assert consumption.consume(updates[0]) == "same"
    assert consumption.consume(terminal) == " 世界"
    assert terminal.text == "same 世界"
    assert "".join(run.text for run in terminal.final_language_runs) == terminal.text
    await engine.close()
