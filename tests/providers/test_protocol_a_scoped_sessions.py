from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import AudioSegmentIdentity, AudioSegmentSettingsSnapshot
from puripuly_heart.core.stt.backend import (
    STTProviderEpochEnded,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnUpdate,
)
from puripuly_heart.providers.stt.deepgram import (
    _CLOSE_STREAM,
    _FINALIZE,
    _DeepgramSDKSession,
)
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
            provider_epoch_id="epoch",
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


def _deepgram_result(
    text: str,
    *,
    event_id: str,
    from_finalize: bool = False,
):
    return SimpleNamespace(
        channel=SimpleNamespace(alternatives=[SimpleNamespace(transcript=text)]),
        is_final=True,
        speech_final=False,
        event_id=event_id,
        from_finalize=from_finalize,
        metadata=SimpleNamespace(request_id="request"),
    )


@pytest.mark.asyncio
async def test_deepgram_scoped_fragments_ack_and_missing_ack_two_drain_path(monkeypatch) -> None:
    session = _DeepgramSDKSession(
        api_key="k",
        model="nova-3",
        language="en",
        sample_rate_hz=16000,
        connect_timeout_s=5.0,
        keyterms=[],
        drain_timeout_s=0.01,
    )
    session._loop = asyncio.get_running_loop()
    writes: list[object] = []
    gate = asyncio.Event()

    async def write(_session, payload) -> None:
        writes.append(payload)
        await gate.wait()

    monkeypatch.setattr(_DeepgramSDKSession, "_write_thread_payload", write)
    request = _request("deepgram")
    await session.begin_turn(request)
    send_task = asyncio.create_task(
        session.send_turn_audio(
            request.identity,
            b"pcm",
            payload_sequence=1,
            source_ranges=_span(),
            context_only=False,
        )
    )
    await _wait(lambda: writes == [b"pcm"])
    assert not send_task.done()
    gate.set()
    await send_task
    await session.seal_turn(
        request.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=224,
    )
    session._build_transcript_event(_deepgram_result("one ", event_id="a"))
    session._build_transcript_event(_deepgram_result("two", event_id="b"))
    session._build_transcript_event(_deepgram_result("", event_id="ack", from_finalize=True))
    await asyncio.sleep(0)
    first = await _next(session)
    second = await _next(session)
    terminal = await _next(session)
    assert [first.text, second.text] == ["one ", "two"]
    assert terminal.outcome == "final"
    assert terminal.text == "one two"
    assert terminal.epoch_disposition == "reuse"

    request_b = _request("deepgram", 2)
    await session.begin_turn(request_b)
    await session.seal_turn(
        request_b.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=224,
    )
    session._build_transcript_event(_deepgram_result("late", event_id="a"))
    await asyncio.sleep(0.03)
    terminal_b = await _next(session)
    assert terminal_b.outcome == "failed"
    assert terminal_b.failure_reason == "deepgram_finalize_ack_missing"
    assert writes.count(_FINALIZE) == 2
    assert writes.count(_CLOSE_STREAM) == 1
    session._build_transcript_event(
        _deepgram_result("unsolicited", event_id="unsolicited", from_finalize=True)
    )
    await asyncio.sleep(0)
    assert session._scoped_events.depth == 0
    await session.close()


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


def _gemini_message(
    *,
    event_id: str,
    final: str | None = None,
    interim: str | None = None,
    ack: bool = False,
):
    content = None
    if final is not None or interim is not None:
        content = SimpleNamespace(
            interim_input_transcription=(
                SimpleNamespace(text=interim) if interim is not None else None
            ),
            input_transcription=(SimpleNamespace(text=final) if final is not None else None),
        )
    activity = None
    if ack:
        activity = SimpleNamespace(voice_activity_type=SimpleNamespace(value="ACTIVITY_END"))
    return SimpleNamespace(id=event_id, server_content=content, voice_activity=activity)


async def _gemini_session(timeout: float = 0.05):
    live = _FakeGeminiLive()
    session = _GeminiTranscribeLiveSession(
        api_key="k",
        language_codes=[],
        custom_vocabulary=[],
        model="model",
        sample_rate_hz=16000,
        connect_timeout_s=10.0,
        finalize_timeout_s=timeout,
    )
    session._live_session = live
    session._send_task = asyncio.create_task(session._send_loop())
    session._recv_task = asyncio.create_task(session._recv_loop())
    return session, live


@pytest.mark.asyncio
async def test_gemini_scoped_requires_both_barriers_in_either_order_and_rejects_late_a() -> None:
    session, live = await _gemini_session()
    live.push(_gemini_message(event_id="unsolicited", final="ignored"))
    await asyncio.sleep(0)
    assert session._scoped_events.depth == 0
    request = _request("gemini_transcribe")
    await session.begin_turn(request)
    live.send_gate.clear()
    send_task = asyncio.create_task(
        session.send_turn_audio(
            request.identity,
            b"pcm",
            payload_sequence=1,
            source_ranges=_span(),
            context_only=False,
        )
    )
    await _wait(lambda: any("audio" in item for item in live.sent))
    assert not send_task.done()
    live.send_gate.set()
    await send_task
    await session.seal_turn(
        request.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=224,
    )
    live.push(_gemini_message(event_id="ack-a", ack=True))
    await asyncio.sleep(0)
    assert session._scoped_events.depth == 0
    live.push(_gemini_message(event_id="final-a", final="one"))
    update = await _next(session)
    terminal = await _next(session)
    assert update.stability == "stable"
    assert terminal.text == "one"
    assert terminal.outcome == "final"

    request_b = _request("gemini_transcribe", 2)
    await session.begin_turn(request_b)
    await session.seal_turn(
        request_b.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=224,
    )
    live.push(_gemini_message(event_id="final-a", final="late-a"))
    live.push(_gemini_message(event_id="final-b", final="two"))
    await asyncio.sleep(0)
    assert session._scoped_events.depth == 1
    live.push(_gemini_message(event_id="ack-b", ack=True))
    update_b = await _next(session)
    terminal_b = await _next(session)
    assert update_b.text == "two"
    assert terminal_b.text == "two"
    request_c = _request("gemini_transcribe", 3)
    await session.begin_turn(request_c)
    await session.seal_turn(
        request_c.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=0,
    )
    live.push(_gemini_message(event_id="empty-c", final=""))
    live.push(_gemini_message(event_id="ack-c", ack=True))
    await _next(session)
    terminal_c = await _next(session)
    assert terminal_c.outcome == "empty"
    assert terminal_c.text_authority == "authoritative"
    await session.close()


@pytest.mark.asyncio
async def test_gemini_scoped_missing_barrier_timeout_receipts() -> None:
    authoritative, live_a = await _gemini_session(timeout=0.01)
    request_a = _request("gemini_transcribe")
    await authoritative.begin_turn(request_a)
    await authoritative.seal_turn(
        request_a.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=0,
    )
    live_a.push(_gemini_message(event_id="final", final="kept"))
    await _next(authoritative)
    terminal_a = await _next(authoritative)
    assert terminal_a.text == "kept"
    assert terminal_a.text_authority == "authoritative"
    assert terminal_a.epoch_disposition == "retire"
    await authoritative.close()

    fallback, live_b = await _gemini_session(timeout=0.01)
    request_b = _request("gemini_transcribe")
    await fallback.begin_turn(request_b)
    live_b.push(_gemini_message(event_id="interim", interim="fallback"))
    await _next(fallback)
    await fallback.seal_turn(
        request_b.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=0,
    )
    live_b.push(_gemini_message(event_id="ack", ack=True))
    terminal_b = await _next(fallback)
    assert terminal_b.outcome == "degraded"
    assert terminal_b.text == "fallback"
    assert terminal_b.epoch_disposition == "retire"
    await fallback.close()


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


@pytest.mark.asyncio
async def test_soniox_scoped_fin_only_barrier_order_runs_and_actual_write_completion() -> None:
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
    )
    session._ws = ws
    session._send_task = asyncio.create_task(session._send_loop())
    session._recv_task = asyncio.create_task(session._recv_loop())
    request = _request("soniox")
    await session.begin_turn(request)
    ws.send_gate.clear()
    write_task = asyncio.create_task(
        session.send_turn_audio(
            request.identity,
            b"pcm",
            payload_sequence=1,
            source_ranges=_span(),
            context_only=False,
        )
    )
    await _wait(lambda: ws.sent == [b"pcm"])
    assert not write_task.done()
    ws.send_gate.set()
    await write_task
    await session.seal_turn(
        request.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=224,
    )
    ws.push(
        json.dumps(
            {
                "request_id": "r",
                "tokens": [
                    {"id": "a", "text": " 안", "language": "ko", "is_final": True},
                    {"id": "b", "text": "녕 ", "language": "ko", "is_final": True},
                    {"text": "<end>", "is_final": True},
                ],
            }
        )
    )
    first = await _next(session)
    second = await _next(session)
    assert isinstance(first, STTProviderTurnUpdate)
    assert isinstance(second, STTProviderTurnUpdate)
    assert session._scoped_events.depth == 0
    ws.push(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))
    terminal = await _next(session)
    assert terminal.text == " 안녕 "
    assert [(run.text, run.language) for run in terminal.final_language_runs] == [(" 안녕 ", "ko")]
    assert terminal.outcome == "final"
    await session.close()


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


@pytest.mark.asyncio
async def test_scribe_scoped_commit_only_terminal_duplicate_partial_and_empty() -> None:
    connection = _FakeScribeConnection()
    session = _ElevenLabsScribeSession(
        api_key="k",
        language_code="en",
        keyterms=(),
        model="model",
        sample_rate_hz=16000,
        connect_timeout_s=10.0,
    )
    session._connection = connection
    session._on_committed(
        {"message_type": "committed_transcript", "id": "unsolicited", "text": "ignored"}
    )
    assert session._scoped_events.depth == 0
    request = _request("elevenlabs_scribe")
    await session.begin_turn(request)
    connection.send_gate.clear()
    write_task = asyncio.create_task(
        session.send_turn_audio(
            request.identity,
            b"pcm",
            payload_sequence=1,
            source_ranges=_span(),
            context_only=False,
        )
    )
    await _wait(lambda: bool(connection.sent))
    assert not write_task.done()
    connection.send_gate.set()
    await write_task
    session._on_partial({"message_type": "partial_transcript", "id": "p", "text": "one"})
    session._on_partial(
        {"message_type": "final_transcript_with_timestamps", "id": "p", "text": "one"}
    )
    update = await _next(session)
    assert update.text == "one"
    assert session._scoped_events.depth == 0
    connection.commit_gate.clear()
    seal_task = asyncio.create_task(
        session.seal_turn(
            request.identity,
            sealed_content_ranges=_span(),
            seal_reason="silence",
            observed_trailing_silence_ms=224,
        )
    )
    await _wait(lambda: connection.commits == 1)
    assert not seal_task.done()
    session._on_committed({"message_type": "committed_transcript", "id": "c", "text": "one"})
    terminal = await _next(session)
    assert terminal.outcome == "final"
    assert terminal.text == "one"
    connection.commit_gate.set()
    await seal_task

    request_b = _request("elevenlabs_scribe", 2)
    await session.begin_turn(request_b)
    await session.seal_turn(
        request_b.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=0,
    )
    session._on_committed({"message_type": "committed_transcript", "id": "c", "text": "late-one"})
    assert session._scoped_events.depth == 0
    session._on_committed({"message_type": "committed_transcript", "id": "empty", "text": ""})
    terminal_b = await _next(session)
    assert terminal_b.outcome == "empty"
    assert terminal_b.text_authority == "authoritative"

    request_c = _request("elevenlabs_scribe", 3)
    await session.begin_turn(request_c)
    session._on_error_event({"message_type": "quota_exceeded"})
    failure = await _next(session)
    ended = await _next(session)
    assert failure.outcome == "failed"
    assert isinstance(ended, STTProviderEpochEnded)
    await session.close()


@pytest.mark.asyncio
async def test_deepgram_scoped_speech_final_is_not_terminal_and_empty_ack_is(
    monkeypatch,
) -> None:
    session = _DeepgramSDKSession(
        api_key="k",
        model="nova-3",
        language="en",
        sample_rate_hz=16000,
        connect_timeout_s=5.0,
        keyterms=[],
        drain_timeout_s=10.0,
    )
    session._loop = asyncio.get_running_loop()

    async def write(_session, payload) -> None:
        _ = payload

    monkeypatch.setattr(_DeepgramSDKSession, "_write_thread_payload", write)
    request = _request("deepgram")
    await session.begin_turn(request)
    await session.seal_turn(
        request.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=0,
    )
    session._build_transcript_event(
        SimpleNamespace(
            channel=SimpleNamespace(alternatives=[SimpleNamespace(transcript="not-boundary")]),
            is_final=False,
            speech_final=True,
            event_id="speech",
            from_finalize=False,
            metadata=SimpleNamespace(request_id="request"),
        )
    )
    await asyncio.sleep(0)
    assert session._scoped_events.depth == 0
    session._build_transcript_event(_deepgram_result("", event_id="empty", from_finalize=True))
    await asyncio.sleep(0)
    terminal = await _next(session)
    assert terminal.outcome == "empty"
    assert terminal.provenance[-1].from_finalize is True
    await session.close()


@pytest.mark.asyncio
async def test_deepgram_scoped_transport_error_retains_stable_text_as_degraded() -> None:
    session = _DeepgramSDKSession(
        api_key="k",
        model="nova-3",
        language="en",
        sample_rate_hz=16000,
        connect_timeout_s=5.0,
        keyterms=[],
        drain_timeout_s=10.0,
    )
    session._loop = asyncio.get_running_loop()
    request = _request("deepgram")
    await session.begin_turn(request)
    session._build_transcript_event(_deepgram_result("kept", event_id="stable"))
    await asyncio.sleep(0)
    await _next(session)
    session._scoped_transport_end(False, "deepgram_transport_error")
    terminal = await _next(session)
    ended = await _next(session)
    assert terminal.outcome == "degraded"
    assert terminal.text == "kept"
    assert terminal.epoch_disposition == "retire"
    assert isinstance(ended, STTProviderEpochEnded)
    await session.close()


@pytest.mark.asyncio
async def test_gemini_scoped_timeout_without_any_text_fails_and_retires() -> None:
    session, _live = await _gemini_session(timeout=0.01)
    request = _request("gemini_transcribe")
    await session.begin_turn(request)
    await session.seal_turn(
        request.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=0,
    )
    terminal = await _next(session)
    assert terminal.outcome == "failed"
    assert terminal.failure_reason == "gemini_finalize_timeout"
    assert terminal.epoch_disposition == "retire"
    await session.close()


@pytest.mark.asyncio
async def test_soniox_scoped_unsolicited_duplicate_empty_and_error_receipts() -> None:
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
    )
    session._ws = ws
    session._send_task = asyncio.create_task(session._send_loop())
    session._handle_message(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))
    assert session._scoped_events.depth == 0

    request = _request("soniox")
    await session.begin_turn(request)
    await session.seal_turn(
        request.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=0,
    )
    duplicate = {"id": "same", "text": "word", "is_final": True}
    session._handle_message(json.dumps({"tokens": [duplicate, duplicate]}))
    update = await _next(session)
    assert update.text == "word"
    assert session._scoped_events.depth == 0
    session._handle_message(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))
    terminal = await _next(session)
    assert terminal.text == "word"

    request_b = _request("soniox", 2)
    await session.begin_turn(request_b)
    await session.seal_turn(
        request_b.identity,
        sealed_content_ranges=_span(),
        seal_reason="silence",
        observed_trailing_silence_ms=0,
    )
    session._handle_message(json.dumps({"tokens": [duplicate]}))
    assert session._scoped_events.depth == 0
    session._handle_message(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))
    empty = await _next(session)
    assert empty.outcome == "empty"

    request_c = _request("soniox", 3)
    await session.begin_turn(request_c)
    session._handle_message(json.dumps({"error": "bad"}))
    failure = await _next(session)
    ended = await _next(session)
    assert failure.outcome == "failed"
    assert isinstance(ended, STTProviderEpochEnded)
    await session.close()


@pytest.mark.asyncio
async def test_scribe_scoped_eof_discards_partial_and_retires_epoch() -> None:
    connection = _FakeScribeConnection()
    session = _ElevenLabsScribeSession(
        api_key="k",
        language_code="en",
        keyterms=(),
        model="model",
        sample_rate_hz=16000,
        connect_timeout_s=10.0,
    )
    session._connection = connection
    request = _request("elevenlabs_scribe")
    await session.begin_turn(request)
    session._on_partial({"message_type": "final_transcript", "id": "partial", "text": "discard"})
    await _next(session)
    session._on_closed({"message_type": "close"})
    terminal = await _next(session)
    ended = await _next(session)
    assert terminal.outcome == "failed"
    assert terminal.text == ""
    assert isinstance(ended, STTProviderEpochEnded)
    await session.close()


@pytest.mark.asyncio
async def test_deepgram_scoped_orderly_eof_without_barrier_fails_and_retires() -> None:
    session = _DeepgramSDKSession(
        api_key="k",
        model="nova-3",
        language="en",
        sample_rate_hz=16000,
        connect_timeout_s=5.0,
        keyterms=[],
        drain_timeout_s=10.0,
    )
    request = _request("deepgram")
    await session.begin_turn(request)
    session._scoped_transport_end(True, "deepgram_eof")
    terminal = await _next(session)
    ended = await _next(session)
    assert terminal.outcome == "failed"
    assert ended.orderly is True
    await session.close()


@pytest.mark.asyncio
async def test_gemini_scoped_receive_error_fails_and_retires() -> None:
    session, live = await _gemini_session()
    request = _request("gemini_transcribe")
    await session.begin_turn(request)
    live.push(RuntimeError("receive failed"))
    terminal = await _next(session)
    ended = await _next(session)
    assert terminal.outcome == "failed"
    assert terminal.failure_reason == "gemini_receive_failed"
    assert isinstance(ended, STTProviderEpochEnded)
    assert ended.orderly is False
    await session.close()


@pytest.mark.asyncio
async def test_gemini_scoped_missing_live_session_eof_fails_and_retires() -> None:
    session, _live = await _gemini_session()
    recv_task = session._recv_task
    assert recv_task is not None
    recv_task.cancel()
    await asyncio.gather(recv_task, return_exceptions=True)
    request = _request("gemini_transcribe")
    await session.begin_turn(request)
    session._live_session = None
    session._recv_task = asyncio.create_task(session._recv_loop())
    terminal = await _next(session)
    ended = await _next(session)
    assert terminal.outcome == "failed"
    assert terminal.failure_reason == "gemini_connection_ended"
    assert isinstance(ended, STTProviderEpochEnded)
    assert ended.orderly is True
    await session.close()


@pytest.mark.asyncio
async def test_soniox_scoped_orderly_eof_without_fin_fails_and_retires() -> None:
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
    )
    session._ws = ws
    session._recv_task = asyncio.create_task(session._recv_loop())
    request = _request("soniox")
    await session.begin_turn(request)
    ws.push(None)
    terminal = await _next(session)
    ended = await _next(session)
    assert terminal.outcome == "failed"
    assert ended.orderly is True
    await session.close()
