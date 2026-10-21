from __future__ import annotations

import asyncio
import logging
from copy import copy
from dataclasses import replace
from uuid import uuid4

import pytest

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import (
    AudioSegmentIdentity,
    AudioSegmentSettingsSnapshot,
)
from puripuly_heart.core.diagnostic_validation import DIAGNOSTIC_SINK_PERSISTED_LOGS
from puripuly_heart.core.diagnostics import DIAGNOSTIC_RECORD_ATTRIBUTE
from puripuly_heart.core.runtime_logging import _DiagnosticRedactionFilter
from puripuly_heart.core.stt.backend import (
    STTProviderEpochEnded,
    STTProviderInputTerminal,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTRecognitionUnit,
    STTSessionProjection,
)
from puripuly_heart.domain.recognition import RecognitionStreamIdentity
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
        drain_timeout_s=timeout,
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


def _go_away(*, final: str | None = None, ack: bool = False, time_left: str | None = "1s"):
    from google.genai import types

    message = _message(final=final, include_final=final is not None, ack=ack)
    message.go_away = types.LiveServerGoAway(time_left=time_left)
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


def _span(start: int, end: int, *, epoch: int = 1) -> tuple[AudioCaptureSpan, ...]:
    return (
        AudioCaptureSpan(
            capture_epoch=epoch,
            callback_sequence=1,
            source_sample_rate_hz=16000,
            source_start_sample=start,
            source_end_sample=end,
            source_start_monotonic_s=start / 16000,
            source_end_monotonic_s=end / 16000,
            normalized_sample_rate_hz=16000,
            normalized_start_sample=start,
            normalized_end_sample=end,
        ),
    )


class _PrivateStatus:
    def __str__(self) -> str:
        raise AssertionError("Provider status must not be rendered")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "ownership"),
    [
        ("receive", "stream"),
        ("receive", "request"),
        ("receive", "none"),
        ("send", "stream"),
        ("send", "request"),
    ],
)
@pytest.mark.parametrize(
    ("code", "status", "safe_code", "safe_status", "kind"),
    [
        (400, "INVALID_ARGUMENT", "400", "INVALID_ARGUMENT", "validation"),
        (503, "UNAVAILABLE", "503", "UNAVAILABLE", "connection_closed"),
        (
            "private-code",
            "UNAVAILABLE private-status api_key=private-key\nprivate-transcript " * 100,
            "none",
            "unclassified",
            "other",
        ),
        (True, ["private-status"], "none", "unclassified", "other"),
        (1006, _PrivateStatus(), "1006", "unclassified", "other"),
        (999999999, None, "none", "none", "other"),
    ],
)
@pytest.mark.parametrize(
    "error_type",
    [RuntimeError, type("private-exception-name" * 100, (RuntimeError,), {})],
    ids=["runtime_error", "external_error"],
)
async def test_transport_failure_preserves_private_safe_metadata_and_retires_consumer(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    ownership: str,
    code: object,
    status: object,
    safe_code: str,
    safe_status: str,
    kind: str,
    error_type: type[RuntimeError],
) -> None:
    error = error_type("private-message api_key=private-key private-transcript private-audio")
    error.code = code
    error.status = status
    session, live = await _session()
    try:
        if ownership == "stream":
            await session.begin_stream(RecognitionStreamIdentity("self", 1, 1, "epoch", ()))
        elif ownership == "request":
            await session.begin_turn(_request(1))
        with caplog.at_level(logging.INFO, logger="puripuly_heart.providers.stt.gemini_transcribe"):
            if operation == "send":

                async def fail_send(**kwargs) -> None:
                    raise error

                monkeypatch.setattr(live, "send_realtime_input", fail_send)
                with pytest.raises(RuntimeError) as raised:
                    await session.send_stream_audio(b"aa", source_ranges=_span(0, 1))
                assert raised.value is error
                assert not session.recognition_source_covers(_span(0, 1))
            else:
                live.push(error)
            if ownership == "request":
                terminal = await _next(session)
                assert isinstance(terminal, STTProviderInputTerminal)
                assert (terminal.outcome, terminal.epoch_disposition) == ("failed", "retire")
            ended = await _next(session)
            assert isinstance(ended, STTProviderEpochEnded)
            assert not ended.orderly
            with pytest.raises(RuntimeError):
                await session.begin_turn(_request(2))
        records = [
            record
            for record in caplog.records
            if record.name == "puripuly_heart.providers.stt.gemini_transcribe"
        ]
        assert records
        redaction = _DiagnosticRedactionFilter(DIAGNOSTIC_SINK_PERSISTED_LOGS)
        fields = []
        for record in records:
            assert record.exc_info is None
            assert record.exc_text is None
            assert record.stack_info is None
            persisted = copy(record)
            assert redaction.filter(persisted)
            message = persisted.getMessage()
            assert "private-" not in message
            assert len(message) < 4096
            diagnostics, _, disposition, stage, _ = getattr(record, DIAGNOSTIC_RECORD_ATTRIBUTE)
            metadata = dict(diagnostics.fields)
            assert metadata["provider"] == "gemini_transcribe"
            assert diagnostics.channel == (
                "self" if ownership == "stream" else ("peer" if ownership == "request" else None)
            )
            assert metadata["provider_epoch_id"] == "epoch"
            assert metadata["exception_class"] == (
                "RuntimeError" if error_type is RuntimeError else "unclassified"
            )
            assert str(metadata.get("api_code", "none")) == safe_code
            assert metadata["api_status"] == safe_status
            assert metadata["message_kind"] == kind
            assert metadata["recognition_cause"] == ended.reason
            assert stage == ("write" if operation == "send" else "receive")
            assert disposition == "failed"
            fields.append(metadata)
        assert len(fields) == 1
        assert ended.failure_diagnostics.diagnostic_id == diagnostics.diagnostic_id
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_stream_admission_delivers_audio_and_native_final_without_local_onset() -> None:
    session, live = await _session()
    stream = RecognitionStreamIdentity("peer", 1, 1, "epoch", ())
    try:
        await session.begin_stream(stream)
        await session.send_stream_audio(b"aabb", source_ranges=_span(10, 12))
        live.push(_message(final="recovered speech", include_final=True))
        unit = await _next(session)
        assert isinstance(unit, STTRecognitionUnit)
        assert unit.identity.stream == stream
        assert unit.text == "recovered speech"
        await session.begin_turn(_request(1))
        await session.send_stream_audio(b"bbcc", source_ranges=_span(11, 13))
        assert [item["audio"]["data"] for item in live.sent if "audio" in item] == [
            b"aabb",
            b"cc",
        ]
        assert session.recognition_source_covers(_span(10, 13))
        assert not session.recognition_source_covers(_span(9, 10))
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("channel", "self"),
        ("activation_generation", 2),
        ("capture_epoch", 2),
        ("provider_epoch_id", "other"),
        ("settings_scope", ("other",)),
    ],
)
async def test_stream_admission_rejects_identity_crossover(field: str, value: object) -> None:
    session, live = await _session()
    stream = RecognitionStreamIdentity("peer", 1, 1, "epoch", ())
    try:
        await session.begin_stream(stream)
        with pytest.raises(RuntimeError, match="stream ownership"):
            await session.begin_stream(replace(stream, **{field: value}))
        await session.send_stream_audio(b"aa", source_ranges=_span(0, 1))
        live.push(_message(final="current", include_final=True))
        unit = await _next(session)
        assert isinstance(unit, STTRecognitionUnit)
        assert unit.identity.stream == stream
        assert unit.text == "current"
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_queued", [False, True])
async def test_cancelled_write_retires_transport_without_sending_queued_audio(
    cancel_queued: bool,
) -> None:
    session, live = await _session()
    writes: list[asyncio.Task[None]] = []
    try:
        await session.begin_turn(_request(1))
        live.send_gate.clear()
        active = asyncio.create_task(session.send_stream_audio(b"aa", source_ranges=_span(0, 1)))
        writes.append(active)
        await _wait_sent(live, "audio")
        pending = asyncio.create_task(session.send_stream_audio(b"bb", source_ranges=_span(1, 2)))
        writes.append(pending)
        await asyncio.sleep(0)
        cancelled = pending if cancel_queued else active
        remaining = active if cancel_queued else pending
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        with pytest.raises(RuntimeError, match="writer"):
            await remaining
        terminal = await _next(session)
        assert isinstance(terminal, STTProviderInputTerminal)
        assert (terminal.outcome, terminal.epoch_disposition) == ("failed", "retire")
        assert terminal.failure_reason == "gemini_write_cancelled"
        assert isinstance(await _next(session), STTProviderEpochEnded)
        with pytest.raises(RuntimeError, match="closed"):
            await session.send_stream_audio(b"cc", source_ranges=_span(2, 3))
        live.send_gate.set()
        await session.close()
        assert live.closed
        assert [item["audio"]["data"] for item in live.sent if "audio" in item] == [b"aa"]
        assert not session.recognition_source_covers(_span(0, 1))
    finally:
        live.send_gate.set()
        await session.close()
        await asyncio.gather(*writes, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pcm", "ranges", "reason"),
    [
        (b"cc", _span(2, 3), "discontinuity"),
        (b"bb", _span(1, 2, epoch=2), "epoch"),
        (b"bbcc", _span(1, 2), "coverage"),
    ],
)
async def test_context_audio_preserves_source_validation(
    pcm: bytes, ranges: tuple[AudioCaptureSpan, ...], reason: str
) -> None:
    session, live = await _session()
    request = _request(1)
    try:
        await session.begin_turn(request)
        await session.send_stream_audio(b"aa", source_ranges=_span(0, 1))
        with pytest.raises(ValueError, match=reason):
            await session.send_turn_audio(
                request.identity,
                pcm,
                payload_sequence=1,
                source_ranges=ranges,
                context_only=True,
            )
        terminal = await _next(session)
        assert isinstance(terminal, STTProviderInputTerminal)
        assert (terminal.outcome, terminal.epoch_disposition) == ("failed", "retire")
        assert isinstance(await _next(session), STTProviderEpochEnded)
        assert [item["audio"]["data"] for item in live.sent if "audio" in item] == [b"aa"]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_seal_submits_input_without_waiting_for_final_and_next_audio_flows() -> None:
    session, live = await _session()
    first, second = _request(1), _request(2)
    try:
        await session.begin_turn(first)
        await session.send_turn_audio(
            first.identity,
            b"\x01\x00\x02\x00",
            payload_sequence=1,
            source_ranges=_span(0, 2),
            context_only=False,
        )
        await _seal(session, first)
        submitted = await _next(session)
        assert isinstance(submitted, STTProviderInputTerminal)
        assert (submitted.identity, submitted.outcome, submitted.epoch_disposition) == (
            first.identity,
            "submitted",
            "reuse",
        )
        await session.begin_turn(second)
        await session.send_turn_audio(
            second.identity,
            b"\x03\x00\x04\x00",
            payload_sequence=1,
            source_ranges=_span(2, 4),
            context_only=False,
        )
        assert live.sent == [
            {"audio": {"data": b"\x01\x00\x02\x00", "mime_type": "audio/pcm;rate=16000"}},
            {"audio_stream_end": True},
            {"audio": {"data": b"\x03\x00\x04\x00", "mime_type": "audio/pcm;rate=16000"}},
        ]
        await _seal(session, second)
        assert (await _next(session)).outcome == "submitted"
        live.push(_message(final="second or first", include_final=True))
        unit = await _next(session)
        assert isinstance(unit, STTRecognitionUnit)
        assert unit.text == "second or first"
        assert unit.identity.stream.provider_epoch_id == "epoch"
        assert unit.identity.stream.capture_epoch == 1
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_final_before_seal_and_repeated_finals_after_seal_are_independent_units() -> None:
    session, live = await _session()
    request = _request(1)
    try:
        await session.begin_turn(request)
        live.push(_message(final="early", include_final=True))
        early = await _next(session)
        assert isinstance(early, STTRecognitionUnit)
        assert early.text == "early"
        await _seal(session, request)
        assert (await _next(session)).outcome == "submitted"
        live.push(_message(final="same", include_final=True))
        live.push(_message(final="same", include_final=True))
        first, second = await _next(session), await _next(session)
        assert all(isinstance(unit, STTRecognitionUnit) for unit in (first, second))
        assert [unit.text for unit in (first, second)] == ["same", "same"]
        assert [unit.identity.receipt_sequence for unit in (early, first, second)] == [1, 2, 3]
        assert len({unit.identity.unit_id for unit in (early, first, second)}) == 3
        assert all(unit.identity.stream == early.identity.stream for unit in (first, second))
        assert all(
            unit.provenance.transcription is not None
            and unit.provenance.transcription.words == ()
            and unit.provenance.transcription.language_code is None
            and unit.provenance.transcription.speaker_label is None
            for unit in (early, first, second)
        )
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
            terminal = await _next(session)
            assert isinstance(terminal, STTProviderInputTerminal)
            assert terminal.outcome == "submitted"
            live.push(_message(final=str(order), include_final=True))
            assert (await _next(session)).text == str(order)
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
async def test_writer_serializes_audio_and_fence_without_native_barrier() -> None:
    session, live = await _session()
    first, second = _request(1), _request(2)
    try:
        await session.begin_turn(first)
        live.send_gate.clear()
        audio = asyncio.create_task(
            session.send_turn_audio(
                first.identity,
                b"\x01\x00",
                payload_sequence=1,
                source_ranges=_span(0, 1),
                context_only=False,
            )
        )
        await _wait_sent(live, "audio")
        seal = asyncio.create_task(_seal(session, first))
        await asyncio.sleep(0)
        assert not audio.done() and not seal.done()
        live.send_gate.set()
        await audio
        await seal
        assert (await _next(session)).outcome == "submitted"
        await session.begin_turn(second)
        await session.send_turn_audio(
            second.identity,
            b"\x02\x00",
            payload_sequence=1,
            source_ranges=_span(1, 2),
            context_only=False,
        )
        assert [next(iter(item)) for item in live.sent] == ["audio", "audio_stream_end", "audio"]
    finally:
        live.send_gate.set()
        await session.close()


@pytest.mark.asyncio
async def test_source_eof_does_not_repeat_local_fence_without_new_audio() -> None:
    session, live = await _session()
    request = _request(1)
    try:
        await session.begin_turn(request)
        await session.send_turn_audio(
            request.identity,
            b"\x01\x00",
            payload_sequence=1,
            source_ranges=_span(0, 1),
            context_only=False,
        )
        await _seal(session, request)
        assert (await _next(session)).outcome == "submitted"
        eof = asyncio.create_task(session.end_stream(reason="source_eof"))
        live.push(_message(final="Yes.", include_final=True))
        assert (await _next(session)).text == "Yes."
        await eof
        assert [next(iter(item)) for item in live.sent] == ["audio", "audio_stream_end"]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_native_activity_end_missing_result_and_interim_do_not_retire_stream() -> None:
    session, live = await _session(timeout=0.01)
    first, second = _request(1), _request(2)
    try:
        await session.begin_turn(first)
        await _seal(session, first)
        assert (await _next(session)).outcome == "submitted"
        live.push(_interim("not final"))
        live.push(_message(ack=True))
        await asyncio.sleep(0.03)
        assert session._event_projection.scoped_event_depth == 0
        await session.begin_turn(second)
        await session.send_turn_audio(
            second.identity,
            b"\x01\x00",
            payload_sequence=1,
            source_ranges=_span(0, 1),
            context_only=False,
        )
        await _seal(session, second)
        assert (await _next(session)).outcome == "submitted"
        live.push(_message(final="actual", include_final=True))
        unit = await _next(session)
        assert isinstance(unit, STTRecognitionUnit)
        assert unit.text == "actual"
        assert not session._protocol_failed
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_source_overlap_is_deduplicated_and_discontinuity_is_explicit() -> None:
    session, live = await _session()
    request = _request(1)
    try:
        await session.begin_turn(request)
        await session.send_stream_audio(b"\x01\x00\x02\x00\x03\x00", source_ranges=_span(0, 3))
        await session.send_stream_audio(b"\x02\x00\x03\x00\x04\x00", source_ranges=_span(1, 4))
        assert [item["audio"]["data"] for item in live.sent if "audio" in item] == [
            b"\x01\x00\x02\x00\x03\x00",
            b"\x04\x00",
        ]
        assert session.recognition_source_covers(_span(1, 4))
        assert not session.recognition_source_covers(_span(4, 5))
        with pytest.raises(ValueError, match="discontinuity"):
            await session.send_stream_audio(b"\x06\x00", source_ranges=_span(5, 6))
        ended = await _next(session)
        assert isinstance(ended, STTProviderInputTerminal)
        assert (ended.outcome, ended.epoch_disposition) == ("failed", "retire")
        assert isinstance(await _next(session), STTProviderEpochEnded)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_initial_context_and_repeated_onsets_send_each_source_sample_once() -> None:
    session, live = await _session()
    first, second = _request(1), _request(2)
    try:
        await session.begin_turn(first)
        await session.send_turn_audio(
            first.identity,
            b"aabb",
            payload_sequence=1,
            source_ranges=_span(0, 2),
            context_only=True,
        )
        await session.send_stream_audio(b"aabbccdd", source_ranges=_span(0, 4))
        await _seal(session, first)
        assert (await _next(session)).outcome == "submitted"
        await session.send_stream_audio(b"eeff", source_ranges=_span(4, 6))
        await session.begin_turn(second)
        await session.send_turn_audio(
            second.identity,
            b"bbcc",
            payload_sequence=1,
            source_ranges=_span(1, 3),
            context_only=True,
        )
        await session.send_turn_audio(
            second.identity,
            b"eeffgg",
            payload_sequence=2,
            source_ranges=_span(4, 7),
            context_only=True,
        )
        await session.send_turn_audio(
            second.identity,
            b"gghh",
            payload_sequence=3,
            source_ranges=_span(6, 8),
            context_only=False,
        )
        assert [next(iter(item)) for item in live.sent] == [
            "audio",
            "audio",
            "audio_stream_end",
            "audio",
            "audio",
            "audio",
        ]
        assert [item["audio"]["data"] for item in live.sent if "audio" in item] == [
            b"aabb",
            b"ccdd",
            b"eeff",
            b"gg",
            b"hh",
        ]
        assert session.recognition_source_covers(_span(0, 8))
        assert not session.recognition_source_covers(_span(8, 9))
        with pytest.raises(ValueError, match="payload_sequence"):
            await session.send_turn_audio(
                second.identity,
                b"bbcc",
                payload_sequence=1,
                source_ranges=_span(1, 3),
                context_only=True,
            )
        assert [item["audio"]["data"] for item in live.sent if "audio" in item][-1] == b"hh"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_stale_stream_generation_and_epoch_cannot_be_admitted() -> None:
    session, _ = await _session()
    first = _request(1)
    try:
        await session.begin_turn(first)
        await _seal(session, first)
        assert (await _next(session)).outcome == "submitted"
        second = _request(2)
        stale_generation = replace(
            second,
            identity=replace(
                second.identity,
                segment=replace(second.identity.segment, activation_generation=2),
            ),
        )
        with pytest.raises(RuntimeError, match="stream ownership"):
            await session.begin_turn(stale_generation)
        with pytest.raises(RuntimeError, match="stream ownership"):
            await session.begin_turn(
                replace(first, identity=replace(first.identity, provider_epoch_id="other"))
            )
        with pytest.raises(RuntimeError, match="unknown or retired"):
            await session.send_turn_audio(
                first.identity,
                b"\x01\x00",
                payload_sequence=2,
                source_ranges=_span(0, 1),
                context_only=False,
            )
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_source_eof_drain_is_observation_not_final_acknowledgement() -> None:
    session, live = await _session(timeout=0.01)
    request = _request(1)
    try:
        await session.begin_turn(request)
        await _seal(session, request)
        assert (await _next(session)).outcome == "submitted"
        await session.end_stream(reason="source_eof")
        await session.begin_turn(_request(2))
        live.push(_message(final="late", include_final=True))
        assert (await _next(session)).text == "late"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_abort_and_go_away_retire_stream_without_promoting_interim() -> None:
    aborted, _ = await _session()
    request = _request(1)
    try:
        await aborted.begin_turn(request)
        await aborted.abort_turn(request.identity, reason="cancelled")
        terminal = await _next(aborted)
        assert isinstance(terminal, STTProviderInputTerminal)
        assert (terminal.outcome, terminal.epoch_disposition) == ("cancelled", "retire")
        assert isinstance(await _next(aborted), STTProviderEpochEnded)
        with pytest.raises(RuntimeError, match="closed"):
            await aborted.begin_turn(_request(2))
    finally:
        await aborted.close()

    retiring, live = await _session()
    try:
        await retiring.begin_turn(request)
        live.push(_interim("partial"))
        live.push(_go_away(final="received", time_left="0.01s"))
        unit = await _next(retiring)
        assert isinstance(unit, STTRecognitionUnit)
        assert unit.text == "received"
        terminal = await _next(retiring)
        assert isinstance(terminal, STTProviderInputTerminal)
        assert terminal.failure_reason == "gemini_go_away"
        assert terminal.epoch_disposition == "retire"
        assert isinstance(await _next(retiring), STTProviderEpochEnded)
    finally:
        await retiring.close()


@pytest.mark.asyncio
async def test_go_away_receives_later_independent_final_before_bounded_retirement() -> None:
    session, live = await _session()
    request = _request(1)
    try:
        await session.begin_turn(request)
        await _seal(session, request)
        assert (await _next(session)).outcome == "submitted"
        live.push(_go_away(time_left="0.25s"))
        live.push(_message(final="after notice", include_final=True))
        unit = await _next(session)
        assert isinstance(unit, STTRecognitionUnit)
        assert unit.text == "after notice"
        assert unit.identity.stream.provider_epoch_id == request.identity.provider_epoch_id
        next_request = _request(2)
        await session.begin_turn(next_request)
        await session.send_turn_audio(
            next_request.identity,
            b"\x01\x00",
            payload_sequence=1,
            source_ranges=_span(0, 1),
            context_only=False,
        )
        await _seal(session, next_request)
        submitted = await _next(session)
        assert isinstance(submitted, STTProviderInputTerminal)
        assert submitted.identity == next_request.identity
        assert submitted.outcome == "submitted"
        live.push(_message(final="still receiving", include_final=True))
        next_unit = await _next(session)
        assert isinstance(next_unit, STTRecognitionUnit)
        assert next_unit.text == "still receiving"
        assert next_unit.identity.receipt_sequence == unit.identity.receipt_sequence + 1
        terminal = await _next(session)
        assert isinstance(terminal, STTProviderEpochEnded)
        assert (terminal.reason, terminal.orderly) == ("gemini_go_away", False)
        with pytest.raises(RuntimeError, match="closed"):
            await session.begin_turn(_request(3))

        async def socket_closed() -> None:
            while not live.closed:
                await asyncio.sleep(0)

        await asyncio.wait_for(socket_closed(), timeout=1)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_duplicate_go_away_cannot_extend_first_deadline() -> None:
    session, live = await _session()
    try:
        await session.begin_turn(_request(1))
        live.push(_go_away(time_left="0.03s"))
        live.push(_go_away(time_left="2s"))
        terminal = await asyncio.wait_for(anext(session.turn_events()), timeout=0.5)
        assert isinstance(terminal, STTProviderInputTerminal)
        assert terminal.failure_reason == "gemini_go_away"
        assert isinstance(await _next(session), STTProviderEpochEnded)
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("time_left", [None, "0s", "private-time-left api_key=private-key"])
async def test_go_away_without_positive_grace_retires_immediately(
    caplog: pytest.LogCaptureFixture, time_left: str | None
) -> None:
    session, live = await _session()
    caplog.set_level(logging.INFO, logger="puripuly_heart.providers.stt.gemini_transcribe")
    try:
        await session.begin_turn(_request(1))
        live.push(_go_away(time_left=time_left))
        terminal = await _next(session)
        assert isinstance(terminal, STTProviderInputTerminal)
        assert terminal.failure_reason == "gemini_go_away"
        assert isinstance(await _next(session), STTProviderEpochEnded)
        with pytest.raises(RuntimeError, match="closed"):
            await session.begin_turn(_request(2))
        records = [
            record
            for record in caplog.records
            if record.name == "puripuly_heart.providers.stt.gemini_transcribe"
        ]
        assert records
        redaction = _DiagnosticRedactionFilter(DIAGNOSTIC_SINK_PERSISTED_LOGS)
        for record in records:
            persisted = copy(record)
            assert redaction.filter(persisted)
            message = persisted.getMessage()
            assert "private-" not in message
            assert record.exc_info is None
            diagnostics, severity, disposition, stage, _ = getattr(
                record, DIAGNOSTIC_RECORD_ATTRIBUTE
            )
            assert diagnostics.fields["recognition_cause"] == terminal.failure_reason
            assert diagnostics.fields["message_kind"] == "go_away"
            assert (severity, disposition, stage) == ("info", "degraded", "receive")
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_go_away_socket_end_retires_without_waiting_for_grace() -> None:
    session, live = await _session()
    try:
        await session.begin_turn(_request(1))
        live.push(_go_away(time_left="5s"))
        live.push(ConnectionError("socket closed"))
        terminal = await _next(session)
        assert isinstance(terminal, STTProviderInputTerminal)
        assert terminal.outcome == "failed"
        assert isinstance(await _next(session), STTProviderEpochEnded)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_repeated_receive_iterators_allow_late_native_final() -> None:
    session, live = await _session()
    request = _request(1)
    try:
        await session.begin_turn(request)
        await _seal(session, request)
        assert (await _next(session)).outcome == "submitted"
        live.push(_ITERATOR_END)
        live.push(_message(final="late", include_final=True))
        assert (await _next(session)).text == "late"
        assert live.receive_calls >= 2
    finally:
        await session.close()
