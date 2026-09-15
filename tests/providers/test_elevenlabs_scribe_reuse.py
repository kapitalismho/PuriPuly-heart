from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import AudioSegmentIdentity, AudioSegmentSettingsSnapshot
from puripuly_heart.core.stt.backend import (
    STTNativeProvenance,
    STTProviderEpochEnded,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTSessionProjection,
)
from puripuly_heart.providers.stt.elevenlabs_scribe import ElevenLabsScribeSTTBackend


class _ControlledConnection:
    def __init__(self) -> None:
        self.handlers: dict = {}
        self.sent: list[dict] = []
        self.commits = 0
        self.closed = 0
        self.commit_gate = asyncio.Event()
        self.commit_gate.set()

    def on(self, event, callback) -> None:
        self.handlers[event] = callback

    async def send(self, payload: dict) -> None:
        self.sent.append(payload)

    async def commit(self) -> None:
        self.commits += 1
        await self.commit_gate.wait()

    async def close(self) -> None:
        self.closed += 1

    def emit(self, event: str, payload: dict) -> None:
        self.handlers[event](payload)


def _request(order: int, *, channel: str = "self") -> STTProviderTurnRequest:
    epoch = "scribe-epoch"
    settings = AudioSegmentSettingsSnapshot(
        provider_id="elevenlabs_scribe",
        provider_signature=("elevenlabs_scribe",),
        runtime_signature=("elevenlabs_scribe",),
        source_mode="microphone" if channel == "self" else "desktop",
        source_language="en",
        expected_languages=("en",),
        target_sample_rate_hz=16000,
        vad_speech_threshold=0.4,
        vad_hangover_ms=800,
        vad_pre_roll_ms=500,
    )
    return STTProviderTurnRequest(
        identity=STTProviderTurnIdentity(
            segment=AudioSegmentIdentity(
                activation_generation=1,
                segment_order=order,
                segment_id=uuid4(),
                capture_epoch=1,
            ),
            provider_epoch_id=epoch,
            provider_turn_id=f"turn-{order}",
        ),
        settings=settings,
        channel=channel,
    )


def _spans() -> tuple[AudioCaptureSpan, ...]:
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


async def _session(connection: _ControlledConnection):
    factories: list = []

    async def factory(options):
        factories.append(options)
        return connection

    backend = ElevenLabsScribeSTTBackend(
        api_key="key",
        scribe_connect_factory=factory,
        keepalive_interval_s=60,
    )
    session = await backend.open_session(
        projection=STTSessionProjection(mode="scoped", provider_epoch_id="scribe-epoch")
    )
    return session, factories


async def _seal(session, request: STTProviderTurnRequest) -> None:
    await session.seal_turn(
        request.identity,
        sealed_content_ranges=_spans(),
        seal_reason="silence",
        observed_trailing_silence_ms=200,
    )


async def _next(session):
    return await asyncio.wait_for(anext(session.turn_events()), timeout=1)


@pytest.mark.asyncio
async def test_consecutive_final_empty_and_identical_turns_reuse_one_connection() -> None:
    connection = _ControlledConnection()
    session, factories = await _session(connection)
    queue_task = session._queue_task
    keepalive_task = session._keepalive_task
    handlers = dict(connection.handlers)
    try:
        expected = (("same", "final"), ("", "empty"), ("same", "final"))
        for order, (text, outcome) in enumerate(expected, 1):
            request = _request(order, channel="self" if order != 2 else "peer")
            await session.begin_turn(request)
            await session.send_turn_audio(
                request.identity,
                bytes((order, order)),
                payload_sequence=1,
                source_ranges=_spans(),
                context_only=False,
            )
            await _seal(session, request)
            connection.emit(
                "committed_transcript",
                {"message_type": "committed_transcript", "text": text},
            )
            terminal = await _next(session)
            assert isinstance(terminal, STTProviderTurnTerminal)
            assert (terminal.identity, terminal.text, terminal.outcome) == (
                request.identity,
                text,
                outcome,
            )
            assert terminal.epoch_disposition == "reuse"
            assert session._scoped_provenance == []

        assert len(factories) == 1
        assert connection.commits == 3
        assert connection.closed == 0
        assert connection.handlers == handlers
        assert session._queue_task is queue_task
        assert session._keepalive_task is keepalive_task
    finally:
        await session.close()
    assert connection.closed == 1


@pytest.mark.asyncio
async def test_preview_variants_and_precommit_marker_never_complete_turn() -> None:
    connection = _ControlledConnection()
    session, _ = await _session(connection)
    request = _request(1)
    try:
        await session.begin_turn(request)
        connection.emit(
            "committed_transcript",
            {"message_type": "committed_transcript", "text": "too early"},
        )
        connection.emit(
            "partial_transcript",
            {"message_type": "partial_transcript", "text": "partial"},
        )
        connection.emit(
            "final_transcript",
            {"message_type": "final_transcript", "text": "preview"},
        )
        connection.emit(
            "final_transcript_with_timestamps",
            {"message_type": "final_transcript_with_timestamps", "text": "timestamped"},
        )
        assert session._event_projection.active_identity == request.identity
        assert not session._event_projection.sealed
        await _seal(session, request)
        connection.emit(
            "final_transcript_with_timestamps",
            {"message_type": "final_transcript_with_timestamps", "text": "still preview"},
        )
        connection.emit(
            "committed_transcript",
            {"message_type": "committed_transcript", "text": "canonical"},
        )

        events = [await _next(session), await _next(session)]
        assert isinstance(events[0], STTProviderTurnUpdate)
        assert events[0].text == "still preview"
        assert isinstance(events[1], STTProviderTurnTerminal)
        assert events[1].text == "canonical"
        assert events[1].epoch_disposition == "reuse"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_callback_during_commit_waits_for_successful_commit_write() -> None:
    connection = _ControlledConnection()
    connection.commit_gate.clear()
    session, _ = await _session(connection)
    request = _request(1)
    try:
        await session.begin_turn(request)
        seal = asyncio.create_task(_seal(session, request))
        while connection.commits == 0:
            await asyncio.sleep(0)
        connection.emit(
            "committed_transcript",
            {"message_type": "committed_transcript", "text": "after write"},
        )
        assert session._event_projection.active_identity == request.identity
        assert session._pending_committed is not None
        connection.commit_gate.set()
        await seal
        terminal = await _next(session)
        assert isinstance(terminal, STTProviderTurnTerminal)
        assert terminal.text == "after write"
        assert terminal.epoch_disposition == "reuse"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_callback_during_failed_commit_cannot_authorize_reuse() -> None:
    class _FailingCommitConnection(_ControlledConnection):
        async def commit(self) -> None:
            self.commits += 1
            self.emit(
                "committed_transcript",
                {"message_type": "committed_transcript", "text": "not authoritative"},
            )
            raise RuntimeError("commit write failed")

    connection = _FailingCommitConnection()
    session, _ = await _session(connection)
    request = _request(1)
    try:
        await session.begin_turn(request)
        with pytest.raises(RuntimeError, match="commit write failed"):
            await _seal(session, request)
        failed = await _next(session)
        ended = await _next(session)
        assert isinstance(failed, STTProviderTurnTerminal)
        assert failed.outcome == "failed"
        assert failed.failure_reason == "scribe_commit_failed"
        assert failed.epoch_disposition == "retire"
        assert isinstance(ended, STTProviderEpochEnded)
        assert session._stopped is True
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_missing_committed_text_is_protocol_failure_but_explicit_empty_is_authoritative() -> (
    None
):
    empty_connection = _ControlledConnection()
    empty_session, _ = await _session(empty_connection)
    empty_request = _request(1)
    await empty_session.begin_turn(empty_request)
    await _seal(empty_session, empty_request)
    empty_connection.emit(
        "committed_transcript",
        {"message_type": "committed_transcript", "text": ""},
    )
    empty = await _next(empty_session)
    assert isinstance(empty, STTProviderTurnTerminal)
    assert (empty.outcome, empty.text_authority, empty.epoch_disposition) == (
        "empty",
        "authoritative",
        "reuse",
    )
    await empty_session.close()

    malformed_connection = _ControlledConnection()
    malformed_session, _ = await _session(malformed_connection)
    malformed_request = _request(1)
    try:
        await malformed_session.begin_turn(malformed_request)
        await _seal(malformed_session, malformed_request)
        malformed_connection.emit(
            "committed_transcript",
            {"message_type": "committed_transcript"},
        )
        failed = await _next(malformed_session)
        ended = await _next(malformed_session)
        assert isinstance(failed, STTProviderTurnTerminal)
        assert failed.failure_reason == "scribe_committed_transcript_missing_text"
        assert failed.epoch_disposition == "retire"
        assert isinstance(ended, STTProviderEpochEnded)
    finally:
        await malformed_session.close()


@pytest.mark.asyncio
async def test_idle_duplicate_commit_retires_without_creating_another_terminal() -> None:
    connection = _ControlledConnection()
    session, _ = await _session(connection)
    request = _request(1)
    try:
        await session.begin_turn(request)
        await _seal(session, request)
        payload = {"message_type": "committed_transcript", "text": "once"}
        connection.emit("committed_transcript", payload)
        terminal = await _next(session)
        assert isinstance(terminal, STTProviderTurnTerminal)
        connection.emit("committed_transcript", payload)
        ended = await _next(session)
        assert isinstance(ended, STTProviderEpochEnded)
        assert ended.reason == "scribe_unsolicited_committed_transcript"
        assert session._stopped is True
    finally:
        await session.close()


@pytest.mark.parametrize(
    ("event_name", "reason"),
    (
        ("queue_overflow", "scribe_queue_overflow"),
        ("session_time_limit_exceeded", "scribe_session_time_limit_exceeded"),
        ("insufficient_audio_activity", "scribe_insufficient_audio_activity"),
        ("close", "scribe_connection_closed"),
    ),
)
@pytest.mark.asyncio
async def test_provider_failure_and_eof_retire_active_turn(
    event_name: str,
    reason: str,
) -> None:
    connection = _ControlledConnection()
    session, _ = await _session(connection)
    request = _request(1)
    try:
        await session.begin_turn(request)
        connection.emit(event_name, {"message_type": event_name})
        failed = await _next(session)
        ended = await _next(session)
        assert isinstance(failed, STTProviderTurnTerminal)
        assert failed.failure_reason == reason
        assert failed.epoch_disposition == "retire"
        assert isinstance(ended, STTProviderEpochEnded)
        assert ended.reason == reason
        assert session._stopped is True
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_one_channels_failure_does_not_retire_or_close_another_channel() -> None:
    self_connection = _ControlledConnection()
    peer_connection = _ControlledConnection()
    self_session, _ = await _session(self_connection)
    peer_session, _ = await _session(peer_connection)
    self_request = _request(1, channel="self")
    peer_request = _request(2, channel="peer")
    try:
        await self_session.begin_turn(self_request)
        await peer_session.begin_turn(peer_request)
        self_connection.emit("queue_overflow", {"message_type": "queue_overflow"})
        self_failed = await _next(self_session)
        assert isinstance(self_failed, STTProviderTurnTerminal)
        assert self_failed.epoch_disposition == "retire"

        await _seal(peer_session, peer_request)
        peer_connection.emit(
            "committed_transcript",
            {"message_type": "committed_transcript", "text": "peer remains healthy"},
        )
        peer_terminal = await _next(peer_session)
        assert isinstance(peer_terminal, STTProviderTurnTerminal)
        assert peer_terminal.epoch_disposition == "reuse"
        await self_session.close()
        assert self_connection.closed == 1
        assert peer_connection.closed == 0
        assert peer_session._stopped is False
    finally:
        await self_session.close()
        await peer_session.close()


@pytest.mark.asyncio
async def test_captured_completed_turn_identity_cannot_finish_successor() -> None:
    connection = _ControlledConnection()
    session, _ = await _session(connection)
    first = _request(1)
    second = _request(2)
    try:
        await session.begin_turn(first)
        await _seal(session, first)
        connection.emit(
            "committed_transcript",
            {"message_type": "committed_transcript", "text": "same"},
        )
        assert isinstance(await _next(session), STTProviderTurnTerminal)

        await session.begin_turn(second)
        await _seal(session, second)
        session._finish_committed(
            first.identity,
            "stale",
            STTNativeProvenance(barrier="committed_transcript"),
        )
        assert session._event_projection.active_identity == second.identity
        connection.emit(
            "committed_transcript",
            {"message_type": "committed_transcript", "text": "same"},
        )
        terminal = await _next(session)
        assert isinstance(terminal, STTProviderTurnTerminal)
        assert (terminal.identity, terminal.text, terminal.epoch_disposition) == (
            second.identity,
            "same",
            "reuse",
        )
    finally:
        await session.close()
