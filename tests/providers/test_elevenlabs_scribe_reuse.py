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


class _NoNetworkWebSocket:
    async def close(self, *_args) -> None:
        pass


class _ControlledConnection:
    def __init__(self) -> None:
        from elevenlabs.realtime import RealtimeConnection

        self._sdk_connection = RealtimeConnection(
            _NoNetworkWebSocket(),
            current_sample_rate=16000,
        )
        self.handlers: dict = {}
        self.sent: list[dict] = []
        self.commits = 0
        self.closed = 0
        self.commit_gate = asyncio.Event()
        self.commit_gate.set()

    def on(self, event, callback) -> None:
        self.handlers[event] = callback
        self._sdk_connection.on(event, callback)

    async def send(self, payload: dict) -> None:
        self.sent.append(payload)

    async def commit(self) -> None:
        self.commits += 1
        await self.commit_gate.wait()

    async def close(self) -> None:
        self.closed += 1
        await self._sdk_connection.close()

    def emit(self, event: str, payload: dict) -> None:
        self._sdk_connection._emit(event, payload)

    def emit_close(self) -> None:
        from elevenlabs.realtime import RealtimeEvents

        self._sdk_connection._emit(RealtimeEvents.CLOSE)


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
                "committed_transcript_with_timestamps",
                {"message_type": "committed_transcript_with_timestamps", "text": text},
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
        await session.close()
    assert connection.closed == 1


@pytest.mark.asyncio
async def test_native_sdk_close_retires_idle_epoch_and_rejects_next_begin() -> None:
    connection = _ControlledConnection()
    session, _ = await _session(connection)
    try:
        connection.emit_close()
        ended = await _next(session)
        assert isinstance(ended, STTProviderEpochEnded)
        assert ended.reason == "scribe_connection_closed"
        assert ended.orderly is True
        with pytest.raises(RuntimeError, match="Scribe session is closed"):
            await session.begin_turn(_request(1))
    finally:
        await session.close()


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
            "committed_transcript_with_timestamps",
            {"message_type": "committed_transcript_with_timestamps", "text": "canonical"},
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
            "committed_transcript_with_timestamps",
            {"message_type": "committed_transcript_with_timestamps", "text": "after write"},
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
                "committed_transcript_with_timestamps",
                {
                    "message_type": "committed_transcript_with_timestamps",
                    "text": "not authoritative",
                },
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
        "committed_transcript_with_timestamps",
        {"message_type": "committed_transcript_with_timestamps", "text": ""},
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
            "committed_transcript_with_timestamps",
            {"message_type": "committed_transcript_with_timestamps"},
        )
        failed = await _next(malformed_session)
        ended = await _next(malformed_session)
        assert isinstance(failed, STTProviderTurnTerminal)
        assert failed.failure_reason == "scribe_timestamped_transcript_missing_text"
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
        payload = {"message_type": "committed_transcript_with_timestamps", "text": "once"}
        connection.emit("committed_transcript_with_timestamps", payload)
        terminal = await _next(session)
        assert isinstance(terminal, STTProviderTurnTerminal)
        connection.emit("committed_transcript_with_timestamps", payload)
        ended = await _next(session)
        assert isinstance(ended, STTProviderEpochEnded)
        assert ended.reason == "scribe_unsolicited_timestamped_transcript"
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
        if event_name == "close":
            connection.emit_close()
        else:
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
            "committed_transcript_with_timestamps",
            {
                "message_type": "committed_transcript_with_timestamps",
                "text": "peer remains healthy",
            },
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
            "committed_transcript_with_timestamps",
            {"message_type": "committed_transcript_with_timestamps", "text": "same"},
        )
        assert isinstance(await _next(session), STTProviderTurnTerminal)

        await session.begin_turn(second)
        await _seal(session, second)
        session._finish_committed(
            first.identity,
            "stale",
            STTNativeProvenance(barrier="committed_transcript_with_timestamps"),
        )
        assert session._event_projection.active_identity == second.identity
        connection.emit(
            "committed_transcript_with_timestamps",
            {"message_type": "committed_transcript_with_timestamps", "text": "same"},
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


def _cwt(text: str, start: float | None, end: float | None, *, speaker: str = "native"):
    return {
        "message_type": "committed_transcript_with_timestamps",
        "text": text,
        "language_code": "zh",
        "words": [
            {"text": text, "start": start, "end": end, "type": "word", "speaker_id": speaker}
        ],
    }


@pytest.mark.asyncio
async def test_cwt_own_text_wins_without_plain_string_pairing_and_preserves_null_clock() -> None:
    connection = _ControlledConnection()
    session, _ = await _session(connection)
    request = _request(1)
    try:
        await session.begin_turn(request)
        await _seal(session, request)
        connection.emit(
            "committed_transcript", {"message_type": "committed_transcript", "text": "wrong plain"}
        )
        assert session._event_projection.scoped_event_depth == 0
        connection.emit("committed_transcript_with_timestamps", _cwt(" 你 好! ", 0.1, None))
        terminal = await _next(session)
        assert terminal.text == " 你 好! "
        assert terminal.epoch_disposition == "reuse"
        evidence = terminal.provenance[-1]
        assert evidence.transcription.words == ((" 你 好! ", "0.1", None),)
        assert evidence.transcription.language_code == "zh"
        assert evidence.transcription.speaker_label is None
        assert evidence.text_timing[0].raw_end is None
        assert evidence.control_kind == "commit"
        assert evidence.control_id == "turn-1:commit"
        assert session._plain_fallback_task is None
        connection.emit(
            "committed_transcript", {"message_type": "committed_transcript", "text": "late"}
        )
        assert session._stopped is False
        assert session._event_projection.scoped_event_depth == 0
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_owned_first_plain_fallback_has_one_irreversible_winner_against_late_cwt() -> None:
    connection = _ControlledConnection()
    session, _ = await _session(connection)
    session._plain_fallback_timeout_s = 0.001
    request = _request(1)
    try:
        await session.begin_turn(request)
        await _seal(session, request)
        connection.emit(
            "committed_transcript", {"message_type": "committed_transcript", "text": " No. "}
        )
        terminal = await _next(session)
        assert (terminal.text, terminal.outcome, terminal.epoch_disposition) == (
            " No. ",
            "degraded",
            "retire",
        )
        assert terminal.failure_reason == "scribe_timestamped_transcript_missing"
        ended = await _next(session)
        assert isinstance(ended, STTProviderEpochEnded)
        connection.emit("committed_transcript_with_timestamps", _cwt("No, three.", 0.1, 0.6))
        assert session._event_projection.scoped_event_depth == 0
        with pytest.raises(RuntimeError, match="closed"):
            await session.begin_turn(_request(2))
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_unmatched_old_plain_cannot_become_sealed_successor_fallback() -> None:
    connection = _ControlledConnection()
    session, _ = await _session(connection)
    session._plain_fallback_timeout_s = 0.001
    try:
        first = _request(1)
        await session.begin_turn(first)
        await _seal(session, first)
        connection.emit("committed_transcript_with_timestamps", _cwt("previous", 0.1, 0.5))
        assert (await _next(session)).epoch_disposition == "reuse"
        second = _request(2)
        await session.begin_turn(second)
        await _seal(session, second)
        connection.emit(
            "committed_transcript", {"message_type": "committed_transcript", "text": "previous"}
        )
        failed = await _next(session)
        assert (failed.identity, failed.outcome, failed.text) == (second.identity, "failed", "")
        assert failed.failure_reason == "scribe_plain_owner_unresolved"
        assert failed.text_authority == "none"
        assert isinstance(await _next(session), STTProviderEpochEnded)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_cwt_prefix_and_terminal_tail_conserve_nonspace_distinct_repetitions() -> None:
    from puripuly_heart.core.stt.backend import STTContributionConsumptionLedger
    from puripuly_heart.core.stt.scoped_normalizer import STTScopedTurnNormalizer

    connection = _ControlledConnection()
    session, _ = await _session(connection)
    request = _request(1)
    try:
        await session.begin_turn(request)
        payload = _cwt("不。", 0.1, 0.3, speaker="first")
        connection.emit("committed_transcript_with_timestamps", payload)
        prefix = await _next(session)
        connection.emit("committed_transcript_with_timestamps", payload)
        assert session._event_projection.scoped_event_depth == 0
        await _seal(session, request)
        connection.emit(
            "committed_transcript_with_timestamps", _cwt("不。", 0.5, 0.8, speaker="other")
        )
        terminal = await _next(session)
        assert terminal.text == "不。不。"
        assert prefix.source_release.text_end == len("不。")
        assert len(terminal.included_contributions) == 2
        assert (
            terminal.included_contributions[0].contribution_id
            != terminal.included_contributions[1].contribution_id
        )
        normalizer = STTScopedTurnNormalizer(request.identity)
        consumption = STTContributionConsumptionLedger()
        assert consumption.consume(normalizer.apply_update(prefix)) == "不。"
        assert consumption.consume(normalizer.apply_terminal(terminal)) == "不。"
        assert not prefix.final_speaker_runs
        assert not terminal.final_speaker_runs
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_scribe_native_clock_reset_keeps_raw_text_and_disables_only_timing() -> None:
    connection = _ControlledConnection()
    session, _ = await _session(connection)
    try:
        first = _request(1)
        await session.begin_turn(first)
        await _seal(session, first)
        connection.emit("committed_transcript_with_timestamps", _cwt("前", 2.0, 2.5))
        assert (await _next(session)).provenance[-1].audio_clocks[0].qualified
        second = _request(2)
        await session.begin_turn(second)
        await _seal(session, second)
        connection.emit("committed_transcript_with_timestamps", _cwt("後", 0.1, 0.4))
        terminal = await _next(session)
        assert terminal.text == "後"
        provenance = terminal.provenance[-1]
        assert not provenance.audio_clocks[0].qualified
        assert provenance.text_timing[0].raw_start == 0.1
        assert terminal.epoch_disposition == "reuse"
    finally:
        await session.close()
