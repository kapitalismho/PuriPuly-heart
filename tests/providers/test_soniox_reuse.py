from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import pytest
from websockets.asyncio.server import serve

from puripuly_heart.core.audio.ownership import AudioSegmentIdentity, AudioSegmentSettingsSnapshot
from puripuly_heart.core.stt.backend import (
    STTContributionConsumptionLedger,
    STTProviderEpochEnded,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTRecognitionUnit,
    STTSessionProjection,
)
from puripuly_heart.core.stt.scoped_normalizer import STTScopedTurnNormalizer
from puripuly_heart.providers.stt.soniox import SonioxRealtimeSTTBackend, _SonioxSession


class _RecordingWebSocket:
    def __init__(self) -> None:
        self.sent: list[object] = []
        self.closed = False
        self.close_calls = 0

    async def send(self, payload: object) -> None:
        self.sent.append(payload)

    async def close(self) -> None:
        self.close_calls += 1
        self.closed = True


def _session(*, epoch: str = "epoch-1") -> tuple[_SonioxSession, _RecordingWebSocket]:
    websocket = _RecordingWebSocket()
    session = _SonioxSession(
        api_key="k",
        model="m",
        endpoint="wss://example",
        sample_rate_hz=16000,
        language_hints=["en"],
        context_terms=[],
        keepalive_interval_s=100,
        trailing_silence_ms=100,
        connect_timeout_s=5,
        enable_language_identification=True,
        enable_speaker_diarization=True,
        projection=STTSessionProjection(mode="scoped", provider_epoch_id=epoch),
    )
    session._ws = websocket
    session._send_task = asyncio.create_task(session._send_loop())
    return session, websocket


def _request(order: int, *, epoch: str = "epoch-1") -> STTProviderTurnRequest:
    settings = AudioSegmentSettingsSnapshot(
        provider_id="soniox",
        provider_signature=("soniox",),
        runtime_signature=("soniox",),
        source_mode="desktop",
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
        channel="peer",
    )


async def _next(session: _SonioxSession):
    return await asyncio.wait_for(anext(session.turn_events()), timeout=1)


async def _seal(session: _SonioxSession, request: STTProviderTurnRequest) -> None:
    await session.seal_turn(
        request.identity,
        sealed_content_ranges=(),
        seal_reason="silence",
        observed_trailing_silence_ms=0,
    )


@pytest.mark.asyncio
async def test_soniox_reuses_one_transport_for_ordered_final_identical_and_empty_turns() -> None:
    session, websocket = _session()
    writer = session._send_task

    first = _request(1)
    await session.begin_turn(first)
    await session.send_turn_audio(
        first.identity,
        b"first-pcm",
        payload_sequence=1,
        source_ranges=(),
        context_only=False,
    )
    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {
                        "text": "same",
                        "is_final": True,
                        "end_ms": 200,
                        "language": "en",
                        "speaker": "7",
                    }
                ]
            }
        )
    )
    update = await _next(session)
    assert isinstance(update, STTProviderTurnUpdate)
    assert update.text == "same"
    with pytest.raises(RuntimeError, match="unresolved turn"):
        await session.begin_turn(_request(2))

    await _seal(session, first)
    session._handle_message(json.dumps({"tokens": [{"text": "<end>", "is_final": True}]}))
    assert session._event_projection.active_identity == first.identity
    assert session._pending_finalize_requests == 1
    release = await _next(session)
    assert isinstance(release, STTProviderTurnUpdate)
    assert release.source_release is not None
    assert (release.source_release.text_start, release.source_release.text_end) == (0, 4)
    session._handle_message(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))
    first_terminal = await _next(session)
    assert isinstance(first_terminal, STTProviderTurnTerminal)
    assert (first_terminal.text, first_terminal.epoch_disposition) == ("same", "reuse")
    assert first_terminal.failure_retryable is False
    assert first_terminal.recovery_pending is False
    assert first_terminal.final_speaker_runs == ()

    second = _request(2)
    await session.begin_turn(second)
    await session.send_turn_audio(
        second.identity,
        b"second-pcm",
        payload_sequence=1,
        source_ranges=(),
        context_only=False,
    )
    await _seal(session, second)
    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {
                        "text": "same",
                        "is_final": True,
                        "end_ms": 100,
                        "language": "en",
                        "speaker": "7",
                    },
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )
    second_source = await _next(session)
    second_terminal = await _next(session)
    assert isinstance(second_source, STTRecognitionUnit)
    assert isinstance(second_terminal, STTProviderTurnTerminal)
    assert second_source.text == "same"
    assert second_terminal.text == ""
    assert second_terminal.identity != first_terminal.identity
    assert second_terminal.epoch_disposition == "reuse"
    assert second_source.final_speaker_runs == ()

    third = _request(3)
    await session.begin_turn(third)
    await _seal(session, third)
    session._handle_message(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))
    empty = await _next(session)
    assert isinstance(empty, STTProviderTurnTerminal)
    assert (empty.outcome, empty.text, empty.epoch_disposition) == ("empty", "", "reuse")
    assert session._send_task is writer
    assert session._event_projection.active_identity is None
    assert session._pending_tokens == []
    assert session._scoped_tokens == []
    assert session._pending_finalize_requests == 0
    assert session._pending_last_end_ms is None

    assert websocket.sent == [
        b"first-pcm",
        json.dumps({"type": "finalize"}),
        b"second-pcm",
        json.dumps({"type": "finalize"}),
        json.dumps({"type": "finalize"}),
    ]
    await session.close()


@pytest.mark.asyncio
async def test_soniox_idle_late_text_retains_connection_ownership_without_seeding_successor() -> (
    None
):
    session, _ = _session()
    request = _request(1)
    await session.begin_turn(request)
    await _seal(session, request)
    session._handle_message(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))
    terminal = await _next(session)
    assert isinstance(terminal, STTProviderTurnTerminal)
    session._handle_message(
        json.dumps({"tokens": [{"text": "late", "is_final": True, "end_ms": 500}]})
    )
    successor = _request(2)
    await session.begin_turn(successor)
    session._handle_message(json.dumps({"tokens": [{"text": "<end>", "is_final": True}]}))
    source = await _next(session)
    assert isinstance(source, STTRecognitionUnit)
    assert source.text == "late"
    assert source.identity.stream.provider_epoch_id == request.identity.provider_epoch_id
    assert source.provenance.audio_clocks[0].origin_sent_sample is None
    assert source.final_speaker_runs == ()
    assert session._event_projection.active_identity == successor.identity
    assert session._scoped_tokens == []
    await session.close()


@pytest.mark.asyncio
async def test_soniox_finished_response_retires_idle_reused_epoch_before_next_begin() -> None:
    session, websocket = _session()
    request = _request(1)
    await session.begin_turn(request)
    await _seal(session, request)
    session._handle_message(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))
    terminal = await _next(session)
    assert isinstance(terminal, STTProviderTurnTerminal)
    assert terminal.epoch_disposition == "reuse"

    session._handle_message(json.dumps({"tokens": [], "finished": True}))
    ended = await _next(session)
    assert isinstance(ended, STTProviderEpochEnded)
    assert (ended.reason, ended.orderly) == ("soniox_stream_finished", True)
    assert ended.failure_retryable is True
    assert session._event_projection.retired is True
    with pytest.raises(RuntimeError, match="session is closed"):
        await session.begin_turn(_request(2))

    session._handle_message(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))
    await asyncio.sleep(0)
    assert session._event_projection.scoped_event_depth == 0
    await session.close()
    await session.close()
    assert websocket.close_calls == 1


@pytest.mark.asyncio
async def test_soniox_finished_response_retires_active_turn_with_preserved_text() -> None:
    session, websocket = _session()
    request = _request(1)
    await session.begin_turn(request)
    await _seal(session, request)

    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {
                        "text": "kept",
                        "is_final": True,
                        "end_ms": 100,
                        "language": "en",
                        "speaker": "4",
                    }
                ],
                "finished": True,
            }
        )
    )
    update = await _next(session)
    release = await _next(session)
    terminal = await _next(session)
    ended = await _next(session)
    assert isinstance(update, STTProviderTurnUpdate)
    assert update.text == "kept"
    assert isinstance(release, STTProviderTurnUpdate)
    assert release.source_release.reason == "drain"
    assert isinstance(terminal, STTProviderTurnTerminal)
    assert (
        terminal.outcome,
        terminal.text,
        terminal.text_authority,
        terminal.failure_reason,
        terminal.epoch_disposition,
    ) == ("degraded", "kept", "degraded", "soniox_stream_finished", "retire")
    assert terminal.final_language_runs[0].text == "kept"
    assert terminal.final_speaker_runs == ()
    assert isinstance(ended, STTProviderEpochEnded)
    assert (ended.reason, ended.orderly) == ("soniox_stream_finished", True)
    assert terminal.failure_retryable is True
    assert ended.failure_retryable is True

    session._handle_message(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))
    await asyncio.sleep(0)
    assert session._event_projection.scoped_event_depth == 0
    await session.close()
    assert websocket.close_calls == 1


@pytest.mark.asyncio
async def test_soniox_abort_then_late_fin_cannot_publish_or_reactivate_epoch() -> None:
    session, _ = _session()
    request = _request(1)
    await session.begin_turn(request)
    await _seal(session, request)
    await session.abort_turn(request.identity, reason="soniox_final_timeout")

    terminal = await _next(session)
    ended = await _next(session)
    assert isinstance(terminal, STTProviderTurnTerminal)
    assert terminal.outcome == "cancelled"
    assert terminal.epoch_disposition == "retire"
    assert isinstance(ended, STTProviderEpochEnded)

    session._handle_message(
        json.dumps(
            {"tokens": [{"text": "late", "is_final": True}, {"text": "<fin>", "is_final": True}]}
        )
    )
    await asyncio.sleep(0)
    assert session._event_projection.scoped_event_depth == 0
    with pytest.raises(RuntimeError, match="epoch is retired"):
        await session.begin_turn(_request(2))
    await session.close()


@pytest.mark.asyncio
async def test_soniox_speaker_scope_is_connection_local() -> None:
    first, _ = _session(epoch="epoch-a")
    second, _ = _session(epoch="epoch-b")
    assert first.speaker_session_scope != second.speaker_session_scope
    await first.close()
    await second.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "retryable"),
    [
        ({"error_code": 408, "error_type": "request_timeout"}, True),
        ({"error_code": 413, "error_type": "max_duration_reached"}, True),
        ({"error_code": 429, "error_type": "limit_exceeded"}, True),
        ({"error_code": 429, "error_type": "max_concurrent_connections_reached"}, True),
        ({"error_code": "429", "error_type": "max_concurrent_connections_reached"}, True),
        ({"error_code": 429, "error": "legacy limit"}, True),
        ({"error_code": 429, "error_type": "unknown"}, False),
        ({"error_code": 429, "error_type": "unauthenticated"}, False),
        ({"error_code": 429, "error_type": None}, False),
        ({"error_code": 429, "error_type": ["limit_exceeded"]}, False),
        ({"error_code": 401, "error_type": "max_concurrent_connections_reached"}, False),
        ({"error_code": 429, "error": {"code": 401}}, False),
        ({"error_code": 500, "error_type": "internal_error"}, True),
        ({"error_code": 503, "error_type": "service_unavailable"}, True),
        ({"error_code": "503"}, True),
        ({"error_code": 400, "error_type": "invalid_request"}, False),
        ({"error_code": 401, "error_type": "unauthenticated"}, False),
        ({"error_code": 403, "error_type": "permission_denied"}, False),
        ({"error_code": 503, "error_type": "unauthenticated"}, False),
        ({"error_code": 503, "error_type": "unknown"}, False),
        ({"error_code": 503, "error_type": None}, False),
        ({"error_code": 503, "error": {"code": 401}}, False),
        ({"error_code": True}, False),
        ({"error_code": "unavailable"}, False),
        ({"error": "service_unavailable"}, False),
        ({"error_type": "service_unavailable"}, False),
    ],
)
@pytest.mark.parametrize("active", [False, True])
async def test_soniox_server_error_retryability_is_carried_on_turn_and_idle_epoch(
    error: dict[str, object], retryable: bool, active: bool
) -> None:
    session, _ = _session()
    if active:
        await session.begin_turn(_request(1))
    session._handle_message(json.dumps({"tokens": [], **error}))
    if active:
        terminal = await _next(session)
        assert isinstance(terminal, STTProviderTurnTerminal)
        assert terminal.failure_reason == "soniox_request_failed"
        assert terminal.failure_retryable is retryable
        assert terminal.recovery_pending is False
    ended = await _next(session)
    assert isinstance(ended, STTProviderEpochEnded)
    assert ended.reason == "soniox_request_failed"
    assert ended.failure_retryable is retryable
    await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "retryable"),
    [
        ("soniox_write_failed", True),
        ("soniox_receive_failed", True),
        ("soniox_keepalive_failed", True),
        ("soniox_connection_ended", True),
        ("soniox_stream_finished", True),
        ("soniox_protocol_ambiguity", False),
        ("soniox_idle_authoritative_text", False),
        ("soniox_token_buffer_overflow", False),
        ("soniox_final_timeout", False),
    ],
)
@pytest.mark.parametrize("active", [False, True])
async def test_soniox_only_known_transport_reasons_are_retryable(
    reason: str, retryable: bool, active: bool
) -> None:
    session, _ = _session()
    if active:
        await session.begin_turn(_request(1))
    session._scoped_transport_failure(reason, orderly=False)
    if active:
        terminal = await _next(session)
        assert isinstance(terminal, STTProviderTurnTerminal)
        assert terminal.failure_retryable is retryable
    ended = await _next(session)
    assert isinstance(ended, STTProviderEpochEnded)
    assert ended.failure_retryable is retryable
    await session.close()


@pytest.mark.asyncio
async def test_soniox_protocol_close_cannot_turn_transport_failure_into_retry() -> None:
    session, websocket = _session()
    websocket.close_code = 1008
    await session.begin_turn(_request(1))
    session._scoped_transport_failure("soniox_receive_failed", orderly=False)
    terminal = await _next(session)
    ended = await _next(session)
    assert isinstance(terminal, STTProviderTurnTerminal)
    assert terminal.failure_retryable is False
    assert isinstance(ended, STTProviderEpochEnded)
    assert ended.failure_retryable is False
    await session.close()


@pytest.mark.asyncio
async def test_soniox_end_releases_exact_prefix_and_manual_only_consumes_residual() -> None:
    session, websocket = _session()
    request = _request(1)
    normalizer = STTScopedTurnNormalizer(request.identity)
    ledger = STTContributionConsumptionLedger()
    await session.begin_turn(request)
    session._handle_message(
        json.dumps(
            {
                "request_id": "real-connection",
                "tokens": [
                    {"text": " 你", "is_final": True, "start_ms": 0, "end_ms": 60},
                    {"text": "好 ", "is_final": True, "start_ms": 60, "end_ms": 100},
                    {"text": "<end>", "is_final": True},
                ],
                "final_audio_proc_ms": 120,
                "total_audio_proc_ms": 240,
            }
        )
    )
    for _ in range(2):
        update = normalizer.apply_update(await _next(session))
        assert update.source_release is None
    release = normalizer.apply_update(await _next(session))
    assert ledger.consume(release) == " 你好 "
    assert release.provenance.native_audio_progress == (
        ("final_audio_proc_ms", 120.0),
        ("total_audio_proc_ms", 240.0),
    )
    assert session._event_projection.active_identity == request.identity
    assert session._event_projection.sealed is False
    assert websocket.sent == []
    session._handle_message(json.dumps({"tokens": [{"text": "<end>", "is_final": True}]}))
    assert session._event_projection.scoped_event_depth == 0
    await _seal(session, request)
    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "no,3", "is_final": True, "start_ms": 20, "end_ms": 20},
                    {"text": "<end>", "is_final": True},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )
    token = await _next(session)
    assert token.provenance.audio_clocks[0].qualified is False
    assert token.provenance.audio_clocks[0].origin_sent_sample is None
    assert token.provenance.text_timing[0].raw_start == 20
    assert (
        token.provenance.text_timing[0].text_start,
        token.provenance.text_timing[0].text_end,
    ) == (4, 8)
    normalizer.apply_update(token)
    residual = normalizer.apply_update(await _next(session))
    assert ledger.consume(residual) == "no,3"
    terminal = normalizer.apply_terminal(await _next(session))
    assert terminal.text == " 你好 no,3"
    assert ledger.consume(terminal) == ""
    assert terminal.provenance[-1].control_sent_sample == 0
    assert terminal.provenance[-1].control_kind == "manual_finalize"
    assert len(websocket.sent) == 1
    await session.close()


@pytest.mark.asyncio
async def test_soniox_manual_coalesces_and_missing_fin_never_resolves_on_end() -> None:
    session, websocket = _session()
    request = _request(1)
    await session.begin_turn(request)
    await asyncio.gather(_seal(session, request), _seal(session, request))
    assert len(websocket.sent) == 1
    session._handle_message(json.dumps({"tokens": [{"text": "<end>", "is_final": True}]}))
    assert session._pending_finalize_requests == 1
    assert session._event_projection.active_identity == request.identity
    assert session._event_projection.scoped_event_depth == 0
    with pytest.raises(RuntimeError, match="unresolved turn"):
        await session.begin_turn(_request(2))
    session._handle_message(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))
    terminal = await _next(session)
    assert terminal.outcome == "empty"
    successor = _request(2)
    await session.begin_turn(successor)
    session._handle_message(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))
    assert session._event_projection.active_identity == successor.identity
    assert session._event_projection.scoped_event_depth == 0
    await session.close()


@pytest.mark.asyncio
async def test_soniox_post_fin_text_stays_stream_owned_even_with_active_successor() -> None:
    session, _ = _session()
    first = _request(1)
    await session.begin_turn(first)
    await _seal(session, first)
    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "<fin>", "is_final": True},
                    {"text": "late ", "is_final": True, "start_ms": -60, "end_ms": 0},
                ]
            }
        )
    )
    assert isinstance(await _next(session), STTProviderTurnTerminal)
    successor = _request(2)
    await session.begin_turn(successor)
    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "late", "is_final": True, "start_ms": 0, "end_ms": 0},
                    {"text": "<end>", "is_final": True},
                ]
            }
        )
    )
    source = await _next(session)
    assert isinstance(source, STTRecognitionUnit)
    assert source.text == "late late"
    assert source.identity.receipt_sequence == 1
    assert [timing.raw_start for timing in source.provenance.text_timing] == [-60, 0]
    assert [timing.text_start for timing in source.provenance.text_timing] == [0, 5]
    assert all(clock.qualified is False for clock in source.provenance.audio_clocks)
    assert session._event_projection.active_identity == successor.identity
    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "late late", "is_final": True},
                    {"text": "<end>", "is_final": True},
                ]
            }
        )
    )
    repeated = await _next(session)
    assert repeated.text == source.text
    assert repeated.identity != source.identity
    assert repeated.identity.receipt_sequence == 2
    await session.close()


@pytest.mark.asyncio
async def test_soniox_native_ids_cannot_change_scoped_source_composition() -> None:
    observed = []
    for speakers in (("1", "1"), ("7", "999")):
        session, _ = _session()
        request = _request(1)
        await session.begin_turn(request)
        session._handle_message(
            json.dumps(
                {
                    "tokens": [
                        {"text": "same", "is_final": True, "speaker": speakers[0]},
                        {"text": " same", "is_final": True, "speaker": speakers[1]},
                        {"text": "<end>", "is_final": True},
                    ]
                }
            )
        )
        events = [await _next(session) for _ in range(3)]
        observed.append(
            [
                (
                    event.text,
                    event.final_speaker_runs,
                    (
                        (event.source_release.text_start, event.source_release.text_end)
                        if event.source_release
                        else None
                    ),
                )
                for event in events
            ]
        )
        assert [event.provenance.transcription.speaker_label for event in events[:2]] == list(
            speakers
        )
        await session.close()
    assert observed[0] == observed[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("endpointing", [False, True])
async def test_soniox_local_wire_dual_control_and_stop_tail_smoke(endpointing: bool) -> None:
    wire = []
    synthetic = []

    async def handler(connection) -> None:
        assert connection.request.headers["Authorization"] == "Bearer local-smoke"
        config = json.loads(await connection.recv())
        wire.append(config)
        assert config["enable_endpoint_detection"] is endpointing
        audio_seen = 0
        while True:
            message = await connection.recv()
            if message == "":
                wire.append("stop")
                await connection.send(
                    json.dumps(
                        {
                            "tokens": [
                                {"text": "!", "is_final": True, "start_ms": 5, "end_ms": 5},
                            ],
                            "finished": True,
                        }
                    )
                )
                await connection.close(code=1000)
                return
            if isinstance(message, bytes):
                wire.append(("audio", len(message)))
                audio_seen += 1
                if audio_seen == 1:
                    tokens = [{"text": " 你好 ", "is_final": True, "start_ms": 0, "end_ms": 100}]
                    if endpointing:
                        tokens.append({"text": "<end>", "is_final": True})
                    await connection.send(json.dumps({"tokens": tokens}))
                continue
            payload = json.loads(message)
            wire.append(payload)
            if payload.get("type") == "finalize":
                await connection.send(
                    json.dumps(
                        {
                            "tokens": [
                                {"text": "no,3", "is_final": True, "start_ms": 10, "end_ms": 10},
                                {"text": "<fin>", "is_final": True},
                            ],
                            "final_audio_proc_ms": 360,
                            "total_audio_proc_ms": 360,
                        }
                    )
                )

    async with serve(handler, "127.0.0.1", 0, ping_interval=None) as server:
        host, port = server.sockets[0].getsockname()[:2]
        backend = SonioxRealtimeSTTBackend(
            api_key="local-smoke",
            language_hints=["en"],
            endpoint=f"ws://{host}:{port}",
            enable_endpoint_detection=endpointing,
            enable_speaker_diarization=False,
        )
        session = await backend.open_session(
            projection=STTSessionProjection(mode="scoped", provider_epoch_id="epoch-1")
        )
        session.set_source_mapping_observer(synthetic.append)
        request = _request(1)
        await session.begin_turn(request)
        await session.send_turn_audio(
            request.identity, bytes(3200), payload_sequence=1, source_ranges=(), context_only=False
        )
        update = await _next(session)
        normalizer = STTScopedTurnNormalizer(request.identity)
        normalizer.apply_update(update)
        ledger = STTContributionConsumptionLedger()
        admitted = []
        if endpointing:
            release = normalizer.apply_update(await _next(session))
            admitted.append(ledger.consume(release))
            assert session._event_projection.sealed is False
        await session.seal_turn(
            request.identity,
            sealed_content_ranges=(),
            seal_reason="delivery_deadline",
            observed_trailing_silence_ms=0,
        )
        normalizer.apply_update(await _next(session))
        terminal = normalizer.apply_terminal(await _next(session))
        admitted.append(ledger.consume(terminal))
        assert "".join(admitted) == " 你好 no,3"
        assert terminal.provenance[-1].control_sent_sample == 4800
        await session.stop()
        tail = await _next(session)
        ended = await _next(session)
        assert isinstance(tail, STTRecognitionUnit)
        assert tail.text == "!"
        assert tail.provenance.barrier == "drain"
        assert tail.provenance.audio_clocks[0].qualified is False
        assert isinstance(ended, STTProviderEpochEnded)
        assert ended.orderly is True
        await session.close()
    assert synthetic == [3200]
    assert wire[1:] == [("audio", 3200), ("audio", 6400), {"type": "finalize"}, "stop"]


def test_soniox_automatic_endpointing_remains_opt_in() -> None:
    assert (
        SonioxRealtimeSTTBackend(api_key="k", language_hints=["en"]).enable_endpoint_detection
        is False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "endpoint", "qualified"),
    [
        ("stt-rt-v5", "wss://stt-rt.soniox.com/transcribe-websocket", True),
        ("custom-model", "wss://stt-rt.soniox.com/transcribe-websocket", False),
        ("stt-rt-v5", "wss://custom.example", False),
    ],
)
async def test_soniox_lexical_clock_qualification_and_raw_progress_are_separate(
    model: str, endpoint: str, qualified: bool
) -> None:
    session, _ = _session()
    session.model = model
    session.endpoint = endpoint
    request = _request(1)
    await session.begin_turn(request)
    session._handle_message(
        json.dumps(
            {
                "tokens": [],
                "final_audio_proc_ms": 240.5,
                "total_audio_proc_ms": 360,
            }
        )
    )
    assert session._event_projection.scoped_event_depth == 0
    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "x", "is_final": True, "start_ms": 60.5, "end_ms": 60.5},
                ]
            }
        )
    )
    first = await _next(session)
    assert first.provenance.audio_clocks[0].qualified is qualified
    assert first.provenance.audio_clocks[0].origin_sent_sample == (0 if qualified else None)
    assert first.provenance.text_timing[0].raw_start == 60.5
    assert first.provenance.native_audio_progress == (
        ("final_audio_proc_ms", 240.5),
        ("total_audio_proc_ms", 360.0),
    )
    await _seal(session, request)
    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "y", "is_final": True, "start_ms": -120, "end_ms": -60},
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )
    after = await _next(session)
    assert after.provenance.audio_clocks[0].qualified is False
    assert after.provenance.audio_clocks[0].origin_sent_sample is None
    assert after.provenance.text_timing[0].raw_start == -120
    assert after.provenance.text_timing[0].raw_end == -60
    assert (await _next(session)).text == "xy"
    await session.close()


@pytest.mark.asyncio
async def test_soniox_empty_tokens_and_markers_never_emit_stream_source() -> None:
    session, _ = _session()
    first = _request(1)
    await session.begin_turn(first)
    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "", "is_final": True},
                    {"text": "<end>", "is_final": True},
                ]
            }
        )
    )
    assert session._event_projection.scoped_event_depth == 0
    await _seal(session, first)
    session._handle_message(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))
    assert (await _next(session)).outcome == "empty"
    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {"text": "", "is_final": True},
                    {"text": "<end>", "is_final": True},
                    {"text": "<fin>", "is_final": True},
                    {"text": "<end>", "is_final": True},
                ]
            }
        )
    )
    assert session._event_projection.scoped_event_depth == 0
    assert session._receipt_sequence == 0
    assert session._scoped_tokens == []
    await session.close()


@pytest.mark.asyncio
async def test_soniox_retired_receiver_reclaims_blocked_writer_without_graceful_wait() -> None:
    session, websocket = _session()
    await session.begin_turn(_request(1))
    session._handle_message(
        json.dumps(
            {
                "error_code": 429,
                "error_type": "max_concurrent_connections_reached",
            }
        )
    )
    session._stopped = True
    session._recv_task = asyncio.create_task(asyncio.sleep(0))
    await session._recv_task
    await asyncio.wait_for(session.close(), timeout=0.1)
    assert session._ws is None
    assert websocket.closed is True
    assert session._send_task.done()
    assert session._event_projection.retired is True


@pytest.mark.asyncio
async def test_soniox_cancelled_graceful_close_still_reclaims_socket_and_tasks() -> None:
    session, websocket = _session()
    await session.begin_turn(_request(1))
    receive_gate = asyncio.Event()
    session._recv_task = asyncio.create_task(receive_gate.wait())
    closing = asyncio.create_task(session.close())
    await asyncio.wait_for(asyncio.shield(session._send_task), timeout=1)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert session._ws is None
    assert websocket.closed is True
    assert session._send_task.done()
    assert session._recv_task.done()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["provider_error", "receive", "write", "keepalive"])
@pytest.mark.parametrize("active", [False, True])
async def test_connection_owned_failure_drains_only_unreleased_immutable_tokens(failure, active):
    session, _ = _session()
    try:
        first = _request(1)
        await session.begin_turn(first)
        await _seal(session, first)
        session._handle_message(
            json.dumps(
                {
                    "tokens": [
                        {"text": "first", "is_final": True},
                        {"text": "<fin>", "is_final": True},
                    ]
                }
            )
        )
        assert isinstance(await _next(session), STTProviderTurnUpdate)
        assert (await _next(session)).text == "first"
        stream = session._stream_identity
        session._handle_message(
            json.dumps(
                {
                    "tokens": [
                        {"text": "same", "is_final": True},
                        {"text": "<end>", "is_final": True},
                    ]
                }
            )
        )
        prefix = await _next(session)
        assert isinstance(prefix, STTRecognitionUnit)
        assert prefix.text == "same"
        if active:
            await session.begin_turn(_request(2))
        session._handle_message(
            json.dumps(
                {
                    "tokens": [
                        {"text": "same", "is_final": True, "start_ms": -120.5, "end_ms": -60},
                        {"text": " !", "is_final": True, "start_ms": -60, "end_ms": 3.25},
                        {"text": "provisional", "is_final": False},
                    ],
                    "final_audio_proc_ms": 360.5,
                }
            )
        )
        if failure == "provider_error":
            session._handle_message(
                json.dumps(
                    {
                        "error_code": 503,
                        "error_type": "service_unavailable",
                    }
                )
            )
        else:
            session._scoped_transport_failure(
                f"soniox_{failure}_failed",
                orderly=False,
                exception=OSError("local failure"),
            )
        tail = await _next(session)
        assert isinstance(tail, STTRecognitionUnit)
        assert tail.text == "same !"
        assert tail.identity.stream == prefix.identity.stream == stream
        assert tail.identity.receipt_sequence == prefix.identity.receipt_sequence + 1
        assert tail.identity.unit_id != prefix.identity.unit_id
        assert tail.provenance.barrier == "drain"
        assert tail.provenance.native_audio_progress == (("final_audio_proc_ms", 360.5),)
        assert tail.provenance.audio_clocks[0].qualified is False
        assert tail.provenance.audio_clocks[0].origin_sent_sample is None
        assert [(timing.raw_start, timing.raw_end) for timing in tail.provenance.text_timing] == [
            (-120.5, -60),
            (-60, 3.25),
        ]
        assert tail.final_speaker_runs == ()
        if active:
            terminal = await _next(session)
            assert terminal.outcome == "failed"
            assert terminal.text == ""
        ended = await _next(session)
        assert isinstance(ended, STTProviderEpochEnded)
        assert not ended.orderly
        assert ended.failure_retryable is True
        assert session._scoped_tokens == session._scoped_contributions == []
        session._scoped_transport_failure("soniox_receive_failed", orderly=False)
        session._handle_message(
            json.dumps(
                {
                    "tokens": [{"text": "repeated cleanup", "is_final": True}],
                    "finished": True,
                }
            )
        )
        assert session._event_projection.scoped_event_depth == 0
        assert session._receipt_sequence == 2
    finally:
        await session.close()
