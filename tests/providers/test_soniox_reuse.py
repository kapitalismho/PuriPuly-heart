from __future__ import annotations

import asyncio
import json
from collections import deque
from uuid import uuid4

import pytest
from websockets.asyncio.server import serve

from puripuly_heart.core.audio.ownership import AudioSegmentIdentity, AudioSegmentSettingsSnapshot
from puripuly_heart.core.stt.backend import (
    STTProviderEpochEnded,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
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


async def _collect_turn(session: _SonioxSession) -> list[object]:
    events = []
    async for event in session.turn_events():
        events.append(event)
        if isinstance(event, STTProviderTurnTerminal):
            return events
    raise AssertionError("session ended without a terminal")


def _burst_tokens(count: int) -> list[dict[str, object]]:
    return [
        {
            "text": f"word{index} ",
            "is_final": True,
            "start_ms": index * 10,
            "end_ms": (index + 1) * 10,
            "language": "en" if index < count // 3 or index >= count * 2 // 3 else "ja",
            "speaker": "A" if index < count // 3 or index >= count * 2 // 3 else "B",
        }
        for index in range(count)
    ]


def _assert_burst_preserved(
    events: list[object],
    request: STTProviderTurnRequest,
    tokens: list[dict[str, object]],
    scope: str,
) -> None:
    updates = [event for event in events if isinstance(event, STTProviderTurnUpdate)]
    terminal = events[-1]
    assert isinstance(terminal, STTProviderTurnTerminal)
    text = "".join(token["text"] for token in tokens)
    assert "".join(update.text for update in updates) == text
    assert terminal.identity == request.identity
    assert (terminal.outcome, terminal.text, terminal.epoch_disposition) == ("final", text, "reuse")
    assert terminal.failure_reason is None
    assert [run.language for run in terminal.final_language_runs] == ["en", "ja", "en"]
    assert [run.speaker_id for run in terminal.final_speaker_runs] == ["A", "B", "A"]
    assert {run.session_scope for run in terminal.final_speaker_runs} == {scope}
    assert "".join(run.text for run in terminal.final_language_runs) == text
    assert "".join(run.text for run in terminal.final_speaker_runs) == text
    normalizer = STTScopedTurnNormalizer(request.identity)
    for update in updates:
        normalizer.apply_update(update)
    normalized = normalizer.apply_terminal(terminal)
    assert normalized.outcome == "final"
    assert normalized.text == text.rstrip()
    assert [run.speaker_id for run in normalized.final_speaker_runs] == ["A", "B", "A"]


@pytest.mark.asyncio
@pytest.mark.parametrize("token_count", [255, 256, 1024])
@pytest.mark.parametrize("fin_in_same_response", [False, True])
async def test_soniox_burst_preserves_complete_turn_and_attribution(
    token_count: int, fin_in_same_response: bool
) -> None:
    session, _ = _session()
    request = _request(1)
    tokens = _burst_tokens(token_count)
    fin = {"text": "<fin>", "is_final": True}
    collector = None
    try:
        await session.begin_turn(request)
        await _seal(session, request)
        collector = asyncio.create_task(_collect_turn(session))
        await asyncio.sleep(0)
        session._handle_message(
            json.dumps(
                {
                    "request_id": "request-1",
                    "tokens": tokens + ([fin] if fin_in_same_response else []),
                }
            )
        )
        if not fin_in_same_response:
            await asyncio.sleep(0)
            session._handle_message(json.dumps({"tokens": [fin]}))
        events = await asyncio.wait_for(collector, timeout=1)
        _assert_burst_preserved(events, request, tokens, session.speaker_session_scope)
        await session.begin_turn(_request(2))
    finally:
        await session.close()
        if collector is not None:
            await asyncio.gather(collector, return_exceptions=True)


@pytest.mark.asyncio
async def test_soniox_buffered_responses_allow_consumer_to_progress() -> None:
    tokens = _burst_tokens(1024)

    class BufferedWebSocket(_RecordingWebSocket):
        def __init__(self) -> None:
            super().__init__()
            self.responses = deque(json.dumps({"tokens": [token]}) for token in tokens)
            self.responses.append(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))

        async def recv(self):
            if self.responses:
                return self.responses.popleft()
            await asyncio.Event().wait()

    session, _ = _session()
    session._ws = BufferedWebSocket()
    request = _request(1)
    collector = None
    try:
        await session.begin_turn(request)
        await _seal(session, request)
        collector = asyncio.create_task(_collect_turn(session))
        session._recv_task = asyncio.create_task(session._recv_loop())
        events = await asyncio.wait_for(collector, timeout=1)
        _assert_burst_preserved(events, request, tokens, session.speaker_session_scope)
    finally:
        await session.close()
        if collector is not None:
            await asyncio.gather(collector, return_exceptions=True)


@pytest.mark.asyncio
async def test_soniox_local_websocket_delivers_burst_and_reuses_connection() -> None:
    tokens = _burst_tokens(1024)
    connections = []

    async def handler(connection) -> None:
        connections.append(connection)
        await connection.recv()
        async for message in connection:
            if isinstance(message, str) and json.loads(message).get("type") == "finalize":
                await connection.send(
                    json.dumps({"tokens": tokens + [{"text": "<fin>", "is_final": True}]})
                )

    async with serve(handler, "127.0.0.1", 0, ping_interval=None) as server:
        host, port = server.sockets[0].getsockname()[:2]
        backend = SonioxRealtimeSTTBackend(
            api_key="synthetic-local-key",
            endpoint=f"ws://{host}:{port}",
            language_hints=["en", "ja"],
            enable_language_identification=True,
            enable_speaker_diarization=True,
        )
        session = await backend.open_session(
            projection=STTSessionProjection(mode="scoped", provider_epoch_id="epoch-1")
        )
        try:
            for order in (1, 2):
                request = _request(order)
                await session.begin_turn(request)
                await _seal(session, request)
                events = await asyncio.wait_for(_collect_turn(session), timeout=2)
                _assert_burst_preserved(events, request, tokens, session.speaker_session_scope)
            assert len(connections) == 1
        finally:
            await session.close()


@pytest.mark.asyncio
async def test_soniox_malformed_token_preserves_preceding_batch_in_failure_terminal() -> None:
    session, _ = _session()
    request = _request(1)
    tokens = _burst_tokens(6)
    try:
        await session.begin_turn(request)
        await _seal(session, request)
        session._handle_message(json.dumps({"request_id": "request-1", "tokens": [*tokens, None]}))
        events = await asyncio.wait_for(_collect_turn(session), timeout=1)
        terminal = events[-1]
        assert isinstance(terminal, STTProviderTurnTerminal)
        assert (terminal.outcome, terminal.failure_reason, terminal.epoch_disposition) == (
            "degraded",
            "soniox_protocol_ambiguity",
            "retire",
        )
        assert terminal.text == "".join(token["text"] for token in tokens)
        assert [run.speaker_id for run in terminal.final_speaker_runs] == ["A", "B", "A"]
        assert [run.language for run in terminal.final_language_runs] == ["en", "ja", "en"]
        assert terminal.provenance[0].native_request_id == "request-1"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_soniox_reuses_one_transport_for_ordered_final_identical_and_empty_turns() -> None:
    session, websocket = _session()
    writer = session._send_task
    scope = session.speaker_session_scope

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
    session._handle_message(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))
    first_terminal = await _next(session)
    assert isinstance(first_terminal, STTProviderTurnTerminal)
    assert (first_terminal.text, first_terminal.epoch_disposition) == ("same", "reuse")
    assert first_terminal.failure_retryable is False
    assert first_terminal.recovery_pending is False
    assert [
        (run.text, run.speaker_id, run.session_scope) for run in first_terminal.final_speaker_runs
    ] == [("same", "7", scope)]

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
    second_update = await _next(session)
    second_terminal = await _next(session)
    assert isinstance(second_update, STTProviderTurnUpdate)
    assert isinstance(second_terminal, STTProviderTurnTerminal)
    assert second_terminal.text == "same"
    assert second_terminal.identity != first_terminal.identity
    assert second_terminal.epoch_disposition == "reuse"
    assert second_terminal.final_speaker_runs[0].session_scope == scope

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
@pytest.mark.parametrize(
    ("late_tokens", "reason"),
    [
        ([{"text": "late", "is_final": True, "end_ms": 500}], "soniox_idle_authoritative_text"),
        ([{"text": "<fin>", "is_final": True}], "soniox_protocol_ambiguity"),
    ],
)
async def test_soniox_detectable_idle_output_retires_reused_epoch_without_seeding_next_turn(
    late_tokens: list[dict[str, object]],
    reason: str,
) -> None:
    session, _ = _session()
    request = _request(1)
    await session.begin_turn(request)
    await _seal(session, request)
    session._handle_message(json.dumps({"tokens": [{"text": "<fin>", "is_final": True}]}))
    terminal = await _next(session)
    assert isinstance(terminal, STTProviderTurnTerminal)
    assert terminal.epoch_disposition == "reuse"

    session._handle_message(json.dumps({"tokens": late_tokens}))
    ended = await _next(session)
    assert isinstance(ended, STTProviderEpochEnded)
    assert ended.reason == reason
    assert ended.failure_retryable is False
    assert session._pending_tokens == []
    with pytest.raises(RuntimeError, match="epoch is retired"):
        await session.begin_turn(_request(2))
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
    terminal = await _next(session)
    ended = await _next(session)
    assert isinstance(update, STTProviderTurnUpdate)
    assert update.text == "kept"
    assert isinstance(terminal, STTProviderTurnTerminal)
    assert (
        terminal.outcome,
        terminal.text,
        terminal.text_authority,
        terminal.failure_reason,
        terminal.epoch_disposition,
    ) == ("degraded", "kept", "degraded", "soniox_stream_finished", "retire")
    assert terminal.final_language_runs[0].text == "kept"
    assert terminal.final_speaker_runs[0].text == "kept"
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
