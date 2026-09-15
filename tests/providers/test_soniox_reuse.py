from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import pytest

from puripuly_heart.core.audio.ownership import AudioSegmentIdentity, AudioSegmentSettingsSnapshot
from puripuly_heart.core.stt.backend import (
    STTProviderEpochEnded,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTSessionProjection,
)
from puripuly_heart.providers.stt.soniox import _SonioxSession


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
