from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import pytest

from puripuly_heart.core.audio.ownership import (
    AudioSegmentIdentity,
    AudioSegmentSettingsSnapshot,
)
from puripuly_heart.core.stt.backend import (
    STTProviderEpochEnded,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTSessionProjection,
)
from puripuly_heart.core.stt.session_projection import STTSessionEventProjection


def _request(epoch: str = "epoch-1") -> STTProviderTurnRequest:
    return STTProviderTurnRequest(
        identity=STTProviderTurnIdentity(
            segment=AudioSegmentIdentity(
                activation_generation=1,
                segment_order=1,
                segment_id=uuid4(),
                capture_epoch=1,
            ),
            provider_epoch_id=epoch,
            provider_turn_id="turn-1",
        ),
        settings=AudioSegmentSettingsSnapshot(
            provider_id="test",
            provider_signature=("test",),
            runtime_signature=("test",),
            source_mode="desktop",
            source_language="en",
            expected_languages=("en",),
            target_sample_rate_hz=16_000,
            vad_speech_threshold=0.4,
            vad_hangover_ms=800,
            vad_pre_roll_ms=500,
        ),
    )


@pytest.mark.asyncio
async def test_projection_allocates_only_selected_stream_and_rejects_wrong_consumer() -> None:
    legacy = STTSessionEventProjection(STTSessionProjection())
    scoped = STTSessionEventProjection(
        STTSessionProjection(mode="scoped", provider_epoch_id="epoch-1")
    )

    assert legacy._legacy_events is not None
    assert legacy._scoped_events is None
    assert scoped._legacy_events is None
    assert scoped._scoped_events is not None

    with pytest.raises(RuntimeError, match="scoped STT events are unavailable"):
        await legacy.turn_events().__anext__()
    with pytest.raises(RuntimeError, match="legacy STT events are unavailable"):
        await scoped.events().__anext__()

    legacy.close()
    scoped.close()


@pytest.mark.asyncio
async def test_scoped_projection_owns_sequences_terminal_and_epoch_end_once() -> None:
    projection = STTSessionEventProjection(
        STTSessionProjection(mode="scoped", provider_epoch_id="epoch-1")
    )
    request = _request()
    projection.begin(request)

    projection.validate_payload(request.identity, 1)
    projection.payload_written(request.identity, 1)
    with pytest.raises(ValueError, match="contiguous"):
        projection.validate_payload(request.identity, 3)
    projection.seal(request.identity)

    terminal = STTProviderTurnTerminal(
        identity=request.identity,
        outcome="failed",
        text_authority="none",
        failure_reason="provider_failed",
        epoch_disposition="retire",
    )
    assert projection.terminal(terminal) is True
    assert projection.terminal(terminal) is False
    assert (
        projection.end_epoch(
            orderly=False,
            reason="provider_failed",
            provider_turn_id=request.identity.provider_turn_id,
        )
        is True
    )
    assert projection.end_epoch(orderly=False, reason="duplicate") is False

    stream = projection.turn_events()
    assert await stream.__anext__() == terminal
    ended = await stream.__anext__()
    assert isinstance(ended, STTProviderEpochEnded)
    assert ended.provider_epoch_id == "epoch-1"
    assert ended.reason == "provider_failed"

    with pytest.raises(RuntimeError, match="epoch is retired"):
        projection.begin(_request())
    projection.close()


@pytest.mark.asyncio
async def test_overlap_projection_keeps_sealed_identity_until_its_own_terminal() -> None:
    projection = STTSessionEventProjection(
        STTSessionProjection(mode="scoped", provider_epoch_id="epoch-1"),
        allows_sealed_turn_overlap=True,
    )
    first = _request()
    second = replace(
        first,
        identity=replace(
            first.identity,
            segment=replace(
                first.identity.segment,
                segment_order=2,
                segment_id=uuid4(),
            ),
            provider_turn_id="turn-2",
        ),
    )

    projection.begin(first)
    projection.payload_written(first.identity, 1)
    projection.seal(first.identity)
    projection.begin(second)
    projection.payload_written(second.identity, 1)
    projection.seal(second.identity)

    second_terminal = STTProviderTurnTerminal(
        identity=second.identity,
        outcome="final",
        text="second",
        text_authority="authoritative",
    )
    first_terminal = replace(
        second_terminal,
        identity=first.identity,
        text="first",
    )
    assert projection.terminal(second_terminal)
    assert projection.is_current(first.identity)
    assert projection.terminal(first_terminal)

    stream = projection.turn_events()
    assert await stream.__anext__() == second_terminal
    assert await stream.__anext__() == first_terminal
    projection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("retirement", ["retire", "end_epoch"])
async def test_explicit_retirement_rejects_late_callbacks(retirement: str) -> None:
    projection = STTSessionEventProjection(
        STTSessionProjection(mode="scoped", provider_epoch_id="epoch-1")
    )
    request = _request()
    projection.begin(request)
    projection.seal(request.identity)

    if retirement == "retire":
        projection.retire()
    else:
        assert projection.end_epoch(orderly=False, reason="connection_closed")

    terminal = STTProviderTurnTerminal(
        identity=request.identity,
        outcome="final",
        text="late",
        text_authority="authoritative",
    )
    assert projection.is_current(request.identity) is False
    assert projection.terminal(terminal) is False
    projection.close()
