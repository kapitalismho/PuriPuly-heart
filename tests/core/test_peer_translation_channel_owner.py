from __future__ import annotations

from uuid import uuid4

import pytest

from puripuly_heart.core.orchestrator.peer_translation_channel import (
    PeerTranslationChannelOwner,
)
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness


def test_peer_owner_rejects_non_peer_runtime() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=RecordingOscQueue())
    current = harness.peer_owner

    with pytest.raises(ValueError, match="requires the Peer channel runtime"):
        PeerTranslationChannelOwner(
            runtime=harness.self_runtime,
            config_snapshot=current.config_snapshot,
            translation_turns=current.translation_turns,
            local_asr_runtime=current.local_asr_runtime,
            translation_requests=current.translation_requests,
            output_projection=current.output_projection,
            diagnostics=current.diagnostics,
            clock=current.clock,
        )


@pytest.mark.asyncio
async def test_peer_owner_rejects_stt_and_vad_after_ingress_closes() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=RecordingOscQueue())
    owner = harness.peer_owner

    await owner.close_ingress()

    with pytest.raises(RuntimeError, match="Peer translation ingress is closed"):
        await owner.handle_stt_event(object())
    with pytest.raises(RuntimeError, match="Peer translation ingress is closed"):
        await owner.handle_peer_vad_event(object())

    await owner.open_ingress()
    await owner.handle_stt_event(object())
    assert owner.accepting_events is True


@pytest.mark.asyncio
async def test_peer_owner_close_clears_runtime_logical_turns_and_latency() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=RecordingOscQueue())
    owner = harness.peer_owner
    parent_id = uuid4()
    child_id = uuid4()
    owner.runtime.get_or_create_bundle(child_id)
    owner.runtime.utterance_start_times[parent_id] = 1.0
    owner.runtime.speech_ended_ids.add(parent_id)
    owner._peer_turn_parent_ids[child_id] = parent_id
    owner._peer_parent_turn_ids[parent_id] = {child_id}
    owner._peer_completed_turn_ids.add(child_id)
    owner._peer_parent_speech_end_times[parent_id] = 1.0
    owner._peer_translation_parent_ids.add(parent_id)

    await owner.close()

    assert owner.accepting_events is False
    assert owner.runtime.utterances == {}
    assert owner.runtime.utterance_start_times == {}
    assert owner.runtime.speech_ended_ids == set()
    assert owner._peer_turn_parent_ids == {}
    assert owner._peer_parent_turn_ids == {}
    assert owner._peer_completed_turn_ids == set()
    assert owner._peer_parent_speech_end_times == {}
    assert owner._peer_translation_parent_ids == set()
    assert owner.diagnostics.snapshot().timeline_keys == frozenset()


@pytest.mark.asyncio
async def test_peer_owner_reset_and_language_clear_reject_non_peer_channels() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=RecordingOscQueue())
    owner = harness.peer_owner

    with pytest.raises(ValueError, match="cannot reset a non-Peer channel"):
        await owner.reset_provider_channel("self")
    with pytest.raises(ValueError, match="cannot clear a non-Peer channel"):
        await owner.clear_language_runtime_state(channel="self")
