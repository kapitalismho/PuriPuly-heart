from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from puripuly_heart.core.orchestrator.peer_translation_channel import (
    PeerTranslationChannelOwner,
)
from puripuly_heart.domain.events import STTFinalEvent
from puripuly_heart.domain.models import Transcript
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness, owned_peer_speech_end


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
        await owner.handle_peer_owned_vad_event(owned_peer_speech_end(uuid4(), speech_end_at=0.0))

    await owner.open_ingress()
    await owner.handle_stt_event(object())
    assert owner.accepting_events is True


@pytest.mark.asyncio
async def test_peer_owned_speech_end_uses_source_ledger_seal_time() -> None:
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
    )
    owner = harness.peer_owner

    class Runtime:
        async def handle_owned_vad_event(self, channel: str, event: object) -> None:
            return None

        async def commit_handoff(self, channel: str) -> None:
            return None

    owner.local_asr_runtime = Runtime()
    utterance_id = uuid4()
    owned = owned_peer_speech_end(utterance_id, speech_end_at=2.5)

    await owner.handle_peer_owned_vad_event(owned)

    assert owner.runtime.utterance_start_times[utterance_id] == 2.5
    assert utterance_id in owner.runtime.speech_ended_ids


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


@pytest.mark.asyncio
async def test_retired_generation_blocks_cancellation_source_only_during_translation() -> None:
    class RecordingOverlay:
        def __init__(self) -> None:
            self.events: list[object] = []

        async def emit(self, event: object) -> None:
            self.events.append(event)

        def active_self_overlay_metadata(self) -> None:
            return None

    overlay = RecordingOverlay()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=overlay,
    )
    started = asyncio.Event()

    async def blocked_process(_child, _cancellation_requested):
        started.set()
        await asyncio.Event().wait()

    harness.translation_turns.process_child = blocked_process
    harness.output_runtime.activate_peer_generation(1)
    await harness.start()
    parent_id = uuid4()
    completion = asyncio.create_task(
        harness.peer_owner.handle_stt_event(
            STTFinalEvent(
                utterance_id=parent_id,
                transcript=Transcript(
                    utterance_id=parent_id,
                    text="must not publish after off",
                    is_final=True,
                    channel="peer",
                    publication_generation=1,
                    source_order=1,
                ),
            )
        )
    )
    await asyncio.wait_for(started.wait(), timeout=0.5)
    harness.output_runtime.retire_peer_generation(1)
    harness.output_runtime.activate_peer_generation(2)
    assert harness.output_runtime.peer_publication_is_authorized(2, 1)
    await harness.translation_turns.cancel_pending(channel="peer")
    await asyncio.wait_for(completion, timeout=0.5)
    await harness.output_runtime.wait_for_peer_output_idle()

    assert overlay.events == []
    assert any(
        decision.reason == "publication_generation_retired"
        for decision in harness.output_runtime.routing_decisions
    )
    await harness.stop()
