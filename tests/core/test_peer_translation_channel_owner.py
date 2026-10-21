from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from puripuly_heart.core.audio.ownership import AudioSegmentIdentity
from puripuly_heart.core.orchestrator.peer_translation_channel import (
    PeerTranslationChannelOwner,
)
from puripuly_heart.core.orchestrator.translation_channel_callbacks import (
    TranslationChannelOwnerCallbacks,
)
from puripuly_heart.core.stt.backend import (
    STTProviderInputTerminal,
    STTProviderTurnIdentity,
    STTRecognitionUnit,
    STTRecognitionUnitTerminal,
)
from puripuly_heart.domain.events import STTFinalEvent, UIEventType
from puripuly_heart.domain.models import Transcript
from puripuly_heart.domain.recognition import RecognitionStreamIdentity, RecognitionUnitIdentity
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
async def test_independent_peer_finals_keep_receipt_order_and_distinct_identities() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=RecordingOscQueue())
    harness.output_runtime.activate_peer_generation(1)
    stream = RecognitionStreamIdentity("peer", 1, 2, "epoch", ("gemini_transcribe",))
    first = STTRecognitionUnit(RecognitionUnitIdentity(stream, uuid4(), 1), "repeat")
    second = STTRecognitionUnit(RecognitionUnitIdentity(stream, uuid4(), 2), "repeat")
    try:
        await harness.start()
        await harness.peer_owner.handle_recognition_unit(first)
        await harness.peer_owner.handle_recognition_unit(second)
        await harness.translation_turns.wait_for_idle()
        await harness.output_runtime.wait_for_peer_output_idle()

        finals = []
        while not harness.ui_events.empty():
            event = harness.ui_events.get_nowait()
            if event.channel == "peer" and event.type is UIEventType.TRANSCRIPT_FINAL:
                finals.append(event.payload)
        assert [item.source_order for item in finals] == [1, 2]
        assert [item.recognition_origins[0].identity for item in finals] == [
            first.identity,
            second.identity,
        ]
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_peer_native_input_terminals_release_local_timing_without_binding_finals() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=RecordingOscQueue())
    harness.output_runtime.activate_peer_generation(1)
    callbacks = TranslationChannelOwnerCallbacks(harness.stt_sessions)
    callbacks.bind_peer(harness.peer_owner)
    callbacks._peer_capture = SimpleNamespace(
        note_input_terminal=lambda _event: (),
        is_current_recognition_stream=lambda identity: identity == stream,
    )
    stream = RecognitionStreamIdentity("peer", 1, 0, "epoch", ("gemini_transcribe",))

    async def accept_vad(_channel: str, _event: object) -> None:
        return None

    harness.peer_owner.local_asr_runtime = SimpleNamespace(
        handle_owned_vad_event=accept_vad,
        commit_handoff=lambda _channel: accept_vad(_channel, None),
        is_current_recognition_stream=lambda _channel, identity: identity == stream,
    )
    local_ids = [uuid4() for _ in range(24)]
    native_units = [
        STTRecognitionUnit(RecognitionUnitIdentity(stream, uuid4(), index), "repeat")
        for index in (1, 2)
    ]
    try:
        await harness.start()
        for order, segment_id in enumerate(local_ids, 1):
            owned_end = owned_peer_speech_end(segment_id, speech_end_at=float(order))
            if order != 1:
                await harness.peer_owner.handle_peer_owned_vad_event(owned_end)
                assert segment_id in harness.peer_owner._peer_parent_speech_end_times
            await callbacks.peer_event_handler(
                STTProviderInputTerminal(
                    STTProviderTurnIdentity(
                        AudioSegmentIdentity(1, order, segment_id, 0),
                        "epoch",
                        f"input-{order}",
                        stream.settings_scope,
                    ),
                    "failed" if order == 1 else "submitted",
                    "peer",
                    independent_recognition=True,
                )
            )
            if order == 1:
                await harness.peer_owner.handle_peer_owned_vad_event(owned_end)
        assert harness.peer_owner._peer_parent_speech_end_times == {}
        assert harness.peer_runtime.utterance_start_times == {}
        assert not (set(local_ids) & harness.peer_runtime.speech_ended_ids)
        assert not (
            set(local_ids)
            & {key for _, key in harness.peer_owner.diagnostics.snapshot().timeline_keys}
        )
        for unit in native_units:
            await callbacks.peer_event_handler(STTRecognitionUnitTerminal(unit, "final"))
        await harness.translation_turns.wait_for_idle()
        await harness.output_runtime.wait_for_peer_output_idle()
        finals = []
        while not harness.ui_events.empty():
            item = harness.ui_events.get_nowait()
            if item.channel == "peer" and item.type is UIEventType.TRANSCRIPT_FINAL:
                finals.append(item.payload)
        assert [item.text for item in finals] == ["repeat", "repeat"]
        assert [item.source_order for item in finals] == [1, 2]
        assert [item.recognition_origins[0].identity for item in finals] == [
            unit.identity for unit in native_units
        ]
        assert all(unit.identity.unit_id not in local_ids for unit in native_units)
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_peer_legacy_logical_turn_inherits_local_speech_end_until_completed() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=RecordingOscQueue())
    owner = harness.peer_owner

    async def accept_vad(_channel: str, _event: object) -> None:
        return None

    owner.local_asr_runtime = SimpleNamespace(
        handle_owned_vad_event=accept_vad,
        commit_handoff=lambda _channel: accept_vad(_channel, None),
    )
    parent_id, peer_turn_id = uuid4(), uuid4()
    owner._register_peer_logical_turn(parent_utterance_id=parent_id, peer_turn_id=peer_turn_id)
    await owner.handle_peer_owned_vad_event(owned_peer_speech_end(parent_id, speech_end_at=4.5))
    assert owner.runtime.utterance_start_times[peer_turn_id] == 4.5
    assert peer_turn_id in owner.runtime.speech_ended_ids
    owner._complete_peer_logical_turn(peer_turn_id)
    assert parent_id not in owner.runtime.utterance_start_times
    assert parent_id not in owner._peer_parent_speech_end_times


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
