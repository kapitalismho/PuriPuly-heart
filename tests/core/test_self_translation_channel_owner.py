from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from puripuly_heart.core.audio.ownership import AudioSegmentIdentity
from puripuly_heart.core.orchestrator.channel_runtime import ContextEntry, _MergeBuffer
from puripuly_heart.core.stt.backend import (
    STTProviderTurnIdentity,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTTextContribution,
)
from puripuly_heart.domain.events import STTFinalEvent
from puripuly_heart.domain.models import Transcript
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import (
    compose_translation_test_harness,
    make_speculative_attempt,
)


@pytest.mark.asyncio
async def test_self_owner_rejects_closed_ingress_and_non_self_events() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=RecordingOscQueue())
    owner = harness.self_owner
    utterance_id = uuid4()

    await owner.close_ingress()
    with pytest.raises(RuntimeError, match="closed"):
        await owner.submit_text("closed")

    await owner.open_ingress()
    with pytest.raises(ValueError, match="non-Self"):
        await owner.handle_stt_event(
            STTFinalEvent(
                utterance_id,
                Transcript(
                    utterance_id,
                    "peer",
                    is_final=True,
                    channel="peer",
                ),
            )
        )


@pytest.mark.asyncio
async def test_self_provider_reset_cancels_speech_without_erasing_manual_or_peer_state() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=RecordingOscQueue())
    owner = harness.self_owner
    speech_id = uuid4()
    manual_id = uuid4()
    peer_id = uuid4()
    owner.runtime.get_or_create_bundle(speech_id)
    owner.runtime.remember_source(speech_id, "Mic")
    owner.runtime.get_or_create_bundle(manual_id)
    owner.runtime.remember_source(manual_id, "You")
    owner.runtime.translation_history.append(ContextEntry("manual", "ko", "en", 1.0))
    harness.peer_runtime.get_or_create_bundle(peer_id)
    harness.peer_runtime.translation_history.append(
        ContextEntry("peer", "ko", "en", 1.0, channel="peer")
    )

    await owner.reset_provider_channel("self")

    assert speech_id not in owner.runtime.utterances
    assert manual_id in owner.runtime.utterances
    assert [entry.text for entry in owner.runtime.translation_history] == ["manual"]
    assert peer_id in harness.peer_runtime.utterances
    assert [entry.text for entry in harness.peer_runtime.translation_history] == ["peer"]


@pytest.mark.asyncio
async def test_scoped_request_keeps_suffix_after_early_publication() -> None:
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
    )
    segment_id = uuid4()
    identity = STTProviderTurnIdentity(
        segment=AudioSegmentIdentity(1, 1, segment_id, 1),
        provider_epoch_id="epoch",
        provider_turn_id="turn",
    )
    contribution_a = STTTextContribution("a", 0, 1)
    contribution_b = STTTextContribution("b", 1, 2)

    await harness.self_owner.handle_stt_event(
        STTProviderTurnUpdate(identity, 1, "stable", "append", "A", contribution=contribution_a)
    )
    first = harness.self_owner.merge_buffer
    assert first is not None
    first.awaiting_vad_end = False
    first.awaiting_vad_utterance_id = None
    await harness.self_owner._commit_merge(first, reason="awaiting_vad_timeout")

    await harness.self_owner.handle_stt_event(
        STTProviderTurnUpdate(identity, 2, "stable", "append", "AB", contribution=contribution_b)
    )
    successor = harness.self_owner.merge_buffer
    assert successor is not None
    assert successor.parts == ["B"]
    assert successor.utterance_ids != first.utterance_ids

    await harness.self_owner.handle_stt_event(
        STTProviderTurnTerminal(
            identity,
            "final",
            text="AB",
            text_authority="authoritative",
            included_contributions=(contribution_a, contribution_b),
        )
    )
    await harness.self_owner.handle_stt_event(
        STTProviderTurnTerminal(
            identity,
            "final",
            text="AB",
            text_authority="authoritative",
            included_contributions=(contribution_a, contribution_b),
        )
    )

    assert successor.parts == ["B"]
    assert identity not in harness.self_owner._scoped_publication_ids


@pytest.mark.asyncio
async def test_self_owner_close_cancels_and_awaits_owned_runtime_tasks() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=RecordingOscQueue())
    owner = harness.self_owner
    utterance_id = uuid4()
    translation_task = asyncio.create_task(asyncio.sleep(60.0))
    spec_task = asyncio.create_task(asyncio.sleep(60.0))
    finalize_task = asyncio.create_task(asyncio.sleep(60.0))
    owner.runtime.translation_tasks[utterance_id] = translation_task
    owner.merge_buffer = _MergeBuffer(
        merge_id=uuid4(),
        utterance_ids=[utterance_id],
        speculative_attempt=make_speculative_attempt(task=spec_task),
        finalize_wait_task=finalize_task,
    )

    await owner.close()

    assert translation_task.done()
    assert spec_task.done()
    assert finalize_task.done()
    assert owner.merge_buffer is None
    assert not owner.accepting_events


def test_self_owner_requires_self_runtime() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=RecordingOscQueue())
    owner = harness.self_owner

    with pytest.raises(ValueError, match="Self channel"):
        type(owner)(
            runtime=harness.peer_runtime,
            config_snapshot=owner.config_snapshot,
            translation_turns=owner.translation_turns,
            local_asr_runtime=owner.local_asr_runtime,
            translation_requests=owner.translation_requests,
            output_projection=owner.output_projection,
            diagnostics=owner.diagnostics,
            clock=owner.clock,
        )
