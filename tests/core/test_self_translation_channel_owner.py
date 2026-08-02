from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from puripuly_heart.core.orchestrator.channel_runtime import ContextEntry, _MergeBuffer
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
async def test_self_owner_reset_clears_only_self_translation_state() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=RecordingOscQueue())
    owner = harness.self_owner
    self_id = uuid4()
    peer_id = uuid4()
    owner.runtime.get_or_create_bundle(self_id)
    owner.runtime.translation_history.append(ContextEntry("self", "ko", "en", 1.0, channel="self"))
    harness.peer_runtime.get_or_create_bundle(peer_id)
    harness.peer_runtime.translation_history.append(
        ContextEntry("peer", "ko", "en", 1.0, channel="peer")
    )

    await owner.reset_provider_channel("self")

    assert owner.runtime.utterances == {}
    assert owner.runtime.translation_history == []
    assert peer_id in harness.peer_runtime.utterances
    assert [entry.text for entry in harness.peer_runtime.translation_history] == ["peer"]


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
