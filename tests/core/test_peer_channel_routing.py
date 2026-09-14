from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.domain.events import STTFinalEvent, UIEventType
from puripuly_heart.domain.models import FinalLanguageRun, Transcript, Translation
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness


@dataclass(slots=True)
class FakeLLM:
    calls: list[str] = field(default_factory=list)

    async def translate(
        self,
        *,
        utterance_id,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
        scene_participant_count: int | None = None,
    ) -> Translation:
        _ = (utterance_id, system_prompt, source_language, target_language, context)
        self.calls.append(text)
        return Translation(utterance_id=utterance_id, text="translated")

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_peer_desktop_transcripts_are_routed_to_peer_runtime_and_never_sent_to_chatbox() -> (
    None
):
    osc = RecordingOscQueue()
    harness = compose_translation_test_harness(
        stt=None, llm=None, osc=osc, clock=FakeClock(_now=10.0)
    )

    utterance_id = await harness.handle_peer_transcript_final_for_test(
        text="peer line",
    )

    bundle = harness.bundle_for(utterance_id, channel="peer")
    event = await harness.ui_events.get()

    assert bundle.final is not None
    assert bundle.final.channel == "peer"
    assert bundle.final.text == "peer line"
    assert osc.messages == []
    assert event.type == UIEventType.TRANSCRIPT_FINAL
    assert event.channel == "peer"


@pytest.mark.asyncio
async def test_peer_final_runs_owner_creates_ordered_children_for_language_runs() -> None:
    harness = compose_translation_test_harness(
        stt=None, llm=None, osc=RecordingOscQueue(), clock=FakeClock(_now=10.0)
    )
    parent_utterance_id = uuid4()
    runs = (
        FinalLanguageRun(text="日本語", language="ja"),
        FinalLanguageRun(text="中文", language="zh"),
    )

    child_ids = await harness.translation_turns.submit_parent(
        harness.admit_peer_transcript_for_test(
            Transcript(
                utterance_id=parent_utterance_id,
                text="日本語中文",
                is_final=True,
                channel="peer",
                final_language_runs=runs,
            )
        ),
        source="Peer",
    )

    assert len(child_ids) == 2
    assert parent_utterance_id not in child_ids
    assert [
        harness.peer_runtime.utterances[child_id].final.final_language_runs
        for child_id in child_ids
    ] == [
        (runs[0],),
        (runs[1],),
    ]


@pytest.mark.asyncio
async def test_peer_final_event_preserves_language_runs_at_the_current_consumer_boundary() -> None:
    harness = compose_translation_test_harness(
        stt=None, llm=None, osc=RecordingOscQueue(), clock=FakeClock(_now=10.0)
    )
    parent_utterance_id = uuid4()
    runs = (FinalLanguageRun(text="中文", language="zh"),)
    transcript = harness.admit_peer_transcript_for_test(
        Transcript(
            utterance_id=parent_utterance_id,
            text="中文",
            is_final=True,
            channel="peer",
            final_language_runs=runs,
        )
    )

    await harness.dispatch_stt_event(STTFinalEvent(parent_utterance_id, transcript))
    await harness.output_runtime.wait_for_peer_output_idle()

    event = await harness.ui_events.get()
    assert event.type == UIEventType.TRANSCRIPT_FINAL
    assert isinstance(event.payload, Transcript)
    assert event.payload.final_language_runs == runs


@pytest.mark.asyncio
async def test_integrated_context_always_includes_peer_entries() -> None:
    clock = FakeClock(_now=112.0)
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        clock=clock,
        integrated_context_enabled=True,
        peer_translation_enabled=True,
    )
    harness.replace_configuration(source_language="en")
    harness.replace_configuration(target_language="ko")
    harness.self_runtime.remember_context(
        "self line",
        timestamp=100.0,
        source_language="en",
        target_language="ko",
    )
    harness.peer_runtime.remember_context(
        "peer line",
        timestamp=105.0,
        source_language="en",
        target_language="ko",
    )

    context, mode = harness.translation_requests.context_resolver.resolve_for_request(
        runtime=harness.self_runtime,
        other_runtime=harness.peer_runtime,
        source_language="en",
        target_language="ko",
    )

    assert mode == "integrated"
    assert "self line" in context
    assert "peer line" in context


def test_integrated_context_includes_opposite_direction_peer_entries() -> None:
    clock = FakeClock(_now=112.0)
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        clock=clock,
        integrated_context_enabled=True,
        peer_translation_enabled=True,
    )
    harness.replace_configuration(source_language="ko")
    harness.replace_configuration(target_language="en")
    harness.replace_configuration(peer_source_language="en")
    harness.replace_configuration(peer_target_language="ko")
    harness.remember_context("self previous", timestamp=100.0, runtime=harness.self_runtime)
    harness.remember_context("peer previous", timestamp=105.0, runtime=harness.peer_runtime)

    _, self_context, _, self_mode = harness.prepare_translation_request_with_mode(
        "self current",
        runtime=harness.self_runtime,
    )
    _, peer_context, _, peer_mode = harness.prepare_translation_request_with_mode(
        "peer current",
        runtime=harness.peer_runtime,
    )

    assert self_mode == "integrated"
    assert peer_mode == "integrated"
    assert self_context == ('- [self] "self previous"\n- [peer] "peer previous"')
    assert peer_context == ('- [self] "self previous"\n- [peer] "peer previous"')


@pytest.mark.asyncio
async def test_peer_translation_respects_master_translation_toggle() -> None:
    llm = FakeLLM()
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=RecordingOscQueue(),
        clock=FakeClock(_now=10.0),
        translation_enabled=False,
        peer_translation_enabled=True,
    )

    utterance_id = await harness.handle_peer_transcript_final_for_test(text="peer line")
    bundle = harness.bundle_for(utterance_id, channel="peer")
    event = await harness.ui_events.get()

    assert event.type == UIEventType.TRANSCRIPT_FINAL
    assert bundle.translation is None
    assert llm.calls == []
