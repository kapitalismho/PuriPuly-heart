from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest

from puripuly_heart.config.prompts import get_default_prompt
from puripuly_heart.core.audio.ownership import AudioSegmentTerminalReceipt
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.orchestrator.channel_runtime import ContextEntry
from puripuly_heart.core.orchestrator.peer_caption import join_action, preview_source
from puripuly_heart.core.stt.backend import (
    STTProviderTurnIdentity,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
)
from puripuly_heart.core.translation_backend import TranslationBackend
from puripuly_heart.domain.events import STTFinalEvent, UIEventType
from puripuly_heart.domain.models import Transcript, Translation
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness, owned_peer_speech_end


class ControlledBackend(TranslationBackend):
    def __init__(self):
        self.calls = []
        self.block = False
        self.completions = []

    async def translate(self, request):
        self.calls.append(request)
        if self.block:
            completion = asyncio.get_running_loop().create_future()
            self.completions.append(completion)
            await completion
        return Translation(utterance_id=request.utterance_id, text=f"translated:{request.text}")

    async def close(self):
        return None


async def turn_loop():
    for _ in range(12):
        await asyncio.sleep(0)


def setup():
    backend = ControlledBackend()
    events = []

    async def emit(event):
        events.append(event)

    clock = FakeClock()
    harness = compose_translation_test_harness(
        stt=None,
        llm=backend,
        osc=RecordingOscQueue(),
        clock=clock,
        overlay_sink=SimpleNamespace(emit=emit, active_self_overlay_metadata=lambda: None),
        source_language="en",
        target_language="ko",
        system_prompt=get_default_prompt(),
        peer_translation_enabled=True,
    )
    return harness, backend, events, clock


def transcript(text, order, *, identity=None, final=False):
    return Transcript(
        identity or uuid4(),
        text,
        final,
        channel="peer",
        publication_generation=1,
        source_order=order,
    )


async def partial(harness, item, provider="soniox"):
    await harness.peer_owner.cloud_captions.partial(
        item, provider_id=provider, scope=(1, 0, "epoch"), config=harness.configuration.snapshot()
    )


async def final(harness, item, provider="soniox", sealed=None):
    await harness.peer_owner.cloud_captions.final(
        replace(item, is_final=True),
        provider_id=provider,
        scope=(1, 0, "epoch"),
        config=harness.configuration.snapshot(),
        sealed=sealed,
    )


async def idle(harness):
    await harness.peer_owner.cloud_captions.wait_for_idle()
    await harness.output_runtime.wait_for_peer_output_idle()
    await turn_loop()


@pytest.mark.asyncio
async def test_preview_first_refresh_growth_gate_and_final_settlement():
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        item = transcript("The observed beginning", 1)
        await partial(harness, item)
        clock.advance(4.99)
        await harness.peer_owner.cloud_captions.check_previews()
        assert backend.calls == []
        clock.advance(0.01)
        await harness.peer_owner.cloud_captions.check_previews()
        await idle(harness)
        assert len(backend.calls) == 1
        assert events[-1].type == "translation_stream_update"
        assert events[-1].peer_caption_revision == 1
        assert harness.peer_runtime.translation_history == []
        clock.advance(4)
        await harness.peer_owner.cloud_captions.check_previews()
        await idle(harness)
        assert len(backend.calls) == 1
        await partial(harness, replace(item, text="The observed beginning and later content"))
        clock.advance(4)
        await harness.peer_owner.cloud_captions.check_previews()
        await idle(harness)
        assert len(backend.calls) == 2
        assert events[-1].utterance_id == item.utterance_id
        assert events[-1].peer_caption_revision == 2
        await final(harness, replace(item, text="The complete corrected sentence."))
        await idle(harness)
        assert len(backend.calls) == 3
        assert events[-1].type == "translation_final"
        assert events[-1].peer_caption_revision == 3
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            "The complete corrected sentence."
        ]
        assert harness.osc.messages == []
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_final_cancels_slow_preview_without_waiting_for_preview_timer():
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        backend.block = True
        item = transcript("unfinished observed source", 1)
        await partial(harness, item)
        clock.advance(5)
        await harness.peer_owner.cloud_captions.check_previews()
        await turn_loop()
        assert len(backend.calls) == 1
        backend.block = False
        await final(harness, replace(item, text="Final source."))
        await idle(harness)
        assert [event.type for event in events] == ["translation_final"]
        assert backend.completions[0].cancelled()
        assert len(backend.calls) == 2
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_merge_preserves_first_context_and_absorbs_preview_atomically(monkeypatch):
    monkeypatch.setattr(
        "puripuly_heart.core.orchestrator.peer_caption.estimate_peer_translation_lines",
        lambda *_: 2.0,
    )
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        harness.peer_runtime.translation_history.append(
            ContextEntry("Earlier.", "en", "ko", 0, "peer")
        )
        first = transcript("The first part,", 1)
        await final(harness, first)
        await idle(harness)
        clock.advance(0.5)
        second = transcript("and the second part", 2)
        await partial(harness, second)
        clock.advance(5)
        await harness.peer_owner.cloud_captions.check_previews()
        await idle(harness)
        assert events[-1].utterance_id == second.utterance_id
        backend.block = True
        await final(harness, second)
        await turn_loop()
        assert events[-1].peer_superseded_unit_ids == ()
        backend.completions[-1].set_result(None)
        await idle(harness)
        repair = events[-1]
        assert repair.type == "translation_final"
        assert repair.utterance_id == first.utterance_id
        assert repair.source_text == "The first part, and the second part"
        assert repair.peer_superseded_unit_ids == (second.utterance_id,)
        assert backend.calls[-1].context == '- [peer] "Earlier."'
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            "Earlier.",
            repair.source_text,
        ]
    finally:
        await harness.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("r1_only", [False, True])
async def test_join_above_two_lines_uses_continuation_and_current_context(monkeypatch, r1_only):
    monkeypatch.setattr(
        "puripuly_heart.core.orchestrator.peer_caption.estimate_peer_translation_lines",
        lambda *_: 3.0,
    )
    harness, backend, events, clock = setup()
    sealed = replace(
        owned_peer_speech_end(uuid4(), speech_end_at=1).segment, seal_reason="delivery_deadline"
    )
    try:
        await harness.start()
        first = transcript("A closed statement." if r1_only else "The earlier part,", 1)
        await final(harness, first, sealed=sealed)
        await idle(harness)
        clock.advance(0.3)
        second = transcript("The next content.", 2)
        await partial(harness, second)
        await final(harness, second)
        await idle(harness)
        assert events[-1].utterance_id == second.utterance_id
        assert backend.calls[-1].text == second.text
        assert "## Peer continuation" in backend.calls[-1].system_prompt
        assert backend.calls[-1].context == f'- [peer] "{first.text}"'
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            first.text,
            second.text,
        ]
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_r1_only_over_four_lines_is_independent(monkeypatch):
    monkeypatch.setattr(
        "puripuly_heart.core.orchestrator.peer_caption.estimate_peer_translation_lines",
        lambda *_: 4.01,
    )
    harness, backend, events, clock = setup()
    sealed = replace(
        owned_peer_speech_end(uuid4(), speech_end_at=1).segment, seal_reason="delivery_deadline"
    )
    try:
        await harness.start()
        await final(harness, transcript("A closed statement.", 1), sealed=sealed)
        await idle(harness)
        clock.advance(0.5)
        second = transcript("The next statement.", 2)
        await partial(harness, second)
        await final(harness, second)
        await idle(harness)
        assert events[-1].utterance_id == second.utterance_id
        assert "## Peer continuation" not in backend.calls[-1].system_prompt
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_chained_repairs_out_of_order_keep_cumulative_absorptions(monkeypatch):
    monkeypatch.setattr(
        "puripuly_heart.core.orchestrator.peer_caption.estimate_peer_translation_lines",
        lambda *_: 2.0,
    )
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        first = transcript("A,", 1)
        await final(harness, first)
        await idle(harness)
        backend.block = True
        second = transcript("second,", 2)
        await partial(harness, second)
        await final(harness, second)
        await turn_loop()
        third = transcript("third.", 3)
        await partial(harness, third)
        await final(harness, third)
        await turn_loop()
        backend.completions[1].set_result(None)
        await turn_loop()
        backend.completions[0].set_result(None)
        await idle(harness)
        assert len(events) == 2
        assert events[-1].source_text == "A, second, third."
        assert events[-1].peer_superseded_unit_ids == (second.utterance_id, third.utterance_id)
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            events[-1].source_text
        ]
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_preview_coalesces_one_running_one_latest_and_reset_fences_work():
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        backend.block = True
        item = transcript("first observed content", 1)
        await partial(harness, item)
        for suffix in ("", " longer", " longer newest"):
            await partial(harness, replace(item, text=item.text + suffix))
            clock.advance(5)
            await harness.peer_owner.cloud_captions.check_previews()
            await turn_loop()
        assert len(backend.calls) == 1
        backend.completions[0].set_result(None)
        await turn_loop()
        assert len(backend.calls) == 2
        assert backend.calls[-1].text.endswith("newest")
        await harness.peer_owner.cloud_captions.reset()
        assert backend.completions[1].cancelled()
        assert harness.peer_runtime.translation_history == []
        assert len(events) <= 1
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_owned_provider_ingress_through_output_contract():
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        item = transcript("Observed provisional speech", 1)
        owned = owned_peer_speech_end(item.utterance_id, speech_end_at=9)
        segment = replace(
            owned.segment, settings=replace(owned.segment.settings, provider_id="soniox")
        )
        identity = STTProviderTurnIdentity(segment.identity, "epoch", "turn", ("soniox",))
        await harness.peer_owner.handle_provider_update(
            STTProviderTurnUpdate(identity, 1, "provisional", "replace", item.text, observed_at=0.0)
        )
        clock.advance(5)
        await harness.peer_owner.cloud_captions.check_previews()
        await idle(harness)
        assert events[-1].type == "translation_stream_update"
        receipt = AudioSegmentTerminalReceipt(segment.identity, "final", segment, 9.0)
        await harness.peer_owner.handle_provider_turn_terminal(
            receipt,
            STTProviderTurnTerminal(
                identity, "final", "Authoritative corrected speech.", text_authority="authoritative"
            ),
        )
        await idle(harness)
        assert events[-1].type == "translation_final"
        assert events[0].utterance_id == events[-1].utterance_id
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            "Authoritative corrected speech."
        ]
        assert harness.osc.messages == []
    finally:
        await harness.stop()


def test_preview_boundary_selection_and_reply_gap_stops():
    assert (
        preview_source("a short, a much longer unfinished observation")
        == "a short, a much longer unfinished observation"
    )
    assert preview_source("A sufficiently long clause, tail") == "A sufficiently long clause,"
    for gap, incoming in ((None, "continuation"), (1.51, "continuation"), (0.2, "yeah okay")):
        assert (
            join_action(
                "the first,",
                incoming,
                language="en",
                gap_s=gap,
                expected_lines=1,
                provider_id="soniox",
                sealed=None,
            )
            == "independent"
        )


def test_gemini_ignores_acoustic_signals_and_question_stops_r1():
    sealed = replace(
        owned_peer_speech_end(uuid4(), speech_end_at=1).segment, seal_reason="delivery_deadline"
    )
    assert (
        join_action(
            "Completed.",
            "Next",
            language="en",
            gap_s=0.2,
            expected_lines=2,
            provider_id="gemini_transcribe",
            sealed=sealed,
        )
        == "independent"
    )
    assert (
        join_action(
            "Completed?",
            "Next",
            language="en",
            gap_s=0.2,
            expected_lines=2,
            provider_id="soniox",
            sealed=sealed,
        )
        == "independent"
    )


@pytest.mark.asyncio
async def test_continuation_caption_can_merge_and_continue_with_its_original_context(monkeypatch):
    estimate = [3.0]
    monkeypatch.setattr(
        "puripuly_heart.core.orchestrator.peer_caption.estimate_peer_translation_lines",
        lambda *_: estimate[0],
    )
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        first = transcript("The beginning,", 1)
        await final(harness, first)
        await idle(harness)
        second = transcript("the middle,", 2)
        await partial(harness, second)
        await final(harness, second)
        await idle(harness)
        assert "## Peer continuation" in backend.calls[-1].system_prompt
        estimate[0] = 2.0
        third = transcript("and more,", 3)
        await partial(harness, third)
        await final(harness, third)
        await idle(harness)
        assert events[-1].utterance_id == second.utterance_id
        assert backend.calls[-1].text == "the middle, and more,"
        assert "## Peer continuation" not in backend.calls[-1].system_prompt
        assert backend.calls[-1].context == f'- [peer] "{first.text}"'
        estimate[0] = 3.0
        fourth = transcript("the end.", 4)
        await partial(harness, fourth)
        await final(harness, fourth)
        await idle(harness)
        assert events[-1].utterance_id == fourth.utterance_id
        assert "## Peer continuation" in backend.calls[-1].system_prompt
        assert backend.calls[-1].context.splitlines()[-1] == '- [peer] "the middle, and more,"'
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            first.text,
            "the middle, and more,",
            fourth.text,
        ]
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_independent_finals_finishing_reverse_preserve_source_order_and_repetition():
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        backend.block = True
        first = transcript("Repeated wording.", 1)
        second = transcript(first.text, 2)
        await final(harness, first)
        await final(harness, second)
        await turn_loop()
        backend.completions[1].set_result(None)
        await turn_loop()
        assert events == []
        backend.completions[0].set_result(None)
        await idle(harness)
        assert [event.utterance_id for event in events] == [first.utterance_id, second.utterance_id]
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            first.text,
            second.text,
        ]
        assert harness.peer_owner.cloud_captions.request_counts == {
            "preview": 0,
            "final": 2,
            "repair": 0,
            "continuation": 0,
        }
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_context_is_five_recent_peer_finals_and_preview_context_is_frozen():
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        clock.advance(61)
        harness.peer_runtime.translation_history.append(
            ContextEntry("Expired.", "en", "ko", 0, "peer")
        )
        for index in range(7):
            harness.peer_runtime.translation_history.append(
                ContextEntry(f"Recent {index}.", "en", "ko", 60, "peer")
            )
        harness.self_runtime.translation_history.append(
            ContextEntry("Private Self.", "en", "ko", 60)
        )
        item = transcript("Observed incomplete", 1)
        await partial(harness, item)
        clock.advance(5)
        await harness.peer_owner.cloud_captions.check_previews()
        await idle(harness)
        expected = "\n".join(f'- [peer] "Recent {index}."' for index in range(2, 7))
        assert backend.calls[-1].context == expected
        harness.peer_runtime.translation_history.append(
            ContextEntry("Later context.", "en", "ko", clock.now(), "peer")
        )
        await final(harness, replace(item, text="Authoritative final."))
        await idle(harness)
        assert backend.calls[-1].context == expected
        ui = []
        while not harness.ui_events.empty():
            ui.append(harness.ui_events.get_nowait())
        assert len([event for event in ui if event.type is UIEventType.TRANSLATION_DONE]) == 1
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_empty_partial_retains_last_nonempty_and_text_capacity_is_explicit():
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        item = transcript("Observed content", 1)
        await partial(harness, item)
        await partial(harness, replace(item, text=""))
        clock.advance(5)
        await harness.peer_owner.cloud_captions.check_previews()
        await idle(harness)
        assert backend.calls[-1].text == item.text
        harness.peer_owner.cloud_captions.max_source_chars = 20
        with pytest.raises(ValueError, match="text capacity exceeded"):
            await partial(harness, replace(item, text="x" * 21))
        with pytest.raises(ValueError, match="text capacity exceeded"):
            await final(harness, replace(item, text="x" * 21))
        assert harness.peer_runtime.translation_history == []
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_local_peer_uses_unchanged_per_final_translation_without_preview_or_join(monkeypatch):
    harness, backend, events, clock = setup()
    monkeypatch.setattr(
        type(harness.peer_owner), "_current_provider_id", lambda _: "local_cpu_auto"
    )
    try:
        await harness.start()
        first = transcript("An incomplete source,", 1)
        from puripuly_heart.domain.events import STTPartialEvent

        await harness.peer_owner.handle_stt_event(STTPartialEvent(first.utterance_id, first))
        clock.advance(10)
        await harness.peer_owner.cloud_captions.check_previews()
        assert backend.calls == []
        await harness.peer_owner.handle_stt_event(
            STTFinalEvent(first.utterance_id, replace(first, is_final=True))
        )
        await harness.translation_turns.wait_for_idle()
        second = transcript("the next local final", 2)
        await harness.peer_owner.handle_stt_event(
            STTFinalEvent(second.utterance_id, replace(second, is_final=True))
        )
        await harness.translation_turns.wait_for_idle()
        await harness.output_runtime.wait_for_peer_output_idle()
        assert [call.text for call in backend.calls] == [first.text, second.text]
        assert all("## Peer provisional" not in call.system_prompt for call in backend.calls)
        assert all("## Peer continuation" not in call.system_prompt for call in backend.calls)
        assert harness.peer_owner.cloud_captions.request_counts == dict.fromkeys(
            ("preview", "final", "repair", "continuation"), 0
        )
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_provider_replacement_discards_inflight_caption_and_fresh_work_progresses():
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        backend.block = True
        await final(harness, transcript("Retired request.", 1))
        await turn_loop()
        replacement = ControlledBackend()
        await harness.replace_llm_provider(replacement)
        backend.completions[0].set_result(None)
        await idle(harness)
        assert events == []
        await final(harness, transcript("Current request.", 2))
        await idle(harness)
        assert events[-1].source_text == "Current request."
    finally:
        await harness.stop()


def test_reply_punctuation_and_overlapping_delivery_gap_follow_join_rules():
    assert (
        join_action(
            "earlier,",
            "yeah, okay!",
            language="en",
            gap_s=0.2,
            expected_lines=1,
            provider_id="soniox",
            sealed=None,
        )
        == "independent"
    )
    assert (
        join_action(
            "earlier,",
            "continued content",
            language="en",
            gap_s=-0.1,
            expected_lines=1,
            provider_id="soniox",
            sealed=None,
        )
        == "merge"
    )


@pytest.mark.asyncio
async def test_shutdown_ingress_drain_keeps_admitted_final_and_cancels_open_preview():
    harness, backend, events, clock = setup()
    try:
        await harness.start()
        backend.block = True
        await final(harness, transcript("Admitted final.", 1))
        await turn_loop()
        await partial(harness, transcript("Unsettled open source", 2))
        draining = asyncio.create_task(harness.peer_owner.close_ingress())
        await turn_loop()
        assert not draining.done()
        backend.completions[0].set_result(None)
        await draining
        await harness.output_runtime.wait_for_peer_output_idle()
        assert [event.source_text for event in events] == ["Admitted final."]
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            "Admitted final."
        ]
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_request_capacity_backpressures_without_losing_final_sources():
    from puripuly_heart.core.orchestrator.peer_caption import PeerCaptionOwner

    harness, backend, events, clock = setup()
    current = harness.peer_owner.cloud_captions
    harness.peer_owner.cloud_captions = PeerCaptionOwner(
        current.runtime, current.requests, current.output, clock, max_requests=2
    )
    third_admission = None
    try:
        await harness.start()
        backend.block = True
        for order in (1, 2):
            await final(harness, transcript(f"Final {order}.", order))
        third_admission = asyncio.create_task(final(harness, transcript("Final 3.", 3)))
        await turn_loop()
        assert len(backend.calls) == 2 and not third_admission.done()
        backend.completions[0].set_result(None)
        await third_admission
        await turn_loop()
        assert len(backend.calls) == 3
        backend.completions[1].set_result(None)
        backend.completions[2].set_result(None)
        await idle(harness)
        assert [event.source_text for event in events] == ["Final 1.", "Final 2.", "Final 3."]
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            "Final 1.",
            "Final 2.",
            "Final 3.",
        ]
    finally:
        if third_admission is not None and not third_admission.done():
            third_admission.cancel()
            await asyncio.gather(third_admission, return_exceptions=True)
        await harness.stop()


@pytest.mark.asyncio
async def test_translation_timeout_preserves_source_and_subsequent_progress():
    harness, backend, events, clock = setup()
    harness.peer_owner.cloud_captions.request_timeout_s = 0.01
    try:
        await harness.start()
        backend.block = True
        await final(harness, transcript("Timeout source.", 1))
        await idle(harness)
        assert events[0].type == "peer_transcript_final" and events[0].text == "Timeout source."
        assert backend.completions[0].cancelled()
        backend.block = False
        await final(harness, transcript("Recovered source.", 2))
        await idle(harness)
        assert events[-1].type == "translation_final"
        assert events[-1].source_text == "Recovered source."
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            "Timeout source.",
            "Recovered source.",
        ]
    finally:
        await harness.stop()
