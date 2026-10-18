from __future__ import annotations

import asyncio
from dataclasses import replace
from uuid import uuid4

import pytest

from puripuly_heart.core.stt.backend import (
    STTContributionConsumptionLedger,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTSourceRelease,
    STTTextContribution,
)
from puripuly_heart.core.vad.gating import SpeechEnd
from puripuly_heart.domain.events import STTFinalEvent
from puripuly_heart.domain.models import Transcript
from puripuly_heart.domain.recognition import (
    NativeTranscriptionEvidence,
    RecognitionStreamIdentity,
    RecognitionTextOrigin,
    RecognitionUnitIdentity,
    recognition_separator_origin,
    validate_recognition_origins,
)
from tests.core.test_output_owner_wiring import RecordingOverlay
from tests.core.test_source_context_contract import ControlledProvider, OutputTrace, source_fixture
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.self_stream import Provider, begin, until
from tests.helpers.translation_owners import compose_translation_test_harness


def origin(text, *, sentence="initial", identity=None):
    identity = identity or RecognitionUnitIdentity(
        RecognitionStreamIdentity("self", 1, 7, "native", ()), uuid4(), 1
    )
    return RecognitionTextOrigin(
        identity,
        0,
        len(text),
        NativeTranscriptionEvidence(native_task_id="task", native_sentence_id=sentence),
        source_binding="local_segment",
        order_basis="source_order",
    )


@pytest.mark.asyncio
async def test_origin_bearing_draft_manual_and_two_targets_register_only_canonical_source():
    provider = Provider()
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=RecordingOscQueue(),
        source_language="ko",
        self_target_languages=("en", "ja"),
        low_latency_mode=True,
        low_latency_finalize_wait_ms=500,
    )
    trace = OutputTrace(harness.translation_turns.output)
    harness.translation_turns.output = trace
    old_text, final_text = "draft source", "draft source."
    try:
        for target in ("en", "ja"):
            provider.release("accepted seed", target)
        seed = await harness.self_owner.submit_text("accepted seed")
        await harness.translation_turns.wait_for_parent(seed.turn_id)
        identity = await begin(harness, old_text)
        old_origin = origin(old_text)
        await harness.dispatch_stt_event(
            STTFinalEvent(
                identity, Transcript(identity, old_text, True, recognition_origins=(old_origin,))
            )
        )
        parent_id = harness.self_owner.merge_buffer.merge_id
        await harness.self_owner.handle_vad_event(SpeechEnd(identity))
        await until(lambda: len([call for call in provider.calls if call["text"] == old_text]) == 2)
        operations = dict(harness.translation_turns._parents[parent_id].operations)
        latest_origin = origin(final_text, sentence="latest", identity=old_origin.identity)
        await harness.dispatch_stt_event(
            STTFinalEvent(
                identity,
                Transcript(identity, final_text, True, recognition_origins=(latest_origin,)),
            )
        )
        for target in ("en", "ja"):
            provider.release("intervening manual", target)
        manual = await harness.self_owner.submit_text("intervening manual")
        await harness.translation_turns.wait_for_parent(manual.turn_id)
        assert [entry.text for entry in harness.self_runtime.translation_history] == [
            "accepted seed",
            "intervening manual",
        ]
        await until(lambda: harness.self_owner.merge_buffer is None)
        parent = harness.translation_turns._parents[parent_id]
        assert all(parent.operations[index] is task for index, task in operations.items())
        sources = [child.source_child for child in parent.children]
        assert sources[0] is sources[1]
        source = sources[0]
        assert source.raw_text == final_text
        assert source.origins == (latest_origin,)
        assert source.revision == parent.children[0].transcript.source_text_revision
        assert all(
            child.prepared_request.context_texts == ("accepted seed", "intervening manual")
            for child in parent.children
        )
        assert all(
            call["context"].count(old_text) == 0 and "intervening manual" not in call["context"]
            for call in provider.calls
            if call["text"] == old_text
        )
        assert [entry.text for entry in harness.self_runtime.translation_history] == [
            "accepted seed",
            "intervening manual",
            final_text,
        ]
        assert harness.self_runtime.translation_history[-1].target_languages == ("en", "ja")
        for target in ("en", "ja"):
            provider.release(old_text, target)
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 2)
        finalized = [item for item in trace.items if item.parent_utterance_id == parent_id]
        assert len(finalized) == 2
        assert all(
            item.source_child is source and item.translation.source_child is source
            for item in finalized
        )
        assert all(
            item.source_text == item.translation.source_text == final_text for item in finalized
        )
        assert all(item.context_texts == ("accepted seed",) for item in finalized)
        assert len([call for call in provider.calls if call["text"] == old_text]) == 2
        assert not any(call["text"] == final_text for call in provider.calls)
        assert provider.cancelled == []
        for target in ("en", "ja"):
            provider.release("successor", target)
        successor = await harness.self_owner.submit_text("successor")
        await harness.translation_turns.wait_for_parent(successor.turn_id)
        successor_outputs = [
            item for item in trace.items if item.parent_utterance_id == successor.turn_id
        ]
        assert all(
            item.context_texts == ("accepted seed", "intervening manual", final_text)
            for item in successor_outputs
        )
        unique_sources = [item.source_child for item in trace.items if item.target_index == 0]
        assert [item.raw_text for item in unique_sources] == [
            "accepted seed",
            "intervening manual",
            final_text,
            "successor",
        ]
        assert [item.order for item in unique_sources] == sorted(
            item.order for item in unique_sources
        )
    finally:
        for text in (old_text, final_text, "accepted seed", "intervening manual", "successor"):
            for target in ("en", "ja"):
                provider.release(text, target)
        await harness.stop()


def test_mixed_consumption_apis_share_ranges_origins_and_terminal_tail():
    _, identity = source_fixture()
    native_origin = origin("ABCDE")
    native_origin = replace(
        native_origin,
        identity=replace(
            native_origin.identity, stream=replace(native_origin.identity.stream, channel="peer")
        ),
    )
    contributions = (STTTextContribution("one", 0, 2), STTTextContribution("two", 2, 4))
    update = STTProviderTurnUpdate(
        identity,
        1,
        "stable",
        "replace",
        "ABCDE",
        source_release=STTSourceRelease("prefix", 0, 3, "end"),
        included_contributions=contributions,
        recognition_origins=(native_origin,),
    )
    ledger = STTContributionConsumptionLedger()
    assert "".join(piece.text for piece in ledger.consume_parts(update, text_range=(0, 1))) == "A"
    text, origins = ledger.consume_with_origins(update)
    assert text == "BC"
    validate_recognition_origins(origins, text)
    assert all(
        item.identity == native_origin.identity and item.native is native_origin.native
        for item in origins
    )
    assert ledger.consume_with_origins(update) == ("", ())
    terminal = STTProviderTurnTerminal(
        identity,
        "final",
        "XBCDE",
        included_contributions=contributions,
        recognition_origins=(native_origin,),
    )
    text, origins = ledger.consume_with_origins(terminal)
    assert text == "DE"
    validate_recognition_origins(origins, text)
    assert ledger.consume_parts(terminal) == ()
    assert ledger.lifecycle(identity)[1]


def test_composition_separator_is_explicit_and_cannot_mask_real_holes():
    left, right = origin("A"), origin("B")
    separator = recognition_separator_origin(left, 1)
    origins = (left, separator, replace(right, text_start=2, text_end=3))
    validate_recognition_origins(origins, "A B")
    assert separator.native is None and separator.source_ranges == ()
    with pytest.raises(ValueError, match="provenance"):
        validate_recognition_origins(origins, "AXB")
    with pytest.raises(ValueError, match="provenance"):
        validate_recognition_origins((left, replace(right, text_start=2, text_end=3)), "A B")


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["translated", "source_only", "failed"])
async def test_progressive_final_transcript_carries_canonical_source_before_close(outcome):
    overlay = RecordingOverlay()
    provider = None if outcome == "source_only" else ControlledProvider()
    text = "fail" if outcome == "failed" else "source text"
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=RecordingOscQueue(),
        overlay_sink=overlay,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
    )
    trace = OutputTrace(harness.translation_turns.output)
    harness.translation_turns.output = trace
    try:
        identity = await begin(harness, text)
        parent_id = harness.self_owner.merge_buffer.merge_id
        await harness.self_owner.handle_vad_event(SpeechEnd(identity))
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 2)
        await until(lambda: not harness.output_runtime.has_active_overlay_deliveries)
        (submission,) = trace.items
        assert submission.outcome == outcome
        source = submission.source_child
        assert source is not None and source.raw_text == text
        canonical = [
            (index, event)
            for index, event in enumerate(overlay.events)
            if getattr(event, "source_child", None) is source
        ]
        assert canonical
        index, event = canonical[-1]
        assert event.utterance_id == parent_id
        assert any(
            later.type == "utterance_closed" and later.utterance_id == parent_id
            for later in overlay.events[index + 1 :]
        )
    finally:
        await harness.stop()


@pytest.mark.parametrize("malformation", ["origin_hole", "skipped_prefix"])
def test_malformed_release_cannot_mutate_or_evict_bounded_identity_state(malformation):
    _, previous = source_fixture(1)
    _, identity = source_fixture(2)
    ledger = STTContributionConsumptionLedger(identity_capacity=1)
    assert ledger.consume(STTProviderTurnTerminal(previous, "final", "previous")) == "previous"
    before = (
        set(ledger._known_identities),
        tuple(ledger._identity_order),
        dict(ledger._retired_floor),
    )
    update = STTProviderTurnUpdate(
        identity,
        1,
        "stable",
        "replace",
        "AB",
        contribution=STTTextContribution("source", 0, 2),
        source_release=STTSourceRelease(
            "release", 1 if malformation == "skipped_prefix" else 0, 2, "end"
        ),
        recognition_origins=(origin("A"),) if malformation == "origin_hole" else (),
    )
    with pytest.raises(ValueError, match="provenance|skips earlier"):
        ledger.consume_with_origins(update)
    assert (
        set(ledger._known_identities),
        tuple(ledger._identity_order),
        dict(ledger._retired_floor),
    ) == before
    assert ledger.consumed_contribution_ids == frozenset()
    assert ledger._releases == ledger._admitted_prefixes == {}
    valid = replace(
        update,
        source_release=STTSourceRelease("release", 0, 2, "end"),
        recognition_origins=(origin("AB"),),
    )
    assert ledger.consume_with_origins(valid)[0] == "AB"
    assert ledger.consume(STTProviderTurnTerminal(previous, "final", "previous")) == ""
