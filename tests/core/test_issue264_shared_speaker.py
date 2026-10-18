from __future__ import annotations

import asyncio
from dataclasses import replace
from uuid import UUID, uuid4

import pytest

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.sink import OverlayEventAdapter
from puripuly_heart.core.speaker.evidence import SpeakerEvidenceView
from puripuly_heart.core.speaker_identity import PeerSpeakerIdentityAllocator
from puripuly_heart.domain.models import FinalSpeakerRun, Transcript
from puripuly_heart.domain.recognition import (
    RecognitionStreamIdentity,
    RecognitionTextOrigin,
    RecognitionUnitIdentity,
)
from puripuly_heart.ui.overlay_calibration import OverlayCalibration
from tests.core.test_causal_batch_speaker import evidence, span
from tests.helpers.overlay_source_cards import source_child, translated_event


def frames(start, end, slot):
    return tuple(evidence(left, min(left + 160, end), (slot,)) for left in range(start, end, 160))


def admitted(view, allocator, order, start, end, *, speaker=None, attempt=None):
    text = f"source {order}"
    source = source_child(order, text, group=UUID(int=264))
    token = view.pin((span(start, end),))
    origin = RecognitionTextOrigin(
        RecognitionUnitIdentity(
            RecognitionStreamIdentity("peer", 1, 7, "provider", ()), uuid4(), order
        ),
        0,
        len(text),
        source_ranges=(span(start, end),),
        speaker_proof_id=token,
    )
    source = replace(source, origins=(origin,))
    transcript = Transcript(
        attempt or uuid4(),
        text,
        True,
        channel="peer",
        source_child=source,
        final_speaker_runs=(
            FinalSpeakerRun(
                text,
                None if speaker is None else str(speaker),
                "11",
                source="nemotron",
                attribution_state="unavailable" if speaker is None else "identified",
            ),
        ),
    )
    allocator.observe(transcript, child_sequence=0)
    return source, transcript, token


async def changed(presenter, predicate):
    for _ in range(50):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError(presenter.snapshot())


@pytest.mark.parametrize("probability", [0.4, 0.6])
def test_same_candidate_and_foreign_leaning_ambiguity_are_different(probability):
    for slot, expected in [(0, "identified"), (1, "uncertain")]:
        view = SpeakerEvidenceView()
        view.observe(frames(0, 15840, 0))
        item = evidence(15840, 16000, ())
        item = replace(
            item,
            frame=replace(
                item.frame,
                probabilities=tuple(probability if i == slot else 0.01 for i in range(8)),
            ),
        )
        view.observe((item,))
        assert view.attribute((span(0, 16000),)).attribution_state == expected


def test_candidates_require_distinct_temporal_confirmation_and_preserve_short_no():
    view = SpeakerEvidenceView()
    view.observe(frames(0, 1600, 0))
    impulse = evidence(1600, 1760, (1,))
    view.observe((impulse,))
    assert list(view.boundaries()) == []
    view.observe((impulse, impulse))
    assert list(view.boundaries()) == []
    assert view.attribute((span(0, 1760),)).attribution_state == "mixed"
    view.observe(frames(1760, 2400, 1) + frames(2400, 3200, 0))
    assert [boundary.normalized_sample for boundary in view.boundaries()] == [1600, 2400]
    assert view.attribute((span(1600, 2400),)).speaker_id == "1"


def test_missing_overlap_revision_reset_and_hard_bounds_are_conservative():
    view = SpeakerEvidenceView()
    token = view.pin((span(0, 320),))
    open_frame = evidence(0, 160, (0,), state="open", immutable=0)
    view.observe((open_frame,))
    assert view.resolve(token).attribution_state == "unavailable"
    view.observe((evidence(0, 160, (1,)), evidence(160, 320, (1,))))
    assert view.resolve(token).speaker_id == "1"
    revision = view.revision
    view.observe((evidence(0, 160, (0,)),))
    assert view.revision == revision
    assert view.resolve(token).speaker_id == "1"
    assert view.attribute((span(0, 321),)).attribution_state == "unavailable"
    view.observe((evidence(320, 480, (0, 1)),))
    assert view.attribute((span(0, 480),)).attribution_state == "mixed"
    view.reset()
    assert not view.has_proof(token)
    assert view.resolve(token).attribution_state == "unavailable"
    for index in range(view.MAX_PROOFS + 10):
        view.pin((span(index, index + 1),))
    assert len(view._proofs) == view.MAX_PROOFS
    view.observe(frames(0, (view.MAX_FRAMES + 10) * 160, 0))
    assert len(view.frames) == view.MAX_FRAMES
    assert not view.has_proof(token)


def test_onset_is_unique_support_while_proven_replay_is_not():
    view = SpeakerEvidenceView()
    view.observe(frames(0, 160, 1) + frames(160, 16160, 0))
    onset = view.pin((span(0, 16160),))
    replay = view.pin((span(160, 16160),), replayed=(span(0, 160),))
    assert view.resolve(onset).attribution_state == "mixed"
    assert view.resolve(replay).speaker_id == "0"


@pytest.mark.asyncio
async def test_readable_text_is_immediate_then_style_changes_in_place_without_ttl_or_text_mutation():
    clock = FakeClock(_now=10)
    view = SpeakerEvidenceView()
    allocator = PeerSpeakerIdentityAllocator(view, clock=clock.now)
    presenter = OverlayPresenter(OverlayCalibration(), clock=clock)
    presenter.bind_speaker_identities(allocator)
    adapter = OverlayEventAdapter(clock=clock)
    source, transcript, token = admitted(view, allocator, 1, 0, 1600)
    try:
        await presenter.emit(
            translated_event(
                adapter, source, "translated", allocator.assignment_for(transcript.utterance_id)
            )
        )
        first = presenter.snapshot().blocks[0]
        assert first.speaker_style == "gray"
        entry = next(iter(presenter._entries.values()))
        deadline = presenter._entry_expiration_deadline(entry)
        allocator.retire(transcript.utterance_id)
        assert view.has_proof(token)
        clock.advance(0.25)
        view.observe(frames(0, 1600, 0))
        await changed(presenter, lambda: presenter.snapshot().blocks[0].speaker_style == "gold")
        final = presenter.snapshot().blocks[0]
        assert final.id == first.id
        assert final.occupant_key == first.occupant_key
        assert final.appearance_seq == first.appearance_seq
        assert final.primary_text == first.primary_text == "translated"
        assert final.secondary_text == first.secondary_text == source.raw_text
        assert final.card_revision > first.card_revision
        assert presenter._entry_expiration_deadline(entry) == deadline
        assert not view.has_proof(token)
        assert allocator.resolved_within_500ms == allocator.resolved_within_1s == 1
        view.observe(frames(0, 1600, 1))
        await asyncio.sleep(0)
        assert presenter.snapshot().blocks[0] == final
    finally:
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("disposition", ["deadline", "reset", "eviction"])
async def test_late_proof_cannot_update_after_termination(disposition):
    clock = FakeClock(_now=10)
    view = SpeakerEvidenceView()
    allocator = PeerSpeakerIdentityAllocator(view, clock=clock.now)
    presenter = OverlayPresenter(OverlayCalibration(), clock=clock)
    presenter.bind_speaker_identities(allocator)
    source, transcript, token = admitted(view, allocator, 1, 0, 1600)
    adapter = OverlayEventAdapter(clock=clock)
    try:
        await presenter.emit(
            translated_event(
                adapter, source, "visible", allocator.assignment_for(transcript.utterance_id)
            )
        )
        if disposition == "deadline":
            clock.advance(1.000)
        elif disposition == "reset":
            allocator.reset()
        else:
            allocator.retire_source(source.source_id)
        view.observe(frames(0, 1600, 0))
        for _ in range(10):
            await asyncio.sleep(0)
        assert presenter.snapshot().blocks[0].speaker_style == "gray"
        assert not view.has_proof(token)
    finally:
        await presenter.close()


def test_reversed_proof_order_defers_only_color_and_settles_in_source_order():
    clock = FakeClock(_now=10)
    view = SpeakerEvidenceView()
    allocator = PeerSpeakerIdentityAllocator(view, clock=clock.now)
    first, a, _ = admitted(view, allocator, 1, 0, 1600)
    second, b, _ = admitted(view, allocator, 2, 1600, 3200)
    view.observe(frames(1600, 3200, 1))
    assert allocator.resolve_source(second, now=10).palette_index is None
    allocator.mark_readable(first, now=10)
    allocator.mark_readable(second, now=10)
    clock.advance(0.4)
    view.observe(frames(0, 1600, 0))
    assert allocator.assignment_for(a.utterance_id).palette_index is None
    allocator.mark_readable(first, now=clock.now())
    allocator.mark_readable(second, now=clock.now())
    assert allocator.assignment_for(a.utterance_id).palette_index == 0
    assert allocator.assignment_for(b.utterance_id).palette_index == 1
    replay = replace(b, utterance_id=uuid4())
    assert allocator.observe(replay, child_sequence=99) == allocator.assignment_for(b.utterance_id)
    assert len(allocator._sources) == 2


@pytest.mark.asyncio
async def test_pre_display_evidence_preserves_append_but_late_returning_a_never_migrates():
    clock = FakeClock(_now=10)
    view = SpeakerEvidenceView()
    allocator = PeerSpeakerIdentityAllocator(view, clock=clock.now)
    presenter = OverlayPresenter(OverlayCalibration(), clock=clock, visible_window_target_blocks=3)
    presenter.bind_speaker_identities(allocator)
    adapter = OverlayEventAdapter(clock=clock)
    try:
        ids = []
        for order, slot in [(1, 0), (2, 1), (3, None)]:
            start = (order - 1) * 1600
            source, transcript, _ = admitted(view, allocator, order, start, start + 1600)
            if slot is not None:
                view.observe(frames(start, start + 1600, slot))
            await presenter.emit(
                translated_event(
                    adapter,
                    source,
                    f"translation {order}",
                    allocator.assignment_for(transcript.utterance_id),
                )
            )
            ids.append(presenter.snapshot().blocks[-1].id)
        assert len(set(ids)) == 3
        clock.advance(0.2)
        view.observe(frames(3200, 4800, 0))
        await changed(presenter, lambda: presenter.snapshot().blocks[-1].speaker_style == "gold")
        assert [block.id for block in presenter.snapshot().blocks] == ids
        assert [block.primary_text for block in presenter.snapshot().blocks] == [
            "translation 1",
            "translation 2",
            "translation 3",
        ]
        source, transcript, _ = admitted(view, allocator, 4, 4800, 6400)
        view.observe(frames(4800, 6400, 0))
        await presenter.emit(
            translated_event(
                adapter, source, "translation 4", allocator.assignment_for(transcript.utterance_id)
            )
        )
        assert presenter.snapshot().blocks[-1].id in ids
        assert presenter.snapshot().blocks[-1].primary_text.endswith("translation 4")
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_hidden_source_only_reveal_starts_clock_at_selected_readable_snapshot():
    clock = FakeClock(_now=10)
    view = SpeakerEvidenceView()
    allocator = PeerSpeakerIdentityAllocator(view, clock=clock.now)
    presenter = OverlayPresenter(OverlayCalibration(), clock=clock, show_peer_original=False)
    presenter.bind_speaker_identities(allocator)
    source, transcript, token = admitted(view, allocator, 1, 0, 1600)
    adapter = OverlayEventAdapter(clock=clock)
    try:
        await presenter.emit(
            adapter.transcript_final(transcript, source_language="en", target_language="ko")
        )
        assert presenter.snapshot().blocks == []
        assert allocator.next_deadline() is None
        clock.advance(2)
        await presenter.update_display_preferences(show_translation=True, show_peer_original=True)
        assert presenter.snapshot().blocks[0].secondary_text == source.raw_text
        assert allocator.next_deadline() == 13
        clock.advance(0.2)
        view.observe(frames(0, 1600, 0))
        await changed(presenter, lambda: presenter.snapshot().blocks[0].speaker_style == "gold")
        assert not view.has_proof(token)
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_producer_completion_and_protected_output_wait_do_not_start_or_destroy_deadline():
    from puripuly_heart.core.runtime.output import OutputRuntime
    from tests.core.runtime.test_output_runtime import RecordingChatbox, _publish_peer_overlay

    clock = FakeClock(_now=10)
    view = SpeakerEvidenceView()
    allocator = PeerSpeakerIdentityAllocator(view, clock=clock.now)
    presenter = OverlayPresenter(OverlayCalibration(), clock=clock, visible_window_target_blocks=1)
    output = OutputRuntime(chatbox=RecordingChatbox(), clock=clock, overlay_sink=presenter)
    output.bind_speaker_identities(allocator)
    adapter = OverlayEventAdapter(clock=clock)
    source, transcript, token = admitted(view, allocator, 1, 0, 1600)
    try:
        self_id = uuid4()
        await presenter.emit(
            adapter.self_active_update(
                text="SELF protected",
                utterance_id=self_id,
                occupant_key=f"self:{self_id}",
                source_language="en",
                target_language="ko",
            )
        )
        event = translated_event(
            adapter,
            source,
            "ready translation",
            allocator.assignment_for(transcript.utterance_id),
            attempt=transcript.utterance_id,
        )
        output.activate_peer_generation(1)
        assert await output.admit_translation_parent(
            parent_id=str(event.parent_utterance_id),
            channel="peer",
            origin="peer",
            turn_generation=1,
            turn_order=1,
            retained_payloads=(source.raw_text, event.text),
            destination_targets={"overlay": frozenset({0})},
        ) == frozenset({"overlay"})
        receipt = await _publish_peer_overlay(output, event)
        allocator.retire(transcript.utterance_id)
        for _ in range(5):
            await asyncio.sleep(0)
        assert receipt.decision.reason == "accepted_handoff"
        assert allocator.next_deadline() is None
        assert view.has_proof(token)
        assert presenter.snapshot().blocks[0].primary_text == "SELF protected"
        clock.advance(2)
        view.observe(frames(0, 1600, 0))
        await presenter.emit(adapter.self_active_clear())
        await output.wait_for_peer_output_idle()
        block = presenter.snapshot().blocks[0]
        assert block.primary_text == "ready translation"
        assert block.speaker_style == "gold"
        assert allocator._sources[source.source_id].first_readable == 12
        assert not view.has_proof(token)
        assert output.chatbox.messages == []
    finally:
        await output.close()
        await presenter.close()


def test_source_order_deadline_unblocks_later_color_without_losing_text_obligation():
    clock = FakeClock(_now=10)
    view = SpeakerEvidenceView()
    allocator = PeerSpeakerIdentityAllocator(view, clock=clock.now)
    first, _, first_token = admitted(view, allocator, 1, 0, 1600)
    second, b, _ = admitted(view, allocator, 2, 1600, 3200)
    view.observe(frames(1600, 3200, 1))
    allocator.mark_readable(first, now=10)
    allocator.mark_readable(second, now=10.2)
    clock.advance(1)
    allocator.refresh()
    assert not view.has_proof(first_token)
    assert allocator.resolve_source(first, now=11).palette_index is None
    assert allocator.assignment_for(b.utterance_id).palette_index is None
    allocator.mark_readable(second, now=clock.now())
    assert allocator.assignment_for(b.utterance_id).palette_index == 0


def test_rejection_releases_only_after_all_attempts_finish_and_applied_output_survives():
    clock = FakeClock(_now=10)
    view = SpeakerEvidenceView()
    allocator = PeerSpeakerIdentityAllocator(view, clock=clock.now)
    source, first, token = admitted(view, allocator, 1, 0, 1600)
    second = replace(first, utterance_id=uuid4())
    allocator.observe(second, child_sequence=0)
    allocator.output_disposition(first.utterance_id, applied=False)
    allocator.retire(first.utterance_id)
    assert view.has_proof(token)
    allocator.output_disposition(second.utterance_id, applied=False)
    assert view.has_proof(token)
    allocator.retire(second.utterance_id)
    assert not view.has_proof(token)
    assert source.source_id not in allocator._sources
    source, first, token = admitted(view, allocator, 2, 1600, 3200)
    allocator.output_disposition(first.utterance_id, applied=True)
    allocator.retire(first.utterance_id)
    assert view.has_proof(token)
    allocator.retire_source(source.source_id)
    assert not view.has_proof(token)


@pytest.mark.asyncio
async def test_native_final_claim_transfers_through_origin_and_survives_asr_close():
    from puripuly_heart.core.stt.backend import STTProviderTurnTerminal
    from puripuly_heart.core.stt.batch_speaker import CausalBatchSession
    from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
    from tests.core.test_causal_batch_speaker import payload, request, seal
    from tests.core.test_stt_scoped_engine import ControlledScopedSession

    view = SpeakerEvidenceView()
    inner = ControlledScopedSession()
    session = CausalBatchSession(inner, evidence_view=view)
    req = request()
    events = session.turn_events()
    delivered = []

    def deliver(event):
        for audio in event.source_audio:
            for token in audio.speaker_proof_ids:
                view.adopt(token)
        delivered.append(event)

    engine = ScopedRecognitionEngine(None, evidence_view=view, event_sink=deliver)
    try:
        await session.begin_turn(req)
        await payload(session, req, 0, 1600, 1)
        await seal(session, req)
        inner.emit(
            STTProviderTurnTerminal(
                inner.requests[0].identity, "final", "No.", text_authority="authoritative"
            )
        )
        final = await asyncio.wait_for(anext(events), 1)
        token = final.source_audio[0].speaker_proof_id
        assert final.final_speaker_runs[0].speaker_id is None
        assert view.has_proof(token)
        await engine._emit(final)
        await events.aclose()
        assert delivered == [final]
        assert view._proofs[token].stage == "source"
        assert view._proofs[token].handoffs == 0
        await session.close()
        assert view.has_proof(token)
        view.observe(frames(0, 1600, 1))
        assert view.resolve(token).speaker_id == "1"
        assert final.final_speaker_runs[0].speaker_id is None
        view.release(token)
    finally:
        await events.aclose()
        await session.close()
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["translated_delay", "source_only", "failed"])
async def test_actual_peer_projection_resolves_during_translation_or_after_source_only_failure(
    outcome,
):
    from puripuly_heart.core.audio.ownership import AudioSegmentTerminalReceipt
    from puripuly_heart.core.stt.backend import (
        STTProviderTurnTerminal,
        STTSourceAudio,
        STTTextContribution,
    )
    from puripuly_heart.core.stt.scoped_normalizer import STTScopedTurnNormalizer
    from tests.core.test_source_context_contract import (
        ControlledProvider,
        close,
        finish,
        harness_for,
        source_fixture,
        until,
    )

    clock = FakeClock(_now=10)
    presenter = OverlayPresenter(OverlayCalibration(), clock=clock)
    provider = ControlledProvider()
    text = "fail" if outcome == "failed" else "same"
    if outcome == "translated_delay":
        provider.gates[text] = asyncio.Event()
    harness, trace = harness_for(
        provider, overlay_sink=presenter, peer_translation_enabled=outcome != "source_only"
    )
    harness.peer_owner.set_clock(clock)
    allocator = harness.peer_owner._speaker_identities
    view = allocator.evidence
    snapshot, identity = source_fixture()
    token = view.pin(snapshot.content_ranges)
    proof = tuple(
        replace(
            item,
            source_map=(
                replace(
                    item.source_map[0],
                    capture=snapshot.content_ranges[0].slice_normalized(
                        item.frame.start_sample, item.frame.end_sample
                    ),
                ),
            ),
        )
        for item in frames(0, 16000, 0)
    )
    try:
        terminal = STTScopedTurnNormalizer(identity).apply_terminal(
            STTProviderTurnTerminal(
                identity,
                "final",
                text,
                text_authority="authoritative",
                final_speaker_runs=(
                    FinalSpeakerRun(
                        text, None, "", source="nemotron", attribution_state="unavailable"
                    ),
                ),
                included_contributions=(STTTextContribution("native-final", 0, len(text)),),
                source_audio=(
                    STTSourceAudio(
                        "native-final",
                        snapshot.content_ranges,
                        speaker_proof_id=token,
                    ),
                ),
            )
        )
        receipt = AudioSegmentTerminalReceipt(identity.segment, "final", snapshot, 2)
        await harness.peer_owner.handle_provider_turn_terminal(receipt, terminal)
        if outcome == "translated_delay":
            await until(lambda: len(provider.calls) == 1)
            assert presenter.snapshot().blocks == []
            clock.advance(2)
            view.observe(proof)
            provider.gates[text].set()
        await finish(harness)
        await harness.output_runtime.wait_for_peer_output_idle()
        assert len(trace.items) == 1
        source = trace.items[0].source_child
        assert source.speaker.speaker_id is None
        assert terminal.final_speaker_runs[0].speaker_id is None
        block = presenter.snapshot().blocks[0]
        entry = next(iter(presenter._entries.values()))
        deadline = presenter._entry_expiration_deadline(entry)
        if outcome != "translated_delay":
            assert block.speaker_style == "gray"
            clock.advance(0.3)
            view.observe(proof)
            await changed(presenter, lambda: presenter.snapshot().blocks[0].speaker_style == "gold")
        assert presenter.snapshot().blocks[0].speaker_style == "gold"
        assert presenter.snapshot().blocks[0].id == block.id
        assert presenter.snapshot().blocks[0].secondary_text == text
        assert presenter._entry_expiration_deadline(entry) == deadline
        assert len(provider.calls) == (0 if outcome == "source_only" else 1)
        assert len(harness.peer_runtime.translation_history) == (
            0 if outcome == "source_only" else 1
        )
        assert harness.output_runtime.chatbox.messages == []
    finally:
        if text in provider.gates:
            provider.gates[text].set()
        await close(harness)
        await presenter.close()


def test_source_and_attempt_pressure_are_bounded_and_cancellation_releases_proof():
    view = SpeakerEvidenceView()
    allocator = PeerSpeakerIdentityAllocator(view, clock=lambda: 10)
    for order in range(1, allocator.MAX_SOURCES + 10):
        source, transcript, token = admitted(view, allocator, order, order * 160, (order + 1) * 160)
    assert len(allocator._sources) == allocator.MAX_SOURCES
    assert len(view._proofs) <= view.MAX_PROOFS
    for _ in range(allocator.MAX_ATTEMPTS + 10):
        allocator.observe(replace(transcript, utterance_id=uuid4()), child_sequence=0)
    assert len(allocator._attempts) == allocator.MAX_ATTEMPTS
    allocator.reset()
    assert len(view._proofs) == 0
    source, transcript, token = admitted(view, allocator, 1000, 0, 1600)
    allocator.mark_readable(source, now=10)
    allocator.cancel_attempt(transcript.utterance_id)
    view.observe(frames(0, 1600, 0))
    assert allocator.source_snapshot(source.source_id) is None
    assert not view.has_proof(token)


def test_one_contribution_claim_transfers_to_all_admitted_source_children_before_settlement():
    view = SpeakerEvidenceView()
    allocator = PeerSpeakerIdentityAllocator(view, clock=lambda: 10)
    token = view.pin((span(0, 1600),))
    view.observe(frames(0, 1600, 0))
    transcripts = []
    for order in (1, 2):
        source = source_child(order, "words", group=UUID(int=264))
        origin = RecognitionTextOrigin(
            RecognitionUnitIdentity(
                RecognitionStreamIdentity("peer", 1, 7, "provider", ()), uuid4(), order
            ),
            0,
            5,
            source_ranges=(span(0, 1600),),
            speaker_proof_id=token,
        )
        source = replace(source, origins=(origin,))
        transcript = Transcript(uuid4(), "words", True, channel="peer", source_child=source)
        transcripts.append(transcript)
        allocator.observe(transcript, child_sequence=0, defer_settlement=True)
        assert view.has_proof(token)
    allocator.refresh()
    assert [allocator.assignment_for(item.utterance_id).palette_index for item in transcripts] == [
        0,
        0,
    ]
    assert not view.has_proof(token)


@pytest.mark.asyncio
async def test_engine_rejects_unowned_claim_but_preserves_queued_and_admitted_claim_through_close():
    from puripuly_heart.core.audio.ownership import PeerAudioSegmentLedger
    from puripuly_heart.core.stt.backend import (
        STTProviderTurnTerminal,
        STTProviderTurnUpdate,
        STTSourceAudio,
        STTSourceRelease,
        STTTextContribution,
    )
    from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
    from tests.core.test_stt_scoped_engine import (
        ControlledScopedSession,
        segment_events,
        settings,
        wait_until,
        watchdogs,
    )

    view = SpeakerEvidenceView()
    allocator = PeerSpeakerIdentityAllocator(view, clock=lambda: 10)
    session = ControlledScopedSession()
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings())
    start, _, _ = segment_events(ledger, start_sample=0, now=0)
    token = view.pin(start.segment.content_ranges)
    rejected = view.pin(start.segment.content_ranges)
    gate = asyncio.Event()
    entered = asyncio.Event()
    source = source_child(1, "yes", group=UUID(int=264))
    origin = RecognitionTextOrigin(
        RecognitionUnitIdentity(
            RecognitionStreamIdentity("peer", 1, 1, "provider", ()), uuid4(), 1
        ),
        0,
        3,
        source_ranges=start.segment.content_ranges,
        speaker_proof_id=token,
    )
    source = replace(source, origins=(origin,))

    async def sink(event):
        if getattr(event, "source_release", None) is None:
            return
        entered.set()
        await gate.wait()
        allocator.observe(
            Transcript(uuid4(), "yes", True, channel="peer", source_child=source), child_sequence=0
        )

    engine = ScopedRecognitionEngine(
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        evidence_view=view,
        watchdog_resolver=lambda _settings: watchdogs(),
    )
    engine.bind_event_sink(sink)
    try:
        await engine.handle_owned_vad_event(start)
        identity = session.requests[0].identity
        session.emit(
            STTProviderTurnUpdate(
                identity,
                1,
                "stable",
                "append",
                "yes",
                contribution=STTTextContribution("c", 0, 3),
                source_release=STTSourceRelease("c", 0, 3, "boundary"),
                source_audio=(
                    STTSourceAudio("c", start.segment.content_ranges, speaker_proof_id=token),
                ),
            )
        )
        await asyncio.wait_for(entered.wait(), 1)
        stale = replace(identity, provider_turn_id="not-owned")
        for claim in (token, rejected):
            session.emit(
                STTProviderTurnTerminal(
                    stale,
                    "final",
                    "ignored",
                    source_audio=(
                        STTSourceAudio(
                            "stale", start.segment.content_ranges, speaker_proof_id=claim
                        ),
                    ),
                )
            )
        await wait_until(lambda: not view.has_proof(rejected))
        assert view.has_proof(token)
        gate.set()
        await wait_until(lambda: allocator.source_snapshot(source.source_id) is not None)
        await engine.close()
        assert view.has_proof(token)
        allocator.retire_source(source.source_id)
        assert not view.has_proof(token)
    finally:
        gate.set()
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("compatible_layout", [False, True])
async def test_hidden_returning_source_resolves_before_reveal_and_keeps_initial_append_choice(
    compatible_layout,
):
    clock = FakeClock(_now=10)
    view = SpeakerEvidenceView()
    allocator = PeerSpeakerIdentityAllocator(view, clock=clock.now)
    presenter = OverlayPresenter(
        OverlayCalibration(), clock=clock, show_peer_original=compatible_layout
    )
    presenter.bind_speaker_identities(allocator)
    adapter = OverlayEventAdapter(clock=clock)
    try:
        for order, slot in ((1, 0), (2, 1)):
            source, transcript, _ = admitted(
                view, allocator, order, (order - 1) * 1600, order * 1600
            )
            view.observe(frames((order - 1) * 1600, order * 1600, slot))
            await presenter.emit(
                translated_event(
                    adapter,
                    source,
                    f"translation {order}",
                    allocator.assignment_for(transcript.utterance_id),
                )
            )
        await presenter.update_display_preferences(show_translation=True, show_peer_original=False)
        a, b = presenter.snapshot().blocks
        source, transcript, _ = admitted(view, allocator, 3, 3200, 4800)
        await presenter.emit(
            adapter.transcript_final(transcript, source_language="en", target_language="ko")
        )
        assert [block.id for block in presenter.snapshot().blocks] == [a.id, b.id]
        assert allocator.source_snapshot(source.source_id).first_readable_at is None
        clock.advance(2)
        view.observe(frames(3200, 4800, 0))
        await presenter.update_display_preferences(show_translation=True, show_peer_original=True)
        blocks = presenter.snapshot().blocks
        assert blocks[0].id == b.id
        if compatible_layout:
            assert blocks[-1].id == a.id
            assert blocks[-1].primary_text == "translation 1"
            assert blocks[-1].secondary_text == "source 1 source 3"
        else:
            assert blocks[-1].id not in (a.id, b.id)
            assert blocks[-1].secondary_text == "source 3"
        assert blocks[-1].speaker_style == "gold"
        assert allocator.source_snapshot(source.source_id).first_readable_at == 12
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_real_monotonic_deadline_releases_absent_proof_without_holding_readable_text():
    from puripuly_heart.core.clock import SystemClock

    clock = SystemClock()
    view = SpeakerEvidenceView()
    allocator = PeerSpeakerIdentityAllocator(view, clock=clock.now)
    presenter = OverlayPresenter(OverlayCalibration(), clock=clock)
    presenter.bind_speaker_identities(allocator)
    adapter = OverlayEventAdapter(clock=clock)
    source, transcript, token = admitted(view, allocator, 1, 0, 1600)
    try:
        await presenter.emit(
            translated_event(
                adapter, source, "immediate text", allocator.assignment_for(transcript.utterance_id)
            )
        )
        first = presenter.snapshot().blocks[0]
        state = allocator.source_snapshot(source.source_id)
        assert first.primary_text == "immediate text" and first.speaker_style == "gray"
        assert state.deadline - state.first_readable_at == 1.0
        async with asyncio.timeout(3):
            while view.has_proof(token):
                await asyncio.sleep(0.01)
        assert clock.now() >= state.deadline
        assert allocator.source_snapshot(source.source_id).assignment.settled
        view.observe(frames(0, 1600, 0))
        await asyncio.sleep(0.01)
        assert presenter.snapshot().blocks[0] == first
    finally:
        await presenter.close()


def test_assignment_owner_close_releases_claims_and_detaches_evidence_wake():
    view = SpeakerEvidenceView()
    allocator = PeerSpeakerIdentityAllocator(view, clock=lambda: 10)
    _, _, token = admitted(view, allocator, 1, 0, 1600)
    allocator.close()
    assert not view.has_proof(token)
    allocator.changed.clear()
    view.observe(frames(0, 1600, 0))
    assert not allocator.changed.is_set()
    allocator.open()
    allocator.open()
    view.observe(frames(1600, 1760, 0))
    assert allocator.changed.is_set()
    allocator.close()
