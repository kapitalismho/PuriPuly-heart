from __future__ import annotations

import asyncio
from dataclasses import replace
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.sink import OverlayEventAdapter
from puripuly_heart.core.speaker.contracts import SourceMapPiece
from puripuly_heart.core.speaker.evidence import SpeakerEvidenceView
from puripuly_heart.core.speaker.runtime import SpeakerRuntime
from puripuly_heart.core.speaker_identity import PeerSpeakerIdentityAllocator
from puripuly_heart.core.stt.backend import STTAudioBoundary, STTProviderTurnTerminal, STTTextTiming
from puripuly_heart.core.stt.streaming_speaker import StreamingSpeakerSession
from puripuly_heart.ui.overlay_calibration import OverlayCalibration
from tests.core.test_causal_batch_speaker import (
    evidence,
    opened,
    payload,
    request,
    seal,
    span,
    terminal_events,
)
from tests.core.test_issue264_shared_speaker import admitted, changed, frames
from tests.core.test_streaming_speaker import Provider, next_text, send, update
from tests.helpers.overlay_source_cards import translated_event


def leaning(start, end, slot, probability):
    item = evidence(start, end, ())
    return replace(
        item,
        frame=replace(
            item.frame,
            probabilities=tuple(probability if i == slot else 0.00001 for i in range(8)),
        ),
    )


def recorded_onsets():
    return (
        frames(0, 16000, 0)
        + (
            leaning(16000, 16160, 1, 0.350615799),
            leaning(16160, 16320, 1, 0.76322),
        )
        + frames(16320, 24000, 1)
        + (
            leaning(24000, 24160, 0, 0.438835800),
            leaning(24160, 24320, 0, 0.73607),
        )
        + frames(24320, 32000, 0)
    )


@pytest.mark.asyncio
async def test_uncertain_successor_onsets_commit_once_and_keep_exact_a_b_a_pcm():
    backend, recognizer, session = await opened()
    req = request()
    await backend._decode_lock.acquire()
    try:
        await session.begin_turn(req)
        await payload(session, req, 0, 32000, 1)
        delivered = recorded_onsets()
        await session.observe_speaker_evidence(delivered)
        assert session._active.submitted_cursor == 24000
        assert session._active.committed_transition_sample == 24320
        assert len(session._active.children) == 2
        boundaries = list(session.evidence.boundaries())
        assert [b.normalized_sample for b in boundaries] == [16320, 24320]
        assert [b.successor_onset_sample for b in boundaries] == [16000, 24000]
        assert [b.transition_end_sample for b in boundaries] == [16320, 24320]
        for _ in range(3):
            await session.observe_speaker_evidence(delivered)
            await session.observe_speaker_evidence((replace(delivered[0], speaker_epoch=10),))
        assert len(session._active.children) == 2
        await seal(session, req)
        backend._decode_lock.release()
        events = await asyncio.wait_for(terminal_events(session), 2)
        assert [call.samples.size for call in recognizer.calls] == [16000, 8000, 8000]
        np.testing.assert_array_equal(
            np.concatenate([call.samples for call in recognizer.calls]),
            np.arange(32000, dtype=np.int16).astype(np.float32) / 32768,
        )
        terminal = events[-1]
        assert [run.speaker_id for run in terminal.final_speaker_runs] == ["0", "1", "0"]
        assert [
            (
                audio.source_ranges[0].normalized_start_sample,
                audio.source_ranges[-1].normalized_end_sample,
            )
            for audio in terminal.source_audio
        ] == [(0, 16000), (16000, 24000), (24000, 32000)]
        assert len(terminal.included_contributions) == 3
        assert all(
            audio.display_group_id == req.display_group_id for audio in terminal.source_audio
        )
    finally:
        if backend._decode_lock.locked():
            backend._decode_lock.release()
        await session.close()
        await backend.close()


def test_callback_map_fragments_are_not_distinct_native_confirmation_frames():
    view = SpeakerEvidenceView()
    view.observe(frames(0, 1600, 0))
    successor = evidence(1600, 1760, (1,))
    fragmented = replace(
        successor,
        source_map=(
            SourceMapPiece(1600, 1680, span(1600, 1680), "original"),
            SourceMapPiece(1680, 1760, span(1680, 1760), "original"),
        ),
    )
    view.observe((fragmented, fragmented))
    assert list(view.boundaries()) == []
    assert view.attribute((span(0, 1760),)).attribution_state == "mixed"
    view.observe((evidence(1760, 1920, (1,)),))
    assert [b.normalized_sample for b in view.boundaries()] == [1600]


def test_single_predecessor_impulse_does_not_establish_a_transition():
    view = SpeakerEvidenceView()
    view.observe((evidence(0, 160, (0,)),) + frames(160, 800, 1))
    assert list(view.boundaries()) == []
    assert view.attribute((span(0, 800),)).attribution_state == "mixed"


def test_valid_omitted_quiet_is_distinct_from_genuinely_missing_coverage():
    view = SpeakerEvidenceView()
    view.observe(frames(0, 320, 0))
    omission = replace(
        evidence(320, 480, ()),
        source_map=(SourceMapPiece(0, 0, span(320, 480), "unobserved"),),
        frame=None,
        reason="omitted_silence",
        processed_through_sample=320,
        immutable_through_sample=320,
    )
    view.observe((omission,))
    assert any(frame.span == span(320, 480) for frame in view.frames)
    token = view.pin((span(0, 480),))
    assert view.resolve(token).speaker_id == "0"
    view.observe(frames(480, 800, 1))
    assert [b.normalized_sample for b in view.boundaries()] == [480]
    view = SpeakerEvidenceView()
    view.observe(frames(0, 320, 0))
    view.observe((replace(omission, reason=None),))
    assert view.attribute((span(0, 480),)).attribution_state == "unavailable"
    view.observe(frames(480, 800, 1))
    assert list(view.boundaries()) == []


@pytest.mark.asyncio
async def test_streaming_uses_same_confirmed_onset_without_internal_untimed_split():
    provider, req = Provider(), request()
    session = StreamingSpeakerSession(provider)
    text = "Alpha. No. Alpha."
    try:
        await session.begin_turn(req)
        await send(session, req, 0, 32000, 1)
        await session.observe_speaker_evidence(recorded_onsets())
        assert [
            (boundary.normalized_sample, boundary.successor_onset_sample)
            for _, boundary in session.audio_boundaries
        ] == [(16320, 16000), (24320, 24000)]
        await session.observe_speaker_evidence(recorded_onsets())
        assert len(session.audio_boundaries) == 2
        event = update(req, text)
        event = replace(
            event,
            provenance=replace(
                event.provenance,
                text_timing=(
                    STTTextTiming("phrase-1", 0, 7, "word_interval", 0, 0.9, "s", "clock", True),
                    STTTextTiming("phrase-1", 7, 11, "word_interval", 1.0, 1.5, "s", "clock", True),
                    STTTextTiming(
                        "phrase-1", 11, 17, "word_interval", 1.5, 2.0, "s", "clock", True
                    ),
                ),
                sent_until_sample=32000,
            ),
        )
        provider.events.put(event)
        result = await next_text(session.turn_events())
        assert [run.text for run in result.final_speaker_runs] == ["Alpha. ", "No. ", "Alpha."]
        assert [run.speaker_id for run in result.final_speaker_runs] == ["0", "1", "0"]
        provider.events.put(
            STTProviderTurnTerminal(req.identity, "final", text, text_authority="authoritative")
        )
        assert not provider.sealed
    finally:
        await session.close()


def test_actual_omission_producer_preserves_positive_quiet_provenance():
    runtime = SpeakerRuntime(None)
    runtime._session = "capture"
    runtime._epoch = 11
    runtime._retain_onset(runtime._block(np.zeros(9600, dtype=np.float32), (span(16000, 25600),)))
    (omission,) = runtime.drain_evidence(1)
    assert omission.reason == "omitted_silence"
    assert omission.source_map == (SourceMapPiece(0, 0, span(16000, 17600), "unobserved"),)
    view = SpeakerEvidenceView()
    view.observe(frames(0, 16000, 0) + (omission,))
    assert view.attribute((span(0, 17600),)).speaker_id == "0"
    assert view.attribute((span(0, 17601),)).attribution_state == "unavailable"


@pytest.mark.asyncio
async def test_late_overflow_keys_update_in_place_divide_and_allow_known_return():
    clock = FakeClock(_now=10)
    view = SpeakerEvidenceView()
    allocator = PeerSpeakerIdentityAllocator(view, clock=clock.now)
    presenter = OverlayPresenter(OverlayCalibration(), clock=clock)
    presenter.bind_speaker_identities(allocator)
    adapter = OverlayEventAdapter(clock=clock)
    overflow_cards = []
    try:
        for order in range(1, 7):
            start = (order - 1) * 1600
            source, transcript, _ = admitted(view, allocator, order, start, start + 1600)
            if order <= 4:
                view.observe(frames(start, start + 1600, order - 1))
            await presenter.emit(
                translated_event(
                    adapter,
                    source,
                    f"translation {order}",
                    allocator.assignment_for(transcript.utterance_id),
                )
            )
            if order <= 4:
                continue
            first = presenter.snapshot().blocks[-1]
            entry = presenter._entries[
                presenter._sources.units[("peer", source.source_id)].card_key
            ]
            deadline = presenter._entry_expiration_deadline(entry)
            clock.advance(0.1)
            view.observe(frames(start, start + 1600, order - 1))
            await changed(presenter, lambda: entry.speaker_gray_reason == "palette_overflow")
            final = next(block for block in presenter.snapshot().blocks if block.id == first.id)
            assert final.speaker_style == first.speaker_style == "gray"
            assert final.primary_text == first.primary_text
            assert final.secondary_text == first.secondary_text
            assert final.appearance_seq == first.appearance_seq
            assert final.occupant_key == first.occupant_key
            assert presenter._entry_expiration_deadline(entry) == deadline
            assert entry.speaker_assignment.attribution.key.local_speaker_id == str(order - 1)
            assert entry.speaker_assignment.palette_overflow
            assert entry.speaker_assignment.palette_index is None
            overflow_cards.append(first.id)
        assert presenter.snapshot().speaker_divider
        source, transcript, _ = admitted(view, allocator, 7, 9600, 11200)
        view.observe(frames(9600, 11200, 4))
        await presenter.emit(
            translated_event(
                adapter,
                source,
                "returning fifth",
                allocator.assignment_for(transcript.utterance_id),
            )
        )
        assert {block.id for block in presenter.snapshot().blocks} == set(overflow_cards)
        returning = next(b for b in presenter.snapshot().blocks if b.id == overflow_cards[0])
        assert returning.primary_text == "translation 5 returning fifth"
        assert presenter.snapshot().speaker_divider
        assert len(allocator._keys) == allocator.CAPACITY == 4
    finally:
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("application_time, expected_style", [(10.95, "gold"), (11.1, "gray")])
async def test_candidate_availability_is_not_late_snapshot_application(
    application_time, expected_style
):
    clock = FakeClock(_now=10)
    view = SpeakerEvidenceView()
    allocator = PeerSpeakerIdentityAllocator(view, clock=clock.now)
    presenter = OverlayPresenter(OverlayCalibration(), clock=clock)
    presenter.bind_speaker_identities(allocator)
    later_target = OverlayPresenter(OverlayCalibration(), clock=clock)
    later_target.bind_speaker_identities(allocator)
    adapter = OverlayEventAdapter(clock=clock)
    source, transcript, token = admitted(view, allocator, 1, 0, 1600)
    try:
        await presenter.emit(
            translated_event(
                adapter, source, "immediate", allocator.assignment_for(transcript.utterance_id)
            )
        )
        first = presenter.snapshot().blocks[0]
        entry = next(iter(presenter._entries.values()))
        deadline = presenter._entry_expiration_deadline(entry)
        async with presenter._ownership_transition_lock:
            clock.advance(0.9)
            view.observe(frames(0, 1600, 0))
            available = allocator.assignment_for(transcript.utterance_id)
            assert available.attribution.key is not None
            assert not available.settled
            assert allocator.resolution_delays == []
            assert allocator.next_deadline() == 11.0
            clock.advance(application_time - clock.now())
        await changed(
            presenter, lambda: allocator.source_snapshot(source.source_id).assignment.settled
        )
        final = presenter.snapshot().blocks[0]
        assert final.speaker_style == expected_style
        assert final.id == first.id
        assert final.primary_text == first.primary_text
        assert final.secondary_text == first.secondary_text
        assert final.appearance_seq == first.appearance_seq
        assert presenter._entry_expiration_deadline(entry) == deadline
        assert not view.has_proof(token)
        retry = replace(transcript, utterance_id=uuid4())
        canonical = allocator.observe(retry, child_sequence=0)
        assert canonical == allocator.assignment_for(transcript.utterance_id)
        clock.advance(11.2 - clock.now())
        await later_target.emit(translated_event(adapter, source, "other target", canonical))
        assert later_target.snapshot().blocks[0].speaker_style == expected_style
        if expected_style == "gold":
            assert allocator.resolution_delays == pytest.approx([application_time - 10])
        else:
            assert canonical.attribution.key is None
            assert allocator.resolution_delays == []
            assert allocator._keys == {}
    finally:
        await presenter.close()
        await later_target.close()


@pytest.mark.parametrize("start,end,onset", [(None, None, 1), (160, 320, 159), (160, 320, 321)])
def test_successor_onset_stays_inside_the_explicit_uncertainty_interval(start, end, onset):
    with pytest.raises(ValueError):
        STTAudioBoundary("invalid-onset", 7, 320, 3.0, "speaker", "11", start, end, onset)
