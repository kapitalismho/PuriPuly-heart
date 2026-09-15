from __future__ import annotations

import numpy as np
import pytest

from puripuly_heart.config.resolved import vad_exit_threshold
from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import (
    AudioSegmentSettingsSnapshot,
    PeerAudioSegmentLedger,
)
from puripuly_heart.core.vad.gating import (
    PEER_VAD_SPEECH_THRESHOLD,
    PEER_VAD_START_COMMIT_CHUNKS,
    PEER_VAD_START_DEBOUNCE_CHUNKS,
    SpeechChunk,
    SpeechEnd,
    SpeechStart,
    VadGating,
    create_peer_vad_gating,
)
from tests.helpers.vad import SequenceVadEngine, chunk_samples


def test_vad_gating_emits_start_and_end_with_hangover():
    # 32ms chunks @16k => 512 samples
    probs = [0.0, 0.0, 0.9, 0.9, 0.0, 0.0, 0.0]
    engine = SequenceVadEngine(probs=probs)
    gating = VadGating(engine, sample_rate_hz=16000, ring_buffer_ms=64, hangover_ms=64)

    events = []
    for i in range(len(probs)):
        events.extend(gating.process_chunk(chunk_samples(float(i), n=gating.chunk_samples)))

    start = next(e for e in events if isinstance(e, SpeechStart))
    end = next(e for e in events if isinstance(e, SpeechEnd))

    assert start.utterance_id == end.utterance_id
    assert start.pre_roll.shape[0] == 1024  # 64ms @ 16k
    assert end.reason == "silence"
    assert end.trailing_silence_ms == 64


def test_vad_gating_does_not_end_continuous_speech_without_silence():
    probs = [0.9] * 6
    gating = VadGating(
        SequenceVadEngine(probs=probs),
        sample_rate_hz=16000,
        ring_buffer_ms=64,
        hangover_ms=64,
    )

    events = []
    for i in range(len(probs)):
        events.extend(gating.process_chunk(chunk_samples(float(i + 1), n=gating.chunk_samples)))

    assert any(isinstance(event, SpeechStart) for event in events)
    assert not any(isinstance(event, SpeechEnd) for event in events)
    assert gating.in_speech is True


def test_vad_gating_pre_roll_contains_previous_audio():
    probs = [0.0, 0.0, 0.9]
    engine = SequenceVadEngine(probs=probs)
    gating = VadGating(engine, sample_rate_hz=16000, ring_buffer_ms=64, hangover_ms=0)

    # append two silent chunks (values 0,1) then speech chunk (value 2)
    gating.process_chunk(chunk_samples(0.0, n=gating.chunk_samples))
    gating.process_chunk(chunk_samples(1.0, n=gating.chunk_samples))
    events = gating.process_chunk(chunk_samples(2.0, n=gating.chunk_samples))

    start = next(e for e in events if isinstance(e, SpeechStart))
    assert start.pre_roll.shape[0] == 1024
    assert np.allclose(start.pre_roll[:512], 0.0)
    assert np.allclose(start.pre_roll[512:], 1.0)
    assert not np.any(np.isclose(start.pre_roll, 2.0))


def test_vad_gating_appends_each_processed_chunk_to_ring_exactly_once():
    gating = VadGating(
        SequenceVadEngine(probs=[0.0, 0.9, 0.9, 0.9, 0.0]),
        sample_rate_hz=16000,
        ring_buffer_ms=160,
        hangover_ms=640,
    )

    for value in range(5):
        gating.process_chunk(chunk_samples(float(value), n=gating.chunk_samples))

    recent_audio = gating._ring.get_last_samples(gating._ring.capacity_samples)
    chunks = recent_audio.reshape(5, gating.chunk_samples)
    assert [float(chunk[0]) for chunk in chunks] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_vad_gating_starts_on_first_positive_chunk_by_default():
    engine = SequenceVadEngine(probs=[0.0, 0.9])
    gating = VadGating(engine, sample_rate_hz=16000, ring_buffer_ms=64, hangover_ms=0)

    assert gating.process_chunk(chunk_samples(0.0, n=gating.chunk_samples)) == []

    events = gating.process_chunk(chunk_samples(1.0, n=gating.chunk_samples))

    assert len(events) == 1
    assert isinstance(events[0], SpeechStart)
    assert np.allclose(events[0].chunk, 1.0)


def test_vad_gating_buffers_candidate_until_commit_threshold():
    probs = [0.0, 0.0, 0.9, 0.9, 0.9]
    gating = VadGating(
        SequenceVadEngine(probs=probs),
        sample_rate_hz=16000,
        ring_buffer_ms=64,
        speech_threshold=0.6,
        hangover_ms=64,
        start_debounce_chunks=3,
        start_commit_chunks=3,
    )

    per_chunk_events = [
        gating.process_chunk(chunk_samples(float(i), n=gating.chunk_samples))
        for i in range(len(probs))
    ]

    assert all(not events for events in per_chunk_events[:4])

    events = per_chunk_events[4]
    start = events[0]
    chunks = [start.chunk] + [event.chunk for event in events[1:] if isinstance(event, SpeechChunk)]

    assert isinstance(start, SpeechStart)
    assert start.pre_roll.shape[0] == 1024
    assert np.allclose(start.pre_roll[:512], 0.0)
    assert np.allclose(start.pre_roll[512:], 1.0)
    assert len(events) == 3
    assert [type(event) for event in events] == [SpeechStart, SpeechChunk, SpeechChunk]
    assert [float(chunk[0]) for chunk in chunks] == [2.0, 3.0, 4.0]


def test_vad_gating_drops_short_candidate_before_commit():
    probs = [0.0, 0.9, 0.9, 0.0]
    gating = VadGating(
        SequenceVadEngine(probs=probs),
        sample_rate_hz=16000,
        ring_buffer_ms=64,
        speech_threshold=0.6,
        hangover_ms=64,
        start_debounce_chunks=3,
        start_commit_chunks=3,
    )

    events: list[object] = []
    for i in range(len(probs)):
        events.extend(gating.process_chunk(chunk_samples(float(i), n=gating.chunk_samples)))

    assert events == []
    assert gating.in_speech is False
    assert gating.utterance_id is None


def test_vad_gating_rejects_commit_threshold_lower_than_debounce_threshold():
    engine = SequenceVadEngine(probs=[0.0])

    with pytest.raises(ValueError, match="start_commit_chunks"):
        VadGating(
            engine,
            sample_rate_hz=16000,
            start_debounce_chunks=3,
            start_commit_chunks=2,
        )


def test_create_peer_vad_gating_leaves_delivery_boundaries_to_controller() -> None:
    gating = create_peer_vad_gating(
        SequenceVadEngine(probs=[0.9, 0.9, 0.9, 0.0, 0.0, 0.0]),
        sample_rate_hz=16000,
        ring_buffer_ms=64,
        hangover_ms=64,
    )

    events = []
    for index in range(6):
        events.extend(gating.process_chunk(chunk_samples(float(index + 1), n=gating.chunk_samples)))

    assert gating.external_delivery_boundaries is True
    assert gating.speech_threshold == PEER_VAD_SPEECH_THRESHOLD
    assert gating.start_debounce_chunks == PEER_VAD_START_DEBOUNCE_CHUNKS
    assert gating.start_commit_chunks == PEER_VAD_START_COMMIT_CHUNKS
    assert not any(isinstance(event, SpeechEnd) for event in events)
    assert gating.in_speech is True


def test_peer_vad_hard_rollover_replays_available_predecessor_audio() -> None:
    gating = create_peer_vad_gating(
        SequenceVadEngine(probs=[0.9, 0.9, 0.9, 0.9]),
        sample_rate_hz=16000,
        ring_buffer_ms=500,
        hangover_ms=500,
    )

    initial = []
    for index in range(3):
        initial.extend(
            gating.process_chunk(chunk_samples(float(index + 1), n=gating.chunk_samples))
        )
    first_start = next(event for event in initial if isinstance(event, SpeechStart))
    first_end = gating.seal_active_for_rollover(reason="delivery_deadline")
    successor = gating.process_chunk(chunk_samples(4.0, n=gating.chunk_samples))
    second_start = next(event for event in successor if isinstance(event, SpeechStart))

    assert first_end is not None
    assert first_end.utterance_id == first_start.utterance_id
    assert first_end.reason == "delivery_deadline"
    assert second_start.utterance_id != first_start.utterance_id
    assert second_start.pre_roll.size == 3 * gating.chunk_samples
    np.testing.assert_array_equal(
        second_start.pre_roll.reshape(3, gating.chunk_samples)[:, 0],
        [1.0, 2.0, 3.0],
    )
    assert second_start.genuine_onset is False


def test_peer_vad_uses_hysteresis_for_continuation_and_pause_metadata() -> None:
    gating = create_peer_vad_gating(
        SequenceVadEngine(probs=[0.5, 0.5, 0.5, 0.39, 0.39, 0.4]),
        sample_rate_hz=16000,
        ring_buffer_ms=64,
        speech_threshold=0.5,
        hangover_ms=500,
    )

    events: list[object] = []
    for index in range(3):
        events.extend(gating.process_chunk(chunk_samples(float(index), n=gating.chunk_samples)))
    assert len([event for event in events if isinstance(event, SpeechStart)]) == 1

    gating.process_chunk(chunk_samples(3.0, n=gating.chunk_samples))
    assert gating.last_observation_was_speech is False
    gating.process_chunk(chunk_samples(4.0, n=gating.chunk_samples))
    assert gating.last_observation_was_speech is False
    gating.process_chunk(chunk_samples(5.0, n=gating.chunk_samples))
    assert gating.last_observation_was_speech is True

    end = gating.seal_active(reason="delivery_pause")
    assert end is not None
    assert end.trailing_silence_ms == 0


def test_peer_vad_floor_and_strict_sub_exit_classification() -> None:
    below_floor = np.nextafter(0.1, 0.0)
    gating = create_peer_vad_gating(
        SequenceVadEngine(probs=[0.1, 0.1, 0.1, 0.1, below_floor]),
        sample_rate_hz=16000,
        ring_buffer_ms=64,
        speech_threshold=0.1,
        hangover_ms=500,
    )

    for index in range(3):
        gating.process_chunk(chunk_samples(float(index), n=gating.chunk_samples))
    gating.process_chunk(chunk_samples(3.0, n=gating.chunk_samples))
    assert gating.last_observation_was_speech is True
    gating.process_chunk(chunk_samples(4.0, n=gating.chunk_samples))
    assert gating.last_observation_was_speech is False

    end = gating.seal_active(reason="delivery_pause")
    assert end is not None
    assert end.trailing_silence_ms == 32


def test_peer_rollover_uses_next_frozen_continuation_with_hard_cut_prefix() -> None:
    gating = create_peer_vad_gating(
        SequenceVadEngine(probs=[0.5, 0.5, 0.5, 0.85]),
        sample_rate_hz=16000,
        ring_buffer_ms=64,
        speech_threshold=0.5,
        hangover_ms=500,
    )
    initial: list[object] = []
    for index in range(3):
        initial.extend(gating.process_chunk(chunk_samples(float(index), n=gating.chunk_samples)))
    first_start = next(event for event in initial if isinstance(event, SpeechStart))
    gating.reconfigure_next_segment(
        speech_threshold=0.9,
        continuation_threshold=0.8,
        hangover_ms=500,
        ring_buffer_ms=64,
    )

    first_end = gating.seal_active_for_rollover(reason="delivery_deadline")
    successor = gating.process_chunk(chunk_samples(3.0, n=gating.chunk_samples))
    second_start = next(event for event in successor if isinstance(event, SpeechStart))

    assert first_end is not None
    assert first_end.utterance_id == first_start.utterance_id
    assert second_start.genuine_onset is False
    assert second_start.pre_roll.size == 2 * gating.chunk_samples
    assert second_start.chunk[0] == 3.0
    assert gating.speech_threshold == 0.9
    assert gating.continuation_threshold == 0.8


def test_peer_hard_rollover_prefix_has_exact_source_accounting_and_single_ownership() -> None:
    gating = create_peer_vad_gating(
        SequenceVadEngine(probs=[0.9] * 14),
        sample_rate_hz=16000,
        ring_buffer_ms=500,
        hangover_ms=500,
    )
    settings = AudioSegmentSettingsSnapshot(
        provider_id="test",
        provider_signature=("test",),
        runtime_signature=("test",),
        source_mode="manual",
        source_language="en",
        expected_languages=(),
        target_sample_rate_hz=16000,
        vad_speech_threshold=0.5,
        vad_hangover_ms=500,
        vad_pre_roll_ms=500,
    )
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings)
    first_events: list[object] = []
    for index in range(12):
        first_events.extend(
            gating.process_owned_chunk(
                chunk_samples(float(index), n=gating.chunk_samples),
                (_capture_span(index),),
            )
        )
    for event in first_events:
        ledger.observe_vad_event(event, now_monotonic_s=0.4)
    first_end = gating.seal_active_for_rollover(reason="delivery_deadline")
    assert first_end is not None
    ledger.observe_vad_event(first_end, now_monotonic_s=0.5)

    successor_events = gating.process_owned_chunk(
        chunk_samples(12.0, n=gating.chunk_samples),
        (_capture_span(12),),
    )
    successor = next(event for event in successor_events if isinstance(event, SpeechStart))
    owned = ledger.observe_vad_event(successor, now_monotonic_s=0.6)

    assert successor.pre_roll.size == 4800
    assert successor.pre_roll_capture[0].normalized_start_sample == 12 * 512 - 4800
    assert successor.pre_roll_capture[-1].normalized_end_sample == 12 * 512
    assert owned.segment.context_sample_count == 4800
    assert owned.segment.content_sample_count == 512
    assert owned.segment.content_ranges == (_capture_span(12),)
    assert ledger.snapshots[0].content_sample_count == 12 * 512
    assert ledger.snapshots[0].content_ranges[-1].normalized_end_sample == 12 * 512


def test_peer_hard_rollover_retains_snapshot_through_delay_and_clears_on_abort() -> None:
    gating = create_peer_vad_gating(
        SequenceVadEngine(probs=[0.9, 0.9, 0.9, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9]),
        sample_rate_hz=16000,
        ring_buffer_ms=500,
        hangover_ms=500,
    )
    for index in range(3):
        gating.process_owned_chunk(
            chunk_samples(float(index + 1), n=gating.chunk_samples),
            (_capture_span(index),),
        )
    assert gating.seal_active_for_rollover(reason="delivery_deadline") is not None
    assert (
        gating.process_owned_chunk(
            chunk_samples(0.0, n=gating.chunk_samples),
            (_capture_span(3),),
        )
        == []
    )
    assert (
        gating.process_owned_chunk(
            chunk_samples(0.0, n=gating.chunk_samples),
            (_capture_span(4),),
        )
        == []
    )
    delayed = gating.process_owned_chunk(
        chunk_samples(6.0, n=gating.chunk_samples),
        (_capture_span(5),),
    )
    start = next(event for event in delayed if isinstance(event, SpeechStart))
    assert start.pre_roll.size == 3 * gating.chunk_samples
    assert start.pre_roll_capture[0].normalized_start_sample == 0
    assert start.pre_roll_capture[-1].normalized_end_sample == 3 * gating.chunk_samples

    assert gating.seal_active_for_rollover(reason="delivery_deadline") is not None
    gating.reset()
    events: list[object] = []
    for index in range(6, 9):
        events.extend(
            gating.process_owned_chunk(
                chunk_samples(float(index + 1), n=gating.chunk_samples),
                (_capture_span(index),),
            )
        )
    start_after_abort = next(event for event in events if isinstance(event, SpeechStart))
    assert start_after_abort.genuine_onset is True
    assert start_after_abort.pre_roll.size == 0


def test_default_self_rollover_and_non_deadline_rollover_have_no_hard_cut_prefix() -> None:
    self_gating = VadGating(
        SequenceVadEngine(probs=[0.9, 0.9]),
        sample_rate_hz=16000,
        ring_buffer_ms=500,
        hangover_ms=500,
    )
    self_gating.process_chunk(chunk_samples(1.0, n=self_gating.chunk_samples))
    assert self_gating.seal_active_for_rollover(reason="delivery_deadline") is not None
    self_start = self_gating.process_chunk(chunk_samples(2.0, n=self_gating.chunk_samples))[0]
    assert isinstance(self_start, SpeechStart)
    assert self_start.pre_roll.size == 0

    peer_gating = create_peer_vad_gating(
        SequenceVadEngine(probs=[0.9, 0.9, 0.9, 0.9]),
        sample_rate_hz=16000,
        ring_buffer_ms=500,
        hangover_ms=500,
    )
    for index in range(3):
        peer_gating.process_chunk(chunk_samples(float(index + 1), n=peer_gating.chunk_samples))
    assert peer_gating.seal_active_for_rollover(reason="max_duration") is not None
    peer_start = peer_gating.process_chunk(chunk_samples(4.0, n=peer_gating.chunk_samples))[0]
    assert isinstance(peer_start, SpeechStart)
    assert peer_start.pre_roll.size == 0


def _capture_span(index: int) -> AudioCaptureSpan:
    start = index * 512
    end = start + 512
    return AudioCaptureSpan(
        capture_epoch=1,
        callback_sequence=index,
        source_sample_rate_hz=16000,
        source_start_sample=start,
        source_end_sample=end,
        source_start_monotonic_s=start / 16000,
        source_end_monotonic_s=end / 16000,
        normalized_sample_rate_hz=16000,
        normalized_start_sample=start,
        normalized_end_sample=end,
    )


def test_peer_ordinary_onset_retains_500ms_pre_roll() -> None:
    gating = create_peer_vad_gating(
        SequenceVadEngine(probs=[0.0] * 16 + [0.9, 0.9, 0.9]),
        sample_rate_hz=16000,
        ring_buffer_ms=500,
        hangover_ms=500,
    )
    events: list[object] = []
    for index in range(19):
        events.extend(gating.process_chunk(chunk_samples(float(index), n=gating.chunk_samples)))
    start = next(event for event in events if isinstance(event, SpeechStart))
    assert start.genuine_onset is True
    assert start.pre_roll.size == 8000


def test_self_hysteresis_cancels_pending_pause_and_preserves_natural_reset() -> None:
    gating = VadGating(
        SequenceVadEngine(probs=[0.5, 0.39, 0.45, 0.39, 0.39, 0.45, 0.5]),
        sample_rate_hz=16000,
        ring_buffer_ms=64,
        speech_threshold=0.5,
        continuation_threshold=vad_exit_threshold(0.5),
        hangover_ms=64,
    )

    assert isinstance(
        gating.process_chunk(chunk_samples(1.0, n=gating.chunk_samples))[0], SpeechStart
    )
    gating.process_chunk(chunk_samples(2.0, n=gating.chunk_samples))
    assert gating.last_observation_was_speech is False
    gating.process_chunk(chunk_samples(3.0, n=gating.chunk_samples))
    assert gating.last_observation_was_speech is True
    gating.process_chunk(chunk_samples(4.0, n=gating.chunk_samples))
    ended = gating.process_chunk(chunk_samples(5.0, n=gating.chunk_samples))

    end = next(event for event in ended if isinstance(event, SpeechEnd))
    assert end.trailing_silence_ms == 64
    assert gating.in_speech is False
    assert gating.process_chunk(chunk_samples(6.0, n=gating.chunk_samples)) == []
    successor = gating.process_chunk(chunk_samples(7.0, n=gating.chunk_samples))
    assert isinstance(successor[0], SpeechStart)
    assert successor[0].utterance_id != end.utterance_id
    assert successor[0].genuine_onset is True
