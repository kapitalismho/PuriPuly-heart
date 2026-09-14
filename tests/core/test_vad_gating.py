from __future__ import annotations

import numpy as np
import pytest

import puripuly_heart.core.vad.gating as gating_module
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


def test_vad_gating_emits_diagnostic_event_summaries() -> None:
    lines: list[str] = []
    probs = [0.9, 0.0, 0.0]
    gating = VadGating(
        SequenceVadEngine(probs=probs),
        sample_rate_hz=16000,
        ring_buffer_ms=64,
        hangover_ms=64,
        diagnostic_event_callback=lines.append,
        diagnostic_label="self",
    )

    for i in range(len(probs)):
        gating.process_chunk(chunk_samples(float(i + 1), n=gating.chunk_samples))

    assert any("[AudioDiag][VAD][self] event=SpeechStart" in line for line in lines)
    assert any("prob=0.900" in line and "threshold=0.4" in line for line in lines)
    assert any("[AudioDiag][VAD][self] event=SpeechEnd" in line for line in lines)


def test_run_diagnostics_preserve_pause_and_hysteresis_timing_without_idle_noise() -> None:
    lines: list[str] = []
    gating = VadGating(
        SequenceVadEngine(probs=[0.1] * 50 + [0.9, 0.6, 0.45, 0.3, 0.2, 0.1, 0.7]),
        sample_rate_hz=16000,
        speech_threshold=0.6,
        continuation_threshold=0.3,
        external_delivery_boundaries=True,
        diagnostic_event_callback=lines.append,
    )
    chunk = chunk_samples(1.0, n=gating.chunk_samples)
    for _ in range(50):
        gating.process_chunk(chunk)
    assert lines == []
    for _ in range(7):
        gating.process_chunk(chunk)
    end = gating.seal_active(reason="delivery_pause")
    assert end is not None
    records = [dict(field.split("=", 1) for field in line.split()[1:]) for line in lines]
    runs = [record for record in records if record["event"] == "VadRun"]
    assert [run["class"] for run in runs] == ["speech", "band", "non_speech", "speech"]
    assert [float(run["start_audio_ms"]) for run in runs] == [0, 64, 128, 192]
    assert [float(run["duration_ms"]) for run in runs] == [64, 64, 64, 32]
    assert [int(run["frame_count"]) for run in runs] == [2, 2, 2, 1]
    assert [float(run["prob_min"]) for run in runs] == [0.6, 0.3, 0.1, 0.7]
    assert [float(run["prob_max"]) for run in runs] == [0.9, 0.45, 0.2, 0.7]
    assert all(run["utterance_id"] == str(end.utterance_id)[:8] for run in runs)
    summary = records[-1]
    assert summary["event"] == "SpeechEnd"
    assert float(summary["max_non_speech_ms"]) == 64
    assert int(summary["band_frame_count"]) == 2
    assert float(summary["observed_audio_ms"]) == 224
    assert summary["observation_complete"] == "true"
    assert float(summary["onset_threshold"]) == 0.6
    assert float(summary["continuation_threshold"]) == 0.3


def test_run_diagnostics_mark_partial_observation_and_do_not_bridge_disabled_audio() -> None:
    lines: list[str] = []
    enabled = False
    gating = VadGating(
        SequenceVadEngine(probs=[0.9, 0.8, 0.1, 0.1, 0.1, 0.1]),
        sample_rate_hz=16000,
        external_delivery_boundaries=True,
        diagnostic_event_callback=lines.append,
        diagnostics_enabled=lambda: enabled,
    )
    chunk = chunk_samples(1.0, n=gating.chunk_samples)
    for _ in range(2):
        gating.process_chunk(chunk)
    assert lines == []
    enabled = True
    gating.process_chunk(chunk)
    gating.process_chunk(chunk)
    enabled = False
    gating.process_chunk(chunk)
    assert lines == []
    enabled = True
    gating.process_chunk(chunk)
    gating.seal_active(reason="delivery_pause")
    records = [dict(field.split("=", 1) for field in line.split()[1:]) for line in lines]
    run, summary = records
    assert run["class"] == "non_speech"
    assert float(run["start_audio_ms"]) == 160
    assert float(run["duration_ms"]) == 32
    assert summary["observation_complete"] == "false"
    assert float(summary["observed_audio_ms"]) == 96
    assert float(summary["speech_audio_ms"]) == 192
    assert float(summary["max_non_speech_ms"]) == 64
    assert int(summary["trailing_silence_ms"]) == 128


def test_run_diagnostics_flush_on_reset_without_leaking_into_next_segment() -> None:
    lines: list[str] = []
    gating = VadGating(
        SequenceVadEngine(probs=[0.9, 0.1, 0.9, 0.9]),
        sample_rate_hz=16000,
        external_delivery_boundaries=True,
        diagnostic_event_callback=lines.append,
    )
    chunk = chunk_samples(1.0, n=gating.chunk_samples)
    gating.process_chunk(chunk)
    first_id = gating.utterance_id
    gating.process_chunk(chunk)
    gating.reset()
    last_run = dict(field.split("=", 1) for field in lines[-1].split()[1:])
    assert last_run["event"] == "VadRun"
    assert last_run["utterance_id"] == str(first_id)[:8]
    assert last_run["class"] == "non_speech"
    assert float(last_run["duration_ms"]) == 32
    gating.process_chunk(chunk)
    gating.process_chunk(chunk)
    gating.seal_active(reason="delivery_deadline")
    run = dict(field.split("=", 1) for field in lines[-2].split()[1:])
    summary = dict(field.split("=", 1) for field in lines[-1].split()[1:])
    assert run["utterance_id"] != str(first_id)[:8]
    assert float(run["start_audio_ms"]) == 0
    assert float(run["duration_ms"]) == 64
    assert float(summary["max_non_speech_ms"]) == 0
    assert summary["observation_complete"] == "true"


def test_run_diagnostics_include_committed_candidate_frames() -> None:
    lines: list[str] = []
    gating = create_peer_vad_gating(
        SequenceVadEngine(probs=[0.9, 0.1, 0.9, 0.8, 0.7, 0.1]),
        sample_rate_hz=16000,
        ring_buffer_ms=64,
        hangover_ms=64,
        diagnostic_event_callback=lines.append,
    )
    chunk = chunk_samples(1.0, n=gating.chunk_samples)
    for _ in range(6):
        gating.process_chunk(chunk)
    end = gating.seal_active(reason="delivery_pause")
    assert end is not None
    records = [dict(field.split("=", 1) for field in line.split()[1:]) for line in lines]
    runs = [
        record
        for record in records
        if record["event"] == "VadRun" and record["utterance_id"] == str(end.utterance_id)[:8]
    ]
    assert [float(run["start_audio_ms"]) for run in runs] == [0, 96]
    assert [float(run["duration_ms"]) for run in runs] == [96, 32]
    assert float(runs[0]["prob_min"]) == 0.7
    assert float(runs[0]["prob_max"]) == 0.9
    assert records[-1]["observation_complete"] == "true"
    assert float(records[-1]["observed_audio_ms"]) == 128


def test_external_end_diagnostics_preserve_boundary_reason_and_audio_duration() -> None:
    lines: list[str] = []
    gating = VadGating(
        SequenceVadEngine(probs=[0.9, 0.0, 0.0, 0.9, 0.0]),
        sample_rate_hz=16000,
        external_delivery_boundaries=True,
        diagnostic_event_callback=lines.append,
        diagnostic_label="peer",
    )
    chunk = chunk_samples(1.0, n=gating.chunk_samples)
    for _ in range(3):
        gating.process_chunk(chunk)
    first_end = gating.seal_active_for_rollover(reason="delivery_deadline")
    gating.process_chunk(chunk)
    gating.process_chunk(chunk)
    second_end = gating.seal_active(reason="delivery_pause")
    assert first_end is not None
    assert second_end is not None
    assert first_end.utterance_id != second_end.utterance_id
    ends = [
        dict(field.split("=", 1) for field in line.split()[1:])
        for line in lines
        if "event=SpeechEnd " in line
    ]
    assert [end["utterance_id"] for end in ends] == [
        str(first_end.utterance_id)[:8],
        str(second_end.utterance_id)[:8],
    ]
    assert [end["reason"] for end in ends] == ["delivery_deadline", "delivery_pause"]
    assert [int(end["trailing_silence_ms"]) for end in ends] == [64, 32]
    assert [float(end["speech_audio_ms"]) for end in ends] == [96, 64]
    runs = [
        dict(field.split("=", 1) for field in line.split()[1:])
        for line in lines
        if "event=VadRun " in line
    ]
    assert [run["utterance_id"] for run in runs] == [
        str(first_end.utterance_id)[:8],
        str(first_end.utterance_id)[:8],
        str(second_end.utterance_id)[:8],
        str(second_end.utterance_id)[:8],
    ]
    assert [float(run["start_audio_ms"]) for run in runs] == [0, 32, 0, 32]
    assert [float(run["duration_ms"]) for run in runs] == [32, 64, 32, 32]
    assert [float(end["max_non_speech_ms"]) for end in ends] == [64, 32]
    assert all(end["observation_complete"] == "true" for end in ends)


def test_vad_gating_diagnostic_callback_failure_does_not_drop_speech_start() -> None:
    def raise_on_diagnostic(_message: str) -> None:
        raise RuntimeError("diagnostic sink unavailable")

    gating = VadGating(
        SequenceVadEngine(probs=[0.9]),
        sample_rate_hz=16000,
        ring_buffer_ms=64,
        hangover_ms=64,
        diagnostic_event_callback=raise_on_diagnostic,
        diagnostic_label="self",
    )

    events = gating.process_chunk(chunk_samples(1.0, n=gating.chunk_samples))

    assert len(events) == 1
    assert isinstance(events[0], SpeechStart)
    assert gating.in_speech is True
    assert gating.utterance_id == events[0].utterance_id


def test_vad_gating_diagnostic_callback_failure_does_not_drop_speech_end_or_reset() -> None:
    def raise_on_end(message: str) -> None:
        if "event=SpeechEnd" in message:
            raise RuntimeError("diagnostic sink unavailable")

    probs = [0.9, 0.0, 0.0]
    gating = VadGating(
        SequenceVadEngine(probs=probs),
        sample_rate_hz=16000,
        ring_buffer_ms=64,
        hangover_ms=64,
        diagnostic_event_callback=raise_on_end,
        diagnostic_label="self",
    )

    events = []
    for i in range(len(probs)):
        events.extend(gating.process_chunk(chunk_samples(float(i + 1), n=gating.chunk_samples)))

    assert any(isinstance(event, SpeechStart) for event in events)
    assert any(isinstance(event, SpeechChunk) for event in events)
    end = next(event for event in events if isinstance(event, SpeechEnd))
    assert end.trailing_silence_ms == 64
    assert gating.in_speech is False
    assert gating.utterance_id is None


def test_vad_gating_diagnostic_metric_failure_does_not_drop_events_or_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gating_module,
        "compute_audio_frame_metrics",
        lambda _frame: (_ for _ in ()).throw(RuntimeError("diagnostic metrics failed")),
        raising=False,
    )
    lines: list[str] = []
    probs = [0.9, 0.0, 0.0]
    gating = VadGating(
        SequenceVadEngine(probs=probs),
        sample_rate_hz=16000,
        ring_buffer_ms=64,
        hangover_ms=64,
        diagnostic_event_callback=lines.append,
        diagnostic_label="self",
    )

    events = []
    for i in range(len(probs)):
        events.extend(gating.process_chunk(chunk_samples(float(i + 1), n=gating.chunk_samples)))

    start = next(event for event in events if isinstance(event, SpeechStart))
    end = next(event for event in events if isinstance(event, SpeechEnd))
    assert start.utterance_id == end.utterance_id
    assert any(isinstance(event, SpeechChunk) for event in events)
    assert end.trailing_silence_ms == 64
    assert gating.in_speech is False
    assert gating.utterance_id is None
    assert any("[AudioDiag][VAD][self] event=SpeechEnd" in line for line in lines)


def test_vad_gating_skips_diagnostic_metrics_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gating_module,
        "compute_audio_frame_metrics",
        lambda _frame: (_ for _ in ()).throw(
            AssertionError("disabled VAD diagnostics must not compute metrics")
        ),
        raising=False,
    )
    lines: list[str] = []
    gating = VadGating(
        SequenceVadEngine(probs=[0.9]),
        sample_rate_hz=16000,
        ring_buffer_ms=64,
        hangover_ms=64,
        diagnostic_event_callback=lines.append,
        diagnostic_label="self",
        diagnostics_enabled=lambda: False,
    )

    gating.process_chunk(chunk_samples(1.0, n=gating.chunk_samples))

    assert lines == []
