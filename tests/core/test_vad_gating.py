from __future__ import annotations

import numpy as np
import pytest

import puripuly_heart.core.vad.gating as gating_module
from puripuly_heart.core.vad.gating import (
    PEER_MAX_SEGMENT_MS,
    PEER_SOFT_BOUNDARY_START_MS,
    PEER_SOFT_PAUSE_MS,
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


class CountingVadEngine:
    def __init__(self, probs: list[float]) -> None:
        self.probs = probs
        self.idx = 0
        self.reset_calls = 0

    def speech_probability(self, samples: np.ndarray, *, sample_rate_hz: int) -> float:
        _ = samples
        _ = sample_rate_hz
        prob = self.probs[self.idx]
        self.idx = min(self.idx + 1, len(self.probs) - 1)
        return prob

    def reset(self) -> None:
        self.reset_calls += 1


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


def test_vad_gating_default_max_segment_disabled_does_not_force_continuous_speech():
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

    assert gating.max_segment_ms is None
    assert any(isinstance(event, SpeechStart) for event in events)
    assert not any(isinstance(event, SpeechEnd) for event in events)
    assert gating.in_speech is True


def test_vad_gating_forces_max_duration_on_above_threshold_chunk_without_engine_reset():
    engine = CountingVadEngine(probs=[0.9, 0.9])
    gating = VadGating(
        engine,
        sample_rate_hz=16000,
        ring_buffer_ms=64,
        hangover_ms=640,
        max_segment_ms=64,
    )

    start_events = gating.process_chunk(chunk_samples(1.0, n=gating.chunk_samples))
    boundary_events = gating.process_chunk(chunk_samples(2.0, n=gating.chunk_samples))

    start = next(event for event in start_events if isinstance(event, SpeechStart))
    chunk = next(event for event in boundary_events if isinstance(event, SpeechChunk))
    end = next(event for event in boundary_events if isinstance(event, SpeechEnd))

    assert chunk.utterance_id == start.utterance_id
    assert end.utterance_id == start.utterance_id
    assert end.reason == "max_duration"
    assert end.trailing_silence_ms == 0
    assert gating.in_speech is False
    assert gating.utterance_id is None
    assert engine.reset_calls == 0
    recent_audio = gating._ring.get_last_samples(gating._ring.capacity_samples)
    assert recent_audio.shape[0] == 1024
    assert np.allclose(recent_audio[:512], 1.0)
    assert np.allclose(recent_audio[512:], 2.0)


def test_vad_gating_does_not_force_on_below_threshold_chunk_that_reaches_budget():
    gating = VadGating(
        SequenceVadEngine(probs=[0.9, 0.0]),
        sample_rate_hz=16000,
        ring_buffer_ms=64,
        hangover_ms=96,
        max_segment_ms=64,
    )

    gating.process_chunk(chunk_samples(1.0, n=gating.chunk_samples))
    events = gating.process_chunk(chunk_samples(2.0, n=gating.chunk_samples))

    assert any(isinstance(event, SpeechChunk) for event in events)
    assert not any(isinstance(event, SpeechEnd) for event in events)
    assert gating.in_speech is True


def test_vad_gating_prefers_silence_end_when_budget_reached_during_hangover():
    gating = VadGating(
        SequenceVadEngine(probs=[0.9, 0.0, 0.0]),
        sample_rate_hz=16000,
        ring_buffer_ms=64,
        hangover_ms=64,
        max_segment_ms=64,
    )

    events = []
    for i in range(3):
        events.extend(gating.process_chunk(chunk_samples(float(i + 1), n=gating.chunk_samples)))

    end = next(event for event in events if isinstance(event, SpeechEnd))
    assert end.reason == "silence"
    assert end.trailing_silence_ms == 64


def test_vad_gating_excludes_pre_roll_from_max_segment_accounting():
    gating = VadGating(
        SequenceVadEngine(probs=[0.0, 0.0, 0.9, 0.9, 0.9]),
        sample_rate_hz=16000,
        ring_buffer_ms=64,
        hangover_ms=640,
        max_segment_ms=96,
    )

    gating.process_chunk(chunk_samples(0.0, n=gating.chunk_samples))
    gating.process_chunk(chunk_samples(1.0, n=gating.chunk_samples))
    start_events = gating.process_chunk(chunk_samples(2.0, n=gating.chunk_samples))
    below_budget_events = gating.process_chunk(chunk_samples(3.0, n=gating.chunk_samples))
    budget_events = gating.process_chunk(chunk_samples(4.0, n=gating.chunk_samples))

    start = next(event for event in start_events if isinstance(event, SpeechStart))
    assert start.pre_roll.shape[0] == 1024
    assert np.allclose(start.pre_roll[:512], 0.0)
    assert np.allclose(start.pre_roll[512:], 1.0)
    assert not any(isinstance(event, SpeechEnd) for event in below_budget_events)
    end = next(event for event in budget_events if isinstance(event, SpeechEnd))
    assert end.reason == "max_duration"


def test_vad_gating_forces_when_debounce_commit_already_exceeds_budget():
    gating = VadGating(
        SequenceVadEngine(probs=[0.9, 0.9, 0.9]),
        sample_rate_hz=16000,
        ring_buffer_ms=96,
        hangover_ms=640,
        max_segment_ms=64,
        start_debounce_chunks=3,
        start_commit_chunks=3,
    )

    assert gating.process_chunk(chunk_samples(1.0, n=gating.chunk_samples)) == []
    assert gating.process_chunk(chunk_samples(2.0, n=gating.chunk_samples)) == []
    events = gating.process_chunk(chunk_samples(3.0, n=gating.chunk_samples))

    assert [type(event) for event in events] == [SpeechStart, SpeechChunk, SpeechChunk, SpeechEnd]
    end = events[-1]
    assert isinstance(end, SpeechEnd)
    assert end.reason == "max_duration"
    assert end.trailing_silence_ms == 0
    assert gating.in_speech is False
    assert gating.utterance_id is None


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
        max_segment_ms=64,
    )

    for value in range(5):
        gating.process_chunk(chunk_samples(float(value), n=gating.chunk_samples))

    recent_audio = gating._ring.get_last_samples(gating._ring.capacity_samples)
    chunks = recent_audio.reshape(5, gating.chunk_samples)
    assert [float(chunk[0]) for chunk in chunks] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_vad_gating_forced_continuation_uses_ring_overlap_after_debounce():
    gating = VadGating(
        SequenceVadEngine(probs=[0.9, 0.9, 0.9, 0.9, 0.9]),
        sample_rate_hz=16000,
        ring_buffer_ms=64,
        hangover_ms=640,
        max_segment_ms=96,
        start_debounce_chunks=2,
        start_commit_chunks=2,
    )

    first_commit_events = []
    first_commit_events.extend(gating.process_chunk(chunk_samples(10.0, n=gating.chunk_samples)))
    first_commit_events.extend(gating.process_chunk(chunk_samples(20.0, n=gating.chunk_samples)))
    boundary_events = gating.process_chunk(chunk_samples(30.0, n=gating.chunk_samples))
    first_candidate_events = gating.process_chunk(chunk_samples(40.0, n=gating.chunk_samples))
    second_commit_events = gating.process_chunk(chunk_samples(50.0, n=gating.chunk_samples))

    first_start = next(event for event in first_commit_events if isinstance(event, SpeechStart))
    boundary_chunk = next(event for event in boundary_events if isinstance(event, SpeechChunk))
    forced_end = next(event for event in boundary_events if isinstance(event, SpeechEnd))
    second_start = next(event for event in second_commit_events if isinstance(event, SpeechStart))
    second_actual_chunks = [
        second_start.chunk,
        *[event.chunk for event in second_commit_events if isinstance(event, SpeechChunk)],
    ]

    assert first_candidate_events == []
    assert boundary_chunk.utterance_id == first_start.utterance_id
    assert forced_end.utterance_id == first_start.utterance_id
    assert forced_end.reason == "max_duration"
    assert second_start.utterance_id != first_start.utterance_id
    assert second_start.pre_roll.shape[0] == 1024
    assert np.allclose(second_start.pre_roll[:512], 20.0)
    assert np.allclose(second_start.pre_roll[512:], 30.0)
    assert [float(chunk[0]) for chunk in second_actual_chunks] == [40.0, 50.0]
    assert all(not np.allclose(chunk, 30.0) for chunk in second_actual_chunks)


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


def test_create_peer_vad_gating_uses_helper_defaults():
    gating = create_peer_vad_gating(
        SequenceVadEngine(probs=[0.0]),
        sample_rate_hz=16000,
        ring_buffer_ms=64,
        hangover_ms=64,
    )

    assert getattr(gating_module, "PEER_MAX_SEGMENT_MS", None) == 7000
    assert gating.max_segment_ms == PEER_MAX_SEGMENT_MS
    assert gating.soft_boundary_start_ms == PEER_SOFT_BOUNDARY_START_MS
    assert gating.soft_pause_ms == PEER_SOFT_PAUSE_MS
    assert gating.speech_threshold == PEER_VAD_SPEECH_THRESHOLD
    assert gating.start_debounce_chunks == PEER_VAD_START_DEBOUNCE_CHUNKS
    assert gating.start_commit_chunks == PEER_VAD_START_COMMIT_CHUNKS
    assert gating.candidate_log_label == "Peer"


def test_peer_vad_gating_uses_160ms_pause_that_starts_inside_soft_window() -> None:
    probs = [0.9] * 157 + [0.0] * 5
    gating = create_peer_vad_gating(
        SequenceVadEngine(probs=probs),
        sample_rate_hz=16000,
        ring_buffer_ms=500,
        hangover_ms=500,
    )

    events = []
    for index in range(len(probs)):
        events.extend(gating.process_chunk(chunk_samples(float(index + 1), n=gating.chunk_samples)))

    end = next(event for event in events if isinstance(event, SpeechEnd))
    assert end.reason == "soft_pause"
    assert end.trailing_silence_ms == 160
    assert len(probs) * 32 == 5184


def test_peer_vad_gating_does_not_reuse_pause_that_started_before_soft_window() -> None:
    probs = [0.9] * 153 + [0.0] * 16
    gating = create_peer_vad_gating(
        SequenceVadEngine(probs=probs),
        sample_rate_hz=16000,
        ring_buffer_ms=500,
        hangover_ms=500,
    )

    events = []
    for index in range(len(probs)):
        events.extend(gating.process_chunk(chunk_samples(float(index + 1), n=gating.chunk_samples)))

    end = next(event for event in events if isinstance(event, SpeechEnd))
    assert end.reason == "silence"
    assert end.trailing_silence_ms == 512


def test_peer_vad_gating_hard_cap_wins_when_soft_pause_completes_after_7000ms() -> None:
    probs = [0.9] * 214 + [0.0] * 5
    gating = create_peer_vad_gating(
        SequenceVadEngine(probs=probs),
        sample_rate_hz=16000,
        ring_buffer_ms=500,
        hangover_ms=500,
    )

    events = []
    for index in range(len(probs)):
        events.extend(gating.process_chunk(chunk_samples(float(index + 1), n=gating.chunk_samples)))

    end = next(event for event in events if isinstance(event, SpeechEnd))
    assert end.reason == "max_duration"
    assert end.trailing_silence_ms == 160
    assert len(probs) * 32 == 7008


def test_peer_vad_gating_soft_continuation_preserves_overlap_and_new_duration_budget() -> None:
    first_segment = [0.9] * 157 + [0.0] * 5
    second_segment = [0.9] * 219
    probs = first_segment + second_segment
    gating = create_peer_vad_gating(
        SequenceVadEngine(probs=probs),
        sample_rate_hz=16000,
        ring_buffer_ms=500,
        hangover_ms=500,
    )

    events_by_chunk = []
    for index in range(len(probs)):
        events_by_chunk.append(
            gating.process_chunk(chunk_samples(float(index + 1), n=gating.chunk_samples))
        )

    first_end = next(
        event
        for events in events_by_chunk
        for event in events
        if isinstance(event, SpeechEnd) and event.reason == "soft_pause"
    )
    second_start = next(
        event
        for events in events_by_chunk[len(first_segment) :]
        for event in events
        if isinstance(event, SpeechStart)
    )
    second_end = next(
        event
        for events in events_by_chunk[len(first_segment) :]
        for event in events
        if isinstance(event, SpeechEnd)
    )

    assert first_end.utterance_id != second_start.utterance_id
    assert second_start.pre_roll.shape[0] == 8000
    overlap_tail = second_start.pre_roll[-5 * gating.chunk_samples :].reshape(
        5, gating.chunk_samples
    )
    assert [float(chunk[0]) for chunk in overlap_tail] == [158.0, 159.0, 160.0, 161.0, 162.0]
    assert not any(
        isinstance(event, SpeechEnd) for events in events_by_chunk[-2:-1] for event in events
    )
    assert second_end.reason == "max_duration"
    assert second_end.trailing_silence_ms == 0


def test_peer_vad_gating_30s_continuous_speech_separates_unique_audio_from_overlap() -> None:
    chunk_count = 938
    probs = [0.9] * chunk_count
    gating = create_peer_vad_gating(
        SequenceVadEngine(probs=probs),
        sample_rate_hz=16000,
        ring_buffer_ms=500,
        hangover_ms=500,
    )

    events = []
    for index in range(chunk_count):
        events.extend(gating.process_chunk(chunk_samples(float(index + 1), n=gating.chunk_samples)))

    starts = [event for event in events if isinstance(event, SpeechStart)]
    chunks = [event for event in events if isinstance(event, SpeechChunk)]
    ends = [event for event in events if isinstance(event, SpeechEnd)]
    actual_sample_count = sum(len(event.chunk) for event in [*starts, *chunks])
    overlap_sample_count = sum(len(event.pre_roll) for event in starts)

    assert len(starts) == 5
    assert len(ends) == 4
    assert all(event.reason == "max_duration" for event in ends)
    assert all(event.trailing_silence_ms == 0 for event in ends)
    assert actual_sample_count == chunk_count * gating.chunk_samples
    assert [len(event.pre_roll) for event in starts] == [0, 8000, 8000, 8000, 8000]
    assert overlap_sample_count == 4 * 8000


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
