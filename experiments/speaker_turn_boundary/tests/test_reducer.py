from __future__ import annotations

import numpy as np
import pytest

from experiments.speaker_turn_boundary.reducer import (
    ReductionProfile,
    StreamingReducer,
    batch_binary_decisions,
    batch_reduce,
)

TRACK_COUNT = 4


def _run_streaming(
    probabilities: np.ndarray,
    profile: ReductionProfile,
    *,
    epoch_end: int = 40000,
) -> StreamingReducer:
    reducer = StreamingReducer(
        profile,
        track_count=probabilities.shape[1],
        audio_epoch=0,
        sample_count_at_epoch_end=epoch_end,
    )
    for frame_index in range(probabilities.shape[0]):
        reducer.emit(frame_index, probabilities[frame_index])
    reducer.finalize(epoch_end_count=epoch_end)
    return reducer


def _signature(boundaries) -> list[tuple]:
    return [
        (item.onset_output_frame, item.confirmed_output_frame, item.track_index)
        for item in boundaries
    ]


@pytest.mark.parametrize("median_width", [1, 3, 11])
@pytest.mark.parametrize("persistence", [1, 2, 3])
@pytest.mark.parametrize("policy", ["new_speaker_onset", "dominant_replacement"])
def test_streaming_equals_batch(median_width, persistence, policy):
    rng = np.random.default_rng(41)
    probabilities = rng.uniform(0.0, 1.0, (200, TRACK_COUNT)).astype(np.float32)
    profile = ReductionProfile(
        threshold=0.5, persistence=persistence, policy=policy, median_width=median_width
    )
    batch = batch_reduce(probabilities, profile, audio_epoch=0)
    streaming = _run_streaming(probabilities, profile)
    assert _signature(batch) == _signature(streaming.boundaries)


def test_onset_persistence_handcrafted():
    probabilities = np.zeros((12, 2), dtype=np.float32)
    probabilities[2:6, 0] = 1.0
    probabilities[9:11, 1] = 1.0
    profile = ReductionProfile(threshold=0.5, persistence=2, policy="new_speaker_onset")
    boundaries = batch_reduce(probabilities, profile, audio_epoch=0)
    assert _signature(boundaries) == [(2, 3, 0), (9, 10, 1)]


def test_sustained_run_emits_single_event():
    probabilities = np.ones((10, 1), dtype=np.float32) * 0.9
    profile = ReductionProfile(threshold=0.5, persistence=2, policy="new_speaker_onset")
    boundaries = batch_reduce(probabilities, profile, audio_epoch=0)
    assert boundaries == []


def test_initial_onset_skipped_like_initial_start():
    probabilities = np.zeros((10, 1), dtype=np.float32)
    probabilities[0:3, 0] = 1.0
    probabilities[6:8, 0] = 1.0
    profile = ReductionProfile(threshold=0.5, persistence=1, policy="new_speaker_onset")
    boundaries = batch_reduce(probabilities, profile, audio_epoch=0)
    assert _signature(boundaries) == [(6, 6, 0)]


def test_onset_needs_prior_inactivity():
    probabilities = np.ones((8, 1), dtype=np.float32)
    probabilities[3, 0] = 0.0
    profile = ReductionProfile(threshold=0.5, persistence=1, policy="new_speaker_onset")
    boundaries = batch_reduce(probabilities, profile, audio_epoch=0)
    assert _signature(boundaries) == [(4, 4, 0)]


def test_dominant_replacement_handcrafted():
    probabilities = np.zeros((10, 2), dtype=np.float32)
    probabilities[:5, 0] = 0.9
    probabilities[5:, 1] = 0.9
    profile = ReductionProfile(threshold=0.5, persistence=2, policy="dominant_replacement")
    boundaries = batch_reduce(probabilities, profile, audio_epoch=0)
    assert _signature(boundaries) == [(5, 6, 1)]


def test_dominant_replacement_requires_old_inactive():
    probabilities = np.zeros((10, 2), dtype=np.float32)
    probabilities[:, 0] = 0.9
    probabilities[5:, 1] = 0.9
    profile = ReductionProfile(threshold=0.5, persistence=2, policy="dominant_replacement")
    boundaries = batch_reduce(probabilities, profile, audio_epoch=0)
    assert boundaries == []


def test_median_filter_changes_events():
    probabilities = np.zeros((20, 1), dtype=np.float32)
    probabilities[7, 0] = 1.0
    probabilities[11:13, 0] = 1.0
    on = batch_reduce(
        probabilities,
        ReductionProfile(0.5, 1, "new_speaker_onset", median_width=11),
    )
    off = batch_reduce(
        probabilities,
        ReductionProfile(0.5, 1, "new_speaker_onset", median_width=1),
    )
    assert len(on) < len(off)


def test_streaming_safe_frontier_progression():
    rng = np.random.default_rng(2)
    probabilities = rng.uniform(0.0, 1.0, (30, 2)).astype(np.float32)
    profile = ReductionProfile(0.5, 2, "new_speaker_onset")
    reducer = StreamingReducer(
        profile, track_count=2, audio_epoch=0, sample_count_at_epoch_end=100000
    )
    previous = -1
    for frame_index in range(probabilities.shape[0]):
        reducer.emit(frame_index, probabilities[frame_index])
        safe = reducer.safe_boundary_frontier_sample()
        assert safe >= previous
        previous = safe
    reducer.finalize(epoch_end_count=100000)
    assert reducer.safe_boundary_frontier_sample() == 100000


def test_streaming_rejects_out_of_order_frames():
    reducer = StreamingReducer(
        ReductionProfile(0.5, 1, "new_speaker_onset"),
        track_count=2,
        audio_epoch=0,
        sample_count_at_epoch_end=100,
    )
    reducer.emit(0, np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError):
        reducer.emit(2, np.zeros(2, dtype=np.float32))


def test_batch_decisions_median_matches_scipy():
    from scipy.signal import medfilt

    rng = np.random.default_rng(8)
    probabilities = rng.uniform(0.0, 1.0, (60, 3)).astype(np.float32)
    profile = ReductionProfile(0.5, 1, "new_speaker_onset", median_width=11)
    binary = batch_binary_decisions(probabilities, profile)
    expected = medfilt((probabilities > 0.5).astype(np.float32), kernel_size=(11, 1)).astype(
        np.float32
    )
    assert np.array_equal(binary, expected)


def test_profile_validation():
    with pytest.raises(ValueError):
        ReductionProfile(1.5, 1, "new_speaker_onset")
    with pytest.raises(ValueError):
        ReductionProfile(0.5, 0, "new_speaker_onset")
    with pytest.raises(ValueError):
        ReductionProfile(0.5, 1, "unknown_policy")
    with pytest.raises(ValueError):
        ReductionProfile(0.5, 1, "new_speaker_onset", median_width=2)
