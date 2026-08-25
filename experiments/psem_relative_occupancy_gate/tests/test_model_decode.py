from __future__ import annotations

import inspect

import numpy as np
import pytest

from experiments.psem_relative_occupancy_gate.contracts import (
    ActivityInterval,
    AnchorEpisode,
    AnchorLifecycle,
    Trace,
)
from experiments.psem_relative_occupancy_gate.model_decode import (
    CausalAnchorTracker,
    CausalEnrollmentConfig,
    ModelObservation,
    oracle_anchor_mapping,
    posterior_cells,
    simulate_causal_session,
)


def _trace(probabilities: list[list[float]], frontiers: list[int]) -> Trace:
    frame_count = len(probabilities)
    return Trace(
        source_id="fixture",
        family="fixture_family",
        slot_ids=("slot-0", "slot-1"),
        probabilities=np.asarray(probabilities, dtype=np.float32),
        frame_start_samples=np.arange(frame_count, dtype=np.int64) * 1600,
        frame_end_samples=(np.arange(frame_count, dtype=np.int64) + 1) * 1600,
        evidence_frontier_samples=np.asarray(frontiers, dtype=np.int64),
        slot_alive=np.ones((frame_count, 2), dtype=np.bool_),
        state_reset=np.asarray([True] + [False] * (frame_count - 1), dtype=np.bool_),
        metadata={},
    )


def _observation(
    start: int,
    probabilities: tuple[float, float],
    frontier: int,
    *,
    masked: bool = False,
    speech_present: bool = True,
    valid: bool = True,
    reset: bool = False,
) -> ModelObservation:
    return ModelObservation(
        start_sample=start,
        end_sample=start + 1600,
        probabilities=probabilities,
        slot_alive=(True, True),
        trace_valid=valid,
        state_reset=reset,
        masked=masked,
        speech_present=speech_present,
        evidence_frontier_sample=max(frontier, start + 1600),
    )


def test_oracle_mapping_is_episode_level_with_lowest_slot_tie() -> None:
    trace = _trace([[0.9, 0.1], [0.1, 0.9]], [1600, 3200])
    intervals = (ActivityInterval(0, 3200, ("A",), False),)
    cells = posterior_cells(trace, intervals, 0, 3200)
    episode = AnchorEpisode("episode", "fixture", "A", 0, 0, 3200, None)
    mapping = oracle_anchor_mapping(episode, cells, trace.slot_ids)
    assert mapping.slot_index == 0
    assert mapping.support_scores == pytest.approx((0.5, 0.5))
    assert [int(np.argmax(value.probabilities)) for value in cells] == [0, 1]


def test_oracle_mapping_excludes_masked_and_non_anchor_active_cells() -> None:
    trace = _trace([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8]], [1600, 3200, 4800])
    intervals = (
        ActivityInterval(0, 1600, ("A",), True),
        ActivityInterval(1600, 3200, ("B",), False),
        ActivityInterval(3200, 4800, ("A",), False),
    )
    cells = posterior_cells(trace, intervals, 0, 4800)
    episode = AnchorEpisode("episode", "fixture", "A", 0, 0, 4800, None)
    mapping = oracle_anchor_mapping(episode, cells, trace.slot_ids)
    assert mapping.slot_index == 1
    assert mapping.support_weight_samples == 1600


def test_causal_tracker_contract_has_no_gt_speaker_identity_input() -> None:
    parameters = inspect.signature(CausalAnchorTracker.advance_group).parameters
    assert set(parameters) == {"self", "observations"}
    observation_fields = set(ModelObservation.__dataclass_fields__)
    assert "active_speakers" not in observation_fields
    assert "anchor_speaker" not in observation_fields


def test_buffered_history_confirms_once_at_recorded_frontier() -> None:
    observations = tuple(
        _observation(index * 1600, (0.9, 0.1), 16680, reset=index == 0) for index in range(5)
    )
    result = simulate_causal_session(
        source_id="fixture",
        slot_ids=("slot-0", "slot-1"),
        observations=observations,
        enrollment_config=CausalEnrollmentConfig(0.8, 0.2, 6400),
        replacement_confirmation_samples=1600,
        anchor_threshold=0.5,
        other_threshold=0.5,
        silence_reset_samples=19200,
        scored_end_sample=8000,
    )
    assert len(result.enrollments) == 1
    assert result.enrollments[0].candidate_start_sample == 0
    assert result.enrollments[0].decoder_emit_sample == 16680
    assert result.episodes[0].end_emit_sample >= result.episodes[0].anchor_emit_sample
    assert all(
        span.lifecycle is not AnchorLifecycle.ANCHORED
        or span.start_sample >= result.episodes[0].anchor_emit_sample
        for span in result.timeline
    )


def test_opening_overlap_and_masked_singleton_do_not_enroll() -> None:
    observations = (
        _observation(0, (0.9, 0.9), 1600),
        _observation(1600, (0.9, 0.1), 3200, masked=True),
        _observation(3200, (0.9, 0.1), 4800),
    )
    result = simulate_causal_session(
        source_id="fixture",
        slot_ids=("slot-0", "slot-1"),
        observations=observations,
        enrollment_config=CausalEnrollmentConfig(0.8, 0.2, 3200),
        replacement_confirmation_samples=1600,
        anchor_threshold=0.5,
        other_threshold=0.5,
        silence_reset_samples=19200,
        scored_end_sample=4800,
    )
    assert result.enrollments == ()


def test_slot_invalidation_enters_uncertain_and_disables_cut() -> None:
    observations = (
        _observation(0, (0.9, 0.1), 1600),
        _observation(1600, (0.9, 0.1), 3200),
        _observation(3200, (0.1, 0.9), 4800, valid=False),
    )
    result = simulate_causal_session(
        source_id="fixture",
        slot_ids=("slot-0", "slot-1"),
        observations=observations,
        enrollment_config=CausalEnrollmentConfig(0.8, 0.2, 3200),
        replacement_confirmation_samples=1600,
        anchor_threshold=0.5,
        other_threshold=0.5,
        silence_reset_samples=19200,
        scored_end_sample=4800,
    )
    assert result.uncertain_entry_count == 1
    assert result.replacement_events == ()
    assert result.timeline[-1].lifecycle is AnchorLifecycle.ANCHOR_UNCERTAIN


def test_post_cut_does_not_inherit_non_anchor_slot() -> None:
    observations = (
        _observation(0, (0.9, 0.1), 1600),
        _observation(1600, (0.9, 0.1), 3200),
        _observation(3200, (0.1, 0.9), 4800),
        _observation(4800, (0.1, 0.9), 6400),
    )
    result = simulate_causal_session(
        source_id="fixture",
        slot_ids=("slot-0", "slot-1"),
        observations=observations,
        enrollment_config=CausalEnrollmentConfig(0.8, 0.2, 3200),
        replacement_confirmation_samples=1600,
        anchor_threshold=0.5,
        other_threshold=0.5,
        silence_reset_samples=19200,
        scored_end_sample=6400,
    )
    assert len(result.replacement_events) == 1
    assert len(result.enrollments) == 1
    assert result.episodes[0].end_reason == "speaker_induced_cut"


def test_final_speech_end_splits_lifecycle_at_exact_qualification() -> None:
    observations = (
        _observation(0, (0.9, 0.1), 1600),
        _observation(1600, (0.9, 0.1), 3200),
        ModelObservation(
            start_sample=3200,
            end_sample=6400,
            probabilities=(0.1, 0.1),
            slot_alive=(True, True),
            trace_valid=True,
            state_reset=False,
            masked=False,
            speech_present=False,
            evidence_frontier_sample=6400,
        ),
    )
    result = simulate_causal_session(
        source_id="fixture",
        slot_ids=("slot-0", "slot-1"),
        observations=observations,
        enrollment_config=CausalEnrollmentConfig(0.8, 0.2, 3200),
        replacement_confirmation_samples=1600,
        anchor_threshold=0.5,
        other_threshold=0.5,
        silence_reset_samples=1600,
        scored_end_sample=6400,
    )
    assert result.episodes[0].end_emit_sample == 4800
    assert result.timeline[-1].start_sample == 4800
    assert result.timeline[-1].lifecycle is AnchorLifecycle.UNANCHORED
