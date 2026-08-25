from __future__ import annotations

import pytest

from experiments.psem_relative_occupancy_gate.contracts import (
    ActivityInterval,
    AnchorLifecycle,
    RelativeState,
    relative_state,
)
from experiments.psem_relative_occupancy_gate.decoder import (
    RelativeObservation,
    ReplacementDecoder,
    simulate_gt_session,
)
from experiments.psem_relative_occupancy_gate.evaluate import (
    audit_gt_session,
    timeline_exposure,
)


@pytest.mark.parametrize(
    ("speakers", "expected"),
    [
        (("A",), RelativeState.ANCHOR_ONLY),
        (("A", "B"), RelativeState.ANCHOR_PLUS_OTHER),
        (("B",), RelativeState.OTHER_ONLY),
        ((), RelativeState.NONE),
        (("A", "B", "C"), RelativeState.ANCHOR_PLUS_OTHER),
        (("B", "C"), RelativeState.OTHER_ONLY),
    ],
)
def test_relative_occupancy_states(speakers: tuple[str, ...], expected: RelativeState) -> None:
    assert relative_state("A", speakers) is expected


def test_activity_interval_mask_round_trip() -> None:
    original = ActivityInterval(100, 200, ("A",), True)
    assert ActivityInterval.from_dict(original.to_dict()) == original


def test_activity_interval_explicit_mask_combines_with_raw_mask_fields() -> None:
    row = {
        "start_sample": 100,
        "end_sample": 200,
        "active_speakers": ["A"],
        "masked": False,
        "ambiguous": True,
    }
    assert ActivityInterval.from_dict(row).masked is True


def test_overlap_never_confirms_replacement() -> None:
    decoder = ReplacementDecoder("fixture", 3200)
    event = decoder.advance(
        RelativeObservation(0, 16000, RelativeState.ANCHOR_PLUS_OTHER, False, 16000),
        lifecycle=AnchorLifecycle.ANCHORED,
        anchor_id="A",
        anchor_episode_id="episode",
    )
    assert event is None
    assert decoder.pending_boundary_sample is None


def test_persistent_other_only_backdates_boundary() -> None:
    decoder = ReplacementDecoder("fixture", 3200)
    assert (
        decoder.advance(
            RelativeObservation(1000, 2600, RelativeState.OTHER_ONLY, False, 2600),
            lifecycle=AnchorLifecycle.ANCHORED,
            anchor_id="A",
            anchor_episode_id="episode",
        )
        is None
    )
    event = decoder.advance(
        RelativeObservation(2600, 4200, RelativeState.OTHER_ONLY, False, 4200),
        lifecycle=AnchorLifecycle.ANCHORED,
        anchor_id="A",
        anchor_episode_id="episode",
    )
    assert event is not None
    assert event.boundary_source_sample == 1000
    assert event.decoder_emit_sample == 4200
    assert event.model_evidence_frontier_sample == 4200


def test_decoder_uses_scalar_observation_frontier_without_interpolation() -> None:
    decoder = ReplacementDecoder("fixture", 500)
    event = decoder.advance(
        RelativeObservation(0, 1000, RelativeState.OTHER_ONLY, False, 10000),
        lifecycle=AnchorLifecycle.ANCHORED,
        anchor_id="A",
        anchor_episode_id="episode",
    )
    assert event is not None
    assert event.boundary_source_sample == 0
    assert event.model_evidence_frontier_sample == 10000
    assert event.decoder_emit_sample == 10000


def test_mask_pauses_replacement_evidence() -> None:
    decoder = ReplacementDecoder("fixture", 3200)
    assert (
        decoder.advance(
            RelativeObservation(0, 1600, RelativeState.OTHER_ONLY, False, 1600),
            lifecycle=AnchorLifecycle.ANCHORED,
            anchor_id="A",
            anchor_episode_id="episode",
        )
        is None
    )
    assert (
        decoder.advance(
            RelativeObservation(1600, 3200, None, True, 3200),
            lifecycle=AnchorLifecycle.ANCHORED,
            anchor_id="A",
            anchor_episode_id="episode",
        )
        is None
    )
    event = decoder.advance(
        RelativeObservation(3200, 4800, RelativeState.OTHER_ONLY, False, 4800),
        lifecycle=AnchorLifecycle.ANCHORED,
        anchor_id="A",
        anchor_episode_id="episode",
    )
    assert event is not None
    assert event.boundary_source_sample == 0
    assert event.decoder_emit_sample == 4800


def test_same_anchor_pause_resume_has_no_cut() -> None:
    intervals = (
        ActivityInterval(0, 1600, ("A",), False),
        ActivityInterval(1600, 4800, (), False),
        ActivityInterval(4800, 12800, ("A",), False),
    )
    result = simulate_gt_session(
        source_id="fixture",
        intervals=intervals,
        confirmation_samples=1600,
        enrollment_samples=3200,
        silence_reset_samples=19200,
        initial_lifecycle=AnchorLifecycle.ANCHORED,
        initial_anchor="A",
    )
    assert result.events == ()
    assert len(result.episodes) == 1


def test_uncertain_lifecycle_disables_cut() -> None:
    decoder = ReplacementDecoder("fixture", 1600)
    event = decoder.advance(
        RelativeObservation(0, 3200, RelativeState.OTHER_ONLY, False, 3200),
        lifecycle=AnchorLifecycle.ANCHOR_UNCERTAIN,
        anchor_id=None,
        anchor_episode_id=None,
    )
    assert event is None
    assert decoder.pending_boundary_sample is None


def test_post_cut_requires_fresh_enrollment() -> None:
    intervals = (
        ActivityInterval(0, 1600, ("A",), False),
        ActivityInterval(1600, 12800, ("B",), False),
    )
    result = simulate_gt_session(
        source_id="fixture",
        intervals=intervals,
        confirmation_samples=1600,
        enrollment_samples=3200,
        silence_reset_samples=19200,
        initial_lifecycle=AnchorLifecycle.ANCHORED,
        initial_anchor="A",
    )
    assert result.events[0].boundary_source_sample == 1600
    assert result.events[0].decoder_emit_sample == 3200
    next_spans = [span for span in result.timeline if span.start_sample >= 3200]
    assert next_spans[0].lifecycle is AnchorLifecycle.UNANCHORED
    assert next_spans[0].end_sample - next_spans[0].start_sample == 3200
    assert result.enrollments[1].anchor_id == "B"
    assert result.enrollments[1].anchor_emit_sample == 6400
    exposure = timeline_exposure(result.timeline)
    assert exposure["unanchored_active_speech_seconds"] == 0.2
    assert (
        exposure["exclusive_other_contamination_upper_bound_seconds"]
        >= exposure["state_seconds"]["OTHER_ONLY"]
    )


def test_gt_replay_audit_proves_exact_boundary_and_emit() -> None:
    result = simulate_gt_session(
        source_id="fixture",
        intervals=(
            ActivityInterval(0, 1600, ("A",), False),
            ActivityInterval(1600, 3200, ("B",), False),
            ActivityInterval(3200, 4000, ("B",), True),
            ActivityInterval(4000, 5600, ("B",), False),
        ),
        confirmation_samples=3200,
        enrollment_samples=3200,
        silence_reset_samples=19200,
        initial_lifecycle=AnchorLifecycle.ANCHORED,
        initial_anchor="A",
    )
    audit = audit_gt_session(result)
    assert audit["passed"] is True
    assert audit["episodes"][0]["expected_boundary_source_sample"] == 1600
    assert audit["episodes"][0]["expected_qualification_sample"] == 5600
    assert result.events[0].model_evidence_frontier_sample == 5600
    assert result.events[0].decoder_emit_sample == 5600


def test_opening_overlap_cannot_enroll() -> None:
    result = simulate_gt_session(
        source_id="fixture",
        intervals=(ActivityInterval(0, 16000, ("A", "B"), False),),
        confirmation_samples=1600,
        enrollment_samples=3200,
        silence_reset_samples=19200,
    )
    assert result.enrollments == ()
    assert all(span.lifecycle is AnchorLifecycle.UNANCHORED for span in result.timeline)


def test_masked_singleton_cannot_enroll() -> None:
    result = simulate_gt_session(
        source_id="fixture",
        intervals=(ActivityInterval(0, 16000, ("A",), True),),
        confirmation_samples=1600,
        enrollment_samples=3200,
        silence_reset_samples=19200,
    )
    assert result.enrollments == ()


def test_long_silence_ends_lifecycle() -> None:
    result = simulate_gt_session(
        source_id="fixture",
        intervals=(
            ActivityInterval(0, 1600, ("A",), False),
            ActivityInterval(1600, 22400, (), False),
        ),
        confirmation_samples=1600,
        enrollment_samples=3200,
        silence_reset_samples=19200,
        initial_lifecycle=AnchorLifecycle.ANCHORED,
        initial_anchor="A",
    )
    assert result.episodes[0].end_emit_sample == 20800
    assert result.timeline[-1].lifecycle is AnchorLifecycle.UNANCHORED
