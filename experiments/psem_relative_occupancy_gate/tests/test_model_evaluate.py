from __future__ import annotations

from experiments.psem_relative_occupancy_gate.contracts import ActivityInterval
from experiments.psem_relative_occupancy_gate.decoder import (
    GTSessionResult,
    ReplacementEvent,
)
from experiments.psem_relative_occupancy_gate.model_decode import (
    CausalAnchorEpisode,
    CausalEnrollmentConfig,
    CausalSessionResult,
)
from experiments.psem_relative_occupancy_gate.model_evaluate import (
    AnnotatedCausalEpisode,
    PrimitiveRecord,
    causal_anchor_metrics,
    causal_product_metrics,
    count_causal_opportunities,
    exact_episode_contamination_samples,
    first_gt_singleton_opportunity,
    gt_singleton_opportunities,
    primitive_metrics,
    product_event_metrics,
    transition_timing,
)
from experiments.psem_relative_occupancy_gate.run_gate1 import _topology_slices


def test_primitive_metrics_preserve_duration_errors_and_frozen_threshold() -> None:
    records = (
        PrimitiveRecord("source", "episode", 800, 1600, True, False, 0.9, 0.1, 3200),
        PrimitiveRecord("source", "episode", 2400, 1600, False, True, 0.1, 0.9, 4800),
        PrimitiveRecord("source", "episode", 4000, 1600, True, False, 0.1, 0.9, 6400),
    )
    result = primitive_metrics(records, [0.5], (0.5, 0.5))
    selected = result["selected_operating_point"]
    assert selected["selection_rule"] == "frozen_gate1_dev_operating_point"
    assert (
        selected["state_duration_errors_seconds"]["false_OTHER_ONLY_inside_GT_ANCHOR_ONLY"] == 0.1
    )


def test_transition_timing_reports_model_evidence_availability() -> None:
    records = (
        PrimitiveRecord("source", "episode", 800, 1600, True, False, 0.9, 0.1, 3200),
        PrimitiveRecord("source", "episode", 2400, 1600, False, True, 0.1, 0.9, 4800),
    )
    result = transition_timing(records, 0.5, 0.5)
    assert result["ANCHOR_ONLY_TO_OTHER_ONLY"]["matched_within_2000ms"] == 1
    assert result["MODEL_EVIDENCE_AVAILABILITY"]["delay_ms"]["p50"] == 150.0


def test_gt_singleton_opportunity_pauses_across_mask() -> None:
    opportunity = first_gt_singleton_opportunity(
        (
            ActivityInterval(0, 1600, ("A",), False),
            ActivityInterval(1600, 3200, ("A",), True),
            ActivityInterval(3200, 4800, ("A",), False),
        ),
        start_sample=0,
        end_sample=4800,
        enrollment_samples=3200,
        silence_reset_samples=19200,
    )
    assert opportunity == ("A", 0, 4800)


def test_gt_opportunity_count_preserves_multiple_outer_sessions() -> None:
    opportunities = gt_singleton_opportunities(
        (
            ActivityInterval(0, 3200, ("A",), False),
            ActivityInterval(3200, 22400, (), False),
            ActivityInterval(22400, 25600, ("B",), False),
        ),
        start_sample=0,
        end_sample=25600,
        enrollment_samples=3200,
        silence_reset_samples=19200,
    )
    assert opportunities == (("A", 0, 3200), ("B", 22400, 25600))


def test_exact_contamination_excludes_masks_and_anchor_activity() -> None:
    intervals = (
        ActivityInterval(0, 1600, ("A",), False),
        ActivityInterval(1600, 3200, ("B",), False),
        ActivityInterval(3200, 4800, ("B",), True),
        ActivityInterval(4800, 6400, ("A", "B"), False),
    )
    assert (
        exact_episode_contamination_samples(
            intervals,
            anchor_speaker="A",
            start_sample=0,
            end_sample=6400,
        )
        == 1600
    )


def test_primary_contamination_partitions_at_next_reference_replacement() -> None:
    intervals = (
        ActivityInterval(0, 16000, ("A",), False),
        ActivityInterval(16000, 32000, ("B",), False),
        ActivityInterval(32000, 48000, ("C",), False),
    )
    reference_events = (
        ReplacementEvent("source", "R1", "A", 16000, 17600, 17600, None, 1600),
        ReplacementEvent("source", "R2", "B", 32000, 33600, 33600, None, 1600),
    )
    reference = GTSessionResult(
        source_id="source",
        confirmation_samples=1600,
        enrollment_samples=1600,
        silence_reset_samples=19200,
        events=reference_events,
        enrollments=(),
        episodes=(),
        timeline=(),
    )
    predicted = (
        ReplacementEvent("source", "P1", "A", 16000, 40000, 40000, None, 1600),
    )
    result = product_event_metrics(
        predicted_events=predicted,
        reference=reference,
        intervals=intervals,
        contamination_episodes=(),
        tolerance_samples=0,
    )
    assert result["contamination_values_seconds_per_true_replacement"] == [1.0, 1.0]
    assert result["exclusive_other_contamination_seconds"] == 2.0


def test_causal_opportunities_count_once_per_unanchored_lifecycle() -> None:
    episode = CausalAnchorEpisode("E1", 0, "slot-0", 0, 16000, 16000, "final")
    session = CausalSessionResult(
        source_id="source",
        enrollment_config=CausalEnrollmentConfig(0.8, 0.2, 1600),
        replacement_confirmation_samples=1600,
        anchor_threshold=0.5,
        other_threshold=0.5,
        enrollments=(),
        episodes=(episode,),
        replacement_events=(),
        timeline=(),
        uncertain_entry_count=0,
        final_reset_count=0,
    )
    intervals = (
        ActivityInterval(0, 3200, ("A",), False),
        ActivityInterval(3200, 6400, (), False),
        ActivityInterval(6400, 9600, ("B",), False),
        ActivityInterval(9600, 12800, (), False),
        ActivityInterval(12800, 16000, ("C",), False),
    )
    assert count_causal_opportunities(
        session=session,
        intervals=intervals,
        scored_start_sample=0,
        scored_end_sample=16000,
        enrollment_samples=1600,
        silence_reset_samples=1600,
    ) == 1


def test_causal_anchor_metrics_separate_unmatched_enrollments() -> None:
    annotations = (
        AnnotatedCausalEpisode(
            CausalAnchorEpisode("E1", 0, "slot-0", 0, 3200, 6400, "cut"),
            "A",
            0,
            0,
            True,
        ),
        AnnotatedCausalEpisode(
            CausalAnchorEpisode("E2", 1, "slot-1", 6400, 8000, 9600, "final"),
            None,
            None,
            None,
            False,
        ),
    )
    metrics = causal_anchor_metrics(annotations, 1)
    assert metrics["enrollment_count"] == 1
    assert metrics["total_enrollment_count"] == 2
    assert metrics["unmatched_enrollment_count"] == 1
    assert metrics["enrollment_failure_count"] == 0


def test_anchor_error_cascade_resets_on_correct_episode_without_event() -> None:
    episodes = (
        AnnotatedCausalEpisode(
            CausalAnchorEpisode("E1", 0, "slot-0", 0, 1000, 2000, "cut"),
            None,
            None,
            None,
            False,
        ),
        AnnotatedCausalEpisode(
            CausalAnchorEpisode("E2", 0, "slot-0", 2000, 3000, 4000, "final"),
            None,
            None,
            0,
            True,
        ),
        AnnotatedCausalEpisode(
            CausalAnchorEpisode("E3", 1, "slot-1", 4000, 5000, 6000, "cut"),
            None,
            None,
            None,
            False,
        ),
    )
    events = (
        ReplacementEvent("source", "E1", "slot-0", 1500, 1800, 1800, None, 100),
        ReplacementEvent("source", "E3", "slot-1", 5500, 5800, 5800, None, 100),
    )
    session = CausalSessionResult(
        source_id="source",
        enrollment_config=CausalEnrollmentConfig(0.8, 0.2, 1600),
        replacement_confirmation_samples=100,
        anchor_threshold=0.5,
        other_threshold=0.5,
        enrollments=(),
        episodes=tuple(value.episode for value in episodes),
        replacement_events=events,
        timeline=(),
        uncertain_entry_count=0,
        final_reset_count=0,
    )
    reference = GTSessionResult("source", 100, 1600, 1600, (), (), (), ())
    result = causal_product_metrics(
        session=session,
        annotated=episodes,
        reference=reference,
        intervals=(ActivityInterval(0, 6000, ("A",), False),),
        tolerance_samples=0,
        expected_opportunity_count=0,
    )
    assert result["anchor_error_cascade_length"]["distribution"] == {1: 2}


def test_overlap_takeover_does_not_credit_early_cut() -> None:
    manifest = [
        {
            "source_id": "source",
            "intervals": [{"start_sample": 0, "end_sample": 10000}],
            "transitions": [
                {"transition_id": "T1", "from_interval_index": 0, "to_interval_index": 0}
            ],
            "topology_episodes": [
                {
                    "primary_topology": "overlap_takeover",
                    "coverage_gate_eligible": True,
                    "transition_ids": ["T1"],
                }
            ],
        }
    ]
    predicted = ReplacementEvent("source", "P1", "A", 4000, 5000, 5000, None, 1600)
    reference_event = ReplacementEvent(
        "source", "R1", "A", 5000, 6000, 6000, None, 1600
    )
    reference = GTSessionResult("source", 1600, 1600, 1600, (reference_event,), (), (), ())
    slices = _topology_slices(
        manifest,
        {"source": (predicted,)},
        {"source": reference},
        tolerance_samples=1600,
    )
    assert slices["overlap_takeover"]["episodes_with_aligned_cut"] == 0
    assert slices["overlap_takeover"]["episodes_with_early_cut"] == 1
    assert slices["overlap_takeover"]["overlap_takeover_success_rate"] == 0.0
