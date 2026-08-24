from __future__ import annotations

import random
from itertools import combinations, permutations

import pytest

from experiments.psem_training_strategy_gate.evaluator import (
    CandidateEvent,
    MetricContractError,
    PredictionScore,
    ReferenceEvent,
    event_average_precision,
    eventize,
    full_threshold_curve,
    shared_score_thresholds,
    sub_resolution_transitions,
)


def test_eventizer_keeps_earlier_plateau_peak_and_suppresses_within_200ms() -> None:
    events = eventize(
        (
            PredictionScore("s", 0, 16000, 0.9),
            PredictionScore("s", 1600, 17600, 0.9),
            PredictionScore("s", 3200, 19200, 0.1),
            PredictionScore("s", 4800, 20800, 0.8),
            PredictionScore("s", 6400, 22400, 0.1),
            PredictionScore("s", 8000, 24000, 0.7),
        )
    )
    assert [(row.boundary_sample, row.score) for row in events] == [(0, 0.9), (4800, 0.8)]


def test_eventizer_excludes_masked_centers_without_turning_them_into_negatives() -> None:
    events = eventize(
        (
            PredictionScore("s", 0, 16000, 0.2),
            PredictionScore("s", 1600, 17600, 0.99, False),
            PredictionScore("s", 3200, 19200, 0.3),
        )
    )
    assert [(row.boundary_sample, row.score) for row in events] == [(3200, 0.3), (0, 0.2)]


def test_matching_is_maximum_cardinality_before_nearest_preference() -> None:
    candidates = (
        CandidateEvent("s", 1600, 17600, 0.9),
        CandidateEvent("s", 3200, 19200, 0.8),
    )
    curve = full_threshold_curve(
        candidates,
        (
            ReferenceEvent("s", 0, "first"),
            ReferenceEvent("s", 1600, "second"),
        ),
        scored_source_samples=57_600_000,
        score_thresholds=shared_score_thresholds((candidates,)),
    )
    assert curve["rows"][0]["metrics"]["100"]["true_positive_count"] == 2
    assert curve["rows"][0]["metrics"]["100"]["false_event_count"] == 0
    assert curve["rows"][0]["matches"]["100"] == [
        {
            "prediction_source_id": "s",
            "prediction_source_sample": 1600,
            "reference_source_id": "s",
            "reference_source_sample": 0,
            "absolute_distance_samples": 1600,
        },
        {
            "prediction_source_id": "s",
            "prediction_source_sample": 3200,
            "reference_source_id": "s",
            "reference_source_sample": 1600,
            "absolute_distance_samples": 1600,
        },
    ]


def test_matching_replaces_a_farther_match_when_cardinality_is_unchanged() -> None:
    candidates = (
        CandidateEvent("s", 0, 16000, 1.0),
        CandidateEvent("s", 1600, 17600, 0.99),
    )
    curve = full_threshold_curve(
        candidates,
        (
            ReferenceEvent("s", 1600, "near"),
            ReferenceEvent("s", 4800, "outside"),
        ),
        scored_source_samples=57_600_000,
        score_thresholds=shared_score_thresholds((candidates,)),
    )
    assert curve["rows"][0]["matches"]["100"] == [
        {
            "prediction_source_id": "s",
            "prediction_source_sample": 1600,
            "reference_source_id": "s",
            "reference_source_sample": 1600,
            "absolute_distance_samples": 0,
        }
    ]


def test_matching_reaches_bruteforce_cardinality_and_nearest_distance() -> None:
    generator = random.Random(76)
    for _ in range(80):
        prediction_samples = sorted(generator.sample(range(0, 12 * 1600, 1600), 5))
        reference_samples = sorted(generator.sample(range(0, 12 * 1600), 4))
        candidates = tuple(
            CandidateEvent("s", sample, sample + 16000, 1.0 - index / 10)
            for index, sample in enumerate(prediction_samples)
        )
        references = tuple(ReferenceEvent("s", sample, "handoff") for sample in reference_samples)
        best_cardinality = 0
        best_distance = 0
        for cardinality in range(1, min(len(candidates), len(references)) + 1):
            distances = []
            for prediction_ids in combinations(range(len(candidates)), cardinality):
                for reference_ids in permutations(range(len(references)), cardinality):
                    values = [
                        abs(prediction_samples[prediction_id] - reference_samples[reference_id])
                        for prediction_id, reference_id in zip(
                            prediction_ids, reference_ids, strict=True
                        )
                    ]
                    if all(value <= 1600 for value in values):
                        distances.append(sum(values))
            if distances:
                best_cardinality = cardinality
                best_distance = min(distances)
        curve = full_threshold_curve(
            candidates,
            references,
            scored_source_samples=57_600_000,
            score_thresholds=shared_score_thresholds((candidates,)),
        )
        matches = curve["rows"][0]["matches"]["100"]
        assert len(matches) == best_cardinality
        assert sum(row["absolute_distance_samples"] for row in matches) == best_distance


def test_full_curve_has_every_unique_threshold_and_no_fe_ceiling() -> None:
    candidates = (
        CandidateEvent("s", 0, 16000, 0.1),
        CandidateEvent("s", 6400, 22400, 0.5),
        CandidateEvent("s", 12800, 28800, 0.9),
    )
    curve = full_threshold_curve(
        candidates,
        (ReferenceEvent("s", 12737, "direct"),),
        scored_source_samples=57_600_000,
        score_thresholds=shared_score_thresholds((candidates,)),
    )
    assert curve["score_thresholds"] == [0.1, 0.5, 0.9]
    assert len(curve["rows"]) == 3
    assert curve["false_events_per_hour_ceiling"] is None
    assert curve["rows"][0]["metrics"]["100"]["false_events_per_hour"] == 2.0
    assert curve["summaries"]["100"]["event_average_precision"] == 1.0


def test_collar_is_sample_exact_and_unsnapped_reference_is_retained() -> None:
    exact_candidates = (
        CandidateEvent("s", 36800, 52800, 0.9),
        CandidateEvent("s", 38400, 54400, 0.1),
    )
    outside_candidates = (
        CandidateEvent("s", 36800, 52800, 0.9),
        CandidateEvent("s", 38400, 54400, 0.1),
    )
    exact = full_threshold_curve(
        exact_candidates,
        (ReferenceEvent("s", 35200, "direct"),),
        scored_source_samples=57_600_000,
        score_thresholds=shared_score_thresholds((exact_candidates,)),
    )
    outside = full_threshold_curve(
        outside_candidates,
        (ReferenceEvent("s", 35199, "direct"),),
        scored_source_samples=57_600_000,
        score_thresholds=shared_score_thresholds((outside_candidates,)),
    )
    assert exact["rows"][1]["metrics"]["100"]["true_positive_count"] == 1
    assert outside["rows"][1]["metrics"]["100"]["true_positive_count"] == 0
    assert exact["rows"][1]["metrics"]["100"]["false_events_per_hour"] == 0.0


def test_short_return_below_duplicate_radius_is_explicit_diagnostic() -> None:
    diagnostics = sub_resolution_transitions(
        (
            ReferenceEvent("s", 35137, "handoff"),
            ReferenceEvent("s", 38337, "short_return"),
            ReferenceEvent("s", 50000, "handoff"),
        )
    )
    assert diagnostics == (
        {
            "artifact_role": "sub_resolution_transition",
            "source_id": "s",
            "left_sample": 35137,
            "right_sample": 38337,
            "distance_samples": 3200,
            "left_topology": "handoff",
            "right_topology": "short_return",
        },
    )


def test_metric_contract_rejects_duplicate_centers_and_nonfixed_collars() -> None:
    candidates = (
        CandidateEvent("s", 0, 16000, 0.9),
        CandidateEvent("s", 0, 16000, 0.8),
    )
    with pytest.raises(MetricContractError, match="unique"):
        full_threshold_curve(
            candidates,
            (),
            scored_source_samples=16000,
            score_thresholds=(0.8, 0.9),
        )
    with pytest.raises(MetricContractError, match="collars"):
        full_threshold_curve(
            candidates[:1],
            (),
            scored_source_samples=16000,
            score_thresholds=(0.8, 0.9),
            collars_ms=(250,),
        )
    with pytest.raises(MetricContractError, match="temporal evidence"):
        full_threshold_curve(
            (CandidateEvent("s", 0, 15999, 0.9),),
            (),
            scored_source_samples=16000,
            score_thresholds=(0.8, 0.9),
        )


def test_shared_threshold_vector_is_required_for_every_output() -> None:
    first = (CandidateEvent("a", 0, 16000, 0.1), CandidateEvent("a", 1600, 17600, 0.9))
    second = (CandidateEvent("b", 0, 16000, 0.2), CandidateEvent("b", 1600, 17600, 0.8))
    thresholds = shared_score_thresholds((first, second))
    assert thresholds == (0.1, 0.2, 0.8, 0.9)
    first_curve = full_threshold_curve(
        first,
        (),
        scored_source_samples=16000,
        score_thresholds=thresholds,
    )
    second_curve = full_threshold_curve(
        second,
        (),
        scored_source_samples=16000,
        score_thresholds=thresholds,
    )
    assert first_curve["score_thresholds"] == second_curve["score_thresholds"] == list(thresholds)


@pytest.mark.parametrize(
    "candidate",
    (
        CandidateEvent("s", 1, 16001, 0.9),
        CandidateEvent("s", True, 16001, 0.9),
        CandidateEvent("s", 0, 16000, True),
        CandidateEvent("s", 0, 16000, "bad"),
    ),
)
def test_metric_contract_rejects_invalid_candidate_domains(candidate) -> None:
    with pytest.raises(MetricContractError, match="temporal evidence"):
        full_threshold_curve(
            (candidate,),
            (),
            scored_source_samples=16000,
            score_thresholds=(0.1, 0.9),
        )


def test_eventizer_requires_contiguous_grid_and_equal_scores_are_atomic() -> None:
    with pytest.raises(MetricContractError, match="contiguous"):
        eventize(
            (
                PredictionScore("s", 0, 16000, 0.9),
                PredictionScore("s", 3200, 19200, 0.8),
            )
        )
    candidates = (
        CandidateEvent("s", 0, 16000, 0.9),
        CandidateEvent("s", 3200, 19200, 0.9),
    )
    curve = full_threshold_curve(
        candidates,
        (),
        scored_source_samples=16000,
        score_thresholds=(0.1, 0.9),
    )
    assert curve["rows"][1]["prediction_count"] == 2


def test_checkpoint_event_average_precision_matches_full_curve() -> None:
    scores = [
        PredictionScore("session", sample, sample + 16000, score)
        for sample, score in (
            (32000, 0.1),
            (33600, 0.9),
            (35200, 0.1),
            (36800, 0.1),
            (38400, 0.8),
            (40000, 0.1),
        )
    ]
    candidates = eventize(scores)
    references = [
        ReferenceEvent("session", 33600, "direct"),
        ReferenceEvent("session", 38600, "gap"),
    ]
    thresholds = shared_score_thresholds((candidates,))
    curve = full_threshold_curve(
        candidates,
        references,
        scored_source_samples=16000 * 60,
        score_thresholds=thresholds,
    )
    summary = event_average_precision(candidates, references, collar_ms=250)
    assert (
        summary["event_average_precision"] == curve["summaries"]["250"]["event_average_precision"]
    )
    assert summary["threshold_count"] == len(thresholds)


def test_checkpoint_event_average_precision_matches_full_curve_randomized() -> None:
    generator = random.Random(7301)
    for case in range(40):
        candidate_samples = sorted(generator.sample(range(0, 40 * 1600, 1600), 8))
        score_order = generator.sample(range(1, 9), 8)
        candidates = tuple(
            CandidateEvent(
                "session",
                sample,
                sample + 16000,
                (score_order[index] + case / 1000) / 10,
            )
            for index, sample in enumerate(candidate_samples)
        )
        references = tuple(
            ReferenceEvent("session", sample, "handoff")
            for sample in sorted(generator.sample(range(0, 40 * 1600), 6))
        )
        thresholds = shared_score_thresholds((candidates,))
        curve = full_threshold_curve(
            candidates,
            references,
            scored_source_samples=16000 * 60,
            score_thresholds=thresholds,
        )
        for collar in (100, 250, 500):
            summary = event_average_precision(candidates, references, collar_ms=collar)
            assert summary["event_average_precision"] == pytest.approx(
                curve["summaries"][str(collar)]["event_average_precision"]
            )
            assert summary["maximum_f1"] == pytest.approx(
                curve["summaries"][str(collar)]["max_f1"]["f1"]
            )
