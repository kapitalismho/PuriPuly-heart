from __future__ import annotations

import pytest

from experiments.speaker_representation_scd.r5_scoring import (
    causal_match_events,
    detect_probability_events,
    event_metrics,
    select_operating_point,
)


def test_probability_events_preserve_onset_and_emit_frontiers() -> None:
    probabilities = [0.1, 0.7, 0.8, 0.2]
    frontiers = [1600, 3200, 4800, 6400]
    events = detect_probability_events(probabilities, frontiers, 0.5, 2)
    assert len(events) == 1
    assert events[0]["onset_sample"] == 3200
    assert events[0]["emit_sample"] == 4800


def test_causal_match_rejects_emit_before_ground_truth() -> None:
    events = [{"onset_sample": 14400, "emit_sample": 15200}]
    assert causal_match_events([16000], events, 500) == []


def test_causal_match_keeps_negative_localization_with_nonnegative_availability() -> None:
    events = [{"onset_sample": 15200, "emit_sample": 17600}]
    matched = causal_match_events([16000], events, 500)
    assert matched[0]["localization_error_ms"] == -50
    assert matched[0]["availability_latency_ms"] == 100


def test_event_metrics_include_raw_false_count_and_exposure() -> None:
    matched = [
        {
            "localization_error_ms": 100.0,
            "availability_latency_ms": 200.0,
        }
    ]
    metrics = event_metrics(matched, 3, 2, 2.0)
    assert metrics["false_event_count"] == 2
    assert metrics["false_events_per_hour"] == pytest.approx(1.0)
    assert metrics["source_hours"] == pytest.approx(2.0)
    assert set(metrics["boundary_f1"]) == {
        "at_100ms",
        "at_250ms",
        "at_500ms",
        "at_750ms",
        "at_1000ms",
        "at_1500ms",
    }
    assert metrics["recall_within_1000ms"] == pytest.approx(0.5)


def test_operating_point_prefers_higher_f1_inside_budget() -> None:
    rows = [
        {
            "config_id": "weak",
            "metrics": {
                "f1_at_250ms": 0.1,
                "recall_within_500ms": 0.2,
                "false_events_per_hour": 0.0,
                "availability_latency_ms": {"median_ms": 100.0},
            },
        },
        {
            "config_id": "strong",
            "metrics": {
                "f1_at_250ms": 0.5,
                "recall_within_500ms": 0.6,
                "false_events_per_hour": 0.8,
                "availability_latency_ms": {"median_ms": 200.0},
            },
        },
    ]
    assert select_operating_point(rows, 1.0)["config_id"] == "strong"
