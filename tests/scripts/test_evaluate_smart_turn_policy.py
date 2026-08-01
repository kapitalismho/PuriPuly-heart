from __future__ import annotations

import pytest

from scripts.experiments.evaluate_smart_turn_policy import summarize_policy


def test_summarize_policy_uses_first_crossing_and_hard_boundary() -> None:
    result = summarize_policy(
        [
            {"endpoint_bool": True, "scores": {"224": 0.7, "416": 0.1, "608": 0.1}},
            {"endpoint_bool": True, "scores": {"224": 0.1, "416": 0.8, "608": 0.1}},
            {"endpoint_bool": False, "scores": {"224": 0.1, "416": 0.2, "608": 0.3}},
        ]
    )

    assert result["first_accept_counts"] == {"224": 1, "416": 1, "608": 0}
    assert result["hard_boundary_count"] == 1
    assert result["decision_ms"]["mean"] == 480.0
    assert result["decision_ms"]["p50"] == 416.0
    assert result["decision_ms"]["p95"] == pytest.approx(761.6)
    assert result["decision_ms"]["mean_delta_vs_512ms_vad"] == -32.0
    assert result["early_false_complete_count"] == 0
    assert result["end_to_end_decision_latency_ms"]["available"] is False
    assert result["end_to_end_decision_latency_ms"]["missing_count"] == 2


def test_summarize_policy_includes_inference_latency_in_decision_latency() -> None:
    result = summarize_policy(
        [
            {
                "endpoint_bool": True,
                "scores": {"224": 0.8, "416": 0.1, "608": 0.1},
                "inference_ms": {"224": 50.0, "416": 51.0, "608": 52.0},
            },
            {
                "endpoint_bool": False,
                "scores": {"224": 0.1, "416": 0.1, "608": 0.1},
                "inference_ms": {"224": 50.0, "416": 51.0, "608": 52.0},
            },
        ]
    )

    latency = result["end_to_end_decision_latency_ms"]
    assert latency["available"] is True
    assert latency["observed_count"] == 2
    assert latency["mean"] == 537.0
    assert result["inference_latency_ms"]["mean"] == 51.0


def test_summarize_policy_rejects_missing_probe_scores() -> None:
    with pytest.raises(ValueError, match="missing probe scores"):
        summarize_policy([{"endpoint_bool": True, "scores": {"224": 0.8, "416": 0.2}}])
