from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import experiments.speaker_representation_scd.r4_gate as gate_module
from experiments.speaker_representation_scd.provenance import with_self_sha256
from experiments.speaker_representation_scd.r1_gate import EXPERIMENT_ROOT
from experiments.speaker_representation_scd.r4_continuous import (
    _event_metrics,
    _prototype_event_detector,
    _rank_encoders,
    _select_operating_point,
    adjacent_scores,
    detect_events,
    match_events,
    prototype_scores,
)
from experiments.speaker_representation_scd.r4_gate import (
    EXPECTED_ACTIONS,
    GATE_PATH,
    validate_r4_gate,
)


def test_r4_gate_is_valid_without_external_execution() -> None:
    result = validate_r4_gate(scan_processes=False)
    assert result.valid, result.errors
    assert result.allowed_actions == EXPECTED_ACTIONS
    assert result.allowed_actions["r4_continuous"] is True
    assert result.allowed_actions["r4_sensitivity"] is True
    assert result.allowed_actions["training"] is False


def test_rehashed_semantic_gate_mutation_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    original_loader = gate_module.load_json
    gate = original_loader(EXPERIMENT_ROOT / GATE_PATH)
    mutated = deepcopy(gate)
    mutated["authorization"]["r4_continuous"] = False
    mutated = with_self_sha256(mutated)

    def load(path: Path) -> dict:
        if path.resolve() == (EXPERIMENT_ROOT / GATE_PATH).resolve():
            return mutated
        return original_loader(path)

    monkeypatch.setattr(gate_module, "load_json", load)
    result = validate_r4_gate(scan_processes=False)
    assert not result.valid
    assert "r4_gate.authorization: differs" in result.errors


def test_gate_file_has_valid_json_identity() -> None:
    document = json.loads((EXPERIMENT_ROOT / GATE_PATH).read_text(encoding="utf-8"))
    assert document["artifact_role"] == "r4_legacy_common_gt_gate"
    assert document["r4"]["primary_context_ms"] == 300
    assert document["r4"]["sensitivity_hop_ms"] == 50
    assert document["authorization"]["r4_continuous"] is True


def test_adjacent_scores_bounds() -> None:
    vectors = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32
    )
    scores = adjacent_scores(vectors)
    assert np.isnan(scores[0])
    assert scores[1] == pytest.approx(0.0)
    assert scores[2] == pytest.approx(1.0)
    assert scores[3] == pytest.approx(0.0)


def test_prototype_scores_tracks_stable_mean() -> None:
    vectors = np.array(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    scores = prototype_scores(vectors)
    assert scores[0] == pytest.approx(0.0)
    assert scores[1] == pytest.approx(0.0, abs=1e-6)
    assert scores[3] == pytest.approx(1.0)
    assert scores[4] == pytest.approx(1.0)


def test_prototype_event_detector_resets_after_emit() -> None:
    vectors = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    events = _prototype_event_detector(vectors, "one_hop", 0.4)
    assert [row["emit_hop"] for row in events] == [3, 8]
    events_two = _prototype_event_detector(vectors, "two_hop", 0.4)
    assert [row["emit_hop"] for row in events_two] == [4, 9]
    stable = np.tile(np.array([1.0, 0.0], dtype=np.float32), (30, 1))
    assert _prototype_event_detector(stable, "one_hop", 0.4) == []


def test_detect_events_families() -> None:
    distances = np.array(
        [0.1, 0.1, 0.6, 0.6, 0.6, 0.1, 0.1, 0.1, 0.8, 0.8, 0.1], dtype=np.float64
    )
    one = detect_events(distances, "one_hop", 0.3)
    assert [row["emit_hop"] for row in one] == [2, 8]
    two = detect_events(distances, "two_hop", 0.3)
    assert [row["emit_hop"] for row in two] == [3, 9]
    assert [row["onset_hop"] for row in two] == [2, 8]
    three = detect_events(distances, "three_hop", 0.3)
    assert [row["emit_hop"] for row in three] == [4]
    hysteresis = detect_events(distances, "hysteresis", 0.5)
    assert [row["emit_hop"] for row in hysteresis] == [3, 9]


def test_detect_events_minimum_separation() -> None:
    distances = np.full(12, 0.8, dtype=np.float64)
    events = detect_events(distances, "one_hop", 0.3)
    assert len(events) == 1
    assert events[0]["emit_hop"] == 0


def test_match_events_one_to_one_within_tolerance() -> None:
    events = [
        {
            "onset_hop": 10,
            "onset_sample": 4800 + 1600 * 10,
            "emit_hop": 10,
            "emit_sample": 4800 + 1600 * 10,
            "availability_hop": 10,
        },
        {
            "onset_hop": 20,
            "onset_sample": 4800 + 1600 * 20,
            "emit_hop": 20,
            "emit_sample": 4800 + 1600 * 20,
            "availability_hop": 20,
        },
    ]
    ground_truth = [4800 + 1600 * 10 - 1600, 4800 + 1600 * 20 - 1600]
    matched = match_events(ground_truth, events, tolerance_ms=500)
    assert len(matched) == 2
    assert matched[0]["localization_error_ms"] == 100
    assert matched[0]["availability_latency_ms"] == 100


def test_match_events_rejects_out_of_tolerance() -> None:
    events = [
        {
            "onset_hop": 10,
            "onset_sample": 4800 + 1600 * 10,
            "emit_hop": 10,
            "emit_sample": 4800 + 1600 * 10,
            "availability_hop": 10,
        },
    ]
    ground_truth = [4800 + 1600 * 40]
    matched = match_events(ground_truth, events, tolerance_ms=500)
    assert matched == []


def test_event_metrics_math() -> None:
    matched = [
        {
            "availability_latency_ms": 100.0,
            "localization_error_ms": -50.0,
        },
        {
            "availability_latency_ms": 300.0,
            "localization_error_ms": 50.0,
        },
    ]
    metrics = _event_metrics(matched, total_events=3, ground_truth_count=4, source_hours=2.0)
    assert metrics["matched_count"] == 2
    assert metrics["false_events_per_hour"] == pytest.approx(0.5)
    assert metrics["missed_change_rate"] == pytest.approx(0.5)
    assert metrics["boundary_f1"]["at_500ms"]["recall"] == pytest.approx(0.5)
    assert metrics["availability_latency_ms"]["median_ms"] == pytest.approx(200.0)
    assert metrics["duplicate_event_rate"] == 0.0


def test_select_operating_point_respects_false_event_budget() -> None:
    configs = [
        {
            "config_id": "prototype|one_hop|0.10",
            "metrics": {
                "recall_within_500ms": 0.95,
                "false_events_per_hour": 2.0,
                "f1_at_250ms": 0.60,
                "availability_latency_ms": {"median_ms": 150.0},
            },
        },
        {
            "config_id": "prototype|one_hop|0.20",
            "metrics": {
                "recall_within_500ms": 0.90,
                "false_events_per_hour": 0.8,
                "f1_at_250ms": 0.65,
                "availability_latency_ms": {"median_ms": 180.0},
            },
        },
        {
            "config_id": "prototype|one_hop|0.30",
            "metrics": {
                "recall_within_500ms": 0.70,
                "false_events_per_hour": 0.2,
                "f1_at_250ms": 0.55,
                "availability_latency_ms": {"median_ms": 220.0},
            },
        },
    ]
    primary = _select_operating_point(configs)
    assert primary["config_id"] == "prototype|one_hop|0.20"


def test_select_operating_point_no_budget_falls_back_to_lowest_rate() -> None:
    configs = [
        {
            "config_id": "adjacent|two_hop|0.10",
            "metrics": {
                "recall_within_500ms": 0.99,
                "false_events_per_hour": 3.0,
                "f1_at_250ms": 0.50,
                "availability_latency_ms": {"median_ms": 100.0},
            },
        },
        {
            "config_id": "adjacent|two_hop|0.40",
            "metrics": {
                "recall_within_500ms": 0.10,
                "false_events_per_hour": 0.3,
                "f1_at_250ms": 0.10,
                "availability_latency_ms": {"median_ms": 400.0},
            },
        },
    ]
    primary = _select_operating_point(configs)
    assert primary["config_id"] == "adjacent|two_hop|0.40"


def test_rank_encoders_deterministic() -> None:
    operating_points = {
        "a-model": {
            "config_id": "x",
            "metrics": {
                "recall_within_500ms": 0.8,
                "false_events_per_hour": 0.5,
                "f1_at_250ms": 0.6,
                "availability_latency_ms": {"median_ms": 200.0},
            },
        },
        "b-model": {
            "config_id": "y",
            "metrics": {
                "recall_within_500ms": 0.8,
                "false_events_per_hour": 0.5,
                "f1_at_250ms": 0.6,
                "availability_latency_ms": {"median_ms": 150.0},
            },
        },
    }
    ranked = _rank_encoders(operating_points, {"a-model": 0.80, "b-model": 0.75})
    assert [row["model_id"] for row in ranked] == ["b-model", "a-model"]


def test_gate_file_matches_expected_contract() -> None:
    document = json.loads((EXPERIMENT_ROOT / GATE_PATH).read_text(encoding="utf-8"))
    assert document["r4"]["false_event_budget_per_hour"] == 1.0
    assert document["r4"]["match_tolerance_ms"] == 500
    assert document["r4"]["sensitivity"]["top_encoder_count"] == 2
