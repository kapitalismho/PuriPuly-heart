from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import experiments.speaker_representation_scd.r3_gate as gate_module
from experiments.speaker_representation_scd.provenance import with_self_sha256
from experiments.speaker_representation_scd.r1_gate import EXPERIMENT_ROOT
from experiments.speaker_representation_scd.r3_gate import (
    EXPECTED_ACTIONS,
    GATE_PATH,
    validate_r3_gate,
)
from experiments.speaker_representation_scd.r3_probe import (
    analyze_anchor_scores,
    bootstrap_block_ci,
    bootstrap_paired_ci,
    build_negative_prototype,
    build_promotion_ledger,
    cosine_distance,
    eer,
    overlap_coefficient,
    positive_prototype,
    quantiles,
    rank_layers,
    roc_auc,
    score_anchor,
    trajectory_metrics,
    wilcoxon_signed_rank_statistic,
)


def test_r3_gate_is_valid_without_external_execution() -> None:
    result = validate_r3_gate(scan_processes=False)
    assert result.valid, result.errors
    assert result.allowed_actions == EXPECTED_ACTIONS
    assert result.allowed_actions["r3_probe"] is True
    assert result.allowed_actions["training"] is False
    assert result.allowed_actions["confirmatory_access"] is False


def test_rehashed_semantic_gate_mutation_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    original_loader = gate_module.load_json
    gate = original_loader(EXPERIMENT_ROOT / GATE_PATH)
    mutated = deepcopy(gate)
    mutated["authorization"]["r3_probe"] = False
    mutated = with_self_sha256(mutated)

    def load(path: Path) -> dict:
        if path.resolve() == (EXPERIMENT_ROOT / GATE_PATH).resolve():
            return mutated
        return original_loader(path)

    monkeypatch.setattr(gate_module, "load_json", load)
    result = validate_r3_gate(scan_processes=False)
    assert not result.valid
    assert "r3_gate.authorization: differs" in result.errors


def test_gate_file_has_valid_json_identity() -> None:
    document = json.loads((EXPERIMENT_ROOT / GATE_PATH).read_text(encoding="utf-8"))
    assert document["artifact_role"] == "r3_legacy_common_gt_gate"
    assert document["experiment_id"] == "speaker_representation_scd_v1"
    assert document["authorization"]["r3_probe"] is True
    assert document["supervision"]["probe_worker_environment"] == "environment_venv"


def test_roc_auc_extremes_and_chance() -> None:
    assert roc_auc([0.9, 0.8, 0.7], [0.2, 0.1, 0.05]) == pytest.approx(1.0)
    assert roc_auc([0.1, 0.2], [0.9, 0.8]) == pytest.approx(0.0)
    assert roc_auc([0.5, 0.6], [0.5, 0.6]) == pytest.approx(0.5)
    assert roc_auc([0.5, 0.6], [0.55, 0.45]) == pytest.approx(0.75)
    assert np.isnan(roc_auc([], [1.0]))


def test_eer_and_overlap_coefficient() -> None:
    assert eer([1.0, 0.9], [0.1, 0.2]) == pytest.approx(0.0)
    assert eer([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.5)
    assert overlap_coefficient([1.0, 1.0], [0.0, 0.0]) == pytest.approx(0.0)
    identical = overlap_coefficient([0.2, 0.4, 0.6], [0.2, 0.4, 0.6])
    assert identical == pytest.approx(1.0)


def test_wilcoxon_and_quantiles() -> None:
    assert wilcoxon_signed_rank_statistic([1.0, 2.0, 3.0]) == pytest.approx(6.0)
    assert wilcoxon_signed_rank_statistic([-1.0, -2.0]) == pytest.approx(0.0)
    stats = quantiles([1.0, 2.0, 3.0, 4.0])
    assert stats["mean"] == pytest.approx(2.5)
    assert stats["median"] == pytest.approx(2.5)


def test_positive_prototype_uses_old_speaker_offsets_only() -> None:
    trajectory = {
        -1000: ("entirely_old", np.array([1.0, 0.0], dtype=np.float32)),
        -300: ("entirely_old", np.array([1.0, 1.0], dtype=np.float32)),
        -100: ("entirely_old", np.array([1.0, 1.0], dtype=np.float32)),
        0: ("boundary_straddling", np.array([0.0, 1.0], dtype=np.float32)),
    }
    prototype = positive_prototype(trajectory)
    assert prototype is not None
    assert prototype[0] > prototype[1]
    assert np.linalg.norm(prototype) == pytest.approx(1.0, abs=1e-5)
    assert positive_prototype({0: ("boundary_straddling", np.array([1.0, 0.0]))}) is None


def test_negative_prototype_leave_one_out() -> None:
    anchor_rows = [
        {"candidate_id": "negative:a", "class": "negative", "block_id": "b1", "kind": "stable"},
        {"candidate_id": "negative:b", "class": "negative", "block_id": "b1", "kind": "stable"},
        {"candidate_id": "negative:c", "class": "negative", "block_id": "b1", "kind": "pause"},
    ]
    z = {
        "negative:a": np.array([1.0, 0.0], dtype=np.float32),
        "negative:b": np.array([0.9, 0.1], dtype=np.float32),
        "negative:c": np.array([0.0, 1.0], dtype=np.float32),
    }
    prototype_a = build_negative_prototype(anchor_rows, "negative:a", z)
    assert prototype_a is not None
    assert prototype_a[0] > prototype_a[1]
    assert np.linalg.norm(prototype_a) == pytest.approx(1.0, abs=1e-5)


def test_score_anchor_positives_and_negatives() -> None:
    anchor_rows = [
        {"candidate_id": "positive:x", "class": "positive", "block_id": "b1", "kind": "hard_boundary"},
        {"candidate_id": "negative:y", "class": "negative", "block_id": "b1", "kind": "stable"},
        {"candidate_id": "negative:z", "class": "negative", "block_id": "b1", "kind": "stable"},
    ]
    z = {
        "positive:x": np.array([0.0, 1.0], dtype=np.float32),
        "negative:y": np.array([1.0, 0.0], dtype=np.float32),
        "negative:z": np.array([1.0, 0.0], dtype=np.float32),
    }
    trajectory = {
        -300: ("entirely_old", np.array([1.0, 0.0], dtype=np.float32)),
        -100: ("entirely_old", np.array([1.0, 0.0], dtype=np.float32)),
        0: ("boundary_straddling", np.array([0.0, 1.0], dtype=np.float32)),
    }
    distance, adjacent = score_anchor(
        anchor_rows[0], z, trajectory, None
    )
    assert distance == pytest.approx(1.0)
    assert adjacent == pytest.approx(1.0)
    negative_prototype = build_negative_prototype(anchor_rows, "negative:y", z)
    distance, _ = score_anchor(anchor_rows[1], z, None, negative_prototype)
    assert distance == pytest.approx(0.0, abs=1e-6)


def test_analyze_anchor_scores_macro_and_pairs() -> None:
    anchor_rows = [
        {"candidate_id": "positive:a", "class": "positive", "block_id": "b1", "kind": "k", "language": "english", "pair_id": "p1"},
        {"candidate_id": "negative:a", "class": "negative", "block_id": "b1", "kind": "k", "language": "english", "pair_id": "p1"},
        {"candidate_id": "positive:b", "class": "positive", "block_id": "b2", "kind": "k", "language": "english", "pair_id": "p2"},
        {"candidate_id": "negative:b", "class": "negative", "block_id": "b2", "kind": "k", "language": "english", "pair_id": "p2"},
    ]
    scores = {"positive:a": 0.9, "negative:a": 0.1, "positive:b": 0.8, "negative:b": 0.2}
    metrics = analyze_anchor_scores(anchor_rows, scores)
    assert metrics["macro_roc_auc"] == pytest.approx(1.0)
    assert metrics["worst_block_roc_auc"] == pytest.approx(1.0)
    assert metrics["pair_metrics"]["paired_auc"] == pytest.approx(1.0)
    assert metrics["pair_metrics"]["matched_pair_count"] == 2
    assert metrics["language_metrics"]["english"]["roc_auc"] == pytest.approx(1.0)


def test_trajectory_metrics_onset_and_recovery() -> None:
    trajectory = {
        -1000: ("entirely_old", np.array([1.0, 0.0], dtype=np.float32)),
        -100: ("entirely_old", np.array([1.0, 0.0], dtype=np.float32)),
        0: ("boundary_straddling", np.array([1.0, 1.0], dtype=np.float32)),
        100: ("entirely_new", np.array([0.0, 1.0], dtype=np.float32)),
        300: ("entirely_new", np.array([1.0, 1.0], dtype=np.float32)),
        1000: ("entirely_new", np.array([1.0, 0.0], dtype=np.float32)),
    }
    metrics = trajectory_metrics({"positive:x": trajectory})
    assert metrics["candidate_count"] == 1
    summary = metrics["candidate_summaries"][0]
    assert summary["onset_offset_ms"] == 100
    assert summary["peak_offset_ms"] == 100
    assert summary["recovery_offset_ms"] == 1000


def test_rank_layers_deterministic_order() -> None:
    metrics = {
        "L6": {"eligible": True, "macro_roc_auc": 0.80, "eer": 0.25, "worst_block_roc_auc": 0.60, "evaluable_positive_count": 1, "evaluable_negative_count": 1},
        "L1": {"eligible": True, "macro_roc_auc": 0.80, "eer": 0.20, "worst_block_roc_auc": 0.65, "evaluable_positive_count": 1, "evaluable_negative_count": 1},
        "L3": {"eligible": False, "macro_roc_auc": 0.99, "eer": 0.01, "worst_block_roc_auc": 0.99, "evaluable_positive_count": 1, "evaluable_negative_count": 1},
    }
    ranked = rank_layers(metrics, ("L1", "L3", "L6", "L9", "L12"), 300)
    assert [row["layer_id"] for row in ranked] == ["L1", "L6"]
    assert ranked[0]["macro_roc_auc"] == pytest.approx(0.80)


def _synthetic_probe_result(
    model_id: str,
    context_macro: dict[int, dict[str, float]],
) -> dict:
    layer_metrics = []
    for context_ms, values in context_macro.items():
        for layer_id, macro in values.items():
            layer_metrics.append(
                {
                    "layer_id": layer_id,
                    "context_ms": context_ms,
                    "eligible": True,
                    "macro_roc_auc": macro,
                    "eer": 0.1,
                    "worst_block_roc_auc": 0.5,
                    "evaluable_positive_count": 2,
                    "evaluable_negative_count": 2,
                }
            )
    return {"model_id": model_id, "layer_metrics": layer_metrics}


def test_build_promotion_ledger_common_context_300() -> None:
    protocol_screen = {
        "r3": {
            "promotion": {
                "fallback_requires_all_evaluable_300ms_auc_below": 0.60,
                "fallback_requires_mean_auc_improvement_at_least": 0.02,
                "no_eligible_layer_status": "not_evaluable",
            }
        }
    }
    results = [
        _synthetic_probe_result("mhubert-147", {300: {"L6": 0.70}, 500: {"L6": 0.75}}),
        _synthetic_probe_result("wavlm-base-plus", {300: {"L9": 0.72}, 500: {"L9": 0.74}}),
    ]
    ledger = build_promotion_ledger(results, protocol_screen)
    assert ledger["common_context_ms"] == 300
    assert ledger["fallback_used"] is False
    assert ledger["promoted_by_encoder"]["mhubert-147"]["layer_id"] == "L6"


def test_build_promotion_ledger_fallback_500() -> None:
    protocol_screen = {
        "r3": {
            "promotion": {
                "fallback_requires_all_evaluable_300ms_auc_below": 0.60,
                "fallback_requires_mean_auc_improvement_at_least": 0.02,
                "no_eligible_layer_status": "not_evaluable",
            }
        }
    }
    results = [
        _synthetic_probe_result("mhubert-147", {300: {"L6": 0.55}, 500: {"L6": 0.60}}),
        _synthetic_probe_result("wavlm-base-plus", {300: {"L9": 0.56}, 500: {"L9": 0.60}}),
    ]
    ledger = build_promotion_ledger(results, protocol_screen)
    assert ledger["common_context_ms"] == 500
    assert ledger["fallback_used"] is True


def test_bootstrap_is_deterministic() -> None:
    block_positive = {"b1": [0.9, 0.8], "b2": [0.7, 0.6], "b3": [0.5, 0.4]}
    block_negative = {"b1": [0.2, 0.1], "b2": [0.3, 0.2], "b3": [0.6, 0.5]}
    first = bootstrap_block_ci(block_positive, block_negative, replicates=40, seed=7)
    second = bootstrap_block_ci(block_positive, block_negative, replicates=40, seed=7)
    assert first == second
    assert first["mean"] == pytest.approx(second["mean"])
    pair_deltas = {"p1": 0.5, "p2": 0.3, "p3": -0.1}
    first_paired = bootstrap_paired_ci(pair_deltas, replicates=40, seed=3)
    second_paired = bootstrap_paired_ci(pair_deltas, replicates=40, seed=3)
    assert first_paired == second_paired


def test_cosine_distance_bounds() -> None:
    assert cosine_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)
    assert cosine_distance(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(0.0)
    assert np.isnan(cosine_distance(np.array([np.nan, 0.0]), np.array([1.0, 0.0])))
