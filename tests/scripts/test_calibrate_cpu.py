from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tools.eot_experiment.calibrate_cpu as calibrate_cpu
from tools.eot_experiment.calibrate_cpu import (
    CpuScorer,
    _final_operating_points,
    _group_splits,
    _policy_trace,
    _prepare_smart_turn_audio,
    _threshold_grid,
    _validate_prediction_rows,
    bootstrap_confidence_intervals,
    simulate_policy,
)


def make_row(
    label: str,
    duration_ms: float,
    *,
    score_224: float | None = None,
    score_512: float | None = None,
    latency_224_ms: float | None = 0.0,
    latency_512_ms: float | None = 0.0,
    turn_id: str = "turn-1",
) -> dict:
    return {
        "language": "en",
        "turn_id": turn_id,
        "label": label,
        "span_duration_ms": duration_ms,
        "score_224": score_224,
        "score_512": score_512,
        "inference_latency_224_ms": latency_224_ms,
        "inference_latency_512_ms": latency_512_ms,
    }


def test_hold_shorter_than_first_probe_cannot_be_cut() -> None:
    metrics = simulate_policy([make_row("hold", 150.0)], "P2", 0.5, 0.5)
    assert metrics["false_cutoffs"] == 0
    assert metrics["hard_timeout_rate"] == 1.0


def test_hold_between_probes_can_only_be_cut_by_first_probe() -> None:
    row = make_row("hold", 300.0, score_224=0.9)
    metrics = simulate_policy([row], "P2", 0.5, 0.5)
    assert metrics["false_cutoffs"] == 1
    assert metrics["acceptance_224_rate"] == 1.0
    assert metrics["acceptance_512_rate"] == 0.0


def test_hold_between_second_probe_and_timeout_can_be_cut_by_either_probe() -> None:
    first = make_row("hold", 700.0, score_224=0.9, score_512=0.9)
    second = make_row("hold", 700.0, score_224=0.1, score_512=0.9, turn_id="turn-2")
    first_metrics = simulate_policy([first], "P2", 0.5, 0.5)
    second_metrics = simulate_policy([second], "P2", 0.5, 0.5)
    assert first_metrics["false_cutoffs"] == 1
    assert second_metrics["false_cutoffs"] == 1
    assert second_metrics["acceptance_512_rate"] == 1.0


def test_hold_longer_than_timeout_is_cut_when_both_probes_reject() -> None:
    row = make_row("hold", 900.0, score_224=0.1, score_512=0.1)
    metrics = simulate_policy([row], "P2", 0.5, 0.5)
    assert metrics["false_cutoffs"] == 1
    assert metrics["hard_timeout_rate"] == 1.0


def test_runtime_aware_first_accepted_result_wins() -> None:
    row = make_row("eot", 700.0, score_224=0.9, score_512=0.9, latency_224_ms=40.0)
    trace = _policy_trace([row], "P2", 0.5, 0.5)
    assert trace["decision_ms"][0] == 264.0
    assert trace["probe"][0] == "224ms"
    assert not trace["second_scheduled"][0]


def test_stale_first_result_does_not_endpoint() -> None:
    row = make_row("hold", 250.0, score_224=0.9, latency_224_ms=40.0)
    metrics = simulate_policy([row], "P1", 0.5)
    assert metrics["false_cutoffs"] == 0
    assert metrics["stale_result_count"] == 1
    assert metrics["stale_result_rate"] == 1.0


def test_hard_timeout_beats_late_model_result() -> None:
    row = make_row(
        "eot",
        900.0,
        score_224=0.1,
        score_512=0.9,
        latency_512_ms=300.0,
    )
    metrics = simulate_policy([row], "P2", 0.5, 0.5)
    assert metrics["mean_endpoint_latency_ms"] == 800.0
    assert metrics["acceptance_512_rate"] == 0.0
    assert metrics["stale_result_count"] == 1


def test_single_worker_overlap_is_recorded_without_parallel_execution() -> None:
    row = make_row(
        "eot",
        750.0,
        score_224=0.1,
        score_512=0.9,
        latency_224_ms=400.0,
        latency_512_ms=40.0,
    )
    trace = _policy_trace([row], "P2", 0.5, 0.5)
    assert trace["second_scheduled"][0]
    assert trace["overlap"][0]
    assert trace["decision_ms"][0] == 664.0


def test_single_probe_and_two_probe_differ_only_through_second_probe() -> None:
    row = make_row("eot", 700.0, score_224=0.1, score_512=0.9)
    p1 = simulate_policy([row], "P1", 0.5)
    p2 = simulate_policy([row], "P2", 0.5, 0.5)
    assert p1["mean_endpoint_latency_ms"] == 800.0
    assert p2["mean_endpoint_latency_ms"] == 512.0


def test_fixed_vad_baselines_are_calculated() -> None:
    rows = [make_row("hold", 600.0), make_row("eot", 600.0, turn_id="turn-2")]
    b0 = simulate_policy(rows, "B0")
    b1 = simulate_policy(rows, "B1")
    assert b0["false_cutoffs"] == 1
    assert b0["mean_endpoint_latency_ms"] == 512.0
    assert b1["false_cutoffs"] == 0
    assert b1["mean_endpoint_latency_ms"] == 800.0


def test_group_split_never_crosses_turn_boundaries() -> None:
    rows = [make_row("hold", 300.0, turn_id=f"turn-{index}") for index in range(10)]
    for _fold, train, test in _group_splits(rows, seed=17):
        train_groups = {row["turn_id"] for row in train}
        test_groups = {row["turn_id"] for row in test}
        assert train_groups.isdisjoint(test_groups)


def test_threshold_grid_is_fine_near_one() -> None:
    values = _threshold_grid()
    assert 0.99 in values
    assert 0.9905 in values
    assert 1.0 in values


def test_causal_audio_preparation_left_pads_and_truncates_from_the_beginning() -> None:
    short = _prepare_smart_turn_audio(np.asarray([1.0, 2.0], dtype=np.float32))
    assert short.shape == (128000,)
    assert np.array_equal(short[-2:], np.asarray([1.0, 2.0], dtype=np.float32))

    long = np.arange(128001, dtype=np.float32)
    prepared = _prepare_smart_turn_audio(long)
    assert prepared.shape == (128000,)
    assert prepared[0] == 1.0
    assert prepared[-1] == 128000.0


def test_provider_audit_preserves_unknown_previous_provider(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"test-model")
    audit = calibrate_cpu.audit_providers(output_dir=tmp_path, model_path=model)
    assert audit["previous_policy_predictions"]["execution_provider"] == "unknown"
    assert audit["previous_local_benchmark"]["execution_provider"] == "unknown"
    assert audit["new_cpu_prediction_run"]["execution_provider"] == "CPUExecutionProvider"


@pytest.mark.skipif(
    not Path(".data/smart-turn/smart-turn-v3.2-cpu.onnx").is_file(),
    reason="CPU-int8 model is not present",
)
def test_cpu_scorer_pins_two_thread_cpu_session() -> None:
    scorer = CpuScorer(Path(".data/smart-turn/smart-turn-v3.2-cpu.onnx"), intra_op_threads=2)
    metadata = scorer.metadata()
    assert metadata["execution_provider"] == "CPUExecutionProvider"
    assert metadata["session_providers"][0] == "CPUExecutionProvider"
    assert metadata["intra_op_threads"] == 2
    assert metadata["inter_op_threads"] == 1
    assert metadata["execution_mode"] == "ORT_SEQUENTIAL"


def test_prediction_validation_requires_surviving_probe_scores() -> None:
    row = make_row("eot", 600.0, score_224=0.9, score_512=None)
    with pytest.raises(ValueError, match="score_512"):
        _validate_prediction_rows([row], language="en")


def test_prediction_validation_permits_missing_second_score_before_probe() -> None:
    row = make_row("eot", 300.0, score_224=0.9, score_512=None, latency_512_ms=None)
    _validate_prediction_rows([row], language="en")


def test_bootstrap_is_deterministic_and_records_requested_resamples(tmp_path: Path) -> None:
    rows = [
        make_row("hold", 300.0, score_224=0.1, turn_id="turn-1"),
        make_row("eot", 600.0, score_224=0.9, turn_id="turn-1"),
        make_row("hold", 300.0, score_224=0.1, turn_id="turn-2"),
        make_row("eot", 600.0, score_224=0.1, turn_id="turn-2"),
    ]
    cv_rows = [
        {
            "language": "en",
            "policy": "P1",
            "target": "stability",
            "selection_kind": "selected",
            "status": "available",
            "threshold224": 0.5,
            "threshold512": None,
            "false_cutoff_rate": 0.0,
            "mean_endpoint_latency_ms": 264.0,
        }
    ]
    points = _final_operating_points(cv_rows)
    first = bootstrap_confidence_intervals(
        {"en": rows}, points, output_dir=tmp_path, resamples=5, seed=123
    )
    second = bootstrap_confidence_intervals(
        {"en": rows}, points, output_dir=tmp_path, resamples=5, seed=123
    )
    assert first == second
    assert len(first) == 4
    assert {row["resamples"] for row in first} == {5}
