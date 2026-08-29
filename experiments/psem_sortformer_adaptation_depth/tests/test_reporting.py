from __future__ import annotations

import copy

import pytest

from experiments.psem_sortformer_adaptation_depth import evaluation as evaluation_module
from experiments.psem_sortformer_adaptation_depth.evaluation import (
    _aggregate_frame_diagnostics,
    _aggregate_mapping_diagnostics,
    _equal_corpus_view,
)
from experiments.psem_sortformer_adaptation_depth.protocol import bind_payload
from experiments.psem_sortformer_adaptation_depth.reporting import (
    DECISION_METRIC_DIRECTIONS,
    _bootstrap_delta,
    _paired_source_maps,
    _stable_gain,
    _timing_compute_report,
    build_bootstrap_report,
    build_final_artifacts,
    decide_outcome,
)
from experiments.psem_training_strategy_gate.sampling import DEV_ROLE, EVAL_ROLE

TOPOLOGY = {
    key: {}
    for key in (
        "clean_direct_different_speaker_handoff",
        "silence_gap_different_speaker_handoff",
        "same_speaker_silence_gap_resume",
        "overlap_return",
        "overlap_takeover",
        "short_backchannel_return",
    )
}


def _prediction(arm: str, seed: int | None, checkpoint: str | None, auth_sha: str) -> dict:
    return bind_payload(
        {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_prediction_set",
            "arm": arm,
            "seed": seed,
            "split_role": EVAL_ROLE,
            "trained_checkpoint_sha256": checkpoint,
            "eval_authorization_sha256": auth_sha,
        }
    )


def _result(
    arm: str,
    seed: int | None,
    prediction_sha: str,
    value: float,
) -> dict:
    full = {
        "active_speech_hours": 1.0,
        "exclusive_other_contamination_seconds_per_active_speech_hour": value,
        "false_cut_count": int(value),
        "missed_replacement_count": int(value),
        "topology": TOPOLOGY,
    }
    metrics = {
        "contamination": value,
        "false_cuts": value,
        "missed_replacements": value,
    }
    frontier = [
        {
            "threshold": threshold,
            "confirmation_ms": confirmation,
            "views": {
                "pooled": {"metrics": metrics, "full_metrics": full},
                "equal_corpus": {"metrics": metrics},
                "corpus_specific": {
                    "AMI": {"metrics": metrics, "full_metrics": full},
                    "AliMeeting": {"metrics": metrics, "full_metrics": full},
                },
            },
        }
        for threshold in (0.35, 0.5, 0.65)
        for confirmation in (100, 300, 500)
    ]
    per_source = [
        {
            "source_id": f"{prefix}-{index}",
            "corpus": corpus,
            "metrics": metrics,
        }
        for corpus, prefix in (("AMI", "ami"), ("AliMeeting", "ali"))
        for index in range(2)
    ]
    per_source.sort(key=lambda row: row["source_id"])
    frame = {
        row["source_id"]: {
            "anchor_only": {},
            "anchor_with_overlap": {},
            "active_anchor_absent": {},
            "gt_overlap_anchor_dropout": {},
        }
        for row in per_source
    }
    mapping = {
        row["source_id"]: {
            "mapping_coverage": 1.0,
            "slot_instability_count": 0,
            "unexpected_reset_count": 0,
        }
        for row in per_source
    }
    return bind_payload(
        {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_eval_result",
            "arm": arm,
            "seed": seed,
            "split_role": EVAL_ROLE,
            "evaluation_roles": [EVAL_ROLE],
            "eval_open_count": 1,
            "passed": True,
            "slot_mapping_coverage_passed": True,
            "timing_gate_passed": True,
            "frontier": frontier,
            "per_source_primary": per_source,
            "per_source_rows": [],
            "mapping_diagnostics": mapping,
            "frame_diagnostics": frame,
            "prediction_set_sha256": prediction_sha,
            "eval_evidence_sha256": "e" * 64,
        }
    )


def test_final_reporting_binds_historical_baselines_bootstrap_and_decision() -> None:
    candidates = [
        {"arm": "F0-FROZEN-FLOAT", "seed": None, "checkpoint_sha256": None},
        {"arm": "H-HEAD", "seed": 7301, "checkpoint_sha256": "1" * 64},
        {"arm": "T2-TOP", "seed": 7301, "checkpoint_sha256": "2" * 64},
    ]
    provisional = bind_payload(
        {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_eval_open_authorization",
            "candidate_set": candidates,
            "evaluation_roles": [EVAL_ROLE],
            "eval_open_count": 1,
            "eval_used_for_development": False,
        }
    )
    predictions = [
        _prediction(
            row["arm"], row["seed"], row["checkpoint_sha256"], provisional["payload_sha256"]
        )
        for row in candidates
    ]
    authorization = provisional
    results = [
        _result("F0-FROZEN-FLOAT", None, predictions[0]["payload_sha256"], 10.0),
        _result("H-HEAD", 7301, predictions[1]["payload_sha256"], 8.0),
        _result("T2-TOP", 7301, predictions[2]["payload_sha256"], 6.0),
    ]
    with pytest.raises(Exception, match="EVAL authorization identity is invalid"):
        build_final_artifacts(
            eval_authorization=authorization,
            eval_results=results,
            eval_prediction_sets=predictions,
            training_results=[],
        )


def test_bootstrap_pareto_uses_delay_boundary_and_both_overlap_metrics() -> None:
    left = {}
    right = {}
    for corpus, prefix in (("AMI", "ami"), ("AliMeeting", "ali")):
        for index in range(2):
            source_id = f"{prefix}-{index}"
            left[source_id] = {
                "corpus": corpus,
                "metrics": {
                    metric: 1.0 if direction == "lower" else 0.9
                    for metric, direction in DECISION_METRIC_DIRECTIONS.items()
                },
            }
            right[source_id] = {
                "corpus": corpus,
                "metrics": {
                    metric: 2.0 if direction == "lower" else 0.5
                    for metric, direction in DECISION_METRIC_DIRECTIONS.items()
                },
            }
    comparison = _paired_source_maps(left, right, "deep", "shallow", 0)
    assert _stable_gain(comparison)
    assert comparison["metrics"]["replacement_delay_p90"]["upper"] < 0
    assert comparison["metrics"]["overlap_takeover_success"]["upper"] < 0
    damaged = {
        source_id: {
            **row,
            "metrics": {**row["metrics"], "replacement_delay_p90": 100.0},
        }
        for source_id, row in left.items()
    }
    damaged_comparison = _paired_source_maps(damaged, right, "deep", "shallow", 100)
    assert not _stable_gain(damaged_comparison)


def test_equal_corpus_and_diagnostic_views_execute_complete_metric_shapes() -> None:
    def corpus_view(scale: float) -> dict:
        full = {
            "active_speech_hours": scale,
            "speaker_induced_cut_count_per_active_speech_hour": scale,
            "exclusive_other_contamination_seconds_per_active_speech_hour": scale * 2,
            "replacement_emit_delay_ms": {"p50": scale * 3, "p90": scale * 4},
            "backdated_boundary_error_ms": {"p50": scale * 5, "p90": scale * 6},
            "overlap_return_preservation_rate": scale / 10,
            "overlap_takeover_success_rate": scale / 20,
            "topology": {"overlap_return": {"episode_count": scale, "success_rate": scale / 10}},
        }
        return {
            "metrics": {
                "contamination": scale * 2,
                "false_cuts": scale * 7,
                "missed_replacements": scale * 8,
            },
            "full_metrics": full,
        }

    equal = _equal_corpus_view({"AMI": corpus_view(1.0), "AliMeeting": corpus_view(3.0)})
    assert equal["full_metrics"]["replacement_emit_delay_ms"]["p90"] == 8.0
    assert equal["full_metrics"]["topology"]["overlap_return"]["success_rate"] == 0.2

    frame_rows = {
        source_id: {
            "anchor_only": {"support_frames": 10, "success_frames": successes},
            "anchor_with_overlap": {"support_frames": 10, "success_frames": successes},
            "active_anchor_absent": {"support_frames": 10, "success_frames": successes},
            "gt_overlap_anchor_dropout": {
                "sustained_100_ms_count": 1,
                "sustained_300_ms_count": 2,
                "sustained_500_ms_count": 3,
            },
        }
        for source_id, successes in (("ami", 8), ("ali", 6))
    }
    corpora = {"ami": "AMI", "ali": "AliMeeting"}
    frame = _aggregate_frame_diagnostics(frame_rows, corpora)
    assert frame["pooled"]["anchor_only"]["recall"] == 0.7
    assert frame["equal_corpus"]["gt_overlap_anchor_dropout"]["mean_sustained_500_ms_count"] == 3.0

    mapping_rows = {
        source_id: {
            "episode_count": 10,
            "mapped_episode_count": mapped,
            "slot_instability_count": 0,
            "reset_exposure_count": 1,
            "unexpected_reset_count": 0,
        }
        for source_id, mapped in (("ami", 10), ("ali", 8))
    }
    mapping = _aggregate_mapping_diagnostics(mapping_rows, corpora)
    assert mapping["pooled"]["mapping_coverage"] == 0.9
    assert mapping["equal_corpus"]["mapping_coverage"] == 0.9


def test_evaluation_boundary_revalidates_evaluator_contract(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        evaluation_module,
        "evaluator_reconstruction_contract",
        lambda: calls.append(True) or {"passed": True},
    )
    monkeypatch.setattr(
        evaluation_module,
        "validate_prediction_set",
        lambda value, authorization, **kwargs: (_ for _ in ()).throw(RuntimeError("stop")),
    )
    with pytest.raises(RuntimeError, match="stop"):
        evaluation_module.evaluate_prediction_set({})
    assert calls == [True]


def test_public_dev_evaluation_is_sealed_after_eval_but_historical_replay_is_pure(
    tmp_path, monkeypatch
) -> None:
    marker = tmp_path / "eval-open.json"
    marker.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(evaluation_module, "_eval_registry_marker", lambda: marker)
    monkeypatch.setattr(evaluation_module, "evaluator_reconstruction_contract", lambda: {})
    monkeypatch.setattr(
        evaluation_module,
        "validate_prediction_set",
        lambda value, authorization, **kwargs: {
            "split_role": DEV_ROLE,
            "arm": "F0-FROZEN-FLOAT",
            "seed": None,
        },
    )
    with pytest.raises(Exception, match="DEV evaluation is sealed"):
        evaluation_module.evaluate_prediction_set({})
    monkeypatch.setattr(
        evaluation_module,
        "load_sessions",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("historical replay reached")),
    )
    with pytest.raises(RuntimeError, match="historical replay reached"):
        evaluation_module.evaluate_prediction_set({}, historical_replay=True)


def test_timing_report_rejects_zero_peak_memory_for_trained_candidate() -> None:
    summary = {
        "training_wall_clock_seconds": 1.0,
        "peak_training_memory_bytes": 0,
        "total_parameters": 2,
        "trainable_parameters": 1,
        "native_diarization_contract_passed": True,
        "native_diarization_contract_evidence_sha256": "e" * 64,
    }
    training_payload = {
        "schema_version": 1,
        "artifact_role": "psem_sortformer_training_result",
        "arm": "H-HEAD",
        "seed": 7301,
        "checkpoint_sha256": "c" * 64,
        "training_summary": summary,
        **summary,
    }
    training = bind_payload(training_payload)
    candidate = {
        "arm": "H-HEAD",
        "seed": 7301,
        "checkpoint_sha256": "c" * 64,
        "checkpoint_receipt_sha256": "r" * 64,
        "training_result_sha256": training["payload_sha256"],
        "training_summary": summary,
    }
    with pytest.raises(Exception, match="compute evidence"):
        _timing_compute_report(
            [{"arm": "H-HEAD", "seed": 7301, "timing_gate_passed": True}],
            [training],
            [candidate],
        )


def _decision_result(arm: str, seed: int | None, value: float, integrity: bool = True) -> dict:
    topology = {
        name: {
            "eligible_episode_count": 10,
            "episodes_with_aligned_cut": round(10 - value),
            "episodes_with_predicted_cut": round(value),
            "episodes_with_reference_replacement": 10,
            "overlap_return_preservation_rate": 1.0 - value / 10
            if name == "overlap_return"
            else None,
            "overlap_takeover_success_rate": 1.0 - value / 10
            if name == "overlap_takeover"
            else None,
        }
        for name in TOPOLOGY
    }
    per_source = []
    for corpus, source_id in (("AMI", "ami"), ("AliMeeting", "ali")):
        per_source.append(
            {
                "source_id": source_id,
                "corpus": corpus,
                "full_metrics": {
                    "active_speech_hours": 1.0,
                    "exclusive_other_contamination_seconds_per_active_speech_hour": value,
                    "speaker_induced_cut_count_per_active_speech_hour": value,
                    "false_cut_count": int(value),
                    "missed_replacement_count": int(value),
                    "replacement_emit_delay_ms": {"p50": value, "p90": value},
                    "backdated_boundary_error_ms": {"p50": value, "p90": value},
                    "overlap_return_preservation_rate": 1.0 - value / 10,
                    "overlap_takeover_success_rate": 1.0 - value / 10,
                    "topology": topology,
                },
            }
        )
    return {
        "arm": arm,
        "seed": seed,
        "per_source_primary": per_source,
        "slot_mapping_coverage_passed": integrity,
        "timing_gate_passed": integrity,
    }


def test_bootstrap_and_integrity_decision_paths_fail_closed() -> None:
    with pytest.raises(Exception, match="no defined metric support"):
        _bootstrap_delta({}, seed=1)
    results = [
        _decision_result("F0-FROZEN-FLOAT", None, 8.0),
        _decision_result("H-HEAD", 7301, 4.0),
        _decision_result("H-HEAD", 7302, 4.0),
        _decision_result("T2-TOP", 7301, 4.0),
    ]
    bootstrap = build_bootstrap_report(results)
    candidates = [
        {
            "arm": row["arm"],
            "seed": row["seed"],
            "training_summary": {"native_diarization_contract_passed": True},
        }
        for row in results
    ]
    accepted = decide_outcome(results, bootstrap, candidates)
    assert accepted["outcome"] == "A"
    assert accepted["evidence_level"] == "engineering"
    assert accepted["improvement_holds_in_overlap_return"] is True
    assert accepted["improvement_holds_in_overlap_takeover"] is True
    damaged = [*results[:-1], {**results[-1], "timing_gate_passed": False}]
    rejected = decide_outcome(damaged, bootstrap, candidates)
    assert rejected["outcome"] == "D"
    assert rejected["adapted_teacher_ready_for_native_causal_binding_gate"] is False
    unsupported = copy.deepcopy(results)
    unsupported[1]["per_source_primary"][0]["full_metrics"]["replacement_emit_delay_ms"]["p90"] = (
        None
    )
    unsupported_bootstrap = build_bootstrap_report(unsupported)
    unsupported_decision = decide_outcome(unsupported, unsupported_bootstrap, candidates)
    assert unsupported_decision["outcome"] == "D"
    comparison = next(
        row
        for row in unsupported_bootstrap["comparisons"]
        if row["left_arm"] == "H-HEAD" and row["right_arm"] == "F0-FROZEN-FLOAT"
    )
    assert comparison["metrics"]["replacement_delay_p90"]["status"] == "unsupported"
    assert comparison["metrics"]["replacement_delay_p90"]["invalid_source_ids"] == ["ami"]


def test_engineering_ta_decision_retains_unsupported_nonprincipal_topology() -> None:
    results = [
        _decision_result("F0-FROZEN-FLOAT", None, 8.0),
        _decision_result("H-HEAD", 7301, 6.0),
        _decision_result("H-HEAD", 7302, 6.0),
        _decision_result("T2-TOP", 7301, 4.0),
        _decision_result("T2-TOP", 7302, 4.0),
        _decision_result("TA-ALL-TEMPORAL", 7301, 2.0),
        _decision_result("TA-ALL-TEMPORAL", 7302, 2.0),
    ]
    results[-1]["per_source_primary"][0]["full_metrics"]["topology"][
        "clean_direct_different_speaker_handoff"
    ]["episodes_with_aligned_cut"] = None
    bootstrap = build_bootstrap_report(results)
    candidates = [
        {
            "arm": row["arm"],
            "seed": row["seed"],
            "training_summary": {"native_diarization_contract_passed": True},
        }
        for row in results
    ]
    decision = decide_outcome(results, bootstrap, candidates)
    assert decision["outcome"] == "C"
    evidence = decision["ta_concentration_evidence"]
    assert evidence["mandatory_topology_concentration_pass"] is False
    interval = evidence["mandatory_topology_bootstrap_by_shallower_arm"]["H-HEAD"][
        "clean_direct_different_speaker_handoff"
    ]
    assert interval["status"] == "unsupported"
    assert interval["invalid_source_ids"] == ["ali", "ami"]
