from __future__ import annotations

import copy

import pytest

from experiments.psem_sortformer_adaptation_depth import evaluation as evaluation_module
from experiments.psem_sortformer_adaptation_depth import execution as execution_module
from experiments.psem_sortformer_adaptation_depth import protocol as protocol_module
from experiments.psem_sortformer_adaptation_depth import reporting as reporting_module
from experiments.psem_sortformer_adaptation_depth.evaluation import (
    _aggregate_frame_diagnostics,
    _aggregate_mapping_diagnostics,
)
from experiments.psem_sortformer_adaptation_depth.execution import build_cost_receipt
from experiments.psem_sortformer_adaptation_depth.protocol import bind_payload
from experiments.psem_sortformer_adaptation_depth.reporting import (
    build_final_artifacts,
    validate_eval_result,
)
from experiments.psem_training_strategy_gate.sampling import DEV_ROLE, EVAL_ROLE


def _singleton_result(arm: str, seed: int | None, value: float) -> dict:
    metrics = {
        "contamination": value,
        "false_cuts": value,
        "missed_replacements": value,
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
            "frontier": [
                {
                    "threshold": 0.5,
                    "confirmation_ms": 500,
                    "views": {
                        "pooled": {"metrics": metrics},
                        "AMI": {"metrics": metrics},
                        "AliMeeting": {"metrics": metrics},
                    },
                }
            ],
            "per_source_rows": [],
        }
    )


def _authorization(candidate_set: list[dict]) -> dict:
    decision = bind_payload(
        {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_operator_dev_decision",
            "decision": "select_candidate",
            "selected_arm": candidate_set[1]["arm"],
            "rationale": "The trusted operator selected the shallower supported winner.",
        }
    )
    return bind_payload(
        {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_eval_open_authorization",
            "candidate_set": candidate_set,
            "candidate_freeze": {"operator_dev_decision": decision},
            "candidate_code_identity_sha256": "c" * 64,
            "evaluation_roles": [EVAL_ROLE],
            "eval_open_count": 1,
            "eval_used_for_development": False,
        }
    )


def _selected_candidate_and_training(arm: str) -> tuple[dict, dict]:
    summary = {
        "training_wall_clock_seconds": 1.0,
        "peak_training_memory_bytes": 1,
        "total_parameters": 10,
        "trainable_parameters": 2,
    }
    training = bind_payload(
        {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_training_result",
            "arm": arm,
            "seed": 7301,
            "checkpoint_sha256": "d" * 64,
            "candidate_code_identity_sha256": "c" * 64,
            "training_summary": summary,
        }
    )
    return (
        {
            "arm": arm,
            "seed": 7301,
            "training_result_sha256": training["payload_sha256"],
            "checkpoint_sha256": "d" * 64,
            "checkpoint_receipt_sha256": "e" * 64,
            "training_summary": summary,
        },
        training,
    )


def test_final_reporting_emits_singleton_f0_plus_winner_engineering_artifacts(
    monkeypatch,
) -> None:
    selected_candidate, training = _selected_candidate_and_training("T2-TOP")
    candidates = [
        {"arm": "F0-FROZEN-FLOAT", "seed": None},
        selected_candidate,
    ]
    authorization = _authorization(candidates)
    results = [
        _singleton_result("F0-FROZEN-FLOAT", None, 10.0),
        _singleton_result("T2-TOP", 7301, 6.0),
    ]
    registered = []
    monkeypatch.setattr(reporting_module, "validate_eval_authorization", lambda value: value)
    monkeypatch.setattr(
        reporting_module,
        "_candidate_results",
        lambda *_args: results,
    )
    monkeypatch.setattr(
        reporting_module,
        "require_registered_execution",
        lambda kind, value: registered.append((kind, value)),
    )
    artifacts, markdown = build_final_artifacts(
        eval_authorization=authorization,
        eval_results=[],
        eval_prediction_sets=[],
        training_results=[training],
    )
    assert artifacts["singleton_metrics.json"]["operating_point"] == {
        "threshold": 0.5,
        "confirmation_ms": 500,
    }
    assert set(artifacts["singleton_metrics.json"]["views"]) == {
        "F0-FROZEN-FLOAT:None",
        "T2-TOP:7301",
    }
    assert artifacts["timing_and_compute.json"]["training_results"] == [training]
    assert registered == [("training-result", training)]
    decision = artifacts["decision_receipt.json"]
    assert decision["selected_arm"] == "T2-TOP"
    assert decision["significance_claim"] is False
    assert decision["seed_stability_claim"] is False
    assert decision["evidence_level"] == "engineering"
    assert "exactly F0 and the operator-selected seed-7301 candidate" in markdown


def test_final_reporting_rejects_more_than_f0_plus_one_winner(monkeypatch) -> None:
    authorization = _authorization(
        [
            {"arm": "F0-FROZEN-FLOAT", "seed": None},
            {"arm": "H-HEAD", "seed": 7301},
            {"arm": "T2-TOP", "seed": 7301},
        ]
    )
    monkeypatch.setattr(reporting_module, "validate_eval_authorization", lambda value: value)
    with pytest.raises(Exception, match="F0 plus one selected candidate"):
        build_final_artifacts(
            eval_authorization=authorization,
            eval_results=[],
            eval_prediction_sets=[],
            training_results=[],
        )


def test_eval_result_requires_the_singleton_operating_point_and_views(monkeypatch) -> None:
    monkeypatch.setattr(reporting_module, "require_registered_execution", lambda *args: {})
    result = _singleton_result("H-HEAD", 7301, 1.0)
    assert validate_eval_result(result) == result
    forged = copy.deepcopy(result)
    forged["frontier"].append(copy.deepcopy(forged["frontier"][0]))
    payload = {key: value for key, value in forged.items() if key != "payload_sha256"}
    forged = bind_payload(payload)
    with pytest.raises(Exception, match="lean integrity gate"):
        validate_eval_result(forged)


def test_diagnostic_aggregates_remain_reproducible() -> None:
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


def test_final_reporting_rejects_missing_or_substituted_training_result(monkeypatch) -> None:
    selected_candidate, training = _selected_candidate_and_training("T2-TOP")
    authorization = _authorization(
        [
            {"arm": "F0-FROZEN-FLOAT", "seed": None},
            selected_candidate,
        ]
    )
    results = [
        _singleton_result("F0-FROZEN-FLOAT", None, 10.0),
        _singleton_result("T2-TOP", 7301, 6.0),
    ]
    monkeypatch.setattr(reporting_module, "validate_eval_authorization", lambda value: value)
    monkeypatch.setattr(reporting_module, "_candidate_results", lambda *_args: results)

    with pytest.raises(Exception, match="selected training result"):
        build_final_artifacts(
            eval_authorization=authorization,
            eval_results=[],
            eval_prediction_sets=[],
            training_results=[],
        )

    substituted_payload = {key: value for key, value in training.items() if key != "payload_sha256"}
    substituted_payload["candidate_code_identity_sha256"] = "f" * 64
    substituted = bind_payload(substituted_payload)
    with pytest.raises(Exception, match="selected training result"):
        build_final_artifacts(
            eval_authorization=authorization,
            eval_results=[],
            eval_prediction_sets=[],
            training_results=[substituted],
        )


def test_public_freeze_to_final_report_preserves_selected_training_summary(
    tmp_path, monkeypatch
) -> None:
    output = (tmp_path / "output").resolve()
    registry = (tmp_path / "registry").resolve()
    output.mkdir()
    registry.mkdir()
    monkeypatch.setattr(protocol_module, "authority_registry_root", lambda: registry)
    monkeypatch.setattr(protocol_module, "validate_dev_result", lambda value: dict(value))
    monkeypatch.setattr(
        execution_module,
        "validate_current_candidate_identity",
        lambda value: value,
    )
    code_identity = bind_payload(
        {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_candidate_code_identity",
            "git_head": "a" * 40,
            "worktree_clean": True,
            "artifact_sha256s": {"run.py": "b" * 64},
        }
    )
    summary = {
        "final_step": 256,
        "training_wall_clock_seconds": 1.0,
        "peak_training_memory_bytes": 1,
        "total_parameters": 10,
        "trainable_parameters": 2,
    }
    training = bind_payload(
        {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_training_result",
            "arm": "T2-TOP",
            "seed": 7301,
            "checkpoint_sha256": "d" * 64,
            "candidate_code_identity_sha256": code_identity["payload_sha256"],
            "training_summary": summary,
        }
    )
    checkpoint = bind_payload(
        {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_checkpoint",
            "arm": "T2-TOP",
            "seed": 7301,
            "final_step": 256,
            "checkpoint_sha256": "d" * 64,
            "training_result_sha256": training["payload_sha256"],
            "training_summary": summary,
        }
    )
    frozen_prediction = bind_payload(
        {
            "artifact_role": "psem_sortformer_prediction_set",
            "arm": "F0-FROZEN-FLOAT",
            "seed": None,
            "experiment_output_root": str(output),
            "protocol_registry_root": str(registry),
        }
    )
    head_prediction = bind_payload(
        {
            "artifact_role": "psem_sortformer_prediction_set",
            "arm": "H-HEAD",
            "seed": 7301,
            "experiment_output_root": str(output),
            "protocol_registry_root": str(registry),
        }
    )
    selected_prediction = bind_payload(
        {
            "artifact_role": "psem_sortformer_prediction_set",
            "arm": "T2-TOP",
            "seed": 7301,
            "trained_checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "trained_checkpoint_receipt_sha256": checkpoint["payload_sha256"],
            "experiment_output_root": str(output),
            "protocol_registry_root": str(registry),
        }
    )
    frozen_result = bind_payload(
        {
            "artifact_role": "psem_sortformer_dev_result",
            "arm": "F0-FROZEN-FLOAT",
            "seed": None,
            "dev_evidence_sha256": "1" * 64,
            "prediction_set": frozen_prediction,
            "prediction_set_sha256": frozen_prediction["payload_sha256"],
        }
    )
    head_result = bind_payload(
        {
            "artifact_role": "psem_sortformer_dev_result",
            "arm": "H-HEAD",
            "seed": 7301,
            "dev_evidence_sha256": "2" * 64,
            "prediction_set": head_prediction,
            "prediction_set_sha256": head_prediction["payload_sha256"],
        }
    )
    selected_result = bind_payload(
        {
            "artifact_role": "psem_sortformer_dev_result",
            "arm": "T2-TOP",
            "seed": 7301,
            "dev_evidence_sha256": "3" * 64,
            "prediction_set": selected_prediction,
            "prediction_set_sha256": selected_prediction["payload_sha256"],
        }
    )
    dev_results = [frozen_result, head_result, selected_result]
    decision = bind_payload(
        {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_operator_dev_decision",
            "decision": "select_candidate",
            "selected_arm": "T2-TOP",
            "rationale": "The trusted operator selected T2.",
            "available_dev_result_sha256s": {
                f"{result['arm']}:{result.get('seed')}": result["payload_sha256"]
                for result in dev_results
            },
            "eval_open_count": 0,
        }
    )
    state = protocol_module.initial_staged_state(frozen_result)
    state = protocol_module.append_dev_result(state, head_result, [frozen_result])
    state = protocol_module.append_dev_result(
        state,
        selected_result,
        [frozen_result, head_result],
    )
    cost = build_cost_receipt(
        hourly_price_usd=1.0,
        hourly_price_source="operator quote",
        actual_gpu_seconds=1.0,
        projected_remaining_gpu_seconds=1.0,
        command="freeze-candidates",
    )
    checkpoint_receipts = {"T2-TOP": checkpoint}
    prediction_sets = {
        "F0-FROZEN-FLOAT": frozen_prediction,
        "T2-TOP": selected_prediction,
    }
    candidate_freeze = protocol_module.freeze_candidate_set(
        state,
        dev_results,
        checkpoint_receipts,
        prediction_sets,
        code_identity,
        operator_decision=decision,
        cost_receipt=cost,
    )
    stale_payload = {key: value for key, value in state.items() if key != "payload_sha256"}
    stale_payload["completed_runs"] = stale_payload["completed_runs"][:-1]
    stale_state = bind_payload(stale_payload)
    with pytest.raises(Exception, match="exact DEV evidence replay"):
        protocol_module.freeze_candidate_set(
            stale_state,
            dev_results,
            checkpoint_receipts,
            prediction_sets,
            code_identity,
            operator_decision=decision,
            cost_receipt=cost,
        )
    authorization = protocol_module.open_eval_once(candidate_freeze, str(output))
    assert authorization["candidate_set"][1]["training_summary"] == summary

    results = [
        _singleton_result("F0-FROZEN-FLOAT", None, 10.0),
        _singleton_result("T2-TOP", 7301, 6.0),
    ]
    monkeypatch.setattr(reporting_module, "validate_eval_authorization", lambda value: value)
    monkeypatch.setattr(reporting_module, "_candidate_results", lambda *_args: results)
    monkeypatch.setattr(reporting_module, "require_registered_execution", lambda *_args: {})
    artifacts, _ = build_final_artifacts(
        eval_authorization=authorization,
        eval_results=[],
        eval_prediction_sets=[],
        training_results=[training],
    )
    assert artifacts["timing_and_compute.json"]["training_results"] == [training]
