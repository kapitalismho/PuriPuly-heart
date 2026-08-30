from __future__ import annotations

import copy

import pytest

from experiments.psem_sortformer_adaptation_depth import evaluation as evaluation_module
from experiments.psem_sortformer_adaptation_depth import reporting as reporting_module
from experiments.psem_sortformer_adaptation_depth.evaluation import (
    _aggregate_frame_diagnostics,
    _aggregate_mapping_diagnostics,
)
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
            "evaluation_roles": [EVAL_ROLE],
            "eval_open_count": 1,
            "eval_used_for_development": False,
        }
    )


def test_final_reporting_emits_singleton_f0_plus_winner_engineering_artifacts(
    monkeypatch,
) -> None:
    candidates = [
        {"arm": "F0-FROZEN-FLOAT", "seed": None},
        {"arm": "T2-TOP", "seed": 7301},
    ]
    authorization = _authorization(candidates)
    results = [
        _singleton_result("F0-FROZEN-FLOAT", None, 10.0),
        _singleton_result("T2-TOP", 7301, 6.0),
    ]
    monkeypatch.setattr(reporting_module, "validate_eval_authorization", lambda value: value)
    monkeypatch.setattr(
        reporting_module,
        "_candidate_results",
        lambda *_args: results,
    )
    artifacts, markdown = build_final_artifacts(
        eval_authorization=authorization,
        eval_results=[],
        eval_prediction_sets=[],
        training_results=[{"arm": "T2-TOP", "seed": 7301}],
    )
    assert artifacts["singleton_metrics.json"]["operating_point"] == {
        "threshold": 0.5,
        "confirmation_ms": 500,
    }
    assert set(artifacts["singleton_metrics.json"]["views"]) == {
        "F0-FROZEN-FLOAT:None",
        "T2-TOP:7301",
    }
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
