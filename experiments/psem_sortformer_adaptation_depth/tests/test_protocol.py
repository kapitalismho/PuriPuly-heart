from __future__ import annotations

import copy
import json

import pytest
import torch

from experiments.psem_sortformer_adaptation_depth import execution as execution_module
from experiments.psem_sortformer_adaptation_depth import protocol as protocol_module
from experiments.psem_sortformer_adaptation_depth.execution import build_cost_receipt
from experiments.psem_sortformer_adaptation_depth.preflight import canonical_sha256, sha256_file
from experiments.psem_sortformer_adaptation_depth.protocol import (
    EVAL_REGISTRY_MARKER,
    _frontier_dominates,
    _validate_checkpoint_receipt,
    append_dev_result,
    bind_payload,
    build_operator_dev_decision,
    initial_staged_state,
    open_eval_once,
    open_ta,
    validate_dev_result,
    validate_eval_authorization,
)
from experiments.psem_training_strategy_gate.sampling import DEV_ROLE


@pytest.fixture(autouse=True)
def _registered_execution_records(monkeypatch):
    monkeypatch.setattr(protocol_module, "require_registered_execution", lambda *args: {})


def _dev_result(
    arm: str,
    seed: int | None,
    *,
    contamination: float,
    false_cuts: float,
    misses: float,
) -> dict:
    prediction = _prediction(arm, seed)
    frontier = []
    for threshold in (0.35, 0.5, 0.65):
        for confirmation in (100, 300, 500):
            metrics = {
                "contamination": contamination,
                "false_cuts": false_cuts,
                "missed_replacements": misses,
            }
            frontier.append(
                {
                    "threshold": threshold,
                    "confirmation_ms": confirmation,
                    "views": {
                        "pooled": {"metrics": metrics},
                        "equal_corpus": {"metrics": metrics},
                        "corpus_specific": {
                            "AMI": {"metrics": metrics},
                            "AliMeeting": {"metrics": metrics},
                        },
                    },
                }
            )
    per_source = []
    for corpus, prefix in (("AMI", "ami"), ("AliMeeting", "ali")):
        for index in range(2):
            per_source.append(
                {
                    "source_id": f"{prefix}-{index}",
                    "corpus": corpus,
                    "metrics": {
                        "contamination": contamination + index * 0.01,
                        "false_cuts": false_cuts + index * 0.01,
                        "missed_replacements": misses + index * 0.01,
                    },
                }
            )
    per_source.sort(key=lambda row: row["source_id"])
    evidence = {"frontier": frontier, "per_source_primary": per_source}
    return bind_payload(
        {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_dev_result",
            "arm": arm,
            "seed": seed,
            "split_role": DEV_ROLE,
            "evaluation_roles": [DEV_ROLE],
            "eval_open_count": 0,
            "passed": True,
            "slot_mapping_coverage_passed": True,
            "timing_gate_passed": True,
            "frontier": frontier,
            "per_source_primary": per_source,
            "dev_evidence_sha256": __import__(
                "experiments.psem_sortformer_adaptation_depth.preflight",
                fromlist=["canonical_sha256"],
            ).canonical_sha256(evidence),
            "prediction_set_sha256": prediction["payload_sha256"],
        }
    )


def _checkpoint(arm: str, seed: int) -> dict:
    return {
        "artifact_role": "psem_sortformer_checkpoint",
        "arm": arm,
        "seed": seed,
        "checkpoint_sha256": f"{seed % 10}" * 64,
    }


def _prediction(arm: str, seed: int | None) -> dict:
    code = _code_identity()
    return bind_payload(
        {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_prediction_set",
            "arm": arm,
            "seed": seed,
            "split_role": DEV_ROLE,
            "trained_checkpoint_sha256": None if seed is None else f"{seed % 10}" * 64,
            "eval_authorization_sha256": None,
            "candidate_git_head": code["git_head"],
            "candidate_code_identity_sha256": code["payload_sha256"],
        }
    )


def _code_identity() -> dict:
    return bind_payload(
        {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_candidate_code_identity",
            "git_head": "a" * 40,
            "worktree_clean": True,
            "artifact_sha256s": {"run.py": "b" * 64},
        }
    )


def _valid_checkpoint_receipt(tmp_path, *, peak_memory: int = 1) -> dict:
    path = (tmp_path / "output" / "checkpoints" / "H-HEAD" / "7301" / "selected.pt").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "arm": "H-HEAD",
            "seed": 7301,
            "model_state_dict": {"weight": torch.ones(2)},
        },
        path,
    )
    code_sha = "c" * 64
    runtime_identity = {
        "model_graph": {
            "state_dict_schema_sha256": canonical_sha256(
                [{"name": "weight", "shape": [2], "dtype": "torch.float32"}]
            ),
            "executable_state_entry_count": 1,
        }
    }
    parameter_policy = {"arm": "H-HEAD", "trainable": ["weight"]}
    dev_source_ids_sha256 = "d" * 64
    material_payload = {
        "schema_version": 1,
        "artifact_role": "material_training_authorization",
        "arm": "H-HEAD",
        "seed": 7301,
        "candidate_code_identity_sha256": code_sha,
        "authorized_output_root": str((tmp_path / "output").resolve()),
        "dev_source_ids_sha256": dev_source_ids_sha256,
        "validation_bundle": {
            "runtime_identity": runtime_identity,
            "parameter_inventory": {
                "parameters": [
                    {
                        "name": "weight",
                        "shape": [2],
                        "dtype": "torch.float32",
                    }
                ]
            },
        },
        "overfit_receipt_sha256": "o" * 64,
        "gradient_receipt_sha256": "g" * 64,
        "timing_receipt_sha256": "t" * 64,
    }
    material_gate = {**material_payload, "payload_sha256": canonical_sha256(material_payload)}
    selected_metrics = {
        "dev_total_loss": 1.0,
        "dev_replacement_average_precision": 0.5,
    }
    summary = {
        "selected_epoch": 1,
        "selected_metrics": selected_metrics,
        "total_parameters": 2,
        "trainable_parameters": 2,
        "training_wall_clock_seconds": 1.0,
        "peak_training_memory_bytes": peak_memory,
        "native_diarization_contract_passed": True,
        "native_diarization_contract_evidence_sha256": canonical_sha256(
            {
                "overfit_receipt_sha256": "o" * 64,
                "gradient_receipt_sha256": "g" * 64,
                "timing_receipt_sha256": "t" * 64,
            }
        ),
    }
    training_payload = {
        "schema_version": 1,
        "artifact_role": "psem_sortformer_training_result",
        "arm": "H-HEAD",
        "seed": 7301,
        "authorization_sha256": material_gate["payload_sha256"],
        "candidate_code_identity_sha256": code_sha,
        "runtime_identity_sha256": canonical_sha256(runtime_identity),
        "runtime_identity": runtime_identity,
        "parameter_policy_sha256": canonical_sha256(parameter_policy),
        "parameter_policy": parameter_policy,
        "split_roles": ["PSEM-STRATEGY-TRAIN", DEV_ROLE],
        "eval_source_count": 0,
        "dev_source_ids_sha256": dev_source_ids_sha256,
        "checkpoint_path": str(path),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_size_bytes": path.stat().st_size,
        "training_summary": summary,
        "selected_checkpoint": {
            "epoch": 1,
            **selected_metrics,
            "selection_roles": [DEV_ROLE],
        },
    }
    training_result = {**training_payload, "payload_sha256": canonical_sha256(training_payload)}
    checkpoint_payload = {
        "schema_version": 1,
        "artifact_role": "psem_sortformer_checkpoint",
        "arm": "H-HEAD",
        "seed": 7301,
        "checkpoint_path": str(path),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_size_bytes": path.stat().st_size,
        "selected_epoch": 1,
        "selected_metrics": selected_metrics,
        "material_gate_sha256": material_gate["payload_sha256"],
        "material_training_authorization": material_gate,
        "authorized_output_root": str((tmp_path / "output").resolve()),
        "candidate_code_identity_sha256": code_sha,
        "runtime_identity_sha256": canonical_sha256(runtime_identity),
        "runtime_identity": runtime_identity,
        "parameter_policy_sha256": canonical_sha256(parameter_policy),
        "parameter_policy": parameter_policy,
        "split_roles": ["PSEM-STRATEGY-TRAIN", DEV_ROLE],
        "eval_source_count": 0,
        "dev_source_ids_sha256": dev_source_ids_sha256,
        "training_summary": summary,
        "training_result_sha256": training_result["payload_sha256"],
        "training_result": training_result,
    }
    return {**checkpoint_payload, "payload_sha256": canonical_sha256(checkpoint_payload)}


def test_protocol_stages_only_f0_h_t2_and_optional_ta_seed_7301(tmp_path, monkeypatch) -> None:
    output = (tmp_path / "output").resolve()
    registry = (tmp_path / "registry").resolve()
    output.mkdir()
    registry.mkdir()
    monkeypatch.setattr(protocol_module, "authority_registry_root", lambda: registry)
    monkeypatch.setattr(protocol_module, "validate_dev_result", lambda value: dict(value))

    def result(arm: str, seed: int | None, index: int) -> dict:
        return bind_payload(
            {
                "schema_version": 1,
                "artifact_role": "psem_sortformer_dev_result",
                "arm": arm,
                "seed": seed,
                "dev_evidence_sha256": f"{index:x}" * 64,
                "prediction_set": {
                    "experiment_output_root": str(output),
                    "protocol_registry_root": str(registry),
                },
            }
        )

    f0 = result("F0-FROZEN-FLOAT", None, 1)
    head = result("H-HEAD", 7301, 2)
    top = result("T2-TOP", 7301, 3)
    ta = result("TA-ALL-TEMPORAL", 7301, 4)
    state = initial_staged_state(f0)
    state = append_dev_result(state, head, [f0])
    state = append_dev_result(state, top, [f0, head])
    assert "ta_escalation" not in state
    assert "confirmation_seed_authorization" not in state
    state = append_dev_result(state, ta, [f0, head, top])
    assert [(row["arm"], row["seed"]) for row in state["completed_runs"]] == [
        ("F0-FROZEN-FLOAT", None),
        ("H-HEAD", 7301),
        ("T2-TOP", 7301),
        ("TA-ALL-TEMPORAL", 7301),
    ]
    with pytest.raises(Exception, match="out of staged order"):
        append_dev_result(state, result("H-HEAD", 7302, 5), [f0, head, top, ta])


def test_dev_pareto_staging_uses_the_complete_equal_corpus_frontier() -> None:
    def result(default: float, exception: float | None = None) -> dict:
        frontier = []
        for index, (threshold, confirmation) in enumerate(
            (threshold, confirmation)
            for threshold in (0.35, 0.5, 0.65)
            for confirmation in (100, 300, 500)
        ):
            value = exception if index == 0 and exception is not None else default
            frontier.append(
                {
                    "threshold": threshold,
                    "confirmation_ms": confirmation,
                    "views": {
                        "equal_corpus": {
                            "metrics": {
                                "contamination": value,
                                "false_cuts": value,
                                "missed_replacements": value,
                            }
                        }
                    },
                }
            )
        return {"frontier": frontier}

    head = result(1.0)
    top_dominated = result(2.0)
    top_with_frontier_gain = result(2.0, exception=0.5)
    assert _frontier_dominates(head, top_dominated)
    assert not _frontier_dominates(head, top_with_frontier_gain)


def test_dev_result_rejects_non_singleton_frontier_or_opened_eval() -> None:
    result = _dev_result("F0-FROZEN-FLOAT", None, contamination=10.0, false_cuts=10.0, misses=10.0)
    with pytest.raises(Exception, match="lean fail-closed"):
        validate_dev_result(result)
    opened = copy.deepcopy(result)
    opened["frontier"] = opened["frontier"][:1]
    opened["eval_open_count"] = 1
    payload = {key: value for key, value in opened.items() if key != "payload_sha256"}
    opened = bind_payload(payload)
    with pytest.raises(Exception, match="lean fail-closed"):
        validate_dev_result(opened)


def test_eval_open_is_global_to_the_authority_registry(tmp_path, monkeypatch) -> None:
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    registry = tmp_path / "registry"
    output_a.mkdir()
    output_b.mkdir()
    registry.mkdir()
    monkeypatch.setattr(protocol_module, "authority_registry_root", lambda: registry)
    monkeypatch.setattr(protocol_module, "validate_candidate_freeze", lambda value: value)
    monkeypatch.setattr(
        execution_module,
        "validate_current_candidate_identity",
        lambda value: value,
    )

    def candidate(output) -> dict:
        identity = _code_identity()
        return bind_payload(
            {
                "schema_version": 1,
                "artifact_role": "psem_sortformer_candidate_freeze",
                "eval_open_count": 0,
                "eval_used_for_development": False,
                "candidate_set": [
                    {"arm": "F0-FROZEN-FLOAT", "seed": None},
                    {"arm": "H-HEAD", "seed": 7301},
                ],
                "thresholds": [0.5],
                "confirmation_ms": [500],
                "cost_receipt": build_cost_receipt(
                    hourly_price_usd=1.0,
                    hourly_price_source="operator quote",
                    actual_gpu_seconds=1.0,
                    projected_remaining_gpu_seconds=1.0,
                    command="open-eval",
                ),
                "candidate_code_identity_sha256": identity["payload_sha256"],
                "candidate_git_head": identity["git_head"],
                "candidate_artifact_sha256s": identity["artifact_sha256s"],
                "experiment_output_root": str(output.resolve()),
                "protocol_registry_root": str(registry.resolve()),
                "candidate_code_identity": identity,
            }
        )

    first = open_eval_once(candidate(output_a), str(output_a.resolve()))
    assert (registry / EVAL_REGISTRY_MARKER).is_file()
    with pytest.raises(Exception, match="already been opened"):
        open_eval_once(candidate(output_b), str(output_b.resolve()))
    assert first["protocol_registry_root"] == str(registry.resolve())


def test_ta_requires_an_explicit_operator_decision_and_cost_receipt(monkeypatch) -> None:
    monkeypatch.setattr(protocol_module, "validate_dev_result", lambda value: dict(value))
    results = {
        "f0": bind_payload(
            {"artifact_role": "psem_sortformer_dev_result", "arm": "F0-FROZEN-FLOAT", "seed": None}
        ),
        "head": bind_payload(
            {"artifact_role": "psem_sortformer_dev_result", "arm": "H-HEAD", "seed": 7301}
        ),
        "top": bind_payload(
            {"artifact_role": "psem_sortformer_dev_result", "arm": "T2-TOP", "seed": 7301}
        ),
    }
    cost = build_cost_receipt(
        hourly_price_usd=1.0,
        hourly_price_source="operator quote",
        actual_gpu_seconds=1.0,
        projected_remaining_gpu_seconds=1.0,
        command="open-ta",
    )
    decision = build_operator_dev_decision(
        decision="open_ta",
        selected_arm="TA-ALL-TEMPORAL",
        rationale="The trusted operator elects to run the optional arm.",
        available_dev_results=results,
    )
    authorization = open_ta(decision, cost_receipt=cost)
    assert authorization["arm"] == "TA-ALL-TEMPORAL"
    assert authorization["seed"] == 7301
    selected = build_operator_dev_decision(
        decision="select_candidate",
        selected_arm="T2-TOP",
        rationale="T2 is selected without opening TA.",
        available_dev_results=results,
    )
    with pytest.raises(Exception, match="explicit open_ta decision"):
        open_ta(selected, cost_receipt=cost)


def test_eval_authorization_rejects_top_level_candidate_substitution(tmp_path, monkeypatch) -> None:
    output = (tmp_path / "output").resolve()
    registry = (tmp_path / "registry").resolve()
    output.mkdir()
    registry.mkdir()
    monkeypatch.setattr(protocol_module, "authority_registry_root", lambda: registry)
    identity = _code_identity()
    freeze = bind_payload(
        {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_candidate_freeze",
            "candidate_set": [{"arm": "H-HEAD", "seed": 7301}],
            "candidate_code_identity_sha256": identity["payload_sha256"],
            "candidate_git_head": identity["git_head"],
            "candidate_artifact_sha256s": identity["artifact_sha256s"],
            "experiment_output_root": str(output),
            "protocol_registry_root": str(registry),
        }
    )
    authorization = bind_payload(
        {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_eval_open_authorization",
            "candidate_freeze_sha256": freeze["payload_sha256"],
            "evaluation_roles": ["PSEM-STRATEGY-EVAL"],
            "eval_open_count": 1,
            "eval_used_for_development": False,
            "candidate_set": [{"arm": "T2-TOP", "seed": 7301}],
            "candidate_code_identity_sha256": identity["payload_sha256"],
            "candidate_git_head": identity["git_head"],
            "candidate_artifact_sha256s": identity["artifact_sha256s"],
            "experiment_output_root": str(output),
            "protocol_registry_root": str(registry),
            "candidate_freeze": freeze,
        }
    )
    (registry / EVAL_REGISTRY_MARKER).write_text(json.dumps(authorization), encoding="utf-8")
    monkeypatch.setattr(protocol_module, "validate_candidate_freeze", lambda value: value)
    with pytest.raises(Exception, match="authorization identity"):
        validate_eval_authorization(authorization)


def test_checkpoint_requires_positive_memory_and_current_material_revalidation(
    tmp_path, monkeypatch
) -> None:
    zero = _valid_checkpoint_receipt(tmp_path, peak_memory=0)
    with pytest.raises(Exception, match="checkpoint receipt is not reproducible"):
        _validate_checkpoint_receipt(zero, "H-HEAD", 7301, "c" * 64)
    receipt = _valid_checkpoint_receipt(tmp_path, peak_memory=1)
    calls = []
    monkeypatch.setattr(
        protocol_module,
        "revalidate_material_training_gate_from_bundle",
        lambda gate, **kwargs: calls.append((gate, kwargs)) or gate,
    )
    assert _validate_checkpoint_receipt(receipt, "H-HEAD", 7301, "c" * 64) == receipt
    assert calls == [(receipt["material_training_authorization"], {})]


def test_checkpoint_rejects_non_checkpoint_bytes(tmp_path, monkeypatch) -> None:
    receipt = _valid_checkpoint_receipt(tmp_path, peak_memory=1)
    path = tmp_path / "output" / "checkpoints" / "H-HEAD" / "7301" / "selected.pt"
    path.write_bytes(b"not a torch checkpoint")
    payload = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    payload["checkpoint_sha256"] = sha256_file(path)
    payload["checkpoint_size_bytes"] = path.stat().st_size
    payload["training_result"] = {
        **payload["training_result"],
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "checkpoint_size_bytes": payload["checkpoint_size_bytes"],
    }
    training_payload = {
        key: value for key, value in payload["training_result"].items() if key != "payload_sha256"
    }
    payload["training_result"]["payload_sha256"] = canonical_sha256(training_payload)
    payload["training_result_sha256"] = payload["training_result"]["payload_sha256"]
    forged = {**payload, "payload_sha256": canonical_sha256(payload)}
    monkeypatch.setattr(
        protocol_module,
        "revalidate_material_training_gate_from_bundle",
        lambda gate, **kwargs: gate,
    )
    with pytest.raises(Exception, match="material gate is not currently valid"):
        _validate_checkpoint_receipt(forged, "H-HEAD", 7301, "c" * 64)


def test_checkpoint_rejects_extra_state_dict_key(tmp_path, monkeypatch) -> None:
    receipt = _valid_checkpoint_receipt(tmp_path, peak_memory=1)
    path = tmp_path / "output" / "checkpoints" / "H-HEAD" / "7301" / "selected.pt"
    torch.save(
        {
            "schema_version": 1,
            "arm": "H-HEAD",
            "seed": 7301,
            "model_state_dict": {
                "weight": torch.ones(2),
                "unauthorized.extra": torch.ones(1),
            },
        },
        path,
    )
    payload = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    payload["checkpoint_sha256"] = sha256_file(path)
    payload["checkpoint_size_bytes"] = path.stat().st_size
    payload["training_result"] = {
        **payload["training_result"],
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "checkpoint_size_bytes": payload["checkpoint_size_bytes"],
    }
    training_payload = {
        key: value for key, value in payload["training_result"].items() if key != "payload_sha256"
    }
    payload["training_result"]["payload_sha256"] = canonical_sha256(training_payload)
    payload["training_result_sha256"] = payload["training_result"]["payload_sha256"]
    forged = {**payload, "payload_sha256": canonical_sha256(payload)}
    monkeypatch.setattr(
        protocol_module,
        "revalidate_material_training_gate_from_bundle",
        lambda gate, **kwargs: gate,
    )
    with pytest.raises(Exception, match="authorized graph"):
        _validate_checkpoint_receipt(forged, "H-HEAD", 7301, "c" * 64)


def test_eval_checkpoint_requires_the_exact_frozen_receipt() -> None:
    candidate = {"checkpoint_receipt_sha256": "a" * 64}
    execution_module._require_frozen_checkpoint_receipt(candidate, {"payload_sha256": "a" * 64})
    with pytest.raises(Exception, match="receipt differs"):
        execution_module._require_frozen_checkpoint_receipt(candidate, {"payload_sha256": "b" * 64})
