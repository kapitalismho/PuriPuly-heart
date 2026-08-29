from __future__ import annotations

import copy
import json

import pytest
import torch

from experiments.psem_sortformer_adaptation_depth import execution as execution_module
from experiments.psem_sortformer_adaptation_depth import protocol as protocol_module
from experiments.psem_sortformer_adaptation_depth.preflight import canonical_sha256, sha256_file
from experiments.psem_sortformer_adaptation_depth.protocol import (
    EVAL_REGISTRY_MARKER,
    _frontier_dominates,
    _validate_checkpoint_receipt,
    authorize_conditional_arm_audit,
    bind_payload,
    initial_staged_state,
    open_eval_once,
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


def test_protocol_opens_deep_arm_then_freezes_before_single_eval_open() -> None:
    f0 = _dev_result("F0-FROZEN-FLOAT", None, contamination=10.0, false_cuts=10.0, misses=10.0)
    head = _dev_result("H-HEAD", 7301, contamination=8.0, false_cuts=8.0, misses=8.0)
    top = _dev_result("T2-TOP", 7301, contamination=6.0, false_cuts=7.0, misses=6.0)
    ta = _dev_result("TA-ALL-TEMPORAL", 7301, contamination=5.0, false_cuts=6.5, misses=5.0)
    for result in (f0, head, top, ta):
        with pytest.raises(Exception, match="embed its prediction evidence"):
            initial_staged_state(result)


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


def test_dev_result_rejects_partial_frontier_or_opened_eval() -> None:
    result = _dev_result("F0-FROZEN-FLOAT", None, contamination=10.0, false_cuts=10.0, misses=10.0)
    partial = copy.deepcopy(result)
    partial["frontier"].pop()
    payload = {key: value for key, value in partial.items() if key != "payload_sha256"}
    partial = bind_payload(payload)
    with pytest.raises(Exception, match="complete fixed frontier"):
        validate_dev_result(partial)
    opened = copy.deepcopy(result)
    opened["eval_open_count"] = 1
    payload = {key: value for key, value in opened.items() if key != "payload_sha256"}
    opened = bind_payload(payload)
    with pytest.raises(Exception, match="fail-closed"):
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
                "candidate_set": [],
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


def test_ta_canary_requires_opened_dev_escalation_and_unopened_eval_registry(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "output"
    registry = tmp_path / "registry"
    output.mkdir()
    registry.mkdir()
    monkeypatch.setattr(protocol_module, "authority_registry_root", lambda: registry)
    payload = {
        "schema_version": 1,
        "artifact_role": "staged_execution_state",
        "eval_open_count": 0,
        "eval_used_for_development": False,
        "experiment_output_root": str(output.resolve()),
        "protocol_registry_root": str(registry.resolve()),
        "completed_runs": [
            {"arm": "F0-FROZEN-FLOAT", "seed": None},
            {"arm": "H-HEAD", "seed": 7301},
            {"arm": "T2-TOP", "seed": 7301},
        ],
        "ta_escalation": {
            "decision": "opened",
            "dev_evidence_sha256": "d" * 64,
        },
    }
    state = bind_payload(payload)
    results = [bind_payload({"artifact_role": "result", "index": index}) for index in range(3)]
    monkeypatch.setattr(
        protocol_module,
        "validate_staged_execution_state",
        lambda value, evidence: value,
    )
    authorization = authorize_conditional_arm_audit("TA-ALL-TEMPORAL", state, results)
    assert authorization["arm"] == "TA-ALL-TEMPORAL"
    (registry / EVAL_REGISTRY_MARKER).write_text("{}", encoding="utf-8")
    with pytest.raises(Exception, match="precedes its frozen DEV escalation gate"):
        authorize_conditional_arm_audit("TA-ALL-TEMPORAL", state, results)


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
