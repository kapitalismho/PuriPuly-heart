from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from torch import nn

from experiments.psem_sortformer_adaptation_depth import execution as execution_module
from experiments.psem_sortformer_adaptation_depth import receipts as receipts_module
from experiments.psem_sortformer_adaptation_depth.preflight import canonical_sha256, sha256_file
from experiments.psem_sortformer_adaptation_depth.receipts import (
    CHECKPOINT_IDENTITY,
    _validate_prediction_artifact,
    _validate_runtime_preflight,
    build_data_split_receipt,
    evaluator_reconstruction_contract,
    paired_source_bootstrap_v1,
    recompute_lineage_numeric_evidence,
    validate_cost_receipt,
)
from experiments.psem_sortformer_adaptation_depth.runtime_audit import (
    LEARNING_RATES,
    authorized_module_paths,
    parameter_inventory,
)
from experiments.psem_training_strategy_gate.sampling import DEV_ROLE, EVAL_ROLE, TRAIN_ROLE


@pytest.fixture(autouse=True)
def _registered_execution_records(monkeypatch):
    monkeypatch.setattr(receipts_module, "require_registered_execution", lambda *args: {})


def test_data_split_receipt_proves_exact_counts_hours_and_eval_seal() -> None:
    receipt = build_data_split_receipt()
    assert receipt["passed"]
    assert receipt["counts"][TRAIN_ROLE] == {"AMI": 50, "AliMeeting": 14}
    assert receipt["counts"][DEV_ROLE] == {"AMI": 7, "AliMeeting": 3}
    assert receipt["counts"][EVAL_ROLE] == {"AMI": 11, "AliMeeting": 8}
    assert receipt["fit_roles"] == [TRAIN_ROLE]
    assert receipt["eval_absent_from_sampling_and_overfit"]


def test_evaluator_contract_rehashes_the_issue_99_implementation() -> None:
    receipt = evaluator_reconstruction_contract()
    assert receipt["passed"]
    assert receipt["primary_cell"] == {"threshold": 0.5, "confirmation_ms": 500}
    assert not receipt["eval_threshold_selection_allowed"]
    assert receipt["q8_posterior_sessions"]["sha256"] == (
        "27b7eaaa5c2ee332c3b81c048f8e2666499da9a3a4d7e46e8c994876c6ddcee8"
    )
    assert Path(receipt["q8_posterior_sessions"]["path"]).is_absolute()
    assert all(Path(value["path"]).is_absolute() for value in receipt["artifacts"].values())


def test_lineage_prediction_artifact_requires_an_absolute_external_path(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(Exception, match="absent, mutable, or in-repository"):
        _validate_prediction_artifact(
            {
                "source_id": "source",
                "prediction_artifact": {
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "row_count": 1,
                },
            }
        )


def test_runtime_preflight_must_equal_a_current_exact_rerun(tmp_path, monkeypatch) -> None:
    checks = [
        {"id": check_id, "passed": True, "observed": str(tmp_path / name)}
        for check_id, name in (
            ("runtime.checkpoint_path", "model.nemo"),
            ("runtime.corpus_root", "corpus"),
            ("runtime.reference_root", "reference"),
            ("runtime.output_root", "output"),
            ("runtime.protocol_registry_root", "registry"),
        )
    ]
    receipt = {"mode": "runtime", "ready_for_runtime_audit": True, "checks": checks}
    monkeypatch.setattr(
        receipts_module,
        "build_preflight",
        lambda paths, static_only=False: receipt,
    )
    _validate_runtime_preflight(receipt)
    forged = copy.deepcopy(receipt)
    forged["checks"][0]["passed"] = False
    with pytest.raises(Exception, match="current exact rerun"):
        _validate_runtime_preflight(forged)


def test_lineage_prediction_requires_all_four_stable_slots_alive(tmp_path) -> None:
    output_root = tmp_path / "output"
    path = output_root / "lineage_predictions" / "source.jsonl"
    path.parent.mkdir(parents=True)
    row = {
        "artifact_role": "psem_sortformer_frame_prediction",
        "source_id": "source",
        "source_frame_start_sample": 0,
        "source_frame_end_sample": 1280,
        "model_evidence_frontier_source_sample": 16640,
        "raw_sortformer_activity_logits": [0.0, 0.0, 0.0, 0.0],
        "raw_anchor_present_logit": 0.0,
        "raw_replacement_evidence_logit": 0.0,
        "slot_alive": [True, True, True, False],
        "state_reset": True,
        "oracle_anchor_slot": 0,
        "anchor_episode_id": "episode",
        "artifact_context": "trainable_checkpoint_lineage",
        "split_role": "PSEM-STRATEGY-DEV",
        "arm": "F0-FROZEN-FLOAT",
        "seed": None,
        "source_waveform_sha256": "w" * 64,
        "base_checkpoint_sha256": CHECKPOINT_IDENTITY["sha256"],
        "trained_checkpoint_sha256": None,
        "trained_checkpoint_receipt_sha256": None,
        "runtime_identity_sha256": "r" * 64,
        "candidate_git_head": "a" * 40,
        "candidate_code_identity_sha256": "c" * 64,
        "experiment_output_root": str(output_root.resolve()),
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    descriptor = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "row_count": 1,
    }
    with pytest.raises(Exception, match="prediction row is invalid"):
        _validate_prediction_artifact(
            {
                "source_id": "source",
                "frame_count": 1,
                "first_frame_start_sample": 0,
                "last_frame_end_sample": 1280,
                "first_evidence_frontier_sample": 16640,
                "last_evidence_frontier_sample": 16640,
                "split_role": "PSEM-STRATEGY-DEV",
                "source_waveform_sha256": "w" * 64,
                "runtime_identity_sha256": "r" * 64,
                "candidate_git_head": "a" * 40,
                "candidate_code_identity_sha256": "c" * 64,
                "experiment_output_root": str(output_root.resolve()),
                "prediction_artifact": descriptor,
            }
        )


def test_paired_source_bootstrap_is_recomputed_from_numeric_deltas() -> None:
    receipt = paired_source_bootstrap_v1(
        {"source-a": -1.0, "source-b": 1.0}, seed=107, resamples=2000
    )
    assert receipt["lower"] == -1.0
    assert receipt["upper"] == 1.0
    assert len(receipt["replicate_estimates_sha256"]) == 64


def test_lineage_replays_the_exact_issue_99_q8_scalar_slot(tmp_path, monkeypatch) -> None:
    from experiments.psem_frozen_ceiling_gate import build_ceiling_examples, evaluate_ceiling
    from experiments.psem_sortformer_adaptation_depth import evaluation

    snapshot = SimpleNamespace(
        source_id="source",
        starts=np.asarray([0, 1280], dtype=np.int64),
        ends=np.asarray([1280, 2560], dtype=np.int64),
        probabilities=np.asarray([[0.9, 0.1, 0.1, 0.1], [0.2, 0.8, 0.1, 0.1]], dtype=np.float32),
    )
    monkeypatch.setattr(build_ceiling_examples, "load_sessions", lambda **_: [snapshot])
    monkeypatch.setattr(
        evaluation,
        "_adapt_session",
        lambda *_: (snapshot, np.asarray([0.4, 0.6]), None, None),
    )
    observed = {}

    def session_row(_session, scores, *, condition, **_kwargs):
        observed[condition] = scores.tolist()
        return {
            "metrics": {
                "exclusive_other_contamination_seconds": float(np.sum(scores)),
                "false_cut_count": int(np.sum(scores)),
                "missed_replacement_count": int(np.sum(scores)),
            }
        }

    monkeypatch.setattr(evaluate_ceiling, "session_row", session_row)
    rows = [
        {
            "source_frame_start_sample": index * 1280,
            "source_frame_end_sample": (index + 1) * 1280,
            "model_evidence_frontier_source_sample": index * 1280 + 16640,
            "raw_sortformer_activity_logits": [0.0, 0.0, 0.0, 0.0],
            "slot_alive": [True, True, True, True],
            "state_reset": index == 0,
            "raw_anchor_present_logit": 0.0,
            "raw_replacement_evidence_logit": 0.0,
        }
        for index in range(2)
    ]
    path = tmp_path / "predictions.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    verified_bytes = path.read_bytes()
    descriptor = {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "row_count": 2,
    }
    path.write_text("changed after verification\n", encoding="utf-8")
    recompute_lineage_numeric_evidence(
        [
            {
                "source_id": "source",
                "prediction_artifact": descriptor,
            }
        ],
        {"q8_posterior_sessions": {"sha256": "q8"}},
        artifact_bytes={"source": verified_bytes},
    )
    assert observed["Q8-S-current"] == [0.0, 1.0]


def _bound_staged(payload: dict) -> dict:
    return {**payload, "payload_sha256": canonical_sha256(payload)}


class _CanaryInventoryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sortformer = nn.Module()
        self.sortformer.transformer_encoder = nn.Module()
        self.sortformer.transformer_encoder.layers = nn.ModuleList(
            [nn.Linear(1, 1, bias=False) for _ in range(18)]
        )
        self.sortformer.sortformer_modules = nn.Module()
        self.sortformer.sortformer_modules.first_hidden_to_hidden = nn.Linear(1, 1, bias=False)
        self.sortformer.sortformer_modules.single_hidden_to_spks = nn.Linear(1, 1, bias=False)
        self.psem_head = nn.Linear(1, 1, bias=False)


def _complete_canary_bundle(arm: str) -> dict:
    model = _CanaryInventoryModel()
    inventory = parameter_inventory(model, arm)
    state_dict_rows = [
        {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in model.state_dict().items()
    ]
    graph_payload = {
        "schema_version": 1,
        "artifact_role": "model_graph_receipt",
        "passed": True,
        "temporal_layer_count": 18,
        "hidden_tensor_identity": "sortformer.transformer_encoder.output",
        "hidden_tap_module_type": "test.CanaryTransformer",
        "hidden_dimension": 192,
        "activity_logit_identity": (
            "sortformer.sortformer_modules.single_hidden_to_spks.output_pre_sigmoid"
        ),
        "activity_head_module_type": "test.CanaryActivityHead",
        "runtime_canary_tap_paths": {
            "final_temporal_hidden": "runtime_taps.final_temporal_hidden",
            "speaker_activity_logits": "runtime_taps.speaker_activity_logits",
            "psem_outputs": "psem_head",
        },
        "slot_count": 4,
        "native_frame_ms": 80,
        "streaming_geometry": {
            "chunk_len": 6,
            "chunk_right_context": 7,
            "fifo_len": 188,
            "spkcache_update_period": 144,
            "spkcache_len": 188,
            "chunk_left_context": 1,
        },
        "algorithmic_evidence_delay_ms": 1040,
        "state_reset_policy": "declared_source_or_reset_boundary_only",
        "slot_alive_policy": "issue_99_all_four_stable_columns_alive",
        "executable_graph_sha256": "a" * 64,
        "parameter_schema_sha256": inventory["parameter_schema_sha256"],
        "state_dict_schema_sha256": canonical_sha256(state_dict_rows),
        "executable_module_count": 24,
        "executable_parameter_count": inventory["parameter_count"],
        "executable_state_entry_count": len(state_dict_rows),
    }
    graph = {
        **graph_payload,
        "payload_sha256": canonical_sha256(graph_payload),
    }
    trainable_names = [row["name"] for row in inventory["parameters"] if row["requires_grad"]]
    inventory_sha = canonical_sha256(inventory)
    graph_sha = canonical_sha256(graph)
    gradient_payload = {
        "schema_version": 1,
        "artifact_role": "gradient_canary_receipt",
        "arm": arm,
        "passed": True,
        "input_kind": "raw_16khz_mono_waveform",
        "input_shape": [1, 480000],
        "loss": 1.0,
        "unclipped_gradient_norm": 1.0,
        "clip_norm": 1.0,
        "module_reach_counts": {path: 1 for path in authorized_module_paths(arm)},
        "tap_output_shapes": {
            "final_temporal_hidden": [1, 375, 192],
            "speaker_activity_logits": [1, 375, 4],
            "psem_outputs": [1, 375],
        },
        "raw_waveform_dependence": {path: True for path in authorized_module_paths(arm)},
        "raw_waveform_gradient_nonzero": True,
        "parameter_inventory_sha256": inventory_sha,
        "model_graph_receipt_sha256": graph_sha,
        "parameters": [
            {
                "name": row["name"],
                "expected_trainable": row["requires_grad"],
                "gradient_present": row["requires_grad"],
                "finite": True,
                "nonzero": row["requires_grad"],
            }
            for row in inventory["parameters"]
        ],
    }
    update_payload = {
        "schema_version": 1,
        "artifact_role": "update_canary_receipt",
        "arm": arm,
        "passed": True,
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "learning_rates": LEARNING_RATES,
        "changed_parameters": sorted(trainable_names),
        "frozen_parameters_unchanged": True,
        "all_trainable_parameters_changed": True,
        "parameter_inventory_sha256": inventory_sha,
        "model_graph_receipt_sha256": graph_sha,
    }
    timing_payload = {
        "schema_version": 1,
        "artifact_role": "timing_receipt",
        "passed": True,
        "sample_rate_hz": 16000,
        "native_frame_samples": 1280,
        "algorithmic_evidence_delay_samples": 16640,
        "frame_counts": [375],
        "slot_count": 4,
        "hidden_dimension": 192,
        "lifecycle_fields_binary": True,
        "all_four_stable_slots_alive": True,
        "state_reset_first_frame_only": True,
        "additional_future_context_observed": False,
        "prefix_causality_passed": True,
        "prefix_causality": {
            "passed": True,
            "algorithmic_evidence_delay_samples": 16640,
            "mutation_start_sample": 240000,
            "protected_frame_count": 100,
            "protected_prefix_unchanged": True,
            "suffix_change_observed": True,
        },
        "streaming_cache_integrity_passed": True,
        "streaming_step_count": 1,
        "streaming_trace_sha256": "f" * 64,
    }
    return {
        "gradient_canary_receipt": {
            **gradient_payload,
            "payload_sha256": canonical_sha256(gradient_payload),
        },
        "update_canary_receipt": {
            **update_payload,
            "payload_sha256": canonical_sha256(update_payload),
        },
        "timing_receipt": {
            **timing_payload,
            "payload_sha256": canonical_sha256(timing_payload),
        },
        "parameter_inventory": inventory,
        "model_graph_receipt": graph,
        "conditional_arm_audit_authorization": None,
    }


def test_cost_receipt_recomputes_formula_and_rejects_tampering() -> None:
    receipt = execution_module.build_cost_receipt(
        hourly_price_usd=2.0,
        hourly_price_source="trusted operator quote",
        actual_gpu_seconds=3600.0,
        projected_remaining_gpu_seconds=1800.0,
        command="train-arm H-HEAD",
    )
    assert validate_cost_receipt(receipt) == receipt
    forged = copy.deepcopy(receipt)
    forged["projected_total_cost_usd"] = 2.5
    payload = {key: value for key, value in forged.items() if key != "payload_sha256"}
    forged["payload_sha256"] = canonical_sha256(payload)
    with pytest.raises(Exception, match="formula or USD-30 hard stop"):
        validate_cost_receipt(forged)
