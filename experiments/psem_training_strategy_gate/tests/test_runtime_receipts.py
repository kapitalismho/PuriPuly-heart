from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.psem_training_strategy_gate import receipts
from experiments.psem_training_strategy_gate import runtime_contract as runtime_contract_module
from experiments.psem_training_strategy_gate.audit import _active_gradient, _metric_contract_audit
from experiments.psem_training_strategy_gate.preflight import (
    BINDING_KEYS,
    _runtime_receipt_valid,
    canonical_sha256,
    sha256_file,
)
from experiments.psem_training_strategy_gate.receipts import (
    ReceiptContractError,
    runtime_receipt,
    write_runtime_receipt,
)
from experiments.psem_training_strategy_gate.runtime_contract import (
    RUNTIME_CHECK_IDS,
    RuntimeEvidenceError,
    runtime_artifact_checks,
)


def _check_rows(receipt_name: str) -> list[dict]:
    return [
        {"id": check_id, "passed": True, "expected": None, "observed": None}
        for check_id in RUNTIME_CHECK_IDS[receipt_name]
    ]


def _sampling_artifact() -> dict:
    topology_families = {
        "clean_direct_different_speaker_handoff",
        "silence_gap_different_speaker_handoff",
        "overlap_takeover",
        "stable_singleton_continuation",
        "same_speaker_silence_gap_resume",
        "overlap_return",
        "overlap_continuation",
        "silence_continuation",
        "source_time_uniform",
    }
    return {
        "schema_version": 1,
        "artifact_role": "psem_sampling_summary",
        "manifest_path": str((Path.cwd() / "sampling_manifest.jsonl").resolve()),
        "row_count": 81_920,
        "manifest_sha256": "a" * 64,
        "epoch_count": 20,
        "windows_per_epoch": 4096,
        "eval_source_count": 0,
        "source_count": 64,
        "sampling_role_counts": {
            "handoff_positive": 20_480,
            "source_time_uniform": 40_960,
            "topology_hard_negative": 20_480,
        },
        "topology_family_counts": {name: 1 for name in topology_families},
        "pool_counts": {name: 1 for name in topology_families},
        "topology_family_mapping": {
            "handoff_positive": {
                "clean_direct_different_speaker_handoff": "clean_direct_different_speaker_handoff",
                "micro_gap_different_speaker_handoff": "silence_gap_different_speaker_handoff",
                "micro_overlap_takeover": "overlap_takeover",
                "overlap_gap_takeover": "overlap_takeover",
                "overlap_takeover": "overlap_takeover",
                "silence_gap_different_speaker_handoff": "silence_gap_different_speaker_handoff",
            },
            "topology_hard_negative": {
                "micro_gap_same_speaker_resume": "same_speaker_silence_gap_resume",
                "micro_overlap_return": "overlap_return",
                "overlap_gap_return": "overlap_return",
                "overlap_return": "overlap_return",
                "same_speaker_silence_gap_resume": "same_speaker_silence_gap_resume",
            },
        },
        "shared_center_and_augmentation_manifest": True,
        "arms": ["FROZEN-WAVLM", "FINETUNE-WAVLM", "SCRATCH-PSEM"],
        "seeds": [7301, 7302],
        "loss_weights": {
            "handoff_positive": 1.0,
            "state_classes": [1.0, 1.0, 1.0],
            "relation_classes": [1.0, 1.0],
        },
        "target_class_counts": {
            "handoff": {"0": 1, "1": 1},
            "state": {"0": 1, "1": 1, "2": 1},
            "relation": {"0": 1, "1": 1},
        },
        "effective_batch_size": 4,
        "minimum_valid_counts_per_batch": {"handoff": 1, "state": 120, "relation": 9},
        "checks": _check_rows("sampling_manifest"),
    }


def test_sampling_receipt_requires_positive_support_in_every_official_batch() -> None:
    artifact = _sampling_artifact()
    assert all(row["passed"] for row in runtime_artifact_checks("sampling_manifest", artifact))
    for field, value in (
        ("effective_batch_size", 8),
        ("minimum_valid_counts_per_batch", {"handoff": 1, "state": 120, "relation": 0}),
        ("minimum_valid_counts_per_batch", {"handoff": 1, "state": 120}),
    ):
        forged = deepcopy(artifact)
        forged[field] = value
        with pytest.raises(RuntimeEvidenceError, match="contradicts artifact semantics"):
            runtime_artifact_checks("sampling_manifest", forged)


def _inventory_row(name: str, trainable: bool, optimizer_group: str | None, numel: int) -> dict:
    return {
        "name": name,
        "shape": [numel],
        "numel": numel,
        "owner_module": name.rsplit(".", 1)[0],
        "trainable": trainable,
        "optimizer_group": optimizer_group,
        "learning_rate": 1e-3 if optimizer_group is not None else None,
    }


def _parameter_artifact(monkeypatch: pytest.MonkeyPatch) -> dict:
    common_names = (
        "projection.weight",
        "head.temporal.weight",
        "head.handoff_head.weight",
        "head.state_head.weight",
        "head.relation_head.weight",
    )
    wavlm_names = (
        "encoder.wavlm.encoder.layers.0.weight",
        "encoder.wavlm.encoder.layers.8.weight",
        "encoder.wavlm.encoder.layers.8.bias",
        "encoder.wavlm.encoder.layers.9.weight",
        "encoder.wavlm.encoder.layers.10.weight",
        "encoder.wavlm.encoder.layers.11.weight",
        "encoder.wavlm.encoder.layer_norm.weight",
    )
    arms = {}
    for arm in ("FROZEN-WAVLM", "FINETUNE-WAVLM"):
        rows = []
        for name in wavlm_names:
            allowed = runtime_contract_module._wavlm_parameter_allowed(name)
            trainable = arm == "FINETUNE-WAVLM" and allowed
            rows.append(
                _inventory_row(name, trainable, "finetuned_wavlm" if trainable else None, 1)
            )
        rows.extend(
            _inventory_row(name, True, "common_head_and_projection", 1) for name in common_names
        )
        groups = [
            {
                "name": "common_head_and_projection",
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "parameter_names": list(common_names),
                "parameter_count": len(common_names),
            }
        ]
        if arm == "FINETUNE-WAVLM":
            groups.append(
                {
                    "name": "finetuned_wavlm",
                    "learning_rate": 1e-5,
                    "weight_decay": 1e-4,
                    "parameter_names": [
                        name
                        for name in wavlm_names
                        if runtime_contract_module._wavlm_parameter_allowed(name)
                    ],
                    "parameter_count": 6,
                }
            )
        arms[arm] = {
            "arm": arm,
            "parameters": rows,
            "total_parameters": sum(row["numel"] for row in rows),
            "trainable_parameters": sum(row["numel"] for row in rows if row["trainable"]),
            "trainable_wavlm_parameters": sum(
                row["numel"]
                for row in rows
                if row["trainable"] and row["name"].startswith("encoder.wavlm.")
            ),
            "optimizer_groups": [
                {
                    "name": group["name"],
                    "learning_rate": group["learning_rate"],
                    "parameter_count": group["parameter_count"],
                }
                for group in groups
            ],
            "optimizer": {
                "type": "torch.optim.adamw.AdamW",
                "defaults": {"learning_rate": 1e-3, "weight_decay": 1e-4},
                "groups": groups,
            },
        }
    scratch_rows = [
        _inventory_row("encoder.stem.weight", True, "scratch_encoder", 5_999_995),
        *(_inventory_row(name, True, "common_head_and_projection", 1) for name in common_names),
    ]
    scratch_groups = [
        {
            "name": "common_head_and_projection",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "parameter_names": list(common_names),
            "parameter_count": len(common_names),
        },
        {
            "name": "scratch_encoder",
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "parameter_names": ["encoder.stem.weight"],
            "parameter_count": 5_999_995,
        },
    ]
    arms["SCRATCH-PSEM"] = {
        "arm": "SCRATCH-PSEM",
        "parameters": scratch_rows,
        "total_parameters": 6_000_000,
        "trainable_parameters": 6_000_000,
        "trainable_wavlm_parameters": 0,
        "optimizer_groups": [
            {
                "name": group["name"],
                "learning_rate": group["learning_rate"],
                "parameter_count": group["parameter_count"],
            }
            for group in scratch_groups
        ],
        "optimizer": {
            "type": "torch.optim.adamw.AdamW",
            "defaults": {"learning_rate": 1e-3, "weight_decay": 1e-4},
            "groups": scratch_groups,
        },
    }
    for arm, values in arms.items():
        monkeypatch.setitem(
            runtime_contract_module._EXPECTED_PARAMETER_INVENTORY_SHA256,
            arm,
            runtime_contract_module._canonical_sha256(values),
        )
    return {
        "schema_version": 1,
        "artifact_role": "psem_parameter_inventory",
        "arms": arms,
        "checks": _check_rows("parameter_inventory"),
    }


def _gradient_stats(names: list[str], state: str) -> dict:
    rows = []
    for name in names:
        present = state != "missing"
        rows.append(
            {
                "name": name,
                "present": present,
                "finite": True if present else None,
                "norm": 1.0 if present else None,
                "sha256": "a" * 64 if present else None,
            }
        )
    return _gradient_stats_from_rows(rows)


def _gradient_stats_from_rows(rows: list[dict]) -> dict:
    finite_norms = [row["norm"] for row in rows if row["finite"] is True]
    return {
        "parameter_tensor_count": len(rows),
        "none_count": sum(row["present"] is False for row in rows),
        "nonfinite_count": sum(row["finite"] is False for row in rows),
        "nonzero_tensor_count": sum(norm > 0.0 for norm in finite_norms),
        "aggregate_norm": sum(norm * norm for norm in finite_norms) ** 0.5,
        "parameters": rows,
    }


def _binding() -> dict:
    value = {key: "1" * 64 for key in BINDING_KEYS}
    value["experiment_id"] = "psem_training_strategy_gate_v1"
    value["git_commit"] = "a" * 40
    return value


def test_runtime_receipt_requires_exact_role_check_inventory_and_artifacts(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(receipts, "current_binding", _binding)
    output_root = tmp_path.resolve()
    artifact_path = output_root / "audits" / "metric_contract.json"
    artifact_path.parent.mkdir(parents=True)
    rows, details = _metric_contract_audit()
    artifact_value = {
        "schema_version": 1,
        "artifact_role": "psem_metric_contract",
        "checks": rows,
        "details": details,
    }
    artifact_path.write_text(json.dumps(artifact_value) + "\n", encoding="utf-8")
    artifact = {
        "path": str(artifact_path),
        "sha256": sha256_file(artifact_path),
        "canonical_sha256": canonical_sha256(artifact_value),
        "size_bytes": artifact_path.stat().st_size,
    }
    receipt = runtime_receipt(
        "metric_contract",
        "psem_metric_contract",
        details={"artifacts": [artifact]},
    )
    passed, observed = _runtime_receipt_valid(
        receipt,
        "metric_contract",
        "psem_metric_contract",
        _binding(),
        output_root,
    )
    assert passed is True
    assert observed["checks_valid"] is True
    forged_binding = json.loads(json.dumps(receipt))
    forged_binding["binding"]["git_commit"] = "b" * 40
    forged_payload = dict(forged_binding)
    forged_payload.pop("payload_sha256")
    forged_binding["payload_sha256"] = canonical_sha256(forged_payload)
    with pytest.raises(ReceiptContractError, match="authoritative"):
        write_runtime_receipt(output_root / "preflight" / "metric_contract.json", forged_binding)
    with pytest.raises(ReceiptContractError, match="identity and details"):
        runtime_receipt("metric_contract", "psem_metric_contract", details={})
    forged_details = json.loads(json.dumps(receipt))
    forged_details["details"]["shared_thresholds"] = ["forged"]
    forged_payload = dict(forged_details)
    forged_payload.pop("payload_sha256")
    forged_details["payload_sha256"] = canonical_sha256(forged_payload)
    passed, observed = _runtime_receipt_valid(
        forged_details,
        "metric_contract",
        "psem_metric_contract",
        _binding(),
        output_root,
    )
    assert passed is False
    assert observed["details_schema_valid"] is False
    forged_artifact = json.loads(json.dumps(artifact_value))
    forged_artifact["details"]["shared_thresholds"] = [0.1, 0.9]
    with pytest.raises(RuntimeEvidenceError, match="contradicts artifact semantics"):
        runtime_artifact_checks("metric_contract", forged_artifact)
    receipt["checks"][0]["id"] = "forged"
    passed, observed = _runtime_receipt_valid(
        receipt,
        "metric_contract",
        "psem_metric_contract",
        _binding(),
        output_root,
    )
    assert passed is False
    assert observed["checks_valid"] is False
    receipt = runtime_receipt(
        "metric_contract",
        "psem_metric_contract",
        details={"artifacts": [artifact]},
    )
    receipt["checks"][0]["expected"] = "forged"
    payload = dict(receipt)
    payload.pop("payload_sha256")
    receipt["payload_sha256"] = canonical_sha256(payload)
    passed, observed = _runtime_receipt_valid(
        receipt,
        "metric_contract",
        "psem_metric_contract",
        _binding(),
        output_root,
    )
    assert passed is False
    assert observed["semantic_checks_valid"] is False
    receipt = runtime_receipt(
        "metric_contract",
        "psem_metric_contract",
        details={"artifacts": [artifact]},
    )
    artifact_path.write_text("{}\n", encoding="utf-8")
    passed, observed = _runtime_receipt_valid(
        receipt,
        "metric_contract",
        "psem_metric_contract",
        _binding(),
        output_root,
    )
    assert passed is False
    assert observed["artifacts_valid"] is False


def test_parameter_inventory_rejects_self_consistent_row_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _parameter_artifact(monkeypatch)
    assert all(row["passed"] for row in runtime_artifact_checks("parameter_inventory", artifact))
    forged = deepcopy(artifact)
    removed = forged["arms"]["FROZEN-WAVLM"]["parameters"].pop(0)
    forged["arms"]["FROZEN-WAVLM"]["total_parameters"] -= removed["numel"]
    with pytest.raises(RuntimeEvidenceError, match="exact constructed models"):
        runtime_artifact_checks("parameter_inventory", forged)


def test_gradient_and_update_artifacts_require_exact_parameter_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter_artifact = _parameter_artifact(monkeypatch)
    gradient_arms = {}
    for arm, inventory in parameter_artifact["arms"].items():
        groups = runtime_contract_module._gradient_expected_groups(inventory)
        allowed_stats = _gradient_stats(
            groups["wavlm_allowed"],
            "active" if arm == "FINETUNE-WAVLM" else "missing",
        )
        frozen_stats = _gradient_stats(groups["wavlm_frozen"], "missing")
        wavlm_by_name = {
            row["name"]: row for row in [*allowed_stats["parameters"], *frozen_stats["parameters"]]
        }
        wavlm_all_stats = _gradient_stats_from_rows(
            [wavlm_by_name[name] for name in groups["wavlm_all"]]
        )
        values = {
            "losses": {
                "total": 1.0,
                "handoff": 1.0,
                "state": 1.0,
                "relation": 1.0,
                "handoff_valid_count": 1,
                "state_valid_count": 1,
                "relation_valid_count": 1,
            },
            "projection": _gradient_stats(groups["projection"], "active"),
            "temporal": _gradient_stats(groups["temporal"], "active"),
            "handoff_head": _gradient_stats(groups["handoff_head"], "active"),
            "state_head": _gradient_stats(groups["state_head"], "active"),
            "relation_head": _gradient_stats(groups["relation_head"], "active"),
            "wavlm_all": wavlm_all_stats,
            "wavlm_allowed": allowed_stats,
            "wavlm_frozen": frozen_stats,
            "scratch_encoder": _gradient_stats(
                groups["scratch_encoder"],
                "active" if arm == "SCRATCH-PSEM" else "missing",
            ),
            "finetuned_blocks": {
                str(index): _gradient_stats(
                    [
                        row["name"]
                        for row in inventory["parameters"]
                        if row["name"].startswith(f"encoder.wavlm.encoder.layers.{index}.")
                    ],
                    "active" if arm == "FINETUNE-WAVLM" else "missing",
                )
                for index in range(8, 12)
            },
            "finetuned_final_normalization": _gradient_stats(
                groups["finetuned_final_normalization"],
                "active" if arm == "FINETUNE-WAVLM" else "missing",
            ),
        }
        gradient_arms[arm] = values
    gradient_artifact = {
        "schema_version": 1,
        "artifact_role": "psem_gradient_canary",
        "real_batch": {},
        "arms": gradient_arms,
        "checks": _check_rows("gradient_canary"),
    }
    context = {"parameter_inventory": parameter_artifact}
    assert all(
        row["passed"]
        for row in runtime_artifact_checks(
            "gradient_canary",
            gradient_artifact,
            validation_context=context,
        )
    )
    for count in (
        "handoff_valid_count",
        "state_valid_count",
        "relation_valid_count",
    ):
        forged_counts = deepcopy(gradient_artifact)
        for values in forged_counts["arms"].values():
            values["losses"][count] = 0
        with pytest.raises(RuntimeEvidenceError, match="contradicts artifact semantics"):
            runtime_artifact_checks(
                "gradient_canary",
                forged_counts,
                validation_context=context,
            )
    forged_gradient = deepcopy(gradient_artifact)
    rows = forged_gradient["arms"]["FINETUNE-WAVLM"]["finetuned_blocks"]["8"]["parameters"][1:]
    forged_gradient["arms"]["FINETUNE-WAVLM"]["finetuned_blocks"]["8"] = _gradient_stats(
        [row["name"] for row in rows], "active"
    )
    with pytest.raises(RuntimeEvidenceError, match="exact parameter inventory"):
        runtime_artifact_checks(
            "gradient_canary",
            forged_gradient,
            validation_context=context,
        )

    update_arms = {}
    for arm, inventory in parameter_artifact["arms"].items():
        update_rows = []
        for row in inventory["parameters"]:
            name = row["name"]
            changed = row["trainable"]
            update_rows.append(
                {
                    "name": name,
                    "owner": runtime_contract_module._parameter_owner(name),
                    "trainable": row["trainable"],
                    "optimizer_group": row["optimizer_group"],
                    "before_sha256": "a" * 64,
                    "after_sha256": "b" * 64 if changed else "a" * 64,
                    "changed": changed,
                }
            )
        update_arms[arm] = {"parameters": update_rows}
    update_artifact = {
        "schema_version": 1,
        "artifact_role": "psem_weight_update_canary",
        "arms": update_arms,
        "checks": _check_rows("weight_update_canary"),
    }
    assert all(
        row["passed"]
        for row in runtime_artifact_checks(
            "weight_update_canary",
            update_artifact,
            validation_context=context,
        )
    )
    forged_update = deepcopy(update_artifact)
    forged_update["arms"]["FROZEN-WAVLM"]["parameters"].pop(0)
    with pytest.raises(RuntimeEvidenceError, match="complete and internally consistent"):
        runtime_artifact_checks(
            "weight_update_canary",
            forged_update,
            validation_context=context,
        )
    forged_hashes = deepcopy(update_artifact)
    projection = next(
        row
        for row in forged_hashes["arms"]["FROZEN-WAVLM"]["parameters"]
        if row["name"].startswith("projection.")
    )
    projection["before_sha256"] = projection["after_sha256"]
    with pytest.raises(RuntimeEvidenceError, match="complete and internally consistent"):
        runtime_artifact_checks(
            "weight_update_canary",
            forged_hashes,
            validation_context=context,
        )


def test_model_graph_requires_exact_shared_head_dimensions() -> None:
    common_head = deepcopy(runtime_contract_module._EXPECTED_COMMON_HEAD)
    shared_losses = deepcopy(runtime_contract_module._EXPECTED_LOSS_CONTRACT)
    frozen_encoder = {
        "type": "WavLMCellEncoder",
        "model_id": "wavlm-base-plus",
        "revision": "4c66d4806a428f2e922ccfa1a962776e232d487b",
        "local_model_root": str(
            (Path.cwd() / "wavlm-base-plus" / "4c66d4806a428f2e922ccfa1a962776e232d487b").resolve()
        ),
        "config": deepcopy(runtime_contract_module._EXPECTED_WAVLM_CONFIG),
        "initial_parameter_sha256": runtime_contract_module._EXPECTED_INITIAL_WAVLM_SHA256,
        "trainable_parameter_names": [],
        "execution_mode": "eval_without_gradients",
        "wavlm_training": False,
    }
    fine_encoder = {
        **deepcopy(frozen_encoder),
        "trainable_parameter_names": ["encoder.wavlm.encoder.layers.8.weight"],
        "execution_mode": "eval_with_gradients",
    }
    scratch_encoder = {
        "type": "ScratchCellEncoder",
        "pretrained_artifacts": [],
        "frontend": {
            "sample_rate_hz": 16000,
            "n_fft": 400,
            "win_length": 400,
            "hop_length": 160,
            "center": False,
            "power": 2.0,
            "mel_bins": 64,
            "mel_norm": "slaney",
            "mel_scale": "slaney",
        },
        "stem": {"input_channels": 64, "output_channels": 320, "kernel": 5},
        "blocks": [
            {"width": 320, "expansion": 2, "kernel": 5, "dilation": value, "groups": 640}
            for value in [1, 2, 4, 8, 16, 1, 2, 4]
        ],
        "final_normalization_channels": 320,
        "implementation_sha256": common_head["implementation_sha256"],
    }
    input_contract = {
        "kind": "raw_waveform",
        "sample_rate_hz": 16000,
        "samples": 48000,
        "upstream_transform": "psem-waveform-augmentation-v1",
        "cached_feature_inputs": [],
    }
    shared = {
        "input": input_contract,
        "cell_output_shape": [1, 30, 256],
        "handoff_output_shape": [1],
        "state_output_shape": [1, 30, 3],
        "common_head": common_head,
        "losses": shared_losses,
        "optimizer": {},
    }
    artifact = {
        "schema_version": 1,
        "artifact_role": "psem_model_graphs",
        "canary_source_id": "source",
        "canary_boundary_sample": 1600,
        "arms": {
            "FROZEN-WAVLM": {
                **deepcopy(shared),
                "encoder": frozen_encoder,
                "projection": {"input_dimension": 768, "output_dimension": 256, "normalization_shape": [256]},
            },
            "FINETUNE-WAVLM": {
                **deepcopy(shared),
                "encoder": fine_encoder,
                "projection": {"input_dimension": 768, "output_dimension": 256, "normalization_shape": [256]},
            },
            "SCRATCH-PSEM": {
                **deepcopy(shared),
                "encoder": scratch_encoder,
                "projection": {"input_dimension": 320, "output_dimension": 256, "normalization_shape": [256]},
            },
        },
        "checks": _check_rows("model_graphs"),
    }
    assert all(row["passed"] for row in runtime_artifact_checks("model_graphs", artifact))
    sparse = deepcopy(artifact)
    sparse["arms"]["FROZEN-WAVLM"]["encoder"]["config"] = {}
    sparse["arms"]["FINETUNE-WAVLM"]["encoder"]["config"] = {}
    with pytest.raises(RuntimeEvidenceError, match="model graph"):
        runtime_artifact_checks("model_graphs", sparse)
    invalid_hash = deepcopy(artifact)
    invalid_hash["arms"]["FROZEN-WAVLM"]["encoder"]["initial_parameter_sha256"] = "0" * 64
    invalid_hash["arms"]["FINETUNE-WAVLM"]["encoder"]["initial_parameter_sha256"] = "0" * 64
    with pytest.raises(RuntimeEvidenceError, match="model graph"):
        runtime_artifact_checks("model_graphs", invalid_hash)
    forged = deepcopy(artifact)
    for values in forged["arms"].values():
        values["common_head"]["temporal"]["hidden_size"] = 1
    with pytest.raises(RuntimeEvidenceError, match="model graph"):
        runtime_artifact_checks("model_graphs", forged)
    for field, value in (
        ("handoff_output_shape", [99]),
        ("state_output_shape", [99]),
    ):
        forged_shapes = deepcopy(artifact)
        for values in forged_shapes["arms"].values():
            values[field] = value
        with pytest.raises(RuntimeEvidenceError, match="model graph"):
            runtime_artifact_checks("model_graphs", forged_shapes)
    for field, value in (
        ("coefficients", {"handoff": 99.0, "relation": 0.5, "state": 0.5}),
        ("implementation_sha256", "c" * 64),
    ):
        forged_losses = deepcopy(artifact)
        for values in forged_losses["arms"].values():
            values["losses"][field] = value
        with pytest.raises(RuntimeEvidenceError, match="model graph"):
            runtime_artifact_checks("model_graphs", forged_losses)


def test_comparability_requires_independent_provenance() -> None:
    expected = {
        "row_id": "row-1",
        "center_id": "source-1:16000",
        "source_id": "source-1",
        "source_waveform_manifest_sha256": "a" * 64,
        "augmentation_decision_sha256": "b" * 64,
        "sampling_role": "handoff_positive",
        "raw_waveform_tensor": {"dtype": "torch.float32", "shape": [48000], "sha256": "c" * 64},
        "augmented_waveform_tensor": {
            "dtype": "torch.float32",
            "shape": [48000],
            "sha256": "d" * 64,
        },
        "target_sha256": "e" * 64,
        "target_batch_tensors": {"handoff": "f" * 64},
        "observed_frontier_sample": 32000,
        "unsnapped_handoff_event_samples": [15920],
    }
    arms = {}
    for arm in ("FROZEN-WAVLM", "FINETUNE-WAVLM", "SCRATCH-PSEM"):
        arms[arm] = {
            **deepcopy(expected),
            "common_head_sha256": runtime_contract_module._EXPECTED_COMMON_HEAD_SHA256,
            "common_head_contract": deepcopy(runtime_contract_module._EXPECTED_COMMON_HEAD),
            "loss_contract": deepcopy(runtime_contract_module._EXPECTED_LOSS_CONTRACT),
            "initial_wavlm_sha256": (
                None
                if arm == "SCRATCH-PSEM"
                else runtime_contract_module._EXPECTED_INITIAL_WAVLM_SHA256
            ),
        }
    artifact = {
        "schema_version": 1,
        "artifact_role": "psem_arm_comparability",
        "row_id": expected["row_id"],
        "source_id": expected["source_id"],
        "boundary_sample": 16000,
        "arms": arms,
        "checks": _check_rows("arm_comparability"),
    }
    context = {"comparability": expected}
    assert all(
        row["passed"]
        for row in runtime_artifact_checks(
            "arm_comparability",
            artifact,
            validation_context=context,
        )
    )
    forged = deepcopy(artifact)
    for values in forged["arms"].values():
        values["row_id"] = "forged-row"
        values["raw_waveform_tensor"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeEvidenceError, match="contradicts artifact semantics"):
        runtime_artifact_checks(
            "arm_comparability",
            forged,
            validation_context=context,
        )


def test_metric_contract_audit_closes_every_declared_check() -> None:
    checks, details = _metric_contract_audit()
    assert tuple(row["id"] for row in checks) == RUNTIME_CHECK_IDS["metric_contract"]
    assert all(row["passed"] for row in checks)
    assert details["false_events_per_hour_ceiling"] is None
    assert details["thresholds"] == "complete_shared_union_of_unique_scores"


def test_gradient_canary_requires_every_parameter_tensor_to_be_active() -> None:
    complete = {
        "parameter_tensor_count": 2,
        "none_count": 0,
        "nonfinite_count": 0,
        "nonzero_tensor_count": 2,
        "aggregate_norm": 1.0,
    }
    assert _active_gradient(complete) is True
    assert _active_gradient({**complete, "none_count": 1, "nonzero_tensor_count": 1}) is False
    assert _active_gradient({**complete, "nonzero_tensor_count": 1}) is False
