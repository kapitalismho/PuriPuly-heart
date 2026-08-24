from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

SAMPLING_CHECK_IDS = (
    "sampling.manifest_identity",
    "sampling.train_only",
    "sampling.mixture_exact",
    "sampling.shared_across_arms_and_seeds",
    "sampling.loss_weights_complete",
)
AUGMENTATION_CHECK_IDS = (
    "augmentation.recipe_exact",
    "augmentation.label_independent_whole_window",
    "augmentation.family_coverage",
    "augmentation.synthetic_policy",
    "augmentation.manifest_binding",
)
MODEL_GRAPH_CHECK_IDS = (
    "model_graph.raw_waveform_paths_exact",
    "model_graph.wavlm_identity_shared",
    "model_graph.finetune_whitelist_exact",
    "model_graph.scratch_architecture_exact",
    "model_graph.common_output_and_head_exact",
)
PARAMETER_CHECK_IDS = (
    "parameters.inventory_complete",
    "parameters.frozen_wavlm_zero_trainable",
    "parameters.finetune_wavlm_nonzero_and_whitelisted",
    "parameters.scratch_no_pretrained_and_size",
    "parameters.optimizer_coverage_and_learning_rates",
)
GRADIENT_CHECK_IDS = (
    "gradient.frozen_wavlm",
    "gradient.finetune_wavlm",
    "gradient.scratch_psem",
    "gradient.common_losses",
)
WEIGHT_UPDATE_CHECK_IDS = (
    "weight_update.frozen_wavlm_unchanged",
    "weight_update.finetune_wavlm_allowed_only",
    "weight_update.scratch_encoder_changed",
    "weight_update.required_common_head_changed",
)
COMPARABILITY_CHECK_IDS = (
    "comparability.raw_waveform_identical",
    "comparability.targets_identical",
    "comparability.sampling_role_identical",
    "comparability.observed_frontier_identical",
    "comparability.evaluation_reference_identical",
    "comparability.only_encoder_strategy_differs",
)
METRIC_CHECK_IDS = (
    "metric.one_to_one",
    "metric.collar_boundaries",
    "metric.duplicate_suppression",
    "metric.source_hour_denominator",
    "metric.unsnapped_references",
    "metric.full_unique_score_range",
    "metric.shared_evaluator_contract",
)
RUNTIME_CHECK_IDS = {
    "sampling_manifest": SAMPLING_CHECK_IDS,
    "augmentation_manifest": AUGMENTATION_CHECK_IDS,
    "model_graphs": MODEL_GRAPH_CHECK_IDS,
    "parameter_inventory": PARAMETER_CHECK_IDS,
    "gradient_canary": GRADIENT_CHECK_IDS,
    "weight_update_canary": WEIGHT_UPDATE_CHECK_IDS,
    "arm_comparability": COMPARABILITY_CHECK_IDS,
    "metric_contract": METRIC_CHECK_IDS,
}

RUNTIME_ARTIFACT_PATHS = {
    "sampling_manifest": (
        "manifests/sampling_manifest.jsonl",
        "manifests/sampling_summary.json",
    ),
    "augmentation_manifest": ("manifests/augmentation_manifest.json",),
    "model_graphs": ("audits/model_graphs.json",),
    "parameter_inventory": ("audits/parameter_inventory.json",),
    "gradient_canary": ("audits/gradient_canary.json",),
    "weight_update_canary": ("audits/weight_update_canary.json",),
    "arm_comparability": ("audits/arm_comparability.json",),
    "metric_contract": ("audits/metric_contract.json",),
}

RUNTIME_CHECK_ARTIFACT_PATHS = {
    "sampling_manifest": "manifests/sampling_summary.json",
    "augmentation_manifest": "manifests/augmentation_manifest.json",
    "model_graphs": "audits/model_graphs.json",
    "parameter_inventory": "audits/parameter_inventory.json",
    "gradient_canary": "audits/gradient_canary.json",
    "weight_update_canary": "audits/weight_update_canary.json",
    "arm_comparability": "audits/arm_comparability.json",
    "metric_contract": "audits/metric_contract.json",
}

RUNTIME_RECEIPT_ROLES = {
    "sampling_manifest": "psem_sampling_manifest",
    "augmentation_manifest": "psem_augmentation_manifest",
    "model_graphs": "psem_model_graph_receipt",
    "parameter_inventory": "psem_parameter_inventory",
    "gradient_canary": "psem_gradient_canary",
    "weight_update_canary": "psem_weight_update_canary",
    "arm_comparability": "psem_arm_comparability",
    "metric_contract": "psem_metric_contract",
}

RUNTIME_ARTIFACT_ROLES = {
    "sampling_manifest": "psem_sampling_summary",
    "augmentation_manifest": "psem_augmentation_manifest",
    "model_graphs": "psem_model_graphs",
    "parameter_inventory": "psem_parameter_inventory",
    "gradient_canary": "psem_gradient_canary",
    "weight_update_canary": "psem_weight_update_canary",
    "arm_comparability": "psem_arm_comparability",
    "metric_contract": "psem_metric_contract",
}

_ARMS = ("FROZEN-WAVLM", "FINETUNE-WAVLM", "SCRATCH-PSEM")
_EXPECTED_PARAMETER_INVENTORY_SHA256 = {
    "FROZEN-WAVLM": "27cefae3c26d82f3248845b9c29a6b19d2a62ec412ed31baf6755e2f5f76f2ba",
    "FINETUNE-WAVLM": "896c554b4c05d5c224c23f4d4cb797c0f07e7ce4d2d2cf65be5e77ff53c8936f",
    "SCRATCH-PSEM": "4ef2d19eadbe96ecf05481a4c3cbc6ac85cda47b501d4f21c8337b8f0516d1e4",
}
_EXPECTED_COMMON_HEAD = {
    "temporal": {
        "type": "GRU",
        "input_size": 256,
        "hidden_size": 128,
        "layers": 2,
        "bidirectional": True,
        "dropout": 0.1,
        "batch_first": True,
    },
    "handoff_head_linear_shapes": [[1024, 256], [256, 1]],
    "state_head_shape": [256, 3],
    "relation_head_linear_shapes": [[512, 256], [256, 1]],
    "implementation_sha256": "2907e18540d2d09e6d5d06d5d30d877cc6e10542d2eac481215fb1197bbb4a14",
}
_EXPECTED_COMMON_HEAD_SHA256 = "1ca79b3c766dbb522ee20b187b5bb0ae32df2b4093135b161dbfe3b3dd8a6fef"
_EXPECTED_INITIAL_WAVLM_SHA256 = "46b0182d148229e2d12d427b30c7d665507f692bcc9785a69eb4767dd3208e81"
_EXPECTED_WAVLM_CONFIG = {
    "model_type": "wavlm",
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
    "conv_stride": [5, 2, 2, 2, 2, 2, 2],
    "do_stable_layer_norm": False,
    "apply_spec_augment": True,
    "mask_time_prob": 0.05,
    "feat_proj_dropout": 0.1,
    "hidden_dropout": 0.1,
    "attention_dropout": 0.1,
    "layerdrop": 0.05,
    "output_frames": 149,
}
_EXPECTED_LOSS_CONTRACT = {
    "class_weights": {
        "handoff_positive": 2.783557848476163,
        "relation_classes": [0.0706411802779254, 1.9293588197220746],
        "state_classes": [0.8617251996399268, 0.35532956672843413, 1.782945233631639],
    },
    "coefficients": {"handoff": 1.0, "relation": 0.5, "state": 0.5},
    "implementation_sha256": "775614a3b1bc53a5cfe882ec8d4ba1bcaaf20f96f8fd7b47198a6872cf3c5cbb",
}


class RuntimeEvidenceError(RuntimeError):
    pass


def _model_graph_complete(
    artifact: Mapping[str, Any],
    validation_context: Mapping[str, Any] | None,
) -> bool:
    if set(artifact) != {
        "schema_version",
        "artifact_role",
        "canary_source_id",
        "canary_boundary_sample",
        "arms",
        "checks",
    }:
        return False
    if (
        not isinstance(artifact.get("canary_source_id"), str)
        or not artifact["canary_source_id"]
        or not isinstance(artifact.get("canary_boundary_sample"), int)
        or isinstance(artifact["canary_boundary_sample"], bool)
        or artifact["canary_boundary_sample"] < 0
        or artifact["canary_boundary_sample"] % 1600
    ):
        return False
    arms = artifact.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(_ARMS):
        return False
    parameter_arms = None
    if validation_context is not None and "parameter_inventory" in validation_context:
        parameter_artifact = validation_context["parameter_inventory"]
        if not isinstance(parameter_artifact, Mapping):
            return False
        parameter_arms = _parameter_inventory_arms(parameter_artifact)
        if parameter_arms is None:
            return False
    expected_graph_keys = {
        "input",
        "encoder",
        "projection",
        "cell_output_shape",
        "handoff_output_shape",
        "state_output_shape",
        "common_head",
        "losses",
        "optimizer",
    }
    for arm, values in arms.items():
        if not isinstance(values, Mapping) or set(values) != expected_graph_keys:
            return False
        if values["input"] != {
            "kind": "raw_waveform",
            "sample_rate_hz": 16000,
            "samples": 48000,
            "upstream_transform": "psem-waveform-augmentation-v1",
            "cached_feature_inputs": [],
        }:
            return False
        expected_projection_input = 320 if arm == "SCRATCH-PSEM" else 768
        if values["projection"] != {
            "input_dimension": expected_projection_input,
            "output_dimension": 256,
            "normalization_shape": [256],
        }:
            return False
        if (
            values["cell_output_shape"] != [1, 30, 256]
            or values["handoff_output_shape"] != [1]
            or values["state_output_shape"] != [1, 30, 3]
            or values["common_head"] != _EXPECTED_COMMON_HEAD
            or values["losses"] != _EXPECTED_LOSS_CONTRACT
        ):
            return False
        if parameter_arms is not None and values["optimizer"] != parameter_arms[arm]["optimizer"]:
            return False
        encoder = values["encoder"]
        if not isinstance(encoder, Mapping):
            return False
        if arm != "SCRATCH-PSEM":
            if set(encoder) != {
                "type",
                "model_id",
                "revision",
                "local_model_root",
                "config",
                "trainable_parameter_names",
                "initial_parameter_sha256",
                "execution_mode",
                "wavlm_training",
            }:
                return False
            root = Path(str(encoder.get("local_model_root", "")))
            if (
                encoder.get("type") != "WavLMCellEncoder"
                or encoder.get("model_id") != "wavlm-base-plus"
                or encoder.get("revision") != "4c66d4806a428f2e922ccfa1a962776e232d487b"
                or not root.is_absolute()
                or root.name != encoder["revision"]
                or root.parent.name != encoder["model_id"]
                or encoder.get("config") != _EXPECTED_WAVLM_CONFIG
                or encoder.get("initial_parameter_sha256") != _EXPECTED_INITIAL_WAVLM_SHA256
                or encoder.get("wavlm_training") is not False
                or encoder.get("execution_mode")
                != (
                    "eval_with_gradients"
                    if arm == "FINETUNE-WAVLM"
                    else "eval_without_gradients"
                )
                or not isinstance(encoder.get("trainable_parameter_names"), list)
            ):
                return False
            if parameter_arms is not None:
                expected_names = [
                    row["name"]
                    for row in parameter_arms[arm]["parameters"]
                    if row["trainable"] and row["name"].startswith("encoder.wavlm.")
                ]
                if encoder["trainable_parameter_names"] != expected_names:
                    return False
        else:
            expected_blocks = [
                {
                    "width": 320,
                    "expansion": 2,
                    "kernel": 5,
                    "dilation": dilation,
                    "groups": 640,
                }
                for dilation in [1, 2, 4, 8, 16, 1, 2, 4]
            ]
            if (
                set(encoder)
                != {
                    "type",
                    "frontend",
                    "stem",
                    "blocks",
                    "final_normalization_channels",
                    "pretrained_artifacts",
                    "implementation_sha256",
                }
                or encoder.get("type") != "ScratchCellEncoder"
                or encoder.get("frontend")
                != {
                    "sample_rate_hz": 16000,
                    "n_fft": 400,
                    "win_length": 400,
                    "hop_length": 160,
                    "center": False,
                    "power": 2.0,
                    "mel_bins": 64,
                    "mel_norm": "slaney",
                    "mel_scale": "slaney",
                }
                or encoder.get("stem")
                != {"input_channels": 64, "output_channels": 320, "kernel": 5}
                or encoder.get("blocks") != expected_blocks
                or encoder.get("final_normalization_channels") != 320
                or encoder.get("pretrained_artifacts") != []
                or encoder.get("implementation_sha256")
                != _EXPECTED_COMMON_HEAD["implementation_sha256"]
            ):
                return False
    return True


def _sampling_summary_complete(artifact: Mapping[str, Any]) -> bool:
    expected_families = {
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
    pool_counts = artifact.get("pool_counts")
    return (
        set(artifact)
        == {
            "schema_version",
            "artifact_role",
            "manifest_path",
            "manifest_sha256",
            "row_count",
            "epoch_count",
            "windows_per_epoch",
            "effective_batch_size",
            "minimum_valid_counts_per_batch",
            "sampling_role_counts",
            "topology_family_counts",
            "pool_counts",
            "source_count",
            "arms",
            "seeds",
            "shared_center_and_augmentation_manifest",
            "topology_family_mapping",
            "eval_source_count",
            "loss_weights",
            "target_class_counts",
            "checks",
        }
        and isinstance(artifact.get("manifest_path"), str)
        and Path(artifact["manifest_path"]).is_absolute()
        and _sha256(artifact.get("manifest_sha256"))
        and artifact.get("epoch_count") == 20
        and artifact.get("windows_per_epoch") == 4096
        and isinstance(pool_counts, Mapping)
        and set(pool_counts) == expected_families
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in pool_counts.values()
        )
        and set(artifact.get("topology_family_counts", {})) == expected_families
    )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _strict_positive_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _strict_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _gradient_rows_consistent(
    value: Any,
    *,
    expected_names: list[str] | None = None,
) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "parameter_tensor_count",
        "none_count",
        "nonfinite_count",
        "nonzero_tensor_count",
        "aggregate_norm",
        "parameters",
    }:
        return False
    rows = value.get("parameters")
    if not isinstance(rows, list):
        return False
    names = [row.get("name") for row in rows if isinstance(row, Mapping)]
    if len(names) != len(rows) or len(names) != len(set(names)):
        return False
    if expected_names is not None and names != expected_names:
        return False
    none_count = 0
    nonfinite_count = 0
    nonzero_tensor_count = 0
    total_squared = 0.0
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "name",
            "present",
            "finite",
            "norm",
            "sha256",
        }:
            return False
        if not isinstance(row.get("name"), str) or not row["name"]:
            return False
        if type(row.get("present")) is not bool:
            return False
        if row["present"] is False:
            if (
                row.get("finite") is not None
                or row.get("norm") is not None
                or row.get("sha256") is not None
            ):
                return False
            none_count += 1
            continue
        if type(row.get("finite")) is not bool or not _sha256(row.get("sha256")):
            return False
        if row["finite"] is False:
            if row.get("norm") is not None:
                return False
            nonfinite_count += 1
            continue
        norm = row.get("norm")
        if (
            not isinstance(norm, (int, float))
            or isinstance(norm, bool)
            or not math.isfinite(norm)
            or norm < 0.0
        ):
            return False
        total_squared += norm * norm
        nonzero_tensor_count += int(norm > 0.0)
    aggregate_norm = value.get("aggregate_norm")
    return (
        value.get("parameter_tensor_count") == len(rows)
        and value.get("none_count") == none_count
        and value.get("nonfinite_count") == nonfinite_count
        and value.get("nonzero_tensor_count") == nonzero_tensor_count
        and isinstance(aggregate_norm, (int, float))
        and not isinstance(aggregate_norm, bool)
        and math.isfinite(aggregate_norm)
        and math.isclose(aggregate_norm, math.sqrt(total_squared), rel_tol=1e-12, abs_tol=1e-12)
    )


def _strict_active_gradient(value: Any) -> bool:
    return (
        _gradient_rows_consistent(value)
        and value["parameter_tensor_count"] > 0
        and value.get("none_count") == 0
        and value.get("nonfinite_count") == 0
        and value.get("nonzero_tensor_count") == value["parameter_tensor_count"]
        and isinstance(value.get("aggregate_norm"), (int, float))
        and not isinstance(value.get("aggregate_norm"), bool)
        and math.isfinite(value["aggregate_norm"])
        and value["aggregate_norm"] > 0.0
    )


def _strict_missing_gradient(value: Any) -> bool:
    return (
        _gradient_rows_consistent(value)
        and value["parameter_tensor_count"] > 0
        and value["none_count"] == value["parameter_tensor_count"]
        and value["nonfinite_count"] == 0
        and value["nonzero_tensor_count"] == 0
        and value["aggregate_norm"] == 0.0
    )


def _parameter_inventory_arms(artifact: Any) -> Mapping[str, Any] | None:
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("schema_version") != 1
        or artifact.get("artifact_role") != "psem_parameter_inventory"
        or set(artifact) != {"schema_version", "artifact_role", "arms", "checks"}
    ):
        return None
    arms = artifact.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(_ARMS):
        return None
    if any(
        not isinstance(arms[arm], Mapping)
        or _canonical_sha256(arms[arm]) != _EXPECTED_PARAMETER_INVENTORY_SHA256[arm]
        for arm in _ARMS
    ):
        return None
    return arms


def _wavlm_parameter_allowed(name: str) -> bool:
    return name.startswith(
        tuple(f"encoder.wavlm.encoder.layers.{index}." for index in range(8, 12))
    ) or name.startswith("encoder.wavlm.encoder.layer_norm.")


def _parameter_owner(name: str) -> str | None:
    if name.startswith("encoder.wavlm."):
        return "wavlm_allowed" if _wavlm_parameter_allowed(name) else "wavlm_frozen"
    if name.startswith("encoder."):
        return "scratch_encoder"
    if name.startswith("projection."):
        return "projection"
    if name.startswith("head."):
        return "common_head"
    return None


def _gradient_expected_groups(inventory: Mapping[str, Any]) -> dict[str, list[str]]:
    names = [row["name"] for row in inventory["parameters"]]
    wavlm = [name for name in names if name.startswith("encoder.wavlm.")]
    return {
        "projection": [name for name in names if name.startswith("projection.")],
        "temporal": [name for name in names if name.startswith("head.temporal.")],
        "handoff_head": [name for name in names if name.startswith("head.handoff_head.")],
        "state_head": [name for name in names if name.startswith("head.state_head.")],
        "relation_head": [name for name in names if name.startswith("head.relation_head.")],
        "wavlm_all": wavlm,
        "wavlm_allowed": [name for name in wavlm if _wavlm_parameter_allowed(name)],
        "wavlm_frozen": [name for name in wavlm if not _wavlm_parameter_allowed(name)],
        "scratch_encoder": [
            name
            for name in names
            if name.startswith("encoder.") and not name.startswith("encoder.wavlm.")
        ],
        "finetuned_final_normalization": [
            name for name in names if name.startswith("encoder.wavlm.encoder.layer_norm.")
        ],
    }


def _gradient_artifact_complete(
    artifact: Mapping[str, Any],
    validation_context: Mapping[str, Any] | None,
) -> bool:
    linked = validation_context.get("parameter_inventory") if validation_context else None
    inventories = _parameter_inventory_arms(linked)
    arms = artifact.get("arms")
    if inventories is None or not isinstance(arms, Mapping) or set(arms) != set(_ARMS):
        return False
    expected_keys = {
        "losses",
        "projection",
        "temporal",
        "handoff_head",
        "state_head",
        "relation_head",
        "wavlm_all",
        "wavlm_allowed",
        "wavlm_frozen",
        "scratch_encoder",
        "finetuned_blocks",
        "finetuned_final_normalization",
    }
    for arm in _ARMS:
        values = arms.get(arm)
        if not isinstance(values, Mapping) or set(values) != expected_keys:
            return False
        losses = values.get("losses")
        if not isinstance(losses, Mapping) or set(losses) != {
            "total",
            "handoff",
            "state",
            "relation",
            "handoff_valid_count",
            "state_valid_count",
            "relation_valid_count",
        }:
            return False
        groups = _gradient_expected_groups(inventories[arm])
        if any(
            not _gradient_rows_consistent(values.get(group), expected_names=names)
            for group, names in groups.items()
        ):
            return False
        blocks = values.get("finetuned_blocks")
        if not isinstance(blocks, Mapping) or set(blocks) != {"8", "9", "10", "11"}:
            return False
        for index in range(8, 12):
            expected = [
                row["name"]
                for row in inventories[arm]["parameters"]
                if row["name"].startswith(f"encoder.wavlm.encoder.layers.{index}.")
            ]
            if not _gradient_rows_consistent(blocks[str(index)], expected_names=expected):
                return False
        wavlm_rows = {row["name"]: row for row in values["wavlm_all"]["parameters"]}
        for group in ("wavlm_allowed", "wavlm_frozen", "finetuned_final_normalization"):
            if values[group]["parameters"] != [wavlm_rows[name] for name in groups[group]]:
                return False
        for index in range(8, 12):
            block_names = [row["name"] for row in blocks[str(index)]["parameters"]]
            if blocks[str(index)]["parameters"] != [wavlm_rows[name] for name in block_names]:
                return False
    return True


def _update_artifact_complete(
    artifact: Mapping[str, Any],
    validation_context: Mapping[str, Any] | None,
) -> bool:
    linked = validation_context.get("parameter_inventory") if validation_context else None
    inventories = _parameter_inventory_arms(linked)
    arms = artifact.get("arms")
    if inventories is None or not isinstance(arms, Mapping) or set(arms) != set(_ARMS):
        return False
    for arm in _ARMS:
        values = arms.get(arm)
        rows = values.get("parameters") if isinstance(values, Mapping) else None
        inventory_rows = inventories[arm]["parameters"]
        if (
            not isinstance(values, Mapping)
            or set(values) != {"parameters"}
            or not isinstance(rows, list)
        ):
            return False
        if [row.get("name") for row in rows if isinstance(row, Mapping)] != [
            row["name"] for row in inventory_rows
        ]:
            return False
        for row, inventory in zip(rows, inventory_rows, strict=True):
            if not isinstance(row, Mapping) or set(row) != {
                "name",
                "owner",
                "trainable",
                "optimizer_group",
                "before_sha256",
                "after_sha256",
                "changed",
            }:
                return False
            if (
                row["owner"] != _parameter_owner(row["name"])
                or row["trainable"] is not inventory["trainable"]
                or row["optimizer_group"] != inventory["optimizer_group"]
                or not _sha256(row["before_sha256"])
                or not _sha256(row["after_sha256"])
                or type(row["changed"]) is not bool
                or row["changed"] is not (row["before_sha256"] != row["after_sha256"])
            ):
                return False
    return True


def _semantic_pass(
    receipt_name: str,
    check_id: str,
    artifact: Mapping[str, Any],
    validation_context: Mapping[str, Any] | None,
) -> bool:
    arms = artifact.get("arms")
    if receipt_name == "sampling_manifest":
        if check_id == "sampling.manifest_identity":
            return (
                artifact.get("row_count") == 81_920
                and isinstance(artifact.get("manifest_sha256"), str)
                and len(artifact["manifest_sha256"]) == 64
            )
        if check_id == "sampling.train_only":
            return artifact.get("eval_source_count") == 0 and artifact.get("source_count") == 64
        if check_id == "sampling.mixture_exact":
            return (
                artifact.get("sampling_role_counts")
                == {
                    "handoff_positive": 20_480,
                    "source_time_uniform": 40_960,
                    "topology_hard_negative": 20_480,
                }
                and set(artifact.get("topology_family_counts", {}))
                == {
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
                and artifact.get("topology_family_mapping")
                == {
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
                }
            )
        if check_id == "sampling.shared_across_arms_and_seeds":
            return (
                artifact.get("shared_center_and_augmentation_manifest") is True
                and artifact.get("arms") == list(_ARMS)
                and artifact.get("seeds") == [7301, 7302]
            )
        if check_id == "sampling.loss_weights_complete":
            weights = artifact.get("loss_weights")
            counts = artifact.get("target_class_counts")
            batch_counts = artifact.get("minimum_valid_counts_per_batch")
            return (
                isinstance(weights, Mapping)
                and set(weights) == {"handoff_positive", "state_classes", "relation_classes"}
                and isinstance(weights.get("state_classes"), list)
                and len(weights["state_classes"]) == 3
                and isinstance(weights.get("relation_classes"), list)
                and len(weights["relation_classes"]) == 2
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and value > 0
                    for value in (
                        weights.get("handoff_positive"),
                        *(weights.get("state_classes") or ()),
                        *(weights.get("relation_classes") or ()),
                    )
                )
                and isinstance(counts, Mapping)
                and set(counts) == {"handoff", "state", "relation"}
                and all(isinstance(counts.get(key), Mapping) for key in counts)
                and set(counts["handoff"]) == {"0", "1"}
                and set(counts["state"]) == {"0", "1", "2"}
                and set(counts["relation"]) == {"0", "1"}
                and all(
                    isinstance(count, int) and not isinstance(count, bool) and count > 0
                    for values in counts.values()
                    if isinstance(values, Mapping)
                    for count in values.values()
                )
                and artifact.get("effective_batch_size") == 4
                and isinstance(batch_counts, Mapping)
                and set(batch_counts) == {"handoff", "state", "relation"}
                and all(_strict_positive_int(value) for value in batch_counts.values())
            )
    if receipt_name == "augmentation_manifest":
        if check_id == "augmentation.recipe_exact":
            return artifact.get(
                "recipe_version"
            ) == "psem-waveform-augmentation-v1" and artifact.get("families") == [
                "global_gain",
                "additive_non_speech_noise",
                "light_reverberation",
                "band_limitation",
                "codec_simulation",
            ]
        if check_id == "augmentation.label_independent_whole_window":
            return (
                artifact.get("label_fields_consulted") == []
                and artifact.get("whole_window_consistency") is True
                and artifact.get("decision_source") == "sampling_manifest.row_id_only"
            )
        if check_id == "augmentation.family_coverage":
            counts = artifact.get("enabled_counts")
            return (
                artifact.get("decision_count") == 81_920
                and isinstance(counts, Mapping)
                and all(isinstance(value, int) and value > 0 for value in counts.values())
            )
        if check_id == "augmentation.synthetic_policy":
            return (
                artifact.get("synthetic_manifest") is None
                and artifact.get("synthetic_optimizer_batch_fraction") == 0.0
                and artifact.get("natural_training_coverage_satisfied") is True
            )
        if check_id == "augmentation.manifest_binding":
            return (
                isinstance(artifact.get("sampling_manifest_sha256"), str)
                and len(artifact["sampling_manifest_sha256"]) == 64
            )
    if receipt_name == "metric_contract":
        details = artifact.get("details")
        if not isinstance(details, Mapping):
            return False
        if check_id == "metric.one_to_one":
            rows = details.get("nearest_remap", {}).get("rows", [])
            replacement_rows = details.get("nearest_replacement", {}).get("rows", [])
            return (
                bool(rows)
                and bool(replacement_rows)
                and [
                    [match.get("prediction_source_sample"), match.get("reference_source_sample")]
                    for match in rows[0].get("matches", {}).get("100", [])
                ]
                == [[1600, 0], [3200, 1600]]
                and [
                    [match.get("prediction_source_sample"), match.get("reference_source_sample")]
                    for match in replacement_rows[0].get("matches", {}).get("100", [])
                ]
                == [[1600, 1600]]
            )
        if check_id == "metric.collar_boundaries":
            return details.get("collar_canaries") == {
                str(collar): {
                    "exact": 1,
                    "outside_by_one_sample": 0,
                    "outside_false_events_per_hour": 1.0,
                }
                for collar in (100, 250, 500)
            }
        if check_id == "metric.duplicate_suppression":
            diagnostics = details.get("sub_resolution_transitions")
            return (
                details.get("eventizer_retained_samples") == [0, 6400]
                and isinstance(diagnostics, list)
                and len(diagnostics) == 1
                and diagnostics[0].get("artifact_role") == "sub_resolution_transition"
            )
        if check_id == "metric.source_hour_denominator":
            return all(
                values.get("outside_false_events_per_hour") == 1.0
                for values in details.get("collar_canaries", {}).values()
            )
        if check_id == "metric.unsnapped_references":
            rows = details.get("unsnapped_curve", {}).get("rows", [])
            return (
                bool(rows)
                and rows[-1].get("metrics", {}).get("100", {}).get("true_positive_count") == 1
            )
        if check_id == "metric.full_unique_score_range":
            frontier = details.get("synthetic_frontier", {})
            equal = details.get("equal_score_curve", {})
            equal_rows = equal.get("rows", [])
            return (
                frontier.get("score_thresholds") == [0.1, 0.5, 0.9]
                and frontier.get("false_events_per_hour_ceiling") is None
                and bool(equal_rows)
                and equal_rows[-1].get("prediction_count") == 2
            )
        if check_id == "metric.shared_evaluator_contract":
            thresholds = details.get("shared_thresholds")
            return (
                thresholds == [0.1, 0.2, 0.8, 0.9]
                and all(
                    curve.get("score_thresholds") == thresholds
                    for curve in details.get("shared_output_curves", [])
                )
                and len(details.get("shared_output_curves", [])) == 2
            )
        return False
    if not isinstance(arms, Mapping) or set(arms) != set(_ARMS):
        return False
    if receipt_name == "model_graphs":
        if check_id == "model_graph.raw_waveform_paths_exact":
            return all(
                values.get("input", {}).get("kind") == "raw_waveform"
                and values["input"].get("sample_rate_hz") == 16000
                and values["input"].get("samples") == 48000
                and values["input"].get("cached_feature_inputs") == []
                for values in arms.values()
            )
        if check_id == "model_graph.wavlm_identity_shared":
            frozen = arms["FROZEN-WAVLM"]["encoder"]
            fine = arms["FINETUNE-WAVLM"]["encoder"]
            return (
                frozen.get("model_id") == fine.get("model_id") == "wavlm-base-plus"
                and frozen.get("revision")
                == fine.get("revision")
                == "4c66d4806a428f2e922ccfa1a962776e232d487b"
                and frozen.get("config") == fine.get("config")
                and frozen.get("initial_parameter_sha256") == fine.get("initial_parameter_sha256")
            )
        if check_id == "model_graph.finetune_whitelist_exact":
            frozen_names = arms["FROZEN-WAVLM"]["encoder"].get("trainable_parameter_names")
            fine_names = arms["FINETUNE-WAVLM"]["encoder"].get("trainable_parameter_names")
            return (
                frozen_names == []
                and bool(fine_names)
                and all(
                    name.startswith(
                        tuple(f"encoder.wavlm.encoder.layers.{index}." for index in range(8, 12))
                    )
                    or name.startswith("encoder.wavlm.encoder.layer_norm.")
                    for name in fine_names
                )
            )
        if check_id == "model_graph.scratch_architecture_exact":
            scratch = arms["SCRATCH-PSEM"]["encoder"]
            return (
                scratch.get("type") == "ScratchCellEncoder"
                and scratch.get("pretrained_artifacts") == []
                and scratch.get("frontend", {}).get("mel_bins") == 64
                and [block.get("dilation") for block in scratch.get("blocks", [])]
                == [1, 2, 4, 8, 16, 1, 2, 4]
            )
        if check_id == "model_graph.common_output_and_head_exact":
            return all(
                values.get("cell_output_shape") == [1, 30, 256]
                and values.get("handoff_output_shape") == [1]
                and values.get("state_output_shape") == [1, 30, 3]
                and values.get("common_head") == _EXPECTED_COMMON_HEAD
                and values.get("losses") == _EXPECTED_LOSS_CONTRACT
                for values in arms.values()
            )
    if receipt_name == "parameter_inventory":
        if check_id == "parameters.inventory_complete":
            return _parameter_inventory_arms(artifact) is not None
        if check_id == "parameters.frozen_wavlm_zero_trainable":
            return arms["FROZEN-WAVLM"].get("trainable_wavlm_parameters") == 0
        if check_id == "parameters.finetune_wavlm_nonzero_and_whitelisted":
            values = arms["FINETUNE-WAVLM"]
            return values.get("trainable_wavlm_parameters", 0) > 0 and all(
                not row.get("trainable")
                or not row.get("name", "").startswith("encoder.wavlm.")
                or row["name"].startswith(
                    tuple(f"encoder.wavlm.encoder.layers.{index}." for index in range(8, 12))
                )
                or row["name"].startswith("encoder.wavlm.encoder.layer_norm.")
                for row in values.get("parameters", [])
            )
        if check_id == "parameters.scratch_no_pretrained_and_size":
            values = arms["SCRATCH-PSEM"]
            return 5_000_000 <= values.get("total_parameters", 0) <= 10_000_000 and all(
                "wavlm" not in row.get("name", "").lower() for row in values.get("parameters", [])
            )
        if check_id == "parameters.optimizer_coverage_and_learning_rates":
            expected_groups = {
                "FROZEN-WAVLM": {"common_head_and_projection": 1e-3},
                "FINETUNE-WAVLM": {
                    "common_head_and_projection": 1e-3,
                    "finetuned_wavlm": 1e-5,
                },
                "SCRATCH-PSEM": {
                    "common_head_and_projection": 1e-3,
                    "scratch_encoder": 3e-4,
                },
            }
            return all(
                values.get("optimizer", {}).get("type") == "torch.optim.adamw.AdamW"
                and values["optimizer"].get("defaults", {}).get("weight_decay") == 1e-4
                and {
                    group.get("name"): group.get("learning_rate")
                    for group in values["optimizer"].get("groups", [])
                }
                == expected_groups[arm]
                and sorted(
                    name
                    for group in values["optimizer"].get("groups", [])
                    for name in group.get("parameter_names", [])
                )
                == sorted(
                    row.get("name") for row in values.get("parameters", []) if row.get("trainable")
                )
                for arm, values in arms.items()
            )
    if receipt_name == "gradient_canary":
        if check_id == "gradient.frozen_wavlm":
            values = arms["FROZEN-WAVLM"].get("wavlm_all", {})
            return _strict_missing_gradient(values)
        if check_id == "gradient.finetune_wavlm":
            values = arms["FINETUNE-WAVLM"]
            frozen = values.get("wavlm_frozen", {})
            return (
                all(
                    _strict_active_gradient(row)
                    for row in values.get("finetuned_blocks", {}).values()
                )
                and _strict_active_gradient(values.get("finetuned_final_normalization"))
                and _strict_missing_gradient(frozen)
            )
        if check_id == "gradient.scratch_psem":
            return _strict_active_gradient(arms["SCRATCH-PSEM"].get("scratch_encoder"))
        if check_id == "gradient.common_losses":
            return all(
                all(
                    _strict_active_gradient(values.get(group))
                    for group in (
                        "projection",
                        "temporal",
                        "handoff_head",
                        "state_head",
                        "relation_head",
                    )
                )
                and all(
                    _strict_positive_number(values.get("losses", {}).get(name))
                    for name in ("total", "handoff", "state", "relation")
                )
                and all(
                    _strict_positive_int(values.get("losses", {}).get(name))
                    for name in (
                        "handoff_valid_count",
                        "state_valid_count",
                        "relation_valid_count",
                    )
                )
                for values in arms.values()
            )
    if receipt_name == "weight_update_canary":
        rows = {arm: values.get("parameters", []) for arm, values in arms.items()}
        if check_id == "weight_update.frozen_wavlm_unchanged":
            selected = [
                row
                for row in rows["FROZEN-WAVLM"]
                if row.get("name", "").startswith("encoder.wavlm.")
            ]
            return bool(selected) and all(row.get("changed") is False for row in selected)
        if check_id == "weight_update.finetune_wavlm_allowed_only":
            selected = [
                row
                for row in rows["FINETUNE-WAVLM"]
                if row.get("name", "").startswith("encoder.wavlm.")
            ]
            return bool(selected) and all(
                row.get("changed") is row.get("trainable") for row in selected
            )
        if check_id == "weight_update.scratch_encoder_changed":
            selected = [
                row for row in rows["SCRATCH-PSEM"] if row.get("name", "").startswith("encoder.")
            ]
            return bool(selected) and all(row.get("changed") is True for row in selected)
        if check_id == "weight_update.required_common_head_changed":
            return all(
                all(
                    row.get("changed") is True
                    for row in values
                    if row.get("owner") in {"projection", "common_head"} and row.get("trainable")
                )
                for values in rows.values()
            )
    if receipt_name == "arm_comparability":
        expected = validation_context.get("comparability") if validation_context else None
        if not isinstance(expected, Mapping):
            return False

        def exact(*keys: str) -> bool:
            return all(
                all(values.get(key) == expected.get(key) for key in keys)
                for values in arms.values()
            )

        if check_id == "comparability.raw_waveform_identical":
            return exact("raw_waveform_tensor", "augmented_waveform_tensor")
        if check_id == "comparability.targets_identical":
            return exact("target_sha256", "target_batch_tensors")
        if check_id == "comparability.sampling_role_identical":
            return exact(
                "row_id",
                "center_id",
                "source_id",
                "source_waveform_manifest_sha256",
                "augmentation_decision_sha256",
                "sampling_role",
            )
        if check_id == "comparability.observed_frontier_identical":
            return exact("observed_frontier_sample")
        if check_id == "comparability.evaluation_reference_identical":
            return exact("unsnapped_handoff_event_samples")
        if check_id == "comparability.only_encoder_strategy_differs":
            return (
                all(
                    values.get("common_head_sha256") == _EXPECTED_COMMON_HEAD_SHA256
                    and values.get("common_head_contract") == _EXPECTED_COMMON_HEAD
                    and values.get("loss_contract") == _EXPECTED_LOSS_CONTRACT
                    for values in arms.values()
                )
                and arms["FROZEN-WAVLM"].get("initial_wavlm_sha256")
                == arms["FINETUNE-WAVLM"].get("initial_wavlm_sha256")
                == _EXPECTED_INITIAL_WAVLM_SHA256
                and arms["SCRATCH-PSEM"].get("initial_wavlm_sha256") is None
            )
    return False


def runtime_artifact_checks(
    receipt_name: str,
    artifact: Mapping[str, Any],
    *,
    validation_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if receipt_name not in RUNTIME_CHECK_IDS:
        raise RuntimeEvidenceError("unknown runtime receipt")
    if artifact.get("artifact_role") != RUNTIME_ARTIFACT_ROLES[receipt_name]:
        raise RuntimeEvidenceError("runtime evidence artifact role differs from its contract")
    if artifact.get("schema_version") != 1:
        raise RuntimeEvidenceError("runtime evidence schema version differs from its contract")
    if receipt_name == "sampling_manifest" and not _sampling_summary_complete(artifact):
        raise RuntimeEvidenceError("sampling summary schema differs from the frozen manifest")
    if receipt_name in {
        "model_graphs",
        "parameter_inventory",
        "gradient_canary",
        "weight_update_canary",
        "arm_comparability",
    }:
        arms = artifact.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != set(_ARMS):
            raise RuntimeEvidenceError("runtime evidence arm inventory differs from its contract")
    if receipt_name == "model_graphs" and not _model_graph_complete(
        artifact, validation_context
    ):
        raise RuntimeEvidenceError("model graph differs from the exact constructed models")
    if receipt_name == "parameter_inventory" and _parameter_inventory_arms(artifact) is None:
        raise RuntimeEvidenceError("parameter inventory differs from the exact constructed models")
    if receipt_name == "gradient_canary" and not _gradient_artifact_complete(
        artifact, validation_context
    ):
        raise RuntimeEvidenceError(
            "gradient evidence is not complete for the exact parameter inventory"
        )
    if receipt_name == "weight_update_canary" and not _update_artifact_complete(
        artifact, validation_context
    ):
        raise RuntimeEvidenceError(
            "weight-update evidence is not complete and internally consistent"
        )
    rows = artifact.get("checks")
    if (
        not isinstance(rows, list)
        or tuple(row.get("id") for row in rows if isinstance(row, Mapping))
        != RUNTIME_CHECK_IDS[receipt_name]
    ):
        raise RuntimeEvidenceError("runtime evidence check inventory differs from its contract")
    result = []
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"id", "passed", "expected", "observed"}
            or not isinstance(row.get("id"), str)
            or type(row.get("passed")) is not bool
            or row["passed"]
            is not _semantic_pass(receipt_name, row["id"], artifact, validation_context)
        ):
            raise RuntimeEvidenceError("runtime evidence check contradicts artifact semantics")
        result.append(dict(row))
    return result
