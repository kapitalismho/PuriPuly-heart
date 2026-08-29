from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from experiments.psem_sortformer_adaptation_depth.parameter_policy import (
    ARMS,
    ParameterPolicyError,
    apply_parameter_policy,
    audit_parameter_graph,
    is_activity_head,
    should_train,
    temporal_layer_index,
)

SAMPLE_RATE_HZ = 16000
WINDOW_SAMPLES = 480000
LEARNING_RATES = {
    "psem_head": 1e-3,
    "sortformer_activity_head": 1e-4,
    "temporal_transformer": 1e-5,
}
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
LOW_LATENCY_STREAMING = {
    "chunk_len": 6,
    "chunk_right_context": 7,
    "fifo_len": 188,
    "spkcache_update_period": 144,
    "spkcache_len": 188,
    "chunk_left_context": 1,
}
STREAMING_SUBSAMPLING_FACTOR = 8


class RuntimeAuditError(RuntimeError):
    pass


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _parameter_group(name: str) -> str | None:
    if name.startswith("psem_head."):
        return "psem_head"
    if is_activity_head(name):
        return "sortformer_activity_head"
    if temporal_layer_index(name) is not None:
        return "temporal_transformer"
    return None


def parameter_inventory(model: nn.Module, arm: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise RuntimeAuditError(f"unknown arm: {arm}")
    policy = apply_parameter_policy(model, arm)
    rows = []
    trainable_counts: Counter[str] = Counter()
    for name, parameter in model.named_parameters():
        expected = should_train(name, arm)
        group = _parameter_group(name) if expected else None
        if parameter.requires_grad != expected:
            raise RuntimeAuditError("parameter trainability differs from the exact policy")
        if expected and group is None:
            raise RuntimeAuditError(f"trainable parameter has no optimizer group: {name}")
        if group is not None:
            trainable_counts[group] += parameter.numel()
        rows.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "numel": parameter.numel(),
                "dtype": str(parameter.dtype),
                "requires_grad": parameter.requires_grad,
                "optimizer_group": group,
                "learning_rate": LEARNING_RATES[group] if group is not None else None,
            }
        )
    expected_groups = {
        group for group in LEARNING_RATES if any(row["optimizer_group"] == group for row in rows)
    }
    if arm != "F0-FROZEN-FLOAT" and not expected_groups:
        raise RuntimeAuditError("trainable arm has no optimizer parameters")
    parameter_schema = [
        {"name": row["name"], "shape": row["shape"], "dtype": row["dtype"]} for row in rows
    ]
    payload = {
        "schema_version": 1,
        "artifact_role": "parameter_inventory",
        "arm": arm,
        "parameters": rows,
        "parameter_count": len(rows),
        "total_parameter_count": sum(row["numel"] for row in rows),
        "trainable_parameter_count": sum(row["numel"] for row in rows if row["requires_grad"]),
        "trainable_count_by_group": dict(sorted(trainable_counts.items())),
        "parameter_schema_sha256": _canonical_sha256(parameter_schema),
        "policy": policy,
    }
    return {**payload, "payload_sha256": _canonical_sha256(payload)}


def parameter_inventory_runtime_passed(
    receipt: Mapping[str, Any],
    arm: str,
    *,
    model_graph_receipt: Mapping[str, Any],
) -> bool:
    expected_keys = {
        "schema_version",
        "artifact_role",
        "arm",
        "parameters",
        "parameter_count",
        "total_parameter_count",
        "trainable_parameter_count",
        "trainable_count_by_group",
        "parameter_schema_sha256",
        "policy",
        "payload_sha256",
    }
    if set(receipt) != expected_keys:
        return False
    payload = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    rows = receipt.get("parameters")
    if (
        arm not in ARMS
        or receipt.get("schema_version") != 1
        or receipt.get("artifact_role") != "parameter_inventory"
        or receipt.get("arm") != arm
        or receipt.get("payload_sha256") != _canonical_sha256(payload)
        or not isinstance(rows, list)
        or not rows
    ):
        return False
    row_keys = {
        "name",
        "shape",
        "numel",
        "dtype",
        "requires_grad",
        "optimizer_group",
        "learning_rate",
    }
    names: list[str] = []
    parameter_schema = []
    total_parameter_count = 0
    trainable_parameter_count = 0
    trainable_counts: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != row_keys:
            return False
        name = row.get("name")
        shape = row.get("shape")
        numel = row.get("numel")
        dtype = row.get("dtype")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(shape, list)
            or any(type(value) is not int or value < 0 for value in shape)
            or type(numel) is not int
            or numel != math.prod(shape)
            or not isinstance(dtype, str)
            or not dtype.startswith("torch.")
        ):
            return False
        expected_trainable = should_train(name, arm)
        expected_group = _parameter_group(name) if expected_trainable else None
        if (
            row.get("requires_grad") is not expected_trainable
            or (expected_trainable and expected_group is None)
            or row.get("optimizer_group") != expected_group
            or row.get("learning_rate")
            != (LEARNING_RATES[expected_group] if expected_group is not None else None)
        ):
            return False
        names.append(name)
        parameter_schema.append({"name": name, "shape": shape, "dtype": dtype})
        total_parameter_count += numel
        if expected_trainable:
            trainable_parameter_count += numel
            trainable_counts[expected_group] += numel
    if len(names) != len(set(names)):
        return False
    try:
        audited_policy = audit_parameter_graph(names)
    except ParameterPolicyError:
        return False
    expected_policy = {
        **audited_policy,
        "arm": arm,
        "trainable": [name for name in names if should_train(name, arm)],
    }
    if (
        receipt.get("parameter_count") != len(rows)
        or receipt.get("total_parameter_count") != total_parameter_count
        or receipt.get("trainable_parameter_count") != trainable_parameter_count
        or receipt.get("trainable_count_by_group") != dict(sorted(trainable_counts.items()))
        or receipt.get("parameter_schema_sha256") != _canonical_sha256(parameter_schema)
        or receipt.get("policy") != expected_policy
    ):
        return False
    return bool(
        model_graph_runtime_passed(model_graph_receipt)
        and model_graph_receipt.get("parameter_schema_sha256")
        == receipt.get("parameter_schema_sha256")
        and model_graph_receipt.get("executable_parameter_count") == len(rows)
    )


def model_graph_runtime_passed(receipt: Mapping[str, Any]) -> bool:
    expected_keys = {
        "schema_version",
        "artifact_role",
        "passed",
        "temporal_layer_count",
        "hidden_tensor_identity",
        "hidden_tap_module_type",
        "hidden_dimension",
        "activity_logit_identity",
        "activity_head_module_type",
        "runtime_canary_tap_paths",
        "slot_count",
        "native_frame_ms",
        "streaming_geometry",
        "algorithmic_evidence_delay_ms",
        "state_reset_policy",
        "slot_alive_policy",
        "executable_graph_sha256",
        "parameter_schema_sha256",
        "state_dict_schema_sha256",
        "executable_module_count",
        "executable_parameter_count",
        "executable_state_entry_count",
        "payload_sha256",
    }
    if set(receipt) != expected_keys:
        return False
    payload = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    return bool(
        receipt.get("schema_version") == 1
        and receipt.get("artifact_role") == "model_graph_receipt"
        and receipt.get("passed") is True
        and receipt.get("payload_sha256") == _canonical_sha256(payload)
        and receipt.get("temporal_layer_count") == 18
        and receipt.get("hidden_tensor_identity") == "sortformer.transformer_encoder.output"
        and isinstance(receipt.get("hidden_tap_module_type"), str)
        and bool(receipt.get("hidden_tap_module_type"))
        and receipt.get("hidden_dimension") == 192
        and receipt.get("activity_logit_identity")
        == "sortformer.sortformer_modules.single_hidden_to_spks.output_pre_sigmoid"
        and isinstance(receipt.get("activity_head_module_type"), str)
        and bool(receipt.get("activity_head_module_type"))
        and receipt.get("runtime_canary_tap_paths")
        == {
            "final_temporal_hidden": "runtime_taps.final_temporal_hidden",
            "speaker_activity_logits": "runtime_taps.speaker_activity_logits",
            "psem_outputs": "psem_head",
        }
        and receipt.get("slot_count") == 4
        and receipt.get("native_frame_ms") == 80
        and receipt.get("streaming_geometry") == LOW_LATENCY_STREAMING
        and receipt.get("algorithmic_evidence_delay_ms") == 1040
        and receipt.get("state_reset_policy") == "declared_source_or_reset_boundary_only"
        and receipt.get("slot_alive_policy") == "issue_99_all_four_stable_columns_alive"
        and _is_sha256(receipt.get("executable_graph_sha256"))
        and _is_sha256(receipt.get("parameter_schema_sha256"))
        and _is_sha256(receipt.get("state_dict_schema_sha256"))
        and type(receipt.get("executable_module_count")) is int
        and receipt["executable_module_count"] > 0
        and type(receipt.get("executable_parameter_count")) is int
        and receipt["executable_parameter_count"] > 0
        and type(receipt.get("executable_state_entry_count")) is int
        and receipt["executable_state_entry_count"] >= receipt["executable_parameter_count"]
    )


def write_parameter_inventory(model: nn.Module, arm: str, output_path: Path) -> dict[str, Any]:
    import json

    receipt = parameter_inventory(model, arm)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def build_optimizer(model: nn.Module, arm: str) -> torch.optim.AdamW:
    inventory = parameter_inventory(model, arm)
    if arm == "F0-FROZEN-FLOAT":
        raise RuntimeAuditError("the frozen arm has no optimizer")
    groups = []
    named = dict(model.named_parameters())
    for group_name, learning_rate in LEARNING_RATES.items():
        parameters = [
            named[row["name"]]
            for row in inventory["parameters"]
            if row["optimizer_group"] == group_name
        ]
        if parameters:
            groups.append(
                {
                    "params": parameters,
                    "lr": learning_rate,
                    "weight_decay": WEIGHT_DECAY,
                    "group_name": group_name,
                }
            )
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def authorized_module_paths(arm: str) -> tuple[str, ...]:
    if arm == "F0-FROZEN-FLOAT":
        return ()
    values = ["psem_head"]
    if arm == "T2-TOP":
        values.extend(
            [
                "sortformer.transformer_encoder.layers.16",
                "sortformer.transformer_encoder.layers.17",
            ]
        )
    if arm == "TA-ALL-TEMPORAL":
        values.extend(f"sortformer.transformer_encoder.layers.{index}" for index in range(18))
    if arm in {"T2-TOP", "TA-ALL-TEMPORAL"}:
        values.extend(
            [
                "sortformer.sortformer_modules.first_hidden_to_hidden",
                "sortformer.sortformer_modules.single_hidden_to_spks",
            ]
        )
    return tuple(values)


def gradient_canary_runtime_passed(
    receipt: Mapping[str, Any],
    arm: str,
    parameter_inventory_receipt: Mapping[str, Any],
    model_graph_receipt: Mapping[str, Any],
) -> bool:
    expected_paths = set(authorized_module_paths(arm))
    dependence = receipt.get("raw_waveform_dependence")
    reach_counts = receipt.get("module_reach_counts")
    payload = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    rows = receipt.get("parameters")
    if not isinstance(rows, list) or not rows:
        return False
    names = [row.get("name") for row in rows if isinstance(row, Mapping)]
    if len(names) != len(rows) or any(not isinstance(name, str) for name in names):
        return False
    if len(set(names)) != len(names):
        return False
    if any(row.get("expected_trainable") is not should_train(row["name"], arm) for row in rows):
        return False
    if not parameter_inventory_runtime_passed(
        parameter_inventory_receipt,
        arm,
        model_graph_receipt=model_graph_receipt,
    ):
        return False
    inventory_rows = parameter_inventory_receipt["parameters"]
    inventory_by_name = {row["name"]: row for row in inventory_rows}
    if set(names) != set(inventory_by_name):
        return False
    for row in rows:
        inventory_row = inventory_by_name.get(row["name"])
        if not isinstance(inventory_row, Mapping) or row.get(
            "expected_trainable"
        ) is not inventory_row.get("requires_grad"):
            return False
    expected_trainable = {row["name"] for row in rows if row.get("expected_trainable") is True}
    if (
        not expected_trainable
        or any(
            not any(name.startswith(f"{path}.") for name in expected_trainable)
            for path in expected_paths
        )
        or any(
            row.get("finite") is not True
            or (
                row.get("expected_trainable") is True
                and (row.get("gradient_present") is not True or row.get("nonzero") is not True)
            )
            or (row.get("expected_trainable") is not True and row.get("nonzero") is not False)
            for row in rows
        )
    ):
        return False
    input_shape = receipt.get("input_shape")
    tap_shapes = receipt.get("tap_output_shapes")
    valid_taps = bool(
        isinstance(input_shape, list)
        and len(input_shape) == 2
        and isinstance(input_shape[0], int)
        and not isinstance(input_shape[0], bool)
        and input_shape[0] > 0
        and input_shape[1] == WINDOW_SAMPLES
        and isinstance(tap_shapes, Mapping)
        and tap_shapes.get("final_temporal_hidden") == [input_shape[0], 375, 192]
        and tap_shapes.get("speaker_activity_logits") == [input_shape[0], 375, 4]
        and tap_shapes.get("psem_outputs") == [input_shape[0], 375]
    )
    return bool(
        receipt.get("schema_version") == 1
        and receipt.get("artifact_role") == "gradient_canary_receipt"
        and receipt.get("arm") == arm
        and receipt.get("passed") is True
        and receipt.get("payload_sha256") == _canonical_sha256(payload)
        and receipt.get("input_kind") == "raw_16khz_mono_waveform"
        and isinstance(receipt.get("loss"), (int, float))
        and not isinstance(receipt.get("loss"), bool)
        and math.isfinite(receipt["loss"])
        and isinstance(receipt.get("unclipped_gradient_norm"), (int, float))
        and not isinstance(receipt.get("unclipped_gradient_norm"), bool)
        and math.isfinite(receipt["unclipped_gradient_norm"])
        and receipt.get("clip_norm") == GRADIENT_CLIP_NORM
        and receipt.get("model_graph_receipt_sha256") == _canonical_sha256(model_graph_receipt)
        and valid_taps
        and bool(expected_paths)
        and isinstance(dependence, Mapping)
        and set(dependence) == expected_paths
        and all(value is True for value in dependence.values())
        and isinstance(reach_counts, Mapping)
        and set(reach_counts) == expected_paths
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in reach_counts.values()
        )
        and receipt.get("raw_waveform_gradient_nonzero") is True
    )


def canary_bundle_runtime_passed(
    gradient: Mapping[str, Any],
    update: Mapping[str, Any],
    timing: Mapping[str, Any],
    arm: str,
    parameter_inventory_receipt: Mapping[str, Any],
    model_graph_receipt: Mapping[str, Any],
) -> bool:
    if not gradient_canary_runtime_passed(
        gradient,
        arm,
        parameter_inventory_receipt=parameter_inventory_receipt,
        model_graph_receipt=model_graph_receipt,
    ):
        return False
    gradient_rows = gradient["parameters"]
    expected_changed = sorted(
        row["name"] for row in gradient_rows if row.get("expected_trainable") is True
    )
    update_payload = {key: value for key, value in update.items() if key != "payload_sha256"}
    timing_payload = {key: value for key, value in timing.items() if key != "payload_sha256"}
    inventory_sha = _canonical_sha256(parameter_inventory_receipt)
    graph_sha = _canonical_sha256(model_graph_receipt)
    frame_counts = timing.get("frame_counts")
    return bool(
        update.get("schema_version") == 1
        and update.get("artifact_role") == "update_canary_receipt"
        and update.get("arm") == arm
        and update.get("passed") is True
        and update.get("payload_sha256") == _canonical_sha256(update_payload)
        and update.get("optimizer") == "AdamW"
        and update.get("weight_decay") == WEIGHT_DECAY
        and update.get("learning_rates") == LEARNING_RATES
        and update.get("changed_parameters") == expected_changed
        and update.get("frozen_parameters_unchanged") is True
        and update.get("all_trainable_parameters_changed") is True
        and update.get("parameter_inventory_sha256") == inventory_sha
        and update.get("model_graph_receipt_sha256") == graph_sha
        and gradient.get("parameter_inventory_sha256") == inventory_sha
        and timing.get("schema_version") == 1
        and timing.get("artifact_role") == "timing_receipt"
        and timing.get("passed") is True
        and timing.get("payload_sha256") == _canonical_sha256(timing_payload)
        and timing.get("sample_rate_hz") == SAMPLE_RATE_HZ
        and timing.get("native_frame_samples") == 1280
        and timing.get("algorithmic_evidence_delay_samples") == 16640
        and isinstance(frame_counts, list)
        and frame_counts
        and all(type(value) is int and value == 375 for value in frame_counts)
        and timing.get("slot_count") == 4
        and timing.get("hidden_dimension") == 192
        and timing.get("lifecycle_fields_binary") is True
        and timing.get("all_four_stable_slots_alive") is True
        and timing.get("state_reset_first_frame_only") is True
        and timing.get("additional_future_context_observed") is False
        and timing.get("prefix_causality_passed") is True
        and timing.get("streaming_cache_integrity_passed") is True
        and _is_sha256(timing.get("streaming_trace_sha256"))
    )


def _resolve_module(model: nn.Module, path: str) -> nn.Module:
    try:
        module = model.get_submodule(path)
    except (AttributeError, KeyError) as exc:
        raise RuntimeAuditError(f"authorized module is absent from the graph: {path}") from exc
    if not isinstance(module, nn.Module):
        raise RuntimeAuditError(f"authorized graph path is not a module: {path}")
    return module


def validate_streaming_graph(model: nn.Module) -> dict[str, Any]:
    try:
        sortformer = model.get_submodule("sortformer")
        modules = sortformer.sortformer_modules
        layers = sortformer.transformer_encoder.layers
    except (AttributeError, KeyError) as exc:
        raise RuntimeAuditError("model does not expose the frozen Sortformer graph") from exc
    observed = {key: int(getattr(modules, key)) for key in LOW_LATENCY_STREAMING}
    if observed != LOW_LATENCY_STREAMING:
        raise RuntimeAuditError(
            f"streaming geometry differs from the #99 low-latency preset: {observed}"
        )
    if len(layers) != 18:
        raise RuntimeAuditError("the checkpoint does not contain 18 temporal layers")
    if int(modules.n_spk) != 4:
        raise RuntimeAuditError("the checkpoint does not expose four arrival-order slots")
    activity_head = _resolve_module(model, "sortformer.sortformer_modules.single_hidden_to_spks")
    hidden_tap = _resolve_module(model, "sortformer.transformer_encoder")
    psem_head = _resolve_module(model, "psem_head")
    runtime_hidden_tap = _resolve_module(model, "runtime_taps.final_temporal_hidden")
    runtime_activity_tap = _resolve_module(model, "runtime_taps.speaker_activity_logits")
    hidden_size = getattr(activity_head, "in_features", None)
    configured_hidden_size = getattr(
        getattr(getattr(sortformer, "_cfg", None), "model_defaults", None),
        "tf_d_model",
        None,
    )
    activity_slots = getattr(activity_head, "out_features", None)
    if configured_hidden_size != 192 or hidden_size != 192:
        raise RuntimeAuditError("the final temporal hidden dimension is not 192")
    if activity_slots != 4:
        raise RuntimeAuditError("the executable activity head does not emit four logits")
    if not isinstance(runtime_hidden_tap, nn.Identity) or not isinstance(
        runtime_activity_tap, nn.Identity
    ):
        raise RuntimeAuditError("the executable runtime evidence taps are absent")
    input_norm = getattr(psem_head, "input_norm", None)
    gru = getattr(psem_head, "gru", None)
    if (
        tuple(getattr(input_norm, "normalized_shape", ())) != (208,)
        or getattr(gru, "input_size", None) != 208
        or getattr(gru, "hidden_size", None) != 64
        or getattr(gru, "num_layers", None) != 1
    ):
        raise RuntimeAuditError("the executable PSEM head differs from the frozen graph")
    graph_rows = [
        {
            "path": name,
            "type": f"{module.__class__.__module__}.{module.__class__.__qualname__}",
        }
        for name, module in model.named_modules()
    ]
    parameter_rows = [
        {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
        for name, parameter in model.named_parameters()
    ]
    state_dict_rows = [
        {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in model.state_dict().items()
    ]
    graph_sha256 = hashlib.sha256(
        json.dumps(
            {
                "modules": graph_rows,
                "parameters": parameter_rows,
                "state_dict": state_dict_rows,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    parameter_schema_sha256 = _canonical_sha256(parameter_rows)
    state_dict_schema_sha256 = _canonical_sha256(state_dict_rows)
    payload = {
        "schema_version": 1,
        "artifact_role": "model_graph_receipt",
        "passed": True,
        "temporal_layer_count": len(layers),
        "hidden_tensor_identity": "sortformer.transformer_encoder.output",
        "hidden_tap_module_type": (
            f"{hidden_tap.__class__.__module__}.{hidden_tap.__class__.__qualname__}"
        ),
        "hidden_dimension": hidden_size,
        "activity_logit_identity": (
            "sortformer.sortformer_modules.single_hidden_to_spks.output_pre_sigmoid"
        ),
        "activity_head_module_type": (
            f"{activity_head.__class__.__module__}.{activity_head.__class__.__qualname__}"
        ),
        "runtime_canary_tap_paths": {
            "final_temporal_hidden": "runtime_taps.final_temporal_hidden",
            "speaker_activity_logits": "runtime_taps.speaker_activity_logits",
            "psem_outputs": "psem_head",
        },
        "slot_count": int(modules.n_spk),
        "native_frame_ms": 80,
        "streaming_geometry": observed,
        "algorithmic_evidence_delay_ms": 1040,
        "state_reset_policy": "declared_source_or_reset_boundary_only",
        "slot_alive_policy": "issue_99_all_four_stable_columns_alive",
        "executable_graph_sha256": graph_sha256,
        "parameter_schema_sha256": parameter_schema_sha256,
        "state_dict_schema_sha256": state_dict_schema_sha256,
        "executable_module_count": len(graph_rows),
        "executable_parameter_count": len(parameter_rows),
        "executable_state_entry_count": len(state_dict_rows),
    }
    return {**payload, "payload_sha256": _canonical_sha256(payload)}


def _validate_raw_waveform(waveform: torch.Tensor) -> None:
    if waveform.ndim != 2 or waveform.shape[1] != WINDOW_SAMPLES:
        raise RuntimeAuditError("canary input must be complete 30-second waveform batches")
    if not torch.is_floating_point(waveform) or not bool(torch.isfinite(waveform).all()):
        raise RuntimeAuditError("canary waveform must be finite floating-point PCM")


def _trace_matches_low_latency(
    streaming_trace: tuple[dict[str, int], ...], frame_count: int
) -> bool:
    expected_steps = math.ceil(frame_count / LOW_LATENCY_STREAMING["chunk_len"])
    if len(streaming_trace) != expected_steps:
        return False
    emitted_total = 0
    prior_cache = 0
    prior_fifo = 0
    for index, row in enumerate(streaming_trace):
        emitted = min(LOW_LATENCY_STREAMING["chunk_len"], frame_count - emitted_total)
        expected_left = (
            0
            if index == 0
            else LOW_LATENCY_STREAMING["chunk_left_context"] * STREAMING_SUBSAMPLING_FACTOR
        )
        expected_right = (
            0
            if index == expected_steps - 1
            else LOW_LATENCY_STREAMING["chunk_right_context"] * STREAMING_SUBSAMPLING_FACTOR
        )
        left_frames = expected_left // STREAMING_SUBSAMPLING_FACTOR
        right_frames = expected_right // STREAMING_SUBSAMPLING_FACTOR
        expected_chunk = left_frames + emitted + right_frames
        fifo_total = prior_fifo + emitted
        if fifo_total > LOW_LATENCY_STREAMING["fifo_len"]:
            pop_count = max(
                LOW_LATENCY_STREAMING["spkcache_update_period"],
                emitted - LOW_LATENCY_STREAMING["fifo_len"] + prior_fifo,
            )
            pop_count = min(pop_count, fifo_total)
            expected_fifo_after = fifo_total - pop_count
            expected_cache_after = min(
                LOW_LATENCY_STREAMING["spkcache_len"], prior_cache + pop_count
            )
        else:
            expected_fifo_after = fifo_total
            expected_cache_after = prior_cache
        if (
            row["step_index"] != index
            or row["left_offset"] != expected_left
            or row["right_offset"] != expected_right
            or row["chunk_feature_frames"] != expected_chunk
            or row["chunk_length_min"] != expected_chunk
            or row["chunk_length_max"] != expected_chunk
            or row["emitted_frames"] != emitted
            or row["cache_before_frames"] != prior_cache
            or row["fifo_before_frames"] != prior_fifo
            or row["cache_after_frames"] != expected_cache_after
            or row["fifo_after_frames"] != expected_fifo_after
        ):
            return False
        prior_cache = row["cache_after_frames"]
        prior_fifo = row["fifo_after_frames"]
        emitted_total += emitted
    return emitted_total == frame_count


def build_timing_receipt(
    waveform_lengths: torch.Tensor,
    probabilities: torch.Tensor,
    activity_logits: torch.Tensor,
    final_temporal_hidden: torch.Tensor,
    slot_alive: torch.Tensor,
    state_reset: torch.Tensor,
    evidence_delay_seconds: torch.Tensor,
    streaming_trace: tuple[dict[str, int], ...],
    prefix_causality: Mapping[str, Any],
) -> dict[str, Any]:
    expected_reset = torch.zeros_like(state_reset, dtype=torch.bool)
    if expected_reset.ndim == 3 and expected_reset.shape[1] > 0:
        expected_reset[:, 0, 0] = True
    if (
        waveform_lengths.ndim != 1
        or probabilities.ndim != 3
        or probabilities.shape[-1] != 4
        or activity_logits.shape != probabilities.shape
        or final_temporal_hidden.shape != (*probabilities.shape[:2], 192)
        or slot_alive.shape != probabilities.shape
        or state_reset.shape != (*probabilities.shape[:2], 1)
        or evidence_delay_seconds.shape != state_reset.shape
        or waveform_lengths.shape[0] != probabilities.shape[0]
        or bool((waveform_lengths <= 0).any())
        or bool((waveform_lengths % 1280 != 0).any())
        or bool((waveform_lengths // 1280 != probabilities.shape[1]).any())
        or not all(
            bool(torch.isfinite(value).all())
            for value in (
                probabilities,
                activity_logits,
                final_temporal_hidden,
                slot_alive,
                state_reset,
                evidence_delay_seconds,
            )
        )
        or not bool(((slot_alive == 0) | (slot_alive == 1)).all())
        or not bool((slot_alive == 1).all())
        or not bool(((state_reset == 0) | (state_reset == 1)).all())
        or not torch.equal(state_reset.to(torch.bool), expected_reset)
        or not bool((evidence_delay_seconds == 1.04).all())
    ):
        raise RuntimeAuditError("timing evidence differs from the exact native frame contract")
    expected_trace_keys = {
        "step_index",
        "left_offset",
        "right_offset",
        "chunk_feature_frames",
        "chunk_length_min",
        "chunk_length_max",
        "cache_before_frames",
        "fifo_before_frames",
        "cache_after_frames",
        "fifo_after_frames",
        "emitted_frames",
    }
    if (
        not streaming_trace
        or any(set(row) != expected_trace_keys for row in streaming_trace)
        or [row["step_index"] for row in streaming_trace] != list(range(len(streaming_trace)))
        or any(
            type(value) is not int or value < 0 for row in streaming_trace for value in row.values()
        )
        or any(row["chunk_length_min"] > row["chunk_length_max"] for row in streaming_trace)
        or any(row["chunk_length_max"] > row["chunk_feature_frames"] for row in streaming_trace)
        or any(
            row["cache_before_frames"] > LOW_LATENCY_STREAMING["spkcache_len"]
            for row in streaming_trace
        )
        or any(
            row["cache_after_frames"] > LOW_LATENCY_STREAMING["spkcache_len"]
            for row in streaming_trace
        )
        or any(
            row["fifo_before_frames"] > LOW_LATENCY_STREAMING["fifo_len"] for row in streaming_trace
        )
        or any(
            row["fifo_after_frames"] > LOW_LATENCY_STREAMING["fifo_len"] for row in streaming_trace
        )
        or any(row["emitted_frames"] <= 0 for row in streaming_trace)
        or sum(row["emitted_frames"] for row in streaming_trace) != probabilities.shape[1]
        or not _trace_matches_low_latency(streaming_trace, probabilities.shape[1])
        or prefix_causality.get("passed") is not True
        or prefix_causality.get("algorithmic_evidence_delay_samples") != 16640
        or type(prefix_causality.get("mutation_start_sample")) is not int
        or type(prefix_causality.get("protected_frame_count")) is not int
        or prefix_causality["protected_frame_count"] <= 0
        or prefix_causality["protected_frame_count"] >= probabilities.shape[1]
        or prefix_causality.get("protected_prefix_unchanged") is not True
        or prefix_causality.get("suffix_change_observed") is not True
    ):
        raise RuntimeAuditError("streaming cache or prefix-causality evidence is invalid")
    trace_rows = [dict(row) for row in streaming_trace]
    payload = {
        "schema_version": 1,
        "artifact_role": "timing_receipt",
        "passed": True,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "native_frame_samples": 1280,
        "algorithmic_evidence_delay_samples": 16640,
        "frame_counts": [int(value) for value in waveform_lengths // 1280],
        "slot_count": 4,
        "hidden_dimension": 192,
        "lifecycle_fields_binary": True,
        "all_four_stable_slots_alive": True,
        "state_reset_first_frame_only": True,
        "additional_future_context_observed": False,
        "prefix_causality_passed": True,
        "prefix_causality": dict(prefix_causality),
        "streaming_cache_integrity_passed": True,
        "streaming_step_count": len(trace_rows),
        "streaming_trace_sha256": _canonical_sha256(trace_rows),
    }
    return {**payload, "payload_sha256": _canonical_sha256(payload)}


def run_prefix_causality_audit(
    model: nn.Module,
    waveform: torch.Tensor,
    lengths: torch.Tensor,
) -> dict[str, Any]:
    from experiments.psem_sortformer_adaptation_depth.nemo_adapter import (
        TrainableSortformerPSEM,
    )

    if type(model) is not TrainableSortformerPSEM:
        raise RuntimeAuditError("prefix-causality audit requires the exact Sortformer wrapper")
    mutation_start = (waveform.shape[1] // 2 // 1280) * 1280
    protected_frame_count = (mutation_start - 16640 + 1279) // 1280
    if protected_frame_count <= 0 or mutation_start >= waveform.shape[1]:
        raise RuntimeAuditError("canary waveform is too short for prefix-causality audit")
    reset = torch.zeros(
        (waveform.shape[0], waveform.shape[1] // 1280, 1),
        dtype=torch.bool,
        device=waveform.device,
    )
    reset[:, 0, 0] = True
    perturbed = waveform.detach().clone()
    perturbed[:, mutation_start:] = -perturbed[:, mutation_start:] + 0.03125
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            baseline = model.sortformer_evidence(waveform, lengths, state_reset=reset)
            changed = model.sortformer_evidence(perturbed, lengths, state_reset=reset)
    finally:
        model.train(was_training)
    tensor_pairs = {
        "probabilities": (baseline.probabilities, changed.probabilities),
        "activity_logits": (baseline.activity_logits, changed.activity_logits),
        "final_temporal_hidden": (
            baseline.final_temporal_hidden,
            changed.final_temporal_hidden,
        ),
    }
    prefix_max_abs_delta = {
        name: float(
            (left[:, :protected_frame_count] - right[:, :protected_frame_count]).abs().max()
        )
        for name, (left, right) in tensor_pairs.items()
    }
    suffix_max_abs_delta = {
        name: float(
            (left[:, protected_frame_count:] - right[:, protected_frame_count:]).abs().max()
        )
        for name, (left, right) in tensor_pairs.items()
    }
    prefix_unchanged = all(value <= 1e-6 for value in prefix_max_abs_delta.values())
    suffix_changed = any(value > 1e-6 for value in suffix_max_abs_delta.values())
    if not prefix_unchanged or not suffix_changed:
        raise RuntimeAuditError(
            "runtime evidence violates charged prefix causality or ignores the waveform suffix"
        )
    return {
        "passed": True,
        "algorithmic_evidence_delay_samples": 16640,
        "mutation_start_sample": mutation_start,
        "protected_frame_count": protected_frame_count,
        "protected_prefix_unchanged": True,
        "suffix_change_observed": True,
        "prefix_max_abs_delta": prefix_max_abs_delta,
        "suffix_max_abs_delta": suffix_max_abs_delta,
    }


def _output_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        return next(
            (tensor for item in value if (tensor := _output_tensor(item)) is not None), None
        )
    if isinstance(value, dict):
        return next(
            (tensor for item in value.values() if (tensor := _output_tensor(item)) is not None),
            None,
        )
    return None


def _waveform_dependence(
    model: nn.Module,
    paths: tuple[str, ...],
    waveform: torch.Tensor,
    lengths: torch.Tensor,
) -> dict[str, bool]:
    captured: dict[str, torch.Tensor] = {}
    hooks = []
    for path in paths:
        module = _resolve_module(model, path)

        def capture(_module: nn.Module, _inputs: Any, output: Any, key: str = path) -> None:
            tensor = _output_tensor(output)
            if tensor is not None:
                captured[key] = tensor.detach().cpu().clone()

        hooks.append(module.register_forward_hook(capture))
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            _exact_runtime_canary_loss(model, waveform, lengths)
        baseline = dict(captured)
        captured.clear()
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        perturbed = waveform * 0.97 + 0.01
        with torch.no_grad():
            _exact_runtime_canary_loss(model, perturbed, lengths)
        changed = {
            path: path in baseline
            and path in captured
            and baseline[path].shape == captured[path].shape
            and not torch.equal(baseline[path], captured[path])
            for path in paths
        }
    finally:
        model.train(was_training)
        for hook in hooks:
            hook.remove()
    if not all(changed.values()):
        raise RuntimeAuditError(f"authorized modules are not raw-waveform dependent: {changed}")
    return changed


def _exact_runtime_canary_loss(
    model: nn.Module,
    waveform: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    from experiments.psem_sortformer_adaptation_depth.nemo_adapter import (
        FIXED_RUNTIME_CANARY_METHODS,
        TrainableSortformerPSEM,
    )

    if type(model) is not TrainableSortformerPSEM or any(
        name in vars(model) or getattr(type(model), name, None) is not implementation
        for name, implementation in FIXED_RUNTIME_CANARY_METHODS
    ):
        raise RuntimeAuditError("model does not expose the exact fixed runtime canary path")
    return dict(FIXED_RUNTIME_CANARY_METHODS)["runtime_canary_loss"](model, waveform, lengths)


def run_gradient_update_canary(
    model: nn.Module,
    arm: str,
    waveform: torch.Tensor,
) -> dict[str, Any]:
    if arm == "F0-FROZEN-FLOAT":
        raise RuntimeAuditError("gradient and update canaries apply to trainable arms only")
    _validate_raw_waveform(waveform)
    graph = validate_streaming_graph(model)
    inventory = parameter_inventory(model, arm)
    optimizer = build_optimizer(model, arm)
    lengths = torch.full(
        (waveform.shape[0],),
        WINDOW_SAMPLES,
        dtype=torch.long,
        device=waveform.device,
    )
    paths = authorized_module_paths(arm)
    dependence = _waveform_dependence(model, paths, waveform, lengths)
    reach_counts = {path: 0 for path in paths}
    tap_outputs: dict[str, torch.Tensor] = {}
    tap_paths = {
        "final_temporal_hidden": "runtime_taps.final_temporal_hidden",
        "speaker_activity_logits": "runtime_taps.speaker_activity_logits",
        "psem_outputs": "psem_head",
    }
    hooks = []
    for path in reach_counts:
        module = _resolve_module(model, path)

        def mark_reached(_module: nn.Module, _inputs: Any, _output: Any, key: str = path) -> None:
            reach_counts[key] += 1

        hooks.append(module.register_forward_hook(mark_reached))
    for tap_name, tap_path in tap_paths.items():
        module = _resolve_module(model, tap_path)

        def capture_tap(_module: nn.Module, _inputs: Any, output: Any, key: str = tap_name) -> None:
            tensor = _output_tensor(output)
            if tensor is not None:
                tap_outputs[key] = tensor

        hooks.append(module.register_forward_hook(capture_tap))
    parameters = dict(model.named_parameters())
    before = {name: _tensor_sha256(parameter) for name, parameter in parameters.items()}
    optimizer.zero_grad(set_to_none=True)
    try:
        canary_waveform = waveform.detach().clone().requires_grad_(True)
        loss = _exact_runtime_canary_loss(model, canary_waveform, lengths)
    finally:
        for hook in hooks:
            hook.remove()
    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
        raise RuntimeAuditError("canary forward must return one finite scalar loss")
    if any(count <= 0 for count in reach_counts.values()):
        raise RuntimeAuditError(
            f"raw waveform did not reach every authorized module: {reach_counts}"
        )
    expected_tap_shapes = {
        "final_temporal_hidden": (waveform.shape[0], 375, 192),
        "speaker_activity_logits": (waveform.shape[0], 375, 4),
        "psem_outputs": (waveform.shape[0], 375),
    }
    if set(tap_outputs) != set(expected_tap_shapes) or any(
        tuple(tap_outputs[key].shape) != shape for key, shape in expected_tap_shapes.items()
    ):
        raise RuntimeAuditError("runtime canary did not execute the exact evidence tap geometry")
    loss.backward()
    if (
        canary_waveform.grad is None
        or not bool(torch.isfinite(canary_waveform.grad).all())
        or not bool(torch.count_nonzero(canary_waveform.grad))
    ):
        raise RuntimeAuditError("canary loss is not differentiably dependent on raw waveform")
    gradient_rows = []
    for name, parameter in parameters.items():
        expected = should_train(name, arm)
        gradient = parameter.grad
        finite = gradient is None or bool(torch.isfinite(gradient).all())
        nonzero = gradient is not None and bool(torch.count_nonzero(gradient))
        if expected and (not finite or not nonzero):
            raise RuntimeAuditError(f"required trainable gradient is absent or invalid: {name}")
        if not expected and gradient is not None and bool(torch.count_nonzero(gradient)):
            raise RuntimeAuditError(f"frozen parameter received a non-zero gradient: {name}")
        gradient_rows.append(
            {
                "name": name,
                "expected_trainable": expected,
                "gradient_present": gradient is not None,
                "finite": finite,
                "nonzero": nonzero,
            }
        )
    unclipped_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in parameters.values() if parameter.requires_grad],
        GRADIENT_CLIP_NORM,
    )
    if not bool(torch.isfinite(unclipped_norm)):
        raise RuntimeAuditError("gradient norm is non-finite")
    optimizer.step()
    changed = {
        name: before[name] != _tensor_sha256(parameter) for name, parameter in parameters.items()
    }
    expected_changed = {name for name in parameters if should_train(name, arm)}
    actual_changed = {name for name, value in changed.items() if value}
    if actual_changed != expected_changed:
        missing = sorted(expected_changed - actual_changed)
        forbidden = sorted(actual_changed - expected_changed)
        raise RuntimeAuditError(
            f"one-step updates differ from the whitelist; missing={missing}, forbidden={forbidden}"
        )
    gradient_payload = {
        "schema_version": 1,
        "artifact_role": "gradient_canary_receipt",
        "arm": arm,
        "passed": True,
        "input_kind": "raw_16khz_mono_waveform",
        "input_shape": list(waveform.shape),
        "loss": float(loss.detach().cpu()),
        "unclipped_gradient_norm": float(unclipped_norm.detach().cpu()),
        "clip_norm": GRADIENT_CLIP_NORM,
        "module_reach_counts": reach_counts,
        "tap_output_shapes": {key: list(tap_outputs[key].shape) for key in sorted(tap_outputs)},
        "raw_waveform_dependence": dependence,
        "raw_waveform_gradient_nonzero": True,
        "parameter_inventory_sha256": _canonical_sha256(inventory),
        "model_graph_receipt_sha256": _canonical_sha256(graph),
        "parameters": gradient_rows,
    }
    update_payload = {
        "schema_version": 1,
        "artifact_role": "update_canary_receipt",
        "arm": arm,
        "passed": True,
        "optimizer": "AdamW",
        "weight_decay": WEIGHT_DECAY,
        "learning_rates": LEARNING_RATES,
        "changed_parameters": sorted(actual_changed),
        "frozen_parameters_unchanged": True,
        "all_trainable_parameters_changed": True,
        "parameter_inventory_sha256": _canonical_sha256(inventory),
        "model_graph_receipt_sha256": _canonical_sha256(graph),
    }
    return {
        "gradient_canary_receipt": {
            **gradient_payload,
            "payload_sha256": _canonical_sha256(gradient_payload),
        },
        "update_canary_receipt": {
            **update_payload,
            "payload_sha256": _canonical_sha256(update_payload),
        },
        "parameter_inventory": inventory,
        "model_graph_receipt": graph,
    }
