from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
from student import StreamingState, StreamingStudent
from torch import Tensor, nn


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def bind_config(config_path: Path, run_root: Path, *, initialize: bool) -> Path:
    frozen = run_root / "frozen_config.json"
    requested = config_path.read_bytes()
    if initialize:
        if frozen.exists():
            raise RuntimeError(
                "frozen probe config already exists; backend stage cannot be replayed"
            )
        temporary = frozen.with_name(f".{frozen.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(requested)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, frozen)
        finally:
            temporary.unlink(missing_ok=True)
    elif not frozen.exists() or frozen.read_bytes() != requested:
        raise RuntimeError("requested config differs from the stage-1 frozen config")
    return frozen


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "PSEM-ISSUE-164-GT-PROBE-1":
        raise RuntimeError("unexpected GT probe config schema")
    root = Path(__file__).resolve().parent
    for relative, expected in value["implementation_sha256"].items():
        actual = sha256_file(root / relative)
        if actual != expected:
            raise RuntimeError(f"implementation identity mismatch for {relative}: {actual}")
    if value["budget"]["backend_fixture_optimizer_steps"] != 1:
        raise RuntimeError("backend fixture must use exactly one optimizer step")
    if not 1 <= value["budget"]["gt_optimizer_steps"] <= 19:
        raise RuntimeError("GT optimizer steps must be between one and nineteen")
    if value["budget"]["total_optimizer_steps"] != 1 + value["budget"]["gt_optimizer_steps"]:
        raise RuntimeError("total optimizer-step budget is inconsistent")
    return value


def local_path(value: str) -> Path:
    if os.name != "nt" and len(value) >= 3 and value[1:3] in (":/", ":\\"):
        return Path("/mnt") / value[0].lower() / value[3:].replace("\\", "/")
    return Path(value)


def runtime_identity(config_path: Path) -> dict[str, object]:
    root = Path(__file__).resolve().parent
    return {
        "implementation_base_commit": load_config(config_path)["implementation_base_commit"],
        "config_sha256": sha256_file(config_path),
        "implementation_sha256": {
            name: sha256_file(root / name)
            for name in ("gt_probe.py", "student.py", "run_gt_probe.sh")
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
    }


def device_report(device: torch.device) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "backend": "PyTorch ROCm through torch.cuda API",
        "device": str(device),
        "name": torch.cuda.get_device_name(device),
        "total_memory_bytes": properties.total_memory,
    }


def require_gpu(config: dict[str, Any]) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch did not expose the WSL ROCm GPU")
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    minimum_free = int(config["resource_guard"]["minimum_free_gpu_bytes"])
    if free < minimum_free:
        raise RuntimeError(f"GPU free-memory guard failed: {free} < {minimum_free}")
    if total != torch.cuda.get_device_properties(device).total_memory:
        raise RuntimeError("GPU memory identity is inconsistent")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parameter_count(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters())


def backend_stage(config_path: Path, run_root: Path) -> None:
    config_path = bind_config(config_path, run_root, initialize=True)
    config = load_config(config_path)
    device = require_gpu(config)
    seed_everything(config["seed"])
    conv = nn.Conv1d(64, 128, 5, stride=2).to(device)
    gru = nn.GRU(128, 128, batch_first=True).to(device)
    head = nn.Linear(128, 4).to(device)
    parameters = list(conv.parameters()) + list(gru.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1.0e-3)
    inputs = torch.randn(2, 64, 65, device=device)
    targets = torch.rand(2, 31, 4, device=device)
    before = conv.weight.detach().clone()
    encoded, _ = gru(torch.nn.functional.silu(conv(inputs)).transpose(1, 2))
    logits = head(encoded)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [value.grad for value in parameters if value.grad is not None]
    gradient_finite = bool(gradients) and all(
        bool(torch.isfinite(value).all().item()) for value in gradients
    )
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0)
    loss_value = float(loss.detach().cpu())
    grad_norm_value = float(grad_norm.detach().cpu())
    if not math.isfinite(loss_value) or not math.isfinite(grad_norm_value) or not gradient_finite:
        raise RuntimeError("representative backend fixture failed pre-update finite checks")
    optimizer.step()
    torch.cuda.synchronize(device)
    delta = float((conv.weight.detach() - before).norm().cpu())
    if not math.isfinite(delta) or not delta > 0.0:
        raise RuntimeError("representative backend fixture did not update its parameter")
    atomic_json(
        run_root / "backend_receipt.json",
        {
            "stage": "backend",
            "fixture_only_not_learned_student": True,
            "optimizer_steps": 1,
            "fixture": "Conv1d(64,128,k5,s2)+GRU(128)+Linear(4)+sigmoid_BCE",
            "loss": loss_value,
            "gradient_finite": gradient_finite,
            "gradient_norm_before_clip": grad_norm_value,
            "conv_weight_delta_norm": delta,
            "device": device_report(device),
            "runtime": runtime_identity(config_path),
        },
    )


def read_wave(
    path: Path, sample_count: int, expected_rate: int, expected_channels: int
) -> tuple[Tensor, dict[str, object]]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        rate = handle.getframerate()
        width = handle.getsampwidth()
        frames = handle.getnframes()
        if rate != expected_rate or channels != expected_channels or width != 2:
            raise RuntimeError(
                f"unexpected WAV geometry rate={rate}, channels={channels}, sample_width={width}"
            )
        if frames < sample_count:
            raise RuntimeError(f"source has {frames} samples but probe needs {sample_count}")
        raw = handle.readframes(sample_count)
    values = np.frombuffer(raw, dtype="<i2").reshape(-1, channels).astype(np.float32)
    mono = values.mean(axis=1) / 32768.0
    return torch.from_numpy(mono.copy()), {
        "sample_rate": rate,
        "source_channels": channels,
        "channel_projection": "arithmetic mean of all channels in each source sample frame",
        "sample_width_bytes": width,
        "source_total_samples": frames,
    }


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"invalid JSONL artifact: {path}")
    return rows


def selected_normalized_labels(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    source = config["source"]
    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "experiments" / "psem_training_strategy_gate" / "data" / "v2"
    manifest_paths = {
        "source_manifest": data_root / "source_manifest.jsonl",
        "split_manifest": data_root / "split_manifest.json",
        "topology_manifest": data_root / "topology_manifest.jsonl",
        "identity_components": data_root / "identity_components.json",
    }
    for name, path in manifest_paths.items():
        if sha256_file(path) != source["manifest_sha256"][name]:
            raise RuntimeError(f"frozen data identity mismatch: {name}")
    split = json.loads(manifest_paths["split_manifest"].read_text(encoding="utf-8"))
    matching_components = [
        component
        for component in split["assignments"]["components"]
        if source["source_id"] in component["source_ids"]
    ]
    if len(matching_components) != 1:
        raise RuntimeError("selected source does not have exactly one split assignment")
    component = matching_components[0]
    if component["component_id"] != source["component_id"] or component["role"] != source["role"]:
        raise RuntimeError("selected source split role/component differs from the frozen choice")
    identities = json.loads(manifest_paths["identity_components"].read_text(encoding="utf-8"))
    identity_component = next(
        (row for row in identities["components"] if row["component_id"] == source["component_id"]),
        None,
    )
    if (
        identity_component is None
        or identity_component["eval_forbidden"]
        or identity_component["selection_exposed_source_ids"]
        or not identity_component["split_assignment_eligible"]
    ):
        raise RuntimeError("selected source component is not eligible for this FIT probe")
    source_rows = [
        row
        for row in jsonl(manifest_paths["source_manifest"])
        if row["source_id"] == source["source_id"]
    ]
    topology_rows = [
        row
        for row in jsonl(manifest_paths["topology_manifest"])
        if row["source_id"] == source["source_id"]
    ]
    if len(source_rows) != 1 or len(topology_rows) != 1:
        raise RuntimeError("selected source has ambiguous frozen manifest records")
    source_row = source_rows[0]
    if source_row["waveform_sha256"] != source["waveform_sha256"]:
        raise RuntimeError("selected source manifest waveform identity differs")
    from experiments.psem_training_strategy_gate.data.reference_normalization import (
        normalize_reference_session,
        open_reference_checkout,
    )

    checkout = open_reference_checkout(local_path(source["reference_root"]))
    normalized = normalize_reference_session(
        source_row, local_path(source["corpus_root"]), checkout
    )
    label_dict = normalized.labels.to_dict()
    label_sha256 = canonical_sha256(label_dict)
    if (
        label_sha256 != source["label_result_sha256"]
        or label_sha256 != topology_rows[0]["label_result_sha256"]
    ):
        raise RuntimeError("selected normalized labels differ from the frozen topology artifact")
    if normalized.reference_sha256 != source["reference_rttm_sha256"]:
        raise RuntimeError("selected RTTM identity differs")
    if normalized.reference_checkout_provenance["commit"] != source["reference_commit"]:
        raise RuntimeError("selected reference checkout revision differs")
    intervals = [
        {
            **interval.to_dict(),
            "masked_for_activity": bool(interval.ambiguous or not interval.speaker_identity_known),
        }
        for interval in normalized.intervals
    ]
    return {
        "contract_version": normalized.labels.contract_version,
        "contract_document_sha256": normalized.labels.contract_document_sha256,
        "label_result_sha256": label_sha256,
        "raw_normalized_intervals": intervals,
        "relation_only_nonlexical_masks": [
            mask.to_dict() for mask in normalized.parsed_nonlexical.masks
        ],
        "generated_label_manifest": normalized.manifest_row(),
        "generated_labels_not_used_as_training_targets": True,
    }, {
        "reference_rttm": normalized.reference_sha256,
        "source_annotation": normalized.source_annotation_sha256,
        "speaker_mapping": normalized.speaker_mapping_sha256,
    }


def first_appearance_slots(timeline: list[dict[str, Any]], permitted_end_sample: int) -> list[str]:
    first: dict[str, int] = {}
    for row in timeline:
        if row["masked_for_activity"] or int(row["start_sample"]) >= permitted_end_sample:
            continue
        for speaker in row["active_speakers"]:
            first[speaker] = min(
                first.get(speaker, int(row["start_sample"])), int(row["start_sample"])
            )
    ordered = sorted(first, key=lambda speaker: (first[speaker], speaker))
    if len(ordered) != 4:
        raise RuntimeError(
            f"selected source does not expose exactly four known speakers: {ordered}"
        )
    return ordered


def projected_targets(
    frontiers: Tensor, timeline: list[dict[str, Any]], slots: list[str], receptive_field: int
) -> tuple[Tensor, Tensor]:
    targets = torch.zeros(frontiers.numel(), len(slots), dtype=torch.float32)
    validity = torch.zeros(frontiers.numel(), dtype=torch.bool)
    for frame, frontier_value in enumerate(frontiers.tolist()):
        left = frontier_value - StreamingStudent.output_hop_samples
        right = frontier_value
        overlaps = [
            row
            for row in timeline
            if int(row["start_sample"]) < right and int(row["end_sample"]) > left
        ]
        covered = sum(
            max(0, min(right, int(row["end_sample"])) - max(left, int(row["start_sample"])))
            for row in overlaps
        )
        valid = (
            left >= 0
            and frontier_value >= receptive_field
            and covered == StreamingStudent.output_hop_samples
            and not any(bool(row["masked_for_activity"]) for row in overlaps)
        )
        validity[frame] = valid
        if not valid:
            continue
        for slot, speaker in enumerate(slots):
            active = sum(
                max(0, min(right, int(row["end_sample"])) - max(left, int(row["start_sample"])))
                for row in overlaps
                if speaker in row["active_speakers"]
            )
            targets[frame, slot] = min(1.0, active / StreamingStudent.output_hop_samples)
    return targets, validity


def prepare_stage(config_path: Path, run_root: Path) -> None:
    config_path = bind_config(config_path, run_root, initialize=False)
    config = load_config(config_path)
    source = config["source"]
    exposure = int(config["training"]["chunk_samples"]) * int(
        config["budget"]["gt_optimizer_steps"]
    )
    maximum = int(config["budget"]["maximum_unique_fit_audio_samples"])
    if exposure > maximum:
        raise RuntimeError(f"unique FIT audio budget exceeded: {exposure} > {maximum}")
    source_path = local_path(source["waveform"])
    if sha256_file(source_path) != source["waveform_sha256"]:
        raise RuntimeError("source waveform identity mismatch")
    waveform, geometry = read_wave(
        source_path, exposure, StreamingStudent.sample_rate, int(source["expected_channels"])
    )
    normalized_gt, annotation_hashes = selected_normalized_labels(config)
    timeline = normalized_gt["raw_normalized_intervals"]
    slots = first_appearance_slots(timeline, exposure)
    model = StreamingStudent()
    count = parameter_count(model)
    if count != config["architecture"]["expected_parameter_count"]:
        raise RuntimeError(f"student parameter count differs from frozen shape: {count}")
    if (
        not config["architecture"]["minimum_parameters"]
        <= count
        <= config["architecture"]["maximum_parameters"]
    ):
        raise RuntimeError(f"student parameter count outside frozen range: {count}")
    projected_frontiers = torch.arange(
        model.output_hop_samples,
        exposure + 1,
        model.output_hop_samples,
        dtype=torch.int64,
    )
    projected_values, projected_validity = projected_targets(
        projected_frontiers, timeline, slots, model.receptive_field_samples
    )
    if projected_frontiers.numel() != config["verification"]["partition_witness_output_frames"]:
        raise RuntimeError("prepared prefix does not have the frozen output-frame count")
    geometry_samples = min(int(config["training"]["chunk_samples"]), 6400)
    with torch.no_grad():
        state = model.initial_state(1, torch.device("cpu"))
        logits, frontiers, _ = model.forward_stream(waveform[:geometry_samples].unsqueeze(0), state)
    targets = projected_values[: frontiers.numel()]
    validity = projected_validity[: frontiers.numel()]
    if (
        not torch.equal(frontiers.cpu(), projected_frontiers[: frontiers.numel()])
        or logits.shape[:2] != targets.unsqueeze(0).shape[:2]
        or not bool(validity.any())
    ):
        raise RuntimeError("CPU-neutral student/data geometry check failed")
    prepared = {
        "waveform": waveform,
        "raw_normalized_gt": normalized_gt,
        "unaltered_annotation_timeline": timeline,
        "slot_mapping": {str(index): speaker for index, speaker in enumerate(slots)},
        "projected_frontiers": projected_frontiers,
        "projected_targets": projected_values,
        "projected_activity_validity": projected_validity,
        "source": {
            "source_id": source["source_id"],
            "meeting": source["meeting"],
            "role": source["role"],
            "component_id": source["component_id"],
            "waveform_sha256": source["waveform_sha256"],
            "annotation_sha256": annotation_hashes,
            "prepared_sample_bounds": [0, exposure],
            "training_sample_bounds": [0, exposure],
            **geometry,
        },
    }
    prepared_path = run_root / "prepared_fit.pt"
    torch.save(prepared, prepared_path)
    atomic_json(
        run_root / "prepare_receipt.json",
        {
            "stage": "prepare",
            "optimizer_steps": 0,
            "teacher_access": False,
            "source_role": source["role"],
            "source": prepared["source"],
            "prepared_sha256": sha256_file(prepared_path),
            "student_parameter_count": count,
            "architecture": config["architecture"],
            "frame_semantics": {
                "sample_rate": model.sample_rate,
                "analysis_window_samples": model.window_samples,
                "mel_hop_samples": model.mel_hop_samples,
                "output_hop_samples": model.output_hop_samples,
                "first_output_frontier_sample": model.output_hop_samples,
                "causal_receptive_field_samples": model.receptive_field_samples,
                "target_interval": "output ordinal k projects exact bin [k*1280,(k+1)*1280); its required physical audio frontier is the bin end",
                "target_value": "raw canonical per-speaker active sample coverage / 1280; four independent occupancy sigmoid/BCE slots",
                "support_phase": {
                    "logit_ordinal_0": "physical frontier and target-bin end 1280; real support [0,1280), with synthetic causal history before source zero",
                    "logit_ordinal_1": "physical frontier and target-bin end 2560; no sample after the target bin is consumed",
                    "first_full_receptive_field_logit_ordinal": 3,
                    "first_full_receptive_field_physical_frontier": 5120,
                    "first_full_receptive_field_target_bin": [3840, 5120],
                    "invalid_initial_outputs_due_to_partial_receptive_field": 3,
                    "frontend_initial_sample_padding": 240,
                    "each_conv_initial_feature_padding": 3,
                },
                "availability_delay_samples_after_target_end": 0,
                "initial_validity": f"false for the first three causal-padding outputs; valid once physical frontier >= {model.receptive_field_samples}",
                "activity_validity": "complete raw GT coverage and no ambiguous or unknown-identity portion; relation-only nonlexical masks do not invalidate activity",
                "future_lookahead_samples": 0,
                "normalization": "fixed pointwise (log mel + 10) / 5; no chunk statistics",
            },
            "slot_mapping": prepared["slot_mapping"],
            "gt_provenance": {
                "contract_version": normalized_gt["contract_version"],
                "label_result_sha256": normalized_gt["label_result_sha256"],
                "raw_normalized_interval_count": len(timeline),
                "relation_only_nonlexical_mask_count": len(
                    normalized_gt["relation_only_nonlexical_masks"]
                ),
                "generated_jitter_reconciled_labels_used_for_targets": False,
            },
            "cpu_neutral_geometry": {
                "input_samples": geometry_samples,
                "output_frames": logits.shape[1],
                "valid_output_frames": int(validity.sum()),
                "output_slots": logits.shape[2],
            },
            "runtime": runtime_identity(config_path),
        },
    )


def state_equal(left: StreamingState, right: StreamingState, tolerance: float) -> float:
    values = [
        (left.sample_buffer, right.sample_buffer),
        (left.hidden, right.hidden),
        *zip(left.conv_buffers, right.conv_buffers, strict=True),
    ]
    maximum = 0.0
    for first, second in values:
        if first.shape != second.shape:
            return math.inf
        if first.numel():
            maximum = max(maximum, float((first - second).abs().max().cpu()))
    if left.total_samples != right.total_samples or left.emitted_outputs != right.emitted_outputs:
        return math.inf
    return maximum


def stream_partitions(
    model: StreamingStudent, audio: Tensor, partitions: list[int], device: torch.device
) -> tuple[Tensor, Tensor, StreamingState]:
    state = model.initial_state(1, device)
    logits: list[Tensor] = []
    frontiers: list[Tensor] = []
    cursor = 0
    for size in partitions:
        piece = audio[cursor : cursor + size].to(device).unsqueeze(0)
        output, output_frontiers, state = model.forward_stream(piece, state)
        logits.append(output)
        frontiers.append(output_frontiers)
        cursor += size
    if cursor != audio.numel():
        raise RuntimeError("stream partition sizes do not cover witness audio")
    return torch.cat(logits, dim=1), torch.cat(frontiers), state


def partition_sizes(total: int) -> list[int]:
    pattern = (997, 65536, 131071, 243200, 77777)
    values: list[int] = []
    remaining = total
    index = 0
    while remaining:
        size = min(remaining, pattern[index % len(pattern)])
        values.append(size)
        remaining -= size
        index += 1
    return values


def rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def train_stage(config_path: Path, run_root: Path) -> None:
    config_path = bind_config(config_path, run_root, initialize=False)
    config = load_config(config_path)
    device = require_gpu(config)
    seed_everything(config["seed"])
    prepared_path = run_root / "prepared_fit.pt"
    prepare_receipt = json.loads((run_root / "prepare_receipt.json").read_text())
    if prepare_receipt["runtime"]["config_sha256"] != sha256_file(config_path):
        raise RuntimeError("prepared input was created under a different frozen config")
    if sha256_file(prepared_path) != prepare_receipt["prepared_sha256"]:
        raise RuntimeError("prepared input identity mismatch")
    prepared = torch.load(prepared_path, map_location="cpu", weights_only=False)
    waveform: Tensor = prepared["waveform"]
    model = StreamingStudent().to(device)
    count = parameter_count(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    chunk_samples = int(config["training"]["chunk_samples"])
    steps = int(config["budget"]["gt_optimizer_steps"])
    state = model.initial_state(1, device)
    records: list[dict[str, object]] = []
    torch.cuda.reset_peak_memory_stats(device)
    first_parameter = next(model.parameters())
    torch.cuda.synchronize(device)
    training_started = time.perf_counter()
    for step in range(steps):
        torch.cuda.synchronize(device)
        step_started = time.perf_counter()
        left = step * chunk_samples
        right = left + chunk_samples
        audio = waveform[left:right].to(device).unsqueeze(0)
        before = first_parameter.detach().clone()
        logits, frontiers, state = model.forward_stream(audio, state)
        frames_per_update = int(config["training"]["frames_per_update"])
        if logits.shape[1] != frames_per_update:
            raise RuntimeError(
                f"GT update {step + 1} emitted {logits.shape[1]} frames instead of the frozen count"
            )
        frame_left = step * frames_per_update
        frame_right = frame_left + frames_per_update
        expected_frontiers = prepared["projected_frontiers"][frame_left:frame_right]
        if not torch.equal(frontiers.cpu(), expected_frontiers):
            raise RuntimeError(f"GT update {step + 1} frontiers differ from prepared targets")
        targets = prepared["projected_targets"][frame_left:frame_right].to(device).unsqueeze(0)
        validity = prepared["projected_activity_validity"][frame_left:frame_right].to(device)
        if not bool(validity.any()):
            raise RuntimeError(f"GT step {step + 1} has no valid causal target frames")
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits[:, validity], targets[:, validity]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"]["maximum_grad_norm"])
        )
        gradients_finite = all(
            value.grad is None or bool(torch.isfinite(value.grad).all().item())
            for value in model.parameters()
        )
        loss_value = float(loss.detach().cpu())
        grad_norm_value = float(grad_norm.detach().cpu())
        if (
            not math.isfinite(loss_value)
            or not math.isfinite(grad_norm_value)
            or not gradients_finite
        ):
            raise RuntimeError(f"GT update {step + 1} failed pre-update finite checks")
        optimizer.step()
        torch.cuda.synchronize(device)
        delta = float((first_parameter.detach() - before).norm().cpu())
        if not math.isfinite(delta) or not delta > 0:
            raise RuntimeError(f"GT update {step + 1} did not change the intended parameter")
        step_elapsed = time.perf_counter() - step_started
        state = state.detached()
        records.append(
            {
                "step": step + 1,
                "loss": loss_value,
                "gradient_norm_before_clip": grad_norm_value,
                "gradients_finite": gradients_finite,
                "first_parameter_delta_norm": delta,
                "elapsed_seconds_synchronized": step_elapsed,
                "input_samples": [left, right],
                "output_frames": frames_per_update,
                "output_frontier_samples": [int(frontiers[0]), int(frontiers[-1])],
                "state_carried_from_previous_step": step > 0,
                "state_detached_not_reset": True,
            }
        )
    torch.cuda.synchronize(device)
    training_loop_seconds = time.perf_counter() - training_started
    optimizer_steps = {
        int(value["step"].item())
        for state_value in optimizer.state.values()
        for key, value in state_value.items()
        if key == "step"
    }
    if optimizer_steps != {steps}:
        raise RuntimeError(f"optimizer step counters do not equal {steps}: {optimizer_steps}")
    training_end = steps * chunk_samples
    if (
        state.total_samples != training_end
        or state.emitted_outputs != config["verification"]["partition_witness_output_frames"]
    ):
        raise RuntimeError(
            "streaming state frontier/frame count differs from chronological training exposure"
        )
    checkpoint = {
        "schema": "PSEM-ISSUE-164-GT-CHECKPOINT-1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "global_step": steps,
        "rng": rng_state(),
        "current_input_frontier": training_end,
        "streaming_state": state.cpu_dict(),
        "streaming_state_semantics": "actual chronological TBPTT trajectory; buffers/hidden were produced immediately before the latest parameter update and are not claimed to equal a frozen-final-weight replay",
        "consumed_source": {
            **prepared["source"],
            "consumed_sample_bounds": [0, training_end],
        },
    }
    checkpoint_path = run_root / "student_gt_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    tolerance = float(config["verification"]["partition_tolerance"])
    witness_samples = min(training_end, int(config["verification"]["partition_witness_samples"]))
    witness = waveform[:witness_samples]
    model.eval()
    with torch.no_grad():
        whole_logits, whole_frontiers, whole_state = stream_partitions(
            model, witness, [witness_samples], device
        )
        chunked_logits, chunked_frontiers, chunked_state = stream_partitions(
            model, witness, partition_sizes(witness_samples), device
        )
        midpoint = witness_samples // 2
        left_logits, left_frontiers, handoff_state = stream_partitions(
            model, witness[:midpoint], [midpoint], device
        )
        restored = model.state_from_dict(handoff_state.cpu_dict(), device)
        right_logits, right_frontiers, restored_state = model.forward_stream(
            witness[midpoint:].to(device).unsqueeze(0), restored
        )
        handoff_logits = torch.cat((left_logits, right_logits), dim=1)
        handoff_frontiers = torch.cat((left_frontiers, right_frontiers))
        inference_samples = int(config["verification"]["inference_samples"])
        inference_audio = waveform[:inference_samples].to(device).unsqueeze(0)
        expected_logits, expected_frontiers, _ = model.forward_stream(
            inference_audio, model.initial_state(1, device)
        )
    if whole_logits.shape[1] != config["verification"]["partition_witness_output_frames"]:
        raise RuntimeError("full-prefix witness emitted an unexpected frame count")
    prefix_delta = float(
        (whole_logits[:, : expected_logits.shape[1]] - expected_logits).abs().max().cpu()
    )
    partition_delta = float((whole_logits - chunked_logits).abs().max().cpu())
    handoff_delta = float((whole_logits - handoff_logits).abs().max().cpu())
    state_delta = state_equal(whole_state, chunked_state, tolerance)
    handoff_state_delta = state_equal(whole_state, restored_state, tolerance)
    if (
        not torch.equal(whole_frontiers, chunked_frontiers)
        or not torch.equal(whole_frontiers, handoff_frontiers)
        or max(prefix_delta, partition_delta, handoff_delta, state_delta, handoff_state_delta)
        > tolerance
    ):
        raise RuntimeError("frozen-weight streaming partition equivalence failed")
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    if max(peak_allocated, peak_reserved) > config["resource_guard"]["maximum_expected_peak_bytes"]:
        raise RuntimeError("student path exceeded the frozen GPU-memory cap")
    atomic_json(
        run_root / "train_receipt.json",
        {
            "stage": "train",
            "arm": "S-GT",
            "teacher_access": False,
            "optimizer_steps": steps,
            "parameter_count": count,
            "updates": records,
            "training_loop_seconds_synchronized": training_loop_seconds,
            "training_timing_scope": "each step and loop include host-to-device input/target transfer, frontend, student forward, BCE, backward, gradient checks/clipping, optimizer step, and synchronization; step 1 includes first-use overhead",
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_contents": [
                "model",
                "optimizer",
                "config",
                "global_step",
                "rng",
                "current_input_frontier",
                "streaming_state",
                "streaming_state_semantics",
                "consumed_source",
            ],
            "streaming_partition_witness": {
                "sample_bounds": [0, witness_samples],
                "whole_vs_chunked_max_abs_logit_delta": partition_delta,
                "whole_vs_chunked_max_abs_state_delta": state_delta,
                "midstream_handoff_max_abs_logit_delta": handoff_delta,
                "midstream_handoff_max_abs_state_delta": handoff_state_delta,
                "prefix_extension_max_abs_logit_delta": prefix_delta,
                "prefix_extension_compares_same_frozen_weights": True,
                "tolerance": tolerance,
                "weights_frozen_during_witness": True,
            },
            "inference_replay_expected": {
                "source_sample_bounds": [0, inference_samples],
                "frontiers": expected_frontiers.cpu().tolist(),
                "probabilities": torch.sigmoid(expected_logits).cpu().tolist(),
            },
            "gpu": {
                **device_report(device),
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
            },
            "runtime": runtime_identity(config_path),
        },
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "reload",
            "--config",
            str(config_path),
            "--run-root",
            str(run_root),
        ],
        check=True,
    )


def reload_stage(config_path: Path, run_root: Path) -> None:
    config_path = bind_config(config_path, run_root, initialize=False)
    config = load_config(config_path)
    seed_everything(config["seed"])
    checkpoint_path = run_root / "student_gt_checkpoint.pt"
    train_receipt = json.loads((run_root / "train_receipt.json").read_text())
    if train_receipt["runtime"]["config_sha256"] != sha256_file(config_path):
        raise RuntimeError("checkpoint receipt was created under a different frozen config")
    if sha256_file(checkpoint_path) != train_receipt["checkpoint_sha256"]:
        raise RuntimeError("checkpoint identity mismatch")
    checkpoint_load_started = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    prepared = torch.load(run_root / "prepared_fit.pt", map_location="cpu", weights_only=False)
    checkpoint_load_seconds = time.perf_counter() - checkpoint_load_started
    if (
        checkpoint["config"] != config
        or checkpoint["global_step"] != config["budget"]["gt_optimizer_steps"]
    ):
        raise RuntimeError("reloaded checkpoint config or global step is incorrect")
    optimizer_validation_started = time.perf_counter()
    model = StreamingStudent()
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    loaded_steps = {
        int(value["step"].item())
        for state_value in optimizer.state.values()
        for key, value in state_value.items()
        if key == "step"
    }
    if loaded_steps != {checkpoint["global_step"]}:
        raise RuntimeError(f"reloaded optimizer counters are incorrect: {loaded_steps}")
    optimizer_validation_seconds = time.perf_counter() - optimizer_validation_started
    del optimizer
    device = require_gpu(config)
    model = model.to(device).eval()
    inference_samples = int(config["verification"]["inference_samples"])
    waveform: Tensor = prepared["waveform"]
    with torch.no_grad():
        warmup_audio = torch.zeros(1, inference_samples, device=device)
        warmup_state = model.initial_state(1, device)
        warmup_started = time.perf_counter()
        model.forward_stream(warmup_audio, warmup_state)
        torch.cuda.synchronize(device)
        warmup_seconds = time.perf_counter() - warmup_started
        del warmup_audio, warmup_state
        torch.cuda.reset_peak_memory_stats(device)
        state = model.initial_state(1, device)
        measured_started = time.perf_counter()
        audio = waveform[:inference_samples].to(device).unsqueeze(0)
        logits, frontiers, _ = model.forward_stream(audio, state)
        probabilities_cpu = torch.sigmoid(logits).cpu()
        torch.cuda.synchronize(device)
        measured_seconds = time.perf_counter() - measured_started
    expected = torch.tensor(train_receipt["inference_replay_expected"]["probabilities"])
    prediction_delta = float((probabilities_cpu - expected).abs().max())
    tolerance = float(config["verification"]["reload_tolerance"])
    if (
        not bool(torch.isfinite(probabilities_cpu).all())
        or probabilities_cpu.shape[2] != 4
        or prediction_delta > tolerance
    ):
        raise RuntimeError("fresh-process checkpoint inference agreement failed")
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    if max(peak_allocated, peak_reserved) > config["resource_guard"]["maximum_expected_peak_bytes"]:
        raise RuntimeError("reload inference exceeded the frozen GPU-memory cap")
    representative_index = next(
        index
        for index, frontier in enumerate(frontiers.tolist())
        if frontier >= model.receptive_field_samples
    )
    atomic_json(
        run_root / "reload_receipt.json",
        {
            "stage": "reload",
            "fresh_python_process": True,
            "optimizer_steps": 0,
            "optimizer_step_counters": sorted(loaded_steps),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_load_seconds_excluded": checkpoint_load_seconds,
            "optimizer_cpu_validation_seconds_excluded": optimizer_validation_seconds,
            "finite_four_independent_probabilities": True,
            "prediction_max_abs_delta": prediction_delta,
            "prediction_tolerance": tolerance,
            "representative_valid_frontier_sample": int(frontiers[representative_index]),
            "representative_valid_probabilities": probabilities_cpu[
                0, representative_index
            ].tolist(),
            "full_path_inference": {
                "definition": "decoder-free student-only 15.2-second block replay: host-to-device mono waveform transfer, GPU fixed log-Mel, causal convolutions, stateful GRU, linear logits, sigmoid, and device-to-host probabilities",
                "not_real_time_admission_cadence": True,
                "input_samples": inference_samples,
                "output_frames": probabilities_cpu.shape[1],
                "warmup_seconds_separate": warmup_seconds,
                "measured_seconds": measured_seconds,
                "audio_seconds": inference_samples / StreamingStudent.sample_rate,
                "synchronized": True,
                "device": device_report(device),
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "peak_scope": "reset after warmup; includes resident model and measured inference only",
            },
            "runtime": runtime_identity(config_path),
        },
    )


def report_stage(config_path: Path, run_root: Path) -> None:
    config_path = bind_config(config_path, run_root, initialize=False)
    config = load_config(config_path)
    backend = json.loads((run_root / "backend_receipt.json").read_text())
    prepared = json.loads((run_root / "prepare_receipt.json").read_text())
    trained = json.loads((run_root / "train_receipt.json").read_text())
    reloaded = json.loads((run_root / "reload_receipt.json").read_text())
    config_sha256 = sha256_file(config_path)
    if any(
        receipt["runtime"]["config_sha256"] != config_sha256
        for receipt in (backend, prepared, trained, reloaded)
    ):
        raise RuntimeError("stage receipts do not share the frozen config identity")
    checkpoint_path = run_root / "student_gt_checkpoint.pt"
    if (
        trained["checkpoint_sha256"] != reloaded["checkpoint_sha256"]
        or sha256_file(checkpoint_path) != trained["checkpoint_sha256"]
    ):
        raise RuntimeError("report checkpoint identities differ")
    result = {
        "schema": "PSEM-ISSUE-164-GT-PROBE-RESULT-1",
        "status": "completed",
        "scope": "bounded GT-only smoke; not quality, generalization, KD, receiver, or production evidence",
        "optimizer_step_accounting": {
            "backend_fixture": backend["optimizer_steps"],
            "gt_student": trained["optimizer_steps"],
            "reload": reloaded["optimizer_steps"],
            "total": backend["optimizer_steps"] + trained["optimizer_steps"],
        },
        "source": prepared["source"],
        "audio_exposure_accounting": {
            "prepared_unique_fit_seconds": config["budget"]["prepared_and_consumed_fit_seconds"],
            "gt_training_consumed_seconds": config["budget"]["prepared_and_consumed_fit_seconds"],
            "partition_and_inference_replays_add_unique_audio_seconds": 0,
            "inference_replay_sample_bounds": config["verification"][
                "inference_source_sample_bounds"
            ],
        },
        "parameter_count": trained["parameter_count"],
        "checkpoint_sha256": trained["checkpoint_sha256"],
        "training": {
            "updates": trained["updates"],
            "loop_seconds_synchronized": trained["training_loop_seconds_synchronized"],
            "timing_scope": trained["training_timing_scope"],
            "gpu": trained["gpu"],
        },
        "streaming_partition_witness": trained["streaming_partition_witness"],
        "reload": reloaded,
        "training_observation_boundary": "few-step training loss is a backend/path diagnostic, not a quality or generalization metric",
        "runtime": runtime_identity(config_path),
    }
    atomic_json(run_root / "result.json", result)
    atomic_json(
        run_root / "report_receipt.json",
        {
            "stage": "report",
            "gpu_or_model_work": False,
            "result_sha256": sha256_file(run_root / "result.json"),
            "checkpoint_sha256": result["checkpoint_sha256"],
            "runtime": runtime_identity(config_path),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("backend", "prepare", "train", "reload", "report"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    stages = {
        "backend": backend_stage,
        "prepare": prepare_stage,
        "train": train_stage,
        "reload": reload_stage,
        "report": report_stage,
    }
    stages[args.stage](args.config.resolve(), args.run_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
