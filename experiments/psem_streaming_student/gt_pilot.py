from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from student import StreamingState, StreamingStudent
from torch import Tensor, nn

from gt_probe import (
    atomic_json,
    canonical_sha256,
    device_report,
    first_appearance_slots,
    local_path,
    parameter_count,
    projected_targets,
    read_wave,
    require_gpu,
    resolve_frozen_source,
    rng_state,
    seed_everything,
    selected_normalized_labels,
    sha256_file,
)

from experiments.psem_relative_occupancy_gate.evaluate import (
    monotonic_boundary_matches,
    weighted_average_precision,
    weighted_binary_confusion,
)

SCHEMA = "PSEM-ISSUE-164-GT-PILOT-1"
CHECKPOINT_SCHEMA = "PSEM-ISSUE-164-GT-PILOT-CHECKPOINT-1"
OUTPUT_SLOTS = 4
HOP = StreamingStudent.output_hop_samples
SAMPLE_RATE = StreamingStudent.sample_rate
RESUME_LOGIT_TOLERANCE = 1.0e-4
COLLARS_MS = (100, 250, 500)
ABA_MAX_SECONDS = 1.0
MATERIAL_AUTHORIZED = False


def bind_pilot_config(config_path: Path, run_root: Path) -> Path:
    frozen = run_root / "frozen_config.json"
    requested = config_path.read_bytes()
    if frozen.exists():
        if frozen.read_bytes() != requested:
            raise RuntimeError("requested config differs from the frozen GT-pilot config")
    else:
        temporary = frozen.with_name(f".{frozen.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(requested)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, frozen)
        finally:
            temporary.unlink(missing_ok=True)
    return frozen


def load_pilot_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise RuntimeError("unexpected GT-pilot config schema")
    if "backend_fixture_optimizer_steps" in value.get("budget", {}) or (
        "prior_failed_gt_optimizer_steps" in value.get("budget", {})
    ):
        raise RuntimeError("old GT-probe step budgets are not authorization for this run")
    root = Path(__file__).resolve().parent
    for relative, expected in value["implementation_sha256"].items():
        actual = sha256_file(root / relative)
        if actual != expected:
            raise RuntimeError(f"implementation identity mismatch for {relative}: {actual}")
    training = value["training"]
    budget = value["budget"]
    if training["seed"] != 20260915:
        raise RuntimeError("pilot seed must be 20260915")
    if training["epochs"] != 10 or training["steps_per_epoch"] != 78:
        raise RuntimeError("pilot must use 10 epochs of 78 chronological chunks")
    if budget["maximum_optimizer_updates"] != 780:
        raise RuntimeError("pilot optimizer cap must be 780")
    if budget["prior_cumulative_optimizer_updates"] != 23:
        raise RuntimeError("prior consumed optimizer updates must remain 23")
    if budget["maximum_cumulative_optimizer_updates"] != 803:
        raise RuntimeError("cumulative optimizer cap must be 803")
    if budget["automatic_retry_after_unknown_partial_update"] is not False:
        raise RuntimeError("automatic retry after unknown partial updates is forbidden")
    if training["student_parameter_count"] != 5940740:
        raise RuntimeError("student parameter count must remain 5940740")
    return value


def pilot_runtime(config_path: Path) -> dict[str, object]:
    root = Path(__file__).resolve().parent
    config = load_pilot_config(config_path)
    return {
        "implementation_base_commit": config["implementation_base_commit"],
        "config_sha256": sha256_file(config_path),
        "implementation_sha256": {
            name: sha256_file(root / name) for name in config["implementation_sha256"]
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
    }


def catalog_sources(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for role in ("FIT", "CAL", "DEV"):
        for source in config["sources"][role]:
            if source["source_id"] in rows:
                raise RuntimeError(f"duplicate source_id {source['source_id']}")
            rows[source["source_id"]] = {**source, "pilot_role": role}
    return rows


def fit_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    chunk = int(config["training"]["chunk_samples"])
    catalog = catalog_sources(config)
    steps: list[dict[str, Any]] = []
    for epoch in range(int(config["training"]["epochs"])):
        for source_id in config["training"]["source_order"]:
            source = catalog[source_id]
            if source["pilot_role"] != "FIT":
                raise RuntimeError(f"{source_id} is not a FIT training source")
            chunks = int(source["prefix_samples"]) // chunk
            if chunks * chunk != int(source["prefix_samples"]):
                raise RuntimeError(f"{source_id} prefix is not an integer chunk count")
            for chunk_index in range(chunks):
                left = chunk_index * chunk
                steps.append(
                    {
                        "epoch": epoch,
                        "source_id": source_id,
                        "chunk_index": chunk_index,
                        "chunks_in_source": chunks,
                        "update_index": len(steps) + 1,
                        "sample_bounds": [left, left + chunk],
                        "source_boundary_after": chunk_index + 1 == chunks,
                    }
                )
    if len(steps) != int(config["training"]["steps_per_epoch"]) * int(
        config["training"]["epochs"]
    ):
        raise RuntimeError("FIT schedule length differs from the frozen 780-step contract")
    return steps


def restore_rng(state: dict[str, Any]) -> None:
    import random

    random.setstate(tuple(state["python"]))
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def append_ledger(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json.loads(line))
    return rows


def ledger_status(rows: list[dict[str, Any]]) -> dict[str, int]:
    completed = 0
    uncertain = 0
    intent = 0
    for row in rows:
        kind = row["kind"]
        index = int(row["update_index"])
        if kind == "completed":
            completed = max(completed, index)
        elif kind == "uncertain":
            uncertain = max(uncertain, index)
        elif kind == "intent":
            intent = max(intent, index)
    return {"completed": completed, "uncertain": uncertain, "intent": intent}


def occupancy_frontiers(prefix_samples: int) -> Tensor:
    return torch.arange(HOP, prefix_samples + 1, HOP, dtype=torch.int64)


def project_source(
    config: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    resolve_frozen_source(config, source)
    normalized, annotation_hashes = selected_normalized_labels(config, source)
    timeline = normalized["raw_normalized_intervals"]
    prefix = int(source["prefix_samples"])
    require_all = source["role"] != "PSEM-STRATEGY-DEV"
    slots = first_appearance_slots(
        timeline, prefix, output_slots=OUTPUT_SLOTS, require_all_slots=require_all
    )
    if require_all and any(slot is None for slot in slots):
        raise RuntimeError(f"{source['source_id']} FIT/CAL slots are incomplete")
    if (not require_all) and all(slot is None for slot in slots):
        raise RuntimeError(f"{source['source_id']} DEV prefix exposes no known speaker")
    model = StreamingStudent()
    frontiers = occupancy_frontiers(prefix)
    targets, validity = projected_targets(
        frontiers, timeline, slots, model.receptive_field_samples
    )
    first_times = {}
    for row in timeline:
        if row["masked_for_activity"] or int(row["start_sample"]) >= prefix:
            continue
        for speaker in row["active_speakers"]:
            first_times[speaker] = min(
                first_times.get(speaker, int(row["start_sample"])), int(row["start_sample"])
            )
    return {
        "source_id": source["source_id"],
        "pilot_role": source.get("pilot_role"),
        "normalized": normalized,
        "annotation_hashes": annotation_hashes,
        "timeline": timeline,
        "slots": slots,
        "first_appearance_samples": first_times,
        "frontiers": frontiers,
        "targets": targets,
        "validity": validity,
        "relation_only_nonlexical_masks": normalized["relation_only_nonlexical_masks"],
        "identity_component": resolve_frozen_source(config, source)["identity_component"],
        "split_component": resolve_frozen_source(config, source)["component"],
    }


def overlap_bin_flags(
    frontiers: Tensor, timeline: list[dict[str, Any]]
) -> dict[str, np.ndarray]:
    physical = np.zeros(frontiers.numel(), dtype=bool)
    mixed_coverage = np.zeros(frontiers.numel(), dtype=bool)
    stable_physical = np.zeros(frontiers.numel(), dtype=bool)
    for frame, frontier_value in enumerate(frontiers.tolist()):
        left = frontier_value - HOP
        right = frontier_value
        clipped: list[tuple[int, int, tuple[str, ...]]] = []
        for row in timeline:
            start = max(left, int(row["start_sample"]))
            end = min(right, int(row["end_sample"]))
            if end > start:
                clipped.append((start, end, tuple(row["active_speakers"])))
        mixed_coverage[frame] = len(clipped) > 1
        if any(len(speakers) >= 2 for _, _, speakers in clipped):
            physical[frame] = True
        else:
            for index, left_span in enumerate(clipped):
                for right_span in clipped[index + 1 :]:
                    shared = min(left_span[1], right_span[1]) - max(left_span[0], right_span[0])
                    if shared > 0 and set(left_span[2]) | set(right_span[2]):
                        if set(left_span[2]) != set(right_span[2]) or len(left_span[2]) >= 2:
                            physical[frame] = True
        if (
            len(clipped) == 1
            and clipped[0][0] == left
            and clipped[0][1] == right
            and len(clipped[0][2]) >= 2
        ):
            stable_physical[frame] = True
    return {
        "physical_overlap": physical,
        "mixed_coverage": mixed_coverage,
        "stable_physical_overlap": stable_physical,
    }


def mask_overlap_frames(
    frontiers: Tensor, masks: list[dict[str, Any]]
) -> Tensor:
    flagged = torch.zeros(frontiers.numel(), dtype=torch.bool)
    for frame, frontier_value in enumerate(frontiers.tolist()):
        left = frontier_value - HOP
        right = frontier_value
        flagged[frame] = any(
            int(mask["start_sample"]) < right and int(mask["end_sample"]) > left
            for mask in masks
        )
    return flagged


def binary_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    validity: np.ndarray,
    threshold: float,
    overlap_flags: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    valid = validity.astype(bool)
    denom = int(valid.sum())
    slot_rows = []
    for slot in range(targets.shape[1]):
        labels = targets[valid, slot] >= threshold
        scores = probabilities[valid, slot]
        weights = np.ones(denom, dtype=np.float64)
        confusion = weighted_binary_confusion(labels, scores, weights, threshold)
        average_precision = weighted_average_precision(labels, scores, weights)
        slot_rows.append(
            {
                "slot": slot,
                "positive_valid_bins": int(labels.sum()),
                "confusion": confusion,
                "average_precision": average_precision,
            }
        )
    gt_count = (targets[valid] >= threshold).sum(axis=1)
    pred_count = (probabilities[valid] >= threshold).sum(axis=1)
    slot_threshold_gt = gt_count >= 2
    slot_threshold_pred = (pred_count >= 2).astype(np.float64)
    overlap = {
        "slot_threshold_overlap_proxy": {
            "not_physical_overlap": True,
            "confounded_by_within_bin_straddle": True,
            "gt_bins": int(slot_threshold_gt.sum()),
            "confusion": weighted_binary_confusion(
                slot_threshold_gt, slot_threshold_pred, np.ones(denom, dtype=np.float64), 0.5
            ),
        }
    }
    if overlap_flags is not None:
        physical = overlap_flags["physical_overlap"][valid]
        mixed = overlap_flags["mixed_coverage"][valid]
        stable = overlap_flags["stable_physical_overlap"][valid]
        confound = slot_threshold_gt & ~physical
        overlap["physical_overlap_from_raw_intervals"] = {
            "gt_bins": int(physical.sum()),
            "stable_physical_overlap_bins": int(stable.sum()),
            "mixed_coverage_valid_bins": int(mixed.sum()),
            "predicted_using_slot_threshold_proxy": True,
            "confusion": weighted_binary_confusion(
                physical, slot_threshold_pred, np.ones(denom, dtype=np.float64), 0.5
            ),
        }
        overlap["confounded_slot_threshold_without_physical_overlap_bins"] = int(confound.sum())
        overlap["denominators"] = {
            "valid_bins": denom,
            "physical_overlap_valid_bins": int(physical.sum()),
            "stable_physical_overlap_valid_bins": int(stable.sum()),
            "mixed_coverage_valid_bins": int(mixed.sum()),
            "slot_threshold_overlap_proxy_gt_bins": int(slot_threshold_gt.sum()),
            "confounded_straddle_or_mixed_bins": int(confound.sum()),
        }
    solo = gt_count == 1
    per_slot_recall = []
    for slot in range(targets.shape[1]):
        relevant = solo & (targets[valid, slot] >= threshold)
        if not bool(relevant.any()):
            per_slot_recall.append(None)
            continue
        predicted_solo = (pred_count == 1) & (probabilities[valid, slot] >= threshold)
        per_slot_recall.append(float(predicted_solo[relevant].mean()))
    present = [value for value in per_slot_recall if value is not None]
    balanced = float(sum(present) / len(present)) if present else None
    histogram = {str(count): int((pred_count == count).sum()) for count in range(OUTPUT_SLOTS + 1)}
    return {
        "valid_bins": denom,
        "per_slot": slot_rows,
        "solo_speaker_balanced_accuracy": balanced,
        "solo_per_slot_recall": per_slot_recall,
        "overlap": overlap,
        "predicted_active_count": histogram,
        "all_silence_valid_fraction": float((pred_count == 0).mean()) if denom else None,
        "one_slot_collapse": bool(
            denom > 0 and (probabilities[valid] >= threshold).any() and
            ((probabilities[valid] >= threshold).sum(axis=0) > 0).sum() == 1
        ),
    }


def active_set_events(
    occupancy: np.ndarray, frontiers: np.ndarray, validity: np.ndarray, threshold: float
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous: frozenset[int] | None = None
    previous_index = None
    for index, valid in enumerate(validity.tolist()):
        if not valid:
            continue
        current = frozenset(
            slot for slot, value in enumerate(occupancy[index]) if value >= threshold
        )
        if previous is not None and current != previous:
            events.append(
                {
                    "frame": index,
                    "sample": int(frontiers[index] - HOP),
                    "before": sorted(previous),
                    "after": sorted(current),
                    "previous_frame": previous_index,
                }
            )
        previous = current
        previous_index = index
    return events


def match_event_rates(
    predicted: list[dict[str, Any]], reference: list[dict[str, Any]], collar_ms: int
) -> dict[str, Any]:
    tolerance = int(round(collar_ms * SAMPLE_RATE / 1000.0))
    predicted_samples = [row["sample"] for row in predicted]
    reference_samples = [row["sample"] for row in reference]
    matches = monotonic_boundary_matches(predicted_samples, reference_samples, tolerance)
    matched_pred = {left for left, _ in matches}
    matched_ref = {right for _, right in matches}
    false_count = len(predicted) - len(matched_pred)
    missed_count = len(reference) - len(matched_ref)
    return {
        "collar_ms": collar_ms,
        "tolerance_samples": tolerance,
        "predicted_events": len(predicted),
        "reference_events": len(reference),
        "matched": len(matches),
        "false_events": false_count,
        "missed_events": missed_count,
        "false_rate": (false_count / len(predicted)) if predicted else None,
        "missed_rate": (missed_count / len(reference)) if reference else None,
        "geometry": "80ms frame-grid active-set transitions; one-to-one sample matching",
        "not_fixed156_policy_admission": True,
    }


def solo_aba_interruptions(
    occupancy: np.ndarray, validity: np.ndarray, threshold: float, hop_samples: int
) -> list[dict[str, Any]]:
    runs: list[tuple[int, int, int]] = []
    start = None
    speaker = None
    for index, valid in enumerate(validity.tolist()):
        current = None
        if valid:
            active = [
                slot for slot, value in enumerate(occupancy[index]) if value >= threshold
            ]
            if len(active) == 1:
                current = active[0]
        if start is not None and current != speaker:
            runs.append((start, index, speaker))
            start = None
            speaker = None
        if current is not None and start is None:
            start = index
            speaker = current
    if start is not None and speaker is not None:
        runs.append((start, occupancy.shape[0], speaker))
    found: list[dict[str, Any]] = []
    limit = int(ABA_MAX_SECONDS * SAMPLE_RATE)
    for left, middle, right in zip(runs, runs[1:], runs[2:]):
        if left[1] != middle[0] or middle[1] != right[0]:
            continue
        if left[2] == right[2] != middle[2]:
            duration = (middle[1] - middle[0]) * hop_samples
            if duration <= limit:
                found.append(
                    {
                        "a_slot": left[2],
                        "b_slot": middle[2],
                        "b_frames": middle[1] - middle[0],
                        "b_samples": duration,
                    }
                )
    return found


def permutation_oracle(
    logits: Tensor, targets: Tensor, validity: Tensor
) -> dict[str, Any]:
    best = None
    for permutation in itertools.permutations(range(OUTPUT_SLOTS)):
        permuted = logits[:, list(permutation)]
        value = float(
            nn.functional.binary_cross_entropy_with_logits(
                permuted[validity], targets[validity]
            )
        )
        row = {"permutation": list(permutation), "masked_bce": value}
        if best is None or value < best["masked_bce"]:
            best = row
    return {
        "role": "source_global_oracle_diagnostic_only",
        "not_used_for_inference_or_selection": True,
        "best": best,
        "permutation_count": math.factorial(OUTPUT_SLOTS),
    }


def score_outputs(
    logits: Tensor,
    targets: Tensor,
    validity: Tensor,
    frontiers: Tensor,
    masks: list[dict[str, Any]],
    threshold: float,
    timeline: list[dict[str, Any]],
) -> dict[str, Any]:
    if not bool(validity.any()):
        raise RuntimeError("evaluation has no valid bins")
    probabilities = torch.sigmoid(logits)
    bce = float(
        nn.functional.binary_cross_entropy_with_logits(logits[validity], targets[validity])
    )
    brier = float(((probabilities[validity] - targets[validity]) ** 2).mean())
    relation = mask_overlap_frames(frontiers, masks)
    exposed = validity & ~relation
    flags = overlap_bin_flags(frontiers, timeline)
    canonical = binary_metrics(
        targets.numpy(), probabilities.numpy(), validity.numpy(), threshold, flags
    )
    aligned = permutation_oracle(logits, targets, validity)
    perm = aligned["best"]["permutation"]
    aligned_probabilities = probabilities[:, perm]
    aligned_metrics = binary_metrics(
        targets.numpy(),
        aligned_probabilities.numpy(),
        validity.numpy(),
        threshold,
        flags,
    )
    pred_events = active_set_events(
        probabilities.numpy(), frontiers.numpy(), validity.numpy(), threshold
    )
    ref_events = active_set_events(
        targets.numpy(), frontiers.numpy(), validity.numpy(), threshold
    )
    event_rates = [match_event_rates(pred_events, ref_events, collar) for collar in COLLARS_MS]
    aba_ref = solo_aba_interruptions(targets.numpy(), validity.numpy(), threshold, HOP)
    aba_pred = solo_aba_interruptions(probabilities.numpy(), validity.numpy(), threshold, HOP)
    return {
        "canonical_masked_bce": bce,
        "canonical_brier": brier,
        "valid_bins": int(validity.sum()),
        "relation_mask_overlapping_valid_bins": int((validity & relation).sum()),
        "relation_mask_excluded_valid_bins": int(exposed.sum()),
        "relation_masks_do_not_erase_activity_targets": True,
        "canonical": canonical,
        "source_global_aligned": {
            **aligned,
            "metrics": aligned_metrics,
            "aligned_masked_bce": aligned["best"]["masked_bce"],
        },
        "event_proxy": {
            "predicted_events": len(pred_events),
            "reference_events": len(ref_events),
            "collars": event_rates,
            "solo_aba_interruptions_le_1s": {
                "reference": len(aba_ref),
                "predicted": len(aba_pred),
            },
            "limitations": "frame-grid 80ms active-set transitions; sub-frame timing is not resolved; not fixed156-policy admission evaluation",
        },
        "operating_point": threshold,
    }


def constant_prior_logits(fit_targets: Tensor, fit_validity: Tensor, frames: int) -> Tensor:
    means = []
    for slot in range(OUTPUT_SLOTS):
        values = fit_targets[fit_validity, slot]
        means.append(float(values.mean()) if values.numel() else 0.0)
    probability = torch.tensor(means, dtype=torch.float32).clamp(1.0e-6, 1.0 - 1.0e-6)
    logit = torch.log(probability) - torch.log1p(-probability)
    return logit.unsqueeze(0).expand(frames, -1).contiguous(), means


def verify_pilot_identities(config: dict[str, Any]) -> dict[str, Any]:
    catalog = catalog_sources(config)
    locked = set(config["locked_identity_closure"]["source_ids"])
    if sorted(locked) != sorted(
        f"ami_{meeting}" for meeting in config["locked_identity_closure"]["meetings"]
    ):
        raise RuntimeError("locked closure source_ids do not match the 12 locked meetings")
    components: dict[str, str] = {}
    speakers: dict[str, set[str]] = {}
    rows = []
    for source in catalog.values():
        resolved = resolve_frozen_source(config, source)
        identity = resolved["identity_component"]
        if source["source_id"] in locked or any(
            item in locked for item in identity["source_ids"]
        ):
            raise RuntimeError(f"{source['source_id']} intersects locked identity closure")
        known = {
            reason["value"]
            for reason in identity["shared_identity_reasons"]
            if reason["axis"] == "known_speaker_identity"
        }
        known.update(f"AMI:{speaker}" for speaker in resolved["source_row"]["speaker_ids"])
        components[source["source_id"]] = identity["component_id"]
        speakers[source["source_id"]] = known
        rows.append(
            {
                "source_id": source["source_id"],
                "pilot_role": source["pilot_role"],
                "split_role": resolved["component"]["role"],
                "component_id": identity["component_id"],
                "component_source_ids": list(identity["source_ids"]),
                "speaker_ids": list(resolved["source_row"]["speaker_ids"]),
                "waveform_sha256": source["waveform_sha256"],
                "label_result_sha256": source["label_result_sha256"],
                "reference_rttm_sha256": source["reference_rttm_sha256"],
            }
        )
    fit_ids = [source["source_id"] for source in config["sources"]["FIT"]]
    cal_ids = [source["source_id"] for source in config["sources"]["CAL"]]
    dev_ids = [source["source_id"] for source in config["sources"]["DEV"]]
    groups = {
        "FIT": set(fit_ids),
        "CAL": set(cal_ids),
        "DEV": set(dev_ids),
    }
    for left, right in itertools.combinations(groups, 2):
        if {components[item] for item in groups[left]} & {
            components[item] for item in groups[right]
        }:
            raise RuntimeError(f"{left}/{right} share an identity component")
        left_speakers = set().union(*(speakers[item] for item in groups[left]))
        right_speakers = set().union(*(speakers[item] for item in groups[right]))
        if left_speakers & right_speakers:
            raise RuntimeError(f"{left}/{right} share known speaker identities")
    if len({components[item] for item in fit_ids}) != len(fit_ids):
        raise RuntimeError("FIT sources are not component-disjoint")
    return {
        "sources": rows,
        "locked_identity_closure": config["locked_identity_closure"],
        "fit_components": {item: components[item] for item in fit_ids},
        "cal_components": {item: components[item] for item in cal_ids},
        "dev_components": {item: components[item] for item in dev_ids},
    }


def synthetic_metric_probes() -> dict[str, Any]:
    hop = HOP
    receptive = StreamingStudent().receptive_field_samples
    timeline = [
        {
            "start_sample": 0,
            "end_sample": 4 * hop,
            "active_speakers": ["A"],
            "masked_for_activity": False,
        },
        {
            "start_sample": 4 * hop,
            "end_sample": 6 * hop,
            "active_speakers": ["A", "B"],
            "masked_for_activity": False,
        },
        {
            "start_sample": 6 * hop,
            "end_sample": 8 * hop,
            "active_speakers": ["A"],
            "masked_for_activity": False,
        },
        {
            "start_sample": 8 * hop,
            "end_sample": 8 * hop + hop // 2,
            "active_speakers": ["A"],
            "masked_for_activity": False,
        },
        {
            "start_sample": 8 * hop + hop // 2,
            "end_sample": 9 * hop,
            "active_speakers": ["B"],
            "masked_for_activity": False,
        },
        {
            "start_sample": 10 * hop,
            "end_sample": 12 * hop,
            "active_speakers": ["A"],
            "masked_for_activity": True,
        },
        {
            "start_sample": 12 * hop,
            "end_sample": 14 * hop,
            "active_speakers": ["A"],
            "masked_for_activity": False,
        },
        {
            "start_sample": 14 * hop,
            "end_sample": 16 * hop,
            "active_speakers": ["B"],
            "masked_for_activity": False,
        },
        {
            "start_sample": 16 * hop,
            "end_sample": 20 * hop,
            "active_speakers": ["A"],
            "masked_for_activity": False,
        },
    ]
    relation_masks = [{"start_sample": 12 * hop, "end_sample": 13 * hop}]
    slots = first_appearance_slots(
        timeline, 20 * hop, output_slots=OUTPUT_SLOTS, require_all_slots=False
    )
    if slots[:2] != ["A", "B"] or slots[2] is not None or slots[3] is not None:
        raise RuntimeError("padded first-appearance slots failed")
    frontiers = occupancy_frontiers(20 * hop)
    targets, validity = projected_targets(frontiers, timeline, slots, receptive)
    if bool(validity[:3].any()):
        raise RuntimeError("first three warmup bins must be invalid")
    overlap_frame = 4
    if not math.isclose(float(targets[overlap_frame, 0]), 1.0) or not math.isclose(
        float(targets[overlap_frame, 1]), 1.0
    ):
        raise RuntimeError("overlap occupancy was not preserved")
    flags = overlap_bin_flags(frontiers, timeline)
    straddle_frame = 8
    if not math.isclose(float(targets[straddle_frame, 0]), 0.5) or not math.isclose(
        float(targets[straddle_frame, 1]), 0.5
    ):
        raise RuntimeError("straddle occupancy targets were not preserved")
    if bool(flags["physical_overlap"][straddle_frame]):
        raise RuntimeError("within-bin A-then-B straddle was labeled physical overlap")
    if not bool(flags["mixed_coverage"][straddle_frame]):
        raise RuntimeError("straddle bin was not marked mixed coverage")
    if not bool(flags["physical_overlap"][overlap_frame]) or not bool(
        flags["stable_physical_overlap"][overlap_frame]
    ):
        raise RuntimeError("simultaneous overlap bin was not labeled from raw intervals")
    if bool(flags["mixed_coverage"][overlap_frame]):
        raise RuntimeError("stable overlap bin was marked mixed coverage")
    slot_threshold_straddle = bool(
        (targets[straddle_frame] >= 0.5).sum() >= 2
    )
    if not slot_threshold_straddle:
        raise RuntimeError("slot-threshold overlap proxy missed the straddle confound")
    if bool(validity[10]):
        raise RuntimeError("ambiguous activity bin was not invalidated")
    relation = mask_overlap_frames(frontiers, relation_masks)
    if not bool(validity[12]) or not bool(relation[12]):
        raise RuntimeError("relation-only mask erased or failed to flag activity")
    if float(targets[12:, 2].sum() + targets[12:, 3].sum()) != 0.0:
        raise RuntimeError("unused padded slots are not zero")
    aba = solo_aba_interruptions(targets.numpy(), validity.numpy(), 0.5, hop)
    if not aba:
        raise RuntimeError("solo A-B-A interruption probe failed")
    logits = torch.log(targets.clamp(1.0e-6, 1.0 - 1.0e-6)) - torch.log1p(
        -targets.clamp(1.0e-6, 1.0 - 1.0e-6)
    )
    swapped = logits[:, [1, 0, 2, 3]]
    oracle = permutation_oracle(swapped, targets, validity)
    if oracle["best"]["permutation"] != [1, 0, 2, 3]:
        raise RuntimeError("source-global permutation oracle did not recover the swap")
    pred_events = active_set_events(
        torch.sigmoid(swapped).numpy(), frontiers.numpy(), validity.numpy(), 0.5
    )
    ref_events = active_set_events(targets.numpy(), frontiers.numpy(), validity.numpy(), 0.5)
    matched = match_event_rates(ref_events, ref_events, 100)
    if matched["matched"] != len(ref_events) or matched["false_events"] != 0:
        raise RuntimeError("self-matched event proxy failed")
    return {
        "padded_slots": slots,
        "invalid_warmup_bins": 3,
        "overlap_preserved": True,
        "ambiguous_invalid": True,
        "relation_mask_does_not_erase_activity": True,
        "solo_aba_interruptions": len(aba),
        "permutation_recovered": oracle["best"]["permutation"],
        "self_matched_events": matched,
        "unused_event_proxy_on_swapped_logits": len(pred_events),
        "straddle_vs_overlap": {
            "activity_targets_unchanged": True,
            "straddle_bin_occupancy": [0.5, 0.5],
            "straddle_physical_overlap": False,
            "straddle_slot_threshold_proxy": True,
            "simultaneous_bin_physical_overlap": True,
            "simultaneous_bin_stable": True,
        },
    }


def check_stage(config_path: Path, run_root: Path) -> None:
    config_path = bind_pilot_config(config_path, run_root)
    config = load_pilot_config(config_path)
    identities = verify_pilot_identities(config)
    schedule = fit_schedule(config)
    if schedule[-1]["update_index"] != 780:
        raise RuntimeError("schedule does not end at update 780")
    per_epoch = {}
    for row in schedule:
        if row["epoch"] == 0:
            per_epoch[row["source_id"]] = per_epoch.get(row["source_id"], 0) + 1
    expected_chunks = {
        source["source_id"]: int(source["prefix_samples"])
        // int(config["training"]["chunk_samples"])
        for source in config["sources"]["FIT"]
    }
    if per_epoch != expected_chunks:
        raise RuntimeError(f"epoch-0 chunk counts differ: {per_epoch}")
    projections = {}
    for source in catalog_sources(config).values():
        projections[source["source_id"]] = project_source(config, source)
    es2006 = projections["ami_ES2006a"]
    fourth = sorted(es2006["first_appearance_samples"].items(), key=lambda item: item[1])[3]
    expected_fourth = int(round(float(config["sources"]["confirmed_es2006a_fourth_speaker_seconds"]) * SAMPLE_RATE))
    if abs(fourth[1] - expected_fourth) > SAMPLE_RATE:
        raise RuntimeError(
            f"ES2006a fourth speaker first appearance {fourth} differs from frozen 306.418s"
        )
    if any(slot is None for slot in es2006["slots"]):
        raise RuntimeError("ES2006a prefix does not expose four speakers")
    en = projections["ami_EN2009d"]
    known = [slot for slot in en["slots"] if slot is not None]
    if len(known) != 2 or len(en["slots"]) != OUTPUT_SLOTS:
        raise RuntimeError("EN2009d prefix-only mapping must keep four columns with two speakers")
    if not torch.equal(en["targets"][:, 2:], torch.zeros_like(en["targets"][:, 2:])):
        raise RuntimeError("EN2009d unused columns are not zero")
    if not bool(en["validity"].any()):
        raise RuntimeError("EN2009d retained no valid complete bins")
    for source_id in ("ami_ES2005a", "ami_ES2007a", "ami_ES2008a", "ami_ES2010a"):
        if any(slot is None for slot in projections[source_id]["slots"]):
            raise RuntimeError(f"{source_id} does not expose four speakers")
    student = StreamingStudent()
    count = parameter_count(student)
    if count != config["training"]["student_parameter_count"]:
        raise RuntimeError(f"student parameter count differs: {count}")
    probes = synthetic_metric_probes()
    slot_report = {
        source_id: {
            "slots": projected["slots"],
            "first_appearance_samples": projected["first_appearance_samples"],
            "valid_bins": int(projected["validity"].sum()),
            "frames": int(projected["frontiers"].numel()),
            "relation_only_nonlexical_mask_count": len(
                projected["relation_only_nonlexical_masks"]
            ),
        }
        for source_id, projected in projections.items()
    }
    atomic_json(
        run_root / "check_receipt.json",
        {
            "stage": "check",
            "optimizer_steps": 0,
            "waveform_read": False,
            "identities": identities,
            "schedule": {
                "steps": len(schedule),
                "epoch_0_chunks": per_epoch,
                "maximum_optimizer_updates": 780,
            },
            "slots": slot_report,
            "es2006a_fourth_speaker": {
                "speaker": fourth[0],
                "first_sample": fourth[1],
                "first_seconds": fourth[1] / SAMPLE_RATE,
                "frozen_seconds": config["sources"]["confirmed_es2006a_fourth_speaker_seconds"],
            },
            "synthetic_metric_probes": probes,
            "student_parameter_count": count,
            "runtime": pilot_runtime(config_path),
        },
    )


def require_authorization() -> None:
    if not MATERIAL_AUTHORIZED:
        raise RuntimeError(
            "material prepare/train/evaluate is blocked until Director GO passes --authorize-material"
        )



def load_or_project(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    return project_source(config, source)


def prepare_stage(config_path: Path, run_root: Path) -> None:
    config_path = bind_pilot_config(config_path, run_root)
    config = load_pilot_config(config_path)
    require_authorization()
    identities = verify_pilot_identities(config)
    prepared: dict[str, Any] = {"sources": {}, "identities": identities}
    fit_targets = []
    fit_validity = []
    for source in catalog_sources(config).values():
        resolve_frozen_source(config, source)
        projection = project_source(config, source)
        path = local_path(source["waveform"])
        if sha256_file(path) != source["waveform_sha256"]:
            raise RuntimeError(f"waveform identity mismatch for {source['source_id']}")
        waveform, geometry = read_wave(
            path,
            int(source["prefix_samples"]),
            SAMPLE_RATE,
            int(source["expected_channels"]),
        )
        payload = {
            "waveform": waveform,
            "targets": projection["targets"],
            "validity": projection["validity"],
            "frontiers": projection["frontiers"],
            "slots": projection["slots"],
            "timeline": projection["timeline"],
            "relation_only_nonlexical_masks": projection["relation_only_nonlexical_masks"],
            "annotation_hashes": projection["annotation_hashes"],
            "geometry": geometry,
            "source": source,
        }
        prepared["sources"][source["source_id"]] = payload
        if source.get("pilot_role") == "FIT":
            fit_targets.append(projection["targets"])
            fit_validity.append(projection["validity"])
    stacked_targets = torch.cat(fit_targets, dim=0)
    stacked_validity = torch.cat(fit_validity, dim=0)
    _, prior_means = constant_prior_logits(stacked_targets, stacked_validity, 1)
    prepared["fit_constant_prior"] = prior_means
    prepared_path = run_root / "prepared_sources.pt"
    atomic_torch_save(prepared_path, prepared)
    atomic_json(
        run_root / "prepare_receipt.json",
        {
            "stage": "prepare",
            "optimizer_steps": 0,
            "prepared_sha256": sha256_file(prepared_path),
            "identities": identities,
            "fit_constant_prior": prior_means,
            "source_ids": sorted(prepared["sources"]),
            "runtime": pilot_runtime(config_path),
        },
    )


def load_prepared(run_root: Path, config_path: Path) -> dict[str, Any]:
    prepared_path = run_root / "prepared_sources.pt"
    receipt = json.loads((run_root / "prepare_receipt.json").read_text(encoding="utf-8"))
    if receipt["runtime"]["config_sha256"] != sha256_file(config_path):
        raise RuntimeError("prepared inputs were created under a different frozen config")
    if sha256_file(prepared_path) != receipt["prepared_sha256"]:
        raise RuntimeError("prepared input identity mismatch")
    return torch.load(prepared_path, map_location="cpu", weights_only=False)


def stream_source(
    model: StreamingStudent, waveform: Tensor, device: torch.device, chunk_samples: int
) -> tuple[Tensor, Tensor]:
    state = model.initial_state(1, device)
    logits = []
    frontiers = []
    offset = 0
    while offset < waveform.numel():
        right = min(offset + chunk_samples, waveform.numel())
        chunk = waveform[offset:right].to(device).unsqueeze(0)
        values, edges, state = model.forward_stream(chunk, state)
        logits.append(values.squeeze(0).detach().cpu())
        frontiers.append(edges.detach().cpu())
        state = state.detached()
        offset = right
    return torch.cat(logits, dim=0), torch.cat(frontiers)


def evaluate_model_on_sources(
    model: StreamingStudent,
    prepared: dict[str, Any],
    source_ids: list[str],
    device: torch.device,
    chunk_samples: int,
    threshold: float,
    label: str,
) -> dict[str, Any]:
    model.eval()
    results = {}
    with torch.no_grad():
        for source_id in source_ids:
            payload = prepared["sources"][source_id]
            logits, frontiers = stream_source(
                model, payload["waveform"], device, chunk_samples
            )
            if not torch.equal(frontiers, payload["frontiers"]):
                raise RuntimeError(f"{source_id} evaluation frontiers differ")
            results[source_id] = {
                "label": label,
                "slots": payload["slots"],
                **score_outputs(
                    logits,
                    payload["targets"],
                    payload["validity"],
                    payload["frontiers"],
                    payload["relation_only_nonlexical_masks"],
                    threshold,
                    payload["timeline"],
                ),
            }
    return results


def build_fresh_model(config: dict[str, Any], device: torch.device) -> StreamingStudent:
    seed_everything(int(config["training"]["seed"]))
    model = StreamingStudent().to(device)
    if parameter_count(model) != config["training"]["student_parameter_count"]:
        raise RuntimeError("fresh student parameter count differs")
    if next(model.parameters()).dtype != torch.float32:
        raise RuntimeError("student precision must be float32")
    return model


def evaluate_stage(config_path: Path, run_root: Path, which: str) -> None:
    config_path = bind_pilot_config(config_path, run_root)
    config = load_pilot_config(config_path)
    require_authorization()
    prepared = load_prepared(run_root, config_path)
    device = require_gpu(config)
    chunk = int(config["training"]["chunk_samples"])
    threshold = float(config["evaluation"]["operating_point"])
    if which == "initial":
        model = build_fresh_model(config, device)
        fit_ids = [source["source_id"] for source in config["sources"]["FIT"]]
        cal_ids = [source["source_id"] for source in config["sources"]["CAL"]]
        dev_ids = [source["source_id"] for source in config["sources"]["DEV"]]
        untrained = evaluate_model_on_sources(
            model, prepared, fit_ids + cal_ids + dev_ids, device, chunk, threshold, "untrained"
        )
        prior_rows = {}
        means = torch.tensor(prepared["fit_constant_prior"], dtype=torch.float32)
        probability = means.clamp(1.0e-6, 1.0 - 1.0e-6)
        logit = torch.log(probability) - torch.log1p(-probability)
        for source_id in fit_ids + cal_ids + dev_ids:
            payload = prepared["sources"][source_id]
            logits = logit.unsqueeze(0).expand(payload["targets"].shape[0], -1).contiguous()
            prior_rows[source_id] = {
                "label": "fit_constant_prior",
                "prior_means": prepared["fit_constant_prior"],
                "slots": payload["slots"],
                **score_outputs(
                    logits,
                    payload["targets"],
                    payload["validity"],
                    payload["frontiers"],
                    payload["relation_only_nonlexical_masks"],
                    threshold,
                    payload["timeline"],
                ),
            }
        atomic_json(
            run_root / "initial_eval.json",
            {
                "stage": "evaluate_initial",
                "untrained": untrained,
                "fit_constant_prior": prior_rows,
                "runtime": pilot_runtime(config_path),
            },
        )
        return
    if which == "cal":
        rows = {}
        for step in config["evaluation"]["calibration_checkpoint_steps"]:
            path = run_root / f"checkpoint_step_{step}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload["schema"] != CHECKPOINT_SCHEMA or payload["completed_updates"] != step:
                raise RuntimeError(f"CAL candidate {step} identity is incorrect")
            model = StreamingStudent().to(device)
            model.load_state_dict(payload["model"])
            rows[str(step)] = {
                "checkpoint_sha256": sha256_file(path),
                "cal": evaluate_model_on_sources(
                    model,
                    prepared,
                    [source["source_id"] for source in config["sources"]["CAL"]],
                    device,
                    chunk,
                    threshold,
                    f"cal_step_{step}",
                ),
            }
        selected = None
        for step in config["evaluation"]["calibration_checkpoint_steps"]:
            bce = rows[str(step)]["cal"][config["sources"]["CAL"][0]["source_id"]][
                "canonical_masked_bce"
            ]
            candidate = {"step": step, "cal_masked_bce": bce, **rows[str(step)]}
            if selected is None or bce < selected["cal_masked_bce"]:
                selected = candidate
        atomic_json(
            run_root / "cal_selection.json",
            {
                "stage": "evaluate_cal",
                "candidates": rows,
                "selected": {
                    "step": selected["step"],
                    "cal_masked_bce": selected["cal_masked_bce"],
                    "checkpoint_sha256": selected["checkpoint_sha256"],
                    "rule": "lowest canonical masked BCE on CAL only; earliest step on exact ties",
                },
                "runtime": pilot_runtime(config_path),
            },
        )
        return
    if which == "final":
        selection = json.loads((run_root / "cal_selection.json").read_text(encoding="utf-8"))
        step = selection["selected"]["step"]
        path = run_root / f"checkpoint_step_{step}.pt"
        if sha256_file(path) != selection["selected"]["checkpoint_sha256"]:
            raise RuntimeError("selected checkpoint identity changed before final evaluation")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = StreamingStudent().to(device)
        model.load_state_dict(payload["model"])
        fit_ids = [source["source_id"] for source in config["sources"]["FIT"]]
        dev_ids = [source["source_id"] for source in config["sources"]["DEV"]]
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        scored = evaluate_model_on_sources(
            model, prepared, fit_ids + dev_ids, device, chunk, threshold, "selected_final"
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        atomic_json(
            run_root / "final_eval.json",
            {
                "stage": "evaluate_final",
                "selected_step": step,
                "checkpoint_sha256": sha256_file(path),
                "scores": scored,
                "inference_cost": {
                    "elapsed_seconds_synchronized": elapsed,
                    "scope": "warm selected-checkpoint source-zero prefix replay over FIT+DEV after model load; excludes training; not live 80ms admission or matched teacher cost",
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    "device": device_report(device),
                },
                "runtime": pilot_runtime(config_path),
            },
        )
        return
    raise RuntimeError(f"unknown evaluation selector {which}")



def optimizer_step_count(optimizer: torch.optim.Optimizer) -> set[int]:
    return {
        int(value.item())
        for state_value in optimizer.state.values()
        for key, value in state_value.items()
        if key == "step"
    }


def save_checkpoint(
    path: Path,
    config: dict[str, Any],
    model: StreamingStudent,
    optimizer: torch.optim.Optimizer,
    state: StreamingState | None,
    completed: int,
    cursor: dict[str, Any] | None,
    witness: dict[str, Any] | None,
) -> str:
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config_schema": config["schema"],
        "completed_updates": completed,
        "rng": rng_state(),
        "streaming_state": None if state is None else state.cpu_dict(),
        "schedule_cursor": cursor,
        "resume_witness": witness,
    }
    atomic_torch_save(path, payload)
    return sha256_file(path)


def refuse_stale_checkpoint(checkpoint: dict[str, Any], ledger_rows: list[dict[str, Any]]) -> None:
    status = ledger_status(ledger_rows)
    completed = int(checkpoint["completed_updates"])
    if status["completed"] > completed or status["uncertain"] > completed:
        raise RuntimeError(
            "refusing stale checkpoint replay; ledger completed/uncertain updates are newer"
        )
    if status["intent"] > completed and status["completed"] < status["intent"]:
        raise RuntimeError("in-flight uncertain update is charged; automatic retry is forbidden")


def next_chunk_logits(
    model: StreamingStudent,
    state: StreamingState,
    waveform: Tensor,
    bounds: list[int],
    device: torch.device,
) -> Tensor:
    model.eval()
    with torch.no_grad():
        audio = waveform[bounds[0] : bounds[1]].to(device).unsqueeze(0)
        logits, _, _ = model.forward_stream(audio, state)
    model.train()
    return logits.squeeze(0).detach().cpu()


def train_stage(config_path: Path, run_root: Path, *, resume: bool, pause_after: int | None) -> None:
    config_path = bind_pilot_config(config_path, run_root)
    config = load_pilot_config(config_path)
    require_authorization()
    prepared = load_prepared(run_root, config_path)
    schedule = fit_schedule(config)
    device = require_gpu(config)
    ledger_path = run_root / "update_ledger.jsonl"
    accounting_path = run_root / "accounting.json"
    latest_path = run_root / "checkpoint_latest.pt"
    started = time.perf_counter()
    deadline = float(config["resource_guard"]["per_run_wall_seconds"])
    max_reserved = int(config["resource_guard"]["max_gpu_allocator_reserved_bytes"])
    chunk = int(config["training"]["chunk_samples"])
    if resume:
        if not latest_path.exists():
            raise RuntimeError("resume requested without checkpoint_latest.pt")
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        refuse_stale_checkpoint(checkpoint, read_ledger(ledger_path))
        seed_everything(int(config["training"]["seed"]))
        model = StreamingStudent().to(device)
        model.load_state_dict(checkpoint["model"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["optimizer"]["learning_rate"]),
            weight_decay=float(config["training"]["optimizer"]["weight_decay"]),
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng(checkpoint["rng"])
        completed = int(checkpoint["completed_updates"])
        cursor = checkpoint["schedule_cursor"]
        if cursor is not None:
            witness = checkpoint.get("resume_witness")
            if witness is None:
                raise RuntimeError("resume checkpoint lacks a no-update next-chunk witness")
            same_stream = witness.get("source_id", cursor["source_id"]) == cursor[
                "source_id"
            ] and witness.get("epoch", cursor["epoch"]) == cursor["epoch"]
            if same_stream:
                state_for_witness = model.state_from_dict(checkpoint["streaming_state"], device)
            else:
                state_for_witness = model.initial_state(1, device)
            witness_payload = prepared["sources"][
                witness.get("source_id", cursor["source_id"])
            ]
            current = next_chunk_logits(
                model,
                state_for_witness,
                witness_payload["waveform"],
                witness["sample_bounds"],
                device,
            )
            expected = witness["logits"]
            delta = float((current - expected).abs().max())
            if delta > RESUME_LOGIT_TOLERANCE:
                raise RuntimeError(
                    f"resume pre-update logits differ from witness by {delta} > {RESUME_LOGIT_TOLERANCE}"
                )
            atomic_json(
                run_root / "resume_witness.json",
                {
                    "completed_updates": completed,
                    "max_abs_logit_delta": delta,
                    "tolerance": RESUME_LOGIT_TOLERANCE,
                    "precision": "float32",
                    "extra_optimizer_updates": 0,
                },
            )
            state = model.state_from_dict(checkpoint["streaming_state"], device)
        else:
            state = model.initial_state(1, device)
    else:
        if latest_path.exists():
            raise RuntimeError("refusing to start a fresh train over an existing latest checkpoint")
        model = build_fresh_model(config, device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["optimizer"]["learning_rate"]),
            weight_decay=float(config["training"]["optimizer"]["weight_decay"]),
        )
        completed = 0
        state = None
        cursor = None
        save_checkpoint(latest_path, config, model, optimizer, None, 0, None, None)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    records = []
    current_source = None if cursor is None else cursor["source_id"]
    current_epoch = None if cursor is None else cursor["epoch"]
    if cursor is not None:
        payload = prepared["sources"][cursor["source_id"]]
        state = model.state_from_dict(checkpoint["streaming_state"], device)
    for row in schedule[completed:]:
        if time.perf_counter() - started > deadline:
            raise RuntimeError("per-run wall cap exhausted")
        if (
            state is None
            or current_source != row["source_id"]
            or current_epoch != row["epoch"]
        ):
            state = model.initial_state(1, device)
            current_source = row["source_id"]
            current_epoch = row["epoch"]
        payload = prepared["sources"][row["source_id"]]
        left, right = row["sample_bounds"]
        append_ledger(
            ledger_path,
            {
                "kind": "intent",
                "update_index": row["update_index"],
                "source_id": row["source_id"],
                "epoch": row["epoch"],
                "chunk_index": row["chunk_index"],
            },
        )
        try:
            audio = payload["waveform"][left:right].to(device).unsqueeze(0)
            logits, frontiers, state = model.forward_stream(audio, state)
            frame_left = row["chunk_index"] * int(config["training"]["frames_per_update"])
            frame_right = frame_left + int(config["training"]["frames_per_update"])
            expected = payload["frontiers"][frame_left:frame_right]
            if not torch.equal(frontiers.cpu(), expected):
                raise RuntimeError(f"update {row['update_index']} frontiers differ")
            targets = payload["targets"][frame_left:frame_right].to(device)
            validity = payload["validity"][frame_left:frame_right].to(device)
            if not bool(validity.any()):
                raise RuntimeError(f"update {row['update_index']} has no valid bins")
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits.squeeze(0)[validity], targets[validity]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config["training"]["optimizer"]["gradient_clip_norm"]),
            )
            loss_value = float(loss.detach().cpu())
            grad_value = float(grad_norm.detach().cpu())
            if not math.isfinite(loss_value) or not math.isfinite(grad_value):
                raise RuntimeError(f"update {row['update_index']} is non-finite")
            optimizer.step()
            torch.cuda.synchronize(device)
            reserved = torch.cuda.max_memory_reserved(device)
            if reserved > max_reserved:
                raise RuntimeError("GPU reserved memory exceeded 4GiB cap")
            state = state.detached()
        except Exception:
            append_ledger(
                ledger_path,
                {
                    "kind": "uncertain",
                    "update_index": row["update_index"],
                    "source_id": row["source_id"],
                },
            )
            raise
        append_ledger(
            ledger_path,
            {
                "kind": "completed",
                "update_index": row["update_index"],
                "source_id": row["source_id"],
                "epoch": row["epoch"],
                "chunk_index": row["chunk_index"],
                "loss": loss_value,
                "gradient_norm_before_clip": grad_value,
                "valid_bins": int(validity.cpu().sum()),
            },
        )
        completed = row["update_index"]
        cursor = {
            "source_id": row["source_id"],
            "epoch": row["epoch"],
            "chunk_index": row["chunk_index"],
        }
        records.append(
            {
                "update_index": completed,
                "loss": loss_value,
                "gradient_norm_before_clip": grad_value,
                "valid_bins": int(validity.cpu().sum()),
                "source_id": row["source_id"],
                "epoch": row["epoch"],
            }
        )
        witness = None
        if pause_after == completed and completed < len(schedule):
            nxt = schedule[completed]
            if nxt["source_id"] == row["source_id"] and nxt["epoch"] == row["epoch"]:
                witness = {
                    "update_index": nxt["update_index"],
                    "source_id": nxt["source_id"],
                    "epoch": nxt["epoch"],
                    "sample_bounds": nxt["sample_bounds"],
                    "logits": next_chunk_logits(
                        model, state, payload["waveform"], nxt["sample_bounds"], device
                    ),
                }
            else:
                witness = {
                    "update_index": nxt["update_index"],
                    "source_id": nxt["source_id"],
                    "epoch": nxt["epoch"],
                    "sample_bounds": nxt["sample_bounds"],
                    "logits": next_chunk_logits(
                        model,
                        model.initial_state(1, device),
                        prepared["sources"][nxt["source_id"]]["waveform"],
                        nxt["sample_bounds"],
                        device,
                    ),
                }
        boundary = row["source_boundary_after"]
        cal_step = completed in set(config["evaluation"]["calibration_checkpoint_steps"])
        if boundary or cal_step or pause_after == completed or completed == len(schedule):
            digest = save_checkpoint(
                latest_path, config, model, optimizer, state, completed, cursor, witness
            )
            if cal_step:
                candidate = run_root / f"checkpoint_step_{completed}.pt"
                if candidate.exists():
                    raise RuntimeError(f"refusing to overwrite immutable CAL checkpoint {completed}")
                atomic_torch_save(candidate, torch.load(latest_path, map_location="cpu", weights_only=False))
            if pause_after == completed:
                pause_path = run_root / f"checkpoint_pause_update{completed}.pt"
                atomic_torch_save(pause_path, torch.load(latest_path, map_location="cpu", weights_only=False))
                atomic_json(
                    accounting_path,
                    {
                        "completed_updates": completed,
                        "uncertain_updates": 0,
                        "pause_after": completed,
                        "latest_checkpoint_sha256": digest,
                        "optimizer_step_counters": sorted(optimizer_step_count(optimizer)),
                        "records": records,
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        "training_elapsed_seconds_synchronized": time.perf_counter() - started,
                        "timing_scope": "this process wall from train_stage entry through pause checkpoint, including host/device transfers and optimizer updates",
                    },
                )
                return
    if optimizer_step_count(optimizer) != {780}:
        raise RuntimeError("optimizer counters do not equal 780 after the full allocation")
    digest = save_checkpoint(
        latest_path, config, model, optimizer, state, 780, cursor, None
    )
    final_path = run_root / "checkpoint_step_780.pt"
    if not final_path.exists():
        atomic_torch_save(final_path, torch.load(latest_path, map_location="cpu", weights_only=False))
    atomic_json(
        accounting_path,
        {
            "completed_updates": 780,
            "uncertain_updates": 0,
            "latest_checkpoint_sha256": digest,
            "optimizer_step_counters": sorted(optimizer_step_count(optimizer)),
            "records": records,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "training_elapsed_seconds_synchronized": time.perf_counter() - started,
            "timing_scope": "this process wall from train_stage entry through final checkpoint, including host/device transfers and optimizer updates",
        },
    )



def report_stage(config_path: Path, run_root: Path) -> None:
    config_path = bind_pilot_config(config_path, run_root)
    config = load_pilot_config(config_path)
    require_authorization()
    check = json.loads((run_root / "check_receipt.json").read_text(encoding="utf-8"))
    prepare = json.loads((run_root / "prepare_receipt.json").read_text(encoding="utf-8"))
    initial = json.loads((run_root / "initial_eval.json").read_text(encoding="utf-8"))
    accounting = json.loads((run_root / "accounting.json").read_text(encoding="utf-8"))
    selection = json.loads((run_root / "cal_selection.json").read_text(encoding="utf-8"))
    final = json.loads((run_root / "final_eval.json").read_text(encoding="utf-8"))
    resume = None
    resume_path = run_root / "resume_witness.json"
    if resume_path.exists():
        resume = json.loads(resume_path.read_text(encoding="utf-8"))
    hashes = {
        name: sha256_file(run_root / name)
        for name in (
            "checkpoint_latest.pt",
            "checkpoint_step_78.pt",
            "checkpoint_step_390.pt",
            "checkpoint_step_780.pt",
        )
        if (run_root / name).exists()
    }
    result = {
        "schema": "PSEM-ISSUE-164-GT-PILOT-RESULT-1",
        "status": "completed",
        "scope": "bounded multi-source GT learning decision pilot; not issue164 completion, generalization, KD, or production evidence",
        "identities": check["identities"],
        "prepare": {
            "prepared_sha256": prepare["prepared_sha256"],
            "fit_constant_prior": prepare["fit_constant_prior"],
        },
        "initial": initial,
        "accounting": accounting,
        "resume_witness": resume,
        "cal_selection": selection["selected"],
        "final": final,
        "checkpoint_sha256": hashes,
        "optimizer_step_accounting": {
            "prior_cumulative": config["budget"]["prior_cumulative_optimizer_updates"],
            "this_run": accounting["completed_updates"],
            "cumulative": config["budget"]["prior_cumulative_optimizer_updates"]
            + accounting["completed_updates"],
            "maximum_cumulative": config["budget"]["maximum_cumulative_optimizer_updates"],
        },
        "limitations": [
            "80ms frame-grid event proxies cannot resolve sub-frame admission timing",
            "source-global permutation scoring is oracle diagnostic only",
            "DEV prefixes are already-exposed directional diagnosis, not a generalization claim",
            "no live 80ms or matched teacher speedup claim",
        ],
        "runtime": pilot_runtime(config_path),
    }
    atomic_json(run_root / "result.json", result)
    atomic_json(
        run_root / "report_receipt.json",
        {
            "stage": "report",
            "result_sha256": sha256_file(run_root / "result.json"),
            "runtime": pilot_runtime(config_path),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("check", "prepare", "train", "evaluate", "report")
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pause-after", type=int, default=None)
    parser.add_argument(
        "--which", choices=("initial", "cal", "final"), default="initial"
    )
    parser.add_argument("--authorize-material", action="store_true")
    args = parser.parse_args()
    global MATERIAL_AUTHORIZED
    MATERIAL_AUTHORIZED = bool(args.authorize_material)
    args.run_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "check":
        check_stage(args.config.resolve(), args.run_root.resolve())
    elif args.stage == "prepare":
        prepare_stage(args.config.resolve(), args.run_root.resolve())
    elif args.stage == "train":
        train_stage(
            args.config.resolve(),
            args.run_root.resolve(),
            resume=args.resume,
            pause_after=args.pause_after,
        )
    elif args.stage == "evaluate":
        evaluate_stage(args.config.resolve(), args.run_root.resolve(), args.which)
    else:
        report_stage(args.config.resolve(), args.run_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
