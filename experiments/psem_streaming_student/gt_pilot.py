from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import shutil
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
HISTORICAL_CHECKPOINT_SCHEMA = "PSEM-ISSUE-164-GT-PILOT-CHECKPOINT-1"
CURRENT_CHECKPOINT_SCHEMA = "PSEM-ISSUE-164-GT-PILOT-CHECKPOINT-2"
EVAL_LOGITS_SCHEMA = "PSEM-ISSUE-164-GT-PILOT-EVAL-LOGITS-1"
EVAL_MANIFEST_SCHEMA = "PSEM-ISSUE-164-GT-PILOT-EVAL-MANIFEST-1"
EVAL_LOGITS_MANIFEST_NAME = "eval_logits_manifest.json"
OBJECTIVE_ACTIVITY_BCE = "activity_bce"
OBJECTIVE_SOLO_CONTRAST = "activity_bce_plus_fit_balanced_solo_ce"
INIT_LOGIT_MATCH_TOLERANCE = 1.0e-6
FIT_SOLO_CLASS_COUNTS = (3446, 1351, 1417, 865)
FIT_SOLO_TOTAL = 7079
FIT_SOLO_CLASS_WEIGHTS = (
    0.5135664538595474,
    1.3099555884529979,
    1.2489414255469302,
    2.0459537572254334,
)
OUTPUT_SLOTS = 4
HOP = StreamingStudent.output_hop_samples
SAMPLE_RATE = StreamingStudent.sample_rate
RESUME_LOGIT_TOLERANCE = 1.0e-4
COLLARS_MS = (100, 250, 500)
ABA_MAX_SECONDS = 1.0
MATERIAL_AUTHORIZED = False


def validate_training_objective(config: dict[str, Any]) -> dict[str, Any]:
    objective = config["training"].get("objective")
    if not isinstance(objective, dict) or "name" not in objective:
        raise RuntimeError("training.objective.name is required")
    name = objective["name"]
    if name == OBJECTIVE_ACTIVITY_BCE:
        if float(objective.get("activity_bce_coefficient", 1.0)) != 1.0:
            raise RuntimeError("activity BCE coefficient must remain 1.0")
        if float(objective.get("solo_ce_coefficient", 0.0)) != 0.0:
            raise RuntimeError("activity-BCE control must not add solo CE")
        return objective
    if name != OBJECTIVE_SOLO_CONTRAST:
        raise RuntimeError(f"unsupported training objective {name}")
    if float(objective["activity_bce_coefficient"]) != 1.0:
        raise RuntimeError("activity BCE coefficient must remain 1.0")
    if float(objective["solo_ce_coefficient"]) != 1.0:
        raise RuntimeError("solo CE coefficient must remain 1.0")
    counts = tuple(int(value) for value in objective["FIT_solo_class_counts"])
    total = int(objective["FIT_solo_total"])
    if counts != FIT_SOLO_CLASS_COUNTS or total != FIT_SOLO_TOTAL:
        raise RuntimeError("FIT solo class counts must remain the frozen diagnostic totals")
    if sum(counts) != total:
        raise RuntimeError("FIT solo total does not match class counts")
    stored = [float(value) for value in objective["class_weights_float64"]]
    computed = [total / (4.0 * count) for count in counts]
    if any(abs(left - right) > 1.0e-15 for left, right in zip(stored, computed)):
        raise RuntimeError("class_weights_float64 do not match 7079/(4*Nk)")
    if stored != list(FIT_SOLO_CLASS_WEIGHTS):
        raise RuntimeError("class_weights_float64 must match the frozen contract values")
    return objective


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
    prior = int(budget["prior_cumulative_optimizer_updates"])
    this_run = int(budget["maximum_optimizer_updates"])
    cumulative = int(budget["maximum_cumulative_optimizer_updates"])
    if this_run != 780:
        raise RuntimeError("pilot optimizer cap must be 780")
    if prior < 0:
        raise RuntimeError("prior consumed optimizer updates must be non-negative")
    if cumulative != prior + this_run:
        raise RuntimeError("cumulative optimizer cap must equal prior plus this run")
    if budget["automatic_retry_after_unknown_partial_update"] is not False:
        raise RuntimeError("automatic retry after unknown partial updates is forbidden")
    if training["student_parameter_count"] != 5940740:
        raise RuntimeError("student parameter count must remain 5940740")
    validate_training_objective(value)
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
            previous = None
            previous_index = None
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

def solo_class_weight_tensor(
    objective: dict[str, Any], device: torch.device | None = None
) -> Tensor:
    return torch.tensor(
        objective["class_weights_float64"], dtype=torch.float32, device=device
    )


def stable_solo_mask(targets: Tensor, validity: Tensor) -> Tensor:
    ones = targets == 1.0
    zeros = targets == 0.0
    return validity & (ones.sum(dim=-1) == 1) & (zeros.sum(dim=-1) == OUTPUT_SLOTS - 1)


def training_objective_terms(
    logits: Tensor,
    targets: Tensor,
    validity: Tensor,
    *,
    objective: dict[str, Any],
    class_weights: Tensor | None,
) -> dict[str, Tensor]:
    if not bool(validity.any()):
        raise RuntimeError("training chunk has no valid bins")
    activity_bce = nn.functional.binary_cross_entropy_with_logits(
        logits[validity], targets[validity]
    )
    if objective["name"] == OBJECTIVE_ACTIVITY_BCE:
        zero = activity_bce * 0
        return {
            "activity_bce": activity_bce,
            "solo_ce": zero,
            "total_objective": activity_bce,
        }
    n_valid = validity.to(dtype=logits.dtype).sum()
    solo = stable_solo_mask(targets, validity)
    if class_weights is None:
        raise RuntimeError("solo contrast requires frozen class weights")
    if bool(solo.any()):
        slots = (targets[solo] == 1.0).to(dtype=torch.int64).argmax(dim=-1)
        ce = nn.functional.cross_entropy(logits[solo], slots, reduction="none")
        solo_ce = (class_weights[slots] * ce).sum() / n_valid
    else:
        solo_ce = logits.new_zeros(())
    total = activity_bce + float(objective["solo_ce_coefficient"]) * solo_ce
    return {
        "activity_bce": activity_bce,
        "solo_ce": solo_ce,
        "total_objective": total,
    }


def fit_solo_class_counts(prepared: dict[str, Any], source_ids: list[str]) -> list[int]:
    counts = [0] * OUTPUT_SLOTS
    for source_id in source_ids:
        payload = prepared["sources"][source_id]
        solo = stable_solo_mask(payload["targets"], payload["validity"])
        if not bool(solo.any()):
            continue
        slots = (payload["targets"][solo] == 1.0).to(dtype=torch.int64).argmax(dim=-1)
        for slot in range(OUTPUT_SLOTS):
            counts[slot] += int((slots == slot).sum().item())
    return counts



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
    excluded = validity & relation
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
    relation_valid = exposed.numpy()
    pred_events = active_set_events(
        probabilities.numpy(), frontiers.numpy(), relation_valid, threshold
    )
    ref_events = active_set_events(
        targets.numpy(), frontiers.numpy(), relation_valid, threshold
    )
    event_rates = [match_event_rates(pred_events, ref_events, collar) for collar in COLLARS_MS]
    aba_ref = solo_aba_interruptions(targets.numpy(), relation_valid, threshold, HOP)
    aba_pred = solo_aba_interruptions(probabilities.numpy(), relation_valid, threshold, HOP)
    return {
        "canonical_masked_bce": bce,
        "canonical_brier": brier,
        "valid_bins": int(validity.sum()),
        "relation_mask_overlapping_valid_bins": int(excluded.sum()),
        "relation_mask_excluded_valid_bins": int(excluded.sum()),
        "relation_mask_exposed_valid_bins": int(exposed.sum()),
        "relation_masks_do_not_erase_activity_targets": True,
        "canonical": canonical,
        "source_global_aligned": {
            **aligned,
            "metrics": aligned_metrics,
            "aligned_masked_bce": aligned["best"]["masked_bce"],
            "event_proxy_uses_unpermuted_occupancy": True,
        },
        "event_proxy": {
            "predicted_events": len(pred_events),
            "reference_events": len(ref_events),
            "collars": event_rates,
            "solo_aba_interruptions_le_1s": {
                "reference": len(aba_ref),
                "predicted": len(aba_pred),
            },
            "validity": "activity_valid_and_relation_unmasked",
            "gap_reset": True,
            "permutation_not_applied": True,
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
    original_validity = validity.clone()
    original_targets = targets.clone()
    scored = score_outputs(logits, targets, validity, frontiers, relation_masks, 0.5, timeline)
    if not torch.equal(validity, original_validity) or not torch.equal(targets, original_targets):
        raise RuntimeError("relation scoring mutated activity validity or targets")
    if scored["relation_mask_excluded_valid_bins"] != scored["relation_mask_overlapping_valid_bins"]:
        raise RuntimeError("excluded relation bins must equal overlapping valid relation-mask bins")
    if scored["relation_mask_excluded_valid_bins"] + scored["relation_mask_exposed_valid_bins"] != scored["valid_bins"]:
        raise RuntimeError("excluded plus exposed relation bins must equal activity-valid bins")
    if scored["relation_mask_exposed_valid_bins"] == scored["valid_bins"] and bool(relation.any()):
        raise RuntimeError("exposed complement was stored as excluded")
    activity_events = active_set_events(targets.numpy(), frontiers.numpy(), validity.numpy(), 0.5)
    relation_events = active_set_events(
        targets.numpy(), frontiers.numpy(), (validity & ~relation).numpy(), 0.5
    )
    if scored["event_proxy"]["reference_events"] != len(relation_events):
        raise RuntimeError("event proxy did not use relation-unmasked activity-valid bins")
    gap_occupancy = np.zeros((3, 4), dtype=np.float32)
    gap_occupancy[0, 0] = 1.0
    gap_occupancy[2, 1] = 1.0
    gap_frontiers = np.array([hop, 2 * hop, 3 * hop], dtype=np.int64)
    carried = active_set_events(
        gap_occupancy, gap_frontiers, np.array([True, False, True]), 0.5
    )
    if carried:
        raise RuntimeError("active-set events bridged an invalid gap")
    relation_gap_valid = np.array([True, False, True])
    relation_gap_occ = np.zeros((3, 4), dtype=np.float32)
    relation_gap_occ[0, 0] = 1.0
    relation_gap_occ[2, 0] = 1.0
    if active_set_events(relation_gap_occ, gap_frontiers, relation_gap_valid, 0.5):
        raise RuntimeError("active-set events bridged a relation-masked gap")
    aba_bridge_occ = np.zeros((3, 4), dtype=np.float32)
    aba_bridge_occ[0, 0] = 1.0
    aba_bridge_occ[1, 1] = 1.0
    aba_bridge_occ[2, 0] = 1.0
    aba_activity = np.array([True, True, True])
    if not solo_aba_interruptions(aba_bridge_occ, aba_activity, 0.5, hop):
        raise RuntimeError("contiguous solo A-B-A probe failed")
    aba_relation_valid = np.array([True, False, True])
    if solo_aba_interruptions(aba_bridge_occ, aba_relation_valid, 0.5, hop):
        raise RuntimeError("ABA bridged a relation or unknown gap")
    swapped_scored = score_outputs(
        swapped, targets, validity, frontiers, relation_masks, 0.5, timeline
    )
    if swapped_scored["event_proxy"]["reference_events"] != scored["event_proxy"]["reference_events"]:
        raise RuntimeError("source-global permutation hid or altered reference transitions")
    if swapped_scored["source_global_aligned"]["best"]["permutation"] != [1, 0, 2, 3]:
        raise RuntimeError("permutation oracle was not independent of event gating")
    matched = match_event_rates(relation_events, relation_events, 100)
    if matched["matched"] != len(relation_events) or matched["false_events"] != 0:
        raise RuntimeError("self-matched event proxy failed")
    return {
        "padded_slots": slots,
        "invalid_warmup_bins": 3,
        "overlap_preserved": True,
        "ambiguous_invalid": True,
        "relation_mask_does_not_erase_activity": True,
        "relation_excluded_bins": scored["relation_mask_excluded_valid_bins"],
        "relation_exposed_bins": scored["relation_mask_exposed_valid_bins"],
        "activity_valid_bins": scored["valid_bins"],
        "activity_events_without_relation_gate": len(activity_events),
        "relation_gated_reference_events": len(relation_events),
        "invalid_gap_reset": True,
        "relation_gap_reset": True,
        "aba_breaks_relation_gap": True,
        "permutation_does_not_hide_reference_transitions": True,
        "solo_aba_interruptions": len(aba),
        "permutation_recovered": oracle["best"]["permutation"],
        "self_matched_events": matched,
        "unused_event_proxy_on_swapped_logits": swapped_scored["event_proxy"]["predicted_events"],
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
    checkpoint_probes = checkpoint_identity_probes(config)
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
            "checkpoint_identity_probes": checkpoint_probes,
            "student_parameter_count": count,
            "runtime": pilot_runtime(config_path),
        },
    )


def require_authorization() -> None:
    if not MATERIAL_AUTHORIZED:
        raise RuntimeError(
            "material prepare/import-prepared/train/evaluate is blocked until Director GO passes --authorize-material"
        )



def load_or_project(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    return project_source(config, source)


def prepare_stage(config_path: Path, run_root: Path) -> None:
    config_path = bind_pilot_config(config_path, run_root)
    config = load_pilot_config(config_path)
    require_authorization()
    if "prepared_import" in config:
        raise RuntimeError("this config reuses origin prepared; use import-prepared")
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


def import_prepared_stage(config_path: Path, run_root: Path) -> None:
    config_path = bind_pilot_config(config_path, run_root)
    config = load_pilot_config(config_path)
    require_authorization()
    spec = config.get("prepared_import")
    if not isinstance(spec, dict):
        raise RuntimeError("import-prepared requires prepared_import")
    repo = Path(__file__).resolve().parent.parent.parent
    origin_prepared = (repo / spec["origin_prepared_path"]).resolve()
    origin_config = (repo / spec["origin_frozen_config_path"]).resolve()
    if sha256_file(origin_prepared) != spec["origin_prepared_sha256"]:
        raise RuntimeError("origin prepared identity mismatch")
    if sha256_file(origin_config) != spec["origin_frozen_config_sha256"]:
        raise RuntimeError("origin frozen config identity mismatch")
    dest = run_root / "prepared_sources.pt"
    if dest.exists():
        raise RuntimeError("refusing to overwrite existing prepared_sources.pt")
    identities = verify_pilot_identities(config)
    shutil.copyfile(origin_prepared, dest)
    if sha256_file(dest) != spec["origin_prepared_sha256"]:
        raise RuntimeError("imported prepared identity mismatch")
    prepared = torch.load(dest, map_location="cpu", weights_only=False)
    catalog = catalog_sources(config)
    if set(prepared["sources"]) != set(catalog):
        raise RuntimeError("imported prepared sources differ from config")
    for source_id, source in catalog.items():
        payload = prepared["sources"][source_id]
        if int(payload["waveform"].numel()) != int(source["prefix_samples"]):
            raise RuntimeError(f"{source_id} imported prefix length differs")
        if tuple(payload["targets"].shape[-1:]) != (OUTPUT_SLOTS,):
            raise RuntimeError(f"{source_id} imported target slots differ")
    atomic_json(
        run_root / "prepare_receipt.json",
        {
            "stage": "import_prepared",
            "optimizer_steps": 0,
            "prepared_sha256": spec["origin_prepared_sha256"],
            "origin_prepared_sha256": spec["origin_prepared_sha256"],
            "origin_config_sha256": spec["origin_frozen_config_sha256"],
            "origin_prepared_path": spec["origin_prepared_path"],
            "origin_frozen_config_path": spec["origin_frozen_config_path"],
            "identities": identities,
            "fit_constant_prior": prepared["fit_constant_prior"],
            "source_ids": sorted(prepared["sources"]),
            "runtime": pilot_runtime(config_path),
            "new_waveform_preparation": False,
            "hardlink": False,
        },
    )


def load_prepared(run_root: Path, config_path: Path) -> dict[str, Any]:
    prepared_path = run_root / "prepared_sources.pt"
    receipt = json.loads((run_root / "prepare_receipt.json").read_text(encoding="utf-8"))
    config = load_pilot_config(config_path)
    if receipt["runtime"]["config_sha256"] != sha256_file(config_path):
        raise RuntimeError("prepared inputs were created under a different frozen config")
    if sha256_file(prepared_path) != receipt["prepared_sha256"]:
        raise RuntimeError("prepared input identity mismatch")
    if receipt.get("stage") == "import_prepared":
        spec = config.get("prepared_import")
        if not isinstance(spec, dict):
            raise RuntimeError("imported prepared receipt requires prepared_import")
        if receipt["origin_prepared_sha256"] != receipt["prepared_sha256"]:
            raise RuntimeError("imported prepared hash does not match origin prepared hash")
        if receipt["origin_prepared_sha256"] != spec["origin_prepared_sha256"]:
            raise RuntimeError("imported prepared hash does not match config origin")
        if receipt["origin_config_sha256"] != spec["origin_frozen_config_sha256"]:
            raise RuntimeError("imported origin config hash does not match")
    elif receipt.get("stage") != "prepare":
        raise RuntimeError("unknown prepared receipt stage")
    return torch.load(prepared_path, map_location="cpu", weights_only=False)


def run_root_posix_relative(run_root: Path, path: Path) -> str:
    return path.resolve().relative_to(run_root.resolve()).as_posix()


def upsert_eval_logit_manifest(
    run_root: Path,
    config_sha256: str,
    prepared_sha256: str,
    entry: dict[str, Any],
) -> None:
    path = run_root / EVAL_LOGITS_MANIFEST_NAME
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema") != EVAL_MANIFEST_SCHEMA:
            raise RuntimeError("unexpected eval logit manifest schema")
        if manifest.get("config_sha256") != config_sha256:
            raise RuntimeError("eval logit manifest config identity differs")
        if manifest.get("prepared_sha256") != prepared_sha256:
            raise RuntimeError("eval logit manifest prepared identity differs")
        entries = [
            row
            for row in manifest["entries"]
            if not (
                row["source_id"] == entry["source_id"]
                and row["weight_id"] == entry["weight_id"]
            )
        ]
    else:
        manifest = {
            "schema": EVAL_MANIFEST_SCHEMA,
            "config_sha256": config_sha256,
            "prepared_sha256": prepared_sha256,
            "entries": [],
        }
        entries = []
    entries.append(entry)
    manifest["entries"] = entries
    atomic_json(path, manifest)


def retain_eval_logits(
    run_root: Path,
    *,
    config_sha256: str,
    prepared_sha256: str,
    source_id: str,
    weight_id: str,
    checkpoint_sha256: str | None,
    logits: Tensor,
    frontiers: Tensor,
) -> None:
    relative = Path("eval_logits") / f"{weight_id}_{source_id}.pt"
    path = run_root / relative
    atomic_torch_save(
        path,
        {
            "schema": EVAL_LOGITS_SCHEMA,
            "source_id": source_id,
            "weight_id": weight_id,
            "checkpoint_sha256": checkpoint_sha256,
            "prepared_sha256": prepared_sha256,
            "config_sha256": config_sha256,
            "logits": logits.to(dtype=torch.float32).contiguous(),
            "frontiers": frontiers.contiguous(),
        },
    )
    upsert_eval_logit_manifest(
        run_root,
        config_sha256,
        prepared_sha256,
        {
            "path": run_root_posix_relative(run_root, path),
            "sha256": sha256_file(path),
            "source_id": source_id,
            "weight_id": weight_id,
            "checkpoint_sha256": checkpoint_sha256,
        },
    )


def compare_same_seed_initial_logits(
    run_root: Path,
    config: dict[str, Any],
    source_ids: list[str],
) -> dict[str, Any] | None:
    spec = config.get("same_initialization")
    if spec is None:
        return None
    repo = Path(__file__).resolve().parent.parent.parent
    origin_root = (repo / spec["origin_logits_root"]).resolve()
    weight_id = spec["weight_id"]
    tolerance = float(spec["max_abs_delta_tolerance"])
    if tolerance != INIT_LOGIT_MATCH_TOLERANCE:
        raise RuntimeError("same-seed initial logit tolerance must remain 1e-6")
    per_source = []
    maxima = []
    for source_id in source_ids:
        ours_path = run_root / "eval_logits" / f"{weight_id}_{source_id}.pt"
        origin_path = origin_root / f"{weight_id}_{source_id}.pt"
        ours = torch.load(ours_path, map_location="cpu", weights_only=False)
        origin = torch.load(origin_path, map_location="cpu", weights_only=False)
        delta = float((ours["logits"] - origin["logits"]).abs().max())
        maxima.append(delta)
        per_source.append({"source_id": source_id, "max_abs_delta": delta})
    global_max = max(maxima)
    matched = global_max <= tolerance
    report = {
        "max_abs_delta": global_max,
        "tolerance": tolerance,
        "matched_init": matched,
        "per_source": per_source,
        "weight_id": weight_id,
        "origin_logits_root": spec["origin_logits_root"],
        "extra_model_forward": False,
    }
    atomic_json(run_root / "same_initialization.json", report)
    if not matched:
        raise RuntimeError(
            f"same-seed initial logits max abs delta {global_max} exceeds {tolerance}"
        )
    return report



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
    *,
    run_root: Path | None = None,
    config_sha256: str | None = None,
    prepared_sha256: str | None = None,
    weight_id: str | None = None,
    checkpoint_sha256: str | None = None,
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
            if weight_id is not None:
                if run_root is None or config_sha256 is None or prepared_sha256 is None:
                    raise RuntimeError("eval logit retention requires run root identities")
                retain_eval_logits(
                    run_root,
                    config_sha256=config_sha256,
                    prepared_sha256=prepared_sha256,
                    source_id=source_id,
                    weight_id=weight_id,
                    checkpoint_sha256=checkpoint_sha256,
                    logits=logits,
                    frontiers=frontiers,
                )
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
    config_sha256 = sha256_file(config_path)
    prepared_sha256 = sha256_file(run_root / "prepared_sources.pt")
    schedule = fit_schedule(config)
    device = require_gpu(config)
    chunk = int(config["training"]["chunk_samples"])
    threshold = float(config["evaluation"]["operating_point"])
    if which == "initial":
        model = build_fresh_model(config, device)
        fit_ids = [source["source_id"] for source in config["sources"]["FIT"]]
        cal_ids = [source["source_id"] for source in config["sources"]["CAL"]]
        dev_ids = [source["source_id"] for source in config["sources"]["DEV"]]
        initial_ids = fit_ids + cal_ids + dev_ids
        untrained = evaluate_model_on_sources(
            model,
            prepared,
            initial_ids,
            device,
            chunk,
            threshold,
            "untrained",
            run_root=run_root,
            config_sha256=config_sha256,
            prepared_sha256=prepared_sha256,
            weight_id="untrained_seed20260915",
            checkpoint_sha256=None,
        )
        same_init = compare_same_seed_initial_logits(run_root, config, initial_ids)
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
                "same_initialization": same_init,
                "runtime": pilot_runtime(config_path),
            },
        )
        return
    if which == "cal":
        rows = {}
        for step in config["evaluation"]["calibration_checkpoint_steps"]:
            path = run_root / f"checkpoint_step_{step}.pt"
            payload = load_current_checkpoint_for_eval(
                path,
                config,
                config_sha256,
                prepared_sha256,
                int(step),
                schedule,
            )
            model = StreamingStudent().to(device)
            model.load_state_dict(payload["model"])
            checkpoint_sha256 = sha256_file(path)
            rows[str(step)] = {
                "checkpoint_sha256": checkpoint_sha256,
                "cal": evaluate_model_on_sources(
                    model,
                    prepared,
                    [source["source_id"] for source in config["sources"]["CAL"]],
                    device,
                    chunk,
                    threshold,
                    f"cal_step_{step}",
                    run_root=run_root,
                    config_sha256=config_sha256,
                    prepared_sha256=prepared_sha256,
                    weight_id=f"checkpoint_{step}",
                    checkpoint_sha256=checkpoint_sha256,
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
        payload = load_current_checkpoint_for_eval(
            path,
            config,
            config_sha256,
            prepared_sha256,
            int(step),
            schedule,
        )
        model = StreamingStudent().to(device)
        model.load_state_dict(payload["model"])
        fit_ids = [source["source_id"] for source in config["sources"]["FIT"]]
        dev_ids = [source["source_id"] for source in config["sources"]["DEV"]]
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        scored = evaluate_model_on_sources(
            model,
            prepared,
            fit_ids + dev_ids,
            device,
            chunk,
            threshold,
            "selected_final",
            run_root=run_root,
            config_sha256=config_sha256,
            prepared_sha256=prepared_sha256,
            weight_id=f"checkpoint_{step}",
            checkpoint_sha256=selection["selected"]["checkpoint_sha256"],
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
                    "scope": "selected-checkpoint source-zero prefix replay over FIT+DEV after model load; timer starts after load, peak reset, and device synchronize; first streamed prefixes are included; not a warmup pass; excludes training; not live 80ms admission or matched teacher cost",
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


def optimizer_settings_from_config(config: dict[str, Any]) -> dict[str, Any]:
    optimizer = config["training"]["optimizer"]
    return {
        "name": optimizer["name"],
        "learning_rate": float(optimizer["learning_rate"]),
        "weight_decay": float(optimizer["weight_decay"]),
        "gradient_clip_norm": float(optimizer["gradient_clip_norm"]),
    }


def optimizer_counters_from_state_dict(state_dict: dict[str, Any]) -> set[int]:
    counts: set[int] = set()
    for value in (state_dict or {}).get("state", {}).values():
        if isinstance(value, dict) and "step" in value:
            step = value["step"]
            counts.add(int(step.item()) if torch.is_tensor(step) else int(step))
    return counts


def optimizer_settings_from_state_dict(state_dict: dict[str, Any]) -> dict[str, float]:
    groups = (state_dict or {}).get("param_groups") or []
    if not groups:
        return {}
    rates = {float(group["lr"]) for group in groups}
    decays = {float(group.get("weight_decay", 0.0)) for group in groups}
    if len(rates) != 1 or len(decays) != 1:
        raise RuntimeError("optimizer param groups have mixed learning rate or weight decay")
    return {"learning_rate": next(iter(rates)), "weight_decay": next(iter(decays))}


def numeric_state_position(state: Any) -> dict[str, int] | None:
    if state is None:
        return None
    if isinstance(state, dict):
        return {
            "total_samples": int(state["total_samples"]),
            "emitted_outputs": int(state["emitted_outputs"]),
        }
    return {
        "total_samples": int(state.total_samples),
        "emitted_outputs": int(state.emitted_outputs),
    }


def expected_numeric_state(
    cursor: dict[str, Any] | None, config: dict[str, Any]
) -> dict[str, int] | None:
    if cursor is None:
        return None
    count = int(cursor["chunk_index"]) + 1
    return {
        "total_samples": count * int(config["training"]["chunk_samples"]),
        "emitted_outputs": count * int(config["training"]["frames_per_update"]),
    }


def expected_cursor(schedule: list[dict[str, Any]], completed: int) -> dict[str, Any] | None:
    if completed == 0:
        return None
    row = schedule[completed - 1]
    return {
        "source_id": row["source_id"],
        "epoch": row["epoch"],
        "chunk_index": row["chunk_index"],
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
    *,
    config_sha256: str,
    prepared_sha256: str,
) -> str:
    numeric = numeric_state_position(None if state is None else state.cpu_dict())
    payload = {
        "schema": CURRENT_CHECKPOINT_SCHEMA,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config_schema": config["schema"],
        "config_sha256": config_sha256,
        "implementation_sha256": dict(config["implementation_sha256"]),
        "prepared_sha256": prepared_sha256,
        "source_order": list(config["training"]["source_order"]),
        "optimizer_settings": optimizer_settings_from_config(config),
        "optimizer_step_counters": sorted(optimizer_step_count(optimizer)),
        "completed_updates": completed,
        "numeric_state": numeric,
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


def validate_current_checkpoint_identity(
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    config_sha256: str,
    prepared_sha256: str,
    schedule: list[dict[str, Any]],
    expected_completed: int | None = None,
) -> None:
    if checkpoint.get("schema") != CURRENT_CHECKPOINT_SCHEMA:
        raise RuntimeError(
            "unbound historical checkpoint cannot be used by the current bound consumer; historical evaluation must use an explicit artifact hash"
        )
    if checkpoint.get("config_schema") != config["schema"]:
        raise RuntimeError("checkpoint config schema does not match the frozen config")
    if checkpoint.get("config_sha256") != config_sha256:
        raise RuntimeError("checkpoint config identity does not match the frozen config")
    if checkpoint.get("implementation_sha256") != config["implementation_sha256"]:
        raise RuntimeError("checkpoint implementation identity does not match")
    if checkpoint.get("prepared_sha256") != prepared_sha256:
        raise RuntimeError("checkpoint prepared-input identity does not match")
    if list(checkpoint.get("source_order") or []) != list(config["training"]["source_order"]):
        raise RuntimeError("checkpoint source order does not match")
    completed = int(checkpoint["completed_updates"])
    if expected_completed is not None and completed != int(expected_completed):
        raise RuntimeError(
            f"checkpoint completed_updates {completed} does not match expected step {expected_completed}"
        )
    if completed < 0 or completed > len(schedule):
        raise RuntimeError("checkpoint completed_updates is outside the frozen schedule")
    cursor = checkpoint.get("schedule_cursor")
    expected = expected_cursor(schedule, completed)
    if expected is None:
        if cursor is not None:
            raise RuntimeError("zero-update checkpoint must not carry a schedule cursor")
    else:
        if cursor is None:
            raise RuntimeError("checkpoint is missing schedule cursor")
        for key in ("source_id", "epoch", "chunk_index"):
            if cursor.get(key) != expected[key]:
                raise RuntimeError(
                    "checkpoint cursor does not match the completed schedule position"
                )
    expected_numeric = expected_numeric_state(cursor, config)
    stored_numeric = checkpoint.get("numeric_state")
    actual_numeric = numeric_state_position(checkpoint.get("streaming_state"))
    if stored_numeric != expected_numeric or actual_numeric != expected_numeric:
        raise RuntimeError("checkpoint numeric stream position does not match the completed cursor")
    settings = optimizer_settings_from_config(config)
    stored_settings = checkpoint.get("optimizer_settings") or {}
    for key in ("name", "learning_rate", "weight_decay"):
        if stored_settings.get(key) != settings[key]:
            raise RuntimeError("checkpoint optimizer settings do not match the frozen config")
    live_settings = optimizer_settings_from_state_dict(checkpoint.get("optimizer") or {})
    if live_settings and (
        live_settings["learning_rate"] != settings["learning_rate"]
        or live_settings["weight_decay"] != settings["weight_decay"]
    ):
        raise RuntimeError("optimizer state settings do not match the frozen config")
    derived = optimizer_counters_from_state_dict(checkpoint.get("optimizer") or {})
    stored_counters = set(int(value) for value in (checkpoint.get("optimizer_step_counters") or []))
    if completed == 0:
        if derived or stored_counters:
            raise RuntimeError("zero-update checkpoint must not carry optimizer step counters")
        return
    if derived != {completed} or stored_counters != {completed}:
        raise RuntimeError("optimizer counters do not match completed updates")


def validate_bound_checkpoint_for_resume(
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    config_sha256: str,
    prepared_sha256: str,
    schedule: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
) -> None:
    refuse_stale_checkpoint(checkpoint, ledger_rows)
    validate_current_checkpoint_identity(
        checkpoint, config, config_sha256, prepared_sha256, schedule
    )


def load_current_checkpoint_for_eval(
    path: Path,
    config: dict[str, Any],
    config_sha256: str,
    prepared_sha256: str,
    expected_completed: int,
    schedule: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_current_checkpoint_identity(
        payload,
        config,
        config_sha256,
        prepared_sha256,
        schedule,
        expected_completed=expected_completed,
    )
    return payload


def load_historical_checkpoint_for_eval(path: Path, expected_sha256: str) -> dict[str, Any]:
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise RuntimeError(f"historical checkpoint hash mismatch for {path.name}: {digest}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise RuntimeError(f"{path.name} lacks model weights")
    return payload


def _expect_resume_rejection(
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    config_sha256: str,
    prepared_sha256: str,
    schedule: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
    needle: str,
) -> str:
    try:
        validate_bound_checkpoint_for_resume(
            checkpoint, config, config_sha256, prepared_sha256, schedule, ledger_rows
        )
    except RuntimeError as exc:
        message = str(exc)
        if needle not in message:
            raise RuntimeError(f"rejected for the wrong reason: {message}") from exc
        return message
    raise RuntimeError(f"expected resume rejection containing {needle}")


def checkpoint_identity_probes(config: dict[str, Any]) -> dict[str, Any]:
    schedule = fit_schedule(config)
    config_sha256 = "a" * 64
    prepared_sha256 = "b" * 64
    settings = optimizer_settings_from_config(config)
    cursor = expected_cursor(schedule, 2)
    numeric = expected_numeric_state(cursor, config)
    ledger = [
        {"kind": "intent", "update_index": 1},
        {"kind": "completed", "update_index": 1},
        {"kind": "intent", "update_index": 2},
        {"kind": "completed", "update_index": 2},
    ]
    bound = {
        "schema": CURRENT_CHECKPOINT_SCHEMA,
        "model": {},
        "optimizer": {
            "state": {0: {"step": torch.tensor(2)}},
            "param_groups": [
                {
                    "lr": settings["learning_rate"],
                    "weight_decay": settings["weight_decay"],
                }
            ],
        },
        "config_schema": config["schema"],
        "config_sha256": config_sha256,
        "implementation_sha256": dict(config["implementation_sha256"]),
        "prepared_sha256": prepared_sha256,
        "source_order": list(config["training"]["source_order"]),
        "optimizer_settings": settings,
        "optimizer_step_counters": [2],
        "completed_updates": 2,
        "numeric_state": numeric,
        "streaming_state": {
            "total_samples": numeric["total_samples"],
            "emitted_outputs": numeric["emitted_outputs"],
        },
        "schedule_cursor": cursor,
    }
    validate_bound_checkpoint_for_resume(
        bound, config, config_sha256, prepared_sha256, schedule, ledger
    )
    foreign = dict(bound)
    foreign["config_sha256"] = "c" * 64
    wrong_cursor = dict(bound)
    wrong_cursor["schedule_cursor"] = {**cursor, "chunk_index": cursor["chunk_index"] + 1}
    wrong_settings = dict(bound)
    wrong_settings["optimizer_settings"] = {**settings, "learning_rate": 0.001}
    wrong_counters = dict(bound)
    wrong_counters["optimizer_step_counters"] = [1]
    wrong_counters["optimizer"] = {
        "state": {0: {"step": torch.tensor(1)}},
        "param_groups": bound["optimizer"]["param_groups"],
    }
    unbound = dict(bound)
    unbound["schema"] = HISTORICAL_CHECKPOINT_SCHEMA
    stale_ledger = ledger + [
        {"kind": "intent", "update_index": 3},
        {"kind": "completed", "update_index": 3},
    ]
    uncertain_ledger = ledger + [{"kind": "intent", "update_index": 3}]
    return {
        "bound_resume_accepted": True,
        "foreign_config_rejected": _expect_resume_rejection(
            foreign, config, config_sha256, prepared_sha256, schedule, ledger, "config identity"
        ),
        "wrong_cursor_rejected": _expect_resume_rejection(
            wrong_cursor, config, config_sha256, prepared_sha256, schedule, ledger, "cursor"
        ),
        "wrong_optimizer_settings_rejected": _expect_resume_rejection(
            wrong_settings,
            config,
            config_sha256,
            prepared_sha256,
            schedule,
            ledger,
            "optimizer settings",
        ),
        "wrong_optimizer_counters_rejected": _expect_resume_rejection(
            wrong_counters, config, config_sha256, prepared_sha256, schedule, ledger, "counters"
        ),
        "unbound_resume_rejected": _expect_resume_rejection(
            unbound, config, config_sha256, prepared_sha256, schedule, ledger, "unbound historical"
        ),
        "stale_ledger_rejected": _expect_resume_rejection(
            bound, config, config_sha256, prepared_sha256, schedule, stale_ledger, "stale checkpoint"
        ),
        "uncertain_ledger_rejected": _expect_resume_rejection(
            bound,
            config,
            config_sha256,
            prepared_sha256,
            schedule,
            uncertain_ledger,
            "uncertain update",
        ),
        "historical_eval_uses_explicit_hash": True,
    }


def _expect_eval_loader_rejection(
    path: Path,
    config: dict[str, Any],
    config_sha256: str,
    prepared_sha256: str,
    expected_completed: int,
    schedule: list[dict[str, Any]],
    needle: str,
) -> str:
    try:
        load_current_checkpoint_for_eval(
            path, config, config_sha256, prepared_sha256, expected_completed, schedule
        )
    except RuntimeError as exc:
        message = str(exc)
        if needle not in message:
            raise RuntimeError(f"rejected for the wrong reason: {message}") from exc
        return message
    raise RuntimeError(f"expected ordinary eval loader rejection containing {needle}")


def _current_bound_checkpoint_payload(
    config: dict[str, Any],
    completed: int,
    config_sha256: str,
    prepared_sha256: str,
    *,
    fixture_role: str,
) -> dict[str, Any]:
    schedule = fit_schedule(config)
    settings = optimizer_settings_from_config(config)
    cursor = expected_cursor(schedule, completed)
    numeric = expected_numeric_state(cursor, config)
    return {
        "schema": CURRENT_CHECKPOINT_SCHEMA,
        "model": {},
        "fixture_role": fixture_role,
        "optimizer": {
            "state": {0: {"step": torch.tensor(completed)}},
            "param_groups": [
                {
                    "lr": settings["learning_rate"],
                    "weight_decay": settings["weight_decay"],
                }
            ],
        },
        "config_schema": config["schema"],
        "config_sha256": config_sha256,
        "implementation_sha256": dict(config["implementation_sha256"]),
        "prepared_sha256": prepared_sha256,
        "source_order": list(config["training"]["source_order"]),
        "optimizer_settings": settings,
        "optimizer_step_counters": [completed],
        "completed_updates": completed,
        "numeric_state": numeric,
        "streaming_state": {
            "total_samples": numeric["total_samples"],
            "emitted_outputs": numeric["emitted_outputs"],
        },
        "schedule_cursor": cursor,
    }


def current_checkpoint_eval_loader_probes(
    config: dict[str, Any], fixture_dir: Path
) -> dict[str, Any]:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    schedule = fit_schedule(config)
    config_sha256 = "a" * 64
    prepared_sha256 = "b" * 64
    bound = _current_bound_checkpoint_payload(
        config,
        2,
        config_sha256,
        prepared_sha256,
        fixture_role="current-bound-eval-loader-fixture-not-training",
    )
    bound_path = fixture_dir / "current_bound_step2.pt"
    atomic_torch_save(bound_path, bound)
    loaded = load_current_checkpoint_for_eval(
        bound_path, config, config_sha256, prepared_sha256, 2, schedule
    )
    if loaded["schema"] != CURRENT_CHECKPOINT_SCHEMA or loaded["completed_updates"] != 2:
        raise RuntimeError("ordinary CAL/final loader rejected a valid current bound checkpoint")
    foreign = dict(bound)
    foreign["config_sha256"] = "c" * 64
    foreign_path = fixture_dir / "foreign_config.pt"
    atomic_torch_save(foreign_path, foreign)
    historical = {
        "schema": HISTORICAL_CHECKPOINT_SCHEMA,
        "model": {},
        "fixture_role": "historical-unbound-v1-fixture-not-a-current-training-checkpoint",
        "completed_updates": 2,
    }
    historical_path = fixture_dir / "historical_v1_fixture.pt"
    atomic_torch_save(historical_path, historical)
    historical_sha256 = sha256_file(historical_path)
    historical_loaded = load_historical_checkpoint_for_eval(historical_path, historical_sha256)
    if historical_loaded["schema"] != HISTORICAL_CHECKPOINT_SCHEMA:
        raise RuntimeError("explicit-hash historical reader failed the marked v1 fixture")
    cal78 = _current_bound_checkpoint_payload(
        config,
        78,
        config_sha256,
        prepared_sha256,
        fixture_role="current-bound-cal-step78-fixture-not-training",
    )
    cal78_path = fixture_dir / "current_bound_step78.pt"
    atomic_torch_save(cal78_path, cal78)
    selected390 = _current_bound_checkpoint_payload(
        config,
        390,
        config_sha256,
        prepared_sha256,
        fixture_role="current-bound-selected-step390-fixture-not-training",
    )
    selected390_path = fixture_dir / "current_bound_step390.pt"
    atomic_torch_save(selected390_path, selected390)
    ledger_780 = [
        {"kind": "intent", "update_index": 780},
        {"kind": "completed", "update_index": 780},
    ]
    loaded78 = load_current_checkpoint_for_eval(
        cal78_path, config, config_sha256, prepared_sha256, 78, schedule
    )
    loaded390 = load_current_checkpoint_for_eval(
        selected390_path, config, config_sha256, prepared_sha256, 390, schedule
    )
    if loaded78["completed_updates"] != 78 or loaded390["completed_updates"] != 390:
        raise RuntimeError("ordinary eval loader failed an earlier bound CAL/final candidate")
    resume_stale_78 = _expect_resume_rejection(
        cal78,
        config,
        config_sha256,
        prepared_sha256,
        schedule,
        ledger_780,
        "stale checkpoint",
    )
    return {
        "ordinary_loader": "load_current_checkpoint_for_eval",
        "ordinary_loader_does_not_consult_training_ledger": True,
        "valid_current_bound_accepted": True,
        "wrong_expected_step_rejected": _expect_eval_loader_rejection(
            bound_path,
            config,
            config_sha256,
            prepared_sha256,
            78,
            schedule,
            "expected step",
        ),
        "wrong_config_rejected": _expect_eval_loader_rejection(
            foreign_path,
            config,
            config_sha256,
            prepared_sha256,
            2,
            schedule,
            "config identity",
        ),
        "unbound_legacy_rejected": _expect_eval_loader_rejection(
            historical_path,
            config,
            config_sha256,
            prepared_sha256,
            2,
            schedule,
            "unbound historical",
        ),
        "historical_explicit_hash_reader_accepted": True,
        "historical_fixture_sha256": historical_sha256,
        "historical_fixture_role": historical_loaded["fixture_role"],
        "end_of_training_ledger_completed": 780,
        "cal_step78_accepted_with_ledger_780": True,
        "selected_step390_accepted_with_ledger_780": True,
        "resume_rejects_stale_step78_versus_ledger_780": resume_stale_78,
    }

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
    config_sha256 = sha256_file(config_path)
    prepared_sha256 = sha256_file(run_root / "prepared_sources.pt")
    started = time.perf_counter()
    deadline = float(config["resource_guard"]["per_run_wall_seconds"])
    max_reserved = int(config["resource_guard"]["max_gpu_allocator_reserved_bytes"])
    chunk = int(config["training"]["chunk_samples"])
    objective = config["training"]["objective"]
    class_weights = None
    if objective["name"] == OBJECTIVE_SOLO_CONTRAST:
        class_weights = solo_class_weight_tensor(objective, device)
    if resume:
        if not latest_path.exists():
            raise RuntimeError("resume requested without checkpoint_latest.pt")
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        validate_bound_checkpoint_for_resume(
            checkpoint,
            config,
            config_sha256,
            prepared_sha256,
            schedule,
            read_ledger(ledger_path),
        )
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
        save_checkpoint(
            latest_path,
            config,
            model,
            optimizer,
            None,
            0,
            None,
            None,
            config_sha256=config_sha256,
            prepared_sha256=prepared_sha256,
        )
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
            terms = training_objective_terms(
                logits.squeeze(0),
                targets,
                validity,
                objective=objective,
                class_weights=class_weights,
            )
            loss = terms["total_objective"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config["training"]["optimizer"]["gradient_clip_norm"]),
            )
            loss_value = float(loss.detach().cpu())
            activity_bce_value = float(terms["activity_bce"].detach().cpu())
            solo_ce_value = float(terms["solo_ce"].detach().cpu())
            grad_value = float(grad_norm.detach().cpu())
            if not all(
                math.isfinite(value)
                for value in (loss_value, activity_bce_value, solo_ce_value, grad_value)
            ):
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
                "total_objective": loss_value,
                "activity_bce": activity_bce_value,
                "solo_ce": solo_ce_value,
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
                "total_objective": loss_value,
                "activity_bce": activity_bce_value,
                "solo_ce": solo_ce_value,
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
                latest_path,
                config,
                model,
                optimizer,
                state,
                completed,
                cursor,
                witness,
                config_sha256=config_sha256,
                prepared_sha256=prepared_sha256,
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
                        "timing_scope": "wall from perf_counter after bind/load config, authorization, prepared load, schedule build, GPU require, and path setup through pause checkpoint, including host/device transfers and optimizer updates; excludes that pre-timer setup. Allocator peak reset is after resume restore/witness or after the initial zero-update checkpoint on a fresh train.",
                    },
                )
                return
    if optimizer_step_count(optimizer) != {780}:
        raise RuntimeError("optimizer counters do not equal 780 after the full allocation")
    digest = save_checkpoint(
        latest_path,
        config,
        model,
        optimizer,
        state,
        780,
        cursor,
        None,
        config_sha256=config_sha256,
        prepared_sha256=prepared_sha256,
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
            "timing_scope": "wall from perf_counter after bind/load config, authorization, prepared load, schedule build, GPU require, and path setup through final checkpoint, including host/device transfers and optimizer updates; excludes that pre-timer setup. Allocator peak reset is after resume restore/witness or after the initial zero-update checkpoint on a fresh train.",
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


def loss_probe_stage(config_path: Path, run_root: Path) -> None:
    started = time.perf_counter()
    config_path = bind_pilot_config(config_path, run_root)
    config = load_pilot_config(config_path)
    objective = config["training"]["objective"]
    if objective["name"] != OBJECTIVE_SOLO_CONTRAST:
        raise RuntimeError("loss-probe requires the solo contrast objective")
    weights = solo_class_weight_tensor(objective)
    graphs: list[dict[str, Any]] = []
    vjps = 0

    def run_graph(
        name: str,
        logits: Tensor,
        targets: Tensor,
        validity: Tensor,
        *,
        expected_solo: float | None = None,
        expect_zero_solo: bool = False,
    ) -> dict[str, Tensor]:
        nonlocal vjps
        if int(logits.numel()) > 32:
            raise RuntimeError("toy logit graph exceeds 32 logits")
        terms = training_objective_terms(
            logits,
            targets,
            validity,
            objective=objective,
            class_weights=weights,
        )
        solo_value = float(terms["solo_ce"].detach())
        if expect_zero_solo and solo_value != 0.0:
            raise RuntimeError(f"{name} solo_ce is not zero")
        if expected_solo is not None and abs(solo_value - expected_solo) > 1.0e-6:
            raise RuntimeError(f"{name} solo_ce {solo_value} != {expected_solo}")
        terms["total_objective"].backward()
        vjps += 1
        if logits.grad is None or not bool(torch.isfinite(logits.grad).all()):
            raise RuntimeError(f"{name} toy VJP is not finite")
        graphs.append(
            {
                "name": name,
                "activity_bce": float(terms["activity_bce"].detach()),
                "solo_ce": solo_value,
                "total_objective": float(terms["total_objective"].detach()),
                "solo_bins": int(stable_solo_mask(targets, validity).sum().item()),
                "valid_bins": int(validity.sum().item()),
                "logit_count": int(logits.numel()),
                "grad_finite": True,
                "synthetic_logit_autograd_vjp": True,
            }
        )
        return terms

    silence_logits = torch.zeros(4, 4, requires_grad=True)
    run_graph(
        "empty_solo_silence",
        silence_logits,
        torch.zeros(4, 4),
        torch.ones(4, dtype=torch.bool),
        expect_zero_solo=True,
    )
    fractional_logits = torch.zeros(2, 4, requires_grad=True)
    run_graph(
        "fractional_overlap_not_solo",
        fractional_logits,
        torch.tensor([[0.5, 0.5, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]]),
        torch.ones(2, dtype=torch.bool),
        expect_zero_solo=True,
    )
    single_logits = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    single_ce = float(
        nn.functional.cross_entropy(
            torch.tensor([[2.0, 0.0, 0.0, 0.0]]), torch.tensor([0])
        )
    )
    weight0 = float(weights[0])
    single_expected = (weight0 * single_ce) / 2.0
    if abs(single_expected - single_ce) <= 1.0e-6:
        raise RuntimeError("single-class case does not distinguish sum/valid from weighted mean")
    run_graph(
        "single_class_plus_silence",
        single_logits,
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        torch.tensor([True, True]),
        expected_solo=single_expected,
    )
    invalid_logits = torch.tensor(
        [[2.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]], requires_grad=True
    )
    invalid_expected = weight0 * single_ce
    run_graph(
        "invalid_excluded_from_denominator",
        invalid_logits,
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([True, False]),
        expected_solo=invalid_expected,
    )
    two_logits = torch.tensor(
        [[3.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]], requires_grad=True
    )
    two_ce0 = float(
        nn.functional.cross_entropy(
            torch.tensor([[3.0, 0.0, 0.0, 0.0]]), torch.tensor([0])
        )
    )
    two_ce1 = float(
        nn.functional.cross_entropy(
            torch.tensor([[0.0, 3.0, 0.0, 0.0]]), torch.tensor([1])
        )
    )
    two_expected = (weight0 * two_ce0 + float(weights[1]) * two_ce1) / 2.0
    run_graph(
        "two_class_solo",
        two_logits,
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        torch.ones(2, dtype=torch.bool),
        expected_solo=two_expected,
    )
    all_solo_logits = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], requires_grad=True
    )
    all_solo_ce = float(
        nn.functional.cross_entropy(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.tensor([0])
        )
    )
    all_solo_expected = weight0 * all_solo_ce
    if abs(all_solo_expected - all_solo_ce) <= 1.0e-6:
        raise RuntimeError("all-solo one-class case cancels weights under a weighted mean")
    run_graph(
        "all_solo_one_class_weight_not_cancelled",
        all_solo_logits,
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        torch.ones(2, dtype=torch.bool),
        expected_solo=all_solo_expected,
    )
    overlap_logits = torch.zeros(1, 4, requires_grad=True)
    run_graph(
        "two_full_ones_not_solo",
        overlap_logits,
        torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
        torch.ones(1, dtype=torch.bool),
        expect_zero_solo=True,
    )
    mixed_logits = torch.zeros(4, 4, requires_grad=True)
    mixed_targets = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    mixed_validity = torch.tensor([True, True, True, False])
    mixed_ce = float(
        nn.functional.cross_entropy(
            torch.zeros(1, 4), torch.tensor([0])
        )
    )
    mixed_expected = (weight0 * mixed_ce) / 3.0
    run_graph(
        "mixed_solo_silence_fractional_invalid",
        mixed_logits,
        mixed_targets,
        mixed_validity,
        expected_solo=mixed_expected,
    )
    if len(graphs) > 8 or vjps > 8:
        raise RuntimeError("loss-probe exceeded toy graph/VJP caps")
    repo = Path(__file__).resolve().parent.parent.parent
    origin_prepared = repo / config["prepared_import"]["origin_prepared_path"]
    if sha256_file(origin_prepared) != config["prepared_import"]["origin_prepared_sha256"]:
        raise RuntimeError("origin prepared identity mismatch during loss-probe")
    prepared = torch.load(origin_prepared, map_location="cpu", weights_only=False)
    fit_ids = [source["source_id"] for source in config["sources"]["FIT"]]
    counted = fit_solo_class_counts(prepared, fit_ids)
    if counted != list(FIT_SOLO_CLASS_COUNTS):
        raise RuntimeError(f"FIT solo counts {counted} != {list(FIT_SOLO_CLASS_COUNTS)}")
    posix_path = run_root_posix_relative(
        run_root, run_root / "eval_logits" / "untrained_seed20260915_ami_ES2005a.pt"
    )
    if posix_path != "eval_logits/untrained_seed20260915_ami_ES2005a.pt":
        raise RuntimeError("eval logit entry.path is not POSIX-relative to the run root")
    elapsed = time.perf_counter() - started
    if elapsed > 120:
        raise RuntimeError("loss-probe exceeded 120s CPU wall")
    atomic_json(
        run_root / "loss_probe_receipt.json",
        {
            "stage": "loss-probe",
            "optimizer_steps": 0,
            "student_constructors": 0,
            "model_forward": 0,
            "model_backward": 0,
            "synthetic_logit_graphs": len(graphs),
            "synthetic_logit_autograd_vjp_calls": vjps,
            "maximum_logits_per_graph": max(row["logit_count"] for row in graphs),
            "graphs": graphs,
            "fit_solo_class_counts": counted,
            "class_weights_float64": list(FIT_SOLO_CLASS_WEIGHTS),
            "eval_logits_manifest": EVAL_LOGITS_MANIFEST_NAME,
            "eval_logits_entry_path_example": posix_path,
            "init_logit_match_tolerance": INIT_LOGIT_MATCH_TOLERANCE,
            "elapsed_seconds": elapsed,
            "runtime": pilot_runtime(config_path),
        },
    )


def probe_stage(config_path: Path, run_root: Path) -> None:
    config_path = bind_pilot_config(config_path, run_root)
    config = load_pilot_config(config_path)
    atomic_json(
        run_root / "probe_receipt.json",
        {
            "stage": "probe",
            "optimizer_steps": 0,
            "model_forward": False,
            "model_backward": False,
            "synthetic_metric_probes": synthetic_metric_probes(),
            "checkpoint_identity_probes": checkpoint_identity_probes(config),
            "current_checkpoint_eval_loader_probes": current_checkpoint_eval_loader_probes(
                config, run_root / "checkpoint_loader_fixtures"
            ),
            "runtime": pilot_runtime(config_path),
        },
    )


def repair_evaluate_stage(config_path: Path, run_root: Path) -> None:
    config_path = bind_pilot_config(config_path, run_root)
    config = load_pilot_config(config_path)
    require_authorization()
    repair = config["metric_repair"]
    origin = (Path(__file__).resolve().parent.parent.parent / repair["origin_run_root"]).resolve()
    if origin == run_root.resolve():
        raise RuntimeError("repair output root must not overwrite the origin runroot")
    origin_prepared = origin / "prepared_sources.pt"
    origin_frozen = origin / "frozen_config.json"
    if sha256_file(origin_prepared) != repair["origin_prepared_sources_sha256"]:
        raise RuntimeError("origin prepared identity mismatch")
    if sha256_file(origin_frozen) != repair["origin_config_sha256"]:
        raise RuntimeError("origin frozen config identity mismatch")
    evaluator_config_sha256 = sha256_file(config_path)
    if evaluator_config_sha256 == repair["origin_config_sha256"]:
        raise RuntimeError("repair evaluator config must be distinct from the origin frozen config")
    prepared = torch.load(origin_prepared, map_location="cpu", weights_only=False)
    device = require_gpu(config)
    max_reserved = int(repair["bounds"]["max_gpu_allocator_reserved_bytes"])
    deadline = time.perf_counter() + float(repair["bounds"]["wall_seconds"])
    chunk = int(config["training"]["chunk_samples"])
    threshold = float(config["evaluation"]["operating_point"])
    fit_ids = [source["source_id"] for source in config["sources"]["FIT"]]
    cal_ids = [source["source_id"] for source in config["sources"]["CAL"]]
    dev_ids = [source["source_id"] for source in config["sources"]["DEV"]]
    logits_root = run_root / "replay_logits"
    logits_root.mkdir(parents=True, exist_ok=True)
    source_passes = 0

    def run_pass(
        model: StreamingStudent,
        source_ids: list[str],
        weight_id: str,
        checkpoint_sha256: str | None,
    ) -> dict[str, Any]:
        nonlocal source_passes
        results: dict[str, Any] = {}
        model.eval()
        with torch.no_grad():
            for source_id in source_ids:
                if time.perf_counter() > deadline:
                    raise RuntimeError("repair wall cap exhausted")
                payload = prepared["sources"][source_id]
                logits, frontiers = stream_source(
                    model, payload["waveform"], device, chunk
                )
                if torch.cuda.max_memory_reserved(device) > max_reserved:
                    raise RuntimeError("repair GPU reserved cap exceeded")
                if not torch.equal(frontiers, payload["frontiers"]):
                    raise RuntimeError(f"{source_id} repair frontiers differ")
                atomic_torch_save(
                    logits_root / f"{weight_id}_{source_id}.pt",
                    {
                        "schema": "PSEM-ISSUE-164-GT-PILOT-REPAIR-LOGITS-1",
                        "weight_id": weight_id,
                        "checkpoint_sha256": checkpoint_sha256,
                        "source_id": source_id,
                        "logits": logits.to(dtype=torch.float32).contiguous(),
                        "frontiers": frontiers,
                        "origin_prepared_sha256": repair["origin_prepared_sources_sha256"],
                        "evaluator_config_sha256": evaluator_config_sha256,
                        "evaluator_implementation_sha256": dict(config["implementation_sha256"]),
                    },
                )
                results[source_id] = {
                    "label": weight_id,
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
                source_passes += 1
        return results

    untrained_model = build_fresh_model(config, device)
    untrained = run_pass(untrained_model, fit_ids + cal_ids + dev_ids, "untrained_seed20260915", None)
    cal_rows: dict[str, Any] = {}
    for step in config["evaluation"]["calibration_checkpoint_steps"]:
        path = origin / f"checkpoint_step_{step}.pt"
        expected = repair["checkpoint_sha256"][str(step)]
        payload = load_historical_checkpoint_for_eval(path, expected)
        model = StreamingStudent().to(device)
        model.load_state_dict(payload["model"])
        cal_rows[str(step)] = {
            "checkpoint_sha256": expected,
            "cal": run_pass(model, cal_ids, f"checkpoint_{step}", expected),
        }
    selected_step = int(repair["frozen_selected_step"])
    selected_sha = repair["checkpoint_sha256"][str(selected_step)]
    selected_payload = load_historical_checkpoint_for_eval(
        origin / f"checkpoint_step_{selected_step}.pt", selected_sha
    )
    selected_model = StreamingStudent().to(device)
    selected_model.load_state_dict(selected_payload["model"])
    selected = run_pass(
        selected_model, fit_ids + dev_ids, f"checkpoint_{selected_step}", selected_sha
    )
    if source_passes != int(repair["maximum_source_passes"]):
        raise RuntimeError(
            f"repair source passes {source_passes} != {repair['maximum_source_passes']}"
        )
    origin_initial = json.loads((origin / "initial_eval.json").read_text(encoding="utf-8"))
    origin_cal = json.loads((origin / "cal_selection.json").read_text(encoding="utf-8"))
    origin_final = json.loads((origin / "final_eval.json").read_text(encoding="utf-8"))
    agreement = float(repair["activity_bce_agreement"])
    bce_comparisons = []

    def agree(label: str, source_id: str, actual: float, expected: float) -> None:
        delta = abs(actual - expected)
        bce_comparisons.append(
            {
                "label": label,
                "source_id": source_id,
                "repaired": actual,
                "origin": expected,
                "abs_delta": delta,
            }
        )
        if delta > agreement:
            raise RuntimeError(f"{label} {source_id} BCE delta {delta} exceeds {agreement}")

    for source_id, row in untrained.items():
        agree(
            "untrained",
            source_id,
            row["canonical_masked_bce"],
            origin_initial["untrained"][source_id]["canonical_masked_bce"],
        )
    for step, row in cal_rows.items():
        source_id = cal_ids[0]
        agree(
            f"cal_{step}",
            source_id,
            row["cal"][source_id]["canonical_masked_bce"],
            origin_cal["candidates"][step]["cal"][source_id]["canonical_masked_bce"],
        )
    for source_id, row in selected.items():
        agree(
            f"selected_{selected_step}",
            source_id,
            row["canonical_masked_bce"],
            origin_final["scores"][source_id]["canonical_masked_bce"],
        )
    atomic_json(
        run_root / "repair_eval.json",
        {
            "stage": "repair_evaluate",
            "optimizer_steps": 0,
            "source_passes": source_passes,
            "origin_run_root": str(origin),
            "origin_prepared_sha256": repair["origin_prepared_sources_sha256"],
            "origin_config_sha256": repair["origin_config_sha256"],
            "evaluator_config_sha256": evaluator_config_sha256,
            "frozen_selected_step": selected_step,
            "untrained": untrained,
            "cal": cal_rows,
            "selected_fit_dev": selected,
            "activity_bce_agreement": bce_comparisons,
            "runtime": pilot_runtime(config_path),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "check",
            "prepare",
            "import-prepared",
            "train",
            "evaluate",
            "report",
            "probe",
            "loss-probe",
            "repair-evaluate",
        ),
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
    elif args.stage == "import-prepared":
        import_prepared_stage(args.config.resolve(), args.run_root.resolve())
    elif args.stage == "train":
        train_stage(
            args.config.resolve(),
            args.run_root.resolve(),
            resume=args.resume,
            pause_after=args.pause_after,
        )
    elif args.stage == "evaluate":
        evaluate_stage(args.config.resolve(), args.run_root.resolve(), args.which)
    elif args.stage == "probe":
        probe_stage(args.config.resolve(), args.run_root.resolve())
    elif args.stage == "loss-probe":
        loss_probe_stage(args.config.resolve(), args.run_root.resolve())
    elif args.stage == "repair-evaluate":
        repair_evaluate_stage(args.config.resolve(), args.run_root.resolve())
    elif args.stage == "report":
        report_stage(args.config.resolve(), args.run_root.resolve())
    else:
        raise RuntimeError(f"unknown stage {args.stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
