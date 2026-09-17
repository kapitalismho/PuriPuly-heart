from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCHEMA = "PSEM-SAVED-SPEAKER-SCORE-DIAGNOSTIC-1"
OUTPUT_SLOTS = 4
HOP = 1280
FIT_PRIOR = (
    0.26317110657691956,
    0.10770361125469208,
    0.11625641584396362,
    0.06539624184370041,
)
HERE = Path(__file__).resolve().parent
ORIGIN = HERE / ".stage-control" / "issue-164-gt-pilot-1"
REPLAY = HERE / ".stage-control" / "issue-164-gt-pilot-metric-repair-1"
WORK = HERE / ".stage-control" / "issue-164-speaker-score-diagnostic-1"
CONTRACT = HERE / ".stage-control" / "autonomous-research" / "speaker-score-diagnostic-contract.json"
MANIFEST = WORK / "input_manifest.json"
RESULT = HERE / "SPEAKER_SCORE_DIAGNOSTIC_RESULT.json"
FIT_IDS = ("ami_ES2005a", "ami_ES2006a", "ami_ES2007a", "ami_ES2008a")
CAL_IDS = ("ami_ES2010a",)
DEV_IDS = ("ami_ES2009d", "ami_EN2009d")
ALL_IDS = FIT_IDS + CAL_IDS + DEV_IDS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def finite(value: float) -> bool:
    return value == value and value != float("inf") and value != float("-inf")


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if not finite(value):
            raise RuntimeError(f"non-finite float {value}")
        return float(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    raise RuntimeError(f"unserializable {type(value)}")


def as_numpy(value: torch.Tensor) -> np.ndarray:
    array = value.detach().cpu().contiguous().numpy()
    if array.dtype == np.bool_ or str(array.dtype).startswith("int"):
        return array
    return np.asarray(array, dtype=np.float64)


def logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    peak = values.max(axis=axis, keepdims=True)
    reduced = peak.squeeze(axis) + np.log(np.exp(values - peak).sum(axis=axis))
    return reduced


def log_softmax(logits: np.ndarray) -> np.ndarray:
    return logits - logsumexp(logits, axis=1)[:, None]


def softmax(logits: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(logits))


def sigmoid(logits: np.ndarray) -> np.ndarray:
    result = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    expv = np.exp(logits[~positive])
    result[~positive] = expv / (1.0 + expv)
    return result


def independent_bce(logits: np.ndarray, targets: np.ndarray, valid: np.ndarray) -> float:
    z = logits[valid]
    y = targets[valid]
    return float(np.mean(np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def independent_brier(logits: np.ndarray, targets: np.ndarray, valid: np.ndarray) -> float:
    return float(np.mean((sigmoid(logits[valid]) - targets[valid]) ** 2))


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        mean_rank = (index + end - 1) / 2.0 + 1.0
        ranks[order[index:end]] = mean_rank
        index = end
    return ranks


def roc_auc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    if positive.size == 0 or negative.size == 0:
        return None
    ranks = average_ranks(np.concatenate([positive, negative]))
    p_ranks = ranks[: positive.size]
    return float((p_ranks.sum() - positive.size * (positive.size + 1) / 2.0) / (positive.size * negative.size))


def logit_from_probability(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1.0e-12, 1.0 - 1.0e-12)
    return np.log(clipped) - np.log1p(-clipped)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def solo_partition(targets: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    ones = targets == 1.0
    zeros = targets == 0.0
    solo = valid & (ones.sum(axis=1) == 1) & (zeros.sum(axis=1) == OUTPUT_SLOTS - 1)
    all_zero = valid & zeros.all(axis=1)
    two_or_more_full = valid & (ones.sum(axis=1) >= 2)
    fractional = valid & np.any((targets > 0.0) & (targets < 1.0), axis=1)
    other = valid & ~solo & ~all_zero & ~two_or_more_full & ~fractional
    labels = np.full(targets.shape[0], -1, dtype=np.int64)
    labels[solo] = ones[solo].argmax(axis=1)
    return {
        "activity_valid": int(valid.sum()),
        "solo_known_complete": int(solo.sum()),
        "excluded_from_solo_diagnostic": int(valid.sum() - solo.sum()),
        "all_zero": int(all_zero.sum()),
        "two_or_more_full_ones": int(two_or_more_full.sum()),
        "any_fractional_occupancy": int(fractional.sum()),
        "other_nonzero_incomplete": int(other.sum()),
        "solo_mask": solo,
        "true_slot": labels,
    }


def mean_or_none(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return float(sum(present) / len(present))


def oracle_permutation(logits: np.ndarray, true_slot: np.ndarray, solo: np.ndarray) -> dict[str, Any]:
    best_perm = None
    best_ba = None
    for perm in itertools.permutations(range(OUTPUT_SLOTS)):
        mapped = logits[:, list(perm)]
        metrics = conditional_core(mapped, true_slot, solo)
        ba = metrics["balanced_accuracy"]
        if ba is None:
            continue
        if best_ba is None or ba > best_ba:
            best_ba = ba
            best_perm = list(perm)
    return {
        "role": "source_global_oracle_diagnostic_only",
        "not_used_for_inference_or_selection": True,
        "not_per_chunk_remapping": True,
        "permutation": best_perm,
        "balanced_accuracy": best_ba,
        "permutation_count": 24,
    }


def conditional_core(logits: np.ndarray, true_slot: np.ndarray, solo: np.ndarray) -> dict[str, Any]:
    n_solo = int(solo.sum())
    counts = [int((solo & (true_slot == slot)).sum()) for slot in range(OUTPUT_SLOTS)]
    observed = [slot for slot, count in enumerate(counts) if count > 0]
    absent = [slot for slot, count in enumerate(counts) if count == 0]
    recalls: list[float | None] = [None] * OUTPUT_SLOTS
    class_nll: list[float | None] = [None] * OUTPUT_SLOTS
    ovr: list[float | None] = [None] * OUTPUT_SLOTS
    confusion = [[0] * OUTPUT_SLOTS for _ in range(OUTPUT_SLOTS)]
    ordinary_nll = None
    balanced_nll = None
    first = {
        "n_slot0": counts[0],
        "n_rest": n_solo - counts[0],
        "auc": None,
        "ordinary_binary_log_loss": None,
        "class_balanced_binary_log_loss": None,
    }
    if n_solo:
        selected = logits[solo]
        labels = true_slot[solo]
        predicted = selected.argmax(axis=1)
        for truth, pred in zip(labels.tolist(), predicted.tolist()):
            confusion[truth][pred] += 1
        log_prob = log_softmax(selected)
        nll = -log_prob[np.arange(n_solo), labels]
        ordinary_nll = float(nll.mean())
        probs = np.exp(log_prob)
        for slot in range(OUTPUT_SLOTS):
            denom = counts[slot]
            if denom == 0:
                continue
            recalls[slot] = float(confusion[slot][slot] / denom)
            class_nll[slot] = float(nll[labels == slot].mean())
            ovr[slot] = roc_auc(probs[labels == slot, slot], probs[labels != slot, slot])
        balanced_nll = mean_or_none(class_nll)
        lse_all = logsumexp(selected, axis=1)
        lse_other = logsumexp(selected[:, 1:], axis=1)
        binary_nll = np.empty(n_solo, dtype=np.float64)
        slot0 = labels == 0
        binary_nll[slot0] = lse_all[slot0] - selected[slot0, 0]
        rest = ~slot0
        binary_nll[rest] = lse_all[rest] - lse_other[rest]
        if counts[0] and counts[0] < n_solo:
            first["auc"] = roc_auc(probs[slot0, 0], probs[rest, 0])
            first["ordinary_binary_log_loss"] = float(binary_nll.mean())
            first["class_balanced_binary_log_loss"] = float(
                0.5 * binary_nll[slot0].mean() + 0.5 * binary_nll[rest].mean()
            )
        elif counts[0] == 0 or counts[0] == n_solo:
            first["auc"] = None
            if n_solo:
                first["ordinary_binary_log_loss"] = float(binary_nll.mean())
    return {
        "solo_bins": n_solo,
        "class_counts": counts,
        "observed_slots": observed,
        "absent_slots": absent,
        "confusion": confusion if n_solo else None,
        "per_observed_slot_recall": recalls,
        "recall_denominators": counts,
        "balanced_accuracy": mean_or_none(recalls),
        "ordinary_conditional_log_loss": ordinary_nll,
        "class_balanced_conditional_log_loss": balanced_nll,
        "ovr_auc": ovr,
        "ovr_auc_denominators": [{"positive": count, "negative": n_solo - count} for count in counts],
        "first_slot0_vs_rest": first,
    }


def score_logits(logits: np.ndarray, true_slot: np.ndarray, solo: np.ndarray) -> dict[str, Any]:
    core = conditional_core(logits, true_slot, solo)
    core["oracle_permutation"] = oracle_permutation(logits, true_slot, solo)
    return core


def centered_logits(logits: np.ndarray) -> np.ndarray:
    return logits - logits.mean(axis=1, keepdims=True)


def common_control(logits: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return logits.mean(axis=1, keepdims=True) + bias[None, :]


def temporal_centered_stats(logits: np.ndarray, valid: np.ndarray) -> list[dict[str, float]]:
    centered = centered_logits(logits[valid])
    rows = []
    for slot in range(OUTPUT_SLOTS):
        column = centered[:, slot]
        rows.append(
            {
                "slot": slot,
                "std": float(column.std(ddof=0)),
                "range": float(column.max() - column.min()) if column.size else 0.0,
            }
        )
    return rows


def constant_prior_logits(frames: int) -> np.ndarray:
    prior = logit_from_probability(np.asarray(FIT_PRIOR, dtype=np.float64))
    return np.broadcast_to(prior[None, :], (frames, OUTPUT_SLOTS)).copy()


def uniform_logits(frames: int) -> np.ndarray:
    return np.zeros((frames, OUTPUT_SLOTS), dtype=np.float64)


def logic_probes() -> dict[str, Any]:
    perfect_pos = np.array([3.0, 2.0, 1.0], dtype=np.float64)
    perfect_neg = np.array([0.0, -1.0, -2.0], dtype=np.float64)
    reversed_pos = np.array([0.0, -1.0], dtype=np.float64)
    reversed_neg = np.array([3.0, 2.0], dtype=np.float64)
    tied = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    require(roc_auc(perfect_pos, perfect_neg) == 1.0, "perfect AUC failed")
    require(roc_auc(reversed_pos, reversed_neg) == 0.0, "reversed AUC failed")
    require(roc_auc(tied, tied) == 0.5, "tied AUC failed")
    require(roc_auc(perfect_pos, np.array([], dtype=np.float64)) is None, "absent-negative AUC must be null")
    require(roc_auc(np.array([], dtype=np.float64), perfect_neg) is None, "absent-positive AUC must be null")

    logits = np.array(
        [
            [4.0, 0.0, 1.0, 2.0],
            [3.0, 0.0, 0.0, 5.0],
            [0.0, 6.0, 0.0, 1.0],
            [0.0, 5.0, 0.0, 4.0],
        ],
        dtype=np.float64,
    )
    true = np.array([0, 0, 1, 1], dtype=np.int64)
    solo = np.array([True, True, True, True])
    core = conditional_core(logits, true, solo)
    require(core["absent_slots"] == [2, 3], "unused heads must be absent/null")
    require(core["per_observed_slot_recall"][2] is None, "absent recall must be null")
    require(core["ovr_auc"][2] is None, "absent AUC must be null")
    require(core["confusion"][0][3] == 1, "prediction into unused head must count")
    require(core["per_observed_slot_recall"][0] == 0.5, "unused-head prediction must reduce observed recall")
    require(core["balanced_accuracy"] == 0.75, "observed-class balanced accuracy failed")

    shifted = logits + np.array([10.0, -3.0, 1.5, 8.0])[:, None]
    require(np.allclose(softmax(logits), softmax(shifted), atol=1e-12), "softmax shift invariance failed")
    require(
        np.allclose(
            conditional_core(shifted, true, solo)["ordinary_conditional_log_loss"],
            core["ordinary_conditional_log_loss"],
            atol=1e-12,
        ),
        "conditional log loss must be shift-invariant",
    )

    activity = np.array([2.0, -1.0, 4.0, 0.5], dtype=np.float64)
    bias = np.array([0.2, -0.1, 0.05, -0.15], dtype=np.float64)
    control = activity[:, None] + bias[None, :]
    control_prob = softmax(control)
    require(np.allclose(control_prob, control_prob[0][None, :], atol=1e-12), "common-logit ranking must be constant")
    require(len(set(control.argmax(axis=1).tolist())) == 1, "common-logit argmax must be constant")
    require(float(centered_logits(control).std(axis=0).max()) < 1e-15, "common-logit centered variation must be zero")

    chunked_true = np.array([0, 0, 0, 0], dtype=np.int64)
    chunked_logits = np.array(
        [
            [5.0, 0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    global_oracle = oracle_permutation(chunked_logits, chunked_true, solo)
    require(global_oracle["balanced_accuracy"] == 0.5, "global permutation must not recover mixed chunk maps")
    require(global_oracle["permutation"] == [0, 1, 2, 3], "lexicographic global permutation tie-break failed")
    chunk_pred = np.array([0, 0, 0, 0], dtype=np.int64)
    chunk_pred[:2] = chunked_logits[:2][:, [0, 1, 2, 3]].argmax(axis=1)
    chunk_pred[2:] = chunked_logits[2:][:, [1, 0, 2, 3]].argmax(axis=1)
    require(float((chunk_pred == chunked_true).mean()) == 1.0, "per-chunk remap probe construction failed")
    require(global_oracle["not_per_chunk_remapping"] is True, "oracle must refuse per-chunk remapping")

    absent_true = np.array([0, 0, 0, 0], dtype=np.int64)
    absent_core = conditional_core(logits, absent_true, solo)
    require(absent_core["ovr_auc"][0] is None, "single-class OVR AUC must be null")
    require(absent_core["first_slot0_vs_rest"]["auc"] is None, "absent rest class AUC must be null")
    empty_core = conditional_core(logits, true, np.zeros(4, dtype=bool))
    require(empty_core["balanced_accuracy"] is None, "empty solo mask BA must be null")
    require(empty_core["confusion"] is None, "empty solo mask confusion must be null")
    return {
        "perfect_auc": 1.0,
        "reversed_auc": 0.0,
        "tied_auc": 0.5,
        "absent_class_auc_null": True,
        "unused_head_predictions_reduce_observed_recall": True,
        "observed_balanced_accuracy": 0.75,
        "softmax_shift_invariant": True,
        "common_logit_no_within_source_ranking_variation": True,
        "global_permutation_lexicographic": global_oracle["permutation"],
        "global_permutation_cannot_use_per_chunk_maps": True,
        "per_chunk_remap_not_used": True,
        "empty_and_absent_nulls": True,
        "model_object_created": False,
    }


def drop_waveforms(prepared: dict[str, Any]) -> None:
    for payload in prepared["sources"].values():
        payload.pop("waveform", None)


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_prepared(prepared: dict[str, Any], manifest: dict[str, Any], prepared_sha: str) -> dict[str, Any]:
    require(prepared_sha == manifest["prepared_sources_sha256"], "prepared_sources.pt hash mismatch")
    require(list(prepared["fit_constant_prior"]) == list(manifest["fit_only_occupancy_prior"]), "FIT prior mismatch")
    require(tuple(FIT_PRIOR) == tuple(manifest["fit_only_occupancy_prior"]), "frozen FIT prior mismatch")
    identities = {}
    for source_id, expected in manifest["source_zero_prefixes"].items():
        payload = prepared["sources"][source_id]
        require("waveform" not in payload, "waveform key must be dropped before use")
        targets = payload["targets"]
        validity = payload["validity"]
        frontiers = payload["frontiers"]
        require(tuple(targets.shape) == (expected["frames"], OUTPUT_SLOTS), f"{source_id} target shape")
        require(tuple(validity.shape) == (expected["frames"],), f"{source_id} validity shape")
        require(tuple(frontiers.shape) == (expected["frames"],), f"{source_id} frontier shape")
        require(int(validity.sum().item()) == expected["valid_bins"], f"{source_id} valid bin count")
        require(int(frontiers[0].item()) == HOP, f"{source_id} first frontier")
        require(int(frontiers[-1].item()) == expected["prefix_samples"], f"{source_id} last frontier")
        require(int((frontiers[1] - frontiers[0]).item()) == HOP, f"{source_id} hop")
        require(payload["source"]["prefix_samples"] == expected["prefix_samples"], f"{source_id} prefix")
        require(payload["source"]["pilot_role"] == expected["role"], f"{source_id} role")
        identities[source_id] = {
            "role": expected["role"],
            "prefix_samples": expected["prefix_samples"],
            "frames": expected["frames"],
            "valid_bins": expected["valid_bins"],
            "slots": list(payload["slots"]),
            "frontiers_start": HOP,
            "frontiers_end": expected["prefix_samples"],
        }
    require(set(prepared["sources"]) == set(manifest["source_zero_prefixes"]), "source set mismatch")
    return identities


def load_logit_file(path: Path, expected: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    require(path.stat().st_size == expected["bytes"], f"{path.name} size mismatch")
    digest = sha256_file(path)
    require(digest == expected["sha256"], f"{path.name} hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(payload["schema"] == "PSEM-ISSUE-164-GT-PILOT-REPAIR-LOGITS-1", f"{path.name} schema")
    require(payload["source_id"] == expected["source_id"], f"{path.name} source")
    require(payload["weight_id"] == expected["weight_id"], f"{path.name} weight")
    require(payload["checkpoint_sha256"] == expected["checkpoint_sha256"], f"{path.name} checkpoint")
    require(payload["origin_prepared_sha256"] == manifest["prepared_sources_sha256"], f"{path.name} prepared bind")
    require(payload["evaluator_config_sha256"] == manifest["evaluator_config_sha256"], f"{path.name} evaluator")
    logits = payload["logits"]
    frontiers = payload["frontiers"]
    require(logits.dtype == torch.float32, f"{path.name} dtype")
    require(tuple(logits.shape) == (expected["frames"], OUTPUT_SLOTS), f"{path.name} logits shape")
    require(bool(logits.is_contiguous()), f"{path.name} contiguous")
    require(bool(torch.isfinite(logits).all()), f"{path.name} finite")
    require(frontiers.device.type == "cpu" and logits.device.type == "cpu", f"{path.name} not cpu")
    return {
        "path": str(path),
        "sha256": digest,
        "bytes": expected["bytes"],
        "weight_id": payload["weight_id"],
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "source_id": payload["source_id"],
        "logits": as_numpy(logits),
        "frontiers": frontiers.cpu().contiguous(),
    }


def fit_bias(rows: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    chunks = [centered_logits(logits[valid]) for logits, valid in rows]
    stacked = np.concatenate(chunks, axis=0)
    return stacked.mean(axis=0)


def attach_control(
    metrics: dict[str, Any],
    logits: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray,
    bias: np.ndarray | None,
) -> dict[str, Any]:
    metrics["independent_activity_valid_bce"] = independent_bce(logits, targets, valid)
    metrics["independent_activity_valid_brier"] = independent_brier(logits, targets, valid)
    metrics["centered_logit_temporal"] = temporal_centered_stats(logits, valid)
    if bias is None:
        metrics["common_logit_control"] = None
        return metrics
    control = common_control(logits, bias)
    metrics["common_logit_control"] = {
        "b": [float(value) for value in bias.tolist()],
        "independent_activity_valid_bce": independent_bce(control, targets, valid),
        "independent_activity_valid_brier": independent_brier(control, targets, valid),
        "softmax_constant_across_valid_frames": bool(
            np.allclose(softmax(control[valid]), softmax(control[valid])[:1], atol=1e-12)
        ),
        "centered_logit_temporal": temporal_centered_stats(control, valid),
    }
    return metrics


def macro_mean(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = []
    sources = []
    class_counts = 0
    solo_bins = 0
    for row in rows:
        value = row[key]
        if value is None:
            continue
        values.append(float(value))
        sources.append(row["source_id"])
        class_counts += len(row["observed_slots"])
        solo_bins += int(row["solo_bins"])
    return {
        "mean": float(sum(values) / len(values)) if values else None,
        "n_sources_with_value": len(values),
        "sources": sources,
        "observed_class_count_sum": class_counts,
        "solo_bins_sum": solo_bins,
    }


def analyze(manifest: dict[str, Any]) -> dict[str, Any]:
    prepared_path = ORIGIN / "prepared_sources.pt"
    frozen_config = ORIGIN / "frozen_config.json"
    repair_eval = REPLAY / "repair_eval.json"
    require(sha256_file(frozen_config) == manifest["origin_config_sha256"], "origin config hash mismatch")
    require(sha256_file(repair_eval) == manifest["replay_result_sha256"], "repair_eval hash mismatch")
    prepared_sha = sha256_file(prepared_path)
    prepared = torch.load(prepared_path, map_location="cpu", weights_only=False)
    drop_waveforms(prepared)
    identities = verify_prepared(prepared, manifest, prepared_sha)
    logit_rows = {}
    for expected in manifest["logits"]:
        loaded = load_logit_file(REPLAY / "replay_logits" / expected["file"], expected, manifest)
        source_id = loaded["source_id"]
        payload = prepared["sources"][source_id]
        require(torch.equal(loaded["frontiers"], payload["frontiers"]), f"{source_id} frontier mismatch")
        logit_rows[(loaded["weight_id"], source_id)] = loaded
    require(len(logit_rows) == 16, "expected 16 retained logit files")

    partitions = {}
    for source_id in ALL_IDS:
        payload = prepared["sources"][source_id]
        targets = as_numpy(payload["targets"])
        valid = payload["validity"].detach().cpu().contiguous().numpy().astype(bool)
        partitions[source_id] = {
            "targets": targets,
            "valid": valid,
            **solo_partition(targets, valid),
            "slots": list(payload["slots"]),
            "role": payload["source"]["pilot_role"],
        }

    untrained_fit = [
        (logit_rows[("untrained_seed20260915", source_id)]["logits"], partitions[source_id]["valid"])
        for source_id in FIT_IDS
    ]
    selected_fit = [
        (logit_rows[("checkpoint_390", source_id)]["logits"], partitions[source_id]["valid"])
        for source_id in FIT_IDS
    ]
    untrained_b = fit_bias(untrained_fit)
    selected_b = fit_bias(selected_fit)

    def pack(weight_id: str, source_id: str, logits: np.ndarray, bias: np.ndarray | None) -> dict[str, Any]:
        part = partitions[source_id]
        metrics = score_logits(logits, part["true_slot"], part["solo_mask"])
        metrics["source_id"] = source_id
        metrics["role"] = part["role"]
        metrics["weight_id"] = weight_id
        metrics["bin_counts"] = {
            "activity_valid": part["activity_valid"],
            "solo_known_complete": part["solo_known_complete"],
            "excluded_from_solo_diagnostic": part["excluded_from_solo_diagnostic"],
            "all_zero": part["all_zero"],
            "two_or_more_full_ones": part["two_or_more_full_ones"],
            "any_fractional_occupancy": part["any_fractional_occupancy"],
            "other_nonzero_incomplete": part["other_nonzero_incomplete"],
        }
        attach_control(metrics, logits, part["targets"], part["valid"], bias)
        return metrics

    per_source: dict[str, Any] = {}
    for source_id in ALL_IDS:
        part = partitions[source_id]
        frames = part["targets"].shape[0]
        scores = {
            "untrained": pack(
                "untrained_seed20260915",
                source_id,
                logit_rows[("untrained_seed20260915", source_id)]["logits"],
                untrained_b,
            ),
            "constant_fit_prior": pack("constant_fit_prior", source_id, constant_prior_logits(frames), None),
            "uniform": pack("uniform", source_id, uniform_logits(frames), None),
        }
        if source_id in FIT_IDS or source_id in DEV_IDS or source_id in CAL_IDS:
            if ("checkpoint_390", source_id) in logit_rows:
                scores["selected_390"] = pack(
                    "checkpoint_390",
                    source_id,
                    logit_rows[("checkpoint_390", source_id)]["logits"],
                    selected_b,
                )
        if source_id in CAL_IDS:
            scores["cal_78"] = pack(
                "checkpoint_78",
                source_id,
                logit_rows[("checkpoint_78", source_id)]["logits"],
                None,
            )
            scores["cal_780"] = pack(
                "checkpoint_780",
                source_id,
                logit_rows[("checkpoint_780", source_id)]["logits"],
                None,
            )
        per_source[source_id] = {
            "role": part["role"],
            "slots": part["slots"],
            "bin_counts": scores["untrained"]["bin_counts"],
            "scores": scores,
        }

    def collect(role_ids: tuple[str, ...], score_key: str) -> list[dict[str, Any]]:
        return [per_source[source_id]["scores"][score_key] for source_id in role_ids if score_key in per_source[source_id]["scores"]]

    role_macro = {}
    for role, ids in (("FIT", FIT_IDS), ("CAL", CAL_IDS), ("DEV", DEV_IDS)):
        role_macro[role] = {}
        keys = ["untrained", "constant_fit_prior", "uniform", "selected_390"]
        if role == "CAL":
            keys.extend(["cal_78", "cal_780"])
        for key in keys:
            rows = collect(ids, key)
            if not rows:
                continue
            role_macro[role][key] = {
                "balanced_accuracy": macro_mean(rows, "balanced_accuracy"),
                "ordinary_conditional_log_loss": macro_mean(rows, "ordinary_conditional_log_loss"),
                "class_balanced_conditional_log_loss": macro_mean(rows, "class_balanced_conditional_log_loss"),
                "independent_activity_valid_bce": macro_mean(rows, "independent_activity_valid_bce"),
                "independent_activity_valid_brier": macro_mean(rows, "independent_activity_valid_brier"),
            }

    cal_trajectory = {
        "source_id": "ami_ES2010a",
        "untrained": per_source["ami_ES2010a"]["scores"]["untrained"],
        "cal_78": per_source["ami_ES2010a"]["scores"]["cal_78"],
        "cal_390": per_source["ami_ES2010a"]["scores"]["selected_390"],
        "cal_780": per_source["ami_ES2010a"]["scores"]["cal_780"],
        "fit_centered_controls_present_for": ["untrained", "cal_390"],
        "fit_centered_controls_absent_for": ["cal_78", "cal_780"],
    }
    logit_identities = []
    for expected in manifest["logits"]:
        loaded = logit_rows[(expected["weight_id"], expected["source_id"])]
        logit_identities.append(
            {
                "file": expected["file"],
                "sha256": loaded["sha256"],
                "bytes": loaded["bytes"],
                "weight_id": loaded["weight_id"],
                "checkpoint_sha256": loaded["checkpoint_sha256"],
                "source_id": loaded["source_id"],
                "frames": int(loaded["logits"].shape[0]),
                "finite": bool(np.isfinite(loaded["logits"]).all()),
            }
        )
    return {
        "identities": identities,
        "logit_identities": logit_identities,
        "fit_common_logit_b": {
            "untrained": [float(value) for value in untrained_b.tolist()],
            "selected_390": [float(value) for value in selected_b.tolist()],
            "fit_sources": list(FIT_IDS),
            "fitted_on": "valid FIT frames only; CAL/DEV logits unused for b",
        },
        "per_source": per_source,
        "role_macro": role_macro,
        "cal_trajectory": cal_trajectory,
    }


def freeze_record(probe_path: Path) -> dict[str, Any]:
    return {
        "schema": "PSEM-SAVED-SCORE-DIAGNOSTIC-FREEZE-1",
        "analyzer_sha256": sha256_file(Path(__file__).resolve()),
        "contract_sha256": sha256_file(CONTRACT),
        "input_manifest_sha256": sha256_file(MANIFEST),
        "probe_receipt_sha256": sha256_file(probe_path),
        "analyzer_path": str(Path(__file__).resolve()),
        "contract_path": str(CONTRACT),
        "input_manifest_path": str(MANIFEST),
    }


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("probes", "analyze"))
    args = parser.parse_args()
    started = time.perf_counter()
    if args.mode == "probes":
        probes = logic_probes()
        receipt = {
            "schema": "PSEM-SAVED-SCORE-DIAGNOSTIC-PROBE-RECEIPT-1",
            "passed": True,
            "probes": probes,
            "model_instantiation_or_forward": 0,
            "gpu_use": False,
            "optimizer_updates": 0,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "wall_seconds": time.perf_counter() - started,
        }
        path = WORK / "probe_receipt.json"
        atomic_json(path, json_safe(receipt))
        freeze = freeze_record(path)
        atomic_json(WORK / "freeze.json", json_safe(freeze))
        print(json.dumps({"status": "probes_passed", "freeze": freeze}, separators=(",", ":")))
        return 0

    probe_path = WORK / "probe_receipt.json"
    freeze_path = WORK / "freeze.json"
    require(probe_path.exists() and freeze_path.exists(), "probes/freeze must exist before analyze")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    current = freeze_record(probe_path)
    for key in ("analyzer_sha256", "contract_sha256", "input_manifest_sha256", "probe_receipt_sha256"):
        require(freeze[key] == current[key], f"freeze mismatch for {key}")
    probes = json.loads(probe_path.read_text(encoding="utf-8"))
    require(probes["passed"] is True, "probes did not pass")
    manifest = load_manifest(MANIFEST)
    analysis = analyze(manifest)
    wall = time.perf_counter() - started
    result = {
        "schema": SCHEMA,
        "status": "Frozen methods for a model-free diagnostic; not an adoption, new checkpoint selection or full issue164 disposition",
        "authority_id": "AUTONOMY-2",
        "decision_question": "Does the selected GT student's loss improvement contain within-source speaker-sensitive score information beyond common speech activity and static slot priors, despite near-silent threshold0.5 predictions?",
        "methods": {
            "solo_mask": "Original activity validity AND exactly one target column equal to 1 AND all other target columns equal to 0. Mixed/straddle/partial/overlap bins are excluded and counted. Relation-only nonlexical masks do not erase known activity.",
            "conditional_identity": "On solo bins, softmax(logits) is the independent-Bernoulli model conditioned on exactly one active slot. Canonical unthresholded argmax uses first-index ties. Absent classes are JSON null with denominators.",
            "baselines": "Same-initialization untrained scores, constant logit(FIT occupancy prior), and uniform four-slot logits, using identical observed-class denominators.",
            "oracle_permutation": "Maximum observed-class balanced argmax accuracy over all 24 fixed source-global four-slot bijections; lexicographic tie-break; oracle scoring only.",
            "first_speaker_vs_rest": "Threshold-free ROC-AUC and ordinary/class-balanced binary log loss for GT first-appearance slot0 versus other slots using conditional softmax probability of slot0. Not production CURRENT/OTHER.",
            "common_logit_control": "b = mean(logits - row_mean(logits)) over valid FIT frames only, for untrained and selected390 separately. Control logits = a(t)+b with a(t)=row_mean(logits(t)). Independent masked BCE/Brier on all original activity-valid bins. No FIT controls for CAL78/780.",
            "cal_trajectory": "Conditional diagnostics for retained CAL78/390/780 plus initial scores. Original selected390 is unchanged.",
            "numerics": "CPU float64; no bootstrap, threshold sweep, significance claim, or fitted hyperparameters beyond the specified FIT-only common-logit vector b.",
        },
        "baselines": ["untrained_seed20260915", "constant_fit_prior", "uniform"],
        "limitations": [
            "Correlated 80ms frames and two DEV sources do not support statistical generalization or significance claims.",
            "Common-logit control isolates a shared activity-plus-static-slot-bias explanation from relative-head variation; it does not prove voice-identity causality or remove temporal/source confounding.",
            "Source-global permutation is oracle scoring, not inference, per-chunk remapping, or checkpoint selection.",
            "First-appearance slot0 versus rest is a GT-anchor score diagnostic, not production 156 CURRENT/OTHER ownership.",
            "This diagnostic does not change thresholds, model, objective, data, or policy, and does not accept or reject the parallel GT pilot candidate.",
        ],
        "provenance": {
            "contract_sha256": freeze["contract_sha256"],
            "analyzer_sha256": freeze["analyzer_sha256"],
            "input_manifest_sha256": freeze["input_manifest_sha256"],
            "probe_receipt_sha256": freeze["probe_receipt_sha256"],
            "origin_config_sha256": manifest["origin_config_sha256"],
            "prepared_sources_sha256": manifest["prepared_sources_sha256"],
            "replay_result_sha256": manifest["replay_result_sha256"],
            "replay_execution_commit": manifest["replay_execution_commit"],
            "evaluator_config_sha256": manifest["evaluator_config_sha256"],
            "fit_only_occupancy_prior": list(FIT_PRIOR),
            "logit_files": analysis["logit_identities"],
            "command": list(sys.argv),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
        },
        "logic_probes": probes["probes"],
        "identities": analysis["identities"],
        "fit_common_logit_b": analysis["fit_common_logit_b"],
        "per_source": analysis["per_source"],
        "role_macro": analysis["role_macro"],
        "cal_trajectory": analysis["cal_trajectory"],
        "accounting": {
            "model_instantiation_or_forward": 0,
            "new_waveform_model_exposure": 0,
            "backward_calls": 0,
            "optimizer_updates": 0,
            "teacher_passes": 0,
            "gpu_use": False,
            "cumulative_optimizer_updates_remain": 803,
            "source_passes": 0,
            "wall_seconds": wall,
            "external_cost": 0,
        },
        "remaining_uncertainty": [
            "Speaker-sensitive relative-head variation, if present, may still be confounded by source, occupancy, channel, or non-identity temporal structure.",
            "Frame-grid 80ms occupancy is not a voice-identity test and does not resolve sub-frame overlap or production admission.",
            "DEV evidence is two already-exposed prefixes only.",
        ],
        "adoption_or_next_experiment_decision": None,
    }
    atomic_json(RESULT, json_safe(result))
    atomic_json(
        WORK / "execution_receipt.json",
        json_safe(
            {
                "schema": "PSEM-SAVED-SCORE-DIAGNOSTIC-EXECUTION-RECEIPT-1",
                "exit": 0,
                "mode": "analyze",
                "command": list(sys.argv),
                "result_sha256": sha256_file(RESULT),
                "result_bytes": RESULT.stat().st_size,
                "analyzer_sha256": freeze["analyzer_sha256"],
                "contract_sha256": freeze["contract_sha256"],
                "input_manifest_sha256": freeze["input_manifest_sha256"],
                "wall_seconds": wall,
                "model_instantiation_or_forward": 0,
                "gpu_use": False,
                "optimizer_updates": 0,
                "finite_result": True,
            }
        ),
    )
    print(
        json.dumps(
            {
                "status": "analyze_complete",
                "result_sha256": sha256_file(RESULT),
                "result_bytes": RESULT.stat().st_size,
                "wall_seconds": wall,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
