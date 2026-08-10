from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import uuid4

import numpy as np

from experiments.speaker_representation_scd.execution_guard import (
    ExecutionGuardError,
    action_receipt_is_authoritative,
    validate_worker_execution,
)
from experiments.speaker_representation_scd.provenance import (
    load_json,
    self_sha256_valid,
    sha256_file,
    with_self_sha256,
)
from experiments.speaker_representation_scd.r1_gate import EXPERIMENT_ROOT, REPOSITORY_ROOT
from experiments.speaker_representation_scd.r3_gate import AUTHORITY, validate_r3_gate
from experiments.speaker_representation_scd.run_provenance import run_provenance

CONTEXTS_MS = (100, 300, 500)
PRIMARY_CONTEXT_MS = 300
FALLBACK_CONTEXT_MS = 500
PROTOTYPE_OFFSET_RANGE_MS = (-1000, -100)
OLD_SPEAKER_CLASSES = ("entirely_old",)
BATCH_SIZE = 32
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 0
HISTOGRAM_BINS = 101

SSL_MODEL_IDS = ("mhubert-147", "wavlm-base-plus", "unispeech-sat-base-plus")
ERES_MODEL_ID = "eres2netv2-standard-prepool"
REGISTRY_LAYER_ORDER = {
    "mhubert-147": ("L1", "L3", "L6", "L9", "L12"),
    "wavlm-base-plus": ("L1", "L3", "L6", "L9", "L12"),
    "unispeech-sat-base-plus": ("L1", "L3", "L6", "L9", "L12"),
    "eres2netv2-standard-prepool": ("S1", "S2", "S3", "S4", "FUSED"),
}

ANCHOR_INDEX_RELATIVE_PATH = Path("data/r3/legacy_common_gt/anchor_index.jsonl")
ANCHOR_INDEX_MANIFEST_RELATIVE_PATH = Path("data/r3/legacy_common_gt/anchor_index.manifest.json")
POOLED_RELATIVE_DIR = Path("data/r3/legacy_common_gt/pooled")
PROBE_RESULT_RELATIVE_PATH = Path("manifests/r3/legacy_common_gt/probe_{model_id}.json")
PROMOTION_LEDGER_RELATIVE_PATH = Path("manifests/r3/legacy_common_gt/promotion_ledger.json")
MODEL_IDS = (
    "mhubert-147",
    "wavlm-base-plus",
    "unispeech-sat-base-plus",
    "eres2netv2-standard-prepool",
)


class R3ProbeError(RuntimeError):
    pass


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        mean_rank = (index + end - 1) / 2 + 1
        for position in range(index, end):
            ranks[order[position]] = mean_rank
        index = end
    return ranks


def roc_auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    p = np.asarray(list(positive), dtype=np.float64)
    n = np.asarray(list(negative), dtype=np.float64)
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    combined = np.concatenate([p, n])
    ranks = _average_ranks(combined)
    p_ranks = ranks[: len(p)]
    return float((p_ranks.sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def eer(positive: Sequence[float], negative: Sequence[float]) -> float:
    p = np.asarray(list(positive), dtype=np.float64)
    n = np.asarray(list(negative), dtype=np.float64)
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    thresholds = np.unique(np.concatenate([p, n]))
    best = 1.0
    for threshold in thresholds:
        false_negative = float((p < threshold).mean())
        false_positive = float((n >= threshold).mean())
        best = min(best, (false_negative + false_positive) / 2)
    return float(best)


def overlap_coefficient(positive: Sequence[float], negative: Sequence[float]) -> float:
    p = np.asarray(list(positive), dtype=np.float64)
    n = np.asarray(list(negative), dtype=np.float64)
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    positive_hist, edges = np.histogram(p, bins=HISTOGRAM_BINS, range=(0.0, 1.0), density=True)
    negative_hist, _ = np.histogram(n, bins=edges, density=True)
    return float(np.minimum(positive_hist, negative_hist).sum() / HISTOGRAM_BINS)


def wilcoxon_signed_rank_statistic(deltas: Sequence[float]) -> float:
    values = np.asarray(list(deltas), dtype=np.float64)
    values = values[values != 0]
    if len(values) == 0:
        return 0.0
    ranks = _average_ranks(np.abs(values))
    return float(ranks[values > 0].sum())


def quantiles(values: Sequence[float]) -> dict[str, float]:
    data = np.asarray(list(values), dtype=np.float64)
    if len(data) == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "q05": float("nan"),
            "q95": float("nan"),
        }
    return {
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "q05": float(np.quantile(data, 0.05)),
        "q95": float(np.quantile(data, 0.95)),
    }


def normalize_mean(vectors: Iterable[np.ndarray]) -> np.ndarray | None:
    stacked = [np.asarray(vector, dtype=np.float32) for vector in vectors]
    if not stacked:
        return None
    if any(not np.isfinite(vector).all() for vector in stacked):
        return None
    mean = np.mean(np.stack(stacked), axis=0)
    norm = np.linalg.norm(mean)
    if not np.isfinite(norm) or norm <= 0:
        return None
    return (mean / norm).astype(np.float32)


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    numerator = float(np.dot(left, right))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(max(0.0, 1.0 - numerator / denominator))


def bootstrap_block_ci(
    block_positive: dict[str, list[float]],
    block_negative: dict[str, list[float]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    block_ids = sorted(set(block_positive) & set(block_negative))
    if len(block_ids) < 2:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.default_rng(seed)
    macro = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        chosen = rng.choice(block_ids, size=len(block_ids), replace=True)
        macro[replicate] = float(
            np.mean([roc_auc(block_positive[block], block_negative[block]) for block in chosen])
        )
    low, high = np.percentile(macro, [2.5, 97.5])
    return {
        "mean": float(macro.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "replicates": replicates,
        "seed": seed,
    }


def bootstrap_paired_ci(
    pair_deltas: dict[str, float],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    pair_ids = sorted(pair_deltas)
    if len(pair_ids) < 2:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        chosen = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        deltas = np.asarray([pair_deltas[pair] for pair in chosen], dtype=np.float64)
        values[replicate] = float((deltas > 0).mean())
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "replicates": replicates,
        "seed": seed,
    }


def build_negative_prototype(
    anchor_rows: Sequence[dict[str, Any]],
    candidate_id: str,
    z_by_candidate: dict[str, np.ndarray],
) -> np.ndarray | None:
    anchor = next(row for row in anchor_rows if row["candidate_id"] == candidate_id)
    same = [
        row["candidate_id"]
        for row in anchor_rows
        if row["class"] == "negative"
        and row["block_id"] == anchor["block_id"]
        and row["kind"] == anchor["kind"]
        and row["candidate_id"] != candidate_id
    ]
    values = [
        z_by_candidate[row]
        for row in same
        if row in z_by_candidate and np.isfinite(z_by_candidate[row]).all()
    ]
    if not values:
        broader = [
            row["candidate_id"]
            for row in anchor_rows
            if row["class"] == "negative"
            and row["block_id"] == anchor["block_id"]
            and row["candidate_id"] != candidate_id
        ]
        values = [
            z_by_candidate[row]
            for row in broader
            if row in z_by_candidate and np.isfinite(z_by_candidate[row]).all()
        ]
    return normalize_mean(values)


def positive_prototype(trajectory: dict[int, tuple[str, np.ndarray]]) -> np.ndarray | None:
    values = [
        value
        for offset, (window_class, value) in trajectory.items()
        if window_class in OLD_SPEAKER_CLASSES
        and PROTOTYPE_OFFSET_RANGE_MS[0] <= offset <= PROTOTYPE_OFFSET_RANGE_MS[1]
    ]
    return normalize_mean(values)


def score_anchor(
    anchor_row: dict[str, Any],
    z_by_candidate: dict[str, np.ndarray],
    trajectory: dict[int, tuple[str, np.ndarray]] | None,
    negative_prototype: np.ndarray | None,
) -> tuple[float | None, float | None]:
    vector = z_by_candidate.get(anchor_row["candidate_id"])
    if vector is None or not np.isfinite(vector).all():
        return None, None
    if anchor_row["class"] == "positive":
        if not trajectory:
            return None, None
        prototype = positive_prototype(trajectory)
        adjacent = None
        sorted_offsets = sorted(trajectory)
        for previous, current in zip(sorted_offsets, sorted_offsets[1:], strict=False):
            if previous == -100 and current == 0:
                adjacent = cosine_distance(trajectory[previous][1], trajectory[current][1])
                break
        if prototype is None:
            return None, adjacent
        return cosine_distance(prototype, vector), adjacent
    if negative_prototype is None:
        return None, None
    return cosine_distance(negative_prototype, vector), None


def analyze_anchor_scores(
    anchor_rows: Sequence[dict[str, Any]],
    scores: dict[str, float],
    adjacent: dict[str, float] | None = None,
) -> dict[str, Any]:
    adjacent = adjacent or {}
    positive = {
        row["candidate_id"]: scores[row["candidate_id"]]
        for row in anchor_rows
        if row["class"] == "positive" and row["candidate_id"] in scores
    }
    negative = {
        row["candidate_id"]: scores[row["candidate_id"]]
        for row in anchor_rows
        if row["class"] == "negative" and row["candidate_id"] in scores
    }
    block_positive: dict[str, list[float]] = defaultdict(list)
    block_negative: dict[str, list[float]] = defaultdict(list)
    for row in anchor_rows:
        if row["class"] == "positive" and row["candidate_id"] in positive:
            block_positive[row["block_id"]].append(positive[row["candidate_id"]])
        if row["class"] == "negative" and row["candidate_id"] in negative:
            block_negative[row["block_id"]].append(negative[row["candidate_id"]])
    block_metrics: dict[str, dict[str, Any]] = {}
    for block_id in sorted(set(block_positive) & set(block_negative)):
        p = block_positive[block_id]
        n = block_negative[block_id]
        block_metrics[block_id] = {
            "roc_auc": roc_auc(p, n),
            "eer": eer(p, n),
            "overlap_coefficient": overlap_coefficient(p, n),
            "positive_count": len(p),
            "negative_count": len(n),
        }
    language_metrics: dict[str, dict[str, float]] = {}
    for language in sorted({row["language"] for row in anchor_rows}):
        p = [
            positive[row["candidate_id"]]
            for row in anchor_rows
            if row["class"] == "positive"
            and row["language"] == language
            and row["candidate_id"] in positive
        ]
        n = [
            negative[row["candidate_id"]]
            for row in anchor_rows
            if row["class"] == "negative"
            and row["language"] == language
            and row["candidate_id"] in negative
        ]
        if p and n:
            language_metrics[language] = {"roc_auc": roc_auc(p, n), "eer": eer(p, n)}
    kind_metrics: dict[str, dict[str, float]] = {}
    for kind in sorted({row["kind"] for row in anchor_rows}):
        p = [
            positive[row["candidate_id"]]
            for row in anchor_rows
            if row["class"] == "positive"
            and row["kind"] == kind
            and row["candidate_id"] in positive
        ]
        n = [
            negative[row["candidate_id"]]
            for row in anchor_rows
            if row["class"] == "negative"
            and row["kind"] == kind
            and row["candidate_id"] in negative
        ]
        if p:
            kind_metrics[kind] = {
                "mean_positive_distance": float(np.mean(p)),
                "mean_negative_distance": float(np.mean(n)) if n else float("nan"),
            }
    paired_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in anchor_rows:
        if row.get("pair_id"):
            paired_by_id[row["pair_id"]].append(row)
    pair_deltas: dict[str, float] = {}
    for pair_id, rows in paired_by_id.items():
        positive_row = next((row for row in rows if row["class"] == "positive"), None)
        negative_row = next((row for row in rows if row["class"] == "negative"), None)
        if positive_row is None or negative_row is None:
            continue
        if positive_row["candidate_id"] in scores and negative_row["candidate_id"] in scores:
            pair_deltas[pair_id] = float(
                scores[positive_row["candidate_id"]] - scores[negative_row["candidate_id"]]
            )
    all_positive = list(positive.values())
    all_negative = list(negative.values())
    macro = (
        float(np.mean([row["roc_auc"] for row in block_metrics.values()]))
        if block_metrics
        else float("nan")
    )
    worst = (
        float(min(row["roc_auc"] for row in block_metrics.values()))
        if block_metrics
        else float("nan")
    )
    return {
        "evaluable_positive_count": len(all_positive),
        "evaluable_negative_count": len(all_negative),
        "macro_roc_auc": macro,
        "worst_block_roc_auc": worst,
        "eer": eer(all_positive, all_negative),
        "overlap_coefficient": overlap_coefficient(all_positive, all_negative),
        "positive_distance_quantiles": quantiles(all_positive),
        "negative_distance_quantiles": quantiles(all_negative),
        "positive_adjacent_distance_quantiles": quantiles(
            [adjacent[candidate] for candidate in positive if candidate in adjacent]
        ),
        "block_metrics": block_metrics,
        "language_metrics": language_metrics,
        "kind_metrics": kind_metrics,
        "pair_metrics": {
            "matched_pair_count": len(pair_deltas),
            "paired_auc": (
                float((np.asarray(list(pair_deltas.values())) > 0).mean())
                if pair_deltas
                else float("nan")
            ),
            "mean_delta": (
                float(np.mean(list(pair_deltas.values()))) if pair_deltas else float("nan")
            ),
            "median_delta": (
                float(np.median(list(pair_deltas.values()))) if pair_deltas else float("nan")
            ),
            "signed_rank_statistic": wilcoxon_signed_rank_statistic(pair_deltas.values()),
            "paired_auc_bootstrap": bootstrap_paired_ci(pair_deltas),
        },
        "macro_roc_auc_bootstrap": bootstrap_block_ci(block_positive, block_negative),
    }


def trajectory_metrics(
    trajectory_by_candidate: dict[str, dict[int, tuple[str, np.ndarray]]],
) -> dict[str, Any]:
    candidate_summaries: list[dict[str, Any]] = []
    by_offset: dict[int, list[float]] = defaultdict(list)
    for candidate_id in sorted(trajectory_by_candidate):
        trajectory = trajectory_by_candidate[candidate_id]
        prototype = positive_prototype(trajectory)
        if prototype is None:
            continue
        distances: dict[int, float] = {}
        for offset, (window_class, vector) in trajectory.items():
            distances[offset] = cosine_distance(prototype, vector)
        old_offsets = [
            offset
            for offset in sorted(distances)
            if offset <= -100
            and trajectory[offset][0] in OLD_SPEAKER_CLASSES
        ]
        if not old_offsets:
            continue
        baseline = float(np.median([distances[offset] for offset in old_offsets]))
        post_offsets = [offset for offset in sorted(distances) if offset >= 0]
        peak_offset = max(post_offsets, key=lambda offset: distances[offset], default=None)
        peak = distances[peak_offset] if peak_offset is not None else float("nan")
        onset = None
        if peak_offset is not None:
            for offset in sorted(distances):
                if offset >= -100 and distances[offset] >= baseline + 0.5 * (peak - baseline):
                    onset = offset
                    break
        recovery = None
        if peak_offset is not None:
            for offset in sorted(distances):
                if offset > peak_offset and distances[offset] <= baseline + 0.1 * (peak - baseline):
                    recovery = offset
                    break
        for offset, value in distances.items():
            by_offset[offset].append(value)
        candidate_summaries.append(
            {
                "candidate_id": candidate_id,
                "baseline_distance": baseline,
                "peak_distance": peak,
                "peak_offset_ms": peak_offset,
                "onset_offset_ms": onset,
                "recovery_offset_ms": recovery,
                "min_distance": float(min(distances.values())),
                "max_distance": float(max(distances.values())),
            }
        )
    onset_values = [row["onset_offset_ms"] for row in candidate_summaries if row["onset_offset_ms"] is not None]
    recovery_values = [
        row["recovery_offset_ms"] for row in candidate_summaries if row["recovery_offset_ms"] is not None
    ]
    return {
        "candidate_count": len(candidate_summaries),
        "mean_curve_by_offset_ms": {
            str(offset): float(np.mean(values)) for offset, values in sorted(by_offset.items())
        },
        "mean_onset_offset_ms": float(np.mean(onset_values)) if onset_values else None,
        "mean_recovery_offset_ms": float(np.mean(recovery_values)) if recovery_values else None,
        "median_peak_offset_ms": (
            float(np.median([row["peak_offset_ms"] for row in candidate_summaries if row["peak_offset_ms"] is not None]))
            if any(row["peak_offset_ms"] is not None for row in candidate_summaries)
            else None
        ),
        "candidate_summaries": candidate_summaries,
    }


def rank_layers(
    layer_metrics: dict[str, dict[str, Any]],
    registry_order: Sequence[str],
    context_ms: int,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for index, layer_id in enumerate(registry_order):
        metrics = layer_metrics.get(layer_id)
        if metrics is None or not metrics.get("eligible"):
            continue
        macro = float(metrics["macro_roc_auc"])
        if not np.isfinite(macro):
            continue
        ranked.append(
            {
                "layer_id": layer_id,
                "context_ms": context_ms,
                "registry_order_index": index,
                "macro_roc_auc": macro,
                "eer": float(metrics["eer"]),
                "worst_block_roc_auc": float(metrics["worst_block_roc_auc"]),
                "evaluable_positive_count": metrics["evaluable_positive_count"],
                "evaluable_negative_count": metrics["evaluable_negative_count"],
            }
        )
    ranked.sort(
        key=lambda row: (
            -row["macro_roc_auc"],
            row["eer"],
            -row["worst_block_roc_auc"],
            row["registry_order_index"],
        )
    )
    return ranked


def build_promotion_ledger(
    probe_results: Sequence[dict[str, Any]],
    protocol_screen: dict[str, Any],
) -> dict[str, Any]:
    promotion = protocol_screen["r3"]["promotion"]
    context_rankings: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for context_ms in (PRIMARY_CONTEXT_MS, FALLBACK_CONTEXT_MS):
        context_rankings[context_ms] = {}
        for result in probe_results:
            model_id = result["model_id"]
            metrics_by_layer = {
                row["layer_id"]: row
                for row in result["layer_metrics"]
                if row["context_ms"] == context_ms
            }
            context_rankings[context_ms][model_id] = rank_layers(
                metrics_by_layer,
                REGISTRY_LAYER_ORDER[model_id],
                context_ms,
            )
    best_300 = {
        model_id: ranking[0] if ranking else None
        for model_id, ranking in context_rankings[PRIMARY_CONTEXT_MS].items()
    }
    best_500 = {
        model_id: ranking[0] if ranking else None
        for model_id, ranking in context_rankings[FALLBACK_CONTEXT_MS].items()
    }
    evaluable_300 = [row["macro_roc_auc"] for row in best_300.values() if row is not None]
    evaluable_500 = [row["macro_roc_auc"] for row in best_500.values() if row is not None]
    use_500 = bool(evaluable_300) and bool(evaluable_500)
    if use_500:
        use_500 = (
            all(
                value < promotion["fallback_requires_all_evaluable_300ms_auc_below"]
                for value in evaluable_300
            )
            and float(np.mean(evaluable_500)) - float(np.mean(evaluable_300))
            >= promotion["fallback_requires_mean_auc_improvement_at_least"]
        )
    common_context_ms = FALLBACK_CONTEXT_MS if use_500 else PRIMARY_CONTEXT_MS
    promoted: dict[str, Any] = {}
    for model_id in [result["model_id"] for result in probe_results]:
        ranking = context_rankings[common_context_ms].get(model_id) or []
        promoted_row = ranking[0] if ranking else None
        if promoted_row is None:
            promoted[model_id] = {
                "status": promotion["no_eligible_layer_status"],
                "reason": "no layer with finite vectors for every shared anchor",
            }
        else:
            promoted[model_id] = {"status": "promoted", **promoted_row}
    return {
        "common_context_ms": common_context_ms,
        "fallback_used": use_500,
        "rule": "lexicographic macro_roc_auc_desc,eer_asc,worst_block_roc_auc_desc,registry_order_asc",
        "best_300ms_by_encoder": best_300,
        "best_500ms_by_encoder": best_500,
        "promoted_by_encoder": promoted,
    }


def _coordinate_rows(cache_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coordinates_root = cache_root / "data" / "r2" / "legacy_common_gt" / "coordinates"
    primary: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    for path in sorted(coordinates_root.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            role = row.get("coordinate_role")
            if role == "r3_primary":
                primary.append(row)
            elif role == "r3_trajectory":
                trajectory.append(row)
    return primary, trajectory


def _waveform_paths(cache_root: Path) -> dict[str, Path]:
    inventory_path = cache_root / "data" / "r2" / "legacy_common_gt" / "waveform_inventory.jsonl"
    rows = [
        json.loads(line)
        for line in inventory_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    receipt = load_json(
        cache_root / "manifests" / "r2" / "legacy_common_gt" / "validation_receipt.json"
    )
    corpus_root = Path(str(receipt["corpus_root"])).resolve()
    result: dict[str, Path] = {}
    for row in rows:
        relative = Path(str(row["artifact_relative_path"]).replace("\\", "/"))
        result[str(row["waveform_id"])] = corpus_root / relative
    return result


def _load_anchor_index(cache_root: Path) -> list[dict[str, Any]]:
    index_path = cache_root / ANCHOR_INDEX_RELATIVE_PATH
    manifest_path = cache_root / ANCHOR_INDEX_MANIFEST_RELATIVE_PATH
    if not index_path.is_file() or not manifest_path.is_file():
        raise R3ProbeError("frozen R3 anchor index is missing; run r3_execute prepare")
    manifest = load_json(manifest_path)
    if not self_sha256_valid(manifest):
        raise R3ProbeError("R3 anchor index manifest self identity is invalid")
    if sha256_file(index_path) != manifest.get("sha256"):
        raise R3ProbeError("R3 anchor index byte identity differs from its manifest")
    rows = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    positive = [row for row in rows if row["class"] == "positive"]
    negative = [row for row in rows if row["class"] == "negative"]
    if len(positive) != 450 or len(negative) != 360:
        raise R3ProbeError("R3 anchor index counts differ from the frozen ledger")
    return rows


def _probe_provenance(model_id: str, requested_argv: tuple[str, ...]) -> dict[str, Any]:
    return {
        "authority": AUTHORITY,
        "execution_identity": {
            "run_id": uuid4().hex,
            "process_id": os.getpid(),
            "started_at_utc": datetime.now(UTC).isoformat(),
        },
        "run_provenance": run_provenance(
            REPOSITORY_ROOT,
            requested_argv,
            deterministic_seed=0,
            deterministic_kernels=True,
        ),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "model_id": model_id,
    }


def _make_extractor(model_id: str, cache_root: Path, *, threads: int = 8):
    acquisition = load_json(cache_root / "manifests" / "r1_model_acquisition.json")
    record = next(
        (row for row in acquisition["models"] if row["model_id"] == model_id), None
    )
    if record is None:
        raise R3ProbeError(f"model was not acquired: {model_id}")
    from experiments.speaker_representation_scd.extraction.eres_prepooling import (
        ERes2NetV2PrepoolExtractor,
    )
    from experiments.speaker_representation_scd.extraction.ssl import SSLExtractor

    if model_id == ERES_MODEL_ID:
        return ERes2NetV2PrepoolExtractor(
            Path(record["checkpoint_root"]),
            Path(record["source_root"]),
            EXPERIMENT_ROOT / "models" / "source_registry.json",
            threads=threads,
        )
    return SSLExtractor(model_id, Path(record["root"]), threads=threads)


def _extract_context(
    extractor,
    context_ms: int,
    by_waveform: dict[str, dict[str, list[dict[str, Any]]]],
    waveform_paths: dict[str, Path],
    layer_ids: Sequence[str],
    pooled_path: Path,
    index_path: Path,
    z_primary: dict[str, dict[str, np.ndarray]],
    trajectory_by_layer: dict[str, dict[str, dict[int, tuple[str, np.ndarray]]]],
) -> dict[str, Any]:
    import soundfile as sf

    total_windows = sum(
        len(
            {
                (int(row["window_start_sample"]), int(row["window_end_sample"]))
                for row in waveform_rows.get(str(context_ms)) or []
            }
        )
        for waveform_rows in by_waveform.values()
    )
    pooled: np.ndarray | None = None
    pooled_rows = 0
    batches = 0
    with index_path.open("w", encoding="utf-8") as index_handle:
        for waveform_id in sorted(by_waveform):
            rows = by_waveform[waveform_id].get(str(context_ms)) or []
            if not rows:
                continue
            path = waveform_paths.get(waveform_id)
            if path is None or not path.is_file():
                raise R3ProbeError(f"waveform path missing: {waveform_id}")
            audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
            if sample_rate != 16000 or audio.shape[1] != 1:
                raise R3ProbeError(f"waveform geometry differs: {waveform_id}")
            waveform = np.ascontiguousarray(audio[:, 0], dtype=np.float32)
            windows_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                windows_by_key[
                    (int(row["window_start_sample"]), int(row["window_end_sample"]))
                ].append(row)
            keys = sorted(windows_by_key)
            for batch_start in range(0, len(keys), BATCH_SIZE):
                batch_keys = keys[batch_start : batch_start + BATCH_SIZE]
                windows = [
                    np.ascontiguousarray(waveform[start:end], dtype=np.float32)
                    for start, end in batch_keys
                ]
                observed = [end for _, end in batch_keys]
                batch = extractor.extract(windows, observed, layer_ids=layer_ids)
                if pooled is None:
                    dimension = int(batch.layers[layer_ids[0]].shape[2])
                    pooled = np.lib.format.open_memmap(
                        pooled_path,
                        mode="w+",
                        dtype=np.float32,
                        shape=(total_windows, len(layer_ids), dimension),
                    )
                for position, (start, end) in enumerate(batch_keys):
                    for layer_index, layer_id in enumerate(layer_ids):
                        values = batch.layers[layer_id]
                        valid = batch.valid_lengths[layer_id]
                        pooled_vector = _mean_pool_l2(values[position], int(valid[position]))
                        pooled[pooled_rows + position, layer_index] = pooled_vector
                        for row in windows_by_key[(start, end)]:
                            candidate_id = str(row["candidate_id"])
                            if row["coordinate_role"] == "r3_primary":
                                z_primary[layer_id][candidate_id] = pooled_vector
                            elif row.get("trajectory_offset_ms") is not None:
                                trajectory_by_layer[layer_id].setdefault(candidate_id, {})[
                                    int(row["trajectory_offset_ms"])
                                ] = (str(row.get("window_class")), pooled_vector)
                    index_handle.write(
                        json.dumps(
                            {
                                "row_index": pooled_rows + position,
                                "window_start_sample": start,
                                "window_end_sample": end,
                                "waveform_id": waveform_id,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                pooled_rows += len(batch_keys)
                batches += 1
    if pooled is not None:
        pooled.flush()
        del pooled
    if pooled_rows != total_windows:
        raise R3ProbeError(f"pooled window row count differs: {pooled_rows} != {total_windows}")
    return {"context_ms": context_ms, "window_rows": pooled_rows, "batches": batches}


def _mean_pool_l2(values: np.ndarray, valid_length: int) -> np.ndarray:
    if valid_length <= 0 or valid_length > values.shape[0]:
        raise R3ProbeError("valid length is outside the feature tensor")
    pooled = values[:valid_length].mean(axis=0, dtype=np.float64)
    norm = np.linalg.norm(pooled)
    if not np.isfinite(norm) or norm <= 0:
        return np.full(values.shape[1], np.nan, dtype=np.float32)
    return (pooled / norm).astype(np.float32)


def _pooled_manifest(
    model_id: str,
    context_ms: int,
    layer_ids: Sequence[str],
    pooled_path: Path,
    index_path: Path,
    vector_meta: dict[str, Any],
) -> dict[str, Any]:
    return with_self_sha256(
        {
            "schema_version": 1,
            "artifact_role": "r3_pooled_shard_manifest",
            "experiment_id": "speaker_representation_scd_v1",
            "authority": AUTHORITY,
            "scope": "legacy-common-gt-v1",
            "model_id": model_id,
            "context_ms": context_ms,
            "layer_ids": list(layer_ids),
            "window_rows": vector_meta["window_rows"],
            "batches": vector_meta["batches"],
            "vectors_relative_path": pooled_path.name,
            "vectors_sha256": sha256_file(pooled_path),
            "index_relative_path": index_path.name,
            "index_sha256": sha256_file(index_path),
        }
    )


def _analyze_context(
    layer_ids: Sequence[str],
    context_ms: int,
    anchor_rows: Sequence[dict[str, Any]],
    z_primary: dict[str, dict[str, np.ndarray]],
    trajectory_by_layer: dict[str, dict[str, dict[int, tuple[str, np.ndarray]]]],
    primary_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    eligible_anchor_ids = {
        str(row["candidate_id"]) for row in primary_rows if int(row["context_ms"]) == context_ms
    }
    results: list[dict[str, Any]] = []
    for layer_id in layer_ids:
        z_layer = z_primary[layer_id]
        trajectory_layer = trajectory_by_layer[layer_id]
        missing = sorted(
            candidate_id
            for candidate_id in eligible_anchor_ids
            if candidate_id not in z_layer or not np.isfinite(z_layer[candidate_id]).all()
        )
        scores: dict[str, float] = {}
        adjacent: dict[str, float] = {}
        for row in anchor_rows:
            candidate_id = row["candidate_id"]
            if candidate_id not in eligible_anchor_ids:
                continue
            trajectory = trajectory_layer.get(candidate_id)
            negative_prototype = (
                None
                if row["class"] == "positive"
                else build_negative_prototype(anchor_rows, candidate_id, z_layer)
            )
            score, adjacency = score_anchor(row, z_layer, trajectory, negative_prototype)
            if score is not None and np.isfinite(score):
                scores[candidate_id] = float(score)
            if adjacency is not None and np.isfinite(adjacency):
                adjacent[candidate_id] = float(adjacency)
        metrics = analyze_anchor_scores(anchor_rows, scores, adjacent)
        metrics["layer_id"] = layer_id
        metrics["eligible"] = not missing
        metrics["missing_anchor_count"] = len(missing)
        metrics["missing_anchor_ids"] = missing
        metrics["trajectory"] = trajectory_metrics(trajectory_layer)
        metrics["score_table"] = [
            {
                "candidate_id": row["candidate_id"],
                "class": row["class"],
                "kind": row["kind"],
                "block_id": row["block_id"],
                "language": row["language"],
                "pair_id": row.get("pair_id"),
                "d_proto": scores.get(row["candidate_id"]),
                "d_adj": adjacent.get(row["candidate_id"]),
            }
            for row in anchor_rows
            if row["candidate_id"] in scores or row["candidate_id"] in adjacent
        ]
        results.append(metrics)
    return results


def run_probe(model_id: str, cache_root: Path, requested_argv: tuple[str, ...]) -> Path:
    if model_id not in REGISTRY_LAYER_ORDER:
        raise R3ProbeError(f"unknown encoder: {model_id}")
    result_path = cache_root / PROBE_RESULT_RELATIVE_PATH.as_posix().format(model_id=model_id)
    validate_worker_execution(cache_root, expected_receipt=result_path)
    if result_path.exists():
        if action_receipt_is_authoritative(cache_root, result_path, f"r3-probe:{model_id}"):
            raise R3ProbeError(f"refusing to rerun a probe with completed evidence: {result_path}")
    gate = validate_r3_gate(cache_root=cache_root.resolve(), scan_processes=False)
    if not gate.valid:
        raise R3ProbeError("; ".join(gate.errors))
    if gate.allowed_actions.get("r3_probe") is not True:
        raise R3ProbeError("R3 probe is not authorized")
    anchor_rows = _load_anchor_index(cache_root)
    primary_rows, trajectory_rows = _coordinate_rows(cache_root)
    waveform_paths = _waveform_paths(cache_root)
    by_waveform: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in primary_rows + trajectory_rows:
        by_waveform[str(row["waveform_id"])][str(row["context_ms"])].append(row)
    extractor = _make_extractor(model_id, cache_root)
    layer_ids = REGISTRY_LAYER_ORDER[model_id]
    pooled_dir = cache_root / POOLED_RELATIVE_DIR / model_id
    pooled_dir.mkdir(parents=True, exist_ok=True)
    layer_metrics_by_context: dict[int, list[dict[str, Any]]] = {}
    for context_ms in CONTEXTS_MS:
        z_primary: dict[str, dict[str, np.ndarray]] = {layer_id: {} for layer_id in layer_ids}
        trajectory_by_layer: dict[
            str, dict[str, dict[int, tuple[str, np.ndarray]]]
        ] = {layer_id: {} for layer_id in layer_ids}
        pooled_path = pooled_dir / f"vectors_{context_ms}.npy"
        index_path = pooled_dir / f"index_{context_ms}.jsonl"
        vector_meta = _extract_context(
            extractor,
            context_ms,
            by_waveform,
            waveform_paths,
            layer_ids,
            pooled_path,
            index_path,
            z_primary,
            trajectory_by_layer,
        )
        layer_metrics_by_context[context_ms] = _analyze_context(
            layer_ids,
            context_ms,
            anchor_rows,
            z_primary,
            trajectory_by_layer,
            primary_rows,
        )
        manifest = _pooled_manifest(
            model_id, context_ms, layer_ids, pooled_path, index_path, vector_meta
        )
        (pooled_dir / f"manifest_{context_ms}.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    result = {
        "schema_version": 1,
        "artifact_role": "r3_probe_result",
        "experiment_id": "speaker_representation_scd_v1",
        "authority": AUTHORITY,
        "scope": "legacy-common-gt-v1",
        "model_id": model_id,
        "contexts_ms": list(CONTEXTS_MS),
        "layer_ids": list(layer_ids),
        "layer_metrics": [
            {**row, "context_ms": context_ms}
            for context_ms in CONTEXTS_MS
            for row in layer_metrics_by_context[context_ms]
        ],
        "pooled_cache": {
            "relative_to_cache_root": POOLED_RELATIVE_DIR.as_posix() + f"/{model_id}",
            "manifest_sha256s": {
                str(context_ms): sha256_file(pooled_dir / f"manifest_{context_ms}.json")
                for context_ms in CONTEXTS_MS
            },
        },
        "supervision_binding": {
            "execution_id": os.environ.get("SRSCD_EXECUTION_LEASE_TOKEN"),
            "expected_receipt_relative_path": PROBE_RESULT_RELATIVE_PATH.as_posix().format(
                model_id=model_id
            ),
            "authority": "requires_completed_usage_attestation",
        },
        "provenance": _probe_provenance(model_id, requested_argv),
    }
    document = with_self_sha256(_json_safe(result))
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result_path


def run_promote(cache_root: Path, requested_argv: tuple[str, ...]) -> Path:
    result_path = cache_root / PROMOTION_LEDGER_RELATIVE_PATH
    validate_worker_execution(cache_root, expected_receipt=result_path)
    if result_path.exists():
        if action_receipt_is_authoritative(cache_root, result_path, "r3-promote"):
            raise R3ProbeError(f"refusing to rerun promotion with completed evidence: {result_path}")
    gate = validate_r3_gate(cache_root=cache_root.resolve(), scan_processes=False)
    if not gate.valid:
        raise R3ProbeError("; ".join(gate.errors))
    if gate.allowed_actions.get("r3_promote") is not True:
        raise R3ProbeError("R3 promotion is not authorized")
    probe_results: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        path = cache_root / PROBE_RESULT_RELATIVE_PATH.as_posix().format(model_id=model_id)
        if not path.is_file():
            raise R3ProbeError(f"R3 probe result missing: {model_id}")
        if not action_receipt_is_authoritative(cache_root, path, f"r3-probe:{model_id}"):
            raise R3ProbeError(f"R3 probe result lacks completed attestation: {model_id}")
        document = load_json(path)
        if not self_sha256_valid(document):
            raise R3ProbeError(f"R3 probe result self identity invalid: {model_id}")
        if document.get("artifact_role") != "r3_probe_result" or document.get("model_id") != model_id:
            raise R3ProbeError(f"R3 probe result identity differs: {model_id}")
        probe_results.append(document)
    protocol_screen = load_json(
        EXPERIMENT_ROOT / "configs/protocol/reduced_pretraining_screen.json"
    )
    if not self_sha256_valid(protocol_screen):
        raise R3ProbeError("protocol screen self identity is invalid")
    ledger = build_promotion_ledger(probe_results, protocol_screen)
    document = with_self_sha256(
        {
            "schema_version": 1,
            "artifact_role": "r3_promotion_ledger",
            "experiment_id": "speaker_representation_scd_v1",
            "authority": AUTHORITY,
            "scope": "legacy-common-gt-v1",
            "promotion": ledger,
            "probe_results": {
                row["model_id"]: {
                    "relative_to_cache_root": PROBE_RESULT_RELATIVE_PATH.as_posix().format(
                        model_id=row["model_id"]
                    ),
                    "sha256": sha256_file(
                        cache_root
                        / PROBE_RESULT_RELATIVE_PATH.as_posix().format(model_id=row["model_id"])
                    ),
                    "self_sha256": row["self_sha256"],
                }
                for row in probe_results
            },
            "supervision_binding": {
                "execution_id": os.environ.get("SRSCD_EXECUTION_LEASE_TOKEN"),
                "expected_receipt_relative_path": PROMOTION_LEDGER_RELATIVE_PATH.as_posix(),
                "authority": "requires_completed_usage_attestation",
            },
            "provenance": {
                "authority": AUTHORITY,
                "execution_identity": {
                    "run_id": uuid4().hex,
                    "process_id": os.getpid(),
                    "started_at_utc": datetime.now(UTC).isoformat(),
                },
                "run_provenance": run_provenance(
                    REPOSITORY_ROOT,
                    requested_argv,
                    deterministic_seed=0,
                    deterministic_kernels=False,
                ),
                "code_sha256": sha256_file(Path(__file__).resolve()),
            },
        }
    )
    document = with_self_sha256(_json_safe(document))
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("probe", "promote"), required=True)
    parser.add_argument("--encoder", choices=sorted(REGISTRY_LAYER_ORDER))
    args = parser.parse_args(argv)
    cache_root = Path(os.environ["SRSCD_CACHE_ROOT"]).resolve()
    requested = tuple(
        [sys.executable, "-m", __package__ + ".r3_probe", *(argv or sys.argv[1:])]
    )
    if args.worker == "probe":
        if args.encoder is None:
            raise SystemExit("probe requires --encoder")
        print(run_probe(args.encoder, cache_root, requested))
    else:
        print(run_promote(cache_root, requested))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExecutionGuardError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
