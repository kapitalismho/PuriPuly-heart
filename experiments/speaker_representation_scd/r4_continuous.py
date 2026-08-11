from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence
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
from experiments.speaker_representation_scd.r2l_gate import AUTHORITY
from experiments.speaker_representation_scd.r3_probe import (
    BATCH_SIZE,
    ERES_MODEL_ID,
    REGISTRY_LAYER_ORDER,
    _json_safe,
    cosine_distance,
)
from experiments.speaker_representation_scd.r4_gate import validate_r4_gate
from experiments.speaker_representation_scd.run_provenance import run_provenance

PRIMARY_CONTEXT_MS = 300
PRIMARY_HOP_SAMPLES = 1600
SENSITIVITY_HOP_SAMPLES = 800
STABILITY_GATE = 0.25
PROTOTYPE_K = 3
MIN_EVENT_SEPARATION_HOPS = 2
MATCH_TOLERANCE_MS = 500
FALSE_EVENT_BUDGET_PER_HOUR = 1.0
THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
SCORE_TYPES = ("adjacent", "prototype")
FAMILIES = ("one_hop", "two_hop", "three_hop", "hysteresis")
MODEL_IDS = (
    "mhubert-147",
    "wavlm-base-plus",
    "unispeech-sat-base-plus",
    "eres2netv2-standard-prepool",
)

CONTINUOUS_RESULT_RELATIVE_PATH = Path("manifests/r4/legacy_common_gt/continuous_{model_id}.json")
SENSITIVITY_RESULT_RELATIVE_PATH = Path("manifests/r4/legacy_common_gt/sensitivity_{model_id}.json")
SELECTION_LEDGER_RELATIVE_PATH = Path("manifests/r4/legacy_common_gt/candidate_selection_ledger.json")
POOLED_RELATIVE_DIR = Path("data/r4/legacy_common_gt/pooled")


class R4Error(RuntimeError):
    pass


def _mean_pool_l2_batch(values: np.ndarray, valid_lengths: np.ndarray) -> np.ndarray:
    if values.ndim != 3:
        raise R4Error("extractor values must have shape batch,time,dimension")
    pooled = np.empty((values.shape[0], values.shape[2]), dtype=np.float32)
    for index, length in enumerate(valid_lengths.tolist()):
        if length <= 0 or length > values.shape[1]:
            raise R4Error("valid length is outside the feature tensor")
        pooled[index] = values[index, :length].mean(axis=0, dtype=np.float64)
    norms = np.linalg.norm(pooled, axis=1)
    result = np.full(pooled.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(norms) & (norms > 0)
    result[finite] = pooled[finite] / norms[finite, None]
    return result


def adjacent_scores(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim != 2:
        raise R4Error("vectors must have shape time,dimension")
    if vectors.shape[0] < 2:
        return np.full(vectors.shape[0], np.nan, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(norms) & (norms > 0)
    if not valid.all():
        raise R4Error("non-finite vector in continuous stream")
    normalized = vectors / norms[:, None]
    products = (normalized[:-1] * normalized[1:]).sum(axis=1)
    distances = np.clip(1.0 - products, 0.0, 1.0)
    return np.concatenate([[np.nan], distances])


def prototype_scores(vectors: np.ndarray, stability_gate: float = STABILITY_GATE) -> np.ndarray:
    if vectors.ndim != 2:
        raise R4Error("vectors must have shape time,dimension")
    time_steps, dimension = vectors.shape
    distances = np.full(time_steps, np.nan, dtype=np.float64)
    if time_steps == 0:
        return distances
    prototypes: list[np.ndarray] = []
    prototype: np.ndarray | None = None
    for index in range(time_steps):
        vector = vectors[index]
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0:
            continue
        if prototype is None:
            prototype = vector
            distances[index] = 0.0
        else:
            distances[index] = cosine_distance(prototype, vector)
        if prototype is not None and distances[index] <= stability_gate:
            prototypes.append(vector)
            if len(prototypes) > PROTOTYPE_K:
                prototypes.pop(0)
            if len(prototypes) >= 2:
                mean = np.mean(np.stack(prototypes), axis=0)
                mean_norm = np.linalg.norm(mean)
                if np.isfinite(mean_norm) and mean_norm > 0:
                    prototype = (mean / mean_norm).astype(np.float32)
    return distances


def detect_events(
    distances: np.ndarray,
    family: str,
    threshold: float,
    *,
    hop_samples: int = PRIMARY_HOP_SAMPLES,
    minimum_separation_hops: int = MIN_EVENT_SEPARATION_HOPS,
) -> list[dict[str, int]]:
    events: list[dict[str, int]] = []
    time_steps = distances.shape[0]
    index = 0
    while index < time_steps:
        value = distances[index]
        if not np.isfinite(value) or value <= threshold:
            index += 1
            continue
        if family == "one_hop":
            emit = index
        elif family == "two_hop":
            if index + 1 >= time_steps or not (
                np.isfinite(distances[index + 1]) and distances[index + 1] > threshold
            ):
                index += 1
                continue
            emit = index + 1
        elif family == "three_hop":
            if index + 2 >= time_steps or not (
                np.isfinite(distances[index + 1])
                and distances[index + 1] > threshold
                and np.isfinite(distances[index + 2])
                and distances[index + 2] > threshold
            ):
                index += 1
                continue
            emit = index + 2
        elif family == "hysteresis":
            stay = 0.5 * threshold
            emit = None
            probe = index
            while probe < time_steps:
                probe_value = distances[probe]
                if not np.isfinite(probe_value):
                    break
                if probe_value <= stay:
                    break
                if probe_value > threshold and probe > index:
                    emit = probe
                    break
                probe += 1
            if emit is None:
                emit = index
        else:
            raise R4Error(f"unknown detector family: {family}")
        if events and emit - events[-1]["emit_hop"] < minimum_separation_hops:
            index = emit + 1
            continue
        events.append(
            {
                "onset_hop": int(index),
                "onset_sample": int(index * hop_samples) + int(PRIMARY_CONTEXT_MS * 16),
                "emit_hop": int(emit),
                "emit_sample": int(emit * hop_samples) + int(PRIMARY_CONTEXT_MS * 16),
                "availability_hop": int(emit),
            }
        )
        index = emit + 1
        while (
            index < time_steps
            and np.isfinite(distances[index])
            and distances[index] > threshold
        ):
            index += 1
    return events


def match_events(
    ground_truth_samples: Sequence[int],
    events: Sequence[dict[str, int]],
    tolerance_ms: int = MATCH_TOLERANCE_MS,
) -> list[dict[str, Any]]:
    tolerance_samples = tolerance_ms * 16
    matched: list[dict[str, Any]] = []
    unmatched_gt = list(ground_truth_samples)
    for event in sorted(events, key=lambda row: row["onset_sample"]):
        best_index = None
        best_error = None
        for index, gt in enumerate(unmatched_gt):
            error = abs(int(event["onset_sample"]) - int(gt))
            if error <= tolerance_samples and (best_error is None or error < best_error):
                best_error = error
                best_index = index
        if best_index is None:
            continue
        gt = unmatched_gt.pop(best_index)
        matched.append(
            {
                "onset_sample": int(event["onset_sample"]),
                "emit_sample": int(event["emit_sample"]),
                "availability_sample": int(event["emit_sample"]),
                "ground_truth_sample": int(gt),
                "localization_error_ms": (int(event["onset_sample"]) - int(gt)) / 16,
                "availability_latency_ms": (int(event["emit_sample"]) - int(gt)) / 16,
            }
        )
    return matched


def _event_metrics(
    matched: Sequence[dict[str, Any]],
    total_events: int,
    ground_truth_count: int,
    source_hours: float,
) -> dict[str, Any]:
    latencies = [row["availability_latency_ms"] for row in matched]
    errors = [row["localization_error_ms"] for row in matched]
    if matched:
        latency_array = np.asarray(latencies, dtype=np.float64)
        latency = {
            "median_ms": float(np.median(latency_array)),
            "p90_ms": float(np.quantile(latency_array, 0.90)),
            "p95_ms": float(np.quantile(latency_array, 0.95)),
        }
        localization = {
            "median_ms": float(np.median(errors)),
            "mean_abs_ms": float(np.mean(np.abs(errors))),
        }
    else:
        latency = {"median_ms": None, "p90_ms": None, "p95_ms": None}
        localization = {"median_ms": None, "mean_abs_ms": None}
    false_events = max(0, total_events - len(matched))

    def f1(precision_value: float, recall_value: float) -> float:
        if not np.isfinite(precision_value + recall_value) or precision_value + recall_value == 0:
            return float("nan")
        return 2 * precision_value * recall_value / (precision_value + recall_value)

    def within(window_ms: int) -> dict[str, float]:
        window_samples = window_ms * 16
        inside = sum(
            1 for row in matched if abs(row["localization_error_ms"]) * 16 <= window_samples
        )
        recall_value = inside / ground_truth_count if ground_truth_count else float("nan")
        precision_value = inside / total_events if total_events else float("nan")
        return {
            "recall": float(recall_value),
            "precision": float(precision_value),
            "f1": f1(precision_value, recall_value),
        }

    return {
        "matched_count": len(matched),
        "ground_truth_count": ground_truth_count,
        "total_confirmed_events": total_events,
        "recall_within_500ms": within(500)["recall"],
        "f1_at_250ms": within(250)["f1"],
        "boundary_f1": {
            "at_100ms": within(100),
            "at_250ms": within(250),
            "at_500ms": within(500),
        },
        "availability_latency_ms": latency,
        "signed_localization_error_ms": localization,
        "false_events_per_hour": false_events / source_hours if source_hours else float("nan"),
        "missed_change_rate": (
            1 - len(matched) / ground_truth_count if ground_truth_count else float("nan")
        ),
        "duplicate_event_rate": 0.0,
    }


def _canonical_config_id(score_type: str, family: str, threshold: float) -> str:
    return f"{score_type}|{family}|{threshold:.2f}"


def _select_operating_point(
    config_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    feasible = [
        row
        for row in config_rows
        if row["metrics"]["false_events_per_hour"] is not None
        and float(row["metrics"]["false_events_per_hour"]) <= FALSE_EVENT_BUDGET_PER_HOUR
    ]
    pool = feasible if feasible else sorted(
        config_rows,
        key=lambda row: (
            float(row["metrics"]["false_events_per_hour"]),
            row["config_id"],
        ),
    )
    primary = min(
        pool,
        key=lambda row: (
            -float(row["metrics"]["recall_within_500ms"]),
            float(row["metrics"]["false_events_per_hour"]),
            -float(row["metrics"]["f1_at_250ms"]),
            float(row["metrics"]["availability_latency_ms"]["median_ms"] or 1e18),
            row["config_id"],
        ),
    )
    return primary


def _rank_encoders(
    operating_points: dict[str, dict[str, Any]],
    r3_macro_auc: dict[str, float],
) -> list[dict[str, Any]]:
    ranked = []
    for model_id, point in operating_points.items():
        metrics = point["metrics"]
        ranked.append(
            {
                "model_id": model_id,
                "config_id": point["config_id"],
                "recall_within_500ms": float(metrics["recall_within_500ms"]),
                "false_events_per_hour": float(metrics["false_events_per_hour"]),
                "f1_at_250ms": float(metrics["f1_at_250ms"]),
                "median_availability_latency_ms": metrics["availability_latency_ms"]["median_ms"],
                "r3_macro_roc_auc": float(r3_macro_auc.get(model_id, float("nan"))),
            }
        )
    ranked.sort(
        key=lambda row: (
            -row["recall_within_500ms"],
            row["false_events_per_hour"],
            -row["f1_at_250ms"],
            row["median_availability_latency_ms"] or 1e18,
            -row["r3_macro_roc_auc"],
            row["model_id"],
        )
    )
    return ranked


def _load_panel_sources(cache_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    forecast = load_json(
        cache_root / "manifests/r2/legacy_common_gt/reduced_r3_r4_forecast.json"
    )
    bounded = forecast.get("bounded_variant") or {}
    included = set(bounded.get("included_session_ids") or [])
    if not included:
        raise R4Error("frozen bounded R4 panel is missing from the reduced forecast")
    ledger = load_json(
        cache_root / "manifests/r2/legacy_common_gt/coordinate_ledger.json"
    )
    sources = [
        row for row in ledger["r4"]["panel_sources"] if str(row["session_id"]) in included
    ]
    if len(sources) != len(included):
        raise R4Error("bounded R4 panel sources do not match the frozen ledger")
    return sources, bounded


def _promoted_layers(cache_root: Path) -> dict[str, dict[str, Any]]:
    ledger = load_json(cache_root / "manifests/r3/legacy_common_gt/promotion_ledger.json")
    promoted = ledger["promotion"]["promoted_by_encoder"]
    result: dict[str, dict[str, Any]] = {}
    for model_id in MODEL_IDS:
        row = promoted.get(model_id) or {}
        if row.get("status") != "promoted" or not row.get("layer_id"):
            raise R4Error(f"R4 requires a promoted configuration for {model_id}")
        result[model_id] = row
    return result


def _r3_macro_auc(cache_root: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    for model_id in MODEL_IDS:
        path = cache_root / f"manifests/r3/legacy_common_gt/probe_{model_id}.json"
        document = load_json(path)
        promoted = None
        for row in document["layer_metrics"]:
            if row["context_ms"] == 300 and row["layer_id"] in REGISTRY_LAYER_ORDER[model_id]:
                candidate = row
                if promoted is None or candidate["macro_roc_auc"] > promoted["macro_roc_auc"]:
                    promoted = candidate
        result[model_id] = float(promoted["macro_roc_auc"] if promoted else float("nan"))
    return result


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


def _anchor_coordinates(cache_root: Path) -> dict[str, list[dict[str, Any]]]:
    index_path = cache_root / "data/r3/legacy_common_gt/anchor_index.jsonl"
    rows = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["class"] == "positive":
            result[str(row["session_id"])].append(
                {"coordinate": int(row["coordinate"]), "candidate_id": row["candidate_id"]}
            )
    return dict(result)


def _session_waveform_map(cache_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted((cache_root / "data/r2/legacy_common_gt/coordinates").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            if row.get("coordinate_role") == "r4_continuous":
                result.setdefault(str(row["session_id"]), str(row["waveform_id"]))
    return result


def _extract_continuous_vectors(
    extractor,
    model_id: str,
    context_ms: int,
    hop_samples: int,
    sources: Sequence[dict[str, Any]],
    waveform_paths: dict[str, Path],
    layer_ids: Sequence[str],
    pooled_path: Path | None,
    index_path: Path | None,
    layer_kwarg: str,
    session_waveform: dict[str, str],
) -> dict[str, dict[str, Any]]:
    import soundfile as sf

    per_source: dict[str, dict[str, Any]] = {}
    for source in sources:
        session_id = str(source["session_id"])
        waveform_id = session_waveform.get(session_id)
        if waveform_id is None:
            raise R4Error(f"waveform mapping missing for session: {session_id}")
        path = waveform_paths.get(waveform_id)
        if path is None or not path.is_file():
            raise R4Error(f"waveform path missing: {waveform_id}")
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        if sample_rate != 16000 or audio.shape[1] != 1:
            raise R4Error(f"waveform geometry differs: {waveform_id}")
        waveform = np.ascontiguousarray(audio[:, 0], dtype=np.float32)
        eligible_start = int(source["eligible_start_sample"])
        eligible_end = int(source["eligible_end_sample"])
        window_samples = context_ms * 16
        frontiers = list(range(eligible_start + window_samples, eligible_end + 1, hop_samples))
        vectors: list[np.ndarray] = []
        observed = [frontier for frontier in frontiers if frontier <= len(waveform)]
        if not observed:
            continue
        windows = [
            np.ascontiguousarray(
                waveform[frontier - window_samples : frontier], dtype=np.float32
            )
            for frontier in observed
        ]
        for batch_start in range(0, len(windows), BATCH_SIZE):
            batch_windows = windows[batch_start : batch_start + BATCH_SIZE]
            batch_observed = observed[batch_start : batch_start + BATCH_SIZE]
            batch = extractor.extract(
                batch_windows, batch_observed, **{layer_kwarg: layer_ids}
            )
            layer_id = layer_ids[0]
            values = batch.layers[layer_id]
            valid = batch.valid_lengths[layer_id]
            pooled_values = _mean_pool_l2_batch(values, valid)
            vectors.append(pooled_values)
        if not vectors:
            continue
        stacked = np.concatenate(vectors, axis=0)
        per_source[session_id] = {
            "session_id": session_id,
            "waveform_id": waveform_id,
            "hop_samples": hop_samples,
            "context_ms": context_ms,
            "frontier_samples": observed,
            "vectors": stacked,
        }
    if pooled_path is not None and index_path is not None and per_source:
        total = sum(int(row["vectors"].shape[0]) for row in per_source.values())
        dimension = int(next(iter(per_source.values()))["vectors"].shape[1])
        pooled = np.lib.format.open_memmap(
            pooled_path, mode="w+", dtype=np.float32, shape=(total, dimension)
        )
        position = 0
        with index_path.open("w", encoding="utf-8") as index_handle:
            for session_id in sorted(per_source):
                row = per_source[session_id]
                count = int(row["vectors"].shape[0])
                pooled[position : position + count] = row["vectors"]
                index_handle.write(
                    json.dumps(
                        {
                            "row_start": position,
                            "row_count": count,
                            "session_id": session_id,
                            "frontier_samples": row["frontier_samples"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                position += count
        pooled.flush()
        del pooled
    return per_source


def _run_continuous(
    model_id: str,
    cache_root: Path,
    requested_argv: tuple[str, ...],
    *,
    sensitivity: bool,
) -> Path:
    if model_id not in MODEL_IDS:
        raise R4Error(f"unknown encoder: {model_id}")
    relative = (
        SENSITIVITY_RESULT_RELATIVE_PATH.as_posix().format(model_id=model_id)
        if sensitivity
        else CONTINUOUS_RESULT_RELATIVE_PATH.as_posix().format(model_id=model_id)
    )
    result_path = cache_root / relative
    validate_worker_execution(cache_root, expected_receipt=result_path)
    action = f"r4-{'sensitivity' if sensitivity else 'continuous'}:{model_id}"
    if result_path.exists():
        if action_receipt_is_authoritative(cache_root, result_path, action):
            raise R4Error(f"refusing to rerun an action with completed evidence: {result_path}")
    gate = validate_r4_gate(cache_root=cache_root.resolve(), scan_processes=False)
    if not gate.valid:
        raise R4Error("; ".join(gate.errors))
    authorization_key = "r4_sensitivity" if sensitivity else "r4_continuous"
    if gate.allowed_actions.get(authorization_key) is not True:
        raise R4Error(f"{authorization_key} is not authorized")
    promoted = _promoted_layers(cache_root)
    promoted_row = promoted[model_id]
    layer_id = str(promoted_row["layer_id"])
    context_ms = int(promoted_row["context_ms"])
    hop_samples = SENSITIVITY_HOP_SAMPLES if sensitivity else PRIMARY_HOP_SAMPLES
    sources, bounded = _load_panel_sources(cache_root)
    if sensitivity:
        sources = [
            row for row in sources if row.get("synthetic_manifest") is not None
        ]
    waveform_paths = _waveform_paths(cache_root)
    session_waveform = _session_waveform_map(cache_root)
    ground_truth = _anchor_coordinates(cache_root)
    extractor = _make_extractor(model_id, cache_root)
    pooled_dir = cache_root / POOLED_RELATIVE_DIR / model_id
    pooled_path = index_path = None
    if not sensitivity:
        pooled_dir.mkdir(parents=True, exist_ok=True)
        pooled_path = pooled_dir / f"vectors_{context_ms}.npy"
        index_path = pooled_dir / f"index_{context_ms}.jsonl"
    per_source = _extract_continuous_vectors(
        extractor,
        model_id,
        context_ms,
        hop_samples,
        sources,
        waveform_paths,
        (layer_id,),
        pooled_path,
        index_path,
        "tap_ids" if model_id == ERES_MODEL_ID else "layer_ids",
        session_waveform,
    )
    config_rows: list[dict[str, Any]] = []
    for score_type in SCORE_TYPES:
        for family in FAMILIES:
            for threshold in THRESHOLDS:
                config_rows.append(
                    _evaluate_config(
                        model_id,
                        score_type,
                        family,
                        threshold,
                        per_source,
                        ground_truth,
                        layer_id,
                        context_ms,
                        hop_samples,
                    )
                )
    operating_point = _select_operating_point(config_rows)
    config_id = str(operating_point["config_id"])
    panel_hours = (
        sum(
            (int(row["eligible_end_sample"]) - int(row["eligible_start_sample"])) / 16000 / 3600
            for row in sources
        )
        if not sensitivity
        else float(bounded.get("r4_source_hours", 0.0))
    )
    if sensitivity:
        panel_hours = sum(
            (int(row["eligible_end_sample"]) - int(row["eligible_start_sample"])) / 16000 / 3600
            for row in sources
        )
    document = with_self_sha256(
        _json_safe(
            {
                "schema_version": 1,
                "artifact_role": "r4_sensitivity_result" if sensitivity else "r4_continuous_result",
                "experiment_id": "speaker_representation_scd_v1",
                "authority": AUTHORITY,
                "scope": "legacy-common-gt-v1",
                "model_id": model_id,
                "promoted_layer_id": layer_id,
                "promoted_context_ms": context_ms,
                "hop_ms": int(hop_samples / 16),
                "panel": {
                    "source_count": len(sources),
                    "source_hours": round(panel_hours, 6),
                    "window_count": sum(
                        int(row["vectors"].shape[0]) for row in per_source.values()
                    ),
                },
                "operating_point": {
                    "config_id": config_id,
                    "score_type": operating_point["score_type"],
                    "family": operating_point["family"],
                    "threshold": operating_point["threshold"],
                    "metrics": operating_point["metrics"],
                },
                "config_grid": [
                    {
                        "config_id": row["config_id"],
                        "score_type": row["score_type"],
                        "family": row["family"],
                        "threshold": row["threshold"],
                        "metrics": row["metrics"],
                    }
                    for row in config_rows
                ],
                "pooled_cache": (
                    {
                        "relative_to_cache_root": POOLED_RELATIVE_DIR.as_posix() + f"/{model_id}",
                        "vectors_sha256": sha256_file(pooled_path),
                        "index_sha256": sha256_file(index_path),
                    }
                    if pooled_path is not None and index_path is not None
                    else None
                ),
                "supervision_binding": {
                    "execution_id": os.environ.get("SRSCD_EXECUTION_LEASE_TOKEN"),
                    "expected_receipt_relative_path": relative,
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
                        deterministic_kernels=True,
                    ),
                    "code_sha256": sha256_file(Path(__file__).resolve()),
                },
            }
        )
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result_path


def _evaluate_config(
    model_id: str,
    score_type: str,
    family: str,
    threshold: float,
    per_source: dict[str, dict[str, Any]],
    ground_truth: dict[str, list[dict[str, Any]]],
    layer_id: str,
    context_ms: int,
    hop_samples: int,
) -> dict[str, Any]:
    total_events = 0
    ground_truth_count = 0
    matched_rows: list[dict[str, Any]] = []
    for session_id, row in per_source.items():
        vectors = row["vectors"]
        if score_type == "adjacent":
            distances = adjacent_scores(vectors)
        else:
            distances = prototype_scores(vectors)
        events = detect_events(distances, family, threshold, hop_samples=hop_samples)
        gt_samples = [int(gt["coordinate"]) for gt in ground_truth.get(session_id, [])]
        total_events += len(events)
        ground_truth_count += len(gt_samples)
        matched_rows.extend(match_events(gt_samples, events))
    source_hours = sum(
        (int(row["vectors"].shape[0]) * hop_samples) / 16000 / 3600
        for row in per_source.values()
    )
    metrics = _event_metrics(matched_rows, total_events, ground_truth_count, source_hours)
    return {
        "config_id": _canonical_config_id(score_type, family, threshold),
        "score_type": score_type,
        "family": family,
        "threshold": float(threshold),
        "metrics": metrics,
    }


def _make_extractor(model_id: str, cache_root: Path, *, threads: int = 8):
    acquisition = load_json(cache_root / "manifests" / "r1_model_acquisition.json")
    record = next(
        (row for row in acquisition["models"] if row["model_id"] == model_id), None
    )
    if record is None:
        raise R4Error(f"model was not acquired: {model_id}")
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


def run_report(cache_root: Path, requested_argv: tuple[str, ...]) -> Path:
    result_path = cache_root / SELECTION_LEDGER_RELATIVE_PATH
    validate_worker_execution(cache_root, expected_receipt=result_path)
    if result_path.exists():
        if action_receipt_is_authoritative(cache_root, result_path, "r4-report"):
            raise R4Error(f"refusing to rerun selection with completed evidence: {result_path}")
    gate = validate_r4_gate(cache_root=cache_root.resolve(), scan_processes=False)
    if not gate.valid:
        raise R4Error("; ".join(gate.errors))
    if gate.allowed_actions.get("r4_report") is not True:
        raise R4Error("r4_report is not authorized")
    operating_points: dict[str, dict[str, Any]] = {}
    for model_id in MODEL_IDS:
        path = cache_root / CONTINUOUS_RESULT_RELATIVE_PATH.as_posix().format(model_id=model_id)
        if not path.is_file():
            raise R4Error(f"R4 continuous result missing: {model_id}")
        if not action_receipt_is_authoritative(cache_root, path, f"r4-continuous:{model_id}"):
            raise R4Error(f"R4 continuous result lacks completed attestation: {model_id}")
        document = load_json(path)
        if not self_sha256_valid(document):
            raise R4Error(f"R4 continuous result self identity invalid: {model_id}")
        operating_points[model_id] = document["operating_point"]
    r3_auc = _r3_macro_auc(cache_root)
    ranked = _rank_encoders(operating_points, r3_auc)
    top_two = [row["model_id"] for row in ranked[:2]]
    sensitivity_results: dict[str, dict[str, Any]] = {}
    for model_id in top_two:
        path = cache_root / SENSITIVITY_RESULT_RELATIVE_PATH.as_posix().format(model_id=model_id)
        if not path.is_file():
            raise R4Error(f"R4 sensitivity result missing: {model_id}")
        if not action_receipt_is_authoritative(cache_root, path, f"r4-sensitivity:{model_id}"):
            raise R4Error(f"R4 sensitivity result lacks completed attestation: {model_id}")
        sensitivity_results[model_id] = load_json(path)["operating_point"]
    document = with_self_sha256(
        _json_safe(
            {
                "schema_version": 1,
                "artifact_role": "r4_candidate_selection_ledger",
                "experiment_id": "speaker_representation_scd_v1",
                "authority": AUTHORITY,
                "scope": "legacy-common-gt-v1",
                "encoder_ranking": ranked,
                "top_two_encoder_ids": top_two,
                "sensitivity_operating_points": sensitivity_results,
                "representation_winner": max(
                    r3_auc, key=lambda model_id: r3_auc[model_id]
                ),
                "zero_shot_event_leader": ranked[0]["model_id"] if ranked else None,
                "efficient_backbone_candidate": min(
                    MODEL_IDS,
                    key=lambda model_id: (
                        operating_points[model_id]["metrics"]["f1_at_250ms"],
                        operating_points[model_id]["metrics"]["availability_latency_ms"][
                            "median_ms"
                        ]
                        or 1e18,
                    ),
                ),
                "claim_level": "exploratory_candidate_selection",
                "supervision_binding": {
                    "execution_id": os.environ.get("SRSCD_EXECUTION_LEASE_TOKEN"),
                    "expected_receipt_relative_path": SELECTION_LEDGER_RELATIVE_PATH.as_posix(),
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
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("continuous", "sensitivity", "report"), required=True)
    parser.add_argument("--encoder", choices=MODEL_IDS)
    args = parser.parse_args(argv)
    cache_root = Path(os.environ["SRSCD_CACHE_ROOT"]).resolve()
    requested = tuple(
        [sys.executable, "-m", __package__ + ".r4_continuous", *(argv or sys.argv[1:])]
    )
    if args.worker == "continuous":
        if args.encoder is None:
            raise SystemExit("continuous requires --encoder")
        print(run_continuous(args.encoder, cache_root, requested))
    elif args.worker == "sensitivity":
        if args.encoder is None:
            raise SystemExit("sensitivity requires --encoder")
        print(run_sensitivity(args.encoder, cache_root, requested))
    else:
        print(run_report(cache_root, requested))
    return 0


def run_continuous(model_id: str, cache_root: Path, requested_argv: tuple[str, ...]) -> Path:
    return _run_continuous(model_id, cache_root, requested_argv, sensitivity=False)


def run_sensitivity(model_id: str, cache_root: Path, requested_argv: tuple[str, ...]) -> Path:
    return _run_continuous(model_id, cache_root, requested_argv, sensitivity=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExecutionGuardError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
