from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import platform
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiments.online_speaker_memory_handoff.protocol import (
    CONFIG_PATH,
    EXPERIMENT_ROOT,
    R6Error,
    SpeakerRegion,
    _corpus_root,
    _regions_for_source,
    _source_rows,
    _waveform_paths,
    cache_root,
    input_paths,
    load_json,
    output_root,
    prepare,
    read_jsonl,
    sha256_file,
    validate_inputs,
    write_json,
    write_jsonl,
)
from experiments.speaker_representation_scd.r4_continuous import _mean_pool_l2_batch

HOP_SAMPLES = 1600
SAMPLE_RATE = 16000
BATCH_SIZE = 16


def _normalise(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1)
    result = np.full(values.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(norms) & (norms > 0)
    result[valid] = values[valid] / norms[valid, None]
    return result


def _model_record(root: Path, model_id: str) -> dict[str, Any]:
    acquisition = load_json(root / "manifests/r1_model_acquisition.json")
    record = next((row for row in acquisition["models"] if row["model_id"] == model_id), None)
    if record is None:
        raise R6Error(f"model acquisition record is missing: {model_id}")
    return record


def _make_extractor(root: Path, representation: str):
    if representation == "m-l1":
        from experiments.speaker_representation_scd.extraction.ssl import SSLExtractor

        record = _model_record(root, "mhubert-147")
        return SSLExtractor("mhubert-147", Path(record["root"]), threads=8)
    if representation in {"e-s3", "e-final"}:
        from experiments.speaker_representation_scd.extraction.eres_prepooling import (
            ERes2NetV2PrepoolExtractor,
        )

        record = _model_record(root, "eres2netv2-standard-prepool")
        return ERes2NetV2PrepoolExtractor(
            Path(record["checkpoint_root"]),
            Path(record["source_root"]),
            EXPERIMENT_ROOT.parent / "speaker_representation_scd" / "models/source_registry.json",
            threads=8,
        )
    raise R6Error(f"unknown representation: {representation}")


def _units(root: Path) -> list[dict[str, Any]]:
    config = load_json(CONFIG_PATH)
    path = output_root(config, root) / "protocol/units.jsonl"
    if not path.is_file():
        prepare(root)
    return read_jsonl(path)


def _needed_frontiers(
    units: Sequence[dict[str, Any]],
    role: str,
    context_ms: int,
    representation: str,
) -> dict[str, list[dict[str, Any]]]:
    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    context_samples = context_ms * 16
    if representation == "e-final" and context_ms in {1500, 2000}:
        for unit in units:
            if unit["role"] != role or int(unit["enrollment_ms"]) != context_ms:
                continue
            key = (str(unit["session_id"]), int(unit["enrollment_end_sample"]), context_ms)
            rows[key] = {
                "session_id": key[0],
                "frontier_sample": key[1],
                "kind": "enrollment",
                "context_ms": context_ms,
            }
    else:
        for unit in units:
            if unit["role"] != role:
                continue
            session_id = str(unit["session_id"])
            start = int(unit["stream_start_sample"])
            end = int(unit["stream_end_sample"])
            first = max(context_samples, ((start + HOP_SAMPLES - 1) // HOP_SAMPLES) * HOP_SAMPLES)
            for frontier in range(first, end + 1, HOP_SAMPLES):
                key = (session_id, frontier, context_ms)
                rows[key] = {
                    "session_id": session_id,
                    "frontier_sample": frontier,
                    "kind": "query",
                    "context_ms": context_ms,
                }
            if representation != "e-final":
                enrollment_start = int(unit["enrollment_start_sample"])
                enrollment_end = int(unit["enrollment_end_sample"])
                first_enrollment = max(
                    context_samples,
                    ((enrollment_start + context_samples + HOP_SAMPLES - 1) // HOP_SAMPLES)
                    * HOP_SAMPLES,
                )
                for frontier in range(first_enrollment, enrollment_end + 1, HOP_SAMPLES):
                    key = (session_id, frontier, context_ms)
                    rows[key] = {
                        "session_id": session_id,
                        "frontier_sample": frontier,
                        "kind": "enrollment",
                        "context_ms": context_ms,
                    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(
        rows.values(),
        key=lambda value: (value["session_id"], value["frontier_sample"], value["kind"]),
    ):
        grouped[str(row["session_id"])].append(row)
    return dict(grouped)


def _extraction_paths(
    root: Path, representation: str, role: str, context_ms: int
) -> tuple[Path, Path, Path]:
    config = load_json(CONFIG_PATH)
    directory = output_root(config, root) / "features" / representation / role
    stem = f"context_{context_ms}ms"
    return (
        directory / f"{stem}.npy",
        directory / f"{stem}.jsonl",
        directory / f"{stem}.manifest.json",
    )


def extract(root: Path, representation: str, role: str, context_ms: int) -> Path:
    config = load_json(CONFIG_PATH)
    validate_inputs(config, root)
    allowed = set(config["representations"][representation]["query_context_ms"])
    if representation == "e-final":
        allowed.update(config["representations"][representation]["enrollment_ms"])
    if context_ms not in allowed:
        raise R6Error(f"context {context_ms} ms is not configured for {representation}")
    vector_path, index_path, manifest_path = _extraction_paths(
        root, representation, role, context_ms
    )
    if vector_path.is_file() and index_path.is_file() and manifest_path.is_file():
        manifest = load_json(manifest_path)
        if manifest.get("vectors_sha256") == sha256_file(vector_path) and manifest.get(
            "index_sha256"
        ) == sha256_file(index_path):
            return manifest_path
        raise R6Error(f"incomplete or modified feature cache: {manifest_path}")
    units = _units(root)
    needed = _needed_frontiers(units, role, context_ms, representation)
    row_count = sum(len(rows) for rows in needed.values())
    if row_count == 0:
        raise R6Error(f"no extraction rows for {representation} {role} {context_ms}")
    paths = input_paths(root)
    source_rows = _source_rows(paths)
    corpus = _corpus_root(root)
    waveform_paths = _waveform_paths(paths, corpus)
    extractor = _make_extractor(root, representation)
    dimension = 192 if representation == "e-final" else (768 if representation == "m-l1" else 10240)
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    temp_vector = vector_path.with_suffix(".npy.tmp")
    temp_index = index_path.with_suffix(".jsonl.tmp")
    vectors = np.lib.format.open_memmap(
        temp_vector, mode="w+", dtype=np.float32, shape=(row_count, dimension)
    )
    index_rows: list[dict[str, Any]] = []
    offset = 0
    started = time.perf_counter()
    import soundfile as sf

    for session_id, rows in needed.items():
        source = source_rows[session_id]
        waveform_path = waveform_paths[str(source["waveform_id"])]
        audio, sample_rate = sf.read(str(waveform_path), dtype="float32", always_2d=True)
        if sample_rate != SAMPLE_RATE or audio.shape[1] != 1:
            raise R6Error(f"waveform geometry differs for {session_id}")
        waveform = np.ascontiguousarray(audio[:, 0], dtype=np.float32)
        for batch_start in range(0, len(rows), BATCH_SIZE):
            batch_rows = rows[batch_start : batch_start + BATCH_SIZE]
            windows = [
                np.ascontiguousarray(
                    waveform[
                        int(row["frontier_sample"]) - context_ms * 16 : int(row["frontier_sample"])
                    ],
                    dtype=np.float32,
                )
                for row in batch_rows
            ]
            if any(window.shape != (context_ms * 16,) for window in windows):
                raise R6Error(f"window geometry differs for {session_id}")
            if representation == "m-l1":
                batch = extractor.extract(
                    windows, [int(row["frontier_sample"]) for row in batch_rows], layer_ids=("L1",)
                )
                values = _mean_pool_l2_batch(batch.layers["L1"], batch.valid_lengths["L1"])
            elif representation == "e-s3":
                batch = extractor.extract(
                    windows, [int(row["frontier_sample"]) for row in batch_rows], tap_ids=("S3",)
                )
                values = _mean_pool_l2_batch(batch.layers["S3"], batch.valid_lengths["S3"])
            else:
                batch = extractor.extract(
                    windows, [int(row["frontier_sample"]) for row in batch_rows], tap_ids=()
                )
                values = _normalise(np.asarray(batch.official_embedding, dtype=np.float32))
            if values.shape != (len(batch_rows), dimension) or not np.isfinite(values).all():
                raise R6Error(f"invalid extracted tensor for {session_id}: {values.shape}")
            vectors[offset : offset + len(batch_rows)] = values
            for local_index, row in enumerate(batch_rows):
                index_rows.append({**row, "row": offset + local_index})
            offset += len(batch_rows)
        vectors.flush()
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "stage": "extract",
                    "representation": representation,
                    "role": role,
                    "context_ms": context_ms,
                    "session_id": session_id,
                    "completed_rows": offset,
                    "total_rows": row_count,
                    "elapsed_seconds": round(elapsed, 3),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    del vectors
    write_jsonl(temp_index, index_rows)
    temp_vector.replace(vector_path)
    temp_index.replace(index_path)
    elapsed = time.perf_counter() - started
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "artifact_role": "r6_feature_cache",
            "representation": representation,
            "role": role,
            "context_ms": context_ms,
            "row_count": row_count,
            "dimension": dimension,
            "vectors_sha256": sha256_file(vector_path),
            "index_sha256": sha256_file(index_path),
            "wall_seconds": round(elapsed, 6),
            "hardware": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
                "backend": "cpu",
            },
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    return manifest_path


class VectorStore:
    def __init__(self, vectors: np.ndarray, rows: Sequence[dict[str, Any]]) -> None:
        self.vectors = vectors
        grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["session_id"])].append((int(row["frontier_sample"]), int(row["row"])))
        self.frontiers: dict[str, np.ndarray] = {}
        self.row_indices: dict[str, np.ndarray] = {}
        for session_id, pairs in grouped.items():
            pairs.sort()
            self.frontiers[session_id] = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
            self.row_indices[session_id] = np.asarray([pair[1] for pair in pairs], dtype=np.int64)

    def exact(self, session_id: str, frontier: int) -> np.ndarray | None:
        values = self.frontiers.get(session_id)
        if values is None:
            return None
        index = int(np.searchsorted(values, frontier))
        if index >= len(values) or int(values[index]) != frontier:
            return None
        return np.asarray(self.vectors[int(self.row_indices[session_id][index])], dtype=np.float32)

    def between(self, session_id: str, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        values = self.frontiers.get(session_id)
        if values is None:
            return np.empty((0,), dtype=np.int64), np.empty(
                (0, self.vectors.shape[1]), dtype=np.float32
            )
        left = int(np.searchsorted(values, start, side="left"))
        right = int(np.searchsorted(values, end, side="right"))
        rows = self.row_indices[session_id][left:right]
        return values[left:right], np.asarray(self.vectors[rows], dtype=np.float32)


def _load_extracted(root: Path, representation: str, role: str, context_ms: int) -> VectorStore:
    vector_path, index_path, manifest_path = _extraction_paths(
        root, representation, role, context_ms
    )
    if not manifest_path.is_file():
        raise R6Error(f"feature cache is missing: {manifest_path}")
    manifest = load_json(manifest_path)
    vectors = np.load(vector_path, mmap_mode="r")
    rows = read_jsonl(index_path)
    if (
        vectors.shape != (int(manifest["row_count"]), int(manifest["dimension"]))
        or len(rows) != vectors.shape[0]
    ):
        raise R6Error(f"feature cache geometry differs: {manifest_path}")
    return VectorStore(vectors, rows)


def _load_r4(root: Path, representation: str) -> VectorStore:
    model_id = "mhubert-147" if representation == "m-l1" else "eres2netv2-standard-prepool"
    directory = root / "data/r4/legacy_common_gt/pooled" / model_id
    vectors = np.load(directory / "vectors_300.npy", mmap_mode="r")
    rows: list[dict[str, Any]] = []
    for source in read_jsonl(directory / "index_300.jsonl"):
        start = int(source["row_start"])
        for local, frontier in enumerate(source["frontier_samples"]):
            rows.append(
                {
                    "session_id": source["session_id"],
                    "frontier_sample": int(frontier),
                    "row": start + local,
                }
            )
    return VectorStore(vectors, rows)


def _store(root: Path, representation: str, role: str, context_ms: int) -> VectorStore:
    if role == "evaluation" and representation in {"m-l1", "e-s3"} and context_ms == 300:
        return _load_r4(root, representation)
    return _load_extracted(root, representation, role, context_ms)


class RegionIndex:
    def __init__(self, regions: Sequence[SpeakerRegion]) -> None:
        self.regions = list(regions)
        self.ends = [region.end_sample for region in self.regions]

    def window(self, start: int, end: int, current_speaker: str) -> dict[str, Any]:
        index = bisect.bisect_right(self.ends, start)
        active = 0
        current = 0
        other = 0
        overlap = 0
        ambiguous = 0
        speakers: set[str] = set()
        while index < len(self.regions):
            region = self.regions[index]
            if region.start_sample >= end:
                break
            amount = min(end, region.end_sample) - max(start, region.start_sample)
            if amount > 0:
                if region.ambiguous:
                    ambiguous += amount
                if region.speakers:
                    active += amount
                    speakers.update(region.speakers)
                    has_current = current_speaker in region.speakers
                    has_other = bool(set(region.speakers) - {current_speaker})
                    if has_current:
                        current += amount
                    if has_other:
                        other += amount
                    if has_current and has_other:
                        overlap += amount
            index += 1
        duration = max(1, end - start)
        if ambiguous:
            label = "ambiguous"
        elif active == 0:
            label = "silence"
        elif overlap > 0 or (current > 0 and other > 0):
            label = "overlap"
        elif other > 0:
            label = "other"
        else:
            label = "same"
        return {
            "label": label,
            "speech_fraction": active / duration,
            "current_fraction": current / duration,
            "other_fraction": other / duration,
            "overlap_fraction": overlap / duration,
            "speakers": sorted(speakers),
        }


def _prototype(vectors: np.ndarray, aggregation: str) -> np.ndarray | None:
    if vectors.shape[0] == 0:
        return None
    normalized = _normalise(vectors)
    if not np.isfinite(normalized).all():
        return None
    if aggregation == "mean":
        value = normalized.mean(axis=0, dtype=np.float64)
    elif aggregation == "medoid":
        similarities = normalized @ normalized.T
        value = normalized[int(np.argmax(similarities.mean(axis=1)))]
    else:
        raise R6Error(f"unknown aggregation: {aggregation}")
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 0:
        return None
    return np.asarray(value / norm, dtype=np.float32)


def raw_score_path(
    root: Path,
    representation: str,
    role: str,
    context_ms: int,
    enrollment_ms: int,
    aggregation: str,
) -> Path:
    config = load_json(CONFIG_PATH)
    name = f"{role}_q{context_ms}_e{enrollment_ms}_{aggregation}.csv"
    return output_root(config, root) / "a1" / representation / "raw_scores" / name


def score_raw(
    root: Path,
    representation: str,
    role: str,
    context_ms: int,
    enrollment_ms: int,
    aggregation: str,
) -> Path:
    path = raw_score_path(root, representation, role, context_ms, enrollment_ms, aggregation)
    if path.is_file():
        return path
    units = [
        row
        for row in _units(root)
        if row["role"] == role and int(row["enrollment_ms"]) == enrollment_ms
    ]
    query_store = _store(root, representation, role, context_ms)
    enrollment_store = (
        _store(root, representation, role, enrollment_ms)
        if representation == "e-final"
        else query_store
    )
    paths = input_paths(root)
    sources = _source_rows(paths)
    corpus = _corpus_root(root)
    region_indices = {
        session_id: RegionIndex(_regions_for_source(sources[session_id], corpus))
        for session_id in sorted({str(unit["session_id"]) for unit in units})
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".csv.tmp")
    fields = [
        "unit_id",
        "session_id",
        "corpus",
        "role",
        "event_kind",
        "current_speaker",
        "stream_start_sample",
        "stream_end_sample",
        "new_speaker_onset_sample",
        "exclusive_new_onset_sample",
        "current_returns_sample",
        "frontier_sample",
        "query_start_sample",
        "same_score",
        "other_score",
        "label",
        "speech_fraction",
        "current_fraction",
        "other_fraction",
        "overlap_fraction",
    ]
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        unavailable = 0
        written = 0
        for unit in units:
            session_id = str(unit["session_id"])
            current = str(unit["current_speaker"])
            if representation == "e-final":
                enrollment_vector = enrollment_store.exact(
                    session_id, int(unit["enrollment_end_sample"])
                )
                prototype = (
                    None
                    if enrollment_vector is None
                    else _prototype(enrollment_vector[None, :], aggregation)
                )
            else:
                enrollment_frontiers, enrollment_vectors = enrollment_store.between(
                    session_id,
                    int(unit["enrollment_start_sample"]) + context_ms * 16,
                    int(unit["enrollment_end_sample"]),
                )
                eligible: list[np.ndarray] = []
                for frontier, vector in zip(
                    enrollment_frontiers.tolist(), enrollment_vectors, strict=True
                ):
                    label = region_indices[session_id].window(
                        frontier - context_ms * 16, frontier, current
                    )
                    if label["label"] == "same" and label["speech_fraction"] >= 0.65:
                        eligible.append(vector)
                prototype = (
                    _prototype(np.asarray(eligible, dtype=np.float32), aggregation)
                    if eligible
                    else None
                )
            if prototype is None:
                unavailable += 1
                continue
            frontiers, vectors = query_store.between(
                session_id,
                int(unit["stream_start_sample"]),
                int(unit["stream_end_sample"]),
            )
            if vectors.shape[0] == 0:
                unavailable += 1
                continue
            scores = np.clip(vectors @ prototype, -1.0, 1.0)
            for frontier, same_score in zip(frontiers.tolist(), scores.tolist(), strict=True):
                labels = region_indices[session_id].window(
                    frontier - context_ms * 16, frontier, current
                )
                writer.writerow(
                    {
                        "unit_id": unit["unit_id"],
                        "session_id": session_id,
                        "corpus": unit["corpus"],
                        "role": role,
                        "event_kind": unit["event_kind"],
                        "current_speaker": current,
                        "stream_start_sample": unit["stream_start_sample"],
                        "stream_end_sample": unit["stream_end_sample"],
                        "new_speaker_onset_sample": unit["new_speaker_onset_sample"],
                        "exclusive_new_onset_sample": unit["exclusive_new_onset_sample"],
                        "current_returns_sample": unit["current_returns_sample"],
                        "frontier_sample": frontier,
                        "query_start_sample": frontier - context_ms * 16,
                        "same_score": f"{same_score:.9g}",
                        "other_score": f"{1.0 - same_score:.9g}",
                        "label": labels["label"],
                        "speech_fraction": f"{labels['speech_fraction']:.9g}",
                        "current_fraction": f"{labels['current_fraction']:.9g}",
                        "other_fraction": f"{labels['other_fraction']:.9g}",
                        "overlap_fraction": f"{labels['overlap_fraction']:.9g}",
                    }
                )
                written += 1
    temp.replace(path)
    write_json(
        path.with_suffix(".manifest.json"),
        {
            "schema_version": 1,
            "artifact_role": "r6_a1_raw_scores",
            "representation": representation,
            "role": role,
            "query_context_ms": context_ms,
            "enrollment_ms": enrollment_ms,
            "aggregation": aggregation,
            "row_count": written,
            "unavailable_unit_count": unavailable,
            "sha256": sha256_file(path),
            "created_at_utc": datetime.now(UTC).isoformat(),
            "config_sha256": sha256_file(CONFIG_PATH),
        },
    )
    return path


def _read_scores(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    integer_fields = {
        "stream_start_sample",
        "stream_end_sample",
        "frontier_sample",
        "query_start_sample",
    }
    optional_integer_fields = {
        "new_speaker_onset_sample",
        "exclusive_new_onset_sample",
        "current_returns_sample",
    }
    float_fields = {
        "same_score",
        "other_score",
        "speech_fraction",
        "current_fraction",
        "other_fraction",
        "overlap_fraction",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for name in integer_fields:
                row[name] = int(row[name])
            for name in optional_integer_fields:
                row[name] = int(row[name]) if row[name] else None
            for name in float_fields:
                row[name] = float(row[name])
            rows.append(row)
    return rows


def _thresholds(rows: Sequence[dict[str, Any]]) -> list[float]:
    values = np.asarray(
        [
            float(row["other_score"])
            for row in rows
            if row["label"] in {"same", "other"}
            and float(row["speech_fraction"]) >= 0.5
            and math.isfinite(float(row["other_score"]))
        ],
        dtype=np.float64,
    )
    if values.size == 0:
        raise R6Error("no finite development SAME/OTHER scores")
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, 101))
    maximum = round(float(np.max(values)), 8)
    return sorted(
        {round(float(value), 8) for value in quantiles} | {float(np.nextafter(maximum, math.inf))}
    )


def _events_for_unit(
    rows: Sequence[dict[str, Any]], threshold: float, persistence_ms: int
) -> tuple[list[int], list[int]]:
    required = max(1, math.ceil(persistence_ms / 100))
    candidate_events: list[int] = []
    handoff_events: list[int] = []
    active = False
    consecutive = 0
    handoff_emitted = False
    last_frontier = None
    for row in rows:
        frontier = int(row["frontier_sample"])
        speech = float(row["speech_fraction"]) >= 0.5
        high = speech and float(row["other_score"]) >= threshold
        contiguous = last_frontier is None or frontier - last_frontier <= HOP_SAMPLES
        if high:
            if not active or not contiguous:
                candidate_events.append(frontier)
                consecutive = 1
                handoff_emitted = False
            else:
                consecutive += 1
            active = True
            if consecutive >= required and not handoff_emitted:
                handoff_events.append(frontier)
                handoff_emitted = True
        elif row["label"] != "silence":
            active = False
            consecutive = 0
            handoff_emitted = False
        last_frontier = frontier
    return candidate_events, handoff_events


def _percentiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"median_ms": None, "p90_ms": None, "p95_ms": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "median_ms": float(np.median(array)),
        "p90_ms": float(np.quantile(array, 0.90)),
        "p95_ms": float(np.quantile(array, 0.95)),
    }


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def _meeting_sensitivity(per_meeting: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total_tp = sum(row["tp_1000"] for row in per_meeting.values())
    total_positives = sum(row["positives"] for row in per_meeting.values())
    leave_one_out: dict[str, dict[str, float | int | None]] = {}
    for session_id, row in sorted(per_meeting.items()):
        positives = total_positives - row["positives"]
        true_positives = total_tp - row["tp_1000"]
        leave_one_out[session_id] = {
            "tp_1000": true_positives,
            "positives": positives,
            "recall_1000": _safe_ratio(true_positives, positives),
        }
    shares = {
        session_id: _safe_ratio(row["tp_1000"], total_tp)
        for session_id, row in sorted(per_meeting.items())
    }
    finite_shares = [value for value in shares.values() if value is not None]
    maximum_share = max(finite_shares) if finite_shares else None
    return {
        "leave_one_meeting_out": leave_one_out,
        "true_positive_share_by_meeting": shares,
        "maximum_true_positive_share": maximum_share,
        "dominated_by_one_meeting": maximum_share is not None and maximum_share > 0.5,
    }


def _strata_summary(
    unit_results: Sequence[dict[str, Any]], return_windows_ms: Sequence[int]
) -> dict[str, Any]:
    strata: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "unit_count": 0,
            "positive_count": 0,
            "hit_1000": 0,
            "miss_1000": 0,
            "prediction_count": 0,
        }
    )
    for row in unit_results:
        stratum = str(row["stratum"])
        summary = strata[stratum]
        summary["unit_count"] += 1
        summary["prediction_count"] += len(row["predictions"])
        if row["ground_truth_sample"] is not None:
            summary["positive_count"] += 1
            if row["matched_at_1000ms"]:
                summary["hit_1000"] += 1
            else:
                summary["miss_1000"] += 1
    by_event_kind = {
        name: {
            **summary,
            "recall_1000": _safe_ratio(summary["hit_1000"], summary["positive_count"]),
        }
        for name, summary in sorted(strata.items())
    }
    backchannel: dict[str, Any] = {}
    for window_ms in return_windows_ms:
        eligible = [
            row
            for row in unit_results
            if row["ground_truth_sample"] is not None
            and row["current_returns_sample"] is not None
            and 0
            <= int(row["current_returns_sample"]) - int(row["ground_truth_sample"])
            <= int(window_ms) * 16
        ]
        rejected = sum(
            1
            for row in eligible
            if not any(
                int(event) <= int(row["current_returns_sample"]) for event in row["predictions"]
            )
        )
        backchannel[str(window_ms)] = {
            "unit_count": len(eligible),
            "rejected_count": rejected,
            "rejection_rate": _safe_ratio(rejected, len(eligible)),
        }
    return {"by_event_kind": by_event_kind, "backchannel_return_windows_ms": backchannel}


def _evaluate_events(
    grouped: dict[str, list[dict[str, Any]]],
    threshold: float,
    persistence_ms: int,
    event_type: str,
    symmetric_windows_ms: Sequence[int],
    causal_windows_ms: Sequence[int],
    return_windows_ms: Sequence[int],
) -> dict[str, Any]:
    predictions = 0
    positives = 0
    evaluated_samples = 0
    latencies: list[float] = []
    early_alerts = 0
    symmetric_hits = {window: 0 for window in symmetric_windows_ms}
    causal_hits = {window: 0 for window in causal_windows_ms}
    per_meeting: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "tp_1000": 0,
            "fp": 0,
            "fn_1000": 0,
            "predictions": 0,
            "positives": 0,
            "evaluated_samples": 0,
        }
    )
    unit_results: list[dict[str, Any]] = []
    for unit_id, rows in grouped.items():
        rows = sorted(rows, key=lambda row: int(row["frontier_sample"]))
        candidate_events, handoff_events = _events_for_unit(rows, threshold, persistence_ms)
        events = candidate_events if event_type == "candidate" else handoff_events
        gt = rows[0]["new_speaker_onset_sample"]
        session_id = str(rows[0]["session_id"])
        duration = int(rows[0]["stream_end_sample"]) - int(rows[0]["stream_start_sample"])
        evaluated_samples += max(0, duration)
        per_meeting[session_id]["evaluated_samples"] += max(0, duration)
        predictions += len(events)
        per_meeting[session_id]["predictions"] += len(events)
        if gt is not None:
            positives += 1
            per_meeting[session_id]["positives"] += 1
        matched_1000 = False
        first_causal = None
        if gt is not None:
            before = [event for event in events if event < gt]
            early_alerts += len(before)
            for window in symmetric_windows_ms:
                if any(abs(event - gt) <= window * 16 for event in events):
                    symmetric_hits[window] += 1
            for window in causal_windows_ms:
                if any(gt <= event <= gt + window * 16 for event in events):
                    causal_hits[window] += 1
            causal = [event for event in events if event >= gt]
            if causal:
                first_causal = min(causal)
                latencies.append((first_causal - gt) / 16)
            matched_1000 = any(gt <= event <= gt + 1000 * 16 for event in events)
        if matched_1000:
            per_meeting[session_id]["tp_1000"] += 1
        elif gt is not None:
            per_meeting[session_id]["fn_1000"] += 1
        matched_predictions: set[int] = set()
        if gt is not None:
            candidates = [
                (abs(event - gt), index)
                for index, event in enumerate(events)
                if abs(event - gt) <= 1000 * 16
            ]
            if candidates:
                matched_predictions.add(min(candidates)[1])
        fp = len(events) - len(matched_predictions)
        per_meeting[session_id]["fp"] += fp
        pre_event = [
            row for row in rows if gt is not None and int(row["frontier_sample"]) < int(gt)
        ]
        event_kind = str(rows[0]["event_kind"])
        if event_kind == "clean" and pre_event and pre_event[-1]["label"] == "silence":
            stratum = "silence_gap"
        else:
            stratum = event_kind
        unit_results.append(
            {
                "unit_id": unit_id,
                "session_id": session_id,
                "event_kind": event_kind,
                "stratum": stratum,
                "ground_truth_sample": gt,
                "predictions": events,
                "matched_at_1000ms": matched_1000,
                "first_causal_prediction": first_causal,
                "current_returns_sample": rows[0]["current_returns_sample"],
            }
        )
    total_hours = evaluated_samples / SAMPLE_RATE / 3600
    false_predictions = sum(row["fp"] for row in per_meeting.values())
    for row in per_meeting.values():
        meeting_hours = row["evaluated_samples"] / SAMPLE_RATE / 3600
        row["evaluated_hours"] = meeting_hours
        row["false_events_per_hour"] = row["fp"] / meeting_hours if meeting_hours else None
    symmetric: dict[str, Any] = {}
    for window, hit_count in symmetric_hits.items():
        precision = _safe_ratio(hit_count, predictions)
        recall = _safe_ratio(hit_count, positives)
        symmetric[str(window)] = {
            "tp": hit_count,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
        }
    causal = {
        str(window): {
            "tp": hit_count,
            "recall": _safe_ratio(hit_count, positives),
        }
        for window, hit_count in causal_hits.items()
    }
    return {
        "event_type": event_type,
        "threshold": threshold,
        "persistence_ms": persistence_ms,
        "prediction_count": predictions,
        "positive_count": positives,
        "false_event_count": false_predictions,
        "evaluated_hours": total_hours,
        "false_events_per_hour": false_predictions / total_hours if total_hours else None,
        "early_alert_count": early_alerts,
        "symmetric": symmetric,
        "causal": causal,
        "latency": _percentiles(latencies),
        "per_meeting": dict(per_meeting),
        "meeting_sensitivity": _meeting_sensitivity(dict(per_meeting)),
        "strata": _strata_summary(unit_results, return_windows_ms),
        "unit_results": unit_results,
    }


def _representation_diagnostics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score, roc_curve

    selected = [
        row
        for row in rows
        if row["label"] in {"same", "other"} and float(row["speech_fraction"]) >= 0.5
    ]
    labels = np.asarray([1 if row["label"] == "other" else 0 for row in selected], dtype=np.int64)
    scores = np.asarray([float(row["other_score"]) for row in selected], dtype=np.float64)
    if len(set(labels.tolist())) < 2:
        return {
            "frame_pooled_roc_auc": None,
            "eer": None,
            "same_count": int((labels == 0).sum()),
            "other_count": int((labels == 1).sum()),
        }
    false_positive, true_positive, _ = roc_curve(labels, scores)
    false_negative = 1.0 - true_positive
    index = int(np.nanargmin(np.abs(false_positive - false_negative)))
    session_auc: dict[str, float] = {}
    for session_id in sorted({str(row["session_id"]) for row in selected}):
        session_rows = [row for row in selected if row["session_id"] == session_id]
        session_labels = [1 if row["label"] == "other" else 0 for row in session_rows]
        if len(set(session_labels)) == 2:
            session_auc[session_id] = float(
                roc_auc_score(session_labels, [float(row["other_score"]) for row in session_rows])
            )
    same_scores = scores[labels == 0]
    other_scores = scores[labels == 1]
    return {
        "frame_pooled_roc_auc": float(roc_auc_score(labels, scores)),
        "session_balanced_roc_auc": float(np.mean(list(session_auc.values())))
        if session_auc
        else None,
        "session_roc_auc": session_auc,
        "eer": float((false_positive[index] + false_negative[index]) / 2),
        "same_count": int(same_scores.size),
        "other_count": int(other_scores.size),
        "same_quantiles": {str(q): float(np.quantile(same_scores, q)) for q in (0.05, 0.5, 0.95)},
        "other_quantiles": {str(q): float(np.quantile(other_scores, q)) for q in (0.05, 0.5, 0.95)},
    }


def _group_scores(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["unit_id"])].append(row)
    return dict(grouped)


def _select_development(curve: Sequence[dict[str, Any]]) -> dict[str, Any]:
    viable = [
        row
        for row in curve
        if row["candidate"]["false_events_per_hour"] is not None
        and row["candidate"]["false_events_per_hour"] <= 5.0
    ]
    pool = viable or list(curve)
    if not pool:
        raise R6Error("development threshold curve is empty")
    return max(
        pool,
        key=lambda row: (
            float(row["candidate"]["causal"]["1000"]["recall"] or 0.0),
            -float(row["candidate"]["false_events_per_hour"] or 1e18),
            -float(row["candidate"]["latency"]["median_ms"] or 1e18),
            row["threshold"],
        ),
    )


def analyse_config(
    root: Path,
    representation: str,
    context_ms: int,
    enrollment_ms: int,
    aggregation: str,
) -> Path:
    config = load_json(CONFIG_PATH)
    directory = output_root(config, root) / "a1" / representation / "metrics"
    name = f"q{context_ms}_e{enrollment_ms}_{aggregation}.json"
    result_path = directory / name
    if result_path.is_file():
        existing = load_json(result_path)
        if existing.get("schema_version") == 2 and existing.get("provenance", {}).get(
            "code_sha256"
        ) == sha256_file(Path(__file__).resolve()):
            return result_path
    development_path = score_raw(
        root, representation, "development", context_ms, enrollment_ms, aggregation
    )
    evaluation_path = score_raw(
        root, representation, "evaluation", context_ms, enrollment_ms, aggregation
    )
    development_rows = _read_scores(development_path)
    evaluation_rows = _read_scores(evaluation_path)
    development_manifest = load_json(development_path.with_suffix(".manifest.json"))
    evaluation_manifest = load_json(evaluation_path.with_suffix(".manifest.json"))
    development_grouped = _group_scores(development_rows)
    evaluation_grouped = _group_scores(evaluation_rows)
    curve: list[dict[str, Any]] = []
    for threshold in _thresholds(development_rows):
        candidate = _evaluate_events(
            development_grouped,
            threshold,
            100,
            "candidate",
            config["candidate_windows_ms"],
            config["causal_recall_windows_ms"],
            config["backchannel_return_windows_ms"],
        )
        handoff = {
            name: _evaluate_events(
                development_grouped,
                threshold,
                int(duration),
                "handoff",
                config["handoff_windows_ms"],
                config["handoff_recall_windows_ms"],
                config["backchannel_return_windows_ms"],
            )
            for name, duration in config["persistence_views_ms"].items()
        }
        for value in (candidate, *handoff.values()):
            value.pop("unit_results", None)
        curve.append({"threshold": threshold, "candidate": candidate, "handoff": handoff})
    selected = _select_development(curve)
    threshold = float(selected["threshold"])
    evaluation_candidate = _evaluate_events(
        evaluation_grouped,
        threshold,
        100,
        "candidate",
        config["candidate_windows_ms"],
        config["causal_recall_windows_ms"],
        config["backchannel_return_windows_ms"],
    )
    evaluation_handoff = {
        name: _evaluate_events(
            evaluation_grouped,
            threshold,
            int(duration),
            "handoff",
            config["handoff_windows_ms"],
            config["handoff_recall_windows_ms"],
            config["backchannel_return_windows_ms"],
        )
        for name, duration in config["persistence_views_ms"].items()
    }
    unit_results = {
        "candidate": evaluation_candidate.pop("unit_results"),
        "handoff": {
            name: metrics.pop("unit_results") for name, metrics in evaluation_handoff.items()
        },
    }
    gate_recall = evaluation_candidate["causal"]["1000"]["recall"]
    gate_fp = evaluation_candidate["false_events_per_hour"]
    dominated = evaluation_candidate["meeting_sensitivity"]["dominated_by_one_meeting"]
    if gate_fp is None or gate_fp > 10.0:
        gate = "stop"
    elif gate_recall is None or gate_recall < 0.30:
        gate = "stop"
    elif gate_recall <= 0.60 or dominated:
        gate = "conditional"
    else:
        gate = "promote"
    write_jsonl(
        result_path.with_name(result_path.stem + ".events.jsonl"),
        [{"view": "candidate", **row} for row in unit_results["candidate"]]
        + [
            {"view": f"handoff_{name}", **row}
            for name, rows in unit_results["handoff"].items()
            for row in rows
        ],
    )
    write_json(
        result_path,
        {
            "schema_version": 2,
            "artifact_role": "r6_a1_metrics",
            "representation": representation,
            "query_context_ms": context_ms,
            "enrollment_ms": enrollment_ms,
            "aggregation": aggregation,
            "development": {
                "diagnostics": _representation_diagnostics(development_rows),
                "unavailable_or_abstained_event_count": development_manifest[
                    "unavailable_unit_count"
                ],
                "selected_threshold": threshold,
                "selected_operating_point": selected,
                "curve": curve,
            },
            "evaluation": {
                "diagnostics": _representation_diagnostics(evaluation_rows),
                "unavailable_or_abstained_event_count": evaluation_manifest[
                    "unavailable_unit_count"
                ],
                "candidate": evaluation_candidate,
                "handoff": evaluation_handoff,
            },
            "gate": {
                "decision": gate,
                "candidate_recall_at_1000ms": gate_recall,
                "candidate_false_events_per_hour": gate_fp,
                "dominated_by_one_meeting": dominated,
                "note": "The threshold was selected only on the disjoint natural-meeting development sessions.",
            },
            "provenance": {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "config_sha256": sha256_file(CONFIG_PATH),
                "code_sha256": sha256_file(Path(__file__).resolve()),
                "worker_job_id": os.environ.get("ORCA_WORKER_JOB_ID"),
            },
        },
    )
    return result_path


def _make_plots(root: Path, metric_path: Path) -> None:
    import matplotlib.pyplot as plt

    document = load_json(metric_path)
    representation = str(document["representation"])
    context_ms = int(document["query_context_ms"])
    enrollment_ms = int(document["enrollment_ms"])
    aggregation = str(document["aggregation"])
    config = load_json(CONFIG_PATH)
    plot_root = output_root(config, root) / "a1" / representation / "plots" / metric_path.stem
    plot_root.mkdir(parents=True, exist_ok=True)
    evaluation_rows = _read_scores(
        raw_score_path(root, representation, "evaluation", context_ms, enrollment_ms, aggregation)
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    same = [
        row["same_score"]
        for row in evaluation_rows
        if row["label"] == "same" and row["speech_fraction"] >= 0.5
    ]
    other = [
        row["same_score"]
        for row in evaluation_rows
        if row["label"] == "other" and row["speech_fraction"] >= 0.5
    ]
    axis.hist(same, bins=60, density=True, alpha=0.55, label="SAME")
    axis.hist(other, bins=60, density=True, alpha=0.55, label="OTHER")
    axis.set_xlabel("CURRENT-to-query cosine")
    axis.set_ylabel("density")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_root / "same_other_distribution.png", dpi=150)
    plt.close(figure)
    curve = document["development"]["curve"]
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(
        [row["candidate"]["false_events_per_hour"] for row in curve],
        [row["candidate"]["causal"]["1000"]["recall"] for row in curve],
        marker=".",
        linewidth=1,
    )
    axis.set_xlabel("development candidate false events/hour")
    axis.set_ylabel("development causal Recall@1000ms")
    axis.set_xscale("symlog", linthresh=1.0)
    figure.tight_layout()
    figure.savefig(plot_root / "recall_false_event_curve.png", dpi=150)
    plt.close(figure)
    grouped = _group_scores(evaluation_rows)
    event_rows = read_jsonl(metric_path.with_name(metric_path.stem + ".events.jsonl"))
    candidate_results = {row["unit_id"]: row for row in event_rows if row["view"] == "candidate"}
    selections: dict[str, str] = {}
    for unit_id, rows in grouped.items():
        result = candidate_results.get(unit_id)
        if result is None:
            continue
        kind = str(rows[0]["event_kind"])
        if kind not in selections:
            selections[kind] = unit_id
        if rows[0]["current_returns_sample"] is not None and "backchannel" not in selections:
            selections["backchannel"] = unit_id
        if (
            result["ground_truth_sample"] is not None
            and not result["matched_at_1000ms"]
            and "miss" not in selections
        ):
            selections["miss"] = unit_id
        if (
            result["ground_truth_sample"] is None
            and result["predictions"]
            and "false_candidate" not in selections
        ):
            selections["false_candidate"] = unit_id
    for label, unit_id in list(selections.items())[:8]:
        rows = sorted(grouped[unit_id], key=lambda row: row["frontier_sample"])
        base = rows[0]["stream_start_sample"]
        x = [(row["frontier_sample"] - base) / SAMPLE_RATE for row in rows]
        y = [row["same_score"] for row in rows]
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.plot(x, y, linewidth=1.2, label="CURRENT cosine")
        threshold = 1.0 - float(document["development"]["selected_threshold"])
        axis.axhline(
            threshold, color="tab:red", linestyle="--", linewidth=1, label="selected threshold"
        )
        gt = rows[0]["new_speaker_onset_sample"]
        if gt is not None:
            axis.axvline(
                (gt - base) / SAMPLE_RATE, color="black", linewidth=1, label="GT new speaker"
            )
        for prediction in candidate_results[unit_id]["predictions"]:
            axis.axvline((prediction - base) / SAMPLE_RATE, color="tab:orange", alpha=0.55)
        axis.set_xlabel("seconds from stream start")
        axis.set_ylabel("cosine")
        axis.set_title(f"{label}: {unit_id}")
        axis.legend(loc="best")
        figure.tight_layout()
        safe = label.replace("/", "_")
        figure.savefig(plot_root / f"timeline_{safe}.png", dpi=150)
        plt.close(figure)


def run_representation(root: Path, representation: str) -> list[Path]:
    config = load_json(CONFIG_PATH)
    spec = config["representations"][representation]
    if representation in {"m-l1", "e-s3"}:
        extract(root, representation, "development", 300)
    else:
        for role in ("development", "evaluation"):
            for context_ms in sorted(set(spec["query_context_ms"] + spec["enrollment_ms"])):
                extract(root, representation, role, int(context_ms))
    results: list[Path] = []
    for context_ms in spec["query_context_ms"]:
        for enrollment_ms in spec["enrollment_ms"]:
            for aggregation in spec["aggregation"]:
                path = analyse_config(
                    root,
                    representation,
                    int(context_ms),
                    int(enrollment_ms),
                    str(aggregation),
                )
                _make_plots(root, path)
                results.append(path)
                print(
                    json.dumps(
                        {
                            "stage": "a1_config_complete",
                            "representation": representation,
                            "query_context_ms": context_ms,
                            "enrollment_ms": enrollment_ms,
                            "aggregation": aggregation,
                            "result": str(path),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return results


def report(root: Path) -> Path:
    config = load_json(CONFIG_PATH)
    base = output_root(config, root)
    metric_paths = sorted((base / "a1").glob("*/metrics/*.json"))
    if not metric_paths:
        raise R6Error("no R6-A1 metrics are available")
    rows: list[dict[str, Any]] = []
    documents: list[tuple[Path, dict[str, Any]]] = []
    for path in metric_paths:
        document = load_json(path)
        documents.append((path, document))
        candidate = document["evaluation"]["candidate"]
        balanced = document["evaluation"]["handoff"]["balanced"]
        rows.append(
            {
                "representation": document["representation"],
                "query_context_ms": document["query_context_ms"],
                "enrollment_ms": document["enrollment_ms"],
                "aggregation": document["aggregation"],
                "threshold": document["development"]["selected_threshold"],
                "roc_auc": document["evaluation"]["diagnostics"]["frame_pooled_roc_auc"],
                "eer": document["evaluation"]["diagnostics"]["eer"],
                "candidate_r500": candidate["causal"]["500"]["recall"],
                "candidate_r1000": candidate["causal"]["1000"]["recall"],
                "candidate_r1500": candidate["causal"]["1500"]["recall"],
                "candidate_fp_h": candidate["false_events_per_hour"],
                "candidate_median_ms": candidate["latency"]["median_ms"],
                "candidate_p95_ms": candidate["latency"]["p95_ms"],
                "handoff_r1000": balanced["causal"]["1000"]["recall"],
                "handoff_r1500": balanced["causal"]["1500"]["recall"],
                "false_handoff_h": balanced["false_events_per_hour"],
                "handoff_median_ms": balanced["latency"]["median_ms"],
                "handoff_p95_ms": balanced["latency"]["p95_ms"],
                "unavailable_events": document["evaluation"].get(
                    "unavailable_or_abstained_event_count", 0
                ),
                "dominated_by_one_meeting": document["gate"].get("dominated_by_one_meeting", False),
                "gate": document["gate"]["decision"],
                "metrics_path": str(path),
            }
        )
    csv_path = base / "a1/per_configuration.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    per_meeting_rows: list[dict[str, Any]] = []
    for path, document in documents:
        views = {"candidate": document["evaluation"]["candidate"]}
        views.update(
            {
                f"handoff_{name}": metrics
                for name, metrics in document["evaluation"]["handoff"].items()
            }
        )
        for view, metrics in views.items():
            for session_id, values in sorted(metrics["per_meeting"].items()):
                per_meeting_rows.append(
                    {
                        "representation": document["representation"],
                        "query_context_ms": document["query_context_ms"],
                        "enrollment_ms": document["enrollment_ms"],
                        "aggregation": document["aggregation"],
                        "view": view,
                        "session_id": session_id,
                        "tp_1000": values["tp_1000"],
                        "fp": values["fp"],
                        "fn_1000": values["fn_1000"],
                        "predictions": values["predictions"],
                        "positives": values["positives"],
                        "evaluated_hours": values.get("evaluated_hours"),
                        "false_events_per_hour": values.get("false_events_per_hour"),
                        "metrics_path": str(path),
                    }
                )
    per_meeting_path = base / "a1/per_meeting.csv"
    with per_meeting_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_meeting_rows[0]))
        writer.writeheader()
        writer.writerows(per_meeting_rows)
    inventory = load_json(base / "protocol/inventory.json")
    configs_path = base / "a1/configs_used.json"
    write_json(
        configs_path,
        {
            "schema_version": 1,
            "artifact_role": "r6_a1_configs_used",
            "config": config,
            "config_sha256": sha256_file(CONFIG_PATH),
            "protocol_inventory_sha256": sha256_file(base / "protocol/inventory.json"),
            "metric_artifacts": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "code_sha256": document["provenance"]["code_sha256"],
                    "worker_job_id": document["provenance"]["worker_job_id"],
                }
                for path, document in documents
            ],
        },
    )
    report_path = base / "a1/REPORT.md"
    lines = [
        "# R6-A1 Fixed Oracle Enrollment Report",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "The threshold for every row was selected only on the disjoint natural-meeting development sessions. The evaluation meetings were opened only after each operating point was frozen.",
        "",
        f"Development inventory: {inventory['summary']['development_hours']:.3f} hours across 5 meetings.",
        f"Evaluation inventory: {inventory['summary']['evaluation_hours']:.3f} hours across 5 meetings.",
        "",
        "Rates use evaluated first-handoff stream hours, while the inventory hours below describe unique source audio.",
        "",
        "| Representation | Query | Enrollment | Aggregation | AUC | EER | Candidate R@500 | Candidate R@1000 | Candidate R@1500 | Candidate FP/h | Candidate latency median / p95 | Handoff R@1000 | Handoff R@1500 | False handoff/h | Dominated | Gate |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]

    def value(number: Any) -> str:
        return "" if number is None else f"{float(number):.4f}"

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["representation"]),
                    str(row["query_context_ms"]),
                    str(row["enrollment_ms"]),
                    str(row["aggregation"]),
                    value(row["roc_auc"]),
                    value(row["eer"]),
                    value(row["candidate_r500"]),
                    value(row["candidate_r1000"]),
                    value(row["candidate_r1500"]),
                    value(row["candidate_fp_h"]),
                    f"{value(row['candidate_median_ms'])} / {value(row['candidate_p95_ms'])}",
                    value(row["handoff_r1000"]),
                    value(row["handoff_r1500"]),
                    value(row["false_handoff_h"]),
                    str(row["dominated_by_one_meeting"]),
                    str(row["gate"]),
                ]
            )
            + " |"
        )
    decisions = defaultdict(list)
    for row in rows:
        decisions[row["representation"]].append(row["gate"])
    lines.extend(["", "## Gate summary", ""])
    for representation, gates in sorted(decisions.items()):
        decision = (
            "promote"
            if "promote" in gates
            else ("conditional" if "conditional" in gates else "stop")
        )
        lines.append(f"- {representation}: {decision}")
    lines.extend(["", "## Natural error strata", ""])
    for representation in sorted(decisions):
        candidates = [
            (path, document)
            for path, document in documents
            if document["representation"] == representation
            and "strata" in document["evaluation"]["candidate"]
        ]
        if not candidates:
            continue
        path, document = max(
            candidates,
            key=lambda item: (
                float(item[1]["evaluation"]["candidate"]["causal"]["1000"]["recall"] or 0.0),
                -float(item[1]["evaluation"]["candidate"]["false_events_per_hour"] or 1e18),
            ),
        )
        candidate = document["evaluation"]["candidate"]
        balanced = document["evaluation"]["handoff"]["balanced"]
        lines.extend(
            [
                f"### {representation}",
                "",
                f"Selected diagnostic configuration: `{path.stem}`.",
                "",
                "| Stratum | Positive units | Candidate R@1000 | Candidate predictions | Balanced handoff R@1000 |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        candidate_strata = candidate["strata"]["by_event_kind"]
        handoff_strata = balanced["strata"]["by_event_kind"]
        for stratum in sorted(set(candidate_strata) | set(handoff_strata)):
            candidate_row = candidate_strata.get(stratum, {})
            handoff_row = handoff_strata.get(stratum, {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        stratum,
                        str(candidate_row.get("positive_count", 0)),
                        value(candidate_row.get("recall_1000")),
                        str(candidate_row.get("prediction_count", 0)),
                        value(handoff_row.get("recall_1000")),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "| Return window | Backchannel units | Balanced handoff rejection rate |",
                "| ---: | ---: | ---: |",
            ]
        )
        for window, values in balanced["strata"]["backchannel_return_windows_ms"].items():
            lines.append(
                f"| {window} | {values['unit_count']} | {value(values['rejection_rate'])} |"
            )
        sensitivity = candidate["meeting_sensitivity"]
        lines.extend(
            [
                "",
                f"Maximum share of Candidate R@1000 true positives from one meeting: {value(sensitivity['maximum_true_positive_share'])}; dominated: {sensitivity['dominated_by_one_meeting']}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundaries",
            "",
            "- Headline results are natural-only. Synthetic anchors remain secondary diagnostics and are not used for threshold selection or promotion.",
            "- Frame-pooled SAME/OTHER AUC is diagnostic; promotion is controlled by causal event recall in a plausible false-event region.",
            "- No A2 or B result is produced for a representation whose A1 gate is stop.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        base / "a1/summary.json",
        {
            "schema_version": 2,
            "artifact_role": "r6_a1_summary",
            "rows": rows,
            "inventory": inventory["summary"],
            "report_sha256": sha256_file(report_path),
            "per_configuration_sha256": sha256_file(csv_path),
            "per_meeting_sha256": sha256_file(per_meeting_path),
            "configs_used_sha256": sha256_file(configs_path),
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    return report_path


def smoke(root: Path, representation: str) -> None:
    config = load_json(CONFIG_PATH)
    validate_inputs(config, root)
    units = _units(root)
    unit = next(row for row in units if row["role"] == "development")
    source = _source_rows(input_paths(root))[str(unit["session_id"])]
    corpus = _corpus_root(root)
    waveform = _waveform_paths(input_paths(root), corpus)[str(source["waveform_id"])]
    import soundfile as sf

    audio, sample_rate = sf.read(str(waveform), dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE or audio.shape[1] != 1:
        raise R6Error("smoke waveform geometry differs")
    context_ms = int(config["representations"][representation]["query_context_ms"][0])
    frontier = int(unit["stream_start_sample"])
    window = np.ascontiguousarray(audio[frontier - context_ms * 16 : frontier, 0], dtype=np.float32)
    extractor = _make_extractor(root, representation)
    if representation == "m-l1":
        batch = extractor.extract([window], [frontier], layer_ids=("L1",))
        vector = _mean_pool_l2_batch(batch.layers["L1"], batch.valid_lengths["L1"])
    elif representation == "e-s3":
        batch = extractor.extract([window], [frontier], tap_ids=("S3",))
        vector = _mean_pool_l2_batch(batch.layers["S3"], batch.valid_lengths["S3"])
    else:
        batch = extractor.extract([window], [frontier], tap_ids=())
        vector = _normalise(np.asarray(batch.official_embedding, dtype=np.float32))
    if vector.shape[0] != 1 or not np.isfinite(vector).all():
        raise R6Error(f"smoke vector is invalid: {vector.shape}")
    print(
        json.dumps(
            {
                "representation": representation,
                "shape": list(vector.shape),
                "norm": float(np.linalg.norm(vector[0])),
            },
            sort_keys=True,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("extract", "score", "run", "report", "smoke"))
    parser.add_argument("--representation", choices=("m-l1", "e-s3", "e-final"))
    parser.add_argument("--role", choices=("development", "evaluation"))
    parser.add_argument("--context-ms", type=int)
    args = parser.parse_args(argv)
    root = cache_root()
    if args.action == "extract":
        if args.representation is None or args.role is None or args.context_ms is None:
            raise SystemExit("extract requires --representation, --role, and --context-ms")
        print(extract(root, args.representation, args.role, args.context_ms))
    elif args.action == "score":
        if args.representation is None:
            raise SystemExit("score requires --representation")
        for path in run_representation(root, args.representation):
            print(path)
    elif args.action == "run":
        representations = (
            [args.representation] if args.representation else ["m-l1", "e-s3", "e-final"]
        )
        for representation in representations:
            run_representation(root, representation)
        print(report(root))
    elif args.action == "report":
        print(report(root))
    else:
        if args.representation is None:
            raise SystemExit("smoke requires --representation")
        smoke(root, args.representation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
