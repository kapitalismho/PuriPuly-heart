from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiments.online_speaker_memory_handoff.protocol import (
    SpeakerRegion,
    _corpus_root,
    _regions_for_source,
    _source_rows,
    _waveform_paths,
    input_paths,
    read_jsonl,
    sha256_file,
    validate_inputs,
    write_json,
    write_jsonl,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "r7"
    / "eres_candidate_relation_verifier.json"
)
CODE_PATH = Path(__file__).resolve()
SAMPLE_RATE = 16000
HOP_SAMPLES = 1600
WINDOW_SAMPLES = 8000
SEQUENCE_OFFSETS_MS = tuple(range(-500, 1001, 100))
FEATURE_SCHEMA_VERSION = 1


class R7Error(RuntimeError):
    pass


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def config() -> dict[str, Any]:
    return load_json(CONFIG_PATH)


def cache_root() -> Path:
    value = os.environ.get("SRSCD_CACHE_ROOT")
    if not value:
        raise R7Error("SRSCD_CACHE_ROOT is required")
    root = Path(value).resolve()
    if not root.is_absolute() or root == REPOSITORY_ROOT or REPOSITORY_ROOT in root.parents:
        raise R7Error("SRSCD_CACHE_ROOT must be an absolute path outside the repository")
    return root


def output_root(root: Path) -> Path:
    return root / str(config()["output_relative_path"])


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "dirty_paths": status}


def _ceil_grid(value: int) -> int:
    return ((value + HOP_SAMPLES - 1) // HOP_SAMPLES) * HOP_SAMPLES


def _event_rows(regions: Sequence[SpeakerRegion]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not regions:
        return events
    last_active: frozenset[str] | None = None
    excluded = bool(regions[0].ambiguous)
    if not excluded and regions[0].speakers:
        last_active = regions[0].speakers
    gap_pending = False
    for previous, current in zip(regions[:-1], regions[1:], strict=True):
        if current.ambiguous:
            excluded = True
            last_active = None
            gap_pending = False
            continue
        if excluded:
            excluded = False
            if current.speakers:
                last_active = current.speakers
            gap_pending = False
            continue
        if current.speakers == previous.speakers:
            continue
        if not current.speakers:
            if last_active is not None:
                gap_pending = True
            continue
        if last_active is None:
            last_active = current.speakers
            gap_pending = False
            continue
        transition_kind = None
        if gap_pending:
            if current.speakers != last_active:
                transition_kind = "gap_speaker_change"
        elif current.speakers - last_active:
            transition_kind = (
                "interruption_onset"
                if current.speakers & last_active
                else "clean_handoff"
            )
        if transition_kind is not None:
            events.append(
                {
                    "sample": int(current.start_sample),
                    "speakers": sorted(current.speakers - last_active),
                    "active_speakers": sorted(current.speakers),
                    "ambiguous": False,
                    "overlap_onset": transition_kind == "interruption_onset",
                    "silence_gap": transition_kind == "gap_speaker_change",
                    "transition_kind": transition_kind,
                }
            )
        last_active = current.speakers
        gap_pending = False
    for index, event in enumerate(events):
        next_sample = events[index + 1]["sample"] if index + 1 < len(events) else None
        event["short_backchannel_or_return"] = (
            next_sample is not None and next_sample - event["sample"] <= 24000
        )
        if event["overlap_onset"]:
            event["stratum"] = "overlap_onset"
        elif event["silence_gap"]:
            event["stratum"] = "silence_gap_change"
        elif event["short_backchannel_or_return"]:
            event["stratum"] = "short_backchannel_or_return"
        else:
            event["stratum"] = "clean_change"
    return events


def prepare_inventory(root: Path) -> Path:
    cfg = config()
    validate_inputs(cfg, root)
    if set(cfg["development_sessions"]) & set(cfg["evaluation_sessions"]):
        raise R7Error("development and evaluation sessions overlap")
    paths = input_paths(root)
    sources = _source_rows(paths)
    corpus = _corpus_root(root)
    waveforms = _waveform_paths(paths, corpus)
    rows: list[dict[str, Any]] = []
    for role, session_ids in (
        ("development", cfg["development_sessions"]),
        ("evaluation", cfg["evaluation_sessions"]),
    ):
        for session_id in session_ids:
            source = sources.get(session_id)
            if source is None:
                raise R7Error(f"source metadata is missing: {session_id}")
            waveform = waveforms.get(str(source["waveform_id"]))
            if waveform is None or not waveform.is_file():
                raise R7Error(f"waveform is missing: {session_id}")
            regions = _regions_for_source(source, corpus)
            if not regions or regions[0].start_sample != 0:
                raise R7Error(f"speaker regions are incomplete: {session_id}")
            start = int(source["eligible_start_sample"])
            end = int(source["eligible_end_sample"])
            events = [
                event
                for event in _event_rows(regions)
                if start <= int(event["sample"]) < end
            ]
            rows.append(
                {
                    "session_id": session_id,
                    "role": role,
                    "corpus": source["corpus"],
                    "language": source["language"],
                    "waveform_id": source["waveform_id"],
                    "waveform_path": str(waveform),
                    "eligible_start_sample": start,
                    "eligible_end_sample": end,
                    "eligible_hours": (end - start) / SAMPLE_RATE / 3600.0,
                    "annotation_sha256": source["annotation_sha256"],
                    "events": events,
                    "event_count": len(events),
                }
            )
    result = {
        "schema_version": 1,
        "experiment_id": cfg["experiment_id"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "code_sha256": sha256_file(CODE_PATH),
        "corpus_root": str(corpus),
        "sessions": rows,
        "summary": {
            "development_hours": sum(
                row["eligible_hours"] for row in rows if row["role"] == "development"
            ),
            "evaluation_hours": sum(
                row["eligible_hours"] for row in rows if row["role"] == "evaluation"
            ),
            "development_events": sum(
                row["event_count"] for row in rows if row["role"] == "development"
            ),
            "evaluation_events": sum(
                row["event_count"] for row in rows if row["role"] == "evaluation"
            ),
        },
        "git": _git_state(),
    }
    path = output_root(root) / "inventory.json"
    write_json(path, result)
    write_json(output_root(root) / "config_used.json", cfg)
    return path


def _feature_paths(root: Path, role: str) -> dict[str, Path]:
    directory = output_root(root) / "features" / role
    return {
        "vectors": directory / "dense_500ms.npy",
        "audio": directory / "dense_audio.npy",
        "index": directory / "dense_index.jsonl",
        "manifest": directory / "dense_manifest.json",
        "candidates": directory / "candidates.jsonl",
        "relations": directory / "relation_features.npz",
        "relation_manifest": directory / "relation_manifest.json",
    }


def _make_extractor(root: Path):
    from experiments.online_speaker_memory_handoff.a1 import _make_extractor

    return _make_extractor(root, "e-final")


def _r6_reuse(root: Path, role: str) -> tuple[np.ndarray | None, dict[tuple[str, int], int], dict[str, Any] | None]:
    directory = (
        root
        / "results/r6/online_speaker_memory_handoff_v1/features/e-final"
        / role
    )
    vectors_path = directory / "context_500ms.npy"
    index_path = directory / "context_500ms.jsonl"
    manifest_path = directory / "context_500ms.manifest.json"
    if not vectors_path.is_file() or not index_path.is_file() or not manifest_path.is_file():
        return None, {}, None
    manifest = load_json(manifest_path)
    if manifest.get("vectors_sha256") != sha256_file(vectors_path) or manifest.get(
        "index_sha256"
    ) != sha256_file(index_path):
        raise R7Error(f"R6 reusable feature identity differs: {manifest_path}")
    vectors = np.load(vectors_path, mmap_mode="r")
    rows = read_jsonl(index_path)
    if vectors.shape != (int(manifest["row_count"]), 192) or len(rows) != vectors.shape[0]:
        raise R7Error(f"R6 reusable feature geometry differs: {manifest_path}")
    lookup = {
        (str(row["session_id"]), int(row["frontier_sample"])): int(row["row"])
        for row in rows
    }
    return vectors, lookup, {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "vectors_sha256": manifest["vectors_sha256"],
        "index_sha256": manifest["index_sha256"],
        "row_count": int(manifest["row_count"]),
    }


def _audio_features(window: np.ndarray) -> tuple[float, float]:
    rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))
    frame = 320
    trimmed = window[: (window.size // frame) * frame]
    frames = trimmed.reshape(-1, frame)
    frame_rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    speech_fraction = float(np.mean(frame_rms >= 0.00316227766))
    return math.log(max(rms, 1e-8)), speech_fraction


def extract_dense(root: Path, role: str) -> Path:
    if role not in {"development", "evaluation"}:
        raise R7Error(f"invalid role: {role}")
    inventory_path = output_root(root) / "inventory.json"
    if not inventory_path.is_file():
        prepare_inventory(root)
    inventory = load_json(inventory_path)
    paths = _feature_paths(root, role)
    if all(paths[name].is_file() for name in ("vectors", "audio", "index", "manifest")):
        manifest = load_json(paths["manifest"])
        if (
            manifest.get("vectors_sha256") == sha256_file(paths["vectors"])
            and manifest.get("audio_sha256") == sha256_file(paths["audio"])
            and manifest.get("index_sha256") == sha256_file(paths["index"])
        ):
            return paths["manifest"]
        raise R7Error(f"modified or incomplete dense feature cache: {paths['manifest']}")
    sessions = [row for row in inventory["sessions"] if row["role"] == role]
    frontiers_by_session: dict[str, list[int]] = {}
    for row in sessions:
        first = _ceil_grid(max(WINDOW_SAMPLES, int(row["eligible_start_sample"]) + WINDOW_SAMPLES))
        last = int(row["eligible_end_sample"])
        frontiers_by_session[row["session_id"]] = list(range(first, last + 1, HOP_SAMPLES))
    row_count = sum(len(values) for values in frontiers_by_session.values())
    if row_count == 0:
        raise R7Error(f"no dense windows for {role}")
    paths["vectors"].parent.mkdir(parents=True, exist_ok=True)
    vector_temp = paths["vectors"].with_suffix(".npy.tmp")
    audio_temp = paths["audio"].with_suffix(".npy.tmp")
    index_temp = paths["index"].with_suffix(".jsonl.tmp")
    vectors = np.lib.format.open_memmap(
        vector_temp, mode="w+", dtype=np.float32, shape=(row_count, 192)
    )
    audio_features = np.lib.format.open_memmap(
        audio_temp, mode="w+", dtype=np.float32, shape=(row_count, 2)
    )
    reused_vectors, reused_lookup, reused_identity = _r6_reuse(root, role)
    extractor = None
    import soundfile as sf

    offset = 0
    reused_count = 0
    inferred_count = 0
    index_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for session in sessions:
        session_id = str(session["session_id"])
        waveform, sample_rate = sf.read(
            str(session["waveform_path"]), dtype="float32", always_2d=True
        )
        if sample_rate != SAMPLE_RATE or waveform.shape[1] != 1:
            raise R7Error(f"waveform geometry differs: {session_id}")
        mono = np.ascontiguousarray(waveform[:, 0], dtype=np.float32)
        frontiers = frontiers_by_session[session_id]
        for batch_start in range(0, len(frontiers), 16):
            batch_frontiers = frontiers[batch_start : batch_start + 16]
            windows = [
                np.ascontiguousarray(mono[frontier - WINDOW_SAMPLES : frontier], dtype=np.float32)
                for frontier in batch_frontiers
            ]
            if any(window.shape != (WINDOW_SAMPLES,) for window in windows):
                raise R7Error(f"window geometry differs: {session_id}")
            values = np.empty((len(windows), 192), dtype=np.float32)
            missing_indices: list[int] = []
            for local_index, frontier in enumerate(batch_frontiers):
                reused_row = reused_lookup.get((session_id, frontier))
                if reused_vectors is None or reused_row is None:
                    missing_indices.append(local_index)
                    continue
                values[local_index] = np.asarray(reused_vectors[reused_row], dtype=np.float32)
                reused_count += 1
            if missing_indices:
                if extractor is None:
                    extractor = _make_extractor(root)
                missing_windows = [windows[index] for index in missing_indices]
                missing_frontiers = [batch_frontiers[index] for index in missing_indices]
                batch = extractor.extract(missing_windows, missing_frontiers, tap_ids=())
                inferred = np.asarray(batch.official_embedding, dtype=np.float32)
                inferred_norms = np.linalg.norm(inferred, axis=1)
                if inferred.shape != (len(missing_windows), 192) or not np.isfinite(
                    inferred
                ).all() or np.any(inferred_norms <= 0):
                    raise R7Error(f"invalid ERes embeddings: {session_id}")
                inferred = inferred / inferred_norms[:, None]
                for inferred_index, local_index in enumerate(missing_indices):
                    values[local_index] = inferred[inferred_index]
                inferred_count += len(missing_indices)
            norms = np.linalg.norm(values, axis=1)
            if not np.isfinite(values).all() or np.any(np.abs(norms - 1.0) > 1e-3):
                raise R7Error(f"invalid normalized ERes embeddings: {session_id}")
            vectors[offset : offset + len(windows)] = values
            for local_index, (frontier, window) in enumerate(zip(batch_frontiers, windows, strict=True)):
                audio_features[offset + local_index] = _audio_features(window)
                index_rows.append(
                    {
                        "session_id": session_id,
                        "frontier_sample": frontier,
                        "window_start_sample": frontier - WINDOW_SAMPLES,
                        "row": offset + local_index,
                    }
                )
            offset += len(windows)
        vectors.flush()
        audio_features.flush()
        print(
            json.dumps(
                {
                    "stage": "r7_extract",
                    "role": role,
                    "session_id": session_id,
                    "completed_rows": offset,
                    "total_rows": row_count,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    del vectors
    del audio_features
    write_jsonl(index_temp, index_rows)
    vector_temp.replace(paths["vectors"])
    audio_temp.replace(paths["audio"])
    index_temp.replace(paths["index"])
    elapsed = time.perf_counter() - started
    source_seconds = sum(float(row["eligible_hours"]) * 3600.0 for row in sessions)
    write_json(
        paths["manifest"],
        {
            "schema_version": 1,
            "role": role,
            "row_count": row_count,
            "dimension": 192,
            "hop_ms": 100,
            "window_ms": 500,
            "vectors_sha256": sha256_file(paths["vectors"]),
            "audio_sha256": sha256_file(paths["audio"]),
            "index_sha256": sha256_file(paths["index"]),
            "wall_seconds": elapsed,
            "source_seconds": source_seconds,
            "source_rtf": elapsed / source_seconds,
            "reused_r6_window_count": reused_count,
            "new_inference_window_count": inferred_count,
            "r6_reuse_identity": reused_identity,
            "hardware": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
                "backend": "cpu",
            },
            "checkpoint_identity": load_json(root / "manifests/r1_model_acquisition.json"),
            "worker_job_id": os.environ.get("ORCA_WORKER_JOB_ID"),
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    return paths["manifest"]


class DenseStore:
    def __init__(self, vectors: np.ndarray, audio: np.ndarray, rows: Sequence[dict[str, Any]]) -> None:
        self.vectors = vectors
        self.audio = audio
        self.lookup = {
            (str(row["session_id"]), int(row["frontier_sample"])): int(row["row"])
            for row in rows
        }

    def exact(self, session_id: str, frontier: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.lookup.get((session_id, frontier))
        if row is None:
            raise R7Error(f"dense feature is missing: {session_id}:{frontier}")
        return (
            np.asarray(self.vectors[row], dtype=np.float32),
            np.asarray(self.audio[row], dtype=np.float32),
        )


def _load_dense(root: Path, role: str) -> DenseStore:
    paths = _feature_paths(root, role)
    if not paths["manifest"].is_file():
        raise R7Error(f"dense extraction is missing: {paths['manifest']}")
    manifest = load_json(paths["manifest"])
    vectors = np.load(paths["vectors"], mmap_mode="r")
    audio = np.load(paths["audio"], mmap_mode="r")
    rows = read_jsonl(paths["index"])
    if vectors.shape != (int(manifest["row_count"]), 192) or audio.shape != (
        int(manifest["row_count"]),
        2,
    ) or len(rows) != int(manifest["row_count"]):
        raise R7Error(f"dense feature geometry differs: {role}")
    return DenseStore(vectors, audio, rows)


def _nearest_event(events: Sequence[dict[str, Any]], boundary: int) -> tuple[dict[str, Any] | None, int]:
    if not events:
        return None, 1 << 60
    event = min(events, key=lambda row: abs(int(row["sample"]) - boundary))
    return event, abs(int(event["sample"]) - boundary)


def _relation_feature(
    embeddings: np.ndarray,
    audio: np.ndarray,
    deadline_ms: int,
) -> np.ndarray:
    offsets = np.asarray(SEQUENCE_OFFSETS_MS, dtype=np.int64)
    usable = offsets <= (700 if deadline_ms == 750 else deadline_ms)
    masked = embeddings.copy()
    masked[~usable] = 0.0
    relation = np.clip(masked @ masked.T, -1.0, 1.0)
    relation[~np.outer(usable, usable)] = 0.0
    adjacent = np.zeros(len(offsets) - 1, dtype=np.float32)
    adjacent_usable = usable[:-1] & usable[1:]
    adjacent[adjacent_usable] = 1.0 - np.sum(
        embeddings[:-1][adjacent_usable] * embeddings[1:][adjacent_usable], axis=1
    )
    left = np.flatnonzero(usable & (offsets <= 0))
    right = np.flatnonzero(usable & (offsets >= 500))
    cross_values = relation[np.ix_(left, right)].reshape(-1)
    left_values = relation[np.ix_(left, left)][np.triu_indices(len(left), 1)]
    right_values = relation[np.ix_(right, right)][np.triu_indices(len(right), 1)]

    def summary(values: np.ndarray) -> list[float]:
        if values.size == 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [
            float(np.mean(values)),
            float(np.std(values)),
            float(np.min(values)),
            float(np.max(values)),
        ]

    summaries = np.asarray(
        summary(left_values)
        + summary(right_values)
        + summary(cross_values)
        + [
            float(adjacent[5]),
            float(np.max(adjacent[5:10])),
            float(np.mean(adjacent[5:10])),
            float(np.std(adjacent[5:10])),
        ],
        dtype=np.float32,
    )
    audio_masked = audio.copy()
    audio_masked[~usable] = 0.0
    return np.concatenate(
        [
            relation.reshape(-1).astype(np.float32),
            adjacent,
            audio_masked[:, 0],
            audio_masked[:, 1],
            usable.astype(np.float32),
            summaries,
        ]
    )


def build_relations(root: Path, role: str) -> Path:
    paths = _feature_paths(root, role)
    if paths["relation_manifest"].is_file() and paths["relations"].is_file() and paths[
        "candidates"
    ].is_file():
        manifest = load_json(paths["relation_manifest"])
        if (
            manifest.get("relations_sha256") == sha256_file(paths["relations"])
            and manifest.get("candidates_sha256") == sha256_file(paths["candidates"])
            and manifest.get("code_sha256") == sha256_file(CODE_PATH)
        ):
            return paths["relation_manifest"]
    inventory = load_json(output_root(root) / "inventory.json")
    store = _load_dense(root, role)
    cfg = config()
    candidates: list[dict[str, Any]] = []
    feature_rows: dict[int, list[np.ndarray]] = {
        int(deadline): [] for deadline in cfg["lookahead_ms"]
    }
    for session in inventory["sessions"]:
        if session["role"] != role:
            continue
        session_id = str(session["session_id"])
        start = _ceil_grid(int(session["eligible_start_sample"]) + SAMPLE_RATE)
        end = int(session["eligible_end_sample"]) - SAMPLE_RATE
        events = session["events"]
        positive_counts: dict[int, int] = defaultdict(int)
        pending: list[tuple[dict[str, Any], dict[int, np.ndarray]]] = []
        for boundary in range(start, end + 1, HOP_SAMPLES):
            left, _ = store.exact(session_id, boundary)
            right, _ = store.exact(session_id, boundary + WINDOW_SAMPLES)
            raw_change = 1.0 - float(np.dot(left, right))
            if raw_change <= float(cfg["candidate_change_threshold"]):
                continue
            embeddings: list[np.ndarray] = []
            audio_rows: list[np.ndarray] = []
            for offset_ms in SEQUENCE_OFFSETS_MS:
                vector, audio = store.exact(session_id, boundary + offset_ms * 16)
                embeddings.append(vector)
                audio_rows.append(audio)
            embedding_matrix = np.stack(embeddings)
            audio_matrix = np.stack(audio_rows)
            nearest, distance = _nearest_event(events, boundary)
            if nearest is not None and distance <= int(cfg["positive_radius_ms"]) * 16:
                label = 1
                event_sample = int(nearest["sample"])
                positive_counts[event_sample] += 1
                stratum = str(nearest["stratum"])
                ambiguous = bool(nearest["ambiguous"])
            elif nearest is None or distance > int(cfg["negative_radius_ms"]) * 16:
                label = 0
                event_sample = None
                stratum = "same_speaker_false_candidate"
                ambiguous = False
            else:
                label = -1
                event_sample = int(nearest["sample"]) if nearest is not None else None
                stratum = "ambiguous_distance"
                ambiguous = True
            row = {
                "candidate_id": f"{session_id}:{boundary}",
                "session_id": session_id,
                "role": role,
                "boundary_sample": boundary,
                "first_available_sample": boundary + WINDOW_SAMPLES,
                "raw_change_score": raw_change,
                "label": label,
                "event_sample": event_sample,
                "event_distance_samples": distance if nearest is not None else None,
                "stratum": stratum,
                "ambiguous": ambiguous,
            }
            pending.append(
                (
                    row,
                    {
                        int(deadline): _relation_feature(
                            embedding_matrix, audio_matrix, int(deadline)
                        )
                        for deadline in cfg["lookahead_ms"]
                    },
                )
            )
        for row, features in pending:
            if row["label"] == 1:
                row["sample_weight"] = 1.0 / positive_counts[int(row["event_sample"])]
            else:
                row["sample_weight"] = 1.0
            candidates.append(row)
            for deadline, values in features.items():
                feature_rows[deadline].append(values)
    if not candidates:
        raise R7Error(f"no candidates generated for {role}")
    arrays = {
        f"x_{deadline}": np.stack(values).astype(np.float32)
        for deadline, values in feature_rows.items()
    }
    arrays["labels"] = np.asarray([row["label"] for row in candidates], dtype=np.int8)
    arrays["weights"] = np.asarray([row["sample_weight"] for row in candidates], dtype=np.float32)
    paths["relations"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths["relations"], **arrays)
    write_jsonl(paths["candidates"], candidates)
    write_json(
        paths["relation_manifest"],
        {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "role": role,
            "candidate_count": len(candidates),
            "positive_count": sum(row["label"] == 1 for row in candidates),
            "negative_count": sum(row["label"] == 0 for row in candidates),
            "ambiguous_count": sum(row["label"] == -1 for row in candidates),
            "feature_dimension": int(next(iter(arrays.values())).shape[1]),
            "lookahead_ms": cfg["lookahead_ms"],
            "sequence_offsets_ms": SEQUENCE_OFFSETS_MS,
            "relations_sha256": sha256_file(paths["relations"]),
            "candidates_sha256": sha256_file(paths["candidates"]),
            "code_sha256": sha256_file(CODE_PATH),
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    return paths["relation_manifest"]


def _balanced_weights(labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result = weights.astype(np.float64, copy=True)
    positive = labels == 1
    negative = labels == 0
    positive_total = float(result[positive].sum())
    negative_total = float(result[negative].sum())
    if positive_total <= 0 or negative_total <= 0:
        raise R7Error("both training classes are required")
    result[positive] *= 0.5 / positive_total
    result[negative] *= 0.5 / negative_total
    result *= len(result)
    return result.astype(np.float32)


def _standardize_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64)
    scale = values.std(axis=0, dtype=np.float64)
    scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _standardize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.asarray((values - mean) / scale, dtype=np.float32)


def _fit_linear(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, cfg: dict[str, Any]
) -> tuple[np.ndarray, float]:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        C=float(cfg["linear"]["c"]),
        max_iter=int(cfg["linear"]["max_iter"]),
        solver="lbfgs",
        random_state=1701,
    )
    model.fit(x, y, sample_weight=weights)
    return np.asarray(model.coef_[0], dtype=np.float32), float(model.intercept_[0])


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _linear_predict(x: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    return _sigmoid(np.asarray(x @ coef + intercept, dtype=np.float64))


def _mlp_module(input_dim: int, hidden: int, dropout: float):
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden, 1),
    )


def _fit_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weights: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    maximum_epochs: int | None = None,
) -> tuple[dict[str, Any], int]:
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    spec = cfg["mlp"]
    model = _mlp_module(train_x.shape[1], int(spec["hidden_width"]), float(spec["dropout"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    generator = torch.Generator().manual_seed(seed)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(train_y.astype(np.float32)),
        torch.from_numpy(train_weights.astype(np.float32)),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(spec["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    epochs = int(maximum_epochs or spec["maximum_epochs"])
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y, batch_w in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x).squeeze(1)
            loss = (loss_fn(logits, batch_y) * batch_w).mean()
            loss.backward()
            optimizer.step()
        if validation is None:
            best_epoch = epoch + 1
            continue
        validation_x, validation_y, validation_weights = validation
        model.eval()
        with torch.inference_mode():
            logits = model(torch.from_numpy(validation_x)).squeeze(1)
            losses = loss_fn(logits, torch.from_numpy(validation_y.astype(np.float32)))
            value = float(
                (losses * torch.from_numpy(validation_weights.astype(np.float32))).sum()
                / max(float(validation_weights.sum()), 1e-12)
            )
        if value < best_loss - 1e-5:
            best_loss = value
            best_epoch = epoch + 1
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(spec["patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}, best_epoch


def _mlp_predict(
    x: np.ndarray, state: dict[str, Any], cfg: dict[str, Any], batch_size: int = 4096
) -> np.ndarray:
    import torch

    model = _mlp_module(x.shape[1], int(cfg["mlp"]["hidden_width"]), float(cfg["mlp"]["dropout"]))
    model.load_state_dict(state)
    model.eval()
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            logits = model(torch.from_numpy(x[start : start + batch_size])).squeeze(1)
            values.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def _fit_calibration(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression

    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1000.0, max_iter=300, solver="lbfgs")
    model.fit(logits, labels, sample_weight=weights)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def _apply_calibration(scores: np.ndarray, calibration: tuple[float, float]) -> np.ndarray:
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    return _sigmoid(calibration[0] * logits + calibration[1])


def _local_events(
    rows: Sequence[dict[str, Any]], scores: np.ndarray, threshold: float
) -> list[tuple[str, int]]:
    cfg = config()
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row, score in zip(rows, scores, strict=True):
        if float(score) >= threshold:
            grouped[str(row["session_id"])].append((int(row["boundary_sample"]), float(score)))
    results: list[tuple[str, int]] = []
    radius = int(cfg["duplicate_suppression_ms"]) * 16
    for session_id, pairs in grouped.items():
        pairs.sort()
        maxima: list[tuple[int, float]] = []
        for index, pair in enumerate(pairs):
            previous = pairs[index - 1][1] if index > 0 and pair[0] - pairs[index - 1][0] <= HOP_SAMPLES else -math.inf
            following = pairs[index + 1][1] if index + 1 < len(pairs) and pairs[index + 1][0] - pair[0] <= HOP_SAMPLES else -math.inf
            if pair[1] >= previous and pair[1] >= following:
                maxima.append(pair)
        accepted: list[tuple[int, float]] = []
        for pair in sorted(maxima, key=lambda value: (-value[1], value[0])):
            if all(abs(pair[0] - other[0]) > radius for other in accepted):
                accepted.append(pair)
        results.extend((session_id, sample) for sample, _ in accepted)
    return sorted(results)


def _matches(
    predictions: Sequence[int], references: Sequence[int], tolerance_samples: int
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    pairs = sorted(
        (
            (abs(prediction - reference), prediction_index, reference_index)
            for prediction_index, prediction in enumerate(predictions)
            for reference_index, reference in enumerate(references)
            if abs(prediction - reference) <= tolerance_samples
        ),
        key=lambda row: (row[0], predictions[row[1]], references[row[2]]),
    )
    used_predictions: set[int] = set()
    used_references: set[int] = set()
    matched: list[tuple[int, int]] = []
    for _, prediction_index, reference_index in pairs:
        if prediction_index in used_predictions or reference_index in used_references:
            continue
        used_predictions.add(prediction_index)
        used_references.add(reference_index)
        matched.append((predictions[prediction_index], references[reference_index]))
    false_predictions = [
        value for index, value in enumerate(predictions) if index not in used_predictions
    ]
    misses = [value for index, value in enumerate(references) if index not in used_references]
    return matched, false_predictions, misses


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _metrics_for_events(
    rows: Sequence[dict[str, Any]],
    predictions: Sequence[tuple[str, int]],
    inventory: dict[str, Any],
    lookahead_ms: int,
) -> dict[str, Any]:
    role = str(rows[0]["role"])
    session_rows = [row for row in inventory["sessions"] if row["role"] == role]
    references_by_session = {
        str(row["session_id"]): [int(event["sample"]) for event in row["events"]]
        for row in session_rows
    }
    predictions_by_session: dict[str, list[int]] = defaultdict(list)
    for session_id, prediction in predictions:
        predictions_by_session[session_id].append(prediction)
    exposure_hours = sum(float(row["eligible_hours"]) for row in session_rows)
    tolerances: dict[str, Any] = {}
    primary_matches: list[tuple[str, int, int]] = []
    primary_false: list[tuple[str, int]] = []
    primary_misses: list[tuple[str, int]] = []
    per_meeting: dict[str, Any] = {}
    for tolerance_ms in (100, 250, 500):
        all_matches: list[tuple[str, int, int]] = []
        all_false: list[tuple[str, int]] = []
        all_misses: list[tuple[str, int]] = []
        for session in session_rows:
            session_id = str(session["session_id"])
            matched, false, misses = _matches(
                predictions_by_session.get(session_id, []),
                references_by_session[session_id],
                tolerance_ms * 16,
            )
            all_matches.extend((session_id, prediction, reference) for prediction, reference in matched)
            all_false.extend((session_id, prediction) for prediction in false)
            all_misses.extend((session_id, reference) for reference in misses)
            if tolerance_ms == 250:
                per_meeting[session_id] = {
                    "reference_count": len(references_by_session[session_id]),
                    "prediction_count": len(predictions_by_session.get(session_id, [])),
                    "true_positive_count": len(matched),
                    "false_event_count": len(false),
                    "miss_count": len(misses),
                    "recall": _safe_ratio(len(matched), len(references_by_session[session_id])),
                    "false_events_per_hour": len(false) / float(session["eligible_hours"]),
                }
        precision = _safe_ratio(len(all_matches), len(all_matches) + len(all_false))
        recall = _safe_ratio(len(all_matches), len(all_matches) + len(all_misses))
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else None
        )
        tolerances[str(tolerance_ms)] = {
            "true_positive_count": len(all_matches),
            "false_event_count": len(all_false),
            "miss_count": len(all_misses),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_events_per_hour": len(all_false) / exposure_hours,
        }
        if tolerance_ms == 250:
            primary_matches = all_matches
            primary_false = all_false
            primary_misses = all_misses
    errors_ms = [
        (prediction - reference) / 16.0 for _, prediction, reference in primary_matches
    ]
    availability = [float(lookahead_ms)] * len(predictions)
    candidate_strata = {
        (str(row["session_id"]), int(row["boundary_sample"])): str(row["stratum"])
        for row in rows
    }
    event_strata = {
        (str(session["session_id"]), int(event["sample"])): str(event["stratum"])
        for session in session_rows
        for event in session["events"]
    }
    false_strata: dict[str, int] = defaultdict(int)
    miss_strata: dict[str, int] = defaultdict(int)
    for key in primary_false:
        false_strata[candidate_strata.get(key, "unclassified_false_candidate")] += 1
    for key in primary_misses:
        miss_strata[event_strata.get(key, "unclassified_change")] += 1
    return {
        "lookahead_ms": lookahead_ms,
        "exposure_hours": exposure_hours,
        "reference_count": sum(len(values) for values in references_by_session.values()),
        "prediction_count": len(predictions),
        "tolerances": tolerances,
        "availability_latency_ms": {
            "median": float(np.median(availability)) if availability else None,
            "p90": float(np.quantile(availability, 0.90)) if availability else None,
            "p95": float(np.quantile(availability, 0.95)) if availability else None,
        },
        "signed_localization_error_ms": {
            "mean": float(np.mean(errors_ms)) if errors_ms else None,
            "median": float(np.median(errors_ms)) if errors_ms else None,
            "p10": float(np.quantile(errors_ms, 0.10)) if errors_ms else None,
            "p90": float(np.quantile(errors_ms, 0.90)) if errors_ms else None,
        },
        "per_meeting": per_meeting,
        "error_strata": {
            "false_events": dict(sorted(false_strata.items())),
            "misses": dict(sorted(miss_strata.items())),
        },
        "matched_pairs": primary_matches,
        "false_event_samples": primary_false,
        "miss_samples": primary_misses,
    }


def _threshold_grid(scores: np.ndarray) -> list[float]:
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, 201))
    values = {round(float(value), 9) for value in quantiles}
    ranked = np.sort(np.unique(scores))[::-1]
    values.update(round(float(value), 9) for value in ranked[:512])
    values.add(float(np.nextafter(float(np.max(scores)), math.inf)))
    return sorted(values, reverse=True)


def _curve(
    rows: Sequence[dict[str, Any]],
    scores: np.ndarray,
    inventory: dict[str, Any],
    lookahead_ms: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for threshold in _threshold_grid(scores):
        events = _local_events(rows, scores, threshold)
        metrics = _metrics_for_events(rows, events, inventory, lookahead_ms)
        primary = metrics["tolerances"]["250"]
        result.append(
            {
                "threshold": threshold,
                "recall_100": metrics["tolerances"]["100"]["recall"],
                "recall_250": primary["recall"],
                "recall_500": metrics["tolerances"]["500"]["recall"],
                "precision_250": primary["precision"],
                "f1_250": primary["f1"],
                "false_event_count": primary["false_event_count"],
                "false_events_per_hour": primary["false_events_per_hour"],
                "prediction_count": metrics["prediction_count"],
            }
        )
    return result


def _select_targets(curve: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target in config()["development_false_event_targets_per_hour"]:
        eligible = [row for row in curve if float(row["false_events_per_hour"]) <= float(target)]
        if not eligible:
            result[str(target)] = None
            continue
        result[str(target)] = max(
            eligible,
            key=lambda row: (
                float(row["recall_250"] or 0.0),
                float(row["recall_500"] or 0.0),
                -float(row["false_events_per_hour"]),
                -float(row["threshold"]),
            ),
        )
    return result


def _raw_metrics(rows: Sequence[dict[str, Any]], inventory: dict[str, Any]) -> dict[str, Any]:
    predictions = [
        (str(row["session_id"]), int(row["boundary_sample"])) for row in rows
    ]
    return _metrics_for_events(rows, predictions, inventory, 500)


def develop(root: Path) -> Path:
    build_relations(root, "development")
    paths = _feature_paths(root, "development")
    rows = read_jsonl(paths["candidates"])
    data = np.load(paths["relations"])
    labels_all = np.asarray(data["labels"], dtype=np.int64)
    weights_all = np.asarray(data["weights"], dtype=np.float32)
    eligible = labels_all >= 0
    labels = labels_all[eligible]
    weights = _balanced_weights(labels, weights_all[eligible])
    meetings = sorted({str(row["session_id"]) for row in rows})
    if len(meetings) != 5:
        raise R7Error(f"meeting-held development requires five meetings: {meetings}")
    cfg = config()
    inventory = load_json(output_root(root) / "inventory.json")
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "feature_dimension": int(data["x_500"].shape[1]),
        "models": {},
        "config_sha256": sha256_file(CONFIG_PATH),
        "code_sha256": sha256_file(CODE_PATH),
    }
    metrics_document: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "raw_candidates": _raw_metrics(rows, inventory),
        "meeting_folds": meetings,
        "methods": {},
    }
    started = time.perf_counter()
    for deadline in cfg["lookahead_ms"]:
        x_all = np.asarray(data[f"x_{deadline}"], dtype=np.float32)
        x = x_all[eligible]
        oof_linear = np.zeros(len(x_all), dtype=np.float64)
        oof_mlp = np.zeros(len(x_all), dtype=np.float64)
        fold_epochs: list[int] = []
        session_ids_all = np.asarray([str(row["session_id"]) for row in rows])
        for held_out in meetings:
            held_out_mask = session_ids_all == held_out
            training_mask = eligible & ~held_out_mask
            validation_mask = eligible & held_out_mask
            mean, scale = _standardize_fit(x_all[training_mask])
            train_x = _standardize(x_all[training_mask], mean, scale)
            validation_x = _standardize(x_all[validation_mask], mean, scale)
            held_out_x = _standardize(x_all[held_out_mask], mean, scale)
            train_weights = _balanced_weights(
                labels_all[training_mask], weights_all[training_mask]
            )
            validation_weights = _balanced_weights(
                labels_all[validation_mask], weights_all[validation_mask]
            )
            coef, intercept = _fit_linear(
                train_x, labels_all[training_mask], train_weights, cfg
            )
            oof_linear[held_out_mask] = _linear_predict(held_out_x, coef, intercept)
            held_out_seed_scores: list[np.ndarray] = []
            seed_epochs: list[int] = []
            for seed in cfg["mlp"]["seeds"]:
                state, best_epoch = _fit_mlp(
                    train_x,
                    labels_all[training_mask],
                    train_weights,
                    cfg,
                    int(seed),
                    (validation_x, labels_all[validation_mask], validation_weights),
                )
                held_out_seed_scores.append(_mlp_predict(held_out_x, state, cfg))
                seed_epochs.append(best_epoch)
            oof_mlp[held_out_mask] = np.mean(held_out_seed_scores, axis=0)
            fold_epochs.append(max(1, int(round(float(np.median(seed_epochs))))))
        mean, scale = _standardize_fit(x)
        standardized = _standardize(x, mean, scale)
        final_weights = _balanced_weights(labels, weights_all[eligible])
        coef, intercept = _fit_linear(standardized, labels, final_weights, cfg)
        mlp_epochs = max(1, int(round(float(np.median(fold_epochs)))))
        final_states: list[dict[str, Any]] = []
        for seed in cfg["mlp"]["seeds"]:
            state, _ = _fit_mlp(
                standardized,
                labels,
                final_weights,
                cfg,
                int(seed),
                None,
                maximum_epochs=mlp_epochs,
            )
            final_states.append(state)
        artifact["models"][str(deadline)] = {
            "mean": mean,
            "scale": scale,
            "linear_coef": coef,
            "linear_intercept": intercept,
            "linear_calibration": _fit_calibration(oof_linear[eligible], labels, weights),
            "mlp_states": final_states,
            "mlp_epochs": mlp_epochs,
            "mlp_calibration": _fit_calibration(oof_mlp[eligible], labels, weights),
        }
        for method, raw_scores, calibration in (
            (
                "linear",
                oof_linear,
                artifact["models"][str(deadline)]["linear_calibration"],
            ),
            ("mlp", oof_mlp, artifact["models"][str(deadline)]["mlp_calibration"]),
        ):
            calibrated = _apply_calibration(raw_scores, tuple(calibration))
            method_curve = _curve(rows, calibrated, inventory, int(deadline))
            metrics_document["methods"][f"{method}_{deadline}"] = {
                "method": method,
                "lookahead_ms": int(deadline),
                "curve": method_curve,
                "selected_operating_points": _select_targets(method_curve),
            }
        print(
            json.dumps(
                {
                    "stage": "r7_development",
                    "lookahead_ms": deadline,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    metrics_document["training_wall_seconds"] = time.perf_counter() - started
    metrics_document["worker_job_id"] = os.environ.get("ORCA_WORKER_JOB_ID")
    import torch

    model_path = output_root(root) / "relation_verifier.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, model_path)
    metrics_document["model_sha256"] = sha256_file(model_path)
    path = output_root(root) / "development_metrics.json"
    write_json(path, metrics_document)
    return path


def _load_artifact(path: Path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def evaluate(root: Path) -> Path:
    development_path = output_root(root) / "development_metrics.json"
    model_path = output_root(root) / "relation_verifier.pt"
    if not development_path.is_file() or not model_path.is_file():
        raise R7Error("development must finish before evaluation")
    development = load_json(development_path)
    artifact = _load_artifact(model_path)
    if artifact.get("config_sha256") != sha256_file(CONFIG_PATH) or artifact.get(
        "code_sha256"
    ) != sha256_file(CODE_PATH):
        raise R7Error("frozen development artifact differs from current code or config")
    build_relations(root, "evaluation")
    paths = _feature_paths(root, "evaluation")
    rows = read_jsonl(paths["candidates"])
    data = np.load(paths["relations"])
    inventory = load_json(output_root(root) / "inventory.json")
    predictions: list[dict[str, Any]] = []
    evaluation_methods: dict[str, Any] = {}
    started = time.perf_counter()
    for deadline in config()["lookahead_ms"]:
        model = artifact["models"][str(deadline)]
        x = _standardize(
            np.asarray(data[f"x_{deadline}"], dtype=np.float32),
            np.asarray(model["mean"], dtype=np.float32),
            np.asarray(model["scale"], dtype=np.float32),
        )
        linear = _apply_calibration(
            _linear_predict(
                x,
                np.asarray(model["linear_coef"], dtype=np.float32),
                float(model["linear_intercept"]),
            ),
            tuple(model["linear_calibration"]),
        )
        mlp_raw = np.mean(
            [_mlp_predict(x, state, config()) for state in model["mlp_states"]], axis=0
        )
        mlp = _apply_calibration(mlp_raw, tuple(model["mlp_calibration"]))
        for index, row in enumerate(rows):
            predictions.append(
                {
                    "candidate_id": row["candidate_id"],
                    "session_id": row["session_id"],
                    "boundary_sample": row["boundary_sample"],
                    "first_available_sample": row["first_available_sample"],
                    "lookahead_ms": int(deadline),
                    "decision_available_sample": int(row["boundary_sample"]) + int(deadline) * 16,
                    "raw_change_score": row["raw_change_score"],
                    "linear_score": float(linear[index]),
                    "mlp_score": float(mlp[index]),
                    "event_sample": row["event_sample"],
                    "stratum": row["stratum"],
                }
            )
        for method, scores in (("linear", linear), ("mlp", mlp)):
            key = f"{method}_{deadline}"
            selected = development["methods"][key]["selected_operating_points"]
            target_metrics: dict[str, Any] = {}
            for target, operating_point in selected.items():
                if operating_point is None:
                    target_metrics[target] = None
                    continue
                threshold = float(operating_point["threshold"])
                events = _local_events(rows, scores, threshold)
                target_metrics[target] = {
                    "threshold": threshold,
                    "development_operating_point": operating_point,
                    "evaluation": _metrics_for_events(
                        rows, events, inventory, int(deadline)
                    ),
                }
            evaluation_methods[key] = {
                "method": method,
                "lookahead_ms": int(deadline),
                "operating_points": target_metrics,
            }
    raw = _raw_metrics(rows, inventory)
    write_jsonl(output_root(root) / "evaluation_predictions.jsonl", predictions)
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "development_metrics_sha256": sha256_file(development_path),
        "model_sha256": sha256_file(model_path),
        "raw_candidates": raw,
        "methods": evaluation_methods,
        "verifier_wall_seconds": time.perf_counter() - started,
        "candidate_count": len(rows),
        "verifier_seconds_per_candidate_view": (
            (time.perf_counter() - started) / max(1, len(rows) * 6)
        ),
        "worker_job_id": os.environ.get("ORCA_WORKER_JOB_ID"),
    }
    path = output_root(root) / "evaluation_metrics.json"
    write_json(path, result)
    make_report(root)
    return path


def _headline(method: dict[str, Any], target: str = "10") -> dict[str, Any] | None:
    row = method["operating_points"].get(target)
    return row["evaluation"] if row is not None else None


def make_report(root: Path) -> Path:
    development = load_json(output_root(root) / "development_metrics.json")
    evaluation = load_json(output_root(root) / "evaluation_metrics.json")
    inventory = load_json(output_root(root) / "inventory.json")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    for key, method in development["methods"].items():
        curve = method["curve"]
        axis.plot(
            [row["false_events_per_hour"] for row in curve],
            [row["recall_250"] for row in curve],
            label=key.replace("_", " "),
            linewidth=1.2,
        )
    axis.set_xscale("symlog", linthresh=1.0)
    axis.set_xlabel("development false events/hour")
    axis.set_ylabel("Recall@250ms")
    axis.legend(fontsize=8)
    figure.tight_layout()
    curve_path = output_root(root) / "recall_false_event_curve.png"
    figure.savefig(curve_path, dpi=160)
    plt.close(figure)
    prediction_rows = read_jsonl(output_root(root) / "evaluation_predictions.jsonl")
    selected_rows = [row for row in prediction_rows if row["lookahead_ms"] == 1000]
    accepted = sorted(selected_rows, key=lambda row: -float(row["mlp_score"]))[:5]
    rejected = sorted(selected_rows, key=lambda row: float(row["mlp_score"]))[:5]
    trajectories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        trajectories[str(row["session_id"])].append(row)
    for values in trajectories.values():
        values.sort(key=lambda row: int(row["boundary_sample"]))
    figure, axes = plt.subplots(5, 2, figsize=(12, 12))
    for axis, row in zip(axes.reshape(-1), accepted + rejected, strict=True):
        center = int(row["boundary_sample"])
        nearby = [
            value
            for value in trajectories[str(row["session_id"])]
            if abs(int(value["boundary_sample"]) - center) <= 32000
        ]
        axis.plot(
            [(int(value["boundary_sample"]) - center) / SAMPLE_RATE for value in nearby],
            [float(value["mlp_score"]) for value in nearby],
            marker=".",
            linewidth=1.0,
            label="MLP score",
        )
        axis.axvline(0.0, color="tab:orange", linewidth=1.0, label="inspected candidate")
        if row["event_sample"] is not None:
            axis.axvline(
                (int(row["event_sample"]) - center) / SAMPLE_RATE,
                color="black",
                linestyle="--",
                linewidth=1.0,
                label="nearest reference",
            )
        axis.set_ylim(0.0, 1.0)
        axis.set_xlim(-2.0, 2.0)
        axis.set_title(f"{row['session_id']} @ {row['boundary_sample'] / SAMPLE_RATE:.1f}s\n{row['stratum']}", fontsize=8)
        axis.set_xlabel("seconds from inspected candidate")
        axis.set_ylabel("score")
    figure.tight_layout()
    timeline_path = output_root(root) / "representative_timelines.png"
    figure.savefig(timeline_path, dpi=150)
    plt.close(figure)
    raw_primary = evaluation["raw_candidates"]["tolerances"]["250"]
    table_rows: list[tuple[str, int, dict[str, Any] | None]] = [
        ("Raw ERes adjacent candidates", 500, evaluation["raw_candidates"])
    ]
    for method in ("linear", "mlp"):
        for deadline in (500, 750, 1000):
            table_rows.append(
                (
                    "Linear relation verifier" if method == "linear" else "Small relation MLP",
                    deadline,
                    _headline(evaluation["methods"][f"{method}_{deadline}"]),
                )
            )
    best_key = None
    best_recall = -1.0
    for key, method in evaluation["methods"].items():
        headline = _headline(method)
        if headline is None:
            continue
        recall = float(headline["tolerances"]["250"]["recall"] or 0.0)
        if recall > best_recall:
            best_recall = recall
            best_key = key
    if best_key is None:
        recommendation = "Outcome D — Inconclusive data or measurement"
    else:
        best = _headline(evaluation["methods"][best_key])
        best_primary = best["tolerances"]["250"]
        false_rate_reduction = (
            float(raw_primary["false_events_per_hour"])
            / max(float(best_primary["false_events_per_hour"]), 1e-12)
        )
        if false_rate_reduction >= 5.0 and float(best_primary["recall"] or 0.0) >= 0.5:
            recommendation = "Outcome A — Candidate-gated verification is useful"
        elif float(best_primary["recall"] or 0.0) >= 0.3:
            recommendation = "Outcome B — Relation evidence works but candidate gating is the bottleneck"
        else:
            recommendation = "Outcome C — Frozen ERes local relation is still insufficient"
    lines = [
        "# R7-A ERes Candidate Relation Verifier Report",
        "",
        f"Recommendation: **{recommendation}**.",
        "",
        "The five evaluation meetings were scored once after meeting-held development training, calibration, duplicate suppression, and operating-point rules were frozen.",
        "",
        "## Natural continuous result",
        "",
        "The verifier rows below use the development-selected 10 false-events/hour target when reachable.",
        "",
        "| Method | Lookahead | Recall@250 | Recall@500 | False events/h | Median / p95 availability | Notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, deadline, metrics in table_rows:
        if metrics is None:
            lines.append(f"| {name} | {deadline} ms | n/a | n/a | n/a | n/a | target unreachable |")
            continue
        primary = metrics["tolerances"]["250"]
        recall_500 = metrics["tolerances"]["500"]["recall"]
        availability = metrics["availability_latency_ms"]
        lines.append(
            f"| {name} | {deadline} ms | {float(primary['recall'] or 0.0):.3f} | {float(recall_500 or 0.0):.3f} | {float(primary['false_events_per_hour']):.3f} | {availability['median']:.0f} / {availability['p95']:.0f} ms | {'fixed raw threshold' if name.startswith('Raw') else 'development-selected OP'} |"
        )
    lines.extend(
        [
            "",
            "## Candidate ceiling and exposure",
            "",
            f"- Development: {inventory['summary']['development_hours']:.3f} h, {inventory['summary']['development_events']} reference changes.",
            f"- Evaluation: {inventory['summary']['evaluation_hours']:.3f} h, {inventory['summary']['evaluation_events']} reference changes.",
            f"- Raw evaluation candidate Recall@250: {float(raw_primary['recall'] or 0.0):.3f}; false events/h: {float(raw_primary['false_events_per_hour']):.3f}.",
            "",
            "## Validity and inspection",
            "",
            "- Development and evaluation session lists are disjoint and evaluation data is not loaded by the development command.",
            "- Every relation feature is masked to an embedding whose audio ends no later than its deadline.",
            "- Event matching is one-to-one and exposure uses the actual eligible duration of each meeting.",
            "- The representative figure contains five highest-scored accepted and five lowest-scored rejected evaluation candidates for manual timeline inspection.",
            "- Boundary timestamps remain at the source candidate; availability is reported separately.",
            "",
            "## Required next decision",
            "",
            "Based on the R7-A result, do you approve planning, implementing, and running R7-B: a 1-second fixed-lag local speaker segmentation model that no longer depends on ERes candidate points?",
            "",
            "R7-B was not planned, implemented, or run.",
        ]
    )
    path = output_root(root) / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def smoke(root: Path) -> dict[str, Any]:
    cfg = config()
    if cfg["development_sessions"] == cfg["evaluation_sessions"]:
        raise R7Error("split collision")
    rng = np.random.default_rng(1701)
    embeddings = rng.normal(size=(len(SEQUENCE_OFFSETS_MS), 192)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    audio = rng.normal(size=(len(SEQUENCE_OFFSETS_MS), 2)).astype(np.float32)
    dimensions = {
        str(deadline): int(_relation_feature(embeddings, audio, int(deadline)).shape[0])
        for deadline in cfg["lookahead_ms"]
    }
    if len(set(dimensions.values())) != 1:
        raise R7Error("feature dimensions differ by deadline")
    predictions = [100, 300, 900]
    references = [120, 910]
    matched, false, misses = _matches(predictions, references, 50)
    if len(matched) != 2 or false != [300] or misses:
        raise R7Error("one-to-one matching smoke failed")
    result = {
        "feature_dimensions": dimensions,
        "matching": {"matched": matched, "false": false, "misses": misses},
        "config_sha256": sha256_file(CONFIG_PATH),
        "code_sha256": sha256_file(CODE_PATH),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("inventory", "extract", "relations", "develop", "evaluate", "report", "smoke"),
    )
    parser.add_argument("--role", choices=("development", "evaluation"))
    args = parser.parse_args(argv)
    root = cache_root()
    if args.action == "inventory":
        print(prepare_inventory(root))
    elif args.action == "extract":
        if args.role is None:
            raise SystemExit("extract requires --role")
        print(extract_dense(root, args.role))
    elif args.action == "relations":
        if args.role is None:
            raise SystemExit("relations requires --role")
        print(build_relations(root, args.role))
    elif args.action == "develop":
        print(develop(root))
    elif args.action == "evaluate":
        print(evaluate(root))
    elif args.action == "report":
        print(make_report(root))
    else:
        smoke(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
