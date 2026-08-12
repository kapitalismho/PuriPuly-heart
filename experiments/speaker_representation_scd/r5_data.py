from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.speaker_representation_scd.provenance import (
    load_json,
    self_sha256_valid,
    sha256_file,
    with_self_sha256,
)
from experiments.speaker_representation_scd.r1_gate import EXPERIMENT_ROOT
from experiments.speaker_representation_scd.r3_probe import REGISTRY_LAYER_ORDER

CONFIG_PATH = EXPERIMENT_ROOT / "configs/r5/frozen_causal_head.json"
ANCHOR_PATH = Path("data/r3/legacy_common_gt/anchor_index.jsonl")
R3_POOLED_DIR = Path("data/r3/legacy_common_gt/pooled")
R4_POOLED_DIR = Path("data/r4/legacy_common_gt/pooled")
COORDINATE_DIR = Path("data/r2/legacy_common_gt/coordinates")
SPLIT_PATH = Path("manifests/r5/split_manifest.json")


class R5DataError(RuntimeError):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_config() -> dict[str, Any]:
    return load_json(CONFIG_PATH)


def group_identity(row: dict[str, Any]) -> str:
    synthetic = row.get("synthetic_manifest")
    if synthetic:
        if isinstance(synthetic, dict):
            return "synthetic:" + json.dumps(synthetic, ensure_ascii=False, sort_keys=True)
        return f"synthetic:{synthetic}"
    return f"block:{row.get('block_id') or row.get('waveform_id') or row['session_id']}"


def _hash_fraction(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def r4_sessions(cache_root: Path) -> set[str]:
    index = cache_root / R4_POOLED_DIR / "mhubert-147/index_300.jsonl"
    if not index.is_file():
        raise R5DataError(f"R4 index missing: {index}")
    return {str(row["session_id"]) for row in read_jsonl(index)}


def build_grouped_split(
    anchors: list[dict[str, Any]],
    excluded_sessions: set[str],
    *,
    dev_fraction: float,
    seed: int,
    search_trials: int,
) -> dict[str, Any]:
    eligible = [row for row in anchors if str(row["session_id"]) not in excluded_sessions]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        groups[group_identity(row)].append(row)
    if not groups:
        raise R5DataError("no R5 groups remain after R4 exclusion")
    total_labels = Counter(str(row["class"]) for row in eligible)
    total_corpora = Counter(str(row["corpus"]) for row in eligible)
    best: tuple[float, int, set[str]] | None = None
    target_count = len(eligible) * dev_fraction
    for trial in range(search_trials):
        dev_groups = {
            key
            for key in groups
            if _hash_fraction(f"{seed}:{trial}:{key}") < dev_fraction
        }
        if not dev_groups or len(dev_groups) == len(groups):
            continue
        dev_rows = [row for key in dev_groups for row in groups[key]]
        dev_labels = Counter(str(row["class"]) for row in dev_rows)
        if any(dev_labels[label] == 0 for label in total_labels):
            continue
        train_labels = total_labels - dev_labels
        if any(train_labels[label] == 0 for label in total_labels):
            continue
        dev_corpora = Counter(str(row["corpus"]) for row in dev_rows)
        train_corpora = total_corpora - dev_corpora
        if any(dev_corpora[corpus] == 0 for corpus in total_corpora):
            continue
        if any(train_corpora[corpus] == 0 for corpus in total_corpora):
            continue
        size_error = abs(len(dev_rows) - target_count) / len(eligible)
        label_error = sum(
            abs(dev_labels[label] / total_labels[label] - dev_fraction)
            for label in sorted(total_labels)
        )
        corpus_error = sum(
            abs(dev_corpora[corpus] / total_corpora[corpus] - dev_fraction)
            for corpus in sorted(total_corpora)
        )
        score = size_error + label_error + 0.5 * corpus_error
        candidate = (score, trial, dev_groups)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise R5DataError("unable to construct a grouped train/dev split")
    _, selected_trial, dev_groups = best
    entries = []
    for row in sorted(eligible, key=lambda value: str(value["candidate_id"])):
        group = group_identity(row)
        entries.append(
            {
                "candidate_id": str(row["candidate_id"]),
                "class": str(row["class"]),
                "corpus": str(row["corpus"]),
                "group_id": group,
                "session_id": str(row["session_id"]),
                "split": "dev" if group in dev_groups else "train",
                "waveform_id": str(row["waveform_id"]),
            }
        )
    return {
        "entries": entries,
        "excluded_r4_sessions": sorted(excluded_sessions),
        "selected_trial": selected_trial,
    }


def create_split_manifest(cache_root: Path, *, replace: bool = False) -> dict[str, Any]:
    config = load_config()
    anchor_path = cache_root / ANCHOR_PATH
    anchors = read_jsonl(anchor_path)
    split = build_grouped_split(
        anchors,
        r4_sessions(cache_root),
        dev_fraction=float(config["split"]["dev_fraction"]),
        seed=int(config["split"]["seed"]),
        search_trials=int(config["split"]["search_trials"]),
    )
    entries = split["entries"]
    counts = Counter((str(row["split"]), str(row["class"])) for row in entries)
    group_sets = {
        split_name: {str(row["group_id"]) for row in entries if row["split"] == split_name}
        for split_name in ("train", "dev")
    }
    if group_sets["train"] & group_sets["dev"]:
        raise R5DataError("group identity crosses train and dev")
    document = with_self_sha256(
        {
            "schema_version": 1,
            "artifact_role": "r5_grouped_split_manifest",
            "experiment_id": "speaker_representation_scd_v1",
            "scope": "legacy-common-gt-v1",
            "config_sha256": sha256_file(CONFIG_PATH),
            "anchor_index_sha256": sha256_file(anchor_path),
            "selection": {
                "seed": int(config["split"]["seed"]),
                "search_trials": int(config["split"]["search_trials"]),
                "selected_trial": int(split["selected_trial"]),
                "dev_fraction_target": float(config["split"]["dev_fraction"]),
                "group_rule": "synthetic_manifest_else_natural_meeting_block",
                "corpus_coverage_rule": "every_corpus_present_in_train_and_dev",
            },
            "counts": {
                "train_positive": counts[("train", "positive")],
                "train_negative": counts[("train", "negative")],
                "dev_positive": counts[("dev", "positive")],
                "dev_negative": counts[("dev", "negative")],
                "train_groups": len(group_sets["train"]),
                "dev_groups": len(group_sets["dev"]),
                "excluded_r4_sessions": len(split["excluded_r4_sessions"]),
            },
            "excluded_r4_sessions": split["excluded_r4_sessions"],
            "entries": entries,
        }
    )
    path = cache_root / SPLIT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = load_json(path)
        if self_sha256_valid(existing) and existing == document:
            return existing
        if not replace:
            raise R5DataError(f"refusing to replace a different split manifest: {path}")
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return document


def load_split_manifest(cache_root: Path) -> dict[str, Any]:
    path = cache_root / SPLIT_PATH
    document = load_json(path)
    if not self_sha256_valid(document):
        raise R5DataError(f"split manifest identity invalid: {path}")
    return document


def _coordinate_rows(
    cache_root: Path, candidate_ids: set[str], context_ms: int
) -> dict[str, dict[int | str, dict[str, Any]]]:
    result: dict[str, dict[int | str, dict[str, Any]]] = defaultdict(dict)
    for path in sorted((cache_root / COORDINATE_DIR).glob("*.jsonl")):
        for row in read_jsonl(path):
            candidate_id = row.get("candidate_id")
            if candidate_id not in candidate_ids or int(row.get("context_ms") or 0) != context_ms:
                continue
            if row.get("coordinate_role") == "r3_primary":
                result[str(candidate_id)]["primary"] = row
            offset = row.get("trajectory_offset_ms")
            if offset is not None:
                result[str(candidate_id)][int(offset)] = row
    return result


def _pooled_row_lookup(cache_root: Path, model_id: str, context_ms: int) -> dict[tuple[str, int, int], int]:
    path = cache_root / R3_POOLED_DIR / model_id / f"index_{context_ms}.jsonl"
    return {
        (str(row["waveform_id"]), int(row["window_start_sample"]), int(row["window_end_sample"])): int(
            row["row_index"]
        )
        for row in read_jsonl(path)
    }


def linear_change_descriptors(
    cache_root: Path, model_id: str
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    config = load_config()
    context_ms = int(config["context_ms"])
    layer_id = str(config["models"][model_id])
    layer_order = REGISTRY_LAYER_ORDER[model_id]
    layer_index = layer_order.index(layer_id)
    split = load_split_manifest(cache_root)
    entries = split["entries"]
    candidate_ids = {str(row["candidate_id"]) for row in entries}
    coordinates = _coordinate_rows(cache_root, candidate_ids, context_ms)
    lookup = _pooled_row_lookup(cache_root, model_id, context_ms)
    vectors = np.load(
        cache_root / R3_POOLED_DIR / model_id / f"vectors_{context_ms}.npy", mmap_mode="r"
    )
    features: list[np.ndarray] = []
    labels: list[float] = []
    metadata: list[dict[str, Any]] = []
    for entry in entries:
        candidate_id = str(entry["candidate_id"])
        candidate_rows = coordinates.get(candidate_id, {})
        before = candidate_rows.get(-100)
        after = candidate_rows.get(300) or candidate_rows.get("primary")
        if before is None or after is None:
            continue
        keys = []
        for row in (before, after):
            keys.append(
                (
                    str(row["waveform_id"]),
                    int(row["window_start_sample"]),
                    int(row["window_end_sample"]),
                )
            )
        if any(key not in lookup for key in keys):
            continue
        z_before = np.asarray(vectors[lookup[keys[0]], layer_index], dtype=np.float32)
        z_after = np.asarray(vectors[lookup[keys[1]], layer_index], dtype=np.float32)
        if not np.isfinite(z_before).all() or not np.isfinite(z_after).all():
            continue
        denominator = float(np.linalg.norm(z_before) * np.linalg.norm(z_after))
        cosine_distance = 1.0 - float(np.dot(z_before, z_after) / denominator)
        features.append(
            np.concatenate(
                [np.abs(z_after - z_before), np.asarray([cosine_distance], dtype=np.float32)]
            )
        )
        labels.append(1.0 if entry["class"] == "positive" else 0.0)
        metadata.append(dict(entry))
    if not features:
        raise R5DataError(f"no linear descriptors available for {model_id}")
    return np.stack(features), np.asarray(labels, dtype=np.float32), metadata


def sequence_rows(cache_root: Path) -> list[dict[str, Any]]:
    config = load_config()
    split = load_split_manifest(cache_root)
    anchors = {str(row["candidate_id"]): row for row in read_jsonl(cache_root / ANCHOR_PATH)}
    source_metadata = {
        str(row["session_id"]): row
        for row in read_jsonl(cache_root / "data/r2/legacy_common_gt/source_metadata.jsonl")
    }
    offsets = list(
        range(
            int(config["sequence"]["start_offset_ms"]),
            int(config["sequence"]["end_offset_ms"]) + 1,
            int(config["sequence"]["step_ms"]),
        )
    )
    positives = {int(value) for value in config["sequence"]["positive_offsets_ms"]}
    context_samples = int(config["context_ms"]) * 16
    rows: list[dict[str, Any]] = []
    for entry in split["entries"]:
        anchor = anchors[str(entry["candidate_id"])]
        source = source_metadata[str(entry["session_id"])]
        available = [
            (offset, int(anchor["coordinate"]) + offset * 16)
            for offset in offsets
            if int(source["eligible_start_sample"]) + context_samples
            <= int(anchor["coordinate"]) + offset * 16
            <= int(source["eligible_end_sample"])
        ]
        if not available:
            continue
        available_offsets = [offset for offset, _ in available]
        frontiers = [frontier for _, frontier in available]
        starts = [frontier - context_samples for frontier in frontiers]
        labels = [
            1.0 if entry["class"] == "positive" and offset in positives else 0.0
            for offset in available_offsets
        ]
        if entry["class"] == "positive" and not any(labels):
            continue
        rows.append(
            {
                **entry,
                "coordinate": int(anchor["coordinate"]),
                "frontier_samples": frontiers,
                "labels": labels,
                "offsets_ms": available_offsets,
                "window_start_samples": starts,
                "window_end_samples": frontiers,
            }
        )
    return rows
