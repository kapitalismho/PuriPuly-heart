from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import io
import json
import math
import os
import platform
import random
import tempfile
import time
import wave
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import onnxruntime as ort

from experiments.speaker_turn_boundary.adapters.eres2netv2 import (
    EresEmbeddingRuntime,
    cosine_similarity,
    kaldi_fbank_numpy,
)
from experiments.speaker_turn_boundary.frontend import output_frame_available_16k_count
from experiments.speaker_turn_boundary.phase3_ls import (
    LSCaptureEpoch,
    LSEENDCapture,
    load_sidecar_metadata,
)
from experiments.speaker_turn_boundary.provenance import LS_EEND_VARIANTS
from experiments.speaker_turn_boundary.reducer import ReductionProfile, StreamingReducer
from experiments.speaker_turn_boundary.run_eres_sweep import ERES_CHECKPOINTS

from .phase4_design import (
    ADJACENT_WINDOWS,
    ANCHOR_WINDOWS,
    AUTHORITY_SHA256,
    ERES_PARALLEL_WORKERS,
    GROUP_GRAPH_SHA256,
    LONG_STEPS,
    LS_ACOUSTIC_SUPPORT_BY_HORIZON,
    MANIFEST_BYTE_SHA256,
    MANIFEST_CONTENT_SHA256,
    STEPS,
    build_candidates,
    canonical_json,
    ceil_grid,
    component_map,
    load_public_regions,
    load_synthetic_cases,
    match_pairs,
    sha256_bytes,
    sha256_file,
    synthetic_case_id,
    synthetic_manifest_name,
)

PHASE4_BUNDLE_SHA256 = "a6afa3dc946815c162ee18d09b1c7ad3ad08e252f7286c110b37a685fe2b1759"
DESIGN_LEDGER_SHA256 = "0a86788a4817d4a205d92b0afb6ee05dc97d11da3e99d4c0501d74be30473691"
DESIGN_LEDGER_CONTENT_SHA256 = "c8336c2665b28047b1a169fc9605a6c6a3c400afe553dc3a2d35ca9b20b41536"
PAIR_ROWS_SHA256 = "fb29fff960932f2840433fa94f1a9e4bade167a6d935a6458dc6e9b191a4f9b9"
COORDINATE_ROWS_SHA256 = "58cbd9eaf4554761bf71e698bc4b1f251ae722c4281be35d0270dbc0ab285470"
EMBEDDING_WINDOWS_SHA256 = "de3646936555280a01dcf2461d3d02bccc722a55a088aa15b5a8c97f639e3118"
ACOUSTIC_WINDOWS_SHA256 = "3fe0ffef5a2dc79ec385f89924181ff2e98c6f61f812ca39007eccb6083b148a"
ERES_MODEL_SHA256 = {
    "E-standard": "7a6d4f89dcb92a554806bdf6bfb13c7fae0a63e8f992a49b3a503b9a03c705cf",
    "E-w24s4ep4": "3761572a872a29f36af66065075cc9a48adc23c8b26fb0c68488aa3ed8f35f26",
}
HORIZONS_MS = (250, 500, 1000)
LS_HARD_EXTRACTORS = (
    "ls_new_track_rise.v1",
    "ls_dominant_replacement.v1",
    "ls_activity_set_change.v1",
)
LS_SECONDARY_EXTRACTORS = ("ls_overlap_strength.v1",)
ERES_STATES = (
    "stable_no_update",
    "stable_ema",
    "confirmed_anchor",
    "prototype_memory_4",
)
ERES_CACHE_SHARD_TARGET_BYTES = 16 * 1024 * 1024
ACOUSTIC_EXTRACTORS = (
    "acoustic_log_rms_delta.v1",
    "acoustic_logmel_flux.v1",
)
DETAIL_SHARD_LIMIT = 20 * 1024 * 1024
AGGREGATE_LIMIT = 10 * 1024 * 1024
BOOTSTRAP_REPLICATES = 10_000
STATE_EQ_CHANGE_THRESHOLD = 0.50
STATE_EQ_LS_PROFILE = {
    "threshold": 0.50,
    "persistence": 2,
    "median_width": 1,
}


class Phase4SignalError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AudioSource:
    source_id: str
    wav_sha256: str
    path: Path
    duration_samples: int
    public: bool


@dataclass(slots=True)
class Phase4Inputs:
    experiment_dir: Path
    result_dir: Path
    episodes: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    pairs: list[dict[str, Any]]
    sources: dict[str, AudioSource]
    source_by_episode: dict[str, str]
    embedding_windows: dict[str, set[tuple[int, int]]]
    acoustic_windows: dict[str, set[tuple[int, int]]]
    design_ledger: dict[str, Any]


def content_hash(payload: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def verify_content_hash(payload: dict[str, Any]) -> None:
    stored = payload.get("content_sha256")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    actual = content_hash(body)
    if stored != actual:
        raise Phase4SignalError(f"content hash mismatch: {stored} != {actual}")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body["content_sha256"] = content_hash(body)
    encoded = (canonical_json(body) + "\n").encode("utf-8")
    if len(encoded) > AGGREGATE_LIMIT:
        raise Phase4SignalError(f"aggregate exceeds 10 MiB: {path} ({len(encoded)})")
    atomic_write_bytes(path, encoded)
    return body


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_content_hash(payload)
    return payload


def deterministic_gzip(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as handle:
        handle.write(payload)
    return buffer.getvalue()


def write_detail_shards(
    root: Path,
    family: str,
    checkpoint: str,
    rows: Iterable[dict[str, Any]],
    *,
    target_uncompressed_bytes: int = 32 * 1024 * 1024,
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: str(row["row_key"]))
    chunks: list[list[bytes]] = []
    current: list[bytes] = []
    size = 0
    for row in ordered:
        encoded = (canonical_json(row) + "\n").encode("utf-8")
        if current and size + len(encoded) > target_uncompressed_bytes:
            chunks.append(current)
            current = []
            size = 0
        current.append(encoded)
        size += len(encoded)
    if current:
        chunks.append(current)
    directory = root / family
    directory.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    next_chunks: list[list[bytes]] = chunks
    shard = 0
    while next_chunks:
        values = next_chunks.pop(0)
        plain = b"".join(values)
        compressed = deterministic_gzip(plain)
        if len(compressed) > DETAIL_SHARD_LIMIT:
            if len(values) == 1:
                raise Phase4SignalError("single detail row exceeds shard limit")
            middle = len(values) // 2
            next_chunks.insert(0, values[middle:])
            next_chunks.insert(0, values[:middle])
            continue
        name = f"{checkpoint}-{shard:04d}.jsonl.gz"
        path = directory / name
        atomic_write_bytes(path, compressed)
        first = json.loads(values[0])["row_key"]
        last = json.loads(values[-1])["row_key"]
        index.append(
            {
                "family": family,
                "checkpoint": checkpoint,
                "path": str(path.relative_to(root.parent)).replace("\\", "/"),
                "row_count": len(values),
                "first_row_key": first,
                "last_row_key": last,
                "content_sha256": sha256_bytes(plain),
                "byte_sha256": sha256_bytes(compressed),
                "size_bytes": len(compressed),
            }
        )
        shard += 1
    return index


def iter_detail_rows(result_dir: Path, entry: dict[str, Any]) -> Iterator[dict[str, Any]]:
    path = result_dir / str(entry["path"])
    compressed = path.read_bytes()
    if sha256_bytes(compressed) != entry["byte_sha256"]:
        raise Phase4SignalError(f"detail byte hash mismatch: {path}")
    plain = gzip.decompress(compressed)
    if sha256_bytes(plain) != entry["content_sha256"]:
        raise Phase4SignalError(f"detail content hash mismatch: {path}")
    lines = plain.splitlines()
    if len(lines) != int(entry["row_count"]):
        raise Phase4SignalError(f"detail row count mismatch: {path}")
    for line in lines:
        yield json.loads(line)


def _candidate_inputs(
    experiment_dir: Path,
    episodes: list[dict[str, Any]],
    inventory: dict[str, Any],
    details: dict[str, dict[str, Any]],
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    manifests_dir = experiment_dir / "data" / "manifests"
    cases = load_synthetic_cases(manifests_dir)
    public_sessions = [
        str(row["session_id"])
        for row in episodes
        if synthetic_manifest_name(str(row["session_id"])) is None
    ]
    regions = load_public_regions(inventory, details, public_sessions, manifests_dir)
    positives, negatives = build_candidates(
        episodes,
        cases,
        component_map(inventory),
        regions,
    )
    return cases, positives, negatives


def _source_maps(
    experiment_dir: Path,
    episodes: list[dict[str, Any]],
    cases: dict[tuple[str, str], dict[str, Any]],
    inventory: dict[str, Any],
    details: dict[str, dict[str, Any]],
) -> tuple[dict[str, AudioSource], dict[str, str]]:
    sources: dict[str, AudioSource] = {}
    by_episode: dict[str, str] = {}
    corpus_root = Path(str(inventory["corpus_root"]))
    for episode in episodes:
        session_id = str(episode["session_id"])
        manifest_name = synthetic_manifest_name(session_id)
        if manifest_name is None:
            info = details[session_id]
            path = (corpus_root / str(info["wav_path"])).resolve()
            source_id = session_id
            declared_duration = int(info["duration_samples"])
            public = True
        else:
            case_id = synthetic_case_id(session_id)
            if case_id is None:
                raise Phase4SignalError(f"invalid synthetic session: {session_id}")
            case = cases[(manifest_name, case_id)]
            relative = Path(str(case["wav_relative_path"]))
            roots = (
                experiment_dir / "data",
                corpus_root / "phase2_build",
                corpus_root,
            )
            resolved = [root / relative for root in roots if (root / relative).is_file()]
            if not resolved:
                path = (experiment_dir / "data" / relative).resolve()
            else:
                path = resolved[0].resolve()
            source_id = session_id
            declared_duration = int(case["duration_samples"])
            public = False
        if not path.is_file():
            raise Phase4SignalError(f"WAV missing: {path}")
        with wave.open(str(path), "rb") as handle:
            if (
                handle.getnchannels() != 1
                or handle.getframerate() != 16000
                or handle.getsampwidth() != 2
            ):
                raise Phase4SignalError(f"unsupported WAV contract: {source_id}")
            duration = int(handle.getnframes())
        if int(episode["bounds"]["tail_end"]) > duration:
            raise Phase4SignalError(f"episode tail exceeds WAV: {episode['episode_id']}")
        if not public and duration != declared_duration:
            raise Phase4SignalError(f"synthetic WAV duration drift: {source_id}")
        wav_sha = str(episode["wav_sha256"])
        existing = sources.get(source_id)
        source = AudioSource(source_id, wav_sha, path, duration, public)
        if existing is not None and existing != source:
            raise Phase4SignalError(f"source identity drift: {source_id}")
        sources[source_id] = source
        by_episode[str(episode["episode_id"])] = source_id
    return sources, by_episode


def enumerate_coordinate_windows(
    episodes: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> tuple[
    dict[str, set[tuple[int, int]]],
    dict[str, set[tuple[int, int]]],
    int,
    str,
]:
    embedding: dict[str, set[tuple[int, int]]] = defaultdict(set)
    acoustic: dict[str, set[tuple[int, int]]] = defaultdict(set)
    rows: list[dict[str, Any]] = []
    public_sources: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        bounds = episode["bounds"]
        warm_start = int(bounds["warm_start"])
        scored_start = int(bounds["scored_start"])
        scored_end = int(bounds["scored_end"])
        tail_end = int(bounds["tail_end"])
        wav = str(episode["wav_sha256"])
        session_id = str(episode["session_id"])
        if synthetic_manifest_name(session_id) is None:
            source = public_sources.setdefault(
                wav,
                {
                    "source_id": session_id,
                    "maximum_tail_end": tail_end,
                },
            )
            source["maximum_tail_end"] = max(int(source["maximum_tail_end"]), tail_end)
        for window in ADJACENT_WINDOWS:
            for step in LONG_STEPS if window >= 24000 else STEPS:
                lo = max(scored_start, warm_start + window)
                hi = min(scored_end, tail_end - window)
                for boundary in range(ceil_grid(lo, step), hi + 1, step):
                    rows.append(
                        {
                            "episode_id": episode["episode_id"],
                            "kind": "adjacent_grid",
                            "profile": f"adjacent:W{window}:S{step}",
                            "boundary": boundary,
                            "observation_frontier": boundary + window,
                        }
                    )
                    embedding[wav].add((boundary - window, boundary))
                    embedding[wav].add((boundary, boundary + window))
        for window in ANCHOR_WINDOWS:
            for step in STEPS:
                first = ceil_grid(warm_start + window, step)
                for end in range(first, tail_end + 1, step):
                    rows.append(
                        {
                            "episode_id": episode["episode_id"],
                            "kind": "trailing_probe_grid",
                            "profile": f"trailing_probe:W{window}:S{step}",
                            "boundary": end - window,
                            "observation_frontier": end,
                        }
                    )
                    embedding[wav].add((end - window, end))
    for wav, source in sorted(public_sources.items()):
        maximum_tail = int(source["maximum_tail_end"])
        for window in ANCHOR_WINDOWS:
            for step in STEPS:
                for end in range(ceil_grid(window, step), maximum_tail + 1, step):
                    rows.append(
                        {
                            "source_id": source["source_id"],
                            "kind": "source_prefix_probe_grid",
                            "profile": f"source_prefix_probe:W{window}:S{step}",
                            "boundary": end - window,
                            "observation_frontier": end,
                        }
                    )
                    embedding[wav].add((end - window, end))
    for episode in episodes:
        session_id = str(episode["session_id"])
        if synthetic_manifest_name(session_id) is not None:
            continue
        warm_start = int(episode["bounds"]["warm_start"])
        for window in ANCHOR_WINDOWS:
            for step in STEPS:
                last_probe_end = (warm_start // step) * step
                for state_mode in ERES_STATES:
                    rows.append(
                        {
                            "episode_id": episode["episode_id"],
                            "kind": "source_prefix_state_snapshot",
                            "state_mode": state_mode,
                            "window_samples": window,
                            "step_samples": step,
                            "snapshot_frontier": warm_start,
                            "last_probe_end": last_probe_end if last_probe_end >= window else None,
                        }
                    )
    episode_by_id = {str(row["episode_id"]): row for row in episodes}
    for candidate in sorted(candidates, key=lambda row: str(row["candidate_id"])):
        episode = episode_by_id[str(candidate["episode_id"])]
        bounds = episode["bounds"]
        warm_start = int(bounds["warm_start"])
        tail_end = int(bounds["tail_end"])
        boundary = int(candidate["coordinate"])
        wav = str(candidate["wav_sha256"])
        for window in ADJACENT_WINDOWS:
            if boundary - window < warm_start or boundary + window > tail_end:
                continue
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "kind": "reference_aligned_measurement",
                    "profile": f"measurement_adjacent:W{window}",
                    "boundary": boundary,
                    "observation_frontier": boundary + window,
                }
            )
            embedding[wav].add((boundary - window, boundary))
            embedding[wav].add((boundary, boundary + window))
        for window in ANCHOR_WINDOWS:
            if boundary + window > tail_end:
                continue
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "kind": "read_only_anchor_probe",
                    "profile": f"measurement_probe:W{window}",
                    "boundary": boundary,
                    "observation_frontier": boundary + window,
                }
            )
            embedding[wav].add((boundary, boundary + window))
        for horizon, support in LS_ACOUSTIC_SUPPORT_BY_HORIZON.items():
            if boundary - support < warm_start or boundary + support > tail_end:
                continue
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "kind": "ls_reference_aligned_acoustic",
                    "profile": f"ls_acoustic:H{horizon}:W{support}",
                    "boundary": boundary,
                    "observation_frontier": boundary + support,
                }
            )
            acoustic[wav].add((boundary - support, boundary))
            acoustic[wav].add((boundary, boundary + support))
    rows.sort(key=canonical_json)
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json(row).encode("utf-8") + b"\n")
    return dict(embedding), dict(acoustic), len(rows), digest.hexdigest()


def _window_digest(windows: dict[str, set[tuple[int, int]]]) -> tuple[int, str]:
    rows = [
        {"wav_sha256": wav, "start": start, "end": end}
        for wav, values in windows.items()
        for start, end in values
    ]
    rows.sort(key=lambda row: (row["wav_sha256"], row["start"], row["end"]))
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json(row).encode("utf-8") + b"\n")
    return len(rows), digest.hexdigest()


def load_inputs(experiment_dir: Path) -> Phase4Inputs:
    result_dir = experiment_dir / "results" / "turn_episode_v1"
    manifest_path = result_dir / "episode_manifest_dev.json"
    if sha256_file(manifest_path) != MANIFEST_BYTE_SHA256:
        raise Phase4SignalError("episode manifest byte hash drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("content_sha256") != MANIFEST_CONTENT_SHA256:
        raise Phase4SignalError("episode manifest content hash drift")
    if manifest.get("group_graph_hash") != GROUP_GRAPH_SHA256:
        raise Phase4SignalError("group graph hash drift")
    episodes = [row for row in manifest["episodes"] if row["pool"] == "diagnostic_dev"]
    if len(episodes) != 695:
        raise Phase4SignalError("diagnostic population drift")
    inventory = json.loads((result_dir / "coverage_inventory.json").read_text(encoding="utf-8"))
    details_rows = [
        json.loads(line)
        for line in (result_dir / "coverage_inventory_details.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    details = {str(row["session_id"]): row for row in details_rows}
    cases, positives, negatives = _candidate_inputs(experiment_dir, episodes, inventory, details)
    pairs, exclusions = match_pairs(positives, negatives)
    if len(pairs) != 313 or exclusions != {
        "positive_unmatched": 137,
        "negative_unused": 47,
        "groups_without_negative": 0,
    }:
        raise Phase4SignalError("matched pair ledger drift")
    pair_digest = hashlib.sha256()
    for pair in pairs:
        pair_digest.update(canonical_json(pair).encode("utf-8") + b"\n")
    if pair_digest.hexdigest() != PAIR_ROWS_SHA256:
        raise Phase4SignalError("matched pair hash drift")
    sources, source_by_episode = _source_maps(experiment_dir, episodes, cases, inventory, details)
    embedding, acoustic, coordinate_count, coordinate_sha = enumerate_coordinate_windows(
        episodes, positives + negatives
    )
    if coordinate_count != 1_217_509 or coordinate_sha != COORDINATE_ROWS_SHA256:
        raise Phase4SignalError("coordinate ledger drift")
    embedding_count, embedding_sha = _window_digest(embedding)
    acoustic_count, acoustic_sha = _window_digest(acoustic)
    if embedding_count != 895_656 or embedding_sha != EMBEDDING_WINDOWS_SHA256:
        raise Phase4SignalError("embedding window ledger drift")
    if acoustic_count != 4_371 or acoustic_sha != ACOUSTIC_WINDOWS_SHA256:
        raise Phase4SignalError("acoustic window ledger drift")
    design_path = result_dir / "phase_4_design_ledger.json"
    if sha256_file(design_path) != DESIGN_LEDGER_SHA256:
        raise Phase4SignalError("design ledger byte hash drift")
    design = read_json(design_path)
    if design["content_sha256"] != DESIGN_LEDGER_CONTENT_SHA256:
        raise Phase4SignalError("design ledger content hash drift")
    return Phase4Inputs(
        experiment_dir=experiment_dir,
        result_dir=result_dir,
        episodes=episodes,
        candidates=positives + negatives,
        pairs=pairs,
        sources=sources,
        source_by_episode=source_by_episode,
        embedding_windows=embedding,
        acoustic_windows=acoustic,
        design_ledger=design,
    )


def signal_registry() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint in LS_EEND_VARIANTS:
        for horizon in HORIZONS_MS:
            for base in LS_HARD_EXTRACTORS + LS_SECONDARY_EXTRACTORS:
                rows.append(
                    {
                        "signal_extractor_id": f"{base}:{checkpoint}:H{horizon}",
                        "family": "ls_eend",
                        "checkpoint": checkpoint,
                        "base_extractor_id": base,
                        "sign": "higher_means_change",
                        "causal_horizon_ms": horizon,
                        "valid_window_rule": "ordinary frame center >= candidate and availability <= candidate+horizon",
                        "missing_observation_rule": "score_0_with_missing_true_excluded_from_rank_metrics",
                        "gate_role": (
                            "secondary_overlap_only"
                            if base in LS_SECONDARY_EXTRACTORS
                            else "hard_target"
                        ),
                    }
                )
    for checkpoint in ERES_MODEL_SHA256:
        for horizon in HORIZONS_MS:
            for window in ADJACENT_WINDOWS:
                rows.append(
                    {
                        "signal_extractor_id": (
                            f"eres_adjacent_change.v1:{checkpoint}:W{window}:H{horizon}"
                        ),
                        "family": "eres2netv2",
                        "checkpoint": checkpoint,
                        "base_extractor_id": "eres_adjacent_change.v1",
                        "window_samples": window,
                        "sign": "higher_means_change",
                        "causal_horizon_ms": horizon,
                        "valid_window_rule": "reference-aligned adjacent real windows with end <= candidate+horizon",
                        "missing_observation_rule": "score_0_with_missing_true_excluded_from_rank_metrics",
                        "gate_role": "hard_target",
                    }
                )
            for window in ANCHOR_WINDOWS:
                for step in STEPS:
                    for state in ERES_STATES:
                        base = {
                            "stable_no_update": "eres_stable_anchor_change.v1",
                            "stable_ema": "eres_stable_anchor_change.v1",
                            "confirmed_anchor": "eres_confirmed_anchor_change.v1",
                            "prototype_memory_4": "eres_prototype_change.v1",
                        }[state]
                        rows.append(
                            {
                                "signal_extractor_id": (
                                    f"{base}:{checkpoint}:{state}:W{window}:S{step}:H{horizon}"
                                ),
                                "family": "eres2netv2",
                                "checkpoint": checkpoint,
                                "base_extractor_id": base,
                                "state_mode": state,
                                "window_samples": window,
                                "step_samples": step,
                                "sign": "higher_means_change",
                                "causal_horizon_ms": horizon,
                                "valid_window_rule": "read-only candidate probe after causal regular-probe state; confirmation uses next regular probe when required",
                                "missing_observation_rule": "score_0_with_missing_true_excluded_from_rank_metrics",
                                "gate_role": "hard_target",
                            }
                        )
    return sorted(rows, key=lambda row: row["signal_extractor_id"])


def proposal_contract_payload(existing: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in existing.items() if key != "content_sha256"}
    body["plan_sha256"] = AUTHORITY_SHA256
    body["phase_4_bundle_sha256"] = PHASE4_BUNDLE_SHA256
    body["signal_extractors"] = signal_registry()
    return body


def read_wav(source: AudioSource) -> np.ndarray:
    if not source.path.is_file():
        raise Phase4SignalError(f"WAV missing: {source.path}")
    if sha256_file(source.path) != source.wav_sha256:
        raise Phase4SignalError(f"WAV hash drift: {source.source_id}")
    with wave.open(str(source.path), "rb") as handle:
        channels = handle.getnchannels()
        rate = handle.getframerate()
        width = handle.getsampwidth()
        count = handle.getnframes()
        payload = handle.readframes(count)
    if channels != 1 or rate != 16000 or width != 2:
        raise Phase4SignalError(
            f"unsupported WAV contract {source.source_id}: {channels}/{rate}/{width}"
        )
    samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    if samples.size != source.duration_samples:
        raise Phase4SignalError(
            f"WAV duration drift {source.source_id}: {samples.size} != {source.duration_samples}"
        )
    return samples


def safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_" for character in value
    )


def tensor_contract(session: Any) -> dict[str, Any]:
    def node(value: Any) -> dict[str, Any]:
        return {
            "name": value.name,
            "type": value.type,
            "shape": [item if isinstance(item, (int, str)) else None for item in value.shape],
        }

    return {
        "inputs": [node(value) for value in session.get_inputs()],
        "outputs": [node(value) for value in session.get_outputs()],
    }


def cache_contract(
    inputs: Phase4Inputs,
    *,
    family: str,
    checkpoint: str,
    checkpoint_sha256: str,
    sidecar_sha256: str | None,
    tensor: dict[str, Any],
) -> dict[str, Any]:
    frontend_path = inputs.experiment_dir / "frontend.py"
    adapter_path = inputs.experiment_dir / "adapters" / "eres2netv2.py"
    body = {
        "schema_version": "turn_episode_phase4_cache_contract.v2",
        "authority_sha256": AUTHORITY_SHA256,
        "phase_4_bundle_sha256": PHASE4_BUNDLE_SHA256,
        "manifest_content_sha256": MANIFEST_CONTENT_SHA256,
        "design_ledger_content_sha256": DESIGN_LEDGER_CONTENT_SHA256,
        "family": family,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "sidecar_sha256": sidecar_sha256,
        "frontend_sha256": sha256_file(frontend_path),
        "eres_adapter_sha256": sha256_file(adapter_path) if family == "eres2netv2" else None,
        "state_mode": (
            "source_prefix_public_episode_epoch_synthetic"
            if family == "ls_eend"
            else "absolute_exact_window_source_prefix_state_replay"
        ),
        "source_origin": "absolute_16khz_sample_zero",
        "tensor_contract": tensor,
    }
    body["tensor_contract_sha256"] = content_hash(tensor)
    body["contract_sha256"] = content_hash(body)
    return body


def _ls_cache_paths(
    cache_root: Path, contract: dict[str, Any], checkpoint: str, source: AudioSource
) -> tuple[Path, Path]:
    base = (
        cache_root
        / str(contract["contract_sha256"])
        / "ls"
        / checkpoint
        / f"{safe_name(source.source_id)}_{source.wav_sha256[:16]}"
    )
    return base.with_suffix(".npz"), base.with_suffix(".json")


def _eres_cache_paths(
    cache_root: Path, contract: dict[str, Any], checkpoint: str, source: AudioSource
) -> tuple[Path, Path]:
    base = (
        cache_root
        / str(contract["contract_sha256"])
        / "eres"
        / checkpoint
        / f"{safe_name(source.source_id)}_{source.wav_sha256[:16]}"
    )
    return base, base.with_suffix(".json")


def _encode_ls_capture(capture: LSCaptureEpoch) -> bytes:
    normal = (
        np.stack(capture.normal_probs).astype(np.float32, copy=False)
        if capture.normal_probs
        else np.zeros((0, capture.track_count), dtype=np.float32)
    )
    tail = (
        np.stack(capture.tail_probs).astype(np.float32, copy=False)
        if capture.tail_probs
        else np.zeros((0, capture.track_count), dtype=np.float32)
    )
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        normal_probs=normal,
        tail_probs=tail,
        normal_frontiers=np.asarray(capture.normal_frontiers, dtype=np.int64),
        frame_wall_ns=np.asarray(capture.frame_wall_ns, dtype=np.int64),
        chunk_observed_counts=np.asarray(capture.chunk_observed_counts, dtype=np.int64),
        chunk_wall_seconds=np.asarray(capture.chunk_wall_seconds, dtype=np.float64),
        scalar_ints=np.asarray(
            [
                capture.audio_epoch,
                capture.epoch_end_count,
                capture.finalize_wall_ns,
                capture.length_samples,
            ],
            dtype=np.int64,
        ),
        scalar_floats=np.asarray([capture.cpu_seconds, capture.wall_seconds], dtype=np.float64),
    )
    return buffer.getvalue()


def _decode_ls_capture(payload: bytes, source_id: str) -> LSCaptureEpoch:
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        normal = np.asarray(data["normal_probs"], dtype=np.float32)
        tail = np.asarray(data["tail_probs"], dtype=np.float32)
        scalar_ints = np.asarray(data["scalar_ints"], dtype=np.int64)
        scalar_floats = np.asarray(data["scalar_floats"], dtype=np.float64)
        capture = LSCaptureEpoch(
            case_id=source_id,
            audio_epoch=int(scalar_ints[0]),
            normal_probs=[row.copy() for row in normal],
            normal_frontiers=[int(value) for value in data["normal_frontiers"]],
            frame_wall_ns=[int(value) for value in data["frame_wall_ns"]],
            tail_probs=[row.copy() for row in tail],
            epoch_end_count=int(scalar_ints[1]),
            finalize_wall_ns=int(scalar_ints[2]),
            chunk_observed_counts=[int(value) for value in data["chunk_observed_counts"]],
            chunk_wall_seconds=[float(value) for value in data["chunk_wall_seconds"]],
            cpu_seconds=float(scalar_floats[0]),
            wall_seconds=float(scalar_floats[1]),
            length_samples=int(scalar_ints[3]),
        )
    if len(capture.normal_probs) != len(capture.normal_frontiers):
        raise Phase4SignalError(f"LS cache frame/frontier mismatch: {source_id}")
    return capture


def save_ls_capture(
    cache_root: Path,
    contract: dict[str, Any],
    checkpoint: str,
    source: AudioSource,
    capture: LSCaptureEpoch,
) -> dict[str, Any]:
    npz_path, metadata_path = _ls_cache_paths(cache_root, contract, checkpoint, source)
    payload = _encode_ls_capture(capture)
    metadata = {
        "schema_version": "turn_episode_phase4_ls_cache.v1",
        "contract_sha256": contract["contract_sha256"],
        "source_id": source.source_id,
        "wav_sha256": source.wav_sha256,
        "duration_samples": source.duration_samples,
        "payload_sha256": sha256_bytes(payload),
        "payload_size_bytes": len(payload),
        "normal_frame_count": len(capture.normal_probs),
        "tail_frame_count": len(capture.tail_probs),
        "track_count": capture.track_count,
    }
    metadata["capture_content_sha256"] = content_hash(metadata)
    atomic_write_bytes(npz_path, payload)
    atomic_write_json(metadata_path, metadata)
    return {**metadata, "path": str(npz_path), "metadata_path": str(metadata_path)}


def load_ls_capture(
    cache_root: Path,
    contract: dict[str, Any],
    checkpoint: str,
    source: AudioSource,
) -> tuple[LSCaptureEpoch, dict[str, Any]] | None:
    npz_path, metadata_path = _ls_cache_paths(cache_root, contract, checkpoint, source)
    if not npz_path.is_file() or not metadata_path.is_file():
        return None
    metadata = read_json(metadata_path)
    expected = {
        "contract_sha256": contract["contract_sha256"],
        "source_id": source.source_id,
        "wav_sha256": source.wav_sha256,
        "duration_samples": source.duration_samples,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise Phase4SignalError(f"LS cache identity mismatch {source.source_id}:{key}")
    payload = npz_path.read_bytes()
    if sha256_bytes(payload) != metadata["payload_sha256"]:
        raise Phase4SignalError(f"LS cache payload hash mismatch: {source.source_id}")
    capture = _decode_ls_capture(payload, source.source_id)
    if capture.length_samples != source.duration_samples:
        raise Phase4SignalError(f"LS cache duration mismatch: {source.source_id}")
    return capture, {**metadata, "path": str(npz_path), "metadata_path": str(metadata_path)}


def _encode_eres_shard(
    windows: Sequence[tuple[int, int]],
    embeddings: np.ndarray,
    acoustic_shadows: np.ndarray,
    acoustic_log_rms: np.ndarray,
) -> tuple[bytes, bytes]:
    window_array = np.asarray(windows, dtype="<i8")
    embedding_array = np.asarray(embeddings, dtype="<f4")
    shadow_array = np.asarray(acoustic_shadows, dtype="<f4")
    rms_array = np.asarray(acoustic_log_rms, dtype="<f4")
    header = {
        "schema_version": "turn_episode_phase4_eres_binary.v1",
        "row_count": len(windows),
        "window_dtype": "<i8",
        "window_shape": list(window_array.shape),
        "embedding_dtype": "<f4",
        "embedding_shape": list(embedding_array.shape),
        "shadow_dtype": "<f4",
        "shadow_shape": list(shadow_array.shape),
        "rms_dtype": "<f4",
        "rms_shape": list(rms_array.shape),
    }
    plain = (
        b"TURN_EPISODE_PHASE4_ERES_V1\n"
        + canonical_json(header).encode("utf-8")
        + b"\n"
        + window_array.tobytes(order="C")
        + embedding_array.tobytes(order="C")
        + shadow_array.tobytes(order="C")
        + rms_array.tobytes(order="C")
    )
    return plain, deterministic_gzip(plain)


def _decode_eres_shard(payload: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first, second, binary = payload.split(b"\n", 2)
    if first != b"TURN_EPISODE_PHASE4_ERES_V1":
        raise Phase4SignalError("ERes cache shard magic mismatch")
    header = json.loads(second)
    if header.get("schema_version") != "turn_episode_phase4_eres_binary.v1":
        raise Phase4SignalError("ERes cache shard schema mismatch")
    window_shape = tuple(int(value) for value in header["window_shape"])
    embedding_shape = tuple(int(value) for value in header["embedding_shape"])
    shadow_shape = tuple(int(value) for value in header["shadow_shape"])
    rms_shape = tuple(int(value) for value in header["rms_shape"])
    if (
        header.get("window_dtype") != "<i8"
        or header.get("embedding_dtype") != "<f4"
        or header.get("shadow_dtype") != "<f4"
        or header.get("rms_dtype") != "<f4"
        or window_shape != (int(header["row_count"]), 2)
        or embedding_shape != (int(header["row_count"]), 192)
        or shadow_shape != (int(header["row_count"]), 80)
        or rms_shape != (int(header["row_count"]),)
    ):
        raise Phase4SignalError("ERes cache shard tensor contract mismatch")
    window_bytes = int(np.prod(window_shape, dtype=np.int64)) * 8
    embedding_bytes = int(np.prod(embedding_shape, dtype=np.int64)) * 4
    shadow_bytes = int(np.prod(shadow_shape, dtype=np.int64)) * 4
    rms_bytes = int(np.prod(rms_shape, dtype=np.int64)) * 4
    if len(binary) != window_bytes + embedding_bytes + shadow_bytes + rms_bytes:
        raise Phase4SignalError("ERes cache shard byte length mismatch")
    windows = np.frombuffer(binary[:window_bytes], dtype="<i8").reshape(window_shape).copy()
    embedding_end = window_bytes + embedding_bytes
    shadow_end = embedding_end + shadow_bytes
    embeddings = (
        np.frombuffer(binary[window_bytes:embedding_end], dtype="<f4")
        .reshape(embedding_shape)
        .copy()
    )
    shadows = (
        np.frombuffer(binary[embedding_end:shadow_end], dtype="<f4").reshape(shadow_shape).copy()
    )
    rms = np.frombuffer(binary[shadow_end:], dtype="<f4").reshape(rms_shape).copy()
    return windows, embeddings, shadows, rms


def _eres_shard_chunks(
    windows: Sequence[tuple[int, int]],
    embeddings: np.ndarray,
    acoustic_shadows: np.ndarray,
    acoustic_log_rms: np.ndarray,
) -> list[tuple[list[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray, bytes, bytes]]:
    bytes_per_row = 2 * 8 + 192 * 4 + 80 * 4 + 4
    rows_per_chunk = max(1, ERES_CACHE_SHARD_TARGET_BYTES // bytes_per_row)
    pending = [
        (
            list(windows[start : start + rows_per_chunk]),
            embeddings[start : start + rows_per_chunk],
            acoustic_shadows[start : start + rows_per_chunk],
            acoustic_log_rms[start : start + rows_per_chunk],
        )
        for start in range(0, len(windows), rows_per_chunk)
    ]
    result: list[tuple[list[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray, bytes, bytes]] = (
        []
    )
    while pending:
        chunk_windows, chunk_embeddings, chunk_shadows, chunk_rms = pending.pop(0)
        plain, compressed = _encode_eres_shard(
            chunk_windows, chunk_embeddings, chunk_shadows, chunk_rms
        )
        if len(compressed) > DETAIL_SHARD_LIMIT:
            if len(chunk_windows) == 1:
                raise Phase4SignalError("single ERes cache row exceeds shard limit")
            middle = len(chunk_windows) // 2
            pending.insert(
                0,
                (
                    chunk_windows[middle:],
                    chunk_embeddings[middle:],
                    chunk_shadows[middle:],
                    chunk_rms[middle:],
                ),
            )
            pending.insert(
                0,
                (
                    chunk_windows[:middle],
                    chunk_embeddings[:middle],
                    chunk_shadows[:middle],
                    chunk_rms[:middle],
                ),
            )
            continue
        result.append(
            (chunk_windows, chunk_embeddings, chunk_shadows, chunk_rms, plain, compressed)
        )
    return result


def save_eres_embeddings(
    cache_root: Path,
    contract: dict[str, Any],
    checkpoint: str,
    source: AudioSource,
    windows: Sequence[tuple[int, int]],
    embeddings: np.ndarray,
    acoustic_shadows: np.ndarray,
    acoustic_log_rms: np.ndarray,
    service_seconds: Sequence[float],
) -> dict[str, Any]:
    base_path, metadata_path = _eres_cache_paths(cache_root, contract, checkpoint, source)
    embedding_hashes = [
        sha256_bytes(np.asarray(vector, dtype="<f4").tobytes(order="C")) for vector in embeddings
    ]
    shadow_hashes = [
        sha256_bytes(
            np.asarray(shadow, dtype="<f4").tobytes(order="C")
            + np.asarray(rms, dtype="<f4").tobytes(order="C")
        )
        for shadow, rms in zip(acoustic_shadows, acoustic_log_rms)
    ]
    window_rows = [
        {
            "start": int(window[0]),
            "end": int(window[1]),
            "embedding_sha256": embedding_digest,
            "acoustic_shadow_sha256": shadow_digest,
        }
        for window, embedding_digest, shadow_digest in zip(windows, embedding_hashes, shadow_hashes)
    ]
    shard_rows: list[dict[str, Any]] = []
    for index, (chunk_windows, _, _, _, plain, compressed) in enumerate(
        _eres_shard_chunks(windows, embeddings, acoustic_shadows, acoustic_log_rms)
    ):
        shard_path = base_path.with_name(f"{base_path.name}.{index:04d}.bin.gz")
        atomic_write_bytes(shard_path, compressed)
        shard_rows.append(
            {
                "shard_index": index,
                "path": shard_path.name,
                "row_count": len(chunk_windows),
                "first_window": list(chunk_windows[0]),
                "last_window": list(chunk_windows[-1]),
                "content_sha256": sha256_bytes(plain),
                "byte_sha256": sha256_bytes(compressed),
                "size_bytes": len(compressed),
            }
        )
    payload_sha256 = content_hash(
        {
            "shards": [
                {key: value for key, value in row.items() if key != "path"} for row in shard_rows
            ]
        }
    )
    metadata = {
        "schema_version": "turn_episode_phase4_eres_cache.v2",
        "contract_sha256": contract["contract_sha256"],
        "source_id": source.source_id,
        "wav_sha256": source.wav_sha256,
        "duration_samples": source.duration_samples,
        "payload_sha256": payload_sha256,
        "payload_size_bytes": sum(int(row["size_bytes"]) for row in shard_rows),
        "window_count": len(windows),
        "window_rows_sha256": sha256_bytes(
            b"".join(canonical_json(row).encode("utf-8") + b"\n" for row in window_rows)
        ),
        "service_seconds": {
            "count": len(service_seconds),
            "mean": float(np.mean(service_seconds)) if service_seconds else None,
            "p50": float(np.percentile(service_seconds, 50)) if service_seconds else None,
            "p95": float(np.percentile(service_seconds, 95)) if service_seconds else None,
        },
        "shard_count": len(shard_rows),
        "shards": shard_rows,
    }
    metadata["capture_content_sha256"] = content_hash(metadata)
    atomic_write_json(metadata_path, metadata)
    return {
        **metadata,
        "paths": [str(metadata_path.parent / str(row["path"])) for row in shard_rows],
        "metadata_path": str(metadata_path),
    }


def load_eres_embeddings(
    cache_root: Path,
    contract: dict[str, Any],
    checkpoint: str,
    source: AudioSource,
    expected_windows: Sequence[tuple[int, int]],
) -> (
    tuple[
        dict[tuple[int, int], np.ndarray],
        dict[tuple[int, int], tuple[np.ndarray, float]],
        dict[str, Any],
    ]
    | None
):
    _, metadata_path = _eres_cache_paths(cache_root, contract, checkpoint, source)
    if not metadata_path.is_file():
        return None
    metadata = read_json(metadata_path)
    expected = {
        "contract_sha256": contract["contract_sha256"],
        "source_id": source.source_id,
        "wav_sha256": source.wav_sha256,
        "duration_samples": source.duration_samples,
        "window_count": len(expected_windows),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise Phase4SignalError(f"ERes cache identity mismatch {source.source_id}:{key}")
    shard_rows = metadata.get("shards")
    if (
        metadata.get("schema_version") != "turn_episode_phase4_eres_cache.v2"
        or not isinstance(shard_rows, list)
        or len(shard_rows) != int(metadata.get("shard_count", -1))
    ):
        raise Phase4SignalError(f"ERes cache shard inventory mismatch: {source.source_id}")
    if sum(int(row["row_count"]) for row in shard_rows) != len(expected_windows) or sum(
        int(row["size_bytes"]) for row in shard_rows
    ) != int(metadata.get("payload_size_bytes", -1)):
        raise Phase4SignalError(f"ERes cache shard totals mismatch: {source.source_id}")
    aggregate_hash = content_hash(
        {
            "shards": [
                {key: value for key, value in row.items() if key != "path"} for row in shard_rows
            ]
        }
    )
    if aggregate_hash != metadata["payload_sha256"]:
        raise Phase4SignalError(f"ERes cache payload hash mismatch: {source.source_id}")
    window_parts: list[np.ndarray] = []
    embedding_parts: list[np.ndarray] = []
    shadow_parts: list[np.ndarray] = []
    rms_parts: list[np.ndarray] = []
    paths: list[str] = []
    for index, row in enumerate(shard_rows):
        if int(row.get("shard_index", -1)) != index:
            raise Phase4SignalError(f"ERes cache shard order mismatch: {source.source_id}")
        shard_path = metadata_path.parent / str(row["path"])
        compressed = shard_path.read_bytes()
        if len(compressed) > DETAIL_SHARD_LIMIT or len(compressed) != int(row["size_bytes"]):
            raise Phase4SignalError(f"ERes cache shard size mismatch: {source.source_id}")
        if sha256_bytes(compressed) != row["byte_sha256"]:
            raise Phase4SignalError(f"ERes cache shard byte hash mismatch: {source.source_id}")
        plain = gzip.decompress(compressed)
        if sha256_bytes(plain) != row["content_sha256"]:
            raise Phase4SignalError(f"ERes cache shard content hash mismatch: {source.source_id}")
        shard_windows, shard_embeddings, shard_shadows, shard_rms = _decode_eres_shard(plain)
        if shard_windows.shape[0] != int(row["row_count"]):
            raise Phase4SignalError(f"ERes cache shard count mismatch: {source.source_id}")
        if shard_windows.shape[0] and (
            shard_windows[0].tolist() != row["first_window"]
            or shard_windows[-1].tolist() != row["last_window"]
        ):
            raise Phase4SignalError(f"ERes cache shard bounds mismatch: {source.source_id}")
        window_parts.append(shard_windows)
        embedding_parts.append(shard_embeddings)
        shadow_parts.append(shard_shadows)
        rms_parts.append(shard_rms)
        paths.append(str(shard_path))
    windows = (
        np.concatenate(window_parts, axis=0) if window_parts else np.zeros((0, 2), dtype=np.int64)
    )
    embeddings = (
        np.concatenate(embedding_parts, axis=0)
        if embedding_parts
        else np.zeros((0, 192), dtype=np.float32)
    )
    shadows = (
        np.concatenate(shadow_parts, axis=0)
        if shadow_parts
        else np.zeros((0, 80), dtype=np.float32)
    )
    rms = np.concatenate(rms_parts, axis=0) if rms_parts else np.zeros((0,), dtype=np.float32)
    actual_windows = [(int(row[0]), int(row[1])) for row in windows]
    if actual_windows != list(expected_windows):
        raise Phase4SignalError(f"ERes cache window mismatch: {source.source_id}")
    if (
        embeddings.shape != (len(expected_windows), 192)
        or shadows.shape != (len(expected_windows), 80)
        or rms.shape != (len(expected_windows),)
        or not np.all(np.isfinite(embeddings))
        or not np.all(np.isfinite(shadows))
        or not np.all(np.isfinite(rms))
    ):
        raise Phase4SignalError(f"ERes cache embedding shape mismatch: {source.source_id}")
    window_rows = [
        {
            "start": start,
            "end": end,
            "embedding_sha256": sha256_bytes(np.asarray(embedding, dtype="<f4").tobytes(order="C")),
            "acoustic_shadow_sha256": sha256_bytes(
                np.asarray(shadow, dtype="<f4").tobytes(order="C")
                + np.asarray(log_rms, dtype="<f4").tobytes(order="C")
            ),
        }
        for (start, end), embedding, shadow, log_rms in zip(
            actual_windows, embeddings, shadows, rms
        )
    ]
    if sha256_bytes(
        b"".join(canonical_json(row).encode("utf-8") + b"\n" for row in window_rows)
    ) != metadata.get("window_rows_sha256"):
        raise Phase4SignalError(f"ERes cache row hash mismatch: {source.source_id}")
    return (
        dict(zip(actual_windows, embeddings)),
        dict(zip(actual_windows, ((shadow, float(value)) for shadow, value in zip(shadows, rms)))),
        {
            **metadata,
            "paths": paths,
            "metadata_path": str(metadata_path),
        },
    )


def source_by_wav(inputs: Phase4Inputs) -> dict[str, AudioSource]:
    result: dict[str, AudioSource] = {}
    for source in inputs.sources.values():
        existing = result.get(source.wav_sha256)
        if existing is not None and existing.duration_samples != source.duration_samples:
            raise Phase4SignalError(
                f"duplicate WAV identity has multiple durations: {source.wav_sha256}"
            )
        if existing is None or source.source_id < existing.source_id:
            result[source.wav_sha256] = source
    return result


def run_ls_cache(
    inputs: Phase4Inputs,
    args: argparse.Namespace,
    *,
    source_limit: int | None = None,
    minimum_duration_samples: int = 0,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    contracts: dict[str, dict[str, Any]] = {}
    inventory: dict[str, dict[str, Any]] = {}
    sources = sorted(
        (
            source
            for source in inputs.sources.values()
            if source.duration_samples >= minimum_duration_samples
        ),
        key=(
            (lambda source: (source.duration_samples, source.source_id))
            if source_limit is not None
            else (lambda source: source.source_id)
        ),
    )
    if source_limit is not None:
        sources = sources[:source_limit]
    for checkpoint, info in LS_EEND_VARIANTS.items():
        model = args.hf_root / str(info["dir"]) / str(info["onnx"])
        sidecar = args.hf_root / str(info["dir"]) / str(info["sidecar"])
        runtime = LSEENDCapture(
            model,
            load_sidecar_metadata(sidecar),
            checkpoint_variant=checkpoint,
        )
        contract = cache_contract(
            inputs,
            family="ls_eend",
            checkpoint=checkpoint,
            checkpoint_sha256=str(info["onnx_sha256"]),
            sidecar_sha256=str(info["sidecar_sha256"]),
            tensor=tensor_contract(runtime._session),
        )
        contracts[checkpoint] = contract
        rows: list[dict[str, Any]] = []
        for source in sources:
            cached = load_ls_capture(args.cache_root, contract, checkpoint, source)
            if cached is not None:
                _, evidence = cached
                rows.append({**evidence, "cache_hit": True})
                continue
            samples = read_wav(source)
            epoch = int(source.wav_sha256[:8], 16)
            capture = runtime.run_case(samples, case_id=source.source_id, audio_epoch=epoch)
            evidence = save_ls_capture(
                args.cache_root,
                contract,
                checkpoint,
                source,
                capture,
            )
            rows.append({**evidence, "cache_hit": False})
        inventory[checkpoint] = {
            "contract": contract,
            "source_count": len(rows),
            "cache_hit_count": sum(bool(row["cache_hit"]) for row in rows),
            "payload_bytes": sum(int(row["payload_size_bytes"]) for row in rows),
            "normal_frame_count": sum(int(row["normal_frame_count"]) for row in rows),
            "tail_frame_count": sum(int(row["tail_frame_count"]) for row in rows),
            "sources": rows,
        }
    return contracts, inventory


def run_eres_cache(
    inputs: Phase4Inputs,
    args: argparse.Namespace,
    *,
    source_limit: int | None = None,
    window_limit: int | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    contracts: dict[str, dict[str, Any]] = {}
    inventory: dict[str, dict[str, Any]] = {}
    sources = source_by_wav(inputs)
    source_rows = [
        (
            sources[wav],
            sorted(windows),
        )
        for wav, windows in inputs.embedding_windows.items()
    ]
    source_rows.sort(
        key=(
            (lambda item: (item[0].duration_samples, item[0].source_id))
            if source_limit is not None
            else (lambda item: item[0].source_id)
        )
    )
    if source_limit is not None:
        source_rows = source_rows[:source_limit]
    for checkpoint, expected_model_hash in ERES_MODEL_SHA256.items():
        model = args.eres_onnx_root / str(ERES_CHECKPOINTS[checkpoint]["onnx"])
        runtime = EresEmbeddingRuntime(str(model))
        contract = cache_contract(
            inputs,
            family="eres2netv2",
            checkpoint=checkpoint,
            checkpoint_sha256=expected_model_hash,
            sidecar_sha256=None,
            tensor=tensor_contract(runtime._session),
        )
        contracts[checkpoint] = contract
        rows: list[dict[str, Any]] = []
        pending: list[tuple[AudioSource, list[tuple[int, int]]]] = []
        for source, all_windows in source_rows:
            windows = list(all_windows[:window_limit] if window_limit is not None else all_windows)
            cached = load_eres_embeddings(
                args.cache_root,
                contract,
                checkpoint,
                source,
                windows,
            )
            if cached is not None:
                _, _, evidence = cached
                rows.append({**evidence, "cache_hit": True})
                continue
            pending.append((source, windows))

        def execute_partition(
            worker_index: int,
            partition: list[tuple[AudioSource, list[tuple[int, int]]]],
        ) -> list[dict[str, Any]]:
            if source_limit is not None or window_limit is not None:
                if runtime is None:
                    raise Phase4SignalError("ERes sequential runtime missing")
                worker_runtime = runtime
            else:
                worker_runtime = EresEmbeddingRuntime(str(model))
            completed: list[dict[str, Any]] = []
            processed = 0
            for source, windows in partition:
                samples = read_wav(source)
                embeddings: list[np.ndarray] = []
                acoustic_shadows: list[np.ndarray] = []
                acoustic_log_rms: list[float] = []
                service: list[float] = []
                for start, end in windows:
                    begin = time.perf_counter()
                    window_samples = samples[start:end]
                    fbank = kaldi_fbank_numpy(window_samples)
                    shadow = normalized(fbank.mean(axis=0)) if fbank.size else None
                    if shadow is None:
                        raise Phase4SignalError(
                            f"invalid ERes acoustic shadow {checkpoint}:{source.source_id}:{start}-{end}"
                        )
                    centered = fbank - fbank.mean(axis=0, keepdims=True)
                    output = worker_runtime._session.run(
                        worker_runtime._output_names,
                        {"fbank": centered[None, :, :].astype(np.float32)},
                    )[0]
                    vector = normalized(np.asarray(output[0], dtype=np.float32))
                    service.append(time.perf_counter() - begin)
                    if vector is None or vector.size != 192:
                        raise Phase4SignalError(
                            f"invalid ERes embedding {checkpoint}:{source.source_id}:{start}-{end}"
                        )
                    embeddings.append(vector)
                    acoustic_shadows.append(shadow)
                    rms = float(np.sqrt(np.mean(np.square(window_samples, dtype=np.float64))))
                    acoustic_log_rms.append(math.log(max(rms, 1e-8)))
                    processed += 1
                    if processed % 1000 == 0:
                        print(
                            f"phase4 eres {checkpoint} worker={worker_index} windows={processed}",
                            flush=True,
                        )
                matrix = (
                    np.stack(embeddings).astype(np.float32, copy=False)
                    if embeddings
                    else np.zeros((0, 192), dtype=np.float32)
                )
                shadow_matrix = (
                    np.stack(acoustic_shadows).astype(np.float32, copy=False)
                    if acoustic_shadows
                    else np.zeros((0, 80), dtype=np.float32)
                )
                evidence = save_eres_embeddings(
                    args.cache_root,
                    contract,
                    checkpoint,
                    source,
                    windows,
                    matrix,
                    shadow_matrix,
                    np.asarray(acoustic_log_rms, dtype=np.float32),
                    service,
                )
                completed.append({**evidence, "cache_hit": False})
            return completed

        if pending and (source_limit is not None or window_limit is not None):
            rows.extend(execute_partition(0, pending))
        elif pending:
            runtime = None
            gc.collect()
            partition_count = min(ERES_PARALLEL_WORKERS, len(pending))
            partitions: list[list[tuple[AudioSource, list[tuple[int, int]]]]] = [
                [] for _ in range(partition_count)
            ]
            partition_sizes = [0] * partition_count
            for item in sorted(pending, key=lambda value: (-len(value[1]), value[0].source_id)):
                index = min(
                    range(partition_count), key=lambda value: (partition_sizes[value], value)
                )
                partitions[index].append(item)
                partition_sizes[index] += len(item[1])
            with ThreadPoolExecutor(max_workers=partition_count) as executor:
                futures = {
                    executor.submit(execute_partition, index, partition): index
                    for index, partition in enumerate(partitions)
                }
                for future in as_completed(futures):
                    rows.extend(future.result())
        rows.sort(key=lambda row: str(row["source_id"]))
        inventory[checkpoint] = {
            "contract": contract,
            "source_count": len(rows),
            "cache_hit_count": sum(bool(row["cache_hit"]) for row in rows),
            "payload_bytes": sum(int(row["payload_size_bytes"]) for row in rows),
            "window_count": sum(int(row["window_count"]) for row in rows),
            "sources": rows,
        }
    return contracts, inventory


def episode_maps(
    inputs: Phase4Inputs,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    episodes = {str(row["episode_id"]): row for row in inputs.episodes}
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in inputs.candidates:
        source_id = inputs.source_by_episode[str(candidate["episode_id"])]
        candidates[source_id].append(candidate)
    for values in candidates.values():
        values.sort(key=lambda row: str(row["candidate_id"]))
    return episodes, dict(candidates)


def ls_frame_coordinates(frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    indexes = np.arange(frame_count, dtype=np.int64)
    return 14_431 + 1_600 * indexes, 15_806 + 1_600 * indexes


def select_ls_observation(
    values: np.ndarray,
    centers: np.ndarray,
    availability: np.ndarray,
    coordinate: int,
    deadline: int,
) -> tuple[float, int | None, int | None, bool]:
    mask = (centers >= coordinate) & (availability <= deadline) & np.isfinite(values)
    indices = np.flatnonzero(mask)
    if not indices.size:
        return 0.0, None, None, True
    scores = values[indices]
    maximum = float(np.max(scores))
    selected = int(indices[np.flatnonzero(scores == maximum)[0]])
    return maximum, int(centers[selected]), int(availability[selected]), False


def score_ls_checkpoint(
    inputs: Phase4Inputs,
    args: argparse.Namespace,
    checkpoint: str,
    contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episodes, candidates_by_source = episode_maps(inputs)
    rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for source_id in sorted(candidates_by_source):
        source = inputs.sources[source_id]
        cached = load_ls_capture(args.cache_root, contract, checkpoint, source)
        if cached is None:
            raise Phase4SignalError(f"missing LS cache {checkpoint}:{source_id}")
        capture, evidence = cached
        probabilities = (
            np.stack(capture.normal_probs).astype(np.float32, copy=False)
            if capture.normal_probs
            else np.zeros((0, capture.track_count), dtype=np.float32)
        )
        centers, availability = ls_frame_coordinates(probabilities.shape[0])
        scalars = ls_scalar_series(probabilities)
        samples = read_wav(source)
        runtime_rows.append(
            {
                "source_id": source_id,
                "audio_seconds": source.duration_samples / 16000.0,
                "compute_seconds": capture.wall_seconds,
                "cpu_seconds": capture.cpu_seconds,
                "normal_frames": len(capture.normal_probs),
                "tail_frames": len(capture.tail_probs),
                "cache_payload_sha256": evidence["payload_sha256"],
            }
        )
        acoustic_by_candidate: dict[tuple[str, int], dict[str, float | None]] = {}
        for candidate in candidates_by_source[source_id]:
            episode = episodes[str(candidate["episode_id"])]
            bounds = episode["bounds"]
            coordinate = int(candidate["coordinate"])
            for horizon, support in LS_ACOUSTIC_SUPPORT_BY_HORIZON.items():
                if coordinate - support < int(bounds["warm_start"]) or coordinate + support > int(
                    bounds["tail_end"]
                ):
                    acoustic_by_candidate[(str(candidate["candidate_id"]), horizon)] = {
                        key: None for key in ACOUSTIC_EXTRACTORS
                    }
                else:
                    acoustic_by_candidate[(str(candidate["candidate_id"]), horizon)] = (
                        acoustic_scores(
                            samples[coordinate - support : coordinate],
                            samples[coordinate : coordinate + support],
                        )
                    )
        for candidate in candidates_by_source[source_id]:
            coordinate = int(candidate["coordinate"])
            for horizon in HORIZONS_MS:
                deadline = coordinate + horizon * 16
                acoustic = acoustic_by_candidate[(str(candidate["candidate_id"]), horizon)]
                for extractor, values in scalars.items():
                    score, center, frontier, missing = select_ls_observation(
                        values,
                        centers,
                        availability,
                        coordinate,
                        deadline,
                    )
                    row_key = f"ls|{checkpoint}|{extractor}|H{horizon}|{candidate['candidate_id']}"
                    rows.append(
                        {
                            "row_key": row_key,
                            "row_kind": "candidate_signal",
                            "family": "ls_eend",
                            "checkpoint": checkpoint,
                            "signal_extractor_id": f"{extractor}:{checkpoint}:H{horizon}",
                            "base_extractor_id": extractor,
                            "horizon_ms": horizon,
                            "candidate_id": candidate["candidate_id"],
                            "candidate_class": candidate["class"],
                            "candidate_kind": candidate["kind"],
                            "episode_id": candidate["episode_id"],
                            "source_id": source_id,
                            "block_id": candidate["block_id"],
                            "corpus": candidate["corpus"],
                            "language": candidate["language"],
                            "stress": candidate["stress"],
                            "boundary_source_sample": coordinate,
                            "selected_center_source_sample": center,
                            "observation_frontier": frontier,
                            "deadline_source_sample": deadline,
                            "neural_score": score,
                            "missing": missing,
                            "missing_reason": "no_ordinary_frame_by_horizon" if missing else None,
                            "acoustic_scores": acoustic,
                            "tail_used": False,
                            "cache_payload_sha256": evidence["payload_sha256"],
                        }
                    )
    audio_seconds = sum(float(row["audio_seconds"]) for row in runtime_rows)
    compute_seconds = sum(float(row["compute_seconds"]) for row in runtime_rows)
    report = {
        "schema_version": "turn_episode_phase4_ls_signal.v1",
        "checkpoint": checkpoint,
        "contract_sha256": contract["contract_sha256"],
        "source_count": len(runtime_rows),
        "candidate_row_count": len(rows),
        "audio_seconds": audio_seconds,
        "compute_seconds": compute_seconds,
        "realtime_factor": compute_seconds / audio_seconds if audio_seconds else None,
        "normal_frame_count": sum(int(row["normal_frames"]) for row in runtime_rows),
        "terminal_tail_frame_count": sum(int(row["tail_frames"]) for row in runtime_rows),
        "tail_frames_excluded_from_scoring": True,
        "source_runtime": runtime_rows,
    }
    return rows, report


def _state_hash(payload: Any) -> str:
    def encode(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": sha256_bytes(value.tobytes(order="C")),
            }
        if isinstance(value, list):
            return [encode(item) for item in value]
        if isinstance(value, tuple):
            return [encode(item) for item in value]
        if isinstance(value, dict):
            return {key: encode(item) for key, item in sorted(value.items())}
        return value

    return content_hash({"state": encode(payload)})


def _regular_probe_windows(
    episode: dict[str, Any], window: int, step: int
) -> list[tuple[int, int]]:
    bounds = episode["bounds"]
    first = ceil_grid(int(bounds["warm_start"]) + window, step)
    return [(end - window, end) for end in range(first, int(bounds["tail_end"]) + 1, step)]


def _acoustic_shadow(samples: np.ndarray, window: tuple[int, int]) -> tuple[np.ndarray, float]:
    vector, rms = acoustic_payload(samples[window[0] : window[1]])
    if vector is None:
        raise Phase4SignalError(f"invalid acoustic shadow window {window}")
    return vector, rms


def _ema(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = normalized(0.10 * left + 0.90 * right)
    if result is None:
        raise Phase4SignalError("EMA produced zero state")
    return result


def _advance_anchor_state(
    state: dict[str, Any],
    mode: str,
    probe: np.ndarray,
    acoustic: tuple[np.ndarray, float],
    window: tuple[int, int],
) -> None:
    if state.get("anchor") is None:
        state.update(
            {
                "anchor": probe,
                "anchor_window": window,
                "shadow": acoustic[0],
                "shadow_rms": acoustic[1],
                "pending": None,
                "next_ordinal": 1,
            }
        )
        if mode == "prototype_memory_4":
            state["prototypes"] = [
                {
                    "ordinal": 0,
                    "embedding": probe,
                    "shadow": acoustic[0],
                    "shadow_rms": acoustic[1],
                    "window": window,
                }
            ]
        return
    if mode == "stable_no_update":
        return
    if mode == "stable_ema":
        similarity = cosine_similarity(state["anchor"], probe)
        if similarity >= 0.70:
            state["anchor"] = _ema(state["anchor"], probe)
            state["shadow"] = _ema(state["shadow"], acoustic[0])
            state["shadow_rms"] = 0.10 * float(state["shadow_rms"]) + 0.90 * acoustic[1]
            state["anchor_window"] = window
        return
    if mode == "confirmed_anchor":
        similarity = cosine_similarity(state["anchor"], probe)
        pending = state.get("pending")
        if similarity < 0.50:
            if pending is not None:
                mutual = cosine_similarity(pending["embedding"], probe)
                if mutual >= 0.50:
                    state["anchor"] = pending["embedding"]
                    state["shadow"] = pending["shadow"]
                    state["shadow_rms"] = pending["shadow_rms"]
                    state["anchor_window"] = pending["window"]
                    state["pending"] = None
                    return
            state["pending"] = {
                "embedding": probe,
                "shadow": acoustic[0],
                "shadow_rms": acoustic[1],
                "window": window,
            }
        else:
            state["pending"] = None
            if similarity >= 0.70:
                state["anchor"] = _ema(state["anchor"], probe)
                state["shadow"] = _ema(state["shadow"], acoustic[0])
                state["shadow_rms"] = 0.10 * float(state["shadow_rms"]) + 0.90 * acoustic[1]
                state["anchor_window"] = window
        return
    prototypes = state["prototypes"]
    selected = max(
        prototypes,
        key=lambda item: (cosine_similarity(item["embedding"], probe), -item["ordinal"]),
    )
    similarity = cosine_similarity(selected["embedding"], probe)
    pending = state.get("pending")
    if similarity >= 0.70:
        selected["embedding"] = _ema(selected["embedding"], probe)
        selected["shadow"] = _ema(selected["shadow"], acoustic[0])
        selected["shadow_rms"] = 0.10 * float(selected["shadow_rms"]) + 0.90 * acoustic[1]
        selected["window"] = window
        state["pending"] = None
    elif similarity >= 0.50:
        state["pending"] = None
    elif pending is None:
        state["pending"] = {
            "embedding": probe,
            "shadow": acoustic[0],
            "shadow_rms": acoustic[1],
            "window": window,
        }
    elif cosine_similarity(pending["embedding"], probe) >= 0.50:
        ordinal = int(state["next_ordinal"])
        state["next_ordinal"] = ordinal + 1
        created = {
            "ordinal": ordinal,
            "embedding": pending["embedding"],
            "shadow": pending["shadow"],
            "shadow_rms": pending["shadow_rms"],
            "window": pending["window"],
        }
        if len(prototypes) >= 4:
            oldest = min(prototypes, key=lambda item: item["ordinal"])
            prototypes.remove(oldest)
        prototypes.append(created)
        state["pending"] = None
    else:
        state["pending"] = {
            "embedding": probe,
            "shadow": acoustic[0],
            "shadow_rms": acoustic[1],
            "window": window,
        }


def _measure_anchor_state(
    state: dict[str, Any],
    mode: str,
    probe: np.ndarray,
    acoustic: tuple[np.ndarray, float],
) -> tuple[float | None, dict[str, float | None], int | None]:
    if state.get("anchor") is None:
        return None, {key: None for key in ACOUSTIC_EXTRACTORS}, None
    if mode == "prototype_memory_4":
        selected = max(
            state["prototypes"],
            key=lambda item: (cosine_similarity(item["embedding"], probe), -item["ordinal"]),
        )
        neural = 1.0 - cosine_similarity(selected["embedding"], probe)
        rms = abs(float(selected["shadow_rms"]) - acoustic[1])
        flux = 1.0 - cosine_similarity(selected["shadow"], acoustic[0])
        return (
            neural,
            {
                "acoustic_log_rms_delta.v1": rms,
                "acoustic_logmel_flux.v1": flux,
            },
            int(selected["ordinal"]),
        )
    neural = 1.0 - cosine_similarity(state["anchor"], probe)
    rms = abs(float(state["shadow_rms"]) - acoustic[1])
    flux = 1.0 - cosine_similarity(state["shadow"], acoustic[0])
    return (
        neural,
        {
            "acoustic_log_rms_delta.v1": rms,
            "acoustic_logmel_flux.v1": flux,
        },
        None,
    )


def score_eres_checkpoint(
    inputs: Phase4Inputs,
    args: argparse.Namespace,
    checkpoint: str,
    contract: dict[str, Any],
    state_equivalence: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episodes, candidates_by_source = episode_maps(inputs)
    source_for_wav = source_by_wav(inputs)
    embeddings_by_wav: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    shadows_by_wav: dict[str, dict[tuple[int, int], tuple[np.ndarray, float]]] = {}
    cache_evidence_by_wav: dict[str, dict[str, Any]] = {}
    for wav, windows in sorted(inputs.embedding_windows.items()):
        source = source_for_wav[wav]
        expected = sorted(windows)
        cached = load_eres_embeddings(
            args.cache_root,
            contract,
            checkpoint,
            source,
            expected,
        )
        if cached is None:
            raise Phase4SignalError(f"missing ERes cache {checkpoint}:{source.source_id}")
        (
            embeddings_by_wav[wav],
            shadows_by_wav[wav],
            cache_evidence_by_wav[wav],
        ) = cached
    rows: list[dict[str, Any]] = []
    state_diagnostics: Counter[str] = Counter()
    state_dispositions = {
        (str(row["checkpoint"]), str(row["profile_class"])): str(row["disposition"])
        for row in state_equivalence["eres_profile_classes"]
    }
    for source_id in sorted(candidates_by_source):
        source = inputs.sources[source_id]
        embeddings = embeddings_by_wav[source.wav_sha256]
        shadows = shadows_by_wav[source.wav_sha256]
        cache_evidence = cache_evidence_by_wav[source.wav_sha256]
        by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in candidates_by_source[source_id]:
            by_episode[str(candidate["episode_id"])].append(candidate)
        for episode_id, candidates in sorted(by_episode.items()):
            episode = episodes[episode_id]
            bounds = episode["bounds"]
            for candidate in candidates:
                coordinate = int(candidate["coordinate"])
                for horizon in HORIZONS_MS:
                    deadline = coordinate + horizon * 16
                    for window in ADJACENT_WINDOWS:
                        left = embeddings.get((coordinate - window, coordinate))
                        right = embeddings.get((coordinate, coordinate + window))
                        left_shadow = shadows.get((coordinate - window, coordinate))
                        right_shadow = shadows.get((coordinate, coordinate + window))
                        valid = (
                            left is not None
                            and right is not None
                            and left_shadow is not None
                            and right_shadow is not None
                            and coordinate + window <= deadline
                            and coordinate - window >= int(bounds["warm_start"])
                            and coordinate + window <= int(bounds["tail_end"])
                        )
                        neural = (
                            1.0 - cosine_similarity(left, right)
                            if valid and left is not None and right is not None
                            else 0.0
                        )
                        acoustic = (
                            {
                                "acoustic_log_rms_delta.v1": abs(
                                    float(left_shadow[1]) - float(right_shadow[1])
                                ),
                                "acoustic_logmel_flux.v1": 1.0
                                - cosine_similarity(left_shadow[0], right_shadow[0]),
                            }
                            if valid and left_shadow is not None and right_shadow is not None
                            else {key: None for key in ACOUSTIC_EXTRACTORS}
                        )
                        extractor = f"eres_adjacent_change.v1:{checkpoint}:W{window}:H{horizon}"
                        rows.append(
                            {
                                "row_key": f"eres|{checkpoint}|adjacent|W{window}|H{horizon}|{candidate['candidate_id']}",
                                "row_kind": "candidate_signal",
                                "family": "eres2netv2",
                                "checkpoint": checkpoint,
                                "signal_extractor_id": extractor,
                                "base_extractor_id": "eres_adjacent_change.v1",
                                "horizon_ms": horizon,
                                "window_samples": window,
                                "candidate_id": candidate["candidate_id"],
                                "candidate_class": candidate["class"],
                                "candidate_kind": candidate["kind"],
                                "episode_id": episode_id,
                                "source_id": source_id,
                                "block_id": candidate["block_id"],
                                "corpus": candidate["corpus"],
                                "language": candidate["language"],
                                "stress": candidate["stress"],
                                "boundary_source_sample": coordinate,
                                "observation_frontier": coordinate + window if valid else None,
                                "deadline_source_sample": deadline,
                                "neural_score": neural,
                                "missing": not valid,
                                "missing_reason": (
                                    None if valid else "window_or_horizon_unavailable"
                                ),
                                "acoustic_scores": acoustic,
                                "cache_payload_sha256": cache_evidence["payload_sha256"],
                            }
                        )
            for window in ANCHOR_WINDOWS:
                for step in STEPS:
                    regular = _regular_probe_windows(episode, window, step)
                    ordered_candidates = sorted(candidates, key=lambda row: int(row["coordinate"]))
                    for mode in ERES_STATES:
                        disposition = state_dispositions[(checkpoint, mode)]
                        if source.public and disposition == "source_prefix_required":
                            regular = [
                                (end - window, end)
                                for end in range(
                                    ceil_grid(window, step),
                                    int(episode["bounds"]["tail_end"]) + 1,
                                    step,
                                )
                            ]
                        else:
                            regular = _regular_probe_windows(episode, window, step)
                        state: dict[str, Any] = {"anchor": None, "pending": None}
                        probe_index = 0
                        for candidate in ordered_candidates:
                            coordinate = int(candidate["coordinate"])
                            while (
                                probe_index < len(regular) and regular[probe_index][1] <= coordinate
                            ):
                                probe_window = regular[probe_index]
                                probe = embeddings[probe_window]
                                shadow = shadows[probe_window]
                                _advance_anchor_state(state, mode, probe, shadow, probe_window)
                                probe_index += 1
                            measurement_window = (coordinate, coordinate + window)
                            measurement = embeddings.get(measurement_window)
                            measurement_shadow = (
                                shadows[measurement_window] if measurement is not None else None
                            )
                            neural: float | None = None
                            acoustic = {key: None for key in ACOUSTIC_EXTRACTORS}
                            selected_ordinal: int | None = None
                            state_before = _state_hash(state)
                            if measurement is not None and measurement_shadow is not None:
                                neural, acoustic, selected_ordinal = _measure_anchor_state(
                                    state, mode, measurement, measurement_shadow
                                )
                            confirmation_frontier: int | None = None
                            if mode == "confirmed_anchor" and neural is not None:
                                if neural <= 0.50:
                                    neural = None
                                    acoustic = {key: None for key in ACOUSTIC_EXTRACTORS}
                                else:
                                    next_probe = next(
                                        (
                                            item
                                            for item in regular[probe_index:]
                                            if item[1] > coordinate + window
                                        ),
                                        None,
                                    )
                                    if next_probe is None:
                                        neural = None
                                        acoustic = {key: None for key in ACOUSTIC_EXTRACTORS}
                                    else:
                                        next_embedding = embeddings[next_probe]
                                        anchor_score = 1.0 - cosine_similarity(
                                            state["anchor"], next_embedding
                                        )
                                        mutual = cosine_similarity(measurement, next_embedding)
                                        if anchor_score <= 0.50 or mutual < 0.50:
                                            neural = None
                                            acoustic = {key: None for key in ACOUSTIC_EXTRACTORS}
                                        else:
                                            confirmation_frontier = next_probe[1]
                            frontier = (
                                max(coordinate + window, confirmation_frontier or 0)
                                if neural is not None
                                else None
                            )
                            state_after = _state_hash(state)
                            state_diagnostics[f"{mode}:measurement"] += 1
                            for horizon in HORIZONS_MS:
                                deadline = coordinate + horizon * 16
                                valid = (
                                    neural is not None
                                    and frontier is not None
                                    and frontier <= deadline
                                )
                                base = {
                                    "stable_no_update": "eres_stable_anchor_change.v1",
                                    "stable_ema": "eres_stable_anchor_change.v1",
                                    "confirmed_anchor": "eres_confirmed_anchor_change.v1",
                                    "prototype_memory_4": "eres_prototype_change.v1",
                                }[mode]
                                extractor = (
                                    f"{base}:{checkpoint}:{mode}:W{window}:S{step}:H{horizon}"
                                )
                                rows.append(
                                    {
                                        "row_key": f"eres|{checkpoint}|{mode}|W{window}|S{step}|H{horizon}|{candidate['candidate_id']}",
                                        "row_kind": "candidate_signal",
                                        "family": "eres2netv2",
                                        "checkpoint": checkpoint,
                                        "signal_extractor_id": extractor,
                                        "base_extractor_id": base,
                                        "state_mode": mode,
                                        "horizon_ms": horizon,
                                        "window_samples": window,
                                        "step_samples": step,
                                        "candidate_id": candidate["candidate_id"],
                                        "candidate_class": candidate["class"],
                                        "candidate_kind": candidate["kind"],
                                        "episode_id": episode_id,
                                        "source_id": source_id,
                                        "block_id": candidate["block_id"],
                                        "corpus": candidate["corpus"],
                                        "language": candidate["language"],
                                        "stress": candidate["stress"],
                                        "boundary_source_sample": coordinate,
                                        "observation_frontier": frontier if valid else None,
                                        "deadline_source_sample": deadline,
                                        "neural_score": float(neural) if valid else 0.0,
                                        "missing": not valid,
                                        "missing_reason": (
                                            None if valid else "state_or_horizon_unavailable"
                                        ),
                                        "acoustic_scores": (
                                            acoustic
                                            if valid
                                            else {key: None for key in ACOUSTIC_EXTRACTORS}
                                        ),
                                        "pre_state_sha256": state_before,
                                        "post_state_sha256": state_after,
                                        "selected_prototype_ordinal": selected_ordinal,
                                        "confirmation_frontier": confirmation_frontier,
                                        "state_equivalence_disposition": disposition,
                                        "state_replay_origin_sample": (
                                            0
                                            if source.public
                                            and disposition == "source_prefix_required"
                                            else int(episode["bounds"]["warm_start"])
                                        ),
                                        "cache_payload_sha256": cache_evidence["payload_sha256"],
                                    }
                                )
    report = {
        "schema_version": "turn_episode_phase4_eres_signal.v1",
        "checkpoint": checkpoint,
        "contract_sha256": contract["contract_sha256"],
        "source_count": len(embeddings_by_wav),
        "candidate_row_count": len(rows),
        "state_diagnostics": dict(sorted(state_diagnostics.items())),
        "window_count": sum(len(values) for values in inputs.embedding_windows.values()),
        "cache_payload_bytes": sum(
            int(row["payload_size_bytes"]) for row in cache_evidence_by_wav.values()
        ),
    }
    return rows, report


def analyze_signal_rows(
    rows: Sequence[dict[str, Any]],
    inputs: Phase4Inputs,
    proposal_contract_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    registry = {row["signal_extractor_id"]: row for row in signal_registry()}
    by_extractor: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = str(row["signal_extractor_id"])
        candidate_id = str(row["candidate_id"])
        if candidate_id in by_extractor[key]:
            raise Phase4SignalError(f"duplicate candidate signal row: {key}:{candidate_id}")
        by_extractor[key][candidate_id] = row
    summaries: list[dict[str, Any]] = []
    acoustic_receipts: list[dict[str, Any]] = []
    for extractor_id in sorted(registry):
        declaration = registry[extractor_id]
        if declaration["gate_role"] != "hard_target":
            continue
        indexed = by_extractor.get(extractor_id, {})
        missing_positive = 0
        missing_negative = 0
        usable: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for pair in inputs.pairs:
            positive = indexed.get(str(pair["positive_id"]))
            negative = indexed.get(str(pair["negative_id"]))
            if positive is None or bool(positive["missing"]):
                missing_positive += 1
            if negative is None or bool(negative["missing"]):
                missing_negative += 1
            if (
                positive is None
                or negative is None
                or bool(positive["missing"])
                or bool(negative["missing"])
            ):
                continue
            usable.append((pair, positive, negative))
        acoustic_auc: dict[str, float] = {}
        for acoustic_id in ACOUSTIC_EXTRACTORS:
            filtered = [
                item
                for item in usable
                if item[1]["acoustic_scores"].get(acoustic_id) is not None
                and item[2]["acoustic_scores"].get(acoustic_id) is not None
            ]
            value = auc(
                [float(item[1]["acoustic_scores"][acoustic_id]) for item in filtered],
                [float(item[2]["acoustic_scores"][acoustic_id]) for item in filtered],
            )
            if value is not None:
                acoustic_auc[acoustic_id] = value
        selected_acoustic = (
            min(acoustic_auc, key=lambda key: (-acoustic_auc[key], key)) if acoustic_auc else None
        )
        paired_rows: list[dict[str, Any]] = []
        if selected_acoustic is not None:
            for pair, positive, negative in usable:
                positive_acoustic = positive["acoustic_scores"].get(selected_acoustic)
                negative_acoustic = negative["acoustic_scores"].get(selected_acoustic)
                if positive_acoustic is None or negative_acoustic is None:
                    continue
                paired_rows.append(
                    {
                        "pair_id": pair["pair_id"],
                        "block_id": pair["block_id"],
                        "positive_neural": float(positive["neural_score"]),
                        "negative_neural": float(negative["neural_score"]),
                        "positive_acoustic": float(positive_acoustic),
                        "negative_acoustic": float(negative_acoustic),
                    }
                )
        neural_auc = auc(
            [float(row["positive_neural"]) for row in paired_rows],
            [float(row["negative_neural"]) for row in paired_rows],
        )
        neural_eer = eer(
            [float(row["positive_neural"]) for row in paired_rows],
            [float(row["negative_neural"]) for row in paired_rows],
        )
        selected_auc = acoustic_auc.get(selected_acoustic) if selected_acoustic else None
        blocks = sorted({str(row["block_id"]) for row in paired_rows})
        seed = int(
            hashlib.sha256(
                f"{proposal_contract_sha256}|{extractor_id}|primary".encode("utf-8")
            ).hexdigest()[:16],
            16,
        )
        lower: float | None = None
        upper: float | None = None
        bootstrap_draw_sha256: str | None = None
        replicate_count = 0
        if neural_auc is not None and selected_auc is not None:
            values, bootstrap_blocks, bootstrap_draw_sha256 = bootstrap_auc_delta(
                paired_rows,
                seed=seed,
            )
            if bootstrap_blocks != blocks:
                raise Phase4SignalError("bootstrap block identity mismatch")
            lower = nearest_rank(values, 0.025)
            upper = nearest_rank(values, 0.975)
            replicate_count = len(values)
        if neural_auc is None or selected_auc is None:
            status = "not_estimable"
        elif len(blocks) < 8:
            status = "low_block"
        elif lower is not None and lower > 0.0:
            status = "eligible_go"
        elif upper is not None and upper <= 0.0:
            status = "eligible_stop"
        else:
            status = "eligible_uncertain"
        summaries.append(
            {
                "signal_extractor_id": extractor_id,
                "family": declaration["family"],
                "checkpoint": declaration["checkpoint"],
                "base_extractor_id": declaration["base_extractor_id"],
                "horizon_ms": declaration["causal_horizon_ms"],
                "pair_count": len(paired_rows),
                "block_count": len(blocks),
                "block_ids": blocks,
                "missing_positive": missing_positive,
                "missing_negative": missing_negative,
                "neural_auc": neural_auc,
                "neural_eer": neural_eer,
                "selected_acoustic_extractor_id": selected_acoustic,
                "selected_acoustic_auc": selected_auc,
                "acoustic_auc": dict(sorted(acoustic_auc.items())),
                "delta_auc": (
                    neural_auc - selected_auc
                    if neural_auc is not None and selected_auc is not None
                    else None
                ),
                "bootstrap_seed": seed,
                "bootstrap_replicates": replicate_count,
                "bootstrap_block_draws_sha256": bootstrap_draw_sha256,
                "delta_auc_ci95": (
                    [lower, upper] if lower is not None and upper is not None else None
                ),
                "status": status,
            }
        )
        acoustic_receipts.append(
            {
                "signal_extractor_id": extractor_id,
                "identical_pair_count": len(paired_rows),
                "selected_acoustic_extractor_id": selected_acoustic,
                "acoustic_auc": dict(sorted(acoustic_auc.items())),
                "selection_rule": "maximum_full_sample_auc_then_lexical_id",
            }
        )
    return summaries, acoustic_receipts


def family_disposition(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in ("ls_eend", "eres2netv2"):
        primary = [
            row for row in summaries if row["family"] == family and int(row["horizon_ms"]) == 500
        ]
        estimable = [row for row in primary if row["status"] != "not_estimable"]
        eligible = [
            row
            for row in primary
            if row["status"] in ("eligible_go", "eligible_stop", "eligible_uncertain")
        ]
        if not estimable:
            disposition = "not_estimable"
        elif not eligible:
            disposition = "signal_limited"
        elif any(row["status"] == "eligible_go" for row in eligible):
            disposition = "signal_go"
        elif all(row["status"] == "eligible_stop" for row in eligible):
            disposition = "signal_stop"
        else:
            disposition = "signal_limited"
        envelope = {
            "signal_go": "full_predeclared_policy_grid",
            "signal_limited": "same_proposal_ladder_plus_one_sentinel_per_policy_family",
            "signal_stop": "b0_b1_plus_no_neural_control_only",
            "not_estimable": "b0_b1_plus_no_neural_control_only",
        }[disposition]
        result[family] = {
            "disposition": disposition,
            "phase_5_compute_envelope": envelope,
            "primary_extractor_count": len(primary),
            "status_counts": dict(sorted(Counter(row["status"] for row in primary).items())),
            "eligible_go_ids": sorted(
                row["signal_extractor_id"] for row in eligible if row["status"] == "eligible_go"
            ),
        }
    return result


def fixture_paths(inputs: Phase4Inputs) -> list[Path]:
    parity_root = (
        Path(os.environ.get("TEMP") or tempfile.gettempdir()) / "opencode" / "parity_cache"
    )
    paths = [
        inputs.experiment_dir / "data" / "generated" / "golden_silence.wav",
        inputs.experiment_dir / "data" / "generated" / "golden_single_utterance.wav",
        inputs.experiment_dir / "data" / "generated" / "golden_two_utterance_gap400.wav",
        parity_root / "speaker1_a_cn_16k.wav",
        parity_root / "speaker1_b_cn_16k.wav",
        parity_root / "speaker2_a_cn_16k.wav",
    ]
    ledger = {
        str(row["name"]): str(row["sha256"])
        for row in inputs.design_ledger["fixture_ledger"]["clips"]
    }
    for path in paths:
        if not path.is_file() or sha256_file(path) != ledger[path.name]:
            raise Phase4SignalError(f"parity fixture drift: {path}")
    return paths


def read_fixture(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getframerate() != 16000:
            raise Phase4SignalError(f"parity fixture format drift: {path}")
        payload = handle.readframes(handle.getnframes())
    return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0


def run_frontend_parity(
    inputs: Phase4Inputs,
    args: argparse.Namespace,
) -> dict[str, Any]:
    clips = fixture_paths(inputs)
    ls_rows: list[dict[str, Any]] = []
    for checkpoint, info in LS_EEND_VARIANTS.items():
        model = args.hf_root / str(info["dir"]) / str(info["onnx"])
        sidecar = args.hf_root / str(info["dir"]) / str(info["sidecar"])
        runtime = LSEENDCapture(
            model, load_sidecar_metadata(sidecar), checkpoint_variant=checkpoint
        )
        for ordinal, path in enumerate(clips):
            samples = read_fixture(path)
            chunked = runtime.run_case(
                samples,
                case_id=f"parity:{path.name}:chunked",
                audio_epoch=ordinal,
                chunk_samples=512,
            )
            whole = runtime.run_case(
                samples,
                case_id=f"parity:{path.name}:whole",
                audio_epoch=ordinal,
                chunk_samples=max(1, samples.size),
            )
            left = (
                np.stack(chunked.normal_probs)
                if chunked.normal_probs
                else np.zeros((0, runtime.real_output_dim), dtype=np.float32)
            )
            right = (
                np.stack(whole.normal_probs)
                if whole.normal_probs
                else np.zeros((0, runtime.real_output_dim), dtype=np.float32)
            )
            maximum = (
                float(np.max(np.abs(left - right)))
                if left.shape == right.shape and left.size
                else 0.0
            )
            passed = left.shape == right.shape and maximum <= 1e-5
            if not passed:
                raise Phase4SignalError(f"LS parity failed {checkpoint}:{path.name}")
            ls_rows.append(
                {
                    "checkpoint": checkpoint,
                    "clip": path.name,
                    "frame_count": int(left.shape[0]),
                    "posterior_max_abs_error": maximum,
                    "source_coordinates_exact": True,
                    "chunked_tail_count": len(chunked.tail_probs),
                    "whole_tail_count": len(whole.tail_probs),
                    "passed": True,
                }
            )
    eres_rows: list[dict[str, Any]] = []
    for checkpoint in ERES_MODEL_SHA256:
        model = args.eres_onnx_root / str(ERES_CHECKPOINTS[checkpoint]["onnx"])
        runtime = EresEmbeddingRuntime(str(model))
        for path in clips:
            samples = read_fixture(path)
            fbank_whole = kaldi_fbank_numpy(samples)
            reconstructed = np.concatenate(
                [samples[index : index + 512] for index in range(0, samples.size, 512)]
            )
            fbank_chunked = kaldi_fbank_numpy(reconstructed)
            feature_error = (
                float(np.max(np.abs(fbank_whole - fbank_chunked))) if fbank_whole.size else 0.0
            )
            first = normalized(runtime.embed(samples))
            second = normalized(runtime.embed(reconstructed))
            if first is None or second is None:
                raise Phase4SignalError(f"ERes parity zero embedding {checkpoint}:{path.name}")
            embedding_error = float(np.mean(np.abs(first - second)))
            embedding_cosine = cosine_similarity(first, second)
            passed = (
                feature_error <= 1e-3 and embedding_error <= 1e-3 and embedding_cosine >= 0.99999
            )
            if not passed:
                raise Phase4SignalError(f"ERes parity failed {checkpoint}:{path.name}")
            eres_rows.append(
                {
                    "checkpoint": checkpoint,
                    "clip": path.name,
                    "fbank_frames": int(fbank_whole.shape[0]),
                    "frontend_max_abs_error": feature_error,
                    "embedding_mean_abs_error": embedding_error,
                    "embedding_cosine": embedding_cosine,
                    "passed": True,
                }
            )
    return {
        "schema_version": "turn_episode_phase4_frontend_parity.v1",
        "fixture_ledger_sha256": inputs.design_ledger["fixture_ledger"]["clips_ledger_sha256"],
        "prior_receipts": {
            "parity_frontend_byte_sha256": inputs.design_ledger["fixture_ledger"][
                "parity_frontend_byte_sha256"
            ],
            "parity_research_byte_sha256": inputs.design_ledger["fixture_ledger"][
                "parity_research_byte_sha256"
            ],
        },
        "ls": ls_rows,
        "eres": eres_rows,
        "passed": True,
    }


def _trace_sha256(rows: Sequence[dict[str, Any]]) -> str:
    return sha256_bytes(b"".join(canonical_json(row).encode("utf-8") + b"\n" for row in rows))


def _sentinel_clusters(proposals: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        proposals,
        key=lambda row: (
            int(row["observation_frontier"]),
            int(row["boundary_source_sample"]),
            str(row["proposal_kind"]),
            str(row["proposal_id"]),
        ),
    )
    clusters: list[dict[str, Any]] = []
    pending = list(ordered)
    while pending:
        first = pending.pop(0)
        members = [first]
        remainder: list[dict[str, Any]] = []
        for row in pending:
            if (
                int(row["observation_frontier"]) == int(first["observation_frontier"])
                and abs(int(row["boundary_source_sample"]) - int(first["boundary_source_sample"]))
                <= 4_000
            ):
                members.append(row)
            else:
                remainder.append(row)
        pending = remainder
        representative = min(
            members,
            key=lambda row: (
                int(row["observation_frontier"]),
                int(row["boundary_source_sample"]),
                str(row["proposal_id"]),
            ),
        )
        clusters.append(
            {
                "boundary_source_sample": representative["boundary_source_sample"],
                "observation_frontier": representative["observation_frontier"],
                "proposal_kind": representative["proposal_kind"],
                "member_proposal_ids": sorted(str(row["proposal_id"]) for row in members),
            }
        )
    return clusters


def _ls_profile_trace(
    probabilities: np.ndarray,
    *,
    offset: int,
    epoch_length: int,
    profile_class: str,
) -> dict[str, list[dict[str, Any]]]:
    centers, availability = ls_frame_coordinates(probabilities.shape[0])
    centers = centers + offset
    availability = availability + offset
    proposals: list[dict[str, Any]] = []
    progress: list[dict[str, Any]] = []
    if profile_class in ("new_track_onset", "dominant_replacement"):
        policy = (
            "new_speaker_onset" if profile_class == "new_track_onset" else "dominant_replacement"
        )
        profile = ReductionProfile(policy=policy, **STATE_EQ_LS_PROFILE)
        reducer = StreamingReducer(
            profile,
            track_count=probabilities.shape[1],
            audio_epoch=0,
            sample_count_at_epoch_end=epoch_length,
        )
        emitted = 0
        for index, vector in enumerate(probabilities):
            reducer.emit(index, vector)
            for ordinal, boundary in enumerate(reducer.boundaries[emitted:], start=emitted):
                proposals.append(
                    {
                        "proposal_id": f"{profile_class}:{ordinal}",
                        "proposal_kind": profile_class,
                        "boundary_source_sample": boundary.boundary_source_sample() + offset,
                        "observation_frontier": int(availability[boundary.confirmed_output_frame]),
                    }
                )
            emitted = len(reducer.boundaries)
            progress.append(
                {
                    "observed_source_sample": int(availability[index]),
                    "safe_boundary_frontier_sample": min(
                        int(availability[index]),
                        reducer.safe_boundary_frontier_sample() + offset,
                    ),
                }
            )
    else:
        active = np.zeros(probabilities.shape[1], dtype=bool)
        ordinal = 0
        for index, vector in enumerate(probabilities):
            changed = False
            for track, value in enumerate(vector):
                before = bool(active[track])
                if before and float(value) < 0.40:
                    active[track] = False
                elif not before and float(value) >= 0.60:
                    active[track] = True
                changed = changed or before != bool(active[track])
            if changed:
                proposals.append(
                    {
                        "proposal_id": f"hysteretic_activity_state:{ordinal}",
                        "proposal_kind": "hysteretic_activity_state",
                        "boundary_source_sample": int(centers[index]),
                        "observation_frontier": int(availability[index]),
                    }
                )
                ordinal += 1
            progress.append(
                {
                    "observed_source_sample": int(availability[index]),
                    "safe_boundary_frontier_sample": int(centers[index]),
                }
            )
    return {
        "proposals": proposals,
        "clusters": _sentinel_clusters(proposals),
        "progress": progress,
    }


def _scored_trace(trace: dict[str, list[dict[str, Any]]], bounds: dict[str, Any]) -> dict[str, Any]:
    start = int(bounds["scored_start"])
    end = int(bounds["scored_end"])
    proposals = [
        row for row in trace["proposals"] if start <= int(row["observation_frontier"]) <= end
    ]
    proposals = [
        {**row, "proposal_id": f"scored:{index}"}
        for index, row in enumerate(
            sorted(
                proposals,
                key=lambda row: (
                    int(row["observation_frontier"]),
                    int(row["boundary_source_sample"]),
                    str(row["proposal_kind"]),
                ),
            )
        )
    ]
    clusters = _sentinel_clusters(proposals)
    progress = [
        row for row in trace["progress"] if start <= int(row["observed_source_sample"]) <= end
    ]
    return {
        "proposal_count": len(proposals),
        "proposal_sha256": _trace_sha256(proposals),
        "cluster_count": len(clusters),
        "cluster_sha256": _trace_sha256(clusters),
        "progress_count": len(progress),
        "progress_sha256": _trace_sha256(progress),
    }


def _eres_state_trace(
    embeddings: dict[tuple[int, int], np.ndarray],
    shadows: dict[tuple[int, int], tuple[np.ndarray, float]],
    *,
    window: int,
    step: int,
    replay_start: int,
    scored_start: int,
    scored_end: int,
    snapshot_frontier: int,
    mode: str,
) -> dict[str, Any]:
    state: dict[str, Any] = {"anchor": None, "pending": None}
    raw: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    progress: list[dict[str, Any]] = []
    snapshot_hash: str | None = None
    first_end = ceil_grid(replay_start + window, step)
    for end in range(first_end, scored_end + 1, step):
        probe_window = (end - window, end)
        probe = embeddings[probe_window]
        shadow = shadows[probe_window]
        if snapshot_hash is None and end > snapshot_frontier:
            snapshot_hash = _state_hash(state)
        before = _state_hash(state)
        score, _, selected = _measure_anchor_state(state, mode, probe, shadow)
        pending_before = state.get("pending")
        pending_window = pending_before.get("window") if pending_before is not None else None
        mutual: float | None = None
        if pending_before is not None:
            mutual = cosine_similarity(pending_before["embedding"], probe)
        emit_boundary: int | None = None
        if score is not None and score > STATE_EQ_CHANGE_THRESHOLD:
            if mode in ("stable_no_update", "stable_ema"):
                emit_boundary = probe_window[0]
            elif pending_before is not None and mutual is not None and mutual >= 0.50:
                emit_boundary = int(pending_window[0])
        _advance_anchor_state(state, mode, probe, shadow, probe_window)
        after = _state_hash(state)
        if scored_start <= end <= scored_end:
            raw.append(
                {
                    "window": list(probe_window),
                    "embedding_sha256": sha256_bytes(
                        np.asarray(probe, dtype="<f4").tobytes(order="C")
                    ),
                }
            )
            scores.append(
                {
                    "window": list(probe_window),
                    "change_score": score,
                    "selected_prototype_ordinal": selected,
                }
            )
            transitions.append(
                {"window": list(probe_window), "pre_state": before, "post_state": after}
            )
            if emit_boundary is not None:
                proposals.append(
                    {
                        "proposal_id": f"{mode}:{window}:{step}:{emit_boundary}:{end}",
                        "proposal_kind": "speaker_change_unknown",
                        "boundary_source_sample": emit_boundary,
                        "observation_frontier": end,
                    }
                )
            pending_after = state.get("pending")
            safe = (
                int(pending_after["window"][0]) - 1
                if pending_after is not None
                else probe_window[0]
            )
            progress.append(
                {
                    "observed_source_sample": end,
                    "safe_boundary_frontier_sample": max(0, min(end, safe)),
                }
            )
    if snapshot_hash is None:
        snapshot_hash = _state_hash(state)
    return {
        "raw": raw,
        "scores": scores,
        "transitions": transitions,
        "proposals": proposals,
        "clusters": _sentinel_clusters(proposals),
        "progress": progress,
        "snapshot_state_sha256": snapshot_hash,
    }


def _eres_trace_comparison(source: dict[str, Any], reset: dict[str, Any]) -> dict[str, Any]:
    source_raw = source["raw"]
    reset_raw = reset["raw"]
    aligned = len(source_raw) == len(reset_raw) and all(
        left["window"] == right["window"] for left, right in zip(source_raw, reset_raw)
    )
    cosine_min = None
    if aligned and source_raw:
        cosine_min = (
            1.0
            if all(
                left["embedding_sha256"] == right["embedding_sha256"]
                for left, right in zip(source_raw, reset_raw)
            )
            else 0.0
        )
    score_differences: list[float] = []
    scores_aligned = len(source["scores"]) == len(reset["scores"])
    if scores_aligned:
        for left, right in zip(source["scores"], reset["scores"]):
            if left["window"] != right["window"]:
                scores_aligned = False
                break
            left_score = left["change_score"]
            right_score = right["change_score"]
            if left_score is None or right_score is None:
                if left_score is not right_score:
                    scores_aligned = False
                    break
            else:
                score_differences.append(abs(float(left_score) - float(right_score)))
    score_max_abs = max(score_differences) if score_differences else 0.0
    exact_fields = ("transitions", "proposals", "clusters", "progress")
    exact = {key: source[key] == reset[key] for key in exact_fields}
    passed = (
        aligned
        and bool(source_raw)
        and cosine_min is not None
        and cosine_min >= 0.99
        and scores_aligned
        and score_max_abs <= 1e-2
        and all(exact.values())
    )
    return {
        "aligned_window_count": len(source_raw) if aligned else 0,
        "aligned_window_cosine_min": cosine_min,
        "similarity_score_max_abs": score_max_abs if scores_aligned else None,
        "source_snapshot_state_sha256": source["snapshot_state_sha256"],
        "reset_initial_state_sha256": reset["snapshot_state_sha256"],
        "source_trace_sha256": _trace_sha256(
            [{key: value for key, value in source.items() if key != "snapshot_state_sha256"}]
        ),
        "reset_trace_sha256": _trace_sha256(
            [{key: value for key, value in reset.items() if key != "snapshot_state_sha256"}]
        ),
        "exact_trace_fields": exact,
        "passed": passed,
    }


def _eres_adjacent_trace(
    embeddings: dict[tuple[int, int], np.ndarray],
    episode: dict[str, Any],
    window: int,
    step: int,
) -> dict[str, Any]:
    bounds = episode["bounds"]
    lo = max(int(bounds["scored_start"]), int(bounds["warm_start"]) + window)
    hi = min(int(bounds["scored_end"]), int(bounds["tail_end"]) - window)
    raw: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    progress: list[dict[str, Any]] = []
    for boundary in range(ceil_grid(lo, step), hi + 1, step):
        left_window = (boundary - window, boundary)
        right_window = (boundary, boundary + window)
        left = embeddings[left_window]
        right = embeddings[right_window]
        score = 1.0 - cosine_similarity(left, right)
        raw.append(
            {
                "left": list(left_window),
                "right": list(right_window),
                "left_sha256": sha256_bytes(np.asarray(left, dtype="<f4").tobytes(order="C")),
                "right_sha256": sha256_bytes(np.asarray(right, dtype="<f4").tobytes(order="C")),
                "change_score": score,
            }
        )
        if score > STATE_EQ_CHANGE_THRESHOLD:
            proposals.append(
                {
                    "proposal_id": f"adjacent:{window}:{step}:{boundary}",
                    "proposal_kind": "speaker_change_unknown",
                    "boundary_source_sample": boundary,
                    "observation_frontier": boundary + window,
                }
            )
        progress.append(
            {
                "observed_source_sample": boundary + window,
                "safe_boundary_frontier_sample": boundary,
            }
        )
    clusters = _sentinel_clusters(proposals)
    return {
        "raw": raw,
        "proposals": proposals,
        "clusters": clusters,
        "progress": progress,
    }


def run_state_equivalence(
    inputs: Phase4Inputs,
    args: argparse.Namespace,
    ls_contracts: dict[str, dict[str, Any]],
    eres_contracts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    public_episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in inputs.episodes:
        source = inputs.sources[inputs.source_by_episode[str(episode["episode_id"])]]
        if source.public:
            public_episodes[source.source_id].append(episode)
    for values in public_episodes.values():
        values.sort(key=lambda row: str(row["episode_id"]))
    ls_classes: list[dict[str, Any]] = []
    for checkpoint, info in LS_EEND_VARIANTS.items():
        model = args.hf_root / str(info["dir"]) / str(info["onnx"])
        sidecar = args.hf_root / str(info["dir"]) / str(info["sidecar"])
        runtime = LSEENDCapture(
            model, load_sidecar_metadata(sidecar), checkpoint_variant=checkpoint
        )
        records_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for source_id in sorted(public_episodes):
            source = inputs.sources[source_id]
            samples = read_wav(source)
            cached = load_ls_capture(
                args.cache_root,
                ls_contracts[checkpoint],
                checkpoint,
                source,
            )
            if cached is None:
                raise Phase4SignalError(f"missing source-prefix LS cache {checkpoint}:{source_id}")
            source_capture, _ = cached
            source_probabilities = (
                np.stack(source_capture.normal_probs)
                if source_capture.normal_probs
                else np.zeros((0, runtime.real_output_dim), dtype=np.float32)
            )
            source_centers, _ = ls_frame_coordinates(source_probabilities.shape[0])
            source_index = {int(value): index for index, value in enumerate(source_centers)}
            source_traces = {
                profile_class: _ls_profile_trace(
                    source_probabilities,
                    offset=0,
                    epoch_length=source.duration_samples,
                    profile_class=profile_class,
                )
                for profile_class in (
                    "new_track_onset",
                    "dominant_replacement",
                    "hysteretic_activity_state",
                )
            }
            for episode in public_episodes[source_id]:
                bounds = episode["bounds"]
                warm = int(bounds["warm_start"])
                tail = int(bounds["tail_end"])
                reset_capture = runtime.run_case(
                    samples[warm:tail],
                    case_id=str(episode["episode_id"]),
                    audio_epoch=int(source.wav_sha256[:8], 16),
                )
                reset_probabilities = (
                    np.stack(reset_capture.normal_probs)
                    if reset_capture.normal_probs
                    else np.zeros((0, runtime.real_output_dim), dtype=np.float32)
                )
                reset_centers, _ = ls_frame_coordinates(reset_probabilities.shape[0])
                reset_centers = reset_centers + warm
                common = [
                    (source_index[int(center)], index)
                    for index, center in enumerate(reset_centers)
                    if int(center) in source_index
                    and int(bounds["scored_start"]) <= int(center) < int(bounds["scored_end"])
                ]
                maximum = (
                    max(
                        float(
                            np.sum(np.abs(source_probabilities[left] - reset_probabilities[right]))
                        )
                        for left, right in common
                    )
                    if common
                    else None
                )
                raw_passed = maximum is not None and maximum <= 1e-2
                records_by_class["raw_posterior"].append(
                    {
                        "episode_id": episode["episode_id"],
                        "aligned_frame_count": len(common),
                        "posterior_max_l1": maximum,
                        "passed": raw_passed,
                        "reason": (None if raw_passed else "no_aligned_frames_or_posterior_drift"),
                    }
                )
                for profile_class, source_trace in source_traces.items():
                    reset_trace = _ls_profile_trace(
                        reset_probabilities,
                        offset=warm,
                        epoch_length=tail - warm,
                        profile_class=profile_class,
                    )
                    source_receipt = _scored_trace(source_trace, bounds)
                    reset_receipt = _scored_trace(reset_trace, bounds)
                    passed = raw_passed and source_receipt == reset_receipt
                    records_by_class[profile_class].append(
                        {
                            "episode_id": episode["episode_id"],
                            "posterior_gate_passed": raw_passed,
                            "source": source_receipt,
                            "episode_reset": reset_receipt,
                            "passed": passed,
                            "reason": None if passed else "posterior_or_trace_mismatch",
                        }
                    )
        for profile_class in (
            "raw_posterior",
            "new_track_onset",
            "dominant_replacement",
            "hysteretic_activity_state",
        ):
            records = records_by_class[profile_class]
            failed = [row for row in records if not row["passed"]]
            ls_classes.append(
                {
                    "family": "ls_eend",
                    "checkpoint": checkpoint,
                    "profile_class": profile_class,
                    "case_count": len(records),
                    "failed_count": len(failed),
                    "disposition": (
                        "episode_reset_permitted" if not failed else "source_prefix_required"
                    ),
                    "source_prefix_cache_contract_sha256": ls_contracts[checkpoint][
                        "contract_sha256"
                    ],
                    "records_sha256": _trace_sha256(records),
                    "records": records,
                }
            )
    eres_classes: list[dict[str, Any]] = []
    source_lookup = source_by_wav(inputs)
    for checkpoint in ERES_MODEL_SHA256:
        embeddings_by_wav: dict[str, dict[tuple[int, int], np.ndarray]] = {}
        shadows_by_wav: dict[str, dict[tuple[int, int], tuple[np.ndarray, float]]] = {}
        cache_receipts: dict[str, str] = {}
        for wav, expected_windows in sorted(inputs.embedding_windows.items()):
            source = source_lookup[wav]
            cached = load_eres_embeddings(
                args.cache_root,
                eres_contracts[checkpoint],
                checkpoint,
                source,
                sorted(expected_windows),
            )
            if cached is None:
                raise Phase4SignalError(f"missing source-prefix ERes cache {checkpoint}:{wav}")
            embeddings_by_wav[wav], shadows_by_wav[wav], evidence = cached
            cache_receipts[wav] = str(evidence["payload_sha256"])
        for profile_class in (
            "adjacent",
            "stable_no_update",
            "stable_ema",
            "confirmed_anchor",
            "prototype_memory_4",
        ):
            class_records: list[dict[str, Any]] = []
            for source_id in sorted(public_episodes):
                source = inputs.sources[source_id]
                embeddings = embeddings_by_wav[source.wav_sha256]
                shadows = shadows_by_wav[source.wav_sha256]
                for episode in public_episodes[source_id]:
                    profile_receipts: list[dict[str, Any]] = []
                    if profile_class == "adjacent":
                        for window in ADJACENT_WINDOWS:
                            for step in LONG_STEPS if window >= 24000 else STEPS:
                                source_trace = _eres_adjacent_trace(
                                    embeddings, episode, window, step
                                )
                                reset_trace = _eres_adjacent_trace(
                                    embeddings, episode, window, step
                                )
                                passed = source_trace == reset_trace and bool(source_trace["raw"])
                                profile_receipts.append(
                                    {
                                        "profile_id": f"adjacent:W{window}:S{step}",
                                        "aligned_window_count": 2 * len(source_trace["raw"]),
                                        "aligned_window_cosine_min": (
                                            1.0 if source_trace["raw"] else None
                                        ),
                                        "source_trace_sha256": _trace_sha256([source_trace]),
                                        "reset_trace_sha256": _trace_sha256([reset_trace]),
                                        "passed": passed,
                                    }
                                )
                    else:
                        bounds = episode["bounds"]
                        for window in ANCHOR_WINDOWS:
                            for step in STEPS:
                                source_trace = _eres_state_trace(
                                    embeddings,
                                    shadows,
                                    window=window,
                                    step=step,
                                    replay_start=0,
                                    scored_start=int(bounds["scored_start"]),
                                    scored_end=int(bounds["scored_end"]),
                                    snapshot_frontier=int(bounds["warm_start"]),
                                    mode=profile_class,
                                )
                                reset_trace = _eres_state_trace(
                                    embeddings,
                                    shadows,
                                    window=window,
                                    step=step,
                                    replay_start=int(bounds["warm_start"]),
                                    scored_start=int(bounds["scored_start"]),
                                    scored_end=int(bounds["scored_end"]),
                                    snapshot_frontier=int(bounds["warm_start"]),
                                    mode=profile_class,
                                )
                                comparison = _eres_trace_comparison(source_trace, reset_trace)
                                profile_receipts.append(
                                    {
                                        "profile_id": f"{profile_class}:W{window}:S{step}",
                                        **comparison,
                                    }
                                )
                    failed_profiles = [
                        row["profile_id"] for row in profile_receipts if not row["passed"]
                    ]
                    class_records.append(
                        {
                            "episode_id": episode["episode_id"],
                            "source_id": source_id,
                            "profile_count": len(profile_receipts),
                            "failed_profile_count": len(failed_profiles),
                            "failed_profile_ids": failed_profiles,
                            "profile_receipts_sha256": _trace_sha256(profile_receipts),
                            "aligned_window_cosine_min": (
                                min(
                                    float(row["aligned_window_cosine_min"])
                                    for row in profile_receipts
                                    if row.get("aligned_window_cosine_min") is not None
                                )
                                if any(
                                    row.get("aligned_window_cosine_min") is not None
                                    for row in profile_receipts
                                )
                                else None
                            ),
                            "cache_payload_sha256": cache_receipts[source.wav_sha256],
                            "passed": not failed_profiles,
                        }
                    )
            failed = [row for row in class_records if not row["passed"]]
            cosine_values = [
                float(record["aligned_window_cosine_min"])
                for record in class_records
                if record["aligned_window_cosine_min"] is not None
            ]
            eres_classes.append(
                {
                    "family": "eres2netv2",
                    "checkpoint": checkpoint,
                    "profile_class": profile_class,
                    "case_count": len(class_records),
                    "failed_count": len(failed),
                    "aligned_window_cosine_min": min(cosine_values) if cosine_values else None,
                    "proposal_and_progress_contract": "executed_exact_source_prefix_vs_episode_reset",
                    "disposition": (
                        "episode_reset_permitted" if not failed else "source_prefix_required"
                    ),
                    "source_prefix_cache_contract_sha256": eres_contracts[checkpoint][
                        "contract_sha256"
                    ],
                    "records_sha256": _trace_sha256(class_records),
                    "records": class_records,
                }
            )
    passed = all(
        row["case_count"] == sum(len(values) for values in public_episodes.values())
        and row["disposition"] in ("episode_reset_permitted", "source_prefix_required")
        for row in ls_classes + eres_classes
    )
    return {
        "schema_version": "turn_episode_phase4_state_equivalence.v2",
        "tolerances": {
            "ls_posterior_max_l1": 1e-2,
            "eres_aligned_window_cosine_min": 0.99,
            "eres_similarity_score_max_abs": 1e-2,
            "proposal_count_kind_boundary_frontier": "exact",
            "sentinel_cluster": {"debounce_ms": 0, "radius_ms": 250, "refractory_ms": 0},
            "sentinel_cluster_progress": "exact",
            "ls_reducer_profile": STATE_EQ_LS_PROFILE,
            "eres_change_threshold": STATE_EQ_CHANGE_THRESHOLD,
        },
        "ls_profile_classes": ls_classes,
        "eres_profile_classes": eres_classes,
        "public_episode_count": sum(len(values) for values in public_episodes.values()),
        "scored_state_mode": {
            "ls_eend": "source_prefix_when_any_class_fails",
            "eres2netv2": "source_prefix_when_profile_class_fails",
        },
        "passed": passed,
    }


def causal_oracle_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["candidate_class"] == "positive":
            grouped[str(row["signal_extractor_id"])].append(row)
    summaries: list[dict[str, Any]] = []
    for extractor_id, values in sorted(grouped.items()):
        valid = [row for row in values if not row["missing"]]
        delays = [
            int(row["observation_frontier"]) - int(row["boundary_source_sample"]) for row in valid
        ]
        errors = [
            abs(
                int(row.get("selected_center_source_sample") or row["boundary_source_sample"])
                - int(row["boundary_source_sample"])
            )
            for row in valid
        ]
        summaries.append(
            {
                "signal_extractor_id": extractor_id,
                "target_count": len(values),
                "covered_count": len(valid),
                "coverage": len(valid) / len(values) if values else None,
                "availability_delay_samples": {
                    "p50": float(np.percentile(delays, 50)) if delays else None,
                    "p95": float(np.percentile(delays, 95)) if delays else None,
                    "maximum": max(delays) if delays else None,
                },
                "boundary_error_samples": {
                    "p50": float(np.percentile(errors, 50)) if errors else None,
                    "p95": float(np.percentile(errors, 95)) if errors else None,
                    "maximum": max(errors) if errors else None,
                },
                "oracle_only": True,
            }
        )
    return summaries


def artifact_receipt(path: Path, result_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(result_dir)).replace("\\", "/"),
        "byte_sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def write_phase4_results(
    inputs: Phase4Inputs,
    args: argparse.Namespace,
    preflight: dict[str, Any],
    parity: dict[str, Any],
    state_equivalence: dict[str, Any],
    ls_inventory: dict[str, dict[str, Any]],
    eres_inventory: dict[str, dict[str, Any]],
    ls_rows_by_checkpoint: dict[str, list[dict[str, Any]]],
    eres_rows_by_checkpoint: dict[str, list[dict[str, Any]]],
    ls_reports: dict[str, dict[str, Any]],
    eres_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result_dir = inputs.result_dir
    detail_root = result_dir / "phase_4_signal_details"
    detail_index: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for checkpoint, rows in ls_rows_by_checkpoint.items():
        detail_index.extend(write_detail_shards(detail_root, "ls_eend", checkpoint, rows))
        all_rows.extend(rows)
    for checkpoint, rows in eres_rows_by_checkpoint.items():
        detail_index.extend(write_detail_shards(detail_root, "eres2netv2", checkpoint, rows))
        all_rows.extend(rows)
    proposal_contract = read_json(result_dir / "proposal_contract.json")
    summaries, acoustic_receipts = analyze_signal_rows(
        all_rows,
        inputs,
        str(proposal_contract["content_sha256"]),
    )
    dispositions = family_disposition(summaries)
    generated_from = {
        "phase4_signal.py": sha256_file(Path(__file__).resolve()),
        "phase4_design.py": sha256_file(Path(__file__).with_name("phase4_design.py")),
    }
    parity_written = atomic_write_json(
        result_dir / "phase_4_frontend_parity.json",
        {**parity, "generated_from": generated_from},
    )
    state_written = atomic_write_json(
        result_dir / "phase_4_state_equivalence.json",
        {**state_equivalence, "generated_from": generated_from},
    )
    ls_summary_rows = [row for row in summaries if row["family"] == "ls_eend"]
    eres_summary_rows = [row for row in summaries if row["family"] == "eres2netv2"]
    oracle = causal_oracle_summary(all_rows)
    ls_written = atomic_write_json(
        result_dir / "phase_4_ls_signal_report.json",
        {
            "schema_version": "turn_episode_phase4_ls_signal_report.v1",
            "authority_sha256": AUTHORITY_SHA256,
            "phase_4_bundle_sha256": PHASE4_BUNDLE_SHA256,
            "proposal_contract_sha256": proposal_contract["content_sha256"],
            "generated_from": generated_from,
            "checkpoints": ls_reports,
            "extractors": ls_summary_rows,
            "causal_oracle": [
                row for row in oracle if row["signal_extractor_id"].startswith("ls_")
            ],
            "accepted_pcm_oracle_sha256": sha256_file(result_dir / "oracle_provider_neutral.json"),
            "detail_shards": [row for row in detail_index if row["family"] == "ls_eend"],
        },
    )
    eres_written = atomic_write_json(
        result_dir / "phase_4_eres_signal_report.json",
        {
            "schema_version": "turn_episode_phase4_eres_signal_report.v1",
            "authority_sha256": AUTHORITY_SHA256,
            "phase_4_bundle_sha256": PHASE4_BUNDLE_SHA256,
            "proposal_contract_sha256": proposal_contract["content_sha256"],
            "generated_from": generated_from,
            "checkpoints": eres_reports,
            "extractors": eres_summary_rows,
            "causal_oracle": [
                row for row in oracle if row["signal_extractor_id"].startswith("eres_")
            ],
            "detail_shards": [row for row in detail_index if row["family"] == "eres2netv2"],
        },
    )
    acoustic_written = atomic_write_json(
        result_dir / "phase_4_acoustic_controls.json",
        {
            "schema_version": "turn_episode_phase4_acoustic_controls.v1",
            "authority_sha256": AUTHORITY_SHA256,
            "phase_4_bundle_sha256": PHASE4_BUNDLE_SHA256,
            "selection_rule": "maximum_full_sample_auc_on_identical_pairs_then_lexical_id",
            "receipts": acoustic_receipts,
            "generated_from": generated_from,
        },
    )
    disposition_written = atomic_write_json(
        result_dir / "phase_4_signal_disposition.json",
        {
            "schema_version": "turn_episode_phase4_signal_disposition.v1",
            "authority_sha256": AUTHORITY_SHA256,
            "phase_4_bundle_sha256": PHASE4_BUNDLE_SHA256,
            "proposal_contract_sha256": proposal_contract["content_sha256"],
            "families": dispositions,
            "primary_horizon_ms": 500,
            "sensitivity_horizons_ms": [250, 1000],
            "all_checkpoints_visible": True,
            "phase_5_envelope_is_mechanical": True,
            "generated_from": generated_from,
        },
    )
    cache_payload = {
        "schema_version": "turn_episode_phase4_cache_inventory.v1",
        "authority_sha256": AUTHORITY_SHA256,
        "phase_4_bundle_sha256": PHASE4_BUNDLE_SHA256,
        "cache_root": str(args.cache_root.resolve()),
        "preflight": preflight,
        "ls": ls_inventory,
        "eres": eres_inventory,
        "detail_shards": detail_index,
        "generated_from": generated_from,
    }
    cache_written = atomic_write_json(result_dir / "phase_4_cache_inventory.json", cache_payload)
    aggregate_paths = [
        result_dir / "proposal_contract.json",
        result_dir / "phase_4_preflight.json",
        result_dir / "phase_4_frontend_parity.json",
        result_dir / "phase_4_state_equivalence.json",
        result_dir / "phase_4_ls_signal_report.json",
        result_dir / "phase_4_eres_signal_report.json",
        result_dir / "phase_4_acoustic_controls.json",
        result_dir / "phase_4_signal_disposition.json",
        result_dir / "phase_4_cache_inventory.json",
    ]
    artifacts = [artifact_receipt(path, result_dir) for path in aggregate_paths]
    artifacts.extend(
        {
            "path": row["path"],
            "byte_sha256": row["byte_sha256"],
            "size_bytes": row["size_bytes"],
        }
        for row in detail_index
    )
    completion = {
        "schema_version": "turn_episode_phase4_completion.v1",
        "authority_sha256": AUTHORITY_SHA256,
        "phase_4_bundle_sha256": PHASE4_BUNDLE_SHA256,
        "population": preflight["population"],
        "signal_row_count": len(all_rows),
        "extractor_summary_count": len(summaries),
        "detail_shard_count": len(detail_index),
        "family_dispositions": dispositions,
        "artifacts": artifacts,
        "generated_from": generated_from,
    }
    return {
        "completion": completion,
        "written": {
            "parity": parity_written,
            "state": state_written,
            "ls": ls_written,
            "eres": eres_written,
            "acoustic": acoustic_written,
            "disposition": disposition_written,
            "cache": cache_written,
        },
    }


def run_smoke(inputs: Phase4Inputs, args: argparse.Namespace) -> dict[str, Any]:
    original = args.cache_root
    args.cache_root = original / "smoke"
    parity = run_frontend_parity(inputs, args)
    minimum_ls_duration = output_frame_available_16k_count(0)
    ls_contracts, ls_inventory = run_ls_cache(
        inputs,
        args,
        source_limit=3,
        minimum_duration_samples=minimum_ls_duration,
    )
    eres_contracts, eres_inventory = run_eres_cache(
        inputs,
        args,
        source_limit=3,
        window_limit=16,
    )
    for checkpoint, contract in ls_contracts.items():
        if ls_inventory[checkpoint]["source_count"] != 3:
            raise Phase4SignalError("LS smoke cache completeness failure")
        if ls_inventory[checkpoint]["normal_frame_count"] <= 0:
            raise Phase4SignalError("LS smoke did not execute a normal streaming frame")
        for source in sorted(
            (
                source
                for source in inputs.sources.values()
                if source.duration_samples >= minimum_ls_duration
            ),
            key=lambda item: (item.duration_samples, item.source_id),
        )[:3]:
            if load_ls_capture(args.cache_root, contract, checkpoint, source) is None:
                raise Phase4SignalError("LS smoke cache reload failure")
    source_lookup = source_by_wav(inputs)
    selected_sources = [
        (
            source_lookup[wav],
            sorted(windows)[:16],
        )
        for wav, windows in inputs.embedding_windows.items()
    ]
    selected_sources.sort(key=lambda item: (item[0].duration_samples, item[0].source_id))
    for checkpoint, contract in eres_contracts.items():
        if eres_inventory[checkpoint]["source_count"] != 3:
            raise Phase4SignalError("ERes smoke cache completeness failure")
        for source, windows in selected_sources[:3]:
            if (
                load_eres_embeddings(
                    args.cache_root,
                    contract,
                    checkpoint,
                    source,
                    windows,
                )
                is None
            ):
                raise Phase4SignalError("ERes smoke cache reload failure")
    args.cache_root = original
    return {
        "schema_version": "turn_episode_phase4_smoke.v1",
        "parity_passed": parity["passed"],
        "ls": {
            key: {
                "source_count": value["source_count"],
                "normal_frame_count": value["normal_frame_count"],
            }
            for key, value in ls_inventory.items()
        },
        "eres": {
            key: {"source_count": value["source_count"], "window_count": value["window_count"]}
            for key, value in eres_inventory.items()
        },
        "passed": True,
    }


def run_full(inputs: Phase4Inputs, args: argparse.Namespace) -> dict[str, Any]:
    preflight = preflight_payload(inputs, args)
    atomic_write_json(inputs.result_dir / "phase_4_preflight.json", preflight)
    parity = run_frontend_parity(inputs, args)
    ls_contracts, ls_inventory = run_ls_cache(inputs, args)
    eres_contracts, eres_inventory = run_eres_cache(inputs, args)
    state = run_state_equivalence(inputs, args, ls_contracts, eres_contracts)
    ls_rows: dict[str, list[dict[str, Any]]] = {}
    ls_reports: dict[str, dict[str, Any]] = {}
    for checkpoint, contract in ls_contracts.items():
        ls_rows[checkpoint], ls_reports[checkpoint] = score_ls_checkpoint(
            inputs, args, checkpoint, contract
        )
    eres_rows: dict[str, list[dict[str, Any]]] = {}
    eres_reports: dict[str, dict[str, Any]] = {}
    for checkpoint, contract in eres_contracts.items():
        eres_rows[checkpoint], eres_reports[checkpoint] = score_eres_checkpoint(
            inputs, args, checkpoint, contract, state
        )
    result = write_phase4_results(
        inputs,
        args,
        preflight,
        parity,
        state,
        ls_inventory,
        eres_inventory,
        ls_rows,
        eres_rows,
        ls_reports,
        eres_reports,
    )
    completion = atomic_write_json(
        inputs.result_dir / "phase_4_completion.json",
        result["completion"],
    )
    return completion


def normalized(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        return None
    return (vector / norm).astype(np.float32, copy=False)


def acoustic_payload(samples: np.ndarray) -> tuple[np.ndarray | None, float]:
    frames = kaldi_fbank_numpy(samples)
    if frames.size == 0:
        return None, float("-inf")
    mean = normalized(frames.mean(axis=0))
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return mean, math.log(max(rms, 1e-8))


def acoustic_scores(
    left_samples: np.ndarray,
    right_samples: np.ndarray,
) -> dict[str, float | None]:
    left_logmel, left_rms = acoustic_payload(left_samples)
    right_logmel, right_rms = acoustic_payload(right_samples)
    flux = None
    if left_logmel is not None and right_logmel is not None:
        flux = 1.0 - cosine_similarity(left_logmel, right_logmel)
    return {
        "acoustic_log_rms_delta.v1": abs(left_rms - right_rms),
        "acoustic_logmel_flux.v1": flux,
    }


def ls_scalar_series(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    count = probabilities.shape[0]
    new_track = np.full(count, np.nan, dtype=np.float64)
    replacement = np.full(count, np.nan, dtype=np.float64)
    activity = np.zeros(count, dtype=np.float64)
    overlap = np.zeros(count, dtype=np.float64)
    if probabilities.ndim != 2:
        raise Phase4SignalError("LS probabilities must be rank two")
    active = np.zeros(probabilities.shape[1], dtype=bool)
    prior = probabilities[0].astype(np.float64) if count else np.zeros(0)
    for index in range(count):
        current = probabilities[index].astype(np.float64)
        if index >= 3:
            preceding = probabilities[index - 3 : index].astype(np.float64)
            previous_max = preceding.max(axis=0)
            new_track[index] = float(np.maximum(0.0, current - previous_max).max())
            q = preceding.mean(axis=0)
            a = int(np.argmax(q))
            choices = [item for item in range(current.size) if item != a]
            if choices:
                b = min(choices, key=lambda item: (-current[item], item))
                replacement[index] = max(0.0, current[b] - current[a]) * max(0.0, q[a] - q[b])
        changed: list[int] = []
        for track, value in enumerate(current):
            before = bool(active[track])
            if before and value < 0.40:
                active[track] = False
            elif not before and value >= 0.60:
                active[track] = True
            if before != bool(active[track]):
                changed.append(track)
        if changed:
            activity[index] = max(abs(current[track] - prior[track]) for track in changed)
        if current.size >= 2:
            top = np.partition(current, -2)[-2:]
            overlap[index] = float(top[0] * top[1])
        prior = current
    return {
        "ls_new_track_rise.v1": new_track,
        "ls_dominant_replacement.v1": replacement,
        "ls_activity_set_change.v1": activity,
        "ls_overlap_strength.v1": overlap,
    }


def auc(scores_positive: Sequence[float], scores_negative: Sequence[float]) -> float | None:
    if not scores_positive or not scores_negative:
        return None
    values = [(float(value), 1) for value in scores_positive] + [
        (float(value), 0) for value in scores_negative
    ]
    values.sort(key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[end][0] == values[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(item[1] for item in values[index:end])
        index = end
    positive_count = len(scores_positive)
    negative_count = len(scores_negative)
    return (rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def eer(scores_positive: Sequence[float], scores_negative: Sequence[float]) -> float | None:
    if not scores_positive or not scores_negative:
        return None
    thresholds = sorted(
        set(map(float, scores_positive)) | set(map(float, scores_negative)), reverse=True
    )
    points: list[tuple[float, float]] = [(0.0, 1.0)]
    for threshold in thresholds:
        false_positive = sum(value >= threshold for value in scores_negative) / len(scores_negative)
        false_negative = sum(value < threshold for value in scores_positive) / len(scores_positive)
        points.append((false_positive, false_negative))
    points.append((1.0, 0.0))
    best = min(points, key=lambda item: (abs(item[0] - item[1]), item[0], item[1]))
    for left, right in zip(points, points[1:]):
        d0 = left[0] - left[1]
        d1 = right[0] - right[1]
        if d0 == 0:
            return left[0]
        if d0 * d1 <= 0 and d0 != d1:
            weight = d0 / (d0 - d1)
            fpr = left[0] + weight * (right[0] - left[0])
            fnr = left[1] + weight * (right[1] - left[1])
            return (fpr + fnr) / 2.0
    return (best[0] + best[1]) / 2.0


def nearest_rank(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def bootstrap_auc_delta(
    pairs: Sequence[dict[str, Any]],
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[list[float], list[str], str]:
    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        by_block[str(row["block_id"])].append(row)
    block_ids = sorted(by_block)
    block_index = {block: index for index, block in enumerate(block_ids)}
    size = len(block_ids)
    neural_matrix = np.zeros((size, size), dtype=np.float64)
    acoustic_matrix = np.zeros((size, size), dtype=np.float64)
    positive_counts = np.zeros(size, dtype=np.float64)
    negative_counts = np.zeros(size, dtype=np.float64)
    for positive in pairs:
        i = block_index[str(positive["block_id"])]
        positive_counts[i] += 1.0
        for negative in pairs:
            j = block_index[str(negative["block_id"])]
            neural_matrix[i, j] += (
                1.0
                if float(positive["positive_neural"]) > float(negative["negative_neural"])
                else (
                    0.5
                    if float(positive["positive_neural"]) == float(negative["negative_neural"])
                    else 0.0
                )
            )
            acoustic_matrix[i, j] += (
                1.0
                if float(positive["positive_acoustic"]) > float(negative["negative_acoustic"])
                else (
                    0.5
                    if float(positive["positive_acoustic"]) == float(negative["negative_acoustic"])
                    else 0.0
                )
            )
    for negative in pairs:
        negative_counts[block_index[str(negative["block_id"])]] += 1.0
    rng = random.Random(seed)
    values: list[float] = []
    draw_digest = hashlib.sha256()
    for _ in range(replicates):
        draws = [rng.randrange(size) for _ in block_ids]
        draw_digest.update(canonical_json(draws).encode("utf-8") + b"\n")
        weights = np.bincount(draws, minlength=size).astype(np.float64)
        positive_total = float(weights @ positive_counts)
        negative_total = float(weights @ negative_counts)
        if positive_total == 0 or negative_total == 0:
            raise Phase4SignalError("bootstrap replicate became non-estimable")
        neural = float(weights @ neural_matrix @ weights) / (positive_total * negative_total)
        acoustic = float(weights @ acoustic_matrix @ weights) / (positive_total * negative_total)
        values.append(neural - acoustic)
    return values, block_ids, draw_digest.hexdigest()


def runtime_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cores": os.cpu_count(),
        "onnxruntime": ort.__version__,
        "provider": "CPUExecutionProvider",
        "intra_op_threads": 1,
        "inter_op_threads": 1,
    }


def preflight_payload(inputs: Phase4Inputs, args: argparse.Namespace) -> dict[str, Any]:
    model_files: dict[str, Any] = {}
    for checkpoint, info in LS_EEND_VARIANTS.items():
        model = args.hf_root / str(info["dir"]) / str(info["onnx"])
        sidecar = args.hf_root / str(info["dir"]) / str(info["sidecar"])
        if sha256_file(model) != str(info["onnx_sha256"]):
            raise Phase4SignalError(f"LS model hash drift: {checkpoint}")
        if sha256_file(sidecar) != str(info["sidecar_sha256"]):
            raise Phase4SignalError(f"LS sidecar hash drift: {checkpoint}")
        model_files[checkpoint] = {
            "model": str(model),
            "model_sha256": str(info["onnx_sha256"]),
            "sidecar": str(sidecar),
            "sidecar_sha256": str(info["sidecar_sha256"]),
        }
    for checkpoint, expected in ERES_MODEL_SHA256.items():
        model = args.eres_onnx_root / str(ERES_CHECKPOINTS[checkpoint]["onnx"])
        if sha256_file(model) != expected:
            raise Phase4SignalError(f"ERes model hash drift: {checkpoint}")
        model_files[checkpoint] = {"model": str(model), "model_sha256": expected}
    missing = [source.source_id for source in inputs.sources.values() if not source.path.is_file()]
    if missing:
        raise Phase4SignalError(f"missing source WAVs: {missing[:5]}")
    forecast = inputs.design_ledger["runtime_forecast"]
    if not forecast["within_ceilings"]:
        raise Phase4SignalError("design forecast exceeds frozen ceilings")
    return {
        "schema_version": "turn_episode_phase4_preflight.v1",
        "authority_sha256": AUTHORITY_SHA256,
        "phase_4_bundle_sha256": PHASE4_BUNDLE_SHA256,
        "design_ledger_sha256": DESIGN_LEDGER_SHA256,
        "population": {
            "episode_count": len(inputs.episodes),
            "source_count": len(inputs.sources),
            "public_source_count": sum(source.public for source in inputs.sources.values()),
            "candidate_count": len(inputs.candidates),
            "pair_count": len(inputs.pairs),
        },
        "coordinate_ledger": inputs.design_ledger["coordinate_ledger"],
        "model_files": model_files,
        "cache_root": str(args.cache_root.resolve()),
        "forecast": forecast,
        "environment": runtime_environment(),
        "network": "forbidden",
        "credentials": "forbidden",
        "confirmatory_access": "forbidden",
    }


def default_cache_root() -> Path:
    return (
        Path(os.environ.get("TEMP") or tempfile.gettempdir())
        / "puripuly_stb_phase4"
        / "turn_episode_v1"
    )


def default_hf_root() -> Path:
    return (
        Path(os.environ.get("TEMP") or tempfile.gettempdir()) / "opencode" / "LS-EEND-ONNX" / "repo"
    )


def default_eres_root() -> Path:
    return Path(os.environ.get("TEMP") or tempfile.gettempdir()) / "opencode" / "eres_onnx"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("registry", "preflight", "smoke", "run"))
    parser.add_argument("--cache-root", type=Path, default=default_cache_root())
    parser.add_argument("--hf-root", type=Path, default=default_hf_root())
    parser.add_argument("--eres-onnx-root", type=Path, default=default_eres_root())
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment_dir = Path(__file__).resolve().parent.parent
    inputs = load_inputs(experiment_dir)
    if args.command == "registry":
        path = inputs.result_dir / "proposal_contract.json"
        existing = json.loads(path.read_text(encoding="utf-8"))
        written = atomic_write_json(path, proposal_contract_payload(existing))
        print(canonical_json({"path": str(path), "content_sha256": written["content_sha256"]}))
        return
    contract = read_json(inputs.result_dir / "proposal_contract.json")
    if len(contract.get("signal_extractors") or []) != len(signal_registry()):
        raise Phase4SignalError("proposal contract signal registry incomplete")
    if args.command == "preflight":
        payload = preflight_payload(inputs, args)
        path = inputs.result_dir / "phase_4_preflight.json"
        written = atomic_write_json(path, payload)
        print(canonical_json({"path": str(path), "content_sha256": written["content_sha256"]}))
        return
    if args.command == "smoke":
        payload = run_smoke(inputs, args)
        path = inputs.result_dir / "phase_4_smoke.json"
        written = atomic_write_json(path, payload)
        print(canonical_json({"path": str(path), "content_sha256": written["content_sha256"]}))
        return
    completion = run_full(inputs, args)
    print(
        canonical_json(
            {
                "path": str(inputs.result_dir / "phase_4_completion.json"),
                "content_sha256": completion["content_sha256"],
                "family_dispositions": completion["family_dispositions"],
            }
        )
    )


if __name__ == "__main__":
    main()
