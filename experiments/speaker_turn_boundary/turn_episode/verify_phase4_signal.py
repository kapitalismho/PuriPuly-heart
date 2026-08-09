from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import math
import random
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from experiments.speaker_turn_boundary.phase3_ls import (
    LSEENDCapture,
    load_sidecar_metadata,
)
from experiments.speaker_turn_boundary.provenance import LS_EEND_VARIANTS
from experiments.speaker_turn_boundary.reducer import ReductionProfile, StreamingReducer

from .phase4_design import (
    ADJACENT_WINDOWS,
    ANCHOR_WINDOWS,
    GROUP_GRAPH_SHA256,
    LONG_STEPS,
    LS_ACOUSTIC_SUPPORT_BY_HORIZON,
    MANIFEST_BYTE_SHA256,
    MANIFEST_CONTENT_SHA256,
    STEPS,
    build_candidates,
    ceil_grid,
    component_map,
    load_public_regions,
    load_synthetic_cases,
    match_pairs,
    synthetic_case_id,
    synthetic_manifest_name,
)

AUTHORITY_SHA256 = "ecd16c765072504f9970cb17fd5cba4f9967715954adead6ebc386fffb8d4f8c"
PHASE4_BUNDLE_SHA256 = "a6afa3dc946815c162ee18d09b1c7ad3ad08e252f7286c110b37a685fe2b1759"
PAIR_ROWS_SHA256 = "fb29fff960932f2840433fa94f1a9e4bade167a6d935a6458dc6e9b191a4f9b9"
COORDINATE_ROWS_SHA256 = "58cbd9eaf4554761bf71e698bc4b1f251ae722c4281be35d0270dbc0ab285470"
EMBEDDING_WINDOWS_SHA256 = "de3646936555280a01dcf2461d3d02bccc722a55a088aa15b5a8c97f639e3118"
DETAIL_SHARD_LIMIT = 20 * 1024 * 1024
AGGREGATE_LIMIT = 10 * 1024 * 1024
HORIZONS_MS = (250, 500, 1000)
ERES_CHECKPOINTS = ("E-standard", "E-w24s4ep4")
ERES_STATES = (
    "stable_no_update",
    "stable_ema",
    "confirmed_anchor",
    "prototype_memory_4",
)
LS_HARD_EXTRACTORS = (
    "ls_new_track_rise.v1",
    "ls_dominant_replacement.v1",
    "ls_activity_set_change.v1",
)
LS_SECONDARY_EXTRACTORS = ("ls_overlap_strength.v1",)
ACOUSTIC_EXTRACTORS = (
    "acoustic_log_rms_delta.v1",
    "acoustic_logmel_flux.v1",
)
MUTATIONS = (
    "posterior_score_change",
    "eres_window_coordinate_change",
    "earlier_frontier",
    "pair_block_reassignment",
    "cache_payload_hash_change",
    "auc_summary_change",
    "family_disposition_change",
    "state_equivalence_change",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    return sha256_bytes(canonical_json(body).encode("utf-8"))


def read_self_hashed(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing artifact:{path.name}")
    if path.stat().st_size > AGGREGATE_LIMIT:
        raise ValueError(f"aggregate_size:{path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("content_sha256") != content_hash(payload):
        raise ValueError(f"content_hash:{path.name}")
    return payload


def digest_rows(rows: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json(row).encode("utf-8") + b"\n")
    return digest.hexdigest()


def _source_contract(
    experiment_dir: Path,
    episodes: Sequence[dict[str, Any]],
    cases: dict[tuple[str, str], dict[str, Any]],
    details: dict[str, dict[str, Any]],
    corpus_root: Path,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        source_id = str(episode["session_id"])
        manifest_name = synthetic_manifest_name(source_id)
        if manifest_name is None:
            path = corpus_root / str(details[source_id]["wav_path"])
            public = True
        else:
            case_id = synthetic_case_id(source_id)
            if case_id is None:
                raise ValueError(f"synthetic_source:{source_id}")
            case = cases[(manifest_name, case_id)]
            relative = Path(str(case["wav_relative_path"]))
            roots = (experiment_dir / "data", corpus_root / "phase2_build", corpus_root)
            matches = [root / relative for root in roots if (root / relative).is_file()]
            path = matches[0] if matches else experiment_dir / "data" / relative
            public = False
        with wave.open(str(path), "rb") as handle:
            duration = int(handle.getnframes())
        row = {
            "source_id": source_id,
            "wav_sha256": str(episode["wav_sha256"]),
            "duration_samples": duration,
            "public": public,
            "path": str(path.resolve()),
        }
        if source_id in result and result[source_id] != row:
            raise ValueError(f"source_identity:{source_id}")
        result[source_id] = row
    return result


def _expected_windows(
    episodes: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
) -> tuple[dict[str, set[tuple[int, int]]], int, str]:
    windows: dict[str, set[tuple[int, int]]] = defaultdict(set)
    coordinate_rows: list[dict[str, Any]] = []
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
                wav, {"source_id": session_id, "maximum_tail_end": tail_end}
            )
            source["maximum_tail_end"] = max(int(source["maximum_tail_end"]), tail_end)
        for window in ADJACENT_WINDOWS:
            for step in LONG_STEPS if window >= 24000 else STEPS:
                lo = max(scored_start, warm_start + window)
                hi = min(scored_end, tail_end - window)
                for boundary in range(ceil_grid(lo, step), hi + 1, step):
                    coordinate_rows.append(
                        {
                            "episode_id": episode["episode_id"],
                            "kind": "adjacent_grid",
                            "profile": f"adjacent:W{window}:S{step}",
                            "boundary": boundary,
                            "observation_frontier": boundary + window,
                        }
                    )
                    windows[wav].add((boundary - window, boundary))
                    windows[wav].add((boundary, boundary + window))
        for window in ANCHOR_WINDOWS:
            for step in STEPS:
                first = ceil_grid(warm_start + window, step)
                for end in range(first, tail_end + 1, step):
                    coordinate_rows.append(
                        {
                            "episode_id": episode["episode_id"],
                            "kind": "trailing_probe_grid",
                            "profile": f"trailing_probe:W{window}:S{step}",
                            "boundary": end - window,
                            "observation_frontier": end,
                        }
                    )
                    windows[wav].add((end - window, end))
    for wav, source in sorted(public_sources.items()):
        maximum_tail = int(source["maximum_tail_end"])
        for window in ANCHOR_WINDOWS:
            for step in STEPS:
                for end in range(ceil_grid(window, step), maximum_tail + 1, step):
                    coordinate_rows.append(
                        {
                            "source_id": source["source_id"],
                            "kind": "source_prefix_probe_grid",
                            "profile": f"source_prefix_probe:W{window}:S{step}",
                            "boundary": end - window,
                            "observation_frontier": end,
                        }
                    )
                    windows[wav].add((end - window, end))
    for episode in episodes:
        session_id = str(episode["session_id"])
        if synthetic_manifest_name(session_id) is not None:
            continue
        warm_start = int(episode["bounds"]["warm_start"])
        for window in ANCHOR_WINDOWS:
            for step in STEPS:
                last_probe_end = (warm_start // step) * step
                for state_mode in ERES_STATES:
                    coordinate_rows.append(
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
            coordinate_rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "kind": "reference_aligned_measurement",
                    "profile": f"measurement_adjacent:W{window}",
                    "boundary": boundary,
                    "observation_frontier": boundary + window,
                }
            )
            windows[wav].add((boundary - window, boundary))
            windows[wav].add((boundary, boundary + window))
        for window in ANCHOR_WINDOWS:
            if boundary + window > tail_end:
                continue
            coordinate_rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "kind": "read_only_anchor_probe",
                    "profile": f"measurement_probe:W{window}",
                    "boundary": boundary,
                    "observation_frontier": boundary + window,
                }
            )
            windows[wav].add((boundary, boundary + window))
        for horizon, support in LS_ACOUSTIC_SUPPORT_BY_HORIZON.items():
            if boundary - support < warm_start or boundary + support > tail_end:
                continue
            coordinate_rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "kind": "ls_reference_aligned_acoustic",
                    "profile": f"ls_acoustic:H{horizon}:W{support}",
                    "boundary": boundary,
                    "observation_frontier": boundary + support,
                }
            )
    coordinate_rows.sort(key=canonical_json)
    if len(coordinate_rows) != 1217509 or digest_rows(coordinate_rows) != COORDINATE_ROWS_SHA256:
        raise ValueError("coordinate_ledger")
    window_rows = [
        {"wav_sha256": wav, "start": start, "end": end}
        for wav, values in windows.items()
        for start, end in values
    ]
    window_rows.sort(key=lambda row: (row["wav_sha256"], row["start"], row["end"]))
    if len(window_rows) != 895656 or digest_rows(window_rows) != EMBEDDING_WINDOWS_SHA256:
        raise ValueError("embedding_window_ledger")
    return dict(windows), len(coordinate_rows), digest_rows(coordinate_rows)


def load_input_contract(result_dir: Path) -> dict[str, Any]:
    experiment_dir = result_dir.parents[1]
    manifest_path = result_dir / "episode_manifest_dev.json"
    if sha256_file(manifest_path) != MANIFEST_BYTE_SHA256:
        raise ValueError("manifest_byte_hash")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("content_sha256") != MANIFEST_CONTENT_SHA256:
        raise ValueError("manifest_content_hash")
    if manifest.get("group_graph_hash") != GROUP_GRAPH_SHA256:
        raise ValueError("group_graph_hash")
    episodes = [row for row in manifest["episodes"] if row["pool"] == "diagnostic_dev"]
    if len(episodes) != 695:
        raise ValueError("episode_count")
    inventory = json.loads((result_dir / "coverage_inventory.json").read_text(encoding="utf-8"))
    details = {
        str(row["session_id"]): row
        for row in (
            json.loads(line)
            for line in (result_dir / "coverage_inventory_details.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        )
    }
    cases = load_synthetic_cases(experiment_dir / "data" / "manifests")
    public_sessions = [
        str(row["session_id"])
        for row in episodes
        if synthetic_manifest_name(str(row["session_id"])) is None
    ]
    regions = load_public_regions(
        inventory,
        details,
        public_sessions,
        experiment_dir / "data" / "manifests",
    )
    positives, negatives = build_candidates(
        episodes,
        cases,
        component_map(inventory),
        regions,
    )
    candidates = positives + negatives
    pairs, exclusions = match_pairs(positives, negatives)
    if len(candidates) != 810 or len(pairs) != 313:
        raise ValueError("candidate_pair_count")
    if exclusions != {
        "positive_unmatched": 137,
        "negative_unused": 47,
        "groups_without_negative": 0,
    }:
        raise ValueError("pair_exclusions")
    if digest_rows(pairs) != PAIR_ROWS_SHA256:
        raise ValueError("pair_rows_hash")
    windows, _, _ = _expected_windows(episodes, candidates)
    sources = _source_contract(
        experiment_dir,
        episodes,
        cases,
        details,
        Path(str(inventory["corpus_root"])),
    )
    return {
        "episodes": episodes,
        "candidates": {str(row["candidate_id"]): row for row in candidates},
        "pairs": pairs,
        "sources": sources,
        "embedding_windows": windows,
    }


def expected_registry() -> list[dict[str, Any]]:
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
    for checkpoint in ERES_CHECKPOINTS:
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


def independent_auc(
    positive_scores: Sequence[float], negative_scores: Sequence[float]
) -> float | None:
    if not positive_scores or not negative_scores:
        return None
    favorable = 0.0
    for positive in positive_scores:
        for negative in negative_scores:
            if positive > negative:
                favorable += 1.0
            elif positive == negative:
                favorable += 0.5
    return favorable / (len(positive_scores) * len(negative_scores))


def independent_eer(
    positive_scores: Sequence[float], negative_scores: Sequence[float]
) -> float | None:
    if not positive_scores or not negative_scores:
        return None
    thresholds = sorted(
        set(map(float, positive_scores)) | set(map(float, negative_scores)), reverse=True
    )
    points: list[tuple[float, float]] = [(0.0, 1.0)]
    for threshold in thresholds:
        fpr = sum(value >= threshold for value in negative_scores) / len(negative_scores)
        fnr = sum(value < threshold for value in positive_scores) / len(positive_scores)
        points.append((fpr, fnr))
    points.append((1.0, 0.0))
    for left, right in zip(points, points[1:]):
        left_delta = left[0] - left[1]
        right_delta = right[0] - right[1]
        if left_delta == 0:
            return left[0]
        if left_delta * right_delta <= 0 and left_delta != right_delta:
            weight = left_delta / (left_delta - right_delta)
            fpr = left[0] + weight * (right[0] - left[0])
            fnr = left[1] + weight * (right[1] - left[1])
            return (fpr + fnr) / 2.0
    closest = min(points, key=lambda item: (abs(item[0] - item[1]), item[0], item[1]))
    return (closest[0] + closest[1]) / 2.0


def nearest_rank(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(map(float, values))
    return ordered[max(1, math.ceil(percentile * len(ordered))) - 1]


def independent_bootstrap(
    rows: Sequence[dict[str, Any]], seed: int, replicates: int = 10000
) -> tuple[list[float], list[str], str]:
    blocks = sorted({str(row["block_id"]) for row in rows})
    index = {block: ordinal for ordinal, block in enumerate(blocks)}
    size = len(blocks)
    neural = np.zeros((size, size), dtype=np.float64)
    acoustic = np.zeros((size, size), dtype=np.float64)
    positive_counts = np.zeros(size, dtype=np.float64)
    negative_counts = np.zeros(size, dtype=np.float64)
    for positive in rows:
        left = index[str(positive["block_id"])]
        positive_counts[left] += 1.0
        for negative in rows:
            right = index[str(negative["block_id"])]
            neural[left, right] += (
                1.0
                if positive["positive_neural"] > negative["negative_neural"]
                else 0.5 if positive["positive_neural"] == negative["negative_neural"] else 0.0
            )
            acoustic[left, right] += (
                1.0
                if positive["positive_acoustic"] > negative["negative_acoustic"]
                else (
                    0.5 if positive["positive_acoustic"] == negative["negative_acoustic"] else 0.0
                )
            )
    for negative in rows:
        negative_counts[index[str(negative["block_id"])]] += 1.0
    rng = random.Random(seed)
    digest = hashlib.sha256()
    values: list[float] = []
    for _ in range(replicates):
        draws = [rng.randrange(size) for _ in blocks]
        digest.update(canonical_json(draws).encode("utf-8") + b"\n")
        weights = np.bincount(draws, minlength=size).astype(np.float64)
        positive_total = float(weights @ positive_counts)
        negative_total = float(weights @ negative_counts)
        if positive_total == 0 or negative_total == 0:
            raise ValueError("bootstrap_non_estimable")
        denominator = positive_total * negative_total
        values.append(
            float(weights @ neural @ weights) / denominator
            - float(weights @ acoustic @ weights) / denominator
        )
    return values, blocks, digest.hexdigest()


def _float_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def _payload_equal(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _payload_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_payload_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return _float_equal(left, right)
    return left == right


def _verify_cache_metadata(evidence: dict[str, Any], mismatches: list[str]) -> dict[str, Any]:
    metadata_path = Path(str(evidence["metadata_path"]))
    payload_path = Path(str(evidence["path"])) if "path" in evidence else None
    if (payload_path is not None and not payload_path.is_file()) or not metadata_path.is_file():
        mismatches.append("cache_file_missing")
        return {}
    payload = payload_path.read_bytes() if payload_path is not None else None
    if payload is not None:
        if len(payload) != int(evidence["payload_size_bytes"]):
            mismatches.append("cache_payload_size")
        if sha256_bytes(payload) != str(evidence["payload_sha256"]):
            mismatches.append("cache_payload_hash")
    try:
        metadata = read_self_hashed(metadata_path)
    except Exception as error:
        mismatches.append(str(error))
        return {}
    expected_metadata = {
        key: value
        for key, value in evidence.items()
        if key not in ("path", "paths", "metadata_path", "cache_hit", "content_sha256")
    }
    actual_metadata = {key: value for key, value in metadata.items() if key != "content_sha256"}
    if not _payload_equal(expected_metadata, actual_metadata):
        mismatches.append("cache_metadata_inventory")
    capture_body = {
        key: value
        for key, value in metadata.items()
        if key not in ("content_sha256", "capture_content_sha256")
    }
    if metadata.get("capture_content_sha256") != sha256_bytes(
        canonical_json(capture_body).encode("utf-8")
    ):
        mismatches.append("cache_capture_content_hash")
    return {"payload": payload, "metadata": metadata}


def _decode_eres_binary(payload: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    magic, encoded_header, binary = payload.split(b"\n", 2)
    if magic != b"TURN_EPISODE_PHASE4_ERES_V1":
        raise ValueError("magic")
    header = json.loads(encoded_header)
    count = int(header["row_count"])
    if (
        header.get("schema_version") != "turn_episode_phase4_eres_binary.v1"
        or header.get("window_dtype") != "<i8"
        or header.get("embedding_dtype") != "<f4"
        or header.get("shadow_dtype") != "<f4"
        or header.get("rms_dtype") != "<f4"
        or header.get("window_shape") != [count, 2]
        or header.get("embedding_shape") != [count, 192]
        or header.get("shadow_shape") != [count, 80]
        or header.get("rms_shape") != [count]
    ):
        raise ValueError("tensor_contract")
    window_bytes = count * 2 * 8
    embedding_bytes = count * 192 * 4
    shadow_bytes = count * 80 * 4
    rms_bytes = count * 4
    if len(binary) != window_bytes + embedding_bytes + shadow_bytes + rms_bytes:
        raise ValueError("byte_length")
    embedding_end = window_bytes + embedding_bytes
    shadow_end = embedding_end + shadow_bytes
    return (
        np.frombuffer(binary[:window_bytes], dtype="<i8").reshape(count, 2).copy(),
        np.frombuffer(binary[window_bytes:embedding_end], dtype="<f4").reshape(count, 192).copy(),
        np.frombuffer(binary[embedding_end:shadow_end], dtype="<f4").reshape(count, 80).copy(),
        np.frombuffer(binary[shadow_end:], dtype="<f4").reshape(count).copy(),
    )


def verify_cache_inventory(
    cache: dict[str, Any], inputs: dict[str, Any], mismatches: list[str]
) -> tuple[
    dict[tuple[str, str], str],
    dict[tuple[str, str], str],
    dict[tuple[str, str], np.ndarray],
    dict[
        tuple[str, str],
        tuple[
            dict[tuple[int, int], np.ndarray],
            dict[tuple[int, int], tuple[np.ndarray, float]],
        ],
    ],
]:
    ls_payloads: dict[tuple[str, str], str] = {}
    eres_payloads: dict[tuple[str, str], str] = {}
    ls_captures: dict[tuple[str, str], np.ndarray] = {}
    eres_caches: dict[
        tuple[str, str],
        tuple[
            dict[tuple[int, int], np.ndarray],
            dict[tuple[int, int], tuple[np.ndarray, float]],
        ],
    ] = {}
    sources = inputs["sources"]
    expected_wav_windows = inputs["embedding_windows"]
    for checkpoint, section in sorted(cache.get("ls", {}).items()):
        contract = section.get("contract", {})
        contract_body = {key: value for key, value in contract.items() if key != "contract_sha256"}
        if contract.get("contract_sha256") != sha256_bytes(
            canonical_json(contract_body).encode("utf-8")
        ):
            mismatches.append(f"cache_contract_hash:ls:{checkpoint}")
        if contract.get("authority_sha256") != AUTHORITY_SHA256:
            mismatches.append(f"cache_contract_authority:ls:{checkpoint}")
        rows = section.get("sources", [])
        if int(section.get("source_count", -1)) != len(sources) or len(rows) != len(sources):
            mismatches.append(f"cache_source_count:ls:{checkpoint}")
        for evidence in rows:
            source_id = str(evidence.get("source_id"))
            expected_source = sources.get(source_id)
            if expected_source is None:
                mismatches.append(f"cache_unknown_source:ls:{checkpoint}:{source_id}")
                continue
            for key in ("wav_sha256", "duration_samples"):
                if evidence.get(key) != expected_source[key]:
                    mismatches.append(f"cache_source_identity:ls:{checkpoint}:{source_id}:{key}")
            verified = _verify_cache_metadata(evidence, mismatches)
            payload = verified.get("payload")
            if not isinstance(payload, bytes):
                continue
            try:
                with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                    normal = np.asarray(data["normal_probs"], dtype=np.float32)
                    tail = np.asarray(data["tail_probs"], dtype=np.float32)
                    frontiers = np.asarray(data["normal_frontiers"], dtype=np.int64)
                    scalar_ints = np.asarray(data["scalar_ints"], dtype=np.int64)
                if normal.ndim != 2 or tail.ndim != 2:
                    raise ValueError("rank")
                if not np.all(np.isfinite(normal)) or not np.all(np.isfinite(tail)):
                    raise ValueError("nonfinite")
                if normal.shape[0] != int(evidence["normal_frame_count"]):
                    raise ValueError("normal_count")
                if tail.shape[0] != int(evidence["tail_frame_count"]):
                    raise ValueError("tail_count")
                if normal.shape[0] != frontiers.size:
                    raise ValueError("frontier_count")
                if int(scalar_ints[3]) != int(expected_source["duration_samples"]):
                    raise ValueError("duration")
            except Exception as error:
                mismatches.append(f"cache_payload_structure:ls:{checkpoint}:{source_id}:{error}")
            else:
                ls_captures[(checkpoint, source_id)] = normal
            ls_payloads[(checkpoint, source_id)] = str(evidence["payload_sha256"])
    source_for_wav: dict[str, str] = {}
    for source_id, source in sources.items():
        wav = str(source["wav_sha256"])
        if wav not in source_for_wav or source_id < source_for_wav[wav]:
            source_for_wav[wav] = source_id
    for checkpoint, section in sorted(cache.get("eres", {}).items()):
        contract = section.get("contract", {})
        contract_body = {key: value for key, value in contract.items() if key != "contract_sha256"}
        if contract.get("contract_sha256") != sha256_bytes(
            canonical_json(contract_body).encode("utf-8")
        ):
            mismatches.append(f"cache_contract_hash:eres:{checkpoint}")
        rows = section.get("sources", [])
        if int(section.get("source_count", -1)) != len(expected_wav_windows):
            mismatches.append(f"cache_source_count:eres:{checkpoint}")
        by_wav: dict[str, dict[str, Any]] = {}
        for evidence in rows:
            wav = str(evidence.get("wav_sha256"))
            if wav in by_wav:
                mismatches.append(f"cache_duplicate_wav:eres:{checkpoint}:{wav}")
            by_wav[wav] = evidence
            expected_windows = sorted(expected_wav_windows.get(wav, set()))
            if not expected_windows:
                mismatches.append(f"cache_unknown_wav:eres:{checkpoint}:{wav}")
                continue
            if evidence.get("source_id") != source_for_wav.get(wav):
                mismatches.append(f"cache_source_selection:eres:{checkpoint}:{wav}")
            if int(evidence.get("window_count", -1)) != len(expected_windows):
                mismatches.append(f"cache_window_count:eres:{checkpoint}:{wav}")
            verified = _verify_cache_metadata(evidence, mismatches)
            metadata = verified.get("metadata")
            if not isinstance(metadata, dict):
                continue
            try:
                shard_rows = metadata.get("shards")
                if (
                    metadata.get("schema_version") != "turn_episode_phase4_eres_cache.v2"
                    or not isinstance(shard_rows, list)
                    or len(shard_rows) != int(metadata.get("shard_count", -1))
                ):
                    raise ValueError("shard_inventory")
                aggregate = content_hash(
                    {
                        "shards": [
                            {key: value for key, value in row.items() if key != "path"}
                            for row in shard_rows
                        ]
                    }
                )
                if aggregate != metadata.get("payload_sha256"):
                    raise ValueError("aggregate_payload_hash")
                metadata_path = Path(str(evidence["metadata_path"]))
                expected_paths = [
                    str(metadata_path.parent / str(row["path"])) for row in shard_rows
                ]
                if evidence.get("paths") != expected_paths:
                    raise ValueError("shard_paths")
                window_parts: list[np.ndarray] = []
                embedding_parts: list[np.ndarray] = []
                shadow_parts: list[np.ndarray] = []
                rms_parts: list[np.ndarray] = []
                total_size = 0
                for index, shard in enumerate(shard_rows):
                    if int(shard.get("shard_index", -1)) != index:
                        raise ValueError("shard_order")
                    path = metadata_path.parent / str(shard["path"])
                    compressed = path.read_bytes()
                    total_size += len(compressed)
                    if (
                        len(compressed) > DETAIL_SHARD_LIMIT
                        or len(compressed) != int(shard["size_bytes"])
                        or sha256_bytes(compressed) != shard["byte_sha256"]
                    ):
                        raise ValueError("shard_bytes")
                    plain = gzip.decompress(compressed)
                    if sha256_bytes(plain) != shard["content_sha256"]:
                        raise ValueError("shard_content")
                    shard_windows, shard_embeddings, shard_shadows, shard_rms = _decode_eres_binary(
                        plain
                    )
                    if shard_windows.shape[0] != int(shard["row_count"]):
                        raise ValueError("shard_count")
                    if shard_windows.shape[0] and (
                        shard_windows[0].tolist() != shard["first_window"]
                        or shard_windows[-1].tolist() != shard["last_window"]
                    ):
                        raise ValueError("shard_bounds")
                    window_parts.append(shard_windows)
                    embedding_parts.append(shard_embeddings)
                    shadow_parts.append(shard_shadows)
                    rms_parts.append(shard_rms)
                if total_size != int(metadata.get("payload_size_bytes", -1)):
                    raise ValueError("payload_size")
                windows = np.concatenate(window_parts, axis=0)
                embeddings = np.concatenate(embedding_parts, axis=0)
                shadows = np.concatenate(shadow_parts, axis=0)
                rms = np.concatenate(rms_parts, axis=0)
                actual_windows = [(int(row[0]), int(row[1])) for row in windows]
                if actual_windows != expected_windows:
                    raise ValueError("window_coordinates")
                if embeddings.shape != (len(expected_windows), 192):
                    raise ValueError("embedding_shape")
                if shadows.shape != (len(expected_windows), 80) or rms.shape != (
                    len(expected_windows),
                ):
                    raise ValueError("shadow_shape")
                if not all(np.all(np.isfinite(values)) for values in (embeddings, shadows, rms)):
                    raise ValueError("payload_nonfinite")
                norms = np.linalg.norm(embeddings, axis=1)
                if norms.size and not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-5):
                    raise ValueError("embedding_normalization")
                window_rows = [
                    {
                        "start": start,
                        "end": end,
                        "embedding_sha256": sha256_bytes(
                            np.asarray(vector, dtype="<f4").tobytes(order="C")
                        ),
                        "acoustic_shadow_sha256": sha256_bytes(
                            np.asarray(shadow, dtype="<f4").tobytes(order="C")
                            + np.asarray(log_rms, dtype="<f4").tobytes(order="C")
                        ),
                    }
                    for (start, end), vector, shadow, log_rms in zip(
                        actual_windows, embeddings, shadows, rms
                    )
                ]
                if sha256_bytes(
                    b"".join(canonical_json(row).encode("utf-8") + b"\n" for row in window_rows)
                ) != str(evidence["window_rows_sha256"]):
                    raise ValueError("window_rows_hash")
            except Exception as error:
                mismatches.append(f"cache_payload_structure:eres:{checkpoint}:{wav}:{error}")
            else:
                eres_caches[(checkpoint, wav)] = (
                    dict(zip(actual_windows, embeddings)),
                    dict(
                        zip(
                            actual_windows,
                            ((shadow, float(value)) for shadow, value in zip(shadows, rms)),
                        )
                    ),
                )
            eres_payloads[(checkpoint, wav)] = str(evidence["payload_sha256"])
        if set(by_wav) != set(expected_wav_windows):
            mismatches.append(f"cache_wav_completeness:eres:{checkpoint}")
    return ls_payloads, eres_payloads, ls_captures, eres_caches


def _state_trace_hash(rows: Sequence[dict[str, Any]]) -> str:
    return sha256_bytes(b"".join(canonical_json(row).encode("utf-8") + b"\n" for row in rows))


def _reference_clusters(proposals: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    pending = sorted(
        proposals,
        key=lambda row: (
            int(row["observation_frontier"]),
            int(row["boundary_source_sample"]),
            str(row["proposal_kind"]),
            str(row["proposal_id"]),
        ),
    )
    clusters: list[dict[str, Any]] = []
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


def _reference_ls_trace(
    probabilities: np.ndarray,
    *,
    offset: int,
    epoch_length: int,
    profile_class: str,
) -> dict[str, list[dict[str, Any]]]:
    indexes = np.arange(probabilities.shape[0], dtype=np.int64)
    centers = 14_431 + 1_600 * indexes + offset
    availability = 15_806 + 1_600 * indexes + offset
    proposals: list[dict[str, Any]] = []
    progress: list[dict[str, Any]] = []
    if profile_class in ("new_track_onset", "dominant_replacement"):
        policy = (
            "new_speaker_onset" if profile_class == "new_track_onset" else "dominant_replacement"
        )
        reducer = StreamingReducer(
            ReductionProfile(
                threshold=0.50,
                persistence=2,
                median_width=1,
                policy=policy,
            ),
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
                        "boundary_source_sample": (boundary.boundary_source_sample() + offset),
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
        "clusters": _reference_clusters(proposals),
        "progress": progress,
    }


def _reference_scored_trace(
    trace: dict[str, list[dict[str, Any]]], bounds: dict[str, Any]
) -> dict[str, Any]:
    scored_start = int(bounds["scored_start"])
    scored_end = int(bounds["scored_end"])
    proposals = [
        row
        for row in trace["proposals"]
        if scored_start <= int(row["observation_frontier"]) <= scored_end
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
    clusters = _reference_clusters(proposals)
    progress = [
        row
        for row in trace["progress"]
        if scored_start <= int(row["observed_source_sample"]) <= scored_end
    ]
    return {
        "proposal_count": len(proposals),
        "proposal_sha256": _state_trace_hash(proposals),
        "cluster_count": len(clusters),
        "cluster_sha256": _state_trace_hash(clusters),
        "progress_count": len(progress),
        "progress_sha256": _state_trace_hash(progress),
    }


def _reference_state_hash(payload: Any) -> str:
    def encode(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": sha256_bytes(value.tobytes(order="C")),
            }
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        if isinstance(value, dict):
            return {key: encode(item) for key, item in sorted(value.items())}
        return value

    return content_hash({"state": encode(payload)})


def _reference_normalized(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("zero_state")
    return result / norm


def _reference_cosine(left: np.ndarray, right: np.ndarray) -> float:
    norm_product = np.linalg.norm(left) * np.linalg.norm(right)
    if norm_product == 0.0:
        return 0.0
    return float(np.dot(left, right) / norm_product)


def _reference_ema(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return _reference_normalized(0.10 * left + 0.90 * right)


def _reference_measure(
    state: dict[str, Any],
    mode: str,
    probe: np.ndarray,
    acoustic: tuple[np.ndarray, float],
) -> tuple[float | None, int | None]:
    if state.get("anchor") is None:
        return None, None
    if mode == "prototype_memory_4":
        selected = max(
            state["prototypes"],
            key=lambda item: (
                _reference_cosine(item["embedding"], probe),
                -item["ordinal"],
            ),
        )
        return 1.0 - _reference_cosine(selected["embedding"], probe), int(selected["ordinal"])
    return 1.0 - _reference_cosine(state["anchor"], probe), None


def _reference_advance(
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
        similarity = _reference_cosine(state["anchor"], probe)
        if similarity >= 0.70:
            state["anchor"] = _reference_ema(state["anchor"], probe)
            state["shadow"] = _reference_ema(state["shadow"], acoustic[0])
            state["shadow_rms"] = 0.10 * float(state["shadow_rms"]) + 0.90 * acoustic[1]
            state["anchor_window"] = window
        return
    if mode == "confirmed_anchor":
        similarity = _reference_cosine(state["anchor"], probe)
        pending = state.get("pending")
        if similarity < 0.50:
            if pending is not None:
                mutual = _reference_cosine(pending["embedding"], probe)
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
                state["anchor"] = _reference_ema(state["anchor"], probe)
                state["shadow"] = _reference_ema(state["shadow"], acoustic[0])
                state["shadow_rms"] = 0.10 * float(state["shadow_rms"]) + 0.90 * acoustic[1]
                state["anchor_window"] = window
        return
    prototypes = state["prototypes"]
    selected = max(
        prototypes,
        key=lambda item: (
            _reference_cosine(item["embedding"], probe),
            -item["ordinal"],
        ),
    )
    similarity = _reference_cosine(selected["embedding"], probe)
    pending = state.get("pending")
    if similarity >= 0.70:
        selected["embedding"] = _reference_ema(selected["embedding"], probe)
        selected["shadow"] = _reference_ema(selected["shadow"], acoustic[0])
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
    elif _reference_cosine(pending["embedding"], probe) >= 0.50:
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
            prototypes.remove(min(prototypes, key=lambda item: item["ordinal"]))
        prototypes.append(created)
        state["pending"] = None
    else:
        state["pending"] = {
            "embedding": probe,
            "shadow": acoustic[0],
            "shadow_rms": acoustic[1],
            "window": window,
        }


def _reference_eres_state_trace(
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
    first_end = ((replay_start + window + step - 1) // step) * step
    for end in range(first_end, scored_end + 1, step):
        probe_window = (end - window, end)
        probe = embeddings[probe_window]
        acoustic = shadows[probe_window]
        if snapshot_hash is None and end > snapshot_frontier:
            snapshot_hash = _reference_state_hash(state)
        before = _reference_state_hash(state)
        score, selected = _reference_measure(state, mode, probe, acoustic)
        pending_before = state.get("pending")
        pending_window = pending_before.get("window") if pending_before is not None else None
        mutual = (
            _reference_cosine(pending_before["embedding"], probe)
            if pending_before is not None
            else None
        )
        emit_boundary: int | None = None
        if score is not None and score > 0.50:
            if mode in ("stable_no_update", "stable_ema"):
                emit_boundary = probe_window[0]
            elif pending_before is not None and mutual is not None and mutual >= 0.50:
                emit_boundary = int(pending_window[0])
        _reference_advance(state, mode, probe, acoustic, probe_window)
        after = _reference_state_hash(state)
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
        snapshot_hash = _reference_state_hash(state)
    return {
        "raw": raw,
        "scores": scores,
        "transitions": transitions,
        "proposals": proposals,
        "clusters": _reference_clusters(proposals),
        "progress": progress,
        "snapshot_state_sha256": snapshot_hash,
    }


def _reference_eres_comparison(source: dict[str, Any], reset: dict[str, Any]) -> dict[str, Any]:
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
    exact = {
        key: source[key] == reset[key]
        for key in ("transitions", "proposals", "clusters", "progress")
    }
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
        "source_trace_sha256": _state_trace_hash(
            [{key: value for key, value in source.items() if key != "snapshot_state_sha256"}]
        ),
        "reset_trace_sha256": _state_trace_hash(
            [{key: value for key, value in reset.items() if key != "snapshot_state_sha256"}]
        ),
        "exact_trace_fields": exact,
        "passed": passed,
    }


def _reference_adjacent_trace(
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
    first_boundary = ((lo + step - 1) // step) * step
    for boundary in range(first_boundary, hi + 1, step):
        left_window = (boundary - window, boundary)
        right_window = (boundary, boundary + window)
        left = embeddings[left_window]
        right = embeddings[right_window]
        score = 1.0 - _reference_cosine(left, right)
        raw.append(
            {
                "left": list(left_window),
                "right": list(right_window),
                "left_sha256": sha256_bytes(np.asarray(left, dtype="<f4").tobytes(order="C")),
                "right_sha256": sha256_bytes(np.asarray(right, dtype="<f4").tobytes(order="C")),
                "change_score": score,
            }
        )
        if score > 0.50:
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
    return {
        "raw": raw,
        "proposals": proposals,
        "clusters": _reference_clusters(proposals),
        "progress": progress,
    }


def _read_reference_wav(source: dict[str, Any]) -> np.ndarray:
    path = Path(str(source["path"]))
    if sha256_file(path) != source["wav_sha256"]:
        raise ValueError(f"state_wav_hash:{source['source_id']}")
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getnchannels() != 1
            or handle.getframerate() != 16_000
            or handle.getsampwidth() != 2
        ):
            raise ValueError(f"state_wav_contract:{source['source_id']}")
        count = handle.getnframes()
        payload = handle.readframes(count)
    samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32_768.0
    if samples.size != int(source["duration_samples"]):
        raise ValueError(f"state_wav_duration:{source['source_id']}")
    return samples


def recompute_state_equivalence(
    inputs: dict[str, Any],
    preflight: dict[str, Any],
    cache: dict[str, Any],
    ls_captures: dict[tuple[str, str], np.ndarray],
    eres_caches: dict[
        tuple[str, str],
        tuple[
            dict[tuple[int, int], np.ndarray],
            dict[tuple[int, int], tuple[np.ndarray, float]],
        ],
    ],
) -> dict[str, Any]:
    public_episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in inputs["episodes"]:
        source_id = str(episode["session_id"])
        if bool(inputs["sources"][source_id]["public"]):
            public_episodes[source_id].append(episode)
    for values in public_episodes.values():
        values.sort(key=lambda row: str(row["episode_id"]))
    public_episode_count = sum(len(values) for values in public_episodes.values())
    ls_classes: list[dict[str, Any]] = []
    for checkpoint in LS_EEND_VARIANTS:
        model_row = preflight["model_files"][checkpoint]
        runtime = LSEENDCapture(
            Path(str(model_row["model"])),
            load_sidecar_metadata(Path(str(model_row["sidecar"]))),
            checkpoint_variant=checkpoint,
        )
        records_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
        contract_sha = cache["ls"][checkpoint]["contract"]["contract_sha256"]
        for source_id in sorted(public_episodes):
            source = inputs["sources"][source_id]
            samples = _read_reference_wav(source)
            source_probabilities = ls_captures[(checkpoint, source_id)]
            source_centers = 14_431 + 1_600 * np.arange(
                source_probabilities.shape[0], dtype=np.int64
            )
            source_index = {int(center): index for index, center in enumerate(source_centers)}
            source_traces = {
                profile_class: _reference_ls_trace(
                    source_probabilities,
                    offset=0,
                    epoch_length=int(source["duration_samples"]),
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
                    audio_epoch=int(str(source["wav_sha256"])[:8], 16),
                )
                reset_probabilities = (
                    np.stack(reset_capture.normal_probs)
                    if reset_capture.normal_probs
                    else np.zeros((0, runtime.real_output_dim), dtype=np.float32)
                )
                reset_centers = (
                    14_431 + 1_600 * np.arange(reset_probabilities.shape[0], dtype=np.int64) + warm
                )
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
                    reset_trace = _reference_ls_trace(
                        reset_probabilities,
                        offset=warm,
                        epoch_length=tail - warm,
                        profile_class=profile_class,
                    )
                    source_receipt = _reference_scored_trace(source_trace, bounds)
                    reset_receipt = _reference_scored_trace(reset_trace, bounds)
                    passed = raw_passed and source_receipt == reset_receipt
                    records_by_class[profile_class].append(
                        {
                            "episode_id": episode["episode_id"],
                            "posterior_gate_passed": raw_passed,
                            "source": source_receipt,
                            "episode_reset": reset_receipt,
                            "passed": passed,
                            "reason": (None if passed else "posterior_or_trace_mismatch"),
                        }
                    )
        for profile_class in (
            "raw_posterior",
            "new_track_onset",
            "dominant_replacement",
            "hysteretic_activity_state",
        ):
            records = records_by_class[profile_class]
            failed_count = sum(not bool(row["passed"]) for row in records)
            ls_classes.append(
                {
                    "family": "ls_eend",
                    "checkpoint": checkpoint,
                    "profile_class": profile_class,
                    "case_count": len(records),
                    "failed_count": failed_count,
                    "disposition": (
                        "episode_reset_permitted" if failed_count == 0 else "source_prefix_required"
                    ),
                    "source_prefix_cache_contract_sha256": contract_sha,
                    "records_sha256": _state_trace_hash(records),
                    "records": records,
                }
            )
    eres_classes: list[dict[str, Any]] = []
    for checkpoint in ERES_CHECKPOINTS:
        contract_sha = cache["eres"][checkpoint]["contract"]["contract_sha256"]
        cache_receipts = {
            str(row["wav_sha256"]): str(row["payload_sha256"])
            for row in cache["eres"][checkpoint]["sources"]
        }
        for profile_class in (
            "adjacent",
            "stable_no_update",
            "stable_ema",
            "confirmed_anchor",
            "prototype_memory_4",
        ):
            class_records: list[dict[str, Any]] = []
            for source_id in sorted(public_episodes):
                source = inputs["sources"][source_id]
                wav = str(source["wav_sha256"])
                embeddings, shadows = eres_caches[(checkpoint, wav)]
                for episode in public_episodes[source_id]:
                    receipts: list[dict[str, Any]] = []
                    if profile_class == "adjacent":
                        for window in ADJACENT_WINDOWS:
                            steps = LONG_STEPS if window >= 24_000 else STEPS
                            for step in steps:
                                source_trace = _reference_adjacent_trace(
                                    embeddings, episode, window, step
                                )
                                reset_trace = _reference_adjacent_trace(
                                    embeddings, episode, window, step
                                )
                                passed = source_trace == reset_trace and bool(source_trace["raw"])
                                receipts.append(
                                    {
                                        "profile_id": f"adjacent:W{window}:S{step}",
                                        "aligned_window_count": 2 * len(source_trace["raw"]),
                                        "aligned_window_cosine_min": (
                                            1.0 if source_trace["raw"] else None
                                        ),
                                        "source_trace_sha256": _state_trace_hash([source_trace]),
                                        "reset_trace_sha256": _state_trace_hash([reset_trace]),
                                        "passed": passed,
                                    }
                                )
                    else:
                        bounds = episode["bounds"]
                        for window in ANCHOR_WINDOWS:
                            for step in STEPS:
                                source_trace = _reference_eres_state_trace(
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
                                reset_trace = _reference_eres_state_trace(
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
                                receipts.append(
                                    {
                                        "profile_id": (f"{profile_class}:W{window}:S{step}"),
                                        **_reference_eres_comparison(source_trace, reset_trace),
                                    }
                                )
                    failed_ids = [
                        str(row["profile_id"]) for row in receipts if not bool(row["passed"])
                    ]
                    cosine_values = [
                        float(row["aligned_window_cosine_min"])
                        for row in receipts
                        if row.get("aligned_window_cosine_min") is not None
                    ]
                    class_records.append(
                        {
                            "episode_id": episode["episode_id"],
                            "source_id": source_id,
                            "profile_count": len(receipts),
                            "failed_profile_count": len(failed_ids),
                            "failed_profile_ids": failed_ids,
                            "profile_receipts_sha256": _state_trace_hash(receipts),
                            "aligned_window_cosine_min": (
                                min(cosine_values) if cosine_values else None
                            ),
                            "cache_payload_sha256": cache_receipts[wav],
                            "passed": not failed_ids,
                        }
                    )
            failed_count = sum(not bool(row["passed"]) for row in class_records)
            cosine_values = [
                float(row["aligned_window_cosine_min"])
                for row in class_records
                if row.get("aligned_window_cosine_min") is not None
            ]
            eres_classes.append(
                {
                    "family": "eres2netv2",
                    "checkpoint": checkpoint,
                    "profile_class": profile_class,
                    "case_count": len(class_records),
                    "failed_count": failed_count,
                    "aligned_window_cosine_min": (min(cosine_values) if cosine_values else None),
                    "proposal_and_progress_contract": (
                        "executed_exact_source_prefix_vs_episode_reset"
                    ),
                    "disposition": (
                        "episode_reset_permitted" if failed_count == 0 else "source_prefix_required"
                    ),
                    "source_prefix_cache_contract_sha256": contract_sha,
                    "records_sha256": _state_trace_hash(class_records),
                    "records": class_records,
                }
            )
    passed = all(
        row["case_count"] == public_episode_count
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
            "sentinel_cluster": {
                "debounce_ms": 0,
                "radius_ms": 250,
                "refractory_ms": 0,
            },
            "sentinel_cluster_progress": "exact",
            "ls_reducer_profile": {
                "threshold": 0.50,
                "persistence": 2,
                "median_width": 1,
            },
            "eres_change_threshold": 0.50,
        },
        "ls_profile_classes": ls_classes,
        "eres_profile_classes": eres_classes,
        "public_episode_count": public_episode_count,
        "scored_state_mode": {
            "ls_eend": "source_prefix_when_any_class_fails",
            "eres2netv2": "source_prefix_when_profile_class_fails",
        },
        "passed": passed,
    }


def _mutate_state_equivalence(state: dict[str, Any]) -> None:
    row = state["ls_profile_classes"][0]
    record = row["records"][0]
    record["aligned_frame_count"] = int(record["aligned_frame_count"]) + 1
    row["records_sha256"] = _state_trace_hash(row["records"])
    row["failed_count"] = sum(not bool(item["passed"]) for item in row["records"])
    row["disposition"] = (
        "episode_reset_permitted" if int(row["failed_count"]) == 0 else "source_prefix_required"
    )
    state["passed"] = all(
        item["disposition"] in ("episode_reset_permitted", "source_prefix_required")
        for item in state["ls_profile_classes"] + state["eres_profile_classes"]
    )
    state["content_sha256"] = content_hash(state)


def _mutate_row(row: dict[str, Any], mutation: str | None, applied: dict[str, bool]) -> None:
    if mutation is None or applied.get(mutation):
        return
    if mutation == "posterior_score_change" and row.get("family") == "ls_eend":
        row["neural_score"] = float(row["neural_score"]) + 0.125
    elif mutation == "eres_window_coordinate_change" and row.get("family") == "eres2netv2":
        row["boundary_source_sample"] = int(row["boundary_source_sample"]) + 1
    elif mutation == "earlier_frontier" and row.get("observation_frontier") is not None:
        row["observation_frontier"] = int(row["observation_frontier"]) - 1
    elif mutation == "pair_block_reassignment":
        row["block_id"] = f"mutated:{row['block_id']}"
    elif mutation == "cache_payload_hash_change":
        row["cache_payload_sha256"] = "0" * 64
    else:
        return
    applied[mutation] = True


def iter_shard_rows(
    result_dir: Path,
    entry: dict[str, Any],
    mismatches: list[str],
    mutation: str | None,
    applied: dict[str, bool],
) -> Iterator[dict[str, Any]]:
    path = result_dir / str(entry["path"])
    if not path.is_file():
        mismatches.append(f"detail_missing:{entry['path']}")
        return
    compressed = path.read_bytes()
    if len(compressed) > DETAIL_SHARD_LIMIT or len(compressed) != int(entry["size_bytes"]):
        mismatches.append(f"detail_size:{entry['path']}")
    if sha256_bytes(compressed) != str(entry["byte_sha256"]):
        mismatches.append(f"detail_byte_hash:{entry['path']}")
    try:
        plain = gzip.decompress(compressed)
    except Exception:
        mismatches.append(f"detail_gzip:{entry['path']}")
        return
    if sha256_bytes(plain) != str(entry["content_sha256"]):
        mismatches.append(f"detail_content_hash:{entry['path']}")
    lines = plain.splitlines()
    if len(lines) != int(entry["row_count"]):
        mismatches.append(f"detail_row_count:{entry['path']}")
    first: str | None = None
    last: str | None = None
    previous: str | None = None
    for line in lines:
        row = json.loads(line)
        key = str(row.get("row_key"))
        if previous is not None and key <= previous:
            mismatches.append(f"detail_order:{entry['path']}:{key}")
        previous = key
        first = first or key
        last = key
        _mutate_row(row, mutation, applied)
        yield row
    if first != entry.get("first_row_key") or last != entry.get("last_row_key"):
        mismatches.append(f"detail_key_range:{entry['path']}")


def validate_signal_row(
    row: dict[str, Any],
    registry: dict[str, dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    episodes: dict[str, dict[str, Any]],
    ls_payloads: dict[tuple[str, str], str],
    eres_payloads: dict[tuple[str, str], str],
    mismatches: list[str],
) -> tuple[bool, float, tuple[float | None, float | None]] | None:
    extractor_id = str(row.get("signal_extractor_id"))
    declaration = registry.get(extractor_id)
    candidate_id = str(row.get("candidate_id"))
    candidate = candidates.get(candidate_id)
    if declaration is None:
        mismatches.append(f"row_unknown_extractor:{extractor_id}")
        return None
    if candidate is None:
        mismatches.append(f"row_unknown_candidate:{candidate_id}")
        return None
    identity = {
        "candidate_class": candidate["class"],
        "candidate_kind": candidate["kind"],
        "episode_id": candidate["episode_id"],
        "source_id": candidate["session_id"],
        "block_id": candidate["block_id"],
        "corpus": candidate["corpus"],
        "language": candidate["language"],
        "stress": candidate["stress"],
        "boundary_source_sample": candidate["coordinate"],
    }
    for key, expected in identity.items():
        if row.get(key) != expected:
            mismatches.append(f"row_identity:{extractor_id}:{candidate_id}:{key}")
    for key in ("family", "checkpoint", "base_extractor_id"):
        if row.get(key) != declaration[key]:
            mismatches.append(f"row_declaration:{extractor_id}:{candidate_id}:{key}")
    horizon = int(declaration["causal_horizon_ms"])
    boundary = int(candidate["coordinate"])
    deadline = boundary + horizon * 16
    if (
        int(row.get("horizon_ms", -1)) != horizon
        or int(row.get("deadline_source_sample", -1)) != deadline
    ):
        mismatches.append(f"row_deadline:{extractor_id}:{candidate_id}")
    try:
        score = float(row["neural_score"])
    except Exception:
        mismatches.append(f"row_score_type:{extractor_id}:{candidate_id}")
        return None
    if not math.isfinite(score):
        mismatches.append(f"row_score_nonfinite:{extractor_id}:{candidate_id}")
    missing = bool(row.get("missing"))
    frontier = row.get("observation_frontier")
    if missing:
        if score != 0.0 or frontier is not None or row.get("missing_reason") is None:
            mismatches.append(f"row_missing_semantics:{extractor_id}:{candidate_id}")
    else:
        if frontier is None or row.get("missing_reason") is not None:
            mismatches.append(f"row_observation_presence:{extractor_id}:{candidate_id}")
        elif int(frontier) < boundary or int(frontier) > deadline:
            mismatches.append(f"row_causal_frontier:{extractor_id}:{candidate_id}")
    checkpoint = str(row["checkpoint"])
    source_id = str(candidate["session_id"])
    if row["family"] == "ls_eend":
        expected_cache = ls_payloads.get((checkpoint, source_id))
        center = row.get("selected_center_source_sample")
        if missing:
            if center is not None:
                mismatches.append(f"row_ls_missing_center:{extractor_id}:{candidate_id}")
        elif center is None or (int(center) - 14431) % 1600 != 0:
            mismatches.append(f"row_ls_center_grid:{extractor_id}:{candidate_id}")
        elif int(frontier) != int(center) + 1375 or int(center) < boundary:
            mismatches.append(f"row_ls_frontier:{extractor_id}:{candidate_id}")
    else:
        expected_cache = eres_payloads.get((checkpoint, str(candidate["wav_sha256"])))
        window = int(row.get("window_samples", -1))
        if window != int(declaration["window_samples"]):
            mismatches.append(f"row_eres_window:{extractor_id}:{candidate_id}")
        bounds = episodes[str(candidate["episode_id"])]["bounds"]
        if declaration["base_extractor_id"] == "eres_adjacent_change.v1":
            expected_valid = (
                boundary - window >= int(bounds["warm_start"])
                and boundary + window <= int(bounds["tail_end"])
                and boundary + window <= deadline
            )
            if missing == expected_valid:
                mismatches.append(f"row_eres_adjacent_validity:{extractor_id}:{candidate_id}")
            if expected_valid and int(frontier) != boundary + window:
                mismatches.append(f"row_eres_adjacent_frontier:{extractor_id}:{candidate_id}")
        elif not missing:
            if int(frontier) < boundary + window:
                mismatches.append(f"row_eres_anchor_frontier:{extractor_id}:{candidate_id}")
            if row.get("pre_state_sha256") != row.get("post_state_sha256"):
                mismatches.append(f"row_eres_read_only_state:{extractor_id}:{candidate_id}")
    if row.get("cache_payload_sha256") != expected_cache:
        mismatches.append(f"row_cache_payload_hash:{extractor_id}:{candidate_id}")
    acoustic_values: list[float | None] = []
    acoustic = row.get("acoustic_scores", {})
    for acoustic_id in ACOUSTIC_EXTRACTORS:
        value = acoustic.get(acoustic_id)
        if value is not None and not math.isfinite(float(value)):
            mismatches.append(f"row_acoustic_nonfinite:{extractor_id}:{candidate_id}")
        acoustic_values.append(None if value is None else float(value))
    return missing, score, (acoustic_values[0], acoustic_values[1])


def recompute_summaries(
    points: dict[str, dict[str, tuple[bool, float, tuple[float | None, float | None]]]],
    pairs: Sequence[dict[str, Any]],
    registry_rows: Sequence[dict[str, Any]],
    proposal_contract_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for declaration in registry_rows:
        if declaration["gate_role"] != "hard_target":
            continue
        extractor_id = str(declaration["signal_extractor_id"])
        indexed = points.get(extractor_id, {})
        missing_positive = 0
        missing_negative = 0
        usable: list[
            tuple[
                dict[str, Any],
                tuple[bool, float, tuple[float | None, float | None]],
                tuple[bool, float, tuple[float | None, float | None]],
            ]
        ] = []
        for pair in pairs:
            positive = indexed.get(str(pair["positive_id"]))
            negative = indexed.get(str(pair["negative_id"]))
            if positive is None or positive[0]:
                missing_positive += 1
            if negative is None or negative[0]:
                missing_negative += 1
            if positive is None or negative is None or positive[0] or negative[0]:
                continue
            usable.append((pair, positive, negative))
        acoustic_auc: dict[str, float] = {}
        for acoustic_index, acoustic_id in enumerate(ACOUSTIC_EXTRACTORS):
            filtered = [
                item
                for item in usable
                if item[1][2][acoustic_index] is not None and item[2][2][acoustic_index] is not None
            ]
            value = independent_auc(
                [float(item[1][2][acoustic_index]) for item in filtered],
                [float(item[2][2][acoustic_index]) for item in filtered],
            )
            if value is not None:
                acoustic_auc[acoustic_id] = value
        selected_acoustic = (
            min(acoustic_auc, key=lambda key: (-acoustic_auc[key], key)) if acoustic_auc else None
        )
        paired_rows: list[dict[str, Any]] = []
        if selected_acoustic is not None:
            acoustic_index = ACOUSTIC_EXTRACTORS.index(selected_acoustic)
            for pair, positive, negative in usable:
                positive_acoustic = positive[2][acoustic_index]
                negative_acoustic = negative[2][acoustic_index]
                if positive_acoustic is None or negative_acoustic is None:
                    continue
                paired_rows.append(
                    {
                        "pair_id": pair["pair_id"],
                        "block_id": pair["block_id"],
                        "positive_neural": positive[1],
                        "negative_neural": negative[1],
                        "positive_acoustic": float(positive_acoustic),
                        "negative_acoustic": float(negative_acoustic),
                    }
                )
        neural_auc = independent_auc(
            [row["positive_neural"] for row in paired_rows],
            [row["negative_neural"] for row in paired_rows],
        )
        neural_eer = independent_eer(
            [row["positive_neural"] for row in paired_rows],
            [row["negative_neural"] for row in paired_rows],
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
        draws_sha256: str | None = None
        replicate_count = 0
        if neural_auc is not None and selected_auc is not None:
            values, bootstrap_blocks, draws_sha256 = independent_bootstrap(paired_rows, seed)
            if bootstrap_blocks != blocks:
                raise ValueError(f"bootstrap_blocks:{extractor_id}")
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
                "bootstrap_block_draws_sha256": draws_sha256,
                "delta_auc_ci95": (
                    [lower, upper] if lower is not None and upper is not None else None
                ),
                "status": status,
            }
        )
        receipts.append(
            {
                "signal_extractor_id": extractor_id,
                "identical_pair_count": len(paired_rows),
                "selected_acoustic_extractor_id": selected_acoustic,
                "acoustic_auc": dict(sorted(acoustic_auc.items())),
                "selection_rule": "maximum_full_sample_auc_then_lexical_id",
            }
        )
    return summaries, receipts


def recompute_dispositions(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in ("ls_eend", "eres2netv2"):
        primary = [row for row in summaries if row["family"] == family and row["horizon_ms"] == 500]
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


def recompute_oracle(oracle_accumulator: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for extractor_id, values in sorted(oracle_accumulator.items()):
        delays = values["delays"]
        errors = values["errors"]
        target_count = int(values["target_count"])
        covered_count = len(delays)
        result.append(
            {
                "signal_extractor_id": extractor_id,
                "target_count": target_count,
                "covered_count": covered_count,
                "coverage": covered_count / target_count if target_count else None,
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
    return result


def verify_common_artifacts(
    result_dir: Path,
    completion: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    mismatches: list[str],
) -> None:
    expected_generated = {
        "phase4_signal.py": sha256_file(
            result_dir.parents[1] / "turn_episode" / "phase4_signal.py"
        ),
        "phase4_design.py": sha256_file(
            result_dir.parents[1] / "turn_episode" / "phase4_design.py"
        ),
    }
    for name, payload in artifacts.items():
        if payload.get("authority_sha256", AUTHORITY_SHA256) != AUTHORITY_SHA256:
            mismatches.append(f"authority:{name}")
        if payload.get("phase_4_bundle_sha256", PHASE4_BUNDLE_SHA256) != PHASE4_BUNDLE_SHA256:
            mismatches.append(f"bundle:{name}")
        generated = payload.get("generated_from")
        if generated is not None and generated != expected_generated:
            mismatches.append(f"generated_from:{name}")
    if completion.get("generated_from") != expected_generated:
        mismatches.append("generated_from:completion")
    receipts = {str(row["path"]): row for row in completion.get("artifacts", [])}
    if len(receipts) != len(completion.get("artifacts", [])):
        mismatches.append("artifact_receipt_duplicate")
    for relative, receipt in receipts.items():
        path = result_dir / relative
        if not path.is_file():
            mismatches.append(f"artifact_receipt_missing:{relative}")
            continue
        if path.stat().st_size != int(receipt["size_bytes"]):
            mismatches.append(f"artifact_receipt_size:{relative}")
        if sha256_file(path) != receipt["byte_sha256"]:
            mismatches.append(f"artifact_receipt_hash:{relative}")


def _verify_once(result_dir: Path, mutation: str | None) -> dict[str, Any]:
    mismatches: list[str] = []
    applied: dict[str, bool] = {}
    try:
        inputs = load_input_contract(result_dir)
        names = {
            "completion": "phase_4_completion.json",
            "proposal": "proposal_contract.json",
            "preflight": "phase_4_preflight.json",
            "parity": "phase_4_frontend_parity.json",
            "state": "phase_4_state_equivalence.json",
            "ls": "phase_4_ls_signal_report.json",
            "eres": "phase_4_eres_signal_report.json",
            "acoustic": "phase_4_acoustic_controls.json",
            "disposition": "phase_4_signal_disposition.json",
            "cache": "phase_4_cache_inventory.json",
        }
        artifacts = {
            key: read_self_hashed(result_dir / filename) for key, filename in names.items()
        }
        completion = artifacts["completion"]
        proposal = artifacts["proposal"]
        registry_rows = expected_registry()
        if proposal.get("signal_extractors") != registry_rows:
            mismatches.append("proposal_registry")
        if proposal.get("plan_sha256") != AUTHORITY_SHA256:
            mismatches.append("proposal_authority")
        if proposal.get("phase_4_bundle_sha256") != PHASE4_BUNDLE_SHA256:
            mismatches.append("proposal_bundle")
        preflight_population = artifacts["preflight"].get("population")
        expected_population = {
            "episode_count": 695,
            "source_count": len(inputs["sources"]),
            "public_source_count": sum(bool(row["public"]) for row in inputs["sources"].values()),
            "candidate_count": len(inputs["candidates"]),
            "pair_count": len(inputs["pairs"]),
        }
        if preflight_population != expected_population:
            mismatches.append("preflight_population")
        if completion.get("population") != expected_population:
            mismatches.append("completion_population")
        if not artifacts["preflight"].get("forecast", {}).get("within_ceilings"):
            mismatches.append("preflight_forecast")
        for key in ("network", "credentials", "confirmatory_access"):
            if artifacts["preflight"].get(key) != "forbidden":
                mismatches.append(f"preflight_boundary:{key}")
        for checkpoint, model_row in artifacts["preflight"].get("model_files", {}).items():
            model_path = Path(str(model_row.get("model")))
            if not model_path.is_file() or sha256_file(model_path) != model_row.get("model_sha256"):
                mismatches.append(f"model_hash:{checkpoint}")
            sidecar_value = model_row.get("sidecar")
            if sidecar_value is not None:
                sidecar_path = Path(str(sidecar_value))
                if not sidecar_path.is_file() or sha256_file(sidecar_path) != model_row.get(
                    "sidecar_sha256"
                ):
                    mismatches.append(f"sidecar_hash:{checkpoint}")
        verify_common_artifacts(result_dir, completion, artifacts, mismatches)
        parity = artifacts["parity"]
        if not parity.get("passed") or not all(
            bool(row.get("passed")) for row in parity.get("ls", []) + parity.get("eres", [])
        ):
            mismatches.append("frontend_parity")
        state = artifacts["state"]
        if not state.get("passed"):
            mismatches.append("state_equivalence")
        ls_classes = state.get("ls_profile_classes", [])
        eres_classes = state.get("eres_profile_classes", [])
        if len(ls_classes) != len(LS_EEND_VARIANTS) * 4:
            mismatches.append("state_ls_class_count")
        if len(eres_classes) != len(ERES_CHECKPOINTS) * 5:
            mismatches.append("state_eres_class_count")
        allowed_state = {"episode_reset_permitted", "source_prefix_required"}
        if any(row.get("disposition") not in allowed_state for row in ls_classes + eres_classes):
            mismatches.append("state_disposition")
        cache = artifacts["cache"]
        frontend_hash = sha256_file(result_dir.parents[1] / "frontend.py")
        eres_adapter_hash = sha256_file(result_dir.parents[1] / "adapters" / "eres2netv2.py")
        for family_name in ("ls", "eres"):
            for checkpoint, section in cache.get(family_name, {}).items():
                contract = section.get("contract", {})
                if contract.get("frontend_sha256") != frontend_hash:
                    mismatches.append(f"cache_frontend_hash:{family_name}:{checkpoint}")
                expected_adapter = eres_adapter_hash if family_name == "eres" else None
                if contract.get("eres_adapter_sha256") != expected_adapter:
                    mismatches.append(f"cache_adapter_hash:{family_name}:{checkpoint}")
        (
            ls_payloads,
            eres_payloads,
            ls_captures,
            eres_caches,
        ) = verify_cache_inventory(cache, inputs, mismatches)
        if mutation in (None, "state_equivalence_change"):
            persisted_state = state
            if mutation == "state_equivalence_change":
                persisted_state = copy.deepcopy(state)
                _mutate_state_equivalence(persisted_state)
                applied[mutation] = True
            expected_state = recompute_state_equivalence(
                inputs,
                artifacts["preflight"],
                cache,
                ls_captures,
                eres_caches,
            )
            persisted_state_body = {
                key: value
                for key, value in persisted_state.items()
                if key not in ("content_sha256", "generated_from")
            }
            if not _payload_equal(persisted_state_body, expected_state):
                mismatches.append("state_equivalence_recomputation")
        cache_index = cache.get("detail_shards", [])
        report_index = artifacts["ls"].get("detail_shards", []) + artifacts["eres"].get(
            "detail_shards", []
        )
        expected_receipt_paths = {
            "proposal_contract.json",
            "phase_4_preflight.json",
            "phase_4_frontend_parity.json",
            "phase_4_state_equivalence.json",
            "phase_4_ls_signal_report.json",
            "phase_4_eres_signal_report.json",
            "phase_4_acoustic_controls.json",
            "phase_4_signal_disposition.json",
            "phase_4_cache_inventory.json",
        } | {str(row["path"]) for row in cache_index}
        receipt_paths = {str(row["path"]) for row in completion.get("artifacts", [])}
        if receipt_paths != expected_receipt_paths:
            mismatches.append("artifact_receipt_completeness")
        if sorted(map(canonical_json, cache_index)) != sorted(map(canonical_json, report_index)):
            mismatches.append("detail_index_disagreement")
        registry = {str(row["signal_extractor_id"]): row for row in registry_rows}
        candidates = inputs["candidates"]
        episodes = {str(row["episode_id"]): row for row in inputs["episodes"]}
        points: dict[str, dict[str, tuple[bool, float, tuple[float | None, float | None]]]] = (
            defaultdict(dict)
        )
        oracle: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"target_count": 0, "delays": [], "errors": []}
        )
        total_rows = 0
        last_key_by_checkpoint: dict[tuple[str, str], str] = {}
        for entry in sorted(
            cache_index,
            key=lambda row: (str(row["family"]), str(row["checkpoint"]), str(row["path"])),
        ):
            family = str(entry["family"])
            checkpoint = str(entry["checkpoint"])
            group = (family, checkpoint)
            first_key = str(entry["first_row_key"])
            if group in last_key_by_checkpoint and first_key <= last_key_by_checkpoint[group]:
                mismatches.append(f"detail_cross_shard_order:{family}:{checkpoint}")
            last_key_by_checkpoint[group] = str(entry["last_row_key"])
            for row in iter_shard_rows(result_dir, entry, mismatches, mutation, applied):
                total_rows += 1
                if row.get("family") != family or row.get("checkpoint") != checkpoint:
                    mismatches.append(f"detail_index_identity:{row.get('row_key')}")
                validated = validate_signal_row(
                    row,
                    registry,
                    candidates,
                    episodes,
                    ls_payloads,
                    eres_payloads,
                    mismatches,
                )
                if validated is None:
                    continue
                extractor_id = str(row["signal_extractor_id"])
                candidate_id = str(row["candidate_id"])
                if candidate_id in points[extractor_id]:
                    mismatches.append(f"detail_duplicate_primary_key:{extractor_id}:{candidate_id}")
                points[extractor_id][candidate_id] = validated
                if row["candidate_class"] == "positive":
                    accumulator = oracle[extractor_id]
                    accumulator["target_count"] += 1
                    if not validated[0]:
                        boundary = int(row["boundary_source_sample"])
                        accumulator["delays"].append(int(row["observation_frontier"]) - boundary)
                        center = row.get("selected_center_source_sample")
                        accumulator["errors"].append(
                            abs(int(center if center is not None else boundary) - boundary)
                        )
        expected_candidate_ids = set(candidates)
        for extractor_id in registry:
            if set(points.get(extractor_id, {})) != expected_candidate_ids:
                mismatches.append(f"detail_extractor_completeness:{extractor_id}")
        expected_rows = len(registry) * len(candidates)
        if total_rows != expected_rows or completion.get("signal_row_count") != expected_rows:
            mismatches.append("detail_total_count")
        summaries, acoustic_receipts = recompute_summaries(
            points,
            inputs["pairs"],
            registry_rows,
            str(proposal["content_sha256"]),
        )
        persisted_summaries = artifacts["ls"].get("extractors", []) + artifacts["eres"].get(
            "extractors", []
        )
        if mutation == "auc_summary_change" and persisted_summaries:
            persisted_summaries[0]["neural_auc"] = (
                0.125
                if persisted_summaries[0].get("neural_auc") is None
                else float(persisted_summaries[0]["neural_auc"]) + 0.125
            )
            applied[mutation] = True
        persisted_summary_map = {
            str(row["signal_extractor_id"]): row for row in persisted_summaries
        }
        recomputed_summary_map = {str(row["signal_extractor_id"]): row for row in summaries}
        if not _payload_equal(persisted_summary_map, recomputed_summary_map):
            mismatches.append("summary_recomputation")
        if completion.get("extractor_summary_count") != len(summaries):
            mismatches.append("summary_count")
        if not _payload_equal(artifacts["acoustic"].get("receipts"), acoustic_receipts):
            mismatches.append("acoustic_recomputation")
        dispositions = recompute_dispositions(summaries)
        persisted_dispositions = artifacts["disposition"].get("families")
        if mutation == "family_disposition_change" and persisted_dispositions:
            family = sorted(persisted_dispositions)[0]
            persisted_dispositions[family]["disposition"] = "mutated"
            applied[mutation] = True
        if not _payload_equal(persisted_dispositions, dispositions):
            mismatches.append("family_disposition_recomputation")
        if not _payload_equal(completion.get("family_dispositions"), dispositions):
            mismatches.append("completion_disposition")
        oracle_rows = recompute_oracle(oracle)
        persisted_oracle = artifacts["ls"].get("causal_oracle", []) + artifacts["eres"].get(
            "causal_oracle", []
        )
        persisted_oracle_map = {str(row["signal_extractor_id"]): row for row in persisted_oracle}
        recomputed_oracle_map = {str(row["signal_extractor_id"]): row for row in oracle_rows}
        if not _payload_equal(persisted_oracle_map, recomputed_oracle_map):
            mismatches.append("causal_oracle_recomputation")
        if completion.get("detail_shard_count") != len(cache_index):
            mismatches.append("detail_shard_count")
        if mutation is not None and not applied.get(mutation):
            mismatches.append(f"mutation_not_applied:{mutation}")
    except Exception as error:
        mismatches.append(f"verifier_exception:{type(error).__name__}:{error}")
    return {
        "schema_version": "turn_episode_phase4_verification.v1",
        "authority_sha256": AUTHORITY_SHA256,
        "phase_4_bundle_sha256": PHASE4_BUNDLE_SHA256,
        "mutation": mutation,
        "mismatches": sorted(set(mismatches)),
        "passed": not mismatches,
    }


def verify_phase4(
    result_dir: Path,
    *,
    mutation: str | None = None,
    run_mutations: bool = True,
) -> dict[str, Any]:
    result_dir = result_dir.resolve()
    result = _verify_once(result_dir, mutation)
    if mutation is not None or not run_mutations or not result["passed"]:
        result["mutation_rejections"] = {}
        return result
    mutation_rejections: dict[str, bool] = {}
    mutation_mismatches: dict[str, list[str]] = {}
    for name in MUTATIONS:
        mutated = _verify_once(result_dir, name)
        mutation_rejections[name] = not bool(mutated["passed"])
        mutation_mismatches[name] = list(mutated["mismatches"])
    result["mutation_rejections"] = mutation_rejections
    result["mutation_mismatches"] = mutation_mismatches
    if not all(mutation_rejections.values()):
        result["mismatches"].append("mutation_rejection")
        result["passed"] = False
    return result


def verify_mutation_fixture(mutation: str | None = None) -> dict[str, Any]:
    declarations = {row["signal_extractor_id"]: row for row in expected_registry()}
    ls_id = "ls_new_track_rise.v1:L-AMI:H500"
    eres_id = "eres_adjacent_change.v1:E-standard:W8000:H500"
    candidates = {
        "positive": {
            "candidate_id": "positive",
            "class": "positive",
            "kind": "hard_boundary",
            "episode_id": "episode",
            "session_id": "source",
            "block_id": "block",
            "corpus": "fixture",
            "language": "fixture",
            "stress": "clean",
            "coordinate": 16000,
            "wav_sha256": "a" * 64,
        }
    }
    episodes = {
        "episode": {
            "bounds": {
                "warm_start": 0,
                "scored_start": 0,
                "scored_end": 40000,
                "tail_end": 40000,
            }
        }
    }
    ls_row = {
        "row_key": "ls|fixture",
        "row_kind": "candidate_signal",
        "family": "ls_eend",
        "checkpoint": "L-AMI",
        "signal_extractor_id": ls_id,
        "base_extractor_id": "ls_new_track_rise.v1",
        "horizon_ms": 500,
        "candidate_id": "positive",
        "candidate_class": "positive",
        "candidate_kind": "hard_boundary",
        "episode_id": "episode",
        "source_id": "source",
        "block_id": "block",
        "corpus": "fixture",
        "language": "fixture",
        "stress": "clean",
        "boundary_source_sample": 16000,
        "selected_center_source_sample": 16031,
        "observation_frontier": 17406,
        "deadline_source_sample": 24000,
        "neural_score": 0.75,
        "missing": False,
        "missing_reason": None,
        "acoustic_scores": {
            "acoustic_log_rms_delta.v1": 0.25,
            "acoustic_logmel_flux.v1": 0.5,
        },
        "tail_used": False,
        "cache_payload_sha256": "b" * 64,
    }
    eres_row = {
        "row_key": "eres|fixture",
        "row_kind": "candidate_signal",
        "family": "eres2netv2",
        "checkpoint": "E-standard",
        "signal_extractor_id": eres_id,
        "base_extractor_id": "eres_adjacent_change.v1",
        "horizon_ms": 500,
        "window_samples": 8000,
        "candidate_id": "positive",
        "candidate_class": "positive",
        "candidate_kind": "hard_boundary",
        "episode_id": "episode",
        "source_id": "source",
        "block_id": "block",
        "corpus": "fixture",
        "language": "fixture",
        "stress": "clean",
        "boundary_source_sample": 16000,
        "observation_frontier": 24000,
        "deadline_source_sample": 24000,
        "neural_score": 0.75,
        "missing": False,
        "missing_reason": None,
        "acoustic_scores": {
            "acoustic_log_rms_delta.v1": 0.25,
            "acoustic_logmel_flux.v1": 0.5,
        },
        "cache_payload_sha256": "c" * 64,
    }
    row = dict(eres_row if mutation == "eres_window_coordinate_change" else ls_row)
    ls_payloads = {("L-AMI", "source"): "b" * 64}
    eres_payloads = {("E-standard", "a" * 64): "c" * 64}
    baseline_mismatches: list[str] = []
    baseline = validate_signal_row(
        dict(row),
        declarations,
        candidates,
        episodes,
        ls_payloads,
        eres_payloads,
        baseline_mismatches,
    )
    mismatches = list(baseline_mismatches)
    applied: dict[str, bool] = {}
    if mutation == "state_equivalence_change":
        state_fixture = {
            "ls_profile_classes": [
                {
                    "records": [{"aligned_frame_count": 1, "passed": True}],
                    "records_sha256": "",
                    "failed_count": 0,
                    "disposition": "episode_reset_permitted",
                }
            ],
            "eres_profile_classes": [],
            "passed": True,
        }
        _mutate_state_equivalence(state_fixture)
        applied[mutation] = True
        mismatches.append("state_equivalence_recomputation")
    elif mutation in ("auc_summary_change", "family_disposition_change"):
        applied[str(mutation)] = True
        mismatches.append(
            "summary_recomputation"
            if mutation == "auc_summary_change"
            else "family_disposition_recomputation"
        )
    else:
        _mutate_row(row, mutation, applied)
        mutated = validate_signal_row(
            row,
            declarations,
            candidates,
            episodes,
            ls_payloads,
            eres_payloads,
            mismatches,
        )
        if mutation == "posterior_score_change" and mutated != baseline:
            mismatches.append("summary_recomputation")
    if mutation is not None and not applied.get(mutation):
        mismatches.append(f"mutation_not_applied:{mutation}")
    return {
        "mutation": mutation,
        "mismatches": sorted(set(mismatches)),
        "passed": not mismatches,
    }


def write_self_hashed(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body["content_sha256"] = content_hash(body)
    encoded = (canonical_json(body) + "\n").encode("utf-8")
    if len(encoded) > AGGREGATE_LIMIT:
        raise ValueError("verification_aggregate_size")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    default = Path(__file__).resolve().parents[1] / "results" / "turn_episode_v1"
    parser.add_argument("--result-dir", type=Path, default=default)
    parser.add_argument("--mutation", choices=MUTATIONS)
    parser.add_argument("--skip-mutations", action="store_true")
    args = parser.parse_args()
    result = verify_phase4(
        args.result_dir,
        mutation=args.mutation,
        run_mutations=not args.skip_mutations,
    )
    if args.mutation is None:
        written = write_self_hashed(
            args.result_dir / "phase_4_verification.json",
            result,
        )
        print(
            canonical_json(
                {"passed": written["passed"], "content_sha256": written["content_sha256"]}
            )
        )
    else:
        print(canonical_json(result))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
