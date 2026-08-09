from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from experiments.speaker_turn_boundary.corpus.phase2_schemas import Phase2Manifest

from .build_episodes import load_session_data, verify_targets_match_phase1

SAMPLE_RATE = 16000
AUTHORITY_SHA256 = "ecd16c765072504f9970cb17fd5cba4f9967715954adead6ebc386fffb8d4f8c"
MANIFEST_CONTENT_SHA256 = "deb1713cd581c93cc13c47103643041c7551993985379869cfa0ca9a407dff68"
MANIFEST_BYTE_SHA256 = "a5df5e56c2917ba174fca7892fbaa092f2b57705cad7a77bf9a347ba94cddfee"
GROUP_GRAPH_SHA256 = "7ebf4dffa0af180910007a318d0e3d1e77f7f048dbae852199ddd45f74cce7eb"
ADJACENT_WINDOWS = (8000, 12000, 16000, 24000, 32000)
ANCHOR_WINDOWS = (8000, 12000, 16000, 24000)
STEPS = (1600, 4000)
LONG_STEPS = (1600, 4000, 8000)
PRIMARY_WINDOW = 8000
STABLE_EXCLUSION = 16000
LS_RTF_FORECAST = 0.05
ERES_SECONDS_PER_EMBEDDING = 0.037
ERES_PARALLEL_WORKERS = 10
ERES_THROUGHPUT_MARGIN = 0.75
LS_CACHE_BYTES_PER_AUDIO_SECOND = 4096
ERES_CACHE_BYTES_PER_EMBEDDING = 2048
ACOUSTIC_CACHE_BYTES_PER_WINDOW = 1024
LS_ACOUSTIC_SUPPORT_BY_HORIZON = {250: 4000, 500: 8000, 1000: 16000}
MATCH_MAX_PAIRS = 450
MATCH_MAX_FEATURE_DISTANCE = 1_000_000_000
MATCH_MAX_TIE = (1 << 256) - 1
MATCH_GAP_WEIGHT = MATCH_MAX_PAIRS * MATCH_MAX_TIE + 1
MATCH_DURATION_WEIGHT = (
    MATCH_MAX_PAIRS * MATCH_MAX_FEATURE_DISTANCE * MATCH_GAP_WEIGHT
    + MATCH_MAX_PAIRS * MATCH_MAX_TIE
    + 1
)
MATCH_STRESS_WEIGHT = (
    MATCH_MAX_PAIRS * MATCH_MAX_FEATURE_DISTANCE * MATCH_DURATION_WEIGHT
    + MATCH_MAX_PAIRS * MATCH_MAX_FEATURE_DISTANCE * MATCH_GAP_WEIGHT
    + MATCH_MAX_PAIRS * MATCH_MAX_TIE
    + 1
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ceil_grid(value: int, step: int) -> int:
    return ((value + step - 1) // step) * step


def corpus_for(session_id: str) -> str:
    if session_id.startswith("ami_"):
        return "ami"
    if session_id.startswith("alimeeting_"):
        return "alimeeting"
    return "librispeech_synthetic"


def language_for(session_id: str) -> str:
    if session_id.startswith("ami_"):
        return "english"
    if session_id.startswith("alimeeting_"):
        return "chinese"
    return "english"


def synthetic_manifest_name(session_id: str) -> str | None:
    if ":" not in session_id:
        return None
    name = session_id.split(":", 1)[0]
    if name in ("ls_dev", "ls_held_out_clean", "ls_held_out_other"):
        return name
    return None


def synthetic_case_id(session_id: str) -> str | None:
    name = synthetic_manifest_name(session_id)
    return session_id.split(":", 1)[1] if name is not None else None


def stress_class(case: dict[str, Any] | None) -> str:
    if case is None:
        return "natural"
    transforms = case.get("transforms") or []
    if transforms:
        return "+".join(sorted(str(item.get("kind") or item) for item in transforms))
    case_id = str(case["case_id"])
    marker = "_stress_"
    if marker in case_id:
        return case_id.split(marker, 1)[1].rsplit("_", 1)[0]
    if str(case.get("kind")) == "gain_variation":
        return "gain"
    return "clean"


def case_duration_ms(case: dict[str, Any] | None, episode: dict[str, Any]) -> int:
    if case is not None and "duration_target_s" in (case.get("condition") or {}):
        return int(round(float(case["condition"]["duration_target_s"]) * 1000))
    bounds = episode["bounds"]
    return int(round((int(bounds["scored_end"]) - int(bounds["scored_start"])) / 16))


def case_gap_ms(case: dict[str, Any] | None, reference: dict[str, Any] | None) -> int:
    if case is not None:
        value = (case.get("condition") or {}).get("gap_ms")
        return int(value) if value is not None else 0
    if reference is not None:
        interval = reference.get("acceptable_interval") or [0, 0]
        onset = reference.get("evidence_onset_sample")
        if onset is not None:
            return max(0, int(round((int(onset) - int(interval[0])) / 16)))
    return 0


def component_map(inventory: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for component, members in inventory["group_graph"]["component_sessions"].items():
        for member in members:
            result[str(member)] = str(component)
    return result


def load_synthetic_cases(manifests_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for name in ("ls_dev", "ls_held_out_clean", "ls_held_out_other"):
        raw = json.loads((manifests_dir / f"{name}.json").read_text(encoding="utf-8"))
        for case in raw["cases"]:
            result[(name, str(case["case_id"]))] = case
    return result


def load_public_regions(
    inventory: dict[str, Any],
    details: dict[str, dict[str, Any]],
    sessions: Iterable[str],
    manifests_dir: Path,
) -> dict[str, list[Any]]:
    opened = sorted(str(item) for item in inventory["completed_materialized_sessions"])
    ranks: dict[str, dict[str, int]] = {}
    for corpus in ("ami", "alimeeting"):
        ids = sorted(item for item in opened if str(details[item]["corpus"]) == corpus)
        ranks[corpus] = {session_id: rank for rank, session_id in enumerate(ids)}
    pilot_cases: dict[str, list[Any]] = {}
    for name in ("ami_dev_pilot.json", "ami_held_out_pilot.json", "alimeeting_eval_pilot.json"):
        manifest = Phase2Manifest.load(manifests_dir / name)
        for case in manifest.cases:
            pilot_cases.setdefault(str(case.case_id), []).append(case)
    corpus_root = Path(str(inventory["corpus_root"]))
    result: dict[str, list[Any]] = {}
    for session_id in sorted(set(sessions)):
        session = load_session_data(
            session_id,
            details[session_id],
            corpus_root,
            manifests_dir,
            pilot_cases,
            ranks,
        )
        verify_targets_match_phase1(session, details[session_id])
        result[session_id] = list(session.regions)
    return result


def reference_coordinates(episode: dict[str, Any]) -> list[int]:
    values: set[int] = set()
    for reference in episode["references"]:
        for key in ("evidence_onset_sample", "target_sample"):
            value = reference.get(key)
            if value is not None:
                values.add(int(value))
        for value in reference.get("acceptable_interval") or []:
            values.add(int(value))
    return sorted(values)


def stable_coordinate(episode: dict[str, Any], regions: list[Any]) -> int | None:
    bounds = episode["bounds"]
    scored_start = int(bounds["scored_start"])
    scored_end = int(bounds["scored_end"])
    excluded = reference_coordinates(episode)
    candidates: list[tuple[int, int]] = []
    for region in regions:
        if isinstance(region, dict):
            speakers = list(region["speakers"])
            ambiguous = bool(region["ambiguous"])
            region_start = int(region["start_sample"])
            region_end = int(region["end_sample"])
        else:
            speakers = list(region.speakers)
            ambiguous = bool(region.ambiguous)
            region_start = int(region.start_sample)
            region_end = int(region.end_sample)
        if len(speakers) != 1 or ambiguous:
            continue
        lo = max(scored_start, region_start) + PRIMARY_WINDOW
        hi = min(scored_end, region_end) - PRIMARY_WINDOW
        first = ceil_grid(lo, 1600)
        for coordinate in range(first, hi + 1, 1600):
            distance = min((abs(coordinate - item) for item in excluded), default=1 << 60)
            if distance >= STABLE_EXCLUSION:
                candidates.append((distance, coordinate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def block_id(episode: dict[str, Any], components: dict[str, str], manifest_name: str | None) -> str:
    if manifest_name is not None:
        return f"synthetic:{manifest_name}"
    return components[str(episode["session_id"])]


def candidate_row(
    *,
    candidate_id: str,
    candidate_class: str,
    kind: str,
    coordinate: int,
    episode: dict[str, Any],
    case: dict[str, Any] | None,
    reference: dict[str, Any] | None,
    components: dict[str, str],
) -> dict[str, Any]:
    manifest_name = synthetic_manifest_name(str(episode["session_id"]))
    return {
        "candidate_id": candidate_id,
        "class": candidate_class,
        "kind": kind,
        "episode_id": episode["episode_id"],
        "session_id": episode["session_id"],
        "wav_sha256": episode["wav_sha256"],
        "coordinate": coordinate,
        "corpus": corpus_for(str(episode["session_id"])),
        "language": language_for(str(episode["session_id"])),
        "block_id": block_id(episode, components, manifest_name),
        "synthetic_manifest": manifest_name,
        "duration_ms": case_duration_ms(case, episode),
        "gap_ms": case_gap_ms(case, reference),
        "stress": stress_class(case),
    }


def build_candidates(
    episodes: list[dict[str, Any]],
    cases: dict[tuple[str, str], dict[str, Any]],
    components: dict[str, str],
    public_regions: dict[str, list[Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    for episode in episodes:
        session_id = str(episode["session_id"])
        manifest_name = synthetic_manifest_name(session_id)
        case_id = synthetic_case_id(session_id)
        case = cases.get((manifest_name, case_id)) if manifest_name and case_id else None
        for reference in episode["references"]:
            if (
                episode["status"] == "scorable"
                and bool(reference.get("scorable"))
                and bool(reference.get("primary_case"))
                and reference["action_kind"] == "hard_boundary"
            ):
                positives.append(
                    candidate_row(
                        candidate_id=f"positive:{reference['reference_id']}",
                        candidate_class="positive",
                        kind="hard_boundary",
                        coordinate=int(reference["evidence_onset_sample"]),
                        episode=episode,
                        case=case,
                        reference=reference,
                        components=components,
                    )
                )
            if (
                episode["status"] == "scorable"
                and bool(reference.get("scorable"))
                and reference["action_kind"] == "neutral_pause"
            ):
                negatives.append(
                    candidate_row(
                        candidate_id=f"negative:pause:{reference['reference_id']}",
                        candidate_class="negative",
                        kind="neutral_pause",
                        coordinate=int(reference["evidence_onset_sample"]),
                        episode=episode,
                        case=case,
                        reference=reference,
                        components=components,
                    )
                )
        if episode["status"] != "scorable":
            continue
        if case is not None and str(case.get("kind")) in ("same_speaker", "gain_variation"):
            coordinate = int(case["splice"]["b_onset_sample"])
            if not any(
                row["episode_id"] == episode["episode_id"] and row["coordinate"] == coordinate
                for row in negatives
            ):
                negatives.append(
                    candidate_row(
                        candidate_id=f"negative:synthetic:{manifest_name}:{case_id}",
                        candidate_class="negative",
                        kind=str(case["kind"]),
                        coordinate=coordinate,
                        episode=episode,
                        case=case,
                        reference=None,
                        components=components,
                    )
                )
        regions = (
            list(case.get("regions") or []) if case is not None else public_regions[session_id]
        )
        coordinate = stable_coordinate(episode, regions)
        if coordinate is not None and not any(
            row["episode_id"] == episode["episode_id"] and row["coordinate"] == coordinate
            for row in negatives
        ):
            negatives.append(
                candidate_row(
                    candidate_id=f"negative:stable:{episode['episode_id']}:{coordinate}",
                    candidate_class="negative",
                    kind="stable_singleton",
                    coordinate=coordinate,
                    episode=episode,
                    case=case,
                    reference=None,
                    components=components,
                )
            )
    return sorted(positives, key=lambda row: row["candidate_id"]), sorted(
        negatives, key=lambda row: row["candidate_id"]
    )


def pair_cost(positive: dict[str, Any], negative: dict[str, Any]) -> int:
    mismatch = 0 if positive["stress"] == negative["stress"] else 1
    duration = abs(int(positive["duration_ms"]) - int(negative["duration_ms"]))
    gap = abs(int(positive["gap_ms"]) - int(negative["gap_ms"]))
    if duration > MATCH_MAX_FEATURE_DISTANCE or gap > MATCH_MAX_FEATURE_DISTANCE:
        raise RuntimeError("matching feature distance exceeds frozen bound")
    tie = int(
        hashlib.sha256(
            f"{positive['candidate_id']}|{negative['candidate_id']}".encode("utf-8")
        ).hexdigest(),
        16,
    )
    return (
        mismatch * MATCH_STRESS_WEIGHT
        + duration * MATCH_DURATION_WEIGHT
        + gap * MATCH_GAP_WEIGHT
        + tie
    )


def hungarian_assignment(costs: list[list[int]]) -> list[tuple[int, int]]:
    if not costs or not costs[0]:
        return []
    row_count = len(costs)
    column_count = len(costs[0])
    transposed = row_count > column_count
    matrix = (
        [[costs[row][column] for row in range(row_count)] for column in range(column_count)]
        if transposed
        else costs
    )
    n = len(matrix)
    m = len(matrix[0])
    u = [0] * (n + 1)
    v = [0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        min_value: list[int | None] = [None] * (m + 1)
        used = [False] * (m + 1)
        j0 = 0
        while True:
            used[j0] = True
            i0 = p[j0]
            delta: int | None = None
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                current = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                if min_value[j] is None or current < min_value[j]:
                    min_value[j] = current
                    way[j] = j0
                if delta is None or min_value[j] < delta or (min_value[j] == delta and j < j1):
                    delta = min_value[j]
                    j1 = j
            if delta is None:
                raise RuntimeError("hungarian assignment exhausted columns")
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                elif min_value[j] is not None:
                    min_value[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]
    if transposed:
        assignment = [(column, row) for row, column in assignment]
    return sorted(assignment)


def match_pairs(
    positives: list[dict[str, Any]], negatives: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped_positive: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    grouped_negative: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in positives:
        grouped_positive[(row["corpus"], row["language"], row["block_id"])].append(row)
    for row in negatives:
        grouped_negative[(row["corpus"], row["language"], row["block_id"])].append(row)
    pairs: list[dict[str, Any]] = []
    matched_positive: set[str] = set()
    matched_negative: set[str] = set()
    for group in sorted(grouped_positive):
        left = sorted(grouped_positive[group], key=lambda row: row["candidate_id"])
        right = sorted(grouped_negative.get(group) or [], key=lambda row: row["candidate_id"])
        if not left or not right:
            continue
        costs = [[pair_cost(a, b) for b in right] for a in left]
        for i, j in hungarian_assignment(costs):
            positive = left[i]
            negative = right[j]
            matched_positive.add(str(positive["candidate_id"]))
            matched_negative.add(str(negative["candidate_id"]))
            pairs.append(
                {
                    "pair_id": sha256_bytes(
                        f"{positive['candidate_id']}|{negative['candidate_id']}".encode("utf-8")
                    ),
                    "block_id": positive["block_id"],
                    "corpus": positive["corpus"],
                    "language": positive["language"],
                    "positive_id": positive["candidate_id"],
                    "negative_id": negative["candidate_id"],
                    "positive_coordinate": positive["coordinate"],
                    "negative_coordinate": negative["coordinate"],
                    "stress_mismatch": positive["stress"] != negative["stress"],
                    "duration_distance_ms": abs(
                        int(positive["duration_ms"]) - int(negative["duration_ms"])
                    ),
                    "gap_distance_ms": abs(int(positive["gap_ms"]) - int(negative["gap_ms"])),
                }
            )
    return sorted(pairs, key=lambda row: row["pair_id"]), {
        "positive_unmatched": len(positives) - len(matched_positive),
        "negative_unused": len(negatives) - len(matched_negative),
        "groups_without_negative": sum(
            1 for group in grouped_positive if not grouped_negative.get(group)
        ),
    }


def ls_candidate_valid(
    candidate: dict[str, Any], episode: dict[str, Any], horizon_ms: int, support: int
) -> bool:
    boundary = int(candidate["coordinate"])
    bounds = episode["bounds"]
    if boundary - support < int(bounds["warm_start"]):
        return False
    if boundary + support > int(bounds["tail_end"]):
        return False
    frame = max(0, ceil_grid(boundary - 14431, 1600) // 1600)
    center = 1600 * frame + 14431
    available = center + 1375
    deadline = boundary + horizon_ms * 16
    return available <= deadline and available <= int(bounds["tail_end"])


def coordinate_ledger(
    episodes: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    windows: set[tuple[str, int, int]] = set()
    acoustic_windows: set[tuple[str, int, int]] = set()
    profile_counts: Counter[str] = Counter()
    public_source_rows: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        bounds = episode["bounds"]
        warm_start = int(bounds["warm_start"])
        scored_start = int(bounds["scored_start"])
        scored_end = int(bounds["scored_end"])
        tail_end = int(bounds["tail_end"])
        wav = str(episode["wav_sha256"])
        session_id = str(episode["session_id"])
        if synthetic_manifest_name(session_id) is None:
            source = public_source_rows.setdefault(
                wav,
                {
                    "source_id": session_id,
                    "wav_sha256": wav,
                    "maximum_tail_end": tail_end,
                    "episode_count": 0,
                },
            )
            source["maximum_tail_end"] = max(int(source["maximum_tail_end"]), tail_end)
            source["episode_count"] = int(source["episode_count"]) + 1
        for window in ADJACENT_WINDOWS:
            for step in LONG_STEPS if window >= 24000 else STEPS:
                lo = max(scored_start, warm_start + window)
                hi = min(scored_end, tail_end - window)
                first = ceil_grid(lo, step)
                profile = f"adjacent:W{window}:S{step}"
                for boundary in range(first, hi + 1, step):
                    rows.append(
                        {
                            "episode_id": episode["episode_id"],
                            "kind": "adjacent_grid",
                            "profile": profile,
                            "boundary": boundary,
                            "observation_frontier": boundary + window,
                        }
                    )
                    profile_counts[profile] += 1
                    windows.add((wav, boundary - window, boundary))
                    windows.add((wav, boundary, boundary + window))
        for window in ANCHOR_WINDOWS:
            for step in STEPS:
                first_end = ceil_grid(warm_start + window, step)
                profile = f"trailing_probe:W{window}:S{step}"
                for end in range(first_end, tail_end + 1, step):
                    rows.append(
                        {
                            "episode_id": episode["episode_id"],
                            "kind": "trailing_probe_grid",
                            "profile": profile,
                            "boundary": end - window,
                            "observation_frontier": end,
                        }
                    )
                    profile_counts[profile] += 1
                    windows.add((wav, end - window, end))
    source_prefix_counts: Counter[str] = Counter()
    state_snapshot_rows: list[dict[str, Any]] = []
    for wav, source in sorted(public_source_rows.items()):
        maximum_tail = int(source["maximum_tail_end"])
        for window in ANCHOR_WINDOWS:
            for step in STEPS:
                profile = f"source_prefix_probe:W{window}:S{step}"
                for end in range(ceil_grid(window, step), maximum_tail + 1, step):
                    rows.append(
                        {
                            "source_id": source["source_id"],
                            "kind": "source_prefix_probe_grid",
                            "profile": profile,
                            "boundary": end - window,
                            "observation_frontier": end,
                        }
                    )
                    source_prefix_counts[profile] += 1
                    windows.add((wav, end - window, end))
    for episode in episodes:
        session_id = str(episode["session_id"])
        if synthetic_manifest_name(session_id) is not None:
            continue
        warm_start = int(episode["bounds"]["warm_start"])
        for window in ANCHOR_WINDOWS:
            for step in STEPS:
                last_probe_end = (warm_start // step) * step
                for state_mode in (
                    "stable_no_update",
                    "stable_ema",
                    "confirmed_anchor",
                    "prototype_memory_4",
                ):
                    row = {
                        "episode_id": episode["episode_id"],
                        "kind": "source_prefix_state_snapshot",
                        "state_mode": state_mode,
                        "window_samples": window,
                        "step_samples": step,
                        "snapshot_frontier": warm_start,
                        "last_probe_end": last_probe_end if last_probe_end >= window else None,
                    }
                    rows.append(row)
                    state_snapshot_rows.append(row)
    episode_by_id = {str(episode["episode_id"]): episode for episode in episodes}
    candidate_by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    measurement_counts: Counter[str] = Counter()
    ls_acoustic_counts: Counter[str] = Counter()
    for candidate in sorted(candidates, key=lambda row: row["candidate_id"]):
        episode = episode_by_id[str(candidate["episode_id"])]
        bounds = episode["bounds"]
        warm_start = int(bounds["warm_start"])
        tail_end = int(bounds["tail_end"])
        boundary = int(candidate["coordinate"])
        wav = str(candidate["wav_sha256"])
        for window in ADJACENT_WINDOWS:
            if boundary - window < warm_start or boundary + window > tail_end:
                continue
            profile = f"measurement_adjacent:W{window}"
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "kind": "reference_aligned_measurement",
                    "profile": profile,
                    "boundary": boundary,
                    "observation_frontier": boundary + window,
                }
            )
            measurement_counts[profile] += 1
            windows.add((wav, boundary - window, boundary))
            windows.add((wav, boundary, boundary + window))
        for window in ANCHOR_WINDOWS:
            if boundary + window > tail_end:
                continue
            profile = f"measurement_probe:W{window}"
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "kind": "read_only_anchor_probe",
                    "profile": profile,
                    "boundary": boundary,
                    "observation_frontier": boundary + window,
                }
            )
            measurement_counts[profile] += 1
            windows.add((wav, boundary, boundary + window))
        for horizon_ms, support in LS_ACOUSTIC_SUPPORT_BY_HORIZON.items():
            if boundary - support < warm_start or boundary + support > tail_end:
                continue
            profile = f"ls_acoustic:H{horizon_ms}:W{support}"
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "kind": "ls_reference_aligned_acoustic",
                    "profile": profile,
                    "boundary": boundary,
                    "observation_frontier": boundary + support,
                }
            )
            ls_acoustic_counts[profile] += 1
            acoustic_windows.add((wav, boundary - support, boundary))
            acoustic_windows.add((wav, boundary, boundary + support))
    rows.sort(key=canonical_json)
    row_digest = hashlib.sha256()
    for row in rows:
        row_digest.update(canonical_json(row).encode("utf-8") + b"\n")
    window_rows = [
        {"wav_sha256": wav, "start": start, "end": end} for wav, start, end in sorted(windows)
    ]
    window_digest = hashlib.sha256()
    for row in window_rows:
        window_digest.update(canonical_json(row).encode("utf-8") + b"\n")
    acoustic_rows = [
        {"wav_sha256": wav, "start": start, "end": end}
        for wav, start, end in sorted(acoustic_windows)
    ]
    acoustic_digest = hashlib.sha256()
    for row in acoustic_rows:
        acoustic_digest.update(canonical_json(row).encode("utf-8") + b"\n")
    paired_valid: dict[str, int] = {}
    for horizon_ms, support in LS_ACOUSTIC_SUPPORT_BY_HORIZON.items():
        count = 0
        for pair in pairs:
            positive = candidate_by_id[str(pair["positive_id"])]
            negative = candidate_by_id[str(pair["negative_id"])]
            if ls_candidate_valid(
                positive, episode_by_id[str(positive["episode_id"])], horizon_ms, support
            ) and ls_candidate_valid(
                negative, episode_by_id[str(negative["episode_id"])], horizon_ms, support
            ):
                count += 1
        if count == 0:
            raise RuntimeError(f"no valid paired LS acoustic comparisons at {horizon_ms} ms")
        paired_valid[str(horizon_ms)] = count
    return {
        "coordinate_row_count": len(rows),
        "coordinate_rows_sha256": row_digest.hexdigest(),
        "grid_profile_counts": dict(sorted(profile_counts.items())),
        "source_prefix_profile_counts": dict(sorted(source_prefix_counts.items())),
        "source_prefix_public_sources": sorted(
            public_source_rows.values(), key=lambda row: row["source_id"]
        ),
        "source_prefix_state_snapshot_count": len(state_snapshot_rows),
        "source_prefix_state_snapshot_rows_sha256": sha256_bytes(
            b"".join(
                canonical_json(row).encode("utf-8") + b"\n"
                for row in sorted(state_snapshot_rows, key=canonical_json)
            )
        ),
        "measurement_profile_counts": dict(sorted(measurement_counts.items())),
        "ls_acoustic_profile_counts": dict(sorted(ls_acoustic_counts.items())),
        "ls_paired_valid_by_horizon": paired_valid,
        "unique_embedding_window_count": len(window_rows),
        "unique_embedding_windows_sha256": window_digest.hexdigest(),
        "first_embedding_window": window_rows[0],
        "last_embedding_window": window_rows[-1],
        "unique_acoustic_window_count": len(acoustic_rows),
        "unique_acoustic_windows_sha256": acoustic_digest.hexdigest(),
        "first_acoustic_window": acoustic_rows[0],
        "last_acoustic_window": acoustic_rows[-1],
    }


def fixture_ledger(experiment_dir: Path) -> dict[str, Any]:
    parity_root = (
        Path(os.environ.get("TEMP") or tempfile.gettempdir()) / "opencode" / "parity_cache"
    )
    clips = []
    for name in (
        "golden_silence.wav",
        "golden_single_utterance.wav",
        "golden_two_utterance_gap400.wav",
    ):
        path = experiment_dir / "data" / "generated" / name
        clips.append({"name": name, "size": path.stat().st_size, "sha256": sha256_file(path)})
    external = {
        "speaker1_a_cn_16k.wav": (
            118932,
            "5f20ce0ddc378ca3239d3ce864b1142726a46a1221ae553912e4e142045df58b",
        ),
        "speaker1_b_cn_16k.wav": (
            157058,
            "20745dc08a4281894d146140b99b9ef7417ac681119b7f7202f553cdf1a85f65",
        ),
        "speaker2_a_cn_16k.wav": (
            170028,
            "8a6cffa452df32ef10503f7992f22ffcdd7f16c4e0273d13311bc5cdcb13abf4",
        ),
    }
    for name, (size, expected) in external.items():
        path = parity_root / name
        actual = sha256_file(path) if path.is_file() else "missing"
        if actual != expected or (path.is_file() and path.stat().st_size != size):
            raise RuntimeError(f"parity clip drift: {name}")
        clips.append({"name": name, "size": size, "sha256": expected})
    return {
        "clips": clips,
        "clips_ledger_sha256": sha256_bytes(canonical_json(clips).encode("utf-8")),
        "parity_frontend_byte_sha256": sha256_file(
            experiment_dir / "results" / "parity_frontend.json"
        ),
        "parity_research_byte_sha256": sha256_file(
            experiment_dir / "results" / "parity_research.json"
        ),
        "research_parity_code_sha256": sha256_file(experiment_dir / "research_parity.py"),
        "run_parity_code_sha256": sha256_file(experiment_dir / "run_parity.py"),
        "run_eres_sweep_code_sha256": sha256_file(experiment_dir / "run_eres_sweep.py"),
    }


def runtime_forecast(
    experiment_dir: Path,
    inventory: dict[str, Any],
    details: dict[str, dict[str, Any]],
    episodes: list[dict[str, Any]],
    coordinate: dict[str, Any],
) -> dict[str, Any]:
    public_sessions: dict[str, int] = {}
    synthetic_seconds = 0.0
    for episode in episodes:
        session_id = str(episode["session_id"])
        bounds = episode["bounds"]
        if synthetic_manifest_name(session_id) is not None:
            synthetic_seconds += (int(bounds["tail_end"]) - int(bounds["warm_start"])) / SAMPLE_RATE
        else:
            public_sessions[session_id] = max(
                public_sessions.get(session_id, 0), int(bounds["tail_end"])
            )
    corpus_root = Path(str(inventory["corpus_root"]))
    source_duration_rows: list[dict[str, Any]] = []
    for session_id in sorted(public_sessions):
        path = corpus_root / str(details[session_id]["wav_path"])
        with wave.open(str(path), "rb") as handle:
            actual_samples = handle.getnframes()
            if (
                handle.getnchannels() != 1
                or handle.getframerate() != SAMPLE_RATE
                or handle.getsampwidth() != 2
            ):
                raise RuntimeError(f"public WAV format drift: {session_id}")
        declared_samples = int(details[session_id]["duration_samples"])
        if int(public_sessions[session_id]) > actual_samples:
            raise RuntimeError(f"public scored tail exceeds WAV: {session_id}")
        source_duration_rows.append(
            {
                "source_id": session_id,
                "declared_samples": declared_samples,
                "actual_wav_samples": actual_samples,
                "difference_samples": actual_samples - declared_samples,
                "maximum_diagnostic_tail": int(public_sessions[session_id]),
            }
        )
    public_source_seconds = (
        sum(int(row["actual_wav_samples"]) for row in source_duration_rows) / SAMPLE_RATE
    )
    ls_audio_seconds_per_checkpoint = public_source_seconds + synthetic_seconds
    ls_jobs_seconds = 4 * ls_audio_seconds_per_checkpoint
    windows_per_checkpoint = int(coordinate["unique_embedding_window_count"])
    embedding_jobs = 2 * windows_per_checkpoint
    ls_wall_seconds = ls_jobs_seconds * LS_RTF_FORECAST
    benchmark_path = (
        experiment_dir / "results" / "turn_episode_v1" / "phase_4_parallel_benchmark.json"
    )
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark_body = {key: value for key, value in benchmark.items() if key != "content_sha256"}
    if benchmark.get("content_sha256") != sha256_bytes(
        canonical_json(benchmark_body).encode("utf-8")
    ):
        raise RuntimeError("parallel benchmark content hash drift")
    if (
        benchmark.get("workers") != ERES_PARALLEL_WORKERS
        or benchmark.get("throughput_margin") != ERES_THROUGHPUT_MARGIN
    ):
        raise RuntimeError("parallel benchmark contract drift")
    throughput = {
        checkpoint: float(row["conservative_jobs_per_second"])
        for checkpoint, row in benchmark["results"].items()
    }
    if set(throughput) != {"E-standard", "E-w24s4ep4"} or any(
        value <= 0 for value in throughput.values()
    ):
        raise RuntimeError("parallel benchmark throughput invalid")
    eres_wall_by_checkpoint = {
        checkpoint: windows_per_checkpoint / value for checkpoint, value in throughput.items()
    }
    eres_wall_seconds = sum(eres_wall_by_checkpoint.values())
    serial_eres_wall_seconds = embedding_jobs * ERES_SECONDS_PER_EMBEDDING
    fixed_overhead_seconds = 900.0
    wall_seconds = ls_wall_seconds + eres_wall_seconds + fixed_overhead_seconds
    cache_bytes = int(
        ls_jobs_seconds * LS_CACHE_BYTES_PER_AUDIO_SECOND
        + embedding_jobs * ERES_CACHE_BYTES_PER_EMBEDDING
        + int(coordinate["unique_acoustic_window_count"]) * ACOUSTIC_CACHE_BYTES_PER_WINDOW
    )
    return {
        "public_source_seconds": round(public_source_seconds, 6),
        "public_session_count": len(public_sessions),
        "public_source_duration_rows": source_duration_rows,
        "synthetic_episode_seconds": round(synthetic_seconds, 6),
        "ls_checkpoint_count": 4,
        "ls_audio_seconds_all_checkpoints": round(ls_jobs_seconds, 6),
        "ls_rtf_forecast": LS_RTF_FORECAST,
        "ls_wall_seconds": round(ls_wall_seconds, 3),
        "eres_checkpoint_count": 2,
        "eres_parallel_workers": ERES_PARALLEL_WORKERS,
        "eres_parallel_benchmark_content_sha256": benchmark["content_sha256"],
        "eres_parallel_benchmark_code_sha256": benchmark["generated_from"][
            "phase4_parallel_benchmark.py"
        ],
        "eres_conservative_jobs_per_second": dict(sorted(throughput.items())),
        "eres_windows_per_checkpoint": windows_per_checkpoint,
        "eres_embedding_jobs": embedding_jobs,
        "eres_seconds_per_embedding": ERES_SECONDS_PER_EMBEDDING,
        "eres_serial_counterfactual_wall_seconds": round(serial_eres_wall_seconds, 3),
        "eres_parallel_wall_seconds_by_checkpoint": {
            key: round(value, 3) for key, value in sorted(eres_wall_by_checkpoint.items())
        },
        "eres_wall_seconds": round(eres_wall_seconds, 3),
        "fixed_overhead_seconds": fixed_overhead_seconds,
        "total_wall_seconds": round(wall_seconds, 3),
        "total_wall_hours": round(wall_seconds / 3600, 6),
        "new_cache_bytes": cache_bytes,
        "peak_rss_bytes": 8 * 1024**3,
        "ceilings": {
            "wall_seconds": 16 * 3600,
            "new_cache_bytes": 8 * 1024**3,
            "peak_rss_bytes": 16 * 1024**3,
        },
        "within_ceilings": (
            wall_seconds <= 16 * 3600 and cache_bytes <= 8 * 1024**3 and 8 * 1024**3 <= 16 * 1024**3
        ),
    }


def build_payload(experiment_dir: Path) -> dict[str, Any]:
    result_dir = experiment_dir / "results" / "turn_episode_v1"
    manifest_path = result_dir / "episode_manifest_dev.json"
    if sha256_file(manifest_path) != MANIFEST_BYTE_SHA256:
        raise RuntimeError("episode manifest byte hash drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["content_sha256"] != MANIFEST_CONTENT_SHA256:
        raise RuntimeError("episode manifest content hash drift")
    if manifest["group_graph_hash"] != GROUP_GRAPH_SHA256:
        raise RuntimeError("group graph hash drift")
    episodes = [row for row in manifest["episodes"] if row["pool"] == "diagnostic_dev"]
    if len(episodes) != 695:
        raise RuntimeError("diagnostic population drift")
    inventory = json.loads((result_dir / "coverage_inventory.json").read_text(encoding="utf-8"))
    details_rows = [
        json.loads(line)
        for line in (result_dir / "coverage_inventory_details.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    details = {str(row["session_id"]): row for row in details_rows}
    manifests_dir = experiment_dir / "data" / "manifests"
    cases = load_synthetic_cases(manifests_dir)
    public_sessions = [
        str(row["session_id"])
        for row in episodes
        if synthetic_manifest_name(str(row["session_id"])) is None
    ]
    regions = load_public_regions(inventory, details, public_sessions, manifests_dir)
    components = component_map(inventory)
    positives, negatives = build_candidates(episodes, cases, components, regions)
    pairs, exclusions = match_pairs(positives, negatives)
    coordinate = coordinate_ledger(episodes, positives + negatives, pairs)
    pair_digest = hashlib.sha256()
    for pair in pairs:
        pair_digest.update(canonical_json(pair).encode("utf-8") + b"\n")
    source_path = Path(__file__).resolve()
    payload = {
        "schema_version": "turn_episode_phase4_design.v1",
        "authority_sha256": AUTHORITY_SHA256,
        "manifest": {
            "byte_sha256": MANIFEST_BYTE_SHA256,
            "content_sha256": MANIFEST_CONTENT_SHA256,
            "group_graph_sha256": GROUP_GRAPH_SHA256,
            "diagnostic_episode_count": len(episodes),
        },
        "generated_from": {"phase4_design.py": sha256_file(source_path)},
        "coordinate_ledger": coordinate,
        "candidate_ledger": {
            "positive_count": len(positives),
            "negative_count": len(negatives),
            "positive_kind_counts": dict(sorted(Counter(row["kind"] for row in positives).items())),
            "negative_kind_counts": dict(sorted(Counter(row["kind"] for row in negatives).items())),
            "positive_rows_sha256": sha256_bytes(
                b"".join(canonical_json(row).encode("utf-8") + b"\n" for row in positives)
            ),
            "negative_rows_sha256": sha256_bytes(
                b"".join(canonical_json(row).encode("utf-8") + b"\n" for row in negatives)
            ),
        },
        "matching": {
            "algorithm": "exact_integer_hungarian.v1",
            "group_key": ["corpus", "language", "block_id"],
            "objective": [
                "maximum_cardinality",
                "stress_mismatch",
                "duration_distance_ms",
                "gap_distance_ms",
                "sha256_tie_rank",
            ],
            "pair_count": len(pairs),
            "pair_rows_sha256": pair_digest.hexdigest(),
            "block_count": len({row["block_id"] for row in pairs}),
            "integer_cost_weights": {
                "stress": str(MATCH_STRESS_WEIGHT),
                "duration": str(MATCH_DURATION_WEIGHT),
                "gap": str(MATCH_GAP_WEIGHT),
                "tie_max": str(MATCH_MAX_TIE),
                "pair_count_bound": MATCH_MAX_PAIRS,
                "feature_distance_bound": MATCH_MAX_FEATURE_DISTANCE,
            },
            "exclusions": exclusions,
            "pairs": pairs,
        },
        "fixture_ledger": fixture_ledger(experiment_dir),
    }
    payload["runtime_forecast"] = runtime_forecast(
        experiment_dir,
        inventory,
        details,
        episodes,
        coordinate,
    )
    return payload


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    with_hash = dict(payload)
    with_hash["content_sha256"] = sha256_bytes(canonical_json(payload).encode("utf-8"))
    encoded = (canonical_json(with_hash) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    experiment_dir = Path(__file__).resolve().parent.parent
    path = args.out or experiment_dir / "results" / "turn_episode_v1" / "phase_4_design_ledger.json"
    payload = build_payload(experiment_dir)
    write_payload(path, payload)
    print(
        json.dumps(
            {
                "path": str(path),
                "byte_sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "content_sha256": json.loads(path.read_text(encoding="utf-8"))["content_sha256"],
                "coordinate_ledger": payload["coordinate_ledger"],
                "candidate_ledger": payload["candidate_ledger"],
                "matching": {
                    key: value for key, value in payload["matching"].items() if key != "pairs"
                },
                "runtime_forecast": payload["runtime_forecast"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
