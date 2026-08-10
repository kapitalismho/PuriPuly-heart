from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import wave
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from experiments.speaker_representation_scd.execution_guard import (
    validate_worker_execution,
)
from experiments.speaker_representation_scd.provenance import (
    canonical_json_bytes,
    load_json,
    sha256_bytes,
    sha256_file,
    with_self_sha256,
)
from experiments.speaker_representation_scd.r1_forecast import (
    FORECAST_CONTRACT_PATH,
    TECHNICAL_VALIDITY_PATH,
)
from experiments.speaker_representation_scd.r1_gate import EXPERIMENT_ROOT, REPOSITORY_ROOT
from experiments.speaker_representation_scd.r2l_forecast import (
    AUTHORITY,
    build_reduced_forecast,
    forecast_provenance,
)
from experiments.speaker_representation_scd.r2l_gate import (
    GATE_PATH,
    validated_r2l_cache_root,
)
from experiments.speaker_representation_scd.run_provenance import run_provenance

CONTEXTS_MS = (100, 300, 500)
HOP_SAMPLES = 1600
TRAJECTORY_OFFSETS_MS = (
    -1000, -750, -500, -300, -200, -100, 0, 100, 200, 300, 500, 750, 1000, 1500, 2000,
)
MAX_R4_SOURCE_HOURS = 6.0
MAX_R4_SOURCE_SECONDS = int(MAX_R4_SOURCE_HOURS * 3600)
MAX_DERIVED_BYTES = 20 * 1024**3
MAX_EXTERNAL_BYTES = 50 * 1024**3
MAX_COORDINATE_ROW_BYTES = 1024
MANIFEST_BYTE_SHA256 = "a5df5e56c2917ba174fca7892fbaa092f2b57705cad7a77bf9a347ba94cddfee"
MANIFEST_CONTENT_SHA256 = "deb1713cd581c93cc13c47103643041c7551993985379869cfa0ca9a407dff68"
INVENTORY_SHA256 = "02a6a118fc90c0d747e9548f07003177b3fc703f33d408d5338427cb6163dd46"
DETAILS_SHA256 = "15b2e4f0efa270985c3bbc6d848ee9ed25496089268e561bff921c5c1be3ef8c"
DESIGN_LEDGER_SHA256 = "0a86788a4817d4a205d92b0afb6ee05dc97d11da3e99d4c0501d74be30473691"
DESIGN_LEDGER_CONTENT_SHA256 = "c8336c2665b28047b1a169fc9605a6c6a3c400afe553dc3a2d35ca9b20b41536"
LS_MANIFESTS = {
    "ls_dev": "14347cdbdb2eff4cc73489f1b59d6755723d9098089dad66ae222984e90370dd",
    "ls_held_out_clean": "c0aabc5ad8c3f00ec53d45f3b372b8ebca7ca9237720a1bb7a70b8de7dda2581",
    "ls_held_out_other": "f0d169394a9fdee9e708bc9cad46c0547946bf967799fa4e2e1a398ddb984079",
}
PHASE4_DESIGN_SHA256 = "bbec4e069165dad309c1dd4103521269494bd8c1dcd115a0d2265f632b6edd4a"
VALIDATION_RECEIPT_PATH = Path("manifests/r2/legacy_common_gt/validation_receipt.json")
COORDINATE_LEDGER_PATH = Path("manifests/r2/legacy_common_gt/coordinate_ledger.json")
REDUCED_FORECAST_PATH = Path("manifests/r2/legacy_common_gt/reduced_r3_r4_forecast.json")
WAVEFORM_INVENTORY_PATH = Path("data/r2/legacy_common_gt/waveform_inventory.jsonl")
SOURCE_METADATA_PATH = Path("data/r2/legacy_common_gt/source_metadata.jsonl")
COORDINATES_DIR = Path("data/r2/legacy_common_gt/coordinates")


class R2LValidationError(RuntimeError):
    pass


def _write_json(path: Path, document: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise R2LValidationError(f"refusing to overwrite an existing R2-L artifact: {path}")
    payload = with_self_sha256(document)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(encoded, encoding="utf-8", newline="\n")
    temporary.replace(path)
    return payload


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            encoded = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            if len(encoded.encode("utf-8")) > MAX_COORDINATE_ROW_BYTES:
                raise R2LValidationError("R2-L coordinate row exceeds its byte ceiling")
            handle.write(encoded)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(path)


def _legacy_result_dir(repo_root: Path) -> Path:
    return repo_root / "experiments" / "speaker_turn_boundary" / "results" / "turn_episode_v1"


def _legacy_manifests_dir(repo_root: Path) -> Path:
    return repo_root / "experiments" / "speaker_turn_boundary" / "data" / "manifests"


def load_legacy_documents(repo_root: Path) -> dict[str, Any]:
    result_dir = _legacy_result_dir(repo_root)
    manifests_dir = _legacy_manifests_dir(repo_root)

    def checked(path: Path, expected: str, label: str) -> Any:
        if not path.is_file():
            raise R2LValidationError(f"{label}: missing")
        if sha256_file(path) != expected:
            raise R2LValidationError(f"{label}: byte identity differs")
        try:
            return load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise R2LValidationError(f"{label}: unreadable: {exc}") from exc

    manifest_path = result_dir / "episode_manifest_dev.json"
    manifest = checked(manifest_path, MANIFEST_BYTE_SHA256, "legacy manifest")
    if manifest.get("content_sha256") != MANIFEST_CONTENT_SHA256:
        raise R2LValidationError("legacy manifest: content identity differs")
    inventory = checked(result_dir / "coverage_inventory.json", INVENTORY_SHA256, "coverage inventory")
    details_rows = [
        json.loads(line)
        for line in (result_dir / "coverage_inventory_details.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    if sha256_file(result_dir / "coverage_inventory_details.jsonl") != DETAILS_SHA256:
        raise R2LValidationError("coverage inventory details: byte identity differs")
    if not details_rows or any(not isinstance(row, dict) for row in details_rows):
        raise R2LValidationError("coverage inventory details: invalid rows")
    design_ledger = checked(
        result_dir / "phase_4_design_ledger.json", DESIGN_LEDGER_SHA256, "phase-4 design ledger"
    )
    if design_ledger.get("content_sha256") != DESIGN_LEDGER_CONTENT_SHA256:
        raise R2LValidationError("phase-4 design ledger: content identity differs")
    cases: dict[str, Any] = {}
    for name, expected in LS_MANIFESTS.items():
        raw = checked(manifests_dir / f"{name}.json", expected, f"ls manifest {name}")
        for case in raw.get("cases") or []:
            cases[(name, str(case["case_id"]))] = case
    phase4_path = repo_root / "experiments" / "speaker_turn_boundary" / "turn_episode" / "phase4_design.py"
    if not phase4_path.is_file() or sha256_file(phase4_path) != PHASE4_DESIGN_SHA256:
        raise R2LValidationError("legacy phase4_design.py: byte identity differs")
    details = {str(row["session_id"]): row for row in details_rows}
    return {
        "manifest": manifest,
        "inventory": inventory,
        "details": details,
        "details_rows": details_rows,
        "design_ledger": design_ledger,
        "cases": cases,
        "manifests_dir": manifests_dir,
    }


def _ensure_src_importable() -> None:
    src = (REPOSITORY_ROOT / "src").resolve()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def derive_candidates(documents: dict[str, Any]) -> dict[str, Any]:
    _ensure_src_importable()
    from experiments.speaker_turn_boundary.turn_episode import phase4_design as legacy

    manifest = documents["manifest"]
    episodes = [row for row in manifest["episodes"] if row["pool"] == "diagnostic_dev"]
    if len(episodes) != 695:
        raise R2LValidationError(f"diagnostic population drift: {len(episodes)}")
    public_sessions = [
        str(row["session_id"])
        for row in episodes
        if legacy.synthetic_manifest_name(str(row["session_id"])) is None
    ]
    regions = legacy.load_public_regions(
        documents["inventory"],
        documents["details"],
        public_sessions,
        documents["manifests_dir"],
    )
    components = legacy.component_map(documents["inventory"])
    positives, negatives = legacy.build_candidates(
        episodes, documents["cases"], components, regions
    )
    pairs, exclusions = legacy.match_pairs(positives, negatives)
    ledger = documents["design_ledger"]["candidate_ledger"]
    matching = documents["design_ledger"]["matching"]

    def digest(rows: list[dict[str, Any]]) -> str:
        value = hashlib.sha256()
        for row in rows:
            value.update(legacy.canonical_json(row).encode("utf-8") + b"\n")
        return value.hexdigest()

    positive_digest = digest(positives)
    negative_digest = digest(negatives)
    pair_digest = digest(pairs)
    if positive_digest != ledger.get("positive_rows_sha256"):
        raise R2LValidationError("R3 positive anchor rows differ from the frozen ledger")
    if negative_digest != ledger.get("negative_rows_sha256"):
        raise R2LValidationError("R3 negative anchor rows differ from the frozen ledger")
    if len(positives) != 450 or len(negatives) != 360:
        raise R2LValidationError("R3 anchor counts differ from the frozen ledger")
    if pair_digest != matching.get("pair_rows_sha256") or len(pairs) != 313:
        raise R2LValidationError("matched pairs differ from the frozen ledger")
    if matching.get("exclusions") != exclusions:
        raise R2LValidationError("pair matching exclusions differ from the frozen ledger")
    return {
        "episodes": episodes,
        "positives": positives,
        "negatives": negatives,
        "pairs": pairs,
        "exclusions": exclusions,
        "regions": regions,
        "components": components,
    }


def _region_dicts(region: Any) -> dict[str, Any]:
    if isinstance(region, dict):
        return {
            "start_sample": int(region["start_sample"]),
            "end_sample": int(region["end_sample"]),
            "speakers": list(region["speakers"]),
            "ambiguous": bool(region["ambiguous"]),
        }
    return {
        "start_sample": int(region.start_sample),
        "end_sample": int(region.end_sample),
        "speakers": list(region.speakers),
        "ambiguous": bool(region.ambiguous),
    }


def regions_by_session(documents: dict[str, Any], derived: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for session_id, regions in derived["regions"].items():
        result[str(session_id)] = [_region_dicts(region) for region in regions]
    for (name, case_id), case in documents["cases"].items():
        for region in case.get("regions") or []:
            session_id = f"{name}:{case_id}"
            result.setdefault(session_id, []).append(
                {
                    "start_sample": int(region["start_sample"]),
                    "end_sample": int(region["end_sample"]),
                    "speakers": list(region["speakers"]),
                    "ambiguous": bool(region["ambiguous"]),
                }
            )
    return result


def resolve_waveforms(
    documents: dict[str, Any],
    derived: dict[str, Any],
    corpus_root: Path,
) -> dict[str, dict[str, Any]]:
    details = documents["details"]
    episodes = derived["episodes"]
    max_end: dict[str, int] = defaultdict(int)
    for episode in episodes:
        session_id = str(episode["session_id"])
        max_end[session_id] = max(max_end[session_id], int(episode["bounds"]["scored_end"]))
    synthetic = {
        (name, case_id)
        for (name, case_id) in documents["cases"]
    }
    waveforms: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        session_id = str(episode["session_id"])
        manifest_name = session_id.split(":", 1)[0]
        if manifest_name in LS_MANIFESTS:
            case_id = session_id.split(":", 1)[1]
            if (manifest_name, case_id) not in synthetic:
                raise R2LValidationError(f"synthetic case missing for {session_id}")
            relative = Path("generated") / f"{case_id}.wav"
        else:
            row = details.get(session_id)
            if row is None or not row.get("wav_path"):
                raise R2LValidationError(f"WAV identity missing for {session_id}")
            relative = Path(str(row["wav_path"]).replace("\\", "/"))
        expected_sha = episode["wav_sha256"]
        path = corpus_root / relative
        if not path.is_file():
            raise R2LValidationError(f"legacy WAV missing: {path}")
        if sha256_file(path) != expected_sha:
            raise R2LValidationError(f"legacy WAV identity differs: {path}")
        with wave.open(str(path), "rb") as handle:
            geometry = {
                "channels": handle.getnchannels(),
                "sample_width": handle.getsampwidth(),
                "sample_rate_hz": handle.getframerate(),
                "num_samples": handle.getnframes(),
            }
        if (
            geometry["channels"] != 1
            or geometry["sample_width"] != 2
            or geometry["sample_rate_hz"] != 16000
        ):
            raise R2LValidationError(f"legacy WAV geometry differs from canonical: {path}")
        eligible_end = (
            geometry["num_samples"]
            if manifest_name in LS_MANIFESTS
            else max_end[session_id]
        )
        if eligible_end < 8000:
            raise R2LValidationError(f"legacy WAV is too short for any context: {path}")
        waveform = waveforms.setdefault(
            expected_sha,
            {
                "waveform_id": expected_sha,
                "source_id": "legacy-common-gt-v1",
                "session_ids": [],
                "artifact_relative_path": relative.as_posix(),
                "artifact_sha256": expected_sha,
                "artifact_size_bytes": path.stat().st_size,
                "sample_rate_hz": 16000,
                "num_samples": geometry["num_samples"],
                "eligible_start_sample": 0,
                "eligible_end_sample": eligible_end,
            },
        )
        if session_id not in waveform["session_ids"]:
            waveform["session_ids"].append(session_id)
        waveform["eligible_end_sample"] = max(waveform["eligible_end_sample"], eligible_end)
    if len(waveforms) != 600:
        raise R2LValidationError(f"unique legacy waveform count differs: {len(waveforms)}")
    return waveforms


def build_source_metadata(
    documents: dict[str, Any],
    derived: dict[str, Any],
    waveforms: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    _ensure_src_importable()
    from experiments.speaker_turn_boundary.turn_episode import phase4_design as legacy

    pair_sources: set[str] = set()
    by_id = {row["candidate_id"]: row for row in derived["positives"] + derived["negatives"]}
    for pair in derived["pairs"]:
        pair_sources.add(str(by_id[pair["positive_id"]]["session_id"]))
        pair_sources.add(str(by_id[pair["negative_id"]]["session_id"]))
    episodes_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    max_scored_end: dict[str, int] = defaultdict(int)
    for episode in derived["episodes"]:
        session_id = str(episode["session_id"])
        episodes_by_session[session_id].append(episode)
        max_scored_end[session_id] = max(
            max_scored_end[session_id], int(episode["bounds"]["scored_end"])
        )
    rows: list[dict[str, Any]] = []
    for session_id in sorted(episodes_by_session):
        episodes = episodes_by_session[session_id]
        manifest_name = legacy.synthetic_manifest_name(session_id)
        wav_sha = str(episodes[0]["wav_sha256"])
        waveform = waveforms[wav_sha]
        annotations = sorted({str(episode["annotation_sha256"]) for episode in episodes})
        eligible_end = (
            waveform["num_samples"] if manifest_name is not None else max_scored_end[session_id]
        )
        row = {
            "session_id": session_id,
            "corpus": legacy.corpus_for(session_id),
            "language": legacy.language_for(session_id),
            "block_id": legacy.block_id(episodes[0], derived["components"], manifest_name),
            "synthetic_manifest": manifest_name,
            "episode_count": len(episodes),
            "annotation_sha256": annotations,
            "waveform_id": wav_sha,
            "num_samples": waveform["num_samples"],
            "eligible_start_sample": 0,
            "eligible_end_sample": eligible_end,
            "duration_seconds": eligible_end / 16000,
            "stratum": "a" if session_id in pair_sources else "b",
        }
        rows.append(row)
    if len(rows) != 616:
        raise R2LValidationError(f"source identity count differs: {len(rows)}")
    return rows


def freeze_r4_panel(
    source_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    def key(row: dict[str, Any]) -> tuple[int, str]:
        return (
            0 if row["stratum"] == "a" else 1,
            hashlib.sha256(str(row["session_id"]).encode("utf-8")).hexdigest(),
        )

    ordered = sorted(source_rows, key=key)
    panel: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    total_seconds = 0.0
    for row in ordered:
        seconds = float(row["duration_seconds"])
        if total_seconds + seconds > MAX_R4_SOURCE_SECONDS:
            excluded.append(
                {
                    "session_id": row["session_id"],
                    "duration_seconds": seconds,
                    "reason": "r4_source_hour_cap",
                }
            )
            continue
        panel.append(row)
        total_seconds += seconds
    return panel, excluded, total_seconds


def _overlaps(window_start: int, window_end: int, region_start: int, region_end: int) -> bool:
    return window_start < region_end and region_start < window_end


def _classify_window(
    window_start: int,
    window_end: int,
    boundary: int,
    regions: list[dict[str, Any]],
    candidate_class: str,
) -> str:
    if candidate_class == "negative":
        for region in regions:
            if region["ambiguous"] and _overlaps(
                window_start, window_end, region["start_sample"], region["end_sample"]
            ):
                return "ambiguous"
            if (
                not region["speakers"]
                and window_start >= region["start_sample"]
                and window_end <= region["end_sample"]
            ):
                return "silence"
            if len(region["speakers"]) >= 2 and _overlaps(
                window_start, window_end, region["start_sample"], region["end_sample"]
            ):
                return "overlap"
            if (
                len(region["speakers"]) == 1
                and window_start >= region["start_sample"]
                and window_end <= region["end_sample"]
            ):
                return "stable_same_speaker"
        return "reference_relative"
    for region in regions:
        if region["ambiguous"] and _overlaps(
            window_start, window_end, region["start_sample"], region["end_sample"]
        ):
            return "ambiguous"
        if (
            not region["speakers"]
            and window_start >= region["start_sample"]
            and window_end <= region["end_sample"]
        ):
            return "silence"
        if len(region["speakers"]) >= 2 and _overlaps(
            window_start, window_end, region["start_sample"], region["end_sample"]
        ):
            return "overlap"
    if window_end <= boundary:
        return "entirely_old"
    if window_start >= boundary:
        return "entirely_new"
    return "boundary_straddling"


def _coordinate_row(
    session_id: str,
    waveform_id: str,
    context_ms: int,
    frontier: int,
    role: str,
    candidate_id: str | None,
    offset_ms: int | None,
    window_class: str | None,
) -> dict[str, Any]:
    payload = {
        "schema": "r2l_trailing_window_coordinate_v1",
        "scope": "legacy-common-gt-v1",
        "session_id": session_id,
        "waveform_id": waveform_id,
        "context_ms": context_ms,
        "window_start_sample": frontier - context_ms * 16,
        "window_end_sample": frontier,
        "observed_frontier_sample": frontier,
        "hop_samples": HOP_SAMPLES,
        "coordinate_role": role,
        "candidate_id": candidate_id,
        "trajectory_offset_ms": offset_ms,
        "window_class": window_class,
    }
    return {
        "coordinate_id": sha256_bytes(canonical_json_bytes(payload)),
        **payload,
    }


def generate_coordinates(
    documents: dict[str, Any],
    derived: dict[str, Any],
    waveforms: dict[str, dict[str, Any]],
    source_rows: list[dict[str, Any]],
    regions: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], list[dict[str, Any]]]:
    rows_by_waveform: dict[str, list[dict[str, Any]]] = {
        waveform_id: [] for waveform_id in waveforms
    }
    r4_counts: dict[str, int] = defaultdict(int)
    for row in source_rows:
        session_id = str(row["session_id"])
        waveform_id = str(row["waveform_id"])
        eligible_start = int(row["eligible_start_sample"])
        eligible_end = int(row["eligible_end_sample"])
        for context_ms in CONTEXTS_MS:
            first = eligible_start + context_ms * 16
            for frontier in range(first, eligible_end + 1, HOP_SAMPLES):
                rows_by_waveform[waveform_id].append(
                    _coordinate_row(
                        session_id,
                        waveform_id,
                        context_ms,
                        frontier,
                        "r4_continuous",
                        None,
                        None,
                        None,
                    )
                )
                r4_counts[str(context_ms)] += 1
    r3_primary = 0
    r3_trajectory = 0
    excluded_rows: list[dict[str, Any]] = []
    for candidate in derived["positives"] + derived["negatives"]:
        session_id = str(candidate["session_id"])
        waveform_id = str(candidate["wav_sha256"])
        boundary = int(candidate["coordinate"])
        eligible = waveforms[waveform_id]
        eligible_start = eligible["eligible_start_sample"]
        eligible_end = eligible["eligible_end_sample"]
        session_regions = regions.get(session_id, [])
        candidate_class = str(candidate["class"])
        for context_ms in CONTEXTS_MS:
            primary_frontier = boundary + context_ms * 16
            if primary_frontier > eligible_end or primary_frontier < eligible_start:
                excluded_rows.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "context_ms": context_ms,
                        "offset_ms": context_ms,
                        "reason": "primary_window_out_of_eligible_range",
                    }
                )
            else:
                rows_by_waveform[waveform_id].append(
                    _coordinate_row(
                        session_id,
                        waveform_id,
                        context_ms,
                        primary_frontier,
                        "r3_primary",
                        candidate["candidate_id"],
                        None,
                        _classify_window(
                            primary_frontier - context_ms * 16,
                            primary_frontier,
                            boundary,
                            session_regions,
                            candidate_class,
                        ),
                    )
                )
                r3_primary += 1
            for offset_ms in TRAJECTORY_OFFSETS_MS:
                frontier = boundary + offset_ms * 16
                if frontier - context_ms * 16 < eligible_start or frontier > eligible_end:
                    excluded_rows.append(
                        {
                            "candidate_id": candidate["candidate_id"],
                            "context_ms": context_ms,
                            "offset_ms": offset_ms,
                            "reason": "trajectory_window_out_of_eligible_range",
                        }
                    )
                    continue
                rows_by_waveform[waveform_id].append(
                    _coordinate_row(
                        session_id,
                        waveform_id,
                        context_ms,
                        frontier,
                        "r3_trajectory",
                        candidate["candidate_id"],
                        offset_ms,
                        _classify_window(
                            frontier - context_ms * 16,
                            frontier,
                            boundary,
                            session_regions,
                            candidate_class,
                        ),
                    )
                )
                r3_trajectory += 1
    return (
        rows_by_waveform,
        {
            "r3_primary": r3_primary,
            "r3_trajectory": r3_trajectory,
            "r4_counts": dict(sorted(r4_counts.items())),
        },
        excluded_rows,
    )


def assemble_ledger(
    cache_root: Path,
    documents: dict[str, Any],
    derived: dict[str, Any],
    source_rows: list[dict[str, Any]],
    panel_excluded: list[dict[str, Any]],
    panel_total_seconds: float,
    rows_by_waveform: dict[str, list[dict[str, Any]]],
    coordinate_counts: dict[str, Any],
    excluded_rows: list[dict[str, Any]],
    supervision_binding: dict[str, Any],
) -> dict[str, Any]:
    shards: list[dict[str, Any]] = []
    total_rows = 0
    for waveform_id in sorted(rows_by_waveform):
        rows = rows_by_waveform[waveform_id]
        if not rows:
            continue
        path = cache_root / COORDINATES_DIR / f"{waveform_id}.jsonl"
        _write_jsonl(path, rows)
        total_rows += len(rows)
        shards.append(
            {
                "waveform_id": waveform_id,
                "relative_to_cache_root": path.relative_to(cache_root).as_posix(),
                "row_count": len(rows),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    panel_sources = []
    for row in source_rows:
        panel_sources.append(
            {
                "session_id": row["session_id"],
                "corpus": row["corpus"],
                "language": row["language"],
                "synthetic_manifest": row["synthetic_manifest"],
                "stratum": row["stratum"],
                "eligible_start_sample": row["eligible_start_sample"],
                "eligible_end_sample": row["eligible_end_sample"],
                "duration_seconds": row["duration_seconds"],
            }
        )
    ledger = {
        "schema_version": 1,
        "artifact_role": "r2l_legacy_common_gt_coordinate_ledger",
        "experiment_id": "speaker_representation_scd_v1",
        "authority": AUTHORITY,
        "scope": "legacy-common-gt-v1",
        "coordinate_contract": {
            "contexts_ms": list(CONTEXTS_MS),
            "hop_samples": HOP_SAMPLES,
            "frontier_rule": "eligible_start_plus_context_through_eligible_end",
            "r3_primary_frontier_rule": "candidate_coordinate_plus_context",
            "r3_trajectory_offsets_ms": list(TRAJECTORY_OFFSETS_MS),
            "one_shard_per_waveform": True,
        },
        "r3": {
            "maximum_shared_anchor_count": 810,
            "positive_anchor_count": len(derived["positives"]),
            "negative_anchor_count": len(derived["negatives"]),
            "positive_rows_sha256": documents["design_ledger"]["candidate_ledger"][
                "positive_rows_sha256"
            ],
            "negative_rows_sha256": documents["design_ledger"]["candidate_ledger"][
                "negative_rows_sha256"
            ],
            "pair_count": len(derived["pairs"]),
            "pair_rows_sha256": documents["design_ledger"]["matching"]["pair_rows_sha256"],
            "block_count": len({row["block_id"] for row in derived["pairs"]}),
            "primary_window_count": coordinate_counts["r3_primary"],
            "trajectory_window_count": coordinate_counts["r3_trajectory"],
            "excluded_rows": excluded_rows,
        },
        "r4": {
            "maximum_source_hours": MAX_R4_SOURCE_HOURS,
            "selection_rule": {
                "source_order_rule": "stratum_then_sha256",
                "stratum_a": "sources_hosting_at_least_one_matched_pair_anchor",
                "cumulative_cap_rule": "include_while_total_seconds_at_or_below_21600",
            },
            "panel_source_count": len(source_rows),
            "panel_total_source_hours": round(panel_total_seconds / 3600, 6),
            "panel_sources": panel_sources,
            "excluded_sources": panel_excluded,
            "windows_by_context_ms": coordinate_counts["r4_counts"],
        },
        "total_window_count": total_rows,
        "coordinate_shards": shards,
        "supervision_binding": supervision_binding,
    }
    return _write_json(cache_root / COORDINATE_LEDGER_PATH, ledger)


def materialize(cache_root: Path, requested_argv: tuple[str, ...]) -> dict[str, Any]:
    validated_r2l_cache_root("coordinate_materialization")
    receipt_path = cache_root / VALIDATION_RECEIPT_PATH
    execution = validate_worker_execution(cache_root, receipt_path)
    if execution.requested_argv != requested_argv:
        raise R2LValidationError("R2-L worker invocation differs from its lease")
    existing = [
        path
        for path in (
            receipt_path,
            cache_root / COORDINATE_LEDGER_PATH,
            cache_root / REDUCED_FORECAST_PATH,
            cache_root / WAVEFORM_INVENTORY_PATH,
            cache_root / SOURCE_METADATA_PATH,
        )
        if path.exists()
    ]
    if existing:
        raise R2LValidationError(f"refusing to overwrite an existing R2-L result: {existing}")
    gate_path = EXPERIMENT_ROOT / GATE_PATH
    gate = load_json(gate_path)
    supervision_binding = {
        "execution_id": execution.execution_id,
        "expected_receipt_relative_path": execution.expected_receipt_relative_path,
        "authority": "requires_completed_usage_attestation",
    }
    documents = load_legacy_documents(REPOSITORY_ROOT)
    derived = derive_candidates(documents)
    corpus_root = Path(str(documents["inventory"]["corpus_root"])).resolve()
    waveforms = resolve_waveforms(documents, derived, corpus_root)
    source_rows = build_source_metadata(documents, derived, waveforms)
    panel, panel_excluded, panel_total_seconds = freeze_r4_panel(source_rows)
    region_map = regions_by_session(documents, derived)
    rows_by_waveform, coordinate_counts, excluded_rows = generate_coordinates(
        documents, derived, waveforms, panel, region_map
    )
    inventory_path = cache_root / WAVEFORM_INVENTORY_PATH
    source_metadata_path = cache_root / SOURCE_METADATA_PATH
    _write_jsonl(
        inventory_path,
        sorted(
            (
                {
                    "waveform_id": waveform["waveform_id"],
                    "source_id": waveform["source_id"],
                    "session_ids": waveform["session_ids"],
                    "artifact_relative_path": waveform["artifact_relative_path"],
                    "artifact_sha256": waveform["artifact_sha256"],
                    "artifact_size_bytes": waveform["artifact_size_bytes"],
                    "sample_rate_hz": waveform["sample_rate_hz"],
                    "num_samples": waveform["num_samples"],
                    "eligible_start_sample": waveform["eligible_start_sample"],
                    "eligible_end_sample": waveform["eligible_end_sample"],
                }
                for waveform in waveforms.values()
            ),
            key=lambda row: row["waveform_id"],
        ),
    )
    _write_jsonl(
        source_metadata_path,
        sorted(source_rows, key=lambda row: row["session_id"]),
    )
    ledger = assemble_ledger(
        cache_root,
        documents,
        derived,
        panel,
        panel_excluded,
        panel_total_seconds,
        rows_by_waveform,
        coordinate_counts,
        excluded_rows,
        supervision_binding,
    )
    technical = load_json(EXPERIMENT_ROOT / TECHNICAL_VALIDITY_PATH)
    contract = load_json(EXPERIMENT_ROOT / FORECAST_CONTRACT_PATH)
    forecast = build_reduced_forecast(
        technical,
        contract,
        ledger,
        cache_root,
        forecast_provenance(requested_argv),
        supervision_binding=supervision_binding,
    )
    forecast_path = cache_root / REDUCED_FORECAST_PATH
    if forecast_path.exists():
        raise R2LValidationError("refusing to overwrite an existing R2-L forecast")
    encoded = json.dumps(forecast, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = forecast_path.with_name(f".{forecast_path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8", newline="\n")
    temporary.replace(forecast_path)
    validation_receipt = _write_json(
        receipt_path,
        {
            "schema_version": 1,
            "artifact_role": "r2l_legacy_common_gt_validation_receipt",
            "experiment_id": "speaker_representation_scd_v1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "authority": AUTHORITY,
            "scope": "legacy-common-gt-v1",
            "corpus_root": str(corpus_root),
            "manifest": {
                "byte_sha256": MANIFEST_BYTE_SHA256,
                "content_sha256": MANIFEST_CONTENT_SHA256,
                "episode_count": documents["manifest"]["episode_count"],
                "diagnostic_episode_count": len(derived["episodes"]),
            },
            "validation_checks": {
                "manifest_identity": True,
                "wav_identity": True,
                "annotation_identity": True,
                "event_identity": True,
                "pair_identity": True,
                "block_identity": True,
                "geometry_16k_mono_s16": True,
                "candidate_digests_match_frozen_ledger": True,
            },
            "waveform_inventory": {
                "relative_to_cache_root": WAVEFORM_INVENTORY_PATH.as_posix(),
                "size_bytes": inventory_path.stat().st_size,
                "sha256": sha256_file(inventory_path),
                "waveform_count": len(waveforms),
            },
            "source_metadata": {
                "relative_to_cache_root": SOURCE_METADATA_PATH.as_posix(),
                "size_bytes": source_metadata_path.stat().st_size,
                "sha256": sha256_file(source_metadata_path),
                "source_count": len(source_rows),
            },
            "coordinate_ledger": {
                "relative_to_cache_root": COORDINATE_LEDGER_PATH.as_posix(),
                "sha256": sha256_file(cache_root / COORDINATE_LEDGER_PATH),
                "self_sha256": ledger["self_sha256"],
                "total_window_count": ledger["total_window_count"],
            },
            "reduced_forecast": {
                "relative_to_cache_root": REDUCED_FORECAST_PATH.as_posix(),
                "sha256": sha256_file(forecast_path),
                "self_sha256": forecast["self_sha256"],
                "status": forecast["status"],
            },
            "r3_anchors": {
                "positive_count": len(derived["positives"]),
                "negative_count": len(derived["negatives"]),
                "pair_count": len(derived["pairs"]),
                "block_count": len({row["block_id"] for row in derived["pairs"]}),
                "positive_rows_sha256": documents["design_ledger"]["candidate_ledger"][
                    "positive_rows_sha256"
                ],
                "negative_rows_sha256": documents["design_ledger"]["candidate_ledger"][
                    "negative_rows_sha256"
                ],
                "pair_rows_sha256": documents["design_ledger"]["matching"][
                    "pair_rows_sha256"
                ],
                "exclusions": derived["exclusions"],
            },
            "r4_panel": {
                "maximum_source_hours": MAX_R4_SOURCE_HOURS,
                "panel_source_count": len(panel),
                "panel_total_source_hours": round(panel_total_seconds / 3600, 6),
                "excluded_sources": panel_excluded,
            },
            "wav_reference_policy": "read_only_in_place",
            "network_access": False,
            "neural_inference": False,
            "training": False,
            "r2l_gate_sha256": sha256_file(gate_path),
            "r2l_gate_self_sha256": gate["self_sha256"],
            "execution_code_manifest_sha256": gate["execution_code"]["manifest_sha256"],
            "supervision_binding": supervision_binding,
            "run_provenance": run_provenance(
                REPOSITORY_ROOT,
                requested_argv,
                deterministic_seed=0,
                deterministic_kernels=True,
            ),
        },
    )
    if _derived_tree_size(cache_root) > MAX_DERIVED_BYTES:
        raise R2LValidationError("R2-L outputs exceed the 20 GiB derived ceiling")
    if _tree_size(cache_root) > MAX_EXTERNAL_BYTES:
        raise R2LValidationError("R2-L external cache exceeds the 50 GiB ceiling")
    return validation_receipt


def _tree_size(root: Path) -> int:
    total = 0
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        for name in tuple(directory_names):
            path = Path(directory) / name
            if path.is_symlink():
                raise R2LValidationError(f"external cache symlink forbidden: {path}")
        for name in file_names:
            path = Path(directory) / name
            if path.is_symlink():
                raise R2LValidationError(f"external cache symlink forbidden: {path}")
            total += path.stat().st_size
    return total


def _derived_tree_size(root: Path) -> int:
    total = 0
    for relative in (
        Path("data/r2"),
        Path("manifests/r2"),
        Path("cache/r2"),
    ):
        path = root / relative
        if path.is_dir():
            total += _tree_size(path)
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, choices=("materialize",))
    parser.parse_args(argv)
    cache_root = validated_r2l_cache_root("coordinate_materialization")
    requested = tuple(json.loads(os.environ.get("SRSCD_REQUESTED_ARGV", "[]")))
    receipt = materialize(cache_root, requested)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except R2LValidationError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
