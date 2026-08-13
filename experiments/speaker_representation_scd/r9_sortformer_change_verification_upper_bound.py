from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "r9"
    / "sortformer_change_verification_upper_bound.json"
)
CODE_PATH = Path(__file__).resolve()
R7B_RELATIVE = "results/r7b/fixed_lag_local_segmentation_v1"


class R9Error(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


@dataclass(slots=True)
class Session:
    session_id: str
    fold: int
    waveform_path: Path
    waveform_sha256: str
    first_boundary: int
    last_boundary: int
    scored_hours: float
    events: list[dict[str, Any]]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def config() -> dict[str, Any]:
    return load_json(CONFIG_PATH)


def cache_root() -> Path:
    value = os.environ.get("SRSCD_CACHE_ROOT")
    if not value:
        raise R9Error("SRSCD_CACHE_ROOT is required")
    root = Path(value).resolve()
    if not root.is_absolute() or root == REPOSITORY_ROOT or REPOSITORY_ROOT in root.parents:
        raise R9Error("SRSCD_CACHE_ROOT must be outside the repository")
    return root


def output_root(root: Path) -> Path:
    return root / str(config()["output_relative_path"])


def r8_output_root(root: Path) -> Path:
    return root / str(config()["r8_output_relative_path"])


def r7b_inventory_path(root: Path) -> Path:
    return root / str(config()["r7b_output_relative_path"]) / "inventory.json"


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


def _fold_map(cfg: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for fold, session_ids in enumerate(cfg["folds"]):
        for session_id in session_ids:
            if session_id in result:
                raise R9Error(f"duplicate session in folds: {session_id}")
            result[str(session_id)] = fold
    return result


def _session_rows(root: Path) -> list[dict[str, Any]]:
    path = r7b_inventory_path(root)
    if not path.is_file():
        raise R9Error(f"R7-B inventory is missing: {path}")
    inventory = load_json(path)
    rows = list(inventory["sessions"])
    fold_map = _fold_map(config())
    if set(fold_map) != {str(row["session_id"]) for row in rows}:
        raise R9Error("R9 folds do not match the R7-B inventory")
    for row in rows:
        if fold_map[str(row["session_id"])] != int(row["fold"]):
            raise R9Error(f"fold drift for {row['session_id']}")
    return rows


def _sessions(root: Path) -> dict[str, Session]:
    result: dict[str, Session] = {}
    for row in _session_rows(root):
        waveform_path = Path(row["waveform_path"])
        if not waveform_path.is_file():
            raise R9Error(f"waveform is missing: {waveform_path}")
        if float(row["scored_hours"]) <= 0.0:
            raise R9Error(f"non-positive scored hours for {row['session_id']}")
        result[str(row["session_id"])] = Session(
            session_id=str(row["session_id"]),
            fold=int(row["fold"]),
            waveform_path=waveform_path,
            waveform_sha256=sha256_file(waveform_path),
            first_boundary=int(row["first_boundary_sample"]),
            last_boundary=int(row["last_boundary_sample"]),
            scored_hours=float(row["scored_hours"]),
            events=list(row["events"]),
        )
    return result


def _load_probabilities(root: Path, session_id: str) -> np.ndarray:
    cfg = config()
    directory = output_root(root)
    reuse_path = directory / "r8_reuse_inventory.json"
    if not reuse_path.is_file():
        raise R9Error("prepare must run before loading R8 probabilities")
    reuse = load_json(reuse_path)["reused"]
    key = f"probabilities_cpu_{session_id}"
    if key not in reuse:
        raise R9Error(f"reuse inventory lacks {key}")
    path = Path(reuse[key]["path"])
    if not path.is_file():
        raise R9Error(f"R8 probabilities are missing: {path}")
    if sha256_file(path) != str(reuse[key]["sha256"]):
        raise R9Error(f"R8 probabilities hash drift since prepare: {path}")
    archive = np.load(path)
    values = archive["probabilities"].astype(np.float32)
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise R9Error(f"invalid probabilities: {path}")
    if int(archive["frame_ms"]) != int(cfg["frame_ms"]):
        raise R9Error(f"probability frame identity drift: {path}")
    if str(archive["backend"]) != "cpu":
        raise R9Error(f"probability backend identity drift: {path}")
    model_key = "r8_model_receipt"
    if model_key not in reuse:
        raise R9Error("reuse inventory lacks the R8 model receipt")
    model_receipt = load_json(Path(reuse[model_key]["path"]))
    expected_model_sha = next(
        (
            str(file_row["sha256"])
            for file_row in model_receipt["files"]
            if str(file_row["filename"]) == str(cfg["model_q8_filename"])
        ),
        None,
    )
    if expected_model_sha is None:
        raise R9Error("R8 model receipt lacks the Q8_0 file hash")
    if str(archive["model_sha256"]) != expected_model_sha:
        raise R9Error(f"probability model identity drift: {path}")
    return values


def _segment_start_candidates(probabilities: np.ndarray, frame_ms: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for speaker in range(probabilities.shape[1]):
        active = probabilities[:, speaker] > 0.5
        start: int | None = None
        for frame in range(len(active) + 1):
            value = frame < len(active) and bool(active[frame])
            if value and start is None:
                start = frame
            elif not value and start is not None:
                result.append(
                    {
                        "speaker_slot": speaker + 1,
                        "start_frame": start,
                        "start_ms": start * frame_ms,
                    }
                )
                start = None
    return sorted(result, key=lambda row: (int(row["start_frame"]), int(row["speaker_slot"])))


def _label_candidate(sample: int, events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    cfg = config()
    positive_tolerance = int(cfg["candidate"]["positive_tolerance_ms"]) * 16
    ambiguous_tolerance = int(cfg["candidate"]["ambiguous_tolerance_ms"]) * 16
    if not events:
        return {
            "label": "negative",
            "nearest_reference_sample": None,
            "nearest_reference_distance_ms": None,
            "nearest_reference_stratum": None,
        }
    nearest = min(events, key=lambda event: abs(int(event["sample"]) - sample))
    distance = abs(int(nearest["sample"]) - sample)
    if distance <= positive_tolerance:
        label = "positive"
    elif distance <= ambiguous_tolerance:
        label = "ambiguous"
    else:
        label = "negative"
    return {
        "label": label,
        "nearest_reference_sample": int(nearest["sample"]),
        "nearest_reference_distance_ms": distance // 16,
        "nearest_reference_stratum": str(nearest.get("stratum")),
    }


def extract_candidates(root: Path) -> Path:
    cfg = config()
    sessions = _sessions(root)
    directory = output_root(root)
    directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    r8_segment_mismatches: list[str] = []
    for session in sorted(sessions.values(), key=lambda value: (value.fold, value.session_id)):
        probabilities = _load_probabilities(root, session.session_id)
        starts = _segment_start_candidates(probabilities, int(cfg["frame_ms"]))
        segment_path = (
            r8_output_root(root) / "speaker_segments" / "cpu" / f"{session.session_id}.json"
        )
        if not segment_path.is_file():
            raise R9Error(f"R8 speaker segments are missing: {segment_path}")
        r8_segments = load_json(segment_path)
        r8_starts = sorted(int(segment["start_ms"]) * 16 for segment in r8_segments)
        candidate_samples = sorted(int(row["start_ms"]) * 16 for row in starts)
        if candidate_samples != r8_starts:
            r8_segment_mismatches.append(session.session_id)
        for row in starts:
            sample = int(row["start_ms"]) * 16
            label = _label_candidate(sample, session.events)
            rows.append(
                {
                    "session_id": session.session_id,
                    "fold": session.fold,
                    "sample": sample,
                    "frame": int(row["start_frame"]),
                    "speaker_slot": int(row["speaker_slot"]),
                    **label,
                }
            )
    rows.sort(key=lambda row: (int(row["fold"]), str(row["session_id"]), int(row["sample"])))
    for index, row in enumerate(rows):
        row["row"] = index
    label_counts = defaultdict(int)
    for row in rows:
        label_counts[str(row["label"])] += 1
    reference_count = sum(len(session.events) for session in sessions.values())
    positive_reference_samples = {
        (str(row["session_id"]), int(row["nearest_reference_sample"]))
        for row in rows
        if row["label"] == "positive"
    }
    covered_references = {
        (session.session_id, int(event["sample"]))
        for session in sessions.values()
        for event in session.events
        if (session.session_id, int(event["sample"])) in positive_reference_samples
    }
    document = {
        "schema_version": 1,
        "candidate_count": len(rows),
        "label_counts": dict(label_counts),
        "reference_count": reference_count,
        "covered_reference_count": len(covered_references),
        "covered_reference_fraction": (
            len(covered_references) / reference_count if reference_count else None
        ),
        "r8_segment_start_mismatch_sessions": r8_segment_mismatches,
        "candidates_per_source_hour": len(rows)
        / sum(session.scored_hours for session in sessions.values()),
        "code_sha256": sha256_file(CODE_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
    }
    if r8_segment_mismatches:
        raise R9Error(
            f"candidate stream diverges from the R8 model policy: {r8_segment_mismatches}"
        )
    path = directory / "candidates.jsonl"
    write_jsonl(path, rows)
    write_json(directory / "candidate_summary.json", document)
    print(json.dumps(document, indent=2, sort_keys=True))
    return path


def _dominant_slot(values: np.ndarray, lo: int, hi: int) -> int | None:
    if hi <= lo:
        return None
    means = values[lo:hi].mean(axis=0)
    return int(np.argmax(means))


def _features_for_candidate(
    probabilities: np.ndarray,
    frame: int,
    slot: int,
    confirmation_frames: int,
    windows: dict[str, Any],
    frame_ms: int,
) -> dict[str, float]:
    total = len(probabilities)
    active_any = (probabilities > 0.5).any(axis=1)
    inactive = ~active_any

    other_mask = np.ones(probabilities.shape[1], dtype=np.bool_)
    other_mask[slot] = False
    if bool((probabilities[frame, other_mask] > 0.5).any()):
        gap_frames = 0
    else:
        gap_frames = 0
        index = frame - 1
        while index >= 0 and bool(inactive[index]) and gap_frames < int(windows["gap_cap_frames"]):
            gap_frames += 1
            index -= 1
        if frame == 0:
            gap_frames = int(windows["gap_cap_frames"])
    gap_ms = float(gap_frames * frame_ms)

    pre_depth_frames = int(windows["pre_depth_frames"])
    pre_start = max(0, frame - gap_frames - pre_depth_frames)
    pre_end = frame - gap_frames
    if pre_end > 0 and pre_end > pre_start:
        previous_dominant = _dominant_slot(probabilities, pre_start, pre_end)
        pre_depth_min = float(probabilities[pre_start:pre_end, previous_dominant].min())
        pre_depth_mean = float(probabilities[pre_start:pre_end, previous_dominant].mean())
    else:
        pre_depth_min = 0.5
        pre_depth_mean = 0.5

    argmax_pre_frames = int(windows["argmax_pre_frames"])
    argmax_pre_lo = max(0, frame - argmax_pre_frames)
    pre_dominant = (
        _dominant_slot(probabilities, argmax_pre_lo, frame) if argmax_pre_lo < frame else None
    )
    post_hi = min(total, frame + confirmation_frames)
    post_dominant = _dominant_slot(probabilities, frame, post_hi)
    argmax_switch = (
        1.0
        if (
            pre_dominant is not None and post_dominant is not None and pre_dominant != post_dominant
        )
        else 0.0
    )

    last = min(total - 1, frame + confirmation_frames - 1)
    cross_probability = float(probabilities[frame, slot])
    peak_probability = float(probabilities[frame : last + 1, slot].max())
    rise_slope = float(probabilities[last, slot] - probabilities[frame, slot])

    run_frames = 0
    run_index = frame
    while (
        run_index < total
        and run_index < frame + confirmation_frames
        and bool(probabilities[run_index, slot] > 0.5)
    ):
        run_frames += 1
        run_index += 1
    persistence_ms = float(run_frames * frame_ms)

    resume_frames = int(windows["resume_window_frames"])
    resume_lo = max(0, frame - resume_frames)
    same_slot_resume = (
        1.0
        if bool((probabilities[resume_lo:frame, slot] > 0.5).any()) and resume_lo < frame
        else 0.0
    )

    co_activity_max = (
        float(probabilities[frame : last + 1, other_mask].max()) if last >= frame else 0.0
    )

    pre_lo = max(0, frame - argmax_pre_frames)
    pre_active_slots = {
        other
        for other in range(probabilities.shape[1])
        if other != slot
        and bool((probabilities[pre_lo:frame, other] > 0.5).any())
        and pre_lo < frame
    }
    return_hi = min(total, frame + int(windows["return_window_frames"]))
    return_flag = 0.0
    if pre_active_slots and frame + 1 < return_hi:
        for other in pre_active_slots:
            if bool((probabilities[frame + 1 : return_hi, other] > 0.5).any()):
                return_flag = 1.0
                break

    return {
        "argmax_switch": argmax_switch,
        "gap_ms": gap_ms,
        "pre_depth_min": pre_depth_min,
        "pre_depth_mean": pre_depth_mean,
        "rise_slope": rise_slope,
        "cross_probability": cross_probability,
        "peak_probability": peak_probability,
        "persistence_ms": persistence_ms,
        "same_slot_resume": same_slot_resume,
        "co_activity_max": co_activity_max,
        "return_flag": return_flag,
    }


def extract_features(root: Path) -> list[Path]:
    cfg = config()
    sessions = _sessions(root)
    directory = output_root(root)
    candidates = read_jsonl(directory / "candidates.jsonl")
    if not candidates:
        raise R9Error("candidates must run before features")
    probabilities: dict[str, np.ndarray] = {
        session_id: _load_probabilities(root, session_id) for session_id in sessions
    }
    windows = cfg["feature_windows"]
    frame_ms = int(cfg["frame_ms"])
    outputs: list[Path] = []
    for name, confirmation_frames in (
        ("features_a", int(cfg["confirmation"]["base_frames"])),
        ("features_a_diagnostic", int(cfg["confirmation"]["diagnostic_frames"])),
    ):
        rows: list[dict[str, Any]] = []
        for candidate in candidates:
            features = _features_for_candidate(
                probabilities[str(candidate["session_id"])],
                int(candidate["frame"]),
                int(candidate["speaker_slot"]) - 1,
                confirmation_frames,
                windows,
                frame_ms,
            )
            rows.append(
                {
                    "row": int(candidate["row"]),
                    "session_id": str(candidate["session_id"]),
                    "sample": int(candidate["sample"]),
                    "label": str(candidate["label"]),
                    "confirmation_frames": confirmation_frames,
                    "features": features,
                }
            )
        path = directory / f"{name}.jsonl"
        write_jsonl(path, rows)
        outputs.append(path)
        print(f"{name}: {len(rows)} rows", flush=True)
    return outputs


def _one_to_one(
    predictions: Sequence[int], references: Sequence[int], tolerance: int
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    prediction_values = sorted(int(value) for value in predictions)
    reference_values = sorted(int(value) for value in references)
    matched: list[tuple[int, int]] = []
    false: list[int] = []
    misses: list[int] = []
    prediction_index = 0
    reference_index = 0
    while prediction_index < len(prediction_values) and reference_index < len(reference_values):
        prediction = prediction_values[prediction_index]
        reference = reference_values[reference_index]
        if prediction < reference - tolerance:
            false.append(prediction)
            prediction_index += 1
        elif reference < prediction - tolerance:
            misses.append(reference)
            reference_index += 1
        else:
            matched.append((prediction, reference))
            prediction_index += 1
            reference_index += 1
    false.extend(prediction_values[prediction_index:])
    misses.extend(reference_values[reference_index:])
    return matched, false, misses


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p90": None, "p95": None, "p99": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "maximum": float(np.max(array)),
    }


def _metrics(
    sessions: dict[str, Session],
    predictions: dict[str, list[int]],
    selected_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    session_ids = sorted(selected_ids if selected_ids is not None else sessions)
    exposure_hours = sum(sessions[session_id].scored_hours for session_id in session_ids)
    if exposure_hours <= 0.0:
        raise R9Error("scored exposure must be positive")
    reference_count = sum(len(sessions[session_id].events) for session_id in session_ids)
    scored_predictions = {
        session_id: [
            sample
            for sample in predictions.get(session_id, [])
            if sessions[session_id].first_boundary
            <= sample
            < sessions[session_id].last_boundary + 1600
        ]
        for session_id in session_ids
    }
    result: dict[str, Any] = {
        "exposure_hours": exposure_hours,
        "reference_count": reference_count,
        "prediction_count": sum(len(scored_predictions[session_id]) for session_id in session_ids),
        "tolerances": {},
    }
    primary_matches: list[tuple[str, int, int]] = []
    primary_false: list[tuple[str, int]] = []
    primary_misses: list[tuple[str, int]] = []
    primary_per_meeting: dict[str, Any] = {}
    for tolerance_ms in (100, 250, 500):
        matched_all: list[tuple[str, int, int]] = []
        false_all: list[tuple[str, int]] = []
        misses_all: list[tuple[str, int]] = []
        per_meeting: dict[str, Any] = {}
        for session_id in session_ids:
            session = sessions[session_id]
            session_predictions = scored_predictions[session_id]
            references = [int(event["sample"]) for event in session.events]
            matched, false, misses = _one_to_one(session_predictions, references, tolerance_ms * 16)
            matched_all.extend(
                (session_id, prediction, reference) for prediction, reference in matched
            )
            false_all.extend((session_id, prediction) for prediction in false)
            misses_all.extend((session_id, reference) for reference in misses)
            per_meeting[session_id] = {
                "reference_count": len(references),
                "prediction_count": len(session_predictions),
                "true_positive_count": len(matched),
                "false_event_count": len(false),
                "miss_count": len(misses),
                "recall": _ratio(len(matched), len(references)),
                "false_events_per_hour": len(false) / session.scored_hours,
            }
        true_positive_count = len(matched_all)
        prediction_count = sum(len(scored_predictions[session_id]) for session_id in session_ids)
        precision = _ratio(true_positive_count, prediction_count)
        recall = _ratio(true_positive_count, reference_count)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0.0
            else None
        )
        result["tolerances"][str(tolerance_ms)] = {
            "true_positive_count": true_positive_count,
            "false_event_count": len(false_all),
            "miss_count": len(misses_all),
            "false_events_per_hour": len(false_all) / exposure_hours,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if tolerance_ms == 250:
            primary_matches = matched_all
            primary_false = false_all
            primary_misses = misses_all
            primary_per_meeting = per_meeting
    event_lookup = {
        session_id: {int(event["sample"]): event for event in sessions[session_id].events}
        for session_id in session_ids
    }
    references_by_stratum: dict[str, int] = defaultdict(int)
    matched_by_stratum: dict[str, int] = defaultdict(int)
    short_reference_count = 0
    short_matched_count = 0
    for session_id in session_ids:
        for event in sessions[session_id].events:
            references_by_stratum[str(event["stratum"])] += 1
            if bool(event.get("short_backchannel_or_return")):
                short_reference_count += 1
    for session_id, _, reference in primary_matches:
        event = event_lookup[session_id][reference]
        matched_by_stratum[str(event["stratum"])] += 1
        if bool(event.get("short_backchannel_or_return")):
            short_matched_count += 1
    meeting_true_positives = [
        int(row["true_positive_count"]) for row in primary_per_meeting.values()
    ]
    result.update(
        {
            "per_meeting": primary_per_meeting,
            "matched_pairs": primary_matches,
            "false_event_samples": primary_false,
            "miss_samples": primary_misses,
            "stratum_recall": {
                name: _ratio(matched_by_stratum[name], count)
                for name, count in sorted(references_by_stratum.items())
            },
            "short_return_recall": _ratio(short_matched_count, short_reference_count),
            "maximum_meeting_true_positive_share": (
                max(meeting_true_positives) / max(sum(meeting_true_positives), 1)
            ),
        }
    )
    return result


def _grouped_events(
    candidates: Sequence[dict[str, Any]], radius_samples: int
) -> list[tuple[int, float]]:
    ordered = sorted(
        candidates,
        key=lambda row: (int(row["sample"]), float(row["score"]), int(row.get("frame", 0))),
    )
    components: list[list[dict[str, Any]]] = []
    for row in ordered:
        if (
            not components
            or int(row["sample"]) - int(components[-1][-1]["sample"]) > radius_samples
        ):
            components.append([row])
        else:
            components[-1].append(row)
    result: list[tuple[int, float]] = []
    for component in components:
        best = max(
            component,
            key=lambda row: (float(row["score"]), -int(row["sample"])),
        )
        result.append((int(best["sample"]), float(best["score"])))
    return sorted(result, key=lambda pair: pair[0])


class ScoreEvaluationCache:
    def __init__(
        self,
        sessions: dict[str, Session],
        grouped: dict[str, list[tuple[int, float]]],
    ) -> None:
        self.sessions = sessions
        self.grouped = grouped
        self.predictions: dict[float, dict[str, list[int]]] = {}

    def events(self, threshold: float) -> dict[str, list[int]]:
        key = float(np.float32(threshold))
        if key not in self.predictions:
            self.predictions[key] = {
                session_id: [
                    sample
                    for sample, score in self.grouped.get(session_id, [])
                    if float(score) >= key
                ]
                for session_id in self.sessions
            }
        return self.predictions[key]

    def metrics(
        self, threshold: float, selected_ids: Iterable[str] | None = None
    ) -> dict[str, Any]:
        return _metrics(self.sessions, self.events(threshold), selected_ids)


def _curve_row(
    cache: ScoreEvaluationCache, threshold: float, selected_ids: Iterable[str] | None = None
) -> dict[str, Any]:
    metrics = cache.metrics(threshold, selected_ids)
    primary = metrics["tolerances"]["250"]
    return {
        "threshold": float(np.float32(threshold)),
        "prediction_count": metrics["prediction_count"],
        "true_positive_count": primary["true_positive_count"],
        "false_event_count": primary["false_event_count"],
        "false_events_per_hour": primary["false_events_per_hour"],
        "recall_250": primary["recall"],
    }


def _select_row(rows: Sequence[dict[str, Any]], target: float) -> dict[str, Any]:
    eligible = [row for row in rows if float(row["false_events_per_hour"]) <= target]
    if not eligible:
        return min(rows, key=lambda row: float(row["false_events_per_hour"]))
    return max(
        eligible,
        key=lambda row: (
            float(row["recall_250"] or 0.0),
            -float(row["false_events_per_hour"]),
            -float(row["threshold"]),
        ),
    )


def _refined_thresholds(
    cache: ScoreEvaluationCache,
    score_values: np.ndarray,
    dense_rows: Sequence[dict[str, Any]],
    target: float,
    search: dict[str, Any],
) -> list[float]:
    selected = _select_row(dense_rows, target)
    dense_values = sorted({float(row["threshold"]) for row in dense_rows})
    position = dense_values.index(float(selected["threshold"]))
    lower = dense_values[max(0, position - 1)]
    upper = dense_values[min(len(dense_values) - 1, position + 1)]
    unique = np.unique(score_values[(score_values >= lower) & (score_values <= upper)]).astype(
        np.float32
    )
    if len(unique) == 0:
        return [float(selected["threshold"])]
    refinement_points = int(search["refinement_points"])
    rounds = int(search["refinement_rounds"])
    evaluated: set[float] = {float(selected["threshold"])}
    current = unique
    for _ in range(rounds):
        if len(current) <= int(search["maximum_exact_values_per_target"]):
            break
        indices = np.linspace(0, len(current) - 1, refinement_points, dtype=np.int64)
        probe_values = [float(current[int(index)]) for index in indices]
        probe_rows = [_curve_row(cache, value) for value in probe_values]
        evaluated.update(probe_values)
        best = _select_row(probe_rows, target)
        best_index = probe_values.index(float(best["threshold"]))
        probe_lower = probe_values[max(0, best_index - 1)]
        probe_upper = probe_values[min(len(probe_values) - 1, best_index + 1)]
        current = current[(current >= probe_lower) & (current <= probe_upper)]
    if len(current) > int(search["maximum_exact_values_per_target"]):
        raise R9Error(
            f"exact threshold bracket remains too large for target {target}: {len(current)}"
        )
    evaluated.update(float(value) for value in current)
    return sorted(evaluated)


def _standardize_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64)
    scale = values.std(axis=0, dtype=np.float64)
    scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _standardize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.asarray((values - mean) / scale, dtype=np.float32)


def _balance_weights(labels: np.ndarray) -> np.ndarray:
    result = np.ones(len(labels), dtype=np.float64)
    positive = labels == 1
    negative = labels == 0
    positive_total = float(positive.sum())
    negative_total = float(negative.sum())
    if positive_total <= 0 or negative_total <= 0:
        raise R9Error("both training classes are required")
    result[positive] *= 0.5 / positive_total
    result[negative] *= 0.5 / negative_total
    result *= len(result)
    return result.astype(np.float32)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_linear(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, cfg: dict[str, Any]
) -> tuple[np.ndarray, float]:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        C=float(cfg["linear"]["c"]),
        max_iter=int(cfg["linear"]["max_iter"]),
        solver="lbfgs",
        random_state=int(cfg["linear"]["random_state"]),
    )
    model.fit(x, y, sample_weight=weights)
    return np.asarray(model.coef_[0], dtype=np.float32), float(model.intercept_[0])


def _linear_predict(x: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    return _sigmoid(np.asarray(x @ coef + intercept, dtype=np.float64))


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    mask = labels != -1
    if not mask.any() or len(set(labels[mask])) < 2:
        return float("nan")
    return float(roc_auc_score(labels[mask], scores[mask]))


def _mlp_module(input_dim: int, hidden: int):
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1),
    )


def _fit_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weights: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> tuple[dict[str, Any], int]:
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    spec = cfg["mlp"]
    model = _mlp_module(train_x.shape[1], int(spec["hidden_width"]))
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
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(int(spec["maximum_epochs"])):
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
            best_state = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }
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

    model = _mlp_module(x.shape[1], int(cfg["mlp"]["hidden_width"]))
    model.load_state_dict(state)
    model.eval()
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            logits = model(torch.from_numpy(x[start : start + batch_size])).squeeze(1)
            values.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def _feature_matrix(
    rows: Sequence[dict[str, Any]], names: Sequence[str] | None = None
) -> np.ndarray:
    if names is None:
        names = list(config()["feature_names"])
    names = list(names)
    return np.asarray(
        [[float(row["features"][name]) for name in names] for row in rows],
        dtype=np.float32,
    )


def _label_vector(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            1 if row["label"] == "positive" else 0 if row["label"] == "negative" else -1
            for row in rows
        ],
        dtype=np.int8,
    )


def _curve_for_scores(
    cache: ScoreEvaluationCache,
    score_values: np.ndarray,
    targets: Sequence[float],
    search: dict[str, Any],
) -> list[dict[str, Any]]:
    dense_thresholds = np.arange(
        float(search["dense_min"]),
        float(search["dense_max"]) + float(search["dense_step"]) / 2.0,
        float(search["dense_step"]),
        dtype=np.float32,
    )
    rows_by_threshold: dict[float, dict[str, Any]] = {}
    for threshold in dense_thresholds:
        row = _curve_row(cache, float(threshold))
        rows_by_threshold[float(row["threshold"])] = row
    dense_rows = list(rows_by_threshold.values())
    for target in targets:
        for threshold in _refined_thresholds(
            cache, score_values, dense_rows, float(target), search
        ):
            key = float(np.float32(threshold))
            if key not in rows_by_threshold:
                rows_by_threshold[key] = _curve_row(cache, key)
    return sorted(rows_by_threshold.values(), key=lambda row: -float(row["threshold"]))


def run_a0(root: Path) -> Path:
    cfg = config()
    sessions = _sessions(root)
    directory = output_root(root)
    candidates = read_jsonl(directory / "candidates.jsonl")
    features_rows = read_jsonl(directory / "features_a.jsonl")
    if len(candidates) != len(features_rows):
        raise R9Error("candidate and feature row counts differ")
    features_by_row = {int(row["row"]): row["features"] for row in features_rows}
    radius_samples = int(cfg["duplicate_suppression_ms"]) * 16
    per_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        features = features_by_row[int(candidate["row"])]
        per_session[str(candidate["session_id"])].append(
            {
                "sample": int(candidate["sample"]),
                "peak_probability": float(features["peak_probability"]),
                "persistence_ms": float(features["persistence_ms"]),
                "same_slot_resume": float(features["same_slot_resume"]),
                "gap_ms": float(features["gap_ms"]),
                "co_activity_max": float(features["co_activity_max"]),
            }
        )
    for rows in per_session.values():
        rows.sort(key=lambda row: int(row["sample"]))

    def passes(
        row: dict[str, Any],
        peak: float,
        persistence: float,
        gap_max: float,
        co_activity_min: float,
        exclude_resume: bool,
    ) -> bool:
        if exclude_resume and float(row["same_slot_resume"]) == 1.0:
            return False
        if float(row["peak_probability"]) < peak:
            return False
        if float(row["persistence_ms"]) < persistence:
            return False
        if float(row["gap_ms"]) > gap_max:
            return False
        if float(row["co_activity_max"]) < co_activity_min:
            return False
        return True

    def predictions_for(
        session_ids: Sequence[str],
        peak: float,
        persistence: float,
        gap_max: float,
        co_activity_min: float,
    ) -> dict[str, list[int]]:
        result: dict[str, list[int]] = {}
        for session_id in session_ids:
            passing = [
                {**row, "score": float(row["peak_probability"])}
                for row in per_session[session_id]
                if passes(row, peak, persistence, gap_max, co_activity_min, exclude_resume)
            ]
            result[session_id] = [sample for sample, _ in _grouped_events(passing, radius_samples)]
        return result

    peak_grid = [float(value) for value in cfg["a0"]["peak_probability_grid"]]
    persistence_grid = [float(value) for value in cfg["a0"]["persistence_ms_grid"]]
    gap_grid = [float(value) for value in cfg["a0"]["gap_ms_max_grid"]]
    co_activity_grid = [float(value) for value in cfg["a0"]["co_activity_min_grid"]]
    budget = float(cfg["a0"]["dev_false_event_budget_per_hour"])
    exclude_resume = bool(cfg["a0"]["exclude_same_slot_resume"])
    folded: dict[str, list[int]] = defaultdict(list)
    fold_selection: dict[str, Any] = {}
    for fold, held_out in enumerate(cfg["folds"]):
        held_out_ids = [str(value) for value in held_out]
        development_ids = [session_id for session_id in sessions if session_id not in held_out_ids]
        best: tuple[Any, ...] | None = None
        best_metrics: dict[str, Any] | None = None
        best_peak = peak_grid[0]
        best_persistence = persistence_grid[0]
        best_gap = gap_grid[0]
        best_co_activity = co_activity_grid[0]
        for peak in peak_grid:
            for persistence in persistence_grid:
                for gap_max in gap_grid:
                    for co_activity_min in co_activity_grid:
                        metrics = _metrics(
                            sessions,
                            predictions_for(
                                development_ids,
                                peak,
                                persistence,
                                gap_max,
                                co_activity_min,
                            ),
                            development_ids,
                        )
                        primary = metrics["tolerances"]["250"]
                        feh = float(primary["false_events_per_hour"])
                        recall = float(primary["recall"] or 0.0)
                        if feh <= budget:
                            candidate_key = (
                                recall,
                                -feh,
                                -peak,
                                -persistence,
                                gap_max,
                                -co_activity_min,
                            )
                        else:
                            candidate_key = (
                                -1.0,
                                -feh,
                                -peak,
                                -persistence,
                                gap_max,
                                -co_activity_min,
                            )
                        if best is None or candidate_key > best:
                            best = candidate_key
                            best_metrics = metrics
                            best_peak = peak
                            best_persistence = persistence
                            best_gap = gap_max
                            best_co_activity = co_activity_min
        held_predictions = predictions_for(
            held_out_ids,
            float(best_peak),
            float(best_persistence),
            float(best_gap),
            float(best_co_activity),
        )
        for session_id in held_out_ids:
            folded[session_id] = held_predictions[session_id]
        held_metrics = _metrics(sessions, held_predictions, held_out_ids)
        fold_selection[str(fold)] = {
            "held_out_sessions": held_out_ids,
            "selected_peak_probability_threshold": float(best_peak),
            "selected_persistence_ms_threshold": float(best_persistence),
            "selected_gap_ms_max": float(best_gap),
            "selected_co_activity_min": float(best_co_activity),
            "development_metrics": best_metrics,
            "held_out_metrics": held_metrics,
        }
    aggregate = _metrics(sessions, dict(folded))
    document = {
        "schema_version": 1,
        "fold_selection": fold_selection,
        "aggregate_metrics": aggregate,
        "aggregate_recall_250": aggregate["tolerances"]["250"]["recall"],
        "aggregate_false_events_per_hour": aggregate["tolerances"]["250"]["false_events_per_hour"],
        "code_sha256": sha256_file(CODE_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
    }
    path = directory / "a0_metrics.json"
    write_json(path, document)
    return path


def _verifier_arm(
    root: Path,
    *,
    arm: str,
    diagnostic: bool,
    feature_file: str,
    feature_names: Sequence[str],
    output_name: str,
    scores_name: str,
    transfer_name: str,
    mlp_trigger_source: str | None = None,
    fold_matrix_transform: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    availability_frames: int | None = None,
    dependencies: dict[str, str] | None = None,
) -> Path:
    cfg = config()
    sessions = _sessions(root)
    directory = output_root(root)
    rows = read_jsonl(directory / f"{feature_file}.jsonl")
    matrix = _feature_matrix(rows, feature_names)
    labels = _label_vector(rows)
    row_session = [str(row["session_id"]) for row in rows]
    radius_samples = int(cfg["duplicate_suppression_ms"]) * 16

    linear_scores = np.full(len(rows), np.nan, dtype=np.float64)
    fold_auroc: dict[str, Any] = {}
    predict_seconds = 0.0
    for fold, held_out in enumerate(cfg["folds"]):
        held_out_ids = [str(value) for value in held_out]
        train_mask = np.asarray(
            [session_id not in held_out_ids for session_id in row_session], dtype=np.bool_
        )
        held_mask = ~train_mask
        fold_matrix = (
            fold_matrix_transform(matrix, train_mask)
            if fold_matrix_transform is not None
            else matrix
        )
        mean, scale = _standardize_fit(fold_matrix[train_mask])
        standardized = _standardize(fold_matrix, mean, scale)
        usable = labels != -1
        train_usable = train_mask & usable
        weights = _balance_weights(labels[train_usable])
        coef, intercept = _fit_linear(
            standardized[train_usable], labels[train_usable], weights, cfg["verifier"]
        )
        started = time.perf_counter()
        fold_scores = _linear_predict(standardized, coef, intercept)
        predict_seconds += time.perf_counter() - started
        linear_scores[held_mask] = fold_scores[held_mask]
        fold_auroc[str(fold)] = _auroc(fold_scores[held_mask], labels[held_mask])

    finite_aurocs = [value for value in fold_auroc.values() if not math.isnan(value)]
    if not finite_aurocs:
        raise R9Error("no fold produced a finite out-of-fold AUROC")
    mean_auroc = float(np.mean(finite_aurocs))
    trigger = float(cfg["verifier"]["mlp"]["fallback_auroc_trigger"])
    if mlp_trigger_source is not None:
        source_path = directory / mlp_trigger_source
        if not source_path.is_file():
            raise R9Error(f"{mlp_trigger_source} must exist before {arm}")
        mlp_triggered = bool(load_json(source_path).get("mlp_triggered", False))
    elif diagnostic:
        a1_path = directory / "a1_metrics.json"
        if not a1_path.is_file():
            raise R9Error("a1 must run before a1-diagnostic")
        mlp_triggered = bool(load_json(a1_path).get("mlp_triggered", False))
    else:
        mlp_triggered = mean_auroc < trigger
    mlp_scores: np.ndarray | None = None
    if mlp_triggered:
        mlp_scores = np.full(len(rows), np.nan, dtype=np.float64)
        for fold, held_out in enumerate(cfg["folds"]):
            held_out_ids = [str(value) for value in held_out]
            train_mask = np.asarray(
                [session_id not in held_out_ids for session_id in row_session], dtype=np.bool_
            )
            held_mask = ~train_mask
            fold_matrix = (
                fold_matrix_transform(matrix, train_mask)
                if fold_matrix_transform is not None
                else matrix
            )
            mean, scale = _standardize_fit(fold_matrix[train_mask])
            standardized = _standardize(fold_matrix, mean, scale)
            usable = labels != -1
            inner_val_ids = [
                str(value)
                for value in cfg["folds"][
                    (fold + int(cfg["verifier"]["mlp"]["inner_validation_fold_offset"]))
                    % len(cfg["folds"])
                ]
            ]
            inner_val_mask = (
                np.asarray(
                    [session_id in inner_val_ids for session_id in row_session], dtype=np.bool_
                )
                & train_mask
                & usable
            )
            inner_train_mask = train_mask & usable & ~inner_val_mask
            validation = (
                standardized[inner_val_mask],
                labels[inner_val_mask],
                _balance_weights(labels[inner_val_mask]),
            )
            seed_scores: list[np.ndarray] = []
            for seed in cfg["verifier"]["mlp"]["seeds"]:
                state, _ = _fit_mlp(
                    standardized[inner_train_mask],
                    labels[inner_train_mask],
                    _balance_weights(labels[inner_train_mask]),
                    cfg["verifier"],
                    int(seed),
                    validation,
                )
                started = time.perf_counter()
                seed_scores.append(_mlp_predict(standardized[held_mask], state, cfg["verifier"]))
                predict_seconds += time.perf_counter() - started
            mlp_scores[held_mask] = np.mean(seed_scores, axis=0)
    scores = mlp_scores if mlp_triggered else linear_scores
    if not np.isfinite(scores).all():
        raise R9Error("verifier scores contain non-finite values")

    grouped: dict[str, list[tuple[int, float]]] = {}
    grouped_selection: dict[str, list[tuple[int, float]]] = {}
    for index, row in enumerate(rows):
        session_id = str(row["session_id"])
        pair = (int(row["sample"]), float(scores[index]))
        grouped.setdefault(session_id, []).append(pair)
        if str(row["label"]) != "ambiguous":
            grouped_selection.setdefault(session_id, []).append(pair)
    for session_id in sessions:
        grouped.setdefault(session_id, [])
        grouped_selection.setdefault(session_id, [])
    for session_id in grouped:
        grouped[session_id] = _grouped_events(
            [{"sample": sample, "score": score} for sample, score in grouped[session_id]],
            radius_samples,
        )
        grouped_selection[session_id] = _grouped_events(
            [{"sample": sample, "score": score} for sample, score in grouped_selection[session_id]],
            radius_samples,
        )
    selection_cache = ScoreEvaluationCache(sessions, grouped_selection)
    all_cache = ScoreEvaluationCache(sessions, grouped)
    score_values = np.asarray(
        [score for pairs in grouped_selection.values() for _, score in pairs],
        dtype=np.float32,
    )
    targets = [float(value) for value in cfg["targets"]["false_events_per_hour"]]
    curve = _curve_for_scores(selection_cache, score_values, targets, cfg["curve_search"])
    evaluation_curve: list[dict[str, Any]] = []
    for row in curve:
        threshold = float(row["threshold"])
        metrics = all_cache.metrics(threshold)
        primary = metrics["tolerances"]["250"]
        evaluation_curve.append(
            {
                "threshold": threshold,
                "prediction_count": metrics["prediction_count"],
                "true_positive_count": primary["true_positive_count"],
                "false_event_count": primary["false_event_count"],
                "false_events_per_hour": primary["false_events_per_hour"],
                "recall_250": primary["recall"],
            }
        )
    selected_points: dict[str, Any] = {}
    for target in targets:
        selected = _select_row(curve, target)
        selected_points[str(target)] = {
            "threshold": selected["threshold"],
            "selection_false_events_per_hour": selected["false_events_per_hour"],
            "selection_recall_250": selected["recall_250"],
            "metrics": all_cache.metrics(float(selected["threshold"])),
        }
    transfer: list[dict[str, Any]] = []
    for fold, held_out in enumerate(cfg["folds"]):
        held_out_ids = [str(value) for value in held_out]
        development_ids = [session_id for session_id in sessions if session_id not in held_out_ids]
        development_rows = [
            _curve_row(selection_cache, float(row["threshold"]), development_ids) for row in curve
        ]
        for target in targets:
            selected = _select_row(development_rows, target)
            held_metrics = all_cache.metrics(float(selected["threshold"]), held_out_ids)
            primary = held_metrics["tolerances"]["250"]
            transfer.append(
                {
                    "fold": fold,
                    "held_out_sessions": held_out_ids,
                    "target_false_events_per_hour": target,
                    "selected_threshold": selected["threshold"],
                    "development_false_events_per_hour": selected["false_events_per_hour"],
                    "development_recall_250": selected["recall_250"],
                    "held_out_false_events_per_hour": primary["false_events_per_hour"],
                    "held_out_recall_250": primary["recall"],
                }
            )
    per_meeting_curves: dict[str, list[dict[str, Any]]] = {
        session_id: [] for session_id in sessions
    }
    for row in curve:
        threshold = float(row["threshold"])
        metrics = all_cache.metrics(threshold)
        for session_id, meeting in metrics["per_meeting"].items():
            per_meeting_curves[session_id].append(
                {
                    "threshold": threshold,
                    "prediction_count": meeting["prediction_count"],
                    "true_positive_count": meeting["true_positive_count"],
                    "false_event_count": meeting["false_event_count"],
                    "false_events_per_hour": meeting["false_events_per_hour"],
                    "recall_250": meeting["recall"],
                }
            )
    gate_cfg = cfg["reference_gate"]
    transfer_check_ok = all(
        float(row["held_out_false_events_per_hour"])
        <= float(row["target_false_events_per_hour"])
        * float(gate_cfg["maximum_transfer_false_event_multiplier"])
        for row in transfer
        if int(row["target_false_events_per_hour"]) in {10, 20}
    )
    reference_gate: dict[str, Any] = {"targets": {}}
    for target in (10.0, 20.0):
        point = selected_points[str(target)]["metrics"]
        primary = point["tolerances"]["250"]
        reference_gate["targets"][str(target)] = {
            "recall_250": primary["recall"],
            "false_events_per_hour": primary["false_events_per_hour"],
            "stratum_recall": point["stratum_recall"],
            "maximum_meeting_true_positive_share": point["maximum_meeting_true_positive_share"],
        }
    reference_gate["checks"] = {
        "recall_at_10_false_events_per_hour": float(
            reference_gate["targets"]["10.0"]["recall_250"] or 0.0
        )
        >= float(gate_cfg["recall_at_10_false_events_per_hour"]),
        "recall_at_20_false_events_per_hour": float(
            reference_gate["targets"]["20.0"]["recall_250"] or 0.0
        )
        >= float(gate_cfg["recall_at_20_false_events_per_hour"]),
        "overlap_onset_nonzero": float(
            reference_gate["targets"]["20.0"]["stratum_recall"].get("overlap_onset") or 0.0
        )
        > 0.0,
        "silence_gap_change_nonzero": float(
            reference_gate["targets"]["20.0"]["stratum_recall"].get("silence_gap_change") or 0.0
        )
        > 0.0,
        "meeting_concentration": float(
            reference_gate["targets"]["20.0"]["maximum_meeting_true_positive_share"]
        )
        <= float(gate_cfg["maximum_single_meeting_true_positive_share"]),
        "threshold_transfer": transfer_check_ok,
    }
    confirmation_frames = int(
        cfg["confirmation"]["diagnostic_frames" if diagnostic else "base_frames"]
    )
    availability_frames = (
        confirmation_frames if availability_frames is None else int(availability_frames)
    )
    confirmation_ms = confirmation_frames * int(cfg["frame_ms"])
    availability_frame_horizon_ms = availability_frames * int(cfg["frame_ms"])
    defect_threshold_ms = float(cfg["reference_gate"]["latency_defect_median_ms"])
    verification_compute_ms = max(predict_seconds / len(rows) * 1000.0, 0.0)
    availability_latency: dict[str, Any] = {
        "confirmation_ms": confirmation_ms,
        "availability_frame_horizon_ms": availability_frame_horizon_ms,
        "verification_compute_ms_per_event": verification_compute_ms,
        "latency_defect_median_ms": defect_threshold_ms,
        "per_target": {},
    }
    for target, point in selected_points.items():
        deltas = [
            (int(prediction) - int(reference)) // 16
            for _, prediction, reference in point["metrics"]["matched_pairs"]
        ]
        availability_values = [
            float(value) + availability_frame_horizon_ms + verification_compute_ms
            for value in deltas
        ]
        availability_percentiles = _percentiles(availability_values)
        availability_latency["per_target"][str(target)] = {
            "boundary_lag_ms": _percentiles([float(value) for value in deltas]),
            "availability_ms": availability_percentiles,
            "latency_defect": availability_percentiles["p50"] is not None
            and float(availability_percentiles["p50"]) >= defect_threshold_ms,
        }
    document = {
        "schema_version": 1,
        "arm": arm,
        "diagnostic": diagnostic,
        "feature_name": feature_file,
        "feature_names": list(feature_names),
        "verifier_form": "mlp" if mlp_triggered else "logistic",
        "mlp_triggered": mlp_triggered,
        "mean_out_of_fold_auroc": mean_auroc,
        "fold_auroc": fold_auroc,
        "threshold_selection_excluded_ambiguous": True,
        "curve": curve,
        "curve_kind": "selection_excluding_ambiguous",
        "evaluation_curve": evaluation_curve,
        "evaluation_curve_kind": "event_level_including_ambiguous",
        "selected_operating_points": selected_points,
        "threshold_transfer": transfer,
        "per_meeting_curves": per_meeting_curves,
        "reference_gate": reference_gate,
        "availability_latency": availability_latency,
        "score_count": len(rows),
        "code_sha256": sha256_file(CODE_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
    }
    path = directory / f"{output_name}.json"
    write_json(path, document)
    transfer_path = directory / transfer_name
    write_json(transfer_path, transfer)
    scores_path = directory / f"scores_{scores_name}.jsonl"
    write_jsonl(
        scores_path,
        [
            {
                "row": int(row["row"]),
                "session_id": str(row["session_id"]),
                "sample": int(row["sample"]),
                "label": str(row["label"]),
                "score": float(scores[int(row["row"])]),
            }
            for row in rows
        ],
    )
    receipt = {
        "schema_version": 1,
        "arm": arm,
        "metrics_sha256": sha256_file(path),
        "feature_file": feature_file,
        "feature_file_sha256": sha256_file(directory / f"{feature_file}.jsonl"),
        "scores_file": scores_path.name,
        "scores_sha256": sha256_file(scores_path),
        "dependencies": dependencies or {},
        "code_sha256": sha256_file(CODE_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
    }
    write_json(directory / f"{output_name}_receipt.json", receipt)
    return path


def run_a1(root: Path, diagnostic: bool) -> Path:
    cfg = config()
    feature_names = list(cfg["feature_names"])
    directory = output_root(root)
    if diagnostic:
        return _verifier_arm(
            root,
            arm="a1-diagnostic",
            diagnostic=True,
            feature_file="features_a_diagnostic",
            feature_names=feature_names,
            output_name="a1_diagnostic_metrics",
            scores_name="a1_diagnostic",
            transfer_name="threshold_transfer_diagnostic_metrics.json",
            dependencies={
                "features_a_diagnostic.jsonl": sha256_file(
                    directory / "features_a_diagnostic.jsonl"
                )
            },
        )
    return _verifier_arm(
        root,
        arm="a1",
        diagnostic=False,
        feature_file="features_a",
        feature_names=feature_names,
        output_name="a1_metrics",
        scores_name="a1",
        transfer_name="threshold_transfer_metrics.json",
        dependencies={
            "candidates.jsonl": sha256_file(directory / "candidates.jsonl"),
            "features_a.jsonl": sha256_file(directory / "features_a.jsonl"),
        },
    )


EMBEDDING_DUMP_HEADER = struct.Struct("<qqiiii")
EMBEDDING_DIM = 512
EMBEDDING_SPEAKERS = 4
EMBEDDING_PINNED_MODEL_SHA256 = "a5dacdc650790266c7a362e54e6bf51952015487edaa606c4e11632bc32442a9"
EMBEDDING_ROW_BYTES = 4 + EMBEDDING_DIM * 4 + EMBEDDING_SPEAKERS * 4
EMBEDDING_EXTERNAL_REPOSITORY = "https://github.com/handy-computer/transcribe.cpp.git"
EMBEDDING_EXTERNAL_COMMIT = "d42c3bbdfa2f63c37e5891e27de47a612d62f221"


def _embedding_config(cfg: dict[str, Any]) -> dict[str, Any]:
    embedding = cfg.get("embedding")
    if not embedding or not embedding.get("enabled"):
        raise R9Error("embedding section is not enabled in the R9 config")
    return embedding


def _parse_dump_records(
    path: Path,
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        while True:
            header_bytes = handle.read(EMBEDDING_DUMP_HEADER.size)
            if not header_bytes:
                break
            if len(header_bytes) < EMBEDDING_DUMP_HEADER.size:
                raise R9Error(f"truncated embedding dump header: {path}")
            chunk, total_n, n_cache, n_fifo, compression, compress_count = (
                EMBEDDING_DUMP_HEADER.unpack(header_bytes)
            )
            row_count = int(n_cache) + int(n_fifo)
            row_bytes = handle.read(row_count * EMBEDDING_ROW_BYTES)
            if len(row_bytes) != row_count * EMBEDDING_ROW_BYTES:
                raise R9Error(f"truncated embedding dump rows: {path}")
            values = np.frombuffer(row_bytes, dtype="<f4").reshape(
                row_count, 1 + EMBEDDING_DIM + EMBEDDING_SPEAKERS
            )
            rows_out.append(
                {
                    "chunk": int(chunk),
                    "total_n": int(total_n),
                    "n_cache": int(n_cache),
                    "n_fifo": int(n_fifo),
                    "compression": int(compression),
                    "compress_count": int(compress_count),
                    "frame_idx": values[:, 0].view("<i4").astype(np.int32),
                    "emb": values[:, 1 : 1 + EMBEDDING_DIM],
                    "preds": values[:, 1 + EMBEDDING_DIM :],
                }
            )
    return rows_out


def _r9_external_root(root: Path) -> Path:
    return root / "external" / "r9" / "transcribe.cpp"


def _clean_external_root(root: Path) -> Path:
    return root / "external" / "r9" / "transcribe-clean"


def r9b_prepare(root: Path) -> Path:
    cfg = config()
    embedding = _embedding_config(cfg)
    if str(embedding["replay_backend"]) != "vulkan":
        raise R9Error("replay_backend must be vulkan (owner decision)")
    directory = output_root(root)
    external = _r9_external_root(root)
    clean = _clean_external_root(root)
    if not (external / "src" / "arch" / "sortformer" / "model.cpp").is_file():
        raise R9Error(f"external r9 checkout missing: {external}")
    if not (clean / "src" / "arch" / "sortformer" / "model.cpp").is_file():
        raise R9Error(f"external clean checkout missing: {clean}")

    def git(command: Sequence[str]) -> list[str]:
        process = subprocess.run(
            ["git", "-C", str(external), *command],
            capture_output=True,
            text=True,
            errors="replace",
        )
        if process.returncode != 0:
            raise R9Error(f"git {command} failed: {process.stderr.strip()}")
        return process.stdout.splitlines()

    commit = git(["rev-parse", "HEAD"])[0]
    if commit != EMBEDDING_EXTERNAL_COMMIT:
        raise R9Error(f"external checkout at {commit}, expected {EMBEDDING_EXTERNAL_COMMIT}")
    clean_commit_process = subprocess.run(
        ["git", "-C", str(clean), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        errors="replace",
    )
    if clean_commit_process.stdout.strip() != EMBEDDING_EXTERNAL_COMMIT:
        raise R9Error("clean checkout is not at the pinned commit")
    clean_status = subprocess.run(
        ["git", "-C", str(clean), "status", "--porcelain"],
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout.strip()
    if clean_status:
        raise R9Error(f"clean checkout is dirty: {clean_status}")
    dirty = git(["status", "--porcelain"])
    allowed_dirty = {
        "M src/arch/sortformer/model.cpp",
        "M src/arch/sortformer/sortformer.h",
        "M src/arch/sortformer/stream.cpp",
    }
    if {line.strip() for line in dirty} != allowed_dirty:
        raise R9Error(f"external checkout dirty paths differ from the R9-B patch: {dirty}")
    patch = subprocess.run(
        ["git", "-C", str(external), "diff", "--", "src/arch/sortformer"],
        capture_output=True,
        text=True,
        errors="replace",
    )
    if patch.returncode != 0:
        raise R9Error("failed to capture the embedding patch diff")
    patch_path = directory / "embedding_patch.diff"
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(patch.stdout, encoding="utf-8")
    bins = {
        "patched_cpu": external / "build-r9-cpu" / "bin" / "Release" / "transcribe-bench.exe",
        "patched_vulkan": root
        / "builds"
        / "r9-vulkan"
        / "bin"
        / "Release"
        / "transcribe-bench.exe",
        "clean_cpu": clean / "build-r9-clean-cpu" / "bin" / "Release" / "transcribe-bench.exe",
    }
    missing = [name for name, path in bins.items() if not path.is_file()]
    if missing:
        raise R9Error(f"missing R9-B binaries: {missing}")
    model_path = root / "models" / "r8" / str(cfg["model_q8_filename"])
    if not model_path.is_file():
        raise R9Error(f"model missing: {model_path}")
    if sha256_file(model_path) != EMBEDDING_PINNED_MODEL_SHA256:
        raise R9Error(f"model hash differs from the plan-pinned R8 model: {model_path}")
    document = {
        "schema_version": 1,
        "external_repository": EMBEDDING_EXTERNAL_REPOSITORY,
        "external_commit": EMBEDDING_EXTERNAL_COMMIT,
        "external_checkout": str(external),
        "clean_checkout": str(clean),
        "dirty_paths": sorted(dirty),
        "patch_path": str(patch_path),
        "patch_sha256": sha256_file(patch_path),
        "binaries": {
            name: {"path": str(path), "sha256": sha256_file(path)} for name, path in bins.items()
        },
        "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
        "replay_backend": embedding["replay_backend"],
        "dump": embedding["dump"],
        "features": embedding["features"],
        "code_sha256": sha256_file(CODE_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
    }
    path = directory / "r9b_prepare.json"
    write_json(path, document)
    return path


def _verify_preparation_artifacts(prepare_doc: dict[str, Any]) -> None:
    patch_path = Path(str(prepare_doc["patch_path"]))
    if not patch_path.is_file() or sha256_file(patch_path) != str(prepare_doc["patch_sha256"]):
        raise R9Error("embedding_patch.diff is missing or modified; rerun r9b-prepare")
    model_path = Path(str(prepare_doc["model"]["path"]))
    if not model_path.is_file() or sha256_file(model_path) != str(prepare_doc["model"]["sha256"]):
        raise R9Error("model file is missing or modified; rerun r9b-prepare")
    if sha256_file(model_path) != EMBEDDING_PINNED_MODEL_SHA256:
        raise R9Error("model hash differs from the plan-pinned R8 model")
    for name, entry in prepare_doc.get("binaries", {}).items():
        binary_path = Path(str(entry["path"]))
        if not binary_path.is_file() or sha256_file(binary_path) != str(entry["sha256"]):
            raise R9Error(f"binary {name} is missing or modified; rebuild and rerun r9b-prepare")


def _peak_private_bytes(process_handle: Any) -> int:
    if os.name != "nt":
        return 0
    import ctypes
    from ctypes import wintypes

    class _COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = _COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    if psapi.GetProcessMemoryInfo(process_handle, ctypes.byref(counters), ctypes.sizeof(counters)):
        return int(counters.PrivateUsage)
    return 0


def _run_bench(
    bench: Path,
    model: Path,
    audio: Path,
    backend: str,
    dump_dir: Path | None,
    embedding_dump: Path | None,
    bench_json: Path,
    log_path: Path,
    *,
    timeout_seconds: int | None = None,
    memory_ceiling_bytes: int | None = None,
) -> dict[str, Any]:
    dump_dir.mkdir(parents=True, exist_ok=True) if dump_dir is not None else None
    bench_json.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    embedding = _embedding_config(config())
    command = [
        str(bench),
        "--model",
        str(model),
        "--sample",
        str(audio),
        "--backend",
        backend,
        "--threads",
        "8",
        "--warmup",
        "0",
        "--iters",
        "1",
        "--json-out",
        str(bench_json),
    ]
    env = os.environ.copy()
    env["TRANSCRIBE_SORTFORMER_STREAM_PRESET"] = "low_latency"
    if dump_dir is not None:
        env["TRANSCRIBE_DUMP_DIR"] = str(dump_dir)
    if embedding_dump is not None:
        env["TRANSCRIBE_SORTFORMER_EMBEDDING_DUMP_PATH"] = str(embedding_dump)
        env["TRANSCRIBE_SORTFORMER_EMBEDDING_DUMP_CADENCE"] = str(
            int(embedding["dump"]["cadence_chunks"])
        )
        env["TRANSCRIBE_SORTFORMER_EMBEDDING_DUMP_HORIZON"] = str(
            int(embedding["dump"]["horizon_frames"])
        )
    started = time.perf_counter()
    peak_private = 0
    memory_sampled = False
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            while process.poll() is None:
                if timeout_seconds is not None and time.perf_counter() - started > timeout_seconds:
                    process.terminate()
                    process.wait(timeout=30)
                    raise R9Error(
                        f"transcribe-bench timeout after {timeout_seconds}s; see {log_path}"
                    )
                if memory_ceiling_bytes is not None:
                    sampled = _peak_private_bytes(process._handle)
                    if sampled > 0:
                        peak_private = max(peak_private, sampled)
                        memory_sampled = True
                        if peak_private > memory_ceiling_bytes:
                            process.terminate()
                            process.wait(timeout=30)
                            raise R9Error(
                                f"transcribe-bench exceeded the memory ceiling "
                                f"({peak_private} > {memory_ceiling_bytes} bytes); see {log_path}"
                            )
                time.sleep(5.0)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
        return_code = int(process.returncode)
        if memory_ceiling_bytes is not None:
            final_sample = _peak_private_bytes(process._handle)
            if final_sample > 0:
                peak_private = max(peak_private, final_sample)
                memory_sampled = True
                if peak_private > memory_ceiling_bytes:
                    raise R9Error(
                        f"transcribe-bench exceeded the memory ceiling "
                        f"({peak_private} > {memory_ceiling_bytes} bytes); see {log_path}"
                    )
    wall_seconds = time.perf_counter() - started
    if return_code != 0:
        raise R9Error(f"transcribe-bench failed ({return_code}); see {log_path}")
    if memory_ceiling_bytes is not None and not memory_sampled:
        raise R9Error("memory sampling failed; the 24 GiB ceiling cannot be certified")
    return {
        "wall_seconds": wall_seconds,
        "peak_private_bytes": peak_private,
        "memory_sampled": memory_sampled,
        "command": command,
    }


def _validate_dump_records(
    records: Sequence[dict[str, Any]], cadence: int, horizon: int
) -> dict[str, Any]:
    expected_chunk = 0
    last_total = -1
    violations: list[str] = []
    if not records:
        violations.append("no dump records")
    for record in records:
        if record["chunk"] != expected_chunk:
            violations.append(f"chunk {record['chunk']} != expected {expected_chunk}")
        expected_chunk = record["chunk"] + cadence
        if record["total_n"] <= last_total:
            violations.append(f"total_n not increasing at chunk {record['chunk']}")
        last_total = record["total_n"]
        if record["n_cache"] < 0 or record["n_fifo"] < 0:
            violations.append(f"negative row counts at chunk {record['chunk']}")
        if not np.isfinite(record["emb"]).all() or not np.isfinite(record["preds"]).all():
            violations.append(f"non-finite dump values at chunk {record['chunk']}")
        if len(record["frame_idx"]) != int(record["n_cache"]) + int(record["n_fifo"]):
            violations.append(f"row count mismatch at chunk {record['chunk']}")
        lower = max(0, record["total_n"] - horizon)
        frame_idx = record["frame_idx"]
        if len(frame_idx) and (
            int(frame_idx.min()) < lower or int(frame_idx.max()) >= record["total_n"]
        ):
            violations.append(
                f"frame index out of horizon at chunk {record['chunk']}: "
                f"min={int(frame_idx.min())} max={int(frame_idx.max())} total_n={record['total_n']}"
            )
    return {"valid": not violations, "violations": violations}


def r9b_fixture(root: Path) -> Path:
    cfg = config()
    embedding = _embedding_config(cfg)
    directory = output_root(root)
    prepared = directory / "r9b_prepare.json"
    if not prepared.is_file():
        raise R9Error("r9b-prepare must run first")
    prepare_doc = load_json(prepared)
    if prepare_doc.get("code_sha256") != sha256_file(CODE_PATH) or prepare_doc.get(
        "config_sha256"
    ) != sha256_file(CONFIG_PATH):
        raise R9Error("r9b_prepare.json is stale; rerun r9b-prepare")
    _verify_preparation_artifacts(prepare_doc)
    patched_cpu = Path(prepare_doc["binaries"]["patched_cpu"]["path"])
    clean_cpu = Path(prepare_doc["binaries"]["clean_cpu"]["path"])
    model = Path(prepare_doc["model"]["path"])
    fixture_root = directory / "embedding_validation"
    fixture_root.mkdir(parents=True, exist_ok=True)
    validation_target = fixture_root / "embedding_validation.json"
    if validation_target.exists():
        validation_target.unlink()
    smoke_root = r8_output_root(root) / "smoke_audio"
    cadence = int(embedding["dump"]["cadence_chunks"])
    horizon = int(embedding["dump"]["horizon_frames"])
    checks: dict[str, Any] = {}
    fixtures: list[dict[str, Any]] = []
    for clip_id in ("r8_smoke_00", "r8_smoke_03"):
        audio = smoke_root / f"{clip_id}.wav"
        if not audio.is_file():
            raise R9Error(f"fixture audio missing: {audio}")
        base = fixture_root / clip_id
        unpatched_dumps = base / "unpatched" / "dumps"
        patched_dumps = base / "patched" / "dumps"
        patched_dump_bin = base / "patched" / "embedding_dump.bin"
        patched_dump_bin.parent.mkdir(parents=True, exist_ok=True)
        if patched_dump_bin.exists():
            patched_dump_bin.unlink()
        _run_bench(
            clean_cpu,
            model,
            audio,
            "cpu",
            unpatched_dumps,
            None,
            base / "unpatched" / "bench.json",
            base / "unpatched" / "run.log",
        )
        _run_bench(
            patched_cpu,
            model,
            audio,
            "cpu",
            patched_dumps,
            patched_dump_bin,
            base / "patched" / "bench.json",
            base / "patched" / "run.log",
        )
        unpatched_probs = unpatched_dumps / "diar.probs.f32"
        patched_probs = patched_dumps / "diar.probs.f32"
        for path in (unpatched_probs, patched_probs):
            if not path.is_file():
                raise R9Error(f"diar.probs missing for {clip_id}: {path}")
        unpatched_segments = _fixed_segments_vulkan(_load_raw_probs_dump(unpatched_dumps))
        patched_segments = _fixed_segments_vulkan(_load_raw_probs_dump(patched_dumps))
        records = _parse_dump_records(patched_dump_bin)
        structure = _validate_dump_records(records, cadence, horizon)
        expected_records = len(
            [
                chunk
                for chunk in range(int(_load_raw_probs_dump(patched_dumps).shape[0]) // 6 + 1)
                if chunk % cadence == 0
            ]
        )
        if len(records) != expected_records:
            raise R9Error(
                f"fixture {clip_id}: dump record count {len(records)} != expected {expected_records}"
            )
        r8_telemetry_path = (
            r8_output_root(root)
            / "runs"
            / "smoke"
            / "cpu"
            / "diar_streaming_sortformer_4spk-v2.1-Q8_0"
            / "telemetry"
            / f"{clip_id}.jsonl"
        )
        if not r8_telemetry_path.is_file():
            raise R9Error(f"R8 smoke telemetry missing for {clip_id}")
        telemetry_flags: dict[int, bool] = {}
        for row in read_jsonl(r8_telemetry_path):
            telemetry_flags[int(row["chunk_index"])] = bool(row.get("compression_called", False))
        expected_telemetry_chunks = int(_load_raw_probs_dump(patched_dumps).shape[0]) // 6 + 1
        if len(telemetry_flags) != expected_telemetry_chunks:
            raise R9Error(
                f"R8 smoke telemetry incomplete for {clip_id}: "
                f"{len(telemetry_flags)} != {expected_telemetry_chunks}"
            )
        compression_mismatches = [
            (record["chunk"], int(record["compression"]), telemetry_flags.get(record["chunk"]))
            for record in records
            if bool(record["compression"])
            != bool(
                telemetry_flags.get(record["chunk"], False)
                or telemetry_flags.get(record["chunk"] - 1, False)
            )
        ]
        fixtures.append(
            {
                "clip_id": clip_id,
                "audio_sha256": sha256_file(audio),
                "unpatched_probability_sha256": sha256_file(unpatched_probs),
                "patched_probability_sha256": sha256_file(patched_probs),
                "probability_bytes_equal": sha256_file(unpatched_probs)
                == sha256_file(patched_probs),
                "speaker_segments_equal": unpatched_segments == patched_segments,
                "dump_records": len(records),
                "dump_schema_complete": bool(structure["valid"]),
                "r8_telemetry_present": True,
                "dump_structure": structure,
                "compression_mismatches_vs_r8_telemetry": compression_mismatches,
            }
        )
    valid = all(
        fixture["probability_bytes_equal"]
        and fixture["speaker_segments_equal"]
        and fixture["dump_schema_complete"]
        and fixture["r8_telemetry_present"]
        and fixture["dump_structure"]["valid"]
        and not fixture["compression_mismatches_vs_r8_telemetry"]
        for fixture in fixtures
    )
    checks["passed"] = bool(valid)
    checks["fixtures"] = fixtures
    document = {
        "schema_version": 1,
        "passed": checks["passed"],
        "fixtures": fixtures,
        "preparation": {
            "patch_sha256": prepare_doc["patch_sha256"],
            "model_sha256": prepare_doc["model"]["sha256"],
            "binaries": {name: value["sha256"] for name, value in prepare_doc["binaries"].items()},
        },
        "code_sha256": sha256_file(CODE_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
    }
    path = fixture_root / "embedding_validation.json"
    write_json(path, document)
    return path


def _load_raw_probs_dump(dump_dir: Path) -> np.ndarray:
    metadata_path = dump_dir / "diar.probs.json"
    data_path = dump_dir / "diar.probs.f32"
    if not metadata_path.is_file() or not data_path.is_file():
        raise R9Error(f"diar.probs dump is missing: {dump_dir}")
    shape = tuple(int(value) for value in load_json(metadata_path)["shape"])
    probabilities = np.fromfile(data_path, dtype="<f4")
    if probabilities.size != int(np.prod(shape)):
        raise R9Error(f"invalid probability dump size: {probabilities.size} != {shape}")
    probabilities = probabilities.reshape(shape).astype(np.float32)
    if probabilities.ndim != 2 or probabilities.shape[1] != EMBEDDING_SPEAKERS:
        raise R9Error(f"unexpected Sortformer probability shape: {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise R9Error("non-finite Sortformer probabilities")
    return probabilities


def r9b_replay(root: Path) -> Path:
    cfg = config()
    embedding = _embedding_config(cfg)
    directory = output_root(root)
    embeddings_dir = directory / "embeddings" / "vulkan"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = directory / "r9b_runs" / "vulkan"
    runs_dir.mkdir(parents=True, exist_ok=True)
    stale_manifest = runs_dir / "replay_manifest.json"
    if stale_manifest.exists():
        stale_manifest.unlink()
    prepared = directory / "r9b_prepare.json"
    if not prepared.is_file():
        raise R9Error("r9b-prepare must run first")
    prepare_doc = load_json(prepared)
    if prepare_doc.get("code_sha256") != sha256_file(CODE_PATH) or prepare_doc.get(
        "config_sha256"
    ) != sha256_file(CONFIG_PATH):
        raise R9Error("r9b_prepare.json is stale; rerun r9b-prepare")
    _verify_preparation_artifacts(prepare_doc)
    bench = Path(prepare_doc["binaries"]["patched_vulkan"]["path"])
    model = Path(prepare_doc["model"]["path"])
    sessions = _sessions(root)
    cadence = int(embedding["dump"]["cadence_chunks"])
    horizon = int(embedding["dump"]["horizon_frames"])
    validation_path = directory / "embedding_validation" / "embedding_validation.json"
    if not validation_path.is_file():
        raise R9Error("r9b-fixture must run and pass before the replay")
    validation = load_json(validation_path)
    if not bool(validation.get("passed")):
        raise R9Error("fixture validation did not pass; replay refused")
    if validation.get("code_sha256") != sha256_file(CODE_PATH) or validation.get(
        "config_sha256"
    ) != sha256_file(CONFIG_PATH):
        raise R9Error(
            "fixture validation hashes do not match the current harness/config; rerun r9b-fixture"
        )
    preparation_binding = validation.get("preparation", {})
    if preparation_binding.get("patch_sha256") != prepare_doc.get("patch_sha256"):
        raise R9Error("fixture validation does not bind to the current patch; rerun r9b-fixture")
    if preparation_binding.get("model_sha256") != prepare_doc.get("model", {}).get("sha256"):
        raise R9Error("fixture validation does not bind to the current model; rerun r9b-fixture")
    current_binary_hashes = {
        name: value.get("sha256") for name, value in prepare_doc.get("binaries", {}).items()
    }
    if preparation_binding.get("binaries") != current_binary_hashes:
        raise R9Error("fixture validation does not bind to the current binaries; rerun r9b-fixture")
    memory_ceiling_bytes = 24 * 1024 * 1024 * 1024
    wall_budget_seconds = 24 * 3600.0
    storage_floor_bytes = 40 * 1024 * 1024 * 1024
    probabilities_dir = directory / "probabilities" / "vulkan"
    probabilities_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = directory / "speaker_segments" / "vulkan"
    segments_dir.mkdir(parents=True, exist_ok=True)
    cpu_probabilities_dir = r8_output_root(root) / "probabilities" / "cpu"
    rows: list[dict[str, Any]] = []
    total_wall = 0.0

    def invalid_receipt(session_id: str, reason: str) -> None:
        write_json(
            runs_dir / f"{session_id}.receipt.invalid.json",
            {
                "schema_version": 1,
                "session_id": session_id,
                "invalid_reason": reason,
                "created_at_utc": datetime.now(UTC).isoformat(),
            },
        )

    for session in sorted(sessions.values(), key=lambda value: (value.fold, value.session_id)):
        session_id = session.session_id
        dump_bin = embeddings_dir / f"{session_id}.bin"
        base = runs_dir / session_id
        dump_dir = base / "dumps"
        bench_json = base / "bench.json"
        try:
            if total_wall > wall_budget_seconds:
                raise R9Error("24-hour per-backend wall budget exhausted")
            free, _, _ = shutil.disk_usage(str(embeddings_dir))
            if free < storage_floor_bytes:
                raise R9Error(f"storage floor breached: {free} free bytes")
            if dump_bin.exists():
                dump_bin.unlink()
            for stale in dump_dir.glob("diar.probs.*"):
                stale.unlink()
            if bench_json.exists():
                bench_json.unlink()
            run_metrics = _run_bench(
                bench,
                model,
                session.waveform_path,
                "vulkan",
                dump_dir,
                dump_bin,
                bench_json,
                base / "run.log",
                timeout_seconds=int(wall_budget_seconds - total_wall),
                memory_ceiling_bytes=memory_ceiling_bytes,
            )
            total_wall += float(run_metrics["wall_seconds"])
            bench_result = load_json(bench_json)
            resolved_backend = str(bench_result.get("backend", "unknown")).lower()
            if "vulkan" not in resolved_backend:
                raise R9Error(
                    f"backend fallback for {session_id}: requested vulkan, resolved {resolved_backend}"
                )
            records = _parse_dump_records(dump_bin)
            structure = _validate_dump_records(records, cadence, horizon)
            if not structure["valid"]:
                raise R9Error(
                    f"dump structure invalid for {session_id}: {structure['violations'][:5]}"
                )
            probabilities = _load_raw_probs_dump(dump_dir)
            cpu_probabilities = np.load(cpu_probabilities_dir / f"{session_id}.npz")[
                "probabilities"
            ]
            if probabilities.shape != cpu_probabilities.shape:
                raise R9Error(
                    f"Vulkan probability geometry differs from the frozen CPU dump for "
                    f"{session_id}: {probabilities.shape} != {cpu_probabilities.shape}"
                )
            np.savez_compressed(
                probabilities_dir / f"{session_id}.npz",
                probabilities=probabilities,
                frame_ms=np.int32(cfg["frame_ms"]),
                backend=np.asarray("vulkan"),
                model_sha256=np.asarray(sha256_file(model)),
            )
            write_json(
                segments_dir / f"{session_id}.json",
                _fixed_segments_vulkan(probabilities),
            )
            expected_records = len(
                [
                    chunk
                    for chunk in range(int(probabilities.shape[0]) // 6 + 1)
                    if chunk % cadence == 0
                ]
            )
            if len(records) != expected_records:
                raise R9Error(
                    f"dump record count mismatch for {session_id}: "
                    f"{len(records)} != {expected_records}"
                )
            rows.append(
                {
                    "session_id": session_id,
                    "fold": session.fold,
                    "backend_requested": "vulkan",
                    "backend_resolved": resolved_backend,
                    "bench_path": str(bench),
                    "bench_sha256": sha256_file(bench),
                    "model_path": str(model),
                    "model_sha256": sha256_file(model),
                    "embedding_dump_path": str(dump_bin),
                    "embedding_dump_sha256": sha256_file(dump_bin),
                    "embedding_dump_records": len(records),
                    "embedding_dump_bytes": dump_bin.stat().st_size,
                    "probability_path": str(probabilities_dir / f"{session_id}.npz"),
                    "probability_sha256": sha256_file(probabilities_dir / f"{session_id}.npz"),
                    "probability_shape": list(probabilities.shape),
                    "wall_seconds": run_metrics["wall_seconds"],
                    "peak_private_bytes": run_metrics["peak_private_bytes"],
                    "memory_sampled": run_metrics["memory_sampled"],
                    "memory_within_ceiling": (
                        run_metrics["peak_private_bytes"] <= memory_ceiling_bytes
                        if run_metrics["memory_sampled"]
                        else None
                    ),
                    "cpu_probability_shape": list(cpu_probabilities.shape),
                }
            )
        except BaseException as error:  # receipt integrity on any limit/cleanup/cancel failure
            invalid_receipt(session_id, f"{type(error).__name__}: {error}")
            raise
    unverified = [
        row["session_id"]
        for row in rows
        if bool(row.get("memory_sampled")) is not True
        or bool(row.get("memory_within_ceiling")) is not True
    ]
    if unverified:
        raise R9Error(
            f"replay rows not certified within the memory ceiling: {unverified}; rerun r9b-replay"
        )
    manifest = {
        "schema_version": 1,
        "backend": "vulkan",
        "memory_ceiling_bytes": memory_ceiling_bytes,
        "wall_budget_seconds": wall_budget_seconds,
        "total_wall_seconds": total_wall,
        "items": rows,
        "code_sha256": sha256_file(CODE_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
    }
    path = runs_dir / "replay_manifest.json"
    write_json(path, manifest)
    return path


def _fixed_segments_vulkan(probabilities: np.ndarray) -> list[dict[str, Any]]:
    frame_ms = int(config()["frame_ms"])
    result: list[dict[str, Any]] = []
    for speaker in range(probabilities.shape[1]):
        active = probabilities[:, speaker] > 0.5
        start: int | None = None
        for frame in range(len(active) + 1):
            value = frame < len(active) and bool(active[frame])
            if value and start is None:
                start = frame
            elif not value and start is not None:
                result.append(
                    {
                        "speaker_slot": speaker + 1,
                        "start_ms": start * frame_ms,
                        "end_ms": frame * frame_ms,
                    }
                )
                start = None
    return sorted(result, key=lambda row: (int(row["start_ms"]), int(row["speaker_slot"])))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left)) * float(np.linalg.norm(right))
    if denominator <= 0.0:
        return float("nan")
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _embedding_features_for_candidate(
    record: dict[str, Any],
    candidate_frame: int,
    candidate_slot: int,
    features: dict[str, Any],
) -> tuple[dict[str, float], bool]:
    before_frames = int(features["before_frames"])
    after_frames = int(features["after_frames"])
    after_lo = int(features["after_lo_frames"])
    frame_idx = record["frame_idx"]
    emb = record["emb"]
    preds = record["preds"]
    active = preds.max(axis=1) >= 0.5
    slots = preds.argmax(axis=1)
    slot = candidate_slot - 1
    after_mask = (frame_idx >= candidate_frame + after_lo) & (
        frame_idx <= candidate_frame + after_frames
    )
    before_mask = (frame_idx >= candidate_frame - before_frames) & (
        frame_idx <= candidate_frame - 1
    )
    after_slot = after_mask & active & (slots == slot)
    before_slot = before_mask & active & (slots == slot)
    mean_after = emb[after_slot].mean(axis=0) if bool(after_slot.any()) else None
    mean_before = emb[before_slot].mean(axis=0) if bool(before_slot.any()) else None
    excluded = mean_after is None and mean_before is None
    if mean_after is not None and mean_before is not None:
        same_slot = _cosine(
            np.asarray(mean_after, dtype=np.float32), np.asarray(mean_before, dtype=np.float32)
        )
        jump = float(np.linalg.norm(mean_after - mean_before))
    else:
        same_slot = float("nan")
        jump = float("nan")
    best_other = float("nan")
    if mean_after is not None:
        best = -2.0
        for other in range(EMBEDDING_SPEAKERS):
            if other == slot:
                continue
            other_mask = after_mask & active & (slots == other)
            if bool(other_mask.any()):
                cosine = _cosine(
                    np.asarray(mean_after, dtype=np.float32),
                    np.asarray(emb[other_mask].mean(axis=0), dtype=np.float32),
                )
                best = max(best, cosine)
        if best > -1.5:
            best_other = best
    return {
        "same_slot_similarity": same_slot,
        "best_other_similarity": best_other,
        "embedding_jump": jump,
    }, excluded


def _replay_manifest_and_dumps(
    root: Path, directory: Path
) -> tuple[dict[str, Any], dict[str, Path], dict[str, Path]]:
    manifest = directory / "r9b_runs" / "vulkan" / "replay_manifest.json"
    if not manifest.is_file():
        raise R9Error("r9b-replay must run before this step")
    manifest_doc = load_json(manifest)
    # Consumers verify content identity (per-file hashes below) plus replay-time gates
    # (memory certification rows, fixture binding at r9b_replay entry). Harness identity at
    # replay time stays recorded in the manifest, but is NOT re-checked here: report-text or
    # downstream-only harness changes must not force a multi-hour GPU replay.
    expected_session_ids = {str(session_id) for session_id in _sessions(root)}
    manifest_items = list(manifest_doc.get("items", []))
    if {str(item["session_id"]) for item in manifest_items} != expected_session_ids:
        raise R9Error("replay manifest does not cover all ten sessions")
    dump_paths: dict[str, Path] = {}
    probability_paths: dict[str, Path] = {}
    for item in manifest_items:
        dump_path = Path(item["embedding_dump_path"])
        if not dump_path.is_file():
            raise R9Error(f"embedding dump missing for {item['session_id']}")
        if sha256_file(dump_path) != str(item.get("embedding_dump_sha256")):
            raise R9Error(f"embedding dump hash mismatch for {item['session_id']}")
        probability_path = Path(item["probability_path"])
        if not probability_path.is_file():
            raise R9Error(f"probability dump missing for {item['session_id']}")
        if sha256_file(probability_path) != str(item.get("probability_sha256")):
            raise R9Error(f"probability dump hash mismatch for {item['session_id']}")
        dump_paths[str(item["session_id"])] = dump_path
        probability_paths[str(item["session_id"])] = probability_path
    return manifest_doc, dump_paths, probability_paths


def extract_embedding_features(root: Path) -> Path:
    cfg = config()
    embedding = _embedding_config(cfg)
    directory = output_root(root)
    features_a = directory / "features_a.jsonl"
    candidates = directory / "candidates.jsonl"
    for required in (features_a, candidates):
        if not required.is_file():
            raise R9Error(f"{required.name} must exist before r9b-extract")
    feature_rows = read_jsonl(features_a)
    candidates_rows = read_jsonl(candidates)
    by_row = {int(row["row"]): row for row in candidates_rows}
    _, dump_paths, _ = _replay_manifest_and_dumps(root, directory)
    records_by_session: dict[str, list[dict[str, Any]]] = {}
    similarity_names = ("same_slot_similarity", "best_other_similarity", "embedding_jump")
    excluded_count = 0
    missing_count = 0
    for row in feature_rows:
        session_id = str(row["session_id"])
        candidate = by_row[int(row["row"])]
        if session_id not in dump_paths:
            raise R9Error(f"no embedding dump for {session_id}")
        if session_id not in records_by_session:
            records_by_session[session_id] = _parse_dump_records(dump_paths[session_id])
        records = records_by_session[session_id]
        target = int(candidate["frame"]) + int(embedding["features"]["after_frames"])
        chosen = next((record for record in records if record["total_n"] > target), records[-1])
        previous = records[records.index(chosen) - 1] if records.index(chosen) > 0 else None
        values, excluded = _embedding_features_for_candidate(
            chosen,
            int(candidate["frame"]),
            int(candidate["speaker_slot"]),
            embedding["features"],
        )
        compression_boundary = (
            1.0
            if chosen["compression"] or (previous is not None and previous["compression"])
            else 0.0
        )
        if compression_boundary == 1.0:
            for name in similarity_names:
                values[name] = float("nan")
        if any(math.isnan(float(values[name])) for name in similarity_names):
            missing_count += 1
        row["features"] = {
            **row["features"],
            **values,
            "compression_boundary": compression_boundary,
        }
        row["excluded_embedding"] = int(bool(excluded))
        excluded_count += int(bool(excluded))
    path = directory / "features_b.jsonl"
    write_jsonl(path, feature_rows)
    summary_path = directory / "embedding_feature_summary.json"
    write_json(
        summary_path,
        {
            "schema_version": 1,
            "row_count": len(feature_rows),
            "excluded_embedding_count": excluded_count,
            "nan_feature_count": missing_count,
            "code_sha256": sha256_file(CODE_PATH),
            "config_sha256": sha256_file(CONFIG_PATH),
        },
    )
    receipt = {
        "schema_version": 1,
        "feature_file": "features_b.jsonl",
        "features_b_sha256": sha256_file(path),
        "dependencies": {
            "features_a.jsonl": sha256_file(features_a),
            "candidates.jsonl": sha256_file(candidates),
            "r9b_runs/vulkan/replay_manifest.json": sha256_file(
                directory / "r9b_runs" / "vulkan" / "replay_manifest.json"
            ),
            "embedding_dumps": {
                session_id: sha256_file(dump_path)
                for session_id, dump_path in sorted(dump_paths.items())
            },
        },
        "code_sha256": sha256_file(CODE_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
    }
    write_json(directory / "features_b_receipt.json", receipt)
    return path


def _b1_fold_transform(matrix: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    transformed = matrix.copy()
    embedding_names = list(config()["embedding"]["features"]["feature_names"])
    continuous = [name for name in embedding_names if name != "compression_boundary"]
    base_count = len(list(config()["feature_names"]))
    for name in continuous:
        index = base_count + embedding_names.index(name)
        train_values = matrix[train_mask, index]
        finite = train_values[np.isfinite(train_values)]
        if len(finite) == 0:
            raise R9Error(f"no finite training values for embedding feature {name}")
        median = float(np.median(finite))
        nan_mask = ~np.isfinite(transformed[:, index])
        transformed[nan_mask, index] = median
    return transformed


def _require_receipt(
    directory: Path, receipt_name: str, expected_arm: str | None = None
) -> dict[str, Any]:
    path = directory / receipt_name
    if not path.is_file():
        raise R9Error(f"{receipt_name} is missing; rerun the producing action")
    receipt = load_json(path)
    if receipt.get("code_sha256") != sha256_file(CODE_PATH) or receipt.get(
        "config_sha256"
    ) != sha256_file(CONFIG_PATH):
        raise R9Error(
            f"{receipt_name} hashes do not match the current harness/config; rerun the producing action"
        )
    if expected_arm is not None and receipt.get("arm") != expected_arm:
        raise R9Error(f"{receipt_name} arm mismatch; rerun the producing action")
    return receipt


def _validate_receipt_dependencies(root: Path, directory: Path, receipt: dict[str, Any]) -> None:
    dependencies = receipt.get("dependencies") or {}
    for name, expected in dependencies.items():
        if name == "embedding_dumps":
            _, dump_paths, _ = _replay_manifest_and_dumps(root, directory)
            for session_id, expected_hash in (expected or {}).items():
                dump_path = dump_paths.get(session_id)
                if dump_path is None or sha256_file(dump_path) != str(expected_hash):
                    raise R9Error(
                        f"embedding dump dependency mismatch for {session_id}; rerun r9b-replay and r9b-extract"
                    )
            continue
        path = directory / str(name)
        if not path.is_file() or sha256_file(path) != str(expected):
            raise R9Error(f"dependency mismatch for {name}; rerun the producing action")


def run_b1(root: Path) -> Path:
    cfg = config()
    embedding = _embedding_config(cfg)
    directory = output_root(root)
    a1_path = directory / "a1_metrics.json"
    if not a1_path.is_file():
        raise R9Error("a1 must run before b1 (no parallel-run authorization)")
    a1_receipt = _require_receipt(directory, "a1_metrics_receipt.json", expected_arm="a1")
    a1_document = load_json(a1_path)
    if (
        a1_document.get("arm") != "a1"
        or a1_document.get("code_sha256") != sha256_file(CODE_PATH)
        or a1_document.get("config_sha256") != sha256_file(CONFIG_PATH)
        or sha256_file(a1_path) != str(a1_receipt.get("metrics_sha256"))
    ):
        raise R9Error("a1_metrics.json is stale or invalid; rerun a1 with the current harness")
    a1_curve = list(a1_document.get("evaluation_curve") or [])
    if not a1_curve or not all(
        np.isfinite(float(point.get("recall_250", np.nan)))
        and np.isfinite(float(point.get("false_events_per_hour", np.nan)))
        for point in a1_curve
    ):
        raise R9Error("a1_metrics.json has no valid finite evaluation curve; rerun a1")
    _validate_receipt_dependencies(root, directory, a1_receipt)
    features_b_receipt = _require_receipt(directory, "features_b_receipt.json")
    _validate_receipt_dependencies(root, directory, features_b_receipt)
    features_b_path = directory / "features_b.jsonl"
    if not features_b_path.is_file() or sha256_file(features_b_path) != str(
        features_b_receipt.get("features_b_sha256")
    ):
        raise R9Error("features_b.jsonl is missing or modified; rerun r9b-extract")
    feature_names = list(cfg["feature_names"]) + list(embedding["features"]["feature_names"])
    return _verifier_arm(
        root,
        arm="b1",
        diagnostic=False,
        feature_file="features_b",
        feature_names=feature_names,
        output_name="b1_metrics",
        scores_name="b1",
        transfer_name="threshold_transfer_b1_metrics.json",
        mlp_trigger_source=None,
        fold_matrix_transform=_b1_fold_transform,
        availability_frames=int(embedding["features"]["after_frames"]),
        dependencies={"features_b.jsonl": sha256_file(features_b_path)},
    )


def _b1_linear_models(root: Path) -> list[dict[str, Any]]:
    cfg = config()
    embedding = _embedding_config(cfg)
    directory = output_root(root)
    b1_metrics_path = directory / "b1_metrics.json"
    if not b1_metrics_path.is_file():
        raise R9Error("b1 must run before b2")
    if bool(load_json(b1_metrics_path).get("mlp_triggered", False)):
        raise R9Error("b2 requires the linear B1 verifier; the MLP fallback was triggered")
    names = list(cfg["feature_names"]) + list(embedding["features"]["feature_names"])
    rows = read_jsonl(directory / "features_b.jsonl")
    matrix = _feature_matrix(rows, names)
    labels = _label_vector(rows)
    row_session = [str(row["session_id"]) for row in rows]
    models: list[dict[str, Any]] = []
    for fold, held_out in enumerate(cfg["folds"]):
        train_mask = np.asarray(
            [session_id not in held_out for session_id in row_session], dtype=np.bool_
        )
        fold_matrix = _b1_fold_transform(matrix, train_mask)
        mean, scale = _standardize_fit(fold_matrix[train_mask])
        standardized = _standardize(fold_matrix, mean, scale)
        usable = labels != -1
        train_usable = train_mask & usable
        weights = _balance_weights(labels[train_usable])
        coef, intercept = _fit_linear(
            standardized[train_usable], labels[train_usable], weights, cfg["verifier"]
        )
        models.append(
            {
                "fold": fold,
                "mean": [float(value) for value in mean],
                "scale": [float(value) for value in scale],
                "coef": [float(value) for value in coef],
                "intercept": float(intercept),
            }
        )
    return models


def _dense_embedding_frames(
    dump_path: Path, total_frames: int
) -> tuple[np.ndarray, np.ndarray, int]:
    records = _parse_dump_records(dump_path)
    effective_frames = int(min(total_frames, max(record["total_n"] for record in records)))
    if effective_frames <= 0:
        raise R9Error(f"no processed frames in {dump_path.name}")
    embeddings = np.zeros((effective_frames, EMBEDDING_DIM), dtype=np.float32)
    preds = np.zeros((effective_frames, EMBEDDING_SPEAKERS), dtype=np.float32)
    filled = np.zeros(effective_frames, dtype=np.bool_)
    for record in records:
        frame_idx = record["frame_idx"]
        fresh = (
            (frame_idx >= 0)
            & (frame_idx < effective_frames)
            & ~filled[frame_idx.clip(0, effective_frames - 1)]
        )
        if bool(fresh.any()):
            targets = frame_idx[fresh]
            embeddings[targets] = record["emb"][fresh]
            preds[targets] = record["preds"][fresh]
            filled[targets] = True
    missing = int((~filled).sum())
    if missing:
        raise R9Error(f"{missing} frames have no dumped embedding in {dump_path.name}")
    return embeddings, preds, effective_frames


def _b2_detect(
    probabilities: np.ndarray,
    dense_emb: np.ndarray,
    dense_preds: np.ndarray,
    b2_cfg: dict[str, Any],
) -> list[tuple[int, int]]:
    window = int(b2_cfg["window_frames"])
    stride = int(b2_cfg["stride_frames"])
    threshold = float(b2_cfg["similarity_threshold"])
    active_value = probabilities.max(axis=1)
    active_slot = probabilities.argmax(axis=1)
    raw: list[tuple[int, int]] = []
    run_slot: int | None = None
    run_start = -1
    for frame in range(len(probabilities) + 1):
        active = frame < len(probabilities) and float(active_value[frame]) >= 0.5
        slot_now = int(active_slot[frame]) if frame < len(probabilities) else -1
        if run_slot is None:
            if active:
                run_slot = slot_now
                run_start = frame
            continue
        if active and slot_now == run_slot:
            continue
        end = frame
        offset = run_start
        while offset + 2 * window <= end:
            before_active = dense_preds[offset : offset + window].max(axis=1) >= 0.5
            after_active = dense_preds[offset + window : offset + 2 * window].max(axis=1) >= 0.5
            before_slot_mask = (
                dense_preds[offset : offset + window].argmax(axis=1) == run_slot
            ) & before_active
            after_slot_mask = (
                dense_preds[offset + window : offset + 2 * window].argmax(axis=1) == run_slot
            ) & after_active
            if bool(before_slot_mask.any()) and bool(after_slot_mask.any()):
                before_mean = dense_emb[offset : offset + window][before_slot_mask].mean(axis=0)
                after_mean = dense_emb[offset + window : offset + 2 * window][after_slot_mask].mean(
                    axis=0
                )
                if _cosine(before_mean, after_mean) < threshold:
                    raw.append((offset + window, run_slot))
            offset += stride
        run_slot = None
        run_start = -1
        if active:
            run_slot = slot_now
            run_start = frame
    dedup_radius_frames = max(
        1,
        int(math.ceil(int(config()["duplicate_suppression_ms"]) / int(config()["frame_ms"]))),
    )
    raw.sort()
    deduplicated: list[tuple[int, int]] = []
    for frame, slot in raw:
        if not deduplicated or frame - deduplicated[-1][0] >= dedup_radius_frames:
            deduplicated.append((frame, slot))
    return deduplicated


def run_b2(root: Path) -> Path:
    cfg = config()
    embedding = _embedding_config(cfg)
    b2_cfg = embedding["b2"]
    directory = output_root(root)
    sessions = _sessions(root)
    fold_map = _fold_map(cfg)
    b1_metrics_path = directory / "b1_metrics.json"
    if not b1_metrics_path.is_file():
        raise R9Error("b1 must run before b2")
    b1_document = load_json(b1_metrics_path)

    def stopped_receipt(stop_reason: str, **extra: Any) -> Path:
        path = directory / "b2_metrics.json"
        write_json(
            path,
            {
                "schema_version": 1,
                "stopped": True,
                "stop_reason": stop_reason,
                **extra,
                "code_sha256": sha256_file(CODE_PATH),
                "config_sha256": sha256_file(CONFIG_PATH),
            },
        )
        return path

    if b1_document.get("code_sha256") != sha256_file(CODE_PATH) or b1_document.get(
        "config_sha256"
    ) != sha256_file(CONFIG_PATH):
        raise R9Error("b1_metrics.json hashes do not match the current harness/config; rerun b1")
    if b1_document.get("arm") != "b1" or "evaluation_curve" not in b1_document:
        raise R9Error("b1_metrics.json is missing required B1 schema fields")
    b1_receipt = _require_receipt(directory, "b1_metrics_receipt.json", expected_arm="b1")
    if sha256_file(b1_metrics_path) != str(b1_receipt.get("metrics_sha256")):
        raise R9Error("b1_metrics.json is missing or modified; rerun b1")
    _validate_receipt_dependencies(root, directory, b1_receipt)
    features_b_receipt = _require_receipt(directory, "features_b_receipt.json")
    _validate_receipt_dependencies(root, directory, features_b_receipt)
    if bool(b1_document.get("mlp_triggered", False)):
        return stopped_receipt("b1_mlp_fallback")
    scores_b1_path = directory / "scores_b1.jsonl"
    features_b_path = directory / "features_b.jsonl"
    if not scores_b1_path.is_file() or sha256_file(scores_b1_path) != str(
        b1_receipt.get("scores_sha256")
    ):
        raise R9Error("scores_b1.jsonl is missing or modified; rerun b1")
    if not features_b_path.is_file() or sha256_file(features_b_path) != str(
        b1_receipt.get("feature_file_sha256")
    ):
        raise R9Error("features_b.jsonl is missing or modified; rerun r9b-extract and b1")
    evaluation_curve = b1_document.get("evaluation_curve")
    if not isinstance(evaluation_curve, list) or not evaluation_curve:
        raise R9Error("b1 evaluation curve is empty; rerun b1 before b2")
    for row in evaluation_curve:
        if (
            not isinstance(row, dict)
            or not math.isfinite(float(row.get("recall_250", float("nan"))))
            or not math.isfinite(float(row.get("false_events_per_hour", float("nan"))))
        ):
            raise R9Error("b1 evaluation curve contains an invalid row; rerun b1")
    b1_recall_100 = float(_select_row(evaluation_curve, 100.0).get("recall_250") or 0.0)
    ceiling_doc_path = directory / "ceiling_summary.json"
    if not ceiling_doc_path.is_file():
        raise R9Error("ceiling_summary.json is missing; rerun the ceiling action before b2")
    ceiling_doc = load_json(ceiling_doc_path)
    if ceiling_doc.get("code_sha256") != sha256_file(CODE_PATH) or ceiling_doc.get(
        "config_sha256"
    ) != sha256_file(CONFIG_PATH):
        raise R9Error(
            "ceiling summary hashes do not match the current harness/config; rerun ceiling"
        )
    oracle_value = ceiling_doc.get("oracle_recall_250")
    if (
        not isinstance(oracle_value, (int, float))
        or not math.isfinite(float(oracle_value))
        or not 0.0 <= float(oracle_value) <= 1.0
    ):
        raise R9Error(
            "ceiling summary has an invalid oracle recall (expected a finite value in [0, 1]); rerun ceiling"
        )
    candidate_ceiling_recall = float(oracle_value)
    entry = bool(b1_recall_100 >= 0.6 or candidate_ceiling_recall < 0.9)
    if not entry:
        return stopped_receipt(
            "entry_condition_not_met",
            b1_recall_100_feh=b1_recall_100,
            candidate_ceiling_recall=candidate_ceiling_recall,
        )
    _, dump_paths, probability_paths = _replay_manifest_and_dumps(root, directory)
    windows = cfg["feature_windows"]
    frame_ms = int(cfg["frame_ms"])
    confirmation_frames = int(cfg["confirmation"]["base_frames"])
    fold_candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)

    def detect_fold_sessions(folds: Sequence[int]) -> None:
        for session in sorted(sessions.values(), key=lambda value: (value.fold, value.session_id)):
            if session.fold not in folds:
                continue
            probabilities = np.load(probability_paths[session.session_id])["probabilities"]
            dense_emb, dense_preds, effective_frames = _dense_embedding_frames(
                dump_paths[session.session_id], int(probabilities.shape[0])
            )
            probabilities = probabilities[:effective_frames]
            detected = _b2_detect(probabilities, dense_emb, dense_preds, b2_cfg)
            for frame, slot in detected:
                sample = frame * int(cfg["samples_per_frame"])
                fold_candidates[fold_map[session.session_id]].append(
                    {
                        "session_id": session.session_id,
                        "frame": frame,
                        "sample": sample,
                        "speaker_slot": slot + 1,
                        "candidate_kind": "intra_slot",
                    }
                )

    detect_fold_sessions([0])
    first_fold_count = len(fold_candidates.get(0, []))
    if first_fold_count < 10:
        return stopped_receipt("fail_fast_noise_floor", first_fold_candidate_count=first_fold_count)
    detect_fold_sessions([1, 2, 3, 4])
    models = _b1_linear_models(root)
    feature_names = list(cfg["feature_names"]) + list(embedding["features"]["feature_names"])
    b2_rows: list[dict[str, Any]] = []
    sessions_by_id = {str(key): value for key, value in sessions.items()}
    records_by_session: dict[str, list[dict[str, Any]]] = {}
    cpu_probabilities: dict[str, np.ndarray] = {}
    features_b_rows = read_jsonl(directory / "features_b.jsonl")
    features_b_matrix = _feature_matrix(features_b_rows, feature_names)
    row_session_main = [str(row["session_id"]) for row in features_b_rows]
    similarity_names = ("same_slot_similarity", "best_other_similarity", "embedding_jump")
    continuous_embedding_names = [
        name for name in embedding["features"]["feature_names"] if name != "compression_boundary"
    ]
    base_count = len(list(cfg["feature_names"]))
    b2_scoring_seconds = 0.0
    for fold, candidates in sorted(fold_candidates.items()):
        model = models[fold]
        mean = np.asarray(model["mean"], dtype=np.float64)
        scale = np.asarray(model["scale"], dtype=np.float64)
        coef = np.asarray(model["coef"], dtype=np.float64)
        train_mask = np.asarray(
            [session_id not in cfg["folds"][fold] for session_id in row_session_main],
            dtype=np.bool_,
        )
        imputation_medians: dict[int, float] = {}
        for name in continuous_embedding_names:
            index = base_count + list(embedding["features"]["feature_names"]).index(name)
            train_values = features_b_matrix[train_mask, index]
            finite = train_values[np.isfinite(train_values)]
            imputation_medians[index] = float(np.median(finite))
        for candidate in candidates:
            started = time.perf_counter()
            session_id = str(candidate["session_id"])
            session = sessions_by_id[session_id]
            if session_id not in cpu_probabilities:
                cpu_probabilities[session_id] = _load_probabilities(root, session_id)
            a_features = _features_for_candidate(
                cpu_probabilities[session_id],
                int(candidate["frame"]),
                int(candidate["speaker_slot"]) - 1,
                confirmation_frames,
                windows,
                frame_ms,
            )
            label = _label_candidate(int(candidate["sample"]), session.events)
            if session_id not in records_by_session:
                records_by_session[session_id] = _parse_dump_records(dump_paths[session_id])
            records = records_by_session[session_id]
            target = int(candidate["frame"]) + int(embedding["features"]["after_frames"])
            chosen = next((record for record in records if record["total_n"] > target), records[-1])
            previous = records[records.index(chosen) - 1] if records.index(chosen) > 0 else None
            values, excluded = _embedding_features_for_candidate(
                chosen,
                int(candidate["frame"]),
                int(candidate["speaker_slot"]),
                embedding["features"],
            )
            compression_boundary = (
                1.0
                if chosen["compression"] or (previous is not None and previous["compression"])
                else 0.0
            )
            if compression_boundary == 1.0:
                for name in similarity_names:
                    values[name] = float("nan")
            features = {**a_features, **values, "compression_boundary": compression_boundary}
            feature_vector = np.asarray(
                [float(features[name]) for name in feature_names], dtype=np.float64
            )
            for index in np.nonzero(~np.isfinite(feature_vector))[0]:
                feature_vector[int(index)] = imputation_medians[int(index)]
            standardized = _standardize(feature_vector[None, :], mean, scale)
            score = float(_linear_predict(standardized, coef, float(model["intercept"]))[0])
            b2_scoring_seconds += time.perf_counter() - started
            b2_rows.append(
                {
                    "session_id": session_id,
                    "sample": int(candidate["sample"]),
                    "frame": int(candidate["frame"]),
                    "speaker_slot": int(candidate["speaker_slot"]),
                    "label": str(label["label"]),
                    "excluded_embedding": int(bool(excluded)),
                    "score": score,
                }
            )
    scores_path = directory / "scores_b2.jsonl"
    write_jsonl(scores_path, b2_rows)
    union_rows = list(read_jsonl(directory / "scores_b1.jsonl")) + b2_rows
    grouped: dict[str, list[tuple[int, float]]] = {}
    grouped_selection: dict[str, list[tuple[int, float]]] = {}
    for row in union_rows:
        session_id = str(row["session_id"])
        pair = (int(row["sample"]), float(row["score"]))
        grouped.setdefault(session_id, []).append(pair)
        if str(row["label"]) != "ambiguous":
            grouped_selection.setdefault(session_id, []).append(pair)
    for session_id in sessions:
        grouped.setdefault(session_id, [])
        grouped_selection.setdefault(session_id, [])
    radius_samples = int(cfg["duplicate_suppression_ms"]) * 16
    for session_id in grouped:
        grouped[session_id] = _grouped_events(
            [{"sample": sample, "score": score} for sample, score in grouped[session_id]],
            radius_samples,
        )
        grouped_selection[session_id] = _grouped_events(
            [{"sample": sample, "score": score} for sample, score in grouped_selection[session_id]],
            radius_samples,
        )
    selection_cache = ScoreEvaluationCache(sessions, grouped_selection)
    all_cache = ScoreEvaluationCache(sessions, grouped)
    score_values = np.asarray(
        [score for pairs in grouped_selection.values() for _, score in pairs],
        dtype=np.float32,
    )
    targets = [float(value) for value in cfg["targets"]["false_events_per_hour"]]
    curve = _curve_for_scores(selection_cache, score_values, targets, cfg["curve_search"])
    evaluation_curve_rows: list[dict[str, Any]] = []
    selected_points: dict[str, Any] = {}
    for row in curve:
        metrics = all_cache.metrics(float(row["threshold"]))
        primary = metrics["tolerances"]["250"]
        evaluation_curve_rows.append(
            {
                "threshold": row["threshold"],
                "prediction_count": metrics["prediction_count"],
                "true_positive_count": primary["true_positive_count"],
                "false_event_count": primary["false_event_count"],
                "false_events_per_hour": primary["false_events_per_hour"],
                "recall_250": primary["recall"],
            }
        )
    for target in targets:
        selected = _select_row(curve, float(target))
        selected_points[str(target)] = {
            "threshold": selected["threshold"],
            "selection_false_events_per_hour": selected["false_events_per_hour"],
            "selection_recall_250": selected["recall_250"],
            "metrics": all_cache.metrics(float(selected["threshold"])),
        }
    transfer: list[dict[str, Any]] = []
    for fold, held_out in enumerate(cfg["folds"]):
        held_out_ids = [str(value) for value in held_out]
        development_ids = [session_id for session_id in sessions if session_id not in held_out_ids]
        development_rows = [
            _curve_row(selection_cache, float(row["threshold"]), development_ids) for row in curve
        ]
        for target in targets:
            selected = _select_row(development_rows, float(target))
            held_metrics = all_cache.metrics(float(selected["threshold"]), held_out_ids)
            primary = held_metrics["tolerances"]["250"]
            transfer.append(
                {
                    "fold": fold,
                    "held_out_sessions": held_out_ids,
                    "target_false_events_per_hour": target,
                    "selected_threshold": selected["threshold"],
                    "development_false_events_per_hour": selected["false_events_per_hour"],
                    "development_recall_250": selected["recall_250"],
                    "held_out_false_events_per_hour": primary["false_events_per_hour"],
                    "held_out_recall_250": primary["recall"],
                }
            )
    per_meeting_curves: dict[str, list[dict[str, Any]]] = {
        session_id: [] for session_id in sessions
    }
    for row in curve:
        metrics = all_cache.metrics(float(row["threshold"]))
        for session_id, meeting in metrics["per_meeting"].items():
            per_meeting_curves[session_id].append(
                {
                    "threshold": row["threshold"],
                    "prediction_count": meeting["prediction_count"],
                    "true_positive_count": meeting["true_positive_count"],
                    "false_event_count": meeting["false_event_count"],
                    "false_events_per_hour": meeting["false_events_per_hour"],
                    "recall_250": meeting["recall"],
                }
            )
    availability_frame_horizon_ms = int(embedding["features"]["after_frames"]) * frame_ms
    verification_compute_ms_per_event_b2 = (
        max(b2_scoring_seconds / len(b2_rows) * 1000.0, 0.0) if b2_rows else 0.0
    )
    b1_compute_value = b1_document.get("availability_latency", {}).get(
        "verification_compute_ms_per_event"
    )
    if not isinstance(b1_compute_value, (int, float)) or not math.isfinite(float(b1_compute_value)):
        raise R9Error("b1_metrics.json lacks a finite verifier compute value; rerun b1")
    verification_compute_ms_per_event_b1 = float(b1_compute_value)
    b1_score_by_key: dict[tuple[str, int], float] = {}
    for row in read_jsonl(directory / "scores_b1.jsonl"):
        key = (str(row["session_id"]), int(row["sample"]))
        score = float(row["score"])
        if key not in b1_score_by_key or score > b1_score_by_key[key]:
            b1_score_by_key[key] = score
    b2_score_by_key: dict[tuple[str, int], float] = {}
    for row in b2_rows:
        key = (str(row["session_id"]), int(row["sample"]))
        score = float(row["score"])
        if key not in b2_score_by_key or score > b2_score_by_key[key]:
            b2_score_by_key[key] = score

    def selected_arm(session_id: str, sample: int) -> str:
        key = (session_id, sample)
        b1_score = b1_score_by_key.get(key, float("-inf"))
        b2_score = b2_score_by_key.get(key, float("-inf"))
        # The union grouping emits the max-score candidate at each sample;
        # ties resolve to B2 (the more expensive compute, conservative).
        if key in b2_score_by_key and b2_score >= b1_score:
            return "b2"
        return "b1"

    defect_threshold_ms = float(cfg["reference_gate"]["latency_defect_median_ms"])
    availability: dict[str, Any] = {
        "availability_frame_horizon_ms": availability_frame_horizon_ms,
        "verification_compute_ms_per_event_b1": verification_compute_ms_per_event_b1,
        "verification_compute_ms_per_event_b2": verification_compute_ms_per_event_b2,
        "latency_defect_median_ms": defect_threshold_ms,
        "per_target": {},
    }
    for target, point in selected_points.items():
        deltas = []
        availability_values = []
        for session_id, prediction, reference in point["metrics"]["matched_pairs"]:
            lag = (int(prediction) - int(reference)) // 16
            compute_ms = (
                verification_compute_ms_per_event_b2
                if selected_arm(str(session_id), int(prediction)) == "b2"
                else verification_compute_ms_per_event_b1
            )
            deltas.append(lag)
            availability_values.append(float(lag) + availability_frame_horizon_ms + compute_ms)
        availability_percentiles = _percentiles(availability_values)
        availability["per_target"][str(target)] = {
            "boundary_lag_ms": _percentiles([float(value) for value in deltas]),
            "availability_ms": availability_percentiles,
            "latency_defect": availability_percentiles["p50"] is not None
            and float(availability_percentiles["p50"]) >= defect_threshold_ms,
        }
    path = directory / "b2_metrics.json"
    write_json(
        path,
        {
            "schema_version": 1,
            "stopped": False,
            "entry_condition_met": bool(entry),
            "b1_recall_100_feh": b1_recall_100,
            "candidate_ceiling_recall": candidate_ceiling_recall,
            "b2_candidate_count": len(b2_rows),
            "first_fold_candidate_count": first_fold_count,
            "b2_candidates_path": str(directory / "scores_b2.jsonl"),
            "curve": curve,
            "curve_kind": "selection_union_excluding_ambiguous",
            "evaluation_curve": evaluation_curve_rows,
            "evaluation_curve_kind": "event_level_union_including_ambiguous",
            "selected_operating_points": selected_points,
            "threshold_transfer": transfer,
            "per_meeting_curves": per_meeting_curves,
            "availability": availability,
            "code_sha256": sha256_file(CODE_PATH),
            "config_sha256": sha256_file(CONFIG_PATH),
        },
    )
    return path


def prepare(root: Path) -> Path:
    cfg = config()
    sessions = _sessions(root)
    directory = output_root(root)
    directory.mkdir(parents=True, exist_ok=True)
    summary = {
        "scored_hours": sum(session.scored_hours for session in sessions.values()),
        "event_count": sum(len(session.events) for session in sessions.values()),
    }
    if not math.isclose(summary["scored_hours"], 4.731361111111111, abs_tol=1e-12):
        raise R9Error("R7-B exposure drifted")
    if summary["event_count"] != 4619:
        raise R9Error("R7-B reference count drifted")
    r8 = r8_output_root(root)
    reused_paths: dict[str, Path] = {
        "r8_config": r8 / "config.json",
        "r8_accuracy_metrics": r8 / "accuracy_metrics.json",
        "r8_input_inventory": r8 / "input_inventory.json",
        "r8_threshold_transfer_metrics": r8 / "threshold_transfer_metrics.json",
        "r8_artifact_inventory": r8 / "artifact_inventory.json",
        "r8_model_receipt": r8 / "model_receipt.json",
        "r7b_inventory": r7b_inventory_path(root),
    }
    for session_id in sessions:
        reused_paths[f"probabilities_cpu_{session_id}"] = (
            r8 / "probabilities" / "cpu" / f"{session_id}.npz"
        )
        reused_paths[f"speaker_segments_cpu_{session_id}"] = (
            r8 / "speaker_segments" / "cpu" / f"{session_id}.json"
        )
    for name, path in reused_paths.items():
        if not path.is_file():
            raise R9Error(f"reused R8/R7-B input is missing: {path}")
    reuse = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in reused_paths.items()
    }
    r8_artifact_inventory = load_json(reused_paths["r8_artifact_inventory"])
    expected_by_relative: dict[str, str] = {
        str(row["relative_path"]): str(row["sha256"]) for row in r8_artifact_inventory["artifacts"]
    }
    mismatches: list[str] = []
    for name, path in reused_paths.items():
        if name in {"r7b_inventory", "r8_artifact_inventory"}:
            continue
        relative = str(path.relative_to(r8)).replace("\\", "/")
        expected = expected_by_relative.get(relative)
        if expected is None:
            mismatches.append(f"{name}: absent from the R8 artifact inventory")
        elif reuse[name]["sha256"] != expected:
            mismatches.append(f"{name}: expected {expected}, got {reuse[name]['sha256']}")
    r8_input_inventory = load_json(reused_paths["r8_input_inventory"])
    if reuse["r7b_inventory"]["sha256"] != str(r8_input_inventory.get("r7b_inventory_sha256")):
        mismatches.append("r7b_inventory: hash differs from the R8 input inventory")
    r8_waveform_hashes = {
        str(row["session_id"]): str(row["waveform_sha256"])
        for row in r8_input_inventory["sessions"]
    }
    for session in sessions.values():
        expected = r8_waveform_hashes.get(session.session_id)
        if expected is None:
            mismatches.append(f"waveform {session.session_id}: absent from the R8 input inventory")
        elif session.waveform_sha256 != expected:
            mismatches.append(
                f"waveform {session.session_id}: computed {session.waveform_sha256}, R8 recorded {expected}"
            )
    if mismatches:
        raise R9Error(f"frozen R8/R7-B input verification failed: {mismatches}")
    document = {
        "schema_version": 1,
        "experiment_id": cfg["experiment_id"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "evidence_mode": cfg["evidence_mode"],
        "summary": summary,
        "git": _git_state(),
        "code_sha256": sha256_file(CODE_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "reused": reuse,
    }
    path = directory / "r8_reuse_inventory.json"
    write_json(path, document)
    write_json(directory / "config.json", cfg)
    print(json.dumps({"summary": summary, "reused_count": len(reuse)}, indent=2, sort_keys=True))
    return path


def _candidate_baseline_points(root: Path, sessions: dict[str, Session]) -> dict[str, Any]:
    directory = output_root(root)
    candidates = read_jsonl(directory / "candidates.jsonl")
    all_predictions: dict[str, list[int]] = defaultdict(list)
    grouped_positive: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        all_predictions[str(candidate["session_id"])].append(int(candidate["sample"]))
        if candidate["label"] == "positive":
            grouped_positive[str(candidate["session_id"])].append(
                {"sample": int(candidate["sample"]), "score": 1.0}
            )
    model_policy = _metrics(sessions, dict(all_predictions))
    oracle: dict[str, list[int]] = {}
    radius_samples = int(config()["duplicate_suppression_ms"]) * 16
    for session_id, rows in grouped_positive.items():
        oracle[session_id] = [sample for sample, _ in _grouped_events(rows, radius_samples)]
    oracle_metrics = _metrics(sessions, oracle)
    return {
        "model_policy": model_policy,
        "oracle": oracle_metrics,
    }


def ceiling_summary(root: Path) -> Path:
    cfg = config()
    sessions = _sessions(root)
    directory = output_root(root)
    baseline = _candidate_baseline_points(root, sessions)
    oracle_recall = float(baseline["oracle"]["tolerances"]["250"]["recall"] or 0.0)
    r8_accuracy = load_json(r8_output_root(root) / "accuracy_metrics.json")
    arms: dict[str, Any] = {}

    def require_current_identity(document: dict[str, Any], name: str) -> None:
        if document.get("code_sha256") != sha256_file(CODE_PATH) or document.get(
            "config_sha256"
        ) != sha256_file(CONFIG_PATH):
            raise R9Error(
                f"{name} was produced by a different harness/config; rerun the {name} action"
            )

    for name in ("a0", "a1", "a1_diagnostic", "b1", "b2"):
        path = directory / f"{name}_metrics.json"
        if name == "a0":
            if path.is_file():
                document = load_json(path)
                require_current_identity(document, "a0")
                recall = float(document["aggregate_recall_250"] or 0.0)
                feh = float(document["aggregate_false_events_per_hour"])
                at_targets = {
                    str(float(target)): (
                        {"recall_250": recall, "false_events_per_hour": feh, "threshold": None}
                        if feh <= float(target)
                        else None
                    )
                    for target in cfg["targets"]["false_events_per_hour"]
                }
                fractions: dict[str, Any] = {}
                for fraction in cfg["ceiling"]["candidate_recall_fractions"]:
                    fractions[str(fraction)] = (
                        {"recall_250": recall, "false_events_per_hour": feh}
                        if recall >= float(fraction) * oracle_recall
                        else None
                    )
                arms["a0"] = {
                    "kind": "point",
                    "recall_250": recall,
                    "false_events_per_hour": feh,
                    "metrics": document["aggregate_metrics"],
                    "at_targets": at_targets,
                    "candidate_ceiling_fractions": fractions,
                    "fold_selection": document["fold_selection"],
                }
            continue
        if name == "b2":
            if not path.is_file():
                continue
            b2_document = load_json(path)
            require_current_identity(b2_document, "b2")
            if bool(b2_document.get("stopped", True)):
                arms["b2"] = {
                    "kind": "stopped",
                    "stop_reason": b2_document.get("stop_reason"),
                    "candidate_count": b2_document.get("b2_candidate_count"),
                    "first_fold_candidate_count": b2_document.get("first_fold_candidate_count"),
                }
                continue
            curve = b2_document["evaluation_curve"]
            at_targets = {}
            for target in cfg["targets"]["false_events_per_hour"]:
                selected = _select_row(curve, float(target))
                at_targets[str(float(target))] = {
                    "threshold": selected["threshold"],
                    "recall_250": selected["recall_250"],
                    "false_events_per_hour": selected["false_events_per_hour"],
                }
            fractions = {}
            for fraction in cfg["ceiling"]["candidate_recall_fractions"]:
                threshold = float(fraction) * oracle_recall
                hit = None
                for row in sorted(curve, key=lambda value: float(value["false_events_per_hour"])):
                    if float(row["recall_250"] or 0.0) >= threshold:
                        hit = {
                            "false_events_per_hour": row["false_events_per_hour"],
                            "recall_250": row["recall_250"],
                            "threshold": row["threshold"],
                        }
                        break
                fractions[str(fraction)] = hit
            arms["b2"] = {
                "kind": "curve",
                "union_main_and_b2": True,
                "candidate_count": b2_document.get("b2_candidate_count"),
                "at_targets": at_targets,
                "candidate_ceiling_fractions": fractions,
                "curve": curve,
                "selection_curve": b2_document.get("curve", curve),
            }
            continue
        if not path.is_file():
            continue
        document = load_json(path)
        require_current_identity(document, name)
        curve = document["evaluation_curve"]
        selection_curve = document.get("curve", curve)
        at_targets: dict[str, Any] = {}
        for target in cfg["targets"]["false_events_per_hour"]:
            selected = _select_row(curve, float(target))
            at_targets[str(float(target))] = {
                "threshold": selected["threshold"],
                "recall_250": selected["recall_250"],
                "false_events_per_hour": selected["false_events_per_hour"],
            }
        fractions: dict[str, Any] = {}
        for fraction in cfg["ceiling"]["candidate_recall_fractions"]:
            threshold = float(fraction) * oracle_recall
            hit = None
            for row in sorted(curve, key=lambda value: float(value["false_events_per_hour"])):
                if float(row["recall_250"] or 0.0) >= threshold:
                    hit = {
                        "false_events_per_hour": row["false_events_per_hour"],
                        "recall_250": row["recall_250"],
                        "threshold": row["threshold"],
                    }
                    break
            fractions[str(fraction)] = hit
        arms[name] = {
            "kind": "curve",
            "at_targets": at_targets,
            "candidate_ceiling_fractions": fractions,
            "curve": curve,
            "selection_curve": selection_curve,
        }
    pareto: dict[str, Any] = {"points": {}, "meaningful": None}
    if "a1" in arms:
        r8_curve = r8_accuracy["curve"]
        counts = 0
        for target in cfg["outcome_a_pareto"]["comparison_points_feh"]:
            r8_best = _select_row(r8_curve, float(target))
            a1_best = arms["a1"]["at_targets"][str(float(target))]
            r8_recall = float(r8_best["recall_250"] or 0.0)
            a1_recall = float(a1_best["recall_250"] or 0.0)
            ratio = (
                a1_recall / r8_recall if r8_recall > 0.0 else (math.inf if a1_recall > 0.0 else 0.0)
            )
            pareto["points"][str(float(target))] = {
                "r8_recall_250": r8_recall,
                "a1_recall_250": a1_recall,
                "a1_evaluated_false_events_per_hour": a1_best["false_events_per_hour"],
                "ratio": ratio,
            }
            if ratio >= float(cfg["outcome_a_pareto"]["minimum_ratio"]):
                counts += 1
        pareto["meaningful"] = counts >= int(cfg["outcome_a_pareto"]["minimum_points"])
    embedding_contribution: dict[str, Any] = {"points": {}}
    if "a1" in arms and "b1" in arms:
        for target in cfg["outcome_a_pareto"]["comparison_points_feh"]:
            a1_best = arms["a1"]["at_targets"][str(float(target))]
            b1_best = arms["b1"]["at_targets"][str(float(target))]
            a1_recall = float(a1_best["recall_250"] or 0.0)
            b1_recall = float(b1_best["recall_250"] or 0.0)
            embedding_contribution["points"][str(float(target))] = {
                "a1_recall_250": a1_recall,
                "b1_recall_250": b1_recall,
                "b1_evaluated_false_events_per_hour": b1_best["false_events_per_hour"],
                "ratio": (
                    b1_recall / a1_recall
                    if a1_recall > 0.0
                    else (math.inf if b1_recall > 0.0 else 0.0)
                ),
            }
    embedding_contribution["raises_ceiling"] = bool(
        embedding_contribution["points"]
        and any(
            float(point["ratio"]) >= 1.05 for point in embedding_contribution["points"].values()
        )
    )
    document = {
        "schema_version": 1,
        "oracle_recall_250": oracle_recall,
        "baseline": {
            "model_policy": {
                "recall_250": baseline["model_policy"]["tolerances"]["250"]["recall"],
                "false_events_per_hour": baseline["model_policy"]["tolerances"]["250"][
                    "false_events_per_hour"
                ],
                "prediction_count": baseline["model_policy"]["prediction_count"],
            },
            "oracle": {
                "recall_250": oracle_recall,
                "false_events_per_hour": baseline["oracle"]["tolerances"]["250"][
                    "false_events_per_hour"
                ],
                "prediction_count": baseline["oracle"]["prediction_count"],
            },
        },
        "arms": arms,
        "outcome_a_pareto": pareto,
        "embedding_contribution": embedding_contribution,
        "code_sha256": sha256_file(CODE_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
    }
    path = directory / "ceiling_summary.json"
    write_json(path, document)
    _plot_ceiling(root, arms, r8_accuracy, baseline)
    _plot_timelines(root)
    return path


def _plot_timelines(root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = config()
    directory = output_root(root)
    a1_path = directory / "a1_metrics.json"
    if not a1_path.is_file():
        return
    a1 = load_json(a1_path)
    sessions = _sessions(root)
    timeline_dir = directory / "representative_timelines"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    point = a1["selected_operating_points"]["20.0"]
    metrics = point["metrics"]
    selected = list(metrics["matched_pairs"][:2]) + [
        [session_id, sample, sample] for session_id, sample in metrics["false_event_samples"][:2]
    ]
    frame_ms = int(cfg["frame_ms"])
    for index, (session_id, prediction, reference) in enumerate(selected):
        values = _load_probabilities(root, str(session_id))
        center_frame = int(round(int(prediction) / (frame_ms * 16)))
        start = max(0, center_frame - 50)
        end = min(len(values), center_frame + 51)
        x_seconds = np.arange(start, end) * (frame_ms / 1000.0)
        figure, axis = plt.subplots(figsize=(9, 4))
        for speaker in range(4):
            axis.plot(x_seconds, values[start:end, speaker], label=f"slot {speaker + 1}")
        axis.axhline(0.5, color="black", linestyle=":", label="0.5 candidate threshold")
        axis.axvline(int(prediction) / 16000.0, color="tab:red", linestyle="--", label="prediction")
        for event in sessions[str(session_id)].events:
            sample = int(event["sample"])
            if start * 1280 <= sample < end * 1280:
                axis.axvline(sample / 16000.0, color="tab:green", alpha=0.35)
        is_match = [session_id, prediction, reference] in metrics["matched_pairs"]
        axis.set_title(f"{session_id}: {'match' if is_match else 'false event'}")
        axis.set_xlabel("Source time (seconds)")
        axis.set_ylabel("Speaker activity probability")
        axis.set_ylim(0, 1)
        axis.legend(ncol=3, fontsize=8)
        figure.tight_layout()
        figure.savefig(timeline_dir / f"timeline_{index:02d}_{session_id}.png", dpi=150)
        plt.close(figure)


def _plot_ceiling(
    root: Path,
    arms: dict[str, Any],
    r8_accuracy: dict[str, Any],
    baseline: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = output_root(root)
    figure, axis = plt.subplots(figsize=(9, 6))
    r8_curve = [row for row in r8_accuracy["curve"] if float(row["false_events_per_hour"]) <= 120.0]
    axis.plot(
        [float(row["false_events_per_hour"]) for row in r8_curve],
        [float(row["recall_250"] or 0.0) for row in r8_curve],
        color="tab:gray",
        linewidth=1.0,
        label="R8 raw-probability curve (incumbent)",
    )
    colors = {
        "a1": "tab:blue",
        "a1_diagnostic": "tab:orange",
        "b1": "tab:green",
        "b2": "tab:purple",
    }
    labels = {
        "a1": "R9-A1 logistic verifier (probability-only)",
        "a1_diagnostic": "R9-A1 diagnostic (+480 ms window)",
        "b1": "R9-B1 logistic verifier (embedding-augmented)",
        "b2": "R9-B2 union stream (intra-slot extension, if run)",
    }
    for name in ("a1", "a1_diagnostic", "b1", "b2"):
        if name not in arms:
            continue
        if name == "b2" and arms[name].get("kind") != "curve":
            continue
        curve = [row for row in arms[name]["curve"] if float(row["false_events_per_hour"]) <= 120.0]
        axis.plot(
            [float(row["false_events_per_hour"]) for row in curve],
            [float(row["recall_250"] or 0.0) for row in curve],
            color=colors[name],
            linestyle="--" if name in ("a1_diagnostic", "b2") else "-",
            label=labels[name],
        )
    if "a0" in arms:
        axis.scatter(
            [float(arms["a0"]["false_events_per_hour"])],
            [float(arms["a0"]["recall_250"] or 0.0)],
            color="tab:green",
            marker="s",
            label="R9-A0 rule stack",
        )
    model_policy = baseline["model_policy"]["tolerances"]["250"]
    axis.annotate(
        f"model 0.5 policy (unfiltered):\n"
        f"recall {float(model_policy['recall'] or 0.0):.2f} @ "
        f"{float(model_policy['false_events_per_hour']):.0f} FE/h (off-scale)",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lavender", "alpha": 0.9},
    )
    axis.scatter(
        [0.0],
        [float(baseline["oracle"]["tolerances"]["250"]["recall"] or 0.0)],
        color="tab:red",
        marker="x",
        s=90,
        label="perfect-filter oracle (candidate ceiling)",
    )
    for target in (1, 5, 10, 20):
        axis.axvline(float(target), color="tab:gray", alpha=0.35, linewidth=0.6)
    axis.axhline(0.3, color="tab:red", linestyle=":", linewidth=1.0)
    axis.axhline(0.5, color="tab:red", linestyle=":", linewidth=1.0)
    axis.annotate(
        "inherited gate reference lines (context only):\n"
        "recall >= 0.3 @ 10 FE/h, recall >= 0.5 @ 20 FE/h",
        xy=(0.62, 0.13),
        xycoords="axes fraction",
        fontsize=7.5,
        color="tab:red",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    axis.set_xlabel("False events per source hour")
    axis.set_ylabel("Recall@250 ms")
    axis.set_xlim(0, 120)
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(directory / "recall_false_event_curves.png", dpi=160)
    plt.close(figure)


def report(root: Path) -> Path:
    directory = output_root(root)
    ceiling = load_json(directory / "ceiling_summary.json")
    candidate_summary = load_json(directory / "candidate_summary.json")
    a1 = load_json(directory / "a1_metrics.json")
    embedding_feature_names = list(
        config().get("embedding", {}).get("features", {}).get("feature_names", [])
    )
    a1_diagnostic_path = directory / "a1_diagnostic_metrics.json"
    a1_diagnostic = load_json(a1_diagnostic_path) if a1_diagnostic_path.is_file() else None
    a0 = (
        load_json(directory / "a0_metrics.json")
        if (directory / "a0_metrics.json").is_file()
        else None
    )
    lines = [
        "# R9 Sortformer Change-Verification Upper-Bound Report",
        "",
        "Evidence status: **development-known internal decision only**. The purpose of R9 is to",
        "measure the performance upper bound of a change-verification layer over Sortformer's own",
        "0.5-threshold candidate stream; the inherited R7-B/R8 gates are reference lines only and",
        "are not continuation criteria (owner instruction, 2026-08-13).",
        "",
        f"Candidate stream: {int(candidate_summary['candidate_count'])} segment starts "
        f"({float(candidate_summary['candidates_per_source_hour']):.1f} per source hour), "
        f"{int(candidate_summary['label_counts'].get('positive', 0))} positive / "
        f"{int(candidate_summary['label_counts'].get('ambiguous', 0))} ambiguous / "
        f"{int(candidate_summary['label_counts'].get('negative', 0))} negative; "
        f"candidate ceiling (perfect-filter oracle) Recall@250 = "
        f"{float(ceiling['oracle_recall_250']):.3f}.",
        "",
        "## Ceiling summary",
        "",
        "Target columns select the best event-level (evaluation) curve point with evaluated false",
        "events/hour at or below the target. Thresholds originate from the non-ambiguous selection",
        "curve (plan section 6); event-level evaluation includes ambiguous candidates. Selection and",
        "evaluation curves are both stored in the metrics artifacts.",
        "",
        "| Arm | 1 FE/h | 5 FE/h | 10 FE/h | 20 FE/h | 50 FE/h | 100 FE/h | FE/h at 50% of ceiling | FE/h at 80% of ceiling |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def cell(arm: dict[str, Any], target: str) -> str:
        point = arm["at_targets"].get(str(target))
        return f"{float(point['recall_250'] or 0.0):.3f}" if point else "—"

    def fraction_cell(arm: dict[str, Any], fraction: str) -> str:
        point = arm["candidate_ceiling_fractions"].get(str(fraction))
        return f"{float(point['false_events_per_hour']):.1f}" if point else "—"

    for name in ("a0", "a1", "a1_diagnostic", "b1", "b2"):
        if name == "a0":
            if "a0" not in ceiling["arms"]:
                continue
            arm = ceiling["arms"]["a0"]
            lines.append(
                f"| R9-A0 rule stack (single point) | {cell(arm, 1.0)} | {cell(arm, 5.0)} | {cell(arm, 10.0)} | "
                f"{cell(arm, 20.0)} | {cell(arm, 50.0)} | {cell(arm, 100.0)} | {fraction_cell(arm, 0.5)} | {fraction_cell(arm, 0.8)} |"
            )
            lines.append(
                f"|  (operating point: {float(arm['recall_250'] or 0.0):.3f} recall at "
                f"{float(arm['false_events_per_hour']):.1f} FE/h) | | | | | | | | |"
            )
            continue
        arm = ceiling["arms"].get(name)
        if arm is None:
            continue
        if name == "b2" and arm.get("kind") == "stopped":
            lines.append(
                f"| R9-B2 intra-slot extension | stopped: {arm.get('stop_reason')} | | | | | | | | |"
            )
            continue
        labels = {
            "a1": "R9-A1 logistic verifier (probability-only)",
            "a1_diagnostic": "R9-A1 diagnostic (non-causal)",
            "b1": "R9-B1 logistic verifier (embedding-augmented)",
            "b2": "R9-B2 union stream (main + intra-slot)",
        }
        lines.append(
            f"| {labels[name]} | {cell(arm, 1.0)} | {cell(arm, 5.0)} | {cell(arm, 10.0)} | {cell(arm, 20.0)} | "
            f"{cell(arm, 50.0)} | {cell(arm, 100.0)} | {fraction_cell(arm, 0.5)} | {fraction_cell(arm, 0.8)} |"
        )
    lines.extend(
        [
            "",
            "## Reference points",
            "",
            f"Model 0.5 policy (unfiltered): Recall@250 {float(ceiling['baseline']['model_policy']['recall_250'] or 0.0):.3f} "
            f"at {float(ceiling['baseline']['model_policy']['false_events_per_hour']):.1f} false events/hour "
            f"({int(ceiling['baseline']['model_policy']['prediction_count'])} predictions).",
            f"Perfect-filter oracle: Recall@250 {float(ceiling['oracle_recall_250']):.3f} at "
            f"{float(ceiling['baseline']['oracle']['false_events_per_hour']):.2f} false events/hour "
            f"({int(ceiling['baseline']['oracle']['prediction_count'])} predictions).",
            "",
            "## Verifier detail",
            "",
            f"Verifier form: **{a1['verifier_form']}** (MLP fallback triggered: {a1['mlp_triggered']}); "
            f"mean out-of-fold AUROC {float(a1['mean_out_of_fold_auroc']):.4f}.",
        ]
    )
    for fold, value in sorted(a1["fold_auroc"].items()):
        lines.append(f"- fold {fold} held-out AUROC: {float(value):.4f}")
    if a0 is not None:
        lines.extend(["", "R9-A0 rule stack selections per fold:", ""])
        for fold, selection in sorted(a0["fold_selection"].items(), key=lambda pair: int(pair[0])):
            lines.append(
                f"- fold {fold}: peak >= {float(selection['selected_peak_probability_threshold']):.2f}, "
                f"persistence >= {int(selection['selected_persistence_ms_threshold'])} ms, "
                f"gap <= {float(selection['selected_gap_ms_max']):.0f} ms, "
                f"co-activity >= {float(selection['selected_co_activity_min']):.2f}, "
                f"same-slot resume excluded"
            )
    lines.extend(
        [
            "",
            "## Inherited-gate reference lines (context only, not continuation criteria)",
            "",
            f"R9-A1 at the <=10 FE/h selection point (threshold chosen on non-ambiguous candidates; "
            f"evaluated {float(a1['reference_gate']['targets']['10.0']['false_events_per_hour']):.2f} FE/h): "
            f"Recall@250 {float(a1['reference_gate']['targets']['10.0']['recall_250'] or 0.0):.3f} "
            f"(reference gate: >= 0.3).",
            f"R9-A1 at the <=20 FE/h selection point (threshold chosen on non-ambiguous candidates; "
            f"evaluated {float(a1['reference_gate']['targets']['20.0']['false_events_per_hour']):.2f} FE/h): "
            f"Recall@250 {float(a1['reference_gate']['targets']['20.0']['recall_250'] or 0.0):.3f} "
            f"(reference gate: >= 0.5).",
            f"20 FE/h stratum recall — overlap onset "
            f"{float(a1['reference_gate']['targets']['20.0']['stratum_recall'].get('overlap_onset') or 0.0):.3f}, "
            f"silence-gap change "
            f"{float(a1['reference_gate']['targets']['20.0']['stratum_recall'].get('silence_gap_change') or 0.0):.3f}, "
            f"maximum single-meeting TP share "
            f"{float(a1['reference_gate']['targets']['20.0']['maximum_meeting_true_positive_share']):.3f}.",
            f"20 FE/h short-return recall "
            f"{float(a1['selected_operating_points']['20.0']['metrics']['short_return_recall'] or 0.0):.3f}.",
        ]
    )
    if "a1" in ceiling["arms"] and ceiling["outcome_a_pareto"]["meaningful"] is not None:
        pareto = ceiling["outcome_a_pareto"]
        verdict = (
            "meaningfully Pareto-dominates the R8 raw curve"
            if pareto["meaningful"]
            else "does NOT meaningfully Pareto-dominate the R8 raw curve"
        )
        lines.extend(
            [
                "",
                "## Outcome-A diagnostic",
                "",
                f"The R9-A1 curve **{verdict}** (twofold recall improvement at "
                f"{int(config()['outcome_a_pareto']['minimum_points'])} of the "
                f"{config()['outcome_a_pareto']['comparison_points_feh']} false-events/hour points).",
            ]
        )
        for target, point in pareto["points"].items():
            lines.append(
                f"- {target} FE/h: R8 {float(point['r8_recall_250']):.4f} vs R9-A1 "
                f"{float(point['a1_recall_250']):.4f} (evaluated "
                f"{float(point['a1_evaluated_false_events_per_hour']):.1f} FE/h, ratio {float(point['ratio']):.2f})"
            )
    if "b1" in ceiling["arms"]:
        b1_doc = load_json(directory / "b1_metrics.json")
        lines.extend(
            [
                "",
                "## R9-B1 (embedding-augmented verifier)",
                "",
                f"Verifier form: **{b1_doc['verifier_form']}** (MLP fallback triggered: {b1_doc['mlp_triggered']}); "
                f"mean out-of-fold AUROC {float(b1_doc['mean_out_of_fold_auroc']):.4f}. "
                "B1 adds the four speaker-cache embedding features "
                f"({', '.join(embedding_feature_names)}) to the A1 features.",
            ]
        )
        for fold, value in sorted(b1_doc["fold_auroc"].items()):
            lines.append(f"- fold {fold} held-out AUROC: {float(value):.4f}")
    if ceiling["embedding_contribution"]["points"]:
        contribution = ceiling["embedding_contribution"]
        lines.extend(
            [
                "",
                "## Embedding contribution (A1 vs B1)",
                "",
                "The comparison isolates how much the speaker-cache embeddings raise the ceiling over",
                "probability-only features.",
            ]
        )
        for target, point in contribution["points"].items():
            lines.append(
                f"- {target} FE/h: A1 {float(point['a1_recall_250']):.4f} vs B1 "
                f"{float(point['b1_recall_250']):.4f} (evaluated "
                f"{float(point['b1_evaluated_false_events_per_hour']):.1f} FE/h, ratio {float(point['ratio']):.2f})"
            )
        lines.append(
            f"- Embeddings raise the ceiling: **{bool(contribution['raises_ceiling'])}** "
            f"(frozen threshold: B1/A1 ratio >= 1.05 at any comparison point)."
        )
    b2_doc_path = directory / "b2_metrics.json"
    if b2_doc_path.is_file():
        b2_doc = load_json(b2_doc_path)
        if bool(b2_doc.get("stopped", True)):
            lines.extend(
                [
                    "",
                    "## R9-B2 (intra-slot candidate extension)",
                    "",
                    f"Stopped: **{b2_doc.get('stop_reason')}** "
                    f"(first-fold candidate count {b2_doc.get('first_fold_candidate_count')}).",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "## R9-B2 (intra-slot candidate extension)",
                    "",
                    f"Ran with {int(b2_doc.get('b2_candidate_count', 0))} intra-slot candidates; the B2 row in the "
                    "ceiling table scores the union stream (main candidates + intra-slot candidates) with the "
                    "B1 verifier.",
                ]
            )
    if a1_diagnostic is not None:
        lines.extend(
            [
                "",
                "## Confirmation-lag diagnostic",
                "",
                "The R9-A1 diagnostic repeats the learned verifier with the frozen +480 ms confirmation",
                "window instead of the base +240 ms window. It is non-causal and shows how much of the",
                "ceiling depends on waiting.",
            ]
        )
    lines.extend(
        [
            "",
            "## Availability latency (R9-A1)",
            "",
            f"Confirmation window: {int(a1['availability_latency']['confirmation_ms'])} ms; "
            f"verification compute: {float(a1['availability_latency']['verification_compute_ms_per_event']):.3f} ms per event. "
            "Boundary timestamps stay at the 0.5 candidate crossing; availability adds the fixed "
            "confirmation window plus verifier compute. "
            f"Latency defect threshold (R8 0.99-confirmation signature): median >= "
            f"{int(a1['availability_latency']['latency_defect_median_ms'])} ms.",
            "",
            "| Target FE/h | Boundary lag p50 ms | Boundary lag p90 ms | Availability p50 ms | Availability p90 ms | Defect |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for target in config()["targets"]["false_events_per_hour"]:
        point = a1["availability_latency"]["per_target"][str(float(target))]
        lines.append(
            f"| {target} | {_latency_cell(point['boundary_lag_ms'], 'p50')} | "
            f"{_latency_cell(point['boundary_lag_ms'], 'p90')} | "
            f"{_latency_cell(point['availability_ms'], 'p50')} | "
            f"{_latency_cell(point['availability_ms'], 'p90')} | "
            f"{'yes' if point.get('latency_defect') else 'no'} |"
        )
    b1_metrics_path = directory / "b1_metrics.json"
    if b1_metrics_path.is_file():
        b1_doc = load_json(b1_metrics_path)
        lines.extend(
            [
                "",
                "## Availability latency (R9-B1)",
                "",
                f"Embedding after-window horizon: "
                f"{int(b1_doc['availability_latency']['availability_frame_horizon_ms'])} ms; "
                f"verification compute: "
                f"{float(b1_doc['availability_latency']['verification_compute_ms_per_event']):.3f} ms per event.",
                "",
                "| Target FE/h | Boundary lag p50 ms | Boundary lag p90 ms | Availability p50 ms | Availability p90 ms | Defect |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for target in config()["targets"]["false_events_per_hour"]:
            point = b1_doc["availability_latency"]["per_target"][str(float(target))]
            lines.append(
                f"| {target} | {_latency_cell(point['boundary_lag_ms'], 'p50')} | "
                f"{_latency_cell(point['boundary_lag_ms'], 'p90')} | "
                f"{_latency_cell(point['availability_ms'], 'p50')} | "
                f"{_latency_cell(point['availability_ms'], 'p90')} | "
                f"{'yes' if point.get('latency_defect') else 'no'} |"
            )
        b1_gate = b1_doc.get("reference_gate", {}).get("targets", {})
        if "20.0" in b1_gate:
            point_20 = b1_gate["20.0"]
            lines.extend(
                [
                    "",
                    "## R9-B1 at the <=20 FE/h selection point",
                    "",
                    f"Evaluated {float(point_20['false_events_per_hour']):.2f} FE/h; "
                    f"Recall@250 {float(point_20['recall_250'] or 0.0):.3f}; "
                    f"stratum recall — overlap onset "
                    f"{float(point_20['stratum_recall'].get('overlap_onset') or 0.0):.3f}, "
                    f"silence-gap change "
                    f"{float(point_20['stratum_recall'].get('silence_gap_change') or 0.0):.3f}; "
                    f"maximum single-meeting TP share "
                    f"{float(point_20['maximum_meeting_true_positive_share']):.3f}.",
                ]
            )
        b1_selected = b1_doc.get("selected_operating_points", {}).get("20.0")
        if b1_selected:
            lines.append(
                f"20 FE/h short-return recall "
                f"{float(b1_selected['metrics'].get('short_return_recall') or 0.0):.3f}."
            )
        b1_per_meeting = b1_doc.get("per_meeting_curves", {})
        if b1_per_meeting:
            selected_threshold = float(
                b1_doc.get("selected_operating_points", {}).get("20.0", {}).get("threshold", 0.0)
            )
            lines.extend(
                [
                    "",
                    "## R9-B1 per-meeting at the <=20 FE/h threshold",
                    "",
                    "| Meeting | Predictions | FE/h | Recall@250 |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )
            for session_id in sorted(b1_per_meeting):
                rows = [
                    row
                    for row in b1_per_meeting[session_id]
                    if abs(float(row["threshold"]) - selected_threshold) < 1e-6
                ]
                if not rows:
                    continue
                row = rows[0]
                lines.append(
                    f"| {session_id} | {int(row['prediction_count'])} | "
                    f"{float(row['false_events_per_hour']):.1f} | "
                    f"{float(row['recall_250'] or 0.0):.3f} |"
                )
    b2_metrics_path = directory / "b2_metrics.json"
    if b2_metrics_path.is_file():
        b2_doc = load_json(b2_metrics_path)
        if not bool(b2_doc.get("stopped", True)):
            lines.extend(
                [
                    "",
                    "## Availability latency (R9-B2 union)",
                    "",
                    f"Embedding after-window horizon: "
                    f"{int(b2_doc['availability']['availability_frame_horizon_ms'])} ms; "
                    f"verification compute per event — B1 "
                    f"{float(b2_doc['availability'].get('verification_compute_ms_per_event_b1') or 0.0):.3f} ms, "
                    f"B2 "
                    f"{float(b2_doc['availability'].get('verification_compute_ms_per_event_b2') or 0.0):.3f} ms "
                    f"(applied to the corresponding emitted events).",
                    "",
                    "| Target FE/h | Boundary lag p50 ms | Boundary lag p90 ms | Availability p50 ms | Availability p90 ms | Defect |",
                    "| ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for target in config()["targets"]["false_events_per_hour"]:
                point = b2_doc["availability"]["per_target"][str(float(target))]
                lines.append(
                    f"| {target} | {_latency_cell(point['boundary_lag_ms'], 'p50')} | "
                    f"{_latency_cell(point['boundary_lag_ms'], 'p90')} | "
                    f"{_latency_cell(point['availability_ms'], 'p50')} | "
                    f"{_latency_cell(point['availability_ms'], 'p90')} | "
                    f"{'yes' if point.get('latency_defect') else 'no'} |"
                )
    transfer = a1.get("threshold_transfer", [])
    if transfer:
        lines.extend(
            [
                "",
                "## Threshold transfer (R9-A1)",
                "",
                "| Fold | Target FE/h | Dev FE/h | Dev recall | Held-out FE/h | Held-out recall |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in transfer:
            lines.append(
                f"| {int(row['fold'])} | {int(row['target_false_events_per_hour'])} | "
                f"{float(row['development_false_events_per_hour']):.1f} | "
                f"{float(row['development_recall_250'] or 0.0):.3f} | "
                f"{float(row['held_out_false_events_per_hour']):.1f} | "
                f"{float(row['held_out_recall_250'] or 0.0):.3f} |"
            )
    if b1_metrics_path.is_file():
        b1_transfer = load_json(b1_metrics_path).get("threshold_transfer", [])
        if b1_transfer:
            lines.extend(
                [
                    "",
                    "## Threshold transfer (R9-B1)",
                    "",
                    "| Fold | Target FE/h | Dev FE/h | Dev recall | Held-out FE/h | Held-out recall |",
                    "| ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in b1_transfer:
                lines.append(
                    f"| {int(row['fold'])} | {int(row['target_false_events_per_hour'])} | "
                    f"{float(row['development_false_events_per_hour']):.1f} | "
                    f"{float(row['development_recall_250'] or 0.0):.3f} | "
                    f"{float(row['held_out_false_events_per_hour']):.1f} | "
                    f"{float(row['held_out_recall_250'] or 0.0):.3f} |"
                )
    if b2_metrics_path.is_file():
        b2_doc = load_json(b2_metrics_path)
        if not bool(b2_doc.get("stopped", True)):
            b2_transfer = b2_doc.get("threshold_transfer", [])
            if b2_transfer:
                lines.extend(
                    [
                        "",
                        "## Threshold transfer (R9-B2 union)",
                        "",
                        "| Fold | Target FE/h | Dev FE/h | Dev recall | Held-out FE/h | Held-out recall |",
                        "| ---: | ---: | ---: | ---: | ---: | ---: |",
                    ]
                )
                for row in b2_transfer:
                    lines.append(
                        f"| {int(row['fold'])} | {int(row['target_false_events_per_hour'])} | "
                        f"{float(row['development_false_events_per_hour']):.1f} | "
                        f"{float(row['development_recall_250'] or 0.0):.3f} | "
                        f"{float(row['held_out_false_events_per_hour']):.1f} | "
                        f"{float(row['held_out_recall_250'] or 0.0):.3f} |"
                    )
            point_20 = b2_doc.get("selected_operating_points", {}).get("20.0")
            if point_20:
                stratum = point_20["metrics"]["stratum_recall"]
                lines.extend(
                    [
                        "",
                        "## R9-B2 union at the <=20 FE/h selection point",
                        "",
                        f"Evaluated {float(point_20['metrics']['tolerances']['250']['false_events_per_hour']):.2f} FE/h; "
                        f"Recall@250 {float(point_20['metrics']['tolerances']['250']['recall'] or 0.0):.3f}; "
                        f"stratum recall — overlap onset {float(stratum.get('overlap_onset') or 0.0):.3f}, "
                        f"silence-gap change {float(stratum.get('silence_gap_change') or 0.0):.3f}, "
                        f"short-return {float(stratum.get('short_backchannel_or_return') or 0.0):.3f}; "
                        f"maximum single-meeting TP share "
                        f"{float(point_20['metrics']['maximum_meeting_true_positive_share']):.3f}.",
                    ]
                )
            b2_per_meeting = b2_doc.get("per_meeting_curves", {})
            if b2_per_meeting:
                selected_threshold = float(
                    b2_doc.get("selected_operating_points", {})
                    .get("20.0", {})
                    .get("threshold", 0.0)
                )
                lines.extend(
                    [
                        "",
                        "## R9-B2 union per-meeting at the <=20 FE/h threshold",
                        "",
                        "| Meeting | Predictions | FE/h | Recall@250 |",
                        "| --- | ---: | ---: | ---: |",
                    ]
                )
                for session_id in sorted(b2_per_meeting):
                    rows = [
                        row
                        for row in b2_per_meeting[session_id]
                        if abs(float(row["threshold"]) - selected_threshold) < 1e-6
                    ]
                    if not rows:
                        continue
                    row = rows[0]
                    lines.append(
                        f"| {session_id} | {int(row['prediction_count'])} | "
                        f"{float(row['false_events_per_hour']):.1f} | "
                        f"{float(row['recall_250'] or 0.0):.3f} |"
                    )
    outcome_lines = ["", "## Outcome", ""]
    pareto = ceiling.get("outcome_a_pareto") or {}
    contribution = ceiling.get("embedding_contribution", {})
    if bool(pareto.get("meaningful")):
        outcome = "A"
        outcome_lines.extend(
            [
                "**Selected predeclared outcome: Outcome A — probability-only verification raises",
                "the ceiling.** The R9-A1 curve meaningfully Pareto-dominates the R8 raw curve across",
                "the low-false-event region per the frozen configuration.",
            ]
        )
    elif bool(contribution.get("raises_ceiling", False)):
        outcome = "B"
        outcome_lines.extend(
            [
                "**Selected predeclared outcome: Outcome B — embeddings carry the ceiling.**",
                "A1 does not dominate meaningfully, but the embedding-augmented B1 verifier does.",
            ]
        )
    elif any(
        bool(arm.get("curve")) or "metrics" in arm
        for arm in ceiling.get("arms", {}).values()
        if isinstance(arm, dict)
    ):
        outcome = "C"
        outcome_lines.extend(
            [
                "**Selected predeclared outcome: Outcome C — neither arm moves the curve.**",
                "Sortformer's own candidate stream is not separable at low false-event rates with",
                "these features; the measured ceiling is the honest product.",
            ]
        )
    else:
        outcome = "D"
        outcome_lines.extend(
            [
                "**Selected predeclared outcome: Outcome D — invalid or inconclusive.**",
                "No valid curve completed under the execution ceilings.",
            ]
        )
    if (directory / "b1_metrics.json").is_file():
        b1_doc = load_json(directory / "b1_metrics.json")
        outcome_lines.extend(
            [
                "The A1-vs-B1 comparison answers whether the speaker-cache embeddings raise the",
                "ceiling over probability-only features "
                f"(raises_ceiling: {bool(contribution.get('raises_ceiling', False))}).",
                f"B1 mean out-of-fold AUROC {float(b1_doc.get('mean_out_of_fold_auroc', float('nan'))):.4f} "
                f"(A1: {float(a1.get('mean_out_of_fold_auroc', float('nan'))):.4f}).",
            ]
        )
        b2_path = directory / "b2_metrics.json"
        if b2_path.is_file():
            b2_doc = load_json(b2_path)
            if bool(b2_doc.get("stopped", True)):
                outcome_lines.append(
                    f"R9-B2 stopped ({b2_doc.get('stop_reason')}); the R9-B ceiling is reported without it."
                )
            else:
                outcome_lines.append(
                    "R9-B2 ran; the B2 row in the ceiling table is the union-stream ceiling with the "
                    "B1 verifier."
                )
        else:
            outcome_lines.append(
                "R9-B2 was not executed: it runs only under its frozen entry condition."
            )
    outcome_lines.extend(
        [
            "No outcome authorizes follow-up work, integration, or publication automatically.",
            "",
            "## Required next decision",
            "",
            f"Outcome {outcome} was selected per plan section 11. Per plan section 15 and the owner's",
            "execution discipline, R9 ends here with a next-decision request: whether to approve",
            "freezing a new untouched panel for confirmatory interpretation of the measured ceiling",
            "(and which arm would carry forward), or to stop R9 without follow-up. No replay,",
            "integration, merge, or publication proceeds before that owner decision.",
        ]
    )
    lines.extend(outcome_lines)
    path = directory / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifact_inventory(root)
    return path


def _latency_cell(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    return "—" if value is None else f"{float(value):.0f}"


def artifact_inventory(root: Path) -> Path:
    output = output_root(root)
    path = output / "artifact_inventory.json"
    rows = [
        {
            "relative_path": str(item.relative_to(output)).replace("\\", "/"),
            "size_bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in sorted(output.rglob("*"))
        if item.is_file() and item != path
    ]
    write_json(
        path,
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "artifact_count": len(rows),
            "artifacts": rows,
        },
    )
    return path


def smoke() -> dict[str, Any]:
    cfg = config()
    probabilities = np.asarray(
        [
            [0.9, 0.1, 0.1, 0.1],
            [0.9, 0.1, 0.1, 0.1],
            [0.9, 0.9, 0.1, 0.1],
            [0.1, 0.9, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.9, 0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    starts = _segment_start_candidates(probabilities, 80)
    if [row["start_frame"] for row in starts] != [0, 2, 6]:
        raise R9Error(f"segment start extraction failed: {starts}")
    features = _features_for_candidate(
        probabilities, 2, 1, int(cfg["confirmation"]["base_frames"]), cfg["feature_windows"], 80
    )
    if features["gap_ms"] != 0.0:
        raise R9Error(f"overlap gap expected 0: {features}")
    if features["same_slot_resume"] != 0.0:
        raise R9Error("fresh slot must not be a resume")
    if features["persistence_ms"] != 160.0:
        raise R9Error(f"persistence expected 160: {features}")
    resume_features = _features_for_candidate(
        probabilities, 6, 0, int(cfg["confirmation"]["base_frames"]), cfg["feature_windows"], 80
    )
    if resume_features["same_slot_resume"] != 1.0:
        raise R9Error("slot 0 at frame 6 must be a resume")
    if resume_features["gap_ms"] != 160.0:
        raise R9Error(f"gap expected 160: {resume_features}")
    grouped = _grouped_events(
        [
            {"sample": 100, "score": 0.4},
            {"sample": 4000, "score": 0.9},
            {"sample": 150, "score": 0.8},
        ],
        3200,
    )
    if grouped != [(150, 0.8), (4000, 0.9)]:
        raise R9Error(f"grouping failed: {grouped}")
    matched, false, misses = _one_to_one([100, 900, 1300], [120, 910], 50)
    if len(matched) != 2 or false != [1300] or misses:
        raise R9Error("one-to-one matcher smoke failed")
    labels = np.asarray([1, 0, 1, 0, -1], dtype=np.int8)
    weights = _balance_weights(labels[labels != -1])
    if not math.isclose(float(weights.sum()), float(len(weights))):
        raise R9Error("balance weights failed")
    matrix = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    y = np.asarray([1, 0, 1, 0], dtype=np.int8)
    coef, intercept = _fit_linear(matrix, y, np.ones(4, dtype=np.float32), cfg["verifier"])
    scores = _linear_predict(matrix, coef, intercept)
    if not np.isfinite(scores).all():
        raise R9Error("linear fit produced non-finite scores")
    return {
        "segment_start_frames": [row["start_frame"] for row in starts],
        "grouped_events": grouped,
        "code_sha256": sha256_file(CODE_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "smoke",
            "prepare",
            "candidates",
            "features",
            "a0",
            "a1",
            "a1-diagnostic",
            "ceiling",
            "report",
            "r9b-prepare",
            "r9b-fixture",
            "r9b-replay",
            "r9b-extract",
            "b1",
            "b2",
        ),
    )
    args = parser.parse_args(argv)
    if args.action == "smoke":
        print(json.dumps(smoke(), indent=2, sort_keys=True))
        return 0
    root = cache_root()
    if args.action == "prepare":
        print(prepare(root))
    elif args.action == "candidates":
        print(extract_candidates(root))
    elif args.action == "features":
        for path in extract_features(root):
            print(path)
    elif args.action == "a0":
        print(run_a0(root))
    elif args.action == "a1":
        print(run_a1(root, diagnostic=False))
    elif args.action == "a1-diagnostic":
        print(run_a1(root, diagnostic=True))
    elif args.action == "ceiling":
        print(ceiling_summary(root))
    elif args.action == "report":
        print(report(root))
    elif args.action == "r9b-prepare":
        print(r9b_prepare(root))
    elif args.action == "r9b-fixture":
        print(r9b_fixture(root))
    elif args.action == "r9b-replay":
        print(r9b_replay(root))
    elif args.action == "r9b-extract":
        print(extract_embedding_features(root))
    elif args.action == "b1":
        print(run_b1(root))
    elif args.action == "b2":
        print(run_b2(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
