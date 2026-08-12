from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

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


def _feature_matrix(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    cfg = config()
    names = list(cfg["feature_names"])
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
    }
    path = directory / "a0_metrics.json"
    write_json(path, document)
    return path


def run_a1(root: Path, diagnostic: bool) -> Path:
    cfg = config()
    sessions = _sessions(root)
    directory = output_root(root)
    feature_name = "features_a_diagnostic" if diagnostic else "features_a"
    output_name = "a1_diagnostic_metrics" if diagnostic else "a1_metrics"
    rows = read_jsonl(directory / f"{feature_name}.jsonl")
    matrix = _feature_matrix(rows)
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
        mean, scale = _standardize_fit(matrix[train_mask])
        standardized = _standardize(matrix, mean, scale)
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
    if diagnostic:
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
            mean, scale = _standardize_fit(matrix[train_mask])
            standardized = _standardize(matrix, mean, scale)
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
    confirmation_ms = int(
        cfg["confirmation"]["diagnostic_frames" if diagnostic else "base_frames"]
    ) * int(cfg["frame_ms"])
    defect_threshold_ms = float(cfg["reference_gate"]["latency_defect_median_ms"])
    verification_compute_ms = max(predict_seconds / len(rows) * 1000.0, 0.0)
    availability_latency: dict[str, Any] = {
        "confirmation_ms": confirmation_ms,
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
            float(value) + confirmation_ms + verification_compute_ms for value in deltas
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
        "diagnostic": diagnostic,
        "feature_name": feature_name,
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
    transfer_path = (
        directory / "threshold_transfer_metrics.json"
        if not diagnostic
        else directory / "threshold_transfer_diagnostic_metrics.json"
    )
    write_json(transfer_path, transfer)
    write_jsonl(
        directory / f"scores_{'a1_diagnostic' if diagnostic else 'a1'}.jsonl",
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
    for name in ("a0", "a1", "a1_diagnostic"):
        path = directory / f"{name}_metrics.json"
        if name == "a0":
            if path.is_file():
                document = load_json(path)
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
        if not path.is_file():
            continue
        document = load_json(path)
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
    colors = {"a1": "tab:blue", "a1_diagnostic": "tab:orange"}
    labels = {
        "a1": "R9-A1 logistic verifier (causal)",
        "a1_diagnostic": "R9-A1 diagnostic (+480 ms window)",
    }
    for name in ("a1", "a1_diagnostic"):
        if name not in arms:
            continue
        curve = [row for row in arms[name]["curve"] if float(row["false_events_per_hour"]) <= 120.0]
        axis.plot(
            [float(row["false_events_per_hour"]) for row in curve],
            [float(row["recall_250"] or 0.0) for row in curve],
            color=colors[name],
            linestyle="--" if name == "a1_diagnostic" else "-",
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

    for name in ("a0", "a1", "a1_diagnostic"):
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
        label = "R9-A1 logistic verifier" if name == "a1" else "R9-A1 diagnostic (non-causal)"
        lines.append(
            f"| {label} | {cell(arm, 1.0)} | {cell(arm, 5.0)} | {cell(arm, 10.0)} | {cell(arm, 20.0)} | "
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
    lines.extend(
        [
            "",
            "## Outcome",
            "",
            "R9 measured the probability-only verification ceiling. The B arms (speaker-cache",
            "embedding features) were not run: they require a separate owner decision naming them",
            "after these A-arm results (plan section 15). No result authorizes follow-up work,",
            "integration, or publication automatically.",
        ]
    )
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
