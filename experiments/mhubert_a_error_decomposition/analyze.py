from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiments.online_speaker_memory_handoff.protocol import read_jsonl, sha256_file
from experiments.psem_trainable_formulation_gate import run as gate
from experiments.speaker_representation_scd.r7b_local_segmentation import (
    HOP_SAMPLES,
    SAMPLE_RATE,
    STATE_OVERLAP,
    STATE_SILENCE,
    STATE_SINGLE,
    _one_to_one,
)

MODEL_ID = "mhubert-147"
ARM = "A-FROZEN-DIRECT"
TOLERANCES_MS = (100, 250, 500)


def _local_maxima(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["session_id"])].append(row)
    result: list[dict[str, Any]] = []
    for session_rows in grouped.values():
        session_rows.sort(key=lambda row: int(row["boundary_sample"]))
        for index, row in enumerate(session_rows):
            previous = (
                float(session_rows[index - 1]["score"])
                if index > 0
                and int(row["boundary_sample"])
                - int(session_rows[index - 1]["boundary_sample"])
                <= HOP_SAMPLES
                else -math.inf
            )
            following = (
                float(session_rows[index + 1]["score"])
                if index + 1 < len(session_rows)
                and int(session_rows[index + 1]["boundary_sample"])
                - int(row["boundary_sample"])
                <= HOP_SAMPLES
                else -math.inf
            )
            if float(row["score"]) >= previous and float(row["score"]) >= following:
                result.append(row)
    return sorted(result, key=lambda row: -float(row["score"]))


def _matches(bundles, predictions, tolerance_ms: int):
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["session_id"])].append(int(row["boundary_sample"]))
    matched: list[tuple[str, int, int]] = []
    false: list[tuple[str, int]] = []
    misses: list[tuple[str, int]] = []
    for session_id, bundle in bundles.items():
        references = [int(event["sample"]) for event in bundle.events]
        local_matched, local_false, local_misses = _one_to_one(
            grouped.get(session_id, []), references, tolerance_ms * 16
        )
        matched.extend((session_id, prediction, reference) for prediction, reference in local_matched)
        false.extend((session_id, prediction) for prediction in local_false)
        misses.extend((session_id, reference) for reference in local_misses)
    return matched, false, misses


def _timing_summary(matches: Sequence[tuple[str, int, int]]) -> dict[str, Any]:
    errors = np.asarray([(prediction - reference) / 16.0 for _, prediction, reference in matches])
    edges = np.arange(-500.0, 501.0, 100.0)
    counts, _ = np.histogram(errors, bins=edges)
    return {
        "count": len(errors),
        "signed_error_ms": {
            "mean": float(np.mean(errors)),
            "median": float(np.median(errors)),
            "p10": float(np.quantile(errors, 0.10)),
            "p90": float(np.quantile(errors, 0.90)),
        },
        "absolute_error_ms": {
            "median": float(np.median(np.abs(errors))),
            "p90": float(np.quantile(np.abs(errors), 0.90)),
        },
        "histogram_100ms": [
            {"from_ms": int(edges[index]), "to_ms": int(edges[index + 1]), "count": int(count)}
            for index, count in enumerate(counts)
        ],
    }


def _coverage(bundles, peaks, tolerance_ms: int) -> dict[str, int | float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in peaks:
        grouped[str(row["session_id"])].append(int(row["boundary_sample"]))
    covered = 0
    total = 0
    radius = tolerance_ms * 16
    for session_id, bundle in bundles.items():
        predictions = grouped.get(session_id, [])
        for event in bundle.events:
            reference = int(event["sample"])
            total += 1
            covered += int(any(abs(prediction - reference) <= radius for prediction in predictions))
    return {"reference_count": total, "covered_count": covered, "missing_count": total - covered, "coverage": covered / total}


def _category(bundle, prediction: int) -> dict[str, Any]:
    center = int(np.searchsorted(bundle.frontiers, prediction))
    if center >= len(bundle.frontiers) or int(bundle.frontiers[center]) != prediction:
        center = max(0, min(center - 1, len(bundle.frontiers) - 2))
    right = min(center + 1, len(bundle.state) - 1)
    left_state = int(bundle.state[center])
    right_state = int(bundle.state[right])
    left_speaker = int(bundle.speaker[center])
    right_speaker = int(bundle.speaker[right])
    if left_state != STATE_OVERLAP and right_state == STATE_OVERLAP:
        name = "overlap_onset_without_nearby_event"
    elif left_state == STATE_OVERLAP and right_state != STATE_OVERLAP:
        name = "overlap_end"
    elif left_state == STATE_OVERLAP and right_state == STATE_OVERLAP:
        name = "overlap_continuation"
    elif left_state == STATE_SINGLE and right_state == STATE_SINGLE:
        name = (
            "singleton_different_speaker_without_nearby_event"
            if left_speaker != right_speaker
            else "continuous_same_speaker_singleton"
        )
    elif left_state == STATE_SILENCE and right_state == STATE_SINGLE:
        source = center - 1
        while source >= 0 and int(bundle.state[source]) == STATE_SILENCE:
            source -= 1
        if source >= 0 and int(bundle.state[source]) == STATE_SINGLE:
            name = (
                "same_speaker_pause_resume"
                if int(bundle.speaker[source]) == right_speaker
                else "different_speaker_after_silence_without_nearby_event"
            )
        else:
            name = "speech_onset_after_silence_unknown_speaker"
    elif left_state == STATE_SINGLE and right_state == STATE_SILENCE:
        name = "speech_offset_to_silence"
    elif left_state == STATE_SILENCE and right_state == STATE_SILENCE:
        name = "silence_continuation"
    else:
        name = "other_state_transition"
    return {
        "category": name,
        "left_state": left_state,
        "right_state": right_state,
        "left_speaker": left_speaker,
        "right_speaker": right_speaker,
    }


def analyze() -> dict[str, Any]:
    root = gate.cache_root()
    cfg = gate.config()
    bundles = gate._load_bundles(root, require_pcm=False)
    prediction_path = gate._arm_paths(root, MODEL_ID, ARM)["predictions"]
    rows = read_jsonl(prediction_path)
    raw_maxima = _local_maxima(rows)
    peaks = gate._peaks(rows, int(cfg["duplicate_suppression_ms"]) * 16)
    _, _, frontier = gate._curve_and_points(bundles, rows, cfg)
    point = frontier["best_macro_f1_operating_point"]
    threshold = float(point["threshold"])
    selected_raw = [row for row in raw_maxima if float(row["score"]) >= threshold]
    selected = [row for row in peaks if float(row["score"]) >= threshold]
    match_sets: dict[int, dict[str, Any]] = {}
    for tolerance in TOLERANCES_MS:
        matched, false, misses = _matches(bundles, selected, tolerance)
        match_sets[tolerance] = {"matched": matched, "false": false, "misses": misses}
    matched_predictions = {
        tolerance: {(session_id, prediction) for session_id, prediction, _ in values["matched"]}
        for tolerance, values in match_sets.items()
    }
    matched_references = {
        tolerance: {(session_id, reference) for session_id, _, reference in values["matched"]}
        for tolerance, values in match_sets.items()
    }
    selected_keys = {(str(row["session_id"]), int(row["boundary_sample"])) for row in selected}
    promotions = {
        "prediction_100fp_to_250tp": len((selected_keys - matched_predictions[100]) & matched_predictions[250]),
        "prediction_100fp_to_500tp": len((selected_keys - matched_predictions[100]) & matched_predictions[500]),
        "prediction_250fp_to_500tp": len((selected_keys - matched_predictions[250]) & matched_predictions[500]),
        "reference_miss_100_to_hit_250": len(matched_references[250] - matched_references[100]),
        "reference_miss_100_to_hit_500": len(matched_references[500] - matched_references[100]),
        "reference_miss_250_to_hit_500": len(matched_references[500] - matched_references[250]),
    }
    raw_by_session: dict[str, list[int]] = defaultdict(list)
    selected_by_session: dict[str, list[int]] = defaultdict(list)
    for row in selected_raw:
        raw_by_session[str(row["session_id"])].append(int(row["boundary_sample"]))
    for row in selected:
        selected_by_session[str(row["session_id"])].append(int(row["boundary_sample"]))
    multiple_raw = 0
    multiple_nms = 0
    excess_raw = 0
    excess_nms = 0
    for session_id, bundle in bundles.items():
        for event in bundle.events:
            reference = int(event["sample"])
            raw_count = sum(abs(value - reference) <= 8000 for value in raw_by_session.get(session_id, []))
            nms_count = sum(abs(value - reference) <= 8000 for value in selected_by_session.get(session_id, []))
            multiple_raw += int(raw_count > 1)
            multiple_nms += int(nms_count > 1)
            excess_raw += max(raw_count - 1, 0)
            excess_nms += max(nms_count - 1, 0)
    false_500 = match_sets[500]["false"]
    proximal_false: list[tuple[str, int]] = []
    remote_false: list[tuple[str, int, float]] = []
    for session_id, prediction in false_500:
        distance = min(abs(prediction - int(event["sample"])) for event in bundles[session_id].events) / 16.0
        if distance <= 500.0:
            proximal_false.append((session_id, prediction))
        else:
            remote_false.append((session_id, prediction, distance))
    score_by_key = {
        (str(row["session_id"]), int(row["boundary_sample"])): float(row["score"])
        for row in selected
    }
    categories: Counter[str] = Counter()
    sessions: Counter[str] = Counter()
    scores_by_category: dict[str, list[float]] = defaultdict(list)
    remote_rows: list[dict[str, Any]] = []
    for session_id, prediction, distance in remote_false:
        category = _category(bundles[session_id], prediction)
        categories[category["category"]] += 1
        sessions[session_id] += 1
        scores_by_category[category["category"]].append(score_by_key[(session_id, prediction)])
        remote_rows.append(
            {
                "session_id": session_id,
                "prediction_sample": prediction,
                "prediction_seconds": prediction / SAMPLE_RATE,
                "score": score_by_key[(session_id, prediction)],
                "nearest_gt_distance_ms": distance,
                **category,
            }
        )
    remote_rows.sort(key=lambda row: -float(row["score"]))
    event_maps = {
        session_id: {int(event["sample"]): event for event in bundle.events}
        for session_id, bundle in bundles.items()
    }
    strata: dict[str, Any] = {}
    all_strata = sorted({str(event["stratum"]) for bundle in bundles.values() for event in bundle.events})
    for stratum in all_strata:
        total = sum(str(event["stratum"]) == stratum for bundle in bundles.values() for event in bundle.events)
        strata[stratum] = {
            str(tolerance): {
                "matched": sum(
                    str(event_maps[session_id][reference]["stratum"]) == stratum
                    for session_id, _, reference in match_sets[tolerance]["matched"]
                ),
                "reference_count": total,
            }
            for tolerance in TOLERANCES_MS
        }
        for tolerance in TOLERANCES_MS:
            strata[stratum][str(tolerance)]["recall"] = (
                strata[stratum][str(tolerance)]["matched"] / total if total else None
            )
    return {
        "schema_version": 1,
        "experiment": "mhubert_a_error_decomposition_v1",
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "input": {
            "model_id": MODEL_ID,
            "arm": ARM,
            "prediction_path": str(prediction_path),
            "prediction_sha256": sha256_file(prediction_path),
            "prediction_row_count": len(rows),
            "reference_count": sum(len(bundle.events) for bundle in bundles.values()),
            "duplicate_suppression_ms": int(cfg["duplicate_suppression_ms"]),
        },
        "operating_point": {
            "selection": "maximum mean F1 across 100/250/500 ms",
            "threshold": threshold,
            "macro_f1": float(point["macro_f1"]),
            "selected_peak_count": len(selected),
            "metrics": {
                tolerance: point["metrics"]["tolerances"][str(tolerance)]
                for tolerance in TOLERANCES_MS
            },
        },
        "timing_error_at_500ms_match": _timing_summary(match_sets[500]["matched"]),
        "collar_promotions": promotions,
        "duplicate_analysis": {
            "selected_raw_local_maxima_count": len(selected_raw),
            "selected_after_200ms_nms_count": len(selected),
            "removed_by_200ms_nms": len(selected_raw) - len(selected),
            "gt_with_multiple_selected_raw_peaks_within_500ms": multiple_raw,
            "raw_excess_peaks_within_500ms": excess_raw,
            "gt_with_multiple_selected_nms_peaks_within_500ms": multiple_nms,
            "nms_excess_peaks_within_500ms": excess_nms,
            "evaluated_500ms_false_peaks_within_500ms_of_any_gt": len(proximal_false),
        },
        "remote_false_positives_at_500ms": {
            "count": len(remote_false),
            "share_of_500ms_false_predictions": len(remote_false) / len(false_500),
            "gt_state_categories": dict(categories.most_common()),
            "session_counts": dict(sessions.most_common()),
            "score_by_category": {
                category: {
                    "count": len(scores),
                    "median": float(np.median(scores)),
                    "p90": float(np.quantile(scores, 0.90)),
                }
                for category, scores in sorted(scores_by_category.items())
            },
            "unobservable_without_audio_annotation": ["laughter", "prosody_change"],
            "highest_scoring_examples": remote_rows[:30],
        },
        "threshold_independent_candidate_coverage": {
            "raw_local_maxima": {
                str(tolerance): _coverage(bundles, raw_maxima, tolerance)
                for tolerance in TOLERANCES_MS
            },
            "after_200ms_nms": {
                str(tolerance): _coverage(bundles, peaks, tolerance)
                for tolerance in TOLERANCES_MS
            },
        },
        "event_stratum_recall_at_operating_point": strata,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
