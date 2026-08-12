from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

SAMPLES_PER_MS = 16
REPORTING_TOLERANCES_MS = (100, 250, 500, 750, 1000, 1500)


def detect_probability_events(
    probabilities: Sequence[float],
    frontiers: Sequence[int],
    threshold: float,
    confirmation_hops: int,
    minimum_separation_hops: int = 2,
) -> list[dict[str, int | float]]:
    values = np.asarray(probabilities, dtype=np.float64)
    frontier_values = np.asarray(frontiers, dtype=np.int64)
    if values.ndim != 1 or frontier_values.ndim != 1 or values.shape != frontier_values.shape:
        raise ValueError("probabilities and frontiers must be equal one-dimensional arrays")
    if confirmation_hops < 1:
        raise ValueError("confirmation_hops must be positive")
    events: list[dict[str, int | float]] = []
    index = 0
    while index < len(values):
        if not np.isfinite(values[index]) or values[index] < threshold:
            index += 1
            continue
        emit_index = index + confirmation_hops - 1
        if emit_index >= len(values):
            break
        span = values[index : emit_index + 1]
        if not np.all(np.isfinite(span)) or not np.all(span >= threshold):
            index += 1
            continue
        if events and emit_index - int(events[-1]["emit_hop"]) < minimum_separation_hops:
            index = emit_index + 1
            continue
        events.append(
            {
                "onset_hop": int(index),
                "onset_sample": int(frontier_values[index]),
                "emit_hop": int(emit_index),
                "emit_sample": int(frontier_values[emit_index]),
                "confidence": float(np.min(span)),
            }
        )
        index = emit_index + 1
        while index < len(values) and np.isfinite(values[index]) and values[index] >= threshold:
            index += 1
    return events


def causal_match_events(
    ground_truth_samples: Sequence[int],
    events: Sequence[dict[str, Any]],
    tolerance_ms: int = 500,
) -> list[dict[str, Any]]:
    tolerance_samples = tolerance_ms * SAMPLES_PER_MS
    unmatched_gt = sorted(int(value) for value in ground_truth_samples)
    matched: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda row: (int(row["emit_sample"]), int(row["onset_sample"]))):
        emit = int(event["emit_sample"])
        onset = int(event["onset_sample"])
        candidates: list[tuple[int, int, int]] = []
        for index, gt in enumerate(unmatched_gt):
            if emit < gt:
                continue
            localization_error = onset - gt
            if abs(localization_error) <= tolerance_samples:
                candidates.append((abs(localization_error), gt, index))
        if not candidates:
            continue
        _, gt, best_index = min(candidates)
        unmatched_gt.pop(best_index)
        matched.append(
            {
                "onset_sample": onset,
                "emit_sample": emit,
                "availability_sample": emit,
                "ground_truth_sample": gt,
                "localization_error_ms": (onset - gt) / SAMPLES_PER_MS,
                "availability_latency_ms": (emit - gt) / SAMPLES_PER_MS,
            }
        )
    return matched


def event_metrics(
    matched: Sequence[dict[str, Any]],
    total_events: int,
    ground_truth_count: int,
    source_hours: float,
) -> dict[str, Any]:
    false_events = max(0, total_events - len(matched))
    latencies = np.asarray(
        [float(row["availability_latency_ms"]) for row in matched], dtype=np.float64
    )
    errors = np.asarray([float(row["localization_error_ms"]) for row in matched], dtype=np.float64)

    def within(tolerance_ms: int) -> dict[str, float | None]:
        inside = sum(abs(float(row["localization_error_ms"])) <= tolerance_ms for row in matched)
        recall = inside / ground_truth_count if ground_truth_count else None
        precision = inside / total_events if total_events else None
        f1 = None
        if precision is not None and recall is not None and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        return {"precision": precision, "recall": recall, "f1": f1}

    latency = {
        "median_ms": float(np.median(latencies)) if len(latencies) else None,
        "p90_ms": float(np.quantile(latencies, 0.9)) if len(latencies) else None,
        "p95_ms": float(np.quantile(latencies, 0.95)) if len(latencies) else None,
    }
    localization = {
        "median_ms": float(np.median(errors)) if len(errors) else None,
        "mean_abs_ms": float(np.mean(np.abs(errors))) if len(errors) else None,
    }
    boundary = {f"at_{value}ms": within(value) for value in REPORTING_TOLERANCES_MS}
    matched_at = {
        tolerance: sum(
            abs(float(row["localization_error_ms"])) <= tolerance for row in matched
        )
        for tolerance in REPORTING_TOLERANCES_MS
    }
    return {
        "matched_count": len(matched),
        "ground_truth_count": ground_truth_count,
        "total_confirmed_events": total_events,
        "false_event_count": false_events,
        "false_events_per_hour": false_events / source_hours if source_hours > 0 else None,
        "strict_false_event_count_at_500ms": max(0, total_events - matched_at[500]),
        "strict_false_events_per_hour_at_500ms": (
            max(0, total_events - matched_at[500]) / source_hours
            if source_hours > 0
            else None
        ),
        "source_hours": source_hours,
        "recall_within_500ms": boundary["at_500ms"]["recall"],
        "recall_within_750ms": boundary["at_750ms"]["recall"],
        "recall_within_1000ms": boundary["at_1000ms"]["recall"],
        "recall_within_1500ms": boundary["at_1500ms"]["recall"],
        "late_detection_fraction_500_to_1000ms": (
            (matched_at[1000] - matched_at[500]) / ground_truth_count
            if ground_truth_count
            else None
        ),
        "late_detection_fraction_1000_to_1500ms": (
            (matched_at[1500] - matched_at[1000]) / ground_truth_count
            if ground_truth_count
            else None
        ),
        "f1_at_250ms": boundary["at_250ms"]["f1"],
        "boundary_f1": boundary,
        "availability_latency_ms": latency,
        "signed_localization_error_ms": localization,
        "missed_change_rate_at_500ms": (
            1 - matched_at[500] / ground_truth_count if ground_truth_count else None
        ),
        "missed_change_rate_at_1000ms": (
            1 - matched_at[1000] / ground_truth_count if ground_truth_count else None
        ),
        "missed_change_rate_at_1500ms": (
            1 - matched_at[1500] / ground_truth_count if ground_truth_count else None
        ),
    }


def select_operating_point(
    rows: Sequence[dict[str, Any]], false_event_budget_per_hour: float
) -> dict[str, Any]:
    if not rows:
        raise ValueError("operating-point rows are empty")
    feasible = [
        row
        for row in rows
        if row["metrics"]["false_events_per_hour"] is not None
        and float(row["metrics"]["false_events_per_hour"]) <= false_event_budget_per_hour
    ]

    def value(row: dict[str, Any], key: str) -> float:
        raw = row["metrics"].get(key)
        return float(raw) if raw is not None and np.isfinite(float(raw)) else 0.0

    if feasible:
        return min(
            feasible,
            key=lambda row: (
                -value(row, "f1_at_250ms"),
                -value(row, "recall_within_500ms"),
                float(row["metrics"]["false_events_per_hour"]),
                float(row["metrics"]["availability_latency_ms"]["median_ms"] or 1e18),
                str(row["config_id"]),
            ),
        )
    return min(
        rows,
        key=lambda row: (
            float(row["metrics"]["false_events_per_hour"] or 1e18),
            -value(row, "f1_at_250ms"),
            -value(row, "recall_within_500ms"),
            float(row["metrics"]["availability_latency_ms"]["median_ms"] or 1e18),
            str(row["config_id"]),
        ),
    )
