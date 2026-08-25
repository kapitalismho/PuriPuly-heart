from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

import numpy as np

from experiments.psem_relative_occupancy_gate.contracts import (
    AnchorEpisode,
    AnchorLifecycle,
    RelativeState,
)
from experiments.psem_relative_occupancy_gate.decoder import GTSessionResult, TimelineSpan
from experiments.psem_relative_occupancy_gate.io_utils import percentile

STATE_ORDER = (
    RelativeState.NONE,
    RelativeState.ANCHOR_ONLY,
    RelativeState.ANCHOR_PLUS_OTHER,
    RelativeState.OTHER_ONLY,
)


def _validated_vectors(
    labels: Iterable[bool | int],
    scores: Iterable[float],
    weights: Iterable[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(list(labels), dtype=np.bool_)
    p = np.asarray(list(scores), dtype=np.float64)
    w = np.asarray(list(weights), dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or w.shape != y.shape:
        raise ValueError("metric vectors must have identical one-dimensional geometry")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("scores must be finite probabilities")
    if not np.isfinite(w).all() or np.any(w < 0.0):
        raise ValueError("weights must be finite and non-negative")
    return y, p, w


def weighted_binary_confusion(
    labels: Iterable[bool | int],
    scores: Iterable[float],
    weights: Iterable[float],
    threshold: float,
) -> dict[str, float]:
    y, p, w = _validated_vectors(labels, scores, weights)
    predicted = p >= threshold
    tp = float(w[y & predicted].sum())
    fp = float(w[~y & predicted].sum())
    fn = float(w[y & ~predicted].sum())
    tn = float(w[~y & ~predicted].sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "tp_weight": tp,
        "fp_weight": fp,
        "fn_weight": fn,
        "tn_weight": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def weighted_binary_pr_curve(
    labels: Iterable[bool | int],
    scores: Iterable[float],
    weights: Iterable[float],
    thresholds: Iterable[float],
) -> list[dict[str, float]]:
    values = sorted({float(value) for value in thresholds})
    if not values or values[0] < 0.0 or values[-1] > 1.0:
        raise ValueError("thresholds must be a non-empty subset of [0, 1]")
    label_values = list(labels)
    score_values = list(scores)
    weight_values = list(weights)
    return [
        weighted_binary_confusion(label_values, score_values, weight_values, value)
        for value in values
    ]


def weighted_average_precision(
    labels: Iterable[bool | int],
    scores: Iterable[float],
    weights: Iterable[float],
) -> float | None:
    y, p, w = _validated_vectors(labels, scores, weights)
    positive_weight = float(w[y].sum())
    if positive_weight == 0.0:
        return None
    order = np.argsort(-p, kind="stable")
    y = y[order]
    p = p[order]
    w = w[order]
    cumulative_tp = np.cumsum(w * y)
    cumulative_fp = np.cumsum(w * ~y)
    group_ends = np.flatnonzero(np.r_[p[1:] != p[:-1], True])
    tp = cumulative_tp[group_ends]
    fp = cumulative_fp[group_ends]
    recall = tp / positive_weight
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall_steps = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_steps * precision))


def state_from_presence(anchor_present: bool, other_present: bool) -> RelativeState:
    if anchor_present and other_present:
        return RelativeState.ANCHOR_PLUS_OTHER
    if anchor_present:
        return RelativeState.ANCHOR_ONLY
    if other_present:
        return RelativeState.OTHER_ONLY
    return RelativeState.NONE


def weighted_relative_state_confusion(
    anchor_labels: Iterable[bool | int],
    other_labels: Iterable[bool | int],
    anchor_scores: Iterable[float],
    other_scores: Iterable[float],
    weights: Iterable[float],
    anchor_threshold: float,
    other_threshold: float,
) -> dict[str, Any]:
    anchor_y, anchor_p, w = _validated_vectors(anchor_labels, anchor_scores, weights)
    other_y, other_p, other_w = _validated_vectors(other_labels, other_scores, weights)
    if not np.array_equal(w, other_w):
        raise ValueError("anchor and other metric weights differ")
    matrix = np.zeros((len(STATE_ORDER), len(STATE_ORDER)), dtype=np.float64)
    state_index = {state: index for index, state in enumerate(STATE_ORDER)}
    for anchor_truth, other_truth, anchor_score, other_score, weight in zip(
        anchor_y, other_y, anchor_p, other_p, w, strict=True
    ):
        truth = state_from_presence(bool(anchor_truth), bool(other_truth))
        predicted = state_from_presence(
            bool(anchor_score >= anchor_threshold), bool(other_score >= other_threshold)
        )
        matrix[state_index[truth], state_index[predicted]] += weight
    per_state: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for index, state in enumerate(STATE_ORDER):
        tp = float(matrix[index, index])
        fp = float(matrix[:, index].sum() - tp)
        fn = float(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_state[state.value] = {"precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)
    return {
        "anchor_threshold": float(anchor_threshold),
        "other_threshold": float(other_threshold),
        "state_order": [state.value for state in STATE_ORDER],
        "confusion_weight": matrix.tolist(),
        "per_state": per_state,
        "macro_f1": float(np.mean(f1_values)),
    }


def timeline_exposure(
    timeline: Sequence[TimelineSpan], sample_rate_hz: int = 16000
) -> dict[str, Any]:
    state_samples: Counter[str] = Counter()
    lifecycle_samples: Counter[str] = Counter()
    masked_samples = 0
    masked_active_speech_samples = 0
    unanchored_active_speech_samples = 0
    uncertain_active_speech_samples = 0
    for span in timeline:
        duration = span.end_sample - span.start_sample
        lifecycle_samples[span.lifecycle.value] += duration
        if span.masked:
            masked_samples += duration
            if span.speech_present:
                masked_active_speech_samples += duration
        elif span.state is not None:
            state_samples[span.state.value] += duration
        elif span.speech_present and span.lifecycle is AnchorLifecycle.UNANCHORED:
            unanchored_active_speech_samples += duration
        elif span.speech_present and span.lifecycle is AnchorLifecycle.ANCHOR_UNCERTAIN:
            uncertain_active_speech_samples += duration
    exact_other = state_samples[RelativeState.OTHER_ONLY.value]
    fail_closed_unknown = (
        masked_active_speech_samples
        + unanchored_active_speech_samples
        + uncertain_active_speech_samples
    )
    return {
        "state_seconds": {
            state.value: state_samples[state.value] / sample_rate_hz for state in STATE_ORDER
        },
        "lifecycle_seconds": {
            key: value / sample_rate_hz for key, value in sorted(lifecycle_samples.items())
        },
        "masked_seconds": masked_samples / sample_rate_hz,
        "masked_active_speech_seconds": masked_active_speech_samples / sample_rate_hz,
        "unanchored_active_speech_seconds": unanchored_active_speech_samples / sample_rate_hz,
        "anchor_uncertain_active_speech_seconds": uncertain_active_speech_samples / sample_rate_hz,
        "fail_closed_unknown_active_speech_seconds": fail_closed_unknown / sample_rate_hz,
        "exclusive_other_contamination_upper_bound_seconds": (
            exact_other + unanchored_active_speech_samples + uncertain_active_speech_samples
        )
        / sample_rate_hz,
    }


def _episode_replacement_expectation(
    timeline: Sequence[TimelineSpan],
    episode: AnchorEpisode,
    confirmation_samples: int,
) -> tuple[int, int] | None:
    boundary: int | None = None
    evidence = 0
    for span in timeline:
        start = max(span.start_sample, episode.anchor_emit_sample)
        end = min(span.end_sample, episode.end_emit_sample)
        if end <= start or span.lifecycle is not AnchorLifecycle.ANCHORED:
            continue
        if span.anchor_id != episode.anchor_speaker:
            boundary = None
            evidence = 0
            continue
        if span.masked:
            continue
        if span.state is not RelativeState.OTHER_ONLY:
            boundary = None
            evidence = 0
            continue
        if boundary is None:
            boundary = start
        needed = confirmation_samples - evidence
        duration = end - start
        if duration >= needed:
            return boundary, start + needed
        evidence += duration
    return None


def audit_gt_session(result: GTSessionResult) -> dict[str, Any]:
    events_by_episode: dict[str, list[Any]] = {}
    for event in result.events:
        events_by_episode.setdefault(event.anchor_episode_id, []).append(event)
    rows = []
    errors = []
    episode_ids = {episode.episode_id for episode in result.episodes}
    for episode in result.episodes:
        expected = _episode_replacement_expectation(
            result.timeline,
            episode,
            result.confirmation_samples,
        )
        observed = events_by_episode.get(episode.episode_id, [])
        if len(observed) > 1:
            errors.append(f"{episode.episode_id}:multiple_events")
        event = observed[0] if len(observed) == 1 else None
        row = {
            "anchor_episode_id": episode.episode_id,
            "expected_boundary_source_sample": expected[0] if expected else None,
            "expected_qualification_sample": expected[1] if expected else None,
            "observed_boundary_source_sample": (
                event.boundary_source_sample if event is not None else None
            ),
            "observed_model_evidence_frontier_sample": (
                event.model_evidence_frontier_sample if event is not None else None
            ),
            "observed_decoder_emit_sample": (
                event.decoder_emit_sample if event is not None else None
            ),
        }
        rows.append(row)
        if expected is None and event is not None:
            errors.append(f"{episode.episode_id}:unexpected_event")
        elif expected is not None and event is None:
            errors.append(f"{episode.episode_id}:missing_event")
        elif expected is not None and event is not None:
            boundary, qualification = expected
            if event.boundary_source_sample != boundary:
                errors.append(f"{episode.episode_id}:boundary_mismatch")
            if event.model_evidence_frontier_sample != qualification:
                errors.append(f"{episode.episode_id}:frontier_mismatch")
            if event.decoder_emit_sample != qualification:
                errors.append(f"{episode.episode_id}:emit_mismatch")
            if event.confirmation_samples != result.confirmation_samples:
                errors.append(f"{episode.episode_id}:confirmation_mismatch")
    for episode_id in sorted(set(events_by_episode) - episode_ids):
        errors.append(f"{episode_id}:unknown_episode")
    return {"episodes": rows, "errors": errors, "passed": not errors}


def gate0_session_metrics(result: GTSessionResult, sample_rate_hz: int = 16000) -> dict[str, Any]:
    exposure = timeline_exposure(result.timeline, sample_rate_hz)
    audit = audit_gt_session(result)
    wall_delays_ms = [
        (event.decoder_emit_sample - event.boundary_source_sample) * 1000.0 / sample_rate_hz
        for event in result.events
    ]
    evidence_delays_ms = [
        (event.model_evidence_frontier_sample - event.boundary_source_sample)
        * 1000.0
        / sample_rate_hz
        for event in result.events
    ]
    boundary_errors = [
        int(row["observed_boundary_source_sample"]) - int(row["expected_boundary_source_sample"])
        for row in audit["episodes"]
        if row["observed_boundary_source_sample"] is not None
        and row["expected_boundary_source_sample"] is not None
    ]
    return {
        "source_id": result.source_id,
        "confirmation_ms": result.confirmation_samples * 1000.0 / sample_rate_hz,
        "speaker_induced_cut_count": len(result.events),
        "anchor_episode_count": len(result.episodes),
        "anchor_enrollment_count": len(result.enrollments),
        "exclusive_other_contamination_seconds": exposure["state_seconds"][
            RelativeState.OTHER_ONLY.value
        ],
        "exclusive_other_contamination_upper_bound_seconds": exposure[
            "exclusive_other_contamination_upper_bound_seconds"
        ],
        "fail_closed_unknown_active_speech_seconds": exposure[
            "fail_closed_unknown_active_speech_seconds"
        ],
        "replacement_emit_delay_ms": {
            "p50": percentile(wall_delays_ms, 50),
            "p90": percentile(wall_delays_ms, 90),
        },
        "model_evidence_delay_ms": {
            "p50": percentile(evidence_delays_ms, 50),
            "p90": percentile(evidence_delays_ms, 90),
        },
        "backdated_boundary_error_ms": {
            "p50": percentile(boundary_errors, 50),
            "p90": percentile(boundary_errors, 90),
        },
        "boundary_backdating_exact": audit["passed"]
        and all(
            event.boundary_source_sample < event.decoder_emit_sample for event in result.events
        ),
        "boundary_audit": audit,
        "exposure": exposure,
    }


def aggregate_gate0_metrics(
    session_metrics: Sequence[dict[str, Any]],
    active_speech_seconds: float,
) -> dict[str, Any]:
    cuts = sum(int(row["speaker_induced_cut_count"]) for row in session_metrics)
    contamination = sum(
        float(row["exclusive_other_contamination_seconds"]) for row in session_metrics
    )
    contamination_upper_bound = sum(
        float(row["exclusive_other_contamination_upper_bound_seconds"]) for row in session_metrics
    )
    fail_closed_unknown = sum(
        float(row["fail_closed_unknown_active_speech_seconds"]) for row in session_metrics
    )
    active_hours = active_speech_seconds / 3600.0
    return {
        "source_count": len(session_metrics),
        "active_speech_hours": active_hours,
        "speaker_induced_cut_count": cuts,
        "speaker_induced_cut_count_per_active_speech_hour": cuts / active_hours
        if active_hours
        else None,
        "exclusive_other_contamination_seconds": contamination,
        "exclusive_other_contamination_seconds_per_active_speech_hour": contamination / active_hours
        if active_hours
        else None,
        "exclusive_other_contamination_upper_bound_seconds": contamination_upper_bound,
        "exclusive_other_contamination_upper_bound_seconds_per_active_speech_hour": contamination_upper_bound
        / active_hours
        if active_hours
        else None,
        "fail_closed_unknown_active_speech_seconds": fail_closed_unknown,
        "boundary_backdating_exact": all(
            bool(row["boundary_backdating_exact"]) for row in session_metrics
        ),
    }


def monotonic_boundary_matches(
    predicted_samples: Sequence[int],
    reference_samples: Sequence[int],
    tolerance_samples: int,
) -> list[tuple[int, int]]:
    if tolerance_samples < 0:
        raise ValueError("tolerance_samples must be non-negative")
    predicted = [int(value) for value in predicted_samples]
    reference = [int(value) for value in reference_samples]
    if predicted != sorted(predicted) or reference != sorted(reference):
        raise ValueError("boundary sequences must be sorted")
    rows = len(predicted)
    columns = len(reference)
    counts = np.zeros((rows + 1, columns + 1), dtype=np.int32)
    costs = np.zeros((rows + 1, columns + 1), dtype=np.int64)
    choices = np.zeros((rows + 1, columns + 1), dtype=np.int8)
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            candidates = [
                (int(counts[row - 1, column]), int(costs[row - 1, column]), 1),
                (int(counts[row, column - 1]), int(costs[row, column - 1]), 2),
            ]
            delta = abs(predicted[row - 1] - reference[column - 1])
            if delta <= tolerance_samples:
                candidates.append(
                    (
                        int(counts[row - 1, column - 1]) + 1,
                        int(costs[row - 1, column - 1]) + delta,
                        3,
                    )
                )
            best = min(candidates, key=lambda value: (-value[0], value[1], -value[2]))
            counts[row, column] = best[0]
            costs[row, column] = best[1]
            choices[row, column] = best[2]
    matches: list[tuple[int, int]] = []
    row = rows
    column = columns
    while row and column:
        choice = int(choices[row, column])
        if choice == 3:
            matches.append((row - 1, column - 1))
            row -= 1
            column -= 1
        elif choice == 1:
            row -= 1
        else:
            column -= 1
    matches.reverse()
    return matches
