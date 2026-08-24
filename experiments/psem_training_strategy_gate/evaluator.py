from __future__ import annotations

import math
from bisect import bisect_left, bisect_right, insort
from dataclasses import dataclass
from numbers import Real
from typing import Any, Iterable, Sequence

SAMPLE_RATE_HZ = 16000
DUPLICATE_SUPPRESSION_SAMPLES = 3200
MATCHING_COLLARS_MS = (100, 250, 500)


class MetricContractError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PredictionScore:
    source_id: str
    boundary_sample: int
    observed_frontier_sample: int
    score: float


@dataclass(frozen=True, slots=True)
class CandidateEvent:
    source_id: str
    boundary_sample: int
    observed_frontier_sample: int
    score: float


@dataclass(frozen=True, slots=True)
class ReferenceEvent:
    source_id: str
    source_sample: int
    topology: str


def _sample(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _score(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _valid_prediction(row: PredictionScore | CandidateEvent) -> bool:
    return (
        isinstance(row.source_id, str)
        and bool(row.source_id)
        and _sample(row.boundary_sample)
        and row.boundary_sample % (SAMPLE_RATE_HZ // 10) == 0
        and _sample(row.observed_frontier_sample)
        and row.observed_frontier_sample == row.boundary_sample + SAMPLE_RATE_HZ
        and _score(row.score)
    )


def _valid_reference(row: ReferenceEvent) -> bool:
    return (
        isinstance(row.source_id, str)
        and bool(row.source_id)
        and _sample(row.source_sample)
        and isinstance(row.topology, str)
        and bool(row.topology)
    )


def _local_maxima(rows: Sequence[PredictionScore]) -> list[PredictionScore]:
    maxima: list[PredictionScore] = []
    for index, row in enumerate(rows):
        previous = rows[index - 1].score if index else float("-inf")
        following = rows[index + 1].score if index + 1 < len(rows) else float("-inf")
        if row.score > previous and row.score >= following:
            maxima.append(row)
    return maxima


def eventize(scores: Iterable[PredictionScore]) -> tuple[CandidateEvent, ...]:
    by_source: dict[str, list[PredictionScore]] = {}
    for row in scores:
        if not _valid_prediction(row):
            raise MetricContractError("prediction score row violates the metric contract")
        by_source.setdefault(row.source_id, []).append(row)
    retained: list[CandidateEvent] = []
    for source_id, rows in sorted(by_source.items()):
        rows.sort(key=lambda row: row.boundary_sample)
        if len({row.boundary_sample for row in rows}) != len(rows):
            raise MetricContractError("prediction centers must be unique within each source")
        if any(
            right.boundary_sample - left.boundary_sample != SAMPLE_RATE_HZ // 10
            for left, right in zip(rows, rows[1:], strict=False)
        ):
            raise MetricContractError("prediction centers must be a contiguous 100 ms sequence")
        candidates = sorted(
            _local_maxima(rows),
            key=lambda row: (-row.score, row.boundary_sample),
        )
        retained_samples: list[int] = []
        retained_by_sample: dict[int, PredictionScore] = {}
        for row in candidates:
            left = bisect_left(
                retained_samples,
                row.boundary_sample - DUPLICATE_SUPPRESSION_SAMPLES,
            )
            right = bisect_right(
                retained_samples,
                row.boundary_sample + DUPLICATE_SUPPRESSION_SAMPLES,
            )
            if left != right:
                continue
            insort(retained_samples, row.boundary_sample)
            retained_by_sample[row.boundary_sample] = row
        retained.extend(
            CandidateEvent(
                source_id=source_id,
                boundary_sample=sample,
                observed_frontier_sample=retained_by_sample[sample].observed_frontier_sample,
                score=retained_by_sample[sample].score,
            )
            for sample in retained_samples
        )
    return tuple(
        sorted(
            retained,
            key=lambda row: (-row.score, row.source_id, row.boundary_sample),
        )
    )


def sub_resolution_transitions(
    references: Iterable[ReferenceEvent],
) -> tuple[dict[str, Any], ...]:
    by_source: dict[str, list[ReferenceEvent]] = {}
    for row in references:
        if not _valid_reference(row):
            raise MetricContractError("reference event violates the metric contract")
        by_source.setdefault(row.source_id, []).append(row)
    diagnostics = []
    for source_id, rows in sorted(by_source.items()):
        rows.sort(key=lambda row: row.source_sample)
        for left, right in zip(rows, rows[1:], strict=False):
            distance = right.source_sample - left.source_sample
            if distance <= DUPLICATE_SUPPRESSION_SAMPLES:
                diagnostics.append(
                    {
                        "artifact_role": "sub_resolution_transition",
                        "source_id": source_id,
                        "left_sample": left.source_sample,
                        "right_sample": right.source_sample,
                        "distance_samples": distance,
                        "left_topology": left.topology,
                        "right_topology": right.topology,
                    }
                )
    return tuple(diagnostics)


def _augment_match(
    prediction_id: int,
    adjacency: dict[int, list[tuple[str, int]]],
    matched_prediction: dict[int, tuple[str, int]],
    matched_reference: dict[tuple[str, int], int],
) -> bool:
    seen_references: set[tuple[str, int]] = set()

    def visit(current: int) -> bool:
        for reference in adjacency[current]:
            if reference in seen_references:
                continue
            seen_references.add(reference)
            other = matched_reference.get(reference)
            if other is None or visit(other):
                matched_prediction[current] = reference
                matched_reference[reference] = current
                return True
        return False

    return visit(prediction_id)


def shared_score_thresholds(
    candidate_sets: Iterable[Sequence[CandidateEvent]],
) -> tuple[float, ...]:
    values: set[float] = set()
    for rows in candidate_sets:
        for row in rows:
            if not isinstance(row, CandidateEvent) or not _valid_prediction(row):
                raise MetricContractError("candidate event violates the temporal evidence contract")
            values.add(float(row.score))
    thresholds = sorted(values)
    if len(thresholds) < 2:
        raise MetricContractError("shared score thresholds require two finite unique values")
    return tuple(thresholds)


def _metric_row(
    prediction_count: int,
    reference_count: int,
    true_positive_count: int,
    exposure_hours: float,
) -> dict[str, float | int]:
    false_event_count = prediction_count - true_positive_count
    precision = true_positive_count / prediction_count if prediction_count else 0.0
    recall = true_positive_count / reference_count if reference_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive_count": true_positive_count,
        "false_event_count": false_event_count,
        "false_events_per_hour": false_event_count / exposure_hours,
        "false_negative_count": reference_count - true_positive_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def full_threshold_curve(
    candidates: Sequence[CandidateEvent],
    references: Sequence[ReferenceEvent],
    *,
    scored_source_samples: int,
    score_thresholds: Sequence[float],
    collars_ms: Sequence[int] = MATCHING_COLLARS_MS,
) -> dict[str, Any]:
    if (
        not isinstance(scored_source_samples, int)
        or isinstance(scored_source_samples, bool)
        or scored_source_samples <= 0
    ):
        raise MetricContractError("scored source exposure must be positive")
    try:
        actual_collars = tuple(collars_ms)
        thresholds = tuple(score_thresholds)
    except TypeError as error:
        raise MetricContractError("metric contract collections must be finite sequences") from error
    if actual_collars != MATCHING_COLLARS_MS:
        raise MetricContractError("matching collars differ from the fixed metric contract")
    if (
        len(thresholds) < 2
        or any(not _score(value) for value in thresholds)
        or any(left >= right for left, right in zip(thresholds, thresholds[1:], strict=False))
    ):
        raise MetricContractError("shared score thresholds must be strictly increasing and unique")
    if any(not isinstance(row, CandidateEvent) or not _valid_prediction(row) for row in candidates):
        raise MetricContractError("candidate event violates the temporal evidence contract")
    if any(not isinstance(row, ReferenceEvent) or not _valid_reference(row) for row in references):
        raise MetricContractError("reference event violates the metric contract")
    ordered = sorted(
        candidates,
        key=lambda row: (-row.score, row.source_id, row.boundary_sample),
    )
    if not {float(row.score) for row in ordered}.issubset(set(thresholds)):
        raise MetricContractError("candidate scores are absent from the shared threshold vector")
    if len({(row.source_id, row.boundary_sample) for row in ordered}) != len(ordered):
        raise MetricContractError("candidate event identities must be unique")
    if len({(row.source_id, row.source_sample) for row in references}) != len(references):
        raise MetricContractError("reference event identities must be unique")
    references_by_source: dict[str, list[ReferenceEvent]] = {}
    for reference in references:
        references_by_source.setdefault(reference.source_id, []).append(reference)
    for rows in references_by_source.values():
        rows.sort(key=lambda row: row.source_sample)
    reference_count = len(references)
    exposure_hours = scored_source_samples / (SAMPLE_RATE_HZ * 3600)
    states = {
        collar: {
            "adjacency": {},
            "matched_prediction": {},
            "matched_reference": {},
        }
        for collar in collars_ms
    }
    descending_rows: list[dict[str, Any]] = []
    index = 0
    for threshold in reversed(thresholds):
        while index < len(ordered) and ordered[index].score >= threshold:
            score = ordered[index].score
            end = index + 1
            while end < len(ordered) and ordered[end].score == score:
                end += 1
            for prediction_id in range(index, end):
                prediction = ordered[prediction_id]
                source_references = references_by_source.get(prediction.source_id, [])
                reference_samples = [row.source_sample for row in source_references]
                for collar in collars_ms:
                    radius = collar * SAMPLE_RATE_HZ // 1000
                    left = bisect_left(reference_samples, prediction.boundary_sample - radius)
                    right = bisect_right(reference_samples, prediction.boundary_sample + radius)
                    neighbors = sorted(
                        range(left, right),
                        key=lambda reference_index: (
                            abs(
                                source_references[reference_index].source_sample
                                - prediction.boundary_sample
                            ),
                            source_references[reference_index].source_sample,
                            reference_index,
                        ),
                    )
                    state = states[collar]
                    state["adjacency"][prediction_id] = [
                        (prediction.source_id, reference_index) for reference_index in neighbors
                    ]
                    _augment_match(
                        prediction_id,
                        state["adjacency"],
                        state["matched_prediction"],
                        state["matched_reference"],
                    )
            index = end
        metrics = {
            str(collar): _metric_row(
                index,
                reference_count,
                len(states[collar]["matched_reference"]),
                exposure_hours,
            )
            for collar in collars_ms
        }
        descending_rows.append(
            {
                "score_threshold": threshold,
                "prediction_count": index,
                "metrics": metrics,
                "matches": {
                    str(collar): [
                        {
                            "prediction_source_id": ordered[prediction_id].source_id,
                            "prediction_source_sample": ordered[prediction_id].boundary_sample,
                            "reference_source_id": reference_key[0],
                            "reference_source_sample": references_by_source[reference_key[0]][
                                reference_key[1]
                            ].source_sample,
                            "absolute_distance_samples": abs(
                                ordered[prediction_id].boundary_sample
                                - references_by_source[reference_key[0]][
                                    reference_key[1]
                                ].source_sample
                            ),
                        }
                        for prediction_id, reference_key in sorted(
                            states[collar]["matched_prediction"].items(),
                            key=lambda item: (
                                ordered[item[0]].source_id,
                                ordered[item[0]].boundary_sample,
                                references_by_source[item[1][0]][item[1][1]].source_sample,
                            ),
                        )
                    ]
                    for collar in collars_ms
                },
            }
        )
    rows = list(reversed(descending_rows))
    output_thresholds = [row["score_threshold"] for row in rows]
    if output_thresholds != list(thresholds):
        raise MetricContractError("full curve thresholds are not strictly increasing and unique")
    summaries: dict[str, Any] = {}
    for collar in collars_ms:
        key = str(collar)
        descending = descending_rows
        previous_recall = 0.0
        average_precision = 0.0
        for row in descending:
            values = row["metrics"][key]
            recall = float(values["recall"])
            if recall > previous_recall:
                average_precision += (recall - previous_recall) * float(values["precision"])
                previous_recall = recall
        max_f1_row = max(
            rows,
            key=lambda row: (
                float(row["metrics"][key]["f1"]),
                -float(row["metrics"][key]["false_events_per_hour"]),
                float(row["score_threshold"]),
            ),
            default=None,
        )
        summaries[key] = {
            "event_average_precision": average_precision,
            "max_f1": (max_f1_row["metrics"][key] if max_f1_row is not None else None),
            "max_f1_score_threshold": (
                max_f1_row["score_threshold"] if max_f1_row is not None else None
            ),
        }
    return {
        "score_thresholds": list(thresholds),
        "rows": rows,
        "summaries": summaries,
        "candidate_count": len(candidates),
        "reference_count": reference_count,
        "scored_source_samples": scored_source_samples,
        "scored_source_hours": exposure_hours,
        "sub_resolution_transitions": list(sub_resolution_transitions(references)),
        "false_events_per_hour_ceiling": None,
    }
