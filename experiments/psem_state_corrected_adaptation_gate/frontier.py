from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


HORIZONS_MS = (100, 300, 500)
RAW_REFERENCE_THRESHOLD = 0.5


class FrontierError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    threshold: float
    false_cuts_per_hour: float
    contamination: float
    miss_rate: float


def unique_thresholds(scores: Sequence[float]) -> list[float]:
    ordered = sorted(set(float(s) for s in scores), reverse=True)
    if not ordered:
        raise FrontierError("no scores for frontier")
    return ordered


def reference_budget(points: Sequence[FrontierPoint]) -> FrontierPoint:
    for point in points:
        if point.threshold == RAW_REFERENCE_THRESHOLD:
            return point
    raise FrontierError("F0 frontier lacks the raw 0.5 reference point")


def select_envelopes(
    f0: FrontierPoint, candidates: Sequence[FrontierPoint]
) -> dict[str, Any]:
    within = [p for p in candidates if p.false_cuts_per_hour <= f0.false_cuts_per_hour]
    if not within:
        return {"budget": f0.false_cuts_per_hour, "c_envelope": None, "m_envelope": None, "useful": False}
    c_best = min(within, key=lambda p: (p.contamination, p.miss_rate))
    m_best = min(within, key=lambda p: (p.miss_rate, p.contamination))
    useful = any(
        p.contamination < f0.contamination and p.miss_rate < f0.miss_rate for p in (c_best, m_best)
    )
    return {
        "budget": f0.false_cuts_per_hour,
        "c_envelope": c_best,
        "m_envelope": m_best,
        "useful": useful,
    }


def build_frontier_from_scores(
    scores: Sequence[float],
    metric_at: Any,
) -> list[FrontierPoint]:
    points: list[FrontierPoint] = []
    for threshold in unique_thresholds(scores):
        false_cuts, contamination, miss_rate = metric_at(threshold)
        points.append(
            FrontierPoint(
                threshold=threshold,
                false_cuts_per_hour=float(false_cuts),
                contamination=float(contamination),
                miss_rate=float(miss_rate),
            )
        )
    return points
