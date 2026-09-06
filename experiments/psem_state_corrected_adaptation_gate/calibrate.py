from __future__ import annotations

import math
from collections.abc import Sequence


class CalibrationError(RuntimeError):
    pass


def sigmoid(value: float) -> float:
    clipped = min(max(value, -80.0), 80.0)
    return 1.0 / (1.0 + math.exp(-clipped))


def nll_loss(logits: Sequence[float], targets: Sequence[float]) -> float:
    if len(logits) != len(targets) or not logits:
        raise CalibrationError("logit/target geometry differs")
    total = 0.0
    for z, y in zip(logits, targets):
        p = sigmoid(z)
        p = min(max(p, 1e-12), 1.0 - 1e-12)
        total += -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
    return total / len(logits)


def brier_score(logits: Sequence[float], targets: Sequence[float]) -> float:
    if len(logits) != len(targets) or not logits:
        raise CalibrationError("logit/target geometry differs")
    return sum((sigmoid(z) - y) ** 2 for z, y in zip(logits, targets)) / len(logits)


def apply_affine(logits: Sequence[float], slope: float, intercept: float) -> list[float]:
    if not slope > 0:
        raise CalibrationError("affine slope must be positive")
    return [slope * z + intercept for z in logits]


def average_precision(scores, targets):
    if len(scores) != len(targets) or not scores:
        raise CalibrationError("score/target geometry differs")
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranked = [float(targets[i]) for i in order]
    total_positive = sum(ranked)
    if total_positive <= 0:
        raise CalibrationError("ranking needs positive support")
    precision_sum = 0.0
    seen_positive = 0
    for rank, value in enumerate(ranked, start=1):
        if value > 0:
            seen_positive += 1
            precision_sum += seen_positive / rank
    return precision_sum / total_positive


_GRID_SLOPES = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
_GRID_INTERCEPTS = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)


def fit_affine_calibrator(
    z_raw: Sequence[float], targets: Sequence[float], role: str
) -> dict[str, object]:
    if role != "TRAIN-CALIB":
        raise CalibrationError("calibration fits TRAIN-CALIB only")
    z_list = list(z_raw)
    y_list = [float(v) for v in targets]
    if len(z_list) != len(y_list) or not z_list:
        raise CalibrationError("logit/target geometry differs")
    if all(v == y_list[0] for v in y_list):
        raise CalibrationError("calibration needs positive and negative support")
    count = len(z_list)
    grid_nll = [0.0] * (len(_GRID_SLOPES) * len(_GRID_INTERCEPTS))
    raw_nll_total = 0.0
    for z, y in zip(z_list, y_list):
        raw_p = min(max(sigmoid(z), 1e-12), 1.0 - 1e-12)
        raw_nll_total += -(y * math.log(raw_p) + (1.0 - y) * math.log(1.0 - raw_p))
        slot = 0
        for slope in _GRID_SLOPES:
            base = slope * z
            for intercept in _GRID_INTERCEPTS:
                p = sigmoid(base + intercept)
                p = min(max(p, 1e-12), 1.0 - 1e-12)
                grid_nll[slot] += -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
                slot += 1
    best_slope = 1.0
    best_intercept = 0.0
    best_nll = raw_nll_total / count
    for slot in range(len(grid_nll)):
        nll = grid_nll[slot] / count
        if nll < float(best_nll):
            best_slope = _GRID_SLOPES[slot // len(_GRID_INTERCEPTS)]
            best_intercept = _GRID_INTERCEPTS[slot % len(_GRID_INTERCEPTS)]
            best_nll = nll
    slope_f = best_slope
    intercept_f = best_intercept
    best: dict[str, object] = {"slope": slope_f, "intercept": intercept_f, "nll": best_nll}
    best["brier"] = (
        sum((sigmoid(slope_f * z + intercept_f) - y) ** 2 for z, y in zip(z_list, y_list))
        / count
    )
    best["raw_nll"] = raw_nll_total / count
    best["raw_brier"] = sum((sigmoid(z) - y) ** 2 for z, y in zip(z_list, y_list)) / count
    best["role"] = role
    return best
