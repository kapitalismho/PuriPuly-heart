from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from tools.eot_experiment import policy_analysis as base


LATENCY_BUDGETS_MS = (50.0, 100.0, 150.0)
FAST_PRECISION_MIN = 0.98
FAST_PRECISION_CI_MIN = 0.97
FAST_COVERAGE_NOMINAL_MIN = 0.05
FAST_COVERAGE_DEPLOY_MIN = 0.10
FAST_HOLD_FALSE_ACCEPT_MAX = 0.005
FAST_FRAGMENTATION_REGRESSION_MAX = 0.005


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_data(rows: list[dict[str, Any]], latency_mode: str) -> dict[str, np.ndarray]:
    data = base._array_data(rows)
    if latency_mode == "measured":
        return data
    if latency_mode == "ideal":
        data["lat224"] = np.where(np.isfinite(data["lat224"]), 0.0, np.nan)
        data["lat512"] = np.where(np.isfinite(data["lat512"]), 0.0, np.nan)
        return data
    if latency_mode == "worst_practical":
        finite224 = data["lat224"][np.isfinite(data["lat224"])]
        finite512 = data["lat512"][np.isfinite(data["lat512"])]
        p95_224 = float(np.percentile(finite224, 95)) if finite224.size else math.nan
        p95_512 = float(np.percentile(finite512, 95)) if finite512.size else math.nan
        data["lat224"] = np.where(np.isfinite(data["lat224"]), p95_224, np.nan)
        data["lat512"] = np.where(np.isfinite(data["lat512"]), p95_512, np.nan)
        return data
    raise ValueError(latency_mode)


def _trace(
    rows: list[dict[str, Any]],
    policy: str,
    *,
    latency_mode: str,
    threshold_fast: float | None = None,
    threshold_guard: float | None = None,
    old_threshold: float | None = None,
    array_data: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    data = _array_data(rows, latency_mode) if array_data is None else array_data
    duration = data["duration"]
    has224 = np.isfinite(data["score224"]) & np.isfinite(data["lat224"])
    has512 = np.isfinite(data["score512"]) & np.isfinite(data["lat512"])
    arrival224 = 224.0 + np.where(has224, data["lat224"], np.inf)
    arrival512_direct = 512.0 + np.where(has512, data["lat512"], np.inf)
    decision = np.full(len(rows), 800.0, dtype=np.float64)
    fast_accept = np.zeros(len(rows), dtype=bool)
    guard_accept = np.zeros(len(rows), dtype=bool)
    stale224 = np.zeros(len(rows), dtype=bool)
    stale512 = np.zeros(len(rows), dtype=bool)
    if policy == "B0":
        decision.fill(512.0)
    elif policy == "B1":
        decision.fill(800.0)
    elif policy == "F1":
        if threshold_fast is None:
            raise ValueError("F1 requires threshold_fast")
        fast_accept = (
            has224
            & (data["score224"] >= float(threshold_fast))
            & (duration > arrival224)
            & (arrival224 < 512.0)
        )
        stale224 = has224 & (data["score224"] >= float(threshold_fast)) & ~fast_accept
        decision.fill(512.0)
        decision[fast_accept] = arrival224[fast_accept]
    elif policy == "G1":
        if threshold_guard is None:
            raise ValueError("G1 requires threshold_guard")
        guard_accept = (
            has512
            & (data["score512"] >= float(threshold_guard))
            & (duration > arrival512_direct)
            & (arrival512_direct < 800.0)
        )
        stale512 = has512 & (data["score512"] >= float(threshold_guard)) & ~guard_accept
        decision[guard_accept] = arrival512_direct[guard_accept]
    elif policy in {"D2", "OLD-P2"}:
        if policy == "OLD-P2":
            if old_threshold is None:
                raise ValueError("OLD-P2 requires old_threshold")
            threshold_fast = old_threshold
            threshold_guard = old_threshold
        if threshold_fast is None or threshold_guard is None:
            raise ValueError(f"{policy} requires both thresholds")
        fast_accept = (
            has224
            & (data["score224"] >= float(threshold_fast))
            & (duration > arrival224)
            & (arrival224 < 800.0)
        )
        stale224 = has224 & (data["score224"] >= float(threshold_fast)) & ~fast_accept
        second_start = np.maximum(512.0, arrival224)
        arrival512 = second_start + np.where(has512, data["lat512"], np.inf)
        second_scheduled = has512 & ~fast_accept
        guard_accept = (
            second_scheduled
            & (data["score512"] >= float(threshold_guard))
            & (duration > arrival512)
            & (arrival512 < 800.0)
        )
        stale512 = (
            second_scheduled
            & (data["score512"] >= float(threshold_guard))
            & ~guard_accept
        )
        decision[fast_accept] = arrival224[fast_accept]
        decision[guard_accept] = arrival512[guard_accept]
    else:
        raise ValueError(policy)
    return {
        "data": data,
        "decision_ms": decision,
        "fast_accept": fast_accept,
        "guard_accept": guard_accept,
        "stale224": stale224,
        "stale512": stale512,
    }


def _metrics(rows: list[dict[str, Any]], trace: dict[str, Any]) -> dict[str, Any]:
    data = trace["data"]
    decision = trace["decision_ms"]
    false_cut = data["hold"] & (data["duration"] > decision)
    eot = data["eot"]
    hold = data["hold"]
    turns = {str(value) for value in data["turns"]}
    fragmented = {
        str(turn)
        for turn, is_false in zip(data["turns"], false_cut, strict=True)
        if is_false
    }
    fast_eot = int((eot & trace["fast_accept"]).sum())
    fast_hold = int((hold & trace["fast_accept"]).sum())
    guard_eot = int((eot & trace["guard_accept"]).sum())
    guard_hold = int((hold & trace["guard_accept"]).sum())
    fast_total = fast_eot + fast_hold
    return {
        "spans": len(rows),
        "eot_spans": int(eot.sum()),
        "hold_spans": int(hold.sum()),
        "false_cutoffs": int(false_cut.sum()),
        "false_cutoff_rate": float(false_cut.sum() / hold.sum()) if hold.sum() else 0.0,
        "mean_eot_latency_ms": float(np.mean(decision[eot])) if eot.sum() else None,
        "eot_timeout_rate": float((eot & (decision >= 800.0 - 1e-6)).sum() / eot.sum())
        if eot.sum()
        else 0.0,
        "turns": len(turns),
        "fragmented_turns": len(fragmented),
        "fragmented_turn_rate": len(fragmented) / len(turns) if turns else 0.0,
        "false_splits_per_100_turns": float(false_cut.sum() / len(turns) * 100.0)
        if turns
        else 0.0,
        "fast_eot_commits": fast_eot,
        "fast_hold_commits": fast_hold,
        "fast_commits": fast_total,
        "fast_precision": fast_eot / fast_total if fast_total else None,
        "fast_eot_coverage": fast_eot / int(eot.sum()) if eot.sum() else 0.0,
        "fast_hold_false_accept_rate": fast_hold / int(hold.sum()) if hold.sum() else 0.0,
        "guard_eot_commits": guard_eot,
        "guard_hold_commits": guard_hold,
        "stale_224_results": int(trace["stale224"].sum()),
        "stale_512_results": int(trace["stale512"].sum()),
    }


def _simulate(
    rows: list[dict[str, Any]],
    policy: str,
    *,
    latency_mode: str = "measured",
    threshold_fast: float | None = None,
    threshold_guard: float | None = None,
    old_threshold: float | None = None,
    array_data: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    return _metrics(
        rows,
        _trace(
            rows,
            policy,
            latency_mode=latency_mode,
            threshold_fast=threshold_fast,
            threshold_guard=threshold_guard,
            old_threshold=old_threshold,
            array_data=array_data,
        ),
    )


def _thresholds(rows: list[dict[str, Any]], key: str) -> list[float]:
    return sorted(
        {
            0.0,
            1.0,
            *(round(float(row[key]), 9) for row in rows if base._finite(row.get(key))),
        }
    )


def _relative_reduction(baseline: float, value: float) -> float:
    return (baseline - value) / baseline if baseline else 0.0


def _wilson_lower(successes: int, total: int, confidence: float = 0.95) -> float | None:
    if total <= 0:
        return None
    z = NormalDist().inv_cdf(confidence)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return (center - margin) / denominator


def _guard_oracle(
    rows: list[dict[str, Any]], language: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    summaries: dict[str, Any] = {}
    for latency_mode in ("ideal", "measured", "worst_practical"):
        baseline = _simulate(rows, "B0", latency_mode=latency_mode)
        mode_rows = []
        for threshold in _thresholds(rows, "score_512"):
            metrics = _simulate(
                rows,
                "G1",
                latency_mode=latency_mode,
                threshold_guard=threshold,
            )
            row = {
                "language": language,
                "latency_mode": latency_mode,
                "threshold_guard": threshold,
                **metrics,
                "mean_eot_added_latency_vs_b0_ms": float(
                    metrics["mean_eot_latency_ms"] - baseline["mean_eot_latency_ms"]
                ),
                "fragmented_turn_relative_reduction_vs_b0": _relative_reduction(
                    float(baseline["fragmented_turn_rate"]),
                    float(metrics["fragmented_turn_rate"]),
                ),
                "fragmented_turn_absolute_reduction_vs_b0": float(
                    baseline["fragmented_turn_rate"] - metrics["fragmented_turn_rate"]
                ),
                "false_splits_per_100_turns_reduction_vs_b0": float(
                    baseline["false_splits_per_100_turns"]
                    - metrics["false_splits_per_100_turns"]
                ),
            }
            mode_rows.append(row)
            output.append(row)
        profiles = {}
        for budget in LATENCY_BUDGETS_MS:
            eligible = [
                row
                for row in mode_rows
                if row["mean_eot_added_latency_vs_b0_ms"] <= budget + 1e-12
            ]
            selected = (
                max(
                    eligible,
                    key=lambda row: (
                        row["fragmented_turn_relative_reduction_vs_b0"],
                        -row["mean_eot_added_latency_vs_b0_ms"],
                    ),
                )
                if eligible
                else None
            )
            profiles[f"Guard-{int(budget)}"] = selected
        best_150 = profiles["Guard-150"]
        summaries[latency_mode] = {
            "baseline": baseline,
            "profiles": profiles,
            "oracle_pass": bool(
                best_150
                and best_150["fragmented_turn_relative_reduction_vs_b0"] >= 0.15
                and best_150["mean_eot_added_latency_vs_b0_ms"] <= 150.0
            ),
        }
    return output, summaries


def _fast_oracle(
    rows: list[dict[str, Any]], language: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    summaries: dict[str, Any] = {}
    for latency_mode in ("ideal", "measured", "worst_practical"):
        baseline = _simulate(rows, "B0", latency_mode=latency_mode)
        mode_rows = []
        for threshold in _thresholds(rows, "score_224"):
            metrics = _simulate(
                rows,
                "F1",
                latency_mode=latency_mode,
                threshold_fast=threshold,
            )
            ci_lower = _wilson_lower(
                int(metrics["fast_eot_commits"]), int(metrics["fast_commits"])
            )
            row = {
                "language": language,
                "latency_mode": latency_mode,
                "threshold_fast": threshold,
                **metrics,
                "fast_precision_one_sided_95_ci_lower": ci_lower,
                "fragmented_turn_regression_vs_b0": float(
                    metrics["fragmented_turn_rate"] - baseline["fragmented_turn_rate"]
                ),
                "mean_eot_latency_savings_vs_b0_ms": float(
                    baseline["mean_eot_latency_ms"] - metrics["mean_eot_latency_ms"]
                ),
            }
            row["passes_hold_false_accept"] = (
                row["fast_hold_false_accept_rate"] <= FAST_HOLD_FALSE_ACCEPT_MAX + 1e-12
            )
            row["passes_nominal_fast_candidate"] = bool(
                row["fast_precision"] is not None
                and row["fast_precision"] >= FAST_PRECISION_MIN
                and row["fast_eot_coverage"] >= FAST_COVERAGE_NOMINAL_MIN
                and row["passes_hold_false_accept"]
                and row["fragmented_turn_regression_vs_b0"]
                <= FAST_FRAGMENTATION_REGRESSION_MAX + 1e-12
            )
            row["passes_deployment_fast_evidence"] = bool(
                ci_lower is not None
                and ci_lower >= FAST_PRECISION_CI_MIN
                and row["fast_eot_coverage"] >= FAST_COVERAGE_DEPLOY_MIN
                and row["passes_hold_false_accept"]
                and row["fragmented_turn_regression_vs_b0"]
                <= FAST_FRAGMENTATION_REGRESSION_MAX + 1e-12
            )
            mode_rows.append(row)
            output.append(row)
        hold_safe = [row for row in mode_rows if row["passes_hold_false_accept"]]
        nominal = [row for row in mode_rows if row["passes_nominal_fast_candidate"]]
        deploy = [row for row in mode_rows if row["passes_deployment_fast_evidence"]]

        def highest_coverage(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
            if not candidates:
                return None
            return max(
                candidates,
                key=lambda row: (
                    row["fast_eot_coverage"],
                    row["fast_precision"] if row["fast_precision"] is not None else -1.0,
                    -row["threshold_fast"],
                ),
            )

        summaries[latency_mode] = {
            "baseline": baseline,
            "best_hold_safe": highest_coverage(hold_safe),
            "best_nominal_candidate": highest_coverage(nominal),
            "best_deployment_candidate": highest_coverage(deploy),
            "hold_safe_thresholds": len(hold_safe),
            "nominal_candidate_thresholds": len(nominal),
            "deployment_candidate_thresholds": len(deploy),
        }
    return output, summaries


def _select_guard_inner(
    rows: list[dict[str, Any]], seed: int
) -> dict[str, Any] | None:
    group_count = len({base._group_key(row) for row in rows})
    inner_folds = min(5, group_count)
    if inner_folds < 2:
        return None
    splits = base._split_groups(rows, seed + 104_729, inner_folds)
    if not splits:
        return None
    inner_training_rows = [row for _, train, _ in splits for row in train]
    validation_rows = [row for _, _, validation in splits for row in validation]
    validation_data = _array_data(validation_rows, "measured")
    baseline = _simulate(
        validation_rows,
        "B0",
        latency_mode="measured",
        array_data=validation_data,
    )
    candidates = []
    for threshold in _thresholds(inner_training_rows, "score_512"):
        metrics = _simulate(
            validation_rows,
            "G1",
            latency_mode="measured",
            threshold_guard=threshold,
            array_data=validation_data,
        )
        added_latency = float(
            metrics["mean_eot_latency_ms"] - baseline["mean_eot_latency_ms"]
        )
        fragmentation_reduction = _relative_reduction(
            float(baseline["fragmented_turn_rate"]),
            float(metrics["fragmented_turn_rate"]),
        )
        false_split_reduction = float(
            baseline["false_splits_per_100_turns"]
            - metrics["false_splits_per_100_turns"]
        )
        if (
            added_latency <= 100.0 + 1e-12
            and fragmentation_reduction >= 0.20 - 1e-12
            and false_split_reduction > 0.0
        ):
            candidates.append(
                {
                    "threshold_guard": threshold,
                    "inner_metrics": metrics,
                    "inner_added_latency_ms": added_latency,
                    "inner_fragmentation_relative_reduction": fragmentation_reduction,
                    "inner_false_splits_reduction_per_100_turns": false_split_reduction,
                }
            )
    if not candidates:
        return None
    selected = max(
        candidates,
        key=lambda item: (
            item["inner_fragmentation_relative_reduction"],
            -item["inner_added_latency_ms"],
            -item["threshold_guard"],
        ),
    )
    return selected | {
        "inner_folds": inner_folds,
        "inner_training_rows": len(inner_training_rows),
        "inner_validation_rows": len(validation_rows),
        "valid_candidate_count": len(candidates),
    }


def _group_fragmentation(
    rows: list[dict[str, Any]], trace: dict[str, Any]
) -> dict[str, bool]:
    data = trace["data"]
    false_cut = data["hold"] & (data["duration"] > trace["decision_ms"])
    output = {base._group_key(row): False for row in rows}
    for row, is_false in zip(rows, false_cut, strict=True):
        if is_false:
            output[base._group_key(row)] = True
    return output


def _bootstrap_fragmentation(
    language: str,
    group_records: dict[str, list[tuple[float, float]]],
    *,
    resamples: int = 10_000,
    seed: int = 20260802,
) -> dict[str, Any]:
    groups = sorted(group_records)
    baseline = np.asarray(
        [np.mean([pair[0] for pair in group_records[group]]) for group in groups],
        dtype=np.float64,
    )
    guard = np.asarray(
        [np.mean([pair[1] for pair in group_records[group]]) for group in groups],
        dtype=np.float64,
    )
    point_baseline = float(np.mean(baseline)) if baseline.size else 0.0
    point_guard = float(np.mean(guard)) if guard.size else 0.0
    rng = np.random.default_rng(seed + sum(ord(char) for char in language))
    indices = rng.integers(0, len(groups), size=(resamples, len(groups)))
    baseline_samples = baseline[indices].mean(axis=1)
    guard_samples = guard[indices].mean(axis=1)
    improvement = baseline_samples - guard_samples
    relative = np.divide(
        improvement,
        baseline_samples,
        out=np.zeros_like(improvement),
        where=baseline_samples != 0,
    )
    return {
        "language": language,
        "comparison": "G1_vs_B0",
        "bootstrap_unit": "unique_conversation",
        "unique_conversations": len(groups),
        "resamples": resamples,
        "seed": seed + sum(ord(char) for char in language),
        "baseline_fragmented_turn_rate": point_baseline,
        "guard_fragmented_turn_rate": point_guard,
        "fragmentation_absolute_improvement": point_baseline - point_guard,
        "fragmentation_absolute_improvement_ci_low": float(
            np.percentile(improvement, 2.5)
        ),
        "fragmentation_absolute_improvement_ci_high": float(
            np.percentile(improvement, 97.5)
        ),
        "fragmentation_relative_improvement": _relative_reduction(
            point_baseline, point_guard
        ),
        "fragmentation_relative_improvement_ci_low": float(
            np.percentile(relative, 2.5)
        ),
        "fragmentation_relative_improvement_ci_high": float(
            np.percentile(relative, 97.5)
        ),
    }


def _nested_guard(
    rows: list[dict[str, Any]], language: str
) -> dict[str, Any]:
    rejected: list[dict[str, Any]] = []
    splits = base._outer_splits(rows, language, rejected)
    cv_rows = []
    group_records: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for split in splits:
        selection = _select_guard_inner(
            split["train_rows"], int(split["seed"]) + int(split["fold"]) * 1_003
        )
        base_data = _array_data(split["test_rows"], "measured")
        baseline_trace = _trace(
            split["test_rows"],
            "B0",
            latency_mode="measured",
            array_data=base_data,
        )
        baseline = _metrics(split["test_rows"], baseline_trace)
        row = {
            "language": language,
            "repeat": split["repeat"],
            "outer_seed": split["seed"],
            "outer_fold": split["fold"],
            "train_groups": split["train_groups"],
            "test_groups": split["test_groups"],
            "test_group_sha256": hashlib.sha256(
                json.dumps(
                    sorted({base._group_key(item) for item in split["test_rows"]}),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "status": "available" if selection else "unavailable",
            "threshold_guard": selection["threshold_guard"] if selection else None,
            "inner_folds": selection["inner_folds"] if selection else None,
            "valid_candidate_count": selection["valid_candidate_count"]
            if selection
            else 0,
            "inner_added_latency_ms": selection["inner_added_latency_ms"]
            if selection
            else None,
            "inner_fragmentation_relative_reduction": selection[
                "inner_fragmentation_relative_reduction"
            ]
            if selection
            else None,
            "baseline_fragmented_turn_rate": baseline["fragmented_turn_rate"],
            "baseline_false_splits_per_100_turns": baseline[
                "false_splits_per_100_turns"
            ],
            "baseline_mean_eot_latency_ms": baseline["mean_eot_latency_ms"],
        }
        if selection is None:
            row.update(
                {
                    "guard_fragmented_turn_rate": None,
                    "fragmented_turn_relative_reduction": None,
                    "fragmented_turn_absolute_reduction": None,
                    "guard_false_splits_per_100_turns": None,
                    "false_splits_reduction_per_100_turns": None,
                    "guard_mean_eot_latency_ms": None,
                    "mean_eot_added_latency_ms": None,
                    "guard_eot_timeout_rate": None,
                }
            )
            cv_rows.append(row)
            continue
        guard_trace = _trace(
            split["test_rows"],
            "G1",
            latency_mode="measured",
            threshold_guard=selection["threshold_guard"],
            array_data=base_data,
        )
        guard = _metrics(split["test_rows"], guard_trace)
        row.update(
            {
                "guard_fragmented_turn_rate": guard["fragmented_turn_rate"],
                "fragmented_turn_relative_reduction": _relative_reduction(
                    baseline["fragmented_turn_rate"], guard["fragmented_turn_rate"]
                ),
                "fragmented_turn_absolute_reduction": baseline[
                    "fragmented_turn_rate"
                ]
                - guard["fragmented_turn_rate"],
                "guard_false_splits_per_100_turns": guard[
                    "false_splits_per_100_turns"
                ],
                "false_splits_reduction_per_100_turns": baseline[
                    "false_splits_per_100_turns"
                ]
                - guard["false_splits_per_100_turns"],
                "guard_mean_eot_latency_ms": guard["mean_eot_latency_ms"],
                "mean_eot_added_latency_ms": guard["mean_eot_latency_ms"]
                - baseline["mean_eot_latency_ms"],
                "guard_eot_timeout_rate": guard["eot_timeout_rate"],
            }
        )
        baseline_groups = _group_fragmentation(split["test_rows"], baseline_trace)
        guard_groups = _group_fragmentation(split["test_rows"], guard_trace)
        for group in baseline_groups:
            group_records[group].append(
                (float(baseline_groups[group]), float(guard_groups[group]))
            )
        cv_rows.append(row)
    available = [row for row in cv_rows if row["status"] == "available"]
    expected = len(splits)
    baseline_fragmentation = float(
        np.mean([row["baseline_fragmented_turn_rate"] for row in available])
    ) if available else None
    guard_fragmentation = float(
        np.mean([row["guard_fragmented_turn_rate"] for row in available])
    ) if available else None
    bootstrap = _bootstrap_fragmentation(language, group_records) if group_records else None
    aggregate = {
        "language": language,
        "expected_outer_evaluations": expected,
        "available_outer_evaluations": len(available),
        "availability": len(available) / expected if expected else 0.0,
        "baseline_fragmented_turn_rate": baseline_fragmentation,
        "guard_fragmented_turn_rate": guard_fragmentation,
        "fragmented_turn_relative_reduction": _relative_reduction(
            baseline_fragmentation, guard_fragmentation
        ) if baseline_fragmentation is not None and guard_fragmentation is not None else None,
        "mean_eot_added_latency_ms": float(
            np.mean([row["mean_eot_added_latency_ms"] for row in available])
        ) if available else None,
        "false_splits_reduction_per_100_turns": float(
            np.mean([row["false_splits_reduction_per_100_turns"] for row in available])
        ) if available else None,
        "threshold_guard_median": float(
            np.median([row["threshold_guard"] for row in available])
        ) if available else None,
        "threshold_guard_iqr": float(
            np.percentile([row["threshold_guard"] for row in available], 75)
            - np.percentile([row["threshold_guard"] for row in available], 25)
        ) if available else None,
        "rejected_split_count": len(rejected),
    }
    aggregate["passes"] = bool(
        aggregate["availability"] >= 0.80
        and aggregate["fragmented_turn_relative_reduction"] is not None
        and aggregate["fragmented_turn_relative_reduction"] >= 0.20
        and aggregate["mean_eot_added_latency_ms"] is not None
        and aggregate["mean_eot_added_latency_ms"] <= 100.0
        and aggregate["false_splits_reduction_per_100_turns"] is not None
        and aggregate["false_splits_reduction_per_100_turns"] > 0.0
        and bootstrap is not None
        and bootstrap["fragmentation_absolute_improvement_ci_low"] > 0.0
    )
    return {
        "cv_rows": cv_rows,
        "aggregate": aggregate,
        "bootstrap": bootstrap,
        "group_records": group_records,
    }


def _bucket(value: float, boundaries: list[tuple[float, str]]) -> str:
    for upper, label in boundaries:
        if value < upper:
            return label
    return boundaries[-1][1]


def _turn_bucket_rows(
    rows: list[dict[str, Any]],
    language: str,
    group_records: dict[str, list[tuple[float, float]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[base._group_key(row)].append(row)
    pause_members: dict[str, list[str]] = defaultdict(list)
    duration_members: dict[str, list[str]] = defaultdict(list)
    for group, group_rows in grouped.items():
        if group not in group_records:
            continue
        pause_count = sum(str(row["label"]) == "hold" for row in group_rows)
        if pause_count == 0:
            pause_bucket = "0"
        elif pause_count <= 2:
            pause_bucket = "1-2"
        elif pause_count <= 4:
            pause_bucket = "3-4"
        else:
            pause_bucket = "5+"
        duration_ms = max(float(row.get("span_end_ms") or 0.0) for row in group_rows)
        duration_bucket = _bucket(
            duration_ms,
            [(2_000.0, "<2s"), (5_000.0, "2-5s"), (10_000.0, "5-10s"), (math.inf, "10s+")],
        )
        pause_members[pause_bucket].append(group)
        duration_members[duration_bucket].append(group)

    def summarize(bucket_kind: str, members: dict[str, list[str]]) -> list[dict[str, Any]]:
        output = []
        for label, groups in members.items():
            baseline_fragmented = np.mean(
                [
                    np.mean([pair[0] for pair in group_records[group]])
                    for group in groups
                ]
            )
            guard_fragmented = np.mean(
                [
                    np.mean([pair[1] for pair in group_records[group]])
                    for group in groups
                ]
            )
            output.append(
                {
                    "language": language,
                    "bucket_kind": bucket_kind,
                    "bucket": label,
                    "unique_turns": len(groups),
                    "b0_unfragmented_turn_rate": 1.0 - float(baseline_fragmented),
                    "g1_unfragmented_turn_rate": 1.0 - float(guard_fragmented),
                    "unfragmented_turn_rate_improvement": float(
                        baseline_fragmented - guard_fragmented
                    ),
                }
            )
        return output

    return summarize("pause_count", pause_members), summarize("duration", duration_members)


def _bridge_checks(predictions_dir: Path, output_dir: Path) -> dict[str, Any]:
    bridge_dir = predictions_dir.parent / "results-v9-bridge-tests-all"
    bridge_summary = bridge_dir / "bridge_summary.json"
    legacy_ko = bridge_dir / "legacy_fallback_ko.csv"
    fingerprints = [bridge_dir / f"split_fingerprints_{lang}.csv" for lang in base.LANGUAGES]
    available = bridge_summary.is_file() and legacy_ko.is_file() and all(
        path.is_file() for path in fingerprints
    )
    payload = {
        "status": "reused_pass" if available else "missing",
        "new_inference_run": False,
        "bridge_summary": str(bridge_summary.resolve()),
        "bridge_summary_sha256": _sha256(bridge_summary) if bridge_summary.is_file() else None,
        "legacy_ko_sha256": _sha256(legacy_ko) if legacy_ko.is_file() else None,
        "split_fingerprint_sha256": {
            path.stem.rsplit("_", 1)[-1]: _sha256(path)
            for path in fingerprints
            if path.is_file()
        },
        "legacy_reference": {
            "threshold224": 0.728035808,
            "false_cutoff_rate": 0.19424573116674754,
            "mean_eot_latency_ms": 380.8370376554386,
            "exact_row_mismatches": 0,
        },
    }
    _write_json(output_dir / "evaluator_bridge_checks.json", payload)
    return payload


def _fixed_threshold_rows(
    rows_by_language: dict[str, list[dict[str, Any]]], bridge_summary_path: Path
) -> list[dict[str, Any]]:
    bridge_summary = json.loads(bridge_summary_path.read_text(encoding="utf-8"))
    output = []
    for language, rows in rows_by_language.items():
        for policy_name in ("B0", "B1"):
            metrics = _simulate(rows, policy_name, latency_mode="measured")
            output.append(
                {
                    "language": language,
                    "policy": policy_name,
                    "threshold_fast": None,
                    "threshold_guard": None,
                    **metrics,
                }
            )
        legacy_threshold = float(
            bridge_summary["legacy_fallback"][language]["aggregate"][
                "threshold224_median"
            ]
        )
        output.append(
            {
                "language": language,
                "policy": "OLD-P2",
                "threshold_fast": legacy_threshold,
                "threshold_guard": legacy_threshold,
                **_simulate(
                    rows,
                    "OLD-P2",
                    latency_mode="measured",
                    old_threshold=legacy_threshold,
                ),
            }
        )
        for threshold in (0.73, 0.80, 0.85, 0.89, 0.92, 0.95):
            output.append(
                {
                    "language": language,
                    "policy": "F1",
                    "threshold_fast": threshold,
                    "threshold_guard": None,
                    **_simulate(
                        rows,
                        "F1",
                        latency_mode="measured",
                        threshold_fast=threshold,
                    ),
                }
            )
            output.append(
                {
                    "language": language,
                    "policy": "G1",
                    "threshold_fast": None,
                    "threshold_guard": threshold,
                    **_simulate(
                        rows,
                        "G1",
                        latency_mode="measured",
                        threshold_guard=threshold,
                    ),
                }
            )
    return output


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Experiment 11: Peer Dual-Threshold Endpoint Policy",
        "",
        "Existing CPU prediction artifacts were reused. No inference or CPU benchmark was rerun.",
        "",
        "## Fail-fast results",
        "",
        "| Language | Guard-100 fragmentation reduction | Added mean EOT latency | HOLD-safe fast coverage | Fast precision |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for language, payload in summary["languages"].items():
        guard = payload["guard_512"]["measured"]["profiles"]["Guard-100"]
        fast = payload["fast_224"]["measured"]["best_hold_safe"]
        lines.append(
            f"| {language} | "
            f"{guard['fragmented_turn_relative_reduction_vs_b0'] * 100:.2f}% | "
            f"{guard['mean_eot_added_latency_vs_b0_ms']:.2f} ms | "
            f"{fast['fast_eot_coverage'] * 100:.2f}% | "
            f"{fast['fast_precision'] * 100:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Nested G1 results",
            "",
            "| Language | Availability | Fragmentation reduction | Added mean EOT latency | Bootstrap CI low | Decision |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for language, payload in summary["languages"].items():
        nested = payload.get("nested_guard")
        decision = summary["decision"][language]["decision"]
        if not nested:
            lines.append(f"| {language} | 0.00% | — | — | — | {decision} |")
            continue
        aggregate = nested["aggregate"]
        bootstrap = nested["bootstrap"]
        lines.append(
            f"| {language} | {aggregate['availability'] * 100:.2f}% | "
            f"{aggregate['fragmented_turn_relative_reduction'] * 100:.2f}% | "
            f"{aggregate['mean_eot_added_latency_ms']:.2f} ms | "
            f"{bootstrap['fragmentation_absolute_improvement_ci_low'] * 100:.2f} pp | "
            f"{decision} |"
        )
    lines.extend(
        [
            "",
            "No language had a full-data 224 ms threshold satisfying the nominal fast-path precision, coverage, HOLD false-accept, and fragmentation gates. D2 was therefore not evaluated.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(predictions_dir: Path, output_dir: Path, languages: tuple[str, ...]) -> dict[str, Any]:
    rows_by_language, validation = base.validate_input_artifacts(predictions_dir, languages)
    output_dir.mkdir(parents=True, exist_ok=True)
    bridge = _bridge_checks(predictions_dir, output_dir)
    if bridge["status"] != "reused_pass":
        raise RuntimeError("validated evaluator bridge artifacts are required")
    guard_rows = []
    fast_rows = []
    summary: dict[str, Any] = {
        "experiment": "Experiment 11: Peer Dual-Threshold Endpoint Policy",
        "stage": "complete",
        "input_validation": validation,
        "evaluator_bridge": bridge,
        "languages": {},
    }
    for language, rows in rows_by_language.items():
        language_guard_rows, guard_summary = _guard_oracle(rows, language)
        language_fast_rows, fast_summary = _fast_oracle(rows, language)
        guard_rows.extend(language_guard_rows)
        fast_rows.extend(language_fast_rows)
        summary["languages"][language] = {
            "guard_512": guard_summary,
            "fast_224": fast_summary,
        }
    _write_csv(output_dir / "guard_512_oracle_pareto.csv", guard_rows)
    _write_csv(output_dir / "fast_224_precision_coverage.csv", fast_rows)
    fixed_rows = _fixed_threshold_rows(
        rows_by_language, Path(bridge["bridge_summary"])
    )
    _write_csv(output_dir / "fixed_threshold_results.csv", fixed_rows)

    all_guard_cv_rows = []
    all_bootstrap_rows = []
    threshold_rows = []
    dual_rows = []
    pause_bucket_rows = []
    duration_bucket_rows = []
    decisions = {}
    for language, rows in rows_by_language.items():
        oracle_pass = summary["languages"][language]["guard_512"]["measured"][
            "oracle_pass"
        ]
        nested = _nested_guard(rows, language) if oracle_pass else None
        if nested:
            all_guard_cv_rows.extend(nested["cv_rows"])
            all_bootstrap_rows.append(nested["bootstrap"])
            threshold_rows.append(
                {
                    "language": language,
                    "policy": "G1",
                    "available_outer_evaluations": nested["aggregate"][
                        "available_outer_evaluations"
                    ],
                    "expected_outer_evaluations": nested["aggregate"][
                        "expected_outer_evaluations"
                    ],
                    "availability": nested["aggregate"]["availability"],
                    "threshold_guard_median": nested["aggregate"][
                        "threshold_guard_median"
                    ],
                    "threshold_guard_iqr": nested["aggregate"]["threshold_guard_iqr"],
                }
            )
            pause_rows, duration_rows = _turn_bucket_rows(
                rows, language, nested["group_records"]
            )
            pause_bucket_rows.extend(pause_rows)
            duration_bucket_rows.extend(duration_rows)
            summary["languages"][language]["nested_guard"] = {
                "aggregate": nested["aggregate"],
                "bootstrap": nested["bootstrap"],
            }
        fast_measured = summary["languages"][language]["fast_224"]["measured"]
        nominal_fast = fast_measured["best_nominal_candidate"]
        if not nested or not nested["aggregate"]["passes"]:
            decision = "KEEP_FIXED_512"
            reason = "G1 nested CV did not pass"
        elif nominal_fast is None:
            decision = "USE_512_GUARD_ONLY"
            reason = "G1 passed; no 224 ms threshold passed nominal fast-path gates"
        elif fast_measured["best_deployment_candidate"] is None:
            decision = "INSUFFICIENT_FAST_PATH_EVIDENCE"
            reason = "G1 passed; 224 ms deployment confidence or coverage was insufficient"
        else:
            decision = "INSUFFICIENT_FAST_PATH_EVIDENCE"
            reason = "D2 held-out evaluation is required before dual-path use"
        decisions[language] = {
            "decision": decision,
            "reason": reason,
            "guard_oracle_pass": oracle_pass,
            "guard_nested_pass": bool(nested and nested["aggregate"]["passes"]),
            "nominal_fast_candidate_available": nominal_fast is not None,
            "deployment_fast_candidate_available": fast_measured[
                "best_deployment_candidate"
            ]
            is not None,
        }
        dual_rows.append(
            {
                "language": language,
                "status": "not_evaluated",
                "reason": "no full-data nominal fast-path candidate"
                if nominal_fast is None
                else "held-out D2 evaluation not reached",
                "outer_evaluations": 0,
            }
        )
    summary["decision"] = decisions
    _write_csv(output_dir / "nested_cv_guard_all.csv", all_guard_cv_rows)
    _write_csv(output_dir / "nested_cv_dual_all.csv", dual_rows)
    _write_csv(
        output_dir / "turn_fragmentation_by_pause_count.csv", pause_bucket_rows
    )
    _write_csv(
        output_dir / "turn_fragmentation_by_duration.csv", duration_bucket_rows
    )
    _write_csv(
        output_dir / "bootstrap_confidence_intervals.csv", all_bootstrap_rows
    )
    _write_csv(output_dir / "threshold_stability.csv", threshold_rows)
    _write_json(output_dir / "decision.json", decisions)
    _write_json(output_dir / "oracle_summary.json", summary)
    _write_report(output_dir / "report.md", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", default=list(base.LANGUAGES))
    args = parser.parse_args()
    summary = run(
        args.predictions_dir.resolve(),
        args.output_dir.resolve(),
        tuple(args.languages),
    )
    compact = {
        language: {
            "guard_measured_100": payload["guard_512"]["measured"]["profiles"][
                "Guard-100"
            ],
            "fast_measured_hold_safe": payload["fast_224"]["measured"][
                "best_hold_safe"
            ],
        }
        for language, payload in summary["languages"].items()
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
