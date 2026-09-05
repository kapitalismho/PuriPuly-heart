from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from tools.eot_experiment import policy_analysis as policy


LEGACY_SEEDS = (17, 29, 43, 71, 97)
FIXED_THRESHOLDS = (0.73, 0.80, 0.85, 0.89, 0.92, 0.95)
LEGACY_EXPECTED_KO = {
    "threshold224_median": 0.728035808,
    "false_cutoff_rate": 0.19424573116674754,
    "mean_endpoint_latency_ms": 380.8370376554386,
}


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


def _thresholds(rows: list[dict[str, Any]]) -> list[float]:
    return sorted(
        {
            0.0,
            1.0,
            *(
                round(float(row["score_224"]), 9)
                for row in rows
                if policy._finite(row.get("score_224"))
            ),
        }
    )


def _legacy_fallback(rows: list[dict[str, Any]], language: str) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    for seed in LEGACY_SEEDS:
        for fold, train_rows, test_rows in policy._split_groups(rows, seed, 5):
            baseline_train = policy.simulate_policy(train_rows, "B0")
            baseline_test = policy.simulate_policy(test_rows, "B0")
            candidates = []
            for threshold in _thresholds(train_rows):
                metrics = policy.simulate_policy(train_rows, "P1", threshold)
                candidates.append((threshold, metrics))
            threshold, training = min(
                candidates,
                key=lambda item: (
                    abs(
                        float(item[1]["false_cutoff_rate"])
                        - float(baseline_train["false_cutoff_rate"])
                    ),
                    float(item[1]["mean_endpoint_latency_ms"]),
                    float(item[0]),
                ),
            )
            heldout = policy.simulate_policy(test_rows, "P1", threshold)
            fold_rows.append(
                {
                    "language": language,
                    "seed": seed,
                    "fold": fold,
                    "threshold224": threshold,
                    "training_b0_false_cutoff_rate": baseline_train["false_cutoff_rate"],
                    "training_p1_false_cutoff_rate": training["false_cutoff_rate"],
                    "training_false_cutoff_difference": abs(
                        float(training["false_cutoff_rate"])
                        - float(baseline_train["false_cutoff_rate"])
                    ),
                    "heldout_b0_false_cutoff_rate": baseline_test["false_cutoff_rate"],
                    "heldout_false_cutoff_rate": heldout["false_cutoff_rate"],
                    "heldout_mean_endpoint_latency_ms": heldout[
                        "mean_endpoint_latency_ms"
                    ],
                    "heldout_eot_timeout_rate": heldout["eot_timeout_rate"],
                    "heldout_turn_fragmentation_rate": heldout[
                        "turn_fragmentation_rate"
                    ],
                }
            )
    aggregate = {
        "language": language,
        "folds": len(fold_rows),
        "threshold224_median": float(
            np.median([row["threshold224"] for row in fold_rows])
        ),
        "threshold224_iqr": float(
            np.percentile([row["threshold224"] for row in fold_rows], 75)
            - np.percentile([row["threshold224"] for row in fold_rows], 25)
        ),
        "false_cutoff_rate": float(
            np.mean([row["heldout_false_cutoff_rate"] for row in fold_rows])
        ),
        "mean_endpoint_latency_ms": float(
            np.mean([row["heldout_mean_endpoint_latency_ms"] for row in fold_rows])
        ),
        "eot_timeout_rate": float(
            np.mean([row["heldout_eot_timeout_rate"] for row in fold_rows])
        ),
        "turn_fragmentation_rate": float(
            np.mean([row["heldout_turn_fragmentation_rate"] for row in fold_rows])
        ),
    }
    expected = LEGACY_EXPECTED_KO if language == "ko" else None
    comparison = None
    if expected:
        comparison = {
            key: {
                "expected": expected[key],
                "actual": aggregate[key],
                "absolute_difference": abs(float(aggregate[key]) - float(expected[key])),
            }
            for key in expected
        }
        comparison["reproduced"] = all(
            comparison[key]["absolute_difference"] <= tolerance
            for key, tolerance in {
                "threshold224_median": 1e-9,
                "false_cutoff_rate": 1e-12,
                "mean_endpoint_latency_ms": 1e-9,
            }.items()
        )
    return {"fold_rows": fold_rows, "aggregate": aggregate, "comparison": comparison}


def _pareto_flags(rows: list[dict[str, Any]], y_key: str) -> list[bool]:
    flags = []
    for index, row in enumerate(rows):
        x = float(row["false_cutoff_rate"])
        y = float(row[y_key])
        dominated = any(
            other_index != index
            and float(other["false_cutoff_rate"]) <= x + 1e-12
            and float(other[y_key]) <= y + 1e-12
            and (
                float(other["false_cutoff_rate"]) < x - 1e-12
                or float(other[y_key]) < y - 1e-12
            )
            for other_index, other in enumerate(rows)
        )
        flags.append(not dominated)
    return flags


def _svg_plot(
    path: Path,
    rows: list[dict[str, Any]],
    y_key: str,
    pareto_key: str,
    title: str,
    y_label: str,
) -> None:
    width = 960
    height = 600
    left = 90
    right = 30
    top = 50
    bottom = 80
    xs = [float(row["false_cutoff_rate"]) * 100.0 for row in rows]
    ys = [float(row[y_key]) * (100.0 if "rate" in y_key else 1.0) for row in rows]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = max((x_max - x_min) * 0.04, 0.1)
    y_pad = max((y_max - y_min) * 0.06, 0.1)
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    def sy(value: float) -> float:
        return height - bottom - (value - y_min) / (y_max - y_min) * (height - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#222"/>',
    ]
    for tick in range(6):
        x_value = x_min + (x_max - x_min) * tick / 5
        y_value = y_min + (y_max - y_min) * tick / 5
        x_pos = sx(x_value)
        y_pos = sy(y_value)
        parts.extend(
            [
                f'<line x1="{x_pos:.2f}" y1="{top}" x2="{x_pos:.2f}" y2="{height - bottom}" stroke="#eee"/>',
                f'<text x="{x_pos:.2f}" y="{height - bottom + 24}" text-anchor="middle" font-family="sans-serif" font-size="12">{x_value:.1f}</text>',
                f'<line x1="{left}" y1="{y_pos:.2f}" x2="{width - right}" y2="{y_pos:.2f}" stroke="#eee"/>',
                f'<text x="{left - 12}" y="{y_pos + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{y_value:.1f}</text>',
            ]
        )
    parts.extend(
        [
            f'<text x="{(left + width - right) / 2}" y="{height - 22}" text-anchor="middle" font-family="sans-serif" font-size="14">False-cutoff rate (%)</text>',
            f'<text x="22" y="{(top + height - bottom) / 2}" text-anchor="middle" transform="rotate(-90 22 {(top + height - bottom) / 2})" font-family="sans-serif" font-size="14">{y_label}</text>',
        ]
    )
    for row, x, y in zip(rows, xs, ys, strict=True):
        color = "#d62728" if row[pareto_key] else "#6baed6"
        radius = 4 if row[pareto_key] else 2
        parts.append(
            f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="{radius}" fill="{color}" opacity="0.8"><title>t={row["threshold224"]:.6f}</title></circle>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _full_data_oracle(
    rows: list[dict[str, Any]], language: str, output_dir: Path
) -> dict[str, Any]:
    baseline = policy.simulate_policy(rows, "B0")
    oracle_rows = []
    for threshold in _thresholds(rows):
        metrics = policy.simulate_policy(rows, "P1", threshold)
        reduction = policy._relative_reduction(
            float(baseline["false_cutoff_rate"]),
            float(metrics["false_cutoff_rate"]),
        )
        oracle_rows.append(
            {
                "language": language,
                "threshold224": threshold,
                "false_cutoff_rate": metrics["false_cutoff_rate"],
                "relative_false_cutoff_reduction": reduction,
                "mean_endpoint_latency_ms": metrics["mean_endpoint_latency_ms"],
                "p50_endpoint_latency_ms": metrics["p50_endpoint_latency_ms"],
                "eot_timeout_rate": metrics["eot_timeout_rate"],
                "eot_recall": metrics["eot_early_detection_rate"],
                "turn_fragmentation_rate": metrics["turn_fragmentation_rate"],
                "passes_low_latency_gate": policy._candidate_is_valid(
                    metrics, baseline, 0.20
                ),
                "passes_stability_gate": policy._candidate_is_valid(
                    metrics, baseline, 0.35
                ),
            }
        )
    latency_flags = _pareto_flags(oracle_rows, "mean_endpoint_latency_ms")
    timeout_flags = _pareto_flags(oracle_rows, "eot_timeout_rate")
    for row, latency_flag, timeout_flag in zip(
        oracle_rows, latency_flags, timeout_flags, strict=True
    ):
        row["pareto_latency"] = latency_flag
        row["pareto_timeout"] = timeout_flag
    _write_csv(output_dir / f"oracle_pareto_{language}.csv", oracle_rows)
    _svg_plot(
        output_dir / f"oracle_pareto_latency_{language}.svg",
        oracle_rows,
        "mean_endpoint_latency_ms",
        "pareto_latency",
        f"{language}: P1 full-data oracle",
        "Mean EOT endpoint latency (ms)",
    )
    _svg_plot(
        output_dir / f"oracle_pareto_timeout_{language}.svg",
        oracle_rows,
        "eot_timeout_rate",
        "pareto_timeout",
        f"{language}: P1 full-data oracle",
        "EOT timeout rate (%)",
    )

    def best(target: str) -> dict[str, Any] | None:
        key = f"passes_{target}_gate"
        available = [row for row in oracle_rows if row[key]]
        if not available:
            return None
        return min(
            available,
            key=lambda row: (
                float(row["mean_endpoint_latency_ms"]),
                float(row["false_cutoff_rate"]),
                float(row["eot_timeout_rate"]),
            ),
        )

    return {
        "language": language,
        "thresholds_evaluated": len(oracle_rows),
        "baseline_false_cutoff_rate": baseline["false_cutoff_rate"],
        "low_latency_gate_candidates": sum(
            bool(row["passes_low_latency_gate"]) for row in oracle_rows
        ),
        "stability_gate_candidates": sum(
            bool(row["passes_stability_gate"]) for row in oracle_rows
        ),
        "best_low_latency": best("low_latency"),
        "best_stability": best("stability"),
        "latency_pareto_points": sum(latency_flags),
        "timeout_pareto_points": sum(timeout_flags),
    }


def _accepted_counts(
    rows: list[dict[str, Any]], threshold: float
) -> tuple[int, int]:
    data = policy._array_data(rows)
    trace = policy._policy_trace(rows, "P1", threshold, array_data=data)
    accepted_eot = int((data["eot"] & trace["accepted_any"]).sum())
    accepted_hold = int((data["hold"] & trace["accepted_any"]).sum())
    return accepted_eot, accepted_hold


def _fixed_thresholds(
    rows: list[dict[str, Any]], language: str, splits: list[dict[str, Any]]
) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    for threshold in FIXED_THRESHOLDS:
        for split in splits:
            test_rows = split["test_rows"]
            metrics = policy.simulate_policy(test_rows, "P1", threshold)
            accepted_eot, accepted_hold = _accepted_counts(test_rows, threshold)
            denominator = accepted_eot + accepted_hold
            fold_rows.append(
                {
                    "language": language,
                    "repeat": split["repeat"],
                    "outer_seed": split["seed"],
                    "outer_fold": split["fold"],
                    "threshold224": threshold,
                    "hold_spans": metrics["hold_spans"],
                    "eot_spans": metrics["eot_spans"],
                    "false_cutoffs": metrics["false_cutoffs"],
                    "accepted_eot": accepted_eot,
                    "accepted_hold": accepted_hold,
                    "false_cutoff_rate": metrics["false_cutoff_rate"],
                    "eot_recall": metrics["eot_early_detection_rate"],
                    "precision": accepted_eot / denominator if denominator else None,
                    "turn_fragmentation_rate": metrics["turn_fragmentation_rate"],
                    "mean_endpoint_latency_ms": metrics["mean_endpoint_latency_ms"],
                    "eot_timeout_rate": metrics["eot_timeout_rate"],
                }
            )
    aggregate_rows = []
    for threshold in FIXED_THRESHOLDS:
        selected = [row for row in fold_rows if row["threshold224"] == threshold]
        total_hold = sum(int(row["hold_spans"]) for row in selected)
        total_eot = sum(int(row["eot_spans"]) for row in selected)
        false_cutoffs = sum(int(row["false_cutoffs"]) for row in selected)
        accepted_eot = sum(int(row["accepted_eot"]) for row in selected)
        accepted_hold = sum(int(row["accepted_hold"]) for row in selected)
        aggregate_rows.append(
            {
                "language": language,
                "threshold224": threshold,
                "outer_evaluations": len(selected),
                "false_cutoff_rate": false_cutoffs / total_hold,
                "eot_recall": accepted_eot / total_eot,
                "precision": accepted_eot / (accepted_eot + accepted_hold),
                "turn_fragmentation_rate_mean": float(
                    np.mean([row["turn_fragmentation_rate"] for row in selected])
                ),
                "mean_endpoint_latency_ms": float(
                    np.average(
                        [row["mean_endpoint_latency_ms"] for row in selected],
                        weights=[row["eot_spans"] for row in selected],
                    )
                ),
                "eot_timeout_rate": 1.0 - accepted_eot / total_eot,
            }
        )
    return {"fold_rows": fold_rows, "aggregate_rows": aggregate_rows}


def _group_hash(groups: set[str]) -> str:
    payload = json.dumps(sorted(groups), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fingerprints(
    rows: list[dict[str, Any]], language: str, splits: list[dict[str, Any]]
) -> dict[str, Any]:
    all_groups = {policy._group_key(row) for row in rows}
    fingerprint_rows = []
    for split in splits:
        train_groups = {policy._group_key(row) for row in split["train_rows"]}
        test_groups = {policy._group_key(row) for row in split["test_rows"]}
        fingerprint_rows.append(
            {
                "language": language,
                "repeat": split["repeat"],
                "outer_seed": split["seed"],
                "outer_fold": split["fold"],
                "test_group_count": len(test_groups),
                "test_group_sha256": _group_hash(test_groups),
                "train_group_count": len(train_groups),
                "train_group_sha256": _group_hash(train_groups),
                "train_test_leakage_count": len(train_groups & test_groups),
                "covers_all_groups": train_groups | test_groups == all_groups,
            }
        )
    repeated = Counter(row["test_group_sha256"] for row in fingerprint_rows)
    rejected_again: list[dict[str, Any]] = []
    rerun_splits = policy._outer_splits(rows, language, rejected_again)
    rerun_hashes = {
        (split["repeat"], split["fold"]): _group_hash(
            {policy._group_key(row) for row in split["test_rows"]}
        )
        for split in rerun_splits
    }
    reproducible = all(
        rerun_hashes[(row["repeat"], row["outer_fold"])]
        == row["test_group_sha256"]
        for row in fingerprint_rows
    )
    repeat_partition_ok = True
    for repeat in sorted({int(row["repeat"]) for row in fingerprint_rows}):
        repeat_tests = [
            {
                policy._group_key(item)
                for item in split["test_rows"]
            }
            for split in splits
            if int(split["repeat"]) == repeat
        ]
        if set().union(*repeat_tests) != all_groups:
            repeat_partition_ok = False
        if sum(len(group_set) for group_set in repeat_tests) != len(all_groups):
            repeat_partition_ok = False
    return {
        "rows": fingerprint_rows,
        "summary": {
            "language": language,
            "outer_partitions": len(fingerprint_rows),
            "unique_test_group_hashes": len(repeated),
            "duplicate_test_partition_count": sum(count - 1 for count in repeated.values()),
            "train_test_leakage_partitions": sum(
                int(row["train_test_leakage_count"] > 0) for row in fingerprint_rows
            ),
            "all_repeats_are_complete_partitions": repeat_partition_ok,
            "deterministic_rerun_matches": reproducible,
            "rerun_rejected_split_count": len(rejected_again),
        },
    }


def run(predictions_dir: Path, output_dir: Path, languages: tuple[str, ...]) -> dict[str, Any]:
    rows_by_language, validation = policy.validate_input_artifacts(
        predictions_dir, languages
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "input_validation": validation,
        "legacy_fallback": {},
        "full_data_oracle": {},
        "fixed_thresholds": {},
        "split_fingerprints": {},
    }
    for language, rows in rows_by_language.items():
        legacy = _legacy_fallback(rows, language)
        _write_csv(output_dir / f"legacy_fallback_{language}.csv", legacy["fold_rows"])
        summary["legacy_fallback"][language] = {
            "aggregate": legacy["aggregate"],
            "comparison": legacy["comparison"],
        }

        rejected: list[dict[str, Any]] = []
        splits = policy._outer_splits(rows, language, rejected)
        for split in splits:
            split["language"] = language

        summary["full_data_oracle"][language] = _full_data_oracle(
            rows, language, output_dir
        )

        fixed = _fixed_thresholds(rows, language, splits)
        _write_csv(output_dir / f"fixed_threshold_folds_{language}.csv", fixed["fold_rows"])
        _write_csv(
            output_dir / f"fixed_threshold_aggregate_{language}.csv",
            fixed["aggregate_rows"],
        )
        summary["fixed_thresholds"][language] = fixed["aggregate_rows"]

        fingerprints = _fingerprints(rows, language, splits)
        _write_csv(
            output_dir / f"split_fingerprints_{language}.csv",
            fingerprints["rows"],
        )
        summary["split_fingerprints"][language] = fingerprints["summary"] | {
            "initial_rejected_split_count": len(rejected)
        }
    _write_json(output_dir / "bridge_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", default=["ko"])
    args = parser.parse_args()
    summary = run(
        args.predictions_dir.resolve(),
        args.output_dir.resolve(),
        tuple(args.languages),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
