from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.eot_experiment import peer_dual_threshold_experiment as exp11
from tools.eot_experiment import policy_analysis as base


LATENCY_BUDGETS_MS = (50.0, 100.0, 150.0)


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
        json.dumps(payload, indent=2, ensure_ascii=False, default=exp11._json_default)
        + "\n",
        encoding="utf-8",
    )


def _eligible_thresholds(rows: list[dict[str, Any]]) -> list[float]:
    return sorted(
        {
            0.0,
            1.0,
            *(
                round(float(row["score_224"]), 9)
                for row in rows
                if float(row["span_duration_ms"]) > 512.0
                and base._finite(row.get("score_224"))
            ),
        }
    )


def _trace(
    rows: list[dict[str, Any]],
    threshold_guard_224: float,
    *,
    latency_mode: str,
    array_data: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    data = exp11._array_data(rows, latency_mode) if array_data is None else array_data
    has224 = np.isfinite(data["score224"]) & np.isfinite(data["lat224"])
    arrival224 = 224.0 + np.where(has224, data["lat224"], np.inf)
    survives_to_512 = data["duration"] > 512.0
    stored = has224 & (arrival224 < 512.0) & (data["duration"] > arrival224)
    commit_at_512 = (
        survives_to_512
        & stored
        & (data["score224"] >= float(threshold_guard_224))
    )
    protect_to_800 = survives_to_512 & ~commit_at_512
    decision = np.full(len(rows), 512.0, dtype=np.float64)
    decision[protect_to_800] = 800.0
    stale224 = has224 & ((data["duration"] <= arrival224) | (arrival224 >= 512.0))
    return {
        "data": data,
        "decision_ms": decision,
        "fast_accept": np.zeros(len(rows), dtype=bool),
        "guard_accept": commit_at_512,
        "stale224": stale224,
        "stale512": np.zeros(len(rows), dtype=bool),
        "stored224": stored,
        "protect_to_800": protect_to_800,
    }


def _simulate(
    rows: list[dict[str, Any]],
    threshold_guard_224: float,
    *,
    latency_mode: str = "measured",
    array_data: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    trace = _trace(
        rows,
        threshold_guard_224,
        latency_mode=latency_mode,
        array_data=array_data,
    )
    metrics = exp11._metrics(rows, trace)
    metrics["stored_224_results"] = int(trace["stored224"].sum())
    metrics["protected_to_800"] = int(trace["protect_to_800"].sum())
    return metrics, trace


def _oracle(
    rows: list[dict[str, Any]], language: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    summary = {}
    for latency_mode in ("ideal", "measured", "worst_practical"):
        data = exp11._array_data(rows, latency_mode)
        baseline = exp11._simulate(
            rows,
            "B0",
            latency_mode=latency_mode,
            array_data=data,
        )
        mode_rows = []
        for threshold in _eligible_thresholds(rows):
            metrics, _ = _simulate(
                rows,
                threshold,
                latency_mode=latency_mode,
                array_data=data,
            )
            row = {
                "language": language,
                "latency_mode": latency_mode,
                "threshold_guard_224": threshold,
                **metrics,
                "mean_eot_added_latency_vs_b0_ms": metrics["mean_eot_latency_ms"]
                - baseline["mean_eot_latency_ms"],
                "fragmented_turn_relative_reduction_vs_b0": exp11._relative_reduction(
                    baseline["fragmented_turn_rate"], metrics["fragmented_turn_rate"]
                ),
                "fragmented_turn_absolute_reduction_vs_b0": baseline[
                    "fragmented_turn_rate"
                ]
                - metrics["fragmented_turn_rate"],
                "false_splits_per_100_turns_reduction_vs_b0": baseline[
                    "false_splits_per_100_turns"
                ]
                - metrics["false_splits_per_100_turns"],
            }
            mode_rows.append(row)
            output.append(row)
        profiles = {}
        for budget in LATENCY_BUDGETS_MS:
            candidates = [
                row
                for row in mode_rows
                if row["mean_eot_added_latency_vs_b0_ms"] <= budget + 1e-12
            ]
            profiles[f"Guard-{int(budget)}"] = (
                max(
                    candidates,
                    key=lambda row: (
                        row["fragmented_turn_relative_reduction_vs_b0"],
                        -row["mean_eot_added_latency_vs_b0_ms"],
                        -row["threshold_guard_224"],
                    ),
                )
                if candidates
                else None
            )
        best150 = profiles["Guard-150"]
        summary[latency_mode] = {
            "baseline": baseline,
            "profiles": profiles,
            "oracle_pass": bool(
                best150
                and best150["fragmented_turn_relative_reduction_vs_b0"] >= 0.15
            ),
        }
    return output, summary


def _select_inner(rows: list[dict[str, Any]], seed: int) -> dict[str, Any] | None:
    group_count = len({base._group_key(row) for row in rows})
    inner_folds = min(5, group_count)
    if inner_folds < 2:
        return None
    splits = base._split_groups(rows, seed + 104_729, inner_folds)
    if not splits:
        return None
    inner_training_rows = [row for _, train, _ in splits for row in train]
    validation_rows = [row for _, _, validation in splits for row in validation]
    data = exp11._array_data(validation_rows, "measured")
    baseline = exp11._simulate(
        validation_rows,
        "B0",
        latency_mode="measured",
        array_data=data,
    )
    candidates = []
    for threshold in _eligible_thresholds(inner_training_rows):
        metrics, _ = _simulate(
            validation_rows,
            threshold,
            latency_mode="measured",
            array_data=data,
        )
        added = metrics["mean_eot_latency_ms"] - baseline["mean_eot_latency_ms"]
        reduction = exp11._relative_reduction(
            baseline["fragmented_turn_rate"], metrics["fragmented_turn_rate"]
        )
        false_split_reduction = (
            baseline["false_splits_per_100_turns"]
            - metrics["false_splits_per_100_turns"]
        )
        if added <= 100.0 + 1e-12 and reduction >= 0.20 - 1e-12 and false_split_reduction > 0:
            candidates.append(
                {
                    "threshold_guard_224": threshold,
                    "inner_added_latency_ms": added,
                    "inner_fragmentation_relative_reduction": reduction,
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
            -item["threshold_guard_224"],
        ),
    )
    return selected | {
        "inner_folds": inner_folds,
        "valid_candidate_count": len(candidates),
    }


def _nested(
    rows: list[dict[str, Any]], language: str
) -> dict[str, Any]:
    rejected: list[dict[str, Any]] = []
    splits = base._outer_splits(rows, language, rejected)
    cv_rows = []
    group_records: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for split in splits:
        selection = _select_inner(
            split["train_rows"], int(split["seed"]) + int(split["fold"]) * 1_003
        )
        data = exp11._array_data(split["test_rows"], "measured")
        baseline_trace = exp11._trace(
            split["test_rows"],
            "B0",
            latency_mode="measured",
            array_data=data,
        )
        baseline = exp11._metrics(split["test_rows"], baseline_trace)
        row = {
            "language": language,
            "repeat": split["repeat"],
            "outer_seed": split["seed"],
            "outer_fold": split["fold"],
            "test_group_sha256": hashlib.sha256(
                json.dumps(
                    sorted({base._group_key(item) for item in split["test_rows"]}),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "status": "available" if selection else "unavailable",
            "threshold_guard_224": selection["threshold_guard_224"] if selection else None,
            "inner_folds": selection["inner_folds"] if selection else None,
            "valid_candidate_count": selection["valid_candidate_count"] if selection else 0,
            "inner_added_latency_ms": selection["inner_added_latency_ms"] if selection else None,
            "inner_fragmentation_relative_reduction": selection[
                "inner_fragmentation_relative_reduction"
            ]
            if selection
            else None,
            "baseline_fragmented_turn_rate": baseline["fragmented_turn_rate"],
            "baseline_false_splits_per_100_turns": baseline["false_splits_per_100_turns"],
            "baseline_mean_eot_latency_ms": baseline["mean_eot_latency_ms"],
        }
        if selection is None:
            row.update(
                {
                    "guard_fragmented_turn_rate": None,
                    "fragmented_turn_relative_reduction": None,
                    "guard_false_splits_per_100_turns": None,
                    "false_splits_reduction_per_100_turns": None,
                    "guard_mean_eot_latency_ms": None,
                    "mean_eot_added_latency_ms": None,
                    "guard_eot_timeout_rate": None,
                }
            )
            cv_rows.append(row)
            continue
        metrics, guard_trace = _simulate(
            split["test_rows"],
            selection["threshold_guard_224"],
            latency_mode="measured",
            array_data=data,
        )
        row.update(
            {
                "guard_fragmented_turn_rate": metrics["fragmented_turn_rate"],
                "fragmented_turn_relative_reduction": exp11._relative_reduction(
                    baseline["fragmented_turn_rate"], metrics["fragmented_turn_rate"]
                ),
                "guard_false_splits_per_100_turns": metrics["false_splits_per_100_turns"],
                "false_splits_reduction_per_100_turns": baseline[
                    "false_splits_per_100_turns"
                ]
                - metrics["false_splits_per_100_turns"],
                "guard_mean_eot_latency_ms": metrics["mean_eot_latency_ms"],
                "mean_eot_added_latency_ms": metrics["mean_eot_latency_ms"]
                - baseline["mean_eot_latency_ms"],
                "guard_eot_timeout_rate": metrics["eot_timeout_rate"],
            }
        )
        baseline_groups = exp11._group_fragmentation(split["test_rows"], baseline_trace)
        guard_groups = exp11._group_fragmentation(split["test_rows"], guard_trace)
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
    bootstrap = exp11._bootstrap_fragmentation(language, group_records) if group_records else None
    aggregate = {
        "language": language,
        "expected_outer_evaluations": expected,
        "available_outer_evaluations": len(available),
        "availability": len(available) / expected if expected else 0.0,
        "baseline_fragmented_turn_rate": baseline_fragmentation,
        "guard_fragmented_turn_rate": guard_fragmentation,
        "fragmented_turn_relative_reduction": exp11._relative_reduction(
            baseline_fragmentation, guard_fragmentation
        ) if baseline_fragmentation is not None and guard_fragmentation is not None else None,
        "mean_eot_added_latency_ms": float(
            np.mean([row["mean_eot_added_latency_ms"] for row in available])
        ) if available else None,
        "false_splits_reduction_per_100_turns": float(
            np.mean([row["false_splits_reduction_per_100_turns"] for row in available])
        ) if available else None,
        "threshold_guard_224_median": float(
            np.median([row["threshold_guard_224"] for row in available])
        ) if available else None,
        "threshold_guard_224_iqr": float(
            np.percentile([row["threshold_guard_224"] for row in available], 75)
            - np.percentile([row["threshold_guard_224"] for row in available], 25)
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
    return {"cv_rows": cv_rows, "aggregate": aggregate, "bootstrap": bootstrap}


def _compare_with_g512(
    language: str,
    pre_rows: list[dict[str, Any]],
    g512_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    lookup = {
        (int(row["repeat"]), int(row["outer_fold"])): row
        for row in g512_rows
        if row["language"] == language and row["status"] == "available"
    }
    paired = []
    for row in pre_rows:
        if row["status"] != "available":
            continue
        other = lookup.get((int(row["repeat"]), int(row["outer_fold"])))
        if not other:
            continue
        paired.append(
            {
                "latency_delta_ms": float(row["guard_mean_eot_latency_ms"])
                - float(other["guard_mean_eot_latency_ms"]),
                "fragmentation_delta": float(row["guard_fragmented_turn_rate"])
                - float(other["guard_fragmented_turn_rate"]),
                "false_splits_delta_per_100_turns": float(
                    row["guard_false_splits_per_100_turns"]
                )
                - float(other["guard_false_splits_per_100_turns"]),
            }
        )
    return {
        "paired_outer_evaluations": len(paired),
        "g224_pre_minus_g512_mean_eot_latency_ms": float(
            np.mean([row["latency_delta_ms"] for row in paired])
        ) if paired else None,
        "g224_pre_minus_g512_fragmented_turn_rate": float(
            np.mean([row["fragmentation_delta"] for row in paired])
        ) if paired else None,
        "g224_pre_minus_g512_false_splits_per_100_turns": float(
            np.mean([row["false_splits_delta_per_100_turns"] for row in paired])
        ) if paired else None,
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# G224-PRE precomputed guard experiment",
        "",
        "| Language | Oracle Guard-100 reduction | Nested availability | Nested reduction | Added latency | vs G512 latency | vs G512 fragmentation | Decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for language, payload in summary["languages"].items():
        oracle = payload["oracle"]["measured"]["profiles"]["Guard-100"]
        nested = payload.get("nested")
        comparison = payload.get("g512_comparison") or {}
        decision = summary["decision"][language]
        if nested:
            aggregate = nested["aggregate"]
            lines.append(
                f"| {language} | {oracle['fragmented_turn_relative_reduction_vs_b0'] * 100:.2f}% | "
                f"{aggregate['availability'] * 100:.2f}% | "
                f"{aggregate['fragmented_turn_relative_reduction'] * 100:.2f}% | "
                f"{aggregate['mean_eot_added_latency_ms']:.2f} ms | "
                f"{comparison['g224_pre_minus_g512_mean_eot_latency_ms']:.2f} ms | "
                f"{comparison['g224_pre_minus_g512_fragmented_turn_rate'] * 100:.2f} pp | "
                f"{decision} |"
            )
        else:
            lines.append(
                f"| {language} | {oracle['fragmented_turn_relative_reduction_vs_b0'] * 100:.2f}% | 0% | — | — | — | — | {decision} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(predictions_dir: Path, output_dir: Path, languages: tuple[str, ...]) -> dict[str, Any]:
    rows_by_language, validation = base.validate_input_artifacts(predictions_dir, languages)
    exp11_dir = predictions_dir.parent / "results-v11-peer-dual-threshold"
    exp11_summary = json.loads((exp11_dir / "oracle_summary.json").read_text(encoding="utf-8"))
    g512_rows = list(csv.DictReader((exp11_dir / "nested_cv_guard_all.csv").open(encoding="utf-8")))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "experiment": "G224-PRE precomputed 224 ms guard",
        "new_inference_run": False,
        "input_validation": validation,
        "languages": {},
        "decision": {},
    }
    oracle_rows = []
    nested_rows = []
    bootstrap_rows = []
    threshold_rows = []
    for language, rows in rows_by_language.items():
        language_oracle_rows, oracle_summary = _oracle(rows, language)
        oracle_rows.extend(language_oracle_rows)
        g512_oracle = exp11_summary["languages"][language]["guard_512"]["measured"][
            "profiles"
        ]["Guard-100"]
        matched_candidates = [
            row
            for row in language_oracle_rows
            if row["latency_mode"] == "measured"
            and row["fragmented_turn_relative_reduction_vs_b0"]
            >= g512_oracle["fragmented_turn_relative_reduction_vs_b0"] - 1e-12
        ]
        matched_g512_protection = (
            min(
                matched_candidates,
                key=lambda row: row["mean_eot_added_latency_vs_b0_ms"],
            )
            if matched_candidates
            else None
        )
        nested = _nested(rows, language) if oracle_summary["measured"]["oracle_pass"] else None
        comparison = None
        if nested:
            nested_rows.extend(nested["cv_rows"])
            bootstrap_rows.append(nested["bootstrap"])
            threshold_rows.append(
                {
                    "language": language,
                    "availability": nested["aggregate"]["availability"],
                    "threshold_guard_224_median": nested["aggregate"][
                        "threshold_guard_224_median"
                    ],
                    "threshold_guard_224_iqr": nested["aggregate"][
                        "threshold_guard_224_iqr"
                    ],
                }
            )
            comparison = _compare_with_g512(language, nested["cv_rows"], g512_rows)
        g512_pass = bool(
            exp11_summary["languages"][language]["nested_guard"]["aggregate"]["passes"]
        )
        if nested and nested["aggregate"]["passes"]:
            if not g512_pass:
                decision = "USE_G224_PRE"
            elif (
                comparison
                and comparison["g224_pre_minus_g512_fragmented_turn_rate"] <= 0.0
                and comparison["g224_pre_minus_g512_mean_eot_latency_ms"] <= 10.0
            ):
                decision = "PREFER_G224_PRE"
            else:
                decision = "KEEP_G512"
        elif g512_pass:
            decision = "KEEP_G512"
        else:
            decision = "KEEP_FIXED_512"
        summary["languages"][language] = {
            "oracle": oracle_summary,
            "nested": {
                "aggregate": nested["aggregate"],
                "bootstrap": nested["bootstrap"],
            }
            if nested
            else None,
            "g512_comparison": comparison,
            "matched_g512_oracle_protection": {
                "g512_fragmented_turn_relative_reduction": g512_oracle[
                    "fragmented_turn_relative_reduction_vs_b0"
                ],
                "g512_mean_eot_added_latency_ms": g512_oracle[
                    "mean_eot_added_latency_vs_b0_ms"
                ],
                "g224_pre": matched_g512_protection,
                "g224_pre_latency_savings_ms": g512_oracle[
                    "mean_eot_added_latency_vs_b0_ms"
                ]
                - matched_g512_protection["mean_eot_added_latency_vs_b0_ms"]
                if matched_g512_protection
                else None,
            },
        }
        summary["decision"][language] = decision
    _write_csv(output_dir / "guard_224_pre_oracle_pareto.csv", oracle_rows)
    _write_csv(output_dir / "nested_cv_guard_224_pre_all.csv", nested_rows)
    _write_csv(output_dir / "bootstrap_confidence_intervals.csv", bootstrap_rows)
    _write_csv(output_dir / "threshold_stability.csv", threshold_rows)
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "decision.json", summary["decision"])
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
            "oracle_guard_100": payload["oracle"]["measured"]["profiles"]["Guard-100"],
            "nested": payload["nested"]["aggregate"] if payload["nested"] else None,
            "g512_comparison": payload["g512_comparison"],
            "decision": summary["decision"][language],
        }
        for language, payload in summary["languages"].items()
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False, default=exp11._json_default))


if __name__ == "__main__":
    main()
