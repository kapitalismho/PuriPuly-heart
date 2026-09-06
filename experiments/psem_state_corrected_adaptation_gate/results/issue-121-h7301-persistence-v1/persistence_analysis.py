from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import config
from experiments.psem_frozen_ceiling_gate.experiment_support import monotonic_boundary_matches
from experiments.psem_state_corrected_adaptation_gate import calibrate
from experiments.psem_state_corrected_adaptation_gate.h_postprocess import (
    fit_calib_from_export,
    load_validated_export,
    prepare_dev_member,
)

DEV_SOURCES = (
    "alimeeting_R1019_M1928",
    "alimeeting_R1021_M4073",
    "alimeeting_R8009_M8019",
    "ami_EN2009d",
    "ami_ES2002b",
    "ami_ES2009a",
    "ami_ES2009b",
    "ami_ES2009c",
    "ami_ES2009d",
    "ami_ES2015d",
)
HORIZONS_MS = (100, 300, 500)
ENVELOPES = ("C", "M")
H500_THRESHOLD_HORIZONS_MS = (100, 300, 500)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()
def sha256_decompressed(path: Path) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values, dtype=np.float64), quantile)), 10)


def summary(values: list[float]) -> dict[str, float | int | None]:
    ordered = [float(value) for value in values]
    return {
        "n": len(ordered),
        "median_ms": percentile(ordered, 50.0),
        "p90_ms": percentile(ordered, 90.0),
        "max_ms": round(max(ordered), 10) if ordered else None,
    }


def _episode_runs(episode_ids: np.ndarray) -> list[tuple[str, list[int]]]:
    runs: list[tuple[str, list[int]]] = []
    start = 0
    count = len(episode_ids)
    while start < count:
        key = str(episode_ids[start])
        end = start + 1
        while end < count and str(episode_ids[end]) == key:
            end += 1
        runs.append((key, list(range(start, end))))
        start = end
    return runs


def positive_runs(member: dict[str, Any], threshold: float, horizon_ms: int) -> list[dict[str, Any]]:
    dev = member["dev"]
    scores = np.asarray(member["cand_raw_prob"], dtype=np.float64)
    confirmation_samples = int(horizon_ms) * 16
    rows: list[dict[str, Any]] = []
    for episode_id, frames in _episode_runs(np.asarray(dev.episode_ids)):
        pending_boundary: int | None = None
        duration_samples = 0
        previous_high_end: int | None = None
        emitted = False
        emitted_row: dict[str, Any] | None = None
        for index in frames:
            if not bool(dev.valid[index]):
                if emitted_row is not None:
                    rows.append(emitted_row)
                    break
                pending_boundary = None
                duration_samples = 0
                previous_high_end = None
                continue
            if bool(dev.masked[index]):
                continue
            start = int(dev.starts[index])
            end = int(dev.ends[index])
            high = bool(dev.speech_present[index]) and float(scores[index]) >= float(threshold)
            if not high or (
                previous_high_end is not None and start != previous_high_end
            ):
                if emitted_row is not None:
                    rows.append(emitted_row)
                    break
                pending_boundary = None
                duration_samples = 0
                previous_high_end = None
            if not high:
                continue
            if pending_boundary is None:
                pending_boundary = start
            duration_samples += end - start
            if emitted_row is not None:
                emitted_row["duration_ms"] = float(duration_samples) / 16.0
            previous_high_end = end
            if not emitted and duration_samples >= confirmation_samples:
                emitted = True
                emitted_row = {
                    "source_id": str(member["source_id"]),
                    "episode_id": episode_id,
                    "boundary_source_sample": int(pending_boundary),
                    "duration_ms": float(duration_samples) / 16.0,
                }
        else:
            if emitted_row is not None:
                rows.append(emitted_row)
    return rows


def classify_runs(member: dict[str, Any], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    references = [int(event.boundary_source_sample) for event in member["dev"].reference.events]
    predicted = [int(row["boundary_source_sample"]) for row in runs]
    matched = monotonic_boundary_matches(predicted, references, 500 * 16)
    matched_indices = {int(left) for left, _right in matched}
    return [
        {**row, "matched": index in matched_indices}
        for index, row in enumerate(runs)
    ]


def selected_thresholds(frontier: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for horizon_ms in HORIZONS_MS:
        horizon = str(horizon_ms)
        for envelope in ENVELOPES:
            envelope_name = f"{envelope.lower()}_envelope"
            path = ["horizons", horizon, "macro", "raw", envelope_name, "threshold"]
            result[f"H{horizon_ms}_{envelope}"] = {
                "horizon_ms": horizon_ms,
                "envelope": envelope,
                "threshold": float(frontier["horizons"][horizon]["macro"]["raw"][envelope_name]["threshold"]),
                "frontier_path": ".".join(path),
            }
    return result


def ap_results(members: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_source: dict[str, Any] = {}
    by_corpus: dict[str, list[str]] = {}
    pooled_cand: list[float] = []
    pooled_f0: list[float] = []
    pooled_target: list[float] = []
    for source_id in DEV_SOURCES:
        member = members[source_id]
        arrays = member
        valid = np.asarray(arrays["dev"].valid, dtype=bool)
        mapped = np.asarray(arrays["mapped"], dtype=bool)
        keep = np.flatnonzero(valid & mapped)
        target = [float(arrays["target"][int(index)]) for index in keep]
        cand = [float(arrays["cand_raw_prob"][int(index)]) for index in keep]
        f0 = [float(arrays["f0_prob"][int(index)]) for index in keep]
        per_source[source_id] = {
            "corpus": str(member["corpus"]),
            "valid_mapped_frames": int(len(keep)),
            "positive_frames": int(sum(value > 0 for value in target)),
            "candidate_ap": float(calibrate.average_precision(cand, target)),
            "f0_ap": float(calibrate.average_precision(f0, target)),
        }
        by_corpus.setdefault(str(member["corpus"]), []).append(source_id)
        pooled_cand.extend(cand)
        pooled_f0.extend(f0)
        pooled_target.extend(target)
    source_macro = {
        corpus: {
            "member_count": len(source_ids),
            "candidate_ap": float(
                np.mean([per_source[source_id]["candidate_ap"] for source_id in source_ids])
            ),
            "f0_ap": float(np.mean([per_source[source_id]["f0_ap"] for source_id in source_ids])),
            "sources": source_ids,
        }
        for corpus, source_ids in sorted(by_corpus.items())
    }
    source_macro["all_sources"] = {
        "member_count": len(DEV_SOURCES),
        "candidate_ap": float(np.mean([per_source[source_id]["candidate_ap"] for source_id in DEV_SOURCES])),
        "f0_ap": float(np.mean([per_source[source_id]["f0_ap"] for source_id in DEV_SOURCES])),
        "sources": list(DEV_SOURCES),
    }
    return {
        "selection": "valid_and_mapped",
        "score_fields": {"candidate": "cand_raw_prob", "f0": "f0_prob"},
        "pooled_valid_mapped": {
            "frame_count": len(pooled_target),
            "candidate_ap": float(calibrate.average_precision(pooled_cand, pooled_target)),
            "f0_ap": float(calibrate.average_precision(pooled_f0, pooled_target)),
        },
        "source_macro": source_macro,
        "per_source": per_source,
    }


def frontier_metric_summary(frontier: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"macro_raw": {}, "independent_corpus_raw": {}}
    for horizon_ms in HORIZONS_MS:
        horizon = str(horizon_ms)
        macro = frontier["horizons"][horizon]["macro"]["raw"]
        result["macro_raw"][horizon] = {
            name: {
                "contamination": float(point["contamination"]),
                "miss_rate": float(point["miss_rate"]),
                "false_cuts_per_hour": float(point["false_cuts_per_hour"]),
                "threshold": float(point["threshold"]),
            }
            for name, point in (
                ("reference", macro["reference"]),
                ("c_envelope", macro["c_envelope"]),
                ("m_envelope", macro["m_envelope"]),
            )
        }
        for corpus in ("alimeeting", "ami"):
            result["independent_corpus_raw"].setdefault(corpus, {})[horizon] = {}
            block = frontier["horizons"][horizon][corpus]["raw"]
            for envelope in ("c_envelope", "m_envelope"):
                point = block[envelope]
                result["independent_corpus_raw"][corpus][horizon][envelope] = {
                    "contamination": float(point["contamination"]),
                    "miss_rate": float(point["miss_rate"]),
                    "false_cuts_per_hour": float(point["false_cuts_per_hour"]),
                    "threshold": float(point["threshold"]),
                }
    return result


def raw_calibrated_metrics_identical(diagnostics: dict[str, Any]) -> bool:
    for horizon in map(str, HORIZONS_MS):
        raw = diagnostics["horizons"][horizon]["raw"]["envelopes"]
        calibrated = diagnostics["horizons"][horizon]["calibrated"]["envelopes"]
        for envelope in ("c_envelope", "m_envelope"):
            for point_name in ("alimeeting_point", "ami_point"):
                left = raw[envelope][point_name]
                right = calibrated[envelope][point_name]
                for metric in ("contamination", "false_cuts_per_hour", "miss_rate"):
                    if float(left[metric]) != float(right[metric]):
                        return False
    return True


def build_markdown(result: dict[str, Any]) -> str:
    ap = result["dev_candidate_ranking_ap"]
    lines = [
        "# Issue 121 H7301 persistence analysis",
        "",
        "This is a posthoc exploratory analysis of the frozen H7301 DEV export. It does not change the Gate 1 decision, open H7302/T2/TA/EVAL, or establish a source-level cause.",
        "",
        "## Decision context",
        "",
        "The scientific decision remains **STOP / inconclusive; retain F0**. The 100 ms point gains on the declared equal-corpus macro criterion, but only one of three horizons is jointly useful. The 300 ms and 500 ms points fail the joint criterion. No causal conclusion is established.",
        "",
        "## DEV ranking AP",
        "",
        "AP is computed posthoc over valid-and-mapped DEV frames using the existing `average_precision` implementation. The earlier report called DEV F0 AP unavailable; this bundle computes it now and does not retroactively treat it as preregistered evidence.",
        "",
        "| Population | Candidate AP | F0 AP |",
        "| --- | ---: | ---: |",
        f"| pooled valid+mapped frames ({ap['pooled_valid_mapped']['frame_count']}) | {ap['pooled_valid_mapped']['candidate_ap']:.15f} | {ap['pooled_valid_mapped']['f0_ap']:.15f} |",
    ]
    for corpus in ("AMI", "AliMeeting", "all_sources"):
        row = ap["source_macro"][corpus]
        label = "all ten source macro" if corpus == "all_sources" else f"{corpus} source macro ({row['member_count']})"
        lines.append(f"| {label} | {row['candidate_ap']:.15f} | {row['f0_ap']:.15f} |")
    lines.extend(
        [
            "",
            "The candidate pooled AP is `0.487681950383668`; candidate source-macro AP is AMI `0.491683249822344`, AliMeeting `0.430875816411725`, all ten `0.473441019799158`. Pooled and source-macro quantities are different estimands.",
            "",
            "## Global frontier metrics",
            "",
            "The following are the canonical equal-corpus macro raw-score metrics. The global threshold is shared across all ten meetings for each envelope; the independent corpus envelope points in the canonical JSON use separate corpus-specific thresholds and are not interchangeable with this table.",
            "",
            "| Horizon | Point | Threshold | Contamination | Miss rate | False cuts/hour |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for horizon in ("100", "300", "500"):
        for name, label in (("reference", "F0 reference"), ("c_envelope", "H C-envelope"), ("m_envelope", "H M-envelope")):
            row = result["frontier_metrics"]["macro_raw"][horizon][name]
            lines.append(
                f"| {horizon} ms | {label} | {row['threshold']} | {row['contamination']:.6f} | {row['miss_rate']:.9f} | {row['false_cuts_per_hour']:.6f} |"
            )
    lines.extend(
        [
            "",
            "Raw and calibrated event metrics are identical at the selected points; calibration only changes score coordinates. Bootstrap intervals in the canonical diagnostics are paired source/meeting-mean intervals (2,000 replicates), not pooled-rate or macro confidence intervals. Timing p90 claims are per meeting, not pooled.",
            "",
            "## Positive event-generating runs",
            "",
            "A run is high when speech is present and `cand_raw_prob >= threshold`. It must be contiguous in source samples; invalid frames close/reset a run, masked frames are skipped without duration or previous-end updates, and speech-absent or below-threshold frames close a run. Duration is the sum of eligible `(end-start)` intervals divided by 16 ms, measured through continuation after emission. A run is matched only by `monotonic_boundary_matches` with the configured 500 ms product-event tolerance; 500 ms is not the horizon.",
            "",
            "| Analysis | Horizon | Threshold | Runs | Matched median/p90/max ms | Unmatched median/p90/max ms |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for key, row in result["run_analyses"].items():
        matched = row["matched_duration_ms"]
        unmatched = row["unmatched_duration_ms"]
        lines.append(
            f"| {key} | {row['horizon_ms']} | {row['threshold']} | {row['positive_event_generating_runs']} | {matched['median_ms']:.1f}/{matched['p90_ms']:.1f}/{matched['max_ms']:.1f} ({matched['n']}) | {unmatched['median_ms']:.1f}/{unmatched['p90_ms']:.1f}/{unmatched['max_ms']:.1f} ({unmatched['n']}) |"
        )
    lines.extend(
        [
            "",
            "At the exact selected global C-envelope thresholds, the matched/unmatched run summaries are H100 `599: 480/589/960 ms` and `663: 300/1000/4700 ms`; H300 `655: 500/668.2/995 ms` and `545: 400/1400/4700 ms`; H500 `344: 500/694.5/972 ms` and `196: 892/2131/4700 ms`. Holding the exact H500 C-envelope score threshold while changing only the horizon gives unmatched `300/1000/4700 ms` at H100, `400/1300/4700 ms` at H300, and `892/2131/4700 ms` at H500. This is descriptive persistence/confirmation evidence; unmatched runs can include timing or matching failures and are not categorical ground-truth false positives.",
            "",
            "## Reproduction and provenance",
            "",
            "Run from the repository root:",
            "",
            "```text",
            "uv run python -m experiments.psem_state_corrected_adaptation_gate.results.issue-121-h7301-persistence-v1.persistence_analysis \\",
            "  --export-dir experiments/psem_state_corrected_adaptation_gate/results/issue-121-h7301-persistence-v1/export/gpu_export \\",
            "  --frontier experiments/psem_state_corrected_adaptation_gate/results/issue-121-h7301-persistence-v1/canonical/dev_frontier.json.gz \\",
            "  --diagnostics experiments/psem_state_corrected_adaptation_gate/results/issue-121-h7301-persistence-v1/canonical/gate1_diagnostics.json \\",
            "  --out-json experiments/psem_state_corrected_adaptation_gate/results/issue-121-h7301-persistence-v1/persistence_analysis.json \\",
            "  --out-md experiments/psem_state_corrected_adaptation_gate/results/issue-121-h7301-persistence-v1/PERSISTENCE_ANALYSIS.md",
            "```",
            "",
            "The gzip frontier is deterministic and decompresses byte-for-byte to the canonical frontier SHA recorded in `bundle_manifest.json`. The durable export contains only numeric NPZ arrays, the immutable export manifest, and training metrics; no audio, transcripts, checkpoints, credentials, or process logs are included.",
            "",
            "Observed frozen execution: 29m18s CPU postprocess, 53 FIT sources, 142 training steps, 11 CALIB NPZ and 10 DEV NPZ. The GPU/source binding, CPU postprocess source hash, trained-head hash, and export manifest hash are recorded in `bundle_manifest.json`.",
            "",
            "Formal commit review remains outstanding. This bundle is prepared for Director review; it has not been committed, pushed, or posted.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    frontier = load_json(args.frontier)
    diagnostics = load_json(args.diagnostics)
    export = load_validated_export(Path(args.export_dir))
    f0_fit, cand_fit, _calibration = fit_calib_from_export(export["calib"])
    members = {
        source_id: prepare_dev_member(source_id, export["dev"][source_id], f0_fit, cand_fit)
        for source_id in DEV_SOURCES
    }
    thresholds = selected_thresholds(frontier)
    analyses: dict[str, Any] = {}
    for key, point in thresholds.items():
        rows: list[dict[str, Any]] = []
        for source_id in DEV_SOURCES:
            rows.extend(
                classify_runs(
                    members[source_id],
                    positive_runs(members[source_id], point["threshold"], point["horizon_ms"]),
                )
            )
        matched_values = [float(row["duration_ms"]) for row in rows if row["matched"]]
        unmatched_values = [float(row["duration_ms"]) for row in rows if not row["matched"]]
        analyses[key] = {
            **point,
            "score_field": "cand_raw_prob",
            "positive_event_generating_runs": len(rows),
            "matched_duration_ms": summary(matched_values),
            "unmatched_duration_ms": summary(unmatched_values),
            "runs": rows,
        }
    held_threshold = thresholds["H500_C"]["threshold"]
    for horizon_ms in H500_THRESHOLD_HORIZONS_MS:
        rows = []
        for source_id in DEV_SOURCES:
            rows.extend(
                classify_runs(
                    members[source_id],
                    positive_runs(members[source_id], held_threshold, horizon_ms),
                )
            )
        matched_values = [float(row["duration_ms"]) for row in rows if row["matched"]]
        unmatched_values = [float(row["duration_ms"]) for row in rows if not row["matched"]]
        key = f"held_H500_C_threshold_H{horizon_ms}"
        analyses[key] = {
            "horizon_ms": horizon_ms,
            "threshold": held_threshold,
            "frontier_path": thresholds["H500_C"]["frontier_path"],
            "score_field": "cand_raw_prob",
            "positive_event_generating_runs": len(rows),
            "matched_duration_ms": summary(matched_values),
            "unmatched_duration_ms": summary(unmatched_values),
            "runs": rows,
        }
    manifest_path = Path(args.export_dir) / "gpu_export_manifest.json"
    return {
        "schema_version": "issue-121-h7301-persistence-analysis.v1",
        "decision": "STOP / inconclusive; retain F0",
        "status": "posthoc_exploratory_not_gate_change",
        "dev_candidate_ranking_ap": ap_results(members),
        "frontier_metrics": frontier_metric_summary(frontier),
        "thresholds": thresholds,
        "run_contract": {
            "score_field": "cand_raw_prob",
            "positive_rule": "speech_present and score >= exact global raw C/M threshold",
            "invalid_closes_run": True,
            "masked_skips_without_duration_or_previous_end_update": True,
            "speech_absent_closes_run": True,
            "below_threshold_closes_run": True,
            "continuity_rule": "high frame start equals previous high frame end",
            "duration_unit": "milliseconds",
            "source_samples_per_ms": 16,
            "duration_continues_after_emit": True,
            "event_boundary": "positive run start",
            "matching": "monotonic_boundary_matches",
            "matching_tolerance_ms": int(config()["product_event_alignment_tolerance_ms"]),
            "matching_tolerance_samples": int(config()["product_event_alignment_tolerance_ms"]) * 16,
            "unmatched_interpretation": "descriptive timing/matching residual, not categorical false positive",
        },
        "run_analyses": analyses,
        "raw_calibrated_event_metrics_identical": raw_calibrated_metrics_identical(diagnostics),
        "provenance": {
            "export_manifest_sha256": sha256_file(manifest_path),
            "frontier_sha256_after_decompression": sha256_decompressed(args.frontier),
            "diagnostics_sha256": sha256_file(args.diagnostics),
            "source_cpu_postprocess_sha256": "674a3639e4245d118960875fbc38a091e0ee9e995b9d8137664ae99417779ca5",
            "gpu_source_binding_sha256": "a3d9003a76ea167c33c644f1e3d15862e0181ed633a0378c91cb6d0fccaa263a",
            "trained_head_sha256": "bb5029da54b01a84763d0513a544cdbcd99bb1592f73b9eef71a1916e59aea3f",
            "seed": 7301,
            "fit_sources": 53,
            "calib_sources": 11,
            "dev_sources": 10,
            "observed_cpu_wall_duration": "29m18s",
            "training_steps": 142,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce issue-121 H7301 persistence analysis")
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    write_json(args.out_json, result)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(build_markdown(result), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
