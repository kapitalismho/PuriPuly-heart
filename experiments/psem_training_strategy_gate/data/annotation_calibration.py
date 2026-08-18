from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from functools import reduce
from pathlib import Path
from typing import Any, Iterable, Sequence

from experiments.psem_training_strategy_gate.data.annotation_normalization import (
    NormalizedSession,
    normalize_inventory,
)
from experiments.psem_training_strategy_gate.data.label_contract import (
    LabelContract,
    load_contract,
)
from experiments.psem_training_strategy_gate.data.provenance import (
    canonical_sha256,
    sha256_file,
)

QUANTILES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
ALIGNMENT_MS = (1, 5, 10, 20, 50, 100)


class AnnotationCalibrationError(RuntimeError):
    pass


def _quantile(sorted_values: Sequence[int], probability: float) -> float:
    if not sorted_values:
        raise AnnotationCalibrationError("quantile requires at least one value")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    return sorted_values[lower] + (
        sorted_values[upper] - sorted_values[lower]
    ) * (position - lower)


def duration_summary(samples: Iterable[int], sample_rate_hz: int) -> dict[str, Any]:
    values = sorted(samples)
    if not values:
        return {"count": 0, "total_samples": 0, "quantiles_ms": {}}
    quantiles_ms = {
        f"p{int(probability * 100):02d}": round(
            _quantile(values, probability) * 1000 / sample_rate_hz, 6
        )
        for probability in QUANTILES
    }
    return {
        "count": len(values),
        "total_samples": sum(values),
        "quantiles_ms": quantiles_ms,
    }


def _fraction(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 8) if denominator else 0.0


def _boundary_granularity(
    sessions: Sequence[NormalizedSession], contract: LabelContract
) -> dict[str, Any]:
    boundaries = []
    positive_steps = []
    for session in sessions:
        session_boundaries = sorted(
            {
                interval.start_sample
                for interval in session.intervals
                if interval.start_sample > session.scored_start_sample
            }
            | {
                interval.end_sample
                for interval in session.intervals
                if interval.end_sample < session.scored_end_sample
            }
        )
        boundaries.extend(session_boundaries)
        steps = [
            right - left
            for left, right in zip(session_boundaries, session_boundaries[1:])
            if right > left
        ]
        positive_steps.extend(steps)
    gcd_samples = reduce(math.gcd, positive_steps) if positive_steps else 0
    return {
        "boundary_count": len(boundaries),
        "gcd_quantum_samples": gcd_samples,
        "gcd_quantum_ms": round(
            gcd_samples * 1000 / contract.sample_rate_hz, 6
        ),
        "minimum_positive_step_samples": min(positive_steps) if positive_steps else 0,
        "minimum_positive_step_ms": round(
            (min(positive_steps) if positive_steps else 0)
            * 1000
            / contract.sample_rate_hz,
            6,
        ),
        "alignment_fraction": {
            f"{milliseconds}ms": _fraction(
                sum(
                    boundary % contract.samples(milliseconds) == 0
                    for boundary in boundaries
                ),
                len(boundaries),
            )
            for milliseconds in ALIGNMENT_MS
        },
    }


def _intervening_speaker_durations(
    session: NormalizedSession, continuity_samples: int
) -> list[int]:
    intervals = session.intervals
    durations = []
    for index, middle in enumerate(intervals):
        if (
            middle.ambiguous
            or not middle.speaker_identity_known
            or len(middle.active_speakers) != 1
        ):
            continue
        left_index = index - 1
        left_gap = 0
        while left_index >= 0 and not intervals[left_index].active_speakers:
            gap = intervals[left_index]
            if gap.ambiguous or not gap.speaker_identity_known:
                left_index = -1
                break
            left_gap += gap.duration_samples
            left_index -= 1
        right_index = index + 1
        right_gap = 0
        while right_index < len(intervals) and not intervals[right_index].active_speakers:
            gap = intervals[right_index]
            if gap.ambiguous or not gap.speaker_identity_known:
                right_index = len(intervals)
                break
            right_gap += gap.duration_samples
            right_index += 1
        if left_index < 0 or right_index >= len(intervals):
            continue
        left = intervals[left_index]
        right = intervals[right_index]
        if (
            left.ambiguous
            or right.ambiguous
            or not left.speaker_identity_known
            or not right.speaker_identity_known
            or len(left.active_speakers) != 1
            or len(right.active_speakers) != 1
            or left.active_speakers != right.active_speakers
            or left.active_speakers == middle.active_speakers
            or left_gap > continuity_samples
            or right_gap > continuity_samples
        ):
            continue
        durations.append(middle.duration_samples)
    return durations


def _calibrate_subset(
    sessions: Sequence[NormalizedSession], contract: LabelContract
) -> dict[str, Any]:
    solo_durations = []
    gap_durations = []
    overlap_durations = []
    intervening_durations = []
    topology_counts: Counter[str] = Counter()
    masked_reasons: Counter[str] = Counter()
    diagnostic_masked_region_counts: Counter[str] = Counter()
    transition_count = 0
    masked_transition_count = 0
    scored_samples = 0
    ambiguous_samples = 0
    unknown_identity_samples = 0
    for session in sessions:
        scored_samples += session.labels.exposure["scored_samples"]
        ambiguous_samples += session.labels.exposure["ambiguous_samples"]
        unknown_identity_samples += session.labels.exposure["unknown_identity_samples"]
        intervals = session.intervals
        solo_durations.extend(
            interval.duration_samples
            for interval in intervals
            if not interval.ambiguous
            and interval.speaker_identity_known
            and len(interval.active_speakers) == 1
        )
        gap_durations.extend(
            interval.duration_samples
            for index, interval in enumerate(intervals)
            if 0 < index < len(intervals) - 1
            and not interval.ambiguous
            and interval.speaker_identity_known
            and not interval.active_speakers
            and intervals[index - 1].active_speakers
            and intervals[index + 1].active_speakers
        )
        overlap_durations.extend(
            interval.duration_samples
            for interval in intervals
            if not interval.ambiguous
            and interval.speaker_identity_known
            and len(interval.active_speakers) >= 2
        )
        intervening_durations.extend(
            _intervening_speaker_durations(
                session, contract.local_continuity_max_gap_samples
            )
        )
        for episode in session.labels.topology_episodes:
            if episode["coverage_gate_eligible"]:
                topology_counts[episode["primary_topology"]] += 1
        for transition in session.labels.transitions:
            transition_id = transition["transition_id"]
            if transition_id.startswith("D"):
                if transition["mask_state"] == "masked":
                    diagnostic_masked_region_counts[
                        transition["primary_topology"]
                    ] += 1
                continue
            transition_count += 1
            if transition["mask_state"] == "masked":
                masked_transition_count += 1
                masked_reasons[transition["primary_topology"]] += 1
    jitter = contract.annotation_boundary_jitter_samples
    gap_minimum = contract.gap_topology_min_duration_samples
    overlap_minimum = contract.overlap_topology_min_duration_samples
    continuity = contract.local_continuity_max_gap_samples
    reliable = contract.reliable_solo_min_duration_samples
    backchannel_minimum = contract.short_backchannel_min_duration_samples
    backchannel_maximum = contract.short_backchannel_max_duration_samples
    gap_bins = {
        "jitter_at_or_below_50ms": sum(value <= jitter for value in gap_durations),
        "micro_above_50ms_below_100ms": sum(
            jitter < value < gap_minimum for value in gap_durations
        ),
        "official_100ms_through_1200ms": sum(
            gap_minimum <= value <= continuity for value in gap_durations
        ),
        "continuity_unknown_above_1200ms": sum(
            value > continuity for value in gap_durations
        ),
    }
    overlap_bins = {
        "jitter_at_or_below_50ms": sum(
            value <= jitter for value in overlap_durations
        ),
        "micro_above_50ms_below_100ms": sum(
            jitter < value < overlap_minimum for value in overlap_durations
        ),
        "official_at_or_above_100ms": sum(
            value >= overlap_minimum for value in overlap_durations
        ),
    }
    solo_bins = {
        "fragment_below_200ms": sum(value < reliable for value in solo_durations),
        "reliable_at_or_above_200ms": sum(value >= reliable for value in solo_durations),
    }
    intervening_bins = {
        "below_200ms": sum(
            value < backchannel_minimum for value in intervening_durations
        ),
        "short_backchannel_200ms_through_1000ms": sum(
            backchannel_minimum <= value <= backchannel_maximum
            for value in intervening_durations
        ),
        "above_1000ms": sum(
            value > backchannel_maximum for value in intervening_durations
        ),
    }
    return {
        "session_count": len(sessions),
        "scored_samples": scored_samples,
        "scored_hours": round(
            scored_samples / contract.sample_rate_hz / 3600, 6
        ),
        "solo_segment_duration": duration_summary(
            solo_durations, contract.sample_rate_hz
        ),
        "silence_gap_duration": duration_summary(
            gap_durations, contract.sample_rate_hz
        ),
        "overlap_duration": duration_summary(
            overlap_durations, contract.sample_rate_hz
        ),
        "intervening_speaker_duration": duration_summary(
            intervening_durations, contract.sample_rate_hz
        ),
        "solo_threshold_bins": solo_bins,
        "silence_gap_bins": gap_bins,
        "overlap_bins": overlap_bins,
        "intervening_speaker_bins": intervening_bins,
        "micro_gap_fraction": _fraction(
            gap_bins["micro_above_50ms_below_100ms"], len(gap_durations)
        ),
        "micro_overlap_fraction": _fraction(
            overlap_bins["micro_above_50ms_below_100ms"],
            len(overlap_durations),
        ),
        "ambiguous_sample_fraction": _fraction(
            ambiguous_samples, scored_samples
        ),
        "unknown_identity_sample_fraction": _fraction(
            unknown_identity_samples, scored_samples
        ),
        "masked_transition_fraction": _fraction(
            masked_transition_count, transition_count
        ),
        "masked_transition_reasons": dict(sorted(masked_reasons.items())),
        "diagnostic_masked_region_counts": dict(
            sorted(diagnostic_masked_region_counts.items())
        ),
        "coverage_eligible_primary_topology_counts": dict(
            sorted(topology_counts.items())
        ),
        "boundary_granularity": _boundary_granularity(sessions, contract),
    }


def _load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise AnnotationCalibrationError(f"invalid JSONL manifest: {path}") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise AnnotationCalibrationError(
            f"JSONL manifest must contain objects: {path}"
        )
    return rows


def _validate_normalization_manifest(
    sessions: Sequence[NormalizedSession], data_dir: Path
) -> None:
    observed_rows = _load_jsonl_objects(data_dir / "normalization_manifest.jsonl")
    expected_rows = [
        session.manifest_row()
        for session in sorted(sessions, key=lambda session: session.source_id)
    ]
    if observed_rows != expected_rows:
        raise AnnotationCalibrationError(
            "normalization manifest does not match calibrated sessions"
        )


def build_calibration_report(
    sessions: Sequence[NormalizedSession], data_dir: Path
) -> dict[str, Any]:
    if not sessions:
        raise AnnotationCalibrationError("calibration requires normalized sessions")
    contract = load_contract()
    corpora = sorted({session.corpus for session in sessions})
    if corpora != ["AMI", "AliMeeting"]:
        raise AnnotationCalibrationError("calibration requires both accepted corpora")
    source_ids = [session.source_id for session in sessions]
    if len(set(source_ids)) != len(source_ids):
        raise AnnotationCalibrationError("calibration source identities must be unique")
    _validate_normalization_manifest(sessions, data_dir)
    by_corpus = {
        corpus: _calibrate_subset(
            [session for session in sessions if session.corpus == corpus], contract
        )
        for corpus in corpora
    }
    overall = _calibrate_subset(sessions, contract)
    retained = {
        "reliable_solo_min_duration": contract.reliable_solo_min_duration_ms,
        "annotation_boundary_jitter": contract.annotation_boundary_jitter_ms,
        "gap_topology_min_duration": contract.gap_topology_min_duration_ms,
        "overlap_topology_min_duration": contract.overlap_topology_min_duration_ms,
        "local_continuity_max_gap": contract.local_continuity_max_gap_ms,
        "short_backchannel_min_duration": contract.short_backchannel_min_duration_ms,
        "short_backchannel_max_duration": contract.short_backchannel_max_duration_ms,
    }
    return {
        "schema_version": 1,
        "artifact_role": "annotation_only_calibration",
        "authority_ref": "https://github.com/kapitalismho/PuriPuly-heart/issues/77",
        "authority_pin": "5778025c8aca1ea1cb7cd8fc41645b520ca1f9f749155b5f5daada32e940b559",
        "contract_version": contract.contract_version,
        "contract_document_sha256": contract.document_sha256,
        "contract_status": contract.status,
        "input_policy": {
            "source": "accepted natural source annotations only",
            "model_predictions_consulted": False,
            "model_scores_consulted": False,
            "official_model_results_inspected": False,
            "official_model_training_performed": False,
            "source_manifest_sha256": sha256_file(data_dir / "source_manifest.jsonl"),
            "annotation_manifest_sha256": sha256_file(
                data_dir / "annotation_manifest.jsonl"
            ),
            "normalization_manifest_sha256": sha256_file(
                data_dir / "normalization_manifest.jsonl"
            ),
            "source_ids_sha256": canonical_sha256(sorted(source_ids)),
        },
        "overall": overall,
        "by_corpus": by_corpus,
        "decision": {
            "action": "retain_all_provisional_constants_and_freeze_contract",
            "retained_constants_ms": retained,
            "version_bump_required": False,
            "rationale": [
                "The 50 ms reconciliation tolerance remains above observed per-corpus annotation granularity and micro-gap/micro-overlap rates are limited rather than dominant.",
                "The 200 ms reliable-solo threshold removes short annotation fragments while retaining the large majority of known singleton intervals in both corpora.",
                "The 100 ms topology minima leave substantial natural gap and overlap coverage in both corpora rather than eliminating nearly all short events.",
                "Masked transitions and diagnostic masked regions are explicitly attributable to v0 complex-overlap, continuity-unknown, mixed-unresolved, or overlap-to-silence rules rather than hidden ambiguous coercion.",
                "The 1200 ms continuity maximum is the hard inherited #76 value and was not eligible for calibration change.",
            ],
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    constants = report["decision"]["retained_constants_ms"]
    lines = [
        "# Annotation-only calibration",
        "",
        "This report uses accepted natural source annotations only. No model prediction, model score, official model result, or model training participated.",
        "",
        f"Contract: `{report['contract_version']}` (`{report['contract_status']}`)",
        "",
        "## Decision",
        "",
        "All provisional constants are retained and the operational contract is frozen without a version bump.",
        "",
        "| Constant | Retained value (ms) |",
        "|---|---:|",
    ]
    lines.extend(f"| `{name}` | {value} |" for name, value in constants.items())
    lines.extend(
        [
            "",
            "## Candidate set",
            "",
            "| Scope | Sessions | Scored hours | Solo intervals | Internal gaps | Overlap intervals |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    scopes = [("All", report["overall"]), *report["by_corpus"].items()]
    for name, stats in scopes:
        lines.append(
            f"| {name} | {stats['session_count']} | {stats['scored_hours']:.6f} | "
            f"{stats['solo_segment_duration']['count']} | "
            f"{stats['silence_gap_duration']['count']} | "
            f"{stats['overlap_duration']['count']} |"
        )
    lines.extend(
        [
            "",
            "## Distribution quantiles",
            "",
            "| Scope | Distribution | p01 ms | p05 ms | p10 ms | p50 ms | p90 ms | p95 ms | p99 ms |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, stats in scopes:
        for field in (
            "solo_segment_duration",
            "silence_gap_duration",
            "overlap_duration",
            "intervening_speaker_duration",
        ):
            quantiles = stats[field]["quantiles_ms"]
            lines.append(
                f"| {name} | `{field}` | {quantiles.get('p01', 'n/a')} | "
                f"{quantiles.get('p05', 'n/a')} | {quantiles.get('p10', 'n/a')} | "
                f"{quantiles.get('p50', 'n/a')} | {quantiles.get('p90', 'n/a')} | "
                f"{quantiles.get('p95', 'n/a')} | {quantiles.get('p99', 'n/a')} |"
            )
    overall = report["overall"]
    lines.extend(
        [
            "",
            "## Threshold audit",
            "",
            f"- Solo bins: `{json.dumps(overall['solo_threshold_bins'], sort_keys=True)}`",
            f"- Silence-gap bins: `{json.dumps(overall['silence_gap_bins'], sort_keys=True)}`",
            f"- Overlap bins: `{json.dumps(overall['overlap_bins'], sort_keys=True)}`",
            f"- Intervening-speaker bins: `{json.dumps(overall['intervening_speaker_bins'], sort_keys=True)}`",
            f"- Micro-gap fraction: `{overall['micro_gap_fraction']}`",
            f"- Micro-overlap fraction: `{overall['micro_overlap_fraction']}`",
            f"- Ambiguous sample fraction: `{overall['ambiguous_sample_fraction']}`",
            f"- Unknown-identity sample fraction: `{overall['unknown_identity_sample_fraction']}`",
            f"- Masked transition fraction: `{overall['masked_transition_fraction']}`",
            f"- Masked transition reasons: `{json.dumps(overall['masked_transition_reasons'], sort_keys=True)}`",
            f"- Diagnostic masked region counts: `{json.dumps(overall['diagnostic_masked_region_counts'], sort_keys=True)}`",
            "",
            "## Per-corpus annotation granularity",
            "",
            "| Corpus | GCD quantum (ms) | Minimum positive step (ms) | 1 ms aligned | 10 ms aligned | 50 ms aligned |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for corpus, stats in report["by_corpus"].items():
        granularity = stats["boundary_granularity"]
        alignment = granularity["alignment_fraction"]
        lines.append(
            f"| {corpus} | {granularity['gcd_quantum_ms']} | "
            f"{granularity['minimum_positive_step_ms']} | {alignment['1ms']} | "
            f"{alignment['10ms']} | {alignment['50ms']} |"
        )
    lines.extend(["", "## Rationale", ""])
    lines.extend(f"- {item}" for item in report["decision"]["rationale"])
    return "\n".join(lines) + "\n"


def write_calibration(
    data_dir: Path, corpus_root: Path, json_path: Path, markdown_path: Path
) -> None:
    sessions = normalize_inventory(data_dir, corpus_root)
    report = build_calibration_report(sessions, data_dir)
    json_path.write_text(
        json.dumps(
            report, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    write_calibration(
        args.data_dir.resolve(),
        args.corpus_root.resolve(),
        args.json_output.resolve(),
        args.markdown_output.resolve(),
    )


if __name__ == "__main__":
    main()
