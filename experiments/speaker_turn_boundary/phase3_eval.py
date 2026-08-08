from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from experiments.speaker_turn_boundary.coalescing import (
    CoalesceConfig,
    CoalescingOutcome,
    coalesce_vad_and_detector,
)
from experiments.speaker_turn_boundary.config import VAD_COALESCE_WINDOW_SAMPLES
from experiments.speaker_turn_boundary.events import SpeakerBoundaryEvent
from experiments.speaker_turn_boundary.metrics import RECALL_DEADLINES_MS
from experiments.speaker_turn_boundary.phase3_data import CaseInputs
from experiments.speaker_turn_boundary.phase3_metrics import (
    ATTRIBUTION_HORIZON_MS,
    PHASE3_METRIC_SCHEMA,
    CausalMatch,
    ObservedCut,
    causal_series,
    events_as_cuts,
    locked_product_series,
    matched_gt_indices,
    percentile_summary,
    rate_with_interval,
    timing_summary,
    validate_series,
    wilson_interval,
)


class Phase3EvaluationError(RuntimeError):
    pass


@dataclass(slots=True)
class CaseEvaluation:
    case: CaseInputs
    detector_events: list[SpeakerBoundaryEvent]
    coalescing: CoalescingOutcome
    b0_cuts: list[ObservedCut]
    detector_cuts: list[ObservedCut]
    added_detector_cuts: list[ObservedCut]
    b0_series: dict[int, list[CausalMatch]]
    detector_series: dict[int, list[CausalMatch]]
    product_series: dict[int, list[CausalMatch]]

    @property
    def recovered_indices(self) -> dict[int, set[int]]:
        b0 = matched_gt_indices(self.b0_series)
        product = matched_gt_indices(self.product_series)
        return {deadline: product[deadline] - b0[deadline] for deadline in RECALL_DEADLINES_MS}

    @property
    def regressed_indices(self) -> dict[int, set[int]]:
        b0 = matched_gt_indices(self.b0_series)
        product = matched_gt_indices(self.product_series)
        return {deadline: b0[deadline] - product[deadline] for deadline in RECALL_DEADLINES_MS}

    def to_evidence_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case.case.case_id,
            "audio_epoch": self.case.audio_epoch,
            "length_samples": self.case.length_samples,
            "active_speech_samples": self.case.active_speech_samples,
            "gt_changes": [
                {
                    "gt_index": index,
                    "change_sample": change.change_sample,
                    "kind": change.kind,
                    "prev_speakers": sorted(change.prev_speakers),
                    "next_speakers": sorted(change.next_speakers),
                }
                for index, change in enumerate(self.case.gt_changes)
            ],
            "vad_boundaries": [event.to_dict() for event in self.case.vad_boundaries],
            "detector_boundaries": [event.to_dict() for event in self.detector_events],
            "coalescing": self.coalescing.to_dict(),
            "matches": {
                "b0": {
                    str(deadline): [match.to_dict() for match in self.b0_series[deadline]]
                    for deadline in RECALL_DEADLINES_MS
                },
                "detector_only": {
                    str(deadline): [match.to_dict() for match in self.detector_series[deadline]]
                    for deadline in RECALL_DEADLINES_MS
                },
                "product_locked_b0": {
                    str(deadline): [match.to_dict() for match in self.product_series[deadline]]
                    for deadline in RECALL_DEADLINES_MS
                },
            },
        }


@dataclass(slots=True)
class ProfileEvaluation:
    profile_id: str
    family: str
    checkpoint: str
    profile_kind: str
    params: dict[str, Any]
    cases: list[CaseEvaluation] = field(default_factory=list)
    compute: dict[str, Any] = field(default_factory=dict)
    aggregate: dict[str, Any] = field(default_factory=dict)

    def evidence_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "experiments.speaker_turn_boundary.phase3.profile_evidence.v2",
            "metric_schema": PHASE3_METRIC_SCHEMA,
            "profile_id": self.profile_id,
            "family": self.family,
            "checkpoint": self.checkpoint,
            "profile_kind": self.profile_kind,
            "params": self.params,
            "compute": self.compute,
            "aggregate": self.aggregate,
            "cases": [case.to_evidence_dict() for case in self.cases],
        }


def evaluate_case(
    case: CaseInputs,
    detector_events: list[SpeakerBoundaryEvent],
    *,
    coalesce_window_samples: int = VAD_COALESCE_WINDOW_SAMPLES,
) -> CaseEvaluation:
    for event in [*case.vad_boundaries, *detector_events]:
        if event.audio_epoch != case.audio_epoch:
            raise Phase3EvaluationError(f"case {case.case.case_id}: event epoch mismatch")
        if event.observed_source_sample_at_emit < event.boundary_source_sample:
            raise Phase3EvaluationError(f"case {case.case.case_id}: anticipatory event")
        if event.observed_source_sample_at_emit > case.length_samples:
            raise Phase3EvaluationError(f"case {case.case.case_id}: event observes past epoch end")
    coalescing = coalesce_vad_and_detector(
        case.vad_boundaries,
        detector_events,
        config=CoalesceConfig(window_samples=coalesce_window_samples),
    )
    added_indices = [
        int(item["event_index"]) for item in coalescing.detections if item["disposition"] == "cut"
    ]
    b0_cuts = events_as_cuts(case.vad_boundaries, kind="vad")
    detector_cuts = events_as_cuts(detector_events, kind="detector")
    added_detector_cuts = events_as_cuts(
        [detector_events[index] for index in added_indices],
        kind="detector",
        source_indices=added_indices,
    )
    b0 = causal_series(case.gt_changes, b0_cuts)
    detector = causal_series(case.gt_changes, detector_cuts)
    product = locked_product_series(case.gt_changes, b0, added_detector_cuts)
    validate_series(case.gt_changes, b0_cuts, b0)
    validate_series(case.gt_changes, detector_cuts, detector)
    product_validation_cuts = [*b0_cuts, *added_detector_cuts]
    for deadline in RECALL_DEADLINES_MS:
        b0_count = len(b0[deadline])
        product_count = len(product[deadline])
        if product_count < b0_count:
            raise Phase3EvaluationError("locked product regressed below B0")
        if any(match.observation_delay_samples < 0 for match in product[deadline]):
            raise Phase3EvaluationError("locked product contains negative causal delay")
        if product_count > len(case.gt_changes):
            raise Phase3EvaluationError("locked product matched more changes than exist")
    if len(product_validation_cuts) != coalescing.report.total_logical_cuts:
        raise Phase3EvaluationError("logical cut count disagrees with coalescing report")
    return CaseEvaluation(
        case=case,
        detector_events=detector_events,
        coalescing=coalescing,
        b0_cuts=b0_cuts,
        detector_cuts=detector_cuts,
        added_detector_cuts=added_detector_cuts,
        b0_series=b0,
        detector_series=detector,
        product_series=product,
    )


def evaluate_profile(
    inputs: list[CaseInputs],
    event_provider: Callable[[CaseInputs], list[SpeakerBoundaryEvent]],
    *,
    profile_id: str,
    family: str,
    checkpoint: str,
    profile_kind: str,
    params: dict[str, Any],
    compute: dict[str, Any] | None = None,
) -> ProfileEvaluation:
    evaluation = ProfileEvaluation(
        profile_id=profile_id,
        family=family,
        checkpoint=checkpoint,
        profile_kind=profile_kind,
        params=params,
        compute=compute or {},
    )
    for case in inputs:
        evaluation.cases.append(evaluate_case(case, event_provider(case)))
    evaluation.aggregate = aggregate_profile(evaluation)
    return evaluation


def evaluate_b0(inputs: list[CaseInputs]) -> ProfileEvaluation:
    return evaluate_profile(
        inputs,
        lambda case: [],
        profile_id="b0_vad_only",
        family="b0",
        checkpoint="b0",
        profile_kind="vad_only",
        params={"policy": "current_peer_vad_segmentation"},
    )


def _recall_from_count(count: int, total: int) -> dict[str, Any]:
    return {
        "matched": count,
        "total": total,
        "recall": count / total if total else 0.0,
        "wilson_95": wilson_interval(count, total),
    }


def aggregate_profile(evaluation: ProfileEvaluation) -> dict[str, Any]:
    cases = evaluation.cases
    gt_count = sum(len(case.case.gt_changes) for case in cases)
    source_samples = sum(case.case.length_samples for case in cases)
    active_samples = sum(case.case.active_speech_samples for case in cases)
    source_hours = source_samples / (16000.0 * 3600.0)
    active_hours = active_samples / (16000.0 * 3600.0)
    b0_counts = {
        deadline: sum(len(case.b0_series[deadline]) for case in cases)
        for deadline in RECALL_DEADLINES_MS
    }
    detector_counts = {
        deadline: sum(len(case.detector_series[deadline]) for case in cases)
        for deadline in RECALL_DEADLINES_MS
    }
    product_counts = {
        deadline: sum(len(case.product_series[deadline]) for case in cases)
        for deadline in RECALL_DEADLINES_MS
    }
    recovered_counts = {
        deadline: sum(len(case.recovered_indices[deadline]) for case in cases)
        for deadline in RECALL_DEADLINES_MS
    }
    regressed_counts = {
        deadline: sum(len(case.regressed_indices[deadline]) for case in cases)
        for deadline in RECALL_DEADLINES_MS
    }
    if any(regressed_counts.values()):
        raise Phase3EvaluationError("locked product reported a regressed B0 match")
    b0_cut_count = sum(len(case.b0_cuts) for case in cases)
    detector_event_count = sum(len(case.detector_cuts) for case in cases)
    added_cut_count = sum(len(case.added_detector_cuts) for case in cases)
    horizon = ATTRIBUTION_HORIZON_MS
    b0_false = b0_cut_count - b0_counts[horizon]
    detector_false = detector_event_count - detector_counts[horizon]
    incremental_false = added_cut_count - recovered_counts[horizon]
    product_false = b0_false + incremental_false
    if min(b0_false, detector_false, incremental_false, product_false) < 0:
        raise Phase3EvaluationError("false-cut accounting became negative")
    product_horizon_matches = [match for case in cases for match in case.product_series[horizon]]
    recovered_horizon_matches = [
        match
        for case in cases
        for match in case.product_series[horizon]
        if match.gt_index in case.recovered_indices[horizon]
    ]
    coalescing = {
        "detector_events": detector_event_count,
        "coalesced": sum(case.coalescing.report.coalesced_count for case in cases),
        "duplicates_near_same_vad": sum(case.coalescing.report.duplicate_count for case in cases),
        "stale": sum(case.coalescing.report.stale_detector_events for case in cases),
        "added_logical_cuts": added_cut_count,
        "total_product_logical_cuts": b0_cut_count + added_cut_count,
    }
    aggregate: dict[str, Any] = {
        "metric_schema": PHASE3_METRIC_SCHEMA,
        "case_count": len(cases),
        "gt_change_count": gt_count,
        "source_samples": source_samples,
        "active_speech_samples": active_samples,
        "source_hours": source_hours,
        "active_speech_hours": active_hours,
        "b0": {
            "recall_at_ms": {
                str(deadline): _recall_from_count(b0_counts[deadline], gt_count)
                for deadline in RECALL_DEADLINES_MS
            },
            "logical_cuts": b0_cut_count,
            "false_cuts": b0_false,
        },
        "detector_only": {
            "recall_at_ms": {
                str(deadline): _recall_from_count(detector_counts[deadline], gt_count)
                for deadline in RECALL_DEADLINES_MS
            },
            "events": detector_event_count,
            "false_cuts": detector_false,
        },
        "product": {
            "matching_policy": "lock_b0_then_match_added_detector_cuts",
            "recall_at_ms": {
                str(deadline): _recall_from_count(product_counts[deadline], gt_count)
                for deadline in RECALL_DEADLINES_MS
            },
            "logical_cuts": b0_cut_count + added_cut_count,
            "false_cuts": product_false,
            "remaining_false_merges_at_2000ms": gt_count - product_counts[horizon],
        },
        "incremental": {
            "recovered_b0_misses_at_ms": {
                str(deadline): recovered_counts[deadline] for deadline in RECALL_DEADLINES_MS
            },
            "regressed_b0_matches_at_ms": {
                str(deadline): regressed_counts[deadline] for deadline in RECALL_DEADLINES_MS
            },
            "added_logical_cuts": added_cut_count,
            "extra_false_cuts": incremental_false,
            "recovered_per_extra_false_cut_at_500ms": (
                None if incremental_false == 0 else recovered_counts[500] / incremental_false
            ),
            "zero_extra_false_with_recovery_at_500ms": (
                incremental_false == 0 and recovered_counts[500] > 0
            ),
        },
        "rates": {
            "incremental_false_per_source_hour": rate_with_interval(
                incremental_false, source_hours
            ),
            "incremental_false_per_active_speech_hour": rate_with_interval(
                incremental_false, active_hours
            ),
            "incremental_false_per_5min_source_session": (
                incremental_false / (source_hours * 12.0) if source_hours > 0 else None
            ),
            "product_false_per_source_hour": rate_with_interval(product_false, source_hours),
            "b0_false_per_source_hour": rate_with_interval(b0_false, source_hours),
        },
        "coalescing": coalescing,
        "timing": {
            "product_matches_at_2000ms": timing_summary(product_horizon_matches),
            "incremental_recoveries_at_2000ms": timing_summary(recovered_horizon_matches),
        },
        "conditions": condition_breakdown(cases),
    }
    return aggregate


def condition_breakdown(cases: list[CaseEvaluation]) -> dict[str, Any]:
    accumulators: dict[tuple[str, str], dict[str, Any]] = {}
    case_axes = ("dataset", "language", "domain", "stress")
    change_axes = ("gap_condition", "turn_duration", "gt_kind")
    for case_eval in cases:
        case = case_eval.case
        b0_sets = matched_gt_indices(case_eval.b0_series)
        product_sets = matched_gt_indices(case_eval.product_series)
        recovered_sets = case_eval.recovered_indices
        for index, label in enumerate(case.change_labels):
            labels = {
                "dataset": case.dataset,
                "language": case.language,
                "domain": case.domain,
                "stress": case.stress,
                "gap_condition": label.gap_bucket,
                "turn_duration": label.turn_bucket,
                "gt_kind": label.gt_kind,
            }
            for axis, bucket in labels.items():
                acc = accumulators.setdefault(
                    (axis, bucket),
                    {
                        "gt_count": 0,
                        "b0": {deadline: 0 for deadline in RECALL_DEADLINES_MS},
                        "product": {deadline: 0 for deadline in RECALL_DEADLINES_MS},
                        "recovered": {deadline: 0 for deadline in RECALL_DEADLINES_MS},
                        "source_samples": 0,
                        "active_speech_samples": 0,
                        "extra_false_cuts": 0,
                    },
                )
                acc["gt_count"] += 1
                for deadline in RECALL_DEADLINES_MS:
                    acc["b0"][deadline] += int(index in b0_sets[deadline])
                    acc["product"][deadline] += int(index in product_sets[deadline])
                    acc["recovered"][deadline] += int(index in recovered_sets[deadline])
        extra_false = len(case_eval.added_detector_cuts) - len(
            case_eval.recovered_indices[ATTRIBUTION_HORIZON_MS]
        )
        for axis in case_axes:
            bucket = str(getattr(case, axis))
            acc = accumulators.setdefault(
                (axis, bucket),
                {
                    "gt_count": 0,
                    "b0": {deadline: 0 for deadline in RECALL_DEADLINES_MS},
                    "product": {deadline: 0 for deadline in RECALL_DEADLINES_MS},
                    "recovered": {deadline: 0 for deadline in RECALL_DEADLINES_MS},
                    "source_samples": 0,
                    "active_speech_samples": 0,
                    "extra_false_cuts": 0,
                },
            )
            acc["source_samples"] += case.length_samples
            acc["active_speech_samples"] += case.active_speech_samples
            acc["extra_false_cuts"] += extra_false
    result: dict[str, Any] = {}
    for (axis, bucket), acc in sorted(accumulators.items()):
        gt_count = int(acc["gt_count"])
        entry: dict[str, Any] = {
            "gt_count": gt_count,
            "b0_matched_at_ms": {
                str(deadline): int(acc["b0"][deadline]) for deadline in RECALL_DEADLINES_MS
            },
            "product_matched_at_ms": {
                str(deadline): int(acc["product"][deadline]) for deadline in RECALL_DEADLINES_MS
            },
            "recovered_b0_misses_at_ms": {
                str(deadline): int(acc["recovered"][deadline]) for deadline in RECALL_DEADLINES_MS
            },
        }
        if axis in change_axes:
            entry["product_recall_at_ms"] = {
                str(deadline): (int(acc["product"][deadline]) / gt_count if gt_count else 0.0)
                for deadline in RECALL_DEADLINES_MS
            }
        if int(acc["source_samples"]) > 0:
            source_hours = int(acc["source_samples"]) / (16000.0 * 3600.0)
            entry["extra_false_cuts"] = int(acc["extra_false_cuts"])
            entry["extra_false_per_source_hour"] = rate_with_interval(
                int(acc["extra_false_cuts"]), source_hours
            )
        result.setdefault(axis, {})[bucket] = entry
    return result


def compact_row(evaluation: ProfileEvaluation) -> dict[str, Any]:
    aggregate = evaluation.aggregate
    incremental = aggregate["incremental"]
    return {
        "profile_id": evaluation.profile_id,
        "family": evaluation.family,
        "checkpoint": evaluation.checkpoint,
        "profile_kind": evaluation.profile_kind,
        "params": evaluation.params,
        "case_count": aggregate["case_count"],
        "gt_change_count": aggregate["gt_change_count"],
        "source_hours": aggregate["source_hours"],
        "active_speech_hours": aggregate["active_speech_hours"],
        "b0_matched_at_ms": {
            deadline: value["matched"]
            for deadline, value in aggregate["b0"]["recall_at_ms"].items()
        },
        "product_matched_at_ms": {
            deadline: value["matched"]
            for deadline, value in aggregate["product"]["recall_at_ms"].items()
        },
        "detector_only_matched_at_ms": {
            deadline: value["matched"]
            for deadline, value in aggregate["detector_only"]["recall_at_ms"].items()
        },
        "recovered_b0_misses_at_ms": incremental["recovered_b0_misses_at_ms"],
        "regressed_b0_matches_at_ms": incremental["regressed_b0_matches_at_ms"],
        "added_logical_cuts": incremental["added_logical_cuts"],
        "extra_false_cuts": incremental["extra_false_cuts"],
        "product_false_cuts": aggregate["product"]["false_cuts"],
        "remaining_false_merges_at_2000ms": aggregate["product"][
            "remaining_false_merges_at_2000ms"
        ],
        "incremental_false_per_source_hour": aggregate["rates"][
            "incremental_false_per_source_hour"
        ],
        "incremental_false_per_active_speech_hour": aggregate["rates"][
            "incremental_false_per_active_speech_hour"
        ],
        "incremental_false_per_5min_source_session": aggregate["rates"][
            "incremental_false_per_5min_source_session"
        ],
        "coalescing": aggregate["coalescing"],
        "timing": aggregate["timing"],
        "compute": evaluation.compute,
    }


def scheduling_backlog_ms(service_seconds: list[float], chunk_samples: int = 512) -> dict[str, Any]:
    interval = chunk_samples / 16000.0
    backlog = 0.0
    delays: list[float] = []
    for service in service_seconds:
        delays.append((backlog + service) * 1000.0)
        backlog = max(0.0, backlog + service - interval)
    return {
        "completion_delay_ms": percentile_summary(delays),
        "final_backlog_ms": backlog * 1000.0,
        "overloaded": backlog > 0.0,
    }
