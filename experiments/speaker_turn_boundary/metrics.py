from __future__ import annotations

from dataclasses import dataclass

from experiments.speaker_turn_boundary.coalescing import CoalescingReport, LogicalCut
from experiments.speaker_turn_boundary.config import VAD_COALESCE_WINDOW_MS
from experiments.speaker_turn_boundary.events import SpeakerBoundaryEvent
from experiments.speaker_turn_boundary.ground_truth import SpeakerChangeGT

RECALL_DEADLINES_MS = (250, 500, 1000, 1500, 2000)
PRODUCT_FALSE_CUT_TOLERANCE_MS = VAD_COALESCE_WINDOW_MS
SPEECH_SAMPLES_PER_HOUR = 16000 * 3600


@dataclass(frozen=True, slots=True)
class CaseBoundaryMetrics:
    case_id: str
    audio_epoch: int
    gt_change_count: int
    recall_at_ms: dict[int, float]
    recall_matched_counts: dict[int, int]
    detector_only_recall_at_ms: dict[int, float]
    detector_only_recall_matched_counts: dict[int, int]
    product_false_cuts: int
    detector_only_false_cuts: int
    b0_cut_count: int
    detector_cut_count: int
    active_speech_samples: int

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "audio_epoch": self.audio_epoch,
            "gt_change_count": self.gt_change_count,
            "recall_at_ms": {str(ms): value for ms, value in self.recall_at_ms.items()},
            "recall_matched_counts": {
                str(ms): value for ms, value in self.recall_matched_counts.items()
            },
            "detector_only_recall_at_ms": {
                str(ms): value for ms, value in self.detector_only_recall_at_ms.items()
            },
            "detector_only_recall_matched_counts": {
                str(ms): value for ms, value in self.detector_only_recall_matched_counts.items()
            },
            "product_false_cuts": self.product_false_cuts,
            "detector_only_false_cuts": self.detector_only_false_cuts,
            "b0_cut_count": self.b0_cut_count,
            "detector_cut_count": self.detector_cut_count,
            "active_speech_samples": self.active_speech_samples,
        }


@dataclass(frozen=True, slots=True)
class SweepAggregate:
    profile_id: str
    case_count: int
    smoke_case_count: int
    gt_change_count: int
    recall_at_ms: dict[int, float]
    detector_only_recall_at_ms: dict[int, float]
    product_false_cuts_total: int
    detector_only_false_cuts_total: int
    false_cuts_per_speech_hour: float
    detector_only_false_cuts_per_speech_hour: float
    active_speech_samples: int
    b0_cut_count_total: int
    detector_cut_count_total: int
    detector_events_total: int
    coalesced_count_total: int
    duplicate_count_total: int
    stale_detector_events_total: int
    smoke_detector_cut_count_total: int

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "case_count": self.case_count,
            "smoke_case_count": self.smoke_case_count,
            "gt_change_count": self.gt_change_count,
            "recall_at_ms": {str(ms): value for ms, value in self.recall_at_ms.items()},
            "detector_only_recall_at_ms": {
                str(ms): value for ms, value in self.detector_only_recall_at_ms.items()
            },
            "product_false_cuts_total": self.product_false_cuts_total,
            "detector_only_false_cuts_total": self.detector_only_false_cuts_total,
            "false_cuts_per_speech_hour": self.false_cuts_per_speech_hour,
            "detector_only_false_cuts_per_speech_hour": (
                self.detector_only_false_cuts_per_speech_hour
            ),
            "active_speech_samples": self.active_speech_samples,
            "b0_cut_count_total": self.b0_cut_count_total,
            "detector_cut_count_total": self.detector_cut_count_total,
            "detector_events_total": self.detector_events_total,
            "coalesced_count_total": self.coalesced_count_total,
            "duplicate_count_total": self.duplicate_count_total,
            "stale_detector_events_total": self.stale_detector_events_total,
            "smoke_detector_cut_count_total": self.smoke_detector_cut_count_total,
        }


def match_gt_to_cuts(
    gt_changes: list[SpeakerChangeGT],
    cuts: list[LogicalCut],
    *,
    window_samples: int,
) -> list[int]:
    changes_by_epoch: dict[int, list[SpeakerChangeGT]] = {}
    for change in gt_changes:
        changes_by_epoch.setdefault(change.audio_epoch, []).append(change)
    cuts_by_epoch: dict[int, list[LogicalCut]] = {}
    for cut in cuts:
        cuts_by_epoch.setdefault(cut.audio_epoch, []).append(cut)
    matched: list[int] = []
    for epoch, changes in changes_by_epoch.items():
        epoch_cuts = sorted(
            cuts_by_epoch.get(epoch, []),
            key=lambda cut: (cut.sample, cut.kind, cut.ref_event_index),
        )
        ordered_changes = sorted(changes, key=lambda change: change.change_sample)
        cut_index = 0
        for change in ordered_changes:
            while cut_index < len(epoch_cuts):
                cut = epoch_cuts[cut_index]
                if cut.sample < change.change_sample - window_samples:
                    cut_index += 1
                    continue
                if cut.sample > change.change_sample + window_samples:
                    break
                matched.append(change.change_sample)
                cut_index += 1
                break
    return matched


def matched_false_cut_count(
    gt_changes: list[SpeakerChangeGT],
    cuts: list[LogicalCut],
    *,
    window_samples: int,
) -> int:
    if not cuts:
        return 0
    matched = match_gt_to_cuts(gt_changes, cuts, window_samples=window_samples)
    return len(cuts) - len(matched)


def _recall_series(
    gt_changes: list[SpeakerChangeGT],
    cuts: list[LogicalCut],
) -> tuple[dict[int, float], dict[int, int]]:
    gt_count = len(gt_changes)
    recalls: dict[int, float] = {}
    matched_counts: dict[int, int] = {}
    for deadline_ms in RECALL_DEADLINES_MS:
        matched = match_gt_to_cuts(gt_changes, cuts, window_samples=deadline_ms * 16)
        matched_counts[deadline_ms] = len(matched)
        recalls[deadline_ms] = len(matched) / gt_count if gt_count else 0.0
    return recalls, matched_counts


def evaluate_case(
    *,
    case_id: str,
    audio_epoch: int,
    gt_changes: list[SpeakerChangeGT],
    cuts: list[LogicalCut],
    detector_events: list[LogicalCut],
    vad_cut_count: int,
    detector_cut_count: int,
    active_speech_samples: int,
) -> CaseBoundaryMetrics:
    gt_count = len(gt_changes)
    recalls, matched_counts = _recall_series(gt_changes, cuts)
    detector_recalls, detector_matched_counts = _recall_series(gt_changes, detector_events)
    false_window = PRODUCT_FALSE_CUT_TOLERANCE_MS * 16
    product_false_cuts = matched_false_cut_count(gt_changes, cuts, window_samples=false_window)
    detector_only_false_cuts = matched_false_cut_count(
        gt_changes, detector_events, window_samples=false_window
    )
    return CaseBoundaryMetrics(
        case_id=case_id,
        audio_epoch=audio_epoch,
        gt_change_count=gt_count,
        recall_at_ms=recalls,
        recall_matched_counts=matched_counts,
        detector_only_recall_at_ms=detector_recalls,
        detector_only_recall_matched_counts=detector_matched_counts,
        product_false_cuts=product_false_cuts,
        detector_only_false_cuts=detector_only_false_cuts,
        b0_cut_count=vad_cut_count,
        detector_cut_count=detector_cut_count,
        active_speech_samples=active_speech_samples,
    )


def aggregate_cases(
    metrics: list[CaseBoundaryMetrics],
    *,
    profile_id: str,
    coalescing_reports: list[CoalescingReport] | None = None,
    smoke_metrics: list[CaseBoundaryMetrics] | None = None,
) -> SweepAggregate:
    gt_total = sum(item.gt_change_count for item in metrics)
    recall: dict[int, float] = {}
    detector_recall: dict[int, float] = {}
    for deadline_ms in RECALL_DEADLINES_MS:
        matched = sum(item.recall_matched_counts[deadline_ms] for item in metrics)
        recall[deadline_ms] = matched / gt_total if gt_total else 0.0
        detector_matched = sum(
            item.detector_only_recall_matched_counts[deadline_ms] for item in metrics
        )
        detector_recall[deadline_ms] = detector_matched / gt_total if gt_total else 0.0
    product_false_cuts = sum(item.product_false_cuts for item in metrics)
    detector_only_false_cuts = sum(item.detector_only_false_cuts for item in metrics)
    active_speech_samples = sum(item.active_speech_samples for item in metrics)
    speech_hours = active_speech_samples / SPEECH_SAMPLES_PER_HOUR
    per_hour = product_false_cuts / speech_hours if speech_hours > 0 else float("inf")
    detector_per_hour = (
        detector_only_false_cuts / speech_hours if speech_hours > 0 else float("inf")
    )
    reports = coalescing_reports or []
    return SweepAggregate(
        profile_id=profile_id,
        case_count=len(metrics),
        smoke_case_count=len(smoke_metrics or []),
        gt_change_count=gt_total,
        recall_at_ms=recall,
        detector_only_recall_at_ms=detector_recall,
        product_false_cuts_total=product_false_cuts,
        detector_only_false_cuts_total=detector_only_false_cuts,
        false_cuts_per_speech_hour=per_hour,
        detector_only_false_cuts_per_speech_hour=detector_per_hour,
        active_speech_samples=active_speech_samples,
        b0_cut_count_total=sum(item.b0_cut_count for item in metrics),
        detector_cut_count_total=sum(item.detector_cut_count for item in metrics),
        detector_events_total=sum(report.detector_events_total for report in reports),
        coalesced_count_total=sum(report.coalesced_count for report in reports),
        duplicate_count_total=sum(report.duplicate_count for report in reports),
        stale_detector_events_total=sum(report.stale_detector_events for report in reports),
        smoke_detector_cut_count_total=sum(
            item.detector_cut_count for item in (smoke_metrics or [])
        ),
    )


def incremental_over_b0(
    baseline: SweepAggregate,
    candidate: SweepAggregate,
) -> dict[str, object]:
    primary = RECALL_DEADLINES_MS[1]
    incremental_false_cuts = max(
        0, candidate.product_false_cuts_total - baseline.product_false_cuts_total
    )
    base_hour = baseline.false_cuts_per_speech_hour
    candidate_hour = candidate.false_cuts_per_speech_hour
    if (
        base_hour == float("inf")
        or candidate_hour == float("inf")
        or base_hour != base_hour
        or candidate_hour != candidate_hour
    ):
        incremental_per_hour: float | None = None
    else:
        incremental_per_hour = max(0.0, candidate_hour - base_hour)
    return {
        "b0_recall_at_500ms": baseline.recall_at_ms[primary],
        "candidate_recall_at_500ms": candidate.recall_at_ms[primary],
        "incremental_recall_at_500ms": max(
            0.0, candidate.recall_at_ms[primary] - baseline.recall_at_ms[primary]
        ),
        "b0_product_false_cuts_total": baseline.product_false_cuts_total,
        "candidate_product_false_cuts_total": candidate.product_false_cuts_total,
        "incremental_false_cuts": incremental_false_cuts,
        "b0_false_cuts_per_speech_hour": baseline.false_cuts_per_speech_hour,
        "candidate_false_cuts_per_speech_hour": candidate.false_cuts_per_speech_hour,
        "incremental_false_cuts_per_speech_hour": incremental_per_hour,
    }


def logical_cut_counts(cuts: list[LogicalCut]) -> tuple[int, int]:
    vad_count = sum(1 for cut in cuts if cut.kind == "vad")
    detector_count = sum(1 for cut in cuts if cut.kind == "detector")
    return vad_count, detector_count


def vad_events_to_boundaries(vad_boundaries: list[SpeakerBoundaryEvent]) -> list[LogicalCut]:
    return [
        LogicalCut(
            audio_epoch=boundary.audio_epoch,
            sample=boundary.boundary_source_sample,
            kind="vad",
            ref_event_index=index,
        )
        for index, boundary in enumerate(vad_boundaries)
    ]


def detector_events_to_cuts(
    detector_boundaries: list[SpeakerBoundaryEvent],
) -> list[LogicalCut]:
    return [
        LogicalCut(
            audio_epoch=boundary.audio_epoch,
            sample=boundary.boundary_source_sample,
            kind="detector",
            ref_event_index=index,
        )
        for index, boundary in enumerate(detector_boundaries)
    ]
