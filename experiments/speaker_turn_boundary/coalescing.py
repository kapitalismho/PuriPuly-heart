from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from experiments.speaker_turn_boundary.config import VAD_COALESCE_WINDOW_SAMPLES
from experiments.speaker_turn_boundary.events import SpeakerBoundaryEvent


@dataclass(frozen=True, slots=True)
class CoalesceConfig:
    window_samples: int = VAD_COALESCE_WINDOW_SAMPLES


@dataclass(frozen=True, slots=True)
class CoalescingReport:
    vad_cut_count: int
    detector_events_total: int
    stale_detector_events: int
    coalesced_count: int
    duplicate_count: int
    detector_cut_count: int
    total_logical_cuts: int

    def to_dict(self) -> dict[str, int]:
        return {
            "vad_cut_count": self.vad_cut_count,
            "detector_events_total": self.detector_events_total,
            "stale_detector_events": self.stale_detector_events,
            "coalesced_count": self.coalesced_count,
            "duplicate_count": self.duplicate_count,
            "detector_cut_count": self.detector_cut_count,
            "total_logical_cuts": self.total_logical_cuts,
        }


LogicalCutKind = Literal["vad", "detector"]


@dataclass(frozen=True, slots=True)
class LogicalCut:
    audio_epoch: int
    sample: int
    kind: LogicalCutKind
    ref_event_index: int

    def to_dict(self) -> dict[str, object]:
        return {
            "audio_epoch": self.audio_epoch,
            "sample": self.sample,
            "kind": self.kind,
            "ref_event_index": self.ref_event_index,
        }


@dataclass(frozen=True, slots=True)
class CoalescingOutcome:
    report: CoalescingReport
    cuts: list[LogicalCut]
    detections: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "report": self.report.to_dict(),
            "cuts": [cut.to_dict() for cut in self.cuts],
            "detections": self.detections,
        }


def _distance(detector: SpeakerBoundaryEvent, boundary: SpeakerBoundaryEvent) -> int:
    return abs(detector.boundary_source_sample - boundary.boundary_source_sample)


def coalesce_vad_and_detector(
    vad_boundaries: list[SpeakerBoundaryEvent],
    detector_events: list[SpeakerBoundaryEvent],
    *,
    config: CoalesceConfig = CoalesceConfig(),
) -> CoalescingOutcome:
    vad_by_epoch: dict[int, list[SpeakerBoundaryEvent]] = {}
    for boundary in vad_boundaries:
        vad_by_epoch.setdefault(boundary.audio_epoch, []).append(boundary)
    current_epoch = max(vad_by_epoch) if vad_by_epoch else None
    ordered = sorted(
        enumerate(detector_events),
        key=lambda item: (
            item[1].audio_epoch,
            item[1].boundary_source_sample,
            item[0],
        ),
    )
    absorbed: set[int] = set()
    cuts: list[LogicalCut] = []
    detections: list[dict[str, object]] = []
    stale_count = 0
    coalesced_count = 0
    duplicate_count = 0
    detector_cut_count = 0
    for index, event in ordered:
        disposition: str
        match_boundary: int | None = None
        if current_epoch is not None and event.audio_epoch < current_epoch:
            stale_count += 1
            disposition = "stale"
        else:
            candidates = vad_by_epoch.get(event.audio_epoch, [])
            best: tuple[int, int, int] | None = None
            for boundary_index, boundary in enumerate(candidates):
                distance = _distance(event, boundary)
                if distance > config.window_samples:
                    continue
                candidate = (distance, boundary.boundary_source_sample, boundary_index)
                if best is None or candidate < best:
                    best = candidate
            if best is None:
                detector_cut_count += 1
                disposition = "cut"
            else:
                _, _, boundary_index = best
                match_boundary = candidates[boundary_index].boundary_source_sample
                if boundary_index in absorbed:
                    duplicate_count += 1
                    disposition = "duplicate"
                else:
                    absorbed.add(boundary_index)
                    coalesced_count += 1
                    disposition = "coalesced"
        detections.append(
            {
                "event_index": index,
                "audio_epoch": event.audio_epoch,
                "boundary_source_sample": event.boundary_source_sample,
                "disposition": disposition,
                "matched_vad_sample": match_boundary,
            }
        )
        if disposition == "cut":
            cuts.append(
                LogicalCut(
                    audio_epoch=event.audio_epoch,
                    sample=event.boundary_source_sample,
                    kind="detector",
                    ref_event_index=index,
                )
            )
    for index, boundary in enumerate(vad_boundaries):
        cuts.append(
            LogicalCut(
                audio_epoch=boundary.audio_epoch,
                sample=boundary.boundary_source_sample,
                kind="vad",
                ref_event_index=index,
            )
        )
    cuts.sort(key=lambda cut: (cut.audio_epoch, cut.sample, cut.ref_event_index))
    vad_cut_count = len(vad_boundaries)
    report = CoalescingReport(
        vad_cut_count=vad_cut_count,
        detector_events_total=len(detector_events),
        stale_detector_events=stale_count,
        coalesced_count=coalesced_count,
        duplicate_count=duplicate_count,
        detector_cut_count=detector_cut_count,
        total_logical_cuts=vad_cut_count + detector_cut_count,
    )
    return CoalescingOutcome(report=report, cuts=cuts, detections=detections)
