from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable, Sequence

import numpy as np


class RelativeState(StrEnum):
    NONE = "NONE"
    ANCHOR_ONLY = "ANCHOR_ONLY"
    ANCHOR_PLUS_OTHER = "ANCHOR_PLUS_OTHER"
    OTHER_ONLY = "OTHER_ONLY"


class AnchorLifecycle(StrEnum):
    UNANCHORED = "UNANCHORED"
    ANCHORED = "ANCHORED"
    ANCHOR_UNCERTAIN = "ANCHOR_UNCERTAIN"


@dataclass(frozen=True, slots=True)
class ActivityInterval:
    start_sample: int
    end_sample: int
    active_speakers: tuple[str, ...]
    masked: bool

    def __post_init__(self) -> None:
        if self.start_sample < 0 or self.end_sample <= self.start_sample:
            raise ValueError("invalid activity interval")
        if tuple(sorted(set(self.active_speakers))) != self.active_speakers:
            raise ValueError("active speakers must be sorted and unique")

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> ActivityInterval:
        explicit_mask = row.get("masked", False)
        if "masked" in row and not isinstance(explicit_mask, bool):
            raise ValueError("masked must be a boolean")
        masks = row.get("handoff_relation_mask_classes") or []
        return cls(
            start_sample=int(row["start_sample"]),
            end_sample=int(row["end_sample"]),
            active_speakers=tuple(sorted(str(value) for value in row["active_speakers"])),
            masked=explicit_mask
            or bool(row.get("ambiguous", False))
            or not bool(row.get("speaker_identity_known", True))
            or bool(masks),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "active_speakers": list(self.active_speakers),
            "masked": self.masked,
        }


@dataclass(frozen=True, slots=True)
class EvaluationCell:
    index: int
    start_sample: int
    end_sample: int
    center_sample: int
    active_speakers: tuple[str, ...]
    masked: bool

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample


@dataclass(frozen=True, slots=True)
class AnchorEpisode:
    episode_id: str
    source_id: str
    anchor_speaker: str
    opportunity_start_sample: int
    anchor_emit_sample: int
    end_emit_sample: int
    replacement_boundary_sample: int | None

    def contains_center(self, sample: int) -> bool:
        return self.anchor_emit_sample <= sample < self.end_emit_sample


@dataclass(frozen=True, slots=True)
class Trace:
    source_id: str
    family: str
    probabilities: np.ndarray
    frame_start_samples: np.ndarray
    frame_end_samples: np.ndarray
    evidence_frontier_samples: np.ndarray
    slot_alive: np.ndarray
    state_reset: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        frame_count = self.probabilities.shape[0]
        if self.probabilities.ndim != 2 or self.probabilities.shape[1] < 1:
            raise ValueError("trace probabilities must be [frames, slots]")
        for values in (
            self.frame_start_samples,
            self.frame_end_samples,
            self.evidence_frontier_samples,
            self.state_reset,
        ):
            if values.shape != (frame_count,):
                raise ValueError("trace vector geometry mismatch")
        if self.slot_alive.shape != self.probabilities.shape:
            raise ValueError("slot-alive geometry mismatch")
        if np.any(self.frame_end_samples <= self.frame_start_samples):
            raise ValueError("trace frame supports are invalid")
        if frame_count and not bool(self.state_reset[0]):
            raise ValueError("trace must mark the initial model-state reset")
        if np.any(self.frame_start_samples[1:] < self.frame_end_samples[:-1]):
            raise ValueError("trace frame supports overlap or regress")
        if np.any(self.evidence_frontier_samples < self.frame_end_samples):
            raise ValueError("evidence frontier precedes used audio")
        if np.any(self.evidence_frontier_samples[1:] < self.evidence_frontier_samples[:-1]):
            raise ValueError("evidence frontier regresses")
        if not np.isfinite(self.probabilities).all():
            raise ValueError("trace probabilities are non-finite")
        if np.any((self.probabilities < 0.0) | (self.probabilities > 1.0)):
            raise ValueError("trace probabilities are outside [0, 1]")


def relative_state(anchor_speaker: str, active_speakers: Iterable[str]) -> RelativeState:
    speakers = set(active_speakers)
    anchor_present = anchor_speaker in speakers
    other_present = any(speaker != anchor_speaker for speaker in speakers)
    if anchor_present and other_present:
        return RelativeState.ANCHOR_PLUS_OTHER
    if anchor_present:
        return RelativeState.ANCHOR_ONLY
    if other_present:
        return RelativeState.OTHER_ONLY
    return RelativeState.NONE


def evaluation_cells(
    intervals: Sequence[ActivityInterval],
    scored_start_sample: int,
    scored_end_sample: int,
    cell_samples: int = 1600,
) -> tuple[EvaluationCell, ...]:
    if cell_samples <= 0:
        raise ValueError("cell_samples must be positive")
    if not intervals or intervals[0].start_sample != scored_start_sample:
        raise ValueError("intervals do not start at the scored boundary")
    if intervals[-1].end_sample != scored_end_sample:
        raise ValueError("intervals do not end at the scored boundary")
    if any(
        previous.end_sample != current.start_sample
        for previous, current in zip(intervals, intervals[1:], strict=False)
    ):
        raise ValueError("canonical interval timeline is not contiguous")
    cells: list[EvaluationCell] = []
    interval_index = 0
    start = scored_start_sample
    index = 0
    while start + cell_samples // 2 < scored_end_sample:
        center = start + cell_samples // 2
        while intervals[interval_index].end_sample <= center:
            interval_index += 1
        interval = intervals[interval_index]
        if not interval.start_sample <= center < interval.end_sample:
            raise ValueError("canonical interval timeline has a gap")
        end = min(start + cell_samples, scored_end_sample)
        cells.append(
            EvaluationCell(
                index=index,
                start_sample=start,
                end_sample=end,
                center_sample=center,
                active_speakers=interval.active_speakers,
                masked=interval.masked,
            )
        )
        index += 1
        start += cell_samples
    return tuple(cells)


def sample_trace_at_cells(trace: Trace, cells: Sequence[EvaluationCell]) -> dict[str, np.ndarray]:
    centers = np.asarray([cell.center_sample for cell in cells], dtype=np.int64)
    probabilities = np.zeros((len(cells), trace.probabilities.shape[1]), dtype=np.float32)
    alive = np.zeros_like(probabilities, dtype=np.bool_)
    frontiers = np.full(len(cells), -1, dtype=np.int64)
    resets = np.zeros(len(cells), dtype=np.bool_)
    if trace.frame_end_samples.size == 0:
        return {
            "probabilities": probabilities,
            "slot_alive": alive,
            "evidence_frontier_samples": frontiers,
            "state_reset": resets,
            "trace_valid": np.zeros(len(cells), dtype=np.bool_),
        }
    indices = np.searchsorted(trace.frame_end_samples, centers, side="right")
    valid = indices < trace.frame_end_samples.size
    valid &= np.where(
        valid,
        trace.frame_start_samples[np.minimum(indices, trace.frame_start_samples.size - 1)]
        <= centers,
        False,
    )
    if valid.any():
        selected = indices[valid]
        probabilities[valid] = trace.probabilities[selected]
        alive[valid] = trace.slot_alive[selected]
        frontiers[valid] = trace.evidence_frontier_samples[selected]
        resets[valid] = trace.state_reset[selected]
    return {
        "probabilities": probabilities,
        "slot_alive": alive,
        "evidence_frontier_samples": frontiers,
        "state_reset": resets,
        "trace_valid": valid,
    }
