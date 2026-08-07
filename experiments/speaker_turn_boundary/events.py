from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from experiments.speaker_turn_boundary.config import CANONICAL_SAMPLE_RATE_HZ


class EventValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SpeakerBoundaryEvent:
    audio_epoch: int
    boundary_source_sample: int
    observed_source_sample_at_emit: int
    emitted_monotonic_ns: int
    confidence: float | None
    source: str
    debug: dict[str, Any]

    def __post_init__(self) -> None:
        if self.audio_epoch < 0:
            raise EventValidationError(f"audio_epoch must be >= 0, got {self.audio_epoch}")
        if self.boundary_source_sample < 0:
            raise EventValidationError(
                f"boundary_source_sample must be >= 0, got {self.boundary_source_sample}"
            )
        if self.observed_source_sample_at_emit < 0:
            raise EventValidationError(
                "observed_source_sample_at_emit must be >= 0, got "
                f"{self.observed_source_sample_at_emit}"
            )
        if self.boundary_source_sample > self.observed_source_sample_at_emit:
            raise EventValidationError(
                "boundary_source_sample cannot exceed observed_source_sample_at_emit "
                f"({self.boundary_source_sample} > {self.observed_source_sample_at_emit})"
            )
        if self.emitted_monotonic_ns < 0:
            raise EventValidationError(
                f"emitted_monotonic_ns must be >= 0, got {self.emitted_monotonic_ns}"
            )
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise EventValidationError(
                f"confidence must be in [0, 1] or None, got {self.confidence}"
            )
        if not self.source:
            raise EventValidationError("source must be a non-empty string")

    @property
    def event_lookback_ms(self) -> float:
        return (
            (self.observed_source_sample_at_emit - self.boundary_source_sample)
            / CANONICAL_SAMPLE_RATE_HZ
            * 1000.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_epoch": self.audio_epoch,
            "boundary_source_sample": self.boundary_source_sample,
            "observed_source_sample_at_emit": self.observed_source_sample_at_emit,
            "emitted_monotonic_ns": self.emitted_monotonic_ns,
            "confidence": self.confidence,
            "source": self.source,
            "debug": dict(sorted(self.debug.items(), key=lambda item: item[0])),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpeakerBoundaryEvent":
        return cls(
            audio_epoch=data["audio_epoch"],
            boundary_source_sample=data["boundary_source_sample"],
            observed_source_sample_at_emit=data["observed_source_sample_at_emit"],
            emitted_monotonic_ns=data["emitted_monotonic_ns"],
            confidence=data["confidence"],
            source=data["source"],
            debug=dict(data.get("debug") or {}),
        )


@dataclass(frozen=True, slots=True)
class DetectorProgress:
    audio_epoch: int
    observed_source_sample: int
    safe_boundary_frontier_sample: int

    def __post_init__(self) -> None:
        if self.audio_epoch < 0:
            raise EventValidationError(f"audio_epoch must be >= 0, got {self.audio_epoch}")
        if self.observed_source_sample < 0:
            raise EventValidationError(
                f"observed_source_sample must be >= 0, got {self.observed_source_sample}"
            )
        if self.safe_boundary_frontier_sample < 0:
            raise EventValidationError(
                "safe_boundary_frontier_sample must be >= 0, got "
                f"{self.safe_boundary_frontier_sample}"
            )
        if self.safe_boundary_frontier_sample > self.observed_source_sample:
            raise EventValidationError(
                "safe_boundary_frontier_sample cannot exceed observed_source_sample "
                f"({self.safe_boundary_frontier_sample} > {self.observed_source_sample})"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_epoch": self.audio_epoch,
            "observed_source_sample": self.observed_source_sample,
            "safe_boundary_frontier_sample": self.safe_boundary_frontier_sample,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectorProgress":
        return cls(
            audio_epoch=data["audio_epoch"],
            observed_source_sample=data["observed_source_sample"],
            safe_boundary_frontier_sample=data["safe_boundary_frontier_sample"],
        )
