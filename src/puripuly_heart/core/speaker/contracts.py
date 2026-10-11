from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from puripuly_heart.core.audio.format import AudioCaptureSpan


@dataclass(frozen=True, slots=True)
class BackendReady:
    tail_padding_samples: int
    max_revision_samples: int
    diagnostics: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ProbabilityFrame:
    start_sample: int
    end_sample: int
    probabilities: tuple[float, ...]
    active_slots: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class BackendUpdate:
    processed_through_sample: int
    immutable_through_sample: int
    frames: tuple[ProbabilityFrame, ...] = ()


class SpeakerBackend(Protocol):
    async def start(self) -> BackendReady: ...

    async def reset(self, epoch: int) -> None: ...

    async def feed(self, pcm: np.ndarray, compact_start_sample: int) -> BackendUpdate: ...

    async def pause(self) -> BackendUpdate: ...

    async def stop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RuntimeLimits:
    max_input_samples: int = 16_000
    max_pending_samples: int = 64_000
    max_pending_items: int = 128
    max_map_pieces: int = 2048
    max_frames_per_update: int = 1024
    max_evidence_items: int = 2048
    max_evidence_map_pieces: int = 8192
    max_capture_sessions: int = 1024
    max_backend_lag_samples: int = 64_000
    max_tail_padding_samples: int = 32_000

    def __post_init__(self) -> None:
        if any(getattr(self, name) <= 0 for name in self.__dataclass_fields__):
            raise ValueError("speaker capacities must be positive")
        if self.max_evidence_map_pieces < self.max_map_pieces:
            raise ValueError("evidence mapping capacity must accommodate one bounded source map")


@dataclass(frozen=True, slots=True)
class SourceMapPiece:
    compact_start_sample: int
    compact_end_sample: int
    capture: AudioCaptureSpan | None
    kind: str


@dataclass(frozen=True, slots=True)
class SpeakerEvidence:
    capture_session_id: str
    speaker_epoch: int
    observation_ready_monotonic_s: float
    source_map: tuple[SourceMapPiece, ...]
    processed_through_sample: int
    immutable_through_sample: int
    frame: ProbabilityFrame | None = None
    range_state: str = "closed"
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SpeakerSnapshot:
    state: str
    ready: bool
    speaker_epoch: int
    capture_session_id: str | None
    paused: bool
    pending_samples: int
    pending_items: int
    retained_map_pieces: int
    onset_samples: int
    evidence_items: int
    evidence_map_pieces: int
    evidence_evictions: int
    overloads: int
    stale_events: int
    copied_samples: int
    max_copy_samples: int
    owned_tasks: int
    processed_through_sample: int
    immutable_through_sample: int
    limits: RuntimeLimits
    diagnostics: tuple[tuple[str, str], ...]
    error: str | None
