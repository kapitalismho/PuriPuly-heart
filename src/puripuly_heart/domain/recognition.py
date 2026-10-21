from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal
from uuid import UUID


@dataclass(frozen=True, slots=True)
class RecognitionStreamIdentity:
    channel: Literal["self", "peer"]
    activation_generation: int
    capture_epoch: int
    provider_epoch_id: str
    settings_scope: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class RecognitionUnitIdentity:
    stream: RecognitionStreamIdentity
    unit_id: UUID
    receipt_sequence: int


@dataclass(frozen=True, slots=True)
class NativeAudioCoverage:
    input_start_sample: int
    input_end_sample: int
    capture_epoch: int
    source_sample_rate_hz: int
    source_start_sample: int
    source_end_sample: int
    normalized_sample_rate_hz: int
    normalized_start_sample: int
    normalized_end_sample: int
    source_start_monotonic_s: float
    source_end_monotonic_s: float
    context_only: bool
    discontinuity_kind: Literal["known_loss", "unknown_loss"] | None = None
    discontinuity_observed_at_monotonic_s: float | None = None
    lost_source_samples: int | None = None


@dataclass(frozen=True, slots=True)
class NativeWordEvidence:
    text: str | None
    punctuation: str | None
    begin_time_ms: float | None
    end_time_ms: float | None


@dataclass(frozen=True, slots=True)
class NativeTranscriptionEvidence:
    finished: bool | None = None
    language_code: str | None = None
    speaker_label: str | None = None
    words: tuple[tuple[str | None, str | None, str | None], ...] = ()
    native_task_id: str | None = None
    native_sentence_id: str | None = None
    timestamp_basis: Literal["task_relative_estimated"] | None = None
    begin_time_ms: float | None = None
    end_time_ms: float | None = None
    input_sample_rate_hz: int | None = None
    task_input_samples: int | None = None
    timed_words: tuple[NativeWordEvidence, ...] = ()
    audio_coverage: tuple[NativeAudioCoverage, ...] = ()

    @property
    def retained_bytes(self) -> int:
        return (
            len(self.words) * 64
            + (64 if self.timestamp_basis is not None else 0)
            + sum(
                len(value.encode("utf-8"))
                for value in (
                    self.language_code,
                    self.speaker_label,
                    self.native_task_id,
                    self.native_sentence_id,
                )
                if value is not None
            )
            + sum(
                len(value.encode("utf-8"))
                for word in self.words
                for value in word
                if value is not None
            )
            + len(self.timed_words) * 64
            + sum(
                len(value.encode("utf-8"))
                for word in self.timed_words
                for value in (word.text, word.punctuation)
                if value is not None
            )
            + len(self.audio_coverage) * 192
        )


@dataclass(frozen=True, slots=True)
class RecognitionTextOrigin:
    identity: RecognitionUnitIdentity
    text_start: int
    text_end: int
    native: NativeTranscriptionEvidence | None = None
    source_binding: Literal["stream_scoped", "local_segment"] = "stream_scoped"
    order_basis: Literal["provider_receipt", "source_order"] = "provider_receipt"


def slice_recognition_origins(
    origins: tuple[RecognitionTextOrigin, ...], start: int, end: int
) -> tuple[RecognitionTextOrigin, ...]:
    return tuple(
        replace(
            origin,
            text_start=max(origin.text_start, start) - start,
            text_end=min(origin.text_end, end) - start,
        )
        for origin in origins
        if origin.text_start < end and origin.text_end > start
    )
