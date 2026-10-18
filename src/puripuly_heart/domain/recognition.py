from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal
from uuid import UUID

if TYPE_CHECKING:
    from puripuly_heart.core.audio.format import AudioCaptureSpan
    from puripuly_heart.core.audio.ownership import AudioSegmentSettingsSnapshot
    from puripuly_heart.core.stt.backend import STTNativeProvenance
    from puripuly_heart.domain.models import SpeakerAttribution


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
    source_binding: Literal["stream_scoped", "local_segment", "request_owned", "composition"] = (
        "stream_scoped"
    )
    order_basis: Literal["provider_receipt", "source_order", "capture_order"] = "provider_receipt"
    request_id: str | None = None
    contribution_id: str | None = None
    contribution_range: tuple[int, int] | None = None
    source_ranges: tuple[AudioCaptureSpan, ...] = ()
    context_ranges: tuple[AudioCaptureSpan, ...] = ()
    display_group_id: UUID | None = None
    ownership: Literal["request", "stream", "composition", "unresolved"] = "stream"
    audio_settings: AudioSegmentSettingsSnapshot | None = None
    source_available_at_monotonic_s: float | None = None
    speaker_attribution: SpeakerAttribution | None = None
    native_provenance: tuple[STTNativeProvenance, ...] = ()
    speaker_proof_id: str | None = None

    def __post_init__(self) -> None:
        if self.text_start < 0 or self.text_end < self.text_start:
            raise ValueError("invalid source origin text range")
        if self.identity.receipt_sequence < 1:
            raise ValueError("source origin requires authoritative receipt order")
        if self.ownership == "unresolved":
            raise ValueError("unresolved source ownership cannot be admitted")
        if self.ownership == "request" and not self.request_id:
            raise ValueError("request ownership requires a request identity")
        if self.contribution_range is not None:
            start, end = self.contribution_range
            if start < 0 or end - start != self.text_end - self.text_start:
                raise ValueError("source contribution range does not conserve text")
        if any(
            span.capture_epoch != self.identity.stream.capture_epoch for span in self.source_ranges
        ):
            raise ValueError("source ranges belong to another capture epoch")


def recognition_separator_origin(
    anchor: RecognitionTextOrigin, start: int
) -> RecognitionTextOrigin:
    return RecognitionTextOrigin(
        anchor.identity,
        start,
        start + 1,
        source_binding="composition",
        order_basis=anchor.order_basis,
        display_group_id=anchor.display_group_id,
        ownership="composition",
    )


def validate_recognition_origins(origins: tuple[RecognitionTextOrigin, ...], text: str) -> None:
    if not origins:
        return
    cursor = 0
    for origin in origins:
        if origin.text_start != cursor or origin.text_end > len(text):
            raise ValueError("source provenance does not conserve text")
        if origin.source_binding == "composition" and (
            origin.ownership != "composition"
            or text[origin.text_start : origin.text_end] != " "
            or origin.native is not None
            or origin.source_ranges
            or origin.context_ranges
            or origin.contribution_range is not None
        ):
            raise ValueError("source provenance has invalid composition separator")
        cursor = origin.text_end
    if cursor != len(text):
        raise ValueError("source provenance leaves unresolved text")


def slice_recognition_origins(
    origins: tuple[RecognitionTextOrigin, ...], start: int, end: int
) -> tuple[RecognitionTextOrigin, ...]:
    return tuple(
        replace(
            origin,
            text_start=max(origin.text_start, start) - start,
            text_end=min(origin.text_end, end) - start,
            contribution_range=(
                (
                    origin.contribution_range[0]
                    + max(origin.text_start, start)
                    - origin.text_start,
                    origin.contribution_range[0] + min(origin.text_end, end) - origin.text_start,
                )
                if origin.contribution_range is not None
                else None
            ),
        )
        for origin in origins
        if origin.text_start < end and origin.text_end > start
    )
