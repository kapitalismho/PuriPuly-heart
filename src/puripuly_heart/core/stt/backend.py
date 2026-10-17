from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import AsyncIterator, Literal, Protocol, TypeVar, runtime_checkable

import numpy as np

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import (
    AudioSegmentIdentity,
    AudioSegmentSettingsSnapshot,
    SegmentTerminalOutcome,
)
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.runtime.local_asr_transition import LocalASRSessionOptions
from puripuly_heart.core.speech_boundary import SpeechBoundaryReason
from puripuly_heart.domain.models import FinalLanguageRun, FinalSpeakerRun
from puripuly_heart.domain.recognition import (
    NativeTranscriptionEvidence,
    RecognitionStreamIdentity,
    RecognitionTextOrigin,
    RecognitionUnitIdentity,
    slice_recognition_origins,
)


class PermanentSTTScopedSessionError(RuntimeError):
    __slots__ = ()


@dataclass(frozen=True, slots=True)
class STTSessionProjection:
    mode: Literal["legacy", "scoped"] = "legacy"
    provider_epoch_id: str | None = None

    def __post_init__(self) -> None:
        if self.mode == "scoped":
            if not self.provider_epoch_id:
                raise ValueError("scoped STT session projection requires a provider epoch")
            return
        if self.mode != "legacy":
            raise ValueError(f"unknown STT session projection mode: {self.mode!r}")
        if self.provider_epoch_id is not None:
            raise ValueError("legacy STT session projection cannot have a provider epoch")


LEGACY_STT_SESSION_PROJECTION = STTSessionProjection()


@dataclass(frozen=True, slots=True)
class STTProviderTurnIdentity:
    segment: AudioSegmentIdentity
    provider_epoch_id: str
    provider_turn_id: str
    settings_scope: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class STTProviderTurnRequest:
    identity: STTProviderTurnIdentity
    settings: AudioSegmentSettingsSnapshot
    channel: Literal["self", "peer"] = "peer"


@dataclass(frozen=True, slots=True)
class STTNativeProvenance:
    native_event_id: str | None = None
    native_request_id: str | None = None
    native_item_id: str | None = None
    native_task_id: str | None = None
    barrier: str | None = None
    from_finalize: bool | None = None
    transcription: NativeTranscriptionEvidence | None = None


@dataclass(frozen=True, slots=True)
class STTTextContribution:
    contribution_id: str
    text_start: int
    text_end: int

    def __post_init__(self) -> None:
        if not self.contribution_id:
            raise ValueError("contribution_id must be non-empty")
        if self.text_start < 0 or self.text_end < self.text_start:
            raise ValueError("invalid contribution text range")


@dataclass(frozen=True, slots=True)
class STTProviderTurnUpdate:
    identity: STTProviderTurnIdentity
    sequence: int
    stability: Literal["provisional", "stable"]
    assembly: Literal["append", "replace"]
    text: str
    final_language_runs: tuple[FinalLanguageRun, ...] = ()
    final_speaker_runs: tuple[FinalSpeakerRun, ...] = ()
    provenance: STTNativeProvenance = STTNativeProvenance()
    contribution: STTTextContribution | None = None
    recognition_origins: tuple[RecognitionTextOrigin, ...] = ()
    is_current: Callable[[], bool] | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class STTProviderTurnTerminal:
    identity: STTProviderTurnIdentity
    outcome: SegmentTerminalOutcome
    text: str = ""
    final_language_runs: tuple[FinalLanguageRun, ...] = ()
    final_speaker_runs: tuple[FinalSpeakerRun, ...] = ()
    text_authority: Literal["authoritative", "degraded", "none"] = "none"
    failure_reason: str | None = None
    epoch_disposition: Literal["reuse", "retire"] = "reuse"
    failure_retryable: bool = False
    recovery_pending: bool = False
    provenance: tuple[STTNativeProvenance, ...] = ()
    included_contributions: tuple[STTTextContribution, ...] = ()
    failure_diagnostics: ErrorDiagnostics | None = None
    recognition_origins: tuple[RecognitionTextOrigin, ...] = ()
    is_current: Callable[[], bool] | None = field(default=None, compare=False, repr=False)


_ProviderEvent = TypeVar("_ProviderEvent")


def scoped_event_for_delivery(
    event: _ProviderEvent,
) -> _ProviderEvent | STTProviderTurnTerminal | None:
    if not isinstance(event, STTProviderTurnUpdate | STTProviderTurnTerminal):
        return event
    if event.is_current is None or event.is_current():
        return event
    if isinstance(event, STTProviderTurnUpdate):
        return None
    return replace(
        event,
        outcome="cancelled",
        text="",
        final_language_runs=(),
        final_speaker_runs=(),
        text_authority="none",
        failure_reason="cancelled",
        epoch_disposition="retire",
        included_contributions=(),
        recognition_origins=(),
        is_current=None,
    )


class STTContributionConsumptionLedger:
    def __init__(self, *, identity_capacity: int = 4096) -> None:
        if identity_capacity < 1:
            raise ValueError("identity_capacity must be positive")
        self._identity_capacity = identity_capacity
        self._consumed: set[tuple[STTProviderTurnIdentity, str]] = set()
        self._consumed_terminal_tails: set[STTProviderTurnIdentity] = set()
        self._known_identities: set[STTProviderTurnIdentity] = set()
        self._identity_order: deque[STTProviderTurnIdentity] = deque()

    @property
    def consumed_contribution_ids(self) -> frozenset[str]:
        return frozenset(contribution_id for _identity, contribution_id in self._consumed)

    def consume(
        self,
        event: STTProviderTurnUpdate | STTProviderTurnTerminal,
    ) -> str:
        text, _origins = self.consume_with_origins(event)
        return text

    def consume_with_origins(
        self,
        event: STTProviderTurnUpdate | STTProviderTurnTerminal,
    ) -> tuple[str, tuple[RecognitionTextOrigin, ...]]:
        self._remember_identity(event.identity)
        ranges: list[tuple[int, int]] = []
        if isinstance(event, STTProviderTurnUpdate):
            contributions = (event.contribution,) if event.contribution is not None else ()
            if event.stability != "stable":
                return "", ()
            include_terminal_only = False
        else:
            contributions = event.included_contributions
            include_terminal_only = event.identity not in self._consumed_terminal_tails
            if not contributions and any(
                identity == event.identity for identity, _item in self._consumed
            ):
                return "", ()
        cursor = 0
        for contribution in contributions:
            if contribution.text_start < cursor or contribution.text_end > len(event.text):
                raise ValueError("invalid assembled contribution range")
            if include_terminal_only and cursor < contribution.text_start:
                ranges.append((cursor, contribution.text_start))
            key = (event.identity, contribution.contribution_id)
            if key not in self._consumed:
                self._consumed.add(key)
                ranges.append((contribution.text_start, contribution.text_end))
            cursor = contribution.text_end
        if include_terminal_only:
            ranges.append((cursor, len(event.text)))
            self._consumed_terminal_tails.add(event.identity)
        text = ""
        origins: tuple[RecognitionTextOrigin, ...] = ()
        for start, end in ranges:
            offset = len(text)
            text += event.text[start:end]
            origins += tuple(
                replace(
                    origin, text_start=origin.text_start + offset, text_end=origin.text_end + offset
                )
                for origin in slice_recognition_origins(event.recognition_origins, start, end)
            )
        return text, origins

    def retire(self, identity: STTProviderTurnIdentity) -> None:
        self._remember_identity(identity)
        self._consumed_terminal_tails.add(identity)

    def _remember_identity(self, identity: STTProviderTurnIdentity) -> None:
        if identity in self._known_identities:
            return
        self._known_identities.add(identity)
        self._identity_order.append(identity)
        while len(self._identity_order) > self._identity_capacity:
            expired = self._identity_order.popleft()
            self._known_identities.discard(expired)
            self._consumed_terminal_tails.discard(expired)
            self._consumed = {item for item in self._consumed if item[0] != expired}


@dataclass(frozen=True, slots=True)
class STTProviderEpochEnded:
    provider_epoch_id: str
    orderly: bool
    reason: str
    provider_turn_id: str | None = None
    failure_retryable: bool = False
    failure_diagnostics: ErrorDiagnostics | None = None


@dataclass(frozen=True, slots=True)
class STTProviderInputTerminal:
    identity: STTProviderTurnIdentity
    outcome: Literal["submitted", "failed", "expired", "cancelled"]
    channel: Literal["self", "peer"]
    failure_reason: str | None = None
    epoch_disposition: Literal["reuse", "retire"] = "reuse"
    recovery_pending: bool = False
    failure_diagnostics: ErrorDiagnostics | None = None
    independent_recognition: bool = False


@dataclass(frozen=True, slots=True)
class STTRecognitionUnit:
    identity: RecognitionUnitIdentity
    text: str
    provenance: STTNativeProvenance = STTNativeProvenance()
    final_language_runs: tuple[FinalLanguageRun, ...] = ()
    final_speaker_runs: tuple[FinalSpeakerRun, ...] = ()
    estimated_last_speech_at: float | None = None

    @property
    def retained_bytes(self) -> int:
        size = len(self.text.encode("utf-8"))
        native = self.provenance.transcription
        if native is not None:
            size += native.retained_bytes
        return size


@dataclass(frozen=True, slots=True)
class STTRecognitionUnitTerminal:
    unit: STTRecognitionUnit
    outcome: Literal["final", "empty", "failed", "cancelled"]
    failure_reason: str | None = None


STTProviderTurnEvent = (
    STTProviderTurnUpdate
    | STTProviderTurnTerminal
    | STTProviderEpochEnded
    | STTProviderInputTerminal
    | STTRecognitionUnit
    | STTRecognitionUnitTerminal
)


@runtime_checkable
class STTScopedTurnSession(Protocol):
    async def begin_turn(self, request: STTProviderTurnRequest) -> None: ...
    async def send_turn_audio(
        self,
        identity: STTProviderTurnIdentity,
        pcm16le: bytes,
        *,
        payload_sequence: int,
        source_ranges: tuple[AudioCaptureSpan, ...],
        context_only: bool,
    ) -> None: ...
    async def seal_turn(
        self,
        identity: STTProviderTurnIdentity,
        *,
        sealed_content_ranges: tuple[AudioCaptureSpan, ...],
        seal_reason: str,
        observed_trailing_silence_ms: int | None,
    ) -> None: ...
    async def abort_turn(self, identity: STTProviderTurnIdentity, *, reason: str) -> None: ...
    async def turn_events(self) -> AsyncIterator[STTProviderTurnEvent]: ...
    async def stop(self) -> None: ...
    async def close(self) -> None: ...


@runtime_checkable
class STTIndependentRecognitionSession(Protocol):
    @property
    def accepts_stream_input(self) -> bool: ...
    @property
    def independent_recognition_units(self) -> bool: ...
    async def begin_stream(self, stream: RecognitionStreamIdentity) -> None: ...
    async def send_stream_audio(
        self, pcm16le: bytes, *, source_ranges: tuple[AudioCaptureSpan, ...]
    ) -> None: ...
    async def end_stream(self, *, reason: str) -> None: ...
    def recognition_source_covers(self, ranges: tuple[AudioCaptureSpan, ...]) -> bool: ...


@dataclass(frozen=True, slots=True)
class STTBackendTranscriptEvent:
    text: str
    is_final: bool
    final_language_runs: tuple[FinalLanguageRun, ...] = ()
    final_speaker_runs: tuple[FinalSpeakerRun, ...] = ()


class RecoverableSTTSessionError(RuntimeError):
    pass


class STTBackendSession(Protocol):
    async def send_audio(self, pcm16le: bytes) -> None: ...
    async def on_speech_end(
        self,
        *,
        trailing_silence_ms: int | None = None,
        reason: SpeechBoundaryReason | None = None,
    ) -> None: ...  # Backend-specific end-of-speech handling
    async def stop(self) -> None: ...
    async def close(self) -> None: ...
    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]: ...


@runtime_checkable
class STTBackendFloat32Session(Protocol):
    async def send_audio_f32(self, samples_f32: np.ndarray) -> None: ...


class STTBackend(Protocol):
    async def open_session(
        self,
        *,
        projection: STTSessionProjection = LEGACY_STT_SESSION_PROJECTION,
    ) -> STTBackendSession: ...


@runtime_checkable
class LocalASRReconfigurableBackend(Protocol):
    async def reconfigure_session_options(self, options: LocalASRSessionOptions) -> None: ...
