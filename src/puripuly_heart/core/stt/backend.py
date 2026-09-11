from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Literal, Protocol, runtime_checkable

import numpy as np

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import (
    AudioSegmentIdentity,
    AudioSegmentSettingsSnapshot,
    SegmentTerminalOutcome,
)
from puripuly_heart.core.runtime.local_asr_transition import LocalASRSessionOptions
from puripuly_heart.core.speech_boundary import SpeechBoundaryReason
from puripuly_heart.domain.models import FinalLanguageRun


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
    provenance: STTNativeProvenance = STTNativeProvenance()
    contribution: STTTextContribution | None = None


@dataclass(frozen=True, slots=True)
class STTTimedToken:
    text: str
    language: str = ""
    start_ms: int | None = None
    end_ms: int | None = None
    timing: Literal["interval", "end_only", "unmapped", "invalid"] = "unmapped"
    source_start_sample: int | None = None
    source_end_sample: int | None = None
    provenance: STTNativeProvenance = STTNativeProvenance()

    def __post_init__(self) -> None:
        if self.timing not in {"interval", "end_only", "unmapped", "invalid"}:
            raise ValueError(f"unknown timed token timing: {self.timing!r}")
        if (
            self.timing == "interval"
            and self.start_ms is not None
            and self.end_ms is not None
            and self.start_ms > self.end_ms
        ):
            raise ValueError("timed token start follows end")


@dataclass(frozen=True, slots=True)
class STTProviderTurnTerminal:
    identity: STTProviderTurnIdentity
    outcome: SegmentTerminalOutcome
    text: str = ""
    final_language_runs: tuple[FinalLanguageRun, ...] = ()
    text_authority: Literal["authoritative", "degraded", "none"] = "none"
    failure_reason: str | None = None
    epoch_disposition: Literal["reuse", "retire"] = "reuse"
    provenance: tuple[STTNativeProvenance, ...] = ()
    timed_tokens: tuple[STTTimedToken, ...] = ()
    included_contributions: tuple[STTTextContribution, ...] = ()


class STTContributionConsumptionLedger:
    def __init__(self) -> None:
        self._consumed: set[tuple[STTProviderTurnIdentity, str]] = set()

    @property
    def consumed_contribution_ids(self) -> frozenset[str]:
        return frozenset(contribution_id for _identity, contribution_id in self._consumed)

    def consume(
        self,
        event: STTProviderTurnUpdate | STTProviderTurnTerminal,
    ) -> str:
        if isinstance(event, STTProviderTurnUpdate):
            contribution = event.contribution
            if event.stability != "stable" or contribution is None:
                return ""
            return self._consume_contribution(event.identity, event.text, contribution)
        pieces = [
            self._consume_contribution(event.identity, event.text, contribution)
            for contribution in event.included_contributions
        ]
        if event.included_contributions:
            return "".join(pieces)
        request_consumed = any(identity == event.identity for identity, _item in self._consumed)
        return event.text if not request_consumed else ""

    def _consume_contribution(
        self,
        identity: STTProviderTurnIdentity,
        text: str,
        contribution: STTTextContribution,
    ) -> str:
        key = (identity, contribution.contribution_id)
        if key in self._consumed:
            return ""
        if contribution.text_end > len(text):
            raise ValueError("contribution range exceeds assembled text")
        self._consumed.add(key)
        return text[contribution.text_start : contribution.text_end]


@dataclass(frozen=True, slots=True)
class STTProviderEpochEnded:
    provider_epoch_id: str
    orderly: bool
    reason: str
    provider_turn_id: str | None = None


STTProviderTurnEvent = STTProviderTurnUpdate | STTProviderTurnTerminal | STTProviderEpochEnded


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


@dataclass(frozen=True, slots=True)
class STTBackendTranscriptEvent:
    text: str
    is_final: bool
    final_language_runs: tuple[FinalLanguageRun, ...] = ()


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
