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
class STTProviderTurnIdentity:
    segment: AudioSegmentIdentity
    provider_epoch_id: str
    provider_turn_id: str


@dataclass(frozen=True, slots=True)
class STTProviderTurnRequest:
    identity: STTProviderTurnIdentity
    settings: AudioSegmentSettingsSnapshot


@dataclass(frozen=True, slots=True)
class STTNativeProvenance:
    native_event_id: str | None = None
    native_request_id: str | None = None
    native_item_id: str | None = None
    native_task_id: str | None = None
    barrier: str | None = None
    from_finalize: bool | None = None


@dataclass(frozen=True, slots=True)
class STTProviderTurnUpdate:
    identity: STTProviderTurnIdentity
    sequence: int
    stability: Literal["provisional", "stable"]
    assembly: Literal["append", "replace"]
    text: str
    final_language_runs: tuple[FinalLanguageRun, ...] = ()
    provenance: STTNativeProvenance = STTNativeProvenance()

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
    async def open_session(self) -> STTBackendSession: ...


@runtime_checkable
class LocalASRReconfigurableBackend(Protocol):
    async def reconfigure_session_options(self, options: LocalASRSessionOptions) -> None: ...
