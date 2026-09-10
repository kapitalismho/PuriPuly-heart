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
    native_request_id: str | None = None


@dataclass(frozen=True, slots=True)
class STTProviderTurnRequest:
    identity: STTProviderTurnIdentity
    settings: AudioSegmentSettingsSnapshot


@dataclass(frozen=True, slots=True)
class STTProviderTurnTerminal:
    identity: STTProviderTurnIdentity
    outcome: SegmentTerminalOutcome
    text: str = ""
    final_language_runs: tuple[FinalLanguageRun, ...] = ()
    text_authority: Literal["authoritative", "degraded", "none"] = "none"


@runtime_checkable
class STTScopedTurnSession(Protocol):
    async def begin_turn(self, request: STTProviderTurnRequest) -> None: ...
    async def send_turn_audio(
        self,
        identity: STTProviderTurnIdentity,
        pcm16le: bytes,
        *,
        source_ranges: tuple[AudioCaptureSpan, ...],
        context_only: bool,
    ) -> None: ...
    async def seal_turn(
        self,
        identity: STTProviderTurnIdentity,
        *,
        sealed_content_ranges: tuple[AudioCaptureSpan, ...],
    ) -> None: ...
    async def abort_turn(self, identity: STTProviderTurnIdentity) -> None: ...
    async def turn_terminals(self) -> AsyncIterator[STTProviderTurnTerminal]: ...


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
