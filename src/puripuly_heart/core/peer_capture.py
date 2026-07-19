from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol
from uuid import UUID


class PeerCaptureSessionState(str, Enum):
    STOPPED = "stopped"
    ADMISSION_PENDING = "admission_pending"
    TARGET_RESOLVING = "target_resolving"
    PROVIDER_PENDING = "provider_pending"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAULTED = "faulted"


class PeerCaptureAdmissionStatus(str, Enum):
    ADMITTED = "admitted"
    PENDING = "pending"
    REJECTED = "rejected"


class PeerCaptureTargetStatus(str, Enum):
    RESOLVED = "resolved"
    PENDING = "pending"
    UNAVAILABLE = "unavailable"


class PeerCaptureProviderStatus(str, Enum):
    DETACHED = "detached"
    PENDING = "pending"
    READY = "ready"
    RELEASING = "releasing"
    FAILED = "failed"


class PeerCaptureProviderMutationStatus(str, Enum):
    APPLIED = "applied"
    PENDING = "pending"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class PeerCaptureFailureReason(str, Enum):
    ADMISSION_REJECTED = "admission_rejected"
    TARGET_UNAVAILABLE = "target_unavailable"
    PROVIDER_FAILED = "provider_failed"
    SOURCE_OPEN_FAILED = "source_open_failed"
    VAD_FAILED = "vad_failed"
    SOURCE_LOST = "source_lost"
    SESSION_FAILED = "session_failed"
    CLEANUP_FAILED = "cleanup_failed"
    PROCESS_TARGET_UNAVAILABLE = "target_unavailable"
    PROCESS_SETUP_FAILED = "source_open_failed"
    PROCESS_TARGET_EXITED = "source_lost"
    PROCESS_SOURCE_FAILED = "source_lost"
    PROCESS_PROVIDER_FAILED = "provider_failed"
    PEER_RUNTIME_FAILED = "session_failed"


class PeerCaptureDiagnosticEvent(str, Enum):
    INTENT_CHANGED = "intent_changed"
    ADMISSION_CHANGED = "admission_changed"
    TARGET_CHANGED = "target_changed"
    PROVIDER_CHANGED = "provider_changed"
    SESSION_CHANGED = "session_changed"
    FAILURE = "failure"


class PeerCaptureFinalLanguageState(str, Enum):
    WHOLE_UTTERANCE = "whole_utterance"
    MIXED = "mixed"
    MISSING = "missing"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class PeerCaptureTargetIntent:
    kind: Literal["default_output_device", "named_output_device", "process"]
    device_name: str | None = None
    process_kind: Literal["generic_executable", "vrchat", "discord"] | None = None
    executable_identity: str | None = None
    discord_channel: str | None = None
    executable_basename: str | None = None


@dataclass(frozen=True, slots=True)
class PeerCaptureResolvedTarget:
    intent: PeerCaptureTargetIntent
    capture_descriptor: object | None = None


@dataclass(frozen=True, slots=True)
class PeerCaptureTargetResolution:
    status: PeerCaptureTargetStatus
    target: PeerCaptureResolvedTarget | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class PeerCaptureLanguageFacts:
    source_mode: Literal["auto", "manual"]
    source_language: str
    expected_languages: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PeerCapturedFinalFacts:
    utterance_id: UUID
    capture_sequence: int
    language: PeerCaptureLanguageFacts
    language_state: PeerCaptureFinalLanguageState
    detected_languages: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PeerCaptureSessionConfig:
    provider_id: str
    provider_signature: tuple[object, ...]
    runtime_signature: tuple[object, ...]
    capture_signature: tuple[object, ...]
    capture_target: PeerCaptureTargetIntent
    language: PeerCaptureLanguageFacts
    target_sample_rate_hz: int
    vad_speech_threshold: float
    vad_hangover_ms: int
    vad_pre_roll_ms: int
    output_device: str = ""
    model_id: str | None = None
    session_options: object | None = None
    provider_context: object | None = None
    local_provider: bool = False
    release_backend_after: float | None = None
    warmup: bool = True

    @property
    def backend(self) -> object | None:
        return self.provider_context

    @property
    def vad_threshold(self) -> float:
        return self.vad_speech_threshold

    @property
    def capture_vad_signature(self) -> tuple[object, ...]:
        return self.capture_signature


@dataclass(frozen=True, slots=True)
class PeerCaptureAdmission:
    status: PeerCaptureAdmissionStatus
    reason: str | None = None
    retain_intent: bool = False


@dataclass(frozen=True, slots=True)
class PeerCaptureProviderMutation:
    status: PeerCaptureProviderMutationStatus
    reason: str | None = None
    attachment_token: object | None = None


@dataclass(frozen=True, slots=True)
class PeerCaptureDiagnostic:
    event: PeerCaptureDiagnosticEvent
    generation: int
    state: PeerCaptureSessionState
    provider_id: str | None
    capture_kind: str | None
    reason: PeerCaptureFailureReason | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class PeerCaptureSessionSnapshot:
    state: PeerCaptureSessionState
    provider_status: PeerCaptureProviderStatus
    target_status: PeerCaptureTargetStatus | None
    desired_active: bool
    effective_active: bool
    generation: int
    provider_id: str | None
    runtime_signature: tuple[object, ...] | None
    capture_target: PeerCaptureTargetIntent | None
    resolved_target: PeerCaptureResolvedTarget | None
    language: PeerCaptureLanguageFacts | None
    failure_reason: PeerCaptureFailureReason | None
    admission_reason: str | None
    target_reason: str | None
    retry_available: bool
    has_source: bool
    has_vad: bool
    has_loop_task: bool
    cleanup_debt: int
    closed: bool


PeerCaptureTerminalFailureHandler = Callable[[Exception], Awaitable[None]]


class PeerCaptureAdmissionPort(Protocol):
    async def admit(self, config: PeerCaptureSessionConfig) -> PeerCaptureAdmission: ...


class PeerCaptureTargetResolverPort(Protocol):
    async def resolve(
        self,
        target: PeerCaptureTargetIntent,
    ) -> PeerCaptureTargetResolution: ...


class PeerCaptureProviderPort(Protocol):
    def is_ready(self, config: PeerCaptureSessionConfig) -> bool: ...

    async def replace(
        self,
        request: object,
        *,
        start: bool,
        on_terminal_failure: PeerCaptureTerminalFailureHandler,
    ) -> PeerCaptureProviderMutation: ...

    async def handoff(
        self,
        request: object,
        *,
        start: bool,
        on_terminal_failure: PeerCaptureTerminalFailureHandler,
    ) -> PeerCaptureProviderMutation: ...

    async def cancel_handoff(self) -> bool: ...

    async def start_ingress(self) -> None: ...

    async def warmup(self) -> None: ...

    async def reconfigure(self, session_options: object) -> None: ...

    async def release(
        self,
        *,
        mode: Literal["drain", "abort"],
        release_backend_after: float | None = None,
    ) -> None: ...


class PeerCaptureSourcePort(Protocol):
    async def close(self) -> None: ...


__all__ = [
    "PeerCapturedFinalFacts",
    "PeerCaptureAdmission",
    "PeerCaptureAdmissionPort",
    "PeerCaptureAdmissionStatus",
    "PeerCaptureDiagnostic",
    "PeerCaptureDiagnosticEvent",
    "PeerCaptureFailureReason",
    "PeerCaptureFinalLanguageState",
    "PeerCaptureLanguageFacts",
    "PeerCaptureProviderMutation",
    "PeerCaptureProviderMutationStatus",
    "PeerCaptureProviderPort",
    "PeerCaptureProviderStatus",
    "PeerCaptureResolvedTarget",
    "PeerCaptureSessionConfig",
    "PeerCaptureSessionSnapshot",
    "PeerCaptureSessionState",
    "PeerCaptureSourcePort",
    "PeerCaptureTargetIntent",
    "PeerCaptureTargetResolution",
    "PeerCaptureTargetResolverPort",
    "PeerCaptureTargetStatus",
    "PeerCaptureTerminalFailureHandler",
]
