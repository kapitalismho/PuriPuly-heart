from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol


class SelfCaptureSessionState(str, Enum):
    STOPPED = "stopped"
    ADMISSION_PENDING = "admission_pending"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAULTED = "faulted"


class SelfCaptureAdmissionStatus(str, Enum):
    ADMITTED = "admitted"
    PENDING = "pending"
    REJECTED = "rejected"


class SelfCaptureProviderStatus(str, Enum):
    DETACHED = "detached"
    PENDING = "pending"
    READY = "ready"
    RELEASING = "releasing"
    FAILED = "failed"


class SelfCaptureProviderMutationStatus(str, Enum):
    APPLIED = "applied"
    PENDING = "pending"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class SelfCaptureFailureReason(str, Enum):
    ADMISSION_REJECTED = "admission_rejected"
    PROVIDER_FAILED = "provider_failed"
    SOURCE_OPEN_FAILED = "source_open_failed"
    VAD_FAILED = "vad_failed"
    SESSION_FAILED = "session_failed"
    CLEANUP_FAILED = "cleanup_failed"


class SelfCaptureDiagnosticEvent(str, Enum):
    INTENT_CHANGED = "intent_changed"
    ADMISSION_CHANGED = "admission_changed"
    PROVIDER_CHANGED = "provider_changed"
    SESSION_CHANGED = "session_changed"
    FAILURE = "failure"


@dataclass(frozen=True, slots=True)
class SelfCaptureSessionConfig:
    provider_id: str
    provider_signature: tuple[object, ...]
    runtime_signature: tuple[object, ...]
    capture_signature: tuple[object, ...]
    target_sample_rate_hz: int
    session_options: object | None = None
    local_cpu: bool = False
    local_gpu: bool = False
    release_backend_after: float | None = None
    warmup: bool = True


@dataclass(frozen=True, slots=True)
class SelfCaptureAdmission:
    status: SelfCaptureAdmissionStatus
    reason: str | None = None
    retain_intent: bool = False


@dataclass(frozen=True, slots=True)
class SelfCaptureProviderMutation:
    status: SelfCaptureProviderMutationStatus
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SelfCaptureDiagnostic:
    event: SelfCaptureDiagnosticEvent
    generation: int
    state: SelfCaptureSessionState
    provider_id: str | None
    reason: SelfCaptureFailureReason | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class SelfCaptureSessionSnapshot:
    state: SelfCaptureSessionState
    provider_status: SelfCaptureProviderStatus
    desired_active: bool
    effective_active: bool
    generation: int
    provider_id: str | None
    runtime_signature: tuple[object, ...] | None
    failure_reason: SelfCaptureFailureReason | None
    admission_reason: str | None
    has_source: bool
    has_vad: bool
    has_loop_task: bool
    cleanup_debt: int
    closed: bool


SelfCaptureTerminalFailureHandler = Callable[[Exception], Awaitable[None]]


class SelfCaptureAdmissionPort(Protocol):
    async def admit(self, config: SelfCaptureSessionConfig) -> SelfCaptureAdmission: ...


class SelfCaptureProviderPort(Protocol):
    def is_ready(self, config: SelfCaptureSessionConfig) -> bool: ...

    async def replace(
        self,
        request: object,
        *,
        start: bool,
        on_terminal_failure: SelfCaptureTerminalFailureHandler,
    ) -> SelfCaptureProviderMutation: ...

    async def handoff(
        self,
        request: object,
        *,
        start: bool,
        on_terminal_failure: SelfCaptureTerminalFailureHandler,
    ) -> SelfCaptureProviderMutation: ...

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


class SelfCaptureSourcePort(Protocol):
    async def close(self) -> None: ...


__all__ = [
    "SelfCaptureAdmission",
    "SelfCaptureAdmissionPort",
    "SelfCaptureAdmissionStatus",
    "SelfCaptureDiagnostic",
    "SelfCaptureDiagnosticEvent",
    "SelfCaptureFailureReason",
    "SelfCaptureProviderMutation",
    "SelfCaptureProviderMutationStatus",
    "SelfCaptureProviderPort",
    "SelfCaptureProviderStatus",
    "SelfCaptureSessionConfig",
    "SelfCaptureSessionSnapshot",
    "SelfCaptureSessionState",
    "SelfCaptureSourcePort",
    "SelfCaptureTerminalFailureHandler",
]
