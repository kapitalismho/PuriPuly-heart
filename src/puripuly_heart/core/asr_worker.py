from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from puripuly_heart.core.messages import ErrorDiagnostics

ASRWorkerMode = Literal["discovery", "persistent"]


@dataclass(frozen=True, slots=True)
class ASRWorkerDevice:
    device_id: str
    registry_index: int
    name: str
    description: str
    device_type: str
    memory_total_bytes: int
    memory_free_bytes: int


@dataclass(frozen=True, slots=True)
class ASRWorkerActivation:
    device: ASRWorkerDevice
    model_load_seconds: float
    warmup_seconds: float


@dataclass(frozen=True, slots=True)
class ASRWorkerTranscription:
    text: str
    detected_language: str | None
    audio_seconds: float
    decode_seconds: float
    rtf: float


@dataclass(frozen=True, slots=True)
class ASRWorkerEvent:
    name: str
    request_id: str | None
    fields: Mapping[str, object]


class ASRWorkerError(RuntimeError):
    pass


class ASRWorkerRequestError(ASRWorkerError):
    def __init__(
        self,
        code: str,
        fields: Mapping[str, object] | None = None,
        *,
        attempt_started: bool = False,
    ) -> None:
        self.code = code
        self.fields = fields or {}
        self.attempt_started = attempt_started
        super().__init__(code)


class ASRWorkerClosedError(ASRWorkerError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "worker_closed",
        exit_code: int | None = None,
        failure_type: str | None = None,
        cleanup_failures: tuple[Mapping[str, object], ...] = (),
        cleanup_complete: bool | None = None,
        failure_diagnostics: ErrorDiagnostics | None = None,
    ) -> None:
        self.code = code
        self.exit_code = exit_code
        self.failure_type = failure_type
        self.cleanup_failures = cleanup_failures
        self.cleanup_complete = cleanup_complete
        self.failure_diagnostics = failure_diagnostics
        super().__init__(message)


class ASRWorkerClientPort(Protocol):
    @property
    def pid(self) -> int | None: ...

    @property
    def is_closed(self) -> bool: ...

    async def discover(self) -> tuple[ASRWorkerDevice, ...]: ...

    async def activate(
        self,
        *,
        model_path: Path,
        device_id: str,
        backend: Literal["cpu", "vulkan"] = "vulkan",
        channel: Literal["self", "peer"] | None = None,
        n_threads: int = 2,
    ) -> ASRWorkerActivation: ...

    async def transcribe(
        self,
        *,
        request_id: str,
        channel: Literal["self", "peer"],
        audio_path: Path,
        language_hint: str | None = None,
        on_request_sent: Callable[[], None] | None = None,
    ) -> ASRWorkerTranscription: ...

    async def cancel(self, target_request_id: str) -> None: ...

    async def next_event(self) -> ASRWorkerEvent: ...

    async def close(self) -> None: ...

    async def force_close(self) -> None: ...


class ASRWorkerProcessFactoryPort(Protocol):
    async def start(
        self,
        *,
        mode: ASRWorkerMode,
        operation_id: str | None = None,
        request_id: str | None = None,
    ) -> ASRWorkerClientPort: ...
    async def close_pending_startups(self) -> None: ...

    def startup_cleanup_snapshot(self) -> Mapping[str, object]: ...


__all__ = [
    "ASRWorkerActivation",
    "ASRWorkerClientPort",
    "ASRWorkerClosedError",
    "ASRWorkerDevice",
    "ASRWorkerError",
    "ASRWorkerEvent",
    "ASRWorkerMode",
    "ASRWorkerProcessFactoryPort",
    "ASRWorkerRequestError",
    "ASRWorkerTranscription",
]
