from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

GpuWorkerMode = Literal["discovery", "persistent"]


@dataclass(frozen=True, slots=True)
class GpuWorkerDevice:
    device_id: str
    registry_index: int
    name: str
    description: str
    device_type: str
    memory_total_bytes: int
    memory_free_bytes: int


@dataclass(frozen=True, slots=True)
class GpuWorkerActivation:
    device: GpuWorkerDevice
    model_load_seconds: float
    warmup_seconds: float


@dataclass(frozen=True, slots=True)
class GpuWorkerTranscription:
    text: str
    detected_language: str | None
    audio_seconds: float
    decode_seconds: float
    rtf: float


@dataclass(frozen=True, slots=True)
class GpuWorkerEvent:
    name: str
    request_id: str | None
    fields: Mapping[str, object]


class GpuWorkerError(RuntimeError):
    pass


class GpuWorkerRequestError(GpuWorkerError):
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


class GpuWorkerClosedError(GpuWorkerError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "worker_closed",
        exit_code: int | None = None,
        failure_type: str | None = None,
    ) -> None:
        self.code = code
        self.exit_code = exit_code
        self.failure_type = failure_type
        super().__init__(message)


class GpuWorkerClientPort(Protocol):
    @property
    def pid(self) -> int | None: ...

    @property
    def is_closed(self) -> bool: ...

    async def discover(self) -> tuple[GpuWorkerDevice, ...]: ...

    async def activate(self, *, model_path: Path, device_id: str) -> GpuWorkerActivation: ...

    async def transcribe(
        self,
        *,
        request_id: str,
        channel: Literal["self", "peer"],
        audio_path: Path,
        language_hint: str | None = None,
        on_request_sent: Callable[[], None] | None = None,
    ) -> GpuWorkerTranscription: ...

    async def cancel(self, target_request_id: str) -> None: ...

    async def next_event(self) -> GpuWorkerEvent: ...

    async def close(self) -> None: ...

    async def force_close(self) -> None: ...


class GpuWorkerProcessFactoryPort(Protocol):
    async def start(self, *, mode: GpuWorkerMode) -> GpuWorkerClientPort: ...


__all__ = [
    "GpuWorkerActivation",
    "GpuWorkerClientPort",
    "GpuWorkerClosedError",
    "GpuWorkerDevice",
    "GpuWorkerError",
    "GpuWorkerEvent",
    "GpuWorkerMode",
    "GpuWorkerProcessFactoryPort",
    "GpuWorkerRequestError",
    "GpuWorkerTranscription",
]
