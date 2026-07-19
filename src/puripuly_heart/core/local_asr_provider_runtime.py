from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from puripuly_heart.config.resolved import ResolvedSTTConfig
from puripuly_heart.core.gpu_worker import (
    GpuWorkerActivation,
    GpuWorkerDevice,
    GpuWorkerTranscription,
)
from puripuly_heart.core.runtime.local_asr_transition import LocalASRSessionOptions

ProviderRuntimeChannel = Literal["self", "peer"]
ProviderRuntimeChannelPhase = Literal[
    "inactive",
    "building",
    "dormant",
    "ready",
    "running",
    "failed",
    "closed",
]
ProviderRuntimeGpuPhase = Literal[
    "inactive",
    "idle",
    "discovering",
    "discovery_pending",
    "unsupported",
    "not_installed",
    "invalid",
    "available",
    "validating",
    "loading",
    "warming",
    "ready",
    "failed",
    "closed",
]
ProviderRuntimeMutationStatus = Literal["applied", "failed", "cancelled"]
ProviderRuntimeReleaseMode = Literal["drain", "dormant", "abort"]

ProviderRuntimeEventHandler = Callable[[object], Awaitable[None]]
ProviderRuntimeExceptionHandler = Callable[[Exception], Awaitable[None] | None]
ProviderRuntimeTerminalFailureSink = Callable[[Exception], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class LocalASRProviderRuntimeCallbacks:
    self_event_handler: ProviderRuntimeEventHandler
    peer_event_handler: ProviderRuntimeEventHandler
    retired_event_handler: ProviderRuntimeEventHandler
    self_exception_handler: ProviderRuntimeExceptionHandler
    peer_exception_handler: ProviderRuntimeExceptionHandler


@dataclass(frozen=True, slots=True)
class ProviderRuntimeBuildRequest:
    config: ResolvedSTTConfig
    gpu_device_id: str = "auto"
    warmup: bool = False
    model_id: str | None = None
    session_options: LocalASRSessionOptions | None = None

    def __post_init__(self) -> None:
        if self.config.channel not in {"self", "peer"}:
            raise ValueError("provider runtime channel must be self or peer")
        if not self.gpu_device_id.strip():
            raise ValueError("gpu_device_id must be non-empty")

    @property
    def channel(self) -> ProviderRuntimeChannel:
        return self.config.channel

    @property
    def provider_id(self) -> str:
        return self.config.provider


@dataclass(frozen=True, slots=True)
class ProviderRuntimeChannelSnapshot:
    channel: ProviderRuntimeChannel
    provider_id: str | None
    model_id: str | None
    phase: ProviderRuntimeChannelPhase
    generation: int
    pending_handoff: bool
    has_resources: bool


@dataclass(frozen=True, slots=True)
class ProviderRuntimeGpuSnapshot:
    phase: ProviderRuntimeGpuPhase
    devices: tuple[GpuWorkerDevice, ...]
    active_channels: frozenset[ProviderRuntimeChannel]
    pending_count: int
    worker_pid: int | None
    configured_device_id: str | None
    model_resident: bool
    retry_required: bool
    failure_code: str | None


@dataclass(frozen=True, slots=True)
class LocalASRProviderRuntimeSnapshot:
    channels: tuple[ProviderRuntimeChannelSnapshot, ...]
    gpu: ProviderRuntimeGpuSnapshot
    revision: int = 0
    closed: bool = False

    def channel_for(self, channel: ProviderRuntimeChannel) -> ProviderRuntimeChannelSnapshot:
        return next(state for state in self.channels if state.channel == channel)


@dataclass(frozen=True, slots=True)
class ProviderRuntimeDiagnostic:
    event: str
    outcome: str | None = None
    channel: ProviderRuntimeChannel | None = None
    provider_id: str | None = None
    phase: str | None = None
    model_id: str | None = None
    device_id: str | None = None
    failure_code: str | None = None
    failure_type: str | None = None
    progress_percent: int | None = None
    model_load_seconds: float | None = None
    warmup_seconds: float | None = None
    audio_seconds: float | None = None
    decode_seconds: float | None = None
    rtf: float | None = None
    queue_wait_seconds: float | None = None
    worker_exit_code: int | None = None


@dataclass(frozen=True, slots=True)
class ProviderRuntimeMutationResult:
    status: ProviderRuntimeMutationStatus
    request: ProviderRuntimeBuildRequest
    previous_provider_id: str | None
    snapshot: LocalASRProviderRuntimeSnapshot
    failure_type: str | None = None


class ProviderGpuRuntimePort(Protocol):
    @property
    def state(self) -> object: ...

    @property
    def discovery_state(self) -> object: ...

    @property
    def active_channels(self) -> frozenset[ProviderRuntimeChannel]: ...

    @property
    def pending_count(self) -> int: ...

    @property
    def worker_pid(self) -> int | None: ...

    @property
    def last_failure_code(self) -> str | None: ...

    @property
    def configured_device_id(self) -> str | None: ...

    async def discover_devices(self) -> tuple[GpuWorkerDevice, ...]: ...

    async def activate_channel(
        self,
        channel: ProviderRuntimeChannel,
        *,
        model_path: Path,
        model_id: str,
        device_id: str,
    ) -> GpuWorkerActivation: ...

    async def retry(self) -> GpuWorkerActivation: ...

    async def submit(
        self,
        channel: ProviderRuntimeChannel,
        samples_f32: np.ndarray,
        *,
        speech_end_at: float,
        language_hint: str | None = None,
    ) -> GpuWorkerTranscription: ...

    async def deactivate_channel(self, channel: ProviderRuntimeChannel) -> None: ...

    async def close(self) -> None: ...


class ProviderRuntimeProviderFactoryPort(Protocol):
    async def create(
        self,
        request: ProviderRuntimeBuildRequest,
        *,
        gpu_runtime: ProviderGpuRuntimePort,
        on_terminal_failure: ProviderRuntimeTerminalFailureSink | None = None,
    ) -> object: ...


class LocalASRProviderRuntimePort(Protocol):
    @property
    def snapshot(self) -> LocalASRProviderRuntimeSnapshot: ...

    @property
    def diagnostics(self) -> tuple[ProviderRuntimeDiagnostic, ...]: ...

    async def start(self) -> None: ...

    async def discover_gpu(self, *, force: bool = False) -> LocalASRProviderRuntimeSnapshot: ...

    async def inspect_gpu_readiness(
        self,
        *,
        explicit_intent: bool,
        device_id: str,
    ) -> LocalASRProviderRuntimeSnapshot: ...

    async def replace_provider(
        self,
        request: ProviderRuntimeBuildRequest,
        *,
        start: bool,
        on_terminal_failure: ProviderRuntimeTerminalFailureSink | None = None,
    ) -> ProviderRuntimeMutationResult: ...

    async def handoff_provider(
        self,
        request: ProviderRuntimeBuildRequest,
        *,
        start: bool,
        on_terminal_failure: ProviderRuntimeTerminalFailureSink | None = None,
    ) -> ProviderRuntimeMutationResult: ...

    async def commit_handoff(self, channel: ProviderRuntimeChannel) -> None: ...

    async def cancel_handoff(self, channel: ProviderRuntimeChannel) -> bool: ...

    async def release_channel(
        self,
        channel: ProviderRuntimeChannel,
        *,
        mode: ProviderRuntimeReleaseMode,
        release_backend_after: float | None = None,
    ) -> None: ...

    async def start_channel(self, channel: ProviderRuntimeChannel) -> None: ...

    async def warmup_channel(self, channel: ProviderRuntimeChannel) -> None: ...

    async def reconfigure_channel(
        self,
        channel: ProviderRuntimeChannel,
        options: LocalASRSessionOptions,
    ) -> None: ...

    async def handle_vad_event(
        self,
        channel: ProviderRuntimeChannel,
        event: object,
    ) -> None: ...

    async def retry_gpu(
        self,
        channels: tuple[ProviderRuntimeChannel, ...],
    ) -> LocalASRProviderRuntimeSnapshot: ...

    async def close(self) -> None: ...


class LocalASRProviderRuntimeFactoryPort(Protocol):
    def create(
        self,
        callbacks: LocalASRProviderRuntimeCallbacks,
    ) -> LocalASRProviderRuntimePort: ...


__all__ = [
    "LocalASRProviderRuntimeCallbacks",
    "LocalASRProviderRuntimeFactoryPort",
    "LocalASRProviderRuntimePort",
    "LocalASRProviderRuntimeSnapshot",
    "ProviderGpuRuntimePort",
    "ProviderRuntimeBuildRequest",
    "ProviderRuntimeChannel",
    "ProviderRuntimeChannelPhase",
    "ProviderRuntimeChannelSnapshot",
    "ProviderRuntimeDiagnostic",
    "ProviderRuntimeEventHandler",
    "ProviderRuntimeExceptionHandler",
    "ProviderRuntimeGpuPhase",
    "ProviderRuntimeGpuSnapshot",
    "ProviderRuntimeMutationResult",
    "ProviderRuntimeMutationStatus",
    "ProviderRuntimeProviderFactoryPort",
    "ProviderRuntimeReleaseMode",
    "ProviderRuntimeTerminalFailureSink",
]
