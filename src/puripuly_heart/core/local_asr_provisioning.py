from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal, Protocol

LocalASRProvisioningBackend = Literal["cpu", "gpu"]
LocalASRIntegrityStatus = Literal["ready", "missing", "invalid", "not_requested"]
LocalASROperationStatus = Literal[
    "idle",
    "downloading",
    "download_failed",
    "cancelled",
]
LocalASRProvisioningStatus = Literal[
    "ready",
    "missing",
    "invalid",
    "not_requested",
    "downloading",
    "download_failed",
    "cancelled",
]


@dataclass(frozen=True, slots=True)
class LocalASRModelProvisioningState:
    model_id: str
    backend: LocalASRProvisioningBackend
    integrity: LocalASRIntegrityStatus
    operation: LocalASROperationStatus = "idle"

    @property
    def status(self) -> LocalASRProvisioningStatus:
        if self.operation != "idle":
            return self.operation
        return self.integrity

    @property
    def available(self) -> bool:
        return self.integrity == "ready"


@dataclass(frozen=True, slots=True)
class LocalASRProvisioningActivity:
    backend: LocalASRProvisioningBackend
    model_id: str
    origin: str
    progress_percent: int | None
    generation: int


@dataclass(frozen=True, slots=True)
class LocalASRProvisioningSnapshot:
    models: tuple[LocalASRModelProvisioningState, ...]
    required_cpu_model_ids: tuple[str, ...]
    gpu_model_id: str
    activities: tuple[LocalASRProvisioningActivity, ...] = ()
    revision: int = 0
    closed: bool = False

    @property
    def cpu_auto_available(self) -> bool:
        required = frozenset(self.required_cpu_model_ids)
        cpu_models = tuple(model for model in self.models if model.model_id in required)
        return (
            len(cpu_models) == len(required)
            and frozenset(model.model_id for model in cpu_models) == required
            and all(model.available for model in cpu_models)
        )

    def state_for(self, model_id: str) -> LocalASRModelProvisioningState:
        matches = tuple(model for model in self.models if model.model_id == model_id)
        if len(matches) != 1:
            raise KeyError(model_id)
        return matches[0]

    def status_for(self, model_ids: tuple[str, ...]) -> LocalASRProvisioningStatus:
        if not model_ids:
            raise ValueError("model_ids must not be empty")
        states = tuple(self.state_for(model_id) for model_id in model_ids)
        for status in ("downloading", "download_failed", "cancelled", "invalid", "missing"):
            if any(state.status == status for state in states):
                return status
        if any(state.status == "not_requested" for state in states):
            return "not_requested"
        return "ready"

    def unavailable_model_ids(self, model_ids: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(model_id for model_id in model_ids if not self.state_for(model_id).available)

    def activity_for(
        self,
        backend: LocalASRProvisioningBackend,
    ) -> LocalASRProvisioningActivity | None:
        return next((activity for activity in self.activities if activity.backend == backend), None)


@dataclass(frozen=True, slots=True)
class LocalASRInstallRequest:
    backend: LocalASRProvisioningBackend
    model_ids: tuple[str, ...]
    locale: str | None
    origin: str
    explicit_gpu_intent: bool = False


@dataclass(frozen=True, slots=True)
class LocalASRInstallResult:
    request: LocalASRInstallRequest
    installed_model_ids: tuple[str, ...]
    failed_model_ids: tuple[str, ...]
    cancelled: bool
    snapshot: LocalASRProvisioningSnapshot


@dataclass(frozen=True, slots=True)
class LocalASRProvisioningDiagnostic:
    event: str
    backend: LocalASRProvisioningBackend | None = None
    model_id: str | None = None
    origin: str | None = None
    outcome: str | None = None
    elapsed_seconds: float | None = None
    failure_type: str | None = None


class LocalASRProvisioningPort(Protocol):
    @property
    def snapshot(self) -> LocalASRProvisioningSnapshot: ...

    @property
    def diagnostics(self) -> tuple[LocalASRProvisioningDiagnostic, ...]: ...

    async def inspect_cpu(
        self,
        model_ids: tuple[str, ...] | None = None,
        *,
        verify_checksums: bool = False,
    ) -> LocalASRProvisioningSnapshot: ...

    async def inspect_gpu(
        self,
        *,
        explicit_intent: bool,
        verify_checksums: bool = False,
    ) -> LocalASRProvisioningSnapshot: ...

    def start_install(
        self,
        request: LocalASRInstallRequest,
    ) -> asyncio.Task[LocalASRInstallResult]: ...

    async def report_model_validation_failure(
        self,
        model_id: str,
        *,
        failure_type: str,
    ) -> LocalASRProvisioningSnapshot: ...

    async def cancel_install(self, backend: LocalASRProvisioningBackend) -> None: ...

    async def close(self) -> None: ...


__all__ = [
    "LocalASRInstallRequest",
    "LocalASRInstallResult",
    "LocalASRIntegrityStatus",
    "LocalASRModelProvisioningState",
    "LocalASROperationStatus",
    "LocalASRProvisioningBackend",
    "LocalASRProvisioningActivity",
    "LocalASRProvisioningDiagnostic",
    "LocalASRProvisioningPort",
    "LocalASRProvisioningSnapshot",
    "LocalASRProvisioningStatus",
]
