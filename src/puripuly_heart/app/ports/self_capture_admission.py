from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum

from puripuly_heart.core.self_capture import SelfCaptureSessionConfig


class SelfCaptureAdmissionEffectType(str, Enum):
    RETAIN_GPU_PENDING_INTENT = "retain_gpu_pending_intent"
    REJECT_UNSUPPORTED_LANGUAGE = "reject_unsupported_language"
    RETAIN_DOWNLOAD_PENDING_INTENT = "retain_download_pending_intent"
    REQUEST_LOCAL_REPAIR = "request_local_repair"


@dataclass(frozen=True, slots=True)
class SelfCaptureAdmissionState:
    settings_available: bool
    runtime_available: bool
    gpu_status: str | None
    local_cpu_supported: bool
    local_runtime_status: str
    activation_generation: int


@dataclass(frozen=True, slots=True)
class SelfCaptureAdmissionEffect:
    type: SelfCaptureAdmissionEffectType
    status: str | None = None
    activation_generation: int | None = None


SelfCaptureAdmissionStateProvider = Callable[
    [SelfCaptureSessionConfig],
    SelfCaptureAdmissionState,
]
SelfCaptureGpuActivationValidator = Callable[[], Awaitable[bool]]
SelfCaptureAdmissionEffectSink = Callable[[SelfCaptureAdmissionEffect], object]


__all__ = [
    "SelfCaptureAdmissionEffect",
    "SelfCaptureAdmissionEffectSink",
    "SelfCaptureAdmissionEffectType",
    "SelfCaptureAdmissionState",
    "SelfCaptureAdmissionStateProvider",
    "SelfCaptureGpuActivationValidator",
]
