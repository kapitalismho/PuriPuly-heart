from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GpuNoticeAction = Literal["install", "repair", "reinstall", "rediscover", "restart"]
ManagedGemmaNoticeAction = Literal["cancel"]


@dataclass
class OptionItem:
    value: str
    label: str
    description: str = ""
    disabled: bool = False
    section: str = ""


@dataclass(frozen=True, slots=True)
class GpuDeviceOption:
    device_id: str
    display_name: str
    backend_name: str


@dataclass(frozen=True, slots=True)
class GpuDashboardNotice:
    status: str
    progress_percent: int | None = None
    action: GpuNoticeAction | None = None


@dataclass(frozen=True, slots=True)
class ManagedGemmaDashboardNotice:
    status: str
    backend: str | None = None
    progress_percent: int | None = None
    action: ManagedGemmaNoticeAction | None = None


@dataclass(frozen=True, slots=True)
class OverlayPeerPresentationState:
    overlay_intent_enabled: bool
    overlay_state: str
    overlay_failure_reason: str | None
    peer_intent_enabled: bool
    peer_effective_enabled: bool
    peer_warning_reason: str | None
    peer_activation_starting: bool


__all__ = [
    "GpuDashboardNotice",
    "GpuDeviceOption",
    "GpuNoticeAction",
    "ManagedGemmaDashboardNotice",
    "ManagedGemmaNoticeAction",
    "OptionItem",
    "OverlayPeerPresentationState",
]
