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


OscControlPresentationName = Literal[
    "PuriPuly_Talk",
    "PuriPuly_Listen",
    "PuriPuly_Trans",
    "PuriPuly_Captions",
    "PuriPuly_PeerAuto",
    "PuriPuly_MuteSync",
    "PuriPuly_ChatboxSource",
    "PuriPuly_SelfSrcLang",
    "PuriPuly_SelfDstLang",
    "PuriPuly_PeerSrcLang",
    "PuriPuly_PeerDstLang",
    "PuriPuly_SelfASR",
    "PuriPuly_PeerASR",
    "PuriPuly_Translator",
    "PuriPuly_Fallback",
]


@dataclass(frozen=True, slots=True)
class OscControlPresentationState:
    changed_control: OscControlPresentationName
    self_capture: bool
    peer_capture: bool
    translation: bool
    captions: bool
    peer_source_mode: str
    mute_sync: bool
    chatbox_source: bool
    self_source_language: str
    self_target_language: str
    peer_source_language: str
    peer_target_language: str
    self_asr: str
    peer_asr: str
    self_asr_setting: str
    peer_asr_setting: str
    custom_stt_mode: str
    custom_stt_compatibility: str
    llm_provider: str
    translation_model: str
    translation_connection: str
    translation_connection_history: tuple[tuple[str, str], ...]
    translation_http_extension_id: str | None
    translation_previous_model: str | None
    fallback: str
    fallback_enabled: bool
    fallback_model: str
    fallback_connection: str


__all__ = [
    "GpuDashboardNotice",
    "GpuDeviceOption",
    "GpuNoticeAction",
    "ManagedGemmaDashboardNotice",
    "ManagedGemmaNoticeAction",
    "OptionItem",
    "OscControlPresentationName",
    "OscControlPresentationState",
    "OverlayPeerPresentationState",
]
