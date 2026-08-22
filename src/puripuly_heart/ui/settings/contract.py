from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias

import flet as ft

from puripuly_heart.ui.gpu_device import GpuDeviceOption

SettingsSnapshot: TypeAlias = object


@dataclass(frozen=True, slots=True)
class SettingsSurfaceIntents:
    settings_changed: Callable[[SettingsSnapshot], None]
    show_snackbar: Callable[[str, str], None]
    runtime_log_basic: Callable[..., None] | None = None
    runtime_log_detailed: Callable[..., None] | None = None


@dataclass(frozen=True, slots=True)
class SettingsProviderIntents:
    providers_changed: Callable[[], None]
    request_openrouter_pkce: Callable[[SettingsSnapshot], None]
    verify_api_key: Callable[[str, str], object]
    provider_secret_change: Callable[[str, str], object]
    secret_cleared: Callable[[str], None]
    local_llm_secret_changed: Callable[[], None]
    gpu_discovery_requested: Callable[[], object]
    custom_stt_secret_changed: Callable[[], None] | None = None


@dataclass(frozen=True, slots=True)
class SettingsGeneralIntents:
    start_microphone_test: Callable[[], None]
    telemetry_consent_change: Callable[[str], None]
    list_loopback_capture_options: Callable[[], object]
    list_loopback_process_options: Callable[[], object]
    list_loopback_device_options: Callable[[], object]
    current_loopback_capture_option: Callable[[], str]
    apply_loopback_capture_option: Callable[[str], None]
    loopback_capture_summary: Callable[[], str]
    osc_effective_ports: Callable[[], tuple[int | None, int | None]] | None = None


@dataclass(frozen=True, slots=True)
class SettingsPromptIntents:
    prompt_apply_settings: Callable[[SettingsSnapshot], None]


@dataclass(frozen=True, slots=True)
class SettingsOverlayIntents:
    desktop_overlay_lock_change: Callable[[bool], None]
    desktop_overlay_size_change: Callable[[str], None]
    desktop_overlay_recovery_action: Callable[[str], None]
    desktop_overlay_position_reset: Callable[[], None]
    view_logs: Callable[[], None]
    calibration_begin: Callable[[], object] | None = None
    calibration_change: Callable[[str, object], object] | None = None
    calibration_apply: Callable[[], object] | None = None
    calibration_cancel: Callable[[], object] | None = None


class SettingsIntentConsumer(Protocol):
    def bind_settings_intents(
        self,
        *,
        surface: SettingsSurfaceIntents,
        provider: SettingsProviderIntents,
        general: SettingsGeneralIntents,
        prompt: SettingsPromptIntents,
        overlay: SettingsOverlayIntents,
    ) -> None: ...


class SettingsProviderStateSink(Protocol):
    has_provider_changes: bool

    def load_from_settings(
        self,
        settings: SettingsSnapshot,
        *,
        config_path: Path,
        preserve_custom_vocab_draft: bool = False,
    ) -> None: ...

    def refresh_after_openrouter_pkce_success(
        self,
        settings: SettingsSnapshot,
        *,
        config_path: Path,
    ) -> None: ...

    def set_managed_key_state(
        self,
        *,
        visible: bool,
        remaining_percent: int | None = None,
        referral_id: str | None = None,
        pass_status: object | None = None,
        remember_referral_id: bool = True,
    ) -> None: ...

    def set_managed_trial_usage_state(
        self,
        *,
        visible: bool,
        remaining_percent: int | None = None,
    ) -> None: ...

    def set_local_cpu_auto_available(self, available: bool) -> None: ...

    def set_gpu_devices(
        self,
        *,
        devices: tuple[GpuDeviceOption, ...] | None = None,
        llm_devices: tuple[GpuDeviceOption, ...] | None = None,
    ) -> None: ...

    def consume_provider_apply_settings(self) -> SettingsSnapshot | None: ...

    def apply_locale(self) -> None: ...


class SettingsApiSlotProvider(Protocol):
    def self_stt_control(self) -> ft.Control: ...

    def peer_stt_control(self) -> ft.Control: ...

    def translation_provider_control(self) -> ft.Control: ...

    def translation_connection_control(self) -> ft.Control: ...

    def http_extension_control(self) -> ft.Control: ...

    def translation_fallback_control(self) -> ft.Control: ...

    def gpu_device_control(self) -> ft.Control: ...

    def gpu_llm_control(self) -> ft.Control: ...

    def gpu_refresh_control(self) -> ft.Control: ...

    def local_llm_connection_control(self) -> ft.Control: ...

    def custom_stt_connection_control(self) -> ft.Control: ...

    def managed_key_control(self) -> ft.Control: ...

    def peer_expected_language_control(self) -> ft.Control: ...

    def api_keys_control(self) -> ft.Control: ...


@dataclass(frozen=True, slots=True)
class SettingsApiSurfaceSlots:
    self_stt: ft.Control
    peer_stt: ft.Control
    translation_provider: ft.Control
    translation_connection: ft.Control
    translation_fallback: ft.Control
    gpu_device: ft.Control
    gpu_llm: ft.Control
    gpu_refresh: ft.Control
    local_llm_connection: ft.Control
    custom_stt_connection: ft.Control
    managed_key: ft.Control
    peer_expected_language: ft.Control
    api_keys: ft.Control
    http_extension: ft.Control | None = None

    @classmethod
    def from_slot_provider(cls, provider: SettingsApiSlotProvider) -> SettingsApiSurfaceSlots:
        extension_factory = getattr(provider, "http_extension_control", None)
        return cls(
            self_stt=provider.self_stt_control(),
            peer_stt=provider.peer_stt_control(),
            translation_provider=provider.translation_provider_control(),
            translation_connection=provider.translation_connection_control(),
            translation_fallback=provider.translation_fallback_control(),
            gpu_device=provider.gpu_device_control(),
            gpu_llm=provider.gpu_llm_control(),
            gpu_refresh=provider.gpu_refresh_control(),
            local_llm_connection=provider.local_llm_connection_control(),
            custom_stt_connection=provider.custom_stt_connection_control(),
            managed_key=provider.managed_key_control(),
            peer_expected_language=provider.peer_expected_language_control(),
            api_keys=provider.api_keys_control(),
            http_extension=(extension_factory() if callable(extension_factory) else None),
        )


@dataclass(frozen=True, slots=True)
class SettingsApiSurfaceRegions:
    rows: tuple[ft.Control, ...]
    provider_row: ft.Container
    provider_controls: ft.Row
    translation_connection_row: ft.Container
    translation_connection_controls: ft.Row
    translation_connection_leading_placeholder: ft.Control
    gpu_device_row: ft.Container
    gpu_device_controls: ft.Row


@dataclass(frozen=True, slots=True)
class SettingsGeneralSurfaceSlots:
    ui: ft.Control
    chatbox_source: ft.Control
    audio_host_api: ft.Control
    microphone: ft.Control
    loopback: ft.Control
    microphone_test: ft.Control
    self_vad: ft.Control
    peer_vad: ft.Control
    clipboard_auto_translate: ft.Control
    vrchat_mic_intercept: ft.Control
    telemetry_consent: ft.Control


@dataclass(frozen=True, slots=True)
class SettingsPromptSurfaceSlots:
    custom_vocabulary: ft.Control
    persona: ft.Control


@dataclass(frozen=True, slots=True)
class SettingsOverlaySurfaceSlots:
    overlay_target: ft.Control
    overlay_translation: ft.Control
    overlay_peer_original: ft.Control
    anchor: ft.Control
    distance: ft.Control
    offset_x: ft.Control
    offset_y: ft.Control
    text_scale: ft.Control
    vr_reset: ft.Control
    desktop_size: ft.Control
    desktop_lock: ft.Control
    desktop_background_alpha: ft.Control
    desktop_swap_caption_languages: ft.Control
    desktop_reset: ft.Control
    desktop_reset_spacer: ft.Control
    desktop_status: ft.Control
    desktop_status_trailing: ft.Control


@dataclass(frozen=True, slots=True)
class SettingsGeneralSurfaceRegions:
    rows: tuple[ft.Control, ...]
    primary_row: ft.Container
    audio_row: ft.Container
    vad_row: ft.Container
    clipboard_row: ft.Container
    primary_row_placeholder: ft.Control


@dataclass(frozen=True, slots=True)
class SettingsPromptSurfaceRegions:
    rows: tuple[ft.Control, ...]


@dataclass(frozen=True, slots=True)
class SettingsOverlaySurfaceRegions:
    rows: tuple[ft.Control, ...]
    target_row: ft.Container
    vr_rows: tuple[ft.Container, ...]
    desktop_rows: tuple[ft.Container, ...]
    desktop_controls_row: ft.Container
    recovery_row: ft.Container
    recovery_row_placeholder: ft.Control


__all__ = [
    "SettingsApiSlotProvider",
    "SettingsApiSurfaceRegions",
    "SettingsApiSurfaceSlots",
    "SettingsGeneralIntents",
    "SettingsGeneralSurfaceRegions",
    "SettingsGeneralSurfaceSlots",
    "SettingsIntentConsumer",
    "SettingsOverlayIntents",
    "SettingsOverlaySurfaceRegions",
    "SettingsOverlaySurfaceSlots",
    "SettingsPromptIntents",
    "SettingsPromptSurfaceRegions",
    "SettingsPromptSurfaceSlots",
    "SettingsProviderIntents",
    "SettingsProviderStateSink",
    "SettingsSnapshot",
    "SettingsSurfaceIntents",
]
