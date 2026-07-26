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


class SettingsIntentConsumer(Protocol):
    def bind_settings_intents(
        self,
        *,
        surface: SettingsSurfaceIntents,
        provider: SettingsProviderIntents,
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

    def set_gpu_devices(self, *, devices: tuple[GpuDeviceOption, ...]) -> None: ...

    def consume_provider_apply_settings(self) -> SettingsSnapshot | None: ...

    def apply_locale(self) -> None: ...


class SettingsApiSlotProvider(Protocol):
    def self_stt_control(self) -> ft.Control: ...

    def peer_stt_control(self) -> ft.Control: ...

    def translation_provider_control(self) -> ft.Control: ...

    def translation_connection_control(self) -> ft.Control: ...

    def translation_fallback_control(self) -> ft.Control: ...

    def gpu_device_control(self) -> ft.Control: ...

    def local_llm_connection_control(self) -> ft.Control: ...

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
    local_llm_connection: ft.Control
    managed_key: ft.Control
    peer_expected_language: ft.Control
    api_keys: ft.Control

    @classmethod
    def from_slot_provider(cls, provider: SettingsApiSlotProvider) -> SettingsApiSurfaceSlots:
        return cls(
            self_stt=provider.self_stt_control(),
            peer_stt=provider.peer_stt_control(),
            translation_provider=provider.translation_provider_control(),
            translation_connection=provider.translation_connection_control(),
            translation_fallback=provider.translation_fallback_control(),
            gpu_device=provider.gpu_device_control(),
            local_llm_connection=provider.local_llm_connection_control(),
            managed_key=provider.managed_key_control(),
            peer_expected_language=provider.peer_expected_language_control(),
            api_keys=provider.api_keys_control(),
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
    gpu_device_placeholders: tuple[ft.Control, ...]


__all__ = [
    "SettingsApiSlotProvider",
    "SettingsApiSurfaceRegions",
    "SettingsApiSurfaceSlots",
    "SettingsIntentConsumer",
    "SettingsProviderIntents",
    "SettingsProviderStateSink",
    "SettingsSnapshot",
    "SettingsSurfaceIntents",
]
