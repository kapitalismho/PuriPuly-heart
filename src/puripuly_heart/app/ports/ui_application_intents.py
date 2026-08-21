from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from puripuly_heart.app.language_selection import LanguageSelectionChange
from puripuly_heart.app.ports.ui_models import (
    GpuNoticeAction,
    ManagedGemmaNoticeAction,
    OverlayPeerPresentationState,
)


class UiInputRuntimePort(Protocol):
    async def submit_text(self, text: str) -> None: ...

    def set_manual_input_activity(self, has_text: bool) -> None: ...

    async def set_translation_enabled(self, enabled: bool) -> object: ...

    async def set_stt_enabled(self, enabled: bool) -> object: ...


class UiPeerCaptureRuntimePort(Protocol):
    async def set_peer_translation_enabled(self, enabled: bool) -> object: ...

    async def retry_peer_process_capture(self) -> bool: ...

    async def apply_loopback_capture_option(self, value: str) -> None: ...

    def list_loopback_capture_options(self) -> object: ...

    def list_loopback_process_options(self) -> object: ...

    def list_loopback_device_options(self) -> object: ...

    def current_loopback_capture_option_value(self) -> object: ...

    def loopback_capture_summary(self) -> object: ...

    def overlay_peer_presentation_state(self) -> OverlayPeerPresentationState | None: ...


class UiSettingsRuntimePort(Protocol):
    async def on_dashboard_language_change(
        self,
        change: LanguageSelectionChange,
    ) -> None: ...

    def capture_settings_view_change(self, settings: object) -> object: ...

    def merge_settings_view_change_with_current(self, captured: object) -> object: ...

    def refresh_settings_projection(
        self,
        *,
        preserve_custom_vocab_draft: bool = False,
    ) -> bool: ...

    def refresh_settings_after_openrouter_pkce_success(self) -> bool: ...

    def merge_settings_tab_apply_with_current_languages(
        self,
        settings: object,
    ) -> object: ...

    async def apply_settings(self, settings: object) -> object: ...

    async def apply_telemetry_consent(self, consent: str) -> object | None: ...


class UiProviderRuntimePort(Protocol):
    async def apply_providers(
        self,
        settings: object | None = None,
        *,
        force_rebuild_llm: bool = False,
        persist_settings: bool = True,
        refresh_ui: bool = True,
    ) -> object: ...

    async def install_selected_gpu_model_if_needed(self) -> None: ...

    async def ensure_gpu_device_discovery(self) -> None: ...

    async def connect_openrouter_via_pkce(
        self,
        *,
        target_settings: object,
        launch_source: str,
    ) -> bool: ...

    def reopen_openrouter_pkce_authorization_url(self) -> object: ...

    def build_managed_openrouter_byok_target_settings(self) -> object | None: ...

    async def verify_api_key(self, provider: str, key: str) -> tuple[bool, str]: ...

    def persist_api_key_verification(
        self,
        provider: str,
        key: str,
        success: bool,
    ) -> None: ...

    async def persist_provider_secret_change(self, key: str, value: str) -> bool: ...

    def clear_provider_verification(self, provider: str) -> None: ...

    def handle_gpu_notice_action(self, action: GpuNoticeAction) -> object: ...

    async def handle_managed_gemma_notice_action(
        self,
        action: ManagedGemmaNoticeAction,
    ) -> object: ...


class UiMicrophoneRuntimePort(Protocol):
    async def start_microphone_test(
        self,
        *,
        meter_callback: Callable[[float], None] | None = None,
    ) -> bool: ...

    async def stop_microphone_test(self) -> None: ...


class UiOverlayRuntimePort(Protocol):
    async def set_overlay_enabled(self, enabled: bool) -> object: ...

    async def set_desktop_overlay_captions_locked(self, locked: bool) -> None: ...

    async def set_desktop_overlay_size_preset(self, size_preset: str) -> None: ...

    async def reset_desktop_overlay_position(self) -> None: ...

    def begin_overlay_calibration(self) -> object: ...

    def set_overlay_calibration_field(
        self,
        field_name: str,
        value: object,
    ) -> object: ...

    def apply_overlay_calibration(self) -> object: ...

    def cancel_overlay_calibration(self) -> object: ...


class UiManagedRuntimePort(Protocol):
    def dashboard_managed_auth_action(self) -> str: ...

    def dashboard_managed_auth_prompt_kind(self) -> str: ...

    async def start_qq_managed_auth_from_dialog(
        self,
        **kwargs: object,
    ) -> object: ...

    async def start_discord_managed_auth_from_dialog(
        self,
        **kwargs: object,
    ) -> object: ...

    def clear_managed_auth_pending_state(self) -> None: ...

    async def refresh_openrouter_usage_after_launch(self) -> bool: ...


class UiEngagementRuntimePort(Protocol):
    def get_event_language_codes(self) -> tuple[str | None, str | None]: ...

    def schedule_github_star_prompt_translation_success_observed(self) -> None: ...

    async def record_telemetry_translation_success_day(self) -> None: ...

    def should_show_github_star_prompt(self) -> bool: ...

    async def persist_github_star_prompt_eligible_launch(self) -> bool: ...

    async def prepare_runtime_after_launch(self) -> None: ...

    async def persist_github_star_prompt_opened(
        self,
        *,
        should_open: Callable[[], bool] | None = None,
    ) -> bool: ...

    async def persist_github_star_prompt_clicked(self) -> None: ...


class UiDiagnosticsRuntimePort(Protocol):
    def set_runtime_logging_mode(self, mode: str) -> None: ...

    def cycle_debug_capture_fault_profile(self) -> str: ...

    def cycle_debug_stt_fault_profile(self) -> str: ...

    def clear_debug_audio_fault_profiles(self) -> None: ...


__all__ = [
    "UiDiagnosticsRuntimePort",
    "UiEngagementRuntimePort",
    "UiInputRuntimePort",
    "UiManagedRuntimePort",
    "UiMicrophoneRuntimePort",
    "UiOverlayRuntimePort",
    "UiPeerCaptureRuntimePort",
    "UiProviderRuntimePort",
    "UiSettingsRuntimePort",
]
