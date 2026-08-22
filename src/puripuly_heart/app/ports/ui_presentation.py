from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from puripuly_heart.app.ports.ui_models import (
    ManagedGemmaDashboardNotice,
    OverlayPeerPresentationState,
)
from puripuly_heart.core.runtime.output import UIEventBridgePort


class UiPresentationPort(Protocol):
    @property
    def debug_ui_preview(self) -> bool: ...

    def refresh_overlay_peer_contract(
        self,
        state: OverlayPeerPresentationState | None,
    ) -> None: ...

    def apply_locale(self) -> None: ...

    def set_locale(self, locale: str) -> None: ...

    def current_locale(self) -> str: ...

    def localize(self, message_key: str, **message_kwargs: object) -> str: ...

    def show_message(self, message_key: str, **message_kwargs: object) -> None: ...

    def attach_runtime_log_sink(self, runtime_logging: object) -> None: ...

    def schedule_task(self, coroutine_factory: object, *args: object) -> bool: ...

    def create_ui_event_bridge(
        self,
        *,
        event_queue: object,
        runtime_logging: object,
    ) -> UIEventBridgePort: ...

    def set_dashboard_translation_enabled(self, enabled: bool) -> None: ...

    def set_dashboard_stt_enabled(self, enabled: bool) -> None: ...

    def set_dashboard_translation_needs_key(self, needs_key: bool) -> None: ...

    def set_dashboard_stt_needs_key(self, needs_key: bool) -> None: ...

    def dashboard_translation_enabled(self) -> bool: ...

    def set_dashboard_managed_auth_pending(self, pending: bool) -> None: ...

    def set_dashboard_gpu_state(
        self,
        *,
        devices: tuple[object, ...],
        state: str,
        progress_percent: int | None,
        notice: object | None,
        publish_notice: bool,
    ) -> None: ...

    def set_dashboard_llm_gpu_devices(self, *, devices: tuple[object, ...]) -> None: ...

    def set_dashboard_local_stt_notice(
        self,
        *,
        status: str | None,
        model_id: str | None,
        percent: int | None,
        starting: bool,
    ) -> None: ...

    def set_dashboard_managed_gemma_notice(
        self,
        notice: ManagedGemmaDashboardNotice | None,
    ) -> None: ...

    def set_dashboard_translation_starting(self, starting: bool) -> None: ...

    def set_dashboard_vrchat_osc_notice(self, active: bool) -> None: ...

    def set_dashboard_overlay_session_fallback_notice(self, active: bool) -> None: ...

    def set_dashboard_languages(
        self,
        *,
        source_language: str,
        target_language: str,
        peer_source_language: str,
        peer_target_language: str,
        peer_source_mode: str,
        recent_source_languages: list[str],
        recent_target_languages: list[str],
        peer_auto_detect_available: bool,
    ) -> None: ...

    def render_settings(
        self,
        settings: object,
        *,
        config_path: Path,
        preserve_custom_vocab_draft: bool = False,
    ) -> bool: ...

    def refresh_settings_after_openrouter_pkce_success(
        self,
        settings: object,
        *,
        config_path: Path,
    ) -> bool: ...

    def set_settings_overlay_calibration(self, calibration: object) -> None: ...

    def refresh_settings_loopback_capture_target(self, settings: object) -> None: ...

    def set_settings_local_cpu_auto_available(self, available: bool) -> None: ...

    def set_settings_managed_key_state(
        self,
        *,
        visible: bool,
        remaining_percent: int | None,
        referral_id: str | None,
        pass_status: object | None,
    ) -> None: ...

    def add_history_entry(self, *args: Any, **kwargs: Any) -> None: ...

    def get_event_language_codes(self) -> tuple[str | None, str | None]: ...

    def is_event_translation_enabled(self) -> bool: ...

    def get_event_stt_state(self) -> object | None: ...

    def clear_managed_auth_pending_state(self) -> None: ...

    def show_snackbar(self, *args: Any, **kwargs: Any) -> None: ...

    def on_github_star_translation_success(self) -> None: ...

    def on_telemetry_translation_success(self) -> None: ...

    def on_overlay_state_changed(self, **kwargs: Any) -> None: ...

    def on_desktop_overlay_state_changed(self, *args: Any, **kwargs: Any) -> None: ...

    def show_qq_managed_auth_dialog(self) -> bool: ...

    def show_founder_letter_dialog(self) -> bool: ...

    def show_local_qwen_hallucination_dialog(self) -> bool: ...

    async def close_after_launch_tasks(self) -> None: ...

    async def close_github_star_prompt_runtime(self) -> None: ...

    async def close_oauth_runtime(self) -> None: ...


__all__ = ["UIEventBridgePort", "UiPresentationPort"]
