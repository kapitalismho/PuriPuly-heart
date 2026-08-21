from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import flet as ft

from puripuly_heart.app.ports.ui_models import (
    ManagedGemmaDashboardNotice,
    OverlayPeerPresentationState,
)
from puripuly_heart.app.ports.ui_presentation import UIEventBridgePort, UiPresentationPort
from puripuly_heart.ui.event_bridge import (
    AppConversationEventDestination,
    AppDashboardEventDestination,
    AppHistoryEventDestination,
    UIEventBridge,
)
from puripuly_heart.ui.i18n import get_locale as get_ui_locale
from puripuly_heart.ui.i18n import set_locale as set_ui_locale
from puripuly_heart.ui.i18n import t
from puripuly_heart.ui.overlay_peer_contract import (
    build_overlay_peer_consumer_contract_from_state,
)


@dataclass(slots=True)
class FletUiPresentationAdapter:
    _app: UiPresentationPort

    @property
    def debug_ui_preview(self) -> bool:
        return bool(getattr(self._app, "debug_ui_preview", False))

    def refresh_overlay_peer_contract(
        self,
        state: OverlayPeerPresentationState | None,
    ) -> None:
        if state is None:
            return
        contract = build_overlay_peer_consumer_contract_from_state(state)
        setattr(self._app, "overlay_peer_contract", contract)
        for view_name in ("view_settings", "view_dashboard"):
            view = getattr(self._app, view_name, None)
            setter = getattr(view, "set_overlay_peer_contract", None)
            if callable(setter):
                setter(contract)

    def apply_locale(self) -> None:
        apply_locale = getattr(self._app, "apply_locale", None)
        if callable(apply_locale):
            apply_locale()

    def set_locale(self, locale: str) -> None:
        set_ui_locale(locale)

    def current_locale(self) -> str:
        return get_ui_locale()

    def localize(self, message_key: str, **message_kwargs: object) -> str:
        return t(message_key, **message_kwargs)

    def show_message(self, message_key: str, **message_kwargs: object) -> None:
        show_snackbar = getattr(self._app, "show_snackbar", None)
        if not callable(show_snackbar):
            show_snackbar = getattr(self._app, "_show_snackbar", None)
        if callable(show_snackbar):
            show_snackbar(
                self.localize(message_key, **message_kwargs),
                ft.Colors.ORANGE_700,
            )

    def attach_runtime_log_sink(self, runtime_logging: object) -> None:
        sink = getattr(self._app, "view_logs", None)
        attach = getattr(runtime_logging, "attach_realtime_sink", None)
        if sink is not None and callable(attach):
            attach(sink)

    def schedule_task(self, coroutine_factory: object, *args: object) -> bool:
        schedule = getattr(self._app, "_run_page_task", None)
        if callable(schedule):
            schedule(coroutine_factory, *args)
            return True
        page = getattr(self._app, "page", None)
        run_task = getattr(page, "run_task", None)
        if callable(run_task):
            run_task(coroutine_factory, *args)
            return True
        return False

    def create_ui_event_bridge(
        self,
        *,
        event_queue: object,
        runtime_logging: object,
    ) -> UIEventBridgePort:
        logs_view = getattr(self._app, "view_logs", None)
        return UIEventBridge(
            event_queue=event_queue,
            runtime_logging=runtime_logging,
            dashboard_destination=AppDashboardEventDestination(
                getattr(self._app, "view_dashboard", None)
            ),
            history_destination=AppHistoryEventDestination(
                getattr(self._app, "add_history_entry", None)
            ),
            conversation_destination=AppConversationEventDestination(
                getattr(logs_view, "append_conversation_record", None)
            ),
            get_language_codes=getattr(self._app, "get_event_language_codes", None),
            is_translation_enabled=getattr(self._app, "is_event_translation_enabled", None),
            get_stt_state=getattr(self._app, "get_event_stt_state", None),
            clear_managed_auth_pending=getattr(
                self._app,
                "clear_managed_auth_pending_state",
                None,
            ),
            show_snackbar=getattr(self._app, "show_snackbar", None),
            on_github_star_translation_success=getattr(
                self._app,
                "on_github_star_translation_success",
                None,
            ),
            on_telemetry_translation_success=getattr(
                self._app,
                "on_telemetry_translation_success",
                None,
            ),
            on_overlay_state_changed=getattr(self._app, "on_overlay_state_changed", None),
        )

    def set_dashboard_translation_enabled(self, enabled: bool) -> None:
        dashboard = getattr(self._app, "view_dashboard", None)
        setter = getattr(dashboard, "set_translation_enabled", None)
        if callable(setter):
            setter(enabled)

    def set_dashboard_stt_enabled(self, enabled: bool) -> None:
        dashboard = getattr(self._app, "view_dashboard", None)
        setter = getattr(dashboard, "set_stt_enabled", None)
        if callable(setter):
            setter(enabled)

    def set_dashboard_translation_needs_key(self, needs_key: bool) -> None:
        dashboard = getattr(self._app, "view_dashboard", None)
        setter = getattr(dashboard, "set_translation_needs_key", None)
        if callable(setter):
            setter(needs_key)
        elif dashboard is not None:
            dashboard.translation_needs_key = needs_key

    def set_dashboard_stt_needs_key(self, needs_key: bool) -> None:
        dashboard = getattr(self._app, "view_dashboard", None)
        setter = getattr(dashboard, "set_stt_needs_key", None)
        if callable(setter):
            setter(needs_key)
        elif dashboard is not None:
            dashboard.stt_needs_key = needs_key

    def dashboard_translation_enabled(self) -> bool:
        dashboard = getattr(self._app, "view_dashboard", None)
        return bool(getattr(dashboard, "is_translation_on", True))

    def set_dashboard_managed_auth_pending(self, pending: bool) -> None:
        dashboard = getattr(self._app, "view_dashboard", None)
        setter = getattr(dashboard, "set_managed_auth_pending", None)
        if callable(setter):
            setter(pending)

    def set_dashboard_gpu_state(
        self,
        *,
        devices: tuple[object, ...],
        state: str,
        progress_percent: int | None,
        notice: object | None,
        publish_notice: bool,
    ) -> None:
        settings_view = getattr(self._app, "view_settings", None)
        set_devices = getattr(settings_view, "set_gpu_devices", None)
        if callable(set_devices):
            set_devices(devices=devices)
        dashboard = getattr(self._app, "view_dashboard", None)
        set_notice = getattr(dashboard, "set_gpu_notice", None)
        if callable(set_notice):
            set_notice(notice if publish_notice else None)
            return
        if publish_notice and not callable(set_devices):
            legacy_setter = getattr(settings_view, "set_gpu_runtime_state", None)
            if callable(legacy_setter):
                legacy_setter(
                    state,
                    devices=devices,
                    progress_percent=progress_percent,
                )

    def set_dashboard_local_stt_notice(
        self,
        *,
        status: str | None,
        model_id: str | None,
        percent: int | None,
        starting: bool,
    ) -> None:
        dashboard = getattr(self._app, "view_dashboard", None)
        set_starting = getattr(dashboard, "set_stt_starting", None)
        if callable(set_starting):
            set_starting(starting)
        set_model = getattr(dashboard, "set_local_stt_notice_model", None)
        if callable(set_model):
            set_model(model_id)
        set_notice = getattr(dashboard, "set_local_stt_notice", None)
        if callable(set_notice):
            set_notice(status, percent=percent)

    def set_dashboard_managed_gemma_notice(
        self,
        notice: ManagedGemmaDashboardNotice | None,
    ) -> None:
        dashboard = getattr(self._app, "view_dashboard", None)
        setter = getattr(dashboard, "set_managed_gemma_notice", None)
        if callable(setter):
            setter(notice)

    def set_dashboard_translation_starting(self, starting: bool) -> None:
        dashboard = getattr(self._app, "view_dashboard", None)
        setter = getattr(dashboard, "set_translation_starting", None)
        if callable(setter):
            setter(bool(starting))

    def set_dashboard_vrchat_osc_notice(self, active: bool) -> None:
        dashboard = getattr(self._app, "view_dashboard", None)
        setter = getattr(dashboard, "set_vrchat_osc_notice", None)
        if callable(setter):
            setter(active)

    def set_dashboard_overlay_session_fallback_notice(self, active: bool) -> None:
        dashboard = getattr(self._app, "view_dashboard", None)
        setter = getattr(dashboard, "set_overlay_session_fallback_notice", None)
        if callable(setter):
            setter(active)

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
    ) -> None:
        dashboard = getattr(self._app, "view_dashboard", None)
        if dashboard is None:
            return
        set_languages = getattr(dashboard, "set_languages_from_codes", None)
        if not callable(set_languages):
            return
        try:
            inspect.signature(set_languages).bind(
                source_language,
                target_language,
                peer_source_language,
                peer_target_language,
                peer_source_mode,
            )
        except (TypeError, ValueError):
            set_languages(
                source_language,
                target_language,
                peer_source_language,
                peer_target_language,
            )
        else:
            set_languages(
                source_language,
                target_language,
                peer_source_language,
                peer_target_language,
                peer_source_mode,
            )
        set_recent = getattr(dashboard, "set_recent_languages", None)
        if callable(set_recent):
            set_recent(
                recent_source_languages,
                recent_target_languages,
            )
        set_peer_auto = getattr(dashboard, "set_peer_auto_detect_available", None)
        if callable(set_peer_auto):
            set_peer_auto(peer_auto_detect_available)

    def render_settings(
        self,
        settings: object,
        *,
        config_path: Path,
        preserve_custom_vocab_draft: bool = False,
    ) -> bool:
        settings_view = getattr(self._app, "view_settings", None)
        loader = getattr(settings_view, "load_from_settings", None)
        if not callable(loader):
            return False
        loader(
            settings,
            config_path=config_path,
            preserve_custom_vocab_draft=preserve_custom_vocab_draft,
        )
        return True

    def refresh_settings_after_openrouter_pkce_success(
        self,
        settings: object,
        *,
        config_path: Path,
    ) -> bool:
        settings_view = getattr(self._app, "view_settings", None)
        refresh = getattr(settings_view, "refresh_after_openrouter_pkce_success", None)
        if not callable(refresh):
            return False
        refresh(settings, config_path=config_path)
        return True

    def set_settings_overlay_calibration(self, calibration: object) -> None:
        settings_view = getattr(self._app, "view_settings", None)
        setter = getattr(settings_view, "set_overlay_calibration", None)
        if callable(setter):
            setter(calibration)

    def refresh_settings_loopback_capture_target(self, settings: object) -> None:
        settings_view = getattr(self._app, "view_settings", None)
        refresh = getattr(settings_view, "refresh_loopback_capture_target", None)
        if callable(refresh):
            refresh(settings)

    def set_settings_local_cpu_auto_available(self, available: bool) -> None:
        settings_view = getattr(self._app, "view_settings", None)
        setter = getattr(settings_view, "set_local_cpu_auto_available", None)
        if callable(setter):
            setter(available)

    def set_settings_managed_key_state(
        self,
        *,
        visible: bool,
        remaining_percent: int | None,
        referral_id: str | None,
        pass_status: object | None,
    ) -> None:
        settings_view = getattr(self._app, "view_settings", None)
        setter = getattr(settings_view, "set_managed_key_state", None)
        if callable(setter):
            try:
                inspect.signature(setter).bind(
                    visible=visible,
                    remaining_percent=remaining_percent,
                    referral_id=referral_id,
                    pass_status=pass_status,
                )
            except (TypeError, ValueError):
                setter(
                    visible=visible,
                    remaining_percent=remaining_percent,
                    referral_id=referral_id,
                )
            else:
                setter(
                    visible=visible,
                    remaining_percent=remaining_percent,
                    referral_id=referral_id,
                    pass_status=pass_status,
                )
            return
        usage_setter = getattr(settings_view, "set_managed_trial_usage_state", None)
        if callable(usage_setter):
            usage_setter(
                visible=visible,
                remaining_percent=remaining_percent,
            )

    def add_history_entry(self, *args: Any, **kwargs: Any) -> None:
        callback = getattr(self._app, "add_history_entry", None)
        if callable(callback):
            callback(*args, **kwargs)

    def get_event_language_codes(self) -> tuple[str | None, str | None]:
        callback = getattr(self._app, "get_event_language_codes", None)
        return callback() if callable(callback) else (None, None)

    def is_event_translation_enabled(self) -> bool:
        callback = getattr(self._app, "is_event_translation_enabled", None)
        return bool(callback()) if callable(callback) else False

    def get_event_stt_state(self) -> object | None:
        callback = getattr(self._app, "get_event_stt_state", None)
        return callback() if callable(callback) else None

    def clear_managed_auth_pending_state(self) -> None:
        callback = getattr(self._app, "clear_managed_auth_pending_state", None)
        if callable(callback):
            callback()

    def show_snackbar(self, *args: Any, **kwargs: Any) -> None:
        callback = getattr(self._app, "show_snackbar", None)
        if not callable(callback):
            callback = getattr(self._app, "_show_snackbar", None)
        if callable(callback):
            callback(*args, **kwargs)

    def on_github_star_translation_success(self) -> None:
        callback = getattr(self._app, "on_github_star_translation_success", None)
        if callable(callback):
            callback()

    def on_telemetry_translation_success(self) -> None:
        callback = getattr(self._app, "on_telemetry_translation_success", None)
        if callable(callback):
            callback()

    def on_overlay_state_changed(self, **kwargs: Any) -> None:
        callback = getattr(self._app, "on_overlay_state_changed", None)
        if callable(callback):
            callback(**kwargs)

    def on_desktop_overlay_state_changed(self, *args: Any, **kwargs: Any) -> None:
        callback = getattr(self._app, "on_desktop_overlay_state_changed", None)
        if callable(callback):
            callback(*args, **kwargs)

    def show_qq_managed_auth_dialog(self) -> bool:
        callback = getattr(self._app, "show_qq_managed_auth_dialog", None)
        if callable(callback):
            callback()
            return True
        return False

    def show_founder_letter_dialog(self) -> bool:
        callback = getattr(self._app, "show_founder_letter_dialog", None)
        if callable(callback):
            callback()
            return True
        return False

    def show_local_qwen_hallucination_dialog(self) -> bool:
        callback = getattr(self._app, "show_local_qwen_hallucination_dialog", None)
        if callable(callback):
            callback()
            return True
        return False

    async def close_after_launch_tasks(self) -> None:
        callback = getattr(self._app, "close_after_launch_tasks", None)
        if callable(callback):
            await callback()

    async def close_github_star_prompt_runtime(self) -> None:
        callback = getattr(self._app, "close_github_star_prompt_runtime", None)
        if callable(callback):
            await callback()

    async def close_oauth_runtime(self) -> None:
        callback = getattr(self._app, "close_oauth_runtime", None)
        if callable(callback):
            await callback()


__all__ = ["FletUiPresentationAdapter"]
