from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from puripuly_heart.app.ports.ui_application import UiApplicationPort
from puripuly_heart.app.services.application_shutdown import application_shutdown_callback
from puripuly_heart.composition.application_runtime import compose_application_runtime
from puripuly_heart.core.lifecycle import SHUTDOWN_PHASE_CLOSE_LOGGING_DIAGNOSTICS
from puripuly_heart.core.runtime_logging import (
    RuntimeLoggingSinks,
    install_headless_console_privacy_filter,
)
from puripuly_heart.domain.events import SelfTurnPresentation
from puripuly_heart.ui.event_dispatch import UIEventBridge
from puripuly_heart.ui.i18n import available_locales, get_locale, set_locale, t

logger = logging.getLogger(__name__)


class _HeadlessDashboard:
    def __init__(self) -> None:
        self.status: str | None = None
        self.transcript: str | None = None
        self.translation: str | None = None
        self.issue: str | None = None

    def publish_status(self, status: str) -> None:
        self.status = status

    def publish_transcript(
        self, text: str, *, language_code: str | None = None, debug_prefix: str | None = None
    ) -> bool:
        self.transcript = text
        return True

    def publish_translation(
        self, text: str, *, language_code: str | None = None, debug_prefix: str | None = None
    ) -> bool:
        self.translation = text
        return True

    def publish_self_turn(self, snapshot: SelfTurnPresentation) -> bool:
        self.transcript = snapshot.source_text
        self.translation = (
            "\n".join(
                target.translation.text
                for target in snapshot.targets
                if target.status == "ready"
                and target.source_revision == snapshot.source_revision
                and target.translation is not None
                and snapshot.status not in ("cancelled", "expired")
            )
            or None
        )
        return True

    def publish_error(self, text: str) -> None:
        self.issue = text


class _HeadlessHistory:
    def append_entry(
        self, source: str, text: str, *, translated: bool = False, language_code: str | None = None
    ) -> None:
        pass

    def publish_self_turn(self, snapshot: SelfTurnPresentation, *, source: str) -> None:
        pass


class HeadlessApplicationPresentation:
    """Nonvisual scheduling, localization, event consumption and interaction state."""

    debug_ui_preview = False

    def __init__(self) -> None:
        self.application: UiApplicationPort | None = None
        self.tasks: set[asyncio.Task] = set()
        self.dashboard = _HeadlessDashboard()
        self.projections: dict[str, object] = {}

    def _remember(self, key: str, value: object) -> None:
        self.projections[key] = value

    def refresh_overlay_peer_contract(self, state: object) -> None:
        self._remember("overlay_peer", state)

    def apply_locale(self) -> None:
        pass

    def set_locale(self, locale: str) -> None:
        set_locale(locale)

    def current_locale(self) -> str:
        return get_locale()

    def available_locales(self) -> tuple[str, ...]:
        return available_locales()

    def localize(self, message_key: str, **kwargs: object) -> str:
        return t(message_key, **kwargs)

    def show_message(
        self, message_key: str, *, is_error: bool = False, **message_kwargs: object
    ) -> None:
        if is_error:
            self.dashboard.publish_error(self.localize(message_key, **message_kwargs))
            logger.error("Application message: %s", message_key)
            return
        logger.warning("Application message: %s", message_key)

    def attach_runtime_log_sink(self, runtime_logging: object) -> None:
        self._remember("runtime_logging", runtime_logging)

    def schedule_task(self, coroutine_factory: object, *args: object) -> bool:
        if not callable(coroutine_factory):
            return False
        task = asyncio.create_task(coroutine_factory(*args))
        self.tasks.add(task)
        task.add_done_callback(self._on_task_done)
        return True

    def _on_task_done(self, task: asyncio.Task) -> None:
        self.tasks.discard(task)
        if not task.cancelled():
            try:
                task.result()
            except Exception:
                logger.exception("Headless application scheduled task failed")

    def create_ui_event_bridge(
        self, *, event_queue: object, runtime_logging: object
    ) -> UIEventBridge:
        app = self.application
        if app is None:
            raise RuntimeError("headless application is not attached")
        return UIEventBridge(
            event_queue=event_queue,
            runtime_logging=runtime_logging,
            dashboard_destination=self.dashboard,
            history_destination=_HeadlessHistory(),
            get_language_codes=app.get_event_language_codes,
            is_translation_enabled=lambda: app.state().translation_enabled,
            get_stt_state=lambda: app.state().stt_state,
            clear_managed_auth_pending=app.clear_managed_auth_pending_state,
            on_github_star_translation_success=app.schedule_github_star_prompt_translation_success_observed,
            on_overlay_state_changed=lambda **values: self._remember("overlay_state", values),
        )

    def set_dashboard_translation_enabled(self, enabled: bool) -> None:
        self._remember("translation_enabled", enabled)

    def set_dashboard_stt_enabled(self, enabled: bool) -> None:
        self._remember("stt_enabled", enabled)

    def set_dashboard_translation_needs_key(self, needs_key: bool) -> None:
        self._remember("translation_needs_key", needs_key)

    def set_dashboard_stt_needs_key(self, needs_key: bool) -> None:
        self._remember("stt_needs_key", needs_key)

    def set_dashboard_peer_needs_key(self, needs_key: bool) -> None:
        self._remember("peer_needs_key", needs_key)

    def dashboard_translation_enabled(self) -> bool:
        return bool(self.projections.get("translation_enabled", False))

    def set_dashboard_managed_auth_pending(self, pending: bool) -> None:
        self._remember("managed_auth_pending", pending)

    def set_dashboard_gpu_state(
        self,
        *,
        devices: tuple,
        state: str,
        progress_percent: int | None,
        notice: object | None,
        publish_notice: bool,
    ) -> None:
        self._remember(
            "gpu",
            {
                "devices": devices,
                "state": state,
                "progress_percent": progress_percent,
                "notice": notice,
            },
        )

    def set_dashboard_llm_gpu_devices(self, *, devices: tuple) -> None:
        self._remember("llm_gpu_devices", devices)

    def set_dashboard_local_stt_notice(
        self, *, status: str | None, model_id: str | None, percent: int | None, starting: bool
    ) -> None:
        self._remember(
            "local_stt",
            {"status": status, "model_id": model_id, "percent": percent, "starting": starting},
        )

    def set_dashboard_managed_gemma_notice(self, notice: object | None) -> None:
        self._remember("managed_gemma", notice)

    def set_dashboard_translation_starting(self, starting: bool) -> None:
        self._remember("translation_starting", starting)

    def set_dashboard_vrchat_osc_notice(self, active: bool) -> None:
        self._remember("vrchat_osc", active)

    def set_dashboard_overlay_session_fallback_notice(self, active: bool) -> None:
        self._remember("overlay_fallback", active)

    def set_dashboard_languages(self, **languages: object) -> None:
        self._remember("languages", languages)

    def project_osc_control_state(self, state: object) -> None:
        self._remember("osc_state", state)

    def render_settings(
        self,
        *,
        provider: object,
        general: object,
        prompt: object,
        overlay: object,
        config_path: Path,
        preserve_custom_vocab_draft: bool = False,
    ) -> bool:
        self._remember("settings", (provider, general, prompt, overlay))
        return True

    def refresh_settings_after_openrouter_pkce_success(
        self, *, provider: object, prompt: object, config_path: Path
    ) -> bool:
        self._remember("provider", provider)
        self._remember("prompt", prompt)
        return True

    def set_settings_overlay_calibration(self, calibration: object) -> None:
        self._remember("calibration", calibration)

    def refresh_settings_loopback_capture_target(self, general: object) -> None:
        self._remember("general", general)

    def set_settings_local_cpu_auto_available(self, available: bool) -> None:
        self._remember("local_cpu_auto", available)

    def set_settings_managed_key_state(
        self,
        *,
        visible: bool,
        remaining_percent: int | None,
        referral_id: str | None,
        pass_status: object | None,
    ) -> None:
        self._remember(
            "managed_key",
            {
                "visible": visible,
                "remaining_percent": remaining_percent,
                "pass_status": pass_status,
            },
        )

    def add_history_entry(self, *args: Any, **kwargs: Any) -> None:
        pass

    def get_event_language_codes(self) -> tuple[str | None, str | None]:
        app = self.application
        return app.get_event_language_codes() if app is not None else (None, None)

    def is_event_translation_enabled(self) -> bool:
        app = self.application
        return app.state().translation_enabled if app is not None else False

    def get_event_stt_state(self) -> object | None:
        app = self.application
        return app.state().stt_state if app is not None else None

    def clear_managed_auth_pending_state(self) -> None:
        self._remember("managed_auth_pending", False)

    def show_snackbar(self, *args: Any, **kwargs: Any) -> None:
        logger.warning("Application notification received")

    def on_github_star_translation_success(self) -> None:
        app = self.application
        if app is not None:
            app.schedule_github_star_prompt_translation_success_observed()

    def on_overlay_state_changed(self, **kwargs: Any) -> None:
        self._remember("overlay_state", kwargs)

    def on_desktop_overlay_state_changed(self, *args: Any, **kwargs: Any) -> None:
        self._remember("desktop_overlay_state", (args, kwargs))

    def show_qq_managed_auth_dialog(self) -> bool:
        self._remember("auth_action_required", "qq_browser_authorization")
        return False

    def show_founder_letter_dialog(self) -> bool:
        return False

    def show_local_qwen_hallucination_dialog(self) -> bool:
        self._remember("model_action_required", "local_qwen_consent")
        return False

    async def close_after_launch_tasks(self) -> None:
        tasks = tuple(self.tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close_github_star_prompt_runtime(self) -> None:
        app = self.application
        if app is not None:
            await app.close_github_star_prompt_runtime()

    async def close_oauth_runtime(self) -> None:
        app = self.application
        if app is not None:
            await app.close_managed_auth_tasks()


def compose_headless_application(
    config_path: Path, *, runtime_logging_sinks: RuntimeLoggingSinks | None = None
) -> UiApplicationPort:
    detach_privacy_filter = install_headless_console_privacy_filter(runtime_logging_sinks)
    try:
        presentation = HeadlessApplicationPresentation()
        application = compose_application_runtime(
            presentation=presentation,
            config_path=config_path,
            runtime_logging_sinks=runtime_logging_sinks,
        )
        presentation.application = application
        application.register_application_shutdown_callbacks(
            (
                application_shutdown_callback(
                    phase=SHUTDOWN_PHASE_CLOSE_LOGGING_DIAGNOSTICS,
                    owner_name="HeadlessConsolePrivacy",
                    callback_name="detach_console_filter",
                    callback=detach_privacy_filter,
                ),
            )
        )
        return application
    except BaseException:
        detach_privacy_filter()
        raise
