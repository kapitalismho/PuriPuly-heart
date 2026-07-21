from __future__ import annotations

import copy
import inspect
import logging
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from puripuly_heart.app.language_selection import LanguageSelectionChange
from puripuly_heart.app.ports.ui_application import UiApplicationState
from puripuly_heart.app.services.application_shutdown import (
    ApplicationShutdownCallback,
    ApplicationShutdownCoordinator,
    ApplicationShutdownDiagnostic,
)
from puripuly_heart.core.runtime.github_star_prompt import GithubStarPromptRuntime
from puripuly_heart.core.runtime.oauth import OAuthRuntime
from puripuly_heart.core.updater import check_for_update


class UiApplicationBoundary:
    def __init__(self, backend: object) -> None:
        self._backend = backend
        self._github_star_prompt_runtime = GithubStarPromptRuntime(
            diagnostics_sink=self._github_star_prompt_runtime_diagnostics_sink,
        )
        self._managed_auth_runtime = OAuthRuntime()

    def wraps(self, backend: object) -> bool:
        return self._backend is backend

    def state(self) -> UiApplicationState:
        settings = getattr(self._backend, "settings", None)
        ui_settings = getattr(settings, "ui", None)
        provider_settings = getattr(settings, "provider", None)
        overlay_settings = getattr(settings, "overlay", None)
        provider = getattr(provider_settings, "llm", None)
        provider_name = getattr(provider, "value", provider)
        overlay_target = getattr(overlay_settings, "target", None)
        overlay_target_name = getattr(overlay_target, "value", overlay_target)
        hub = getattr(self._backend, "hub", None)
        stt_state = None
        stt_session_state = getattr(hub, "stt_session_state", None)
        if callable(stt_session_state):
            stt_state = stt_session_state("self")
        return UiApplicationState(
            config_path=Path(getattr(self._backend, "config_path", "settings.json")),
            runtime_logging_mode=str(getattr(self._backend, "runtime_logging_mode", "basic")),
            translation_enabled=bool(getattr(hub, "translation_enabled", False)),
            stt_state=stt_state,
            peer_translation_eula_accepted=(
                bool(getattr(ui_settings, "peer_translation_eula_accepted", False))
                if ui_settings is not None
                else None
            ),
            microphone_test_active=bool(getattr(self._backend, "microphone_test_active", False)),
            provider_name=str(provider_name) if provider_name is not None else None,
            overlay_target=(str(overlay_target_name) if overlay_target_name is not None else None),
            desktop_overlay_captions_locked=bool(
                getattr(self._backend, "desktop_overlay_captions_locked", False)
            ),
            managed_auth_referral_bonus_applied=bool(
                getattr(self._backend, "last_discord_managed_auth_referral_bonus_applied", False)
                is True
            ),
        )

    def compatibility_settings(self) -> Any | None:
        settings = getattr(self._backend, "settings", None)
        return copy.deepcopy(settings) if settings is not None else None

    @property
    def overlay_calibration(self) -> object | None:
        calibration = getattr(self._backend, "overlay_calibration", None)
        return copy.deepcopy(calibration)

    async def start(self) -> None:
        await self._backend.start()

    async def stop(self) -> None:
        await self._backend.stop()

    def application_shutdown_callbacks(self) -> Sequence[ApplicationShutdownCallback]:
        callbacks = getattr(self._backend, "application_shutdown_callbacks", None)
        return callbacks() if callable(callbacks) else ()

    def bind_application_lifecycle(self, lifecycle: ApplicationShutdownCoordinator) -> None:
        bind = getattr(self._backend, "bind_application_lifecycle", None)
        if callable(bind):
            bind(lifecycle)

    def emit_application_shutdown_diagnostic(
        self,
        diagnostic: ApplicationShutdownDiagnostic,
    ) -> Awaitable[None] | None:
        emit = getattr(self._backend, "emit_application_shutdown_diagnostic", None)
        return emit(diagnostic) if callable(emit) else None

    def log_basic(self, message: str, *, level: int = logging.INFO) -> None:
        log = getattr(self._backend, "log_basic", None)
        if callable(log):
            log(message, level=level)

    def log_detailed(self, message: str, *, level: int = logging.INFO) -> None:
        log = getattr(self._backend, "log_detailed", None)
        if callable(log):
            log(message, level=level)

    async def submit_text(self, text: str) -> None:
        await self._backend.submit_text(text)

    def set_manual_input_activity(self, has_text: bool) -> None:
        self._backend.set_manual_input_activity(has_text)

    async def set_translation_enabled(self, enabled: bool) -> object:
        return await self._backend.set_translation_enabled(enabled)

    async def set_stt_enabled(self, enabled: bool) -> object:
        return await self._backend.set_stt_enabled(enabled)

    async def set_peer_translation_enabled(self, enabled: bool) -> object:
        return await self._backend.set_peer_translation_enabled(enabled)

    async def set_overlay_enabled(self, enabled: bool) -> object:
        return await self._backend.set_overlay_enabled(enabled)

    async def retry_peer_process_capture(self) -> None:
        await self._backend.retry_peer_process_capture()

    async def apply_loopback_capture_option(self, value: str) -> None:
        await self._backend.apply_loopback_capture_option(value)

    def list_loopback_capture_options(self) -> object:
        return self._backend.list_loopback_capture_options()

    def list_loopback_process_options(self) -> object:
        return self._backend.list_loopback_process_options()

    def list_loopback_device_options(self) -> object:
        return self._backend.list_loopback_device_options()

    def current_loopback_capture_option_value(self) -> object:
        return self._backend.current_loopback_capture_option_value()

    def loopback_capture_summary(self) -> object:
        return self._backend.loopback_capture_summary()

    async def on_dashboard_language_change(self, change: LanguageSelectionChange) -> None:
        await self._backend.on_dashboard_language_change(change)

    def capture_settings_view_change(self, settings: Any) -> object:
        capture = getattr(self._backend, "capture_settings_view_change", None)
        return capture(settings) if callable(capture) else settings

    def merge_settings_view_change_with_current(self, captured: object) -> Any:
        merge = getattr(self._backend, "merge_settings_view_change_with_current", None)
        return merge(captured) if callable(merge) else captured

    def merge_settings_tab_apply_with_current_languages(self, settings: Any) -> Any:
        return self._backend.merge_settings_tab_apply_with_current_languages(settings)

    async def apply_settings(self, settings: Any) -> object:
        return await self._backend.apply_settings(settings)

    async def apply_providers(
        self,
        settings: Any | None = None,
        *,
        force_rebuild_llm: bool = False,
    ) -> object:
        if force_rebuild_llm:
            if settings is None:
                return await self._backend.apply_providers(force_rebuild_llm=True)
            return await self._backend.apply_providers(
                settings,
                force_rebuild_llm=True,
            )
        if settings is None:
            return await self._backend.apply_providers()
        return await self._backend.apply_providers(settings)

    async def install_selected_gpu_model_if_needed(self) -> None:
        await self._backend.install_selected_gpu_model_if_needed()

    async def ensure_gpu_device_discovery(self) -> None:
        await self._backend.ensure_gpu_device_discovery()

    async def start_microphone_test(
        self,
        *,
        meter_callback: Callable[[float], None] | None = None,
    ) -> bool:
        start = self._backend.start_microphone_test
        if meter_callback is not None and _accepts_keyword(start, "meter_callback"):
            result = start(meter_callback=meter_callback)
        else:
            result = start()
        return bool(await result if inspect.isawaitable(result) else result)

    async def stop_microphone_test(self) -> None:
        stop = getattr(self._backend, "stop_microphone_test", None)
        if not callable(stop):
            return
        result = stop()
        if inspect.isawaitable(result):
            await result

    def set_runtime_logging_mode(self, mode: str) -> str:
        self._backend.set_runtime_logging_mode(mode)
        return self.state().runtime_logging_mode

    async def set_desktop_overlay_captions_locked(self, locked: bool) -> None:
        await self._backend.set_desktop_overlay_captions_locked(locked)

    async def set_desktop_overlay_size_preset(self, size_preset: str) -> None:
        await self._backend.set_desktop_overlay_size_preset(size_preset)

    async def reset_desktop_overlay_position(self) -> None:
        await self._backend.reset_desktop_overlay_position()

    def begin_overlay_calibration(self) -> object:
        return self._backend.begin_overlay_calibration()

    def set_overlay_calibration_field(self, *args: Any, **kwargs: Any) -> object:
        return self._backend.set_overlay_calibration_field(*args, **kwargs)

    def apply_overlay_calibration(self) -> object:
        return self._backend.apply_overlay_calibration()

    def cancel_overlay_calibration(self) -> object:
        return self._backend.cancel_overlay_calibration()

    def build_overlay_peer_consumer_contract(self) -> object | None:
        build = getattr(self._backend, "build_overlay_peer_consumer_contract", None)
        return build() if callable(build) else None

    def dashboard_managed_auth_action(self) -> str:
        action = getattr(self._backend, "dashboard_managed_auth_action", None)
        return str(action()) if callable(action) else "continue"

    def dashboard_managed_auth_prompt_kind(self) -> str:
        prompt = getattr(self._backend, "dashboard_managed_auth_prompt_kind", None)
        return str(prompt()) if callable(prompt) else "discord"

    async def apply_telemetry_consent(self, consent: str) -> Any | None:
        apply = getattr(self._backend, "apply_telemetry_consent", None)
        if callable(apply):
            return await apply(consent)
        return None

    async def accept_peer_translation_eula_and_enable(self) -> object:
        settings = self.compatibility_settings()
        if settings is not None:
            settings.ui.peer_translation_eula_accepted = True
            await self.apply_settings(settings)
        return await self.set_peer_translation_enabled(True)

    def local_llm_selected(self) -> bool:
        return self.state().provider_name == "local_llm"

    async def connect_openrouter_via_pkce(
        self, *, target_settings: Any, launch_source: str
    ) -> bool:
        return bool(
            await self._backend.connect_openrouter_via_pkce(
                target_settings=target_settings,
                launch_source=launch_source,
            )
        )

    def reopen_openrouter_pkce_authorization_url(self) -> None:
        reopen = getattr(self._backend, "reopen_openrouter_pkce_authorization_url", None)
        if callable(reopen):
            reopen()

    def build_managed_openrouter_byok_target_settings(self) -> Any | None:
        return self._backend.build_managed_openrouter_byok_target_settings()

    async def verify_api_key(self, provider: str, key: str) -> tuple[bool, str]:
        return await self._backend.verify_api_key(provider, key)

    def persist_api_key_verification(self, provider: str, key: str, success: bool) -> None:
        self._backend.persist_api_key_verification(provider, key, success)

    async def persist_provider_secret_change(self, key: str, value: str) -> bool:
        return bool(await self._backend.persist_provider_secret_change(key, value))

    def clear_provider_verification(self, provider: str) -> None:
        clear = getattr(self._backend, "clear_provider_verification", None)
        if callable(clear):
            clear(provider)

    async def start_qq_managed_auth_from_dialog(self, **kwargs: Any) -> object:
        return await self._backend.start_qq_managed_auth_from_dialog(**kwargs)

    async def start_discord_managed_auth_from_dialog(self, **kwargs: Any) -> object:
        return await self._backend.start_discord_managed_auth_from_dialog(**kwargs)

    def reopen_discord_managed_auth_browser(self) -> object:
        reopen = getattr(self._backend, "reopen_discord_managed_auth_browser", None)
        return reopen() if callable(reopen) else None

    def supports_discord_managed_auth_reopen(self) -> bool:
        return callable(getattr(self._backend, "reopen_discord_managed_auth_browser", None))

    def cancel_discord_managed_auth(self) -> object:
        cancel = getattr(self._backend, "cancel_discord_managed_auth", None)
        return cancel() if callable(cancel) else None

    def translation_enable_succeeded(self, result: object) -> bool:
        if result is False:
            return False
        state = self.state()
        hub = getattr(self._backend, "hub", None)
        if hub is not None:
            return bool(getattr(hub, "llm", None) is not None and state.translation_enabled)
        return result is True

    def clear_managed_auth_pending_state(self) -> None:
        self._backend.clear_managed_auth_pending_state()

    def get_event_language_codes(self) -> tuple[str | None, str | None]:
        return self._backend.get_event_language_codes()

    def schedule_github_star_prompt_translation_success_observed(self) -> None:
        self._backend.schedule_github_star_prompt_translation_success_observed()

    async def record_telemetry_translation_success_day(self) -> None:
        record = getattr(self._backend, "record_telemetry_translation_success_day", None)
        if callable(record):
            await record()

    def should_show_github_star_prompt(self) -> bool:
        should_show = getattr(self._backend, "should_show_github_star_prompt", None)
        return bool(should_show()) if callable(should_show) else False

    async def persist_github_star_prompt_eligible_launch(self) -> bool:
        persist = getattr(self._backend, "persist_github_star_prompt_eligible_launch", None)
        return bool(await persist()) if callable(persist) else False

    async def refresh_openrouter_usage_after_launch(self) -> bool:
        refresh = getattr(self._backend, "refresh_openrouter_usage_after_launch", None)
        return bool(await refresh()) if callable(refresh) else False

    async def prepare_runtime_after_launch(self) -> None:
        prepare = getattr(self._backend, "prepare_runtime_after_launch", None)
        if callable(prepare):
            await prepare()

    async def check_for_update(self) -> object | None:
        return await check_for_update()

    def start_github_star_prompt(
        self,
        run_prompt: Callable[[int], Awaitable[bool]],
    ) -> Awaitable[bool]:
        return self._github_star_prompt_runtime.start_launch_prompt(run_prompt)

    def is_current_github_star_prompt_generation(self, generation: int) -> bool:
        return self._github_star_prompt_runtime.is_current_generation(generation)

    def stop_github_star_prompt_ingress(self) -> None:
        self._github_star_prompt_runtime.stop_ingress()

    async def close_github_star_prompt_runtime(self) -> None:
        await self._github_star_prompt_runtime.close()

    def start_managed_auth_task(
        self,
        *,
        task_runner: Callable[[Callable[[], Awaitable[Any]]], object],
        task_factory: Callable[[], Awaitable[Any]],
        task_name: str,
        generation: int,
    ) -> object:
        return self._managed_auth_runtime.start_external_task(
            task_runner=task_runner,
            task_factory=task_factory,
            task_name=task_name,
            generation=generation,
        )

    def clear_managed_auth_task(self, task_name: str) -> None:
        self._managed_auth_runtime.clear_external_task(task_name)

    def cancel_managed_auth_task(
        self,
        handle: object | None,
        *,
        task_name: str,
    ) -> None:
        self._managed_auth_runtime.cancel_external_task(handle, task_name=task_name)

    def managed_auth_task_names(self) -> tuple[str, ...]:
        return self._managed_auth_runtime.external_task_names

    def managed_auth_tasks_open(self) -> bool:
        return not self._managed_auth_runtime.is_closed

    async def close_managed_auth_tasks(self) -> None:
        await self._managed_auth_runtime.close()

    def _github_star_prompt_runtime_diagnostics_sink(
        self,
        event: str,
        metadata: object,
    ) -> None:
        details = dict(metadata) if isinstance(metadata, dict) else {}
        self.log_detailed(
            f"[Lifecycle][GithubStarPromptRuntime] event={event} metadata={details}",
            level=logging.WARNING,
        )

    async def persist_github_star_prompt_opened(
        self,
        *,
        should_open: Callable[[], bool] | None = None,
    ) -> bool:
        persist = getattr(self._backend, "persist_github_star_prompt_opened", None)
        return bool(await persist(should_open=should_open)) if callable(persist) else False

    async def persist_github_star_prompt_clicked(self) -> None:
        persist = getattr(self._backend, "persist_github_star_prompt_clicked", None)
        if callable(persist):
            await persist()

    def cycle_debug_capture_fault_profile(self) -> str:
        return str(self._backend.cycle_debug_capture_fault_profile())

    def cycle_debug_stt_fault_profile(self) -> str:
        return str(self._backend.cycle_debug_stt_fault_profile())

    def clear_debug_audio_fault_profiles(self) -> None:
        self._backend.clear_debug_audio_fault_profiles()

    def handle_gpu_notice_action(self) -> object:
        return self._backend.handle_gpu_notice_action()


def _accepts_keyword(callable_obj: object, keyword: str) -> bool:
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return True
    return keyword in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


__all__ = ["UiApplicationBoundary"]
