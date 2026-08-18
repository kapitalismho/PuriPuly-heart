from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from puripuly_heart.app.language_selection import LanguageSelectionChange
from puripuly_heart.app.ports.application_runtime_logging import (
    ApplicationRuntimeLoggingPort,
)
from puripuly_heart.app.ports.application_runtime_shutdown import (
    ApplicationRuntimeShutdownPort,
)
from puripuly_heart.app.ports.ui_application import UiApplicationState
from puripuly_heart.app.ports.ui_application_intents import (
    UiDiagnosticsRuntimePort,
    UiEngagementRuntimePort,
    UiInputRuntimePort,
    UiManagedRuntimePort,
    UiMicrophoneRuntimePort,
    UiOverlayRuntimePort,
    UiPeerCaptureRuntimePort,
    UiProviderRuntimePort,
    UiSettingsRuntimePort,
)
from puripuly_heart.app.ports.ui_application_state import UiApplicationStatePort
from puripuly_heart.app.ports.ui_models import (
    GpuNoticeAction,
    ManagedGemmaNoticeAction,
    OverlayPeerPresentationState,
)
from puripuly_heart.app.services.application_runtime_shutdown import (
    compose_application_runtime_shutdown_callbacks,
)
from puripuly_heart.app.services.application_shutdown import (
    ApplicationShutdownCallback,
    ApplicationShutdownCoordinator,
    ApplicationShutdownDiagnostic,
    application_shutdown_callback,
)
from puripuly_heart.app.services.application_startup import ApplicationStartupOwner
from puripuly_heart.core.lifecycle import (
    SHUTDOWN_PHASE_FREEZE_INGRESS,
    SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS,
)
from puripuly_heart.core.runtime.github_star_prompt import GithubStarPromptRuntime
from puripuly_heart.core.runtime.oauth import OAuthRuntime
from puripuly_heart.core.updater import check_for_update

UI_APPLICATION_USER_INTENT_METHODS = frozenset(
    {
        "accept_peer_translation_eula_and_enable",
        "apply_loopback_capture_option",
        "apply_overlay_calibration",
        "apply_providers",
        "apply_settings",
        "apply_telemetry_consent",
        "begin_overlay_calibration",
        "cancel_discord_managed_auth",
        "cancel_overlay_calibration",
        "capture_settings_view_change",
        "check_for_update",
        "clear_debug_audio_fault_profiles",
        "clear_managed_auth_pending_state",
        "clear_provider_verification",
        "connect_openrouter_via_pkce",
        "cycle_debug_capture_fault_profile",
        "cycle_debug_stt_fault_profile",
        "ensure_gpu_device_discovery",
        "handle_gpu_notice_action",
        "handle_managed_gemma_notice_action",
        "install_selected_gpu_model_if_needed",
        "on_dashboard_language_change",
        "persist_api_key_verification",
        "persist_github_star_prompt_clicked",
        "persist_github_star_prompt_eligible_launch",
        "persist_github_star_prompt_opened",
        "persist_provider_secret_change",
        "prepare_runtime_after_launch",
        "record_telemetry_translation_success_day",
        "refresh_openrouter_usage_after_launch",
        "reopen_discord_managed_auth_browser",
        "reopen_openrouter_pkce_authorization_url",
        "reset_desktop_overlay_position",
        "retry_peer_process_capture",
        "schedule_github_star_prompt_translation_success_observed",
        "set_desktop_overlay_captions_locked",
        "set_desktop_overlay_size_preset",
        "set_manual_input_activity",
        "set_overlay_calibration_field",
        "set_overlay_enabled",
        "set_peer_translation_enabled",
        "set_runtime_logging_mode",
        "set_stt_enabled",
        "set_translation_enabled",
        "start",
        "start_discord_managed_auth_from_dialog",
        "start_github_star_prompt",
        "start_managed_auth_task",
        "start_microphone_test",
        "start_qq_managed_auth_from_dialog",
        "stop_microphone_test",
        "submit_text",
        "verify_api_key",
    }
)


def _guard_application_intent(method: Callable[..., Any]) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(method):

        @functools.wraps(method)
        async def invoke_async(self, *args: Any, **kwargs: Any) -> Any:
            self._admit_application_intent(method.__name__)
            return await method(self, *args, **kwargs)

        return invoke_async

    @functools.wraps(method)
    def invoke_sync(self, *args: Any, **kwargs: Any) -> Any:
        self._admit_application_intent(method.__name__)
        return method(self, *args, **kwargs)

    return invoke_sync


class UiApplicationBoundary:
    def __init__(
        self,
        *,
        startup: ApplicationStartupOwner,
        input_runtime: UiInputRuntimePort,
        peer_capture: UiPeerCaptureRuntimePort,
        settings: UiSettingsRuntimePort,
        provider: UiProviderRuntimePort,
        microphone: UiMicrophoneRuntimePort,
        overlay: UiOverlayRuntimePort,
        managed: UiManagedRuntimePort,
        engagement: UiEngagementRuntimePort,
        diagnostics: UiDiagnosticsRuntimePort,
        state: UiApplicationStatePort,
        runtime_shutdown: ApplicationRuntimeShutdownPort,
        runtime_logging: ApplicationRuntimeLoggingPort,
        osc_state_publisher: Callable[[], object] | None = None,
        http_extension_registry: object | None = None,
    ) -> None:
        self._startup = startup
        self._input_runtime = input_runtime
        self._peer_capture = peer_capture
        self._settings = settings
        self._provider = provider
        self._microphone = microphone
        self._overlay = overlay
        self._managed = managed
        self._engagement = engagement
        self._diagnostics = diagnostics
        self._runtime_logging = runtime_logging
        self._http_extension_registry = http_extension_registry
        self._runtime_shutdown = runtime_shutdown
        self._state_owner = state
        self._osc_state_publisher = osc_state_publisher
        self._github_star_prompt_runtime = GithubStarPromptRuntime(
            diagnostics_sink=self._github_star_prompt_runtime_diagnostics_sink,
        )
        self._managed_auth_runtime = OAuthRuntime()
        self._registered_application_shutdown_callbacks: list[ApplicationShutdownCallback] = []
        self._owned_application_shutdown_callbacks = (
            *self._boundary_application_shutdown_callbacks(),
            *compose_application_runtime_shutdown_callbacks(runtime_shutdown),
        )
        self._application_lifecycle: ApplicationShutdownCoordinator | None = None

    def _boundary_application_shutdown_callbacks(
        self,
    ) -> tuple[ApplicationShutdownCallback, ...]:
        return (
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_FREEZE_INGRESS,
                owner_name="GithubStarPromptRuntime",
                callback_name="stop_ingress",
                callback=self.stop_github_star_prompt_ingress,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS,
                owner_name="GithubStarPromptRuntime",
                callback_name="close",
                callback=self.close_github_star_prompt_runtime,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS,
                owner_name="OAuthRuntime",
                callback_name="close",
                callback=self.close_managed_auth_tasks,
            ),
        )

    def http_extension_registry(self) -> object | None:
        return self._http_extension_registry

    def state(self) -> UiApplicationState:
        return self._state_owner.snapshot()

    def effective_osc_ports(self) -> tuple[int | None, int | None]:
        return self._runtime_shutdown.effective_osc_ports()

    def compatibility_settings(self) -> Any | None:
        return self._state_owner.compatibility_settings()

    @property
    def overlay_calibration(self) -> object | None:
        return self._state_owner.overlay_calibration()

    async def start(self) -> None:
        try:
            await self._startup.start()
        except BaseException:
            try:
                await self.stop()
            except BaseException:
                pass
            raise

    async def stop(self) -> None:
        await self.application_lifecycle().shutdown()

    def application_lifecycle(self) -> ApplicationShutdownCoordinator:
        lifecycle = self._application_lifecycle
        if lifecycle is None:
            lifecycle = ApplicationShutdownCoordinator(
                (
                    *self._registered_application_shutdown_callbacks,
                    *self._owned_application_shutdown_callbacks,
                ),
                diagnostics_sink=self._runtime_shutdown.emit_application_shutdown_diagnostic,
            )
            self._application_lifecycle = lifecycle
        return lifecycle

    def register_application_shutdown_callbacks(
        self,
        callbacks: Sequence[ApplicationShutdownCallback],
    ) -> None:
        lifecycle = self._application_lifecycle
        if lifecycle is not None:
            lifecycle.register_callbacks(
                callbacks,
                before_existing=True,
            )
        else:
            self._registered_application_shutdown_callbacks.extend(callbacks)

    def _admit_application_intent(self, intent_name: str) -> None:
        self.application_lifecycle().admit_intent(intent_name)

    async def _publish_osc_state(self) -> None:
        callback = self._osc_state_publisher
        if callback is None:
            return
        result = callback()
        if inspect.isawaitable(result):
            await result

    def emit_application_shutdown_diagnostic(
        self,
        diagnostic: ApplicationShutdownDiagnostic,
    ) -> Awaitable[None] | None:
        return self._runtime_shutdown.emit_application_shutdown_diagnostic(diagnostic)

    def log_basic(self, message: str, *, level: int = logging.INFO) -> None:
        self._runtime_logging.emit_basic(message, level=level)

    def log_detailed(self, message: str, *, level: int = logging.INFO) -> None:
        self._runtime_logging.emit_detailed(message, level=level)

    async def submit_text(self, text: str) -> None:
        await self._input_runtime.submit_text(text)

    def set_manual_input_activity(self, has_text: bool) -> None:
        self._input_runtime.set_manual_input_activity(has_text)

    async def set_translation_enabled(self, enabled: bool) -> object:
        result = await self._input_runtime.set_translation_enabled(enabled)
        await self._publish_osc_state()
        return result

    async def set_stt_enabled(self, enabled: bool) -> object:
        result = await self._input_runtime.set_stt_enabled(enabled)
        await self._publish_osc_state()
        return result

    async def set_peer_translation_enabled(self, enabled: bool) -> object:
        result = await self._peer_capture.set_peer_translation_enabled(enabled)
        await self._publish_osc_state()
        return result

    async def set_overlay_enabled(self, enabled: bool) -> object:
        result = await self._overlay.set_overlay_enabled(enabled)
        await self._publish_osc_state()
        return result

    async def retry_peer_process_capture(self) -> bool:
        return bool(await self._peer_capture.retry_peer_process_capture())

    async def apply_loopback_capture_option(self, value: str) -> None:
        await self._peer_capture.apply_loopback_capture_option(value)

    def list_loopback_capture_options(self) -> object:
        return self._peer_capture.list_loopback_capture_options()

    def list_loopback_process_options(self) -> object:
        return self._peer_capture.list_loopback_process_options()

    def list_loopback_device_options(self) -> object:
        return self._peer_capture.list_loopback_device_options()

    def current_loopback_capture_option_value(self) -> object:
        return self._peer_capture.current_loopback_capture_option_value()

    def loopback_capture_summary(self) -> object:
        return self._peer_capture.loopback_capture_summary()

    async def on_dashboard_language_change(self, change: LanguageSelectionChange) -> None:
        await self._settings.on_dashboard_language_change(change)

    def capture_settings_view_change(self, settings: Any) -> object:
        return self._settings.capture_settings_view_change(settings)

    def merge_settings_view_change_with_current(self, captured: object) -> Any:
        return self._settings.merge_settings_view_change_with_current(captured)

    def refresh_settings_projection(
        self,
        *,
        preserve_custom_vocab_draft: bool = False,
    ) -> bool:
        return bool(
            self._settings.refresh_settings_projection(
                preserve_custom_vocab_draft=preserve_custom_vocab_draft,
            )
        )

    def refresh_settings_after_openrouter_pkce_success(self) -> bool:
        return bool(self._settings.refresh_settings_after_openrouter_pkce_success())

    def merge_settings_tab_apply_with_current_languages(self, settings: Any) -> Any:
        return self._settings.merge_settings_tab_apply_with_current_languages(settings)

    async def apply_settings(self, settings: Any) -> object:
        result = await self._settings.apply_settings(settings)
        await self._publish_osc_state()
        return result

    async def apply_providers(
        self,
        settings: Any | None = None,
        *,
        force_rebuild_llm: bool = False,
        persist_settings: bool = True,
        refresh_ui: bool = True,
    ) -> object:
        if not refresh_ui:
            result = await self._provider.apply_providers(
                settings,
                force_rebuild_llm=force_rebuild_llm,
                persist_settings=persist_settings,
                refresh_ui=False,
            )
        elif force_rebuild_llm:
            if settings is None:
                if persist_settings:
                    result = await self._provider.apply_providers(force_rebuild_llm=True)
                else:
                    result = await self._provider.apply_providers(
                        force_rebuild_llm=True,
                        persist_settings=False,
                    )
            else:
                if persist_settings:
                    result = await self._provider.apply_providers(
                        settings,
                        force_rebuild_llm=True,
                    )
                else:
                    result = await self._provider.apply_providers(
                        settings,
                        force_rebuild_llm=True,
                        persist_settings=False,
                    )
        elif settings is None:
            if persist_settings:
                result = await self._provider.apply_providers()
            else:
                result = await self._provider.apply_providers(persist_settings=False)
        else:
            if persist_settings:
                result = await self._provider.apply_providers(settings)
            else:
                result = await self._provider.apply_providers(
                    settings,
                    persist_settings=False,
                )
        await self._publish_osc_state()
        return result

    async def install_selected_gpu_model_if_needed(self) -> None:
        await self._provider.install_selected_gpu_model_if_needed()

    async def ensure_gpu_device_discovery(self) -> None:
        await self._provider.ensure_gpu_device_discovery()

    async def start_microphone_test(
        self,
        *,
        meter_callback: Callable[[float], None] | None = None,
    ) -> bool:
        start = self._microphone.start_microphone_test
        if meter_callback is not None and _accepts_keyword(start, "meter_callback"):
            result = start(meter_callback=meter_callback)
        else:
            result = start()
        return bool(await result if inspect.isawaitable(result) else result)

    async def stop_microphone_test(self) -> None:
        result = self._microphone.stop_microphone_test()
        if inspect.isawaitable(result):
            await result

    def set_runtime_logging_mode(self, mode: str) -> str:
        self._diagnostics.set_runtime_logging_mode(mode)
        return self.state().runtime_logging_mode

    async def set_desktop_overlay_captions_locked(self, locked: bool) -> None:
        await self._overlay.set_desktop_overlay_captions_locked(locked)

    async def set_desktop_overlay_size_preset(self, size_preset: str) -> None:
        await self._overlay.set_desktop_overlay_size_preset(size_preset)

    async def reset_desktop_overlay_position(self) -> None:
        await self._overlay.reset_desktop_overlay_position()

    def begin_overlay_calibration(self) -> object:
        return self._overlay.begin_overlay_calibration()

    def set_overlay_calibration_field(self, *args: Any, **kwargs: Any) -> object:
        return self._overlay.set_overlay_calibration_field(*args, **kwargs)

    def apply_overlay_calibration(self) -> object:
        return self._overlay.apply_overlay_calibration()

    def cancel_overlay_calibration(self) -> object:
        return self._overlay.cancel_overlay_calibration()

    def overlay_peer_presentation_state(self) -> OverlayPeerPresentationState | None:
        return self._peer_capture.overlay_peer_presentation_state()

    def dashboard_managed_auth_action(self) -> str:
        return str(self._managed.dashboard_managed_auth_action())

    def dashboard_managed_auth_prompt_kind(self) -> str:
        return str(self._managed.dashboard_managed_auth_prompt_kind())

    async def apply_telemetry_consent(self, consent: str) -> Any | None:
        return await self._settings.apply_telemetry_consent(consent)

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
            await self._provider.connect_openrouter_via_pkce(
                target_settings=target_settings,
                launch_source=launch_source,
            )
        )

    def reopen_openrouter_pkce_authorization_url(self) -> None:
        self._provider.reopen_openrouter_pkce_authorization_url()

    def build_managed_openrouter_byok_target_settings(self) -> Any | None:
        return self._provider.build_managed_openrouter_byok_target_settings()

    async def verify_api_key(self, provider: str, key: str) -> tuple[bool, str]:
        return await self._provider.verify_api_key(provider, key)

    def persist_api_key_verification(self, provider: str, key: str, success: bool) -> None:
        self._provider.persist_api_key_verification(provider, key, success)

    async def persist_provider_secret_change(self, key: str, value: str) -> bool:
        return bool(await self._provider.persist_provider_secret_change(key, value))

    def clear_provider_verification(self, provider: str) -> None:
        self._provider.clear_provider_verification(provider)

    async def start_qq_managed_auth_from_dialog(self, **kwargs: Any) -> object:
        return await self._managed.start_qq_managed_auth_from_dialog(**kwargs)

    async def start_discord_managed_auth_from_dialog(self, **kwargs: Any) -> object:
        return await self._managed.start_discord_managed_auth_from_dialog(**kwargs)

    def reopen_discord_managed_auth_browser(self) -> object:
        return None

    def supports_discord_managed_auth_reopen(self) -> bool:
        return False

    def cancel_discord_managed_auth(self) -> object:
        return None

    def translation_enable_succeeded(self, result: object) -> bool:
        if result is False:
            return False
        state = self.state()
        if state.translation_runtime_ready is not None:
            return bool(state.translation_runtime_ready and state.translation_enabled)
        return result is True

    def clear_managed_auth_pending_state(self) -> None:
        self._managed.clear_managed_auth_pending_state()

    def get_event_language_codes(self) -> tuple[str | None, str | None]:
        return self._engagement.get_event_language_codes()

    def schedule_github_star_prompt_translation_success_observed(self) -> None:
        self._engagement.schedule_github_star_prompt_translation_success_observed()

    async def record_telemetry_translation_success_day(self) -> None:
        await self._engagement.record_telemetry_translation_success_day()

    def should_show_github_star_prompt(self) -> bool:
        return bool(self._engagement.should_show_github_star_prompt())

    async def persist_github_star_prompt_eligible_launch(self) -> bool:
        return bool(await self._engagement.persist_github_star_prompt_eligible_launch())

    async def refresh_openrouter_usage_after_launch(self) -> bool:
        return bool(await self._managed.refresh_openrouter_usage_after_launch())

    async def prepare_runtime_after_launch(self) -> None:
        await self._engagement.prepare_runtime_after_launch()

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
        return bool(
            await self._engagement.persist_github_star_prompt_opened(
                should_open=should_open,
            )
        )

    async def persist_github_star_prompt_clicked(self) -> None:
        await self._engagement.persist_github_star_prompt_clicked()

    def cycle_debug_capture_fault_profile(self) -> str:
        return str(self._diagnostics.cycle_debug_capture_fault_profile())

    def cycle_debug_stt_fault_profile(self) -> str:
        return str(self._diagnostics.cycle_debug_stt_fault_profile())

    def clear_debug_audio_fault_profiles(self) -> None:
        self._diagnostics.clear_debug_audio_fault_profiles()

    def handle_gpu_notice_action(self, action: GpuNoticeAction) -> object:
        return self._provider.handle_gpu_notice_action(action)

    async def handle_managed_gemma_notice_action(
        self,
        action: ManagedGemmaNoticeAction,
    ) -> object:
        return await self._provider.handle_managed_gemma_notice_action(action)


for _intent_method_name in UI_APPLICATION_USER_INTENT_METHODS:
    setattr(
        UiApplicationBoundary,
        _intent_method_name,
        _guard_application_intent(getattr(UiApplicationBoundary, _intent_method_name)),
    )


def _accepts_keyword(callable_obj: object, keyword: str) -> bool:
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return True
    return keyword in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


__all__ = ["UI_APPLICATION_USER_INTENT_METHODS", "UiApplicationBoundary"]
