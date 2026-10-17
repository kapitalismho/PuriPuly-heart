from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from puripuly_heart.app.language_selection import LanguageSelectionChange
from puripuly_heart.app.ports.application_control import ApplicationControl
from puripuly_heart.app.ports.application_runtime_logging import (
    ApplicationRuntimeLoggingPort,
)
from puripuly_heart.app.ports.application_runtime_shutdown import (
    ApplicationRuntimeShutdownPort,
)
from puripuly_heart.app.ports.application_startup import ApplicationStartupDiagnostic
from puripuly_heart.app.ports.chatgpt_account import (
    ChatGptAccountSnapshot,
    ChatGptConnectResult,
    ChatGptSignOutResult,
)
from puripuly_heart.app.ports.settings_secrets import SettingsSecretsPort
from puripuly_heart.app.ports.settings_view import (
    AlibabaConnectionApplyResult,
    AlibabaConnectionDraftSnapshot,
    GeneralSettingsSnapshot,
    ImmediateSettingsIntent,
    OpenRouterPkceTarget,
    OverlaySettingsSnapshot,
    PromptApplyIntent,
    ProviderApplyIntent,
)
from puripuly_heart.app.ports.ui_application import (
    TranslationEnablePreparation,
    UiApplicationState,
)
from puripuly_heart.app.ports.ui_application_intents import (
    UiAudioTestRuntimePort,
    UiDiagnosticsRuntimePort,
    UiEngagementRuntimePort,
    UiInputRuntimePort,
    UiManagedRuntimePort,
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
    ApplicationShutdownStallDiagnostic,
    application_shutdown_callback,
)
from puripuly_heart.app.services.application_startup import ApplicationStartupOwner
from puripuly_heart.app.services.index_selection import IndexSelectionOwner
from puripuly_heart.config.alibaba_connection import AlibabaEndpointMode, AlibabaRegion
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.config.translation_values import provider_llm_for_translation
from puripuly_heart.core.diagnostics import emit_failure, safe_exception_fields
from puripuly_heart.core.lifecycle import (
    SHUTDOWN_PHASE_FREEZE_INGRESS,
    SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS,
)
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.orchestrator.self_translation_channel import TextSubmissionAdmission
from puripuly_heart.core.runtime.github_star_prompt import GithubStarPromptRuntime
from puripuly_heart.core.runtime.oauth import OAuthRuntime
from puripuly_heart.core.updater import check_for_update

UI_APPLICATION_USER_INTENT_METHODS = frozenset(
    {
        "accept_peer_translation_eula_and_enable",
        "apply_loopback_capture_option",
        "apply_overlay_calibration",
        "apply_providers",
        "apply_alibaba_connection_draft",
        "apply_settings",
        "apply_settings_intent",
        "apply_prompt_intent",
        "apply_provider_intent",
        "apply_telemetry_enabled",
        "begin_overlay_calibration",
        "begin_alibaba_connection_draft",
        "cancel_alibaba_connection_draft",
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
        "download_model_asset",
        "import_model_asset",
        "cancel_model_asset",
        "delete_model_asset",
        "refresh_model_assets",
        "edit_alibaba_connection_draft",
        "on_dashboard_language_change",
        "persist_api_key_verification",
        "persist_github_star_prompt_clicked",
        "persist_github_star_prompt_eligible_launch",
        "persist_github_star_prompt_opened",
        "persist_provider_secret_change",
        "prepare_runtime_after_launch",
        "record_app_active_day",
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
        "set_stt_enabled",
        "set_translation_enabled",
        "start",
        "start_discord_managed_auth_from_dialog",
        "start_github_star_prompt",
        "start_managed_auth_task",
        "start_audio_test",
        "start_qq_managed_auth_from_dialog",
        "stop_audio_test",
        "submit_text",
        "verify_alibaba_connection_draft",
        "verify_api_key",
    }
)


_ORDERED_INTENTS = frozenset(
    {
        "apply_loopback_capture_option",
        "apply_providers",
        "apply_settings",
        "apply_alibaba_connection_draft",
        "apply_settings_intent",
        "apply_prompt_intent",
        "edit_alibaba_connection_draft",
        "apply_provider_intent",
        "apply_telemetry_enabled",
        "on_dashboard_language_change",
        "set_stt_enabled",
        "set_peer_translation_enabled",
        "set_translation_enabled",
        "set_overlay_enabled",
        "persist_provider_secret_change",
        "start_audio_test",
    }
)


def _guard_application_intent(method: Callable[..., Any]) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(method):

        @functools.wraps(method)
        async def invoke_async(self, *args: Any, **kwargs: Any) -> Any:
            self._admit_application_intent(method.__name__)
            if method.__name__ == "start_audio_test" and kwargs.get("admission_revision") is None:
                kwargs["admission_revision"] = self.audio_test_start_revision()
            control = self._control
            if method.__name__ == "set_translation_enabled":
                preparation = kwargs.get("preparation")
                if preparation is not None:
                    if (
                        preparation.intent_revision != self._translation_selection_intent
                        or not preparation.ready
                        or self.translation_preparation_needs_refresh(preparation)
                    ):
                        return False
                self._translation_selection_intent += 1
                intent = self._translation_selection_intent
                enabled = args[0] if args else kwargs.get("enabled")
                if (
                    enabled
                    and preparation is None
                    and self.translation_preparation_needs_refresh(None)
                ):
                    if control is not None and control.lock_owned_by_current_task():
                        return False
                    while True:
                        preparation = await self.prepare_translation_enable(intent)
                        if (
                            preparation is None
                            or not preparation.ready
                            or intent != self._translation_selection_intent
                        ):
                            return False
                        self._admit_application_intent(method.__name__)
                        if control is None:
                            return await method(self, *args, **kwargs)
                        async with control._resource_lock:
                            async with control._lock:
                                if intent != self._translation_selection_intent:
                                    return False
                                if not self.translation_preparation_needs_refresh(preparation):
                                    return await method(self, *args, **kwargs)
            if (
                method.__name__ not in _ORDERED_INTENTS
                or control is None
                or control.lock_owned_by_current_task()
            ):
                return await method(self, *args, **kwargs)
            capture_off = (
                method.__name__ in {"set_stt_enabled", "set_peer_translation_enabled"}
                and (args[0] if args else kwargs.get("enabled")) is False
            )
            if not capture_off:
                async with control._resource_lock:
                    async with control._lock:
                        return await method(self, *args, **kwargs)
            async with control._lock:
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
        audio_test: UiAudioTestRuntimePort,
        overlay: UiOverlayRuntimePort,
        managed: UiManagedRuntimePort,
        engagement: UiEngagementRuntimePort,
        diagnostics: UiDiagnosticsRuntimePort,
        state: UiApplicationStatePort,
        runtime_shutdown: ApplicationRuntimeShutdownPort,
        runtime_logging: ApplicationRuntimeLoggingPort,
        settings_secrets: SettingsSecretsPort,
        osc_state_publisher: Callable[[], object] | None = None,
        output_status_provider: Callable[[], dict[str, dict[str, object]]] | None = None,
        http_extension_registry: object | None = None,
        scene_observation_provider: Callable[[], dict[str, object] | None] | None = None,
        index_selection: IndexSelectionOwner | None = None,
    ) -> None:
        self._startup = startup
        self._input_runtime = input_runtime
        self._peer_capture = peer_capture
        self._settings = settings
        self._provider = provider
        self._audio_test = audio_test
        self._overlay = overlay
        self._managed = managed
        self._engagement = engagement
        self._diagnostics = diagnostics
        self._runtime_logging = runtime_logging
        self._settings_secrets = settings_secrets
        self._http_extension_registry = http_extension_registry
        self._runtime_shutdown = runtime_shutdown
        self._state_owner = state
        self._osc_state_publisher = osc_state_publisher
        self._output_status_provider = output_status_provider
        self._scene_observation_provider = scene_observation_provider
        self._index_selection = index_selection
        self._translation_selection_intent = 0
        self._audio_test_start_revision = 0
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
        self._control: ApplicationControl | None = None
        self._presentation_close: Callable[[], Awaitable[None]] | None = None

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

    @property
    def diagnostic_instance_id(self) -> str | None:
        return self.control().instance_id

    def control(self) -> ApplicationControl:
        if self._control is None:
            raise RuntimeError("application control is not composed")
        return self._control

    def attach_control(self, control: ApplicationControl) -> None:
        self._control = control

    def bind_presentation_close(self, callback: Callable[[], Awaitable[None]]) -> None:
        if self._presentation_close is not None:
            raise RuntimeError("presentation close is already bound")
        self._presentation_close = callback

    async def close_presentation(self) -> None:
        callback = self._presentation_close
        if callback is None:
            raise RuntimeError("GUI presentation close is unavailable")
        await callback()

    def http_extension_registry(self) -> object | None:
        return self._http_extension_registry

    def settings_secrets(self) -> SettingsSecretsPort:
        return self._settings_secrets

    def state(self) -> UiApplicationState:
        return self._state_owner.snapshot()

    def effective_osc_ports(self) -> tuple[int | None, int | None]:
        return self._runtime_shutdown.effective_osc_ports()

    def output_status(self) -> dict[str, dict[str, object]]:
        if self._output_status_provider is None:
            raise RuntimeError("application output owners are not composed")
        return self._output_status_provider()

    def scene_observation(self) -> dict[str, object]:
        snapshot = (
            self._scene_observation_provider()
            if self._scene_observation_provider is not None
            else None
        )
        return {
            "observation_succeeded": (
                snapshot.get("process_observation_succeeded") if snapshot else None
            ),
            "process_present": snapshot.get("process_present") if snapshot else None,
            "observation_diagnostic_id": (
                snapshot.get("observation_diagnostic_id") if snapshot else None
            ),
        }

    async def wait_overlay_transition(self) -> dict[str, object]:
        return await self._overlay.wait_overlay_transition()

    def compatibility_settings(self) -> AppSettingsVNext | None:
        return self._state_owner.compatibility_settings()

    @property
    def overlay_calibration(self) -> object | None:
        return self._state_owner.overlay_calibration()

    async def start(self) -> None:
        attempt_id = f"startup-{uuid4().hex}"
        process_id = os.getpid()
        self._emit_startup_diagnostic(
            ApplicationStartupDiagnostic(
                outcome="entered",
                attempt_id=attempt_id,
                process_id=process_id,
                monotonic_ns=time.monotonic_ns(),
            )
        )
        try:
            await self._startup.start()
        except BaseException as exc:
            cancelled = isinstance(exc, asyncio.CancelledError)
            primary = getattr(exc, "diagnostics", None)
            if not isinstance(primary, ErrorDiagnostics):
                phase = getattr(self._startup, "phase", "startup_prepare")
                primary = ErrorDiagnostics(
                    component="application_startup",
                    operation=phase,
                    code="startup_prepare_failed",
                    category="lifecycle",
                    visibility="basic",
                    content_policy="metadata_only",
                    status_code=None,
                    retry_after_ms=None,
                    fields={"phase": phase, **safe_exception_fields(exc)},
                )
            self._emit_startup_diagnostic(
                ApplicationStartupDiagnostic(
                    outcome="cancelled" if cancelled else "failed",
                    attempt_id=attempt_id,
                    process_id=process_id,
                    monotonic_ns=time.monotonic_ns(),
                    exception_class=None if cancelled else type(exc).__name__,
                    diagnostics=primary,
                )
            )
            self.application_lifecycle().set_primary_failure(primary.diagnostic_id)
            try:
                await self.stop()
            except BaseException:
                pass
            raise
        self._emit_startup_diagnostic(
            ApplicationStartupDiagnostic(
                outcome="completed",
                attempt_id=attempt_id,
                process_id=process_id,
                monotonic_ns=time.monotonic_ns(),
            )
        )

    def _emit_startup_diagnostic(self, diagnostic: ApplicationStartupDiagnostic) -> None:
        try:
            self._runtime_logging.emit_startup_diagnostic(diagnostic)
        except Exception:
            pass

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
                runtime_state_supplier=(self._runtime_shutdown.application_shutdown_runtime_states),
            )
            self._application_lifecycle = lifecycle
        return lifecycle

    def capture_application_shutdown_stall_diagnostic(
        self,
    ) -> ApplicationShutdownStallDiagnostic:
        return self.application_lifecycle().capture_stall_diagnostic()

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
        try:
            result = callback()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            emit_failure(
                logging.getLogger(__name__),
                ErrorDiagnostics(
                    component="application_notification",
                    operation="osc_state_notification",
                    code="application_state_notification_failed",
                    category="lifecycle",
                    visibility="basic",
                    content_policy="metadata_only",
                    status_code=None,
                    retry_after_ms=None,
                    fields={"phase": "osc_state_notification", **safe_exception_fields(exc)},
                ),
                severity="warning",
                disposition="degraded",
            )

    def emit_application_shutdown_diagnostic(
        self,
        diagnostic: ApplicationShutdownDiagnostic,
    ) -> Awaitable[None] | None:
        return self._runtime_shutdown.emit_application_shutdown_diagnostic(diagnostic)

    def log_basic(self, message: str, *, level: int = logging.INFO) -> None:
        self._runtime_logging.emit_basic(message, level=level)

    def log_diagnostic(self, message: str, *, level: int = logging.INFO) -> None:
        self._runtime_logging.emit_diagnostic(message, level=level)

    async def submit_text(self, text: str) -> TextSubmissionAdmission | None:
        return await self._input_runtime.submit_text(text)

    def set_manual_input_activity(self, has_text: bool) -> None:
        self._input_runtime.set_manual_input_activity(has_text)

    def _translation_preparation_required(self) -> bool:
        selection = self._index_selection
        if selection is None:
            return False
        current = self.compatibility_settings()
        return bool(
            (current is not None and current.intent.translation.model == "index_translate_35b")
            or not selection.ready_for_enable
        )

    async def prepare_translation_enable(
        self, intent_revision: int | None = None
    ) -> TranslationEnablePreparation | None:
        required = self._translation_preparation_required()
        if intent_revision is None:
            if not required:
                return None
            intent_revision = self._translation_selection_intent
        selection = self._index_selection
        ready = intent_revision == self._translation_selection_intent
        if ready and required and selection is not None:
            ready = await selection.prepare_enable()
        return TranslationEnablePreparation(
            intent_revision=intent_revision,
            selection_generation=selection.generation if selection is not None else None,
            ready=ready,
        )

    def translation_preparation_needs_refresh(
        self, preparation: TranslationEnablePreparation | None
    ) -> bool:
        if preparation is None:
            return self._translation_preparation_required()
        if preparation.intent_revision != self._translation_selection_intent:
            return False
        selection = self._index_selection
        generation = selection.generation if selection is not None else None
        return bool(
            preparation.selection_generation != generation
            or (preparation.ready and selection is not None and not selection.ready_for_enable)
        )

    async def set_translation_enabled(
        self,
        enabled: bool,
        *,
        allow_authorization: bool = True,
        preparation: TranslationEnablePreparation | None = None,
    ) -> object:
        result = await self._input_runtime.set_translation_enabled(
            enabled,
            allow_authorization=allow_authorization,
        )
        await self._publish_osc_state()
        return result

    async def set_stt_enabled(self, enabled: bool) -> object:
        await self._yield_audio_test_capture("self", enabled)
        result = await self._input_runtime.set_stt_enabled(enabled)
        await self._publish_osc_state()
        return result

    async def set_peer_translation_enabled(self, enabled: bool) -> object:
        await self._yield_audio_test_capture("peer", enabled)
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

    def capture_settings_view_change(self, settings: AppSettingsVNext) -> object:
        return self._settings.capture_settings_view_change(settings)

    def merge_settings_view_change_with_current(self, captured: object) -> AppSettingsVNext:
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

    def merge_settings_tab_apply_with_current_languages(
        self, settings: AppSettingsVNext
    ) -> AppSettingsVNext:
        return self._settings.merge_settings_tab_apply_with_current_languages(settings)

    async def apply_settings(
        self,
        settings: AppSettingsVNext,
        *,
        provider_requested: bool = False,
    ) -> object:
        result = await self._settings.apply_settings(
            settings, provider_requested=provider_requested
        )
        await self._publish_osc_state()
        return result

    async def apply_providers(
        self,
        settings: AppSettingsVNext | None = None,
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

    def audio_test_start_revision(self) -> int:
        return self._audio_test_start_revision

    async def start_audio_test(
        self,
        *,
        observation_callback: Callable[[str, tuple[object, ...]], object] | None = None,
        admission_revision: int | None = None,
    ) -> bool:
        if admission_revision != self._audio_test_start_revision:
            return False
        return bool(
            await self._audio_test.start_audio_test(observation_callback=observation_callback)
        )

    def audio_test_snapshot(self) -> dict[str, object]:
        return self._audio_test.audio_test_snapshot()

    async def wait_audio_test_ready(self) -> bool:
        return await self._audio_test.wait_audio_test_ready()

    async def stop_audio_test(self) -> None:
        self._audio_test_start_revision += 1
        await self._audio_test.stop_audio_test()

    async def _yield_audio_test_capture(self, channel: str, enabled: bool) -> None:
        self._audio_test.release_audio_test_capture_restore(channel)
        if enabled:
            self._audio_test_start_revision += 1
        if enabled and self.state().audio_test_active:
            await self._audio_test.stop_audio_test()
            if self.state().audio_test_active:
                raise RuntimeError("Audio test resources are still being released")

    async def set_desktop_overlay_captions_locked(self, locked: bool) -> dict[str, str]:
        return await self._overlay.set_desktop_overlay_captions_locked(locked)

    async def set_desktop_overlay_size_preset(self, size_preset: str) -> dict[str, str]:
        return await self._overlay.set_desktop_overlay_size_preset(size_preset)

    async def reset_desktop_overlay_position(self) -> dict[str, str]:
        return await self._overlay.reset_desktop_overlay_position()

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

    def managed_auth_last_failure_kind(self) -> str:
        accessor = getattr(self._managed, "managed_auth_last_failure_kind", None)
        if not callable(accessor):
            return "failed"
        kind = accessor()
        return kind if isinstance(kind, str) else "failed"

    async def apply_telemetry_enabled(self, enabled: bool) -> Any | None:
        return await self._settings.apply_telemetry_enabled(enabled)

    async def apply_settings_intent(self, intent: ImmediateSettingsIntent) -> object:
        result = await self._settings.apply_settings_intent(intent)
        await self._publish_osc_state()
        return result

    async def apply_prompt_intent(self, intent: PromptApplyIntent) -> object:
        result = await self._settings.apply_prompt_intent(intent)
        await self._publish_osc_state()
        return result

    async def apply_provider_intent(self, intent: ProviderApplyIntent) -> object:
        settings = self._settings.materialize_provider_intent(intent)
        self.log_basic(
            "[Settings] provider_apply_requested "
            f"self={settings.intent.stt.provider} peer={settings.intent.peer_stt.provider}"
        )
        result = await self._provider.apply_providers(settings)
        await self._publish_osc_state()
        return result

    def settings_overlay_snapshot(self) -> OverlaySettingsSnapshot | None:
        return self._settings.overlay_snapshot()

    def settings_general_snapshot(self) -> GeneralSettingsSnapshot | None:
        return self._settings.general_snapshot()

    async def accept_peer_translation_eula_and_enable(self) -> object:
        settings = self.compatibility_settings()
        if settings is not None:
            from dataclasses import replace

            settings = replace(
                settings,
                state=replace(
                    settings.state,
                    peer_translation=replace(
                        settings.state.peer_translation,
                        eula_accepted=True,
                    ),
                ),
            )
            await self.apply_settings(settings)
        return await self.set_peer_translation_enabled(True)

    def local_llm_selected(self) -> bool:
        return self.state().provider_name == "local_llm"

    async def connect_openrouter_via_pkce(
        self,
        *,
        target: OpenRouterPkceTarget,
        launch_source: str,
        open_browser: bool = True,
        authorization_url_sink: Callable[[str], None] | None = None,
    ) -> bool:
        return bool(
            await self._provider.connect_openrouter_via_pkce(
                target=target,
                launch_source=launch_source,
                open_browser=open_browser,
                authorization_url_sink=authorization_url_sink,
            )
        )

    def reopen_openrouter_pkce_authorization_url(self) -> None:
        self._provider.reopen_openrouter_pkce_authorization_url()

    def chatgpt_account_snapshot(self) -> ChatGptAccountSnapshot:
        return self._provider.chatgpt_account_snapshot()

    async def connect_chatgpt(
        self,
        *,
        open_browser: bool = True,
        authorization_url_sink: Callable[[str], None] | None = None,
    ) -> ChatGptConnectResult:
        result = await self._provider.connect_chatgpt(
            open_browser=open_browser,
            authorization_url_sink=authorization_url_sink,
        )
        await self._publish_osc_state()
        return result

    def reopen_chatgpt_authorization_url(self) -> bool:
        return bool(self._provider.reopen_chatgpt_authorization_url())

    def cancel_chatgpt_sign_in(self) -> None:
        self._provider.cancel_chatgpt_sign_in()

    def chatgpt_sign_in_required(self) -> bool:
        return bool(self._provider.chatgpt_sign_in_required())

    async def select_chatgpt_translation(self) -> object:
        from dataclasses import replace

        from puripuly_heart.app.ports.settings_view import TranslationSelectionEdit
        from puripuly_heart.app.services.settings.settings_application import (
            settings_view_surface_snapshots,
        )
        from puripuly_heart.config.translation_values import (
            TranslationConnection,
            TranslationModel,
        )

        settings = self.compatibility_settings()
        if settings is None:
            return None
        selection = settings_view_surface_snapshots(settings)[0].translation
        model = TranslationModel.GPT_6_LUNA
        connection = TranslationConnection.CHATGPT
        if selection.model == model and selection.connection == connection:
            return True
        history = dict(selection.connection_history)
        history[model] = connection
        edit = TranslationSelectionEdit(
            replace(
                selection,
                model=model,
                connection=connection,
                connection_history=tuple(
                    (candidate, history[candidate])
                    for candidate in TranslationModel
                    if candidate in history
                ),
                previous_llm_model=None,
            ),
            ((model, connection),),
        )
        return await self.apply_provider_intent(ProviderApplyIntent((edit,)))

    async def sign_out_chatgpt(self) -> ChatGptSignOutResult:
        settings = self.compatibility_settings()
        if settings is not None and self.state().translation_enabled:
            translation = settings.intent.translation
            if provider_llm_for_translation(translation.model, translation.connection) == "chatgpt":
                await self.set_translation_enabled(False)
        result = await self._provider.sign_out_chatgpt()
        await self._publish_osc_state()
        return result

    def build_managed_openrouter_byok_target(self) -> OpenRouterPkceTarget | None:
        return self._provider.build_managed_openrouter_byok_target()

    async def begin_alibaba_connection_draft(self) -> AlibabaConnectionDraftSnapshot:
        return await self._provider.begin_alibaba_connection_draft()

    async def alibaba_connection_draft(self) -> AlibabaConnectionDraftSnapshot:
        return await self._provider.alibaba_connection_draft()

    async def alibaba_active_connection(self) -> AlibabaConnectionDraftSnapshot:
        return await self._provider.alibaba_active_connection()

    async def edit_alibaba_connection_draft(
        self,
        *,
        token: str,
        region: AlibabaRegion | None = None,
        endpoint_mode: AlibabaEndpointMode | None = None,
        api_host: str | None = None,
    ) -> AlibabaConnectionDraftSnapshot:
        return await self._provider.edit_alibaba_connection_draft(
            token=token,
            region=region,
            endpoint_mode=endpoint_mode,
            api_host=api_host,
        )

    async def verify_alibaba_connection_draft(
        self,
        *,
        token: str,
        capability: Literal["asr", "translation", "both"],
        api_key: str | None = None,
    ) -> AlibabaConnectionDraftSnapshot:
        return await self._provider.verify_alibaba_connection_draft(
            token=token,
            capability=capability,
            api_key=api_key,
        )

    async def apply_alibaba_connection_draft(self, *, token: str) -> AlibabaConnectionApplyResult:
        result = await self._provider.apply_alibaba_connection_draft(token=token)
        await self._publish_osc_state()
        return result

    def cancel_alibaba_connection_draft(self, *, token: str) -> None:
        self._provider.cancel_alibaba_connection_draft(token=token)

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

    async def logout_local_managed(self, provider: str) -> dict[str, object]:
        return await self._managed.logout_local_managed(provider)

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

    async def record_app_active_day(self, active_date_utc: str) -> object:
        return await self._engagement.record_app_active_day(active_date_utc)

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
        self.log_diagnostic(
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

    def model_asset_entries(self) -> tuple[object, ...]:
        return self._provider.model_asset_entries()

    async def refresh_model_assets(self) -> tuple[object, ...]:
        return await self._provider.refresh_model_assets()

    async def download_model_asset(self, model_id: str, *, source: str | None = None) -> object:
        return await self._provider.download_model_asset(model_id, source=source)

    async def import_model_asset(self, model_id: str, directory: Path) -> object:
        return await self._provider.import_model_asset(model_id, directory)

    async def cancel_model_asset(self, model_id: str) -> bool:
        return await self._provider.cancel_model_asset(model_id)

    async def delete_model_asset(self, model_id: str) -> object:
        return await self._provider.delete_model_asset(model_id)


for _intent_method_name in UI_APPLICATION_USER_INTENT_METHODS:
    setattr(
        UiApplicationBoundary,
        _intent_method_name,
        _guard_application_intent(getattr(UiApplicationBoundary, _intent_method_name)),
    )


__all__ = ["UI_APPLICATION_USER_INTENT_METHODS", "UiApplicationBoundary"]
