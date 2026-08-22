from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from puripuly_heart.app.language_selection import LanguageSelectionChange
from puripuly_heart.app.ports.ui_models import (
    GpuDeviceOption,
    GpuNoticeAction,
    ManagedGemmaNoticeAction,
    OverlayPeerPresentationState,
)
from puripuly_heart.app.services.application_after_launch import (
    ApplicationAfterLaunchOwner,
)
from puripuly_heart.app.services.application_runtime_logging import (
    ApplicationRuntimeLoggingOwner,
)
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.services.desktop_overlay_application import (
    DesktopOverlayApplicationOwner,
)
from puripuly_heart.app.services.github_star_prompt import GithubStarPromptOwner
from puripuly_heart.app.services.gpu_runtime_interaction import GpuRuntimeInteractionOwner
from puripuly_heart.app.services.managed_gemma_translation import (
    ManagedGemmaTranslationOwner,
)
from puripuly_heart.app.services.manual_typing import ManualTypingOwner
from puripuly_heart.app.services.overlay_application import OverlayApplicationOwner
from puripuly_heart.app.services.overlay_calibration_application import (
    OverlayCalibrationApplicationOwner,
)
from puripuly_heart.app.services.provider_credential_verification import (
    ProviderCredentialVerificationInteractionOwner,
)
from puripuly_heart.app.services.provider_settings import (
    ProviderApplicationOwner,
    ProviderSettingsOwner,
)
from puripuly_heart.app.services.self_capture_application import (
    SelfCaptureApplicationOwner,
)
from puripuly_heart.app.services.settings_application import SettingsApplicationOwner
from puripuly_heart.app.services.settings_projection import (
    SettingsProjectionOwner,
)
from puripuly_heart.app.services.translation_enable import TranslationEnableOwner
from puripuly_heart.app.wiring_managed_account import ManagedAccountComponents
from puripuly_heart.app.wiring_microphone_test import MicrophoneTestRuntime
from puripuly_heart.app.wiring_peer_application import PeerApplicationRuntime
from puripuly_heart.app.wiring_runtime_pipeline import RuntimePipelineHandle
from puripuly_heart.core.http_extensions import (
    http_extension_secret_key_prefix,
)
from puripuly_heart.core.local_translation.devices import list_llama_vulkan_devices
from puripuly_heart.core.telemetry import (
    TranslationSuccessTelemetryResult,
    TranslationSuccessTelemetryService,
)


def _llama_gpu_device_options() -> tuple[GpuDeviceOption, ...]:
    return tuple(
        GpuDeviceOption(
            device_id=device.device_id,
            display_name=device.display_name,
            backend_name=device.device_id,
        )
        for device in list_llama_vulkan_devices()
    )


@dataclass(slots=True)
class UiInputRuntimeAdapter:
    pipeline: RuntimePipelineHandle
    manual_typing: ManualTypingOwner
    translation: TranslationEnableOwner
    self_capture: SelfCaptureApplicationOwner

    async def submit_text(self, text: str) -> None:
        owner = self.pipeline.self_translation_channel
        submit = None if owner is None else lambda: owner.submit_text(text, source="You")
        await self.manual_typing.submit(submit)

    def set_manual_input_activity(self, has_text: bool) -> None:
        self.manual_typing.set_input_activity(has_text)

    async def set_translation_enabled(self, enabled: bool) -> object:
        return await self.translation.set_enabled(enabled)

    async def set_stt_enabled(self, enabled: bool) -> object:
        return await self.self_capture.set_enabled(enabled)


@dataclass(slots=True)
class UiPeerCaptureRuntimeAdapter:
    peer: PeerApplicationRuntime
    overlay: OverlayApplicationOwner

    async def set_peer_translation_enabled(self, enabled: bool) -> object:
        return await self.peer.owner.set_enabled(enabled)

    async def retry_peer_process_capture(self) -> bool:
        return await self.peer.owner.retry_process_capture()

    async def apply_loopback_capture_option(self, value: str) -> None:
        await self.peer.target.apply(value)

    def list_loopback_capture_options(self) -> object:
        return self.peer.target.options()

    def list_loopback_process_options(self) -> object:
        return self.peer.target.process_options()

    def list_loopback_device_options(self) -> object:
        return self.peer.target.device_options()

    def current_loopback_capture_option_value(self) -> object:
        return self.peer.target.current_value()

    def loopback_capture_summary(self) -> object:
        return self.peer.target.summary()

    def overlay_peer_presentation_state(self) -> OverlayPeerPresentationState | None:
        return self.overlay.presentation_state()


@dataclass(slots=True)
class UiMicrophoneRuntimeAdapter:
    microphone: MicrophoneTestRuntime
    level_log_interval_seconds: float

    async def start_microphone_test(
        self,
        *,
        meter_callback=None,
    ) -> bool:
        return await self.microphone.start(
            meter_callback=meter_callback,
            level_log_interval_s=self.level_log_interval_seconds,
        )

    async def stop_microphone_test(self) -> None:
        await self.microphone.stop()


@dataclass(slots=True)
class UiOverlayRuntimeAdapter:
    overlay: OverlayApplicationOwner
    desktop: DesktopOverlayApplicationOwner
    calibration: OverlayCalibrationApplicationOwner

    async def set_overlay_enabled(self, enabled: bool) -> object:
        return await self.overlay.set_enabled(enabled)

    async def set_desktop_overlay_captions_locked(self, locked: bool) -> None:
        await self.desktop.set_captions_locked(locked)

    async def set_desktop_overlay_size_preset(self, size_preset: str) -> None:
        await self.desktop.set_size_preset(size_preset)

    async def reset_desktop_overlay_position(self) -> None:
        await self.desktop.reset_position()

    def begin_overlay_calibration(self) -> object:
        return self.calibration.begin()

    def set_overlay_calibration_field(
        self,
        field_name: object,
        value: object,
    ) -> object:
        return self.calibration.set_field(str(field_name), value)

    def apply_overlay_calibration(self) -> object:
        return self.calibration.apply()

    def cancel_overlay_calibration(self) -> object:
        return self.calibration.cancel()


@dataclass(slots=True)
class UiManagedRuntimeAdapter:
    managed: ManagedAccountComponents

    def dashboard_managed_auth_action(self) -> str:
        return self.managed.auth.dashboard_action()

    def dashboard_managed_auth_prompt_kind(self) -> str:
        return self.managed.auth.dashboard_prompt_kind()

    async def start_qq_managed_auth_from_dialog(self, **kwargs: object) -> object:
        return await self.managed.auth.start_qq(
            qq_identity=str(kwargs.get("qq_identity", "")),
            credential=str(kwargs.get("credential", "")),
        )

    async def start_discord_managed_auth_from_dialog(self, **kwargs: object) -> object:
        callback = kwargs.get("on_callback_received")
        referral_id = kwargs.get("referral_id")
        return await self.managed.auth.start_discord(
            on_callback_received=callback if callable(callback) else None,
            referral_id=str(referral_id) if referral_id is not None else None,
        )

    def clear_managed_auth_pending_state(self) -> None:
        self.managed.auth.clear_pending()

    async def refresh_openrouter_usage_after_launch(self) -> bool:
        await self.managed.usage.refresh(auto_show_founder_letter=True)
        return self.managed.usage.is_exhausted


@dataclass(slots=True)
class UiSettingsRuntimeAdapter:
    settings: SettingsOwner
    projection: SettingsProjectionOwner
    application: SettingsApplicationOwner
    merge_provider_settings: Callable[[object], object]
    telemetry_consent_settings: Callable[[object, str], object]

    async def on_dashboard_language_change(
        self,
        change: LanguageSelectionChange,
    ) -> None:
        await self.application.apply_language_selection(change)

    def capture_settings_view_change(self, settings: object) -> object:
        return self.projection.capture(settings)

    def merge_settings_view_change_with_current(self, captured: object) -> object:
        return self.projection.merge_with_current(captured)

    def refresh_settings_projection(
        self,
        *,
        preserve_custom_vocab_draft: bool = False,
    ) -> bool:
        settings = self.settings.current
        if settings is None:
            return False
        return bool(
            self.projection.render(
                settings,
                preserve_custom_vocab_draft=preserve_custom_vocab_draft,
            )
        )

    def refresh_settings_after_openrouter_pkce_success(self) -> bool:
        settings = self.settings.current
        if settings is None:
            return False
        return bool(self.projection.refresh_after_openrouter_pkce_success(settings))

    def merge_settings_tab_apply_with_current_languages(
        self,
        settings: object,
    ) -> object:
        return self.merge_provider_settings(settings)

    async def apply_settings(self, settings: object) -> object:
        return await self.application.apply(settings)

    async def apply_telemetry_consent(self, consent: str) -> object | None:
        settings = self.settings.current
        if settings is None:
            return None
        await self.application.apply(self.telemetry_consent_settings(settings, consent))
        return self.settings.current


@dataclass(slots=True)
class UiProviderRuntimeAdapter:
    settings: SettingsOwner
    provider_application: ProviderApplicationOwner
    gpu: GpuRuntimeInteractionOwner
    managed: ManagedAccountComponents
    credential_verification: ProviderCredentialVerificationInteractionOwner
    provider_settings: ProviderSettingsOwner
    build_byok_target_settings: Callable[[object | None], object | None]
    managed_gemma: ManagedGemmaTranslationOwner | None = None
    llm_devices_sink: Callable[[tuple[GpuDeviceOption, ...]], None] | None = None

    async def apply_providers(
        self,
        settings: object | None = None,
        *,
        force_rebuild_llm: bool = False,
        persist_settings: bool = True,
        refresh_ui: bool = True,
    ) -> object:
        if persist_settings:
            return await self.provider_application.apply(
                settings,
                force_rebuild_llm=force_rebuild_llm,
                refresh_ui=refresh_ui,
            )
        return await self.provider_application.apply(
            settings,
            force_rebuild_llm=force_rebuild_llm,
            persist_settings=persist_settings,
            refresh_ui=refresh_ui,
        )

    async def install_selected_gpu_model_if_needed(self) -> None:
        await self.gpu.install_selected_model_if_needed()

    async def ensure_gpu_device_discovery(self) -> None:
        await self.gpu.ensure_device_discovery(
            force=False,
            origin="settings",
        )
        sink = self.llm_devices_sink
        if sink is None:
            return
        sink(await asyncio.to_thread(_llama_gpu_device_options))

    async def connect_openrouter_via_pkce(
        self,
        *,
        target_settings: object,
        launch_source: str,
    ) -> bool:
        return await self.managed.pkce.connect(
            target_settings=target_settings,
            launch_source=launch_source,
        )

    def reopen_openrouter_pkce_authorization_url(self) -> object:
        return self.managed.pkce_flow.reopen_authorization_url()

    def build_managed_openrouter_byok_target_settings(self) -> object | None:
        return self.build_byok_target_settings(self.settings.current)

    async def verify_api_key(self, provider: str, key: str) -> tuple[bool, str]:
        return await self.credential_verification.verify(provider, key)

    def persist_api_key_verification(
        self,
        provider: str,
        key: str,
        success: bool,
    ) -> None:
        self.provider_settings.persist_verification(provider, key, success)

    async def persist_provider_secret_change(self, key: str, value: str) -> bool:
        succeeded = await self.provider_settings.change_secret(key, value)
        current = self.settings.current
        http_extension_id = None if current is None else current.translation.http_extension_id
        if (
            succeeded
            and key.startswith("http_extension.")
            and current is not None
            and current.translation.model == "custom_http"
            and http_extension_id is not None
            and key.startswith(http_extension_secret_key_prefix(http_extension_id))
        ):
            await self.apply_providers(
                force_rebuild_llm=True,
                persist_settings=False,
            )
        return succeeded

    def clear_provider_verification(self, provider: str) -> None:
        self.provider_settings.persist_verification(provider, "", False)

    def handle_gpu_notice_action(self, action: GpuNoticeAction) -> object:
        return self.gpu.handle_notice_action(action)

    async def handle_managed_gemma_notice_action(
        self,
        action: ManagedGemmaNoticeAction,
    ) -> object:
        if action == "cancel":
            return False if self.managed_gemma is None else self.managed_gemma.cancel()
        raise ValueError(f"unsupported managed Gemma notice action: {action}")


@dataclass(slots=True)
class UiEngagementRuntimeAdapter:
    settings: SettingsOwner
    settings_application: SettingsApplicationOwner
    github_prompt: GithubStarPromptOwner
    telemetry: TranslationSuccessTelemetryService
    after_launch: ApplicationAfterLaunchOwner

    def get_event_language_codes(self) -> tuple[str | None, str | None]:
        settings = self.settings.current
        if settings is None:
            return None, None
        return (
            settings.languages.source_language,
            settings.languages.target_language,
        )

    def schedule_github_star_prompt_translation_success_observed(self) -> None:
        self.github_prompt.schedule_translation_success_observed()

    async def record_telemetry_translation_success_day(
        self,
    ) -> TranslationSuccessTelemetryResult:
        settings = self.settings.current
        if settings is None:
            return TranslationSuccessTelemetryResult(status="skipped_no_settings")

        async def persist(updated: object) -> bool:
            await self.settings_application.apply(updated)
            return self.settings_application.results.committed()

        return await self.telemetry.record_translation_success_day(
            settings,
            persist_sent_date=persist,
        )

    def should_show_github_star_prompt(self) -> bool:
        return self.github_prompt.should_show()

    async def persist_github_star_prompt_eligible_launch(self) -> bool:
        return await self.github_prompt.persist_eligible_launch()

    async def prepare_runtime_after_launch(self) -> None:
        await self.after_launch.prepare()

    async def persist_github_star_prompt_opened(
        self,
        *,
        should_open: Callable[[], bool] | None = None,
    ) -> bool:
        return await self.github_prompt.persist_opened(should_open=should_open)

    async def persist_github_star_prompt_clicked(self) -> None:
        await self.github_prompt.persist_clicked()


@dataclass(slots=True)
class UiDiagnosticsRuntimeAdapter:
    runtime_logging: ApplicationRuntimeLoggingOwner
    overlay: OverlayApplicationOwner
    cycle_capture_fault: Callable[[], str]
    cycle_stt_fault: Callable[[], str]
    clear_audio_faults: Callable[[], None]

    def set_runtime_logging_mode(self, mode: str) -> None:
        def mode_changed(normalized_mode: str) -> None:
            runtime = self.overlay.runtime
            manager = runtime.process_manager if runtime is not None else None
            if manager is not None:
                set_logging_mode = getattr(manager, "set_logging_mode", None)
                if callable(set_logging_mode):
                    set_logging_mode(normalized_mode)
            self.runtime_logging.schedule_overlay_logging_mode_update()

        self.runtime_logging.set_mode(
            mode,
            detailed_enabled=self.runtime_logging.schedule_audio_environment_snapshot,
            mode_changed=mode_changed,
        )

    def cycle_debug_capture_fault_profile(self) -> str:
        return self.cycle_capture_fault()

    def cycle_debug_stt_fault_profile(self) -> str:
        return self.cycle_stt_fault()

    def clear_debug_audio_fault_profiles(self) -> None:
        self.clear_audio_faults()
