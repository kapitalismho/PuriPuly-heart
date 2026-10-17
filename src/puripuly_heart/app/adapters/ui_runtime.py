from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from puripuly_heart.app.language_selection import LanguageSelectionChange
from puripuly_heart.app.ports.chatgpt_account import (
    ChatGptAccountSnapshot,
    ChatGptConnectResult,
    ChatGptSignOutResult,
)
from puripuly_heart.app.ports.settings_view import (
    AlibabaConnectionApplyResult,
    AlibabaConnectionDraftSnapshot,
    GeneralSettingsSnapshot,
    ImmediateSettingsIntent,
    OpenRouterPkceTarget,
    OverlaySettingsSnapshot,
    PromptApplyIntent,
    ProviderApplyIntent,
    TranslationSelectionEdit,
)
from puripuly_heart.app.ports.ui_models import (
    GpuDeviceOption,
    GpuNoticeAction,
    ManagedGemmaNoticeAction,
    OverlayPeerPresentationState,
)
from puripuly_heart.app.services.application_after_launch import (
    ApplicationAfterLaunchOwner,
)
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.services.chatgpt_account import ChatGptAccountOwner
from puripuly_heart.app.services.desktop_overlay_application import (
    DesktopOverlayApplicationOwner,
)
from puripuly_heart.app.services.github_star_prompt import GithubStarPromptOwner
from puripuly_heart.app.services.gpu_runtime_interaction import GpuRuntimeInteractionOwner
from puripuly_heart.app.services.llm_connection_readiness import LlmConnectionReadinessOwner
from puripuly_heart.app.services.managed_gemma_translation import (
    ManagedGemmaTranslationOwner,
)
from puripuly_heart.app.services.manual_typing import ManualTypingOwner
from puripuly_heart.app.services.model_assets import ModelAssetsApplication
from puripuly_heart.app.services.overlay_application import OverlayApplicationOwner
from puripuly_heart.app.services.overlay_calibration_application import (
    OverlayCalibrationApplicationOwner,
)
from puripuly_heart.app.services.provider.alibaba_workspace import AlibabaWorkspaceOwner
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
from puripuly_heart.app.services.settings_application import (
    SettingsApplicationOwner,
    materialize_immediate_settings_intent,
    materialize_prompt_apply_intent,
    materialize_provider_apply_intent,
    settings_view_surface_snapshots,
)
from puripuly_heart.app.services.settings_projection import (
    SettingsProjectionOwner,
)
from puripuly_heart.app.services.translation_enable import TranslationEnableOwner
from puripuly_heart.app.wiring_audio_test import AudioTestRuntime
from puripuly_heart.app.wiring_managed_account import ManagedAccountComponents
from puripuly_heart.app.wiring_peer_application import PeerApplicationRuntime
from puripuly_heart.app.wiring_runtime_pipeline import RuntimePipelineHandle
from puripuly_heart.config.alibaba_connection import AlibabaEndpointMode, AlibabaRegion
from puripuly_heart.config.prompts import resolve_system_prompt
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.config.translation_values import provider_llm_for_translation
from puripuly_heart.core.http_extensions import (
    http_extension_secret_key_prefix,
)
from puripuly_heart.core.local_translation.devices import list_llama_vulkan_devices
from puripuly_heart.core.orchestrator.self_translation_channel import TextSubmissionAdmission
from puripuly_heart.core.telemetry import (
    AppActiveDayTelemetryResult,
    AppActiveDayTelemetryService,
    AppActiveDayTelemetryState,
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
    connection_readiness: LlmConnectionReadinessOwner | None = None

    async def submit_text(self, text: str) -> TextSubmissionAdmission | None:
        owner = self.pipeline.self_translation_channel
        submit = None if owner is None else lambda: owner.submit_text(text, source="You")
        return await self.manual_typing.submit(submit)

    def set_manual_input_activity(self, has_text: bool) -> None:
        self.manual_typing.set_input_activity(has_text)

    async def set_translation_enabled(
        self, enabled: bool, *, allow_authorization: bool = True
    ) -> object:
        result = await self.translation.set_enabled(
            enabled, allow_authorization=allow_authorization
        )
        self._sync_connection_readiness()
        return result

    async def set_stt_enabled(self, enabled: bool) -> object:
        result = await self.self_capture.set_enabled(enabled)
        self._sync_connection_readiness()
        return result

    def _sync_connection_readiness(self) -> None:
        if self.connection_readiness is not None:
            self.connection_readiness.sync()


@dataclass(slots=True)
class UiPeerCaptureRuntimeAdapter:
    peer: PeerApplicationRuntime
    overlay: OverlayApplicationOwner
    connection_readiness: LlmConnectionReadinessOwner | None = None

    async def set_peer_translation_enabled(self, enabled: bool) -> object:
        result = await self.peer.owner.set_enabled(enabled)
        if self.connection_readiness is not None:
            self.connection_readiness.sync()
        return result

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
class UiAudioTestRuntimeAdapter:
    audio_test: AudioTestRuntime

    async def start_audio_test(self, *, observation_callback=None) -> bool:
        return await self.audio_test.start(observation_callback=observation_callback)

    def audio_test_snapshot(self) -> dict[str, object]:
        return self.audio_test.snapshot

    async def wait_audio_test_ready(self) -> bool:
        return await self.audio_test.wait_ready()

    async def stop_audio_test(self) -> None:
        await self.audio_test.stop()

    def release_audio_test_capture_restore(self, channel: str) -> None:
        self.audio_test.release_capture_restore(channel)


@dataclass(slots=True)
class UiOverlayRuntimeAdapter:
    overlay: OverlayApplicationOwner
    desktop: DesktopOverlayApplicationOwner
    calibration: OverlayCalibrationApplicationOwner

    async def set_overlay_enabled(self, enabled: bool) -> object:
        return await self.overlay.set_enabled(enabled)

    async def wait_overlay_transition(self) -> dict[str, object]:
        return {
            **await self.overlay.wait_start_outcome(),
            "calibration_draft_active": self.calibration.draft is not None,
        }

    async def set_desktop_overlay_captions_locked(self, locked: bool) -> dict[str, str]:
        return await self.desktop.set_captions_locked(locked)

    async def set_desktop_overlay_size_preset(self, size_preset: str) -> dict[str, str]:
        return await self.desktop.set_size_preset(size_preset)

    async def reset_desktop_overlay_position(self) -> dict[str, str]:
        return await self.desktop.reset_position()

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
            referral_id=(
                str(kwargs["referral_id"]) if kwargs.get("referral_id") is not None else None
            ),
        )

    async def start_discord_managed_auth_from_dialog(self, **kwargs: object) -> object:
        callback = kwargs.get("on_callback_received")
        recovery_hook = kwargs.get("on_recovery_started")
        referral_id = kwargs.get("referral_id")
        url_sink = kwargs.get("authorization_url_sink")
        return await self.managed.auth.start_discord(
            on_callback_received=callback if callable(callback) else None,
            on_recovery_started=recovery_hook if callable(recovery_hook) else None,
            referral_id=str(referral_id) if referral_id is not None else None,
            authorization_url_sink=url_sink if callable(url_sink) else None,
            open_browser=kwargs.get("open_browser", True) is True,
        )

    async def logout_local_managed(self, provider: str) -> dict[str, object]:
        return await self.managed.auth_runtime.logout_local_managed(provider)

    def managed_auth_last_failure_kind(self) -> str:
        kind = getattr(self.managed.auth, "last_failure_kind", "failed")
        return kind if isinstance(kind, str) else "failed"

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
    merge_provider_settings: Callable[[AppSettingsVNext], AppSettingsVNext]
    telemetry_enabled_settings: Callable[[AppSettingsVNext, bool], AppSettingsVNext]

    async def on_dashboard_language_change(
        self,
        change: LanguageSelectionChange,
    ) -> None:
        await self.application.apply_language_selection(change)

    def capture_settings_view_change(self, settings: AppSettingsVNext) -> object:
        return self.projection.capture(settings)

    def merge_settings_view_change_with_current(self, captured: object) -> AppSettingsVNext:
        return self.projection.merge_with_current(captured)

    def refresh_settings_projection(
        self,
        *,
        preserve_custom_vocab_draft: bool = False,
    ) -> bool:
        settings = self.settings.canonical
        if settings is None:
            return False
        provider, general, prompt, overlay = settings_view_surface_snapshots(settings)
        return bool(
            self.projection.render_surfaces(
                provider=provider,
                general=general,
                prompt=prompt,
                overlay=overlay,
                compatibility_settings=settings,
                preserve_custom_vocab_draft=preserve_custom_vocab_draft,
            )
        )

    def refresh_settings_after_openrouter_pkce_success(self) -> bool:
        settings = self.settings.canonical
        if settings is None:
            return False
        provider, _general, prompt, _overlay = settings_view_surface_snapshots(settings)
        return bool(
            self.projection.refresh_after_openrouter_pkce_success(
                provider=provider,
                prompt=prompt,
                compatibility_settings=settings,
            )
        )

    def merge_settings_tab_apply_with_current_languages(
        self,
        settings: AppSettingsVNext,
    ) -> AppSettingsVNext:
        return self.merge_provider_settings(settings)

    async def apply_settings(
        self,
        settings: AppSettingsVNext,
        *,
        provider_requested: bool = False,
    ) -> object:
        return await self.application.apply(settings, provider_requested=provider_requested)

    async def apply_telemetry_enabled(self, enabled: bool) -> object | None:
        settings = self.settings.canonical
        if settings is None:
            return None
        await self.application.apply(self.telemetry_enabled_settings(settings, enabled))
        return self.settings.canonical

    async def apply_settings_intent(self, intent: ImmediateSettingsIntent) -> object:
        settings = self.settings.canonical
        if settings is None:
            return False
        return await self.application.apply(materialize_immediate_settings_intent(settings, intent))

    async def apply_prompt_intent(self, intent: PromptApplyIntent) -> object:
        settings = self.settings.canonical
        if settings is None:
            return False
        return await self.application.apply(materialize_prompt_apply_intent(settings, intent))

    def materialize_provider_intent(self, intent: ProviderApplyIntent) -> object:
        settings = self.settings.canonical
        if settings is None:
            raise RuntimeError("settings owner has no compatibility settings")
        return materialize_provider_apply_intent(
            settings,
            intent,
            materialize_translation=self.settings.materialize_translation,
        )

    def overlay_snapshot(self) -> OverlaySettingsSnapshot | None:
        settings = self.settings.canonical
        if settings is None:
            return None
        return settings_view_surface_snapshots(settings)[3]

    def general_snapshot(self) -> GeneralSettingsSnapshot | None:
        settings = self.settings.canonical
        if settings is None:
            return None
        return settings_view_surface_snapshots(settings)[1]


@dataclass(slots=True)
class UiProviderRuntimeAdapter:
    settings: SettingsOwner
    provider_application: ProviderApplicationOwner
    gpu: GpuRuntimeInteractionOwner
    managed: ManagedAccountComponents
    credential_verification: ProviderCredentialVerificationInteractionOwner
    provider_settings: ProviderSettingsOwner
    build_byok_target_settings: Callable[[AppSettingsVNext | None], AppSettingsVNext | None]
    managed_gemma: ManagedGemmaTranslationOwner | None = None
    llm_devices_sink: Callable[[tuple[GpuDeviceOption, ...]], None] | None = None
    chatgpt: ChatGptAccountOwner | None = None
    connection_readiness: LlmConnectionReadinessOwner | None = None
    model_assets: ModelAssetsApplication | None = None
    _alibaba_workspace: AlibabaWorkspaceOwner | None = field(default=None, init=False)

    def alibaba_workspace(self) -> AlibabaWorkspaceOwner:
        if self._alibaba_workspace is None:
            self._alibaba_workspace = AlibabaWorkspaceOwner(
                self.settings,
                self.provider_settings,
                self.credential_verification.verification_owner.verifier,
            )
        return self._alibaba_workspace

    async def begin_alibaba_connection_draft(self) -> AlibabaConnectionDraftSnapshot:
        return await self.alibaba_workspace().begin()

    async def alibaba_connection_draft(self) -> AlibabaConnectionDraftSnapshot:
        return await self.alibaba_workspace().read()

    async def alibaba_active_connection(self) -> AlibabaConnectionDraftSnapshot:
        return await self.alibaba_workspace().active()

    async def edit_alibaba_connection_draft(
        self,
        *,
        token: str,
        region: AlibabaRegion | None = None,
        endpoint_mode: AlibabaEndpointMode | None = None,
        api_host: str | None = None,
    ) -> AlibabaConnectionDraftSnapshot:
        return await self.alibaba_workspace().edit(
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
        return await self.alibaba_workspace().verify(
            token=token,
            capability=capability,
            api_key=api_key,
        )

    async def apply_alibaba_connection_draft(self, *, token: str) -> AlibabaConnectionApplyResult:
        affected = (await self.alibaba_workspace().read()).affected_consumers
        committed = await self.alibaba_workspace().apply(
            token=token, apply_settings=self.apply_providers
        )
        return AlibabaConnectionApplyResult(
            committed=bool(committed),
            affected_consumers=affected,
            transaction=self.provider_application.results.current,
        )

    def cancel_alibaba_connection_draft(self, *, token: str) -> None:
        self.alibaba_workspace().cancel(token=token)

    async def apply_providers(
        self,
        settings: AppSettingsVNext | None = None,
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
        target: OpenRouterPkceTarget,
        launch_source: str,
        open_browser: bool = True,
        authorization_url_sink: Callable[[str], None] | None = None,
    ) -> bool:
        return await self.managed.pkce.connect(
            target=target,
            launch_source=launch_source,
            open_browser=open_browser,
            authorization_url_sink=authorization_url_sink,
        )

    def reopen_openrouter_pkce_authorization_url(self) -> object:
        return self.managed.pkce_flow.reopen_authorization_url()

    def chatgpt_account_snapshot(self) -> ChatGptAccountSnapshot:
        if self.chatgpt is None:
            return ChatGptAccountSnapshot(signed_in=False)
        return self.chatgpt.snapshot()

    async def connect_chatgpt(
        self,
        *,
        open_browser: bool = True,
        authorization_url_sink: Callable[[str], None] | None = None,
    ) -> ChatGptConnectResult:
        if self.chatgpt is None:
            return ChatGptConnectResult(succeeded=False, failure_code="unavailable")
        result = await self.chatgpt.connect(
            open_browser=open_browser,
            authorization_url_sink=authorization_url_sink,
        )
        if result.succeeded and self.chatgpt_route_active():
            await self.apply_providers(force_rebuild_llm=True, persist_settings=False)
        return result

    def reopen_chatgpt_authorization_url(self) -> bool:
        return self.chatgpt is not None and self.chatgpt.reopen_authorization_url()

    async def sign_out_chatgpt(self) -> ChatGptSignOutResult:
        if self.chatgpt is None:
            return ChatGptSignOutResult(remote_revoked=False)
        readiness = self.connection_readiness if self.chatgpt_route_active() else None
        if readiness is not None:
            await readiness.suspend()
        try:
            result = await self.chatgpt.sign_out()
            if self.chatgpt_route_active():
                await self.apply_providers(force_rebuild_llm=True, persist_settings=False)
            return result
        finally:
            if readiness is not None:
                readiness.resume()

    def cancel_chatgpt_sign_in(self) -> None:
        if self.chatgpt is not None:
            self.chatgpt.cancel()

    def chatgpt_sign_in_required(self) -> bool:
        return self.chatgpt_route_active() and not self.chatgpt_account_snapshot().signed_in

    def chatgpt_route_active(self) -> bool:
        canonical = self.settings.canonical
        if canonical is None:
            return False
        translation = canonical.intent.translation
        return provider_llm_for_translation(translation.model, translation.connection) == "chatgpt"

    def build_managed_openrouter_byok_target(self) -> OpenRouterPkceTarget | None:
        from puripuly_heart.config.provider_values import OpenRouterSelectionAlias

        target_settings = self.build_byok_target_settings(self.settings.canonical)
        if target_settings is None:
            return None
        alias = target_settings.intent.translation.openrouter_selection_alias
        if alias is None:
            return None
        provider, _general, _prompt, _overlay = settings_view_surface_snapshots(target_settings)
        return OpenRouterPkceTarget(
            OpenRouterSelectionAlias(alias),
            provider_intent=ProviderApplyIntent(
                (
                    TranslationSelectionEdit(
                        provider.translation,
                        (
                            (
                                provider.translation.model,
                                provider.translation.connection,
                            ),
                        ),
                    ),
                )
            ),
            system_prompt=resolve_system_prompt(
                target_settings.intent.prompts.system_prompt_override
            ),
        )

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
        if succeeded:
            rebind = getattr(self.provider_application, "rebind_rolling_stt_secret", None)
            if callable(rebind):
                rebind(key, value)
        if succeeded and key in {"alibaba_api_key_beijing", "alibaba_api_key_singapore"}:
            await self.apply_providers(persist_settings=False)
        current = self.settings.canonical
        if succeeded and current is not None:
            translation = current.intent.translation
            active_secret = None
            if translation.connection == "official_byok":
                active_secret = {
                    "gemini": "google_api_key",
                    "openai": "openai_api_key",
                    "deepseek": "deepseek_api_key",
                }.get(provider_llm_for_translation(translation.model, translation.connection))
            elif (
                translation.connection == "openrouter"
                and translation.openrouter_selected_source == "byok"
            ):
                active_secret = "openrouter_api_key"
            if key == active_secret:
                await self.apply_providers(
                    force_rebuild_llm=True,
                    persist_settings=False,
                )
                return succeeded
        http_extension_id = (
            None if current is None else current.intent.translation.http_extension_id
        )
        if (
            succeeded
            and key.startswith("http_extension.")
            and current is not None
            and current.intent.translation.model == "custom_http"
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

    def model_asset_entries(self) -> tuple[object, ...]:
        return self.model_assets.entries() if self.model_assets is not None else ()

    async def refresh_model_assets(self) -> tuple[object, ...]:
        if self.model_assets is None:
            return ()
        return await self.model_assets.refresh()

    async def download_model_asset(self, model_id: str, *, source: str | None = None) -> object:
        return await self._require_model_assets().download(model_id, source=source)

    async def import_model_asset(self, model_id: str, directory: Path) -> object:
        return await self._require_model_assets().import_directory(model_id, directory)

    async def cancel_model_asset(self, model_id: str) -> bool:
        return await self._require_model_assets().cancel(model_id)

    async def delete_model_asset(self, model_id: str) -> object:
        return await self._require_model_assets().delete(model_id)

    def _require_model_assets(self) -> ModelAssetsApplication:
        if self.model_assets is None:
            raise RuntimeError("Model assets are unavailable")
        return self.model_assets


@dataclass(slots=True)
class UiEngagementRuntimeAdapter:
    settings: SettingsOwner
    settings_application: SettingsApplicationOwner
    github_prompt: GithubStarPromptOwner
    telemetry: AppActiveDayTelemetryService
    after_launch: ApplicationAfterLaunchOwner

    def get_event_language_codes(self) -> tuple[str | None, str | None]:
        settings = self.settings.canonical
        if settings is None:
            return None, None
        return (
            settings.intent.languages.source_language,
            settings.intent.languages.target_language,
        )

    def schedule_github_star_prompt_translation_success_observed(self) -> None:
        self.github_prompt.schedule_translation_success_observed()

    async def record_app_active_day(
        self,
        active_date_utc: str,
    ) -> AppActiveDayTelemetryResult:
        canonical = self.settings.canonical
        if canonical is None:
            return AppActiveDayTelemetryResult(status="skipped_no_settings")
        state = AppActiveDayTelemetryState(
            enabled=canonical.intent.telemetry.enabled,
            anonymous_id=canonical.state.telemetry.anonymous_id,
            last_sent_date_utc=canonical.state.telemetry.last_sent_date_utc,
            broker_base_url=(canonical.intent.translation.openrouter_broker_base_url),
        )

        async def persist(updated: AppActiveDayTelemetryState) -> bool:
            current = self.settings.canonical
            if current is None:
                return False
            next_settings = replace(
                current,
                state=replace(
                    current.state,
                    telemetry=replace(
                        current.state.telemetry,
                        last_sent_date_utc=updated.last_sent_date_utc,
                    ),
                ),
            )
            await self.settings_application.apply(next_settings)
            return self.settings_application.results.committed()

        return await self.telemetry.record_app_active_day(
            state,
            active_date_utc=active_date_utc,
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
    cycle_capture_fault: Callable[[], str]
    cycle_stt_fault: Callable[[], str]
    clear_audio_faults: Callable[[], None]

    def cycle_debug_capture_fault_profile(self) -> str:
        return self.cycle_capture_fault()

    def cycle_debug_stt_fault_profile(self) -> str:
        return self.cycle_stt_fault()

    def clear_debug_audio_fault_profiles(self) -> None:
        self.clear_audio_faults()
