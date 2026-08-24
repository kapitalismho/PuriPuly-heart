from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from puripuly_heart.app.ports.desktop_overlay import DesktopOverlayRuntimeEffectsPort
from puripuly_heart.app.ports.overlay_calibration import (
    OverlayCalibrationRuntimeEffectsPort,
)
from puripuly_heart.app.ports.settings_runtime_effects import (
    SettingsRuntimeState,
    SettingsRuntimeTransition,
)
from puripuly_heart.app.ports.ui_presentation import UiPresentationPort
from puripuly_heart.app.services.application_runtime_logging import (
    ApplicationRuntimeLoggingOwner,
)
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.services.clipboard_auto_translation import (
    ClipboardAutoTranslationOwner,
)
from puripuly_heart.app.services.github_star_prompt import GithubStarPromptOwner
from puripuly_heart.app.services.gpu_runtime_interaction import GpuRuntimeInteractionOwner
from puripuly_heart.app.services.osc.control_runtime import OscControlIntegrationOwner
from puripuly_heart.app.services.overlay_application import (
    OverlayApplicationOwner,
    OverlayApplicationState,
)
from puripuly_heart.app.wiring_microphone_test import MicrophoneTestRuntime
from puripuly_heart.app.wiring_peer_application import PeerApplicationRuntime
from puripuly_heart.app.wiring_provider_runtime import (
    ProviderRuntimeSignatures,
    project_translation_runtime_settings,
)
from puripuly_heart.app.wiring_runtime_pipeline import RuntimePipelineHandle
from puripuly_heart.app.wiring_stt_factory import (
    build_peer_stt_runtime_signature,
    build_self_capture_vad_signature,
    build_self_stt_runtime_signature,
)
from puripuly_heart.app.wiring_translation_runtime_configuration import (
    replace_translation_runtime_settings,
)
from puripuly_heart.config.settings import (
    OVERLAY_TARGET_DESKTOP,
    AppSettings,
    LLMProviderName,
    TranslationModel,
)
from puripuly_heart.core.local_asr_provisioning import LocalASRProvisioningPort
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfigChange,
)
from puripuly_heart.core.runtime.self_capture import SelfCaptureSessionOwner

from .settings_projection import SettingsProjectionOwner


@dataclass(slots=True)
class SettingsRuntimeEffectsState:
    microphone_audio_signature: object | None = None


def managed_gemma_prefix_refresh_required(
    transition: SettingsRuntimeTransition[AppSettings],
) -> bool:
    settings = transition.settings
    if settings.translation.model not in {
        TranslationModel.MANAGED_GEMMA,
        TranslationModel.MANAGED_GEMMA_12B,
    }:
        return False
    previous = transition.previous_settings
    return bool(
        previous is None
        or previous.translation.model != settings.translation.model
        or transition.source_language_changed
        or transition.target_language_changed
        or previous.system_prompt != settings.system_prompt
    )


async def refresh_managed_gemma_prefix(
    transition: SettingsRuntimeTransition[AppSettings],
    *,
    rebuild: Callable[[], Awaitable[bool]],
) -> None:
    if not managed_gemma_prefix_refresh_required(transition):
        return
    if not await rebuild():
        raise RuntimeError("managed Gemma prefix rebuild failed")


class SettingsRuntimeEffectsAdapter:
    def __init__(
        self,
        *,
        state: SettingsRuntimeEffectsState,
        settings: SettingsOwner,
        presentation: UiPresentationPort,
        runtime_logging: ApplicationRuntimeLoggingOwner,
        pipeline: RuntimePipelineHandle,
        runtime_signatures: ProviderRuntimeSignatures,
        microphone: MicrophoneTestRuntime,
        clipboard: ClipboardAutoTranslationOwner,
        provisioning: LocalASRProvisioningPort,
        gpu: GpuRuntimeInteractionOwner,
        vrc_mic_sync: OscControlIntegrationOwner,
        projection: SettingsProjectionOwner,
        github_prompt: Callable[[], GithubStarPromptOwner],
        desktop_overlay: DesktopOverlayRuntimeEffectsPort[AppSettings],
        calibration: OverlayCalibrationRuntimeEffectsPort,
        overlay: OverlayApplicationOwner,
        overlay_state_provider: Callable[[AppSettings | None], OverlayApplicationState],
        peer: PeerApplicationRuntime,
        self_capture: Callable[[], SelfCaptureSessionOwner | None],
        clear_local_pending: Callable[[], None],
        replace_self_stt: Callable[[bool], Awaitable[None]],
        rebuild_managed_gemma: Callable[[], Awaitable[bool]],
    ) -> None:
        self._state = state
        self._settings = settings
        self._presentation = presentation
        self._runtime_logging = runtime_logging
        self._pipeline = pipeline
        self._runtime_signatures = runtime_signatures
        self._microphone = microphone
        self._clipboard = clipboard
        self._provisioning = provisioning
        self._gpu = gpu
        self._vrc_mic_sync = vrc_mic_sync
        self._projection = projection
        self._github_prompt = github_prompt
        self._desktop_overlay = desktop_overlay
        self._calibration = calibration
        self._overlay = overlay
        self._overlay_state_provider = overlay_state_provider
        self._peer = peer
        self._self_capture = self_capture
        self._clear_local_pending = clear_local_pending
        self._replace_self_stt = replace_self_stt
        self._rebuild_managed_gemma = rebuild_managed_gemma

    async def preserve_before_replace(self, settings: AppSettings) -> None:
        await self._github_prompt().preserve_before_settings_replace(settings)

    def capture_runtime_signatures(self) -> None:
        settings = self._settings.current
        if settings is None:
            return
        self._runtime_signatures.capture_peer_before_canonical_mutation(
            settings,
            canonical=self._canonical_settings(settings),
            peer=self._peer.owner,
        )

    async def prepare(
        self,
        current_settings: AppSettings | None,
        next_settings: AppSettings,
    ) -> SettingsRuntimeTransition[AppSettings]:
        microphone_owner = self._microphone.owner_if_created
        previous_microphone_signature = (
            self._state.microphone_audio_signature
            or (microphone_owner.audio_signature if microphone_owner is not None else None)
            or self._microphone.audio_signature(current_settings)
        )
        next_microphone_signature = self._microphone.audio_signature(next_settings)
        if (
            previous_microphone_signature is not None
            and previous_microphone_signature != next_microphone_signature
        ):
            await self._microphone.stop()

        previous_locale = self._presentation.current_locale()
        previous_overlay_enabled = (
            current_settings.ui.overlay_enabled if current_settings is not None else False
        )
        previous_settings = (
            copy.deepcopy(current_settings) if current_settings is not None else None
        )
        previous_settings_overlay_target = self._overlay.target_for_state(
            self._overlay_state_provider(current_settings)
        )
        next_overlay_target = self._overlay.target_for_state(
            self._overlay_state_provider(next_settings)
        )
        if self._overlay.snapshot.fallback_active:
            previous_overlay_target = previous_settings_overlay_target
        else:
            previous_overlay_target = self._overlay.previous_target_for_apply()
        if next_overlay_target == OVERLAY_TARGET_DESKTOP:
            self._overlay.clear_fallback()
        if (
            previous_overlay_target != next_overlay_target
            and previous_overlay_enabled
            and next_settings.ui.overlay_enabled
            and self._overlay.runtime_is_active()
        ):
            self._runtime_logging.emit_basic(
                "[Overlay] Target changed while running; stopping current overlay before switch"
            )
            next_settings = copy.deepcopy(next_settings)
            next_settings.ui.overlay_enabled = False
            self._overlay.clear_fallback()
        desktop_runtime_controls = tuple(
            self._desktop_overlay.prepare_settings_update(
                previous_settings,
                next_settings,
            )
        )
        peer = self._peer.owner
        previous_peer_translation_enabled = (
            peer.last_intent_enabled
            if peer.last_intent_enabled is not None
            else (
                current_settings.ui.peer_translation_enabled
                if current_settings is not None
                else False
            )
        )
        previous_peer_activation_requested = (
            peer.last_activation_requested
            if peer.last_activation_requested is not None
            else (
                peer.activation_requested(
                    intent_enabled=current_settings.ui.peer_translation_enabled,
                    eula_accepted=current_settings.ui.peer_translation_eula_accepted,
                )
                if current_settings is not None
                else False
            )
        )
        previous_self_signature = self._runtime_signatures.last_self_runtime
        previous_peer_signature = peer.last_runtime_signature
        output_projection = self._pipeline.translation_output_projection
        config_owner = self._pipeline.translation_runtime_configuration
        previous_configuration = config_owner.snapshot().value if config_owner is not None else None
        previous_source_language = (
            previous_configuration.source_language if previous_configuration is not None else None
        )
        previous_target_language = (
            previous_configuration.target_language if previous_configuration is not None else None
        )
        previous_peer_source_language = (
            previous_configuration.peer_source_language
            if previous_configuration is not None
            else None
        )
        previous_peer_target_language = (
            previous_configuration.peer_target_language
            if previous_configuration is not None
            else None
        )
        previous_peer_source_mode = (
            previous_settings.languages.peer_source_mode if previous_settings is not None else None
        )
        previous_effective_peer_source = (
            self._effective_peer_language(
                previous_source_language,
                previous_peer_source_language,
            )
            if previous_source_language is not None and previous_peer_source_language is not None
            else None
        )
        previous_effective_peer_target = (
            self._effective_peer_language(
                previous_target_language,
                previous_peer_target_language,
            )
            if previous_target_language is not None and previous_peer_target_language is not None
            else None
        )
        source_language_changed = (
            previous_source_language is not None
            and previous_source_language != next_settings.languages.source_language
        )
        target_language_changed = (
            previous_target_language is not None
            and previous_target_language != next_settings.languages.target_language
        )
        effective_peer_source_changed = (
            previous_effective_peer_source is not None
            and previous_effective_peer_source
            != self._effective_peer_language(
                next_settings.languages.source_language,
                next_settings.languages.peer_source_language,
            )
        )
        effective_peer_target_changed = (
            previous_effective_peer_target is not None
            and previous_effective_peer_target
            != self._effective_peer_language(
                next_settings.languages.target_language,
                next_settings.languages.peer_target_language,
            )
        )
        peer_source_language_changed = (
            previous_peer_source_language is not None
            and previous_peer_source_language != next_settings.languages.peer_source_language
        )
        peer_target_language_changed = (
            previous_peer_target_language is not None
            and previous_peer_target_language != next_settings.languages.peer_target_language
        )
        peer_source_mode_changed = (
            previous_peer_source_mode is not None
            and previous_peer_source_mode != next_settings.languages.peer_source_mode
        )
        if source_language_changed or target_language_changed:
            presenter = self._overlay.current_presenter()
            bridge = self._overlay.current_bridge()
            self._runtime_logging.emit_basic(
                "[Settings] Applying languages: "
                f"source={previous_source_language}->{next_settings.languages.source_language} "
                f"target={previous_target_language}->{next_settings.languages.target_language}"
            )
            self._runtime_logging.emit_detailed(
                "[Settings] Language apply detail: "
                f"overlay_state={self._overlay.snapshot.state} "
                f"presenter_attached={presenter is not None} "
                f"bridge_attached={bridge is not None} "
                "overlay_sink_matches_presenter="
                f"{output_projection is not None and presenter is not None and output_projection.overlay_sink is presenter}"
            )
        return SettingsRuntimeTransition(
            settings=next_settings,
            previous_settings=previous_settings,
            previous_locale=previous_locale,
            previous_overlay_enabled=previous_overlay_enabled,
            previous_self_signature=previous_self_signature,
            previous_peer_signature=previous_peer_signature,
            previous_peer_translation_enabled=previous_peer_translation_enabled,
            previous_peer_activation_requested=previous_peer_activation_requested,
            source_language_changed=source_language_changed,
            target_language_changed=target_language_changed,
            effective_peer_source_changed=effective_peer_source_changed,
            effective_peer_target_changed=effective_peer_target_changed,
            peer_source_language_changed=peer_source_language_changed,
            peer_target_language_changed=peer_target_language_changed,
            peer_source_mode_changed=peer_source_mode_changed,
            desktop_runtime_controls=desktop_runtime_controls,
        )

    def activate_before_persist(
        self,
        transition: SettingsRuntimeTransition[AppSettings],
    ) -> None:
        settings = transition.settings
        self._state.microphone_audio_signature = self._microphone.audio_signature(settings)
        self._calibration.sync_from_settings(settings)
        self._desktop_overlay.sync_from_settings(settings)

    async def prepare_overlay_persistence(
        self,
        previous_settings: AppSettings,
        next_settings: AppSettings,
    ) -> None:
        await self._desktop_overlay.prepare_persistence(
            previous_settings,
            next_settings,
        )

    def restore_memory(self, settings: AppSettings) -> None:
        restored_settings = copy.deepcopy(settings)
        self._settings.current = restored_settings
        self._calibration.sync_from_settings(restored_settings)
        config_owner = self._pipeline.translation_runtime_configuration
        if config_owner is not None:
            peer_enabled = self._peer.owner.effective_enabled(
                self._peer.state_for(restored_settings)
            )
            replace_translation_runtime_settings(
                config_owner,
                project_translation_runtime_settings(restored_settings),
                peer_translation_enabled=peer_enabled,
                integrated_context_enabled=peer_enabled,
            )
        self._sync_signatures(restored_settings)

    def sync_signatures(self, settings: AppSettings) -> None:
        self._sync_signatures(settings)

    def state(self, settings: AppSettings) -> SettingsRuntimeState:
        local_asr_runtime = self._pipeline.local_asr_runtime
        llm_runtime = self._pipeline.llm_runtime
        self_capture = self._self_capture()
        return SettingsRuntimeState(
            runtime_available=(local_asr_runtime is not None and llm_runtime is not None),
            self_stt_desired=bool(
                self_capture is not None and self_capture.snapshot.desired_active
            ),
            self_stt_available=(
                local_asr_runtime is not None
                and local_asr_runtime.snapshot.channel_for("self").provider_id is not None
            ),
            peer_stt_desired=self._peer.owner.desired_active(self._peer.state_for(settings)),
            peer_stt_available=(
                local_asr_runtime is not None
                and local_asr_runtime.snapshot.channel_for("peer").provider_id is not None
            ),
            qwen_llm_desired=(
                settings.translation.model != TranslationModel.CUSTOM_HTTP
                and settings.provider.llm == LLMProviderName.QWEN
            ),
            llm_available=llm_runtime is not None and llm_runtime.provider is not None,
        )

    async def apply_after_persist(
        self,
        transition: SettingsRuntimeTransition[AppSettings],
        *,
        strict_runtime_errors: bool,
        reload_settings_view: bool,
    ) -> None:
        settings = transition.settings
        await self._desktop_overlay.apply_controls(transition.desktop_runtime_controls)
        previous_strict_runtime_errors = self._clipboard.strict_runtime_errors
        self._clipboard.strict_runtime_errors = strict_runtime_errors
        try:
            await self._clipboard.sync(
                enabled=settings.ui.clipboard_auto_translate_enabled,
            )
        finally:
            self._clipboard.strict_runtime_errors = previous_strict_runtime_errors
        await self._provisioning.inspect_cpu()
        await self._provisioning.inspect_gpu(
            explicit_intent=self._gpu.state_provider().selected_provider_requires_model,
        )
        self._clear_local_pending()

        config_owner = self._pipeline.translation_runtime_configuration
        config_change: TranslationRuntimeConfigChange | None = None
        if config_owner is not None:
            peer_enabled = self._peer.owner.effective_enabled(self._peer.state_for(settings))
            config_change = replace_translation_runtime_settings(
                config_owner,
                project_translation_runtime_settings(settings),
                peer_translation_enabled=peer_enabled,
                integrated_context_enabled=peer_enabled,
            )

        if config_change is not None and config_change.self_language_changed:
            await self._clear_language_runtime_state(
                "self",
                strict_runtime_errors=strict_runtime_errors,
            )
        if config_change is not None and config_change.peer_language_changed:
            await self._clear_language_runtime_state(
                "peer",
                strict_runtime_errors=strict_runtime_errors,
            )

        await refresh_managed_gemma_prefix(
            transition,
            rebuild=self._rebuild_managed_gemma,
        )

        presenter = self._overlay.current_presenter()
        if presenter is not None:
            await presenter.update_display_preferences(
                show_translation=settings.overlay.show_translation,
                show_peer_original=settings.overlay.show_peer_original,
            )

        if transition.previous_overlay_enabled != settings.ui.overlay_enabled:
            await self._overlay.set_enabled(settings.ui.overlay_enabled)

        configure_connection = getattr(self._vrc_mic_sync, "configure_connection", None)
        if callable(configure_connection):
            await configure_connection(
                mode=settings.osc.connection_mode,
                send_port=settings.osc.send_port or settings.osc.port,
                receive_port=settings.osc.receive_port,
                host=settings.osc.host,
            )

        if self._vrc_mic_sync.last_enabled != settings.osc.vrc_mic_intercept:
            self._runtime_logging.emit_detailed(
                f"[Settings] VRC mic sync enabled: {settings.osc.vrc_mic_intercept}"
            )
            await self._vrc_mic_sync.configure(enabled=settings.osc.vrc_mic_intercept)

        current_self_signature = build_self_stt_runtime_signature(settings)
        current_peer_signature = build_peer_stt_runtime_signature(
            settings,
            canonical_settings=self._canonical_settings(settings),
        )
        next_peer_activation_requested = self._peer.owner.activation_requested(
            intent_enabled=settings.ui.peer_translation_enabled,
            eula_accepted=settings.ui.peer_translation_eula_accepted,
        )
        should_restart_stt = (
            transition.previous_self_signature is not None
            and current_self_signature != transition.previous_self_signature
        )
        should_refresh_peer = (
            transition.previous_peer_signature is None
            or current_peer_signature != transition.previous_peer_signature
            or transition.previous_peer_translation_enabled != settings.ui.peer_translation_enabled
            or transition.previous_peer_activation_requested != next_peer_activation_requested
        )

        self._sync_signatures(settings)

        if transition.source_language_changed or transition.target_language_changed:
            self._runtime_logging.emit_detailed(
                "[Settings] Language runtime impact: "
                f"should_restart_stt={should_restart_stt} "
                f"should_refresh_peer={should_refresh_peer} "
                f"prev_overlay_enabled={transition.previous_overlay_enabled} "
                f"next_overlay_enabled={settings.ui.overlay_enabled}"
            )

        if should_refresh_peer and self._pipeline.peer_translation_channel is not None:
            await self._peer.owner.refresh_runtime()
            self._sync_effective_translation_flags(settings)

        if should_restart_stt:
            smooth_local = bool(
                transition.previous_settings is not None
                and build_self_capture_vad_signature(transition.previous_settings)
                == build_self_capture_vad_signature(settings)
            )
            await self._replace_self_stt(smooth_local)

        if reload_settings_view and (
            transition.source_language_changed
            or transition.target_language_changed
            or transition.peer_source_language_changed
            or transition.peer_target_language_changed
            or transition.peer_source_mode_changed
        ):
            self._projection.render(
                settings,
                preserve_custom_vocab_draft=True,
            )

        if transition.previous_locale != settings.ui.locale:
            self._presentation.set_locale(settings.ui.locale)
            try:
                self._presentation.apply_locale()
            except Exception:
                self._runtime_logging.emit_basic("Failed to apply locale")
                if strict_runtime_errors:
                    raise

        self._overlay.publish_presentation()
        publish_delta = getattr(self._vrc_mic_sync, "publish_delta", None)
        if callable(publish_delta):
            publish_delta()

    async def _clear_language_runtime_state(
        self,
        channel: str,
        *,
        strict_runtime_errors: bool,
    ) -> None:
        owner = (
            self._pipeline.self_translation_channel
            if channel == "self"
            else self._pipeline.peer_translation_channel
        )
        if owner is None:
            return
        try:
            if channel == "self":
                await owner.clear_language_runtime_state()
            else:
                await owner.clear_language_runtime_state(channel=channel)
        except Exception as exc:
            if strict_runtime_errors:
                self._runtime_logging.emit_basic(
                    f"Failed to clear language runtime state for {channel}"
                )
            else:
                self._runtime_logging.emit_basic(
                    f"Failed to clear language runtime state for {channel}: {exc}"
                )
            if strict_runtime_errors:
                raise

    def _canonical_settings(self, settings: AppSettings) -> object:
        return self._settings.project(
            settings,
            authoritative=self._settings.authoritative,
        )

    def _sync_signatures(self, settings: AppSettings) -> None:
        self._runtime_signatures.sync(
            settings,
            canonical=self._canonical_settings(settings),
            peer=self._peer.owner,
        )
        self._state.microphone_audio_signature = self._microphone.audio_signature(settings)

    def _sync_effective_translation_flags(self, settings: AppSettings) -> None:
        self._peer.owner.sync_effective_flags(self._peer.state_for(settings))

    @staticmethod
    def _effective_peer_language(language: str, peer_language: str) -> str:
        return peer_language or language


__all__ = [
    "SettingsRuntimeEffectsAdapter",
    "SettingsRuntimeEffectsState",
    "managed_gemma_prefix_refresh_required",
    "refresh_managed_gemma_prefix",
]
