from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, Protocol

from puripuly_heart.app.ports.desktop_overlay import DesktopOverlayRuntimeEffectsPort
from puripuly_heart.app.ports.overlay_calibration import (
    OverlayCalibrationRuntimeEffectsPort,
)
from puripuly_heart.app.ports.settings_runtime_effects import (
    SettingsRuntimeState,
    SettingsRuntimeTransition,
)
from puripuly_heart.app.services.overlay_application import (
    OverlayApplicationOwner,
    OverlayApplicationState,
)
from puripuly_heart.app.wiring_stt_factory import (
    build_peer_stt_runtime_signature,
    build_self_capture_vad_signature,
    build_self_stt_runtime_signature,
)
from puripuly_heart.config.settings import OVERLAY_TARGET_DESKTOP, AppSettings
from puripuly_heart.config.vad_defaults import DEFAULT_STABLE_VAD_HANGOVER_MS
from puripuly_heart.core.translation_policy import FIXED_TRANSLATION_POLICY


class SettingsRuntimeEffectsHost(Protocol):
    app: Any
    hub: Any
    settings: AppSettings | None
    vrc_mic_audio_gate: Any
    _last_microphone_test_audio_settings_signature: object | None
    _last_peer_stt_runtime_signature: object | None
    _last_peer_translation_activation_requested: bool | None
    _last_peer_translation_enabled: bool | None
    _last_self_stt_runtime_signature: object | None
    _last_stt_runtime_signature: object | None
    _last_vrc_mic_sync_enabled: bool | None
    _stt_desired: bool

    async def _preserve_github_star_prompt_observation_before_settings_replace(
        self,
        settings: AppSettings,
    ) -> None: ...

    def _capture_runtime_signatures_before_canonical_mutation(self) -> None: ...

    def _microphone_test_audio_settings_signature(
        self,
        settings: AppSettings | None,
    ) -> object | None: ...

    async def stop_microphone_test_for_audio_settings_change(self) -> None: ...

    def _peer_translation_activation_requested_for(
        self,
        settings: AppSettings,
    ) -> bool: ...

    def _peer_runtime_should_be_active(self, settings: object) -> bool: ...

    def _is_qwen_llm(self, settings: AppSettings) -> bool: ...

    def log_basic(self, message: str) -> None: ...

    def log_detailed(self, message: str) -> None: ...

    async def _sync_clipboard_watcher_with_policy(
        self,
        *,
        strict_runtime_errors: bool,
    ) -> None: ...

    def _get_local_asr_provisioning_owner(self) -> Any: ...

    def _gpu_runtime_interaction_state(self) -> Any: ...

    def _clear_local_stt_pending_enable_if_provider_switched_away(self) -> None: ...

    def _sync_effective_hub_flags(self, settings: AppSettings) -> None: ...

    def _log_error(self, message: str) -> None: ...

    async def set_overlay_enabled(self, enabled: bool) -> None: ...

    async def _configure_vrc_mic_receiver(self, *, enabled: bool) -> None: ...

    def _canonical_vnext_settings_for(self, settings: AppSettings) -> object: ...

    async def _refresh_peer_stt_runtime(self) -> None: ...

    async def _apply_stt_runtime_replacement(self, *, smooth_local: bool) -> None: ...

    def _sync_signature_caches(self, settings: AppSettings) -> None: ...

    def _settings_projection(self) -> Any: ...

    def _refresh_overlay_peer_consumers(self) -> None: ...


class SettingsRuntimeEffectsAdapter:
    def __init__(
        self,
        host: SettingsRuntimeEffectsHost,
        *,
        desktop_overlay: DesktopOverlayRuntimeEffectsPort[AppSettings],
        calibration: OverlayCalibrationRuntimeEffectsPort,
        overlay: OverlayApplicationOwner,
        overlay_state_provider: Callable[[AppSettings | None], OverlayApplicationState],
    ) -> None:
        self._host = host
        self._desktop_overlay = desktop_overlay
        self._calibration = calibration
        self._overlay = overlay
        self._overlay_state_provider = overlay_state_provider

    async def preserve_before_replace(self, settings: AppSettings) -> None:
        await self._host._preserve_github_star_prompt_observation_before_settings_replace(settings)

    def capture_runtime_signatures(self) -> None:
        self._host._capture_runtime_signatures_before_canonical_mutation()

    async def prepare(
        self,
        current_settings: AppSettings | None,
        next_settings: AppSettings,
    ) -> SettingsRuntimeTransition[AppSettings]:
        host = self._host
        previous_microphone_signature = (
            host._last_microphone_test_audio_settings_signature
            or host._microphone_test_audio_settings_signature(current_settings)
        )
        next_microphone_signature = host._microphone_test_audio_settings_signature(next_settings)
        if (
            previous_microphone_signature is not None
            and previous_microphone_signature != next_microphone_signature
        ):
            await host.stop_microphone_test_for_audio_settings_change()

        previous_locale = host.app.current_locale()
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
            host.log_basic(
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
        previous_peer_translation_enabled = (
            host._last_peer_translation_enabled
            if host._last_peer_translation_enabled is not None
            else (
                current_settings.ui.peer_translation_enabled
                if current_settings is not None
                else False
            )
        )
        previous_peer_activation_requested = (
            host._last_peer_translation_activation_requested
            if host._last_peer_translation_activation_requested is not None
            else (
                host._peer_translation_activation_requested_for(current_settings)
                if current_settings is not None
                else False
            )
        )
        previous_self_signature = (
            host._last_self_stt_runtime_signature or host._last_stt_runtime_signature
        )
        previous_peer_signature = host._last_peer_stt_runtime_signature
        previous_source_language = host.hub.source_language if host.hub else None
        previous_target_language = host.hub.target_language if host.hub else None
        previous_peer_source_language = (
            getattr(host.hub, "peer_source_language", None) if host.hub else None
        )
        previous_peer_target_language = (
            getattr(host.hub, "peer_target_language", None) if host.hub else None
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
            host.log_basic(
                "[Settings] Applying languages: "
                f"source={previous_source_language}->{next_settings.languages.source_language} "
                f"target={previous_target_language}->{next_settings.languages.target_language}"
            )
            host.log_detailed(
                "[Settings] Language apply detail: "
                f"overlay_state={self._overlay.snapshot.state} "
                f"presenter_attached={presenter is not None} "
                f"bridge_attached={bridge is not None} "
                "overlay_sink_matches_presenter="
                f"{host.hub is not None and presenter is not None and getattr(host.hub, 'overlay_sink', None) is presenter}"
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
        host = self._host
        settings = transition.settings
        host._last_microphone_test_audio_settings_signature = (
            host._microphone_test_audio_settings_signature(settings)
        )
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
        host = self._host
        restored_settings = copy.deepcopy(settings)
        host.settings = restored_settings
        self._calibration.sync_from_settings(restored_settings)
        if host.hub is not None:
            host.hub.source_language = restored_settings.languages.source_language
            host.hub.target_language = restored_settings.languages.target_language
            host.hub.peer_source_language = restored_settings.languages.peer_source_language
            host.hub.peer_target_language = restored_settings.languages.peer_target_language
            host.hub.system_prompt = restored_settings.system_prompt
            host.hub.low_latency_mode = FIXED_TRANSLATION_POLICY.fast_translation_enabled
            host.hub.low_latency_merge_gap_ms = restored_settings.stt.low_latency_merge_gap_ms
            host.hub.low_latency_spec_retry_max = restored_settings.stt.low_latency_spec_retry_max
            host.hub.hangover_s = (
                restored_settings.stt.low_latency_vad_hangover_ms / 1000.0
                if FIXED_TRANSLATION_POLICY.fast_translation_enabled
                else DEFAULT_STABLE_VAD_HANGOVER_MS / 1000.0
            )
            host.hub.peer_hangover_s = restored_settings.desktop_audio.vad_hangover_ms / 1000.0
            host.hub.chatbox_include_source = restored_settings.osc.chatbox_include_source
            host._sync_effective_hub_flags(restored_settings)
        host._sync_signature_caches(restored_settings)

    def sync_signatures(self, settings: AppSettings) -> None:
        self._host._sync_signature_caches(settings)

    def state(self, settings: AppSettings) -> SettingsRuntimeState:
        host = self._host
        hub = host.hub
        return SettingsRuntimeState(
            runtime_available=hub is not None,
            self_stt_desired=host._stt_desired,
            self_stt_available=hub is not None and hub.has_stt_provider("self"),
            peer_stt_desired=host._peer_runtime_should_be_active(settings),
            peer_stt_available=hub is not None and hub.has_stt_provider("peer"),
            qwen_llm_desired=host._is_qwen_llm(settings),
            llm_available=hub is not None and hub.llm is not None,
        )

    async def apply_after_persist(
        self,
        transition: SettingsRuntimeTransition[AppSettings],
        *,
        strict_runtime_errors: bool,
        reload_settings_view: bool,
    ) -> None:
        host = self._host
        settings = transition.settings
        await self._desktop_overlay.apply_controls(transition.desktop_runtime_controls)
        await host._sync_clipboard_watcher_with_policy(
            strict_runtime_errors=strict_runtime_errors,
        )
        provisioning = host._get_local_asr_provisioning_owner()
        await provisioning.inspect_cpu()
        await provisioning.inspect_gpu(
            explicit_intent=host._gpu_runtime_interaction_state().selected_provider_requires_model,
        )
        host._clear_local_stt_pending_enable_if_provider_switched_away()

        if host.hub is not None:
            host.hub.source_language = settings.languages.source_language
            host.hub.target_language = settings.languages.target_language
            host.hub.peer_source_language = settings.languages.peer_source_language
            host.hub.peer_target_language = settings.languages.peer_target_language
            host.hub.system_prompt = settings.system_prompt
            host.hub.low_latency_mode = FIXED_TRANSLATION_POLICY.fast_translation_enabled
            host.hub.low_latency_merge_gap_ms = settings.stt.low_latency_merge_gap_ms
            host.hub.low_latency_spec_retry_max = settings.stt.low_latency_spec_retry_max
            host.hub.hangover_s = (
                settings.stt.low_latency_vad_hangover_ms / 1000.0
                if FIXED_TRANSLATION_POLICY.fast_translation_enabled
                else DEFAULT_STABLE_VAD_HANGOVER_MS / 1000.0
            )
            host.hub.peer_hangover_s = settings.desktop_audio.vad_hangover_ms / 1000.0
            host.hub.chatbox_include_source = settings.osc.chatbox_include_source
            host._sync_effective_hub_flags(settings)

            if transition.source_language_changed or transition.target_language_changed:
                await self._clear_language_runtime_state(
                    "self",
                    strict_runtime_errors=strict_runtime_errors,
                )
            if transition.effective_peer_source_changed or transition.effective_peer_target_changed:
                await self._clear_language_runtime_state(
                    "peer",
                    strict_runtime_errors=strict_runtime_errors,
                )

        presenter = self._overlay.current_presenter()
        if presenter is not None:
            await presenter.update_display_preferences(
                show_translation=settings.overlay.show_translation,
                show_peer_original=settings.overlay.show_peer_original,
            )

        if transition.previous_overlay_enabled != settings.ui.overlay_enabled:
            await host.set_overlay_enabled(settings.ui.overlay_enabled)

        if host._last_vrc_mic_sync_enabled != settings.osc.vrc_mic_intercept:
            if host.vrc_mic_audio_gate is not None:
                host.vrc_mic_audio_gate.set_enabled(settings.osc.vrc_mic_intercept)
            host.log_detailed(f"[Settings] VRC mic sync enabled: {settings.osc.vrc_mic_intercept}")
            await host._configure_vrc_mic_receiver(enabled=settings.osc.vrc_mic_intercept)

        current_self_signature = build_self_stt_runtime_signature(settings)
        current_peer_signature = build_peer_stt_runtime_signature(
            settings,
            canonical_settings=host._canonical_vnext_settings_for(settings),
        )
        next_peer_activation_requested = host._peer_translation_activation_requested_for(settings)
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

        host._sync_signature_caches(settings)

        if transition.source_language_changed or transition.target_language_changed:
            host.log_detailed(
                "[Settings] Language runtime impact: "
                f"should_restart_stt={should_restart_stt} "
                f"should_refresh_peer={should_refresh_peer} "
                f"prev_overlay_enabled={transition.previous_overlay_enabled} "
                f"next_overlay_enabled={settings.ui.overlay_enabled}"
            )

        if should_refresh_peer and host.hub is not None:
            await host._refresh_peer_stt_runtime()
            host._sync_effective_hub_flags(settings)

        if should_restart_stt:
            smooth_local = bool(
                transition.previous_settings is not None
                and build_self_capture_vad_signature(transition.previous_settings)
                == build_self_capture_vad_signature(settings)
            )
            await host._apply_stt_runtime_replacement(smooth_local=smooth_local)

        if reload_settings_view and (
            transition.source_language_changed
            or transition.target_language_changed
            or transition.peer_source_language_changed
            or transition.peer_target_language_changed
            or transition.peer_source_mode_changed
        ):
            host._settings_projection().render(
                settings,
                preserve_custom_vocab_draft=True,
            )

        if transition.previous_locale != settings.ui.locale:
            host.app.set_locale(settings.ui.locale)
            try:
                host.app.apply_locale()
            except Exception:
                host._log_error("Failed to apply locale")
                if strict_runtime_errors:
                    raise

        host._refresh_overlay_peer_consumers()

    async def _clear_language_runtime_state(
        self,
        channel: str,
        *,
        strict_runtime_errors: bool,
    ) -> None:
        host = self._host
        try:
            await host.hub.clear_language_runtime_state(channel=channel)
        except Exception as exc:
            if strict_runtime_errors:
                host._log_error(f"Failed to clear language runtime state for {channel}")
            else:
                host._log_error(f"Failed to clear language runtime state for {channel}: {exc}")
            if strict_runtime_errors:
                raise

    @staticmethod
    def _effective_peer_language(language: str, peer_language: str) -> str:
        return peer_language or language


__all__ = [
    "SettingsRuntimeEffectsAdapter",
    "SettingsRuntimeEffectsHost",
]
