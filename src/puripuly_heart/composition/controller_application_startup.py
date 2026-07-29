from __future__ import annotations

import contextlib

from puripuly_heart.app.ports.application_startup import ApplicationStartupState
from puripuly_heart.app.services.application_startup import ApplicationStartupOwner
from puripuly_heart.core.runtime_logging import SessionLoggingMode
from puripuly_heart.ui.controller import GuiController


class ControllerApplicationStartupAdapter:
    def __init__(self, runtime: GuiController) -> None:
        self._runtime = runtime

    async def prepare_startup_settings(self) -> ApplicationStartupState:
        runtime = self._runtime
        runtime.settings = runtime._load_or_init_settings(runtime.config_path)
        settings_owner = runtime._get_settings_owner()
        settings_owner.authoritative = True
        settings_owner.remember_projection(runtime.settings)
        provisioning = runtime._get_local_asr_provisioning_owner()
        await provisioning.inspect_cpu()
        await provisioning.inspect_gpu(
            explicit_intent=(
                runtime._gpu_runtime_interaction_state().selected_provider_requires_model
            ),
        )
        loaded_settings = runtime.settings
        if loaded_settings is None:
            raise RuntimeError("Application settings were not loaded")
        fallback_plan = runtime.manual_local_asr_fallback_owner.plan(
            runtime.manual_local_asr_fallback_owner.state(
                loaded_settings,
                cpu_auto_available=provisioning.snapshot.cpu_auto_available,
            )
        )
        fallback_channels = fallback_plan.fallback_channels
        installation_fallback = fallback_plan.installation_fallback
        if fallback_plan.changed:
            normalized_settings = runtime.manual_local_asr_fallback_owner.apply(
                loaded_settings,
                fallback_plan,
            )
            runtime.settings = normalized_settings
            if not settings_owner.save_current(
                failure_sink=lambda exc: runtime._log_error(f"Failed to save settings: {exc}")
            ):
                runtime.settings = loaded_settings
                fallback_channels = ()
            else:
                loaded_settings.provider.stt = normalized_settings.provider.stt
                loaded_settings.provider.peer_stt = normalized_settings.provider.peer_stt
                runtime.settings = loaded_settings
        settings = runtime.settings
        if settings is None:
            raise RuntimeError("Application settings were not retained")
        settings.ui.overlay_enabled = False
        settings.ui.peer_translation_enabled = False
        return ApplicationStartupState(
            settings=settings,
            fallback_channels=fallback_channels,
            installation_fallback=installation_fallback,
        )

    def apply_startup_presentation(self, state: ApplicationStartupState) -> None:
        runtime = self._runtime
        settings = state.settings
        calibration = runtime._get_overlay_calibration_application_owner()
        calibration.sync_from_settings(settings)
        calibration.replace_draft(None)
        runtime.app.set_locale(settings.ui.locale)
        runtime._sync_ui_from_settings()
        runtime._get_settings_application_owner().notify_fallback(
            state.fallback_channels,
            state.installation_fallback,
        )
        with contextlib.suppress(Exception):
            runtime.app.apply_locale()
        runtime_logging = runtime.runtime_logging
        runtime_logging.set_mode(SessionLoggingMode.BASIC)
        runtime.app.attach_runtime_log_sink(runtime_logging)

    async def launch_startup_runtime(self, state: ApplicationStartupState) -> None:
        runtime = self._runtime
        settings = state.settings
        runtime._sync_signature_caches(settings)
        await runtime.runtime_composition.pipeline_launcher.launch(
            settings,
            vrc_mic_state=runtime.vrc_mic_state,
            vrc_mic_audio_gate=runtime.vrc_mic_audio_gate,
            receiver_active=runtime.receiver is not None,
        )
        runtime._get_local_asr_application_runtime().adapters.notice.sync()
        hub = runtime.hub
        if hub is None:
            raise RuntimeError("Application pipeline did not provide a ClientHub")
        stt_provider = settings.provider.stt.value
        if runtime._stt_provider_requires_secret(settings.provider.stt):
            stt_key_map = {"qwen_asr": runtime._get_alibaba_verified_key()}
            stt_verified_key = stt_key_map.get(stt_provider, stt_provider)
            stt_verified = getattr(
                settings.api_key_verified,
                stt_verified_key,
                False,
            )
            stt_needs_key = (not runtime._hub_has_stt_provider("self")) or (not stt_verified)
        else:
            stt_needs_key = False
        runtime.app.set_dashboard_stt_needs_key(stt_needs_key)
        llm_provider = settings.provider.llm.value
        if runtime._llm_provider_requires_secret(settings.provider.llm):
            llm_key_map = {
                "gemini": "google",
                "openrouter": "openrouter",
                "deepseek": "deepseek",
                "qwen": runtime._get_alibaba_verified_key(),
            }
            llm_verified_key = llm_key_map.get(llm_provider, llm_provider)
            llm_verified = getattr(
                settings.api_key_verified,
                llm_verified_key,
                False,
            )
            translation_needs_key = (
                False
                if runtime._managed_openrouter_can_attempt_translation()
                else (hub.llm is None) or (not llm_verified)
            )
        else:
            translation_needs_key = False
        runtime.app.set_dashboard_translation_needs_key(translation_needs_key)
        runtime.app.set_dashboard_translation_enabled(False)
        runtime.app.set_dashboard_stt_enabled(False)
        hub.translation_enabled = False
        await hub.start(auto_flush_osc=True)

    async def start_application_events(self) -> None:
        runtime = self._runtime
        bridge = runtime._create_ui_event_bridge(
            runtime_logging=runtime.runtime_logging,
        )
        runtime._start_ui_event_bridge_task(bridge)
        await runtime._wait_for_ui_event_bridge_started()
        await runtime._sync_clipboard_watcher()


def compose_controller_application_startup(
    runtime: GuiController,
) -> ApplicationStartupOwner:
    adapter = ControllerApplicationStartupAdapter(runtime)
    return ApplicationStartupOwner(
        settings=adapter,
        presentation=adapter,
        runtime=adapter,
        events=adapter,
    )
