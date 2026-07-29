from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from puripuly_heart.app.ports.application_startup import ApplicationStartupState
from puripuly_heart.app.ports.ui_presentation import UIEventBridgePort, UiPresentationPort
from puripuly_heart.app.services.application_runtime_logging import (
    ApplicationRuntimeLoggingOwner,
)
from puripuly_heart.app.services.application_startup import ApplicationStartupOwner
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.services.gpu_runtime_interaction import GpuRuntimeInteractionState
from puripuly_heart.app.services.manual_local_asr_fallback import (
    ManualLocalASRFallbackOwner,
)
from puripuly_heart.app.services.overlay_calibration_application import (
    OverlayCalibrationApplicationOwner,
)
from puripuly_heart.app.wiring_runtime_pipeline import (
    RuntimePipelineHandle,
    RuntimePipelineLauncher,
)
from puripuly_heart.core.local_asr_provisioning import LocalASRProvisioningPort
from puripuly_heart.core.runtime_logging import SessionLoggingMode


@dataclass(slots=True)
class ApplicationStartupAdapter:
    settings: SettingsOwner
    settings_loader: Callable[[], object]
    provisioning: LocalASRProvisioningPort
    gpu_state: Callable[[], GpuRuntimeInteractionState]
    manual_fallback: ManualLocalASRFallbackOwner
    save_failure_sink: Callable[[BaseException], None]
    calibration: OverlayCalibrationApplicationOwner
    presentation: UiPresentationPort
    sync_presentation: Callable[[], None]
    notify_fallback: Callable[[tuple[str, ...], bool], None]
    runtime_logging: ApplicationRuntimeLoggingOwner
    sync_runtime_signatures: Callable[[object], None]
    pipeline_launcher: RuntimePipelineLauncher
    pipeline: RuntimePipelineHandle
    sync_local_asr_notice: Callable[[], None]
    stt_requires_secret: Callable[[object], bool]
    llm_requires_secret: Callable[[object], bool]
    alibaba_verified_key: Callable[[], str]
    managed_translation_available: Callable[[], bool]
    receiver_active: Callable[[], bool]
    create_event_bridge: Callable[[ApplicationRuntimeLoggingOwner], UIEventBridgePort]
    start_event_bridge: Callable[[UIEventBridgePort], None]
    wait_for_event_bridge: Callable[[], Awaitable[None]]
    sync_clipboard: Callable[[], Awaitable[None]]

    async def prepare_startup_settings(self) -> ApplicationStartupState:
        loaded_settings = self.settings_loader()
        self.settings.current = loaded_settings
        self.settings.authoritative = True
        self.settings.remember_projection(loaded_settings)
        await self.provisioning.inspect_cpu()
        await self.provisioning.inspect_gpu(
            explicit_intent=self.gpu_state().selected_provider_requires_model,
        )
        fallback_plan = self.manual_fallback.plan(
            self.manual_fallback.state(
                loaded_settings,
                cpu_auto_available=self.provisioning.snapshot.cpu_auto_available,
            )
        )
        fallback_channels = fallback_plan.fallback_channels
        installation_fallback = fallback_plan.installation_fallback
        if fallback_plan.changed:
            normalized_settings = self.manual_fallback.apply(
                loaded_settings,
                fallback_plan,
            )
            self.settings.current = normalized_settings
            if not self.settings.save_current(failure_sink=self.save_failure_sink):
                self.settings.current = loaded_settings
                fallback_channels = ()
            else:
                loaded_settings.provider.stt = normalized_settings.provider.stt
                loaded_settings.provider.peer_stt = normalized_settings.provider.peer_stt
                self.settings.current = loaded_settings
        settings = self.settings.current
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
        settings = state.settings
        self.calibration.sync_from_settings(settings)
        self.calibration.replace_draft(None)
        self.presentation.set_locale(settings.ui.locale)
        self.sync_presentation()
        self.notify_fallback(
            state.fallback_channels,
            state.installation_fallback,
        )
        with contextlib.suppress(Exception):
            self.presentation.apply_locale()
        self.runtime_logging.initialize_mode(SessionLoggingMode.BASIC)
        self.presentation.attach_runtime_log_sink(self.runtime_logging)

    async def launch_startup_runtime(self, state: ApplicationStartupState) -> None:
        settings = state.settings
        self.sync_runtime_signatures(settings)
        await self.pipeline_launcher.launch(
            settings,
            vrc_mic_state=self.pipeline.vrc_mic_state,
            vrc_mic_audio_gate=self.pipeline.vrc_mic_audio_gate,
            receiver_active=self.receiver_active(),
        )
        self.sync_local_asr_notice()
        hub = self.pipeline.hub
        if hub is None:
            raise RuntimeError("Application pipeline did not provide a ClientHub")
        stt_provider = settings.provider.stt.value
        if self.stt_requires_secret(settings.provider.stt):
            stt_key_map = {"qwen_asr": self.alibaba_verified_key()}
            stt_verified_key = stt_key_map.get(stt_provider, stt_provider)
            stt_verified = getattr(settings.api_key_verified, stt_verified_key, False)
            stt_needs_key = (not hub.has_stt_provider("self")) or (not stt_verified)
        else:
            stt_needs_key = False
        self.presentation.set_dashboard_stt_needs_key(stt_needs_key)
        llm_provider = settings.provider.llm.value
        if self.llm_requires_secret(settings.provider.llm):
            llm_key_map = {
                "gemini": "google",
                "openrouter": "openrouter",
                "deepseek": "deepseek",
                "qwen": self.alibaba_verified_key(),
            }
            llm_verified_key = llm_key_map.get(llm_provider, llm_provider)
            llm_verified = getattr(settings.api_key_verified, llm_verified_key, False)
            translation_needs_key = (
                False
                if self.managed_translation_available()
                else (hub.llm is None) or (not llm_verified)
            )
        else:
            translation_needs_key = False
        self.presentation.set_dashboard_translation_needs_key(translation_needs_key)
        self.presentation.set_dashboard_translation_enabled(False)
        self.presentation.set_dashboard_stt_enabled(False)
        hub.translation_enabled = False
        await hub.start(auto_flush_osc=True)

    async def start_application_events(self) -> None:
        bridge = self.create_event_bridge(self.runtime_logging)
        self.start_event_bridge(bridge)
        await self.wait_for_event_bridge()
        await self.sync_clipboard()


def compose_application_startup(
    adapter: ApplicationStartupAdapter,
) -> ApplicationStartupOwner:
    return ApplicationStartupOwner(
        settings=adapter,
        presentation=adapter,
        runtime=adapter,
        events=adapter,
    )
