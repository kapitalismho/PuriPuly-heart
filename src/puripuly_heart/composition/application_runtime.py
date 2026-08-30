from __future__ import annotations

import contextlib
import copy
import logging
import sys
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

from puripuly_heart.app.adapters.application_runtime_shutdown import (
    ApplicationRuntimeShutdownAdapter,
)
from puripuly_heart.app.adapters.local_asr_application import LocalASRApplicationSettings
from puripuly_heart.app.adapters.peer_application_state import PeerApplicationSettings
from puripuly_heart.app.adapters.system_directory_opener import SystemDirectoryOpener
from puripuly_heart.app.adapters.ui_runtime import (
    UiDiagnosticsRuntimeAdapter,
    UiEngagementRuntimeAdapter,
    UiInputRuntimeAdapter,
    UiManagedRuntimeAdapter,
    UiMicrophoneRuntimeAdapter,
    UiOverlayRuntimeAdapter,
    UiPeerCaptureRuntimeAdapter,
    UiProviderRuntimeAdapter,
    UiSettingsRuntimeAdapter,
)
from puripuly_heart.app.ports.capture_vad_runtime import (
    PeerCaptureVadEventRuntime,
    SelfCaptureVadEventRuntime,
)
from puripuly_heart.app.ports.local_asr_production_evidence import (
    LocalASRProductionCompositionAccessPort,
)
from puripuly_heart.app.ports.provider_channel_runtime import ProviderChannelResetPort
from puripuly_heart.app.ports.provider_verifier import ProviderVerifierPort
from puripuly_heart.app.ports.runtime_pipeline_lifecycle import (
    RuntimePipelineStartCallbacks,
)
from puripuly_heart.app.ports.settings_secrets import SettingsSecretStorePort
from puripuly_heart.app.ports.ui_application import UiApplicationPort
from puripuly_heart.app.ports.ui_models import (
    ManagedGemmaDashboardNotice,
    OscControlPresentationName,
    OscControlPresentationState,
)
from puripuly_heart.app.ports.ui_presentation import UIEventBridgePort, UiPresentationPort
from puripuly_heart.app.ports.vrchat_osc_presence import VrchatOscPresencePort
from puripuly_heart.app.services.application_after_launch import (
    ApplicationAfterLaunchOwner,
)
from puripuly_heart.app.services.application_ingress import ApplicationIngressGate
from puripuly_heart.app.services.audio_diagnostics_application import (
    AudioDiagnosticsApplicationOwner,
)
from puripuly_heart.app.services.canonical_settings_persistence import (
    compose_settings_owner,
)
from puripuly_heart.app.services.clipboard_auto_translation import (
    ClipboardAutoTranslationOwner,
)
from puripuly_heart.app.services.desktop_overlay_application import (
    DESKTOP_INTERACTION_MODE_EDIT,
    DesktopOverlayApplicationOwner,
)
from puripuly_heart.app.services.github_star_prompt import GithubStarPromptOwner
from puripuly_heart.app.services.github_star_prompt_settings import (
    compose_github_star_prompt_owner,
)
from puripuly_heart.app.services.gpu_provider_recovery import (
    GpuProviderRecoveryDiagnostic,
)
from puripuly_heart.app.services.gpu_provider_recovery_application import (
    GpuProviderRecoveryApplicationOwner,
    GpuProviderRecoveryApplicationRequest,
)
from puripuly_heart.app.services.gpu_runtime_interaction import (
    GpuRuntimeInteractionOwner,
    GpuRuntimeInteractionState,
)
from puripuly_heart.app.services.http_extension_registry import (
    HttpExtensionRegistryService,
)
from puripuly_heart.app.services.local_asr_diagnostics import LocalASRDiagnosticsOwner
from puripuly_heart.app.services.local_asr_gpu_provisioning import (
    LocalASRGpuProvisioningDiagnostic,
)
from puripuly_heart.app.services.local_asr_selection import (
    LOCAL_CPU_PROVIDERS,
    resolve_local_asr_selection,
)
from puripuly_heart.app.services.managed_gemma_translation import (
    ManagedGemmaTranslationOwner,
    ManagedGemmaTranslationSnapshot,
)
from puripuly_heart.app.services.manual_local_asr_fallback import (
    ManualLocalASRFallbackOwner,
)
from puripuly_heart.app.services.manual_typing import ManualTypingOwner
from puripuly_heart.app.services.osc.control_runtime import OscControlIntegrationOwner
from puripuly_heart.app.services.osc.state_publisher import state_from_settings
from puripuly_heart.app.services.overlay_application import (
    OverlayApplicationOwner,
    OverlayApplicationState,
)
from puripuly_heart.app.services.overlay_calibration_application import (
    OverlayCalibrationApplicationOwner,
)
from puripuly_heart.app.services.provider_credential_verification import (
    ProviderCredentialVerificationInteractionOwner,
)
from puripuly_heart.app.services.provider_runtime_apply import ProviderRuntimeApplyPlan
from puripuly_heart.app.services.provider_settings import (
    ProviderApplicationOwner,
    ProviderSettingsOwner,
    provider_verification_context,
)
from puripuly_heart.app.services.provider_verification_binding import (
    ProviderVerificationBindingOwner,
)
from puripuly_heart.app.services.self_capture_application import (
    SelfCaptureApplicationOwner,
    SelfCaptureApplicationSettings,
)
from puripuly_heart.app.services.settings_application import (
    SettingsApplicationOwner,
    osc_control_presentation_state,
)
from puripuly_heart.app.services.settings_projection import SettingsProjectionOwner
from puripuly_heart.app.services.settings_runtime_effects import (
    SettingsRuntimeEffectsAdapter,
    SettingsRuntimeEffectsState,
)
from puripuly_heart.app.services.settings_secrets import SettingsSecretsOwner
from puripuly_heart.app.services.ui_application import UiApplicationBoundary
from puripuly_heart.app.services.ui_application_state import UiApplicationStateOwner
from puripuly_heart.app.wiring import (
    LocalASRProviderRuntimeFactory,
    ManagedSTTProviderFactory,
    build_peer_capture_session_config_from_vnext,
    build_peer_stt_provider_request,
    build_peer_stt_runtime_signature_from_vnext,
    build_self_capture_session_config_from_vnext,
    build_self_stt_provider_request_from_vnext,
    build_self_stt_runtime_signature_from_vnext,
    create_local_asr_provisioning_owner,
    create_provider_verifier,
    create_secret_store,
    create_self_capture_admission_adapter,
    create_sync_secret_store_adapter,
    resolve_overlay_config_from_vnext,
    runtime_pipeline_inputs_from_vnext,
)
from puripuly_heart.app.wiring.wiring_managed_gemma import (
    create_managed_gemma_runtime,
    managed_gemma_selection,
    managed_gemma_translation_desired,
    sync_managed_gemma_demand,
)
from puripuly_heart.app.wiring.wiring_provider_runtime_policy import (
    provider_llm_for_translation,
)
from puripuly_heart.app.wiring_application_runtime_logging import (
    compose_application_runtime_logging,
)
from puripuly_heart.app.wiring_capture_runtime import CaptureOwnerFactory
from puripuly_heart.app.wiring_composition import (
    create_desktop_overlay_policy,
    create_gpu_provider_recovery_application_owner,
    create_gpu_runtime_interaction_owner,
    create_local_asr_diagnostics_owner,
    create_manual_typing_owner,
    create_provider_credential_verification_interaction_owner,
    create_vrchat_osc_presence_probe_owner,
    create_windows_desktop_work_area,
)
from puripuly_heart.app.wiring_local_asr_application import (
    LocalASRApplicationRuntime,
    compose_local_asr_application,
)
from puripuly_heart.app.wiring_managed_account import (
    ManagedAccountComponents,
    ManagedOpenRouterReleaseRuntime,
    ManagedTranslationRuntimeAccess,
    compose_managed_account,
)
from puripuly_heart.app.wiring_microphone_test import (
    MicrophoneTestAudioSettings,
    MicrophoneTestRuntime,
)
from puripuly_heart.app.wiring_peer_application import (
    PeerApplicationRuntime,
    compose_peer_application,
)
from puripuly_heart.app.wiring_provider_runtime import (
    ProviderRuntimeComponents,
    ProviderRuntimeSignatures,
    compose_provider_runtime,
)
from puripuly_heart.app.wiring_runtime_composition import RuntimeCompositionComponents
from puripuly_heart.app.wiring_runtime_pipeline import (
    RuntimePipelineComponents,
    RuntimePipelineHandle,
    RuntimePipelineLauncher,
)
from puripuly_heart.app.wiring_vrc_mic_sync import compose_vrc_mic_sync
from puripuly_heart.composition.application_settings import load_application_settings
from puripuly_heart.composition.application_startup import (
    ApplicationStartupAdapter,
    compose_application_startup,
)
from puripuly_heart.composition.application_state import ApplicationUiStateAdapter
from puripuly_heart.config.paths import default_http_extensions_dir, user_config_dir
from puripuly_heart.config.provider_values import (
    STT_INTERNAL_SAMPLE_RATE_HZ,
    LLMProviderName,
    OpenRouterCredentialSource,
    QwenLLMModel,
    QwenRegion,
    STTProviderName,
)
from puripuly_heart.config.resolved import OVERLAY_TARGET_STEAMVR
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.config.translation_values import TranslationModel
from puripuly_heart.core.clipboard.watcher import create_clipboard_watcher
from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.http_extensions import HttpExtensionRegistry
from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimePort,
)
from puripuly_heart.core.local_asr_provisioning import (
    LocalASRProvisioningDiagnostic,
    LocalASRProvisioningPort,
    LocalASRProvisioningSnapshot,
)
from puripuly_heart.core.local_gpu_assets import local_gpu_model_path
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfigurationOwner,
)
from puripuly_heart.core.peer_capture import PeerCaptureSessionSnapshot
from puripuly_heart.core.runtime.gpu_asr import GpuASRChannel
from puripuly_heart.core.runtime.local_asr_provider_runtime import (
    LocalASRProviderRuntimeOwner,
)
from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle
from puripuly_heart.core.runtime.self_capture import SelfCaptureSessionOwner
from puripuly_heart.core.runtime.vrchat_osc_presence import (
    VrchatOscPresenceProbeOwner,
)
from puripuly_heart.core.runtime_logging import RuntimeLoggingSinks
from puripuly_heart.core.self_capture import (
    SelfCaptureSessionSnapshot,
    SelfCaptureSessionState,
)
from puripuly_heart.core.telemetry import (
    AppActiveDayTelemetryService,
    HttpAppActiveDayTelemetryClient,
)
from puripuly_heart.core.translation_policy import FIXED_TRANSLATION_POLICY

STT_RESET_DEADLINE_S = 300.0
MANUAL_INPUT_TYPING_IDLE_TIMEOUT_S = 3.0
MANUAL_SUBMIT_TYPING_TIMEOUT_S = 10.0


def _require_self_capture_owner(
    pipeline: RuntimePipelineHandle,
    capture_factory: CaptureOwnerFactory,
) -> SelfCaptureSessionOwner:
    owner = pipeline.self_capture
    if owner is not None:
        return owner
    self_translation_channel = pipeline.self_translation_channel
    local_asr_runtime = pipeline.local_asr_runtime
    audio_gate = pipeline.vrc_mic_audio_gate
    if self_translation_channel is None:
        raise RuntimeError("Self translation channel owner is unavailable")
    if local_asr_runtime is None:
        raise RuntimeError("Local ASR runtime owner is unavailable")
    if audio_gate is None:
        raise RuntimeError("VRChat microphone audio gate is unavailable")
    owner = capture_factory.compose_self(
        self_translation_channel,
        local_asr_runtime,
        self_translation_channel,
        audio_gate,
    )
    pipeline.self_capture = owner
    return owner


@dataclass(frozen=True, slots=True)
class _LocalASRProductionCompositionAccess:
    config_path: Path
    settings_loader: Callable[[], AppSettingsVNext]
    runtime_initializer: Callable[[AppSettingsVNext], Awaitable[None]]
    components_provider: Callable[[], RuntimePipelineComponents | None]
    gpu_retry: Callable[[], Awaitable[None]]

    def load_compatibility_settings(self) -> AppSettingsVNext:
        return self.settings_loader()

    async def initialize(self, settings: AppSettingsVNext) -> None:
        await self.runtime_initializer(settings)

    @property
    def _components(self) -> RuntimePipelineComponents:
        components = self.components_provider()
        if components is None:
            raise RuntimeError("production application did not compose runtime components")
        return components

    @property
    def owner(self) -> LocalASRProviderRuntimeOwner:
        owner = self._components.local_asr_runtime
        if not isinstance(owner, LocalASRProviderRuntimeOwner):
            raise RuntimeError("production application did not compose the canonical owner")
        return owner

    @property
    def llm_runtime(self) -> ProviderRuntimeHandle:
        return self._components.llm_runtime

    @property
    def translation_runtime_configuration(self) -> TranslationRuntimeConfigurationOwner:
        return self._components.translation_runtime_configuration

    @property
    def self_vad(self) -> SelfCaptureVadEventRuntime:
        return self._components.self_translation_channel

    @property
    def peer_vad(self) -> PeerCaptureVadEventRuntime:
        return self._components.peer_translation_channel

    @property
    def channel_reset(self) -> ProviderChannelResetPort:
        return self._components.channel_reset

    @property
    def start_callbacks(self) -> RuntimePipelineStartCallbacks:
        return self._components.start_callbacks

    async def retry_gpu_activation(self) -> None:
        await self.gpu_retry()


def _copy_provider_prompt_apply_fields(
    source: AppSettingsVNext,
    target: AppSettingsVNext,
) -> AppSettingsVNext:
    source_managed = source.state.managed_connection
    if source.intent.translation.openrouter_selected_source == "managed":
        next_managed = replace(
            target.state.managed_connection,
            verified_hardware_hash=source_managed.verified_hardware_hash,
            verified_hardware_hash_salt_version=source_managed.verified_hardware_hash_salt_version,
        )
    else:
        next_managed = replace(
            target.state.managed_connection,
            verified_hardware_hash=None,
            verified_hardware_hash_salt_version=None,
        )
    return replace(
        target,
        intent=replace(
            target.intent,
            stt=replace(target.intent.stt, provider=source.intent.stt.provider),
            peer_stt=replace(target.intent.peer_stt, provider=source.intent.peer_stt.provider),
            translation=copy.deepcopy(source.intent.translation),
            local_llm=copy.deepcopy(source.intent.local_llm),
            prompts=replace(
                target.intent.prompts,
                system_prompt=source.intent.prompts.system_prompt,
            ),
        ),
        state=replace(target.state, managed_connection=next_managed),
    )


def compose_application_runtime(
    *,
    presentation: UiPresentationPort,
    config_path: Path,
    runtime_logging_sinks: RuntimeLoggingSinks | None = None,
    vrchat_osc_presence: VrchatOscPresencePort | None = None,
    local_asr_evidence_sink: (
        Callable[[LocalASRProductionCompositionAccessPort], None] | None
    ) = None,
) -> UiApplicationPort:
    settings = compose_settings_owner(config_path)
    http_extensions = HttpExtensionRegistry(default_http_extensions_dir())
    http_extensions.reload()
    clock = SystemClock()
    ingress = ApplicationIngressGate()
    pipeline = RuntimePipelineHandle()
    signatures = ProviderRuntimeSignatures(http_extensions=http_extensions)
    effects_state = SettingsRuntimeEffectsState()
    manual_fallback = ManualLocalASRFallbackOwner()
    event_bridge: UIEventBridgePort | None = None
    bridge_task = None
    overlay: OverlayApplicationOwner | None = None
    desktop: DesktopOverlayApplicationOwner | None = None
    calibration: OverlayCalibrationApplicationOwner | None = None
    peer: PeerApplicationRuntime | None = None
    local_asr: LocalASRApplicationRuntime | None = None
    provisioning: LocalASRProvisioningPort | None = None
    gpu: GpuRuntimeInteractionOwner | None = None
    gpu_recovery: GpuProviderRecoveryApplicationOwner | None = None
    local_asr_diagnostics: LocalASRDiagnosticsOwner | None = None
    audio_diagnostics: AudioDiagnosticsApplicationOwner | None = None
    self_application: SelfCaptureApplicationOwner | None = None
    microphone: MicrophoneTestRuntime | None = None
    vrc_mic_sync: OscControlIntegrationOwner | None = None
    manual_typing: ManualTypingOwner | None = None
    clipboard: ClipboardAutoTranslationOwner | None = None
    settings_projection: SettingsProjectionOwner | None = None
    settings_application: SettingsApplicationOwner | None = None
    provider_verifier: ProviderVerifierPort | None = None
    provider_settings: ProviderSettingsOwner | None = None
    credential_verification: ProviderCredentialVerificationInteractionOwner | None = None
    github_prompt: GithubStarPromptOwner | None = None
    vrchat_presence: VrchatOscPresenceProbeOwner | None = None
    managed_account: ManagedAccountComponents | None = None
    runtime_components: RuntimeCompositionComponents | None = None

    def current_settings() -> AppSettingsVNext | None:
        return settings.canonical

    def create_settings_secret_store() -> SettingsSecretStorePort:
        canonical = settings.canonical
        if canonical is None:
            raise RuntimeError("Settings are not loaded")
        return create_secret_store(canonical.intent.secrets, config_path=config_path)

    def canonical_settings(value: AppSettingsVNext) -> AppSettingsVNext:
        return settings.project(value, authoritative=True)

    def peer_application_settings() -> PeerApplicationSettings | None:
        canonical = settings.canonical
        if canonical is None:
            return None
        return PeerApplicationSettings(
            intent_enabled=settings.peer_translation_enabled(),
            eula_accepted=canonical.state.peer_translation.eula_accepted,
            overlay_intent_enabled=settings.overlay_enabled(),
            provider_id=canonical.intent.peer_stt.provider,
        )

    def microphone_audio_settings() -> MicrophoneTestAudioSettings | None:
        canonical = settings.canonical
        if canonical is None:
            return None
        return MicrophoneTestAudioSettings(
            input_host_api=canonical.intent.audio.input_host_api,
            input_device=canonical.intent.audio.input_device,
            internal_sample_rate_hz=STT_INTERNAL_SAMPLE_RATE_HZ,
            internal_channels=1,
        )

    def current_canonical() -> AppSettingsVNext:
        canonical = settings.canonical
        if canonical is None:
            raise RuntimeError("Settings are not loaded")
        return canonical

    def log_basic(message: str, *, level: int = logging.INFO) -> None:
        runtime_logging.emit_basic(message, level=level)

    def log_detailed(
        message: str,
        *,
        level: int = logging.INFO,
        exception: BaseException | None = None,
    ) -> bool:
        return runtime_logging.emit_detailed(
            message,
            level=level,
            exception=exception,
        )

    def log_error(message: str) -> None:
        log_basic(message, level=logging.ERROR)

    async def emit_overlay_logging_mode_update() -> None:
        owner = overlay
        bridge = owner.current_bridge() if owner is not None else None
        if bridge is not None:
            await bridge.broadcast_runtime_control(logging_mode=runtime_logging.mode)

    runtime_logging = compose_application_runtime_logging(
        presentation=presentation,
        sinks=runtime_logging_sinks,
        overlay_logging_mode_update=emit_overlay_logging_mode_update,
        overlay_logging_mode_update_available=lambda: (
            overlay is not None and overlay.current_bridge() is not None
        ),
    )

    def managed_gemma_status(snapshot: ManagedGemmaTranslationSnapshot) -> None:
        fields = [f"state={snapshot.state}"]
        if snapshot.backend is not None:
            fields.append(f"backend={snapshot.backend}")
        if snapshot.progress_percent is not None:
            fields.append(f"progress_percent={snapshot.progress_percent}")
        if snapshot.error_type is not None:
            fields.append(f"error_type={snapshot.error_type}")
        log_detailed("[ManagedGemma] " + " ".join(fields))
        if snapshot.state not in {
            "checking",
            "downloading",
            "preparing",
            "failed",
            "cancelled",
        }:
            presentation.set_dashboard_managed_gemma_notice(None)
            return
        presentation.set_dashboard_managed_gemma_notice(
            ManagedGemmaDashboardNotice(
                status=snapshot.state,
                backend=snapshot.backend,
                progress_percent=snapshot.progress_percent,
            )
        )

    managed_gemma = ManagedGemmaTranslationOwner(
        runtime=create_managed_gemma_runtime(
            log_sink=lambda message, level: log_detailed(message, level=level),
        ),
        status_sink=managed_gemma_status,
        lifecycle_diagnostic_sink=lambda event: log_detailed(
            "[ManagedGemma] lifecycle_diagnostic "
            + " ".join(f"{key}={value}" for key, value in event.fields.items()),
            level=logging.ERROR,
        ),
    )

    def _managed_gemma_demand() -> tuple[bool, object | None]:
        settings_value = current_settings()
        config = pipeline.translation_runtime_configuration
        desired = managed_gemma_translation_desired(
            translation_enabled=bool(
                config is not None and config.snapshot().value.translation_enabled
            ),
            peer_translation_enabled=bool(settings.peer_translation_enabled()),
        )
        return desired, settings_value

    async def sync_local_translation_demand() -> None:
        desired, settings_value = _managed_gemma_demand()
        await sync_managed_gemma_demand(
            managed_gemma=managed_gemma,
            settings=settings_value,
            desired=desired,
        )

    def schedule_local_translation_demand() -> None:
        desired, settings_value = _managed_gemma_demand()
        selection = None
        if desired and settings_value is not None:
            with contextlib.suppress(ValueError):
                selection = managed_gemma_selection(settings_value)
        managed_gemma.schedule_demand_sync(desired=desired, selection=selection)

    def disable_peer_intent() -> None:
        require_peer().owner.disable_for_overlay()
        schedule_local_translation_demand()

    def require_runtime_components() -> RuntimeCompositionComponents:
        if runtime_components is None:
            raise RuntimeError("runtime composition is incomplete")
        return runtime_components

    def overlay_state(
        canonical: AppSettingsVNext | None = None,
        *,
        overlay_enabled: bool | None = None,
    ) -> OverlayApplicationState:
        resolved = settings.canonical if canonical is None else canonical
        enabled = settings.overlay_enabled() if overlay_enabled is None else overlay_enabled
        return OverlayApplicationState(
            settings_available=resolved is not None,
            overlay_intent_enabled=enabled,
            configured_target=(
                resolved.intent.overlay.target if resolved is not None else OVERLAY_TARGET_STEAMVR
            ),
            locale=resolved.intent.ui.locale if resolved is not None else "en",
        )

    def report_overlay_state(state: str, failure_reason: str | None) -> None:
        if event_bridge is not None:
            event_bridge.report_overlay_state(state, failure_reason=failure_reason)

    def refresh_overlay_presentation() -> None:
        require_overlay().publish_presentation()

    def sync_effective_flags() -> None:
        if settings.canonical is None:
            return
        runtime = require_peer()
        runtime.owner.sync_effective_flags(runtime.state_for(peer_application_settings()))

    async def refresh_overlay_runtime_dependencies(
        *,
        peer_stop_mode: str = "retain",
    ) -> None:
        await require_peer().owner.refresh_runtime(
            stop_mode="release" if peer_stop_mode == "release" else "retain"
        )
        sync_effective_flags()
        refresh_overlay_presentation()

    def require_desktop() -> DesktopOverlayApplicationOwner:
        nonlocal desktop
        if desktop is None:
            desktop = DesktopOverlayApplicationOwner(
                settings=settings,
                settings_application_provider=require_settings_application,
                overlay_provider=require_overlay,
                work_area=create_windows_desktop_work_area(),
                policy=create_desktop_overlay_policy(),
                presentation_sink=lambda mode, locked: (
                    presentation.on_desktop_overlay_state_changed(
                        interaction_mode=mode,
                        captions_locked=locked,
                    )
                ),
                log_detailed=lambda message, level, exception: log_detailed(
                    message,
                    level=level,
                    exception=exception,
                ),
            )
        return desktop

    def require_calibration() -> OverlayCalibrationApplicationOwner:
        nonlocal calibration
        if calibration is None:
            calibration = OverlayCalibrationApplicationOwner(
                settings=settings,
                settings_application_provider=require_settings_application,
                overlay_provider=require_overlay,
                schedule_task=presentation.schedule_task,
                log_detailed=log_detailed,
                ingress_available=lambda: not ingress.frozen,
            )
        return calibration

    def require_overlay() -> OverlayApplicationOwner:
        nonlocal overlay
        if overlay is None:
            desktop_owner = require_desktop()
            calibration_owner = require_calibration()
            overlay = OverlayApplicationOwner(
                state_provider=overlay_state,
                config_provider=lambda: resolve_overlay_config_from_vnext(
                    current_canonical(),
                    enabled=settings.overlay_enabled(),
                    locked=settings.overlay_desktop_locked(),
                ),
                overlay_intent_sink=settings.set_overlay_enabled,
                output_provider=lambda: pipeline.translation_output_projection,
                diagnostics_provider=lambda: pipeline.translation_diagnostics,
                peer_snapshot_provider=lambda: require_peer().owner.snapshot(),
                disable_peer_intent=disable_peer_intent,
                sync_peer_effective=lambda: require_peer().owner.sync_effective_flags(),
                cancel_peer_activation=(lambda: require_peer().owner.cancel_activation_starting()),
                refresh_peer_dependencies=refresh_overlay_runtime_dependencies,
                presentation_sink=presentation.refresh_overlay_peer_contract,
                state_sink=report_overlay_state,
                fallback_notice_sink=(presentation.set_dashboard_overlay_session_fallback_notice),
                cancel_bounds_persistence=desktop_owner.bounds_owner.cancel,
                clear_bounds_suppressed=desktop_owner.bounds_owner.clear_suppressed,
                calibration_provider=lambda: calibration_owner.current.copy(),
                logging_mode_provider=lambda: runtime_logging.mode,
                log_dir_provider=lambda: str(user_config_dir()),
                desktop_controls_factory=desktop_owner.initial_controls,
                interaction_mode_sink=desktop_owner.set_interaction_mode,
                bounds_control_sink=desktop_owner.bounds_owner.track_apply_control,
                renderer_event_consumer=desktop_owner.consume_renderer_events,
                edit_interaction_mode=DESKTOP_INTERACTION_MODE_EDIT,
                clock=clock,
                log_basic=lambda message, level: log_basic(message, level=level),
                log_detailed=lambda message, level, exception: log_detailed(
                    message,
                    level=level,
                    exception=exception,
                ),
            )
        return overlay

    def peer_activation_requested() -> bool:
        peer_settings = peer_application_settings()
        if peer_settings is None:
            return False
        runtime = require_peer()
        return runtime.owner.activation_requested(
            intent_enabled=peer_settings.intent_enabled,
            eula_accepted=peer_settings.eula_accepted,
        )

    def peer_runtime_desired() -> bool:
        runtime = require_peer()
        return runtime.owner.desired_active(runtime.state_for(peer_application_settings()))

    def peer_local_stt_requested() -> bool:
        runtime = require_peer()
        return runtime.owner.local_stt_requested(runtime.state_for(peer_application_settings()))

    def enqueue_peer_disclosure() -> None:
        output_projection = pipeline.translation_output_projection
        if output_projection is None:
            return
        output_projection.publish_system_disclosure(
            presentation.localize("peer_translation.disclosure")
        )

    def require_peer() -> PeerApplicationRuntime:
        nonlocal peer
        if peer is None:
            peer = compose_peer_application(
                settings_provider=peer_application_settings,
                settings_owner=settings,
                canonical_settings=current_canonical,
                peer_intent_sink=settings.set_peer_translation_enabled,
                overlay_intent_sink=settings.set_overlay_enabled,
                runtime_provider=lambda: pipeline.local_asr_runtime,
                translation_runtime_configuration_provider=(
                    lambda: pipeline.translation_runtime_configuration
                ),
                overlay_provider=require_overlay,
                ingress_frozen=lambda: ingress.frozen,
                persist_manual_fallback=lambda: (
                    require_settings_application().persist_manual_fallback(channel="peer")
                ),
                ensure_local_ready=lambda generation: require_local_asr().ensure_peer_ready(
                    activation_generation=generation,
                ),
                clear_cpu_pending=lambda: require_local_asr().cpu_repair.reset_peer(),
                clear_gpu_pending=lambda: require_gpu().clear_pending("peer"),
                clear_switched_pending=lambda: (
                    require_local_asr().cpu_repair.clear_if_provider_switched_away()
                ),
                sync_local_notice=lambda: require_local_asr().adapters.notice.sync(),
                presentation_changed=refresh_overlay_presentation,
                disclosure_sink=enqueue_peer_disclosure,
                superseded_sink=lambda: signatures.mark_superseded(settings.canonical),
                localize=presentation.localize,
                settings_presentation_sink=(presentation.refresh_settings_loopback_capture_target),
                log_basic=log_basic,
                log_detailed=log_detailed,
                translation_demand_sink=sync_local_translation_demand,
            )
        return peer

    def gpu_state() -> GpuRuntimeInteractionState:
        value = current_settings()
        return GpuRuntimeInteractionState(
            settings_available=value is not None,
            selected_provider_requires_model=bool(
                value is not None
                and (
                    value.intent.stt.provider == STTProviderName.LOCAL_QWEN_GPU.value
                    or value.intent.peer_stt.provider == STTProviderName.LOCAL_QWEN_GPU.value
                )
            ),
            locale=value.intent.ui.locale if value is not None else None,
            device_id=value.intent.stt.gpu_device_id if value is not None else "auto",
        )

    def local_asr_runtime() -> LocalASRProviderRuntimePort | None:
        return pipeline.local_asr_runtime

    def on_gpu_install_diagnostic(
        diagnostic: LocalASRGpuProvisioningDiagnostic,
    ) -> None:
        log_detailed(
            "[GPU ASR] model_install failure=unexpected",
            level=logging.WARNING,
            exception=diagnostic.exception,
        )

    async def retry_gpu_activation() -> None:
        value = current_settings()
        if value is None:
            return
        await require_gpu_recovery().recover(
            lambda: gpu_recovery_request(
                current_canonical(),
                reason="manual_retry",
                plan=None,
                peer_enabled=settings.peer_translation_enabled(),
            )
        )

    def require_gpu() -> GpuRuntimeInteractionOwner:
        nonlocal gpu
        if gpu is None:

            def runtime_provider() -> LocalASRProviderRuntimePort:
                runtime = local_asr_runtime()
                if runtime is None:
                    raise RuntimeError("local ASR provider runtime is unavailable")
                return runtime

            gpu = create_gpu_runtime_interaction_owner(
                runtime_provider=runtime_provider,
                provisioning_provider=require_provisioning,
                state_provider=gpu_state,
                presentation_sink=lambda state: presentation.set_dashboard_gpu_state(
                    devices=state.devices,
                    state=state.state,
                    progress_percent=state.progress_percent,
                    notice=state.notice,
                    publish_notice=state.publish_notice,
                ),
                detailed_log_sink=log_detailed,
                retry_activation=retry_gpu_activation,
                install_diagnostic_sink=on_gpu_install_diagnostic,
            )
        return gpu

    def require_local_asr_diagnostics() -> LocalASRDiagnosticsOwner:
        nonlocal local_asr_diagnostics
        if local_asr_diagnostics is None:
            local_asr_diagnostics = create_local_asr_diagnostics_owner(
                basic_log_sink=lambda message, level: log_basic(
                    message,
                    level=level,
                ),
                detailed_log_sink=log_detailed,
                gpu_effect_sink=require_gpu().apply_diagnostics_effect,
                gpu_discovery_origin_provider=lambda: require_gpu().snapshot.discovery_origin,
                gpu_provider_id=STTProviderName.LOCAL_QWEN_GPU.value,
            )
        return local_asr_diagnostics

    def on_provisioning_diagnostic(
        diagnostic: LocalASRProvisioningDiagnostic,
    ) -> None:
        fields = [
            f"model={diagnostic.model_id or 'unknown'}",
            f"origin={diagnostic.origin or 'runtime'}",
            f"outcome={diagnostic.outcome or 'observed'}",
        ]
        if diagnostic.elapsed_seconds is not None:
            fields.append(f"elapsed_seconds={diagnostic.elapsed_seconds:.3f}")
        if diagnostic.failure_type is not None:
            fields.append(f"failure_type={diagnostic.failure_type}")
        log_basic(
            f"[LocalASR][{diagnostic.event.title()}] {' '.join(fields)}",
            level=(logging.ERROR if diagnostic.outcome == "failed" else logging.INFO),
        )

    def on_provisioning_state(
        snapshot: LocalASRProvisioningSnapshot,
    ) -> None:
        presentation.set_settings_local_cpu_auto_available(snapshot.cpu_auto_available)
        require_gpu().observe_provisioning(snapshot)
        require_local_asr().adapters.notice.sync()

    def require_provisioning() -> LocalASRProvisioningPort:
        nonlocal provisioning
        if provisioning is None:
            provisioning = create_local_asr_provisioning_owner(
                state_changed=on_provisioning_state,
                diagnostic_sink=on_provisioning_diagnostic,
            )
        return provisioning

    async def validate_gpu_activation() -> bool:
        return await require_gpu().validate_activation()

    async def resume_self_after_cpu_repair() -> bool:
        snapshot = await require_self_application().run_switch(desired=True)
        presentation.set_dashboard_stt_enabled(
            bool(
                snapshot is not None
                and snapshot.desired_active
                and snapshot.state is not SelfCaptureSessionState.FAULTED
                and snapshot.has_loop_task
            )
        )
        require_local_asr().adapters.notice.sync()
        return True

    async def resume_peer_after_cpu_repair() -> None:
        await refresh_overlay_runtime_dependencies()

    def show_short_message(message_key: str, **message_kwargs: object) -> None:
        try:
            presentation.show_message(message_key, **message_kwargs)
        except Exception:
            log_error(presentation.localize(message_key, **message_kwargs))

    def require_local_asr() -> LocalASRApplicationRuntime:
        nonlocal local_asr
        if local_asr is None:

            def local_asr_application_settings() -> LocalASRApplicationSettings | None:
                canonical = settings.canonical
                if canonical is None:
                    return None
                languages = canonical.intent.languages
                self_provider = canonical.intent.stt.provider
                peer_provider = canonical.intent.peer_stt.provider
                return LocalASRApplicationSettings(
                    locale=canonical.intent.ui.locale,
                    self_provider=self_provider,
                    peer_provider=peer_provider,
                    self_source_language=languages.source_language,
                    peer_source_language=(
                        languages.peer_source_language or languages.source_language
                    ),
                    self_gpu_provider=self_provider == STTProviderName.LOCAL_QWEN_GPU.value,
                    peer_gpu_provider=peer_provider == STTProviderName.LOCAL_QWEN_GPU.value,
                    peer_requested=peer_local_stt_requested(),
                    peer_activation_requested=peer_activation_requested(),
                )

            local_asr = compose_local_asr_application(
                settings_provider=local_asr_application_settings,
                runtime_provider=lambda: pipeline.local_asr_runtime,
                self_capture_provider=lambda: pipeline.self_capture,
                peer_provider=lambda: require_peer().owner,
                provisioning_provider=require_provisioning,
                gpu_state_provider=lambda: require_gpu().snapshot.ui_state,
                retain_gpu_pending=lambda channel: require_gpu().retain_pending(
                    cast(GpuASRChannel, channel)
                ),
                validate_gpu_activation=validate_gpu_activation,
                dashboard_enabled_sink=presentation.set_dashboard_stt_enabled,
                dashboard_needs_key_sink=presentation.set_dashboard_stt_needs_key,
                message_sink=show_short_message,
                notice_sink=presentation.set_dashboard_local_stt_notice,
                rebuild_self_provider=lambda: (
                    require_runtime_components().provider_runtime.effects.rebuild_self_stt()
                ),
                resume_self=resume_self_after_cpu_repair,
                resume_peer=resume_peer_after_cpu_repair,
                persist_manual_fallback=lambda channel: (
                    require_settings_application().persist_manual_fallback(
                        channel=cast(Literal["self", "peer"], channel)
                    )
                ),
                load_log_sink=require_local_asr_diagnostics().log_load_result,
            )
        return local_asr

    def stt_requires_secret(provider: STTProviderName) -> bool:
        return provider in {
            STTProviderName.DEEPGRAM,
            STTProviderName.QWEN_ASR,
            STTProviderName.SONIOX,
        }

    def dashboard_stt_needs_key(*, stt_available: bool) -> bool:
        value = current_settings()
        if value is None:
            return not stt_available
        return stt_requires_secret(STTProviderName(value.intent.stt.provider)) and not stt_available

    def publish_osc_state_from_runtime() -> None:
        if vrc_mic_sync is not None:
            vrc_mic_sync.publish_delta()

    def on_self_capture_state(_snapshot: SelfCaptureSessionSnapshot) -> None:
        require_local_asr().adapters.notice.sync()
        publish_osc_state_from_runtime()

    def on_peer_capture_state(snapshot: PeerCaptureSessionSnapshot) -> None:
        require_peer().owner.on_runtime_state_changed(snapshot)
        publish_osc_state_from_runtime()

    def require_self_application() -> SelfCaptureApplicationOwner:
        nonlocal self_application
        if self_application is None:
            self_application = SelfCaptureApplicationOwner(
                settings_provider=lambda: (
                    SelfCaptureApplicationSettings(
                        config=build_self_capture_session_config_from_vnext(canonical),
                        provider_id=canonical.intent.stt.provider,
                        qwen_region=canonical.intent.translation.qwen.region,
                    )
                    if (canonical := settings.canonical) is not None
                    else None
                ),
                runtime_available=lambda: pipeline.self_translation_channel is not None,
                capture_owner=lambda: require_runtime_components().self_capture_owner(),
                capture_owner_if_created=lambda: pipeline.self_capture,
                persist_manual_fallback=lambda: (
                    require_settings_application().persist_manual_fallback(channel="self")
                ),
                reset_local_pending=lambda: require_local_asr().cpu_repair.reset_self(),
                clear_gpu_pending=lambda: require_gpu().clear_pending("self"),
                overlay_state_provider=lambda: require_overlay().snapshot.state,
                mark_promo_eligible=lambda: (
                    pipeline.self_translation_channel.mark_promo_eligible()
                    if pipeline.self_translation_channel is not None
                    else None
                ),
                dashboard_enabled_sink=presentation.set_dashboard_stt_enabled,
                dashboard_needs_key_sink=presentation.set_dashboard_stt_needs_key,
                dashboard_needs_key=dashboard_stt_needs_key,
                state_sink=on_self_capture_state,
                sync_effective_flags=sync_effective_flags,
                sync_local_notice=lambda: require_local_asr().adapters.notice.sync(),
                log_basic=log_basic,
                log_detailed=lambda message, level: log_detailed(
                    message,
                    level=level,
                ),
            )
        return self_application

    async def stop_self_capture() -> None:
        await require_self_application().set_enabled(False)

    def require_vrc_mic_sync() -> OscControlIntegrationOwner:
        nonlocal vrc_mic_sync
        if vrc_mic_sync is None:

            def osc_state() -> object:
                value = current_settings()
                if value is None:
                    if settings.canonical is None:
                        raise RuntimeError("Settings are not loaded")
                    return state_from_settings(settings.require_canonical())
                self_capture = pipeline.self_capture
                peer_owner = require_peer().owner
                return state_from_settings(
                    value,
                    self_capture=bool(
                        self_capture is not None and self_capture.snapshot.desired_active
                    ),
                    peer_capture=bool(peer_owner.snapshot().effective_enabled),
                    translation=bool(
                        managed_account is not None
                        and managed_account.translation.state_provider().translation_enabled
                    ),
                    captions=settings.overlay_enabled(),
                )

            def language_state() -> tuple[str, str, str, str]:
                value = current_settings()
                if value is None:
                    return ("ko", "en", "en", "ko")
                return (
                    value.intent.languages.source_language,
                    value.intent.languages.target_language,
                    value.intent.languages.peer_source_language,
                    value.intent.languages.peer_target_language,
                )

            def osc_ui_state(
                control: OscControlPresentationName,
            ) -> OscControlPresentationState:
                value = current_settings()
                if value is None:
                    if settings.canonical is None:
                        raise RuntimeError("Settings are not loaded")
                    value = settings.require_canonical()
                return osc_control_presentation_state(
                    value,
                    canonical_state=osc_state(),
                    changed_control=control,
                    self_capture_effective=bool(
                        pipeline.self_capture is not None
                        and pipeline.self_capture.snapshot.effective_active
                    ),
                )

            vrc_mic_sync = compose_vrc_mic_sync(
                state_provider=lambda: pipeline.vrc_mic_state,
                gate_provider=lambda: pipeline.vrc_mic_audio_gate,
                log_detailed=lambda message, level: log_detailed(
                    message,
                    level=level,
                ),
                error_sink=log_error,
                settings_provider=current_settings,
                apply_settings=lambda next_settings: require_settings_application().apply(
                    next_settings,
                    reload_settings_view=False,
                ),
                application_provider=lambda: application,
                sender_provider=lambda: pipeline.sender,
                osc_state_provider=osc_state,
                ui_state_provider=osc_ui_state,
                ui_state_sink=presentation.project_osc_control_state,
                language_state_provider=language_state,
                translation_model_normalizer=settings.materialize_translation,
            )
        return vrc_mic_sync

    def require_microphone() -> MicrophoneTestRuntime:
        nonlocal microphone
        if microphone is None:
            microphone = MicrophoneTestRuntime(
                audio_provider=microphone_audio_settings,
                self_capture_provider=lambda: pipeline.self_capture,
                local_pending_provider=lambda: require_local_asr().self_pending,
                disable_self_capture=stop_self_capture,
                clock=clock,
                log_sink=log_basic,
                detailed_sink=lambda message, level, exception: log_detailed(
                    message,
                    level=level,
                    exception=exception,
                ),
                error_sink=log_error,
            )
        return microphone

    def require_manual_typing() -> ManualTypingOwner:
        nonlocal manual_typing
        if manual_typing is None:

            def output_provider():
                return pipeline.translation_output_projection

            def completion_provider(utterance_id: object) -> object | None:
                owner = pipeline.self_translation_channel
                runtime = getattr(owner, "runtime", None)
                tasks = getattr(runtime, "translation_tasks", None)
                return tasks.get(utterance_id) if isinstance(tasks, dict) else None

            manual_typing = create_manual_typing_owner(
                output_provider=output_provider,
                completion_provider=completion_provider,
                log_detailed=log_detailed,
                log_error=log_error,
                idle_timeout_seconds=MANUAL_INPUT_TYPING_IDLE_TIMEOUT_S,
                submit_timeout_seconds=MANUAL_SUBMIT_TYPING_TIMEOUT_S,
            )
        return manual_typing

    async def submit_clipboard(text: str) -> None:
        owner = pipeline.self_translation_channel
        if owner is not None:
            await owner.submit_text(text, source="Clipboard")

    def require_clipboard() -> ClipboardAutoTranslationOwner:
        nonlocal clipboard
        if clipboard is None:
            clipboard = ClipboardAutoTranslationOwner(
                watcher_factory=create_clipboard_watcher,
                submit_text=submit_clipboard,
                failure_sink=log_error,
                platform_provider=lambda: sys.platform,
            )
        return clipboard

    async def sync_clipboard() -> None:
        canonical = settings.canonical
        await require_clipboard().sync(
            enabled=bool(
                canonical is not None and canonical.intent.clipboard.auto_translate_enabled
            )
        )

    def require_projection() -> SettingsProjectionOwner:
        nonlocal settings_projection
        if settings_projection is None:
            settings_projection = SettingsProjectionOwner(
                presentation=presentation,
                config_path=config_path,
                current_settings=current_settings,
            )
        return settings_projection

    def sync_ui_from_settings() -> None:
        canonical = settings.canonical
        if canonical is None:
            return
        languages = canonical.intent.languages
        presentation.set_dashboard_languages(
            source_language=languages.source_language,
            target_language=languages.target_language,
            peer_source_language=languages.peer_source_language,
            peer_target_language=languages.peer_target_language,
            peer_source_mode=languages.peer_source_mode,
            recent_source_languages=languages.recent_source_languages,
            recent_target_languages=languages.recent_target_languages,
            peer_auto_detect_available=(
                canonical.intent.peer_stt.provider
                in {
                    STTProviderName.SONIOX.value,
                    STTProviderName.LOCAL_QWEN_GPU.value,
                }
            ),
        )
        loaded = require_projection().render(settings.canonical)
        if loaded:
            with contextlib.suppress(Exception):
                presentation.set_settings_overlay_calibration(require_calibration().current)
        refresh_overlay_presentation()

    def log_manual_fallbacks(
        previous,
        normalized,
        channels: tuple[str, ...],
    ) -> None:
        previous_canonical = settings.project(previous, authoritative=True)
        normalized_canonical = settings.project(normalized, authoritative=True)
        for channel in channels:
            if channel == "self":
                requested = previous_canonical.intent.stt.provider
                actual = normalized_canonical.intent.stt.provider
                source_language = normalized_canonical.intent.languages.source_language
            else:
                requested = previous_canonical.intent.peer_stt.provider
                actual = normalized_canonical.intent.peer_stt.provider
                languages = normalized_canonical.intent.languages
                source_language = languages.peer_source_language or languages.source_language
            decision = resolve_local_asr_selection(actual, source_language)
            log_basic(
                "[LocalASR][Selection] "
                f"channel={channel} requested={requested} actual={actual} "
                f"model={decision.model_id or 'unknown'} "
                "reason=preferred_model_unavailable"
            )

    def active_local_asr_change(
        base: AppSettingsVNext,
        next_value: AppSettingsVNext,
    ) -> bool:
        local_providers = {
            *LOCAL_CPU_PROVIDERS,
            STTProviderName.LOCAL_QWEN_GPU.value,
        }
        self_owner = pipeline.self_capture
        self_changed = (
            self_owner is not None
            and self_owner.snapshot.desired_active
            and (
                base.intent.stt.provider in local_providers
                or next_value.intent.stt.provider in local_providers
            )
            and build_self_stt_runtime_signature_from_vnext(base)
            != build_self_stt_runtime_signature_from_vnext(next_value)
        )
        peer_changed = (
            peer_runtime_desired()
            and (
                base.intent.peer_stt.provider in local_providers
                or next_value.intent.peer_stt.provider in local_providers
            )
            and build_peer_stt_runtime_signature_from_vnext(base)
            != build_peer_stt_runtime_signature_from_vnext(next_value)
        )
        return self_changed or peer_changed

    def require_settings_application() -> SettingsApplicationOwner:
        nonlocal settings_application
        if settings_application is None:
            settings_application = SettingsApplicationOwner(
                settings=settings,
                projection=require_projection(),
                runtime_effects=SettingsRuntimeEffectsAdapter(
                    state=effects_state,
                    settings=settings,
                    presentation=presentation,
                    runtime_logging=runtime_logging,
                    pipeline=pipeline,
                    runtime_signatures=signatures,
                    microphone=require_microphone(),
                    clipboard=require_clipboard(),
                    provisioning=require_provisioning(),
                    gpu=require_gpu(),
                    vrc_mic_sync=require_vrc_mic_sync(),
                    projection=require_projection(),
                    github_prompt=require_github_prompt,
                    desktop_overlay=require_desktop(),
                    calibration=require_calibration(),
                    overlay=require_overlay(),
                    overlay_state_provider=overlay_state,
                    peer=require_peer(),
                    self_capture=lambda: pipeline.self_capture,
                    clear_local_pending=lambda: (
                        require_local_asr().cpu_repair.clear_if_provider_switched_away()
                    ),
                    replace_self_stt=lambda smooth: require_self_application().replace_provider(
                        smooth_local=smooth
                    ),
                    rebuild_managed_gemma=lambda: provider_runtime.llm_rebuild.rebuild(),
                ),
                manual_fallback=manual_fallback,
                cpu_auto_available=lambda: require_provisioning().snapshot.cpu_auto_available,
                inspect_cpu=require_provisioning().inspect_cpu,
                fallback_sink=lambda channels, installation_fallback: (
                    show_short_message(
                        "local_stt.installation_fallback_qwen"
                        if installation_fallback
                        else "local_stt.language_fallback_qwen"
                    )
                    if channels
                    else None
                ),
                sync_ui=sync_ui_from_settings,
                fallback_log_sink=log_manual_fallbacks,
                mutation_service_provider=lambda: None,
                consume_superseded_settings=signatures.consume_superseded,
                active_local_asr_change=active_local_asr_change,
                failure_sink=log_error,
            )
        return settings_application

    def merge_provider_settings(pending):
        value = current_settings()
        if value is None:
            return copy.deepcopy(pending)
        merged = _copy_provider_prompt_apply_fields(pending, copy.deepcopy(value))
        config_owner = pipeline.translation_runtime_configuration
        if config_owner is not None:
            configuration = config_owner.snapshot().value
            merged = replace(
                merged,
                intent=replace(
                    merged.intent,
                    languages=replace(
                        merged.intent.languages,
                        source_language=configuration.source_language,
                        target_language=configuration.target_language,
                        peer_source_language=configuration.peer_source_language,
                        peer_target_language=configuration.peer_target_language,
                    ),
                ),
            )
        if (
            merged.intent.peer_stt.provider
            not in {
                STTProviderName.SONIOX.value,
                STTProviderName.LOCAL_QWEN_GPU.value,
            }
            and merged.intent.languages.peer_source_mode == "auto"
        ):
            merged = replace(
                merged,
                intent=replace(
                    merged.intent,
                    languages=replace(merged.intent.languages, peer_source_mode="manual"),
                ),
            )
        return merged

    def capture_runtime_signatures_before_mutation() -> None:
        value = current_settings()
        if value is None:
            return
        signatures.capture_peer_before_canonical_mutation(
            value,
            canonical=canonical_settings(value),
            peer=require_peer().owner,
        )

    def sync_non_provider_signatures(_value=None) -> None:
        effects_state.microphone_audio_signature = MicrophoneTestRuntime.audio_signature(
            microphone_audio_settings()
        )

    def sync_signature_caches(value) -> None:
        signatures.sync(
            value,
            canonical=canonical_settings(value),
            peer=require_peer().owner,
            peer_translation_enabled=settings.peer_translation_enabled(),
        )
        sync_non_provider_signatures()

    def require_provider_verifier() -> ProviderVerifierPort:
        nonlocal provider_verifier
        if provider_verifier is None:
            provider_verifier = create_provider_verifier()
        return provider_verifier

    def require_provider_settings() -> ProviderSettingsOwner:
        nonlocal provider_settings
        if provider_settings is None:
            provider_settings = ProviderSettingsOwner(
                settings=settings,
                binding=ProviderVerificationBindingOwner(
                    context_provider=lambda provider: provider_verification_context(
                        current_settings(),
                        provider,
                        low_latency=(FIXED_TRANSLATION_POLICY.fast_translation_enabled),
                    ),
                ),
                secret_store_factory=lambda value: create_sync_secret_store_adapter(
                    create_secret_store(
                        value.intent.secrets,
                        config_path=config_path,
                    )
                ),
                active_secret_provider=lambda value, secret_key: create_secret_store(
                    value.intent.secrets,
                    config_path=config_path,
                ).get(secret_key),
                save_failure_sink=log_error,
                results=require_settings_application().results,
            )
        return provider_settings

    def require_credential_verification() -> ProviderCredentialVerificationInteractionOwner:
        nonlocal credential_verification
        if credential_verification is None:
            credential_verification = create_provider_credential_verification_interaction_owner(
                verifier=require_provider_verifier(),
                selected_model_provider=lambda provider: (
                    require_provider_settings().binding.selected_model(provider)
                ),
                fallback_models=tuple(model.value for model in QwenLLMModel),
                low_latency=(FIXED_TRANSLATION_POLICY.fast_translation_enabled),
                diagnostics_sink=lambda event, metadata, exception: log_detailed(
                    "[ProviderVerification] Credential verification "
                    f"failed event={event} "
                    f"provider={metadata.get('provider')} "
                    f"error_type={metadata.get('error_type')}",
                    level=logging.WARNING,
                    exception=exception,
                ),
                error_sink=lambda provider, error_text: log_error(
                    f"Verification error for {provider}: {error_text}"
                ),
            )
        return credential_verification

    def github_save_failure(
        failure_context: str,
        exc: Exception,
    ) -> None:
        log_basic(
            "[GitHubStar] Failed to persist prompt "
            f"{failure_context}: exception_class={type(exc).__name__}",
            level=logging.WARNING,
        )

    def github_diagnostic(
        event: str,
        metadata: Mapping[str, object],
    ) -> None:
        log_detailed(
            f"[Lifecycle][GithubStarPromptRuntime] event={event} metadata={dict(metadata)}",
            level=logging.WARNING,
        )

    def require_github_prompt() -> GithubStarPromptOwner:
        nonlocal github_prompt
        if github_prompt is None:
            github_prompt = compose_github_star_prompt_owner(
                settings=settings,
                managed_remaining_percent=lambda: (
                    require_runtime_components().managed_account.usage.remaining_percent
                ),
                transaction_result_sink=(require_settings_application().results.set),
                save_failure_sink=github_save_failure,
                runtime_diagnostics_sink=github_diagnostic,
                mutation_service_provider=lambda: None,
            )
        return github_prompt

    def telemetry_diagnostic(
        event: str,
        metadata: Mapping[str, object],
    ) -> None:
        log_detailed(
            f"[Telemetry] event={event} metadata={dict(metadata)}",
            level=logging.INFO,
        )

    def telemetry_service() -> AppActiveDayTelemetryService:
        return AppActiveDayTelemetryService(
            HttpAppActiveDayTelemetryClient(),
            diagnostics_sink=telemetry_diagnostic,
        )

    def vrchat_probe_port() -> int:
        value = current_settings()
        if value is None:
            return 9000
        port = getattr(value.intent.osc, "port", 9000)
        return port if isinstance(port, int) and 0 < port <= 65535 else 9000

    def require_vrchat_presence() -> VrchatOscPresenceProbeOwner:
        nonlocal vrchat_presence
        if vrchat_presence is None:
            vrchat_presence = create_vrchat_osc_presence_probe_owner(
                presence_provider=lambda: vrchat_osc_presence,
                port_provider=vrchat_probe_port,
                publish_notice=presentation.set_dashboard_vrchat_osc_notice,
                diagnostics_sink=lambda _event, _metadata, exception: log_detailed(
                    "[OSC] VRChat OSC presence probe failed",
                    level=logging.WARNING,
                    exception=exception,
                ),
            )
        return vrchat_presence

    def require_audio_diagnostics() -> AudioDiagnosticsApplicationOwner:
        nonlocal audio_diagnostics
        if audio_diagnostics is None:
            audio_diagnostics = AudioDiagnosticsApplicationOwner(
                presentation=presentation,
                runtime_logging=runtime_logging,
            )
        return audio_diagnostics

    def build_local_asr_factory(
        secrets: object,
    ) -> LocalASRProviderRuntimeFactory:
        return LocalASRProviderRuntimeFactory(
            provider_factory=ManagedSTTProviderFactory(
                secrets=secrets,
                clock=clock,
                reset_deadline_s=STT_RESET_DEADLINE_S,
                gpu_model_path=local_gpu_model_path(),
                diagnostics_enabled=require_audio_diagnostics().detailed_enabled,
                on_final_transcript_suppressed=(
                    require_audio_diagnostics().on_final_transcript_suppressed
                ),
                runtime_logging=runtime_logging.service,
                fault_profile_provider=lambda: (
                    require_audio_diagnostics().stt_fault_profile
                    if require_audio_diagnostics().debug_allowed()
                    else "none"
                ),
            ),
            provisioning=require_provisioning(),
            clock=clock,
            state_changed=lambda snapshot: require_gpu().observe_runtime(snapshot),
            diagnostic_sink=(require_local_asr_diagnostics().provider_runtime_diagnostic),
        )

    def gpu_recovery_request(
        canonical: AppSettingsVNext,
        *,
        reason: Literal["manual_retry", "settings_restart"],
        plan: ProviderRuntimeApplyPlan | None,
        peer_enabled: bool,
    ) -> GpuProviderRecoveryApplicationRequest:
        if reason == "settings_restart" and plan is None:
            raise RuntimeError("settings GPU recovery requires a runtime apply plan")
        gpu_device_id = canonical.intent.stt.gpu_device_id
        return GpuProviderRecoveryApplicationRequest(
            device_id=gpu_device_id,
            reason=reason,
            self_gpu_selected=(
                canonical.intent.stt.provider == STTProviderName.LOCAL_QWEN_GPU.value
            ),
            peer_gpu_selected=(
                canonical.intent.peer_stt.provider == STTProviderName.LOCAL_QWEN_GPU.value
            ),
            self_desired=bool(
                pipeline.self_capture is not None and pipeline.self_capture.snapshot.desired_active
            ),
            peer_enabled=peer_enabled,
            self_config_factory=lambda: build_self_capture_session_config_from_vnext(canonical),
            peer_config_factory=lambda: build_peer_capture_session_config_from_vnext(canonical),
            self_request_factory=lambda: build_self_stt_provider_request_from_vnext(
                canonical,
                warmup=True,
            ),
            peer_request_factory=lambda config: build_peer_stt_provider_request(
                config,
                gpu_device_id=gpu_device_id,
                warmup=True,
            ),
            should_refresh_self=bool(plan is not None and plan.should_refresh_self_stt),
            should_refresh_peer=bool(plan is not None and plan.should_refresh_peer),
        )

    async def suspend_gpu_consumers(
        channels: tuple[GpuASRChannel, ...],
    ) -> None:
        if "self" in channels and pipeline.self_capture is not None:
            snapshot = await pipeline.self_capture.suspend_provider_consumer()
            on_self_capture_state(snapshot)
        peer_owner = require_peer().owner.runtime
        if "peer" in channels and peer_owner is not None:
            await peer_owner.suspend_provider_consumer()

    def on_gpu_recovery_diagnostic(
        diagnostic: GpuProviderRecoveryDiagnostic,
    ) -> None:
        fields = [
            f"outcome={diagnostic.outcome}",
            f"reason={diagnostic.reason}",
            f"channels={','.join(diagnostic.channels) or 'none'}",
        ]
        if diagnostic.failure_type is not None:
            fields.append(f"failure_type={diagnostic.failure_type}")
        log_detailed(
            f"[GPU ASR] provider_recovery {' '.join(fields)}",
            level=(
                logging.WARNING
                if diagnostic.outcome in {"failed", "prepare_failed"}
                else logging.INFO
            ),
        )

    def require_gpu_recovery() -> GpuProviderRecoveryApplicationOwner:
        nonlocal gpu_recovery
        if gpu_recovery is None:
            gpu_recovery = create_gpu_provider_recovery_application_owner(
                runtime_provider=local_asr_runtime,
                pending_provider=lambda: require_gpu().snapshot.pending_channels,
                pending_clear=require_gpu().complete_manual_recovery,
                failure_sink=lambda reason: require_gpu().set_ui_state(
                    "activation_failed",
                    publish_notice=True,
                    origin=("manual_retry" if reason == "manual_retry" else "settings_apply"),
                ),
                runtime_state_sink=lambda snapshot: require_gpu().observe_runtime(snapshot),
                quiesce=suspend_gpu_consumers,
                self_owner_factory=(require_runtime_components().self_capture_owner),
                peer_owner_provider=lambda: require_peer().owner.runtime,
                self_state_sink=on_self_capture_state,
                ensure_self_switch=require_self_application().run_switch,
                refresh_self=(
                    require_runtime_components().provider_runtime.effects.refresh_self_stt
                ),
                refresh_peer=(require_runtime_components().provider_runtime.effects.refresh_peer),
                diagnostic_sink=on_gpu_recovery_diagnostic,
            )
        return gpu_recovery

    capture_factory = CaptureOwnerFactory(
        canonical_provider=lambda: None if current_settings() is None else current_canonical(),
        self_admission=create_self_capture_admission_adapter(
            state_provider=(require_local_asr().adapters.state.self_admission),
            validate_gpu_activation=validate_gpu_activation,
            effect_sink=(require_local_asr().adapters.effects.apply_self_admission),
        ),
        ensure_peer_local_ready=lambda generation: require_local_asr().ensure_peer_ready(
            activation_generation=generation,
        ),
        clock=clock,
        log_detailed=log_detailed,
        detailed_enabled=require_audio_diagnostics().detailed_enabled,
        source_wrapper=lambda source, channel: (
            require_audio_diagnostics()
            .capture_adapter()
            .wrap_source(
                source,
                channel_label=channel,
            )
        ),
        self_state_sink=on_self_capture_state,
        self_diagnostic_sink=(require_audio_diagnostics().capture_adapter().self_capture),
        peer_state_sink=on_peer_capture_state,
        peer_diagnostic_sink=require_peer().owner.on_runtime_diagnostic,
        local_asr_diagnostic_sink=(require_local_asr_diagnostics().transition_diagnostic),
    )

    def self_capture_owner() -> SelfCaptureSessionOwner:
        return _require_self_capture_owner(pipeline, capture_factory)

    def require_managed_account() -> ManagedAccountComponents:
        if managed_account is None:
            raise RuntimeError("managed-account composition is incomplete")
        return managed_account

    def managed_release() -> ManagedOpenRouterReleaseRuntime:
        return require_managed_account().release

    def set_managed_pending(pending: bool) -> None:
        require_managed_account().auth.set_pending(pending)

    def managed_pending() -> bool:
        return bool(managed_account is not None and managed_account.auth.pending)

    def managed_delegate_ready() -> None:
        require_managed_account().usage.delegate_ready()

    async def refresh_managed_usage() -> None:
        await require_managed_account().usage.refresh_best_effort()

    async def recover_gpu(value, plan: ProviderRuntimeApplyPlan) -> None:
        await require_gpu_recovery().recover(
            lambda: gpu_recovery_request(
                settings.project(value, authoritative=True),
                reason="settings_restart",
                plan=plan,
                peer_enabled=settings.peer_translation_enabled(),
            )
        )

    provider_runtime: ProviderRuntimeComponents = compose_provider_runtime(
        config_path=config_path,
        settings=settings,
        llm_runtime_provider=lambda: pipeline.llm_runtime,
        local_asr_runtime_provider=lambda: pipeline.local_asr_runtime,
        translation_runtime_configuration_provider=(
            lambda: pipeline.translation_runtime_configuration
        ),
        http_extensions=http_extensions,
        self_capture_provider=lambda: pipeline.self_capture,
        self_capture_owner=self_capture_owner,
        peer=lambda: require_peer().owner,
        peer_desired=peer_runtime_desired,
        canonical_settings=canonical_settings,
        clear_local_pending=lambda: (
            require_local_asr().cpu_repair.clear_if_provider_switched_away()
        ),
        sync_local_notice=lambda: require_local_asr().adapters.notice.sync(),
        managed_pending_sink=set_managed_pending,
        managed_pending_provider=managed_pending,
        dashboard_managed_pending_sink=(presentation.set_dashboard_managed_auth_pending),
        sync_effective_flags=sync_effective_flags,
        refresh_overlay=refresh_overlay_presentation,
        refresh_peer_runtime=lambda: require_peer().owner.refresh_runtime(),
        replace_self_stt=lambda smooth: require_self_application().replace_provider(
            smooth_local=smooth
        ),
        self_state_sink=on_self_capture_state,
        self_availability=require_self_application().project_availability,
        gpu_recovery=recover_gpu,
        managed_release=managed_release,
        managed_delegate_ready=managed_delegate_ready,
        runtime_logging=runtime_logging,
        translation_needs_key_sink=(presentation.set_dashboard_translation_needs_key),
        usage_refresh=refresh_managed_usage,
        failure_sink=log_error,
        success_sink=log_basic,
        additional_signature_sink=sync_non_provider_signatures,
        managed_gemma=managed_gemma,
        signatures=signatures,
    )

    def apply_managed_usage_view(state) -> None:
        presentation.set_settings_managed_key_state(
            visible=state.visible,
            remaining_percent=state.remaining_percent,
            referral_id=state.referral_id,
            pass_status=state.pass_status,
        )

    def maybe_show_founder_letter(launch_source: str) -> None:
        if launch_source != "letter":
            return
        with contextlib.suppress(Exception):
            presentation.show_founder_letter_dialog()

    managed_account = compose_managed_account(
        config_path=config_path,
        settings=settings,
        provider_settings=require_provider_settings(),
        provider_runtime=provider_runtime.runtime,
        verifier=require_provider_verifier(),
        results=require_settings_application().results,
        runtime=ManagedTranslationRuntimeAccess(
            llm_runtime_provider=lambda: pipeline.llm_runtime,
            context_provider=lambda: pipeline.translation_requests,
            translation_runtime_configuration_provider=(
                lambda: pipeline.translation_runtime_configuration
            ),
            rebuild_llm=provider_runtime.llm_rebuild.rebuild,
        ),
        ingress_provider=lambda: ingress.frozen,
        pending_sink=presentation.set_dashboard_managed_auth_pending,
        usage_view_sink=apply_managed_usage_view,
        dashboard_sink=presentation.set_dashboard_translation_enabled,
        starting_sink=presentation.set_dashboard_translation_starting,
        runtime_state_changed=lambda: require_vrc_mic_sync().publish_delta(),
        message_sink=lambda key, values: show_short_message(
            key,
            **dict(values),
        ),
        qq_dialog_sink=presentation.show_qq_managed_auth_dialog,
        founder_dialog=presentation.show_founder_letter_dialog,
        failure_route=maybe_show_founder_letter,
        log_basic=log_basic,
        log_detailed=log_detailed,
        log_error=log_error,
        basic_warning_sink=lambda message: log_basic(
            message,
            level=logging.WARNING,
        ),
        detailed_warning_sink=lambda message, exception: log_detailed(
            message,
            level=logging.WARNING,
            exception=exception,
        ),
        managed_gemma=managed_gemma,
        sync_local_translation_demand=sync_local_translation_demand,
    )

    provider_application = ProviderApplicationOwner(
        settings=settings,
        runtime=provider_runtime.runtime,
        merge_settings=merge_provider_settings,
        preserve_before_replace=(require_github_prompt().preserve_before_settings_replace),
        sync_ui=sync_ui_from_settings,
        order24_patch_provider=(require_projection().order24_patch_base_and_values),
        apply_order24=(require_settings_application().apply_ui_prompt_clipboard_state),
        remember_order22=require_projection().remember_order22,
        mutation_service_provider=lambda: None,
        save_failure_sink=log_error,
        results=require_settings_application().results,
        sync_memory=(require_settings_application().runtime_effects.restore_memory),
        capture_runtime_signatures=capture_runtime_signatures_before_mutation,
        sync_signatures=sync_signature_caches,
        consume_superseded_settings=signatures.consume_superseded,
        active_local_asr_change=active_local_asr_change,
        compensate_local_asr=(
            require_settings_application().compensate_failed_local_asr_settings_apply
        ),
        llm_retry_pending=lambda: signatures.last_llm_provider == (),
        mark_llm_retry=signatures.mark_llm_retry,
    )

    def install_pipeline(components: RuntimePipelineComponents) -> None:
        pipeline.install(components)

    pipeline_launcher = RuntimePipelineLauncher(
        config_path=config_path,
        clock=clock,
        runtime_logging=runtime_logging,
        managed_release=managed_account.release,
        managed_delegate_ready=managed_delegate_ready,
        local_asr_factory=build_local_asr_factory,
        self_capture_factory=capture_factory.compose_self,
        peer_capture_factory=capture_factory.compose_peer,
        previous_self_capture=lambda: pipeline.self_capture,
        component_sink=install_pipeline,
        peer_application=lambda: require_peer().owner,
        configure_vrc_mic=lambda *, enabled: (require_vrc_mic_sync().configure(enabled=enabled)),
        stt_failure_sink=log_error,
        cleanup_failure_sink=lambda message, exc: log_error(f"{message}: {exc}"),
        managed_gemma=managed_gemma,
        http_extensions=http_extensions,
    )

    runtime_components = RuntimeCompositionComponents(
        self_capture_owner=self_capture_owner,
        provider_runtime=provider_runtime,
        managed_account=managed_account,
        provider_application=provider_application,
        pipeline_launcher=pipeline_launcher,
    )

    def alibaba_verified_key() -> str:
        if current_canonical().intent.translation.qwen.region == QwenRegion.BEIJING.value:
            return "alibaba_beijing"
        return "alibaba_singapore"

    def llm_requires_secret(provider: LLMProviderName) -> bool:
        canonical = settings.canonical
        if (
            canonical is not None
            and canonical.intent.translation.model == TranslationModel.CUSTOM_HTTP.value
        ):
            return False
        return provider in {
            LLMProviderName.GEMINI,
            LLMProviderName.OPENROUTER,
            LLMProviderName.QWEN,
            LLMProviderName.DEEPSEEK,
        }

    def managed_translation_available() -> bool:
        canonical = settings.canonical
        llm_runtime = pipeline.llm_runtime
        if canonical is None or llm_runtime is None or llm_runtime.provider is None:
            return False
        translation = canonical.intent.translation
        return (
            translation.model != TranslationModel.CUSTOM_HTTP.value
            and translation.openrouter_selected_source == OpenRouterCredentialSource.MANAGED.value
            and provider_llm_for_translation(translation.model, translation.connection)
            == LLMProviderName.OPENROUTER.value
        )

    def create_event_bridge(
        active_runtime_logging: object,
    ) -> UIEventBridgePort:
        event_queue = pipeline.ui_events
        if event_queue is None:
            raise RuntimeError("UI Event Bridge owner is unavailable")
        return presentation.create_ui_event_bridge(
            event_queue=event_queue,
            runtime_logging=active_runtime_logging,
        )

    def start_event_bridge(bridge: UIEventBridgePort) -> None:
        nonlocal event_bridge, bridge_task
        output_runtime = pipeline.output_runtime
        if output_runtime is None:
            raise RuntimeError("UI Event Bridge owner is unavailable")
        event_bridge = bridge
        bridge_task = output_runtime.start_ui_event_bridge(bridge)

    async def wait_for_event_bridge() -> None:
        output_runtime = pipeline.output_runtime
        if output_runtime is None:
            raise RuntimeError("UI Event Bridge owner is unavailable")
        await output_runtime.wait_for_ui_event_bridge_started()

    startup = compose_application_startup(
        ApplicationStartupAdapter(
            settings=settings,
            settings_loader=lambda: load_application_settings(
                settings=settings,
            ),
            provisioning=require_provisioning(),
            gpu_state=gpu_state,
            manual_fallback=manual_fallback,
            save_failure_sink=lambda exc: log_error(f"Failed to save settings: {exc}"),
            calibration=require_calibration(),
            presentation=presentation,
            sync_presentation=sync_ui_from_settings,
            notify_fallback=require_settings_application().notify_fallback,
            runtime_logging=runtime_logging,
            sync_runtime_signatures=sync_signature_caches,
            pipeline_launcher=pipeline_launcher,
            pipeline=pipeline,
            sync_local_asr_notice=lambda: require_local_asr().adapters.notice.sync(),
            stt_requires_secret=stt_requires_secret,
            llm_requires_secret=llm_requires_secret,
            alibaba_verified_key=alibaba_verified_key,
            managed_translation_available=managed_translation_available,
            receiver_active=lambda: require_vrc_mic_sync().receiver is not None,
            create_event_bridge=create_event_bridge,
            start_event_bridge=start_event_bridge,
            wait_for_event_bridge=wait_for_event_bridge,
            sync_clipboard=sync_clipboard,
        )
    )

    async def close_local_asr() -> None:
        if local_asr is not None:
            await local_asr.close()
        elif provisioning is not None:
            await provisioning.close()

    async def close_openrouter_oauth() -> None:
        await managed_account.pkce_flow.close()

    def clear_event_runtime() -> None:
        nonlocal event_bridge, bridge_task
        event_bridge = None
        bridge_task = None

    runtime_shutdown = ApplicationRuntimeShutdownAdapter(
        ingress=ingress,
        pipeline=pipeline,
        runtime_logging=runtime_logging,
        managed=managed_account,
        pipeline_launcher=pipeline_launcher,
        stop_self_capture=stop_self_capture,
        release_manual_typing_owner=require_manual_typing().release,
        close_local_asr_provisioning_owner=close_local_asr,
        close_openrouter_oauth_owner=close_openrouter_oauth,
        clear_ui_event_runtime=clear_event_runtime,
        peer=lambda: peer.owner if peer is not None else None,
        overlay=lambda: overlay,
        vrchat_presence=lambda: vrchat_presence,
        vrc_mic_sync=lambda: vrc_mic_sync,
        github_prompt=lambda: github_prompt,
        clipboard=lambda: clipboard,
        microphone=lambda: microphone,
        close_managed_gemma_owner=managed_gemma.close,
    )

    overlay_owner = require_overlay()
    peer_runtime = require_peer()
    microphone_runtime = require_microphone()
    desktop_owner = require_desktop()
    calibration_owner = require_calibration()
    settings_owner = require_settings_application()
    gpu_owner = require_gpu()
    github_owner = require_github_prompt()
    audio_owner = require_audio_diagnostics()

    application = UiApplicationBoundary(
        startup=startup,
        input_runtime=UiInputRuntimeAdapter(
            pipeline=pipeline,
            manual_typing=require_manual_typing(),
            translation=managed_account.translation,
            self_capture=require_self_application(),
        ),
        peer_capture=UiPeerCaptureRuntimeAdapter(
            peer=peer_runtime,
            overlay=overlay_owner,
        ),
        settings=UiSettingsRuntimeAdapter(
            settings=settings,
            projection=require_projection(),
            application=settings_owner,
            merge_provider_settings=merge_provider_settings,
            telemetry_enabled_settings=settings.with_telemetry_enabled,
        ),
        provider=UiProviderRuntimeAdapter(
            settings=settings,
            provider_application=provider_application,
            gpu=gpu_owner,
            managed=managed_account,
            credential_verification=require_credential_verification(),
            provider_settings=require_provider_settings(),
            build_byok_target_settings=settings.build_managed_openrouter_byok_target,
            managed_gemma=managed_gemma,
            llm_devices_sink=lambda devices: presentation.set_dashboard_llm_gpu_devices(
                devices=devices
            ),
        ),
        microphone=UiMicrophoneRuntimeAdapter(
            microphone=microphone_runtime,
            level_log_interval_seconds=1.0,
        ),
        overlay=UiOverlayRuntimeAdapter(
            overlay=overlay_owner,
            desktop=desktop_owner,
            calibration=calibration_owner,
        ),
        managed=UiManagedRuntimeAdapter(managed_account),
        engagement=UiEngagementRuntimeAdapter(
            settings=settings,
            settings_application=settings_owner,
            github_prompt=github_owner,
            telemetry=telemetry_service(),
            after_launch=ApplicationAfterLaunchOwner(
                vrchat_presence=require_vrchat_presence(),
                gpu=gpu_owner,
            ),
        ),
        diagnostics=UiDiagnosticsRuntimeAdapter(
            runtime_logging=runtime_logging,
            overlay=overlay_owner,
            cycle_capture_fault=audio_owner.cycle_capture_fault_profile,
            cycle_stt_fault=audio_owner.cycle_stt_fault_profile,
            clear_audio_faults=audio_owner.clear_fault_profiles,
        ),
        state=UiApplicationStateOwner(
            ApplicationUiStateAdapter(
                config_path=config_path,
                settings=settings,
                pipeline=pipeline,
                desktop_overlay=desktop_owner,
                managed=managed_account,
                calibration=calibration_owner,
                microphone=lambda: microphone,
            ),
            runtime_logging=runtime_logging,
        ),
        runtime_shutdown=runtime_shutdown,
        runtime_logging=runtime_logging,
        settings_secrets=SettingsSecretsOwner(
            secret_store_factory=create_settings_secret_store,
        ),
        osc_state_publisher=lambda: require_vrc_mic_sync().publish_delta(),
        http_extension_registry=HttpExtensionRegistryService(
            http_extensions,
            SystemDirectoryOpener(),
        ),
    )

    async def initialize_local_asr_evidence(value: AppSettingsVNext) -> None:
        settings.canonical = value
        canonical = settings.project(value, authoritative=True)
        require_provisioning()
        sync_signature_caches(value)
        await pipeline_launcher.launch(
            runtime_pipeline_inputs_from_vnext(
                canonical,
                peer_translation_enabled=settings.peer_translation_enabled(),
            ),
            secrets=create_secret_store(canonical.intent.secrets, config_path=config_path),
            vrc_mic_state=pipeline.vrc_mic_state,
            vrc_mic_audio_gate=pipeline.vrc_mic_audio_gate,
            receiver_active=require_vrc_mic_sync().receiver is not None,
        )

    if local_asr_evidence_sink is not None:
        local_asr_evidence_sink(
            _LocalASRProductionCompositionAccess(
                config_path=config_path,
                settings_loader=lambda: load_application_settings(
                    settings=settings,
                ),
                runtime_initializer=initialize_local_asr_evidence,
                components_provider=lambda: pipeline.current,
                gpu_retry=retry_gpu_activation,
            )
        )
    return application
