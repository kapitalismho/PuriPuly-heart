from __future__ import annotations

from collections.abc import Awaitable, Callable

from puripuly_heart.app.ports.desktop_overlay import (
    DesktopOverlayPolicy,
    DesktopWorkAreaPort,
)
from puripuly_heart.app.ports.manual_typing import SelfChatboxTypingPort
from puripuly_heart.app.ports.microphone_test import (
    MicrophoneTestCapturePort,
    MicrophoneTestMeterCallback,
)
from puripuly_heart.app.ports.provider_verifier import ProviderVerifierPort
from puripuly_heart.app.ports.self_capture_admission import (
    SelfCaptureAdmissionEffectSink,
    SelfCaptureAdmissionStateProvider,
    SelfCaptureGpuActivationValidator,
)
from puripuly_heart.app.ports.vrchat_osc_presence import VrchatOscPresencePort
from puripuly_heart.app.services.gpu_provider_recovery import (
    GpuProviderRecoveryDiagnosticSink,
    GpuProviderRecoveryOwner,
)
from puripuly_heart.app.services.gpu_provider_recovery_application import (
    GpuProviderRecoveryApplicationOwner,
    GpuProviderRecoveryAsyncEffect,
    GpuProviderRecoveryFailureSink,
    GpuProviderRecoveryPeerOwnerProvider,
    GpuProviderRecoveryPendingClear,
    GpuProviderRecoveryPendingProvider,
    GpuProviderRecoveryRuntimeProvider,
    GpuProviderRecoverySelfOwnerFactory,
    GpuProviderRecoverySelfStateSink,
    GpuProviderRecoveryStateSink,
)
from puripuly_heart.app.services.gpu_runtime_interaction import (
    GpuRuntimeActivationRetry,
    GpuRuntimeDetailedLogSink,
    GpuRuntimeInstallDiagnosticSink,
    GpuRuntimeInteractionOwner,
    GpuRuntimeInteractionStateProvider,
    GpuRuntimePresentationSink,
    GpuRuntimeProvider,
    GpuRuntimeProvisioningProvider,
)
from puripuly_heart.app.services.local_asr_cpu_repair import (
    LocalASRCpuModelIdsForProvider,
    LocalASRCpuPeerResume,
    LocalASRCpuProvisioningProvider,
    LocalASRCpuRepairEffectSink,
    LocalASRCpuRepairOwner,
    LocalASRCpuRepairRuntimeStateProvider,
    LocalASRCpuSelfProviderRebuild,
    LocalASRCpuSelfResume,
    LocalASRCpuStatusForProvider,
)
from puripuly_heart.app.services.local_asr_diagnostics import (
    LocalASRBasicLogSink,
    LocalASRDetailedLogSink,
    LocalASRDiagnosticsGpuEffectSink,
    LocalASRDiagnosticsOwner,
    LocalASRGpuDiscoveryOriginProvider,
)
from puripuly_heart.app.services.local_asr_readiness import (
    LocalASRReadinessAsyncEffect,
    LocalASRReadinessChannelProvider,
    LocalASRReadinessEffectSink,
    LocalASRReadinessFallback,
    LocalASRReadinessGpuPendingSink,
    LocalASRReadinessGpuStateProvider,
    LocalASRReadinessGpuValidator,
    LocalASRReadinessLoadLogSink,
    LocalASRReadinessOwner,
    LocalASRReadinessProviderAvailable,
    LocalASRReadinessProvisioningProvider,
    LocalASRReadinessStateProvider,
)
from puripuly_heart.app.services.manual_typing import ManualTypingOwner
from puripuly_heart.app.services.provider_credential_verification import (
    ProviderCredentialSelectedModelProvider,
    ProviderCredentialVerificationDiagnosticsSink,
    ProviderCredentialVerificationErrorSink,
    ProviderCredentialVerificationInteractionOwner,
    ProviderCredentialVerificationOwner,
)
from puripuly_heart.core.local_asr_provider_runtime import ProviderRuntimeRecoveryQuiesce
from puripuly_heart.core.peer_capture import (
    PeerCaptureAdmissionPort,
    PeerCaptureTargetResolverPort,
)
from puripuly_heart.core.runtime.peer_channel import (
    PeerCaptureAudioLoop,
    PeerCaptureSourceFactory,
    PeerCaptureVadFactory,
)
from puripuly_heart.core.runtime.self_capture import (
    SelfCaptureAudioLoop,
    SelfCaptureSourceFactory,
    SelfCaptureVadFactory,
)
from puripuly_heart.core.runtime.vrchat_osc_presence import (
    VrchatOscPresenceProbeOwner,
    VrchatOscProbeDiagnosticsSink,
)
from puripuly_heart.core.self_capture import SelfCaptureAdmissionPort
from puripuly_heart.core.vad.smart_turn import SmartTurnExperimentConfig


def create_microphone_test_capture_adapter(
    *,
    clock: object,
    log_sink: Callable[[str], None],
    meter_sink: Callable[
        [float, MicrophoneTestMeterCallback | None, int | None],
        Awaitable[None],
    ],
    route_observer: Callable[..., object],
    channel_decision: Callable[..., object],
    source_factory: Callable[..., object],
) -> MicrophoneTestCapturePort:
    from puripuly_heart.app.adapters.microphone_test_capture import (
        MicrophoneTestCaptureAdapter,
    )

    return MicrophoneTestCaptureAdapter(
        clock=clock,
        log_sink=log_sink,
        meter_sink=meter_sink,
        route_observer=route_observer,
        channel_decision=channel_decision,
        source_factory=source_factory,
    )


def create_manual_typing_owner(
    *,
    output_provider: Callable[[], SelfChatboxTypingPort | None],
    completion_provider: Callable[[object], object | None],
    log_detailed: Callable[[str], object],
    log_error: Callable[[str], object],
    idle_timeout_seconds: float,
    submit_timeout_seconds: float,
) -> ManualTypingOwner:
    return ManualTypingOwner(
        output_provider=output_provider,
        completion_provider=completion_provider,
        log_detailed=log_detailed,
        log_error=log_error,
        idle_timeout_seconds=idle_timeout_seconds,
        submit_timeout_seconds=submit_timeout_seconds,
    )


def create_provider_credential_verification_interaction_owner(
    *,
    verifier: ProviderVerifierPort,
    selected_model_provider: ProviderCredentialSelectedModelProvider,
    fallback_models: tuple[str, ...],
    low_latency: bool,
    diagnostics_sink: ProviderCredentialVerificationDiagnosticsSink | None = None,
    error_sink: ProviderCredentialVerificationErrorSink | None = None,
) -> ProviderCredentialVerificationInteractionOwner:
    return ProviderCredentialVerificationInteractionOwner(
        verification_owner=ProviderCredentialVerificationOwner(
            verifier=verifier,
            diagnostics_sink=diagnostics_sink,
        ),
        selected_model_provider=selected_model_provider,
        fallback_models=fallback_models,
        low_latency=low_latency,
        error_sink=error_sink,
    )


def create_gpu_runtime_interaction_owner(
    *,
    runtime_provider: GpuRuntimeProvider,
    provisioning_provider: GpuRuntimeProvisioningProvider,
    state_provider: GpuRuntimeInteractionStateProvider,
    presentation_sink: GpuRuntimePresentationSink,
    detailed_log_sink: GpuRuntimeDetailedLogSink,
    retry_activation: GpuRuntimeActivationRetry,
    install_diagnostic_sink: GpuRuntimeInstallDiagnosticSink | None = None,
) -> GpuRuntimeInteractionOwner:
    return GpuRuntimeInteractionOwner(
        runtime_provider=runtime_provider,
        provisioning_provider=provisioning_provider,
        state_provider=state_provider,
        presentation_sink=presentation_sink,
        detailed_log_sink=detailed_log_sink,
        retry_activation=retry_activation,
        install_diagnostic_sink=install_diagnostic_sink,
    )


def create_gpu_provider_recovery_application_owner(
    *,
    runtime_provider: GpuProviderRecoveryRuntimeProvider,
    pending_provider: GpuProviderRecoveryPendingProvider,
    pending_clear: GpuProviderRecoveryPendingClear,
    failure_sink: GpuProviderRecoveryFailureSink,
    runtime_state_sink: GpuProviderRecoveryStateSink,
    quiesce: ProviderRuntimeRecoveryQuiesce,
    self_owner_factory: GpuProviderRecoverySelfOwnerFactory,
    peer_owner_provider: GpuProviderRecoveryPeerOwnerProvider,
    self_state_sink: GpuProviderRecoverySelfStateSink,
    ensure_self_switch: GpuProviderRecoveryAsyncEffect,
    refresh_self: GpuProviderRecoveryAsyncEffect,
    refresh_peer: GpuProviderRecoveryAsyncEffect,
    diagnostic_sink: GpuProviderRecoveryDiagnosticSink,
) -> GpuProviderRecoveryApplicationOwner:
    return GpuProviderRecoveryApplicationOwner(
        recovery_owner=GpuProviderRecoveryOwner(
            diagnostic_sink=diagnostic_sink,
        ),
        runtime_provider=runtime_provider,
        pending_provider=pending_provider,
        pending_clear=pending_clear,
        failure_sink=failure_sink,
        runtime_state_sink=runtime_state_sink,
        quiesce=quiesce,
        self_owner_factory=self_owner_factory,
        peer_owner_provider=peer_owner_provider,
        self_state_sink=self_state_sink,
        ensure_self_switch=ensure_self_switch,
        refresh_self=refresh_self,
        refresh_peer=refresh_peer,
    )


def create_local_asr_diagnostics_owner(
    *,
    basic_log_sink: LocalASRBasicLogSink,
    detailed_log_sink: LocalASRDetailedLogSink,
    gpu_effect_sink: LocalASRDiagnosticsGpuEffectSink,
    gpu_discovery_origin_provider: LocalASRGpuDiscoveryOriginProvider,
    gpu_provider_id: str,
) -> LocalASRDiagnosticsOwner:
    return LocalASRDiagnosticsOwner(
        basic_log_sink=basic_log_sink,
        detailed_log_sink=detailed_log_sink,
        gpu_effect_sink=gpu_effect_sink,
        gpu_discovery_origin_provider=gpu_discovery_origin_provider,
        gpu_provider_id=gpu_provider_id,
    )


def create_local_asr_cpu_repair_owner(
    *,
    provisioning_provider: LocalASRCpuProvisioningProvider,
    state_provider: LocalASRCpuRepairRuntimeStateProvider,
    model_ids_for_provider: LocalASRCpuModelIdsForProvider,
    status_for_provider: LocalASRCpuStatusForProvider,
    effect_sink: LocalASRCpuRepairEffectSink,
    rebuild_self_provider: LocalASRCpuSelfProviderRebuild,
    resume_self: LocalASRCpuSelfResume,
    resume_peer: LocalASRCpuPeerResume,
) -> LocalASRCpuRepairOwner:
    return LocalASRCpuRepairOwner(
        provisioning_provider=provisioning_provider,
        state_provider=state_provider,
        model_ids_for_provider=model_ids_for_provider,
        status_for_provider=status_for_provider,
        effect_sink=effect_sink,
        rebuild_self_provider=rebuild_self_provider,
        resume_self=resume_self,
        resume_peer=resume_peer,
    )


def create_local_asr_readiness_owner(
    *,
    provisioning_provider: LocalASRReadinessProvisioningProvider,
    cpu_repair_owner: LocalASRCpuRepairOwner,
    state_provider: LocalASRReadinessStateProvider,
    effect_sink: LocalASRReadinessEffectSink,
    self_provider_available: LocalASRReadinessProviderAvailable,
    self_channel_provider: LocalASRReadinessChannelProvider,
    rebuild_self_provider: LocalASRReadinessAsyncEffect,
    probe_self_provider: LocalASRReadinessAsyncEffect,
    persist_manual_fallback: LocalASRReadinessFallback,
    validate_gpu_activation: LocalASRReadinessGpuValidator,
    gpu_state_provider: LocalASRReadinessGpuStateProvider,
    retain_gpu_pending: LocalASRReadinessGpuPendingSink,
    load_log_sink: LocalASRReadinessLoadLogSink,
) -> LocalASRReadinessOwner:
    return LocalASRReadinessOwner(
        provisioning_provider=provisioning_provider,
        cpu_repair_owner=cpu_repair_owner,
        state_provider=state_provider,
        effect_sink=effect_sink,
        self_provider_available=self_provider_available,
        self_channel_provider=self_channel_provider,
        rebuild_self_provider=rebuild_self_provider,
        probe_self_provider=probe_self_provider,
        persist_manual_fallback=persist_manual_fallback,
        validate_gpu_activation=validate_gpu_activation,
        gpu_state_provider=gpu_state_provider,
        retain_gpu_pending=retain_gpu_pending,
        load_log_sink=load_log_sink,
    )


def create_vrchat_osc_presence_probe_owner(
    *,
    presence_provider: Callable[[], VrchatOscPresencePort | None],
    port_provider: Callable[[], int],
    publish_notice: Callable[[bool], None],
    diagnostics_sink: VrchatOscProbeDiagnosticsSink | None = None,
) -> VrchatOscPresenceProbeOwner:
    return VrchatOscPresenceProbeOwner(
        presence_provider=presence_provider,
        port_provider=port_provider,
        publish_notice=publish_notice,
        diagnostics_sink=diagnostics_sink,
    )


def create_self_capture_source_adapter(
    *,
    log_detailed: Callable[..., object],
    wrap_source: Callable[[object], object],
) -> SelfCaptureSourceFactory:
    from puripuly_heart.app.adapters.self_capture_source import SelfCaptureSourceAdapter
    from puripuly_heart.config.audio_host_api import normalize_input_host_api
    from puripuly_heart.core.audio.source import (
        SoundDeviceAudioSource,
        determine_self_mic_capture_channels,
        resolve_sounddevice_input_device,
    )

    return SelfCaptureSourceAdapter(
        normalize_host_api=normalize_input_host_api,
        resolve_device=resolve_sounddevice_input_device,
        channel_decision=determine_self_mic_capture_channels,
        source_factory=SoundDeviceAudioSource,
        log_detailed=log_detailed,
        wrap_source=wrap_source,
    )


def create_self_capture_admission_adapter(
    *,
    state_provider: SelfCaptureAdmissionStateProvider,
    validate_gpu_activation: SelfCaptureGpuActivationValidator,
    effect_sink: SelfCaptureAdmissionEffectSink,
) -> SelfCaptureAdmissionPort:
    from puripuly_heart.app.adapters.self_capture_admission import (
        SelfCaptureAdmissionAdapter,
    )

    return SelfCaptureAdmissionAdapter(
        state_provider=state_provider,
        validate_gpu_activation=validate_gpu_activation,
        effect_sink=effect_sink,
    )


def create_self_capture_vad_adapter(
    *,
    log_detailed: Callable[[str], object],
    diagnostics_enabled: Callable[[], bool],
    smart_turn_config_provider: Callable[[], SmartTurnExperimentConfig] | None = None,
) -> SelfCaptureVadFactory:
    from puripuly_heart.app.adapters.self_capture_vad import SelfCaptureVadAdapter
    from puripuly_heart.core.vad.bundled import ensure_silero_vad_onnx
    from puripuly_heart.core.vad.gating import VadGating
    from puripuly_heart.core.vad.silero import SileroVadOnnx

    config_provider = smart_turn_config_provider or SmartTurnExperimentConfig.from_environment

    return SelfCaptureVadAdapter(
        model_path_resolver=ensure_silero_vad_onnx,
        engine_factory=SileroVadOnnx,
        gating_factory=VadGating,
        log_detailed=log_detailed,
        diagnostics_enabled=diagnostics_enabled,
        smart_turn_config_provider=config_provider,
    )


def create_self_capture_audio_loop_adapter(
    *,
    audio_gate_provider: Callable[[], object | None],
    log_detailed: Callable[[str], object],
    is_detailed_enabled: Callable[[], bool],
    smart_turn_config_provider: Callable[[], SmartTurnExperimentConfig] | None = None,
) -> SelfCaptureAudioLoop:
    from puripuly_heart.app.adapters.self_capture_audio_loop import (
        SelfCaptureAudioLoopAdapter,
    )
    from puripuly_heart.core.runtime.audio_vad_loop import run_audio_vad_loop
    from puripuly_heart.core.vad.smart_turn import (
        create_smart_turn_event_sink_factory,
    )

    config_provider = smart_turn_config_provider or SmartTurnExperimentConfig.from_environment

    return SelfCaptureAudioLoopAdapter(
        runner=run_audio_vad_loop,
        audio_gate_provider=audio_gate_provider,
        log_detailed=log_detailed,
        is_detailed_enabled=is_detailed_enabled,
        smart_turn_config_provider=config_provider,
        smart_turn_event_sink_factory=create_smart_turn_event_sink_factory(
            log_detailed=log_detailed,
        ),
    )


def create_peer_capture_source_adapter(
    *,
    log_detailed: Callable[[str], object],
    wrap_source: Callable[[object], object],
    is_detailed_enabled: Callable[[], bool],
) -> PeerCaptureSourceFactory:
    from puripuly_heart.app.adapters.peer_capture_source import PeerCaptureSourceAdapter
    from puripuly_heart.core.audio.desktop_pipeline import DesktopPeerPipeline
    from puripuly_heart.core.audio.desktop_source import DesktopLoopbackAudioSource
    from puripuly_heart.core.audio.process_identity import PsutilProcessIdentityWatcher
    from puripuly_heart.core.audio.process_source import ProcessAudioCaptureSource

    return PeerCaptureSourceAdapter(
        loopback_source_factory=DesktopLoopbackAudioSource,
        process_source_factory=ProcessAudioCaptureSource,
        process_watcher_factory=PsutilProcessIdentityWatcher,
        pipeline_factory=DesktopPeerPipeline,
        log_detailed=log_detailed,
        wrap_source=wrap_source,
        is_detailed_enabled=is_detailed_enabled,
    )


def create_peer_capture_target_resolver_adapter() -> PeerCaptureTargetResolverPort:
    from puripuly_heart.app.adapters.peer_capture_target_resolver import (
        PeerCaptureTargetResolverAdapter,
    )
    from puripuly_heart.config.process_capture_resolution import ProcessCaptureResolver
    from puripuly_heart.core.audio.process_identity import PsutilCurrentUserProcessSnapshots

    def create_process_resolver() -> ProcessCaptureResolver:
        return ProcessCaptureResolver(snapshots=PsutilCurrentUserProcessSnapshots())

    return PeerCaptureTargetResolverAdapter(resolver_factory=create_process_resolver)


def create_peer_capture_vad_adapter(
    *,
    log_detailed: Callable[[str], object],
    diagnostics_enabled: Callable[[], bool],
    smart_turn_config_provider: Callable[[], SmartTurnExperimentConfig] | None = None,
) -> PeerCaptureVadFactory:
    from puripuly_heart.app.adapters.peer_capture_vad import PeerCaptureVadAdapter
    from puripuly_heart.core.vad.bundled import ensure_silero_vad_onnx
    from puripuly_heart.core.vad.gating import create_peer_vad_gating
    from puripuly_heart.core.vad.silero import SileroVadOnnx

    config_provider = smart_turn_config_provider or SmartTurnExperimentConfig.from_environment

    return PeerCaptureVadAdapter(
        model_path_resolver=ensure_silero_vad_onnx,
        engine_factory=SileroVadOnnx,
        gating_factory=create_peer_vad_gating,
        log_detailed=log_detailed,
        diagnostics_enabled=diagnostics_enabled,
        smart_turn_config_provider=config_provider,
    )


def create_peer_capture_audio_loop_adapter(
    *,
    log_detailed: Callable[[str], object],
    is_detailed_enabled: Callable[[], bool],
    smart_turn_config_provider: Callable[[], SmartTurnExperimentConfig] | None = None,
) -> PeerCaptureAudioLoop:
    from puripuly_heart.app.adapters.peer_capture_audio_loop import (
        PeerCaptureAudioLoopAdapter,
    )
    from puripuly_heart.core.runtime.audio_vad_loop import run_audio_vad_loop
    from puripuly_heart.core.vad.smart_turn import (
        create_smart_turn_event_sink_factory,
    )

    config_provider = smart_turn_config_provider or SmartTurnExperimentConfig.from_environment

    return PeerCaptureAudioLoopAdapter(
        runner=run_audio_vad_loop,
        log_detailed=log_detailed,
        is_detailed_enabled=is_detailed_enabled,
        smart_turn_config_provider=config_provider,
        smart_turn_event_sink_factory=create_smart_turn_event_sink_factory(
            log_detailed=log_detailed,
        ),
    )


def create_peer_capture_admission_adapter(
    *,
    runtime_available: Callable[[], bool],
    ensure_local_ready: Callable[[], Awaitable[bool]],
) -> PeerCaptureAdmissionPort:
    from puripuly_heart.app.adapters.peer_capture_admission import (
        PeerCaptureAdmissionAdapter,
    )

    return PeerCaptureAdmissionAdapter(
        runtime_available=runtime_available,
        ensure_local_ready=ensure_local_ready,
    )


def create_peer_capture_vad_sink_adapter(
    *,
    runtime_provider: Callable[[], object | None],
) -> object:
    from puripuly_heart.app.adapters.peer_capture_vad_sink import (
        PeerCaptureVadSinkAdapter,
    )

    return PeerCaptureVadSinkAdapter(runtime_provider=runtime_provider)


def create_self_capture_vad_sink_adapter(
    *,
    runtime_provider: Callable[[], object | None],
) -> object:
    from puripuly_heart.app.adapters.self_capture_vad_sink import (
        SelfCaptureVadSinkAdapter,
    )

    return SelfCaptureVadSinkAdapter(runtime_provider=runtime_provider)


def create_provider_verifier() -> ProviderVerifierPort:
    from puripuly_heart.app.adapters.provider_verifier import ProviderVerifierAdapter

    return ProviderVerifierAdapter()


def create_desktop_overlay_policy() -> DesktopOverlayPolicy:
    from puripuly_heart.config.settings import (
        DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA,
        DESKTOP_FLET_DEFAULT_SIZE_PRESET,
        DESKTOP_FLET_DEFAULT_TEXT_SCALE,
        DESKTOP_FLET_MIN_HEIGHT,
        DESKTOP_FLET_MIN_WIDTH,
        DESKTOP_FLET_SIZE_PRESETS,
    )

    return DesktopOverlayPolicy(
        minimum_width=DESKTOP_FLET_MIN_WIDTH,
        minimum_height=DESKTOP_FLET_MIN_HEIGHT,
        default_text_scale=DESKTOP_FLET_DEFAULT_TEXT_SCALE,
        default_background_alpha=DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA,
        default_size_preset=DESKTOP_FLET_DEFAULT_SIZE_PRESET,
        size_presets=DESKTOP_FLET_SIZE_PRESETS,
    )


def create_windows_desktop_work_area() -> DesktopWorkAreaPort:
    from puripuly_heart.app.adapters.windows_desktop_work_area import (
        WindowsDesktopWorkAreaAdapter,
    )

    return WindowsDesktopWorkAreaAdapter()
