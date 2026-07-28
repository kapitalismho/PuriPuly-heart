from __future__ import annotations

from collections.abc import Awaitable, Callable

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
from puripuly_heart.app.services.local_asr_gpu_provisioning import (
    LocalASRGpuActivationRetry,
    LocalASRGpuProvisioningDiagnosticSink,
    LocalASRGpuProvisioningEffectSink,
    LocalASRGpuProvisioningOwner,
    LocalASRGpuProvisioningStateProvider,
    LocalASRProvisioningProvider,
)
from puripuly_heart.app.services.manual_typing import ManualTypingOwner
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


def create_local_asr_gpu_provisioning_owner(
    *,
    provisioning_provider: LocalASRProvisioningProvider,
    state_provider: LocalASRGpuProvisioningStateProvider,
    effect_sink: LocalASRGpuProvisioningEffectSink,
    retry_activation: LocalASRGpuActivationRetry,
    diagnostic_sink: LocalASRGpuProvisioningDiagnosticSink | None = None,
) -> LocalASRGpuProvisioningOwner:
    return LocalASRGpuProvisioningOwner(
        provisioning_provider=provisioning_provider,
        state_provider=state_provider,
        effect_sink=effect_sink,
        retry_activation=retry_activation,
        diagnostic_sink=diagnostic_sink,
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
) -> SelfCaptureVadFactory:
    from puripuly_heart.app.adapters.self_capture_vad import SelfCaptureVadAdapter
    from puripuly_heart.core.vad.bundled import ensure_silero_vad_onnx
    from puripuly_heart.core.vad.gating import VadGating
    from puripuly_heart.core.vad.silero import SileroVadOnnx

    return SelfCaptureVadAdapter(
        model_path_resolver=ensure_silero_vad_onnx,
        engine_factory=SileroVadOnnx,
        gating_factory=VadGating,
        log_detailed=log_detailed,
        diagnostics_enabled=diagnostics_enabled,
    )


def create_self_capture_audio_loop_adapter(
    *,
    audio_gate_provider: Callable[[], object | None],
    log_detailed: Callable[[str], object],
    is_detailed_enabled: Callable[[], bool],
) -> SelfCaptureAudioLoop:
    from puripuly_heart.app.adapters.self_capture_audio_loop import (
        SelfCaptureAudioLoopAdapter,
    )
    from puripuly_heart.core.runtime.audio_vad_loop import run_audio_vad_loop

    return SelfCaptureAudioLoopAdapter(
        runner=run_audio_vad_loop,
        audio_gate_provider=audio_gate_provider,
        log_detailed=log_detailed,
        is_detailed_enabled=is_detailed_enabled,
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
) -> PeerCaptureVadFactory:
    from puripuly_heart.app.adapters.peer_capture_vad import PeerCaptureVadAdapter
    from puripuly_heart.core.vad.bundled import ensure_silero_vad_onnx
    from puripuly_heart.core.vad.gating import create_peer_vad_gating
    from puripuly_heart.core.vad.silero import SileroVadOnnx

    return PeerCaptureVadAdapter(
        model_path_resolver=ensure_silero_vad_onnx,
        engine_factory=SileroVadOnnx,
        gating_factory=create_peer_vad_gating,
        log_detailed=log_detailed,
        diagnostics_enabled=diagnostics_enabled,
    )


def create_peer_capture_audio_loop_adapter(
    *,
    log_detailed: Callable[[str], object],
    is_detailed_enabled: Callable[[], bool],
) -> PeerCaptureAudioLoop:
    from puripuly_heart.app.adapters.peer_capture_audio_loop import (
        PeerCaptureAudioLoopAdapter,
    )
    from puripuly_heart.core.runtime.audio_vad_loop import run_audio_vad_loop

    return PeerCaptureAudioLoopAdapter(
        runner=run_audio_vad_loop,
        log_detailed=log_detailed,
        is_detailed_enabled=is_detailed_enabled,
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
