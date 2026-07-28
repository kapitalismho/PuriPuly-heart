from __future__ import annotations

from collections.abc import Awaitable, Callable

from puripuly_heart.app.ports.microphone_test import (
    MicrophoneTestCapturePort,
    MicrophoneTestMeterCallback,
)
from puripuly_heart.app.ports.provider_verifier import ProviderVerifierPort
from puripuly_heart.core.peer_capture import PeerCaptureTargetResolverPort
from puripuly_heart.core.runtime.peer_channel import (
    PeerCaptureAudioLoop,
    PeerCaptureSourceFactory,
    PeerCaptureVadFactory,
)
from puripuly_heart.core.runtime.self_capture import (
    SelfCaptureSourceFactory,
    SelfCaptureVadFactory,
)


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


def create_provider_verifier() -> ProviderVerifierPort:
    from puripuly_heart.app.adapters.provider_verifier import ProviderVerifierAdapter

    return ProviderVerifierAdapter()
