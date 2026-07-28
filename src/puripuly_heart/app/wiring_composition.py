from __future__ import annotations

from collections.abc import Awaitable, Callable

from puripuly_heart.app.ports.microphone_test import (
    MicrophoneTestCapturePort,
    MicrophoneTestMeterCallback,
)
from puripuly_heart.app.ports.provider_verifier import ProviderVerifierPort
from puripuly_heart.core.runtime.peer_channel import PeerCaptureSourceFactory
from puripuly_heart.core.runtime.self_capture import SelfCaptureSourceFactory


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


def create_provider_verifier() -> ProviderVerifierPort:
    from puripuly_heart.app.adapters.provider_verifier import ProviderVerifierAdapter

    return ProviderVerifierAdapter()
