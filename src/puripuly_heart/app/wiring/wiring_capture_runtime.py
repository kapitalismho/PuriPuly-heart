from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from puripuly_heart.app.ports.capture_vad_runtime import (
    PeerCaptureVadEventRuntime,
    SelfCaptureVadEventRuntime,
)
from puripuly_heart.app.ports.provider_channel_runtime import ProviderChannelResetPort
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.audio.diagnostics import AudioFaultProfile, FaultInjectingAudioSource
from puripuly_heart.core.audio.gate import VrcMicAudioGate
from puripuly_heart.core.audio.source import AudioSource
from puripuly_heart.core.clock import Clock
from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimePort,
    ProviderRuntimeBuildRequest,
)
from puripuly_heart.core.peer_capture import (
    PeerCaptureDiagnostic,
    PeerCaptureSessionSnapshot,
)
from puripuly_heart.core.runtime.peer_channel import PeerCaptureSessionOwner
from puripuly_heart.core.runtime.self_capture import SelfCaptureSessionOwner
from puripuly_heart.core.self_capture import (
    SelfCaptureAdmissionPort,
    SelfCaptureDiagnostic,
    SelfCaptureDiagnosticEvent,
    SelfCaptureSessionConfig,
    SelfCaptureSessionSnapshot,
)

from .root import (
    compose_peer_capture_session_owner,
    compose_self_capture_session_owner,
)
from .wiring_composition import (
    create_peer_capture_admission_adapter,
    create_peer_capture_audio_loop_adapter,
    create_peer_capture_source_adapter,
    create_peer_capture_target_resolver_adapter,
    create_peer_capture_vad_adapter,
    create_peer_capture_vad_sink_adapter,
    create_self_capture_audio_loop_adapter,
    create_self_capture_source_adapter,
    create_self_capture_vad_adapter,
    create_self_capture_vad_sink_adapter,
)
from .wiring_stt_factory import (
    build_peer_stt_provider_request,
    build_self_stt_provider_request_from_vnext,
)


@dataclass(frozen=True, slots=True)
class CaptureDiagnosticsAdapter:
    debug_allowed: Callable[[], bool]
    capture_fault_profile: Callable[[], str]
    log_diagnostic: Callable[[str], None]
    log_basic: Callable[[str], None]

    def wrap_source(self, source: AudioSource) -> AudioSource:
        return FaultInjectingAudioSource(
            source=source,
            fault_profile_provider=lambda: (
                self.capture_fault_profile()
                if self.debug_allowed()
                else AudioFaultProfile.NONE.value
            ),
        )

    def self_capture(self, diagnostic: SelfCaptureDiagnostic) -> None:
        if diagnostic.event is not SelfCaptureDiagnosticEvent.FAILURE:
            return
        fields = [
            "[SelfCapture] failed",
            f"state={diagnostic.state.value}",
        ]
        if diagnostic.provider_id is not None:
            fields.append(f"provider={diagnostic.provider_id}")
        if diagnostic.reason is not None:
            fields.append(f"cause={diagnostic.reason.value}")
        if diagnostic.detail is not None:
            fields.append(f"detail={diagnostic.detail}")
        self.log_basic(" ".join(fields))
        self.log_diagnostic(" ".join(fields))


@dataclass(frozen=True, slots=True)
class CaptureOwnerFactory:
    canonical_provider: Callable[[], AppSettingsVNext | None]
    self_admission: SelfCaptureAdmissionPort
    ensure_peer_local_ready: Callable[[int | None], Awaitable[bool]]
    clock: Clock
    log_basic: Callable[[str], None]
    log_diagnostic: Callable[[str], None]
    source_wrapper: Callable[[AudioSource], AudioSource]
    self_state_sink: Callable[[SelfCaptureSessionSnapshot], None]
    self_diagnostic_sink: Callable[[SelfCaptureDiagnostic], None]
    peer_state_sink: Callable[[PeerCaptureSessionSnapshot], None]
    peer_diagnostic_sink: Callable[[PeerCaptureDiagnostic], None]
    local_asr_diagnostic_sink: Callable[[object], None]

    def compose_self(
        self,
        vad_runtime: SelfCaptureVadEventRuntime | None,
        provider_runtime: LocalASRProviderRuntimePort | None,
        channel_reset: ProviderChannelResetPort | None,
        audio_gate: VrcMicAudioGate | None,
    ) -> SelfCaptureSessionOwner:
        return compose_self_capture_session_owner(
            provider_runtime=provider_runtime,
            channel_reset=channel_reset,
            admission=self.self_admission,
            provider_request_factory=self.self_provider_request,
            source_factory=create_self_capture_source_adapter(
                log_diagnostic=self.log_diagnostic,
                wrap_source=self.source_wrapper,
            ),
            vad_factory=create_self_capture_vad_adapter(),
            run_audio_loop=create_self_capture_audio_loop_adapter(
                audio_gate_provider=lambda: audio_gate,
                log_basic=self.log_basic,
            ),
            vad_sink=create_self_capture_vad_sink_adapter(runtime_provider=lambda: vad_runtime),
            state_changed=self.self_state_sink,
            diagnostic_sink=self.self_diagnostic_sink,
            audio_gate_reset=audio_gate.reset if audio_gate is not None else None,
        )

    def compose_peer(
        self,
        vad_runtime: PeerCaptureVadEventRuntime,
        provider_runtime: LocalASRProviderRuntimePort,
        channel_reset: ProviderChannelResetPort,
    ) -> PeerCaptureSessionOwner:
        return compose_peer_capture_session_owner(
            provider_runtime=provider_runtime,
            channel_reset=channel_reset,
            admission=create_peer_capture_admission_adapter(
                runtime_available=lambda: (
                    self.canonical_provider() is not None and vad_runtime is not None
                ),
                ensure_local_ready=lambda: self.ensure_peer_local_ready(None),
            ),
            target_resolver=create_peer_capture_target_resolver_adapter(),
            clock=self.clock,
            provider_request_factory=lambda config, warmup: build_peer_stt_provider_request(
                config,
                gpu_device_id=self._canonical().intent.stt.gpu_device_id,
                warmup=warmup,
            ),
            source_factory=create_peer_capture_source_adapter(
                log_diagnostic=self.log_diagnostic,
                wrap_source=self.source_wrapper,
            ),
            vad_factory=create_peer_capture_vad_adapter(),
            run_audio_loop=create_peer_capture_audio_loop_adapter(
                log_basic=self.log_basic,
            ),
            vad_sink=create_peer_capture_vad_sink_adapter(runtime_provider=lambda: vad_runtime),
            state_changed=self.peer_state_sink,
            diagnostic_sink=self.peer_diagnostic_sink,
            local_asr_diagnostic_sink=self.local_asr_diagnostic_sink,
        )

    def self_provider_request(
        self,
        config: SelfCaptureSessionConfig,
        warmup: bool,
    ) -> ProviderRuntimeBuildRequest:
        _ = config
        return build_self_stt_provider_request_from_vnext(self._canonical(), warmup=warmup)

    def _canonical(self) -> AppSettingsVNext:
        settings = self.canonical_provider()
        if settings is None:
            raise RuntimeError("Capture provider request requires canonical settings")
        return settings


__all__ = [
    "CaptureDiagnosticsAdapter",
    "CaptureOwnerFactory",
]
