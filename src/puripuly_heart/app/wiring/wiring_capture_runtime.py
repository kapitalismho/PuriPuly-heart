from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from puripuly_heart.app.ports.capture_vad_runtime import (
    PeerCaptureVadEventRuntime,
    SelfCaptureVadEventRuntime,
)
from puripuly_heart.app.ports.provider_channel_runtime import ProviderChannelResetPort
from puripuly_heart.config.settings import AppSettings
from puripuly_heart.core.audio.diagnostics import AudioFaultProfile, DiagnosticAudioSource
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
    SelfCaptureSessionConfig,
    SelfCaptureSessionSnapshot,
)
from puripuly_heart.core.vad.smart_turn import SmartTurnExperimentConfig

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
    build_self_stt_provider_request,
)


@dataclass(frozen=True, slots=True)
class CaptureDiagnosticsAdapter:
    detailed_enabled: Callable[[], bool]
    debug_allowed: Callable[[], bool]
    capture_fault_profile: Callable[[], str]
    log_detailed: Callable[[str], None]

    def wrap_source(
        self,
        source: AudioSource,
        *,
        channel_label: str,
    ) -> AudioSource:
        def extra_fields() -> dict[str, object]:
            return {
                "queue_drops": getattr(source, "queue_drop_count", 0),
                "callback_statuses": getattr(source, "callback_status_count", 0),
                "last_callback_status": getattr(source, "last_callback_status", None),
                "resolved_device_name": getattr(source, "resolved_device_name", None),
                "resolved_device_index": getattr(source, "resolved_device_index", None),
                "resolved_channels": getattr(source, "resolved_channels", None),
                "actual_sample_rate_hz": getattr(source, "actual_sample_rate_hz", None),
                "used_default_fallback": getattr(source, "used_default_fallback", None),
            }

        return DiagnosticAudioSource(
            source=source,
            channel_label=channel_label,
            is_detailed_enabled=self.detailed_enabled,
            log_detailed=self.log_detailed,
            fault_profile_provider=lambda: (
                self.capture_fault_profile()
                if self.debug_allowed()
                else AudioFaultProfile.NONE.value
            ),
            extra_fields_provider=extra_fields,
        )

    def self_capture(self, diagnostic: SelfCaptureDiagnostic) -> None:
        fields = [
            f"event={diagnostic.event.value}",
            f"generation={diagnostic.generation}",
            f"state={diagnostic.state.value}",
        ]
        if diagnostic.provider_id is not None:
            fields.append(f"provider={diagnostic.provider_id}")
        if diagnostic.reason is not None:
            fields.append(f"reason={diagnostic.reason.value}")
        if diagnostic.detail is not None:
            fields.append(f"detail={diagnostic.detail}")
        self.log_detailed(f"[SelfCapture] {' '.join(fields)}")


@dataclass(frozen=True, slots=True)
class CaptureOwnerFactory:
    settings_provider: Callable[[], AppSettings | None]
    self_admission: SelfCaptureAdmissionPort
    ensure_peer_local_ready: Callable[[int | None], Awaitable[bool]]
    clock: Clock
    log_detailed: Callable[[str], None]
    detailed_enabled: Callable[[], bool]
    source_wrapper: Callable[[AudioSource, str], AudioSource]
    self_state_sink: Callable[[SelfCaptureSessionSnapshot], None]
    self_diagnostic_sink: Callable[[SelfCaptureDiagnostic], None]
    peer_state_sink: Callable[[PeerCaptureSessionSnapshot], None]
    peer_diagnostic_sink: Callable[[PeerCaptureDiagnostic], None]
    local_asr_diagnostic_sink: Callable[[object], None]
    smart_turn_config_provider: Callable[[], SmartTurnExperimentConfig] = (
        SmartTurnExperimentConfig.from_environment
    )

    def compose_self(
        self,
        vad_runtime: SelfCaptureVadEventRuntime | None,
        provider_runtime: LocalASRProviderRuntimePort | None,
        channel_reset: ProviderChannelResetPort | None,
        audio_gate: VrcMicAudioGate | None,
    ) -> SelfCaptureSessionOwner:
        smart_turn_config = self.smart_turn_config_provider()
        return compose_self_capture_session_owner(
            provider_runtime=provider_runtime,
            channel_reset=channel_reset,
            admission=self.self_admission,
            provider_request_factory=self.self_provider_request,
            source_factory=create_self_capture_source_adapter(
                log_detailed=self.log_detailed,
                wrap_source=lambda source: self.source_wrapper(source, "self"),
            ),
            vad_factory=create_self_capture_vad_adapter(
                log_detailed=self.log_detailed,
                diagnostics_enabled=self.detailed_enabled,
                smart_turn_config_provider=lambda: smart_turn_config,
            ),
            run_audio_loop=create_self_capture_audio_loop_adapter(
                audio_gate_provider=lambda: audio_gate,
                log_detailed=self.log_detailed,
                is_detailed_enabled=self.detailed_enabled,
                smart_turn_config_provider=lambda: smart_turn_config,
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
        smart_turn_config = self.smart_turn_config_provider()
        return compose_peer_capture_session_owner(
            provider_runtime=provider_runtime,
            channel_reset=channel_reset,
            admission=create_peer_capture_admission_adapter(
                runtime_available=lambda: (
                    self.settings_provider() is not None and vad_runtime is not None
                ),
                ensure_local_ready=lambda: self.ensure_peer_local_ready(None),
            ),
            target_resolver=create_peer_capture_target_resolver_adapter(),
            clock=self.clock,
            provider_request_factory=lambda config, warmup: build_peer_stt_provider_request(
                config,
                gpu_device_id=self._settings().stt.gpu_device_id,
                warmup=warmup,
            ),
            source_factory=create_peer_capture_source_adapter(
                log_detailed=self.log_detailed,
                wrap_source=lambda source: self.source_wrapper(source, "peer"),
                is_detailed_enabled=self.detailed_enabled,
            ),
            vad_factory=create_peer_capture_vad_adapter(
                log_detailed=self.log_detailed,
                diagnostics_enabled=self.detailed_enabled,
                smart_turn_config_provider=lambda: smart_turn_config,
            ),
            run_audio_loop=create_peer_capture_audio_loop_adapter(
                log_detailed=self.log_detailed,
                is_detailed_enabled=self.detailed_enabled,
                smart_turn_config_provider=lambda: smart_turn_config,
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
        return build_self_stt_provider_request(self._settings(), warmup=warmup)

    def _settings(self) -> AppSettings:
        settings = self.settings_provider()
        if settings is None:
            raise RuntimeError("Capture provider request requires settings")
        return settings


__all__ = [
    "CaptureDiagnosticsAdapter",
    "CaptureOwnerFactory",
]
