from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from puripuly_heart.app.adapters.gpu_worker_process import DefaultGpuWorkerProcessFactory
from puripuly_heart.config.settings import STTProviderName
from puripuly_heart.core.clock import Clock
from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimeCallbacks,
    LocalASRProviderRuntimePort,
    ProviderGpuRuntimePort,
    ProviderRuntimeBuildRequest,
    ProviderRuntimeProviderFactoryPort,
    ProviderRuntimeTerminalFailureSink,
)
from puripuly_heart.core.local_asr_provisioning import LocalASRProvisioningPort
from puripuly_heart.core.runtime.gpu_asr import SharedGpuASRRuntime
from puripuly_heart.core.runtime.local_asr_provider_runtime import (
    LocalASRProviderRuntimeOwner,
    ProviderRuntimeDiagnosticSink,
    ProviderRuntimeStateChanged,
)
from puripuly_heart.core.runtime_logging import SessionRuntimeLoggingService
from puripuly_heart.core.storage.secrets import SecretStore
from puripuly_heart.core.stt.controller import (
    FinalTranscriptSuppressedNotification,
    ManagedSTTProvider,
)

from .wiring_stt_factory import create_stt_backend_from_resolved_config

FinalTranscriptSuppressedSink = Callable[[FinalTranscriptSuppressedNotification], object]
DiagnosticsEnabled = Callable[[], bool]
FaultProfileProvider = Callable[[], object]


@dataclass(slots=True)
class ManagedSTTProviderFactory(ProviderRuntimeProviderFactoryPort):
    secrets: SecretStore
    clock: Clock
    reset_deadline_s: float
    gpu_model_path: Path
    diagnostics_enabled: DiagnosticsEnabled | None = None
    on_final_transcript_suppressed: FinalTranscriptSuppressedSink | None = None
    runtime_logging: SessionRuntimeLoggingService | None = None
    fault_profile_provider: FaultProfileProvider | None = None
    event_ingress_observer: Callable[..., object] | None = None

    async def create(
        self,
        request: ProviderRuntimeBuildRequest,
        *,
        gpu_runtime: ProviderGpuRuntimePort,
        on_terminal_failure: ProviderRuntimeTerminalFailureSink | None = None,
    ) -> ManagedSTTProvider:
        config = request.config
        backend = create_stt_backend_from_resolved_config(
            config,
            secrets=self.secrets,
            diagnostics_enabled=self.diagnostics_enabled,
            gpu_runtime=cast(SharedGpuASRRuntime, gpu_runtime),
            gpu_model_path=self.gpu_model_path,
            gpu_device_id=request.gpu_device_id,
        )
        provider = ManagedSTTProvider(
            backend=backend,
            sample_rate_hz=config.sample_rate_hz,
            stt_provider_name=STTProviderName(config.provider),
            channel=config.channel,
            clock=self.clock,
            reset_deadline_s=self.reset_deadline_s,
            drain_timeout_s=config.drain_timeout_s,
            bridging_ms=max(
                1,
                config.vad_pre_roll_ms if config.channel == "peer" else config.ring_buffer_ms,
            ),
            on_terminal_failure=on_terminal_failure,
            on_final_transcript_suppressed=self.on_final_transcript_suppressed,
            runtime_logging=self.runtime_logging,
            stt_input_fault_profile_provider=self.fault_profile_provider,
            event_ingress_observer=self.event_ingress_observer,
        )
        if request.session_options is not None:
            await provider.reconfigure_session_options(request.session_options)
        return provider


@dataclass(slots=True)
class LocalASRProviderRuntimeFactory:
    provider_factory: ProviderRuntimeProviderFactoryPort
    provisioning: LocalASRProvisioningPort
    clock: Clock
    state_changed: ProviderRuntimeStateChanged | None = None
    diagnostic_sink: ProviderRuntimeDiagnosticSink | None = None

    def bind_stt_event_ingress_observer(
        self,
        observer: Callable[..., object] | None,
    ) -> None:
        factory = self.provider_factory
        if isinstance(factory, ManagedSTTProviderFactory):
            factory.event_ingress_observer = observer

    def create(
        self,
        callbacks: LocalASRProviderRuntimeCallbacks,
    ) -> LocalASRProviderRuntimePort:
        return LocalASRProviderRuntimeOwner(
            provider_factory=self.provider_factory,
            gpu_runtime_factory=lambda diagnostic_sink: SharedGpuASRRuntime(
                process_factory=DefaultGpuWorkerProcessFactory(),
                clock=self.clock,
                diagnostic_sink=diagnostic_sink,
            ),
            provisioning=self.provisioning,
            self_event_handler=callbacks.self_event_handler,
            peer_event_handler=callbacks.peer_event_handler,
            retired_event_handler=callbacks.retired_event_handler,
            self_exception_handler=callbacks.self_exception_handler,
            peer_exception_handler=callbacks.peer_exception_handler,
            state_changed=self.state_changed,
            diagnostic_sink=self.diagnostic_sink,
        )


__all__ = ["LocalASRProviderRuntimeFactory", "ManagedSTTProviderFactory"]
