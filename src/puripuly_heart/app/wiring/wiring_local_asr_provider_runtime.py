from __future__ import annotations

import contextlib
import inspect
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from puripuly_heart.app.adapters.gpu_worker_process import DefaultGpuWorkerProcessFactory
from puripuly_heart.config.provider_values import (
    STTProviderName,
    custom_stt_selection_for_provider,
    is_custom_stt_provider,
)
from puripuly_heart.core.audio.listen_delivery import (
    LISTEN_RETAINED_SEGMENT_SLOTS,
    ListenDeliveryController,
)
from puripuly_heart.core.audio.ownership import AudioSegmentSettingsSnapshot
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
from puripuly_heart.core.stt.backend import STTScopedTurnSession, STTSessionProjection
from puripuly_heart.core.stt.custom import validate_peer_custom_stt_configuration
from puripuly_heart.core.stt.notifications import FinalTranscriptSuppressedNotification
from puripuly_heart.core.stt.scoped_engine import (
    PermanentSTTScopedSessionError,
    ScopedRecognitionEngine,
    STTRecognitionWatchdogs,
    STTRetentionProfile,
)
from puripuly_heart.core.stt.scoped_normalizer import STTNormalizationDiagnostic

from .wiring_stt_factory import create_stt_backend_from_resolved_config

FinalTranscriptSuppressedSink = Callable[[FinalTranscriptSuppressedNotification], object]
FaultProfileProvider = Callable[[], object]
DiagnosticsEnabled = Callable[[], bool]
_ACTUAL_DEGRADATION_REASONS = frozenset(
    {
        "language_run_conservation_fallback",
        "language_run_limit_fallback",
        "invalid_language_run_fallback",
        "speaker_run_conservation_fallback",
        "speaker_run_limit_fallback",
        "invalid_speaker_run_fallback",
    }
)


@dataclass(slots=True)
class SharedSTTProviderFactory(ProviderRuntimeProviderFactoryPort):
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
    ) -> object:
        config = request.config
        provider_name = STTProviderName(config.provider)
        if config.channel == "peer" and is_custom_stt_provider(provider_name):
            provider_mode, _compatibility = custom_stt_selection_for_provider(
                provider_name,
                stored_mode=str(config.provider_options.get("mode") or ""),
                stored_compatibility=str(config.provider_options.get("compatibility") or ""),
            )
            validate_peer_custom_stt_configuration(
                mode=provider_mode,
                extra=config.provider_options.get("extra"),
            )
        backend = create_stt_backend_from_resolved_config(
            config,
            secrets=self.secrets,
            diagnostics_enabled=self.diagnostics_enabled,
            gpu_runtime=cast(SharedGpuASRRuntime, gpu_runtime),
            gpu_model_path=self.gpu_model_path,
            gpu_device_id=request.gpu_device_id,
        )
        if request.recognition_projection != "scoped":
            raise ValueError("production recognition requires the scoped projection")
        if request.provider_signature is None or request.runtime_signature is None:
            raise ValueError("scoped provider request requires configuration scope signatures")

        async def open_scoped_session(
            _settings: AudioSegmentSettingsSnapshot,
            provider_epoch_id: str,
        ) -> STTScopedTurnSession:
            session = await backend.open_session(
                projection=STTSessionProjection("scoped", provider_epoch_id)
            )
            if isinstance(session, STTScopedTurnSession) and isinstance(
                getattr(session, "inner", session),
                STTScopedTurnSession,
            ):
                return session
            try:
                await session.close()
            finally:
                raise PermanentSTTScopedSessionError(
                    f"{provider_name.value} does not implement scoped recognition"
                )

        async def close_backend() -> None:
            close = getattr(backend, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result

        channel = str(config.channel)
        runtime_logging = self.runtime_logging
        provider_id = str(config.provider)

        def _normalization_diagnostic_sink(
            diagnostic: STTNormalizationDiagnostic,
        ) -> None:
            service = runtime_logging
            if service is None:
                return
            reason = diagnostic.reason
            degraded = reason in _ACTUAL_DEGRADATION_REASONS
            message = (
                "[STT][Normalization] "
                f"channel={channel} provider={provider_id} "
                f"reason={reason} degraded={str(degraded).lower()}"
            )
            with contextlib.suppress(Exception):
                if degraded:
                    service.emit_basic(message, level=logging.WARNING)
                else:
                    service.emit_detailed(message)

        return ScopedRecognitionEngine(
            channel=config.channel,
            session_factory=open_scoped_session,
            watchdog_resolver=lambda _settings: _recognition_watchdogs(config),
            accepted_settings_scope=(
                config.provider,
                request.provider_signature,
                request.runtime_signature,
            ),
            retention_profile_resolver=lambda settings: _recognition_retention_profile(
                config,
                settings,
            ),
            backend_close=close_backend,
            event_drain_timeout_s=config.drain_timeout_s,
            terminal_failure_sink=on_terminal_failure,
            diagnostic_sink=(
                _normalization_diagnostic_sink if runtime_logging is not None else None
            ),
        )


def _recognition_watchdogs(config: object) -> STTRecognitionWatchdogs:
    provider_id = str(getattr(config, "provider"))
    drain_timeout_s = float(getattr(config, "drain_timeout_s"))
    local_provider_ids = {
        STTProviderName.LOCAL_CPU_AUTO.value,
        STTProviderName.LOCAL_PARAKEET_V3.value,
        STTProviderName.LOCAL_PARAKEET_JAPANESE.value,
        STTProviderName.LOCAL_QWEN.value,
        STTProviderName.LOCAL_QWEN_GPU.value,
    }
    if provider_id in local_provider_ids:
        readiness_timeout_s = 60.0
        final_timeout_s = 30.0
    elif provider_id == STTProviderName.CUSTOM_OFFLINE.value:
        readiness_timeout_s = 30.0
        final_timeout_s = 50.0
    elif provider_id == STTProviderName.GEMINI_TRANSCRIBE.value:
        readiness_timeout_s = 30.0
        final_timeout_s = 2.0
    elif provider_id == STTProviderName.QWEN_AUDIO.value:
        readiness_timeout_s = 30.0
        final_timeout_s = 5.0
    elif provider_id == STTProviderName.DEEPGRAM.value:
        readiness_timeout_s = 30.0
        final_timeout_s = drain_timeout_s * 2.0 + 5.0
    else:
        readiness_timeout_s = 30.0
        final_timeout_s = 20.0
    return STTRecognitionWatchdogs(
        readiness_timeout_s=readiness_timeout_s,
        write_timeout_s=5.0,
        final_timeout_s=final_timeout_s,
        drain_timeout_s=drain_timeout_s,
    )


def _recognition_retention_profile(
    config: object,
    settings: AudioSegmentSettingsSnapshot,
) -> STTRetentionProfile:
    sample_rate_hz = int(getattr(config, "sample_rate_hz"))
    channel = str(getattr(config, "channel"))
    provider_id = str(getattr(config, "provider"))
    if channel == "self":
        max_samples = 2_880_000
    else:
        max_context_ms = max(
            settings.vad_pre_roll_ms,
            ListenDeliveryController.HARD_CUT_OVERLAP_MS,
        )
        max_samples = int(
            sample_rate_hz
            * (ListenDeliveryController.HARD_LIMIT_S + max_context_ms / 1000.0)
            * LISTEN_RETAINED_SEGMENT_SLOTS
        )
    provider_options = getattr(config, "provider_options", {})
    custom_mode = (
        str(provider_options.get("mode", ""))
        if isinstance(provider_options, dict | Mapping)
        else ""
    )
    custom_offline = provider_id == STTProviderName.CUSTOM_OFFLINE.value or (
        is_custom_stt_provider(provider_id) and custom_mode == "offline"
    )
    retained_until_terminal = (
        provider_id
        in {
            STTProviderName.LOCAL_CPU_AUTO.value,
            STTProviderName.LOCAL_PARAKEET_V3.value,
            STTProviderName.LOCAL_PARAKEET_JAPANESE.value,
            STTProviderName.LOCAL_QWEN.value,
            STTProviderName.LOCAL_QWEN_GPU.value,
        }
        or custom_offline
    )
    return STTRetentionProfile(
        max_retained_samples=max_samples,
        max_retained_bytes=max_samples * 4,
        release_after_write=not retained_until_terminal,
        retained_bytes_per_sample=(
            4
            if provider_id
            in {
                STTProviderName.LOCAL_CPU_AUTO.value,
                STTProviderName.LOCAL_PARAKEET_V3.value,
                STTProviderName.LOCAL_PARAKEET_JAPANESE.value,
                STTProviderName.LOCAL_QWEN.value,
                STTProviderName.LOCAL_QWEN_GPU.value,
            }
            else 2
        ),
    )


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
        if isinstance(factory, SharedSTTProviderFactory):
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


__all__ = [
    "LocalASRProviderRuntimeFactory",
    "SharedSTTProviderFactory",
]
