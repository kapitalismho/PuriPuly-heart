from __future__ import annotations

from dataclasses import dataclass

from puripuly_heart.app.ports.self_capture_admission import (
    SelfCaptureAdmissionEffect,
    SelfCaptureAdmissionEffectSink,
    SelfCaptureAdmissionEffectType,
    SelfCaptureAdmissionStateProvider,
    SelfCaptureGpuActivationValidator,
)
from puripuly_heart.core.self_capture import (
    SelfCaptureAdmission,
    SelfCaptureAdmissionStatus,
    SelfCaptureSessionConfig,
)

_GPU_PENDING_STATUSES = frozenset(
    {
        "not_installed",
        "invalid",
        "install_failed",
        "installing",
    }
)
_LOCAL_REPAIR_STATUSES = frozenset({"missing", "invalid", "download_failed"})


@dataclass(frozen=True, slots=True)
class SelfCaptureAdmissionAdapter:
    state_provider: SelfCaptureAdmissionStateProvider
    validate_gpu_activation: SelfCaptureGpuActivationValidator
    effect_sink: SelfCaptureAdmissionEffectSink

    async def admit(self, config: SelfCaptureSessionConfig) -> SelfCaptureAdmission:
        state = self.state_provider(config)
        if not state.settings_available:
            return SelfCaptureAdmission(
                SelfCaptureAdmissionStatus.REJECTED,
                reason="runtime_unavailable",
            )

        if config.local_gpu:
            if await self.validate_gpu_activation():
                return SelfCaptureAdmission(SelfCaptureAdmissionStatus.ADMITTED)
            state = self.state_provider(config)
            if state.gpu_status in _GPU_PENDING_STATUSES:
                self.effect_sink(
                    SelfCaptureAdmissionEffect(
                        SelfCaptureAdmissionEffectType.RETAIN_GPU_PENDING_INTENT,
                        status=state.gpu_status,
                    )
                )
                return SelfCaptureAdmission(
                    SelfCaptureAdmissionStatus.PENDING,
                    reason=state.gpu_status,
                    retain_intent=True,
                )
            return SelfCaptureAdmission(
                SelfCaptureAdmissionStatus.REJECTED,
                reason=state.gpu_status or "gpu_unavailable",
            )

        if not config.local_cpu:
            return self._runtime_admission(state.runtime_available)

        if not state.local_cpu_supported:
            self.effect_sink(
                SelfCaptureAdmissionEffect(
                    SelfCaptureAdmissionEffectType.REJECT_UNSUPPORTED_LANGUAGE,
                )
            )
            return SelfCaptureAdmission(
                SelfCaptureAdmissionStatus.REJECTED,
                reason="language_unsupported",
            )

        if state.local_runtime_status == "downloading":
            self.effect_sink(
                SelfCaptureAdmissionEffect(
                    SelfCaptureAdmissionEffectType.RETAIN_DOWNLOAD_PENDING_INTENT,
                    status=state.local_runtime_status,
                    activation_generation=state.activation_generation,
                )
            )
            return SelfCaptureAdmission(
                SelfCaptureAdmissionStatus.PENDING,
                reason=state.local_runtime_status,
                retain_intent=True,
            )

        if state.local_runtime_status in _LOCAL_REPAIR_STATUSES:
            self.effect_sink(
                SelfCaptureAdmissionEffect(
                    SelfCaptureAdmissionEffectType.REQUEST_LOCAL_REPAIR,
                    status=state.local_runtime_status,
                    activation_generation=state.activation_generation,
                )
            )
            return SelfCaptureAdmission(
                SelfCaptureAdmissionStatus.PENDING,
                reason=state.local_runtime_status,
                retain_intent=True,
            )

        return self._runtime_admission(state.runtime_available)

    @staticmethod
    def _runtime_admission(runtime_available: bool) -> SelfCaptureAdmission:
        if runtime_available:
            return SelfCaptureAdmission(SelfCaptureAdmissionStatus.ADMITTED)
        return SelfCaptureAdmission(
            SelfCaptureAdmissionStatus.REJECTED,
            reason="runtime_unavailable",
        )


__all__ = ["SelfCaptureAdmissionAdapter"]
