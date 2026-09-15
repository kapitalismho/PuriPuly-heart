from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field

from puripuly_heart.core.local_asr_provider_runtime import ProviderRuntimeDiagnostic

LocalASRBasicLogSink = Callable[[str, int], None]
LocalASRDiagnosticLogSink = Callable[[str], object]
LocalASRGpuDiscoveryOriginProvider = Callable[[], str]


@dataclass(frozen=True, slots=True)
class LocalASRDiagnosticsGpuEffect:
    state: str
    origin: str
    publish_notice: bool = False


LocalASRDiagnosticsGpuEffectSink = Callable[[LocalASRDiagnosticsGpuEffect], None]


@dataclass(slots=True)
class LocalASRDiagnosticsOwner:
    basic_log_sink: LocalASRBasicLogSink = field(repr=False)
    diagnostic_log_sink: LocalASRDiagnosticLogSink = field(repr=False)
    gpu_effect_sink: LocalASRDiagnosticsGpuEffectSink = field(repr=False)
    gpu_discovery_origin_provider: LocalASRGpuDiscoveryOriginProvider = field(repr=False)
    gpu_provider_id: str

    @property
    def owner_name(self) -> str:
        return "LocalASRDiagnosticsOwner"

    def provider_runtime_diagnostic(
        self,
        diagnostic: ProviderRuntimeDiagnostic,
    ) -> None:
        fields = [f"event={diagnostic.event}"]
        for name in (
            "channel",
            "provider_id",
            "model_id",
            "device_id",
            "phase",
            "outcome",
            "failure_code",
            "failure_type",
        ):
            value = getattr(diagnostic, name)
            if value is not None:
                fields.append(f"{name}={value}")
        self.diagnostic_log_sink(f"[LocalASR][ProviderRuntime] {' '.join(fields)}")
        if diagnostic.event == "activation_ready":
            self.log_load_result(
                channel=diagnostic.channel or "unknown",
                model_id=diagnostic.model_id or "unknown",
                backend="Vulkan",
                device=diagnostic.device_id or "unknown",
                outcome="ready",
                load_seconds=diagnostic.model_load_seconds,
                warmup_seconds=diagnostic.warmup_seconds,
            )
        elif diagnostic.event == "activation_failed":
            self.log_load_result(
                channel=diagnostic.channel or "unknown",
                model_id=diagnostic.model_id or "unknown",
                backend="Vulkan",
                outcome="failed",
                load_seconds=diagnostic.model_load_seconds,
                failure_code=diagnostic.failure_code or "activation_failed",
            )
        elif diagnostic.event == "worker_failed":
            exit_code = (
                f" exit_code={diagnostic.worker_exit_code}"
                if diagnostic.worker_exit_code is not None
                else ""
            )
            self.basic_log_sink(
                "[LocalASR][Worker] backend=Vulkan outcome=failed "
                f"failure_code={diagnostic.failure_code or 'worker_failed'}{exit_code}",
                logging.ERROR,
            )
        elif diagnostic.event == "worker_recovery_started":
            self.basic_log_sink(
                "[LocalASR][Worker] backend=Vulkan outcome=restarting "
                f"failure_code={diagnostic.failure_code or 'decode_failure'} "
                "utterance_retry=false",
                logging.WARNING,
            )
        elif diagnostic.event == "worker_recovery_ready":
            self.basic_log_sink(
                "[LocalASR][Worker] backend=Vulkan outcome=recovered utterance_retry=false",
                logging.INFO,
            )
        elif (
            diagnostic.event == "decode_attempt"
            and diagnostic.audio_seconds is not None
            and diagnostic.audio_seconds > 0
            and diagnostic.decode_seconds is not None
            and diagnostic.decode_seconds >= 0
            and diagnostic.rtf is not None
            and math.isfinite(diagnostic.rtf)
        ):
            label = "Self" if diagnostic.channel == "self" else "Peer"
            parts = [
                f"[{label} · Recognition]",
                f"Audio {diagnostic.audio_seconds:.2f} s",
                f"Decode {diagnostic.decode_seconds:.2f} s",
                f"RTF {diagnostic.rtf:.3f}",
                f"Result {diagnostic.outcome or 'unknown'}",
            ]
            if diagnostic.queue_wait_seconds is not None and diagnostic.queue_wait_seconds >= 0:
                parts.append(f"Queue {diagnostic.queue_wait_seconds:.2f} s")
            self.basic_log_sink(" · ".join(parts), logging.INFO)
        if diagnostic.event == "worker_lifecycle" and diagnostic.phase in {
            "validating",
            "loading",
            "warming",
            "ready",
        }:
            self.gpu_effect_sink(
                LocalASRDiagnosticsGpuEffect(
                    state=diagnostic.phase,
                    origin="worker_lifecycle",
                )
            )
        elif diagnostic.event == "activation_ready":
            self.gpu_effect_sink(
                LocalASRDiagnosticsGpuEffect(
                    state="ready",
                    origin="activation",
                )
            )
        elif diagnostic.event == "discovery_pending":
            self.gpu_effect_sink(
                LocalASRDiagnosticsGpuEffect(
                    state="discovery_pending",
                    origin=self.gpu_discovery_origin_provider(),
                )
            )
        elif diagnostic.event in {"activation_failed", "worker_failed"}:
            self.gpu_effect_sink(
                LocalASRDiagnosticsGpuEffect(
                    state="activation_failed",
                    origin="worker",
                    publish_notice=True,
                )
            )

    def transition_diagnostic(self, fields: dict[str, object]) -> None:
        ordered = " ".join(f"{key}={value}" for key, value in fields.items())
        self.diagnostic_log_sink(f"[LocalASR][Transition] {ordered}")
        actual_provider = str(fields.get("actual_provider") or "")
        if actual_provider == self.gpu_provider_id:
            return
        outcome = str(fields.get("outcome") or "")
        if outcome not in {"applied", "failed"}:
            return
        load_ms = fields.get("load_ms")
        load_seconds = (
            max(0.0, float(load_ms) / 1000.0)
            if isinstance(load_ms, int | float) and not isinstance(load_ms, bool)
            else None
        )
        self.log_load_result(
            channel=str(fields.get("channel") or "unknown"),
            model_id=str(fields.get("model_id") or "unknown"),
            backend="CPU",
            outcome="ready" if outcome == "applied" else "failed",
            load_seconds=load_seconds,
            failure_type=(
                str(fields["failure_type"]) if fields.get("failure_type") is not None else None
            ),
        )

    def log_load_result(
        self,
        *,
        channel: str,
        model_id: str,
        backend: str,
        outcome: str,
        load_seconds: float | None,
        failure_type: str | None = None,
        device: str | None = None,
        warmup_seconds: float | None = None,
        failure_code: str | None = None,
    ) -> None:
        fields = [
            f"channel={channel}",
            f"model={model_id}",
            f"backend={backend}",
        ]
        if device is not None:
            fields.append(f"device={device}")
        fields.append(f"outcome={outcome}")
        if load_seconds is not None:
            fields.append(f"load_seconds={max(0.0, load_seconds):.3f}")
        if warmup_seconds is not None:
            fields.append(f"warmup_seconds={max(0.0, warmup_seconds):.3f}")
        if failure_type is not None:
            fields.append(f"failure_type={failure_type}")
        if failure_code is not None:
            fields.append(f"failure_code={failure_code}")
        self.basic_log_sink(
            f"[LocalASR][Load] {' '.join(fields)}",
            logging.ERROR if outcome == "failed" else logging.INFO,
        )


__all__ = [
    "LocalASRBasicLogSink",
    "LocalASRDiagnosticLogSink",
    "LocalASRDiagnosticsGpuEffect",
    "LocalASRDiagnosticsGpuEffectSink",
    "LocalASRDiagnosticsOwner",
    "LocalASRGpuDiscoveryOriginProvider",
]
