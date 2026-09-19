from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field

from puripuly_heart.core.local_asr_provider_runtime import ProviderRuntimeDiagnostic

LocalASRBasicLogSink = Callable[[str, int], None]
LocalASRDiagnosticLogSink = Callable[[str], object]
LocalASRGpuDiscoveryOriginProvider = Callable[[], str]


_SAFE_TOKEN_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.:+-,"
)
_FAILURE_CAUSE_LABELS = {
    "activation_failed": "activation failed",
    "channel_cancel_timeout": "channel cancellation timeout",
    "decode_failure": "decode failure",
    "heartbeat_timeout": "heartbeat timeout",
    "saved_device_missing": "saved device missing",
    "unclassified": "unknown error",
    "worker_closed": "worker stopped",
    "worker_failed": "worker failure",
    "worker_shutdown_failed": "worker shutdown failed",
}


def _safe_token(value: object, fallback: str = "") -> str:
    if value is None:
        return fallback
    try:
        token = str(value).strip()
    except Exception:
        return fallback
    if (
        not token
        or len(token) > 64
        or any(character not in _SAFE_TOKEN_CHARS for character in token)
    ):
        return fallback
    return token


def _finite_nonnegative_seconds(value: object) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except OverflowError, ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _seconds_from_milliseconds(value: object) -> float | None:
    milliseconds = _finite_nonnegative_seconds(value)
    if milliseconds is None:
        return None
    return _finite_nonnegative_seconds(milliseconds / 1000.0)


def _channel_label(channel: object) -> str:
    token = _safe_token(channel, "").casefold()
    return {"self": "Self", "peer": "Peer"}.get(token, "Local")


def _model_label(model_id: object) -> str:
    token = _safe_token(model_id, "")
    normalized = token.casefold()
    if "parakeet" in normalized:
        return "Parakeet"
    if "qwen" in normalized:
        return "Qwen"
    return "selected model"


def _failure_cause(value: object, fallback: str) -> str:
    token = _safe_token(value, "").casefold()
    return _FAILURE_CAUSE_LABELS.get(token, fallback)


def _load_failure_cause(failure_code: object, failure_type: object) -> str:
    cause = _failure_cause(failure_code, "")
    return cause or _failure_cause(failure_type, "model load failed")


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
                failure_type=diagnostic.failure_type,
            )
        elif diagnostic.event == "worker_failed":
            cause = _failure_cause(diagnostic.failure_code, "worker failure")
            exit_code = (
                f"; exit code={diagnostic.worker_exit_code}"
                if isinstance(diagnostic.worker_exit_code, int)
                and not isinstance(diagnostic.worker_exit_code, bool)
                else ""
            )
            self.basic_log_sink(
                f"Local ASR worker failed (cause={cause}{exit_code})",
                logging.ERROR,
            )
        elif diagnostic.event == "worker_recovery_started":
            cause = _failure_cause(diagnostic.failure_code, "decode failure")
            self.basic_log_sink(
                "Local ASR worker restarting " f"(cause={cause}; utterance retry=false)",
                logging.WARNING,
            )
        elif diagnostic.event == "worker_recovery_ready":
            self.basic_log_sink(
                "Local ASR worker recovered (utterance retry=false)",
                logging.INFO,
            )
        elif diagnostic.event == "decode_attempt":
            audio_seconds = _finite_nonnegative_seconds(diagnostic.audio_seconds)
            decode_seconds = _finite_nonnegative_seconds(diagnostic.decode_seconds)
            rtf = _finite_nonnegative_seconds(diagnostic.rtf)
            if (
                audio_seconds is not None
                and audio_seconds > 0
                and decode_seconds is not None
                and rtf is not None
            ):
                label = _channel_label(diagnostic.channel)
                parts = [
                    f"[{label} · Recognition]",
                    f"Audio {audio_seconds:.2f} s",
                    f"Decode {decode_seconds:.2f} s",
                    f"RTF {rtf:.3f}",
                    f"Result {_safe_token(diagnostic.outcome, 'unknown')}",
                ]
                queue_wait_seconds = _finite_nonnegative_seconds(diagnostic.queue_wait_seconds)
                if queue_wait_seconds is not None:
                    parts.append(f"Queue {queue_wait_seconds:.2f} s")
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
        actual_provider = str(fields.get("actual_provider") or "")
        if actual_provider == self.gpu_provider_id:
            return
        outcome = _safe_token(fields.get("outcome"), "")
        if outcome not in {"applied", "failed"}:
            return
        self.log_load_result(
            channel=str(fields.get("channel") or "unknown"),
            model_id=str(fields.get("model_id") or "unknown"),
            backend="CPU",
            outcome="ready" if outcome == "applied" else "failed",
            load_seconds=_seconds_from_milliseconds(fields.get("load_ms")),
            failure_type=(
                _safe_token(fields["failure_type"], "") or None
                if fields.get("failure_type") is not None
                else None
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
        channel_token = _safe_token(channel, "unknown")
        model_token = _safe_token(model_id, "unknown")
        backend_token = _safe_token(backend, "unknown")
        outcome_token = _safe_token(outcome, "unknown")
        fields = [
            f"channel={channel_token}",
            f"model={model_token}",
            f"backend={backend_token}",
        ]
        device_token = _safe_token(device, "")
        if device_token:
            fields.append(f"device={device_token}")
        fields.append(f"outcome={outcome_token}")
        measured_load_seconds = _finite_nonnegative_seconds(load_seconds)
        if measured_load_seconds is not None:
            fields.append(f"load_seconds={measured_load_seconds:.3f}")
        measured_warmup_seconds = _finite_nonnegative_seconds(warmup_seconds)
        if measured_warmup_seconds is not None:
            fields.append(f"warmup_seconds={measured_warmup_seconds:.3f}")
        failure_type_token = _safe_token(failure_type, "")
        if failure_type_token:
            fields.append(f"failure_type={failure_type_token}")
        failure_code_token = _safe_token(failure_code, "")
        if failure_code_token:
            fields.append(f"failure_code={failure_code_token}")
        self.diagnostic_log_sink(f"[LocalASR][Load] {' '.join(fields)}")

        if outcome_token == "ready":
            message = (
                f"{_channel_label(channel)} recognition model ready " f"({_model_label(model_id)})"
            )
            level = logging.INFO
        elif outcome_token == "failed":
            message = (
                f"{_channel_label(channel)} recognition model failed "
                f"({_model_label(model_id)}; "
                f"cause={_load_failure_cause(failure_code, failure_type)})"
            )
            level = logging.ERROR
        else:
            message = (
                f"{_channel_label(channel)} recognition model update " f"({_model_label(model_id)})"
            )
            level = logging.INFO
        self.basic_log_sink(message, level)


__all__ = [
    "LocalASRBasicLogSink",
    "LocalASRDiagnosticLogSink",
    "LocalASRDiagnosticsGpuEffect",
    "LocalASRDiagnosticsGpuEffectSink",
    "LocalASRDiagnosticsOwner",
    "LocalASRGpuDiscoveryOriginProvider",
]
