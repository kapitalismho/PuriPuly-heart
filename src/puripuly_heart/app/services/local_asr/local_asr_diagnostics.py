from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field

from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.config.runtime_resolution import (
    DEEPGRAM_STT_MODEL_NOVA_3,
    ELEVENLABS_SCRIBE_STT_MODEL,
    GEMINI_TRANSCRIBE_STT_MODEL,
    QWEN_AUDIO_STT_MODEL,
    SONIOX_STT_MODEL_RT_V5,
)
from puripuly_heart.core.diagnostic_validation import (
    stt_startup_failure_code,
    stt_startup_failure_stage,
    stt_startup_failure_type,
)
from puripuly_heart.core.local_asr.local_asr_provider_runtime import ProviderRuntimeDiagnostic
from puripuly_heart.core.local_asr.local_stt_assets import (
    LOCAL_QWEN_GPU_MODEL_ID,
    REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
)

LocalASRBasicLogSink = Callable[[str, int], None]
LocalASRDiagnosticLogSink = Callable[[str], object]
LocalASRGpuDiscoveryOriginProvider = Callable[[], str]


_SAFE_TOKEN_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.:+-,"
)
_PROVIDER_IDENTITIES = frozenset(provider.value for provider in STTProviderName)
_MODEL_IDENTITIES = frozenset(
    (
        *REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
        LOCAL_QWEN_GPU_MODEL_ID,
        DEEPGRAM_STT_MODEL_NOVA_3,
        ELEVENLABS_SCRIBE_STT_MODEL,
        GEMINI_TRANSCRIBE_STT_MODEL,
        QWEN_AUDIO_STT_MODEL,
        SONIOX_STT_MODEL_RT_V5,
    )
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


def _known_identity(value: str | None, identities: frozenset[str]) -> str:
    if value is None:
        return "none"
    return value if value in identities else "unknown"


def _finite_nonnegative_seconds(value: object) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except OverflowError, ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _channel_label(channel: object) -> str:
    token = _safe_token(channel, "").casefold()
    return {"self": "Self", "peer": "Peer"}.get(token, "Local")


def _model_label(model_id: object) -> str:
    token = _safe_token(model_id, "")
    normalized = token.casefold()
    if "orukeet" in normalized:
        return "Orukeet"
    if "fun-asr-mlt" in normalized:
        return "Fun-ASR-MLT-Nano"
    if "fun-asr-nano" in normalized:
        return "Fun-ASR-Nano"
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
    _applied_selections: dict[str, tuple[str | None, str | None, str | None]] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def owner_name(self) -> str:
        return "LocalASRDiagnosticsOwner"

    def selection_applied(
        self,
        *,
        channel: str,
        provider_id: str,
        model_id: str | None,
        source_language: str,
    ) -> None:
        self.provider_runtime_diagnostic(
            ProviderRuntimeDiagnostic(
                event="language_selection",
                outcome="applied",
                channel=channel,
                provider_id=provider_id,
                model_id=model_id,
                source_language=source_language,
            )
        )

    def provider_runtime_diagnostic(
        self,
        diagnostic: ProviderRuntimeDiagnostic,
    ) -> None:
        if (
            diagnostic.event in {"provider_build", "provider_warmup"}
            and diagnostic.outcome == "failed"
        ):
            channel = (
                diagnostic.channel if diagnostic.channel in {"self", "peer"} else "unclassified"
            )
            self.diagnostic_log_sink(
                "[LocalASR] failed "
                f"channel={channel} "
                f"failure_stage={stt_startup_failure_stage(diagnostic.event)} "
                f"failure_code={stt_startup_failure_code(diagnostic.failure_code)} "
                f"failure_type={stt_startup_failure_type(diagnostic.failure_type)}"
            )
        if diagnostic.event in {"provider_handoff", "provider_replace", "language_selection"}:
            previous_provider_id = diagnostic.previous_provider_id
            previous_model_id = diagnostic.previous_model_id
            previous_language = diagnostic.previous_source_language
            if diagnostic.outcome == "applied" and diagnostic.channel in {"self", "peer"}:
                selection = (
                    diagnostic.provider_id,
                    diagnostic.model_id,
                    diagnostic.source_language,
                )
                previous_selection = self._applied_selections.get(diagnostic.channel)
                if previous_selection == selection:
                    return
                if previous_selection is not None:
                    previous_provider_id, previous_model_id, previous_language = previous_selection
                self._applied_selections[diagnostic.channel] = selection
            previous_provider = _known_identity(previous_provider_id, _PROVIDER_IDENTITIES)
            provider = _known_identity(diagnostic.provider_id, _PROVIDER_IDENTITIES)
            previous_model = _known_identity(previous_model_id, _MODEL_IDENTITIES)
            model = _known_identity(diagnostic.model_id, _MODEL_IDENTITIES)
            self.basic_log_sink(
                "[ASR] provider_transition "
                f"event={diagnostic.event} "
                f"channel={_safe_token(diagnostic.channel, 'unknown')} "
                f"previous_provider={previous_provider} provider={provider} "
                f"previous_model={previous_model} model={model} "
                f"previous_language={_safe_token(previous_language, 'none')} "
                f"language={_safe_token(diagnostic.source_language, 'none')} "
                f"outcome={_safe_token(diagnostic.outcome, 'unknown')} "
                f"cause={stt_startup_failure_code(diagnostic.failure_code)} "
                f"stage={'selection' if diagnostic.event == 'language_selection' else 'provider'}",
                logging.WARNING if diagnostic.outcome == "failed" else logging.INFO,
            )
        if diagnostic.event == "activation_failed":
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
                f"Local ASR worker restarting (cause={cause}; utterance retry=false)",
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
        if outcome != "failed":
            return
        self.basic_log_sink(
            f"{_channel_label(fields.get('channel'))} recognition provider update failed "
            f"({_model_label(fields.get('model_id'))}; "
            f"cause={_failure_cause(fields.get('failure_type'), 'unknown error')})",
            logging.ERROR,
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
        if outcome != "failed":
            return
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

        self.basic_log_sink(
            f"{_channel_label(channel)} recognition model failed "
            f"({_model_label(model_id)}; "
            f"cause={_load_failure_cause(failure_code, failure_type)})",
            logging.ERROR,
        )


__all__ = [
    "LocalASRBasicLogSink",
    "LocalASRDiagnosticLogSink",
    "LocalASRDiagnosticsGpuEffect",
    "LocalASRDiagnosticsGpuEffectSink",
    "LocalASRDiagnosticsOwner",
    "LocalASRGpuDiscoveryOriginProvider",
]
