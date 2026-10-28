from __future__ import annotations

import logging
import math
from dataclasses import replace

from puripuly_heart.app.services.local_asr.local_asr_diagnostics import (
    LocalASRDiagnosticsGpuEffect,
    LocalASRDiagnosticsOwner,
)
from puripuly_heart.core.local_asr.local_asr_provider_runtime import ProviderRuntimeDiagnostic


def _owner_with_logs() -> tuple[
    LocalASRDiagnosticsOwner,
    list[tuple[str, int]],
    list[str],
    list[LocalASRDiagnosticsGpuEffect],
]:
    basic: list[tuple[str, int]] = []
    detailed: list[str] = []
    effects: list[LocalASRDiagnosticsGpuEffect] = []
    owner = LocalASRDiagnosticsOwner(
        basic_log_sink=lambda message, level: basic.append((message, level)),
        diagnostic_log_sink=lambda message: detailed.append(message),
        gpu_effect_sink=effects.append,
        gpu_discovery_origin_provider=lambda: "settings",
        gpu_provider_id="local_qwen_gpu",
    )
    return owner, basic, detailed, effects


def test_native_load_logs_only_failures_and_preserves_failure_details() -> None:
    owner, basic, detailed, _effects = _owner_with_logs()

    owner.log_load_result(
        channel="self",
        model_id="parakeet-v3",
        backend="CPU",
        outcome="ready",
        load_seconds=2.4184,
    )
    owner.log_load_result(
        channel="peer",
        model_id="qwen",
        backend="CPU",
        outcome="failed",
        load_seconds=0.7314,
        failure_type="LocalQwenSherpaLoadError",
    )

    assert len(basic) == 1
    assert basic[0][1] == logging.ERROR
    assert "LocalQwenSherpaLoadError" not in basic[0][0]

    assert len(detailed) == 1
    assert "channel=peer" in detailed[0]
    assert "model=qwen" in detailed[0]
    assert "outcome=failed" in detailed[0]
    assert "failure_type=LocalQwenSherpaLoadError" in detailed[0]


def test_python_provider_transition_does_not_claim_native_model_readiness() -> None:
    owner, basic, detailed, _effects = _owner_with_logs()

    for outcome in ("applied", "superseded", "closed"):
        owner.transition_diagnostic(
            {
                "channel": "peer",
                "actual_provider": "local_orukeet",
                "model_id": "orukeet-q8-0-transcribe-cpu",
                "load_ms": 0,
                "outcome": outcome,
            }
        )

    assert basic == detailed == []

    owner.transition_diagnostic(
        {
            "channel": "self",
            "actual_provider": "local_fun_asr_mlt_nano",
            "model_id": "fun-asr-mlt-nano-2512-q8-0-transcribe-cpu",
            "load_ms": 0,
            "outcome": "failed",
            "failure_type": "RuntimeError",
        }
    )

    assert len(basic) == 1
    assert basic[0][1] == logging.ERROR
    assert "provider update failed" in basic[0][0]
    assert "model ready" not in basic[0][0]
    assert detailed == []


def test_provider_transition_preserves_channel_and_old_new_identity() -> None:
    owner, basic, detailed, _effects = _owner_with_logs()

    owner.provider_runtime_diagnostic(
        ProviderRuntimeDiagnostic(
            event="provider_replace",
            channel="peer",
            previous_provider_id="local_cpu_auto",
            provider_id="local_cpu_auto",
            previous_model_id="orukeet-q8-0-transcribe-cpu",
            model_id="fun-asr-mlt-nano-2512-q8-0-transcribe-cpu",
            previous_source_language="en",
            source_language="ko",
            outcome="applied",
        )
    )

    assert len(basic) == 1
    message, level = basic[0]
    fields = dict(token.split("=", 1) for token in message.split()[2:])
    assert level == logging.INFO
    assert fields["channel"] == "peer"
    assert fields["previous_provider"] == fields["provider"] == "local_cpu_auto"
    assert fields["previous_model"] == "orukeet-q8-0-transcribe-cpu"
    assert fields["model"] == "fun-asr-mlt-nano-2512-q8-0-transcribe-cpu"
    assert fields["outcome"] == "applied"
    assert fields["stage"] == "provider"
    assert fields["previous_language"] == "en"
    assert fields["language"] == "ko"
    assert detailed == []


def test_language_selection_logs_once_without_utterance_load_or_unchanged_refresh() -> None:
    owner, basic, _detailed, _effects = _owner_with_logs()
    changed = ProviderRuntimeDiagnostic(
        event="provider_replace",
        channel="self",
        previous_provider_id="local_cpu_auto",
        provider_id="local_cpu_auto",
        previous_model_id="orukeet-q8-0-transcribe-cpu",
        model_id="orukeet-q8-0-transcribe-cpu",
        previous_source_language="en",
        source_language="es",
        outcome="applied",
    )
    owner.provider_runtime_diagnostic(changed)
    owner.provider_runtime_diagnostic(replace(changed, previous_source_language="es"))
    owner.provider_runtime_diagnostic(
        replace(
            changed,
            previous_provider_id=None,
            previous_model_id=None,
            previous_source_language=None,
        )
    )
    owner.provider_runtime_diagnostic(
        ProviderRuntimeDiagnostic(
            event="activation_ready",
            channel="self",
            model_id="orukeet-q8-0-transcribe-cpu",
            outcome="ready",
        )
    )
    assert len(basic) == 1
    fields = dict(token.split("=", 1) for token in basic[0][0].split()[2:])
    assert fields["previous_language"] == "en"
    assert fields["language"] == "es"
    assert fields["previous_model"] == fields["model"] == "orukeet-q8-0-transcribe-cpu"
    owner.selection_applied(
        channel="self",
        provider_id="local_cpu_auto",
        model_id="orukeet-q8-0-transcribe-cpu",
        source_language="en",
    )
    owner.selection_applied(
        channel="self",
        provider_id="local_cpu_auto",
        model_id="orukeet-q8-0-transcribe-cpu",
        source_language="en",
    )
    assert len(basic) == 2
    reapplied = dict(token.split("=", 1) for token in basic[1][0].split()[2:])
    assert reapplied["previous_language"] == "es"
    assert reapplied["language"] == "en"
    assert reapplied["stage"] == "selection"


def test_provider_transition_keeps_known_cloud_identity_without_unregistered_values() -> None:
    owner, basic, detailed, _effects = _owner_with_logs()

    owner.provider_runtime_diagnostic(
        ProviderRuntimeDiagnostic(
            event="provider_handoff",
            channel="self",
            previous_provider_id="custom_offline",
            provider_id="deepgram",
            previous_model_id="private-model-id",
            model_id="nova-3",
            outcome="failed",
            failure_code="private-cause",
        )
    )

    message, level = basic[0]
    fields = dict(token.split("=", 1) for token in message.split()[2:])
    assert level == logging.WARNING
    assert fields["previous_provider"] == "custom_offline"
    assert fields["provider"] == "deepgram"
    assert fields["previous_model"] == "unknown"
    assert fields["model"] == "nova-3"
    assert fields["cause"] == "unclassified"
    assert "private-model-id" not in message
    assert "private-cause" not in message
    assert detailed == []


def test_gpu_ready_and_worker_failure_keep_basic_facts_without_internal_fields() -> None:
    owner, basic, detailed, effects = _owner_with_logs()

    owner.provider_runtime_diagnostic(
        ProviderRuntimeDiagnostic(
            event="activation_ready",
            channel="self",
            model_id="qwen-gpu",
            device_id="vulkan-index-0",
            outcome="ready",
            model_load_seconds=4.12,
            warmup_seconds=0.382,
        )
    )
    owner.provider_runtime_diagnostic(
        ProviderRuntimeDiagnostic(
            event="worker_failed",
            outcome="failed",
            failure_code="heartbeat_timeout",
            worker_exit_code=1,
        )
    )

    assert detailed == []
    assert len(basic) == 1
    assert basic[0][1] == logging.ERROR
    assert "cause=heartbeat timeout" in basic[0][0]
    assert "exit code=1" in basic[0][0]
    assert "failure_code=" not in basic[0][0]
    assert effects == [
        LocalASRDiagnosticsGpuEffect(state="ready", origin="activation"),
        LocalASRDiagnosticsGpuEffect(
            state="activation_failed",
            origin="worker",
            publish_notice=True,
        ),
    ]


def test_gpu_decode_attempt_emits_one_truthful_basic_rtf_row_only() -> None:
    owner, basic, detailed, _effects = _owner_with_logs()

    owner.provider_runtime_diagnostic(
        ProviderRuntimeDiagnostic(
            event="decode_attempt",
            channel="self",
            model_id="qwen-gpu",
            audio_seconds=2.0,
            decode_seconds=0.25,
            rtf=0.125,
            outcome="success",
            queue_wait_seconds=0.031,
        )
    )

    assert detailed == []
    assert len(basic) == 1
    message, level = basic[0]
    assert level == logging.INFO
    assert "[Self · Recognition]" in message
    assert "Audio 2.00 s" in message
    assert "Decode 0.25 s" in message
    assert "RTF 0.125" in message
    assert "Result success" in message
    assert "Queue 0.03 s" in message
    assert "qwen-gpu" not in message


def test_gpu_decode_attempt_omits_missing_zero_or_nonfinite_measurements() -> None:
    owner, basic, detailed, _effects = _owner_with_logs()

    invalid_attempts = (
        {"audio_seconds": None, "decode_seconds": 0.25, "rtf": 0.125},
        {"audio_seconds": 0.0, "decode_seconds": 0.25, "rtf": 0.125},
        {"audio_seconds": 2.0, "decode_seconds": None, "rtf": 0.125},
        {"audio_seconds": 2.0, "decode_seconds": -0.25, "rtf": 0.125},
        {"audio_seconds": 2.0, "decode_seconds": 0.25, "rtf": math.nan},
        {"audio_seconds": math.inf, "decode_seconds": 0.25, "rtf": 0.125},
    )
    for fields in invalid_attempts:
        owner.provider_runtime_diagnostic(
            ProviderRuntimeDiagnostic(
                event="decode_attempt",
                channel="peer",
                outcome="success",
                **fields,
            )
        )

    assert basic == []
    assert detailed == []

    owner.provider_runtime_diagnostic(
        ProviderRuntimeDiagnostic(
            event="decode_attempt",
            channel="peer",
            audio_seconds=1.0,
            decode_seconds=0.0,
            rtf=0.0,
            outcome="success",
        )
    )

    assert len(basic) == 1
    assert "Decode 0.00 s" in basic[0][0]
    assert "RTF 0.000" in basic[0][0]
    assert "Queue " not in basic[0][0]


def test_gpu_worker_recovery_is_a_concise_basic_fact() -> None:
    owner, basic, detailed, _effects = _owner_with_logs()

    owner.provider_runtime_diagnostic(
        ProviderRuntimeDiagnostic(
            event="worker_recovery_started",
            failure_code="decode_failure",
        )
    )
    owner.provider_runtime_diagnostic(ProviderRuntimeDiagnostic(event="worker_recovery_ready"))

    assert detailed == []
    assert [level for _message, level in basic] == [
        logging.WARNING,
        logging.INFO,
    ]
    assert "Local ASR worker restarting" in basic[0][0]
    assert "cause=decode failure" in basic[0][0]
    assert "utterance retry=false" in basic[0][0]
    assert "Local ASR worker recovered" in basic[1][0]
    assert "utterance retry=false" in basic[1][0]
    assert all("failure_code=" not in message for message, _level in basic)


def test_gpu_operational_effects_are_independent_of_basic_log_wording() -> None:
    owner, basic, detailed, effects = _owner_with_logs()

    for phase in ("validating", "loading", "warming", "ready"):
        owner.provider_runtime_diagnostic(
            ProviderRuntimeDiagnostic(event="worker_lifecycle", phase=phase)
        )
    owner.provider_runtime_diagnostic(ProviderRuntimeDiagnostic(event="discovery_pending"))

    assert basic == []
    assert detailed == []
    assert effects == [
        LocalASRDiagnosticsGpuEffect(state="validating", origin="worker_lifecycle"),
        LocalASRDiagnosticsGpuEffect(state="loading", origin="worker_lifecycle"),
        LocalASRDiagnosticsGpuEffect(state="warming", origin="worker_lifecycle"),
        LocalASRDiagnosticsGpuEffect(state="ready", origin="worker_lifecycle"),
        LocalASRDiagnosticsGpuEffect(state="discovery_pending", origin="settings"),
    ]
