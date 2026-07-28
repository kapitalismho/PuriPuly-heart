from __future__ import annotations

import logging

from puripuly_heart.app.services.local_asr_diagnostics import (
    LocalASRDiagnosticsGpuEffect,
    LocalASRDiagnosticsOwner,
)
from puripuly_heart.core.local_asr_provider_runtime import ProviderRuntimeDiagnostic


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
        detailed_log_sink=lambda message: detailed.append(message),
        gpu_effect_sink=effects.append,
        gpu_discovery_origin_provider=lambda: "settings",
        gpu_provider_id="local_qwen_gpu",
    )
    return owner, basic, detailed, effects


def test_local_asr_load_result_is_basic_and_bounded() -> None:
    owner, basic, _detailed, _effects = _owner_with_logs()

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

    assert basic == [
        (
            "[LocalASR][Load] channel=self model=parakeet-v3 backend=CPU "
            "outcome=ready load_seconds=2.418",
            logging.INFO,
        ),
        (
            "[LocalASR][Load] channel=peer model=qwen backend=CPU "
            "outcome=failed load_seconds=0.731 failure_type=LocalQwenSherpaLoadError",
            logging.ERROR,
        ),
    ]


def test_cpu_transition_promotes_only_terminal_load_results_to_basic() -> None:
    owner, basic, detailed, _effects = _owner_with_logs()

    owner.transition_diagnostic(
        {
            "channel": "peer",
            "actual_provider": "local_parakeet_v3",
            "model_id": "parakeet-v3",
            "load_ms": 1250,
            "outcome": "applied",
        }
    )
    owner.transition_diagnostic(
        {
            "channel": "self",
            "actual_provider": "local_qwen",
            "model_id": "qwen",
            "load_ms": 300,
            "outcome": "failed",
            "failure_type": "LocalQwenSherpaLoadError",
        }
    )
    owner.transition_diagnostic(
        {
            "channel": "self",
            "actual_provider": "local_qwen",
            "model_id": "qwen",
            "load_ms": 100,
            "outcome": "superseded",
        }
    )

    assert len(detailed) == 3
    assert basic == [
        (
            "[LocalASR][Load] channel=peer model=parakeet-v3 backend=CPU "
            "outcome=ready load_seconds=1.250",
            logging.INFO,
        ),
        (
            "[LocalASR][Load] channel=self model=qwen backend=CPU "
            "outcome=failed load_seconds=0.300 failure_type=LocalQwenSherpaLoadError",
            logging.ERROR,
        ),
    ]


def test_gpu_ready_and_worker_failure_are_basic_terminal_logs() -> None:
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

    assert len(detailed) == 2
    assert basic == [
        (
            "[LocalASR][Load] channel=self model=qwen-gpu backend=Vulkan "
            "device=vulkan-index-0 outcome=ready load_seconds=4.120 warmup_seconds=0.382",
            logging.INFO,
        ),
        (
            "[LocalASR][Worker] backend=Vulkan outcome=failed "
            "failure_code=heartbeat_timeout exit_code=1",
            logging.ERROR,
        ),
    ]
    assert effects == [
        LocalASRDiagnosticsGpuEffect(state="ready", origin="activation"),
        LocalASRDiagnosticsGpuEffect(
            state="activation_failed",
            origin="worker",
            publish_notice=True,
        ),
    ]


def test_gpu_decode_attempt_logs_rtf_in_basic_mode() -> None:
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

    assert len(detailed) == 1
    assert basic == [
        (
            "[LocalASR][Attempt] channel=self model=qwen-gpu backend=Vulkan "
            "audio_seconds=2.000 decode_seconds=0.250 rtf=0.125000 "
            "result=success queue_wait_seconds=0.031",
            logging.INFO,
        )
    ]


def test_gpu_worker_recovery_logs_restart_without_utterance_retry() -> None:
    owner, basic, detailed, _effects = _owner_with_logs()

    owner.provider_runtime_diagnostic(
        ProviderRuntimeDiagnostic(
            event="worker_recovery_started",
            failure_code="decode_failure",
        )
    )
    owner.provider_runtime_diagnostic(ProviderRuntimeDiagnostic(event="worker_recovery_ready"))

    assert len(detailed) == 2
    assert basic == [
        (
            "[LocalASR][Worker] backend=Vulkan outcome=restarting "
            "failure_code=decode_failure utterance_retry=false",
            logging.WARNING,
        ),
        (
            "[LocalASR][Worker] backend=Vulkan outcome=recovered utterance_retry=false",
            logging.INFO,
        ),
    ]
