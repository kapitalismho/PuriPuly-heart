from __future__ import annotations

import logging
import math

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
        diagnostic_log_sink=lambda message: detailed.append(message),
        gpu_effect_sink=effects.append,
        gpu_discovery_origin_provider=lambda: "settings",
        gpu_provider_id="local_qwen_gpu",
    )
    return owner, basic, detailed, effects


def test_local_asr_load_result_separates_sparse_file_evidence_from_basic_summary() -> None:
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

    assert len(basic) == 2
    assert basic[0][1] == logging.INFO
    assert "Self recognition model ready" in basic[0][0]
    assert "parakeet-v3" not in basic[0][0]
    assert "model=" not in basic[0][0]
    assert "backend=" not in basic[0][0]
    assert basic[1][1] == logging.ERROR
    assert "Peer recognition model failed" in basic[1][0]
    assert "LocalQwenSherpaLoadError" not in basic[1][0]
    assert "cause=" in basic[1][0]

    assert len(detailed) == 2
    assert detailed[0].startswith("[LocalASR][Load] ")
    assert "channel=self" in detailed[0]
    assert "model=parakeet-v3" in detailed[0]
    assert "backend=CPU" in detailed[0]
    assert "outcome=ready" in detailed[0]
    assert "load_seconds=2.418" in detailed[0]
    assert "failure_type=" not in detailed[0]
    assert "channel=peer" in detailed[1]
    assert "model=qwen" in detailed[1]
    assert "failure_type=LocalQwenSherpaLoadError" in detailed[1]


def test_local_asr_load_result_omits_missing_or_invalid_timings() -> None:
    owner, basic, detailed, _effects = _owner_with_logs()

    for load_seconds, warmup_seconds in (
        (None, None),
        (math.nan, math.inf),
        (-1.0, -2.0),
    ):
        owner.log_load_result(
            channel="self",
            model_id="qwen",
            backend="Vulkan",
            outcome="ready",
            load_seconds=load_seconds,
            warmup_seconds=warmup_seconds,
        )

    assert len(basic) == 3
    assert all(level == logging.INFO for _message, level in basic)
    assert len(detailed) == 3
    assert all("load_seconds=" not in message for message in detailed)
    assert all("warmup_seconds=" not in message for message in detailed)


def test_cpu_transition_logs_only_actual_terminal_results() -> None:
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
    owner.transition_diagnostic(
        {
            "channel": "self",
            "actual_provider": "local_qwen_gpu",
            "model_id": "qwen-gpu",
            "load_ms": 100,
            "outcome": "applied",
        }
    )

    assert len(basic) == 2
    assert [level for _message, level in basic] == [logging.INFO, logging.ERROR]
    assert len(detailed) == 2
    assert "channel=peer" in detailed[0]
    assert "outcome=ready" in detailed[0]
    assert "load_seconds=1.250" in detailed[0]
    assert "channel=self" in detailed[1]
    assert "outcome=failed" in detailed[1]
    assert "load_seconds=0.300" in detailed[1]
    assert all("superseded" not in message for message in detailed)
    assert all("qwen-gpu" not in message for message in detailed)


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

    assert len(detailed) == 1
    assert "model=qwen-gpu" in detailed[0]
    assert "backend=Vulkan" in detailed[0]
    assert "device=vulkan-index-0" in detailed[0]
    assert "load_seconds=4.120" in detailed[0]
    assert len(basic) == 2
    assert basic[0][1] == logging.INFO
    assert "Self recognition model ready" in basic[0][0]
    assert "qwen-gpu" not in basic[0][0]
    assert basic[1][1] == logging.ERROR
    assert "Local ASR worker failed" in basic[1][0]
    assert "cause=heartbeat timeout" in basic[1][0]
    assert "exit code=1" in basic[1][0]
    assert "failure_code=" not in basic[1][0]
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
