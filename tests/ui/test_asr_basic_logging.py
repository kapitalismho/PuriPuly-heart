from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from puripuly_heart.config.settings import AppSettings, STTProviderName
from puripuly_heart.core.local_stt_assets import (
    LocalQwenSherpaLoadError,
    LocalSTTInstallState,
)
from puripuly_heart.core.runtime.gpu_asr import GpuASRDiagnostic
from puripuly_heart.ui.controller import GuiController


def _controller_with_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[GuiController, list[tuple[str, int]], list[str]]:
    basic: list[tuple[str, int]] = []
    detailed: list[str] = []
    monkeypatch.setattr(
        GuiController,
        "log_basic",
        lambda _self, message, *, level=logging.INFO: basic.append((message, level)),
    )
    monkeypatch.setattr(
        GuiController,
        "log_detailed",
        lambda _self, message, **_kwargs: detailed.append(message) or True,
    )
    controller = GuiController(
        page=SimpleNamespace(),
        app=SimpleNamespace(),
        config_path=Path("settings.json"),
    )
    return controller, basic, detailed


def test_local_asr_load_result_is_basic_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, basic, _detailed = _controller_with_logs(monkeypatch)

    controller._log_local_asr_load_result(
        channel="self",
        model_id="parakeet-v3",
        backend="CPU",
        outcome="ready",
        load_seconds=2.4184,
    )
    controller._log_local_asr_load_result(
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


def test_cpu_transition_promotes_only_terminal_load_results_to_basic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, basic, detailed = _controller_with_logs(monkeypatch)

    controller._local_asr_transition_diagnostic(
        {
            "channel": "peer",
            "actual_provider": "local_parakeet_v3",
            "model_id": "parakeet-v3",
            "load_ms": 1250,
            "outcome": "applied",
        }
    )
    controller._local_asr_transition_diagnostic(
        {
            "channel": "self",
            "actual_provider": "local_qwen",
            "model_id": "qwen",
            "load_ms": 300,
            "outcome": "failed",
            "failure_type": "LocalQwenSherpaLoadError",
        }
    )
    controller._local_asr_transition_diagnostic(
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


@pytest.mark.asyncio
async def test_self_cpu_first_load_logs_resolved_model_and_elapsed_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, basic, _detailed = _controller_with_logs(monkeypatch)
    settings = AppSettings()
    settings.provider.stt = STTProviderName.LOCAL_QWEN
    controller.settings = settings
    controller._local_stt_install_state = LocalSTTInstallState(status="ready")
    controller.hub = SimpleNamespace(
        stt=SimpleNamespace(
            backend=SimpleNamespace(is_loaded=False, model_id="qwen-model"),
        ),
        peer_stt=None,
    )

    async def probe(_self, *, activation_generation=None) -> None:
        _ = activation_generation

    monkeypatch.setattr(GuiController, "_probe_self_local_stt_runtime_load", probe)
    monkeypatch.setattr(
        GuiController,
        "_record_strict_local_stt_ready",
        lambda _self, _model_ids: None,
    )
    monkeypatch.setattr(GuiController, "_sync_local_stt_notice", lambda _self: None)
    assert await controller._ensure_local_stt_ready() is True
    message, level = basic[-1]
    assert message.startswith(
        "[LocalASR][Load] channel=self model=qwen-model backend=CPU " "outcome=ready load_seconds="
    )
    assert level == logging.INFO


@pytest.mark.asyncio
async def test_self_cpu_load_failure_is_basic_error_without_private_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, basic, _detailed = _controller_with_logs(monkeypatch)
    settings = AppSettings()
    settings.provider.stt = STTProviderName.LOCAL_QWEN
    controller.settings = settings
    controller._local_stt_install_state = LocalSTTInstallState(status="ready")
    controller.hub = SimpleNamespace(
        stt=SimpleNamespace(
            backend=SimpleNamespace(is_loaded=False, model_id="qwen-model"),
        ),
        peer_stt=None,
    )

    async def fail_probe(_self, *, activation_generation=None) -> None:
        _ = activation_generation
        raise LocalQwenSherpaLoadError("C:/Users/private/model")

    monkeypatch.setattr(GuiController, "_probe_self_local_stt_runtime_load", fail_probe)
    monkeypatch.setattr(
        GuiController,
        "_handle_local_stt_unavailable",
        lambda *_args, **_kwargs: False,
    )
    assert await controller._ensure_local_stt_ready() is False
    message, level = basic[-1]
    assert message.startswith(
        "[LocalASR][Load] channel=self model=qwen3-asr-0.6b-int8-sherpa "
        "backend=CPU outcome=failed load_seconds="
    )
    assert message.endswith("failure_type=LocalQwenSherpaLoadError")
    assert level == logging.ERROR
    assert "C:/Users/private/model" not in message


@pytest.mark.asyncio
async def test_gpu_ready_and_worker_failure_are_basic_terminal_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, basic, detailed = _controller_with_logs(monkeypatch)
    controller._gpu_asr_runtime = SimpleNamespace(active_channels=frozenset({"self", "peer"}))
    monkeypatch.setattr(GuiController, "_set_gpu_ui_state", lambda *_args, **_kwargs: None)

    await controller._on_gpu_asr_diagnostic(
        GpuASRDiagnostic(
            kind="activation_ready",
            fields={
                "model": "qwen-gpu",
                "device": "vulkan-index-0",
                "model_load_seconds": 4.12,
                "warmup_seconds": 0.382,
            },
        )
    )
    await controller._on_gpu_asr_diagnostic(
        GpuASRDiagnostic(
            kind="worker_failed",
            fields={"failure": "heartbeat_timeout", "exit_code": 1},
        )
    )

    assert len(detailed) == 2
    assert basic == [
        (
            "[LocalASR][Load] channel=peer,self model=qwen-gpu backend=Vulkan "
            "device=vulkan-index-0 outcome=ready load_seconds=4.120 warmup_seconds=0.382",
            logging.INFO,
        ),
        (
            "[LocalASR][Worker] backend=Vulkan outcome=failed "
            "failure_code=heartbeat_timeout exit_code=1",
            logging.ERROR,
        ),
    ]


@pytest.mark.asyncio
async def test_gpu_decode_attempt_logs_rtf_in_basic_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, basic, detailed = _controller_with_logs(monkeypatch)

    await controller._on_gpu_asr_diagnostic(
        GpuASRDiagnostic(
            kind="decode_attempt",
            fields={
                "channel": "self",
                "model": "qwen-gpu",
                "backend": "Vulkan",
                "audio_seconds": 2.0,
                "decode_seconds": 0.25,
                "rtf": 0.125,
                "result": "success",
                "queue_wait_seconds": 0.031,
            },
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


@pytest.mark.asyncio
async def test_gpu_worker_recovery_logs_restart_without_utterance_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, basic, detailed = _controller_with_logs(monkeypatch)

    await controller._on_gpu_asr_diagnostic(
        GpuASRDiagnostic(
            kind="worker_recovery_started",
            fields={"failure": "decode_failure", "utterance_retry": False},
        )
    )
    await controller._on_gpu_asr_diagnostic(
        GpuASRDiagnostic(
            kind="worker_recovery_ready",
            fields={"utterance_retry": False},
        )
    )

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
