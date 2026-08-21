from __future__ import annotations

from pathlib import Path

import pytest

from puripuly_heart.app.wiring.wiring_translation_backend import create_translation_backend
from puripuly_heart.core.http_extensions import HttpExtensionRegistry
from puripuly_heart.core.storage.secrets import InMemorySecretStore
from puripuly_heart.release_evidence import managed_gemma_production as evidence


def test_performance_log_parser_preserves_operational_metrics_without_text() -> None:
    source_text = "private source subtitle"
    translated_text = "private translated subtitle"
    system_prompt = "private system prompt"
    parsed = evidence._parse_performance_log(
        "[ManagedGemma][Performance] backend=vulkan language_pair=en->ko "
        "prompt_tokens=31 cached_prompt_tokens=28 completion_tokens=7 "
        "prompt_ms=4.500 generation_ms=70.000 generation_tps=100.000"
    )

    evidence._validate_metrics(parsed, effective_backend="vulkan")

    serialized = str(parsed)
    assert parsed == {
        "backend": "vulkan",
        "language_pair": "en->ko",
        "prompt_tokens": 31,
        "cached_prompt_tokens": 28,
        "completion_tokens": 7,
        "prompt_ms": 4.5,
        "generation_ms": 70.0,
        "generation_tps": 100.0,
    }
    assert source_text not in serialized
    assert translated_text not in serialized
    assert system_prompt not in serialized


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("prompt_tokens", 0, "prompt token"),
        ("cached_prompt_tokens", 0, "cached prompt token"),
        ("completion_tokens", 0, "completion token"),
        ("generation_ms", 0.0, "generation timing"),
        ("generation_tps", 0.0, "generation speed"),
        ("prompt_ms", 0.0, "prompt timing"),
    ],
)
def test_metric_validation_rejects_incomplete_observable_evidence(
    field: str, value: int | float, message: str
) -> None:
    metrics: dict[str, object] = {
        "backend": "cpu",
        "prompt_tokens": 10,
        "cached_prompt_tokens": 8,
        "completion_tokens": 5,
        "prompt_ms": 1.0,
        "generation_ms": 2.0,
        "generation_tps": 3.0,
    }
    metrics[field] = value

    with pytest.raises(RuntimeError, match=message):
        evidence._validate_metrics(metrics, effective_backend="cpu")


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("prompt_ms", "prompt timing"),
        ("generation_ms", "generation timing"),
        ("generation_tps", "generation speed"),
    ],
)
def test_metric_validation_rejects_non_finite_values(field: str, message: str) -> None:
    metrics: dict[str, object] = {
        "backend": "vulkan",
        "prompt_tokens": 10,
        "cached_prompt_tokens": 8,
        "completion_tokens": 5,
        "prompt_ms": 1.0,
        "generation_ms": 2.0,
        "generation_tps": 3.0,
    }
    metrics[field] = float("nan")

    with pytest.raises(RuntimeError, match=message):
        evidence._validate_metrics(metrics, effective_backend="vulkan")


def test_vulkan_observation_requires_full_device_offload() -> None:
    class Capture:
        returncode = 1
        command = (
            "llama-server.exe",
            "--device",
            "Vulkan0",
            "--n-gpu-layers",
            "99",
        )
        output = (
            "llama_prepare_model_devices: using device Vulkan0 "
            "(AMD Radeon RX 7900 XTX) - 23737 MiB free\n"
            "load_tensors: offloaded 43/43 layers to GPU\n"
        )

    assert evidence._runtime_observation(Capture(), backend="gpu") == {
        "device": "Vulkan0",
        "device_name": "AMD Radeon RX 7900 XTX",
        "n_gpu_layers": 99,
        "offloaded_layers": 43,
        "total_layers": 43,
        "mtp_enabled": False,
        "process_exited": True,
    }


@pytest.mark.asyncio
async def test_cleanup_retries_owner_when_backend_close_fails() -> None:
    class Backend:
        async def close(self) -> None:
            raise RuntimeError("backend close failed")

    class Owner:
        closed = False

        async def close(self) -> None:
            self.closed = True

    owner = Owner()
    with pytest.raises(RuntimeError, match="backend close failed"):
        await evidence._close_backend_and_owner(Backend(), owner)
    assert owner.closed


def test_managed_gemma_evidence_settings_disable_provider_fallback(tmp_path: Path) -> None:
    for backend in ("cpu", "gpu"):
        settings = evidence._settings(backend)
        translation_backend = create_translation_backend(
            settings,
            secrets=InMemorySecretStore(),
            http_extensions=HttpExtensionRegistry(tmp_path),
            managed_gemma_runtime=object(),
        )

        assert settings.translation.model.value == "managed_gemma"
        assert settings.translation.connection.value == backend
        assert settings.translation.fallback.enabled is True
        assert evidence._is_managed_only_backend(translation_backend)
