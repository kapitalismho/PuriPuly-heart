from __future__ import annotations

import importlib
import inspect

import pytest

from puripuly_heart.app.ports.provider_verifier import (
    PROVIDER_VERIFICATION_STATUS_FAILED,
    PROVIDER_VERIFICATION_STATUS_VERIFIED,
    ProviderVerificationRequest,
)


@pytest.mark.asyncio
async def test_provider_verifier_adapter_maps_controller_provider_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_module = importlib.import_module("puripuly_heart.app.adapters.provider_verifier")
    adapter = adapter_module.ProviderVerifierAdapter()
    calls: list[tuple[str, str, str | None]] = []

    async def fake_gemini(api_key: str, *, model: str) -> bool:
        calls.append(("google", api_key, model))
        return True

    async def fake_openrouter(api_key: str) -> bool:
        calls.append(("openrouter", api_key, None))
        return True

    async def fake_deepseek(api_key: str) -> bool:
        calls.append(("deepseek", api_key, None))
        return True

    async def fake_cerebras(api_key: str, *, model: str | None = None) -> bool:
        calls.append(("cerebras", api_key, model))
        return True

    async def fake_deepgram(api_key: str) -> bool:
        calls.append(("deepgram", api_key, None))
        return True

    async def fake_soniox(api_key: str) -> bool:
        calls.append(("soniox", api_key, None))
        return True

    monkeypatch.setattr(
        adapter_module.GeminiLLMProvider,
        "verify_api_key",
        staticmethod(fake_gemini),
    )
    monkeypatch.setattr(
        adapter_module.OpenRouterLLMProvider,
        "verify_api_key",
        staticmethod(fake_openrouter),
    )
    monkeypatch.setattr(
        adapter_module.DeepSeekLLMProvider,
        "verify_api_key",
        staticmethod(fake_deepseek),
    )
    monkeypatch.setattr(
        adapter_module.CerebrasLLMProvider,
        "verify_api_key",
        staticmethod(fake_cerebras),
    )
    monkeypatch.setattr(
        adapter_module.DeepgramRealtimeSTTBackend,
        "verify_api_key",
        staticmethod(fake_deepgram),
    )
    monkeypatch.setattr(
        adapter_module.SonioxRealtimeSTTBackend,
        "verify_api_key",
        staticmethod(fake_soniox),
    )

    assert await adapter.verify_api_key("google", "secret", model="gemini-custom")
    assert await adapter.verify_api_key("openrouter", "secret")
    assert await adapter.verify_api_key("deepseek", "secret")
    assert await adapter.verify_api_key("cerebras", "secret", model="gemma-4-31b")
    assert await adapter.verify_api_key("deepgram", "secret")
    assert await adapter.verify_api_key("soniox", "secret")

    assert calls == [
        ("google", "secret", "gemini-custom"),
        ("openrouter", "secret", None),
        ("deepseek", "secret", None),
        ("cerebras", "secret", "gemma-4-31b"),
        ("deepgram", "secret", None),
        ("soniox", "secret", None),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("historical_value", [True, False])
async def test_provider_verifier_adapter_always_uses_async_qwen_compatible_mode(
    monkeypatch: pytest.MonkeyPatch,
    historical_value: bool,
) -> None:
    adapter_module = importlib.import_module("puripuly_heart.app.adapters.provider_verifier")
    adapter = adapter_module.ProviderVerifierAdapter()
    calls: list[tuple[str, str, str]] = []

    async def fake_async_qwen(api_key: str, *, base_url: str, model: str) -> bool:
        calls.append((api_key, base_url, model))
        return True

    monkeypatch.setattr(
        adapter_module.AsyncQwenLLMProvider,
        "verify_api_key",
        staticmethod(fake_async_qwen),
    )
    assert await adapter.verify_qwen_llm_api_key(
        "secret",
        base_url="https://dashscope.aliyuncs.com/api/v1",
        model="qwen3.5-flash",
        low_latency=historical_value,
    )
    assert calls == [
        (
            "secret",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "qwen3.5-flash",
        )
    ]


@pytest.mark.asyncio
async def test_provider_verifier_adapter_returns_safe_failed_result_when_provider_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_module = importlib.import_module("puripuly_heart.app.adapters.provider_verifier")
    adapter = adapter_module.ProviderVerifierAdapter()
    secret_value = "super-secret-provider-key"

    async def raise_secret_error(_api_key: str) -> bool:
        raise RuntimeError(f"token {secret_value} was rejected")

    monkeypatch.setattr(
        adapter_module.DeepSeekLLMProvider,
        "verify_api_key",
        staticmethod(raise_secret_error),
    )

    result = await adapter.verify_provider_secret(
        ProviderVerificationRequest(
            provider="deepseek",
            secret_key="providers.deepseek.api_key",
            secret_value=secret_value,
            secret_revision="secret-r1",
            context={"flow": "settings.verify_api_key"},
        )
    )

    assert result.status == PROVIDER_VERIFICATION_STATUS_FAILED
    assert result.provider == "deepseek"
    assert result.secret_key == "providers.deepseek.api_key"
    assert result.diagnostics is not None
    assert result.diagnostics.fields["error_type"] == "RuntimeError"
    assert secret_value not in repr(result)
    assert secret_value not in repr(result.diagnostics)


@pytest.mark.asyncio
async def test_provider_verifier_adapter_returns_verified_port_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_module = importlib.import_module("puripuly_heart.app.adapters.provider_verifier")
    adapter = adapter_module.ProviderVerifierAdapter()

    async def fake_openrouter(_api_key: str) -> bool:
        return True

    monkeypatch.setattr(
        adapter_module.OpenRouterLLMProvider,
        "verify_api_key",
        staticmethod(fake_openrouter),
    )

    result = await adapter.verify_provider_secret(
        ProviderVerificationRequest(
            provider="openrouter",
            secret_key="providers.openrouter.api_key",
            secret_value="not-logged",
            secret_revision="secret-r1",
            context={"flow": "pkce"},
        )
    )

    assert result.status == PROVIDER_VERIFICATION_STATUS_VERIFIED
    assert result.diagnostics is None
    assert result.evidence["verifier"] == "provider_adapter"
    assert "not-logged" not in repr(result)


@pytest.mark.asyncio
async def test_provider_verifier_adapter_returns_cerebras_safe_bound_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_module = importlib.import_module("puripuly_heart.app.adapters.provider_verifier")
    adapter = adapter_module.ProviderVerifierAdapter()

    async def fake_cerebras(_api_key: str, *, model: str | None = None) -> bool:
        assert model == "gemma-4-31b"
        return True

    monkeypatch.setattr(
        adapter_module.CerebrasLLMProvider,
        "verify_api_key",
        staticmethod(fake_cerebras),
    )

    result = await adapter.verify_provider_secret(
        ProviderVerificationRequest(
            provider="cerebras",
            secret_key="cerebras_api_key",
            secret_value="raw-cerebras-secret",
            secret_revision="cerebras-r1",
            context={"flow": "settings.verify_api_key", "model": "gemma-4-31b"},
        )
    )

    assert result.status == PROVIDER_VERIFICATION_STATUS_VERIFIED
    assert result.provider == "cerebras"
    assert result.secret_key == "cerebras_api_key"
    assert result.secret_revision == "cerebras-r1"
    assert result.evidence == {
        "verifier": "provider_adapter",
        "provider": "cerebras",
        "context_count": 2,
    }
    assert "raw-cerebras-secret" not in repr(result)


def test_wiring_factory_and_openrouter_metadata_compatibility_imports() -> None:
    adapter_module = importlib.import_module("puripuly_heart.app.adapters.provider_verifier")
    wiring_module = importlib.import_module("puripuly_heart.app.wiring")
    core_metadata = importlib.import_module("puripuly_heart.core.openrouter_metadata")
    provider_openrouter = importlib.import_module("puripuly_heart.providers.llm.openrouter")

    verifier = wiring_module.create_provider_verifier()

    assert isinstance(verifier, adapter_module.ProviderVerifierAdapter)
    assert provider_openrouter.OpenRouterKeyMetadata is core_metadata.OpenRouterKeyMetadata
    assert inspect.iscoroutinefunction(
        adapter_module.ProviderVerifierAdapter.verify_provider_secret
    )
