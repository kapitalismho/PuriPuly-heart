from __future__ import annotations

import importlib

import httpx
import pytest

from puripuly_heart.app.ports.provider_verifier import (
    PROVIDER_VERIFICATION_STATUS_FAILED,
    PROVIDER_VERIFICATION_STATUS_VERIFIED,
    ProviderVerificationRequest,
)
from puripuly_heart.core import network_clients


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_status"),
    [(200, PROVIDER_VERIFICATION_STATUS_VERIFIED), (403, PROVIDER_VERIFICATION_STATUS_FAILED)],
)
async def test_shared_gemini_key_verification_is_an_authentication_probe(
    monkeypatch: pytest.MonkeyPatch, status_code: int, expected_status: str
) -> None:
    from puripuly_heart.app.adapters.provider_verifier import ProviderVerifierAdapter

    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status_code, json={"models": []})

    monkeypatch.setattr(
        network_clients,
        "external_client",
        lambda **kwargs: httpx.Client(transport=httpx.MockTransport(respond), **kwargs),
    )
    result = await ProviderVerifierAdapter().verify_provider_secret(
        ProviderVerificationRequest(
            provider="gemini",
            secret_key="gemini_api_key",
            secret_value="synthetic-shared-key",
            secret_revision="revision-1",
            context={"flow": "settings.verify_api_key"},
        )
    )

    assert result.status == expected_status
    assert result.evidence["verification_scope"] == "key_probe"
    assert result.evidence["recognition_readiness"] == "unobserved"
    assert [(request.method, request.url.path) for request in requests] == [
        ("GET", "/v1beta/models")
    ]
    assert requests[0].headers["x-goog-api-key"] == "synthetic-shared-key"
    assert "synthetic-shared-key" not in repr(result)


@pytest.mark.asyncio
async def test_direct_openai_verification_requires_the_luna_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_module = importlib.import_module("puripuly_heart.app.adapters.provider_verifier")
    calls: list[tuple[str, str]] = []

    async def probe(api_key: str, *, model: str) -> bool:
        calls.append((api_key, model))
        return True

    monkeypatch.setattr(
        adapter_module.OpenAILLMProvider,
        "verify_api_key",
        staticmethod(probe),
    )
    adapter = adapter_module.ProviderVerifierAdapter()
    assert await adapter.verify_api_key("openai", "luna-secret", model="gpt-6-luna")
    assert not await adapter.verify_api_key("openai", "luna-secret", model="other-model")
    assert calls == [("luna-secret", "gpt-6-luna")]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["cerebras", "google", "gemini_transcribe"])
async def test_retired_provider_cannot_be_verified(provider: str) -> None:
    adapter_module = importlib.import_module("puripuly_heart.app.adapters.provider_verifier")

    with pytest.raises(ValueError, match="Unknown provider"):
        await adapter_module.ProviderVerifierAdapter().verify_api_key(provider, "retired-key")


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
    assert result.diagnostics.fields["exception_class"] == "RuntimeError"
    assert result.diagnostics.category == "unknown"
    assert result.diagnostics.fields["verification_scope"] == "key_probe"
    assert result.diagnostics.fields["recognition_readiness"] == "unobserved"
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
