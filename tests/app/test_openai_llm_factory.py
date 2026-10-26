from __future__ import annotations

import pytest

from puripuly_heart.app.wiring.wiring_llm_factory import create_llm_provider_from_resolved_config
from puripuly_heart.config.resolved import (
    CREDENTIAL_SOURCE_MANAGED,
    CREDENTIAL_SOURCE_SECRET_STORE,
    ResolvedCredentialRequirement,
    ResolvedLLMConfig,
    ResolvedLLMTarget,
)
from puripuly_heart.config.runtime_resolution import (
    CREDENTIAL_REF_OPENAI_BYOK,
    OPENAI_MODEL_GPT_6_LUNA,
    PROVIDER_OPENAI,
)
from puripuly_heart.core.llm.provider import SemaphoreLLMProvider
from puripuly_heart.core.storage.secrets import InMemorySecretStore
from puripuly_heart.providers.llm.openai import OpenAILLMProvider


def _openai_config(
    *,
    model: str = OPENAI_MODEL_GPT_6_LUNA,
    credential_source: str = CREDENTIAL_SOURCE_SECRET_STORE,
    credential_reference: str = CREDENTIAL_REF_OPENAI_BYOK,
) -> ResolvedLLMConfig:
    return ResolvedLLMConfig(
        primary=ResolvedLLMTarget(
            provider=PROVIDER_OPENAI,
            model=model,
            credential=ResolvedCredentialRequirement(
                source=credential_source,
                required=True,
                reference=credential_reference,
            ),
        ),
        concurrency_limit=3,
    )


def _secret_store(**values: str) -> InMemorySecretStore:
    secrets = InMemorySecretStore()
    for key, value in values.items():
        secrets.set(key, value)
    return secrets


def test_openai_factory_uses_only_its_secret_and_exact_luna_model() -> None:
    secrets = _secret_store(
        openai_api_key="openai-key",
        openrouter_api_key="openrouter-key",
    )

    provider = create_llm_provider_from_resolved_config(_openai_config(), secrets=secrets)

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, OpenAILLMProvider)
    assert provider.inner.api_key == "openai-key"
    assert provider.inner.model == OPENAI_MODEL_GPT_6_LUNA
    assert "openai-key" not in repr(provider)


def test_openai_factory_uses_existing_environment_key_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "environment-openai-key")
    provider = create_llm_provider_from_resolved_config(
        _openai_config(),
        secrets=InMemorySecretStore(),
    )

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, OpenAILLMProvider)
    assert provider.inner.api_key == "environment-openai-key"


def test_openai_factory_rejects_managed_or_other_model_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secrets = _secret_store(openai_api_key="openai-key")
    with pytest.raises(ValueError, match="official BYOK"):
        create_llm_provider_from_resolved_config(
            _openai_config(credential_source=CREDENTIAL_SOURCE_MANAGED),
            secrets=secrets,
        )
    with pytest.raises(ValueError, match="official BYOK"):
        create_llm_provider_from_resolved_config(
            _openai_config(credential_reference="openrouter:byok"),
            secrets=secrets,
        )
    with pytest.raises(ValueError):
        create_llm_provider_from_resolved_config(
            _openai_config(model="some-other-model"),
            secrets=secrets,
        )


def test_openai_factory_requires_its_key_when_environment_and_secret_are_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="openai_api_key"):
        create_llm_provider_from_resolved_config(
            _openai_config(),
            secrets=InMemorySecretStore(),
        )
