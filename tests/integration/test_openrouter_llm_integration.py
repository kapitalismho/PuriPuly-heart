from __future__ import annotations

import os
from uuid import UUID

import pytest

from puripuly_heart.app.wiring import create_llm_provider
from puripuly_heart.config.settings import (
    AppSettings,
    LLMProviderName,
    OpenRouterCredentialSource,
    OpenRouterFallbackSelectionAlias,
    OpenRouterLLMModel,
    OpenRouterSelectionAlias,
)
from puripuly_heart.core.llm.fallback_racing import FallbackRacingLLMProvider
from puripuly_heart.core.openrouter_credentials import OPENROUTER_MANAGED_API_KEY_SECRET
from puripuly_heart.core.storage.secrets import InMemorySecretStore
from puripuly_heart.domain.models import Translation
from puripuly_heart.providers.llm.openrouter import OpenRouterLLMProvider
from tests.integration.helpers import (
    integration_mark,
    require_env,
    run_llm_smoke,
    suppressed_runtime_logger,
)

pytestmark = integration_mark()


class FailFastManagedReleaseService:
    def __init__(self) -> None:
        self.accessed: list[str] = []
        self.managed_state = None

    def __getattr__(self, name: str) -> object:
        self.accessed.append(name)
        raise AssertionError(f"managed release service unexpectedly used: {name}")


def fail_if_managed_delegate_ready() -> None:
    raise AssertionError("managed delegate-ready callback unexpectedly invoked")


class CloseTrackingOpenRouterBranch:
    def __init__(self, inner: OpenRouterLLMProvider) -> None:
        self.inner = inner
        self.close_calls = 0

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ) -> Translation:
        return await self.inner.translate(
            utterance_id=utterance_id,
            text=text,
            system_prompt=system_prompt,
            source_language=source_language,
            target_language=target_language,
            context=context,
        )

    async def close(self) -> None:
        self.close_calls += 1
        await self.inner.close()


def openrouter_fallback_timeout_ms() -> int:
    if os.getenv("OPENROUTER_FORCE_FALLBACK_RACE") == "1":
        return 1
    return 60_000


@pytest.mark.asyncio
async def test_openrouter_byok_translation_smoke() -> None:
    api_key = require_env("OPENROUTER_API_KEY")

    provider = OpenRouterLLMProvider(
        api_key=api_key,
        model=os.getenv(
            "OPENROUTER_LLM_MODEL",
            OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value,
        ),
        runtime_logging=suppressed_runtime_logger(),
    )

    await run_llm_smoke(provider)


@pytest.mark.asyncio
async def test_openrouter_cached_managed_key_translation_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_key = require_env("OPENROUTER_TEST_CACHED_MANAGED_API_KEY")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.llm_model = OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.openrouter.selection_alias = OpenRouterSelectionAlias.GEMMA4_MANAGED
    settings.openrouter.fallback_selection_alias = OpenRouterFallbackSelectionAlias.NONE

    secrets = InMemorySecretStore()
    secrets.set(OPENROUTER_MANAGED_API_KEY_SECRET, api_key)

    managed_release_service = FailFastManagedReleaseService()
    provider = create_llm_provider(
        settings,
        secrets=secrets,
        managed_release_service=managed_release_service,
        managed_delegate_ready=fail_if_managed_delegate_ready,
        runtime_logging=suppressed_runtime_logger(),
    )

    await run_llm_smoke(provider)
    assert managed_release_service.accessed == []


@pytest.mark.asyncio
async def test_openrouter_fallback_configuration_translation_smoke() -> None:
    api_key = require_env("OPENROUTER_API_KEY")

    primary = CloseTrackingOpenRouterBranch(
        OpenRouterLLMProvider(
            api_key=api_key,
            model=os.getenv(
                "OPENROUTER_PRIMARY_MODEL",
                OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value,
            ),
            runtime_logging=suppressed_runtime_logger(),
        )
    )
    fallback = CloseTrackingOpenRouterBranch(
        OpenRouterLLMProvider(
            api_key=api_key,
            model=os.getenv(
                "OPENROUTER_FALLBACK_MODEL",
                OpenRouterLLMModel.DEEPSEEK_V4_FLASH.value,
            ),
            runtime_logging=suppressed_runtime_logger(),
        )
    )
    provider = FallbackRacingLLMProvider(
        primary=primary,
        fallback=fallback,
        fallback_timeout_ms=openrouter_fallback_timeout_ms(),
        runtime_logging=suppressed_runtime_logger(),
    )

    # Default to validating the fallback configuration without starting a concurrent
    # fallback request; explicit OPENROUTER_FORCE_FALLBACK_RACE=1 opts into racing cost.
    await run_llm_smoke(provider)

    assert primary.close_calls == 1
    assert fallback.close_calls == 1
