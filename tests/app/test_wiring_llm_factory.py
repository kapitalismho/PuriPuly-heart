from __future__ import annotations

import pytest
from puripuly_heart.app.wiring_managed_auth_factory import (
    ManagedIdentityStateAdapter,
    build_openrouter_release_runtime_config,
)
from puripuly_heart.core.managed_openrouter_release import (
    ManagedOpenRouterReleaseBehavior,
    ManagedOpenRouterReleaseResult,
    ManagedOpenRouterReleaseService,
    UnavailableManagedOpenRouterReleaseClient,
)
from puripuly_heart.core.openrouter_credentials import (
    OPENROUTER_MANAGED_API_KEY_SECRET,
    OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
)

from puripuly_heart.app import wiring_llm_factory
from puripuly_heart.app.wiring import create_llm_provider_from_resolved_config
from puripuly_heart.config.resolved import (
    CREDENTIAL_SOURCE_MANAGED,
    ResolvedCredentialRequirement,
    ResolvedLLMConfig,
    ResolvedLLMFallbackPlan,
    ResolvedLLMTarget,
)
from puripuly_heart.config.settings import (
    AppSettings,
    LLMProviderName,
    OpenRouterCredentialSource,
    TranslationConnection,
    TranslationModel,
)
from puripuly_heart.core.llm.provider import SemaphoreLLMProvider
from puripuly_heart.core.storage.secrets import InMemorySecretStore
from puripuly_heart.providers.llm.managed_gemma import ManagedGemmaLLMProvider
from puripuly_heart.providers.llm.openrouter import OpenRouterLLMProvider


class RecordingManagedState:
    def __init__(self, *, local_claim_sources: tuple[str, ...] = ()) -> None:
        self.local_managed_claim_sources = local_claim_sources
        self.persist_calls = 0

    def persist(self) -> None:
        self.persist_calls += 1


class RecordingReleaseService:
    def __init__(self, managed_state: RecordingManagedState) -> None:
        self.managed_state = managed_state
        self.ensure_calls = 0

    async def ensure_key_for_llm_start(self) -> ManagedOpenRouterReleaseResult:
        self.ensure_calls += 1
        return ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            message_key="managed_release.ready",
            api_key="issued-managed-key",
            local_key_available=True,
        )


def _managed_china_settings() -> AppSettings:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.translation.model = TranslationModel.DEEPSEEK_V4_FLASH
    settings.translation.connection = TranslationConnection.MANAGED_CHINA
    settings.managed_identity.active_managed_credential_ref = "qq-managed-ref"
    return settings


def _managed_china_resolved_config() -> ResolvedLLMConfig:
    return ResolvedLLMConfig(
        primary=ResolvedLLMTarget(
            provider="openrouter",
            model="deepseek/deepseek-v4-flash-0731",
            credential=ResolvedCredentialRequirement(
                source=CREDENTIAL_SOURCE_MANAGED,
                required=True,
                reference="openrouter:managed_qq",
            ),
            provider_routing="deepseek_only",
        )
    )


def _standard_managed_resolved_config() -> ResolvedLLMConfig:
    return ResolvedLLMConfig(
        primary=ResolvedLLMTarget(
            provider="openrouter",
            model="google/gemma-4-26b-a4b-it",
            credential=ResolvedCredentialRequirement(
                source=CREDENTIAL_SOURCE_MANAGED,
                required=True,
                reference="openrouter:managed",
            ),
        )
    )


def _qq_managed_credential() -> ResolvedCredentialRequirement:
    return ResolvedCredentialRequirement(
        source=CREDENTIAL_SOURCE_MANAGED,
        required=True,
        reference="openrouter:managed_qq",
    )


@pytest.mark.asyncio
async def test_managed_gemma_target_uses_injected_runtime_and_activation_release() -> None:
    runtime = object()
    releases = 0

    async def release() -> None:
        nonlocal releases
        releases += 1

    provider = create_llm_provider_from_resolved_config(
        ResolvedLLMConfig(
            primary=ResolvedLLMTarget(
                provider="managed_gemma",
                model="puripuly-gemma-4-e4b-q4",
                provider_options={"backend": "gpu"},
            )
        ),
        secrets=InMemorySecretStore(),
        managed_gemma_runtime=runtime,
        managed_gemma_release=release,
    )

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, ManagedGemmaLLMProvider)
    assert provider.inner.runtime is runtime
    assert provider.inner.backend == "gpu"

    await provider.close()
    await provider.close()
    assert releases == 1


def test_managed_china_direct_provider_uses_qq_managed_secret() -> None:
    settings = _managed_china_settings()
    secrets = InMemorySecretStore()
    secrets.set(OPENROUTER_MANAGED_QQ_API_KEY_SECRET, "qq-managed-key")

    provider = create_llm_provider_from_resolved_config(
        _managed_china_resolved_config(),
        secrets=secrets,
        managed_release_service=object(),
        compatibility_settings=settings,
    )

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, OpenRouterLLMProvider)
    assert provider.inner.api_key == "qq-managed-key"


def test_managed_china_direct_provider_survives_missing_active_managed_state() -> None:
    settings = _managed_china_settings()
    settings.managed_identity.active_managed_credential_ref = None
    secrets = InMemorySecretStore()
    secrets.set(OPENROUTER_MANAGED_QQ_API_KEY_SECRET, "qq-managed-key")

    provider = create_llm_provider_from_resolved_config(
        _managed_china_resolved_config(),
        secrets=secrets,
        managed_release_service=object(),
        compatibility_settings=settings,
    )

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, OpenRouterLLMProvider)
    assert provider.inner.api_key == "qq-managed-key"


def test_managed_china_blocks_opposite_discord_managed_secret() -> None:
    settings = _managed_china_settings()
    secrets = InMemorySecretStore()
    secrets.set(OPENROUTER_MANAGED_API_KEY_SECRET, "standard-managed-key")

    with pytest.raises(ValueError, match="managed local claim conflict"):
        create_llm_provider_from_resolved_config(
            _managed_china_resolved_config(),
            secrets=secrets,
            managed_release_service=object(),
            compatibility_settings=settings,
        )


def test_resolved_qq_managed_credential_projects_release_runtime_kind_without_mutating_intent() -> (
    None
):
    settings = AppSettings()
    settings.translation.connection = TranslationConnection.MANAGED

    projected = wiring_llm_factory._settings_for_resolved_managed_credential(
        settings,
        _qq_managed_credential(),
    )

    assert settings.translation.connection == TranslationConnection.MANAGED
    assert projected.translation.connection == TranslationConnection.MANAGED_CHINA
    assert build_openrouter_release_runtime_config(projected).managed_credential_kind == "qq"


@pytest.mark.asyncio
async def test_mixed_managed_fallback_rebuilds_release_service_for_fallback_kind() -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.translation.connection = TranslationConnection.MANAGED
    secrets = InMemorySecretStore()
    release_service = ManagedOpenRouterReleaseService(
        openrouter_config=build_openrouter_release_runtime_config(settings),
        managed_state=ManagedIdentityStateAdapter(settings, lambda _updated: None),
        secrets=secrets,
        client=UnavailableManagedOpenRouterReleaseClient(),
        app_version="test",
    )
    config = ResolvedLLMConfig(
        primary=_standard_managed_resolved_config().primary,
        fallback=ResolvedLLMFallbackPlan(
            target=_managed_china_resolved_config().primary,
            force_managed_wrapper=True,
        ),
    )

    provider = create_llm_provider_from_resolved_config(
        config,
        secrets=secrets,
        managed_release_service=release_service,
        compatibility_settings=settings,
    )

    primary_release = provider.inner.primary.release_service  # type: ignore[attr-defined]
    fallback_delegate = await provider.inner.fallback._ensure_delegate()  # type: ignore[attr-defined]
    fallback_release = fallback_delegate.release_service

    assert primary_release.claim_source == "discord"
    assert primary_release.release_service.openrouter_config.managed_credential_kind == "standard"
    assert fallback_release.claim_source == "qq"
    assert fallback_release.release_service.openrouter_config.managed_credential_kind == "qq"


@pytest.mark.asyncio
async def test_standard_lazy_managed_provider_blocks_qq_claim_before_release_ensure() -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    managed_state = RecordingManagedState(local_claim_sources=("qq",))
    release_service = RecordingReleaseService(managed_state)

    provider = create_llm_provider_from_resolved_config(
        _standard_managed_resolved_config(),
        secrets=InMemorySecretStore(),
        managed_release_service=release_service,
        compatibility_settings=settings,
    )

    result = await provider.inner.release_service.ensure_key_for_llm_start()  # type: ignore[attr-defined]

    assert result.message_key == "discord_auth.error.already_claimed_qq"
    assert release_service.ensure_calls == 0
    assert managed_state.local_managed_claim_sources == ("qq",)


@pytest.mark.asyncio
async def test_standard_lazy_managed_provider_records_discord_claim_after_release_ready() -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    managed_state = RecordingManagedState()
    release_service = RecordingReleaseService(managed_state)

    provider = create_llm_provider_from_resolved_config(
        _standard_managed_resolved_config(),
        secrets=InMemorySecretStore(),
        managed_release_service=release_service,
        compatibility_settings=settings,
    )

    result = await provider.inner.release_service.ensure_key_for_llm_start()  # type: ignore[attr-defined]

    assert result.message_key == "managed_release.ready"
    assert release_service.ensure_calls == 1
    assert managed_state.local_managed_claim_sources == ("discord",)
    assert managed_state.persist_calls == 1
