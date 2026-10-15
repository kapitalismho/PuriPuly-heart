from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
from types import SimpleNamespace
from uuid import uuid4

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
from puripuly_heart.app.wiring.wiring_llm_factory import llm_factory_extras_from_vnext
from puripuly_heart.config.provider_values import OpenRouterCredentialSource
from puripuly_heart.config.resolved import (
    CREDENTIAL_SOURCE_MANAGED,
    ResolvedCredentialRequirement,
    ResolvedLLMConfig,
    ResolvedLLMFallbackPlan,
    ResolvedLLMTarget,
)
from puripuly_heart.config.runtime_resolution import (
    RuntimeResolutionInput,
    TranslationRuntimeIntent,
    resolve_llm_config,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.llm.fallback_racing import FallbackRacingLLMProvider
from puripuly_heart.core.llm.provider import SemaphoreLLMProvider
from puripuly_heart.core.openrouter_routing import OpenRouterProviderRouting
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
            message_key="managed_release.retry",
            api_key="issued-managed-key",
            local_key_available=True,
        )


def _managed_china_settings() -> AppSettingsVNext:
    settings = AppSettingsVNext()
    return replace(
        settings,
        intent=replace(
            settings.intent,
            translation=replace(
                settings.intent.translation,
                model="deepseek_v4_flash",
                connection="managed_china",
                openrouter_selected_source="managed",
            ),
        ),
        state=replace(
            settings.state,
            managed_connection=replace(
                settings.state.managed_connection,
                active_managed_credential_ref="qq-managed-ref",
            ),
        ),
    )


def _managed_settings() -> AppSettingsVNext:
    settings = AppSettingsVNext()
    return replace(
        settings,
        intent=replace(
            settings.intent,
            translation=replace(
                settings.intent.translation,
                connection="managed",
                openrouter_selected_source="managed",
            ),
        ),
    )


def _identity_adapter(settings: AppSettingsVNext) -> ManagedIdentityStateAdapter:
    return ManagedIdentityStateAdapter(
        SimpleNamespace(**asdict(settings.state.managed_connection)),
        lambda _updated: None,
    )


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

    assert isinstance(provider, ManagedGemmaLLMProvider)
    assert provider.runtime is runtime
    assert provider.backend == "gpu"
    assert provider.spec.model_id == "gemma-4-e4b-it-qat-ud-q4-k-xl"

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
        extras=llm_factory_extras_from_vnext(settings),
    )

    assert isinstance(provider, SemaphoreLLMProvider)
    assert isinstance(provider.inner, OpenRouterLLMProvider)
    assert provider.inner.api_key == "qq-managed-key"


def test_managed_china_direct_provider_survives_missing_active_managed_state() -> None:
    settings = _managed_china_settings()
    settings = replace(
        settings,
        state=replace(
            settings.state,
            managed_connection=replace(
                settings.state.managed_connection,
                active_managed_credential_ref=None,
            ),
        ),
    )
    secrets = InMemorySecretStore()
    secrets.set(OPENROUTER_MANAGED_QQ_API_KEY_SECRET, "qq-managed-key")

    provider = create_llm_provider_from_resolved_config(
        _managed_china_resolved_config(),
        secrets=secrets,
        managed_release_service=object(),
        extras=llm_factory_extras_from_vnext(settings),
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
            extras=llm_factory_extras_from_vnext(settings),
        )


def test_resolved_qq_managed_credential_projects_release_runtime_kind_without_mutating_intent() -> (
    None
):
    settings = _managed_settings()

    projected = wiring_llm_factory._openrouter_release_config_from_resolved_fields(
        model=settings.intent.translation.openrouter_model,
        credential=_qq_managed_credential(),
        selected_source=OpenRouterCredentialSource.MANAGED,
        include_selection_alias=True,
    )

    assert settings.intent.translation.connection == "managed"
    assert projected.managed_credential_kind == "qq"


@pytest.mark.asyncio
async def test_mixed_managed_fallback_rebuilds_release_service_for_fallback_kind() -> None:
    settings = _managed_settings()
    secrets = InMemorySecretStore()
    release_service = ManagedOpenRouterReleaseService(
        openrouter_config=build_openrouter_release_runtime_config(settings),
        managed_state=_identity_adapter(settings),
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
        extras=llm_factory_extras_from_vnext(settings),
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
    settings = _managed_settings()
    managed_state = RecordingManagedState(local_claim_sources=("qq",))
    release_service = RecordingReleaseService(managed_state)

    provider = create_llm_provider_from_resolved_config(
        _standard_managed_resolved_config(),
        secrets=InMemorySecretStore(),
        managed_release_service=release_service,
        extras=llm_factory_extras_from_vnext(settings),
    )

    result = await provider.inner.release_service.ensure_key_for_llm_start()  # type: ignore[attr-defined]

    assert result.message_key == "discord_auth.error.already_claimed_qq"
    assert release_service.ensure_calls == 0
    assert managed_state.local_managed_claim_sources == ("qq",)


@pytest.mark.asyncio
async def test_standard_lazy_managed_provider_records_discord_claim_after_release_ready() -> None:
    settings = _managed_settings()
    managed_state = RecordingManagedState()
    release_service = RecordingReleaseService(managed_state)

    provider = create_llm_provider_from_resolved_config(
        _standard_managed_resolved_config(),
        secrets=InMemorySecretStore(),
        managed_release_service=release_service,
        extras=llm_factory_extras_from_vnext(settings),
    )

    result = await provider.inner.release_service.ensure_key_for_llm_start()  # type: ignore[attr-defined]

    assert result.message_key == "managed_release.retry"
    assert release_service.ensure_calls == 1
    assert managed_state.local_managed_claim_sources == ("discord",)
    assert managed_state.persist_calls == 1


@pytest.mark.parametrize("connection", ["managed", "openrouter"])
def test_combined_gemma_composes_pooled_attempts_and_single_model_emergency(
    connection: str,
) -> None:
    from puripuly_heart.core.managed_openrouter_release import ManagedOpenRouterLLMProvider

    secrets = InMemorySecretStore()
    if connection == "managed":
        secrets.set(OPENROUTER_MANAGED_API_KEY_SECRET, "fixture-managed-key")
    else:
        secrets.set("openrouter_api_key", "fixture-byok-key")
    config = resolve_llm_config(
        RuntimeResolutionInput(
            translation=TranslationRuntimeIntent(model="gemma4_31b", connection=connection)
        )
    )
    provider = create_llm_provider_from_resolved_config(
        config,
        secrets=secrets,
        managed_release_service=object() if connection == "managed" else None,
    )

    assert isinstance(provider.inner, FallbackRacingLLMProvider)
    attempts = provider.inner.attempts
    assert [attempt.start_after_ms for attempt in attempts] == [0, 1300, 4400]
    assert [attempt.start_on_primary_error for attempt in attempts] == [False, True, False]
    assert provider.inner.loser_grace_ms == 50
    first = attempts[0].provider
    assert isinstance(first, OpenRouterLLMProvider)
    for attempt in attempts[1:]:
        candidate = attempt.provider.factory()
        if connection == "managed":
            assert isinstance(candidate, ManagedOpenRouterLLMProvider)
            candidate = candidate.delegate_factory("fixture-managed-key")
        assert isinstance(candidate, OpenRouterLLMProvider)
        if attempt.start_after_ms == 1300:
            assert candidate.models == (
                "google/gemma-4-26b-a4b-it",
                "google/gemma-4-31b-it",
            )
            assert candidate.provider_routing == OpenRouterProviderRouting.GEMMA4_26B_31B_LATENCY
        else:
            assert candidate.models == ("google/gemma-4-31b-it",)
            assert candidate.provider_routing == OpenRouterProviderRouting.GEMMA4_31B_MODELRUN_ONLY
    assert first.models == ("google/gemma-4-26b-a4b-it", "google/gemma-4-31b-it")
    assert first.provider_routing == OpenRouterProviderRouting.GEMMA4_26B_31B_LATENCY


def test_combined_gemma_byok_without_key_does_not_fall_back_to_managed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    secrets = InMemorySecretStore()
    secrets.set(OPENROUTER_MANAGED_API_KEY_SECRET, "fixture-managed-key")
    config = resolve_llm_config(
        RuntimeResolutionInput(
            translation=TranslationRuntimeIntent(
                model="gemma4_31b",
                connection="openrouter",
            )
        )
    )

    assert config.primary.credential.reference == "openrouter:byok"
    with pytest.raises(ValueError, match="Missing secret `openrouter_api_key`"):
        create_llm_provider_from_resolved_config(
            config,
            secrets=secrets,
            managed_release_service=object(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "model_id"),
    [
        ("managed_gemma", "gemma-4-e4b-it-qat-ud-q4-k-xl"),
        ("managed_gemma_26b", "gemma-4-26b-a4b-it-qat-ud-q4-k-xl"),
    ],
)
async def test_managed_factory_leaves_gpu_admission_to_shared_runtime_even_with_limit_one(
    model: str, model_id: str
) -> None:
    both_entered = asyncio.Event()
    finish = asyncio.Event()
    calls = []

    class Runtime:
        async def translate(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                both_entered.set()
            await finish.wait()
            return SimpleNamespace(text=kwargs["user_message"])

    provider = create_llm_provider_from_resolved_config(
        resolve_llm_config(
            RuntimeResolutionInput(
                translation=TranslationRuntimeIntent(
                    model=model, connection="gpu", concurrency_limit=1
                )
            )
        ),
        secrets=InMemorySecretStore(),
        managed_gemma_runtime=Runtime(),
    )
    identities = [uuid4(), uuid4()]
    tasks = [
        asyncio.create_task(
            provider.translate(
                utterance_id=identity,
                text=f"synthetic {index}",
                system_prompt="translate",
                source_language="en",
                target_language="ko",
            )
        )
        for index, identity in enumerate(identities)
    ]
    try:
        await asyncio.wait_for(both_entered.wait(), timeout=2)
        assert all(call["spec"].model_id == model_id for call in calls)
        finish.set()
        results = await asyncio.gather(*tasks)
        assert [result.utterance_id for result in results] == identities
        assert results[0].text != results[1].text
    finally:
        finish.set()
        await asyncio.gather(*tasks, return_exceptions=True)
