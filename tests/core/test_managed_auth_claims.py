from __future__ import annotations

from puripuly_heart.config.settings import (
    MANAGED_AUTH_CLAIM_SOURCE_DISCORD,
    MANAGED_AUTH_CLAIM_SOURCE_QQ,
    AppSettings,
)
from puripuly_heart.core.managed_auth_claims import (
    backfill_local_managed_claim_sources,
    local_managed_auth_blocking_source,
    record_local_managed_claim_source,
)
from puripuly_heart.core.openrouter_credentials import (
    OPENROUTER_MANAGED_API_KEY_SECRET,
    OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
)
from puripuly_heart.core.storage.secrets import InMemorySecretStore


def test_backfill_local_managed_claim_sources_from_existing_managed_secrets() -> None:
    settings = AppSettings()
    secrets = InMemorySecretStore()
    secrets.set(OPENROUTER_MANAGED_API_KEY_SECRET, " discord-key ")
    secrets.set(OPENROUTER_MANAGED_QQ_API_KEY_SECRET, "qq-key")
    persisted: list[tuple[str, ...]] = []

    sources = backfill_local_managed_claim_sources(
        settings,
        secrets,
        persist_settings=lambda updated: persisted.append(
            updated.managed_identity.local_managed_claim_sources
        ),
    )

    assert sources == (MANAGED_AUTH_CLAIM_SOURCE_DISCORD, MANAGED_AUTH_CLAIM_SOURCE_QQ)
    assert settings.managed_identity.local_managed_claim_sources == sources
    assert persisted == [sources]


def test_backfill_ignores_blank_or_missing_managed_secrets_without_persisting() -> None:
    settings = AppSettings()
    secrets = InMemorySecretStore()
    secrets.set(OPENROUTER_MANAGED_API_KEY_SECRET, "   ")
    persisted: list[tuple[str, ...]] = []

    sources = backfill_local_managed_claim_sources(
        settings,
        secrets,
        persist_settings=lambda updated: persisted.append(
            updated.managed_identity.local_managed_claim_sources
        ),
    )

    assert sources == ()
    assert persisted == []


def test_local_managed_auth_blocking_source_blocks_only_cross_source_claims() -> None:
    settings = AppSettings()

    assert record_local_managed_claim_source(settings, MANAGED_AUTH_CLAIM_SOURCE_DISCORD) is True
    assert (
        local_managed_auth_blocking_source(settings, MANAGED_AUTH_CLAIM_SOURCE_QQ)
        == MANAGED_AUTH_CLAIM_SOURCE_DISCORD
    )
    assert local_managed_auth_blocking_source(settings, MANAGED_AUTH_CLAIM_SOURCE_DISCORD) is None

    assert record_local_managed_claim_source(settings, MANAGED_AUTH_CLAIM_SOURCE_QQ) is True
    assert (
        local_managed_auth_blocking_source(settings, MANAGED_AUTH_CLAIM_SOURCE_DISCORD)
        == MANAGED_AUTH_CLAIM_SOURCE_QQ
    )
