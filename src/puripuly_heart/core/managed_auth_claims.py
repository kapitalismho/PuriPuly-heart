from __future__ import annotations

from collections.abc import Callable

from puripuly_heart.config.settings import (
    MANAGED_AUTH_CLAIM_SOURCE_DISCORD,
    MANAGED_AUTH_CLAIM_SOURCE_QQ,
    MANAGED_AUTH_CLAIM_SOURCES,
    AppSettings,
    normalize_managed_claim_sources,
)
from puripuly_heart.core.openrouter_credentials import (
    OPENROUTER_MANAGED_API_KEY_SECRET,
    OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
)
from puripuly_heart.core.storage.secrets import SecretStore

ManagedAuthClaimSource = str
PersistSettings = Callable[[AppSettings], None]


def record_local_managed_claim_source(
    settings: AppSettings,
    source: ManagedAuthClaimSource,
) -> bool:
    normalized_source = _normalize_claim_source(source)
    if normalized_source is None:
        return False

    previous_sources = settings.managed_identity.local_managed_claim_sources
    current_sources = set(
        normalize_managed_claim_sources(settings.managed_identity.local_managed_claim_sources)
    )
    canonical_sources = tuple(
        source for source in MANAGED_AUTH_CLAIM_SOURCES if source in current_sources
    )
    if normalized_source in current_sources:
        settings.managed_identity.local_managed_claim_sources = canonical_sources
        return previous_sources != canonical_sources

    current_sources.add(normalized_source)
    settings.managed_identity.local_managed_claim_sources = tuple(
        source for source in MANAGED_AUTH_CLAIM_SOURCES if source in current_sources
    )
    return True


def backfill_local_managed_claim_sources(
    settings: AppSettings,
    secrets: SecretStore,
    *,
    persist_settings: PersistSettings | None = None,
) -> tuple[str, ...]:
    previous_sources = normalize_managed_claim_sources(
        settings.managed_identity.local_managed_claim_sources
    )
    changed = False
    for source, secret_key in (
        (MANAGED_AUTH_CLAIM_SOURCE_DISCORD, OPENROUTER_MANAGED_API_KEY_SECRET),
        (MANAGED_AUTH_CLAIM_SOURCE_QQ, OPENROUTER_MANAGED_QQ_API_KEY_SECRET),
    ):
        if _secret_available(secrets, secret_key):
            changed = record_local_managed_claim_source(settings, source) or changed

    settings.managed_identity.local_managed_claim_sources = normalize_managed_claim_sources(
        settings.managed_identity.local_managed_claim_sources
    )
    if changed and persist_settings is not None:
        try:
            persist_settings(settings)
        except Exception:
            settings.managed_identity.local_managed_claim_sources = previous_sources
            raise
    return settings.managed_identity.local_managed_claim_sources


def local_managed_auth_blocking_source(
    settings: AppSettings,
    requested_source: ManagedAuthClaimSource,
) -> str | None:
    normalized_requested_source = _normalize_claim_source(requested_source)
    if normalized_requested_source is None:
        return None
    for source in normalize_managed_claim_sources(
        settings.managed_identity.local_managed_claim_sources
    ):
        if source != normalized_requested_source:
            return source
    return None


def _secret_available(secrets: SecretStore, key: str) -> bool:
    try:
        value = secrets.get(key)
    except Exception:
        return False
    return isinstance(value, str) and bool(value.strip())


def _normalize_claim_source(source: object) -> str | None:
    if not isinstance(source, str):
        return None
    normalized = source.strip().lower()
    if normalized not in MANAGED_AUTH_CLAIM_SOURCES:
        return None
    return normalized
