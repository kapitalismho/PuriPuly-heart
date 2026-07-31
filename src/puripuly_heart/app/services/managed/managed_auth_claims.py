from __future__ import annotations

from dataclasses import dataclass

from puripuly_heart.app.ports.managed_identity_state import ManagedIdentityStatePort
from puripuly_heart.app.ports.secret_store import SecretStorePort
from puripuly_heart.core.messages import (
    CONTENT_POLICY_METADATA_ONLY,
    DIAGNOSTIC_CATEGORY_AUTH,
    DIAGNOSTIC_VISIBILITY_BASIC,
    SEVERITY_ERROR,
    TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED,
    ErrorDiagnostics,
    TransactionResult,
    UserMessageRef,
)

MANAGED_AUTH_CLAIM_SOURCE_DISCORD = "discord"
MANAGED_AUTH_CLAIM_SOURCE_QQ = "qq"
MANAGED_AUTH_CLAIM_SOURCES = (
    MANAGED_AUTH_CLAIM_SOURCE_DISCORD,
    MANAGED_AUTH_CLAIM_SOURCE_QQ,
)
OPENROUTER_MANAGED_API_KEY_SECRET = "openrouter_managed_api_key"
OPENROUTER_MANAGED_QQ_API_KEY_SECRET = "openrouter_managed_qq_api_key"
OPENROUTER_MANAGED_USER_ID_SECRET = "openrouter_managed_user_id"
OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET = "openrouter_managed_user_installation_id"
OPENROUTER_MANAGED_USER_ID_MAX_LENGTH = 256


@dataclass(frozen=True, slots=True)
class ManagedAuthClaimGuard:
    managed_state: ManagedIdentityStatePort
    secret_store: SecretStorePort

    async def preflight(self, requested_source: str) -> TransactionResult | None:
        normalized_requested_source = normalize_managed_claim_source(requested_source)
        if normalized_requested_source is None:
            return _blocked_result(
                requested_source="unknown",
                blocking_source="unknown",
                code="managed_auth_claim_source_invalid",
                message_key="managed_auth.error.claim_conflict",
            )

        changed = await self.backfill_from_local_secrets()
        if changed:
            try:
                self.managed_state.persist()
            except Exception:
                return _blocked_result(
                    requested_source=normalized_requested_source,
                    blocking_source="unknown",
                    code="managed_auth_claim_backfill_persist_failed",
                    message_key=_message_key_for_block(
                        normalized_requested_source,
                        "unknown",
                    ),
                )

        blocking_source = local_managed_auth_blocking_source(
            self.managed_state.local_managed_claim_sources,
            normalized_requested_source,
        )
        if blocking_source is None:
            return None
        return _blocked_result(
            requested_source=normalized_requested_source,
            blocking_source=blocking_source,
            code="managed_auth_claim_source_blocked",
            message_key=_message_key_for_block(normalized_requested_source, blocking_source),
        )

    async def backfill_from_local_secrets(self) -> bool:
        changed = False
        for source, secret_key in (
            (MANAGED_AUTH_CLAIM_SOURCE_DISCORD, OPENROUTER_MANAGED_API_KEY_SECRET),
            (MANAGED_AUTH_CLAIM_SOURCE_QQ, OPENROUTER_MANAGED_QQ_API_KEY_SECRET),
        ):
            if await self._secret_available(secret_key):
                changed = record_local_managed_claim_source(self.managed_state, source) or changed
        canonical = normalize_managed_claim_sources(self.managed_state.local_managed_claim_sources)
        if self.managed_state.local_managed_claim_sources != canonical:
            self.managed_state.local_managed_claim_sources = canonical
            changed = True
        return changed

    def record_success(self, source: str) -> bool:
        return record_local_managed_claim_source(self.managed_state, source)

    async def _secret_available(self, key: str) -> bool:
        try:
            result = await self.secret_store.get_secret(key)
        except Exception:
            return False
        value = result.value
        return isinstance(value, str) and bool(value.strip())


def record_local_managed_claim_source(
    managed_state: ManagedIdentityStatePort,
    source: str,
) -> bool:
    normalized_source = normalize_managed_claim_source(source)
    if normalized_source is None:
        return False
    previous_sources = managed_state.local_managed_claim_sources
    current_sources = set(normalize_managed_claim_sources(previous_sources))
    canonical_sources = tuple(
        source for source in MANAGED_AUTH_CLAIM_SOURCES if source in current_sources
    )
    if normalized_source in current_sources:
        managed_state.local_managed_claim_sources = canonical_sources
        return previous_sources != canonical_sources
    current_sources.add(normalized_source)
    managed_state.local_managed_claim_sources = tuple(
        source for source in MANAGED_AUTH_CLAIM_SOURCES if source in current_sources
    )
    return True


def local_managed_auth_blocking_source(
    sources: object,
    requested_source: str,
) -> str | None:
    normalized_requested_source = normalize_managed_claim_source(requested_source)
    if normalized_requested_source is None:
        return None
    for source in normalize_managed_claim_sources(sources):
        if source != normalized_requested_source:
            return source
    return None


def normalize_managed_claim_sources(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        candidates: tuple[object, ...] = (value,)
    elif isinstance(value, (list, tuple, set, frozenset)):
        candidates = tuple(value)
    else:
        candidates = ()
    normalized = {
        item.strip().lower()
        for item in candidates
        if isinstance(item, str) and item.strip().lower() in MANAGED_AUTH_CLAIM_SOURCES
    }
    return tuple(source for source in MANAGED_AUTH_CLAIM_SOURCES if source in normalized)


def normalize_managed_claim_source(source: object) -> str | None:
    if not isinstance(source, str):
        return None
    normalized = source.strip().lower()
    if normalized not in MANAGED_AUTH_CLAIM_SOURCES:
        return None
    return normalized


def _blocked_result(
    *,
    requested_source: str,
    blocking_source: str,
    code: str,
    message_key: str,
) -> TransactionResult:
    return TransactionResult(
        status=TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED,
        message=UserMessageRef(key=message_key, params={}, severity=SEVERITY_ERROR),
        diagnostics=ErrorDiagnostics(
            component="managed_auth_claims",
            operation="preflight_managed_auth_claim",
            code=code,
            category=DIAGNOSTIC_CATEGORY_AUTH,
            visibility=DIAGNOSTIC_VISIBILITY_BASIC,
            content_policy=CONTENT_POLICY_METADATA_ONLY,
            status_code=None,
            retry_after_ms=None,
            fields={
                "requested_source": requested_source,
                "blocking_source": blocking_source,
            },
        ),
    )


def _message_key_for_block(requested_source: str, blocking_source: str) -> str:
    if requested_source == MANAGED_AUTH_CLAIM_SOURCE_DISCORD:
        if blocking_source == MANAGED_AUTH_CLAIM_SOURCE_QQ:
            return "discord_auth.error.already_claimed_qq"
        return "managed_auth.error.claim_conflict"
    if requested_source == MANAGED_AUTH_CLAIM_SOURCE_QQ:
        if blocking_source == MANAGED_AUTH_CLAIM_SOURCE_DISCORD:
            return "qq_managed_auth.already_claimed_discord"
        return "managed_auth.error.claim_conflict"
    return "managed_auth.error.claim_conflict"


__all__ = [
    "MANAGED_AUTH_CLAIM_SOURCE_DISCORD",
    "MANAGED_AUTH_CLAIM_SOURCE_QQ",
    "MANAGED_AUTH_CLAIM_SOURCES",
    "OPENROUTER_MANAGED_API_KEY_SECRET",
    "OPENROUTER_MANAGED_QQ_API_KEY_SECRET",
    "OPENROUTER_MANAGED_USER_ID_MAX_LENGTH",
    "OPENROUTER_MANAGED_USER_ID_SECRET",
    "OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET",
    "ManagedAuthClaimGuard",
    "local_managed_auth_blocking_source",
    "normalize_managed_claim_source",
    "normalize_managed_claim_sources",
    "record_local_managed_claim_source",
]
