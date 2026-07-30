from __future__ import annotations

from puripuly_heart.app.ports.managed_identity_state import ManagedIdentityStatePort

from .openrouter_metadata import OpenRouterKeyMetadata

MANAGED_EFFECTIVE_EXHAUSTION_USD = 0.0007


def is_effectively_exhausted(metadata: OpenRouterKeyMetadata | None) -> bool:
    return bool(
        metadata is not None
        and metadata.remaining_usd is not None
        and metadata.remaining_usd <= MANAGED_EFFECTIVE_EXHAUSTION_USD
    )


def store_managed_entitlement_snapshot(
    state: ManagedIdentityStatePort,
    *,
    managed_credential_ref: str | None,
    expires_at: str | None,
) -> None:
    existing_ref = state.active_managed_credential_ref
    normalized_ref = (
        (managed_credential_ref or "").strip()
        or existing_ref
        or (expires_at or "").strip()
        or state.installation_id
        or "managed-entitlement"
    )
    if state.active_managed_credential_ref != normalized_ref:
        state.founder_letter_seen_credential_ref = None
    state.active_managed_credential_ref = normalized_ref
    state.active_managed_expires_at = (expires_at or "").strip() or None


def should_auto_show_founder_letter(
    state: ManagedIdentityStatePort,
    metadata: OpenRouterKeyMetadata | None,
) -> bool:
    active_ref = state.active_managed_credential_ref
    return bool(
        active_ref
        and is_effectively_exhausted(metadata)
        and state.founder_letter_seen_credential_ref != active_ref
    )


def mark_founder_letter_shown(state: ManagedIdentityStatePort) -> None:
    state.founder_letter_seen_credential_ref = state.active_managed_credential_ref
