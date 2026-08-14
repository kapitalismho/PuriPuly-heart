from __future__ import annotations

from puripuly_heart.core.openrouter_handoff import (
    MANAGED_EFFECTIVE_EXHAUSTION_USD,
    is_effectively_exhausted,
    mark_founder_letter_shown,
    should_auto_show_founder_letter,
    store_managed_entitlement_snapshot,
)

from puripuly_heart.app.wiring import ManagedIdentityStateAdapter
from puripuly_heart.config.settings import AppSettings
from puripuly_heart.providers.llm.openrouter import OpenRouterKeyMetadata


def _managed_state(settings: AppSettings) -> ManagedIdentityStateAdapter:
    return ManagedIdentityStateAdapter(settings, lambda _updated: None)


def test_is_effectively_exhausted_uses_raw_remaining_usd_floor() -> None:
    assert MANAGED_EFFECTIVE_EXHAUSTION_USD == 0.0007
    assert (
        is_effectively_exhausted(
            OpenRouterKeyMetadata(limit_usd=0.07, remaining_usd=0.0007, usage_usd=0.0693)
        )
        is True
    )
    assert (
        is_effectively_exhausted(
            OpenRouterKeyMetadata(limit_usd=0.07, remaining_usd=0.00071, usage_usd=0.06929)
        )
        is False
    )


def test_store_managed_entitlement_snapshot_resets_seen_flag_for_new_ref() -> None:
    settings = AppSettings()
    settings.managed_identity.founder_letter_seen_credential_ref = "hash_old"

    store_managed_entitlement_snapshot(
        _managed_state(settings),
        managed_credential_ref="hash_new",
        expires_at="2026-10-17T12:34:56Z",
    )

    assert settings.managed_identity.active_managed_credential_ref == "hash_new"
    assert settings.managed_identity.active_managed_expires_at == "2026-10-17T12:34:56Z"
    assert settings.managed_identity.founder_letter_seen_credential_ref is None


def test_should_auto_show_founder_letter_is_true_once_per_entitlement() -> None:
    settings = AppSettings()
    store_managed_entitlement_snapshot(
        _managed_state(settings),
        managed_credential_ref="hash_123",
        expires_at=None,
    )

    metadata = OpenRouterKeyMetadata(limit_usd=0.07, remaining_usd=0.0007, usage_usd=0.0693)
    assert should_auto_show_founder_letter(_managed_state(settings), metadata) is True

    mark_founder_letter_shown(_managed_state(settings))
    assert should_auto_show_founder_letter(_managed_state(settings), metadata) is False


def test_store_managed_entitlement_snapshot_preserves_existing_ref_on_partial_reissue() -> None:
    settings = AppSettings()
    state = _managed_state(settings)
    store_managed_entitlement_snapshot(
        state,
        managed_credential_ref="hash_real",
        expires_at="2026-10-17T12:34:56Z",
    )
    mark_founder_letter_shown(state)

    store_managed_entitlement_snapshot(
        state,
        managed_credential_ref=None,
        expires_at="2026-11-01T00:00:00Z",
    )

    metadata = OpenRouterKeyMetadata(limit_usd=0.07, remaining_usd=0.0007, usage_usd=0.0693)
    assert settings.managed_identity.active_managed_credential_ref == "hash_real"
    assert settings.managed_identity.active_managed_expires_at == "2026-11-01T00:00:00Z"
    assert settings.managed_identity.founder_letter_seen_credential_ref == "hash_real"
    assert should_auto_show_founder_letter(_managed_state(settings), metadata) is False
