from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

from puripuly_heart.core.openrouter_handoff import (
    MANAGED_EFFECTIVE_EXHAUSTION_USD,
    is_effectively_exhausted,
    mark_founder_letter_shown,
    should_auto_show_founder_letter,
    store_managed_entitlement_snapshot,
)

from puripuly_heart.app.wiring import ManagedIdentityStateAdapter
from puripuly_heart.config.settings_vnext.schema import ManagedConnectionState
from puripuly_heart.providers.llm.openrouter import OpenRouterKeyMetadata


def _managed_state(
    *,
    founder_letter_seen_credential_ref: str | None = None,
) -> ManagedIdentityStateAdapter:
    values = asdict(ManagedConnectionState())
    values["founder_letter_seen_credential_ref"] = founder_letter_seen_credential_ref
    return ManagedIdentityStateAdapter(SimpleNamespace(**values), lambda: None)


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
    state = _managed_state(founder_letter_seen_credential_ref="hash_old")

    store_managed_entitlement_snapshot(
        state,
        managed_credential_ref="hash_new",
        expires_at="2026-10-17T12:34:56Z",
    )

    assert state.active_managed_credential_ref == "hash_new"
    assert state.active_managed_expires_at == "2026-10-17T12:34:56Z"
    assert state.founder_letter_seen_credential_ref is None


def test_should_auto_show_founder_letter_is_true_once_per_entitlement() -> None:
    state = _managed_state()
    store_managed_entitlement_snapshot(
        state,
        managed_credential_ref="hash_123",
        expires_at=None,
    )

    metadata = OpenRouterKeyMetadata(limit_usd=0.07, remaining_usd=0.0007, usage_usd=0.0693)
    assert should_auto_show_founder_letter(state, metadata) is True

    mark_founder_letter_shown(state)
    assert should_auto_show_founder_letter(state, metadata) is False


def test_store_managed_entitlement_snapshot_preserves_existing_ref_on_partial_reissue() -> None:
    state = _managed_state()
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
    assert state.active_managed_credential_ref == "hash_real"
    assert state.active_managed_expires_at == "2026-11-01T00:00:00Z"
    assert state.founder_letter_seen_credential_ref == "hash_real"
    assert should_auto_show_founder_letter(state, metadata) is False
