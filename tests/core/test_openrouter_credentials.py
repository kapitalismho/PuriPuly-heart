from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import pytest
from puripuly_heart.core.openrouter_credentials import (
    OPENROUTER_BYOK_API_KEY_SECRET,
    OPENROUTER_MANAGED_API_KEY_SECRET,
    OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
    OPENROUTER_MANAGED_USER_ID_MAX_LENGTH,
    OPENROUTER_MANAGED_USER_ID_SECRET,
    OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET,
    OpenRouterCredentialRuntimeConfig,
    OpenRouterManagedRecoveryAction,
    clear_temporary_managed_release_state,
    handle_managed_availability,
    handle_managed_release_error,
    load_managed_openrouter_user_identifier,
    normalize_managed_openrouter_user_identifier,
    resolve_openrouter_credentials,
)

from puripuly_heart.app.wiring import ManagedIdentityStateAdapter
from puripuly_heart.config.provider_values import OpenRouterCredentialSource
from puripuly_heart.config.settings_vnext.schema import ManagedConnectionState
from puripuly_heart.core.storage.secrets import InMemorySecretStore


def _credential_config(
    *,
    selected_source: OpenRouterCredentialSource = OpenRouterCredentialSource.NONE,
    installation_id: str = "",
    managed_credential_kind: str = "standard",
    active_managed_credential_ref: str | None = None,
    active_managed_expires_at: str | None = None,
) -> OpenRouterCredentialRuntimeConfig:
    return OpenRouterCredentialRuntimeConfig(
        selected_source=selected_source,
        installation_id=installation_id,
        managed_credential_kind=managed_credential_kind,
        active_managed_credential_ref=active_managed_credential_ref,
        active_managed_expires_at=active_managed_expires_at,
    )


class TrackingSecretStore(InMemorySecretStore):
    def __init__(self) -> None:
        super().__init__()
        self.get_calls: list[str] = []

    def get(self, key: str) -> str | None:
        self.get_calls.append(key)
        return super().get(key)


def _managed_state(**fields: object) -> ManagedIdentityStateAdapter:
    values = asdict(ManagedConnectionState())
    values.update(fields)
    return ManagedIdentityStateAdapter(SimpleNamespace(**values), lambda: None)


def test_resolve_openrouter_credentials_respects_explicit_none_selection_even_with_stored_keys() -> (
    None
):
    store = InMemorySecretStore()
    store.set(OPENROUTER_BYOK_API_KEY_SECRET, "byok-key")
    store.set(OPENROUTER_MANAGED_API_KEY_SECRET, "managed-key")

    resolution = resolve_openrouter_credentials(
        _credential_config(selected_source=OpenRouterCredentialSource.NONE),
        secrets=store,
    )

    assert resolution.selected_source == OpenRouterCredentialSource.NONE
    assert resolution.api_key is None
    assert resolution.requires_managed_challenge is False


def test_resolve_openrouter_credentials_uses_selected_byok_key_without_managed_fallback() -> None:
    store = InMemorySecretStore()
    store.set(OPENROUTER_BYOK_API_KEY_SECRET, "byok-key")
    store.set(OPENROUTER_MANAGED_API_KEY_SECRET, "managed-key")

    resolution = resolve_openrouter_credentials(
        _credential_config(selected_source=OpenRouterCredentialSource.BYOK),
        secrets=store,
    )

    assert resolution.selected_source == OpenRouterCredentialSource.BYOK
    assert resolution.api_key == "byok-key"
    assert resolution.requires_managed_challenge is False


def test_resolve_openrouter_credentials_uses_selected_managed_key_without_byok_fallback() -> None:
    store = InMemorySecretStore()
    store.set(OPENROUTER_BYOK_API_KEY_SECRET, "byok-key")
    store.set(OPENROUTER_MANAGED_API_KEY_SECRET, "managed-key")

    resolution = resolve_openrouter_credentials(
        _credential_config(selected_source=OpenRouterCredentialSource.MANAGED),
        secrets=store,
    )

    assert resolution.selected_source == OpenRouterCredentialSource.MANAGED
    assert resolution.api_key == "managed-key"
    assert resolution.requires_managed_challenge is False


def test_resolve_openrouter_credentials_blocks_dual_standard_and_qq_managed_keys() -> None:
    store = InMemorySecretStore()
    store.set(OPENROUTER_MANAGED_API_KEY_SECRET, "standard-managed-key")
    store.set(OPENROUTER_MANAGED_QQ_API_KEY_SECRET, "qq-managed-key")

    with pytest.raises(ValueError, match="managed local claim conflict"):
        resolve_openrouter_credentials(
            _credential_config(
                selected_source=OpenRouterCredentialSource.MANAGED,
                active_managed_credential_ref="managed-ref-qq",
            ),
            secrets=store,
        )
    with pytest.raises(ValueError, match="managed local claim conflict"):
        resolve_openrouter_credentials(
            _credential_config(
                selected_source=OpenRouterCredentialSource.MANAGED,
                managed_credential_kind="qq",
                active_managed_credential_ref="managed-ref-qq",
            ),
            secrets=store,
        )


def test_resolve_openrouter_credentials_never_falls_back_between_managed_key_kinds() -> None:
    store = InMemorySecretStore()
    store.set(OPENROUTER_MANAGED_API_KEY_SECRET, "standard-managed-key")

    with pytest.raises(ValueError, match="managed local claim conflict"):
        resolve_openrouter_credentials(
            _credential_config(
                selected_source=OpenRouterCredentialSource.MANAGED,
                managed_credential_kind="qq",
                active_managed_credential_ref="managed-ref-qq",
            ),
            secrets=store,
            request_intent="TRANS",
        )


def test_resolve_openrouter_credentials_standard_managed_blocks_qq_key() -> None:
    store = TrackingSecretStore()
    store.set(OPENROUTER_MANAGED_QQ_API_KEY_SECRET, "qq-managed-key")

    with pytest.raises(ValueError, match="managed local claim conflict"):
        resolve_openrouter_credentials(
            _credential_config(selected_source=OpenRouterCredentialSource.MANAGED),
            secrets=store,
            request_intent="TRANS",
        )
    assert store.get_calls == [
        OPENROUTER_MANAGED_API_KEY_SECRET,
        OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
    ]


def test_resolve_openrouter_credentials_qq_managed_blocks_standard_key() -> None:
    store = TrackingSecretStore()
    store.set(OPENROUTER_MANAGED_API_KEY_SECRET, "standard-managed-key")

    with pytest.raises(ValueError, match="managed local claim conflict"):
        resolve_openrouter_credentials(
            _credential_config(
                selected_source=OpenRouterCredentialSource.MANAGED,
                managed_credential_kind="qq",
                active_managed_credential_ref="managed-ref-qq",
            ),
            secrets=store,
            request_intent="TRANS",
        )
    assert store.get_calls == [
        OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
        OPENROUTER_MANAGED_API_KEY_SECRET,
    ]


def test_resolve_openrouter_credentials_qq_key_usable_without_active_managed_ref() -> None:
    store = InMemorySecretStore()
    store.set(OPENROUTER_MANAGED_QQ_API_KEY_SECRET, "qq-managed-key")

    resolution = resolve_openrouter_credentials(
        _credential_config(
            selected_source=OpenRouterCredentialSource.MANAGED,
            managed_credential_kind="qq",
            active_managed_credential_ref=None,
        ),
        secrets=store,
        request_intent="TRANS",
    )

    assert resolution.api_key == "qq-managed-key"
    assert resolution.requires_managed_challenge is False


def test_resolve_openrouter_credentials_requires_explicit_trans_intent_before_managed_release() -> (
    None
):
    store = InMemorySecretStore()
    store.set(OPENROUTER_BYOK_API_KEY_SECRET, "byok-key")

    config = _credential_config(selected_source=OpenRouterCredentialSource.MANAGED)
    resolution = resolve_openrouter_credentials(config, secrets=store)
    trans_resolution = resolve_openrouter_credentials(
        config,
        secrets=store,
        request_intent="TRANS",
    )

    assert resolution.api_key is None
    assert resolution.requires_managed_challenge is False
    assert trans_resolution.api_key is None
    assert trans_resolution.requires_managed_challenge is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (" user-123 ", "user-123"),
        ("", None),
        ("   ", None),
        (None, None),
        (123, None),
        ("x" * OPENROUTER_MANAGED_USER_ID_MAX_LENGTH, "x" * OPENROUTER_MANAGED_USER_ID_MAX_LENGTH),
        ("x" * (OPENROUTER_MANAGED_USER_ID_MAX_LENGTH + 1), None),
    ],
)
def test_normalize_managed_openrouter_user_identifier(
    value: object,
    expected: str | None,
) -> None:
    assert normalize_managed_openrouter_user_identifier(value) == expected


def test_load_managed_openrouter_user_identifier_returns_cached_user_for_matching_installation() -> (
    None
):
    store = InMemorySecretStore()
    store.set(OPENROUTER_MANAGED_USER_ID_SECRET, " user-123 ")
    store.set(OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET, " install-123 ")

    assert (
        load_managed_openrouter_user_identifier(
            _credential_config(installation_id="install-123"),
            secrets=store,
        )
        == "user-123"
    )


@pytest.mark.parametrize(
    ("current_installation_id", "cached_installation_id", "cached_user_id"),
    [
        ("", "install-123", "user-123"),
        ("install-123", "other-install-123", "user-123"),
        ("install-123", "   ", "user-123"),
        ("install-123", "install-123", "   "),
        (
            "install-123",
            "install-123",
            "x" * (OPENROUTER_MANAGED_USER_ID_MAX_LENGTH + 1),
        ),
    ],
)
def test_load_managed_openrouter_user_identifier_returns_none_without_matching_valid_cache(
    current_installation_id: str,
    cached_installation_id: str,
    cached_user_id: str,
) -> None:
    store = InMemorySecretStore()
    store.set(OPENROUTER_MANAGED_USER_ID_SECRET, cached_user_id)
    store.set(OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET, cached_installation_id)

    assert (
        load_managed_openrouter_user_identifier(
            _credential_config(installation_id=current_installation_id),
            secrets=store,
        )
        is None
    )


def test_load_managed_openrouter_user_identifier_fails_open_when_secret_store_raises() -> None:
    class BrokenSecretStore:
        def get(self, key: str) -> str | None:
            raise RuntimeError(f"boom: {key}")

    assert (
        load_managed_openrouter_user_identifier(
            _credential_config(installation_id="install-123"),
            secrets=BrokenSecretStore(),
        )
        is None
    )


def test_clear_temporary_managed_release_state_clears_verified_snapshot_fields() -> None:
    state = _managed_state(
        release_token="release-1",
        release_token_expires_at="2026-04-08T06:00:45.000Z",
        verified_hardware_hash="hardware-hash-1",
        verified_hardware_hash_salt_version=7,
    )

    clear_temporary_managed_release_state(state)

    assert state.release_token is None
    assert state.release_token_expires_at is None
    assert state.verified_hardware_hash is None
    assert state.verified_hardware_hash_salt_version is None


@pytest.mark.parametrize(
    ("selected_source", "managed_availability"),
    [
        (OpenRouterCredentialSource.MANAGED, "not_eligible"),
        (OpenRouterCredentialSource.BYOK, "unavailable"),
    ],
)
def test_handle_managed_availability_stops_flow_without_switching_sources(
    selected_source: OpenRouterCredentialSource,
    managed_availability: str,
) -> None:
    state = _managed_state(
        release_token="release-1",
        release_token_expires_at="2026-04-08T06:00:45.000Z",
    )

    result = handle_managed_availability(
        state,
        managed_availability=managed_availability,
        selected_source=selected_source,
    )

    assert result.action == OpenRouterManagedRecoveryAction.STOP
    assert result.reason == managed_availability
    assert result.selected_source == selected_source
    assert result.managed_availability == managed_availability
    assert state.release_token is None
    assert state.release_token_expires_at is None


@pytest.mark.parametrize(
    ("selected_source", "error_code"),
    [
        (OpenRouterCredentialSource.MANAGED, "challenge_expired"),
        (OpenRouterCredentialSource.BYOK, "security_fail"),
    ],
)
def test_handle_managed_release_error_restarts_from_challenge_without_switching_sources(
    selected_source: OpenRouterCredentialSource,
    error_code: str,
) -> None:
    state = _managed_state(
        release_token="release-1",
        release_token_expires_at="2026-04-08T06:00:45.000Z",
    )

    result = handle_managed_release_error(
        state,
        error_code=error_code,
        selected_source=selected_source,
    )

    assert result.action == OpenRouterManagedRecoveryAction.RESTART_CHALLENGE
    assert result.reason == error_code
    assert result.selected_source == selected_source
    assert result.managed_availability is None
    assert state.release_token is None
    assert state.release_token_expires_at is None
