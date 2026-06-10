from __future__ import annotations

import logging

import pytest

from puripuly_heart.config.settings import (
    AppSettings,
    OpenRouterCredentialSource,
    TranslationConnection,
)
from puripuly_heart.core.openrouter_credentials import (
    OPENROUTER_BYOK_API_KEY_SECRET,
    OPENROUTER_MANAGED_API_KEY_SECRET,
    OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
    OPENROUTER_MANAGED_USER_ID_MAX_LENGTH,
    OPENROUTER_MANAGED_USER_ID_SECRET,
    OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET,
    OpenRouterManagedRecoveryAction,
    clear_temporary_managed_release_state,
    handle_managed_availability,
    handle_managed_release_error,
    load_managed_openrouter_user_identifier,
    normalize_managed_openrouter_user_identifier,
    resolve_openrouter_credentials,
)
from puripuly_heart.core.storage.secrets import InMemorySecretStore


class ReadGuardSecretStore(InMemorySecretStore):
    def __init__(self, *, raise_on_get: set[str]) -> None:
        super().__init__()
        self.raise_on_get = raise_on_get
        self.get_calls: list[str] = []

    def get(self, key: str) -> str | None:
        self.get_calls.append(key)
        if key in self.raise_on_get:
            raise RuntimeError(f"unexpected secret read: {key}")
        return super().get(key)


def test_resolve_openrouter_credentials_respects_explicit_none_selection_even_with_stored_keys() -> (
    None
):
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.NONE
    store = InMemorySecretStore()
    store.set(OPENROUTER_BYOK_API_KEY_SECRET, "byok-key")
    store.set(OPENROUTER_MANAGED_API_KEY_SECRET, "managed-key")

    resolution = resolve_openrouter_credentials(settings, secrets=store)

    assert resolution.selected_source == OpenRouterCredentialSource.NONE
    assert resolution.api_key is None
    assert resolution.requires_managed_challenge is False


def test_resolve_openrouter_credentials_uses_selected_byok_key_without_managed_fallback() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    store = InMemorySecretStore()
    store.set(OPENROUTER_BYOK_API_KEY_SECRET, "byok-key")
    store.set(OPENROUTER_MANAGED_API_KEY_SECRET, "managed-key")

    resolution = resolve_openrouter_credentials(settings, secrets=store)

    assert resolution.selected_source == OpenRouterCredentialSource.BYOK
    assert resolution.api_key == "byok-key"
    assert resolution.requires_managed_challenge is False


def test_resolve_openrouter_credentials_uses_selected_managed_key_without_byok_fallback() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    store = InMemorySecretStore()
    store.set(OPENROUTER_BYOK_API_KEY_SECRET, "byok-key")
    store.set(OPENROUTER_MANAGED_API_KEY_SECRET, "managed-key")

    resolution = resolve_openrouter_credentials(settings, secrets=store)

    assert resolution.selected_source == OpenRouterCredentialSource.MANAGED
    assert resolution.api_key == "managed-key"
    assert resolution.requires_managed_challenge is False


def test_resolve_openrouter_credentials_requires_explicit_trans_intent_before_managed_release() -> (
    None
):
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    store = InMemorySecretStore()
    store.set(OPENROUTER_BYOK_API_KEY_SECRET, "byok-key")

    resolution = resolve_openrouter_credentials(settings, secrets=store)
    trans_resolution = resolve_openrouter_credentials(
        settings,
        secrets=store,
        request_intent="TRANS",
    )

    assert resolution.api_key is None
    assert resolution.requires_managed_challenge is False
    assert trans_resolution.api_key is None
    assert trans_resolution.requires_managed_challenge is True


def test_resolve_openrouter_credentials_managed_china_uses_only_qq_key() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.translation.connection = TranslationConnection.MANAGED_CHINA
    store = InMemorySecretStore()
    store.set(OPENROUTER_MANAGED_API_KEY_SECRET, "discord-managed-key")
    store.set(OPENROUTER_MANAGED_QQ_API_KEY_SECRET, " qq-managed-key ")

    resolution = resolve_openrouter_credentials(settings, secrets=store)

    assert resolution.selected_source == OpenRouterCredentialSource.MANAGED
    assert resolution.api_key == "qq-managed-key"
    assert resolution.requires_managed_challenge is False


def test_resolve_openrouter_credentials_managed_china_ignores_discord_key() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.translation.connection = TranslationConnection.MANAGED_CHINA
    store = InMemorySecretStore()
    store.set(OPENROUTER_MANAGED_API_KEY_SECRET, "discord-managed-key")

    resolution = resolve_openrouter_credentials(
        settings,
        secrets=store,
        request_intent="TRANS",
    )

    assert resolution.selected_source == OpenRouterCredentialSource.MANAGED
    assert resolution.api_key is None
    assert resolution.requires_managed_challenge is True


def test_resolve_openrouter_credentials_non_china_managed_uses_only_discord_key() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.translation.connection = TranslationConnection.MANAGED
    store = InMemorySecretStore()
    store.set(OPENROUTER_MANAGED_API_KEY_SECRET, " discord-managed-key ")
    store.set(OPENROUTER_MANAGED_QQ_API_KEY_SECRET, "qq-managed-key")

    resolution = resolve_openrouter_credentials(settings, secrets=store)

    assert resolution.selected_source == OpenRouterCredentialSource.MANAGED
    assert resolution.api_key == "discord-managed-key"
    assert resolution.requires_managed_challenge is False


def test_resolve_openrouter_credentials_non_china_managed_ignores_qq_key() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.translation.connection = TranslationConnection.MANAGED
    store = InMemorySecretStore()
    store.set(OPENROUTER_MANAGED_QQ_API_KEY_SECRET, "qq-managed-key")

    resolution = resolve_openrouter_credentials(
        settings,
        secrets=store,
        request_intent="TRANS",
    )

    assert resolution.selected_source == OpenRouterCredentialSource.MANAGED
    assert resolution.api_key is None
    assert resolution.requires_managed_challenge is True


def test_resolve_openrouter_credentials_whitespace_qq_key_is_unavailable() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.translation.connection = TranslationConnection.MANAGED_CHINA
    store = InMemorySecretStore()
    store.set(OPENROUTER_MANAGED_QQ_API_KEY_SECRET, "   ")

    resolution = resolve_openrouter_credentials(
        settings,
        secrets=store,
        request_intent="TRANS",
    )

    assert resolution.selected_source == OpenRouterCredentialSource.MANAGED
    assert resolution.api_key is None
    assert resolution.requires_managed_challenge is True


def test_resolve_openrouter_credentials_managed_china_does_not_read_discord_secret(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.translation.connection = TranslationConnection.MANAGED_CHINA
    store = ReadGuardSecretStore(raise_on_get={OPENROUTER_MANAGED_API_KEY_SECRET})
    store.set(OPENROUTER_MANAGED_QQ_API_KEY_SECRET, "qq-managed-key")
    caplog.set_level(logging.INFO, logger="puripuly_heart.core.openrouter_credentials")

    resolution = resolve_openrouter_credentials(settings, secrets=store)

    assert resolution.api_key == "qq-managed-key"
    assert resolution.requires_managed_challenge is False
    assert store.get_calls == [OPENROUTER_MANAGED_QQ_API_KEY_SECRET]
    assert "discord_key" not in caplog.text
    assert "ignored" not in caplog.text


def test_resolve_openrouter_credentials_non_china_managed_does_not_read_qq_secret(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.translation.connection = TranslationConnection.MANAGED
    store = ReadGuardSecretStore(raise_on_get={OPENROUTER_MANAGED_QQ_API_KEY_SECRET})
    store.set(OPENROUTER_MANAGED_API_KEY_SECRET, "discord-managed-key")
    caplog.set_level(logging.INFO, logger="puripuly_heart.core.openrouter_credentials")

    resolution = resolve_openrouter_credentials(settings, secrets=store)

    assert resolution.api_key == "discord-managed-key"
    assert resolution.requires_managed_challenge is False
    assert store.get_calls == [OPENROUTER_MANAGED_API_KEY_SECRET]
    assert "qq_key" not in caplog.text
    assert "ignored" not in caplog.text


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
    settings = AppSettings()
    settings.managed_identity.installation_id = "install-123"
    store = InMemorySecretStore()
    store.set(OPENROUTER_MANAGED_USER_ID_SECRET, " user-123 ")
    store.set(OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET, " install-123 ")

    assert load_managed_openrouter_user_identifier(settings, secrets=store) == "user-123"


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
    settings = AppSettings()
    settings.managed_identity.installation_id = current_installation_id
    store = InMemorySecretStore()
    store.set(OPENROUTER_MANAGED_USER_ID_SECRET, cached_user_id)
    store.set(OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET, cached_installation_id)

    assert load_managed_openrouter_user_identifier(settings, secrets=store) is None


def test_load_managed_openrouter_user_identifier_fails_open_when_secret_store_raises() -> None:
    class BrokenSecretStore:
        def get(self, key: str) -> str | None:
            raise RuntimeError(f"boom: {key}")

    settings = AppSettings()
    settings.managed_identity.installation_id = "install-123"

    assert load_managed_openrouter_user_identifier(settings, secrets=BrokenSecretStore()) is None


def test_clear_temporary_managed_release_state_clears_verified_snapshot_fields() -> None:
    settings = AppSettings()
    settings.managed_identity.release_token = "release-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:00:45.000Z"
    settings.managed_identity.verified_hardware_hash = "hardware-hash-1"
    settings.managed_identity.verified_hardware_hash_salt_version = 7

    clear_temporary_managed_release_state(settings)

    assert settings.managed_identity.release_token is None
    assert settings.managed_identity.release_token_expires_at is None
    assert settings.managed_identity.verified_hardware_hash is None
    assert settings.managed_identity.verified_hardware_hash_salt_version is None


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
    settings = AppSettings()
    settings.openrouter.selected_source = selected_source
    settings.managed_identity.release_token = "release-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:00:45.000Z"

    result = handle_managed_availability(
        settings,
        managed_availability=managed_availability,
    )

    assert result.action == OpenRouterManagedRecoveryAction.STOP
    assert result.reason == managed_availability
    assert result.selected_source == selected_source
    assert result.managed_availability == managed_availability
    assert settings.openrouter.selected_source == selected_source
    assert settings.managed_identity.release_token is None
    assert settings.managed_identity.release_token_expires_at is None


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
    settings = AppSettings()
    settings.openrouter.selected_source = selected_source
    settings.managed_identity.release_token = "release-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:00:45.000Z"

    result = handle_managed_release_error(settings, error_code=error_code)

    assert result.action == OpenRouterManagedRecoveryAction.RESTART_CHALLENGE
    assert result.reason == error_code
    assert result.selected_source == selected_source
    assert result.managed_availability is None
    assert settings.openrouter.selected_source == selected_source
    assert settings.managed_identity.release_token is None
    assert settings.managed_identity.release_token_expires_at is None
