from __future__ import annotations

from dataclasses import replace

from puripuly_heart.config.provider_values import OpenRouterCredentialSource
from puripuly_heart.config.settings_vnext import serialization
from puripuly_heart.config.settings_vnext.schema import (
    DEFAULT_OPENROUTER_BROKER_BASE_URL,
    AppSettingsVNext,
)


def _with_managed_connection(settings: AppSettingsVNext, **updates: object) -> AppSettingsVNext:
    return replace(
        settings,
        state=replace(
            settings.state,
            managed_connection=replace(settings.state.managed_connection, **updates),
        ),
    )


def test_managed_identity_settings_round_trip() -> None:
    settings = _with_managed_connection(
        AppSettingsVNext(),
        installation_id="01961ad7-a7c1-7000-8000-0123456789ab",
        release_token="release-1",
        release_token_expires_at="2026-04-08T06:00:45.000Z",
        verified_hardware_hash="hardware-hash-1",
        verified_hardware_hash_salt_version=7,
    )

    restored = serialization.from_dict(serialization.to_dict(settings))

    assert restored.state.managed_connection == settings.state.managed_connection


def test_managed_identity_settings_round_trip_includes_handoff_fields() -> None:
    settings = _with_managed_connection(
        AppSettingsVNext(),
        installation_id="01961ad7-a7c1-7000-8000-0123456789ab",
        active_managed_credential_ref="hash_123",
        active_managed_expires_at="2026-10-17T12:34:56Z",
        founder_letter_seen_credential_ref="hash_123",
    )

    restored = serialization.from_dict(serialization.to_dict(settings))

    assert restored.state.managed_connection.active_managed_credential_ref == "hash_123"
    assert restored.state.managed_connection.active_managed_expires_at == "2026-10-17T12:34:56Z"
    assert restored.state.managed_connection.founder_letter_seen_credential_ref == "hash_123"


def test_managed_identity_referral_id_defaults_to_none_and_round_trips() -> None:
    settings = AppSettingsVNext()

    assert settings.state.managed_connection.referral_id is None
    assert settings.state.managed_connection.referral_source is None

    default_payload = serialization.to_dict(settings)
    assert default_payload["state"]["managed_connection"]["referral_id"] is None
    assert default_payload["state"]["managed_connection"]["referral_source"] is None

    settings = _with_managed_connection(settings, referral_id="7KQ9M2", referral_source="qq")
    restored = serialization.from_dict(serialization.to_dict(settings))

    assert restored.state.managed_connection.referral_id == "7KQ9M2"
    assert restored.state.managed_connection.referral_source == "qq"
    payload = serialization.to_dict(restored)["state"]["managed_connection"]
    assert payload["referral_id"] == "7KQ9M2"
    assert payload["referral_source"] == "qq"


def test_managed_identity_referral_source_defaults_to_discord_when_referral_id_is_set() -> None:
    settings = _with_managed_connection(AppSettingsVNext(), referral_id="7KQ9M2")

    assert settings.state.managed_connection.referral_source == "discord"


def test_managed_identity_settings_do_not_persist_talk_together_pass_status() -> None:
    settings = _with_managed_connection(AppSettingsVNext(), referral_id="7KQ9M2")

    serialized = serialization.to_dict(settings)
    managed_identity = serialized["state"]["managed_connection"]

    assert managed_identity["referral_id"] == "7KQ9M2"
    assert "talk_together_pass" not in managed_identity
    assert "invite_count" not in managed_identity
    assert "invite_limit" not in managed_identity

    loaded = serialization.from_dict(
        {
            **serialized,
            "state": {
                **serialized["state"],
                "managed_connection": {
                    **managed_identity,
                    "talk_together_pass": {"pass_id": "7KQ9M2", "invite_count": 1},
                    "invite_count": 1,
                    "invite_limit": 3,
                },
            },
        }
    )

    assert loaded.state.managed_connection.referral_id == "7KQ9M2"
    assert not hasattr(loaded.state.managed_connection, "talk_together_pass")
    assert not hasattr(loaded.state.managed_connection, "invite_count")
    assert not hasattr(loaded.state.managed_connection, "invite_limit")


def test_openrouter_selected_source_round_trip() -> None:
    current = AppSettingsVNext()
    settings = replace(
        current,
        intent=replace(
            current.intent,
            translation=replace(
                current.intent.translation,
                openrouter_selected_source=OpenRouterCredentialSource.MANAGED.value,
            ),
        ),
    )

    restored = serialization.from_dict(serialization.to_dict(settings))

    assert (
        restored.intent.translation.openrouter_selected_source
        == OpenRouterCredentialSource.MANAGED.value
    )


def test_openrouter_broker_base_url_round_trip() -> None:
    current = AppSettingsVNext()
    settings = replace(
        current,
        intent=replace(
            current.intent,
            translation=replace(
                current.intent.translation,
                openrouter_broker_base_url="https://broker.example.test",
            ),
        ),
    )

    restored = serialization.from_dict(serialization.to_dict(settings))

    assert restored.intent.translation.openrouter_broker_base_url == "https://broker.example.test"


def test_openrouter_broker_base_url_default_is_production_broker() -> None:
    settings = AppSettingsVNext()

    assert (
        settings.intent.translation.openrouter_broker_base_url == DEFAULT_OPENROUTER_BROKER_BASE_URL
    )
