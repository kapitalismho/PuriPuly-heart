from __future__ import annotations

import json

import pytest

from puripuly_heart.config.settings import (
    DEFAULT_OPENROUTER_BROKER_BASE_URL,
    SETTINGS_SCHEMA_VERSION,
    AppSettings,
    OpenRouterCredentialSource,
    _migrate_settings_dict,
    from_dict,
    load_settings,
    to_dict,
)
from tests.config.settings_vnext_test_helpers import legacy_projected_settings_file

PRE_MANAGED_HANDOFF_SETTINGS_SCHEMA_VERSION = 15


def test_managed_identity_settings_round_trip() -> None:
    settings = AppSettings()
    settings.managed_identity.installation_id = "01961ad7-a7c1-7000-8000-0123456789ab"
    settings.managed_identity.release_token = "release-1"
    settings.managed_identity.release_token_expires_at = "2026-04-08T06:00:45.000Z"
    settings.managed_identity.verified_hardware_hash = "hardware-hash-1"
    settings.managed_identity.verified_hardware_hash_salt_version = 7

    restored = from_dict(to_dict(settings))

    assert restored.managed_identity == settings.managed_identity


def test_managed_identity_settings_round_trip_includes_handoff_fields() -> None:
    settings = AppSettings()
    settings.managed_identity.installation_id = "01961ad7-a7c1-7000-8000-0123456789ab"
    settings.managed_identity.active_managed_credential_ref = "hash_123"
    settings.managed_identity.active_managed_expires_at = "2026-10-17T12:34:56Z"
    settings.managed_identity.founder_letter_seen_credential_ref = "hash_123"

    restored = from_dict(to_dict(settings))

    assert restored.managed_identity.active_managed_credential_ref == "hash_123"
    assert restored.managed_identity.active_managed_expires_at == "2026-10-17T12:34:56Z"
    assert restored.managed_identity.founder_letter_seen_credential_ref == "hash_123"


def test_managed_identity_referral_id_defaults_to_none_and_round_trips_uppercase() -> None:
    settings = AppSettings()

    assert settings.managed_identity.referral_id is None

    default_payload = to_dict(settings)
    assert default_payload["managed_identity"]["referral_id"] is None

    settings.managed_identity.referral_id = " 7kq9m2 "

    restored = from_dict(to_dict(settings))

    assert restored.managed_identity.referral_id == "7KQ9M2"
    assert to_dict(restored)["managed_identity"]["referral_id"] == "7KQ9M2"


def test_managed_identity_settings_do_not_persist_talk_together_pass_status() -> None:
    settings = AppSettings()
    settings.managed_identity.referral_id = "7KQ9M2"

    serialized = to_dict(settings)
    managed_identity = serialized["managed_identity"]

    assert managed_identity["referral_id"] == "7KQ9M2"
    assert "talk_together_pass" not in managed_identity
    assert "invite_count" not in managed_identity
    assert "invite_limit" not in managed_identity

    loaded = from_dict(
        {
            **serialized,
            "managed_identity": {
                **managed_identity,
                "talk_together_pass": {"pass_id": "7KQ9M2", "invite_count": 1},
                "invite_count": 1,
                "invite_limit": 5,
            },
        }
    )

    assert loaded.managed_identity.referral_id == "7KQ9M2"
    assert not hasattr(loaded.managed_identity, "talk_together_pass")
    assert not hasattr(loaded.managed_identity, "invite_count")
    assert not hasattr(loaded.managed_identity, "invite_limit")


@pytest.mark.parametrize(
    ("persisted_value", "expected"),
    [
        ("7kq9m2", "7KQ9M2"),
        ("  7KQ9M2  ", "7KQ9M2"),
        ("ABCDE", None),
        ("ABCDEFG", None),
        ("7KQ0M2", None),
        ("7KQOM2", None),
        ("7KQ1M2", None),
        ("7KQIM2", None),
        ("7KQLM2", None),
        ("", None),
        ("   ", None),
        (123456, None),
    ],
)
def test_load_settings_migrates_v22_referral_id_values(tmp_path, persisted_value, expected) -> None:
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy["settings_version"] = 22
    legacy["managed_identity"]["referral_id"] = persisted_value
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert SETTINGS_SCHEMA_VERSION == 25
    assert loaded.settings_version == SETTINGS_SCHEMA_VERSION
    assert loaded.managed_identity.referral_id == expected
    assert persisted["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert persisted["managed_identity"]["referral_id"] == expected


def test_openrouter_selected_source_round_trip() -> None:
    settings = AppSettings()
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED

    restored = from_dict(to_dict(settings))

    assert restored.openrouter.selected_source == OpenRouterCredentialSource.MANAGED


def test_openrouter_broker_base_url_round_trip() -> None:
    settings = AppSettings()
    settings.openrouter.broker_base_url = "https://broker.example.test"

    restored = from_dict(to_dict(settings))

    assert restored.openrouter.broker_base_url == "https://broker.example.test"


def test_load_settings_backfills_managed_identity_defaults(tmp_path) -> None:
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy["settings_version"] = SETTINGS_SCHEMA_VERSION - 1
    legacy.pop("managed_identity", None)
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.managed_identity.installation_id == ""
    assert loaded.managed_identity.release_token is None
    assert loaded.managed_identity.release_token_expires_at is None
    assert loaded.managed_identity.verified_hardware_hash is None
    assert loaded.managed_identity.verified_hardware_hash_salt_version is None
    assert persisted["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert persisted["managed_identity"] == {
        "installation_id": "",
        "release_token": None,
        "release_token_expires_at": None,
        "verified_hardware_hash": None,
        "verified_hardware_hash_salt_version": None,
        "active_managed_credential_ref": None,
        "active_managed_expires_at": None,
        "founder_letter_seen_credential_ref": None,
        "referral_id": None,
        "local_managed_claim_sources": [],
        "pending_delivery_ack_source": None,
        "pending_delivery_ack_delivery_id": None,
        "pending_delivery_ack_managed_credential_ref": None,
        "pending_delivery_ack_expires_at": None,
    }


def test_load_settings_backfills_openrouter_defaults(tmp_path) -> None:
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy["settings_version"] = SETTINGS_SCHEMA_VERSION - 1
    legacy["openrouter"].pop("selected_source", None)
    legacy["openrouter"].pop("broker_base_url", None)
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert loaded.openrouter.broker_base_url == DEFAULT_OPENROUTER_BROKER_BASE_URL
    assert persisted["openrouter"]["selected_source"] == OpenRouterCredentialSource.MANAGED.value
    assert persisted["openrouter"]["broker_base_url"] == DEFAULT_OPENROUTER_BROKER_BASE_URL


def test_load_settings_backfills_managed_handoff_defaults(tmp_path) -> None:
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy["settings_version"] = PRE_MANAGED_HANDOFF_SETTINGS_SCHEMA_VERSION
    legacy["managed_identity"].pop("active_managed_credential_ref", None)
    legacy["managed_identity"].pop("active_managed_expires_at", None)
    legacy["managed_identity"].pop("founder_letter_seen_credential_ref", None)
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.managed_identity.active_managed_credential_ref is None
    assert loaded.managed_identity.active_managed_expires_at is None
    assert loaded.managed_identity.founder_letter_seen_credential_ref is None
    assert persisted["managed_identity"]["active_managed_credential_ref"] is None
    assert persisted["managed_identity"]["active_managed_expires_at"] is None
    assert persisted["managed_identity"]["founder_letter_seen_credential_ref"] is None


def test_load_settings_normalizes_invalid_managed_handoff_values_to_none(tmp_path) -> None:
    path = tmp_path / "settings.json"
    current = to_dict(AppSettings())
    current["settings_version"] = SETTINGS_SCHEMA_VERSION
    current["managed_identity"]["active_managed_credential_ref"] = "   "
    current["managed_identity"]["active_managed_expires_at"] = 123
    current["managed_identity"]["founder_letter_seen_credential_ref"] = "\t"

    migrated, changed = _migrate_settings_dict(current)

    assert changed is True
    assert migrated["managed_identity"]["active_managed_credential_ref"] is None
    assert migrated["managed_identity"]["active_managed_expires_at"] is None
    assert migrated["managed_identity"]["founder_letter_seen_credential_ref"] is None

    path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.managed_identity.active_managed_credential_ref is None
    assert loaded.managed_identity.active_managed_expires_at is None
    assert loaded.managed_identity.founder_letter_seen_credential_ref is None
    assert persisted["managed_identity"]["active_managed_credential_ref"] is None
    assert persisted["managed_identity"]["active_managed_expires_at"] is None
    assert persisted["managed_identity"]["founder_letter_seen_credential_ref"] is None
