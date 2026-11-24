from __future__ import annotations

import json
from dataclasses import replace

import pytest

from puripuly_heart.config.settings_vnext import compat, serialization
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext

_LEGACY_BROKER_ORIGIN = "https://puripuly-heart-broker.kapitalismho.workers.dev"


def _with_managed_connection(settings: AppSettingsVNext, **updates: object) -> AppSettingsVNext:
    return replace(
        settings,
        state=replace(
            settings.state,
            managed_connection=replace(settings.state.managed_connection, **updates),
        ),
    )


def test_loading_and_writing_old_referral_settings_preserves_managed_authority(tmp_path) -> None:
    settings = _with_managed_connection(
        AppSettingsVNext(),
        installation_id="01961ad7-a7c1-7000-8000-0123456789ab",
        release_token="release-1",
        release_token_expires_at="2026-12-08T06:00:45.000Z",
        verified_hardware_hash="hardware-hash-1",
        verified_hardware_hash_salt_version=7,
        active_managed_credential_ref="managed-ref-qq",
        active_managed_expires_at="2026-12-17T12:34:56Z",
        founder_letter_seen_credential_ref="managed-ref-qq",
        local_managed_claim_sources=("discord", "qq"),
        pending_delivery_ack_source="qq",
        pending_delivery_ack_delivery_id="delivery-qq",
        pending_delivery_ack_managed_credential_ref="managed-ref-qq",
        pending_delivery_ack_expires_at="2026-12-17T12:34:56Z",
        pending_managed_operation_id="ph-mop-v1_0123456789abcdef0123456789abcdef",
        pending_managed_operation_source="qq",
        pending_managed_operation_installation_id="01961ad7-a7c1-7000-8000-0123456789ab",
        pending_managed_operation_state="DELIVERY_PENDING",
    )
    payload = serialization.to_dict(settings)
    payload["settings_version"] = 52
    obsolete = {
        "referral_id": "7KQ9M2",
        "referral_source": "qq",
        "talk_together_pass": {"pass_id": "7KQ9M2", "invite_count": 1},
        "invite_count": 1,
        "invite_limit": 3,
    }
    payload["state"]["managed_connection"].update(obsolete)
    payload["state"]["managed_connection"]["unrelated_extension"] = {"retained": True}
    payload["managed_identity"] = {**obsolete, "unrelated_extension": "identity"}
    payload["managed_connection"] = {**obsolete, "unrelated_extension": "connection"}
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = compat.load_vnext_settings(path)

    assert loaded.settings is not None
    assert loaded.settings.state.managed_connection == settings.state.managed_connection
    saved = compat.save_vnext_settings(path, loaded.settings)
    assert saved.ok
    written = json.loads(path.read_text(encoding="utf-8"))
    managed = written["state"]["managed_connection"]
    assert managed == {
        **json.loads(json.dumps(serialization.to_dict(settings)))["state"]["managed_connection"],
        "unrelated_extension": {"retained": True},
    }
    assert written["managed_identity"] == {"unrelated_extension": "identity"}
    assert written["managed_connection"] == {"unrelated_extension": "connection"}
    assert not set(obsolete).intersection(managed)
    reloaded = compat.load_vnext_settings(path)
    assert reloaded.settings is not None
    assert reloaded.settings.state.managed_connection == settings.state.managed_connection


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


@pytest.mark.parametrize(
    ("saved_origin", "migrates_default"),
    [
        (_LEGACY_BROKER_ORIGIN, True),
        (f"{_LEGACY_BROKER_ORIGIN}///", True),
        (_LEGACY_BROKER_ORIGIN.upper(), False),
        (f"{_LEGACY_BROKER_ORIGIN}:443", False),
        (f" {_LEGACY_BROKER_ORIGIN}", False),
        (f"{_LEGACY_BROKER_ORIGIN} ", False),
        ("https://broker.example.test", False),
        (f"{_LEGACY_BROKER_ORIGIN}/api", False),
        (f"{_LEGACY_BROKER_ORIGIN}:8443", False),
        (f"{_LEGACY_BROKER_ORIGIN}.example.test", False),
        (f"{_LEGACY_BROKER_ORIGIN}?custom", False),
        (f"{_LEGACY_BROKER_ORIGIN}#custom", False),
        (_LEGACY_BROKER_ORIGIN.replace("://", "://user@", 1), False),
        (_LEGACY_BROKER_ORIGIN.replace("https:", "http:", 1), False),
    ],
)
def test_saved_broker_origin_migrates_only_recognized_legacy_defaults(
    tmp_path, saved_origin, migrates_default
) -> None:
    settings = _with_managed_connection(
        AppSettingsVNext(),
        installation_id="01961ad7-a7c1-7000-8000-0123456789ab",
        active_managed_credential_ref="synthetic-credential-ref",
        active_managed_expires_at="2026-12-17T12:34:56Z",
    )
    raw = serialization.to_persisted_dict(settings)
    raw["settings_version"] = 58
    raw["intent"]["translation"]["openrouter_broker_base_url"] = saved_origin
    path = tmp_path / "settings.json"
    original = json.dumps(raw).encode("utf-8")
    path.write_bytes(original)

    loaded = compat.load_vnext_settings(path)

    expected = "https://puripuly-broker.salee.sarl" if migrates_default else saved_origin
    assert loaded.migrated and loaded.backup_path is not None
    assert loaded.backup_path.read_bytes() == original
    assert loaded.settings is not None
    assert loaded.settings.intent.translation.openrouter_broker_base_url == expected
    assert loaded.settings.state == settings.state
    saved = json.loads(path.read_bytes())
    assert saved["intent"]["translation"]["openrouter_broker_base_url"] == expected
    assert saved["state"] == json.loads(original)["state"]
    reloaded = compat.load_vnext_settings(path)
    assert reloaded.ok and not reloaded.migrated
    assert reloaded.settings == loaded.settings


def test_current_settings_can_explicitly_use_the_legacy_broker_origin(tmp_path) -> None:
    path = tmp_path / "settings.json"
    raw = serialization.to_persisted_dict(AppSettingsVNext())
    raw["intent"]["translation"]["openrouter_broker_base_url"] = _LEGACY_BROKER_ORIGIN
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = compat.load_vnext_settings(path)

    assert loaded.ok and not loaded.migrated
    assert loaded.settings is not None
    assert loaded.settings.intent.translation.openrouter_broker_base_url == _LEGACY_BROKER_ORIGIN
