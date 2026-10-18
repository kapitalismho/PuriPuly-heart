from __future__ import annotations

import json
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


def test_loading_and_writing_old_referral_settings_preserves_managed_authority(tmp_path) -> None:
    from puripuly_heart.config.settings_vnext import compat

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
