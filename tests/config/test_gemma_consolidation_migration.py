from __future__ import annotations

import json
from pathlib import Path

import pytest

from puripuly_heart.app.wiring.wiring_llm_factory import runtime_resolution_input_from_vnext
from puripuly_heart.config.runtime_resolution import resolve_llm_config
from puripuly_heart.config.settings_vnext import compat, migration, serialization
from puripuly_heart.config.settings_vnext.schema import (
    VNEXT_SETTINGS_SCHEMA_VERSION,
    AppSettingsVNext,
)


@pytest.mark.parametrize("model", ["gemma4", "gemma4_31b", "gemma4_26b_31b", "deepseek_v4_flash"])
@pytest.mark.parametrize("connection", ["managed", "openrouter"])
@pytest.mark.parametrize("alias", [None, "gemma4_31b_managed", "gemma4_byok"])
def test_v48_consolidates_active_pair_without_trusting_stale_alias(
    model: str, connection: str, alias: str | None
) -> None:
    raw = serialization.to_dict(AppSettingsVNext())
    raw["settings_version"] = 48
    raw["intent"]["ui"]["locale"] = "ja"
    raw["intent"]["translation"].update(
        model=model,
        connection=connection,
        connection_history={"gpt_6_luna": "official_byok", model: connection},
        openrouter_model="google/gemma-4-31b-it",
        openrouter_provider_routing="gemma4_31b_latency",
        openrouter_selected_source="byok" if connection == "managed" else "managed",
        openrouter_selection_alias=alias,
    )
    managed = raw["state"]["managed_connection"]
    managed.update(
        installation_id="01961ad7-a7c1-7000-8000-0123456789ab",
        release_token="kept-release",
        release_token_expires_at="2099-10-17T12:34:56Z",
        verified_hardware_hash="kept-hardware-hash",
        verified_hardware_hash_salt_version=7,
        active_managed_credential_ref="kept-managed-ref",
        active_managed_expires_at="2099-10-17T12:34:56Z",
        founder_letter_seen_credential_ref="kept-managed-ref",
        local_managed_claim_sources=("discord", "qq"),
        pending_delivery_ack_source="discord",
        pending_delivery_ack_delivery_id="kept-delivery",
        pending_delivery_ack_managed_credential_ref="kept-managed-ref",
        pending_delivery_ack_expires_at="2099-10-17T12:34:56Z",
        pending_managed_operation_id="ph-mop-v1_0123456789abcdef0123456789abcdef",
        pending_managed_operation_source="discord",
        pending_managed_operation_installation_id="01961ad7-a7c1-7000-8000-0123456789ab",
        pending_managed_operation_state="DELIVERY_PENDING",
    )
    preserved_authority = managed.copy()
    managed.update(referral_id="kept-referral", referral_source="discord")
    migrated = migration.from_dict(raw)
    result = serialization.to_dict(migrated)
    translation = result["intent"]["translation"]
    assert result["settings_version"] == VNEXT_SETTINGS_SCHEMA_VERSION
    assert result["intent"]["ui"]["locale"] == "ja"
    assert result["state"]["managed_connection"] == preserved_authority
    assert translation["connection"] == connection
    assert translation["connection_history"]["gpt_6_luna"] == "official_byok"
    target = resolve_llm_config(runtime_resolution_input_from_vnext(migrated)).primary
    assert target.provider == "openrouter"
    assert target.credential.source == ("managed" if connection == "managed" else "secret_store")
    assert target.credential.reference == (
        "openrouter:managed" if connection == "managed" else "openrouter:byok"
    )
    if model == "deepseek_v4_flash" and connection == "managed":
        assert translation["model"] == "deepseek_v4_flash_41"
        assert translation["connection_history"]["deepseek_v4_flash_41"] == "managed"
        assert "deepseek_v4_flash" not in translation["connection_history"]
        assert target.model == "deepseek/deepseek-v4.1-flash-1017"
        assert translation["openrouter_model"] == "deepseek/deepseek-v4.1-flash-1017"
        assert translation["openrouter_selection_alias"] == "deepseek_v4_flash_41_managed"
        assert translation["openrouter_selected_source"] == "managed"
        assert translation["openrouter_provider_routing"] == "deepseek_v4_flash_41_strict"
    elif model == "deepseek_v4_flash":
        assert translation["model"] == model
        assert target.model == "deepseek/deepseek-v4-flash-0731"
        assert target.models == ("deepseek/deepseek-v4-flash-0731",)
        assert translation["openrouter_model"] == "deepseek/deepseek-v4-flash-0731"
        assert translation["openrouter_selection_alias"] == "deepseek_v4_flash_byok"
        assert translation["openrouter_selected_source"] == "byok"
    else:
        assert translation["model"] == "gemma4_26b_31b"
        assert target.model == "google/gemma-4-26b-a4b-it"
        assert target.models == (
            "google/gemma-4-26b-a4b-it",
            "google/gemma-4-31b-it",
        )
        assert target.provider_routing == "gemma4_26b_31b_latency"
        assert translation["connection_history"]["gemma4_26b_31b"] == connection
        assert model not in translation["connection_history"] or model == "gemma4_26b_31b"
        assert translation["openrouter_model"] == "google/gemma-4-26b-a4b-it"
        assert translation["openrouter_provider_routing"] == "gemma4_26b_31b_latency"
        assert translation["openrouter_selected_source"] == (
            "managed" if connection == "managed" else "byok"
        )
        assert translation["openrouter_selection_alias"] == (
            f"gemma4_26b_31b_{'managed' if connection == 'managed' else 'byok'}"
        )


@pytest.mark.parametrize(
    ("previous", "history", "expected"),
    [
        ("gemma4_31b", {"gemma4_26b_31b": "openrouter", "gemma4": "managed"}, "openrouter"),
        ("gemma4_31b", {"gemma4_31b": "managed", "gemma4": "openrouter"}, "managed"),
        ("gemma4", {"gemma4_31b": "managed", "gemma4": "openrouter"}, "openrouter"),
        (None, {"gemma4_31b": "managed", "gemma4": "openrouter"}, "openrouter"),
        (None, {"gemma4_31b": "managed", "gemma4": "invalid"}, "managed"),
        (None, {"gemma4_26b_31b": "invalid", "gemma4": "invalid"}, "managed"),
        ("gemma4_31b", {}, "managed"),
    ],
)
def test_dormant_gemma_history_precedence_and_restoration(
    previous: str | None, history: dict[str, str], expected: str
) -> None:
    raw = serialization.to_dict(AppSettingsVNext())
    raw["settings_version"] = 48
    raw["intent"]["translation"].update(
        model="custom_http",
        connection="custom_http",
        previous_llm_model=previous,
        connection_history={**history, "gpt_6_luna": "openrouter"},
        openrouter_selection_alias="gemma4_31b_byok",
    )
    translated = serialization.to_dict(migration.from_dict(raw))["intent"]["translation"]
    assert translated["model"] == "custom_http"
    assert translated["connection"] == "custom_http"
    assert translated["previous_llm_model"] == ("gemma4_26b_31b" if previous else None)
    assert translated["connection_history"] == {
        "gpt_6_luna": "openrouter",
        "gemma4_26b_31b": expected,
    }
    assert translated["openrouter_selection_alias"] == "gemma4_26b_31b_byok"


@pytest.mark.parametrize(
    "version,model,connection",
    [(31, "gemma4", "openrouter"), (41, "gemma4_31b_cerebras", "official_byok")],
)
def test_older_chain_reaches_combined(version: int, model: str, connection: str) -> None:
    raw = serialization.to_dict(AppSettingsVNext())
    raw["settings_version"] = version
    raw["intent"]["translation"].update(
        model=model, connection=connection, connection_history={model: connection}
    )
    translation = serialization.to_dict(migration.from_dict(raw))["intent"]["translation"]
    assert translation["model"] == "gemma4_26b_31b"
    assert translation["connection"] == "openrouter"
    assert translation["connection_history"]["gemma4_26b_31b"] == "openrouter"
    assert translation["openrouter_selection_alias"] == "gemma4_26b_31b_byok"


@pytest.mark.parametrize("existing_combined", [None, "managed"])
def test_v31_dormant_previous_history_precedes_synthesized_combined(
    existing_combined: str | None,
) -> None:
    raw = serialization.to_dict(AppSettingsVNext())
    raw["settings_version"] = 31
    history = {"gemma4": "managed", "gemma4_31b": "openrouter"}
    if existing_combined is not None:
        history["gemma4_26b_31b"] = existing_combined
    raw["intent"]["translation"].update(
        model="custom_http",
        connection="custom_http",
        previous_llm_model="gemma4_31b",
        connection_history=history,
    )

    translation = migration.from_dict(raw).intent.translation

    assert translation.previous_llm_model == "gemma4_26b_31b"
    assert translation.connection_history == {"gemma4_26b_31b": existing_combined or "openrouter"}


def test_v48_loader_keeps_original_then_reloads_without_new_backup(tmp_path: Path) -> None:
    raw = serialization.to_dict(AppSettingsVNext())
    raw["settings_version"] = 48
    raw["intent"]["translation"].update(
        model="gemma4_31b",
        connection="openrouter",
        openrouter_selection_alias=None,
        connection_history={"gemma4_31b": "managed"},
    )
    raw["state"]["provider_verification"]["openrouter"] = {"status": "unknown"}
    path = tmp_path / "settings.json"
    original = json.dumps(raw, ensure_ascii=False, indent=2).encode("utf-8")
    path.write_bytes(original)
    first = compat.load_vnext_settings(path)
    assert first.ok and first.migrated and first.settings is not None
    assert first.backup_path is not None and first.backup_path.read_bytes() == original
    persisted = path.read_bytes()
    assert f'"settings_version": {VNEXT_SETTINGS_SCHEMA_VERSION}'.encode() in persisted
    assert first.settings.intent.translation.connection == "openrouter"
    assert first.settings.intent.translation.openrouter_selected_source == "byok"
    assert first.settings.state.provider_verification.openrouter.status == "unknown"
    second = compat.load_vnext_settings(path)
    assert second.ok and not second.migrated and second.backup_path is None
    assert path.read_bytes() == persisted
    assert list(tmp_path.glob("*.bak")) == [first.backup_path]
    assert second.settings == first.settings


@pytest.mark.parametrize("secret", [None, "existing-openrouter-key"])
def test_v48_byok_missing_or_present_secret_does_not_change_connection_or_auth(
    tmp_path: Path, secret: str | None
) -> None:
    raw = serialization.to_dict(AppSettingsVNext())
    raw["settings_version"] = 48
    raw["intent"]["translation"].update(
        model="gemma4_31b",
        connection="openrouter",
        openrouter_selected_source="managed",
        openrouter_selection_alias="gemma4_31b_managed",
        connection_history={"gemma4_31b": "managed"},
    )
    raw["state"]["managed_connection"]["active_managed_credential_ref"] = "account-ref"
    evidence = {
        "status": "verified",
        "provider": "openrouter",
        "secret_key": "openrouter_api_key",
        "secret_revision": "old-revision",
        "secret_fingerprint": "sha256:existing",
        "verifier_context": {"flow": "settings.verify_api_key"},
        "verifier_evidence": {"verifier": "openrouter"},
    }
    if secret is not None:
        raw["state"]["provider_verification"]["openrouter"] = evidence
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    secret_path = tmp_path / "secrets.json"
    secret_bytes = json.dumps({"openrouter_api_key": secret}).encode("utf-8")
    secret_path.write_bytes(secret_bytes)

    loaded = compat.load_vnext_settings(path)

    assert loaded.ok and loaded.settings is not None
    assert loaded.settings.intent.translation.model == "gemma4_26b_31b"
    assert loaded.settings.intent.translation.connection == "openrouter"
    assert loaded.settings.intent.translation.openrouter_selected_source == "byok"
    target = resolve_llm_config(runtime_resolution_input_from_vnext(loaded.settings)).primary
    assert target.models == ("google/gemma-4-26b-a4b-it", "google/gemma-4-31b-it")
    assert target.credential.source == "secret_store"
    assert target.credential.reference == "openrouter:byok"
    assert loaded.settings.state.managed_connection.active_managed_credential_ref == "account-ref"
    expected_evidence = evidence if secret is not None else {"status": "unknown"}
    assert (
        json.loads(path.read_text(encoding="utf-8"))["state"]["provider_verification"]["openrouter"]
        == expected_evidence
    )
    assert secret_path.read_bytes() == secret_bytes
