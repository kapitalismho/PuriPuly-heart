from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from puripuly_heart.config.settings import (
    SETTINGS_SCHEMA_VERSION,
    AppSettings,
    from_dict,
    load_settings,
    load_settings_with_result,
    new_settings_for_first_run,
    save_settings,
    to_dict,
)
from puripuly_heart.config.settings_vnext.schema import (
    VNEXT_SETTINGS_SCHEMA_VERSION,
    AppSettingsVNext,
    CaptureTargetIntent,
    PersistedOperationalState,
    ProcessCaptureTargetIntent,
    ProviderVerificationEntry,
    ProviderVerificationState,
    TelemetryOperationalState,
    TranslationFallbackIntent,
    with_capture_target,
    with_telemetry_consent,
)
from tests.config.settings_migration_fixtures import (
    legacy_compatibility_settings_fixture,
    maximal_v24_settings_fixture,
)

PROVIDER_VERIFICATION_FIELDS = (
    "deepgram",
    "soniox",
    "google",
    "openrouter",
    "deepseek",
    "cerebras",
    "alibaba_beijing",
    "alibaba_singapore",
)
PROVIDER_VERIFICATION_SECRET_KEYS = {
    "deepgram": "deepgram_api_key",
    "soniox": "soniox_api_key",
    "google": "google_api_key",
    "openrouter": "openrouter_api_key",
    "deepseek": "deepseek_api_key",
    "cerebras": "cerebras_api_key",
    "alibaba_beijing": "alibaba_api_key_beijing",
    "alibaba_singapore": "alibaba_api_key_singapore",
}


def _load_module(name: str) -> ModuleType:
    try:
        return import_module(name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"{name} should import: {exc}")


def _migration() -> ModuleType:
    return _load_module("puripuly_heart.config.settings_vnext.migration")


def _serialization() -> ModuleType:
    return _load_module("puripuly_heart.config.settings_vnext.serialization")


def _compat() -> ModuleType:
    return _load_module("puripuly_heart.config.settings_vnext.compat")


def _facade() -> ModuleType:
    return _load_module("puripuly_heart.config.settings_vnext.facade")


def _leaf_paths(value: object, prefix: str = "") -> set[str]:
    if isinstance(value, dict):
        paths: set[str] = set()
        for key, child in value.items():
            child_path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, dict) and child:
                paths.update(_leaf_paths(child, child_path))
            else:
                paths.add(child_path)
        return paths
    return {prefix} if prefix else set()


def _write_json_bytes(path: Path, data: dict[str, Any]) -> bytes:
    raw_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    path.write_bytes(raw_bytes)
    return raw_bytes


def _final_dev_v30_fixture() -> dict[str, Any]:
    fixture_path = Path(__file__).parent / "fixtures" / "final_dev_v30_settings.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def test_v24_maximal_fixture_migrates_to_canonical_vnext_serialization() -> None:
    migration = _migration()
    serialization = _serialization()

    settings = migration.from_dict(maximal_v24_settings_fixture())
    serialized = serialization.to_dict(settings)

    assert isinstance(settings, AppSettingsVNext)
    assert settings.settings_version == VNEXT_SETTINGS_SCHEMA_VERSION
    assert set(serialized) == {"settings_version", "intent", "state"}
    assert serialized["settings_version"] == VNEXT_SETTINGS_SCHEMA_VERSION
    assert serialized["intent"]["translation"]["model"] == "local_llm"
    assert serialized["intent"]["translation"]["connection"] == "ollama"
    assert serialized["intent"]["translation"]["qwen"]["region"] == "singapore"
    assert serialized["intent"]["translation"]["cerebras"]["llm_model"] == "gemma-4-31b"
    assert serialized["intent"]["translation"]["openrouter_model"] == ("qwen/qwen3.5-flash-02-23")
    assert serialized["intent"]["translation"]["openrouter_selected_source"] == "byok"
    assert serialized["intent"]["translation"]["openrouter_selection_alias"] == (
        "qwen35_flash_byok"
    )
    assert serialized["intent"]["translation"]["openrouter_provider_routing"] == "default"
    assert serialized["intent"]["local_llm"]["base_url"] == "http://127.0.0.1:12345/v1"
    assert serialized["intent"]["stt"]["provider"] == "deepgram"
    assert serialized["intent"]["peer_stt"]["provider"] == "soniox"
    assert serialized["intent"]["ui"]["locale"] == "ja"
    assert serialized["intent"]["stt"]["low_latency_mode"] is True
    assert serialized["intent"]["integrated_context"]["enabled"] is True
    assert serialized["state"]["integrated_context"]["bootstrapped"] is True
    assert serialized["state"]["peer_translation"]["eula_accepted"] is True
    assert serialized["state"]["provider_verification"]["deepgram"]["status"] == "verified"
    assert "ui" not in serialized["state"]
    assert "provider" not in serialized
    assert "openrouter" not in serialized
    assert "api_key_verified" not in serialized


def test_peer_auto_detection_intent_roundtrips_through_legacy_compatibility_projection() -> None:
    migration = _migration()
    serialization = _serialization()
    legacy = AppSettings()
    legacy.languages.peer_source_mode = "auto"
    legacy.languages.peer_expected_languages = ["ja", "zh-TW", "ja"]

    vnext = migration.from_legacy_app_settings(legacy)
    serialized = serialization.to_dict(vnext)
    projected = migration.to_legacy_dict(vnext)

    assert serialized["intent"]["languages"]["peer_source_mode"] == "auto"
    assert serialized["intent"]["languages"]["peer_expected_languages"] == ["ja", "zh-TW"]
    assert projected["languages"]["peer_source_mode"] == "auto"
    assert projected["languages"]["peer_expected_languages"] == ["ja", "zh-TW"]


def test_v30_soniox_auto_is_backed_up_and_migrated_once(tmp_path: Path) -> None:
    compat = _compat()
    serialization = _serialization()
    raw = serialization.to_dict(AppSettingsVNext())
    raw["settings_version"] = 30
    raw["intent"]["languages"]["peer_source_mode"] = "soniox_auto"
    path = tmp_path / "settings.json"
    original_bytes = _write_json_bytes(path, raw)
    fixed_now = datetime(2026, 7, 18, 1, 2, 3, tzinfo=timezone.utc)

    first = compat.load_vnext_settings(path, now=fixed_now)

    assert first.ok
    assert first.migrated is True
    assert first.backup_path == tmp_path / "settings.json.pre-v30.20260718T010203Z.bak"
    assert first.backup_path.read_bytes() == original_bytes
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["settings_version"] == VNEXT_SETTINGS_SCHEMA_VERSION
    assert persisted["intent"]["languages"]["peer_source_mode"] == "auto"
    assert "soniox_auto" not in path.read_text(encoding="utf-8")

    second = compat.load_vnext_settings(path, now=fixed_now)

    assert second.ok
    assert second.migrated is False
    assert second.backup_path is None
    assert list(tmp_path.glob("*.bak")) == [first.backup_path]


def test_high_version_legacy_shape_migrates_by_shape_not_settings_version() -> None:
    migration = _migration()
    serialization = _serialization()
    raw = maximal_v24_settings_fixture()
    raw["settings_version"] = VNEXT_SETTINGS_SCHEMA_VERSION + 100

    settings = migration.from_dict(raw)
    serialized = serialization.to_dict(settings)

    assert isinstance(settings, AppSettingsVNext)
    assert settings.settings_version == VNEXT_SETTINGS_SCHEMA_VERSION
    assert set(serialized) == {"settings_version", "intent", "state"}
    assert serialized["settings_version"] == VNEXT_SETTINGS_SCHEMA_VERSION
    assert serialized["intent"]["translation"]["model"] == "local_llm"
    assert serialized["intent"]["translation"]["connection"] == "ollama"


def test_final_dev_v30_fixture_migrates_with_semantic_and_verification_continuity(
    tmp_path: Path,
) -> None:
    compat = _compat()
    migration = _migration()
    serialization = _serialization()
    raw = _final_dev_v30_fixture()
    path = tmp_path / "settings.json"
    original_bytes = _write_json_bytes(path, raw)
    fixed_now = datetime(2026, 7, 17, 1, 2, 3, tzinfo=timezone.utc)

    first = compat.load_vnext_settings(path, now=fixed_now)

    assert first.ok
    assert first.migrated is True
    assert first.backup_path == tmp_path / "settings.json.pre-v30.20260717T010203Z.bak"
    assert first.backup_path.read_bytes() == original_bytes
    assert first.settings is not None
    canonical = serialization.to_dict(first.settings)
    assert canonical["intent"]["translation"]["model"] == "local_llm"
    assert canonical["intent"]["translation"]["connection"] == "ollama"
    assert canonical["intent"]["translation"]["fallback"] == {
        "enabled": True,
        "model": "deepseek_v4_flash",
        "connection": "openrouter",
        "selection_alias": "openrouter_deepseek_v4_flash",
    }
    assert canonical["intent"]["translation"]["qwen"]["region"] == "singapore"
    assert canonical["intent"]["translation"]["gemini"]["llm_model"] == "gemini-3.7-flash"
    assert canonical["intent"]["translation"]["deepseek"]["llm_model"] == "deepseek-v4-flash"
    assert canonical["intent"]["stt"]["custom_terms"] == raw["stt"]["custom_terms"]
    assert (
        canonical["intent"]["languages"]["recent_source_languages"]
        == raw["languages"]["recent_source_languages"]
    )
    assert canonical["state"]["telemetry"] == {
        "anonymous_id": "fixture-telemetry-anonymous-id",
        "sent_translation_success_dates_utc": ("2026-07-01", "2026-07-02"),
    }
    assert canonical["state"]["managed_connection"]["pending_delivery_ack_delivery_id"] == (
        "fixture-delivery-id"
    )
    assert all(
        canonical["state"]["provider_verification"][provider]["status"] == "verified"
        for provider in PROVIDER_VERIFICATION_FIELDS
    )
    projected = migration.to_legacy_dict(first.settings)
    assert projected["api_key_verified"] == raw["api_key_verified"]
    assert projected["gemini"] == {"llm_model": "gemini-3.7-flash"}
    assert projected["deepseek"] == {"llm_model": "deepseek-v4-flash"}
    assert projected["telemetry_state"] == {
        "anonymous_id": raw["telemetry"]["identifier"],
        "sent_translation_success_dates_utc": raw["telemetry"]["sent_utc_dates"],
    }
    assert (
        projected["managed_identity"]["pending_delivery_ack_delivery_id"]
        == raw["managed_identity"]["pending_delivery_ack_id"]
    )

    canonical_bytes = path.read_bytes()
    second = compat.load_vnext_settings(path, now=fixed_now)

    assert second.ok
    assert second.migrated is False
    assert second.backup_path is None
    assert path.read_bytes() == canonical_bytes
    assert list(tmp_path.glob("*.bak")) == [first.backup_path]


def test_final_dev_v30_ui_equivalent_save_preserves_migrated_and_changed_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    _write_json_bytes(path, _final_dev_v30_fixture())
    loaded = load_settings(path)

    loaded.ui.locale = "ko"
    loaded.audio.ring_buffer_ms = 875
    save_settings(path, loaded)
    reloaded = load_settings(path)

    assert reloaded.ui.locale == "ko"
    assert reloaded.audio.ring_buffer_ms == 875
    assert reloaded.translation.model.value == "local_llm"
    assert reloaded.gemini.llm_model.value == "gemini-3.7-flash"
    assert reloaded.deepseek.llm_model.value == "deepseek-v4-flash"
    assert reloaded.languages.recent_source_languages == ["fr", "de", "it", "ko", "en", "zh-CN"]
    assert reloaded.managed_identity.pending_delivery_ack_delivery_id == "fixture-delivery-id"
    assert all(
        getattr(reloaded.api_key_verified, provider) is True
        for provider in PROVIDER_VERIFICATION_FIELDS
    )


def test_vnext_dict_migrates_gemini_3_flash_nested_fields() -> None:
    from puripuly_heart.config.settings_vnext import migration, serialization

    canonical = serialization.to_dict(AppSettingsVNext())
    translation = canonical["intent"]["translation"]
    translation["model"] = "gemini3_flash"
    translation["connection"] = "official_byok"
    translation["gemini"] = {"llm_model": "gemini-3-flash-preview"}
    translation["openrouter_model"] = "google/gemini-3-flash-preview"
    translation["openrouter_selection_alias"] = "gemini3_flash_byok"
    translation["fallback"] = {
        "enabled": True,
        "model": "gemini3_flash",
        "connection": "openrouter",
        "selection_alias": "none",
    }

    migrated = migration.from_dict(canonical)
    result = serialization.to_dict(migrated)["intent"]["translation"]

    assert result["model"] == "gemini37_flash"
    assert result["gemini"]["llm_model"] == "gemini-3.7-flash"
    assert result["openrouter_model"] == "google/gemini-3.7-flash"
    assert result["openrouter_selection_alias"] == "gemini37_flash_byok"
    assert result["fallback"] == {
        "enabled": True,
        "model": "gemma4_26b_31b",
        "connection": "openrouter",
        "selection_alias": "openrouter_gemma4_26b_31b",
    }


def test_vnext_dict_migrates_legacy_timestamp_prompt_to_new_default() -> None:
    from puripuly_heart.config.prompts import load_prompt_for_provider
    from puripuly_heart.config.settings import LEGACY_TIMESTAMP_PROMPT
    from puripuly_heart.config.settings_vnext import migration, serialization

    canonical = serialization.to_dict(AppSettingsVNext())
    canonical["settings_version"] = VNEXT_SETTINGS_SCHEMA_VERSION - 1
    canonical["intent"]["prompts"]["system_prompt"] = LEGACY_TIMESTAMP_PROMPT

    migrated = migration.from_dict(canonical)

    assert migrated.intent.prompts.system_prompt == load_prompt_for_provider("gemini")


def test_vnext_dict_migrates_legacy_timestamp_prompt_from_legacy_shape() -> None:
    from puripuly_heart.config.prompts import load_prompt_for_provider
    from puripuly_heart.config.settings import LEGACY_TIMESTAMP_PROMPT
    from puripuly_heart.config.settings_vnext import migration

    raw = {
        "settings_version": SETTINGS_SCHEMA_VERSION - 1,
        "system_prompt": LEGACY_TIMESTAMP_PROMPT,
    }

    migrated = migration.from_dict(raw)

    assert migrated.intent.prompts.system_prompt == load_prompt_for_provider("gemini")


def test_vnext_dict_preserves_custom_prompt_through_migration() -> None:
    from puripuly_heart.config.settings_vnext import migration, serialization

    canonical = serialization.to_dict(AppSettingsVNext())
    canonical["settings_version"] = VNEXT_SETTINGS_SCHEMA_VERSION - 1
    canonical["intent"]["prompts"]["system_prompt"] = "my customized prompt"

    migrated = migration.from_dict(canonical)

    assert migrated.intent.prompts.system_prompt == "my customized prompt"


def test_vnext_dict_preserves_prompt_with_boundary_whitespace() -> None:
    from puripuly_heart.config.settings import LEGACY_TIMESTAMP_PROMPT
    from puripuly_heart.config.settings_vnext import migration, serialization

    stored_prompt = f"  {LEGACY_TIMESTAMP_PROMPT}  "
    canonical = serialization.to_dict(AppSettingsVNext())
    canonical["settings_version"] = VNEXT_SETTINGS_SCHEMA_VERSION - 1
    canonical["intent"]["prompts"]["system_prompt"] = stored_prompt

    migrated = migration.from_dict(canonical)

    assert migrated.intent.prompts.system_prompt == stored_prompt


def test_vnext_dict_migrates_disabled_gemini_3_flash_fallback_to_none() -> None:
    from puripuly_heart.config.settings_vnext import migration, serialization

    canonical = serialization.to_dict(AppSettingsVNext())
    translation = canonical["intent"]["translation"]
    translation["model"] = "gemini3_flash"
    translation["connection"] = "official_byok"
    translation["fallback"] = {
        "enabled": False,
        "model": "gemini3_flash",
        "connection": "openrouter",
        "selection_alias": "none",
    }

    migrated = migration.from_dict(canonical)
    result = serialization.to_dict(migrated)["intent"]["translation"]

    assert result["model"] == "gemini37_flash"
    assert result["fallback"] == {
        "enabled": False,
        "model": "deepseek_v4_flash",
        "connection": "official_byok",
        "selection_alias": "none",
    }


def test_v24_boolean_api_key_verification_migrates_every_provider_to_verified() -> None:
    migration = _migration()
    serialization = _serialization()
    raw = maximal_v24_settings_fixture()

    assert all(
        raw["api_key_verified"][provider] is True for provider in PROVIDER_VERIFICATION_FIELDS
    )

    serialized = serialization.to_dict(migration.from_dict(raw))
    provider_entries = serialized["state"]["provider_verification"]

    assert provider_entries == {
        provider: {
            "status": "verified",
            "provider": provider,
            "secret_key": PROVIDER_VERIFICATION_SECRET_KEYS[provider],
            "secret_revision": "legacy-dev-settings",
            "secret_fingerprint": None,
            "verifier_context": {"flow": "legacy_settings_migration"},
            "verifier_evidence": {"source": "legacy_boolean"},
        }
        for provider in PROVIDER_VERIFICATION_FIELDS
    }


def test_public_facade_save_does_not_fabricate_legacy_verification_provenance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    settings = AppSettings()
    for provider in PROVIDER_VERIFICATION_FIELDS:
        setattr(settings.api_key_verified, provider, True)

    save_settings(path, settings)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert set(raw) == {"settings_version", "intent", "state"}
    assert raw["settings_version"] == VNEXT_SETTINGS_SCHEMA_VERSION
    assert {
        provider: raw["state"]["provider_verification"][provider]
        for provider in PROVIDER_VERIFICATION_FIELDS
    } == {provider: {"status": "unknown"} for provider in PROVIDER_VERIFICATION_FIELDS}

    loaded = load_settings(path)
    assert all(
        getattr(loaded.api_key_verified, provider) is False
        for provider in PROVIDER_VERIFICATION_FIELDS
    )


def test_public_facade_save_preserves_existing_credential_bound_verification(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    verified = ProviderVerificationEntry(
        status="verified",
        provider="openrouter",
        secret_key="openrouter_api_key",
        secret_revision=None,
        secret_fingerprint="sha256:credential-bound",
        verifier_context={"flow": "settings_api_key_verification"},
        verifier_evidence={"source": "provider_verifier"},
    )
    canonical = AppSettingsVNext(
        state=PersistedOperationalState(
            provider_verification=ProviderVerificationState(openrouter=verified)
        )
    )
    save_settings(path, canonical)
    legacy = load_settings(path)
    legacy.ui.locale = "ja"

    save_settings(path, legacy)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["intent"]["ui"]["locale"] == "ja"
    assert raw["state"]["provider_verification"]["openrouter"] == {
        "status": "verified",
        "provider": "openrouter",
        "secret_key": "openrouter_api_key",
        "secret_revision": None,
        "secret_fingerprint": "sha256:credential-bound",
        "verifier_context": {"flow": "settings_api_key_verification"},
        "verifier_evidence": {"source": "provider_verifier"},
    }


def test_evidence_bound_verified_entry_serializes_and_projects_legacy_true(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    verified_openrouter = ProviderVerificationEntry(
        status="verified",
        provider="openrouter",
        secret_key="openrouter_api_key",
        secret_revision="secret-r1",
        secret_fingerprint="sha256:0123456789abcdef",
        verifier_context={"flow": "settings.verify_api_key"},
        verifier_evidence={"verifier": "openrouter", "latency_ms": 12.5},
    )
    settings = AppSettingsVNext(
        state=PersistedOperationalState(
            provider_verification=ProviderVerificationState(openrouter=verified_openrouter)
        )
    )

    save_settings(path, settings)

    raw = json.loads(path.read_text(encoding="utf-8"))
    openrouter_entry = raw["state"]["provider_verification"]["openrouter"]
    assert openrouter_entry == {
        "status": "verified",
        "provider": "openrouter",
        "secret_key": "openrouter_api_key",
        "secret_revision": "secret-r1",
        "secret_fingerprint": "sha256:0123456789abcdef",
        "verifier_context": {"flow": "settings.verify_api_key"},
        "verifier_evidence": {"verifier": "openrouter", "latency_ms": 12.5},
    }

    loaded = load_settings(path)
    assert loaded.api_key_verified.openrouter is True
    assert loaded.api_key_verified.deepgram is False


def test_cerebras_evidence_bound_provider_verification_entry_projects_legacy_true(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    verified_cerebras = ProviderVerificationEntry(
        status="verified",
        provider="cerebras",
        secret_key="cerebras_api_key",
        secret_revision=None,
        secret_fingerprint="sha256:cerebras-test-fingerprint",
        verifier_context={"flow": "settings.verify_api_key"},
        verifier_evidence={"verifier": "cerebras"},
    )
    settings = AppSettingsVNext(
        state=PersistedOperationalState(
            provider_verification=ProviderVerificationState(cerebras=verified_cerebras)
        )
    )

    save_settings(path, settings)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["state"]["provider_verification"]["cerebras"] == {
        "status": "verified",
        "provider": "cerebras",
        "secret_key": "cerebras_api_key",
        "secret_revision": None,
        "secret_fingerprint": "sha256:cerebras-test-fingerprint",
        "verifier_context": {"flow": "settings.verify_api_key"},
        "verifier_evidence": {"verifier": "cerebras"},
    }

    loaded = load_settings(path)
    assert loaded.api_key_verified.cerebras is True


def test_china_first_run_defaults_project_to_vnext_intent() -> None:
    migration = _migration()
    serialization = _serialization()

    settings = new_settings_for_first_run("zh-CN")
    serialized = serialization.to_dict(migration.from_legacy_app_settings(settings))

    assert serialized["intent"]["ui"]["locale"] == "zh-CN"
    assert serialized["intent"]["translation"]["model"] == "deepseek_v4_flash"
    assert serialized["intent"]["translation"]["connection"] == "managed_china"
    assert serialized["intent"]["translation"]["openrouter_model"] == (
        "deepseek/deepseek-v4-flash-0731"
    )
    assert serialized["intent"]["translation"]["openrouter_selected_source"] == "managed"
    assert serialized["intent"]["translation"]["openrouter_selection_alias"] == (
        "deepseek_v4_flash_managed"
    )
    assert serialized["intent"]["translation"]["openrouter_provider_routing"] == ("deepseek_only")
    assert serialized["intent"]["translation"]["fallback"] == {
        "enabled": True,
        "model": "gemma4_26b_31b",
        "connection": "openrouter",
        "selection_alias": "openrouter_gemma4_26b_31b",
    }
    assert "openrouter_fallback_selection_alias" not in serialized["intent"]["translation"]


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("none", (False, "deepseek_v4_flash", "official_byok", "none")),
        (
            "deepseek_v4_flash_official",
            (True, "deepseek_v4_flash", "official_byok", "deepseek_v4_flash_official"),
        ),
        (
            "openrouter_deepseek_v4_flash",
            (True, "deepseek_v4_flash", "openrouter", "openrouter_deepseek_v4_flash"),
        ),
        (
            "openrouter_gemma4_26b_a4b",
            (True, "gemma4", "openrouter", "openrouter_gemma4_26b_a4b"),
        ),
        (
            "openrouter_gemma4_26b_31b",
            (True, "gemma4_26b_31b", "openrouter", "openrouter_gemma4_26b_31b"),
        ),
        (
            "cerebras_gemma4_31b",
            (True, "gemma4_31b", "cerebras", "cerebras_gemma4_31b"),
        ),
    ],
)
def test_vnext_fallback_selection_alias_is_canonical_product_intent(
    alias: str,
    expected: tuple[bool, str, str, str],
) -> None:
    serialization = _serialization()
    raw = serialization.to_dict(AppSettingsVNext())
    raw["intent"]["translation"]["fallback"] = {"selection_alias": alias}

    loaded = serialization.from_dict(raw)
    fallback = loaded.intent.translation.fallback

    assert (
        fallback.enabled,
        fallback.model,
        fallback.connection,
        fallback.selection_alias,
    ) == expected
    assert serialization.to_dict(loaded)["intent"]["translation"]["fallback"] == {
        "enabled": expected[0],
        "model": expected[1],
        "connection": expected[2],
        "selection_alias": expected[3],
    }


@pytest.mark.parametrize(
    ("container", "alias", "selected_source", "expected_alias"),
    [
        ("openrouter", "deepseek_v4_flash", "byok", "openrouter_deepseek_v4_flash"),
        ("openrouter", "deepseek_v4_flash", "managed", "openrouter_deepseek_v4_flash"),
        ("openrouter", "deepseek_v4_flash_china", "managed", "deepseek_v4_flash_china"),
        ("openrouter", "qwen35_flash", "byok", "none"),
        ("openrouter", "broken-alias", "byok", "none"),
        ("translation", "deepseek_v4_flash_official", "byok", "deepseek_v4_flash_official"),
        ("translation", "openrouter_gemma4_26b_a4b", "byok", "openrouter_gemma4_26b_a4b"),
        ("translation", "cerebras_gemma4_31b", "byok", "cerebras_gemma4_31b"),
    ],
)
def test_legacy_fallback_aliases_load_to_safe_vnext_fallback_intent(
    container: str,
    alias: str,
    selected_source: str,
    expected_alias: str,
) -> None:
    migration = _migration()
    serialization = _serialization()
    raw = maximal_v24_settings_fixture()
    raw["translation"].pop("fallback", None)
    raw["openrouter"]["selected_source"] = selected_source
    if container == "openrouter":
        raw["openrouter"]["fallback_selection_alias"] = alias
    else:
        raw["translation"]["fallback_selection_alias"] = alias

    serialized = serialization.to_dict(migration.from_dict(raw))
    fallback = serialized["intent"]["translation"]["fallback"]

    assert fallback["selection_alias"] == expected_alias
    assert "fallback_selection_alias" not in serialized["intent"]["translation"]
    assert "openrouter_fallback_selection_alias" not in serialized["intent"]["translation"]


def test_current_vnext_unknown_fallback_alias_falls_back_to_none() -> None:
    serialization = _serialization()
    raw = serialization.to_dict(AppSettingsVNext())
    raw["intent"]["translation"]["fallback"] = {
        "enabled": True,
        "model": "deepseek_v4_flash",
        "connection": "openrouter",
        "selection_alias": "not-real",
    }

    loaded = serialization.from_dict(raw)

    assert loaded.intent.translation.fallback == TranslationFallbackIntent()


def test_current_vnext_explicit_none_fallback_alias_disables_stale_enabled_fields() -> None:
    serialization = _serialization()
    raw = serialization.to_dict(AppSettingsVNext())
    raw["intent"]["translation"]["fallback"] = {
        "enabled": True,
        "model": "gemma4_31b_cerebras",
        "connection": "official_byok",
        "selection_alias": "none",
    }

    loaded = serialization.from_dict(raw)

    assert loaded.intent.translation.fallback == TranslationFallbackIntent()


@pytest.mark.parametrize("source_version", [33, 34])
def test_pre_v35_cerebras_model_migrates_to_gemma31_connection_and_preserves_return_target(
    source_version: int,
) -> None:
    migration = _migration()
    serialization = _serialization()
    raw = serialization.to_dict(AppSettingsVNext())
    raw["settings_version"] = source_version
    raw["intent"]["translation"].update(
        {
            "model": "custom_http",
            "connection": "custom_http",
            "previous_llm_model": "gemma4_31b_cerebras",
            "connection_history": {
                "gemma4_31b": "openrouter",
                "gemma4_31b_cerebras": "official_byok",
            },
            "fallback": {
                "enabled": True,
                "model": "gemma4_31b_cerebras",
                "connection": "official_byok",
                "selection_alias": "cerebras_gemma4_31b",
            },
        }
    )

    loaded = migration.from_dict(raw)
    translated = loaded.intent.translation

    assert loaded.settings_version == VNEXT_SETTINGS_SCHEMA_VERSION
    assert translated.previous_llm_model == "gemma4_31b"
    assert translated.connection_history == {"gemma4_31b": "cerebras"}
    assert translated.fallback == TranslationFallbackIntent(selection_alias="cerebras_gemma4_31b")
    assert "gemma4_31b_cerebras" not in json.dumps(serialization.to_dict(loaded))


@pytest.mark.parametrize("source_version", [33, 34])
def test_pre_v35_active_cerebras_model_migrates_without_losing_explicit_disabled_fallback(
    source_version: int,
) -> None:
    migration = _migration()
    serialization = _serialization()
    raw = serialization.to_dict(AppSettingsVNext())
    raw["settings_version"] = source_version
    raw["intent"]["translation"].update(
        {
            "model": "gemma4_31b_cerebras",
            "connection": "official_byok",
            "connection_history": {
                "gemma4_31b": "openrouter",
                "gemma4_31b_cerebras": "official_byok",
            },
            "fallback": {
                "enabled": True,
                "model": "gemma4_31b_cerebras",
                "connection": "official_byok",
                "selection_alias": "none",
            },
        }
    )

    loaded = migration.from_dict(raw)
    translated = loaded.intent.translation

    assert translated.model == "gemma4_31b"
    assert translated.connection == "cerebras"
    assert translated.connection_history == {"gemma4_31b": "cerebras"}
    assert translated.fallback == TranslationFallbackIntent()


@pytest.mark.parametrize("source_version", [34, 35])
def test_pre_v36_deepseek_v4_pro_migrates_to_deepseek_v4_flash(
    source_version: int,
) -> None:
    migration = _migration()
    serialization = _serialization()
    raw = serialization.to_dict(AppSettingsVNext())
    raw["settings_version"] = source_version
    raw["intent"]["translation"].update(
        {
            "model": "deepseek_v4_pro",
            "connection": "official_byok",
            "connection_history": {
                "gemma4_26b_31b": "managed",
                "deepseek_v4_pro": "official_byok",
            },
            "fallback": {
                "enabled": True,
                "model": "deepseek_v4_pro",
                "connection": "official_byok",
                "selection_alias": "none",
            },
        }
    )

    loaded = migration.from_dict(raw)
    translated = loaded.intent.translation

    assert loaded.settings_version == VNEXT_SETTINGS_SCHEMA_VERSION
    assert translated.model == "deepseek_v4_flash"
    assert translated.connection == "official_byok"
    assert translated.connection_history == {
        "gemma4_26b_31b": "managed",
        "deepseek_v4_flash": "official_byok",
    }
    assert translated.fallback == TranslationFallbackIntent(
        enabled=True,
        model="deepseek_v4_flash",
        connection="official_byok",
        selection_alias="none",
    )
    assert "deepseek_v4_pro" not in json.dumps(serialization.to_dict(loaded))


def test_current_vnext_missing_fallback_alias_still_infers_compatibility_fields() -> None:
    serialization = _serialization()
    raw = serialization.to_dict(AppSettingsVNext())
    raw["intent"]["translation"]["fallback"] = {
        "enabled": True,
        "model": "deepseek_v4_flash",
        "connection": "managed_china",
    }

    loaded = serialization.from_dict(raw)

    assert loaded.intent.translation.fallback == TranslationFallbackIntent(
        selection_alias="deepseek_v4_flash_china"
    )


@pytest.mark.parametrize("loader_name", ["serialization", "migration"])
def test_missing_fallback_uses_unified_gemma_default(loader_name: str) -> None:
    migration = _migration()
    serialization = _serialization()
    raw = serialization.to_dict(AppSettingsVNext())
    raw["intent"]["translation"].pop("fallback")

    loader = serialization.from_dict if loader_name == "serialization" else migration.from_dict
    fallback = loader(raw).intent.translation.fallback

    assert fallback == TranslationFallbackIntent(selection_alias="openrouter_gemma4_26b_31b")


def test_legacy_missing_fallback_uses_unified_gemma_default() -> None:
    migration = _migration()
    raw = maximal_v24_settings_fixture()
    raw["translation"].pop("fallback", None)
    raw["openrouter"].pop("fallback_selection_alias", None)

    fallback = migration.from_dict(raw).intent.translation.fallback

    assert fallback == TranslationFallbackIntent(selection_alias="openrouter_gemma4_26b_31b")


def test_maximal_v24_fixture_preserves_telemetry_consent_and_identifier() -> None:
    migration = _migration()
    serialization = _serialization()

    serialized = serialization.to_dict(migration.from_dict(maximal_v24_settings_fixture()))

    assert serialized["intent"]["telemetry"] == {"consent": "allow"}
    assert serialized["state"]["telemetry"] == {
        "anonymous_id": "fixture-telemetry-anonymous-id",
        "sent_translation_success_dates_utc": ("2026-07-01", "2026-07-02"),
    }


def test_existing_settings_upgrade_unknown_telemetry_to_allow_with_identifier() -> None:
    migration = _migration()
    serialization = _serialization()

    existing = maximal_v24_settings_fixture()
    existing["telemetry"] = {}
    existing["telemetry_state"] = {}

    serialized = serialization.to_dict(migration.from_dict(existing))

    assert serialized["intent"]["telemetry"] == {"consent": "allow"}
    assert serialized["state"]["telemetry"]["anonymous_id"]
    assert serialized["state"]["telemetry"]["sent_translation_success_dates_utc"] == ()


def test_telemetry_consent_transitions_manage_operational_state() -> None:
    base = AppSettingsVNext(
        state=PersistedOperationalState(
            telemetry=TelemetryOperationalState(
                anonymous_id="existing-id",
                sent_translation_success_dates_utc=("2026-07-01",),
            )
        )
    )

    allowed = with_telemetry_consent(base, "allow", identifier_factory=lambda: "new-id")
    declined = with_telemetry_consent(allowed, "decline")
    allowed_again = with_telemetry_consent(declined, "allow", identifier_factory=lambda: "new-id")

    assert allowed.intent.telemetry.consent == "allow"
    assert allowed.state.telemetry.anonymous_id == "existing-id"
    assert allowed.state.telemetry.sent_translation_success_dates_utc == ("2026-07-01",)
    assert declined.intent.telemetry.consent == "decline"
    assert declined.state.telemetry.anonymous_id is None
    assert declined.state.telemetry.sent_translation_success_dates_utc == ()
    assert allowed_again.intent.telemetry.consent == "allow"
    assert allowed_again.state.telemetry.anonymous_id == "new-id"


def test_malformed_telemetry_sent_dates_are_ignored_and_deduplicated() -> None:
    serialization = _serialization()
    raw = serialization.to_dict(AppSettingsVNext())
    raw["state"]["telemetry"] = {
        "anonymous_id": " telemetry-id ",
        "sent_translation_success_dates_utc": [
            "2026-07-01",
            "bad-date",
            "2026-07-01",
            7,
            "2026-07-02",
        ],
    }

    loaded = serialization.from_dict(raw)

    assert loaded.state.telemetry.anonymous_id == "telemetry-id"
    assert loaded.state.telemetry.sent_translation_success_dates_utc == (
        "2026-07-01",
        "2026-07-02",
    )


def test_current_vnext_status_only_provider_verification_entries_load_as_unknown() -> None:
    migration = _migration()
    serialization = _serialization()
    raw = serialization.to_dict(AppSettingsVNext())
    raw["state"]["provider_verification"] = {
        "deepgram": {"status": "verified"},
        "soniox": {"status": "failed"},
        "google": {"status": "skipped"},
        "openrouter": {"status": "verified"},
        "deepseek": {"status": "failed"},
        "cerebras": {"status": "verified"},
        "alibaba_beijing": {"status": "skipped"},
        "alibaba_singapore": {"status": "verified"},
    }

    for loader in (serialization.from_dict, migration.from_dict):
        settings = loader(raw)
        serialized = serialization.to_dict(settings)

        assert serialized["state"]["provider_verification"] == {
            provider: {
                "status": "unknown",
                "provider": None,
                "secret_key": None,
                "secret_revision": None,
                "secret_fingerprint": None,
                "verifier_context": {},
                "verifier_evidence": {},
            }
            for provider in PROVIDER_VERIFICATION_FIELDS
        }
        assert migration.to_legacy_dict(settings)["api_key_verified"] == {
            provider: False for provider in PROVIDER_VERIFICATION_FIELDS
        }


def test_current_vnext_evidence_bound_provider_verification_entry_survives_compatibility_shim() -> (
    None
):
    migration = _migration()
    serialization = _serialization()
    raw = serialization.to_dict(AppSettingsVNext())
    raw["state"]["provider_verification"]["openrouter"] = {
        "status": "verified",
        "provider": "openrouter",
        "secret_key": "openrouter_api_key",
        "secret_revision": None,
        "secret_fingerprint": "sha256:0123456789abcdef",
        "verifier_context": {"flow": "settings.verify_api_key"},
        "verifier_evidence": {"verifier": "openrouter"},
    }

    settings = migration.from_dict(raw)
    serialized = serialization.to_dict(settings)

    assert (
        serialized["state"]["provider_verification"]["openrouter"]
        == raw["state"]["provider_verification"]["openrouter"]
    )
    assert migration.to_legacy_dict(settings)["api_key_verified"]["openrouter"] is True


def test_legacy_accepted_keys_read_without_reintroducing_legacy_write_projection() -> None:
    migration = _migration()
    serialization = _serialization()

    settings = migration.from_dict(legacy_compatibility_settings_fixture())
    serialized = serialization.to_dict(settings)

    assert settings.intent.overlay.calibration.offset_x == 0.42
    assert settings.intent.overlay.show_translation is False
    assert settings.intent.overlay.show_peer_original is False
    assert settings.intent.peer_stt.provider == "soniox"
    serialized_paths = _leaf_paths(serialized)
    assert {
        "overlay_calibration.offset_x",
        "intent.ui.overlay_enabled",
        "state.ui.overlay_enabled",
        "intent.ui.peer_translation_enabled",
        "state.ui.peer_translation_enabled",
        "peer_qwen_asr_stt.model",
        "peer_soniox_stt.endpoint",
        "system_prompts.legacy",
    }.isdisjoint(serialized_paths)


def test_current_vnext_dict_reads_and_serializes_idempotently() -> None:
    migration = _migration()
    serialization = _serialization()

    original = with_telemetry_consent(
        AppSettingsVNext(),
        "allow",
        identifier_factory=lambda: "current-settings-test-id",
    )
    raw = serialization.to_dict(original)

    loaded = migration.from_dict(raw)
    serialized = serialization.to_dict(loaded)

    assert loaded == original
    assert serialized == raw


def test_vnext_settings_version_is_loaded_as_metadata_not_input() -> None:
    migration = _migration()
    serialization = _serialization()
    raw = serialization.to_dict(AppSettingsVNext())
    raw["settings_version"] = "not-a-schema-discriminator"

    loaded = migration.from_dict(raw)

    assert loaded.settings_version == VNEXT_SETTINGS_SCHEMA_VERSION
    assert serialization.to_dict(loaded)["settings_version"] == VNEXT_SETTINGS_SCHEMA_VERSION


@pytest.mark.parametrize(
    ("legacy_output_device", "kind", "device_name"),
    [
        ("", "default_output_device", None),
        ("Fixture Speakers", "named_output_device", "Fixture Speakers"),
    ],
)
def test_legacy_output_device_migrates_to_canonical_capture_target(
    legacy_output_device: str,
    kind: str,
    device_name: str | None,
) -> None:
    migration = _migration()
    serialization = _serialization()
    raw = to_dict(AppSettings())
    raw["desktop_audio"]["output_device"] = legacy_output_device

    settings = migration.from_dict(raw)
    serialized = serialization.to_dict(settings)

    assert settings.intent.desktop_audio.capture_target.kind == kind
    assert settings.intent.desktop_audio.capture_target.device_name == device_name
    assert serialized["intent"]["desktop_audio"]["capture_target"] == {
        "kind": kind,
        "device_name": device_name,
        "process": None,
    }
    assert serialized["intent"]["desktop_audio"]["output_device"] == legacy_output_device
    assert (
        migration.to_legacy_dict(settings)["desktop_audio"]["output_device"] == legacy_output_device
    )


@pytest.mark.parametrize(
    ("legacy_output_device", "kind", "device_name"),
    [
        ("", "default_output_device", None),
        ("Fixture Speakers", "named_output_device", "Fixture Speakers"),
    ],
)
def test_pre_v27_canonical_vnext_output_device_migration_backs_up_before_rewrite(
    tmp_path: Path,
    legacy_output_device: str,
    kind: str,
    device_name: str | None,
) -> None:
    compat = _compat()
    serialization = _serialization()
    fixed_now = datetime(2026, 7, 10, 1, 2, 3, tzinfo=timezone.utc)
    path = tmp_path / "settings.json"
    raw = serialization.to_dict(AppSettingsVNext())
    raw["settings_version"] = 26
    raw["intent"]["desktop_audio"].pop("capture_target")
    raw["intent"]["desktop_audio"]["output_device"] = legacy_output_device
    original_bytes = _write_json_bytes(path, raw)

    result = compat.load_vnext_settings(path, now=fixed_now)

    assert result.status == compat.SettingsPersistenceStatus.SUCCESS
    assert result.migrated is True
    assert result.backup_path == tmp_path / "settings.json.pre-v26.20260710T010203Z.bak"
    assert result.backup_path.read_bytes() == original_bytes
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["intent"]["desktop_audio"]["capture_target"] == {
        "kind": kind,
        "device_name": device_name,
        "process": None,
    }
    assert persisted["intent"]["desktop_audio"]["output_device"] == legacy_output_device


@pytest.mark.parametrize(
    ("input_channel", "channel", "basename"),
    [
        ("STABLE", "stable", "Discord.exe"),
        ("PTB", "ptb", "DiscordPTB.exe"),
        ("Canary", "canary", "DiscordCanary.exe"),
    ],
)
def test_process_capture_target_round_trips_with_discord_update_resistant_identity(
    input_channel: str,
    channel: str,
    basename: str,
) -> None:
    serialization = _serialization()
    migration = _migration()
    from puripuly_heart.config.capture_target_resolution import resolve_desktop_audio_capture_target

    settings = with_telemetry_consent(
        AppSettingsVNext(
            intent=replace(
                AppSettingsVNext().intent,
                desktop_audio=replace(
                    AppSettingsVNext().intent.desktop_audio,
                    capture_target=CaptureTargetIntent.process_target(
                        ProcessCaptureTargetIntent.discord(input_channel)
                    ),
                ),
            ),
        ),
        "allow",
        identifier_factory=lambda: "capture-target-test-id",
    )

    serialized = serialization.to_dict(settings)
    restored = migration.from_dict(serialized)

    target = serialized["intent"]["desktop_audio"]["capture_target"]
    assert target["kind"] == "process"
    assert target["process"] == {
        "kind": "discord",
        "executable_identity": None,
        "discord_channel": channel,
        "executable_basename": basename,
    }
    assert restored == settings
    resolved = resolve_desktop_audio_capture_target(settings.intent.desktop_audio.capture_target)
    assert resolved.discord_channel == channel
    assert resolved.executable_basename == basename
    assert migration.to_legacy_dict(settings)["desktop_audio"]["output_device"] == ""


def test_generic_process_identity_is_normalized_without_relocation() -> None:
    serialization = _serialization()
    migration = _migration()
    original_identity = r"C:/Apps/Example/Example.EXE"
    target = CaptureTargetIntent.process_target(
        ProcessCaptureTargetIntent.generic_executable(original_identity)
    )
    settings = with_capture_target(AppSettingsVNext(), target)

    restored = migration.from_dict(serialization.to_dict(settings))

    assert restored.intent.desktop_audio.capture_target.process is not None
    assert restored.intent.desktop_audio.capture_target.process.executable_identity == (
        r"c:\apps\example\example.exe"
    )
    assert (
        restored.intent.desktop_audio.capture_target.process.executable_identity
        != original_identity
    )


@pytest.mark.parametrize(
    "process",
    [
        ProcessCaptureTargetIntent.generic_executable(r"\\server\share\example.exe"),
        ProcessCaptureTargetIntent.vrchat(r"\\server\share\VRChat.exe"),
    ],
)
def test_persisted_process_targets_accept_fully_qualified_unc_identities(
    process: ProcessCaptureTargetIntent,
) -> None:
    serialization = _serialization()
    migration = _migration()
    settings = AppSettingsVNext(
        intent=replace(
            AppSettingsVNext().intent,
            desktop_audio=replace(
                AppSettingsVNext().intent.desktop_audio,
                capture_target=CaptureTargetIntent.process_target(process),
            ),
        )
    )

    restored = migration.from_dict(serialization.to_dict(settings))

    assert restored.intent.desktop_audio.capture_target.process == process


@pytest.mark.parametrize(
    ("process_kind", "identity", "accepted"),
    [
        ("generic_executable", r"C:\Apps\example.exe", True),
        ("generic_executable", r"\\server\share\example.exe", True),
        ("generic_executable", r"C:example.exe", False),
        ("generic_executable", r"\??\C:\example.exe", False),
        ("generic_executable", r"\Device\Audio\example.exe", False),
        ("vrchat", r"C:\Apps\VRChat.exe", True),
        ("vrchat", r"\\server\share\VRChat.exe", True),
        ("vrchat", r"C:VRChat.exe", False),
        ("vrchat", r"\??\C:\VRChat.exe", False),
        ("vrchat", r"\Device\Audio\VRChat.exe", False),
    ],
)
def test_persisted_process_identity_path_matrix(
    process_kind: str,
    identity: str,
    accepted: bool,
) -> None:
    factory = getattr(ProcessCaptureTargetIntent, process_kind)

    if accepted:
        assert factory(identity).executable_identity is not None
        return
    with pytest.raises(ValueError, match="executable"):
        factory(identity)


@pytest.mark.parametrize(
    ("process_kind", "identity", "accepted"),
    [
        ("generic_executable", r"c:\apps\example.exe", True),
        ("generic_executable", r"\\server\share\example.exe", True),
        ("generic_executable", r"c:example.exe", False),
        ("generic_executable", r"\??\c:\example.exe", False),
        ("generic_executable", r"\device\audio\example.exe", False),
        ("vrchat", r"c:\apps\vrchat.exe", True),
        ("vrchat", r"\\server\share\vrchat.exe", True),
        ("vrchat", r"c:vrchat.exe", False),
        ("vrchat", r"\??\c:\vrchat.exe", False),
        ("vrchat", r"\device\audio\vrchat.exe", False),
    ],
)
def test_resolved_process_identity_path_matrix(
    process_kind: str,
    identity: str,
    accepted: bool,
) -> None:
    from puripuly_heart.config.resolved import ResolvedDesktopAudioCaptureTarget

    kwargs = {
        "kind": "process",
        "process_kind": process_kind,
        "executable_identity": identity,
    }
    if accepted:
        assert ResolvedDesktopAudioCaptureTarget(**kwargs).executable_identity == identity
        return
    with pytest.raises(ValueError, match="executable identity"):
        ResolvedDesktopAudioCaptureTarget(**kwargs)


def test_capture_target_validation_rejects_ambiguous_or_path_bound_discord_values() -> None:
    with pytest.raises(ValueError, match="non-empty device name"):
        CaptureTargetIntent.named_output_device("   ")
    with pytest.raises(ValueError, match="VRChat.exe"):
        ProcessCaptureTargetIntent.vrchat(r"C:\Apps\Other.exe")
    with pytest.raises(ValueError, match="installation path"):
        ProcessCaptureTargetIntent(
            kind="discord",
            executable_identity=r"C:\Users\example\AppData\Discord\app-1.0\Discord.exe",
            discord_channel="stable",
        )
    with pytest.raises(ValueError, match="executable"):
        ProcessCaptureTargetIntent.generic_executable("Example.exe")
    with pytest.raises(ValueError, match="executable"):
        ProcessCaptureTargetIntent.vrchat(r"\VRChat\VRChat.exe")
    with pytest.raises(ValueError, match="executable"):
        ProcessCaptureTargetIntent.generic_executable(r"\\server\example.exe")
    with pytest.raises(ValueError, match="executable"):
        ProcessCaptureTargetIntent.vrchat(r"\\server\VRChat.exe")
    with pytest.raises(ValueError, match="executable"):
        ProcessCaptureTargetIntent.generic_executable(r"\\.\pipe\capture.exe")
    with pytest.raises(ValueError, match="executable"):
        ProcessCaptureTargetIntent.vrchat(r"\\.\pipe\VRChat.exe")
    with pytest.raises(ValueError, match="executable"):
        ProcessCaptureTargetIntent.generic_executable(r"\\?\c:\capture.exe")
    with pytest.raises(ValueError, match="executable"):
        ProcessCaptureTargetIntent.vrchat(r"\\?\c:\VRChat.exe")
    with pytest.raises(ValueError, match="Discord channel identity"):
        ProcessCaptureTargetIntent.generic_executable(r"C:\Apps\Discord\Discord.exe")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "process", "process_kind": "generic_executable"},
        {
            "kind": "process",
            "process_kind": "generic_executable",
            "executable_identity": r"C:\Apps\Discord\Discord.exe",
        },
        {
            "kind": "process",
            "process_kind": "generic_executable",
            "executable_identity": "example.exe",
        },
        {
            "kind": "process",
            "process_kind": "vrchat",
            "executable_identity": r"C:\Apps\Other.exe",
        },
        {
            "kind": "process",
            "process_kind": "vrchat",
            "executable_identity": r"\VRChat\VRChat.exe",
        },
        {
            "kind": "process",
            "process_kind": "generic_executable",
            "executable_identity": r"\\server\example.exe",
        },
        {
            "kind": "process",
            "process_kind": "vrchat",
            "executable_identity": r"\\server\vrchat.exe",
        },
        {
            "kind": "process",
            "process_kind": "generic_executable",
            "executable_identity": r"\\.\pipe\capture.exe",
        },
        {
            "kind": "process",
            "process_kind": "vrchat",
            "executable_identity": r"\\.\pipe\vrchat.exe",
        },
        {
            "kind": "process",
            "process_kind": "generic_executable",
            "executable_identity": r"\\?\c:\capture.exe",
        },
        {
            "kind": "process",
            "process_kind": "vrchat",
            "executable_identity": r"\\?\c:\vrchat.exe",
        },
        {
            "kind": "process",
            "process_kind": "discord",
            "discord_channel": "stable",
            "executable_basename": "DiscordPTB.exe",
        },
        {
            "kind": "process",
            "process_kind": "discord",
            "executable_identity": r"C:\Apps\Discord\Discord.exe",
            "discord_channel": "stable",
            "executable_basename": "Discord.exe",
        },
        {
            "kind": "process",
            "process_kind": "discord",
            "discord_channel": "Stable",
            "executable_basename": "Discord.exe",
        },
    ],
)
def test_resolved_process_capture_target_rejects_incomplete_or_malformed_values(
    kwargs: dict[str, object],
) -> None:
    from puripuly_heart.config.resolved import ResolvedDesktopAudioCaptureTarget

    with pytest.raises(ValueError):
        ResolvedDesktopAudioCaptureTarget(**kwargs)


def test_resolved_process_capture_target_accepts_canonical_generic_vrchat_and_discord_values() -> (
    None
):
    from puripuly_heart.config.resolved import ResolvedDesktopAudioCaptureTarget

    generic = ResolvedDesktopAudioCaptureTarget(
        kind="process",
        process_kind="generic_executable",
        executable_identity=r"c:\apps\example\example.exe",
    )
    vrchat = ResolvedDesktopAudioCaptureTarget(
        kind="process",
        process_kind="vrchat",
        executable_identity=r"c:\vrchat\vrchat.exe",
    )
    discord = ResolvedDesktopAudioCaptureTarget(
        kind="process",
        process_kind="discord",
        discord_channel="canary",
        executable_basename="DiscordCanary.exe",
    )
    generic_unc = ResolvedDesktopAudioCaptureTarget(
        kind="process",
        process_kind="generic_executable",
        executable_identity=r"\\server\share\example.exe",
    )
    vrchat_unc = ResolvedDesktopAudioCaptureTarget(
        kind="process",
        process_kind="vrchat",
        executable_identity=r"\\server\share\vrchat.exe",
    )

    assert generic.process_kind == "generic_executable"
    assert vrchat.process_kind == "vrchat"
    assert discord.discord_channel == "canary"
    assert generic_unc.executable_identity == r"\\server\share\example.exe"
    assert vrchat_unc.executable_identity == r"\\server\share\vrchat.exe"


def test_capture_target_mutation_updates_only_immutable_vnext_intent() -> None:
    original = AppSettingsVNext()
    target = CaptureTargetIntent.process_target(
        ProcessCaptureTargetIntent.generic_executable(r"C:\Apps\Example\Example.exe")
    )

    updated = with_capture_target(original, target)

    assert updated is not original
    assert original.intent.desktop_audio.capture_target.kind == "default_output_device"
    assert updated.intent.desktop_audio.capture_target == target
    assert updated.state == original.state


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (
            CaptureTargetIntent.default_output_device(),
            {"kind": "default_output_device", "device_name": None},
        ),
        (
            CaptureTargetIntent.named_output_device("Fixture Speakers"),
            {"kind": "named_output_device", "device_name": "Fixture Speakers"},
        ),
        (
            CaptureTargetIntent.process_target(
                ProcessCaptureTargetIntent.generic_executable(r"C:\Apps\example.exe")
            ),
            {
                "kind": "process",
                "process_kind": "generic_executable",
                "executable_identity": r"c:\apps\example.exe",
            },
        ),
        (
            CaptureTargetIntent.process_target(
                ProcessCaptureTargetIntent.vrchat(r"C:\Apps\VRChat.exe")
            ),
            {
                "kind": "process",
                "process_kind": "vrchat",
                "executable_identity": r"c:\apps\vrchat.exe",
            },
        ),
        (
            CaptureTargetIntent.process_target(ProcessCaptureTargetIntent.discord("PTB")),
            {
                "kind": "process",
                "process_kind": "discord",
                "discord_channel": "ptb",
                "executable_basename": "DiscordPTB.exe",
            },
        ),
    ],
)
def test_capture_target_resolution_covers_all_target_kinds(
    target: CaptureTargetIntent,
    expected: dict[str, str | None],
) -> None:
    from puripuly_heart.config.capture_target_resolution import resolve_desktop_audio_capture_target

    resolved = resolve_desktop_audio_capture_target(target)

    for field, value in expected.items():
        assert getattr(resolved, field) == value


@pytest.mark.parametrize(
    ("target", "legacy_output_device"),
    [
        (CaptureTargetIntent.default_output_device(), ""),
        (CaptureTargetIntent.named_output_device("Fixture Speakers"), "Fixture Speakers"),
        (
            CaptureTargetIntent.process_target(
                ProcessCaptureTargetIntent.generic_executable(r"C:\Apps\example.exe")
            ),
            "",
        ),
    ],
)
def test_capture_target_legacy_facade_projection(
    target: CaptureTargetIntent,
    legacy_output_device: str,
) -> None:
    migration = _migration()
    settings = with_capture_target(AppSettingsVNext(), target)

    legacy = migration.to_legacy_dict(settings)

    assert legacy["desktop_audio"]["output_device"] == legacy_output_device


@pytest.mark.parametrize(
    ("target", "legacy_output_device"),
    [
        (CaptureTargetIntent.default_output_device(), ""),
        (CaptureTargetIntent.named_output_device("Fixture Speakers"), "Fixture Speakers"),
        (
            CaptureTargetIntent.process_target(
                ProcessCaptureTargetIntent.generic_executable(r"C:\Apps\Game\Game.exe")
            ),
            "",
        ),
    ],
)
def test_legacy_facade_save_preserves_existing_canonical_process_target(
    tmp_path: Path,
    target: CaptureTargetIntent,
    legacy_output_device: str,
) -> None:
    facade = _facade()
    path = tmp_path / "settings.json"
    settings = with_capture_target(AppSettingsVNext(), target)
    facade.save_vnext_settings(path, settings)

    legacy = facade.load_settings(path)
    facade.save_settings(path, legacy)
    reloaded = facade.load_vnext_settings(path)

    assert legacy.desktop_audio.output_device == legacy_output_device
    assert reloaded.settings is not None
    assert reloaded.settings.intent.desktop_audio.capture_target == target


def test_legacy_facade_device_change_replaces_process_projection_with_named_device(
    tmp_path: Path,
) -> None:
    facade = _facade()
    path = tmp_path / "settings.json"
    process_target = CaptureTargetIntent.process_target(
        ProcessCaptureTargetIntent.generic_executable(r"C:\Apps\Game\Game.exe")
    )
    settings = with_capture_target(AppSettingsVNext(), process_target)
    facade.save_vnext_settings(path, settings)

    legacy = facade.load_settings(path)
    legacy.desktop_audio.output_device = "Replacement Speakers"
    from puripuly_heart.config.resolved import ResolvedDesktopAudioCaptureTarget

    legacy.desktop_audio.runtime_capture_target = ResolvedDesktopAudioCaptureTarget(
        kind="named_output_device",
        device_name="Replacement Speakers",
    )
    facade.save_settings(path, legacy)
    reloaded = facade.load_vnext_settings(path)

    assert reloaded.settings is not None
    assert reloaded.settings.intent.desktop_audio.capture_target == (
        CaptureTargetIntent.named_output_device("Replacement Speakers")
    )


def test_legacy_facade_device_change_without_runtime_projection_replaces_process_target(
    tmp_path: Path,
) -> None:
    facade = _facade()
    path = tmp_path / "settings.json"
    process_target = CaptureTargetIntent.process_target(
        ProcessCaptureTargetIntent.generic_executable(r"C:\Apps\Game\Game.exe")
    )
    facade.save_vnext_settings(path, with_capture_target(AppSettingsVNext(), process_target))

    legacy = facade.load_settings(path)
    legacy.desktop_audio.output_device = "Replacement Speakers"
    legacy.desktop_audio.runtime_capture_target = None
    facade.save_settings(path, legacy)
    reloaded = facade.load_vnext_settings(path)

    assert reloaded.settings is not None
    assert reloaded.settings.intent.desktop_audio.capture_target == (
        CaptureTargetIntent.named_output_device("Replacement Speakers")
    )


def test_capture_target_resolution_excludes_pids_and_runtime_state() -> None:
    from puripuly_heart.config.capture_target_resolution import resolve_desktop_audio_capture_target

    target = CaptureTargetIntent.process_target(
        ProcessCaptureTargetIntent.vrchat(r"C:\VRChat\VRChat.exe")
    )

    resolved = resolve_desktop_audio_capture_target(target)
    serialized = _serialization().to_dict(
        AppSettingsVNext(
            intent=replace(
                AppSettingsVNext().intent,
                desktop_audio=replace(
                    AppSettingsVNext().intent.desktop_audio, capture_target=target
                ),
            )
        )
    )

    assert resolved.kind == "process"
    assert resolved.process_kind == "vrchat"
    assert resolved.executable_identity == r"c:\vrchat\vrchat.exe"
    assert not {"pid", "active", "warning", "retry", "capture_state"}.intersection(
        serialized["intent"]["desktop_audio"]["capture_target"]
    )


def test_save_vnext_settings_normalizes_stale_settings_version_to_current(
    tmp_path: Path,
) -> None:
    compat = _compat()
    path = tmp_path / "settings.json"
    stale = replace(AppSettingsVNext(), settings_version=VNEXT_SETTINGS_SCHEMA_VERSION - 1)

    result = compat.save_vnext_settings(path, stale)

    assert result.ok
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert set(raw) == {"settings_version", "intent", "state"}
    assert raw["settings_version"] == VNEXT_SETTINGS_SCHEMA_VERSION


def test_migration_on_load_creates_byte_identical_backup_and_writes_vnext_with_collision(
    tmp_path: Path,
) -> None:
    compat = _compat()
    fixed_now = datetime(2026, 6, 9, 1, 2, 3, tzinfo=timezone.utc)
    path = tmp_path / "settings.json"
    original_bytes = _write_json_bytes(path, maximal_v24_settings_fixture())
    colliding_backup = tmp_path / "settings.json.pre-v25.20260609T010203Z.bak"
    colliding_backup.write_bytes(b"existing backup")

    result = compat.load_vnext_settings(path, now=fixed_now)

    assert result.status == compat.SettingsPersistenceStatus.SUCCESS
    assert result.migrated is True
    assert result.backup_path == tmp_path / "settings.json.pre-v25.20260609T010203Z.1.bak"
    assert result.backup_path.read_bytes() == original_bytes
    assert colliding_backup.read_bytes() == b"existing backup"
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert set(persisted) == {"settings_version", "intent", "state"}
    assert persisted["settings_version"] == VNEXT_SETTINGS_SCHEMA_VERSION


def test_backup_creation_failure_aborts_vnext_save_and_leaves_original_bytes(
    tmp_path: Path,
) -> None:
    compat = _compat()
    fixed_now = datetime(2026, 6, 9, 1, 2, 3, tzinfo=timezone.utc)
    path = tmp_path / "settings.json"
    original_bytes = _write_json_bytes(path, maximal_v24_settings_fixture())
    first_backup = tmp_path / "settings.json.pre-v25.20260609T010203Z.bak"
    first_backup.write_bytes(b"collision")

    result = compat.load_vnext_settings(path, now=fixed_now, max_backup_attempts=1)

    assert result.status == compat.SettingsPersistenceStatus.BACKUP_FAILED
    assert result.settings is None
    assert path.read_bytes() == original_bytes
    assert first_backup.read_bytes() == b"collision"


def test_save_failure_before_final_replace_leaves_original_and_backup_safe(
    tmp_path: Path,
) -> None:
    compat = _compat()
    fixed_now = datetime(2026, 6, 9, 1, 2, 3, tzinfo=timezone.utc)
    path = tmp_path / "settings.json"
    original_bytes = _write_json_bytes(path, maximal_v24_settings_fixture())
    (tmp_path / "settings.json.tmp").mkdir()

    result = compat.load_vnext_settings(path, now=fixed_now)

    assert result.status == compat.SettingsPersistenceStatus.SAVE_FAILED
    assert result.settings is None
    assert path.read_bytes() == original_bytes
    backup_path = tmp_path / "settings.json.pre-v25.20260609T010203Z.bak"
    assert backup_path.read_bytes() == original_bytes


def test_post_replace_validation_failure_restores_original_bytes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    compat = _compat()
    fixed_now = datetime(2026, 7, 17, 1, 2, 3, tzinfo=timezone.utc)
    path = tmp_path / "settings.json"
    original_bytes = _write_json_bytes(path, _final_dev_v30_fixture())

    def fail_validation(_path: Path, _settings: AppSettingsVNext) -> None:
        raise ValueError("injected validation failure")

    monkeypatch.setattr(compat, "_validate_persisted_settings", fail_validation)

    result = compat.load_vnext_settings(path, now=fixed_now)

    assert result.status == compat.SettingsPersistenceStatus.SAVE_FAILED
    assert result.settings is None
    assert path.read_bytes() == original_bytes
    assert result.backup_path is not None
    assert result.backup_path.read_bytes() == original_bytes


def test_post_replace_validation_and_restoration_failure_returns_safe_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    compat = _compat()
    path = tmp_path / "settings.json"
    original_bytes = _write_json_bytes(path, _final_dev_v30_fixture())
    validation_sentinel = "private-validation-setting-sentinel"
    restoration_sentinel = "private-restoration-setting-sentinel"

    def fail_validation(_path: Path, _settings: AppSettingsVNext) -> None:
        raise ValueError(validation_sentinel)

    def fail_restoration(_path: Path, _content: bytes) -> None:
        raise OSError(restoration_sentinel)

    monkeypatch.setattr(compat, "_validate_persisted_settings", fail_validation)
    monkeypatch.setattr(compat, "_atomic_write_bytes", fail_restoration)

    result = compat.load_vnext_settings(path)

    assert result.status == compat.SettingsPersistenceStatus.SAVE_FAILED
    assert result.settings is None
    assert result.error is not None
    assert result.error.message == "save_failed:RuntimeError"
    assert validation_sentinel not in result.error.message
    assert restoration_sentinel not in result.error.message
    assert result.backup_path is not None
    assert result.backup_path.read_bytes() == original_bytes


def test_migration_diagnostics_include_only_approved_metadata(caplog, tmp_path: Path) -> None:
    compat = _compat()
    raw = _final_dev_v30_fixture()
    prohibited_values = (
        raw["system_prompt"],
        raw["osc"]["host"],
        raw["managed_identity"]["installation_id"],
        raw["telemetry"]["identifier"],
    )
    success_path = tmp_path / "success.json"
    _write_json_bytes(success_path, raw)

    with caplog.at_level("INFO", logger=compat.__name__):
        success = compat.load_vnext_settings(success_path)

    assert success.ok
    assert "source_shape=legacy destination_shape=canonical status=success" in caplog.text
    assert all(value not in caplog.text for value in prohibited_values)

    caplog.clear()
    failure_path = tmp_path / "failure.json"
    failure_path.write_text("not-json-user-value", encoding="utf-8")
    with caplog.at_level("WARNING", logger=compat.__name__):
        failure = compat.load_vnext_settings(failure_path)

    assert failure.status == compat.SettingsPersistenceStatus.PARSE_FAILED
    assert failure.error is not None
    assert failure.error.message == "parse_failed:JSONDecodeError"
    assert "not-json-user-value" not in failure.error.message
    assert "failure_category=parse_failed" in caplog.text
    assert "not-json-user-value" not in caplog.text


def test_parse_and_migration_failures_return_explicit_results_without_overwrite(
    tmp_path: Path,
) -> None:
    compat = _compat()
    parse_path = tmp_path / "parse-settings.json"
    parse_path.write_text("not json", encoding="utf-8")

    parse_result = compat.load_vnext_settings(parse_path)

    assert parse_result.status == compat.SettingsPersistenceStatus.PARSE_FAILED
    assert parse_path.read_text(encoding="utf-8") == "not json"

    migration_path = tmp_path / "migration-settings.json"
    migration_bytes = _write_json_bytes(
        migration_path,
        {
            **maximal_v24_settings_fixture(),
            "overlay": {"calibration": {"anchor": "unsupported_anchor"}},
        },
    )

    migration_result = compat.load_vnext_settings(migration_path)
    assert migration_result.status == compat.SettingsPersistenceStatus.MIGRATION_FAILED
    assert migration_result.error is not None
    assert migration_result.error.message == "migration_failed:ValueError"
    assert "unsupported_anchor" not in migration_result.error.message
    assert migration_path.read_bytes() == migration_bytes


@pytest.mark.parametrize(
    ("raw", "case_id"),
    [
        (
            {"settings_version": VNEXT_SETTINGS_SCHEMA_VERSION, "intent": {}},
            "missing-state",
        ),
        (
            {"settings_version": VNEXT_SETTINGS_SCHEMA_VERSION, "intent": [], "state": {}},
            "non-object-intent",
        ),
        (
            {"settings_version": VNEXT_SETTINGS_SCHEMA_VERSION, "intent": {}, "state": []},
            "non-object-state",
        ),
    ],
)
def test_malformed_current_vnext_top_level_shape_fails_without_backup_or_overwrite(
    tmp_path: Path,
    raw: dict[str, Any],
    case_id: str,
) -> None:
    compat = _compat()
    path = tmp_path / f"{case_id}.json"
    original_bytes = _write_json_bytes(path, raw)

    result = compat.load_vnext_settings(path)

    assert result.status == compat.SettingsPersistenceStatus.MIGRATION_FAILED
    assert result.settings is None
    assert result.backup_path is None
    assert path.read_bytes() == original_bytes
    assert not list(tmp_path.glob("*.bak"))


def test_vnext_settings_version_only_difference_does_not_backup_or_overwrite(
    tmp_path: Path,
) -> None:
    compat = _compat()
    serialization = _serialization()
    fixed_now = datetime(2026, 6, 9, 1, 2, 3, tzinfo=timezone.utc)
    path = tmp_path / "settings.json"
    settings = with_telemetry_consent(
        AppSettingsVNext(),
        "allow",
        identifier_factory=lambda: "version-only-test-id",
    )
    raw = serialization.to_dict(settings)
    raw["settings_version"] = VNEXT_SETTINGS_SCHEMA_VERSION - 1
    raw["intent"]["ui"]["locale"] = "ja"
    original_bytes = _write_json_bytes(path, raw)

    result = compat.load_vnext_settings(path, now=fixed_now)

    assert result.status == compat.SettingsPersistenceStatus.SUCCESS
    assert result.settings is not None
    assert result.settings.settings_version == VNEXT_SETTINGS_SCHEMA_VERSION
    assert result.settings.intent.ui.locale == "ja"
    assert result.migrated is False
    assert result.backup_path is None
    assert path.read_bytes() == original_bytes
    assert not list(tmp_path.glob("*.bak"))


def test_facade_projection_failure_returns_explicit_result_without_overwrite(
    tmp_path: Path,
) -> None:
    compat = _compat()
    serialization = _serialization()
    path = tmp_path / "settings.json"
    settings = with_telemetry_consent(
        AppSettingsVNext(),
        "allow",
        identifier_factory=lambda: "projection-failure-test-id",
    )
    raw = serialization.to_dict(settings)
    raw["intent"]["osc"]["port"] = "not-an-int"
    original_bytes = _write_json_bytes(path, raw)

    result = load_settings_with_result(path)

    assert result.status == compat.SettingsPersistenceStatus.MIGRATION_FAILED
    assert result.settings is None
    assert result.error is not None
    assert result.error.message == "migration_failed:ValueError"
    assert "not-an-int" not in result.error.message
    assert path.read_bytes() == original_bytes
    with pytest.raises(RuntimeError, match="migration_failed:ValueError") as exc_info:
        load_settings(path)
    assert "not-an-int" not in str(exc_info.value)


def test_raw_provider_api_key_fields_are_absent_from_vnext_serialized_output() -> None:
    migration = _migration()
    serialization = _serialization()
    raw = maximal_v24_settings_fixture()
    raw.update(
        {
            "google_api_key": "raw-google-secret",
            "openrouter_api_key": "raw-openrouter-secret",
            "deepgram_api_key": "raw-deepgram-secret",
            "soniox_api_key": "raw-soniox-secret",
            "local_llm_api_key": "raw-local-secret",
            "cerebras_api_key": "raw-cerebras-secret",
            "openrouter_managed_qq_api_key": "raw-managed-qq-secret",
        }
    )

    serialized = serialization.to_dict(migration.from_dict(raw))
    encoded = json.dumps(serialized, ensure_ascii=False)
    forbidden_names = {
        "alibaba_api_key",
        "alibaba_api_key_beijing",
        "alibaba_api_key_singapore",
        "deepgram_api_key",
        "deepseek_api_key",
        "cerebras_api_key",
        "google_api_key",
        "local_llm_api_key",
        "openrouter_api_key",
        "openrouter_managed_api_key",
        "openrouter_managed_qq_api_key",
        "soniox_api_key",
    }

    assert forbidden_names.isdisjoint(
        path.rsplit(".", maxsplit=1)[-1] for path in _leaf_paths(serialized)
    )
    assert "raw-google-secret" not in encoded
    assert "raw-openrouter-secret" not in encoded
    assert "raw-deepgram-secret" not in encoded
    assert "raw-soniox-secret" not in encoded
    assert "raw-local-secret" not in encoded
    assert "raw-cerebras-secret" not in encoded
    assert "raw-managed-qq-secret" not in encoded


def test_secret_bearing_legacy_local_llm_extra_body_is_repaired_before_vnext_output() -> None:
    migration = _migration()
    serialization = _serialization()
    raw = maximal_v24_settings_fixture()
    raw["local_llm"]["extra_body"] = {
        "temperature": 0.2,
        "api_key": "raw-local-llm-secret",
    }

    serialized = serialization.to_dict(migration.from_dict(raw))
    encoded = json.dumps(serialized, ensure_ascii=False)

    assert serialized["intent"]["local_llm"]["extra_body"] == {"reasoning_effort": "none"}
    assert "raw-local-llm-secret" not in encoded


def test_public_settings_facade_keeps_legacy_imports_and_reads_vnext_dict() -> None:
    settings_module = import_module("puripuly_heart.config.settings")
    serialization = _serialization()

    assert settings_module.AppSettings is AppSettings
    assert settings_module.AppSettingsVNext is AppSettingsVNext
    assert "provider" in to_dict(AppSettings())
    assert set(serialization.to_dict(AppSettingsVNext())) == {"settings_version", "intent", "state"}

    vnext = replace(
        AppSettingsVNext(),
        settings_version=VNEXT_SETTINGS_SCHEMA_VERSION,
    )
    legacy = from_dict(serialization.to_dict(vnext))

    assert isinstance(legacy, AppSettings)
    assert legacy.settings_version == SETTINGS_SCHEMA_VERSION
    assert hasattr(settings_module, "load_vnext_settings")
    assert hasattr(settings_module, "save_settings_with_result")


def test_public_settings_facade_load_save_functions_are_owned_by_vnext_facade() -> None:
    settings_module = import_module("puripuly_heart.config.settings")
    facade = _facade()

    assert settings_module.FacadeSettingsLoadResult is facade.FacadeSettingsLoadResult
    assert settings_module._FacadeSettingsLoadResult is facade.FacadeSettingsLoadResult
    for name in (
        "load_settings",
        "load_settings_with_result",
        "save_settings",
        "save_settings_with_result",
        "load_vnext_settings",
        "save_vnext_settings",
    ):
        assert getattr(settings_module, name) is getattr(facade, name)
