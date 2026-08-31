from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from puripuly_heart.config.settings_vnext.schema import (
    VNEXT_SETTINGS_SCHEMA_VERSION,
    AppSettingsVNext,
    CaptureTargetIntent,
    PersistedOperationalState,
    ProcessCaptureTargetIntent,
    TelemetryOperationalState,
    TranslationFallbackIntent,
    with_capture_target,
    with_telemetry_enabled,
)
from tests.config.settings_migration_fixtures import (
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


def _write_json_bytes(path: Path, data: dict[str, Any]) -> bytes:
    raw_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    path.write_bytes(raw_bytes)
    return raw_bytes


def _final_dev_v30_fixture() -> dict[str, Any]:
    fixture_path = Path(__file__).parent / "fixtures" / "final_dev_v30_settings.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("settings_version", [25, VNEXT_SETTINGS_SCHEMA_VERSION + 100])
def test_canonical_migration_loader_rejects_flat_shape(settings_version: int) -> None:
    migration = _migration()
    raw = maximal_v24_settings_fixture()
    raw["settings_version"] = settings_version

    with pytest.raises(ValueError, match="canonical settings"):
        migration.from_dict(raw)


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


def test_final_dev_v30_flat_fixture_archives_then_resets_without_value_continuity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compat = _compat()
    serialization = _serialization()
    monkeypatch.setattr(compat.defaults, "detect_system_locale", lambda: "en_US")
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
    assert canonical["intent"]["translation"]["model"] == "gemma4_26b_31b"
    assert canonical["intent"]["translation"]["connection"] == "managed"
    assert canonical["intent"]["translation"]["fallback"] == {
        "enabled": True,
        "model": "gemma4_26b_31b",
        "connection": "openrouter",
        "selection_alias": "openrouter_gemma4_26b_31b",
    }
    assert canonical["intent"]["translation"]["qwen"]["region"] == "beijing"
    assert canonical["intent"]["stt"]["custom_terms"] == {}
    assert canonical["state"]["telemetry"]["anonymous_id"]
    assert canonical["state"]["telemetry"]["anonymous_id"] != raw["telemetry"]["identifier"]
    assert canonical["state"]["telemetry"]["last_sent_date_utc"] is None
    assert canonical["state"]["managed_connection"]["pending_delivery_ack_delivery_id"] is None
    assert all(
        canonical["state"]["provider_verification"][provider]["status"] == "unknown"
        for provider in PROVIDER_VERIFICATION_FIELDS
    )

    canonical_bytes = path.read_bytes()
    second = compat.load_vnext_settings(path, now=fixed_now)

    assert second.ok
    assert second.migrated is False
    assert second.backup_path is None
    assert path.read_bytes() == canonical_bytes
    assert list(tmp_path.glob("*.bak")) == [first.backup_path]


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


def test_vnext_dict_migrates_shared_qwen_audio_model_to_per_channel_provider() -> None:
    from puripuly_heart.config.settings_vnext import migration, serialization

    canonical = serialization.to_dict(AppSettingsVNext())
    canonical["intent"]["stt"]["provider"] = "qwen_asr"
    canonical["intent"]["stt"]["qwen_asr"] = {"model": "qwen-audio-3.0-asr-flash-streaming"}
    canonical["intent"]["peer_stt"]["provider"] = "qwen_asr"

    migrated = migration.from_dict(canonical)
    result = serialization.to_dict(migrated)["intent"]

    assert result["stt"]["provider"] == "qwen_audio"
    assert result["peer_stt"]["provider"] == "qwen_audio"
    assert result["stt"]["qwen_asr"]["model"] == "qwen3-asr-flash-realtime"


def test_vnext_dict_preserves_split_qwen_cloud_providers_with_leftover_audio_model() -> None:
    from puripuly_heart.config.settings_vnext import migration, serialization

    canonical = serialization.to_dict(AppSettingsVNext())
    canonical["intent"]["stt"]["provider"] = "qwen_asr"
    canonical["intent"]["stt"]["qwen_asr"] = {"model": "qwen-audio-3.0-asr-flash-streaming"}
    canonical["intent"]["peer_stt"]["provider"] = "qwen_audio"

    migrated = migration.from_dict(canonical)
    result = serialization.to_dict(migrated)["intent"]

    assert result["stt"]["provider"] == "qwen_asr"
    assert result["peer_stt"]["provider"] == "qwen_audio"
    assert result["stt"]["qwen_asr"]["model"] == "qwen3-asr-flash-realtime"


def test_vnext_dict_migrates_qwen_35_plus_nested_fields() -> None:
    from puripuly_heart.config.settings_vnext import migration, serialization

    canonical = serialization.to_dict(AppSettingsVNext())
    translation = canonical["intent"]["translation"]
    translation["model"] = "qwen35_plus"
    translation["previous_llm_model"] = "qwen35_plus"
    translation["connection"] = "official_byok"
    translation["connection_history"] = {"qwen35_plus": "official_byok"}
    translation["qwen"] = {"region": "singapore", "llm_model": "qwen3.5-plus"}

    migrated = migration.from_dict(canonical)
    result = serialization.to_dict(migrated)["intent"]["translation"]

    assert result["model"] == "qwen38_flash"
    assert result["previous_llm_model"] == "qwen38_flash"
    assert result["qwen"] == {"region": "singapore", "llm_model": "qwen3.8-flash"}
    assert result["connection_history"] == {"qwen38_flash": "official_byok"}


def test_vnext_dict_migrates_legacy_gemini_byok_alias_only() -> None:
    from puripuly_heart.config.settings_vnext import migration, serialization

    canonical = serialization.to_dict(AppSettingsVNext())
    translation = canonical["intent"]["translation"]
    translation["model"] = "gemini37_flash"
    translation["connection"] = "openrouter"
    translation["openrouter_selected_source"] = "byok"
    translation["openrouter_selection_alias"] = "gemini31_flash_lite_byok"
    translation.pop("openrouter_model", None)

    migrated = migration.from_dict(canonical)
    result = serialization.to_dict(migrated)["intent"]["translation"]

    assert result["model"] == "gemini37_flash"
    assert result["openrouter_selection_alias"] == "gemini37_flash_byok"


def test_vnext_dict_migrates_legacy_timestamp_prompt_to_new_default() -> None:
    from puripuly_heart.config.prompts import load_prompt_for_provider
    from puripuly_heart.config.settings_vnext import migration, serialization
    from puripuly_heart.config.settings_vnext.migration import LEGACY_TIMESTAMP_PROMPT

    canonical = serialization.to_dict(AppSettingsVNext())
    canonical["settings_version"] = VNEXT_SETTINGS_SCHEMA_VERSION - 1
    canonical["intent"]["prompts"]["system_prompt"] = LEGACY_TIMESTAMP_PROMPT

    migrated = migration.from_dict(canonical)

    assert migrated.intent.prompts.system_prompt == load_prompt_for_provider("gemini")


def test_vnext_dict_preserves_custom_prompt_through_migration() -> None:
    from puripuly_heart.config.settings_vnext import migration, serialization

    canonical = serialization.to_dict(AppSettingsVNext())
    canonical["settings_version"] = VNEXT_SETTINGS_SCHEMA_VERSION - 1
    canonical["intent"]["prompts"]["system_prompt"] = "my customized prompt"

    migrated = migration.from_dict(canonical)

    assert migrated.intent.prompts.system_prompt == "my customized prompt"


def test_vnext_dict_preserves_prompt_with_boundary_whitespace() -> None:
    from puripuly_heart.config.settings_vnext import migration, serialization
    from puripuly_heart.config.settings_vnext.migration import LEGACY_TIMESTAMP_PROMPT

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


@pytest.mark.parametrize(
    ("legacy_consent", "expected_enabled"),
    [
        ("allow", True),
        ("unknown", True),
        ("decline", False),
        (None, True),
        ("corrupt", False),
    ],
)
def test_v36_telemetry_states_migrate_once_to_boolean_without_legacy_surfaces(
    legacy_consent: str | None,
    expected_enabled: bool,
) -> None:
    migration = _migration()
    serialization = _serialization()
    raw = serialization.to_dict(AppSettingsVNext())
    raw["settings_version"] = 36
    raw["intent"]["telemetry"] = {} if legacy_consent is None else {"consent": legacy_consent}
    raw["state"]["telemetry"] = {
        "anonymous_id": "existing-id",
        "sent_translation_success_dates_utc": ["2026-07-01", "2026-07-03"],
    }

    serialized = serialization.to_dict(migration.from_dict(raw))

    assert serialized["intent"]["telemetry"] == {"enabled": expected_enabled}
    assert serialized["state"]["telemetry"] == (
        {"anonymous_id": "existing-id", "last_sent_date_utc": "2026-07-03"}
        if expected_enabled
        else {"anonymous_id": None, "last_sent_date_utc": None}
    )
    persisted_text = json.dumps(serialized)
    assert '"consent"' not in persisted_text
    assert "sent_translation_success_dates_utc" not in persisted_text


def test_current_schema_rejects_non_boolean_telemetry_enabled() -> None:
    migration = _migration()
    serialization = _serialization()
    raw = serialization.to_dict(AppSettingsVNext())
    raw["intent"]["telemetry"] = {"enabled": "yes"}

    with pytest.raises(
        ValueError,
        match=r"settings\.intent\.telemetry\.enabled has an invalid type",
    ):
        migration.from_dict(raw)


def test_current_schema_normalizes_legacy_telemetry_fields_without_extensions() -> None:
    migration = _migration()
    serialization = _serialization()
    raw = serialization.to_dict(AppSettingsVNext())
    raw["intent"]["telemetry"] = {"consent": "decline"}
    raw["state"]["telemetry"] = {
        "anonymous_id": "existing-id",
        "sent_translation_success_dates_utc": ["2026-07-01", "2026-07-03"],
    }

    serialized = serialization.to_dict(migration.from_dict(raw))

    assert serialized["intent"]["telemetry"] == {"enabled": False}
    assert serialized["state"]["telemetry"] == {
        "anonymous_id": None,
        "last_sent_date_utc": None,
    }
    assert "consent" not in json.dumps(serialized)
    assert "sent_translation_success_dates_utc" not in json.dumps(serialized)


@pytest.mark.parametrize("malformed", [None, []])
def test_present_malformed_telemetry_blocks_fail_closed(malformed: object) -> None:
    migration = _migration()
    serialization = _serialization()
    raw = serialization.to_dict(AppSettingsVNext())
    raw["settings_version"] = 36
    raw["intent"]["telemetry"] = malformed

    with pytest.raises(
        ValueError,
        match=r"settings\.intent\.telemetry must be a JSON object",
    ):
        migration.from_dict(raw)


def test_telemetry_enabled_transitions_manage_operational_state() -> None:
    base = AppSettingsVNext(
        state=PersistedOperationalState(
            telemetry=TelemetryOperationalState(
                anonymous_id="existing-id",
                last_sent_date_utc="2026-07-01",
            )
        )
    )

    enabled = with_telemetry_enabled(base, True, identifier_factory=lambda: "new-id")
    disabled = with_telemetry_enabled(enabled, False)
    enabled_again = with_telemetry_enabled(disabled, True, identifier_factory=lambda: "new-id")

    assert enabled.intent.telemetry.enabled is True
    assert enabled.state.telemetry.anonymous_id == "existing-id"
    assert enabled.state.telemetry.last_sent_date_utc == "2026-07-01"
    assert disabled.intent.telemetry.enabled is False
    assert disabled.state.telemetry.anonymous_id is None
    assert disabled.state.telemetry.last_sent_date_utc is None
    assert enabled_again.intent.telemetry.enabled is True
    assert enabled_again.state.telemetry.anonymous_id == "new-id"


def test_malformed_telemetry_last_sent_date_is_ignored() -> None:
    serialization = _serialization()
    raw = serialization.to_dict(AppSettingsVNext())
    raw["state"]["telemetry"] = {
        "anonymous_id": " telemetry-id ",
        "last_sent_date_utc": "bad-date",
    }

    loaded = serialization.from_dict(raw)

    assert loaded.state.telemetry.anonymous_id == "telemetry-id"
    assert loaded.state.telemetry.last_sent_date_utc is None


def test_current_vnext_dict_reads_and_serializes_idempotently() -> None:
    migration = _migration()
    serialization = _serialization()

    original = with_telemetry_enabled(
        AppSettingsVNext(),
        True,
        identifier_factory=lambda: "current-settings-test-id",
    )
    raw = serialization.to_dict(original)

    loaded = migration.from_dict(raw)
    serialized = serialization.to_dict(loaded)

    assert loaded == original
    assert serialized == raw


def test_canonical_settings_version_must_be_a_supported_integer() -> None:
    migration = _migration()
    serialization = _serialization()
    raw = serialization.to_dict(AppSettingsVNext())
    raw["settings_version"] = "not-a-schema-discriminator"

    with pytest.raises(ValueError, match="positive integer"):
        migration.from_dict(raw)

    raw["settings_version"] = VNEXT_SETTINGS_SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="unsupported canonical"):
        migration.from_dict(raw)


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


def test_current_vnext_unknown_fields_round_trip_through_compatibility_extensions(
    tmp_path: Path,
) -> None:
    serialization = _serialization()
    compat = _compat()
    raw = serialization.to_dict(AppSettingsVNext())
    raw["future_product_flag"] = True
    raw["intent"]["ui"]["future_toggle"] = "keep-me"
    path = tmp_path / "settings.json"
    _write_json_bytes(path, raw)

    persisted = serialization.to_dict(serialization.from_dict(raw))
    assert persisted["future_product_flag"] is True
    assert persisted["intent"]["ui"]["future_toggle"] == "keep-me"

    result = compat.load_vnext_settings(path)
    assert result.ok
    assert result.migrated is False
    assert result.settings is not None
    saved = serialization.to_dict(result.settings)
    assert saved["future_product_flag"] is True
    assert saved["intent"]["ui"]["future_toggle"] == "keep-me"


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


@pytest.mark.parametrize(
    "raw",
    [
        {"ui": {"locale": "ja"}, "provider": {"llm": "deepseek"}},
        {"settings_version": VNEXT_SETTINGS_SCHEMA_VERSION + 100, "unknown": "value"},
        {"settings_version": 25, "unrecognized_product_field": {"nested": True}},
        {
            "settings_version": 25,
            "secrets": {"encrypted_file_path": "retired-secrets.json"},
            "api_key_verified": {"openrouter": True},
            "openrouter_api_key": "raw-secret-must-not-survive",
        },
    ],
    ids=("ordinary", "high-version", "unknown-fields", "secret-references"),
)
def test_flat_shape_reset_matrix_discards_all_values(
    raw: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    compat = _compat()
    serialization = _serialization()
    monkeypatch.setattr(compat.defaults, "detect_system_locale", lambda: "en_US")
    path = tmp_path / "settings.json"
    original_bytes = _write_json_bytes(path, raw)
    secret_file = tmp_path / "retired-secrets.json"
    secret_file.write_bytes(b"secret-storage-sentinel")

    result = compat.load_vnext_settings(path)

    assert result.ok
    assert result.migrated is True
    assert result.settings is not None
    assert result.backup_path is not None
    assert result.backup_path.read_bytes() == original_bytes
    persisted = serialization.to_dict(result.settings)
    assert persisted["intent"]["ui"]["locale"] == "en"
    assert persisted["intent"]["translation"]["model"] == "gemma4_26b_31b"
    assert persisted["intent"]["secrets"]["encrypted_file_path"] == "secrets.json"
    assert persisted["state"]["provider_verification"]["openrouter"]["status"] == "unknown"
    assert "raw-secret-must-not-survive" not in path.read_text(encoding="utf-8")
    assert secret_file.read_bytes() == b"secret-storage-sentinel"


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


def test_reported_write_failure_restores_original_bytes_and_retains_backup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    compat = _compat()
    path = tmp_path / "settings.json"
    original_bytes = _write_json_bytes(path, maximal_v24_settings_fixture())

    def fail_after_replace(target: Path, _settings: AppSettingsVNext):
        target.write_bytes(b"partial replacement")
        return compat.VNextSettingsSaveResult(
            status=compat.SettingsPersistenceStatus.SAVE_FAILED,
            error=compat.SettingsPersistenceError(
                compat.SettingsPersistenceStatus.SAVE_FAILED,
                "save_failed:OSError",
            ),
        )

    monkeypatch.setattr(compat, "save_vnext_settings", fail_after_replace)

    result = compat.load_vnext_settings(path)

    assert result.status == compat.SettingsPersistenceStatus.SAVE_FAILED
    assert path.read_bytes() == original_bytes
    assert result.backup_path is not None
    assert result.backup_path.read_bytes() == original_bytes


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
    assert "source_shape=pre_vnext destination_shape=canonical status=success" in caplog.text
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
            "settings_version": VNEXT_SETTINGS_SCHEMA_VERSION,
            "intent": {},
            "state": {"provider_verification": []},
        },
    )

    migration_result = compat.load_vnext_settings(migration_path)
    assert migration_result.status == compat.SettingsPersistenceStatus.MIGRATION_FAILED
    assert migration_result.error is not None
    assert migration_result.error.message == "migration_failed:ValueError"
    assert "unsupported_anchor" not in migration_result.error.message
    assert migration_path.read_bytes() == migration_bytes


@pytest.mark.parametrize("original_bytes", [b"[]", b"null", b'"text"'])
def test_top_level_non_object_json_fails_without_backup_or_overwrite(
    tmp_path: Path,
    original_bytes: bytes,
) -> None:
    compat = _compat()
    path = tmp_path / "settings.json"
    path.write_bytes(original_bytes)

    result = compat.load_vnext_settings(path)

    assert result.status == compat.SettingsPersistenceStatus.PARSE_FAILED
    assert result.settings is None
    assert result.backup_path is None
    assert path.read_bytes() == original_bytes
    assert not list(tmp_path.glob("*.bak"))


@pytest.mark.parametrize(
    ("field_path", "invalid_value"),
    [
        (("intent", "ui", "locale"), []),
        (("intent", "osc", "chatbox_send"), "true"),
        (("state", "github_star_prompt", "clicked"), 1),
        (("intent", "translation"), []),
        (("intent", "translation", "fallback"), []),
        (("intent", "desktop_audio"), []),
        (("intent", "prompts"), []),
        (("intent", "translation", "concurrency_limit"), "5"),
        (("intent", "languages", "peer_expected_languages"), [123]),
        (("intent", "languages", "recent_source_languages"), [123]),
        (("intent", "translation", "connection_history"), {"gemma4_26b_31b": 123}),
        (("intent", "stt", "custom_terms"), {"ko": [123]}),
        (("state", "telemetry", "last_sent_date_utc"), [7]),
    ],
)
def test_malformed_canonical_nested_value_fails_without_backup_or_overwrite(
    tmp_path: Path,
    field_path: tuple[str, ...],
    invalid_value: object,
) -> None:
    compat = _compat()
    serialization = _serialization()
    raw = serialization.to_dict(AppSettingsVNext())
    target = raw
    for segment in field_path[:-1]:
        target = target[segment]
    target[field_path[-1]] = invalid_value
    path = tmp_path / "settings.json"
    original_bytes = _write_json_bytes(path, raw)

    result = compat.load_vnext_settings(path)

    assert result.status == compat.SettingsPersistenceStatus.MIGRATION_FAILED
    assert result.settings is None
    assert result.backup_path is None
    assert path.read_bytes() == original_bytes
    assert not list(tmp_path.glob("*.bak"))


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("source_shape", ["flat", "canonical"])
def test_nonstandard_json_constants_follow_shape_specific_cutover_policy(
    tmp_path: Path,
    constant: str,
    source_shape: str,
) -> None:
    compat = _compat()
    path = tmp_path / f"{source_shape}-{constant}.json"
    if source_shape == "canonical":
        raw_text = (
            f'{{"settings_version": {VNEXT_SETTINGS_SCHEMA_VERSION}, '
            f'"intent": {{"ui": {{"locale": {constant}}}}}, "state": {{}}}}'
        )
    else:
        raw_text = f'{{"legacy_value": {constant}}}'
    original_bytes = raw_text.encode("utf-8")
    path.write_bytes(original_bytes)

    result = compat.load_vnext_settings(path)

    if source_shape == "flat":
        assert result.status == compat.SettingsPersistenceStatus.SUCCESS
        assert result.settings is not None
        assert result.settings.intent.local_llm.extra_body == {
            "reasoning_effort": "none",
            "temperature": 0.6,
        }
        assert result.backup_path is not None
        assert result.backup_path.read_bytes() == original_bytes
        assert path.read_bytes() != original_bytes
        return

    assert result.status == compat.SettingsPersistenceStatus.PARSE_FAILED
    assert result.settings is None
    assert result.backup_path is None
    assert path.read_bytes() == original_bytes
    assert not list(tmp_path.glob("*.bak"))


def test_save_rejects_non_finite_canonical_value_without_overwrite(tmp_path: Path) -> None:
    compat = _compat()
    path = tmp_path / "settings.json"
    original_bytes = b"existing-settings"
    path.write_bytes(original_bytes)
    settings = AppSettingsVNext()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            overlay=replace(
                settings.intent.overlay,
                desktop_flet=replace(
                    settings.intent.overlay.desktop_flet,
                    visual=replace(
                        settings.intent.overlay.desktop_flet.visual,
                        background_alpha=float("nan"),
                    ),
                ),
            ),
        ),
    )

    result = compat.save_vnext_settings(path, settings)

    assert result.status == compat.SettingsPersistenceStatus.SAVE_FAILED
    assert path.read_bytes() == original_bytes
    assert not (tmp_path / "settings.json.tmp").exists()


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


def test_unsupported_future_canonical_version_fails_without_backup_or_overwrite(
    tmp_path: Path,
) -> None:
    compat = _compat()
    serialization = _serialization()
    path = tmp_path / "future-settings.json"
    raw = serialization.to_dict(AppSettingsVNext())
    raw["settings_version"] = VNEXT_SETTINGS_SCHEMA_VERSION + 1
    original_bytes = _write_json_bytes(path, raw)

    result = compat.load_vnext_settings(path)

    assert result.status == compat.SettingsPersistenceStatus.MIGRATION_FAILED
    assert result.settings is None
    assert result.backup_path is None
    assert path.read_bytes() == original_bytes
    assert not list(tmp_path.glob("*.bak"))


def test_older_vnext_version_is_backed_up_and_forward_migrated(
    tmp_path: Path,
) -> None:
    compat = _compat()
    serialization = _serialization()
    fixed_now = datetime(2026, 6, 9, 1, 2, 3, tzinfo=timezone.utc)
    path = tmp_path / "settings.json"
    settings = with_telemetry_enabled(
        AppSettingsVNext(),
        True,
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
    assert result.migrated is True
    assert result.backup_path == tmp_path / (
        f"settings.json.pre-v{VNEXT_SETTINGS_SCHEMA_VERSION - 1}.20260609T010203Z.bak"
    )
    assert result.backup_path.read_bytes() == original_bytes
    assert path.read_bytes() != original_bytes
    assert json.loads(path.read_text(encoding="utf-8"))["settings_version"] == (
        VNEXT_SETTINGS_SCHEMA_VERSION
    )
