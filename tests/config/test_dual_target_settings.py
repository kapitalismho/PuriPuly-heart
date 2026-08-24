from __future__ import annotations

import pytest

from puripuly_heart.config.settings import AppSettings, from_dict, to_dict
from puripuly_heart.config.settings_vnext import migration, serialization
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, LanguageIntent


def test_existing_settings_without_secondary_target_load_as_single_target() -> None:
    raw = to_dict(AppSettings())
    raw["languages"].pop("secondary_target_language")

    loaded = from_dict(raw)

    assert loaded.languages.target_language == "en"
    assert loaded.languages.secondary_target_language == ""


def test_secondary_target_roundtrips_through_canonical_and_compatibility_settings() -> None:
    legacy = AppSettings()
    legacy.languages.target_language = " zh-CN "
    legacy.languages.secondary_target_language = " ja "
    legacy.validate()

    canonical = migration.from_legacy_app_settings(legacy)
    persisted = serialization.to_dict(canonical)
    restored = migration.from_dict(persisted)
    projected = migration.to_legacy_dict(restored)

    assert canonical.intent.languages.target_language == "zh-CN"
    assert canonical.intent.languages.secondary_target_language == "ja"
    assert persisted["intent"]["languages"]["secondary_target_language"] == "ja"
    assert projected["languages"]["secondary_target_language"] == "ja"


def test_duplicate_secondary_target_normalizes_to_single_target() -> None:
    legacy = AppSettings()
    legacy.languages.target_language = " ja "
    legacy.languages.secondary_target_language = "ja"

    legacy.validate()
    canonical = LanguageIntent(
        target_language=" ja ",
        secondary_target_language="ja",
    )

    assert legacy.languages.target_language == "ja"
    assert legacy.languages.secondary_target_language == ""
    assert canonical.target_language == "ja"
    assert canonical.secondary_target_language == ""


def test_canonical_settings_reject_empty_primary_target() -> None:
    with pytest.raises(ValueError, match="target_language must be non-empty"):
        LanguageIntent(target_language="   ", secondary_target_language="ja")


def test_current_vnext_without_secondary_target_loads_with_empty_default() -> None:
    raw = serialization.to_dict(AppSettingsVNext())
    raw["intent"]["languages"].pop("secondary_target_language")

    loaded = serialization.from_dict(raw)

    assert loaded.intent.languages.secondary_target_language == ""
