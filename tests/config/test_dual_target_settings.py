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


def test_language_selection_change_defaults_to_single_target() -> None:
    from puripuly_heart.app.language_selection import LanguageSelectionChange

    change = LanguageSelectionChange(
        source_code="ko",
        target_code="en",
        peer_source_code="",
        peer_target_code="",
        peer_source_mode="manual",
        recent_source_codes=(),
        recent_target_codes=(),
    )

    assert change.secondary_target_code == ""


def test_language_selection_change_carries_the_secondary_target() -> None:
    from puripuly_heart.app.language_selection import LanguageSelectionChange

    change = LanguageSelectionChange(
        source_code="ko",
        target_code="en",
        peer_source_code="",
        peer_target_code="",
        peer_source_mode="manual",
        recent_source_codes=(),
        recent_target_codes=(),
        secondary_target_code="ja",
    )

    assert change.secondary_target_code == "ja"


def test_apply_language_selection_writes_the_secondary_target() -> None:
    import asyncio

    from puripuly_heart.app.language_selection import LanguageSelectionChange
    from puripuly_heart.app.services.settings.settings_application import (
        SettingsApplicationOwner,
    )

    applied: list[AppSettings] = []

    class Stub:
        def __init__(self) -> None:
            self.settings = type("S", (), {"current": AppSettings()})()
            self.projection = type("P", (), {"render": lambda self, *a, **k: None})()

        async def apply(self, updated: AppSettings) -> None:
            applied.append(updated)

    stub = Stub()
    change = LanguageSelectionChange(
        source_code="ko",
        target_code="en",
        peer_source_code="",
        peer_target_code="",
        peer_source_mode="manual",
        recent_source_codes=(),
        recent_target_codes=(),
        secondary_target_code="ja",
    )

    asyncio.run(SettingsApplicationOwner.apply_language_selection(stub, change))

    assert applied[-1].languages.secondary_target_language == "ja"
    assert applied[-1].languages.target_language == "en"


def test_apply_language_selection_clears_the_secondary_target() -> None:
    import asyncio

    from puripuly_heart.app.language_selection import LanguageSelectionChange
    from puripuly_heart.app.services.settings.settings_application import (
        SettingsApplicationOwner,
    )

    applied: list[AppSettings] = []
    current = AppSettings()
    current.languages.secondary_target_language = "ja"

    class Stub:
        def __init__(self) -> None:
            self.settings = type("S", (), {"current": current})()
            self.projection = type("P", (), {"render": lambda self, *a, **k: None})()

        async def apply(self, updated: AppSettings) -> None:
            applied.append(updated)

    change = LanguageSelectionChange(
        source_code="ko",
        target_code="en",
        peer_source_code="",
        peer_target_code="",
        peer_source_mode="manual",
        recent_source_codes=(),
        recent_target_codes=(),
    )

    asyncio.run(SettingsApplicationOwner.apply_language_selection(Stub(), change))

    assert applied[-1].languages.secondary_target_language == ""


@pytest.mark.parametrize("secondary", ["", "ja"])
def test_apply_language_selection_survives_validation(secondary: str) -> None:
    settings = AppSettings()
    settings.languages.target_language = "en"
    settings.languages.secondary_target_language = secondary

    settings.validate()

    assert settings.languages.secondary_target_language == secondary
