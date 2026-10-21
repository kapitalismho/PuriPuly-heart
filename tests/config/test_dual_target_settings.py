from __future__ import annotations

from dataclasses import replace

import pytest

from puripuly_heart.config.settings_vnext import migration, serialization
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, LanguageIntent


def test_existing_settings_without_secondary_target_load_as_single_target() -> None:
    raw = serialization.to_dict(AppSettingsVNext())
    raw["intent"]["languages"].pop("secondary_target_language")

    loaded = serialization.from_dict(raw)

    assert loaded.intent.languages.target_language == "en"
    assert loaded.intent.languages.secondary_target_language == ""


def test_secondary_target_roundtrips_through_canonical_settings() -> None:
    baseline = AppSettingsVNext()
    canonical = replace(
        baseline,
        intent=replace(
            baseline.intent,
            languages=LanguageIntent(
                source_language=baseline.intent.languages.source_language,
                target_language=" zh-CN ",
                secondary_target_language=" ja ",
            ),
        ),
    )
    persisted = serialization.to_dict(canonical)
    restored = migration.from_dict(persisted)

    assert canonical.intent.languages.target_language == "zh-CN"
    assert canonical.intent.languages.secondary_target_language == "ja"
    assert persisted["intent"]["languages"]["secondary_target_language"] == "ja"
    assert restored.intent.languages.secondary_target_language == "ja"


def test_duplicate_secondary_target_normalizes_to_single_target() -> None:
    canonical = LanguageIntent(
        target_language=" ja ",
        secondary_target_language="ja",
    )

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


def test_materialize_language_selection_stores_one_recent_queue() -> None:
    from puripuly_heart.app.language_selection import LanguageSelectionChange
    from puripuly_heart.app.services.settings.settings_application import (
        materialize_language_selection,
    )

    updated = materialize_language_selection(
        AppSettingsVNext(),
        LanguageSelectionChange(
            source_code="ko",
            target_code="en",
            peer_source_code="",
            peer_target_code="",
            peer_source_mode="manual",
            recent_source_codes=("ja", "ko", "en", "fr"),
            recent_target_codes=("de", "ja", "es"),
        ),
    )

    languages = updated.intent.languages
    assert languages.recent_source_languages == ["ja", "ko", "en"]
    assert languages.recent_target_languages == ["ja", "ko", "en"]


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

    applied: list[AppSettingsVNext] = []

    class Stub:
        def __init__(self) -> None:
            self.settings = type("S", (), {"canonical": AppSettingsVNext()})()
            self.projection = type("P", (), {"render": lambda self, *a, **k: None})()

        async def apply(self, updated: AppSettingsVNext) -> None:
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

    assert applied[-1].intent.languages.secondary_target_language == "ja"
    assert applied[-1].intent.languages.target_language == "en"


def test_apply_language_selection_clears_the_secondary_target() -> None:
    import asyncio

    from puripuly_heart.app.language_selection import LanguageSelectionChange
    from puripuly_heart.app.services.settings.settings_application import (
        SettingsApplicationOwner,
    )

    applied: list[AppSettingsVNext] = []
    baseline = AppSettingsVNext()
    current = replace(
        baseline,
        intent=replace(
            baseline.intent,
            languages=replace(baseline.intent.languages, secondary_target_language="ja"),
        ),
    )

    class Stub:
        def __init__(self) -> None:
            self.settings = type("S", (), {"canonical": current})()
            self.projection = type("P", (), {"render": lambda self, *a, **k: None})()

        async def apply(self, updated: AppSettingsVNext) -> None:
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

    assert applied[-1].intent.languages.secondary_target_language == ""


@pytest.mark.parametrize("secondary", ["", "ja"])
def test_apply_language_selection_survives_validation(secondary: str) -> None:
    LanguageIntent(
        target_language="en",
        secondary_target_language=secondary,
    )


def test_peer_expected_languages_intent_stores_the_shared_recent_queue() -> None:
    from puripuly_heart.app.ports.settings_view import PeerExpectedLanguagesIntent
    from puripuly_heart.app.services.settings.settings_application import (
        materialize_immediate_settings_intent,
    )

    with_recent = materialize_immediate_settings_intent(
        AppSettingsVNext(),
        PeerExpectedLanguagesIntent(("ja", "ar", "en"), ("en", "ar", "ja", "fr")),
    )
    without_recent = materialize_immediate_settings_intent(
        AppSettingsVNext(),
        PeerExpectedLanguagesIntent(("ja",)),
    )

    assert with_recent.intent.languages.peer_expected_languages == ["ja", "ar", "en"]
    assert with_recent.intent.languages.recent_source_languages == ["en", "ar", "ja"]
    assert with_recent.intent.languages.recent_target_languages == ["en", "ar", "ja"]
    assert without_recent.intent.languages.peer_expected_languages == ["ja"]
    assert without_recent.intent.languages.recent_source_languages == ["en", "zh-CN", "ja"]


def test_recent_only_language_selection_does_not_reload_the_settings_view() -> None:
    import asyncio

    from puripuly_heart.app.language_selection import LanguageSelectionChange
    from puripuly_heart.app.services.settings.settings_application import (
        SettingsApplicationOwner,
    )

    current = AppSettingsVNext()
    renders: list[object] = []

    class Stub:
        def __init__(self) -> None:
            self.settings = type("S", (), {"canonical": current})()
            self.projection = type(
                "P",
                (),
                {"render": lambda self, *args, **kwargs: renders.append((args, kwargs))},
            )()

        async def apply(self, updated: AppSettingsVNext) -> None:
            self.settings.canonical = updated

    languages = current.intent.languages
    change = LanguageSelectionChange(
        source_code=languages.source_language,
        target_code=languages.target_language,
        secondary_target_code=languages.secondary_target_language,
        peer_source_code=languages.peer_source_language,
        peer_target_code=languages.peer_target_language,
        peer_source_mode=languages.peer_source_mode,
        recent_source_codes=("ja", "en", "zh-CN"),
        recent_target_codes=("ar", "ja"),
    )

    stub = Stub()
    asyncio.run(SettingsApplicationOwner.apply_language_selection(stub, change))

    assert renders == []
    assert stub.settings.canonical.intent.languages.recent_source_languages == [
        "ja",
        "en",
        "zh-CN",
    ]
    assert stub.settings.canonical.intent.languages.peer_expected_languages == []


def test_visible_language_selection_still_reloads_the_settings_view() -> None:
    import asyncio

    from puripuly_heart.app.language_selection import LanguageSelectionChange
    from puripuly_heart.app.services.settings.settings_application import (
        SettingsApplicationOwner,
    )

    current = AppSettingsVNext()
    renders: list[object] = []

    class Stub:
        def __init__(self) -> None:
            self.settings = type("S", (), {"canonical": current})()
            self.projection = type(
                "P",
                (),
                {"render": lambda self, *args, **kwargs: renders.append(True)},
            )()

        async def apply(self, updated: AppSettingsVNext) -> None:
            self.settings.canonical = updated

    languages = current.intent.languages
    change = LanguageSelectionChange(
        source_code="ja",
        target_code=languages.target_language,
        secondary_target_code=languages.secondary_target_language,
        peer_source_code=languages.peer_source_language,
        peer_target_code=languages.peer_target_language,
        peer_source_mode=languages.peer_source_mode,
        recent_source_codes=("ja",),
        recent_target_codes=("ja",),
    )

    asyncio.run(SettingsApplicationOwner.apply_language_selection(Stub(), change))

    assert len(renders) == 1
