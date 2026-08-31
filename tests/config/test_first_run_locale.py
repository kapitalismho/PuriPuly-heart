from __future__ import annotations

from dataclasses import replace

import pytest

from puripuly_heart.app.services.canonical_settings_persistence import compose_settings_owner
from puripuly_heart.config.prompts import load_prompt_for_provider
from puripuly_heart.config.provider_values import LLMProviderName
from puripuly_heart.config.settings_vnext import defaults as canonical_defaults
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.main import _load_settings_or_default
from tests.config.settings_vnext_test_helpers import assert_raw_vnext_settings_file


def _resolve_first_run_locale(system_locale: str | None) -> str:
    return canonical_defaults.resolve_first_run_ui_locale(system_locale)


def _new_first_run_settings(system_locale: str | None = None) -> AppSettingsVNext:
    return canonical_defaults.new_settings_for_first_run(system_locale)


def test_detect_system_locale_uses_locale_getlocale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(canonical_defaults.locale, "getlocale", lambda: ("Korean_Korea", "949"))

    assert canonical_defaults.detect_system_locale() == "Korean_Korea"


@pytest.mark.parametrize(
    "exc", [ValueError("bad locale"), canonical_defaults.locale.Error("bad locale")]
)
def test_first_run_settings_falls_back_to_english_when_system_locale_is_invalid(
    exc: Exception,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_invalid_locale() -> tuple[str | None, str | None]:
        raise exc

    monkeypatch.setattr(canonical_defaults.locale, "getlocale", raise_invalid_locale)

    assert canonical_defaults.detect_system_locale() is None
    assert _new_first_run_settings().intent.ui.locale == "en"


@pytest.mark.parametrize(
    "system_locale",
    ["ko", "ko-KR", "ko_KR", "KO_kr", "Korean_Korea.949"],
)
def test_first_run_locale_maps_korean_locales_to_korean(system_locale: str) -> None:
    assert _resolve_first_run_locale(system_locale) == "ko"


@pytest.mark.parametrize(
    "system_locale",
    [
        "zh",
        "zh-CN",
        "zh_CN",
        "zh-Hans",
        "zh-SG",
        "Chinese_China.936",
        "zh-TW",
        "zh-HK",
        "zh-Hant",
        "Chinese_Taiwan.950",
    ],
)
def test_first_run_locale_maps_chinese_locales_to_simplified_chinese(
    system_locale: str,
) -> None:
    assert _resolve_first_run_locale(system_locale) == "zh-CN"


@pytest.mark.parametrize(
    "system_locale",
    ["ja", "ja-JP", "ja_JP", "JA_jp", "Japanese_Japan.932"],
)
def test_first_run_locale_maps_japanese_locales_to_japanese(system_locale: str) -> None:
    assert _resolve_first_run_locale(system_locale) == "ja"


@pytest.mark.parametrize(
    "system_locale",
    ["ru", "ru-RU", "ru_RU", "RU_ru", "Russian_Russia.1251"],
)
def test_first_run_locale_maps_russian_locales_to_russian(system_locale: str) -> None:
    assert _resolve_first_run_locale(system_locale) == "ru"


@pytest.mark.parametrize("system_locale", ["en_US", "fr_FR", None])
def test_first_run_locale_defaults_to_english(system_locale: str | None) -> None:
    assert _resolve_first_run_locale(system_locale) == "en"


def test_load_settings_preserves_existing_saved_locale(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "settings.json"
    saved = AppSettingsVNext()
    saved = replace(
        saved,
        intent=replace(saved.intent, ui=replace(saved.intent.ui, locale="ja")),
    )
    owner = compose_settings_owner(path)
    owner.canonical = saved
    owner.persist()
    monkeypatch.setattr(canonical_defaults, "detect_system_locale", lambda: "ko_KR")

    loaded = compose_settings_owner(path).start().settings

    assert loaded.intent.ui.locale == "ja"
    assert assert_raw_vnext_settings_file(path)["intent"]["ui"]["locale"] == "ja"


def test_first_run_settings_preserve_prompt_defaults() -> None:
    settings = _new_first_run_settings("ko_KR")
    default_prompt = load_prompt_for_provider("gemini")

    assert settings.intent.prompts.system_prompt == default_prompt


def test_first_run_settings_preserve_provider_defaults() -> None:
    settings = _new_first_run_settings("zh_CN")

    translation = settings.intent.translation
    assert settings.intent.stt.provider == "local_cpu_auto"
    assert translation.model == "deepseek_v4_flash"
    assert translation.connection == "managed_china"
    assert translation.openrouter_selected_source == "managed"
    assert translation.fallback.enabled is True
    assert translation.fallback.model == "gemma4_26b_31b"
    assert translation.fallback.connection == "openrouter"


@pytest.mark.parametrize("system_locale", ["en_US", "ko_KR", "ja_JP", None])
def test_first_run_settings_use_openrouter_unified_gemma_fallback_default(
    system_locale: str | None,
) -> None:
    settings = _new_first_run_settings(system_locale)

    translation = settings.intent.translation
    assert translation.model == "gemma4_26b_31b"
    assert translation.connection == "managed"
    assert translation.fallback.enabled is True
    assert translation.fallback.model == "gemma4_26b_31b"
    assert translation.fallback.connection == "openrouter"


def test_first_run_settings_roundtrip_through_dict_serialization() -> None:
    settings = _new_first_run_settings("Korean_Korea.949")

    from puripuly_heart.config.settings_vnext import serialization

    restored = serialization.from_dict(serialization.to_dict(settings))

    assert restored.intent.ui.locale == "ko"
    assert restored.intent.stt.provider == "local_cpu_auto"
    assert restored.intent.translation.openrouter_selected_source == "managed"
    assert restored.intent.prompts.system_prompt == settings.intent.prompts.system_prompt


def test_first_run_settings_without_explicit_locale_detects_system_locale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(canonical_defaults.locale, "getlocale", lambda: ("zh_TW", "UTF-8"))

    settings = _new_first_run_settings()

    assert settings.intent.ui.locale == "zh-CN"


def test_settings_owner_first_run_uses_detected_system_locale(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(canonical_defaults, "detect_system_locale", lambda: "ko_KR")
    path = tmp_path / "settings.json"

    loaded = compose_settings_owner(path).start().settings

    assert loaded.intent.ui.locale == "ko"
    assert assert_raw_vnext_settings_file(path)["intent"]["ui"]["locale"] == "ko"


def test_main_first_run_uses_detected_system_locale(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(canonical_defaults, "detect_system_locale", lambda: "zh_CN")
    path = tmp_path / "settings.json"

    loaded = _load_settings_or_default(path)

    assert loaded.intent.ui.locale == "zh-CN"
    assert loaded.intent.translation.model == "deepseek_v4_flash"
    assert loaded.intent.translation.connection == "managed_china"
    assert loaded.intent.translation.openrouter_selection_alias == "deepseek_v4_flash_managed"
    assert loaded.intent.translation.openrouter_provider_routing == "deepseek_only"
    assert loaded.intent.translation.fallback.selection_alias == "openrouter_gemma4_26b_31b"
    assert not path.exists()


def test_main_first_run_non_china_uses_openrouter_unified_gemma_fallback_default(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(canonical_defaults, "detect_system_locale", lambda: "ko_KR")
    path = tmp_path / "settings.json"

    loaded = _load_settings_or_default(path)

    assert loaded.intent.ui.locale == "ko"
    assert loaded.intent.translation.model == "gemma4_26b_31b"
    assert loaded.intent.translation.connection == "managed"
    assert loaded.intent.translation.fallback.selection_alias == "openrouter_gemma4_26b_31b"
    assert not path.exists()


def test_main_first_run_populates_default_system_prompt(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(canonical_defaults, "detect_system_locale", lambda: None)
    path = tmp_path / "settings.json"

    loaded = _load_settings_or_default(path)

    expected = load_prompt_for_provider(LLMProviderName.GEMINI.value)
    assert loaded.intent.prompts.system_prompt == expected
    assert loaded.intent.prompts.system_prompt != ""
    assert not path.exists()


def test_main_existing_invalid_stable_settings_are_not_replaced_with_defaults(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stable_path = tmp_path / "puripuly-heart" / "settings.json"
    stable_path.parent.mkdir()
    stable_path.write_text("not-json", encoding="utf-8")

    with pytest.raises(RuntimeError):
        _load_settings_or_default(stable_path)

    assert stable_path.read_text(encoding="utf-8") == "not-json"


def test_settings_owner_existing_invalid_settings_are_not_replaced_with_defaults(
    tmp_path,
) -> None:
    stable_path = tmp_path / "puripuly-heart" / "settings.json"
    stable_path.parent.mkdir()
    stable_path.write_text("not-json", encoding="utf-8")

    with pytest.raises(RuntimeError):
        compose_settings_owner(stable_path).start()

    assert stable_path.read_text(encoding="utf-8") == "not-json"
