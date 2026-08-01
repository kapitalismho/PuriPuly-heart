from __future__ import annotations

import pytest

from puripuly_heart.app.services.canonical_settings_persistence import compose_settings_owner
from puripuly_heart.config import settings as settings_module
from puripuly_heart.config.prompts import load_prompt_for_provider
from puripuly_heart.config.settings import (
    AppSettings,
    LLMProviderName,
    OpenRouterCredentialSource,
    STTProviderName,
    TranslationConnection,
    TranslationModel,
    from_dict,
    load_settings,
    save_settings,
    to_dict,
)
from puripuly_heart.main import _load_settings_or_default
from tests.config.settings_vnext_test_helpers import legacy_projected_settings_file


def _resolve_first_run_locale(system_locale: str | None) -> str:
    assert hasattr(settings_module, "resolve_first_run_ui_locale")
    return settings_module.resolve_first_run_ui_locale(system_locale)


def _new_first_run_settings(system_locale: str | None = None) -> AppSettings:
    assert hasattr(settings_module, "new_settings_for_first_run")
    return settings_module.new_settings_for_first_run(system_locale)


def test_detect_system_locale_uses_locale_getlocale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings_module.locale, "getlocale", lambda: ("Korean_Korea", "949"))

    assert settings_module.detect_system_locale() == "Korean_Korea"


@pytest.mark.parametrize(
    "exc", [ValueError("bad locale"), settings_module.locale.Error("bad locale")]
)
def test_first_run_settings_falls_back_to_english_when_system_locale_is_invalid(
    exc: Exception,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_invalid_locale() -> tuple[str | None, str | None]:
        raise exc

    monkeypatch.setattr(settings_module.locale, "getlocale", raise_invalid_locale)

    assert settings_module.detect_system_locale() is None
    assert _new_first_run_settings().ui.locale == "en"


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
    saved = AppSettings()
    saved.ui.locale = "ja"
    save_settings(path, saved)
    monkeypatch.setattr(settings_module, "detect_system_locale", lambda: "ko_KR", raising=False)

    loaded = load_settings(path)

    assert loaded.ui.locale == "ja"
    assert legacy_projected_settings_file(path)["ui"]["locale"] == "ja"


def test_first_run_settings_preserve_prompt_defaults() -> None:
    settings = _new_first_run_settings("ko_KR")
    default_prompt = load_prompt_for_provider("gemini")

    assert settings.system_prompt == default_prompt
    assert settings.system_prompts == {}


def test_first_run_settings_preserve_provider_defaults() -> None:
    settings = _new_first_run_settings("zh_CN")

    assert settings.provider.stt == STTProviderName.LOCAL_CPU_AUTO
    assert settings.provider.llm == LLMProviderName.OPENROUTER
    assert settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert settings.translation.model == TranslationModel.DEEPSEEK_V4_FLASH
    assert settings.translation.connection == TranslationConnection.MANAGED_CHINA
    assert settings.translation.fallback.enabled is True
    assert settings.translation.fallback.model == TranslationModel.GEMMA4
    assert settings.translation.fallback.connection == TranslationConnection.OPENROUTER


@pytest.mark.parametrize("system_locale", ["en_US", "ko_KR", "ja_JP", None])
def test_first_run_settings_use_openrouter_deepseek_fallback_default(
    system_locale: str | None,
) -> None:
    settings = _new_first_run_settings(system_locale)

    assert settings.provider.llm == LLMProviderName.OPENROUTER
    assert settings.translation.model == TranslationModel.GEMMA4_26B_31B
    assert settings.translation.connection == TranslationConnection.MANAGED
    assert settings.translation.fallback.enabled is True
    assert settings.translation.fallback.model == TranslationModel.DEEPSEEK_V4_FLASH
    assert settings.translation.fallback.connection == TranslationConnection.OPENROUTER


def test_first_run_settings_roundtrip_through_dict_serialization() -> None:
    settings = _new_first_run_settings("Korean_Korea.949")

    restored = from_dict(to_dict(settings))

    assert restored.ui.locale == "ko"
    assert restored.provider.stt == STTProviderName.LOCAL_CPU_AUTO
    assert restored.provider.llm == LLMProviderName.OPENROUTER
    assert restored.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert restored.system_prompt == settings.system_prompt
    assert restored.system_prompts == {}


def test_first_run_settings_without_explicit_locale_detects_system_locale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings_module.locale, "getlocale", lambda: ("zh_TW", "UTF-8"))

    settings = _new_first_run_settings()

    assert settings.ui.locale == "zh-CN"


def test_settings_owner_first_run_uses_detected_system_locale(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings_module, "detect_system_locale", lambda: "ko_KR", raising=False)
    path = tmp_path / "settings.json"

    loaded = compose_settings_owner(path).start().settings

    assert loaded.ui.locale == "ko"
    assert legacy_projected_settings_file(path)["ui"]["locale"] == "ko"


def test_main_first_run_uses_detected_system_locale(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings_module, "detect_system_locale", lambda: "zh_CN", raising=False)
    path = tmp_path / "settings.json"

    loaded = _load_settings_or_default(path)

    assert loaded.intent.ui.locale == "zh-CN"
    assert loaded.intent.translation.model == "deepseek_v4_flash"
    assert loaded.intent.translation.connection == "managed_china"
    assert loaded.intent.translation.openrouter_selection_alias == "deepseek_v4_flash_managed"
    assert loaded.intent.translation.openrouter_provider_routing == "deepseek_only"
    assert loaded.intent.translation.fallback.selection_alias == "openrouter_gemma4_26b_a4b"
    assert not path.exists()


def test_main_first_run_non_china_uses_openrouter_deepseek_fallback_default(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings_module, "detect_system_locale", lambda: "ko_KR", raising=False)
    path = tmp_path / "settings.json"

    loaded = _load_settings_or_default(path)

    assert loaded.intent.ui.locale == "ko"
    assert loaded.intent.translation.model == "gemma4_26b_31b"
    assert loaded.intent.translation.connection == "managed"
    assert loaded.intent.translation.fallback.selection_alias == "openrouter_deepseek_v4_flash"
    assert not path.exists()


def test_main_first_run_populates_default_system_prompt(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings_module, "detect_system_locale", lambda: None, raising=False)
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
