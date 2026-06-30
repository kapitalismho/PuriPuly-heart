from __future__ import annotations

import json

import pytest

from puripuly_heart.config import settings as settings_module
from puripuly_heart.config.prompts import load_prompt_for_provider
from puripuly_heart.config.settings import (
    AppSettings,
    LLMProviderName,
    OpenRouterCredentialSource,
    OpenRouterFallbackSelectionAlias,
    OpenRouterLLMModel,
    OpenRouterProviderRouting,
    OpenRouterSelectionAlias,
    STTProviderName,
    TranslationConnection,
    TranslationModel,
    from_dict,
    load_settings,
    save_settings,
    to_dict,
)
from puripuly_heart.main import _load_settings_or_default
from puripuly_heart.ui.controller import GuiController


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
    assert json.loads(path.read_text(encoding="utf-8"))["ui"]["locale"] == "ja"


def test_first_run_settings_preserve_prompt_defaults() -> None:
    settings = _new_first_run_settings("ko_KR")
    default_prompt = load_prompt_for_provider("gemini")

    assert settings.system_prompt == default_prompt
    assert settings.system_prompts == {}


def test_first_run_china_locale_preserves_prompt_defaults() -> None:
    settings = _new_first_run_settings("zh_CN")
    default_prompt = load_prompt_for_provider("gemini")

    assert settings.system_prompt == default_prompt
    assert settings.system_prompts == {}


def test_first_run_settings_preserve_provider_defaults() -> None:
    settings = _new_first_run_settings("zh_CN")

    assert settings.provider.stt == STTProviderName.LOCAL_QWEN
    assert settings.provider.llm == LLMProviderName.OPENROUTER
    assert settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED


def test_first_run_settings_roundtrip_through_dict_serialization() -> None:
    settings = _new_first_run_settings("Korean_Korea.949")

    restored = from_dict(to_dict(settings))

    assert restored.ui.locale == "ko"
    assert restored.provider.stt == STTProviderName.LOCAL_QWEN
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


def test_controller_first_run_uses_detected_system_locale(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings_module, "detect_system_locale", lambda: "ko_KR", raising=False)
    path = tmp_path / "settings.json"
    controller = GuiController(page=object(), app=object(), config_path=path)

    loaded = controller._load_or_init_settings(path)

    assert loaded.ui.locale == "ko"
    assert json.loads(path.read_text(encoding="utf-8"))["ui"]["locale"] == "ko"


def test_main_first_run_uses_detected_system_locale(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings_module, "detect_system_locale", lambda: "zh_CN", raising=False)
    path = tmp_path / "settings.json"

    loaded = _load_settings_or_default(path)

    assert loaded.ui.locale == "zh-CN"
    assert loaded.translation.connection == TranslationConnection.MANAGED_CHINA
    assert loaded.openrouter.provider_routing == OpenRouterProviderRouting.DEEPSEEK_ONLY
    assert loaded.openrouter.llm_model == OpenRouterLLMModel.DEEPSEEK_V4_FLASH
    assert not path.exists()


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
def test_first_run_china_locale_selects_managed_china_defaults(
    system_locale: str,
) -> None:
    settings = _new_first_run_settings(system_locale)

    assert settings.ui.locale == "zh-CN"
    assert settings.translation.model == TranslationModel.DEEPSEEK_V4_FLASH
    assert settings.translation.connection == TranslationConnection.MANAGED_CHINA
    assert settings.openrouter.llm_model == OpenRouterLLMModel.DEEPSEEK_V4_FLASH
    assert settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert settings.openrouter.selection_alias == OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED
    assert settings.openrouter.provider_routing == OpenRouterProviderRouting.DEEPSEEK_ONLY
    assert (
        settings.openrouter.fallback_selection_alias
        == OpenRouterFallbackSelectionAlias.DEEPSEEK_V4_FLASH_CHINA
    )
    assert (
        settings.translation.connection_history.get(TranslationModel.DEEPSEEK_V4_FLASH.value)
        == TranslationConnection.MANAGED_CHINA
    )


@pytest.mark.parametrize("system_locale", ["ko_KR", "ja_JP", "en_US", "fr_FR"])
def test_first_run_non_china_locale_preserves_default_managed_connection(
    system_locale: str,
) -> None:
    settings = _new_first_run_settings(system_locale)

    assert settings.translation.model == TranslationModel.GEMMA4
    assert settings.translation.connection == TranslationConnection.MANAGED
    assert settings.openrouter.provider_routing == OpenRouterProviderRouting.DEFAULT
    assert settings.openrouter.llm_model == OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
    assert settings.openrouter.selection_alias == OpenRouterSelectionAlias.GEMMA4_MANAGED


def test_first_run_china_locale_managed_china_roundtrip_through_dict() -> None:
    settings = _new_first_run_settings("zh_CN")

    restored = from_dict(to_dict(settings))

    assert restored.ui.locale == "zh-CN"
    assert restored.translation.model == TranslationModel.DEEPSEEK_V4_FLASH
    assert restored.translation.connection == TranslationConnection.MANAGED_CHINA
    assert restored.openrouter.provider_routing == OpenRouterProviderRouting.DEEPSEEK_ONLY
    assert restored.openrouter.llm_model == OpenRouterLLMModel.DEEPSEEK_V4_FLASH
    assert restored.openrouter.selection_alias == OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED
    assert (
        restored.openrouter.fallback_selection_alias
        == OpenRouterFallbackSelectionAlias.DEEPSEEK_V4_FLASH_CHINA
    )


def test_first_run_china_locale_saved_and_loaded_preserves_managed_china(
    tmp_path,
) -> None:
    path = tmp_path / "settings.json"
    settings = _new_first_run_settings("zh_CN")
    save_settings(path, settings)

    loaded = load_settings(path)

    assert loaded.ui.locale == "zh-CN"
    assert loaded.translation.connection == TranslationConnection.MANAGED_CHINA
    assert loaded.translation.model == TranslationModel.DEEPSEEK_V4_FLASH
    assert loaded.openrouter.provider_routing == OpenRouterProviderRouting.DEEPSEEK_ONLY
    assert loaded.openrouter.selection_alias == OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED


def test_first_run_china_locale_controller_persists_managed_china(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings_module, "detect_system_locale", lambda: "zh_CN", raising=False)
    path = tmp_path / "settings.json"
    controller = GuiController(page=object(), app=object(), config_path=path)

    loaded = controller._load_or_init_settings(path)

    assert loaded.ui.locale == "zh-CN"
    assert loaded.translation.connection == TranslationConnection.MANAGED_CHINA
    assert loaded.openrouter.provider_routing == OpenRouterProviderRouting.DEEPSEEK_ONLY
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["ui"]["locale"] == "zh-CN"
    assert persisted["translation"]["connection"] == TranslationConnection.MANAGED_CHINA.value
    assert (
        persisted["openrouter"]["provider_routing"] == OpenRouterProviderRouting.DEEPSEEK_ONLY.value
    )


def test_existing_settings_not_overwritten_by_china_locale_detection(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings_module, "detect_system_locale", lambda: "zh_CN", raising=False)
    path = tmp_path / "settings.json"
    saved = AppSettings()
    saved.ui.locale = "en"
    save_settings(path, saved)

    loaded = load_settings(path)

    assert loaded.ui.locale == "en"
    assert loaded.translation.connection == TranslationConnection.MANAGED
    assert loaded.openrouter.provider_routing == OpenRouterProviderRouting.DEFAULT
