from __future__ import annotations

from dataclasses import replace

import pytest

from puripuly_heart.app.services.canonical_settings_persistence import compose_settings_owner
from puripuly_heart.config.prompts import load_prompt_for_provider, resolve_system_prompt
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

    assert settings.intent.prompts.system_prompt_override is None
    assert resolve_system_prompt(settings.intent.prompts.system_prompt_override) == default_prompt


def test_first_run_settings_preserve_provider_defaults() -> None:
    settings = _new_first_run_settings("zh_CN")

    translation = settings.intent.translation
    assert settings.intent.stt.provider == "local_cpu_auto"
    assert translation.model == "index_translate_35b"
    assert translation.connection == "public_api"
    assert translation.openrouter_selected_source == "none"


@pytest.mark.parametrize("system_locale", ["en_US", "ko_KR", "ja_JP", "zh_CN", "ru_RU", None])
def test_first_run_settings_use_public_index_default(
    system_locale: str | None,
) -> None:
    settings = _new_first_run_settings(system_locale)

    translation = settings.intent.translation
    assert translation.model == "index_translate_35b"
    assert translation.connection == "public_api"


def test_first_run_settings_roundtrip_through_dict_serialization() -> None:
    settings = _new_first_run_settings("Korean_Korea.949")

    from puripuly_heart.config.settings_vnext import serialization

    restored = serialization.from_dict(serialization.to_dict(settings))

    assert restored.intent.ui.locale == "ko"
    assert restored.intent.stt.provider == "local_cpu_auto"
    assert restored.intent.translation.model == "index_translate_35b"
    assert restored.intent.translation.connection == "public_api"
    assert restored.intent.translation.openrouter_selected_source == "none"
    assert restored.intent.prompts.system_prompt_override is None


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
    assert loaded.intent.translation.model == "index_translate_35b"
    assert loaded.intent.translation.connection == "public_api"
    assert loaded.intent.translation.openrouter_selection_alias is None
    assert loaded.intent.translation.openrouter_provider_routing == "default"
    assert not path.exists()


def test_main_first_run_non_china_uses_public_index_default(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(canonical_defaults, "detect_system_locale", lambda: "ko_KR")
    path = tmp_path / "settings.json"

    loaded = _load_settings_or_default(path)

    assert loaded.intent.ui.locale == "ko"
    assert loaded.intent.translation.model == "index_translate_35b"
    assert loaded.intent.translation.connection == "public_api"
    assert not path.exists()


@pytest.mark.parametrize("system_locale", ["zh_CN", "en_US", "ko_KR", "ja_JP", "ru_RU"])
def test_first_run_locale_resolves_public_index_and_only_delayed_duplicate(
    system_locale: str,
) -> None:
    from puripuly_heart.app.wiring.wiring_llm_factory import runtime_resolution_input_from_vnext
    from puripuly_heart.config.runtime_resolution import resolve_llm_config

    config = resolve_llm_config(
        runtime_resolution_input_from_vnext(_new_first_run_settings(system_locale))
    )

    assert config.primary.provider == "index_translate"
    assert config.primary.model == "Index-Translate-35B-A3B"
    assert config.primary.credential.source == "none"
    assert config.primary.credential.reference is None
    assert not config.primary.credential.required
    assert config.fallback is not None
    assert config.fallback.target == config.primary
    assert not config.fallback.start_on_primary_error
    assert len(config.attempts) == 2
    assert [attempt.start_after_ms for attempt in config.attempts] == [0, 1300]
    assert not any(attempt.start_on_primary_error for attempt in config.attempts)


def test_main_first_run_populates_default_system_prompt(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(canonical_defaults, "detect_system_locale", lambda: None)
    path = tmp_path / "settings.json"

    loaded = _load_settings_or_default(path)

    expected = load_prompt_for_provider(LLMProviderName.GEMINI.value)
    assert loaded.intent.prompts.system_prompt_override is None
    assert resolve_system_prompt(loaded.intent.prompts.system_prompt_override) == expected
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


@pytest.mark.parametrize(
    ("system_locale", "ui_locale", "model", "connection", "alias"),
    [
        ("en_US", "en", "gemma4_26b_31b", "managed", "gemma4_26b_31b_managed"),
        ("ko_KR", "ko", "gemma4_26b_31b", "managed", "gemma4_26b_31b_managed"),
        ("ja_JP", "ja", "gemma4_26b_31b", "managed", "gemma4_26b_31b_managed"),
        ("ru_RU", "ru", "gemma4_26b_31b", "managed", "gemma4_26b_31b_managed"),
        (
            "zh_CN",
            "zh-CN",
            "deepseek_v4_flash_41",
            "managed_china",
            "deepseek_v4_flash_41_managed",
        ),
    ],
)
def test_existing_flat_settings_reset_retains_previous_regional_defaults(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    system_locale: str,
    ui_locale: str,
    model: str,
    connection: str,
    alias: str,
) -> None:
    monkeypatch.setattr(canonical_defaults, "detect_system_locale", lambda: system_locale)
    path = tmp_path / "settings.json"
    original_bytes = b'{"settings_version":30,"provider":{"llm":"qwen"}}'
    path.write_bytes(original_bytes)

    result = compose_settings_owner(path).start()

    translation = result.settings.intent.translation
    assert result.migrated
    assert result.backup_path is not None
    assert result.backup_path.read_bytes() == original_bytes
    assert result.settings.intent.ui.locale == ui_locale
    assert translation.model == model
    assert translation.connection == connection
    assert translation.openrouter_selection_alias == alias
    assert translation.openrouter_selected_source == "managed"
    assert translation.connection_history[model] == connection
    assert "index_translate_35b" not in translation.connection_history
    assert compose_settings_owner(path).start().settings.intent.translation == translation
    assert _new_first_run_settings(system_locale).intent.translation.model == "index_translate_35b"
