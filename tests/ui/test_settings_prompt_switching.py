from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("flet")

from puripuly_heart.config.prompts import load_prompt_for_provider
from puripuly_heart.config.settings import (
    AppSettings,
    GeminiLLMModel,
    LLMProviderName,
    OpenRouterSelectionAlias,
    QwenLLMModel,
    STTProviderName,
    TranslationConnection,
    TranslationFallbackSettings,
    TranslationModel,
    TranslationSettings,
)
from puripuly_heart.ui import i18n as i18n_module
from puripuly_heart.ui.i18n import provider_label, t
from puripuly_heart.ui.views import settings as settings_view
from tests.helpers.flet_page import attach_dummy_page


class DummySecretStore:
    def get(self, _key: str) -> str | None:
        return None


def _make_settings_view(monkeypatch):
    monkeypatch.setattr(settings_view.SettingsView, "_populate_host_apis", lambda self: None)
    monkeypatch.setattr(settings_view.SettingsView, "_refresh_microphones", lambda self: None)
    monkeypatch.setattr(settings_view.SettingsView, "update", lambda self: None)
    monkeypatch.setattr(
        settings_view, "create_secret_store", lambda *args, **kwargs: DummySecretStore()
    )
    return settings_view.SettingsView()


def test_settings_view_loads_qwen_prompt(monkeypatch) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.QWEN

    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert view._prompt_editor.value == load_prompt_for_provider("qwen")
    assert settings.system_prompt == view._prompt_editor.value
    assert settings.system_prompts == {}


def test_settings_view_switches_prompt_on_llm_change(monkeypatch) -> None:
    settings = AppSettings()

    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert view._prompt_editor.value == load_prompt_for_provider("openrouter")
    assert view._prompt_for_text.value == t(
        "settings.prompt_for",
        provider=provider_label(LLMProviderName.OPENROUTER.value),
    )

    view._on_llm_selected(TranslationModel.QWEN_35_PLUS.value)
    pending = view.build_provider_apply_settings()

    assert view._prompt_editor.value == load_prompt_for_provider("qwen")
    assert view._prompt_for_text.value == t(
        "settings.prompt_for",
        provider=provider_label(LLMProviderName.QWEN.value),
    )
    assert settings.provider.llm == LLMProviderName.OPENROUTER
    assert pending is not None
    assert pending.provider.llm == LLMProviderName.QWEN
    assert pending.qwen.llm_model == QwenLLMModel.QWEN_35_PLUS

    view._on_llm_selected(TranslationModel.LOCAL_LLM.value)
    pending = view.build_provider_apply_settings()

    assert view._prompt_editor.value == load_prompt_for_provider("local_llm")
    assert view._prompt_for_text.value == t(
        "settings.prompt_for",
        provider=provider_label(LLMProviderName.LOCAL_LLM.value),
    )
    assert settings.provider.llm == LLMProviderName.OPENROUTER
    assert pending is not None
    assert pending.provider.llm == LLMProviderName.LOCAL_LLM

    view._on_llm_selected(TranslationModel.GEMINI_37_FLASH.value)
    pending = view.build_provider_apply_settings()

    assert view._prompt_editor.value == load_prompt_for_provider("gemini")
    assert view._prompt_for_text.value == t(
        "settings.prompt_for",
        provider=provider_label(LLMProviderName.GEMINI.value),
    )
    assert settings.provider.llm == LLMProviderName.OPENROUTER
    assert pending is not None
    assert pending.provider.llm == LLMProviderName.GEMINI

    view._on_llm_selected(TranslationModel.GEMMA4.value)
    pending = view.build_provider_apply_settings()

    assert view._prompt_editor.value == load_prompt_for_provider("openrouter")
    assert view._prompt_for_text.value == t(
        "settings.prompt_for",
        provider=provider_label(LLMProviderName.OPENROUTER.value),
    )
    assert settings.provider.llm == LLMProviderName.OPENROUTER
    assert pending is not None
    assert pending.provider.llm == LLMProviderName.OPENROUTER


def test_deepseek_managed_and_fallback_keep_single_prompt(monkeypatch) -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.GEMINI_37_FLASH,
        connection=TranslationConnection.OFFICIAL_BYOK,
    )
    settings.provider.llm = LLMProviderName.GEMINI
    settings.system_prompts = {
        "gemini": "GEMINI CUSTOM",
        "openrouter": "OPENROUTER CUSTOM",
        "qwen": "QWEN CUSTOM",
    }
    settings.system_prompt = "GEMINI CUSTOM"

    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    view._on_llm_selected(TranslationModel.DEEPSEEK_V4_FLASH.value)
    pending = view.build_provider_apply_settings()

    assert view._prompt_editor.value == "GEMINI CUSTOM"
    assert view._prompt_for_text.value == t(
        "settings.prompt_for",
        provider=provider_label(LLMProviderName.OPENROUTER.value),
    )
    assert pending is not None
    assert pending.translation.model == TranslationModel.DEEPSEEK_V4_FLASH
    assert pending.translation.connection == TranslationConnection.MANAGED
    assert pending.provider.llm == LLMProviderName.OPENROUTER
    assert pending.openrouter.selection_alias == OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED
    assert pending.system_prompt == "GEMINI CUSTOM"

    view._on_openrouter_fallback_selected("openrouter_deepseek_v4_flash")
    pending = view.build_provider_apply_settings()

    assert view._prompt_editor.value == "GEMINI CUSTOM"
    assert view._prompt_for_text.value == t(
        "settings.prompt_for",
        provider=provider_label(LLMProviderName.OPENROUTER.value),
    )
    assert pending is not None
    assert pending.translation.fallback == TranslationFallbackSettings(
        enabled=True,
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.OPENROUTER,
    )
    assert pending.system_prompt == "GEMINI CUSTOM"


def test_prompt_tab_labels_and_tag_editor_copy_render_from_i18n(monkeypatch) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.QWEN
    settings.languages.source_language = "en"

    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    previous_locale = i18n_module.get_locale()
    try:
        i18n_module.set_locale("ko")
        view.apply_locale()

        assert view._persona_title.value == t("settings.section.persona")
        assert view._custom_vocab_title.value == t("settings.section.custom_vocabulary")
        assert view._prompt_for_text.value == t(
            "settings.prompt_for",
            provider=provider_label(LLMProviderName.QWEN.value),
        )
        assert view._custom_vocab_description_text.value == t(
            "settings.custom_vocabulary.description"
        )
        assert view._custom_vocab_tag_editor._input_field.hint_text == ""  # noqa: SLF001
    finally:
        i18n_module.set_locale(previous_locale)


def test_settings_view_shows_qwen_model_label(monkeypatch) -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.QWEN_35_PLUS,
        connection=TranslationConnection.OFFICIAL_BYOK,
    )
    settings.provider.llm = LLMProviderName.QWEN
    settings.qwen.llm_model = QwenLLMModel.QWEN_35_PLUS

    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert view._llm_text.content.value == t("provider.qwen35_plus")


def test_settings_view_uses_single_prompt_across_provider_switches(monkeypatch) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.GEMINI
    settings.system_prompts = {
        "gemini": "GEMINI CUSTOM",
        "openrouter": "OPENROUTER CUSTOM",
        "qwen": "QWEN CUSTOM",
    }
    settings.system_prompt = "GEMINI CUSTOM"

    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert view._prompt_editor.value == "GEMINI CUSTOM"

    view._on_llm_selected(TranslationModel.QWEN_35_PLUS.value)
    pending = view.build_provider_apply_settings()
    assert view._prompt_editor.value == "GEMINI CUSTOM"
    assert settings.system_prompt == "GEMINI CUSTOM"
    assert pending is not None
    assert pending.system_prompt == "GEMINI CUSTOM"

    view._on_prompt_change("QWEN EDITED")
    pending = view.build_provider_apply_settings()
    assert settings.system_prompts == {}
    assert pending is not None
    assert pending.system_prompt == "QWEN EDITED"
    assert pending.system_prompts == {}

    view._on_llm_selected(TranslationModel.GEMINI_37_FLASH.value)
    pending = view.build_provider_apply_settings()
    assert view._prompt_editor.value == "QWEN EDITED"
    assert settings.system_prompt == "GEMINI CUSTOM"
    assert pending is not None
    assert pending.system_prompt == "QWEN EDITED"

    view._on_llm_selected(TranslationModel.GEMMA4.value)
    pending = view.build_provider_apply_settings()
    assert view._prompt_editor.value == "QWEN EDITED"
    assert settings.system_prompt == "GEMINI CUSTOM"
    assert pending is not None
    assert pending.system_prompt == "QWEN EDITED"


def test_prompt_draft_survives_provider_round_trip_until_commit(monkeypatch) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.GEMINI
    settings.system_prompts = {
        "gemini": "GEMINI CUSTOM",
        "qwen": "QWEN CUSTOM",
    }
    settings.system_prompt = "GEMINI CUSTOM"

    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    view._on_prompt_change("GEMINI DRAFT")
    view._on_llm_selected(TranslationModel.QWEN_35_PLUS.value)
    view._on_llm_selected(TranslationModel.GEMINI_37_FLASH.value)

    assert view._prompt_editor.value == "GEMINI DRAFT"
    assert settings.system_prompt == "GEMINI CUSTOM"


def test_single_prompt_whitespace_survives_provider_switch(monkeypatch) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.GEMINI
    settings.system_prompt = "  CUSTOM PROMPT\n"

    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    view._on_llm_selected(TranslationModel.QWEN_35_PLUS.value)
    pending = view.build_provider_apply_settings()

    assert view._prompt_editor.value == "  CUSTOM PROMPT\n"
    assert pending is not None
    assert pending.system_prompt == "  CUSTOM PROMPT\n"
    assert pending.system_prompts == {}


def test_prompt_commit_uses_prompt_apply_callback_without_generic_settings_emit(
    monkeypatch,
) -> None:
    settings = AppSettings()
    prompt_applied: list[AppSettings] = []
    generic_changed: list[AppSettings] = []

    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    view.on_prompt_apply_settings = lambda incoming: prompt_applied.append(incoming)
    view.on_settings_changed = lambda incoming: generic_changed.append(incoming)

    view._on_prompt_change("custom prompt")
    view._on_prompt_commit("custom prompt")

    assert view.has_pending_prompt_changes is False
    assert len(prompt_applied) == 1
    assert prompt_applied[0].system_prompt == "custom prompt"
    assert prompt_applied[0].system_prompts == {}
    assert generic_changed == []


def test_settings_view_llm_modal_lists_logical_translation_models_once(monkeypatch) -> None:
    settings = AppSettings()
    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    attach_dummy_page(monkeypatch, view)

    captured: dict[str, object] = {}

    class DummyModal:
        def __init__(
            self,
            _page,
            _title,
            options,
            _on_select,
            *,
            show_description=False,
            two_column=False,
            left_column_sections=1,
        ):
            captured["options"] = options
            captured["show_description"] = show_description
            captured["two_column"] = two_column
            captured["left_column_sections"] = left_column_sections

        def open(self, current: str) -> None:
            captured["current"] = current

    monkeypatch.setattr(settings_view, "SettingsModal", DummyModal)

    view._on_llm_click(None)

    assert captured["show_description"] is True
    assert captured["two_column"] is True
    assert captured["left_column_sections"] == 2
    options = captured["options"]
    values = [option.value for option in options]

    assert values == [
        TranslationModel.GEMMA4_26B_31B.value,
        TranslationModel.GEMMA4_31B.value,
        TranslationModel.DEEPSEEK_V4_FLASH.value,
        "managed_gemma_cpu",
        "managed_gemma_gpu",
        TranslationModel.GEMMA4.value,
        TranslationModel.LOCAL_LLM.value,
        TranslationModel.CUSTOM_HTTP.value,
        TranslationModel.GEMINI_37_FLASH.value,
        TranslationModel.GEMINI_31_FLASH_LITE.value,
        TranslationModel.QWEN_35_PLUS.value,
    ]
    assert TranslationModel.QWEN_35_PLUS.value in values
    assert TranslationModel.LOCAL_LLM.value in values
    assert all("qwen35_flash" not in value for value in values)
    assert len(values) == len(set(values))
    assert TranslationModel.MANAGED_GEMMA.value not in values

    managed = {option.value: option for option in options}
    assert managed["managed_gemma_cpu"].label == t("provider.managed_gemma_cpu")
    assert managed["managed_gemma_cpu"].description == t(
        "settings.translation_model.managed_gemma_cpu.description"
    )
    assert managed["managed_gemma_gpu"].label == t("provider.managed_gemma_gpu")
    assert managed["managed_gemma_gpu"].description == t(
        "settings.translation_model.managed_gemma_gpu.description"
    )
    assert managed["managed_gemma_cpu"].section == t(
        "settings.translation_model.section.recommended_local"
    )
    assert managed["managed_gemma_gpu"].section == t(
        "settings.translation_model.section.gpu_inference"
    )

    gemma31 = next(
        option for option in options if option.value == TranslationModel.GEMMA4_31B.value
    )
    assert gemma31.section == t("settings.translation_model.section.recommended_cloud")
    assert gemma31.description == t("settings.translation_model.gemma4_31b.description")

    sections: list[str] = []
    for option in options:
        if option.section and option.section not in sections:
            sections.append(option.section)
    assert sections == [
        t("settings.translation_model.section.recommended_cloud"),
        t("settings.translation_model.section.recommended_local"),
        t("settings.translation_model.section.gpu_inference"),
        t("settings.translation_model.section.gemma"),
        t("settings.translation_model.section.user_settings"),
        t("settings.translation_model.section.others"),
    ]


def test_gemma31_connection_modal_lists_managed_openrouter_and_cerebras(monkeypatch) -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.GEMMA4_31B,
        connection=TranslationConnection.OPENROUTER,
        connection_history={
            TranslationModel.GEMMA4_31B.value: TranslationConnection.OPENROUTER,
        },
    )
    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    attach_dummy_page(monkeypatch, view)
    captured: dict[str, object] = {}

    class DummyModal:
        def __init__(self, _page, _title, options, _on_select, **_kwargs):
            captured["options"] = options
            captured["show_description"] = _kwargs.get("show_description")

        def open(self, current: str) -> None:
            captured["current"] = current

    monkeypatch.setattr(settings_view, "SettingsModal", DummyModal)

    view._on_translation_connection_click(None)

    assert [option.value for option in captured["options"]] == [
        TranslationConnection.MANAGED.value,
        TranslationConnection.OPENROUTER.value,
        TranslationConnection.CEREBRAS.value,
    ]
    assert captured["current"] == TranslationConnection.OPENROUTER.value
    assert captured["show_description"] is True
    assert captured["options"][0].description == ""
    assert captured["options"][1].description == ""
    assert captured["options"][2].description == t(
        "settings.translation_connection.cerebras.description"
    )


def test_gemma31_cerebras_connection_materializes_provider_and_key_visibility(monkeypatch) -> None:
    settings = AppSettings()
    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    view._on_llm_selected(TranslationModel.GEMMA4_31B.value)
    view._on_translation_connection_selected(TranslationConnection.CEREBRAS.value)
    pending = view.build_provider_apply_settings()

    assert pending is not None
    assert pending.translation.model == TranslationModel.GEMMA4_31B
    assert pending.translation.connection == TranslationConnection.CEREBRAS
    assert pending.provider.llm == LLMProviderName.CEREBRAS
    assert view._cerebras_key.visible is True
    assert view._openrouter_key.visible is False


def test_settings_view_updates_gemini_model_without_provider_switch(monkeypatch) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.GEMINI
    settings.gemini.llm_model = GeminiLLMModel.GEMINI_37_FLASH
    settings.system_prompts = {
        "gemini": "GEMINI CUSTOM",
        "qwen": "QWEN CUSTOM",
    }
    settings.system_prompt = "GEMINI CUSTOM"

    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    view._on_llm_selected(TranslationModel.GEMINI_31_FLASH_LITE.value)
    pending = view.build_provider_apply_settings()

    assert settings.provider.llm == LLMProviderName.GEMINI
    assert settings.gemini.llm_model == GeminiLLMModel.GEMINI_37_FLASH
    assert pending is not None
    assert pending.gemini.llm_model == GeminiLLMModel.GEMINI_31_FLASH_LITE
    assert settings.system_prompt == "GEMINI CUSTOM"
    assert view._prompt_editor.value == "GEMINI CUSTOM"


def test_settings_view_toggles_qwen_region_visibility_with_stt_provider(monkeypatch) -> None:
    settings = AppSettings()
    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert view._qwen_region_btn.visible is False

    view._on_stt_selected(STTProviderName.QWEN_ASR.value)
    assert view._qwen_region_btn.visible is True

    view._on_stt_selected(STTProviderName.DEEPGRAM.value)
    assert view._qwen_region_btn.visible is False
