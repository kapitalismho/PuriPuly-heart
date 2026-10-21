from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("flet")

from puripuly_heart.app.services.settings_secrets import SettingsSecretsOwner
from puripuly_heart.app.wiring.wiring_provider_runtime_policy import (
    provider_llm_for_translation,
)
from puripuly_heart.config.prompts import load_prompt_for_provider
from puripuly_heart.config.provider_values import (
    GeminiLLMModel,
    LLMProviderName,
    OpenRouterSelectionAlias,
    QwenLLMModel,
    STTProviderName,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.config.translation_values import TranslationConnection, TranslationModel
from puripuly_heart.ui import i18n as i18n_module
from puripuly_heart.ui.i18n import t
from puripuly_heart.ui.views import settings as settings_view
from tests.helpers.flet_page import attach_dummy_page


class DummySecretStore:
    def get(self, _key: str) -> str | None:
        return None


def _settings(
    *,
    model: str | None = None,
    connection: str | None = None,
    prompt: str = "",
    source_language: str | None = None,
    history: dict[str, str] | None = None,
) -> AppSettingsVNext:
    settings = AppSettingsVNext()
    translation = settings.intent.translation
    if model is not None or connection is not None or history is not None:
        gemini = translation.gemini
        if model == "gemini_flash":
            gemini = replace(gemini, llm_model="gemini-3.8-flash")
        translation = replace(
            translation,
            model=model or translation.model,
            connection=connection or translation.connection,
            connection_history=history or dict(translation.connection_history),
            gemini=gemini,
        )
    languages = settings.intent.languages
    if source_language is not None:
        languages = replace(languages, source_language=source_language)
    return replace(
        settings,
        intent=replace(
            settings.intent,
            translation=translation,
            languages=languages,
            prompts=replace(settings.intent.prompts, system_prompt_override=prompt),
        ),
    )


def _translation(pending: AppSettingsVNext):
    return pending.intent.translation


def _llm(pending: AppSettingsVNext) -> str:
    translation = _translation(pending)
    return provider_llm_for_translation(translation.model, translation.connection)


def _prompt(pending: AppSettingsVNext) -> str:
    return pending.intent.prompts.system_prompt_override


def _make_settings_view(monkeypatch):
    monkeypatch.setattr(settings_view.SettingsView, "_populate_host_apis", lambda self: None)
    monkeypatch.setattr(settings_view.SettingsView, "_refresh_microphones", lambda self: None)
    monkeypatch.setattr(settings_view.SettingsView, "update", lambda self: None)
    view = settings_view.SettingsView()
    view._settings_secrets = SettingsSecretsOwner(
        secret_store_factory=DummySecretStore,
    )
    return view


def test_luna_direct_route_uses_shared_translation_prompt(monkeypatch) -> None:
    view = _make_settings_view(monkeypatch)
    view.load_from_settings(AppSettingsVNext(), config_path=Path("settings.json"))

    view._on_llm_selected(TranslationModel.GPT_6_LUNA.value)
    view._on_translation_connection_selected(TranslationConnection.OFFICIAL_BYOK.value)

    pending = view.build_provider_apply_settings()
    assert pending is not None
    assert _translation(pending).model == "gpt_6_luna"
    assert _translation(pending).connection == "official_byok"
    assert view._active_prompt_key() == "openai"
    assert view._prompt_editor.value == load_prompt_for_provider("openai")
    assert view._prompt_editor.value == load_prompt_for_provider("openrouter")


def test_settings_view_loads_qwen_prompt(monkeypatch) -> None:
    settings = _settings(model="qwen38_flash", connection="official_byok")

    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert view._prompt_editor.value == load_prompt_for_provider("qwen")
    assert view._settings.intent.prompts.system_prompt_override is None


def test_settings_view_switches_prompt_on_llm_change(monkeypatch) -> None:
    settings = AppSettingsVNext()

    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert view._prompt_editor.value == load_prompt_for_provider("openrouter")

    view._on_llm_selected(TranslationModel.QWEN_38_FLASH.value)
    pending = view.build_provider_apply_settings()

    assert view._prompt_editor.value == load_prompt_for_provider("qwen")
    assert _llm(settings) == LLMProviderName.OPENROUTER.value
    assert pending is not None
    assert _llm(pending) == LLMProviderName.QWEN.value
    assert _translation(pending).qwen.llm_model == QwenLLMModel.QWEN_38_FLASH.value

    view._on_llm_selected(TranslationModel.LOCAL_LLM.value)
    pending = view.build_provider_apply_settings()

    assert view._prompt_editor.value == load_prompt_for_provider("local_llm")
    assert _llm(settings) == LLMProviderName.OPENROUTER.value
    assert pending is not None
    assert _llm(pending) == LLMProviderName.LOCAL_LLM.value

    view._on_llm_selected(TranslationModel.GEMINI_FLASH.value)
    pending = view.build_provider_apply_settings()

    assert view._prompt_editor.value == load_prompt_for_provider("gemini")
    assert _llm(settings) == LLMProviderName.OPENROUTER.value
    assert pending is not None
    assert _llm(pending) == LLMProviderName.GEMINI.value

    view._on_llm_selected(TranslationModel.GEMMA4_26B_31B.value)
    pending = view.build_provider_apply_settings()

    assert view._prompt_editor.value == load_prompt_for_provider("openrouter")
    assert _llm(settings) == LLMProviderName.OPENROUTER.value
    assert pending is not None
    assert _llm(pending) == LLMProviderName.OPENROUTER.value


def test_deepseek_managed_keeps_single_prompt(monkeypatch) -> None:
    settings = _settings(
        model="gemini_flash",
        connection="official_byok",
        prompt="GEMINI CUSTOM",
    )

    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    view._on_llm_selected(TranslationModel.DEEPSEEK_V4_FLASH.value)
    pending = view.build_provider_apply_settings()

    assert view._prompt_editor.value == "GEMINI CUSTOM"
    assert pending is not None
    assert _translation(pending).model == TranslationModel.DEEPSEEK_V4_FLASH.value
    assert _translation(pending).connection == TranslationConnection.OPENROUTER.value
    assert _llm(pending) == LLMProviderName.OPENROUTER.value
    assert (
        _translation(pending).openrouter_selection_alias
        == OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_BYOK.value
    )
    assert _prompt(pending) == "GEMINI CUSTOM"


def test_prompt_tab_labels_and_tag_editor_copy_render_from_i18n(monkeypatch) -> None:
    settings = _settings(
        model="qwen38_flash",
        connection="official_byok",
        source_language="en",
    )

    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    previous_locale = i18n_module.get_locale()
    try:
        i18n_module.set_locale("ko")
        view.apply_locale()

        assert view._persona_title.value == t("settings.section.persona")
        assert view._custom_vocab_title.value == t("settings.section.custom_vocabulary")
        assert view._custom_vocab_description_text.value == t(
            "settings.custom_vocabulary.description"
        )
        assert view._custom_vocab_tag_editor._input_field.hint_text == ""  # noqa: SLF001
    finally:
        i18n_module.set_locale(previous_locale)


def test_settings_view_shows_qwen_model_label(monkeypatch) -> None:
    settings = _settings(model="qwen38_flash", connection="official_byok")

    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert view._llm_text.content.value == t("provider.qwen38_flash")


def test_settings_view_uses_single_prompt_across_provider_switches(monkeypatch) -> None:
    settings = _settings(
        model="gemini_flash",
        connection="official_byok",
        prompt="GEMINI CUSTOM",
    )

    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert view._prompt_editor.value == "GEMINI CUSTOM"

    view._on_llm_selected(TranslationModel.QWEN_38_FLASH.value)
    pending = view.build_provider_apply_settings()
    assert view._prompt_editor.value == "GEMINI CUSTOM"
    assert _prompt(settings) == "GEMINI CUSTOM"
    assert pending is not None
    assert _prompt(pending) == "GEMINI CUSTOM"

    view._on_prompt_change("QWEN EDITED")
    pending = view.build_provider_apply_settings()
    assert pending is not None
    assert _prompt(pending) == "QWEN EDITED"

    view._on_llm_selected(TranslationModel.GEMINI_FLASH.value)
    pending = view.build_provider_apply_settings()
    assert view._prompt_editor.value == "QWEN EDITED"
    assert _prompt(settings) == "GEMINI CUSTOM"
    assert pending is not None
    assert _prompt(pending) == "QWEN EDITED"

    view._on_llm_selected(TranslationModel.GEMMA4_26B_31B.value)
    pending = view.build_provider_apply_settings()
    assert view._prompt_editor.value == "QWEN EDITED"
    assert _prompt(settings) == "GEMINI CUSTOM"
    assert pending is not None
    assert _prompt(pending) == "QWEN EDITED"


def test_prompt_draft_survives_provider_round_trip_until_commit(monkeypatch) -> None:
    settings = _settings(
        model="gemini_flash",
        connection="official_byok",
        prompt="GEMINI CUSTOM",
    )

    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    view._on_prompt_change("GEMINI DRAFT")
    view._on_llm_selected(TranslationModel.QWEN_38_FLASH.value)
    view._on_llm_selected(TranslationModel.GEMINI_FLASH.value)

    assert view._prompt_editor.value == "GEMINI DRAFT"
    assert _prompt(settings) == "GEMINI CUSTOM"


def test_single_prompt_whitespace_survives_provider_switch(monkeypatch) -> None:
    settings = _settings(
        model="gemini_flash",
        connection="official_byok",
        prompt="  CUSTOM PROMPT\n",
    )

    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    view._on_llm_selected(TranslationModel.QWEN_38_FLASH.value)
    pending = view.build_provider_apply_settings()

    assert view._prompt_editor.value == "  CUSTOM PROMPT\n"
    assert pending is not None
    assert _prompt(pending) == "  CUSTOM PROMPT\n"


@pytest.fixture(params=["en", "ja", "ko", "ru", "zh-CN"])
def selection_locale(request):
    previous = i18n_module.get_locale()
    i18n_module.set_locale(request.param)
    yield request.param
    i18n_module.set_locale(previous)


def test_settings_view_llm_modal_lists_logical_translation_models_once(
    monkeypatch, selection_locale
) -> None:
    settings = AppSettingsVNext()
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
    assert captured["left_column_sections"] == 1
    options = captured["options"]
    values = [option.value for option in options]

    assert TranslationModel.GPT_6_LUNA.value in values
    assert TranslationModel.QWEN_38_FLASH.value in values
    assert TranslationModel.LOCAL_LLM.value in values
    assert all("qwen35_flash" not in value for value in values)
    assert len(values) == len(set(values))
    assert TranslationModel.MANAGED_GEMMA.value not in values

    managed = {option.value: option for option in options}
    assert managed[TranslationModel.GPT_6_LUNA.value].description == t(
        "settings.translation_model.gpt_6_luna.description"
    )
    assert managed[TranslationModel.GPT_6_LUNA.value].description
    deepseek_v4_flash = managed[TranslationModel.DEEPSEEK_V4_FLASH.value]
    assert deepseek_v4_flash.section == t("settings.translation_model.section.others")
    assert deepseek_v4_flash.description == ""
    deepseek_v4_flash_41 = managed[TranslationModel.DEEPSEEK_V4_FLASH_41.value]
    assert deepseek_v4_flash_41.section == t("settings.translation_model.section.recommended_cloud")
    assert deepseek_v4_flash_41.description == t(
        "settings.translation_model.deepseek_v4_flash_41.description"
    )
    assert managed["managed_gemma_cpu"].label == t("provider.managed_gemma_cpu")
    assert managed["managed_gemma_cpu"].description == t(
        "settings.translation_model.managed_gemma_cpu.description"
    )
    assert managed["managed_gemma_gpu"].label == t("provider.managed_gemma_gpu")
    assert managed["managed_gemma_gpu"].description == t(
        "settings.translation_model.managed_gemma_gpu.description"
    )
    assert managed["managed_gemma_cpu"].section == t(
        "settings.translation_model.section.cpu_inference"
    )
    assert managed["managed_gemma_gpu"].section == t(
        "settings.translation_model.section.gpu_inference"
    )
    gemma_26b = managed["managed_gemma_26b"]
    assert gemma_26b.label == t("provider.managed_gemma_26b")
    assert gemma_26b.label != managed["managed_gemma_gpu"].label
    assert gemma_26b.description == t("settings.translation_model.managed_gemma_26b.description")
    assert gemma_26b.description
    assert gemma_26b.section == t("settings.translation_model.section.gpu_inference")
    assert not gemma_26b.disabled

    cloud_gemma = next(
        option for option in options if option.value == TranslationModel.GEMMA4_26B_31B.value
    )
    assert cloud_gemma.section == t("settings.translation_model.section.recommended_cloud")
    assert cloud_gemma.label == t("provider.gemma4_26b_31b") == "Gemma 4 26B + 31B"
    assert values.count(TranslationModel.GEMMA4_26B_31B.value) == 1
    assert not {"gemma4", "gemma4_31b"} & set(values)

    sections: list[str] = []
    for option in options:
        if option.section and option.section not in sections:
            sections.append(option.section)
    assert sections == [
        t("settings.translation_model.section.recommended_cloud"),
        t("settings.translation_model.section.cpu_inference"),
        t("settings.translation_model.section.gpu_inference"),
        t("settings.translation_model.section.user_settings"),
        t("settings.translation_model.section.others"),
    ]

    others_options = [
        option
        for option in options
        if option.section == t("settings.translation_model.section.others")
    ]
    assert others_options[0] is deepseek_v4_flash
    assert all(option.value != cloud_gemma.value for option in others_options)


def test_combined_gemma_connection_modal_lists_managed_and_openrouter(monkeypatch) -> None:
    settings = _settings(
        model="gemma4_26b_31b",
        connection="openrouter",
        history={"gemma4_26b_31b": "openrouter"},
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
    ]
    assert captured["current"] == TranslationConnection.OPENROUTER.value
    assert all(option.description == "" for option in captured["options"])


def test_luna_connection_modal_describes_chatgpt_connection(monkeypatch) -> None:
    settings = _settings(
        model=TranslationModel.GPT_6_LUNA.value,
        connection=TranslationConnection.CHATGPT.value,
        history={TranslationModel.GPT_6_LUNA.value: TranslationConnection.CHATGPT.value},
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

    descriptions = {option.value: option.description for option in captured["options"]}
    assert captured["show_description"] is True
    assert descriptions == {
        TranslationConnection.CHATGPT.value: t(
            "settings.translation_connection.chatgpt.description"
        ),
        TranslationConnection.OPENROUTER.value: "",
        TranslationConnection.OFFICIAL_BYOK.value: "",
    }
    assert descriptions[TranslationConnection.CHATGPT.value]


@pytest.mark.parametrize(
    ("model", "expected_connections"),
    [
        (
            TranslationModel.DEEPSEEK_V4_FLASH,
            [TranslationConnection.OPENROUTER],
        ),
        (
            TranslationModel.DEEPSEEK_V4_FLASH_41,
            [
                TranslationConnection.MANAGED,
                TranslationConnection.MANAGED_CHINA,
                TranslationConnection.OPENROUTER,
                TranslationConnection.OFFICIAL_BYOK,
            ],
        ),
    ],
)
def test_deepseek_connection_modal_exposes_version_specific_choices(
    monkeypatch,
    model: TranslationModel,
    expected_connections: list[TranslationConnection],
) -> None:
    connection = (
        TranslationConnection.OPENROUTER
        if model == TranslationModel.DEEPSEEK_V4_FLASH
        else TranslationConnection.MANAGED
    )
    settings = _settings(
        model=model.value,
        connection=connection.value,
        history={model.value: connection.value},
    )
    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))
    attach_dummy_page(monkeypatch, view)
    captured: dict[str, object] = {}

    class DummyModal:
        def __init__(self, _page, _title, options, _on_select, **_kwargs):
            captured["options"] = options

        def open(self, current: str) -> None:
            captured["current"] = current

    monkeypatch.setattr(settings_view, "SettingsModal", DummyModal)
    view._on_translation_connection_click(None)

    assert [option.value for option in captured["options"]] == [
        connection.value for connection in expected_connections
    ]
    assert captured["current"] == connection.value


def test_settings_view_keeps_gemini_model_without_provider_switch(monkeypatch) -> None:
    settings = _settings(
        model="gemini_flash",
        connection="official_byok",
        prompt="GEMINI CUSTOM",
    )

    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    view._on_llm_selected(TranslationModel.GEMINI_FLASH.value)
    pending = view.build_provider_apply_settings()

    assert _llm(settings) == LLMProviderName.GEMINI.value
    assert _translation(settings).gemini.llm_model == GeminiLLMModel.GEMINI_FLASH.value
    assert pending is not None
    assert _translation(pending).gemini.llm_model == GeminiLLMModel.GEMINI_FLASH.value
    assert _prompt(settings) == "GEMINI CUSTOM"
    assert view._prompt_editor.value == "GEMINI CUSTOM"


def test_settings_view_toggles_qwen_region_visibility_with_stt_provider(monkeypatch) -> None:
    settings = AppSettingsVNext()
    view = _make_settings_view(monkeypatch)
    view.load_from_settings(settings, config_path=Path("settings.json"))

    assert view._qwen_region_btn.visible is False

    view._on_stt_selected(STTProviderName.QWEN_AUDIO.value)
    assert view._qwen_region_btn.visible is True

    view._on_stt_selected(STTProviderName.DEEPGRAM.value)
    assert view._qwen_region_btn.visible is False
