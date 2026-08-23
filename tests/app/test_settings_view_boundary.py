from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest
from puripuly_heart.app.services.settings_application import (
    materialize_immediate_settings_intent,
    materialize_prompt_apply_intent,
    materialize_provider_apply_intent,
    settings_view_surface_snapshots,
)

from puripuly_heart.app.ports.settings_view import (
    CustomVocabularySettingsIntent,
    LocaleSettingsIntent,
    PromptApplyIntent,
    ProviderApplyIntent,
    QwenRegionEdit,
    SystemPromptEdit,
    TranslationProviderEdit,
)
from puripuly_heart.config.provider_values import LLMProviderName, QwenRegion
from puripuly_heart.config.settings import AppSettings
from puripuly_heart.config.translation_values import (
    TranslationConnection,
    TranslationModel,
)


def test_surface_projection_returns_independent_frozen_snapshots() -> None:
    settings = AppSettings()
    settings.languages.source_language = "ko"
    settings.stt.custom_terms = {"ko": ["PuriPuly"], "en": ["Avatar"]}

    provider, general, prompt, overlay = settings_view_surface_snapshots(settings)

    assert provider.translation.model == settings.translation.model
    assert general.locale == settings.ui.locale
    assert prompt.custom_vocabulary_terms == ("PuriPuly",)
    assert prompt.custom_vocabulary_other_languages_have_terms is True
    assert overlay.target == settings.overlay.target
    with pytest.raises(FrozenInstanceError):
        general.locale = "ja"


def test_immediate_intents_rebase_onto_latest_settings_without_surface_displacement() -> None:
    displayed = AppSettings()
    displayed.languages.source_language = "en"
    displayed.stt.custom_terms = {"en": ["old"], "ja": ["既存"]}
    current = AppSettings()
    current.languages.source_language = "ja"
    current.languages.target_language = "ko"
    current.stt.custom_terms = {"en": ["old"], "ja": ["最新"]}

    localized = materialize_immediate_settings_intent(current, LocaleSettingsIntent("ko"))
    updated = materialize_immediate_settings_intent(
        localized,
        CustomVocabularySettingsIntent("en", ("new",), True),
    )

    assert updated.ui.locale == "ko"
    assert updated.languages.source_language == "ja"
    assert updated.languages.target_language == "ko"
    assert updated.stt.custom_terms == {"en": ["new"], "ja": ["最新"]}
    assert current.ui.locale == "en"
    assert displayed.stt.custom_terms["en"] == ["old"]


def test_provider_edit_journal_replays_only_owned_fields_onto_latest_settings() -> None:
    displayed = AppSettings()
    provider, _general, _prompt, _overlay = settings_view_surface_snapshots(displayed)
    selection = replace(
        provider.translation,
        model=TranslationModel.GEMINI_37_FLASH,
        connection=TranslationConnection.OPENROUTER,
    )
    current = AppSettings()
    current.languages.source_language = "ja"
    current.audio.input_device = "latest microphone"

    updated = materialize_provider_apply_intent(
        current,
        ProviderApplyIntent(
            (
                TranslationProviderEdit(selection),
                QwenRegionEdit(QwenRegion.SINGAPORE),
                SystemPromptEdit("focused prompt"),
            )
        ),
    )

    assert updated.provider.llm == LLMProviderName.OPENROUTER
    assert updated.translation.model == TranslationModel.GEMINI_37_FLASH
    assert updated.translation.connection == TranslationConnection.OPENROUTER
    assert updated.qwen.region == QwenRegion.SINGAPORE
    assert updated.system_prompt == "focused prompt"
    assert updated.languages.source_language == "ja"
    assert updated.audio.input_device == "latest microphone"


def test_prompt_intent_preserves_latest_languages_and_provider_selection() -> None:
    current = AppSettings()
    current.languages.source_language = "ja"
    current.languages.target_language = "ko"
    current.provider.llm = LLMProviderName.QWEN

    updated = materialize_prompt_apply_intent(current, PromptApplyIntent("new prompt"))

    assert updated.system_prompt == "new prompt"
    assert updated.system_prompts == {}
    assert updated.languages.source_language == "ja"
    assert updated.languages.target_language == "ko"
    assert updated.provider.llm == LLMProviderName.QWEN
