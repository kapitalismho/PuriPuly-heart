from __future__ import annotations

from puripuly_heart.config.settings import (
    AppSettings,
    LLMProviderName,
    OpenRouterCredentialSource,
    OpenRouterLLMModel,
    OpenRouterProviderRouting,
    OpenRouterSelectionAlias,
    TranslationConnection,
    TranslationModel,
    TranslationSettings,
    build_managed_openrouter_byok_target_settings,
    to_dict,
)


def test_byok_target_rejects_absent_settings() -> None:
    assert build_managed_openrouter_byok_target_settings(None) is None


def test_byok_target_rejects_non_openrouter_settings() -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.GEMINI

    assert build_managed_openrouter_byok_target_settings(settings) is None


def test_byok_target_rejects_non_managed_openrouter_source() -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK

    assert build_managed_openrouter_byok_target_settings(settings) is None


def test_byok_target_projects_managed_qwen_without_mutating_source() -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.openrouter.selection_alias = OpenRouterSelectionAlias.QWEN35_FLASH_MANAGED
    settings.openrouter.llm_model = OpenRouterLLMModel.QWEN_35_FLASH_02_23
    baseline = to_dict(settings)

    target = build_managed_openrouter_byok_target_settings(settings)

    assert target is not None
    assert target is not settings
    assert target.openrouter.selected_source is OpenRouterCredentialSource.BYOK
    assert target.openrouter.selection_alias is OpenRouterSelectionAlias.QWEN35_FLASH_BYOK
    assert target.openrouter.llm_model is OpenRouterLLMModel.QWEN_35_FLASH_02_23
    assert target.openrouter.provider_routing is OpenRouterProviderRouting.DEFAULT
    assert target.translation.connection is TranslationConnection.OPENROUTER
    assert to_dict(settings) == baseline


def test_byok_target_clears_managed_china_translation_state() -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.translation = TranslationSettings(
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.MANAGED_CHINA,
        connection_history={
            TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.MANAGED_CHINA,
        },
    )
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.openrouter.selection_alias = OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED
    settings.openrouter.llm_model = OpenRouterLLMModel.DEEPSEEK_V4_FLASH
    settings.openrouter.provider_routing = OpenRouterProviderRouting.DEEPSEEK_ONLY

    target = build_managed_openrouter_byok_target_settings(settings)

    assert target is not None
    assert target.openrouter.selected_source is OpenRouterCredentialSource.BYOK
    assert target.openrouter.selection_alias is OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_BYOK
    assert target.openrouter.provider_routing is OpenRouterProviderRouting.DEFAULT
    assert target.translation.connection is TranslationConnection.OPENROUTER
    assert (
        target.translation.connection_history[TranslationModel.DEEPSEEK_V4_FLASH.value]
        is TranslationConnection.OPENROUTER
    )
