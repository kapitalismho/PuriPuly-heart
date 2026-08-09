from __future__ import annotations

import pytest

from puripuly_heart.app.services.osc.control_application import (
    SettingsBackedOscControlApplication,
)
from puripuly_heart.config.settings import (
    AppSettings,
    DeepSeekLLMModel,
    LLMProviderName,
    TranslationConnection,
    TranslationModel,
    materialize_translation_settings,
)


@pytest.mark.asyncio
async def test_translation_model_control_materializes_provider_and_connection() -> None:
    current = AppSettings()
    current.translation.connection = TranslationConnection.MANAGED
    applied: list[AppSettings] = []

    async def apply_settings(settings: object) -> object:
        nonlocal current
        assert isinstance(settings, AppSettings)
        applied.append(settings)
        current = settings
        return settings

    application = SettingsBackedOscControlApplication(
        settings_provider=lambda: current,
        apply_settings=apply_settings,
        translation_model_normalizer=materialize_translation_settings,
    )

    result = await application.set_translation_model(TranslationModel.DEEPSEEK_V4_PRO.value)

    assert result is applied[0]
    updated = applied[0]
    assert updated.translation.model == TranslationModel.DEEPSEEK_V4_PRO
    assert updated.translation.connection == TranslationConnection.OFFICIAL_BYOK
    assert (
        updated.translation.connection_history[TranslationModel.DEEPSEEK_V4_PRO.value]
        == TranslationConnection.OFFICIAL_BYOK
    )
    assert updated.provider.llm == LLMProviderName.DEEPSEEK
    assert updated.deepseek.llm_model == DeepSeekLLMModel.DEEPSEEK_V4_PRO


@pytest.mark.asyncio
async def test_custom_http_control_preserves_the_previous_llm_selection() -> None:
    current = AppSettings()
    current.translation.model = TranslationModel.DEEPSEEK_V4_PRO
    current.translation.connection = TranslationConnection.OFFICIAL_BYOK
    current.translation.http_extension_id = "demo"

    async def apply_settings(settings: object) -> object:
        nonlocal current
        assert isinstance(settings, AppSettings)
        current = settings
        return settings

    application = SettingsBackedOscControlApplication(
        settings_provider=lambda: current,
        apply_settings=apply_settings,
        translation_model_normalizer=materialize_translation_settings,
    )

    await application.set_translation_model(TranslationModel.CUSTOM_HTTP.value)

    assert current.translation.model is TranslationModel.CUSTOM_HTTP
    assert current.translation.connection is TranslationConnection.CUSTOM_HTTP
    assert current.translation.http_extension_id == "demo"
    assert current.translation.previous_llm_model is TranslationModel.DEEPSEEK_V4_PRO


@pytest.mark.asyncio
async def test_gemma_31b_control_keeps_an_active_cerebras_selection() -> None:
    current = AppSettings()
    current.translation.model = TranslationModel.GEMMA4_31B
    current.translation.connection = TranslationConnection.CEREBRAS
    applied = 0

    async def apply_settings(_settings: object) -> object:
        nonlocal applied
        applied += 1
        return True

    application = SettingsBackedOscControlApplication(
        settings_provider=lambda: current,
        apply_settings=apply_settings,
        translation_model_normalizer=materialize_translation_settings,
    )

    result = await application.set_translation_model(TranslationModel.GEMMA4_31B.value)

    assert result is True
    assert applied == 0
    assert current.translation.model is TranslationModel.GEMMA4_31B
    assert current.translation.connection is TranslationConnection.CEREBRAS


@pytest.mark.asyncio
async def test_settings_control_rejects_when_application_keeps_the_previous_state() -> None:
    current = AppSettings()

    async def apply_settings(_settings: object) -> object:
        return True

    application = SettingsBackedOscControlApplication(
        settings_provider=lambda: current,
        apply_settings=apply_settings,
        translation_model_normalizer=materialize_translation_settings,
    )

    result = await application.set_mute_sync(True)

    assert result is False
    assert current.osc.vrc_mic_intercept is False
