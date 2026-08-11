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


@pytest.mark.asyncio
async def test_set_languages_skips_apply_when_every_value_matches() -> None:
    current = AppSettings()
    current.languages.source_language = "fr"
    current.languages.target_language = "fr"
    current.languages.peer_source_language = "fr"
    current.languages.peer_target_language = "fr"
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

    result = await application.set_languages(
        self_source="fr",
        self_target="fr",
        peer_source="fr",
        peer_target="fr",
    )

    assert result is True
    assert applied == 0


@pytest.mark.asyncio
async def test_set_languages_applies_when_any_value_differs() -> None:
    current = AppSettings()
    current.languages.source_language = "ko"
    current.languages.target_language = "en"
    current.languages.peer_source_language = "en"
    current.languages.peer_target_language = "ko"
    applied = 0

    async def apply_settings(settings: object) -> object:
        nonlocal applied
        assert isinstance(settings, AppSettings)
        applied += 1
        current.languages.source_language = settings.languages.source_language
        current.languages.target_language = settings.languages.target_language
        current.languages.peer_source_language = settings.languages.peer_source_language
        current.languages.peer_target_language = settings.languages.peer_target_language
        return True

    application = SettingsBackedOscControlApplication(
        settings_provider=lambda: current,
        apply_settings=apply_settings,
        translation_model_normalizer=materialize_translation_settings,
    )

    result = await application.set_languages(
        self_source="ko",
        self_target="fr",
        peer_source="en",
        peer_target="ko",
    )

    assert result is True
    assert applied == 1
    assert current.languages.target_language == "fr"


@pytest.mark.asyncio
async def test_asr_controls_skip_apply_when_provider_matches() -> None:
    current = AppSettings()
    current.provider.stt = type(current.provider.stt)("deepgram")
    current.provider.peer_stt = type(current.provider.peer_stt)("soniox")
    applied = 0

    async def apply_settings(settings: object) -> object:
        nonlocal applied
        assert isinstance(settings, AppSettings)
        applied += 1
        current.provider.stt = settings.provider.stt
        current.provider.peer_stt = settings.provider.peer_stt
        return True

    application = SettingsBackedOscControlApplication(
        settings_provider=lambda: current,
        apply_settings=apply_settings,
        translation_model_normalizer=materialize_translation_settings,
    )

    assert (await application.set_self_asr("deepgram")) is True
    assert (await application.set_peer_asr("soniox")) is True
    assert applied == 0

    assert (await application.set_self_asr("local_parakeet_v3")) is True

    assert applied == 1
    assert current.provider.stt.value == "local_parakeet_v3"
    assert current.provider.peer_stt.value == "soniox"


@pytest.mark.asyncio
async def test_fallback_control_skips_apply_when_alias_matches() -> None:
    current = AppSettings()
    applied = 0

    async def apply_settings(settings: object) -> object:
        nonlocal applied
        assert isinstance(settings, AppSettings)
        applied += 1
        current.translation.fallback = settings.translation.fallback
        return True

    application = SettingsBackedOscControlApplication(
        settings_provider=lambda: current,
        apply_settings=apply_settings,
        translation_model_normalizer=materialize_translation_settings,
    )

    assert (await application.set_fallback("none")) is True
    assert applied == 0

    assert (await application.set_fallback("cerebras_gemma4_31b")) is True
    assert applied == 1
    assert current.translation.fallback.enabled is True
    assert current.translation.fallback.model.value == "gemma4_31b"
    assert current.translation.fallback.connection.value == "cerebras"

    assert (await application.set_fallback("cerebras_gemma4_31b")) is True
    assert applied == 1


@pytest.mark.asyncio
async def test_auxiliary_controls_skip_apply_when_unchanged() -> None:
    current = AppSettings()
    current.osc.vrc_mic_intercept = False
    current.osc.chatbox_include_source = False
    current.languages.peer_source_mode = "manual"
    applied = 0

    async def apply_settings(settings: object) -> object:
        nonlocal applied
        assert isinstance(settings, AppSettings)
        applied += 1
        current.osc.vrc_mic_intercept = settings.osc.vrc_mic_intercept
        current.osc.chatbox_include_source = settings.osc.chatbox_include_source
        current.languages.peer_source_mode = settings.languages.peer_source_mode
        return True

    application = SettingsBackedOscControlApplication(
        settings_provider=lambda: current,
        apply_settings=apply_settings,
        translation_model_normalizer=materialize_translation_settings,
    )

    assert (await application.set_mute_sync(False)) is True
    assert (await application.set_chatbox_source(False)) is True
    assert (await application.set_peer_auto_detect(False)) is True
    assert applied == 0

    assert (await application.set_mute_sync(True)) is True
    assert applied == 1
    assert current.osc.vrc_mic_intercept is True

    assert (await application.set_chatbox_source(True)) is True
    assert applied == 2
    assert current.osc.chatbox_include_source is True

    assert (await application.set_peer_auto_detect(True)) is True
    assert applied == 3
    assert current.languages.peer_source_mode == "auto"
