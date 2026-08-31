from __future__ import annotations

from dataclasses import replace

import pytest

from puripuly_heart.app.services.canonical_settings_persistence import (
    materialize_canonical_translation_settings,
)
from puripuly_heart.config.provider_values import LLMProviderName, QwenRegion
from puripuly_heart.config.runtime_resolution import TranslationFallbackRuntimeIntent
from puripuly_heart.config.settings_vnext import serialization
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, TranslationFallbackIntent
from puripuly_heart.config.translation_values import TranslationConnection, TranslationModel


def _custom_settings() -> AppSettingsVNext:
    current = AppSettingsVNext()
    return replace(
        current,
        intent=replace(
            current.intent,
            translation=replace(
                current.intent.translation,
                model=TranslationModel.CUSTOM_HTTP.value,
                connection=TranslationConnection.CUSTOM_HTTP.value,
                connection_history={
                    TranslationModel.GEMMA4_26B_31B.value: TranslationConnection.MANAGED.value,
                    TranslationModel.QWEN_38_FLASH.value: TranslationConnection.OFFICIAL_BYOK.value,
                    TranslationModel.CUSTOM_HTTP.value: TranslationConnection.CUSTOM_HTTP.value,
                },
                fallback=TranslationFallbackIntent(selection_alias="openrouter_gemma4_26b_a4b"),
                http_extension_id="libretranslate",
                previous_llm_model=TranslationModel.QWEN_38_FLASH.value,
                qwen=replace(
                    current.intent.translation.qwen,
                    region=QwenRegion.SINGAPORE.value,
                    llm_model="qwen3.8-flash",
                ),
            ),
            telemetry=replace(current.intent.telemetry, enabled=True),
        ),
        state=replace(
            current.state,
            telemetry=replace(current.state.telemetry, anonymous_id="custom-translation-test"),
        ),
    )


def test_custom_http_settings_roundtrip_preserves_inactive_llm_state() -> None:
    settings = _custom_settings()

    persisted = serialization.to_dict(settings)
    loaded = serialization.from_dict(persisted)

    assert persisted["intent"]["translation"]["model"] == TranslationModel.CUSTOM_HTTP.value
    assert (
        persisted["intent"]["translation"]["connection"] == TranslationConnection.CUSTOM_HTTP.value
    )
    assert persisted["intent"]["translation"]["http_extension_id"] == "libretranslate"
    assert persisted["intent"]["translation"]["previous_llm_model"] == (
        TranslationModel.QWEN_38_FLASH.value
    )
    assert "credential-value" not in repr(persisted)
    assert loaded.intent.translation.model == TranslationModel.CUSTOM_HTTP.value
    assert loaded.intent.translation.connection == TranslationConnection.CUSTOM_HTTP.value
    assert loaded.intent.translation.http_extension_id == "libretranslate"
    assert loaded.intent.translation.previous_llm_model == TranslationModel.QWEN_38_FLASH.value
    assert loaded.intent.translation.fallback == settings.intent.translation.fallback
    assert loaded.intent.translation.qwen.region == QwenRegion.SINGAPORE.value
    assert loaded.intent.translation.qwen.llm_model == "qwen3.8-flash"


def test_custom_http_cannot_be_used_as_translation_fallback() -> None:
    with pytest.raises(ValueError, match="cannot be used as fallback"):
        TranslationFallbackRuntimeIntent(
            enabled=True,
            model=TranslationModel.CUSTOM_HTTP.value,
            connection=TranslationConnection.CUSTOM_HTTP.value,
        )


def test_switching_custom_http_and_llm_preserves_model_connection_and_fallback() -> None:
    current = AppSettingsVNext()
    settings = replace(
        current,
        intent=replace(
            current.intent,
            translation=replace(
                current.intent.translation,
                model=TranslationModel.QWEN_38_FLASH.value,
                connection=TranslationConnection.OFFICIAL_BYOK.value,
                connection_history={
                    TranslationModel.GEMMA4_26B_31B.value: TranslationConnection.MANAGED.value,
                    TranslationModel.QWEN_38_FLASH.value: TranslationConnection.OFFICIAL_BYOK.value,
                },
                fallback=TranslationFallbackIntent(selection_alias="openrouter_gemma4_26b_a4b"),
                qwen=replace(
                    current.intent.translation.qwen,
                    region=QwenRegion.SINGAPORE.value,
                    llm_model="qwen3.8-flash",
                ),
            ),
        ),
    )
    expected_region = settings.intent.translation.qwen.region
    expected_qwen_model = settings.intent.translation.qwen.llm_model
    expected_fallback = settings.intent.translation.fallback
    expected_history = dict(settings.intent.translation.connection_history)

    custom = materialize_canonical_translation_settings(
        replace(
            settings,
            intent=replace(
                settings.intent,
                translation=replace(
                    settings.intent.translation,
                    model=TranslationModel.CUSTOM_HTTP.value,
                    connection=TranslationConnection.CUSTOM_HTTP.value,
                    http_extension_id="libretranslate",
                    previous_llm_model=TranslationModel.QWEN_38_FLASH.value,
                ),
            ),
        )
    )

    assert custom.intent.translation.qwen.region == expected_region
    assert custom.intent.translation.qwen.llm_model == expected_qwen_model
    assert custom.intent.translation.fallback == expected_fallback
    assert custom.intent.translation.model == TranslationModel.CUSTOM_HTTP.value
    assert custom.intent.translation.connection == TranslationConnection.CUSTOM_HTTP.value
    for key, value in expected_history.items():
        assert custom.intent.translation.connection_history[key] == value

    restored = materialize_canonical_translation_settings(
        replace(
            custom,
            intent=replace(
                custom.intent,
                translation=replace(
                    custom.intent.translation,
                    model=TranslationModel.QWEN_38_FLASH.value,
                    connection=TranslationConnection.OFFICIAL_BYOK.value,
                ),
            ),
        )
    )

    assert restored.intent.translation.qwen.region == expected_region
    assert restored.intent.translation.qwen.llm_model == expected_qwen_model
    assert restored.intent.translation.fallback == expected_fallback
    assert LLMProviderName.QWEN.value == "qwen"
