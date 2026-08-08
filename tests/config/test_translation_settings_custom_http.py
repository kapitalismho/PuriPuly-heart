from __future__ import annotations

import copy

import pytest

from puripuly_heart.config.settings import (
    AppSettings,
    LLMProviderName,
    QwenLLMModel,
    QwenRegion,
    TranslationConnection,
    TranslationFallbackSettings,
    TranslationModel,
    TranslationSettings,
    from_dict,
    materialize_translation_settings,
    to_dict,
)
from puripuly_heart.config.settings_vnext import migration, serialization


def _custom_settings() -> AppSettings:
    settings = AppSettings()
    settings.telemetry.consent = "allow"
    settings.telemetry_state.anonymous_id = "custom-translation-test"
    settings.provider.llm = LLMProviderName.QWEN
    settings.qwen.region = QwenRegion.SINGAPORE
    settings.qwen.llm_model = QwenLLMModel.QWEN_35_PLUS
    settings.translation = TranslationSettings(
        model=TranslationModel.CUSTOM_HTTP,
        connection=TranslationConnection.CUSTOM_HTTP,
        connection_history={
            TranslationModel.GEMMA4_26B_31B.value: TranslationConnection.MANAGED,
            TranslationModel.QWEN_35_PLUS.value: TranslationConnection.OFFICIAL_BYOK,
            TranslationModel.CUSTOM_HTTP.value: TranslationConnection.CUSTOM_HTTP,
        },
        fallback=TranslationFallbackSettings(
            enabled=True,
            model=TranslationModel.GEMMA4,
            connection=TranslationConnection.OPENROUTER,
        ),
        extension_id="libretranslate",
        previous_llm_model=TranslationModel.QWEN_35_PLUS,
    )
    return settings


def test_custom_http_settings_roundtrip_preserves_inactive_llm_state() -> None:
    settings = _custom_settings()

    persisted = to_dict(settings)
    loaded = from_dict(persisted)

    assert persisted["translation"]["model"] == TranslationModel.CUSTOM_HTTP.value
    assert persisted["translation"]["connection"] == TranslationConnection.CUSTOM_HTTP.value
    assert persisted["translation"]["extension_id"] == "libretranslate"
    assert persisted["translation"]["previous_llm_model"] == TranslationModel.QWEN_35_PLUS.value
    assert "credential-value" not in repr(persisted)
    assert loaded.translation.model is TranslationModel.CUSTOM_HTTP
    assert loaded.translation.connection is TranslationConnection.CUSTOM_HTTP
    assert loaded.translation.extension_id == "libretranslate"
    assert loaded.translation.previous_llm_model is TranslationModel.QWEN_35_PLUS
    assert loaded.translation.fallback == settings.translation.fallback
    assert loaded.provider.llm is LLMProviderName.QWEN
    assert loaded.qwen.region is QwenRegion.SINGAPORE
    assert loaded.qwen.llm_model is QwenLLMModel.QWEN_35_PLUS


def test_custom_http_cannot_be_used_as_translation_fallback() -> None:
    with pytest.raises(ValueError, match="cannot be used as fallback"):
        TranslationFallbackSettings(
            enabled=True,
            model=TranslationModel.CUSTOM_HTTP,
            connection=TranslationConnection.CUSTOM_HTTP,
        ).validate()


def test_switching_custom_http_and_llm_preserves_model_connection_and_fallback() -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.QWEN
    settings.qwen.region = QwenRegion.SINGAPORE
    settings.qwen.llm_model = QwenLLMModel.QWEN_35_PLUS
    settings.translation = TranslationSettings(
        model=TranslationModel.QWEN_35_PLUS,
        connection=TranslationConnection.OFFICIAL_BYOK,
        connection_history={
            TranslationModel.GEMMA4_26B_31B.value: TranslationConnection.MANAGED,
            TranslationModel.QWEN_35_PLUS.value: TranslationConnection.OFFICIAL_BYOK,
        },
        fallback=TranslationFallbackSettings(
            enabled=True,
            model=TranslationModel.GEMMA4,
            connection=TranslationConnection.OPENROUTER,
        ),
    )
    expected_provider = settings.provider.llm
    expected_region = settings.qwen.region
    expected_qwen_model = settings.qwen.llm_model
    expected_fallback = copy.deepcopy(settings.translation.fallback)
    expected_history = copy.deepcopy(settings.translation.connection_history)

    settings.translation.model = TranslationModel.CUSTOM_HTTP
    settings.translation.connection = TranslationConnection.CUSTOM_HTTP
    settings.translation.extension_id = "libretranslate"
    settings.translation.previous_llm_model = TranslationModel.QWEN_35_PLUS
    materialize_translation_settings(settings)

    assert settings.provider.llm is expected_provider
    assert settings.qwen.region is expected_region
    assert settings.qwen.llm_model is expected_qwen_model
    assert settings.translation.fallback == expected_fallback
    assert settings.translation.connection_history == {
        **expected_history,
        TranslationModel.CUSTOM_HTTP.value: TranslationConnection.CUSTOM_HTTP,
    }

    settings.translation = TranslationSettings(
        model=TranslationModel.QWEN_35_PLUS,
        connection=TranslationConnection.OFFICIAL_BYOK,
        connection_history=copy.deepcopy(settings.translation.connection_history),
        fallback=copy.deepcopy(settings.translation.fallback),
    )
    materialize_translation_settings(settings)

    assert settings.provider.llm is expected_provider
    assert settings.qwen.region is expected_region
    assert settings.qwen.llm_model is expected_qwen_model
    assert settings.translation.fallback == expected_fallback


def test_vnext_custom_http_migration_is_idempotent_and_projects_legacy_provider() -> None:
    legacy = _custom_settings()

    canonical = serialization.to_dict(migration.from_legacy_app_settings(legacy))
    reloaded = serialization.to_dict(migration.from_dict(canonical))
    projected = migration.to_legacy_dict(migration.from_dict(canonical))

    assert reloaded == canonical
    assert canonical["intent"]["translation"]["extension_id"] == "libretranslate"
    assert canonical["intent"]["translation"]["previous_llm_model"] == (
        TranslationModel.QWEN_35_PLUS.value
    )
    assert projected["translation"]["model"] == TranslationModel.CUSTOM_HTTP.value
    assert projected["translation"]["connection"] == TranslationConnection.CUSTOM_HTTP.value
    assert projected["translation"]["extension_id"] == "libretranslate"
    assert projected["translation"]["previous_llm_model"] == TranslationModel.QWEN_35_PLUS.value
    assert projected["provider"]["llm"] == LLMProviderName.QWEN.value

    legacy_roundtrip = migration.from_dict(to_dict(legacy))
    assert serialization.to_dict(legacy_roundtrip)["intent"]["translation"]["model"] == (
        TranslationModel.CUSTOM_HTTP.value
    )
