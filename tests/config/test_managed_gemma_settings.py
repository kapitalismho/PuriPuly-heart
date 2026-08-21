from __future__ import annotations

import pytest

from puripuly_heart.config.settings import (
    AppSettings,
    LLMProviderName,
    TranslationConnection,
    TranslationFallbackSettings,
    TranslationModel,
    TranslationSettings,
    default_translation_connection,
    materialize_translation_settings,
    supported_translation_connections,
)
from puripuly_heart.config.settings_vnext import migration, serialization


def test_managed_gemma_exposes_exact_cpu_gpu_product_choices() -> None:
    assert supported_translation_connections(TranslationModel.MANAGED_GEMMA) == (
        TranslationConnection.CPU,
        TranslationConnection.GPU,
    )
    assert default_translation_connection(TranslationModel.MANAGED_GEMMA) == (
        TranslationConnection.CPU
    )


@pytest.mark.parametrize(
    "connection",
    [TranslationConnection.CPU, TranslationConnection.GPU],
)
def test_managed_gemma_materializes_and_round_trips_as_distinct_provider(
    connection: TranslationConnection,
) -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.MANAGED_GEMMA,
        connection=connection,
        fallback=TranslationFallbackSettings(
            enabled=True,
            model=TranslationModel.DEEPSEEK_V4_FLASH,
            connection=TranslationConnection.OPENROUTER,
        ),
    )

    materialize_translation_settings(settings)
    canonical = migration.from_legacy_app_settings(settings)
    serialized = serialization.to_dict(canonical)
    restored = migration.from_dict(serialized)

    assert settings.provider.llm == LLMProviderName.MANAGED_GEMMA
    assert serialized["intent"]["translation"]["model"] == "managed_gemma"
    assert serialized["intent"]["translation"]["connection"] == connection.value
    assert restored.intent.translation.model == "managed_gemma"
    assert restored.intent.translation.connection == connection.value
    assert restored.intent.translation.fallback.enabled is True


def test_managed_gemma_cannot_be_configured_as_provider_fallback() -> None:
    fallback = TranslationFallbackSettings(
        enabled=True,
        model=TranslationModel.MANAGED_GEMMA,
        connection=TranslationConnection.CPU,
    )

    with pytest.raises(ValueError, match="cannot be used as provider fallback"):
        fallback.validate()


def test_managed_gemma_legacy_projection_preserves_active_and_previous_provider_identity() -> None:
    active = AppSettings()
    active.translation = TranslationSettings(
        model=TranslationModel.MANAGED_GEMMA,
        connection=TranslationConnection.CPU,
    )
    materialize_translation_settings(active)

    projected_active = migration.to_legacy_dict(migration.from_legacy_app_settings(active))

    custom = AppSettings()
    custom.translation = TranslationSettings(
        model=TranslationModel.CUSTOM_HTTP,
        connection=TranslationConnection.CUSTOM_HTTP,
        previous_llm_model=TranslationModel.MANAGED_GEMMA,
        connection_history={
            TranslationModel.MANAGED_GEMMA: TranslationConnection.GPU,
        },
        http_extension_id="managed-gemma-roundtrip",
    )
    custom.provider.llm = LLMProviderName.MANAGED_GEMMA

    projected_custom = migration.to_legacy_dict(migration.from_legacy_app_settings(custom))

    assert projected_active["provider"]["llm"] == LLMProviderName.MANAGED_GEMMA.value
    assert projected_custom["provider"]["llm"] == LLMProviderName.MANAGED_GEMMA.value
    assert projected_custom["translation"]["previous_llm_model"] == (
        TranslationModel.MANAGED_GEMMA.value
    )
