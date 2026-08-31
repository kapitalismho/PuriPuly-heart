from __future__ import annotations

from dataclasses import replace

import pytest

from puripuly_heart.app.services.canonical_settings_persistence import (
    materialize_canonical_translation_settings,
)
from puripuly_heart.config.provider_values import LLMProviderName
from puripuly_heart.config.runtime_resolution import TranslationFallbackRuntimeIntent
from puripuly_heart.config.settings_vnext import serialization
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, TranslationFallbackIntent
from puripuly_heart.config.translation_values import (
    TranslationConnection,
    TranslationModel,
    default_translation_connection,
    provider_llm_for_translation,
    supported_translation_connections,
)


def test_managed_gemma_exposes_exact_cpu_gpu_product_choices() -> None:
    assert supported_translation_connections(TranslationModel.MANAGED_GEMMA) == (
        TranslationConnection.CPU,
        TranslationConnection.GPU,
    )
    assert default_translation_connection(TranslationModel.MANAGED_GEMMA) == (
        TranslationConnection.CPU
    )


def test_managed_gemma_12b_exposes_gpu_only_product_choice() -> None:
    assert supported_translation_connections(TranslationModel.MANAGED_GEMMA_12B) == (
        TranslationConnection.GPU,
    )
    assert default_translation_connection(TranslationModel.MANAGED_GEMMA_12B) == (
        TranslationConnection.GPU
    )


@pytest.mark.parametrize(
    "connection",
    [TranslationConnection.CPU, TranslationConnection.GPU],
)
def test_managed_gemma_materializes_and_round_trips_as_distinct_provider(
    connection: TranslationConnection,
) -> None:
    current = AppSettingsVNext()
    settings = materialize_canonical_translation_settings(
        replace(
            current,
            intent=replace(
                current.intent,
                translation=replace(
                    current.intent.translation,
                    model=TranslationModel.MANAGED_GEMMA.value,
                    connection=connection.value,
                    fallback=TranslationFallbackIntent(
                        selection_alias="openrouter_deepseek_v4_flash"
                    ),
                ),
            ),
        )
    )
    serialized = serialization.to_dict(settings)
    restored = serialization.from_dict(serialized)

    assert (
        provider_llm_for_translation(
            settings.intent.translation.model,
            settings.intent.translation.connection,
        )
        == LLMProviderName.MANAGED_GEMMA.value
    )
    assert serialized["intent"]["translation"]["model"] == "managed_gemma"
    assert serialized["intent"]["translation"]["connection"] == connection.value
    assert restored.intent.translation.model == "managed_gemma"
    assert restored.intent.translation.connection == connection.value
    assert restored.intent.translation.fallback.enabled is True


def test_managed_gemma_12b_materializes_and_round_trips_as_gpu_local_model() -> None:
    current = AppSettingsVNext()
    settings = materialize_canonical_translation_settings(
        replace(
            current,
            intent=replace(
                current.intent,
                translation=replace(
                    current.intent.translation,
                    model=TranslationModel.MANAGED_GEMMA_12B.value,
                    connection=TranslationConnection.GPU.value,
                ),
            ),
        )
    )
    serialized = serialization.to_dict(settings)
    restored = serialization.from_dict(serialized)

    assert (
        provider_llm_for_translation(
            settings.intent.translation.model,
            settings.intent.translation.connection,
        )
        == LLMProviderName.MANAGED_GEMMA.value
    )
    assert serialized["intent"]["translation"]["model"] == "managed_gemma_12b"
    assert serialized["intent"]["translation"]["connection"] == TranslationConnection.GPU.value
    assert restored.intent.translation.model == "managed_gemma_12b"
    assert restored.intent.translation.connection == TranslationConnection.GPU.value


def test_managed_gemma_cannot_be_configured_as_provider_fallback() -> None:
    with pytest.raises(ValueError, match="cannot be used as provider fallback"):
        TranslationFallbackRuntimeIntent(
            enabled=True,
            model=TranslationModel.MANAGED_GEMMA.value,
            connection=TranslationConnection.CPU.value,
        )


def test_managed_gemma_12b_cannot_be_configured_as_provider_fallback() -> None:
    with pytest.raises(ValueError, match="cannot be used as provider fallback"):
        TranslationFallbackRuntimeIntent(
            enabled=True,
            model=TranslationModel.MANAGED_GEMMA_12B.value,
            connection=TranslationConnection.GPU.value,
        )
