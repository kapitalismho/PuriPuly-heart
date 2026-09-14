from __future__ import annotations

from dataclasses import replace

import pytest

from puripuly_heart.app.services.canonical_settings_persistence import (
    materialize_canonical_translation_settings,
)
from puripuly_heart.config.provider_values import LLMProviderName
from puripuly_heart.config.settings_vnext import serialization
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
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
