from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.app.services.canonical_settings_persistence import (
    SettingsOwner,
    materialize_canonical_translation_settings,
)
from puripuly_heart.config.provider_values import (
    OpenRouterCredentialSource,
    OpenRouterLLMModel,
    OpenRouterSelectionAlias,
)
from puripuly_heart.config.settings_vnext import serialization
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.config.translation_values import TranslationConnection, TranslationModel
from puripuly_heart.core.openrouter_routing import OpenRouterProviderRouting


def _vnext(
    *,
    model: str,
    connection: str,
    openrouter_source: str,
    openrouter_alias: str,
    openrouter_model: str,
    openrouter_routing: str | None = None,
    connection_history: dict[str, str] | None = None,
) -> AppSettingsVNext:
    current = AppSettingsVNext()
    translation = replace(
        current.intent.translation,
        model=model,
        connection=connection,
        openrouter_selected_source=openrouter_source,
        openrouter_selection_alias=openrouter_alias,
        openrouter_model=openrouter_model,
        openrouter_provider_routing=openrouter_routing
        or current.intent.translation.openrouter_provider_routing,
        connection_history=current.intent.translation.connection_history
        if connection_history is None
        else connection_history,
    )
    return materialize_canonical_translation_settings(
        replace(current, intent=replace(current.intent, translation=translation))
    )


def _byok_target(settings: AppSettingsVNext | None) -> AppSettingsVNext | None:
    owner = SettingsOwner(
        path=Path("settings.json"),
        persistence=SettingsVNextCanonicalPersistenceAdapter(),
        canonical=settings,
    )
    return owner.build_managed_openrouter_byok_target(settings)


def test_byok_target_rejects_absent_settings() -> None:
    assert _byok_target(None) is None


def test_byok_target_rejects_non_openrouter_settings() -> None:
    settings = _vnext(
        model=TranslationModel.GEMINI_31_FLASH_LITE.value,
        connection=TranslationConnection.OFFICIAL_BYOK.value,
        openrouter_source="managed",
        openrouter_alias=OpenRouterSelectionAlias.GEMMA4_MANAGED.value,
        openrouter_model=OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value,
    )
    assert _byok_target(settings) is None


def test_byok_target_rejects_non_managed_openrouter_source() -> None:
    settings = _vnext(
        model=TranslationModel.GEMMA4.value,
        connection=TranslationConnection.OPENROUTER.value,
        openrouter_source="byok",
        openrouter_alias=OpenRouterSelectionAlias.GEMMA4_BYOK.value,
        openrouter_model=OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value,
    )
    assert _byok_target(settings) is None


def test_byok_target_projects_managed_qwen_without_mutating_source() -> None:
    settings = _vnext(
        model="openrouter_qwen35_flash",
        connection="managed",
        openrouter_source="managed",
        openrouter_alias=OpenRouterSelectionAlias.QWEN35_FLASH_MANAGED.value,
        openrouter_model=OpenRouterLLMModel.QWEN_35_FLASH_02_23.value,
    )
    baseline = serialization.to_dict(settings)

    target = _byok_target(settings)

    assert target is not None
    assert target is not settings
    assert target.intent.translation.openrouter_selected_source == (
        OpenRouterCredentialSource.BYOK.value
    )
    assert target.intent.translation.openrouter_selection_alias == (
        OpenRouterSelectionAlias.QWEN35_FLASH_BYOK.value
    )
    assert target.intent.translation.openrouter_model == (
        OpenRouterLLMModel.QWEN_35_FLASH_02_23.value
    )
    assert target.intent.translation.openrouter_provider_routing == (
        OpenRouterProviderRouting.DEFAULT.value
    )
    assert target.intent.translation.connection == TranslationConnection.OPENROUTER.value
    assert serialization.to_dict(settings) == baseline


def test_byok_target_clears_managed_china_translation_state() -> None:
    settings = _vnext(
        model=TranslationModel.DEEPSEEK_V4_FLASH.value,
        connection=TranslationConnection.MANAGED_CHINA.value,
        openrouter_source="managed",
        openrouter_alias=OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED.value,
        openrouter_model=OpenRouterLLMModel.DEEPSEEK_V4_FLASH.value,
        openrouter_routing=OpenRouterProviderRouting.DEEPSEEK_ONLY.value,
        connection_history={
            TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.MANAGED_CHINA.value,
        },
    )

    target = _byok_target(settings)

    assert target is not None
    assert target.intent.translation.openrouter_selected_source == (
        OpenRouterCredentialSource.BYOK.value
    )
    assert target.intent.translation.openrouter_selection_alias == (
        OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_BYOK.value
    )
    assert target.intent.translation.openrouter_provider_routing == (
        OpenRouterProviderRouting.DEFAULT.value
    )
    assert target.intent.translation.connection == TranslationConnection.OPENROUTER.value
    assert (
        target.intent.translation.connection_history[TranslationModel.DEEPSEEK_V4_FLASH.value]
        == TranslationConnection.OPENROUTER.value
    )
