from __future__ import annotations

from dataclasses import replace

import pytest

from puripuly_heart.app.services.canonical_settings_persistence import (
    compose_settings_owner,
    materialize_canonical_translation_settings,
)
from puripuly_heart.app.wiring.wiring_llm_factory import runtime_resolution_input_from_vnext
from puripuly_heart.config.runtime_resolution import (
    derive_translation_runtime_intent_from_compatibility,
    normalize_translation_runtime_intent,
    resolve_llm_config,
)
from puripuly_heart.config.settings_vnext import migration, serialization
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, TranslationIntent
from puripuly_heart.config.translation_values import (
    TranslationConnection,
    TranslationModel,
    default_translation_connection,
    provider_llm_for_translation,
    supported_translation_connections,
)


def test_index_is_a_distinct_public_api_selection():
    assert supported_translation_connections(TranslationModel.INDEX_TRANSLATE_35B) == (
        TranslationConnection.PUBLIC_API,
    )
    assert default_translation_connection(TranslationModel.INDEX_TRANSLATE_35B) == (
        TranslationConnection.PUBLIC_API
    )
    assert provider_llm_for_translation("index_translate_35b", "public_api") == "index_translate"
    intent = normalize_translation_runtime_intent(
        model="index_translate_35b", connection="openrouter"
    )
    assert intent.model == "index_translate_35b"
    assert intent.connection == "public_api"
    compatible = derive_translation_runtime_intent_from_compatibility(
        provider_llm="index_translate",
        openrouter_model="deepseek/deepseek-v4.1-flash",
        openrouter_selected_source="managed",
    )
    assert compatible == intent


def test_schema_backfill_keeps_previous_default_instead_of_first_run_index():
    assert TranslationIntent().model == "gemma4_26b_31b"
    restored = serialization.from_dict({"intent": {}, "state": {}})
    assert restored.intent.translation.model == "gemma4_26b_31b"
    assert restored.intent.translation.connection == "managed"


@pytest.mark.parametrize(
    ("model", "connection"),
    [
        (model.value, connection.value)
        for model in TranslationModel
        if model != TranslationModel.INDEX_TRANSLATE_35B
        for connection in supported_translation_connections(model)
    ],
)
def test_existing_supported_selection_prompt_and_connection_history_survive_load(
    tmp_path, model, connection
):
    settings = AppSettingsVNext()
    history = {
        "gemma4_26b_31b": "openrouter",
        "deepseek_v4_flash_41": "managed_china",
        "gpt_6_luna": "official_byok",
        model: connection,
    }
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            translation=replace(
                settings.intent.translation,
                model=model,
                connection=connection,
                connection_history=history,
            ),
            prompts=replace(settings.intent.prompts, system_prompt_override="Preserved override"),
            ui=replace(settings.intent.ui, locale="zh-CN"),
        ),
    )
    settings = migration.from_dict(serialization.to_dict(settings))
    settings = materialize_canonical_translation_settings(settings)
    path = tmp_path / "settings.json"
    owner = compose_settings_owner(path)
    owner.canonical = settings
    owner.persist()
    loaded = compose_settings_owner(path).start().settings
    assert loaded.intent.translation == settings.intent.translation
    assert loaded.intent.translation.model == model
    assert loaded.intent.translation.connection == connection
    assert loaded.intent.translation.connection_history == history
    assert loaded.intent.prompts == settings.intent.prompts
    assert loaded.intent.ui == settings.intent.ui
    assert loaded.state.managed_connection == settings.state.managed_connection


def test_saved_index_cannot_be_reinterpreted_by_stale_openrouter_alias(tmp_path):
    settings = AppSettingsVNext()
    translation = replace(
        settings.intent.translation,
        model="index_translate_35b",
        connection="public_api",
        connection_history={"gpt_6_luna": "official_byok", "index_translate_35b": "public_api"},
        openrouter_selection_alias="deepseek_v4_flash_41_managed",
        openrouter_selected_source="managed",
        openrouter_provider_routing="deepseek_v4_flash_41_strict",
    )
    settings = replace(settings, intent=replace(settings.intent, translation=translation))
    config = resolve_llm_config(runtime_resolution_input_from_vnext(settings))
    assert config.primary.provider == "index_translate"
    assert config.primary.credential.source == "none"
    materialized = materialize_canonical_translation_settings(settings)
    assert materialized.intent.translation.connection == "public_api"
    assert materialized.intent.translation.openrouter_selection_alias is None
    assert materialized.intent.translation.openrouter_selected_source == "none"
    assert materialized.intent.translation.openrouter_provider_routing == "default"
    assert materialized.intent.translation.connection_history == translation.connection_history
    path = tmp_path / "settings.json"
    owner = compose_settings_owner(path)
    owner.canonical = materialized
    owner.persist()
    restored = compose_settings_owner(path).start().settings
    assert restored.intent.translation == materialized.intent.translation
    assert resolve_llm_config(runtime_resolution_input_from_vnext(restored)) == config
