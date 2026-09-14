from __future__ import annotations

from puripuly_heart.config import llm_profiles, runtime_resolution
from puripuly_heart.config.settings_vnext import migration, serialization
from puripuly_heart.config.settings_vnext.schema import (
    VNEXT_SETTINGS_SCHEMA_VERSION,
    AppSettingsVNext,
)


def _runtime_input(
    *,
    model: str,
    connection: str = runtime_resolution.TRANSLATION_CONNECTION_OPENROUTER,
) -> runtime_resolution.RuntimeResolutionInput:
    return runtime_resolution.RuntimeResolutionInput(
        translation=runtime_resolution.TranslationRuntimeIntent(
            model=model,
            connection=connection,
        )
    )


def test_gemma_product_catalog_has_distinct_single_and_unified_profiles() -> None:
    assert llm_profiles.PROFILE_BY_ALIAS[
        llm_profiles.OPENROUTER_SELECTION_ALIAS_GEMMA4_26B_31B_BYOK
    ].openrouter_models == (
        llm_profiles.OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT,
        llm_profiles.OPENROUTER_MODEL_GEMMA_4_31B_IT,
    )
    assert llm_profiles.PROFILE_BY_ALIAS[
        llm_profiles.OPENROUTER_SELECTION_ALIAS_GEMMA4_31B_BYOK
    ].openrouter_models == (llm_profiles.OPENROUTER_MODEL_GEMMA_4_31B_IT,)
    assert llm_profiles.PROFILE_BY_ALIAS[
        llm_profiles.OPENROUTER_SELECTION_ALIAS_GEMMA4_BYOK
    ].openrouter_models == (llm_profiles.OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT,)


def test_runtime_resolves_unified_and_standalone_gemma_targets() -> None:
    unified = runtime_resolution.resolve_llm_config(
        _runtime_input(model=runtime_resolution.TRANSLATION_MODEL_GEMMA4_26B_31B)
    )
    standalone_31b = runtime_resolution.resolve_llm_config(
        _runtime_input(model=runtime_resolution.TRANSLATION_MODEL_GEMMA4_31B)
    )
    compatibility_26b = runtime_resolution.resolve_llm_config(
        _runtime_input(model=runtime_resolution.TRANSLATION_MODEL_GEMMA4)
    )

    assert unified.primary.models == (
        llm_profiles.OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT,
        llm_profiles.OPENROUTER_MODEL_GEMMA_4_31B_IT,
    )
    assert unified.primary.provider_routing == "gemma4_26b_31b_latency"
    assert standalone_31b.primary.models == (llm_profiles.OPENROUTER_MODEL_GEMMA_4_31B_IT,)
    assert standalone_31b.primary.provider_routing == "gemma4_31b_latency"
    assert compatibility_26b.primary.models == (llm_profiles.OPENROUTER_MODEL_GEMMA_4_26B_A4B_IT,)
    assert compatibility_26b.primary.provider_routing == "gemma4_26b_latency"


def test_runtime_resolves_three_stage_plan_without_deduplicating_targets() -> None:
    config = runtime_resolution.resolve_llm_config(
        _runtime_input(
            model=runtime_resolution.TRANSLATION_MODEL_GEMMA4_26B_31B,
        )
    )

    assert len(config.attempts) == 3
    assert config.attempts[0].start_after_ms == 0
    assert config.attempts[1].start_after_ms == 1300
    assert config.attempts[1].start_on_primary_error is True
    assert config.attempts[1].target == config.attempts[0].target
    assert config.attempts[2].start_after_ms == 4400
    assert config.attempts[2].start_on_primary_error is False
    assert config.attempts[2].target.model == llm_profiles.OPENROUTER_MODEL_GEMMA_4_31B_IT
    assert config.attempts[2].target.provider_routing == "gemma4_31b_modelrun_only"
    assert config.loser_grace_ms == 50


def test_non_openrouter_primary_does_not_get_emergency_attempt() -> None:
    config = runtime_resolution.resolve_llm_config(
        _runtime_input(
            model=runtime_resolution.TRANSLATION_MODEL_DEEPSEEK_V4_FLASH_41,
            connection=runtime_resolution.TRANSLATION_CONNECTION_OFFICIAL_BYOK,
        )
    )

    assert config.primary.provider == runtime_resolution.PROVIDER_DEEPSEEK
    assert len(config.attempts) == 2
    assert config.attempts[1].start_after_ms == 1300


def test_vnext_gemma_migration_is_idempotent_after_round_trip() -> None:
    raw = serialization.to_dict(AppSettingsVNext())
    raw["settings_version"] = 31
    raw["intent"]["translation"]["model"] = "gemma4"

    once = serialization.to_dict(migration.from_dict(raw))
    twice = serialization.to_dict(migration.from_dict(once))

    assert once == twice
    assert once["intent"]["translation"]["model"] == "gemma4_26b_31b"
    assert once["settings_version"] == VNEXT_SETTINGS_SCHEMA_VERSION
