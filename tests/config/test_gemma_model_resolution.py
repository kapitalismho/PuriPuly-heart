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
    fallback: runtime_resolution.TranslationFallbackRuntimeIntent | None = None,
) -> runtime_resolution.RuntimeResolutionInput:
    return runtime_resolution.RuntimeResolutionInput(
        translation=runtime_resolution.TranslationRuntimeIntent(
            model=model,
            connection=connection,
        ),
        translation_fallback=fallback or runtime_resolution.TranslationFallbackRuntimeIntent(),
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
            fallback=runtime_resolution.TranslationFallbackRuntimeIntent(
                enabled=True,
                model=runtime_resolution.TRANSLATION_MODEL_GEMMA4_26B_31B,
                connection=runtime_resolution.TRANSLATION_CONNECTION_OPENROUTER,
            ),
        )
    )

    assert len(config.attempts) == 3
    assert config.attempts[0].start_after_ms == 0
    assert config.attempts[1].start_after_ms == 1300
    assert config.attempts[1].start_on_primary_error is True
    assert config.attempts[1].target == config.attempts[0].target
    assert config.attempts[2].start_after_ms == 4500
    assert config.attempts[2].start_on_primary_error is False
    assert config.attempts[2].target.model == llm_profiles.OPENROUTER_MODEL_GEMMA_4_31B_IT
    assert config.attempts[2].target.provider_routing == "gemma4_31b_cerebras_only"
    assert config.loser_grace_ms == 50


def test_non_openrouter_primary_does_not_get_emergency_attempt() -> None:
    config = runtime_resolution.resolve_llm_config(
        _runtime_input(
            model=runtime_resolution.TRANSLATION_MODEL_DEEPSEEK_V4_FLASH,
            connection=runtime_resolution.TRANSLATION_CONNECTION_OFFICIAL_BYOK,
            fallback=runtime_resolution.TranslationFallbackRuntimeIntent(
                enabled=True,
                model=runtime_resolution.TRANSLATION_MODEL_GEMMA4_31B,
                connection=runtime_resolution.TRANSLATION_CONNECTION_OPENROUTER,
            ),
        )
    )

    assert config.primary.provider == runtime_resolution.PROVIDER_DEEPSEEK
    assert len(config.attempts) == 2
    assert config.attempts[1].start_after_ms == 1300


def test_vnext_migration_converts_primary_and_only_unmarked_gemma_fallback() -> None:
    raw = serialization.to_dict(AppSettingsVNext())
    raw["settings_version"] = 31
    translation = raw["intent"]["translation"]
    translation["model"] = "gemma4"
    translation["connection"] = "openrouter"
    translation["connection_history"] = {"gemma4": "openrouter"}
    translation["fallback"] = {
        "enabled": True,
        "model": "gemma4",
        "connection": "managed",
    }

    migrated = migration.from_dict(raw)
    migrated_translation = migrated.intent.translation

    assert migrated_translation.model == "gemma4_26b_31b"
    assert migrated_translation.connection == "openrouter"
    assert migrated_translation.connection_history["gemma4_26b_31b"] == "openrouter"
    assert migrated_translation.fallback.model == "gemma4_26b_31b"
    assert migrated_translation.fallback.connection == "managed"
    assert migrated_translation.fallback.selection_alias == "managed_gemma4_26b_31b"


def test_vnext_migration_preserves_explicit_old_and_deepseek_fallbacks() -> None:
    raw = serialization.to_dict(AppSettingsVNext())
    raw["settings_version"] = 31
    translation = raw["intent"]["translation"]
    translation["model"] = "gemma4"
    translation["fallback"] = {
        "enabled": True,
        "model": "gemma4",
        "connection": "openrouter",
        "selection_alias": "openrouter_gemma4_26b_a4b",
    }
    preserved_old = migration.from_dict(raw).intent.translation.fallback
    assert preserved_old.model == "gemma4"
    assert preserved_old.selection_alias == "openrouter_gemma4_26b_a4b"

    translation["fallback"] = {
        "enabled": True,
        "model": "deepseek_v4_flash",
        "connection": "openrouter",
        "selection_alias": "openrouter_deepseek_v4_flash",
    }
    preserved_deepseek = migration.from_dict(raw).intent.translation.fallback
    assert preserved_deepseek.model == "deepseek_v4_flash"
    assert preserved_deepseek.selection_alias == "openrouter_deepseek_v4_flash"


def test_vnext_gemma_migration_is_idempotent_after_round_trip() -> None:
    raw = serialization.to_dict(AppSettingsVNext())
    raw["settings_version"] = 31
    raw["intent"]["translation"]["model"] = "gemma4"

    once = serialization.to_dict(migration.from_dict(raw))
    twice = serialization.to_dict(migration.from_dict(once))

    assert once == twice
    assert once["intent"]["translation"]["model"] == "gemma4_26b_31b"
    assert once["settings_version"] == VNEXT_SETTINGS_SCHEMA_VERSION
    assert once["intent"]["translation"]["fallback"] == {
        "enabled": True,
        "model": "gemma4_26b_31b",
        "connection": "openrouter",
        "selection_alias": "openrouter_gemma4_26b_31b",
    }


def test_missing_vnext_fallback_uses_unified_gemma_default() -> None:
    raw = serialization.to_dict(AppSettingsVNext())
    raw["intent"]["translation"].pop("fallback")

    fallback = migration.from_dict(raw).intent.translation.fallback

    assert fallback.enabled is True
    assert fallback.model == "gemma4_26b_31b"
    assert fallback.connection == "openrouter"
    assert fallback.selection_alias == "openrouter_gemma4_26b_31b"
