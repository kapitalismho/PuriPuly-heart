from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
from types import SimpleNamespace

import pytest
from puripuly_heart.app.wiring_provider_runtime import (
    ProviderRuntimeEffects,
    project_translation_runtime_settings_from_vnext,
)
from puripuly_heart.app.wiring_runtime_pipeline import runtime_pipeline_inputs_from_vnext
from puripuly_heart.app.wiring_translation_runtime_configuration import (
    build_translation_runtime_config,
    replace_translation_runtime_effective_flags,
    replace_translation_runtime_enabled,
    replace_translation_runtime_settings,
)

from puripuly_heart.config.llm_profiles import OPENROUTER_MODEL_GPT_6_LUNA
from puripuly_heart.config.prompts import get_translation_prompt_template
from puripuly_heart.config.runtime_resolution import OPENAI_MODEL_GPT_6_LUNA
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigurationOwner,
)


@pytest.mark.parametrize(
    ("model", "connection", "prompt_model"),
    (
        ("gpt_6_luna", "official_byok", OPENAI_MODEL_GPT_6_LUNA),
        ("gpt_6_luna", "openrouter", OPENROUTER_MODEL_GPT_6_LUNA),
        ("gpt_6_luna", "chatgpt", OPENAI_MODEL_GPT_6_LUNA),
        ("gemma4_26b_31b", "managed", None),
        ("deepseek_v4_flash", "openrouter", None),
        ("deepseek_v4_flash_41", "official_byok", None),
        ("gemini_flash", "official_byok", None),
        ("gemini_flash", "openrouter", None),
        ("qwen38_flash", "official_byok", None),
        ("managed_gemma", "cpu", None),
        ("managed_gemma", "gpu", None),
        ("local_llm", "ollama", None),
        ("custom_http", "custom_http", None),
    ),
)
@pytest.mark.parametrize("override", (None, "  Custom {source_lang} prompt.\n"))
def test_startup_runtime_selects_prompt_for_resolved_model(
    model: str,
    connection: str,
    prompt_model: str | None,
    override: str | None,
) -> None:
    baseline = AppSettingsVNext()
    settings = replace(
        baseline,
        intent=replace(
            baseline.intent,
            translation=replace(baseline.intent.translation, model=model, connection=connection),
            prompts=replace(baseline.intent.prompts, system_prompt_override=override),
        ),
    )

    inputs = runtime_pipeline_inputs_from_vnext(settings, peer_translation_enabled=False)

    expected = (
        override
        if override is not None and override.strip()
        else get_translation_prompt_template(model=prompt_model)
    )
    assert inputs.translation_runtime.system_prompt == expected
    assert settings.intent.prompts.system_prompt_override == override


@pytest.mark.parametrize("connection", ("official_byok", "openrouter", "chatgpt"))
@pytest.mark.parametrize("override", (None, "  Custom {target_lang} prompt.\n"))
def test_provider_apply_switches_default_prompt_without_changing_override(
    connection: str,
    override: str | None,
) -> None:
    baseline = AppSettingsVNext()
    settings = replace(
        baseline,
        intent=replace(
            baseline.intent,
            prompts=replace(baseline.intent.prompts, system_prompt_override=override),
        ),
    )
    owner = TranslationRuntimeConfigurationOwner(
        build_translation_runtime_config(project_translation_runtime_settings_from_vnext(settings))
    )
    effects = ProviderRuntimeEffects.__new__(ProviderRuntimeEffects)
    effects.settings = SimpleNamespace(canonical=settings)
    effects.canonical_settings = lambda value: value
    effects.clear_local_pending = lambda: None
    effects.sync_local_notice = lambda: None
    effects.managed_pending_sink = lambda _value: None
    effects.managed_pending_provider = lambda: False
    effects.dashboard_managed_pending_sink = lambda _value: None
    effects.translation_runtime_configuration_provider = lambda: owner
    effects.peer = lambda: SimpleNamespace(effective_enabled=lambda: False)
    luna_settings = replace(
        settings,
        intent=replace(
            settings.intent,
            translation=replace(
                settings.intent.translation,
                model="gpt_6_luna",
                connection=connection,
            ),
        ),
    )

    effects.apply_common(luna_settings)

    prompt_model = (
        OPENROUTER_MODEL_GPT_6_LUNA if connection == "openrouter" else OPENAI_MODEL_GPT_6_LUNA
    )
    assert owner.snapshot().value.system_prompt == (
        override if override is not None else get_translation_prompt_template(model=prompt_model)
    )
    assert effects.settings.canonical.intent.prompts.system_prompt_override == override

    effects.apply_common(settings)

    assert owner.snapshot().value.system_prompt == (
        override if override is not None else get_translation_prompt_template()
    )
    assert effects.settings.canonical.intent.prompts.system_prompt_override == override


def test_settings_replace_is_one_atomic_revision_and_preserves_runtime_only_values() -> None:
    initial = TranslationRuntimeConfig(
        fallback_transcript_only=True,
        translation_enabled=False,
        peer_translation_enabled=True,
        integrated_context_enabled=True,
        low_latency_finalize_wait_ms=225,
    )
    owner = TranslationRuntimeConfigurationOwner(initial)
    baseline = AppSettingsVNext()
    settings = replace(
        baseline,
        intent=replace(
            baseline.intent,
            languages=replace(
                baseline.intent.languages,
                source_language="ja",
                target_language="fr",
                peer_source_language="ko",
                peer_target_language="en",
            ),
            prompts=replace(
                baseline.intent.prompts,
                system_prompt_override="runtime prompt",
            ),
            osc=replace(baseline.intent.osc, chatbox_include_source=False),
            stt=replace(
                baseline.intent.stt,
                low_latency_merge_gap_ms=725,
                low_latency_spec_retry_max=3,
                low_latency_vad_hangover_ms=815,
            ),
            desktop_audio=replace(baseline.intent.desktop_audio, vad_hangover_ms=935),
        ),
    )
    settings_values = project_translation_runtime_settings_from_vnext(settings)

    change = replace_translation_runtime_settings(
        owner,
        settings_values,
        peer_translation_enabled=False,
        integrated_context_enabled=False,
    )

    assert change.before.revision == 0
    assert change.after.revision == 1
    assert owner.snapshot() is change.after
    assert change.after.value == build_translation_runtime_config(
        settings_values,
        current=initial,
        peer_translation_enabled=False,
        integrated_context_enabled=False,
    )
    assert change.after.value.fallback_transcript_only is True
    assert change.after.value.translation_enabled is False
    assert change.after.value.peer_translation_enabled is False
    assert change.after.value.integrated_context_enabled is False
    assert change.after.value.low_latency_finalize_wait_ms == 225


def test_effective_flag_replace_changes_both_flags_in_one_revision() -> None:
    owner = TranslationRuntimeConfigurationOwner()

    change = replace_translation_runtime_effective_flags(
        owner,
        peer_translation_enabled=True,
        integrated_context_enabled=True,
    )

    assert change.before.revision == 0
    assert change.after.revision == 1
    assert change.changed_fields == {
        "peer_translation_enabled",
        "integrated_context_enabled",
    }


def test_translation_enable_replace_preserves_every_other_value() -> None:
    initial = TranslationRuntimeConfig(system_prompt="prompt")
    owner = TranslationRuntimeConfigurationOwner(initial)

    change = replace_translation_runtime_enabled(owner, False)

    assert change.after.revision == 1
    assert change.after.value == replace(initial, translation_enabled=False)


def test_concurrent_cross_mutators_preserve_both_atomic_changes() -> None:
    barrier = Barrier(2)

    class CoordinatedOwner(TranslationRuntimeConfigurationOwner):
        def transform(self, transformer):
            barrier.wait(timeout=1)
            return super().transform(transformer)

    owner = CoordinatedOwner()

    with ThreadPoolExecutor(max_workers=2) as executor:
        enable_future = executor.submit(replace_translation_runtime_enabled, owner, False)
        flags_future = executor.submit(
            replace_translation_runtime_effective_flags,
            owner,
            peer_translation_enabled=True,
            integrated_context_enabled=True,
        )
        enable_future.result()
        flags_future.result()

    snapshot = owner.snapshot()
    assert snapshot.revision == 2
    assert snapshot.value.translation_enabled is False
    assert snapshot.value.peer_translation_enabled is True
    assert snapshot.value.integrated_context_enabled is True
