from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

from puripuly_heart.app.wiring_provider_runtime import (
    project_translation_runtime_settings_from_vnext,
)
from puripuly_heart.app.wiring_translation_runtime_configuration import (
    build_translation_runtime_config,
    replace_translation_runtime_effective_flags,
    replace_translation_runtime_enabled,
    replace_translation_runtime_settings,
)

from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigurationOwner,
)


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
            prompts=replace(baseline.intent.prompts, system_prompt="runtime prompt"),
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
