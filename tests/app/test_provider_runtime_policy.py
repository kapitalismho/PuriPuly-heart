from __future__ import annotations

from dataclasses import replace

from puripuly_heart.app.wiring_provider_runtime_policy import (
    build_llm_provider_signature,
)

from puripuly_heart.config.provider_values import OpenRouterSelectionAlias
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.openrouter_routing import OpenRouterProviderRouting


def _signature(settings: AppSettingsVNext) -> tuple[object, ...]:
    return build_llm_provider_signature(settings)


def _with_translation(settings: AppSettingsVNext, **changes: object) -> AppSettingsVNext:
    return replace(
        settings,
        intent=replace(
            settings.intent,
            translation=replace(settings.intent.translation, **changes),
        ),
    )


def test_llm_provider_signature_tracks_all_runtime_inputs() -> None:
    baseline = AppSettingsVNext()
    canonical = _with_translation(
        baseline,
        model="gemma4_26b_31b",
        connection="openrouter",
        openrouter_selected_source="managed",
        openrouter_selection_alias=OpenRouterSelectionAlias.GEMMA4_MANAGED.value,
    )
    different_selection = _with_translation(
        canonical,
        openrouter_selection_alias=OpenRouterSelectionAlias.QWEN35_FLASH_MANAGED.value,
    )

    assert _signature(canonical) != _signature(different_selection)

    direct = _with_translation(
        baseline,
        model="gemini_flash",
        connection="official_byok",
    )
    different_identity = replace(
        direct,
        state=replace(
            direct.state,
            managed_connection=replace(
                direct.state.managed_connection,
                verified_hardware_hash="unrelated-managed-hash",
            ),
        ),
    )
    assert _signature(direct) == _signature(different_identity)

    routed = _with_translation(
        canonical,
        openrouter_selection_alias=OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED.value,
        openrouter_provider_routing=OpenRouterProviderRouting.DEFAULT.value,
    )
    deepseek_only = _with_translation(
        routed,
        openrouter_provider_routing=OpenRouterProviderRouting.DEEPSEEK_ONLY.value,
    )
    assert _signature(routed) != _signature(deepseek_only)

    local = replace(
        baseline,
        intent=replace(
            baseline.intent,
            translation=replace(
                baseline.intent.translation,
                model="local_llm",
                connection="ollama",
            ),
            local_llm=replace(
                baseline.intent.local_llm,
                backend="ollama",
                base_url="http://127.0.0.1:11434/v1",
                model="llama3.1:8b",
                extra_body={"thinking": {"type": "disabled", "budget": 0}},
            ),
        ),
    )
    same_json_different_order = replace(
        local,
        intent=replace(
            local.intent,
            local_llm=replace(
                local.intent.local_llm,
                extra_body={"thinking": {"budget": 0, "type": "disabled"}},
            ),
        ),
    )
    changed_model = replace(
        local,
        intent=replace(
            local.intent,
            local_llm=replace(local.intent.local_llm, model="qwen2.5:7b"),
        ),
    )
    changed_body = replace(
        local,
        intent=replace(
            local.intent,
            local_llm=replace(local.intent.local_llm, extra_body={"enable_thinking": False}),
        ),
    )

    assert _signature(local) == _signature(same_json_different_order)
    assert _signature(local) != _signature(changed_model)
    assert _signature(local) != _signature(changed_body)


def test_managed_gemma_signature_tracks_prefix_and_ignores_cloud_state() -> None:
    baseline = AppSettingsVNext()
    base = _with_translation(
        baseline,
        model="managed_gemma",
        connection="cpu",
    )
    changed_cloud_state = replace(
        base,
        intent=replace(
            base.intent,
            translation=replace(
                base.intent.translation,
                openrouter_broker_base_url="https://different.example",
            ),
        ),
        state=replace(
            base.state,
            managed_connection=replace(
                base.state.managed_connection,
                verified_hardware_hash="different",
            ),
        ),
    )
    changed_language = replace(
        base,
        intent=replace(
            base.intent,
            languages=replace(base.intent.languages, target_language="ja"),
        ),
    )
    changed_prompt = replace(
        base,
        intent=replace(
            base.intent,
            prompts=replace(
                base.intent.prompts,
                system_prompt_override="different prompt",
            ),
        ),
    )
    changed_backend = _with_translation(base, connection="gpu")

    assert _signature(base) == _signature(changed_cloud_state)
    assert _signature(base) != _signature(changed_language)
    assert _signature(base) != _signature(changed_prompt)
    assert _signature(base) != _signature(changed_backend)
