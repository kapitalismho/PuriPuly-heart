from __future__ import annotations

import copy

from puripuly_heart.app.wiring_provider_runtime_policy import (
    build_llm_provider_signature,
)

from puripuly_heart.config.settings import (
    AppSettings,
    LLMProviderName,
    LocalLLMBackend,
    LocalLLMSettings,
    OpenRouterCredentialSource,
    OpenRouterProviderRouting,
    OpenRouterSelectionAlias,
    TranslationConnection,
    TranslationFallbackSettings,
    TranslationModel,
)


def test_llm_provider_signature_tracks_all_runtime_inputs() -> None:
    base = AppSettings()
    base.provider.llm = LLMProviderName.OPENROUTER
    base.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    base.openrouter.selection_alias = OpenRouterSelectionAlias.GEMMA4_MANAGED
    base.translation.fallback = TranslationFallbackSettings(
        enabled=True,
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.OPENROUTER,
    )
    different_selection = copy.deepcopy(base)
    different_selection.openrouter.selection_alias = OpenRouterSelectionAlias.QWEN35_FLASH_MANAGED
    different_fallback = copy.deepcopy(base)
    different_fallback.translation.fallback = TranslationFallbackSettings(enabled=False)

    assert build_llm_provider_signature(base) != build_llm_provider_signature(different_selection)
    assert build_llm_provider_signature(base) != build_llm_provider_signature(different_fallback)

    managed_fallback = AppSettings()
    managed_fallback.provider.llm = LLMProviderName.GEMINI
    managed_fallback.translation.fallback = TranslationFallbackSettings(
        enabled=True,
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.MANAGED_CHINA,
    )
    different_identity = copy.deepcopy(managed_fallback)
    different_identity.managed_identity.verified_hardware_hash = "fallback-managed-hash"
    assert build_llm_provider_signature(managed_fallback) != build_llm_provider_signature(
        different_identity
    )

    routed = copy.deepcopy(base)
    routed.openrouter.selection_alias = OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED
    routed.openrouter.provider_routing = OpenRouterProviderRouting.DEFAULT
    deepseek_only = copy.deepcopy(routed)
    deepseek_only.openrouter.provider_routing = OpenRouterProviderRouting.DEEPSEEK_ONLY
    assert build_llm_provider_signature(routed) != build_llm_provider_signature(deepseek_only)

    local = AppSettings()
    local.provider.llm = LLMProviderName.LOCAL_LLM
    local.local_llm = LocalLLMSettings(
        backend=LocalLLMBackend.OLLAMA,
        base_url="http://127.0.0.1:11434/v1",
        model="llama3.1:8b",
        extra_body={"thinking": {"type": "disabled", "budget": 0}},
    )
    same_json_different_order = copy.deepcopy(local)
    same_json_different_order.local_llm.extra_body = {"thinking": {"budget": 0, "type": "disabled"}}
    changed_model = copy.deepcopy(local)
    changed_model.local_llm.model = "qwen2.5:7b"
    changed_body = copy.deepcopy(local)
    changed_body.local_llm.extra_body = {"enable_thinking": False}

    assert build_llm_provider_signature(local) == build_llm_provider_signature(
        same_json_different_order
    )
    assert build_llm_provider_signature(local) != build_llm_provider_signature(changed_model)
    assert build_llm_provider_signature(local) != build_llm_provider_signature(changed_body)


def test_managed_gemma_signature_ignores_disabled_provider_fallback_but_tracks_prefix() -> None:
    base = AppSettings()
    base.translation.model = TranslationModel.MANAGED_GEMMA
    base.translation.connection = TranslationConnection.CPU
    base.provider.llm = LLMProviderName.MANAGED_GEMMA
    base.translation.fallback = TranslationFallbackSettings(
        enabled=True,
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.MANAGED_CHINA,
    )

    changed_fallback = copy.deepcopy(base)
    changed_fallback.translation.fallback = TranslationFallbackSettings(enabled=False)
    changed_fallback.openrouter.broker_base_url = "https://different.example"
    changed_fallback.managed_identity.verified_hardware_hash = "different"
    changed_language = copy.deepcopy(base)
    changed_language.languages.target_language = "ja"
    changed_prompt = copy.deepcopy(base)
    changed_prompt.system_prompt = "different prompt"
    changed_backend = copy.deepcopy(base)
    changed_backend.translation.connection = TranslationConnection.GPU

    assert build_llm_provider_signature(base) == build_llm_provider_signature(changed_fallback)
    assert build_llm_provider_signature(base) != build_llm_provider_signature(changed_language)
    assert build_llm_provider_signature(base) != build_llm_provider_signature(changed_prompt)
    assert build_llm_provider_signature(base) != build_llm_provider_signature(changed_backend)
