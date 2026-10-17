from __future__ import annotations

from dataclasses import replace

import pytest

from puripuly_heart.app.services.canonical_settings_persistence import (
    materialize_canonical_translation_settings,
)
from puripuly_heart.app.services.osc.control_application import (
    OscControlApplyResult,
    SettingsBackedOscControlApplication,
)
from puripuly_heart.app.wiring.wiring_llm_factory import runtime_resolution_input_from_vnext
from puripuly_heart.app.wiring.wiring_provider_runtime_policy import (
    provider_llm_for_translation,
)
from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.config.runtime_resolution import resolve_llm_config
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.config.translation_values import TranslationConnection, TranslationModel


def _with_translation(settings: AppSettingsVNext, **changes: object) -> AppSettingsVNext:
    return replace(
        settings,
        intent=replace(
            settings.intent,
            translation=replace(settings.intent.translation, **changes),
        ),
    )


def _with_languages(settings: AppSettingsVNext, **changes: object) -> AppSettingsVNext:
    return replace(
        settings,
        intent=replace(
            settings.intent,
            languages=replace(settings.intent.languages, **changes),
        ),
    )


def _with_stt_provider(settings: AppSettingsVNext, provider: str) -> AppSettingsVNext:
    return replace(
        settings,
        intent=replace(
            settings.intent,
            stt=replace(settings.intent.stt, provider=provider),
        ),
    )


def _with_peer_stt_provider(settings: AppSettingsVNext, provider: str) -> AppSettingsVNext:
    return replace(
        settings,
        intent=replace(
            settings.intent,
            peer_stt=replace(settings.intent.peer_stt, provider=provider),
        ),
    )


def _with_osc(settings: AppSettingsVNext, **changes: object) -> AppSettingsVNext:
    return replace(
        settings,
        intent=replace(
            settings.intent,
            osc=replace(settings.intent.osc, **changes),
        ),
    )


@pytest.mark.asyncio
async def test_translation_model_control_materializes_provider_and_connection() -> None:
    current = _with_translation(AppSettingsVNext(), connection=TranslationConnection.MANAGED.value)
    applied: list[AppSettingsVNext] = []

    async def apply_settings(settings: object) -> object:
        nonlocal current
        assert isinstance(settings, AppSettingsVNext)
        applied.append(settings)
        current = settings
        return settings

    application = SettingsBackedOscControlApplication(
        settings_provider=lambda: current,
        apply_settings=apply_settings,
        translation_model_normalizer=materialize_canonical_translation_settings,
    )

    result = await application.set_translation_model(TranslationModel.DEEPSEEK_V4_FLASH.value)

    assert result is applied[0]
    updated = applied[0]
    assert updated.intent.translation.model == TranslationModel.DEEPSEEK_V4_FLASH.value
    assert updated.intent.translation.connection == TranslationConnection.OPENROUTER.value
    assert (
        provider_llm_for_translation(
            updated.intent.translation.model,
            updated.intent.translation.connection,
        )
        == "openrouter"
    )


@pytest.mark.asyncio
async def test_osc_luna_first_selection_and_saved_route_survive_model_switch() -> None:
    from puripuly_heart.config.settings_vnext import serialization
    from puripuly_heart.config.settings_vnext.migration import from_dict

    current = AppSettingsVNext()

    async def apply_settings(settings: object) -> object:
        nonlocal current
        assert isinstance(settings, AppSettingsVNext)
        current = from_dict(serialization.to_dict(settings))
        return current

    application = SettingsBackedOscControlApplication(
        settings_provider=lambda: current,
        apply_settings=apply_settings,
        translation_model_normalizer=materialize_canonical_translation_settings,
    )
    await application.set_translation_model("gpt_6_luna")
    assert current.intent.translation.connection == "chatgpt"
    assert current.intent.translation.openrouter_selected_source == "none"
    chatgpt_runtime = resolve_llm_config(runtime_resolution_input_from_vnext(current)).primary
    assert (chatgpt_runtime.provider, chatgpt_runtime.model) == ("chatgpt", "gpt-6-luna")
    assert chatgpt_runtime.credential.reference == "chatgpt:oauth"

    await application.set_translation_model("gpt_6_luna", "official_byok")
    assert current.intent.translation.connection == "official_byok"
    await application.set_translation_model("gemma4_26b_31b")
    assert current.intent.translation.connection == "managed"
    assert current.intent.translation.connection_history["gemma4_26b_31b"] == "managed"
    gemma_runtime = resolve_llm_config(runtime_resolution_input_from_vnext(current)).primary
    assert (gemma_runtime.provider, gemma_runtime.model) == (
        "openrouter",
        "google/gemma-4-26b-a4b-it",
    )
    assert (gemma_runtime.credential.source, gemma_runtime.credential.reference) == (
        "managed",
        "openrouter:managed",
    )
    assert current.intent.translation.connection_history["gpt_6_luna"] == "official_byok"
    await application.set_translation_model("gpt_6_luna")
    assert current.intent.translation.connection == "official_byok"
    assert current.intent.translation.openrouter_selection_alias is None
    luna_runtime = resolve_llm_config(runtime_resolution_input_from_vnext(current)).primary
    assert (luna_runtime.provider, luna_runtime.model) == ("openai", "gpt-6-luna")
    assert luna_runtime.credential.reference == "openai:byok"


@pytest.mark.asyncio
async def test_osc_model_only_selection_restores_other_model_history_or_default() -> None:
    current = _with_translation(
        AppSettingsVNext(),
        model="gemma4_26b_31b",
        connection="openrouter",
        connection_history={"gemma4_26b_31b": "openrouter", "gpt_6_luna": "official_byok"},
    )

    async def apply_settings(settings: object) -> object:
        nonlocal current
        assert isinstance(settings, AppSettingsVNext)
        current = settings
        return current

    application = SettingsBackedOscControlApplication(
        settings_provider=lambda: current,
        apply_settings=apply_settings,
        translation_model_normalizer=materialize_canonical_translation_settings,
    )
    await application.set_translation_model("gpt_6_luna")
    assert current.intent.translation.connection == "official_byok"
    await application.set_translation_model("gemma4_26b_31b")
    assert current.intent.translation.connection == "openrouter"
    assert current.intent.translation.connection_history == {
        "gemma4_26b_31b": "openrouter",
        "gpt_6_luna": "official_byok",
    }
    gemma_runtime = resolve_llm_config(runtime_resolution_input_from_vnext(current)).primary
    assert (gemma_runtime.credential.source, gemma_runtime.credential.reference) == (
        "secret_store",
        "openrouter:byok",
    )

    await application.set_translation_model("gemini_flash")
    assert current.intent.translation.connection == "official_byok"
    gemini_runtime = resolve_llm_config(runtime_resolution_input_from_vnext(current)).primary
    assert (gemini_runtime.provider, gemini_runtime.credential.reference) == (
        "gemini",
        "gemini:byok",
    )
    await application.set_translation_model("gpt_6_luna")
    assert current.intent.translation.connection == "official_byok"
    assert current.intent.translation.connection_history["gemma4_26b_31b"] == "openrouter"


@pytest.mark.asyncio
async def test_osc_same_model_repairs_invalid_connection_from_saved_history() -> None:
    current = _with_translation(
        AppSettingsVNext(),
        model="gemma4_26b_31b",
        connection="official_byok",
        connection_history={"gemma4_26b_31b": "openrouter"},
    )

    async def apply_settings(settings: object) -> object:
        nonlocal current
        assert isinstance(settings, AppSettingsVNext)
        current = settings
        return settings

    application = SettingsBackedOscControlApplication(
        settings_provider=lambda: current,
        apply_settings=apply_settings,
        translation_model_normalizer=materialize_canonical_translation_settings,
    )
    await application.set_translation_model("gemma4_26b_31b")
    assert current.intent.translation.connection == "openrouter"
    assert current.intent.translation.connection_history["gemma4_26b_31b"] == "openrouter"
    runtime = resolve_llm_config(runtime_resolution_input_from_vnext(current)).primary
    assert runtime.credential.reference == "openrouter:byok"


@pytest.mark.asyncio
async def test_osc_model_only_restores_regional_managed_connection_history() -> None:
    current = materialize_canonical_translation_settings(
        _with_translation(
            AppSettingsVNext(),
            model="deepseek_v4_flash_41",
            connection="managed_china",
            connection_history={"deepseek_v4_flash_41": "managed_china"},
        )
    )

    async def apply_settings(settings: object) -> object:
        nonlocal current
        assert isinstance(settings, AppSettingsVNext)
        current = settings
        return settings

    application = SettingsBackedOscControlApplication(
        settings_provider=lambda: current,
        apply_settings=apply_settings,
        translation_model_normalizer=materialize_canonical_translation_settings,
    )
    await application.set_translation_model("gpt_6_luna")
    assert current.intent.translation.connection == "chatgpt"
    await application.set_translation_model("deepseek_v4_flash_41")
    assert current.intent.translation.model == "deepseek_v4_flash_41"
    assert current.intent.translation.connection == "managed_china"
    assert current.intent.translation.connection_history["deepseek_v4_flash_41"] == "managed_china"
    runtime = resolve_llm_config(runtime_resolution_input_from_vnext(current)).primary
    assert runtime.credential.reference == "openrouter:managed_qq"


@pytest.mark.asyncio
async def test_managed_local_models_control_materializes_provider_and_connection() -> None:
    current = _with_translation(AppSettingsVNext(), connection=TranslationConnection.MANAGED.value)
    applied: list[AppSettingsVNext] = []

    async def apply_settings(settings: object) -> object:
        nonlocal current
        assert isinstance(settings, AppSettingsVNext)
        applied.append(settings)
        current = settings
        return settings

    application = SettingsBackedOscControlApplication(
        settings_provider=lambda: current,
        apply_settings=apply_settings,
        translation_model_normalizer=materialize_canonical_translation_settings,
    )

    await application.set_translation_model(
        TranslationModel.MANAGED_GEMMA.value,
        TranslationConnection.CPU.value,
    )

    updated = applied[0]
    assert updated.intent.translation.model == TranslationModel.MANAGED_GEMMA.value
    assert updated.intent.translation.connection == TranslationConnection.CPU.value
    assert (
        provider_llm_for_translation(
            updated.intent.translation.model,
            updated.intent.translation.connection,
        )
        == "managed_gemma"
    )

    await application.set_translation_model(
        TranslationModel.MANAGED_GEMMA.value,
        TranslationConnection.GPU.value,
    )

    updated = applied[1]
    assert updated.intent.translation.model == TranslationModel.MANAGED_GEMMA.value
    assert updated.intent.translation.connection == TranslationConnection.GPU.value


@pytest.mark.asyncio
async def test_custom_http_control_preserves_the_previous_llm_selection() -> None:
    current = _with_translation(
        AppSettingsVNext(),
        model=TranslationModel.DEEPSEEK_V4_FLASH.value,
        connection=TranslationConnection.OFFICIAL_BYOK.value,
        http_extension_id="demo",
    )

    async def apply_settings(settings: object) -> object:
        nonlocal current
        assert isinstance(settings, AppSettingsVNext)
        current = settings
        return settings

    application = SettingsBackedOscControlApplication(
        settings_provider=lambda: current,
        apply_settings=apply_settings,
        translation_model_normalizer=materialize_canonical_translation_settings,
    )

    await application.set_translation_model(TranslationModel.CUSTOM_HTTP.value)

    assert current.intent.translation.model == TranslationModel.CUSTOM_HTTP.value
    assert current.intent.translation.connection == TranslationConnection.CUSTOM_HTTP.value
    assert current.intent.translation.http_extension_id == "demo"
    assert current.intent.translation.previous_llm_model == TranslationModel.DEEPSEEK_V4_FLASH.value


@pytest.mark.asyncio
async def test_settings_control_rejects_when_application_keeps_the_previous_state() -> None:
    current = AppSettingsVNext()

    async def apply_settings(_settings: object) -> object:
        return True

    application = SettingsBackedOscControlApplication(
        settings_provider=lambda: current,
        apply_settings=apply_settings,
        translation_model_normalizer=materialize_canonical_translation_settings,
    )

    result = await application.set_mute_sync(True)

    assert result is False
    assert current.intent.osc.vrc_mic_intercept is False


@pytest.mark.asyncio
async def test_settings_control_reports_a_committed_normalized_canonical_state() -> None:
    current = _with_stt_provider(AppSettingsVNext(), STTProviderName.DEEPGRAM.value)

    async def apply_settings(settings: object) -> object:
        nonlocal current
        assert isinstance(settings, AppSettingsVNext)
        current = _with_stt_provider(settings, STTProviderName.LOCAL_CPU_AUTO.value)
        return True

    application = SettingsBackedOscControlApplication(
        settings_provider=lambda: current,
        apply_settings=apply_settings,
        translation_model_normalizer=materialize_canonical_translation_settings,
    )

    result = await application.set_self_asr(STTProviderName.LOCAL_QWEN_GPU.value)

    assert result == OscControlApplyResult(
        applied=False,
        canonical_state_changed=True,
    )
    assert current.intent.stt.provider == STTProviderName.LOCAL_CPU_AUTO.value


@pytest.mark.asyncio
async def test_set_languages_skips_apply_when_every_value_matches() -> None:
    current = _with_languages(
        AppSettingsVNext(),
        source_language="fr",
        target_language="fr",
        peer_source_language="fr",
        peer_target_language="fr",
    )
    applied = 0

    async def apply_settings(_settings: object) -> object:
        nonlocal applied
        applied += 1
        return True

    application = SettingsBackedOscControlApplication(
        settings_provider=lambda: current,
        apply_settings=apply_settings,
        translation_model_normalizer=materialize_canonical_translation_settings,
    )

    result = await application.set_languages(
        self_source="fr",
        self_target="fr",
        peer_source="fr",
        peer_target="fr",
    )

    assert result is True
    assert applied == 0


@pytest.mark.asyncio
async def test_set_languages_applies_when_any_value_differs() -> None:
    current = _with_languages(
        AppSettingsVNext(),
        source_language="ko",
        target_language="en",
        peer_source_language="en",
        peer_target_language="ko",
    )
    applied = 0

    async def apply_settings(settings: object) -> object:
        nonlocal current
        nonlocal applied
        assert isinstance(settings, AppSettingsVNext)
        applied += 1
        current = settings
        return True

    application = SettingsBackedOscControlApplication(
        settings_provider=lambda: current,
        apply_settings=apply_settings,
        translation_model_normalizer=materialize_canonical_translation_settings,
    )

    result = await application.set_languages(
        self_source="ko",
        self_target="fr",
        peer_source="en",
        peer_target="ko",
    )

    assert result is True
    assert applied == 1
    assert current.intent.languages.target_language == "fr"


@pytest.mark.asyncio
async def test_primary_osc_target_change_clears_an_equal_secondary_target() -> None:
    current = _with_languages(
        AppSettingsVNext(),
        target_language="en",
        secondary_target_language="fr",
    )

    async def apply_settings(settings: object) -> object:
        nonlocal current
        assert isinstance(settings, AppSettingsVNext)
        current = settings
        return True

    application = SettingsBackedOscControlApplication(
        settings_provider=lambda: current,
        apply_settings=apply_settings,
        translation_model_normalizer=materialize_canonical_translation_settings,
    )

    result = await application.set_languages(
        self_source="ko",
        self_target="fr",
        peer_source="en",
        peer_target="ko",
    )

    assert result is True
    assert current.intent.languages.target_language == "fr"
    assert current.intent.languages.secondary_target_language == ""


@pytest.mark.asyncio
async def test_primary_osc_target_change_preserves_a_distinct_secondary_target() -> None:
    current = _with_languages(
        AppSettingsVNext(),
        target_language="en",
        secondary_target_language="ja",
    )

    async def apply_settings(settings: object) -> object:
        nonlocal current
        assert isinstance(settings, AppSettingsVNext)
        current = settings
        return True

    application = SettingsBackedOscControlApplication(
        settings_provider=lambda: current,
        apply_settings=apply_settings,
        translation_model_normalizer=materialize_canonical_translation_settings,
    )

    result = await application.set_languages(
        self_source="ko",
        self_target="fr",
        peer_source="en",
        peer_target="ko",
    )

    assert result is True
    assert current.intent.languages.target_language == "fr"
    assert current.intent.languages.secondary_target_language == "ja"

    result = await application.set_secondary_target_language("de")

    assert result is True
    assert current.intent.languages.secondary_target_language == "de"

    result = await application.set_secondary_target_language("fr")

    assert result is True
    assert current.intent.languages.secondary_target_language == ""


@pytest.mark.asyncio
async def test_asr_controls_skip_apply_when_provider_matches() -> None:
    current = _with_peer_stt_provider(
        _with_stt_provider(AppSettingsVNext(), "deepgram"),
        "soniox",
    )
    applied = 0

    async def apply_settings(settings: object) -> object:
        nonlocal current
        nonlocal applied
        assert isinstance(settings, AppSettingsVNext)
        applied += 1
        current = settings
        return True

    application = SettingsBackedOscControlApplication(
        settings_provider=lambda: current,
        apply_settings=apply_settings,
        translation_model_normalizer=materialize_canonical_translation_settings,
    )

    assert (await application.set_self_asr("deepgram")) is True
    assert (await application.set_peer_asr("soniox")) is True
    assert applied == 0

    assert (await application.set_self_asr("local_parakeet_v3")) is True

    assert applied == 1
    assert current.intent.stt.provider == "local_parakeet_v3"
    assert current.intent.peer_stt.provider == "soniox"


@pytest.mark.asyncio
async def test_auxiliary_controls_skip_apply_when_unchanged() -> None:
    current = _with_languages(
        _with_osc(
            AppSettingsVNext(),
            vrc_mic_intercept=False,
            chatbox_include_source=False,
        ),
        peer_source_mode="manual",
    )
    applied = 0

    async def apply_settings(settings: object) -> object:
        nonlocal current
        nonlocal applied
        assert isinstance(settings, AppSettingsVNext)
        applied += 1
        current = settings
        return True

    application = SettingsBackedOscControlApplication(
        settings_provider=lambda: current,
        apply_settings=apply_settings,
        translation_model_normalizer=materialize_canonical_translation_settings,
    )

    assert (await application.set_mute_sync(False)) is True
    assert (await application.set_chatbox_source(False)) is True
    assert (await application.set_peer_auto_detect(False)) is True
    assert applied == 0

    assert (await application.set_mute_sync(True)) is True
    assert applied == 1
    assert current.intent.osc.vrc_mic_intercept is True

    assert (await application.set_chatbox_source(True)) is True
    assert applied == 2
    assert current.intent.osc.chatbox_include_source is True

    assert (await application.set_peer_auto_detect(True)) is True
    assert applied == 3
    assert current.intent.languages.peer_source_mode == "auto"
