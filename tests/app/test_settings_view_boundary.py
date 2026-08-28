from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest
from puripuly_heart.app.services.settings_application import (
    materialize_immediate_settings_intent,
    materialize_prompt_apply_intent,
    materialize_provider_apply_intent,
    settings_view_surface_snapshots,
)

from puripuly_heart.app.adapters.ui_runtime import UiProviderRuntimeAdapter
from puripuly_heart.app.ports.settings_view import (
    AudioInputSettingsIntent,
    AudioSettingsIntent,
    ChatboxSourceSettingsIntent,
    CustomSttEndpointEdit,
    CustomVocabularySettingsIntent,
    DesktopOverlayBackgroundAlphaIntent,
    LocaleSettingsIntent,
    LocalLlmBaseUrlEdit,
    OverlayTargetSettingsIntent,
    PeerVadHangoverIntent,
    PromptApplyIntent,
    ProviderApplyIntent,
    QwenRegionEdit,
    SelfSttProviderEdit,
    SttGpuDeviceEdit,
    SystemPromptEdit,
    TranslationSelectionEdit,
    VrcMicInterceptSettingsIntent,
)
from puripuly_heart.app.services.canonical_settings_persistence import (
    materialize_compatibility_translation_settings,
)
from puripuly_heart.config.provider_values import (
    LLMProviderName,
    OpenRouterCredentialSource,
    OpenRouterLLMModel,
    OpenRouterSelectionAlias,
    QwenRegion,
    STTProviderName,
)
from puripuly_heart.config.settings import (
    AppSettings,
    build_managed_openrouter_byok_target_settings,
)
from puripuly_heart.config.translation_values import (
    TranslationConnection,
    TranslationModel,
)


def test_surface_projection_returns_independent_frozen_snapshots() -> None:
    settings = AppSettings()
    settings.languages.source_language = "ko"
    settings.stt.custom_terms = {"ko": ["PuriPuly"], "en": ["Avatar"]}

    provider, general, prompt, overlay = settings_view_surface_snapshots(settings)

    assert provider.translation.model == settings.translation.model
    assert general.locale == settings.ui.locale
    assert prompt.custom_vocabulary_terms == ("PuriPuly",)
    assert prompt.custom_vocabulary_other_languages_have_terms is True
    assert overlay.target == settings.overlay.target
    with pytest.raises(FrozenInstanceError):
        general.locale = "ja"


def test_immediate_intents_rebase_onto_latest_settings_without_surface_displacement() -> None:
    displayed = AppSettings()
    displayed.languages.source_language = "en"
    displayed.stt.custom_terms = {"en": ["old"], "ja": ["既存"]}
    current = AppSettings()
    current.languages.source_language = "ja"
    current.languages.target_language = "ko"
    current.stt.custom_terms = {"en": ["old"], "ja": ["最新"]}

    localized = materialize_immediate_settings_intent(current, LocaleSettingsIntent("ko"))
    updated = materialize_immediate_settings_intent(
        localized,
        CustomVocabularySettingsIntent("en", ("new",)),
    )

    assert updated.ui.locale == "ko"
    assert updated.languages.source_language == "ja"
    assert updated.languages.target_language == "ko"
    assert updated.stt.custom_terms == {"en": ["new"], "ja": ["最新"]}
    assert current.ui.locale == "en"
    assert displayed.stt.custom_terms["en"] == ["old"]


def test_custom_vocabulary_intent_derives_enabled_from_latest_rebased_terms() -> None:
    current = AppSettings()
    current.stt.custom_terms = {"en": ["stale"], "ja": ["latest"]}
    current.stt.custom_vocabulary_enabled = True

    updated = materialize_immediate_settings_intent(
        current,
        CustomVocabularySettingsIntent("en", ()),
    )

    assert updated.stt.custom_terms == {"en": [], "ja": ["latest"]}
    assert updated.stt.custom_vocabulary_enabled is True

    no_other_terms = AppSettings()
    no_other_terms.stt.custom_terms = {"en": ["stale"]}
    cleared = materialize_immediate_settings_intent(
        no_other_terms,
        CustomVocabularySettingsIntent("en", ()),
    )
    assert cleared.stt.custom_vocabulary_enabled is False


def test_focused_immediate_intents_preserve_latest_sibling_values() -> None:
    current = AppSettings()
    current.osc.connection_mode = "manual"
    current.osc.send_port = 9010
    current.osc.receive_port = 9011
    current.osc.vrc_mic_intercept = False
    current.osc.chatbox_include_source = False
    current.desktop_audio.vad_speech_threshold = 0.73
    current.desktop_audio.vad_hangover_ms = 900
    current.desktop_audio.vad_pre_roll_ms = 225
    current.overlay.target = "steamvr"
    current.overlay.show_translation = False
    current.overlay.desktop_flet.visual.background_alpha = 0.62
    current.overlay.desktop_flet.swap_caption_languages = True
    current.desktop_audio.output_device = "latest output"

    updated = materialize_immediate_settings_intent(
        current,
        AudioSettingsIntent((AudioInputSettingsIntent("MME", "staged microphone"),)),
    )
    updated = materialize_immediate_settings_intent(updated, VrcMicInterceptSettingsIntent(True))
    updated = materialize_immediate_settings_intent(
        updated,
        ChatboxSourceSettingsIntent(True),
    )
    updated = materialize_immediate_settings_intent(updated, PeerVadHangoverIntent(1200))
    updated = materialize_immediate_settings_intent(
        updated,
        OverlayTargetSettingsIntent("desktop"),
    )
    updated = materialize_immediate_settings_intent(
        updated,
        DesktopOverlayBackgroundAlphaIntent(0.4),
    )

    assert updated.osc.connection_mode == "manual"
    assert updated.osc.send_port == 9010
    assert updated.osc.receive_port == 9011
    assert updated.osc.vrc_mic_intercept is True
    assert updated.osc.chatbox_include_source is True
    assert updated.desktop_audio.vad_speech_threshold == 0.73
    assert updated.desktop_audio.vad_hangover_ms == 1200
    assert updated.desktop_audio.vad_pre_roll_ms == 225
    assert updated.overlay.target == "desktop"
    assert updated.overlay.show_translation is False
    assert updated.overlay.desktop_flet.visual.background_alpha == 0.4
    assert updated.overlay.desktop_flet.swap_caption_languages is True
    assert updated.audio.input_host_api == "MME"
    assert updated.audio.input_device == "staged microphone"
    assert updated.desktop_audio.output_device == "latest output"


def test_provider_edit_journal_replays_only_owned_fields_onto_latest_settings() -> None:
    displayed = AppSettings()
    displayed.translation.connection_history = {
        TranslationModel.GEMMA4.value: TranslationConnection.MANAGED,
        TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.MANAGED_CHINA,
    }
    provider, _general, _prompt, _overlay = settings_view_surface_snapshots(displayed)
    selection = replace(
        provider.translation,
        model=TranslationModel.GEMINI_37_FLASH,
        connection=TranslationConnection.OPENROUTER,
    )
    current = AppSettings()
    current.languages.source_language = "ja"
    current.audio.input_device = "latest microphone"
    current.translation.gpu_device_id = "latest-llm-gpu"
    current.translation.connection_history = {
        TranslationModel.GEMMA4.value: TranslationConnection.OPENROUTER,
        TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.OFFICIAL_BYOK,
    }
    current.custom_stt.model = "latest-custom-model"
    current.custom_stt.extra = {"latest": True}

    updated = materialize_provider_apply_intent(
        current,
        ProviderApplyIntent(
            (
                TranslationSelectionEdit(
                    selection,
                    ((TranslationModel.GEMINI_37_FLASH, TranslationConnection.OPENROUTER),),
                ),
                SelfSttProviderEdit(STTProviderName.DEEPGRAM),
                SttGpuDeviceEdit("staged-stt-gpu"),
                LocalLlmBaseUrlEdit("http://draft.local:11434"),
                CustomSttEndpointEdit("https://draft.invalid/v1/audio/transcriptions"),
                QwenRegionEdit(QwenRegion.SINGAPORE),
                SystemPromptEdit("focused prompt"),
            )
        ),
        materialize_translation=materialize_compatibility_translation_settings,
    )

    assert updated.provider.llm == LLMProviderName.OPENROUTER
    assert updated.translation.model == TranslationModel.GEMINI_37_FLASH
    assert updated.translation.connection == TranslationConnection.OPENROUTER
    assert updated.translation.connection_history[TranslationModel.GEMMA4.value] == (
        TranslationConnection.OPENROUTER
    )
    assert updated.translation.connection_history[TranslationModel.DEEPSEEK_V4_FLASH.value] == (
        TranslationConnection.OFFICIAL_BYOK
    )
    assert updated.translation.connection_history[TranslationModel.GEMINI_37_FLASH.value] == (
        TranslationConnection.OPENROUTER
    )
    assert updated.provider.stt == STTProviderName.DEEPGRAM
    assert updated.provider.peer_stt == current.provider.peer_stt
    assert updated.local_llm.base_url == "http://draft.local:11434"
    assert updated.local_llm.model == current.local_llm.model
    assert updated.local_llm.extra_body == current.local_llm.extra_body
    assert updated.stt.gpu_device_id == "staged-stt-gpu"
    assert updated.translation.gpu_device_id == "latest-llm-gpu"
    assert updated.custom_stt.endpoint == "https://draft.invalid/v1/audio/transcriptions"
    assert updated.custom_stt.model == "latest-custom-model"
    assert updated.custom_stt.extra == {"latest": True}
    assert updated.qwen.region == QwenRegion.SINGAPORE
    assert updated.system_prompt == "focused prompt"
    assert updated.languages.source_language == "ja"
    assert updated.audio.input_device == "latest microphone"


def test_prompt_intent_preserves_latest_languages_and_provider_selection() -> None:
    current = AppSettings()
    current.languages.source_language = "ja"
    current.languages.target_language = "ko"
    current.provider.llm = LLMProviderName.QWEN

    updated = materialize_prompt_apply_intent(current, PromptApplyIntent("new prompt"))

    assert updated.system_prompt == "new prompt"
    assert updated.system_prompts == {}
    assert updated.languages.source_language == "ja"
    assert updated.languages.target_language == "ko"
    assert updated.provider.llm == LLMProviderName.QWEN


def test_managed_byok_pkce_target_carries_focused_translation_change() -> None:
    current = AppSettings()
    current.provider.llm = LLMProviderName.OPENROUTER
    current.translation.connection = TranslationConnection.MANAGED
    current.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    current.openrouter.selection_alias = OpenRouterSelectionAlias.GEMMA4_MANAGED
    current.openrouter.llm_model = OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
    adapter = UiProviderRuntimeAdapter.__new__(UiProviderRuntimeAdapter)
    adapter.settings = SimpleNamespace(current=current)
    adapter.build_byok_target_settings = build_managed_openrouter_byok_target_settings

    target = adapter.build_managed_openrouter_byok_target()

    assert target is not None
    updated = materialize_provider_apply_intent(
        current,
        target.provider_intent,
        materialize_translation=materialize_compatibility_translation_settings,
    )
    assert updated.translation.connection == TranslationConnection.OPENROUTER
    assert current.translation.connection == TranslationConnection.MANAGED
