from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from puripuly_heart.app.services.settings_application import (
    materialize_immediate_settings_intent,
    materialize_prompt_apply_intent,
    materialize_provider_apply_intent,
    settings_view_surface_snapshots,
)

from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
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
    PeerSttRollingEnabledEdit,
    PeerVadHangoverIntent,
    PromptApplyIntent,
    ProviderApplyIntent,
    QwenAsrModelEdit,
    QwenRegionEdit,
    SelfSttProviderEdit,
    SttGpuDeviceEdit,
    SttRollingEnabledEdit,
    SystemPromptEdit,
    TranslationSelectionEdit,
    VrcMicInterceptSettingsIntent,
)
from puripuly_heart.app.services.canonical_settings_persistence import (
    SettingsOwner,
    materialize_canonical_translation_settings,
)
from puripuly_heart.app.wiring.wiring_provider_runtime_policy import (
    provider_llm_for_translation,
)
from puripuly_heart.config.provider_values import (
    OpenRouterCredentialSource,
    QwenRegion,
    STTProviderName,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.config.translation_values import (
    TranslationConnection,
    TranslationModel,
)


def _vnext(settings: AppSettingsVNext | None = None, **intent_fields: object) -> AppSettingsVNext:
    current = AppSettingsVNext() if settings is None else settings
    return replace(current, intent=replace(current.intent, **intent_fields))


def test_surface_projection_returns_independent_frozen_snapshots() -> None:
    settings = _vnext(
        languages=replace(AppSettingsVNext().intent.languages, source_language="ko"),
        stt=replace(
            AppSettingsVNext().intent.stt,
            custom_terms={"ko": ["PuriPuly"], "en": ["Avatar"]},
        ),
    )

    provider, general, prompt, overlay = settings_view_surface_snapshots(settings)

    assert provider.translation.model.value == settings.intent.translation.model
    assert provider.qwen_asr_model == settings.intent.stt.qwen_asr.model
    assert general.locale == settings.intent.ui.locale
    assert prompt.custom_vocabulary_terms == ("PuriPuly",)
    assert prompt.custom_vocabulary_other_languages_have_terms is True
    assert overlay.target == settings.intent.overlay.target
    with pytest.raises(FrozenInstanceError):
        general.locale = "ja"


def test_immediate_intents_rebase_onto_latest_settings_without_surface_displacement() -> None:
    displayed = _vnext(
        languages=replace(AppSettingsVNext().intent.languages, source_language="en"),
        stt=replace(
            AppSettingsVNext().intent.stt,
            custom_terms={"en": ["old"], "ja": ["既存"]},
        ),
    )
    current = _vnext(
        languages=replace(
            AppSettingsVNext().intent.languages,
            source_language="ja",
            target_language="ko",
        ),
        stt=replace(
            AppSettingsVNext().intent.stt,
            custom_terms={"en": ["old"], "ja": ["最新"]},
        ),
    )

    localized = materialize_immediate_settings_intent(current, LocaleSettingsIntent("ko"))
    updated = materialize_immediate_settings_intent(
        localized,
        CustomVocabularySettingsIntent("en", ("new",)),
    )

    assert updated.intent.ui.locale == "ko"
    assert updated.intent.languages.source_language == "ja"
    assert updated.intent.languages.target_language == "ko"
    assert updated.intent.stt.custom_terms == {"en": ["new"], "ja": ["最新"]}
    assert current.intent.ui.locale == "en"
    assert displayed.intent.stt.custom_terms["en"] == ["old"]


def test_custom_vocabulary_intent_derives_enabled_from_latest_rebased_terms() -> None:
    current = _vnext(
        stt=replace(
            AppSettingsVNext().intent.stt,
            custom_terms={"en": ["stale"], "ja": ["latest"]},
            custom_vocabulary_enabled=True,
        ),
    )

    updated = materialize_immediate_settings_intent(
        current,
        CustomVocabularySettingsIntent("en", ()),
    )

    assert updated.intent.stt.custom_terms == {"en": [], "ja": ["latest"]}
    assert updated.intent.stt.custom_vocabulary_enabled is True

    no_other_terms = _vnext(
        stt=replace(AppSettingsVNext().intent.stt, custom_terms={"en": ["stale"]}),
    )
    cleared = materialize_immediate_settings_intent(
        no_other_terms,
        CustomVocabularySettingsIntent("en", ()),
    )
    assert cleared.intent.stt.custom_vocabulary_enabled is False


def test_focused_immediate_intents_preserve_latest_sibling_values() -> None:
    baseline = AppSettingsVNext()
    current = _vnext(
        osc=replace(
            baseline.intent.osc,
            connection_mode="manual",
            send_port=9010,
            receive_port=9011,
            vrc_mic_intercept=False,
            chatbox_include_source=False,
        ),
        desktop_audio=replace(
            baseline.intent.desktop_audio,
            vad_speech_threshold=0.73,
            vad_hangover_ms=900,
            vad_pre_roll_ms=225,
            output_device="latest output",
        ),
        overlay=replace(
            baseline.intent.overlay,
            target="steamvr",
            show_translation=False,
            desktop_flet=replace(
                baseline.intent.overlay.desktop_flet,
                swap_caption_languages=True,
                visual=replace(
                    baseline.intent.overlay.desktop_flet.visual,
                    background_alpha=0.62,
                ),
            ),
        ),
    )

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

    assert updated.intent.osc.connection_mode == "manual"
    assert updated.intent.osc.send_port == 9010
    assert updated.intent.osc.receive_port == 9011
    assert updated.intent.osc.vrc_mic_intercept is True
    assert updated.intent.osc.chatbox_include_source is True
    assert updated.intent.desktop_audio.vad_speech_threshold == 0.73
    assert updated.intent.desktop_audio.vad_hangover_ms == 1200
    assert updated.intent.desktop_audio.vad_pre_roll_ms == 225
    assert updated.intent.overlay.target == "desktop"
    assert updated.intent.overlay.show_translation is False
    assert updated.intent.overlay.desktop_flet.visual.background_alpha == 0.4
    assert updated.intent.overlay.desktop_flet.swap_caption_languages is True
    assert updated.intent.audio.input_host_api == "MME"
    assert updated.intent.audio.input_device == "staged microphone"
    assert updated.intent.desktop_audio.output_device == "latest output"


def test_provider_edit_journal_replays_only_owned_fields_onto_latest_settings() -> None:
    displayed = _vnext(
        translation=replace(
            AppSettingsVNext().intent.translation,
            connection_history={
                TranslationModel.GEMMA4.value: TranslationConnection.MANAGED.value,
                TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.MANAGED_CHINA.value,
            },
        ),
    )
    provider, _general, _prompt, _overlay = settings_view_surface_snapshots(displayed)
    selection = replace(
        provider.translation,
        model=TranslationModel.GEMINI_37_FLASH,
        connection=TranslationConnection.OPENROUTER,
    )
    current = _vnext(
        languages=replace(AppSettingsVNext().intent.languages, source_language="ja"),
        audio=replace(AppSettingsVNext().intent.audio, input_device="latest microphone"),
        translation=replace(
            AppSettingsVNext().intent.translation,
            gpu_device_id="latest-llm-gpu",
            connection_history={
                TranslationModel.GEMMA4.value: TranslationConnection.OPENROUTER.value,
                TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.OFFICIAL_BYOK.value,
            },
        ),
        stt=replace(
            AppSettingsVNext().intent.stt,
            custom=replace(
                AppSettingsVNext().intent.stt.custom,
                model="latest-custom-model",
                extra={"latest": True},
            ),
        ),
    )

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
                QwenAsrModelEdit("qwen-audio-3.0-asr-flash-streaming"),
                SystemPromptEdit("focused prompt"),
            )
        ),
        materialize_translation=materialize_canonical_translation_settings,
    )

    translation = updated.intent.translation
    assert provider_llm_for_translation(translation.model, translation.connection) == "openrouter"
    assert translation.model == TranslationModel.GEMINI_37_FLASH.value
    assert translation.connection == TranslationConnection.OPENROUTER.value
    assert translation.connection_history[TranslationModel.GEMMA4.value] == (
        TranslationConnection.OPENROUTER.value
    )
    assert translation.connection_history[TranslationModel.DEEPSEEK_V4_FLASH.value] == (
        TranslationConnection.OFFICIAL_BYOK.value
    )
    assert translation.connection_history[TranslationModel.GEMINI_37_FLASH.value] == (
        TranslationConnection.OPENROUTER.value
    )
    assert updated.intent.stt.provider == STTProviderName.DEEPGRAM.value
    assert updated.intent.peer_stt.provider == current.intent.peer_stt.provider
    assert updated.intent.local_llm.base_url == "http://draft.local:11434"
    assert updated.intent.local_llm.model == current.intent.local_llm.model
    assert updated.intent.local_llm.extra_body == current.intent.local_llm.extra_body
    assert updated.intent.stt.gpu_device_id == "staged-stt-gpu"
    assert translation.gpu_device_id == "latest-llm-gpu"
    assert updated.intent.stt.custom.endpoint == "https://draft.invalid/v1/audio/transcriptions"
    assert updated.intent.stt.custom.model == "latest-custom-model"
    assert updated.intent.stt.custom.extra == {"latest": True}
    assert translation.qwen.region == QwenRegion.SINGAPORE.value
    assert updated.intent.stt.qwen_asr.model == "qwen-audio-3.0-asr-flash-streaming"
    assert updated.intent.prompts.system_prompt == "focused prompt"
    assert updated.intent.languages.source_language == "ja"
    assert updated.intent.audio.input_device == "latest microphone"


def test_prompt_intent_preserves_latest_languages_and_provider_selection() -> None:
    current = _vnext(
        languages=replace(
            AppSettingsVNext().intent.languages,
            source_language="ja",
            target_language="ko",
        ),
        translation=replace(AppSettingsVNext().intent.translation, model="qwen38_flash"),
    )

    updated = materialize_prompt_apply_intent(current, PromptApplyIntent("new prompt"))

    assert updated.intent.prompts.system_prompt == "new prompt"
    assert updated.intent.languages.source_language == "ja"
    assert updated.intent.languages.target_language == "ko"
    assert (
        provider_llm_for_translation(
            updated.intent.translation.model,
            updated.intent.translation.connection,
        )
        == "qwen"
    )


def test_managed_byok_pkce_target_carries_focused_translation_change() -> None:
    current = _vnext(
        translation=replace(
            AppSettingsVNext().intent.translation,
            connection="managed",
            openrouter_selected_source=OpenRouterCredentialSource.MANAGED.value,
            openrouter_selection_alias="gemma4_26b_31b_managed",
            openrouter_model="google/gemma-4-26b-a4b-it",
        ),
    )
    owner = SettingsOwner(
        path=Path("settings.json"),
        persistence=SettingsVNextCanonicalPersistenceAdapter(),
        canonical=current,
    )
    adapter = UiProviderRuntimeAdapter.__new__(UiProviderRuntimeAdapter)
    adapter.settings = owner
    adapter.build_byok_target_settings = owner.build_managed_openrouter_byok_target

    target = adapter.build_managed_openrouter_byok_target()

    assert target is not None
    updated = materialize_provider_apply_intent(
        current,
        target.provider_intent,
        materialize_translation=materialize_canonical_translation_settings,
    )
    assert updated.intent.translation.connection == TranslationConnection.OPENROUTER.value
    assert current.intent.translation.connection == "managed"


def test_rolling_enabled_edits_persist_self_and_peer_flags() -> None:
    current = AppSettingsVNext()
    assert current.intent.stt.rolling_enabled is False
    assert current.intent.peer_stt.rolling_enabled is False

    updated = materialize_provider_apply_intent(
        current,
        ProviderApplyIntent((SttRollingEnabledEdit(True),)),
        materialize_translation=materialize_canonical_translation_settings,
    )
    assert updated.intent.stt.rolling_enabled is True
    assert updated.intent.peer_stt.rolling_enabled is False

    peer_updated = materialize_provider_apply_intent(
        updated,
        ProviderApplyIntent((PeerSttRollingEnabledEdit(True),)),
        materialize_translation=materialize_canonical_translation_settings,
    )
    assert peer_updated.intent.stt.rolling_enabled is True
    assert peer_updated.intent.peer_stt.rolling_enabled is True

    provider_snapshot, _general, _prompt, _overlay = settings_view_surface_snapshots(peer_updated)
    assert provider_snapshot.stt_rolling_enabled is True
    assert provider_snapshot.peer_stt_rolling_enabled is True
