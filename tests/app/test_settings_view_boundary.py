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
from puripuly_heart.app.wiring_provider_runtime import (
    project_translation_runtime_settings_from_vnext,
)

from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.app.adapters.ui_runtime import UiProviderRuntimeAdapter
from puripuly_heart.app.ports.settings_view import (
    ActivationNoticeSettingsIntent,
    AudioInputSettingsIntent,
    AudioSettingsIntent,
    ChatboxSourceSettingsIntent,
    CloudFreeTierProvidersEdit,
    CustomSttEndpointEdit,
    CustomVocabularySettingsIntent,
    DesktopOverlayBackgroundAlphaIntent,
    LocaleSettingsIntent,
    LocalLlmBaseUrlEdit,
    OverlayTargetSettingsIntent,
    PeerVadHangoverIntent,
    PeerVadSpeechThresholdIntent,
    PromptApplyIntent,
    ProviderApplyIntent,
    QwenBeijingApiHostEdit,
    QwenRegionEdit,
    QwenSingaporeApiHostEdit,
    SelfSttProviderEdit,
    SelfVadSettingsIntent,
    SonioxSpeakerDiarizationEdit,
    SttGpuDeviceEdit,
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
from puripuly_heart.config.alibaba_connection import AlibabaRegionalSettings
from puripuly_heart.config.prompts import get_translation_prompt_template
from puripuly_heart.config.provider_values import (
    OpenRouterCredentialSource,
    QwenRegion,
    STTProviderName,
)
from puripuly_heart.config.runtime_resolution import OPENAI_MODEL_GPT_6_LUNA
from puripuly_heart.config.settings_vnext.schema import (
    AppSettingsVNext,
    ProviderVerificationEntry,
)
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
    assert general.locale == settings.intent.ui.locale
    assert prompt.custom_vocabulary_terms == ("PuriPuly",)
    assert prompt.custom_vocabulary_other_languages_have_terms is True
    assert overlay.target == settings.intent.overlay.target
    with pytest.raises(FrozenInstanceError):
        general.locale = "ja"


def test_activation_notice_intent_replays_only_its_canonical_preference() -> None:
    baseline = AppSettingsVNext()
    current = _vnext(
        osc=replace(
            baseline.intent.osc,
            connection_mode="manual",
            send_port=9130,
            receive_port=9131,
            chatbox_include_source=True,
            vrc_mic_intercept=True,
        ),
        overlay=replace(baseline.intent.overlay, show_translation=False),
        ui=replace(baseline.intent.ui, locale="ja"),
    )

    updated = materialize_immediate_settings_intent(current, ActivationNoticeSettingsIntent(False))

    assert updated == _vnext(
        current,
        osc=replace(current.intent.osc, activation_notice_enabled=False),
    )
    assert settings_view_surface_snapshots(current)[1].activation_notice_enabled is True
    assert settings_view_surface_snapshots(updated)[1].activation_notice_enabled is False


@pytest.mark.parametrize("invalid", [None, 0, 1, "false", [], {}])
def test_activation_notice_intent_rejects_non_boolean_values(invalid: object) -> None:
    with pytest.raises(ValueError, match="activation_notice_enabled"):
        materialize_immediate_settings_intent(
            AppSettingsVNext(),
            ActivationNoticeSettingsIntent(invalid),
        )


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


def test_vad_threshold_intents_enforce_shared_range_and_independent_values() -> None:
    current = AppSettingsVNext()

    self_updated = materialize_immediate_settings_intent(
        current,
        SelfVadSettingsIntent(0.10),
    )
    assert self_updated.intent.stt.vad_speech_threshold == 0.10
    assert (
        self_updated.intent.desktop_audio.vad_speech_threshold
        == current.intent.desktop_audio.vad_speech_threshold
    )

    peer_updated = materialize_immediate_settings_intent(
        self_updated,
        PeerVadSpeechThresholdIntent(0.75),
    )
    assert peer_updated.intent.stt.vad_speech_threshold == 0.10
    assert peer_updated.intent.desktop_audio.vad_speech_threshold == 0.75

    for intent in (
        SelfVadSettingsIntent(0.09),
        SelfVadSettingsIntent(1.01),
        PeerVadSpeechThresholdIntent(0.09),
        PeerVadSpeechThresholdIntent(1.01),
    ):
        with pytest.raises(ValueError, match="0.10..1.00"):
            materialize_immediate_settings_intent(current, intent)


def test_provider_edit_journal_replays_only_owned_fields_onto_latest_settings() -> None:
    displayed = _vnext(
        translation=replace(
            AppSettingsVNext().intent.translation,
            connection_history={
                TranslationModel.GEMMA4_26B_31B.value: TranslationConnection.MANAGED.value,
                TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.MANAGED_CHINA.value,
            },
        ),
    )
    provider, _general, _prompt, _overlay = settings_view_surface_snapshots(displayed)
    selection = replace(
        provider.translation,
        model=TranslationModel.GEMINI_FLASH,
        connection=TranslationConnection.OPENROUTER,
    )
    current = _vnext(
        languages=replace(AppSettingsVNext().intent.languages, source_language="ja"),
        audio=replace(AppSettingsVNext().intent.audio, input_device="latest microphone"),
        translation=replace(
            AppSettingsVNext().intent.translation,
            gpu_device_id="latest-llm-gpu",
            connection_history={
                TranslationModel.GEMMA4_26B_31B.value: TranslationConnection.OPENROUTER.value,
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
                    ((TranslationModel.GEMINI_FLASH, TranslationConnection.OPENROUTER),),
                ),
                SelfSttProviderEdit(STTProviderName.DEEPGRAM),
                SttGpuDeviceEdit("staged-stt-gpu"),
                LocalLlmBaseUrlEdit("http://draft.local:11434"),
                CustomSttEndpointEdit("https://draft.invalid/v1/audio/transcriptions"),
                QwenRegionEdit(QwenRegion.SINGAPORE),
                SystemPromptEdit("focused prompt"),
            )
        ),
        materialize_translation=materialize_canonical_translation_settings,
    )

    translation = updated.intent.translation
    assert provider_llm_for_translation(translation.model, translation.connection) == "openrouter"
    assert translation.model == TranslationModel.GEMINI_FLASH.value
    assert translation.connection == TranslationConnection.OPENROUTER.value
    assert translation.connection_history[TranslationModel.GEMMA4_26B_31B.value] == (
        TranslationConnection.OPENROUTER.value
    )
    assert translation.connection_history[TranslationModel.DEEPSEEK_V4_FLASH.value] == (
        TranslationConnection.OFFICIAL_BYOK.value
    )
    assert translation.connection_history[TranslationModel.GEMINI_FLASH.value] == (
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
    assert updated.intent.prompts.system_prompt_override == "focused prompt"
    assert updated.intent.languages.source_language == "ja"
    assert updated.intent.audio.input_device == "latest microphone"


@pytest.mark.parametrize(
    "connection",
    (
        TranslationConnection.OFFICIAL_BYOK,
        TranslationConnection.OPENROUTER,
        TranslationConnection.CHATGPT,
    ),
)
@pytest.mark.parametrize("override", (None, "  Custom translation prompt.\n"))
def test_model_selection_preserves_prompt_override_and_generic_editor_default(
    connection: TranslationConnection,
    override: str | None,
) -> None:
    baseline = AppSettingsVNext()
    current = _vnext(
        baseline,
        prompts=replace(baseline.intent.prompts, system_prompt_override=override),
    )
    for model, selected_connection in (
        (TranslationModel.GPT_6_LUNA, connection),
        (TranslationModel.GEMINI_FLASH, TranslationConnection.OFFICIAL_BYOK),
    ):
        provider, _general, _prompt, _overlay = settings_view_surface_snapshots(current)
        selection = replace(
            provider.translation,
            model=model,
            connection=selected_connection,
        )
        current = materialize_provider_apply_intent(
            current,
            ProviderApplyIntent(
                (TranslationSelectionEdit(selection, ((model, selected_connection),)),)
            ),
            materialize_translation=materialize_canonical_translation_settings,
        )

        assert current.intent.prompts.system_prompt_override == override
        editor_prompt = settings_view_surface_snapshots(current)[2].system_prompt
        assert editor_prompt == (
            override if override is not None else get_translation_prompt_template()
        )
        if model is TranslationModel.GPT_6_LUNA:
            expected_runtime_prompt = get_translation_prompt_template(model=OPENAI_MODEL_GPT_6_LUNA)
        else:
            expected_runtime_prompt = get_translation_prompt_template()
        runtime = project_translation_runtime_settings_from_vnext(current)
        assert runtime.system_prompt == (
            override if override is not None else expected_runtime_prompt
        )

        saved = materialize_prompt_apply_intent(current, PromptApplyIntent(editor_prompt))
        assert saved.intent.prompts.system_prompt_override == override


def _with_qwen(settings: AppSettingsVNext, **qwen_fields: object) -> AppSettingsVNext:
    translation = settings.intent.translation
    return _vnext(
        settings,
        translation=replace(translation, qwen=replace(translation.qwen, **qwen_fields)),
    )


def _apply_qwen_host_edits(current: AppSettingsVNext, *edits: object) -> AppSettingsVNext:
    return materialize_provider_apply_intent(
        current,
        ProviderApplyIntent(edits),
        materialize_translation=materialize_canonical_translation_settings,
    )


def test_qwen_api_host_edit_selects_dedicated_mode_and_preserves_other_region() -> None:
    beijing = AlibabaRegionalSettings(
        "workspace_dedicated", "work-1.cn-beijing.maas.aliyuncs.com", 4
    )
    current = _with_qwen(AppSettingsVNext(), beijing=beijing)
    current = replace(
        current,
        state=replace(
            current.state,
            provider_verification=replace(
                current.state.provider_verification,
                alibaba_singapore=ProviderVerificationEntry(
                    status="verified",
                    provider="alibaba_singapore",
                    secret_key="alibaba_api_key_singapore",
                    secret_fingerprint="sha256:previous",
                    verifier_context={"host": "dashscope-intl.aliyuncs.com"},
                ),
            ),
        ),
    )

    updated = _apply_qwen_host_edits(
        current,
        QwenRegionEdit(QwenRegion.SINGAPORE),
        QwenSingaporeApiHostEdit("https://Work-2.ap-southeast-1.maas.aliyuncs.com/api/v1"),
    )

    qwen = updated.intent.translation.qwen
    assert qwen.region == QwenRegion.SINGAPORE.value
    assert qwen.singapore.endpoint_mode == "workspace_dedicated"
    assert qwen.singapore.api_host == "work-2.ap-southeast-1.maas.aliyuncs.com"
    assert qwen.singapore.revision == current.intent.translation.qwen.singapore.revision + 1
    assert qwen.beijing == beijing
    assert updated.state.provider_verification.alibaba_singapore.status == "unknown"
    provider, _general, _prompt, _overlay = settings_view_surface_snapshots(updated)
    assert provider.qwen_api_host_singapore == "work-2.ap-southeast-1.maas.aliyuncs.com"
    assert provider.qwen_api_host_beijing == "work-1.cn-beijing.maas.aliyuncs.com"


def test_clearing_qwen_api_host_returns_region_to_shared_mode() -> None:
    current = _with_qwen(
        AppSettingsVNext(),
        beijing=AlibabaRegionalSettings(
            "workspace_dedicated", "work-1.cn-beijing.maas.aliyuncs.com", 4
        ),
    )

    updated = _apply_qwen_host_edits(current, QwenBeijingApiHostEdit(""))

    beijing = updated.intent.translation.qwen.beijing
    assert (beijing.endpoint_mode, beijing.api_host, beijing.revision) == ("legacy_shared", "", 5)
    provider, _general, _prompt, _overlay = settings_view_surface_snapshots(updated)
    assert provider.qwen_api_host_beijing == ""


def test_unchanged_qwen_api_host_keeps_connection_revision_and_verification() -> None:
    dedicated = AlibabaRegionalSettings(
        "workspace_dedicated", "work-1.cn-beijing.maas.aliyuncs.com", 4
    )
    current = _with_qwen(
        AppSettingsVNext(),
        beijing=dedicated,
        singapore=AlibabaRegionalSettings(
            "legacy_shared", "work-2.ap-southeast-1.maas.aliyuncs.com"
        ),
    )

    updated = _apply_qwen_host_edits(
        current,
        QwenBeijingApiHostEdit("work-1.cn-beijing.maas.aliyuncs.com"),
        QwenSingaporeApiHostEdit(""),
    )

    assert updated == current
    provider, _general, _prompt, _overlay = settings_view_surface_snapshots(current)
    assert provider.qwen_api_host_singapore == ""


def test_qwen_api_host_edit_rejects_host_from_another_region() -> None:
    with pytest.raises(ValueError) as failure:
        _apply_qwen_host_edits(
            AppSettingsVNext(),
            QwenBeijingApiHostEdit("work-2.ap-southeast-1.maas.aliyuncs.com"),
        )

    assert "work-2" not in str(failure.value)


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

    assert updated.intent.prompts.system_prompt_override == "new prompt"
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


def test_rolling_free_provider_selection_persists_self_and_peer() -> None:
    from puripuly_heart.app.ports.settings_view import PeerSttProviderEdit
    from puripuly_heart.config.provider_values import STTProviderName

    current = AppSettingsVNext()
    assert current.intent.stt.provider == "local_cpu_auto"
    assert current.intent.peer_stt.provider == "local_cpu_auto"

    updated = materialize_provider_apply_intent(
        current,
        ProviderApplyIntent((SelfSttProviderEdit(STTProviderName.ROLLING_FREE),)),
        materialize_translation=materialize_canonical_translation_settings,
    )
    assert updated.intent.stt.provider == STTProviderName.ROLLING_FREE.value
    assert updated.intent.peer_stt.provider == "local_cpu_auto"

    peer_updated = materialize_provider_apply_intent(
        updated,
        ProviderApplyIntent((PeerSttProviderEdit(STTProviderName.ROLLING_FREE),)),
        materialize_translation=materialize_canonical_translation_settings,
    )
    assert peer_updated.intent.stt.provider == STTProviderName.ROLLING_FREE.value
    assert peer_updated.intent.peer_stt.provider == STTProviderName.ROLLING_FREE.value


def test_cloud_free_tier_provider_selection_persists() -> None:
    current = AppSettingsVNext()
    assert current.intent.stt.cloud_free_tier_providers == ["gemini_transcribe"]

    updated = materialize_provider_apply_intent(
        current,
        ProviderApplyIntent(
            (
                CloudFreeTierProvidersEdit(
                    (
                        STTProviderName.GEMINI_TRANSCRIBE,
                        STTProviderName.DEEPGRAM,
                    )
                ),
            )
        ),
        materialize_translation=materialize_canonical_translation_settings,
    )
    assert updated.intent.stt.cloud_free_tier_providers == [
        "gemini_transcribe",
        "deepgram",
    ]


def test_soniox_speaker_diarization_preference_persists_independently_of_provider() -> None:
    from puripuly_heart.composition.application_runtime import (
        _copy_provider_prompt_apply_fields,
    )

    current = AppSettingsVNext()
    updated = materialize_provider_apply_intent(
        current,
        ProviderApplyIntent((SonioxSpeakerDiarizationEdit(False),)),
        materialize_translation=materialize_canonical_translation_settings,
    )
    merged = _copy_provider_prompt_apply_fields(updated, current)

    assert updated.intent.stt.provider == current.intent.stt.provider
    assert updated.intent.stt.soniox.enable_speaker_diarization is False
    assert merged.intent.stt.soniox.enable_speaker_diarization is False


def test_provider_apply_merge_keeps_cloud_free_tier_pool() -> None:
    from puripuly_heart.composition.application_runtime import (
        _copy_provider_prompt_apply_fields,
    )

    baseline = AppSettingsVNext()
    current = replace(
        baseline,
        intent=replace(
            baseline.intent,
            stt=replace(baseline.intent.stt, vad_speech_threshold=0.25),
        ),
    )
    pending = materialize_provider_apply_intent(
        current,
        ProviderApplyIntent(
            (
                CloudFreeTierProvidersEdit(
                    (
                        STTProviderName.GEMINI_TRANSCRIBE,
                        STTProviderName.DEEPGRAM,
                    )
                ),
            )
        ),
        materialize_translation=materialize_canonical_translation_settings,
    )

    merged = _copy_provider_prompt_apply_fields(pending, current)

    assert merged.intent.stt.cloud_free_tier_providers == [
        "gemini_transcribe",
        "deepgram",
    ]
    assert merged.intent.stt.vad_speech_threshold == 0.25


def test_legacy_deepseek_openrouter_model_still_builds_provider_snapshot() -> None:
    from puripuly_heart.config.provider_values import OpenRouterLLMModel

    baseline = AppSettingsVNext()
    legacy = replace(
        baseline,
        intent=replace(
            baseline.intent,
            translation=replace(
                baseline.intent.translation,
                openrouter_model="deepseek/deepseek-v4-flash",
                openrouter_selected_source="byok",
                openrouter_selection_alias="deepseek_v4_flash_byok",
            ),
        ),
    )

    provider, _general, _prompt, _overlay = settings_view_surface_snapshots(legacy)

    assert provider.openrouter_llm_model == OpenRouterLLMModel.DEEPSEEK_V4_FLASH


def test_legacy_deepseek_openrouter_model_builds_release_runtime_config() -> None:
    from puripuly_heart.app.wiring.wiring_managed_auth_factory import (
        build_openrouter_release_runtime_config_from_vnext,
    )
    from puripuly_heart.config.provider_values import OpenRouterLLMModel

    baseline = AppSettingsVNext()
    legacy = replace(
        baseline,
        intent=replace(
            baseline.intent,
            translation=replace(
                baseline.intent.translation,
                openrouter_model="deepseek/deepseek-v4-flash",
                openrouter_selected_source="byok",
                openrouter_selection_alias="deepseek_v4_flash_byok",
            ),
        ),
    )

    config = build_openrouter_release_runtime_config_from_vnext(legacy)

    assert config.llm_model == OpenRouterLLMModel.DEEPSEEK_V4_FLASH
