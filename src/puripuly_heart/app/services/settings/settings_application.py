from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace

from puripuly_heart.app.language_selection import LanguageSelectionChange
from puripuly_heart.app.ports.runtime_apply import RuntimeApplyPort
from puripuly_heart.app.ports.settings_runtime_effects import SettingsRuntimeEffectsPort
from puripuly_heart.app.ports.settings_view import (
    AudioInputSettingsIntent,
    AudioSettingsIntent,
    ChatboxSourceSettingsIntent,
    ClipboardSettingsIntent,
    CustomSttEndpointEdit,
    CustomSttExtraEdit,
    CustomSttModelEdit,
    CustomVocabularySettingsIntent,
    DesktopAudioOutputSettingsIntent,
    DesktopOverlayBackgroundAlphaIntent,
    DesktopOverlayPositionResetIntent,
    DesktopOverlaySizeIntent,
    DesktopOverlaySwapCaptionLanguagesIntent,
    GeneralSettingsSnapshot,
    ImmediateSettingsIntent,
    LlmGpuDeviceEdit,
    LocaleSettingsIntent,
    LocalLlmBaseUrlEdit,
    LocalLlmExtraBodyEdit,
    LocalLlmModelEdit,
    ManagedReferralEdit,
    OscConnectionSettingsIntent,
    OverlayCalibrationSettingsIntent,
    OverlayCalibrationSnapshot,
    OverlayPeerOriginalSettingsIntent,
    OverlaySettingsSnapshot,
    OverlayTargetSettingsIntent,
    OverlayTranslationSettingsIntent,
    PeerExpectedLanguagesIntent,
    PeerSttProviderEdit,
    PeerVadHangoverIntent,
    PeerVadPreRollIntent,
    PeerVadSpeechThresholdIntent,
    PromptApplyIntent,
    PromptSettingsSnapshot,
    ProviderApplyIntent,
    ProviderSettingsSnapshot,
    ProviderVerificationSnapshot,
    QwenRegionEdit,
    SelfSttProviderEdit,
    SelfVadSettingsIntent,
    SttGpuDeviceEdit,
    SystemPromptEdit,
    TranslationFallbackEdit,
    TranslationFallbackSnapshot,
    TranslationHttpExtensionEdit,
    TranslationSelectionEdit,
    TranslationSelectionSnapshot,
    VrcMicInterceptSettingsIntent,
)
from puripuly_heart.app.ports.ui_models import (
    OscControlPresentationName,
    OscControlPresentationState,
)
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.services.manual_local_asr_fallback import (
    ManualLocalASRFallbackOwner,
    ManualLocalASRFallbackPlan,
)
from puripuly_heart.app.services.osc.state_publisher import OscCanonicalState
from puripuly_heart.app.services.provider_runtime_apply import (
    NoopRuntimeApply,
    OverlayOscOutputRuntimeApplyAdapter,
    SttLanguageAudioRuntimeApplyAdapter,
    UiPromptClipboardStateRuntimeApplyAdapter,
    _overlay_osc_output_runtime_degraded_transaction_result,
    _overlay_osc_output_save_failed_transaction_result,
    _runtime_apply_result_as_degraded_transaction,
    _stt_language_audio_runtime_degraded_transaction_result,
    _stt_language_audio_runtime_unavailable_result,
    _stt_language_audio_save_failed_transaction_result,
    _ui_prompt_clipboard_state_runtime_degraded_transaction_result,
    _ui_prompt_clipboard_state_save_failed_transaction_result,
)
from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.config.provider_values import (
    LLMProviderName,
    OpenRouterCredentialSource,
    OpenRouterLLMModel,
    OpenRouterSelectionAlias,
    QwenRegion,
    STTProviderName,
)
from puripuly_heart.config.settings_vnext.schema import (
    AppSettingsVNext,
    DesktopFletOverlayPositionIntent,
    TranslationFallbackIntent,
    _infer_translation_fallback_alias,
)
from puripuly_heart.config.translation_values import (
    TranslationConnection,
    TranslationModel,
    provider_llm_for_translation,
)
from puripuly_heart.core.messages import (
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    TransactionResult,
)

from .settings_mutation import (
    OverlayOscOutputSettingsMutation,
    SettingsMutationCommand,
    SettingsMutationService,
    SttLanguageAudioSettingsMutation,
    UiPromptClipboardStateSettingsMutation,
)
from .settings_mutation_legacy import (
    apply_settings_path_patch,
    build_overlay_osc_output_settings_path_patch,
    settings_path_mutation_validator_for_command,
)
from .settings_projection import SettingsProjectionOwner
from .settings_transaction_result import (
    SettingsTransactionResultOwner,
)


class StrictSettingsSaveFailed(Exception):
    pass


SettingsMutationServiceProvider = Callable[[], SettingsMutationService | None]
SettingsPredicate = Callable[[AppSettingsVNext, AppSettingsVNext], bool]
SettingsFailureSink = Callable[[str], None]
SettingsAsyncEffect = Callable[[], Awaitable[None]]
SettingsFallbackSink = Callable[[tuple[str, ...], bool], None]
SettingsFallbackLogSink = Callable[
    [object, object, tuple[str, ...]],
    None,
]


def _settings_mutation_committed(result: TransactionResult) -> bool:
    return result.status in {
        TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
        TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    }


def _copy_runtime_only_ui_state(source: object, target: object) -> object:
    return target


def _active_prompt_key(settings: AppSettingsVNext) -> str:
    provider = provider_llm_for_translation(
        settings.intent.translation.model,
        settings.intent.translation.connection,
    )
    if provider in {"gemini", "openrouter", "deepseek", "local_llm", "managed_gemma"}:
        return provider
    return "qwen"


def _translation_model(value: object) -> TranslationModel:
    return value if isinstance(value, TranslationModel) else TranslationModel(str(value))


def _translation_connection(value: object) -> TranslationConnection:
    return value if isinstance(value, TranslationConnection) else TranslationConnection(str(value))


def _optional_translation_model(value: object) -> TranslationModel | None:
    if value is None or value == "":
        return None
    return _translation_model(value)


def _openrouter_selection_alias(value: object) -> OpenRouterSelectionAlias | None:
    if value is None or value == "":
        return None
    return (
        value
        if isinstance(value, OpenRouterSelectionAlias)
        else OpenRouterSelectionAlias(str(value))
    )


def _verified(entry: object) -> bool:
    return getattr(entry, "status", None) == "verified"


def settings_view_surface_snapshots(
    settings: object,
) -> tuple[
    ProviderSettingsSnapshot,
    GeneralSettingsSnapshot,
    PromptSettingsSnapshot,
    OverlaySettingsSnapshot,
]:
    if not isinstance(settings, AppSettingsVNext):
        raise TypeError("settings snapshots require AppSettingsVNext")
    intent = settings.intent
    translation = intent.translation
    verification = settings.state.provider_verification
    provider = ProviderSettingsSnapshot(
        stt_provider=STTProviderName(intent.stt.provider),
        peer_stt_provider=STTProviderName(intent.peer_stt.provider),
        llm_provider=LLMProviderName(
            provider_llm_for_translation(translation.model, translation.connection)
        ),
        translation=TranslationSelectionSnapshot(
            model=_translation_model(translation.model),
            connection=_translation_connection(translation.connection),
            connection_history=tuple(
                (_translation_model(model), _translation_connection(connection))
                for model, connection in translation.connection_history.items()
            ),
            fallback=TranslationFallbackSnapshot(
                enabled=translation.fallback.enabled,
                model=_translation_model(translation.fallback.model),
                connection=_translation_connection(translation.fallback.connection),
            ),
            http_extension_id=translation.http_extension_id,
            previous_llm_model=_optional_translation_model(translation.previous_llm_model),
            gpu_device_id=translation.gpu_device_id,
        ),
        stt_gpu_device_id=intent.stt.gpu_device_id,
        qwen_region=QwenRegion(intent.translation.qwen.region),
        local_llm_base_url=intent.local_llm.base_url,
        local_llm_model=intent.local_llm.model,
        local_llm_extra_body_json=json.dumps(
            intent.local_llm.extra_body,
            ensure_ascii=False,
            indent=2,
        ),
        custom_stt_mode=intent.stt.custom.mode,
        custom_stt_compatibility=intent.stt.custom.compatibility,
        custom_stt_endpoint=intent.stt.custom.endpoint,
        custom_stt_model=intent.stt.custom.model,
        custom_stt_extra_json=json.dumps(
            intent.stt.custom.extra,
            ensure_ascii=False,
            indent=2,
        ),
        openrouter_llm_model=OpenRouterLLMModel(translation.openrouter_model),
        openrouter_selected_source=OpenRouterCredentialSource(
            translation.openrouter_selected_source
        ),
        openrouter_selection_alias=_openrouter_selection_alias(
            translation.openrouter_selection_alias
        ),
        verified=ProviderVerificationSnapshot(
            deepgram=_verified(verification.deepgram),
            soniox=_verified(verification.soniox),
            google=_verified(verification.google),
            openrouter=_verified(verification.openrouter),
            deepseek=_verified(verification.deepseek),
            alibaba_beijing=_verified(verification.alibaba_beijing),
            alibaba_singapore=_verified(verification.alibaba_singapore),
            cerebras=_verified(verification.cerebras),
        ),
        managed_referral_id=settings.state.managed_connection.referral_id,
    )
    general = GeneralSettingsSnapshot(
        locale=intent.ui.locale,
        effective_peer_source_language=intent.languages.effective_peer_source,
        input_host_api=intent.audio.input_host_api,
        input_device=intent.audio.input_device,
        output_device=intent.desktop_audio.output_device,
        self_vad_speech_threshold=intent.stt.vad_speech_threshold,
        peer_vad_speech_threshold=intent.desktop_audio.vad_speech_threshold,
        peer_vad_hangover_ms=intent.desktop_audio.vad_hangover_ms,
        peer_vad_pre_roll_ms=intent.desktop_audio.vad_pre_roll_ms,
        osc_connection_mode=intent.osc.connection_mode,
        osc_port=intent.osc.port,
        osc_send_port=intent.osc.send_port,
        osc_receive_port=intent.osc.receive_port,
        vrc_mic_intercept=intent.osc.vrc_mic_intercept,
        chatbox_include_source=intent.osc.chatbox_include_source,
        clipboard_auto_translate_enabled=intent.clipboard.auto_translate_enabled,
        telemetry_enabled=intent.telemetry.enabled,
        peer_expected_languages=tuple(intent.languages.peer_expected_languages),
    )
    source_language = intent.languages.source_language
    prompt = PromptSettingsSnapshot(
        active_provider_key=_active_prompt_key(settings),
        source_language=source_language,
        system_prompt=intent.prompts.system_prompt,
        custom_vocabulary_enabled=intent.stt.custom_vocabulary_enabled,
        custom_vocabulary_terms=tuple(intent.stt.custom_terms.get(source_language, ())),
        custom_vocabulary_other_languages_have_terms=any(
            bool(terms)
            for language, terms in intent.stt.custom_terms.items()
            if language != source_language
        ),
    )
    calibration = intent.overlay.calibration
    overlay = OverlaySettingsSnapshot(
        target=intent.overlay.target,
        show_translation=intent.overlay.show_translation,
        show_peer_original=intent.overlay.show_peer_original,
        desktop_size_preset=intent.overlay.desktop_flet.size_preset,
        desktop_background_alpha=intent.overlay.desktop_flet.visual.background_alpha,
        desktop_swap_caption_languages=intent.overlay.desktop_flet.swap_caption_languages,
        calibration=OverlayCalibrationSnapshot(
            anchor=calibration.anchor,
            distance=calibration.distance,
            offset_x=calibration.offset_x,
            offset_y=calibration.offset_y,
            text_scale=calibration.text_scale,
        ),
    )
    return provider, general, prompt, overlay


def osc_control_presentation_state(
    settings: object,
    *,
    canonical_state: OscCanonicalState,
    changed_control: OscControlPresentationName,
    self_capture_effective: bool | None = None,
) -> OscControlPresentationState:
    if not isinstance(settings, AppSettingsVNext):
        raise TypeError("OSC presentation requires AppSettingsVNext")
    intent = settings.intent
    translation = intent.translation
    fallback = translation.fallback
    source_language = intent.languages.source_language
    return OscControlPresentationState(
        changed_control=changed_control,
        self_capture=(
            canonical_state.self_capture
            if self_capture_effective is None
            else bool(self_capture_effective)
        ),
        peer_capture=canonical_state.peer_capture,
        translation=canonical_state.translation,
        captions=canonical_state.captions,
        peer_source_mode=intent.languages.peer_source_mode,
        mute_sync=canonical_state.mute_sync,
        chatbox_source=canonical_state.chatbox_source,
        self_source_language=canonical_state.self_source_language,
        self_target_language=canonical_state.self_target_language,
        self_secondary_target_language=canonical_state.self_secondary_target_language,
        peer_source_language=canonical_state.peer_source_language,
        peer_target_language=canonical_state.peer_target_language,
        self_asr=canonical_state.self_asr,
        peer_asr=canonical_state.peer_asr,
        self_asr_setting=intent.stt.provider,
        peer_asr_setting=intent.peer_stt.provider,
        custom_stt_mode=intent.stt.custom.mode,
        custom_stt_compatibility=intent.stt.custom.compatibility,
        custom_vocabulary_enabled=bool(intent.stt.custom_vocabulary_enabled),
        custom_vocabulary_terms=tuple(intent.stt.custom_terms.get(source_language, ())),
        custom_vocabulary_other_languages_have_terms=any(
            bool(terms)
            for language, terms in intent.stt.custom_terms.items()
            if language != source_language
        ),
        llm_provider=provider_llm_for_translation(translation.model, translation.connection),
        openrouter_llm_model=translation.openrouter_model,
        openrouter_selected_source=translation.openrouter_selected_source,
        openrouter_selection_alias=translation.openrouter_selection_alias,
        translation_model=translation.model,
        translation_connection=translation.connection,
        translation_connection_history=tuple(
            sorted(
                (str(model), str(connection))
                for model, connection in translation.connection_history.items()
            )
        ),
        translation_http_extension_id=translation.http_extension_id,
        translation_previous_model=translation.previous_llm_model,
        fallback=canonical_state.fallback,
        fallback_enabled=bool(fallback.enabled),
        fallback_model=fallback.model,
        fallback_connection=fallback.connection,
    )


def _with_intent(settings: AppSettingsVNext, **changes: object) -> AppSettingsVNext:
    return replace(settings, intent=replace(settings.intent, **changes))


def materialize_immediate_settings_intent(
    current: object,
    intent: ImmediateSettingsIntent,
) -> object:
    if not isinstance(current, AppSettingsVNext):
        raise TypeError("immediate settings intents require AppSettingsVNext")
    updated = current
    if isinstance(intent, LocaleSettingsIntent):
        updated = _with_intent(updated, ui=replace(updated.intent.ui, locale=intent.locale))
    elif isinstance(intent, AudioSettingsIntent):
        for change in intent.changes:
            if isinstance(change, AudioInputSettingsIntent):
                updated = _with_intent(
                    updated,
                    audio=replace(
                        updated.intent.audio,
                        input_host_api=change.input_host_api,
                        input_device=change.input_device,
                    ),
                )
            elif isinstance(change, DesktopAudioOutputSettingsIntent):
                updated = _with_intent(
                    updated,
                    desktop_audio=replace(
                        updated.intent.desktop_audio,
                        output_device=change.output_device,
                    ),
                )
    elif isinstance(intent, SelfVadSettingsIntent):
        updated = _with_intent(
            updated,
            stt=replace(updated.intent.stt, vad_speech_threshold=intent.speech_threshold),
        )
    elif isinstance(intent, PeerVadSpeechThresholdIntent):
        updated = _with_intent(
            updated,
            desktop_audio=replace(
                updated.intent.desktop_audio,
                vad_speech_threshold=intent.speech_threshold,
            ),
        )
    elif isinstance(intent, PeerVadHangoverIntent):
        updated = _with_intent(
            updated,
            desktop_audio=replace(
                updated.intent.desktop_audio,
                vad_hangover_ms=intent.hangover_ms,
            ),
        )
    elif isinstance(intent, PeerVadPreRollIntent):
        updated = _with_intent(
            updated,
            desktop_audio=replace(
                updated.intent.desktop_audio,
                vad_pre_roll_ms=intent.pre_roll_ms,
            ),
        )
    elif isinstance(intent, OscConnectionSettingsIntent):
        updated = _with_intent(
            updated,
            osc=replace(
                updated.intent.osc,
                connection_mode=intent.connection_mode,
                send_port=intent.send_port,
                receive_port=intent.receive_port,
            ),
        )
    elif isinstance(intent, VrcMicInterceptSettingsIntent):
        updated = _with_intent(
            updated,
            osc=replace(updated.intent.osc, vrc_mic_intercept=intent.enabled),
        )
    elif isinstance(intent, ChatboxSourceSettingsIntent):
        updated = _with_intent(
            updated,
            osc=replace(updated.intent.osc, chatbox_include_source=intent.enabled),
        )
    elif isinstance(intent, ClipboardSettingsIntent):
        updated = _with_intent(
            updated,
            clipboard=replace(updated.intent.clipboard, auto_translate_enabled=intent.enabled),
        )
    elif isinstance(intent, PeerExpectedLanguagesIntent):
        updated = _with_intent(
            updated,
            languages=replace(
                updated.intent.languages,
                peer_expected_languages=list(intent.languages),
            ),
        )
    elif isinstance(intent, CustomVocabularySettingsIntent):
        custom_terms = dict(updated.intent.stt.custom_terms)
        custom_terms[intent.source_language] = list(intent.terms)
        updated = _with_intent(
            updated,
            stt=replace(
                updated.intent.stt,
                custom_terms=custom_terms,
                custom_vocabulary_enabled=any(custom_terms.values()),
            ),
        )
    elif isinstance(intent, OverlayTargetSettingsIntent):
        updated = _with_intent(
            updated,
            overlay=replace(updated.intent.overlay, target=intent.target),
        )
    elif isinstance(intent, OverlayTranslationSettingsIntent):
        updated = _with_intent(
            updated,
            overlay=replace(updated.intent.overlay, show_translation=intent.enabled),
        )
    elif isinstance(intent, OverlayPeerOriginalSettingsIntent):
        updated = _with_intent(
            updated,
            overlay=replace(updated.intent.overlay, show_peer_original=intent.enabled),
        )
    elif isinstance(intent, DesktopOverlayBackgroundAlphaIntent):
        desktop = updated.intent.overlay.desktop_flet
        updated = _with_intent(
            updated,
            overlay=replace(
                updated.intent.overlay,
                desktop_flet=replace(
                    desktop,
                    visual=replace(desktop.visual, background_alpha=intent.background_alpha),
                ),
            ),
        )
    elif isinstance(intent, DesktopOverlaySwapCaptionLanguagesIntent):
        updated = _with_intent(
            updated,
            overlay=replace(
                updated.intent.overlay,
                desktop_flet=replace(
                    updated.intent.overlay.desktop_flet,
                    swap_caption_languages=intent.enabled,
                ),
            ),
        )
    elif isinstance(intent, DesktopOverlaySizeIntent):
        updated = _with_intent(
            updated,
            overlay=replace(
                updated.intent.overlay,
                desktop_flet=replace(
                    updated.intent.overlay.desktop_flet,
                    size_preset=intent.size_preset,
                ),
            ),
        )
    elif isinstance(intent, DesktopOverlayPositionResetIntent):
        updated = _with_intent(
            updated,
            overlay=replace(
                updated.intent.overlay,
                desktop_flet=replace(
                    updated.intent.overlay.desktop_flet,
                    position=DesktopFletOverlayPositionIntent(x=None, y=None),
                ),
            ),
        )
    elif isinstance(intent, OverlayCalibrationSettingsIntent):
        calibration = intent.calibration
        updated = _with_intent(
            updated,
            overlay=replace(
                updated.intent.overlay,
                calibration=OverlayCalibration(
                    anchor=calibration.anchor,
                    distance=calibration.distance,
                    offset_x=calibration.offset_x,
                    offset_y=calibration.offset_y,
                    text_scale=calibration.text_scale,
                    background_alpha=updated.intent.overlay.calibration.background_alpha,
                ),
            ),
        )
    return updated


def materialize_prompt_apply_intent(
    current: object,
    intent: PromptApplyIntent,
) -> object:
    if not isinstance(current, AppSettingsVNext):
        raise TypeError("prompt apply intents require AppSettingsVNext")
    return _with_intent(
        current,
        prompts=replace(current.intent.prompts, system_prompt=intent.value),
    )


def materialize_provider_apply_intent(
    current: object,
    intent: ProviderApplyIntent,
    *,
    materialize_translation: Callable[[object], object],
) -> object:
    if not isinstance(current, AppSettingsVNext):
        raise TypeError("provider apply intents require AppSettingsVNext")
    updated = current
    for edit in intent.edits:
        if isinstance(edit, SelfSttProviderEdit):
            updated = _with_intent(
                updated,
                stt=replace(
                    updated.intent.stt,
                    provider=str(getattr(edit.provider, "value", edit.provider)),
                ),
            )
        elif isinstance(edit, PeerSttProviderEdit):
            updated = _with_intent(
                updated,
                peer_stt=replace(
                    updated.intent.peer_stt,
                    provider=str(getattr(edit.provider, "value", edit.provider)),
                ),
            )
        elif isinstance(edit, SttGpuDeviceEdit):
            updated = _with_intent(
                updated,
                stt=replace(updated.intent.stt, gpu_device_id=edit.device_id),
            )
        elif isinstance(edit, LlmGpuDeviceEdit):
            updated = _with_intent(
                updated,
                translation=replace(updated.intent.translation, gpu_device_id=edit.device_id),
            )
        elif isinstance(edit, TranslationSelectionEdit):
            selection = edit.selection
            history = dict(updated.intent.translation.connection_history)
            for model, connection in edit.history_updates:
                history[str(getattr(model, "value", model))] = str(
                    getattr(connection, "value", connection)
                )
            previous = selection.previous_llm_model
            updated = _with_intent(
                updated,
                translation=replace(
                    updated.intent.translation,
                    model=str(getattr(selection.model, "value", selection.model)),
                    connection=str(getattr(selection.connection, "value", selection.connection)),
                    connection_history=history,
                    previous_llm_model=(
                        None if previous is None else str(getattr(previous, "value", previous))
                    ),
                ),
            )
            materialized = materialize_translation(updated)
            if isinstance(materialized, AppSettingsVNext):
                updated = materialized
        elif isinstance(edit, TranslationFallbackEdit):
            fallback_model = str(getattr(edit.fallback.model, "value", edit.fallback.model))
            fallback_connection = str(
                getattr(edit.fallback.connection, "value", edit.fallback.connection)
            )
            updated = _with_intent(
                updated,
                translation=replace(
                    updated.intent.translation,
                    fallback=TranslationFallbackIntent(
                        selection_alias=_infer_translation_fallback_alias(
                            enabled=edit.fallback.enabled,
                            model=fallback_model,
                            connection=fallback_connection,
                        )
                    ),
                ),
            )
        elif isinstance(edit, TranslationHttpExtensionEdit):
            updated = _with_intent(
                updated,
                translation=replace(
                    updated.intent.translation,
                    http_extension_id=edit.extension_id,
                ),
            )
        elif isinstance(edit, QwenRegionEdit):
            updated = _with_intent(
                updated,
                translation=replace(
                    updated.intent.translation,
                    qwen=replace(
                        updated.intent.translation.qwen,
                        region=str(getattr(edit.region, "value", edit.region)),
                    ),
                ),
            )
        elif isinstance(edit, LocalLlmBaseUrlEdit):
            updated = _with_intent(
                updated,
                local_llm=replace(updated.intent.local_llm, base_url=edit.base_url),
            )
        elif isinstance(edit, LocalLlmModelEdit):
            updated = _with_intent(
                updated,
                local_llm=replace(updated.intent.local_llm, model=edit.model),
            )
        elif isinstance(edit, LocalLlmExtraBodyEdit):
            updated = _with_intent(
                updated,
                local_llm=replace(
                    updated.intent.local_llm,
                    extra_body=json.loads(edit.extra_body_json),
                ),
            )
        elif isinstance(edit, CustomSttEndpointEdit):
            updated = _with_intent(
                updated,
                stt=replace(
                    updated.intent.stt,
                    custom=replace(updated.intent.stt.custom, endpoint=edit.endpoint),
                ),
            )
        elif isinstance(edit, CustomSttModelEdit):
            updated = _with_intent(
                updated,
                stt=replace(
                    updated.intent.stt,
                    custom=replace(updated.intent.stt.custom, model=edit.model),
                ),
            )
        elif isinstance(edit, CustomSttExtraEdit):
            updated = _with_intent(
                updated,
                stt=replace(
                    updated.intent.stt,
                    custom=replace(
                        updated.intent.stt.custom,
                        extra=json.loads(edit.extra_json),
                    ),
                ),
            )
        elif isinstance(edit, ManagedReferralEdit):
            updated = replace(
                updated,
                state=replace(
                    updated.state,
                    managed_connection=replace(
                        updated.state.managed_connection,
                        referral_id=edit.referral_id,
                    ),
                ),
            )
        elif isinstance(edit, SystemPromptEdit):
            updated = _with_intent(
                updated,
                prompts=replace(updated.intent.prompts, system_prompt=edit.value),
            )
    return updated


@dataclass(slots=True)
class SettingsApplicationOwner:
    settings: SettingsOwner
    projection: SettingsProjectionOwner
    runtime_effects: SettingsRuntimeEffectsPort[AppSettingsVNext]
    manual_fallback: ManualLocalASRFallbackOwner
    cpu_auto_available: Callable[[], bool]
    inspect_cpu: SettingsAsyncEffect
    fallback_sink: SettingsFallbackSink
    sync_ui: Callable[[], None]
    fallback_log_sink: SettingsFallbackLogSink
    mutation_service_provider: SettingsMutationServiceProvider
    consume_superseded_settings: Callable[[AppSettingsVNext], bool]
    active_local_asr_change: SettingsPredicate
    failure_sink: SettingsFailureSink
    results: SettingsTransactionResultOwner = field(default_factory=SettingsTransactionResultOwner)

    async def apply(
        self,
        next_settings: AppSettingsVNext,
        *,
        reload_settings_view: bool = True,
    ) -> bool:
        current = self.settings.canonical
        if current is not None:
            self.settings.normalize_compatibility(current)
        self.settings.normalize_compatibility(next_settings)
        fallback_channels: tuple[str, ...] = ()
        installation_fallback = False
        normalization_channels = self.manual_fallback.normalization_channels(
            current=(
                self.manual_fallback.state(
                    current,
                    cpu_auto_available=self.cpu_auto_available(),
                )
                if current is not None
                else None
            ),
            pending=self.manual_fallback.state(
                next_settings,
                cpu_auto_available=self.cpu_auto_available(),
            ),
        )
        if normalization_channels:
            await self.inspect_cpu()
            fallback_plan = self.manual_fallback.plan(
                self.manual_fallback.state(
                    next_settings,
                    cpu_auto_available=self.cpu_auto_available(),
                )
            )
            fallback_channels = tuple(
                channel
                for channel in fallback_plan.fallback_channels
                if channel in normalization_channels
            )
            if fallback_channels:
                next_settings = self.manual_fallback.apply(
                    next_settings,
                    ManualLocalASRFallbackPlan(
                        self_provider=(
                            fallback_plan.self_provider
                            if "self" in fallback_channels
                            else next_settings.intent.stt.provider
                        ),
                        peer_provider=(
                            fallback_plan.peer_provider
                            if "peer" in fallback_channels
                            else next_settings.intent.peer_stt.provider
                        ),
                        fallback_channels=fallback_channels,
                        installation_fallback=bool(
                            fallback_plan.installation_fallback and fallback_channels
                        ),
                    ),
                )
            installation_fallback = bool(fallback_plan.installation_fallback and fallback_channels)
        if next_settings is not self.settings.canonical:
            if await self._route(
                next_settings,
                reload_settings_view=reload_settings_view,
            ):
                self.fallback_sink(fallback_channels, installation_fallback)
                return True
        await self.apply_direct(
            next_settings,
            reload_settings_view=reload_settings_view,
        )
        self.fallback_sink(fallback_channels, installation_fallback)
        return True

    async def _route(
        self,
        next_settings: AppSettingsVNext,
        *,
        reload_settings_view: bool,
    ) -> bool:
        if await self._apply_combined(
            next_settings,
            reload_settings_view=reload_settings_view,
        ):
            return True
        if await self._apply_stt_language_audio(
            next_settings,
            reload_settings_view=reload_settings_view,
        ):
            return True
        if await self._apply_overlay_osc_output(next_settings):
            return True
        return await self._apply_ui_prompt_clipboard_state(next_settings)

    def notify_fallback(
        self,
        channels: tuple[str, ...],
        installation_fallback: bool,
    ) -> None:
        self.fallback_sink(channels, installation_fallback)

    def persist_manual_fallback(self, *, channel: str | None = None) -> bool:
        previous = self.settings.canonical
        if previous is None:
            return False
        plan = self.manual_fallback.plan(
            self.manual_fallback.state(
                previous,
                cpu_auto_available=self.cpu_auto_available(),
            ),
            channel=channel,
        )
        if not plan.changed:
            return True
        normalized = self.manual_fallback.apply(previous, plan)
        self.settings.canonical = normalized
        if not self.settings.save_current(
            failure_sink=lambda exc: self.failure_sink(f"Failed to save settings: {exc}")
        ):
            self.settings.canonical = previous
            return False
        self.sync_ui()
        self.fallback_sink(
            plan.fallback_channels,
            plan.installation_fallback,
        )
        self.fallback_log_sink(
            previous,
            normalized,
            plan.fallback_channels,
        )
        return True

    async def apply_direct(
        self,
        next_settings: AppSettingsVNext,
        *,
        persist: bool = True,
        strict_runtime_errors: bool = False,
        strict_persistence_errors: bool = False,
        reload_settings_view: bool = True,
    ) -> None:
        next_settings = await self.runtime_effects.preserve_before_replace(next_settings)
        if persist:
            baseline = self.settings.projection_snapshot or self.settings.canonical
            self.settings.begin(snapshot=baseline)
        committed = not persist
        try:
            if persist:
                self.runtime_effects.capture_runtime_signatures()
                self.settings.apply_canonical_delta(baseline, next_settings)
            transition = await self.runtime_effects.prepare(
                self.settings.canonical,
                next_settings,
            )
            self.settings.canonical = transition.settings
            self.runtime_effects.activate_before_persist(transition)
            if persist:
                if strict_persistence_errors:
                    try:
                        self.settings.persist()
                    except Exception:
                        raise StrictSettingsSaveFailed from None
                    self.settings.remember_projection(transition.settings)
                elif not self.settings.save_current(
                    failure_sink=lambda exc: self.failure_sink(f"Failed to save settings: {exc}")
                ):
                    return
                committed = True
            await self.runtime_effects.apply_after_persist(
                transition,
                strict_runtime_errors=strict_runtime_errors,
                reload_settings_view=reload_settings_view,
            )
            self.projection.remember_all(self.settings.canonical)
        finally:
            if persist:
                if committed:
                    self.settings.complete()
                else:
                    self.settings.rollback()

    async def _apply_runtime_effect(
        self,
        settings: object,
        reload_settings_view: bool,
    ) -> None:
        await self.apply_direct(
            settings,
            persist=False,
            strict_runtime_errors=True,
            reload_settings_view=reload_settings_view,
        )

    async def apply_ui_prompt_clipboard_state(
        self,
        next_settings: AppSettingsVNext,
    ) -> bool:
        return await self._apply_ui_prompt_clipboard_state(next_settings)

    async def apply_overlay_osc_output(
        self,
        next_settings: AppSettingsVNext,
    ) -> bool:
        return await self._apply_overlay_osc_output(next_settings)

    async def apply_language_selection(
        self,
        change: LanguageSelectionChange,
    ) -> None:
        current = self.settings.canonical
        if current is None:
            return
        updated = _with_intent(
            current,
            languages=replace(
                current.intent.languages,
                source_language=change.source_code,
                target_language=change.target_code,
                secondary_target_language=change.secondary_target_code,
                peer_source_mode=change.peer_source_mode,
                peer_source_language=change.peer_source_code,
                peer_target_language=change.peer_target_code,
                recent_source_languages=list(change.recent_source_codes),
                recent_target_languages=list(change.recent_target_codes),
            ),
        )
        await self.apply(updated)
        if self.settings.canonical is not None:
            self.projection.render(
                self.settings.canonical,
                preserve_custom_vocab_draft=True,
            )

    async def compensate_failed_local_asr_settings_apply(
        self,
        *,
        base_settings: AppSettingsVNext,
        committed_settings: AppSettingsVNext,
        reload_settings_view: bool = True,
    ) -> None:
        self.settings.begin(snapshot=committed_settings)
        committed = False
        try:
            self.settings.apply_canonical_delta(
                committed_settings,
                base_settings,
            )
            await asyncio.to_thread(self.settings.persist)
            self.settings.remember_projection(base_settings)
            committed = True
        finally:
            if committed:
                self.settings.complete()
            else:
                self.settings.rollback()
        await self.apply_direct(
            copy.deepcopy(base_settings),
            persist=False,
            strict_runtime_errors=False,
            reload_settings_view=reload_settings_view,
        )
        self.settings.canonical = copy.deepcopy(base_settings)
        if reload_settings_view:
            self.projection.render(
                self.settings.canonical,
                preserve_custom_vocab_draft=True,
            )

    async def _apply_combined(
        self,
        next_settings: AppSettingsVNext,
        *,
        reload_settings_view: bool,
    ) -> bool:
        order22 = self.projection.order22_patch_base_and_values(next_settings)
        order23 = self.projection.order23_patch_base_and_values(next_settings)
        order24 = self.projection.order24_patch_base_and_values(next_settings)
        if order22 is None or order23 is None or order24 is None:
            return False
        patches = (order22[1], order23[1], order24[1])
        if sum(bool(values) for values in patches) < 2:
            return False
        committed_results: list[TransactionResult] = []

        async def route(
            values: dict[str, object],
            apply_patch: Callable[[object], Awaitable[bool]],
            *,
            runtime_source: object | None = None,
        ) -> bool:
            current = self.settings.canonical
            if current is None:
                return False
            patch_settings = apply_settings_path_patch(current, values)
            if runtime_source is not None:
                _copy_runtime_only_ui_state(runtime_source, patch_settings)
            if not await apply_patch(patch_settings):
                return False
            if self.results.current is not None and _settings_mutation_committed(
                self.results.current
            ):
                committed_results.append(self.results.current)
            return True

        if patches[0]:
            if not await route(
                patches[0],
                lambda settings: self._apply_stt_language_audio(
                    settings,
                    reload_settings_view=False,
                ),
            ):
                return False
            if not self._last_result_committed():
                return True
        if patches[1]:
            if not await route(patches[1], self._apply_overlay_osc_output):
                return False
            if not self._last_result_committed():
                return True
        if patches[2]:
            if not await route(
                patches[2],
                self._apply_ui_prompt_clipboard_state,
                runtime_source=next_settings,
            ):
                return False
            if not self._last_result_committed():
                return True

        committed_before_full_draft = (
            copy.deepcopy(self.settings.canonical) if self.settings.canonical is not None else None
        )
        if self.settings.canonical is not None and self.settings.snapshot_values(
            self.settings.canonical
        ) != self.settings.snapshot_values(next_settings):
            try:
                await self.apply_direct(
                    next_settings,
                    strict_runtime_errors=True,
                    strict_persistence_errors=True,
                    reload_settings_view=reload_settings_view,
                )
            except StrictSettingsSaveFailed:
                if committed_before_full_draft is not None:
                    await self._resync_committed_runtime(
                        base_settings=committed_before_full_draft,
                        committed_settings=committed_before_full_draft,
                        failure_message="Failed to resync committed order24 settings runtime",
                    )
                self._set_result(
                    _ui_prompt_clipboard_state_save_failed_transaction_result(
                        operation="apply_order22_order23_order24_full_draft_save"
                    )
                )
            except Exception:
                self._set_result(_ui_prompt_clipboard_state_runtime_degraded_transaction_result())

        if reload_settings_view and self.settings.canonical is not None:
            self.projection.render(
                self.settings.canonical,
                preserve_custom_vocab_draft=True,
            )
        if (
            self.results.current is not None
            and self.results.current.status
            == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
        ):
            degraded = next(
                (
                    result
                    for result in committed_results
                    if result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
                ),
                None,
            )
            if degraded is not None:
                self._set_result(degraded)
        return True

    async def _apply_stt_language_audio(
        self,
        next_settings: AppSettingsVNext,
        *,
        reload_settings_view: bool = True,
    ) -> bool:
        base_and_patch = self.projection.order22_patch_base_and_values(next_settings)
        if base_and_patch is None:
            return False
        base_settings, patch_values = base_and_patch
        if not patch_values:
            return False
        committed_settings = apply_settings_path_patch(base_settings, patch_values)
        has_out_of_scope_draft = self.settings.snapshot_values(
            committed_settings
        ) != self.settings.snapshot_values(next_settings)
        runtime_apply = (
            NoopRuntimeApply()
            if has_out_of_scope_draft
            else SttLanguageAudioRuntimeApplyAdapter(
                apply_settings=self._apply_runtime_effect,
                state_provider=self.runtime_effects.state,
                settings=committed_settings,
                reload_settings_view=reload_settings_view,
            )
        )
        result = await self._mutate(
            command=SttLanguageAudioSettingsMutation(values=patch_values),
            base_settings=base_settings,
            committed_settings=committed_settings,
            surface="stt_language_audio",
            runtime_apply=runtime_apply,
        )
        if not _settings_mutation_committed(result):
            self.settings.canonical = copy.deepcopy(base_settings)
            self.projection.remember_order22(self.settings.canonical)
            return True
        if self.consume_superseded_settings(committed_settings):
            self.projection.remember_order22(self.settings.canonical)
            return True
        if (
            not has_out_of_scope_draft
            and result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
            and self.active_local_asr_change(
                self.settings.project(base_settings, authoritative=True),
                self.settings.project(committed_settings, authoritative=True),
            )
        ):
            try:
                await self.compensate_failed_local_asr_settings_apply(
                    base_settings=base_settings,
                    committed_settings=committed_settings,
                    reload_settings_view=reload_settings_view,
                )
            except Exception:
                self.failure_sink("Failed to compensate local ASR settings apply")
            self.projection.remember_order22(self.settings.canonical)
            return True
        if has_out_of_scope_draft:
            try:
                await self.apply_direct(
                    next_settings,
                    strict_runtime_errors=True,
                    strict_persistence_errors=True,
                    reload_settings_view=reload_settings_view,
                )
            except StrictSettingsSaveFailed:
                await self._resync_committed_runtime(
                    base_settings=base_settings,
                    committed_settings=committed_settings,
                    failure_message="Failed to resync committed order22 settings runtime",
                )
                self._set_result(
                    _stt_language_audio_save_failed_transaction_result(
                        operation="apply_stt_language_audio_full_draft_save"
                    )
                )
            except Exception:
                self._set_result(_stt_language_audio_runtime_degraded_transaction_result())
            else:
                unavailable = _stt_language_audio_runtime_unavailable_result(
                    state=self.runtime_effects.state(next_settings),
                    settings=next_settings,
                )
                if unavailable is not None:
                    self._set_result(_runtime_apply_result_as_degraded_transaction(unavailable))
        else:
            self.settings.canonical = committed_settings
            if result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED:
                self.runtime_effects.sync_signatures(committed_settings)
        self.projection.remember_order22(self.settings.canonical)
        return True

    async def _apply_overlay_osc_output(self, next_settings: AppSettingsVNext) -> bool:
        base_and_patch = self.projection.order23_patch_base_and_values(next_settings)
        if base_and_patch is None:
            return False
        base_settings, patch_values = base_and_patch
        if not patch_values:
            return False
        next_settings = copy.deepcopy(next_settings)
        await self.runtime_effects.prepare_overlay_persistence(
            base_settings,
            next_settings,
        )
        patch_values = build_overlay_osc_output_settings_path_patch(
            base_settings,
            next_settings,
        )
        committed_settings = apply_settings_path_patch(base_settings, patch_values)
        has_out_of_scope_draft = self.settings.snapshot_values(
            committed_settings
        ) != self.settings.snapshot_values(next_settings)
        runtime_apply = (
            NoopRuntimeApply()
            if has_out_of_scope_draft
            else OverlayOscOutputRuntimeApplyAdapter(
                apply_settings=self._apply_runtime_effect,
                settings=committed_settings,
            )
        )
        result = await self._mutate(
            command=OverlayOscOutputSettingsMutation(values=patch_values),
            base_settings=base_settings,
            committed_settings=committed_settings,
            surface="overlay_osc_output",
            runtime_apply=runtime_apply,
        )
        if not _settings_mutation_committed(result):
            self.settings.canonical = copy.deepcopy(base_settings)
            self.projection.remember_order23(self.settings.canonical)
            return True
        if has_out_of_scope_draft:
            try:
                await self.apply_direct(
                    next_settings,
                    strict_runtime_errors=True,
                    strict_persistence_errors=True,
                )
            except StrictSettingsSaveFailed:
                await self._resync_committed_runtime(
                    base_settings=base_settings,
                    committed_settings=committed_settings,
                    failure_message="Failed to resync committed order23 settings runtime",
                )
                self._set_result(
                    _overlay_osc_output_save_failed_transaction_result(
                        operation="apply_overlay_osc_output_full_draft_save"
                    )
                )
            except Exception:
                self._set_result(_overlay_osc_output_runtime_degraded_transaction_result())
        elif self.settings.canonical is None or self.settings.canonical is base_settings:
            self.settings.canonical = committed_settings
        self.projection.remember_order23(self.settings.canonical)
        return True

    async def _apply_ui_prompt_clipboard_state(
        self,
        next_settings: AppSettingsVNext,
    ) -> bool:
        base_and_patch = self.projection.order24_patch_base_and_values(next_settings)
        if base_and_patch is None:
            return False
        base_settings, patch_values = base_and_patch
        if not patch_values:
            return False
        committed_settings = apply_settings_path_patch(base_settings, patch_values)
        has_out_of_scope_draft = self.settings.snapshot_values(
            committed_settings
        ) != self.settings.snapshot_values(next_settings)
        runtime_settings = copy.deepcopy(next_settings)
        runtime_apply = (
            NoopRuntimeApply()
            if has_out_of_scope_draft
            else UiPromptClipboardStateRuntimeApplyAdapter(
                apply_settings=self._apply_runtime_effect,
                settings=runtime_settings,
            )
        )
        result = await self._mutate(
            command=UiPromptClipboardStateSettingsMutation(values=patch_values),
            base_settings=self.settings.canonical or committed_settings,
            committed_settings=committed_settings,
            surface="ui_prompt_clipboard_state",
            runtime_apply=runtime_apply,
        )
        if not _settings_mutation_committed(result):
            self.settings.canonical = copy.deepcopy(base_settings)
            self.projection.remember_order24(self.settings.canonical)
            return True
        if has_out_of_scope_draft:
            try:
                await self.apply_direct(
                    next_settings,
                    strict_runtime_errors=True,
                    strict_persistence_errors=True,
                )
            except StrictSettingsSaveFailed:
                await self._resync_committed_runtime(
                    base_settings=base_settings,
                    committed_settings=committed_settings,
                    failure_message="Failed to resync committed order24 settings runtime",
                )
                self._set_result(
                    _ui_prompt_clipboard_state_save_failed_transaction_result(
                        operation="apply_ui_prompt_clipboard_state_full_draft_save"
                    )
                )
            except Exception:
                self._set_result(_ui_prompt_clipboard_state_runtime_degraded_transaction_result())
        else:
            self.settings.canonical = runtime_settings
            if result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED:
                self.runtime_effects.sync_signatures(runtime_settings)
        self.projection.remember_order24(self.settings.canonical)
        return True

    async def _mutate(
        self,
        *,
        command: SettingsMutationCommand,
        base_settings: AppSettingsVNext,
        committed_settings: AppSettingsVNext,
        surface: str,
        runtime_apply: RuntimeApplyPort,
    ) -> TransactionResult:
        repository = self.settings.create_canonical_patch_repository(
            base_settings=base_settings,
            committed_settings=committed_settings,
            surface=surface,
            save_failure_sink=self.failure_sink,
        )
        service = self.mutation_service_provider() or SettingsMutationService(
            settings_repository=repository,
            runtime_apply=runtime_apply,
            validator=settings_path_mutation_validator_for_command(command),
        )
        result: TransactionResult | None = None
        try:
            result = await service.mutate(
                command.to_mutation_request(
                    expected_revision=None,
                    correlation_id=None,
                )
            )
        finally:
            if getattr(repository, "commit_succeeded", False) or (
                result is not None and _settings_mutation_committed(result)
            ):
                self.settings.complete()
            else:
                self.settings.rollback()
        if result is None:
            raise RuntimeError("settings mutation completed without a result")
        self._set_result(result)
        return result

    async def _resync_committed_runtime(
        self,
        *,
        base_settings: AppSettingsVNext,
        committed_settings: AppSettingsVNext,
        failure_message: str,
    ) -> None:
        self.runtime_effects.restore_memory(base_settings)
        try:
            await self.apply_direct(
                copy.deepcopy(committed_settings),
                persist=False,
                strict_runtime_errors=True,
            )
        except Exception:
            self.failure_sink(failure_message)
            self.runtime_effects.restore_memory(committed_settings)

    def _set_result(self, result: TransactionResult) -> None:
        self.results.set(result)

    def _last_result_committed(self) -> bool:
        return self.results.committed()


__all__ = [
    "SettingsApplicationOwner",
    "StrictSettingsSaveFailed",
]
