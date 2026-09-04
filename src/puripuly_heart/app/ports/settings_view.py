from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from puripuly_heart.config.provider_values import (
    LLMProviderName,
    OpenRouterCredentialSource,
    OpenRouterLLMModel,
    OpenRouterSelectionAlias,
    QwenRegion,
    STTProviderName,
)
from puripuly_heart.config.translation_values import (
    TranslationConnection,
    TranslationModel,
)


@dataclass(frozen=True, slots=True)
class TranslationFallbackSnapshot:
    enabled: bool
    model: TranslationModel
    connection: TranslationConnection


@dataclass(frozen=True, slots=True)
class TranslationSelectionSnapshot:
    model: TranslationModel
    connection: TranslationConnection
    connection_history: tuple[tuple[TranslationModel, TranslationConnection], ...]
    fallback: TranslationFallbackSnapshot
    http_extension_id: str | None
    previous_llm_model: TranslationModel | None
    gpu_device_id: str


@dataclass(frozen=True, slots=True)
class ProviderVerificationSnapshot:
    deepgram: bool
    gemini_transcribe: bool
    elevenlabs_scribe: bool
    soniox: bool
    google: bool
    openrouter: bool
    deepseek: bool
    alibaba_beijing: bool
    alibaba_singapore: bool
    cerebras: bool


@dataclass(frozen=True, slots=True)
class ProviderSettingsSnapshot:
    stt_provider: STTProviderName
    peer_stt_provider: STTProviderName
    cloud_free_tier_providers: tuple[STTProviderName, ...]
    llm_provider: LLMProviderName
    translation: TranslationSelectionSnapshot
    stt_gpu_device_id: str
    qwen_region: QwenRegion
    qwen_asr_model: str
    local_llm_base_url: str
    local_llm_model: str
    local_llm_extra_body_json: str
    custom_stt_mode: str
    custom_stt_compatibility: str
    custom_stt_endpoint: str
    custom_stt_model: str
    custom_stt_extra_json: str
    openrouter_llm_model: OpenRouterLLMModel
    openrouter_selected_source: OpenRouterCredentialSource
    openrouter_selection_alias: OpenRouterSelectionAlias | None
    verified: ProviderVerificationSnapshot
    managed_referral_id: str | None


@dataclass(frozen=True, slots=True)
class GeneralSettingsSnapshot:
    locale: str
    effective_peer_source_language: str
    input_host_api: str
    input_device: str
    output_device: str
    self_vad_speech_threshold: float
    peer_vad_speech_threshold: float
    peer_vad_hangover_ms: int
    peer_vad_pre_roll_ms: int
    osc_connection_mode: str
    osc_port: int
    osc_send_port: int | None
    osc_receive_port: int
    vrc_mic_intercept: bool
    chatbox_include_source: bool
    clipboard_auto_translate_enabled: bool
    telemetry_enabled: bool
    peer_expected_languages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PromptSettingsSnapshot:
    active_provider_key: str
    source_language: str
    system_prompt: str
    custom_vocabulary_enabled: bool
    custom_vocabulary_terms: tuple[str, ...]
    custom_vocabulary_other_languages_have_terms: bool


@dataclass(frozen=True, slots=True)
class OverlayCalibrationSnapshot:
    anchor: str
    distance: float
    offset_x: float
    offset_y: float
    text_scale: float


@dataclass(frozen=True, slots=True)
class OverlaySettingsSnapshot:
    target: str
    show_translation: bool
    show_peer_original: bool
    desktop_size_preset: str
    desktop_background_alpha: float
    desktop_swap_caption_languages: bool
    calibration: OverlayCalibrationSnapshot


@dataclass(frozen=True, slots=True)
class LocaleSettingsIntent:
    locale: str


@dataclass(frozen=True, slots=True)
class AudioInputSettingsIntent:
    input_host_api: str
    input_device: str


@dataclass(frozen=True, slots=True)
class DesktopAudioOutputSettingsIntent:
    output_device: str


AudioSettingsChange: TypeAlias = AudioInputSettingsIntent | DesktopAudioOutputSettingsIntent


@dataclass(frozen=True, slots=True)
class AudioSettingsIntent:
    changes: tuple[AudioSettingsChange, ...]


@dataclass(frozen=True, slots=True)
class SelfVadSettingsIntent:
    speech_threshold: float


@dataclass(frozen=True, slots=True)
class PeerVadSpeechThresholdIntent:
    speech_threshold: float


@dataclass(frozen=True, slots=True)
class PeerVadHangoverIntent:
    hangover_ms: int


@dataclass(frozen=True, slots=True)
class PeerVadPreRollIntent:
    pre_roll_ms: int


@dataclass(frozen=True, slots=True)
class OscConnectionSettingsIntent:
    connection_mode: str
    send_port: int | None
    receive_port: int


@dataclass(frozen=True, slots=True)
class VrcMicInterceptSettingsIntent:
    enabled: bool


@dataclass(frozen=True, slots=True)
class ChatboxSourceSettingsIntent:
    enabled: bool


@dataclass(frozen=True, slots=True)
class ClipboardSettingsIntent:
    enabled: bool


@dataclass(frozen=True, slots=True)
class PeerExpectedLanguagesIntent:
    languages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CustomVocabularySettingsIntent:
    source_language: str
    terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OverlayTargetSettingsIntent:
    target: str


@dataclass(frozen=True, slots=True)
class OverlayTranslationSettingsIntent:
    enabled: bool


@dataclass(frozen=True, slots=True)
class OverlayPeerOriginalSettingsIntent:
    enabled: bool


@dataclass(frozen=True, slots=True)
class DesktopOverlayBackgroundAlphaIntent:
    background_alpha: float


@dataclass(frozen=True, slots=True)
class DesktopOverlaySwapCaptionLanguagesIntent:
    enabled: bool


@dataclass(frozen=True, slots=True)
class DesktopOverlaySizeIntent:
    size_preset: str


@dataclass(frozen=True, slots=True)
class DesktopOverlayPositionResetIntent:
    pass


@dataclass(frozen=True, slots=True)
class OverlayCalibrationSettingsIntent:
    calibration: OverlayCalibrationSnapshot


ImmediateSettingsIntent: TypeAlias = (
    LocaleSettingsIntent
    | AudioSettingsIntent
    | SelfVadSettingsIntent
    | PeerVadSpeechThresholdIntent
    | PeerVadHangoverIntent
    | PeerVadPreRollIntent
    | OscConnectionSettingsIntent
    | VrcMicInterceptSettingsIntent
    | ChatboxSourceSettingsIntent
    | ClipboardSettingsIntent
    | PeerExpectedLanguagesIntent
    | CustomVocabularySettingsIntent
    | OverlayTargetSettingsIntent
    | OverlayTranslationSettingsIntent
    | OverlayPeerOriginalSettingsIntent
    | DesktopOverlayBackgroundAlphaIntent
    | DesktopOverlaySwapCaptionLanguagesIntent
    | DesktopOverlaySizeIntent
    | DesktopOverlayPositionResetIntent
    | OverlayCalibrationSettingsIntent
)


@dataclass(frozen=True, slots=True)
class SelfSttProviderEdit:
    provider: STTProviderName


@dataclass(frozen=True, slots=True)
class CloudFreeTierProvidersEdit:
    providers: tuple[STTProviderName, ...]


@dataclass(frozen=True, slots=True)
class PeerSttProviderEdit:
    provider: STTProviderName


@dataclass(frozen=True, slots=True)
class SttGpuDeviceEdit:
    device_id: str


@dataclass(frozen=True, slots=True)
class LlmGpuDeviceEdit:
    device_id: str


@dataclass(frozen=True, slots=True)
class TranslationSelectionEdit:
    selection: TranslationSelectionSnapshot
    history_updates: tuple[tuple[TranslationModel, TranslationConnection], ...]


@dataclass(frozen=True, slots=True)
class TranslationFallbackEdit:
    fallback: TranslationFallbackSnapshot


@dataclass(frozen=True, slots=True)
class TranslationHttpExtensionEdit:
    extension_id: str


@dataclass(frozen=True, slots=True)
class QwenRegionEdit:
    region: QwenRegion


@dataclass(frozen=True, slots=True)
class QwenAsrModelEdit:
    model: str


@dataclass(frozen=True, slots=True)
class LocalLlmBaseUrlEdit:
    base_url: str


@dataclass(frozen=True, slots=True)
class LocalLlmModelEdit:
    model: str


@dataclass(frozen=True, slots=True)
class LocalLlmExtraBodyEdit:
    extra_body_json: str


@dataclass(frozen=True, slots=True)
class CustomSttEndpointEdit:
    endpoint: str


@dataclass(frozen=True, slots=True)
class CustomSttModelEdit:
    model: str


@dataclass(frozen=True, slots=True)
class CustomSttExtraEdit:
    extra_json: str


@dataclass(frozen=True, slots=True)
class ManagedReferralEdit:
    referral_id: str | None


@dataclass(frozen=True, slots=True)
class SystemPromptEdit:
    value: str


ProviderSettingsEdit: TypeAlias = (
    SelfSttProviderEdit
    | CloudFreeTierProvidersEdit
    | PeerSttProviderEdit
    | SttGpuDeviceEdit
    | LlmGpuDeviceEdit
    | TranslationSelectionEdit
    | TranslationFallbackEdit
    | TranslationHttpExtensionEdit
    | QwenRegionEdit
    | QwenAsrModelEdit
    | LocalLlmBaseUrlEdit
    | LocalLlmModelEdit
    | LocalLlmExtraBodyEdit
    | CustomSttEndpointEdit
    | CustomSttModelEdit
    | CustomSttExtraEdit
    | ManagedReferralEdit
    | SystemPromptEdit
)


@dataclass(frozen=True, slots=True)
class ProviderApplyIntent:
    edits: tuple[ProviderSettingsEdit, ...]


@dataclass(frozen=True, slots=True)
class PromptApplyIntent:
    value: str


@dataclass(frozen=True, slots=True)
class OpenRouterPkceTarget:
    selection_alias: OpenRouterSelectionAlias
    provider_intent: ProviderApplyIntent = ProviderApplyIntent(())
    system_prompt: str | None = None


__all__ = [
    "AudioInputSettingsIntent",
    "AudioSettingsChange",
    "AudioSettingsIntent",
    "ChatboxSourceSettingsIntent",
    "CloudFreeTierProvidersEdit",
    "ClipboardSettingsIntent",
    "CustomSttEndpointEdit",
    "CustomSttExtraEdit",
    "CustomSttModelEdit",
    "CustomVocabularySettingsIntent",
    "DesktopAudioOutputSettingsIntent",
    "DesktopOverlayBackgroundAlphaIntent",
    "DesktopOverlaySizeIntent",
    "DesktopOverlayPositionResetIntent",
    "DesktopOverlaySwapCaptionLanguagesIntent",
    "GeneralSettingsSnapshot",
    "ImmediateSettingsIntent",
    "LlmGpuDeviceEdit",
    "LocalLlmBaseUrlEdit",
    "LocalLlmExtraBodyEdit",
    "LocalLlmModelEdit",
    "LocaleSettingsIntent",
    "ManagedReferralEdit",
    "OpenRouterPkceTarget",
    "OscConnectionSettingsIntent",
    "OverlayCalibrationSnapshot",
    "OverlayCalibrationSettingsIntent",
    "OverlayPeerOriginalSettingsIntent",
    "OverlaySettingsSnapshot",
    "OverlayTargetSettingsIntent",
    "OverlayTranslationSettingsIntent",
    "PeerExpectedLanguagesIntent",
    "PeerSttProviderEdit",
    "PeerVadHangoverIntent",
    "PeerVadPreRollIntent",
    "PeerVadSpeechThresholdIntent",
    "PromptApplyIntent",
    "PromptSettingsSnapshot",
    "ProviderApplyIntent",
    "ProviderSettingsEdit",
    "ProviderSettingsSnapshot",
    "ProviderVerificationSnapshot",
    "QwenRegionEdit",
    "QwenAsrModelEdit",
    "SelfSttProviderEdit",
    "SelfVadSettingsIntent",
    "SttGpuDeviceEdit",
    "SystemPromptEdit",
    "TranslationFallbackSnapshot",
    "TranslationFallbackEdit",
    "TranslationSelectionSnapshot",
    "TranslationSelectionEdit",
    "TranslationHttpExtensionEdit",
    "VrcMicInterceptSettingsIntent",
]
