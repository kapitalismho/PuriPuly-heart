"""Settings view - Bento grid layout with SegmentedButton providers."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import inspect
import json
import logging
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Callable, Mapping

import flet as ft
from puripuly_heart.app.services.local_asr_selection import resolve_local_asr_selection
from puripuly_heart.core.managed_openrouter_release import TalkTogetherPassStatus

from puripuly_heart.app.ports.settings_secrets import (
    SettingsSecretKey,
    SettingsSecretLoadResult,
    SettingsSecretMutation,
    SettingsSecretSnapshot,
    SettingsSecretsPort,
)
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
    OpenRouterPkceTarget,
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
    ProviderSettingsEdit,
    ProviderSettingsSnapshot,
    QwenRegionEdit,
    SelfSttProviderEdit,
    SelfVadSettingsIntent,
    SttGpuDeviceEdit,
    SttRollingEnabledEdit,
    SystemPromptEdit,
    TranslationFallbackEdit,
    TranslationFallbackSnapshot,
    TranslationHttpExtensionEdit,
    TranslationSelectionEdit,
    TranslationSelectionSnapshot,
    VrcMicInterceptSettingsIntent,
)
from puripuly_heart.app.ports.ui_models import OscControlPresentationState
from puripuly_heart.app.services.http_extension_registry import (
    HttpExtensionRegistryService,
)
from puripuly_heart.config.desktop_overlay_values import (
    DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA,
    DESKTOP_FLET_SIZE_PRESET_DISPLAY_ORDER,
    DESKTOP_FLET_SIZE_PRESET_ORDER,
)
from puripuly_heart.config.llm_profiles import (
    profile_for_alias,
)
from puripuly_heart.config.overlay_calibration import (
    OVERLAY_CALIBRATION_ANCHORS,
    OverlayCalibration,
)
from puripuly_heart.config.prompts import load_prompt_for_provider
from puripuly_heart.config.provider_values import (
    LOCAL_LLM_RESERVED_EXTRA_BODY_KEYS,
    LOCAL_LLM_SENSITIVE_EXTRA_BODY_KEYS,
    MAX_CUSTOM_VOCAB_TERMS,
    LLMProviderName,
    OpenRouterCredentialSource,
    OpenRouterLLMModel,
    OpenRouterSelectionAlias,
    QwenRegion,
    STTProviderName,
    display_stt_provider,
    is_custom_stt_provider,
    is_qwen_cloud_stt_provider,
    normalize_local_llm_base_url,
    normalize_owned_referral_id,
)
from puripuly_heart.config.resolved import (
    OVERLAY_TARGET_DESKTOP,
    OVERLAY_TARGET_STEAMVR,
)
from puripuly_heart.config.translation_values import (
    TranslationConnection,
    TranslationModel,
    default_translation_connection,
    supported_translation_connections,
)
from puripuly_heart.core.http_extensions import http_extension_secret_key
from puripuly_heart.core.language import get_stt_compatibility_warning
from puripuly_heart.core.stt.custom import (
    CustomSTTConfigurationError,
    normalize_custom_stt_extra,
)
from puripuly_heart.ui.components.managed_trial_usage_bar import ManagedTrialUsageBar
from puripuly_heart.ui.components.settings import (
    ApiKeyField,
    AudioSettings,
    CustomVocabularyTagEditor,
    LanguageHintEditor,
    OptionItem,
    OscConnectionModal,
    PromptEditor,
    SettingsModal,
    SettingsUnitCard,
)
from puripuly_heart.ui.components.shared_card_wrapper import SharedCardWrapper
from puripuly_heart.ui.components.subtab_shell import TextSubtab, TextSubtabShell
from puripuly_heart.ui.flet_runtime import (
    is_control_mounted,
    is_hover_active,
    update_control_if_mounted,
)
from puripuly_heart.ui.fonts import font_for_language
from puripuly_heart.ui.gpu_device import GpuDeviceOption
from puripuly_heart.ui.i18n import (
    available_locales,
    get_locale,
    language_name,
    locale_label,
    provider_label,
    t,
)
from puripuly_heart.ui.overlay_peer_contract import OverlayPeerConsumerContract
from puripuly_heart.ui.settings.contract import (
    SettingsApiSurfaceSlots,
    SettingsGeneralIntents,
    SettingsGeneralSurfaceSlots,
    SettingsOverlayIntents,
    SettingsOverlaySurfaceSlots,
    SettingsPromptIntents,
    SettingsPromptSurfaceSlots,
    SettingsProviderIntents,
    SettingsSurfaceIntents,
)
from puripuly_heart.ui.settings.renderer import (
    SETTINGS_ROW_SPACING,
    compose_settings_api_surface,
    compose_settings_general_surface,
    compose_settings_overlay_surface,
    compose_settings_prompt_surface,
)
from puripuly_heart.ui.theme import (
    COLOR_DIVIDER,
    COLOR_NEUTRAL_DARK,
    COLOR_ON_BACKGROUND,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
)

logger = logging.getLogger(__name__)

_CJK_START = 0x3000
_CENTER_ALIGNMENT = ft.Alignment(0, 0)
_CENTER_RIGHT_ALIGNMENT = ft.Alignment(1, 0)
_SETTINGS_SUBTAB_ORDER = ("api", "general", "prompt", "overlay")
_OVERLAY_DISTANCE_MIN = 0.5
_OVERLAY_DISTANCE_MAX = 2.0
_OVERLAY_DISTANCE_DIVISIONS = 30
_OVERLAY_OFFSET_STEP = 0.05
_DESKTOP_OVERLAY_BACKGROUND_ALPHA_STEP = 0.1
_OVERLAY_TEXT_SCALE_PRESETS = (
    ("large", 1.2),
    ("normal", 1.0),
    ("small", 0.8),
)
_DESKTOP_OVERLAY_REOPEN_FAILURE_REASONS = frozenset({"window_configuration_failed"})
_CUSTOM_VOCAB_DELIMITER_RE = re.compile(r"\s+")
_STT_UI_PROVIDERS = (
    STTProviderName.LOCAL_CPU_AUTO,
    STTProviderName.LOCAL_PARAKEET_V3,
    STTProviderName.LOCAL_PARAKEET_JAPANESE,
    STTProviderName.LOCAL_QWEN,
    STTProviderName.LOCAL_QWEN_GPU,
    STTProviderName.DEEPGRAM,
    STTProviderName.GEMINI_TRANSCRIBE,
    STTProviderName.ELEVENLABS_SCRIBE,
    STTProviderName.QWEN_ASR,
    STTProviderName.QWEN_AUDIO,
    STTProviderName.SONIOX,
    STTProviderName.CUSTOM_OFFLINE,
    STTProviderName.CUSTOM_REALTIME,
)
_STT_SECTION_ORDER = (
    "settings.stt.section.recommended_cloud",
    "settings.stt.section.recommended_local",
    "settings.stt.section.cloud",
    "settings.stt.section.gpu_inference",
    "settings.stt.section.cpu_inference",
    "settings.stt.section.custom",
)
_STT_SECTION_BY_PROVIDER: dict[STTProviderName, str] = {
    STTProviderName.DEEPGRAM: "settings.stt.section.recommended_cloud",
    STTProviderName.GEMINI_TRANSCRIBE: "settings.stt.section.cloud",
    STTProviderName.ELEVENLABS_SCRIBE: "settings.stt.section.cloud",
    STTProviderName.SONIOX: "settings.stt.section.recommended_cloud",
    STTProviderName.LOCAL_CPU_AUTO: "settings.stt.section.recommended_local",
    STTProviderName.QWEN_ASR: "settings.stt.section.cloud",
    STTProviderName.QWEN_AUDIO: "settings.stt.section.cloud",
    STTProviderName.CUSTOM: "settings.stt.section.custom",
    STTProviderName.CUSTOM_OFFLINE: "settings.stt.section.custom",
    STTProviderName.CUSTOM_REALTIME: "settings.stt.section.custom",
    STTProviderName.LOCAL_QWEN_GPU: "settings.stt.section.gpu_inference",
    STTProviderName.LOCAL_PARAKEET_V3: "settings.stt.section.cpu_inference",
    STTProviderName.LOCAL_PARAKEET_JAPANESE: "settings.stt.section.cpu_inference",
    STTProviderName.LOCAL_QWEN: "settings.stt.section.cpu_inference",
}
_TRANSLATION_MODEL_LABEL_KEYS = {
    TranslationModel.MANAGED_GEMMA: "provider.managed_gemma",
    TranslationModel.MANAGED_GEMMA_12B: "provider.managed_gemma_12b",
    TranslationModel.GEMMA4_26B_31B: "provider.gemma4_26b_31b",
    TranslationModel.GEMMA4_31B: "provider.gemma4_31b",
    TranslationModel.GEMMA4: "provider.gemma4_26b_a4b_it",
    TranslationModel.DEEPSEEK_V4_FLASH: "provider.deepseek_v4_flash",
    TranslationModel.GEMINI_37_FLASH: "provider.gemini37_flash",
    TranslationModel.QWEN_38_FLASH: "provider.qwen38_flash",
    TranslationModel.LOCAL_LLM: "provider.local_llms",
    TranslationModel.CUSTOM_HTTP: "provider.custom_http",
}
_TRANSLATION_CONNECTION_LABEL_KEYS = {
    TranslationConnection.CPU: "settings.translation_connection.cpu",
    TranslationConnection.GPU: "settings.translation_connection.gpu",
    TranslationConnection.MANAGED: "settings.translation_connection.managed",
    TranslationConnection.MANAGED_CHINA: "settings.translation_connection.managed_china",
    TranslationConnection.OPENROUTER: "settings.translation_connection.openrouter",
    TranslationConnection.CEREBRAS: "settings.translation_connection.cerebras",
    TranslationConnection.OFFICIAL_BYOK: "settings.translation_connection.official_byok",
    TranslationConnection.OLLAMA: "settings.translation_connection.ollama",
    TranslationConnection.CUSTOM_HTTP: "settings.translation_connection.custom_http",
}
_TRANSLATION_CONNECTION_DESCRIPTION_KEYS = {
    TranslationConnection.CEREBRAS: "settings.translation_connection.cerebras.description",
}
_TRANSLATION_CONNECTION_ONLY_SUPPORTED_KEY = "settings.translation_connection.only_supported"
_TRANSLATION_MODELS = (
    TranslationModel.MANAGED_GEMMA,
    TranslationModel.MANAGED_GEMMA_12B,
    TranslationModel.GEMMA4_26B_31B,
    TranslationModel.GEMMA4_31B,
    TranslationModel.GEMMA4,
    TranslationModel.DEEPSEEK_V4_FLASH,
    TranslationModel.LOCAL_LLM,
    TranslationModel.CUSTOM_HTTP,
    TranslationModel.GEMINI_37_FLASH,
    TranslationModel.QWEN_38_FLASH,
)
_TRANSLATION_MODEL_SECTION_ORDER = (
    "settings.translation_model.section.recommended_cloud",
    "settings.translation_model.section.recommended_local",
    "settings.translation_model.section.gpu_inference",
    "settings.translation_model.section.user_settings",
    "settings.translation_model.section.others",
)
_TRANSLATION_MODEL_SECTION_BY_MODEL: dict[TranslationModel, str] = {
    TranslationModel.MANAGED_GEMMA: "settings.translation_model.section.recommended_local",
    TranslationModel.MANAGED_GEMMA_12B: "settings.translation_model.section.gpu_inference",
    TranslationModel.GEMMA4_26B_31B: "settings.translation_model.section.recommended_cloud",
    TranslationModel.GEMMA4_31B: "settings.translation_model.section.recommended_cloud",
    TranslationModel.DEEPSEEK_V4_FLASH: "settings.translation_model.section.recommended_cloud",
    TranslationModel.GEMMA4: "settings.translation_model.section.others",
    TranslationModel.LOCAL_LLM: "settings.translation_model.section.user_settings",
    TranslationModel.CUSTOM_HTTP: "settings.translation_model.section.user_settings",
    TranslationModel.GEMINI_37_FLASH: "settings.translation_model.section.others",
    TranslationModel.QWEN_38_FLASH: "settings.translation_model.section.others",
}
_TRANSLATION_MODELS_WITHOUT_PROVIDER_FALLBACK = frozenset(
    {
        TranslationModel.CUSTOM_HTTP,
        TranslationModel.MANAGED_GEMMA,
        TranslationModel.MANAGED_GEMMA_12B,
        TranslationModel.LOCAL_LLM,
    }
)
_TRANSLATION_FALLBACK_PRESETS: tuple[tuple[str, TranslationFallbackSnapshot, str], ...] = (
    (
        "none",
        TranslationFallbackSnapshot(
            enabled=False,
            model=TranslationModel.DEEPSEEK_V4_FLASH,
            connection=TranslationConnection.OFFICIAL_BYOK,
        ),
        "settings.fallback.none",
    ),
    (
        "deepseek_v4_flash_official",
        TranslationFallbackSnapshot(
            enabled=True,
            model=TranslationModel.DEEPSEEK_V4_FLASH,
            connection=TranslationConnection.OFFICIAL_BYOK,
        ),
        "settings.fallback.deepseek_v4_flash_official",
    ),
    (
        "openrouter_deepseek_v4_flash",
        TranslationFallbackSnapshot(
            enabled=True,
            model=TranslationModel.DEEPSEEK_V4_FLASH,
            connection=TranslationConnection.OPENROUTER,
        ),
        "settings.fallback.openrouter_deepseek_v4_flash",
    ),
    (
        "openrouter_gemma4_26b_31b",
        TranslationFallbackSnapshot(
            enabled=True,
            model=TranslationModel.GEMMA4_26B_31B,
            connection=TranslationConnection.OPENROUTER,
        ),
        "settings.fallback.openrouter_gemma4_26b_31b",
    ),
    (
        "openrouter_gemma4_31b",
        TranslationFallbackSnapshot(
            enabled=True,
            model=TranslationModel.GEMMA4_31B,
            connection=TranslationConnection.OPENROUTER,
        ),
        "settings.fallback.openrouter_gemma4_31b",
    ),
    (
        "openrouter_gemma4_26b_a4b",
        TranslationFallbackSnapshot(
            enabled=True,
            model=TranslationModel.GEMMA4,
            connection=TranslationConnection.OPENROUTER,
        ),
        "settings.fallback.openrouter_gemma4_26b_a4b",
    ),
    (
        "cerebras_gemma4_31b",
        TranslationFallbackSnapshot(
            enabled=True,
            model=TranslationModel.GEMMA4_31B,
            connection=TranslationConnection.CEREBRAS,
        ),
        "settings.fallback.cerebras_gemma4_31b",
    ),
)
_TRANSLATION_FALLBACK_PRESET_BY_VALUE = {
    value: fallback for value, fallback, _label_key in _TRANSLATION_FALLBACK_PRESETS
}
_TRANSLATION_FALLBACK_LABEL_KEY_BY_VALUE = {
    value: label_key for value, _fallback, label_key in _TRANSLATION_FALLBACK_PRESETS
}
_TRANSLATION_FALLBACK_DESCRIPTION_KEY_BY_VALUE = {
    "openrouter_gemma4_26b_31b": "settings.fallback.openrouter_gemma4_26b_31b.description",
    "openrouter_gemma4_31b": "settings.fallback.openrouter_gemma4_31b.description",
    "cerebras_gemma4_31b": "settings.fallback.cerebras_gemma4_31b.description",
}


def _make_text_button(label: str, **kwargs) -> ft.TextButton:
    return ft.TextButton(content=label, **kwargs)


def _settings_secondary_text_button_style() -> ft.ButtonStyle:
    return ft.ButtonStyle(
        color={
            ft.ControlState.HOVERED: COLOR_PRIMARY,
            ft.ControlState.DEFAULT: COLOR_SECONDARY,
        },
        icon_color={
            ft.ControlState.HOVERED: COLOR_PRIMARY,
            ft.ControlState.DEFAULT: COLOR_SECONDARY,
        },
        text_style=ft.TextStyle(
            size=20,
            font_family=font_for_language(get_locale()),
        ),
        overlay_color=ft.Colors.TRANSPARENT,
        animation_duration=0,
    )


def _set_text_button_label(button: ft.TextButton, label: str) -> None:
    button.content = label


def _reject_json_constant(value: str) -> None:
    raise json.JSONDecodeError(f"invalid JSON constant: {value}", value, 0)


def _custom_stt_extra_to_text(extra: Mapping[str, object]) -> str:
    if not extra:
        return "{}"
    return json.dumps(extra, ensure_ascii=False, indent=2)


def _update_control_if_mounted(control: ft.Control) -> None:
    update_control_if_mounted(control)


def _make_overlay_anchor_dropdown(value: str, on_change) -> ft.Dropdown:
    return ft.Dropdown(
        value=value,
        options=[
            ft.dropdown.Option(
                key=anchor,
                text=t(f"settings.overlay.calibration.anchor.{anchor}"),
            )
            for anchor in OVERLAY_CALIBRATION_ANCHORS
        ],
        text_size=14,
        border_radius=10,
        border_color=COLOR_DIVIDER,
        focused_border_color=COLOR_PRIMARY,
        on_select=on_change,
    )


def _weighted_len(text: str) -> int:
    return sum(2 if ord(char) >= _CJK_START else 1 for char in text)


def _setting_action_text_size(text: str) -> int:
    length = _weighted_len(text or "")
    if length <= 6:
        return 22
    if length <= 10:
        return 20
    if length <= 18:
        return 18
    return 16


def _derive_openrouter_selection_alias(
    llm_model: OpenRouterLLMModel,
    selected_source: OpenRouterCredentialSource,
) -> OpenRouterSelectionAlias:
    if llm_model == OpenRouterLLMModel.QWEN_35_FLASH_02_23:
        if selected_source == OpenRouterCredentialSource.MANAGED:
            return OpenRouterSelectionAlias.QWEN35_FLASH_MANAGED
        return OpenRouterSelectionAlias.QWEN35_FLASH_BYOK
    if llm_model == OpenRouterLLMModel.DEEPSEEK_V4_FLASH:
        if selected_source == OpenRouterCredentialSource.MANAGED:
            return OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED
        return OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_BYOK
    if selected_source == OpenRouterCredentialSource.MANAGED:
        return OpenRouterSelectionAlias.GEMMA4_MANAGED
    return OpenRouterSelectionAlias.GEMMA4_BYOK


class SettingsView(ft.Column):
    """Settings view with Bento grid layout."""

    def __init__(
        self,
        http_extension_registry: HttpExtensionRegistryService | None = None,
    ):
        super().__init__(expand=True, spacing=16)

        # Callbacks (assigned by App)
        self.on_settings_changed: Callable[[ImmediateSettingsIntent], None] | None = None
        self.on_prompt_apply_settings: Callable[[PromptApplyIntent], None] | None = None
        self.on_providers_changed: Callable[[], None] | None = None
        self.on_local_llm_secret_changed: Callable[[], None] | None = None
        self.on_custom_stt_secret_changed: Callable[[], None] | None = None
        self.on_request_openrouter_pkce: Callable[[OpenRouterPkceTarget], None] | None = None
        self.on_verify_api_key: Callable[[str, str], object] | None = None
        self.on_provider_secret_change: Callable[[str, str], object] | None = None
        self.on_secret_cleared: Callable[[str], None] | None = None  # key name
        self.on_overlay_calibration_begin: Callable[[], OverlayCalibration] | None = None
        self.on_overlay_calibration_change: Callable[[str, object], OverlayCalibration] | None = (
            None
        )
        self.on_overlay_calibration_apply: Callable[[], OverlayCalibration] | None = None
        self.on_overlay_calibration_cancel: Callable[[], OverlayCalibration] | None = None
        self.on_desktop_overlay_lock_change: Callable[[bool], None] | None = None
        self.on_desktop_overlay_size_change: Callable[[str], None] | None = None
        self.on_desktop_overlay_recovery_action: Callable[[str], None] | None = None
        self.on_desktop_overlay_position_reset: Callable[[], None] | None = None
        self.on_view_logs: Callable[[], None] | None = None
        self.on_start_microphone_test: Callable[[], None] | None = None
        self.on_gpu_discovery_requested: Callable[[], object] | None = None
        self.on_telemetry_enabled_change: Callable[[bool], None] | None = None
        self.on_list_loopback_capture_options: Callable[[], object] | None = None
        self.on_list_loopback_process_options: Callable[[], object] | None = None
        self.on_list_loopback_device_options: Callable[[], object] | None = None
        self.on_current_loopback_capture_option: Callable[[], str] | None = None
        self.on_apply_loopback_capture_option: Callable[[str], None] | None = None
        self.on_loopback_capture_summary: Callable[[], str] | None = None
        self.on_osc_effective_ports: Callable[[], tuple[int | None, int | None]] | None = None
        self.show_snackbar: Callable[[str, str], None] | None = None
        self.runtime_log_basic: Callable[..., None] | None = None
        self.runtime_log_detailed: Callable[..., None] | None = None
        self._settings_secrets: SettingsSecretsPort | None = None

        self._http_extensions = (
            http_extension_registry
            if http_extension_registry is not None
            else HttpExtensionRegistryService.from_default_directory()
        )
        self._http_extension_secret_fields: dict[str, ft.TextField] = {}
        self._http_extension_secret_dirty: set[str] = set()
        self._http_extension_selected_id: str | None = None
        self._http_extension_snapshot = self._http_extensions.snapshot
        self._http_extension_runtime_reload_pending = False

        # State
        self._provider_snapshot: ProviderSettingsSnapshot | None = None
        self._provider_draft: ProviderSettingsSnapshot | None = None
        self._provider_edits: dict[type, ProviderSettingsEdit] = {}
        self._general_snapshot: GeneralSettingsSnapshot | None = None
        self._prompt_snapshot: PromptSettingsSnapshot | None = None
        self._overlay_snapshot: OverlaySettingsSnapshot | None = None
        self._config_path: Path | None = None
        self.has_provider_changes: bool = False
        self.has_pending_prompt_changes: bool = False
        self._overlay_state: str = "off"
        self._overlay_failure_reason: str | None = None
        self._overlay_runtime_target: str = OVERLAY_TARGET_STEAMVR
        self._desktop_overlay_captions_locked = False
        self._desktop_overlay_pending_locked: bool | None = None
        self._desktop_overlay_primary_action_kind: str | None = None
        self._desktop_overlay_pending_size_preset: str | None = None
        self._desktop_overlay_pending_position_reset = False
        self._overlay_calibration = OverlayCalibration()
        self._overlay_calibration_draft = self._overlay_calibration.copy()
        self._overlay_calibration_session_active = False
        self._managed_trial_usage_visible = False
        self._managed_trial_usage_remaining_percent: int | None = None
        self._managed_key_referral_id: str | None = None
        self._managed_key_pass_status: TalkTogetherPassStatus | None = None
        self._overlay_peer_contract: OverlayPeerConsumerContract | None = None
        self._gpu_devices: tuple[GpuDeviceOption, ...] = ()
        self._llm_gpu_devices: tuple[GpuDeviceOption, ...] = ()
        self._local_cpu_auto_available = False

        # Build UI components
        self._build_ui()

    def set_local_cpu_auto_available(self, available: bool) -> None:
        self._local_cpu_auto_available = bool(available)

    def self_stt_control(self) -> ft.Control:
        return ft.Column(
            [
                self._self_stt_card,
                self._stt_rolling_switch,
            ],
            spacing=8,
            tight=True,
        )

    def stt_rolling_control(self) -> ft.Control:
        return self._stt_rolling_switch

    def peer_stt_control(self) -> ft.Control:
        return self._peer_stt_card

    def translation_provider_control(self) -> ft.Control:
        return self._translation_provider_card

    def translation_connection_control(self) -> ft.Control:
        return self._translation_connection_card

    def http_extension_control(self) -> ft.Control:
        return self._http_extension_host

    def set_http_extension_registry(
        self,
        registry: HttpExtensionRegistryService | None,
    ) -> None:
        if registry is None:
            return
        self._http_extensions = registry
        self._http_extension_snapshot = registry.snapshot
        self._sync_http_extension_card(force_credentials=True)

    def translation_fallback_control(self) -> ft.Control:
        return self._openrouter_fallback_card

    def gpu_device_control(self) -> ft.Control:
        return self._gpu_device_card

    def gpu_llm_control(self) -> ft.Control:
        return self._gpu_llm_card

    def gpu_refresh_control(self) -> ft.Control:
        return self._gpu_refresh_card

    def local_llm_connection_control(self) -> ft.Control:
        return self._local_llm_connection_card

    def custom_stt_connection_control(self) -> ft.Control:
        return self._custom_stt_connection_card

    def managed_key_control(self) -> ft.Control:
        return self._managed_key_card

    def peer_expected_language_control(self) -> ft.Control:
        return self._peer_auto_languages_card

    def api_keys_control(self) -> ft.Control:
        return self._api_keys_card

    def bind_settings_intents(
        self,
        *,
        surface: SettingsSurfaceIntents,
        provider: SettingsProviderIntents,
        general: SettingsGeneralIntents,
        prompt: SettingsPromptIntents,
        overlay: SettingsOverlayIntents,
    ) -> None:
        self.on_settings_changed = surface.settings_changed
        self.show_snackbar = surface.show_snackbar
        if surface.runtime_log_basic is not None:
            self.runtime_log_basic = surface.runtime_log_basic
        if surface.runtime_log_detailed is not None:
            self.runtime_log_detailed = surface.runtime_log_detailed
        self.on_providers_changed = provider.providers_changed
        self.on_request_openrouter_pkce = provider.request_openrouter_pkce
        self.on_verify_api_key = provider.verify_api_key
        self.on_provider_secret_change = provider.provider_secret_change
        self.on_secret_cleared = provider.secret_cleared
        self.on_local_llm_secret_changed = provider.local_llm_secret_changed
        self.on_custom_stt_secret_changed = provider.custom_stt_secret_changed
        self.on_gpu_discovery_requested = provider.gpu_discovery_requested
        self._settings_secrets = provider.settings_secrets
        self.on_start_microphone_test = general.start_microphone_test
        self.on_telemetry_enabled_change = general.telemetry_enabled_change
        self.on_list_loopback_capture_options = general.list_loopback_capture_options
        self.on_list_loopback_process_options = general.list_loopback_process_options
        self.on_list_loopback_device_options = general.list_loopback_device_options
        self.on_current_loopback_capture_option = general.current_loopback_capture_option
        self.on_apply_loopback_capture_option = general.apply_loopback_capture_option
        self.on_loopback_capture_summary = general.loopback_capture_summary
        self.on_osc_effective_ports = general.osc_effective_ports
        self.on_prompt_apply_settings = prompt.prompt_apply_settings
        self.on_desktop_overlay_lock_change = overlay.desktop_overlay_lock_change
        self.on_desktop_overlay_size_change = overlay.desktop_overlay_size_change
        self.on_desktop_overlay_recovery_action = overlay.desktop_overlay_recovery_action
        self.on_desktop_overlay_position_reset = overlay.desktop_overlay_position_reset
        self.on_view_logs = overlay.view_logs
        if overlay.calibration_begin is not None:
            self.on_overlay_calibration_begin = overlay.calibration_begin
        if overlay.calibration_change is not None:
            self.on_overlay_calibration_change = overlay.calibration_change
        if overlay.calibration_apply is not None:
            self.on_overlay_calibration_apply = overlay.calibration_apply
        if overlay.calibration_cancel is not None:
            self.on_overlay_calibration_cancel = overlay.calibration_cancel

    # --- Card Wrapper (About page pattern) ---
    def _wrap_card(
        self,
        content: ft.Control,
        *,
        expand: bool | None = None,
        height: float | int | None = SharedCardWrapper.DEFAULT_HEIGHT,
    ) -> SharedCardWrapper:
        """Wrap content in the shared card shell used across settings/about."""
        return SharedCardWrapper(
            content,
            expand=expand,
            height=height,
        )

    def _wrap_unit_card(
        self,
        *,
        title: ft.Control,
        value: ft.Control,
        extra_controls: tuple[ft.Control, ...] = (),
        height: float | int | None = SettingsUnitCard.DEFAULT_HEIGHT,
    ) -> SettingsUnitCard:
        return SettingsUnitCard(
            title=title,
            value=value,
            extra_controls=extra_controls,
            height=height,
        )

    def _wrap_empty_unit_card(
        self,
        *,
        height: float | int | None = SettingsUnitCard.DEFAULT_HEIGHT,
    ) -> SharedCardWrapper:
        card = self._wrap_card(ft.Container(expand=True), expand=True, height=height)
        card.ignore_interactions = True
        return card

    # --- Clickable Text Builders ---
    def _build_clickable_text(
        self,
        text: str,
        on_click,
        *,
        size: int = 28,
        text_align: ft.TextAlign = ft.TextAlign.CENTER,
        alignment=_CENTER_ALIGNMENT,
        no_wrap: bool = False,
        max_lines: int | None = None,
        overflow: ft.TextOverflow | None = None,
        width: float | int | None = None,
        height: float | int | None = None,
        expand: bool | int | None = True,
    ) -> ft.Container:
        """Build a clickable centered text with hover effect."""
        text_control = ft.Text(
            text,
            size=size,
            font_family=font_for_language(get_locale()),
            color=COLOR_ON_BACKGROUND,
            text_align=text_align,
            no_wrap=no_wrap,
            max_lines=max_lines,
            overflow=overflow,
        )
        return ft.Container(
            content=text_control,
            alignment=alignment,
            width=width,
            height=height,
            expand=expand,
            on_click=on_click,
            on_hover=self._on_text_hover,
        )

    def _build_setting_action_text(self, text: str, on_click) -> ft.Container:
        return self._build_clickable_text(
            text,
            on_click,
            size=_setting_action_text_size(text),
            text_align=ft.TextAlign.RIGHT,
            alignment=_CENTER_RIGHT_ALIGNMENT,
            no_wrap=True,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
        )

    def _set_setting_action_text(self, control: ft.Container, text: str) -> None:
        text_control = control.content
        text_control.value = text
        text_control.size = _setting_action_text_size(text)

    def _set_unit_card_value_text(
        self, control: ft.Container, text: str, *, size: int = 28
    ) -> None:
        text_control = control.content
        text_control.value = text
        text_control.size = size

    def _iter_locale_sensitive_clickable_text_controls(self) -> tuple[ft.Container, ...]:
        return (
            self._stt_text,
            self._peer_stt_text,
            self._gpu_device_text,
            self._gpu_llm_text,
            self._llm_text,
            self._ui_text,
            self._chatbox_source_text,
            self._osc_connection_text,
            self._clipboard_auto_translate_text,
            self._microphone_test_text,
            self._vrc_mic_text,
            self._mic_audio_text,
            self._audio_host_api_text,
            self._loopback_audio_text,
            self._overlay_translation_button,
            self._overlay_peer_original_button,
            self._overlay_target_button,
            self._overlay_anchor_button,
            self._overlay_text_scale_text,
            self._desktop_overlay_size_button,
            self._desktop_overlay_lock_button,
            self._desktop_overlay_swap_caption_languages_button,
            self._overlay_vr_reset_button,
            self._overlay_desktop_reset_button,
            self._desktop_overlay_primary_action,
            self._desktop_overlay_view_logs_action,
            self._translation_connection_text,
            self._openrouter_fallback_text,
            self._telemetry_enabled_text,
            self._http_extension_text,
            self._http_extension_path_text,
        )

    def _sync_clickable_text_control_fonts(self, font_family: str | None) -> None:
        for control in self._iter_locale_sensitive_clickable_text_controls():
            if control:
                control.content.font_family = font_family

    def _sync_general_audio_card_texts(self) -> None:
        default_label = t("settings.default_option")
        self._set_unit_card_value_text(
            self._mic_audio_text,
            self._audio_settings.microphone or default_label,
        )
        self._set_unit_card_value_text(
            self._audio_host_api_text,
            self._audio_settings.host_api_display_label,
        )
        loopback_summary = (
            self.on_loopback_capture_summary()
            if callable(getattr(self, "on_loopback_capture_summary", None))
            else (self._audio_settings.desktop_output_device or default_label)
        )
        self._set_unit_card_value_text(
            self._loopback_audio_text,
            loopback_summary or default_label,
        )

    def _sync_osc_connection_card(self, settings: GeneralSettingsSnapshot) -> None:
        mode = settings.osc_connection_mode
        if mode not in {"automatic", "manual", "off"}:
            mode = "automatic"
        self._set_unit_card_value_text(
            self._osc_connection_text,
            t(f"settings.osc.mode.{mode}"),
        )

    def refresh_loopback_capture_target(self, settings: GeneralSettingsSnapshot) -> None:
        self._general_snapshot = settings
        self._audio_settings.desktop_output_device = settings.output_device
        summary = (
            self.on_loopback_capture_summary()
            if callable(getattr(self, "on_loopback_capture_summary", None))
            else (self._audio_settings.desktop_output_device or t("settings.default_option"))
        )
        self._set_unit_card_value_text(
            self._loopback_audio_text,
            summary or t("settings.default_option"),
        )
        if is_control_mounted(self._loopback_audio_text):
            self._loopback_audio_text.update()

    def _on_text_hover(self, e: ft.ControlEvent) -> None:
        """Handle hover effect on clickable text."""
        container = e.control
        text_control = container.content
        next_color = COLOR_PRIMARY if is_hover_active(e) else COLOR_ON_BACKGROUND
        if text_control.color == next_color:
            return
        text_control.color = next_color
        container.update()

    def _make_overlay_step_hover_handler(self, text_control: ft.Text):
        def _on_hover(e: ft.ControlEvent) -> None:
            next_color = COLOR_PRIMARY if is_hover_active(e) else COLOR_ON_BACKGROUND
            if text_control.color == next_color:
                return
            text_control.color = next_color
            if is_control_mounted(text_control):
                text_control.update()

        return _on_hover

    def _build_overlay_step_hit_lane(self, on_click, *, on_hover=None) -> ft.Container:
        return ft.Container(
            content=ft.Container(expand=True),
            expand=1,
            on_click=on_click,
            on_hover=on_hover,
        )

    def _build_overlay_step_visual_lane(
        self, text: str, *, alignment
    ) -> tuple[ft.Container, ft.Text]:
        text_control = ft.Text(
            text,
            size=22,
            font_family=font_for_language(get_locale()),
            color=COLOR_ON_BACKGROUND,
            text_align=ft.TextAlign.CENTER,
        )
        return (
            ft.Container(
                content=text_control,
                expand=1,
                alignment=alignment,
            ),
            text_control,
        )

    def _build_overlay_step_split_layout(
        self,
        *,
        title: ft.Text,
        value_text: ft.Text,
        decrease_text: str,
        increase_text: str,
        on_decrease,
        on_increase,
    ) -> tuple[ft.Stack, ft.Container, ft.Container, ft.Text, ft.Text]:
        decrease_visual, decrease_glyph = self._build_overlay_step_visual_lane(
            decrease_text,
            alignment=ft.Alignment.CENTER_RIGHT,
        )
        increase_visual, increase_glyph = self._build_overlay_step_visual_lane(
            increase_text,
            alignment=ft.Alignment.CENTER_LEFT,
        )
        decrease_lane = self._build_overlay_step_hit_lane(
            on_decrease,
            on_hover=self._make_overlay_step_hover_handler(decrease_glyph),
        )
        increase_lane = self._build_overlay_step_hit_lane(
            on_increase,
            on_hover=self._make_overlay_step_hover_handler(increase_glyph),
        )
        visual_row = ft.Row(
            controls=[
                decrease_visual,
                ft.Container(
                    content=value_text,
                    width=84,
                    alignment=ft.Alignment.CENTER,
                ),
                increase_visual,
            ],
            spacing=4,
            expand=1,
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        visual_column = ft.Column(
            controls=[
                title,
                ft.Container(
                    content=visual_row,
                    expand=True,
                    alignment=ft.Alignment.CENTER,
                ),
            ],
            spacing=0,
            expand=True,
        )
        stack = ft.Stack(
            controls=[
                ft.Row(
                    controls=[decrease_lane, increase_lane],
                    spacing=0,
                    expand=1,
                    vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                ),
                ft.TransparentPointer(content=visual_column),
            ],
            fit=ft.StackFit.EXPAND,
            expand=True,
            alignment=ft.Alignment.CENTER,
        )
        return stack, decrease_lane, increase_lane, decrease_glyph, increase_glyph

    def _get_button_style(
        self,
        font_family: str,
        *,
        size: int = 20,
        default_color: str = COLOR_SECONDARY,
        disabled_color: str | None = None,
    ) -> ft.ButtonStyle:
        """Create a complete ButtonStyle with the specified font."""
        color = {
            ft.ControlState.HOVERED: COLOR_PRIMARY,
            ft.ControlState.DEFAULT: default_color,
        }
        if disabled_color is not None:
            color[ft.ControlState.DISABLED] = disabled_color
        return ft.ButtonStyle(
            color=color,
            icon_color=color,
            text_style=ft.TextStyle(
                size=size,
                font_family=font_family,
            ),
            overlay_color=ft.Colors.TRANSPARENT,
            animation_duration=0,
        )

    def _settings_subtab_label(self, key: str) -> str:
        return t(f"settings.subtab.{key}")

    def _build_settings_subtab_shell(
        self, tab_rows: dict[str, list[ft.Control]]
    ) -> TextSubtabShell:
        return TextSubtabShell(
            tabs=[
                TextSubtab(key, self._settings_subtab_label(key), tuple(tab_rows[key]))
                for key in _SETTINGS_SUBTAB_ORDER
            ],
            font_family=font_for_language(get_locale()),
            initial_key=_SETTINGS_SUBTAB_ORDER[0],
            subtab_bar_position="bottom",
        )

    def _build_setting_action_row(self, label: ft.Text, action: ft.Control) -> ft.Row:
        return ft.Row(
            controls=[label, ft.Container(expand=True), action],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

    def _emit_runtime_basic(self, message: str, *, level: int = logging.INFO) -> None:
        runtime_log_basic = getattr(self, "runtime_log_basic", None)
        if runtime_log_basic is not None:
            runtime_log_basic(message, level=level)
            return
        logger.log(level, message)

    def _emit_runtime_detailed(self, message: str, *, level: int = logging.INFO) -> None:
        runtime_log_detailed = getattr(self, "runtime_log_detailed", None)
        if runtime_log_detailed is not None:
            runtime_log_detailed(message, level=level)
            return
        logger.log(level, message)

    def _build_action_button(
        self,
        text: str,
        on_click,
        *,
        size: int = 20,
        default_color: str = COLOR_SECONDARY,
        disabled_color: str | None = None,
        width: float | int | None = None,
        height: float | int | None = None,
    ) -> ft.TextButton:
        return _make_text_button(
            text,
            style=self._get_button_style(
                font_for_language(get_locale()),
                size=size,
                default_color=default_color,
                disabled_color=disabled_color,
            ),
            on_click=on_click,
            width=width,
            height=height,
        )

    def _build_overlay_calibration_field(
        self,
        *,
        value: float,
        on_blur,
    ) -> ft.TextField:
        return ft.TextField(
            value=self._format_overlay_calibration_number(value),
            text_size=14,
            width=120,
            border_radius=10,
            border_color=COLOR_DIVIDER,
            focused_border_color=COLOR_PRIMARY,
            on_blur=on_blur,
        )

    def _build_numeric_setting_field(
        self,
        *,
        label: str,
        value: str,
        on_change_end,
    ) -> ft.TextField:
        return ft.TextField(
            label=label,
            value=value,
            dense=True,
            expand=True,
            text_align=ft.TextAlign.CENTER,
            border_radius=10,
            border_color=COLOR_DIVIDER,
            focused_border_color=COLOR_PRIMARY,
            on_blur=on_change_end,
            on_submit=on_change_end,
        )

    def _build_overlay_calibration_column(
        self,
        *,
        label: ft.Text,
        control: ft.Control,
    ) -> ft.Column:
        return ft.Column(
            controls=[label, control],
            spacing=6,
            expand=True,
        )

    def _format_overlay_calibration_number(self, value: float) -> str:
        return f"{value:.2f}"

    def _overlay_anchor_label_for(self, anchor: str) -> str:
        return t(f"settings.overlay.calibration.anchor.{anchor}")

    def _overlay_text_scale_label_for(self, value: float) -> str:
        return t(
            f"settings.overlay.calibration.text_scale.{self._overlay_text_scale_preset_key_for(value)}"
        )

    def _overlay_text_scale_preset_key_for(self, value: float) -> str:
        return min(
            _OVERLAY_TEXT_SCALE_PRESETS,
            key=lambda preset: abs(preset[1] - value),
        )[0]

    def _overlay_text_scale_value_for(self, preset_key: str) -> float:
        for key, scale in _OVERLAY_TEXT_SCALE_PRESETS:
            if key == preset_key:
                return scale
        try:
            return float(preset_key)
        except (TypeError, ValueError):
            return 1.0

    def _parse_setting_float(
        self,
        raw_value: str,
        *,
        fallback: float,
        minimum: float,
        maximum: float | None = None,
    ) -> float:
        try:
            parsed = float(raw_value)
        except (TypeError, ValueError):
            parsed = fallback
        if parsed < minimum:
            parsed = minimum
        if maximum is not None and parsed > maximum:
            parsed = maximum
        return parsed

    def _parse_setting_int(
        self,
        raw_value: str,
        *,
        fallback: int,
        minimum: int,
    ) -> int:
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError):
            parsed = fallback
        return max(minimum, parsed)

    def _build_ui(self) -> None:
        """Build the settings UI with Bento grid layout."""
        # === API provider surfaces: Self STT + Peer STT + Shared Translation ===
        self._stt_text = self._build_clickable_text(
            provider_label(STTProviderName.LOCAL_CPU_AUTO.value),
            self._on_stt_click,
        )
        self._stt_title = ft.Text(
            t("settings.section.stt"), size=24, weight=ft.FontWeight.BOLD, color=COLOR_SECONDARY
        )
        self._stt_provider_label = ft.Text(
            t("settings.self_stt_provider"), size=16, color=COLOR_ON_BACKGROUND
        )
        self._self_stt_card = self._wrap_unit_card(
            title=self._stt_title,
            value=self._stt_text,
        )
        self._stt_rolling_switch = ft.Switch(
            label=t("settings.stt_rolling"),
            value=False,
            active_color=COLOR_PRIMARY,
            on_change=self._on_stt_rolling_toggle,
        )

        self._llm_text = self._build_clickable_text(
            t("provider.gemini37_flash"),
            self._on_llm_click,
        )
        self._trans_title = ft.Text(
            t("settings.section.translation"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._translation_provider_label = ft.Text(
            t("settings.shared_translation_provider"), size=16, color=COLOR_ON_BACKGROUND
        )
        self._translation_provider_card = self._wrap_unit_card(
            title=self._trans_title,
            value=self._llm_text,
        )

        # === Row 2: API Keys (2x1) ===
        # Qwen region selection button (in header)
        self._qwen_region_btn = _make_text_button(
            f"{t('settings.qwen_region')} {t('region.beijing')}",
            style=ft.ButtonStyle(
                color={
                    ft.ControlState.HOVERED: COLOR_PRIMARY,
                    ft.ControlState.DEFAULT: COLOR_SECONDARY,
                },
                text_style=ft.TextStyle(
                    size=20,
                    font_family=font_for_language(get_locale()),
                ),
                overlay_color=ft.Colors.TRANSPARENT,
                animation_duration=0,
            ),
            on_click=self._on_qwen_region_click,
            visible=False,  # Hidden by default, updated by visibility logic
        )

        # API Key fields
        self._deepgram_key = ApiKeyField(
            "settings.deepgram_api_key",
            "deepgram_api_key",
            "deepgram",
            on_verify=self._verify_key,
            on_save=self._on_secret_change,
            show_snackbar=lambda msg, bg: (
                self.show_snackbar(msg, bg) if self.show_snackbar else None
            ),
        )
        self._gemini_transcribe_key = ApiKeyField(
            "settings.gemini_transcribe_api_key",
            "gemini_transcribe_api_key",
            "gemini_transcribe",
            on_verify=self._verify_key,
            on_save=self._on_secret_change,
            show_snackbar=lambda msg, bg: (
                self.show_snackbar(msg, bg) if self.show_snackbar else None
            ),
        )
        self._elevenlabs_scribe_key = ApiKeyField(
            "settings.elevenlabs_scribe_api_key",
            "elevenlabs_scribe_api_key",
            "elevenlabs_scribe",
            on_verify=self._verify_key,
            on_save=self._on_secret_change,
            show_snackbar=lambda msg, bg: (
                self.show_snackbar(msg, bg) if self.show_snackbar else None
            ),
        )
        self._soniox_key = ApiKeyField(
            "settings.soniox_api_key",
            "soniox_api_key",
            "soniox",
            on_verify=self._verify_key,
            on_save=self._on_secret_change,
            show_snackbar=lambda msg, bg: (
                self.show_snackbar(msg, bg) if self.show_snackbar else None
            ),
        )
        self._peer_auto_languages_title = ft.Text(
            t("settings.peer_auto_languages.title"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._peer_auto_languages_editor = LanguageHintEditor(
            on_add=self._on_peer_auto_languages_add,
            on_remove=self._on_peer_auto_languages_remove,
        )
        self._peer_auto_languages_card = self._wrap_card(
            ft.Column(
                [
                    self._peer_auto_languages_title,
                    self._peer_auto_languages_editor,
                ],
                spacing=12,
            ),
            height=None,
        )
        self._peer_auto_languages_card.visible = False
        self._google_key = ApiKeyField(
            "settings.google_api_key",
            "google_api_key",
            "google",
            on_verify=self._verify_key,
            on_save=self._on_secret_change,
            show_snackbar=lambda msg, bg: (
                self.show_snackbar(msg, bg) if self.show_snackbar else None
            ),
        )
        self._openrouter_key = ApiKeyField(
            "settings.openrouter_api_key",
            "openrouter_api_key",
            "openrouter",
            on_verify=self._verify_key,
            on_save=self._on_secret_change,
            show_snackbar=lambda msg, bg: (
                self.show_snackbar(msg, bg) if self.show_snackbar else None
            ),
        )
        self._deepseek_key = ApiKeyField(
            "settings.deepseek_api_key",
            "deepseek_api_key",
            "deepseek",
            on_verify=self._verify_key,
            on_save=self._on_secret_change,
            show_snackbar=lambda msg, bg: (
                self.show_snackbar(msg, bg) if self.show_snackbar else None
            ),
        )
        self._cerebras_key = ApiKeyField(
            "settings.cerebras_api_key",
            "cerebras_api_key",
            "cerebras",
            on_verify=self._verify_key,
            on_save=self._on_secret_change,
            show_snackbar=lambda msg, bg: (
                self.show_snackbar(msg, bg) if self.show_snackbar else None
            ),
        )
        self._openrouter_pkce_button = self._build_action_button(
            t("settings.openrouter_authenticate"),
            self._on_openrouter_pkce_click,
            size=20,
            default_color=COLOR_NEUTRAL_DARK,
            disabled_color=COLOR_NEUTRAL_DARK,
        )
        self._openrouter_pkce_button.disabled = False
        self._openrouter_pkce_button_row = ft.Row(
            controls=[self._openrouter_pkce_button],
            alignment=ft.MainAxisAlignment.END,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        self._managed_trial_usage_bar = ManagedTrialUsageBar()
        self._managed_key_title = ft.Text(
            t("settings.managed_key.title"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._managed_key_referral_id_label = ft.Text(
            t("settings.managed_key.referral_id.label"),
            size=16,
            weight=ft.FontWeight.BOLD,
            color=COLOR_ON_BACKGROUND,
        )
        self._managed_key_referral_id_value = ft.Text(
            t("settings.managed_key.referral_id.empty"),
            size=22,
            weight=ft.FontWeight.BOLD,
            color=COLOR_ON_BACKGROUND,
            selectable=True,
        )
        self._managed_key_invite_progress_label = ft.Text(
            t("settings.managed_key.invite_progress.label"),
            size=16,
            weight=ft.FontWeight.BOLD,
            color=COLOR_ON_BACKGROUND,
        )
        self._managed_key_invite_progress_value = ft.Text(
            "",
            size=20,
            weight=ft.FontWeight.BOLD,
            color=COLOR_ON_BACKGROUND,
        )
        self._managed_key_invite_progress_row = ft.Row(
            [
                self._managed_key_invite_progress_label,
                ft.Container(expand=True),
                self._managed_key_invite_progress_value,
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            visible=False,
        )
        self._managed_key_card = self._wrap_card(
            ft.Column(
                [
                    self._managed_key_title,
                    ft.Container(height=4),
                    self._managed_trial_usage_bar,
                    ft.Container(height=8),
                    ft.Column(
                        [
                            ft.Row(
                                [
                                    self._managed_key_referral_id_label,
                                    ft.Container(expand=True),
                                    self._managed_key_referral_id_value,
                                ],
                                spacing=8,
                                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                            self._managed_key_invite_progress_row,
                        ],
                        spacing=4,
                    ),
                ],
                spacing=0,
            ),
            height=None,
            expand=False,
        )
        self._managed_key_card.visible = False
        self._alibaba_key_beijing = ApiKeyField(
            "settings.alibaba_api_key_beijing",
            "alibaba_api_key_beijing",
            "alibaba_beijing",
            on_verify=self._verify_key,
            on_save=self._on_secret_change,
            show_snackbar=lambda msg, bg: (
                self.show_snackbar(msg, bg) if self.show_snackbar else None
            ),
        )
        self._alibaba_key_singapore = ApiKeyField(
            "settings.alibaba_api_key_singapore",
            "alibaba_api_key_singapore",
            "alibaba_singapore",
            on_verify=self._verify_key,
            on_save=self._on_secret_change,
            show_snackbar=lambda msg, bg: (
                self.show_snackbar(msg, bg) if self.show_snackbar else None
            ),
        )

        self._api_keys_column = ft.Column(
            [
                # self._qwen_region_row removed
                self._deepgram_key,
                self._gemini_transcribe_key,
                self._elevenlabs_scribe_key,
                self._soniox_key,
                self._google_key,
                self._deepseek_key,
                self._cerebras_key,
                self._alibaba_key_beijing,
                self._alibaba_key_singapore,
                self._openrouter_key,
                self._openrouter_pkce_button_row,
            ],
            spacing=12,
        )

        self._api_title = ft.Text(
            t("settings.section.api_keys"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._api_credentials_helper_text = ft.Text(
            t("settings.api_credentials_helper"),
            size=16,
            color=COLOR_SECONDARY,
        )
        # Header row with title and region button
        api_header = ft.Row(
            controls=[
                self._api_title,
                ft.Container(expand=True),
                self._qwen_region_btn,
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        self._api_keys_card = self._wrap_card(
            ft.Column(
                [
                    api_header,
                    ft.Container(height=16),
                    self._api_keys_column,
                ],
                spacing=0,
            ),
            height=None,
        )

        # === General Tab Row 1 ===
        self._ui_text = self._build_clickable_text(
            locale_label(get_locale()),
            self._on_ui_click,
        )
        self._ui_title = ft.Text(
            t("settings.section.ui"), size=24, weight=ft.FontWeight.BOLD, color=COLOR_SECONDARY
        )
        ui_card = self._wrap_unit_card(
            title=self._ui_title,
            value=self._ui_text,
        )

        self._audio_settings = AudioSettings(on_change=self._on_audio_change)
        self._chatbox_source_text = self._build_clickable_text(
            t("settings.chatbox_source.on"),
            self._on_chatbox_source_click,
        )
        self._chatbox_source_title = ft.Text(
            t("settings.chatbox_include_source"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        chatbox_source_card = self._wrap_unit_card(
            title=self._chatbox_source_title,
            value=self._chatbox_source_text,
        )

        self._osc_connection_text = self._build_clickable_text(
            t("settings.osc.mode.automatic"),
            self._on_osc_connection_click,
        )
        self._osc_connection_title = ft.Text(
            t("settings.osc.connection.title"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._vrchat_osc_card = self._wrap_unit_card(
            title=self._osc_connection_title,
            value=self._osc_connection_text,
        )

        self._clipboard_auto_translate_text = self._build_clickable_text(
            t("settings.clipboard_auto_translate.off"),
            self._on_clipboard_auto_translate_click,
        )
        self._clipboard_auto_translate_title = ft.Text(
            t("settings.clipboard_auto_translate"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        clipboard_auto_translate_card = self._wrap_unit_card(
            title=self._clipboard_auto_translate_title,
            value=self._clipboard_auto_translate_text,
        )

        self._telemetry_enabled_text = self._build_clickable_text(
            t("settings.telemetry.state.off"),
            self._on_telemetry_enabled_click,
        )
        self._telemetry_enabled_title = ft.Text(
            t("settings.telemetry.title"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._telemetry_enabled_card = self._wrap_unit_card(
            title=self._telemetry_enabled_title,
            value=self._telemetry_enabled_text,
        )

        self._vrc_mic_text = self._build_clickable_text(
            t("settings.vrc_mic.on"),
            self._on_vrc_mic_click,
        )
        self._vrc_mic_title = ft.Text(
            t("settings.vrc_mic_intercept"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        vrc_mic_card = self._wrap_unit_card(
            title=self._vrc_mic_title,
            value=self._vrc_mic_text,
        )

        self._microphone_test_text = self._build_clickable_text(
            t("settings.microphone_test.action"),
            self._on_microphone_test_click,
        )
        self._microphone_test_title = ft.Text(
            t("settings.microphone_test"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        microphone_test_card = self._wrap_unit_card(
            title=self._microphone_test_title,
            value=self._microphone_test_text,
        )

        self._general_ui_card = ui_card
        self._general_chatbox_source_card = chatbox_source_card

        # === General Tab Row 2: Host API / Microphone Audio / Loopback Audio ===
        self._mic_audio_text = self._build_clickable_text(
            t("settings.default_option"),
            self._on_mic_audio_click,
        )
        self._audio_host_api_title = ft.Text(
            t("settings.audio_host_api"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._audio_host_api_text = self._build_clickable_text(
            t("settings.default_option"),
            self._on_mic_host_api_click,
        )
        host_api_card = self._wrap_unit_card(
            title=self._audio_host_api_title,
            value=self._audio_host_api_text,
        )
        self._mic_audio_title = ft.Text(
            t("settings.section.microphone_audio"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        mic_audio_card = self._wrap_unit_card(
            title=self._mic_audio_title,
            value=self._mic_audio_text,
        )

        self._loopback_audio_text = self._build_clickable_text(
            t("settings.default_option"),
            self._on_loopback_audio_click,
        )
        self._loopback_audio_title = ft.Text(
            t("settings.section.loopback_audio"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        loopback_audio_card = self._wrap_unit_card(
            title=self._loopback_audio_title,
            value=self._loopback_audio_text,
        )
        self._general_host_api_card = host_api_card
        self._general_mic_audio_card = mic_audio_card
        self._general_loopback_audio_card = loopback_audio_card

        # === General Tab Row 3: VRChat Mute Sync / Self VAD / Peer VAD ===
        self._self_vad_title = ft.Text(
            t("settings.section.self_vad_sensitivity"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._vad_slider = ft.Slider(
            min=0.0,
            max=1.0,
            divisions=20,
            value=0.4,
            label="0.40",
            active_color=COLOR_PRIMARY,
            on_change=self._handle_vad_visual_change,
            on_change_end=self._handle_vad_change,
        )
        self._self_vad_card = self._wrap_unit_card(
            title=self._self_vad_title,
            value=ft.Container(content=self._vad_slider, alignment=_CENTER_ALIGNMENT, expand=True),
        )

        self._peer_vad_title = ft.Text(
            t("settings.section.peer_vad_sensitivity"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._peer_vad_slider = ft.Slider(
            min=0.0,
            max=1.0,
            divisions=20,
            value=0.5,
            label="0.50",
            active_color=COLOR_PRIMARY,
            on_change=self._handle_peer_vad_visual_change,
            on_change_end=self._handle_peer_vad_change,
        )
        self._peer_vad_field = self._build_numeric_setting_field(
            label=t("settings.vad.peer"),
            value="0.50",
            on_change_end=self._on_peer_vad_threshold_change,
        )
        self._peer_hangover_field = self._build_numeric_setting_field(
            label=t("settings.vad.peer_hangover_ms"),
            value="700",
            on_change_end=self._on_peer_hangover_change,
        )
        self._peer_pre_roll_field = self._build_numeric_setting_field(
            label=t("settings.vad.peer_pre_roll_ms"),
            value="500",
            on_change_end=self._on_peer_pre_roll_change,
        )
        self._peer_vad_card = self._wrap_unit_card(
            title=self._peer_vad_title,
            value=ft.Container(
                content=self._peer_vad_slider,
                alignment=_CENTER_ALIGNMENT,
                expand=True,
            ),
        )
        self._general_surface = compose_settings_general_surface(
            SettingsGeneralSurfaceSlots(
                ui=self._general_ui_card,
                chatbox_source=self._general_chatbox_source_card,
                audio_host_api=self._general_host_api_card,
                microphone=self._general_mic_audio_card,
                loopback=self._general_loopback_audio_card,
                microphone_test=microphone_test_card,
                self_vad=self._self_vad_card,
                peer_vad=self._peer_vad_card,
                clipboard_auto_translate=clipboard_auto_translate_card,
                vrchat_mic_intercept=vrc_mic_card,
                telemetry_enabled=self._telemetry_enabled_card,
            ),
            placeholder_factory=lambda: self._vrchat_osc_card,
        )

        # === Peer STT card ===
        self._peer_provider_title = ft.Text(
            t("settings.section.peer_stt"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._dashboard_language_redirect_text = ft.Text(
            t("settings.dashboard_language_redirect"),
            size=16,
            color=COLOR_SECONDARY,
        )
        self._peer_stt_text = self._build_clickable_text(
            provider_label(STTProviderName.LOCAL_CPU_AUTO.value),
            self._on_peer_stt_click,
        )
        self._peer_stt_label = ft.Text(
            t("settings.peer_stt_provider"),
            size=16,
            color=COLOR_ON_BACKGROUND,
        )
        self._peer_stt_card = self._wrap_unit_card(
            title=self._peer_provider_title,
            value=self._peer_stt_text,
        )

        self._gpu_device_title = ft.Text(
            t("settings.gpu_device.asr"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._gpu_device_text = self._build_clickable_text(
            t("settings.gpu_device.auto"),
            self._on_gpu_device_click,
            no_wrap=True,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        self._gpu_device_card = self._wrap_unit_card(
            title=self._gpu_device_title,
            value=self._gpu_device_text,
        )
        self._gpu_llm_title = ft.Text(
            t("settings.gpu_device.llm"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._gpu_llm_text = self._build_clickable_text(
            t("settings.gpu_device.auto"),
            self._on_llm_gpu_device_click,
            no_wrap=True,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        self._gpu_llm_card = self._wrap_unit_card(
            title=self._gpu_llm_title,
            value=self._gpu_llm_text,
        )
        self._gpu_refresh_title = ft.Text(
            t("settings.gpu_device.refresh"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._gpu_refresh_icon = ft.Container(
            content=ft.Icon(
                ft.Icons.REFRESH_ROUNDED,
                size=44,
                color=COLOR_ON_BACKGROUND,
            ),
            alignment=_CENTER_ALIGNMENT,
            expand=True,
            on_click=self._on_gpu_refresh_click,
            on_hover=self._on_text_hover,
        )
        self._gpu_refresh_card = self._wrap_unit_card(
            title=self._gpu_refresh_title,
            value=self._gpu_refresh_icon,
        )

        self._overlay_translation_title = ft.Text(
            t("settings.overlay.show_translation"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._overlay_translation_button = self._build_clickable_text(
            t("settings.option.on"),
            self._on_overlay_translation_click,
        )
        self._overlay_translation_card = self._wrap_unit_card(
            title=self._overlay_translation_title,
            value=self._overlay_translation_button,
        )

        self._overlay_peer_original_title = ft.Text(
            t("settings.overlay.show_peer_original"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._overlay_peer_original_button = self._build_clickable_text(
            t("settings.option.on"),
            self._on_overlay_peer_original_click,
        )
        self._overlay_peer_original_card = self._wrap_unit_card(
            title=self._overlay_peer_original_title,
            value=self._overlay_peer_original_button,
        )

        self._overlay_target_title = ft.Text(
            t("settings.overlay.caption_location"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._overlay_target_button = self._build_clickable_text(
            self._overlay_target_label_for(OVERLAY_TARGET_STEAMVR),
            self._on_overlay_target_click,
            size=28,
            max_lines=2,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        self._overlay_target_card = self._wrap_unit_card(
            title=self._overlay_target_title,
            value=self._overlay_target_button,
        )

        self._overlay_anchor_title = ft.Text(
            t("settings.overlay.calibration.anchor"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._overlay_anchor_button = self._build_clickable_text(
            self._overlay_anchor_label_for(self._overlay_calibration.anchor),
            self._on_overlay_anchor_click,
        )
        self._overlay_anchor_card = self._wrap_unit_card(
            title=self._overlay_anchor_title,
            value=self._overlay_anchor_button,
        )

        self._overlay_distance_title = ft.Text(
            t("settings.overlay.calibration.distance"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._overlay_distance_value_text = ft.Text(
            self._format_overlay_calibration_number(self._overlay_calibration.distance),
            size=28,
            color=COLOR_ON_BACKGROUND,
            text_align=ft.TextAlign.CENTER,
        )
        (
            self._overlay_distance_card_content,
            self._overlay_distance_decrease_button,
            self._overlay_distance_increase_button,
            self._overlay_distance_decrease_glyph,
            self._overlay_distance_increase_glyph,
        ) = self._build_overlay_step_split_layout(
            title=self._overlay_distance_title,
            value_text=self._overlay_distance_value_text,
            decrease_text="－",
            increase_text="＋",
            on_decrease=lambda _e: self._on_overlay_distance_step(-_OVERLAY_OFFSET_STEP),
            on_increase=lambda _e: self._on_overlay_distance_step(_OVERLAY_OFFSET_STEP),
        )
        self._overlay_distance_card = self._wrap_card(
            self._overlay_distance_card_content,
            expand=True,
            height=SettingsUnitCard.DEFAULT_HEIGHT,
        )

        self._overlay_offset_x_title = ft.Text(
            t("settings.overlay.calibration.offset_x"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._overlay_offset_x_value_text = ft.Text(
            self._format_overlay_calibration_number(self._overlay_calibration.offset_x),
            size=28,
            color=COLOR_ON_BACKGROUND,
            text_align=ft.TextAlign.CENTER,
        )
        (
            self._overlay_offset_x_card_content,
            self._overlay_offset_x_decrease_button,
            self._overlay_offset_x_increase_button,
            self._overlay_offset_x_decrease_glyph,
            self._overlay_offset_x_increase_glyph,
        ) = self._build_overlay_step_split_layout(
            title=self._overlay_offset_x_title,
            value_text=self._overlay_offset_x_value_text,
            decrease_text="◀",
            increase_text="▶",
            on_decrease=lambda _e: self._on_overlay_offset_x_step(-_OVERLAY_OFFSET_STEP),
            on_increase=lambda _e: self._on_overlay_offset_x_step(_OVERLAY_OFFSET_STEP),
        )
        self._overlay_offset_x_card = self._wrap_card(
            self._overlay_offset_x_card_content,
            expand=True,
            height=SettingsUnitCard.DEFAULT_HEIGHT,
        )

        self._overlay_offset_y_title = ft.Text(
            t("settings.overlay.calibration.offset_y"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._overlay_offset_y_value_text = ft.Text(
            self._format_overlay_calibration_number(self._overlay_calibration.offset_y),
            size=28,
            color=COLOR_ON_BACKGROUND,
            text_align=ft.TextAlign.CENTER,
        )
        (
            self._overlay_offset_y_card_content,
            self._overlay_offset_y_decrease_button,
            self._overlay_offset_y_increase_button,
            self._overlay_offset_y_decrease_glyph,
            self._overlay_offset_y_increase_glyph,
        ) = self._build_overlay_step_split_layout(
            title=self._overlay_offset_y_title,
            value_text=self._overlay_offset_y_value_text,
            decrease_text="▲",
            increase_text="▼",
            on_decrease=lambda _e: self._on_overlay_offset_y_step(-_OVERLAY_OFFSET_STEP),
            on_increase=lambda _e: self._on_overlay_offset_y_step(_OVERLAY_OFFSET_STEP),
        )
        self._overlay_offset_y_card = self._wrap_card(
            self._overlay_offset_y_card_content,
            expand=True,
            height=SettingsUnitCard.DEFAULT_HEIGHT,
        )

        self._overlay_text_scale_title = ft.Text(
            t("settings.overlay.calibration.text_scale"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._overlay_text_scale_text = self._build_clickable_text(
            self._overlay_text_scale_label_for(self._overlay_calibration.text_scale),
            self._on_overlay_text_scale_click,
        )
        self._overlay_text_scale_card = self._wrap_unit_card(
            title=self._overlay_text_scale_title,
            value=self._overlay_text_scale_text,
        )

        self._overlay_vr_reset_title = ft.Text(
            t("settings.overlay.position_reset.vr.title"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._overlay_vr_reset_button = self._build_clickable_text(
            t("settings.overlay.position_reset.action.vr"),
            self._on_overlay_position_reset,
            height=72,
            expand=False,
        )
        self._overlay_vr_reset_card = self._wrap_unit_card(
            title=self._overlay_vr_reset_title,
            value=self._overlay_vr_reset_button,
        )

        self._overlay_desktop_reset_title = ft.Text(
            t("settings.overlay.position_reset.desktop.title"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._overlay_desktop_reset_button = self._build_clickable_text(
            t("settings.overlay.position_reset.action.desktop"),
            self._on_desktop_overlay_position_reset,
            height=72,
            expand=False,
        )
        self._overlay_desktop_reset_card = self._wrap_unit_card(
            title=self._overlay_desktop_reset_title,
            value=self._overlay_desktop_reset_button,
        )
        self._overlay_reset_title = self._overlay_vr_reset_title

        self._desktop_overlay_size_title = ft.Text(
            t("settings.overlay.desktop.size.title"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._desktop_overlay_size_button = self._build_clickable_text(
            self._desktop_overlay_size_label_for("medium"),
            self._on_desktop_overlay_size_click,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        self._desktop_overlay_size_card = self._wrap_unit_card(
            title=self._desktop_overlay_size_title,
            value=self._desktop_overlay_size_button,
        )

        self._desktop_overlay_background_alpha_title = ft.Text(
            t("settings.overlay.desktop.background_alpha.title"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._desktop_overlay_background_alpha_value_text = ft.Text(
            "40%",
            size=28,
            color=COLOR_ON_BACKGROUND,
            text_align=ft.TextAlign.CENTER,
        )
        (
            self._desktop_overlay_background_alpha_card_content,
            self._desktop_overlay_background_alpha_decrease_button,
            self._desktop_overlay_background_alpha_increase_button,
            self._desktop_overlay_background_alpha_decrease_glyph,
            self._desktop_overlay_background_alpha_increase_glyph,
        ) = self._build_overlay_step_split_layout(
            title=self._desktop_overlay_background_alpha_title,
            value_text=self._desktop_overlay_background_alpha_value_text,
            decrease_text="－",
            increase_text="＋",
            on_decrease=lambda _e: self._on_desktop_overlay_background_alpha_step(
                -_DESKTOP_OVERLAY_BACKGROUND_ALPHA_STEP
            ),
            on_increase=lambda _e: self._on_desktop_overlay_background_alpha_step(
                _DESKTOP_OVERLAY_BACKGROUND_ALPHA_STEP
            ),
        )
        self._desktop_overlay_background_alpha_card = self._wrap_card(
            self._desktop_overlay_background_alpha_card_content,
            expand=True,
            height=SettingsUnitCard.DEFAULT_HEIGHT,
        )

        self._desktop_overlay_lock_title = ft.Text(
            t("settings.overlay.desktop.lock.title"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._desktop_overlay_lock_button = self._build_clickable_text(
            self._desktop_overlay_lock_label_for(False),
            self._on_desktop_overlay_lock_click,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        self._desktop_overlay_lock_card = self._wrap_unit_card(
            title=self._desktop_overlay_lock_title,
            value=self._desktop_overlay_lock_button,
        )

        self._desktop_overlay_swap_caption_languages_title = ft.Text(
            t("settings.overlay.desktop.swap_caption_languages.title"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._desktop_overlay_swap_caption_languages_button = self._build_clickable_text(
            t("settings.option.off"),
            self._on_desktop_overlay_swap_caption_languages_click,
        )
        self._desktop_overlay_swap_caption_languages_card = self._wrap_unit_card(
            title=self._desktop_overlay_swap_caption_languages_title,
            value=self._desktop_overlay_swap_caption_languages_button,
        )

        self._desktop_overlay_status_title = ft.Text(
            t("settings.overlay.status.off"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._desktop_overlay_reason_text = ft.Text(
            "",
            size=15,
            color=COLOR_SECONDARY,
            text_align=ft.TextAlign.CENTER,
            max_lines=2,
            overflow=ft.TextOverflow.ELLIPSIS,
            visible=False,
        )
        self._desktop_overlay_helper_text = ft.Text(
            "",
            size=14,
            color=COLOR_SECONDARY,
            text_align=ft.TextAlign.CENTER,
            max_lines=2,
            overflow=ft.TextOverflow.ELLIPSIS,
            visible=False,
        )
        self._desktop_overlay_primary_action = self._build_clickable_text(
            "",
            self._on_desktop_overlay_primary_action,
            size=20,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        self._desktop_overlay_primary_action.visible = False
        self._desktop_overlay_view_logs_action = self._build_clickable_text(
            t("settings.overlay.desktop.recovery.action.view_details"),
            self._on_desktop_overlay_view_logs,
            size=16,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        self._desktop_overlay_view_logs_action.visible = False
        self._desktop_overlay_status_body = ft.Column(
            [
                self._desktop_overlay_reason_text,
                self._desktop_overlay_primary_action,
                self._desktop_overlay_view_logs_action,
                self._desktop_overlay_helper_text,
            ],
            spacing=6,
            expand=True,
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )
        self._desktop_overlay_status_card = self._wrap_unit_card(
            title=self._desktop_overlay_status_title,
            value=self._desktop_overlay_status_body,
        )
        self._overlay_empty_card = self._wrap_empty_unit_card()
        self._overlay_desktop_reset_spacer = self._wrap_empty_unit_card()

        self._overlay_surface = compose_settings_overlay_surface(
            SettingsOverlaySurfaceSlots(
                overlay_target=self._overlay_target_card,
                overlay_translation=self._overlay_translation_card,
                overlay_peer_original=self._overlay_peer_original_card,
                anchor=self._overlay_anchor_card,
                distance=self._overlay_distance_card,
                offset_x=self._overlay_offset_x_card,
                offset_y=self._overlay_offset_y_card,
                text_scale=self._overlay_text_scale_card,
                vr_reset=self._overlay_vr_reset_card,
                desktop_size=self._desktop_overlay_size_card,
                desktop_lock=self._desktop_overlay_lock_card,
                desktop_background_alpha=self._desktop_overlay_background_alpha_card,
                desktop_swap_caption_languages=self._desktop_overlay_swap_caption_languages_card,
                desktop_reset=self._overlay_desktop_reset_card,
                desktop_reset_spacer=self._overlay_desktop_reset_spacer,
                desktop_status=self._desktop_overlay_status_card,
                desktop_status_trailing=self._overlay_empty_card,
            ),
            placeholder_factory=self._wrap_empty_unit_card,
        )
        self._overlay_vr_rows = self._overlay_surface.vr_rows
        self._overlay_desktop_rows = self._overlay_surface.desktop_rows
        self._desktop_overlay_controls_row = self._overlay_surface.desktop_controls_row
        self._desktop_overlay_recovery_row = self._overlay_surface.recovery_row
        self._sync_overlay_target_specific_visibility()

        self._translation_connection_title = ft.Text(
            t("settings.translation_connection"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._translation_connection_text = self._build_clickable_text(
            t("settings.translation_connection.managed"),
            self._on_translation_connection_click,
        )
        self._translation_connection_card = self._wrap_unit_card(
            title=self._translation_connection_title,
            value=self._translation_connection_text,
        )
        self._openrouter_fallback_title = ft.Text(
            t("settings.fallback"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._openrouter_fallback_text = self._build_clickable_text(
            t("settings.fallback.none"),
            self._on_openrouter_fallback_click,
        )
        self._openrouter_fallback_helper_text = ft.Text(
            t("settings.fallback.inactive_helper"),
            size=16,
            color=COLOR_SECONDARY,
        )
        self._openrouter_fallback_card = self._wrap_unit_card(
            title=self._openrouter_fallback_title,
            value=self._openrouter_fallback_text,
        )

        self._local_llm_connection_title = ft.Text(
            t("settings.local_llm.connection"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._local_llm_base_url = ft.TextField(
            label=t("settings.local_llm.base_url"),
            value="http://127.0.0.1:11434/v1",
            border_radius=12,
            border_color=COLOR_DIVIDER,
            focused_border_color=COLOR_PRIMARY,
            expand=True,
            text_size=24,
            color=COLOR_NEUTRAL_DARK,
            label_style=ft.TextStyle(size=18, weight=ft.FontWeight.BOLD, color=COLOR_NEUTRAL_DARK),
            on_change=self._on_local_llm_field_change,
            on_blur=self._on_local_llm_base_url_change_end,
            on_submit=self._on_local_llm_base_url_change_end,
        )
        self._local_llm_model = ft.TextField(
            label=t("settings.local_llm.model"),
            value="llama3.1:8b",
            border_radius=12,
            border_color=COLOR_DIVIDER,
            focused_border_color=COLOR_PRIMARY,
            expand=True,
            text_size=24,
            color=COLOR_NEUTRAL_DARK,
            label_style=ft.TextStyle(size=18, weight=ft.FontWeight.BOLD, color=COLOR_NEUTRAL_DARK),
            on_change=self._on_local_llm_field_change,
            on_blur=self._on_local_llm_model_change_end,
            on_submit=self._on_local_llm_model_change_end,
        )
        self._local_llm_api_key = ApiKeyField(
            "settings.local_llm.api_key",
            "local_llm_api_key",
            "local_llm",
            on_verify=None,
            on_save=self._on_local_llm_secret_change,
            show_snackbar=lambda msg, bg: (
                self.show_snackbar(msg, bg) if self.show_snackbar else None
            ),
            show_status=False,
        )
        local_llm_api_key_description = t("settings.local_llm.api_key.description")
        self._local_llm_api_key_helper = ft.Text(
            local_llm_api_key_description,
            size=15,
            color=COLOR_SECONDARY,
            visible=bool(local_llm_api_key_description.strip()),
        )
        self._local_llm_extra_body = ft.TextField(
            label=t("settings.local_llm.extra_body"),
            value=json.dumps(
                {"reasoning_effort": "none", "temperature": 0.6},
                ensure_ascii=False,
                indent=2,
            ),
            multiline=True,
            min_lines=3,
            max_lines=6,
            border_radius=12,
            border_color=COLOR_DIVIDER,
            focused_border_color=COLOR_PRIMARY,
            expand=True,
            text_size=24,
            color=COLOR_NEUTRAL_DARK,
            label_style=ft.TextStyle(size=18, weight=ft.FontWeight.BOLD, color=COLOR_NEUTRAL_DARK),
            on_change=self._on_local_llm_field_change,
            on_blur=self._on_local_llm_extra_body_change_end,
            on_submit=self._on_local_llm_extra_body_change_end,
        )
        self._local_llm_extra_body_helper = ft.Text(
            t("settings.local_llm.extra_body.description"),
            size=15,
            color=COLOR_SECONDARY,
        )
        self._local_llm_extra_body_error = ft.Text(
            "",
            size=13,
            color=ft.Colors.RED_600,
            visible=False,
        )
        self._local_llm_extra_body_error_key = ""
        self._local_llm_extra_body_error_kwargs: dict[str, object] = {}
        self._local_llm_connection_card = self._wrap_card(
            ft.Column(
                [
                    self._local_llm_connection_title,
                    ft.Container(height=4),
                    self._local_llm_extra_body_helper,
                    self._local_llm_base_url,
                    self._local_llm_model,
                    self._local_llm_api_key,
                    self._local_llm_api_key_helper,
                    self._local_llm_extra_body,
                    self._local_llm_extra_body_error,
                ],
                spacing=8,
            ),
            height=None,
        )
        self._local_llm_connection_card.visible = False

        self._custom_stt_connection_title = ft.Text(
            t("settings.custom_stt.title"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._custom_stt_endpoint = ft.TextField(
            label=t("settings.custom_stt.endpoint"),
            value="",
            border_radius=12,
            border_color=COLOR_DIVIDER,
            focused_border_color=COLOR_PRIMARY,
            expand=True,
            text_size=24,
            color=COLOR_NEUTRAL_DARK,
            label_style=ft.TextStyle(size=18, weight=ft.FontWeight.BOLD, color=COLOR_NEUTRAL_DARK),
            on_change=self._on_custom_stt_field_change,
            on_blur=self._on_custom_stt_endpoint_change_end,
            on_submit=self._on_custom_stt_endpoint_change_end,
        )
        self._custom_stt_model = ft.TextField(
            label=t("settings.custom_stt.model"),
            value="",
            border_radius=12,
            border_color=COLOR_DIVIDER,
            focused_border_color=COLOR_PRIMARY,
            expand=True,
            text_size=24,
            color=COLOR_NEUTRAL_DARK,
            label_style=ft.TextStyle(size=18, weight=ft.FontWeight.BOLD, color=COLOR_NEUTRAL_DARK),
            on_change=self._on_custom_stt_field_change,
            on_blur=self._on_custom_stt_model_change_end,
            on_submit=self._on_custom_stt_model_change_end,
        )
        self._custom_stt_api_key = ApiKeyField(
            "settings.custom_stt.api_key",
            "custom_stt_api_key",
            "custom",
            on_verify=None,
            on_save=self._on_custom_stt_secret_change,
            show_snackbar=lambda msg, bg: (
                self.show_snackbar(msg, bg) if self.show_snackbar else None
            ),
            show_status=False,
        )
        custom_stt_api_key_description = t("settings.custom_stt.api_key.description")
        self._custom_stt_api_key_helper = ft.Text(
            custom_stt_api_key_description,
            size=15,
            color=COLOR_SECONDARY,
            visible=bool(custom_stt_api_key_description.strip()),
        )
        self._custom_stt_extra = ft.TextField(
            label=t("settings.custom_stt.extra"),
            value="{}",
            multiline=True,
            min_lines=1,
            max_lines=12,
            border_radius=12,
            border_color=COLOR_DIVIDER,
            focused_border_color=COLOR_PRIMARY,
            expand=True,
            text_size=24,
            color=COLOR_NEUTRAL_DARK,
            label_style=ft.TextStyle(size=18, weight=ft.FontWeight.BOLD, color=COLOR_NEUTRAL_DARK),
            on_change=self._on_custom_stt_field_change,
            on_blur=self._on_custom_stt_extra_change_end,
            on_submit=self._on_custom_stt_extra_change_end,
        )
        self._custom_stt_extra_error = ft.Text(
            "",
            size=13,
            color=ft.Colors.RED_600,
            visible=False,
        )
        self._custom_stt_extra_error_key = ""
        self._custom_stt_extra_error_kwargs: dict[str, object] = {}
        self._custom_stt_connection_card = self._wrap_card(
            ft.Column(
                [
                    self._custom_stt_connection_title,
                    ft.Container(height=4),
                    self._custom_stt_endpoint,
                    self._custom_stt_model,
                    self._custom_stt_extra,
                    self._custom_stt_extra_error,
                    self._custom_stt_api_key,
                    self._custom_stt_api_key_helper,
                ],
                spacing=8,
            ),
            height=None,
        )
        self._custom_stt_connection_card.visible = False

        self._http_extension_title = ft.Text(
            t("settings.http_extension.title"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._http_extension_text = self._build_clickable_text(
            t("settings.http_extension.none"),
            self._on_http_extension_click,
            no_wrap=True,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        self._http_extension_selection_card = self._wrap_unit_card(
            title=self._http_extension_title,
            value=self._http_extension_text,
        )

        self._http_extension_path_title = ft.Text(
            t("settings.http_extension.path"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._http_extension_path_text = self._build_clickable_text(
            t("settings.http_extension.open"),
            self._on_http_extension_open_folder,
            no_wrap=True,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        self._http_extension_path_card = self._wrap_unit_card(
            title=self._http_extension_path_title,
            value=self._http_extension_path_text,
        )

        self._http_extension_refresh_title = ft.Text(
            t("settings.http_extension.refresh"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._http_extension_refresh_icon = ft.Container(
            content=ft.Icon(
                ft.Icons.REFRESH_ROUNDED,
                size=44,
                color=COLOR_ON_BACKGROUND,
            ),
            alignment=_CENTER_ALIGNMENT,
            expand=True,
            on_click=self._on_http_extension_reload,
            on_hover=self._on_text_hover,
        )
        self._http_extension_refresh_card = self._wrap_unit_card(
            title=self._http_extension_refresh_title,
            value=self._http_extension_refresh_icon,
        )

        self._http_extension_credentials = ft.Column([], spacing=12, visible=False)
        self._api_keys_column.controls.append(self._http_extension_credentials)

        self._http_extension_row = ft.Row(
            [
                self._http_extension_selection_card,
                self._http_extension_path_card,
                self._http_extension_refresh_card,
            ],
            spacing=SETTINGS_ROW_SPACING,
            expand=True,
        )
        self._http_extension_host = ft.Container(
            content=self._http_extension_row,
            visible=False,
        )

        # === Row 8: Persona (2x2) - Licenses style ===
        self._prompt_editor = PromptEditor(
            on_change=self._on_prompt_change,
            on_commit=self._on_prompt_commit,
        )
        self._persona_title = ft.Text(
            t("settings.section.persona"), size=24, weight=ft.FontWeight.BOLD, color=COLOR_SECONDARY
        )
        self._prompt_for_text = ft.Text(
            self._prompt_provider_copy(),
            size=16,
            color=COLOR_SECONDARY,
        )

        # Reset button (matches Persona title color, hover -> primary)
        self._reset_prompt_btn = _make_text_button(
            t("settings.reset_prompt"),
            icon=ft.Icons.REFRESH_ROUNDED,
            style=_settings_secondary_text_button_style(),
            on_click=self._on_reset_prompt,
        )

        # Header row with title and reset button
        persona_header = ft.Row(
            controls=[self._persona_title, ft.Container(expand=True), self._reset_prompt_btn],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        # Simple container like Licenses (no border, no internal scroll)
        prompt_container = ft.Container(
            content=self._prompt_editor,
            width=float("inf"),
        )

        persona_card = SharedCardWrapper(
            ft.Column(
                [
                    persona_header,
                    ft.Container(height=16),
                    prompt_container,
                ],
                spacing=0,
            ),
            height=None,
            expand=False,
        )
        # === Row 9: Custom Vocabulary (2x1) ===
        self._custom_vocab_title = ft.Text(
            t("settings.section.custom_vocabulary"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._custom_vocab_description_text = ft.Text(
            t("settings.custom_vocabulary.description"),
            size=16,
            color=COLOR_SECONDARY,
        )
        self._custom_vocab_tag_editor = CustomVocabularyTagEditor(
            on_add_terms=self._on_custom_vocabulary_add_terms,
            on_remove_term=self._on_custom_vocabulary_remove_term,
        )
        self._apply_custom_vocabulary_tag_editor_locale()
        row7 = SharedCardWrapper(
            ft.Column(
                [
                    self._custom_vocab_title,
                    ft.Container(height=6),
                    self._custom_vocab_description_text,
                    ft.Container(height=12),
                    self._custom_vocab_tag_editor,
                ],
                spacing=0,
            ),
            height=None,
            expand=False,
        )

        self._api_surface = compose_settings_api_surface(
            SettingsApiSurfaceSlots.from_slot_provider(self),
            placeholder_factory=self._wrap_empty_unit_card,
        )
        self._translation_connection_row = self._api_surface.translation_connection_row
        self._openrouter_routing_row = self._translation_connection_row
        self._gpu_device_row = self._api_surface.gpu_device_row

        self._prompt_surface = compose_settings_prompt_surface(
            SettingsPromptSurfaceSlots(custom_vocabulary=row7, persona=persona_card)
        )

        self._settings_subtab_shell = self._build_settings_subtab_shell(
            {
                "api": list(self._api_surface.rows),
                "general": list(self._general_surface.rows),
                "prompt": list(self._prompt_surface.rows),
                "overlay": list(self._overlay_surface.rows),
            }
        )
        self.controls = [self._settings_subtab_shell]

    def _gpu_asr_selected(self, settings: ProviderSettingsSnapshot) -> bool:
        return (
            settings.stt_provider == STTProviderName.LOCAL_QWEN_GPU
            or settings.peer_stt_provider == STTProviderName.LOCAL_QWEN_GPU
        )

    def _gpu_llm_selected(self, settings: ProviderSettingsSnapshot) -> bool:
        model = settings.translation.model
        if model == TranslationModel.MANAGED_GEMMA_12B:
            return True
        return (
            model == TranslationModel.MANAGED_GEMMA
            and settings.translation.connection == TranslationConnection.GPU
        )

    def _gpu_selected(self, settings: ProviderSettingsSnapshot | None = None) -> bool:
        current = settings or self._build_settings_with_provider_draft()
        return bool(
            current is not None
            and (self._gpu_asr_selected(current) or self._gpu_llm_selected(current))
        )

    def _gpu_column_label(
        self,
        selected: str,
        devices: tuple[GpuDeviceOption, ...],
    ) -> str:
        if selected == "auto":
            return t("settings.gpu_device.auto")
        selected_device = next(
            (device for device in devices if device.device_id == selected),
            None,
        )
        if selected_device is not None:
            return selected_device.display_name
        return t("settings.gpu_device.unavailable", device=selected)

    def _sync_gpu_device_card(self) -> None:
        if not hasattr(self, "_gpu_device_text"):
            return
        settings = self._build_settings_with_provider_draft()
        asr_selected = settings.stt_gpu_device_id if settings is not None else "auto"
        llm_selected = settings.translation.gpu_device_id if settings is not None else "auto"
        asr_label = self._gpu_column_label(asr_selected, getattr(self, "_gpu_devices", ()))
        llm_label = self._gpu_column_label(llm_selected, getattr(self, "_llm_gpu_devices", ()))
        self._set_unit_card_value_text(self._gpu_device_text, asr_label)
        self._set_unit_card_value_text(self._gpu_llm_text, llm_label)
        visible = self._gpu_selected(settings)
        self._gpu_device_row.visible = visible
        _update_control_if_mounted(self._gpu_device_row)

    def set_gpu_devices(
        self,
        *,
        devices: tuple[GpuDeviceOption, ...] | None = None,
        llm_devices: tuple[GpuDeviceOption, ...] | None = None,
    ) -> None:
        if devices is not None:
            self._gpu_devices = devices
        if llm_devices is not None:
            self._llm_gpu_devices = llm_devices
        self._sync_gpu_device_card()

    @staticmethod
    def _gpu_backend_label(name: str) -> str:
        match = re.fullmatch(r"Vulkan\s*(\d+)", name.strip(), flags=re.IGNORECASE)
        if match is not None:
            return f"Vulkan {match.group(1)}"
        return name.strip()

    def _gpu_column_options(
        self,
        selected: str,
        devices: tuple[GpuDeviceOption, ...],
    ) -> list[OptionItem]:
        options = [
            OptionItem(
                value="auto",
                label=t("settings.gpu_device.auto"),
            )
        ]
        options.extend(
            OptionItem(
                value=device.device_id,
                label=device.display_name,
                description=self._gpu_backend_label(device.backend_name),
            )
            for device in devices
        )
        if selected != "auto" and all(device.device_id != selected for device in devices):
            options.append(
                OptionItem(
                    value=selected,
                    label=t("settings.gpu_device.unavailable", device=selected),
                )
            )
        return options

    def _open_gpu_column_modal(
        self,
        *,
        title: str,
        selected: str,
        devices: tuple[GpuDeviceOption, ...],
        on_select,
    ) -> None:
        if self.on_gpu_discovery_requested is not None:
            self.on_gpu_discovery_requested()
        SettingsModal(
            self.page,
            title,
            self._gpu_column_options(selected, devices),
            on_select,
            show_description=True,
        ).open(selected)

    def _on_gpu_device_click(self, _event) -> None:
        if not is_control_mounted(self):
            return
        settings = self._build_settings_with_provider_draft()
        selected = settings.stt_gpu_device_id if settings is not None else "auto"
        self._open_gpu_column_modal(
            title=t("settings.gpu_device.asr"),
            selected=selected,
            devices=self._gpu_devices,
            on_select=self._on_gpu_device_selected,
        )

    def _on_llm_gpu_device_click(self, _event) -> None:
        if not is_control_mounted(self):
            return
        settings = self._build_settings_with_provider_draft()
        selected = settings.translation.gpu_device_id if settings is not None else "auto"
        self._open_gpu_column_modal(
            title=t("settings.gpu_device.llm"),
            selected=selected,
            devices=self._llm_gpu_devices,
            on_select=self._on_llm_gpu_device_selected,
        )

    def _on_gpu_refresh_click(self, _event) -> None:
        if self.on_gpu_discovery_requested is not None:
            self.on_gpu_discovery_requested()

    def _on_gpu_device_selected(self, value: str) -> None:
        if self._provider_snapshot is None:
            return
        draft = self._ensure_provider_settings_draft()
        self._provider_draft = replace(draft, stt_gpu_device_id=value or "auto")
        self._record_provider_edit(SttGpuDeviceEdit(self._provider_draft.stt_gpu_device_id))
        self.has_provider_changes = True
        self._sync_gpu_device_card()

    def _on_llm_gpu_device_selected(self, value: str) -> None:
        if self._provider_snapshot is None:
            return
        draft = self._ensure_provider_settings_draft()
        translation = replace(draft.translation, gpu_device_id=value or "auto")
        self._provider_draft = replace(draft, translation=translation)
        self._record_provider_edit(LlmGpuDeviceEdit(translation.gpu_device_id))
        self.has_provider_changes = True
        self._sync_gpu_device_card()

    def _populate_host_apis(self) -> None:
        """Legacy hook for tests; host APIs are handled by AudioSettings."""
        return None

    def _refresh_microphones(self) -> None:
        """Legacy hook for tests; microphone list is handled by AudioSettings."""
        return None

    def _build_locale_options(self) -> list[ft.dropdown.Option]:
        """Build locale dropdown options."""
        return [
            ft.dropdown.Option(key=code, text=locale_label(code)) for code in available_locales()
        ]

    def _http_extension_modal_options(self) -> list[OptionItem]:
        options = [OptionItem(value="", label=t("settings.http_extension.none"))]
        options.extend(
            OptionItem(
                value=loaded.definition.id,
                label=loaded.definition.name,
                description=loaded.definition.description or None,
            )
            for loaded in self._http_extension_snapshot.extensions
        )
        return options

    def _sync_http_extension_credentials(self, extension) -> None:
        self._http_extension_secret_fields = {}
        self._http_extension_secret_dirty.clear()
        secret_values: dict[str, str] = {}
        if extension is not None and self._settings_secrets is not None:
            keys = tuple(
                http_extension_secret_key(extension.id, secret.id) for secret in extension.secrets
            )
            result = self._settings_secrets.load_values(keys)
            if result.error_message is not None:
                self._emit_runtime_basic(result.error_message, level=logging.WARNING)
            if result.snapshot is not None:
                secret_values = {
                    secret.id: result.snapshot.get(
                        http_extension_secret_key(extension.id, secret.id)
                    )
                    or ""
                    for secret in extension.secrets
                }
            if result.read_error is not None:
                self._emit_runtime_basic(
                    f"Failed to load HTTP extension secrets: {result.read_error}",
                    level=logging.WARNING,
                )
        controls: list[ft.Control] = []
        if extension is not None:
            for secret in extension.secrets:
                field = ft.TextField(
                    label=t("settings.http_extension.api_key"),
                    value=secret_values.get(secret.id, ""),
                    password=True,
                    can_reveal_password=False,
                    border_radius=12,
                    border_color=COLOR_DIVIDER,
                    focused_border_color=COLOR_PRIMARY,
                    expand=True,
                    text_size=24,
                    color=COLOR_NEUTRAL_DARK,
                    label_style=ft.TextStyle(
                        size=18,
                        weight=ft.FontWeight.BOLD,
                        color=COLOR_NEUTRAL_DARK,
                    ),
                    on_change=lambda _event, secret_id=secret.id: (
                        self._http_extension_secret_dirty.add(secret_id)
                    ),
                    on_blur=lambda _event, secret_id=secret.id: self._on_http_extension_secret_blur(
                        secret_id
                    ),
                )
                reveal_button = ft.IconButton(
                    icon=ft.Icons.VISIBILITY_OFF_ROUNDED,
                    icon_color=COLOR_DIVIDER,
                    icon_size=24,
                )

                def _on_toggle_secret_reveal(
                    _event,
                    field: ft.TextField = field,
                    button: ft.IconButton = reveal_button,
                ) -> None:
                    field.password = not field.password
                    button.icon = (
                        ft.Icons.VISIBILITY_OFF_ROUNDED
                        if field.password
                        else ft.Icons.VISIBILITY_ROUNDED
                    )
                    update_control_if_mounted(field)
                    update_control_if_mounted(button)

                reveal_button.on_click = _on_toggle_secret_reveal
                field.suffix = reveal_button
                self._http_extension_secret_fields[secret.id] = field
                controls.append(field)
        self._http_extension_credentials.controls = controls
        _update_control_if_mounted(self._http_extension_credentials)

    def _sync_http_extension_card(
        self,
        settings: ProviderSettingsSnapshot | None = None,
        *,
        force_credentials: bool = False,
    ) -> None:
        if not hasattr(self, "_http_extension_row"):
            return
        if settings is None:
            settings = self._build_settings_with_provider_draft()
        if settings is None:
            return
        is_custom = settings.translation.model == TranslationModel.CUSTOM_HTTP
        self._http_extension_row.visible = is_custom
        self._http_extension_host.visible = is_custom
        self._http_extension_credentials.visible = is_custom
        if not is_custom:
            _update_control_if_mounted(self._http_extension_host)
            return
        selected_id = settings.translation.http_extension_id
        loaded = self._http_extension_snapshot.get(selected_id)
        selected_changed = selected_id != self._http_extension_selected_id
        self._set_unit_card_value_text(
            self._http_extension_text,
            loaded.definition.name if loaded else t("settings.http_extension.none"),
        )
        if selected_changed or force_credentials:
            self._sync_http_extension_credentials(loaded.definition if loaded else None)
            self._http_extension_selected_id = selected_id
        _update_control_if_mounted(self._http_extension_host)
        _update_control_if_mounted(self._http_extension_credentials)

    def _on_http_extension_click(self, _event) -> None:
        if not is_control_mounted(self):
            return
        settings = self._build_settings_with_provider_draft()
        selected = settings.translation.http_extension_id if settings is not None else ""
        SettingsModal(
            self.page,
            t("settings.http_extension.title"),
            self._http_extension_modal_options(),
            self._on_http_extension_selected,
            show_description=True,
        ).open(selected)

    def _on_http_extension_selected(self, value: str) -> None:
        if self._provider_snapshot is None:
            return
        draft = self._ensure_provider_settings_draft()
        translation = replace(draft.translation, http_extension_id=value or "")
        self._provider_draft = replace(draft, translation=translation)
        self._record_provider_edit(TranslationHttpExtensionEdit(translation.http_extension_id))
        self.has_provider_changes = True
        self._sync_http_extension_card(self._provider_draft, force_credentials=True)

    def _on_http_extension_secret_blur(self, secret_id: str) -> None:
        http_extension_id = self._http_extension_selected_id
        field = self._http_extension_secret_fields.get(secret_id)
        if (
            not http_extension_id
            or field is None
            or secret_id not in self._http_extension_secret_dirty
        ):
            return
        value = (field.value or "").strip()
        self._http_extension_secret_dirty.discard(secret_id)
        result = self._on_secret_change(
            http_extension_secret_key(http_extension_id, secret_id),
            value,
        )
        if inspect.isawaitable(result):
            self._schedule_page_task(self._finish_http_extension_secret_save, result)

    async def _finish_http_extension_secret_save(self, result) -> None:
        succeeded = await result
        if succeeded is False and self.show_snackbar is not None:
            self.show_snackbar(
                t("settings.http_extension.credential_save_failed"),
                ft.Colors.RED_400,
            )

    def _schedule_page_task(self, callback: Callable[..., object], *args: object) -> None:
        page = getattr(self, "page", None)
        if page is not None:
            page.run_task(callback, *args)

    def _on_http_extension_open_folder(self, _event) -> None:
        try:
            self._http_extensions.open_directory()
        except Exception:
            if self.show_snackbar is not None:
                self.show_snackbar(
                    t("settings.http_extension.open_folder_failed"),
                    ft.Colors.RED_400,
                )

    def _on_http_extension_reload(self, _event) -> None:
        settings = self._build_settings_with_provider_draft()
        previous_snapshot = self._http_extension_snapshot
        active_settings = self._provider_snapshot
        selected_id = (
            active_settings.translation.http_extension_id
            if active_settings is not None
            and active_settings.translation.model == TranslationModel.CUSTOM_HTTP
            else None
        )
        previous_selected = previous_snapshot.get(selected_id) if selected_id else None
        self._http_extension_snapshot = self._http_extensions.reload()
        current_selected = self._http_extension_snapshot.get(selected_id) if selected_id else None
        self._sync_http_extension_card(settings, force_credentials=True)
        if self._http_extension_snapshot.errors and self.show_snackbar is not None:
            self.show_snackbar(
                t(
                    "settings.http_extension.reload_errors",
                    count=len(self._http_extension_snapshot.errors),
                ),
                ft.Colors.ORANGE_700,
            )
        if (
            active_settings is not None
            and active_settings.translation.model == TranslationModel.CUSTOM_HTTP
            and (
                previous_selected is None
                and current_selected is not None
                or previous_selected is not None
                and current_selected is None
                or previous_selected is not None
                and current_selected is not None
                and previous_selected.fingerprint != current_selected.fingerprint
            )
            and self.on_providers_changed is not None
        ):
            self._http_extension_runtime_reload_pending = True
            self.on_providers_changed()

    def consume_http_extension_runtime_reload(self) -> bool:
        pending = self._http_extension_runtime_reload_pending
        self._http_extension_runtime_reload_pending = False
        return pending

    def _get_llm_modal_value(self, settings: ProviderSettingsSnapshot) -> str:
        model = settings.translation.model
        if model == TranslationModel.MANAGED_GEMMA:
            connection = settings.translation.connection
            if connection == TranslationConnection.GPU:
                return "managed_gemma_gpu"
            return "managed_gemma_cpu"
        return model.value

    def _translation_model_display_label(self, model: TranslationModel) -> str:
        return t(_TRANSLATION_MODEL_LABEL_KEYS[model])

    def _translation_connection_display_label(self, connection: TranslationConnection) -> str:
        return t(_TRANSLATION_CONNECTION_LABEL_KEYS[connection])

    def _translation_connection_display_description(self, connection: TranslationConnection) -> str:
        return t(_TRANSLATION_CONNECTION_DESCRIPTION_KEYS[connection], default="")

    def _translation_connection_only_supported_description(self) -> str:
        return t(_TRANSLATION_CONNECTION_ONLY_SUPPORTED_KEY, default="")

    def _set_translation_connection_text(self, text: str) -> None:
        text_control = self._translation_connection_text.content
        text_control.value = text
        text_control.size = 28

    def _sync_translation_connection_title(self, settings: ProviderSettingsSnapshot) -> None:
        title = getattr(self, "_translation_connection_title", None)
        if title is None:
            return
        title.value = t("settings.translation_connection")

    def _stored_openrouter_selection_alias(
        self, settings: ProviderSettingsSnapshot
    ) -> OpenRouterSelectionAlias | None:
        if settings.openrouter_selection_alias is None:
            if settings.openrouter_selected_source == OpenRouterCredentialSource.NONE:
                return None
            return _derive_openrouter_selection_alias(
                settings.openrouter_llm_model,
                settings.openrouter_selected_source,
            )
        try:
            profile_for_alias(settings.openrouter_selection_alias.value)
            return settings.openrouter_selection_alias
        except KeyError:
            if settings.openrouter_selected_source == OpenRouterCredentialSource.NONE:
                return None
            return _derive_openrouter_selection_alias(
                settings.openrouter_llm_model,
                settings.openrouter_selected_source,
            )

    def _display_openrouter_selection_alias(
        self, settings: ProviderSettingsSnapshot
    ) -> OpenRouterSelectionAlias:
        stored_alias = self._stored_openrouter_selection_alias(settings)
        if stored_alias is not None:
            return stored_alias
        if settings.openrouter_llm_model == OpenRouterLLMModel.QWEN_35_FLASH_02_23:
            return OpenRouterSelectionAlias.QWEN35_FLASH_MANAGED
        if settings.openrouter_llm_model == OpenRouterLLMModel.DEEPSEEK_V4_FLASH:
            return OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED
        return OpenRouterSelectionAlias.GEMMA4_MANAGED

    def _openrouter_selection_profile(self, settings: ProviderSettingsSnapshot | None):
        if settings is None:
            return None
        try:
            return profile_for_alias(self._display_openrouter_selection_alias(settings).value)
        except KeyError:
            return None

    def _translation_fallback_preset_value(self, fallback: TranslationFallbackSnapshot) -> str:
        for value, preset, _label_key in _TRANSLATION_FALLBACK_PRESETS:
            if (
                preset.enabled == fallback.enabled
                and preset.model == fallback.model
                and preset.connection == fallback.connection
            ):
                return value
        if fallback.connection in (
            TranslationConnection.MANAGED,
            TranslationConnection.MANAGED_CHINA,
        ):
            return "none"
        return "custom"

    def _translation_fallback_display_label(
        self,
        fallback: TranslationFallbackSnapshot,
    ) -> str:
        preset_value = self._translation_fallback_preset_value(fallback)
        label_key = _TRANSLATION_FALLBACK_LABEL_KEY_BY_VALUE.get(preset_value)
        if label_key is not None:
            return t(label_key)
        model_label = self._translation_model_display_label(fallback.model)
        connection_label = self._translation_connection_display_label(fallback.connection)
        return f"{model_label} · {connection_label}"

    def _openrouter_fallback_source(
        self, settings: ProviderSettingsSnapshot | None
    ) -> OpenRouterCredentialSource:
        if settings is None or not self._translation_uses_provider_fallback(settings):
            return OpenRouterCredentialSource.NONE
        fallback = settings.translation.fallback
        if not fallback.enabled:
            return OpenRouterCredentialSource.NONE
        if fallback.connection == TranslationConnection.OPENROUTER:
            return OpenRouterCredentialSource.BYOK
        if fallback.connection in (
            TranslationConnection.MANAGED,
            TranslationConnection.MANAGED_CHINA,
        ):
            return OpenRouterCredentialSource.MANAGED
        return OpenRouterCredentialSource.NONE

    def _openrouter_profile_display_label(self, profile) -> str:
        return t(profile.label_key)

    def _openrouter_profile_display_description(self, profile) -> str:
        return t(profile.description_key, default="")

    def _get_llm_display_label(self, settings: ProviderSettingsSnapshot) -> str:
        model = settings.translation.model
        if model == TranslationModel.MANAGED_GEMMA:
            if settings.translation.connection == TranslationConnection.GPU:
                return t("provider.managed_gemma_gpu")
            return t("provider.managed_gemma_cpu")
        return self._translation_model_display_label(model)

    def _get_translation_connection_display_label(
        self,
        settings: ProviderSettingsSnapshot | None,
    ) -> str:
        if settings is None:
            return self._translation_connection_display_label(TranslationConnection.MANAGED)
        return self._translation_connection_display_label(settings.translation.connection)

    def _get_openrouter_fallback_display_label(
        self,
        settings: ProviderSettingsSnapshot | None,
    ) -> str:
        if settings is None:
            return t("settings.fallback.none")
        return self._translation_fallback_display_label(settings.translation.fallback)

    def _get_openrouter_fallback_helper_text(
        self,
        settings: ProviderSettingsSnapshot | None,
    ) -> str:
        if settings is None:
            return t("settings.fallback.inactive_helper")
        if not settings.translation.fallback.enabled:
            return t("settings.fallback.none.description")
        return t("settings.fallback.active_helper")

    def _telemetry_enabled_display_label(
        self,
        settings: GeneralSettingsSnapshot | None,
    ) -> str:
        enabled = settings.telemetry_enabled if settings is not None else True
        return t("settings.telemetry.state.on" if enabled else "settings.telemetry.state.off")

    def _sync_telemetry_enabled_card(
        self,
        settings: GeneralSettingsSnapshot | None = None,
    ) -> None:
        if settings is None:
            settings = self._general_snapshot
        self._set_unit_card_value_text(
            self._telemetry_enabled_text,
            self._telemetry_enabled_display_label(settings),
        )

    def _set_openrouter_fallback_text(self, text: str) -> None:
        text_control = self._openrouter_fallback_text.content
        text_control.value = text
        text_control.size = 28

    def _sync_openrouter_fallback_card(
        self,
        settings: ProviderSettingsSnapshot | None = None,
    ) -> None:
        if settings is None:
            settings = self._build_settings_with_provider_draft()
        self._set_openrouter_fallback_text(self._get_openrouter_fallback_display_label(settings))
        self._openrouter_fallback_helper_text.value = self._get_openrouter_fallback_helper_text(
            settings
        )

    def _active_prompt_key_for_settings(
        self,
        settings: ProviderSettingsSnapshot | None,
    ) -> str:
        if settings is None:
            return "gemini"
        if settings.llm_provider == LLMProviderName.GEMINI:
            return "gemini"
        if settings.llm_provider == LLMProviderName.OPENROUTER:
            return "openrouter"
        if settings.llm_provider == LLMProviderName.DEEPSEEK:
            return "deepseek"
        if settings.llm_provider == LLMProviderName.LOCAL_LLM:
            return "local_llm"
        if settings.llm_provider == LLMProviderName.MANAGED_GEMMA:
            return "managed_gemma"
        return "qwen"

    def _active_prompt_key(self) -> str:
        return self._active_prompt_key_for_settings(self._build_settings_with_provider_draft())

    def _ensure_provider_prompt_value(
        self,
        settings: ProviderSettingsSnapshot,
        provider_name: str,
    ) -> str:
        prompt = self._prompt_editor.value
        if prompt.strip():
            return prompt
        prompt = load_prompt_for_provider(provider_name)
        return prompt

    def _current_source_language(self) -> str:
        if self._prompt_snapshot is None:
            return "en"
        return self._prompt_snapshot.source_language

    def _prompt_provider_copy(self) -> str:
        return t(
            "settings.prompt_for",
            provider=provider_label(self._active_prompt_key()),
        )

    def _custom_vocabulary_description_copy(self) -> str:
        return t("settings.custom_vocabulary.description")

    def _apply_custom_vocabulary_tag_editor_locale(self) -> None:
        self._custom_vocab_tag_editor.set_placeholder(
            t("settings.custom_vocabulary.add_placeholder")
        )
        self._custom_vocab_tag_editor.set_add_label(t("settings.custom_vocabulary.add_action"))
        self._custom_vocab_tag_editor.set_empty_text(t("settings.custom_vocabulary.empty"))
        self._custom_vocab_tag_editor.set_remove_label_template(
            t("settings.custom_vocabulary.remove_hint")
        )

    def _sync_prompt_tab_copy(self) -> None:
        self._prompt_for_text.value = self._prompt_provider_copy()
        self._custom_vocab_description_text.value = self._custom_vocabulary_description_copy()
        self._apply_custom_vocabulary_tag_editor_locale()
        peer_auto_languages_title = getattr(self, "_peer_auto_languages_title", None)
        if peer_auto_languages_title is not None:
            peer_auto_languages_title.value = t("settings.peer_auto_languages.title")
        peer_auto_languages_editor = getattr(self, "_peer_auto_languages_editor", None)
        if peer_auto_languages_editor is not None and hasattr(
            peer_auto_languages_editor, "apply_locale"
        ):
            peer_auto_languages_editor.apply_locale()
        if is_control_mounted(self):
            for control in (
                self._prompt_for_text,
                self._custom_vocab_description_text,
                peer_auto_languages_title,
            ):
                if control is not None:
                    with contextlib.suppress(Exception):
                        control.update()

    def _sync_custom_vocabulary_editor_from_settings(self) -> None:
        if self._prompt_snapshot is None:
            self._custom_vocab_tag_editor.set_terms([])
            self._custom_vocab_tag_editor.clear_input()
            return

        self._custom_vocab_tag_editor.set_terms(list(self._prompt_snapshot.custom_vocabulary_terms))
        self._custom_vocab_tag_editor.clear_input()

    def _sync_peer_auto_languages_editor(
        self,
        settings: GeneralSettingsSnapshot | None = None,
    ) -> None:
        if not hasattr(self, "_peer_auto_languages_editor"):
            return
        settings = settings or self._general_snapshot
        languages = () if settings is None else settings.peer_expected_languages
        self._peer_auto_languages_editor.set_terms(list(languages))

    def _set_peer_auto_languages(self, languages: list[str]) -> None:
        if self._general_snapshot is None:
            return
        normalized = list(
            dict.fromkeys(language.strip() for language in languages if language.strip())
        )
        if self._general_snapshot.peer_expected_languages == tuple(normalized):
            return
        self._general_snapshot = replace(
            self._general_snapshot,
            peer_expected_languages=tuple(normalized),
        )
        self._sync_peer_auto_languages_editor()
        self._emit_settings_changed(PeerExpectedLanguagesIntent(tuple(normalized)))

    def _on_peer_auto_languages_add(self, language: str) -> None:
        if self._general_snapshot is None:
            return
        self._set_peer_auto_languages([*self._general_snapshot.peer_expected_languages, language])

    def _on_peer_auto_languages_remove(self, language: str) -> None:
        if self._general_snapshot is None:
            return
        self._set_peer_auto_languages(
            [
                current
                for current in self._general_snapshot.peer_expected_languages
                if current != language
            ]
        )

    def _normalize_custom_vocabulary_submitted_terms(self, raw_terms: list[str]) -> list[str]:
        terms: list[str] = []
        for raw_term in raw_terms:
            for part in _CUSTOM_VOCAB_DELIMITER_RE.split(str(raw_term)):
                normalized = part.strip()
                if normalized:
                    terms.append(normalized)
        return terms

    @property
    def managed_trial_usage_state(self) -> dict[str, object]:
        return {
            "visible": self._managed_trial_usage_visible,
            "remaining_percent": self._managed_trial_usage_remaining_percent,
        }

    def _translation_uses_provider_fallback(
        self,
        settings: ProviderSettingsSnapshot | None,
    ) -> bool:
        return bool(
            settings is not None
            and settings.translation.model not in _TRANSLATION_MODELS_WITHOUT_PROVIDER_FALLBACK
        )

    def _is_managed_translation_connection_selected(
        self,
        settings: ProviderSettingsSnapshot | None,
    ) -> bool:
        if settings is None:
            return False
        if not self._translation_uses_provider_fallback(settings):
            return False
        managed_connections = (TranslationConnection.MANAGED, TranslationConnection.MANAGED_CHINA)
        return bool(
            settings.translation.connection in managed_connections
            or (
                settings.translation.fallback.enabled
                and settings.translation.fallback.connection in managed_connections
            )
        )

    def _managed_key_card_visible_for(
        self,
        settings: ProviderSettingsSnapshot | None,
    ) -> bool:
        return self._is_managed_translation_connection_selected(settings)

    def _sync_managed_key_referral_row_value(self, referral_id: str | None) -> None:
        referral_id = normalize_owned_referral_id(referral_id)
        self._managed_key_referral_id = referral_id

        self._managed_key_referral_id_value.value = referral_id or t(
            "settings.managed_key.referral_id.empty"
        )

    def _remember_managed_key_referral_id(self, referral_id: str | None) -> str | None:
        referral_id = normalize_owned_referral_id(referral_id)
        if referral_id is None:
            return None

        referral_changed = (
            self._provider_snapshot is not None
            and self._provider_snapshot.managed_referral_id != referral_id
        )
        if self._provider_snapshot is not None:
            self._provider_snapshot = replace(
                self._provider_snapshot,
                managed_referral_id=referral_id,
            )
        if self._provider_draft is not None:
            self._provider_draft = replace(
                self._provider_draft,
                managed_referral_id=referral_id,
            )
        if referral_changed:
            self._record_provider_edit(ManagedReferralEdit(referral_id))
            self.has_provider_changes = True
        return referral_id

    def _sync_managed_key_invite_progress_row(
        self,
        referral_id: str | None,
        pass_status: TalkTogetherPassStatus | None,
    ) -> None:
        normalized_referral_id = normalize_owned_referral_id(referral_id)
        if (
            normalized_referral_id is None
            or pass_status is None
            or pass_status.pass_id != normalized_referral_id
            or pass_status.invite_limit <= 0
            or pass_status.invite_count < 0
        ):
            self._managed_key_pass_status = None
            self._managed_key_invite_progress_label.value = t(
                "settings.managed_key.invite_progress.label"
            )
            self._managed_key_invite_progress_row.visible = normalized_referral_id is not None
            self._managed_key_invite_progress_value.value = "- / -"
            return

        self._managed_key_pass_status = pass_status
        displayed_count = min(pass_status.invite_count, pass_status.invite_limit)
        self._managed_key_invite_progress_label.value = t(
            "settings.managed_key.invite_progress.label"
        )
        self._managed_key_invite_progress_value.value = (
            f"{displayed_count} / {pass_status.invite_limit}"
        )
        self._managed_key_invite_progress_row.visible = True

    def _sync_managed_key_referral_row(
        self,
        settings: ProviderSettingsSnapshot | None,
    ) -> None:
        referral_id = (
            None if settings is None else normalize_owned_referral_id(settings.managed_referral_id)
        )
        self._sync_managed_key_referral_row_value(referral_id)

    def _sync_managed_key_card(
        self,
        settings: ProviderSettingsSnapshot | None = None,
    ) -> None:
        if settings is None:
            settings = self._build_settings_with_provider_draft()
        visible = self._managed_key_card_visible_for(settings)
        self._managed_key_card.visible = visible
        self._sync_managed_key_referral_row(settings)
        self._sync_managed_key_invite_progress_row(
            self._managed_key_referral_id,
            self._managed_key_pass_status if visible else None,
        )
        self._sync_managed_trial_usage_bar(settings)

    def _repaint_managed_key_card(self) -> None:
        self._repaint_managed_key_dynamic_controls()
        _update_control_if_mounted(self._managed_key_card)
        _update_control_if_mounted(self._api_keys_column)
        if hasattr(self, "_settings_subtab_shell"):
            api_body = self._settings_subtab_shell.body_by_key.get("api")
            if api_body is not None:
                _update_control_if_mounted(api_body)

    def _repaint_managed_key_dynamic_controls(self) -> None:
        usage_repaint = getattr(self._managed_trial_usage_bar, "repaint_dynamic_controls", None)
        if callable(usage_repaint):
            usage_repaint()
        else:
            for control_name in ("_fill_segments", "_remaining_text"):
                control = getattr(self._managed_trial_usage_bar, control_name, None)
                if control is not None:
                    _update_control_if_mounted(control)
        for control in (
            self._managed_trial_usage_bar,
            self._managed_key_referral_id_value,
            self._managed_key_invite_progress_label,
            self._managed_key_invite_progress_value,
            self._managed_key_invite_progress_row,
        ):
            _update_control_if_mounted(control)

    def set_managed_trial_usage_state(
        self, *, visible: bool, remaining_percent: int | None = None
    ) -> None:
        self._managed_trial_usage_visible = bool(visible)
        if self._managed_trial_usage_visible and remaining_percent is not None:
            self._managed_trial_usage_remaining_percent = max(0, min(100, int(remaining_percent)))
        else:
            self._managed_trial_usage_remaining_percent = None
        self._sync_managed_key_card()
        if is_control_mounted(self):
            with contextlib.suppress(Exception):
                self._repaint_managed_key_card()

    def set_managed_key_state(
        self,
        *,
        visible: bool,
        remaining_percent: int | None = None,
        referral_id: str | None = None,
        pass_status: TalkTogetherPassStatus | None = None,
        remember_referral_id: bool = True,
    ) -> None:
        referral_id = (
            self._remember_managed_key_referral_id(referral_id)
            if remember_referral_id
            else normalize_owned_referral_id(referral_id)
        )
        usage_visible = bool(visible)
        card_visible = self._managed_key_card_visible_for(
            self._build_settings_with_provider_draft()
        )
        self._managed_trial_usage_visible = usage_visible
        if usage_visible and remaining_percent is not None:
            self._managed_trial_usage_remaining_percent = max(0, min(100, int(remaining_percent)))
        else:
            self._managed_trial_usage_remaining_percent = None

        self._managed_key_card.visible = card_visible
        self._managed_trial_usage_bar.visible = card_visible
        self._managed_trial_usage_bar.set_percent(
            self._managed_trial_usage_remaining_percent if card_visible else None
        )
        self._sync_managed_key_referral_row_value(referral_id)
        self._sync_managed_key_invite_progress_row(
            referral_id,
            pass_status if card_visible else None,
        )
        self._repaint_managed_key_card()

    def _build_settings_with_provider_draft(self) -> ProviderSettingsSnapshot | None:
        return self._provider_draft or self._provider_snapshot

    def _ensure_provider_settings_draft(self) -> ProviderSettingsSnapshot:
        assert self._provider_snapshot is not None
        if self._provider_draft is None:
            self._provider_draft = self._provider_snapshot
        return self._provider_draft

    def _record_provider_edit(self, edit: ProviderSettingsEdit) -> None:
        self._provider_edits[type(edit)] = edit

    def _on_stt_rolling_toggle(self, e) -> None:
        if self._provider_snapshot is None:
            self._stt_rolling_switch.value = False
            return
        enabled = bool(e.control.value)
        current_settings = self._build_settings_with_provider_draft()
        if current_settings is not None and current_settings.stt_rolling_enabled == enabled:
            return
        draft = self._ensure_provider_settings_draft()
        self._provider_draft = replace(draft, stt_rolling_enabled=enabled)
        self._record_provider_edit(SttRollingEnabledEdit(enabled))
        self.has_provider_changes = True
        self._emit_runtime_basic(f"[Settings] STT rolling {'enabled' if enabled else 'disabled'}")

    def sync_stt_rolling_switch(self, settings: ProviderSettingsSnapshot | None) -> None:
        switch = getattr(self, "_stt_rolling_switch", None)
        if switch is None or settings is None:
            return
        switch.value = settings.stt_rolling_enabled

    def _translation_selection_edit(
        self,
        selection: TranslationSelectionSnapshot,
    ) -> TranslationSelectionEdit:
        committed_history = (
            {}
            if self._provider_snapshot is None
            else dict(self._provider_snapshot.translation.connection_history)
        )
        return TranslationSelectionEdit(
            selection,
            tuple(
                (model, connection)
                for model, connection in selection.connection_history
                if committed_history.get(model) != connection
            ),
        )

    def _stt_provider_display_label(
        self,
        provider: STTProviderName,
        *,
        custom_mode: str = "offline",
        qwen_asr_model: str | None = None,
    ) -> str:
        return provider_label(
            display_stt_provider(
                provider,
                custom_mode=custom_mode,
                qwen_asr_model=qwen_asr_model,
            ).value
        )

    def _display_stt_choice(
        self,
        settings: ProviderSettingsSnapshot,
        *,
        peer: bool = False,
    ) -> STTProviderName:
        provider = self._effective_peer_stt_provider(settings) if peer else settings.stt_provider
        return display_stt_provider(
            provider,
            custom_mode=settings.custom_stt_mode,
            qwen_asr_model=settings.qwen_asr_model,
        )

    def _normalized_peer_stt_provider(self, provider: STTProviderName) -> STTProviderName:
        return provider

    def _effective_peer_stt_provider(
        self,
        settings: ProviderSettingsSnapshot | None,
    ) -> STTProviderName:
        if settings is None:
            return STTProviderName.LOCAL_CPU_AUTO
        return self._normalized_peer_stt_provider(settings.peer_stt_provider)

    def _peer_stt_option_item(self, provider: STTProviderName) -> OptionItem:
        return self._stt_option_item(provider)

    def _classified_peer_stt_option_item(self, provider: STTProviderName) -> OptionItem:
        item = self._peer_stt_option_item(provider)
        section_key = _STT_SECTION_BY_PROVIDER.get(provider, "")
        section = t(section_key) if section_key else ""
        return OptionItem(
            value=item.value,
            label=item.label,
            description=item.description,
            disabled=item.disabled,
            section=section,
        )

    def _stt_option_item(self, provider: STTProviderName) -> OptionItem:
        auto_unavailable = (
            provider == STTProviderName.LOCAL_CPU_AUTO and not self._local_cpu_auto_available
        )
        description = (
            ""
            if provider == STTProviderName.QWEN_AUDIO
            else t(f"provider.{provider.value}.description", default="")
        )
        return OptionItem(
            value=provider.value,
            label=provider_label(provider.value),
            description=description,
            disabled=auto_unavailable,
        )

    def _classified_stt_option_item(self, provider: STTProviderName) -> OptionItem:
        item = self._stt_option_item(provider)
        section_key = _STT_SECTION_BY_PROVIDER.get(provider, "")
        section = t(section_key) if section_key else ""
        return OptionItem(
            value=item.value,
            label=item.label,
            description=item.description,
            disabled=item.disabled,
            section=section,
        )

    def _local_llm_extra_body_error_message(
        self,
        message_key: str,
        **kwargs: object,
    ) -> str:
        if "key" not in kwargs:
            return t(message_key, **kwargs)
        template = t(message_key)
        with contextlib.suppress(Exception):
            return template.format(**kwargs)
        return template

    def _show_local_llm_extra_body_error(self, message_key: str, **kwargs: object) -> None:
        message = self._local_llm_extra_body_error_message(message_key, **kwargs)
        self._local_llm_extra_body_error_key = message_key
        self._local_llm_extra_body_error_kwargs = dict(kwargs)
        self._local_llm_extra_body_error.value = message
        self._local_llm_extra_body_error.visible = True
        self._local_llm_extra_body.error = message
        _update_control_if_mounted(self._local_llm_extra_body)
        _update_control_if_mounted(self._local_llm_extra_body_error)

    def _on_local_llm_field_change(self, e) -> None:
        _ = e
        if self._provider_snapshot is None:
            return
        current = self._provider_draft or self._provider_snapshot
        if current.llm_provider != LLMProviderName.LOCAL_LLM:
            return
        self._ensure_provider_settings_draft()
        self.has_provider_changes = True

    def _clear_local_llm_extra_body_error(self) -> None:
        self._local_llm_extra_body_error_key = ""
        self._local_llm_extra_body_error_kwargs = {}
        self._local_llm_extra_body_error.value = ""
        self._local_llm_extra_body_error.visible = False
        self._local_llm_extra_body.error = None
        _update_control_if_mounted(self._local_llm_extra_body)
        _update_control_if_mounted(self._local_llm_extra_body_error)

    def _on_local_llm_base_url_change_end(self, e) -> None:
        _ = e
        if self._provider_snapshot is None:
            return
        raw_value = self._local_llm_base_url.value or ""
        try:
            normalized = normalize_local_llm_base_url(raw_value)
        except ValueError:
            self._local_llm_base_url.error = t("settings.local_llm.base_url.invalid")
            _update_control_if_mounted(self._local_llm_base_url)
            return

        self._local_llm_base_url.error = None
        self._local_llm_base_url.value = normalized
        current = self._provider_draft or self._provider_snapshot
        if current.local_llm_base_url != normalized:
            draft = self._ensure_provider_settings_draft()
            self._provider_draft = replace(draft, local_llm_base_url=normalized)
            self._record_provider_edit(LocalLlmBaseUrlEdit(self._provider_draft.local_llm_base_url))
            self.has_provider_changes = True
        _update_control_if_mounted(self._local_llm_base_url)

    def _on_local_llm_model_change_end(self, e) -> None:
        _ = e
        if self._provider_snapshot is None:
            return
        model = (self._local_llm_model.value or "").strip()
        if not model:
            self._local_llm_model.error = t("settings.local_llm.model.required")
            _update_control_if_mounted(self._local_llm_model)
            return

        self._local_llm_model.error = None
        self._local_llm_model.value = model
        current = self._provider_draft or self._provider_snapshot
        if current.local_llm_model != model:
            draft = self._ensure_provider_settings_draft()
            self._provider_draft = replace(draft, local_llm_model=model)
            self._record_provider_edit(LocalLlmModelEdit(self._provider_draft.local_llm_model))
            self.has_provider_changes = True
        _update_control_if_mounted(self._local_llm_model)

    def _on_local_llm_extra_body_change_end(self, e) -> None:
        _ = e
        if self._provider_snapshot is None:
            return
        raw = (self._local_llm_extra_body.value or "").strip()
        try:
            parsed = (
                {"reasoning_effort": "none", "temperature": 0.6}
                if not raw
                else json.loads(raw, parse_constant=_reject_json_constant)
            )
        except json.JSONDecodeError:
            self._show_local_llm_extra_body_error("settings.local_llm.extra_body.invalid_json")
            return

        if not isinstance(parsed, dict):
            self._show_local_llm_extra_body_error("settings.local_llm.extra_body.must_be_object")
            return

        lowered = {str(key).lower() for key in parsed}
        reserved = LOCAL_LLM_RESERVED_EXTRA_BODY_KEYS.intersection(lowered)
        if reserved:
            self._show_local_llm_extra_body_error(
                "settings.local_llm.extra_body.reserved_key",
                key=sorted(reserved)[0],
            )
            return

        sensitive = LOCAL_LLM_SENSITIVE_EXTRA_BODY_KEYS.intersection(lowered)
        if sensitive:
            self._show_local_llm_extra_body_error(
                "settings.local_llm.extra_body.sensitive_key",
                key=sorted(sensitive)[0],
            )
            return

        try:
            json.dumps(parsed, allow_nan=False)
        except (TypeError, ValueError):
            self._show_local_llm_extra_body_error("settings.local_llm.extra_body.not_serializable")
            return

        normalized = copy.deepcopy(parsed)
        normalized_text = json.dumps(
            normalized,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        current = self._provider_draft or self._provider_snapshot
        if current.local_llm_extra_body_json != normalized_text:
            draft = self._ensure_provider_settings_draft()
            self._provider_draft = replace(draft, local_llm_extra_body_json=normalized_text)
            self._record_provider_edit(
                LocalLlmExtraBodyEdit(self._provider_draft.local_llm_extra_body_json)
            )
            self.has_provider_changes = True
        self._local_llm_extra_body.value = normalized_text
        self._clear_local_llm_extra_body_error()
        _update_control_if_mounted(self._local_llm_extra_body)

    def _commit_local_llm_fields_from_controls(self) -> None:
        if self._provider_snapshot is None:
            return
        current = self._provider_draft or self._provider_snapshot
        if current.llm_provider != LLMProviderName.LOCAL_LLM:
            return
        self._on_local_llm_base_url_change_end(None)
        self._on_local_llm_model_change_end(None)
        self._on_local_llm_extra_body_change_end(None)

    def _stage_prompt_draft(self, value: str) -> None:
        if self._prompt_snapshot is None:
            return
        committed_prompt = self._committed_prompt_value()
        self.has_pending_prompt_changes = value != committed_prompt
        if self.has_pending_prompt_changes:
            self._record_provider_edit(SystemPromptEdit(value))
        else:
            self._provider_edits.pop(SystemPromptEdit, None)

    def _committed_prompt_value(self) -> str:
        if self._prompt_snapshot is None:
            return ""
        return self._prompt_snapshot.system_prompt

    def _build_provider_apply_intent(self) -> ProviderApplyIntent | None:
        self._commit_local_llm_fields_from_controls()
        if self._provider_snapshot is None:
            return None
        return ProviderApplyIntent(tuple(self._provider_edits.values()))

    def build_provider_apply_settings(self) -> ProviderApplyIntent | None:
        return self._build_provider_apply_intent()

    def consume_provider_apply_settings(self) -> ProviderApplyIntent | None:
        intent = self.build_provider_apply_settings()
        if intent is None:
            return None
        if self._provider_draft is not None:
            self._provider_snapshot = self._provider_draft
        if self._prompt_snapshot is not None and self.has_pending_prompt_changes:
            self._prompt_snapshot = replace(
                self._prompt_snapshot,
                system_prompt=self._prompt_editor.value,
            )
        self._provider_draft = None
        self._provider_edits.clear()
        self.has_provider_changes = False
        self.has_pending_prompt_changes = False
        return intent

    def consume_prompt_apply_settings(self) -> PromptApplyIntent | None:
        if not self.has_pending_prompt_changes:
            return None
        if self._prompt_snapshot is None:
            return None
        value = self._prompt_editor.value
        self._prompt_snapshot = replace(self._prompt_snapshot, system_prompt=value)
        self.has_pending_prompt_changes = False
        self._provider_edits.pop(SystemPromptEdit, None)
        return PromptApplyIntent(value)

    # --- Load Settings ---
    def load_from_settings(
        self,
        *,
        provider: ProviderSettingsSnapshot,
        general: GeneralSettingsSnapshot,
        prompt: PromptSettingsSnapshot,
        overlay: OverlaySettingsSnapshot,
        config_path: Path,
        preserve_custom_vocab_draft: bool = False,
    ) -> None:
        """Load current settings into the UI."""
        self._provider_snapshot = provider
        self._provider_draft = None
        self._provider_edits.clear()
        self._general_snapshot = general
        self._prompt_snapshot = prompt
        self._overlay_snapshot = overlay
        self._config_path = config_path
        self._http_extension_runtime_reload_pending = False
        self._http_extension_secret_dirty.clear()
        self.has_provider_changes = False
        self.has_pending_prompt_changes = False
        self._desktop_overlay_pending_size_preset = None
        self._desktop_overlay_pending_position_reset = False
        self._desktop_overlay_pending_locked = None
        self._desktop_overlay_captions_locked = False
        if self._overlay_state == "off":
            self._overlay_runtime_target = self._current_overlay_target()
        self._sync_clickable_text_control_fonts(font_for_language(general.locale))

        # UI Language
        self._ui_text.content.value = locale_label(general.locale)

        # STT Provider
        self._set_unit_card_value_text(
            self._stt_text,
            self._stt_provider_display_label(
                provider.stt_provider,
                custom_mode=provider.custom_stt_mode,
                qwen_asr_model=provider.qwen_asr_model,
            ),
        )
        self._set_unit_card_value_text(
            self._peer_stt_text,
            self._stt_provider_display_label(
                self._effective_peer_stt_provider(provider),
                custom_mode=provider.custom_stt_mode,
                qwen_asr_model=provider.qwen_asr_model,
            ),
        )
        self._update_api_visibility()
        self._sync_gpu_device_card()

        # LLM Provider
        self._set_unit_card_value_text(
            self._llm_text,
            self._get_llm_display_label(provider),
        )
        self._set_translation_connection_text(
            self._get_translation_connection_display_label(provider),
        )
        self._sync_translation_connection_title(provider)
        self._sync_openrouter_fallback_card(provider)
        self._local_llm_base_url.value = provider.local_llm_base_url
        self._local_llm_base_url.error = None
        self._local_llm_model.value = provider.local_llm_model
        self._local_llm_model.error = None
        self._local_llm_extra_body.value = provider.local_llm_extra_body_json
        self._clear_local_llm_extra_body_error()
        self._sync_custom_stt_card(provider)

        # Qwen Region
        region_label = t(f"region.{provider.qwen_region.value}")
        _set_text_button_label(self._qwen_region_btn, f"{t('settings.qwen_region')} {region_label}")

        # Audio Settings
        self._audio_settings.host_api = general.input_host_api
        self._audio_settings.microphone = general.input_device
        self._audio_settings.desktop_output_device = general.output_device
        self._sync_general_audio_card_texts()

        # VAD
        self._vad_slider.value = general.self_vad_speech_threshold
        self._vad_slider.label = f"{general.self_vad_speech_threshold:.2f}"
        self._peer_vad_slider.value = general.peer_vad_speech_threshold
        self._peer_vad_slider.label = f"{general.peer_vad_speech_threshold:.2f}"
        self._peer_vad_field.value = f"{general.peer_vad_speech_threshold:.2f}"
        self._peer_hangover_field.value = str(general.peer_vad_hangover_ms)
        self._peer_pre_roll_field.value = str(general.peer_vad_pre_roll_ms)
        # --- 新增：读取 VRChat 同步开关状态 ---
        self._vrc_mic_text.content.value = t(
            "settings.vrc_mic.on" if general.vrc_mic_intercept else "settings.vrc_mic.off"
        )
        self._sync_osc_connection_card(general)
        self._chatbox_source_text.content.value = t(
            "settings.chatbox_source.on"
            if general.chatbox_include_source
            else "settings.chatbox_source.off"
        )
        self._clipboard_auto_translate_text.content.value = t(
            "settings.clipboard_auto_translate.on"
            if general.clipboard_auto_translate_enabled
            else "settings.clipboard_auto_translate.off"
        )
        self._sync_telemetry_enabled_card(general)
        # Prompt
        provider_name = self._active_prompt_key()
        self._prompt_editor.set_provider(provider_name)
        if prompt.system_prompt.strip():
            self._prompt_editor.value = prompt.system_prompt
        else:
            self._prompt_editor.load_default_prompt(emit_change=False)
            self._prompt_snapshot = replace(prompt, system_prompt=self._prompt_editor.value)

        _ = preserve_custom_vocab_draft
        self._sync_custom_vocabulary_editor_from_settings()
        self._sync_prompt_tab_copy()
        self._overlay_peer_contract = None
        self._sync_overlay_controls()
        self.set_overlay_calibration(
            OverlayCalibration(
                anchor=overlay.calibration.anchor,
                distance=overlay.calibration.distance,
                offset_x=overlay.calibration.offset_x,
                offset_y=overlay.calibration.offset_y,
                text_scale=overlay.calibration.text_scale,
            ),
            preserve_draft=self._overlay_calibration_session_active,
        )

        # Load secrets
        self._load_secrets(provider, config_path)

        if is_control_mounted(self):
            self.update()

    def refresh_after_openrouter_pkce_success(
        self,
        *,
        provider: ProviderSettingsSnapshot,
        prompt: PromptSettingsSnapshot,
        config_path: Path,
    ) -> None:
        self._provider_snapshot = provider
        self._provider_draft = None
        self._provider_edits.clear()
        self._prompt_snapshot = prompt
        self._config_path = config_path
        self.has_provider_changes = False
        self.has_pending_prompt_changes = False
        self._desktop_overlay_pending_size_preset = None
        self._desktop_overlay_pending_position_reset = False
        self._desktop_overlay_pending_locked = None
        self._desktop_overlay_captions_locked = False

        self._set_unit_card_value_text(
            self._llm_text,
            self._get_llm_display_label(provider),
        )
        self._set_translation_connection_text(
            self._get_translation_connection_display_label(provider),
        )
        self._sync_openrouter_fallback_card(provider)
        self._update_api_visibility()

        provider_name = self._active_prompt_key()
        self._prompt_editor.set_provider(provider_name)
        if prompt.system_prompt.strip():
            self._prompt_editor.value = prompt.system_prompt
        else:
            self._prompt_editor.load_default_prompt(emit_change=False)
            self._prompt_snapshot = replace(prompt, system_prompt=self._prompt_editor.value)
        self._sync_custom_vocabulary_editor_from_settings()
        self._sync_prompt_tab_copy()

        result = (
            self._settings_secrets.load_openrouter_pkce()
            if self._settings_secrets is not None
            else None
        )
        if result is not None:
            if result.error_message is not None:
                self._emit_runtime_basic(result.error_message, level=logging.WARNING)
            snapshot = result.snapshot
            if snapshot is not None:
                if snapshot.openrouter_api_key is not None:
                    self._openrouter_key.value = snapshot.openrouter_api_key
                if snapshot.deepseek_api_key is not None:
                    self._deepseek_key.value = snapshot.deepseek_api_key
                if snapshot.cerebras_api_key is not None:
                    self._cerebras_key.value = snapshot.cerebras_api_key
            if result.read_error is not None:
                raise result.read_error
            if snapshot is not None:
                self._restore_api_key_icons(provider)

        if is_control_mounted(self):
            self.update()

    def sync_telemetry_settings(self, settings: GeneralSettingsSnapshot) -> None:
        self._general_snapshot = settings
        self._sync_telemetry_enabled_card(settings)
        if is_control_mounted(self):
            _update_control_if_mounted(self._telemetry_enabled_card)

    def project_osc_control_state(self, state: OscControlPresentationState) -> None:
        control = state.changed_control
        if control in {"PuriPuly_Talk", "PuriPuly_Trans"}:
            return

        if control == "PuriPuly_SelfSrcLang" and self._prompt_snapshot is not None:
            self._prompt_snapshot = replace(
                self._prompt_snapshot,
                source_language=state.self_source_language,
                custom_vocabulary_enabled=state.custom_vocabulary_enabled,
                custom_vocabulary_terms=state.custom_vocabulary_terms,
                custom_vocabulary_other_languages_have_terms=(
                    state.custom_vocabulary_other_languages_have_terms
                ),
            )
            self._custom_vocab_tag_editor.set_terms(list(state.custom_vocabulary_terms))
        if control in {"PuriPuly_PeerAuto", "PuriPuly_PeerSrcLang"}:
            if self._general_snapshot is not None:
                self._general_snapshot = replace(
                    self._general_snapshot,
                    effective_peer_source_language=state.peer_source_language,
                )
        if control in {
            "PuriPuly_PeerAuto",
            "PuriPuly_SelfSrcLang",
            "PuriPuly_SelfDstLang",
            "PuriPuly_SelfDstLang2",
            "PuriPuly_PeerSrcLang",
            "PuriPuly_PeerDstLang",
        }:
            return

        if self._provider_snapshot is not None:
            self._provider_snapshot = self._project_osc_provider_snapshot(
                self._provider_snapshot,
                state,
            )
        if self._provider_draft is not None:
            self._provider_draft = self._project_osc_provider_snapshot(
                self._provider_draft,
                state,
            )
            self._rebase_provider_edits_after_osc(control)
        if self._general_snapshot is not None:
            if control == "PuriPuly_MuteSync":
                self._general_snapshot = replace(
                    self._general_snapshot,
                    vrc_mic_intercept=state.mute_sync,
                )
            elif control == "PuriPuly_ChatboxSource":
                self._general_snapshot = replace(
                    self._general_snapshot,
                    chatbox_include_source=state.chatbox_source,
                )

        display_settings = self._build_settings_with_provider_draft()
        if display_settings is None:
            return
        if control in {"PuriPuly_SelfASR", "PuriPuly_PeerASR"}:
            self._sync_stt_provider_labels(display_settings)
            self._sync_custom_stt_card(display_settings)
            self._update_api_visibility()
            _update_control_if_mounted(self._stt_text)
            _update_control_if_mounted(self._peer_stt_text)
            _update_control_if_mounted(self._api_keys_column)
        elif control == "PuriPuly_Translator":
            self._set_unit_card_value_text(
                self._llm_text,
                self._get_llm_display_label(display_settings),
            )
            self._set_translation_connection_text(
                self._get_translation_connection_display_label(display_settings)
            )
            self._sync_translation_connection_title(display_settings)
            self._update_api_visibility()
            _update_control_if_mounted(self._llm_text)
            _update_control_if_mounted(self._translation_connection_row)
            _update_control_if_mounted(self._api_keys_column)
        elif control == "PuriPuly_Fallback":
            self._sync_openrouter_fallback_card(display_settings)
            self._update_api_visibility()
            _update_control_if_mounted(self._openrouter_fallback_text)
            _update_control_if_mounted(self._openrouter_fallback_helper_text)
            _update_control_if_mounted(self._api_keys_column)
        elif control == "PuriPuly_MuteSync":
            self._vrc_mic_text.content.value = t(
                "settings.vrc_mic.on" if state.mute_sync else "settings.vrc_mic.off"
            )
            _update_control_if_mounted(self._vrc_mic_text)
        elif control == "PuriPuly_ChatboxSource":
            self._chatbox_source_text.content.value = t(
                "settings.chatbox_source.on"
                if state.chatbox_source
                else "settings.chatbox_source.off"
            )
            _update_control_if_mounted(self._chatbox_source_text)
        elif control in {"PuriPuly_Listen", "PuriPuly_Captions"}:
            self._sync_overlay_controls()

    @staticmethod
    def _project_osc_provider_snapshot(
        snapshot: ProviderSettingsSnapshot,
        state: OscControlPresentationState,
    ) -> ProviderSettingsSnapshot:
        control = state.changed_control
        if control in {"PuriPuly_SelfASR", "PuriPuly_PeerASR"}:
            if control == "PuriPuly_SelfASR":
                return replace(
                    snapshot,
                    stt_provider=STTProviderName(state.self_asr_setting),
                    custom_stt_mode=state.custom_stt_mode,
                    custom_stt_compatibility=state.custom_stt_compatibility,
                )
            return replace(
                snapshot,
                peer_stt_provider=STTProviderName(state.peer_asr_setting),
                custom_stt_mode=state.custom_stt_mode,
                custom_stt_compatibility=state.custom_stt_compatibility,
            )
        if control == "PuriPuly_Translator":
            translation = replace(
                snapshot.translation,
                model=TranslationModel(state.translation_model),
                connection=TranslationConnection(state.translation_connection),
                connection_history=tuple(
                    (TranslationModel(model), TranslationConnection(connection))
                    for model, connection in state.translation_connection_history
                ),
                http_extension_id=state.translation_http_extension_id,
                previous_llm_model=(
                    None
                    if state.translation_previous_model is None
                    else TranslationModel(state.translation_previous_model)
                ),
            )
            return replace(
                snapshot,
                llm_provider=LLMProviderName(state.llm_provider),
                translation=translation,
                openrouter_llm_model=OpenRouterLLMModel(state.openrouter_llm_model),
                openrouter_selected_source=OpenRouterCredentialSource(
                    state.openrouter_selected_source
                ),
                openrouter_selection_alias=(
                    None
                    if state.openrouter_selection_alias is None
                    else OpenRouterSelectionAlias(state.openrouter_selection_alias)
                ),
            )
        if control == "PuriPuly_Fallback":
            fallback = TranslationFallbackSnapshot(
                enabled=state.fallback_enabled,
                model=TranslationModel(state.fallback_model),
                connection=TranslationConnection(state.fallback_connection),
            )
            return replace(
                snapshot,
                translation=replace(snapshot.translation, fallback=fallback),
            )
        return snapshot

    def _rebase_provider_edits_after_osc(self, control: str) -> None:
        draft = self._provider_draft
        if draft is None:
            return
        if control == "PuriPuly_SelfASR" and SelfSttProviderEdit in self._provider_edits:
            self._record_provider_edit(SelfSttProviderEdit(draft.stt_provider))
        if control == "PuriPuly_PeerASR" and PeerSttProviderEdit in self._provider_edits:
            self._record_provider_edit(PeerSttProviderEdit(draft.peer_stt_provider))
        if control == "PuriPuly_Translator" and (TranslationSelectionEdit in self._provider_edits):
            self._record_provider_edit(self._translation_selection_edit(draft.translation))
        if control == "PuriPuly_Fallback" and TranslationFallbackEdit in self._provider_edits:
            self._record_provider_edit(TranslationFallbackEdit(draft.translation.fallback))

    def _load_secrets(self, settings: ProviderSettingsSnapshot, config_path: Path) -> None:
        """Load secret values into fields."""
        _ = config_path
        result = self._load_secret_snapshot()
        if result is None or result.snapshot is None:
            return
        snapshot = result.snapshot

        if snapshot.google_api_key is not None:
            self._google_key.value = snapshot.google_api_key
        if snapshot.openrouter_api_key is not None:
            self._openrouter_key.value = snapshot.openrouter_api_key
        if snapshot.deepseek_api_key is not None:
            self._deepseek_key.value = snapshot.deepseek_api_key
        if snapshot.cerebras_api_key is not None:
            self._cerebras_key.value = snapshot.cerebras_api_key
        if snapshot.deepgram_api_key is not None:
            self._deepgram_key.value = snapshot.deepgram_api_key
        if snapshot.gemini_transcribe_api_key is not None:
            gemini_transcribe_key = getattr(self, "_gemini_transcribe_key", None)
            if gemini_transcribe_key is not None:
                gemini_transcribe_key.value = snapshot.gemini_transcribe_api_key
        if snapshot.elevenlabs_scribe_api_key is not None:
            elevenlabs_scribe_key = getattr(self, "_elevenlabs_scribe_key", None)
            if elevenlabs_scribe_key is not None:
                elevenlabs_scribe_key.value = snapshot.elevenlabs_scribe_api_key
        if snapshot.soniox_api_key is not None:
            self._soniox_key.value = snapshot.soniox_api_key
        if snapshot.local_llm_api_key is not None:
            self._local_llm_api_key.value = snapshot.local_llm_api_key
        if snapshot.custom_stt_api_key is not None:
            self._custom_stt_api_key.value = snapshot.custom_stt_api_key
        if (
            snapshot.alibaba_api_key_beijing is not None
            and snapshot.alibaba_api_key_singapore is not None
        ):
            self._alibaba_key_beijing.value = snapshot.alibaba_api_key_beijing
            self._alibaba_key_singapore.value = snapshot.alibaba_api_key_singapore

        if result.read_error is not None:
            raise result.read_error

        # Restore verification status icons from saved settings
        self._restore_api_key_icons(settings)

    def _load_secret_snapshot(
        self,
    ) -> SettingsSecretLoadResult[SettingsSecretSnapshot] | None:
        if self._settings_secrets is None:
            return None
        result = self._settings_secrets.load()
        if result.error_message is not None:
            self._emit_runtime_basic(result.error_message, level=logging.WARNING)
        return result

    def _restore_api_key_icons(self, settings: ProviderSettingsSnapshot) -> None:
        """Restore API key field icons based on saved verification status."""
        verified = settings.verified

        # Map field -> (has_key, is_verified)
        gemini_transcribe_key = getattr(self, "_gemini_transcribe_key", None)
        elevenlabs_scribe_key = getattr(self, "_elevenlabs_scribe_key", None)
        field_map = [
            (self._deepgram_key, self._deepgram_key.value, verified.deepgram),
            (
                gemini_transcribe_key,
                gemini_transcribe_key.value if gemini_transcribe_key else None,
                verified.gemini_transcribe,
            ),
            (
                elevenlabs_scribe_key,
                elevenlabs_scribe_key.value if elevenlabs_scribe_key else None,
                verified.elevenlabs_scribe,
            ),
            (self._soniox_key, self._soniox_key.value, verified.soniox),
            (self._google_key, self._google_key.value, verified.google),
            (self._openrouter_key, self._openrouter_key.value, verified.openrouter),
            (self._deepseek_key, self._deepseek_key.value, verified.deepseek),
            (self._cerebras_key, self._cerebras_key.value, verified.cerebras),
            (self._alibaba_key_beijing, self._alibaba_key_beijing.value, verified.alibaba_beijing),
            (
                self._alibaba_key_singapore,
                self._alibaba_key_singapore.value,
                verified.alibaba_singapore,
            ),
        ]
        field_map = [
            (field, has_key, is_verified)
            for field, has_key, is_verified in field_map
            if field is not None
        ]

        for field, has_key, is_verified in field_map:
            if not has_key:
                field._set_status("idle")
                field._last_verified_hash = ""
            elif is_verified:
                field._set_status("success")
                # Restore hash to prevent re-verification on blur
                field._last_verified_hash = field._get_key_hash(has_key)
            else:
                field._set_status("error")
                field._last_verified_hash = ""
        self._sync_openrouter_pkce_button_state(settings)

    def _sync_openrouter_pkce_button_state(
        self,
        settings: ProviderSettingsSnapshot | None = None,
    ) -> None:
        if settings is None:
            settings = self._build_settings_with_provider_draft()
        authenticated = bool(
            settings is not None and settings.verified.openrouter and self._openrouter_key.value
        )
        _set_text_button_label(
            self._openrouter_pkce_button,
            t(
                "settings.openrouter_authenticated"
                if authenticated
                else "settings.openrouter_authenticate"
            ),
        )
        self._openrouter_pkce_button.disabled = authenticated
        self._openrouter_pkce_button.style = self._get_button_style(
            font_for_language(get_locale()),
            default_color=COLOR_NEUTRAL_DARK,
            disabled_color=COLOR_NEUTRAL_DARK,
        )
        if is_control_mounted(self._openrouter_pkce_button):
            self._openrouter_pkce_button.update()

    # --- Visibility Updates ---
    def _sync_managed_trial_usage_bar(
        self,
        settings: ProviderSettingsSnapshot | None = None,
    ) -> None:
        if settings is None:
            settings = self._build_settings_with_provider_draft()
        managed_key_visible = self._managed_key_card_visible_for(settings)
        self._managed_trial_usage_bar.visible = managed_key_visible
        self._managed_trial_usage_bar.set_percent(
            self._managed_trial_usage_remaining_percent
            if managed_key_visible and self._managed_trial_usage_visible
            else None
        )

    def _update_api_visibility(self) -> None:
        """Update API key field visibility based on selected providers."""
        settings = self._build_settings_with_provider_draft()
        if settings is None:
            return

        stt = settings.stt_provider
        llm = settings.llm_provider
        is_custom_http = settings.translation.model == TranslationModel.CUSTOM_HTTP
        peer_stt = self._effective_peer_stt_provider(settings)
        fallback = settings.translation.fallback
        fallback_source = self._openrouter_fallback_source(settings)
        active_stt_providers = {stt, peer_stt}
        self._deepgram_key.visible = STTProviderName.DEEPGRAM in active_stt_providers
        gemini_transcribe_key = getattr(self, "_gemini_transcribe_key", None)
        if gemini_transcribe_key is not None:
            gemini_transcribe_key.visible = (
                STTProviderName.GEMINI_TRANSCRIBE in active_stt_providers
            )
        elevenlabs_scribe_key = getattr(self, "_elevenlabs_scribe_key", None)
        if elevenlabs_scribe_key is not None:
            elevenlabs_scribe_key.visible = (
                STTProviderName.ELEVENLABS_SCRIBE in active_stt_providers
            )
        self._soniox_key.visible = STTProviderName.SONIOX in active_stt_providers
        peer_auto_languages_card = getattr(self, "_peer_auto_languages_card", None)
        if peer_auto_languages_card is not None:
            peer_auto_languages_card.visible = peer_stt == STTProviderName.SONIOX
            self._sync_peer_auto_languages_editor()
            if is_control_mounted(self):
                try:
                    peer_auto_languages_card.update()
                except Exception:
                    pass

        self._google_key.visible = not is_custom_http and llm == LLMProviderName.GEMINI
        self._sync_managed_key_card(settings)
        if is_custom_http:
            self._managed_key_card.visible = False
            self._managed_trial_usage_bar.visible = False
        if hasattr(self, "_http_extension_credentials"):
            self._http_extension_credentials.visible = is_custom_http
            _update_control_if_mounted(self._http_extension_credentials)
        openrouter_byok_selected = bool(
            not is_custom_http
            and llm == LLMProviderName.OPENROUTER
            and settings.openrouter_selected_source == OpenRouterCredentialSource.BYOK
        )
        self._openrouter_key.visible = bool(
            not is_custom_http
            and (openrouter_byok_selected or fallback_source == OpenRouterCredentialSource.BYOK)
        )
        self._openrouter_pkce_button_row.visible = openrouter_byok_selected
        uses_provider_fallback = self._translation_uses_provider_fallback(settings)
        self._deepseek_key.visible = bool(
            not is_custom_http
            and (
                llm == LLMProviderName.DEEPSEEK
                or (
                    uses_provider_fallback
                    and fallback.enabled
                    and fallback.model == TranslationModel.DEEPSEEK_V4_FLASH
                    and fallback.connection == TranslationConnection.OFFICIAL_BYOK
                )
            )
        )
        self._cerebras_key.visible = bool(
            not is_custom_http
            and (
                llm == LLMProviderName.CEREBRAS
                or (
                    uses_provider_fallback
                    and fallback.enabled
                    and fallback.model == TranslationModel.GEMMA4_31B
                    and fallback.connection == TranslationConnection.CEREBRAS
                )
            )
        )
        self._sync_openrouter_pkce_button_state(settings)
        self._translation_connection_row.visible = (
            not is_custom_http
            and settings.translation.model
            not in {TranslationModel.MANAGED_GEMMA, TranslationModel.MANAGED_GEMMA_12B}
        )
        self._local_llm_connection_card.visible = (
            not is_custom_http and llm == LLMProviderName.LOCAL_LLM
        )
        custom_stt_card = getattr(self, "_custom_stt_connection_card", None)
        if custom_stt_card is not None:
            custom_stt_card.visible = any(
                is_custom_stt_provider(provider) for provider in active_stt_providers
            )
            if custom_stt_card.visible:
                self._sync_custom_stt_card(settings)
        self._sync_openrouter_fallback_card(settings)
        openrouter_fallback_card = getattr(self, "_openrouter_fallback_card", None)
        if openrouter_fallback_card is not None:
            openrouter_fallback_card.visible = self._translation_uses_provider_fallback(settings)
        self._sync_http_extension_card(settings)

        qwen_regions: set[QwenRegion] = set()
        if (
            is_qwen_cloud_stt_provider(stt)
            or (not is_custom_http and llm == LLMProviderName.QWEN)
            or is_qwen_cloud_stt_provider(peer_stt)
        ):
            qwen_regions.add(settings.qwen_region)

        qwen_stt_selected = is_qwen_cloud_stt_provider(stt) or is_qwen_cloud_stt_provider(peer_stt)
        self._qwen_region_btn.visible = qwen_stt_selected or (
            not is_custom_http and llm == LLMProviderName.QWEN
        )
        self._alibaba_key_beijing.visible = QwenRegion.BEIJING in qwen_regions
        self._alibaba_key_singapore.visible = QwenRegion.SINGAPORE in qwen_regions
        api_keys_card = getattr(self, "_api_keys_card", None)
        if api_keys_card is not None:
            api_keys_card.visible = any(
                getattr(control, "visible", False)
                for control in (
                    self._deepgram_key,
                    getattr(self, "_gemini_transcribe_key", None),
                    getattr(self, "_elevenlabs_scribe_key", None),
                    self._soniox_key,
                    self._google_key,
                    self._deepseek_key,
                    self._cerebras_key,
                    self._alibaba_key_beijing,
                    self._alibaba_key_singapore,
                    self._openrouter_pkce_button_row,
                    self._qwen_region_btn,
                    getattr(self, "_http_extension_credentials", None),
                )
                if control is not None
            )

    # --- Event Handlers ---
    def _on_stt_click(self, e) -> None:
        """Open STT provider selection modal."""
        if not is_control_mounted(self):
            return
        ordered_providers = [
            provider
            for section_key in _STT_SECTION_ORDER
            for provider in _STT_UI_PROVIDERS
            if _STT_SECTION_BY_PROVIDER.get(provider) == section_key
        ]
        options = [self._classified_stt_option_item(provider) for provider in ordered_providers]
        display_settings = self._build_settings_with_provider_draft()
        current = (
            self._display_stt_choice(display_settings).value
            if display_settings is not None
            else STTProviderName.LOCAL_CPU_AUTO.value
        )
        modal = SettingsModal(
            self.page,
            t("settings.section.stt"),
            options,
            self._on_stt_selected,
            show_description=True,
            two_column=True,
            left_column_sections=2,
        )
        modal.open(current)

    def _on_stt_selected(self, value: str) -> None:
        """Handle STT provider selection from modal."""
        if self._provider_snapshot is None:
            return
        current_settings = self._build_settings_with_provider_draft()
        assert current_settings is not None
        selected = STTProviderName(value)
        if selected == STTProviderName.LOCAL_CPU_AUTO and not self._local_cpu_auto_available:
            return
        if self._display_stt_choice(current_settings) == selected:
            return
        old_provider = current_settings.stt_provider.value
        self._emit_runtime_basic(
            f"[Settings] STT provider changed: {old_provider} -> {selected.value}"
        )
        draft = self._ensure_provider_settings_draft()
        self._provider_draft = replace(
            draft,
            stt_provider=selected,
        )
        if selected != current_settings.stt_provider:
            self._record_provider_edit(SelfSttProviderEdit(selected))
        if (
            selected == STTProviderName.LOCAL_QWEN_GPU
            and self.on_gpu_discovery_requested is not None
        ):
            self.on_gpu_discovery_requested()
        self._update_api_visibility()
        self._sync_gpu_device_card()
        self.has_provider_changes = True

        self._sync_stt_provider_labels(self._provider_draft)

        source_lang = self._current_source_language()
        selection = resolve_local_asr_selection(selected.value, source_lang)
        if selection.fallback_applied:
            self._show_stt_selection_notice(t("local_stt.language_fallback_qwen"))
        elif not selection.supported:
            self._show_stt_selection_notice(t("local_stt.language_unsupported"))
        warning = (
            None
            if selection.fallback_applied or not selection.supported
            else get_stt_compatibility_warning(
                source_lang,
                selected.value,
            )
        )
        if warning:
            message = t(warning.key, language=language_name(warning.language_code))
            self._show_stt_selection_notice(message)

        if is_control_mounted(self):
            self._qwen_region_btn.update()
            self._api_keys_column.update()
            self._stt_text.update()
            self._peer_stt_text.update()

    def _on_peer_stt_click(self, e) -> None:
        if not is_control_mounted(self):
            return
        ordered_providers = [
            provider
            for section_key in _STT_SECTION_ORDER
            for provider in _STT_UI_PROVIDERS
            if _STT_SECTION_BY_PROVIDER.get(provider) == section_key
        ]
        options = [
            self._classified_peer_stt_option_item(provider) for provider in ordered_providers
        ]
        display_settings = self._build_settings_with_provider_draft()
        current = (
            self._display_stt_choice(display_settings, peer=True).value
            if display_settings is not None
            else STTProviderName.LOCAL_CPU_AUTO.value
        )
        SettingsModal(
            self.page,
            t("settings.section.peer_stt"),
            options,
            self._on_peer_stt_selected,
            show_description=True,
            two_column=True,
            left_column_sections=2,
        ).open(current)

    def _on_peer_stt_selected(self, value: str) -> None:
        if self._provider_snapshot is None:
            return
        current_settings = self._build_settings_with_provider_draft()
        assert current_settings is not None
        selected = STTProviderName(value)
        if selected == STTProviderName.LOCAL_CPU_AUTO and not self._local_cpu_auto_available:
            return
        if self._display_stt_choice(current_settings, peer=True) == selected:
            return
        draft = self._ensure_provider_settings_draft()
        self._provider_draft = replace(
            draft,
            peer_stt_provider=selected,
        )
        if selected != current_settings.peer_stt_provider:
            self._record_provider_edit(PeerSttProviderEdit(selected))
        if (
            selected == STTProviderName.LOCAL_QWEN_GPU
            and self.on_gpu_discovery_requested is not None
        ):
            self.on_gpu_discovery_requested()
        selection = resolve_local_asr_selection(
            selected.value,
            (
                self._general_snapshot.effective_peer_source_language
                if self._general_snapshot is not None
                else "en"
            ),
        )
        if selection.fallback_applied:
            self._show_stt_selection_notice(t("local_stt.language_fallback_qwen"))
        elif not selection.supported:
            self._show_stt_selection_notice(t("local_stt.language_unsupported"))
        self._sync_stt_provider_labels(self._provider_draft)
        self._update_api_visibility()
        self._sync_gpu_device_card()
        if is_control_mounted(self):
            self._peer_stt_text.update()
            self._stt_text.update()
            self._qwen_region_btn.update()
            self._api_keys_column.update()
        self.has_provider_changes = True

    def _sync_stt_provider_labels(self, settings: ProviderSettingsSnapshot) -> None:
        self._set_unit_card_value_text(
            self._stt_text,
            self._stt_provider_display_label(
                settings.stt_provider,
                custom_mode=settings.custom_stt_mode,
                qwen_asr_model=settings.qwen_asr_model,
            ),
        )
        self._set_unit_card_value_text(
            self._peer_stt_text,
            self._stt_provider_display_label(
                self._effective_peer_stt_provider(settings),
                custom_mode=settings.custom_stt_mode,
                qwen_asr_model=settings.qwen_asr_model,
            ),
        )
        stt_rolling_switch = getattr(self, "_stt_rolling_switch", None)
        if stt_rolling_switch is not None:
            stt_rolling_switch.value = settings.stt_rolling_enabled

    def _show_stt_selection_notice(self, message: str) -> None:
        if self.show_snackbar:
            self.show_snackbar(message, ft.Colors.ORANGE_700)
        elif is_control_mounted(self):
            self.page.show_dialog(
                ft.SnackBar(
                    ft.Text(message, color=ft.Colors.WHITE),
                    bgcolor=ft.Colors.ORANGE_700,
                    duration=4000,
                    behavior=ft.SnackBarBehavior.FLOATING,
                    elevation=0,
                    margin=ft.Margin.only(bottom=90),
                    padding=20,
                )
            )

    def _on_llm_click(self, e) -> None:
        """Open LLM provider selection modal."""
        if not is_control_mounted(self):
            return
        options: list[OptionItem] = []
        for section_key in _TRANSLATION_MODEL_SECTION_ORDER:
            for model in _TRANSLATION_MODELS:
                if model == TranslationModel.MANAGED_GEMMA:
                    if section_key == "settings.translation_model.section.recommended_local":
                        options.append(
                            OptionItem(
                                value="managed_gemma_cpu",
                                label=t("provider.managed_gemma_cpu"),
                                description=t(
                                    "settings.translation_model.managed_gemma_cpu.description",
                                    default="",
                                ),
                                section=t(section_key),
                            )
                        )
                    elif section_key == "settings.translation_model.section.gpu_inference":
                        options.append(
                            OptionItem(
                                value="managed_gemma_gpu",
                                label=t("provider.managed_gemma_gpu"),
                                description=t(
                                    "settings.translation_model.managed_gemma_gpu.description",
                                    default="",
                                ),
                                section=t(section_key),
                            )
                        )
                    continue
                if _TRANSLATION_MODEL_SECTION_BY_MODEL.get(model) != section_key:
                    continue
                options.append(
                    OptionItem(
                        value=model.value,
                        label=self._translation_model_display_label(model),
                        description=t(
                            f"settings.translation_model.{model.value}.description",
                            default="",
                        ),
                        section=t(section_key),
                    )
                )
        display_settings = self._build_settings_with_provider_draft()
        current = (
            self._get_llm_modal_value(display_settings)
            if display_settings is not None
            else TranslationModel.GEMMA4.value
        )
        modal = SettingsModal(
            self.page,
            t("settings.section.translation"),
            options,
            self._on_llm_selected,
            show_description=True,
            two_column=True,
            left_column_sections=2,
        )
        modal.open(current)

    def _restore_translation_connection_for_model(
        self,
        model: TranslationModel,
        history: tuple[tuple[TranslationModel, TranslationConnection], ...],
    ) -> TranslationConnection:
        connection = dict(history).get(model)
        if connection in supported_translation_connections(model):
            return connection
        return default_translation_connection(model)

    def _sync_translation_selection_controls(self, settings: ProviderSettingsSnapshot) -> None:
        self._set_unit_card_value_text(
            self._llm_text,
            self._get_llm_display_label(settings),
        )
        self._set_translation_connection_text(
            self._get_translation_connection_display_label(settings),
        )
        self._sync_translation_connection_title(settings)
        self._sync_openrouter_fallback_card(settings)

    def _provider_snapshot_with_translation(
        self,
        settings: ProviderSettingsSnapshot,
        selection: TranslationSelectionSnapshot,
    ) -> ProviderSettingsSnapshot:
        model = selection.model
        connection = selection.connection
        llm_provider = settings.llm_provider
        openrouter_model = settings.openrouter_llm_model
        openrouter_source = settings.openrouter_selected_source
        openrouter_alias = settings.openrouter_selection_alias
        if model == TranslationModel.GEMMA4_26B_31B:
            llm_provider = LLMProviderName.OPENROUTER
            openrouter_model = OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
            openrouter_source = (
                OpenRouterCredentialSource.MANAGED
                if connection == TranslationConnection.MANAGED
                else OpenRouterCredentialSource.BYOK
            )
            openrouter_alias = (
                OpenRouterSelectionAlias.GEMMA4_26B_31B_MANAGED
                if openrouter_source == OpenRouterCredentialSource.MANAGED
                else OpenRouterSelectionAlias.GEMMA4_26B_31B_BYOK
            )
        elif model == TranslationModel.GEMMA4_31B:
            llm_provider = (
                LLMProviderName.CEREBRAS
                if connection == TranslationConnection.CEREBRAS
                else LLMProviderName.OPENROUTER
            )
            if llm_provider == LLMProviderName.OPENROUTER:
                openrouter_model = OpenRouterLLMModel.GEMMA_4_31B_IT
                openrouter_source = (
                    OpenRouterCredentialSource.MANAGED
                    if connection == TranslationConnection.MANAGED
                    else OpenRouterCredentialSource.BYOK
                )
                openrouter_alias = (
                    OpenRouterSelectionAlias.GEMMA4_31B_MANAGED
                    if openrouter_source == OpenRouterCredentialSource.MANAGED
                    else OpenRouterSelectionAlias.GEMMA4_31B_BYOK
                )
        elif model == TranslationModel.GEMMA4:
            llm_provider = LLMProviderName.OPENROUTER
            openrouter_model = OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
            openrouter_source = (
                OpenRouterCredentialSource.MANAGED
                if connection == TranslationConnection.MANAGED
                else OpenRouterCredentialSource.BYOK
            )
            openrouter_alias = (
                OpenRouterSelectionAlias.GEMMA4_MANAGED
                if openrouter_source == OpenRouterCredentialSource.MANAGED
                else OpenRouterSelectionAlias.GEMMA4_BYOK
            )
        elif model == TranslationModel.DEEPSEEK_V4_FLASH:
            llm_provider = (
                LLMProviderName.DEEPSEEK
                if connection == TranslationConnection.OFFICIAL_BYOK
                else LLMProviderName.OPENROUTER
            )
            if llm_provider == LLMProviderName.OPENROUTER:
                openrouter_model = OpenRouterLLMModel.DEEPSEEK_V4_FLASH
                openrouter_source = (
                    OpenRouterCredentialSource.MANAGED
                    if connection
                    in {TranslationConnection.MANAGED, TranslationConnection.MANAGED_CHINA}
                    else OpenRouterCredentialSource.BYOK
                )
                openrouter_alias = (
                    OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED
                    if openrouter_source == OpenRouterCredentialSource.MANAGED
                    else OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_BYOK
                )
        elif model == TranslationModel.GEMINI_37_FLASH:
            llm_provider = (
                LLMProviderName.GEMINI
                if connection == TranslationConnection.OFFICIAL_BYOK
                else LLMProviderName.OPENROUTER
            )
            if llm_provider == LLMProviderName.OPENROUTER:
                openrouter_source = OpenRouterCredentialSource.BYOK
                openrouter_model = OpenRouterLLMModel.GEMINI_37_FLASH
                openrouter_alias = OpenRouterSelectionAlias.GEMINI37_FLASH_BYOK
        elif model == TranslationModel.QWEN_38_FLASH:
            llm_provider = LLMProviderName.QWEN
        elif model in {TranslationModel.MANAGED_GEMMA, TranslationModel.MANAGED_GEMMA_12B}:
            llm_provider = LLMProviderName.MANAGED_GEMMA
        elif model == TranslationModel.LOCAL_LLM:
            llm_provider = LLMProviderName.LOCAL_LLM
        return replace(
            settings,
            llm_provider=llm_provider,
            translation=selection,
            openrouter_llm_model=openrouter_model,
            openrouter_selected_source=openrouter_source,
            openrouter_selection_alias=openrouter_alias,
        )

    def _apply_translation_selection(
        self,
        model: TranslationModel,
        connection: TranslationConnection,
    ) -> None:
        if self._provider_snapshot is None:
            return
        if connection not in supported_translation_connections(model):
            return

        current_settings = self._build_settings_with_provider_draft()
        assert current_settings is not None
        old_model = current_settings.translation.model
        old_connection = current_settings.translation.connection
        old_provider = current_settings.llm_provider
        if old_model == model and old_connection == connection:
            return

        draft = self._ensure_provider_settings_draft()
        history = dict(current_settings.translation.connection_history)
        history[model] = connection
        previous_llm_model = current_settings.translation.previous_llm_model
        if model == TranslationModel.CUSTOM_HTTP and old_model != TranslationModel.CUSTOM_HTTP:
            previous_llm_model = old_model
        elif model != TranslationModel.CUSTOM_HTTP:
            previous_llm_model = None
        selection = replace(
            current_settings.translation,
            model=model,
            connection=connection,
            connection_history=tuple(
                (candidate, history[candidate])
                for candidate in TranslationModel
                if candidate in history
            ),
            previous_llm_model=previous_llm_model,
        )
        self._provider_draft = self._provider_snapshot_with_translation(draft, selection)
        self._record_provider_edit(self._translation_selection_edit(selection))
        new_provider = self._provider_draft.llm_provider

        changes: list[str] = []
        if old_model != model:
            changes.append(f"model={old_model.value}->{model.value}")
        if old_connection != connection:
            changes.append(f"connection={old_connection.value}->{connection.value}")
        if old_provider != new_provider:
            changes.append(f"provider={old_provider.value}->{new_provider.value}")
            self._emit_runtime_basic(
                f"[Settings] LLM provider changed: {old_provider.value} -> {new_provider.value}"
            )
        if changes:
            self._emit_runtime_detailed(
                f"[Settings] Translation selection changed: {', '.join(changes)}"
            )

        self.has_provider_changes = True
        self._update_api_visibility()
        if (
            self._gpu_llm_selected(self._provider_draft)
            and not self._gpu_llm_selected(current_settings)
            and self.on_gpu_discovery_requested is not None
        ):
            self.on_gpu_discovery_requested()
        self._sync_gpu_device_card()

        if (
            connection in (TranslationConnection.MANAGED, TranslationConnection.MANAGED_CHINA)
            or model in {TranslationModel.MANAGED_GEMMA, TranslationModel.MANAGED_GEMMA_12B}
            or old_model in {TranslationModel.MANAGED_GEMMA, TranslationModel.MANAGED_GEMMA_12B}
        ) and getattr(self, "on_providers_changed", None) is not None:
            self.on_providers_changed()

        display_settings = self._build_settings_with_provider_draft()
        assert display_settings is not None
        self._sync_translation_selection_controls(display_settings)

        if old_provider != display_settings.llm_provider:
            provider_name = self._active_prompt_key()
            self._prompt_editor.set_provider(provider_name)
            next_prompt = self._ensure_provider_prompt_value(self._provider_draft, provider_name)
            self._prompt_editor.value = next_prompt
            self._stage_prompt_draft(next_prompt)
        self._sync_prompt_tab_copy()

        if is_control_mounted(self):
            self._qwen_region_btn.update()
            self._repaint_managed_key_card()
            self._llm_text.update()
            self._translation_connection_row.update()
            self._local_llm_connection_card.update()
            http_extension_host = getattr(self, "_http_extension_host", None)
            if http_extension_host is not None:
                http_extension_host.update()

    def _on_llm_selected(self, value: str) -> None:
        """Handle LLM provider selection from modal."""
        if self._provider_snapshot is None:
            return
        current_settings = self._build_settings_with_provider_draft()
        assert current_settings is not None
        if value in ("managed_gemma_cpu", "managed_gemma_gpu"):
            model = TranslationModel.MANAGED_GEMMA
            connection = (
                TranslationConnection.GPU
                if value == "managed_gemma_gpu"
                else TranslationConnection.CPU
            )
        else:
            try:
                model = TranslationModel(value)
            except (TypeError, ValueError):
                if value == LLMProviderName.OPENROUTER.value:
                    model = TranslationModel.GEMMA4
                else:
                    return
            connection = None

        if current_settings.translation.model == model:
            if connection is not None:
                if current_settings.translation.connection == connection:
                    return
            else:
                return
        history = current_settings.translation.connection_history
        if connection is None:
            connection = self._restore_translation_connection_for_model(model, history)
        self._apply_translation_selection(model, connection)

    def _on_translation_connection_click(self, e) -> None:
        if not is_control_mounted(self):
            return
        display_settings = self._build_settings_with_provider_draft()
        model = (
            display_settings.translation.model
            if display_settings is not None
            else TranslationModel.GEMMA4
        )
        if model in {TranslationModel.MANAGED_GEMMA, TranslationModel.MANAGED_GEMMA_12B}:
            return
        connections = supported_translation_connections(model)
        options = [
            OptionItem(
                value=connection.value,
                label=self._translation_connection_display_label(connection),
                description=(
                    self._translation_connection_display_description(connection)
                    if connection == TranslationConnection.CEREBRAS
                    else ""
                ),
            )
            for connection in connections
        ]
        current = (
            display_settings.translation.connection.value
            if display_settings is not None
            else default_translation_connection(model).value
        )
        modal = SettingsModal(
            self.page,
            t("settings.translation_connection"),
            options,
            self._on_translation_connection_selected,
            show_description=True,
        )
        modal.open(current)

    def _on_translation_connection_selected(self, value: str) -> None:
        if self._provider_snapshot is None:
            return
        current_settings = self._build_settings_with_provider_draft()
        assert current_settings is not None
        model = current_settings.translation.model
        try:
            connection = TranslationConnection(value)
        except (TypeError, ValueError):
            return
        if connection not in supported_translation_connections(model):
            return
        self._apply_translation_selection(model, connection)

    def _on_openrouter_fallback_click(self, e) -> None:
        if not is_control_mounted(self):
            return
        display_settings = self._build_settings_with_provider_draft()
        if not self._translation_uses_provider_fallback(display_settings):
            return
        options: list[OptionItem] = [
            OptionItem(
                value=value,
                label=t(label_key),
                description=t(
                    _TRANSLATION_FALLBACK_DESCRIPTION_KEY_BY_VALUE.get(value, ""),
                    default="",
                ),
            )
            for value, _fallback, label_key in _TRANSLATION_FALLBACK_PRESETS
        ]
        display_settings = self._build_settings_with_provider_draft()
        current = "none"
        if display_settings is not None:
            current = self._translation_fallback_preset_value(display_settings.translation.fallback)
            if current == "custom":
                current = "none"
        modal = SettingsModal(
            self.page,
            t("settings.fallback.modal_title"),
            options,
            self._on_openrouter_fallback_selected,
            show_description=True,
        )
        modal.open(current)

    def _on_openrouter_fallback_selected(self, value: str) -> None:
        if self._provider_snapshot is None:
            return

        current_settings = self._build_settings_with_provider_draft()
        assert current_settings is not None
        new_value = _TRANSLATION_FALLBACK_PRESET_BY_VALUE.get(
            value,
            _TRANSLATION_FALLBACK_PRESET_BY_VALUE["none"],
        )

        old_value = current_settings.translation.fallback
        if (
            old_value.enabled == new_value.enabled
            and old_value.model == new_value.model
            and old_value.connection == new_value.connection
        ):
            return

        self._emit_runtime_detailed(
            "[Settings] Fallback selection changed: "
            f"{old_value.enabled}:{old_value.model.value}:{old_value.connection.value}->"
            f"{new_value.enabled}:{new_value.model.value}:{new_value.connection.value}"
        )
        draft = self._ensure_provider_settings_draft()
        translation = replace(current_settings.translation, fallback=new_value)
        self._provider_draft = replace(draft, translation=translation)
        self._record_provider_edit(TranslationFallbackEdit(translation.fallback))
        self.has_provider_changes = True
        self._update_api_visibility()

        display_settings = self._build_settings_with_provider_draft()
        self._sync_openrouter_fallback_card(display_settings)
        if is_control_mounted(self):
            self._api_keys_column.update()
            self._translation_connection_row.update()

        if self.on_providers_changed is not None:
            self.on_providers_changed()

    def _on_ui_click(self, e) -> None:
        """Open UI language selection modal."""
        if not is_control_mounted(self):
            return
        options = [OptionItem(value=code, label=locale_label(code)) for code in available_locales()]
        current = self._general_snapshot.locale if self._general_snapshot else "en"
        modal = SettingsModal(
            self.page,
            t("settings.section.ui"),
            options,
            self._on_ui_selected,
            show_description=False,
        )
        modal.open(current)

    def _on_ui_selected(self, value: str) -> None:
        """Handle UI language selection from modal."""
        if self._general_snapshot is None:
            return
        old_locale = self._general_snapshot.locale
        self._emit_runtime_basic(f"[Settings] Language changed: {old_locale} -> {value}")
        self._general_snapshot = replace(self._general_snapshot, locale=value)

        # Update text
        self._ui_text.content.value = locale_label(value)
        if is_control_mounted(self):
            self._ui_text.update()
        self._emit_settings_changed(LocaleSettingsIntent(value))

    def _on_qwen_region_click(self, e) -> None:
        """Open Qwen region selection modal."""
        if not is_control_mounted(self):
            return
        options = [OptionItem(value=r.value, label=t(f"region.{r.value}")) for r in QwenRegion]
        display_settings = self._build_settings_with_provider_draft()
        current = (
            display_settings.qwen_region.value
            if display_settings is not None
            else QwenRegion.BEIJING.value
        )
        modal = SettingsModal(
            self.page,
            t("settings.qwen_region"),
            options,
            self._on_qwen_region_selected,
            show_description=False,
        )
        modal.open(current)

    def _on_qwen_region_selected(self, value: str) -> None:
        if self._provider_snapshot is None:
            return

        current_settings = self._build_settings_with_provider_draft()
        assert current_settings is not None
        old_region = current_settings.qwen_region.value
        if old_region == value:
            return
        self._emit_runtime_detailed(f"[Settings] Qwen region changed: {old_region} -> {value}")
        draft = self._ensure_provider_settings_draft()
        self._provider_draft = replace(draft, qwen_region=QwenRegion(value))
        self._record_provider_edit(QwenRegionEdit(self._provider_draft.qwen_region))
        self.has_provider_changes = True

        # Update text
        _set_text_button_label(
            self._qwen_region_btn,
            f"{t('settings.qwen_region')} {t(f'region.{value}')}",
        )
        if is_control_mounted(self):
            self._qwen_region_btn.update()

        self._update_api_visibility()
        if is_control_mounted(self):
            self._api_keys_column.update()

    def _on_openrouter_pkce_click(self, _e) -> None:
        settings = self._build_settings_with_provider_draft()
        if settings is None or self.on_request_openrouter_pkce is None:
            return
        if settings.verified.openrouter and self._openrouter_key.value:
            return
        if settings.llm_provider != LLMProviderName.OPENROUTER:
            return
        if settings.openrouter_selected_source != OpenRouterCredentialSource.BYOK:
            return
        profile = self._openrouter_selection_profile(settings)
        if profile is None or profile.openrouter_source != OpenRouterCredentialSource.BYOK.value:
            return
        provider_intent = self._build_provider_apply_intent()
        if provider_intent is None:
            return

        self.on_request_openrouter_pkce(
            OpenRouterPkceTarget(
                OpenRouterSelectionAlias(profile.alias),
                provider_intent=provider_intent,
                system_prompt=self._ensure_provider_prompt_value(settings, "openrouter"),
            )
        )

    def _write_secret_value(self, key: str, value: str) -> bool:
        if (
            self._provider_snapshot is None
            or not self._config_path
            or self._settings_secrets is None
        ):
            return False

        try:
            secret_key = SettingsSecretKey(key)
        except ValueError:
            return False
        result = self._settings_secrets.mutate(SettingsSecretMutation(key=secret_key, value=value))
        if not result.succeeded:
            self._emit_runtime_basic(
                f"Failed to update secret {key}: {result.error_type or 'unknown'}",
                level=logging.WARNING,
            )
            return False
        return True

    def _sync_custom_stt_card(
        self,
        settings: ProviderSettingsSnapshot | None = None,
    ) -> None:
        if getattr(self, "_custom_stt_connection_card", None) is None:
            return
        current = settings or self._build_settings_with_provider_draft()
        if current is None:
            return
        self._custom_stt_endpoint.value = current.custom_stt_endpoint
        self._custom_stt_endpoint.error = None
        self._custom_stt_model.value = current.custom_stt_model
        self._custom_stt_extra.value = current.custom_stt_extra_json
        self._clear_custom_stt_extra_error()
        if is_control_mounted(self):
            _update_control_if_mounted(self._custom_stt_connection_card)

    def _on_custom_stt_field_change(self, e) -> None:
        _ = e
        if self._provider_snapshot is None:
            return
        current = self._build_settings_with_provider_draft()
        if current is None or not (
            is_custom_stt_provider(current.stt_provider)
            or is_custom_stt_provider(current.peer_stt_provider)
        ):
            return
        self._ensure_provider_settings_draft()
        self.has_provider_changes = True

    def _on_custom_stt_endpoint_change_end(self, e) -> None:
        _ = e
        if self._provider_snapshot is None:
            return
        endpoint = (self._custom_stt_endpoint.value or "").strip()
        current = self._provider_draft or self._provider_snapshot
        if current.custom_stt_endpoint != endpoint:
            draft = self._ensure_provider_settings_draft()
            self._provider_draft = replace(draft, custom_stt_endpoint=endpoint)
            self._record_provider_edit(
                CustomSttEndpointEdit(self._provider_draft.custom_stt_endpoint)
            )
            self.has_provider_changes = True
        self._custom_stt_endpoint.value = endpoint
        _update_control_if_mounted(self._custom_stt_endpoint)

    def _on_custom_stt_model_change_end(self, e) -> None:
        _ = e
        if self._provider_snapshot is None:
            return
        model = (self._custom_stt_model.value or "").strip()
        current = self._provider_draft or self._provider_snapshot
        if current.custom_stt_model != model:
            draft = self._ensure_provider_settings_draft()
            self._provider_draft = replace(draft, custom_stt_model=model)
            self._record_provider_edit(CustomSttModelEdit(self._provider_draft.custom_stt_model))
            self.has_provider_changes = True
        self._custom_stt_model.value = model
        _update_control_if_mounted(self._custom_stt_model)

    def _custom_stt_extra_error_message(self, message_key: str, **kwargs: object) -> str:
        if not kwargs:
            return t(message_key)
        template = t(message_key)
        with contextlib.suppress(Exception):
            return template.format(**kwargs)
        return template

    def _show_custom_stt_extra_error(self, message_key: str, **kwargs: object) -> None:
        message = self._custom_stt_extra_error_message(message_key, **kwargs)
        self._custom_stt_extra_error_key = message_key
        self._custom_stt_extra_error_kwargs = dict(kwargs)
        self._custom_stt_extra_error.value = message
        self._custom_stt_extra_error.visible = True
        self._custom_stt_extra.error = message
        _update_control_if_mounted(self._custom_stt_extra)
        _update_control_if_mounted(self._custom_stt_extra_error)

    def _clear_custom_stt_extra_error(self) -> None:
        self._custom_stt_extra_error_key = ""
        self._custom_stt_extra_error_kwargs = {}
        self._custom_stt_extra_error.value = ""
        self._custom_stt_extra_error.visible = False
        self._custom_stt_extra.error = None
        _update_control_if_mounted(self._custom_stt_extra)
        _update_control_if_mounted(self._custom_stt_extra_error)

    def _on_custom_stt_extra_change_end(self, e) -> None:
        _ = e
        if self._provider_snapshot is None:
            return
        raw = (self._custom_stt_extra.value or "").strip()
        try:
            parsed = {} if not raw else json.loads(raw, parse_constant=_reject_json_constant)
        except json.JSONDecodeError:
            self._show_custom_stt_extra_error("settings.custom_stt.extra.invalid_json")
            return
        if not isinstance(parsed, dict):
            self._show_custom_stt_extra_error("settings.custom_stt.extra.must_be_object")
            return
        try:
            normalized = normalize_custom_stt_extra(parsed)
        except CustomSTTConfigurationError as exc:
            self._show_custom_stt_extra_error(
                "settings.custom_stt.extra.rejected_key",
                key=str(exc),
            )
            return
        normalized_text = _custom_stt_extra_to_text(normalized)
        current = self._provider_draft or self._provider_snapshot
        if current.custom_stt_extra_json != normalized_text:
            draft = self._ensure_provider_settings_draft()
            self._provider_draft = replace(draft, custom_stt_extra_json=normalized_text)
            self._record_provider_edit(
                CustomSttExtraEdit(self._provider_draft.custom_stt_extra_json)
            )
            self.has_provider_changes = True
        self._custom_stt_extra.value = normalized_text
        self._clear_custom_stt_extra_error()
        _update_control_if_mounted(self._custom_stt_extra)

    def _on_custom_stt_secret_change(self, key: str, value: str) -> None:
        if key != "custom_stt_api_key":
            return
        stripped = value.strip()
        if not self._write_secret_value(key, stripped):
            if self.show_snackbar:
                self.show_snackbar(t("settings.custom_stt.api_key.save_failed"), ft.Colors.RED_400)
            return
        self._custom_stt_api_key.value = stripped
        from puripuly_heart.core.stt.custom import bump_custom_stt_secret_generation

        bump_custom_stt_secret_generation()
        if self.on_custom_stt_secret_changed:
            self.on_custom_stt_secret_changed()

    def _on_local_llm_secret_change(self, key: str, value: str) -> None:
        if key != "local_llm_api_key":
            return
        stripped = value.strip()
        if not self._write_secret_value(key, stripped):
            if self.show_snackbar:
                self.show_snackbar(t("settings.local_llm.api_key.save_failed"), ft.Colors.RED_400)
            return
        self._local_llm_api_key.value = stripped
        if self.on_local_llm_secret_changed:
            self.on_local_llm_secret_changed()

    def _on_secret_change(self, key: str, value: str) -> object:
        if self._provider_snapshot is None or not self._config_path:
            return False
        if self.on_provider_secret_change is not None:
            return self.on_provider_secret_change(key, value)

        if not self._write_secret_value(key, value):
            return False
        if not value and self.on_secret_cleared:
            with contextlib.suppress(Exception):
                self.on_secret_cleared(key)
        if key == "openrouter_api_key":
            self._sync_openrouter_pkce_button_state()
        return True

    def _on_audio_change(self) -> None:
        if self._general_snapshot is None:
            return

        new_host = self._audio_settings.host_api
        new_device = self._audio_settings.microphone
        new_desktop_output = self._audio_settings.desktop_output_device
        old_host = self._general_snapshot.input_host_api
        old_device = self._general_snapshot.input_device
        old_desktop_output = self._general_snapshot.output_device

        if old_host != new_host:
            self._emit_runtime_basic(f"[Settings] Audio Host changed: {old_host} -> {new_host}")
        if old_device != new_device:
            self._emit_runtime_basic(f"[Settings] Microphone changed: {old_device} -> {new_device}")
        if old_desktop_output != new_desktop_output:
            self._emit_runtime_basic(
                f"[Settings] Desktop loopback output changed: {old_desktop_output} -> {new_desktop_output}"
            )

        self._general_snapshot = replace(
            self._general_snapshot,
            input_host_api=new_host,
            input_device=new_device,
            output_device=new_desktop_output,
        )
        changes = []
        if old_host != new_host or old_device != new_device:
            changes.append(AudioInputSettingsIntent(new_host, new_device))
        if old_desktop_output != new_desktop_output:
            changes.append(DesktopAudioOutputSettingsIntent(new_desktop_output))
        if changes:
            self._emit_settings_changed(AudioSettingsIntent(tuple(changes)))

    def _on_mic_host_api_click(self, e) -> None:
        if not is_control_mounted(self):
            return
        options = self._audio_settings._get_host_api_options()
        modal = SettingsModal(
            self.page,
            t("settings.audio_host_api"),
            options,
            self._on_mic_host_api_selected,
            show_description=False,
        )
        modal.open(self._audio_settings.host_api)

    def _on_mic_host_api_selected(self, value: str) -> None:
        self._audio_settings.host_api = value
        self._audio_settings.microphone = ""
        self._sync_general_audio_card_texts()
        if is_control_mounted(self):
            self._mic_audio_text.update()
            self._audio_host_api_text.update()
        self._on_audio_change()

    def _on_mic_audio_click(self, e) -> None:
        if not is_control_mounted(self):
            return
        options = self._audio_settings._get_microphone_options()
        modal = SettingsModal(
            self.page,
            t("settings.section.microphone_audio"),
            options,
            self._on_mic_audio_selected,
            show_description=False,
        )
        modal.open(self._audio_settings.microphone)

    def _on_mic_audio_selected(self, value: str) -> None:
        self._audio_settings.microphone = value
        self._sync_general_audio_card_texts()
        if is_control_mounted(self):
            self._mic_audio_text.update()
        self._on_audio_change()

    def _on_loopback_audio_click(self, e) -> None:
        if not is_control_mounted(self):
            return
        list_process_options = getattr(self, "on_list_loopback_process_options", None)
        list_device_options = getattr(self, "on_list_loopback_device_options", None)
        list_options = getattr(self, "on_list_loopback_capture_options", None)
        if callable(list_options):
            current = (
                self.on_current_loopback_capture_option()
                if callable(getattr(self, "on_current_loopback_capture_option", None))
                else "device:"
            )
            if callable(list_device_options) and callable(list_process_options):
                device_options = list_device_options()
                process_section = t("settings.desktop_audio.section.process")
                initial_options: list[OptionItem] = [
                    OptionItem(value="", label="", section=process_section),
                    *device_options,
                ]
                modal = SettingsModal(
                    self.page,
                    t("settings.section.loopback_audio"),
                    initial_options,
                    self._on_loopback_audio_selected,
                    show_description=False,
                    two_column=True,
                )
                modal.open(current, loading_section=process_section)
                self._schedule_page_task(self._load_process_capture_options, modal, current)
            else:
                options = list_options()
                modal = SettingsModal(
                    self.page,
                    t("settings.section.loopback_audio"),
                    options,
                    self._on_loopback_audio_selected,
                    show_description=False,
                    two_column=True,
                )
                modal.open(current)
        else:
            options = self._audio_settings._get_desktop_output_options()
            current = self._audio_settings.desktop_output_device
            modal = SettingsModal(
                self.page,
                t("settings.section.loopback_audio"),
                options,
                self._on_loopback_audio_selected,
                show_description=False,
            )
            modal.open(current)

    async def _load_process_capture_options(self, modal: SettingsModal, current: str) -> None:
        list_process_options = getattr(self, "on_list_loopback_process_options", None)
        list_device_options = getattr(self, "on_list_loopback_device_options", None)
        if not callable(list_process_options) or not callable(list_device_options):
            return
        process_options = await asyncio.to_thread(list_process_options)
        device_options = await asyncio.to_thread(list_device_options)
        full_options: list[OptionItem] = [*process_options, *device_options]
        modal.replace_options(full_options)

    def _on_loopback_audio_selected(self, value: str) -> None:
        apply_option = getattr(self, "on_apply_loopback_capture_option", None)
        if callable(apply_option):
            apply_option(value)
            summary = (
                self.on_loopback_capture_summary()
                if callable(getattr(self, "on_loopback_capture_summary", None))
                else value
            )
            if value.startswith("device:"):
                self._audio_settings.desktop_output_device = value[len("device:") :]
            self._set_unit_card_value_text(
                self._loopback_audio_text,
                summary or t("settings.default_option"),
            )
            if is_control_mounted(self):
                self._loopback_audio_text.update()
            return
        self._audio_settings.desktop_output_device = value
        self._sync_general_audio_card_texts()
        if is_control_mounted(self):
            self._loopback_audio_text.update()
        self._on_audio_change()

    def _normalized_overlay_target(self, value: object) -> str:
        return OVERLAY_TARGET_DESKTOP if value == OVERLAY_TARGET_DESKTOP else OVERLAY_TARGET_STEAMVR

    def _current_overlay_target(self) -> str:
        if self._overlay_snapshot is None:
            return OVERLAY_TARGET_STEAMVR
        return self._normalized_overlay_target(self._overlay_snapshot.target)

    def _overlay_target_label_for(self, target: object) -> str:
        normalized_target = self._normalized_overlay_target(target)
        return t(f"settings.overlay.target.{normalized_target}")

    def _sync_overlay_target_control(self) -> None:
        self._set_unit_card_value_text(
            self._overlay_target_button,
            self._overlay_target_label_for(self._current_overlay_target()),
            size=28,
        )
        self._overlay_target_button.disabled = self._overlay_snapshot is None

    def _sync_overlay_target_specific_visibility(self) -> None:
        desktop_selected = self._current_overlay_target() == OVERLAY_TARGET_DESKTOP
        for row in getattr(self, "_overlay_vr_rows", ()):
            row.visible = not desktop_selected
        for row in getattr(self, "_overlay_desktop_rows", ()):
            row.visible = desktop_selected

    @staticmethod
    def _normalize_desktop_overlay_size_preset(value: object) -> str:
        if isinstance(value, str) and value in DESKTOP_FLET_SIZE_PRESET_ORDER:
            return value
        return "medium"

    @staticmethod
    def _normalize_desktop_overlay_background_alpha(value: object) -> float:
        if isinstance(value, bool):
            return DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA
        try:
            alpha = float(value)
        except (TypeError, ValueError):
            return DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA
        if not math.isfinite(alpha):
            return DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA
        return max(0.0, min(1.0, alpha))

    def _desktop_overlay_background_alpha_label_for(self, value: object) -> str:
        alpha = self._normalize_desktop_overlay_background_alpha(value)
        transparency = 1.0 - alpha
        return f"{int(round(transparency * 100))}%"

    def _desktop_overlay_size_label_for(self, size_preset: object) -> str:
        normalized = self._normalize_desktop_overlay_size_preset(size_preset)
        return t(f"settings.overlay.desktop.size.option.{normalized}")

    def _current_desktop_overlay_size_preset(self) -> str:
        pending_size_preset = getattr(self, "_desktop_overlay_pending_size_preset", None)
        if pending_size_preset is not None:
            return pending_size_preset
        if self._overlay_snapshot is None:
            return "medium"
        return self._normalize_desktop_overlay_size_preset(
            self._overlay_snapshot.desktop_size_preset
        )

    def _current_desktop_overlay_background_alpha(self) -> float:
        if self._overlay_snapshot is None:
            return DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA
        return self._normalize_desktop_overlay_background_alpha(
            self._overlay_snapshot.desktop_background_alpha
        )

    def _desktop_overlay_lock_label_for(self, locked: bool) -> str:
        return t(
            "settings.overlay.desktop.lock.value.locked"
            if locked
            else "settings.overlay.desktop.lock.value.move"
        )

    def _current_desktop_overlay_locked(self) -> bool:
        if self._overlay_snapshot is None:
            return False
        if getattr(self, "_desktop_overlay_pending_position_reset", False):
            return False
        if not self._desktop_overlay_runtime_lock_applies():
            return False
        pending_locked = getattr(self, "_desktop_overlay_pending_locked", None)
        if pending_locked is not None:
            return bool(pending_locked)
        return bool(getattr(self, "_desktop_overlay_captions_locked", False))

    def _desktop_overlay_runtime_lock_applies(self) -> bool:
        if getattr(self, "_overlay_state", "off") not in {"connected", "running"}:
            return False
        return (
            self._normalized_overlay_target(
                getattr(self, "_overlay_runtime_target", OVERLAY_TARGET_STEAMVR)
            )
            == OVERLAY_TARGET_DESKTOP
        )

    def _sync_desktop_overlay_main_controls(self) -> None:
        self._set_unit_card_value_text(
            self._desktop_overlay_size_button,
            self._desktop_overlay_size_label_for(self._current_desktop_overlay_size_preset()),
        )
        self._set_unit_card_value_text(
            self._desktop_overlay_lock_button,
            self._desktop_overlay_lock_label_for(self._current_desktop_overlay_locked()),
        )
        self._set_unit_card_value_text(
            self._desktop_overlay_swap_caption_languages_button,
            t(
                "settings.option.on"
                if self._current_desktop_overlay_swap_caption_languages()
                else "settings.option.off"
            ),
        )
        self._desktop_overlay_background_alpha_value_text.value = (
            self._desktop_overlay_background_alpha_label_for(
                self._current_desktop_overlay_background_alpha()
            )
        )
        disabled = self._overlay_snapshot is None
        self._desktop_overlay_size_button.disabled = disabled
        self._desktop_overlay_background_alpha_decrease_button.disabled = disabled
        self._desktop_overlay_background_alpha_increase_button.disabled = disabled
        self._desktop_overlay_lock_button.disabled = disabled
        self._desktop_overlay_swap_caption_languages_button.disabled = disabled
        self._overlay_vr_reset_button.disabled = disabled
        self._overlay_desktop_reset_button.disabled = disabled

    def _desktop_overlay_status_is_visible(self) -> bool:
        return bool(
            self._current_overlay_target() == OVERLAY_TARGET_DESKTOP
            or self._normalized_overlay_target(self._overlay_runtime_target)
            == OVERLAY_TARGET_DESKTOP
        )

    def _desktop_overlay_failure_action_kind(self) -> str:
        if self._overlay_failure_reason in _DESKTOP_OVERLAY_REOPEN_FAILURE_REASONS:
            return "reopen"
        return "retry"

    def _set_desktop_overlay_primary_action(
        self,
        *,
        label_key: str | None,
        action_kind: str | None,
        visible: bool,
    ) -> None:
        self._set_unit_card_value_text(
            self._desktop_overlay_primary_action,
            t(label_key) if label_key else "",
            size=20,
        )
        self._desktop_overlay_primary_action_kind = action_kind
        self._desktop_overlay_primary_action.visible = visible

    def _sync_desktop_overlay_status_control(self) -> None:
        state = self._overlay_state
        desktop_status_visible = self._desktop_overlay_status_is_visible() and state == "failed"
        self._desktop_overlay_status_card.visible = desktop_status_visible
        self._desktop_overlay_recovery_row.visible = desktop_status_visible
        self._desktop_overlay_reason_text.visible = False
        self._desktop_overlay_reason_text.value = ""
        self._desktop_overlay_helper_text.visible = False
        self._desktop_overlay_helper_text.value = ""
        self._desktop_overlay_view_logs_action.visible = False
        self._desktop_overlay_view_logs_action.disabled = False

        if state == "failed":
            self._desktop_overlay_status_title.value = t("settings.overlay.desktop.status.failed")
            action_kind = self._desktop_overlay_failure_action_kind()
            self._desktop_overlay_reason_text.value = t(
                f"settings.overlay.desktop.recovery.message.{action_kind}",
                default=t("settings.overlay.desktop.recovery.message.retry"),
            )
            self._desktop_overlay_reason_text.visible = True
            action_key = (
                "settings.overlay.desktop.recovery.action.reopen"
                if action_kind == "reopen"
                else "settings.overlay.desktop.recovery.action.retry"
            )
            self._set_desktop_overlay_primary_action(
                label_key=action_key,
                action_kind=action_kind,
                visible=True,
            )
            self._desktop_overlay_view_logs_action.visible = True
        else:
            self._desktop_overlay_status_title.value = t(
                "settings.overlay.status.stopping"
                if state == "stopping"
                else "settings.overlay.status.off"
            )
            self._set_desktop_overlay_primary_action(
                label_key=None,
                action_kind=None,
                visible=False,
            )

    def _on_overlay_target_click(self, e) -> None:
        _ = e
        if not is_control_mounted(self) or self._overlay_snapshot is None:
            return
        options = [
            OptionItem(
                value=OVERLAY_TARGET_STEAMVR,
                label=self._overlay_target_label_for(OVERLAY_TARGET_STEAMVR),
            ),
            OptionItem(
                value=OVERLAY_TARGET_DESKTOP,
                label=self._overlay_target_label_for(OVERLAY_TARGET_DESKTOP),
            ),
        ]
        modal = SettingsModal(
            self.page,
            t("settings.overlay.caption_location"),
            options,
            self._on_overlay_target_selected,
            show_description=True,
        )
        modal.open(self._current_overlay_target())

    def _on_overlay_target_selected(self, value: str) -> None:
        if self._overlay_snapshot is None:
            return
        target = self._normalized_overlay_target(value)
        if self._current_overlay_target() == target:
            return
        self._overlay_snapshot = replace(self._overlay_snapshot, target=target)
        if self._overlay_state == "off":
            self._overlay_runtime_target = target
        self._sync_overlay_controls()
        self._emit_settings_changed(OverlayTargetSettingsIntent(self._overlay_snapshot.target))

    def _on_desktop_overlay_size_click(self, e) -> None:
        _ = e
        if (
            not is_control_mounted(self)
            or self._overlay_snapshot is None
            or self._desktop_overlay_size_button.disabled
        ):
            return
        options = [
            OptionItem(
                value=preset,
                label=self._desktop_overlay_size_label_for(preset),
            )
            for preset in DESKTOP_FLET_SIZE_PRESET_DISPLAY_ORDER
        ]
        modal = SettingsModal(
            self.page,
            t("settings.overlay.desktop.size.title"),
            options,
            self._on_desktop_overlay_size_selected,
            show_description=False,
        )
        modal.open(self._current_desktop_overlay_size_preset())

    def _on_desktop_overlay_size_selected(self, value: str) -> None:
        if self._overlay_snapshot is None:
            return
        size_preset = self._normalize_desktop_overlay_size_preset(value)
        if self._current_desktop_overlay_size_preset() == size_preset:
            return
        if self.on_desktop_overlay_size_change:
            self._desktop_overlay_pending_size_preset = size_preset
            self._sync_desktop_overlay_main_controls()
            self.on_desktop_overlay_size_change(size_preset)
            return
        self._overlay_snapshot = replace(
            self._overlay_snapshot,
            desktop_size_preset=size_preset,
        )
        self._desktop_overlay_pending_size_preset = None
        self._sync_desktop_overlay_main_controls()
        self._emit_settings_changed(DesktopOverlaySizeIntent(size_preset))

    def _current_desktop_overlay_swap_caption_languages(self) -> bool:
        if self._overlay_snapshot is None:
            return False
        return self._overlay_snapshot.desktop_swap_caption_languages

    def _on_desktop_overlay_swap_caption_languages_click(self, e) -> None:
        if (
            self._overlay_snapshot is None
            or self._desktop_overlay_swap_caption_languages_button.disabled
        ):
            return
        next_value = "off" if self._current_desktop_overlay_swap_caption_languages() else "on"
        self._on_desktop_overlay_swap_caption_languages_selected(next_value)

    def _on_desktop_overlay_swap_caption_languages_selected(self, value: str) -> None:
        if self._overlay_snapshot is None:
            return
        enabled = value == "on"
        if self._current_desktop_overlay_swap_caption_languages() == enabled:
            self._sync_desktop_overlay_main_controls()
            return
        self._overlay_snapshot = replace(
            self._overlay_snapshot,
            desktop_swap_caption_languages=enabled,
        )
        self._sync_desktop_overlay_main_controls()
        self._emit_settings_changed(DesktopOverlaySwapCaptionLanguagesIntent(enabled))

    def _on_desktop_overlay_lock_click(self, e) -> None:
        _ = e
        if self._overlay_snapshot is None or self._desktop_overlay_lock_button.disabled:
            return
        next_value = "move" if self._current_desktop_overlay_locked() else "locked"
        self._on_desktop_overlay_lock_selected(next_value)

    def _on_desktop_overlay_lock_selected(self, value: str) -> None:
        if self._overlay_snapshot is None:
            return
        locked = value == "locked"
        if self._current_desktop_overlay_locked() == locked:
            return
        if not self._desktop_overlay_runtime_lock_applies():
            self._sync_desktop_overlay_main_controls()
            return
        if self.on_desktop_overlay_lock_change:
            self._desktop_overlay_pending_locked = locked
            self._desktop_overlay_captions_locked = locked
            self._sync_desktop_overlay_main_controls()
            self.on_desktop_overlay_lock_change(locked)
            return
        self._desktop_overlay_pending_locked = locked
        self._desktop_overlay_captions_locked = locked
        self._sync_desktop_overlay_main_controls()

    def _on_desktop_overlay_background_alpha_step(self, delta: float) -> None:
        if (
            self._overlay_snapshot is None
            or self._desktop_overlay_background_alpha_decrease_button.disabled
        ):
            return
        current = self._current_desktop_overlay_background_alpha()
        current_transparency = 1.0 - current
        next_transparency = self._normalize_desktop_overlay_background_alpha(
            round(current_transparency + delta, 2)
        )
        next_alpha = self._normalize_desktop_overlay_background_alpha(
            round(1.0 - next_transparency, 2)
        )
        if current == next_alpha:
            self._sync_desktop_overlay_main_controls()
            if is_control_mounted(self):
                self.update()
            return
        self._overlay_snapshot = replace(
            self._overlay_snapshot,
            desktop_background_alpha=next_alpha,
        )
        self._sync_desktop_overlay_main_controls()
        if is_control_mounted(self):
            self.update()
        self._emit_settings_changed(DesktopOverlayBackgroundAlphaIntent(next_alpha))

    def _on_desktop_overlay_primary_action(self, e) -> None:
        _ = e
        action_kind = self._desktop_overlay_primary_action_kind
        if action_kind == "lock" and self.on_desktop_overlay_lock_change:
            self.on_desktop_overlay_lock_change(True)
        elif action_kind == "edit" and self.on_desktop_overlay_lock_change:
            self.on_desktop_overlay_lock_change(False)
        elif action_kind in {"retry", "reopen"} and self.on_desktop_overlay_recovery_action:
            self.on_desktop_overlay_recovery_action(action_kind)

    def _on_desktop_overlay_view_logs(self, e) -> None:
        _ = e
        if self.on_view_logs:
            self.on_view_logs()

    def set_overlay_calibration(
        self,
        calibration: OverlayCalibration,
        *,
        preserve_draft: bool = False,
    ) -> None:
        calibration.validate()
        self._overlay_calibration = calibration.copy()
        if self._overlay_snapshot is not None:
            self._overlay_snapshot = replace(
                self._overlay_snapshot,
                calibration=OverlayCalibrationSnapshot(
                    anchor=calibration.anchor,
                    distance=calibration.distance,
                    offset_x=calibration.offset_x,
                    offset_y=calibration.offset_y,
                    text_scale=calibration.text_scale,
                ),
            )

        if preserve_draft and self._overlay_calibration_session_active:
            self._sync_overlay_calibration_controls(self._overlay_calibration_draft)
            return

        self._overlay_calibration_draft = calibration.copy()
        self._overlay_calibration_session_active = False
        self._sync_overlay_calibration_controls(self._overlay_calibration)

    def _sync_overlay_calibration_controls(
        self,
        calibration: OverlayCalibration | None = None,
    ) -> None:
        current = (calibration or self._overlay_calibration).copy()
        self._set_unit_card_value_text(
            self._overlay_anchor_button,
            self._overlay_anchor_label_for(current.anchor),
        )
        self._overlay_distance_value_text.value = self._format_overlay_calibration_number(
            current.distance
        )
        self._overlay_offset_x_value_text.value = self._format_overlay_calibration_number(
            current.offset_x
        )
        self._overlay_offset_y_value_text.value = self._format_overlay_calibration_number(
            current.offset_y
        )
        self._overlay_text_scale_text.content.value = self._overlay_text_scale_label_for(
            current.text_scale
        )

    def _begin_overlay_calibration_session(self) -> OverlayCalibration:
        if self._overlay_calibration_session_active:
            return self._overlay_calibration_draft.copy()

        if self.on_overlay_calibration_begin:
            calibration = self.on_overlay_calibration_begin()
        else:
            calibration = self._overlay_calibration.copy()

        calibration.validate()
        self._overlay_calibration_draft = calibration.copy()
        self._overlay_calibration_session_active = True
        self._sync_overlay_calibration_controls(self._overlay_calibration_draft)
        return self._overlay_calibration_draft.copy()

    def _update_overlay_calibration_draft(
        self,
        field_name: str,
        value: object,
    ) -> OverlayCalibration:
        self._begin_overlay_calibration_session()

        if self.on_overlay_calibration_change:
            calibration = self.on_overlay_calibration_change(field_name, value)
            calibration.validate()
            self._overlay_calibration_draft = calibration.copy()
        else:
            if field_name == "anchor":
                setattr(self._overlay_calibration_draft, field_name, str(value))
            else:
                setattr(self._overlay_calibration_draft, field_name, float(value))
            self._overlay_calibration_draft.validate()

        self._sync_overlay_calibration_controls(self._overlay_calibration_draft)
        return self._overlay_calibration_draft.copy()

    def _commit_overlay_calibration_draft(self) -> OverlayCalibration:
        if self.on_overlay_calibration_apply:
            calibration = self.on_overlay_calibration_apply()
            calibration.validate()
        else:
            if not self._overlay_calibration_session_active:
                self._begin_overlay_calibration_session()
            calibration = self._overlay_calibration_draft.copy()

        self._overlay_calibration = calibration.copy()
        self._overlay_calibration_draft = calibration.copy()
        self._overlay_calibration_session_active = False
        calibration_snapshot = OverlayCalibrationSnapshot(
            anchor=calibration.anchor,
            distance=calibration.distance,
            offset_x=calibration.offset_x,
            offset_y=calibration.offset_y,
            text_scale=calibration.text_scale,
        )
        if self._overlay_snapshot is not None:
            self._overlay_snapshot = replace(
                self._overlay_snapshot,
                calibration=calibration_snapshot,
            )
        self._sync_overlay_calibration_controls(self._overlay_calibration)

        if is_control_mounted(self):
            self.update()

        if self.on_overlay_calibration_apply is None:
            self._emit_settings_changed(OverlayCalibrationSettingsIntent(calibration_snapshot))

        return calibration.copy()

    def _apply_overlay_calibration_field_immediately(
        self,
        field_name: str,
        value: object,
    ) -> OverlayCalibration | None:
        try:
            self._update_overlay_calibration_draft(field_name, value)
        except ValueError:
            self._sync_overlay_calibration_controls(self._overlay_calibration)
            return None

        return self._commit_overlay_calibration_draft()

    def _on_overlay_distance_step(self, delta: float) -> None:
        current = self._overlay_calibration.distance
        next_value = max(_OVERLAY_DISTANCE_MIN, min(_OVERLAY_DISTANCE_MAX, current + delta))
        self._apply_overlay_calibration_field_immediately("distance", round(next_value, 2))

    def _on_overlay_anchor_click(self, e) -> None:
        if not is_control_mounted(self) or self._overlay_snapshot is None:
            return
        options = [
            OptionItem(
                value=anchor,
                label=t(f"settings.overlay.calibration.anchor.{anchor}"),
                description=t(
                    f"settings.overlay.calibration.anchor.{anchor}.description",
                    default="",
                ),
            )
            for anchor in OVERLAY_CALIBRATION_ANCHORS
        ]
        modal = SettingsModal(
            self.page,
            t("settings.overlay.calibration.anchor"),
            options,
            self._on_overlay_anchor_selected,
            show_description=True,
        )
        modal.open(self._overlay_calibration.anchor)

    def _on_overlay_anchor_selected(self, value: str) -> None:
        self._apply_overlay_calibration_field_immediately("anchor", value)

    def _on_overlay_offset_x_step(self, delta: float) -> None:
        current = self._overlay_calibration.offset_x
        self._apply_overlay_calibration_field_immediately("offset_x", current + delta)

    def _on_overlay_offset_y_step(self, delta: float) -> None:
        current = self._overlay_calibration.offset_y
        self._apply_overlay_calibration_field_immediately("offset_y", current + delta)

    def _on_overlay_text_scale_click(self, e) -> None:
        if not is_control_mounted(self) or self._overlay_snapshot is None:
            return
        options = [
            OptionItem(
                value=key,
                label=t(f"settings.overlay.calibration.text_scale.{key}"),
            )
            for key, _scale in _OVERLAY_TEXT_SCALE_PRESETS
        ]
        modal = SettingsModal(
            self.page,
            t("settings.overlay.calibration.text_scale"),
            options,
            self._on_overlay_text_scale_selected,
            show_description=False,
        )
        modal.open(self._overlay_text_scale_preset_key_for(self._overlay_calibration.text_scale))

    def _on_overlay_text_scale_selected(self, value: str) -> None:
        self._apply_overlay_calibration_field_immediately(
            "text_scale", self._overlay_text_scale_value_for(value)
        )

    def _on_overlay_position_reset(self, e) -> None:
        _ = e
        defaults = OverlayCalibration()
        for field_name in OverlayCalibration.__dataclass_fields__:
            self._update_overlay_calibration_draft(field_name, getattr(defaults, field_name))
        self._commit_overlay_calibration_draft()

    def _on_desktop_overlay_position_reset(self, e) -> None:
        _ = e
        if self._overlay_snapshot is None or self._overlay_desktop_reset_button.disabled:
            return
        if self.on_desktop_overlay_position_reset:
            self._desktop_overlay_pending_position_reset = True
            self._desktop_overlay_captions_locked = False
            self._sync_desktop_overlay_main_controls()
            self.on_desktop_overlay_position_reset()
            return
        self._desktop_overlay_captions_locked = False
        self._desktop_overlay_pending_position_reset = False
        self._sync_desktop_overlay_main_controls()
        self._emit_settings_changed(DesktopOverlayPositionResetIntent())

    def sync_desktop_overlay_settings(self, settings: OverlaySettingsSnapshot) -> None:
        self._overlay_snapshot = settings
        self._desktop_overlay_pending_size_preset = None
        self._desktop_overlay_pending_position_reset = False
        self._desktop_overlay_pending_locked = None
        self._desktop_overlay_captions_locked = False
        if self._overlay_state == "off":
            self._overlay_runtime_target = self._current_overlay_target()
        self._sync_overlay_controls()

    def set_overlay_peer_contract(self, contract: OverlayPeerConsumerContract) -> None:
        self._overlay_peer_contract = contract
        if self._provider_snapshot is not None:
            self._update_api_visibility()
            if is_control_mounted(self):
                self._api_keys_column.update()
        self._sync_overlay_controls()

    def _sync_overlay_controls(self) -> None:
        overlay_translation_enabled = bool(
            self._overlay_snapshot and self._overlay_snapshot.show_translation
        )
        overlay_peer_original_enabled = bool(
            self._overlay_snapshot and self._overlay_snapshot.show_peer_original
        )
        self._set_unit_card_value_text(
            self._overlay_translation_button,
            t("settings.option.on" if overlay_translation_enabled else "settings.option.off"),
        )
        self._set_unit_card_value_text(
            self._overlay_peer_original_button,
            t("settings.option.on" if overlay_peer_original_enabled else "settings.option.off"),
        )
        self._sync_overlay_target_control()
        self._sync_overlay_target_specific_visibility()
        self._sync_desktop_overlay_main_controls()
        self._sync_desktop_overlay_status_control()

        disabled = self._overlay_snapshot is None
        self._overlay_translation_button.disabled = disabled
        self._overlay_peer_original_button.disabled = disabled
        self._overlay_target_button.disabled = disabled
        self._overlay_anchor_button.disabled = disabled
        self._overlay_distance_decrease_button.disabled = disabled
        self._overlay_distance_increase_button.disabled = disabled
        self._overlay_offset_x_decrease_button.disabled = disabled
        self._overlay_offset_x_increase_button.disabled = disabled
        self._overlay_offset_y_decrease_button.disabled = disabled
        self._overlay_offset_y_increase_button.disabled = disabled
        self._desktop_overlay_background_alpha_decrease_button.disabled = disabled
        self._desktop_overlay_background_alpha_increase_button.disabled = disabled
        self._overlay_vr_reset_button.disabled = disabled
        self._overlay_desktop_reset_button.disabled = disabled
        if is_control_mounted(self):
            self.update()

    def set_overlay_runtime_state(
        self,
        state: str,
        *,
        failure_reason: str | None = None,
        overlay_target: str | None = None,
        desktop_captions_locked: bool | None = None,
    ) -> None:
        self._overlay_state = state
        self._overlay_failure_reason = failure_reason
        if overlay_target is not None:
            self._overlay_runtime_target = self._normalized_overlay_target(overlay_target)
        elif state == "off":
            self._overlay_runtime_target = self._current_overlay_target()
        if desktop_captions_locked is not None:
            if self._desktop_overlay_runtime_lock_applies():
                self._desktop_overlay_pending_locked = None
                self._desktop_overlay_captions_locked = bool(desktop_captions_locked)
            else:
                self._desktop_overlay_pending_locked = None
                self._desktop_overlay_captions_locked = False
        self._sync_overlay_controls()

    def _on_overlay_calibration_reset(self, e) -> None:
        _ = e
        self._begin_overlay_calibration_session()
        self._overlay_calibration_draft = OverlayCalibration()
        self._sync_overlay_calibration_controls(self._overlay_calibration_draft)

        if is_control_mounted(self):
            self.update()

    def _on_overlay_translation_click(self, e) -> None:
        if self._overlay_snapshot is None or self._overlay_translation_button.disabled:
            return
        next_value = "off" if self._overlay_snapshot.show_translation else "on"
        self._on_overlay_translation_selected(next_value)

    def _on_overlay_translation_selected(self, value: str) -> None:
        if self._overlay_snapshot is None:
            return
        self._overlay_snapshot = replace(
            self._overlay_snapshot,
            show_translation=value == "on",
        )
        self._sync_overlay_controls()
        self._emit_settings_changed(
            OverlayTranslationSettingsIntent(self._overlay_snapshot.show_translation)
        )

    def _on_overlay_peer_original_click(self, e) -> None:
        if self._overlay_snapshot is None or self._overlay_peer_original_button.disabled:
            return
        next_value = "off" if self._overlay_snapshot.show_peer_original else "on"
        self._on_overlay_peer_original_selected(next_value)

    def _on_overlay_peer_original_selected(self, value: str) -> None:
        if self._overlay_snapshot is None:
            return
        self._overlay_snapshot = replace(
            self._overlay_snapshot,
            show_peer_original=value == "on",
        )
        self._sync_overlay_controls()
        self._emit_settings_changed(
            OverlayPeerOriginalSettingsIntent(self._overlay_snapshot.show_peer_original)
        )

    def _handle_vad_visual_change(self, e) -> None:
        self._vad_slider.label = f"{float(e.control.value):.2f}"
        _update_control_if_mounted(self._vad_slider)

    def _handle_vad_change(self, e) -> None:
        if self._general_snapshot is None:
            return

        new_vad = float(e.control.value)
        old_vad = self._general_snapshot.self_vad_speech_threshold

        if abs(old_vad - new_vad) > 0.001:
            self._emit_runtime_detailed(
                f"[Settings] VAD sensitivity changed: {old_vad:.2f} -> {new_vad:.2f}"
            )

        self._general_snapshot = replace(
            self._general_snapshot,
            self_vad_speech_threshold=new_vad,
        )
        self._emit_settings_changed(SelfVadSettingsIntent(new_vad))

    def _handle_peer_vad_visual_change(self, e) -> None:
        self._peer_vad_slider.label = f"{float(e.control.value):.2f}"
        _update_control_if_mounted(self._peer_vad_slider)

    def _handle_peer_vad_change(self, e) -> None:
        if self._general_snapshot is None:
            return

        new_vad = float(e.control.value)
        old_vad = self._general_snapshot.peer_vad_speech_threshold

        if abs(old_vad - new_vad) > 0.001:
            self._emit_runtime_detailed(
                f"[Settings] Peer VAD threshold changed: {old_vad:.2f} -> {new_vad:.2f}"
            )

        self._general_snapshot = replace(
            self._general_snapshot,
            peer_vad_speech_threshold=new_vad,
        )
        self._peer_vad_field.value = f"{new_vad:.2f}"
        self._peer_vad_slider.label = f"{new_vad:.2f}"
        _update_control_if_mounted(self._peer_vad_field)
        _update_control_if_mounted(self._peer_vad_slider)
        self._emit_settings_changed(PeerVadSpeechThresholdIntent(new_vad))

    def _on_peer_vad_threshold_change(self, e) -> None:
        if self._general_snapshot is None:
            return

        old_value = self._general_snapshot.peer_vad_speech_threshold
        new_value = self._parse_setting_float(
            e.control.value,
            fallback=old_value,
            minimum=0.0,
            maximum=1.0,
        )
        if abs(old_value - new_value) > 0.001:
            self._emit_runtime_detailed(
                f"[Settings] Peer VAD threshold changed: {old_value:.2f} -> {new_value:.2f}"
            )

        self._general_snapshot = replace(
            self._general_snapshot,
            peer_vad_speech_threshold=new_value,
        )
        self._peer_vad_field.value = f"{new_value:.2f}"
        _update_control_if_mounted(self._peer_vad_field)
        self._emit_settings_changed(PeerVadSpeechThresholdIntent(new_value))

    def _on_peer_hangover_change(self, e) -> None:
        if self._general_snapshot is None:
            return

        old_value = self._general_snapshot.peer_vad_hangover_ms
        new_value = self._parse_setting_int(
            e.control.value,
            fallback=old_value,
            minimum=0,
        )
        if old_value != new_value:
            self._emit_runtime_detailed(
                f"[Settings] Peer hangover changed: {old_value} -> {new_value}"
            )

        self._general_snapshot = replace(
            self._general_snapshot,
            peer_vad_hangover_ms=new_value,
        )
        self._peer_hangover_field.value = str(new_value)
        _update_control_if_mounted(self._peer_hangover_field)
        self._emit_settings_changed(PeerVadHangoverIntent(new_value))

    def _on_peer_pre_roll_change(self, e) -> None:
        if self._general_snapshot is None:
            return

        old_value = self._general_snapshot.peer_vad_pre_roll_ms
        new_value = self._parse_setting_int(
            e.control.value,
            fallback=old_value,
            minimum=0,
        )
        if old_value != new_value:
            self._emit_runtime_detailed(
                f"[Settings] Peer pre-roll changed: {old_value} -> {new_value}"
            )

        self._general_snapshot = replace(
            self._general_snapshot,
            peer_vad_pre_roll_ms=new_value,
        )
        self._peer_pre_roll_field.value = str(new_value)
        _update_control_if_mounted(self._peer_pre_roll_field)
        self._emit_settings_changed(PeerVadPreRollIntent(new_value))

    def _on_vrc_mic_click(self, e) -> None:
        """Toggle VRC mic intercept immediately from the unit card."""
        if self._general_snapshot is None:
            return
        next_value = "off" if self._general_snapshot.vrc_mic_intercept else "on"
        self._on_vrc_mic_selected(next_value)

    def _on_microphone_test_click(self, e) -> None:
        """Request the app/controller-owned microphone-test lifecycle."""
        _ = e
        if self.on_start_microphone_test is not None:
            self.on_start_microphone_test()

    def _on_vrc_mic_selected(self, value: str) -> None:
        """处理选项卡的选择结果

        Handle VRC mic intercept selection result.
        """
        if self._general_snapshot is None:
            return
        new_value = value == "on"
        self._emit_runtime_basic(f"[Settings] VRC mic intercept toggled: {new_value}")
        self._general_snapshot = replace(
            self._general_snapshot,
            vrc_mic_intercept=new_value,
        )

        self._vrc_mic_text.content.value = t(
            "settings.vrc_mic.on" if new_value else "settings.vrc_mic.off"
        )
        if is_control_mounted(self):
            self._vrc_mic_text.update()
        self._emit_settings_changed(VrcMicInterceptSettingsIntent(new_value))

    def _on_chatbox_source_click(self, e) -> None:
        """Open chatbox source inclusion selection modal."""
        if not is_control_mounted(self):
            return
        options = [
            OptionItem(value="on", label=t("settings.chatbox_source.on")),
            OptionItem(value="off", label=t("settings.chatbox_source.off")),
        ]
        if self._general_snapshot is None:
            return
        current = "on" if self._general_snapshot.chatbox_include_source else "off"
        modal = SettingsModal(
            self.page,
            t("settings.chatbox_include_source"),
            options,
            self._on_chatbox_source_selected,
            show_description=False,
        )
        modal.open(current)

    def _on_chatbox_source_selected(self, value: str) -> None:
        """Handle chatbox source inclusion selection result."""
        if self._general_snapshot is None:
            return
        new_value = value == "on"
        self._emit_runtime_basic(f"[Settings] Chatbox include source toggled: {new_value}")
        self._general_snapshot = replace(
            self._general_snapshot,
            chatbox_include_source=new_value,
        )

        self._chatbox_source_text.content.value = t(
            "settings.chatbox_source.on" if new_value else "settings.chatbox_source.off"
        )
        if is_control_mounted(self):
            self._chatbox_source_text.update()
        self._emit_settings_changed(ChatboxSourceSettingsIntent(new_value))

    def _on_osc_connection_click(self, e) -> None:
        _ = e
        if self._general_snapshot is None or not is_control_mounted(self):
            return
        self._osc_connection_modal = OscConnectionModal(
            self.page,
            self._on_osc_connection_selected,
            effective_ports_provider=self.on_osc_effective_ports,
        )
        self._osc_connection_modal.open(
            self._general_snapshot.osc_connection_mode,
            int(self._general_snapshot.osc_send_port or self._general_snapshot.osc_port),
            int(self._general_snapshot.osc_receive_port),
        )

    def _on_osc_connection_selected(self, mode: str, send_port: int, receive_port: int) -> None:
        if self._general_snapshot is None:
            return
        if mode not in {"automatic", "manual", "off"}:
            return
        try:
            send_port = int(send_port)
            receive_port = int(receive_port)
        except (TypeError, ValueError):
            return
        if not 1 <= send_port <= 65535 or not 1 <= receive_port <= 65535:
            return
        self._general_snapshot = replace(
            self._general_snapshot,
            osc_connection_mode=mode,
            osc_send_port=send_port,
            osc_receive_port=receive_port,
        )
        self._sync_osc_connection_card(self._general_snapshot)
        if is_control_mounted(self):
            self._osc_connection_text.update()
        self._emit_settings_changed(
            OscConnectionSettingsIntent(
                connection_mode=self._general_snapshot.osc_connection_mode,
                send_port=self._general_snapshot.osc_send_port,
                receive_port=self._general_snapshot.osc_receive_port,
            )
        )

    def _on_clipboard_auto_translate_click(self, e) -> None:
        """Toggle clipboard auto-translate immediately from the unit card."""
        if self._general_snapshot is None:
            return
        next_value = "off" if self._general_snapshot.clipboard_auto_translate_enabled else "on"
        self._on_clipboard_auto_translate_selected(next_value)

    def _on_clipboard_auto_translate_selected(self, value: str) -> None:
        """Handle clipboard auto-translate selection result."""
        if self._general_snapshot is None:
            return
        new_value = value == "on"
        self._emit_runtime_basic(f"[Settings] Clipboard auto translate toggled: {new_value}")
        self._general_snapshot = replace(
            self._general_snapshot,
            clipboard_auto_translate_enabled=new_value,
        )
        self._clipboard_auto_translate_text.content.value = t(
            "settings.clipboard_auto_translate.on"
            if new_value
            else "settings.clipboard_auto_translate.off"
        )
        if is_control_mounted(self):
            self._clipboard_auto_translate_text.update()
        self._emit_settings_changed(ClipboardSettingsIntent(new_value))

    def _on_telemetry_enabled_click(self, e) -> None:
        _ = e
        if not is_control_mounted(self) or self._general_snapshot is None:
            return

        enabled = not self._general_snapshot.telemetry_enabled
        self._general_snapshot = replace(
            self._general_snapshot,
            telemetry_enabled=enabled,
        )
        self._sync_telemetry_enabled_card(self._general_snapshot)
        if self.on_telemetry_enabled_change is not None:
            self.on_telemetry_enabled_change(enabled)

    def _on_prompt_change(self, value: str) -> None:
        self._prompt_editor.value = value
        self._stage_prompt_draft(value)

    def _on_prompt_commit(self, value: str) -> None:
        if not self.has_pending_prompt_changes and value == self._committed_prompt_value():
            return
        self._stage_prompt_draft(value)
        if self.has_provider_changes:
            return
        pending = self.consume_prompt_apply_settings()
        if pending is None:
            return
        self._emit_prompt_apply_settings(pending)

    def _on_reset_prompt(self, e) -> None:
        """Reset prompt to default for current provider."""
        self._prompt_editor.load_default_prompt()
        self._on_prompt_commit(self._prompt_editor.value)

    def _show_custom_vocabulary_limit_snackbar(self) -> None:
        if self.show_snackbar:
            self.show_snackbar(
                t(
                    "snackbar.custom_vocabulary_limit",
                    max_terms=MAX_CUSTOM_VOCAB_TERMS,
                ),
                ft.Colors.ORANGE_700,
            )

    def _set_custom_vocabulary_terms_for_current_language(self, next_terms: list[str]) -> None:
        if self._prompt_snapshot is None:
            return

        source_language = self._current_source_language()
        current_terms = list(self._prompt_snapshot.custom_vocabulary_terms)
        applied_terms = list(next_terms)
        next_enabled = bool(applied_terms) or (
            self._prompt_snapshot.custom_vocabulary_other_languages_have_terms
        )

        if (
            current_terms == applied_terms
            and self._prompt_snapshot.custom_vocabulary_enabled == next_enabled
        ):
            return

        self._prompt_snapshot = replace(
            self._prompt_snapshot,
            custom_vocabulary_enabled=next_enabled,
            custom_vocabulary_terms=tuple(applied_terms),
        )
        self._custom_vocab_tag_editor.set_terms(applied_terms)
        self._emit_runtime_detailed(
            f"[Settings] Custom vocabulary applied: language={source_language}, terms={len(applied_terms)}"
        )
        self._emit_settings_changed(
            CustomVocabularySettingsIntent(
                source_language=source_language,
                terms=tuple(applied_terms),
            )
        )

    def _on_custom_vocabulary_add_terms(self, raw_terms: list[str]) -> None:
        if self._prompt_snapshot is None:
            return

        raw_values = [str(term) for term in raw_terms]
        if any(value != "" for value in raw_values):
            self._custom_vocab_tag_editor.clear_input()
        submitted_terms = self._normalize_custom_vocabulary_submitted_terms(raw_values)
        if not submitted_terms:
            return

        source_language = self._current_source_language()
        current_terms = list(self._prompt_snapshot.custom_vocabulary_terms)
        next_terms = list(current_terms)
        seen_terms = set(current_terms)
        unique_requested_count = len(current_terms)
        cap_exceeded = False

        for term in submitted_terms:
            if term in seen_terms:
                continue
            seen_terms.add(term)
            unique_requested_count += 1
            if len(next_terms) >= MAX_CUSTOM_VOCAB_TERMS:
                cap_exceeded = True
                continue
            next_terms.append(term)

        next_enabled = bool(next_terms) or (
            self._prompt_snapshot.custom_vocabulary_other_languages_have_terms
        )
        will_change = (
            current_terms != next_terms
            or self._prompt_snapshot.custom_vocabulary_enabled != next_enabled
        )
        if cap_exceeded:
            if will_change:
                self._emit_runtime_detailed(
                    "[Settings] Custom vocabulary capped: "
                    f"language={source_language}, requested={unique_requested_count}, "
                    f"applied={MAX_CUSTOM_VOCAB_TERMS}"
                )
            self._show_custom_vocabulary_limit_snackbar()

        self._set_custom_vocabulary_terms_for_current_language(next_terms)

    def _on_custom_vocabulary_remove_term(self, term: str) -> None:
        if self._prompt_snapshot is None:
            return

        current_terms = list(self._prompt_snapshot.custom_vocabulary_terms)
        try:
            current_terms.remove(term)
        except ValueError:
            return
        self._set_custom_vocabulary_terms_for_current_language(current_terms)

    async def _verify_key(self, provider: str, key: str) -> tuple[bool, str]:
        """Verify API key."""
        if self.on_verify_api_key:
            result = await self.on_verify_api_key(provider, key)
            if provider == "openrouter" and self._provider_snapshot is not None:
                current = self._provider_draft or self._provider_snapshot
                updated = replace(
                    current,
                    verified=replace(current.verified, openrouter=bool(result[0])),
                )
                if self._provider_draft is None:
                    self._provider_snapshot = updated
                else:
                    self._provider_draft = updated
                self._sync_openrouter_pkce_button_state()
            return result
        return False, "Verification not available"

    def _emit_settings_changed(self, intent: ImmediateSettingsIntent) -> None:
        if self.on_settings_changed:
            self.on_settings_changed(intent)

    def _emit_prompt_apply_settings(self, intent: PromptApplyIntent) -> None:
        if self.on_prompt_apply_settings:
            self.on_prompt_apply_settings(intent)

    # --- Locale ---
    def apply_locale(self) -> None:
        """Update all labels when locale changes."""
        self._settings_subtab_shell.set_font_family(font_for_language(get_locale()))
        for key in _SETTINGS_SUBTAB_ORDER:
            self._settings_subtab_shell.set_tab_label(key, self._settings_subtab_label(key))

        # Section titles
        self._stt_title.value = t("settings.section.stt")
        self._stt_rolling_switch.label = t("settings.stt_rolling")
        self._trans_title.value = t("settings.section.translation")
        self._api_title.value = t("settings.section.api_keys")
        self._managed_key_title.value = t("settings.managed_key.title")
        self._managed_key_referral_id_label.value = t("settings.managed_key.referral_id.label")
        self._managed_key_invite_progress_label.value = t(
            "settings.managed_key.invite_progress.label"
        )
        self._stt_provider_label.value = t("settings.self_stt_provider")
        self._translation_provider_label.value = t("settings.shared_translation_provider")
        self._api_credentials_helper_text.value = t("settings.api_credentials_helper")
        self._ui_title.value = t("settings.section.ui")
        self._audio_host_api_title.value = t("settings.audio_host_api")
        self._mic_audio_title.value = t("settings.section.microphone_audio")
        self._loopback_audio_title.value = t("settings.section.loopback_audio")
        self._self_vad_title.value = t("settings.section.self_vad_sensitivity")
        self._peer_vad_title.value = t("settings.section.peer_vad_sensitivity")
        self._microphone_test_title.value = t("settings.microphone_test")
        self._peer_vad_field.label = t("settings.vad.peer")
        self._peer_hangover_field.label = t("settings.vad.peer_hangover_ms")
        self._peer_pre_roll_field.label = t("settings.vad.peer_pre_roll_ms")
        self._translation_connection_title.value = t("settings.translation_connection")
        self._openrouter_fallback_title.value = t("settings.fallback")
        self._local_llm_connection_title.value = t("settings.local_llm.connection")
        self._custom_stt_connection_title.value = t("settings.custom_stt.title")
        self._custom_stt_endpoint.label = t("settings.custom_stt.endpoint")
        self._custom_stt_model.label = t("settings.custom_stt.model")
        self._custom_stt_api_key.apply_locale()
        custom_stt_api_key_description = t("settings.custom_stt.api_key.description")
        self._custom_stt_api_key_helper.value = custom_stt_api_key_description
        self._custom_stt_api_key_helper.visible = bool(custom_stt_api_key_description.strip())
        self._sync_custom_stt_card()
        self._http_extension_title.value = t("settings.http_extension.title")
        self._http_extension_path_title.value = t("settings.http_extension.path")
        self._http_extension_refresh_title.value = t("settings.http_extension.refresh")
        self._set_unit_card_value_text(
            self._http_extension_path_text,
            t("settings.http_extension.open"),
        )
        self._sync_http_extension_card()
        self._local_llm_base_url.label = t("settings.local_llm.base_url")
        self._local_llm_model.label = t("settings.local_llm.model")
        self._local_llm_api_key.apply_locale()
        local_llm_api_key_description = t("settings.local_llm.api_key.description")
        self._local_llm_api_key_helper.value = local_llm_api_key_description
        self._local_llm_api_key_helper.visible = bool(local_llm_api_key_description.strip())
        self._local_llm_extra_body.label = t("settings.local_llm.extra_body")
        self._local_llm_extra_body_helper.value = t("settings.local_llm.extra_body.description")
        if self._local_llm_base_url.error:
            self._local_llm_base_url.error = t("settings.local_llm.base_url.invalid")
        if self._local_llm_model.error:
            self._local_llm_model.error = t("settings.local_llm.model.required")
        if self._local_llm_extra_body_error.visible:
            error_key = self._local_llm_extra_body_error_key
            error_kwargs = self._local_llm_extra_body_error_kwargs
            if error_key:
                message = self._local_llm_extra_body_error_message(error_key, **error_kwargs)
                self._local_llm_extra_body_error.value = message
                self._local_llm_extra_body.error = message
        self._persona_title.value = t("settings.section.persona")
        self._custom_vocab_title.value = t("settings.section.custom_vocabulary")
        self._vrc_mic_title.value = t("settings.vrc_mic_intercept")
        self._osc_connection_title.value = t("settings.osc.connection.title")
        self._chatbox_source_title.value = t("settings.chatbox_include_source")
        self._clipboard_auto_translate_title.value = t("settings.clipboard_auto_translate")
        self._telemetry_enabled_title.value = t("settings.telemetry.title")
        self._peer_provider_title.value = t("settings.section.peer_stt")
        self._dashboard_language_redirect_text.value = t("settings.dashboard_language_redirect")
        self._peer_stt_label.value = t("settings.peer_stt_provider")
        self._gpu_device_title.value = t("settings.gpu_device.asr")
        self._gpu_llm_title.value = t("settings.gpu_device.llm")
        self._gpu_refresh_title.value = t("settings.gpu_device.refresh")
        self._overlay_target_title.value = t("settings.overlay.caption_location")
        self._overlay_translation_title.value = t("settings.overlay.show_translation")
        self._overlay_peer_original_title.value = t("settings.overlay.show_peer_original")
        self._audio_settings.apply_locale()
        self._sync_general_audio_card_texts()
        self._overlay_anchor_title.value = t("settings.overlay.calibration.anchor")
        self._overlay_distance_title.value = t("settings.overlay.calibration.distance")
        self._overlay_offset_x_title.value = t("settings.overlay.calibration.offset_x")
        self._overlay_offset_y_title.value = t("settings.overlay.calibration.offset_y")
        self._overlay_text_scale_title.value = t("settings.overlay.calibration.text_scale")
        self._overlay_vr_reset_title.value = t("settings.overlay.position_reset.vr.title")
        self._overlay_desktop_reset_title.value = t("settings.overlay.position_reset.desktop.title")
        self._desktop_overlay_size_title.value = t("settings.overlay.desktop.size.title")
        self._desktop_overlay_background_alpha_title.value = t(
            "settings.overlay.desktop.background_alpha.title"
        )
        self._desktop_overlay_lock_title.value = t("settings.overlay.desktop.lock.title")
        self._desktop_overlay_swap_caption_languages_title.value = t(
            "settings.overlay.desktop.swap_caption_languages.title"
        )
        self._set_unit_card_value_text(
            self._overlay_vr_reset_button, t("settings.overlay.position_reset.action.vr")
        )
        self._set_unit_card_value_text(
            self._overlay_desktop_reset_button,
            t("settings.overlay.position_reset.action.desktop"),
        )
        _set_text_button_label(self._reset_prompt_btn, t("settings.reset_prompt"))
        self._sync_prompt_tab_copy()

        # Update dynamic buttons by replacing the entire style object
        ui_font = font_for_language(get_locale())
        display_settings = self._build_settings_with_provider_draft()

        if self._reset_prompt_btn:
            self._reset_prompt_btn.style = self._get_button_style(ui_font)

        if self._qwen_region_btn:
            self._qwen_region_btn.style = self._get_button_style(ui_font)
        if self._openrouter_pkce_button:
            self._sync_openrouter_pkce_button_state(display_settings)
        self._sync_clickable_text_control_fonts(ui_font)
        for glyph_text in (
            getattr(self, "_overlay_distance_decrease_glyph", None),
            getattr(self, "_overlay_distance_increase_glyph", None),
            getattr(self, "_overlay_offset_x_decrease_glyph", None),
            getattr(self, "_overlay_offset_x_increase_glyph", None),
            getattr(self, "_overlay_offset_y_decrease_glyph", None),
            getattr(self, "_overlay_offset_y_increase_glyph", None),
            getattr(self, "_desktop_overlay_background_alpha_decrease_glyph", None),
            getattr(self, "_desktop_overlay_background_alpha_increase_glyph", None),
        ):
            if glyph_text:
                glyph_text.font_family = ui_font
                glyph_text.size = 22
        # Update text controls with current selection labels
        if display_settings:
            self._sync_stt_provider_labels(display_settings)
            self._set_unit_card_value_text(
                self._llm_text,
                self._get_llm_display_label(display_settings),
            )
            self._set_translation_connection_text(
                self._get_translation_connection_display_label(display_settings),
            )
            self._sync_translation_connection_title(display_settings)
            self._sync_openrouter_fallback_card(display_settings)
            self._sync_http_extension_card(display_settings, force_credentials=True)
            self._sync_managed_key_card(display_settings)
            self._sync_managed_key_invite_progress_row(
                self._managed_key_referral_id,
                self._managed_key_pass_status,
            )
            if self._general_snapshot is not None:
                self._ui_text.content.value = locale_label(self._general_snapshot.locale)
            self._vrc_mic_text.content.value = t(
                "settings.vrc_mic.on"
                if self._general_snapshot is not None and self._general_snapshot.vrc_mic_intercept
                else "settings.vrc_mic.off"
            )
            if self._general_snapshot is not None:
                self._sync_osc_connection_card(self._general_snapshot)
            self._chatbox_source_text.content.value = t(
                "settings.chatbox_source.on"
                if self._general_snapshot is not None
                and self._general_snapshot.chatbox_include_source
                else "settings.chatbox_source.off"
            )
            self._clipboard_auto_translate_text.content.value = t(
                "settings.clipboard_auto_translate.on"
                if self._general_snapshot is not None
                and self._general_snapshot.clipboard_auto_translate_enabled
                else "settings.clipboard_auto_translate.off"
            )
            if self._general_snapshot is not None:
                self._sync_telemetry_enabled_card(self._general_snapshot)
            self._set_unit_card_value_text(
                self._microphone_test_text,
                t("settings.microphone_test.action"),
            )
            self._sync_overlay_controls()
            self._sync_overlay_calibration_controls()

        # Qwen Region label
        if display_settings:
            region_val = display_settings.qwen_region.value
            _set_text_button_label(
                self._qwen_region_btn,
                f"{t('settings.qwen_region')} {t(f'region.{region_val}')}",
            )

        # Components
        self._deepgram_key.apply_locale()
        self._gemini_transcribe_key.apply_locale()
        self._elevenlabs_scribe_key.apply_locale()
        self._soniox_key.apply_locale()
        self._google_key.apply_locale()
        self._managed_trial_usage_bar.apply_locale()
        self._openrouter_key.apply_locale()
        self._deepseek_key.apply_locale()
        self._cerebras_key.apply_locale()
        self._alibaba_key_beijing.apply_locale()
        self._alibaba_key_singapore.apply_locale()
        self._audio_settings.apply_locale()
        self._prompt_editor.apply_locale()
        self._sync_gpu_device_card()

        if is_control_mounted(self):
            self.update()

    def refresh_prompt_if_empty(self) -> None:
        """Load default prompt if current is empty."""
        was_empty = not self._prompt_editor.value.strip()
        self._prompt_editor.load_default_if_empty()
        if was_empty and self._prompt_editor.value.strip():
            if self._prompt_editor.value != self._committed_prompt_value():
                self._stage_prompt_draft(self._prompt_editor.value)
