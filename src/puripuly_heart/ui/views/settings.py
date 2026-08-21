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
from pathlib import Path
from typing import Callable, Mapping

import flet as ft
from puripuly_heart.app.services.local_asr_selection import resolve_local_asr_selection
from puripuly_heart.core.managed_openrouter_release import TalkTogetherPassStatus

from puripuly_heart.app.services.http_extension_registry import (
    HttpExtensionRegistryService,
)
from puripuly_heart.app.wiring import create_secret_store
from puripuly_heart.config.llm_profiles import (
    profile_for_alias,
)
from puripuly_heart.config.overlay_calibration import (
    OVERLAY_CALIBRATION_ANCHORS,
    OverlayCalibration,
)
from puripuly_heart.config.prompts import load_prompt_for_provider
from puripuly_heart.config.settings import (
    DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA,
    DESKTOP_FLET_SIZE_PRESET_DISPLAY_ORDER,
    DESKTOP_FLET_SIZE_PRESET_ORDER,
    LOCAL_LLM_RESERVED_EXTRA_BODY_KEYS,
    LOCAL_LLM_SENSITIVE_EXTRA_BODY_KEYS,
    MAX_CUSTOM_VOCAB_TERMS,
    OVERLAY_TARGET_DESKTOP,
    OVERLAY_TARGET_STEAMVR,
    AppSettings,
    LLMProviderName,
    OpenRouterCredentialSource,
    OpenRouterLLMModel,
    OpenRouterSelectionAlias,
    QwenRegion,
    STTProviderName,
    TranslationConnection,
    TranslationFallbackSettings,
    TranslationModel,
    _normalize_local_llm_base_url,
    default_translation_connection,
    display_stt_provider,
    is_custom_stt_provider,
    materialize_translation_settings,
    normalize_owned_referral_id,
    supported_translation_connections,
    with_telemetry_consent,
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
    STTProviderName.QWEN_ASR,
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
    STTProviderName.SONIOX: "settings.stt.section.recommended_cloud",
    STTProviderName.LOCAL_CPU_AUTO: "settings.stt.section.recommended_local",
    STTProviderName.QWEN_ASR: "settings.stt.section.cloud",
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
    TranslationModel.GEMMA4_26B_31B: "provider.gemma4_26b_31b",
    TranslationModel.GEMMA4_31B: "provider.gemma4_31b",
    TranslationModel.GEMMA4: "provider.gemma4_26b_a4b_it",
    TranslationModel.DEEPSEEK_V4_FLASH: "provider.deepseek_v4_flash",
    TranslationModel.GEMINI_37_FLASH: "provider.gemini37_flash",
    TranslationModel.GEMINI_31_FLASH_LITE: "provider.gemini31_flash_lite",
    TranslationModel.QWEN_35_PLUS: "provider.qwen35_plus",
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
    TranslationModel.GEMMA4_26B_31B,
    TranslationModel.GEMMA4_31B,
    TranslationModel.GEMMA4,
    TranslationModel.DEEPSEEK_V4_FLASH,
    TranslationModel.LOCAL_LLM,
    TranslationModel.CUSTOM_HTTP,
    TranslationModel.GEMINI_37_FLASH,
    TranslationModel.GEMINI_31_FLASH_LITE,
    TranslationModel.QWEN_35_PLUS,
)
_TRANSLATION_MODEL_SECTION_ORDER = (
    "settings.translation_model.section.recommended_cloud",
    "settings.translation_model.section.recommended_local",
    "settings.translation_model.section.gpu_inference",
    "settings.translation_model.section.gemma",
    "settings.translation_model.section.user_settings",
    "settings.translation_model.section.others",
)
_TRANSLATION_MODEL_SECTION_BY_MODEL: dict[TranslationModel, str] = {
    TranslationModel.MANAGED_GEMMA: "settings.translation_model.section.recommended_local",
    TranslationModel.GEMMA4_26B_31B: "settings.translation_model.section.recommended_cloud",
    TranslationModel.GEMMA4_31B: "settings.translation_model.section.recommended_cloud",
    TranslationModel.DEEPSEEK_V4_FLASH: "settings.translation_model.section.recommended_cloud",
    TranslationModel.GEMMA4: "settings.translation_model.section.gemma",
    TranslationModel.LOCAL_LLM: "settings.translation_model.section.user_settings",
    TranslationModel.CUSTOM_HTTP: "settings.translation_model.section.user_settings",
    TranslationModel.GEMINI_37_FLASH: "settings.translation_model.section.others",
    TranslationModel.GEMINI_31_FLASH_LITE: "settings.translation_model.section.others",
    TranslationModel.QWEN_35_PLUS: "settings.translation_model.section.others",
}
_TRANSLATION_FALLBACK_PRESETS: tuple[tuple[str, TranslationFallbackSettings, str], ...] = (
    (
        "none",
        TranslationFallbackSettings(enabled=False),
        "settings.fallback.none",
    ),
    (
        "deepseek_v4_flash_official",
        TranslationFallbackSettings(
            enabled=True,
            model=TranslationModel.DEEPSEEK_V4_FLASH,
            connection=TranslationConnection.OFFICIAL_BYOK,
        ),
        "settings.fallback.deepseek_v4_flash_official",
    ),
    (
        "openrouter_deepseek_v4_flash",
        TranslationFallbackSettings(
            enabled=True,
            model=TranslationModel.DEEPSEEK_V4_FLASH,
            connection=TranslationConnection.OPENROUTER,
        ),
        "settings.fallback.openrouter_deepseek_v4_flash",
    ),
    (
        "openrouter_gemma4_26b_31b",
        TranslationFallbackSettings(
            enabled=True,
            model=TranslationModel.GEMMA4_26B_31B,
            connection=TranslationConnection.OPENROUTER,
        ),
        "settings.fallback.openrouter_gemma4_26b_31b",
    ),
    (
        "openrouter_gemma4_31b",
        TranslationFallbackSettings(
            enabled=True,
            model=TranslationModel.GEMMA4_31B,
            connection=TranslationConnection.OPENROUTER,
        ),
        "settings.fallback.openrouter_gemma4_31b",
    ),
    (
        "openrouter_gemma4_26b_a4b",
        TranslationFallbackSettings(
            enabled=True,
            model=TranslationModel.GEMMA4,
            connection=TranslationConnection.OPENROUTER,
        ),
        "settings.fallback.openrouter_gemma4_26b_a4b",
    ),
    (
        "cerebras_gemma4_31b",
        TranslationFallbackSettings(
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


def _load_secret_value(store, key: str, *, legacy_keys: tuple[str, ...] = ()) -> str:
    """Load secret value with legacy key fallback."""
    value = store.get(key) or ""
    if value or not legacy_keys:
        return value
    for legacy_key in legacy_keys:
        legacy_value = store.get(legacy_key) or ""
        if legacy_value:
            with contextlib.suppress(Exception):
                store.set(key, legacy_value)
            return legacy_value
    return ""


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
        self.on_settings_changed: Callable[[AppSettings], None] | None = None
        self.on_prompt_apply_settings: Callable[[AppSettings], None] | None = None
        self.on_providers_changed: Callable[[], None] | None = None
        self.on_local_llm_secret_changed: Callable[[], None] | None = None
        self.on_custom_stt_secret_changed: Callable[[], None] | None = None
        self.on_request_openrouter_pkce: Callable[[AppSettings], None] | None = None
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
        self.on_telemetry_consent_change: Callable[[str], None] | None = None
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
        self._settings: AppSettings | None = None
        self._provider_settings_draft: AppSettings | None = None
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
        self._local_cpu_auto_available = False

        # Build UI components
        self._build_ui()

    def set_local_cpu_auto_available(self, available: bool) -> None:
        self._local_cpu_auto_available = bool(available)

    def self_stt_control(self) -> ft.Control:
        return self._self_stt_card

    def peer_stt_control(self) -> ft.Control:
        return self._peer_stt_card

    def translation_provider_control(self) -> ft.Control:
        return self._translation_provider_card

    def translation_connection_control(self) -> ft.Control:
        return self._translation_connection_card

    def http_extension_control(self) -> ft.Control:
        return ft.Container(content=self._http_extension_row)

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
        self.on_start_microphone_test = general.start_microphone_test
        self.on_telemetry_consent_change = general.telemetry_consent_change
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
            self._overlay_vr_reset_button,
            self._overlay_desktop_reset_button,
            self._desktop_overlay_primary_action,
            self._desktop_overlay_view_logs_action,
            self._translation_connection_text,
            self._openrouter_fallback_text,
            self._telemetry_consent_text,
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

    def _sync_osc_connection_card(self, settings: AppSettings) -> None:
        mode = settings.osc.connection_mode
        if mode not in {"automatic", "manual", "off"}:
            mode = "automatic"
        self._set_unit_card_value_text(
            self._osc_connection_text,
            t(f"settings.osc.mode.{mode}"),
        )

    def refresh_loopback_capture_target(self, settings: AppSettings) -> None:
        self._rebase_retained_loopback_capture_target(settings)
        self._audio_settings.desktop_output_device = settings.desktop_audio.output_device
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

    def _rebase_retained_loopback_capture_target(self, settings: AppSettings) -> None:
        for retained in (self._settings, self._provider_settings_draft):
            if retained is None:
                continue
            retained.desktop_audio.output_device = settings.desktop_audio.output_device
            retained.desktop_audio.runtime_capture_target = (
                settings.desktop_audio.runtime_capture_target
            )

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
        self._managed_key_referral_helper_text = ft.Text(
            t("settings.managed_key.referral_id.pending_helper"),
            size=14,
            color=COLOR_SECONDARY,
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
                            self._managed_key_referral_helper_text,
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

        self._telemetry_consent_text = self._build_clickable_text(
            t("settings.telemetry.state.off"),
            self._on_telemetry_consent_click,
        )
        self._telemetry_consent_title = ft.Text(
            t("settings.telemetry.title"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._telemetry_consent_card = self._wrap_unit_card(
            title=self._telemetry_consent_title,
            value=self._telemetry_consent_text,
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
                telemetry_consent=self._telemetry_consent_card,
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
            t("settings.gpu_device.title"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
        )
        self._gpu_device_text = self._build_clickable_text(
            t("settings.gpu_device.auto"),
            self._on_gpu_device_click,
            max_lines=2,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        self._gpu_device_card = self._wrap_unit_card(
            title=self._gpu_device_title,
            value=self._gpu_device_text,
        )
        self._gpu_device_card.visible = False

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
        self._overlay_desktop_reset_spacer_a = self._wrap_empty_unit_card()
        self._overlay_desktop_reset_spacer_b = self._wrap_empty_unit_card()

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
                desktop_reset=self._overlay_desktop_reset_card,
                desktop_reset_spacer_a=self._overlay_desktop_reset_spacer_a,
                desktop_reset_spacer_b=self._overlay_desktop_reset_spacer_b,
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
            value=json.dumps({"reasoning_effort": "none"}, ensure_ascii=False, indent=2),
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
        self._http_extension_row.visible = False

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

    def _gpu_selected(self, settings: AppSettings | None = None) -> bool:
        current = settings or self._build_settings_with_provider_draft()
        return bool(
            current is not None
            and (
                current.provider.stt == STTProviderName.LOCAL_QWEN_GPU
                or current.provider.peer_stt == STTProviderName.LOCAL_QWEN_GPU
            )
        )

    def _sync_gpu_device_card(self) -> None:
        if not hasattr(self, "_gpu_device_text"):
            return
        settings = self._build_settings_with_provider_draft()
        selected = settings.stt.gpu_device_id if settings is not None else "auto"
        devices = getattr(self, "_gpu_devices", ())
        selected_device = next(
            (device for device in devices if device.device_id == selected),
            None,
        )
        if selected == "auto":
            label = t("settings.gpu_device.auto")
        elif selected_device is not None:
            label = selected_device.display_name
        else:
            label = t("settings.gpu_device.unavailable", device=selected)
        self._set_unit_card_value_text(self._gpu_device_text, label)
        visible = self._gpu_selected(settings)
        self._gpu_device_card.visible = visible
        self._gpu_device_row.visible = visible
        _update_control_if_mounted(self._gpu_device_row)

    def set_gpu_devices(
        self,
        *,
        devices: tuple[GpuDeviceOption, ...],
    ) -> None:
        self._gpu_devices = devices
        self._sync_gpu_device_card()

    @staticmethod
    def _gpu_backend_label(name: str) -> str:
        match = re.fullmatch(r"Vulkan\s*(\d+)", name.strip(), flags=re.IGNORECASE)
        if match is not None:
            return f"Vulkan {match.group(1)}"
        return name.strip()

    def _on_gpu_device_click(self, _event) -> None:
        if not is_control_mounted(self):
            return
        settings = self._build_settings_with_provider_draft()
        selected = settings.stt.gpu_device_id if settings is not None else "auto"
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
            for device in self._gpu_devices
        )
        if selected != "auto" and all(device.device_id != selected for device in self._gpu_devices):
            options.append(
                OptionItem(
                    value=selected,
                    label=t("settings.gpu_device.unavailable", device=selected),
                )
            )
        SettingsModal(
            self.page,
            t("settings.gpu_device.title"),
            options,
            self._on_gpu_device_selected,
            show_description=True,
        ).open(selected)

    def _on_gpu_device_selected(self, value: str) -> None:
        if self._settings is None:
            return
        draft = self._ensure_provider_settings_draft()
        draft.stt.gpu_device_id = value or "auto"
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
        controls: list[ft.Control] = []
        if extension is not None:
            for secret in extension.secrets:
                field = ft.TextField(
                    label=t("settings.http_extension.api_key"),
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
        settings: AppSettings | None = None,
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
        self._http_extension_credentials.visible = is_custom
        if not is_custom:
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
        _update_control_if_mounted(self._http_extension_row)
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
        if self._settings is None:
            return
        draft = self._ensure_provider_settings_draft()
        draft.translation.http_extension_id = value or ""
        self.has_provider_changes = True
        self._sync_http_extension_card(draft, force_credentials=True)

    def _on_http_extension_secret_blur(self, secret_id: str) -> None:
        http_extension_id = self._http_extension_selected_id
        field = self._http_extension_secret_fields.get(secret_id)
        if not http_extension_id or field is None:
            return
        value = (field.value or "").strip()
        if not value and secret_id not in self._http_extension_secret_dirty:
            return
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
        active_settings = self._settings
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

    def _get_llm_modal_value(self, settings: AppSettings) -> str:
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

    def _sync_translation_connection_title(self, settings: AppSettings) -> None:
        title = getattr(self, "_translation_connection_title", None)
        if title is None:
            return
        title.value = t("settings.translation_connection")

    def _stored_openrouter_selection_alias(
        self, settings: AppSettings
    ) -> OpenRouterSelectionAlias | None:
        if settings.openrouter.selection_alias is None:
            if settings.openrouter.selected_source == OpenRouterCredentialSource.NONE:
                return None
            return _derive_openrouter_selection_alias(
                settings.openrouter.llm_model,
                settings.openrouter.selected_source,
            )
        try:
            profile_for_alias(settings.openrouter.selection_alias.value)
            return settings.openrouter.selection_alias
        except KeyError:
            if settings.openrouter.selected_source == OpenRouterCredentialSource.NONE:
                return None
            return _derive_openrouter_selection_alias(
                settings.openrouter.llm_model,
                settings.openrouter.selected_source,
            )

    def _display_openrouter_selection_alias(
        self, settings: AppSettings
    ) -> OpenRouterSelectionAlias:
        stored_alias = self._stored_openrouter_selection_alias(settings)
        if stored_alias is not None:
            return stored_alias
        if settings.openrouter.llm_model == OpenRouterLLMModel.QWEN_35_FLASH_02_23:
            return OpenRouterSelectionAlias.QWEN35_FLASH_MANAGED
        if settings.openrouter.llm_model == OpenRouterLLMModel.DEEPSEEK_V4_FLASH:
            return OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED
        return OpenRouterSelectionAlias.GEMMA4_MANAGED

    def _openrouter_selection_profile(self, settings: AppSettings | None):
        if settings is None:
            return None
        try:
            return profile_for_alias(self._display_openrouter_selection_alias(settings).value)
        except KeyError:
            return None

    def _translation_fallback_preset_value(self, fallback: TranslationFallbackSettings) -> str:
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
        fallback: TranslationFallbackSettings,
    ) -> str:
        preset_value = self._translation_fallback_preset_value(fallback)
        label_key = _TRANSLATION_FALLBACK_LABEL_KEY_BY_VALUE.get(preset_value)
        if label_key is not None:
            return t(label_key)
        model_label = self._translation_model_display_label(fallback.model)
        connection_label = self._translation_connection_display_label(fallback.connection)
        return f"{model_label} · {connection_label}"

    def _openrouter_fallback_source(
        self, settings: AppSettings | None
    ) -> OpenRouterCredentialSource:
        if settings is None:
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

    def _get_llm_display_label(self, settings: AppSettings) -> str:
        model = settings.translation.model
        if model == TranslationModel.MANAGED_GEMMA:
            if settings.translation.connection == TranslationConnection.GPU:
                return t("provider.managed_gemma_gpu")
            return t("provider.managed_gemma_cpu")
        return self._translation_model_display_label(model)

    def _get_translation_connection_display_label(self, settings: AppSettings | None) -> str:
        if settings is None:
            return self._translation_connection_display_label(TranslationConnection.MANAGED)
        return self._translation_connection_display_label(settings.translation.connection)

    def _get_openrouter_fallback_display_label(self, settings: AppSettings | None) -> str:
        if settings is None:
            return t("settings.fallback.none")
        return self._translation_fallback_display_label(settings.translation.fallback)

    def _get_openrouter_fallback_helper_text(self, settings: AppSettings | None) -> str:
        if settings is None:
            return t("settings.fallback.inactive_helper")
        if not settings.translation.fallback.enabled:
            return t("settings.fallback.none.description")
        return t("settings.fallback.active_helper")

    def _telemetry_consent_display_label(self, settings: AppSettings | None) -> str:
        consent = getattr(getattr(settings, "telemetry", None), "consent", "unknown")
        return t(
            "settings.telemetry.state.on"
            if consent != "decline"
            else "settings.telemetry.state.off"
        )

    def _sync_telemetry_consent_card(self, settings: AppSettings | None = None) -> None:
        if settings is None:
            settings = self._settings
        self._set_unit_card_value_text(
            self._telemetry_consent_text,
            self._telemetry_consent_display_label(settings),
        )

    def _set_openrouter_fallback_text(self, text: str) -> None:
        text_control = self._openrouter_fallback_text.content
        text_control.value = text
        text_control.size = 28

    def _sync_openrouter_fallback_card(self, settings: AppSettings | None = None) -> None:
        if settings is None:
            settings = self._build_settings_with_provider_draft()
        self._set_openrouter_fallback_text(self._get_openrouter_fallback_display_label(settings))
        self._openrouter_fallback_helper_text.value = self._get_openrouter_fallback_helper_text(
            settings
        )

    def _active_prompt_key_for_settings(self, settings: AppSettings | None) -> str:
        if settings is None:
            return "gemini"
        if settings.provider.llm == LLMProviderName.GEMINI:
            return "gemini"
        if settings.provider.llm == LLMProviderName.OPENROUTER:
            return "openrouter"
        if settings.provider.llm == LLMProviderName.DEEPSEEK:
            return "deepseek"
        if settings.provider.llm == LLMProviderName.LOCAL_LLM:
            return "local_llm"
        if settings.provider.llm == LLMProviderName.MANAGED_GEMMA:
            return "managed_gemma"
        return "qwen"

    def _active_prompt_key(self) -> str:
        return self._active_prompt_key_for_settings(self._build_settings_with_provider_draft())

    def _ensure_provider_prompt_value(self, settings: AppSettings, provider_name: str) -> str:
        prompt = settings.system_prompt
        if prompt.strip():
            settings.system_prompts = {}
            return prompt
        prompt = load_prompt_for_provider(provider_name)
        settings.system_prompt = prompt
        settings.system_prompts = {}
        return prompt

    def _current_source_language(self) -> str:
        if not self._settings:
            return "en"
        return self._settings.languages.source_language

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
        if not self._settings:
            self._custom_vocab_tag_editor.set_terms([])
            self._custom_vocab_tag_editor.clear_input()
            return

        source_language = self._current_source_language()
        self._custom_vocab_tag_editor.set_terms(
            list(self._settings.stt.custom_terms.get(source_language, []))
        )
        self._custom_vocab_tag_editor.clear_input()

    def _sync_peer_auto_languages_editor(self, settings: AppSettings | None = None) -> None:
        if not hasattr(self, "_peer_auto_languages_editor"):
            return
        settings = settings or self._settings
        languages = [] if settings is None else settings.languages.peer_expected_languages
        self._peer_auto_languages_editor.set_terms(list(languages))

    def _set_peer_auto_languages(self, languages: list[str]) -> None:
        if self._settings is None:
            return
        normalized = list(
            dict.fromkeys(language.strip() for language in languages if language.strip())
        )
        if self._settings.languages.peer_expected_languages == normalized:
            return
        self._settings.languages.peer_expected_languages = normalized
        self._sync_peer_auto_languages_editor()
        self._emit_settings_changed()

    def _on_peer_auto_languages_add(self, language: str) -> None:
        if self._settings is None:
            return
        self._set_peer_auto_languages([*self._settings.languages.peer_expected_languages, language])

    def _on_peer_auto_languages_remove(self, language: str) -> None:
        if self._settings is None:
            return
        self._set_peer_auto_languages(
            [
                current
                for current in self._settings.languages.peer_expected_languages
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

    def _is_managed_translation_connection_selected(self, settings: AppSettings | None) -> bool:
        if settings is None:
            return False
        if settings.translation.model == TranslationModel.CUSTOM_HTTP:
            return False
        managed_connections = (TranslationConnection.MANAGED, TranslationConnection.MANAGED_CHINA)
        return bool(
            settings.translation.connection in managed_connections
            or (
                settings.translation.fallback.enabled
                and settings.translation.fallback.connection in managed_connections
            )
        )

    def _managed_key_card_visible_for(self, settings: AppSettings | None) -> bool:
        return self._is_managed_translation_connection_selected(settings)

    def _sync_managed_key_referral_row_value(self, referral_id: str | None) -> None:
        referral_id = normalize_owned_referral_id(referral_id)
        self._managed_key_referral_id = referral_id

        self._managed_key_referral_id_value.value = referral_id or t(
            "settings.managed_key.referral_id.empty"
        )
        self._managed_key_referral_helper_text.value = t(
            "settings.managed_key.referral_id.helper"
            if referral_id is not None
            else "settings.managed_key.referral_id.pending_helper"
        )

    def _remember_managed_key_referral_id(self, referral_id: str | None) -> str | None:
        referral_id = normalize_owned_referral_id(referral_id)
        if referral_id is None:
            return None

        if self._settings is not None:
            self._settings.managed_identity.referral_id = referral_id
        if self._provider_settings_draft is not None:
            self._provider_settings_draft.managed_identity.referral_id = referral_id
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

    def _sync_managed_key_referral_row(self, settings: AppSettings | None) -> None:
        referral_id = None
        if settings is not None:
            referral_id = normalize_owned_referral_id(
                getattr(settings.managed_identity, "referral_id", None)
            )
        self._sync_managed_key_referral_row_value(referral_id)

    def _sync_managed_key_card(self, settings: AppSettings | None = None) -> None:
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
            self._managed_key_referral_helper_text,
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

    def _copy_provider_draft_fields(self, source: AppSettings, target: AppSettings) -> None:
        target.provider.stt = source.provider.stt
        target.provider.peer_stt = source.provider.peer_stt
        target.stt.gpu_device_id = source.stt.gpu_device_id
        target.provider.llm = source.provider.llm
        target.translation = copy.deepcopy(source.translation)
        target.gemini.llm_model = source.gemini.llm_model
        target.openrouter.llm_model = source.openrouter.llm_model
        target.openrouter.routing_mode = source.openrouter.routing_mode
        target.openrouter.provider_routing = source.openrouter.provider_routing
        target.openrouter.selected_source = source.openrouter.selected_source
        target.openrouter.selection_alias = source.openrouter.selection_alias
        target.qwen.llm_model = source.qwen.llm_model
        target.qwen.region = source.qwen.region
        target.deepseek.llm_model = source.deepseek.llm_model
        target.local_llm = copy.deepcopy(source.local_llm)
        target.custom_stt = copy.deepcopy(source.custom_stt)
        if source.openrouter.selected_source == OpenRouterCredentialSource.MANAGED:
            target.managed_identity.verified_hardware_hash = (
                source.managed_identity.verified_hardware_hash
            )
            target.managed_identity.verified_hardware_hash_salt_version = (
                source.managed_identity.verified_hardware_hash_salt_version
            )
        target.system_prompt = source.system_prompt
        target.system_prompts = {}

    def _build_settings_with_provider_draft(self) -> AppSettings | None:
        if self._settings is None:
            return None
        if self._provider_settings_draft is None:
            return self._settings
        merged = copy.deepcopy(self._settings)
        self._copy_provider_draft_fields(self._provider_settings_draft, merged)
        return merged

    def _ensure_provider_settings_draft(self) -> AppSettings:
        assert self._settings is not None
        if self._provider_settings_draft is None:
            self._provider_settings_draft = copy.deepcopy(self._settings)
        return self._provider_settings_draft

    def _stt_provider_display_label(
        self,
        provider: STTProviderName,
        *,
        custom_mode: str = "offline",
    ) -> str:
        return provider_label(display_stt_provider(provider, custom_mode=custom_mode).value)

    def _normalized_peer_stt_provider(self, provider: STTProviderName) -> STTProviderName:
        return provider

    def _effective_peer_stt_provider(self, settings: AppSettings | None) -> STTProviderName:
        if settings is None:
            return STTProviderName.LOCAL_CPU_AUTO
        return self._normalized_peer_stt_provider(settings.provider.peer_stt)

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
        return OptionItem(
            value=provider.value,
            label=provider_label(provider.value),
            description=t(f"provider.{provider.value}.description", default=""),
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
        if not self._settings:
            return
        current = self._provider_settings_draft or self._settings
        if current.provider.llm != LLMProviderName.LOCAL_LLM:
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
        if not self._settings:
            return
        raw_value = self._local_llm_base_url.value or ""
        try:
            normalized = _normalize_local_llm_base_url(raw_value)
        except ValueError:
            self._local_llm_base_url.error = t("settings.local_llm.base_url.invalid")
            _update_control_if_mounted(self._local_llm_base_url)
            return

        self._local_llm_base_url.error = None
        self._local_llm_base_url.value = normalized
        current = self._provider_settings_draft or self._settings
        if current.local_llm.base_url != normalized:
            draft = self._ensure_provider_settings_draft()
            draft.local_llm.base_url = normalized
            self.has_provider_changes = True
        _update_control_if_mounted(self._local_llm_base_url)

    def _on_local_llm_model_change_end(self, e) -> None:
        _ = e
        if not self._settings:
            return
        model = (self._local_llm_model.value or "").strip()
        if not model:
            self._local_llm_model.error = t("settings.local_llm.model.required")
            _update_control_if_mounted(self._local_llm_model)
            return

        self._local_llm_model.error = None
        self._local_llm_model.value = model
        current = self._provider_settings_draft or self._settings
        if current.local_llm.model != model:
            draft = self._ensure_provider_settings_draft()
            draft.local_llm.model = model
            self.has_provider_changes = True
        _update_control_if_mounted(self._local_llm_model)

    def _on_local_llm_extra_body_change_end(self, e) -> None:
        _ = e
        if not self._settings:
            return
        raw = (self._local_llm_extra_body.value or "").strip()
        try:
            parsed = (
                {"reasoning_effort": "none"}
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
        current = self._provider_settings_draft or self._settings
        if current.local_llm.extra_body != normalized:
            draft = self._ensure_provider_settings_draft()
            draft.local_llm.extra_body = normalized
            self.has_provider_changes = True
        self._local_llm_extra_body.value = json.dumps(
            normalized,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        self._clear_local_llm_extra_body_error()
        _update_control_if_mounted(self._local_llm_extra_body)

    def _commit_local_llm_fields_from_controls(self) -> None:
        if not self._settings:
            return
        current = self._provider_settings_draft or self._settings
        if current.provider.llm != LLMProviderName.LOCAL_LLM:
            return
        self._on_local_llm_base_url_change_end(None)
        self._on_local_llm_model_change_end(None)
        self._on_local_llm_extra_body_change_end(None)

    def _settings_with_desktop_overlay_runtime_state(
        self,
        settings: AppSettings | None,
    ) -> AppSettings | None:
        if settings is None:
            return None
        pending_position_reset = getattr(self, "_desktop_overlay_pending_position_reset", False)
        desktop_settings = settings.overlay.desktop_flet
        size_preset = self._current_desktop_overlay_size_preset()
        needs_copy = desktop_settings.size_preset != size_preset or pending_position_reset
        if not needs_copy:
            return settings

        updated = copy.deepcopy(settings)
        updated_desktop = updated.overlay.desktop_flet
        updated_desktop.size_preset = size_preset
        if pending_position_reset:
            updated_desktop.position.x = None
            updated_desktop.position.y = None
            updated_desktop.locked = False
        updated_desktop.validate()
        return updated

    def _sanitize_provider_apply_settings(self, settings: AppSettings | None) -> AppSettings | None:
        if settings is not None:
            settings.system_prompts = {}
        return settings

    def _stage_prompt_draft(self, value: str) -> None:
        if not self._settings:
            return
        committed_prompt = self._committed_prompt_value()
        draft = self._ensure_provider_settings_draft()
        draft.system_prompt = value
        draft.system_prompts = {}
        self.has_pending_prompt_changes = value != committed_prompt
        if not self.has_pending_prompt_changes and not self.has_provider_changes:
            self._provider_settings_draft = None

    def _committed_prompt_value(self) -> str:
        if not self._settings:
            return ""
        return self._settings.system_prompt

    def build_provider_apply_settings(self) -> AppSettings | None:
        self._commit_local_llm_fields_from_controls()
        return self._sanitize_provider_apply_settings(
            self._settings_with_desktop_overlay_runtime_state(
                self._build_settings_with_provider_draft()
            )
        )

    def consume_provider_apply_settings(self) -> AppSettings | None:
        settings = self.build_provider_apply_settings()
        if settings is None:
            return None
        self._settings = settings
        self._provider_settings_draft = None
        self.has_provider_changes = False
        self.has_pending_prompt_changes = False
        return settings

    def consume_prompt_apply_settings(self) -> AppSettings | None:
        if not self.has_pending_prompt_changes:
            return None
        settings = self._sanitize_provider_apply_settings(
            self._settings_with_desktop_overlay_runtime_state(
                self._build_settings_with_provider_draft()
            )
        )
        if settings is None:
            return None
        self._settings = settings
        self.has_pending_prompt_changes = False
        if not self.has_provider_changes:
            self._provider_settings_draft = None
        return settings

    # --- Load Settings ---
    def load_from_settings(
        self,
        settings: AppSettings,
        *,
        config_path: Path,
        preserve_custom_vocab_draft: bool = False,
    ) -> None:
        """Load current settings into the UI."""
        self._settings = settings
        self._provider_settings_draft = None
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
        self._sync_clickable_text_control_fonts(font_for_language(settings.ui.locale))

        # UI Language
        self._ui_text.content.value = locale_label(settings.ui.locale)

        # STT Provider
        self._set_unit_card_value_text(
            self._stt_text,
            self._stt_provider_display_label(
                settings.provider.stt,
                custom_mode=settings.custom_stt.mode,
            ),
        )
        self._set_unit_card_value_text(
            self._peer_stt_text,
            self._stt_provider_display_label(
                self._effective_peer_stt_provider(settings),
                custom_mode=settings.custom_stt.mode,
            ),
        )
        self._update_api_visibility()
        self._sync_gpu_device_card()

        # LLM Provider
        self._set_unit_card_value_text(
            self._llm_text,
            self._get_llm_display_label(settings),
        )
        self._set_translation_connection_text(
            self._get_translation_connection_display_label(settings),
        )
        self._sync_translation_connection_title(settings)
        self._sync_openrouter_fallback_card(settings)
        self._local_llm_base_url.value = settings.local_llm.base_url
        self._local_llm_base_url.error = None
        self._local_llm_model.value = settings.local_llm.model
        self._local_llm_model.error = None
        self._local_llm_extra_body.value = json.dumps(
            settings.local_llm.extra_body,
            ensure_ascii=False,
            indent=2,
        )
        self._clear_local_llm_extra_body_error()
        self._sync_custom_stt_card(settings)

        # Qwen Region
        region_label = t(f"region.{settings.qwen.region.value}")
        _set_text_button_label(self._qwen_region_btn, f"{t('settings.qwen_region')} {region_label}")

        # Audio Settings
        self._audio_settings.host_api = settings.audio.input_host_api
        self._audio_settings.microphone = settings.audio.input_device
        self._audio_settings.desktop_output_device = settings.desktop_audio.output_device
        self._sync_general_audio_card_texts()

        # VAD
        self._vad_slider.value = settings.stt.vad_speech_threshold
        self._vad_slider.label = f"{settings.stt.vad_speech_threshold:.2f}"
        self._peer_vad_slider.value = settings.desktop_audio.vad_speech_threshold
        self._peer_vad_slider.label = f"{settings.desktop_audio.vad_speech_threshold:.2f}"
        self._peer_vad_field.value = f"{settings.desktop_audio.vad_speech_threshold:.2f}"
        self._peer_hangover_field.value = str(settings.desktop_audio.vad_hangover_ms)
        self._peer_pre_roll_field.value = str(settings.desktop_audio.vad_pre_roll_ms)
        # --- 新增：读取 VRChat 同步开关状态 ---
        self._vrc_mic_text.content.value = t(
            "settings.vrc_mic.on" if settings.osc.vrc_mic_intercept else "settings.vrc_mic.off"
        )
        self._sync_osc_connection_card(settings)
        self._chatbox_source_text.content.value = t(
            "settings.chatbox_source.on"
            if settings.osc.chatbox_include_source
            else "settings.chatbox_source.off"
        )
        self._clipboard_auto_translate_text.content.value = t(
            "settings.clipboard_auto_translate.on"
            if settings.ui.clipboard_auto_translate_enabled
            else "settings.clipboard_auto_translate.off"
        )
        self._sync_telemetry_consent_card(settings)
        # Prompt
        provider_name = self._active_prompt_key()
        self._prompt_editor.set_provider(provider_name)
        settings.system_prompts = {}
        if settings.system_prompt.strip():
            self._prompt_editor.value = settings.system_prompt
        else:
            self._prompt_editor.load_default_prompt(emit_change=False)
            settings.system_prompt = self._prompt_editor.value

        _ = preserve_custom_vocab_draft
        self._sync_custom_vocabulary_editor_from_settings()
        self._sync_prompt_tab_copy()
        self._overlay_peer_contract = None
        self._sync_overlay_controls()
        self.set_overlay_calibration(
            settings.overlay.calibration,
            preserve_draft=self._overlay_calibration_session_active,
        )

        # Load secrets
        self._load_secrets(settings, config_path)

        if is_control_mounted(self):
            self.update()

    def refresh_after_openrouter_pkce_success(
        self,
        settings: AppSettings,
        *,
        config_path: Path,
    ) -> None:
        self._settings = settings
        self._provider_settings_draft = None
        self._config_path = config_path
        self.has_provider_changes = False
        self.has_pending_prompt_changes = False
        self._desktop_overlay_pending_size_preset = None
        self._desktop_overlay_pending_position_reset = False
        self._desktop_overlay_pending_locked = None
        self._desktop_overlay_captions_locked = False

        self._set_unit_card_value_text(
            self._llm_text,
            self._get_llm_display_label(settings),
        )
        self._set_translation_connection_text(
            self._get_translation_connection_display_label(settings),
        )
        self._sync_openrouter_fallback_card(settings)
        self._update_api_visibility()

        provider_name = self._active_prompt_key()
        self._prompt_editor.set_provider(provider_name)
        settings.system_prompts = {}
        if settings.system_prompt.strip():
            self._prompt_editor.value = settings.system_prompt
        else:
            self._prompt_editor.load_default_prompt(emit_change=False)
            settings.system_prompt = self._prompt_editor.value
        self._sync_custom_vocabulary_editor_from_settings()
        self._sync_prompt_tab_copy()

        try:
            store = create_secret_store(settings.secrets, config_path=config_path)
        except Exception as exc:
            self._emit_runtime_basic(f"Failed to load secrets: {exc}", level=logging.WARNING)
        else:
            self._openrouter_key.value = store.get("openrouter_api_key") or ""
            self._deepseek_key.value = store.get("deepseek_api_key") or ""
            self._cerebras_key.value = store.get("cerebras_api_key") or ""
            self._restore_api_key_icons(settings)

        if is_control_mounted(self):
            self.update()

    def sync_telemetry_settings(self, settings: AppSettings) -> None:
        self._settings = settings
        self._sync_telemetry_consent_card(settings)
        if is_control_mounted(self):
            _update_control_if_mounted(self._telemetry_consent_card)

    def _load_secrets(self, settings: AppSettings, config_path: Path) -> None:
        """Load secret values into fields."""
        try:
            store = create_secret_store(settings.secrets, config_path=config_path)
        except Exception as exc:
            self._emit_runtime_basic(f"Failed to load secrets: {exc}", level=logging.WARNING)
            return

        self._google_key.value = store.get("google_api_key") or ""
        self._openrouter_key.value = store.get("openrouter_api_key") or ""
        self._deepseek_key.value = store.get("deepseek_api_key") or ""
        self._cerebras_key.value = store.get("cerebras_api_key") or ""
        self._deepgram_key.value = store.get("deepgram_api_key") or ""
        self._soniox_key.value = store.get("soniox_api_key") or ""
        self._local_llm_api_key.value = store.get("local_llm_api_key") or ""
        self._custom_stt_api_key.value = store.get("custom_stt_api_key") or ""

        # Alibaba keys with legacy fallback
        beijing_key = _load_secret_value(
            store, "alibaba_api_key_beijing", legacy_keys=("alibaba_api_key",)
        )
        singapore_key = _load_secret_value(
            store, "alibaba_api_key_singapore", legacy_keys=("alibaba_api_key",)
        )

        self._alibaba_key_beijing.value = beijing_key
        self._alibaba_key_singapore.value = singapore_key

        # Restore verification status icons from saved settings
        self._restore_api_key_icons(settings)

    def _restore_api_key_icons(self, settings: AppSettings) -> None:
        """Restore API key field icons based on saved verification status."""
        verified = settings.api_key_verified

        # Map field -> (has_key, is_verified)
        field_map = [
            (self._deepgram_key, self._deepgram_key.value, verified.deepgram),
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

    def _sync_openrouter_pkce_button_state(self, settings: AppSettings | None = None) -> None:
        if settings is None:
            settings = self._build_settings_with_provider_draft()
        authenticated = bool(
            settings is not None
            and settings.api_key_verified.openrouter
            and self._openrouter_key.value
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
    def _sync_managed_trial_usage_bar(self, settings: AppSettings | None = None) -> None:
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

        stt = settings.provider.stt
        llm = settings.provider.llm
        is_custom_http = settings.translation.model == TranslationModel.CUSTOM_HTTP
        peer_stt = self._effective_peer_stt_provider(settings)
        fallback = settings.translation.fallback
        fallback_source = self._openrouter_fallback_source(settings)
        active_stt_providers = {stt, peer_stt}
        self._deepgram_key.visible = STTProviderName.DEEPGRAM in active_stt_providers
        self._soniox_key.visible = STTProviderName.SONIOX in active_stt_providers
        peer_auto_languages_card = getattr(self, "_peer_auto_languages_card", None)
        if peer_auto_languages_card is not None:
            peer_auto_languages_card.visible = peer_stt == STTProviderName.SONIOX
            self._sync_peer_auto_languages_editor(settings)
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
            and settings.openrouter.selected_source == OpenRouterCredentialSource.BYOK
        )
        self._openrouter_key.visible = bool(
            not is_custom_http
            and (openrouter_byok_selected or fallback_source == OpenRouterCredentialSource.BYOK)
        )
        self._openrouter_pkce_button_row.visible = openrouter_byok_selected
        self._deepseek_key.visible = bool(
            not is_custom_http
            and (
                llm == LLMProviderName.DEEPSEEK
                or (
                    fallback.enabled
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
                    fallback.enabled
                    and fallback.model == TranslationModel.GEMMA4_31B
                    and fallback.connection == TranslationConnection.CEREBRAS
                )
            )
        )
        self._sync_openrouter_pkce_button_state(settings)
        self._translation_connection_row.visible = (
            not is_custom_http and settings.translation.model != TranslationModel.MANAGED_GEMMA
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
            openrouter_fallback_card.visible = not is_custom_http and (
                settings.translation.model != TranslationModel.MANAGED_GEMMA
            )
        self._sync_http_extension_card(settings)

        qwen_regions: set[QwenRegion] = set()
        if (
            stt == STTProviderName.QWEN_ASR
            or (not is_custom_http and llm == LLMProviderName.QWEN)
            or peer_stt == STTProviderName.QWEN_ASR
        ):
            qwen_regions.add(settings.qwen.region)

        self._qwen_region_btn.visible = (
            stt == STTProviderName.QWEN_ASR
            or (not is_custom_http and llm == LLMProviderName.QWEN)
            or peer_stt == STTProviderName.QWEN_ASR
        )
        self._alibaba_key_beijing.visible = QwenRegion.BEIJING in qwen_regions
        self._alibaba_key_singapore.visible = QwenRegion.SINGAPORE in qwen_regions

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
            display_stt_provider(
                display_settings.provider.stt,
                custom_mode=display_settings.custom_stt.mode,
            ).value
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
        if not self._settings:
            return
        current_settings = self._build_settings_with_provider_draft()
        assert current_settings is not None
        provider = STTProviderName(value)
        if provider == STTProviderName.LOCAL_CPU_AUTO and not self._local_cpu_auto_available:
            return
        old_provider = current_settings.provider.stt.value
        if old_provider == provider.value:
            return
        self._emit_runtime_basic(
            f"[Settings] STT provider changed: {old_provider} -> {provider.value}"
        )
        draft = self._ensure_provider_settings_draft()
        draft.provider.stt = provider
        if (
            provider == STTProviderName.LOCAL_QWEN_GPU
            and self.on_gpu_discovery_requested is not None
        ):
            self.on_gpu_discovery_requested()
        self._update_api_visibility()
        self._sync_gpu_device_card()
        self.has_provider_changes = True

        # Update text
        self._set_unit_card_value_text(self._stt_text, provider_label(provider.value))

        source_lang = self._settings.languages.source_language
        selection = resolve_local_asr_selection(provider.value, source_lang)
        if selection.fallback_applied:
            self._show_stt_selection_notice(t("local_stt.language_fallback_qwen"))
        elif not selection.supported:
            self._show_stt_selection_notice(t("local_stt.language_unsupported"))
        warning = (
            None
            if selection.fallback_applied or not selection.supported
            else get_stt_compatibility_warning(source_lang, provider.value)
        )
        if warning:
            message = t(warning.key, language=language_name(warning.language_code))
            self._show_stt_selection_notice(message)

        if is_control_mounted(self):
            self._qwen_region_btn.update()
            self._api_keys_column.update()
            self._stt_text.update()

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
        current_provider = (
            display_settings.provider.peer_stt
            if display_settings is not None
            else STTProviderName.LOCAL_CPU_AUTO
        )
        current = display_stt_provider(
            self._normalized_peer_stt_provider(current_provider),
            custom_mode=(
                display_settings.custom_stt.mode if display_settings is not None else "offline"
            ),
        ).value
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
        if not self._settings:
            return
        current_settings = self._build_settings_with_provider_draft()
        assert current_settings is not None
        provider = STTProviderName(value)
        if provider == STTProviderName.LOCAL_CPU_AUTO and not self._local_cpu_auto_available:
            return
        if current_settings.provider.peer_stt == provider:
            return
        draft = self._ensure_provider_settings_draft()
        draft.provider.peer_stt = provider
        if (
            provider == STTProviderName.LOCAL_QWEN_GPU
            and self.on_gpu_discovery_requested is not None
        ):
            self.on_gpu_discovery_requested()
        selection = resolve_local_asr_selection(
            provider.value,
            current_settings.languages.effective_peer_source,
        )
        if selection.fallback_applied:
            self._show_stt_selection_notice(t("local_stt.language_fallback_qwen"))
        elif not selection.supported:
            self._show_stt_selection_notice(t("local_stt.language_unsupported"))
        self._set_unit_card_value_text(self._peer_stt_text, provider_label(value))
        self._update_api_visibility()
        self._sync_gpu_device_card()
        if is_control_mounted(self):
            self._peer_stt_text.update()
            self._qwen_region_btn.update()
            self._api_keys_column.update()
        self.has_provider_changes = True

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
        history: dict[str, TranslationConnection],
    ) -> TranslationConnection:
        connection = history.get(model.value)
        if not isinstance(connection, TranslationConnection):
            try:
                connection = TranslationConnection(str(connection))
            except (TypeError, ValueError):
                connection = None
        if connection in supported_translation_connections(model):
            return connection
        return default_translation_connection(model)

    def _sync_translation_selection_controls(self, settings: AppSettings) -> None:
        self._set_unit_card_value_text(
            self._llm_text,
            self._get_llm_display_label(settings),
        )
        self._set_translation_connection_text(
            self._get_translation_connection_display_label(settings),
        )
        self._sync_translation_connection_title(settings)
        self._sync_openrouter_fallback_card(settings)

    def _apply_translation_selection(
        self,
        model: TranslationModel,
        connection: TranslationConnection,
    ) -> None:
        if not self._settings:
            return
        if connection not in supported_translation_connections(model):
            return

        current_settings = self._build_settings_with_provider_draft()
        assert current_settings is not None
        old_model = current_settings.translation.model
        old_connection = current_settings.translation.connection
        old_provider = current_settings.provider.llm
        if old_model == model and old_connection == connection:
            return

        draft = self._ensure_provider_settings_draft()
        draft.translation = copy.deepcopy(current_settings.translation)
        draft.translation.model = model
        draft.translation.connection = connection
        draft.translation.connection_history = copy.deepcopy(
            current_settings.translation.connection_history
        )
        draft.translation.connection_history[model.value] = connection
        if model == TranslationModel.CUSTOM_HTTP and old_model != TranslationModel.CUSTOM_HTTP:
            draft.translation.previous_llm_model = old_model
        elif model != TranslationModel.CUSTOM_HTTP:
            draft.translation.previous_llm_model = None
        materialize_translation_settings(draft)
        new_provider = draft.provider.llm

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
            connection in (TranslationConnection.MANAGED, TranslationConnection.MANAGED_CHINA)
            or model == TranslationModel.MANAGED_GEMMA
            or old_model == TranslationModel.MANAGED_GEMMA
        ) and getattr(self, "on_providers_changed", None) is not None:
            self.on_providers_changed()

        display_settings = self._build_settings_with_provider_draft()
        assert display_settings is not None
        self._sync_translation_selection_controls(display_settings)

        if old_provider != display_settings.provider.llm:
            provider_name = self._active_prompt_key()
            self._prompt_editor.set_provider(provider_name)
            next_prompt = self._ensure_provider_prompt_value(draft, provider_name)
            self._prompt_editor.value = next_prompt
            draft.system_prompt = next_prompt
        self._sync_prompt_tab_copy()

        if is_control_mounted(self):
            self._qwen_region_btn.update()
            self._repaint_managed_key_card()
            self._llm_text.update()
            self._translation_connection_row.update()
            self._local_llm_connection_card.update()
            http_extension_row = getattr(self, "_http_extension_row", None)
            if http_extension_row is not None:
                http_extension_row.update()

    def _on_llm_selected(self, value: str) -> None:
        """Handle LLM provider selection from modal."""
        if not self._settings:
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
        history = copy.deepcopy(current_settings.translation.connection_history)
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
        if model == TranslationModel.MANAGED_GEMMA:
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
        if not self._settings:
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
        if (
            display_settings is not None
            and display_settings.translation.model == TranslationModel.MANAGED_GEMMA
        ):
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
        if not self._settings:
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
        draft.translation = copy.deepcopy(current_settings.translation)
        draft.translation.fallback = copy.deepcopy(new_value)
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
        current = self._settings.ui.locale if self._settings else "en"
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
        if not self._settings:
            return
        old_locale = self._settings.ui.locale
        self._emit_runtime_basic(f"[Settings] Language changed: {old_locale} -> {value}")
        self._settings.ui.locale = value

        # Update text
        self._ui_text.content.value = locale_label(value)
        if is_control_mounted(self):
            self._ui_text.update()
        self._emit_settings_changed()

    def _on_qwen_region_click(self, e) -> None:
        """Open Qwen region selection modal."""
        if not is_control_mounted(self):
            return
        options = [OptionItem(value=r.value, label=t(f"region.{r.value}")) for r in QwenRegion]
        display_settings = self._build_settings_with_provider_draft()
        current = (
            display_settings.qwen.region.value
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
        if not self._settings:
            return

        current_settings = self._build_settings_with_provider_draft()
        assert current_settings is not None
        old_region = current_settings.qwen.region.value
        if old_region == value:
            return
        self._emit_runtime_detailed(f"[Settings] Qwen region changed: {old_region} -> {value}")
        draft = self._ensure_provider_settings_draft()
        draft.qwen.region = QwenRegion(value)
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
        if settings.api_key_verified.openrouter and self._openrouter_key.value:
            return
        if settings.provider.llm != LLMProviderName.OPENROUTER:
            return
        if settings.openrouter.selected_source != OpenRouterCredentialSource.BYOK:
            return
        profile = self._openrouter_selection_profile(settings)
        if profile is None or profile.openrouter_source != OpenRouterCredentialSource.BYOK.value:
            return

        target = copy.deepcopy(settings)
        target.provider.llm = LLMProviderName.OPENROUTER
        target.openrouter.selection_alias = OpenRouterSelectionAlias(profile.alias)
        target.openrouter.selected_source = OpenRouterCredentialSource.BYOK
        assert profile.openrouter_model is not None
        target.openrouter.llm_model = OpenRouterLLMModel(profile.openrouter_model)
        target.system_prompt = self._ensure_provider_prompt_value(target, "openrouter")
        self.on_request_openrouter_pkce(target)

    def _write_secret_value(self, key: str, value: str) -> bool:
        if not self._settings or not self._config_path:
            return False

        try:
            store = create_secret_store(self._settings.secrets, config_path=self._config_path)
            if value:
                store.set(key, value)
            else:
                store.delete(key)
            return True
        except Exception as exc:
            self._emit_runtime_basic(
                f"Failed to update secret {key}: {type(exc).__name__}",
                level=logging.WARNING,
            )
            return False

    def _sync_custom_stt_card(self, settings: AppSettings | None = None) -> None:
        if getattr(self, "_custom_stt_connection_card", None) is None:
            return
        current = settings or self._build_settings_with_provider_draft()
        if current is None:
            return
        custom = current.custom_stt
        self._custom_stt_endpoint.value = custom.endpoint
        self._custom_stt_endpoint.error = None
        self._custom_stt_model.value = custom.model
        self._custom_stt_extra.value = _custom_stt_extra_to_text(custom.extra)
        self._clear_custom_stt_extra_error()
        if is_control_mounted(self):
            _update_control_if_mounted(self._custom_stt_connection_card)

    def _on_custom_stt_field_change(self, e) -> None:
        _ = e
        if not self._settings:
            return
        current = self._build_settings_with_provider_draft()
        if current is None or not (
            is_custom_stt_provider(current.provider.stt)
            or is_custom_stt_provider(current.provider.peer_stt)
        ):
            return
        self._ensure_provider_settings_draft()
        self.has_provider_changes = True

    def _on_custom_stt_endpoint_change_end(self, e) -> None:
        _ = e
        if not self._settings:
            return
        endpoint = (self._custom_stt_endpoint.value or "").strip()
        current = self._provider_settings_draft or self._settings
        if current.custom_stt.endpoint != endpoint:
            draft = self._ensure_provider_settings_draft()
            draft.custom_stt.endpoint = endpoint
            self.has_provider_changes = True
        self._custom_stt_endpoint.value = endpoint
        _update_control_if_mounted(self._custom_stt_endpoint)

    def _on_custom_stt_model_change_end(self, e) -> None:
        _ = e
        if not self._settings:
            return
        model = (self._custom_stt_model.value or "").strip()
        current = self._provider_settings_draft or self._settings
        if current.custom_stt.model != model:
            draft = self._ensure_provider_settings_draft()
            draft.custom_stt.model = model
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
        if not self._settings:
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
        current = self._provider_settings_draft or self._settings
        if current.custom_stt.extra != normalized:
            draft = self._ensure_provider_settings_draft()
            draft.custom_stt.extra = normalized
            self.has_provider_changes = True
        self._custom_stt_extra.value = _custom_stt_extra_to_text(normalized)
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
        if not self._settings or not self._config_path:
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
        if not self._settings:
            return

        new_host = self._audio_settings.host_api
        new_device = self._audio_settings.microphone
        new_desktop_output = self._audio_settings.desktop_output_device
        old_host = self._settings.audio.input_host_api
        old_device = self._settings.audio.input_device
        old_desktop_output = self._settings.desktop_audio.output_device

        if old_host != new_host:
            self._emit_runtime_basic(f"[Settings] Audio Host changed: {old_host} -> {new_host}")
        if old_device != new_device:
            self._emit_runtime_basic(f"[Settings] Microphone changed: {old_device} -> {new_device}")
        if old_desktop_output != new_desktop_output:
            self._emit_runtime_basic(
                f"[Settings] Desktop loopback output changed: {old_desktop_output} -> {new_desktop_output}"
            )

        self._settings.audio.input_host_api = new_host
        self._settings.audio.input_device = new_device
        self._settings.desktop_audio.output_device = new_desktop_output
        self._emit_settings_changed()

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
        if self._settings is None:
            return OVERLAY_TARGET_STEAMVR
        return self._normalized_overlay_target(self._settings.overlay.target)

    def _overlay_target_label_for(self, target: object) -> str:
        normalized_target = self._normalized_overlay_target(target)
        return t(f"settings.overlay.target.{normalized_target}")

    def _sync_overlay_target_control(self) -> None:
        self._set_unit_card_value_text(
            self._overlay_target_button,
            self._overlay_target_label_for(self._current_overlay_target()),
            size=28,
        )
        self._overlay_target_button.disabled = self._settings is None

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
        if self._settings is None:
            return "medium"
        return self._normalize_desktop_overlay_size_preset(
            self._settings.overlay.desktop_flet.size_preset
        )

    def _current_desktop_overlay_background_alpha(self) -> float:
        if self._settings is None:
            return DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA
        return self._normalize_desktop_overlay_background_alpha(
            self._settings.overlay.desktop_flet.visual.background_alpha
        )

    def _desktop_overlay_lock_label_for(self, locked: bool) -> str:
        return t(
            "settings.overlay.desktop.lock.value.locked"
            if locked
            else "settings.overlay.desktop.lock.value.move"
        )

    def _current_desktop_overlay_locked(self) -> bool:
        if self._settings is None:
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
        self._desktop_overlay_background_alpha_value_text.value = (
            self._desktop_overlay_background_alpha_label_for(
                self._current_desktop_overlay_background_alpha()
            )
        )
        disabled = self._settings is None
        self._desktop_overlay_size_button.disabled = disabled
        self._desktop_overlay_background_alpha_decrease_button.disabled = disabled
        self._desktop_overlay_background_alpha_increase_button.disabled = disabled
        self._desktop_overlay_lock_button.disabled = disabled
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
        if not is_control_mounted(self) or not self._settings:
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
        if not self._settings:
            return
        target = self._normalized_overlay_target(value)
        if self._current_overlay_target() == target:
            return
        self._settings.overlay.target = target
        if self._overlay_state == "off":
            self._overlay_runtime_target = target
        self._sync_overlay_controls()
        self._emit_settings_changed()

    def _on_desktop_overlay_size_click(self, e) -> None:
        _ = e
        if (
            not is_control_mounted(self)
            or not self._settings
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
        if not self._settings:
            return
        size_preset = self._normalize_desktop_overlay_size_preset(value)
        if self._current_desktop_overlay_size_preset() == size_preset:
            return
        if self.on_desktop_overlay_size_change:
            self._desktop_overlay_pending_size_preset = size_preset
            self._sync_desktop_overlay_main_controls()
            self.on_desktop_overlay_size_change(size_preset)
            return
        self._settings.overlay.desktop_flet.size_preset = size_preset
        self._desktop_overlay_pending_size_preset = None
        self._sync_desktop_overlay_main_controls()
        self._emit_settings_changed()

    def _on_desktop_overlay_lock_click(self, e) -> None:
        _ = e
        if not self._settings or self._desktop_overlay_lock_button.disabled:
            return
        next_value = "move" if self._current_desktop_overlay_locked() else "locked"
        self._on_desktop_overlay_lock_selected(next_value)

    def _on_desktop_overlay_lock_selected(self, value: str) -> None:
        if not self._settings:
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
        if not self._settings or self._desktop_overlay_background_alpha_decrease_button.disabled:
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
        updated = copy.deepcopy(self._settings)
        desktop_visual = updated.overlay.desktop_flet.visual
        desktop_visual.background_alpha = next_alpha
        desktop_visual.validate()
        self._settings = updated
        self._sync_desktop_overlay_main_controls()
        if is_control_mounted(self):
            self.update()
        self._emit_settings_changed()

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
        if self._settings is not None:
            self._settings.overlay.calibration = calibration.copy()
        self._sync_overlay_calibration_controls(self._overlay_calibration)

        if is_control_mounted(self):
            self.update()

        if self.on_overlay_calibration_apply is None:
            self._emit_settings_changed()

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
        if not is_control_mounted(self) or not self._settings:
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
        if not is_control_mounted(self) or not self._settings:
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
        if not self._settings or self._overlay_desktop_reset_button.disabled:
            return
        if self.on_desktop_overlay_position_reset:
            self._desktop_overlay_pending_position_reset = True
            self._desktop_overlay_captions_locked = False
            self._sync_desktop_overlay_main_controls()
            self.on_desktop_overlay_position_reset()
            return
        desktop_settings = self._settings.overlay.desktop_flet
        desktop_settings.position.x = None
        desktop_settings.position.y = None
        desktop_settings.locked = False
        desktop_settings.validate()
        self._desktop_overlay_captions_locked = False
        self._desktop_overlay_pending_position_reset = False
        self._sync_desktop_overlay_main_controls()
        self._emit_settings_changed()

    def sync_desktop_overlay_settings(self, settings: AppSettings) -> None:
        self._settings = settings
        self._desktop_overlay_pending_size_preset = None
        self._desktop_overlay_pending_position_reset = False
        self._desktop_overlay_pending_locked = None
        self._desktop_overlay_captions_locked = False
        if self._overlay_state == "off":
            self._overlay_runtime_target = self._current_overlay_target()
        self._sync_overlay_controls()

    def set_overlay_peer_contract(self, contract: OverlayPeerConsumerContract) -> None:
        self._overlay_peer_contract = contract
        if self._settings is not None:
            self._settings.ui.overlay_enabled = contract.overlay.intent_enabled
            self._settings.ui.peer_translation_enabled = contract.peer.intent_enabled
            self._update_api_visibility()
            if is_control_mounted(self):
                self._api_keys_column.update()
        self._sync_overlay_controls()

    def _sync_overlay_controls(self) -> None:
        overlay_translation_enabled = bool(
            self._settings and self._settings.overlay.show_translation
        )
        overlay_peer_original_enabled = bool(
            self._settings and self._settings.overlay.show_peer_original
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

        self._overlay_translation_button.disabled = self._settings is None
        self._overlay_peer_original_button.disabled = self._settings is None
        self._overlay_target_button.disabled = self._settings is None
        self._overlay_anchor_button.disabled = self._settings is None
        self._overlay_distance_decrease_button.disabled = self._settings is None
        self._overlay_distance_increase_button.disabled = self._settings is None
        self._overlay_offset_x_decrease_button.disabled = self._settings is None
        self._overlay_offset_x_increase_button.disabled = self._settings is None
        self._overlay_offset_y_decrease_button.disabled = self._settings is None
        self._overlay_offset_y_increase_button.disabled = self._settings is None
        self._desktop_overlay_background_alpha_decrease_button.disabled = self._settings is None
        self._desktop_overlay_background_alpha_increase_button.disabled = self._settings is None
        self._overlay_vr_reset_button.disabled = self._settings is None
        self._overlay_desktop_reset_button.disabled = self._settings is None
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
        if not self._settings or self._overlay_translation_button.disabled:
            return
        next_value = "off" if self._settings.overlay.show_translation else "on"
        self._on_overlay_translation_selected(next_value)

    def _on_overlay_translation_selected(self, value: str) -> None:
        if not self._settings:
            return
        self._settings.overlay.show_translation = value == "on"
        self._sync_overlay_controls()
        self._emit_settings_changed()

    def _on_overlay_peer_original_click(self, e) -> None:
        if not self._settings or self._overlay_peer_original_button.disabled:
            return
        next_value = "off" if self._settings.overlay.show_peer_original else "on"
        self._on_overlay_peer_original_selected(next_value)

    def _on_overlay_peer_original_selected(self, value: str) -> None:
        if not self._settings:
            return
        self._settings.overlay.show_peer_original = value == "on"
        self._sync_overlay_controls()
        self._emit_settings_changed()

    def _handle_vad_visual_change(self, e) -> None:
        self._vad_slider.label = f"{float(e.control.value):.2f}"
        _update_control_if_mounted(self._vad_slider)

    def _handle_vad_change(self, e) -> None:
        if not self._settings:
            return

        new_vad = float(e.control.value)
        old_vad = self._settings.stt.vad_speech_threshold

        if abs(old_vad - new_vad) > 0.001:
            self._emit_runtime_detailed(
                f"[Settings] VAD sensitivity changed: {old_vad:.2f} -> {new_vad:.2f}"
            )

        self._settings.stt.vad_speech_threshold = new_vad
        self._emit_settings_changed()

    def _handle_peer_vad_visual_change(self, e) -> None:
        self._peer_vad_slider.label = f"{float(e.control.value):.2f}"
        _update_control_if_mounted(self._peer_vad_slider)

    def _handle_peer_vad_change(self, e) -> None:
        if not self._settings:
            return

        new_vad = float(e.control.value)
        old_vad = self._settings.desktop_audio.vad_speech_threshold

        if abs(old_vad - new_vad) > 0.001:
            self._emit_runtime_detailed(
                f"[Settings] Peer VAD threshold changed: {old_vad:.2f} -> {new_vad:.2f}"
            )

        self._settings.desktop_audio.vad_speech_threshold = new_vad
        self._peer_vad_field.value = f"{new_vad:.2f}"
        self._peer_vad_slider.label = f"{new_vad:.2f}"
        _update_control_if_mounted(self._peer_vad_field)
        _update_control_if_mounted(self._peer_vad_slider)
        self._emit_settings_changed()

    def _on_peer_vad_threshold_change(self, e) -> None:
        if not self._settings:
            return

        old_value = self._settings.desktop_audio.vad_speech_threshold
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

        self._settings.desktop_audio.vad_speech_threshold = new_value
        self._peer_vad_field.value = f"{new_value:.2f}"
        _update_control_if_mounted(self._peer_vad_field)
        self._emit_settings_changed()

    def _on_peer_hangover_change(self, e) -> None:
        if not self._settings:
            return

        old_value = self._settings.desktop_audio.vad_hangover_ms
        new_value = self._parse_setting_int(
            e.control.value,
            fallback=old_value,
            minimum=0,
        )
        if old_value != new_value:
            self._emit_runtime_detailed(
                f"[Settings] Peer hangover changed: {old_value} -> {new_value}"
            )

        self._settings.desktop_audio.vad_hangover_ms = new_value
        self._peer_hangover_field.value = str(new_value)
        _update_control_if_mounted(self._peer_hangover_field)
        self._emit_settings_changed()

    def _on_peer_pre_roll_change(self, e) -> None:
        if not self._settings:
            return

        old_value = self._settings.desktop_audio.vad_pre_roll_ms
        new_value = self._parse_setting_int(
            e.control.value,
            fallback=old_value,
            minimum=0,
        )
        if old_value != new_value:
            self._emit_runtime_detailed(
                f"[Settings] Peer pre-roll changed: {old_value} -> {new_value}"
            )

        self._settings.desktop_audio.vad_pre_roll_ms = new_value
        self._peer_pre_roll_field.value = str(new_value)
        _update_control_if_mounted(self._peer_pre_roll_field)
        self._emit_settings_changed()

    def _on_vrc_mic_click(self, e) -> None:
        """Toggle VRC mic intercept immediately from the unit card."""
        if not self._settings:
            return
        next_value = "off" if self._settings.osc.vrc_mic_intercept else "on"
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
        if not self._settings:
            return
        new_value = value == "on"
        self._emit_runtime_basic(f"[Settings] VRC mic intercept toggled: {new_value}")
        self._settings.osc.vrc_mic_intercept = new_value

        self._vrc_mic_text.content.value = t(
            "settings.vrc_mic.on" if new_value else "settings.vrc_mic.off"
        )
        if is_control_mounted(self):
            self._vrc_mic_text.update()
        self._emit_settings_changed()

    def _on_chatbox_source_click(self, e) -> None:
        """Open chatbox source inclusion selection modal."""
        if not is_control_mounted(self):
            return
        options = [
            OptionItem(value="on", label=t("settings.chatbox_source.on")),
            OptionItem(value="off", label=t("settings.chatbox_source.off")),
        ]
        current = "on" if self._settings.osc.chatbox_include_source else "off"
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
        if not self._settings:
            return
        new_value = value == "on"
        self._emit_runtime_basic(f"[Settings] Chatbox include source toggled: {new_value}")
        self._settings.osc.chatbox_include_source = new_value

        self._chatbox_source_text.content.value = t(
            "settings.chatbox_source.on" if new_value else "settings.chatbox_source.off"
        )
        if is_control_mounted(self):
            self._chatbox_source_text.update()
        self._emit_settings_changed()

    def _on_osc_connection_click(self, e) -> None:
        _ = e
        if not self._settings or not is_control_mounted(self):
            return
        self._osc_connection_modal = OscConnectionModal(
            self.page,
            self._on_osc_connection_selected,
            effective_ports_provider=self.on_osc_effective_ports,
        )
        self._osc_connection_modal.open(
            self._settings.osc.connection_mode,
            int(self._settings.osc.send_port or self._settings.osc.port),
            int(self._settings.osc.receive_port),
        )

    def _on_osc_connection_selected(self, mode: str, send_port: int, receive_port: int) -> None:
        if not self._settings:
            return
        if mode not in {"automatic", "manual", "off"}:
            return
        candidate = copy.deepcopy(self._settings)
        candidate.osc.connection_mode = mode
        candidate.osc.send_port = int(send_port)
        candidate.osc.receive_port = int(receive_port)
        try:
            candidate.osc.validate()
        except (TypeError, ValueError):
            return
        self._settings.osc.connection_mode = mode
        self._settings.osc.send_port = int(send_port)
        self._settings.osc.receive_port = int(receive_port)
        self._sync_osc_connection_card(self._settings)
        if is_control_mounted(self):
            self._osc_connection_text.update()
        self._emit_settings_changed()

    def _on_clipboard_auto_translate_click(self, e) -> None:
        """Toggle clipboard auto-translate immediately from the unit card."""
        if not self._settings:
            return
        next_value = "off" if self._settings.ui.clipboard_auto_translate_enabled else "on"
        self._on_clipboard_auto_translate_selected(next_value)

    def _on_clipboard_auto_translate_selected(self, value: str) -> None:
        """Handle clipboard auto-translate selection result."""
        if not self._settings:
            return
        new_value = value == "on"
        self._emit_runtime_basic(f"[Settings] Clipboard auto translate toggled: {new_value}")
        self._settings.ui.clipboard_auto_translate_enabled = new_value
        self._clipboard_auto_translate_text.content.value = t(
            "settings.clipboard_auto_translate.on"
            if new_value
            else "settings.clipboard_auto_translate.off"
        )
        if is_control_mounted(self):
            self._clipboard_auto_translate_text.update()
        self._emit_settings_changed()

    def _on_telemetry_consent_click(self, e) -> None:
        _ = e
        if not is_control_mounted(self) or not self._settings:
            return

        def _select(value: str) -> None:
            if value not in {"allow", "decline"} or self._settings is None:
                return
            updated = with_telemetry_consent(self._settings, value)
            self._settings = updated
            self._sync_telemetry_consent_card(updated)
            if self.on_telemetry_consent_change is not None:
                self.on_telemetry_consent_change(value)

        options = [
            OptionItem(
                "allow",
                t("settings.telemetry.option.allow"),
                t("settings.telemetry.option.allow.description"),
            ),
            OptionItem(
                "decline",
                t("settings.telemetry.option.decline"),
                "",
            ),
        ]
        modal = SettingsModal(
            self.page,
            t("settings.telemetry.modal.title"),
            options,
            _select,
            show_description=True,
        )
        modal.open("decline" if self._settings.telemetry.consent == "decline" else "allow")

    def _on_prompt_change(self, value: str) -> None:
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
        if not self._settings:
            return

        source_language = self._current_source_language()
        updated_terms = dict(self._settings.stt.custom_terms)
        current_terms = list(updated_terms.get(source_language, []))
        applied_terms = list(next_terms)
        updated_terms[source_language] = applied_terms
        next_enabled = any(bool(terms) for terms in updated_terms.values())

        if (
            current_terms == applied_terms
            and self._settings.stt.custom_vocabulary_enabled == next_enabled
        ):
            return

        self._settings.stt.custom_terms = updated_terms
        self._settings.stt.custom_vocabulary_enabled = next_enabled
        self._custom_vocab_tag_editor.set_terms(applied_terms)
        self._emit_runtime_detailed(
            f"[Settings] Custom vocabulary applied: language={source_language}, terms={len(applied_terms)}"
        )
        self._emit_settings_changed()

    def _on_custom_vocabulary_add_terms(self, raw_terms: list[str]) -> None:
        if not self._settings:
            return

        raw_values = [str(term) for term in raw_terms]
        if any(value != "" for value in raw_values):
            self._custom_vocab_tag_editor.clear_input()
        submitted_terms = self._normalize_custom_vocabulary_submitted_terms(raw_values)
        if not submitted_terms:
            return

        source_language = self._current_source_language()
        current_terms = list(self._settings.stt.custom_terms.get(source_language, []))
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

        updated_terms = dict(self._settings.stt.custom_terms)
        updated_terms[source_language] = list(next_terms)
        next_enabled = any(bool(terms) for terms in updated_terms.values())
        will_change = (
            current_terms != next_terms
            or self._settings.stt.custom_vocabulary_enabled != next_enabled
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
        if not self._settings:
            return

        source_language = self._current_source_language()
        current_terms = list(self._settings.stt.custom_terms.get(source_language, []))
        try:
            current_terms.remove(term)
        except ValueError:
            return
        self._set_custom_vocabulary_terms_for_current_language(current_terms)

    async def _verify_key(self, provider: str, key: str) -> tuple[bool, str]:
        """Verify API key."""
        if self.on_verify_api_key:
            result = await self.on_verify_api_key(provider, key)
            if provider == "openrouter":
                self._sync_openrouter_pkce_button_state()
            return result
        return False, "Verification not available"

    def _emit_settings_changed(self) -> None:
        if self._settings and self.on_settings_changed:
            settings = copy.deepcopy(self._settings)
            self.on_settings_changed(
                self._sanitize_provider_apply_settings(
                    self._settings_with_desktop_overlay_runtime_state(settings)
                )
            )

    def _emit_prompt_apply_settings(self, settings: AppSettings) -> None:
        sanitized = self._sanitize_provider_apply_settings(settings)
        if sanitized is None:
            return
        if self.on_prompt_apply_settings:
            self.on_prompt_apply_settings(sanitized)
            return
        if self.on_settings_changed:
            self.on_settings_changed(sanitized)

    # --- Locale ---
    def apply_locale(self) -> None:
        """Update all labels when locale changes."""
        self._settings_subtab_shell.set_font_family(font_for_language(get_locale()))
        for key in _SETTINGS_SUBTAB_ORDER:
            self._settings_subtab_shell.set_tab_label(key, self._settings_subtab_label(key))

        # Section titles
        self._stt_title.value = t("settings.section.stt")
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
        self._telemetry_consent_title.value = t("settings.telemetry.title")
        self._peer_provider_title.value = t("settings.section.peer_stt")
        self._dashboard_language_redirect_text.value = t("settings.dashboard_language_redirect")
        self._peer_stt_label.value = t("settings.peer_stt_provider")
        self._gpu_device_title.value = t("settings.gpu_device.title")
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
            self._set_unit_card_value_text(
                self._stt_text,
                self._stt_provider_display_label(
                    display_settings.provider.stt,
                    custom_mode=display_settings.custom_stt.mode,
                ),
            )
            self._set_unit_card_value_text(
                self._peer_stt_text,
                self._stt_provider_display_label(
                    self._effective_peer_stt_provider(display_settings),
                    custom_mode=display_settings.custom_stt.mode,
                ),
            )
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
            self._ui_text.content.value = locale_label(display_settings.ui.locale)
            self._vrc_mic_text.content.value = t(
                "settings.vrc_mic.on"
                if display_settings.osc.vrc_mic_intercept
                else "settings.vrc_mic.off"
            )
            self._sync_osc_connection_card(display_settings)
            self._chatbox_source_text.content.value = t(
                "settings.chatbox_source.on"
                if display_settings.osc.chatbox_include_source
                else "settings.chatbox_source.off"
            )
            self._clipboard_auto_translate_text.content.value = t(
                "settings.clipboard_auto_translate.on"
                if display_settings.ui.clipboard_auto_translate_enabled
                else "settings.clipboard_auto_translate.off"
            )
            self._sync_telemetry_consent_card(display_settings)
            self._set_unit_card_value_text(
                self._microphone_test_text,
                t("settings.microphone_test.action"),
            )
            self._sync_overlay_controls()
            self._sync_overlay_calibration_controls()

        # Qwen Region label
        if display_settings:
            region_val = display_settings.qwen.region.value
            _set_text_button_label(
                self._qwen_region_btn,
                f"{t('settings.qwen_region')} {t(f'region.{region_val}')}",
            )

        # Components
        self._deepgram_key.apply_locale()
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
