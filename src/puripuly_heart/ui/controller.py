from __future__ import annotations

import asyncio
import contextlib
import copy
import inspect
import json
import logging
import math
import os
import secrets
import sys
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

import flet as ft
import numpy as np

from puripuly_heart.app.wiring import (
    build_peer_stt_provider_signature,
    create_llm_provider,
    create_peer_stt_backend,
    create_secret_store,
    create_stt_backend,
    resolve_peer_stt_config,
)
from puripuly_heart.config.audio_host_api import normalize_input_host_api
from puripuly_heart.config.llm_profiles import profile_for_alias
from puripuly_heart.config.settings import (
    DESKTOP_FLET_MIN_HEIGHT,
    DESKTOP_FLET_MIN_WIDTH,
    DESKTOP_FLET_SIZE_PRESETS,
    OVERLAY_TARGET_DESKTOP,
    OVERLAY_TARGET_STEAMVR,
    AppSettings,
    LLMProviderName,
    OpenRouterCredentialSource,
    OpenRouterLLMModel,
    OpenRouterProviderRouting,
    OpenRouterSelectionAlias,
    QwenLLMModel,
    QwenRegion,
    STTProviderName,
    TranslationConnection,
    load_settings,
    new_settings_for_first_run,
    normalize_owned_referral_id,
    save_settings,
)
from puripuly_heart.config.vad_defaults import DEFAULT_STABLE_VAD_HANGOVER_MS
from puripuly_heart.core.audio.desktop_pipeline import DesktopPeerPipeline
from puripuly_heart.core.audio.desktop_source import DesktopLoopbackAudioSource
from puripuly_heart.core.audio.diagnostics import compute_audio_frame_metrics
from puripuly_heart.core.audio.gate import VrcMicAudioGate
from puripuly_heart.core.audio.source import (
    AudioSource,
    MicrophoneTestRouteObservation,
    SelfMicCaptureChannelDecision,
    SoundDeviceAudioSource,
    determine_self_mic_capture_channels,
    observe_microphone_test_route,
    resolve_sounddevice_input_device,
)
from puripuly_heart.core.clipboard.watcher import create_clipboard_watcher
from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.hardware_fingerprint import get_raw_hardware_fingerprint
from puripuly_heart.core.llm.provider import SemaphoreLLMProvider
from puripuly_heart.core.local_stt_assets import (
    LocalSTTInstallState,
    LocalSTTManifestInvalidError,
    LocalSTTModelMissingError,
    inspect_local_stt_install_state,
)
from puripuly_heart.core.local_stt_runtime_installer import (
    LocalSTTRuntimeInstallCancelled,
    LocalSTTRuntimeInstallError,
    RuntimeLocalSTTStatusUpdate,
    ensure_local_stt_installed,
)
from puripuly_heart.core.managed_openrouter_broker_client import (
    HttpManagedOpenRouterBrokerClient,
)
from puripuly_heart.core.managed_openrouter_release import (
    ManagedOpenRouterReleaseBehavior,
    ManagedOpenRouterReleaseResult,
    ManagedOpenRouterReleaseService,
    ManagedOpenRouterStatusRefreshResult,
    TalkTogetherPassStatus,
    UnavailableManagedOpenRouterReleaseClient,
    format_managed_openrouter_diagnostics,
)
from puripuly_heart.core.openrouter_credentials import (
    OPENROUTER_BYOK_API_KEY_SECRET,
    resolve_openrouter_credentials,
)
from puripuly_heart.core.openrouter_handoff import (
    is_effectively_exhausted,
    mark_founder_letter_shown,
    should_auto_show_founder_letter,
)
from puripuly_heart.core.openrouter_pkce import OpenRouterPKCEClient
from puripuly_heart.core.orchestrator.hub import ClientHub
from puripuly_heart.core.osc.chatbox_paginator import ChatboxPaginator
from puripuly_heart.core.osc.receiver import (
    VRC_OSC_RECEIVER_HOST,
    VRC_OSC_RECEIVER_PORT,
    VrcMicState,
    VrcOscReceiver,
)
from puripuly_heart.core.osc.udp_sender import VrchatOscUdpSender
from puripuly_heart.core.overlay.bridge import OverlayBridge
from puripuly_heart.core.overlay.diagnostics import OverlayDiagnosticsRecorder
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.process import (
    DefaultOverlayProcessRunner,
    DesktopFletOverlayRunner,
    OverlayProcessManager,
    OverlayProcessRunner,
)
from puripuly_heart.core.runtime.peer_channel import PeerChannelRuntime, PeerRuntimeConfig
from puripuly_heart.core.runtime_logging import SessionLoggingMode, SessionRuntimeLoggingService
from puripuly_heart.core.stt.controller import (
    FinalTranscriptSuppressedNotification,
    ManagedSTTProvider,
)
from puripuly_heart.core.stt.custom_vocab import get_effective_custom_terms
from puripuly_heart.core.vad.bundled import SILERO_VAD_VERSION, ensure_silero_vad_onnx
from puripuly_heart.core.vad.gating import VadGating, create_peer_vad_gating
from puripuly_heart.core.vad.silero import SileroVadOnnx
from puripuly_heart.providers.llm.cerebras import CerebrasLLMProvider
from puripuly_heart.providers.llm.deepseek import DeepSeekLLMProvider
from puripuly_heart.providers.llm.gemini import GeminiLLMProvider
from puripuly_heart.providers.llm.openrouter import OpenRouterKeyMetadata, OpenRouterLLMProvider
from puripuly_heart.providers.llm.qwen import QwenLLMProvider
from puripuly_heart.providers.llm.qwen_async import AsyncQwenLLMProvider
from puripuly_heart.providers.stt.deepgram import DeepgramRealtimeSTTBackend
from puripuly_heart.providers.stt.local_qwen_sherpa import LocalQwenSherpaLoadError
from puripuly_heart.providers.stt.soniox import SonioxRealtimeSTTBackend
from puripuly_heart.ui.event_bridge import UIEventBridge
from puripuly_heart.ui.i18n import get_locale, set_locale, t
from puripuly_heart.ui.overlay_calibration import OverlayCalibration
from puripuly_heart.ui.overlay_peer_contract import (
    OverlayPeerConsumerContract,
    build_overlay_peer_consumer_contract,
)
from puripuly_heart.ui.views.logs import FletLogHandler

logger = logging.getLogger(__name__)

# Hardcoded STT session reset deadline (not configurable via settings)
STT_RESET_DEADLINE_S = 300.0
OVERLAY_STARTUP_TIMEOUT_MS = 3000
OVERLAY_SHUTDOWN_GRACE_S = 0.05
DESKTOP_BOUNDS_PERSIST_DEBOUNCE_S = 0.05
DESKTOP_INTERACTION_MODE_EDIT = "edit"
DESKTOP_INTERACTION_MODE_PASS_THROUGH = "pass_through"
DESKTOP_INTERACTION_MODES = frozenset(
    {DESKTOP_INTERACTION_MODE_EDIT, DESKTOP_INTERACTION_MODE_PASS_THROUGH}
)
_PASS_STATUS_UNSET = object()
_OVERLAY_FAILURE_REASONS = frozenset(
    {
        "missing_executable",
        "spawn_failed",
        "manifest_invalid",
        "contract_mismatch",
        "bridge_auth_failed",
        "startup_timeout",
        "stale_overlay_build",
        "vendored_openvr_dll_missing",
        "packaged_openvr_dll_missing",
        "openvr_dll_hash_mismatch",
        "steamvr_not_installed",
        "steamvr_not_running",
        "hmd_not_found",
        "openvr_init_failed",
        "renderer_init_failed",
        "runtime_disconnected",
        "window_configuration_failed",
        "runtime_control_invalid",
        "runtime_crashed",
        "unknown",
    }
)
GITHUB_STAR_PROMPT_MANAGED_REMAINING_PERCENT_THRESHOLD = 60
GITHUB_STAR_PROMPT_ELIGIBLE_LAUNCH_THRESHOLD = 3
GITHUB_STAR_PROMPT_RECENCY_WINDOW = timedelta(days=14)
_GITHUB_STAR_PROMPT_MANAGED_CONNECTIONS = frozenset(
    {
        TranslationConnection.MANAGED,
        TranslationConnection.MANAGED_CHINA,
    }
)
_GITHUB_STAR_PROMPT_USER_OWNED_CLOUD_CONNECTIONS = frozenset(
    {
        TranslationConnection.OPENROUTER,
        TranslationConnection.OFFICIAL_BYOK,
    }
)
DISCORD_AUTH_ERROR_KEY_BY_SUBCODE = {
    "discord_email_unverified": "discord_auth.error.email_unverified",
    "discord_account_too_new": "discord_auth.error.account_too_new",
    "discord_lifetime_used": "discord_auth.error.lifetime_used",
    "hardware_duplicate": "discord_auth.error.hardware_duplicate",
    "global_cap_reached": "discord_auth.error.daily_cap",
    "oauth_session_expired": "discord_auth.error.expired",
    "loopback_unavailable": "discord_auth.error.loopback_unavailable",
}
_MICROPHONE_TEST_LEVEL_INTERVAL_S = 1.0
LOCAL_QWEN_HALLUCINATION_GUIDANCE_TRIGGER_COUNT = 2


def _mic_test_log_value(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, float):
        return str(value)
    return str(value)


def _mic_test_db(value: float) -> float:
    if value <= 0.0:
        return -120.0
    return round(float(20.0 * math.log10(max(value, 1e-6))), 1)


@dataclass(slots=True)
class _MicrophoneTestLevelStats:
    frames: int = 0
    audio_ms: float = 0.0
    sample_count: int = 0
    square_sum: float = 0.0
    peak_abs: float = 0.0
    zero_count: int = 0

    def add_frame(self, frame) -> None:  # noqa: ANN001
        metrics = compute_audio_frame_metrics(frame)
        samples = np.asarray(frame.samples, dtype=np.float32)
        self.frames += 1
        self.audio_ms += metrics.audio_ms
        self.sample_count += int(samples.size)
        if samples.size == 0:
            return

        abs_samples = np.abs(samples)
        self.square_sum += float(np.sum(np.square(samples, dtype=np.float32)))
        self.peak_abs = max(self.peak_abs, float(np.max(abs_samples)))
        self.zero_count += int(np.count_nonzero(abs_samples < 1e-6))

    @property
    def rms_db(self) -> float:
        if self.sample_count <= 0:
            return -120.0
        return _mic_test_db(math.sqrt(self.square_sum / float(self.sample_count)))

    @property
    def peak_db(self) -> float:
        return _mic_test_db(self.peak_abs)

    @property
    def zero_ratio(self) -> float:
        if self.sample_count <= 0:
            return 1.0
        return round(float(self.zero_count) / float(self.sample_count), 3)

    def reset(self) -> None:
        self.frames = 0
        self.audio_ms = 0.0
        self.sample_count = 0
        self.square_sum = 0.0
        self.peak_abs = 0.0
        self.zero_count = 0


def _canonical_json_signature(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _callable_accepts_keyword(callable_obj: object, keyword: str) -> bool:
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return True
    return keyword in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _github_star_prompt_utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _github_star_prompt_utc_timestamp(value: datetime | None = None) -> str:
    resolved = value or _github_star_prompt_utc_now()
    if resolved.tzinfo is None:
        resolved = resolved.replace(tzinfo=timezone.utc)
    return (
        resolved.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )


def _parse_github_star_prompt_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    parse_value = f"{normalized[:-1]}+00:00" if normalized.endswith("Z") else normalized
    try:
        parsed = datetime.fromisoformat(parse_value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        return None
    return parsed.astimezone(timezone.utc)


def _github_star_prompt_non_negative_count(value: object) -> int:
    if type(value) is int and value >= 0:
        return value
    return 0


def _github_star_prompt_latest_timestamp(*values: str | None) -> str | None:
    latest: tuple[datetime, str] | None = None
    for value in values:
        parsed = _parse_github_star_prompt_timestamp(value)
        if parsed is None:
            continue
        normalized_value = _github_star_prompt_utc_timestamp(parsed)
        if latest is None or parsed > latest[0]:
            latest = (parsed, normalized_value)
    return latest[1] if latest is not None else None


class ClipboardWatcherRuntime(Protocol):
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


@dataclass(slots=True)
class _HubVadSink:
    hub: ClientHub
    channel: str = "self"

    async def handle_vad_event(self, event) -> None:  # noqa: ANN001
        if self.channel == "peer":
            await self.hub.handle_peer_vad_event(event)
            return
        await self.hub.handle_vad_event(event)


@dataclass(slots=True)
class GuiController:
    page: ft.Page
    app: object
    config_path: Path

    settings: AppSettings | None = None
    clock: SystemClock = SystemClock()
    _managed_openrouter_release_service: ManagedOpenRouterReleaseService | None = None
    _openrouter_pkce_client: OpenRouterPKCEClient | None = None

    sender: VrchatOscUdpSender | None = None
    osc: ChatboxPaginator | None = None
    hub: ClientHub | None = None
    _peer_runtime: PeerChannelRuntime | None = None
    receiver: VrcOscReceiver | None = None
    vrc_mic_state: VrcMicState | None = None
    vrc_mic_audio_gate: VrcMicAudioGate | None = None

    _bridge_task: asyncio.Task[None] | None = None
    _mic_task: asyncio.Task[None] | None = None
    _audio_source: AudioSource | None = None
    _last_mic_loop_close_exception: BaseException | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _microphone_test_meter_level: float = field(init=False, default=0.0)
    _microphone_test_task: asyncio.Task[None] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _microphone_test_lifecycle_lock: asyncio.Lock | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _debug_capture_fault_profile: str = field(init=False, default="none")
    _debug_stt_fault_profile: str = field(init=False, default="none")
    _vad: VadGating | None = None
    _stt_desired: bool = False
    _stt_switch_lock: asyncio.Lock | None = None
    _stt_switch_task: asyncio.Task[None] | None = None
    _stt_restart_requested: bool = False
    _last_stt_runtime_signature: tuple[object, ...] | None = None
    _last_self_stt_runtime_signature: tuple[object, ...] | None = None
    _last_peer_stt_runtime_signature: tuple[object, ...] | None = None
    _last_self_stt_provider_signature: tuple[object, ...] | None = None
    _last_peer_stt_provider_signature: tuple[object, ...] | None = None
    _last_llm_provider_signature: tuple[object, ...] | None = None
    _last_microphone_test_audio_settings_signature: tuple[object, ...] | None = None
    _last_peer_translation_enabled: bool | None = None
    _last_peer_translation_activation_requested: bool | None = None
    _last_vrc_mic_sync_enabled: bool | None = None
    _vrc_receiver_lock: asyncio.Lock | None = None
    _ui_event_bridge: UIEventBridge | None = None
    _clipboard_watcher: ClipboardWatcherRuntime | None = field(init=False, default=None)
    _clipboard_loop: asyncio.AbstractEventLoop | None = field(init=False, default=None)
    _clipboard_watcher_lock: asyncio.Lock | None = field(init=False, default=None)
    _local_stt_install_state: LocalSTTInstallState = field(
        init=False,
        default_factory=lambda: LocalSTTInstallState(status="ready"),
    )
    _local_stt_runtime_status: str = field(init=False, default="ready")
    _local_stt_download_origin: str | None = field(init=False, default=None)
    _local_stt_download_percent: int | None = field(init=False, default=None)
    _local_stt_download_task: asyncio.Task[object] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _local_stt_download_cancel_event: threading.Event | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _local_stt_pending_enable_after_install: bool = field(init=False, default=False)
    _local_stt_pending_peer_enable_after_install: bool = field(init=False, default=False)
    _overlay_bridge: OverlayBridge | None = None
    _overlay_presenter: OverlayPresenter | None = None
    _overlay_manager: OverlayProcessManager | None = None
    _overlay_diagnostics: OverlayDiagnosticsRecorder | None = None
    _overlay_start_task: asyncio.Task[None] | None = None
    _overlay_monitor_task: asyncio.Task[None] | None = None
    _overlay_lock: asyncio.Lock | None = None
    _active_overlay_target: str | None = field(init=False, default=None)
    _desktop_renderer_events: asyncio.Queue[dict[str, object]] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _desktop_renderer_events_task: asyncio.Task[None] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _desktop_bounds_persist_task: asyncio.Task[None] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _pending_desktop_bounds: dict[str, int | float] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _desktop_suppressed_bounds_signatures: set[tuple[float, float, float, float]] = field(
        init=False,
        default_factory=set,
        repr=False,
    )
    _managed_trial_pending_auth: bool = field(init=False, default=False)
    _discord_managed_auth_in_progress: bool = field(init=False, default=False)
    _qq_managed_auth_in_progress: bool = field(init=False, default=False)
    _qq_managed_auth_generation: int = field(init=False, default=0)
    _qq_managed_auth_cancelled: bool = field(init=False, default=False)
    _qq_managed_auth_task_handle: asyncio.Task[object] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _discord_managed_auth_callback_received_hook: Callable[[], None] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    last_discord_managed_auth_referral_bonus_applied: bool = field(
        init=False,
        default=False,
    )
    _managed_trial_usage_metadata: OpenRouterKeyMetadata | None = field(init=False, default=None)
    _managed_trial_usage_metadata_entitlement_ref: str | None = field(
        init=False,
        default=None,
    )
    _talk_together_pass_status: TalkTogetherPassStatus | None = field(
        init=False,
        default=None,
    )
    _talk_together_pass_status_key: tuple[str | None, str | None, str | None] | None = field(
        init=False,
        default=None,
    )
    _translation_toggle_intent_enabled: bool = field(init=False, default=False)
    _translation_toggle_generation: int = field(init=False, default=0)
    _github_star_prompt_translation_success_task: asyncio.Task[bool] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _github_star_prompt_persistence_lock: asyncio.Lock | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _runtime_logging: SessionRuntimeLoggingService | None = field(init=False, default=None)
    _local_qwen_hallucination_detection_count: int = field(init=False, default=0)
    _local_qwen_hallucination_modal_shown: bool = field(init=False, default=False)

    overlay_state: str = "off"
    failure_reason: str | None = None
    auto_restart_scheduled: bool = False
    desktop_overlay_interaction_mode: str = field(
        init=False,
        default=DESKTOP_INTERACTION_MODE_EDIT,
    )
    overlay_calibration: OverlayCalibration = field(default_factory=OverlayCalibration)
    _overlay_calibration_draft: OverlayCalibration | None = None

    @property
    def effective_peer_translation_enabled(self) -> bool:
        if self.settings is None:
            return False
        return self._effective_peer_translation_enabled_for(self.settings)

    @property
    def managed_auth_pending(self) -> bool:
        return self._managed_trial_pending_auth

    @property
    def desktop_overlay_captions_locked(self) -> bool:
        return self.desktop_overlay_interaction_mode == DESKTOP_INTERACTION_MODE_PASS_THROUGH

    @property
    def discord_managed_auth_in_progress(self) -> bool:
        return self._discord_managed_auth_in_progress

    @property
    def effective_context_mode(self) -> str:
        if self.settings is None:
            return "local"
        if self._effective_integrated_context_enabled_for(self.settings):
            return "integrated"
        return "local"

    def _effective_peer_translation_enabled_for(self, settings: AppSettings) -> bool:
        return bool(
            self._peer_translation_activation_requested_for(settings)
            and self._effective_peer_overlay_enabled_for(settings)
            and self.hub is not None
            and getattr(self.hub, "peer_stt", None) is not None
        )

    def _peer_translation_eula_accepted_for(self, settings: AppSettings) -> bool:
        return bool(settings.ui.peer_translation_eula_accepted)

    def _peer_translation_activation_requested_for(self, settings: AppSettings) -> bool:
        return bool(
            settings.ui.peer_translation_enabled
            and self._peer_translation_eula_accepted_for(settings)
        )

    def _effective_peer_overlay_enabled_for(self, settings: AppSettings) -> bool:
        _ = settings
        return self.overlay_state == "connected"

    def _effective_integrated_context_enabled_for(self, settings: AppSettings) -> bool:
        return bool(
            settings.ui.integrated_context_enabled
            and self._effective_peer_translation_enabled_for(settings)
        )

    def _sync_effective_hub_flags(self, settings: AppSettings | None = None) -> None:
        resolved_settings = settings or self.settings
        if resolved_settings is None or self.hub is None:
            return
        self.hub.peer_translation_enabled = self._effective_peer_translation_enabled_for(
            resolved_settings
        )
        self.hub.integrated_context_enabled = self._effective_integrated_context_enabled_for(
            resolved_settings
        )

    def build_overlay_peer_consumer_contract(self) -> OverlayPeerConsumerContract | None:
        if self.settings is None:
            return None
        return build_overlay_peer_consumer_contract(
            overlay_intent_enabled=bool(self.settings.ui.overlay_enabled),
            overlay_state=self.overlay_state,
            overlay_failure_reason=self.failure_reason,
            peer_intent_enabled=bool(self.settings.ui.peer_translation_enabled),
            peer_effective_enabled=self._effective_peer_translation_enabled_for(self.settings),
        )

    def _refresh_overlay_peer_consumers(self) -> None:
        refresh_contract = getattr(self.app, "refresh_overlay_peer_contract", None)
        if callable(refresh_contract):
            with contextlib.suppress(Exception):
                refresh_contract()

    async def _refresh_overlay_runtime_dependencies(self) -> None:
        if self.settings is None or self.hub is None:
            return

        await self._refresh_peer_stt_runtime()
        self._sync_effective_hub_flags(self.settings)
        self._refresh_overlay_peer_consumers()

    async def start(self) -> None:
        self.settings = self._load_or_init_settings(self.config_path)
        self.settings.ui.overlay_enabled = False
        self.settings.ui.peer_translation_enabled = False
        self._sync_overlay_calibration_cache(self.settings)
        self._overlay_calibration_draft = None
        set_locale(self.settings.ui.locale)
        self._sync_ui_from_settings()
        with contextlib.suppress(Exception):
            apply_locale = getattr(self.app, "apply_locale", None)
            if callable(apply_locale):
                apply_locale()
        with contextlib.suppress(Exception):
            show_telemetry_consent_dialog = getattr(self.app, "show_telemetry_consent_dialog", None)
            if callable(show_telemetry_consent_dialog):
                show_telemetry_consent_dialog()

        runtime_logging = self.runtime_logging
        runtime_logging.set_mode(SessionLoggingMode.BASIC)

        # Attach realtime sink to LogsView for GUI log display
        logs_view = getattr(self.app, "view_logs", None)
        if logs_view is not None:
            runtime_logging.attach_realtime_sink(logs_view)

        await self._init_pipeline()
        self._refresh_local_stt_runtime_state()

        assert self.hub is not None

        dash = getattr(self.app, "view_dashboard", None)
        if dash is not None:
            # Set needs_key flags based on saved verification status & key existence
            # STT: check current provider's verification status
            stt_provider = self.settings.provider.stt.value
            if self._stt_provider_requires_secret(self.settings.provider.stt):
                # Map stt provider to api_key_verified field name (qwen_asr uses alibaba keys)
                stt_key_map = {"qwen_asr": self._get_alibaba_verified_key()}
                stt_verified_key = stt_key_map.get(stt_provider, stt_provider)
                stt_verified = getattr(self.settings.api_key_verified, stt_verified_key, False)
                dash.stt_needs_key = (self.hub.stt is None) or (not stt_verified)
            else:
                dash.stt_needs_key = False

            # LLM: check current provider's verification status
            llm_provider = self.settings.provider.llm.value
            if self._llm_provider_requires_secret(self.settings.provider.llm):
                # Map llm provider to api_key_verified field name
                llm_key_map = {
                    "gemini": "google",
                    "openrouter": "openrouter",
                    "deepseek": "deepseek",
                    "qwen": self._get_alibaba_verified_key(),
                }
                llm_verified_key = llm_key_map.get(llm_provider, llm_provider)
                llm_verified = getattr(self.settings.api_key_verified, llm_verified_key, False)
                dash.translation_needs_key = (
                    False
                    if self._managed_openrouter_can_attempt_translation()
                    else (self.hub.llm is None) or (not llm_verified)
                )
            else:
                dash.translation_needs_key = False

            # Set initial enabled states (all start as off/gray)
            dash.set_translation_enabled(False)
            dash.set_stt_enabled(False)
            self.hub.translation_enabled = False
            await self._refresh_managed_trial_usage_state_impl(auto_show_founder_letter=False)

        await self.hub.start(auto_flush_osc=True)

        bridge = UIEventBridge(
            app=self.app,
            event_queue=self.hub.ui_events,
            runtime_logging=runtime_logging,
        )
        self._ui_event_bridge = bridge
        self._bridge_task = asyncio.create_task(bridge.run())
        await self._sync_clipboard_watcher()

    def _get_alibaba_verified_key(self) -> str:
        """Get the api_key_verified field name based on Qwen region."""
        from puripuly_heart.config.settings import QwenRegion

        if self.settings.qwen.region == QwenRegion.BEIJING:
            return "alibaba_beijing"
        return "alibaba_singapore"

    def _stt_provider_applies_custom_vocabulary(self, settings: AppSettings) -> bool:
        return settings.provider.stt in (
            STTProviderName.DEEPGRAM,
            STTProviderName.LOCAL_QWEN,
            STTProviderName.SONIOX,
        )

    def _stt_provider_requires_secret(self, provider: STTProviderName) -> bool:
        return provider in (
            STTProviderName.DEEPGRAM,
            STTProviderName.QWEN_ASR,
            STTProviderName.SONIOX,
        )

    def _llm_provider_requires_secret(self, provider: LLMProviderName) -> bool:
        return provider in (
            LLMProviderName.GEMINI,
            LLMProviderName.OPENROUTER,
            LLMProviderName.QWEN,
            LLMProviderName.DEEPSEEK,
            LLMProviderName.CEREBRAS,
        )

    def _selected_stt_provider(self) -> STTProviderName | None:
        if self.settings is None:
            return None
        return self.settings.provider.stt

    def _dashboard_stt_needs_key(self, *, stt_available: bool) -> bool:
        provider = self._selected_stt_provider()
        if provider is None:
            return not stt_available
        return self._stt_provider_requires_secret(provider) and not stt_available

    def _stt_runtime_custom_vocabulary_signature(
        self, settings: AppSettings
    ) -> tuple[bool, tuple[str, ...]]:
        if not self._stt_provider_applies_custom_vocabulary(settings):
            return False, ()
        if settings.provider.stt == STTProviderName.LOCAL_QWEN:
            from puripuly_heart.core.stt.custom_vocab import get_effective_local_qwen_hotwords

            return (
                settings.stt.custom_vocabulary_enabled,
                tuple(
                    get_effective_local_qwen_hotwords(settings, settings.languages.source_language)
                ),
            )
        return (
            settings.stt.custom_vocabulary_enabled,
            tuple(get_effective_custom_terms(settings, settings.languages.source_language)),
        )

    def _peer_stt_runtime_custom_vocabulary_signature(
        self, settings: AppSettings
    ) -> tuple[bool, tuple[str, ...]]:
        _ = settings
        return (False, ())

    def _build_self_stt_runtime_signature(self, settings: AppSettings) -> tuple[object, ...]:
        custom_vocab_enabled, custom_terms = self._stt_runtime_custom_vocabulary_signature(settings)
        return (
            settings.languages.source_language,
            settings.audio.input_host_api,
            settings.audio.input_device,
            settings.provider.stt,
            settings.stt.vad_speech_threshold,
            settings.stt.low_latency_mode,
            settings.stt.low_latency_merge_gap_ms,
            settings.stt.low_latency_spec_retry_max,
            settings.stt.low_latency_vad_hangover_ms,
            settings.stt.drain_timeout_s,
            settings.audio.ring_buffer_ms,
            settings.audio.internal_sample_rate_hz,
            settings.audio.internal_channels,
            custom_vocab_enabled,
            custom_terms,
        )

    def _build_self_stt_provider_signature(self, settings: AppSettings) -> tuple[object, ...]:
        local_qwen_identity = None
        if settings.provider.stt == STTProviderName.LOCAL_QWEN:
            from puripuly_heart.core.local_stt_assets import default_local_stt_model_dir

            local_qwen_identity = str(default_local_stt_model_dir())

        return (
            settings.provider.stt,
            (
                settings.deepgram_stt.model
                if settings.provider.stt == STTProviderName.DEEPGRAM
                else None
            ),
            settings.qwen.region if settings.provider.stt == STTProviderName.QWEN_ASR else None,
            (
                settings.qwen_asr_stt.model
                if settings.provider.stt == STTProviderName.QWEN_ASR
                else None
            ),
            settings.soniox_stt.model if settings.provider.stt == STTProviderName.SONIOX else None,
            (
                settings.soniox_stt.endpoint
                if settings.provider.stt == STTProviderName.SONIOX
                else None
            ),
            (
                settings.soniox_stt.keepalive_interval_s
                if settings.provider.stt == STTProviderName.SONIOX
                else None
            ),
            (
                settings.soniox_stt.trailing_silence_ms
                if settings.provider.stt == STTProviderName.SONIOX
                else None
            ),
            local_qwen_identity,
        )

    def _build_stt_runtime_signature(self, settings: AppSettings) -> tuple[object, ...]:
        return self._build_self_stt_runtime_signature(settings)

    def _build_peer_stt_runtime_signature(self, settings: AppSettings) -> tuple[object, ...]:
        return self._build_peer_runtime_config(settings).runtime_signature

    def _build_peer_stt_provider_signature(self, settings: AppSettings) -> tuple[object, ...]:
        return build_peer_stt_provider_signature(settings)

    def _managed_openrouter_can_attempt_translation(self) -> bool:
        can_attempt = bool(
            self.settings is not None
            and self.settings.provider.llm == LLMProviderName.OPENROUTER
            and self.settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
            and self.hub is not None
            and self.hub.llm is not None
        )
        if not can_attempt:
            return False
        if self._is_managed_china_connection():
            return self._managed_openrouter_local_key_available()
        return True

    def _sync_managed_auth_dashboard_notice(self) -> None:
        dash = getattr(self.app, "view_dashboard", None)
        setter = getattr(dash, "set_managed_auth_pending", None) if dash is not None else None
        if callable(setter):
            setter(self._managed_trial_pending_auth)

    def _set_managed_trial_pending_auth(self, pending: bool) -> None:
        self._managed_trial_pending_auth = bool(pending)
        self._sync_managed_auth_dashboard_notice()

    def clear_managed_auth_pending_state(self) -> None:
        self._set_managed_trial_pending_auth(False)

    def _record_translation_toggle_intent(self, enabled: bool) -> int:
        self._translation_toggle_intent_enabled = bool(enabled)
        self._translation_toggle_generation += 1
        return self._translation_toggle_generation

    def _translation_toggle_intent_matches(self, *, enabled: bool, generation: int) -> bool:
        return generation == self._translation_toggle_generation and (
            self._translation_toggle_intent_enabled == bool(enabled)
        )

    def _should_show_managed_auth_pending_before_prepare(self) -> bool:
        if self.settings is None:
            return False
        try:
            secrets = create_secret_store(self.settings.secrets, config_path=self.config_path)
            resolution = resolve_openrouter_credentials(
                self.settings,
                secrets=secrets,
                request_intent="TRANS",
            )
        except Exception:
            return True
        return resolution.api_key is None

    def _managed_openrouter_selected(self) -> bool:
        return bool(
            self.settings is not None
            and self.settings.provider.llm == LLMProviderName.OPENROUTER
            and self.settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
        )

    def _managed_openrouter_local_key_available(self) -> bool:
        if self.settings is None:
            return False
        try:
            secrets = create_secret_store(self.settings.secrets, config_path=self.config_path)
            resolution = resolve_openrouter_credentials(
                self.settings,
                secrets=secrets,
                request_intent="TRANS",
            )
        except Exception:
            return False
        has_key = resolution.api_key is not None
        if self._is_managed_china_connection():
            logger.info(
                "[QQAuth] _managed_openrouter_local_key_available: China mode, qq_key=%s",
                has_key,
            )
        else:
            logger.info(
                "[ManagedAuth] _managed_openrouter_local_key_available: non-China mode, has_key=%s",
                has_key,
            )
        return has_key

    def _managed_qq_key_available(self) -> bool:
        if self.settings is None:
            return False
        try:
            secrets = create_secret_store(self.settings.secrets, config_path=self.config_path)
            resolution = resolve_openrouter_credentials(
                self.settings,
                secrets=secrets,
                request_intent="TRANS",
            )
            result = resolution.api_key is not None
            logger.info("[QQAuth] _managed_qq_key_available: %s", result)
            return result
        except Exception as exc:
            logger.warning("[QQAuth] _managed_qq_key_available: exception: %s", exc)
            return False

    def dashboard_managed_auth_action(self) -> str:
        if not self._managed_openrouter_selected():
            return "continue"
        if (
            self._discord_managed_auth_in_progress
            or self._qq_managed_auth_in_progress
            or self._managed_trial_pending_auth
        ):
            logger.info(
                "[ManagedAuth] dashboard_managed_auth_action: in_progress (discord=%s qq=%s pending=%s)",
                self._discord_managed_auth_in_progress,
                self._qq_managed_auth_in_progress,
                self._managed_trial_pending_auth,
            )
            return "in_progress"
        if self._managed_openrouter_local_key_available():
            logger.info("[ManagedAuth] dashboard_managed_auth_action: continue (key available)")
            return "continue"
        is_china = self._is_managed_china_connection()
        logger.info("[ManagedAuth] dashboard_managed_auth_action: prompt (is_china=%s)", is_china)
        return "prompt"

    def _discord_auth_message_key(self, result) -> str:  # noqa: ANN001
        diagnostics = getattr(result, "diagnostics", None)
        subcode = getattr(diagnostics, "subcode", None)
        if subcode is not None:
            mapped_key = DISCORD_AUTH_ERROR_KEY_BY_SUBCODE.get(subcode)
            if mapped_key is not None:
                return mapped_key
        if getattr(diagnostics, "code", None) == "discord_loopback_unavailable":
            return DISCORD_AUTH_ERROR_KEY_BY_SUBCODE["loopback_unavailable"]
        return getattr(result, "message_key", "discord_auth.error.retry")

    async def start_discord_managed_auth_from_dialog(
        self,
        *,
        on_callback_received: Callable[[], None] | None = None,
        referral_id: str | None = None,
    ) -> bool:
        self.last_discord_managed_auth_referral_bonus_applied = False
        service = self._managed_openrouter_release_service
        if service is None:
            self._discord_managed_auth_in_progress = False
            self._set_managed_trial_pending_auth(False)
            self._show_short_message("discord_auth.error.retry")
            return False

        previous_callback = self._discord_managed_auth_callback_received_hook
        self._discord_managed_auth_callback_received_hook = on_callback_received
        self._discord_managed_auth_in_progress = True
        self._set_managed_trial_pending_auth(True)
        try:
            try:
                result = await service.prepare_for_translation(referral_id=referral_id)
            except Exception as exc:
                self.log_basic(
                    f"[ManagedAuth] Discord auth start failed: {exc}",
                    level=logging.ERROR,
                )
                self._show_short_message("discord_auth.error.retry")
                return False

            if (
                result.behavior == ManagedOpenRouterReleaseBehavior.READY
                and result.local_key_available
            ):
                self.last_discord_managed_auth_referral_bonus_applied = (
                    getattr(result, "referral_bonus_applied", False) is True
                )
                if self.hub is None:
                    self._show_short_message("discord_auth.error.retry")
                    return False
                if self.hub.llm is None:
                    await self._rebuild_llm_provider()
                if self.hub.llm is None:
                    self._show_short_message("discord_auth.error.retry")
                    return False
                result_referral_id = normalize_owned_referral_id(
                    getattr(result, "referral_id", None)
                )
                self._set_managed_usage_view_state(
                    view_settings=getattr(self.app, "view_settings", None),
                    visible=True,
                    remaining_percent=None,
                    referral_id=result_referral_id or self._current_owned_referral_id(),
                    pass_status=getattr(result, "pass_status", None),
                )
                self._schedule_managed_trial_usage_refresh()
                return True

            message_key = self._discord_auth_message_key(result)
            diagnostics = result.diagnostics
            error_class = getattr(diagnostics, "error_class", None)
            self.log_basic(
                "[ManagedAuth] Discord auth failed: "
                f"message_key={message_key} class={error_class or 'unknown'}",
                level=logging.ERROR,
            )
            self._show_short_message(
                message_key,
                **dict(result.message_kwargs),
            )
            return False
        finally:
            if self._discord_managed_auth_callback_received_hook is on_callback_received:
                self._discord_managed_auth_callback_received_hook = previous_callback
            self._discord_managed_auth_in_progress = False
            self._set_managed_trial_pending_auth(False)

    def _is_managed_china_connection(self) -> bool:
        if self.settings is None:
            return False
        return self.settings.translation.connection == TranslationConnection.MANAGED_CHINA

    async def start_qq_managed_auth_from_dialog(
        self,
        *,
        qq_identity: str,
        credential: str,
    ) -> bool | ManagedOpenRouterReleaseResult:
        logger.info("[QQAuth] start_qq_managed_auth_from_dialog: starting")
        current_task = asyncio.current_task()
        previous_task = self._qq_managed_auth_task_handle
        if (
            previous_task is not None
            and previous_task is not current_task
            and not previous_task.done()
        ):
            previous_task.cancel()
        if current_task is not None:
            self._qq_managed_auth_task_handle = current_task
        self._qq_managed_auth_generation += 1
        auth_generation = self._qq_managed_auth_generation
        self._qq_managed_auth_cancelled = False

        service = self._managed_openrouter_release_service
        if service is None:
            logger.warning("[QQAuth] start_qq_managed_auth_from_dialog: service is None")
            if current_task is not None and self._qq_managed_auth_task_handle is current_task:
                self._qq_managed_auth_task_handle = None
            self._qq_managed_auth_in_progress = False
            self._set_managed_trial_pending_auth(False)
            return self._neutral_qq_managed_auth_result()

        normalized_qq_identity = (qq_identity or "").strip()
        normalized_credential = (credential or "").strip()
        if not normalized_qq_identity or not normalized_credential:
            logger.warning("[QQAuth] start_qq_managed_auth_from_dialog: empty input")
            if current_task is not None and self._qq_managed_auth_task_handle is current_task:
                self._qq_managed_auth_task_handle = None
            self._qq_managed_auth_in_progress = False
            self._set_managed_trial_pending_auth(False)
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.RETRY,
                message_key="qq_auth.error.invalid_input",
            )

        self._qq_managed_auth_in_progress = True
        self._set_managed_trial_pending_auth(True)

        def issue_persistence_allowed() -> bool:
            return self._is_current_qq_managed_auth_generation(auth_generation)

        try:
            try:
                logger.info(
                    "[QQAuth] start_qq_managed_auth_from_dialog: calling prepare_from_qq_assertion"
                )
                result = await service.prepare_from_qq_assertion(
                    qq_identity=normalized_qq_identity,
                    credential=normalized_credential,
                    issue_persistence_allowed=issue_persistence_allowed,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._is_current_qq_managed_auth_generation(auth_generation):
                    return self._neutral_qq_managed_auth_result()
                self.log_basic(
                    "[QQAuth] QQ auth start failed",
                    level=logging.ERROR,
                )
                logger.warning(
                    "[QQAuth] start_qq_managed_auth_from_dialog: release error class=%s",
                    exc.__class__.__name__,
                )
                return ManagedOpenRouterReleaseResult(
                    behavior=ManagedOpenRouterReleaseBehavior.RETRY,
                    message_key="qq_auth.error.retry",
                )

            if not self._is_current_qq_managed_auth_generation(auth_generation):
                return self._neutral_qq_managed_auth_result()
            logger.info(
                "[QQAuth] start_qq_managed_auth_from_dialog: result behavior=%s local_key=%s",
                result.behavior,
                result.local_key_available,
            )

            if (
                result.behavior == ManagedOpenRouterReleaseBehavior.READY
                and result.local_key_available
            ):
                if self.hub is None:
                    return ManagedOpenRouterReleaseResult(
                        behavior=ManagedOpenRouterReleaseBehavior.RETRY,
                        message_key="qq_auth.error.retry",
                    )
                if self.hub.llm is None:
                    await self._rebuild_llm_provider()
                    if not self._is_current_qq_managed_auth_generation(auth_generation):
                        return self._neutral_qq_managed_auth_result()
                if self.hub.llm is None:
                    return ManagedOpenRouterReleaseResult(
                        behavior=ManagedOpenRouterReleaseBehavior.RETRY,
                        message_key="qq_auth.error.retry",
                    )
                result_referral_id = normalize_owned_referral_id(
                    getattr(result, "referral_id", None)
                )
                self._set_managed_usage_view_state(
                    view_settings=getattr(self.app, "view_settings", None),
                    visible=True,
                    remaining_percent=None,
                    referral_id=result_referral_id or self._current_owned_referral_id(),
                    pass_status=getattr(result, "pass_status", None),
                )
                self._schedule_managed_trial_usage_refresh()
                return True

            result = self._qq_managed_auth_result_for_dialog(result)
            diagnostics = result.diagnostics
            subcode = getattr(diagnostics, "subcode", None)
            self.log_basic(
                "[ManagedAuth] QQ auth failed: "
                f"message_key={result.message_key} "
                f"subcode={subcode or 'none'} "
                f"class={getattr(diagnostics, 'error_class', None) or 'unknown'}",
                level=logging.ERROR,
            )
            return result
        finally:
            if self._is_current_qq_managed_auth_generation(auth_generation) and (
                current_task is None or self._qq_managed_auth_task_handle is current_task
            ):
                if current_task is not None:
                    self._qq_managed_auth_task_handle = None
                self._qq_managed_auth_in_progress = False
                self._set_managed_trial_pending_auth(False)

    def _neutral_qq_managed_auth_result(self) -> ManagedOpenRouterReleaseResult:
        return ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.RETRY,
            message_key="qq_auth.error.retry",
        )

    def _is_current_qq_managed_auth_generation(self, generation: int) -> bool:
        return bool(
            generation == self._qq_managed_auth_generation and not self._qq_managed_auth_cancelled
        )

    def _qq_managed_auth_result_for_dialog(
        self,
        result: ManagedOpenRouterReleaseResult,
    ) -> ManagedOpenRouterReleaseResult:
        diagnostics = result.diagnostics
        subcode = getattr(diagnostics, "subcode", None)
        if subcode in {"qq_credential_invalid", "qq_credential_mismatch"}:
            if result.message_key == "qq_auth.error.credential_mismatch":
                return result
            return replace(result, message_key="qq_auth.error.credential_mismatch")
        if subcode == "qq_lifetime_used":
            if result.message_key == "qq_auth.error.lifetime_used":
                return result
            return replace(result, message_key="qq_auth.error.lifetime_used")
        return result

    def cancel_qq_managed_auth(self) -> None:
        self._qq_managed_auth_cancelled = True
        task_handle = self._qq_managed_auth_task_handle
        cancel = getattr(task_handle, "cancel", None)
        if callable(cancel):
            with contextlib.suppress(Exception):
                cancel()
        self._qq_managed_auth_task_handle = None
        self._qq_managed_auth_in_progress = False
        self._set_managed_trial_pending_auth(False)

    def _managed_trial_remaining_percent(
        self, usage_metadata: OpenRouterKeyMetadata | None
    ) -> int | None:
        if usage_metadata is None:
            return None
        if usage_metadata.limit_usd is None or usage_metadata.remaining_usd is None:
            return None
        if usage_metadata.limit_usd <= 0:
            return None
        return max(
            0, min(100, round((usage_metadata.remaining_usd / usage_metadata.limit_usd) * 100))
        )

    def _github_star_prompt_translation_connection_for(
        self,
        settings: AppSettings | None,
    ) -> TranslationConnection | None:
        if settings is None:
            return None
        connection = settings.translation.connection
        if isinstance(connection, TranslationConnection):
            return connection
        with contextlib.suppress(ValueError, TypeError):
            return TranslationConnection(connection)
        return None

    def _github_star_prompt_current_translation_connection(self) -> TranslationConnection | None:
        return self._github_star_prompt_translation_connection_for(self.settings)

    def _github_star_prompt_settings_has_user_owned_cloud_connection(
        self,
        settings: AppSettings | None,
    ) -> bool:
        return (
            self._github_star_prompt_translation_connection_for(settings)
            in _GITHUB_STAR_PROMPT_USER_OWNED_CLOUD_CONNECTIONS
        )

    def _github_star_prompt_has_managed_connection(self) -> bool:
        return (
            self._github_star_prompt_current_translation_connection()
            in _GITHUB_STAR_PROMPT_MANAGED_CONNECTIONS
        )

    def _github_star_prompt_has_user_owned_cloud_connection(self) -> bool:
        return (
            self._github_star_prompt_current_translation_connection()
            in _GITHUB_STAR_PROMPT_USER_OWNED_CLOUD_CONNECTIONS
        )

    def is_github_star_prompt_eligible(self) -> bool:
        if self.settings is None:
            return False
        if self._github_star_prompt_has_managed_connection():
            remaining_percent = self._managed_trial_remaining_percent(
                self._managed_trial_usage_metadata
            )
            return (
                remaining_percent is not None
                and remaining_percent <= GITHUB_STAR_PROMPT_MANAGED_REMAINING_PERCENT_THRESHOLD
            )
        if self._github_star_prompt_has_user_owned_cloud_connection():
            return bool(self.settings.ui.github_star_prompt_translation_success_observed)
        return False

    def _github_star_prompt_initial_launch_gate_satisfied(self, settings: AppSettings) -> bool:
        if _github_star_prompt_non_negative_count(settings.ui.github_star_prompt_show_count) > 0:
            return True
        return (
            _github_star_prompt_non_negative_count(
                settings.ui.github_star_prompt_eligible_launch_count
            )
            >= GITHUB_STAR_PROMPT_ELIGIBLE_LAUNCH_THRESHOLD
        )

    def should_show_github_star_prompt(self, *, now: datetime | None = None) -> bool:
        settings = self.settings
        if settings is None:
            return False
        if settings.ui.github_star_prompt_clicked:
            return False
        if not self.is_github_star_prompt_eligible():
            return False
        if not self._github_star_prompt_initial_launch_gate_satisfied(settings):
            return False

        last_shown_at = _parse_github_star_prompt_timestamp(
            settings.ui.github_star_prompt_last_shown_at
        )
        if last_shown_at is None:
            return True

        resolved_now = now or _github_star_prompt_utc_now()
        if resolved_now.tzinfo is None:
            resolved_now = resolved_now.replace(tzinfo=timezone.utc)
        elapsed = resolved_now.astimezone(timezone.utc) - last_shown_at
        return elapsed >= GITHUB_STAR_PROMPT_RECENCY_WINDOW

    def _get_github_star_prompt_persistence_lock(self) -> asyncio.Lock:
        if self._github_star_prompt_persistence_lock is None:
            self._github_star_prompt_persistence_lock = asyncio.Lock()
        return self._github_star_prompt_persistence_lock

    def _github_star_prompt_state_snapshot(self, settings: AppSettings) -> tuple[object, ...]:
        return (
            settings.ui.github_star_prompt_clicked,
            settings.ui.github_star_prompt_last_shown_at,
            settings.ui.github_star_prompt_show_count,
            settings.ui.github_star_prompt_translation_success_observed,
            settings.ui.github_star_prompt_eligible_launch_count,
        )

    def _restore_github_star_prompt_state_snapshot(
        self,
        settings: AppSettings,
        snapshot: tuple[object, ...],
    ) -> None:
        (
            clicked,
            last_shown_at,
            show_count,
            translation_success_observed,
            eligible_launch_count,
        ) = snapshot
        settings.ui.github_star_prompt_clicked = bool(clicked)
        settings.ui.github_star_prompt_last_shown_at = (
            last_shown_at if isinstance(last_shown_at, str) else None
        )
        settings.ui.github_star_prompt_show_count = _github_star_prompt_non_negative_count(
            show_count
        )
        settings.ui.github_star_prompt_translation_success_observed = bool(
            translation_success_observed
        )
        settings.ui.github_star_prompt_eligible_launch_count = (
            _github_star_prompt_non_negative_count(eligible_launch_count)
        )

    def _log_github_star_prompt_save_failure(
        self,
        failure_context: str,
        exc: Exception,
    ) -> None:
        self.log_basic(
            f"[GitHubStar] Failed to persist prompt {failure_context}: {exc}",
            level=logging.WARNING,
        )

    async def _persist_github_star_prompt_mutation(
        self,
        *,
        failure_context: str,
        mutate,
    ) -> bool:
        attempted_mutation = False
        while True:
            async with self._get_github_star_prompt_persistence_lock():
                settings = self.settings
                if settings is None:
                    return False
                snapshot = self._github_star_prompt_state_snapshot(settings)
                if not mutate(settings):
                    return attempted_mutation
                attempted_mutation = True
                try:
                    await asyncio.to_thread(save_settings, self.config_path, settings)
                except asyncio.CancelledError:
                    if self.settings is settings:
                        self._restore_github_star_prompt_state_snapshot(settings, snapshot)
                    raise
                except Exception as exc:
                    if self.settings is settings:
                        self._restore_github_star_prompt_state_snapshot(settings, snapshot)
                    self._log_github_star_prompt_save_failure(failure_context, exc)
                    return False
                if self.settings is settings:
                    return True
            await asyncio.sleep(0)

    async def persist_github_star_prompt_opened(
        self,
        *,
        opened_at: datetime | None = None,
        should_open: Callable[[], bool] | None = None,
    ) -> bool:
        opened_timestamp = _github_star_prompt_utc_timestamp(opened_at)

        while True:
            async with self._get_github_star_prompt_persistence_lock():
                settings = self.settings
                if settings is None:
                    return False
                if should_open is not None and not should_open():
                    return False
                snapshot = self._github_star_prompt_state_snapshot(settings)
                settings.ui.github_star_prompt_last_shown_at = opened_timestamp
                settings.ui.github_star_prompt_show_count = (
                    _github_star_prompt_non_negative_count(
                        settings.ui.github_star_prompt_show_count
                    )
                    + 1
                )
                try:
                    await asyncio.to_thread(save_settings, self.config_path, settings)
                except asyncio.CancelledError:
                    if self.settings is settings:
                        self._restore_github_star_prompt_state_snapshot(settings, snapshot)
                    raise
                except Exception as exc:
                    if self.settings is settings:
                        self._restore_github_star_prompt_state_snapshot(settings, snapshot)
                    self._log_github_star_prompt_save_failure("open state", exc)
                    return False
                if self.settings is settings:
                    if should_open is not None and not should_open():
                        self._restore_github_star_prompt_state_snapshot(settings, snapshot)
                        try:
                            await asyncio.to_thread(save_settings, self.config_path, settings)
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            self._log_github_star_prompt_save_failure(
                                "open state rollback",
                                exc,
                            )
                        return False
                    return True
            await asyncio.sleep(0)

    async def persist_github_star_prompt_eligible_launch(self) -> bool:
        settings = self.settings
        if settings is None:
            return False
        if settings.ui.github_star_prompt_clicked:
            return False
        if not self.is_github_star_prompt_eligible():
            return False
        if self._github_star_prompt_initial_launch_gate_satisfied(settings):
            return True

        def _mutate(settings: AppSettings) -> bool:
            if settings.ui.github_star_prompt_clicked:
                return False
            if not self.is_github_star_prompt_eligible():
                return False
            if self._github_star_prompt_initial_launch_gate_satisfied(settings):
                return False
            current_count = _github_star_prompt_non_negative_count(
                settings.ui.github_star_prompt_eligible_launch_count
            )
            settings.ui.github_star_prompt_eligible_launch_count = min(
                current_count + 1,
                GITHUB_STAR_PROMPT_ELIGIBLE_LAUNCH_THRESHOLD,
            )
            return True

        await self._persist_github_star_prompt_mutation(
            failure_context="eligible launch state",
            mutate=_mutate,
        )
        settings = self.settings
        return bool(
            settings is not None
            and not settings.ui.github_star_prompt_clicked
            and self.is_github_star_prompt_eligible()
            and self._github_star_prompt_initial_launch_gate_satisfied(settings)
        )

    def _run_github_star_prompt_persistence_sync(self, coro) -> bool:  # noqa: ANN001
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return bool(asyncio.run(coro))
        close = getattr(coro, "close", None)
        if callable(close):
            close()
        return False

    def record_github_star_prompt_opened(self, *, opened_at: datetime | None = None) -> bool:
        return self._run_github_star_prompt_persistence_sync(
            self.persist_github_star_prompt_opened(opened_at=opened_at)
        )

    async def persist_github_star_prompt_clicked(self) -> bool:
        def _mutate(settings: AppSettings) -> bool:
            settings.ui.github_star_prompt_clicked = True
            return True

        return await self._persist_github_star_prompt_mutation(
            failure_context="click state",
            mutate=_mutate,
        )

    def record_github_star_prompt_clicked(self) -> bool:
        return self._run_github_star_prompt_persistence_sync(
            self.persist_github_star_prompt_clicked()
        )

    async def persist_github_star_prompt_translation_success_observed(self) -> bool:
        def _mutate(settings: AppSettings) -> bool:
            if not self._github_star_prompt_settings_has_user_owned_cloud_connection(settings):
                return False
            if settings.ui.github_star_prompt_translation_success_observed:
                return False
            settings.ui.github_star_prompt_translation_success_observed = True
            return True

        return await self._persist_github_star_prompt_mutation(
            failure_context="translation success observation",
            mutate=_mutate,
        )

    def record_github_star_prompt_translation_success_observed(self) -> bool:
        return self._run_github_star_prompt_persistence_sync(
            self.persist_github_star_prompt_translation_success_observed()
        )

    def schedule_github_star_prompt_translation_success_observed(self) -> bool:
        if self.settings is None:
            return False
        if not self._github_star_prompt_has_user_owned_cloud_connection():
            return False
        if self.settings.ui.github_star_prompt_translation_success_observed:
            return False
        existing_task = self._github_star_prompt_translation_success_task
        if existing_task is not None and not existing_task.done():
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return self.record_github_star_prompt_translation_success_observed()

        task = loop.create_task(self.persist_github_star_prompt_translation_success_observed())
        self._github_star_prompt_translation_success_task = task

        def _clear_completed_task(completed_task: asyncio.Task[bool]) -> None:
            if self._github_star_prompt_translation_success_task is completed_task:
                self._github_star_prompt_translation_success_task = None

        task.add_done_callback(_clear_completed_task)
        return True

    async def _drain_github_star_prompt_translation_success_task(self) -> None:
        task = self._github_star_prompt_translation_success_task
        if task is None:
            return
        await asyncio.gather(task, return_exceptions=True)
        if self._github_star_prompt_translation_success_task is task:
            self._github_star_prompt_translation_success_task = None

    async def _preserve_github_star_prompt_observation_before_settings_replace(
        self,
        replacement_settings: AppSettings,
    ) -> None:
        await self._drain_github_star_prompt_translation_success_task()
        async with self._get_github_star_prompt_persistence_lock():
            if self.settings is None:
                return
            current_ui = self.settings.ui
            replacement_ui = replacement_settings.ui
            replacement_ui.github_star_prompt_clicked = bool(
                replacement_ui.github_star_prompt_clicked or current_ui.github_star_prompt_clicked
            )
            replacement_ui.github_star_prompt_translation_success_observed = bool(
                replacement_ui.github_star_prompt_translation_success_observed
                or current_ui.github_star_prompt_translation_success_observed
            )
            replacement_ui.github_star_prompt_eligible_launch_count = max(
                _github_star_prompt_non_negative_count(
                    replacement_ui.github_star_prompt_eligible_launch_count
                ),
                _github_star_prompt_non_negative_count(
                    current_ui.github_star_prompt_eligible_launch_count
                ),
            )
            replacement_ui.github_star_prompt_show_count = max(
                _github_star_prompt_non_negative_count(
                    replacement_ui.github_star_prompt_show_count
                ),
                _github_star_prompt_non_negative_count(current_ui.github_star_prompt_show_count),
            )
            replacement_ui.github_star_prompt_last_shown_at = _github_star_prompt_latest_timestamp(
                replacement_ui.github_star_prompt_last_shown_at,
                current_ui.github_star_prompt_last_shown_at,
            )

    def _current_owned_referral_id(self) -> str | None:
        if self.settings is None:
            return None
        return normalize_owned_referral_id(self.settings.managed_identity.referral_id)

    def _managed_identity_scope(
        self,
        referral_id: str | None,
    ) -> tuple[str | None, str | None, str | None] | None:
        if self.settings is None:
            return None
        installation_id = self.settings.managed_identity.installation_id.strip() or None
        active_ref = self.settings.managed_identity.active_managed_credential_ref
        normalized_active_ref = active_ref.strip() if isinstance(active_ref, str) else None
        normalized_referral_id = normalize_owned_referral_id(referral_id)
        return (installation_id, normalized_active_ref or None, normalized_referral_id)

    def _talk_together_pass_cache_key(
        self,
        referral_id: str | None,
    ) -> tuple[str | None, str | None, str | None] | None:
        normalized_referral_id = normalize_owned_referral_id(referral_id)
        if normalized_referral_id is None:
            return None
        return self._managed_identity_scope(normalized_referral_id)

    def _clear_talk_together_pass_status_cache(self) -> None:
        self._talk_together_pass_status = None
        self._talk_together_pass_status_key = None

    def _cached_talk_together_pass_status_for(
        self,
        referral_id: str | None,
    ) -> TalkTogetherPassStatus | None:
        cache_key = self._talk_together_pass_cache_key(referral_id)
        if cache_key is None or cache_key != self._talk_together_pass_status_key:
            self._clear_talk_together_pass_status_cache()
            return None
        return self._talk_together_pass_status

    def _set_managed_usage_view_state(
        self,
        *,
        view_settings: object | None,
        visible: bool,
        remaining_percent: int | None,
        referral_id: str | None,
        pass_status: TalkTogetherPassStatus | None | object = _PASS_STATUS_UNSET,
    ) -> None:
        normalized_referral_id = normalize_owned_referral_id(referral_id)
        if not visible or normalized_referral_id is None:
            self._clear_talk_together_pass_status_cache()
        elif pass_status is _PASS_STATUS_UNSET:
            pass
        elif (
            isinstance(pass_status, TalkTogetherPassStatus)
            and pass_status.pass_id == normalized_referral_id
        ):
            self._talk_together_pass_status = pass_status
            self._talk_together_pass_status_key = self._talk_together_pass_cache_key(
                normalized_referral_id
            )
        else:
            self._clear_talk_together_pass_status_cache()

        effective_pass_status = self._cached_talk_together_pass_status_for(normalized_referral_id)
        if view_settings is None:
            return
        managed_key_setter = getattr(view_settings, "set_managed_key_state", None)
        if callable(managed_key_setter):
            if _callable_accepts_keyword(managed_key_setter, "pass_status"):
                managed_key_setter(
                    visible=visible,
                    remaining_percent=remaining_percent,
                    referral_id=normalized_referral_id,
                    pass_status=effective_pass_status,
                )
            else:
                managed_key_setter(
                    visible=visible,
                    remaining_percent=remaining_percent,
                    referral_id=normalized_referral_id,
                )
            return
        usage_setter = getattr(view_settings, "set_managed_trial_usage_state", None)
        if callable(usage_setter):
            usage_setter(visible=visible, remaining_percent=remaining_percent)

    def _managed_key_card_visible_from_settings(self) -> bool:
        if self.settings is None:
            return False
        return self.settings.translation.connection in (
            TranslationConnection.MANAGED,
            TranslationConnection.MANAGED_CHINA,
        )

    async def _refresh_managed_status_best_effort(
        self,
        *,
        service: object | None = None,
    ) -> ManagedOpenRouterStatusRefreshResult:
        current_referral_id = self._current_owned_referral_id()
        if service is None:
            service = self._managed_openrouter_release_service
        if service is None:
            return ManagedOpenRouterStatusRefreshResult(
                referral_id=current_referral_id,
                pass_status=self._cached_talk_together_pass_status_for(current_referral_id),
                succeeded=False,
            )
        refresh_status = getattr(service, "refresh_managed_status", None)
        if callable(refresh_status):
            try:
                return await refresh_status()
            except Exception as exc:
                self.log_basic(
                    f"[ManagedAuth] Managed status refresh failed: {exc}",
                    level=logging.WARNING,
                )
                return ManagedOpenRouterStatusRefreshResult(
                    referral_id=current_referral_id,
                    pass_status=self._cached_talk_together_pass_status_for(current_referral_id),
                    succeeded=False,
                )
        refresh_status = getattr(service, "refresh_owned_referral_id_from_status", None)
        if callable(refresh_status):
            try:
                return ManagedOpenRouterStatusRefreshResult(
                    referral_id=normalize_owned_referral_id(await refresh_status())
                    or current_referral_id,
                    pass_status=None,
                    succeeded=True,
                )
            except Exception as exc:
                self.log_basic(
                    f"[ManagedAuth] Referral ID status refresh failed: {exc}",
                    level=logging.WARNING,
                )
        return ManagedOpenRouterStatusRefreshResult(
            referral_id=current_referral_id,
            pass_status=self._cached_talk_together_pass_status_for(current_referral_id),
            succeeded=False,
        )

    async def _refresh_owned_referral_id_from_managed_status_best_effort(
        self,
        *,
        service: object | None = None,
    ) -> str | None:
        return (await self._refresh_managed_status_best_effort(service=service)).referral_id

    def _schedule_owned_referral_id_status_refresh(
        self,
        *,
        view_settings: object | None,
        remaining_percent: int | None,
        current_referral_id: str | None,
    ) -> None:
        service = self._managed_openrouter_release_service
        if service is None:
            return
        refresh_status = getattr(service, "refresh_managed_status", None)
        legacy_refresh_status = getattr(service, "refresh_owned_referral_id_from_status", None)
        if not callable(refresh_status) and not callable(legacy_refresh_status):
            return
        scheduled_identity_scope = self._managed_identity_scope(current_referral_id)
        scheduled_identity_base = (
            scheduled_identity_scope[:2] if scheduled_identity_scope is not None else None
        )

        async def _run_status_refresh() -> None:
            try:
                result = await self._refresh_managed_status_best_effort(
                    service=service,
                )
                if service is not self._managed_openrouter_release_service:
                    return
                if (
                    self.settings is None
                    or self.settings.provider.llm != LLMProviderName.OPENROUTER
                    or self.settings.openrouter.selected_source
                    != OpenRouterCredentialSource.MANAGED
                    or not self._managed_key_card_visible_from_settings()
                ):
                    return
                refreshed_referral_id = (
                    normalize_owned_referral_id(result.referral_id) or current_referral_id
                )
                current_identity_scope = self._managed_identity_scope(
                    self._current_owned_referral_id()
                )
                allowed_identity_scopes = {scheduled_identity_scope}
                if scheduled_identity_base is not None:
                    allowed_identity_scopes.add((*scheduled_identity_base, refreshed_referral_id))
                if current_identity_scope not in allowed_identity_scopes:
                    return
                if result.succeeded:
                    self._set_managed_usage_view_state(
                        view_settings=view_settings,
                        visible=True,
                        remaining_percent=remaining_percent,
                        referral_id=refreshed_referral_id,
                        pass_status=result.pass_status,
                    )
                    return
                self._set_managed_usage_view_state(
                    view_settings=view_settings,
                    visible=True,
                    remaining_percent=remaining_percent,
                    referral_id=refreshed_referral_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.log_basic(
                    f"[ManagedAuth] Referral ID status refresh failed: {exc}",
                    level=logging.WARNING,
                )

        with contextlib.suppress(RuntimeError):
            asyncio.get_running_loop().create_task(_run_status_refresh())

    def _schedule_managed_trial_usage_refresh(self) -> None:
        async def _run_refresh() -> None:
            await self._refresh_managed_trial_usage_state_best_effort()

        with contextlib.suppress(RuntimeError):
            asyncio.get_running_loop().create_task(_run_refresh())

    def _on_managed_trial_delegate_ready(self) -> None:
        self._set_managed_trial_pending_auth(False)
        self._schedule_managed_trial_usage_refresh()

    async def _refresh_managed_trial_usage_state_best_effort(self) -> None:
        try:
            await self._refresh_managed_trial_usage_state()
        except Exception as exc:
            self.log_basic(
                f"[ManagedAuth] Usage refresh failed: {exc}",
                level=logging.WARNING,
            )

    async def _refresh_managed_trial_usage_state(self) -> None:
        await self._refresh_managed_trial_usage_state_impl(auto_show_founder_letter=True)

    def _clear_managed_trial_usage_metadata_cache(self) -> None:
        self._managed_trial_usage_metadata = None
        self._managed_trial_usage_metadata_entitlement_ref = None

    def _sync_managed_trial_usage_metadata_scope(self) -> str | None:
        if self.settings is None:
            self._clear_managed_trial_usage_metadata_cache()
            return None
        entitlement_ref = self.settings.managed_identity.active_managed_credential_ref
        if entitlement_ref != self._managed_trial_usage_metadata_entitlement_ref:
            self._managed_trial_usage_metadata = None
            self._managed_trial_usage_metadata_entitlement_ref = entitlement_ref
        return entitlement_ref

    async def _refresh_managed_trial_usage_state_impl(
        self,
        *,
        auto_show_founder_letter: bool,
    ) -> None:
        view_settings = getattr(self.app, "view_settings", None)
        if self.settings is None:
            self._clear_managed_trial_usage_metadata_cache()
            self._set_managed_trial_pending_auth(False)
            self._set_managed_usage_view_state(
                view_settings=view_settings,
                visible=False,
                remaining_percent=None,
                referral_id=self._current_owned_referral_id(),
            )
            return
        managed_key_visible = self._managed_key_card_visible_from_settings()
        if not managed_key_visible:
            self._clear_managed_trial_usage_metadata_cache()
            self._set_managed_trial_pending_auth(False)
            self._set_managed_usage_view_state(
                view_settings=view_settings,
                visible=False,
                remaining_percent=None,
                referral_id=self._current_owned_referral_id(),
            )
            return
        if (
            self.settings.provider.llm != LLMProviderName.OPENROUTER
            or self.settings.openrouter.selected_source != OpenRouterCredentialSource.MANAGED
        ):
            self._clear_managed_trial_usage_metadata_cache()
            self._set_managed_trial_pending_auth(False)
            self._set_managed_usage_view_state(
                view_settings=view_settings,
                visible=True,
                remaining_percent=None,
                referral_id=self._current_owned_referral_id(),
            )
            return

        entitlement_ref = self._sync_managed_trial_usage_metadata_scope()

        try:
            secrets = create_secret_store(self.settings.secrets, config_path=self.config_path)
            resolution = resolve_openrouter_credentials(self.settings, secrets=secrets)
        except Exception:
            resolution = None

        usage_metadata: OpenRouterKeyMetadata | None = None
        api_key = resolution.api_key if resolution is not None else None
        if api_key:
            self._set_managed_trial_pending_auth(False)
            usage_metadata = await OpenRouterLLMProvider.fetch_key_metadata(api_key)

        self._managed_trial_usage_metadata = usage_metadata
        self._managed_trial_usage_metadata_entitlement_ref = entitlement_ref

        remaining_percent = self._managed_trial_remaining_percent(usage_metadata)
        current_referral_id = self._current_owned_referral_id()
        self._set_managed_usage_view_state(
            view_settings=view_settings,
            visible=True,
            remaining_percent=remaining_percent,
            referral_id=current_referral_id,
        )

        if auto_show_founder_letter and is_effectively_exhausted(usage_metadata):
            self._disable_translation_for_managed_exhaustion(
                reopen_founder_letter=should_auto_show_founder_letter(self.settings, usage_metadata)
            )

        self._schedule_owned_referral_id_status_refresh(
            view_settings=view_settings,
            remaining_percent=remaining_percent,
            current_referral_id=current_referral_id,
        )

    def _show_founder_letter_dialog(self) -> None:
        if self.settings is None:
            return
        show_founder_letter_dialog = getattr(self.app, "show_founder_letter_dialog", None)
        if not callable(show_founder_letter_dialog):
            return
        show_founder_letter_dialog()
        mark_founder_letter_shown(self.settings)
        with contextlib.suppress(Exception):
            self._save_settings()

    def _disable_translation_for_managed_exhaustion(
        self,
        *,
        reopen_founder_letter: bool,
    ) -> None:
        self._record_translation_toggle_intent(False)
        self._set_managed_trial_pending_auth(False)
        if reopen_founder_letter:
            self._show_founder_letter_dialog()
        if self.hub is not None:
            self.hub.translation_enabled = False
        dash = getattr(self.app, "view_dashboard", None)
        if dash is not None:
            dash.set_translation_enabled(False)

    async def _should_route_managed_trans_to_founder_letter(self) -> bool:
        if self.settings is None:
            return False
        with contextlib.suppress(Exception):
            await self._refresh_managed_trial_usage_state_impl(auto_show_founder_letter=False)
        if not is_effectively_exhausted(self._managed_trial_usage_metadata):
            return False

        self._disable_translation_for_managed_exhaustion(reopen_founder_letter=True)
        return True

    def _build_llm_provider_signature(self, settings: AppSettings) -> tuple[object, ...]:
        return (
            settings.provider.llm,
            settings.gemini.llm_model if settings.provider.llm == LLMProviderName.GEMINI else None,
            (
                settings.openrouter.llm_model
                if settings.provider.llm == LLMProviderName.OPENROUTER
                else None
            ),
            (
                settings.openrouter.routing_mode
                if settings.provider.llm == LLMProviderName.OPENROUTER
                else None
            ),
            (
                settings.openrouter.provider_routing
                if settings.provider.llm == LLMProviderName.OPENROUTER
                else OpenRouterProviderRouting.DEFAULT
            ),
            (
                settings.openrouter.selected_source
                if settings.provider.llm == LLMProviderName.OPENROUTER
                else None
            ),
            (
                settings.openrouter.fallback_selection_alias
                if settings.provider.llm == LLMProviderName.OPENROUTER
                else None
            ),
            settings.qwen.llm_model if settings.provider.llm == LLMProviderName.QWEN else None,
            settings.qwen.region if settings.provider.llm == LLMProviderName.QWEN else None,
            (
                settings.deepseek.llm_model
                if settings.provider.llm == LLMProviderName.DEEPSEEK
                else None
            ),
            (
                settings.cerebras.llm_model
                if settings.provider.llm == LLMProviderName.CEREBRAS
                else None
            ),
            (
                (
                    settings.local_llm.backend,
                    settings.local_llm.base_url,
                    settings.local_llm.model,
                    _canonical_json_signature(settings.local_llm.extra_body),
                )
                if settings.provider.llm == LLMProviderName.LOCAL_LLM
                else None
            ),
        )

    def _sync_signature_caches(self, settings: AppSettings) -> None:
        current_self_signature = self._build_self_stt_runtime_signature(settings)
        self._last_stt_runtime_signature = current_self_signature
        self._last_self_stt_runtime_signature = current_self_signature
        self._last_peer_stt_runtime_signature = self._build_peer_stt_runtime_signature(settings)
        self._last_self_stt_provider_signature = self._build_self_stt_provider_signature(settings)
        self._last_peer_stt_provider_signature = self._build_peer_stt_provider_signature(settings)
        self._last_llm_provider_signature = self._build_llm_provider_signature(settings)
        self._last_microphone_test_audio_settings_signature = (
            self._microphone_test_audio_settings_signature(settings)
        )
        self._last_peer_translation_enabled = settings.ui.peer_translation_enabled
        self._last_peer_translation_activation_requested = (
            self._peer_translation_activation_requested_for(settings)
        )

    def _copy_provider_prompt_apply_fields(self, source: AppSettings, target: AppSettings) -> None:
        target.provider.stt = source.provider.stt
        target.provider.peer_stt = source.provider.peer_stt
        target.provider.llm = source.provider.llm
        target.translation = copy.deepcopy(source.translation)
        target.gemini.llm_model = source.gemini.llm_model
        target.openrouter.llm_model = source.openrouter.llm_model
        target.openrouter.routing_mode = source.openrouter.routing_mode
        target.openrouter.provider_routing = source.openrouter.provider_routing
        target.openrouter.selected_source = source.openrouter.selected_source
        target.openrouter.selection_alias = source.openrouter.selection_alias
        target.openrouter.fallback_selection_alias = source.openrouter.fallback_selection_alias
        target.qwen.llm_model = source.qwen.llm_model
        target.qwen.region = source.qwen.region
        target.deepseek.llm_model = source.deepseek.llm_model
        target.local_llm = copy.deepcopy(source.local_llm)
        if source.openrouter.selected_source == OpenRouterCredentialSource.MANAGED:
            target.managed_identity.verified_hardware_hash = (
                source.managed_identity.verified_hardware_hash
            )
            target.managed_identity.verified_hardware_hash_salt_version = (
                source.managed_identity.verified_hardware_hash_salt_version
            )
        else:
            target.managed_identity.verified_hardware_hash = None
            target.managed_identity.verified_hardware_hash_salt_version = None
        target.system_prompt = source.system_prompt
        target.system_prompts = {}

    def merge_settings_tab_apply_with_current_languages(self, pending: AppSettings) -> AppSettings:
        if self.settings is None:
            return copy.deepcopy(pending)

        merged = copy.deepcopy(self.settings)
        self._copy_provider_prompt_apply_fields(pending, merged)
        if self.hub is not None:
            merged.languages.source_language = self.hub.source_language
            merged.languages.target_language = self.hub.target_language
            merged.languages.peer_source_language = getattr(
                self.hub,
                "peer_source_language",
                merged.languages.peer_source_language,
            )
            merged.languages.peer_target_language = getattr(
                self.hub,
                "peer_target_language",
                merged.languages.peer_target_language,
            )
        return merged

    def _peer_runtime_should_be_active(self, settings: AppSettings) -> bool:
        return bool(
            self._peer_translation_activation_requested_for(settings)
            and self._effective_peer_overlay_enabled_for(settings)
            and self.hub is not None
            and self._overlay_bridge is not None
        )

    @staticmethod
    def _normalized_overlay_target(value: object) -> str:
        if value == OVERLAY_TARGET_DESKTOP:
            return OVERLAY_TARGET_DESKTOP
        return OVERLAY_TARGET_STEAMVR

    def _overlay_target_for_settings(self, settings: AppSettings | None = None) -> str:
        resolved_settings = settings or self.settings
        if resolved_settings is None:
            return OVERLAY_TARGET_STEAMVR
        return self._normalized_overlay_target(resolved_settings.overlay.target)

    def _overlay_runtime_is_active(self) -> bool:
        start_task = self._overlay_start_task
        return bool(
            self.overlay_state in {"starting", "connected"}
            or self._overlay_bridge is not None
            or self._overlay_manager is not None
            or (start_task is not None and not start_task.done())
        )

    def _previous_overlay_target_for_apply(self) -> str:
        if self._overlay_runtime_is_active() and self._active_overlay_target is not None:
            return self._active_overlay_target
        return self._overlay_target_for_settings(self.settings)

    def _overlay_process_runner_for_target(self, target: str) -> OverlayProcessRunner:
        if target == OVERLAY_TARGET_DESKTOP:
            return DesktopFletOverlayRunner()
        return DefaultOverlayProcessRunner()

    def _build_initial_desktop_runtime_controls(
        self,
        settings: AppSettings,
    ) -> list[dict[str, object]]:
        desktop_settings = copy.deepcopy(settings.overlay.desktop_flet)
        desktop_settings.validate()
        bounds = self._desktop_launch_bounds_for_current_launch(desktop_settings)
        visual = desktop_settings.visual
        interaction_mode = DESKTOP_INTERACTION_MODE_EDIT
        self.log_detailed(
            "[DesktopOverlay][Launch] "
            f"target=desktop locked={desktop_settings.locked} "
            f"interaction_mode={interaction_mode} "
            f"size_preset={desktop_settings.size_preset} "
            f"x={bounds['x']} y={bounds['y']} width={bounds['width']} "
            f"height={bounds['height']} "
            f"text_scale={visual.text_scale} "
            f"background_alpha={visual.background_alpha} "
            f"outline_width={visual.outline_width}"
        )
        return [
            {
                "command": "apply_window_bounds",
                "x": bounds["x"],
                "y": bounds["y"],
                "width": bounds["width"],
                "height": bounds["height"],
            },
            {
                "command": "apply_visual_config",
                "text_scale": visual.text_scale,
                "background_alpha": visual.background_alpha,
                "outline_width": visual.outline_width,
            },
            {"command": "set_interaction_mode", "mode": interaction_mode},
        ]

    @staticmethod
    def _desktop_dimensions_for_size_preset(size_preset: object) -> tuple[int, int]:
        if isinstance(size_preset, str) and size_preset in DESKTOP_FLET_SIZE_PRESETS:
            return DESKTOP_FLET_SIZE_PRESETS[size_preset]
        return DESKTOP_FLET_SIZE_PRESETS["medium"]

    def _desktop_launch_bounds_for_current_launch(
        self,
        desktop_settings: object,
    ) -> dict[str, int | float]:
        position = getattr(desktop_settings, "position", None)
        x = getattr(position, "x", None)
        y = getattr(position, "y", None)
        width, height = self._desktop_dimensions_for_size_preset(
            getattr(desktop_settings, "size_preset", None)
        )
        if self._is_finite_non_bool_number(x) and self._is_finite_non_bool_number(y):
            return {"x": x, "y": y, "width": width, "height": height}  # type: ignore[dict-item]
        return self._desktop_centered_bounds_for_dimensions(width=width, height=height)

    def _desktop_centered_bounds_for_dimensions(
        self,
        *,
        width: int | float,
        height: int | float,
    ) -> dict[str, int | float]:
        work_area = self._desktop_work_area_for_current_launch()
        if work_area is None:
            return {"x": 0, "y": 0, "width": width, "height": height}
        left, top, work_width, work_height = work_area
        if not (
            self._is_finite_non_bool_number(left)
            and self._is_finite_non_bool_number(top)
            and self._is_finite_non_bool_number(work_width)
            and self._is_finite_non_bool_number(work_height)
            and work_width > 0
            and work_height > 0
        ):
            return {"x": 0, "y": 0, "width": width, "height": height}

        return {
            "x": left + ((work_width - width) / 2),
            "y": top + ((work_height - height) / 2),
            "width": width,
            "height": height,
        }

    @staticmethod
    def _is_finite_non_bool_number(value: object) -> bool:
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        )

    @staticmethod
    def _desktop_bounds_signature(
        bounds: dict[str, int | float],
    ) -> tuple[float, float, float, float]:
        return (
            float(bounds["x"]),
            float(bounds["y"]),
            float(bounds["width"]),
            float(bounds["height"]),
        )

    def _desktop_bounds_from_payload(
        self,
        payload: dict[object, object],
    ) -> dict[str, int | float] | None:
        x = payload.get("x")
        y = payload.get("y")
        width = payload.get("width")
        height = payload.get("height")
        if not (
            self._is_finite_non_bool_number(x)
            and self._is_finite_non_bool_number(y)
            and self._is_finite_non_bool_number(width)
            and self._is_finite_non_bool_number(height)
        ):
            return None
        if width < DESKTOP_FLET_MIN_WIDTH or height < DESKTOP_FLET_MIN_HEIGHT:  # type: ignore[operator]
            return None
        return {
            "x": x,  # type: ignore[dict-item]
            "y": y,  # type: ignore[dict-item]
            "width": width,  # type: ignore[dict-item]
            "height": height,  # type: ignore[dict-item]
        }

    def _is_valid_desktop_window_bounds_event_payload(
        self,
        payload: dict[object, object],
    ) -> bool:
        source = payload.get("source")
        persist = payload.get("persist")
        if source not in {"user", "reset", "programmatic", "launch_repair"}:
            return False
        expected_persist = source in {"user", "reset"}
        return bool(
            payload.get("event") == "window_bounds_changed"
            and isinstance(persist, bool)
            and persist is expected_persist
            and self._desktop_bounds_from_payload(payload) is not None
        )

    def _track_desktop_apply_window_bounds_control(self, payload: dict[str, object]) -> None:
        if payload.get("command") != "apply_window_bounds":
            return
        bounds = self._desktop_bounds_from_payload(payload)
        if bounds is None:
            return
        self._desktop_suppressed_bounds_signatures.add(self._desktop_bounds_signature(bounds))

    def _consume_suppressed_desktop_bounds(self, bounds: dict[str, int | float]) -> bool:
        signature = self._desktop_bounds_signature(bounds)
        if signature not in self._desktop_suppressed_bounds_signatures:
            return False
        self._desktop_suppressed_bounds_signatures.discard(signature)
        return True

    def _discard_suppressed_desktop_bounds(self, bounds: dict[str, int | float]) -> None:
        self._desktop_suppressed_bounds_signatures.discard(self._desktop_bounds_signature(bounds))

    @staticmethod
    def _is_desktop_user_window_bounds_event(event: object) -> bool:
        if not isinstance(event, dict):
            return False
        payload = event.get("payload")
        if not isinstance(payload, dict):
            return False
        return bool(
            payload.get("event") == "window_bounds_changed"
            and payload.get("source") == "user"
            and payload.get("persist") is True
        )

    def _drain_pending_desktop_user_bounds_events(self) -> None:
        queue = self._desktop_renderer_events
        if queue is None:
            return
        retained: list[dict[str, object]] = []
        dropped = 0
        while True:
            try:
                event = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if self._is_desktop_user_window_bounds_event(event):
                dropped += 1
                continue
            retained.append(event)
        for event in retained:
            queue.put_nowait(event)
        if dropped:
            self.log_detailed(
                f"[DesktopOverlay][Bounds] drained_pending_user_bounds count={dropped}"
            )

    def _set_desktop_overlay_interaction_mode(self, mode: object) -> bool:
        if not isinstance(mode, str) or mode not in DESKTOP_INTERACTION_MODES:
            return False
        previous_mode = self.desktop_overlay_interaction_mode
        self.desktop_overlay_interaction_mode = mode
        if previous_mode != mode:
            self._notify_desktop_overlay_interaction_mode()
        return True

    def _notify_desktop_overlay_interaction_mode(self) -> None:
        handler = getattr(self.app, "on_desktop_overlay_state_changed", None)
        if callable(handler):
            handler(
                interaction_mode=self.desktop_overlay_interaction_mode,
                captions_locked=self.desktop_overlay_captions_locked,
            )

    async def set_desktop_overlay_captions_locked(self, locked: bool) -> None:
        if self.settings is None:
            return
        if self.overlay_state != "connected":
            return
        if self._active_overlay_target != OVERLAY_TARGET_DESKTOP or self._overlay_bridge is None:
            return

        mode = DESKTOP_INTERACTION_MODE_PASS_THROUGH if locked else DESKTOP_INTERACTION_MODE_EDIT
        if not await self._broadcast_desktop_runtime_control(
            {
                "command": "set_interaction_mode",
                "mode": mode,
            }
        ):
            return
        self._set_desktop_overlay_interaction_mode(mode)

    async def set_desktop_overlay_size_preset(self, size_preset: str) -> None:
        if self.settings is None:
            return
        normalized_size_preset = (
            size_preset if size_preset in DESKTOP_FLET_SIZE_PRESETS else "medium"
        )
        if self.settings.overlay.desktop_flet.size_preset == normalized_size_preset:
            return
        updated = copy.deepcopy(self.settings)
        updated.overlay.desktop_flet.size_preset = normalized_size_preset
        await self.apply_settings(updated)

    async def reset_desktop_overlay_position(self) -> None:
        await self._handle_desktop_overlay_reset_requested()

    async def _broadcast_desktop_runtime_control(self, payload: dict[str, object]) -> bool:
        if self._active_overlay_target != OVERLAY_TARGET_DESKTOP:
            return False
        bridge = self._overlay_bridge
        if bridge is None:
            return False
        broadcast = getattr(bridge, "broadcast_desktop_runtime_control", None)
        if not callable(broadcast):
            return False
        try:
            await broadcast(payload)
        except Exception as exc:
            self.log_detailed(
                "[Overlay] Failed to send desktop runtime control",
                level=logging.WARNING,
                exception=exc,
            )
            return False
        return True

    async def _broadcast_desktop_window_bounds_control(
        self,
        bounds: dict[str, int | float],
    ) -> None:
        payload: dict[str, object] = {
            "command": "apply_window_bounds",
            "x": bounds["x"],
            "y": bounds["y"],
            "width": bounds["width"],
            "height": bounds["height"],
        }
        if await self._broadcast_desktop_runtime_control(payload):
            self._track_desktop_apply_window_bounds_control(payload)

    async def _consume_desktop_renderer_events(
        self,
        queue: asyncio.Queue[dict[str, object]],
    ) -> None:
        try:
            while True:
                event = await queue.get()
                try:
                    await self._handle_desktop_renderer_event(event)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.log_detailed(
                        "[Overlay] Ignoring desktop renderer event after controller error",
                        level=logging.WARNING,
                        exception=exc,
                    )
        except asyncio.CancelledError:
            raise

    async def _handle_desktop_renderer_event(self, event: object) -> None:
        if self._active_overlay_target != OVERLAY_TARGET_DESKTOP:
            return
        if not isinstance(event, dict):
            return
        payload = event.get("payload")
        if not isinstance(payload, dict):
            return
        event_type = payload.get("event")
        if event_type == "window_bounds_changed":
            await self._handle_desktop_window_bounds_changed(payload)
            return
        if event_type == "reset_to_bottom_center_requested":
            await self._handle_desktop_overlay_reset_requested()
            return
        if event_type == "interaction_mode_changed":
            self._set_desktop_overlay_interaction_mode(payload.get("mode"))

    async def _handle_desktop_window_bounds_changed(
        self,
        payload: dict[object, object],
    ) -> None:
        if not self._is_valid_desktop_window_bounds_event_payload(payload):
            self.log_detailed(
                "[DesktopOverlay][Bounds] ignored reason=invalid_payload "
                f"keys={sorted(str(key) for key in payload)} "
                f"source={payload.get('source')} persist={payload.get('persist')}"
            )
            return
        bounds = self._desktop_bounds_from_payload(payload)
        if bounds is None:
            self.log_detailed(
                "[DesktopOverlay][Bounds] ignored reason=invalid_bounds "
                f"source={payload.get('source')} persist={payload.get('persist')}"
            )
            return
        source = payload.get("source")
        interaction_mode = self.desktop_overlay_interaction_mode
        self.log_detailed(
            "[DesktopOverlay][Bounds] received "
            f"source={source} persist={payload.get('persist')} "
            f"interaction_mode={interaction_mode} "
            f"x={bounds['x']} y={bounds['y']} width={bounds['width']} "
            f"height={bounds['height']}"
        )
        if source in {"programmatic", "launch_repair"}:
            self.log_detailed(
                "[DesktopOverlay][Bounds] ignored reason=programmatic_source "
                f"source={source} x={bounds['x']} y={bounds['y']} "
                f"width={bounds['width']} height={bounds['height']}"
            )
            self._discard_suppressed_desktop_bounds(bounds)
            return
        if source == "reset":
            self.log_detailed(
                "[DesktopOverlay][Bounds] reset_requested "
                f"x={bounds['x']} y={bounds['y']} width={bounds['width']} "
                f"height={bounds['height']}"
            )
            await self._handle_desktop_overlay_reset_requested(bounds=bounds)
            return
        if source == "user" and interaction_mode != DESKTOP_INTERACTION_MODE_EDIT:
            self.log_detailed(
                "[DesktopOverlay][Bounds] ignored reason=locked_interaction_mode "
                f"interaction_mode={interaction_mode} x={bounds['x']} y={bounds['y']} "
                f"width={bounds['width']} height={bounds['height']}"
            )
            return
        if self._consume_suppressed_desktop_bounds(bounds):
            self.log_detailed(
                "[DesktopOverlay][Bounds] ignored reason=suppressed_signature "
                f"x={bounds['x']} y={bounds['y']} width={bounds['width']} "
                f"height={bounds['height']}"
            )
            return
        self._schedule_desktop_bounds_persistence(bounds)
        self.log_detailed(
            "[DesktopOverlay][Bounds] scheduled_persist "
            f"x={bounds['x']} y={bounds['y']} width={bounds['width']} "
            f"height={bounds['height']}"
        )

    def _schedule_desktop_bounds_persistence(
        self,
        bounds: dict[str, int | float],
    ) -> None:
        self._pending_desktop_bounds = dict(bounds)
        task = self._desktop_bounds_persist_task
        if task is not None and not task.done():
            task.cancel()
        self._desktop_bounds_persist_task = asyncio.create_task(
            self._persist_desktop_bounds_after_debounce()
        )

    async def _persist_desktop_bounds_after_debounce(self) -> None:
        current_task = asyncio.current_task()
        try:
            await asyncio.sleep(DESKTOP_BOUNDS_PERSIST_DEBOUNCE_S)
            bounds = self._pending_desktop_bounds
            self._pending_desktop_bounds = None
            if bounds is None:
                return
            self._persist_desktop_bounds(bounds)
        except asyncio.CancelledError:
            raise
        finally:
            if self._desktop_bounds_persist_task is current_task:
                self._desktop_bounds_persist_task = None

    def _persist_desktop_bounds(self, bounds: dict[str, int | float]) -> None:
        if self.settings is None or self._active_overlay_target != OVERLAY_TARGET_DESKTOP:
            return
        if self._desktop_bounds_from_payload({"event": "window_bounds_changed", **bounds}) is None:
            return
        desktop_settings = self.settings.overlay.desktop_flet
        desktop_settings.position.x = bounds["x"]
        desktop_settings.position.y = bounds["y"]
        desktop_settings.position.validate()
        self.log_detailed(
            "[DesktopOverlay][Bounds] persisted "
            f"x={bounds['x']} y={bounds['y']} width={bounds['width']} "
            f"height={bounds['height']} size_preset={desktop_settings.size_preset}"
        )
        self._save_settings()

    async def _handle_desktop_overlay_reset_requested(
        self,
        *,
        bounds: dict[str, int | float] | None = None,
    ) -> None:
        if self.settings is None:
            return
        configured_for_desktop = (
            self._overlay_target_for_settings(self.settings) == OVERLAY_TARGET_DESKTOP
        )
        desktop_renderer_active = bool(
            self._active_overlay_target == OVERLAY_TARGET_DESKTOP
            and self._overlay_bridge is not None
        )
        if not configured_for_desktop and not desktop_renderer_active:
            return
        await self._cancel_desktop_bounds_persistence()
        self._drain_pending_desktop_user_bounds_events()
        _ = bounds
        desktop_settings = self.settings.overlay.desktop_flet
        desktop_settings.position.x = None
        desktop_settings.position.y = None
        desktop_settings.locked = False
        desktop_settings.validate()
        self._set_desktop_overlay_interaction_mode(DESKTOP_INTERACTION_MODE_EDIT)
        self._save_settings()
        if not desktop_renderer_active:
            return
        await self._broadcast_desktop_runtime_control(
            {
                "command": "set_interaction_mode",
                "mode": DESKTOP_INTERACTION_MODE_EDIT,
            }
        )
        await self._broadcast_desktop_window_bounds_control(
            self._desktop_center_bounds_for_current_preset()
        )

    def _desktop_center_bounds_for_current_preset(self) -> dict[str, int | float]:
        assert self.settings is not None
        width, height = self._desktop_dimensions_for_size_preset(
            self.settings.overlay.desktop_flet.size_preset
        )
        return self._desktop_centered_bounds_for_dimensions(width=width, height=height)

    def _desktop_work_area_for_current_launch(
        self,
    ) -> tuple[int | float, int | float, int | float, int | float] | None:
        _ = self
        if sys.platform != "win32":
            return None
        try:
            import ctypes
            from ctypes import wintypes

            rect = wintypes.RECT()
            # SPI_GETWORKAREA returns the primary monitor work area excluding taskbars.
            if not ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
                return None
            return (
                rect.left,
                rect.top,
                rect.right - rect.left,
                rect.bottom - rect.top,
            )
        except Exception:
            return None

    async def _cancel_desktop_renderer_event_task(self) -> None:
        current_task = asyncio.current_task()
        task = self._desktop_renderer_events_task
        self._desktop_renderer_events_task = None
        self._desktop_renderer_events = None
        if task is not None and task is not current_task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _cancel_desktop_bounds_persistence(self) -> None:
        current_task = asyncio.current_task()
        task = self._desktop_bounds_persist_task
        self._desktop_bounds_persist_task = None
        self._pending_desktop_bounds = None
        if task is not None and task is not current_task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def _discard_pending_desktop_bounds_persistence(self) -> None:
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            current_task = None
        task = self._desktop_bounds_persist_task
        self._desktop_bounds_persist_task = None
        self._pending_desktop_bounds = None
        if task is not None and task is not current_task and not task.done():
            task.cancel()

    def _desktop_runtime_is_running_for_settings_update(
        self,
        settings: AppSettings,
    ) -> bool:
        return bool(
            settings.ui.overlay_enabled
            and self._active_overlay_target == OVERLAY_TARGET_DESKTOP
            and self._overlay_bridge is not None
        )

    def _desktop_center_preserving_bounds_for_size_preset_change(
        self,
        *,
        previous_desktop_settings: object,
        next_size_preset: object,
    ) -> dict[str, int | float]:
        previous_bounds = self._desktop_launch_bounds_for_current_launch(previous_desktop_settings)
        next_width, next_height = self._desktop_dimensions_for_size_preset(next_size_preset)
        old_center_x = previous_bounds["x"] + (previous_bounds["width"] / 2)
        old_center_y = previous_bounds["y"] + (previous_bounds["height"] / 2)
        return {
            "x": old_center_x - (next_width / 2),
            "y": old_center_y - (next_height / 2),
            "width": next_width,
            "height": next_height,
        }

    def _prepare_desktop_runtime_settings_update(
        self,
        previous_settings: AppSettings | None,
        next_settings: AppSettings,
    ) -> list[dict[str, object]]:
        if previous_settings is None:
            return []
        previous_desktop = copy.deepcopy(previous_settings.overlay.desktop_flet)
        previous_desktop.validate()
        next_desktop = next_settings.overlay.desktop_flet
        next_desktop.validate()

        if not self._desktop_runtime_is_running_for_settings_update(next_settings):
            return []

        controls: list[dict[str, object]] = []
        if previous_desktop.size_preset != next_desktop.size_preset:
            self._discard_pending_desktop_bounds_persistence()
            self._drain_pending_desktop_user_bounds_events()
            bounds = self._desktop_center_preserving_bounds_for_size_preset_change(
                previous_desktop_settings=previous_desktop,
                next_size_preset=next_desktop.size_preset,
            )
            if previous_desktop.position.x is not None and previous_desktop.position.y is not None:
                next_desktop.position.x = bounds["x"]
                next_desktop.position.y = bounds["y"]
                next_desktop.position.validate()
            controls.append({"command": "apply_window_bounds", **bounds})

        previous_visual = previous_desktop.visual
        next_visual = next_desktop.visual
        if (
            previous_visual.text_scale != next_visual.text_scale
            or previous_visual.background_alpha != next_visual.background_alpha
            or previous_visual.outline_width != next_visual.outline_width
        ):
            controls.append(
                {
                    "command": "apply_visual_config",
                    "text_scale": next_visual.text_scale,
                    "background_alpha": next_visual.background_alpha,
                    "outline_width": next_visual.outline_width,
                }
            )
        return controls

    def _sync_desktop_overlay_interaction_mode_from_settings(
        self,
        settings: AppSettings,
    ) -> None:
        if self._overlay_target_for_settings(settings) != OVERLAY_TARGET_DESKTOP:
            return
        if (
            self._active_overlay_target == OVERLAY_TARGET_DESKTOP
            and self._overlay_bridge is not None
        ):
            return
        self._set_desktop_overlay_interaction_mode(DESKTOP_INTERACTION_MODE_EDIT)

    async def _broadcast_desktop_runtime_control_payloads(
        self,
        payloads: list[dict[str, object]],
    ) -> None:
        for payload in payloads:
            if payload.get("command") == "apply_window_bounds":
                bounds = self._desktop_bounds_from_payload(payload)
                if bounds is not None:
                    await self._broadcast_desktop_window_bounds_control(bounds)
                continue
            await self._broadcast_desktop_runtime_control(payload)

    def _build_peer_runtime_config(self, settings: AppSettings) -> PeerRuntimeConfig:
        backend = resolve_peer_stt_config(settings)
        provider_signature = build_peer_stt_provider_signature(settings)
        return PeerRuntimeConfig(
            backend=backend,
            output_device=settings.desktop_audio.output_device,
            vad_threshold=settings.desktop_audio.vad_speech_threshold,
            vad_hangover_ms=settings.desktop_audio.vad_hangover_ms,
            vad_pre_roll_ms=settings.desktop_audio.vad_pre_roll_ms,
            provider_signature=provider_signature,
            runtime_signature=(
                backend.source_language,
                settings.desktop_audio.output_device,
                settings.desktop_audio.vad_speech_threshold,
                settings.desktop_audio.vad_hangover_ms,
                settings.desktop_audio.vad_pre_roll_ms,
                provider_signature,
            ),
        )

    async def stop(self) -> None:
        await self._drain_github_star_prompt_translation_success_task()
        await self._stop_clipboard_watcher()
        await self._cancel_local_stt_download()
        await self.stop_microphone_test()
        await self.set_stt_enabled(False)
        await self._configure_vrc_mic_receiver(enabled=False)
        await self._shutdown_overlay_runtime(preserve_failure_reason=True)
        if self._peer_runtime is not None:
            with contextlib.suppress(Exception):
                await self._peer_runtime.close()
            self._peer_runtime = None

        if self._bridge_task:
            self._bridge_task.cancel()
            await asyncio.gather(self._bridge_task, return_exceptions=True)
            self._bridge_task = None
        self._ui_event_bridge = None

        if self.hub is not None:
            with contextlib.suppress(Exception):
                await self.hub.stop()
            self.hub = None

        if self.sender is not None:
            with contextlib.suppress(Exception):
                self.sender.close()
            self.sender = None
        self.osc = None
        await self._replace_managed_openrouter_release_service(None)
        if self._runtime_logging is not None:
            with contextlib.suppress(Exception):
                self._runtime_logging.close()
            self._runtime_logging = None

    async def set_overlay_enabled(self, enabled: bool) -> None:
        if self.settings is None:
            return

        self.log_basic(f"[Overlay] Toggle request: enabled={enabled}")
        self.log_detailed(
            "[Overlay] Toggle detail: "
            f"current_state={self.overlay_state} "
            f"has_bridge={self._overlay_bridge is not None} "
            f"has_manager={self._overlay_manager is not None}"
        )
        self.settings.ui.overlay_enabled = bool(enabled)
        if not enabled:
            self.settings.ui.peer_translation_enabled = False
            self._last_peer_translation_enabled = False
            self._last_peer_translation_activation_requested = False
        self._refresh_overlay_peer_consumers()

        if enabled:
            await self._begin_overlay_start()
            return

        await self._shutdown_overlay_runtime(preserve_failure_reason=True)

    async def set_peer_translation_enabled(self, enabled: bool) -> None:
        if self.settings is None:
            return

        enabled = bool(enabled)
        self.log_basic(f"[Peer] Toggle request: enabled={enabled}")
        self.log_detailed(
            "[Peer] Toggle detail: "
            f"overlay_enabled={self.settings.ui.overlay_enabled} "
            f"overlay_state={self.overlay_state} "
            f"peer_stt_available={self.hub is not None and getattr(self.hub, 'peer_stt', None) is not None} "
            f"eula_accepted={self.settings.ui.peer_translation_eula_accepted}"
        )

        if enabled and not self._peer_translation_eula_accepted_for(self.settings):
            self.settings.ui.peer_translation_enabled = False
            self._last_peer_translation_enabled = False
            self._last_peer_translation_activation_requested = False
            self._sync_effective_hub_flags(self.settings)
            self._refresh_overlay_peer_consumers()
            self.log_basic("[Peer] Toggle ignored: eula_accepted=False")
            return

        if enabled and not self.settings.ui.overlay_enabled:
            self.settings.ui.overlay_enabled = True
        self.settings.ui.peer_translation_enabled = enabled
        self._last_peer_translation_enabled = enabled
        self._last_peer_translation_activation_requested = (
            self._peer_translation_activation_requested_for(self.settings)
        )
        if enabled:
            await self._ensure_peer_local_stt_ready()
        self._clear_local_stt_pending_enable_if_provider_switched_away()
        self._sync_local_stt_notice()
        self._refresh_overlay_peer_consumers()

        if enabled and self.overlay_state not in {"starting", "connected"}:
            await self._begin_overlay_start()
        else:
            await self._refresh_overlay_runtime_dependencies()
        self._sync_effective_hub_flags(self.settings)
        if enabled:
            self._enqueue_peer_translation_disclosure()
        self._refresh_overlay_peer_consumers()

    def _enqueue_peer_translation_disclosure(self) -> None:
        hub = self.hub
        if hub is None:
            return
        enqueue_disclosure = getattr(hub, "enqueue_peer_translation_disclosure", None)
        if callable(enqueue_disclosure):
            enqueue_disclosure(t("peer_translation.disclosure"))

    def on_overlay_start_failed(self, failure_reason: str | None) -> None:
        previous_state = self.overlay_state
        self.overlay_state = "failed"
        self.failure_reason = self._normalize_overlay_failure_reason(failure_reason)
        self.auto_restart_scheduled = False
        self._log_overlay_state_transition(previous_state, self.overlay_state)
        self._sync_effective_hub_flags()
        self._notify_overlay_state()

    def on_overlay_runtime_disconnected(self) -> None:
        self.on_overlay_start_failed("runtime_disconnected")

    def on_overlay_runtime_crashed(self) -> None:
        self.on_overlay_start_failed("runtime_crashed")

    async def _begin_overlay_start(self) -> None:
        if self._overlay_lock is None:
            self._overlay_lock = asyncio.Lock()

        async with self._overlay_lock:
            if self.overlay_state in {"starting", "connected"}:
                return

            await self._teardown_overlay_runtime(preserve_presenter_state=True)
            self._active_overlay_target = self._overlay_target_for_settings(self.settings)
            previous_state = self.overlay_state
            self.overlay_state = "starting"
            self.auto_restart_scheduled = False
            self._log_overlay_state_transition(previous_state, self.overlay_state)
            self._notify_overlay_state()
            self._overlay_start_task = asyncio.create_task(self._run_overlay_start())

    async def _run_overlay_start(self) -> None:
        current_task = asyncio.current_task()
        try:
            if self.settings is None or self.hub is None:
                self._active_overlay_target = None
                self.on_overlay_start_failed("unknown")
                return

            presenter = self._overlay_presenter
            overlay_instance_id = f"overlay-{secrets.token_hex(8)}"
            diagnostics = OverlayDiagnosticsRecorder(overlay_instance_id=overlay_instance_id)
            overlay_target = self._active_overlay_target or self._overlay_target_for_settings(
                self.settings
            )
            self._active_overlay_target = overlay_target
            peer_presentation_refresh_burst = overlay_target != OVERLAY_TARGET_DESKTOP
            self_presentation_refresh_burst = overlay_target != OVERLAY_TARGET_DESKTOP
            self.log_detailed(
                "[Overlay][Start] "
                f"target={overlay_target} "
                f"overlay_instance_id={overlay_instance_id} "
                f"logging_mode={self.runtime_logging_mode} "
                f"peer_presentation_refresh_burst={peer_presentation_refresh_burst} "
                f"self_presentation_refresh_burst={self_presentation_refresh_burst}"
            )

            if presenter is None:
                presenter = OverlayPresenter(
                    calibration=self.overlay_calibration.copy(),
                    clock=self.clock,
                    diagnostics=diagnostics,
                    runtime_log_detailed=self.log_detailed,
                    show_translation=self.settings.overlay.show_translation,
                    show_peer_original=self.settings.overlay.show_peer_original,
                    peer_presentation_refresh_burst=peer_presentation_refresh_burst,
                    self_presentation_refresh_burst=self_presentation_refresh_burst,
                )
                self._overlay_presenter = presenter
            else:
                presenter.diagnostics = diagnostics
                presenter.runtime_log_detailed = self.log_detailed
                await presenter.update_peer_presentation_refresh_burst(
                    peer_presentation_refresh_burst
                )
                await presenter.update_self_presentation_refresh_burst(
                    self_presentation_refresh_burst
                )
            bridge = OverlayBridge(
                session_token=secrets.token_urlsafe(16),
                initial_snapshot=presenter.snapshot(),
                overlay_instance_id=overlay_instance_id,
                diagnostics=diagnostics,
                runtime_logging_mode=self.runtime_logging_mode,
                desktop_runtime_controls_enabled=overlay_target == OVERLAY_TARGET_DESKTOP,
            )
            if overlay_target == OVERLAY_TARGET_DESKTOP:
                initial_desktop_controls = self._build_initial_desktop_runtime_controls(
                    self.settings
                )
                initial_interaction_control = initial_desktop_controls[-1]
                self._set_desktop_overlay_interaction_mode(initial_interaction_control.get("mode"))
                for payload in initial_desktop_controls:
                    self._track_desktop_apply_window_bounds_control(payload)
                bridge.set_initial_desktop_runtime_controls(initial_desktop_controls)
            await bridge.start()
            presenter.attach_bridge(bridge)
            latest_snapshot = presenter.snapshot()
            if bridge.snapshot() != latest_snapshot:
                await bridge.replace_snapshot(latest_snapshot)
            self._overlay_bridge = bridge
            self._overlay_diagnostics = diagnostics
            self.hub.overlay_sink = presenter
            self.hub.overlay_diagnostics = diagnostics

            renderer_events: asyncio.Queue[dict[str, object]] | None = None
            if overlay_target == OVERLAY_TARGET_DESKTOP:
                renderer_events = asyncio.Queue(maxsize=64)
                self._desktop_renderer_events = renderer_events
                self._desktop_renderer_events_task = asyncio.create_task(
                    self._consume_desktop_renderer_events(renderer_events)
                )

            manager = OverlayProcessManager(
                process_runner=self._overlay_process_runner_for_target(overlay_target),
                bridge_url=bridge.url,
                bridge_messages=bridge.messages,
                session_token=bridge.session_token,
                locale=self.settings.ui.locale,
                startup_timeout_ms=OVERLAY_STARTUP_TIMEOUT_MS,
                renderer_events=renderer_events,
                overlay_instance_id=overlay_instance_id,
                logging_mode=self.runtime_logging_mode,
                diagnostics=diagnostics,
            )
            self._overlay_manager = manager
            await manager.start()

            if self._overlay_manager is not manager:
                return

            if manager.state != "connected":
                await self._handle_overlay_start_failure(manager.failure_reason)
                return

            self._mark_overlay_connected()
            await self._refresh_overlay_runtime_dependencies()
            monitor_task = getattr(manager, "_monitor_task", None)
            if monitor_task is not None:
                self._overlay_monitor_task = asyncio.create_task(
                    self._watch_overlay_runtime(manager, monitor_task)
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log_detailed(
                "[Overlay] Failed to start overlay runtime",
                level=logging.ERROR,
                exception=exc,
            )
            await self._handle_overlay_start_failure("unknown")
        finally:
            if self._overlay_start_task is current_task:
                self._overlay_start_task = None

    async def _watch_overlay_runtime(
        self,
        manager: OverlayProcessManager,
        monitor_task: asyncio.Task[None],
    ) -> None:
        current_task = asyncio.current_task()
        try:
            await monitor_task
            if self._overlay_manager is not manager:
                return
            if manager.state != "failed":
                return

            reason = self._normalize_overlay_failure_reason(manager.failure_reason)
            if reason == "runtime_disconnected":
                self.on_overlay_runtime_disconnected()
            elif reason == "runtime_crashed":
                self.on_overlay_runtime_crashed()
            else:
                self.on_overlay_start_failed(reason)
            await self._teardown_overlay_runtime(preserve_presenter_state=True)
            await self._refresh_overlay_runtime_dependencies()
        except asyncio.CancelledError:
            raise
        finally:
            if self._overlay_monitor_task is current_task:
                self._overlay_monitor_task = None

    async def _handle_overlay_start_failure(self, failure_reason: str | None) -> None:
        self.on_overlay_start_failed(failure_reason)
        await self._teardown_overlay_runtime(preserve_presenter_state=True)
        await self._refresh_overlay_runtime_dependencies()

    async def _shutdown_overlay_runtime(self, *, preserve_failure_reason: bool) -> None:
        if self._overlay_lock is None:
            self._overlay_lock = asyncio.Lock()

        self.log_basic("[Overlay] Shutdown requested")
        self.log_detailed(
            "[Overlay] Shutdown detail: "
            f"preserve_failure_reason={preserve_failure_reason} "
            f"state={self.overlay_state} "
            f"has_bridge={self._overlay_bridge is not None} "
            f"has_manager={self._overlay_manager is not None} "
            f"presenter_attached={self._overlay_presenter is not None}"
        )
        async with self._overlay_lock:
            has_runtime = (
                self._overlay_bridge is not None
                or self._overlay_manager is not None
                or (self._overlay_start_task is not None and not self._overlay_start_task.done())
            )
            if not has_runtime and self.overlay_state == "off":
                return

            previous_state = self.overlay_state
            self.overlay_state = "stopping"
            self.auto_restart_scheduled = False
            self._log_overlay_state_transition(previous_state, self.overlay_state)
            self._notify_overlay_state()

            await self._emit_overlay_shutdown()
            await self._teardown_overlay_runtime(preserve_presenter_state=False)
            previous_state = self.overlay_state
            self.overlay_state = "off"
            if not preserve_failure_reason:
                self.failure_reason = None
            self._log_overlay_state_transition(previous_state, self.overlay_state)
            self._sync_effective_hub_flags()
            await self._refresh_overlay_runtime_dependencies()
            self._notify_overlay_state()

    async def _emit_overlay_shutdown(self) -> None:
        presenter = self._overlay_presenter
        if presenter is None:
            return
        with contextlib.suppress(Exception):
            await presenter.broadcast_shutdown()
            await asyncio.sleep(OVERLAY_SHUTDOWN_GRACE_S)

    async def _teardown_overlay_runtime(self, *, preserve_presenter_state: bool) -> None:
        current_task = asyncio.current_task()

        start_task = self._overlay_start_task
        if start_task is not None and start_task is not current_task and not start_task.done():
            start_task.cancel()
            await asyncio.gather(start_task, return_exceptions=True)
        if start_task is not None and start_task.done():
            self._overlay_start_task = None

        monitor_task = self._overlay_monitor_task
        if (
            monitor_task is not None
            and monitor_task is not current_task
            and not monitor_task.done()
        ):
            monitor_task.cancel()
            await asyncio.gather(monitor_task, return_exceptions=True)
        if monitor_task is not None and monitor_task.done():
            self._overlay_monitor_task = None

        await self._cancel_desktop_renderer_event_task()
        await self._cancel_desktop_bounds_persistence()

        presenter = self._overlay_presenter
        if not preserve_presenter_state and presenter is not None:
            with contextlib.suppress(Exception):
                await presenter.clear_for_runtime_detach()
        if presenter is not None:
            presenter.detach_bridge()
        if (
            presenter is not None
            and self.hub is not None
            and getattr(self.hub, "overlay_sink", None) is presenter
        ):
            if preserve_presenter_state:
                self.hub.overlay_sink = presenter
            else:
                self.hub.overlay_sink = None
                self.hub.overlay_diagnostics = None
                with contextlib.suppress(Exception):
                    await self.hub.reset_overlay_preview()
        if not preserve_presenter_state and presenter is not None:
            presenter.reset_scene()
            self._overlay_presenter = None

        manager = self._overlay_manager
        self._overlay_manager = None
        if manager is not None:
            with contextlib.suppress(Exception):
                await manager.stop()

        bridge = self._overlay_bridge
        self._overlay_bridge = None
        if bridge is not None:
            with contextlib.suppress(Exception):
                await bridge.stop()
        self._active_overlay_target = None
        self._desktop_suppressed_bounds_signatures.clear()
        if not preserve_presenter_state:
            self._set_desktop_overlay_interaction_mode(DESKTOP_INTERACTION_MODE_EDIT)
        if not preserve_presenter_state:
            self._overlay_diagnostics = None

    def _mark_overlay_connected(self) -> None:
        previous_state = self.overlay_state
        self.overlay_state = "connected"
        self.failure_reason = None
        self.auto_restart_scheduled = False
        self._log_overlay_state_transition(previous_state, self.overlay_state)
        self._sync_effective_hub_flags()
        self._notify_overlay_state()

    def _normalize_overlay_failure_reason(self, failure_reason: str | None) -> str:
        if isinstance(failure_reason, str) and failure_reason in _OVERLAY_FAILURE_REASONS:
            return failure_reason
        return "unknown"

    def _notify_overlay_state(self) -> None:
        bridge = self._ui_event_bridge
        if bridge is not None:
            bridge.report_overlay_state(self.overlay_state, failure_reason=self.failure_reason)

    def _log_overlay_state_transition(self, previous_state: str, next_state: str) -> None:
        manager = self._overlay_manager
        transition_message = f"[Overlay] State transition: {previous_state} -> {next_state}"
        if self.failure_reason is not None:
            transition_message = f"{transition_message} failure_reason={self.failure_reason}"
        self.log_basic(transition_message)
        self.log_detailed(
            "[Overlay] State detail: "
            f"presenter_attached={self._overlay_presenter is not None} "
            f"bridge_attached={self._overlay_bridge is not None} "
            f"manager_state={manager.state if manager is not None else None}"
        )

    def begin_overlay_calibration(self) -> OverlayCalibration:
        if self._overlay_calibration_draft is None:
            self._overlay_calibration_draft = self.overlay_calibration.copy()
        return self._overlay_calibration_draft.copy()

    def set_overlay_calibration_field(
        self,
        field_name: str,
        value: object,
    ) -> OverlayCalibration:
        if self._overlay_calibration_draft is None:
            self._overlay_calibration_draft = self.overlay_calibration.copy()

        if field_name not in OverlayCalibration.__dataclass_fields__:
            raise ValueError(f"unknown overlay calibration field: {field_name}")

        if field_name == "anchor":
            setattr(self._overlay_calibration_draft, field_name, str(value))
        else:
            setattr(self._overlay_calibration_draft, field_name, float(value))

        self._overlay_calibration_draft.validate()
        return self._overlay_calibration_draft.copy()

    def apply_overlay_calibration(self) -> OverlayCalibration:
        if self._overlay_calibration_draft is None:
            return self.overlay_calibration.copy()

        self._overlay_calibration_draft.validate()
        self.overlay_calibration = self._overlay_calibration_draft.copy()
        self._overlay_calibration_draft = None
        if self.settings is not None:
            self.settings.overlay.calibration = self.overlay_calibration.copy()
            self._save_settings()
        self._schedule_overlay_calibration_emit()
        return self.overlay_calibration.copy()

    def cancel_overlay_calibration(self) -> OverlayCalibration:
        self._overlay_calibration_draft = None
        return self.overlay_calibration.copy()

    def _sync_overlay_calibration_cache(self, settings: AppSettings | None = None) -> None:
        resolved_settings = settings or self.settings
        if resolved_settings is None:
            return
        self.overlay_calibration = resolved_settings.overlay.calibration.copy()

    async def _emit_overlay_calibration_update(self) -> None:
        presenter = self._overlay_presenter
        if presenter is None:
            return
        with contextlib.suppress(Exception):
            await presenter.update_calibration(self.overlay_calibration.copy())

    def _schedule_overlay_calibration_emit(self) -> None:
        if self._overlay_presenter is None:
            return
        run_task = getattr(self.page, "run_task", None)
        if callable(run_task):
            try:
                run_task(self._emit_overlay_calibration_update)
                return
            except Exception as exc:
                self.log_detailed(
                    "[Overlay] Failed to schedule calibration update via page.run_task",
                    level=logging.WARNING,
                    exception=exc,
                )
                return

        try:
            asyncio.get_running_loop().create_task(self._emit_overlay_calibration_update())
        except RuntimeError:
            self.log_detailed(
                "[Overlay] Skipping calibration update; no running loop and page.run_task unavailable",
                level=logging.WARNING,
            )

    def begin_overlay_calibration_for_test(self) -> None:
        self.begin_overlay_calibration()

    def set_overlay_calibration_field_for_test(self, field_name: str, value: object) -> None:
        self.set_overlay_calibration_field(field_name, value)

    def apply_overlay_calibration_for_test(self) -> None:
        self.apply_overlay_calibration()

    def cancel_overlay_calibration_for_test(self) -> None:
        self.cancel_overlay_calibration()

    async def set_translation_enabled(self, enabled: bool) -> bool:
        request_generation = self._record_translation_toggle_intent(enabled)
        if not enabled:
            self._set_managed_trial_pending_auth(False)
        if self.hub is None:
            return False
        self.log_basic(f"[Translation] Toggle request: enabled={enabled}")
        self.log_detailed(
            "[Translation] Toggle detail: "
            f"current_enabled={self.hub.translation_enabled} "
            f"llm_available={self.hub.llm is not None}"
        )
        if enabled and await self._handle_managed_translation_enable(request_generation) is False:
            return False
        if enabled and not self._translation_toggle_intent_matches(
            enabled=True,
            generation=request_generation,
        ):
            self.log_detailed(
                "[Translation] Skipping stale enable request after newer toggle intent"
            )
            return False
        if enabled and self.hub.llm is None:
            self.hub.translation_enabled = False
            dash = getattr(self.app, "view_dashboard", None)
            if dash is not None:
                dash.set_translation_enabled(False)
            self._log_error("Translation is ON but LLM provider is not configured.")
            return False

        # Log provider info when enabling
        if enabled and self.settings is not None:
            provider = self.settings.provider.llm.value
            if provider == "qwen":
                region = self.settings.qwen.region.value
                self.log_basic(f"[Translation] Enabled with provider: {provider}")
                self.log_detailed(
                    f"[Translation] Provider detail: provider={provider} region={region}"
                )
            else:
                self.log_basic(f"[Translation] Enabled with provider: {provider}")

        # Clear context history when toggling translation
        self.hub.clear_context()
        self.hub.translation_enabled = bool(enabled)
        if enabled and self.hub.llm is not None:
            llm = self.hub.llm
            if isinstance(llm, SemaphoreLLMProvider):
                llm = llm.inner
            if isinstance(llm, (GeminiLLMProvider, QwenLLMProvider, AsyncQwenLLMProvider)):
                with contextlib.suppress(Exception):
                    await llm.warmup()
        return bool(self.hub.translation_enabled)

    async def set_stt_enabled(self, enabled: bool) -> None:
        self.log_basic(f"[STT] Toggle request: enabled={enabled}")
        self.log_detailed(
            "[STT] Toggle detail: "
            f"desired_before={self._stt_desired} overlay_state={self.overlay_state}"
        )
        self._stt_desired = bool(enabled)
        if not enabled:
            self._reset_local_stt_pending_enable_after_install()

        # Log provider info when enabling
        if enabled and self.settings is not None:
            provider = self.settings.provider.stt.value
            if provider == "qwen_asr":
                region = self.settings.qwen.region.value
                self.log_basic(f"[STT] Enabled with provider: {provider}")
                self.log_detailed(f"[STT] Provider detail: provider={provider} region={region}")
            else:
                self.log_basic(f"[STT] Enabled with provider: {provider}")

        if (
            enabled
            and self.settings is not None
            and self.settings.provider.stt == STTProviderName.LOCAL_QWEN
        ):
            current_status = self._current_local_stt_runtime_status()
            if current_status == "downloading":
                self._local_stt_pending_enable_after_install = True
                self._stt_desired = False
                dash = getattr(self.app, "view_dashboard", None)
                if dash is not None:
                    dash.set_stt_enabled(False)
                self._show_short_stt_message("local_stt.download_in_progress")
                return
            if current_status in ("missing", "invalid", "download_failed"):
                self._handle_local_stt_unavailable(
                    current_status,
                    resume_self=True,
                    resume_peer=self._peer_local_stt_requested(self.settings),
                )
                return

        # Mark promo eligible when user explicitly enables STT via button
        if enabled and self.hub is not None:
            self.hub.mark_promo_eligible()

        await self._ensure_stt_switch()

    def _show_short_stt_message(self, message_key: str) -> None:
        self._show_short_message(message_key)

    def _show_short_message(self, message_key: str, **message_kwargs: object) -> None:
        message = t(message_key, **message_kwargs)
        show_snackbar = getattr(self.app, "_show_snackbar", None)
        if callable(show_snackbar):
            with contextlib.suppress(Exception):
                show_snackbar(message, ft.Colors.ORANGE_700)
                return
        opener = getattr(self.page, "open", None)
        if callable(opener):
            with contextlib.suppress(Exception):
                opener(
                    ft.SnackBar(
                        ft.Text(message, color=ft.Colors.WHITE),
                        bgcolor=ft.Colors.ORANGE_700,
                        duration=4000,
                        behavior=ft.SnackBarBehavior.FLOATING,
                        margin=ft.margin.only(bottom=90),
                        padding=20,
                    )
                )
                return
        self._log_error(message)

    async def _handle_managed_translation_enable(self, request_generation: int) -> bool:
        if self.settings is None or self.hub is None:
            return True
        if self.settings.provider.llm != LLMProviderName.OPENROUTER:
            return True
        if self.settings.openrouter.selected_source != OpenRouterCredentialSource.MANAGED:
            return True
        if (
            self._is_managed_china_connection()
            and not self._managed_openrouter_local_key_available()
        ):
            self._set_managed_trial_pending_auth(False)
            self.hub.translation_enabled = False
            dash = getattr(self.app, "view_dashboard", None)
            if dash is not None:
                dash.set_translation_enabled(False)
            self._show_short_message("qq_auth.error.key_unavailable")
            return False
        if await self._should_route_managed_trans_to_founder_letter():
            return False
        service = self._managed_openrouter_release_service
        if service is None:
            return True

        self._set_managed_trial_pending_auth(
            self._should_show_managed_auth_pending_before_prepare()
        )
        try:
            result = await service.prepare_for_translation()
        except Exception:
            self._set_managed_trial_pending_auth(False)
            raise

        self._set_managed_trial_pending_auth(False)

        if not self._translation_toggle_intent_matches(
            enabled=True,
            generation=request_generation,
        ):
            self.log_detailed(
                "[Translation] Skipping stale managed enable result after newer toggle intent"
            )
            return False

        if result.behavior == ManagedOpenRouterReleaseBehavior.READY and result.local_key_available:
            if self.hub.llm is None:
                await self._rebuild_llm_provider()
            else:
                self._schedule_managed_trial_usage_refresh()
            return True

        diagnostics_text = format_managed_openrouter_diagnostics(result.diagnostics)
        if diagnostics_text:
            self.log_basic(f"[ManagedAuth] {diagnostics_text}", level=logging.ERROR)
        await self._refresh_managed_trial_usage_state_impl(auto_show_founder_letter=False)
        self.hub.translation_enabled = False
        dash = getattr(self.app, "view_dashboard", None)
        if dash is not None:
            dash.set_translation_enabled(False)
        self._show_short_message(result.message_key, **dict(result.message_kwargs))
        return False

    def _refresh_local_stt_runtime_state(self) -> None:
        if self.settings is None:
            return
        self._local_stt_install_state = inspect_local_stt_install_state()
        if self._local_stt_runtime_status not in ("downloading", "download_failed"):
            self._local_stt_runtime_status = self._local_stt_install_state.status
        self._sync_local_stt_notice()

    def _current_local_stt_runtime_status(self) -> str:
        if self._local_stt_runtime_status in ("downloading", "download_failed"):
            return self._local_stt_runtime_status
        return self._local_stt_install_state.status

    def _peer_local_stt_requested(self, settings: AppSettings | None = None) -> bool:
        resolved_settings = settings or self.settings
        return bool(
            resolved_settings is not None
            and resolved_settings.provider.peer_stt == STTProviderName.LOCAL_QWEN
            and self._peer_translation_activation_requested_for(resolved_settings)
        )

    def _reset_local_stt_pending_enable_after_install(self) -> None:
        self._local_stt_pending_enable_after_install = False

    def _reset_local_stt_pending_peer_enable_after_install(self) -> None:
        self._local_stt_pending_peer_enable_after_install = False

    def _clear_local_stt_pending_enable_if_provider_switched_away(self) -> None:
        if self.settings is None:
            return
        if self.settings.provider.stt != STTProviderName.LOCAL_QWEN:
            self._reset_local_stt_pending_enable_after_install()
        if not self._peer_local_stt_requested(self.settings):
            self._reset_local_stt_pending_peer_enable_after_install()

    def _sync_local_stt_notice(self) -> None:
        dash = getattr(self.app, "view_dashboard", None)
        if dash is None or self.settings is None:
            return
        status = self._current_local_stt_runtime_status()
        should_show = status == "downloading" or (
            (
                self.settings.provider.stt == STTProviderName.LOCAL_QWEN
                or self._peer_local_stt_requested(self.settings)
            )
            and status != "ready"
        )
        with contextlib.suppress(Exception):
            dash.set_local_stt_notice(
                status if should_show else None,
                percent=self._local_stt_download_percent if status == "downloading" else None,
            )

    def _start_local_stt_download(self, *, origin: str) -> bool:
        task = self._local_stt_download_task
        if task is not None and not task.done():
            return False
        self._local_stt_download_origin = origin
        self._local_stt_download_percent = 0
        self._local_stt_download_cancel_event = threading.Event()
        self._local_stt_download_task = asyncio.create_task(
            self._run_local_stt_download(origin=origin)
        )
        return True

    async def _run_local_stt_download(self, *, origin: str) -> None:
        current_task = asyncio.current_task()
        cancel_event = self._local_stt_download_cancel_event
        if self.settings is None:
            return
        self._local_stt_runtime_status = "downloading"
        self._local_stt_download_percent = 0
        self._sync_local_stt_notice()
        try:
            installed = await ensure_local_stt_installed(
                locale=self.settings.ui.locale,
                on_status=self._handle_local_stt_download_status,
                cancel_event=cancel_event,
            )
        except (asyncio.CancelledError, LocalSTTRuntimeInstallCancelled):
            return
        except LocalSTTRuntimeInstallError as exc:
            self._local_stt_runtime_status = "download_failed"
            self._local_stt_download_percent = None
            self._sync_local_stt_notice()
            if origin == "manual":
                self._show_short_stt_message("local_stt.download_failed")
            self._log_error(f"Local STT download failed: {exc}")
            return
        finally:
            if self._local_stt_download_task is current_task:
                self._local_stt_download_task = None
            if self._local_stt_download_cancel_event is cancel_event:
                self._local_stt_download_cancel_event = None
            if self._local_stt_download_origin == origin:
                self._local_stt_download_origin = None

        self._local_stt_install_state = LocalSTTInstallState(
            status="ready",
            installed_manifest=installed,
        )
        self._local_stt_runtime_status = "ready"
        self._local_stt_download_percent = None
        self._clear_local_stt_pending_enable_if_provider_switched_away()
        self._sync_local_stt_notice()

        should_resume_self_local_stt = (
            origin == "manual"
            and self.settings is not None
            and self.settings.provider.stt == STTProviderName.LOCAL_QWEN
            and self._local_stt_pending_enable_after_install
        )
        should_resume_peer_local_stt = (
            origin == "manual"
            and self.settings is not None
            and self._peer_local_stt_requested(self.settings)
            and self._local_stt_pending_peer_enable_after_install
        )

        if should_resume_self_local_stt:
            self._reset_local_stt_pending_enable_after_install()
            await self._rebuild_stt_provider()
            self._stt_desired = True
            dash = getattr(self.app, "view_dashboard", None)
            if dash is not None:
                dash.set_stt_enabled(True)
            await self._ensure_stt_switch()

        if should_resume_peer_local_stt:
            self._reset_local_stt_pending_peer_enable_after_install()
            await self._refresh_overlay_runtime_dependencies()

    async def _handle_local_stt_download_status(self, update: RuntimeLocalSTTStatusUpdate) -> None:
        self._local_stt_runtime_status = update.status
        self._local_stt_download_percent = update.percent
        self._sync_local_stt_notice()

    def _handle_local_stt_unavailable(
        self,
        status: str,
        *,
        resume_self: bool,
        resume_peer: bool,
    ) -> bool:
        if status in ("missing", "invalid"):
            self._local_stt_install_state = LocalSTTInstallState(status=status)
        if self._local_stt_runtime_status != "downloading":
            self._local_stt_runtime_status = status
            self._local_stt_download_percent = None
        if resume_self:
            self._local_stt_pending_enable_after_install = True
            self._stt_desired = False
        if resume_peer:
            self._local_stt_pending_peer_enable_after_install = True
        dash = getattr(self.app, "view_dashboard", None)
        if resume_self and dash is not None:
            dash.set_stt_enabled(False)
            dash.set_stt_needs_key(False)
        self._sync_local_stt_notice()
        self._start_local_stt_download(origin="manual")
        return False

    async def _ensure_local_stt_ready(self) -> bool:
        if self.settings is None or self.settings.provider.stt != STTProviderName.LOCAL_QWEN:
            return True
        current_status = self._current_local_stt_runtime_status()
        if current_status == "downloading":
            self._stt_desired = False
            self._local_stt_pending_enable_after_install = True
            if self._peer_local_stt_requested(self.settings):
                self._local_stt_pending_peer_enable_after_install = True
            dash = getattr(self.app, "view_dashboard", None)
            if dash is not None:
                dash.set_stt_enabled(False)
            self._show_short_stt_message("local_stt.download_in_progress")
            return False
        if current_status in ("missing", "invalid", "download_failed"):
            return self._handle_local_stt_unavailable(
                current_status,
                resume_self=True,
                resume_peer=self._peer_local_stt_requested(self.settings),
            )
        if self.hub is None or self.hub.stt is None:
            self._stt_desired = False
            dash = getattr(self.app, "view_dashboard", None)
            if dash is not None:
                dash.set_stt_enabled(False)
                dash.set_stt_needs_key(False)
            self._show_short_stt_message("error.local_stt_model_invalid")
            return False
        try:
            await self.hub.stt.warmup()
            self._local_stt_install_state = LocalSTTInstallState(status="ready")
            if self._local_stt_runtime_status != "downloading":
                self._local_stt_runtime_status = "ready"
            self._sync_local_stt_notice()
            return True
        except LocalSTTModelMissingError:
            return self._handle_local_stt_unavailable(
                "missing",
                resume_self=True,
                resume_peer=self._peer_local_stt_requested(self.settings),
            )
        except (LocalSTTManifestInvalidError, LocalQwenSherpaLoadError):
            return self._handle_local_stt_unavailable(
                "invalid",
                resume_self=True,
                resume_peer=self._peer_local_stt_requested(self.settings),
            )

    async def _ensure_peer_local_stt_ready(self) -> bool:
        if self.settings is None or not self._peer_local_stt_requested(self.settings):
            return True
        current_status = self._current_local_stt_runtime_status()
        if current_status == "downloading":
            self._local_stt_pending_peer_enable_after_install = True
            self._sync_local_stt_notice()
            return False
        if current_status in ("missing", "invalid", "download_failed"):
            return self._handle_local_stt_unavailable(
                current_status,
                resume_self=False,
                resume_peer=True,
            )
        try:
            await self._probe_peer_local_stt_runtime_load()
            self._local_stt_install_state = LocalSTTInstallState(status="ready")
            if self._local_stt_runtime_status != "downloading":
                self._local_stt_runtime_status = "ready"
            self._sync_local_stt_notice()
            return True
        except LocalSTTModelMissingError:
            return self._handle_local_stt_unavailable(
                "missing",
                resume_self=False,
                resume_peer=True,
            )
        except (LocalSTTManifestInvalidError, LocalQwenSherpaLoadError):
            return self._handle_local_stt_unavailable(
                "invalid",
                resume_self=False,
                resume_peer=True,
            )

    async def _probe_peer_local_stt_runtime_load(self) -> None:
        assert self.settings is not None
        secrets = create_secret_store(self.settings.secrets, config_path=self.config_path)
        peer_backend = create_peer_stt_backend(
            self.settings,
            secrets=secrets,
            diagnostics_enabled=self._detailed_audio_diag_enabled,
        )
        session = None
        try:
            session = await peer_backend.open_session()
        finally:
            if session is not None:
                with contextlib.suppress(Exception):
                    await session.close()
            close_backend = getattr(peer_backend, "close", None)
            if callable(close_backend):
                with contextlib.suppress(Exception):
                    await close_backend()

    async def _cancel_local_stt_download(self) -> None:
        task = self._local_stt_download_task
        cancel_event = self._local_stt_download_cancel_event
        self._reset_local_stt_pending_enable_after_install()
        self._reset_local_stt_pending_peer_enable_after_install()
        if cancel_event is not None:
            cancel_event.set()
        if task is None:
            self._local_stt_download_cancel_event = None
            return
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._local_stt_download_task = None
        self._local_stt_download_cancel_event = None

    async def _ensure_stt_switch(self) -> None:
        if self._stt_switch_task is None or self._stt_switch_task.done():
            self._stt_switch_task = asyncio.create_task(self._run_stt_switch())
        await self._stt_switch_task

    async def _replace_runtime_stt_provider(self) -> None:
        self.log_detailed(
            "[STT] Replacing runtime provider detail: "
            f"desired={self._stt_desired} mic_task_active={self._mic_task is not None}"
        )
        if self._mic_task is not None:
            await self._stop_mic_loop()
        self._stt_restart_requested = False
        await self._rebuild_stt_provider()
        if self._stt_desired:
            await self._ensure_stt_switch()

    async def _run_stt_switch(self) -> None:
        if self._stt_switch_lock is None:
            self._stt_switch_lock = asyncio.Lock()
        async with self._stt_switch_lock:
            while True:
                desired = self._stt_desired
                restart = self._stt_restart_requested
                self._stt_restart_requested = False

                if not desired:
                    await self._stop_mic_loop()
                    if self.hub is not None:
                        with contextlib.suppress(Exception):
                            await self.hub.stt.close()
                else:
                    if self.hub is None:
                        self.log_detailed(
                            "[STT] Enable requested before hub is ready",
                            level=logging.WARNING,
                        )
                        break
                    if restart:
                        await self._stop_mic_loop()
                        with contextlib.suppress(Exception):
                            await self.hub.stt.close()
                    if not await self._ensure_local_stt_ready():
                        break
                    await self._start_mic_loop()
                    # Pre-warm STT session for faster first response
                    if (
                        self.hub is not None
                        and self.hub.stt is not None
                        and self._selected_stt_provider() != STTProviderName.LOCAL_QWEN
                    ):
                        with contextlib.suppress(Exception):
                            await self.hub.stt.warmup()

                if desired == self._stt_desired and not self._stt_restart_requested:
                    break

    def _get_clipboard_watcher_lock(self) -> asyncio.Lock:
        if self._clipboard_watcher_lock is None:
            self._clipboard_watcher_lock = asyncio.Lock()
        return self._clipboard_watcher_lock

    async def _sync_clipboard_watcher(self) -> None:
        enabled = bool(
            self.settings is not None and self.settings.ui.clipboard_auto_translate_enabled
        )
        if not enabled or sys.platform != "win32":
            await self._stop_clipboard_watcher()
            return
        async with self._get_clipboard_watcher_lock():
            if self._clipboard_watcher is not None:
                return

            self._clipboard_loop = asyncio.get_running_loop()
            watcher = create_clipboard_watcher(self._on_clipboard_text_from_thread)
            try:
                await asyncio.to_thread(watcher.start)
            except Exception as exc:
                self._clipboard_loop = None
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(watcher.stop)
                self._log_error(f"Clipboard watcher failed to start: {exc}")
                return
            self._clipboard_watcher = watcher

    async def _stop_clipboard_watcher(self) -> None:
        async with self._get_clipboard_watcher_lock():
            watcher = self._clipboard_watcher
            self._clipboard_watcher = None
            self._clipboard_loop = None
            if watcher is None:
                return
            try:
                await asyncio.to_thread(watcher.stop)
            except Exception as exc:
                self._log_error(f"Clipboard watcher failed to stop: {exc}")

    def _on_clipboard_text_from_thread(self, text: str) -> None:
        trimmed = text.strip()
        if not trimmed or len(trimmed) > 300:
            return
        loop = self._clipboard_loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._schedule_clipboard_submit, trimmed)

    def _schedule_clipboard_submit(self, text: str) -> None:
        try:
            asyncio.create_task(self._submit_clipboard_text(text))
        except RuntimeError as exc:
            self._log_error(f"Clipboard submit scheduling failed: {exc}")

    async def _submit_clipboard_text(self, text: str) -> None:
        if self.hub is None:
            return
        try:
            await self.hub.submit_text(text, source="Clipboard")
        except Exception as exc:
            self._log_error(f"Clipboard submit failed: {exc}")

    async def submit_text(self, text: str) -> None:
        if self.hub is None:
            return
        try:
            await self.hub.submit_text(text, source="You")
        except Exception as exc:
            self._log_error(f"Submit failed: {exc}")

    async def on_dashboard_language_change(
        self,
        *,
        source_code: str,
        target_code: str,
        peer_source_code: str = "",
        peer_target_code: str = "",
    ) -> None:
        if self.settings is None:
            return

        updated = copy.deepcopy(self.settings)
        updated.languages.source_language = source_code
        updated.languages.target_language = target_code
        updated.languages.peer_source_language = peer_source_code
        updated.languages.peer_target_language = peer_target_code
        await self.apply_settings(updated)

    async def apply_settings(self, settings: AppSettings) -> None:
        def _effective_peer_language(language: str, peer_language: str) -> str:
            return peer_language or language

        await self._preserve_github_star_prompt_observation_before_settings_replace(settings)
        prev_microphone_test_audio_signature = (
            self._last_microphone_test_audio_settings_signature
            or self._microphone_test_audio_settings_signature(self.settings)
        )
        next_microphone_test_audio_signature = self._microphone_test_audio_settings_signature(
            settings
        )
        if (
            prev_microphone_test_audio_signature is not None
            and prev_microphone_test_audio_signature != next_microphone_test_audio_signature
        ):
            await self.stop_microphone_test_for_audio_settings_change()

        prev_locale = get_locale()
        prev_overlay_enabled = (
            self.settings.ui.overlay_enabled if self.settings is not None else False
        )
        previous_settings_for_desktop = (
            copy.deepcopy(self.settings) if self.settings is not None else None
        )
        prev_overlay_target = self._previous_overlay_target_for_apply()
        next_overlay_target = self._overlay_target_for_settings(settings)
        if (
            prev_overlay_target != next_overlay_target
            and prev_overlay_enabled
            and settings.ui.overlay_enabled
            and self._overlay_runtime_is_active()
        ):
            self.log_basic(
                "[Overlay] Target changed while running; stopping current overlay before switch"
            )
            settings = copy.deepcopy(settings)
            settings.ui.overlay_enabled = False
        desktop_runtime_controls = self._prepare_desktop_runtime_settings_update(
            previous_settings_for_desktop,
            settings,
        )
        prev_peer_translation_enabled = (
            self._last_peer_translation_enabled
            if self._last_peer_translation_enabled is not None
            else (self.settings.ui.peer_translation_enabled if self.settings is not None else False)
        )
        prev_peer_activation_requested = (
            self._last_peer_translation_activation_requested
            if self._last_peer_translation_activation_requested is not None
            else (
                self._peer_translation_activation_requested_for(self.settings)
                if self.settings is not None
                else False
            )
        )
        prev_self_signature = (
            self._last_self_stt_runtime_signature or self._last_stt_runtime_signature
        )
        prev_peer_signature = self._last_peer_stt_runtime_signature
        # hub.source_language를 기준으로 비교 (settings 객체는 이미 수정되어 전달될 수 있음)
        prev_source_lang = self.hub.source_language if self.hub else None
        prev_target_lang = self.hub.target_language if self.hub else None
        prev_peer_source_lang = (
            getattr(self.hub, "peer_source_language", None) if self.hub else None
        )
        prev_peer_target_lang = (
            getattr(self.hub, "peer_target_language", None) if self.hub else None
        )
        prev_effective_peer_source = (
            _effective_peer_language(prev_source_lang, prev_peer_source_lang)
            if prev_source_lang is not None and prev_peer_source_lang is not None
            else None
        )
        prev_effective_peer_target = (
            _effective_peer_language(prev_target_lang, prev_peer_target_lang)
            if prev_target_lang is not None and prev_peer_target_lang is not None
            else None
        )
        prev_low_latency = self.hub.low_latency_mode if self.hub else None
        source_language_changed = (
            prev_source_lang is not None and prev_source_lang != settings.languages.source_language
        )
        target_language_changed = (
            prev_target_lang is not None and prev_target_lang != settings.languages.target_language
        )
        effective_peer_source_changed = (
            prev_effective_peer_source is not None
            and prev_effective_peer_source
            != _effective_peer_language(
                settings.languages.source_language,
                settings.languages.peer_source_language,
            )
        )
        effective_peer_target_changed = (
            prev_effective_peer_target is not None
            and prev_effective_peer_target
            != _effective_peer_language(
                settings.languages.target_language,
                settings.languages.peer_target_language,
            )
        )
        if source_language_changed or target_language_changed:
            presenter = self._overlay_presenter
            self.log_basic(
                "[Settings] Applying languages: "
                f"source={prev_source_lang}->{settings.languages.source_language} "
                f"target={prev_target_lang}->{settings.languages.target_language}"
            )
            self.log_detailed(
                "[Settings] Language apply detail: "
                f"overlay_state={self.overlay_state} "
                f"presenter_attached={presenter is not None} "
                f"bridge_attached={self._overlay_bridge is not None} "
                "overlay_sink_matches_presenter="
                f"{self.hub is not None and presenter is not None and getattr(self.hub, 'overlay_sink', None) is presenter}"
            )
        self.settings = settings
        self._last_microphone_test_audio_settings_signature = next_microphone_test_audio_signature
        self._sync_overlay_calibration_cache(settings)
        self._sync_desktop_overlay_interaction_mode_from_settings(settings)
        self._save_settings()
        await self._broadcast_desktop_runtime_control_payloads(desktop_runtime_controls)
        await self._sync_clipboard_watcher()
        self._refresh_local_stt_runtime_state()
        self._clear_local_stt_pending_enable_if_provider_switched_away()

        # low_latency_mode 변경 시 Qwen LLM 프로바이더 재생성 필요
        # (AsyncQwenLLMProvider vs QwenLLMProvider 전환)
        if (
            prev_low_latency is not None
            and prev_low_latency != settings.stt.low_latency_mode
            and self.settings.provider.llm.value == "qwen"
        ):
            self.log_detailed(
                "[Settings] Low latency detail: "
                f"mode={prev_low_latency}->{settings.stt.low_latency_mode} rebuilding_llm_provider=True"
            )
            await self._rebuild_llm_provider()

        if self.hub is not None:
            self.hub.source_language = settings.languages.source_language
            self.hub.target_language = settings.languages.target_language
            self.hub.peer_source_language = settings.languages.peer_source_language
            self.hub.peer_target_language = settings.languages.peer_target_language
            self.hub.system_prompt = settings.system_prompt
            self.hub.low_latency_mode = settings.stt.low_latency_mode
            self.hub.low_latency_merge_gap_ms = settings.stt.low_latency_merge_gap_ms
            self.hub.low_latency_spec_retry_max = settings.stt.low_latency_spec_retry_max
            self.hub.hangover_s = (
                settings.stt.low_latency_vad_hangover_ms / 1000.0
                if settings.stt.low_latency_mode
                else DEFAULT_STABLE_VAD_HANGOVER_MS / 1000.0
            )
            self.hub.peer_hangover_s = settings.desktop_audio.vad_hangover_ms / 1000.0
            self.hub.chatbox_include_source = settings.osc.chatbox_include_source
            self._sync_effective_hub_flags(settings)

            async def _clear_language_runtime_state(channel: str) -> None:
                try:
                    await self.hub.clear_language_runtime_state(channel=channel)
                except Exception as exc:
                    self._log_error(f"Failed to clear language runtime state for {channel}: {exc}")

            if source_language_changed or target_language_changed:
                await _clear_language_runtime_state("self")
            if effective_peer_source_changed or effective_peer_target_changed:
                await _clear_language_runtime_state("peer")

        presenter = self._overlay_presenter
        if presenter is not None:
            await presenter.update_display_preferences(
                show_translation=settings.overlay.show_translation,
                show_peer_original=settings.overlay.show_peer_original,
            )

        if prev_overlay_enabled != settings.ui.overlay_enabled:
            await self.set_overlay_enabled(settings.ui.overlay_enabled)

        if self._last_vrc_mic_sync_enabled != settings.osc.vrc_mic_intercept:
            if self.vrc_mic_audio_gate is not None:
                self.vrc_mic_audio_gate.set_enabled(settings.osc.vrc_mic_intercept)
            self.log_detailed(f"[Settings] VRC mic sync enabled: {settings.osc.vrc_mic_intercept}")
            await self._configure_vrc_mic_receiver(enabled=settings.osc.vrc_mic_intercept)

        current_self_signature = self._build_self_stt_runtime_signature(settings)
        current_peer_signature = self._build_peer_stt_runtime_signature(settings)
        next_peer_activation_requested = self._peer_translation_activation_requested_for(settings)
        should_restart_stt = (
            prev_self_signature is not None and current_self_signature != prev_self_signature
        )
        should_refresh_peer = (
            prev_peer_signature is None
            or current_peer_signature != prev_peer_signature
            or prev_peer_translation_enabled != settings.ui.peer_translation_enabled
            or prev_peer_activation_requested != next_peer_activation_requested
        )

        self._sync_signature_caches(settings)

        if source_language_changed or target_language_changed:
            self.log_detailed(
                "[Settings] Language runtime impact: "
                f"should_restart_stt={should_restart_stt} "
                f"should_refresh_peer={should_refresh_peer} "
                f"prev_overlay_enabled={prev_overlay_enabled} "
                f"next_overlay_enabled={settings.ui.overlay_enabled}"
            )

        if should_refresh_peer and self.hub is not None:
            await self._refresh_peer_stt_runtime()
            self._sync_effective_hub_flags(settings)

        if should_restart_stt:
            await self._replace_runtime_stt_provider()

        if source_language_changed or target_language_changed:
            view_settings = getattr(self.app, "view_settings", None)
            if view_settings is not None:
                with contextlib.suppress(Exception):
                    view_settings.load_from_settings(
                        settings,
                        config_path=self.config_path,
                        preserve_custom_vocab_draft=True,
                    )

        if prev_locale != settings.ui.locale:
            set_locale(settings.ui.locale)
            apply_locale = getattr(self.app, "apply_locale", None)
            if callable(apply_locale):
                try:
                    apply_locale()
                except Exception as exc:
                    self._log_error(f"Failed to apply locale: {exc}")

        self._refresh_overlay_peer_consumers()

    async def verify_api_key(self, provider: str, key: str) -> tuple[bool, str]:
        """Verify API key using the respective provider's static check. Returns (success, error_msg)."""
        if not key:
            return False, "API Key is empty"

        try:
            success = False
            if provider == "google":
                success = await GeminiLLMProvider.verify_api_key(
                    key,
                    model=self.settings.gemini.llm_model.value,
                )
            elif provider == "openrouter":
                success = await OpenRouterLLMProvider.verify_api_key(key)
            elif provider == "deepseek":
                success = await DeepSeekLLMProvider.verify_api_key(key)
            elif provider == "cerebras":
                success = await CerebrasLLMProvider.verify_api_key(key)
            elif provider == "alibaba_beijing":
                return await self._verify_qwen_key_with_model_fallback(
                    key,
                    base_url="https://dashscope.aliyuncs.com/api/v1",
                )
            elif provider == "alibaba_singapore":
                return await self._verify_qwen_key_with_model_fallback(
                    key,
                    base_url="https://dashscope-intl.aliyuncs.com/api/v1",
                )
            elif provider == "deepgram":
                success = await DeepgramRealtimeSTTBackend.verify_api_key(key)
            elif provider == "soniox":
                success = await SonioxRealtimeSTTBackend.verify_api_key(key)
            else:
                return False, f"Unknown provider: {provider}"

            if success:
                return True, "Verification successful"
            else:
                return False, "Verification failed (check logs/console for details)"
        except Exception as exc:
            msg = f"Verification error for {provider}: {exc}"
            self._log_error(msg)
            return False, str(exc)

    async def apply_providers(
        self,
        settings: AppSettings | None = None,
        *,
        force_rebuild_llm: bool = False,
    ) -> None:
        next_settings = (
            self.settings
            if settings is None
            else self.merge_settings_tab_apply_with_current_languages(settings)
        )
        if next_settings is None:
            return

        await self._preserve_github_star_prompt_observation_before_settings_replace(next_settings)

        prev_settings = self.settings
        prev_self_provider_signature = self._last_self_stt_provider_signature
        prev_peer_provider_signature = self._last_peer_stt_provider_signature
        prev_llm_provider_signature = self._last_llm_provider_signature

        if prev_settings is not None:
            if prev_self_provider_signature is None:
                prev_self_provider_signature = self._build_self_stt_provider_signature(
                    prev_settings
                )
            if prev_peer_provider_signature is None:
                prev_peer_provider_signature = self._build_peer_stt_provider_signature(
                    prev_settings
                )
            if prev_llm_provider_signature is None:
                prev_llm_provider_signature = self._build_llm_provider_signature(prev_settings)

        next_self_provider_signature = self._build_self_stt_provider_signature(next_settings)
        next_peer_provider_signature = self._build_peer_stt_provider_signature(next_settings)
        next_llm_provider_signature = self._build_llm_provider_signature(next_settings)

        should_rebuild_llm = force_rebuild_llm or (
            prev_llm_provider_signature is None
            or next_llm_provider_signature != prev_llm_provider_signature
        )
        should_refresh_peer = (
            prev_peer_provider_signature is None
            or next_peer_provider_signature != prev_peer_provider_signature
        )
        should_refresh_self_stt = (
            prev_self_provider_signature is None
            or next_self_provider_signature != prev_self_provider_signature
        )

        self.settings = next_settings
        self._save_settings()
        self._clear_local_stt_pending_enable_if_provider_switched_away()
        self._sync_local_stt_notice()
        if (
            next_settings.provider.llm != LLMProviderName.OPENROUTER
            or next_settings.openrouter.selected_source != OpenRouterCredentialSource.MANAGED
        ):
            self._set_managed_trial_pending_auth(False)
        else:
            self._sync_managed_auth_dashboard_notice()

        if self.hub is not None:
            self.hub.source_language = next_settings.languages.source_language
            self.hub.target_language = next_settings.languages.target_language
            self.hub.peer_source_language = next_settings.languages.peer_source_language
            self.hub.peer_target_language = next_settings.languages.peer_target_language
            self.hub.system_prompt = next_settings.system_prompt
            self.hub.low_latency_mode = next_settings.stt.low_latency_mode
            self.hub.low_latency_merge_gap_ms = next_settings.stt.low_latency_merge_gap_ms
            self.hub.low_latency_spec_retry_max = next_settings.stt.low_latency_spec_retry_max
            self.hub.hangover_s = (
                next_settings.stt.low_latency_vad_hangover_ms / 1000.0
                if next_settings.stt.low_latency_mode
                else DEFAULT_STABLE_VAD_HANGOVER_MS / 1000.0
            )
            self.hub.peer_hangover_s = next_settings.desktop_audio.vad_hangover_ms / 1000.0
            self.hub.chatbox_include_source = next_settings.osc.chatbox_include_source
            self._sync_effective_hub_flags(next_settings)

        if should_rebuild_llm:
            await self._rebuild_llm_provider()

        if should_refresh_peer:
            await self._refresh_peer_stt_runtime()
            self._sync_effective_hub_flags(next_settings)
            self._refresh_overlay_peer_consumers()

        if should_refresh_self_stt:
            if self._stt_desired:
                await self._replace_runtime_stt_provider()
            else:
                await self._rebuild_stt_provider()

        self._sync_signature_caches(next_settings)

    def _load_or_init_settings(self, path: Path) -> AppSettings:
        if path.exists():
            return load_settings(path)
        settings = new_settings_for_first_run()
        path.parent.mkdir(parents=True, exist_ok=True)
        save_settings(path, settings)
        return settings

    async def _rebuild_llm_provider(self) -> None:
        """Rebuild only the LLM provider without tearing down the entire pipeline."""
        if self.hub is None or self.settings is None:
            return

        # Close existing LLM provider
        previous_llm = self.hub.llm
        self.hub.llm = None
        if previous_llm is not None:
            with contextlib.suppress(Exception):
                await previous_llm.close()

        # Create new LLM provider with current settings
        llm = None
        llm_error: Exception | None = None
        try:
            secrets = create_secret_store(self.settings.secrets, config_path=self.config_path)
            new_managed_release_service = self._create_managed_openrouter_release_service(
                secrets=secrets
            )
            await self._replace_managed_openrouter_release_service(new_managed_release_service)
            llm = create_llm_provider(
                self.settings,
                secrets=secrets,
                managed_release_service=self._managed_openrouter_release_service,
                managed_delegate_ready=self._on_managed_trial_delegate_ready,
                runtime_logging=self.runtime_logging,
            )
        except Exception as exc:
            llm_error = exc

        # Update hub's LLM provider
        self.hub.llm = llm

        # Update dashboard status
        dash = getattr(self.app, "view_dashboard", None)
        if dash is not None:
            dash.set_translation_needs_key(
                (llm is None) and self._llm_provider_requires_secret(self.settings.provider.llm)
            )

        await self._refresh_managed_trial_usage_state_best_effort()

        if llm is None:
            message = "LLM provider not available"
            if llm_error is not None:
                message = f"{message}: {llm_error}"
            self._log_error(message)
            return

        self.log_basic("[Settings] LLM provider rebuilt successfully")

    async def _rebuild_stt_provider(self) -> None:
        """Rebuild only the STT provider so later enable uses current settings."""
        if self.hub is None or self.settings is None:
            return

        stt = None
        stt_error: Exception | None = None
        try:
            secrets = create_secret_store(self.settings.secrets, config_path=self.config_path)
            backend = create_stt_backend(
                self.settings,
                secrets=secrets,
                diagnostics_enabled=self._detailed_audio_diag_enabled,
            )
            stt = ManagedSTTProvider(
                backend=backend,
                sample_rate_hz=self.settings.audio.internal_sample_rate_hz,
                stt_provider_name=self.settings.provider.stt,
                clock=self.clock,
                reset_deadline_s=STT_RESET_DEADLINE_S,
                drain_timeout_s=self.settings.stt.drain_timeout_s,
                bridging_ms=self.settings.audio.ring_buffer_ms,
                on_final_transcript_suppressed=self._on_final_transcript_suppressed,
                runtime_logging=self.runtime_logging,
                stt_input_fault_profile_provider=lambda: (
                    self._debug_stt_fault_profile if self._debug_audio_fault_allowed() else "none"
                ),
            )
        except Exception as exc:
            stt_error = exc

        await self.hub.replace_stt_provider(stt)
        self._sync_effective_hub_flags(self.settings)

        dash = getattr(self.app, "view_dashboard", None)
        if dash is not None:
            dash.set_stt_needs_key(self._dashboard_stt_needs_key(stt_available=stt is not None))
            if stt is None:
                dash.set_stt_enabled(False)

        if stt is None:
            assert stt_error is not None
            self._log_error(f"STT backend not available: {stt_error}")
            return

        self.log_basic("[Settings] STT provider replacement completed successfully")

    def _create_peer_stt_provider_from_runtime_config(
        self,
        config: PeerRuntimeConfig,
        on_terminal_failure,
    ) -> ManagedSTTProvider:
        assert self.settings is not None
        secrets = create_secret_store(self.settings.secrets, config_path=self.config_path)
        peer_backend = create_peer_stt_backend(
            self.settings,
            secrets=secrets,
            diagnostics_enabled=self._detailed_audio_diag_enabled,
        )
        return ManagedSTTProvider(
            backend=peer_backend,
            sample_rate_hz=config.backend.sample_rate_hz,
            stt_provider_name=config.backend.provider,
            channel="peer",
            clock=self.clock,
            reset_deadline_s=STT_RESET_DEADLINE_S,
            drain_timeout_s=self.settings.stt.drain_timeout_s,
            bridging_ms=max(1, config.vad_pre_roll_ms),
            on_terminal_failure=on_terminal_failure,
            on_final_transcript_suppressed=self._on_final_transcript_suppressed,
            runtime_logging=self.runtime_logging,
            stt_input_fault_profile_provider=lambda: (
                self._debug_stt_fault_profile if self._debug_audio_fault_allowed() else "none"
            ),
        )

    def _create_peer_audio_source_from_runtime_config(self, config: PeerRuntimeConfig):
        raw_source = DesktopLoopbackAudioSource(device_name=config.output_device)
        self.log_detailed(
            "[AudioDiag][Loopback][peer] "
            f"requested_device={config.output_device!r} "
            f"resolved_device_name={getattr(raw_source, 'resolved_device_name', None)!r} "
            f"resolved_device_index={getattr(raw_source, 'resolved_device_index', None)} "
            f"resolved_channels={getattr(raw_source, 'resolved_channels', None)} "
            f"actual_sample_rate_hz={getattr(raw_source, 'actual_sample_rate_hz', None)} "
            f"used_default_fallback={getattr(raw_source, 'used_default_fallback', None)}"
        )
        wrapped_source = self._wrap_diagnostic_audio_source(raw_source, channel_label="peer")
        return DesktopPeerPipeline(
            source=wrapped_source,
            target_sample_rate_hz=config.backend.sample_rate_hz,
            is_detailed_enabled=self._detailed_audio_diag_enabled,
            log_detailed=lambda message: self.log_detailed(message),
        )

    @property
    def debug_capture_fault_profile(self) -> str:
        return self._debug_capture_fault_profile

    @property
    def debug_stt_fault_profile(self) -> str:
        return self._debug_stt_fault_profile

    def _debug_audio_fault_allowed(self) -> bool:
        return bool(getattr(self.app, "debug_ui_preview", False))

    def _detailed_audio_diag_enabled(self) -> bool:
        return self.runtime_logging.mode is SessionLoggingMode.DETAILED

    def _on_final_transcript_suppressed(
        self,
        notification: FinalTranscriptSuppressedNotification,
    ) -> None:
        self.log_detailed(
            "[STT][SuppressedFinalNotification] "
            f"provider={notification.stt_provider_name.value} "
            f"channel={notification.channel} "
            f"utterance_id={str(notification.utterance_id)[:8]}"
        )
        if notification.stt_provider_name is STTProviderName.LOCAL_QWEN:
            self._record_local_qwen_hallucination_guidance_detection(notification)

    def _record_local_qwen_hallucination_guidance_detection(
        self,
        notification: FinalTranscriptSuppressedNotification,
    ) -> None:
        self._local_qwen_hallucination_detection_count += 1
        count = self._local_qwen_hallucination_detection_count
        self.log_detailed(
            "[STT][SuppressedFinalNotification] "
            f"local_qwen_guidance count={count} "
            f"channel={notification.channel} "
            f"modal_shown={self._local_qwen_hallucination_modal_shown}"
        )
        if count < LOCAL_QWEN_HALLUCINATION_GUIDANCE_TRIGGER_COUNT:
            return
        if self._local_qwen_hallucination_modal_shown:
            return

        show_dialog = getattr(self.app, "show_local_qwen_hallucination_dialog", None)
        if not callable(show_dialog):
            self.log_detailed(
                "[STT][SuppressedFinalNotification] "
                f"local_qwen_guidance count={count} guidance_modal=unavailable"
            )
            return

        self._local_qwen_hallucination_modal_shown = True
        show_dialog()

    def cycle_debug_capture_fault_profile(self) -> str:
        if not self._debug_audio_fault_allowed():
            return "none"

        from puripuly_heart.core.audio.diagnostics import (
            EXPECTED_FAULT_SIGNATURES,
            AudioFaultProfile,
        )

        profiles = [
            AudioFaultProfile.NONE,
            AudioFaultProfile.CAPTURE_SILENT_FIRST_CHANNEL,
            AudioFaultProfile.CAPTURE_ATTENUATE_40DB,
            AudioFaultProfile.CAPTURE_NEAR_SILENCE_NOISE,
            AudioFaultProfile.CAPTURE_BUFFER_DROPOUTS,
        ]
        current = AudioFaultProfile(self._debug_capture_fault_profile)
        next_profile = profiles[(profiles.index(current) + 1) % len(profiles)]
        self._debug_capture_fault_profile = next_profile.value
        self.log_detailed(
            "[AudioDiag][DebugFault] "
            f"capture_profile={next_profile.value} "
            "expected_signature="
            f"{EXPECTED_FAULT_SIGNATURES.get(next_profile.value, 'none')}"
        )
        return self._debug_capture_fault_profile

    def cycle_debug_stt_fault_profile(self) -> str:
        if not self._debug_audio_fault_allowed():
            return "none"

        from puripuly_heart.core.audio.diagnostics import (
            EXPECTED_FAULT_SIGNATURES,
            AudioFaultProfile,
        )

        profiles = [AudioFaultProfile.NONE, AudioFaultProfile.STT_INPUT_LOW_SNR_VAD_PASS]
        current = AudioFaultProfile(self._debug_stt_fault_profile)
        next_profile = profiles[(profiles.index(current) + 1) % len(profiles)]
        self._debug_stt_fault_profile = next_profile.value
        self.log_detailed(
            "[AudioDiag][DebugFault] "
            f"stt_profile={next_profile.value} "
            "expected_signature="
            f"{EXPECTED_FAULT_SIGNATURES.get(next_profile.value, 'none')}"
        )
        return self._debug_stt_fault_profile

    def clear_debug_audio_fault_profiles(self) -> None:
        self._debug_capture_fault_profile = "none"
        self._debug_stt_fault_profile = "none"
        self.log_detailed("[AudioDiag][DebugFault] capture_profile=none stt_profile=none")

    def _wrap_diagnostic_audio_source(
        self,
        source: AudioSource,
        *,
        channel_label: str,
    ) -> AudioSource:
        from puripuly_heart.core.audio.diagnostics import AudioFaultProfile, DiagnosticAudioSource

        def extra_fields() -> dict[str, object]:
            return {
                "queue_drops": getattr(source, "queue_drop_count", 0),
                "callback_statuses": getattr(source, "callback_status_count", 0),
                "last_callback_status": getattr(source, "last_callback_status", None),
                "resolved_device_name": getattr(source, "resolved_device_name", None),
                "resolved_device_index": getattr(source, "resolved_device_index", None),
                "resolved_channels": getattr(source, "resolved_channels", None),
                "actual_sample_rate_hz": getattr(source, "actual_sample_rate_hz", None),
                "used_default_fallback": getattr(source, "used_default_fallback", None),
            }

        return DiagnosticAudioSource(
            source=source,
            channel_label=channel_label,
            is_detailed_enabled=self._detailed_audio_diag_enabled,
            log_detailed=lambda message: self.log_detailed(message),
            fault_profile_provider=lambda: (
                self._debug_capture_fault_profile
                if self._debug_audio_fault_allowed()
                else AudioFaultProfile.NONE.value
            ),
            extra_fields_provider=extra_fields,
        )

    def _create_peer_vad_from_runtime_config(self, config: PeerRuntimeConfig, model_path: Path):
        return create_peer_vad_gating(
            engine=SileroVadOnnx(model_path=model_path),
            sample_rate_hz=config.backend.sample_rate_hz,
            ring_buffer_ms=config.vad_pre_roll_ms,
            speech_threshold=config.vad_threshold,
            hangover_ms=config.vad_hangover_ms,
            diagnostic_event_callback=lambda message: self.log_detailed(message),
            diagnostics_enabled=self._detailed_audio_diag_enabled,
            diagnostic_label="peer",
        )

    async def _run_peer_audio_vad_loop(self, **kwargs: object) -> None:
        from puripuly_heart.app.headless_mic import run_audio_vad_loop

        await run_audio_vad_loop(
            **kwargs,
            channel_label="peer",
            is_detailed_enabled=self._detailed_audio_diag_enabled,
            log_detailed=lambda message: self.log_detailed(message),
        )

    async def _refresh_peer_stt_runtime(self) -> None:
        if self.settings is None or self.hub is None or self._peer_runtime is None:
            return

        config = self._build_peer_runtime_config(self.settings)
        desired_active = self._peer_runtime_should_be_active(self.settings)
        if desired_active and not await self._ensure_peer_local_stt_ready():
            desired_active = False
        await self._peer_runtime.apply_policy(config=config, desired_active=desired_active)
        self._last_peer_stt_runtime_signature = config.runtime_signature
        self._sync_effective_hub_flags(self.settings)

    async def _rebuild_pipeline(self, *, rebuild_stt: bool) -> None:
        self.log_detailed(
            f"[Settings] Rebuilding pipeline detail: rebuild_stt={rebuild_stt} overlay_state={self.overlay_state}"
        )
        _ = rebuild_stt
        restore_stt_enabled = self._stt_desired
        if self._bridge_task:
            self._bridge_task.cancel()
            await asyncio.gather(self._bridge_task, return_exceptions=True)
            self._bridge_task = None

        peer_runtime = self._peer_runtime
        if peer_runtime is not None:
            with contextlib.suppress(Exception):
                await peer_runtime.close()
            self._peer_runtime = None

        await self.set_stt_enabled(False)
        await self._configure_vrc_mic_receiver(enabled=False)
        if self.hub is not None:
            with contextlib.suppress(Exception):
                await self.hub.stop()
        if self.sender is not None:
            with contextlib.suppress(Exception):
                self.sender.close()
        self.sender = None
        self.osc = None
        self.hub = None
        await self._init_pipeline()
        assert self.hub is not None
        presenter = self._overlay_presenter
        if presenter is not None:
            self.hub.overlay_sink = presenter

        dash = getattr(self.app, "view_dashboard", None)
        if dash is not None:
            dash.set_translation_needs_key(
                (self.hub.llm is None)
                and self._llm_provider_requires_secret(self.settings.provider.llm)
            )
            dash.set_stt_needs_key(
                self._dashboard_stt_needs_key(stt_available=self.hub.stt is not None)
            )

            self.hub.translation_enabled = (
                bool(getattr(dash, "is_translation_on", True)) and self.hub.llm is not None
            )
            dash.set_translation_enabled(self.hub.translation_enabled)

        await self.hub.start(auto_flush_osc=True)

        bridge = UIEventBridge(
            app=self.app,
            event_queue=self.hub.ui_events,
            runtime_logging=self.runtime_logging,
        )
        self._bridge_task = asyncio.create_task(bridge.run())

        if self.overlay_state == "connected" and presenter is not None:
            await self._refresh_overlay_runtime_dependencies()

        if restore_stt_enabled:
            await self.set_stt_enabled(True)

        # Trigger background verification to sync button colors
        asyncio.create_task(self._verify_and_update_status())

    async def _init_pipeline(self) -> None:
        assert self.settings is not None
        self._sync_signature_caches(self.settings)
        secrets = create_secret_store(self.settings.secrets, config_path=self.config_path)
        new_managed_release_service = self._create_managed_openrouter_release_service(
            secrets=secrets
        )
        await self._replace_managed_openrouter_release_service(new_managed_release_service)

        llm = None
        with contextlib.suppress(Exception):
            llm = create_llm_provider(
                self.settings,
                secrets=secrets,
                managed_release_service=self._managed_openrouter_release_service,
                managed_delegate_ready=self._on_managed_trial_delegate_ready,
                runtime_logging=self.runtime_logging,
            )

        stt = None
        try:
            backend = create_stt_backend(
                self.settings,
                secrets=secrets,
                diagnostics_enabled=self._detailed_audio_diag_enabled,
            )
            stt = ManagedSTTProvider(
                backend=backend,
                sample_rate_hz=self.settings.audio.internal_sample_rate_hz,
                stt_provider_name=self.settings.provider.stt,
                clock=self.clock,
                reset_deadline_s=STT_RESET_DEADLINE_S,
                drain_timeout_s=self.settings.stt.drain_timeout_s,
                bridging_ms=self.settings.audio.ring_buffer_ms,
                on_final_transcript_suppressed=self._on_final_transcript_suppressed,
                runtime_logging=self.runtime_logging,
                stt_input_fault_profile_provider=lambda: (
                    self._debug_stt_fault_profile if self._debug_audio_fault_allowed() else "none"
                ),
            )
        except Exception as exc:
            self._log_error(f"STT backend not available: {exc}")

        sender = VrchatOscUdpSender(
            host=self.settings.osc.host,
            port=self.settings.osc.port,
            chatbox_address=self.settings.osc.chatbox_address,
            chatbox_send=self.settings.osc.chatbox_send,
            chatbox_clear=self.settings.osc.chatbox_clear,
        )
        osc = ChatboxPaginator(
            sender=sender,
            clock=self.clock,
            max_chars=self.settings.osc.chatbox_max_chars,
            runtime_logging=self.runtime_logging,
        )

        hub = ClientHub(
            stt=stt,
            llm=llm,
            osc=osc,
            peer_stt=None,
            clock=self.clock,
            runtime_logging=self.runtime_logging,
            source_language=self.settings.languages.source_language,
            target_language=self.settings.languages.target_language,
            peer_source_language=self.settings.languages.peer_source_language,
            peer_target_language=self.settings.languages.peer_target_language,
            system_prompt=self.settings.system_prompt,
            chatbox_include_source=self.settings.osc.chatbox_include_source,
            fallback_transcript_only=True,
            translation_enabled=True,
            peer_translation_enabled=False,
            integrated_context_enabled=False,
            low_latency_mode=self.settings.stt.low_latency_mode,
            low_latency_merge_gap_ms=self.settings.stt.low_latency_merge_gap_ms,
            low_latency_spec_retry_max=self.settings.stt.low_latency_spec_retry_max,
            hangover_s=(
                self.settings.stt.low_latency_vad_hangover_ms / 1000.0
                if self.settings.stt.low_latency_mode
                else DEFAULT_STABLE_VAD_HANGOVER_MS / 1000.0
            ),
            peer_hangover_s=self.settings.desktop_audio.vad_hangover_ms / 1000.0,
        )

        if self.vrc_mic_state is None:
            self.vrc_mic_state = VrcMicState()
        if self.vrc_mic_audio_gate is None:
            self.vrc_mic_audio_gate = VrcMicAudioGate(
                state=self.vrc_mic_state,
                enabled=self.settings.osc.vrc_mic_intercept,
            )
        else:
            self.vrc_mic_audio_gate.state = self.vrc_mic_state
            self.vrc_mic_audio_gate.set_enabled(self.settings.osc.vrc_mic_intercept)
        self.vrc_mic_audio_gate.set_receiver_active(self.receiver is not None)
        self.vrc_mic_audio_gate.reset()

        self.sender = sender
        self.osc = osc
        self.hub = hub

        self._peer_runtime = PeerChannelRuntime(
            hub=hub,
            clock=self.clock,
            stt_factory=self._create_peer_stt_provider_from_runtime_config,
            source_factory=self._create_peer_audio_source_from_runtime_config,
            vad_factory=self._create_peer_vad_from_runtime_config,
            vad_model_resolver=ensure_silero_vad_onnx,
            run_audio_loop=self._run_peer_audio_vad_loop,
        )
        self._last_peer_translation_enabled = self.settings.ui.peer_translation_enabled
        await self._configure_vrc_mic_receiver(enabled=self.settings.osc.vrc_mic_intercept)

    async def _replace_managed_openrouter_release_service(
        self,
        service: ManagedOpenRouterReleaseService | None,
    ) -> None:
        previous = self._managed_openrouter_release_service
        self._managed_openrouter_release_service = service
        if previous is not None and previous is not service:
            with contextlib.suppress(Exception):
                await previous.close()

    def _create_managed_openrouter_release_service(
        self, *, secrets
    ) -> ManagedOpenRouterReleaseService | None:
        if self.settings is None:
            return None
        if self.settings.provider.llm != LLMProviderName.OPENROUTER:
            return None
        if self.settings.openrouter.selected_source != OpenRouterCredentialSource.MANAGED:
            return None

        from puripuly_heart import __version__

        try:
            client = HttpManagedOpenRouterBrokerClient(
                base_url=self.settings.openrouter.broker_base_url,
            )
        except ValueError as exc:
            logger.warning(
                "[Managed OpenRouter] Invalid broker base URL %r; using unavailable fallback: %s",
                self.settings.openrouter.broker_base_url,
                exc,
            )
            client = UnavailableManagedOpenRouterReleaseClient()

        return ManagedOpenRouterReleaseService(
            settings=self.settings,
            secrets=secrets,
            client=client,
            persist_settings=lambda updated: save_settings(self.config_path, updated),
            raw_hardware_fingerprint_provider=get_raw_hardware_fingerprint,
            app_version=__version__,
            on_discord_callback_received=self._on_discord_managed_auth_callback_received,
        )

    def _on_discord_managed_auth_callback_received(self) -> None:
        hook = self._discord_managed_auth_callback_received_hook
        if callable(hook):
            hook()

    @property
    def microphone_test_meter_level(self) -> float:
        return self._microphone_test_meter_level

    @property
    def microphone_test_active(self) -> bool:
        task = self._microphone_test_task
        return task is not None and not task.done()

    def _get_microphone_test_lifecycle_lock(self) -> asyncio.Lock:
        if self._microphone_test_lifecycle_lock is None:
            self._microphone_test_lifecycle_lock = asyncio.Lock()
        return self._microphone_test_lifecycle_lock

    @staticmethod
    def _microphone_test_audio_settings_signature(
        settings: AppSettings | None,
    ) -> tuple[object, ...] | None:
        if settings is None:
            return None
        return (
            settings.audio.input_host_api,
            settings.audio.input_device,
            settings.audio.internal_sample_rate_hz,
            settings.audio.internal_channels,
        )

    def _self_stt_active_or_desired_for_microphone_test(self) -> bool:
        return bool(
            self._stt_desired
            or self._local_stt_pending_enable_after_install
            or self._mic_task is not None
            or self._audio_source is not None
        )

    def _log_microphone_test_stt_auto_off(
        self,
        *,
        requested: bool,
        completed: bool,
        exception: BaseException | None = None,
    ) -> None:
        self.log_basic(
            "[MicTest] stt_auto_off "
            f"requested={requested} "
            f"completed={completed} "
            "exception_class="
            f"{_mic_test_log_value(type(exception).__name__ if exception else None)} "
            "exception_message="
            f"{_mic_test_log_value(str(exception) if exception else None)}"
        )

    async def _prepare_microphone_test_capture(self) -> bool:
        requested = self._self_stt_active_or_desired_for_microphone_test()
        if not requested:
            if self._last_mic_loop_close_exception is not None:
                self._log_microphone_test_stt_auto_off(
                    requested=False,
                    completed=False,
                    exception=self._last_mic_loop_close_exception,
                )
                return False
            self._log_microphone_test_stt_auto_off(
                requested=False,
                completed=True,
            )
            return True

        try:
            await self.set_stt_enabled(False)
            if self._last_mic_loop_close_exception is not None:
                raise self._last_mic_loop_close_exception
            if self._mic_task is not None or self._audio_source is not None:
                raise RuntimeError("self microphone source still open after STT auto-off")
        except Exception as exc:
            self._log_microphone_test_stt_auto_off(
                requested=True,
                completed=False,
                exception=exc,
            )
            return False

        self._log_microphone_test_stt_auto_off(
            requested=True,
            completed=True,
        )
        return True

    async def start_microphone_test(
        self,
        *,
        meter_callback: Callable[[float], object] | None = None,
        level_log_interval_s: float = _MICROPHONE_TEST_LEVEL_INTERVAL_S,
    ) -> bool:
        if self.settings is None:
            return False
        if self._last_microphone_test_audio_settings_signature is None:
            self._last_microphone_test_audio_settings_signature = (
                self._microphone_test_audio_settings_signature(self.settings)
            )
        async with self._get_microphone_test_lifecycle_lock():
            task = self._microphone_test_task
            if task is not None:
                if not task.done():
                    return False
                await asyncio.gather(task, return_exceptions=True)
                if self._microphone_test_task is task:
                    self._microphone_test_task = None

            if not await self._prepare_microphone_test_capture():
                return False

            self._microphone_test_task = asyncio.create_task(
                self._run_microphone_test_session(
                    meter_callback=meter_callback,
                    level_log_interval_s=level_log_interval_s,
                )
            )
            return True

    async def _run_microphone_test_session(
        self,
        *,
        meter_callback: Callable[[float], object] | None,
        level_log_interval_s: float,
    ) -> None:
        current_task = asyncio.current_task()
        try:
            await self.run_microphone_test_capture(
                meter_callback=meter_callback,
                level_log_interval_s=level_log_interval_s,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_error(f"Microphone test error: {exc}")
        finally:
            if self._microphone_test_task is current_task:
                self._microphone_test_task = None

    async def stop_microphone_test(self) -> None:
        async with self._get_microphone_test_lifecycle_lock():
            task = self._microphone_test_task
            if task is None:
                return
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if self._microphone_test_task is task:
                self._microphone_test_task = None

    async def stop_microphone_test_for_audio_settings_change(self) -> None:
        await self.stop_microphone_test()

    async def _set_microphone_test_meter_level(
        self,
        value: float,
        meter_callback: Callable[[float], object] | None,
    ) -> None:
        level = max(0.0, min(1.0, float(value)))
        if level <= 1e-6:
            level = 0.0
        self._microphone_test_meter_level = level
        if meter_callback is None:
            return
        try:
            result = meter_callback(level)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.debug("Microphone-test meter callback raised", exc_info=True)

    @staticmethod
    def _microphone_test_meter_level_from_frame(frame) -> float:  # noqa: ANN001
        samples = np.asarray(frame.samples, dtype=np.float32)
        if samples.size == 0:
            return 0.0
        peak_abs = float(np.max(np.abs(samples)))
        if peak_abs <= 1e-6:
            return 0.0
        return min(1.0, peak_abs)

    @staticmethod
    def _format_microphone_test_route_log(
        observation: MicrophoneTestRouteObservation,
    ) -> str:
        return (
            "[MicTest] route "
            f"saved_host_api={_mic_test_log_value(observation.saved_host_api)} "
            f"actual_host_api={_mic_test_log_value(observation.actual_host_api)} "
            f"requested_device={_mic_test_log_value(observation.requested_device)} "
            f"hostapi_index={_mic_test_log_value(observation.hostapi_index)} "
            f"resolved_device_idx={_mic_test_log_value(observation.resolved_device_idx)} "
            f"resolved_device_name={_mic_test_log_value(observation.resolved_device_name)} "
            "resolution_exception_class="
            f"{_mic_test_log_value(observation.resolution_exception_class)} "
            "resolution_exception_message="
            f"{_mic_test_log_value(observation.resolution_exception_message)}"
        )

    @staticmethod
    def _microphone_test_source_value(source: object | None, attr: str, fallback: object) -> object:
        if source is None:
            return fallback
        try:
            return getattr(source, attr, fallback)
        except Exception:
            return fallback

    @staticmethod
    def _microphone_test_source_int(
        source: object | None,
        attr: str,
        fallback: int | None,
    ) -> int | None:
        if source is None:
            return fallback
        try:
            value = getattr(source, attr, fallback)
            if value is None:
                return None
            return int(value)
        except Exception:
            return fallback

    def _log_microphone_test_open(
        self,
        *,
        attempted: bool,
        opened: bool,
        requested_channels: int | None,
        source: object | None,
        observation: MicrophoneTestRouteObservation,
        exception: BaseException | None = None,
    ) -> None:
        opened_channels = self._microphone_test_source_int(source, "opened_channels", None)
        frame_channels = self._microphone_test_source_int(source, "frame_channels", opened_channels)
        actual_sample_rate_hz = self._microphone_test_source_value(
            source,
            "actual_sample_rate_hz",
            None,
        )
        self.log_basic(
            "[MicTest] open "
            f"attempted={attempted} "
            f"opened={opened} "
            f"requested_channels={_mic_test_log_value(requested_channels)} "
            f"opened_channels={_mic_test_log_value(opened_channels)} "
            f"frame_channels={_mic_test_log_value(frame_channels)} "
            "requested_sample_rate_hz=None "
            f"actual_sample_rate_hz={_mic_test_log_value(actual_sample_rate_hz)} "
            f"wasapi_auto_convert={observation.wasapi_auto_convert} "
            f"wasapi_exclusive={observation.wasapi_exclusive} "
            "exception_class="
            f"{_mic_test_log_value(type(exception).__name__ if exception else None)} "
            "exception_message="
            f"{_mic_test_log_value(str(exception) if exception else None)}"
        )

    def _log_microphone_test_level(
        self,
        stats: _MicrophoneTestLevelStats,
        *,
        source: object | None,
    ) -> None:
        self.log_basic(
            "[MicTest] level "
            f"rms_db={stats.rms_db:.1f} "
            f"peak_db={stats.peak_db:.1f} "
            f"zero_ratio={stats.zero_ratio:.3f} "
            f"frames={stats.frames} "
            f"audio_ms={stats.audio_ms:.1f} "
            f"queue_drops={self._microphone_test_source_int(source, 'queue_drop_count', 0)} "
            "callback_statuses="
            f"{self._microphone_test_source_int(source, 'callback_status_count', 0)}"
        )

    def _log_microphone_test_end(
        self,
        *,
        opened: bool,
        stats: _MicrophoneTestLevelStats,
        source: object | None,
        exception: BaseException | None,
    ) -> None:
        self.log_basic(
            "[MicTest] end "
            f"opened={opened} "
            f"frames_total={stats.frames} "
            f"audio_ms_total={stats.audio_ms:.1f} "
            f"rms_db_total={stats.rms_db:.1f} "
            f"peak_db_max={stats.peak_db:.1f} "
            f"zero_ratio_total={stats.zero_ratio:.3f} "
            f"queue_drops={self._microphone_test_source_int(source, 'queue_drop_count', 0)} "
            "callback_statuses="
            f"{self._microphone_test_source_int(source, 'callback_status_count', 0)} "
            "exception_class="
            f"{_mic_test_log_value(type(exception).__name__ if exception else None)} "
            "exception_message="
            f"{_mic_test_log_value(str(exception) if exception else None)}"
        )

    async def run_microphone_test_capture(
        self,
        *,
        meter_callback: Callable[[float], object] | None = None,
        level_log_interval_s: float = _MICROPHONE_TEST_LEVEL_INTERVAL_S,
    ) -> None:
        assert self.settings is not None
        level_log_interval_s = max(0.0, float(level_log_interval_s))
        source: object | None = None
        opened = False
        end_exception: BaseException | None = None
        level_logged = False
        pending_frame: asyncio.Task[object] | None = None
        interval_stats = _MicrophoneTestLevelStats()
        total_stats = _MicrophoneTestLevelStats()

        await self._set_microphone_test_meter_level(0.0, meter_callback)
        observation = observe_microphone_test_route(
            saved_host_api=self.settings.audio.input_host_api,
            requested_device=self.settings.audio.input_device,
        )
        self.log_basic(self._format_microphone_test_route_log(observation))

        try:
            if not observation.should_attempt_open:
                self._log_microphone_test_open(
                    attempted=False,
                    opened=False,
                    requested_channels=None,
                    source=None,
                    observation=observation,
                )
                self._log_microphone_test_level(interval_stats, source=None)
                level_logged = True
                return

            decision = determine_self_mic_capture_channels(
                device_idx=observation.resolved_device_idx,
                internal_channels=self.settings.audio.internal_channels,
            )
            requested_channels = decision.preferred_capture_channels
            try:
                source = SoundDeviceAudioSource(
                    sample_rate_hz=None,
                    channels=requested_channels,
                    device=observation.resolved_device_idx,
                    wasapi_auto_convert=observation.wasapi_auto_convert,
                    wasapi_exclusive=observation.wasapi_exclusive,
                )
            except Exception as exc:
                end_exception = exc
                self._log_microphone_test_open(
                    attempted=True,
                    opened=False,
                    requested_channels=requested_channels,
                    source=None,
                    observation=observation,
                    exception=exc,
                )
                self._log_microphone_test_level(interval_stats, source=None)
                level_logged = True
                return

            opened = True
            self._log_microphone_test_open(
                attempted=True,
                opened=True,
                requested_channels=requested_channels,
                source=source,
                observation=observation,
            )

            frame_iterator = source.frames()  # type: ignore[attr-defined]
            pending_frame = asyncio.create_task(anext(frame_iterator))
            last_level_log_s = self.clock.now()
            while True:
                if level_log_interval_s > 0.0:
                    elapsed_s = max(0.0, self.clock.now() - last_level_log_s)
                    timeout_s = max(0.0, level_log_interval_s - elapsed_s)
                    done, _pending = await asyncio.wait({pending_frame}, timeout=timeout_s)
                    if not done:
                        self._log_microphone_test_level(interval_stats, source=source)
                        level_logged = True
                        interval_stats.reset()
                        last_level_log_s = self.clock.now()
                        continue
                else:
                    await asyncio.wait({pending_frame})

                try:
                    frame = pending_frame.result()
                except StopAsyncIteration:
                    pending_frame = None
                    break

                interval_stats.add_frame(frame)
                total_stats.add_frame(frame)
                await self._set_microphone_test_meter_level(
                    self._microphone_test_meter_level_from_frame(frame),
                    meter_callback,
                )
                pending_frame = asyncio.create_task(anext(frame_iterator))

                if level_log_interval_s <= 0.0 or (
                    self.clock.now() - last_level_log_s >= level_log_interval_s
                ):
                    self._log_microphone_test_level(interval_stats, source=source)
                    level_logged = True
                    interval_stats.reset()
                    last_level_log_s = self.clock.now()
        except asyncio.CancelledError as exc:
            end_exception = exc
            raise
        except Exception as exc:
            end_exception = exc
        finally:
            if pending_frame is not None and not pending_frame.done():
                pending_frame.cancel()
                with contextlib.suppress(Exception):
                    await asyncio.gather(pending_frame, return_exceptions=True)

            if source is not None:
                with contextlib.suppress(Exception):
                    await source.close()  # type: ignore[attr-defined]

            if source is not None and interval_stats.frames > 0:
                self._log_microphone_test_level(interval_stats, source=source)
                level_logged = True
            elif source is not None and not level_logged:
                self._log_microphone_test_level(interval_stats, source=source)

            self._log_microphone_test_end(
                opened=opened,
                stats=total_stats,
                source=source,
                exception=end_exception,
            )
            await self._set_microphone_test_meter_level(0.0, meter_callback)

    async def _start_mic_loop(self) -> None:
        assert self.settings is not None
        assert self.hub is not None

        if self._mic_task is not None:
            return

        if self._audio_source is not None or self._last_mic_loop_close_exception is not None:
            await self._stop_mic_loop()
            if self._audio_source is not None or self._last_mic_loop_close_exception is not None:
                self.log_detailed(
                    "[STT] Skipping microphone start while previous microphone source close is pending",
                    level=logging.WARNING,
                    exception=self._last_mic_loop_close_exception,
                )
                return

        try:
            model_path = ensure_silero_vad_onnx()
        except Exception as exc:
            self._log_error(f"Failed to prepare Silero VAD model ({SILERO_VAD_VERSION}): {exc}")
            return

        if self._mic_task is None:
            vad = VadGating(
                engine=SileroVadOnnx(model_path=model_path),
                sample_rate_hz=self.settings.audio.internal_sample_rate_hz,
                ring_buffer_ms=self.settings.audio.ring_buffer_ms,
                speech_threshold=self.settings.stt.vad_speech_threshold,
                hangover_ms=(
                    self.settings.stt.low_latency_vad_hangover_ms
                    if self.settings.stt.low_latency_mode
                    else DEFAULT_STABLE_VAD_HANGOVER_MS
                ),
                diagnostic_event_callback=lambda message: self.log_detailed(message),
                diagnostics_enabled=self._detailed_audio_diag_enabled,
                diagnostic_label="self",
            )

            def _resolve_device(host_api: str, device: str) -> int | None:
                try:
                    return resolve_sounddevice_input_device(host_api=host_api, device=device)
                except Exception as exc:
                    self.log_detailed(
                        "[STT] Device resolution detail: "
                        f"host_api={host_api!r} device={device!r} error={exc}",
                        level=logging.WARNING,
                    )
                    return None

            def _source_int(source: SoundDeviceAudioSource, attr: str, fallback: int) -> int:
                try:
                    value = getattr(source, attr, fallback)
                    return int(value)
                except Exception:
                    return fallback

            def _log_mic_capture_format(
                *,
                attempt: str,
                dev_idx: int | None,
                requested_channels: int,
                decision: SelfMicCaptureChannelDecision,
                source: SoundDeviceAudioSource,
                host_api_for_log: str,
                device_for_log: str,
                wasapi_auto_convert: bool,
                wasapi_exclusive: bool,
            ) -> None:
                metadata = decision.metadata
                opened_channels = _source_int(source, "opened_channels", requested_channels)
                frame_channels = _source_int(source, "frame_channels", opened_channels)
                frame_channels_source = "opened_fallback"
                actual_sample_rate_hz = _source_int(source, "actual_sample_rate_hz", 0)
                self.log_detailed(
                    "[STT] Microphone capture format: "
                    f"attempt={attempt!r} "
                    f"internal_channels={decision.internal_channels} "
                    f"preferred_capture_channels={decision.preferred_capture_channels} "
                    f"requested_channels={requested_channels} "
                    f"opened_channels={opened_channels} "
                    f"frame_channels={frame_channels} "
                    f"frame_channels_source={frame_channels_source!r} "
                    f"saved_host_api={saved_host_api!r} "
                    f"actual_host_api={host_api_for_log!r} "
                    f"device={device_for_log!r} "
                    f"device_idx={dev_idx} "
                    f"wasapi_auto_convert={wasapi_auto_convert} "
                    f"wasapi_exclusive={wasapi_exclusive} "
                    f"actual_sample_rate_hz={actual_sample_rate_hz or None} "
                    f"metadata_device_idx={metadata.device_idx} "
                    f"metadata_device_name={metadata.name!r} "
                    f"device_max_input_channels={metadata.max_input_channels} "
                    f"device_default_samplerate={metadata.default_samplerate} "
                    f"metadata_status={metadata.metadata_status!r} "
                    f"metadata_error={metadata.metadata_error!r}"
                )

            def _open_source_once(
                dev_idx: int | None,
                *,
                attempt: str,
                requested_channels: int,
                decision: SelfMicCaptureChannelDecision,
                host_api_for_log: str,
                device_for_log: str,
                wasapi_auto_convert: bool = False,
                wasapi_exclusive: bool = False,
            ) -> SoundDeviceAudioSource:
                source = SoundDeviceAudioSource(
                    sample_rate_hz=None,
                    channels=requested_channels,
                    device=dev_idx,
                    wasapi_auto_convert=wasapi_auto_convert,
                    wasapi_exclusive=wasapi_exclusive,
                )
                _log_mic_capture_format(
                    attempt=attempt,
                    dev_idx=dev_idx,
                    requested_channels=requested_channels,
                    decision=decision,
                    source=source,
                    host_api_for_log=host_api_for_log,
                    device_for_log=device_for_log,
                    wasapi_auto_convert=wasapi_auto_convert,
                    wasapi_exclusive=wasapi_exclusive,
                )
                return source

            def _open_source_with_mono_retry(
                dev_idx: int | None,
                *,
                attempt: str,
                host_api_for_log: str,
                device_for_log: str,
                wasapi_auto_convert: bool = False,
                wasapi_exclusive: bool = False,
            ) -> SoundDeviceAudioSource:
                decision = determine_self_mic_capture_channels(
                    device_idx=dev_idx,
                    internal_channels=self.settings.audio.internal_channels,
                )
                try:
                    return _open_source_once(
                        dev_idx,
                        attempt=attempt,
                        requested_channels=decision.preferred_capture_channels,
                        decision=decision,
                        host_api_for_log=host_api_for_log,
                        device_for_log=device_for_log,
                        wasapi_auto_convert=wasapi_auto_convert,
                        wasapi_exclusive=wasapi_exclusive,
                    )
                except Exception as exc:
                    if decision.preferred_capture_channels <= self.settings.audio.internal_channels:
                        raise
                    self.log_detailed(
                        "[STT] Microphone open detail: "
                        f"attempt={attempt!r} "
                        f"host_api={host_api_for_log!r} "
                        f"device={device_for_log!r} "
                        f"device_idx={dev_idx} "
                        f"preferred_capture_channels={decision.preferred_capture_channels} "
                        f"requested_channels={decision.preferred_capture_channels} "
                        f"wasapi_auto_convert={wasapi_auto_convert} "
                        f"wasapi_exclusive={wasapi_exclusive} "
                        f"metadata_status={decision.metadata.metadata_status!r} "
                        f"will_retry_mono=True "
                        f"error={exc}",
                        level=logging.WARNING,
                    )
                    retry_attempt = f"{attempt}_mono_retry"
                    return _open_source_once(
                        dev_idx,
                        attempt=retry_attempt,
                        requested_channels=self.settings.audio.internal_channels,
                        decision=decision,
                        host_api_for_log=host_api_for_log,
                        device_for_log=device_for_log,
                        wasapi_auto_convert=wasapi_auto_convert,
                        wasapi_exclusive=wasapi_exclusive,
                    )

            saved_host_api = self.settings.audio.input_host_api
            host_api_profile = normalize_input_host_api(saved_host_api)
            host_api = host_api_profile.actual_host_api
            first_open_used_wasapi_flags = (
                host_api_profile.wasapi_auto_convert or host_api_profile.wasapi_exclusive
            )
            device_name = self.settings.audio.input_device

            # 1차 시도: 설정된 Host API + 마이크
            device_idx = _resolve_device(host_api, device_name)
            source: SoundDeviceAudioSource | None = None

            try:
                source = _open_source_with_mono_retry(
                    device_idx,
                    attempt="primary",
                    host_api_for_log=host_api,
                    device_for_log=device_name,
                    wasapi_auto_convert=host_api_profile.wasapi_auto_convert,
                    wasapi_exclusive=host_api_profile.wasapi_exclusive,
                )
                self.log_detailed(
                    "[STT] Microphone opened: "
                    f"saved_host_api={saved_host_api!r} "
                    f"actual_host_api={host_api!r} "
                    f"device={device_name!r} "
                    f"device_idx={device_idx} "
                    f"wasapi_auto_convert={host_api_profile.wasapi_auto_convert} "
                    f"wasapi_exclusive={host_api_profile.wasapi_exclusive}"
                )
            except Exception as exc:
                self.log_detailed(
                    "[STT] Microphone open detail: "
                    f"host_api={host_api!r} device={device_name!r} error={exc}",
                    level=logging.ERROR,
                )

            # 2차 시도: Host API 무시, 마이크 이름만
            if source is None and device_name:
                fallback_idx = _resolve_device("", device_name)
                if fallback_idx != device_idx or first_open_used_wasapi_flags:
                    try:
                        source = _open_source_with_mono_retry(
                            fallback_idx,
                            attempt="name_fallback",
                            host_api_for_log="",
                            device_for_log=device_name,
                            wasapi_auto_convert=False,
                            wasapi_exclusive=False,
                        )
                        self.log_detailed(
                            f"[STT] Microphone opened with fallback: device_idx={fallback_idx}"
                        )
                    except Exception as exc:
                        self.log_detailed(
                            f"[STT] Fallback microphone detail: error={exc}",
                            level=logging.ERROR,
                        )

            # 3차 시도: 시스템 기본 장치
            if source is None:
                try:
                    source = _open_source_with_mono_retry(
                        None,
                        attempt="system_default",
                        host_api_for_log="",
                        device_for_log="",
                        wasapi_auto_convert=False,
                        wasapi_exclusive=False,
                    )
                    self.log_detailed("[STT] Microphone opened with system default")
                except Exception as exc:
                    self.log_detailed(
                        f"[STT] System default microphone detail: error={exc}",
                        level=logging.ERROR,
                    )

            if source is None:
                self._log_error("All microphone attempts failed")
                return

            self._vad = vad
            self._audio_source = self._wrap_diagnostic_audio_source(source, channel_label="self")
            self._mic_task = asyncio.create_task(self._run_mic_loop())

    async def _stop_mic_loop(self) -> None:
        if self._mic_task is not None:
            self._mic_task.cancel()
            await asyncio.gather(self._mic_task, return_exceptions=True)
            self._mic_task = None

        if self._audio_source is not None:
            try:
                await self._audio_source.close()
            except Exception as exc:
                self._last_mic_loop_close_exception = exc
            else:
                self._last_mic_loop_close_exception = None
                self._audio_source = None
        self._vad = None
        if self.vrc_mic_audio_gate is not None:
            self.vrc_mic_audio_gate.reset()

    async def _run_mic_loop(self) -> None:
        assert self.hub is not None
        assert self._audio_source is not None
        assert self._vad is not None

        from puripuly_heart.app.headless_mic import run_audio_vad_loop

        try:
            await run_audio_vad_loop(
                source=self._audio_source,
                vad=self._vad,
                sink=_HubVadSink(hub=self.hub),
                target_sample_rate_hz=self.settings.audio.internal_sample_rate_hz,  # type: ignore[union-attr]
                audio_gate=self.vrc_mic_audio_gate,
                channel_label="self",
                is_detailed_enabled=self._detailed_audio_diag_enabled,
                log_detailed=lambda message: self.log_detailed(message),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_error(f"Mic loop error: {exc}")

    async def _configure_vrc_mic_receiver(self, *, enabled: bool) -> None:
        if self._vrc_receiver_lock is None:
            self._vrc_receiver_lock = asyncio.Lock()

        async with self._vrc_receiver_lock:
            self._last_vrc_mic_sync_enabled = enabled
            if self.vrc_mic_audio_gate is not None:
                self.vrc_mic_audio_gate.set_enabled(enabled)

            if not enabled:
                self._stop_vrc_mic_receiver()
                return

            if self.receiver is not None or self.vrc_mic_state is None:
                if self.vrc_mic_audio_gate is not None:
                    self.vrc_mic_audio_gate.set_receiver_active(self.receiver is not None)
                return

            receiver = VrcOscReceiver(
                state=self.vrc_mic_state,
                host=VRC_OSC_RECEIVER_HOST,
                port=VRC_OSC_RECEIVER_PORT,
            )
            try:
                await receiver.start()
            except OSError as exc:
                if self.vrc_mic_audio_gate is not None:
                    self.vrc_mic_audio_gate.set_receiver_active(False)
                self._log_error(
                    "VRChat mic sync receiver unavailable on "
                    f"{VRC_OSC_RECEIVER_HOST}:{VRC_OSC_RECEIVER_PORT}: {exc}"
                )
                return

            self.receiver = receiver
            if self.vrc_mic_audio_gate is not None:
                self.vrc_mic_audio_gate.set_receiver_active(True)
                self.vrc_mic_audio_gate.reset()

    def _stop_vrc_mic_receiver(self) -> None:
        if self.receiver is not None:
            with contextlib.suppress(Exception):
                self.receiver.stop()
            self.receiver = None
        if self.vrc_mic_audio_gate is not None:
            self.vrc_mic_audio_gate.set_receiver_active(False)

    def _create_openrouter_pkce_client(self) -> OpenRouterPKCEClient:
        return OpenRouterPKCEClient(callback_origin="http://localhost:3000")

    def reopen_openrouter_pkce_authorization_url(self) -> bool:
        if self._openrouter_pkce_client is None:
            return False
        return self._openrouter_pkce_client.reopen_authorization_url()

    async def connect_openrouter_via_pkce(
        self,
        *,
        target_settings: AppSettings,
        launch_source: str,
    ) -> bool:
        assert self.settings is not None
        selection_alias = target_settings.openrouter.selection_alias
        if selection_alias is None:
            raise ValueError("PKCE connection requires a BYOK OpenRouter alias")

        profile = profile_for_alias(selection_alias.value)
        if profile.openrouter_source != OpenRouterCredentialSource.BYOK.value:
            raise ValueError("PKCE connection requires a BYOK OpenRouter alias")
        if profile.openrouter_model is None:
            raise ValueError("PKCE connection requires a BYOK OpenRouter model")
        previous_settings = copy.deepcopy(self.settings)

        try:
            pkce_client = self._create_openrouter_pkce_client()
            self._openrouter_pkce_client = pkce_client
            try:
                result = await pkce_client.run_desktop_flow()
            finally:
                self._openrouter_pkce_client = None
        except Exception as exc:
            self._show_short_message("openrouter.pkce.failed")
            self._log_error(f"OpenRouter PKCE failed: {exc}")
            if launch_source == "letter":
                show_founder_letter_dialog = getattr(self.app, "show_founder_letter_dialog", None)
                if callable(show_founder_letter_dialog):
                    with contextlib.suppress(Exception):
                        show_founder_letter_dialog()
            return False

        try:
            if not await OpenRouterLLMProvider.verify_api_key(result.api_key):
                raise RuntimeError("OpenRouter PKCE key verification failed")
            secrets = create_secret_store(self.settings.secrets, config_path=self.config_path)
            previous_api_key = secrets.get(OPENROUTER_BYOK_API_KEY_SECRET)
            secrets.set(OPENROUTER_BYOK_API_KEY_SECRET, result.api_key)
            updated = copy.deepcopy(target_settings)
            updated.provider.llm = LLMProviderName.OPENROUTER
            updated.openrouter.selection_alias = OpenRouterSelectionAlias(profile.alias)
            updated.openrouter.selected_source = OpenRouterCredentialSource.BYOK
            updated.openrouter.llm_model = OpenRouterLLMModel(profile.openrouter_model)
            updated.api_key_verified.openrouter = True
            try:
                await self.apply_providers(updated, force_rebuild_llm=True)
                self.settings.api_key_verified.openrouter = True
                self._save_settings()
            except Exception:
                with contextlib.suppress(Exception):
                    if previous_api_key is None:
                        secrets.delete(OPENROUTER_BYOK_API_KEY_SECRET)
                    else:
                        secrets.set(OPENROUTER_BYOK_API_KEY_SECRET, previous_api_key)
                try:
                    await self.apply_providers(previous_settings, force_rebuild_llm=True)
                except Exception as rollback_exc:
                    self.settings = previous_settings
                    with contextlib.suppress(Exception):
                        self._save_settings()
                    self._log_error(f"OpenRouter PKCE rollback failed: {rollback_exc}")
                raise
        except Exception as exc:
            self._show_short_message("openrouter.pkce.failed")
            self._log_error(f"OpenRouter PKCE apply failed: {exc}")
            if launch_source == "letter":
                show_founder_letter_dialog = getattr(self.app, "show_founder_letter_dialog", None)
                if callable(show_founder_letter_dialog):
                    with contextlib.suppress(Exception):
                        show_founder_letter_dialog()
            return False
        return True

    def _save_settings(self) -> None:
        assert self.settings is not None
        try:
            save_settings(self.config_path, self.settings)
        except Exception as exc:
            self._log_error(f"Failed to save settings: {exc}")

    def _sync_ui_from_settings(self) -> None:
        settings = self.settings
        if settings is None:
            return

        # Dashboard language dropdowns are initialized by the view; set values if present.
        with contextlib.suppress(Exception):
            dash = getattr(self.app, "view_dashboard", None)
            if dash is not None:
                dash.set_languages_from_codes(
                    settings.languages.source_language,
                    settings.languages.target_language,
                    settings.languages.peer_source_language,
                    settings.languages.peer_target_language,
                )
                # Load recent languages from settings
                dash.set_recent_languages(
                    settings.languages.recent_source_languages,
                    settings.languages.recent_target_languages,
                )
                # Connect callback for persistence
                dash.on_recent_languages_change = self._on_recent_languages_change

        with contextlib.suppress(Exception):
            view_settings = getattr(self.app, "view_settings", None)
            if view_settings is not None:
                view_settings.load_from_settings(settings, config_path=self.config_path)
                view_settings.set_overlay_calibration(self.overlay_calibration)

        self._refresh_overlay_peer_consumers()

    def _on_recent_languages_change(self, source: list[str], target: list[str]) -> None:
        """Callback when recent languages change in dashboard."""
        if self.settings is None:
            return
        self.settings.languages.recent_source_languages = list(source)
        self.settings.languages.recent_target_languages = list(target)
        self._save_settings()

    @property
    def runtime_logging(self) -> SessionRuntimeLoggingService:
        if self._runtime_logging is None:
            self._runtime_logging = SessionRuntimeLoggingService(ui_handler_factory=FletLogHandler)
        logs_view = getattr(self.app, "view_logs", None)
        if logs_view is not None:
            self._runtime_logging.attach_realtime_sink(logs_view)
        return self._runtime_logging

    @property
    def runtime_logging_mode(self) -> str:
        return self.runtime_logging.mode.value

    def set_runtime_logging_mode(self, mode: SessionLoggingMode | str) -> None:
        previous_mode = self.runtime_logging.mode
        self.runtime_logging.set_mode(mode)
        normalized_mode = self.runtime_logging.mode.value
        if (
            previous_mode is not SessionLoggingMode.DETAILED
            and self.runtime_logging.mode is SessionLoggingMode.DETAILED
        ):
            self._schedule_audio_environment_snapshot()
        manager = self._overlay_manager
        if manager is not None:
            set_logging_mode = getattr(manager, "set_logging_mode", None)
            if callable(set_logging_mode):
                set_logging_mode(normalized_mode)
        self._schedule_overlay_runtime_logging_mode_update()

    def _schedule_audio_environment_snapshot(self) -> None:
        async def _task() -> None:
            await self._log_audio_environment_snapshot_async()

        run_task = getattr(self.page, "run_task", None)
        if callable(run_task):
            try:
                run_task(_task)
                return
            except Exception as exc:
                self.log_detailed(
                    "[AudioDiag][Snapshot] failed to schedule via page.run_task",
                    level=logging.WARNING,
                    exception=exc,
                )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.log_detailed(
                "[AudioDiag][Snapshot] skipped reason=no_running_loop",
                level=logging.WARNING,
            )
            return

        task_coro = _task()
        try:
            loop.create_task(task_coro)
        except Exception as exc:
            task_coro.close()
            self.log_detailed(
                "[AudioDiag][Snapshot] skipped reason=create_task_failed",
                level=logging.WARNING,
                exception=exc,
            )

    async def _log_audio_environment_snapshot_async(self) -> None:
        from puripuly_heart.core.audio.diagnostics import (
            collect_pyaudiowpatch_snapshot_lines,
            collect_sounddevice_snapshot_lines,
        )

        sounddevice_lines, loopback_lines = await asyncio.gather(
            asyncio.to_thread(collect_sounddevice_snapshot_lines),
            asyncio.to_thread(collect_pyaudiowpatch_snapshot_lines),
        )
        for line in sounddevice_lines:
            self.log_detailed(line)
        for line in loopback_lines:
            self.log_detailed(line)

    async def _emit_overlay_runtime_logging_mode_update(self) -> None:
        bridge = self._overlay_bridge
        if bridge is None:
            return
        await bridge.broadcast_runtime_control(logging_mode=self.runtime_logging_mode)

    def _schedule_overlay_runtime_logging_mode_update(self) -> None:
        bridge = self._overlay_bridge
        if bridge is None:
            return

        run_task = getattr(self.page, "run_task", None)
        if callable(run_task):
            try:
                run_task(self._emit_overlay_runtime_logging_mode_update)
                return
            except Exception as exc:
                self.log_detailed(
                    "[Overlay] Failed to schedule logging mode update via page.run_task",
                    level=logging.WARNING,
                    exception=exc,
                )
                return

        try:
            asyncio.get_running_loop().create_task(self._emit_overlay_runtime_logging_mode_update())
        except RuntimeError:
            self.log_detailed(
                "[Overlay] Skipping logging mode update; no running loop and page.run_task unavailable",
                level=logging.WARNING,
            )

    def log_basic(self, message: str, *, level: int = logging.INFO) -> None:
        try:
            self.runtime_logging.emit_basic(message, level=level)
            return
        except Exception:
            logger.log(level, message)

    def log_detailed(
        self,
        message: str,
        *,
        level: int = logging.INFO,
        exception: BaseException | None = None,
    ) -> bool:
        rendered_message = message
        exc_info = None
        if exception is not None:
            exc_info = (type(exception), exception, exception.__traceback__)
            rendered_message = (
                f"{message}\n{''.join(traceback.format_exception(*exc_info)).rstrip()}"
            )
        try:
            return self.runtime_logging.emit_detailed(rendered_message, level=level)
        except Exception:
            logger.log(level, message, exc_info=exc_info)
            return True

    def log_detailed_lazy(
        self,
        build_message: Callable[[], str],
        *,
        level: int = logging.INFO,
        exception: BaseException | None = None,
    ) -> bool:
        exc_info = None
        if exception is not None:
            exc_info = (type(exception), exception, exception.__traceback__)

        def render_message() -> str:
            rendered_message = build_message()
            if exc_info is None:
                return rendered_message
            return f"{rendered_message}\n{''.join(traceback.format_exception(*exc_info)).rstrip()}"

        try:
            return self.runtime_logging.emit_detailed_lazy(render_message, level=level)
        except Exception:
            logger.log(level, build_message(), exc_info=exc_info)
            return True

    def _log_error(self, message: str) -> None:
        self.log_basic(message, level=logging.ERROR)

    def _get_qwen_key_and_base_url(self, secrets) -> tuple[str, str]:
        if self.settings is None:
            return "", ""
        if self.settings.qwen.region == QwenRegion.BEIJING:
            target_key = "alibaba_api_key_beijing"
        else:
            target_key = "alibaba_api_key_singapore"

        api_key = secrets.get(target_key) or ""
        if api_key:
            return api_key, self.settings.qwen.get_llm_base_url()

        # Backward compatibility: legacy single-key storage from older versions.
        legacy_key = secrets.get("alibaba_api_key") or ""
        if legacy_key:
            setter = getattr(secrets, "set", None)
            if callable(setter):
                with contextlib.suppress(Exception):
                    setter(target_key, legacy_key)
            return legacy_key, self.settings.qwen.get_llm_base_url()

        return "", self.settings.qwen.get_llm_base_url()

    async def _verify_qwen_key_with_model_fallback(
        self,
        api_key: str,
        *,
        base_url: str,
    ) -> tuple[bool, str]:
        if self.settings is None:
            return False, "Verification failed (check logs/console for details)"

        selected_model = self.settings.qwen.llm_model.value
        if await self._verify_qwen_llm_api_key(api_key, base_url=base_url, model=selected_model):
            return True, "Verification successful"

        for fallback_model in (
            model.value for model in QwenLLMModel if model.value != selected_model
        ):
            if await self._verify_qwen_llm_api_key(
                api_key,
                base_url=base_url,
                model=fallback_model,
            ):
                return False, f"qwen_model_unavailable:{selected_model}"

        return False, "Verification failed (check logs/console for details)"

    async def _verify_qwen_llm_api_key(
        self,
        api_key: str,
        *,
        base_url: str,
        model: str | None = None,
    ) -> bool:
        if self.settings is None:
            return False
        runtime_model = model or self.settings.qwen.llm_model.value
        if self.settings.stt.low_latency_mode:
            async_base_url = base_url.replace("/api/v1", "/compatible-mode/v1")
            return await AsyncQwenLLMProvider.verify_api_key(
                api_key,
                base_url=async_base_url,
                model=runtime_model,
            )
        return await QwenLLMProvider.verify_api_key(
            api_key,
            base_url=base_url,
            model=runtime_model,
        )

    async def _verify_and_update_status(self) -> None:
        """Background task to verify keys and update dashboard status."""
        if self.settings is None:
            return

        dash = getattr(self.app, "view_dashboard", None)
        if dash is None:
            return

        secrets = None
        with contextlib.suppress(Exception):
            secrets = create_secret_store(self.settings.secrets, config_path=self.config_path)

        alibaba_selected_valid_cache: bool | None = None
        alibaba_any_valid_cache: bool | None = None

        async def _verify_alibaba_selected() -> bool:
            nonlocal alibaba_selected_valid_cache
            if alibaba_selected_valid_cache is not None:
                return alibaba_selected_valid_cache
            if secrets is None:
                alibaba_selected_valid_cache = False
                return False
            key, base_url = self._get_qwen_key_and_base_url(secrets)
            selected_model = self.settings.qwen.llm_model.value
            alibaba_selected_valid_cache = await self._verify_qwen_llm_api_key(
                key,
                base_url=base_url,
                model=selected_model,
            )
            return alibaba_selected_valid_cache

        async def _verify_alibaba_any_model() -> bool:
            nonlocal alibaba_any_valid_cache
            if alibaba_any_valid_cache is not None:
                return alibaba_any_valid_cache
            if await _verify_alibaba_selected():
                alibaba_any_valid_cache = True
                return True
            if secrets is None:
                alibaba_any_valid_cache = False
                return False
            key, base_url = self._get_qwen_key_and_base_url(secrets)
            selected_model = self.settings.qwen.llm_model.value
            for fallback_model in (
                model.value for model in QwenLLMModel if model.value != selected_model
            ):
                if await self._verify_qwen_llm_api_key(
                    key,
                    base_url=base_url,
                    model=fallback_model,
                ):
                    alibaba_any_valid_cache = True
                    return True
            alibaba_any_valid_cache = False
            return False

        # 1. Verify LLM
        llm_valid = False
        if self.hub and self.hub.llm:
            # It was created, but is the key valid?
            try:
                provider_name = self.settings.provider.llm
                key = ""
                if provider_name == "gemini":
                    key = secrets.get("google_api_key") or "" if secrets is not None else ""
                    llm_valid = await GeminiLLMProvider.verify_api_key(
                        key,
                        model=self.settings.gemini.llm_model.value,
                    )
                elif provider_name == LLMProviderName.OPENROUTER:
                    resolution = (
                        resolve_openrouter_credentials(self.settings, secrets=secrets)
                        if secrets is not None
                        else None
                    )
                    if (
                        self.settings.openrouter.selected_source
                        == OpenRouterCredentialSource.MANAGED
                        and (resolution is None or resolution.api_key is None)
                    ):
                        llm_valid = self._managed_openrouter_can_attempt_translation()
                    else:
                        key = (
                            resolution.api_key
                            if resolution is not None and resolution.api_key
                            else ""
                        )
                        llm_valid = bool(key) and await OpenRouterLLMProvider.verify_api_key(key)
                elif provider_name == LLMProviderName.DEEPSEEK:
                    key = (
                        (secrets.get("deepseek_api_key") if secrets is not None else None)
                        or os.getenv("DEEPSEEK_API_KEY")
                        or ""
                    )
                    llm_valid = bool(key) and await DeepSeekLLMProvider.verify_api_key(key)
                elif provider_name == LLMProviderName.CEREBRAS:
                    key = (
                        (secrets.get("cerebras_api_key") if secrets is not None else None)
                        or os.getenv("CEREBRAS_API_KEY")
                        or ""
                    )
                    llm_valid = bool(key) and await CerebrasLLMProvider.verify_api_key(key)
                elif provider_name == "qwen":
                    llm_valid = await _verify_alibaba_selected()
                elif provider_name == LLMProviderName.LOCAL_LLM:
                    llm_valid = True
                else:
                    # Assume valid for others or if no key usage known
                    llm_valid = True
            except Exception:
                llm_valid = False

        llm_requires_secret = self._llm_provider_requires_secret(self.settings.provider.llm)
        # If LLM verification failed, only key-backed providers should show needs-key state.
        if not llm_valid:
            dash.set_translation_needs_key(llm_requires_secret)
            # If it was enabled, we potentially disable it or just let the warning show on next interaction
            # User request: "Validation Fail -> Orange". Implicitly, if it's ON and fails, maybe we should turn it OFF?
            # For now, setting needs_key=True ensures that if they try to toggle, it warns.
            # If it is currently ON, we might want to flag it.
            if self.hub:
                self.hub.translation_enabled = False  # Disable internally
            dash.set_translation_enabled(False)  # Visually turn off
        else:
            dash.set_translation_needs_key(False)
            if self.settings.provider.llm == LLMProviderName.LOCAL_LLM and self.hub is not None:
                dash.set_translation_enabled(bool(self.hub.translation_enabled))

        # 2. Verify STT
        stt_requires_secret = self._stt_provider_requires_secret(self.settings.provider.stt)
        stt_valid = not stt_requires_secret
        if self.hub and self.hub.stt and stt_requires_secret:
            try:
                provider_name = self.settings.provider.stt

                if provider_name == STTProviderName.DEEPGRAM:
                    key = secrets.get("deepgram_api_key") or "" if secrets is not None else ""
                    stt_valid = await DeepgramRealtimeSTTBackend.verify_api_key(key)
                elif provider_name == STTProviderName.QWEN_ASR:
                    stt_valid = await _verify_alibaba_any_model()
                elif provider_name == STTProviderName.SONIOX:
                    key = secrets.get("soniox_api_key") or "" if secrets is not None else ""
                    stt_valid = await SonioxRealtimeSTTBackend.verify_api_key(key)
                else:
                    stt_valid = True
            except Exception:
                stt_valid = False

        if not stt_valid:
            dash.set_stt_needs_key(stt_requires_secret)
            if self.hub:
                # Close STT backend?
                pass
            dash.set_stt_enabled(False)
        else:
            dash.set_stt_needs_key(False)

        await self._refresh_managed_trial_usage_state_impl(auto_show_founder_letter=False)
