from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import inspect
import json
import logging
import os
import secrets
import sys
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

from puripuly_heart.app.language_selection import LanguageSelectionChange
from puripuly_heart.app.ports.canonical_settings_persistence import (
    ProviderVerificationBinding,
)
from puripuly_heart.app.ports.gpu_worker import GpuWorkerDevice
from puripuly_heart.app.ports.microphone_test import (
    MicrophoneTestCapturePort,
    MicrophoneTestCaptureRequest,
)
from puripuly_heart.app.ports.provider_verifier import ProviderVerifierPort
from puripuly_heart.app.ports.runtime_apply import RuntimeApplyRequest
from puripuly_heart.app.ports.settings_repository import CommittedSettingsRepositoryPort
from puripuly_heart.app.ports.ui_models import (
    GpuDashboardNotice,
    GpuDeviceOption,
    GpuNoticeAction,
    OptionItem,
    OverlayPeerPresentationState,
)
from puripuly_heart.app.ports.ui_presentation import UiPresentationPort
from puripuly_heart.app.ports.vrchat_osc_presence import VrchatOscPresencePort
from puripuly_heart.app.services.application_runtime_logging import (
    ApplicationRuntimeLoggingOwner,
)
from puripuly_heart.app.services.application_shutdown import (
    ApplicationShutdownCallback,
    ApplicationShutdownContext,
    ApplicationShutdownCoordinator,
    ApplicationShutdownDiagnostic,
    application_shutdown_callback,
)
from puripuly_heart.app.services.canonical_settings_persistence import (
    SettingsOwner,
    compose_settings_owner,
)
from puripuly_heart.app.services.clipboard_auto_translation import (
    ClipboardAutoTranslationOwner,
)
from puripuly_heart.app.services.github_star_prompt import (
    GithubStarPromptOwner,
)
from puripuly_heart.app.services.github_star_prompt import (
    github_star_prompt_utc_timestamp as _github_star_prompt_utc_timestamp,
)
from puripuly_heart.app.services.local_asr_selection import (
    LOCAL_CPU_AUTO_PROVIDER,
    LOCAL_CPU_DIRECT_MODEL_BY_PROVIDER,
    LOCAL_CPU_PROVIDERS,
    resolve_local_asr_selection,
)
from puripuly_heart.app.services.managed_auth_claims import (
    MANAGED_AUTH_CLAIM_SOURCE_DISCORD,
    ManagedAuthClaimGuard,
)
from puripuly_heart.app.services.managed_connection_auth import (
    ManagedConnectionAuthRequest,
    ManagedConnectionAuthService,
)
from puripuly_heart.app.services.managed_status_refresh import ManagedStatusRefreshOwner
from puripuly_heart.app.services.manual_typing import ManualTypingOwner
from puripuly_heart.app.services.microphone_test import (
    MicrophoneTestSelfCaptureState,
    MicrophoneTestSessionOwner,
    MicrophoneTestSessionRequest,
)
from puripuly_heart.app.services.openrouter_pkce_flow import OpenRouterPkceFlowOwner
from puripuly_heart.app.services.overlay_calibration import OverlayCalibrationOwner
from puripuly_heart.app.services.peer_capture_target import PeerCaptureTargetResolutionService
from puripuly_heart.app.services.provider_credential_verification import (
    PROVIDER_CREDENTIAL_EMPTY,
    PROVIDER_CREDENTIAL_ERROR,
    PROVIDER_CREDENTIAL_MODEL_UNAVAILABLE,
    PROVIDER_CREDENTIAL_UNKNOWN,
    PROVIDER_CREDENTIAL_VERIFIED,
    ProviderCredentialVerificationOwner,
    ProviderCredentialVerificationRequest,
)
from puripuly_heart.app.services.provider_runtime_apply import (
    _ControllerNoopRuntimeApply,
    _ControllerOverlayOscOutputRuntimeApply,
    _ControllerProviderRuntimeApply,
    _ControllerSttLanguageAudioRuntimeApply,
    _ControllerUiPromptClipboardStateRuntimeApply,
    _overlay_osc_output_runtime_degraded_transaction_result,
    _overlay_osc_output_save_failed_transaction_result,
    _provider_runtime_apply_unavailable_result,
    _ProviderRuntimeApplyPlan,
    _runtime_apply_failed_result,
    _runtime_apply_result_as_degraded_transaction,
    _stt_language_audio_runtime_degraded_transaction_result,
    _stt_language_audio_runtime_unavailable_result,
    _stt_language_audio_save_failed_transaction_result,
    _translation_provider_save_failed_transaction_result,
    _ui_prompt_clipboard_state_runtime_degraded_transaction_result,
    _ui_prompt_clipboard_state_save_failed_transaction_result,
)
from puripuly_heart.app.services.provider_secret_change import (
    ProviderSecretChangeExecution,
    ProviderSecretChangeOwner,
    ProviderSecretChangeRequest,
)
from puripuly_heart.app.services.provider_status_verification import (
    ConfiguredProviderStatusVerificationRequest,
    ConfiguredProviderStatusVerificationResult,
    ProviderStatusVerificationOwner,
)
from puripuly_heart.app.services.qq_managed_auth import QqManagedAuthRequest, QqManagedAuthService
from puripuly_heart.app.services.secret_settings_transaction import (
    SecretSetRequest,
    SecretSettingsTransaction,
)
from puripuly_heart.app.services.settings_mutation import (
    OverlayOscOutputSettingsMutation,
    SettingsMutationService,
    SttLanguageAudioSettingsMutation,
    TranslationProviderSettingsMutation,
    UiPromptClipboardStateSettingsMutation,
)
from puripuly_heart.app.services.settings_mutation_legacy import (
    _apply_settings_path_patch,
    build_overlay_osc_output_settings_path_patch,
    build_stt_language_audio_settings_path_patch,
    build_translation_provider_settings_path_patch,
    build_ui_prompt_clipboard_state_settings_path_patch,
    settings_path_mutation_validator_for_command,
)
from puripuly_heart.app.services.settings_projection import (
    SettingsProjectionOwner,
    SettingsViewSettingsChange,
)
from puripuly_heart.app.services.vrc_mic_sync import VrcMicSyncOwner
from puripuly_heart.app.wiring import (
    DiscordManagedBrokerClientAdapter,
    DiscordOAuthAuthAdapter,
    LocalASRProviderRuntimeFactory,
    ManagedIdentityPreflightAdapter,
    ManagedSTTProviderFactory,
    build_custom_vocabulary_runtime_config,
    build_managed_identity_state_port,
    build_openrouter_credential_runtime_config,
    build_openrouter_release_runtime_config,
    build_peer_stt_provider_signature_from_vnext,
    compose_peer_capture_session_owner,
    compose_self_capture_session_owner,
    copy_stable_secrets_to_vnext_namespace,
    create_llm_provider,
    create_local_asr_provisioning_owner,
    create_microphone_test_capture_adapter,
    create_provider_verifier,
    create_secret_store,
    create_sync_secret_store_adapter,
    resolve_overlay_config,
    resolve_peer_stt_runtime_config_from_vnext,
    resolve_self_stt_runtime_config,
)
from puripuly_heart.config.audio_host_api import normalize_input_host_api
from puripuly_heart.config.capture_target_resolution import resolve_desktop_audio_capture_target
from puripuly_heart.config.llm_profiles import (
    get_openrouter_selection_alias_for_model_and_source,
    profile_for_alias,
)
from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.config.paths import user_config_dir
from puripuly_heart.config.process_capture_resolution import (
    ProcessCaptureResolver,
    ProcessCaptureTargetUnavailableError,
)
from puripuly_heart.config.resolved import (
    ResolvedDesktopAudioCaptureTarget,
    ResolvedOverlayConfig,
    ResolvedSTTConfig,
)
from puripuly_heart.config.settings import (
    DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA,
    DESKTOP_FLET_DEFAULT_TEXT_SCALE,
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
    TranslationModel,
    normalize_owned_referral_id,
    with_telemetry_consent,
)
from puripuly_heart.config.settings_vnext.schema import (
    AppSettingsVNext,
    CaptureTargetIntent,
    ProcessCaptureTargetIntent,
)
from puripuly_heart.config.vad_defaults import DEFAULT_STABLE_VAD_HANGOVER_MS
from puripuly_heart.core.audio.desktop_pipeline import DesktopPeerPipeline
from puripuly_heart.core.audio.desktop_source import DesktopLoopbackAudioSource
from puripuly_heart.core.audio.gate import VrcMicAudioGate
from puripuly_heart.core.audio.process_identity import (
    PsutilCurrentUserProcessSnapshots,
    PsutilProcessIdentityWatcher,
)
from puripuly_heart.core.audio.process_source import ProcessAudioCaptureSource
from puripuly_heart.core.audio.source import (
    AudioSource,
    SelfMicCaptureChannelDecision,
    SoundDeviceAudioSource,
    determine_self_mic_capture_channels,
    observe_microphone_test_route,
    resolve_sounddevice_input_device,
)
from puripuly_heart.core.clipboard.watcher import create_clipboard_watcher
from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.hardware_fingerprint import get_raw_hardware_fingerprint
from puripuly_heart.core.lifecycle import (
    SHUTDOWN_PHASE_CLOSE_LOGGING_DIAGNOSTICS,
    SHUTDOWN_PHASE_CLOSE_PROVIDERS_OUTPUT_ADAPTERS,
    SHUTDOWN_PHASE_FINAL_DIAGNOSTICS,
    SHUTDOWN_PHASE_FREEZE_INGRESS,
    SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL,
    SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS,
)
from puripuly_heart.core.llm.provider import SemaphoreLLMProvider
from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimePort,
    LocalASRProviderRuntimeSnapshot,
    ProviderRuntimeBuildRequest,
    ProviderRuntimeDiagnostic,
    ProviderRuntimeGpuRecoveryRequest,
    ProviderRuntimeRecoveryChannel,
)
from puripuly_heart.core.local_asr_provisioning import (
    LocalASRInstallRequest,
    LocalASRInstallResult,
    LocalASRProvisioningDiagnostic,
    LocalASRProvisioningPort,
    LocalASRProvisioningSnapshot,
)
from puripuly_heart.core.local_gpu_assets import local_gpu_model_path
from puripuly_heart.core.local_stt_assets import (
    LOCAL_QWEN_GPU_MODEL_ID,
    LOCAL_STT_MODEL_ID,
    REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
    LocalParakeetSherpaLoadError,
    LocalQwenSherpaLoadError,
    LocalSTTManifestInvalidError,
    LocalSTTModelMissingError,
)
from puripuly_heart.core.local_stt_catalog import (
    LocalCPUAutoUnavailableError,
)
from puripuly_heart.core.managed_openrouter_broker_client import (
    HttpManagedOpenRouterBrokerClient,
)
from puripuly_heart.core.managed_openrouter_release import (
    ManagedOpenRouterReleaseBehavior,
    ManagedOpenRouterReleaseService,
    ManagedOpenRouterStatusRefreshResult,
    TalkTogetherPassStatus,
    UnavailableManagedOpenRouterReleaseClient,
    format_managed_openrouter_diagnostics,
)
from puripuly_heart.core.messages import (
    RUNTIME_APPLY_STATUS_APPLIED,
    TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    TransactionResult,
)
from puripuly_heart.core.openrouter_credentials import (
    OPENROUTER_BYOK_API_KEY_SECRET,
    OPENROUTER_MANAGED_API_KEY_SECRET,
    resolve_openrouter_credentials,
)
from puripuly_heart.core.openrouter_handoff import (
    is_effectively_exhausted,
    mark_founder_letter_shown,
    should_auto_show_founder_letter,
)
from puripuly_heart.core.openrouter_metadata import OpenRouterKeyMetadata
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
from puripuly_heart.core.peer_capture import (
    PeerCaptureAdmission,
    PeerCaptureAdmissionStatus,
    PeerCaptureDiagnostic,
    PeerCaptureFailureReason,
    PeerCaptureLanguageFacts,
    PeerCaptureResolvedTarget,
    PeerCaptureSessionConfig,
    PeerCaptureSessionSnapshot,
    PeerCaptureTargetIntent,
    PeerCaptureTargetResolution,
    PeerCaptureTargetStatus,
)
from puripuly_heart.core.runtime.clipboard import ClipboardRuntime
from puripuly_heart.core.runtime.desktop_overlay_bounds import (
    DesktopOverlayBoundsOwner,
    is_finite_non_bool_number,
)
from puripuly_heart.core.runtime.github_star_prompt import GithubStarPromptRuntime
from puripuly_heart.core.runtime.gpu_asr import GpuASRChannel
from puripuly_heart.core.runtime.local_asr_transition import (
    LocalASRSessionOptions,
    LocalASRTransitionCoordinator,
    LocalASRTransitionRequest,
    PreparedLocalASRTransition,
)
from puripuly_heart.core.runtime.local_qwen_lifecycle import LOCAL_QWEN_IDLE_RELEASE_SECONDS
from puripuly_heart.core.runtime.logging import RuntimeLoggingService
from puripuly_heart.core.runtime.mic_test import MicTestRuntime
from puripuly_heart.core.runtime.oauth import OAuthRuntime
from puripuly_heart.core.runtime.overlay import OverlayRuntimeHandle
from puripuly_heart.core.runtime.overlay_session_fallback import (
    OverlaySessionFallbackOwner,
)
from puripuly_heart.core.runtime.peer_channel import (
    PeerCaptureSessionOwner,
    PeerLocalASRTransitionSuperseded,
)
from puripuly_heart.core.runtime.provider_rebuild import ProviderRuntimeRebuildService
from puripuly_heart.core.runtime.receiver import VrcMicReceiverRuntime
from puripuly_heart.core.runtime.self_capture import SelfCaptureSessionOwner
from puripuly_heart.core.runtime.vrchat_osc_presence import (
    VrchatOscPresenceProbeOwner,
)
from puripuly_heart.core.runtime_logging import (
    RealtimeLogHandler,
    RuntimeLoggingSinks,
    SessionLoggingMode,
    SessionRuntimeLoggingService,
)
from puripuly_heart.core.self_capture import (
    SelfCaptureAdmission,
    SelfCaptureAdmissionStatus,
    SelfCaptureDiagnostic,
    SelfCaptureProviderStatus,
    SelfCaptureSessionConfig,
    SelfCaptureSessionSnapshot,
    SelfCaptureSessionState,
)
from puripuly_heart.core.stt.controller import FinalTranscriptSuppressedNotification
from puripuly_heart.core.stt.custom_vocab import get_effective_custom_terms
from puripuly_heart.core.telemetry import (
    TranslationSuccessTelemetryClientPort,
    TranslationSuccessTelemetryResult,
    TranslationSuccessTelemetryService,
)
from puripuly_heart.core.translation_policy import FIXED_TRANSLATION_POLICY
from puripuly_heart.core.vad.bundled import ensure_silero_vad_onnx
from puripuly_heart.core.vad.gating import VadGating, create_peer_vad_gating
from puripuly_heart.core.vad.silero import SileroVadOnnx

logger = logging.getLogger(__name__)

QQ_AUTH_DIALOG_MESSAGE_KEY_BY_SERVICE_KEY = {
    "qq_managed_auth.already_claimed_discord": "qq_auth.error.already_claimed_discord",
    "qq_managed_auth.invalid_credential": "qq_auth.error.credential_mismatch",
    "qq_managed_auth.mismatch": "qq_auth.error.credential_mismatch",
    "qq_managed_auth.lifetime_used": "qq_auth.error.lifetime_used",
    "qq_managed_auth.rate_limited": "qq_auth.error.rate_limited",
    "qq_managed_auth.key_unavailable": "qq_auth.error.key_unavailable",
    "qq_managed_auth.broker_unavailable": "qq_auth.error.broker_unavailable",
    "qq_managed_auth.settings_commit_failed": "qq_auth.error.settings_commit_failed",
    "qq_managed_auth.secret_write_failed": "qq_auth.error.secret_write_failed",
    "qq_managed_auth.error.retry": "qq_auth.error.retry",
}

# Hardcoded STT session reset deadline (not configurable via settings)
STT_RESET_DEADLINE_S = 300.0
OVERLAY_STARTUP_TIMEOUT_MS = 3000
OVERLAY_SHUTDOWN_GRACE_S = 0.05
DESKTOP_BOUNDS_PERSIST_DEBOUNCE_S = 0.05
MANUAL_INPUT_TYPING_IDLE_TIMEOUT_S = 3.0
MANUAL_SUBMIT_TYPING_TIMEOUT_S = 10.0
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
_GITHUB_STAR_PROMPT_MANAGED_CONNECTIONS = frozenset(
    {
        TranslationConnection.MANAGED,
        TranslationConnection.MANAGED_CHINA,
    }
)
_MANAGED_OPENROUTER_CONNECTIONS = frozenset(
    {
        TranslationConnection.MANAGED,
        TranslationConnection.MANAGED_CHINA,
    }
)
_MANAGED_OPENROUTER_MODEL_BY_TRANSLATION_MODEL = {
    TranslationModel.GEMMA4: OpenRouterLLMModel.GEMMA_4_26B_A4B_IT,
    TranslationModel.DEEPSEEK_V4_FLASH: OpenRouterLLMModel.DEEPSEEK_V4_FLASH,
}
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


def _canonical_json_signature(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _managed_connection_auth_settings_values(settings: AppSettings) -> dict[str, Any]:
    managed = settings.managed_identity
    selection_alias = settings.openrouter.selection_alias
    return {
        "intent": {
            "translation": {
                "connection": settings.translation.connection.value,
                "model": settings.translation.model.value,
            },
            "openrouter": {
                "selected_source": settings.openrouter.selected_source.value,
                "llm_model": settings.openrouter.llm_model.value,
                "selection_alias": selection_alias.value if selection_alias is not None else None,
            },
        },
        "state": {
            "managed_connection": {
                "installation_id": managed.installation_id,
                "active_managed_credential_ref": managed.active_managed_credential_ref,
                "active_managed_expires_at": managed.active_managed_expires_at,
                "founder_letter_seen_credential_ref": managed.founder_letter_seen_credential_ref,
                "referral_id": managed.referral_id,
                "local_managed_claim_sources": list(managed.local_managed_claim_sources),
            }
        },
    }


def _callable_accepts_keyword(callable_obj: object, keyword: str) -> bool:
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return True
    return keyword in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _callable_accepts_positional_arguments(callable_obj: object, count: int) -> bool:
    try:
        inspect.signature(callable_obj).bind(*([None] * count))
    except TypeError:
        return False
    except ValueError:
        return True
    return True


def _raise_lifecycle_cleanup_failures(message: str, failures: list[Exception]) -> None:
    if not failures:
        return
    if len(failures) == 1:
        raise failures[0]
    raise ExceptionGroup(message, failures)


class _StrictSettingsSaveFailed(Exception):
    pass


class _LocalASRTransitionSuperseded(Exception):
    pass


def _managed_identity_delta(baseline: object, current: object) -> dict[str, object]:
    baseline_values = asdict(baseline)
    current_values = asdict(current)
    return {
        field_name: copy.deepcopy(value)
        for field_name, value in current_values.items()
        if baseline_values.get(field_name) != value
    }


def _apply_managed_identity_delta(settings: AppSettings, values: Mapping[str, object]) -> None:
    for field_name, value in values.items():
        setattr(settings.managed_identity, field_name, copy.deepcopy(value))


def _restore_managed_identity(settings: AppSettings, snapshot: object) -> None:
    for field_name, value in asdict(snapshot).items():
        setattr(settings.managed_identity, field_name, copy.deepcopy(value))


def _copy_runtime_only_ui_state(source: AppSettings, target: AppSettings) -> None:
    target.ui.overlay_enabled = bool(source.ui.overlay_enabled)
    target.ui.peer_translation_enabled = bool(source.ui.peer_translation_enabled)


def _settings_mutation_committed(result: TransactionResult) -> bool:
    return result.status in {
        TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
        TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    }


def _provider_verification_fingerprint(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


_PROVIDER_VERIFICATION_SECRET_KEY_BY_PROVIDER = {
    "deepgram": "deepgram_api_key",
    "soniox": "soniox_api_key",
    "google": "google_api_key",
    "openrouter": OPENROUTER_BYOK_API_KEY_SECRET,
    "deepseek": "deepseek_api_key",
    "cerebras": "cerebras_api_key",
    "alibaba_beijing": "alibaba_api_key_beijing",
    "alibaba_singapore": "alibaba_api_key_singapore",
}
_PROVIDER_BY_VERIFICATION_SECRET_KEY = {
    secret_key: provider
    for provider, secret_key in _PROVIDER_VERIFICATION_SECRET_KEY_BY_PROVIDER.items()
}


def _sensitive_optional_text_signature(value: str | None) -> tuple[int, str] | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return (len(normalized), digest)


def _managed_openrouter_identity_signature(settings: AppSettings) -> tuple[object, ...]:
    identity = settings.managed_identity
    return (
        identity.installation_id,
        _sensitive_optional_text_signature(identity.release_token),
        identity.release_token_expires_at,
        identity.verified_hardware_hash,
        identity.verified_hardware_hash_salt_version,
        identity.active_managed_credential_ref,
        identity.active_managed_expires_at,
        identity.referral_id,
    )


@dataclass(slots=True)
class _SelfCaptureAdmissionAdapter:
    callback: Callable[[SelfCaptureSessionConfig], Awaitable[SelfCaptureAdmission]]

    async def admit(self, config: SelfCaptureSessionConfig) -> SelfCaptureAdmission:
        return await self.callback(config)


@dataclass(slots=True)
class _SelfCaptureVadSink:
    hub_provider: Callable[[], ClientHub | None]

    async def handle_vad_event(self, event: object) -> None:
        hub = self.hub_provider()
        if hub is None:
            raise RuntimeError("Self VAD sink requires the production hub")
        await hub.handle_vad_event(event)


@dataclass(slots=True)
class _PeerCaptureAdmissionAdapter:
    callback: Callable[[PeerCaptureSessionConfig], Awaitable[PeerCaptureAdmission]]

    async def admit(self, config: PeerCaptureSessionConfig) -> PeerCaptureAdmission:
        return await self.callback(config)


@dataclass(slots=True)
class _PeerCaptureTargetResolverAdapter:
    callback: Callable[
        [PeerCaptureTargetIntent],
        Awaitable[PeerCaptureTargetResolution],
    ]

    async def resolve(
        self,
        target: PeerCaptureTargetIntent,
    ) -> PeerCaptureTargetResolution:
        return await self.callback(target)


@dataclass(slots=True)
class _PeerCaptureVadSink:
    hub_provider: Callable[[], ClientHub | None]

    async def handle_vad_event(self, event: object) -> None:
        hub = self.hub_provider()
        if hub is None:
            raise RuntimeError("Peer VAD sink requires the production hub")
        await hub.handle_peer_vad_event(event)


@dataclass(slots=True)
class GuiController:
    page: object
    app: UiPresentationPort
    config_path: Path
    allow_stable_settings_import: bool = False
    runtime_logging_sinks: RuntimeLoggingSinks | None = field(default=None, repr=False)
    settings_mutation_service: SettingsMutationService | None = None
    provider_verifier: ProviderVerifierPort | None = None
    telemetry_client: TranslationSuccessTelemetryClientPort | None = None
    vrchat_osc_presence: VrchatOscPresencePort | None = field(default=None, repr=False)
    local_asr_provisioning: LocalASRProvisioningPort | None = field(
        default=None,
        repr=False,
    )
    settings_owner: SettingsOwner | None = field(default=None, repr=False)

    settings: AppSettings | None = None
    _vnext_settings_authoritative: bool = field(init=False, default=False, repr=False)
    _canonical_mutation_rollback_legacy_snapshot: AppSettings | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _canonical_mutation_rollback_active_settings: AppSettings | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _provider_secret_change_owner: ProviderSecretChangeOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _canonical_mutation_rollback_authoritative: bool = field(
        init=False,
        default=False,
        repr=False,
    )
    _canonical_mutation_rollback_pending: bool = field(
        init=False,
        default=False,
        repr=False,
    )
    _canonical_mutation_depth: int = field(init=False, default=0, repr=False)
    _canonical_legacy_projection_snapshot: AppSettings | None = field(
        init=False,
        default=None,
        repr=False,
    )
    last_settings_mutation_result: TransactionResult | None = field(
        init=False,
        default=None,
    )
    clock: SystemClock = SystemClock()
    _managed_openrouter_release_service: ManagedOpenRouterReleaseService | None = None
    _openrouter_pkce_flow_owner: OpenRouterPkceFlowOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )

    sender: VrchatOscUdpSender | None = None
    osc: ChatboxPaginator | None = None
    hub: ClientHub | None = None
    _self_capture_owner: SelfCaptureSessionOwner | None = field(init=False, default=None)
    _provider_rebuild_runtime: ProviderRuntimeRebuildService = field(
        init=False,
        default_factory=ProviderRuntimeRebuildService,
        repr=False,
    )
    _peer_capture_target_resolution: PeerCaptureTargetResolutionService = field(
        init=False,
        default_factory=PeerCaptureTargetResolutionService,
        repr=False,
    )
    _peer_runtime: PeerCaptureSessionOwner | None = None
    _vrc_mic_sync_owner: VrcMicSyncOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    vrc_mic_state: VrcMicState | None = None
    vrc_mic_audio_gate: VrcMicAudioGate | None = None

    _bridge_task: asyncio.Task[None] | None = None
    _mic_task: asyncio.Task[None] | None = None
    _manual_typing_owner: ManualTypingOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _audio_source: AudioSource | None = None
    _last_mic_loop_close_exception: BaseException | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _microphone_test_owner: MicrophoneTestSessionOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _debug_capture_fault_profile: str = field(init=False, default="none")
    _debug_stt_fault_profile: str = field(init=False, default="none")
    _vad: VadGating | None = None
    _stt_desired: bool = False
    _stt_restart_requested: bool = False
    _stt_force_immediate: bool = False
    _stt_activation_generation: int = field(init=False, default=0)
    _stt_activation_starting: bool = field(init=False, default=False)
    _stt_activation_failed: bool = field(init=False, default=False)
    _self_asr_model_loading: bool = field(init=False, default=False)
    _self_local_asr_transition: LocalASRTransitionCoordinator = field(
        init=False,
        default_factory=lambda: LocalASRTransitionCoordinator(channel="self"),
        repr=False,
    )
    _peer_local_asr_transition: LocalASRTransitionCoordinator = field(
        init=False,
        default_factory=lambda: LocalASRTransitionCoordinator(channel="peer"),
        repr=False,
    )
    _last_stt_runtime_signature: tuple[object, ...] | None = None
    _last_self_stt_runtime_signature: tuple[object, ...] | None = None
    _last_peer_stt_runtime_signature: tuple[object, ...] | None = None
    _last_self_stt_provider_signature: tuple[object, ...] | None = None
    _last_peer_stt_provider_signature: tuple[object, ...] | None = None
    _last_llm_provider_signature: tuple[object, ...] | None = None
    _last_peer_translation_enabled: bool | None = None
    _last_peer_translation_activation_requested: bool | None = None
    _peer_activation_generation: int = field(init=False, default=0)
    _peer_activation_starting: bool = field(init=False, default=False)
    _peer_asr_model_loading: bool = field(init=False, default=False)
    _superseded_local_asr_settings_ids: set[int] = field(
        init=False,
        default_factory=set,
        repr=False,
    )
    _process_idle_preparation_scheduled: bool = field(init=False, default=False)
    _peer_process_warning_reason: str | None = field(init=False, default=None)
    _settings_projection_owner: SettingsProjectionOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _ui_event_bridge: object | None = None
    _clipboard_auto_translation_owner: ClipboardAutoTranslationOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _local_stt_pending_enable_after_install: bool = field(init=False, default=False)
    _local_stt_pending_enable_generation: int | None = field(init=False, default=None)
    _local_stt_pending_peer_enable_after_install: bool = field(init=False, default=False)
    _gpu_pending_enable_channels: frozenset[GpuASRChannel] = field(
        init=False,
        default=frozenset(),
        repr=False,
    )
    _gpu_ui_state: str | None = field(init=False, default=None, repr=False)
    _gpu_devices: tuple[GpuWorkerDevice, ...] = field(init=False, default=(), repr=False)
    _gpu_discovery_attempted: bool = field(init=False, default=False, repr=False)
    _gpu_discovery_failed: bool = field(init=False, default=False, repr=False)
    _gpu_discovery_failure_state: str | None = field(init=False, default=None, repr=False)
    _gpu_discovery_generation: int = field(init=False, default=0, repr=False)
    _gpu_discovery_origin: str = field(init=False, default="settings", repr=False)
    _gpu_provider_recovery_lock: asyncio.Lock | None = field(
        init=False,
        default=None,
        repr=False,
    )
    # Overlay runtime internals are owned by OverlayRuntimeHandle.
    _overlay_runtime: OverlayRuntimeHandle | None = None
    _overlay_lock: asyncio.Lock | None = None
    _active_overlay_target: str | None = field(init=False, default=None)
    _overlay_session_fallback_owner: OverlaySessionFallbackOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _vrchat_osc_presence_owner: VrchatOscPresenceProbeOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _desktop_overlay_bounds_owner: DesktopOverlayBoundsOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _managed_status_refresh_owner: ManagedStatusRefreshOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _provider_status_verification_owner: ProviderStatusVerificationOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _provider_credential_verification_owner: ProviderCredentialVerificationOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _managed_trial_pending_auth: bool = field(init=False, default=False)
    _discord_managed_auth_in_progress: bool = field(init=False, default=False)
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
    _github_star_prompt_owner: GithubStarPromptOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _runtime_logging_owner: ApplicationRuntimeLoggingOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _application_lifecycle: ApplicationShutdownCoordinator | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _shutdown_ingress_frozen: bool = field(init=False, default=False, repr=False)
    _local_qwen_hallucination_detection_count: int = field(init=False, default=0)
    _local_qwen_hallucination_modal_shown: bool = field(init=False, default=False)
    _stop_complete: bool = field(init=False, default=False, repr=False)
    _stop_exception: BaseException | None = field(init=False, default=None, repr=False)

    overlay_state: str = "off"
    failure_reason: str | None = None
    auto_restart_scheduled: bool = False
    desktop_overlay_interaction_mode: str = field(
        init=False,
        default=DESKTOP_INTERACTION_MODE_EDIT,
    )
    _overlay_calibration_owner: OverlayCalibrationOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )

    @property
    def overlay_calibration(self) -> OverlayCalibration:
        return self._get_overlay_calibration_owner().current

    @overlay_calibration.setter
    def overlay_calibration(self, calibration: OverlayCalibration) -> None:
        self._get_overlay_calibration_owner().replace_current(calibration)

    @property
    def _overlay_calibration_draft(self) -> OverlayCalibration | None:
        return self._get_overlay_calibration_owner().draft

    @_overlay_calibration_draft.setter
    def _overlay_calibration_draft(
        self,
        calibration: OverlayCalibration | None,
    ) -> None:
        self._get_overlay_calibration_owner().replace_draft(calibration)

    def _get_settings_owner(self) -> SettingsOwner:
        if self.settings_owner is None:
            self.settings_owner = compose_settings_owner(self.config_path)
        return self.settings_owner

    def _legacy_settings_patch_repository(
        self,
        *,
        committed_settings: AppSettings,
        base_settings: AppSettings | None = None,
        surface: str = "translation_provider",
        provider_verification_binding: ProviderVerificationBinding | None = None,
    ) -> CommittedSettingsRepositoryPort[AppSettings]:
        return self._get_settings_owner().create_legacy_patch_repository(
            current_settings=lambda: self.settings,
            canonical_projection_snapshot=lambda: self._canonical_legacy_projection_snapshot,
            begin_canonical_mutation=self._begin_canonical_mutation,
            update_canonical_from_legacy_delta=(self._update_canonical_settings_from_legacy_delta),
            bind_provider_verification=self._bind_provider_verification,
            persist_settings=self._persist_settings_at_controller_boundary,
            rollback_canonical_mutation=self._rollback_canonical_mutation,
            save_failure_sink=self._log_error,
            remember_canonical_projection=self._remember_canonical_legacy_projection,
            committed_settings=committed_settings,
            base_settings=base_settings,
            surface=surface,
            provider_verification_binding=provider_verification_binding,
        )

    @property
    def vnext_settings(self) -> AppSettingsVNext | None:
        return self._get_settings_owner().canonical

    @vnext_settings.setter
    def vnext_settings(self, settings: AppSettingsVNext | None) -> None:
        self._get_settings_owner().canonical = settings

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
            and self._hub_has_stt_provider("peer")
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
        return self._effective_peer_translation_enabled_for(settings)

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

    def get_event_language_codes(self) -> tuple[str | None, str | None]:
        if self.settings is None:
            return None, None
        return self.settings.languages.source_language, self.settings.languages.target_language

    def overlay_peer_presentation_state(self) -> OverlayPeerPresentationState | None:
        if self.settings is None:
            return None
        peer_effective = self._effective_peer_translation_enabled_for(self.settings)
        if peer_effective or not self.settings.ui.peer_translation_enabled:
            self._peer_process_warning_reason = None
        return OverlayPeerPresentationState(
            overlay_intent_enabled=bool(self.settings.ui.overlay_enabled),
            overlay_state=self.overlay_state,
            overlay_failure_reason=self.failure_reason,
            peer_intent_enabled=bool(self.settings.ui.peer_translation_enabled),
            peer_effective_enabled=peer_effective,
            peer_warning_reason=self._peer_process_warning_reason,
            peer_activation_starting=(
                self._peer_activation_starting or self._peer_asr_model_loading
            ),
        )

    def _refresh_overlay_peer_consumers(self) -> None:
        with contextlib.suppress(Exception):
            self.app.refresh_overlay_peer_contract(self.overlay_peer_presentation_state())

    async def _refresh_overlay_runtime_dependencies(
        self,
        *,
        peer_stop_mode: str = "retain",
    ) -> None:
        if self.settings is None or self.hub is None:
            return

        if peer_stop_mode == "retain":
            await self._refresh_peer_stt_runtime()
        else:
            await self._refresh_peer_stt_runtime(stop_mode="release")
        self._sync_effective_hub_flags(self.settings)
        self._refresh_overlay_peer_consumers()

    async def start(self) -> None:
        try:
            await self._start_impl()
        except BaseException:
            with contextlib.suppress(BaseException):
                await self.stop()
            raise

    async def _start_impl(self) -> None:
        self.settings = self._load_or_init_settings(self.config_path)
        self._vnext_settings_authoritative = True
        self._remember_canonical_legacy_projection(self.settings)
        provisioning = self._get_local_asr_provisioning_owner()
        await provisioning.inspect_cpu()
        await provisioning.inspect_gpu(
            explicit_intent=self._selected_gpu_provider_requires_model(),
        )
        loaded_settings = self.settings
        (
            normalized_settings,
            fallback_channels,
            installation_fallback,
        ) = self._normalize_manual_local_asr_fallbacks(loaded_settings)
        if normalized_settings is not loaded_settings:
            self.settings = normalized_settings
            if not self._save_settings():
                self.settings = loaded_settings
                fallback_channels = ()
            else:
                loaded_settings.provider.stt = normalized_settings.provider.stt
                loaded_settings.provider.peer_stt = normalized_settings.provider.peer_stt
                self.settings = loaded_settings
        self.settings.ui.overlay_enabled = False
        self.settings.ui.peer_translation_enabled = False
        self._sync_overlay_calibration_cache(self.settings)
        self._overlay_calibration_draft = None
        self.app.set_locale(self.settings.ui.locale)
        self._sync_ui_from_settings()
        self._notify_manual_local_asr_fallback(
            fallback_channels,
            installation_fallback=installation_fallback,
        )
        with contextlib.suppress(Exception):
            apply_locale = getattr(self.app, "apply_locale", None)
            if callable(apply_locale):
                apply_locale()

        runtime_logging = self.runtime_logging
        runtime_logging.set_mode(SessionLoggingMode.BASIC)

        self.app.attach_runtime_log_sink(runtime_logging)

        await self._init_pipeline()
        self._sync_local_stt_notice()

        assert self.hub is not None

        stt_provider = self.settings.provider.stt.value
        if self._stt_provider_requires_secret(self.settings.provider.stt):
            stt_key_map = {"qwen_asr": self._get_alibaba_verified_key()}
            stt_verified_key = stt_key_map.get(stt_provider, stt_provider)
            stt_verified = getattr(self.settings.api_key_verified, stt_verified_key, False)
            stt_needs_key = (not self._hub_has_stt_provider("self")) or (not stt_verified)
        else:
            stt_needs_key = False
        self.app.set_dashboard_stt_needs_key(stt_needs_key)

        llm_provider = self.settings.provider.llm.value
        if self._llm_provider_requires_secret(self.settings.provider.llm):
            llm_key_map = {
                "gemini": "google",
                "openrouter": "openrouter",
                "deepseek": "deepseek",
                "qwen": self._get_alibaba_verified_key(),
            }
            llm_verified_key = llm_key_map.get(llm_provider, llm_provider)
            llm_verified = getattr(self.settings.api_key_verified, llm_verified_key, False)
            translation_needs_key = (
                False
                if self._managed_openrouter_can_attempt_translation()
                else (self.hub.llm is None) or (not llm_verified)
            )
        else:
            translation_needs_key = False
        self.app.set_dashboard_translation_needs_key(translation_needs_key)
        self.app.set_dashboard_translation_enabled(False)
        self.app.set_dashboard_stt_enabled(False)
        self.hub.translation_enabled = False
        await self.hub.start(auto_flush_osc=True)

        bridge = self._create_ui_event_bridge(runtime_logging=runtime_logging)
        self._start_ui_event_bridge_task(bridge)
        await self._wait_for_ui_event_bridge_started()
        await self._sync_clipboard_watcher()

    async def refresh_openrouter_usage_after_launch(self) -> bool:
        await self._refresh_managed_trial_usage_state_impl(auto_show_founder_letter=True)
        return is_effectively_exhausted(self._managed_trial_usage_metadata)

    async def prepare_runtime_after_launch(self) -> None:
        self._schedule_process_discovery_idle_preparation()
        self._schedule_vrchat_osc_presence_probe(force=True)
        await self.preload_saved_gpu_device_discovery()

    async def preload_saved_gpu_device_discovery(self) -> tuple[GpuWorkerDevice, ...]:
        if self.settings is None:
            return ()
        if self.settings.provider.stt != STTProviderName.LOCAL_QWEN_GPU and (
            self.settings.provider.peer_stt != STTProviderName.LOCAL_QWEN_GPU
        ):
            return ()
        return await self.ensure_gpu_device_discovery(origin="startup")

    def _schedule_process_discovery_idle_preparation(self) -> None:
        if self._process_idle_preparation_scheduled:
            return
        self._process_idle_preparation_scheduled = True
        with contextlib.suppress(Exception):
            self.app.schedule_task(self._prepare_process_discovery_idle)

    async def _prepare_process_discovery_idle(self) -> None:
        with contextlib.suppress(Exception):
            await asyncio.to_thread(lambda: tuple(PsutilCurrentUserProcessSnapshots().snapshots()))

    def _hub_has_stt_provider(self, channel: str) -> bool:
        if self.hub is None:
            return False
        return bool(self.hub.has_stt_provider(channel))

    def _hub_local_asr_provider_runtime(self) -> LocalASRProviderRuntimePort | None:
        return (
            getattr(self.hub, "local_asr_provider_runtime", None) if self.hub is not None else None
        )

    def _set_gpu_ui_state(
        self,
        state: str,
        *,
        progress_percent: int | None = None,
        publish_notice: bool = False,
        origin: str = "runtime",
    ) -> None:
        self._gpu_ui_state = state
        fields = [f"state={state}", f"origin={origin}"]
        if progress_percent is not None:
            fields.append(f"progress_percent={progress_percent}")
        self.log_detailed(f"[GPU ASR] {' '.join(fields)}")
        devices = tuple(
            GpuDeviceOption(
                device_id=device.device_id,
                display_name=device.description.strip() or device.name,
                backend_name=device.name,
            )
            for device in self._gpu_devices
        )
        action_by_state: dict[str, GpuNoticeAction] = {
            "discovery_failed": "rediscover",
            "activation_failed": "restart",
        }
        notice = (
            GpuDashboardNotice(
                status=state,
                progress_percent=progress_percent,
                action=action_by_state.get(state),
            )
            if publish_notice
            else None
        )
        self.app.set_dashboard_gpu_state(
            devices=devices,
            state=state,
            progress_percent=progress_percent,
            notice=notice,
            publish_notice=publish_notice,
        )

    async def handle_gpu_notice_action(self, action: GpuNoticeAction) -> None:
        if action in {"install", "repair", "reinstall"}:
            await self.install_or_repair_gpu_model()
            return
        if action == "rediscover":
            await self.ensure_gpu_device_discovery(force=True, origin="manual_rediscovery")
            if self._gpu_discovery_failed:
                self._set_gpu_ui_state(
                    self._gpu_discovery_failure_state or "discovery_failed",
                    publish_notice=True,
                    origin="manual_rediscovery",
                )
            elif not self._gpu_devices:
                self._set_gpu_ui_state(
                    "unsupported",
                    publish_notice=True,
                    origin="manual_rediscovery",
                )
            return
        if action == "restart":
            await self.retry_gpu_activation()

    async def ensure_gpu_device_discovery(
        self,
        *,
        force: bool = False,
        origin: str = "settings",
    ) -> tuple[GpuWorkerDevice, ...]:
        owned_runtime = self._hub_local_asr_provider_runtime()
        if owned_runtime is None:
            raise RuntimeError("local ASR provider runtime is unavailable")
        self._gpu_discovery_origin = origin
        self._set_gpu_ui_state("discovering", origin=origin)
        snapshot = await owned_runtime.discover_gpu(force=force)
        self._on_local_asr_provider_runtime_state_changed(snapshot)
        if snapshot.gpu.phase == "unsupported":
            self._set_gpu_ui_state("unsupported", origin=origin)
        elif snapshot.gpu.phase == "failed":
            self._set_gpu_ui_state(
                "discovery_failed",
                publish_notice=True,
                origin=origin,
            )
        else:
            self._set_gpu_ui_state(self._gpu_idle_ui_state(), origin=origin)
        return snapshot.gpu.devices

    def _get_local_asr_provisioning_owner(self) -> LocalASRProvisioningPort:
        if self.local_asr_provisioning is None:
            self.local_asr_provisioning = create_local_asr_provisioning_owner(
                state_changed=self._on_local_asr_provisioning_state_changed,
                diagnostic_sink=self._on_local_asr_provisioning_diagnostic,
            )
        return self.local_asr_provisioning

    def _on_local_asr_provisioning_state_changed(
        self,
        snapshot: LocalASRProvisioningSnapshot,
    ) -> None:
        self._sync_local_cpu_auto_availability(snapshot.cpu_auto_available)
        activity = snapshot.activity_for("gpu")
        if activity is not None:
            self._set_gpu_ui_state(
                "installing",
                progress_percent=activity.progress_percent,
                publish_notice=True,
                origin=activity.origin,
            )
        self._sync_local_stt_notice()

    def _on_local_asr_provisioning_diagnostic(
        self,
        diagnostic: LocalASRProvisioningDiagnostic,
    ) -> None:
        fields = [
            f"model={diagnostic.model_id or 'unknown'}",
            f"origin={diagnostic.origin or 'runtime'}",
            f"outcome={diagnostic.outcome or 'observed'}",
        ]
        if diagnostic.elapsed_seconds is not None:
            fields.append(f"elapsed_seconds={diagnostic.elapsed_seconds:.3f}")
        if diagnostic.failure_type is not None:
            fields.append(f"failure_type={diagnostic.failure_type}")
        level = logging.ERROR if diagnostic.outcome == "failed" else logging.INFO
        self.log_basic(f"[LocalASR][{diagnostic.event.title()}] {' '.join(fields)}", level=level)

    def _gpu_idle_ui_state(self) -> str:
        snapshot = self._get_local_asr_provisioning_owner().snapshot
        status = snapshot.state_for(LOCAL_QWEN_GPU_MODEL_ID).status
        if status in {"not_requested", "missing"}:
            return "not_installed"
        if status in {"invalid", "download_failed", "cancelled"}:
            return "invalid"
        if status == "downloading":
            return "installing"
        return "installed"

    async def _validate_gpu_activation(self) -> bool:
        if self.settings is None:
            return False
        owned_runtime = self._hub_local_asr_provider_runtime()
        if owned_runtime is None:
            raise RuntimeError("local ASR provider runtime is unavailable")
        self._set_gpu_ui_state("validating", origin="activation")
        snapshot = await owned_runtime.inspect_gpu_readiness(
            explicit_intent=True,
            device_id=self.settings.stt.gpu_device_id,
        )
        self._on_local_asr_provider_runtime_state_changed(snapshot)
        phase = snapshot.gpu.phase
        if phase in {"available", "ready"}:
            self._set_gpu_ui_state(
                "ready" if phase == "ready" else "loading",
                origin="activation",
            )
            return True
        state_by_phase = {
            "unsupported": "unsupported",
            "not_installed": "not_installed",
            "invalid": "invalid",
            "failed": (
                "unavailable_device"
                if snapshot.gpu.failure_code == "saved_device_missing"
                else "activation_failed"
            ),
        }
        self._set_gpu_ui_state(
            state_by_phase.get(phase, "activation_failed"),
            publish_notice=True,
            origin="activation",
        )
        return False

    def _selected_gpu_provider_requires_model(self) -> bool:
        return bool(
            self.settings is not None
            and (
                self.settings.provider.stt == STTProviderName.LOCAL_QWEN_GPU
                or self.settings.provider.peer_stt == STTProviderName.LOCAL_QWEN_GPU
            )
        )

    async def install_selected_gpu_model_if_needed(self) -> bool:
        if not self._selected_gpu_provider_requires_model():
            return False
        provisioning = self._get_local_asr_provisioning_owner()
        if provisioning.snapshot.activity_for("gpu") is not None:
            return False
        snapshot = await provisioning.inspect_gpu(
            explicit_intent=True,
            verify_checksums=False,
        )
        if not self._selected_gpu_provider_requires_model():
            return False
        if snapshot.state_for(LOCAL_QWEN_GPU_MODEL_ID).status == "ready":
            return False
        await self.install_or_repair_gpu_model(origin="settings_exit")
        return True

    async def install_or_repair_gpu_model(self, *, origin: str = "manual") -> None:
        provisioning = self._get_local_asr_provisioning_owner()
        if provisioning.snapshot.activity_for("gpu") is not None:
            return
        self._set_gpu_ui_state(
            "installing",
            progress_percent=0,
            publish_notice=True,
            origin=origin,
        )
        task = provisioning.start_install(
            LocalASRInstallRequest(
                backend="gpu",
                model_ids=(LOCAL_QWEN_GPU_MODEL_ID,),
                locale=self.settings.ui.locale if self.settings is not None else None,
                origin=origin,
                explicit_gpu_intent=True,
            )
        )
        try:
            result = await task
            if result.cancelled:
                return
            if result.failed_model_ids:
                self._set_gpu_ui_state(
                    "install_failed",
                    publish_notice=True,
                    origin=origin,
                )
                return
            pending = self._gpu_pending_enable_channels
            self._set_gpu_ui_state("installed", origin=origin)
            if pending:
                await self.retry_gpu_activation()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log_detailed(
                "[GPU ASR] model_install failure=unexpected",
                level=logging.WARNING,
                exception=exc,
            )
            self._set_gpu_ui_state(
                "install_failed",
                publish_notice=True,
                origin=origin,
            )

    async def retry_gpu_activation(self) -> None:
        async with self._get_gpu_provider_recovery_lock():
            await self._execute_gpu_provider_recovery_retry()

    async def _execute_gpu_provider_recovery_retry(self) -> None:
        if self.settings is None:
            return
        owned_runtime = self._hub_local_asr_provider_runtime()
        if owned_runtime is None:
            raise RuntimeError("local ASR provider runtime is unavailable")
        desired_channels = frozenset(
            {
                *self._desired_gpu_channels(self.settings),
                *self._gpu_pending_enable_channels,
            }
        )
        recovery, peer_config = self._build_gpu_recovery_request(
            self.settings,
            desired_channels,
            reason="manual_retry",
        )
        if not recovery.channels:
            return
        try:
            snapshot = await owned_runtime.recover_gpu(
                recovery,
                quiesce=self._suspend_gpu_provider_consumers,
            )
            recovered_channels = frozenset(item.request.channel for item in recovery.channels)
            if snapshot.gpu.retry_required or not recovered_channels.issubset(
                snapshot.gpu.active_channels
            ):
                self._on_local_asr_provider_runtime_state_changed(snapshot)
                return
            await self._resume_gpu_provider_consumers(
                settings=self.settings,
                channels=recovered_channels,
                peer_config=peer_config,
                recovery=recovery,
            )
            self._gpu_pending_enable_channels = frozenset(
                channel
                for channel in self._gpu_pending_enable_channels
                if channel not in recovered_channels
            )
            self._set_gpu_ui_state("ready", origin="manual_retry")
        except Exception:
            self._set_gpu_ui_state(
                "activation_failed",
                publish_notice=True,
                origin="manual_retry",
            )
        finally:
            self._abort_provider_recoveries(recovery)

    def _create_ui_event_bridge(self, *, runtime_logging) -> object:  # noqa: ANN001
        assert self.hub is not None
        return self.app.create_ui_event_bridge(
            event_queue=self.hub.ui_events,
            runtime_logging=runtime_logging,
        )

    def _start_ui_event_bridge_task(self, bridge: object) -> None:
        assert self.hub is not None
        self._ui_event_bridge = bridge
        self._bridge_task = self.hub.output_runtime.start_ui_event_bridge(bridge)

    async def _wait_for_ui_event_bridge_started(self) -> None:
        if self.hub is None:
            raise RuntimeError("UI Event Bridge owner is unavailable")
        await self.hub.output_runtime.wait_for_ui_event_bridge_started()

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
                    get_effective_local_qwen_hotwords(
                        build_custom_vocabulary_runtime_config(settings),
                        settings.languages.source_language,
                    )
                ),
            )
        return (
            settings.stt.custom_vocabulary_enabled,
            tuple(
                get_effective_custom_terms(
                    build_custom_vocabulary_runtime_config(settings),
                    settings.languages.source_language,
                )
            ),
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
            FIXED_TRANSLATION_POLICY.fast_translation_enabled,
            settings.stt.low_latency_merge_gap_ms,
            settings.stt.low_latency_spec_retry_max,
            settings.stt.low_latency_vad_hangover_ms,
            settings.stt.drain_timeout_s,
            settings.audio.ring_buffer_ms,
            settings.audio.internal_sample_rate_hz,
            settings.audio.internal_channels,
            (
                settings.stt.gpu_device_id
                if settings.provider.stt == STTProviderName.LOCAL_QWEN_GPU
                else None
            ),
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
            (
                settings.qwen_asr_stt.endpoint
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
            (
                settings.stt.gpu_device_id
                if settings.provider.stt == STTProviderName.LOCAL_QWEN_GPU
                else None
            ),
        )

    @staticmethod
    def _self_capture_vad_signature(settings: AppSettings) -> tuple[object, ...]:
        return (
            settings.audio.input_host_api,
            settings.audio.input_device,
            settings.stt.vad_speech_threshold,
            settings.stt.low_latency_vad_hangover_ms,
            settings.audio.ring_buffer_ms,
            settings.audio.internal_sample_rate_hz,
            settings.audio.internal_channels,
            settings.stt.gpu_device_id,
        )

    def _build_stt_runtime_signature(self, settings: AppSettings) -> tuple[object, ...]:
        return self._build_self_stt_runtime_signature(settings)

    def _build_peer_stt_runtime_signature(self, settings: AppSettings) -> tuple[object, ...]:
        return self._build_peer_runtime_config(settings).runtime_signature

    def _build_peer_stt_provider_signature(
        self,
        settings: AppSettings,
        *,
        canonical_settings: AppSettingsVNext | None = None,
    ) -> tuple[object, ...]:
        return build_peer_stt_provider_signature_from_vnext(
            canonical_settings or self._canonical_vnext_settings_for(settings)
        )

    def _managed_openrouter_can_attempt_translation(self) -> bool:
        return bool(
            self.settings is not None
            and self.settings.provider.llm == LLMProviderName.OPENROUTER
            and self.settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
            and self.hub is not None
            and self.hub.llm is not None
        )

    def _sync_managed_auth_dashboard_notice(self) -> None:
        self.app.set_dashboard_managed_auth_pending(self._managed_trial_pending_auth)

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

    def _managed_openrouter_fallback_branch_settings_for(
        self,
        settings: AppSettings,
    ) -> AppSettings | None:
        fallback = settings.translation.fallback
        if not fallback.enabled or fallback.connection not in _MANAGED_OPENROUTER_CONNECTIONS:
            return None
        llm_model = _MANAGED_OPENROUTER_MODEL_BY_TRANSLATION_MODEL.get(fallback.model)
        if llm_model is None:
            return None
        branch_settings = copy.deepcopy(settings)
        branch_settings.provider.llm = LLMProviderName.OPENROUTER
        branch_settings.translation.connection = fallback.connection
        branch_settings.openrouter.llm_model = llm_model
        branch_settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
        alias = get_openrouter_selection_alias_for_model_and_source(
            llm_model.value,
            OpenRouterCredentialSource.MANAGED.value,
        )
        branch_settings.openrouter.selection_alias = (
            OpenRouterSelectionAlias(alias) if alias is not None else None
        )
        return branch_settings

    def _managed_openrouter_branch_settings_for(
        self,
        settings: AppSettings,
    ) -> tuple[AppSettings, ...]:
        branches: list[AppSettings] = []
        if (
            settings.provider.llm == LLMProviderName.OPENROUTER
            and settings.translation.connection in _MANAGED_OPENROUTER_CONNECTIONS
            and settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
        ):
            branches.append(settings)
        fallback_branch = self._managed_openrouter_fallback_branch_settings_for(settings)
        if fallback_branch is not None:
            branches.append(fallback_branch)
        return tuple(branches)

    def _managed_openrouter_release_settings(self) -> AppSettings | None:
        if self.settings is None:
            return None
        if not self._managed_openrouter_selected():
            return None
        return self.settings

    def _should_show_managed_auth_pending_before_prepare(self) -> bool:
        if self.settings is None:
            return False
        if not self._managed_openrouter_selected():
            return False
        try:
            secrets = create_secret_store(self.settings.secrets, config_path=self.config_path)
            resolution = resolve_openrouter_credentials(
                build_openrouter_credential_runtime_config(self.settings),
                secrets=secrets,
                request_intent="TRANS",
            )
            if resolution.api_key is None:
                return True
        except Exception:
            return True
        return False

    def _managed_openrouter_selected(self) -> bool:
        return bool(
            self.settings is not None
            and self.settings.provider.llm == LLMProviderName.OPENROUTER
            and self.settings.translation.connection in _MANAGED_OPENROUTER_CONNECTIONS
            and self.settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
        )

    def _managed_openrouter_local_key_available(self) -> bool:
        if self.settings is None:
            return False
        if not self._managed_openrouter_selected():
            return False
        try:
            secrets = create_secret_store(self.settings.secrets, config_path=self.config_path)
            resolution = resolve_openrouter_credentials(
                build_openrouter_credential_runtime_config(self.settings),
                secrets=secrets,
                request_intent="TRANS",
            )
            if resolution.api_key is None:
                return False
        except Exception:
            return False
        return True

    def dashboard_managed_auth_action(self) -> str:
        if not self._managed_openrouter_selected():
            return "continue"
        if self._discord_managed_auth_in_progress or self._managed_trial_pending_auth:
            return "in_progress"
        if self._managed_openrouter_local_key_available():
            return "continue"
        return "prompt"

    def dashboard_managed_auth_prompt_kind(self) -> str:
        if self._managed_china_auth_relevant_for_translation_enable():
            return "qq"
        return "discord"

    def _managed_china_auth_relevant_for_translation_enable(self) -> bool:
        if self.settings is None:
            return False
        return self.settings.translation.connection == TranslationConnection.MANAGED_CHINA

    def _show_qq_managed_auth_dialog(self) -> None:
        if not self._managed_china_auth_relevant_for_translation_enable():
            return
        show_dialog = getattr(self.app, "show_qq_managed_auth_dialog", None)
        if callable(show_dialog):
            show_dialog()

    def _managed_auth_claim_guard_for_settings(
        self,
        settings: AppSettings,
    ) -> ManagedAuthClaimGuard:
        secret_store = create_secret_store(settings.secrets, config_path=self.config_path)
        secret_store_port = create_sync_secret_store_adapter(secret_store)
        managed_state = build_managed_identity_state_port(
            settings,
            self._managed_identity_persistence_callback(settings),
        )
        return ManagedAuthClaimGuard(
            managed_state=managed_state,
            secret_store=secret_store_port,
        )

    async def start_qq_managed_auth_from_dialog(
        self,
        *,
        qq_identity: str,
        credential: str,
    ) -> bool | tuple[str, dict[str, object]]:
        if not self._managed_china_auth_relevant_for_translation_enable() or self.settings is None:
            return "qq_auth.error.retry", {}
        service = self._managed_openrouter_release_service
        broker_client = getattr(service, "client", None)
        if broker_client is None:
            return "qq_auth.error.retry", {}
        secret_store = create_secret_store(self.settings.secrets, config_path=self.config_path)
        secret_store_port = create_sync_secret_store_adapter(secret_store)
        managed_state = build_managed_identity_state_port(
            self.settings,
            self._managed_identity_persistence_callback(self.settings),
        )
        auth_service = QqManagedAuthService(
            broker_client=broker_client,
            secret_store=secret_store_port,
            managed_state=managed_state,
            claim_guard=ManagedAuthClaimGuard(
                managed_state=managed_state,
                secret_store=secret_store_port,
            ),
        )
        result = await auth_service.authenticate(
            QqManagedAuthRequest(
                qq_identity=qq_identity,
                credential=credential,
                asserted_at=_github_star_prompt_utc_timestamp(),
                metadata={"flow": "qq_managed_auth_dialog"},
            )
        )
        if _settings_mutation_committed(result):
            self._set_managed_trial_pending_auth(False)
            if self.hub is not None and self.hub.llm is None:
                await self._rebuild_llm_provider()
            else:
                self._schedule_managed_trial_usage_refresh()
            return True
        message = result.message
        if message is None:
            return "qq_auth.error.retry", {}
        return (
            QQ_AUTH_DIALOG_MESSAGE_KEY_BY_SERVICE_KEY.get(message.key, message.key),
            dict(message.params),
        )

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
        release_service = self._managed_openrouter_release_service
        if release_service is None or self.settings is None:
            self._discord_managed_auth_in_progress = False
            self._set_managed_trial_pending_auth(False)
            self._show_short_message("discord_auth.error.retry")
            return False

        previous_callback = self._discord_managed_auth_callback_received_hook
        self._discord_managed_auth_callback_received_hook = on_callback_received
        self._discord_managed_auth_in_progress = True
        self._set_managed_trial_pending_auth(True)
        try:
            if not self._discord_release_service_supports_transaction_auth(release_service):
                return await self._start_discord_managed_auth_via_release_service(
                    release_service,
                    referral_id=referral_id,
                )
            updated = copy.deepcopy(self.settings)
            secret_store = create_secret_store(updated.secrets, config_path=self.config_path)
            secret_store_port = create_sync_secret_store_adapter(secret_store)
            managed_state = build_managed_identity_state_port(
                updated,
                self._managed_identity_persistence_callback(updated),
            )
            identity = ManagedIdentityPreflightAdapter(
                managed_state=managed_state,
                secrets=secret_store,
            )
            discord_auth = DiscordOAuthAuthAdapter(
                identity=identity,
                client=release_service.client,
                app_version=release_service.app_version,
                raw_hardware_fingerprint_provider=release_service.raw_hardware_fingerprint_provider,
                hardware_hash_provider=getattr(
                    release_service,
                    "_legacy_hardware_hash_provider",
                    None,
                ),
                oauth_runtime=release_service.oauth_runtime,
                listener_factory=release_service.discord_oauth_listener_factory,
                callback_runner=release_service.discord_oauth_callback_runner,
                referral_id=referral_id,
                on_callback_received=on_callback_received,
            )
            broker = DiscordManagedBrokerClientAdapter(
                identity=identity,
                client=release_service.client,
                openrouter_config=release_service.openrouter_config,
                app_version=release_service.app_version,
                signed_at_provider=release_service.signed_at_provider,
            )
            auth_service = ManagedConnectionAuthService(
                local_identity=identity,
                discord_auth=discord_auth,
                broker_client=broker,
                secret_store=secret_store_port,
                settings_repository=self._legacy_settings_patch_repository(
                    base_settings=self.settings,
                    committed_settings=updated,
                    surface="managed_connection_auth",
                ),
                claim_guard=ManagedAuthClaimGuard(
                    managed_state=managed_state,
                    secret_store=secret_store_port,
                ),
            )
            result = await auth_service.authorize(
                ManagedConnectionAuthRequest(
                    local_secret_key=OPENROUTER_MANAGED_API_KEY_SECRET,
                    settings_values=_managed_connection_auth_settings_values(updated),
                    expected_settings_revision=None,
                    reason="managed_connection_auth",
                    correlation_id=None,
                    broker_metadata={"flow": "managed_connection_auth"},
                )
            )
            self._complete_canonical_mutation()
            self.last_settings_mutation_result = result
            if result.status == TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING:
                self.settings = updated
            if _settings_mutation_committed(result):
                issue = broker.last_issue_response
                self.settings = updated
                self.last_discord_managed_auth_referral_bonus_applied = bool(
                    getattr(issue, "referral_bonus_applied", False)
                )
                if self.hub is None:
                    self._show_short_message("discord_auth.error.retry")
                    return False
                await self._rebuild_llm_provider()
                if self.hub.llm is None:
                    self._show_short_message("discord_auth.error.retry")
                    return False
                result_referral_id = normalize_owned_referral_id(
                    getattr(issue, "referral_id", None)
                )
                self._set_managed_usage_view_state(
                    visible=True,
                    remaining_percent=None,
                    referral_id=result_referral_id or self._current_owned_referral_id(),
                    pass_status=getattr(issue, "pass_status", None),
                )
                self._schedule_managed_trial_usage_refresh()
                return True

            message = result.message
            message_key = message.key if message is not None else "discord_auth.error.retry"
            message_kwargs = dict(message.params) if message is not None else {}
            diagnostics = result.diagnostics
            error_class = getattr(diagnostics, "category", None)
            self.log_basic(
                "[ManagedAuth] Discord auth failed: "
                f"message_key={message_key} class={error_class or 'unknown'}",
                level=logging.ERROR,
            )
            self._show_short_message(
                message_key,
                **message_kwargs,
            )
            return False
        finally:
            if self._discord_managed_auth_callback_received_hook is on_callback_received:
                self._discord_managed_auth_callback_received_hook = previous_callback
            self._discord_managed_auth_in_progress = False
            self._set_managed_trial_pending_auth(False)

    def _discord_release_service_supports_transaction_auth(self, release_service: object) -> bool:
        return all(
            hasattr(release_service, attr)
            for attr in (
                "app_version",
                "client",
                "discord_oauth_callback_runner",
                "discord_oauth_listener_factory",
                "oauth_runtime",
                "openrouter_config",
                "signed_at_provider",
            )
        )

    async def _start_discord_managed_auth_via_release_service(
        self,
        release_service: object,
        *,
        referral_id: str | None,
    ) -> bool:
        claim_guard: ManagedAuthClaimGuard | None = None
        if self.settings is not None:
            try:
                claim_guard = self._managed_auth_claim_guard_for_settings(self.settings)
                claim_result = await claim_guard.preflight(MANAGED_AUTH_CLAIM_SOURCE_DISCORD)
            except Exception:
                self._show_short_message("discord_auth.error.retry")
                return False
            if claim_result is not None:
                self.last_settings_mutation_result = claim_result
                message = claim_result.message
                message_key = message.key if message is not None else "discord_auth.error.retry"
                message_kwargs = dict(message.params) if message is not None else {}
                self._show_short_message(message_key, **message_kwargs)
                return False
        try:
            result = await release_service.prepare_for_translation(referral_id=referral_id)
        except Exception as exc:
            self.log_basic(
                f"[ManagedAuth] Discord auth start failed: {exc}",
                level=logging.ERROR,
            )
            self._show_short_message("discord_auth.error.retry")
            return False

        if result.behavior == ManagedOpenRouterReleaseBehavior.READY and result.local_key_available:
            if claim_guard is not None:
                with contextlib.suppress(Exception):
                    claim_guard.record_success(MANAGED_AUTH_CLAIM_SOURCE_DISCORD)
                    claim_guard.managed_state.persist()
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
            result_referral_id = normalize_owned_referral_id(getattr(result, "referral_id", None))
            self._set_managed_usage_view_state(
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
        return self._get_github_star_prompt_owner().initial_launch_gate_satisfied(settings)

    def should_show_github_star_prompt(self, *, now: datetime | None = None) -> bool:
        return self._get_github_star_prompt_owner().should_show(now=now)

    def _get_github_star_prompt_persistence_lock(self) -> asyncio.Lock:
        return self._get_github_star_prompt_owner().persistence_lock

    def _github_star_prompt_state_snapshot(self, settings: AppSettings) -> tuple[object, ...]:
        return self._get_github_star_prompt_owner().state_snapshot(settings)

    def _restore_github_star_prompt_state_snapshot(
        self,
        settings: AppSettings,
        snapshot: tuple[object, ...],
    ) -> None:
        self._get_github_star_prompt_owner().restore_state_snapshot(settings, snapshot)

    def _log_github_star_prompt_save_failure(
        self,
        failure_context: str,
        exc: Exception,
    ) -> None:
        self.log_basic(
            "[GitHubStar] Failed to persist prompt "
            f"{failure_context}: exception_class={type(exc).__name__}",
            level=logging.WARNING,
        )

    async def _persist_order24_state_mutation(
        self,
        *,
        base_settings: AppSettings,
        committed_settings: AppSettings,
    ) -> bool:
        patch_values = build_ui_prompt_clipboard_state_settings_path_patch(
            base_settings,
            committed_settings,
        )
        if not patch_values:
            return True
        result = await self._mutate_order24_settings_patch(
            patch_values=patch_values,
            committed_settings=copy.deepcopy(committed_settings),
            runtime_apply=_ControllerNoopRuntimeApply(),
        )
        self.last_settings_mutation_result = result
        return _settings_mutation_committed(result)

    async def _persist_github_star_prompt_mutation(
        self,
        *,
        failure_context: str,
        mutate,
    ) -> bool:
        return await self._get_github_star_prompt_owner().persist_mutation(
            failure_context=failure_context,
            mutate=mutate,
        )

    async def persist_github_star_prompt_opened(
        self,
        *,
        opened_at: datetime | None = None,
        should_open: Callable[[], bool] | None = None,
    ) -> bool:
        return await self._get_github_star_prompt_owner().persist_opened(
            opened_at=opened_at,
            should_open=should_open,
        )

    async def persist_github_star_prompt_eligible_launch(self) -> bool:
        return await self._get_github_star_prompt_owner().persist_eligible_launch()

    def _run_github_star_prompt_persistence_sync(self, coro) -> bool:  # noqa: ANN001
        return self._get_github_star_prompt_owner().run_sync(coro)

    def record_github_star_prompt_opened(self, *, opened_at: datetime | None = None) -> bool:
        return self._get_github_star_prompt_owner().record_opened(opened_at=opened_at)

    async def persist_github_star_prompt_clicked(self) -> bool:
        return await self._get_github_star_prompt_owner().persist_clicked()

    def record_github_star_prompt_clicked(self) -> bool:
        return self._get_github_star_prompt_owner().record_clicked()

    async def persist_github_star_prompt_translation_success_observed(self) -> bool:
        return await self._get_github_star_prompt_owner().persist_translation_success_observed()

    def record_github_star_prompt_translation_success_observed(self) -> bool:
        return self._get_github_star_prompt_owner().record_translation_success_observed()

    def _get_github_star_prompt_runtime(self) -> GithubStarPromptRuntime:
        return self._get_github_star_prompt_owner().get_runtime()

    @property
    def _github_star_prompt_runtime(self) -> GithubStarPromptRuntime | None:
        owner = self._github_star_prompt_owner
        return owner.runtime if owner is not None else None

    @_github_star_prompt_runtime.setter
    def _github_star_prompt_runtime(
        self,
        runtime: GithubStarPromptRuntime | None,
    ) -> None:
        self._get_github_star_prompt_owner().runtime = runtime

    def _get_github_star_prompt_owner(self) -> GithubStarPromptOwner:
        owner = self._github_star_prompt_owner
        if owner is None:
            owner = GithubStarPromptOwner(
                settings_provider=lambda: self.settings,
                persist_settings_state=lambda base, committed: (
                    self._persist_order24_state_mutation(
                        base_settings=cast(AppSettings, base),
                        committed_settings=cast(AppSettings, committed),
                    )
                ),
                is_eligible=self.is_github_star_prompt_eligible,
                has_user_owned_cloud_connection=lambda settings: (
                    self._github_star_prompt_settings_has_user_owned_cloud_connection(
                        cast(AppSettings | None, settings)
                    )
                ),
                log_save_failure=self._log_github_star_prompt_save_failure,
                runtime_diagnostics_sink=self._github_star_prompt_runtime_diagnostics_sink,
                translation_success_observation=lambda: (
                    self.persist_github_star_prompt_translation_success_observed()
                ),
            )
            self._github_star_prompt_owner = owner
        return owner

    def _github_star_prompt_runtime_diagnostics_sink(
        self,
        event: str,
        metadata: Mapping[str, object],
    ) -> None:
        self.log_detailed(
            f"[Lifecycle][GithubStarPromptRuntime] event={event} metadata={dict(metadata)}",
            level=logging.WARNING,
        )

    async def _close_github_star_prompt_runtime_for_release(
        self,
        failures: list[Exception],
    ) -> None:
        owner = self._github_star_prompt_owner
        if owner is None:
            return
        try:
            await owner.close()
        except Exception as exc:
            failures.append(exc)

    async def _close_app_github_star_prompt_runtime_for_release(
        self,
        failures: list[Exception],
    ) -> None:
        close_prompt_runtime = getattr(self.app, "close_after_launch_tasks", None)
        if not callable(close_prompt_runtime):
            close_prompt_runtime = getattr(self.app, "close_github_star_prompt_runtime", None)
        if not callable(close_prompt_runtime):
            return
        try:
            result = close_prompt_runtime()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            failures.append(exc)

    def schedule_github_star_prompt_translation_success_observed(self) -> bool:
        return self._get_github_star_prompt_owner().schedule_translation_success_observed()

    async def _drain_github_star_prompt_translation_success_observation(self) -> None:
        owner = self._github_star_prompt_owner
        if owner is not None:
            await owner.drain_translation_success_observation()

    def _get_telemetry_service(self) -> TranslationSuccessTelemetryService:
        return TranslationSuccessTelemetryService(
            self.telemetry_client,
            diagnostics_sink=self._telemetry_diagnostics_sink,
        )

    def _telemetry_diagnostics_sink(
        self,
        event: str,
        metadata: Mapping[str, object],
    ) -> None:
        self.log_detailed(
            f"[Telemetry] event={event} metadata={dict(metadata)}",
            level=logging.INFO,
        )

    async def record_telemetry_translation_success_day(
        self,
    ) -> TranslationSuccessTelemetryResult:
        if self.settings is None:
            return TranslationSuccessTelemetryResult(status="skipped_no_settings")

        async def _persist(updated: AppSettings) -> bool:
            await self.apply_settings(updated)
            return self.last_settings_mutation_result is not None and _settings_mutation_committed(
                self.last_settings_mutation_result
            )

        return await self._get_telemetry_service().record_translation_success_day(
            self.settings,
            persist_sent_date=_persist,
        )

    async def _preserve_github_star_prompt_observation_before_settings_replace(
        self,
        replacement_settings: AppSettings,
    ) -> None:
        await self._get_github_star_prompt_owner().preserve_before_settings_replace(
            replacement_settings
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
        self.app.set_settings_managed_key_state(
            visible=visible,
            remaining_percent=remaining_percent,
            referral_id=normalized_referral_id,
            pass_status=effective_pass_status,
        )

    def _managed_key_card_visible_from_settings(self) -> bool:
        if self.settings is None:
            return False
        return self._managed_openrouter_selected()

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
        remaining_percent: int | None,
        current_referral_id: str | None,
    ) -> None:
        if self._shutdown_ingress_frozen:
            return
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
                if self._shutdown_ingress_frozen:
                    return
                if service is not self._managed_openrouter_release_service:
                    return
                if (
                    self.settings is None
                    or self._managed_openrouter_release_settings() is None
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
                        visible=True,
                        remaining_percent=remaining_percent,
                        referral_id=refreshed_referral_id,
                        pass_status=result.pass_status,
                    )
                    return
                self._set_managed_usage_view_state(
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

        self._get_managed_status_refresh_owner().schedule_status_refresh(_run_status_refresh)

    def _schedule_managed_trial_usage_refresh(self) -> None:
        if self._shutdown_ingress_frozen:
            return

        self._get_managed_status_refresh_owner().schedule_trial_usage_refresh(
            self._refresh_managed_trial_usage_state_best_effort
        )

    def _get_managed_status_refresh_owner(self) -> ManagedStatusRefreshOwner:
        owner = self._managed_status_refresh_owner
        if owner is None:
            owner = ManagedStatusRefreshOwner(
                diagnostics_sink=lambda event, metadata, exception: self.log_detailed(
                    "[ManagedAuth] Background refresh failed "
                    f"event={event} kind={metadata.get('kind')} "
                    f"error_type={metadata.get('error_type')}",
                    level=logging.WARNING,
                    exception=exception,
                )
            )
            self._managed_status_refresh_owner = owner
        return owner

    def _on_managed_trial_delegate_ready(self) -> None:
        if self._shutdown_ingress_frozen:
            return
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
        if self._shutdown_ingress_frozen:
            return
        if self.settings is None:
            self._clear_managed_trial_usage_metadata_cache()
            self._set_managed_trial_pending_auth(False)
            self._set_managed_usage_view_state(
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
                visible=False,
                remaining_percent=None,
                referral_id=self._current_owned_referral_id(),
            )
            return
        release_settings = self._managed_openrouter_release_settings()
        if release_settings is None:
            self._clear_managed_trial_usage_metadata_cache()
            self._set_managed_trial_pending_auth(False)
            self._set_managed_usage_view_state(
                visible=True,
                remaining_percent=None,
                referral_id=self._current_owned_referral_id(),
            )
            return

        entitlement_ref = self._sync_managed_trial_usage_metadata_scope()

        try:
            secrets = create_secret_store(self.settings.secrets, config_path=self.config_path)
            resolution = resolve_openrouter_credentials(
                build_openrouter_credential_runtime_config(release_settings), secrets=secrets
            )
        except Exception:
            resolution = None

        usage_metadata: OpenRouterKeyMetadata | None = None
        api_key = resolution.api_key if resolution is not None else None
        if api_key:
            self._set_managed_trial_pending_auth(False)
            usage_metadata = await self._get_provider_verifier().fetch_openrouter_key_metadata(
                api_key
            )
            if self._shutdown_ingress_frozen:
                return

        self._managed_trial_usage_metadata = usage_metadata
        self._managed_trial_usage_metadata_entitlement_ref = entitlement_ref

        remaining_percent = self._managed_trial_remaining_percent(usage_metadata)
        current_referral_id = self._current_owned_referral_id()
        self._set_managed_usage_view_state(
            visible=True,
            remaining_percent=remaining_percent,
            referral_id=current_referral_id,
        )

        if auto_show_founder_letter and is_effectively_exhausted(usage_metadata):
            self._disable_translation_for_managed_exhaustion(
                reopen_founder_letter=should_auto_show_founder_letter(
                    build_managed_identity_state_port(
                        self.settings,
                        lambda _settings: None,
                    ),
                    usage_metadata,
                )
            )

        self._schedule_owned_referral_id_status_refresh(
            remaining_percent=remaining_percent,
            current_referral_id=current_referral_id,
        )

    def _show_founder_letter_dialog(self) -> None:
        if self.settings is None:
            return
        if not self.app.show_founder_letter_dialog():
            return
        mark_founder_letter_shown(
            build_managed_identity_state_port(
                self.settings,
                lambda _settings: None,
            )
        )
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
        self.app.set_dashboard_translation_enabled(False)

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
        primary_uses_openrouter = settings.provider.llm == LLMProviderName.OPENROUTER
        fallback_uses_openrouter = bool(
            settings.translation.fallback.enabled
            and settings.translation.fallback.connection
            in (
                TranslationConnection.OPENROUTER,
                TranslationConnection.MANAGED,
                TranslationConnection.MANAGED_CHINA,
            )
        )
        uses_openrouter = primary_uses_openrouter or fallback_uses_openrouter
        uses_managed_openrouter = bool(
            (
                primary_uses_openrouter
                and settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
            )
            or (
                settings.translation.fallback.enabled
                and settings.translation.fallback.connection
                in (TranslationConnection.MANAGED, TranslationConnection.MANAGED_CHINA)
            )
        )
        return (
            settings.provider.llm,
            settings.llm.concurrency_limit,
            settings.gemini.llm_model if settings.provider.llm == LLMProviderName.GEMINI else None,
            (settings.openrouter.llm_model if primary_uses_openrouter else None),
            (settings.openrouter.routing_mode if uses_openrouter else None),
            (
                settings.openrouter.provider_routing
                if uses_openrouter
                else OpenRouterProviderRouting.DEFAULT
            ),
            (settings.openrouter.selected_source if primary_uses_openrouter else None),
            (settings.openrouter.selection_alias if primary_uses_openrouter else None),
            (
                settings.translation.fallback.enabled,
                settings.translation.fallback.model,
                settings.translation.fallback.connection,
            ),
            (settings.openrouter.broker_base_url if uses_openrouter else None),
            (_managed_openrouter_identity_signature(settings) if uses_managed_openrouter else None),
            settings.qwen.llm_model if settings.provider.llm == LLMProviderName.QWEN else None,
            settings.qwen.region if settings.provider.llm == LLMProviderName.QWEN else None,
            (
                settings.deepseek.llm_model
                if settings.provider.llm == LLMProviderName.DEEPSEEK
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
        target.openrouter.broker_base_url = source.openrouter.broker_base_url
        target.qwen.llm_model = source.qwen.llm_model
        target.qwen.region = source.qwen.region
        target.deepseek.llm_model = source.deepseek.llm_model
        target.local_llm = copy.deepcopy(source.local_llm)
        target.llm.concurrency_limit = source.llm.concurrency_limit
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
        if (
            merged.provider.peer_stt
            not in {
                STTProviderName.SONIOX,
                STTProviderName.LOCAL_QWEN_GPU,
            }
            and merged.languages.peer_source_mode == "auto"
        ):
            merged.languages.peer_source_mode = "manual"
        return merged

    def _peer_runtime_should_be_active(self, settings: AppSettings) -> bool:
        return bool(
            self._peer_translation_activation_requested_for(settings)
            and self._effective_peer_overlay_enabled_for(settings)
            and self.hub is not None
            and self._current_overlay_bridge_for_direct_runtime_command() is not None
        )

    def _active_local_asr_change(
        self,
        base_settings: AppSettings,
        next_settings: AppSettings,
    ) -> bool:
        local_providers = {*LOCAL_CPU_PROVIDERS, STTProviderName.LOCAL_QWEN_GPU.value}
        self_changed = (
            self._stt_desired
            and (
                base_settings.provider.stt.value in local_providers
                or next_settings.provider.stt.value in local_providers
            )
            and self._build_self_stt_runtime_signature(base_settings)
            != self._build_self_stt_runtime_signature(next_settings)
        )
        peer_changed = (
            (
                self._peer_runtime_should_be_active(base_settings)
                or self._peer_runtime_should_be_active(next_settings)
            )
            and (
                base_settings.provider.peer_stt.value in local_providers
                or next_settings.provider.peer_stt.value in local_providers
            )
            and self._build_peer_stt_runtime_signature(base_settings)
            != self._build_peer_stt_runtime_signature(next_settings)
        )
        return self_changed or peer_changed

    async def _compensate_failed_local_asr_settings_apply(
        self,
        *,
        base_settings: AppSettings,
        committed_settings: AppSettings,
    ) -> None:
        self._update_canonical_settings_from_legacy_delta(
            committed_settings,
            base_settings,
        )
        await asyncio.to_thread(
            self._persist_settings_at_controller_boundary,
            base_settings,
        )
        self._remember_canonical_legacy_projection(base_settings)
        await self._apply_settings_direct(
            copy.deepcopy(base_settings),
            persist=False,
            strict_runtime_errors=False,
        )
        self.settings = copy.deepcopy(base_settings)
        self.refresh_settings_projection(preserve_custom_vocab_draft=True)

    def _is_qwen_llm(self, settings: object) -> bool:
        # Boundary accessor used by the app-service runtime-apply adapters to
        # keep ``LLMProviderName`` settings-shape knowledge inside the
        # controller instead of leaking it across the app-service boundary.
        return bool(
            isinstance(settings, AppSettings) and settings.provider.llm == LLMProviderName.QWEN
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

    def _effective_overlay_target_for_start(self) -> str:
        if self._get_overlay_session_fallback_owner().active:
            return OVERLAY_TARGET_DESKTOP
        return self._overlay_target_for_settings(self.settings)

    def _clear_overlay_session_desktop_fallback(self) -> None:
        self._get_overlay_session_fallback_owner().clear()

    def _set_overlay_session_fallback_notice_active(self, active: bool) -> None:
        self._get_overlay_session_fallback_owner().publish(active)

    def _should_session_fallback_overlay_to_desktop(self, reason: str) -> bool:
        return self._get_overlay_session_fallback_owner().should_fallback(
            reason=reason,
            active_target=self._active_overlay_target,
            configured_enabled=bool(self.settings is not None and self.settings.ui.overlay_enabled),
            configured_target=self._overlay_target_for_settings(self.settings),
            desktop_target=OVERLAY_TARGET_DESKTOP,
            steamvr_target=OVERLAY_TARGET_STEAMVR,
        )

    def _new_overlay_runtime_handle(self) -> OverlayRuntimeHandle:
        runtime = OverlayRuntimeHandle(shutdown_grace_s=OVERLAY_SHUTDOWN_GRACE_S)
        self._overlay_runtime = runtime
        return runtime

    def _ensure_overlay_runtime_handle(self) -> OverlayRuntimeHandle:
        runtime = self._overlay_runtime
        if runtime is None:
            runtime = self._new_overlay_runtime_handle()
        return runtime

    def _overlay_runtime_is_current(
        self,
        runtime: OverlayRuntimeHandle,
        *,
        overlay_instance_id: str | None = None,
    ) -> bool:
        if self._overlay_runtime is not runtime:
            return False
        if overlay_instance_id is None:
            return True
        return runtime.is_current_instance_id(overlay_instance_id)

    def _overlay_runtime_has_resources(self, runtime: OverlayRuntimeHandle | None) -> bool:
        if runtime is None:
            return False
        return any(
            resource is not None
            for resource in (
                runtime.presenter,
                runtime.bridge,
                runtime.process_manager,
                runtime.diagnostics,
                runtime.renderer_events,
                runtime.start_task,
                runtime.monitor_task,
                runtime.renderer_event_task,
            )
        )

    async def _replace_hub_overlay_sink(
        self,
        overlay_sink: object | None,
        *,
        expected_current: object | None = None,
        require_match: bool = False,
    ) -> bool:
        hub = self.hub
        if hub is None:
            return False
        replace_overlay_sink = getattr(hub, "replace_overlay_sink", None)
        if callable(replace_overlay_sink):
            return bool(
                await replace_overlay_sink(
                    overlay_sink,
                    expected_current=expected_current,
                    require_match=require_match,
                )
            )
        if require_match and getattr(hub, "overlay_sink", None) is not expected_current:
            return False
        setattr(hub, "overlay_sink", overlay_sink)
        return True

    async def _close_stale_overlay_start_runtime(
        self,
        runtime: OverlayRuntimeHandle,
    ) -> None:
        stale_presenter = runtime.presenter
        stale_diagnostics = runtime.diagnostics
        try:
            await runtime.close(
                preserve_presenter_state=True,
                hub=self.hub,
                emit_shutdown=False,
            )
        except Exception as exc:
            self.log_detailed(
                "[Overlay] Stale overlay start cleanup reported failure",
                level=logging.WARNING,
                exception=exc,
            )
        if self.hub is not None:
            if getattr(self.hub, "overlay_sink", None) is stale_presenter:
                await self._replace_hub_overlay_sink(
                    None,
                    expected_current=stale_presenter,
                    require_match=True,
                )
            if getattr(self.hub, "overlay_diagnostics", None) is stale_diagnostics:
                self.hub.overlay_diagnostics = None

    def _overlay_runtime_is_active(self) -> bool:
        runtime = self._overlay_runtime
        start_task = runtime.start_task if runtime is not None else None
        return bool(
            self.overlay_state in {"starting", "connected"}
            or (runtime is not None and runtime.bridge is not None)
            or (runtime is not None and runtime.process_manager is not None)
            or (start_task is not None and not start_task.done())
        )

    def _current_overlay_presenter_for_direct_runtime_command(self) -> OverlayPresenter | None:
        runtime = self._overlay_runtime
        if runtime is None or self.overlay_state not in {"starting", "connected"}:
            return None
        return cast(OverlayPresenter | None, runtime.current_presenter_for_ingress())

    def _current_overlay_bridge_for_direct_runtime_command(self) -> OverlayBridge | None:
        runtime = self._overlay_runtime
        if runtime is None:
            return None
        return cast(OverlayBridge | None, runtime.current_bridge_for_runtime_command())

    def _previous_overlay_target_for_apply(self) -> str:
        if self._overlay_runtime_is_active() and self._active_overlay_target is not None:
            return self._active_overlay_target
        return self._overlay_target_for_settings(self.settings)

    def _overlay_process_runner_for_target(
        self,
        target: str,
        *,
        task_factory: object | None = None,
    ) -> OverlayProcessRunner:
        if target == OVERLAY_TARGET_DESKTOP:
            return self._instantiate_overlay_process_runner(
                DesktopFletOverlayRunner,
                task_factory=task_factory,
            )
        return self._instantiate_overlay_process_runner(
            DefaultOverlayProcessRunner,
            task_factory=task_factory,
        )

    @staticmethod
    def _instantiate_overlay_process_runner(
        runner_cls: Callable[..., OverlayProcessRunner],
        *,
        task_factory: object | None,
    ) -> OverlayProcessRunner:
        try:
            return runner_cls(task_factory=task_factory)
        except TypeError:
            runner = runner_cls()
            with contextlib.suppress(Exception):
                setattr(runner, "task_factory", task_factory)
            return runner

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

    def _build_initial_desktop_runtime_controls_from_resolved_config(
        self,
        config: ResolvedOverlayConfig,
    ) -> list[dict[str, object]]:
        desktop_options = config.desktop_overlay_options
        position = desktop_options.get("position")
        if not isinstance(position, Mapping):
            position = {}
        visual_options = desktop_options.get("visual")
        if not isinstance(visual_options, Mapping):
            visual_options = {}

        width, height = self._desktop_dimensions_for_size_preset(desktop_options.get("size_preset"))
        x = position.get("x")
        y = position.get("y")
        if self._is_finite_non_bool_number(x) and self._is_finite_non_bool_number(y):
            bounds = {"x": x, "y": y, "width": width, "height": height}
        else:
            bounds = self._desktop_centered_bounds_for_dimensions(width=width, height=height)

        text_scale = visual_options.get("text_scale", DESKTOP_FLET_DEFAULT_TEXT_SCALE)
        background_alpha = visual_options.get(
            "background_alpha",
            DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA,
        )
        outline_width = visual_options.get("outline_width")
        interaction_mode = DESKTOP_INTERACTION_MODE_EDIT
        self.log_detailed(
            "[DesktopOverlay][Launch] "
            f"target=desktop locked={bool(desktop_options.get('locked', False))} "
            f"interaction_mode={interaction_mode} "
            f"size_preset={desktop_options.get('size_preset')} "
            f"x={bounds['x']} y={bounds['y']} width={bounds['width']} "
            f"height={bounds['height']} "
            f"text_scale={text_scale} "
            f"background_alpha={background_alpha} "
            f"outline_width={outline_width}"
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
                "text_scale": text_scale,
                "background_alpha": background_alpha,
                "outline_width": outline_width,
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
        return is_finite_non_bool_number(value)

    def _desktop_bounds_from_payload(
        self,
        payload: dict[object, object],
    ) -> dict[str, int | float] | None:
        return self._get_desktop_overlay_bounds_owner().bounds_from_payload(payload)

    def _is_valid_desktop_window_bounds_event_payload(
        self,
        payload: dict[object, object],
    ) -> bool:
        return self._get_desktop_overlay_bounds_owner().is_valid_event_payload(payload)

    def _track_desktop_apply_window_bounds_control(self, payload: dict[str, object]) -> None:
        self._get_desktop_overlay_bounds_owner().track_apply_control(payload)

    def _consume_suppressed_desktop_bounds(self, bounds: dict[str, int | float]) -> bool:
        return self._get_desktop_overlay_bounds_owner().consume_suppressed(bounds)

    def _discard_suppressed_desktop_bounds(self, bounds: dict[str, int | float]) -> None:
        self._get_desktop_overlay_bounds_owner().discard_suppressed(bounds)

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
        runtime = self._overlay_runtime
        queue = runtime.renderer_events_or_none() if runtime is not None else None
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
        if (
            self._active_overlay_target != OVERLAY_TARGET_DESKTOP
            or self._current_overlay_bridge_for_direct_runtime_command() is None
        ):
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
        bridge = self._current_overlay_bridge_for_direct_runtime_command()
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
        *,
        overlay_instance_id: str | None = None,
    ) -> None:
        try:
            while True:
                event = await queue.get()
                try:
                    await self._handle_desktop_renderer_event(
                        event,
                        overlay_instance_id=overlay_instance_id,
                    )
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

    async def _handle_desktop_renderer_event(
        self,
        event: object,
        *,
        overlay_instance_id: str | None = None,
    ) -> None:
        if overlay_instance_id is not None:
            runtime = self._overlay_runtime
            if runtime is None or not runtime.is_current_instance_id(overlay_instance_id):
                return
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
        self._get_desktop_overlay_bounds_owner().schedule_persistence(bounds)

    async def _persist_desktop_bounds_after_debounce(self) -> None:
        await self._get_desktop_overlay_bounds_owner().persist_after_debounce()

    async def _persist_desktop_bounds(self, bounds: dict[str, int | float]) -> None:
        if self.settings is None or self._active_overlay_target != OVERLAY_TARGET_DESKTOP:
            return
        if self._desktop_bounds_from_payload({"event": "window_bounds_changed", **bounds}) is None:
            return
        next_settings = copy.deepcopy(self.settings)
        desktop_settings = next_settings.overlay.desktop_flet
        desktop_settings.position.x = bounds["x"]
        desktop_settings.position.y = bounds["y"]
        desktop_settings.position.validate()
        routed = await self._apply_overlay_osc_output_settings_via_mutation_service(next_settings)
        if not routed or self.last_settings_mutation_result is None:
            return
        if not _settings_mutation_committed(self.last_settings_mutation_result):
            return
        self.log_detailed(
            "[DesktopOverlay][Bounds] persisted "
            f"x={bounds['x']} y={bounds['y']} width={bounds['width']} "
            f"height={bounds['height']} size_preset={desktop_settings.size_preset}"
        )

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
            and self._current_overlay_bridge_for_direct_runtime_command() is not None
        )
        if not configured_for_desktop and not desktop_renderer_active:
            return
        await self._cancel_desktop_bounds_persistence()
        self._drain_pending_desktop_user_bounds_events()
        _ = bounds
        next_settings = copy.deepcopy(self.settings)
        desktop_settings = next_settings.overlay.desktop_flet
        desktop_settings.position.x = None
        desktop_settings.position.y = None
        desktop_settings.validate()
        routed = await self._apply_overlay_osc_output_settings_via_mutation_service(next_settings)
        if routed and (
            self.last_settings_mutation_result is None
            or not _settings_mutation_committed(self.last_settings_mutation_result)
        ):
            return
        if self.settings is not None:
            self.settings.overlay.desktop_flet.locked = False
        self._set_desktop_overlay_interaction_mode(DESKTOP_INTERACTION_MODE_EDIT)
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

    async def _cancel_desktop_bounds_persistence(self) -> None:
        await self._get_desktop_overlay_bounds_owner().cancel()

    def _discard_pending_desktop_bounds_persistence(self) -> None:
        self._get_desktop_overlay_bounds_owner().discard()

    @property
    def _desktop_bounds_persist_task(self) -> asyncio.Task[None] | None:
        owner = self._desktop_overlay_bounds_owner
        return owner.persist_task if owner is not None else None

    @property
    def _pending_desktop_bounds(self) -> dict[str, int | float] | None:
        owner = self._desktop_overlay_bounds_owner
        return owner.pending_bounds if owner is not None else None

    @_pending_desktop_bounds.setter
    def _pending_desktop_bounds(
        self,
        bounds: dict[str, int | float] | None,
    ) -> None:
        self._get_desktop_overlay_bounds_owner().replace_pending_bounds(bounds)

    def _get_desktop_overlay_bounds_owner(self) -> DesktopOverlayBoundsOwner:
        owner = self._desktop_overlay_bounds_owner
        if owner is None:
            owner = DesktopOverlayBoundsOwner(
                persist_bounds=self._persist_desktop_bounds,
                debounce_seconds=lambda: DESKTOP_BOUNDS_PERSIST_DEBOUNCE_S,
                minimum_width=DESKTOP_FLET_MIN_WIDTH,
                minimum_height=DESKTOP_FLET_MIN_HEIGHT,
                diagnostics_sink=lambda event, metadata: self.log_detailed(
                    f"[DesktopOverlay][Bounds] event={event} metadata={dict(metadata)}",
                    level=logging.WARNING,
                ),
            )
            self._desktop_overlay_bounds_owner = owner
        return owner

    def _desktop_runtime_is_running_for_settings_update(
        self,
        settings: AppSettings,
    ) -> bool:
        return bool(
            settings.ui.overlay_enabled
            and self._active_overlay_target == OVERLAY_TARGET_DESKTOP
            and self._current_overlay_bridge_for_direct_runtime_command() is not None
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
            and self._current_overlay_bridge_for_direct_runtime_command() is not None
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

    async def _apply_desktop_size_preset_persistence_adjustment(
        self,
        previous_settings: AppSettings,
        next_settings: AppSettings,
    ) -> None:
        previous_desktop = copy.deepcopy(previous_settings.overlay.desktop_flet)
        previous_desktop.validate()
        next_desktop = next_settings.overlay.desktop_flet
        next_desktop.validate()
        if previous_desktop.size_preset == next_desktop.size_preset:
            return
        if not self._desktop_runtime_is_running_for_settings_update(next_settings):
            return
        await self._cancel_desktop_bounds_persistence()
        self._drain_pending_desktop_user_bounds_events()
        if previous_desktop.position.x is None or previous_desktop.position.y is None:
            return
        bounds = self._desktop_center_preserving_bounds_for_size_preset_change(
            previous_desktop_settings=previous_desktop,
            next_size_preset=next_desktop.size_preset,
        )
        next_desktop.position.x = bounds["x"]
        next_desktop.position.y = bounds["y"]
        next_desktop.position.validate()

    def _build_peer_runtime_config(self, settings: AppSettings) -> PeerCaptureSessionConfig:
        vnext_settings = self._canonical_vnext_settings_for(settings)
        backend = resolve_peer_stt_runtime_config_from_vnext(vnext_settings)
        provider_signature = build_peer_stt_provider_signature_from_vnext(vnext_settings)
        desktop_audio = vnext_settings.intent.desktop_audio
        capture_target = resolve_desktop_audio_capture_target(desktop_audio.capture_target)
        model_id = None
        if backend.provider == STTProviderName.LOCAL_QWEN_GPU.value:
            model_id = LOCAL_QWEN_GPU_MODEL_ID
        elif backend.provider in LOCAL_CPU_PROVIDERS:
            decision = resolve_local_asr_selection(
                backend.provider,
                backend.source_language,
            )
            model_id = decision.model_id
        session_options = (
            self._local_asr_session_options(
                source_language=backend.source_language,
                source_mode=backend.source_mode,
            )
            if backend.provider in {*LOCAL_CPU_PROVIDERS, STTProviderName.LOCAL_QWEN_GPU.value}
            else None
        )
        capture_vad_signature = (
            desktop_audio.output_device,
            capture_target,
            desktop_audio.vad_speech_threshold,
            desktop_audio.vad_hangover_ms,
            desktop_audio.vad_pre_roll_ms,
            backend.sample_rate_hz,
        )
        target = PeerCaptureTargetIntent(
            kind=capture_target.kind,
            device_name=capture_target.device_name,
            process_kind=capture_target.process_kind,
            executable_identity=capture_target.executable_identity,
            discord_channel=capture_target.discord_channel,
            executable_basename=capture_target.executable_basename,
        )
        return PeerCaptureSessionConfig(
            provider_id=backend.provider,
            output_device=desktop_audio.output_device,
            vad_speech_threshold=desktop_audio.vad_speech_threshold,
            vad_hangover_ms=desktop_audio.vad_hangover_ms,
            vad_pre_roll_ms=desktop_audio.vad_pre_roll_ms,
            provider_signature=provider_signature,
            runtime_signature=(
                backend.source_language,
                desktop_audio.output_device,
                target,
                desktop_audio.vad_speech_threshold,
                desktop_audio.vad_hangover_ms,
                desktop_audio.vad_pre_roll_ms,
                provider_signature,
            ),
            capture_signature=capture_vad_signature,
            capture_target=target,
            language=PeerCaptureLanguageFacts(
                source_mode=backend.source_mode,
                source_language=backend.source_language,
                expected_languages=tuple(vnext_settings.intent.languages.peer_expected_languages),
            ),
            target_sample_rate_hz=backend.sample_rate_hz,
            model_id=model_id,
            session_options=session_options,
            provider_context=backend,
            local_provider=backend.provider
            in {*LOCAL_CPU_PROVIDERS, STTProviderName.LOCAL_QWEN_GPU.value},
            release_backend_after=(
                LOCAL_QWEN_IDLE_RELEASE_SECONDS
                if backend.provider == STTProviderName.LOCAL_QWEN.value
                else None
            ),
            warmup=backend.provider != STTProviderName.LOCAL_QWEN.value,
        )

    async def _admit_peer_capture(
        self,
        config: PeerCaptureSessionConfig,
    ) -> PeerCaptureAdmission:
        if self.settings is None or self.hub is None:
            return PeerCaptureAdmission(
                PeerCaptureAdmissionStatus.REJECTED,
                reason="runtime_unavailable",
            )
        if config.local_provider and not await self._ensure_peer_local_stt_ready():
            return PeerCaptureAdmission(
                PeerCaptureAdmissionStatus.PENDING,
                reason="provider_unavailable",
                retain_intent=True,
            )
        return PeerCaptureAdmission(PeerCaptureAdmissionStatus.ADMITTED)

    def _on_peer_capture_state_changed(
        self,
        snapshot: PeerCaptureSessionSnapshot,
    ) -> None:
        if snapshot.state.value == "running":
            self._peer_process_warning_reason = None

    def _resolve_peer_capture_target(
        self,
        settings: AppSettings,
    ) -> ResolvedDesktopAudioCaptureTarget:
        persisted_capture_target = getattr(settings.desktop_audio, "runtime_capture_target", None)
        if not isinstance(persisted_capture_target, ResolvedDesktopAudioCaptureTarget):
            persisted_capture_target = None
        return self._peer_capture_target_resolution.resolve(
            legacy_output_device=settings.desktop_audio.output_device,
            persisted_capture_target=persisted_capture_target,
        )

    async def _close_peer_runtime_for_release(self, failures: list[Exception]) -> None:
        peer_runtime = self._peer_runtime
        if peer_runtime is None:
            return
        try:
            await peer_runtime.close()
        except Exception as exc:
            failures.append(exc)
            return
        if self._peer_runtime is peer_runtime:
            self._peer_runtime = None

    async def _stop_hub_for_release(self, failures: list[Exception]) -> None:
        hub = self.hub
        if hub is None:
            return
        try:
            await hub.stop()
        except Exception as exc:
            failures.append(exc)
            return
        if self.hub is hub:
            self.hub = None

    async def _close_vrc_mic_receiver_runtime_for_release(
        self,
        failures: list[Exception],
    ) -> None:
        try:
            await self._get_vrc_mic_sync_owner().close()
        except Exception as exc:
            failures.append(exc)

    async def _close_self_capture_owner_for_release(
        self,
        failures: list[Exception],
    ) -> None:
        owner = self._self_capture_owner
        if owner is None:
            return
        try:
            await owner.close()
        except Exception as exc:
            failures.append(exc)
            return
        if self._self_capture_owner is owner:
            self._self_capture_owner = None
        self._mic_task = None
        self._audio_source = None
        self._vad = None
        self._last_mic_loop_close_exception = None

    def bind_application_lifecycle(
        self,
        lifecycle: ApplicationShutdownCoordinator,
    ) -> None:
        if self._application_lifecycle is not None and self._application_lifecycle is not lifecycle:
            raise RuntimeError("GuiController application lifecycle is already bound")
        self._application_lifecycle = lifecycle

    def application_shutdown_callbacks(self) -> tuple[ApplicationShutdownCallback, ...]:
        return (
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_FREEZE_INGRESS,
                owner_name="GuiControllerCompatibilityFacade",
                callback_name="freeze_ingress",
                callback=self._freeze_application_ingress,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_FREEZE_INGRESS,
                owner_name="GithubStarPromptRuntime",
                callback_name="stop_ingress",
                callback=self._stop_github_star_prompt_ingress,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS,
                owner_name="GuiControllerCompatibilityFacade",
                callback_name="release_manual_typing",
                callback=self.release_manual_typing,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS,
                owner_name="ClipboardRuntime",
                callback_name="close",
                callback=self._close_clipboard_runtime_for_shutdown,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS,
                owner_name="VrchatOscPresenceProbeOwner",
                callback_name="cancel",
                callback=self._cancel_vrchat_osc_presence_probe,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL,
                owner_name="SelfCaptureSessionOwner",
                callback_name="stop_ingress",
                callback=self._stop_self_capture_ingress,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL,
                owner_name="VrcMicReceiverRuntime",
                callback_name="close",
                callback=self._close_vrc_mic_receiver_runtime,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL,
                owner_name="OverlayRuntimeHandle",
                callback_name="close",
                callback=self._close_overlay_runtime,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL,
                owner_name="OverlaySessionFallbackOwner",
                callback_name="close",
                callback=self._close_overlay_session_fallback_owner,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL,
                owner_name="PeerCaptureSessionOwner",
                callback_name="close",
                callback=self._close_peer_runtime,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL,
                owner_name="SelfLocalASRTransitionCoordinator",
                callback_name="close",
                callback=self._self_local_asr_transition.close,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL,
                owner_name="PeerLocalASRTransitionCoordinator",
                callback_name="close",
                callback=self._peer_local_asr_transition.close,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS,
                owner_name="GithubStarPromptRuntime",
                callback_name="close",
                callback=self._close_github_star_prompt_runtime,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS,
                owner_name="OAuthRuntime",
                callback_name="close",
                callback=self._close_oauth_runtime,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL,
                owner_name="LocalASRProvisioningRuntime",
                callback_name="close",
                callback=self._close_local_asr_provisioning,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL,
                owner_name="MicrophoneTestSessionOwner",
                callback_name="close",
                callback=self._close_microphone_test_runtime,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL,
                owner_name="SelfCaptureSessionOwner",
                callback_name="close",
                callback=self._close_self_capture_owner,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL,
                owner_name="ApplicationRuntimeLoggingOwner",
                callback_name="close_background_tasks",
                callback=self._close_runtime_logging_background_tasks,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL,
                owner_name="ManagedStatusRefreshOwner",
                callback_name="close",
                callback=self._close_managed_status_refresh_owner,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL,
                owner_name="ProviderStatusVerificationOwner",
                callback_name="close",
                callback=self._close_provider_status_verification_owner,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_CLOSE_PROVIDERS_OUTPUT_ADAPTERS,
                owner_name="ClientHub",
                callback_name="stop_owned_runtimes",
                callback=self._stop_hub,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_CLOSE_PROVIDERS_OUTPUT_ADAPTERS,
                owner_name="VrchatOscUdpSender",
                callback_name="close",
                callback=self._close_sender,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_CLOSE_PROVIDERS_OUTPUT_ADAPTERS,
                owner_name="ManagedOpenRouterReleaseService",
                callback_name="close",
                callback=self._close_managed_openrouter_release_service,
            ),
            ApplicationShutdownCallback(
                phase=SHUTDOWN_PHASE_FINAL_DIAGNOSTICS,
                owner_name="GuiControllerCompatibilityFacade",
                callback_name="emit_final_shutdown_diagnostics",
                callback=self._emit_final_application_shutdown_diagnostics,
            ),
            ApplicationShutdownCallback(
                phase=SHUTDOWN_PHASE_CLOSE_LOGGING_DIAGNOSTICS,
                owner_name="RuntimeLoggingService",
                callback_name="close_after_producers_stop",
                callback=self._close_runtime_logging,
            ),
        )

    async def stop(self) -> None:
        lifecycle = self._application_lifecycle
        if lifecycle is None:
            lifecycle = ApplicationShutdownCoordinator(
                self._standalone_application_shutdown_callbacks(),
                diagnostics_sink=self.emit_application_shutdown_diagnostic,
            )
            self.bind_application_lifecycle(lifecycle)
        try:
            await lifecycle.shutdown()
        except BaseException as exc:
            self._stop_exception = exc
            raise
        finally:
            self._stop_complete = lifecycle.is_terminal

    def _standalone_application_shutdown_callbacks(
        self,
    ) -> tuple[ApplicationShutdownCallback, ...]:
        app_callbacks = (
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS,
                owner_name="TranslatorAppCompatibilityFacade",
                callback_name="close_after_launch_tasks",
                callback=self._close_app_github_star_prompt_runtime,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_STOP_EXTERNAL_PRODUCERS,
                owner_name="TranslatorAppCompatibilityFacade",
                callback_name="close_oauth_runtime",
                callback=self._close_app_oauth_runtime,
            ),
        )
        return app_callbacks + self.application_shutdown_callbacks()

    def _freeze_application_ingress(self) -> None:
        self._shutdown_ingress_frozen = True
        self._stt_desired = False
        self._stt_activation_generation += 1
        self._peer_activation_generation += 1
        self._gpu_discovery_generation += 1
        owner = self._vrchat_osc_presence_owner
        if owner is not None:
            owner.stop_ingress()
        fallback_owner = self._overlay_session_fallback_owner
        if fallback_owner is not None:
            fallback_owner.stop_ingress()
        logging_owner = self._runtime_logging_owner
        if logging_owner is not None:
            logging_owner.stop_ingress()
        managed_status_owner = self._managed_status_refresh_owner
        if managed_status_owner is not None:
            managed_status_owner.stop_ingress()
        provider_status_owner = self._provider_status_verification_owner
        if provider_status_owner is not None:
            provider_status_owner.stop_ingress()

    def _stop_github_star_prompt_ingress(self) -> None:
        owner = self._github_star_prompt_owner
        if owner is not None:
            owner.stop_ingress()

    async def _close_github_star_prompt_runtime(self) -> None:
        failures: list[Exception] = []
        await self._close_github_star_prompt_runtime_for_release(failures)
        _raise_lifecycle_cleanup_failures("GitHub prompt shutdown failed", failures)

    async def _close_app_github_star_prompt_runtime(self) -> None:
        failures: list[Exception] = []
        await self._close_app_github_star_prompt_runtime_for_release(failures)
        _raise_lifecycle_cleanup_failures("App prompt shutdown failed", failures)

    async def _close_app_oauth_runtime(self) -> None:
        failures: list[Exception] = []
        await self._close_app_oauth_runtime_for_release(failures)
        _raise_lifecycle_cleanup_failures("App OAuth shutdown failed", failures)

    async def _close_runtime_logging_background_tasks(self) -> None:
        owner = self._runtime_logging_owner
        if owner is not None:
            await owner.close_background_tasks()

    async def _close_managed_status_refresh_owner(self) -> None:
        owner = self._managed_status_refresh_owner
        if owner is not None:
            await owner.close()

    async def _close_provider_status_verification_owner(self) -> None:
        owner = self._provider_status_verification_owner
        if owner is not None:
            await owner.close()

    async def _close_microphone_test_runtime(self) -> None:
        failures: list[Exception] = []
        await self._close_microphone_test_runtime_for_release(failures)
        _raise_lifecycle_cleanup_failures("Microphone test shutdown failed", failures)

    async def _close_clipboard_runtime_for_shutdown(self) -> None:
        owner = self._clipboard_auto_translation_owner
        if owner is not None:
            await owner.close(strict_runtime_errors=True)

    async def _stop_self_capture_ingress(self) -> None:
        await self.set_stt_enabled(False)

    async def _close_self_capture_owner(self) -> None:
        failures: list[Exception] = []
        await self._close_self_capture_owner_for_release(failures)
        _raise_lifecycle_cleanup_failures("Self capture shutdown failed", failures)

    async def _close_vrc_mic_receiver_runtime(self) -> None:
        failures: list[Exception] = []
        await self._close_vrc_mic_receiver_runtime_for_release(failures)
        _raise_lifecycle_cleanup_failures("VRChat microphone receiver shutdown failed", failures)

    async def _close_overlay_runtime(self) -> None:
        await self._shutdown_overlay_runtime(preserve_failure_reason=True)
        self._clear_overlay_session_desktop_fallback()

    async def _close_overlay_session_fallback_owner(self) -> None:
        owner = self._overlay_session_fallback_owner
        if owner is not None:
            await owner.close()

    async def _close_peer_runtime(self) -> None:
        failures: list[Exception] = []
        await self._close_peer_runtime_for_release(failures)
        _raise_lifecycle_cleanup_failures("Peer capture shutdown failed", failures)

    async def _stop_hub(self) -> None:
        failures: list[Exception] = []
        await self._stop_hub_for_release(failures)
        if self.hub is None:
            self._bridge_task = None
            self._ui_event_bridge = None
        _raise_lifecycle_cleanup_failures("Hub owner shutdown failed", failures)

    def _close_sender(self) -> None:
        sender = self.sender
        if sender is None:
            self.osc = None
            return
        sender.close()
        self.sender = None
        self.osc = None

    async def _close_managed_openrouter_release_service(self) -> None:
        service = self._managed_openrouter_release_service
        self._managed_openrouter_release_service = None
        if service is not None:
            await service.close()

    def _emit_final_application_shutdown_diagnostics(
        self,
        context: ApplicationShutdownContext,
    ) -> None:
        owner = self._runtime_logging_owner
        if owner is not None:
            owner.emit_terminal_summary(context)

    def _close_runtime_logging(self, context: ApplicationShutdownContext) -> None:
        owner = self._runtime_logging_owner
        if owner is not None:
            owner.close_after_producers_stop(context)

    def emit_application_shutdown_diagnostic(
        self,
        diagnostic: ApplicationShutdownDiagnostic,
    ) -> None:
        owner = self._runtime_logging_owner
        if owner is None:
            owner = self._get_runtime_logging_owner()
        owner.emit_shutdown_diagnostic(diagnostic)

    async def set_overlay_enabled(self, enabled: bool) -> None:
        if self.settings is None:
            return

        self.log_basic(f"[Overlay] Toggle request: enabled={enabled}")
        runtime = self._overlay_runtime
        self.log_detailed(
            "[Overlay] Toggle detail: "
            f"current_state={self.overlay_state} "
            f"has_bridge={runtime is not None and runtime.bridge is not None} "
            f"has_manager={runtime is not None and runtime.process_manager is not None}"
        )
        self.settings.ui.overlay_enabled = bool(enabled)
        if not enabled:
            self._peer_activation_generation += 1
            self.settings.ui.peer_translation_enabled = False
            self._last_peer_translation_enabled = False
            self._last_peer_translation_activation_requested = False
            self._clear_overlay_session_desktop_fallback()
        self._refresh_overlay_peer_consumers()

        if enabled:
            await self._begin_overlay_start()
            return

        await self._shutdown_overlay_runtime(preserve_failure_reason=True)

    async def set_peer_translation_enabled(self, enabled: bool) -> None:
        if self.settings is None:
            return

        enabled = bool(enabled)
        if (
            enabled
            and self.settings.provider.peer_stt != STTProviderName.LOCAL_CPU_AUTO
            and not self._persist_current_manual_local_asr_fallback(channel="peer")
        ):
            return
        self._peer_activation_generation += 1
        activation_generation = self._peer_activation_generation
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
        self._peer_activation_starting = enabled
        self._refresh_overlay_peer_consumers()
        if enabled:
            ready = await self._ensure_peer_local_stt_ready(
                activation_generation=activation_generation
            )
            if activation_generation != self._peer_activation_generation:
                return
            if not ready:
                self._peer_activation_starting = False
        else:
            self._reset_local_stt_pending_peer_enable_after_install()
            self._gpu_pending_enable_channels = frozenset(
                channel for channel in self._gpu_pending_enable_channels if channel != "peer"
            )
        self._clear_local_stt_pending_enable_if_provider_switched_away()
        self._sync_local_stt_notice()
        self._refresh_overlay_peer_consumers()

        if enabled and self.overlay_state not in {"starting", "connected"}:
            await self._begin_overlay_start()
        else:
            refresh_dependencies = self._refresh_overlay_runtime_dependencies
            if not enabled and _callable_accepts_keyword(
                refresh_dependencies,
                "peer_stop_mode",
            ):
                await refresh_dependencies(peer_stop_mode="release")
            else:
                await refresh_dependencies()
        if activation_generation != self._peer_activation_generation:
            return
        self._sync_effective_hub_flags(self.settings)
        self._peer_activation_starting = False
        if enabled:
            self._enqueue_peer_translation_disclosure()
        self._refresh_overlay_peer_consumers()

    async def retry_peer_process_capture(self) -> bool:
        if (
            self.settings is None
            or self._peer_runtime is None
            or not self._peer_runtime_should_be_active(self.settings)
        ):
            return False
        if not await self._ensure_peer_local_stt_ready():
            return False
        config = self._build_peer_runtime_config(self.settings)
        retried = await self._peer_runtime.retry_process_capture(config=config)
        if retried:
            self._peer_process_warning_reason = None
        self._sync_effective_hub_flags(self.settings)
        self._refresh_overlay_peer_consumers()
        return retried

    def _enqueue_peer_translation_disclosure(self) -> None:
        hub = self.hub
        if hub is None:
            return
        enqueue_disclosure = getattr(hub, "enqueue_peer_translation_disclosure", None)
        if callable(enqueue_disclosure):
            enqueue_disclosure(self.app.localize("peer_translation.disclosure"))

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

            previous_runtime = self._overlay_runtime
            teardown_succeeded = await self._teardown_overlay_runtime(
                preserve_presenter_state=True,
            )
            if not teardown_succeeded:
                return
            preserved_presenter = None
            if previous_runtime is not None and previous_runtime.is_closed:
                preserved_presenter = cast(
                    OverlayPresenter | None,
                    previous_runtime.detach_preserved_presenter(),
                )
            runtime = self._new_overlay_runtime_handle()
            if preserved_presenter is not None:
                runtime.adopt_presenter(preserved_presenter)
            self._active_overlay_target = self._effective_overlay_target_for_start()
            previous_state = self.overlay_state
            self.overlay_state = "starting"
            self.auto_restart_scheduled = False
            self._log_overlay_state_transition(previous_state, self.overlay_state)
            self._notify_overlay_state()
            runtime.create_start_task(self._run_overlay_start(runtime))

    async def _apply_overlay_retry_ownership(
        self,
        runtime: OverlayRuntimeHandle,
        presenter: OverlayPresenter,
        manager: OverlayProcessManager,
        *,
        confirmed: bool,
    ) -> None:
        if not self._overlay_runtime_is_current(runtime) or runtime.process_manager is not manager:
            return
        await presenter.update_native_retry_ownership(confirmed)

    async def _run_overlay_start(self, runtime: OverlayRuntimeHandle | None = None) -> None:
        if runtime is None:
            runtime = self._overlay_runtime
            if runtime is None:
                runtime = self._new_overlay_runtime_handle()
        overlay_instance_id: str | None = None
        try:
            if self.settings is None or self.hub is None:
                self._active_overlay_target = None
                if self._overlay_runtime_is_current(runtime):
                    self.on_overlay_start_failed("unknown")
                return

            presenter = cast(OverlayPresenter | None, runtime.presenter)
            overlay_instance_id = f"overlay-{secrets.token_hex(8)}"
            runtime.set_overlay_instance_id(overlay_instance_id)
            diagnostics = OverlayDiagnosticsRecorder(overlay_instance_id=overlay_instance_id)
            runtime.attach_diagnostics(diagnostics)
            resolved_overlay_config = resolve_overlay_config(self.settings)
            overlay_target = self._active_overlay_target or self._normalized_overlay_target(
                resolved_overlay_config.target
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
                    show_translation=resolved_overlay_config.show_translation,
                    show_peer_original=resolved_overlay_config.show_peer_original,
                    task_factory=runtime.create_child_task,
                    peer_presentation_refresh_burst=peer_presentation_refresh_burst,
                    self_presentation_refresh_burst=self_presentation_refresh_burst,
                )
            else:
                presenter.runtime_log_detailed = self.log_detailed
            presenter = cast(OverlayPresenter, runtime.adopt_presenter(presenter))
            presenter.runtime_log_detailed = self.log_detailed
            if overlay_target != OVERLAY_TARGET_DESKTOP:
                await presenter.update_native_retry_ownership(False)
            await presenter.update_calibration(self.overlay_calibration.copy())
            await presenter.update_display_preferences(
                show_translation=resolved_overlay_config.show_translation,
                show_peer_original=resolved_overlay_config.show_peer_original,
            )
            await presenter.update_peer_presentation_refresh_burst(peer_presentation_refresh_burst)
            await presenter.update_self_presentation_refresh_burst(self_presentation_refresh_burst)
            bridge = OverlayBridge(
                session_token=secrets.token_urlsafe(16),
                initial_snapshot=presenter.snapshot(),
                overlay_instance_id=overlay_instance_id,
                diagnostics=diagnostics,
                runtime_logging_mode=self.runtime_logging_mode,
                desktop_runtime_controls_enabled=overlay_target == OVERLAY_TARGET_DESKTOP,
                task_factory=runtime.create_child_task,
            )
            if overlay_target == OVERLAY_TARGET_DESKTOP:
                initial_desktop_controls = (
                    self._build_initial_desktop_runtime_controls_from_resolved_config(
                        resolved_overlay_config
                    )
                )
                initial_interaction_control = initial_desktop_controls[-1]
                self._set_desktop_overlay_interaction_mode(initial_interaction_control.get("mode"))
                for payload in initial_desktop_controls:
                    self._track_desktop_apply_window_bounds_control(payload)
                bridge.set_initial_desktop_runtime_controls(initial_desktop_controls)
            runtime.attach_bridge(bridge)
            await bridge.start()
            if not self._overlay_runtime_is_current(
                runtime,
                overlay_instance_id=overlay_instance_id,
            ):
                await self._close_stale_overlay_start_runtime(runtime)
                return
            current_presenter = cast(
                OverlayPresenter | None,
                runtime.current_presenter_for_ingress(),
            )
            if current_presenter is not presenter:
                await self._close_stale_overlay_start_runtime(runtime)
                return
            presenter = current_presenter
            presenter.attach_bridge(bridge)
            latest_snapshot = presenter.snapshot()
            if bridge.snapshot() != latest_snapshot:
                await bridge.replace_snapshot(latest_snapshot)
            runtime.attach_diagnostics(diagnostics)
            await self._replace_hub_overlay_sink(presenter)
            self.hub.overlay_diagnostics = diagnostics

            renderer_events: asyncio.Queue[dict[str, object]] | None = None
            if overlay_target == OVERLAY_TARGET_DESKTOP:
                renderer_events = asyncio.Queue(maxsize=64)
                runtime.attach_renderer_events(renderer_events)
                runtime.create_renderer_event_task(
                    self._consume_desktop_renderer_events(
                        renderer_events,
                        overlay_instance_id=overlay_instance_id,
                    )
                )
            else:
                runtime.attach_renderer_events(None)

            manager = OverlayProcessManager(
                process_runner=self._overlay_process_runner_for_target(
                    overlay_target,
                    task_factory=runtime.create_child_task,
                ),
                bridge_url=bridge.url,
                bridge_messages=bridge.messages,
                session_token=bridge.session_token,
                locale=self.settings.ui.locale,
                log_dir=str(user_config_dir()),
                startup_timeout_ms=OVERLAY_STARTUP_TIMEOUT_MS,
                renderer_events=renderer_events,
                overlay_instance_id=overlay_instance_id,
                logging_mode=self.runtime_logging_mode,
                diagnostics=diagnostics,
                task_factory=runtime.create_child_task,
                retry_ownership_changed=(
                    None
                    if overlay_target == OVERLAY_TARGET_DESKTOP
                    else lambda confirmed: self._apply_overlay_retry_ownership(
                        runtime,
                        presenter,
                        manager,
                        confirmed=confirmed,
                    )
                ),
            )
            runtime.attach_process_manager(manager)
            await manager.start()

            if not self._overlay_runtime_is_current(
                runtime,
                overlay_instance_id=overlay_instance_id,
            ):
                await self._close_stale_overlay_start_runtime(runtime)
                return

            if runtime.process_manager is not manager:
                return

            if manager.state != "connected":
                await self._handle_overlay_start_failure(manager.failure_reason)
                return

            self._mark_overlay_connected()
            await self._refresh_overlay_runtime_dependencies()
            monitor_task = getattr(manager, "_monitor_task", None)
            if monitor_task is not None:
                runtime.create_monitor_task(
                    self._watch_overlay_runtime(
                        manager,
                        monitor_task,
                        runtime=runtime,
                        overlay_instance_id=overlay_instance_id,
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._overlay_runtime_is_current(
                runtime,
                overlay_instance_id=overlay_instance_id,
            ):
                self.log_detailed(
                    "[Overlay] Ignoring stale overlay runtime start failure",
                    level=logging.WARNING,
                    exception=exc,
                )
                await self._close_stale_overlay_start_runtime(runtime)
                return
            self.log_detailed(
                "[Overlay] Failed to start overlay runtime",
                level=logging.ERROR,
                exception=exc,
            )
            await self._handle_overlay_start_failure("unknown")

    async def _watch_overlay_runtime(
        self,
        manager: OverlayProcessManager,
        monitor_task: asyncio.Task[None],
        *,
        runtime: OverlayRuntimeHandle | None = None,
        overlay_instance_id: str | None = None,
    ) -> None:
        if runtime is None:
            runtime = self._overlay_runtime
        try:
            await monitor_task
            if runtime is not None and not self._overlay_runtime_is_current(
                runtime,
                overlay_instance_id=overlay_instance_id,
            ):
                return
            if runtime is None or runtime.process_manager is not manager:
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

    async def _handle_overlay_start_failure(self, failure_reason: str | None) -> None:
        reason = self._normalize_overlay_failure_reason(failure_reason)
        if self._should_session_fallback_overlay_to_desktop(reason):
            self.log_basic(f"[Overlay] Session fallback to desktop: reason={reason}")
            self._get_overlay_session_fallback_owner().activate()
            await self._teardown_overlay_runtime(preserve_presenter_state=True)
            previous_state = self.overlay_state
            self.overlay_state = "off"
            self.failure_reason = None
            self._log_overlay_state_transition(previous_state, self.overlay_state)
            self._sync_effective_hub_flags()
            await self._refresh_overlay_runtime_dependencies()
            self._notify_overlay_state()
            self._set_overlay_session_fallback_notice_active(True)
            # Restart outside this start task so runtime close/reopen is not re-entrant.
            self._schedule_overlay_session_desktop_fallback_start()
            return
        self.on_overlay_start_failed(failure_reason)
        await self._teardown_overlay_runtime(preserve_presenter_state=True)
        await self._refresh_overlay_runtime_dependencies()

    def _schedule_overlay_session_desktop_fallback_start(self) -> None:
        self._get_overlay_session_fallback_owner().schedule()

    @property
    def _overlay_session_desktop_fallback_active(self) -> bool:
        owner = self._overlay_session_fallback_owner
        return owner.active if owner is not None else False

    def _get_overlay_session_fallback_owner(self) -> OverlaySessionFallbackOwner:
        owner = self._overlay_session_fallback_owner
        if owner is None:
            owner = OverlaySessionFallbackOwner(
                can_start=lambda: bool(
                    self.settings is not None
                    and self.settings.ui.overlay_enabled
                    and self.overlay_state not in {"starting", "connected"}
                ),
                start_overlay=self._begin_overlay_start,
                publish_notice=self.app.set_dashboard_overlay_session_fallback_notice,
                diagnostics_sink=lambda _event, _metadata, exception: self.log_detailed(
                    "[Overlay] Failed to schedule session desktop fallback",
                    level=logging.WARNING,
                    exception=exception,
                ),
            )
            self._overlay_session_fallback_owner = owner
        return owner

    async def _shutdown_overlay_runtime(self, *, preserve_failure_reason: bool) -> None:
        if self._overlay_lock is None:
            self._overlay_lock = asyncio.Lock()

        self.log_basic("[Overlay] Shutdown requested")
        runtime = self._overlay_runtime
        self.log_detailed(
            "[Overlay] Shutdown detail: "
            f"preserve_failure_reason={preserve_failure_reason} "
            f"state={self.overlay_state} "
            f"has_bridge={runtime is not None and runtime.bridge is not None} "
            f"has_manager={runtime is not None and runtime.process_manager is not None} "
            f"presenter_attached={runtime is not None and runtime.presenter is not None}"
        )
        async with self._overlay_lock:
            runtime = self._overlay_runtime
            has_runtime = self._overlay_runtime_has_resources(runtime)
            if not has_runtime and self.overlay_state == "off":
                return

            previous_state = self.overlay_state
            self.overlay_state = "stopping"
            self.auto_restart_scheduled = False
            self._log_overlay_state_transition(previous_state, self.overlay_state)
            self._notify_overlay_state()

            teardown_succeeded = await self._teardown_overlay_runtime(
                preserve_presenter_state=False,
                emit_shutdown=True,
            )
            if not teardown_succeeded and self._overlay_runtime_has_resources(
                self._overlay_runtime
            ):
                previous_state = self.overlay_state
                self.overlay_state = "failed"
                if not preserve_failure_reason or self.failure_reason is None:
                    self.failure_reason = self._normalize_overlay_failure_reason(None)
                self._log_overlay_state_transition(previous_state, self.overlay_state)
                self._sync_effective_hub_flags()
                await self._refresh_overlay_runtime_dependencies()
                self._notify_overlay_state()
                return
            previous_state = self.overlay_state
            self.overlay_state = "off"
            if not preserve_failure_reason:
                self.failure_reason = None
            self._log_overlay_state_transition(previous_state, self.overlay_state)
            self._sync_effective_hub_flags()
            await self._refresh_overlay_runtime_dependencies()
            self._notify_overlay_state()

    async def _emit_overlay_shutdown(self) -> None:
        presenter = self._current_overlay_presenter_for_direct_runtime_command()
        if presenter is None:
            return
        with contextlib.suppress(Exception):
            await presenter.broadcast_shutdown()
            await asyncio.sleep(OVERLAY_SHUTDOWN_GRACE_S)

    async def _teardown_overlay_runtime(
        self,
        *,
        preserve_presenter_state: bool,
        emit_shutdown: bool = False,
    ) -> bool:
        runtime = self._ensure_overlay_runtime_handle()

        await self._cancel_desktop_bounds_persistence()
        close_succeeded = True
        try:
            await runtime.close(
                preserve_presenter_state=preserve_presenter_state,
                hub=self.hub,
                emit_shutdown=emit_shutdown,
            )
        except Exception as exc:
            close_succeeded = False
            message = "[Overlay] Overlay runtime close reported cleanup failure"
            detailed_emitted = self.log_detailed(
                message,
                level=logging.WARNING,
                exception=exc,
            )
            if not detailed_emitted:
                self.log_basic(message, level=logging.WARNING)
        if close_succeeded and not self._overlay_runtime_has_resources(runtime):
            self._overlay_runtime = None
        self._active_overlay_target = None
        self._get_desktop_overlay_bounds_owner().clear_suppressed()
        if not preserve_presenter_state:
            self._set_desktop_overlay_interaction_mode(DESKTOP_INTERACTION_MODE_EDIT)
        return close_succeeded

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
        self._refresh_overlay_peer_consumers()

    def _log_overlay_state_transition(self, previous_state: str, next_state: str) -> None:
        runtime = self._overlay_runtime
        manager = runtime.process_manager if runtime is not None else None
        transition_message = f"[Overlay] State transition: {previous_state} -> {next_state}"
        if self.failure_reason is not None:
            transition_message = f"{transition_message} failure_reason={self.failure_reason}"
        self.log_basic(transition_message)
        self.log_detailed(
            "[Overlay] State detail: "
            f"presenter_attached={runtime is not None and runtime.presenter is not None} "
            f"bridge_attached={runtime is not None and runtime.bridge is not None} "
            f"manager_state={manager.state if manager is not None else None}"
        )

    def begin_overlay_calibration(self) -> OverlayCalibration:
        return self._get_overlay_calibration_owner().begin()

    def set_overlay_calibration_field(
        self,
        field_name: str,
        value: object,
    ) -> OverlayCalibration:
        return self._get_overlay_calibration_owner().set_field(field_name, value)

    def apply_overlay_calibration(self) -> OverlayCalibration:
        return self._get_overlay_calibration_owner().apply()

    async def _apply_overlay_calibration_persistence(
        self,
        calibration: OverlayCalibration,
    ) -> None:
        await self._get_overlay_calibration_owner().persist_calibration(calibration)

    async def _persist_overlay_calibration(
        self,
        calibration: OverlayCalibration,
    ) -> None:
        if self.settings is None:
            return
        next_settings = copy.deepcopy(self.settings)
        next_settings.overlay.calibration = calibration.copy()
        await self._apply_overlay_osc_output_settings_via_mutation_service(next_settings)

    def _schedule_overlay_calibration_persistence(
        self,
        calibration: OverlayCalibration,
    ) -> None:
        self._get_overlay_calibration_owner().schedule_persistence(calibration)

    def cancel_overlay_calibration(self) -> OverlayCalibration:
        return self._get_overlay_calibration_owner().cancel()

    def _sync_overlay_calibration_cache(self, settings: AppSettings | None = None) -> None:
        resolved_settings = settings or self.settings
        if resolved_settings is None:
            return
        self._get_overlay_calibration_owner().replace_current(resolved_settings.overlay.calibration)

    async def _emit_overlay_calibration_update(self) -> None:
        await self._get_overlay_calibration_owner().emit_current()

    async def _emit_overlay_calibration_to_runtime(
        self,
        calibration: OverlayCalibration,
    ) -> None:
        presenter = self._current_overlay_presenter_for_direct_runtime_command()
        if presenter is None:
            return
        await presenter.update_calibration(calibration.copy())

    def _schedule_overlay_calibration_emit(self) -> None:
        self._get_overlay_calibration_owner().schedule_emit()

    def _get_overlay_calibration_owner(self) -> OverlayCalibrationOwner:
        owner = self._overlay_calibration_owner
        if owner is None:
            owner = OverlayCalibrationOwner(
                schedule_task=lambda task: self.app.schedule_task(task),
                persist=self._persist_overlay_calibration,
                emit=self._emit_overlay_calibration_to_runtime,
                can_persist=lambda: self.settings is not None,
                can_emit=lambda: (
                    not self._shutdown_ingress_frozen
                    and self._current_overlay_presenter_for_direct_runtime_command() is not None
                ),
                log_detailed=self.log_detailed,
            )
            self._overlay_calibration_owner = owner
        return owner

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
            self.app.set_dashboard_translation_enabled(False)
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
            warmup = getattr(llm, "warmup", None)
            if callable(warmup):
                with contextlib.suppress(Exception):
                    result = warmup()
                    if inspect.isawaitable(result):
                        await result
        return bool(self.hub.translation_enabled)

    async def set_stt_enabled(self, enabled: bool, *, force_immediate: bool = False) -> None:
        if enabled and not self._persist_current_manual_local_asr_fallback(channel="self"):
            self.app.set_dashboard_stt_enabled(False)
            return
        self.log_basic(f"[STT] Toggle request: enabled={enabled}")
        self.log_detailed(
            "[STT] Toggle detail: "
            f"desired_before={self._stt_desired} overlay_state={self.overlay_state}"
        )
        self._stt_activation_starting = bool(enabled)
        self._stt_activation_failed = False
        self._sync_local_stt_notice()
        self._stt_desired = bool(enabled)
        self._stt_force_immediate = force_immediate
        if not enabled:
            self._reset_local_stt_pending_enable_after_install()
            self._gpu_pending_enable_channels = frozenset(
                channel for channel in self._gpu_pending_enable_channels if channel != "self"
            )

        # Log provider info when enabling
        if enabled and self.settings is not None:
            provider = self.settings.provider.stt.value
            if provider == "qwen_asr":
                region = self.settings.qwen.region.value
                self.log_basic(f"[STT] Enabled with provider: {provider}")
                self.log_detailed(f"[STT] Provider detail: provider={provider} region={region}")
            else:
                self.log_basic(f"[STT] Enabled with provider: {provider}")

        # Mark promo eligible when user explicitly enables STT via button
        if enabled and self.hub is not None:
            self.hub.mark_promo_eligible()

        snapshot = await self._ensure_stt_switch()
        self._stt_activation_starting = False
        if snapshot is None or snapshot.state is not SelfCaptureSessionState.ADMISSION_PENDING:
            self.app.set_dashboard_stt_enabled(
                bool(
                    self._stt_desired
                    and not self._stt_activation_failed
                    and self._mic_task is not None
                )
            )
        self._sync_local_stt_notice()

    def _show_short_stt_message(self, message_key: str) -> None:
        self._show_short_message(message_key)

    def _vrchat_osc_probe_port(self) -> int:
        if self.settings is None:
            return 9000
        port = getattr(self.settings.osc, "port", 9000)
        return port if isinstance(port, int) and 0 < port <= 65535 else 9000

    def _set_vrchat_osc_notice_active(self, active: bool) -> None:
        self._get_vrchat_osc_presence_owner().publish(active)

    def _schedule_vrchat_osc_presence_probe(self, *, force: bool = False) -> None:
        self._get_vrchat_osc_presence_owner().schedule(force=force)

    async def _run_vrchat_osc_presence_probe_loop(self, generation: int) -> None:
        await self._get_vrchat_osc_presence_owner().run(generation)

    async def _cancel_vrchat_osc_presence_probe(self) -> None:
        await self._get_vrchat_osc_presence_owner().cancel()

    @property
    def _vrchat_osc_probe_task(self) -> asyncio.Task[None] | None:
        owner = self._vrchat_osc_presence_owner
        return owner.task if owner is not None else None

    def _get_vrchat_osc_presence_owner(self) -> VrchatOscPresenceProbeOwner:
        owner = self._vrchat_osc_presence_owner
        if owner is None:
            owner = VrchatOscPresenceProbeOwner(
                presence_provider=lambda: self.vrchat_osc_presence,
                port_provider=self._vrchat_osc_probe_port,
                publish_notice=self.app.set_dashboard_vrchat_osc_notice,
                diagnostics_sink=lambda _event, _metadata, exception: self.log_detailed(
                    "[OSC] VRChat OSC presence probe failed",
                    level=logging.WARNING,
                    exception=exception,
                ),
            )
            self._vrchat_osc_presence_owner = owner
        return owner

    def _show_short_message(self, message_key: str, **message_kwargs: object) -> None:
        try:
            self.app.show_message(message_key, **message_kwargs)
        except Exception:
            self._log_error(self.app.localize(message_key, **message_kwargs))

    async def _handle_managed_translation_enable(self, request_generation: int) -> bool:
        if self.settings is None or self.hub is None:
            return True
        if not self._managed_openrouter_selected():
            return True
        if await self._should_route_managed_trans_to_founder_letter():
            return False
        service = self._managed_openrouter_release_service
        if service is None:
            return True
        discord_claim_guard: ManagedAuthClaimGuard | None = None
        if not self._managed_china_auth_relevant_for_translation_enable():
            try:
                discord_claim_guard = self._managed_auth_claim_guard_for_settings(self.settings)
                claim_result = await discord_claim_guard.preflight(
                    MANAGED_AUTH_CLAIM_SOURCE_DISCORD
                )
            except Exception:
                self._show_short_message("discord_auth.error.retry")
                return False
            if claim_result is not None:
                self.last_settings_mutation_result = claim_result
                message = claim_result.message
                message_key = message.key if message is not None else "discord_auth.error.retry"
                message_kwargs = dict(message.params) if message is not None else {}
                self._show_short_message(message_key, **message_kwargs)
                return False

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
            if discord_claim_guard is not None:
                with contextlib.suppress(Exception):
                    discord_claim_guard.record_success(MANAGED_AUTH_CLAIM_SOURCE_DISCORD)
                    discord_claim_guard.managed_state.persist()
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
        self.app.set_dashboard_translation_enabled(False)
        if (
            result.message_key == "qq_managed_auth.required"
            and self._managed_china_auth_relevant_for_translation_enable()
        ):
            self._show_qq_managed_auth_dialog()
            return False
        self._show_short_message(result.message_key, **dict(result.message_kwargs))
        return False

    def _normalize_manual_local_asr_fallbacks(
        self,
        settings: AppSettings,
    ) -> tuple[AppSettings, tuple[str, ...], bool]:
        cpu_auto_requested = (
            settings.provider.stt == STTProviderName.LOCAL_CPU_AUTO
            or settings.provider.peer_stt == STTProviderName.LOCAL_CPU_AUTO
        )
        cpu_auto_available = True
        if cpu_auto_requested:
            cpu_auto_available = (
                self._get_local_asr_provisioning_owner().snapshot.cpu_auto_available
            )
        self_decision = resolve_local_asr_selection(
            settings.provider.stt.value,
            settings.languages.source_language,
            cpu_auto_available=cpu_auto_available,
        )
        peer_decision = resolve_local_asr_selection(
            settings.provider.peer_stt.value,
            settings.languages.effective_peer_source,
            cpu_auto_available=cpu_auto_available,
        )
        fallback_channels: list[str] = []
        normalized = settings
        if self_decision.fallback_applied or peer_decision.fallback_applied:
            normalized = copy.deepcopy(settings)
        if self_decision.fallback_applied:
            normalized.provider.stt = STTProviderName(self_decision.effective_provider)
            fallback_channels.append("self")
        if peer_decision.fallback_applied:
            normalized.provider.peer_stt = STTProviderName(peer_decision.effective_provider)
            fallback_channels.append("peer")
        installation_fallback = not cpu_auto_available and (
            (
                self_decision.fallback_applied
                and settings.provider.stt == STTProviderName.LOCAL_CPU_AUTO
            )
            or (
                peer_decision.fallback_applied
                and settings.provider.peer_stt == STTProviderName.LOCAL_CPU_AUTO
            )
        )
        return normalized, tuple(fallback_channels), installation_fallback

    def _manual_local_asr_fallback_normalization_channels(
        self,
        settings: AppSettings,
    ) -> frozenset[str]:
        current = self.settings
        if current is None:
            return frozenset({"self", "peer"})
        channels: set[str] = set()
        if (
            current.provider.stt != settings.provider.stt
            and settings.provider.stt.value in LOCAL_CPU_PROVIDERS
        ) or (
            settings.provider.stt.value in LOCAL_CPU_DIRECT_MODEL_BY_PROVIDER
            and settings.provider.stt != STTProviderName.LOCAL_QWEN
            and current.languages.source_language != settings.languages.source_language
        ):
            channels.add("self")
        if (
            current.provider.peer_stt != settings.provider.peer_stt
            and settings.provider.peer_stt.value in LOCAL_CPU_PROVIDERS
        ) or (
            settings.provider.peer_stt.value in LOCAL_CPU_DIRECT_MODEL_BY_PROVIDER
            and settings.provider.peer_stt != STTProviderName.LOCAL_QWEN
            and current.languages.effective_peer_source != settings.languages.effective_peer_source
        ):
            channels.add("peer")
        return frozenset(channels)

    def _notify_manual_local_asr_fallback(
        self,
        channels: tuple[str, ...],
        *,
        installation_fallback: bool = False,
    ) -> None:
        if not channels:
            return
        self._show_short_message(
            "local_stt.installation_fallback_qwen"
            if installation_fallback
            else "local_stt.language_fallback_qwen"
        )

    def _persist_current_manual_local_asr_fallback(self, *, channel: str | None = None) -> bool:
        if self.settings is None:
            return False
        previous = self.settings
        if channel is None:
            normalized, fallback_channels, installation_fallback = (
                self._normalize_manual_local_asr_fallbacks(previous)
            )
            if normalized is previous:
                return True
            self.settings = normalized
            if self._save_settings() is False:
                self.settings = previous
                return False
            self._sync_ui_from_settings()
            self._notify_manual_local_asr_fallback(
                fallback_channels,
                installation_fallback=installation_fallback,
            )
            self._log_manual_local_asr_fallbacks(previous, normalized, fallback_channels)
            return True
        scoped = copy.deepcopy(previous)
        if channel == "self":
            scoped.provider.peer_stt = STTProviderName.LOCAL_QWEN
        elif channel == "peer":
            scoped.provider.stt = STTProviderName.LOCAL_QWEN
        else:
            raise ValueError("channel must be 'self' or 'peer'")
        normalized_scoped, fallback_channels, installation_fallback = (
            self._normalize_manual_local_asr_fallbacks(scoped)
        )
        normalized = previous
        if channel == "self" and normalized_scoped.provider.stt != previous.provider.stt:
            normalized = copy.deepcopy(previous)
            normalized.provider.stt = normalized_scoped.provider.stt
        elif (
            channel == "peer" and normalized_scoped.provider.peer_stt != previous.provider.peer_stt
        ):
            normalized = copy.deepcopy(previous)
            normalized.provider.peer_stt = normalized_scoped.provider.peer_stt
        if normalized is previous:
            return True
        self.settings = normalized
        if self._save_settings() is False:
            self.settings = previous
            return False
        self._sync_ui_from_settings()
        self._notify_manual_local_asr_fallback(
            fallback_channels,
            installation_fallback=installation_fallback,
        )
        self._log_manual_local_asr_fallbacks(previous, normalized, fallback_channels)
        return True

    def _log_manual_local_asr_fallbacks(
        self,
        previous: AppSettings,
        normalized: AppSettings,
        channels: tuple[str, ...],
    ) -> None:
        for channel in channels:
            if channel == "self":
                requested = previous.provider.stt.value
                actual = normalized.provider.stt.value
                source_language = normalized.languages.source_language
            else:
                requested = previous.provider.peer_stt.value
                actual = normalized.provider.peer_stt.value
                source_language = normalized.languages.effective_peer_source
            decision = resolve_local_asr_selection(actual, source_language)
            self.log_basic(
                "[LocalASR][Selection] "
                f"channel={channel} requested={requested} actual={actual} "
                f"model={decision.model_id or 'unknown'} reason=preferred_model_unavailable"
            )

    def _required_local_stt_model_ids_for_provider(self, provider: str) -> tuple[str, ...]:
        if provider == LOCAL_CPU_AUTO_PROVIDER:
            return REQUIRED_CPU_LOCAL_STT_MODEL_IDS
        model_id = LOCAL_CPU_DIRECT_MODEL_BY_PROVIDER.get(provider)
        return (model_id,) if model_id is not None else ()

    def _sync_local_cpu_auto_availability(self, available: bool) -> None:
        self.app.set_settings_local_cpu_auto_available(available)

    def _local_stt_runtime_status_for_provider(self, provider: str) -> str:
        model_ids = self._required_local_stt_model_ids_for_provider(provider)
        if not model_ids:
            return "ready"
        return self._get_local_asr_provisioning_owner().snapshot.status_for(model_ids)

    def _current_local_stt_runtime_status(self) -> str:
        if self.settings is None:
            return "ready"
        return self._local_stt_runtime_status_for_provider(self.settings.provider.stt.value)

    def _local_stt_models_needing_install(self, provider: str) -> tuple[str, ...]:
        model_ids = self._required_local_stt_model_ids_for_provider(provider)
        return self._get_local_asr_provisioning_owner().snapshot.unavailable_model_ids(model_ids)

    def _peer_local_stt_requested(self, settings: AppSettings | None = None) -> bool:
        resolved_settings = settings or self.settings
        return bool(
            resolved_settings is not None
            and resolved_settings.provider.peer_stt.value in LOCAL_CPU_PROVIDERS
            and self._peer_translation_activation_requested_for(resolved_settings)
        )

    def _reset_local_stt_pending_enable_after_install(self) -> None:
        self._local_stt_pending_enable_after_install = False
        self._local_stt_pending_enable_generation = None

    def _reset_local_stt_pending_peer_enable_after_install(self) -> None:
        self._local_stt_pending_peer_enable_after_install = False

    def _self_local_stt_activation_is_current(self, generation: int | None) -> bool:
        return generation is None or (
            generation == self._stt_activation_generation and self._stt_desired
        )

    def _peer_local_stt_activation_is_current(self, generation: int | None) -> bool:
        return generation is None or (
            generation == self._peer_activation_generation
            and self._peer_local_stt_requested(self.settings)
        )

    def _clear_local_stt_pending_enable_if_provider_switched_away(self) -> None:
        if self.settings is None:
            return
        if self.settings.provider.stt.value not in LOCAL_CPU_PROVIDERS:
            self._reset_local_stt_pending_enable_after_install()
        if not self._peer_local_stt_requested(self.settings):
            self._reset_local_stt_pending_peer_enable_after_install()

    def _sync_local_stt_notice(self) -> None:
        if self.settings is None:
            return
        self_provider = self.settings.provider.stt.value
        peer_provider = self.settings.provider.peer_stt.value
        self_local = self_provider in LOCAL_CPU_PROVIDERS
        peer_local = self._peer_local_stt_requested(self.settings)
        self_local_asr = self_local or self_provider == STTProviderName.LOCAL_QWEN_GPU.value
        peer_local_asr = peer_local or (
            peer_provider == STTProviderName.LOCAL_QWEN_GPU.value
            and self._peer_translation_activation_requested_for(self.settings)
        )
        status = (
            self._local_stt_runtime_status_for_provider(self_provider)
            if self_local
            else self._local_stt_runtime_status_for_provider(peer_provider)
        )
        provisioning_snapshot = self._get_local_asr_provisioning_owner().snapshot
        visible_model_ids = self._required_local_stt_model_ids_for_provider(
            self_provider if self_local else peer_provider
        )
        activity = provisioning_snapshot.activity_for("cpu")
        notice_model_id = (
            activity.model_id
            if activity is not None and activity.model_id in visible_model_ids
            else next(
                (
                    model_id
                    for model_id in visible_model_ids
                    if provisioning_snapshot.state_for(model_id).status != "ready"
                ),
                None,
            )
        )
        if (self._stt_activation_starting or self._self_asr_model_loading) and self_local_asr:
            status = "self_loading"
        elif (self._peer_activation_starting or self._peer_asr_model_loading) and peer_local_asr:
            status = "peer_loading"
        elif self._stt_activation_failed and self_local_asr:
            status = "start_failed"
        should_show = status in {"self_loading", "peer_loading", "downloading"} or (
            (self_local_asr or peer_local_asr) and status != "ready"
        )
        with contextlib.suppress(Exception):
            self.app.set_dashboard_local_stt_notice(
                status=status if should_show else None,
                model_id=notice_model_id if should_show else None,
                percent=(
                    activity.progress_percent
                    if status == "downloading" and activity is not None
                    else None
                ),
                starting=self._stt_activation_starting or self._self_asr_model_loading,
            )

    def _request_local_asr_install(
        self,
        *,
        origin: str,
        model_ids: tuple[str, ...] | None = None,
    ) -> bool:
        if self.settings is None:
            return False
        provisioning = self._get_local_asr_provisioning_owner()
        if provisioning.snapshot.activity_for("cpu") is not None:
            return False
        requested_model_ids = model_ids or (LOCAL_STT_MODEL_ID,)
        try:
            provisioning.start_install(
                LocalASRInstallRequest(
                    backend="cpu",
                    model_ids=requested_model_ids,
                    locale=self.settings.ui.locale,
                    origin=origin,
                ),
                result_handler=lambda result: self._handle_local_stt_install_result(
                    result,
                    origin=origin,
                ),
            )
        except RuntimeError:
            return False
        return True

    async def _handle_local_stt_install_result(
        self,
        result: LocalASRInstallResult,
        *,
        origin: str,
    ) -> None:
        if result.cancelled:
            return
        if result.failed_model_ids:
            if origin == "manual":
                self._show_short_stt_message("local_stt.download_failed")
            return

        self._clear_local_stt_pending_enable_if_provider_switched_away()
        should_resume_self_local_stt = (
            origin == "manual"
            and self.settings is not None
            and self.settings.provider.stt.value in LOCAL_CPU_PROVIDERS
            and self._local_stt_runtime_status_for_provider(self.settings.provider.stt.value)
            == "ready"
            and self._local_stt_pending_enable_after_install
            and (
                self._local_stt_pending_enable_generation is None
                or self._local_stt_pending_enable_generation == self._stt_activation_generation
            )
        )
        should_resume_peer_local_stt = (
            origin == "manual"
            and self.settings is not None
            and self._peer_local_stt_requested(self.settings)
            and self._local_stt_runtime_status_for_provider(self.settings.provider.peer_stt.value)
            == "ready"
            and self._local_stt_pending_peer_enable_after_install
        )

        if should_resume_self_local_stt:
            resume_generation = (
                self._local_stt_pending_enable_generation
                if self._local_stt_pending_enable_generation is not None
                else self._stt_activation_generation
            )
            await self._rebuild_stt_provider()
            self._clear_local_stt_pending_enable_if_provider_switched_away()
            if (
                not self._local_stt_pending_enable_after_install
                or self._local_stt_pending_enable_generation != resume_generation
            ):
                return
            self._reset_local_stt_pending_enable_after_install()
            self._stt_desired = True
            self._stt_activation_starting = True
            self._sync_local_stt_notice()
            snapshot = await self._ensure_stt_switch()
            if snapshot is not None and snapshot.generation != self._stt_activation_generation:
                return
            self._stt_activation_starting = False
            self.app.set_dashboard_stt_enabled(
                bool(
                    self._stt_desired
                    and not self._stt_activation_failed
                    and self._mic_task is not None
                )
            )
            self._sync_local_stt_notice()

        if should_resume_peer_local_stt:
            self._reset_local_stt_pending_peer_enable_after_install()
            await self._refresh_overlay_runtime_dependencies()

    def _request_unavailable_local_asr_repair(
        self,
        status: str,
        *,
        channel: str,
        model_ids: tuple[str, ...] | None = None,
        activation_generation: int | None = None,
    ) -> bool:
        if self.settings is None:
            return False
        if channel == "self" and not self._self_local_stt_activation_is_current(
            activation_generation
        ):
            return False
        if channel == "peer" and not self._peer_local_stt_activation_is_current(
            activation_generation
        ):
            return False
        provider = (
            self.settings.provider.stt.value
            if channel == "self"
            else self.settings.provider.peer_stt.value
        )
        if channel == "self":
            self._local_stt_pending_enable_after_install = True
            self._local_stt_pending_enable_generation = (
                activation_generation
                if activation_generation is not None
                else self._stt_activation_generation
            )
            self._stt_desired = False
        else:
            self._local_stt_pending_peer_enable_after_install = True
        affected_model_ids = model_ids or self._local_stt_models_needing_install(provider)
        if not affected_model_ids:
            affected_model_ids = self._required_local_stt_model_ids_for_provider(provider)
        if channel == "self":
            self.app.set_dashboard_stt_enabled(False)
            self.app.set_dashboard_stt_needs_key(False)
        self._sync_local_stt_notice()
        self._request_local_asr_install(
            origin="manual",
            model_ids=affected_model_ids,
        )
        return False

    @staticmethod
    def _snapshot_unavailable_status(
        snapshot: LocalASRProvisioningSnapshot,
        model_ids: tuple[str, ...],
    ) -> str:
        if any(snapshot.state_for(model_id).integrity == "invalid" for model_id in model_ids):
            return "invalid"
        return "missing"

    @staticmethod
    def _snapshot_unavailable_model_ids(
        snapshot: LocalASRProvisioningSnapshot,
        model_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        return snapshot.unavailable_model_ids(model_ids)

    async def _probe_self_local_stt_runtime_load(
        self,
        *,
        activation_generation: int | None = None,
    ) -> None:
        if not self._self_local_stt_activation_is_current(activation_generation):
            return
        if self.hub is None or not self._hub_has_stt_provider("self"):
            raise RuntimeError("self STT provider is unavailable")
        if self._hub_local_asr_provider_runtime() is None:
            raise RuntimeError("local ASR provider runtime is unavailable")
        await self.hub.warmup_stt_channel("self")

    async def _ensure_local_stt_ready(
        self,
        *,
        activation_generation: int | None = None,
    ) -> bool:
        if not self._self_local_stt_activation_is_current(activation_generation):
            return False
        if self.settings is None or self.settings.provider.stt.value not in LOCAL_CPU_PROVIDERS:
            return True
        decision = resolve_local_asr_selection(
            self.settings.provider.stt.value,
            self.settings.languages.source_language,
        )
        if not decision.supported:
            self._stt_desired = False
            self.app.set_dashboard_stt_enabled(False)
            self.app.set_dashboard_stt_needs_key(False)
            self._show_short_stt_message("local_stt.language_unsupported")
            return False
        current_status = self._current_local_stt_runtime_status()
        if current_status == "downloading":
            self._stt_desired = False
            self._local_stt_pending_enable_after_install = True
            self._local_stt_pending_enable_generation = (
                activation_generation
                if activation_generation is not None
                else self._stt_activation_generation
            )
            self.app.set_dashboard_stt_enabled(False)
            self._show_short_stt_message("local_stt.download_in_progress")
            return False
        if current_status in ("missing", "invalid", "download_failed"):
            return self._request_unavailable_local_asr_repair(
                current_status,
                channel="self",
                activation_generation=activation_generation,
            )
        if self.hub is not None and not self._hub_has_stt_provider("self"):
            await self._rebuild_stt_provider()
            if not self._self_local_stt_activation_is_current(activation_generation):
                return False
        if self.hub is None or not self._hub_has_stt_provider("self"):
            self._stt_desired = False
            self.app.set_dashboard_stt_enabled(False)
            self.app.set_dashboard_stt_needs_key(False)
            self._show_short_stt_message("error.local_stt_model_invalid")
            return False
        runtime = self._hub_local_asr_provider_runtime()
        if runtime is None:
            raise RuntimeError("local ASR provider runtime is unavailable")
        channel_snapshot = runtime.snapshot.channel_for("self")
        was_loaded = channel_snapshot.phase in {"ready", "running"}
        load_started_at = time.monotonic()
        try:
            await self._probe_self_local_stt_runtime_load(
                activation_generation=activation_generation
            )
            if not self._self_local_stt_activation_is_current(activation_generation):
                return False
            loaded_model_id = channel_snapshot.model_id or decision.model_id
            if not was_loaded:
                self._log_local_asr_load_result(
                    channel="self",
                    model_id=str(loaded_model_id or "unknown"),
                    backend="CPU",
                    outcome="ready",
                    load_seconds=time.monotonic() - load_started_at,
                )
            await self._get_local_asr_provisioning_owner().inspect_cpu(
                self._required_local_stt_model_ids_for_provider(self.settings.provider.stt.value)
            )
            self._sync_local_stt_notice()
            return True
        except LocalCPUAutoUnavailableError:
            required_model_ids = self._required_local_stt_model_ids_for_provider(
                self.settings.provider.stt.value
            )
            snapshot = await self._get_local_asr_provisioning_owner().inspect_cpu(
                required_model_ids
            )
            if (
                self.settings is not None
                and self.settings.provider.stt == STTProviderName.LOCAL_CPU_AUTO
            ):
                if not self._persist_current_manual_local_asr_fallback(channel="self"):
                    return False
                await self._rebuild_stt_provider()
                return await self._ensure_local_stt_ready(
                    activation_generation=activation_generation
                )
            return self._request_unavailable_local_asr_repair(
                self._snapshot_unavailable_status(snapshot, required_model_ids),
                channel="self",
                model_ids=self._snapshot_unavailable_model_ids(snapshot, required_model_ids),
                activation_generation=activation_generation,
            )
        except LocalSTTModelMissingError as exc:
            self._log_local_asr_load_result(
                channel="self",
                model_id=str(decision.model_id or "unknown"),
                backend="CPU",
                outcome="failed",
                load_seconds=time.monotonic() - load_started_at,
                failure_type=type(exc).__name__,
            )
            return self._request_unavailable_local_asr_repair(
                "missing",
                channel="self",
                model_ids=((decision.model_id,) if decision.model_id is not None else None),
                activation_generation=activation_generation,
            )
        except (
            LocalSTTManifestInvalidError,
            LocalQwenSherpaLoadError,
            LocalParakeetSherpaLoadError,
        ) as exc:
            self._log_local_asr_load_result(
                channel="self",
                model_id=str(decision.model_id or "unknown"),
                backend="CPU",
                outcome="failed",
                load_seconds=time.monotonic() - load_started_at,
                failure_type=type(exc).__name__,
            )
            if decision.model_id is not None:
                await self._get_local_asr_provisioning_owner().report_model_validation_failure(
                    decision.model_id,
                    failure_type=type(exc).__name__,
                )
            return self._request_unavailable_local_asr_repair(
                "invalid",
                channel="self",
                model_ids=((decision.model_id,) if decision.model_id is not None else None),
                activation_generation=activation_generation,
            )

    async def _ensure_peer_local_stt_ready(
        self,
        *,
        activation_generation: int | None = None,
    ) -> bool:
        if not self._peer_local_stt_activation_is_current(activation_generation):
            return False
        if (
            self.settings is not None
            and self.settings.provider.peer_stt == STTProviderName.LOCAL_QWEN_GPU
        ):
            ready = await self._validate_gpu_activation()
            if not ready and self._gpu_ui_state in {
                "not_installed",
                "invalid",
                "install_failed",
                "installing",
            }:
                self._gpu_pending_enable_channels = frozenset(
                    {*self._gpu_pending_enable_channels, "peer"}
                )
            return ready
        if self.settings is None or not self._peer_local_stt_requested(self.settings):
            return True
        decision = resolve_local_asr_selection(
            self.settings.provider.peer_stt.value,
            self.settings.languages.effective_peer_source,
        )
        if not decision.supported:
            self.settings.ui.peer_translation_enabled = False
            self._last_peer_translation_enabled = False
            self._last_peer_translation_activation_requested = False
            self._sync_effective_hub_flags(self.settings)
            self._show_short_stt_message("local_stt.language_unsupported")
            return False
        current_status = self._local_stt_runtime_status_for_provider(
            self.settings.provider.peer_stt.value
        )
        if current_status == "downloading":
            self._local_stt_pending_peer_enable_after_install = True
            self._sync_local_stt_notice()
            return False
        if current_status in ("missing", "invalid", "download_failed"):
            return self._request_unavailable_local_asr_repair(
                current_status,
                channel="peer",
                activation_generation=activation_generation,
            )
        required_model_ids = self._required_local_stt_model_ids_for_provider(
            self.settings.provider.peer_stt.value
        )
        strict_snapshot = await self._get_local_asr_provisioning_owner().inspect_cpu(
            required_model_ids
        )
        if not self._peer_local_stt_activation_is_current(activation_generation):
            return False
        unavailable_model_ids = self._snapshot_unavailable_model_ids(
            strict_snapshot,
            required_model_ids,
        )
        if unavailable_model_ids:
            return self._request_unavailable_local_asr_repair(
                self._snapshot_unavailable_status(strict_snapshot, required_model_ids),
                channel="peer",
                model_ids=unavailable_model_ids,
                activation_generation=activation_generation,
            )
        self._sync_local_stt_notice()
        return True

    async def _close_local_asr_provisioning(self) -> None:
        self._reset_local_stt_pending_enable_after_install()
        self._reset_local_stt_pending_peer_enable_after_install()
        if self.local_asr_provisioning is not None:
            await self.local_asr_provisioning.close()

    async def _ensure_stt_switch(self) -> SelfCaptureSessionSnapshot | None:
        return await self._run_stt_switch()

    async def _replace_runtime_stt_provider(self, *, smooth_local: bool = False) -> None:
        self.log_detailed(
            "[STT] Replacing runtime provider detail: "
            f"desired={self._stt_desired} mic_task_active={self._mic_task is not None}"
        )
        if self.settings is None or self.hub is None:
            return
        owner = self._get_self_capture_owner()
        config = self._build_self_capture_session_config(self.settings)
        if self._stt_desired:
            snapshot = await owner.apply_intent(
                config,
                enabled=True,
                restart=not smooth_local,
                explicit_toggle_off=False,
            )
        else:
            snapshot = await owner.prepare_provider(config)
        self._on_self_capture_state_changed(snapshot)
        self._project_self_provider_availability(snapshot)
        self._stt_restart_requested = False

    def _project_self_provider_availability(
        self,
        snapshot: SelfCaptureSessionSnapshot,
    ) -> bool:
        available = snapshot.provider_status is SelfCaptureProviderStatus.READY
        self._sync_effective_hub_flags(self.settings)
        self.app.set_dashboard_stt_needs_key(self._dashboard_stt_needs_key(stt_available=available))
        if not available:
            self.app.set_dashboard_stt_enabled(False)
        return available

    async def _apply_stt_runtime_replacement(self, *, smooth_local: bool) -> None:
        replacement = self._replace_runtime_stt_provider
        try:
            inspect.signature(replacement).bind(smooth_local=smooth_local)
        except (TypeError, ValueError):
            await replacement()
            return
        await replacement(smooth_local=smooth_local)

    async def _transition_active_self_local_asr(self) -> bool:
        if (
            not self._stt_desired
            or self.hub is None
            or self.settings is None
            or not self._hub_has_stt_provider("self")
        ):
            return False
        transition_settings = self.settings
        request = self._self_local_asr_transition_request(
            self.settings,
            trigger="settings",
        )
        if request is None:
            return False
        runtime = self._hub_local_asr_provider_runtime()
        if runtime is None:
            raise RuntimeError("local ASR provider runtime is unavailable")
        current_model_id = runtime.snapshot.channel_for("self").model_id
        if current_model_id is None:
            return False
        if current_model_id == request.model_id:
            await self.hub.reconfigure_stt_channel("self", request.session_options)
            self.log_detailed(
                "[LocalASR][Transition] "
                f"channel=self requested_provider={request.requested_provider} "
                f"actual_provider={request.actual_provider} model_id={request.model_id} "
                "outcome=reconfigured"
            )
            return True

        coordinator = self._self_local_asr_transition
        coordinator.diagnostic_sink = self._local_asr_transition_diagnostic
        target_settings = copy.deepcopy(self.settings)
        self._self_asr_model_loading = True
        self._sync_local_stt_notice()

        async def prepare_owned(
            prepared_request: LocalASRTransitionRequest,
            generation: int,
        ) -> PreparedLocalASRTransition:
            return PreparedLocalASRTransition(
                request=prepared_request,
                provider=self._self_stt_provider_request(target_settings, warmup=True),
                generation=generation,
            )

        async def commit_owned(prepared: PreparedLocalASRTransition) -> None:
            if not isinstance(prepared.provider, ProviderRuntimeBuildRequest):
                raise TypeError("owned Self STT transition requires a build request")
            try:
                result = await self.hub.handoff_stt_provider_request(
                    prepared.provider,
                    start=True,
                )
            except asyncio.CancelledError:
                await self.hub.cancel_stt_provider_request_handoff()
                raise
            if result.status != "applied":
                raise RuntimeError("owned Self STT handoff failed")

        try:
            outcome = await coordinator.request_transition(
                request,
                prepare=prepare_owned,
                commit=commit_owned,
            )
        finally:
            self._self_asr_model_loading = False
            self._sync_local_stt_notice()
        if outcome.status == "failed":
            self._stt_activation_failed = True
            self._sync_local_stt_notice()
            raise RuntimeError("local ASR transition failed")
        if outcome.status == "superseded":
            self._superseded_local_asr_settings_ids.add(id(transition_settings))
            raise _LocalASRTransitionSuperseded
        if outcome.status == "closed":
            raise RuntimeError("local ASR transition coordinator is closed")
        return outcome.status == "applied"

    async def _run_stt_switch(self) -> SelfCaptureSessionSnapshot | None:
        if self.settings is None:
            self.log_detailed(
                "[STT] Enable requested before hub is ready",
                level=logging.WARNING,
            )
            self._stt_desired = False
            self._stt_activation_failed = True
            return None
        desired = self._stt_desired
        restart = self._stt_restart_requested
        force_immediate = self._stt_force_immediate
        self._stt_restart_requested = False
        self._stt_force_immediate = False
        owner = self._get_self_capture_owner()
        snapshot = await owner.apply_intent(
            self._build_self_capture_session_config(self.settings),
            enabled=desired,
            restart=restart,
            force_immediate=force_immediate,
            explicit_toggle_off=not desired,
        )
        self._on_self_capture_state_changed(snapshot)
        return snapshot

    def _get_clipboard_watcher_lock(self) -> asyncio.Lock:
        return self._get_clipboard_auto_translation_owner().lock

    def _get_clipboard_runtime(self) -> ClipboardRuntime:
        return self._get_clipboard_auto_translation_owner().get_runtime()

    @property
    def _clipboard_runtime(self) -> ClipboardRuntime | None:
        owner = self._clipboard_auto_translation_owner
        return owner.runtime if owner is not None else None

    @_clipboard_runtime.setter
    def _clipboard_runtime(self, runtime: ClipboardRuntime | None) -> None:
        self._get_clipboard_auto_translation_owner().runtime = runtime

    def _get_clipboard_auto_translation_owner(self) -> ClipboardAutoTranslationOwner:
        owner = self._clipboard_auto_translation_owner
        if owner is None:
            owner = ClipboardAutoTranslationOwner(
                watcher_factory=create_clipboard_watcher,
                submit_text=self._submit_clipboard_text_to_hub,
                failure_sink=self._log_error,
                platform_provider=lambda: sys.platform,
            )
            self._clipboard_auto_translation_owner = owner
        return owner

    async def _sync_clipboard_watcher(self) -> None:
        enabled = bool(
            self.settings is not None and self.settings.ui.clipboard_auto_translate_enabled
        )
        await self._get_clipboard_auto_translation_owner().sync(enabled=enabled)

    async def _sync_clipboard_watcher_with_policy(self, *, strict_runtime_errors: bool) -> None:
        owner = self._get_clipboard_auto_translation_owner()
        previous_strict_runtime_errors = owner.strict_runtime_errors
        owner.strict_runtime_errors = strict_runtime_errors
        try:
            await self._sync_clipboard_watcher()
        finally:
            owner.strict_runtime_errors = previous_strict_runtime_errors

    async def _stop_clipboard_watcher(self) -> None:
        owner = self._clipboard_auto_translation_owner
        if owner is not None:
            await owner.stop()

    async def _close_clipboard_runtime(self) -> None:
        owner = self._clipboard_auto_translation_owner
        if owner is not None:
            await owner.close()

    def _on_clipboard_text_from_thread(self, text: str) -> None:
        self._get_clipboard_auto_translation_owner().on_text_from_thread(text)

    def _schedule_clipboard_submit(self, text: str) -> None:
        self._get_clipboard_auto_translation_owner().submit_from_loop(text)

    async def _submit_clipboard_text(self, text: str) -> None:
        await self._get_clipboard_auto_translation_owner().submit_now(text)

    async def _submit_clipboard_text_to_hub(self, text: str) -> None:
        if self.hub is None:
            return
        await self.hub.submit_text(text, source="Clipboard")

    async def submit_text(self, text: str) -> None:
        hub = self.hub
        submit = None if hub is None else lambda: hub.submit_text(text, source="You")
        await self._get_manual_typing_owner().submit(submit)

    def set_manual_input_activity(self, has_text: bool) -> None:
        self._get_manual_typing_owner().set_input_activity(has_text)

    async def release_manual_typing(self) -> None:
        await self._get_manual_typing_owner().release()

    def _begin_manual_submit_typing(self) -> str:
        return self._get_manual_typing_owner().begin_submit()

    @property
    def _manual_typing_idle_task(self) -> asyncio.Task[None] | None:
        owner = self._manual_typing_owner
        return owner.idle_task if owner is not None else None

    def _get_manual_typing_owner(self) -> ManualTypingOwner:
        owner = self._manual_typing_owner
        if owner is None:

            def output_provider():
                hub = self.hub
                set_reason = getattr(hub, "set_self_chatbox_typing_reason", None)
                clear_reasons = getattr(hub, "clear_self_chatbox_typing_reasons", None)
                if callable(set_reason) and callable(clear_reasons):
                    return hub
                return None

            def completion_provider(utterance_id: object) -> object | None:
                hub = self.hub
                runtime = getattr(hub, "self_runtime", None)
                tasks = getattr(runtime, "translation_tasks", None)
                return tasks.get(utterance_id) if isinstance(tasks, dict) else None

            owner = ManualTypingOwner(
                output_provider=output_provider,
                completion_provider=completion_provider,
                log_detailed=lambda message: self.log_detailed(message),
                log_error=lambda message: self._log_error(message),
                idle_timeout_seconds=MANUAL_INPUT_TYPING_IDLE_TIMEOUT_S,
                submit_timeout_seconds=MANUAL_SUBMIT_TYPING_TIMEOUT_S,
            )
            self._manual_typing_owner = owner
        return owner

    async def on_dashboard_language_change(
        self,
        change: LanguageSelectionChange,
    ) -> None:
        if self.settings is None:
            return

        updated = copy.deepcopy(self.settings)
        updated.languages.source_language = change.source_code
        updated.languages.target_language = change.target_code
        updated.languages.peer_source_mode = change.peer_source_mode
        updated.languages.peer_source_language = change.peer_source_code
        updated.languages.peer_target_language = change.peer_target_code
        updated.languages.recent_source_languages = list(change.recent_source_codes)
        updated.languages.recent_target_languages = list(change.recent_target_codes)
        self._begin_canonical_mutation(
            legacy_snapshot=self._canonical_legacy_projection_snapshot or self.settings
        )
        self._capture_runtime_signatures_before_canonical_mutation()
        self._update_canonical_settings_from_legacy_delta(self.settings, updated)
        try:
            await self.apply_settings(updated)
            if self.settings is not None:
                self.refresh_settings_projection(preserve_custom_vocab_draft=True)
        finally:
            self._complete_canonical_mutation()

    def _settings_projection(self) -> SettingsProjectionOwner:
        owner = self._settings_projection_owner
        if owner is None:
            owner = SettingsProjectionOwner(
                presentation=self.app,
                config_path=self.config_path,
                current_settings=lambda: self.settings,
            )
            self._settings_projection_owner = owner
        return owner

    def capture_settings_view_change(
        self,
        pending_settings: AppSettings,
    ) -> SettingsViewSettingsChange:
        return self._settings_projection().capture(pending_settings)

    def merge_settings_view_change_with_current(
        self,
        change: SettingsViewSettingsChange,
    ) -> AppSettings:
        return cast(AppSettings, self._settings_projection().merge_with_current(change))

    def refresh_settings_projection(
        self,
        *,
        preserve_custom_vocab_draft: bool = False,
    ) -> bool:
        settings = self.settings
        if settings is None:
            return False
        return bool(
            self._settings_projection().render(
                settings,
                preserve_custom_vocab_draft=preserve_custom_vocab_draft,
            )
        )

    def refresh_settings_after_openrouter_pkce_success(self) -> bool:
        settings = self.settings
        if settings is None:
            return False
        return bool(self._settings_projection().refresh_after_openrouter_pkce_success(settings))

    def _sync_memory_runtime_fields_from_settings(self, settings: AppSettings) -> None:
        restored_settings = copy.deepcopy(settings)
        self.settings = restored_settings
        self._sync_overlay_calibration_cache(restored_settings)
        if self.hub is not None:
            self.hub.source_language = restored_settings.languages.source_language
            self.hub.target_language = restored_settings.languages.target_language
            self.hub.peer_source_language = restored_settings.languages.peer_source_language
            self.hub.peer_target_language = restored_settings.languages.peer_target_language
            self.hub.system_prompt = restored_settings.system_prompt
            self.hub.low_latency_mode = FIXED_TRANSLATION_POLICY.fast_translation_enabled
            self.hub.low_latency_merge_gap_ms = restored_settings.stt.low_latency_merge_gap_ms
            self.hub.low_latency_spec_retry_max = restored_settings.stt.low_latency_spec_retry_max
            self.hub.hangover_s = (
                restored_settings.stt.low_latency_vad_hangover_ms / 1000.0
                if FIXED_TRANSLATION_POLICY.fast_translation_enabled
                else DEFAULT_STABLE_VAD_HANGOVER_MS / 1000.0
            )
            self.hub.peer_hangover_s = restored_settings.desktop_audio.vad_hangover_ms / 1000.0
            self.hub.chatbox_include_source = restored_settings.osc.chatbox_include_source
            self._sync_effective_hub_flags(restored_settings)
        self._sync_signature_caches(restored_settings)

    async def _resync_committed_order22_settings_after_strict_save_failure(
        self,
        *,
        base_settings: AppSettings,
        committed_settings: AppSettings,
    ) -> None:
        self._sync_memory_runtime_fields_from_settings(base_settings)
        try:
            await self._apply_settings_direct(
                copy.deepcopy(committed_settings),
                persist=False,
                strict_runtime_errors=True,
            )
        except Exception:
            self._log_error("Failed to resync committed order22 settings runtime")
            self._sync_memory_runtime_fields_from_settings(committed_settings)

    async def _resync_committed_order22_provider_runtime_after_strict_save_failure(
        self,
        *,
        base_settings: AppSettings,
        committed_settings: AppSettings,
        plan: _ProviderRuntimeApplyPlan,
    ) -> None:
        self._sync_memory_runtime_fields_from_settings(base_settings)
        try:
            await self._apply_provider_runtime_plan(copy.deepcopy(committed_settings), plan)
        except Exception:
            self._log_error("Failed to resync committed order22 provider runtime")
            self._sync_memory_runtime_fields_from_settings(committed_settings)

    async def _resync_committed_order23_settings_after_strict_save_failure(
        self,
        *,
        base_settings: AppSettings,
        committed_settings: AppSettings,
    ) -> None:
        self._sync_memory_runtime_fields_from_settings(base_settings)
        try:
            await self._apply_settings_direct(
                copy.deepcopy(committed_settings),
                persist=False,
                strict_runtime_errors=True,
            )
        except Exception:
            self._log_error("Failed to resync committed order23 settings runtime")
            self._sync_memory_runtime_fields_from_settings(committed_settings)

    async def _resync_committed_order24_settings_after_strict_save_failure(
        self,
        *,
        base_settings: AppSettings,
        committed_settings: AppSettings,
    ) -> None:
        self._sync_memory_runtime_fields_from_settings(base_settings)
        try:
            await self._apply_settings_direct(
                copy.deepcopy(committed_settings),
                persist=False,
                strict_runtime_errors=True,
            )
        except Exception:
            self._log_error("Failed to resync committed order24 settings runtime")
            self._sync_memory_runtime_fields_from_settings(committed_settings)

    async def apply_settings(self, settings: AppSettings) -> None:
        owner = self._get_settings_owner()
        if self.settings is not None:
            owner.normalize_compatibility(self.settings)
        owner.normalize_compatibility(settings)
        fallback_channels: tuple[str, ...] = ()
        installation_fallback = False
        normalization_channels = self._manual_local_asr_fallback_normalization_channels(settings)
        if normalization_channels:
            await self._get_local_asr_provisioning_owner().inspect_cpu()
            normalized_settings, normalized_channels, installation_fallback = (
                self._normalize_manual_local_asr_fallbacks(settings)
            )
            fallback_channels = tuple(
                channel for channel in normalized_channels if channel in normalization_channels
            )
            if fallback_channels:
                scoped_settings = copy.deepcopy(settings)
                if "self" in fallback_channels:
                    scoped_settings.provider.stt = normalized_settings.provider.stt
                if "peer" in fallback_channels:
                    scoped_settings.provider.peer_stt = normalized_settings.provider.peer_stt
                settings = scoped_settings
            installation_fallback = bool(installation_fallback and fallback_channels)
        if settings is not self.settings:
            routed = await self._apply_order22_order23_order24_settings_via_mutation_services(
                settings
            )
            if routed:
                self._notify_manual_local_asr_fallback(
                    fallback_channels,
                    installation_fallback=installation_fallback,
                )
                return
            routed = await self._apply_stt_language_audio_settings_via_mutation_service(settings)
            if routed:
                self._notify_manual_local_asr_fallback(
                    fallback_channels,
                    installation_fallback=installation_fallback,
                )
                return
            routed = await self._apply_overlay_osc_output_settings_via_mutation_service(settings)
            if routed:
                self._notify_manual_local_asr_fallback(
                    fallback_channels,
                    installation_fallback=installation_fallback,
                )
                return
            routed = await self._apply_ui_prompt_clipboard_state_settings_via_mutation_service(
                settings
            )
            if routed:
                self._notify_manual_local_asr_fallback(
                    fallback_channels,
                    installation_fallback=installation_fallback,
                )
                return
        await self._apply_settings_direct(settings)
        self._notify_manual_local_asr_fallback(
            fallback_channels,
            installation_fallback=installation_fallback,
        )

    async def apply_telemetry_consent(self, consent: str) -> AppSettings | None:
        if self.settings is None:
            return None
        await self.apply_settings(with_telemetry_consent(self.settings, consent))
        return self.settings

    async def _apply_settings_direct(
        self,
        settings: AppSettings,
        *,
        persist: bool = True,
        strict_runtime_errors: bool = False,
        strict_persistence_errors: bool = False,
        reload_settings_view: bool = True,
    ) -> None:
        await self._preserve_github_star_prompt_observation_before_settings_replace(settings)
        if persist:
            self._begin_canonical_mutation(
                legacy_snapshot=self._canonical_legacy_projection_snapshot or self.settings
            )
            self._capture_runtime_signatures_before_canonical_mutation()
            self._update_canonical_settings_from_legacy_delta(
                self._canonical_legacy_projection_snapshot or self.settings,
                settings,
            )

        def _effective_peer_language(language: str, peer_language: str) -> str:
            return peer_language or language

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

        prev_locale = self.app.current_locale()
        prev_overlay_enabled = (
            self.settings.ui.overlay_enabled if self.settings is not None else False
        )
        previous_settings_for_desktop = (
            copy.deepcopy(self.settings) if self.settings is not None else None
        )
        prev_settings_overlay_target = self._overlay_target_for_settings(self.settings)
        next_overlay_target = self._overlay_target_for_settings(settings)
        if self._get_overlay_session_fallback_owner().active:
            # Session fallback keeps settings on SteamVR while runtime is desktop.
            # Compare settings targets so unrelated applies do not look like a switch.
            prev_overlay_target = prev_settings_overlay_target
        else:
            prev_overlay_target = self._previous_overlay_target_for_apply()
        if next_overlay_target == OVERLAY_TARGET_DESKTOP:
            self._clear_overlay_session_desktop_fallback()
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
            self._clear_overlay_session_desktop_fallback()
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
        prev_peer_source_mode = (
            previous_settings_for_desktop.languages.peer_source_mode
            if previous_settings_for_desktop is not None
            else None
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
        peer_source_language_changed = (
            prev_peer_source_lang is not None
            and prev_peer_source_lang != settings.languages.peer_source_language
        )
        peer_target_language_changed = (
            prev_peer_target_lang is not None
            and prev_peer_target_lang != settings.languages.peer_target_language
        )
        peer_source_mode_changed = (
            prev_peer_source_mode is not None
            and prev_peer_source_mode != settings.languages.peer_source_mode
        )
        if source_language_changed or target_language_changed:
            presenter = self._current_overlay_presenter_for_direct_runtime_command()
            bridge = self._current_overlay_bridge_for_direct_runtime_command()
            self.log_basic(
                "[Settings] Applying languages: "
                f"source={prev_source_lang}->{settings.languages.source_language} "
                f"target={prev_target_lang}->{settings.languages.target_language}"
            )
            self.log_detailed(
                "[Settings] Language apply detail: "
                f"overlay_state={self.overlay_state} "
                f"presenter_attached={presenter is not None} "
                f"bridge_attached={bridge is not None} "
                "overlay_sink_matches_presenter="
                f"{self.hub is not None and presenter is not None and getattr(self.hub, 'overlay_sink', None) is presenter}"
            )
        self.settings = settings
        self._last_microphone_test_audio_settings_signature = next_microphone_test_audio_signature
        self._sync_overlay_calibration_cache(settings)
        self._sync_desktop_overlay_interaction_mode_from_settings(settings)
        if persist:
            if strict_persistence_errors:
                try:
                    self._persist_settings_at_controller_boundary(self.settings)
                except Exception:
                    self._rollback_canonical_mutation()
                    raise _StrictSettingsSaveFailed from None
                else:
                    self._remember_canonical_legacy_projection(self.settings)
            else:
                if self._save_settings() is False:
                    return
        await self._broadcast_desktop_runtime_control_payloads(desktop_runtime_controls)
        await self._sync_clipboard_watcher_with_policy(
            strict_runtime_errors=strict_runtime_errors,
        )
        provisioning = self._get_local_asr_provisioning_owner()
        await provisioning.inspect_cpu()
        await provisioning.inspect_gpu(
            explicit_intent=self._selected_gpu_provider_requires_model(),
        )
        self._clear_local_stt_pending_enable_if_provider_switched_away()

        if self.hub is not None:
            self.hub.source_language = settings.languages.source_language
            self.hub.target_language = settings.languages.target_language
            self.hub.peer_source_language = settings.languages.peer_source_language
            self.hub.peer_target_language = settings.languages.peer_target_language
            self.hub.system_prompt = settings.system_prompt
            self.hub.low_latency_mode = FIXED_TRANSLATION_POLICY.fast_translation_enabled
            self.hub.low_latency_merge_gap_ms = settings.stt.low_latency_merge_gap_ms
            self.hub.low_latency_spec_retry_max = settings.stt.low_latency_spec_retry_max
            self.hub.hangover_s = (
                settings.stt.low_latency_vad_hangover_ms / 1000.0
                if FIXED_TRANSLATION_POLICY.fast_translation_enabled
                else DEFAULT_STABLE_VAD_HANGOVER_MS / 1000.0
            )
            self.hub.peer_hangover_s = settings.desktop_audio.vad_hangover_ms / 1000.0
            self.hub.chatbox_include_source = settings.osc.chatbox_include_source
            self._sync_effective_hub_flags(settings)

            async def _clear_language_runtime_state(channel: str) -> None:
                try:
                    await self.hub.clear_language_runtime_state(channel=channel)
                except Exception as exc:
                    if strict_runtime_errors:
                        self._log_error(f"Failed to clear language runtime state for {channel}")
                    else:
                        self._log_error(
                            f"Failed to clear language runtime state for {channel}: {exc}"
                        )
                    if strict_runtime_errors:
                        raise

            if source_language_changed or target_language_changed:
                await _clear_language_runtime_state("self")
            if effective_peer_source_changed or effective_peer_target_changed:
                await _clear_language_runtime_state("peer")

        presenter = self._current_overlay_presenter_for_direct_runtime_command()
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
            smooth_local = bool(
                previous_settings_for_desktop is not None
                and self._self_capture_vad_signature(previous_settings_for_desktop)
                == self._self_capture_vad_signature(settings)
            )
            await self._apply_stt_runtime_replacement(smooth_local=smooth_local)

        if reload_settings_view and (
            source_language_changed
            or target_language_changed
            or peer_source_language_changed
            or peer_target_language_changed
            or peer_source_mode_changed
        ):
            self._settings_projection().render(
                settings,
                preserve_custom_vocab_draft=True,
            )

        if prev_locale != settings.ui.locale:
            self.app.set_locale(settings.ui.locale)
            try:
                self.app.apply_locale()
            except Exception:
                self._log_error("Failed to apply locale")
                if strict_runtime_errors:
                    raise

        self._refresh_overlay_peer_consumers()
        self._settings_projection().remember_all(self.settings)
        if persist:
            self._complete_canonical_mutation()

    async def _apply_order22_order23_order24_settings_via_mutation_services(
        self,
        next_settings: AppSettings,
    ) -> bool:
        projection = self._settings_projection()
        order22_base_and_patch = projection.order22_patch_base_and_values(next_settings)
        order23_base_and_patch = projection.order23_patch_base_and_values(next_settings)
        order24_base_and_patch = projection.order24_patch_base_and_values(next_settings)
        if (
            order22_base_and_patch is None
            or order23_base_and_patch is None
            or order24_base_and_patch is None
        ):
            return False
        _order22_base_settings, order22_patch_values = order22_base_and_patch
        _order23_base_settings, order23_patch_values = order23_base_and_patch
        _order24_base_settings, order24_patch_values = order24_base_and_patch
        patch_count = sum(
            1
            for patch_values in (
                order22_patch_values,
                order23_patch_values,
                order24_patch_values,
            )
            if patch_values
        )
        if patch_count < 2:
            return False

        committed_results: list[TransactionResult] = []

        async def _route_patch_only(
            patch_values: dict[str, object],
            route,
            *,
            route_stt: bool = False,
            reload_settings_view: bool = True,
            runtime_only_source: AppSettings | None = None,
        ) -> bool:
            if self.settings is None:
                return False
            patch_only_settings = copy.deepcopy(self.settings)
            _apply_settings_path_patch(patch_only_settings, patch_values)
            if runtime_only_source is not None:
                _copy_runtime_only_ui_state(runtime_only_source, patch_only_settings)
            if route_stt:
                routed = await route(
                    patch_only_settings,
                    reload_settings_view=reload_settings_view,
                )
            else:
                routed = await route(patch_only_settings)
            if not routed:
                return False
            result = self.last_settings_mutation_result
            if result is None or not _settings_mutation_committed(result):
                return True
            committed_results.append(result)
            return True

        if order22_patch_values:
            routed_order22 = await _route_patch_only(
                order22_patch_values,
                self._apply_stt_language_audio_settings_via_mutation_service,
                route_stt=True,
                reload_settings_view=False,
            )
            if not routed_order22:
                return False
            if self.last_settings_mutation_result is None or not _settings_mutation_committed(
                self.last_settings_mutation_result,
            ):
                return True

        if order23_patch_values:
            routed_order23 = await _route_patch_only(
                order23_patch_values,
                self._apply_overlay_osc_output_settings_via_mutation_service,
            )
            if not routed_order23:
                return False
            if self.last_settings_mutation_result is None or not _settings_mutation_committed(
                self.last_settings_mutation_result,
            ):
                return True

        if order24_patch_values:
            routed_order24 = await _route_patch_only(
                order24_patch_values,
                self._apply_ui_prompt_clipboard_state_settings_via_mutation_service,
                runtime_only_source=next_settings,
            )
            if not routed_order24:
                return False
            if self.last_settings_mutation_result is None or not _settings_mutation_committed(
                self.last_settings_mutation_result,
            ):
                return True

        committed_settings_before_full_draft = (
            copy.deepcopy(self.settings) if self.settings is not None else None
        )
        if self.settings is not None and self._get_settings_owner().legacy_snapshot_values(
            self.settings
        ) != self._get_settings_owner().legacy_snapshot_values(next_settings):
            try:
                await self._apply_settings_direct(
                    next_settings,
                    strict_runtime_errors=True,
                    strict_persistence_errors=True,
                )
            except _StrictSettingsSaveFailed:
                if committed_settings_before_full_draft is not None:
                    await self._resync_committed_order24_settings_after_strict_save_failure(
                        base_settings=committed_settings_before_full_draft,
                        committed_settings=committed_settings_before_full_draft,
                    )
                self.last_settings_mutation_result = (
                    _ui_prompt_clipboard_state_save_failed_transaction_result(
                        operation="apply_order22_order23_order24_full_draft_save"
                    )
                )
            except Exception:
                self.last_settings_mutation_result = (
                    _ui_prompt_clipboard_state_runtime_degraded_transaction_result()
                )

        if self.settings is not None:
            self.refresh_settings_projection(preserve_custom_vocab_draft=True)

        if (
            self.last_settings_mutation_result is not None
            and self.last_settings_mutation_result.status
            == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
        ):
            degraded_result = next(
                (
                    result
                    for result in committed_results
                    if result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
                ),
                None,
            )
            if degraded_result is not None:
                self.last_settings_mutation_result = degraded_result
        return True

    async def _apply_stt_language_audio_settings_via_mutation_service(
        self,
        next_settings: AppSettings,
        *,
        reload_settings_view: bool = True,
    ) -> bool:
        base_and_patch = self._settings_projection().order22_patch_base_and_values(next_settings)
        if base_and_patch is None:
            return False
        base_settings, patch_values = base_and_patch
        if not patch_values:
            return False

        committed_settings = copy.deepcopy(base_settings)
        _apply_settings_path_patch(committed_settings, patch_values)
        has_out_of_scope_draft = self._get_settings_owner().legacy_snapshot_values(
            committed_settings
        ) != self._get_settings_owner().legacy_snapshot_values(next_settings)
        repository = self._legacy_settings_patch_repository(
            base_settings=base_settings,
            committed_settings=committed_settings,
            surface="stt_language_audio",
        )
        runtime_apply = (
            _ControllerNoopRuntimeApply()
            if has_out_of_scope_draft
            else _ControllerSttLanguageAudioRuntimeApply(
                controller=self,
                settings=committed_settings,
                reload_settings_view=reload_settings_view,
            )
        )
        command = SttLanguageAudioSettingsMutation(values=patch_values)
        service = self.settings_mutation_service or SettingsMutationService(
            settings_repository=repository,
            runtime_apply=runtime_apply,
            validator=settings_path_mutation_validator_for_command(command),
        )
        request = command.to_mutation_request(
            expected_revision=None,
            correlation_id=None,
        )

        result = await service.mutate(request)
        self._complete_canonical_mutation()
        self.last_settings_mutation_result = result
        if not _settings_mutation_committed(result):
            self.settings = copy.deepcopy(base_settings)
            self._settings_projection().remember_order22(self.settings)
            return True
        if id(committed_settings) in self._superseded_local_asr_settings_ids:
            self._superseded_local_asr_settings_ids.discard(id(committed_settings))
            self._settings_projection().remember_order22(self.settings)
            return True
        if (
            not has_out_of_scope_draft
            and result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
            and self._active_local_asr_change(base_settings, committed_settings)
        ):
            try:
                await self._compensate_failed_local_asr_settings_apply(
                    base_settings=base_settings,
                    committed_settings=committed_settings,
                )
            except Exception:
                self._log_error("Failed to compensate local ASR settings apply")
            self._settings_projection().remember_order22(self.settings)
            return True

        if has_out_of_scope_draft:
            try:
                await self._apply_settings_direct(
                    next_settings,
                    strict_runtime_errors=True,
                    strict_persistence_errors=True,
                    reload_settings_view=reload_settings_view,
                )
            except _StrictSettingsSaveFailed:
                await self._resync_committed_order22_settings_after_strict_save_failure(
                    base_settings=base_settings,
                    committed_settings=committed_settings,
                )
                self.last_settings_mutation_result = (
                    _stt_language_audio_save_failed_transaction_result(
                        operation="apply_stt_language_audio_full_draft_save"
                    )
                )
            except Exception:
                self.last_settings_mutation_result = (
                    _stt_language_audio_runtime_degraded_transaction_result()
                )
            else:
                unavailable_result = _stt_language_audio_runtime_unavailable_result(
                    controller=self,
                    settings=next_settings,
                )
                if unavailable_result is not None:
                    self.last_settings_mutation_result = (
                        _runtime_apply_result_as_degraded_transaction(unavailable_result)
                    )
        else:
            self.settings = committed_settings
            if result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED:
                self._sync_signature_caches(committed_settings)
        self._settings_projection().remember_order22(self.settings)
        return True

    async def _apply_overlay_osc_output_settings_via_mutation_service(
        self,
        next_settings: AppSettings,
    ) -> bool:
        base_and_patch = self._settings_projection().order23_patch_base_and_values(next_settings)
        if base_and_patch is None:
            return False
        base_settings, patch_values = base_and_patch
        if not patch_values:
            return False

        next_settings = copy.deepcopy(next_settings)
        await self._apply_desktop_size_preset_persistence_adjustment(
            base_settings,
            next_settings,
        )
        patch_values = build_overlay_osc_output_settings_path_patch(
            base_settings,
            next_settings,
        )
        committed_settings = copy.deepcopy(base_settings)
        _apply_settings_path_patch(committed_settings, patch_values)
        has_out_of_scope_draft = self._get_settings_owner().legacy_snapshot_values(
            committed_settings
        ) != self._get_settings_owner().legacy_snapshot_values(next_settings)
        repository = self._legacy_settings_patch_repository(
            base_settings=base_settings,
            committed_settings=committed_settings,
            surface="overlay_osc_output",
        )
        runtime_apply = (
            _ControllerNoopRuntimeApply()
            if has_out_of_scope_draft
            else _ControllerOverlayOscOutputRuntimeApply(
                controller=self,
                settings=committed_settings,
            )
        )
        command = OverlayOscOutputSettingsMutation(values=patch_values)
        service = self.settings_mutation_service or SettingsMutationService(
            settings_repository=repository,
            runtime_apply=runtime_apply,
            validator=settings_path_mutation_validator_for_command(command),
        )
        request = command.to_mutation_request(
            expected_revision=None,
            correlation_id=None,
        )

        result = await service.mutate(request)
        self._complete_canonical_mutation()
        self.last_settings_mutation_result = result
        if not _settings_mutation_committed(result):
            self.settings = copy.deepcopy(base_settings)
            self._settings_projection().remember_order23(self.settings)
            return True

        if has_out_of_scope_draft:
            try:
                await self._apply_settings_direct(
                    next_settings,
                    strict_runtime_errors=True,
                    strict_persistence_errors=True,
                )
            except _StrictSettingsSaveFailed:
                await self._resync_committed_order23_settings_after_strict_save_failure(
                    base_settings=base_settings,
                    committed_settings=committed_settings,
                )
                self.last_settings_mutation_result = (
                    _overlay_osc_output_save_failed_transaction_result(
                        operation="apply_overlay_osc_output_full_draft_save"
                    )
                )
            except Exception:
                self.last_settings_mutation_result = (
                    _overlay_osc_output_runtime_degraded_transaction_result()
                )
        else:
            if self.settings is None or self.settings is base_settings:
                self.settings = committed_settings
        self._settings_projection().remember_order23(self.settings)
        return True

    async def _mutate_order24_settings_patch(
        self,
        *,
        patch_values: Mapping[str, object],
        committed_settings: AppSettings,
        runtime_apply,
    ) -> TransactionResult:
        repository = self._legacy_settings_patch_repository(
            base_settings=self.settings or committed_settings,
            committed_settings=committed_settings,
            surface="ui_prompt_clipboard_state",
        )
        command = UiPromptClipboardStateSettingsMutation(values=patch_values)
        service = self.settings_mutation_service or SettingsMutationService(
            settings_repository=repository,
            runtime_apply=runtime_apply,
            validator=settings_path_mutation_validator_for_command(command),
        )
        request = command.to_mutation_request(
            expected_revision=None,
            correlation_id=None,
        )
        result = await service.mutate(request)
        self._complete_canonical_mutation()
        return result

    async def _apply_ui_prompt_clipboard_state_settings_via_mutation_service(
        self,
        next_settings: AppSettings,
    ) -> bool:
        base_and_patch = self._settings_projection().order24_patch_base_and_values(next_settings)
        if base_and_patch is None:
            return False
        base_settings, patch_values = base_and_patch
        if not patch_values:
            return False

        committed_settings = copy.deepcopy(base_settings)
        _apply_settings_path_patch(committed_settings, patch_values)
        has_out_of_scope_draft = self._get_settings_owner().legacy_snapshot_values(
            committed_settings
        ) != self._get_settings_owner().legacy_snapshot_values(next_settings)
        runtime_settings = copy.deepcopy(next_settings)
        runtime_apply = (
            _ControllerNoopRuntimeApply()
            if has_out_of_scope_draft
            else _ControllerUiPromptClipboardStateRuntimeApply(
                controller=self,
                settings=runtime_settings,
            )
        )

        result = await self._mutate_order24_settings_patch(
            patch_values=patch_values,
            committed_settings=committed_settings,
            runtime_apply=runtime_apply,
        )
        self.last_settings_mutation_result = result
        if not _settings_mutation_committed(result):
            self.settings = copy.deepcopy(base_settings)
            self._settings_projection().remember_order24(self.settings)
            return True

        if has_out_of_scope_draft:
            try:
                await self._apply_settings_direct(
                    next_settings,
                    strict_runtime_errors=True,
                    strict_persistence_errors=True,
                )
            except _StrictSettingsSaveFailed:
                await self._resync_committed_order24_settings_after_strict_save_failure(
                    base_settings=base_settings,
                    committed_settings=committed_settings,
                )
                self.last_settings_mutation_result = (
                    _ui_prompt_clipboard_state_save_failed_transaction_result(
                        operation="apply_ui_prompt_clipboard_state_full_draft_save"
                    )
                )
            except Exception:
                self.last_settings_mutation_result = (
                    _ui_prompt_clipboard_state_runtime_degraded_transaction_result()
                )
        else:
            self.settings = runtime_settings
            if result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED:
                self._sync_signature_caches(runtime_settings)
        self._settings_projection().remember_order24(self.settings)
        return True

    async def verify_api_key(self, provider: str, key: str) -> tuple[bool, str]:
        selected_model = None
        if self.settings is not None:
            if provider == "google":
                selected_model = self.settings.gemini.llm_model.value
            elif provider == "cerebras":
                selected_model = self.settings.cerebras.llm_model.value
            elif provider in {"alibaba_beijing", "alibaba_singapore"}:
                selected_model = self.settings.qwen.llm_model.value
        outcome = await self._get_provider_credential_verification_owner().verify(
            ProviderCredentialVerificationRequest(
                provider=provider,
                api_key=key,
                selected_model=selected_model,
                fallback_models=tuple(model.value for model in QwenLLMModel),
                low_latency=FIXED_TRANSLATION_POLICY.fast_translation_enabled,
            )
        )
        if outcome.status == PROVIDER_CREDENTIAL_VERIFIED:
            return True, "Verification successful"
        if outcome.status == PROVIDER_CREDENTIAL_EMPTY:
            return False, "API Key is empty"
        if outcome.status == PROVIDER_CREDENTIAL_UNKNOWN:
            return False, f"Unknown provider: {provider}"
        if outcome.status == PROVIDER_CREDENTIAL_MODEL_UNAVAILABLE:
            return False, f"qwen_model_unavailable:{outcome.unavailable_model}"
        if outcome.status == PROVIDER_CREDENTIAL_ERROR:
            error_text = outcome.error_text or ""
            self._log_error(f"Verification error for {provider}: {error_text}")
            return False, error_text
        return False, "Verification failed (check logs/console for details)"

    async def apply_providers(
        self,
        settings: AppSettings | None = None,
        *,
        force_rebuild_llm: bool = False,
    ) -> bool:
        next_settings = (
            self.settings
            if settings is None
            else self.merge_settings_tab_apply_with_current_languages(settings)
        )
        if next_settings is None:
            return False

        await self._preserve_github_star_prompt_observation_before_settings_replace(next_settings)

        try:
            if settings is not None and not force_rebuild_llm:
                routed = await self._apply_order21_order22_order24_provider_settings_via_mutation_services(
                    next_settings,
                )
                if routed:
                    return bool(
                        self.last_settings_mutation_result is not None
                        and _settings_mutation_committed(self.last_settings_mutation_result)
                    )
                routed = await self._apply_translation_provider_settings_via_mutation_service(
                    next_settings,
                )
                if routed:
                    return bool(
                        self.last_settings_mutation_result is not None
                        and _settings_mutation_committed(self.last_settings_mutation_result)
                    )
                routed = await self._apply_ui_prompt_clipboard_state_settings_via_mutation_service(
                    next_settings,
                )
                if routed:
                    return bool(
                        self.last_settings_mutation_result is not None
                        and _settings_mutation_committed(self.last_settings_mutation_result)
                    )

            return await self._apply_providers_direct(
                next_settings,
                force_rebuild_llm=force_rebuild_llm,
            )
        finally:
            self._sync_ui_from_settings()

    def _build_provider_runtime_apply_plan(
        self,
        next_settings: AppSettings,
        *,
        force_rebuild_llm: bool,
        canonical_settings: AppSettingsVNext | None = None,
    ) -> _ProviderRuntimeApplyPlan:
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
        next_peer_provider_signature = self._build_peer_stt_provider_signature(
            next_settings,
            canonical_settings=canonical_settings,
        )
        next_llm_provider_signature = self._build_llm_provider_signature(next_settings)

        return _ProviderRuntimeApplyPlan(
            should_rebuild_llm=force_rebuild_llm
            or (
                prev_llm_provider_signature is None
                or next_llm_provider_signature != prev_llm_provider_signature
            ),
            should_refresh_peer=(
                prev_peer_provider_signature is None
                or next_peer_provider_signature != prev_peer_provider_signature
            ),
            should_refresh_self_stt=(
                prev_self_provider_signature is None
                or next_self_provider_signature != prev_self_provider_signature
            ),
            coordinated_gpu_restart=(
                prev_settings is not None
                and prev_settings.stt.gpu_device_id != next_settings.stt.gpu_device_id
                and (
                    prev_settings.provider.stt == STTProviderName.LOCAL_QWEN_GPU
                    or prev_settings.provider.peer_stt == STTProviderName.LOCAL_QWEN_GPU
                    or next_settings.provider.stt == STTProviderName.LOCAL_QWEN_GPU
                    or next_settings.provider.peer_stt == STTProviderName.LOCAL_QWEN_GPU
                )
            ),
        )

    async def _apply_order21_order22_order24_provider_settings_via_mutation_services(
        self,
        next_settings: AppSettings,
    ) -> bool:
        base_settings = self.settings
        if base_settings is None:
            return False

        order21_patch_values = build_translation_provider_settings_path_patch(
            base_settings,
            next_settings,
        )
        order22_patch_values = build_stt_language_audio_settings_path_patch(
            base_settings,
            next_settings,
        )
        order24_base_and_patch = self._settings_projection().order24_patch_base_and_values(
            next_settings
        )
        if order24_base_and_patch is None:
            return False
        _order24_base_settings, order24_patch_values = order24_base_and_patch
        patch_count = sum(
            1
            for patch_values in (
                order21_patch_values,
                order22_patch_values,
                order24_patch_values,
            )
            if patch_values
        )
        if patch_count < 2:
            return False

        committed_results: list[TransactionResult] = []

        async def _route_provider_patch_only(patch_values: dict[str, object], route) -> bool:
            if self.settings is None:
                return False
            patch_only_settings = copy.deepcopy(self.settings)
            _apply_settings_path_patch(patch_only_settings, patch_values)
            routed = await route(patch_only_settings)
            if not routed:
                return False
            result = self.last_settings_mutation_result
            if result is None or not _settings_mutation_committed(result):
                return True
            committed_results.append(result)
            return True

        if order21_patch_values:
            routed_order21 = await _route_provider_patch_only(
                order21_patch_values,
                self._apply_translation_provider_settings_via_mutation_service,
            )
            if not routed_order21:
                return False
            if self.last_settings_mutation_result is None or not _settings_mutation_committed(
                self.last_settings_mutation_result,
            ):
                return True

        if order22_patch_values:
            routed_order22 = await _route_provider_patch_only(
                order22_patch_values,
                self._apply_stt_language_audio_provider_settings_via_mutation_service,
            )
            if not routed_order22:
                return False
            if self.last_settings_mutation_result is None or not _settings_mutation_committed(
                self.last_settings_mutation_result,
            ):
                return True

        if order24_patch_values:
            if self.settings is None:
                return True
            order24_only_settings = copy.deepcopy(self.settings)
            _apply_settings_path_patch(order24_only_settings, order24_patch_values)
            _copy_runtime_only_ui_state(next_settings, order24_only_settings)
            routed_order24 = (
                await self._apply_ui_prompt_clipboard_state_settings_via_mutation_service(
                    order24_only_settings,
                )
            )
            if not routed_order24:
                return False
            if self.last_settings_mutation_result is None or not _settings_mutation_committed(
                self.last_settings_mutation_result,
            ):
                return True

        committed_settings_before_full_draft = (
            copy.deepcopy(self.settings) if self.settings is not None else None
        )
        if self.settings is not None and self._get_settings_owner().legacy_snapshot_values(
            self.settings
        ) != self._get_settings_owner().legacy_snapshot_values(next_settings):
            try:
                await self._apply_providers_direct(
                    next_settings,
                    force_rebuild_llm=False,
                    route_order22=False,
                    strict_persistence_errors=True,
                )
            except _StrictSettingsSaveFailed:
                if committed_settings_before_full_draft is not None:
                    self._sync_memory_runtime_fields_from_settings(
                        committed_settings_before_full_draft
                    )
                self.last_settings_mutation_result = (
                    _translation_provider_save_failed_transaction_result(
                        operation="apply_order21_order22_order24_provider_full_draft_save"
                    )
                )
            except Exception:
                self.last_settings_mutation_result = _runtime_apply_result_as_degraded_transaction(
                    _runtime_apply_failed_result(
                        operation="apply_order21_order22_order24_provider_runtime",
                        code="provider_runtime_apply_exception",
                        surface="translation_provider",
                    )
                )

        if (
            self.last_settings_mutation_result is not None
            and self.last_settings_mutation_result.status
            == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
            and committed_results
        ):
            degraded_result = next(
                (
                    result
                    for result in committed_results
                    if result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
                ),
                None,
            )
            if degraded_result is not None:
                self.last_settings_mutation_result = degraded_result
        return True

    async def _apply_translation_provider_settings_via_mutation_service(
        self,
        next_settings: AppSettings,
    ) -> bool:
        base_settings = self.settings
        if base_settings is None:
            return False

        patch_values = build_translation_provider_settings_path_patch(
            base_settings,
            next_settings,
        )
        if not patch_values:
            return False

        committed_settings = copy.deepcopy(base_settings)
        _apply_settings_path_patch(committed_settings, patch_values)
        has_out_of_scope_draft = self._get_settings_owner().legacy_snapshot_values(
            committed_settings
        ) != self._get_settings_owner().legacy_snapshot_values(next_settings)
        plan = self._build_provider_runtime_apply_plan(
            committed_settings,
            force_rebuild_llm=False,
            canonical_settings=self._canonical_vnext_after_legacy_delta(
                base_settings,
                committed_settings,
            ),
        )
        repository = self._legacy_settings_patch_repository(
            base_settings=base_settings,
            committed_settings=committed_settings,
        )
        runtime_apply = _ControllerProviderRuntimeApply(
            controller=self,
            settings=committed_settings,
            plan=plan,
        )
        command = TranslationProviderSettingsMutation(values=patch_values)
        service = self.settings_mutation_service or SettingsMutationService(
            settings_repository=repository,
            runtime_apply=runtime_apply,
            validator=settings_path_mutation_validator_for_command(command),
        )
        request = command.to_mutation_request(
            expected_revision=None,
            correlation_id=None,
        )

        result = await service.mutate(request)
        self._complete_canonical_mutation()
        self.last_settings_mutation_result = result
        if not _settings_mutation_committed(result):
            return True

        self.settings = committed_settings
        if has_out_of_scope_draft:
            fallback_plan = self._build_provider_runtime_apply_plan(
                next_settings,
                force_rebuild_llm=False,
                canonical_settings=self._canonical_vnext_after_legacy_delta(
                    committed_settings,
                    next_settings,
                ),
            )
            try:
                await self._apply_providers_direct(
                    next_settings,
                    force_rebuild_llm=False,
                    plan=fallback_plan,
                    route_order22=False,
                    strict_persistence_errors=True,
                )
            except _StrictSettingsSaveFailed:
                preserve_llm_provider_retry_marker = self._last_llm_provider_signature == ()
                self._sync_memory_runtime_fields_from_settings(committed_settings)
                if preserve_llm_provider_retry_marker:
                    self._last_llm_provider_signature = ()
                self.last_settings_mutation_result = (
                    _translation_provider_save_failed_transaction_result(
                        operation="apply_translation_provider_full_draft_save"
                    )
                )
            except Exception:
                self.last_settings_mutation_result = _runtime_apply_result_as_degraded_transaction(
                    _runtime_apply_failed_result(
                        operation="apply_translation_provider_runtime",
                        code="provider_runtime_apply_exception",
                        surface="translation_provider",
                    )
                )
            else:
                unavailable_result = _provider_runtime_apply_unavailable_result(
                    controller=self,
                    settings=next_settings,
                    plan=fallback_plan,
                    operation="apply_translation_provider_runtime",
                    surface="translation_provider",
                )
                if unavailable_result is not None:
                    self.last_settings_mutation_result = (
                        _runtime_apply_result_as_degraded_transaction(unavailable_result)
                    )
        elif result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED:
            self._sync_signature_caches(committed_settings)
        self._settings_projection().remember_order22(self.settings)
        return True

    async def _apply_stt_language_audio_provider_settings_via_mutation_service(
        self,
        next_settings: AppSettings,
    ) -> bool:
        base_settings = self.settings
        if base_settings is None:
            return False

        patch_values = build_stt_language_audio_settings_path_patch(
            base_settings,
            next_settings,
        )
        if not patch_values:
            return False

        committed_settings = copy.deepcopy(base_settings)
        _apply_settings_path_patch(committed_settings, patch_values)
        has_out_of_scope_draft = self._get_settings_owner().legacy_snapshot_values(
            committed_settings
        ) != self._get_settings_owner().legacy_snapshot_values(next_settings)
        plan = self._build_provider_runtime_apply_plan(
            committed_settings,
            force_rebuild_llm=False,
            canonical_settings=self._canonical_vnext_after_legacy_delta(
                base_settings,
                committed_settings,
            ),
        )
        repository = self._legacy_settings_patch_repository(
            base_settings=base_settings,
            committed_settings=committed_settings,
            surface="stt_language_audio",
        )
        runtime_apply = (
            _ControllerNoopRuntimeApply()
            if has_out_of_scope_draft
            else _ControllerProviderRuntimeApply(
                controller=self,
                settings=committed_settings,
                plan=plan,
                surface="stt_language_audio",
                operation="apply_stt_language_audio_provider_runtime",
            )
        )
        command = SttLanguageAudioSettingsMutation(values=patch_values)
        service = self.settings_mutation_service or SettingsMutationService(
            settings_repository=repository,
            runtime_apply=runtime_apply,
            validator=settings_path_mutation_validator_for_command(command),
        )
        request = command.to_mutation_request(
            expected_revision=None,
            correlation_id=None,
        )

        result = await service.mutate(request)
        self._complete_canonical_mutation()
        self.last_settings_mutation_result = result
        if not _settings_mutation_committed(result):
            self.settings = copy.deepcopy(base_settings)
            self._settings_projection().remember_order22(self.settings)
            return True
        if id(committed_settings) in self._superseded_local_asr_settings_ids:
            self._superseded_local_asr_settings_ids.discard(id(committed_settings))
            self._settings_projection().remember_order22(self.settings)
            return True
        if (
            not has_out_of_scope_draft
            and result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
            and self._active_local_asr_change(base_settings, committed_settings)
        ):
            try:
                await self._compensate_failed_local_asr_settings_apply(
                    base_settings=base_settings,
                    committed_settings=committed_settings,
                )
            except Exception:
                self._log_error("Failed to compensate local ASR provider settings apply")
            self._settings_projection().remember_order22(self.settings)
            return True

        if has_out_of_scope_draft:
            fallback_plan = self._build_provider_runtime_apply_plan(
                next_settings,
                force_rebuild_llm=False,
                canonical_settings=self._canonical_vnext_after_legacy_delta(
                    committed_settings,
                    next_settings,
                ),
            )
            try:
                await self._apply_providers_direct(
                    next_settings,
                    force_rebuild_llm=False,
                    plan=fallback_plan,
                    route_order22=False,
                    strict_persistence_errors=True,
                )
            except _StrictSettingsSaveFailed:
                await self._resync_committed_order22_provider_runtime_after_strict_save_failure(
                    base_settings=base_settings,
                    committed_settings=committed_settings,
                    plan=plan,
                )
                self.last_settings_mutation_result = (
                    _stt_language_audio_save_failed_transaction_result(
                        operation="apply_stt_language_audio_provider_full_draft_save"
                    )
                )
            except Exception:
                self.last_settings_mutation_result = _runtime_apply_result_as_degraded_transaction(
                    _runtime_apply_failed_result(
                        operation="apply_stt_language_audio_provider_runtime",
                        code="provider_runtime_apply_exception",
                        surface="stt_language_audio",
                    )
                )
            else:
                unavailable_result = _provider_runtime_apply_unavailable_result(
                    controller=self,
                    settings=next_settings,
                    plan=fallback_plan,
                    operation="apply_stt_language_audio_provider_runtime",
                    surface="stt_language_audio",
                )
                if unavailable_result is not None:
                    self.last_settings_mutation_result = (
                        _runtime_apply_result_as_degraded_transaction(unavailable_result)
                    )
        else:
            self.settings = committed_settings
            if result.status == TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED:
                self._sync_signature_caches(committed_settings)
        self._settings_projection().remember_order22(self.settings)
        return True

    async def _apply_providers_direct(
        self,
        next_settings: AppSettings,
        *,
        force_rebuild_llm: bool,
        plan: _ProviderRuntimeApplyPlan | None = None,
        route_order22: bool = True,
        strict_persistence_errors: bool = False,
    ) -> bool:
        if route_order22 and not force_rebuild_llm and plan is None:
            routed = await self._apply_stt_language_audio_provider_settings_via_mutation_service(
                next_settings,
            )
            if routed:
                return bool(
                    self.last_settings_mutation_result is not None
                    and _settings_mutation_committed(self.last_settings_mutation_result)
                )
        self._begin_canonical_mutation(
            legacy_snapshot=self._canonical_legacy_projection_snapshot or self.settings
        )
        self._capture_runtime_signatures_before_canonical_mutation()
        self._update_canonical_settings_from_legacy_delta(
            self._canonical_legacy_projection_snapshot or self.settings,
            next_settings,
        )
        if plan is None:
            plan = self._build_provider_runtime_apply_plan(
                next_settings,
                force_rebuild_llm=force_rebuild_llm,
            )
        self.settings = next_settings
        if strict_persistence_errors:
            try:
                self._persist_settings_at_controller_boundary(self.settings)
            except Exception:
                self._rollback_canonical_mutation()
                raise _StrictSettingsSaveFailed from None
            else:
                self._remember_canonical_legacy_projection(self.settings)
        else:
            if self._save_settings() is False:
                return False
        await self._apply_provider_runtime_plan(next_settings, plan)
        self._settings_projection().remember_order22(self.settings)
        self._complete_canonical_mutation()
        return True

    async def _apply_provider_runtime_plan(
        self,
        next_settings: AppSettings,
        plan: _ProviderRuntimeApplyPlan,
    ) -> None:
        self.settings = next_settings

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
            self.hub.low_latency_mode = FIXED_TRANSLATION_POLICY.fast_translation_enabled
            self.hub.low_latency_merge_gap_ms = next_settings.stt.low_latency_merge_gap_ms
            self.hub.low_latency_spec_retry_max = next_settings.stt.low_latency_spec_retry_max
            self.hub.hangover_s = (
                next_settings.stt.low_latency_vad_hangover_ms / 1000.0
                if FIXED_TRANSLATION_POLICY.fast_translation_enabled
                else DEFAULT_STABLE_VAD_HANGOVER_MS / 1000.0
            )
            self.hub.peer_hangover_s = next_settings.desktop_audio.vad_hangover_ms / 1000.0
            self.hub.chatbox_include_source = next_settings.osc.chatbox_include_source
            self._sync_effective_hub_flags(next_settings)

        if plan.should_rebuild_llm:
            await self._rebuild_llm_provider()

        if plan.coordinated_gpu_restart:
            await self._apply_gpu_runtime_owner_recovery(next_settings, plan)
            self._sync_signature_caches(next_settings)
            if plan.should_rebuild_llm and self.hub is not None and self.hub.llm is None:
                self._last_llm_provider_signature = ()
            return

        if plan.should_refresh_peer:
            await self._refresh_peer_stt_runtime()
            self._sync_effective_hub_flags(next_settings)
            self._refresh_overlay_peer_consumers()

        if plan.should_refresh_self_stt:
            if self._stt_desired:
                await self._apply_stt_runtime_replacement(smooth_local=True)
            else:
                await self._rebuild_stt_provider()

        self._sync_signature_caches(next_settings)
        if plan.should_rebuild_llm and self.hub is not None and self.hub.llm is None:
            self._last_llm_provider_signature = ()

    async def _apply_gpu_runtime_owner_recovery(
        self,
        next_settings: AppSettings,
        plan: _ProviderRuntimeApplyPlan,
    ) -> None:
        async with self._get_gpu_provider_recovery_lock():
            await self._apply_gpu_runtime_owner_recovery_locked(next_settings, plan)

    async def _apply_gpu_runtime_owner_recovery_locked(
        self,
        next_settings: AppSettings,
        plan: _ProviderRuntimeApplyPlan,
    ) -> None:
        owned_runtime = self._hub_local_asr_provider_runtime()
        if owned_runtime is None:
            raise RuntimeError("local ASR provider runtime is unavailable")
        desired_channels = self._desired_gpu_channels(next_settings)
        recovery, peer_config = self._build_gpu_recovery_request(
            next_settings,
            desired_channels,
            reason="settings_restart",
        )
        try:
            snapshot = await owned_runtime.recover_gpu(
                recovery,
                quiesce=self._suspend_gpu_provider_consumers,
            )
            recovered_channels = frozenset(item.request.channel for item in recovery.channels)
            if recovered_channels and (
                snapshot.gpu.retry_required
                or not recovered_channels.issubset(snapshot.gpu.active_channels)
            ):
                self._on_local_asr_provider_runtime_state_changed(snapshot)
                return
            await self._resume_gpu_provider_consumers(
                settings=next_settings,
                channels=recovered_channels,
                peer_config=peer_config,
                recovery=recovery,
            )
            if plan.should_refresh_self_stt and "self" not in recovered_channels:
                if self._stt_desired:
                    await self._replace_runtime_stt_provider()
                else:
                    await self._rebuild_stt_provider()
            if plan.should_refresh_peer and "peer" not in recovered_channels:
                await self._refresh_peer_stt_runtime()
                self._sync_effective_hub_flags(next_settings)
                self._refresh_overlay_peer_consumers()
        except Exception:
            self._set_gpu_ui_state(
                "activation_failed",
                publish_notice=True,
                origin="settings_apply",
            )
        finally:
            self._abort_provider_recoveries(recovery)

    def _get_gpu_provider_recovery_lock(self) -> asyncio.Lock:
        if self._gpu_provider_recovery_lock is None:
            self._gpu_provider_recovery_lock = asyncio.Lock()
        return self._gpu_provider_recovery_lock

    def _desired_gpu_channels(self, settings: AppSettings) -> frozenset[GpuASRChannel]:
        owned_runtime = self._hub_local_asr_provider_runtime()
        active_channels = (
            owned_runtime.snapshot.gpu.active_channels if owned_runtime is not None else frozenset()
        )
        desired: set[GpuASRChannel] = set()
        if settings.provider.stt == STTProviderName.LOCAL_QWEN_GPU and (
            self._stt_desired or "self" in active_channels
        ):
            desired.add("self")
        if settings.provider.peer_stt == STTProviderName.LOCAL_QWEN_GPU and (
            settings.ui.peer_translation_enabled or "peer" in active_channels
        ):
            desired.add("peer")
        return frozenset(desired)

    def _build_gpu_recovery_request(
        self,
        settings: AppSettings,
        channels: frozenset[GpuASRChannel],
        *,
        reason: Literal["manual_retry", "settings_restart"],
    ) -> tuple[ProviderRuntimeGpuRecoveryRequest, PeerCaptureSessionConfig | None]:
        targets: list[ProviderRuntimeRecoveryChannel] = []
        peer_config = None
        self_recovery_owner = None
        self_recovery_handler = None
        peer_recovery_owner = None
        peer_recovery_handler = None
        try:
            if "self" in channels and settings.provider.stt == STTProviderName.LOCAL_QWEN_GPU:
                self_config = self._build_self_capture_session_config(settings)
                self_recovery_owner = self._get_self_capture_owner()
                self_recovery_handler = self_recovery_owner.prepare_provider_recovery(self_config)
                targets.append(
                    ProviderRuntimeRecoveryChannel(
                        request=self._self_stt_provider_request(settings, warmup=True),
                        start=self._stt_desired,
                        on_terminal_failure=self_recovery_handler,
                    )
                )
            if "peer" in channels and settings.provider.peer_stt == STTProviderName.LOCAL_QWEN_GPU:
                peer_config = self._build_peer_runtime_config(settings)
                peer_recovery_owner = self._peer_runtime
                if peer_recovery_owner is None:
                    raise RuntimeError("Peer recovery requires a capture owner")
                peer_recovery_handler = peer_recovery_owner.prepare_provider_recovery(peer_config)
                targets.append(
                    ProviderRuntimeRecoveryChannel(
                        request=self._peer_stt_provider_request(peer_config, warmup=True),
                        start=False,
                        on_terminal_failure=peer_recovery_handler,
                    )
                )
        except BaseException:
            if self_recovery_owner is not None and self_recovery_handler is not None:
                self_recovery_owner.abort_provider_recovery(self_recovery_handler)
            if peer_recovery_owner is not None and peer_recovery_handler is not None:
                peer_recovery_owner.abort_provider_recovery(peer_recovery_handler)
            raise
        return (
            ProviderRuntimeGpuRecoveryRequest(
                device_id=settings.stt.gpu_device_id,
                channels=tuple(targets),
                reason=reason,
            ),
            peer_config,
        )

    def _abort_provider_recoveries(
        self,
        recovery: ProviderRuntimeGpuRecoveryRequest,
    ) -> None:
        for channel, owner in (
            ("self", self._self_capture_owner),
            ("peer", self._peer_runtime),
        ):
            if owner is None:
                continue
            target = next(
                (item for item in recovery.channels if item.request.channel == channel),
                None,
            )
            if target is not None and target.on_terminal_failure is not None:
                owner.abort_provider_recovery(target.on_terminal_failure)

    async def _suspend_gpu_provider_consumers(
        self,
        channels: tuple[GpuASRChannel, ...],
    ) -> None:
        if "self" in channels and self._self_capture_owner is not None:
            snapshot = await self._self_capture_owner.suspend_provider_consumer()
            self._on_self_capture_state_changed(snapshot)
        if "peer" in channels and self._peer_runtime is not None:
            await self._peer_runtime.suspend_provider_consumer()

    async def _resume_gpu_provider_consumers(
        self,
        *,
        settings: AppSettings,
        channels: frozenset[GpuASRChannel],
        peer_config: PeerCaptureSessionConfig | None,
        recovery: ProviderRuntimeGpuRecoveryRequest,
    ) -> None:
        if "peer" in channels and self._peer_runtime is not None:
            if peer_config is None:
                peer_config = self._build_peer_runtime_config(settings)
            peer_target = next(item for item in recovery.channels if item.request.channel == "peer")
            if peer_target.on_terminal_failure is None:
                raise RuntimeError("Peer recovery requires an owner failure callback")
            await self._peer_runtime.adopt_recovered_provider(
                peer_config,
                on_terminal_failure=peer_target.on_terminal_failure,
            )
            await self._refresh_peer_stt_runtime()
            self._sync_effective_hub_flags(settings)
            self._refresh_overlay_peer_consumers()
        if "self" in channels:
            if self._self_capture_owner is not None:
                self_target = next(
                    item for item in recovery.channels if item.request.channel == "self"
                )
                if self_target.on_terminal_failure is None:
                    raise RuntimeError("Self recovery requires an owner failure callback")
                snapshot = await self._self_capture_owner.adopt_recovered_provider(
                    self._build_self_capture_session_config(settings),
                    on_terminal_failure=self_target.on_terminal_failure,
                )
                self._on_self_capture_state_changed(snapshot)
            if self._stt_desired:
                await self._ensure_stt_switch()

    def _load_or_init_settings(self, path: Path) -> AppSettings:
        if path != self.config_path:
            raise ValueError("settings owner path does not match controller config path")
        result = self._get_settings_owner().start(
            allow_stable_settings_import=self.allow_stable_settings_import
        )
        if result.stable_source_path is not None and result.imported_settings is not None:
            with contextlib.suppress(Exception):
                copy_stable_secrets_to_vnext_namespace(
                    (result.stable_source_settings or result.imported_settings).intent.secrets,
                    stable_config_path=result.stable_source_path,
                    vnext_config_path=path,
                    vnext_settings=result.imported_settings.intent.secrets,
                )
        return result.settings

    def _persist_settings_at_controller_boundary(self, settings: AppSettings) -> None:
        _ = settings
        self._get_settings_owner().persist()

    def _canonical_vnext_settings_for(self, settings: AppSettings) -> AppSettingsVNext:
        projected = self._get_settings_owner().project(
            settings,
            authoritative=self._vnext_settings_authoritative,
        )
        return projected

    def _update_canonical_settings_from_legacy_delta(
        self,
        base_settings: AppSettings | None,
        next_settings: AppSettings,
    ) -> AppSettingsVNext:
        self.vnext_settings = self._canonical_vnext_after_legacy_delta(
            base_settings,
            next_settings,
        )
        self._vnext_settings_authoritative = True
        return self.vnext_settings

    def _canonical_vnext_after_legacy_delta(
        self,
        base_settings: AppSettings | None,
        next_settings: AppSettings,
    ) -> AppSettingsVNext:
        return self._get_settings_owner().project_legacy_delta(base_settings, next_settings)

    def _provider_verification_binding(
        self,
        provider: str,
        key: str,
        *,
        flow: str,
        context_values: Mapping[str, object] | None = None,
    ) -> ProviderVerificationBinding:
        secret_key = _PROVIDER_VERIFICATION_SECRET_KEY_BY_PROVIDER.get(provider)
        if secret_key is None:
            raise ValueError(f"unsupported provider verification binding: {provider}")
        context: dict[str, object] = {"flow": flow}
        settings = self.settings
        if settings is not None:
            if provider == "google":
                context["model"] = settings.gemini.llm_model.value
            elif provider == "cerebras":
                context["model"] = settings.cerebras.llm_model.value
            elif provider in {"alibaba_beijing", "alibaba_singapore"}:
                context["base_url"] = (
                    "https://dashscope.aliyuncs.com/api/v1"
                    if provider == "alibaba_beijing"
                    else "https://dashscope-intl.aliyuncs.com/api/v1"
                )
                context["model"] = settings.qwen.llm_model.value
                context["low_latency"] = FIXED_TRANSLATION_POLICY.fast_translation_enabled
        if context_values is not None:
            context.update(context_values)
        return ProviderVerificationBinding(
            provider=provider,
            secret_key=secret_key,
            secret_revision=None,
            secret_fingerprint=_provider_verification_fingerprint(key),
            verifier_context=context,
            verifier_evidence={"source": "provider_verifier"},
        )

    def _bind_provider_verification(
        self,
        binding: ProviderVerificationBinding,
    ) -> None:
        self._get_settings_owner().bind_provider_verification(binding)

    async def persist_provider_secret_change(
        self,
        secret_key: str,
        value: str,
    ) -> bool:
        return await self._get_provider_secret_change_owner().change(
            lambda: self._provider_secret_change_execution(secret_key, value)
        )

    def _get_provider_secret_change_owner(
        self,
    ) -> ProviderSecretChangeOwner:
        owner = self._provider_secret_change_owner
        if owner is None:
            owner = ProviderSecretChangeOwner()
            self._provider_secret_change_owner = owner
        return owner

    def _provider_secret_change_execution(
        self,
        secret_key: str,
        value: str,
    ) -> ProviderSecretChangeExecution:
        assert self.settings is not None
        provider = _PROVIDER_BY_VERIFICATION_SECRET_KEY.get(secret_key)
        if provider is None:
            raise ValueError(f"unsupported provider secret key: {secret_key}")
        updated = copy.deepcopy(self.settings)
        setattr(updated.api_key_verified, provider, False)
        secret_store = create_secret_store(
            self.settings.secrets,
            config_path=self.config_path,
        )
        repository = self._legacy_settings_patch_repository(
            base_settings=self.settings,
            committed_settings=updated,
            surface="provider_secret_change",
        )
        transaction = SecretSettingsTransaction(
            secret_store=create_sync_secret_store_adapter(secret_store),
            settings_repository=repository,
        )
        return ProviderSecretChangeExecution(
            transaction=transaction,
            request=ProviderSecretChangeRequest(
                provider=provider,
                secret_key=secret_key,
                secret_value=value,
                settings_values=self._get_settings_owner().legacy_snapshot_values(updated),
            ),
            result_handler=lambda result, succeeded: self._apply_provider_secret_change_result(
                repository,
                result,
                succeeded,
            ),
        )

    def _apply_provider_secret_change_result(
        self,
        repository: CommittedSettingsRepositoryPort[AppSettings],
        result: TransactionResult,
        succeeded: bool,
    ) -> None:
        self.last_settings_mutation_result = result
        if succeeded:
            self.settings = repository.committed_settings
            self._remember_canonical_legacy_projection(self.settings)
            self._complete_canonical_mutation()

    def persist_api_key_verification(
        self,
        provider: str,
        key: str,
        success: bool,
    ) -> None:
        assert self.settings is not None
        baseline = copy.deepcopy(self._canonical_legacy_projection_snapshot or self.settings)
        self._begin_canonical_mutation(legacy_snapshot=baseline)
        setattr(self.settings.api_key_verified, provider, success)
        try:
            self._update_canonical_settings_from_legacy_delta(
                baseline,
                self.settings,
            )
            if success:
                binding = self._provider_verification_binding(
                    provider,
                    key,
                    flow="settings_api_key_verification",
                )
                secret_store = create_secret_store(
                    self.settings.secrets,
                    config_path=self.config_path,
                )
                if secret_store.get(binding.secret_key) != key:
                    raise RuntimeError(
                        "verified credential does not match the active SecretStore value"
                    )
                self._bind_provider_verification(binding)
            self._persist_settings_at_controller_boundary(self.settings)
        except Exception:
            self._rollback_canonical_mutation()
            raise
        self._remember_canonical_legacy_projection(self.settings)
        self._complete_canonical_mutation()

    def clear_provider_verification(self, provider: str) -> None:
        self.persist_api_key_verification(provider, "", False)

    def _update_canonical_settings_from_compatibility_mutation(
        self,
        settings: AppSettings,
    ) -> AppSettingsVNext:
        return self._update_canonical_settings_from_legacy_delta(self.settings, settings)

    def _remember_canonical_legacy_projection(self, settings: AppSettings) -> None:
        self._canonical_legacy_projection_snapshot = copy.deepcopy(settings)

    def _managed_identity_persistence_callback(
        self,
        bound_settings: AppSettings,
    ) -> Callable[[AppSettings], None]:
        bound_snapshot = copy.deepcopy(bound_settings.managed_identity)

        def persist(settings: AppSettings) -> None:
            nonlocal bound_snapshot
            self._persist_active_controller_settings(
                settings,
                bound_managed_snapshot=bound_snapshot,
            )
            bound_snapshot = copy.deepcopy(settings.managed_identity)

        return persist

    def _persist_active_controller_settings(
        self,
        settings: AppSettings,
        *,
        bound_managed_snapshot: object | None = None,
    ) -> None:
        active_settings = self.settings or settings
        baseline = self._canonical_legacy_projection_snapshot or active_settings
        managed_baseline = (
            bound_managed_snapshot
            if bound_managed_snapshot is not None
            else baseline.managed_identity
        )
        managed_delta = _managed_identity_delta(managed_baseline, settings.managed_identity)
        next_settings = copy.deepcopy(active_settings)
        _apply_managed_identity_delta(next_settings, managed_delta)
        self._begin_canonical_mutation(legacy_snapshot=baseline)
        self._update_canonical_settings_from_legacy_delta(baseline, next_settings)
        try:
            self._persist_settings_at_controller_boundary(next_settings)
        except Exception:
            self._rollback_canonical_mutation()
            _restore_managed_identity(settings, managed_baseline)
            raise
        self.settings = next_settings
        self._remember_canonical_legacy_projection(next_settings)
        self._complete_canonical_mutation()

    def _begin_canonical_mutation(
        self,
        *,
        legacy_snapshot: AppSettings | None = None,
    ) -> None:
        if self._canonical_mutation_depth == 0:
            self._canonical_mutation_rollback_active_settings = self.settings
            self._canonical_mutation_rollback_legacy_snapshot = copy.deepcopy(
                legacy_snapshot if legacy_snapshot is not None else self.settings
            )
            self._canonical_mutation_rollback_authoritative = self._vnext_settings_authoritative
            self._canonical_mutation_rollback_pending = True
        self._get_settings_owner().begin()
        self._canonical_mutation_depth += 1

    def _rollback_canonical_mutation(self) -> None:
        if not self._canonical_mutation_rollback_pending:
            return
        self._get_settings_owner().rollback()
        active_settings = self._canonical_mutation_rollback_active_settings
        legacy_snapshot = self._canonical_mutation_rollback_legacy_snapshot
        if active_settings is not None and legacy_snapshot is not None:
            for settings_field in fields(AppSettings):
                setattr(
                    active_settings,
                    settings_field.name,
                    copy.deepcopy(getattr(legacy_snapshot, settings_field.name)),
                )
            self.settings = active_settings
        else:
            self.settings = legacy_snapshot
        self._vnext_settings_authoritative = self._canonical_mutation_rollback_authoritative
        self._canonical_mutation_depth = 1
        self._complete_canonical_mutation()

    def _complete_canonical_mutation(self) -> None:
        if self._canonical_mutation_depth == 0:
            return
        self._canonical_mutation_depth -= 1
        self._get_settings_owner().complete()
        if self._canonical_mutation_depth:
            return
        self._canonical_mutation_rollback_legacy_snapshot = None
        self._canonical_mutation_rollback_active_settings = None
        self._canonical_mutation_rollback_authoritative = False
        self._canonical_mutation_rollback_pending = False

    def _capture_runtime_signatures_before_canonical_mutation(self) -> None:
        if self.settings is None:
            return
        if self._last_peer_stt_provider_signature is None:
            self._last_peer_stt_provider_signature = self._build_peer_stt_provider_signature(
                self.settings
            )
        if self._last_peer_stt_runtime_signature is None:
            self._last_peer_stt_runtime_signature = self._build_peer_stt_runtime_signature(
                self.settings
            )

    def _get_provider_verifier(self) -> ProviderVerifierPort:
        if self.provider_verifier is None:
            self.provider_verifier = create_provider_verifier()
        return self.provider_verifier

    def _get_provider_credential_verification_owner(
        self,
    ) -> ProviderCredentialVerificationOwner:
        owner = self._provider_credential_verification_owner
        if owner is None:
            owner = ProviderCredentialVerificationOwner(
                verifier=self._get_provider_verifier(),
                diagnostics_sink=lambda event, metadata, exception: self.log_detailed(
                    "[ProviderVerification] Credential verification failed "
                    f"event={event} provider={metadata.get('provider')} "
                    f"error_type={metadata.get('error_type')}",
                    level=logging.WARNING,
                    exception=exception,
                ),
            )
            self._provider_credential_verification_owner = owner
        return owner

    def _get_provider_status_verification_owner(self) -> ProviderStatusVerificationOwner:
        owner = self._provider_status_verification_owner
        if owner is None:
            owner = ProviderStatusVerificationOwner(
                verifier=self._get_provider_verifier(),
                diagnostics_sink=lambda event, metadata, exception: self.log_detailed(
                    "[ProviderVerification] Background status verification failed "
                    f"event={event} error_type={metadata.get('error_type')}",
                    level=logging.WARNING,
                    exception=exception,
                ),
            )
            self._provider_status_verification_owner = owner
        return owner

    def _schedule_provider_status_verification(self) -> None:
        if self._shutdown_ingress_frozen:
            return
        self._get_provider_status_verification_owner().schedule(
            request_factory=self._build_provider_status_verification_request,
            result_handler=self._apply_provider_status_verification_result,
        )

    def _build_provider_status_verification_request(
        self,
    ) -> ConfiguredProviderStatusVerificationRequest | None:
        settings = self.settings
        if settings is None:
            return None

        secrets = None
        with contextlib.suppress(Exception):
            secrets = create_secret_store(settings.secrets, config_path=self.config_path)

        def secret_value(secret_key: str) -> str:
            if secrets is None:
                return ""
            try:
                return secrets.get(secret_key) or ""
            except Exception:
                return ""

        qwen_api_key = ""
        qwen_base_url = settings.qwen.get_llm_base_url()
        if secrets is not None:
            with contextlib.suppress(Exception):
                qwen_api_key, qwen_base_url = self._get_qwen_key_and_base_url(secrets)

        openrouter_api_key = ""
        if secrets is not None:
            with contextlib.suppress(Exception):
                resolution = resolve_openrouter_credentials(
                    build_openrouter_credential_runtime_config(settings),
                    secrets=secrets,
                )
                openrouter_api_key = resolution.api_key or ""

        managed_openrouter_can_attempt = False
        with contextlib.suppress(Exception):
            managed_openrouter_can_attempt = self._managed_openrouter_can_attempt_translation()

        llm_provider = getattr(settings.provider.llm, "value", settings.provider.llm)
        stt_provider = getattr(settings.provider.stt, "value", settings.provider.stt)
        return ConfiguredProviderStatusVerificationRequest(
            llm_runtime_present=bool(self.hub is not None and self.hub.llm),
            stt_runtime_present=bool(self.hub is not None and self._hub_has_stt_provider("self")),
            llm_provider=str(llm_provider),
            stt_provider=str(stt_provider),
            llm_requires_secret=self._llm_provider_requires_secret(settings.provider.llm),
            stt_requires_secret=self._stt_provider_requires_secret(settings.provider.stt),
            runtime_translation_enabled=bool(self.hub is not None and self.hub.translation_enabled),
            managed_openrouter_can_attempt=managed_openrouter_can_attempt,
            openrouter_managed_selected=(
                settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
            ),
            gemini_model=settings.gemini.llm_model.value,
            qwen_selected_model=settings.qwen.llm_model.value,
            qwen_fallback_models=tuple(model.value for model in QwenLLMModel),
            qwen_base_url=qwen_base_url,
            fast_translation_enabled=FIXED_TRANSLATION_POLICY.fast_translation_enabled,
            google_api_key=secret_value("google_api_key"),
            openrouter_api_key=openrouter_api_key,
            deepseek_api_key=secret_value("deepseek_api_key")
            or os.getenv("DEEPSEEK_API_KEY")
            or "",
            qwen_api_key=qwen_api_key,
            deepgram_api_key=secret_value("deepgram_api_key"),
            soniox_api_key=secret_value("soniox_api_key"),
        )

    async def _apply_provider_status_verification_result(
        self,
        result: ConfiguredProviderStatusVerificationResult,
    ) -> None:
        if self._shutdown_ingress_frozen:
            return
        self.app.set_dashboard_translation_needs_key(result.translation_needs_key)
        if result.translation_enabled_update is not None:
            if not result.translation_enabled_update and self.hub is not None:
                self.hub.translation_enabled = False
            self.app.set_dashboard_translation_enabled(result.translation_enabled_update)
        self.app.set_dashboard_stt_needs_key(result.stt_needs_key)
        if result.stt_enabled_update is not None:
            self.app.set_dashboard_stt_enabled(result.stt_enabled_update)
        if not self._shutdown_ingress_frozen:
            await self._refresh_managed_trial_usage_state_impl(auto_show_founder_letter=False)

    async def _rebuild_llm_provider(self) -> None:
        """Rebuild only the LLM provider without tearing down the entire pipeline."""
        if self.hub is None or self.settings is None:
            return

        async def create_provider() -> object | None:
            secrets = create_secret_store(self.settings.secrets, config_path=self.config_path)
            new_managed_release_service = self._create_managed_openrouter_release_service(
                secrets=secrets
            )
            await self._replace_managed_openrouter_release_service(new_managed_release_service)
            return create_llm_provider(
                self.settings,
                secrets=secrets,
                managed_release_service=self._managed_openrouter_release_service,
                managed_delegate_ready=self._on_managed_trial_delegate_ready,
                runtime_logging=self.runtime_logging,
            )

        outcome = await self._provider_rebuild_runtime.rebuild_llm_provider(
            replace_provider=self.hub.replace_llm_provider,
            create_provider=create_provider,
        )
        llm = outcome.provider

        self.app.set_dashboard_translation_needs_key(
            (llm is None) and self._llm_provider_requires_secret(self.settings.provider.llm)
        )

        await self._refresh_managed_trial_usage_state_best_effort()

        if llm is None:
            self._log_error("LLM provider not available")
            return

        self.log_basic("[Settings] LLM provider rebuilt successfully")

    def _self_stt_provider_request(
        self,
        settings: AppSettings,
        *,
        warmup: bool = False,
    ) -> ProviderRuntimeBuildRequest:
        config = resolve_self_stt_runtime_config(settings)
        transition = self._self_local_asr_transition_request(settings, trigger="runtime")
        return ProviderRuntimeBuildRequest(
            config=config,
            gpu_device_id=settings.stt.gpu_device_id,
            warmup=warmup,
            model_id=transition.model_id if transition is not None else config.model,
            session_options=(
                transition.session_options
                if transition is not None
                else self._local_asr_session_options(
                    source_language=config.source_language,
                    source_mode=config.source_mode,
                )
            ),
        )

    def _peer_stt_provider_request(
        self,
        config: PeerCaptureSessionConfig,
        *,
        warmup: bool = False,
    ) -> ProviderRuntimeBuildRequest:
        assert self.settings is not None
        backend = config.provider_context
        if not isinstance(backend, ResolvedSTTConfig):
            raise TypeError("Peer capture config requires a resolved STT provider context")
        return ProviderRuntimeBuildRequest(
            config=backend,
            gpu_device_id=self.settings.stt.gpu_device_id,
            warmup=warmup,
            model_id=config.model_id or backend.model,
            session_options=config.session_options,
        )

    def _build_local_asr_provider_runtime_factory(
        self,
        *,
        secrets,
    ) -> LocalASRProviderRuntimeFactory:
        assert self.local_asr_provisioning is not None
        return LocalASRProviderRuntimeFactory(
            provider_factory=ManagedSTTProviderFactory(
                secrets=secrets,
                clock=self.clock,
                reset_deadline_s=STT_RESET_DEADLINE_S,
                gpu_model_path=local_gpu_model_path(),
                diagnostics_enabled=self._detailed_audio_diag_enabled,
                on_final_transcript_suppressed=self._on_final_transcript_suppressed,
                runtime_logging=self.runtime_logging,
                fault_profile_provider=lambda: (
                    self._debug_stt_fault_profile if self._debug_audio_fault_allowed() else "none"
                ),
            ),
            provisioning=self.local_asr_provisioning,
            clock=self.clock,
            state_changed=self._on_local_asr_provider_runtime_state_changed,
            diagnostic_sink=self._on_local_asr_provider_runtime_diagnostic,
        )

    def _on_local_asr_provider_runtime_state_changed(
        self,
        snapshot: LocalASRProviderRuntimeSnapshot,
    ) -> None:
        self._gpu_devices = snapshot.gpu.devices
        self._gpu_discovery_attempted = bool(snapshot.gpu.devices) or snapshot.gpu.phase not in {
            "inactive",
            "idle",
        }
        self._gpu_discovery_failed = snapshot.gpu.phase in {"failed", "unsupported"}
        self._gpu_discovery_failure_state = (
            "unsupported"
            if snapshot.gpu.phase == "unsupported"
            else "discovery_failed" if snapshot.gpu.phase == "failed" else None
        )

    def _on_local_asr_provider_runtime_diagnostic(
        self,
        diagnostic: ProviderRuntimeDiagnostic,
    ) -> None:
        fields = [f"event={diagnostic.event}"]
        for name in (
            "channel",
            "provider_id",
            "model_id",
            "device_id",
            "phase",
            "outcome",
            "failure_code",
            "failure_type",
        ):
            value = getattr(diagnostic, name)
            if value is not None:
                fields.append(f"{name}={value}")
        self.log_detailed(f"[LocalASR][ProviderRuntime] {' '.join(fields)}")
        if diagnostic.event == "activation_ready":
            self._log_local_asr_load_result(
                channel=diagnostic.channel or "unknown",
                model_id=diagnostic.model_id or "unknown",
                backend="Vulkan",
                device=diagnostic.device_id or "unknown",
                outcome="ready",
                load_seconds=diagnostic.model_load_seconds or 0.0,
                warmup_seconds=diagnostic.warmup_seconds or 0.0,
            )
        elif diagnostic.event == "activation_failed":
            self._log_local_asr_load_result(
                channel=diagnostic.channel or "unknown",
                model_id=diagnostic.model_id or "unknown",
                backend="Vulkan",
                outcome="failed",
                load_seconds=diagnostic.model_load_seconds or 0.0,
                failure_code=diagnostic.failure_code or "activation_failed",
            )
        elif diagnostic.event == "worker_failed":
            exit_code = (
                f" exit_code={diagnostic.worker_exit_code}"
                if diagnostic.worker_exit_code is not None
                else ""
            )
            self.log_basic(
                "[LocalASR][Worker] backend=Vulkan outcome=failed "
                f"failure_code={diagnostic.failure_code or 'worker_failed'}{exit_code}",
                level=logging.ERROR,
            )
        elif diagnostic.event == "worker_recovery_started":
            self.log_basic(
                "[LocalASR][Worker] backend=Vulkan outcome=restarting "
                f"failure_code={diagnostic.failure_code or 'decode_failure'} "
                "utterance_retry=false",
                level=logging.WARNING,
            )
        elif diagnostic.event == "worker_recovery_ready":
            self.log_basic(
                "[LocalASR][Worker] backend=Vulkan outcome=recovered utterance_retry=false"
            )
        elif diagnostic.event == "decode_attempt" and all(
            value is not None
            for value in (
                diagnostic.audio_seconds,
                diagnostic.decode_seconds,
                diagnostic.rtf,
                diagnostic.queue_wait_seconds,
            )
        ):
            self.log_basic(
                "[LocalASR][Attempt] "
                f"channel={diagnostic.channel or 'unknown'} "
                f"model={diagnostic.model_id or 'unknown'} "
                "backend=Vulkan "
                f"audio_seconds={diagnostic.audio_seconds:.3f} "
                f"decode_seconds={diagnostic.decode_seconds:.3f} "
                f"rtf={diagnostic.rtf:.6f} "
                f"result={diagnostic.outcome or 'unknown'} "
                f"queue_wait_seconds={diagnostic.queue_wait_seconds:.3f}"
            )
        if diagnostic.event == "worker_lifecycle" and diagnostic.phase in {
            "validating",
            "loading",
            "warming",
            "ready",
        }:
            self._set_gpu_ui_state(diagnostic.phase, origin="worker_lifecycle")
        elif diagnostic.event == "activation_ready":
            self._set_gpu_ui_state("ready", origin="activation")
        elif diagnostic.event == "discovery_pending":
            self._set_gpu_ui_state(
                "discovery_pending",
                origin=self._gpu_discovery_origin,
            )
        elif diagnostic.event in {"activation_failed", "worker_failed"}:
            self._set_gpu_ui_state(
                "activation_failed",
                publish_notice=True,
                origin="worker",
            )

    async def _rebuild_stt_provider(self) -> None:
        """Rebuild only the STT provider so later enable uses current settings."""
        if self.hub is None or self.settings is None:
            return

        owner = self._get_self_capture_owner()
        config = self._build_self_capture_session_config(self.settings)
        if self._stt_desired:
            snapshot = await owner.apply_intent(config, enabled=True)
        else:
            snapshot = await owner.prepare_provider(config)
        self._on_self_capture_state_changed(snapshot)
        available = self._project_self_provider_availability(snapshot)
        if not available:
            self._log_error("STT backend not available")
            return
        self.log_basic("[Settings] STT provider replacement completed successfully")

    @staticmethod
    def _local_asr_session_options(
        *,
        source_language: str,
        source_mode: str = "manual",
    ) -> LocalASRSessionOptions:
        from puripuly_heart.core.language import get_local_qwen_language_hint

        return LocalASRSessionOptions(
            source_language=source_language,
            source_mode=source_mode,
            language_hint=(
                None if source_mode == "auto" else get_local_qwen_language_hint(source_language)
            ),
        )

    def _self_local_asr_transition_request(
        self,
        settings: AppSettings,
        *,
        trigger: str,
    ) -> LocalASRTransitionRequest | None:
        provider = settings.provider.stt.value
        if provider == STTProviderName.LOCAL_QWEN_GPU.value:
            model_id = LOCAL_QWEN_GPU_MODEL_ID
            actual_provider = provider
        elif provider in LOCAL_CPU_PROVIDERS:
            decision = resolve_local_asr_selection(
                provider,
                settings.languages.source_language,
            )
            if not decision.supported:
                return None
            model_id = decision.model_id
            actual_provider = decision.effective_provider
        else:
            return None
        return LocalASRTransitionRequest(
            channel="self",
            requested_provider=provider,
            actual_provider=actual_provider,
            model_id=model_id,
            session_options=self._local_asr_session_options(
                source_language=settings.languages.source_language,
            ),
            trigger=trigger,
        )

    def _local_asr_transition_diagnostic(self, fields: dict[str, object]) -> None:
        ordered = " ".join(f"{key}={value}" for key, value in fields.items())
        self.log_detailed(f"[LocalASR][Transition] {ordered}")
        actual_provider = str(fields.get("actual_provider") or "")
        if actual_provider == STTProviderName.LOCAL_QWEN_GPU.value:
            return
        outcome = str(fields.get("outcome") or "")
        if outcome not in {"applied", "failed"}:
            return
        self._log_local_asr_load_result(
            channel=str(fields.get("channel") or "unknown"),
            model_id=str(fields.get("model_id") or "unknown"),
            backend="CPU",
            outcome="ready" if outcome == "applied" else "failed",
            load_seconds=max(0.0, float(fields.get("load_ms") or 0) / 1000.0),
            failure_type=(
                str(fields["failure_type"]) if fields.get("failure_type") is not None else None
            ),
        )

    def _log_local_asr_load_result(
        self,
        *,
        channel: str,
        model_id: str,
        backend: str,
        outcome: str,
        load_seconds: float,
        failure_type: str | None = None,
        device: str | None = None,
        warmup_seconds: float | None = None,
        failure_code: str | None = None,
    ) -> None:
        fields = [
            f"channel={channel}",
            f"model={model_id}",
            f"backend={backend}",
        ]
        if device is not None:
            fields.append(f"device={device}")
        fields.extend((f"outcome={outcome}", f"load_seconds={max(0.0, load_seconds):.3f}"))
        if warmup_seconds is not None:
            fields.append(f"warmup_seconds={max(0.0, warmup_seconds):.3f}")
        if failure_type is not None:
            fields.append(f"failure_type={failure_type}")
        if failure_code is not None:
            fields.append(f"failure_code={failure_code}")
        self.log_basic(
            f"[LocalASR][Load] {' '.join(fields)}",
            level=logging.ERROR if outcome == "failed" else logging.INFO,
        )

    def _on_peer_runtime_diagnostic(self, diagnostic: PeerCaptureDiagnostic) -> None:
        unavailable_reason = getattr(
            diagnostic,
            "detail",
            getattr(diagnostic, "process_unavailable_reason", None),
        )
        self.log_detailed(
            "[PeerRuntime] "
            f"reason={diagnostic.reason.value} "
            f"capture_kind={diagnostic.capture_kind} "
            f"unavailable_reason={unavailable_reason}"
        )
        if diagnostic.reason is not PeerCaptureFailureReason.PROCESS_PROVIDER_FAILED:
            suffix = (
                f" unavailable_reason={unavailable_reason}"
                if unavailable_reason is not None
                else ""
            )
            self.log_basic(
                "[PeerRuntime] outcome=failed "
                f"reason={diagnostic.reason.value} capture_kind={diagnostic.capture_kind}{suffix}",
                level=logging.ERROR,
            )
        if diagnostic.capture_kind == "process":
            self._peer_process_warning_reason = self._peer_process_warning_reason_for_diagnostic(
                diagnostic
            )
            self._refresh_overlay_peer_consumers()

    @staticmethod
    def _peer_process_warning_reason_for_diagnostic(
        diagnostic: PeerCaptureDiagnostic,
    ) -> str:
        if diagnostic.reason is PeerCaptureFailureReason.PROCESS_TARGET_UNAVAILABLE:
            unavailable = (
                getattr(
                    diagnostic,
                    "detail",
                    getattr(diagnostic, "process_unavailable_reason", None),
                )
                or "no_process"
            )
            return f"process_unavailable_{unavailable}"
        return diagnostic.reason.value

    def loopback_capture_summary(self, settings: AppSettings | None = None) -> str:
        resolved_settings = settings or self.settings
        if resolved_settings is None:
            return self.app.localize("settings.default_option")
        target = self._resolve_peer_capture_target(resolved_settings)
        if target.kind == "named_output_device":
            return target.device_name or self.app.localize("settings.default_option")
        if target.kind == "process":
            return self._process_capture_display_name(target)
        return self.app.localize("settings.default_option")

    def list_loopback_capture_options(self) -> list[OptionItem]:
        options = self.list_loopback_process_options()
        options.extend(self.list_loopback_device_options())
        return options

    def list_loopback_process_options(self) -> list[OptionItem]:
        process_section = self.app.localize("settings.desktop_audio.section.process")
        options: list[OptionItem] = []
        seen_process_values: set[str] = set()
        for candidate in ProcessCaptureResolver(
            snapshots=PsutilCurrentUserProcessSnapshots()
        ).enumerate_candidates():
            value = self._encode_process_capture_option(candidate.target)
            seen_process_values.add(value)
            options.append(
                OptionItem(
                    value=value,
                    label=self._process_option_label(candidate.target, candidate.name),
                    description="",
                    disabled=not candidate.enabled,
                    section=process_section,
                )
            )
        current_value = self.current_loopback_capture_option_value()
        if current_value.startswith("process:") and current_value not in seen_process_values:
            process = self._decode_capture_option(current_value).process
            if process is not None:
                options.insert(
                    0,
                    OptionItem(
                        value=current_value,
                        label=self._process_option_label(process, ""),
                        description="",
                        disabled=False,
                        section=process_section,
                    ),
                )
        options.sort(key=lambda o: o.disabled)
        return options

    def list_loopback_device_options(self) -> list[OptionItem]:
        device_section = self.app.localize("settings.desktop_audio.section.device")
        options: list[OptionItem] = [
            OptionItem(
                value="device:",
                label=self.app.localize("settings.default_option"),
                description="",
                disabled=False,
                section=device_section,
            )
        ]
        for device in self._enumerate_loopback_device_names():
            options.append(
                OptionItem(
                    value=f"device:{device}",
                    label=device,
                    description="",
                    disabled=False,
                    section=device_section,
                )
            )
        return options

    def current_loopback_capture_option_value(self, settings: AppSettings | None = None) -> str:
        resolved_settings = settings or self.settings
        if resolved_settings is None:
            return "device:"
        target = self._resolve_peer_capture_target(resolved_settings)
        if target.kind == "process":
            process = self._process_target_from_resolved(target)
            return self._encode_process_capture_option(process)
        if target.kind == "named_output_device":
            return f"device:{target.device_name or ''}"
        return "device:"

    async def apply_loopback_capture_option(self, value: str) -> None:
        if self.settings is None:
            return
        capture_target = self._decode_capture_option(value)
        next_settings = self._get_settings_owner().update_capture_target(
            self.settings,
            capture_target,
        )
        # Keep non-capture runtime fields from the live session.
        next_settings.ui.overlay_enabled = self.settings.ui.overlay_enabled
        next_settings.ui.peer_translation_enabled = self.settings.ui.peer_translation_enabled
        self.settings = next_settings
        self._vnext_settings_authoritative = True
        self._remember_canonical_legacy_projection(next_settings)
        self._peer_process_warning_reason = None
        await self._refresh_peer_stt_runtime()
        self._sync_effective_hub_flags(self.settings)
        self._refresh_overlay_peer_consumers()
        with contextlib.suppress(Exception):
            self.app.refresh_settings_loopback_capture_target(self.settings)

    def peer_warning_action_is_retry(self) -> bool:
        if self.settings is None or not self.settings.ui.peer_translation_enabled:
            return False
        reason = self._peer_process_warning_reason
        return reason is not None and (
            reason.startswith("process_") or reason.startswith("process_unavailable_")
        )

    @staticmethod
    def _encode_process_capture_option(target: ProcessCaptureTargetIntent) -> str:
        if target.kind == "discord":
            return f"process:discord:{target.discord_channel}"
        if target.kind == "vrchat":
            return f"process:vrchat:{target.executable_identity}"
        return f"process:generic:{target.executable_identity}"

    def _decode_capture_option(self, value: str) -> CaptureTargetIntent:
        if value.startswith("process:"):
            payload = value[len("process:") :]
            kind, _, rest = payload.partition(":")
            if kind == "discord":
                return CaptureTargetIntent.process_target(ProcessCaptureTargetIntent.discord(rest))
            if kind == "vrchat":
                return CaptureTargetIntent.process_target(ProcessCaptureTargetIntent.vrchat(rest))
            return CaptureTargetIntent.process_target(
                ProcessCaptureTargetIntent.generic_executable(rest)
            )
        device_name = value[len("device:") :] if value.startswith("device:") else value
        if device_name:
            return CaptureTargetIntent.named_output_device(device_name)
        return CaptureTargetIntent.default_output_device()

    def _process_target_from_resolved(
        self,
        target: ResolvedDesktopAudioCaptureTarget,
    ) -> ProcessCaptureTargetIntent:
        if target.process_kind == "discord":
            return ProcessCaptureTargetIntent.discord(target.discord_channel or "")
        if target.process_kind == "vrchat":
            return ProcessCaptureTargetIntent.vrchat(target.executable_identity or "")
        return ProcessCaptureTargetIntent.generic_executable(target.executable_identity or "")

    def _process_capture_display_name(self, target: ResolvedDesktopAudioCaptureTarget) -> str:
        process = self._process_target_from_resolved(target)
        return self._process_option_label(process, "")

    def _process_option_label(
        self,
        target: ProcessCaptureTargetIntent,
        fallback_name: str,
    ) -> str:
        if target.kind == "vrchat":
            base = self.app.localize("settings.desktop_audio.process.vrchat")
        elif target.kind == "discord":
            channel = target.discord_channel or "stable"
            if channel == "ptb":
                base = self.app.localize("settings.desktop_audio.process.discord_ptb")
            elif channel == "canary":
                base = self.app.localize("settings.desktop_audio.process.discord_canary")
            else:
                base = self.app.localize("settings.desktop_audio.process.discord_stable")
        elif fallback_name:
            return fallback_name
        else:
            path = target.executable_identity or ""
            basename = path.rsplit("\\", 1)[-1]
            if basename.lower().endswith(".exe"):
                basename = basename[:-4]
            base = basename or self.app.localize("settings.default_option")
        if target.kind in {"vrchat", "discord"} and fallback_name:
            count_suffix = fallback_name.rsplit(" (", 1)
            if len(count_suffix) == 2 and count_suffix[1].endswith(")"):
                count = count_suffix[1][:-1]
                if count.isdigit():
                    return f"{base} ({count})"
        return base

    @staticmethod
    def _enumerate_loopback_device_names() -> list[str]:
        names: list[str] = []
        manager = None
        try:
            import pyaudiowpatch as pyaudio  # type: ignore

            manager = pyaudio.PyAudio()
            seen: set[str] = set()
            for info in manager.get_loopback_device_info_generator():
                name = str(info.get("name", "") or "").strip()
                if not name or name in seen:
                    continue
                seen.add(name)
                names.append(name)
        except Exception:
            return names
        finally:
            if manager is not None:
                with contextlib.suppress(Exception):
                    manager.terminate()
        return names

    async def _resolve_peer_capture_target_for_owner(
        self,
        target: PeerCaptureTargetIntent,
    ) -> PeerCaptureTargetResolution:
        if target.kind != "process":
            return PeerCaptureTargetResolution(
                PeerCaptureTargetStatus.RESOLVED,
                target=PeerCaptureResolvedTarget(intent=target),
            )
        process_target = self._process_target_from_capture_target(target)
        resolution = await asyncio.to_thread(
            lambda: ProcessCaptureResolver(
                snapshots=PsutilCurrentUserProcessSnapshots()
            ).resolve_for_start(process_target)
        )
        if resolution.identity is None:
            return PeerCaptureTargetResolution(
                PeerCaptureTargetStatus.UNAVAILABLE,
                reason=resolution.unavailable_reason,
            )
        return PeerCaptureTargetResolution(
            PeerCaptureTargetStatus.RESOLVED,
            target=PeerCaptureResolvedTarget(
                intent=target,
                capture_descriptor=resolution,
            ),
        )

    async def _create_peer_audio_source_from_runtime_config(
        self,
        config: PeerCaptureSessionConfig,
        resolved_target: PeerCaptureResolvedTarget | None = None,
    ) -> DesktopPeerPipeline:
        if resolved_target is None:
            resolution = await self._resolve_peer_capture_target_for_owner(config.capture_target)
            if resolution.target is None:
                raise ProcessCaptureTargetUnavailableError(
                    cast(Any, resolution.reason or "no_process")
                )
            resolved_target = resolution.target
        target = resolved_target.intent
        if target.kind == "process":
            return self._create_process_peer_audio_source(
                config,
                resolution=resolved_target.capture_descriptor,
            )

        device_name = target.device_name or config.output_device
        raw_source = DesktopLoopbackAudioSource(device_name=device_name)
        self.log_detailed(
            "[AudioDiag][Loopback][peer] "
            f"requested_device={device_name!r} "
            f"resolved_device_name={getattr(raw_source, 'resolved_device_name', None)!r} "
            f"resolved_device_index={getattr(raw_source, 'resolved_device_index', None)} "
            f"resolved_channels={getattr(raw_source, 'resolved_channels', None)} "
            f"actual_sample_rate_hz={getattr(raw_source, 'actual_sample_rate_hz', None)} "
            f"used_default_fallback={getattr(raw_source, 'used_default_fallback', None)}"
        )
        wrapped_source = self._wrap_diagnostic_audio_source(raw_source, channel_label="peer")
        return DesktopPeerPipeline(
            source=wrapped_source,
            target_sample_rate_hz=self._peer_capture_sample_rate(config),
            is_detailed_enabled=self._detailed_audio_diag_enabled,
            log_detailed=lambda message: self.log_detailed(message),
        )

    def _create_process_peer_audio_source(
        self,
        config: PeerCaptureSessionConfig,
        *,
        resolution: object,
    ) -> DesktopPeerPipeline:
        identity = getattr(resolution, "identity", resolution)
        if identity is None:
            raise RuntimeError("resolved process capture requires a process identity")
        raw_source = ProcessAudioCaptureSource(
            identity=identity,
            watcher=PsutilProcessIdentityWatcher(),
        )
        self.log_detailed(
            "[AudioDiag][ProcessCapture][peer] "
            f"target_kind={config.capture_target.process_kind} capture=process"
        )
        wrapped_source = self._wrap_diagnostic_audio_source(raw_source, channel_label="peer")
        return DesktopPeerPipeline(
            source=wrapped_source,
            target_sample_rate_hz=self._peer_capture_sample_rate(config),
            is_detailed_enabled=self._detailed_audio_diag_enabled,
            log_detailed=lambda message: self.log_detailed(message),
        )

    @staticmethod
    def _process_target_from_capture_target(
        target: PeerCaptureTargetIntent,
    ) -> ProcessCaptureTargetIntent:
        if target.kind != "process" or target.process_kind is None:
            raise ValueError("process peer source requires a process capture target")
        if target.process_kind == "discord":
            return ProcessCaptureTargetIntent.discord(target.discord_channel or "")
        if target.process_kind == "vrchat":
            return ProcessCaptureTargetIntent.vrchat(target.executable_identity or "")
        return ProcessCaptureTargetIntent.generic_executable(target.executable_identity or "")

    @staticmethod
    def _peer_capture_sample_rate(config: object) -> int:
        sample_rate = getattr(config, "target_sample_rate_hz", None)
        if sample_rate is not None:
            return int(sample_rate)
        return int(getattr(getattr(config, "backend"), "sample_rate_hz"))

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

        if not self.app.show_local_qwen_hallucination_dialog():
            self.log_detailed(
                "[STT][SuppressedFinalNotification] "
                f"local_qwen_guidance count={count} guidance_modal=unavailable"
            )
            return

        self._local_qwen_hallucination_modal_shown = True

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

    def _create_peer_vad_from_runtime_config(
        self,
        config: PeerCaptureSessionConfig,
        model_path: Path | None = None,
    ) -> VadGating:
        model_path = model_path or ensure_silero_vad_onnx()
        return create_peer_vad_gating(
            engine=SileroVadOnnx(model_path=model_path),
            sample_rate_hz=self._peer_capture_sample_rate(config),
            ring_buffer_ms=config.vad_pre_roll_ms,
            speech_threshold=config.vad_speech_threshold,
            hangover_ms=config.vad_hangover_ms,
            diagnostic_event_callback=lambda message: self.log_detailed(message),
            diagnostics_enabled=self._detailed_audio_diag_enabled,
            diagnostic_label="peer",
        )

    async def _run_peer_audio_vad_loop(self, **kwargs: object) -> None:
        from puripuly_heart.core.runtime.audio_vad_loop import run_audio_vad_loop

        await run_audio_vad_loop(
            **kwargs,
            channel_label="peer",
            is_detailed_enabled=self._detailed_audio_diag_enabled,
            log_detailed=lambda message: self.log_detailed(message),
        )

    async def _refresh_peer_stt_runtime(self, *, stop_mode: str = "retain") -> None:
        if self.settings is None or self.hub is None or self._peer_runtime is None:
            return

        config = self._build_peer_runtime_config(self.settings)
        desired_active = self._peer_runtime_should_be_active(self.settings)
        previous_signature = getattr(self._peer_runtime, "current_signature", None)
        peer_local_provider = bool(desired_active and config.local_provider)
        peer_local_transition = bool(
            peer_local_provider
            and previous_signature is not None
            and previous_signature != config.runtime_signature
        )
        peer_local_loading = bool(
            peer_local_provider
            and (
                not self._hub_has_stt_provider("peer")
                or previous_signature != config.runtime_signature
            )
        )
        if peer_local_loading:
            self._peer_asr_model_loading = True
            self._sync_local_stt_notice()
            self._refresh_overlay_peer_consumers()
        transition_settings = self.settings
        try:
            await self._provider_rebuild_runtime.apply_peer_policy(
                peer_runtime=self._peer_runtime,
                config=config,
                desired_active=desired_active,
                stop_mode="release" if stop_mode == "release" else "retain",
            )
            transition_status = getattr(
                self._peer_runtime,
                "last_local_asr_transition_status",
                "idle",
            )
            if peer_local_transition and transition_status == "superseded":
                raise PeerLocalASRTransitionSuperseded
            if peer_local_transition and transition_status == "failed":
                raise RuntimeError("peer local ASR transition failed")
        except PeerLocalASRTransitionSuperseded:
            self._superseded_local_asr_settings_ids.add(id(transition_settings))
            raise
        finally:
            if peer_local_loading:
                self._peer_asr_model_loading = False
                self._sync_local_stt_notice()
                self._refresh_overlay_peer_consumers()
        self._last_peer_stt_runtime_signature = config.runtime_signature
        self._sync_effective_hub_flags(self.settings)

    async def _rebuild_pipeline(self, *, rebuild_stt: bool) -> None:
        self.log_detailed(
            f"[Settings] Rebuilding pipeline detail: rebuild_stt={rebuild_stt} overlay_state={self.overlay_state}"
        )
        _ = rebuild_stt
        cleanup_failures: list[Exception] = []
        restore_stt_enabled = self._stt_desired

        await self._close_peer_runtime_for_release(cleanup_failures)

        try:
            await self.set_stt_enabled(False)
        except Exception as exc:
            cleanup_failures.append(exc)
        await self._close_self_capture_owner_for_release(cleanup_failures)
        await self._configure_vrc_mic_receiver(enabled=False)
        await self._stop_hub_for_release(cleanup_failures)
        if self.hub is None:
            self._bridge_task = None
            self._ui_event_bridge = None
        if self.sender is not None:
            with contextlib.suppress(Exception):
                self.sender.close()
        self.sender = None
        self.osc = None
        _raise_lifecycle_cleanup_failures(
            "GUI controller pipeline rebuild cleanup failed",
            cleanup_failures,
        )
        await self._init_pipeline()
        assert self.hub is not None
        presenter = self._current_overlay_presenter_for_direct_runtime_command()
        if presenter is not None:
            await self._replace_hub_overlay_sink(presenter)

        self.app.set_dashboard_translation_needs_key(
            (self.hub.llm is None)
            and self._llm_provider_requires_secret(self.settings.provider.llm)
        )
        self.app.set_dashboard_stt_needs_key(
            self._dashboard_stt_needs_key(stt_available=self._hub_has_stt_provider("self"))
        )
        self.hub.translation_enabled = (
            self.app.dashboard_translation_enabled() and self.hub.llm is not None
        )
        self.app.set_dashboard_translation_enabled(self.hub.translation_enabled)

        await self.hub.start(auto_flush_osc=True)

        bridge = self._create_ui_event_bridge(runtime_logging=self.runtime_logging)
        self._start_ui_event_bridge_task(bridge)

        if self.overlay_state == "connected" and presenter is not None:
            await self._refresh_overlay_runtime_dependencies()

        if restore_stt_enabled:
            await self.set_stt_enabled(True)

        self._schedule_provider_status_verification()

    async def _init_pipeline(self) -> None:
        assert self.settings is not None
        self._get_local_asr_provisioning_owner()
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

        stt_request = None
        if self.settings.provider.stt != STTProviderName.LOCAL_QWEN_GPU:
            try:
                stt_request = self._self_stt_provider_request(self.settings)
            except Exception:
                self._log_error("STT backend not available")

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
            stt=None,
            llm=llm,
            osc=osc,
            peer_stt=None,
            clock=self.clock,
            runtime_logging=self.runtime_logging,
            local_asr_provider_runtime_factory=self._build_local_asr_provider_runtime_factory(
                secrets=secrets
            ),
            source_language=self.settings.languages.source_language,
            target_language=self.settings.languages.target_language,
            peer_source_language=self.settings.languages.peer_source_language,
            peer_target_language=self.settings.languages.peer_target_language,
            system_prompt=self.settings.system_prompt,
            chatbox_include_source=self.settings.osc.chatbox_include_source,
            fallback_transcript_only=True,
            translation_enabled=True,
            peer_translation_enabled=False,
            integrated_context_enabled=True,
            low_latency_mode=FIXED_TRANSLATION_POLICY.fast_translation_enabled,
            low_latency_merge_gap_ms=self.settings.stt.low_latency_merge_gap_ms,
            low_latency_spec_retry_max=self.settings.stt.low_latency_spec_retry_max,
            hangover_s=(
                self.settings.stt.low_latency_vad_hangover_ms / 1000.0
                if FIXED_TRANSLATION_POLICY.fast_translation_enabled
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

        prior_self_capture_owner = self._self_capture_owner
        if prior_self_capture_owner is not None:
            await prior_self_capture_owner.close()
        self.sender = sender
        self.osc = osc
        self.hub = hub
        self._self_capture_owner = None
        self_capture_owner = self._get_self_capture_owner()
        if stt_request is not None:
            snapshot = await self_capture_owner.prepare_provider(
                self._build_self_capture_session_config(self.settings)
            )
            if snapshot.provider_status.value != "ready":
                self._log_error("STT backend not available")

        self._peer_runtime = compose_peer_capture_session_owner(
            hub=hub,
            admission=_PeerCaptureAdmissionAdapter(self._admit_peer_capture),
            target_resolver=_PeerCaptureTargetResolverAdapter(
                self._resolve_peer_capture_target_for_owner
            ),
            clock=self.clock,
            provider_request_factory=lambda config, warmup: self._peer_stt_provider_request(
                config,
                warmup=warmup,
            ),
            source_factory=self._create_peer_audio_source_from_runtime_config,
            vad_factory=self._create_peer_vad_from_runtime_config,
            run_audio_loop=self._run_peer_audio_vad_loop,
            vad_sink=_PeerCaptureVadSink(lambda: self.hub),
            state_changed=self._on_peer_capture_state_changed,
            diagnostic_sink=self._on_peer_runtime_diagnostic,
            local_asr_diagnostic_sink=self._local_asr_transition_diagnostic,
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
            self.telemetry_client = None
            return None
        release_settings = self._managed_openrouter_release_settings()
        if release_settings is None:
            self.telemetry_client = None
            return None

        from puripuly_heart import __version__

        try:
            client = HttpManagedOpenRouterBrokerClient(
                base_url=self.settings.openrouter.broker_base_url,
            )
            self.telemetry_client = client
        except ValueError as exc:
            logger.warning(
                "[Managed OpenRouter] Invalid broker base URL %r; using unavailable fallback: %s",
                self.settings.openrouter.broker_base_url,
                exc,
            )
            client = UnavailableManagedOpenRouterReleaseClient()
            self.telemetry_client = None

        return ManagedOpenRouterReleaseService(
            openrouter_config=build_openrouter_release_runtime_config(release_settings),
            managed_state=build_managed_identity_state_port(
                self.settings,
                self._managed_identity_persistence_callback(self.settings),
            ),
            secrets=secrets,
            client=client,
            raw_hardware_fingerprint_provider=get_raw_hardware_fingerprint,
            app_version=__version__,
            on_discord_callback_received=self._on_discord_managed_auth_callback_received,
        )

    def _get_oauth_runtime(self) -> OAuthRuntime:
        return self._get_openrouter_pkce_flow_owner().get_runtime()

    @property
    def _oauth_runtime(self) -> OAuthRuntime | None:
        owner = self._openrouter_pkce_flow_owner
        return owner.runtime if owner is not None else None

    @_oauth_runtime.setter
    def _oauth_runtime(self, runtime: OAuthRuntime | None) -> None:
        self._get_openrouter_pkce_flow_owner().runtime = runtime

    @property
    def _openrouter_pkce_client(self) -> object | None:
        owner = self._openrouter_pkce_flow_owner
        return owner.active_client if owner is not None else None

    @_openrouter_pkce_client.setter
    def _openrouter_pkce_client(self, client: object | None) -> None:
        self._get_openrouter_pkce_flow_owner().active_client = client

    def _get_openrouter_pkce_flow_owner(self) -> OpenRouterPkceFlowOwner:
        owner = self._openrouter_pkce_flow_owner
        if owner is None:
            owner = OpenRouterPkceFlowOwner(
                client_factory=lambda: self._create_openrouter_pkce_client(),
            )
            self._openrouter_pkce_flow_owner = owner
        return owner

    async def _close_oauth_runtime(self) -> None:
        owner = self._openrouter_pkce_flow_owner
        if owner is not None:
            await owner.close()

    async def _close_oauth_runtime_for_release(self, failures: list[Exception]) -> None:
        try:
            await self._close_oauth_runtime()
        except Exception as exc:
            failures.append(exc)

    async def _close_app_oauth_runtime_for_release(self, failures: list[Exception]) -> None:
        close_oauth_runtime = getattr(self.app, "close_oauth_runtime", None)
        if not callable(close_oauth_runtime):
            return
        try:
            result = close_oauth_runtime()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            failures.append(exc)

    def _on_discord_managed_auth_callback_received(self) -> None:
        hook = self._discord_managed_auth_callback_received_hook
        if callable(hook):
            hook()

    @property
    def _microphone_test_runtime(self) -> MicTestRuntime | None:
        owner = self._microphone_test_owner
        return owner.runtime_if_created if owner is not None else None

    @_microphone_test_runtime.setter
    def _microphone_test_runtime(self, runtime: MicTestRuntime | None) -> None:
        self._get_microphone_test_owner().runtime = runtime

    @property
    def _microphone_test_meter_level(self) -> float:
        owner = self._microphone_test_owner
        return owner.meter_level if owner is not None else 0.0

    @_microphone_test_meter_level.setter
    def _microphone_test_meter_level(self, value: float) -> None:
        self._get_microphone_test_owner().meter_level = value

    @property
    def _last_microphone_test_audio_settings_signature(
        self,
    ) -> tuple[object, ...] | None:
        owner = self._microphone_test_owner
        return owner.audio_signature if owner is not None else None

    @_last_microphone_test_audio_settings_signature.setter
    def _last_microphone_test_audio_settings_signature(
        self,
        signature: tuple[object, ...] | None,
    ) -> None:
        self._get_microphone_test_owner().audio_signature = signature

    @property
    def microphone_test_meter_level(self) -> float:
        return self._microphone_test_meter_level

    @property
    def microphone_test_active(self) -> bool:
        owner = self._microphone_test_owner
        return owner.active if owner is not None else False

    def _get_microphone_test_owner(self) -> MicrophoneTestSessionOwner:
        owner = self._microphone_test_owner
        if owner is None:
            owner = MicrophoneTestSessionOwner(
                capture_port=self._build_microphone_test_capture_adapter(),
                capture_request_factory=self._microphone_test_capture_request,
                self_capture_snapshot=self._microphone_test_self_capture_state,
                disable_self_capture=lambda: self.set_stt_enabled(False),
                log_sink=self.log_basic,
                diagnostics_sink=self._on_microphone_test_session_diagnostic,
            )
            self._microphone_test_owner = owner
        return owner

    def _on_microphone_test_session_diagnostic(
        self,
        event: str,
        metadata: Mapping[str, object],
        exception: BaseException | None,
    ) -> None:
        if event == "session_failed":
            self._log_error(f"Microphone test error: {exception}")
            return
        if event == "cleanup_retry_failed":
            self._log_error(f"Microphone test cleanup retry failed: {exception}")
            return
        if event == "meter_callback_failed":
            exc_info = (
                (type(exception), exception, exception.__traceback__)
                if exception is not None
                else None
            )
            logger.debug("Microphone-test meter callback raised", exc_info=exc_info)
            return
        self.log_detailed(
            f"[MicTest] owner event={event} error_type={metadata.get('error_type')}",
            level=logging.WARNING,
            exception=exception,
        )

    def _get_microphone_test_runtime(self) -> MicTestRuntime:
        return self._get_microphone_test_owner().runtime

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

    def _microphone_test_self_capture_state(self) -> MicrophoneTestSelfCaptureState:
        source_open = self._mic_task is not None or self._audio_source is not None
        return MicrophoneTestSelfCaptureState(
            stop_required=bool(
                self._stt_desired or self._local_stt_pending_enable_after_install or source_open
            ),
            source_open=source_open,
            close_exception=self._last_mic_loop_close_exception,
        )

    async def start_microphone_test(
        self,
        *,
        meter_callback: Callable[[float], object] | None = None,
        level_log_interval_s: float = _MICROPHONE_TEST_LEVEL_INTERVAL_S,
    ) -> bool:
        if self.settings is None:
            return False
        signature = self._microphone_test_audio_settings_signature(self.settings)
        assert signature is not None
        return await self._get_microphone_test_owner().start(
            MicrophoneTestSessionRequest(
                audio_signature=signature,
                meter_callback=meter_callback,
                level_log_interval_s=level_log_interval_s,
            )
        )

    async def stop_microphone_test(self) -> None:
        owner = self._microphone_test_owner
        if owner is not None:
            await owner.stop()

    async def stop_microphone_test_for_audio_settings_change(self) -> None:
        await self.stop_microphone_test()

    async def _close_microphone_test_runtime_for_release(
        self,
        cleanup_failures: list[Exception],
    ) -> None:
        owner = self._microphone_test_owner
        if owner is None:
            return
        try:
            await owner.close()
        except Exception as exc:
            cleanup_failures.append(exc)

    async def _set_microphone_test_meter_level(
        self,
        value: float,
        meter_callback: Callable[[float], object] | None,
        *,
        generation: int | None = None,
    ) -> None:
        await self._get_microphone_test_owner().set_meter_level(
            value,
            meter_callback,
            generation=generation,
        )

    def _build_microphone_test_capture_adapter(self) -> MicrophoneTestCapturePort:
        return create_microphone_test_capture_adapter(
            clock=self.clock,
            log_sink=self.log_basic,
            meter_sink=lambda value, meter_callback, generation: (
                self._set_microphone_test_meter_level(
                    value,
                    meter_callback,
                    generation=generation,
                )
            ),
            route_observer=observe_microphone_test_route,
            channel_decision=determine_self_mic_capture_channels,
            source_factory=SoundDeviceAudioSource,
        )

    def _microphone_test_capture_request(
        self,
        generation: int | None,
        meter_callback: Callable[[float], object] | None,
        level_log_interval_s: float,
    ) -> MicrophoneTestCaptureRequest:
        assert self.settings is not None
        return MicrophoneTestCaptureRequest(
            saved_host_api=self.settings.audio.input_host_api,
            requested_device=self.settings.audio.input_device,
            internal_channels=self.settings.audio.internal_channels,
            generation=generation,
            meter_callback=meter_callback,
            level_log_interval_s=level_log_interval_s,
        )

    def _build_self_capture_session_config(
        self,
        settings: AppSettings,
    ) -> SelfCaptureSessionConfig:
        provider = settings.provider.stt.value
        transition = self._self_local_asr_transition_request(settings, trigger="runtime")
        return SelfCaptureSessionConfig(
            provider_id=provider,
            provider_signature=self._build_self_stt_provider_signature(settings),
            runtime_signature=self._build_self_stt_runtime_signature(settings),
            capture_signature=self._self_capture_vad_signature(settings),
            target_sample_rate_hz=settings.audio.internal_sample_rate_hz,
            input_host_api=settings.audio.input_host_api,
            input_device=settings.audio.input_device,
            internal_channels=settings.audio.internal_channels,
            ring_buffer_ms=settings.audio.ring_buffer_ms,
            vad_speech_threshold=settings.stt.vad_speech_threshold,
            vad_hangover_ms=(
                settings.stt.low_latency_vad_hangover_ms
                if FIXED_TRANSLATION_POLICY.fast_translation_enabled
                else 1100
            ),
            session_options=(transition.session_options if transition is not None else None),
            local_cpu=provider in LOCAL_CPU_PROVIDERS,
            local_gpu=provider == STTProviderName.LOCAL_QWEN_GPU.value,
            release_backend_after=(
                LOCAL_QWEN_IDLE_RELEASE_SECONDS if provider in LOCAL_CPU_PROVIDERS else None
            ),
            warmup=provider != STTProviderName.LOCAL_QWEN.value,
        )

    async def _admit_self_capture(
        self,
        config: SelfCaptureSessionConfig,
    ) -> SelfCaptureAdmission:
        if self.settings is None:
            return SelfCaptureAdmission(
                SelfCaptureAdmissionStatus.REJECTED,
                reason="runtime_unavailable",
            )
        if config.local_gpu:
            if await self._validate_gpu_activation():
                return SelfCaptureAdmission(SelfCaptureAdmissionStatus.ADMITTED)
            if self._gpu_ui_state in {
                "not_installed",
                "invalid",
                "install_failed",
                "installing",
            }:
                self._gpu_pending_enable_channels = frozenset(
                    {*self._gpu_pending_enable_channels, "self"}
                )
                return SelfCaptureAdmission(
                    SelfCaptureAdmissionStatus.PENDING,
                    reason=self._gpu_ui_state,
                    retain_intent=True,
                )
            return SelfCaptureAdmission(
                SelfCaptureAdmissionStatus.REJECTED,
                reason=self._gpu_ui_state or "gpu_unavailable",
            )
        if not config.local_cpu:
            return SelfCaptureAdmission(
                (
                    SelfCaptureAdmissionStatus.ADMITTED
                    if self.hub is not None
                    else SelfCaptureAdmissionStatus.REJECTED
                ),
                reason=None if self.hub is not None else "runtime_unavailable",
            )
        decision = resolve_local_asr_selection(
            self.settings.provider.stt.value,
            self.settings.languages.source_language,
        )
        if not decision.supported:
            self.app.set_dashboard_stt_enabled(False)
            self.app.set_dashboard_stt_needs_key(False)
            self._show_short_stt_message("local_stt.language_unsupported")
            return SelfCaptureAdmission(
                SelfCaptureAdmissionStatus.REJECTED,
                reason="language_unsupported",
            )
        current_status = self._current_local_stt_runtime_status()
        if current_status == "downloading":
            self._local_stt_pending_enable_after_install = True
            self._local_stt_pending_enable_generation = self._stt_activation_generation
            self.app.set_dashboard_stt_enabled(False)
            self._show_short_stt_message("local_stt.download_in_progress")
            return SelfCaptureAdmission(
                SelfCaptureAdmissionStatus.PENDING,
                reason=current_status,
                retain_intent=True,
            )
        if current_status in {"missing", "invalid", "download_failed"}:
            self._request_unavailable_local_asr_repair(
                current_status,
                channel="self",
                activation_generation=self._stt_activation_generation,
            )
            return SelfCaptureAdmission(
                SelfCaptureAdmissionStatus.PENDING,
                reason=current_status,
                retain_intent=True,
            )
        return SelfCaptureAdmission(
            (
                SelfCaptureAdmissionStatus.ADMITTED
                if self.hub is not None
                else SelfCaptureAdmissionStatus.REJECTED
            ),
            reason=None if self.hub is not None else "runtime_unavailable",
        )

    def _get_self_capture_owner(self) -> SelfCaptureSessionOwner:
        if self._self_capture_owner is not None:
            return self._self_capture_owner
        self._self_capture_owner = compose_self_capture_session_owner(
            hub=self.hub,
            admission=_SelfCaptureAdmissionAdapter(self._admit_self_capture),
            provider_request_factory=self._self_capture_provider_request,
            source_factory=self._create_self_capture_source,
            vad_factory=self._create_self_capture_vad,
            run_audio_loop=self._run_self_capture_audio_loop,
            vad_sink=_SelfCaptureVadSink(lambda: self.hub),
            state_changed=self._on_self_capture_state_changed,
            diagnostic_sink=self._on_self_capture_diagnostic,
            audio_gate_reset=(
                self.vrc_mic_audio_gate.reset if self.vrc_mic_audio_gate is not None else None
            ),
        )
        return self._self_capture_owner

    def _self_capture_provider_request(
        self,
        config: SelfCaptureSessionConfig,
        warmup: bool,
    ) -> ProviderRuntimeBuildRequest:
        _ = config
        if self.settings is None:
            raise RuntimeError("Self provider request requires settings")
        return self._self_stt_provider_request(self.settings, warmup=warmup)

    def _on_self_capture_state_changed(
        self,
        snapshot: SelfCaptureSessionSnapshot,
    ) -> None:
        owner = self._self_capture_owner
        if owner is not None:
            self._mic_task = owner.loop_task
            self._audio_source = cast(
                AudioSource | None,
                owner.source if owner.source is not None else owner.cleanup_source,
            )
            self._vad = cast(VadGating | None, owner.vad)
            self._last_mic_loop_close_exception = owner.last_cleanup_exception
        self._stt_desired = snapshot.desired_active
        self._stt_activation_generation = snapshot.generation
        self._stt_activation_starting = snapshot.state in {
            SelfCaptureSessionState.STARTING,
            SelfCaptureSessionState.ADMISSION_PENDING,
        }
        self._stt_activation_failed = snapshot.state is SelfCaptureSessionState.FAULTED

    def _on_self_capture_diagnostic(self, diagnostic: SelfCaptureDiagnostic) -> None:
        fields = [
            f"event={diagnostic.event.value}",
            f"generation={diagnostic.generation}",
            f"state={diagnostic.state.value}",
        ]
        if diagnostic.provider_id is not None:
            fields.append(f"provider={diagnostic.provider_id}")
        if diagnostic.reason is not None:
            fields.append(f"reason={diagnostic.reason.value}")
        if diagnostic.detail is not None:
            fields.append(f"detail={diagnostic.detail}")
        self.log_detailed(f"[SelfCapture] {' '.join(fields)}")

    def _create_self_capture_vad(self, config: SelfCaptureSessionConfig) -> VadGating:
        model_path = ensure_silero_vad_onnx()
        return VadGating(
            engine=SileroVadOnnx(model_path=model_path),
            sample_rate_hz=config.target_sample_rate_hz,
            ring_buffer_ms=config.ring_buffer_ms,
            speech_threshold=config.vad_speech_threshold,
            hangover_ms=config.vad_hangover_ms,
            diagnostic_event_callback=lambda message: self.log_detailed(message),
            diagnostics_enabled=self._detailed_audio_diag_enabled,
            diagnostic_label="self",
        )

    def _create_self_capture_source(self, config: SelfCaptureSessionConfig) -> AudioSource:
        def resolve_device(host_api: str, device: str) -> int | None:
            try:
                return resolve_sounddevice_input_device(host_api=host_api, device=device)
            except Exception as exc:
                self.log_detailed(
                    "[STT] Device resolution detail: "
                    f"host_api={host_api!r} device={device!r} error={exc}",
                    level=logging.WARNING,
                )
                return None

        def source_int(source: SoundDeviceAudioSource, attr: str, fallback: int) -> int:
            try:
                return int(getattr(source, attr, fallback))
            except Exception:
                return fallback

        def open_source_once(
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
            metadata = decision.metadata
            opened_channels = source_int(source, "opened_channels", requested_channels)
            frame_channels = source_int(source, "frame_channels", opened_channels)
            actual_sample_rate_hz = source_int(source, "actual_sample_rate_hz", 0)
            self.log_detailed(
                "[STT] Microphone capture format: "
                f"attempt={attempt!r} "
                f"internal_channels={decision.internal_channels} "
                f"preferred_capture_channels={decision.preferred_capture_channels} "
                f"requested_channels={requested_channels} "
                f"opened_channels={opened_channels} "
                f"frame_channels={frame_channels} "
                "frame_channels_source='opened_fallback' "
                f"saved_host_api={config.input_host_api!r} "
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
            return source

        def open_source_with_mono_retry(
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
                internal_channels=config.internal_channels,
            )
            try:
                return open_source_once(
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
                if decision.preferred_capture_channels <= config.internal_channels:
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
                    "will_retry_mono=True "
                    f"error={exc}",
                    level=logging.WARNING,
                )
                return open_source_once(
                    dev_idx,
                    attempt=f"{attempt}_mono_retry",
                    requested_channels=config.internal_channels,
                    decision=decision,
                    host_api_for_log=host_api_for_log,
                    device_for_log=device_for_log,
                    wasapi_auto_convert=wasapi_auto_convert,
                    wasapi_exclusive=wasapi_exclusive,
                )

        host_api_profile = normalize_input_host_api(config.input_host_api)
        host_api = host_api_profile.actual_host_api
        first_open_used_wasapi_flags = (
            host_api_profile.wasapi_auto_convert or host_api_profile.wasapi_exclusive
        )
        device_idx = resolve_device(host_api, config.input_device)
        source: SoundDeviceAudioSource | None = None
        try:
            source = open_source_with_mono_retry(
                device_idx,
                attempt="primary",
                host_api_for_log=host_api,
                device_for_log=config.input_device,
                wasapi_auto_convert=host_api_profile.wasapi_auto_convert,
                wasapi_exclusive=host_api_profile.wasapi_exclusive,
            )
            self.log_detailed(
                "[STT] Microphone opened: "
                f"saved_host_api={config.input_host_api!r} "
                f"actual_host_api={host_api!r} "
                f"device={config.input_device!r} "
                f"device_idx={device_idx} "
                f"wasapi_auto_convert={host_api_profile.wasapi_auto_convert} "
                f"wasapi_exclusive={host_api_profile.wasapi_exclusive}"
            )
        except Exception as exc:
            self.log_detailed(
                "[STT] Microphone open detail: "
                f"host_api={host_api!r} device={config.input_device!r} error={exc}",
                level=logging.ERROR,
            )
        if source is None and config.input_device:
            fallback_idx = resolve_device("", config.input_device)
            if fallback_idx != device_idx or first_open_used_wasapi_flags:
                try:
                    source = open_source_with_mono_retry(
                        fallback_idx,
                        attempt="name_fallback",
                        host_api_for_log="",
                        device_for_log=config.input_device,
                    )
                    self.log_detailed(
                        f"[STT] Microphone opened with fallback: device_idx={fallback_idx}"
                    )
                except Exception as exc:
                    self.log_detailed(
                        f"[STT] Fallback microphone detail: error={exc}",
                        level=logging.ERROR,
                    )
        if source is None:
            try:
                source = open_source_with_mono_retry(
                    None,
                    attempt="system_default",
                    host_api_for_log="",
                    device_for_log="",
                )
                self.log_detailed("[STT] Microphone opened with system default")
            except Exception as exc:
                self.log_detailed(
                    f"[STT] System default microphone detail: error={exc}",
                    level=logging.ERROR,
                )
        if source is None:
            raise RuntimeError("All microphone attempts failed")
        return self._wrap_diagnostic_audio_source(source, channel_label="self")

    async def _run_self_capture_audio_loop(self, **kwargs: object) -> None:
        from puripuly_heart.core.runtime.audio_vad_loop import run_audio_vad_loop

        await run_audio_vad_loop(
            **kwargs,
            audio_gate=self.vrc_mic_audio_gate,
            channel_label="self",
            is_detailed_enabled=self._detailed_audio_diag_enabled,
            log_detailed=lambda message: self.log_detailed(message),
        )

    def _create_vrc_osc_receiver_for_runtime(self, **kwargs: object) -> VrcOscReceiver:
        return VrcOscReceiver(**kwargs)  # type: ignore[arg-type]

    @property
    def receiver(self) -> object | None:
        owner = self._vrc_mic_sync_owner
        return owner.receiver if owner is not None else None

    @receiver.setter
    def receiver(self, receiver: object | None) -> None:
        self._get_vrc_mic_sync_owner().receiver = receiver

    @property
    def _vrc_mic_receiver_runtime(self) -> VrcMicReceiverRuntime | None:
        owner = self._vrc_mic_sync_owner
        return owner.runtime if owner is not None else None

    @_vrc_mic_receiver_runtime.setter
    def _vrc_mic_receiver_runtime(self, runtime: VrcMicReceiverRuntime | None) -> None:
        self._get_vrc_mic_sync_owner().runtime = runtime

    @property
    def _last_vrc_mic_sync_enabled(self) -> bool | None:
        owner = self._vrc_mic_sync_owner
        return owner.last_enabled if owner is not None else None

    @_last_vrc_mic_sync_enabled.setter
    def _last_vrc_mic_sync_enabled(self, enabled: bool | None) -> None:
        self._get_vrc_mic_sync_owner().last_enabled = enabled

    def _get_vrc_mic_sync_owner(self) -> VrcMicSyncOwner:
        owner = self._vrc_mic_sync_owner
        if owner is None:
            owner = VrcMicSyncOwner(
                state_provider=lambda: self.vrc_mic_state,
                gate_provider=lambda: self.vrc_mic_audio_gate,
                receiver_factory=self._create_vrc_osc_receiver_for_runtime,
                diagnostics_sink=self._vrc_mic_receiver_runtime_diagnostics_sink,
                error_sink=self._log_error,
                host=VRC_OSC_RECEIVER_HOST,
                port=VRC_OSC_RECEIVER_PORT,
            )
            self._vrc_mic_sync_owner = owner
        return owner

    def _get_vrc_mic_receiver_runtime(self) -> VrcMicReceiverRuntime | None:
        return self._get_vrc_mic_sync_owner().get_runtime()

    def _sync_vrc_mic_receiver_runtime_aliases(
        self,
        runtime: VrcMicReceiverRuntime | None = None,
    ) -> None:
        self._get_vrc_mic_sync_owner().sync_runtime_receiver(runtime)

    def _vrc_mic_receiver_runtime_diagnostics_sink(
        self,
        event: str,
        metadata: Mapping[str, object],
    ) -> None:
        self.log_detailed(
            f"[Lifecycle][VrcMicReceiverRuntime] event={event} metadata={dict(metadata)}",
            level=logging.WARNING,
        )

    async def _configure_vrc_mic_receiver(self, *, enabled: bool) -> None:
        await self._get_vrc_mic_sync_owner().configure(enabled=enabled)

    async def _stop_vrc_mic_receiver(self) -> None:
        await self._get_vrc_mic_sync_owner().stop()

    def _create_openrouter_pkce_client(self) -> OpenRouterPKCEClient:
        return OpenRouterPKCEClient(callback_origin="http://localhost:3000")

    def reopen_openrouter_pkce_authorization_url(self) -> bool:
        owner = self._openrouter_pkce_flow_owner
        return owner.reopen_authorization_url() if owner is not None else False

    def build_managed_openrouter_byok_target_settings(self) -> AppSettings | None:
        """Build a BYOK OpenRouter target settings draft from the current managed state.

        The controller owns this projection so UI renderers do not perform dynamic
        ``getattr(controller, "settings")`` shape reads. Returns ``None`` when the
        current settings are not a managed OpenRouter configuration.
        """
        current_settings = self.settings
        if current_settings is None:
            return None
        if current_settings.provider.llm != LLMProviderName.OPENROUTER:
            return None
        if current_settings.openrouter.selected_source != OpenRouterCredentialSource.MANAGED:
            return None

        openrouter_model = None
        selection_alias = current_settings.openrouter.selection_alias
        if selection_alias is not None:
            try:
                profile = profile_for_alias(selection_alias.value)
            except KeyError:
                profile = None
            if profile is not None:
                openrouter_model = profile.openrouter_model
        if openrouter_model is None:
            openrouter_model = current_settings.openrouter.llm_model.value

        alias_value = get_openrouter_selection_alias_for_model_and_source(
            openrouter_model,
            OpenRouterCredentialSource.BYOK.value,
        )
        if alias_value is None:
            return None

        target_settings = copy.deepcopy(current_settings)
        target_settings.provider.llm = LLMProviderName.OPENROUTER
        target_settings.openrouter.selection_alias = OpenRouterSelectionAlias(alias_value)
        target_settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
        target_settings.openrouter.llm_model = OpenRouterLLMModel(openrouter_model)
        target_settings.openrouter.provider_routing = OpenRouterProviderRouting.DEFAULT
        target_settings.translation.connection = TranslationConnection.OPENROUTER
        target_settings.translation.connection_history[target_settings.translation.model.value] = (
            TranslationConnection.OPENROUTER
        )
        return target_settings

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

        try:
            result = await self._get_openrouter_pkce_flow_owner().run_flow()
        except Exception:
            self._show_short_message("openrouter.pkce.failed")
            self._log_error("OpenRouter PKCE flow failed")
            self._maybe_show_founder_letter_after_pkce_failure(launch_source)
            return False

        try:
            verified = await self._get_provider_verifier().verify_api_key(
                "openrouter",
                result.api_key,
            )
        except Exception:
            verified = False
        if not verified:
            self._show_short_message("openrouter.pkce.failed")
            self._log_error("OpenRouter PKCE key verification failed")
            self._maybe_show_founder_letter_after_pkce_failure(launch_source)
            return False

        updated = copy.deepcopy(target_settings)
        updated.provider.llm = LLMProviderName.OPENROUTER
        updated.openrouter.selection_alias = OpenRouterSelectionAlias(profile.alias)
        updated.openrouter.selected_source = OpenRouterCredentialSource.BYOK
        updated.openrouter.llm_model = OpenRouterLLMModel(profile.openrouter_model)
        updated.api_key_verified.openrouter = True

        plan = self._build_provider_runtime_apply_plan(updated, force_rebuild_llm=True)
        secret_store = create_secret_store(self.settings.secrets, config_path=self.config_path)
        secret_store_port = create_sync_secret_store_adapter(secret_store)
        settings_repository = self._legacy_settings_patch_repository(
            base_settings=self.settings,
            committed_settings=updated,
            surface="openrouter_pkce",
            provider_verification_binding=self._provider_verification_binding(
                "openrouter",
                result.api_key,
                flow="openrouter_pkce",
                context_values={"launch_source": launch_source},
            ),
        )
        transaction = SecretSettingsTransaction(
            secret_store=secret_store_port,
            settings_repository=settings_repository,
        )
        runtime_apply_port = _ControllerProviderRuntimeApply(
            controller=self,
            settings=updated,
            plan=plan,
            surface="openrouter_pkce",
            operation="openrouter_pkce_runtime_apply",
        )

        commit_result = await transaction.set_provider_secret(
            SecretSetRequest(
                secret_key=OPENROUTER_BYOK_API_KEY_SECRET,
                secret_value=result.api_key,
                settings_values=self._get_settings_owner().legacy_snapshot_values(updated),
                expected_settings_revision=None,
                reason="openrouter_pkce",
                correlation_id=None,
            )
        )
        if commit_result.status != TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED:
            self._show_short_message("openrouter.pkce.failed")
            self._log_error("OpenRouter PKCE settings commit failed")
            self.last_settings_mutation_result = commit_result
            self._maybe_show_founder_letter_after_pkce_failure(launch_source)
            self._complete_canonical_mutation()
            return False

        runtime_result = await runtime_apply_port.apply_runtime(
            RuntimeApplyRequest(
                settings_values=self._get_settings_owner().legacy_snapshot_values(updated),
                reason="openrouter_pkce",
                correlation_id=None,
            )
        )
        if runtime_result.status == RUNTIME_APPLY_STATUS_APPLIED:
            self.last_settings_mutation_result = TransactionResult(
                status=TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
                message=runtime_result.message,
                diagnostics=runtime_result.diagnostics,
            )
            self._complete_canonical_mutation()
            return True

        self.last_settings_mutation_result = _runtime_apply_result_as_degraded_transaction(
            runtime_result
        )
        self._complete_canonical_mutation()
        return True

    def _maybe_show_founder_letter_after_pkce_failure(self, launch_source: str) -> None:
        if launch_source != "letter":
            return
        show_founder_letter_dialog = getattr(self.app, "show_founder_letter_dialog", None)
        if callable(show_founder_letter_dialog):
            with contextlib.suppress(Exception):
                show_founder_letter_dialog()

    def _save_settings(self) -> bool:
        assert self.settings is not None
        owns_mutation = self._canonical_mutation_depth == 0
        baseline = self._canonical_legacy_projection_snapshot or self.settings
        if owns_mutation:
            self._begin_canonical_mutation(legacy_snapshot=baseline)
        self._update_canonical_settings_from_legacy_delta(baseline, self.settings)
        try:
            self._persist_settings_at_controller_boundary(self.settings)
        except Exception as exc:
            self._rollback_canonical_mutation()
            self._log_error(f"Failed to save settings: {exc}")
            return False
        else:
            self._remember_canonical_legacy_projection(self.settings)
            if owns_mutation:
                self._complete_canonical_mutation()
            return True

    def persist_settings(self) -> None:
        """Persist current settings, propagating persistence errors to the caller."""
        assert self.settings is not None
        owns_mutation = self._canonical_mutation_depth == 0
        baseline = self._canonical_legacy_projection_snapshot or self.settings
        if owns_mutation:
            self._begin_canonical_mutation(legacy_snapshot=baseline)
        self._update_canonical_settings_from_legacy_delta(baseline, self.settings)
        try:
            self._persist_settings_at_controller_boundary(self.settings)
        except Exception:
            self._rollback_canonical_mutation()
            raise
        self._remember_canonical_legacy_projection(self.settings)
        if owns_mutation:
            self._complete_canonical_mutation()

    def _sync_ui_from_settings(self) -> None:
        settings = self.settings
        if settings is None:
            return

        self.app.set_dashboard_languages(
            source_language=settings.languages.source_language,
            target_language=settings.languages.target_language,
            peer_source_language=settings.languages.peer_source_language,
            peer_target_language=settings.languages.peer_target_language,
            peer_source_mode=settings.languages.peer_source_mode,
            recent_source_languages=settings.languages.recent_source_languages,
            recent_target_languages=settings.languages.recent_target_languages,
            peer_auto_detect_available=(
                settings.provider.peer_stt
                in {STTProviderName.SONIOX, STTProviderName.LOCAL_QWEN_GPU}
            ),
        )

        loaded = self._settings_projection().render(settings)
        if loaded is None:
            return
        if loaded:
            with contextlib.suppress(Exception):
                self.app.set_settings_overlay_calibration(self.overlay_calibration)

        self._refresh_overlay_peer_consumers()

    def _get_runtime_logging_owner(self) -> ApplicationRuntimeLoggingOwner:
        owner = self._runtime_logging_owner
        if owner is None:
            owner = ApplicationRuntimeLoggingOwner(
                presentation=self.app,
                service_factory=lambda: RuntimeLoggingService(
                    session_factory=lambda: SessionRuntimeLoggingService(
                        sinks=self.runtime_logging_sinks,
                        ui_handler_factory=RealtimeLogHandler,
                    ),
                    fallback_logger=logger,
                ),
                fallback_logger=logger,
                overlay_logging_mode_update=self._emit_overlay_runtime_logging_mode_update,
                overlay_logging_mode_update_available=lambda: (
                    self._current_overlay_bridge_for_direct_runtime_command() is not None
                ),
            )
            self._runtime_logging_owner = owner
        return owner

    @property
    def _runtime_logging(self) -> Any | None:
        owner = self._runtime_logging_owner
        return owner.installed_service if owner is not None else None

    @_runtime_logging.setter
    def _runtime_logging(self, service: Any | None) -> None:
        self._get_runtime_logging_owner().install_service(service)

    @property
    def runtime_logging(self) -> Any:
        return self._get_runtime_logging_owner().service

    @property
    def runtime_logging_mode(self) -> str:
        return self._get_runtime_logging_owner().mode

    def set_runtime_logging_mode(self, mode: SessionLoggingMode | str) -> None:
        def mode_changed(normalized_mode: str) -> None:
            runtime = self._overlay_runtime
            manager = runtime.process_manager if runtime is not None else None
            if manager is not None:
                set_logging_mode = getattr(manager, "set_logging_mode", None)
                if callable(set_logging_mode):
                    set_logging_mode(normalized_mode)
            self._schedule_overlay_runtime_logging_mode_update()

        self._get_runtime_logging_owner().set_mode(
            mode,
            detailed_enabled=self._schedule_audio_environment_snapshot,
            mode_changed=mode_changed,
        )

    def _schedule_audio_environment_snapshot(self) -> None:
        self._get_runtime_logging_owner().schedule_audio_environment_snapshot()

    async def _log_audio_environment_snapshot_async(self) -> None:
        await self._get_runtime_logging_owner().log_audio_environment_snapshot()

    async def _emit_overlay_runtime_logging_mode_update(self) -> None:
        bridge = self._current_overlay_bridge_for_direct_runtime_command()
        if bridge is None:
            return
        await bridge.broadcast_runtime_control(logging_mode=self.runtime_logging_mode)

    def _schedule_overlay_runtime_logging_mode_update(self) -> None:
        self._get_runtime_logging_owner().schedule_overlay_logging_mode_update()

    def log_basic(self, message: str, *, level: int = logging.INFO) -> None:
        self._get_runtime_logging_owner().emit_basic(message, level=level)

    def log_detailed(
        self,
        message: str,
        *,
        level: int = logging.INFO,
        exception: BaseException | None = None,
    ) -> bool:
        return self._get_runtime_logging_owner().emit_detailed(
            message,
            level=level,
            exception=exception,
        )

    def log_detailed_lazy(
        self,
        build_message: Callable[[], str],
        *,
        level: int = logging.INFO,
        exception: BaseException | None = None,
    ) -> bool:
        return self._get_runtime_logging_owner().emit_detailed_lazy(
            build_message,
            level=level,
            exception=exception,
        )

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
        return await self._get_provider_verifier().verify_qwen_llm_api_key(
            api_key,
            base_url=base_url,
            model=runtime_model,
            low_latency=FIXED_TRANSLATION_POLICY.fast_translation_enabled,
        )

    async def _verify_and_update_status(self) -> None:
        if self._shutdown_ingress_frozen:
            return
        request = self._build_provider_status_verification_request()
        if request is None:
            return
        result = await self._get_provider_status_verification_owner().verify(request)
        await self._apply_provider_status_verification_result(result)
