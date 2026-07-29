from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import inspect
import json
import logging
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
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
from puripuly_heart.app.ports.self_capture_admission import (
    SelfCaptureAdmissionEffect,
    SelfCaptureAdmissionEffectType,
    SelfCaptureAdmissionState,
)
from puripuly_heart.app.ports.settings_repository import CommittedSettingsRepositoryPort
from puripuly_heart.app.ports.ui_models import (
    GpuNoticeAction,
    OptionItem,
    OverlayPeerPresentationState,
)
from puripuly_heart.app.ports.ui_presentation import UIEventBridgePort, UiPresentationPort
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
from puripuly_heart.app.services.github_star_prompt_settings import (
    compose_github_star_prompt_owner,
)
from puripuly_heart.app.services.gpu_provider_recovery import (
    GpuProviderRecoveryDiagnostic,
)
from puripuly_heart.app.services.gpu_provider_recovery_application import (
    GpuProviderRecoveryApplicationOwner,
    GpuProviderRecoveryApplicationRequest,
)
from puripuly_heart.app.services.gpu_runtime_interaction import (
    GpuRuntimeInteractionOwner,
    GpuRuntimeInteractionState,
)
from puripuly_heart.app.services.local_asr_cpu_repair import (
    LocalASRCpuRepairEffect,
    LocalASRCpuRepairEffectType,
    LocalASRCpuRepairOwner,
    LocalASRCpuRepairRequest,
    LocalASRCpuRepairRuntimeState,
)
from puripuly_heart.app.services.local_asr_diagnostics import LocalASRDiagnosticsOwner
from puripuly_heart.app.services.local_asr_gpu_provisioning import (
    LocalASRGpuProvisioningDiagnostic,
)
from puripuly_heart.app.services.local_asr_readiness import (
    LocalASRReadinessEffect,
    LocalASRReadinessEffectType,
    LocalASRReadinessOwner,
    LocalASRReadinessState,
)
from puripuly_heart.app.services.local_asr_selection import (
    LOCAL_CPU_PROVIDERS,
    local_asr_status_for_provider,
    required_local_asr_model_ids,
    resolve_local_asr_selection,
)
from puripuly_heart.app.services.managed_auth import ManagedAuthOwner
from puripuly_heart.app.services.managed_usage import (
    ManagedUsageMetadataResult,
    ManagedUsageOwner,
    ManagedUsageState,
    ManagedUsageViewState,
)
from puripuly_heart.app.services.manual_local_asr_fallback import (
    ManualLocalASRFallbackOwner,
    ManualLocalASRFallbackPlan,
    ManualLocalASRFallbackState,
)
from puripuly_heart.app.services.manual_typing import ManualTypingOwner
from puripuly_heart.app.services.microphone_test import (
    MicrophoneTestSelfCaptureState,
    MicrophoneTestSessionOwner,
    MicrophoneTestSessionRequest,
)
from puripuly_heart.app.services.openrouter_pkce_flow import OpenRouterPkceFlowOwner
from puripuly_heart.app.services.overlay_application import (
    OverlayApplicationOwner,
    OverlayApplicationState,
)
from puripuly_heart.app.services.overlay_calibration import OverlayCalibrationOwner
from puripuly_heart.app.services.peer_application import (
    PeerApplicationOwner,
    PeerApplicationState,
)
from puripuly_heart.app.services.provider_credential_verification import (
    ProviderCredentialVerificationInteractionOwner,
)
from puripuly_heart.app.services.provider_runtime_apply import (
    LlmProviderRebuildContext,
    LlmProviderRebuildOwner,
    NoopRuntimeApply,
    OverlayOscOutputRuntimeApplyAdapter,
    ProviderRuntimeApplyAdapter,
    ProviderRuntimeApplyPlan,
    ProviderRuntimeOwner,
    ProviderRuntimeState,
    SettingsRuntimeState,
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
from puripuly_heart.app.services.provider_settings import (
    ProviderApplicationOwner,
    ProviderSettingsOwner,
    provider_verification_context,
)
from puripuly_heart.app.services.provider_verification_binding import (
    ProviderVerificationBindingOwner,
)
from puripuly_heart.app.services.secret_settings_transaction import (
    SecretSetRequest,
    SecretSettingsTransaction,
)
from puripuly_heart.app.services.settings_mutation import (
    OverlayOscOutputSettingsMutation,
    SettingsMutationService,
    SttLanguageAudioSettingsMutation,
    UiPromptClipboardStateSettingsMutation,
)
from puripuly_heart.app.services.settings_mutation_legacy import (
    _apply_settings_path_patch,
    build_overlay_osc_output_settings_path_patch,
    settings_path_mutation_validator_for_command,
)
from puripuly_heart.app.services.settings_projection import (
    SettingsProjectionOwner,
    SettingsViewSettingsChange,
)
from puripuly_heart.app.services.translation_enable import TranslationEnableOwner
from puripuly_heart.app.services.vrc_mic_sync import VrcMicSyncOwner
from puripuly_heart.app.wiring import (
    LocalASRProviderRuntimeFactory,
    ManagedSTTProviderFactory,
    build_managed_identity_state_port,
    build_openrouter_credential_runtime_config,
    build_openrouter_release_runtime_config,
    build_peer_capture_session_config,
    build_peer_stt_provider_request,
    build_peer_stt_provider_signature_from_vnext,
    build_peer_stt_runtime_signature,
    build_self_capture_session_config,
    build_self_capture_vad_signature,
    build_self_stt_provider_request,
    build_self_stt_provider_signature,
    build_self_stt_runtime_signature,
    compose_peer_capture_session_owner,
    compose_self_capture_session_owner,
    copy_stable_secrets_to_vnext_namespace,
    create_llm_provider,
    create_local_asr_provisioning_owner,
    create_microphone_test_capture_adapter,
    create_peer_capture_admission_adapter,
    create_peer_capture_audio_loop_adapter,
    create_peer_capture_source_adapter,
    create_peer_capture_target_resolver_adapter,
    create_peer_capture_vad_adapter,
    create_peer_capture_vad_sink_adapter,
    create_provider_verifier,
    create_secret_store,
    create_self_capture_admission_adapter,
    create_self_capture_audio_loop_adapter,
    create_self_capture_source_adapter,
    create_self_capture_vad_adapter,
    create_self_capture_vad_sink_adapter,
    create_sync_secret_store_adapter,
    resolve_overlay_config,
)
from puripuly_heart.app.wiring_composition import (
    create_gpu_provider_recovery_application_owner,
    create_gpu_runtime_interaction_owner,
    create_local_asr_cpu_repair_owner,
    create_local_asr_diagnostics_owner,
    create_local_asr_readiness_owner,
    create_manual_typing_owner,
    create_provider_credential_verification_interaction_owner,
    create_vrchat_osc_presence_probe_owner,
)
from puripuly_heart.app.wiring_managed_auth_factory import (
    ManagedAuthRuntimeAdapter,
    ManagedTranslationRuntimeAdapter,
)
from puripuly_heart.config.llm_profiles import profile_for_alias
from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.config.paths import user_config_dir
from puripuly_heart.config.process_capture_resolution import (
    ProcessCaptureResolver,
)
from puripuly_heart.config.resolved import (
    ResolvedDesktopAudioCaptureTarget,
    ResolvedOverlayConfig,
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
    STTProviderName,
    TranslationConnection,
    build_managed_openrouter_byok_target_settings,
    normalize_owned_referral_id,
    with_telemetry_consent,
)
from puripuly_heart.config.settings_vnext.schema import (
    AppSettingsVNext,
    CaptureTargetIntent,
    ProcessCaptureTargetIntent,
)
from puripuly_heart.config.vad_defaults import DEFAULT_STABLE_VAD_HANGOVER_MS
from puripuly_heart.core.audio.gate import VrcMicAudioGate
from puripuly_heart.core.audio.process_identity import PsutilCurrentUserProcessSnapshots
from puripuly_heart.core.audio.source import (
    AudioSource,
    SoundDeviceAudioSource,
    determine_self_mic_capture_channels,
    observe_microphone_test_route,
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
from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimePort,
    LocalASRProviderRuntimeSnapshot,
    ProviderRuntimeBuildRequest,
    ProviderRuntimeChannelSnapshot,
)
from puripuly_heart.core.local_asr_provisioning import (
    LocalASRProvisioningDiagnostic,
    LocalASRProvisioningPort,
    LocalASRProvisioningSnapshot,
)
from puripuly_heart.core.local_gpu_assets import local_gpu_model_path
from puripuly_heart.core.managed_openrouter_broker_client import (
    HttpManagedOpenRouterBrokerClient,
)
from puripuly_heart.core.managed_openrouter_release import (
    ManagedOpenRouterReleaseService,
    UnavailableManagedOpenRouterReleaseClient,
)
from puripuly_heart.core.messages import (
    RUNTIME_APPLY_STATUS_APPLIED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    TransactionResult,
)
from puripuly_heart.core.openrouter_credentials import (
    OPENROUTER_BYOK_API_KEY_SECRET,
    resolve_openrouter_credentials,
)
from puripuly_heart.core.openrouter_handoff import should_auto_show_founder_letter
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
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.process import OverlayProcessRunner
from puripuly_heart.core.peer_capture import (
    PeerCaptureDiagnostic,
    PeerCaptureSessionSnapshot,
)
from puripuly_heart.core.runtime.desktop_overlay_bounds import (
    DesktopOverlayBoundsOwner,
    is_finite_non_bool_number,
)
from puripuly_heart.core.runtime.gpu_asr import GpuASRChannel
from puripuly_heart.core.runtime.logging import RuntimeLoggingService
from puripuly_heart.core.runtime.overlay import OverlayRuntimeHandle
from puripuly_heart.core.runtime.peer_channel import (
    PeerCaptureSessionOwner,
)
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
    SelfCaptureDiagnostic,
    SelfCaptureProviderStatus,
    SelfCaptureSessionConfig,
    SelfCaptureSessionSnapshot,
    SelfCaptureSessionState,
)
from puripuly_heart.core.stt.controller import FinalTranscriptSuppressedNotification
from puripuly_heart.core.telemetry import (
    TranslationSuccessTelemetryClientPort,
    TranslationSuccessTelemetryResult,
    TranslationSuccessTelemetryService,
)
from puripuly_heart.core.translation_policy import FIXED_TRANSLATION_POLICY

logger = logging.getLogger(__name__)

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
_MANAGED_OPENROUTER_CONNECTIONS = frozenset(
    {
        TranslationConnection.MANAGED,
        TranslationConnection.MANAGED_CHINA,
    }
)
_MICROPHONE_TEST_LEVEL_INTERVAL_S = 1.0
LOCAL_QWEN_HALLUCINATION_GUIDANCE_TRIGGER_COUNT = 2


def _canonical_json_signature(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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
    provider_settings_owner: ProviderSettingsOwner | None = field(default=None, repr=False)
    manual_local_asr_fallback_owner: ManualLocalASRFallbackOwner = field(
        default_factory=ManualLocalASRFallbackOwner,
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
    _provider_runtime_owner: ProviderRuntimeOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _provider_application_owner: ProviderApplicationOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _llm_provider_rebuild_owner: LlmProviderRebuildOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _peer_application_owner: PeerApplicationOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
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
    _vad: object | None = None
    _stt_desired: bool = False
    _stt_restart_requested: bool = False
    _stt_force_immediate: bool = False
    _stt_activation_generation: int = field(init=False, default=0)
    _stt_activation_starting: bool = field(init=False, default=False)
    _stt_activation_failed: bool = field(init=False, default=False)
    _last_stt_runtime_signature: tuple[object, ...] | None = None
    _last_self_stt_runtime_signature: tuple[object, ...] | None = None
    _last_self_stt_provider_signature: tuple[object, ...] | None = None
    _last_llm_provider_signature: tuple[object, ...] | None = None
    _superseded_local_asr_settings_ids: set[int] = field(
        init=False,
        default_factory=set,
        repr=False,
    )
    _process_idle_preparation_scheduled: bool = field(init=False, default=False)
    _settings_projection_owner: SettingsProjectionOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _ui_event_bridge: UIEventBridgePort | None = None
    _clipboard_auto_translation_owner: ClipboardAutoTranslationOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _local_asr_cpu_repair_owner: LocalASRCpuRepairOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _local_asr_readiness_owner: LocalASRReadinessOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _gpu_runtime_interaction_owner: GpuRuntimeInteractionOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _gpu_provider_recovery_owner: GpuProviderRecoveryApplicationOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _local_asr_diagnostics_owner: LocalASRDiagnosticsOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    # Overlay runtime internals are owned by OverlayRuntimeHandle.
    _overlay_application_owner: OverlayApplicationOwner | None = field(
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
    _managed_usage_owner: ManagedUsageOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _provider_credential_verification_owner: (
        ProviderCredentialVerificationInteractionOwner | None
    ) = field(init=False, default=None, repr=False)
    _managed_auth_runtime_adapter: ManagedAuthRuntimeAdapter | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _managed_auth_owner: ManagedAuthOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _managed_translation_runtime_adapter: ManagedTranslationRuntimeAdapter | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _translation_enable_owner: TranslationEnableOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
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

    @property
    def settings(self) -> AppSettings | None:
        return self._get_settings_owner().current

    @settings.setter
    def settings(self, settings: AppSettings | None) -> None:
        self._get_settings_owner().current = settings

    def _legacy_settings_patch_repository(
        self,
        *,
        committed_settings: AppSettings,
        base_settings: AppSettings | None = None,
        surface: str = "translation_provider",
        provider_verification_binding: ProviderVerificationBinding | None = None,
    ) -> CommittedSettingsRepositoryPort[AppSettings]:
        return self._get_settings_owner().create_legacy_patch_repository(
            committed_settings=committed_settings,
            base_settings=base_settings,
            surface=surface,
            provider_verification_binding=provider_verification_binding,
            save_failure_sink=self._log_error,
        )

    @property
    def vnext_settings(self) -> AppSettingsVNext | None:
        return self._get_settings_owner().canonical

    @vnext_settings.setter
    def vnext_settings(self, settings: AppSettingsVNext | None) -> None:
        self._get_settings_owner().canonical = settings

    @property
    def managed_auth_pending(self) -> bool:
        owner = self._managed_auth_owner
        return owner.pending if owner is not None else False

    @property
    def last_discord_managed_auth_referral_bonus_applied(self) -> bool:
        owner = self._managed_auth_owner
        return owner.last_referral_bonus_applied if owner is not None else False

    @property
    def _overlay_runtime(self) -> OverlayRuntimeHandle | None:
        owner = self._overlay_application_owner
        return owner.runtime if owner is not None else None

    @_overlay_runtime.setter
    def _overlay_runtime(self, runtime: OverlayRuntimeHandle | None) -> None:
        self._get_overlay_application_owner().runtime = runtime

    @property
    def _active_overlay_target(self) -> str | None:
        owner = self._overlay_application_owner
        return owner.active_target if owner is not None else None

    @_active_overlay_target.setter
    def _active_overlay_target(self, target: str | None) -> None:
        self._get_overlay_application_owner().active_target = target

    @property
    def overlay_state(self) -> str:
        owner = self._overlay_application_owner
        return owner.state if owner is not None else "off"

    @overlay_state.setter
    def overlay_state(self, state: str) -> None:
        self._get_overlay_application_owner().state = state

    @property
    def failure_reason(self) -> str | None:
        owner = self._overlay_application_owner
        return owner.failure_reason if owner is not None else None

    @failure_reason.setter
    def failure_reason(self, reason: str | None) -> None:
        self._get_overlay_application_owner().failure_reason = reason

    @property
    def auto_restart_scheduled(self) -> bool:
        owner = self._overlay_application_owner
        return owner.auto_restart_scheduled if owner is not None else False

    @auto_restart_scheduled.setter
    def auto_restart_scheduled(self, scheduled: bool) -> None:
        self._get_overlay_application_owner().auto_restart_scheduled = scheduled

    def _report_overlay_application_state(
        self,
        state: str,
        failure_reason: str | None,
    ) -> None:
        bridge = self._ui_event_bridge
        if bridge is not None:
            bridge.report_overlay_state(state, failure_reason=failure_reason)

    def _overlay_application_state(
        self,
        settings: AppSettings | None = None,
    ) -> OverlayApplicationState:
        resolved = settings or self.settings
        return OverlayApplicationState(
            settings_available=resolved is not None,
            overlay_intent_enabled=bool(resolved is not None and resolved.ui.overlay_enabled),
            configured_target=(
                resolved.overlay.target if resolved is not None else OVERLAY_TARGET_STEAMVR
            ),
            locale=resolved.ui.locale if resolved is not None else "en",
        )

    def _get_overlay_application_owner(self) -> OverlayApplicationOwner:
        owner = self._overlay_application_owner
        if owner is None:
            owner = OverlayApplicationOwner(
                state_provider=self._overlay_application_state,
                config_provider=lambda: resolve_overlay_config(cast(AppSettings, self.settings)),
                overlay_intent_sink=lambda enabled: setattr(
                    cast(AppSettings, self.settings).ui,
                    "overlay_enabled",
                    enabled,
                ),
                hub_provider=lambda: self.hub,
                peer_snapshot_provider=lambda: self._get_peer_application_owner().snapshot(),
                disable_peer_intent=lambda: self._get_peer_application_owner().disable_for_overlay(),
                sync_peer_effective=lambda: self._get_peer_application_owner().sync_effective_flags(),
                refresh_peer_dependencies=self._refresh_overlay_runtime_dependencies,
                presentation_sink=self.app.refresh_overlay_peer_contract,
                state_sink=self._report_overlay_application_state,
                fallback_notice_sink=(self.app.set_dashboard_overlay_session_fallback_notice),
                cancel_bounds_persistence=self._cancel_desktop_bounds_persistence,
                clear_bounds_suppressed=lambda: self._get_desktop_overlay_bounds_owner().clear_suppressed(),
                calibration_provider=lambda: self.overlay_calibration.copy(),
                logging_mode_provider=lambda: self.runtime_logging_mode,
                log_dir_provider=lambda: str(user_config_dir()),
                desktop_controls_factory=(
                    self._build_initial_desktop_runtime_controls_from_resolved_config
                ),
                interaction_mode_sink=self._set_desktop_overlay_interaction_mode,
                bounds_control_sink=self._track_desktop_apply_window_bounds_control,
                renderer_event_consumer=lambda queue, instance_id: (
                    self._consume_desktop_renderer_events(
                        queue,
                        overlay_instance_id=instance_id,
                    )
                ),
                edit_interaction_mode=DESKTOP_INTERACTION_MODE_EDIT,
                clock=self.clock,
                log_basic=lambda message, level: self.log_basic(
                    message,
                    level=level,
                ),
                log_detailed=lambda message, level, exception: self.log_detailed(
                    message,
                    level=level,
                    exception=exception,
                ),
            )
            self._overlay_application_owner = owner
        return owner

    @property
    def desktop_overlay_captions_locked(self) -> bool:
        return self.desktop_overlay_interaction_mode == DESKTOP_INTERACTION_MODE_PASS_THROUGH

    @property
    def _peer_runtime(self) -> PeerCaptureSessionOwner | None:
        owner = self._peer_application_owner
        return owner.runtime if owner is not None else None

    @_peer_runtime.setter
    def _peer_runtime(self, runtime: PeerCaptureSessionOwner | None) -> None:
        self._get_peer_application_owner().bind_runtime(runtime)

    @property
    def _last_peer_stt_runtime_signature(self) -> tuple[object, ...] | None:
        return self._get_peer_application_owner().last_runtime_signature

    @_last_peer_stt_runtime_signature.setter
    def _last_peer_stt_runtime_signature(self, value: tuple[object, ...] | None) -> None:
        self._get_peer_application_owner().last_runtime_signature = value

    @property
    def _last_peer_stt_provider_signature(self) -> tuple[object, ...] | None:
        return self._get_peer_application_owner().last_provider_signature

    @_last_peer_stt_provider_signature.setter
    def _last_peer_stt_provider_signature(self, value: tuple[object, ...] | None) -> None:
        self._get_peer_application_owner().last_provider_signature = value

    @property
    def _last_peer_translation_enabled(self) -> bool | None:
        return self._get_peer_application_owner().last_intent_enabled

    @_last_peer_translation_enabled.setter
    def _last_peer_translation_enabled(self, value: bool | None) -> None:
        self._get_peer_application_owner().last_intent_enabled = value

    @property
    def _last_peer_translation_activation_requested(self) -> bool | None:
        return self._get_peer_application_owner().last_activation_requested

    @_last_peer_translation_activation_requested.setter
    def _last_peer_translation_activation_requested(self, value: bool | None) -> None:
        self._get_peer_application_owner().last_activation_requested = value

    @property
    def _peer_activation_generation(self) -> int:
        return self._get_peer_application_owner().activation_generation

    @_peer_activation_generation.setter
    def _peer_activation_generation(self, value: int) -> None:
        self._get_peer_application_owner().activation_generation = value

    @property
    def _peer_activation_starting(self) -> bool:
        return self._get_peer_application_owner().activation_starting

    @_peer_activation_starting.setter
    def _peer_activation_starting(self, value: bool) -> None:
        self._get_peer_application_owner().activation_starting = value

    @property
    def _peer_asr_model_loading(self) -> bool:
        return self._get_peer_application_owner().model_loading

    @_peer_asr_model_loading.setter
    def _peer_asr_model_loading(self, value: bool) -> None:
        self._get_peer_application_owner().model_loading = value

    @property
    def _peer_process_warning_reason(self) -> str | None:
        return self._get_peer_application_owner().process_warning_reason

    @_peer_process_warning_reason.setter
    def _peer_process_warning_reason(self, value: str | None) -> None:
        self._get_peer_application_owner().process_warning_reason = value

    def _peer_application_state(
        self,
        settings: AppSettings | None = None,
    ) -> PeerApplicationState:
        resolved_settings = settings or self.settings
        hub = self.hub
        activation_requested = bool(
            resolved_settings is not None
            and PeerApplicationOwner.activation_requested(
                intent_enabled=resolved_settings.ui.peer_translation_enabled,
                eula_accepted=resolved_settings.ui.peer_translation_eula_accepted,
            )
        )
        return PeerApplicationState(
            settings_available=resolved_settings is not None,
            peer_intent_enabled=bool(
                resolved_settings is not None and resolved_settings.ui.peer_translation_enabled
            ),
            eula_accepted=bool(
                resolved_settings is not None
                and resolved_settings.ui.peer_translation_eula_accepted
            ),
            overlay_intent_enabled=bool(
                resolved_settings is not None and resolved_settings.ui.overlay_enabled
            ),
            peer_provider_id=(
                resolved_settings.provider.peer_stt.value if resolved_settings is not None else None
            ),
            runtime_available=hub is not None,
            peer_provider_available=bool(
                activation_requested and hub is not None and self._hub_has_stt_provider("peer")
            ),
            overlay_state=self.overlay_state,
            overlay_command_available=(
                self._current_overlay_bridge_for_direct_runtime_command() is not None
            ),
            ingress_frozen=self._shutdown_ingress_frozen,
        )

    def _apply_peer_effective_flags(
        self,
        peer_translation_enabled: bool,
        integrated_context_enabled: bool,
    ) -> None:
        if self.hub is None:
            return
        self.hub.peer_translation_enabled = peer_translation_enabled
        self.hub.integrated_context_enabled = integrated_context_enabled

    def _get_peer_application_owner(self) -> PeerApplicationOwner:
        owner = self._peer_application_owner
        if owner is None:
            owner = PeerApplicationOwner(
                state_provider=self._peer_application_state,
                config_factory=lambda: build_peer_capture_session_config(
                    cast(AppSettings, self.settings),
                    canonical_settings=self._canonical_vnext_settings_for(
                        cast(AppSettings, self.settings)
                    ),
                ),
                peer_intent_sink=lambda enabled: setattr(
                    cast(AppSettings, self.settings).ui,
                    "peer_translation_enabled",
                    enabled,
                ),
                overlay_intent_sink=lambda enabled: setattr(
                    cast(AppSettings, self.settings).ui,
                    "overlay_enabled",
                    enabled,
                ),
                persist_manual_fallback=lambda: self._persist_current_manual_local_asr_fallback(
                    channel="peer"
                ),
                ensure_local_ready=lambda generation: self._ensure_peer_local_stt_ready(
                    activation_generation=generation
                ),
                clear_cpu_pending=self._reset_local_stt_pending_peer_enable_after_install,
                clear_gpu_pending=lambda: self._get_gpu_runtime_interaction_owner().clear_pending(
                    "peer"
                ),
                clear_switched_pending=self._clear_local_stt_pending_enable_if_provider_switched_away,
                sync_local_notice=self._sync_local_stt_notice,
                presentation_changed=self._refresh_overlay_peer_consumers,
                begin_overlay_start=self._begin_overlay_start,
                effective_sink=self._apply_peer_effective_flags,
                disclosure_sink=self._enqueue_peer_translation_disclosure,
                superseded_sink=lambda: self._superseded_local_asr_settings_ids.add(
                    id(self.settings)
                ),
                log_basic=self.log_basic,
                log_detailed=self.log_detailed,
                log_failure=lambda message: self.log_basic(
                    message,
                    level=logging.ERROR,
                ),
            )
            self._peer_application_owner = owner
        return owner

    def _effective_peer_translation_enabled_for(self, settings: AppSettings) -> bool:
        return self._get_peer_application_owner().effective_enabled(
            self._peer_application_state(settings)
        )

    def _peer_translation_eula_accepted_for(self, settings: AppSettings) -> bool:
        return bool(settings.ui.peer_translation_eula_accepted)

    def _peer_translation_activation_requested_for(self, settings: AppSettings) -> bool:
        return self._get_peer_application_owner().activation_requested(
            intent_enabled=settings.ui.peer_translation_enabled,
            eula_accepted=settings.ui.peer_translation_eula_accepted,
        )

    def _effective_peer_overlay_enabled_for(self, settings: AppSettings) -> bool:
        _ = settings
        return self.overlay_state == "connected"

    def _effective_integrated_context_enabled_for(self, settings: AppSettings) -> bool:
        return self._effective_peer_translation_enabled_for(settings)

    def _sync_effective_hub_flags(self, settings: AppSettings | None = None) -> None:
        resolved_settings = settings or self.settings
        if resolved_settings is None:
            return
        state = self._peer_application_state(resolved_settings)
        self._get_peer_application_owner().sync_effective_flags(state)

    def get_event_language_codes(self) -> tuple[str | None, str | None]:
        if self.settings is None:
            return None, None
        return self.settings.languages.source_language, self.settings.languages.target_language

    def overlay_peer_presentation_state(self) -> OverlayPeerPresentationState | None:
        return self._get_overlay_application_owner().presentation_state()

    def _refresh_overlay_peer_consumers(self) -> None:
        self._get_overlay_application_owner().publish_presentation()

    async def _refresh_overlay_runtime_dependencies(
        self,
        *,
        peer_stop_mode: str = "retain",
    ) -> None:
        if peer_stop_mode == "release":
            await self._refresh_peer_stt_runtime(stop_mode="release")
        else:
            await self._refresh_peer_stt_runtime()
        self._sync_effective_hub_flags()
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
        self._get_settings_owner().authoritative = True
        self._get_settings_owner().remember_projection(self.settings)
        provisioning = self._get_local_asr_provisioning_owner()
        await provisioning.inspect_cpu()
        await provisioning.inspect_gpu(
            explicit_intent=self._gpu_runtime_interaction_state().selected_provider_requires_model,
        )
        loaded_settings = self.settings
        fallback_plan = self.manual_local_asr_fallback_owner.plan(
            self._manual_local_asr_fallback_state(loaded_settings)
        )
        fallback_channels = fallback_plan.fallback_channels
        installation_fallback = fallback_plan.installation_fallback
        if fallback_plan.changed:
            normalized_settings = self._settings_with_manual_local_asr_fallback_plan(
                loaded_settings,
                fallback_plan,
            )
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
        owner = self._get_managed_usage_owner()
        await owner.refresh(auto_show_founder_letter=True)
        return owner.is_exhausted

    async def prepare_runtime_after_launch(self) -> None:
        self._schedule_process_discovery_idle_preparation()
        self._schedule_vrchat_osc_presence_probe(force=True)
        await self.preload_saved_gpu_device_discovery()

    async def preload_saved_gpu_device_discovery(self) -> tuple[GpuWorkerDevice, ...]:
        return await self._get_gpu_runtime_interaction_owner().preload_saved_device_discovery()

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

    async def handle_gpu_notice_action(self, action: GpuNoticeAction) -> None:
        await self._get_gpu_runtime_interaction_owner().handle_notice_action(action)

    async def ensure_gpu_device_discovery(
        self,
        *,
        force: bool = False,
        origin: str = "settings",
    ) -> tuple[GpuWorkerDevice, ...]:
        return await self._get_gpu_runtime_interaction_owner().ensure_device_discovery(
            force=force,
            origin=origin,
        )

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
        self._get_gpu_runtime_interaction_owner().observe_provisioning(snapshot)
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

    async def _validate_gpu_activation(self) -> bool:
        return await self._get_gpu_runtime_interaction_owner().validate_activation()

    def _gpu_runtime_interaction_state(self) -> GpuRuntimeInteractionState:
        settings = self.settings
        return GpuRuntimeInteractionState(
            settings_available=settings is not None,
            selected_provider_requires_model=bool(
                settings is not None
                and (
                    settings.provider.stt == STTProviderName.LOCAL_QWEN_GPU
                    or settings.provider.peer_stt == STTProviderName.LOCAL_QWEN_GPU
                )
            ),
            locale=settings.ui.locale if settings is not None else None,
            device_id=settings.stt.gpu_device_id if settings is not None else "auto",
        )

    def _on_local_asr_gpu_provisioning_diagnostic(
        self,
        diagnostic: LocalASRGpuProvisioningDiagnostic,
    ) -> None:
        self.log_detailed(
            "[GPU ASR] model_install failure=unexpected",
            level=logging.WARNING,
            exception=diagnostic.exception,
        )

    def _get_gpu_runtime_interaction_owner(self) -> GpuRuntimeInteractionOwner:
        owner = self._gpu_runtime_interaction_owner
        if owner is None:

            def runtime_provider() -> LocalASRProviderRuntimePort:
                runtime = self._hub_local_asr_provider_runtime()
                if runtime is None:
                    raise RuntimeError("local ASR provider runtime is unavailable")
                return runtime

            owner = create_gpu_runtime_interaction_owner(
                runtime_provider=runtime_provider,
                provisioning_provider=self._get_local_asr_provisioning_owner,
                state_provider=self._gpu_runtime_interaction_state,
                presentation_sink=lambda presentation: self.app.set_dashboard_gpu_state(
                    devices=presentation.devices,
                    state=presentation.state,
                    progress_percent=presentation.progress_percent,
                    notice=presentation.notice,
                    publish_notice=presentation.publish_notice,
                ),
                detailed_log_sink=self.log_detailed,
                retry_activation=self.retry_gpu_activation,
                install_diagnostic_sink=self._on_local_asr_gpu_provisioning_diagnostic,
            )
            self._gpu_runtime_interaction_owner = owner
        return owner

    def _get_local_asr_diagnostics_owner(self) -> LocalASRDiagnosticsOwner:
        owner = self._local_asr_diagnostics_owner
        if owner is None:
            owner = create_local_asr_diagnostics_owner(
                basic_log_sink=lambda message, level: self.log_basic(
                    message,
                    level=level,
                ),
                detailed_log_sink=self.log_detailed,
                gpu_effect_sink=self._get_gpu_runtime_interaction_owner().apply_diagnostics_effect,
                gpu_discovery_origin_provider=lambda: (
                    self._get_gpu_runtime_interaction_owner().snapshot.discovery_origin
                ),
                gpu_provider_id=STTProviderName.LOCAL_QWEN_GPU.value,
            )
            self._local_asr_diagnostics_owner = owner
        return owner

    async def install_selected_gpu_model_if_needed(self) -> bool:
        return await self._get_gpu_runtime_interaction_owner().install_selected_model_if_needed()

    async def install_or_repair_gpu_model(self, *, origin: str = "manual") -> None:
        await self._get_gpu_runtime_interaction_owner().install_or_repair(origin=origin)

    async def retry_gpu_activation(self) -> None:
        if self.settings is None:
            return
        await self._get_gpu_provider_recovery_owner().recover(
            lambda: self._gpu_provider_recovery_request(
                self.settings,
                reason="manual_retry",
                plan=None,
            )
        )

    def _create_ui_event_bridge(
        self,
        *,
        runtime_logging,
    ) -> UIEventBridgePort:  # noqa: ANN001
        assert self.hub is not None
        return self.app.create_ui_event_bridge(
            event_queue=self.hub.ui_events,
            runtime_logging=runtime_logging,
        )

    def _start_ui_event_bridge_task(self, bridge: UIEventBridgePort) -> None:
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

    def _managed_openrouter_can_attempt_translation(self) -> bool:
        return bool(
            self.settings is not None
            and self.settings.provider.llm == LLMProviderName.OPENROUTER
            and self.settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
            and self.hub is not None
            and self.hub.llm is not None
        )

    def _get_managed_auth_runtime_adapter(self) -> ManagedAuthRuntimeAdapter:
        adapter = self._managed_auth_runtime_adapter
        if adapter is None:
            adapter = ManagedAuthRuntimeAdapter(
                config_path=self.config_path,
                secret_store_factory=create_secret_store,
                settings_provider=lambda: self.settings,
                settings_sink=lambda settings: setattr(self, "settings", settings),
                release_service_provider=lambda: self._managed_openrouter_release_service,
                persistence_callback_factory=self._managed_identity_persistence_callback,
                settings_repository_factory=lambda base, committed, surface: (
                    self._legacy_settings_patch_repository(
                        base_settings=base,
                        committed_settings=committed,
                        surface=surface,
                    )
                ),
                settings_owner_complete=self._get_settings_owner().complete,
                runtime_presence_provider=lambda: (
                    self.hub is not None,
                    self.hub is not None and self.hub.llm is not None,
                ),
                ingress_provider=lambda: self._shutdown_ingress_frozen,
            )
            self._managed_auth_runtime_adapter = adapter
        return adapter

    def _get_managed_auth_owner(self) -> ManagedAuthOwner:
        owner = self._managed_auth_owner
        if owner is None:
            adapter = self._get_managed_auth_runtime_adapter()
            owner = ManagedAuthOwner(
                state_provider=adapter.state,
                pending_sink=self.app.set_dashboard_managed_auth_pending,
                qq_executor=adapter.execute_qq,
                discord_executor=adapter.execute_discord,
                runtime_ensurer=self._ensure_managed_auth_runtime,
                usage_view_sink=self._apply_managed_auth_usage_view,
                usage_refresh_sink=self._get_managed_usage_owner().schedule_usage_refresh,
                message_sink=lambda key, values: self._show_short_message(
                    key,
                    **dict(values),
                ),
                result_sink=lambda result: setattr(
                    self,
                    "last_settings_mutation_result",
                    result,
                ),
                log_sink=lambda message: self.log_basic(
                    message,
                    level=logging.ERROR,
                ),
            )
            self._managed_auth_owner = owner
        return owner

    async def _ensure_managed_auth_runtime(self, mode: str) -> bool:
        if self.hub is None:
            return False
        if mode == "always" or (mode == "if_missing" and self.hub.llm is None):
            await self._get_llm_provider_rebuild_owner().rebuild()
        return self.hub.llm is not None

    def _apply_managed_auth_usage_view(
        self,
        referral_id: str | None,
        pass_status,
    ) -> None:
        owner = self._get_managed_usage_owner()
        owner.set_view_state(
            visible=True,
            remaining_percent=None,
            referral_id=referral_id or owner.current_referral_id,
            pass_status=pass_status,
        )

    def _set_managed_trial_pending_auth(self, pending: bool) -> None:
        self._get_managed_auth_owner().set_pending(pending)

    def clear_managed_auth_pending_state(self) -> None:
        self._get_managed_auth_owner().clear_pending()

    def _get_managed_translation_runtime_adapter(
        self,
    ) -> ManagedTranslationRuntimeAdapter:
        adapter = self._managed_translation_runtime_adapter
        if adapter is None:
            adapter = ManagedTranslationRuntimeAdapter(
                auth=self._get_managed_auth_runtime_adapter(),
                settings_provider=lambda: self.settings,
                release_service_provider=lambda: self._managed_openrouter_release_service,
                runtime_snapshot_provider=lambda: (
                    self.hub is not None,
                    self.hub.translation_enabled if self.hub is not None else False,
                    self.hub.llm if self.hub is not None else None,
                ),
                ingress_provider=lambda: self._shutdown_ingress_frozen,
                founder_dialog=self.app.show_founder_letter_dialog,
                persist_settings=self._save_settings,
            )
            self._managed_translation_runtime_adapter = adapter
        return adapter

    def _get_translation_enable_owner(self) -> TranslationEnableOwner:
        owner = self._translation_enable_owner
        if owner is None:
            adapter = self._get_managed_translation_runtime_adapter()
            owner = TranslationEnableOwner(
                state_provider=adapter.state,
                managed_prepare=adapter.prepare,
                founder_route=self._get_managed_usage_owner().should_route_to_founder_letter,
                pending_sink=self._set_managed_trial_pending_auth,
                runtime_ensurer=self._ensure_managed_auth_runtime,
                usage_refresh_sink=self._get_managed_usage_owner().schedule_usage_refresh,
                usage_refresh_now=lambda: self._get_managed_usage_owner().refresh(
                    auto_show_founder_letter=False
                ),
                runtime_sink=self._set_translation_runtime_state,
                dashboard_sink=self.app.set_dashboard_translation_enabled,
                clear_context=lambda: self.hub.clear_context() if self.hub is not None else None,
                warmup=adapter.warmup,
                message_sink=lambda key, values: self._show_short_message(
                    key,
                    **dict(values),
                ),
                qq_dialog_sink=lambda: (
                    self.app.show_qq_managed_auth_dialog()
                    if callable(getattr(self.app, "show_qq_managed_auth_dialog", None))
                    else None
                ),
                result_sink=lambda result: setattr(
                    self,
                    "last_settings_mutation_result",
                    result,
                ),
                log_basic=self.log_basic,
                log_detailed=self.log_detailed,
                log_error=self._log_error,
                founder_letter_sink=adapter.show_founder_letter,
            )
            self._translation_enable_owner = owner
        return owner

    def _set_translation_runtime_state(self, enabled: bool) -> None:
        if self.hub is not None:
            self.hub.translation_enabled = bool(enabled)

    def _managed_openrouter_release_settings(self) -> AppSettings | None:
        if self.settings is None:
            return None
        if not self._managed_openrouter_selected():
            return None
        return self.settings

    def _managed_openrouter_selected(self) -> bool:
        return bool(
            self.settings is not None
            and self.settings.provider.llm == LLMProviderName.OPENROUTER
            and self.settings.translation.connection in _MANAGED_OPENROUTER_CONNECTIONS
            and self.settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
        )

    def dashboard_managed_auth_action(self) -> str:
        return self._get_managed_auth_owner().dashboard_action()

    def dashboard_managed_auth_prompt_kind(self) -> str:
        return self._get_managed_auth_owner().dashboard_prompt_kind()

    async def start_qq_managed_auth_from_dialog(
        self,
        *,
        qq_identity: str,
        credential: str,
    ) -> bool | tuple[str, dict[str, object]]:
        return await self._get_managed_auth_owner().start_qq(
            qq_identity=qq_identity,
            credential=credential,
        )

    async def start_discord_managed_auth_from_dialog(
        self,
        *,
        on_callback_received: Callable[[], None] | None = None,
        referral_id: str | None = None,
    ) -> bool:
        return await self._get_managed_auth_owner().start_discord(
            on_callback_received=on_callback_received,
            referral_id=referral_id,
        )

    def is_github_star_prompt_eligible(self) -> bool:
        return self._get_github_star_prompt_owner().is_eligible()

    def should_show_github_star_prompt(self, *, now: datetime | None = None) -> bool:
        return self._get_github_star_prompt_owner().should_show(now=now)

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

    async def persist_github_star_prompt_clicked(self) -> bool:
        return await self._get_github_star_prompt_owner().persist_clicked()

    async def persist_github_star_prompt_translation_success_observed(self) -> bool:
        return await self._get_github_star_prompt_owner().persist_translation_success_observed()

    def _get_github_star_prompt_owner(self) -> GithubStarPromptOwner:
        owner = self._github_star_prompt_owner
        if owner is None:
            owner = compose_github_star_prompt_owner(
                settings=self._get_settings_owner(),
                managed_remaining_percent=lambda: self._get_managed_usage_owner().remaining_percent,
                transaction_result_sink=lambda result: setattr(
                    self,
                    "last_settings_mutation_result",
                    result,
                ),
                save_failure_sink=self._log_github_star_prompt_save_failure,
                runtime_diagnostics_sink=self._github_star_prompt_runtime_diagnostics_sink,
                mutation_service_provider=lambda: self.settings_mutation_service,
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

    def _managed_usage_state(self) -> ManagedUsageState:
        settings = self.settings
        if settings is None:
            return ManagedUsageState(
                settings_available=False,
                managed_key_visible=False,
                release_settings_available=False,
                installation_id=None,
                entitlement_ref=None,
                referral_id=None,
                ingress_frozen=self._shutdown_ingress_frozen,
            )
        active_ref = settings.managed_identity.active_managed_credential_ref
        entitlement_ref = active_ref.strip() if isinstance(active_ref, str) else None
        return ManagedUsageState(
            settings_available=True,
            managed_key_visible=self._managed_openrouter_selected(),
            release_settings_available=self._managed_openrouter_release_settings() is not None,
            installation_id=settings.managed_identity.installation_id.strip() or None,
            entitlement_ref=entitlement_ref or None,
            referral_id=normalize_owned_referral_id(settings.managed_identity.referral_id),
            ingress_frozen=self._shutdown_ingress_frozen,
        )

    async def _fetch_managed_usage_metadata(self) -> ManagedUsageMetadataResult:
        settings = self.settings
        release_settings = self._managed_openrouter_release_settings()
        if settings is None or release_settings is None:
            return ManagedUsageMetadataResult(key_available=False, metadata=None)
        try:
            secrets = create_secret_store(settings.secrets, config_path=self.config_path)
            resolution = resolve_openrouter_credentials(
                build_openrouter_credential_runtime_config(release_settings),
                secrets=secrets,
            )
        except Exception:
            return ManagedUsageMetadataResult(key_available=False, metadata=None)
        api_key = resolution.api_key
        if not api_key:
            return ManagedUsageMetadataResult(key_available=False, metadata=None)
        metadata = await self._get_provider_verifier().fetch_openrouter_key_metadata(api_key)
        return ManagedUsageMetadataResult(key_available=True, metadata=metadata)

    def _apply_managed_usage_view_state(self, state: ManagedUsageViewState) -> None:
        self.app.set_settings_managed_key_state(
            visible=state.visible,
            remaining_percent=state.remaining_percent,
            referral_id=state.referral_id,
            pass_status=state.pass_status,
        )

    def _managed_usage_auto_show_founder_letter(self, metadata) -> bool:
        if self.settings is None:
            return False
        return should_auto_show_founder_letter(
            build_managed_identity_state_port(
                self.settings,
                lambda _settings: None,
            ),
            metadata,
        )

    def _managed_usage_warning_sink(
        self,
        message: str,
        exception: BaseException | None,
    ) -> None:
        if message.startswith("[ManagedAuth] Background refresh failed"):
            self.log_detailed(
                message,
                level=logging.WARNING,
                exception=exception,
            )
            return
        self.log_basic(message, level=logging.WARNING)

    def _get_managed_usage_owner(self) -> ManagedUsageOwner:
        owner = self._managed_usage_owner
        if owner is None:
            owner = ManagedUsageOwner(
                state_provider=self._managed_usage_state,
                release_service_provider=lambda: self._managed_openrouter_release_service,
                metadata_fetcher=self._fetch_managed_usage_metadata,
                pending_sink=self._set_managed_trial_pending_auth,
                view_sink=self._apply_managed_usage_view_state,
                disable_translation_sink=lambda reopen: (
                    self._get_translation_enable_owner().disable_for_managed_exhaustion(
                        reopen_founder_letter=reopen
                    )
                ),
                auto_show_founder_letter_provider=self._managed_usage_auto_show_founder_letter,
                normalize_referral_id=normalize_owned_referral_id,
                warning_sink=self._managed_usage_warning_sink,
            )
            self._managed_usage_owner = owner
        return owner

    def _on_managed_trial_delegate_ready(self) -> None:
        self._get_managed_usage_owner().delegate_ready()

    async def _refresh_managed_trial_usage_state_best_effort(self) -> None:
        await self._get_managed_usage_owner().refresh_best_effort()

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
        current_self_signature = build_self_stt_runtime_signature(settings)
        self._last_stt_runtime_signature = current_self_signature
        self._last_self_stt_runtime_signature = current_self_signature
        self._last_peer_stt_runtime_signature = build_peer_stt_runtime_signature(
            settings,
            canonical_settings=self._canonical_vnext_settings_for(settings),
        )
        self._last_self_stt_provider_signature = build_self_stt_provider_signature(settings)
        self._last_peer_stt_provider_signature = build_peer_stt_provider_signature_from_vnext(
            self._canonical_vnext_settings_for(settings)
        )
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
        return self._get_peer_application_owner().desired_active(
            self._peer_application_state(settings)
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
            and build_self_stt_runtime_signature(base_settings)
            != build_self_stt_runtime_signature(next_settings)
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
            and build_peer_stt_runtime_signature(
                base_settings,
                canonical_settings=self._canonical_vnext_settings_for(base_settings),
            )
            != build_peer_stt_runtime_signature(
                next_settings,
                canonical_settings=self._canonical_vnext_settings_for(next_settings),
            )
        )
        return self_changed or peer_changed

    async def _compensate_failed_local_asr_settings_apply(
        self,
        *,
        base_settings: AppSettings,
        committed_settings: AppSettings,
    ) -> None:
        self._get_settings_owner().apply_legacy_delta(
            committed_settings,
            base_settings,
        )
        await asyncio.to_thread(self._get_settings_owner().persist)
        self._get_settings_owner().remember_projection(base_settings)
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
        return OverlayApplicationOwner.normalized_target(value)

    def _overlay_target_for_settings(self, settings: AppSettings | None = None) -> str:
        return self._get_overlay_application_owner().target_for_state(
            self._overlay_application_state(settings)
        )

    def _effective_overlay_target_for_start(self) -> str:
        return self._get_overlay_application_owner().effective_target_for_start()

    def _clear_overlay_session_desktop_fallback(self) -> None:
        self._get_overlay_application_owner().clear_fallback()

    def _set_overlay_session_fallback_notice_active(self, active: bool) -> None:
        self._get_overlay_application_owner().publish_fallback(active)

    def _should_session_fallback_overlay_to_desktop(self, reason: str) -> bool:
        return self._get_overlay_application_owner().should_fallback(reason)

    def _new_overlay_runtime_handle(self) -> OverlayRuntimeHandle:
        return self._get_overlay_application_owner().new_runtime()

    def _ensure_overlay_runtime_handle(self) -> OverlayRuntimeHandle:
        return self._get_overlay_application_owner().ensure_runtime()

    def _overlay_runtime_is_current(
        self,
        runtime: OverlayRuntimeHandle,
        *,
        overlay_instance_id: str | None = None,
    ) -> bool:
        return self._get_overlay_application_owner().runtime_is_current(
            runtime,
            overlay_instance_id=overlay_instance_id,
        )

    def _overlay_runtime_has_resources(self, runtime: OverlayRuntimeHandle | None) -> bool:
        return OverlayApplicationOwner.runtime_has_resources(runtime)

    async def _replace_hub_overlay_sink(
        self,
        overlay_sink: object | None,
        *,
        expected_current: object | None = None,
        require_match: bool = False,
    ) -> bool:
        return await self._get_overlay_application_owner().replace_hub_sink(
            overlay_sink,
            expected_current=expected_current,
            require_match=require_match,
        )

    async def _close_stale_overlay_start_runtime(
        self,
        runtime: OverlayRuntimeHandle,
    ) -> None:
        await self._get_overlay_application_owner().close_stale_start(runtime)

    def _overlay_runtime_is_active(self) -> bool:
        return self._get_overlay_application_owner().runtime_is_active()

    def _current_overlay_presenter_for_direct_runtime_command(self) -> OverlayPresenter | None:
        return self._get_overlay_application_owner().current_presenter()

    def _current_overlay_bridge_for_direct_runtime_command(self) -> OverlayBridge | None:
        return self._get_overlay_application_owner().current_bridge()

    def _previous_overlay_target_for_apply(self) -> str:
        return self._get_overlay_application_owner().previous_target_for_apply()

    def _overlay_process_runner_for_target(
        self,
        target: str,
        *,
        task_factory: object | None = None,
    ) -> OverlayProcessRunner:
        return self._get_overlay_application_owner().process_runner(target, task_factory)

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

    def _on_peer_capture_state_changed(
        self,
        snapshot: PeerCaptureSessionSnapshot,
    ) -> None:
        self._get_peer_application_owner().on_runtime_state_changed(snapshot)

    def _resolve_peer_capture_target(
        self,
        settings: AppSettings,
    ) -> ResolvedDesktopAudioCaptureTarget:
        return self._get_peer_application_owner().resolve_capture_target(
            legacy_output_device=settings.desktop_audio.output_device,
            persisted_capture_target=getattr(
                settings.desktop_audio,
                "runtime_capture_target",
                None,
            ),
        )

    async def _close_peer_runtime_for_release(self, failures: list[Exception]) -> None:
        owner = self._peer_application_owner
        if owner is None:
            return
        try:
            await owner.close()
        except Exception as exc:
            failures.append(exc)

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
                owner_name="OverlayApplicationOwner",
                callback_name="close",
                callback=self._close_overlay_runtime,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL,
                owner_name="PeerApplicationOwner",
                callback_name="close",
                callback=self._close_peer_runtime,
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
                owner_name="ManagedAuthOwner",
                callback_name="close",
                callback=self._close_managed_auth_owner,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL,
                owner_name="TranslationEnableOwner",
                callback_name="close",
                callback=self._close_translation_enable_owner,
            ),
            application_shutdown_callback(
                phase=SHUTDOWN_PHASE_OWNER_DRAIN_CANCEL,
                owner_name="ManagedUsageOwner",
                callback_name="close",
                callback=self._close_managed_usage_owner,
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
        peer_owner = self._peer_application_owner
        if peer_owner is not None:
            peer_owner.stop_ingress()
        owner = self._vrchat_osc_presence_owner
        if owner is not None:
            owner.stop_ingress()
        overlay_owner = self._overlay_application_owner
        if overlay_owner is not None:
            overlay_owner.stop_ingress()
        vrc_mic_sync_owner = self._vrc_mic_sync_owner
        if vrc_mic_sync_owner is not None:
            vrc_mic_sync_owner.stop_ingress()
        logging_owner = self._runtime_logging_owner
        if logging_owner is not None:
            logging_owner.stop_ingress()
        managed_usage_owner = self._managed_usage_owner
        if managed_usage_owner is not None:
            managed_usage_owner.stop_ingress()
        managed_auth_owner = self._managed_auth_owner
        if managed_auth_owner is not None:
            managed_auth_owner.stop_ingress()
        translation_enable_owner = self._translation_enable_owner
        if translation_enable_owner is not None:
            translation_enable_owner.stop_ingress()

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

    async def _close_managed_usage_owner(self) -> None:
        owner = self._managed_usage_owner
        if owner is not None:
            await owner.close()

    async def _close_managed_auth_owner(self) -> None:
        owner = self._managed_auth_owner
        if owner is not None:
            await owner.close()

    async def _close_translation_enable_owner(self) -> None:
        owner = self._translation_enable_owner
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
        owner = self._get_overlay_application_owner()
        owner.stop_ingress()
        await self._shutdown_overlay_runtime(preserve_failure_reason=True)
        owner.clear_fallback()
        await owner.fallback_owner.close()

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
        await self._get_overlay_application_owner().set_enabled(enabled)

    async def set_peer_translation_enabled(self, enabled: bool) -> None:
        await self._get_peer_application_owner().set_enabled(enabled)

    async def retry_peer_process_capture(self) -> bool:
        return await self._get_peer_application_owner().retry_process_capture()

    def _enqueue_peer_translation_disclosure(self) -> None:
        hub = self.hub
        if hub is None:
            return
        enqueue_disclosure = getattr(hub, "enqueue_peer_translation_disclosure", None)
        if callable(enqueue_disclosure):
            enqueue_disclosure(self.app.localize("peer_translation.disclosure"))

    def on_overlay_start_failed(self, failure_reason: str | None) -> None:
        self._get_overlay_application_owner().on_start_failed(failure_reason)

    def on_overlay_runtime_disconnected(self) -> None:
        self._get_overlay_application_owner().on_runtime_disconnected()

    def on_overlay_runtime_crashed(self) -> None:
        self._get_overlay_application_owner().on_runtime_crashed()

    async def _begin_overlay_start(self) -> None:
        await self._get_overlay_application_owner().begin_start()

    async def _run_overlay_start(self, runtime: OverlayRuntimeHandle | None = None) -> None:
        await self._get_overlay_application_owner().run_start(runtime)

    async def _shutdown_overlay_runtime(self, *, preserve_failure_reason: bool) -> None:
        await self._get_overlay_application_owner().shutdown(
            preserve_failure_reason=preserve_failure_reason
        )

    async def _teardown_overlay_runtime(
        self,
        *,
        preserve_presenter_state: bool,
        emit_shutdown: bool = False,
    ) -> bool:
        return await self._get_overlay_application_owner().teardown(
            preserve_presenter_state=preserve_presenter_state,
            emit_shutdown=emit_shutdown,
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

    async def _persist_overlay_calibration(
        self,
        calibration: OverlayCalibration,
    ) -> None:
        if self.settings is None:
            return
        next_settings = copy.deepcopy(self.settings)
        next_settings.overlay.calibration = calibration.copy()
        await self._apply_overlay_osc_output_settings_via_mutation_service(next_settings)

    def cancel_overlay_calibration(self) -> OverlayCalibration:
        return self._get_overlay_calibration_owner().cancel()

    def _sync_overlay_calibration_cache(self, settings: AppSettings | None = None) -> None:
        resolved_settings = settings or self.settings
        if resolved_settings is None:
            return
        self._get_overlay_calibration_owner().replace_current(resolved_settings.overlay.calibration)

    async def _emit_overlay_calibration_to_runtime(
        self,
        calibration: OverlayCalibration,
    ) -> None:
        presenter = self._current_overlay_presenter_for_direct_runtime_command()
        if presenter is None:
            return
        await presenter.update_calibration(calibration.copy())

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
        return await self._get_translation_enable_owner().set_enabled(enabled)

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
            self._get_gpu_runtime_interaction_owner().clear_pending("self")

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

    def _schedule_vrchat_osc_presence_probe(self, *, force: bool = False) -> None:
        self._get_vrchat_osc_presence_owner().schedule(force=force)

    async def _cancel_vrchat_osc_presence_probe(self) -> None:
        await self._get_vrchat_osc_presence_owner().cancel()

    def _get_vrchat_osc_presence_owner(self) -> VrchatOscPresenceProbeOwner:
        owner = self._vrchat_osc_presence_owner
        if owner is None:
            owner = create_vrchat_osc_presence_probe_owner(
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

    def _manual_local_asr_fallback_state(
        self,
        settings: AppSettings,
    ) -> ManualLocalASRFallbackState:
        cpu_auto_requested = (
            settings.provider.stt == STTProviderName.LOCAL_CPU_AUTO
            or settings.provider.peer_stt == STTProviderName.LOCAL_CPU_AUTO
        )
        cpu_auto_available = True
        if cpu_auto_requested:
            cpu_auto_available = (
                self._get_local_asr_provisioning_owner().snapshot.cpu_auto_available
            )
        return ManualLocalASRFallbackState(
            self_provider=settings.provider.stt.value,
            peer_provider=settings.provider.peer_stt.value,
            self_source_language=settings.languages.source_language,
            peer_source_language=settings.languages.effective_peer_source,
            cpu_auto_available=cpu_auto_available,
        )

    @staticmethod
    def _settings_with_manual_local_asr_fallback_plan(
        settings: AppSettings,
        plan: ManualLocalASRFallbackPlan,
    ) -> AppSettings:
        normalized = copy.deepcopy(settings)
        normalized.provider.stt = STTProviderName(plan.self_provider)
        normalized.provider.peer_stt = STTProviderName(plan.peer_provider)
        return normalized

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
        plan = self.manual_local_asr_fallback_owner.plan(
            self._manual_local_asr_fallback_state(previous),
            channel=channel,
        )
        if not plan.changed:
            return True
        normalized = self._settings_with_manual_local_asr_fallback_plan(
            previous,
            plan,
        )
        self.settings = normalized
        if self._save_settings() is False:
            self.settings = previous
            return False
        self._sync_ui_from_settings()
        self._notify_manual_local_asr_fallback(
            plan.fallback_channels,
            installation_fallback=plan.installation_fallback,
        )
        self._log_manual_local_asr_fallbacks(
            previous,
            normalized,
            plan.fallback_channels,
        )
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

    def _sync_local_cpu_auto_availability(self, available: bool) -> None:
        self.app.set_settings_local_cpu_auto_available(available)

    def _current_local_stt_runtime_status(self) -> str:
        if self.settings is None:
            return "ready"
        return local_asr_status_for_provider(
            self._get_local_asr_provisioning_owner().snapshot,
            self.settings.provider.stt.value,
        )

    def _peer_local_stt_requested(self, settings: AppSettings | None = None) -> bool:
        return self._get_peer_application_owner().local_stt_requested(
            self._peer_application_state(settings)
        )

    def _local_asr_cpu_repair_state(self) -> LocalASRCpuRepairRuntimeState:
        settings = self.settings
        self_provider = settings.provider.stt.value if settings is not None else None
        peer_provider = settings.provider.peer_stt.value if settings is not None else None
        return LocalASRCpuRepairRuntimeState(
            settings_available=settings is not None,
            locale=settings.ui.locale if settings is not None else None,
            self_provider=self_provider,
            peer_provider=peer_provider,
            self_provider_local=bool(self_provider in LOCAL_CPU_PROVIDERS),
            peer_requested=self._peer_local_stt_requested(settings),
            self_activation_generation=self._stt_activation_generation,
            peer_activation_generation=self._peer_activation_generation,
            self_desired=self._stt_desired,
        )

    def _apply_local_asr_cpu_repair_effect(
        self,
        effect: LocalASRCpuRepairEffect,
    ) -> None:
        if effect.type is LocalASRCpuRepairEffectType.DISABLE_SELF_INTENT:
            self._stt_desired = False
            return
        if effect.type is LocalASRCpuRepairEffectType.DISABLE_SELF_DASHBOARD:
            self.app.set_dashboard_stt_enabled(False)
            self.app.set_dashboard_stt_needs_key(False)
            return
        if effect.type is LocalASRCpuRepairEffectType.SYNC_NOTICE:
            self._sync_local_stt_notice()
            return
        if effect.type is LocalASRCpuRepairEffectType.SHOW_DOWNLOAD_FAILED:
            self._show_short_stt_message("local_stt.download_failed")
            return
        raise ValueError(f"Unsupported Local ASR CPU repair effect: {effect.type}")

    async def _resume_self_after_local_asr_cpu_repair(self) -> bool:
        self._stt_desired = True
        self._stt_activation_starting = True
        self._sync_local_stt_notice()
        snapshot = await self._ensure_stt_switch()
        if snapshot is not None and snapshot.generation != self._stt_activation_generation:
            return False
        self._stt_activation_starting = False
        self.app.set_dashboard_stt_enabled(
            bool(
                self._stt_desired and not self._stt_activation_failed and self._mic_task is not None
            )
        )
        self._sync_local_stt_notice()
        return True

    async def _resume_peer_after_local_asr_cpu_repair(self) -> None:
        await self._refresh_overlay_runtime_dependencies()

    def _get_local_asr_cpu_repair_owner(self) -> LocalASRCpuRepairOwner:
        owner = self._local_asr_cpu_repair_owner
        if owner is None:
            owner = create_local_asr_cpu_repair_owner(
                provisioning_provider=lambda: self._get_local_asr_provisioning_owner(),
                state_provider=lambda: self._local_asr_cpu_repair_state(),
                model_ids_for_provider=required_local_asr_model_ids,
                status_for_provider=lambda provider: local_asr_status_for_provider(
                    self._get_local_asr_provisioning_owner().snapshot,
                    provider,
                ),
                effect_sink=lambda effect: self._apply_local_asr_cpu_repair_effect(effect),
                rebuild_self_provider=lambda: self._rebuild_stt_provider(),
                resume_self=lambda: self._resume_self_after_local_asr_cpu_repair(),
                resume_peer=lambda: self._resume_peer_after_local_asr_cpu_repair(),
            )
            self._local_asr_cpu_repair_owner = owner
        return owner

    def _local_asr_readiness_state(self) -> LocalASRReadinessState:
        settings = self.settings
        return LocalASRReadinessState(
            settings_available=settings is not None,
            runtime_available=self.hub is not None,
            self_provider=settings.provider.stt.value if settings is not None else None,
            peer_provider=settings.provider.peer_stt.value if settings is not None else None,
            self_source_language=(
                settings.languages.source_language if settings is not None else ""
            ),
            peer_source_language=(
                settings.languages.effective_peer_source if settings is not None else ""
            ),
            self_desired=self._stt_desired,
            peer_requested=self._peer_local_stt_requested(settings),
            self_activation_generation=self._stt_activation_generation,
            peer_activation_generation=self._peer_activation_generation,
        )

    def _apply_local_asr_readiness_effect(
        self,
        effect: LocalASRReadinessEffect,
    ) -> None:
        if effect.type is LocalASRReadinessEffectType.DISABLE_SELF_UNSUPPORTED:
            self._stt_desired = False
            self.app.set_dashboard_stt_enabled(False)
            self.app.set_dashboard_stt_needs_key(False)
            self._show_short_stt_message("local_stt.language_unsupported")
            return
        if effect.type is LocalASRReadinessEffectType.DISABLE_SELF_INVALID:
            self._stt_desired = False
            self.app.set_dashboard_stt_enabled(False)
            self.app.set_dashboard_stt_needs_key(False)
            self._show_short_stt_message("error.local_stt_model_invalid")
            return
        if effect.type is LocalASRReadinessEffectType.SELF_DOWNLOAD_IN_PROGRESS:
            self._stt_desired = False
            self.app.set_dashboard_stt_enabled(False)
            self._show_short_stt_message("local_stt.download_in_progress")
            return
        if effect.type is LocalASRReadinessEffectType.DISABLE_PEER_UNSUPPORTED:
            owner = self._get_peer_application_owner()
            owner.disable_intent()
            owner.sync_effective_flags()
            self._show_short_stt_message("local_stt.language_unsupported")
            return
        if effect.type is LocalASRReadinessEffectType.SYNC_NOTICE:
            self._sync_local_stt_notice()
            return
        raise ValueError(f"Unsupported Local ASR readiness effect: {effect.type}")

    def _get_local_asr_readiness_owner(self) -> LocalASRReadinessOwner:
        owner = self._local_asr_readiness_owner
        if owner is None:

            async def probe_self_provider() -> None:
                if self.hub is None or not self._hub_has_stt_provider("self"):
                    raise RuntimeError("self STT provider is unavailable")
                if self._hub_local_asr_provider_runtime() is None:
                    raise RuntimeError("local ASR provider runtime is unavailable")
                await self.hub.warmup_stt_channel("self")

            def self_channel_provider() -> ProviderRuntimeChannelSnapshot | None:
                runtime = self._hub_local_asr_provider_runtime()
                return runtime.snapshot.channel_for("self") if runtime is not None else None

            owner = create_local_asr_readiness_owner(
                provisioning_provider=self._get_local_asr_provisioning_owner,
                cpu_repair_owner=self._get_local_asr_cpu_repair_owner(),
                state_provider=self._local_asr_readiness_state,
                effect_sink=self._apply_local_asr_readiness_effect,
                self_provider_available=lambda: (
                    self.hub is not None and self._hub_has_stt_provider("self")
                ),
                self_channel_provider=self_channel_provider,
                rebuild_self_provider=self._rebuild_stt_provider,
                probe_self_provider=probe_self_provider,
                persist_manual_fallback=lambda channel: (
                    self._persist_current_manual_local_asr_fallback(channel=channel)
                ),
                validate_gpu_activation=self._validate_gpu_activation,
                gpu_state_provider=lambda: (
                    self._get_gpu_runtime_interaction_owner().snapshot.ui_state
                ),
                retain_gpu_pending=lambda channel: (
                    self._get_gpu_runtime_interaction_owner().retain_pending(
                        cast(GpuASRChannel, channel)
                    )
                ),
                load_log_sink=self._get_local_asr_diagnostics_owner().log_load_result,
            )
            self._local_asr_readiness_owner = owner
        return owner

    @property
    def _local_stt_pending_enable_after_install(self) -> bool:
        return self._get_local_asr_cpu_repair_owner().snapshot.self_pending

    @_local_stt_pending_enable_after_install.setter
    def _local_stt_pending_enable_after_install(self, pending: bool) -> None:
        self._get_local_asr_cpu_repair_owner().set_self_pending(pending)

    def _reset_local_stt_pending_enable_after_install(self) -> None:
        self._get_local_asr_cpu_repair_owner().reset_self()

    def _reset_local_stt_pending_peer_enable_after_install(self) -> None:
        self._get_local_asr_cpu_repair_owner().reset_peer()

    def _clear_local_stt_pending_enable_if_provider_switched_away(self) -> None:
        self._get_local_asr_cpu_repair_owner().clear_if_provider_switched_away()

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
        provisioning_snapshot = self._get_local_asr_provisioning_owner().snapshot
        status = (
            local_asr_status_for_provider(provisioning_snapshot, self_provider)
            if self_local
            else local_asr_status_for_provider(provisioning_snapshot, peer_provider)
        )
        visible_model_ids = required_local_asr_model_ids(
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
        if self._stt_activation_starting and self_local_asr:
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
                starting=self._stt_activation_starting,
            )

    def _request_unavailable_local_asr_repair(
        self,
        status: str,
        *,
        channel: Literal["self", "peer"],
        model_ids: tuple[str, ...] | None = None,
        activation_generation: int | None = None,
    ) -> bool:
        return self._get_local_asr_cpu_repair_owner().request_repair(
            LocalASRCpuRepairRequest(
                status=status,
                channel=channel,
                model_ids=model_ids,
                activation_generation=activation_generation,
            )
        )

    async def _ensure_local_stt_ready(
        self,
        *,
        activation_generation: int | None = None,
    ) -> bool:
        return await self._get_local_asr_readiness_owner().ensure_self_ready(
            activation_generation=activation_generation,
        )

    async def _ensure_peer_local_stt_ready(
        self,
        *,
        activation_generation: int | None = None,
    ) -> bool:
        return await self._get_local_asr_readiness_owner().ensure_peer_ready(
            activation_generation=activation_generation,
            gpu_provider_id=STTProviderName.LOCAL_QWEN_GPU.value,
        )

    async def _close_local_asr_provisioning(self) -> None:
        self._get_local_asr_cpu_repair_owner().reset_all()
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
        config = build_self_capture_session_config(self.settings)
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
            build_self_capture_session_config(self.settings),
            enabled=desired,
            restart=restart,
            force_immediate=force_immediate,
            explicit_toggle_off=not desired,
        )
        self._on_self_capture_state_changed(snapshot)
        return snapshot

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

            owner = create_manual_typing_owner(
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
        self._get_settings_owner().begin(
            legacy_snapshot=self._get_settings_owner().projection_snapshot or self.settings
        )
        self._capture_runtime_signatures_before_canonical_mutation()
        self._get_settings_owner().apply_legacy_delta(self.settings, updated)
        try:
            await self.apply_settings(updated)
            if self.settings is not None:
                self.refresh_settings_projection(preserve_custom_vocab_draft=True)
        finally:
            self._get_settings_owner().complete()

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
        plan: ProviderRuntimeApplyPlan,
    ) -> None:
        self._sync_memory_runtime_fields_from_settings(base_settings)
        try:
            await self._get_provider_runtime_owner().apply(
                copy.deepcopy(committed_settings),
                plan,
            )
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
        normalization_channels = self.manual_local_asr_fallback_owner.normalization_channels(
            current=(
                self._manual_local_asr_fallback_state(self.settings)
                if self.settings is not None
                else None
            ),
            pending=self._manual_local_asr_fallback_state(settings),
        )
        if normalization_channels:
            await self._get_local_asr_provisioning_owner().inspect_cpu()
            fallback_plan = self.manual_local_asr_fallback_owner.plan(
                self._manual_local_asr_fallback_state(settings)
            )
            fallback_channels = tuple(
                channel
                for channel in fallback_plan.fallback_channels
                if channel in normalization_channels
            )
            if fallback_channels:
                settings = self._settings_with_manual_local_asr_fallback_plan(
                    settings,
                    ManualLocalASRFallbackPlan(
                        self_provider=(
                            fallback_plan.self_provider
                            if "self" in fallback_channels
                            else settings.provider.stt.value
                        ),
                        peer_provider=(
                            fallback_plan.peer_provider
                            if "peer" in fallback_channels
                            else settings.provider.peer_stt.value
                        ),
                        fallback_channels=fallback_channels,
                        installation_fallback=bool(
                            fallback_plan.installation_fallback and fallback_channels
                        ),
                    ),
                )
            installation_fallback = bool(fallback_plan.installation_fallback and fallback_channels)
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
            self._get_settings_owner().begin(
                legacy_snapshot=self._get_settings_owner().projection_snapshot or self.settings
            )
            self._capture_runtime_signatures_before_canonical_mutation()
            self._get_settings_owner().apply_legacy_delta(
                self._get_settings_owner().projection_snapshot or self.settings,
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
        if self._get_overlay_application_owner().fallback_owner.active:
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
                    self._get_settings_owner().persist()
                except Exception:
                    self._get_settings_owner().rollback()
                    raise _StrictSettingsSaveFailed from None
                else:
                    self._get_settings_owner().remember_projection(self.settings)
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
            explicit_intent=self._gpu_runtime_interaction_state().selected_provider_requires_model,
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

        current_self_signature = build_self_stt_runtime_signature(settings)
        current_peer_signature = build_peer_stt_runtime_signature(
            settings,
            canonical_settings=self._canonical_vnext_settings_for(settings),
        )
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
                and build_self_capture_vad_signature(previous_settings_for_desktop)
                == build_self_capture_vad_signature(settings)
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
            self._get_settings_owner().complete()

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
            NoopRuntimeApply()
            if has_out_of_scope_draft
            else SttLanguageAudioRuntimeApplyAdapter(
                apply_settings=self._apply_settings_runtime_effect,
                state_provider=self._settings_runtime_state,
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
        self._get_settings_owner().complete()
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
                    state=self._settings_runtime_state(next_settings),
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
            NoopRuntimeApply()
            if has_out_of_scope_draft
            else OverlayOscOutputRuntimeApplyAdapter(
                apply_settings=self._apply_settings_runtime_effect,
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
        self._get_settings_owner().complete()
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
        self._get_settings_owner().complete()
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
            NoopRuntimeApply()
            if has_out_of_scope_draft
            else UiPromptClipboardStateRuntimeApplyAdapter(
                apply_settings=self._apply_settings_runtime_effect,
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
        return await self._get_provider_credential_verification_owner().verify(provider, key)

    async def apply_providers(
        self,
        settings: AppSettings | None = None,
        *,
        force_rebuild_llm: bool = False,
    ) -> bool:
        return await self._get_provider_application_owner().apply(
            settings,
            force_rebuild_llm=force_rebuild_llm,
        )

    def _get_provider_application_owner(self) -> ProviderApplicationOwner:
        owner = self._provider_application_owner
        if owner is None:
            owner = ProviderApplicationOwner(
                settings=self._get_settings_owner(),
                runtime=self._get_provider_runtime_owner(),
                merge_settings=self.merge_settings_tab_apply_with_current_languages,
                preserve_before_replace=(
                    self._preserve_github_star_prompt_observation_before_settings_replace
                ),
                sync_ui=self._sync_ui_from_settings,
                order24_patch_provider=(self._settings_projection().order24_patch_base_and_values),
                apply_order24=(self._apply_ui_prompt_clipboard_state_settings_via_mutation_service),
                remember_order22=self._settings_projection().remember_order22,
                mutation_service_provider=lambda: self.settings_mutation_service,
                persist_current_settings=self._save_settings,
                save_failure_sink=self._log_error,
                result_sink=lambda result: setattr(
                    self,
                    "last_settings_mutation_result",
                    result,
                ),
                result_provider=lambda: self.last_settings_mutation_result,
                sync_memory=self._sync_memory_runtime_fields_from_settings,
                capture_runtime_signatures=(
                    self._capture_runtime_signatures_before_canonical_mutation
                ),
                sync_signatures=self._sync_signature_caches,
                consume_superseded_settings=(self._consume_superseded_local_asr_settings),
                active_local_asr_change=self._active_local_asr_change,
                compensate_local_asr=self._compensate_failed_local_asr_settings_apply,
                copy_runtime_only_ui_state=_copy_runtime_only_ui_state,
                llm_retry_pending=lambda: self._last_llm_provider_signature == (),
                mark_llm_retry=self._mark_llm_provider_retry,
            )
            self._provider_application_owner = owner
        return owner

    def _consume_superseded_local_asr_settings(self, settings: AppSettings) -> bool:
        settings_id = id(settings)
        if settings_id not in self._superseded_local_asr_settings_ids:
            return False
        self._superseded_local_asr_settings_ids.discard(settings_id)
        return True

    def _get_provider_runtime_owner(self) -> ProviderRuntimeOwner:
        owner = self._provider_runtime_owner
        if owner is None:
            owner = ProviderRuntimeOwner(
                state_provider=self._provider_runtime_state,
                common_effect=self._apply_provider_runtime_common_effects,
                rebuild_llm=self._get_llm_provider_rebuild_owner().rebuild,
                recover_gpu=self._apply_gpu_runtime_owner_recovery,
                refresh_peer=self._refresh_provider_runtime_peer_effect,
                refresh_self_stt=self._refresh_provider_runtime_self_stt_effect,
                signature_sink=self._sync_signature_caches,
                llm_retry_sink=self._mark_llm_provider_retry,
                current_settings_provider=lambda: self.settings,
                signature_cache_provider=lambda: (
                    self._last_self_stt_provider_signature,
                    self._last_peer_stt_provider_signature,
                    self._last_llm_provider_signature,
                ),
                self_signature_builder=build_self_stt_provider_signature,
                peer_signature_builder=lambda settings, canonical: (
                    build_peer_stt_provider_signature_from_vnext(
                        canonical or self._canonical_vnext_settings_for(settings)
                    )
                ),
                llm_signature_builder=self._build_llm_provider_signature,
                gpu_restart_decision=self._provider_runtime_requires_gpu_restart,
            )
            self._provider_runtime_owner = owner
        return owner

    def _provider_runtime_state(self, settings: object) -> ProviderRuntimeState:
        hub = self.hub
        return ProviderRuntimeState(
            runtime_available=hub is not None,
            llm_available=hub is not None and hub.llm is not None,
            self_stt_available=hub is not None and hub.has_stt_provider("self"),
            peer_stt_available=hub is not None and hub.has_stt_provider("peer"),
            self_stt_desired=self._stt_desired,
            peer_stt_desired=self._peer_runtime_should_be_active(settings),
        )

    async def _apply_settings_runtime_effect(
        self,
        settings: object,
        reload_settings_view: bool,
    ) -> None:
        if not isinstance(settings, AppSettings):
            raise TypeError("settings runtime effect requires AppSettings")
        await self._apply_settings_direct(
            settings,
            persist=False,
            strict_runtime_errors=True,
            reload_settings_view=reload_settings_view,
        )

    def _settings_runtime_state(self, settings: object) -> SettingsRuntimeState:
        if not isinstance(settings, AppSettings):
            raise TypeError("settings runtime state requires AppSettings")
        hub = self.hub
        return SettingsRuntimeState(
            runtime_available=hub is not None,
            self_stt_desired=self._stt_desired,
            self_stt_available=hub is not None and hub.has_stt_provider("self"),
            peer_stt_desired=self._peer_runtime_should_be_active(settings),
            peer_stt_available=hub is not None and hub.has_stt_provider("peer"),
            qwen_llm_desired=self._is_qwen_llm(settings),
            llm_available=hub is not None and hub.llm is not None,
        )

    def _apply_provider_runtime_common_effects(self, settings: object) -> None:
        if not isinstance(settings, AppSettings):
            raise TypeError("provider runtime settings must be AppSettings")
        next_settings = settings
        self.settings = next_settings
        self._clear_local_stt_pending_enable_if_provider_switched_away()
        self._sync_local_stt_notice()
        if (
            next_settings.provider.llm != LLMProviderName.OPENROUTER
            or next_settings.openrouter.selected_source != OpenRouterCredentialSource.MANAGED
        ):
            self._set_managed_trial_pending_auth(False)
        else:
            self.app.set_dashboard_managed_auth_pending(self.managed_auth_pending)

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

    async def _refresh_provider_runtime_peer_effect(self) -> None:
        await self._refresh_peer_stt_runtime()
        if self.settings is not None:
            self._sync_effective_hub_flags(self.settings)
        self._refresh_overlay_peer_consumers()

    async def _refresh_provider_runtime_self_stt_effect(self) -> None:
        if self._stt_desired:
            await self._apply_stt_runtime_replacement(smooth_local=True)
        else:
            await self._rebuild_stt_provider()

    def _mark_llm_provider_retry(self) -> None:
        self._last_llm_provider_signature = ()

    @staticmethod
    def _provider_runtime_requires_gpu_restart(
        current_settings: object,
        next_settings: object,
    ) -> bool:
        if not isinstance(current_settings, AppSettings) or not isinstance(
            next_settings,
            AppSettings,
        ):
            return False
        return current_settings.stt.gpu_device_id != next_settings.stt.gpu_device_id and (
            current_settings.provider.stt == STTProviderName.LOCAL_QWEN_GPU
            or current_settings.provider.peer_stt == STTProviderName.LOCAL_QWEN_GPU
            or next_settings.provider.stt == STTProviderName.LOCAL_QWEN_GPU
            or next_settings.provider.peer_stt == STTProviderName.LOCAL_QWEN_GPU
        )

    async def _apply_gpu_runtime_owner_recovery(
        self,
        next_settings: AppSettings,
        plan: ProviderRuntimeApplyPlan,
    ) -> None:
        await self._get_gpu_provider_recovery_owner().recover(
            lambda: self._gpu_provider_recovery_request(
                next_settings,
                reason="settings_restart",
                plan=plan,
            )
        )

    def _get_gpu_provider_recovery_owner(self) -> GpuProviderRecoveryApplicationOwner:
        owner = self._gpu_provider_recovery_owner
        if owner is None:
            owner = create_gpu_provider_recovery_application_owner(
                runtime_provider=self._hub_local_asr_provider_runtime,
                pending_provider=lambda: (
                    self._get_gpu_runtime_interaction_owner().snapshot.pending_channels
                ),
                pending_clear=lambda channels: (
                    self._get_gpu_runtime_interaction_owner().complete_manual_recovery(channels)
                ),
                failure_sink=lambda reason: (
                    self._get_gpu_runtime_interaction_owner().set_ui_state(
                        "activation_failed",
                        publish_notice=True,
                        origin=("manual_retry" if reason == "manual_retry" else "settings_apply"),
                    )
                ),
                runtime_state_sink=self._on_local_asr_provider_runtime_state_changed,
                quiesce=self._suspend_gpu_provider_consumers,
                self_owner_factory=self._get_self_capture_owner,
                peer_owner_provider=lambda: self._peer_runtime,
                self_state_sink=self._on_self_capture_state_changed,
                ensure_self_switch=self._ensure_stt_switch,
                refresh_self=self._refresh_provider_runtime_self_stt_effect,
                refresh_peer=self._refresh_provider_runtime_peer_effect,
                diagnostic_sink=self._on_gpu_provider_recovery_diagnostic,
            )
            self._gpu_provider_recovery_owner = owner
        return owner

    def _gpu_provider_recovery_request(
        self,
        settings: AppSettings,
        *,
        reason: Literal["manual_retry", "settings_restart"],
        plan: ProviderRuntimeApplyPlan | None,
    ) -> GpuProviderRecoveryApplicationRequest:
        if reason == "settings_restart" and plan is None:
            raise RuntimeError("settings GPU recovery requires a runtime apply plan")
        return GpuProviderRecoveryApplicationRequest(
            device_id=settings.stt.gpu_device_id,
            reason=reason,
            self_gpu_selected=settings.provider.stt == STTProviderName.LOCAL_QWEN_GPU,
            peer_gpu_selected=settings.provider.peer_stt == STTProviderName.LOCAL_QWEN_GPU,
            self_desired=self._stt_desired,
            peer_enabled=settings.ui.peer_translation_enabled,
            self_config_factory=lambda: build_self_capture_session_config(settings),
            peer_config_factory=lambda: build_peer_capture_session_config(
                settings,
                canonical_settings=self._canonical_vnext_settings_for(settings),
            ),
            self_request_factory=lambda: build_self_stt_provider_request(
                settings,
                warmup=True,
            ),
            peer_request_factory=lambda config: build_peer_stt_provider_request(
                config,
                gpu_device_id=settings.stt.gpu_device_id,
                warmup=True,
            ),
            should_refresh_self=bool(plan is not None and plan.should_refresh_self_stt),
            should_refresh_peer=bool(plan is not None and plan.should_refresh_peer),
        )

    async def _suspend_gpu_provider_consumers(
        self,
        channels: tuple[GpuASRChannel, ...],
    ) -> None:
        if "self" in channels and self._self_capture_owner is not None:
            snapshot = await self._self_capture_owner.suspend_provider_consumer()
            self._on_self_capture_state_changed(snapshot)
        if "peer" in channels and self._peer_runtime is not None:
            await self._peer_runtime.suspend_provider_consumer()

    def _on_gpu_provider_recovery_diagnostic(
        self,
        diagnostic: GpuProviderRecoveryDiagnostic,
    ) -> None:
        fields = [
            f"outcome={diagnostic.outcome}",
            f"reason={diagnostic.reason}",
            f"channels={','.join(diagnostic.channels) or 'none'}",
        ]
        if diagnostic.failure_type is not None:
            fields.append(f"failure_type={diagnostic.failure_type}")
        self.log_detailed(
            f"[GPU ASR] provider_recovery {' '.join(fields)}",
            level=(
                logging.WARNING
                if diagnostic.outcome in {"failed", "prepare_failed"}
                else logging.INFO
            ),
        )

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

    def _canonical_vnext_settings_for(self, settings: AppSettings) -> AppSettingsVNext:
        projected = self._get_settings_owner().project(
            settings,
            authoritative=self._get_settings_owner().authoritative,
        )
        return projected

    def _provider_verification_binding(
        self,
        provider: str,
        key: str,
        *,
        flow: str,
        context_values: Mapping[str, object] | None = None,
    ) -> ProviderVerificationBinding:
        return self._get_provider_settings_owner().verification_binding(
            provider,
            key,
            flow=flow,
            context_values=context_values,
        )

    async def persist_provider_secret_change(
        self,
        secret_key: str,
        value: str,
    ) -> bool:
        owner = self._get_provider_settings_owner()
        succeeded = await owner.change_secret(secret_key, value)
        self.last_settings_mutation_result = owner.last_result
        return succeeded

    def persist_api_key_verification(
        self,
        provider: str,
        key: str,
        success: bool,
    ) -> None:
        self._get_provider_settings_owner().persist_verification(provider, key, success)

    def clear_provider_verification(self, provider: str) -> None:
        self.persist_api_key_verification(provider, "", False)

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
        baseline = self._get_settings_owner().projection_snapshot or active_settings
        managed_baseline = (
            bound_managed_snapshot
            if bound_managed_snapshot is not None
            else baseline.managed_identity
        )
        managed_delta = _managed_identity_delta(managed_baseline, settings.managed_identity)
        next_settings = copy.deepcopy(active_settings)
        _apply_managed_identity_delta(next_settings, managed_delta)
        self._get_settings_owner().begin(legacy_snapshot=baseline)
        self._get_settings_owner().apply_legacy_delta(baseline, next_settings)
        try:
            self._get_settings_owner().persist()
        except Exception:
            self._get_settings_owner().rollback()
            _restore_managed_identity(settings, managed_baseline)
            raise
        self.settings = next_settings
        self._get_settings_owner().remember_projection(next_settings)
        self._get_settings_owner().complete()

    def _capture_runtime_signatures_before_canonical_mutation(self) -> None:
        if self.settings is None:
            return
        if self._last_peer_stt_provider_signature is None:
            self._last_peer_stt_provider_signature = build_peer_stt_provider_signature_from_vnext(
                self._canonical_vnext_settings_for(self.settings)
            )
        if self._last_peer_stt_runtime_signature is None:
            self._last_peer_stt_runtime_signature = build_peer_stt_runtime_signature(
                self.settings,
                canonical_settings=self._canonical_vnext_settings_for(self.settings),
            )

    def _get_provider_verifier(self) -> ProviderVerifierPort:
        if self.provider_verifier is None:
            self.provider_verifier = create_provider_verifier()
        return self.provider_verifier

    def _get_provider_credential_verification_owner(
        self,
    ) -> ProviderCredentialVerificationInteractionOwner:
        owner = self._provider_credential_verification_owner
        if owner is None:
            owner = create_provider_credential_verification_interaction_owner(
                verifier=self._get_provider_verifier(),
                selected_model_provider=self._provider_credential_selected_model,
                fallback_models=tuple(model.value for model in QwenLLMModel),
                low_latency=FIXED_TRANSLATION_POLICY.fast_translation_enabled,
                diagnostics_sink=lambda event, metadata, exception: self.log_detailed(
                    "[ProviderVerification] Credential verification failed "
                    f"event={event} provider={metadata.get('provider')} "
                    f"error_type={metadata.get('error_type')}",
                    level=logging.WARNING,
                    exception=exception,
                ),
                error_sink=lambda provider, error_text: self._log_error(
                    f"Verification error for {provider}: {error_text}"
                ),
            )
            self._provider_credential_verification_owner = owner
        return owner

    def _provider_credential_selected_model(self, provider: str) -> str | None:
        return self._get_provider_settings_owner().binding.selected_model(provider)

    def _get_provider_settings_owner(
        self,
    ) -> ProviderSettingsOwner:
        owner = self.provider_settings_owner
        if owner is None:
            owner = ProviderSettingsOwner(
                settings=self._get_settings_owner(),
                binding=ProviderVerificationBindingOwner(
                    context_provider=lambda provider: provider_verification_context(
                        self.settings,
                        provider,
                        low_latency=FIXED_TRANSLATION_POLICY.fast_translation_enabled,
                    ),
                ),
                secret_store_factory=lambda settings: create_sync_secret_store_adapter(
                    create_secret_store(settings.secrets, config_path=self.config_path)
                ),
                active_secret_provider=lambda settings, secret_key: create_secret_store(
                    settings.secrets,
                    config_path=self.config_path,
                ).get(secret_key),
                save_failure_sink=self._log_error,
            )
            self.provider_settings_owner = owner
        return owner

    def _get_llm_provider_rebuild_owner(self) -> LlmProviderRebuildOwner:
        owner = self._llm_provider_rebuild_owner
        if owner is None:
            owner = LlmProviderRebuildOwner(
                context_provider=self._llm_provider_rebuild_context,
                provider_factory=self._create_llm_provider_for_rebuild,
                availability_sink=self.app.set_dashboard_translation_needs_key,
                usage_refresh=self._refresh_managed_trial_usage_state_best_effort,
                failure_sink=self._log_error,
                success_sink=self.log_basic,
            )
            self._llm_provider_rebuild_owner = owner
        return owner

    def _llm_provider_rebuild_context(self) -> LlmProviderRebuildContext | None:
        if self.hub is None or self.settings is None:
            return None
        return LlmProviderRebuildContext(
            settings=self.settings,
            replace_provider=self.hub.replace_llm_provider,
            requires_secret=self._llm_provider_requires_secret(self.settings.provider.llm),
        )

    async def _create_llm_provider_for_rebuild(self, settings: object) -> object | None:
        if not isinstance(settings, AppSettings):
            raise TypeError("LLM provider rebuild settings must be AppSettings")
        secrets = create_secret_store(settings.secrets, config_path=self.config_path)
        new_managed_release_service = self._create_managed_openrouter_release_service(
            secrets=secrets
        )
        await self._replace_managed_openrouter_release_service(new_managed_release_service)
        return create_llm_provider(
            settings,
            secrets=secrets,
            managed_release_service=self._managed_openrouter_release_service,
            managed_delegate_ready=self._on_managed_trial_delegate_ready,
            runtime_logging=self.runtime_logging,
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
            diagnostic_sink=self._get_local_asr_diagnostics_owner().provider_runtime_diagnostic,
        )

    def _on_local_asr_provider_runtime_state_changed(
        self,
        snapshot: LocalASRProviderRuntimeSnapshot,
    ) -> None:
        self._get_gpu_runtime_interaction_owner().observe_runtime(snapshot)

    async def _rebuild_stt_provider(self) -> None:
        """Rebuild only the STT provider so later enable uses current settings."""
        if self.hub is None or self.settings is None:
            return

        owner = self._get_self_capture_owner()
        config = build_self_capture_session_config(self.settings)
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

    def _on_peer_runtime_diagnostic(self, diagnostic: PeerCaptureDiagnostic) -> None:
        self._get_peer_application_owner().on_runtime_diagnostic(diagnostic)

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
        self._get_settings_owner().authoritative = True
        self._get_settings_owner().remember_projection(next_settings)
        self._peer_process_warning_reason = None
        await self._refresh_peer_stt_runtime()
        self._sync_effective_hub_flags(self.settings)
        self._refresh_overlay_peer_consumers()
        with contextlib.suppress(Exception):
            self.app.refresh_settings_loopback_capture_target(self.settings)

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

    async def _refresh_peer_stt_runtime(self, *, stop_mode: str = "retain") -> None:
        await self._get_peer_application_owner().refresh_runtime(
            stop_mode="release" if stop_mode == "release" else "retain"
        )

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
                stt_request = build_self_stt_provider_request(self.settings)
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
                build_self_capture_session_config(self.settings)
            )
            if snapshot.provider_status.value != "ready":
                self._log_error("STT backend not available")

        peer_runtime = compose_peer_capture_session_owner(
            hub=hub,
            admission=create_peer_capture_admission_adapter(
                runtime_available=lambda: self.settings is not None and self.hub is not None,
                ensure_local_ready=self._ensure_peer_local_stt_ready,
            ),
            target_resolver=create_peer_capture_target_resolver_adapter(),
            clock=self.clock,
            provider_request_factory=lambda config, warmup: build_peer_stt_provider_request(
                config,
                gpu_device_id=self.settings.stt.gpu_device_id,
                warmup=warmup,
            ),
            source_factory=create_peer_capture_source_adapter(
                log_detailed=self.log_detailed,
                wrap_source=lambda source: self._wrap_diagnostic_audio_source(
                    cast(AudioSource, source),
                    channel_label="peer",
                ),
                is_detailed_enabled=self._detailed_audio_diag_enabled,
            ),
            vad_factory=create_peer_capture_vad_adapter(
                log_detailed=self.log_detailed,
                diagnostics_enabled=self._detailed_audio_diag_enabled,
            ),
            run_audio_loop=create_peer_capture_audio_loop_adapter(
                log_detailed=self.log_detailed,
                is_detailed_enabled=self._detailed_audio_diag_enabled,
            ),
            vad_sink=create_peer_capture_vad_sink_adapter(runtime_provider=lambda: self.hub),
            state_changed=self._on_peer_capture_state_changed,
            diagnostic_sink=self._on_peer_runtime_diagnostic,
            local_asr_diagnostic_sink=(
                self._get_local_asr_diagnostics_owner().transition_diagnostic
            ),
        )
        await self._get_peer_application_owner().replace_runtime(peer_runtime)
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
        self._get_managed_auth_owner().on_callback_received()

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

    def _self_capture_admission_state(
        self,
        config: SelfCaptureSessionConfig,
    ) -> SelfCaptureAdmissionState:
        settings = self.settings
        decision = (
            resolve_local_asr_selection(
                settings.provider.stt.value,
                settings.languages.source_language,
            )
            if settings is not None and config.local_cpu
            else None
        )
        return SelfCaptureAdmissionState(
            settings_available=settings is not None,
            runtime_available=self.hub is not None,
            gpu_status=self._get_gpu_runtime_interaction_owner().snapshot.ui_state,
            local_cpu_supported=bool(decision is None or decision.supported),
            local_runtime_status=(
                self._current_local_stt_runtime_status()
                if decision is not None and decision.supported
                else "ready"
            ),
            activation_generation=self._stt_activation_generation,
        )

    def _apply_self_capture_admission_effect(
        self,
        effect: SelfCaptureAdmissionEffect,
    ) -> None:
        if effect.type is SelfCaptureAdmissionEffectType.RETAIN_GPU_PENDING_INTENT:
            self._get_gpu_runtime_interaction_owner().retain_pending("self")
            return
        if effect.type is SelfCaptureAdmissionEffectType.REJECT_UNSUPPORTED_LANGUAGE:
            self.app.set_dashboard_stt_enabled(False)
            self.app.set_dashboard_stt_needs_key(False)
            self._show_short_stt_message("local_stt.language_unsupported")
            return
        if effect.type is SelfCaptureAdmissionEffectType.RETAIN_DOWNLOAD_PENDING_INTENT:
            self._get_local_asr_cpu_repair_owner().retain_pending(
                "self",
                activation_generation=effect.activation_generation,
            )
            self.app.set_dashboard_stt_enabled(False)
            self._show_short_stt_message("local_stt.download_in_progress")
            return
        if effect.type is SelfCaptureAdmissionEffectType.REQUEST_LOCAL_REPAIR:
            assert effect.status is not None
            self._request_unavailable_local_asr_repair(
                effect.status,
                channel="self",
                activation_generation=effect.activation_generation,
            )
            return
        raise ValueError(f"Unsupported Self capture admission effect: {effect.type}")

    def _get_self_capture_owner(self) -> SelfCaptureSessionOwner:
        if self._self_capture_owner is not None:
            return self._self_capture_owner
        self._self_capture_owner = compose_self_capture_session_owner(
            hub=self.hub,
            admission=create_self_capture_admission_adapter(
                state_provider=self._self_capture_admission_state,
                validate_gpu_activation=self._validate_gpu_activation,
                effect_sink=self._apply_self_capture_admission_effect,
            ),
            provider_request_factory=self._self_capture_provider_request,
            source_factory=create_self_capture_source_adapter(
                log_detailed=self.log_detailed,
                wrap_source=lambda source: self._wrap_diagnostic_audio_source(
                    cast(AudioSource, source),
                    channel_label="self",
                ),
            ),
            vad_factory=create_self_capture_vad_adapter(
                log_detailed=self.log_detailed,
                diagnostics_enabled=self._detailed_audio_diag_enabled,
            ),
            run_audio_loop=create_self_capture_audio_loop_adapter(
                audio_gate_provider=lambda: self.vrc_mic_audio_gate,
                log_detailed=self.log_detailed,
                is_detailed_enabled=self._detailed_audio_diag_enabled,
            ),
            vad_sink=create_self_capture_vad_sink_adapter(runtime_provider=lambda: self.hub),
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
        return build_self_stt_provider_request(self.settings, warmup=warmup)

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
            self._vad = owner.vad
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

    def _create_openrouter_pkce_client(self) -> OpenRouterPKCEClient:
        return OpenRouterPKCEClient(callback_origin="http://localhost:3000")

    def reopen_openrouter_pkce_authorization_url(self) -> bool:
        owner = self._openrouter_pkce_flow_owner
        return owner.reopen_authorization_url() if owner is not None else False

    def build_managed_openrouter_byok_target_settings(self) -> AppSettings | None:
        return build_managed_openrouter_byok_target_settings(self.settings)

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

        plan = self._get_provider_runtime_owner().build_plan(
            updated,
            force_rebuild_llm=True,
        )
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
        runtime_apply_port = ProviderRuntimeApplyAdapter(
            owner=self._get_provider_runtime_owner(),
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
            self._get_settings_owner().complete()
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
            self._get_settings_owner().complete()
            return True

        self.last_settings_mutation_result = _runtime_apply_result_as_degraded_transaction(
            runtime_result
        )
        self._get_settings_owner().complete()
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
        owner = self._get_settings_owner()
        owns_mutation = owner.mutation_depth == 0
        baseline = owner.projection_snapshot or self.settings
        if owns_mutation:
            owner.begin(legacy_snapshot=baseline)
        owner.apply_legacy_delta(baseline, self.settings)
        try:
            owner.persist()
        except Exception as exc:
            owner.rollback()
            self._log_error(f"Failed to save settings: {exc}")
            return False
        else:
            owner.remember_projection(self.settings)
            if owns_mutation:
                owner.complete()
            return True

    def persist_settings(self) -> None:
        """Persist current settings, propagating persistence errors to the caller."""
        assert self.settings is not None
        owner = self._get_settings_owner()
        owns_mutation = owner.mutation_depth == 0
        baseline = owner.projection_snapshot or self.settings
        if owns_mutation:
            owner.begin(legacy_snapshot=baseline)
        owner.apply_legacy_delta(baseline, self.settings)
        try:
            owner.persist()
        except Exception:
            owner.rollback()
            raise
        owner.remember_projection(self.settings)
        if owns_mutation:
            owner.complete()

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
