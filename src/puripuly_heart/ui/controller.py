from __future__ import annotations

import asyncio
import contextlib
import copy
import inspect
import logging
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

from puripuly_heart.app.language_selection import LanguageSelectionChange
from puripuly_heart.app.ports.gpu_worker import GpuWorkerDevice
from puripuly_heart.app.ports.provider_verifier import ProviderVerifierPort
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
from puripuly_heart.app.services.desktop_overlay_application import (
    DESKTOP_INTERACTION_MODE_EDIT,
    DesktopOverlayApplicationOwner,
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
from puripuly_heart.app.services.local_asr_diagnostics import LocalASRDiagnosticsOwner
from puripuly_heart.app.services.local_asr_gpu_provisioning import (
    LocalASRGpuProvisioningDiagnostic,
)
from puripuly_heart.app.services.local_asr_selection import (
    LOCAL_CPU_PROVIDERS,
    resolve_local_asr_selection,
)
from puripuly_heart.app.services.managed_auth import ManagedAuthOwner
from puripuly_heart.app.services.managed_usage import (
    ManagedUsageOwner,
    ManagedUsageViewState,
)
from puripuly_heart.app.services.manual_local_asr_fallback import (
    ManualLocalASRFallbackOwner,
)
from puripuly_heart.app.services.manual_typing import ManualTypingOwner
from puripuly_heart.app.services.openrouter_pkce_flow import (
    OpenRouterPkceApplicationOwner,
    OpenRouterPkceFlowOwner,
)
from puripuly_heart.app.services.overlay_application import (
    OverlayApplicationOwner,
    OverlayApplicationState,
)
from puripuly_heart.app.services.overlay_calibration_application import (
    OverlayCalibrationApplicationOwner,
)
from puripuly_heart.app.services.provider_credential_verification import (
    ProviderCredentialVerificationInteractionOwner,
)
from puripuly_heart.app.services.provider_runtime_apply import (
    ProviderRuntimeApplyPlan,
)
from puripuly_heart.app.services.provider_settings import (
    ProviderSettingsOwner,
    provider_verification_context,
)
from puripuly_heart.app.services.provider_verification_binding import (
    ProviderVerificationBindingOwner,
)
from puripuly_heart.app.services.self_capture_application import (
    SelfCaptureApplicationOwner,
    SelfCaptureApplicationSettings,
)
from puripuly_heart.app.services.settings_application import SettingsApplicationOwner
from puripuly_heart.app.services.settings_mutation import SettingsMutationService
from puripuly_heart.app.services.settings_projection import (
    SettingsProjectionOwner,
    SettingsViewSettingsChange,
)
from puripuly_heart.app.services.settings_runtime_effects import (
    SettingsRuntimeEffectsAdapter,
)
from puripuly_heart.app.services.translation_enable import TranslationEnableOwner
from puripuly_heart.app.services.vrc_mic_sync import VrcMicSyncOwner
from puripuly_heart.app.wiring import (
    LocalASRProviderRuntimeFactory,
    ManagedSTTProviderFactory,
    build_peer_capture_session_config,
    build_peer_stt_provider_request,
    build_peer_stt_runtime_signature,
    build_self_capture_session_config,
    build_self_stt_provider_request,
    build_self_stt_runtime_signature,
    copy_stable_secrets_to_vnext_namespace,
    create_local_asr_provisioning_owner,
    create_provider_verifier,
    create_secret_store,
    create_sync_secret_store_adapter,
    resolve_overlay_config,
)
from puripuly_heart.app.wiring_capture_runtime import (
    CaptureDiagnosticsAdapter,
)
from puripuly_heart.app.wiring_composition import (
    create_desktop_overlay_policy,
    create_gpu_provider_recovery_application_owner,
    create_gpu_runtime_interaction_owner,
    create_local_asr_diagnostics_owner,
    create_manual_typing_owner,
    create_provider_credential_verification_interaction_owner,
    create_vrchat_osc_presence_probe_owner,
    create_windows_desktop_work_area,
)
from puripuly_heart.app.wiring_local_asr_application import (
    LocalASRApplicationRuntime,
    compose_local_asr_application,
)
from puripuly_heart.app.wiring_microphone_test import MicrophoneTestRuntime
from puripuly_heart.app.wiring_peer_application import (
    PeerApplicationRuntime,
    compose_peer_application,
)
from puripuly_heart.app.wiring_provider_runtime import (
    ProviderRuntimeSignatures,
)
from puripuly_heart.app.wiring_runtime_composition import RuntimeCompositionComponents
from puripuly_heart.app.wiring_runtime_pipeline import (
    RuntimePipelineComponents,
)
from puripuly_heart.app.wiring_vrc_mic_sync import compose_vrc_mic_sync
from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.config.paths import user_config_dir
from puripuly_heart.config.settings import (
    OVERLAY_TARGET_STEAMVR,
    AppSettings,
    LLMProviderName,
    OpenRouterCredentialSource,
    QwenLLMModel,
    STTProviderName,
    build_managed_openrouter_byok_target_settings,
    with_telemetry_consent,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.audio.gate import VrcMicAudioGate
from puripuly_heart.core.audio.process_identity import PsutilCurrentUserProcessSnapshots
from puripuly_heart.core.clipboard.watcher import create_clipboard_watcher
from puripuly_heart.core.clock import SystemClock
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
)
from puripuly_heart.core.local_asr_provisioning import (
    LocalASRProvisioningDiagnostic,
    LocalASRProvisioningPort,
    LocalASRProvisioningSnapshot,
)
from puripuly_heart.core.local_gpu_assets import local_gpu_model_path
from puripuly_heart.core.messages import (
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
    TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    TransactionResult,
)
from puripuly_heart.core.orchestrator.hub import ClientHub
from puripuly_heart.core.osc.chatbox_paginator import ChatboxPaginator
from puripuly_heart.core.osc.receiver import (
    VrcMicState,
)
from puripuly_heart.core.osc.udp_sender import VrchatOscUdpSender
from puripuly_heart.core.peer_capture import (
    PeerCaptureSessionSnapshot,
)
from puripuly_heart.core.runtime.gpu_asr import GpuASRChannel
from puripuly_heart.core.runtime.logging import RuntimeLoggingService
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
    SelfCaptureSessionSnapshot,
    SelfCaptureSessionState,
)
from puripuly_heart.core.stt.controller import FinalTranscriptSuppressedNotification
from puripuly_heart.core.telemetry import (
    TranslationSuccessTelemetryResult,
    TranslationSuccessTelemetryService,
)
from puripuly_heart.core.translation_policy import FIXED_TRANSLATION_POLICY

logger = logging.getLogger(__name__)

# Hardcoded STT session reset deadline (not configurable via settings)
STT_RESET_DEADLINE_S = 300.0
OVERLAY_STARTUP_TIMEOUT_MS = 3000
OVERLAY_SHUTDOWN_GRACE_S = 0.05
MANUAL_INPUT_TYPING_IDLE_TIMEOUT_S = 3.0
MANUAL_SUBMIT_TYPING_TIMEOUT_S = 10.0
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
_MICROPHONE_TEST_LEVEL_INTERVAL_S = 1.0
LOCAL_QWEN_HALLUCINATION_GUIDANCE_TRIGGER_COUNT = 2


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


def _settings_mutation_committed(result: TransactionResult) -> bool:
    return result.status in {
        TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
        TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
    }


@dataclass(slots=True)
class GuiController:
    page: object
    app: UiPresentationPort
    config_path: Path
    allow_stable_settings_import: bool = False
    runtime_logging_sinks: RuntimeLoggingSinks | None = field(default=None, repr=False)
    settings_mutation_service: SettingsMutationService | None = None
    provider_verifier: ProviderVerifierPort | None = None
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

    clock: SystemClock = SystemClock()
    _runtime_composition: RuntimeCompositionComponents | None = field(
        init=False,
        default=None,
        repr=False,
    )

    sender: VrchatOscUdpSender | None = None
    osc: ChatboxPaginator | None = None
    hub: ClientHub | None = None
    _self_capture_owner: SelfCaptureSessionOwner | None = field(init=False, default=None)
    _capture_diagnostics_adapter: CaptureDiagnosticsAdapter | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _provider_runtime_signatures: ProviderRuntimeSignatures = field(
        init=False,
        default_factory=ProviderRuntimeSignatures,
        repr=False,
    )
    _settings_application_owner: SettingsApplicationOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _peer_application_runtime: PeerApplicationRuntime | None = field(
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
    _manual_typing_owner: ManualTypingOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _microphone_test_runtime: MicrophoneTestRuntime | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _debug_capture_fault_profile: str = field(init=False, default="none")
    _debug_stt_fault_profile: str = field(init=False, default="none")
    _self_capture_application_owner: SelfCaptureApplicationOwner | None = field(
        init=False,
        default=None,
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
    _local_asr_application_runtime: LocalASRApplicationRuntime | None = field(
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
    _desktop_overlay_application_owner: DesktopOverlayApplicationOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _provider_credential_verification_owner: (
        ProviderCredentialVerificationInteractionOwner | None
    ) = field(init=False, default=None, repr=False)
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

    _overlay_calibration_application_owner: OverlayCalibrationApplicationOwner | None = field(
        init=False,
        default=None,
        repr=False,
    )

    @property
    def runtime_composition(self) -> RuntimeCompositionComponents:
        components = self._runtime_composition
        if components is None:
            raise RuntimeError("GuiController runtime composition is not installed")
        return components

    def install_runtime_composition(
        self,
        components: RuntimeCompositionComponents,
    ) -> None:
        current = self._runtime_composition
        if current is not None and current is not components:
            raise RuntimeError("GuiController runtime composition is already installed")
        self._runtime_composition = components

    @property
    def overlay_calibration(self) -> OverlayCalibration:
        return self._get_overlay_calibration_application_owner().current

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

    @property
    def _last_stt_runtime_signature(self) -> tuple[object, ...] | None:
        return self._provider_runtime_signatures.last_self_runtime

    @_last_stt_runtime_signature.setter
    def _last_stt_runtime_signature(self, signature: tuple[object, ...] | None) -> None:
        self._provider_runtime_signatures.last_self_runtime = signature

    @property
    def _last_self_stt_runtime_signature(self) -> tuple[object, ...] | None:
        return self._provider_runtime_signatures.last_self_runtime

    @_last_self_stt_runtime_signature.setter
    def _last_self_stt_runtime_signature(
        self,
        signature: tuple[object, ...] | None,
    ) -> None:
        self._provider_runtime_signatures.last_self_runtime = signature

    @property
    def _last_self_stt_provider_signature(self) -> tuple[object, ...] | None:
        return self._provider_runtime_signatures.last_self_provider

    @_last_self_stt_provider_signature.setter
    def _last_self_stt_provider_signature(
        self,
        signature: tuple[object, ...] | None,
    ) -> None:
        self._provider_runtime_signatures.last_self_provider = signature

    @property
    def _last_llm_provider_signature(self) -> tuple[object, ...] | None:
        return self._provider_runtime_signatures.last_llm_provider

    @_last_llm_provider_signature.setter
    def _last_llm_provider_signature(
        self,
        signature: tuple[object, ...] | None,
    ) -> None:
        self._provider_runtime_signatures.last_llm_provider = signature

    @property
    def vnext_settings(self) -> AppSettingsVNext | None:
        return self._get_settings_owner().canonical

    @vnext_settings.setter
    def vnext_settings(self, settings: AppSettingsVNext | None) -> None:
        self._get_settings_owner().canonical = settings

    @property
    def managed_auth_pending(self) -> bool:
        components = self._runtime_composition
        return components.managed_account.auth.pending if components is not None else False

    @property
    def last_discord_managed_auth_referral_bonus_applied(self) -> bool:
        components = self._runtime_composition
        return (
            components.managed_account.auth.last_referral_bonus_applied
            if components is not None
            else False
        )

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
            desktop = self._get_desktop_overlay_application_owner()
            calibration = self._get_overlay_calibration_application_owner()
            owner = OverlayApplicationOwner(
                state_provider=self._overlay_application_state,
                config_provider=lambda: resolve_overlay_config(cast(AppSettings, self.settings)),
                overlay_intent_sink=lambda enabled: setattr(
                    cast(AppSettings, self.settings).ui,
                    "overlay_enabled",
                    enabled,
                ),
                hub_provider=lambda: self.hub,
                peer_snapshot_provider=lambda: (
                    self._get_peer_application_runtime().owner.snapshot()
                ),
                disable_peer_intent=lambda: (
                    self._get_peer_application_runtime().owner.disable_for_overlay()
                ),
                sync_peer_effective=lambda: (
                    self._get_peer_application_runtime().owner.sync_effective_flags()
                ),
                refresh_peer_dependencies=self._refresh_overlay_runtime_dependencies,
                presentation_sink=self.app.refresh_overlay_peer_contract,
                state_sink=self._report_overlay_application_state,
                fallback_notice_sink=(self.app.set_dashboard_overlay_session_fallback_notice),
                cancel_bounds_persistence=desktop.bounds_owner.cancel,
                clear_bounds_suppressed=desktop.bounds_owner.clear_suppressed,
                calibration_provider=lambda: calibration.current.copy(),
                logging_mode_provider=lambda: self.runtime_logging_mode,
                log_dir_provider=lambda: str(user_config_dir()),
                desktop_controls_factory=desktop.initial_controls,
                interaction_mode_sink=desktop.set_interaction_mode,
                bounds_control_sink=desktop.bounds_owner.track_apply_control,
                renderer_event_consumer=desktop.consume_renderer_events,
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
        return self._get_desktop_overlay_application_owner().captions_locked

    def _get_desktop_overlay_application_owner(self) -> DesktopOverlayApplicationOwner:
        owner = self._desktop_overlay_application_owner
        if owner is None:
            owner = DesktopOverlayApplicationOwner(
                settings=self._get_settings_owner(),
                settings_application_provider=self._get_settings_application_owner,
                overlay_provider=self._get_overlay_application_owner,
                work_area=create_windows_desktop_work_area(),
                policy=create_desktop_overlay_policy(),
                presentation_sink=lambda mode, locked: (
                    self.app.on_desktop_overlay_state_changed(
                        interaction_mode=mode,
                        captions_locked=locked,
                    )
                ),
                log_detailed=lambda message, level, exception: self.log_detailed(
                    message,
                    level=level,
                    exception=exception,
                ),
            )
            self._desktop_overlay_application_owner = owner
        return owner

    def _get_overlay_calibration_application_owner(
        self,
    ) -> OverlayCalibrationApplicationOwner:
        owner = self._overlay_calibration_application_owner
        if owner is None:
            owner = OverlayCalibrationApplicationOwner(
                settings=self._get_settings_owner(),
                settings_application_provider=self._get_settings_application_owner,
                overlay_provider=self._get_overlay_application_owner,
                schedule_task=lambda task: self.app.schedule_task(task),
                log_detailed=self.log_detailed,
                ingress_available=lambda: not self._shutdown_ingress_frozen,
            )
            self._overlay_calibration_application_owner = owner
        return owner

    def _get_peer_application_runtime(self) -> PeerApplicationRuntime:
        runtime = self._peer_application_runtime
        if runtime is None:
            runtime = compose_peer_application(
                settings_provider=lambda: self.settings,
                settings_owner=self._get_settings_owner(),
                canonical_settings=self._canonical_vnext_settings_for,
                hub_provider=lambda: self.hub,
                overlay_provider=self._get_overlay_application_owner,
                ingress_frozen=lambda: self._shutdown_ingress_frozen,
                persist_manual_fallback=lambda: (
                    self._get_settings_application_owner().persist_manual_fallback(channel="peer")
                ),
                ensure_local_ready=lambda generation: (
                    self._get_local_asr_application_runtime().ensure_peer_ready(
                        activation_generation=generation
                    )
                ),
                clear_cpu_pending=lambda: (
                    self._get_local_asr_application_runtime().cpu_repair.reset_peer()
                ),
                clear_gpu_pending=lambda: self._get_gpu_runtime_interaction_owner().clear_pending(
                    "peer"
                ),
                clear_switched_pending=lambda: (
                    self._get_local_asr_application_runtime().cpu_repair.clear_if_provider_switched_away()
                ),
                sync_local_notice=lambda: (
                    self._get_local_asr_application_runtime().adapters.notice.sync()
                ),
                presentation_changed=self._refresh_overlay_peer_consumers,
                disclosure_sink=self._enqueue_peer_translation_disclosure,
                superseded_sink=lambda: self._provider_runtime_signatures.mark_superseded(
                    cast(AppSettings, self.settings)
                ),
                localize=self.app.localize,
                settings_presentation_sink=self.app.refresh_settings_loopback_capture_target,
                log_basic=self.log_basic,
                log_detailed=self.log_detailed,
            )
            self._peer_application_runtime = runtime
        return runtime

    def _effective_peer_translation_enabled_for(self, settings: AppSettings) -> bool:
        runtime = self._get_peer_application_runtime()
        return runtime.owner.effective_enabled(runtime.state_for(settings))

    def _peer_translation_eula_accepted_for(self, settings: AppSettings) -> bool:
        return bool(settings.ui.peer_translation_eula_accepted)

    def _peer_translation_activation_requested_for(self, settings: AppSettings) -> bool:
        return self._get_peer_application_runtime().owner.activation_requested(
            intent_enabled=settings.ui.peer_translation_enabled,
            eula_accepted=settings.ui.peer_translation_eula_accepted,
        )

    def _effective_peer_overlay_enabled_for(self, settings: AppSettings) -> bool:
        _ = settings
        return self._get_overlay_application_owner().snapshot.state == "connected"

    def _effective_integrated_context_enabled_for(self, settings: AppSettings) -> bool:
        return self._effective_peer_translation_enabled_for(settings)

    def _sync_effective_hub_flags(self, settings: AppSettings | None = None) -> None:
        resolved_settings = settings or self.settings
        if resolved_settings is None:
            return
        runtime = self._get_peer_application_runtime()
        runtime.owner.sync_effective_flags(runtime.state_for(resolved_settings))

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
        await self._get_peer_application_runtime().owner.refresh_runtime(
            stop_mode="release" if peer_stop_mode == "release" else "retain"
        )
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
            self.manual_local_asr_fallback_owner.state(
                loaded_settings,
                cpu_auto_available=provisioning.snapshot.cpu_auto_available,
            )
        )
        fallback_channels = fallback_plan.fallback_channels
        installation_fallback = fallback_plan.installation_fallback
        if fallback_plan.changed:
            normalized_settings = self.manual_local_asr_fallback_owner.apply(
                loaded_settings,
                fallback_plan,
            )
            self.settings = normalized_settings
            if not self._get_settings_owner().save_current(
                failure_sink=lambda exc: self._log_error(f"Failed to save settings: {exc}")
            ):
                self.settings = loaded_settings
                fallback_channels = ()
            else:
                loaded_settings.provider.stt = normalized_settings.provider.stt
                loaded_settings.provider.peer_stt = normalized_settings.provider.peer_stt
                self.settings = loaded_settings
        self.settings.ui.overlay_enabled = False
        self.settings.ui.peer_translation_enabled = False
        self._get_overlay_calibration_application_owner().sync_from_settings(self.settings)
        self._get_overlay_calibration_application_owner().replace_draft(None)
        self.app.set_locale(self.settings.ui.locale)
        self._sync_ui_from_settings()
        self._get_settings_application_owner().notify_fallback(
            fallback_channels,
            installation_fallback,
        )
        with contextlib.suppress(Exception):
            apply_locale = getattr(self.app, "apply_locale", None)
            if callable(apply_locale):
                apply_locale()

        runtime_logging = self.runtime_logging
        runtime_logging.set_mode(SessionLoggingMode.BASIC)

        self.app.attach_runtime_log_sink(runtime_logging)

        self._get_local_asr_provisioning_owner()
        self._sync_signature_caches(self.settings)
        await self.runtime_composition.pipeline_launcher.launch(
            self.settings,
            vrc_mic_state=self.vrc_mic_state,
            vrc_mic_audio_gate=self.vrc_mic_audio_gate,
            receiver_active=self.receiver is not None,
        )
        self._get_local_asr_application_runtime().adapters.notice.sync()

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
        self._get_local_asr_application_runtime().adapters.notice.sync()

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

    def _get_managed_auth_owner(self) -> ManagedAuthOwner:
        return self.runtime_composition.managed_account.auth

    def _set_managed_trial_pending_auth(self, pending: bool) -> None:
        self._get_managed_auth_owner().set_pending(pending)

    def clear_managed_auth_pending_state(self) -> None:
        self._get_managed_auth_owner().clear_pending()

    def _get_translation_enable_owner(self) -> TranslationEnableOwner:
        return self.runtime_composition.managed_account.translation

    def _managed_openrouter_selected(self) -> bool:
        return self.runtime_composition.managed_account.release.selected()

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
                transaction_result_sink=self._get_settings_application_owner().results.set,
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
            self.runtime_composition.managed_account.release.telemetry_client,
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
            return self._get_settings_application_owner().results.committed()

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

    def _apply_managed_usage_view_state(self, state: ManagedUsageViewState) -> None:
        self.app.set_settings_managed_key_state(
            visible=state.visible,
            remaining_percent=state.remaining_percent,
            referral_id=state.referral_id,
            pass_status=state.pass_status,
        )

    def _get_managed_usage_owner(self) -> ManagedUsageOwner:
        return self.runtime_composition.managed_account.usage

    def _on_managed_trial_delegate_ready(self) -> None:
        self._get_managed_usage_owner().delegate_ready()

    async def _refresh_managed_trial_usage_state_best_effort(self) -> None:
        await self._get_managed_usage_owner().refresh_best_effort()

    def _sync_signature_caches(self, settings: AppSettings) -> None:
        self._provider_runtime_signatures.sync(
            settings,
            canonical=self._canonical_vnext_settings_for(settings),
            peer=self._get_peer_application_runtime().owner,
        )
        self._sync_non_provider_runtime_signatures(settings)

    def _sync_non_provider_runtime_signatures(self, settings: AppSettings) -> None:
        self._last_microphone_test_audio_settings_signature = (
            self._microphone_test_audio_settings_signature(settings)
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
        runtime = self._get_peer_application_runtime()
        return runtime.owner.desired_active(runtime.state_for(settings))

    def _active_local_asr_change(
        self,
        base_settings: AppSettings,
        next_settings: AppSettings,
    ) -> bool:
        local_providers = {*LOCAL_CPU_PROVIDERS, STTProviderName.LOCAL_QWEN_GPU.value}
        self_owner = self._self_capture_owner
        self_changed = (
            self_owner is not None
            and self_owner.snapshot.desired_active
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

    def _is_qwen_llm(self, settings: object) -> bool:
        # Boundary accessor used by the app-service runtime-apply adapters to
        # keep ``LLMProviderName`` settings-shape knowledge inside the
        # controller instead of leaking it across the app-service boundary.
        return bool(
            isinstance(settings, AppSettings) and settings.provider.llm == LLMProviderName.QWEN
        )

    async def set_desktop_overlay_captions_locked(self, locked: bool) -> None:
        await self._get_desktop_overlay_application_owner().set_captions_locked(locked)

    async def set_desktop_overlay_size_preset(self, size_preset: str) -> None:
        await self._get_desktop_overlay_application_owner().set_size_preset(size_preset)

    async def reset_desktop_overlay_position(self) -> None:
        await self._get_desktop_overlay_application_owner().reset_position()

    def _on_peer_capture_state_changed(
        self,
        snapshot: PeerCaptureSessionSnapshot,
    ) -> None:
        self._get_peer_application_runtime().owner.on_runtime_state_changed(snapshot)

    async def _close_peer_runtime_for_release(self, failures: list[Exception]) -> None:
        runtime = self._peer_application_runtime
        if runtime is None:
            return
        try:
            await runtime.owner.close()
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
                owner_name="RuntimePipelineLauncher",
                callback_name="close_failed_resources",
                callback=self._close_runtime_pipeline_launcher,
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
        self_owner = self._self_capture_owner
        if self_owner is not None:
            self_owner.invalidate_intent()
        peer_runtime = self._peer_application_runtime
        if peer_runtime is not None:
            peer_runtime.owner.stop_ingress()
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
        composition = self._runtime_composition
        managed = composition.managed_account if composition is not None else None
        if managed is not None:
            managed.usage.stop_ingress()
            managed.auth.stop_ingress()
            managed.translation.stop_ingress()

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
        composition = self._runtime_composition
        managed = composition.managed_account if composition is not None else None
        if managed is not None:
            await managed.usage.close()

    async def _close_managed_auth_owner(self) -> None:
        composition = self._runtime_composition
        managed = composition.managed_account if composition is not None else None
        if managed is not None:
            await managed.auth.close()

    async def _close_translation_enable_owner(self) -> None:
        composition = self._runtime_composition
        managed = composition.managed_account if composition is not None else None
        if managed is not None:
            await managed.translation.close()

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
        await owner.shutdown(preserve_failure_reason=True)
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

    async def _close_runtime_pipeline_launcher(self) -> None:
        composition = self._runtime_composition
        if composition is None:
            return
        await composition.pipeline_launcher.close()

    def _close_sender(self) -> None:
        sender = self.sender
        if sender is None:
            self.osc = None
            return
        sender.close()
        self.sender = None
        self.osc = None

    async def _close_managed_openrouter_release_service(self) -> None:
        composition = self._runtime_composition
        managed = composition.managed_account if composition is not None else None
        if managed is not None:
            await managed.release.close()

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
        await self._get_peer_application_runtime().owner.set_enabled(enabled)

    async def retry_peer_process_capture(self) -> bool:
        return await self._get_peer_application_runtime().owner.retry_process_capture()

    def _enqueue_peer_translation_disclosure(self) -> None:
        hub = self.hub
        if hub is None:
            return
        enqueue_disclosure = getattr(hub, "enqueue_peer_translation_disclosure", None)
        if callable(enqueue_disclosure):
            enqueue_disclosure(self.app.localize("peer_translation.disclosure"))

    def begin_overlay_calibration(self) -> OverlayCalibration:
        return self._get_overlay_calibration_application_owner().begin()

    def set_overlay_calibration_field(
        self,
        field_name: str,
        value: object,
    ) -> OverlayCalibration:
        return self._get_overlay_calibration_application_owner().set_field(
            field_name,
            value,
        )

    def apply_overlay_calibration(self) -> OverlayCalibration:
        return self._get_overlay_calibration_application_owner().apply()

    def cancel_overlay_calibration(self) -> OverlayCalibration:
        return self._get_overlay_calibration_application_owner().cancel()

    async def set_translation_enabled(self, enabled: bool) -> bool:
        return await self._get_translation_enable_owner().set_enabled(enabled)

    @property
    def _stt_restart_requested(self) -> bool:
        return self._get_self_capture_application_owner().restart_requested

    @_stt_restart_requested.setter
    def _stt_restart_requested(self, requested: bool) -> None:
        self._get_self_capture_application_owner().restart_requested = requested

    def _get_self_capture_application_owner(self) -> SelfCaptureApplicationOwner:
        owner = self._self_capture_application_owner
        if owner is None:
            owner = SelfCaptureApplicationOwner(
                settings_provider=lambda: (
                    SelfCaptureApplicationSettings(
                        config=build_self_capture_session_config(self.settings),
                        provider_id=self.settings.provider.stt.value,
                        qwen_region=self.settings.qwen.region.value,
                    )
                    if self.settings is not None
                    else None
                ),
                runtime_available=lambda: self.hub is not None,
                capture_owner=lambda: self.runtime_composition.self_capture_owner(),
                capture_owner_if_created=lambda: self._self_capture_owner,
                persist_manual_fallback=lambda: (
                    self._get_settings_application_owner().persist_manual_fallback(channel="self")
                ),
                reset_local_pending=lambda: (
                    self._get_local_asr_application_runtime().cpu_repair.reset_self()
                ),
                clear_gpu_pending=lambda: self._get_gpu_runtime_interaction_owner().clear_pending(
                    "self"
                ),
                overlay_state_provider=lambda: (
                    self._get_overlay_application_owner().snapshot.state
                ),
                mark_promo_eligible=lambda: (
                    self.hub.mark_promo_eligible() if self.hub is not None else None
                ),
                dashboard_enabled_sink=self.app.set_dashboard_stt_enabled,
                dashboard_needs_key_sink=self.app.set_dashboard_stt_needs_key,
                dashboard_needs_key=lambda available: self._dashboard_stt_needs_key(
                    stt_available=available
                ),
                state_sink=self._on_self_capture_state_changed,
                sync_effective_flags=self._sync_effective_hub_flags,
                sync_local_notice=lambda: (
                    self._get_local_asr_application_runtime().adapters.notice.sync()
                ),
                log_basic=self.log_basic,
                log_detailed=lambda message, level: self.log_detailed(
                    message,
                    level=level,
                ),
            )
            self._self_capture_application_owner = owner
        return owner

    async def set_stt_enabled(self, enabled: bool, *, force_immediate: bool = False) -> None:
        await self._get_self_capture_application_owner().set_enabled(
            enabled,
            force_immediate=force_immediate,
        )

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

    def _peer_local_stt_requested(self, settings: AppSettings | None = None) -> bool:
        runtime = self._get_peer_application_runtime()
        return runtime.owner.local_stt_requested(runtime.state_for(settings))

    def _get_local_asr_application_runtime(self) -> LocalASRApplicationRuntime:
        runtime = self._local_asr_application_runtime
        if runtime is None:
            runtime = compose_local_asr_application(
                settings_provider=lambda: self.settings,
                hub_provider=lambda: self.hub,
                self_capture_provider=lambda: self._self_capture_owner,
                peer_provider=lambda: self._get_peer_application_runtime().owner,
                peer_requested=self._peer_local_stt_requested,
                peer_activation_requested=self._peer_translation_activation_requested_for,
                provisioning_provider=self._get_local_asr_provisioning_owner,
                gpu_state_provider=lambda: (
                    self._get_gpu_runtime_interaction_owner().snapshot.ui_state
                ),
                retain_gpu_pending=lambda channel: (
                    self._get_gpu_runtime_interaction_owner().retain_pending(
                        cast(GpuASRChannel, channel)
                    )
                ),
                validate_gpu_activation=self._validate_gpu_activation,
                dashboard_enabled_sink=self.app.set_dashboard_stt_enabled,
                dashboard_needs_key_sink=self.app.set_dashboard_stt_needs_key,
                message_sink=self._show_short_stt_message,
                notice_sink=self.app.set_dashboard_local_stt_notice,
                rebuild_self_provider=lambda: (
                    self.runtime_composition.provider_runtime.effects.rebuild_self_stt()
                ),
                resume_self=self._resume_self_after_local_asr_cpu_repair,
                resume_peer=self._resume_peer_after_local_asr_cpu_repair,
                persist_manual_fallback=lambda channel: (
                    self._get_settings_application_owner().persist_manual_fallback(
                        channel=cast(Literal["self", "peer"], channel)
                    )
                ),
                load_log_sink=self._get_local_asr_diagnostics_owner().log_load_result,
            )
            self._local_asr_application_runtime = runtime
        return runtime

    async def _resume_self_after_local_asr_cpu_repair(self) -> bool:
        snapshot = await self._get_self_capture_application_owner().run_switch(desired=True)
        self.app.set_dashboard_stt_enabled(
            bool(
                snapshot is not None
                and snapshot.desired_active
                and snapshot.state is not SelfCaptureSessionState.FAULTED
                and snapshot.has_loop_task
            )
        )
        self._get_local_asr_application_runtime().adapters.notice.sync()
        return True

    async def _resume_peer_after_local_asr_cpu_repair(self) -> None:
        await self._refresh_overlay_runtime_dependencies()

    async def _close_local_asr_provisioning(self) -> None:
        runtime = self._local_asr_application_runtime
        if runtime is not None:
            await runtime.close()
        elif self.local_asr_provisioning is not None:
            await self.local_asr_provisioning.close()

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
        await self._get_settings_application_owner().apply_language_selection(change)

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

    def _get_settings_application_owner(self) -> SettingsApplicationOwner:
        owner = self._settings_application_owner
        if owner is None:
            owner = SettingsApplicationOwner(
                settings=self._get_settings_owner(),
                projection=self._settings_projection(),
                runtime_effects=SettingsRuntimeEffectsAdapter(
                    self,
                    desktop_overlay=self._get_desktop_overlay_application_owner(),
                    calibration=self._get_overlay_calibration_application_owner(),
                    overlay=self._get_overlay_application_owner(),
                    overlay_state_provider=self._overlay_application_state,
                    peer=lambda: self._get_peer_application_runtime().owner,
                    self_capture=lambda: self._self_capture_owner,
                    clear_local_pending=lambda: (
                        self._get_local_asr_application_runtime().cpu_repair.clear_if_provider_switched_away()
                    ),
                    replace_self_stt=lambda smooth: (
                        self._get_self_capture_application_owner().replace_provider(
                            smooth_local=smooth
                        )
                    ),
                ),
                manual_fallback=self.manual_local_asr_fallback_owner,
                cpu_auto_available=lambda: (
                    self._get_local_asr_provisioning_owner().snapshot.cpu_auto_available
                ),
                inspect_cpu=self._get_local_asr_provisioning_owner().inspect_cpu,
                fallback_sink=lambda channels, installation_fallback: (
                    self._show_short_message(
                        "local_stt.installation_fallback_qwen"
                        if installation_fallback
                        else "local_stt.language_fallback_qwen"
                    )
                    if channels
                    else None
                ),
                sync_ui=self._sync_ui_from_settings,
                fallback_log_sink=self._log_manual_local_asr_fallbacks,
                mutation_service_provider=lambda: self.settings_mutation_service,
                consume_superseded_settings=self._consume_superseded_local_asr_settings,
                active_local_asr_change=self._active_local_asr_change,
                failure_sink=self._log_error,
            )
            self._settings_application_owner = owner
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

    async def apply_settings(self, settings: AppSettings) -> None:
        await self._get_settings_application_owner().apply(settings)

    async def apply_telemetry_consent(self, consent: str) -> AppSettings | None:
        if self.settings is None:
            return None
        await self.apply_settings(with_telemetry_consent(self.settings, consent))
        return self.settings

    async def verify_api_key(self, provider: str, key: str) -> tuple[bool, str]:
        return await self._get_provider_credential_verification_owner().verify(provider, key)

    async def apply_providers(
        self,
        settings: AppSettings | None = None,
        *,
        force_rebuild_llm: bool = False,
    ) -> bool:
        return await self.runtime_composition.provider_application.apply(
            settings,
            force_rebuild_llm=force_rebuild_llm,
        )

    def _consume_superseded_local_asr_settings(self, settings: AppSettings) -> bool:
        return self._provider_runtime_signatures.consume_superseded(settings)

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
                self_owner_factory=self.runtime_composition.self_capture_owner,
                peer_owner_provider=lambda: (self._get_peer_application_runtime().owner.runtime),
                self_state_sink=self._on_self_capture_state_changed,
                ensure_self_switch=lambda: (
                    self._get_self_capture_application_owner().run_switch()
                ),
                refresh_self=(self.runtime_composition.provider_runtime.effects.refresh_self_stt),
                refresh_peer=(self.runtime_composition.provider_runtime.effects.refresh_peer),
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
            self_desired=bool(
                self._self_capture_owner is not None
                and self._self_capture_owner.snapshot.desired_active
            ),
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
        peer_runtime = self._get_peer_application_runtime().owner.runtime
        if "peer" in channels and peer_runtime is not None:
            await peer_runtime.suspend_provider_consumer()

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

    async def persist_provider_secret_change(
        self,
        secret_key: str,
        value: str,
    ) -> bool:
        owner = self._get_provider_settings_owner()
        return await owner.change_secret(secret_key, value)

    def persist_api_key_verification(
        self,
        provider: str,
        key: str,
        success: bool,
    ) -> None:
        self._get_provider_settings_owner().persist_verification(provider, key, success)

    def clear_provider_verification(self, provider: str) -> None:
        self.persist_api_key_verification(provider, "", False)

    def _capture_runtime_signatures_before_canonical_mutation(self) -> None:
        if self.settings is None:
            return
        self._provider_runtime_signatures.capture_peer_before_canonical_mutation(
            self.settings,
            canonical=self._canonical_vnext_settings_for(self.settings),
            peer=self._get_peer_application_runtime().owner,
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
                results=self._get_settings_application_owner().results,
            )
            self.provider_settings_owner = owner
        return owner

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

    def loopback_capture_summary(self, settings: AppSettings | None = None) -> str:
        return self._get_peer_application_runtime().target.summary(settings)

    def list_loopback_capture_options(self) -> list[OptionItem]:
        return self._get_peer_application_runtime().target.options()

    def list_loopback_process_options(self) -> list[OptionItem]:
        return self._get_peer_application_runtime().target.process_options()

    def list_loopback_device_options(self) -> list[OptionItem]:
        return self._get_peer_application_runtime().target.device_options()

    def current_loopback_capture_option_value(self, settings: AppSettings | None = None) -> str:
        return self._get_peer_application_runtime().target.current_value(settings)

    async def apply_loopback_capture_option(self, value: str) -> None:
        await self._get_peer_application_runtime().target.apply(value)

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

    def _get_capture_diagnostics_adapter(self) -> CaptureDiagnosticsAdapter:
        adapter = self._capture_diagnostics_adapter
        if adapter is None:
            adapter = CaptureDiagnosticsAdapter(
                detailed_enabled=self._detailed_audio_diag_enabled,
                debug_allowed=self._debug_audio_fault_allowed,
                capture_fault_profile=lambda: self._debug_capture_fault_profile,
                log_detailed=lambda message: self.log_detailed(message),
            )
            self._capture_diagnostics_adapter = adapter
        return adapter

    def _apply_runtime_pipeline_components(
        self,
        pipeline: RuntimePipelineComponents,
    ) -> None:
        self.sender = pipeline.sender
        self.osc = pipeline.osc
        self.hub = pipeline.hub
        self.vrc_mic_state = pipeline.vrc_mic_state
        self.vrc_mic_audio_gate = pipeline.vrc_mic_audio_gate
        self._self_capture_owner = pipeline.self_capture

    def _get_openrouter_pkce_flow_owner(self) -> OpenRouterPkceFlowOwner:
        return self.runtime_composition.managed_account.pkce_flow

    def _get_openrouter_pkce_application_owner(self) -> OpenRouterPkceApplicationOwner:
        return self.runtime_composition.managed_account.pkce

    async def _close_oauth_runtime(self) -> None:
        composition = self._runtime_composition
        managed = composition.managed_account if composition is not None else None
        if managed is not None:
            await managed.pkce_flow.close()

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

    @property
    def _last_microphone_test_audio_settings_signature(
        self,
    ) -> tuple[object, ...] | None:
        runtime = self._microphone_test_runtime
        owner = runtime.owner_if_created if runtime is not None else None
        return owner.audio_signature if owner is not None else None

    @_last_microphone_test_audio_settings_signature.setter
    def _last_microphone_test_audio_settings_signature(
        self,
        signature: tuple[object, ...] | None,
    ) -> None:
        self._get_microphone_test_runtime().owner().audio_signature = signature

    @property
    def microphone_test_active(self) -> bool:
        runtime = self._microphone_test_runtime
        return runtime.active if runtime is not None else False

    def _get_microphone_test_runtime(self) -> MicrophoneTestRuntime:
        runtime = self._microphone_test_runtime
        if runtime is None:
            runtime = MicrophoneTestRuntime(
                settings_provider=lambda: self.settings,
                self_capture_provider=lambda: self._self_capture_owner,
                local_pending_provider=lambda: (
                    self._get_local_asr_application_runtime().self_pending
                ),
                disable_self_capture=lambda: self.set_stt_enabled(False),
                clock=self.clock,
                log_sink=self.log_basic,
                detailed_sink=lambda message, level, exception: self.log_detailed(
                    message,
                    level=level,
                    exception=exception,
                ),
                error_sink=self._log_error,
            )
            self._microphone_test_runtime = runtime
        return runtime

    @staticmethod
    def _microphone_test_audio_settings_signature(
        settings: AppSettings | None,
    ) -> tuple[object, ...] | None:
        return MicrophoneTestRuntime.audio_signature(settings)

    async def start_microphone_test(
        self,
        *,
        meter_callback: Callable[[float], object] | None = None,
        level_log_interval_s: float = _MICROPHONE_TEST_LEVEL_INTERVAL_S,
    ) -> bool:
        return await self._get_microphone_test_runtime().start(
            meter_callback=meter_callback,
            level_log_interval_s=level_log_interval_s,
        )

    async def stop_microphone_test(self) -> None:
        runtime = self._microphone_test_runtime
        if runtime is not None:
            await runtime.stop()

    async def stop_microphone_test_for_audio_settings_change(self) -> None:
        await self.stop_microphone_test()

    async def _close_microphone_test_runtime_for_release(
        self,
        cleanup_failures: list[Exception],
    ) -> None:
        runtime = self._microphone_test_runtime
        if runtime is None:
            return
        try:
            await runtime.close()
        except Exception as exc:
            cleanup_failures.append(exc)

    def _on_self_capture_state_changed(
        self,
        snapshot: SelfCaptureSessionSnapshot,
    ) -> None:
        _ = snapshot
        self._get_local_asr_application_runtime().adapters.notice.sync()

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
            owner = compose_vrc_mic_sync(
                state_provider=lambda: self.vrc_mic_state,
                gate_provider=lambda: self.vrc_mic_audio_gate,
                log_detailed=lambda message, level: self.log_detailed(
                    message,
                    level=level,
                ),
                error_sink=self._log_error,
            )
            self._vrc_mic_sync_owner = owner
        return owner

    async def _configure_vrc_mic_receiver(self, *, enabled: bool) -> None:
        await self._get_vrc_mic_sync_owner().configure(enabled=enabled)

    def reopen_openrouter_pkce_authorization_url(self) -> bool:
        composition = self._runtime_composition
        managed = composition.managed_account if composition is not None else None
        return managed.pkce_flow.reopen_authorization_url() if managed is not None else False

    def build_managed_openrouter_byok_target_settings(self) -> AppSettings | None:
        return build_managed_openrouter_byok_target_settings(self.settings)

    async def connect_openrouter_via_pkce(
        self,
        *,
        target_settings: AppSettings,
        launch_source: str,
    ) -> bool:
        return await self._get_openrouter_pkce_application_owner().connect(
            target_settings=target_settings,
            launch_source=launch_source,
        )

    def _maybe_show_founder_letter_after_pkce_failure(self, launch_source: str) -> None:
        if launch_source != "letter":
            return
        show_founder_letter_dialog = getattr(self.app, "show_founder_letter_dialog", None)
        if callable(show_founder_letter_dialog):
            with contextlib.suppress(Exception):
                show_founder_letter_dialog()

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
                    self._get_overlay_application_owner().current_bridge() is not None
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
            runtime = self._get_overlay_application_owner().runtime
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
        bridge = self._get_overlay_application_owner().current_bridge()
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
