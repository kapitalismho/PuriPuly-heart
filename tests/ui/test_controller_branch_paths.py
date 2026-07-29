from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Awaitable, Callable, Literal
from uuid import uuid4

import flet as ft
import numpy as np
import pytest

pytest.importorskip("flet")

from puripuly_heart.app import wiring_managed_auth_factory as managed_auth_runtime_module
from puripuly_heart.app.adapters import (
    settings_vnext_canonical_persistence as canonical_persistence_adapter_module,
)
from puripuly_heart.app.adapters.self_capture_source import SelfCaptureSourceAdapter
from puripuly_heart.app.adapters.self_capture_vad import SelfCaptureVadAdapter
from puripuly_heart.app.adapters.windows_desktop_work_area import (
    WindowsDesktopWorkAreaAdapter,
)
from puripuly_heart.app.language_selection import LanguageSelectionChange
from puripuly_heart.app.ports.microphone_test import (
    MicrophoneTestCaptureRequest,
    MicrophoneTestRuntimePort,
)
from puripuly_heart.app.ports.self_capture_admission import (
    SelfCaptureAdmissionEffect,
    SelfCaptureAdmissionEffectType,
)
from puripuly_heart.app.ports.settings_repository import SettingsCommitRequest
from puripuly_heart.app.services import (
    desktop_overlay_application as desktop_overlay_application_module,
)
from puripuly_heart.app.services import overlay_application as overlay_application_module
from puripuly_heart.app.services import (
    overlay_generation_start as overlay_generation_start_module,
)
from puripuly_heart.app.services import provider_runtime_apply as provider_runtime_apply_module
from puripuly_heart.app.services import provider_settings as provider_settings_module
from puripuly_heart.app.services import settings_mutation
from puripuly_heart.app.services.canonical_settings_persistence import (
    SettingsOwner,
    legacy_settings_snapshot_values,
)
from puripuly_heart.app.services.desktop_overlay_application import (
    DesktopOverlayApplicationOwner,
)
from puripuly_heart.app.services.managed_connection_auth import ManagedConnectionAuthService
from puripuly_heart.app.services.managed_usage import ManagedUsageOwner
from puripuly_heart.app.services.overlay_application import OverlayApplicationOwner
from puripuly_heart.app.services.settings_application import SettingsApplicationOwner
from puripuly_heart.app.wiring import (
    build_peer_capture_session_config,
    build_peer_stt_provider_request,
    build_peer_stt_provider_signature_from_vnext,
    build_peer_stt_runtime_signature,
    build_self_capture_session_config,
    build_self_stt_provider_signature,
    build_self_stt_runtime_signature,
    resolve_overlay_config,
)
from puripuly_heart.app.wiring_composition import create_desktop_overlay_policy
from puripuly_heart.config.audio_host_api import (
    WINDOWS_MME_HOST_API,
    WINDOWS_WASAPI_COMPATIBILITY_HOST_API,
    WINDOWS_WASAPI_HOST_API,
    normalize_input_host_api,
)
from puripuly_heart.config.prompts import load_prompt_for_provider
from puripuly_heart.config.settings import (
    OVERLAY_TARGET_DESKTOP,
    AppSettings,
    LLMProviderName,
    LocalLLMBackend,
    LocalLLMSettings,
    OpenRouterCredentialSource,
    OpenRouterLLMModel,
    OpenRouterProviderRouting,
    OpenRouterRoutingMode,
    OpenRouterSelectionAlias,
    ProviderSettings,
    QwenLLMModel,
    QwenRegion,
    SecretsBackend,
    STTProviderName,
    TranslationConnection,
    TranslationFallbackSettings,
    TranslationModel,
    TranslationSettings,
    save_settings,
    to_dict,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core import messages
from puripuly_heart.core.audio import source as audio_source_module
from puripuly_heart.core.audio.format import AudioFrameF32
from puripuly_heart.core.audio.gate import VrcMicAudioGate
from puripuly_heart.core.audio.source import (
    MicrophoneTestRouteObservation,
    SelfMicCaptureChannelDecision,
    SoundDeviceInputMetadata,
)
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.llm.provider import SemaphoreLLMProvider
from puripuly_heart.core.local_asr_provisioning import (
    LocalASRModelProvisioningState,
    LocalASRProvisioningSnapshot,
)
from puripuly_heart.core.local_stt_assets import (
    LOCAL_QWEN_GPU_MODEL_ID,
    LOCAL_STT_MODEL_ID,
    PARAKEET_JAPANESE_MODEL_ID,
    PARAKEET_V3_MODEL_ID,
    REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
)
from puripuly_heart.core.managed_openrouter_broker_client import (
    HttpManagedOpenRouterBrokerClient,
)
from puripuly_heart.core.managed_openrouter_release import (
    ManagedOpenRouterReleaseBehavior,
    ManagedOpenRouterReleaseDiagnostics,
    ManagedOpenRouterReleaseResult,
    ManagedOpenRouterReleaseService,
    ManagedOpenRouterStatusRefreshResult,
    TalkTogetherPassStatus,
    UnavailableManagedOpenRouterReleaseClient,
)
from puripuly_heart.core.openrouter_metadata import OpenRouterKeyMetadata
from puripuly_heart.core.openrouter_pkce import OpenRouterPKCEExchangeResult
from puripuly_heart.core.orchestrator.hub import ClientHub
from puripuly_heart.core.osc.chatbox_paginator import ChatboxPaginator
from puripuly_heart.core.osc.receiver import VrcMicState
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.sink import (
    OverlayEventAdapter,
    PeerTranscriptFinal,
    SelfTranscriptFinal,
    TranslationFinal,
)
from puripuly_heart.core.runtime.github_star_prompt import GithubStarPromptRuntime
from puripuly_heart.core.runtime.overlay import OverlayRuntimeHandle
from puripuly_heart.core.runtime.peer_channel import PeerRuntimeConfig
from puripuly_heart.core.runtime_logging import (
    RuntimeLoggingSinks,
    SessionLoggingMode,
    SessionRuntimeLoggingService,
)
from puripuly_heart.core.self_capture import (
    SelfCaptureAdmissionStatus,
    SelfCaptureFailureReason,
    SelfCaptureProviderStatus,
    SelfCaptureSessionConfig,
    SelfCaptureSessionSnapshot,
    SelfCaptureSessionState,
)
from puripuly_heart.core.stt.controller import FinalTranscriptSuppressedNotification
from puripuly_heart.domain.models import Transcript
from puripuly_heart.providers.llm.gemini import GeminiLLMProvider
from puripuly_heart.providers.llm.openrouter import OpenRouterLLMProvider
from puripuly_heart.providers.llm.qwen import QwenLLMProvider
from puripuly_heart.providers.stt.deepgram import DeepgramRealtimeSTTBackend
from puripuly_heart.providers.stt.soniox import SonioxRealtimeSTTBackend
from puripuly_heart.ui import controller as controller_module
from puripuly_heart.ui import presentation_adapter as presentation_adapter_module
from puripuly_heart.ui.app import TranslatorApp
from puripuly_heart.ui.controller import GuiController
from puripuly_heart.ui.i18n import set_locale, t
from puripuly_heart.ui.overlay_calibration import OverlayCalibration
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter
from tests.helpers.fakes import FakeSender

PEER_DISCLOSURE_KEY = "peer_translation.disclosure"


@pytest.fixture(autouse=True)
def _isolate_relative_settings_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)


class DummySecrets:
    def __init__(self, values: dict[str, str]):
        self._values = dict(values)
        self.set_calls: list[tuple[str, str]] = []
        self.delete_calls: list[str] = []

    def get(self, key: str) -> str | None:
        return self._values.get(key)

    def set(self, key: str, value: str) -> None:
        self.set_calls.append((key, value))
        self._values[key] = value

    def delete(self, key: str) -> None:
        self.delete_calls.append(key)
        self._values.pop(key, None)


class DummyDashboard:
    def __init__(self) -> None:
        self.translation_needs_key: bool | None = None
        self.translation_enabled: bool | None = None
        self.stt_needs_key: bool | None = None
        self.stt_enabled: bool | None = None
        self.stt_starting: bool | None = None
        self.stt_starting_calls: list[bool] = []
        self.local_stt_notice_status: str | None = None
        self.local_stt_notice_percent: int | None = None
        self.languages: tuple[str, str] | None = None
        self.recent_languages: tuple[list[str], list[str]] | None = None
        self.managed_trial_state: dict[str, object] | None = None
        self.managed_trial_calls: list[dict[str, object]] = []
        self.managed_auth_pending: bool | None = None
        self.managed_auth_pending_calls: list[bool] = []
        self.is_translation_on: bool = True
        self.on_recent_languages_change = None

    def set_translation_needs_key(self, value: bool) -> None:
        self.translation_needs_key = value

    def set_translation_enabled(self, value: bool) -> None:
        self.translation_enabled = value

    def set_stt_needs_key(self, value: bool) -> None:
        self.stt_needs_key = value

    def set_stt_enabled(self, value: bool) -> None:
        self.stt_enabled = value

    def set_stt_starting(self, value: bool) -> None:
        self.stt_starting = value
        self.stt_starting_calls.append(value)

    def set_local_stt_notice(self, status: str | None, percent: int | None = None) -> None:
        self.local_stt_notice_status = status
        self.local_stt_notice_percent = percent

    def set_languages_from_codes(
        self,
        source: str,
        target: str,
        peer_source: str = "",
        peer_target: str = "",
    ) -> None:
        self.languages = (source, target, peer_source, peer_target)

    def set_recent_languages(self, source: list[str], target: list[str]) -> None:
        self.recent_languages = (source, target)

    def set_peer_auto_detect_available(self, available: bool) -> None:
        pass

    def set_managed_trial_state(self, **state: object) -> None:
        self.managed_trial_calls.append(dict(state))
        self.managed_trial_state = dict(state)

    def set_managed_auth_pending(self, pending: bool) -> None:
        self.managed_auth_pending = bool(pending)
        self.managed_auth_pending_calls.append(self.managed_auth_pending)


class DummySettingsView:
    def __init__(self) -> None:
        self.calls: list[tuple[AppSettings, Path, bool]] = []
        self.managed_trial_usage_state: dict[str, object] | None = None

    def load_from_settings(
        self,
        settings: AppSettings,
        *,
        config_path: Path,
        preserve_custom_vocab_draft: bool = False,
    ) -> None:
        self.calls.append((settings, config_path, preserve_custom_vocab_draft))

    def set_managed_trial_usage_state(
        self, *, visible: bool, remaining_percent: int | None = None
    ) -> None:
        self.managed_trial_usage_state = {
            "visible": visible,
            "remaining_percent": remaining_percent,
        }


class DummyLogsView:
    def __init__(self) -> None:
        self.logs: list[str] = []
        self.attach_calls = 0

    def append_log(self, message: str) -> None:
        self.logs.append(message)

    def attach_log_handler(self) -> None:
        self.attach_calls += 1


class RuntimeLoggingSpy:
    def __init__(
        self,
        *,
        detailed_enabled: bool = True,
        basic_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.mode = SessionLoggingMode.DETAILED if detailed_enabled else SessionLoggingMode.BASIC
        self.basic_messages: list[tuple[int, str]] = []
        self.detailed_messages: list[tuple[int, str]] = []
        self.shutdown_failure_summaries: list[tuple[str, ...]] = []
        self.close_calls = 0
        self.basic_error = basic_error
        self.close_error = close_error

    def emit_basic(self, message: str, *, level: int = logging.INFO) -> None:
        if self.basic_error is not None:
            raise self.basic_error
        self.basic_messages.append((level, message))

    def emit_detailed(self, message: str, *, level: int = logging.INFO) -> bool:
        if self.mode.value != "detailed":
            return False
        self.detailed_messages.append((level, message))
        return True

    def emit_detailed_lazy(
        self,
        build_message: Callable[[], str],
        *,
        level: int = logging.INFO,
    ) -> bool:
        if self.mode.value != "detailed":
            return False
        self.detailed_messages.append((level, build_message()))
        return True

    def attach_realtime_sink(self, sink) -> None:
        _ = sink

    def set_mode(self, mode) -> None:
        normalized = SessionLoggingMode(mode)
        self.mode = normalized

    def close_after_producers_stop(self, *, cleanup_failures=()) -> None:
        self.close_calls += 1
        self.shutdown_failure_summaries.append(
            tuple(type(failure).__name__ for failure in cleanup_failures)
        )
        if self.close_error is not None:
            raise self.close_error


class DummyHub:
    def __init__(
        self,
        *,
        llm: object | None = object(),
        stt: object | None = object(),
        peer_stt: object | None = None,
    ) -> None:
        self.llm = llm
        self.stt = stt
        self.peer_stt = peer_stt
        self.local_asr_provider_runtime = None
        self.translation_enabled = True
        self.peer_translation_enabled = False
        self.integrated_context_enabled = False
        self.source_language = "ko"
        self.target_language = "en"
        self.system_prompt = ""
        self.low_latency_mode = False
        self.low_latency_merge_gap_ms = 600
        self.low_latency_spec_retry_max = 10
        self.hangover_s = 1.1
        self.peer_hangover_s = 0.6
        self.clear_context_calls = 0
        self.promo_calls = 0
        self.replace_stt_calls: list[object | None] = []
        self.replace_peer_stt_calls: list[object | None] = []
        self.replace_stt_request_calls: list[tuple[object, bool]] = []
        self.drain_self_stt_calls: list[float | None] = []
        self.abort_self_stt_calls = 0
        self.warmup_stt_calls: list[str] = []
        self.replace_llm_calls: list[object | None] = []
        self.start_calls: list[bool] = []
        self.stop_calls = 0
        self.submit_calls: list[tuple[str, str]] = []
        self.submit_event = asyncio.Event()
        self.reset_overlay_preview_calls = 0
        self.clear_language_runtime_state_calls: list[str] = []
        self.clear_language_runtime_state_errors: dict[str, Exception] = {}
        self.ui_events: asyncio.Queue[object] = asyncio.Queue()
        self.output_runtime = DummyOutputRuntime()

    def clear_context(self) -> None:
        self.clear_context_calls += 1

    def mark_promo_eligible(self) -> None:
        self.promo_calls += 1

    async def start(self, *, auto_flush_osc: bool) -> None:
        self.start_calls.append(auto_flush_osc)

    async def stop(self) -> None:
        self.stop_calls += 1
        await self.output_runtime.close()

    async def submit_text(self, text: str, *, source: str) -> None:
        self.submit_calls.append((text, source))
        self.submit_event.set()

    async def reset_overlay_preview(self) -> None:
        self.reset_overlay_preview_calls += 1

    async def clear_language_runtime_state(self, *, channel: str) -> None:
        self.clear_language_runtime_state_calls.append(channel)
        if channel in self.clear_language_runtime_state_errors:
            raise self.clear_language_runtime_state_errors[channel]

    async def replace_stt_provider(self, stt: object | None) -> None:
        old_stt = self.stt
        self.replace_stt_calls.append(stt)
        if old_stt is not None and hasattr(old_stt, "close"):
            await old_stt.close()
        self.stt = stt

    def has_stt_provider(self, channel: str) -> bool:
        return self.stt is not None if channel == "self" else self.peer_stt is not None

    async def replace_stt_provider_request(self, request: object, *, start: bool):
        self.replace_stt_request_calls.append((request, start))
        self.stt = request
        return SimpleNamespace(status="applied")

    async def drain_self_stt_for_toggle_off(
        self,
        *,
        release_backend_after: float | None = None,
    ) -> None:
        self.drain_self_stt_calls.append(release_backend_after)

    async def abort_self_stt_for_toggle_off(self) -> None:
        self.abort_self_stt_calls += 1
        self.stt = None

    async def warmup_stt_channel(self, channel: str) -> None:
        self.warmup_stt_calls.append(channel)

    async def replace_peer_stt_provider(self, stt: object | None) -> None:
        old_stt = self.peer_stt
        self.replace_peer_stt_calls.append(stt)
        if old_stt is not None and hasattr(old_stt, "close"):
            await old_stt.close()
        self.peer_stt = stt

    async def replace_llm_provider(self, llm: object | None) -> None:
        old_llm = self.llm
        self.replace_llm_calls.append(llm)
        self.llm = llm
        if old_llm is not None and old_llm is not llm and hasattr(old_llm, "close"):
            await old_llm.close()


class DummyOutputRuntime:
    def __init__(self) -> None:
        self.started_bridges: list[object] = []
        self.bridge_tasks: list[asyncio.Task[object]] = []
        self.close_calls = 0

    def start_ui_event_bridge(self, bridge: object) -> asyncio.Task[object]:
        self.started_bridges.append(bridge)
        task = asyncio.create_task(bridge.run())  # type: ignore[attr-defined]
        self.bridge_tasks.append(task)
        return task

    async def wait_for_ui_event_bridge_started(self) -> None:
        bridge = self.started_bridges[-1]
        bridge_task = self.bridge_tasks[-1]
        wait_started = getattr(bridge, "wait_started", None)
        if not callable(wait_started):
            await asyncio.sleep(0)
            if bridge_task.done():
                await bridge_task
            return
        started_task = asyncio.create_task(
            wait_started(),
            name="OutputRuntime:ui-event-bridge-started-wait",
        )
        try:
            done, _ = await asyncio.wait(
                {bridge_task, started_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if bridge_task in done:
                await bridge_task
                raise RuntimeError("UI Event Bridge stopped before reporting started")
            await started_task
            if bridge_task.done():
                await bridge_task
                raise RuntimeError("UI Event Bridge stopped during startup")
        finally:
            if not started_task.done():
                started_task.cancel()
            await asyncio.gather(started_task, return_exceptions=True)

    async def close(self) -> None:
        self.close_calls += 1
        for task in self.bridge_tasks:
            if not task.done():
                task.cancel()
        if self.bridge_tasks:
            await asyncio.gather(*self.bridge_tasks, return_exceptions=True)
        for bridge in self.started_bridges:
            close = getattr(bridge, "close", None)
            if callable(close):
                close()


class FakeClipboardWatcher:
    def __init__(self, on_text: Callable[[str], None]) -> None:
        self.on_text = on_text
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def emit(self, text: str) -> None:
        self.on_text(text)


class DisclosureDummyHub(DummyHub):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.disclosures: list[str] = []

    def enqueue_peer_translation_disclosure(self, text: str) -> None:
        self.disclosures.append(text)


class DummyPeerRuntime:
    def __init__(self) -> None:
        self.policy_calls: list[dict[str, object]] = []
        self.closed = False
        self.warmup_calls = 0

    async def apply_policy(self, *, config: PeerRuntimeConfig, desired_active: bool) -> None:
        self.policy_calls.append({"config": config, "desired_active": desired_active})

    async def warmup(self) -> None:
        self.warmup_calls += 1

    async def close(self) -> None:
        self.closed = True


class RetryPeerRuntime(DummyPeerRuntime):
    def __init__(self, result: bool) -> None:
        super().__init__()
        self.result = result
        self.retry_configs: list[PeerRuntimeConfig] = []

    async def retry_process_capture(self, *, config: PeerRuntimeConfig) -> bool:
        self.retry_configs.append(config)
        return self.result


class DummyGate:
    def __init__(self) -> None:
        self.state = None
        self.enabled_calls: list[bool] = []
        self.receiver_active_calls: list[bool] = []
        self.reset_calls = 0

    def set_enabled(self, enabled: bool) -> None:
        self.enabled_calls.append(enabled)

    def set_receiver_active(self, active: bool) -> None:
        self.receiver_active_calls.append(active)

    def reset(self) -> None:
        self.reset_calls += 1


class DummyManagedReleaseService:
    def __init__(self, result: ManagedOpenRouterReleaseResult) -> None:
        self.result = result
        self.prepare_calls = 0
        self.prepare_referral_ids: list[str | None] = []
        self.close_calls = 0

    async def prepare_for_translation(
        self,
        *,
        referral_id: str | None = None,
    ) -> ManagedOpenRouterReleaseResult:
        self.prepare_calls += 1
        self.prepare_referral_ids.append(referral_id)
        return self.result

    async def close(self) -> None:
        self.close_calls += 1


class RecordingSettingsMutationService:
    def __init__(self, result: messages.TransactionResult | None = None) -> None:
        self.requests: list[settings_mutation.SettingsMutationRequest] = []
        self.result = result or messages.TransactionResult(
            status=messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
            message=None,
            diagnostics=None,
        )

    async def mutate(
        self,
        request: settings_mutation.SettingsMutationRequest,
    ) -> messages.TransactionResult:
        self.requests.append(request)
        return self.result


class InspectingManagedReleaseService(DummyManagedReleaseService):
    def __init__(
        self,
        result: ManagedOpenRouterReleaseResult,
        *,
        on_prepare: Callable[[], object] | None = None,
    ) -> None:
        super().__init__(result)
        self.on_prepare = on_prepare

    async def prepare_for_translation(
        self,
        *,
        referral_id: str | None = None,
    ) -> ManagedOpenRouterReleaseResult:
        self.prepare_calls += 1
        self.prepare_referral_ids.append(referral_id)
        if self.on_prepare is not None:
            prepare_result = self.on_prepare()
            if asyncio.iscoroutine(prepare_result):
                await prepare_result
        return self.result


class FailingManagedReleaseService(DummyManagedReleaseService):
    def __init__(self, exc: Exception) -> None:
        super().__init__(
            ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.RETRY,
                message_key="managed_release.retry",
            )
        )
        self.exc = exc

    async def prepare_for_translation(
        self,
        *,
        referral_id: str | None = None,
    ) -> ManagedOpenRouterReleaseResult:
        self.prepare_calls += 1
        self.prepare_referral_ids.append(referral_id)
        raise self.exc


class FakeOverlayBridge:
    instances: list["FakeOverlayBridge"] = []

    def __init__(self, *, session_token: str, initial_snapshot=None, **_kwargs) -> None:
        self.session_token = session_token
        self.initial_snapshot = initial_snapshot
        self.current_snapshot = initial_snapshot
        self.messages: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self.url = "ws://127.0.0.1:8765"
        self.desktop_runtime_controls_enabled = bool(
            _kwargs.get("desktop_runtime_controls_enabled", False)
        )
        self.started = False
        self.stopped = False
        self.snapshots: list[object] = []
        self.shutdown_calls = 0
        self.runtime_control_messages: list[str] = []
        self.desktop_runtime_control_payloads: list[dict[str, object]] = []
        self.initial_desktop_runtime_controls: list[dict[str, object]] = []
        self.__class__.instances.append(self)

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def replace_snapshot(self, snapshot: object) -> None:
        self.current_snapshot = snapshot
        self.snapshots.append(snapshot)

    async def broadcast_shutdown(self) -> None:
        self.shutdown_calls += 1

    async def broadcast_runtime_control(self, *, logging_mode: str) -> None:
        self.runtime_control_messages.append(logging_mode)

    async def broadcast_desktop_runtime_control(self, payload) -> None:
        self.desktop_runtime_control_payloads.append(dict(payload))

    def set_initial_desktop_runtime_controls(self, sequence) -> None:
        self.initial_desktop_runtime_controls = [dict(payload) for payload in sequence]

    def snapshot(self):
        return self.current_snapshot


class FakeOverlayProcessManager:
    instances: list["FakeOverlayProcessManager"] = []

    def __init__(
        self,
        *,
        bridge_url: str,
        bridge_messages: asyncio.Queue[dict[str, object]],
        session_token: str,
        locale: str,
        startup_timeout_ms: int,
        **_kwargs,
    ) -> None:
        self.bridge_url = bridge_url
        self.bridge_messages = bridge_messages
        self.session_token = session_token
        self.locale = locale
        self.startup_timeout_ms = startup_timeout_ms
        self.process_runner = _kwargs.get("process_runner")
        self.extra_kwargs = dict(_kwargs)
        self.renderer_events = _kwargs.get("renderer_events")
        self.retry_ownership_changed = _kwargs.get("retry_ownership_changed")
        self.state = "off"
        self.failure_reason: str | None = None
        self.restart_scheduled = False
        self.stop_calls = 0
        self._start_gate = asyncio.Event()
        self._start_failure_reason: str | None = None
        self._runtime_failure_reason: str | None = None
        self._monitor_release: asyncio.Event | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self.__class__.instances.append(self)

    async def start(self) -> None:
        self.state = "starting"
        if self.retry_ownership_changed is not None:
            await self.retry_ownership_changed(False)
        await self._start_gate.wait()
        if self._start_failure_reason is not None:
            self.state = "failed"
            self.failure_reason = self._start_failure_reason
            return

        self.state = "connected"
        self.failure_reason = None
        self._monitor_release = asyncio.Event()

        async def _monitor() -> None:
            assert self._monitor_release is not None
            await self._monitor_release.wait()
            if self._runtime_failure_reason is not None:
                self.state = "failed"
                self.failure_reason = self._runtime_failure_reason

        self._monitor_task = asyncio.create_task(_monitor())

    async def confirm_native_retry_ownership(self) -> None:
        assert self.retry_ownership_changed is not None
        await self.retry_ownership_changed(True)

    async def stop(self) -> None:
        self.stop_calls += 1
        if self._monitor_task is not None and not self._monitor_task.done():
            self._monitor_task.cancel()
            await asyncio.gather(self._monitor_task, return_exceptions=True)
        self.state = "off"

    def complete_startup(self, *, failure_reason: str | None = None) -> None:
        self._start_failure_reason = failure_reason
        self._start_gate.set()

    def trigger_runtime_failure(self, failure_reason: str) -> None:
        self._runtime_failure_reason = failure_reason
        assert self._monitor_release is not None
        self._monitor_release.set()


class ReadyProvisioningPort:
    def __init__(self) -> None:
        self.snapshot = LocalASRProvisioningSnapshot(
            models=(
                LocalASRModelProvisioningState(PARAKEET_V3_MODEL_ID, "cpu", "ready"),
                LocalASRModelProvisioningState(PARAKEET_JAPANESE_MODEL_ID, "cpu", "ready"),
                LocalASRModelProvisioningState(LOCAL_STT_MODEL_ID, "cpu", "ready"),
                LocalASRModelProvisioningState(LOCAL_QWEN_GPU_MODEL_ID, "gpu", "ready"),
            ),
            required_cpu_model_ids=REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
            gpu_model_id=LOCAL_QWEN_GPU_MODEL_ID,
        )

    @property
    def diagnostics(self):
        return ()

    async def inspect_cpu(self, model_ids=None, *, verify_checksums=False):
        _ = (model_ids, verify_checksums)
        return self.snapshot

    async def inspect_gpu(self, *, explicit_intent, verify_checksums=False):
        _ = (explicit_intent, verify_checksums)
        return self.snapshot

    def start_install(self, request, *, result_handler=None):
        raise AssertionError(
            f"unexpected provisioning install: {request}, handler={result_handler}"
        )

    async def report_model_validation_failure(self, model_id, *, failure_type):
        _ = (model_id, failure_type)
        return self.snapshot

    async def cancel_install(self, backend):
        _ = backend

    async def close(self):
        return

    def lifecycle_owner_snapshot(self):
        return {"owner": "LocalASRProvisioningOwner"}


def _presentation(
    app: object,
    *,
    page: object | None = None,
) -> FletUiPresentationAdapter:
    if page is not None:
        setattr(app, "page", page)
    return FletUiPresentationAdapter(app)


def _make_controller(*, app: object) -> GuiController:
    return GuiController(
        page=SimpleNamespace(),
        app=_presentation(app),
        config_path=Path("settings.json"),
        local_asr_provisioning=ReadyProvisioningPort(),
    )


def _settings_result(controller: GuiController) -> messages.TransactionResult | None:
    return controller._get_settings_application_owner().results.current


@pytest.mark.asyncio
async def test_self_capture_admission_rejects_unsupported_language_before_status_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dashboard = DummyDashboard()
    controller = _make_controller(app=SimpleNamespace(view_dashboard=dashboard))
    controller.settings = AppSettings()
    controller.settings.provider.stt = STTProviderName.LOCAL_CPU_AUTO
    controller.settings.languages.source_language = "he"
    controller.hub = SimpleNamespace()
    messages: list[str] = []

    def unexpected_status(_self: GuiController) -> str:
        raise RuntimeError("status probe must not run")

    def show_message(_self: GuiController, message_key: str) -> None:
        messages.append(message_key)

    monkeypatch.setattr(
        GuiController,
        "_current_local_stt_runtime_status",
        unexpected_status,
    )
    monkeypatch.setattr(GuiController, "_show_short_stt_message", show_message)
    admission = controller._get_self_capture_owner()._admission

    result = await admission.admit(
        SelfCaptureSessionConfig(
            provider_id=STTProviderName.LOCAL_CPU_AUTO.value,
            provider_signature=("provider",),
            runtime_signature=("runtime",),
            capture_signature=("capture",),
            target_sample_rate_hz=16000,
            local_cpu=True,
        )
    )

    assert result.status is SelfCaptureAdmissionStatus.REJECTED
    assert result.reason == "language_unsupported"
    assert dashboard.stt_enabled is False
    assert dashboard.stt_needs_key is False
    assert messages == ["local_stt.language_unsupported"]


def test_self_capture_admission_effects_preserve_controller_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dashboard = DummyDashboard()
    controller = _make_controller(app=SimpleNamespace(view_dashboard=dashboard))
    controller._get_gpu_runtime_interaction_owner().retain_pending("peer")
    messages: list[str] = []
    repairs: list[tuple[str, str, int | None]] = []

    def show_message(_self: GuiController, message_key: str) -> None:
        messages.append(message_key)

    def request_repair(
        _self: GuiController,
        status: str,
        *,
        channel: str,
        model_ids: tuple[str, ...] | None = None,
        activation_generation: int | None = None,
    ) -> bool:
        assert model_ids is None
        repairs.append((status, channel, activation_generation))
        return False

    monkeypatch.setattr(GuiController, "_show_short_stt_message", show_message)
    monkeypatch.setattr(
        GuiController,
        "_request_unavailable_local_asr_repair",
        request_repair,
    )

    controller._apply_self_capture_admission_effect(
        SelfCaptureAdmissionEffect(
            SelfCaptureAdmissionEffectType.RETAIN_GPU_PENDING_INTENT,
            status="installing",
        )
    )
    controller._apply_self_capture_admission_effect(
        SelfCaptureAdmissionEffect(
            SelfCaptureAdmissionEffectType.REJECT_UNSUPPORTED_LANGUAGE,
        )
    )
    controller._apply_self_capture_admission_effect(
        SelfCaptureAdmissionEffect(
            SelfCaptureAdmissionEffectType.RETAIN_DOWNLOAD_PENDING_INTENT,
            status="downloading",
            activation_generation=21,
        )
    )
    controller._apply_self_capture_admission_effect(
        SelfCaptureAdmissionEffect(
            SelfCaptureAdmissionEffectType.REQUEST_LOCAL_REPAIR,
            status="invalid",
            activation_generation=23,
        )
    )

    assert controller._get_gpu_runtime_interaction_owner().snapshot.pending_channels == frozenset(
        {"peer", "self"}
    )
    assert controller._local_stt_pending_enable_after_install is True
    assert controller._get_local_asr_cpu_repair_owner().snapshot.self_activation_generation == 21
    assert dashboard.stt_enabled is False
    assert dashboard.stt_needs_key is False
    assert messages == [
        "local_stt.language_unsupported",
        "local_stt.download_in_progress",
    ]
    assert repairs == [("invalid", "self", 23)]


def _patch_settings_save(
    monkeypatch: pytest.MonkeyPatch,
    callback: Callable[[Path, AppSettings], object],
) -> None:
    def save(path: Path, canonical: AppSettingsVNext) -> object:
        compatibility = canonical_persistence_adapter_module.SettingsVNextCanonicalPersistenceAdapter().compatibility_projection(
            canonical
        )
        callback(path, compatibility)
        return SimpleNamespace(ok=True)

    monkeypatch.setattr(canonical_persistence_adapter_module, "save_vnext_settings", save)


def _controller_with_persisted_settings(
    tmp_path: Path,
    settings: AppSettings,
) -> tuple[GuiController, Path]:
    path = tmp_path / "settings.json"
    save_settings(path, settings)
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.config_path = path
    controller.settings = controller._load_or_init_settings(path)
    controller._get_settings_owner().authoritative = True
    controller._get_settings_owner().remember_projection(controller.settings)
    return controller, path


def _language_selection_change(
    *,
    source_code: str,
    target_code: str,
    peer_source_code: str = "",
    peer_target_code: str = "",
    peer_source_mode: str = "manual",
    recent_source_codes: tuple[str, ...] = (),
    recent_target_codes: tuple[str, ...] = (),
) -> LanguageSelectionChange:
    return LanguageSelectionChange(
        source_code=source_code,
        target_code=target_code,
        peer_source_code=peer_source_code,
        peer_target_code=peer_target_code,
        peer_source_mode=peer_source_mode,
        recent_source_codes=recent_source_codes,
        recent_target_codes=recent_target_codes,
    )


def _managed_china_settings() -> AppSettings:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.translation.connection = TranslationConnection.MANAGED_CHINA
    return settings


def _local_qwen_suppressed_notification(
    *,
    channel: Literal["self", "peer"] = "self",
) -> FinalTranscriptSuppressedNotification:
    return FinalTranscriptSuppressedNotification(
        utterance_id=uuid4(),
        channel=channel,
        stt_provider_name=STTProviderName.LOCAL_QWEN,
    )


@pytest.mark.asyncio
async def test_managed_china_required_prepare_opens_qq_dialog_instead_of_snackbar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qq_dialog_calls: list[str] = []
    snackbars: list[tuple[str, object]] = []
    dashboard_enabled: list[bool] = []
    app = SimpleNamespace(
        view_dashboard=SimpleNamespace(
            set_translation_enabled=lambda enabled: dashboard_enabled.append(enabled)
        ),
        show_qq_managed_auth_dialog=lambda: qq_dialog_calls.append("show"),
        _show_snackbar=lambda *args: snackbars.append(args),
    )
    controller = _make_controller(app=app)
    controller.settings = _managed_china_settings()
    controller.hub = SimpleNamespace(llm=None, translation_enabled=False)

    async def prepare_required() -> ManagedOpenRouterReleaseResult:
        return ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.RETRY,
            message_key="qq_managed_auth.required",
            local_key_available=False,
        )

    controller._managed_openrouter_release_service = SimpleNamespace(
        prepare_for_translation=prepare_required
    )

    async def no_founder_letter(_self) -> bool:
        return False

    async def refresh_noop(_self, **_kwargs) -> None:
        return None

    monkeypatch.setattr(ManagedUsageOwner, "should_route_to_founder_letter", no_founder_letter)
    monkeypatch.setattr(ManagedUsageOwner, "refresh", refresh_noop)
    monkeypatch.setattr(GuiController, "log_detailed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(GuiController, "log_basic", lambda *_args, **_kwargs: None)

    result = await controller.set_translation_enabled(True)

    assert result is False
    assert qq_dialog_calls == ["show"]
    assert snackbars == []
    assert dashboard_enabled == [False]


def test_local_qwen_suppression_first_gui_detection_counts_without_modal() -> None:
    modal_calls: list[str] = []
    app = SimpleNamespace(
        debug_ui_preview=False,
        show_local_qwen_hallucination_dialog=lambda: modal_calls.append("modal"),
    )
    controller = _make_controller(app=app)
    controller._runtime_logging = RuntimeLoggingSpy()

    controller._on_final_transcript_suppressed(_local_qwen_suppressed_notification())

    assert controller._local_qwen_hallucination_detection_count == 1
    assert controller._local_qwen_hallucination_modal_shown is False
    assert modal_calls == []


def test_local_qwen_suppression_second_gui_detection_opens_modal_once_without_settings_persistence() -> (
    None
):
    settings = AppSettings()
    before = to_dict(settings)
    modal_calls: list[str] = []
    app = SimpleNamespace(
        debug_ui_preview=False,
        show_local_qwen_hallucination_dialog=lambda: modal_calls.append("modal"),
    )
    controller = _make_controller(app=app)
    controller._runtime_logging = RuntimeLoggingSpy()
    controller.settings = settings

    controller._on_final_transcript_suppressed(_local_qwen_suppressed_notification(channel="self"))
    controller._on_final_transcript_suppressed(_local_qwen_suppressed_notification(channel="peer"))
    controller._on_final_transcript_suppressed(_local_qwen_suppressed_notification(channel="self"))

    assert controller._local_qwen_hallucination_detection_count == 3
    assert controller._local_qwen_hallucination_modal_shown is True
    assert modal_calls == ["modal"]
    assert to_dict(settings) == before
    assert not any("hallucination" in key for key in to_dict(settings)["ui"])


def test_local_qwen_suppression_provider_switch_does_not_reset_same_session_modal_state() -> None:
    modal_calls: list[str] = []
    app = SimpleNamespace(
        debug_ui_preview=False,
        show_local_qwen_hallucination_dialog=lambda: modal_calls.append("modal"),
    )
    controller = _make_controller(app=app)
    controller._runtime_logging = RuntimeLoggingSpy()
    controller.settings = AppSettings()

    controller._on_final_transcript_suppressed(_local_qwen_suppressed_notification())
    controller._on_final_transcript_suppressed(_local_qwen_suppressed_notification())
    controller.settings.provider.stt = STTProviderName.DEEPGRAM
    controller.settings.provider.stt = STTProviderName.LOCAL_QWEN
    controller._on_final_transcript_suppressed(_local_qwen_suppressed_notification())

    assert controller._local_qwen_hallucination_detection_count == 3
    assert controller._local_qwen_hallucination_modal_shown is True
    assert modal_calls == ["modal"]


def test_local_qwen_suppression_new_gui_session_resets_counter_and_modal_state() -> None:
    first_session_modal_calls: list[str] = []
    first_controller = _make_controller(
        app=SimpleNamespace(
            debug_ui_preview=False,
            show_local_qwen_hallucination_dialog=lambda: first_session_modal_calls.append("modal"),
        )
    )
    first_controller._runtime_logging = RuntimeLoggingSpy()
    first_controller._on_final_transcript_suppressed(_local_qwen_suppressed_notification())
    first_controller._on_final_transcript_suppressed(_local_qwen_suppressed_notification())
    assert first_session_modal_calls == ["modal"]

    new_session_modal_calls: list[str] = []
    new_controller = _make_controller(
        app=SimpleNamespace(
            debug_ui_preview=False,
            show_local_qwen_hallucination_dialog=lambda: new_session_modal_calls.append("modal"),
        )
    )
    new_controller._runtime_logging = RuntimeLoggingSpy()

    new_controller._on_final_transcript_suppressed(_local_qwen_suppressed_notification())

    assert new_controller._local_qwen_hallucination_detection_count == 1
    assert new_controller._local_qwen_hallucination_modal_shown is False
    assert new_session_modal_calls == []


def test_local_qwen_suppression_non_gui_callback_logs_only_without_modal_attempt() -> None:
    app = SimpleNamespace(debug_ui_preview=False)
    controller = _make_controller(app=app)
    controller._runtime_logging = RuntimeLoggingSpy()

    controller._on_final_transcript_suppressed(_local_qwen_suppressed_notification())
    controller._on_final_transcript_suppressed(_local_qwen_suppressed_notification())

    assert controller._local_qwen_hallucination_detection_count == 2
    assert controller._local_qwen_hallucination_modal_shown is False
    messages = [message for _level, message in controller._runtime_logging.detailed_messages]
    assert any("guidance_modal=unavailable" in message for message in messages)


def test_debug_capture_fault_is_disabled_without_debug_preview() -> None:
    controller = _make_controller(app=SimpleNamespace(debug_ui_preview=False))
    controller._debug_capture_fault_profile = "capture_attenuate_40db"

    assert controller.cycle_debug_capture_fault_profile() == "none"
    assert controller.debug_capture_fault_profile == "capture_attenuate_40db"


async def _wait_until(predicate, *, attempts: int = 20, delay_s: float = 0.0) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(delay_s)
    raise AssertionError("condition was not met in time")


def _patch_overlay_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeOverlayBridge.instances = []
    FakeOverlayProcessManager.instances = []
    monkeypatch.setattr(overlay_generation_start_module, "OverlayBridge", FakeOverlayBridge)
    monkeypatch.setattr(
        overlay_generation_start_module,
        "OverlayProcessManager",
        FakeOverlayProcessManager,
    )


def _overlay_runtime(controller: GuiController) -> OverlayRuntimeHandle:
    runtime = _overlay_owner(controller).runtime
    assert runtime is not None
    return runtime


def _overlay_owner(controller: GuiController) -> OverlayApplicationOwner:
    return controller._get_overlay_application_owner()


def _peer_runtime_config(
    controller: GuiController,
    settings: AppSettings,
):
    return build_peer_capture_session_config(
        settings,
        canonical_settings=controller._canonical_vnext_settings_for(settings),
    )


def _peer_runtime_signature(
    controller: GuiController,
    settings: AppSettings,
) -> tuple[object, ...]:
    return build_peer_stt_runtime_signature(
        settings,
        canonical_settings=controller._canonical_vnext_settings_for(settings),
    )


def _peer_provider_signature(
    controller: GuiController,
    settings: AppSettings,
) -> tuple[object, ...]:
    return build_peer_stt_provider_signature_from_vnext(
        controller._canonical_vnext_settings_for(settings)
    )


def _peer_provider_request(
    controller: GuiController,
    config,
    *,
    warmup: bool = False,
):
    assert controller.settings is not None
    return build_peer_stt_provider_request(
        config,
        gpu_device_id=controller.settings.stt.gpu_device_id,
        warmup=warmup,
    )


def _attach_overlay_presenter(controller: GuiController, presenter: object | None) -> None:
    _overlay_owner(controller).ensure_runtime().attach_presenter(presenter)


def _attach_overlay_bridge(controller: GuiController, bridge: object | None) -> None:
    if bridge is None:
        runtime = _overlay_owner(controller).runtime
        if runtime is not None:
            runtime.attach_bridge(None)
        return
    _overlay_owner(controller).ensure_runtime().attach_bridge(bridge)


def _attach_overlay_manager(controller: GuiController, manager: object | None) -> None:
    _overlay_owner(controller).ensure_runtime().attach_process_manager(manager)


def _attach_overlay_diagnostics(controller: GuiController, diagnostics: object | None) -> None:
    _overlay_owner(controller).ensure_runtime().attach_diagnostics(diagnostics)


def _attach_desktop_renderer_events(
    controller: GuiController,
    renderer_events: asyncio.Queue[dict[str, object]] | None,
) -> None:
    _overlay_owner(controller).ensure_runtime().attach_renderer_events(renderer_events)


def _microphone_test_task(controller: GuiController) -> asyncio.Task[None] | None:
    runtime = controller._get_microphone_test_owner().runtime_if_created
    return runtime.session_task if runtime is not None else None


def test_settings_view_order22_baseline_uses_path_snapshot_not_legacy_settings() -> None:
    controller = _make_controller(app=SimpleNamespace())
    projection = controller._settings_projection()
    baseline = AppSettings()
    baseline.languages.source_language = "ko"
    projection.remember_order22(baseline)

    assert not isinstance(projection.order22_baseline, AppSettings)

    current = AppSettings()
    current.languages.source_language = "ja"
    current.provider.llm = LLMProviderName.DEEPSEEK
    controller.settings = current
    pending = copy.deepcopy(current)

    base_settings, patch_values = projection.order22_patch_base_and_values(pending)

    assert patch_values == {"languages.source_language": "ja"}
    assert base_settings is not current
    assert base_settings.languages.source_language == "ko"
    assert base_settings.provider.llm == LLMProviderName.DEEPSEEK


def test_settings_view_change_rebases_audio_patch_without_restoring_stale_peer_language() -> None:
    controller = _make_controller(app=SimpleNamespace())
    projection = controller._settings_projection()
    settings_view_snapshot = AppSettings()
    settings_view_snapshot.provider.peer_stt = STTProviderName.LOCAL_CPU_AUTO
    projection.remember_all(settings_view_snapshot)

    pending = copy.deepcopy(settings_view_snapshot)
    pending.desktop_audio.output_device = "Speakers (Loopback)"
    change = controller.capture_settings_view_change(pending)

    current = copy.deepcopy(settings_view_snapshot)
    current.languages.peer_source_language = "ja"
    controller.settings = current
    projection.remember_order22(current)

    merged = controller.merge_settings_view_change_with_current(change)

    assert merged.desktop_audio.output_device == "Speakers (Loopback)"
    assert merged.languages.peer_source_language == "ja"
    assert controller.settings.languages.peer_source_language == "ja"
    decision = controller_module.resolve_local_asr_selection(
        merged.provider.peer_stt.value,
        merged.languages.effective_peer_source,
    )
    assert decision.model_id == "parakeet-tdt-ctc-0.6b-ja-int8-sherpa"


@pytest.mark.parametrize("reload_method", ["reload", "sync"])
def test_failed_settings_view_reload_preserves_displayed_mutation_baseline(
    reload_method: str,
) -> None:
    class FailingSettingsView:
        def load_from_settings(self, *_args, **_kwargs) -> None:
            raise RuntimeError("settings view reload failed")

    controller = _make_controller(app=SimpleNamespace(view_settings=FailingSettingsView()))
    projection = controller._settings_projection()
    displayed = AppSettings()
    displayed.languages.source_language = "en"
    projection.remember_all(displayed)

    committed = copy.deepcopy(displayed)
    committed.languages.source_language = "ja"
    controller.settings = committed
    if reload_method == "reload":
        projection.render(
            committed,
            preserve_custom_vocab_draft=True,
        )
    else:
        controller._sync_ui_from_settings()

    stale_edit = copy.deepcopy(displayed)
    stale_edit.overlay.show_translation = False
    change = controller.capture_settings_view_change(stale_edit)
    merged = controller.merge_settings_view_change_with_current(change)

    assert change.values_by_path == {"overlay.show_translation": False}
    assert merged.languages.source_language == "ja"
    assert merged.overlay.show_translation is False


@pytest.mark.parametrize(
    "failure_reason",
    [
        "missing_executable",
        "spawn_failed",
        "manifest_invalid",
        "contract_mismatch",
        "startup_timeout",
        "bridge_auth_failed",
        "renderer_init_failed",
        "runtime_disconnected",
        "window_configuration_failed",
        "runtime_control_invalid",
        "runtime_crashed",
        "unknown",
    ],
)
def test_desktop_gui_overlay_failure_i18n_reasons_survive_controller_normalization(
    failure_reason: str,
) -> None:
    reported: list[tuple[str, str | None]] = []
    controller = _make_controller(app=SimpleNamespace())
    controller._ui_event_bridge = SimpleNamespace(
        report_overlay_state=lambda state, failure_reason=None: reported.append(
            (state, failure_reason)
        )
    )

    _overlay_owner(controller).on_start_failed(failure_reason)

    assert _overlay_owner(controller).snapshot.state == "failed"
    assert _overlay_owner(controller).snapshot.failure_reason == failure_reason
    assert reported == [("failed", failure_reason)]


def _patch_init_pipeline_dependencies(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    created: dict[str, object] = {}

    monkeypatch.setattr(controller_module, "create_secret_store", lambda *_a, **_k: object())
    monkeypatch.setattr(controller_module, "create_llm_provider", lambda *_a, **_k: "llm")

    class FakeSender:
        def close(self) -> None:
            return None

    def fake_sender(*_args, **_kwargs):
        sender = FakeSender()
        created["sender"] = sender
        return sender

    def fake_osc(*_args, **_kwargs):
        created["osc_kwargs"] = dict(_kwargs)
        osc = object()
        created["osc"] = osc
        return osc

    def fake_hub(*_args, **kwargs):
        hub = SimpleNamespace(
            llm=kwargs.get("llm"),
            stt=kwargs.get("stt"),
            peer_stt=kwargs.get("peer_stt"),
            local_asr_provider_runtime=None,
            peer_translation_enabled=kwargs.get("peer_translation_enabled", False),
            integrated_context_enabled=kwargs.get("integrated_context_enabled", False),
        )

        def has_stt_provider(channel: str) -> bool:
            return hub.stt is not None if channel == "self" else hub.peer_stt is not None

        async def replace_stt_provider_request(
            request,
            *,
            start,
            on_terminal_failure,
        ):
            _ = start, on_terminal_failure
            created["stt_request"] = request
            hub.stt = "owned-stt"
            return SimpleNamespace(status="applied", failure_type=None)

        hub.has_stt_provider = has_stt_provider
        hub.replace_stt_provider_request = replace_stt_provider_request
        created["hub"] = hub
        return hub

    monkeypatch.setattr(controller_module, "VrchatOscUdpSender", fake_sender)
    monkeypatch.setattr(controller_module, "ChatboxPaginator", fake_osc)
    monkeypatch.setattr(controller_module, "ClientHub", fake_hub)

    return created


@pytest.mark.asyncio
async def test_init_pipeline_wires_self_stt_fault_provider_with_debug_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stt_calls: list[dict[str, object]] = []
    created = _patch_init_pipeline_dependencies(monkeypatch)

    def fake_stt_provider_factory(*_args, **kwargs):
        stt_calls.append(dict(kwargs))
        return SimpleNamespace()

    app = SimpleNamespace(debug_ui_preview=True)
    controller = _make_controller(app=app)
    controller.settings = AppSettings()
    prior_owner_closed: list[str] = []

    class PriorOwner:
        async def close(self) -> None:
            prior_owner_closed.append("closed")

    prior_owner = PriorOwner()
    controller._self_capture_owner = prior_owner
    monkeypatch.setattr(
        controller_module,
        "ManagedSTTProviderFactory",
        fake_stt_provider_factory,
    )
    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        lambda self, *, enabled: asyncio.sleep(0),
    )

    assert controller.cycle_debug_stt_fault_profile() == "stt_input_low_snr_vad_pass"
    await controller._init_pipeline()

    provider = stt_calls[0]["fault_profile_provider"]
    assert prior_owner_closed == ["closed"]
    assert controller._self_capture_owner is not prior_owner
    assert created["stt_request"].provider_id == controller.settings.provider.stt.value
    assert controller._hub_has_stt_provider("self")
    assert callable(stt_calls[0]["on_final_transcript_suppressed"])
    assert callable(provider)
    assert provider() == "stt_input_low_snr_vad_pass"

    app.debug_ui_preview = False
    assert provider() == "none"
    assert controller.debug_stt_fault_profile == "stt_input_low_snr_vad_pass"


@pytest.mark.asyncio
async def test_rebuild_stt_provider_delegates_immutable_owner_request() -> None:
    app = SimpleNamespace(debug_ui_preview=False)
    controller = _make_controller(app=app)
    controller._runtime_logging = RuntimeLoggingSpy()
    controller.settings = AppSettings()
    configs: list[object] = []

    class FakeOwner:
        loop_task = None
        source = None
        cleanup_source = None
        vad = None
        last_cleanup_exception = None

        async def prepare_provider(self, config):
            configs.append(config)
            return SelfCaptureSessionSnapshot(
                state=SelfCaptureSessionState.STOPPED,
                provider_status=SelfCaptureProviderStatus.READY,
                desired_active=False,
                effective_active=False,
                generation=1,
                provider_id=config.provider_id,
                runtime_signature=config.runtime_signature,
                failure_reason=None,
                admission_reason=None,
                has_source=False,
                has_vad=False,
                has_loop_task=False,
                cleanup_debt=0,
                closed=False,
            )

    controller.hub = SimpleNamespace()
    controller._self_capture_owner = FakeOwner()
    controller._debug_stt_fault_profile = "stt_input_low_snr_vad_pass"

    await controller._rebuild_stt_provider()

    config = configs[0]
    request = controller._self_capture_provider_request(config, False)
    assert request.provider_id == controller.settings.provider.stt.value
    assert request.config.source_language == controller.settings.languages.source_language
    assert (
        config.provider_signature
        == build_self_capture_session_config(controller.settings).provider_signature
    )


def test_peer_stt_provider_request_preserves_resolved_runtime_config() -> None:
    app = SimpleNamespace(debug_ui_preview=True)
    controller = _make_controller(app=app)
    controller.settings = AppSettings()
    config = _peer_runtime_config(controller, controller.settings)

    request = _peer_provider_request(controller, config)

    assert request.config is config.backend
    assert request.provider_id == config.backend.provider
    assert request.model_id == config.model_id or config.backend.model
    assert request.session_options is config.session_options


def test_cycle_debug_stt_fault_profile_requires_debug_preview() -> None:
    app = SimpleNamespace(debug_ui_preview=False)
    controller = _make_controller(app=app)
    controller._debug_stt_fault_profile = "stt_input_low_snr_vad_pass"

    assert controller.cycle_debug_stt_fault_profile() == "none"
    assert controller.debug_stt_fault_profile == "stt_input_low_snr_vad_pass"

    app.debug_ui_preview = True
    assert controller.cycle_debug_stt_fault_profile() == "none"
    assert controller.cycle_debug_stt_fault_profile() == "stt_input_low_snr_vad_pass"


@pytest.mark.parametrize("channel_label", ["self", "peer"])
def test_wrap_diagnostic_audio_source_wires_capture_fault_provider_with_debug_gate(
    channel_label: str,
) -> None:
    class FakeAudioSource:
        async def frames(self):
            if False:
                yield None

        async def close(self) -> None:
            return None

    app = SimpleNamespace(debug_ui_preview=True)
    controller = _make_controller(app=app)
    controller._runtime_logging = RuntimeLoggingSpy()
    controller._debug_capture_fault_profile = "capture_attenuate_40db"

    wrapped = controller._wrap_diagnostic_audio_source(
        FakeAudioSource(),
        channel_label=channel_label,
    )

    assert getattr(wrapped, "channel_label") == channel_label
    provider = getattr(wrapped, "fault_profile_provider")
    assert callable(provider)
    assert provider() == "capture_attenuate_40db"

    app.debug_ui_preview = False
    assert provider() == "none"
    assert controller.debug_capture_fault_profile == "capture_attenuate_40db"


@pytest.mark.asyncio
async def test_clipboard_watcher_starts_and_stops_from_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    watchers: list[FakeClipboardWatcher] = []

    def watcher_factory(on_text: Callable[[str], None]) -> FakeClipboardWatcher:
        watcher = FakeClipboardWatcher(on_text)
        watchers.append(watcher)
        return watcher

    monkeypatch.setattr(controller_module, "create_clipboard_watcher", watcher_factory)
    monkeypatch.setattr(controller_module.sys, "platform", "win32")

    controller = GuiController(
        page=SimpleNamespace(),
        app=_presentation(SimpleNamespace()),
        config_path=tmp_path / "settings.json",
    )
    controller.settings = AppSettings()
    controller.hub = DummyHub()

    controller.settings.ui.clipboard_auto_translate_enabled = True
    await controller._sync_clipboard_watcher()

    assert len(watchers) == 1
    assert watchers[0].started is True

    controller.settings.ui.clipboard_auto_translate_enabled = False
    await controller._sync_clipboard_watcher()

    assert watchers[0].stopped is True


@pytest.mark.asyncio
async def test_clipboard_watcher_submits_valid_text_through_existing_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    watchers: list[FakeClipboardWatcher] = []

    def watcher_factory(on_text: Callable[[str], None]) -> FakeClipboardWatcher:
        watcher = FakeClipboardWatcher(on_text)
        watchers.append(watcher)
        return watcher

    monkeypatch.setattr(controller_module, "create_clipboard_watcher", watcher_factory)
    monkeypatch.setattr(controller_module.sys, "platform", "win32")

    controller = GuiController(
        page=SimpleNamespace(),
        app=_presentation(SimpleNamespace()),
        config_path=tmp_path / "settings.json",
    )
    controller.settings = AppSettings()
    controller.settings.ui.clipboard_auto_translate_enabled = True
    controller.hub = DummyHub()

    await controller._sync_clipboard_watcher()
    watchers[0].emit("  hello clipboard  ")
    await asyncio.wait_for(controller.hub.submit_event.wait(), timeout=1.0)

    assert controller.hub.submit_calls == [("hello clipboard", "Clipboard")]


@pytest.mark.asyncio
async def test_clipboard_watcher_does_not_block_manual_fallback_when_translation_off(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    watchers: list[FakeClipboardWatcher] = []

    def watcher_factory(on_text: Callable[[str], None]) -> FakeClipboardWatcher:
        watcher = FakeClipboardWatcher(on_text)
        watchers.append(watcher)
        return watcher

    monkeypatch.setattr(controller_module, "create_clipboard_watcher", watcher_factory)
    monkeypatch.setattr(controller_module.sys, "platform", "win32")

    controller = GuiController(
        page=SimpleNamespace(),
        app=_presentation(SimpleNamespace()),
        config_path=tmp_path / "settings.json",
    )
    controller.settings = AppSettings()
    controller.settings.ui.clipboard_auto_translate_enabled = True
    controller.hub = DummyHub(llm=None)
    controller.hub.translation_enabled = False

    await controller._sync_clipboard_watcher()
    watchers[0].emit("source fallback")
    await asyncio.wait_for(controller.hub.submit_event.wait(), timeout=1.0)

    assert controller.hub.submit_calls == [("source fallback", "Clipboard")]


@pytest.mark.asyncio
async def test_clipboard_watcher_ignores_empty_and_long_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    watchers: list[FakeClipboardWatcher] = []

    def watcher_factory(on_text: Callable[[str], None]) -> FakeClipboardWatcher:
        watcher = FakeClipboardWatcher(on_text)
        watchers.append(watcher)
        return watcher

    monkeypatch.setattr(controller_module, "create_clipboard_watcher", watcher_factory)
    monkeypatch.setattr(controller_module.sys, "platform", "win32")

    controller = GuiController(
        page=SimpleNamespace(),
        app=_presentation(SimpleNamespace()),
        config_path=tmp_path / "settings.json",
    )
    controller.settings = AppSettings()
    controller.settings.ui.clipboard_auto_translate_enabled = True
    controller.hub = DummyHub()

    await controller._sync_clipboard_watcher()
    watchers[0].emit("   ")
    watchers[0].emit("x" * 301)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert controller.hub.submit_calls == []


@pytest.mark.asyncio
async def test_clipboard_watcher_not_started_on_non_windows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called = False

    def watcher_factory(on_text: Callable[[str], None]) -> FakeClipboardWatcher:
        nonlocal called
        called = True
        return FakeClipboardWatcher(on_text)

    monkeypatch.setattr(controller_module, "create_clipboard_watcher", watcher_factory)
    monkeypatch.setattr(controller_module.sys, "platform", "linux")

    controller = GuiController(
        page=SimpleNamespace(),
        app=_presentation(SimpleNamespace()),
        config_path=tmp_path / "settings.json",
    )
    controller.settings = AppSettings()
    controller.settings.ui.clipboard_auto_translate_enabled = True
    controller.hub = DummyHub()

    await controller._sync_clipboard_watcher()

    assert called is False
    assert controller._get_clipboard_auto_translation_owner().runtime is None


def test_manual_local_asr_mismatches_persist_qwen_for_self_and_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.LOCAL_PARAKEET_V3
    settings.provider.peer_stt = STTProviderName.LOCAL_PARAKEET_JAPANESE
    settings.languages.source_language = "ko"
    settings.languages.peer_source_language = "en"
    saved: list[AppSettings] = []
    messages: list[str] = []
    controller = GuiController(
        page=SimpleNamespace(),
        app=FletUiPresentationAdapter(
            SimpleNamespace(
                show_snackbar=lambda message, _color: messages.append(message),
            )
        ),
        config_path=Path("settings.json"),
    )
    controller.settings = settings

    monkeypatch.setattr(GuiController, "_sync_ui_from_settings", lambda self: None)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: saved.append(self.current) or True,
    )

    assert controller._get_settings_application_owner().persist_manual_fallback() is True
    assert controller.settings.provider.stt == STTProviderName.LOCAL_QWEN
    assert controller.settings.provider.peer_stt == STTProviderName.LOCAL_QWEN
    assert len(saved) == 1
    assert messages == [t("local_stt.language_fallback_qwen")]


@pytest.mark.asyncio
async def test_start_local_llm_without_runtime_does_not_show_api_key_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.LOCAL_LLM
    settings.translation = TranslationSettings(
        model=TranslationModel.LOCAL_LLM,
        connection=TranslationConnection.OLLAMA,
    )
    dash = DummyDashboard()
    hub = DummyHub(llm=None, stt=object())

    class FakeBridge:
        def __init__(self, **kwargs) -> None:
            _ = kwargs

        async def run(self) -> None:
            await asyncio.sleep(0)

    async def fake_init_pipeline(self: GuiController) -> None:
        self.hub = hub

    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)
    monkeypatch.setattr(GuiController, "_load_or_init_settings", lambda self, path: settings)
    monkeypatch.setattr(GuiController, "_sync_ui_from_settings", lambda self: None)
    monkeypatch.setattr(GuiController, "_init_pipeline", fake_init_pipeline)
    monkeypatch.setattr(presentation_adapter_module, "set_ui_locale", lambda _locale: None)
    monkeypatch.setattr(presentation_adapter_module, "UIEventBridge", FakeBridge)
    monkeypatch.setattr(
        controller_module, "create_secret_store", lambda *_a, **_k: DummySecrets({})
    )
    controller = _make_controller(app=SimpleNamespace(view_dashboard=dash))
    await controller.start()
    await asyncio.sleep(0)

    assert dash.translation_needs_key is False
    assert dash.translation_enabled is False


@pytest.mark.parametrize(
    ("result_map", "expected"),
    [
        ({"qwen3.5-flash": True}, (True, "Verification successful")),
        (
            {"qwen3.5-flash": False, "qwen3.5-plus": True},
            (False, "qwen_model_unavailable:qwen3.5-flash"),
        ),
        (
            {"qwen3.5-flash": False, "qwen3.5-plus": False},
            (False, "Verification failed (check logs/console for details)"),
        ),
    ],
)
@pytest.mark.asyncio
async def test_verify_qwen_key_with_model_fallback_paths(
    result_map: dict[str, bool],
    expected: tuple[bool, str],
) -> None:
    settings = AppSettings()
    settings.qwen.llm_model = QwenLLMModel.QWEN_35_FLASH
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = settings

    async def fake_verify_qwen(
        api_key: str,
        *,
        base_url: str,
        model: str | None,
        low_latency: bool,
    ) -> bool:
        _ = (api_key, base_url, low_latency)
        assert model is not None
        return result_map.get(model, False)

    controller.provider_verifier = SimpleNamespace(
        verify_qwen_llm_api_key=fake_verify_qwen,
    )

    result = await controller.verify_api_key("alibaba_beijing", "secret")
    assert result == expected


@pytest.mark.asyncio
async def test_verify_api_key_preserves_model_and_error_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs = DummyLogsView()
    controller = _make_controller(app=SimpleNamespace(view_logs=logs))
    controller.settings = AppSettings()
    calls: list[tuple[str, str]] = []

    async def succeed(key: str, *, model: str) -> bool:
        calls.append((key, model))
        return True

    monkeypatch.setattr(GeminiLLMProvider, "verify_api_key", staticmethod(succeed))

    empty = await controller.verify_api_key("google", "")
    unknown = await controller.verify_api_key("mystery", "x")
    verified = await controller.verify_api_key("google", "secret")

    async def raise_error(*_args, **_kwargs) -> bool:
        raise RuntimeError("bad key")

    monkeypatch.setattr(GeminiLLMProvider, "verify_api_key", staticmethod(raise_error))
    errored = await controller.verify_api_key("google", "x")

    assert empty == (False, "API Key is empty")
    assert unknown == (False, "Unknown provider: mystery")
    assert verified == (True, "Verification successful")
    assert calls == [("secret", "gemini-3.1-flash-lite")]
    assert errored == (False, "bad key")
    assert getattr(controller, "runtime_logging_mode", None) == "basic"
    assert any("[ERROR]" in line and "bad key" in line for line in logs.logs)


def test_log_error_falls_back_to_standard_logger_without_direct_logs_view_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs = DummyLogsView()
    controller = _make_controller(app=SimpleNamespace(view_logs=logs))

    class BrokenRuntimeLogging:
        def attach_realtime_sink(self, _sink) -> None:
            return None

        def emit_basic(self, _message: str, *, level: int = logging.INFO) -> None:
            _ = level
            raise RuntimeError("emit failed")

    controller._runtime_logging = BrokenRuntimeLogging()
    seen: list[tuple[int, str]] = []
    monkeypatch.setattr(
        controller_module.logger,
        "log",
        lambda level, message: seen.append((level, message)),
    )

    controller._log_error("fallback message")

    assert seen == [(logging.ERROR, "fallback message")]
    assert logs.logs == []


def test_sync_ui_from_settings_updates_dashboard_and_settings_view() -> None:
    settings = AppSettings()
    settings.languages.source_language = "ko"
    settings.languages.target_language = "en"
    settings.languages.recent_source_languages = ["ko", "ja"]
    settings.languages.recent_target_languages = ["en", "zh"]

    dash = DummyDashboard()
    settings_view = DummySettingsView()
    controller = _make_controller(
        app=SimpleNamespace(view_dashboard=dash, view_settings=settings_view)
    )
    controller.settings = settings

    controller._sync_ui_from_settings()

    assert dash.languages == ("ko", "en", "en", "ko")
    assert dash.recent_languages == (["ko", "ja"], ["en", "zh"])
    assert dash.on_recent_languages_change is None
    assert settings_view.calls == [(settings, Path("settings.json"), False)]


@pytest.mark.asyncio
async def test_set_translation_enabled_warms_supported_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.QWEN
    qwen_provider = QwenLLMProvider(api_key="secret")
    controller.hub = DummyHub(
        llm=SemaphoreLLMProvider(inner=qwen_provider, semaphore=asyncio.Semaphore(1))
    )
    called: list[tuple[str, str, str]] = []

    async def fake_verify(
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/api/v1",
        model: str = "qwen3.5-plus",
    ) -> bool:
        called.append((api_key, base_url, model))
        return True

    monkeypatch.setattr(QwenLLMProvider, "verify_api_key", staticmethod(fake_verify))

    await controller.set_translation_enabled(True)

    assert controller.hub.translation_enabled is True
    assert controller.hub.clear_context_calls == 1
    assert called == [("secret", "https://dashscope.aliyuncs.com/api/v1", "qwen3.5-plus")]


@pytest.mark.asyncio
async def test_set_translation_enabled_keeps_managed_translation_disabled_until_local_key_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dash = DummyDashboard()
    settings_view = DummySettingsView()
    controller = _make_controller(
        app=SimpleNamespace(view_dashboard=dash, view_settings=settings_view)
    )
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.hub = DummyHub(llm=object())
    observed_pending: list[bool | None] = []
    controller._managed_openrouter_release_service = InspectingManagedReleaseService(
        ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            message_key="managed_release.ready",
            pending_issue=True,
            local_key_available=False,
        ),
        on_prepare=lambda: observed_pending.append(dash.managed_auth_pending),
    )

    async def fail_fetch_key_metadata(_api_key: str):
        raise AssertionError("fetch_key_metadata should not run without a managed key")

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecrets({}),
    )
    monkeypatch.setattr(
        OpenRouterLLMProvider,
        "fetch_key_metadata",
        staticmethod(fail_fetch_key_metadata),
    )

    await controller.set_translation_enabled(True)

    assert controller._managed_openrouter_release_service.prepare_calls == 1
    assert controller.hub.translation_enabled is False
    assert controller.hub.clear_context_calls == 0
    assert observed_pending == [True]
    assert dash.managed_auth_pending is False
    assert dash.managed_auth_pending_calls == [True, False]
    assert settings_view.managed_trial_usage_state == {
        "visible": True,
        "remaining_percent": None,
    }
    assert dash.managed_trial_calls == []


@pytest.mark.asyncio
async def test_set_translation_enabled_transitions_pending_true_to_false_after_managed_preissue_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dash = DummyDashboard()
    controller = _make_controller(app=SimpleNamespace(view_dashboard=dash))
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.hub = DummyHub(llm=object())
    observed_pending: list[bool | None] = []
    scheduled_refreshes: list[str] = []
    controller._managed_openrouter_release_service = InspectingManagedReleaseService(
        ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            message_key="managed_release.ready",
            api_key="managed-key",
            local_key_available=True,
            pending_issue=False,
        ),
        on_prepare=lambda: observed_pending.append(dash.managed_auth_pending),
    )

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecrets({}),
    )
    monkeypatch.setattr(
        ManagedUsageOwner,
        "schedule_usage_refresh",
        lambda self: scheduled_refreshes.append("scheduled"),
    )

    await controller.set_translation_enabled(True)

    assert controller._managed_openrouter_release_service.prepare_calls == 1
    assert controller.hub.translation_enabled is True
    assert controller.hub.clear_context_calls == 1
    assert observed_pending == [True]
    assert dash.managed_auth_pending is False
    assert dash.managed_auth_pending_calls == [True, False]
    assert scheduled_refreshes == ["scheduled"]


@pytest.mark.asyncio
async def test_set_translation_enabled_rebuild_path_keeps_success_when_managed_usage_refresh_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dash = DummyDashboard()
    controller = _make_controller(app=SimpleNamespace(view_dashboard=dash))
    controller._runtime_logging = RuntimeLoggingSpy()
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.hub = DummyHub(llm=None)
    controller._managed_openrouter_release_service = DummyManagedReleaseService(
        ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            message_key="managed_release.ready",
            api_key="managed-key",
            local_key_available=True,
            pending_issue=False,
        )
    )

    monkeypatch.setattr(controller_module, "create_secret_store", lambda *_a, **_k: object())
    monkeypatch.setattr(
        GuiController,
        "_create_managed_openrouter_release_service",
        lambda self, *, secrets: None,
    )
    monkeypatch.setattr(controller_module, "create_llm_provider", lambda *_a, **_k: object())

    async def fail_usage_refresh(self):
        raise RuntimeError("usage refresh boom")

    monkeypatch.setattr(
        GuiController,
        "_fetch_managed_usage_metadata",
        fail_usage_refresh,
    )

    await controller.set_translation_enabled(True)

    assert controller.hub.llm is not None
    assert controller.hub.translation_enabled is True
    assert controller.hub.clear_context_calls == 1
    assert dash.managed_auth_pending_calls == [True, False]
    assert (
        logging.WARNING,
        "[ManagedAuth] Usage refresh failed: usage refresh boom",
    ) in controller._runtime_logging.basic_messages
    assert (
        logging.INFO,
        "[Settings] LLM provider rebuilt successfully",
    ) in controller._runtime_logging.basic_messages
    assert (
        logging.INFO,
        "[Translation] Enabled with provider: openrouter",
    ) in controller._runtime_logging.basic_messages


@pytest.mark.asyncio
async def test_set_translation_enabled_rebuild_path_turns_translation_back_off_when_refresh_discovers_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shown: list[str] = []
    dash = DummyDashboard()
    settings_view = DummySettingsView()
    controller = _make_controller(
        app=SimpleNamespace(
            view_dashboard=dash,
            view_settings=settings_view,
            show_founder_letter_dialog=lambda: shown.append("shown"),
        )
    )
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.settings.managed_identity.active_managed_credential_ref = "hash_123"
    controller.hub = DummyHub(llm=None)
    controller._managed_openrouter_release_service = DummyManagedReleaseService(
        ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            message_key="managed_release.ready",
            api_key="managed-key",
            local_key_available=True,
            pending_issue=False,
        )
    )

    metadata_responses = [
        OpenRouterKeyMetadata(
            limit_usd=0.07,
            remaining_usd=0.05,
            usage_usd=0.02,
        ),
        OpenRouterKeyMetadata(
            limit_usd=0.07,
            remaining_usd=0.0007,
            usage_usd=0.0693,
        ),
    ]

    async def fake_fetch_key_metadata(_api_key: str):
        return metadata_responses.pop(0)

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_a, **_k: DummySecrets({"openrouter_managed_api_key": "managed-key"}),
    )
    monkeypatch.setattr(
        GuiController,
        "_create_managed_openrouter_release_service",
        lambda self, *, secrets: None,
    )
    monkeypatch.setattr(controller_module, "create_llm_provider", lambda *_a, **_k: object())
    monkeypatch.setattr(
        OpenRouterLLMProvider,
        "fetch_key_metadata",
        staticmethod(fake_fetch_key_metadata),
    )

    await controller.set_translation_enabled(True)

    assert shown == ["shown"]
    assert controller.hub.llm is not None
    assert controller.hub.translation_enabled is False
    assert controller.hub.clear_context_calls == 0
    assert dash.translation_enabled is False
    assert settings_view.managed_trial_usage_state == {
        "visible": True,
        "remaining_percent": 1,
    }


@pytest.mark.asyncio
async def test_set_translation_enabled_keeps_managed_translation_disabled_on_retry_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snackbar_calls: list[tuple[str, str]] = []
    dash = DummyDashboard()
    settings_view = DummySettingsView()
    controller = _make_controller(
        app=SimpleNamespace(
            _show_snackbar=lambda message, color: snackbar_calls.append((message, color)),
            view_dashboard=dash,
            view_settings=settings_view,
        )
    )
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.hub = DummyHub(llm=object())
    controller._runtime_logging = RuntimeLoggingSpy()
    controller._managed_openrouter_release_service = DummyManagedReleaseService(
        ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.RETRY,
            message_key="managed_release.retry_after_ms",
            message_kwargs={"retry_after_ms": 5000},
            diagnostics=ManagedOpenRouterReleaseDiagnostics(
                operation="issue",
                code="trial_unavailable",
                error_class="retryable",
                subcode="broker_backoff",
                retry_after_ms=5000,
                message="broker is temporarily unavailable",
            ),
            retry_after_ms=5000,
        )
    )

    async def fail_fetch_key_metadata(_api_key: str):
        raise AssertionError("fetch_key_metadata should not run without a managed key")

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecrets({}),
    )
    monkeypatch.setattr(
        OpenRouterLLMProvider,
        "fetch_key_metadata",
        staticmethod(fail_fetch_key_metadata),
    )

    await controller.set_translation_enabled(True)

    assert controller._managed_openrouter_release_service.prepare_calls == 1
    assert controller.hub.translation_enabled is False
    assert controller.hub.clear_context_calls == 0
    assert dash.managed_auth_pending is False
    assert dash.managed_auth_pending_calls == [True, False]
    assert snackbar_calls == [
        (t("managed_release.retry_after_ms", retry_after_ms=5000), ft.Colors.ORANGE_700)
    ]
    assert (
        logging.ERROR,
        "[ManagedAuth] operation=issue code=trial_unavailable class=retryable subcode=broker_backoff retry_after_ms=5000 message=<redacted>",
    ) in controller._runtime_logging.basic_messages
    assert settings_view.managed_trial_usage_state == {
        "visible": True,
        "remaining_percent": None,
    }
    assert dash.managed_trial_calls == []


def test_on_managed_trial_delegate_ready_clears_dashboard_pending_notice() -> None:
    dash = DummyDashboard()
    controller = _make_controller(app=SimpleNamespace(view_dashboard=dash))

    controller._get_managed_auth_owner().set_pending(True)

    controller._on_managed_trial_delegate_ready()

    assert controller.managed_auth_pending is False
    assert dash.managed_auth_pending is False
    assert dash.managed_auth_pending_calls == [True, False]


@pytest.mark.asyncio
async def test_managed_trial_delegate_refresh_is_cancelled_by_managed_status_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dash = DummyDashboard()
    controller = _make_controller(app=SimpleNamespace(view_dashboard=dash))
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocked_refresh(self) -> None:
        _ = self
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(
        ManagedUsageOwner,
        "refresh_best_effort",
        blocked_refresh,
    )

    controller._on_managed_trial_delegate_ready()
    await entered.wait()

    managed_usage_owner = controller._managed_usage_owner
    assert managed_usage_owner is not None
    owner = managed_usage_owner.refresh_owner
    assert owner.active_task_names
    controller._freeze_application_ingress()
    await owner.close()

    assert cancelled.is_set()
    assert owner.active_task_names == ()

    controller._on_managed_trial_delegate_ready()
    assert owner.active_task_names == ()


def test_application_ingress_freeze_is_terminal_for_late_work_owners() -> None:
    app = SimpleNamespace(
        set_dashboard_overlay_session_fallback_notice=lambda _active: None,
        set_dashboard_vrchat_osc_notice=lambda _active: None,
    )
    controller = _make_controller(app=app)
    overlay_owner = controller._get_overlay_application_owner().fallback_owner
    osc_owner = controller._get_vrchat_osc_presence_owner()
    mic_owner = controller._get_vrc_mic_sync_owner()

    controller._freeze_application_ingress()

    assert overlay_owner.accepting_ingress is False
    assert osc_owner.accepting_ingress is False
    assert mic_owner.accepting_ingress is False


@pytest.mark.asyncio
async def test_set_translation_enabled_off_wins_against_inflight_managed_enable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dash = DummyDashboard()
    controller = _make_controller(app=SimpleNamespace(view_dashboard=dash))
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.hub = DummyHub(llm=object())
    prepare_started = asyncio.Event()
    release_prepare = asyncio.Event()

    async def block_prepare() -> None:
        prepare_started.set()
        await release_prepare.wait()

    controller._managed_openrouter_release_service = InspectingManagedReleaseService(
        ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            message_key="managed_release.ready",
            api_key="managed-key",
            local_key_available=True,
            pending_issue=False,
        ),
        on_prepare=block_prepare,
    )

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecrets({}),
    )
    monkeypatch.setattr(
        ManagedUsageOwner,
        "schedule_usage_refresh",
        lambda self: None,
    )

    enable_task = asyncio.create_task(controller.set_translation_enabled(True))
    await prepare_started.wait()

    await controller.set_translation_enabled(False)

    assert controller.hub.translation_enabled is False
    assert controller.managed_auth_pending is False
    assert dash.managed_auth_pending is False

    release_prepare.set()
    await enable_task

    assert controller._managed_openrouter_release_service.prepare_calls == 1
    assert controller.hub.translation_enabled is False
    assert controller.hub.clear_context_calls == 1
    assert controller.managed_auth_pending is False
    assert dash.managed_auth_pending is False
    assert dash.managed_auth_pending_calls[:2] == [True, False]
    assert dash.managed_auth_pending_calls[-1] is False


@pytest.mark.asyncio
async def test_apply_providers_clears_dashboard_pending_notice_when_switching_away_from_managed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dash = DummyDashboard()
    controller = _make_controller(app=SimpleNamespace(view_dashboard=dash))
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.hub = DummyHub(llm=object())

    controller._get_managed_auth_owner().set_pending(True)

    next_settings = AppSettings()
    next_settings.provider.llm = LLMProviderName.GEMINI

    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    monkeypatch.setattr(
        provider_runtime_apply_module.LlmProviderRebuildOwner,
        "rebuild",
        lambda self: asyncio.sleep(0),
    )

    await controller.apply_providers(next_settings)

    assert controller.managed_auth_pending is False
    assert dash.managed_auth_pending is False
    assert dash.managed_auth_pending_calls == [True, False]


@pytest.mark.asyncio
async def test_apply_providers_force_rebuild_local_llm_reads_updated_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ClosableLLM:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    async def noop_replace_managed_service(self, service) -> None:
        _ = (self, service)

    async def noop_refresh_managed_usage(self) -> None:
        _ = self

    settings = AppSettings(provider=ProviderSettings(llm=LLMProviderName.LOCAL_LLM))
    controller = _make_controller(
        app=SimpleNamespace(view_dashboard=DummyDashboard(), view_settings=DummySettingsView())
    )
    controller.settings = settings
    previous_llm = ClosableLLM()
    controller.hub = DummyHub(llm=previous_llm)

    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    monkeypatch.setattr(
        GuiController,
        "_create_managed_openrouter_release_service",
        lambda self, *, secrets: object(),
    )
    monkeypatch.setattr(
        GuiController,
        "_replace_managed_openrouter_release_service",
        noop_replace_managed_service,
    )
    monkeypatch.setattr(
        GuiController,
        "_refresh_managed_trial_usage_state_best_effort",
        noop_refresh_managed_usage,
    )
    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_a, **_k: DummySecrets({"local_llm_api_key": "new-secret"}),
    )

    await controller.apply_providers(force_rebuild_llm=True)

    assert previous_llm.closed is True
    assert isinstance(controller.hub.llm, SemaphoreLLMProvider)
    assert controller.hub.llm.inner.api_key == "new-secret"


@pytest.mark.asyncio
async def test_set_translation_enabled_clears_dashboard_pending_notice_when_prepare_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dash = DummyDashboard()
    controller = _make_controller(app=SimpleNamespace(view_dashboard=dash))
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.hub = DummyHub(llm=object())
    controller._managed_openrouter_release_service = FailingManagedReleaseService(
        RuntimeError("boom")
    )

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecrets({}),
    )

    with pytest.raises(RuntimeError, match="boom"):
        await controller.set_translation_enabled(True)

    assert controller.managed_auth_pending is False
    assert dash.managed_auth_pending is False
    assert dash.managed_auth_pending_calls == [True, False]


@pytest.mark.asyncio
async def test_apply_providers_resyncs_dashboard_pending_notice_when_staying_on_managed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dash = DummyDashboard()
    controller = _make_controller(app=SimpleNamespace(view_dashboard=dash))
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.hub = DummyHub(llm=object())

    controller._get_managed_auth_owner().set_pending(True)

    next_settings = AppSettings()
    next_settings.provider.llm = LLMProviderName.OPENROUTER
    next_settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED

    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    monkeypatch.setattr(
        provider_runtime_apply_module.LlmProviderRebuildOwner,
        "rebuild",
        lambda self: asyncio.sleep(0),
    )

    await controller.apply_providers(next_settings)

    assert controller.managed_auth_pending is True
    assert dash.managed_auth_pending is True
    assert dash.managed_auth_pending_calls == [True, True]


@pytest.mark.asyncio
async def test_apply_providers_staying_on_managed_does_not_prepare_managed_translation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.hub = DummyHub(llm=object())

    class TrackingManagedReleaseService:
        def __init__(self) -> None:
            self.prepare_calls = 0
            self.close_calls = 0

        async def prepare_for_translation(self):
            self.prepare_calls += 1
            raise AssertionError("apply_providers must not prepare managed translation")

        async def close(self) -> None:
            self.close_calls += 1

    initial_service = TrackingManagedReleaseService()
    created_services: list[TrackingManagedReleaseService] = []
    controller._managed_openrouter_release_service = initial_service

    updated = copy.deepcopy(controller.settings)
    updated.openrouter.provider_routing = OpenRouterProviderRouting.DEEPSEEK_ONLY

    async def fake_refresh_managed_usage(self) -> None:
        return None

    def fake_create_managed_release_service(self, *, secrets):
        _ = (self, secrets)
        service = TrackingManagedReleaseService()
        created_services.append(service)
        return service

    _patch_settings_save(monkeypatch, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(controller_module, "create_secret_store", lambda *_a, **_k: object())
    monkeypatch.setattr(controller_module, "create_llm_provider", lambda *_a, **_k: object())
    monkeypatch.setattr(
        GuiController,
        "_create_managed_openrouter_release_service",
        fake_create_managed_release_service,
    )
    monkeypatch.setattr(
        GuiController,
        "_refresh_managed_trial_usage_state_best_effort",
        fake_refresh_managed_usage,
    )

    await controller.apply_providers(updated)

    assert initial_service.close_calls == 1
    assert len(created_services) == 1
    assert created_services[0].prepare_calls == 0
    assert controller.settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED


def test_dashboard_trans_missing_managed_key_opens_discord_auth_dialog_without_prepare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = SimpleNamespace(tasks=[], run_task=lambda task: app.page.tasks.append(task))
    dash = DummyDashboard()
    app.view_dashboard = dash
    dialog_calls: list[bool] = []
    app.show_discord_managed_auth_dialog = lambda *, preview=False: dialog_calls.append(preview)

    controller = _make_controller(app=app)
    app.controller = controller
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.hub = DummyHub(llm=object())
    controller._managed_openrouter_release_service = DummyManagedReleaseService(
        ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            message_key="managed_release.ready",
            api_key="managed-key",
            local_key_available=True,
        )
    )

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecrets({}),
    )

    handled = app._on_translation_toggle(True)

    assert handled is False
    assert dialog_calls == [False]
    assert dash.translation_enabled is False
    assert app.page.tasks == []
    assert controller._managed_openrouter_release_service.prepare_calls == 0


def test_dashboard_trans_in_progress_managed_oauth_prevents_second_dialog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = SimpleNamespace(tasks=[], run_task=lambda task: app.page.tasks.append(task))
    dash = DummyDashboard()
    app.view_dashboard = dash
    dialog_calls: list[bool] = []
    app.show_discord_managed_auth_dialog = lambda *, preview=False: dialog_calls.append(preview)

    controller = _make_controller(app=app)
    app.controller = controller
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.hub = DummyHub(llm=object())
    controller._get_managed_auth_owner().set_pending(True)
    controller._managed_openrouter_release_service = DummyManagedReleaseService(
        ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            message_key="managed_release.ready",
            api_key="managed-key",
            local_key_available=True,
        )
    )

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecrets({}),
    )

    handled = app._on_translation_toggle(True)

    assert handled is False
    assert dialog_calls == []
    assert dash.translation_enabled is False
    assert app.page.tasks == []
    assert controller._managed_openrouter_release_service.prepare_calls == 0


def test_managed_connection_auth_settings_values_are_service_safe() -> None:
    from puripuly_heart.app.services import managed_connection_auth

    settings = AppSettings()
    settings.secrets.backend = SecretsBackend.ENCRYPTED_FILE

    values = managed_auth_runtime_module._managed_connection_auth_settings_values(settings)

    assert managed_connection_auth._caller_settings_values_are_unsafe(
        legacy_settings_snapshot_values(settings)
    )
    assert not managed_connection_auth._caller_settings_values_are_unsafe(values)


@pytest.mark.asyncio
async def test_discord_managed_auth_transaction_rebuilds_existing_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.hub = DummyHub(llm=object())
    controller._managed_openrouter_release_service = SimpleNamespace(
        app_version="test",
        client=object(),
        raw_hardware_fingerprint_provider=None,
        _legacy_hardware_hash_provider=None,
        oauth_runtime=object(),
        discord_oauth_listener_factory=object(),
        discord_oauth_callback_runner=object(),
        openrouter_config=controller_module.build_openrouter_release_runtime_config(
            controller.settings
        ),
        signed_at_provider=lambda: "2026-01-01T00:00:00Z",
    )
    rebuild_calls: list[str] = []

    async def fake_authorize(_self, _request):
        return messages.TransactionResult(
            status=messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
            message=None,
            diagnostics=None,
        )

    async def fake_rebuild_llm_provider(_owner) -> None:
        rebuild_calls.append("rebuild")
        assert controller.hub is not None
        controller.hub.llm = object()

    monkeypatch.setattr(
        controller_module, "create_secret_store", lambda *_args, **_kwargs: DummySecrets({})
    )
    _patch_settings_save(monkeypatch, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ManagedConnectionAuthService, "authorize", fake_authorize)
    monkeypatch.setattr(
        provider_runtime_apply_module.LlmProviderRebuildOwner, "rebuild", fake_rebuild_llm_provider
    )

    ok = await controller.start_discord_managed_auth_from_dialog()

    assert ok is True
    assert rebuild_calls == ["rebuild"]


@pytest.mark.asyncio
async def test_discord_managed_auth_pending_ack_installs_runtime_settings_without_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.hub = DummyHub(llm=object())
    controller._managed_openrouter_release_service = SimpleNamespace(
        app_version="test",
        client=object(),
        raw_hardware_fingerprint_provider=None,
        _legacy_hardware_hash_provider=None,
        oauth_runtime=object(),
        discord_oauth_listener_factory=object(),
        discord_oauth_callback_runner=object(),
        openrouter_config=controller_module.build_openrouter_release_runtime_config(
            controller.settings
        ),
        signed_at_provider=lambda: "2026-01-01T00:00:00Z",
    )
    rebuild_calls: list[str] = []

    async def fake_authorize(service, _request):
        service.settings_repository.committed_settings.managed_identity.pending_delivery_ack_source = (
            "discord"
        )
        service.settings_repository.committed_settings.managed_identity.pending_delivery_ack_delivery_id = (
            "delivery-discord"
        )
        service.settings_repository.committed_settings.managed_identity.pending_delivery_ack_managed_credential_ref = (
            "managed-ref-discord"
        )
        return messages.TransactionResult(
            status=messages.TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING,
            message=None,
            diagnostics=None,
        )

    async def fake_rebuild_llm_provider(self: GuiController) -> None:
        rebuild_calls.append("rebuild")

    monkeypatch.setattr(
        controller_module, "create_secret_store", lambda *_args, **_kwargs: DummySecrets({})
    )
    _patch_settings_save(monkeypatch, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ManagedConnectionAuthService, "authorize", fake_authorize)
    monkeypatch.setattr(
        provider_runtime_apply_module.LlmProviderRebuildOwner, "rebuild", fake_rebuild_llm_provider
    )

    ok = await controller.start_discord_managed_auth_from_dialog()

    assert ok is False
    assert rebuild_calls == []
    assert controller.settings.managed_identity.pending_delivery_ack_source == "discord"
    assert (
        controller.settings.managed_identity.pending_delivery_ack_delivery_id == "delivery-discord"
    )


@pytest.mark.asyncio
async def test_start_discord_managed_auth_from_dialog_success_rebuilds_missing_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dash = DummyDashboard()
    controller = _make_controller(app=SimpleNamespace(view_dashboard=dash))
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.hub = DummyHub(llm=None)
    controller._managed_openrouter_release_service = DummyManagedReleaseService(
        ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            message_key="managed_release.ready",
            api_key="managed-key",
            local_key_available=True,
        )
    )
    rebuild_calls: list[str] = []

    async def fake_rebuild_llm_provider(_owner) -> None:
        rebuild_calls.append("rebuild")
        controller.hub.llm = object()

    monkeypatch.setattr(
        provider_runtime_apply_module.LlmProviderRebuildOwner, "rebuild", fake_rebuild_llm_provider
    )

    ok = await controller.start_discord_managed_auth_from_dialog()

    assert ok is True
    assert controller._managed_openrouter_release_service.prepare_calls == 1
    assert rebuild_calls == ["rebuild"]
    assert controller.hub.llm is not None
    assert controller.managed_auth_pending is False
    assert dash.managed_auth_pending_calls == [True, False]


@pytest.mark.asyncio
async def test_start_discord_managed_auth_from_dialog_passes_referral_id_without_persisting_friend_id() -> (
    None
):
    dash = DummyDashboard()
    controller = _make_controller(app=SimpleNamespace(view_dashboard=dash))
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.hub = DummyHub(llm=object())
    service = DummyManagedReleaseService(
        ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            message_key="managed_release.ready",
            api_key="managed-key",
            local_key_available=True,
        )
    )
    controller._managed_openrouter_release_service = service

    ok = await controller.start_discord_managed_auth_from_dialog(referral_id=" 7kq9m2 ")

    assert ok is True
    assert service.prepare_referral_ids == [" 7kq9m2 "]
    assert controller.settings.managed_identity.referral_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("referral_bonus_applied", "expected"),
    [(True, True), (False, False), (None, False), ("true", False), (1, False)],
)
async def test_start_discord_managed_auth_from_dialog_exposes_only_boolean_true_referral_bonus(
    referral_bonus_applied: object,
    expected: bool,
) -> None:
    dash = DummyDashboard()
    controller = _make_controller(app=SimpleNamespace(view_dashboard=dash))
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.hub = DummyHub(llm=object())
    controller._managed_openrouter_release_service = DummyManagedReleaseService(
        ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            message_key="managed_release.ready",
            api_key="managed-key",
            local_key_available=True,
            referral_bonus_applied=referral_bonus_applied,  # type: ignore[arg-type]
        )
    )

    ok = await controller.start_discord_managed_auth_from_dialog()

    assert ok is True
    assert controller.last_discord_managed_auth_referral_bonus_applied is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result_referral_id", "persisted_referral_id", "expected_referral_id"),
    [
        ("7kq9m2", None, "7KQ9M2"),
        (None, "7KQ9M2", "7KQ9M2"),
    ],
)
async def test_start_discord_managed_auth_from_dialog_updates_managed_key_referral_row_immediately(
    monkeypatch: pytest.MonkeyPatch,
    result_referral_id: str | None,
    persisted_referral_id: str | None,
    expected_referral_id: str,
) -> None:
    dash = DummyDashboard()

    class ManagedKeySettingsView(DummySettingsView):
        def __init__(self) -> None:
            super().__init__()
            self.managed_key_state_calls: list[dict[str, object]] = []

        def set_managed_key_state(
            self,
            *,
            visible: bool,
            remaining_percent: int | None = None,
            referral_id: str | None = None,
            pass_status: object | None = None,
        ) -> None:
            self.managed_key_state_calls.append(
                {
                    "visible": visible,
                    "remaining_percent": remaining_percent,
                    "referral_id": referral_id,
                    "pass_status": pass_status,
                }
            )

    settings_view = ManagedKeySettingsView()
    controller = _make_controller(
        app=SimpleNamespace(view_dashboard=dash, view_settings=settings_view)
    )
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.settings.managed_identity.referral_id = persisted_referral_id
    controller.hub = DummyHub(llm=object())
    controller._managed_openrouter_release_service = DummyManagedReleaseService(
        ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            message_key="managed_release.ready",
            api_key="managed-key",
            local_key_available=True,
            referral_id=result_referral_id,
        )
    )
    scheduled_refreshes: list[str] = []
    monkeypatch.setattr(
        ManagedUsageOwner,
        "schedule_usage_refresh",
        lambda self: scheduled_refreshes.append("usage"),
    )

    ok = await controller.start_discord_managed_auth_from_dialog()

    assert ok is True
    assert settings_view.managed_key_state_calls == [
        {
            "visible": True,
            "remaining_percent": None,
            "referral_id": expected_referral_id,
            "pass_status": None,
        }
    ]
    assert scheduled_refreshes == ["usage"]


@pytest.mark.asyncio
async def test_start_discord_managed_auth_from_dialog_repaints_pass_status_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass_status = TalkTogetherPassStatus(
        pass_id="7KQ9M2",
        invite_count=1,
        invite_limit=5,
        bonus_translations_per_friend=200,
    )
    dash = DummyDashboard()
    settings_view = CapturingManagedKeySettingsView()
    controller = _make_controller(
        app=SimpleNamespace(view_dashboard=dash, view_settings=settings_view)
    )
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.hub = DummyHub(llm=object())
    controller._managed_openrouter_release_service = DummyManagedReleaseService(
        ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            message_key="managed_release.ready",
            api_key="managed-key",
            local_key_available=True,
            referral_id="7KQ9M2",
            pass_status=pass_status,
        )
    )
    scheduled_refreshes: list[str] = []
    monkeypatch.setattr(
        ManagedUsageOwner,
        "schedule_usage_refresh",
        lambda self: scheduled_refreshes.append("usage"),
    )

    ok = await controller.start_discord_managed_auth_from_dialog()

    assert ok is True
    assert settings_view.managed_key_state_calls == [
        {
            "visible": True,
            "remaining_percent": None,
            "referral_id": "7KQ9M2",
            "pass_status": pass_status,
        }
    ]
    assert scheduled_refreshes == ["usage"]


@pytest.mark.asyncio
async def test_start_discord_managed_auth_from_dialog_issue_success_does_not_repaint_stale_usage_percent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dash = DummyDashboard()

    class ManagedKeySettingsView(DummySettingsView):
        def __init__(self) -> None:
            super().__init__()
            self.managed_key_state_calls: list[dict[str, object]] = []

        def set_managed_key_state(
            self,
            *,
            visible: bool,
            remaining_percent: int | None = None,
            referral_id: str | None = None,
            pass_status: object | None = None,
        ) -> None:
            self.managed_key_state_calls.append(
                {
                    "visible": visible,
                    "remaining_percent": remaining_percent,
                    "referral_id": referral_id,
                    "pass_status": pass_status,
                }
            )

    settings_view = ManagedKeySettingsView()
    controller = _make_controller(
        app=SimpleNamespace(view_dashboard=dash, view_settings=settings_view)
    )
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.settings.managed_identity.active_managed_credential_ref = "new-ref"
    controller.hub = DummyHub(llm=object())
    controller._get_managed_usage_owner().usage_metadata = OpenRouterKeyMetadata(
        limit_usd=0.10,
        remaining_usd=0.02,
        usage_usd=0.08,
    )
    controller._get_managed_usage_owner().usage_metadata_entitlement_ref = "old-ref"
    controller._managed_openrouter_release_service = DummyManagedReleaseService(  # noqa: SLF001
        ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            message_key="managed_release.ready",
            api_key="managed-key",
            local_key_available=True,
            referral_id="7KQ9M2",
        )
    )
    scheduled_refreshes: list[str] = []
    monkeypatch.setattr(
        ManagedUsageOwner,
        "schedule_usage_refresh",
        lambda self: scheduled_refreshes.append("usage"),
    )

    ok = await controller.start_discord_managed_auth_from_dialog()

    assert ok is True
    assert settings_view.managed_key_state_calls == [
        {
            "visible": True,
            "remaining_percent": None,
            "referral_id": "7KQ9M2",
            "pass_status": None,
        }
    ]
    assert scheduled_refreshes == ["usage"]


def test_discord_managed_auth_callback_received_runs_active_hook_only() -> None:
    calls: list[str] = []
    controller = _make_controller(app=SimpleNamespace())
    controller._get_managed_auth_owner().callback_received_hook = lambda: calls.append("received")

    controller._on_discord_managed_auth_callback_received()

    assert calls == ["received"]


@pytest.mark.asyncio
async def test_start_discord_managed_auth_from_dialog_rebuild_failure_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snackbar_calls: list[tuple[str, str]] = []
    controller = _make_controller(
        app=SimpleNamespace(
            _show_snackbar=lambda message, color: snackbar_calls.append((message, color))
        )
    )
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.hub = DummyHub(llm=None)
    controller._managed_openrouter_release_service = DummyManagedReleaseService(
        ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            message_key="managed_release.ready",
            api_key="managed-key",
            local_key_available=True,
        )
    )
    rebuild_calls: list[str] = []

    async def fake_rebuild_llm_provider(_owner) -> None:
        rebuild_calls.append("rebuild")
        controller.hub.llm = None

    monkeypatch.setattr(
        provider_runtime_apply_module.LlmProviderRebuildOwner, "rebuild", fake_rebuild_llm_provider
    )
    monkeypatch.setattr(presentation_adapter_module, "t", lambda key, **_kwargs: key)

    ok = await controller.start_discord_managed_auth_from_dialog()

    assert ok is False
    assert rebuild_calls == ["rebuild"]
    assert controller.hub.llm is None
    assert snackbar_calls == [("discord_auth.error.retry", ft.Colors.ORANGE_700)]


@pytest.mark.asyncio
async def test_start_discord_managed_auth_from_dialog_does_not_log_raw_broker_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snackbar_calls: list[tuple[str, str]] = []
    controller = _make_controller(
        app=SimpleNamespace(
            _show_snackbar=lambda message, color: snackbar_calls.append((message, color))
        )
    )
    controller._runtime_logging = RuntimeLoggingSpy()
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.hub = DummyHub(llm=object())
    raw_subcode = "discord_email_unverified"
    raw_message = "raw broker eligibility message"
    controller._managed_openrouter_release_service = DummyManagedReleaseService(
        ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.STOP,
            message_key="managed_release.not_eligible",
            diagnostics=ManagedOpenRouterReleaseDiagnostics(
                operation="discord_issue",
                code="trial_not_eligible",
                error_class="terminal",
                subcode=raw_subcode,
                message=raw_message,
            ),
        )
    )
    monkeypatch.setattr(presentation_adapter_module, "t", lambda key, **_kwargs: key)

    ok = await controller.start_discord_managed_auth_from_dialog()

    assert ok is False
    assert snackbar_calls == [("discord_auth.error.email_unverified", ft.Colors.ORANGE_700)]
    logged_messages = [message for _level, message in controller._runtime_logging.basic_messages]
    assert not any(raw_subcode in message for message in logged_messages)
    assert not any(raw_message in message for message in logged_messages)


def test_discord_auth_message_key_falls_back_to_result_message_key() -> None:
    result = ManagedOpenRouterReleaseResult(
        behavior=ManagedOpenRouterReleaseBehavior.RETRY,
        message_key="managed_release.retry_after_ms",
        message_kwargs={"retry_after_ms": 5000},
    )

    assert (
        managed_auth_runtime_module._discord_auth_message_key(result)
        == "managed_release.retry_after_ms"
    )


def test_discord_auth_message_key_maps_loopback_bind_failure_diagnostic() -> None:
    result = ManagedOpenRouterReleaseResult(
        behavior=ManagedOpenRouterReleaseBehavior.RETRY,
        message_key="managed_release.retry",
        diagnostics=ManagedOpenRouterReleaseDiagnostics(
            operation="discord_start",
            code="discord_loopback_unavailable",
            error_class="retryable",
            message="bind failed",
        ),
    )

    assert managed_auth_runtime_module._discord_auth_message_key(result) == (
        "discord_auth.error.loopback_unavailable"
    )


def test_verified_key_and_runtime_signature_depend_on_region_and_settings() -> None:
    controller = _make_controller(app=SimpleNamespace())
    settings = AppSettings()
    controller.settings = settings

    settings.qwen.region = QwenRegion.BEIJING
    key_beijing = controller._get_alibaba_verified_key()
    settings.qwen.region = QwenRegion.SINGAPORE
    key_singapore = controller._get_alibaba_verified_key()

    baseline = build_self_stt_runtime_signature(settings)
    settings.audio.input_device = "Microphone 2"
    changed = build_self_stt_runtime_signature(settings)

    assert key_beijing == "alibaba_beijing"
    assert key_singapore == "alibaba_singapore"
    assert baseline != changed


def test_build_llm_provider_signature_tracks_openrouter_selection_and_fallback_branch() -> None:
    controller = _make_controller(app=SimpleNamespace())
    base = AppSettings()
    base.provider.llm = LLMProviderName.OPENROUTER
    base.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    base.openrouter.selection_alias = OpenRouterSelectionAlias.GEMMA4_MANAGED
    base.translation.fallback = TranslationFallbackSettings(
        enabled=True,
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.OPENROUTER,
    )

    different_selection = copy.deepcopy(base)
    different_selection.openrouter.selection_alias = OpenRouterSelectionAlias.QWEN35_FLASH_MANAGED

    different_fallback = copy.deepcopy(base)
    different_fallback.translation.fallback = TranslationFallbackSettings(enabled=False)

    assert controller._build_llm_provider_signature(
        base
    ) != controller._build_llm_provider_signature(different_selection)
    assert controller._build_llm_provider_signature(
        base
    ) != controller._build_llm_provider_signature(different_fallback)


def test_build_llm_provider_signature_tracks_managed_fallback_identity() -> None:
    controller = _make_controller(app=SimpleNamespace())
    base = AppSettings()
    base.provider.llm = LLMProviderName.GEMINI
    base.translation.fallback = TranslationFallbackSettings(
        enabled=True,
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.MANAGED_CHINA,
    )

    different_identity = copy.deepcopy(base)
    different_identity.managed_identity.verified_hardware_hash = "fallback-managed-hash"

    assert controller._build_llm_provider_signature(
        base
    ) != controller._build_llm_provider_signature(different_identity)


def test_build_llm_provider_signature_tracks_openrouter_provider_routing() -> None:
    controller = _make_controller(app=SimpleNamespace())
    base = AppSettings()
    base.provider.llm = LLMProviderName.OPENROUTER
    base.openrouter.selection_alias = OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED
    base.openrouter.provider_routing = OpenRouterProviderRouting.DEFAULT

    deepseek_only = copy.deepcopy(base)
    deepseek_only.openrouter.provider_routing = OpenRouterProviderRouting.DEEPSEEK_ONLY

    assert controller._build_llm_provider_signature(
        base
    ) != controller._build_llm_provider_signature(deepseek_only)


def test_build_llm_provider_signature_tracks_local_llm_runtime_fields() -> None:
    controller = _make_controller(app=SimpleNamespace())
    base = AppSettings()
    base.provider.llm = LLMProviderName.LOCAL_LLM
    base.local_llm = LocalLLMSettings(
        backend=LocalLLMBackend.OLLAMA,
        base_url="http://127.0.0.1:11434/v1",
        model="llama3.1:8b",
        extra_body={"thinking": {"type": "disabled", "budget": 0}},
    )

    same_json_different_order = copy.deepcopy(base)
    same_json_different_order.local_llm.extra_body = {"thinking": {"budget": 0, "type": "disabled"}}
    changed_model = copy.deepcopy(base)
    changed_model.local_llm.model = "qwen2.5:7b"
    changed_body = copy.deepcopy(base)
    changed_body.local_llm.extra_body = {"enable_thinking": False}

    assert controller._build_llm_provider_signature(
        base
    ) == controller._build_llm_provider_signature(same_json_different_order)
    assert controller._build_llm_provider_signature(
        base
    ) != controller._build_llm_provider_signature(changed_model)
    assert controller._build_llm_provider_signature(
        base
    ) != controller._build_llm_provider_signature(changed_body)


def test_merge_settings_tab_apply_copies_translation_selection_for_provider_apply() -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.translation = TranslationSettings(
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.MANAGED,
        connection_history={
            TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.MANAGED,
        },
    )

    pending = copy.deepcopy(controller.settings)
    pending.translation = TranslationSettings(
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.MANAGED_CHINA,
        connection_history={
            TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.MANAGED_CHINA,
        },
    )
    pending.openrouter.provider_routing = OpenRouterProviderRouting.DEEPSEEK_ONLY

    merged = controller.merge_settings_tab_apply_with_current_languages(pending)

    assert merged.translation.connection == TranslationConnection.MANAGED_CHINA
    assert (
        merged.translation.connection_history[TranslationModel.DEEPSEEK_V4_FLASH.value]
        == TranslationConnection.MANAGED_CHINA
    )
    assert merged.openrouter.provider_routing == OpenRouterProviderRouting.DEEPSEEK_ONLY


def test_merge_settings_tab_apply_copies_local_llm_settings_for_provider_apply() -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.GEMINI

    pending = copy.deepcopy(controller.settings)
    pending.provider.llm = LLMProviderName.LOCAL_LLM
    pending.translation = TranslationSettings(
        model=TranslationModel.LOCAL_LLM,
        connection=TranslationConnection.OLLAMA,
        connection_history={TranslationModel.LOCAL_LLM.value: TranslationConnection.OLLAMA},
    )
    pending.local_llm = LocalLLMSettings(
        backend=LocalLLMBackend.OLLAMA,
        base_url="http://mac-studio.local:11434/v1",
        model="gemma3:4b",
        extra_body={"think": False},
    )

    merged = controller.merge_settings_tab_apply_with_current_languages(pending)

    assert merged.provider.llm == LLMProviderName.LOCAL_LLM
    assert merged.translation.model == TranslationModel.LOCAL_LLM
    assert merged.translation.connection == TranslationConnection.OLLAMA
    assert merged.local_llm.base_url == "http://mac-studio.local:11434/v1"
    assert merged.local_llm.model == "gemma3:4b"
    assert merged.local_llm.extra_body == {"think": False}


def test_stt_runtime_signature_includes_custom_vocabulary_state() -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.DEEPGRAM
    settings.languages.source_language = "ko"
    settings.stt.custom_terms = {"ko": [" Puripuly ", "VRChat", "Puripuly"], "en": ["Avatar"]}
    settings.stt.custom_vocabulary_enabled = False

    disabled_signature = build_self_stt_runtime_signature(settings)

    settings.stt.custom_vocabulary_enabled = True
    enabled_signature = build_self_stt_runtime_signature(settings)

    assert disabled_signature != enabled_signature
    assert enabled_signature[-2] is True
    assert enabled_signature[-1] == ("Puripuly", "VRChat")


def test_stt_runtime_signature_includes_source_language() -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.DEEPGRAM
    settings.languages.source_language = "ko"

    ko_signature = build_self_stt_runtime_signature(settings)
    settings.languages.source_language = "en"
    en_signature = build_self_stt_runtime_signature(settings)

    assert ko_signature != en_signature
    assert ko_signature[0] == "ko"
    assert en_signature[0] == "en"


def test_stt_runtime_signature_differs_between_plain_wasapi_and_compatibility_mode() -> None:
    plain = AppSettings()
    plain.audio.input_host_api = WINDOWS_WASAPI_HOST_API
    compat = copy.deepcopy(plain)
    compat.audio.input_host_api = WINDOWS_WASAPI_COMPATIBILITY_HOST_API

    assert build_self_stt_runtime_signature(plain) != build_self_stt_runtime_signature(compat)


def test_stt_runtime_signature_ignores_custom_vocabulary_for_qwen_asr() -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.QWEN_ASR
    settings.languages.source_language = "ko"
    settings.stt.custom_terms = {"ko": ["Puripuly", "VRChat"]}

    disabled_signature = build_self_stt_runtime_signature(settings)

    settings.stt.custom_vocabulary_enabled = True
    enabled_signature = build_self_stt_runtime_signature(settings)

    assert disabled_signature == enabled_signature
    assert enabled_signature[-2] is False
    assert enabled_signature[-1] == ()


def test_stt_runtime_signature_uses_capped_custom_vocabulary_for_local_qwen() -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.LOCAL_QWEN
    settings.languages.source_language = "ko"
    settings.stt.custom_terms = {"ko": [f"term-{i:02d}" for i in range(20)]}
    settings.stt.custom_vocabulary_enabled = False

    disabled_signature = build_self_stt_runtime_signature(settings)

    settings.stt.custom_vocabulary_enabled = True
    enabled_signature = build_self_stt_runtime_signature(settings)

    assert disabled_signature != enabled_signature
    assert enabled_signature[-2] is True
    assert enabled_signature[-1] == tuple(f"term-{i:02d}" for i in range(12))


def test_peer_runtime_config_disables_self_custom_vocabulary() -> None:
    controller = _make_controller(app=SimpleNamespace())
    settings = AppSettings()
    settings.languages.peer_source_language = "zh-CN"
    settings.stt.custom_vocabulary_enabled = True
    settings.stt.custom_terms = {
        "ko": ["Puripuly"],
        "zh-CN": ["airi", "shinano"],
    }

    backend = _peer_runtime_config(controller, settings).provider_context

    assert backend.custom_vocabulary_enabled is False
    assert backend.custom_terms == {}


def test_self_stt_runtime_signature_ignores_overlay_and_peer_desktop_settings() -> None:
    controller = _make_controller(app=SimpleNamespace())
    settings = AppSettings()

    baseline = build_self_stt_runtime_signature(settings)

    settings.ui.peer_translation_enabled = True
    _overlay_owner(controller).state = "connected"
    settings.desktop_audio.output_device = "Headphones (Loopback)"
    settings.desktop_audio.vad_speech_threshold = 0.72
    settings.desktop_audio.vad_hangover_ms = 950
    settings.desktop_audio.vad_pre_roll_ms = 420
    changed = build_self_stt_runtime_signature(settings)

    assert baseline == changed


def test_peer_stt_runtime_signature_includes_peer_desktop_settings() -> None:
    controller = _make_controller(app=SimpleNamespace())
    settings = AppSettings()

    baseline = _peer_runtime_signature(controller, settings)

    settings.ui.peer_translation_enabled = True
    settings.desktop_audio.output_device = "Headphones (Loopback)"
    settings.desktop_audio.vad_speech_threshold = 0.72
    settings.desktop_audio.vad_hangover_ms = 950
    settings.desktop_audio.vad_pre_roll_ms = 420
    changed = _peer_runtime_signature(controller, settings)

    assert baseline != changed


def test_peer_stt_runtime_signature_includes_peer_source_language() -> None:
    controller = _make_controller(app=SimpleNamespace())
    settings = AppSettings()

    baseline = _peer_runtime_signature(controller, settings)

    settings.languages.peer_source_language = "zh-CN"
    changed = _peer_runtime_signature(controller, settings)

    assert baseline != changed


def test_peer_runtime_uses_canonical_vnext_intent_over_legacy_projection() -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.provider.peer_stt = STTProviderName.DEEPGRAM
    controller.settings.languages.peer_source_mode = "manual"
    vnext_settings = AppSettingsVNext()
    controller.vnext_settings = replace(
        vnext_settings,
        intent=replace(
            vnext_settings.intent,
            peer_stt=replace(vnext_settings.intent.peer_stt, provider="soniox"),
            languages=replace(
                vnext_settings.intent.languages,
                peer_source_mode="auto",
                peer_expected_languages=["ja"],
            ),
        ),
    )
    controller._get_settings_owner().authoritative = True

    config = _peer_runtime_config(controller, controller.settings)

    assert config.backend.provider == "soniox"
    assert config.backend.provider_options["enable_language_identification"] is True
    assert config.backend.provider_options["language_hints"] == ("ja",)


def test_direct_peer_settings_mutation_refreshes_canonical_runtime_intent() -> None:
    controller = _make_controller(app=SimpleNamespace())
    settings = AppSettings()
    settings.provider.peer_stt = STTProviderName.SONIOX
    controller.settings = settings
    vnext_settings = AppSettingsVNext()
    controller.vnext_settings = replace(
        vnext_settings,
        intent=replace(
            vnext_settings.intent,
            peer_stt=replace(vnext_settings.intent.peer_stt, provider="soniox"),
        ),
    )
    controller._get_settings_owner().authoritative = True

    pending = copy.deepcopy(settings)
    pending.languages.peer_source_mode = "auto"
    pending.languages.peer_expected_languages = ["ja"]
    pending.desktop_audio.output_device = "Headphones (Loopback)"
    pending.desktop_audio.vad_speech_threshold = 0.72

    controller._get_settings_owner().apply_legacy_delta(controller.settings, pending)
    config = _peer_runtime_config(controller, pending)

    assert controller.vnext_settings.intent.languages.peer_source_mode == "auto"
    assert controller.vnext_settings.intent.languages.peer_expected_languages == ["ja"]
    assert config.backend.provider_options["enable_language_identification"] is True
    assert config.output_device == "Headphones (Loopback)"
    assert config.vad_threshold == 0.72


def test_unrelated_legacy_apply_preserves_canonical_peer_auto_intent_after_save_reload(
    tmp_path: Path,
) -> None:
    from puripuly_heart.config.settings_vnext.facade import load_vnext_settings

    controller = _make_controller(app=SimpleNamespace())
    controller.config_path = tmp_path / "settings.json"
    legacy = AppSettings()
    legacy.provider.peer_stt = STTProviderName.DEEPGRAM
    controller.settings = legacy
    vnext_settings = AppSettingsVNext()
    controller.vnext_settings = replace(
        vnext_settings,
        intent=replace(
            vnext_settings.intent,
            peer_stt=replace(vnext_settings.intent.peer_stt, provider="soniox"),
            languages=replace(
                vnext_settings.intent.languages,
                peer_source_mode="auto",
                peer_expected_languages=["ja"],
            ),
        ),
    )
    controller._get_settings_owner().authoritative = True
    expected_signature = _peer_provider_signature(controller, legacy)

    pending = copy.deepcopy(legacy)
    pending.ui.locale = "ja"
    controller._get_settings_owner().begin()
    controller._get_settings_owner().apply_legacy_delta(controller.settings, pending)
    controller.settings = pending
    controller._get_settings_owner().persist_current()

    loaded = load_vnext_settings(controller.config_path)
    runtime = _peer_runtime_config(controller, pending)

    assert loaded.settings is not None
    assert loaded.settings.intent.languages.peer_source_mode == "auto"
    assert loaded.settings.intent.languages.peer_expected_languages == ["ja"]
    assert runtime.backend.provider == "soniox"
    assert runtime.backend.provider_options["enable_language_identification"] is True
    assert _peer_provider_signature(controller, pending) == expected_signature


def test_failed_canonical_persistence_rolls_back_peer_auto_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    legacy = AppSettings()
    legacy_before_mutation = copy.deepcopy(legacy)
    controller.settings = legacy
    vnext_settings = AppSettingsVNext()
    canonical = replace(
        vnext_settings,
        intent=replace(
            vnext_settings.intent,
            peer_stt=replace(vnext_settings.intent.peer_stt, provider="soniox"),
            languages=replace(
                vnext_settings.intent.languages,
                peer_source_mode="auto",
                peer_expected_languages=["ja"],
            ),
        ),
    )
    controller.vnext_settings = canonical
    controller._get_settings_owner().authoritative = True
    pending = copy.deepcopy(legacy)
    pending.ui.locale = "ja"
    controller._get_settings_owner().begin()
    controller._get_settings_owner().apply_legacy_delta(controller.settings, pending)
    controller.settings = pending
    monkeypatch.setattr(
        canonical_persistence_adapter_module,
        "save_vnext_settings",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("save failed")),
    )

    controller._get_settings_owner().save_current()
    runtime = _peer_runtime_config(controller, pending)

    assert controller.vnext_settings == canonical
    assert controller.settings == legacy_before_mutation
    assert runtime.backend.provider == "soniox"
    assert runtime.backend.provider_options["language_hints"] == ("ja",)


def test_stale_managed_adapter_persists_only_managed_delta_on_current_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    stale_settings = AppSettings()
    state = controller_module.build_managed_identity_state_port(
        stale_settings,
        controller._get_settings_owner().managed_identity_persistence_callback(stale_settings),
    )
    active_settings = copy.deepcopy(stale_settings)
    active_settings.ui.locale = "ja"
    controller.settings = active_settings
    vnext_settings = AppSettingsVNext()
    controller.vnext_settings = replace(
        vnext_settings,
        intent=replace(
            vnext_settings.intent,
            ui=replace(vnext_settings.intent.ui, locale="ja"),
            peer_stt=replace(vnext_settings.intent.peer_stt, provider="soniox"),
            languages=replace(
                vnext_settings.intent.languages,
                peer_source_mode="auto",
                peer_expected_languages=["ja"],
            ),
        ),
    )
    controller._get_settings_owner().authoritative = True
    controller._get_settings_owner().remember_projection(active_settings)
    saved: list[AppSettingsVNext] = []
    monkeypatch.setattr(
        canonical_persistence_adapter_module,
        "save_vnext_settings",
        lambda _path, settings: saved.append(settings) or SimpleNamespace(ok=True),
    )

    state.referral_id = "234567"
    state.persist()

    assert controller.settings.ui.locale == "ja"
    assert controller.settings.managed_identity.referral_id == "234567"
    assert len(saved) == 1
    assert saved[0].intent.ui.locale == "ja"
    assert saved[0].state.managed_connection.referral_id == "234567"
    assert saved[0].intent.peer_stt.provider == "soniox"
    assert saved[0].intent.languages.peer_source_mode == "auto"
    assert saved[0].intent.languages.peer_expected_languages == ["ja"]


def test_failed_stale_managed_adapter_persistence_restores_active_and_bound_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    stale_settings = AppSettings()
    state = controller_module.build_managed_identity_state_port(
        stale_settings,
        controller._get_settings_owner().managed_identity_persistence_callback(stale_settings),
    )
    active_settings = copy.deepcopy(stale_settings)
    active_settings.ui.locale = "ja"
    active_before_mutation = copy.deepcopy(active_settings)
    stale_before_mutation = copy.deepcopy(stale_settings)
    controller.settings = active_settings
    vnext_settings = AppSettingsVNext()
    canonical = replace(
        vnext_settings,
        intent=replace(
            vnext_settings.intent,
            ui=replace(vnext_settings.intent.ui, locale="ja"),
            peer_stt=replace(vnext_settings.intent.peer_stt, provider="soniox"),
            languages=replace(
                vnext_settings.intent.languages,
                peer_source_mode="auto",
                peer_expected_languages=["ja"],
            ),
        ),
    )
    controller.vnext_settings = canonical
    controller._get_settings_owner().authoritative = True
    controller._get_settings_owner().remember_projection(active_settings)
    monkeypatch.setattr(
        canonical_persistence_adapter_module,
        "save_vnext_settings",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("save failed")),
    )

    state.referral_id = "234567"
    with pytest.raises(OSError, match="save failed"):
        state.persist()

    assert controller.settings == active_before_mutation
    assert stale_settings == stale_before_mutation
    assert controller.vnext_settings == canonical


def test_nested_canonical_completion_keeps_outer_rollback_snapshot() -> None:
    controller = _make_controller(app=SimpleNamespace())
    legacy = AppSettings()
    legacy_before_mutation = copy.deepcopy(legacy)
    controller.settings = legacy
    vnext_settings = AppSettingsVNext()
    canonical = replace(
        vnext_settings,
        intent=replace(
            vnext_settings.intent,
            peer_stt=replace(vnext_settings.intent.peer_stt, provider="soniox"),
        ),
    )
    controller.vnext_settings = canonical
    controller._get_settings_owner().authoritative = True

    controller._get_settings_owner().begin()
    controller.settings.ui.locale = "ja"
    controller._get_settings_owner().begin()
    controller._get_settings_owner().complete()

    assert controller._get_settings_owner().rollback_pending is True
    assert controller._get_settings_owner().mutation_depth == 1
    controller._get_settings_owner().rollback()

    assert controller.settings == legacy_before_mutation
    assert controller.vnext_settings == canonical
    assert controller._get_settings_owner().mutation_depth == 0


@pytest.mark.asyncio
async def test_settings_repository_commits_only_scoped_delta_to_canonical_vnext(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    legacy = AppSettings()
    controller.settings = legacy
    vnext_settings = AppSettingsVNext()
    controller.vnext_settings = replace(
        vnext_settings,
        intent=replace(
            vnext_settings.intent,
            peer_stt=replace(vnext_settings.intent.peer_stt, provider="soniox"),
            languages=replace(
                vnext_settings.intent.languages,
                peer_source_mode="auto",
                peer_expected_languages=["ja"],
            ),
        ),
    )
    controller._get_settings_owner().authoritative = True
    stale_full_draft = copy.deepcopy(legacy)
    stale_full_draft.provider.peer_stt = STTProviderName.DEEPGRAM
    saved: list[AppSettingsVNext] = []
    monkeypatch.setattr(
        canonical_persistence_adapter_module,
        "save_vnext_settings",
        lambda _path, settings: saved.append(settings) or SimpleNamespace(ok=True),
    )
    repository = controller._get_settings_owner().create_legacy_patch_repository(
        base_settings=legacy,
        committed_settings=stale_full_draft,
        surface="ui_prompt_clipboard_state",
        save_failure_sink=controller._log_error,
    )

    result = await repository.save(
        SettingsCommitRequest(
            values={"ui.locale": "ja"},
            expected_revision=None,
            reason="settings.ui_prompt_clipboard_state",
        )
    )

    assert result.succeeded is True
    assert len(saved) == 1
    assert saved[0].intent.languages.peer_source_mode == "auto"
    assert saved[0].intent.languages.peer_expected_languages == ["ja"]
    assert saved[0].intent.peer_stt.provider == "soniox"
    assert saved[0].intent.ui.locale == "ja"


@pytest.mark.asyncio
async def test_failed_scoped_persistence_restores_canonical_and_legacy_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    legacy = AppSettings()
    controller.settings = legacy
    vnext_settings = AppSettingsVNext()
    canonical = replace(
        vnext_settings,
        intent=replace(
            vnext_settings.intent,
            peer_stt=replace(vnext_settings.intent.peer_stt, provider="soniox"),
            languages=replace(
                vnext_settings.intent.languages,
                peer_source_mode="auto",
                peer_expected_languages=["ja"],
            ),
        ),
    )
    controller.vnext_settings = canonical
    controller._get_settings_owner().authoritative = True
    controller._get_settings_owner().remember_projection(legacy)
    runtime_calls: list[str] = []
    monkeypatch.setattr(
        canonical_persistence_adapter_module,
        "save_vnext_settings",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("save failed")),
    )
    monkeypatch.setattr(
        GuiController,
        "_refresh_peer_stt_runtime",
        lambda self: runtime_calls.append("peer") or asyncio.sleep(0),
    )

    updated = copy.deepcopy(legacy)
    updated.languages.peer_source_mode = "manual"
    updated.languages.peer_source_language = "ko"
    await controller.apply_settings(updated)

    assert controller.settings == legacy
    assert controller.vnext_settings == canonical
    assert runtime_calls == []


def test_dashboard_sync_does_not_swallow_supported_setter_type_error() -> None:
    calls: list[tuple[object, ...]] = []

    class FailingDashboard:
        def set_languages_from_codes(self, *values: object) -> None:
            calls.append(values)
            raise TypeError("dashboard failure")

    controller = _make_controller(app=SimpleNamespace(view_dashboard=FailingDashboard()))
    controller.settings = AppSettings()

    with pytest.raises(TypeError, match="dashboard failure"):
        controller._sync_ui_from_settings()

    assert len(calls) == 1
    assert len(calls[0]) == 5


def test_build_peer_runtime_config_includes_provider_signature_and_desktop_settings() -> None:
    controller = _make_controller(app=SimpleNamespace())
    settings = AppSettings()
    settings.provider.peer_stt = STTProviderName.SONIOX
    settings.desktop_audio.output_device = "Headphones (Loopback)"
    settings.desktop_audio.vad_speech_threshold = 0.72
    settings.desktop_audio.vad_hangover_ms = 950
    settings.desktop_audio.vad_pre_roll_ms = 420

    config = _peer_runtime_config(controller, settings)

    assert config.backend.provider == STTProviderName.SONIOX
    assert config.output_device == "Headphones (Loopback)"
    assert config.vad_threshold == 0.72
    assert config.runtime_signature == (
        config.backend.source_language,
        config.output_device,
        config.capture_target,
        config.vad_threshold,
        config.vad_hangover_ms,
        config.vad_pre_roll_ms,
        config.provider_signature,
    )


@pytest.mark.asyncio
async def test_apply_settings_updates_peer_translation_flags_on_hub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.hub = DummyHub(llm=object(), stt=object(), peer_stt=object())
    _overlay_owner(controller).state = "connected"
    controller._last_self_stt_runtime_signature = build_self_stt_runtime_signature(
        controller.settings
    )
    controller._last_peer_stt_runtime_signature = _peer_runtime_signature(
        controller, controller.settings
    )
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    monkeypatch.setattr(
        GuiController, "_replace_runtime_stt_provider", lambda self: asyncio.sleep(0)
    )
    monkeypatch.setattr(GuiController, "_refresh_peer_stt_runtime", lambda self: asyncio.sleep(0))

    updated = AppSettings()
    updated.ui.peer_translation_enabled = True
    updated.ui.peer_translation_eula_accepted = True
    updated.ui.integrated_context_enabled = True

    await controller.apply_settings(updated)

    assert controller.hub.peer_translation_enabled is True
    assert controller.hub.integrated_context_enabled is True


@pytest.mark.asyncio
async def test_apply_settings_copies_self_and_peer_vad_hangovers_to_hub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    settings = AppSettings()
    settings.stt.low_latency_mode = True
    settings.stt.low_latency_vad_hangover_ms = 650
    settings.desktop_audio.vad_hangover_ms = 950
    controller.settings = settings
    controller.hub = DummyHub(llm=object(), stt=object(), peer_stt=object())
    controller._last_self_stt_runtime_signature = build_self_stt_runtime_signature(settings)
    controller._last_peer_stt_runtime_signature = _peer_runtime_signature(controller, settings)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    monkeypatch.setattr(
        GuiController, "_replace_runtime_stt_provider", lambda self: asyncio.sleep(0)
    )
    monkeypatch.setattr(GuiController, "_refresh_peer_stt_runtime", lambda self: asyncio.sleep(0))

    await controller.apply_settings(settings)

    assert controller.hub.hangover_s == 0.65
    assert controller.hub.peer_hangover_s == 0.95


@pytest.mark.asyncio
async def test_set_peer_translation_enabled_enqueues_peer_disclosure_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_locale("ko")
    try:
        controller = _make_controller(app=SimpleNamespace())
        controller.settings = AppSettings()
        controller.settings.ui.peer_translation_eula_accepted = True
        controller.hub = DisclosureDummyHub(llm=object(), stt=object(), peer_stt=object())
        _overlay_owner(controller).state = "connected"
        monkeypatch.setattr(
            SettingsOwner,
            "save_current",
            lambda self, **_kwargs: True,
        )
        monkeypatch.setattr(
            GuiController,
            "_refresh_overlay_runtime_dependencies",
            lambda self: asyncio.sleep(0),
        )

        await controller.set_peer_translation_enabled(True)

        expected_disclosure = t(PEER_DISCLOSURE_KEY)
        assert expected_disclosure != PEER_DISCLOSURE_KEY
        assert controller.hub.disclosures == [expected_disclosure]
    finally:
        set_locale("en")


@pytest.mark.asyncio
async def test_overlay_composition_replacement_cancels_old_owner_delivery() -> None:
    class BlockingOverlaySink:
        def __init__(self) -> None:
            self.events: list[object] = []
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def emit(self, event: object) -> None:
            self.events.append(event)
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

        def active_self_overlay_metadata(self) -> None:
            return None

    class RecordingOverlaySink:
        def __init__(self) -> None:
            self.events: list[object] = []

        async def emit(self, event: object) -> None:
            self.events.append(event)

        def active_self_overlay_metadata(self) -> None:
            return None

    controller = _make_controller(app=SimpleNamespace())
    old_sink = BlockingOverlaySink()
    replacement = RecordingOverlaySink()
    hub = ClientHub(
        stt=None,
        llm=None,
        osc=ChatboxPaginator(sender=FakeSender(), clock=FakeClock()),
        overlay_sink=old_sink,
    )
    controller.hub = hub
    old_event = hub.overlay_event_adapter.utterance_closed(
        utterance_id=uuid4(),
        channel="peer",
        is_final=True,
    )

    await hub.start()
    old_publication = asyncio.create_task(hub._emit_overlay_event(old_event))
    await asyncio.wait_for(old_sink.started.wait(), timeout=0.5)
    replaced = await _overlay_owner(controller).replace_hub_sink(replacement)
    await old_publication
    new_id = await hub.submit_text("replacement composition", source="You")

    assert replaced is True
    assert old_sink.cancelled.is_set()
    assert not hub.output_runtime.has_active_overlay_deliveries
    assert hub.overlay_sink is replacement
    assert old_sink.events == [old_event]
    assert [getattr(event, "utterance_id", None) for event in replacement.events] == [
        new_id,
        new_id,
    ]

    await hub.stop()


@pytest.mark.asyncio
async def test_init_pipeline_keeps_peer_original_runtime_available_without_peer_translation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _patch_init_pipeline_dependencies(monkeypatch)
    monkeypatch.setattr(
        GuiController, "_configure_vrc_mic_receiver", lambda self, enabled: asyncio.sleep(0)
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.ui.overlay_enabled = True
    _overlay_owner(controller).state = "connected"

    await controller._init_pipeline()

    hub = created["hub"]
    assert hub.peer_stt is None
    assert hub.peer_translation_enabled is False
    assert controller._peer_runtime is not None


@pytest.mark.asyncio
async def test_init_pipeline_passes_chatbox_and_peer_language_settings_to_hub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    llm_create_kwargs: dict[str, object] = {}

    monkeypatch.setattr(controller_module, "create_secret_store", lambda *_a, **_k: object())

    def fake_create_llm_provider(*_args, **kwargs):
        llm_create_kwargs.update(kwargs)
        return "llm"

    monkeypatch.setattr(controller_module, "create_llm_provider", fake_create_llm_provider)
    monkeypatch.setattr(controller_module, "VrchatOscUdpSender", lambda *a, **k: object())
    monkeypatch.setattr(controller_module, "ChatboxPaginator", lambda *a, **k: object())

    def fake_hub(*_args, **kwargs):
        captured.update(kwargs)
        hub = SimpleNamespace(
            llm=kwargs.get("llm"),
            stt=kwargs.get("stt"),
            peer_stt=kwargs.get("peer_stt"),
            local_asr_provider_runtime=None,
        )

        async def replace_stt_provider_request(request, *, start):
            _ = request, start
            hub.stt = "owned-stt"
            return SimpleNamespace(status="applied")

        hub.has_stt_provider = lambda channel: (
            hub.stt is not None if channel == "self" else hub.peer_stt is not None
        )
        hub.replace_stt_provider_request = replace_stt_provider_request
        return hub

    monkeypatch.setattr(controller_module, "ClientHub", fake_hub)
    monkeypatch.setattr(
        GuiController, "_configure_vrc_mic_receiver", lambda self, enabled: asyncio.sleep(0)
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.osc.chatbox_include_source = False
    controller.settings.languages.peer_source_language = "ja"
    controller.settings.languages.peer_target_language = "en"

    await controller._init_pipeline()

    assert captured["chatbox_include_source"] is False
    assert captured["peer_source_language"] == "ja"
    assert captured["peer_target_language"] == "en"
    assert llm_create_kwargs["runtime_logging"] is controller.runtime_logging


@pytest.mark.asyncio
async def test_init_pipeline_constructs_production_output_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[object] = []

    class ProviderFactory:
        async def create(
            self,
            request,
            *,
            gpu_runtime,
            on_terminal_failure=None,
        ):
            _ = gpu_runtime, on_terminal_failure
            requests.append(request)
            return object()

    class Chatbox:
        def enqueue(self, message) -> None:
            _ = message

        def send_typing(self, is_typing: bool) -> None:
            _ = is_typing

        def set_typing_reason(self, reason: str, active: bool) -> None:
            _ = (reason, active)

        def clear_typing_reasons(self) -> None:
            return

        def process_due(self) -> None:
            return

        def send_immediate(self, text: str) -> bool:
            _ = text
            return True

    chatbox = Chatbox()
    monkeypatch.setattr(controller_module, "create_secret_store", lambda *_a, **_k: object())
    monkeypatch.setattr(controller_module, "create_llm_provider", lambda *_a, **_k: None)
    monkeypatch.setattr(
        controller_module,
        "ManagedSTTProviderFactory",
        lambda **_kwargs: ProviderFactory(),
    )
    monkeypatch.setattr(controller_module, "VrchatOscUdpSender", lambda *a, **k: object())
    monkeypatch.setattr(controller_module, "ChatboxPaginator", lambda *a, **k: chatbox)
    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        lambda self, enabled: asyncio.sleep(0),
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()

    await controller._init_pipeline()

    assert type(controller.hub) is controller_module.ClientHub
    assert controller.hub.output_runtime.chatbox is chatbox
    assert controller.hub.output_runtime.overlay_sink is None
    assert controller.hub.output_runtime.state == "open"
    assert controller.hub.local_asr_provider_runtime is not None
    assert (
        controller.hub.local_asr_provider_runtime.snapshot.channel_for("self").provider_id
        == controller.settings.provider.stt.value
    )
    assert set(controller.hub.provider_runtime_handles) == {"llm"}
    assert [request.provider_id for request in requests] == [controller.settings.provider.stt.value]
    assert controller.local_asr_provisioning is not None
    assert controller.local_asr_provisioning.lifecycle_owner_snapshot()["owner"] == (
        "LocalASRProvisioningOwner"
    )


@pytest.mark.asyncio
async def test_real_controller_composition_routes_all_output_channels_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingChatbox:
        def __init__(self) -> None:
            self.messages: list[object] = []
            self.typing_reasons: set[str] = set()

        def enqueue(self, message: object) -> None:
            self.messages.append(message)

        def send_typing(self, is_typing: bool) -> None:
            _ = is_typing

        def set_typing_reason(self, reason: str, active: bool) -> None:
            if active:
                self.typing_reasons.add(reason)
            else:
                self.typing_reasons.discard(reason)

        def clear_typing_reasons(self) -> None:
            self.typing_reasons.clear()

        def process_due(self) -> None:
            return

        def send_immediate(self, text: str) -> bool:
            _ = text
            return True

    class RecordingOverlay:
        def __init__(self) -> None:
            self.events: list[object] = []

        async def emit(self, event: object) -> None:
            self.events.append(event)

        def active_self_overlay_metadata(self) -> None:
            return None

    chatbox = RecordingChatbox()
    overlay = RecordingOverlay()
    monkeypatch.setattr(controller_module, "create_secret_store", lambda *_a, **_k: object())
    monkeypatch.setattr(controller_module, "create_llm_provider", lambda *_a, **_k: None)
    monkeypatch.setattr(controller_module, "VrchatOscUdpSender", lambda *a, **k: object())
    monkeypatch.setattr(controller_module, "ChatboxPaginator", lambda *a, **k: chatbox)
    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        lambda self, enabled: asyncio.sleep(0),
    )
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()

    await controller._init_pipeline()
    hub = controller.hub
    assert type(hub) is controller_module.ClientHub
    await _overlay_owner(controller).replace_hub_sink(overlay)
    await controller.submit_text("manual self text")
    manual_id = getattr(chatbox.messages[0], "utterance_id")
    peer_id = await hub.handle_peer_transcript_final_for_test("peer presentation text")
    hub.enqueue_peer_translation_disclosure("system disclosure")

    chatbox_ids = [getattr(message, "utterance_id") for message in chatbox.messages]
    assert len(chatbox_ids) == 2
    assert chatbox_ids[0] == manual_id
    assert chatbox_ids[1] not in {manual_id, peer_id}
    assert [getattr(message, "text") for message in chatbox.messages] == [
        "manual self text",
        "system disclosure",
    ]
    assert [getattr(event, "channel") for event in overlay.events] == [
        "self",
        "self",
        "peer",
        "peer",
    ]
    assert [getattr(event, "utterance_id") for event in overlay.events] == [
        manual_id,
        manual_id,
        peer_id,
        peer_id,
    ]
    assert [getattr(event, "type") for event in overlay.events] == [
        "self_transcript_final",
        "utterance_closed",
        "peer_transcript_final",
        "utterance_closed",
    ]
    peer_denials = [
        decision
        for decision in hub.output_runtime.routing_decisions
        if decision.publication_kind == "peer_subtitle" and decision.route == "self_chatbox"
    ]
    assert len(peer_denials) == 1
    assert peer_denials[0].reason == "peer_chatbox_denied"
    assert all(
        "peer presentation text" not in getattr(message, "text") for message in chatbox.messages
    )

    await hub.stop()
    assert hub.output_runtime.state == "closed"
    assert not hub.output_runtime.has_resources


@pytest.mark.asyncio
async def test_initial_peer_local_activation_publishes_starting_until_provider_attaches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contracts = []
    app = SimpleNamespace(
        view_dashboard=SimpleNamespace(
            set_overlay_peer_contract=contracts.append,
        )
    )
    controller = _make_controller(app=app)
    controller.settings = AppSettings()
    controller.settings.provider.peer_stt = STTProviderName.LOCAL_QWEN
    controller.settings.ui.peer_translation_enabled = True
    controller.settings.ui.peer_translation_eula_accepted = True
    controller.hub = DummyHub(llm=object(), stt=object(), peer_stt=None)
    _overlay_owner(controller).state = "connected"
    _attach_overlay_bridge(controller, object())
    activation_started = asyncio.Event()
    finish_activation = asyncio.Event()

    class DelayedPeerRuntime:
        current_signature = None

        async def apply_policy(self, *, config, desired_active) -> None:
            _ = config
            assert desired_active is True
            activation_started.set()
            await finish_activation.wait()
            controller.hub.peer_stt = object()

    async def ready(_self: GuiController, **_kwargs) -> bool:
        return True

    controller._peer_runtime = DelayedPeerRuntime()
    monkeypatch.setattr(GuiController, "_ensure_peer_local_stt_ready", ready)

    task = asyncio.create_task(controller._refresh_peer_stt_runtime())
    await activation_started.wait()

    assert controller._peer_asr_model_loading is True
    assert contracts[-1].peer.state == "starting"

    finish_activation.set()
    await task

    assert controller._peer_asr_model_loading is False
    assert contracts[-1].peer.state == "on"


@pytest.mark.asyncio
async def test_refresh_overlay_runtime_dependencies_applies_peer_runtime_policy() -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.provider.peer_stt = STTProviderName.SONIOX
    controller.settings.ui.peer_translation_enabled = True
    controller.settings.ui.peer_translation_eula_accepted = True
    controller.hub = DummyHub(llm=object(), stt=object(), peer_stt=None)
    _overlay_owner(controller).state = "connected"
    _attach_overlay_bridge(controller, object())

    peer_runtime = DummyPeerRuntime()
    controller._peer_runtime = peer_runtime

    await controller._refresh_overlay_runtime_dependencies()

    assert len(peer_runtime.policy_calls) == 1
    assert peer_runtime.policy_calls[0]["desired_active"] is True


@pytest.mark.asyncio
async def test_refresh_overlay_runtime_dependencies_disables_peer_runtime_when_overlay_fails() -> (
    None
):
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.ui.peer_translation_enabled = True
    controller.settings.ui.peer_translation_eula_accepted = True
    controller.hub = DummyHub(llm=object(), stt=object(), peer_stt=object())
    _overlay_owner(controller).state = "failed"
    _attach_overlay_bridge(controller, None)

    peer_runtime = DummyPeerRuntime()
    controller._peer_runtime = peer_runtime

    await controller._refresh_overlay_runtime_dependencies()

    assert peer_runtime.policy_calls[-1]["desired_active"] is False
    assert controller.hub.replace_stt_calls == []


def test_dashboard_stt_needs_key_remains_self_oriented_when_peer_provider_differs() -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.provider.stt = STTProviderName.LOCAL_QWEN
    controller.settings.provider.peer_stt = STTProviderName.DEEPGRAM

    assert controller._dashboard_stt_needs_key(stt_available=True) is False


@pytest.mark.asyncio
async def test_overlay_toggle_starts_and_stops_overlay_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    monkeypatch.setattr(controller_module, "user_config_dir", lambda: tmp_path)

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)

    manager = FakeOverlayProcessManager.instances[0]
    bridge = FakeOverlayBridge.instances[0]

    assert manager.extra_kwargs["log_dir"] == str(tmp_path)
    assert controller.settings.ui.overlay_enabled is True
    assert _overlay_owner(controller).snapshot.state == "starting"
    assert controller.hub.overlay_sink is _overlay_runtime(controller).presenter
    assert bridge.started is True

    manager.complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")

    assert _overlay_owner(controller).snapshot.failure_reason is None

    await controller.set_overlay_enabled(False)

    assert controller.settings.ui.overlay_enabled is False
    assert _overlay_owner(controller).snapshot.state == "off"
    assert controller.hub.overlay_sink is None
    assert controller.hub.reset_overlay_preview_calls == 1
    assert bridge.stopped is True
    assert manager.stop_calls == 1


@pytest.mark.asyncio
async def test_closing_desktop_overlay_runtime_rejects_direct_bridge_commands() -> None:
    from puripuly_heart.core.runtime.overlay import OverlayRuntimeHandle

    class BlockingShutdownOverlayBridge(FakeOverlayBridge):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.shutdown_entered = asyncio.Event()
            self.shutdown_released = asyncio.Event()

        async def broadcast_shutdown(self) -> None:
            await super().broadcast_shutdown()
            self.shutdown_entered.set()
            await self.shutdown_released.wait()

    class FakePage:
        def __init__(self) -> None:
            self.tasks: list[object] = []

        def run_task(self, coro_fn) -> None:
            self.tasks.append(coro_fn)

    page = FakePage()
    controller = GuiController(
        page=page,
        app=_presentation(SimpleNamespace(), page=page),
        config_path=Path("settings.json"),
    )
    controller.settings = AppSettings()
    controller.settings.ui.overlay_enabled = True
    controller.settings.overlay.target = OVERLAY_TARGET_DESKTOP
    _overlay_owner(controller).active_target = OVERLAY_TARGET_DESKTOP
    _overlay_owner(controller).state = "connected"
    controller.hub = DummyHub()
    controller._runtime_logging = RuntimeLoggingSpy(detailed_enabled=True)

    bridge = BlockingShutdownOverlayBridge(session_token="token")
    runtime = OverlayRuntimeHandle(shutdown_grace_s=0)
    runtime.attach_bridge(bridge)
    _overlay_owner(controller).runtime = runtime
    _attach_overlay_bridge(controller, bridge)

    close_task = asyncio.create_task(
        runtime.close(
            preserve_presenter_state=False,
            hub=controller.hub,
            emit_shutdown=True,
        )
    )
    await _wait_until(lambda: bridge.shutdown_entered.is_set())

    try:
        assert runtime.is_closing is True
        assert _overlay_runtime(controller).bridge is bridge

        sent = await controller._get_desktop_overlay_application_owner().broadcast(
            {"command": "set_interaction_mode", "mode": "edit"}
        )

        assert sent is False
        assert bridge.desktop_runtime_control_payloads == []
        assert (
            controller._get_desktop_overlay_application_owner().runtime_is_running_for_settings(
                controller.settings
            )
            is False
        )

        await controller._emit_overlay_runtime_logging_mode_update()

        assert bridge.runtime_control_messages == []

        controller._schedule_overlay_runtime_logging_mode_update()

        assert page.tasks == []
    finally:
        bridge.shutdown_released.set()
        await close_task


@pytest.mark.asyncio
async def test_closing_overlay_runtime_rejects_direct_presenter_commands() -> None:
    from puripuly_heart.core.runtime.overlay import OverlayRuntimeHandle

    class BlockingShutdownPresenter:
        def __init__(self) -> None:
            self.shutdown_entered = asyncio.Event()
            self.shutdown_released = asyncio.Event()
            self.calibration_updates: list[OverlayCalibration] = []
            self.display_preferences: list[dict[str, bool]] = []

        async def broadcast_shutdown(self) -> None:
            self.shutdown_entered.set()
            await self.shutdown_released.wait()

        async def update_calibration(self, calibration: OverlayCalibration) -> None:
            self.calibration_updates.append(calibration)

        async def update_display_preferences(
            self,
            *,
            show_translation: bool,
            show_peer_original: bool,
        ) -> None:
            self.display_preferences.append(
                {
                    "show_translation": show_translation,
                    "show_peer_original": show_peer_original,
                }
            )

    class FakePage:
        def __init__(self) -> None:
            self.tasks: list[object] = []

        def run_task(self, coro_fn) -> None:
            self.tasks.append(coro_fn)

    page = FakePage()
    controller = GuiController(
        page=page,
        app=_presentation(SimpleNamespace(), page=page),
        config_path=Path("settings.json"),
    )
    controller.settings = AppSettings()
    _overlay_owner(controller).state = "connected"
    controller._last_vrc_mic_sync_enabled = controller.settings.osc.vrc_mic_intercept
    presenter = BlockingShutdownPresenter()
    runtime = OverlayRuntimeHandle(shutdown_grace_s=0)
    runtime.attach_presenter(presenter)
    _overlay_owner(controller).runtime = runtime
    _attach_overlay_presenter(controller, presenter)

    close_task = asyncio.create_task(
        runtime.close(
            preserve_presenter_state=True,
            hub=None,
            emit_shutdown=True,
        )
    )
    await _wait_until(lambda: presenter.shutdown_entered.is_set())

    try:
        assert runtime.is_closing is True
        assert _overlay_runtime(controller).presenter is presenter

        controller._get_overlay_calibration_application_owner().schedule_emit()

        assert page.tasks == []

        await controller._get_overlay_calibration_application_owner().emit_current()

        assert presenter.calibration_updates == []

        pending = copy.deepcopy(controller.settings)
        pending.overlay.show_translation = not pending.overlay.show_translation

        await controller._get_settings_application_owner().apply_direct(
            pending,
            persist=False,
            reload_settings_view=False,
        )

        assert presenter.display_preferences == []
    finally:
        presenter.shutdown_released.set()
        await close_task


@pytest.mark.asyncio
async def test_overlay_teardown_close_failure_falls_back_to_basic_runtime_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from puripuly_heart.core.runtime.overlay import OverlayRuntimeHandle

    class FailingStopManager:
        async def stop(self) -> None:
            raise RuntimeError("raw cleanup failure details must stay detailed-only")

    detailed_calls: list[tuple[str, int, BaseException | None]] = []
    basic_calls: list[tuple[str, int]] = []
    controller = _make_controller(app=SimpleNamespace())
    controller.hub = DummyHub()
    runtime = OverlayRuntimeHandle(shutdown_grace_s=0)
    runtime.attach_process_manager(FailingStopManager())
    _overlay_owner(controller).runtime = runtime

    def fake_log_detailed(
        self: GuiController,
        message: str,
        *,
        level: int = logging.INFO,
        exception: BaseException | None = None,
    ) -> bool:
        assert self is controller
        detailed_calls.append((message, level, exception))
        return False

    def fake_log_basic(
        self: GuiController,
        message: str,
        *,
        level: int = logging.INFO,
    ) -> None:
        assert self is controller
        basic_calls.append((message, level))

    monkeypatch.setattr(GuiController, "log_detailed", fake_log_detailed)
    monkeypatch.setattr(GuiController, "log_basic", fake_log_basic)

    await _overlay_owner(controller).teardown(preserve_presenter_state=True)

    warning = "[Overlay] Overlay runtime close reported cleanup failure"
    assert len(detailed_calls) == 1
    assert detailed_calls[0][:2] == (warning, logging.WARNING)
    assert isinstance(detailed_calls[0][2], RuntimeError)
    assert basic_calls == [(warning, logging.WARNING)]
    assert "RuntimeError" not in basic_calls[0][0]
    assert "raw cleanup failure" not in basic_calls[0][0]


@pytest.mark.asyncio
async def test_stale_desktop_renderer_event_is_ignored_after_overlay_instance_change() -> None:
    from puripuly_heart.core.runtime.overlay import OverlayRuntimeHandle

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.overlay.target = OVERLAY_TARGET_DESKTOP
    _overlay_owner(controller).active_target = OVERLAY_TARGET_DESKTOP
    _attach_overlay_bridge(controller, FakeOverlayBridge(session_token="token"))
    _overlay_owner(controller).runtime = OverlayRuntimeHandle(overlay_instance_id="overlay-new")

    await controller._get_desktop_overlay_application_owner().handle_renderer_event(
        {
            "type": "overlay_event",
            "payload": {
                "event": "window_bounds_changed",
                "source": "user",
                "persist": True,
                "x": 111,
                "y": 222,
                "width": 1152,
                "height": 288,
            },
        },
        overlay_instance_id="overlay-old",
    )

    assert controller._get_desktop_overlay_application_owner().bounds_owner.pending_bounds is None


@pytest.mark.asyncio
async def test_overlay_target_routing_installs_steamvr_runner_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)

    class FakeSteamVrRunner:
        pass

    class FakeDesktopRunner:
        pass

    monkeypatch.setattr(
        overlay_application_module,
        "DefaultOverlayProcessRunner",
        FakeSteamVrRunner,
        raising=False,
    )
    monkeypatch.setattr(
        overlay_application_module,
        "DesktopFletOverlayRunner",
        FakeDesktopRunner,
        raising=False,
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.overlay.target = "steamvr"
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)

    manager = FakeOverlayProcessManager.instances[0]
    assert isinstance(manager.process_runner, FakeSteamVrRunner)
    assert not isinstance(manager.process_runner, FakeDesktopRunner)

    manager.complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")
    await controller.set_overlay_enabled(False)


@pytest.mark.asyncio
async def test_overlay_session_fallback_to_desktop_when_steamvr_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    notices: list[bool] = []

    class FakeSteamVrRunner:
        pass

    class FakeDesktopRunner:
        pass

    monkeypatch.setattr(
        overlay_application_module,
        "DefaultOverlayProcessRunner",
        FakeSteamVrRunner,
        raising=False,
    )
    monkeypatch.setattr(
        overlay_application_module,
        "DesktopFletOverlayRunner",
        FakeDesktopRunner,
        raising=False,
    )

    controller = _make_controller(
        app=SimpleNamespace(
            view_dashboard=SimpleNamespace(
                set_overlay_session_fallback_notice=lambda active: notices.append(bool(active))
            )
        )
    )
    controller.settings = AppSettings()
    controller.settings.overlay.target = "steamvr"
    controller.settings.ui.overlay_enabled = False
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)
    first = FakeOverlayProcessManager.instances[0]
    assert isinstance(first.process_runner, FakeSteamVrRunner)
    first.complete_startup(failure_reason="steamvr_not_running")

    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 2)
    second = FakeOverlayProcessManager.instances[1]
    assert isinstance(second.process_runner, FakeDesktopRunner)
    assert controller.settings.overlay.target == "steamvr"
    assert controller._get_overlay_application_owner().fallback_owner.active is True
    assert True in notices
    second.complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")
    assert _overlay_owner(controller).snapshot.active_target == "desktop"

    await controller.set_overlay_enabled(False)
    assert controller._get_overlay_application_owner().fallback_owner.active is False
    assert False in notices


@pytest.mark.asyncio
async def test_desktop_initial_control_manifest_always_launches_edit_even_with_legacy_locked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    state_changes: list[dict[str, object]] = []

    class FakeSteamVrRunner:
        pass

    class FakeDesktopRunner:
        pass

    monkeypatch.setattr(
        overlay_application_module,
        "DefaultOverlayProcessRunner",
        FakeSteamVrRunner,
        raising=False,
    )
    monkeypatch.setattr(
        overlay_application_module,
        "DesktopFletOverlayRunner",
        FakeDesktopRunner,
        raising=False,
    )

    controller = _make_controller(
        app=SimpleNamespace(
            on_desktop_overlay_state_changed=lambda **state: state_changes.append(dict(state))
        )
    )
    controller.settings = AppSettings()
    controller.settings.overlay.target = "desktop"
    controller.settings.overlay.desktop_flet.size_preset = "small"
    controller.settings.overlay.desktop_flet.position.x = 24
    controller.settings.overlay.desktop_flet.position.y = 48
    controller.settings.overlay.desktop_flet.locked = True
    controller.settings.overlay.desktop_flet.visual.background_alpha = 0.44
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)

    manager = FakeOverlayProcessManager.instances[0]
    bridge = FakeOverlayBridge.instances[0]
    assert isinstance(manager.process_runner, FakeDesktopRunner)
    assert not isinstance(manager.process_runner, FakeSteamVrRunner)
    assert "overlay_target" not in manager.extra_kwargs
    assert bridge.desktop_runtime_controls_enabled is True
    assert bridge.initial_desktop_runtime_controls == [
        {
            "command": "apply_window_bounds",
            "x": 24,
            "y": 48,
            "width": 1152,
            "height": 288,
        },
        {
            "command": "apply_visual_config",
            "text_scale": 1.0,
            "background_alpha": 0.44,
            "outline_width": None,
        },
        {"command": "set_interaction_mode", "mode": "edit"},
    ]
    assert controller.desktop_overlay_captions_locked is False
    assert controller._get_desktop_overlay_application_owner().interaction_mode == "edit"
    assert state_changes == []

    manager.complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")
    await controller.set_overlay_enabled(False)


@pytest.mark.asyncio
async def test_desktop_move_persistence_debounces_position_only_and_ignores_programmatic_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    saved_desktop: list[tuple[object, object, str]] = []

    def record_saved_settings(_path, settings) -> None:
        desktop = settings.overlay.desktop_flet
        saved_desktop.append((desktop.position.x, desktop.position.y, desktop.size_preset))

    _patch_settings_save(monkeypatch, record_saved_settings)

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.overlay.target = "desktop"
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)
    manager = FakeOverlayProcessManager.instances[0]
    manager.complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")

    renderer_events = manager.renderer_events
    assert isinstance(renderer_events, asyncio.Queue)

    await renderer_events.put(
        {
            "type": "overlay_event",
            "payload": {
                "event": "window_bounds_changed",
                "source": "programmatic",
                "persist": False,
                "x": 24,
                "y": 48,
                "width": 960,
                "height": 240,
            },
        }
    )
    await renderer_events.put(
        {
            "type": "overlay_event",
            "payload": {
                "event": "window_bounds_changed",
                "source": "user",
                "persist": True,
                "x": True,
                "y": 48,
                "width": 960,
                "height": 240,
            },
        }
    )
    await asyncio.sleep(0.05)
    assert saved_desktop == []

    await renderer_events.put(
        {
            "type": "overlay_event",
            "payload": {
                "event": "window_bounds_changed",
                "source": "user",
                "persist": True,
                "x": 111,
                "y": 222,
                "width": 1792,
                "height": 448,
            },
        }
    )
    await renderer_events.put(
        {
            "type": "overlay_event",
            "payload": {
                "event": "window_bounds_changed",
                "source": "user",
                "persist": True,
                "x": 333,
                "y": 444,
                "width": 1152,
                "height": 288,
            },
        }
    )

    await _wait_until(
        lambda: len(saved_desktop) == 1
        and controller.settings.overlay.desktop_flet.position.x == 333,
        attempts=20,
        delay_s=0.02,
    )

    assert saved_desktop == [(333, 444, "medium")]
    assert controller.settings.overlay.desktop_flet.position.x == 333
    assert controller.settings.overlay.desktop_flet.position.y == 444
    assert controller.settings.overlay.desktop_flet.size_preset == "medium"

    await controller.set_overlay_enabled(False)


@pytest.mark.asyncio
async def test_desktop_bounds_debounce_routes_position_through_order23_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.overlay.target = "desktop"
    _overlay_owner(controller).active_target = OVERLAY_TARGET_DESKTOP
    service = RecordingSettingsMutationService()
    controller.settings_mutation_service = service

    def fail_direct_save(*_args, **_kwargs) -> None:
        raise AssertionError("desktop bounds must not use direct settings save")

    _patch_settings_save(monkeypatch, fail_direct_save)
    monkeypatch.setattr(
        desktop_overlay_application_module,
        "DESKTOP_BOUNDS_PERSIST_DEBOUNCE_S",
        0,
    )
    bounds_owner = controller._get_desktop_overlay_application_owner().bounds_owner
    bounds_owner.replace_pending_bounds(
        {
            "x": 321,
            "y": 654,
            "width": 1152,
            "height": 288,
        }
    )

    await bounds_owner.persist_after_debounce()

    assert len(service.requests) == 1
    request = service.requests[0]
    assert request.reason == settings_mutation.SETTINGS_MUTATION_SURFACE_OVERLAY_OSC_OUTPUT
    assert request.values == {
        "overlay.desktop_flet.position.x": 321,
        "overlay.desktop_flet.position.y": 654,
    }
    assert controller.settings.overlay.desktop_flet.position.x == 321
    assert controller.settings.overlay.desktop_flet.position.y == 654


@pytest.mark.asyncio
async def test_desktop_locked_mode_user_bounds_events_do_not_persist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    saved_desktop: list[tuple[object, object, str]] = []

    def record_saved_settings(_path, settings) -> None:
        desktop = settings.overlay.desktop_flet
        saved_desktop.append((desktop.position.x, desktop.position.y, desktop.size_preset))

    _patch_settings_save(monkeypatch, record_saved_settings)

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.overlay.target = "desktop"
    controller.settings.overlay.desktop_flet.locked = True
    controller.settings.overlay.desktop_flet.position.x = 320
    controller.settings.overlay.desktop_flet.position.y = 720
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)
    manager = FakeOverlayProcessManager.instances[0]
    manager.complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")
    await controller.set_desktop_overlay_captions_locked(True)
    assert controller.desktop_overlay_captions_locked is True
    assert saved_desktop == []

    renderer_events = manager.renderer_events
    assert isinstance(renderer_events, asyncio.Queue)
    await renderer_events.put(
        {
            "type": "overlay_event",
            "payload": {
                "event": "window_bounds_changed",
                "source": "user",
                "persist": True,
                "x": 608,
                "y": 1117,
                "width": 1344,
                "height": 336,
            },
        }
    )
    await asyncio.sleep(0.10)

    assert saved_desktop == []
    assert controller.settings.overlay.desktop_flet.position.x == 320
    assert controller.settings.overlay.desktop_flet.position.y == 720

    await controller.set_overlay_enabled(False)


@pytest.mark.asyncio
async def test_desktop_lock_toggle_is_runtime_only_and_does_not_save_or_mutate_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    saved_desktop: list[tuple[object, object, str, bool]] = []
    state_changes: list[dict[str, object]] = []

    def fake_save_settings(self: SettingsOwner, **_kwargs: object) -> bool:
        assert self.current is not None
        desktop = self.current.overlay.desktop_flet
        saved_desktop.append(
            (desktop.position.x, desktop.position.y, desktop.size_preset, desktop.locked)
        )
        return True

    monkeypatch.setattr(SettingsOwner, "save_current", fake_save_settings)

    controller = _make_controller(
        app=SimpleNamespace(
            on_desktop_overlay_state_changed=lambda **state: state_changes.append(dict(state))
        )
    )
    controller.settings = AppSettings()
    controller.settings.overlay.target = "desktop"
    controller.settings.overlay.desktop_flet.locked = False
    controller.settings.overlay.desktop_flet.position.x = 320
    controller.settings.overlay.desktop_flet.position.y = 720
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)
    manager = FakeOverlayProcessManager.instances[0]
    bridge = FakeOverlayBridge.instances[0]
    manager.complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")
    saved_desktop.clear()

    await controller.set_desktop_overlay_captions_locked(True)

    assert controller.desktop_overlay_captions_locked is True
    assert controller.settings.overlay.desktop_flet.locked is False
    assert saved_desktop == []
    assert bridge.desktop_runtime_control_payloads[-1] == {
        "command": "set_interaction_mode",
        "mode": "pass_through",
    }
    assert state_changes[-1] == {
        "interaction_mode": "pass_through",
        "captions_locked": True,
    }

    await controller.set_desktop_overlay_captions_locked(False)

    assert controller.desktop_overlay_captions_locked is False
    assert controller.settings.overlay.desktop_flet.locked is False
    assert saved_desktop == []
    assert bridge.desktop_runtime_control_payloads[-1] == {
        "command": "set_interaction_mode",
        "mode": "edit",
    }

    await controller.set_overlay_enabled(False)


@pytest.mark.asyncio
async def test_desktop_size_preset_change_preserves_current_center_without_clamping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    saved_desktop: list[tuple[object, object, str]] = []

    def record_saved_settings(_path, settings) -> None:
        desktop = settings.overlay.desktop_flet
        saved_desktop.append((desktop.position.x, desktop.position.y, desktop.size_preset))

    _patch_settings_save(monkeypatch, record_saved_settings)
    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        lambda self, *, enabled: asyncio.sleep(0),
    )
    monkeypatch.setattr(
        WindowsDesktopWorkAreaAdapter,
        "primary_work_area",
        lambda self: (0, 0, 800, 600),
        raising=False,
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.overlay.target = "desktop"
    controller.settings.overlay.desktop_flet.size_preset = "small"
    controller.settings.overlay.desktop_flet.position.x = -100
    controller.settings.overlay.desktop_flet.position.y = 20
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)
    manager = FakeOverlayProcessManager.instances[0]
    bridge = FakeOverlayBridge.instances[0]
    manager.complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")
    saved_desktop.clear()

    await controller.set_desktop_overlay_size_preset("xlarge")

    expected_x = -420
    expected_y = -60
    assert saved_desktop[-1] == (pytest.approx(expected_x), pytest.approx(expected_y), "xlarge")
    assert controller.settings.overlay.desktop_flet.position.x == pytest.approx(expected_x)
    assert controller.settings.overlay.desktop_flet.position.y == pytest.approx(expected_y)
    assert bridge.desktop_runtime_control_payloads[-1] == {
        "command": "apply_window_bounds",
        "x": pytest.approx(expected_x),
        "y": pytest.approx(expected_y),
        "width": 1792,
        "height": 448,
    }

    await controller.set_desktop_overlay_size_preset("tiny")

    tiny_expected_x = 156
    tiny_expected_y = 84
    assert saved_desktop[-1] == (
        pytest.approx(tiny_expected_x),
        pytest.approx(tiny_expected_y),
        "tiny",
    )
    assert controller.settings.overlay.desktop_flet.position.x == pytest.approx(tiny_expected_x)
    assert controller.settings.overlay.desktop_flet.position.y == pytest.approx(tiny_expected_y)
    assert bridge.desktop_runtime_control_payloads[-1] == {
        "command": "apply_window_bounds",
        "x": pytest.approx(tiny_expected_x),
        "y": pytest.approx(tiny_expected_y),
        "width": 640,
        "height": 160,
    }

    await controller.set_overlay_enabled(False)


@pytest.mark.asyncio
async def test_desktop_size_preset_change_drains_queued_pre_resize_user_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    saved_desktop: list[tuple[object, object, str]] = []

    def record_saved_settings(_path, settings) -> None:
        desktop = settings.overlay.desktop_flet
        saved_desktop.append((desktop.position.x, desktop.position.y, desktop.size_preset))

    _patch_settings_save(monkeypatch, record_saved_settings)
    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        lambda self, *, enabled: asyncio.sleep(0),
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.overlay.target = "desktop"
    controller.settings.overlay.desktop_flet.size_preset = "small"
    controller.settings.overlay.desktop_flet.position.x = 300
    controller.settings.overlay.desktop_flet.position.y = 400
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)
    manager = FakeOverlayProcessManager.instances[0]
    manager.complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")

    runtime = _overlay_runtime(controller)
    event_task = runtime.renderer_event_task
    assert event_task is not None
    event_task.cancel()
    await asyncio.gather(event_task, return_exceptions=True)
    runtime._renderer_event_task = None  # noqa: SLF001 - freeze owner queue

    renderer_events = manager.renderer_events
    assert isinstance(renderer_events, asyncio.Queue)
    await renderer_events.put(
        {
            "type": "overlay_event",
            "payload": {
                "event": "window_bounds_changed",
                "source": "user",
                "persist": True,
                "x": 300,
                "y": 400,
                "width": 1152,
                "height": 288,
            },
        }
    )

    await controller.set_desktop_overlay_size_preset("xlarge")

    assert renderer_events.empty()
    assert saved_desktop == [(-20, 320, "xlarge")]

    await controller.set_overlay_enabled(False)


@pytest.mark.asyncio
async def test_desktop_size_preset_change_supersedes_pending_user_position_debounce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    saved_desktop: list[tuple[object, object, str]] = []

    def record_saved_settings(_path, settings) -> None:
        desktop = settings.overlay.desktop_flet
        saved_desktop.append((desktop.position.x, desktop.position.y, desktop.size_preset))

    _patch_settings_save(monkeypatch, record_saved_settings)
    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        lambda self, *, enabled: asyncio.sleep(0),
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.overlay.target = "desktop"
    controller.settings.overlay.desktop_flet.size_preset = "small"
    controller.settings.overlay.desktop_flet.position.x = -100
    controller.settings.overlay.desktop_flet.position.y = 20
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)
    manager = FakeOverlayProcessManager.instances[0]
    manager.complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")
    saved_desktop.clear()

    renderer_events = manager.renderer_events
    assert isinstance(renderer_events, asyncio.Queue)
    await renderer_events.put(
        {
            "type": "overlay_event",
            "payload": {
                "event": "window_bounds_changed",
                "source": "user",
                "persist": True,
                "x": 111,
                "y": 222,
                "width": 1152,
                "height": 288,
            },
        }
    )
    bounds_owner = controller._get_desktop_overlay_application_owner().bounds_owner
    await _wait_until(
        lambda: bounds_owner.persist_task is not None and bounds_owner.pending_bounds is not None
    )

    updated = copy.deepcopy(controller.settings)
    updated.overlay.desktop_flet.size_preset = "xlarge"

    await controller.apply_settings(updated)
    await asyncio.sleep(desktop_overlay_application_module.DESKTOP_BOUNDS_PERSIST_DEBOUNCE_S * 2)

    expected_x = -420
    expected_y = -60
    assert saved_desktop == [(pytest.approx(expected_x), pytest.approx(expected_y), "xlarge")]
    assert controller.settings.overlay.desktop_flet.position.x == pytest.approx(expected_x)
    assert controller.settings.overlay.desktop_flet.position.y == pytest.approx(expected_y)
    assert controller.settings.overlay.desktop_flet.size_preset == "xlarge"

    await controller.set_overlay_enabled(False)


@pytest.mark.asyncio
async def test_desktop_size_preset_change_cancels_pending_bounds_before_order23_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        lambda self, *, enabled: asyncio.sleep(0),
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.overlay.target = "desktop"
    controller.settings.overlay.desktop_flet.size_preset = "small"
    controller.settings.overlay.desktop_flet.position.x = -100
    controller.settings.overlay.desktop_flet.position.y = 20
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)
    manager = FakeOverlayProcessManager.instances[0]
    manager.complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")

    renderer_events = manager.renderer_events
    assert isinstance(renderer_events, asyncio.Queue)
    await renderer_events.put(
        {
            "type": "overlay_event",
            "payload": {
                "event": "window_bounds_changed",
                "source": "user",
                "persist": True,
                "x": 111,
                "y": 222,
                "width": 1152,
                "height": 288,
            },
        }
    )
    bounds_owner = controller._get_desktop_overlay_application_owner().bounds_owner
    await _wait_until(
        lambda: bounds_owner.persist_task is not None and bounds_owner.pending_bounds is not None
    )

    class InspectingOrder23Service(RecordingSettingsMutationService):
        async def mutate(
            self,
            request: settings_mutation.SettingsMutationRequest,
        ) -> messages.TransactionResult:
            if request.reason == settings_mutation.SETTINGS_MUTATION_SURFACE_OVERLAY_OSC_OUTPUT:
                task = bounds_owner.persist_task
                assert task is None or task.cancelled() or task.done()
                assert bounds_owner.pending_bounds is None
            return await super().mutate(request)

    controller.settings_mutation_service = InspectingOrder23Service()

    await controller.set_desktop_overlay_size_preset("xlarge")
    await asyncio.sleep(desktop_overlay_application_module.DESKTOP_BOUNDS_PERSIST_DEBOUNCE_S * 2)

    request = controller.settings_mutation_service.requests[0]
    assert request.values["overlay.desktop_flet.size_preset"] == "xlarge"
    assert request.values["overlay.desktop_flet.position.x"] == pytest.approx(-420)
    assert request.values["overlay.desktop_flet.position.y"] == pytest.approx(-60)

    await controller.set_overlay_enabled(False)


@pytest.mark.asyncio
async def test_desktop_reset_clears_position_unlocks_preserves_size_and_alpha_and_centers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    saved_desktop: list[tuple[object, object, str, bool, float]] = []
    state_changes: list[dict[str, object]] = []

    def record_saved_settings(_path, settings) -> None:
        desktop = settings.overlay.desktop_flet
        saved_desktop.append(
            (
                desktop.position.x,
                desktop.position.y,
                desktop.size_preset,
                desktop.locked,
                desktop.visual.background_alpha,
            )
        )

    _patch_settings_save(monkeypatch, record_saved_settings)
    monkeypatch.setattr(
        WindowsDesktopWorkAreaAdapter,
        "primary_work_area",
        lambda self: (0, 0, 1920, 1080),
        raising=False,
    )

    controller = _make_controller(
        app=SimpleNamespace(
            on_desktop_overlay_state_changed=lambda **state: state_changes.append(dict(state))
        )
    )
    controller.settings = AppSettings()
    controller.settings.overlay.target = "desktop"
    controller.settings.overlay.desktop_flet.size_preset = "large"
    controller.settings.overlay.desktop_flet.position.x = 80
    controller.settings.overlay.desktop_flet.position.y = 90
    controller.settings.overlay.desktop_flet.locked = True
    controller.settings.overlay.desktop_flet.visual.background_alpha = 0.44
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)
    manager = FakeOverlayProcessManager.instances[0]
    bridge = FakeOverlayBridge.instances[0]
    manager.complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")
    await controller.set_desktop_overlay_captions_locked(True)
    saved_desktop.clear()
    bridge.desktop_runtime_control_payloads.clear()

    await controller.reset_desktop_overlay_position()

    await _wait_until(lambda: len(saved_desktop) == 1, attempts=20, delay_s=0.02)

    assert saved_desktop == [(None, None, "large", False, 0.44)]
    assert controller.settings.overlay.desktop_flet.position.x is None
    assert controller.settings.overlay.desktop_flet.position.y is None
    assert controller.settings.overlay.desktop_flet.size_preset == "large"
    assert controller.settings.overlay.desktop_flet.locked is False
    assert controller.settings.overlay.desktop_flet.visual.background_alpha == 0.44
    assert controller.desktop_overlay_captions_locked is False
    assert state_changes[-1] == {"interaction_mode": "edit", "captions_locked": False}
    assert {"command": "set_interaction_mode", "mode": "edit"} in (
        bridge.desktop_runtime_control_payloads
    )
    assert bridge.desktop_runtime_control_payloads[-1] == {
        "command": "apply_window_bounds",
        "x": pytest.approx(160),
        "y": pytest.approx(340),
        "width": 1600,
        "height": 400,
    }

    await controller.set_overlay_enabled(False)


@pytest.mark.asyncio
async def test_desktop_reset_persists_configured_desktop_target_without_running_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved_desktop: list[tuple[object, object, str, bool, float]] = []
    runtime_payloads: list[dict[str, object]] = []
    bounds_payloads: list[dict[str, int | float]] = []

    def record_saved_settings(_path, settings) -> None:
        desktop = settings.overlay.desktop_flet
        saved_desktop.append(
            (
                desktop.position.x,
                desktop.position.y,
                desktop.size_preset,
                desktop.locked,
                desktop.visual.background_alpha,
            )
        )

    async def fake_broadcast_runtime_control(
        self: DesktopOverlayApplicationOwner,
        payload: dict[str, object],
    ) -> bool:
        _ = self
        runtime_payloads.append(dict(payload))
        return True

    async def fake_broadcast_window_bounds_control(
        self: DesktopOverlayApplicationOwner,
        bounds: dict[str, int | float],
    ) -> None:
        _ = self
        bounds_payloads.append(dict(bounds))

    _patch_settings_save(monkeypatch, record_saved_settings)
    monkeypatch.setattr(
        DesktopOverlayApplicationOwner,
        "broadcast",
        fake_broadcast_runtime_control,
    )
    monkeypatch.setattr(
        DesktopOverlayApplicationOwner,
        "broadcast_bounds",
        fake_broadcast_window_bounds_control,
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.overlay.target = "desktop"
    controller.settings.overlay.desktop_flet.size_preset = "large"
    controller.settings.overlay.desktop_flet.position.x = 80
    controller.settings.overlay.desktop_flet.position.y = 90
    controller.settings.overlay.desktop_flet.locked = True
    controller.settings.overlay.desktop_flet.visual.background_alpha = 0.44

    await controller.reset_desktop_overlay_position()

    assert saved_desktop == [(None, None, "large", False, 0.44)]
    assert controller.settings.overlay.desktop_flet.position.x is None
    assert controller.settings.overlay.desktop_flet.position.y is None
    assert controller.settings.overlay.desktop_flet.size_preset == "large"
    assert controller.settings.overlay.desktop_flet.locked is False
    assert controller.settings.overlay.desktop_flet.visual.background_alpha == 0.44
    assert runtime_payloads == []
    assert bounds_payloads == []


@pytest.mark.asyncio
async def test_desktop_reset_routes_position_clear_through_order23_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.overlay.target = "desktop"
    controller.settings.overlay.desktop_flet.size_preset = "large"
    controller.settings.overlay.desktop_flet.position.x = 80
    controller.settings.overlay.desktop_flet.position.y = 90
    controller.settings.overlay.desktop_flet.locked = True
    controller._get_desktop_overlay_application_owner().set_interaction_mode("pass_through")
    service = RecordingSettingsMutationService()
    controller.settings_mutation_service = service

    def fail_direct_save(*_args, **_kwargs) -> None:
        raise AssertionError("desktop reset must not use direct settings save")

    _patch_settings_save(monkeypatch, fail_direct_save)

    await controller.reset_desktop_overlay_position()

    assert len(service.requests) == 1
    request = service.requests[0]
    assert request.reason == settings_mutation.SETTINGS_MUTATION_SURFACE_OVERLAY_OSC_OUTPUT
    assert request.values == {
        "overlay.desktop_flet.position.x": None,
        "overlay.desktop_flet.position.y": None,
    }
    assert "overlay.desktop_flet.locked" not in request.values
    assert controller.settings.overlay.desktop_flet.position.x is None
    assert controller.settings.overlay.desktop_flet.position.y is None
    assert controller.settings.overlay.desktop_flet.size_preset == "large"
    assert controller.settings.overlay.desktop_flet.locked is False
    assert controller.desktop_overlay_captions_locked is False


@pytest.mark.asyncio
async def test_desktop_reset_keeps_runtime_state_when_order23_commit_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_payloads: list[dict[str, object]] = []
    bounds_payloads: list[dict[str, int | float]] = []

    async def fake_broadcast_runtime_control(
        self: DesktopOverlayApplicationOwner,
        payload: dict[str, object],
    ) -> bool:
        _ = self
        runtime_payloads.append(dict(payload))
        return True

    async def fake_broadcast_window_bounds_control(
        self: DesktopOverlayApplicationOwner,
        bounds: dict[str, int | float],
    ) -> None:
        _ = self
        bounds_payloads.append(dict(bounds))

    monkeypatch.setattr(
        DesktopOverlayApplicationOwner,
        "broadcast",
        fake_broadcast_runtime_control,
    )
    monkeypatch.setattr(
        DesktopOverlayApplicationOwner,
        "broadcast_bounds",
        fake_broadcast_window_bounds_control,
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.overlay.target = "desktop"
    controller.settings.overlay.desktop_flet.size_preset = "large"
    controller.settings.overlay.desktop_flet.position.x = 80
    controller.settings.overlay.desktop_flet.position.y = 90
    controller.settings.overlay.desktop_flet.locked = True
    _overlay_owner(controller).active_target = OVERLAY_TARGET_DESKTOP
    _attach_overlay_bridge(controller, object())
    controller._get_desktop_overlay_application_owner().set_interaction_mode("pass_through")
    failed_result = messages.TransactionResult(
        status=messages.TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED,
        message=None,
        diagnostics=None,
    )
    service = RecordingSettingsMutationService(failed_result)
    controller.settings_mutation_service = service

    await controller.reset_desktop_overlay_position()

    assert len(service.requests) == 1
    request = service.requests[0]
    assert request.reason == settings_mutation.SETTINGS_MUTATION_SURFACE_OVERLAY_OSC_OUTPUT
    assert request.values == {
        "overlay.desktop_flet.position.x": None,
        "overlay.desktop_flet.position.y": None,
    }
    assert controller.settings.overlay.desktop_flet.position.x == 80
    assert controller.settings.overlay.desktop_flet.position.y == 90
    assert controller.settings.overlay.desktop_flet.locked is True
    assert controller._get_desktop_overlay_application_owner().interaction_mode == "pass_through"
    assert controller.desktop_overlay_captions_locked is True
    assert runtime_payloads == []
    assert bounds_payloads == []


@pytest.mark.asyncio
async def test_desktop_reset_persistence_cancels_pending_user_position_debounce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    saved_desktop: list[tuple[object, object, str, bool]] = []

    def record_saved_settings(_path, settings) -> None:
        desktop = settings.overlay.desktop_flet
        saved_desktop.append(
            (desktop.position.x, desktop.position.y, desktop.size_preset, desktop.locked)
        )

    _patch_settings_save(monkeypatch, record_saved_settings)
    monkeypatch.setattr(
        WindowsDesktopWorkAreaAdapter,
        "primary_work_area",
        lambda self: (0, 0, 1920, 1080),
        raising=False,
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.overlay.target = "desktop"
    controller.settings.overlay.desktop_flet.size_preset = "medium"
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)
    manager = FakeOverlayProcessManager.instances[0]
    manager.complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")

    renderer_events = manager.renderer_events
    assert isinstance(renderer_events, asyncio.Queue)
    await renderer_events.put(
        {
            "type": "overlay_event",
            "payload": {
                "event": "window_bounds_changed",
                "source": "user",
                "persist": True,
                "x": 111,
                "y": 222,
                "width": 800,
                "height": 220,
            },
        }
    )
    await renderer_events.put(
        {
            "type": "overlay_event",
            "payload": {"event": "reset_to_bottom_center_requested"},
        }
    )

    bounds_owner = controller._get_desktop_overlay_application_owner().bounds_owner
    await _wait_until(
        lambda: bounds_owner.persist_task is None and bounds_owner.pending_bounds is None,
        attempts=20,
        delay_s=0.02,
    )
    await asyncio.sleep(desktop_overlay_application_module.DESKTOP_BOUNDS_PERSIST_DEBOUNCE_S * 2)

    assert saved_desktop == []
    assert controller.settings.overlay.desktop_flet.position.x is None
    assert controller.settings.overlay.desktop_flet.position.y is None

    await controller.set_overlay_enabled(False)


@pytest.mark.asyncio
async def test_desktop_reset_drains_queued_pre_reset_user_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    saved_desktop: list[tuple[object, object, str, bool]] = []

    def record_saved_settings(_path, settings) -> None:
        desktop = settings.overlay.desktop_flet
        saved_desktop.append(
            (desktop.position.x, desktop.position.y, desktop.size_preset, desktop.locked)
        )

    _patch_settings_save(monkeypatch, record_saved_settings)
    monkeypatch.setattr(
        WindowsDesktopWorkAreaAdapter,
        "primary_work_area",
        lambda self: (0, 0, 1920, 1080),
        raising=False,
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.overlay.target = "desktop"
    controller.settings.overlay.desktop_flet.size_preset = "large"
    controller.settings.overlay.desktop_flet.position.x = 300
    controller.settings.overlay.desktop_flet.position.y = 400
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)
    manager = FakeOverlayProcessManager.instances[0]
    manager.complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")

    runtime = _overlay_runtime(controller)
    event_task = runtime.renderer_event_task
    assert event_task is not None
    event_task.cancel()
    await asyncio.gather(event_task, return_exceptions=True)
    runtime._renderer_event_task = None  # noqa: SLF001 - freeze owner queue

    renderer_events = manager.renderer_events
    assert isinstance(renderer_events, asyncio.Queue)
    await renderer_events.put(
        {
            "type": "overlay_event",
            "payload": {
                "event": "window_bounds_changed",
                "source": "user",
                "persist": True,
                "x": 300,
                "y": 400,
                "width": 1600,
                "height": 400,
            },
        }
    )

    await controller.reset_desktop_overlay_position()

    assert renderer_events.empty()
    assert saved_desktop == [(None, None, "large", False)]

    await controller.set_overlay_enabled(False)


@pytest.mark.asyncio
async def test_desktop_source_reset_ignores_event_size_and_cancels_pending_user_position_debounce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    saved_desktop: list[tuple[object, object, str, bool]] = []

    def record_saved_settings(_path, settings) -> None:
        desktop = settings.overlay.desktop_flet
        saved_desktop.append(
            (desktop.position.x, desktop.position.y, desktop.size_preset, desktop.locked)
        )

    _patch_settings_save(monkeypatch, record_saved_settings)
    monkeypatch.setattr(
        WindowsDesktopWorkAreaAdapter,
        "primary_work_area",
        lambda self: (0, 0, 1920, 1080),
        raising=False,
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.overlay.target = "desktop"
    controller.settings.overlay.desktop_flet.size_preset = "small"
    controller.settings.overlay.desktop_flet.locked = True
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)
    manager = FakeOverlayProcessManager.instances[0]
    manager.complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")

    renderer_events = manager.renderer_events
    assert isinstance(renderer_events, asyncio.Queue)
    await renderer_events.put(
        {
            "type": "overlay_event",
            "payload": {
                "event": "window_bounds_changed",
                "source": "user",
                "persist": True,
                "x": 111,
                "y": 222,
                "width": 800,
                "height": 220,
            },
        }
    )
    await renderer_events.put(
        {
            "type": "overlay_event",
            "payload": {
                "event": "window_bounds_changed",
                "source": "reset",
                "persist": True,
                "x": 320,
                "y": 750,
                "width": 960,
                "height": 240,
            },
        }
    )

    bounds_owner = controller._get_desktop_overlay_application_owner().bounds_owner
    await _wait_until(
        lambda: bounds_owner.persist_task is None and bounds_owner.pending_bounds is None,
        attempts=20,
        delay_s=0.02,
    )
    await asyncio.sleep(desktop_overlay_application_module.DESKTOP_BOUNDS_PERSIST_DEBOUNCE_S * 2)

    assert saved_desktop == []
    assert controller.settings.overlay.desktop_flet.position.x is None
    assert controller.settings.overlay.desktop_flet.position.y is None
    assert controller.settings.overlay.desktop_flet.size_preset == "small"

    await controller.set_overlay_enabled(False)


def test_vr_overlay_calibration_reset_does_not_mutate_desktop_overlay_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePage:
        def __init__(self) -> None:
            self.tasks: list[object] = []

        def run_task(self, coro_fn) -> None:
            self.tasks.append(coro_fn)

    def fail_direct_save(*_args, **_kwargs) -> None:
        raise AssertionError("overlay calibration must not use direct settings save")

    _patch_settings_save(monkeypatch, fail_direct_save)

    page = FakePage()
    controller = GuiController(
        page=page,
        app=_presentation(SimpleNamespace(), page=page),
        config_path=Path("settings.json"),
    )
    controller.settings = AppSettings()
    service = RecordingSettingsMutationService()
    controller.settings_mutation_service = service
    controller.settings.overlay.desktop_flet.size_preset = "xlarge"
    controller.settings.overlay.desktop_flet.position.x = 123
    controller.settings.overlay.desktop_flet.position.y = 456
    controller.settings.overlay.desktop_flet.locked = True
    controller.settings.overlay.desktop_flet.visual.background_alpha = 0.33
    controller._get_overlay_calibration_application_owner().replace_current(
        OverlayCalibration(distance=3.0, offset_x=2.0)
    )
    controller.settings.overlay.calibration = controller.overlay_calibration.copy()
    controller._get_overlay_calibration_application_owner().replace_draft(OverlayCalibration())

    controller.apply_overlay_calibration()

    assert len(page.tasks) == 1

    asyncio.run(page.tasks[0]())

    assert len(service.requests) == 1
    assert (
        service.requests[0].reason == settings_mutation.SETTINGS_MUTATION_SURFACE_OVERLAY_OSC_OUTPUT
    )
    assert service.requests[0].values == {
        "overlay.calibration.offset_x": 0.0,
        "overlay.calibration.distance": 1.1,
    }
    assert controller.settings.overlay.desktop_flet.position.x == 123
    assert controller.settings.overlay.desktop_flet.position.y == 456
    assert controller.settings.overlay.desktop_flet.size_preset == "xlarge"
    assert controller.settings.overlay.desktop_flet.locked is True
    assert controller.settings.overlay.desktop_flet.visual.background_alpha == 0.33


def test_resolved_desktop_initial_controls_emit_launch_diagnostics_only_in_detailed_mode() -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.overlay.target = "desktop"
    controller.settings.overlay.desktop_flet.size_preset = "medium"
    controller.settings.overlay.desktop_flet.position.x = 597
    controller.settings.overlay.desktop_flet.position.y = 1017
    controller.settings.overlay.desktop_flet.locked = True
    controller.settings.overlay.desktop_flet.visual.background_alpha = 0.5
    controller._runtime_logging = RuntimeLoggingSpy(detailed_enabled=True)

    controls = controller._get_desktop_overlay_application_owner().initial_controls(
        resolve_overlay_config(controller.settings)
    )

    assert controls[-1] == {"command": "set_interaction_mode", "mode": "edit"}
    assert "bounds_epoch" not in controls[0]
    messages = [message for _level, message in controller._runtime_logging.detailed_messages]
    assert any(
        message.startswith("[DesktopOverlay][Launch]")
        and "target=desktop" in message
        and "locked=True" in message
        and "interaction_mode=edit" in message
        and "size_preset=medium" in message
        and "x=597" in message
        and "y=1017" in message
        and "width=1344" in message
        and "height=336" in message
        and "background_alpha=0.5" in message
        for message in messages
    )

    basic_controller = _make_controller(app=SimpleNamespace())
    basic_controller.settings = copy.deepcopy(controller.settings)
    basic_controller._runtime_logging = RuntimeLoggingSpy(detailed_enabled=False)

    basic_controller._get_desktop_overlay_application_owner().initial_controls(
        resolve_overlay_config(basic_controller.settings)
    )

    assert basic_controller._runtime_logging.detailed_messages == []


def test_desktop_initial_controls_can_be_built_from_resolved_overlay_config() -> None:
    from puripuly_heart.config.resolved import ResolvedOverlayConfig  # noqa: PLC0415

    owner = DesktopOverlayApplicationOwner(
        settings=SimpleNamespace(current=None),
        settings_application_provider=lambda: SimpleNamespace(),
        overlay_provider=lambda: SimpleNamespace(),
        work_area=SimpleNamespace(primary_work_area=lambda: (0, 0, 1920, 1080)),
        policy=create_desktop_overlay_policy(),
        presentation_sink=lambda _mode, _locked: None,
        log_detailed=lambda _message, _level, _exception: None,
    )
    resolved = ResolvedOverlayConfig(
        enabled=True,
        target="desktop",
        show_translation=False,
        show_peer_original=True,
        calibration={"distance": 2.0},
        desktop_overlay_options={
            "size_preset": "medium",
            "position": {"x": 597, "y": 1017},
            "locked": True,
            "visual": {
                "text_scale": 1.0,
                "background_alpha": 0.5,
                "outline_width": None,
            },
        },
    )

    controls = owner.initial_controls(resolved)

    assert controls == [
        {
            "command": "apply_window_bounds",
            "x": 597,
            "y": 1017,
            "width": 1344,
            "height": 336,
        },
        {
            "command": "apply_visual_config",
            "text_scale": 1.0,
            "background_alpha": 0.5,
            "outline_width": None,
        },
        {"command": "set_interaction_mode", "mode": "edit"},
    ]
    assert "desktop_flet" not in resolved.desktop_overlay_options

    centered = owner.initial_controls(
        ResolvedOverlayConfig(
            enabled=True,
            target="desktop",
            show_translation=False,
            show_peer_original=True,
            calibration={},
            desktop_overlay_options={
                "size_preset": "medium",
                "position": {"x": None, "y": None},
                "locked": False,
                "visual": {},
            },
        )
    )
    assert centered[0] == {
        "command": "apply_window_bounds",
        "x": 288,
        "y": 372,
        "width": 1344,
        "height": 336,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overlay_target", "expected_refresh_burst"),
    [("desktop", "False"), ("steamvr", "True")],
)
async def test_overlay_start_logs_selected_target_refresh_flags_for_experiment_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    overlay_target: str,
    expected_refresh_burst: str,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.overlay.target = overlay_target
    controller.hub = DummyHub()
    controller._runtime_logging = RuntimeLoggingSpy(detailed_enabled=True)

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)

    messages = [message for _level, message in controller._runtime_logging.detailed_messages]
    assert any(
        message.startswith("[Overlay][Start]")
        and f"target={overlay_target}" in message
        and "overlay_instance_id=overlay-" in message
        and "logging_mode=detailed" in message
        and f"peer_presentation_refresh_burst={expected_refresh_burst}" in message
        and f"self_presentation_refresh_burst={expected_refresh_burst}" in message
        for message in messages
    )

    FakeOverlayProcessManager.instances[0].complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")
    await controller.set_overlay_enabled(False)


@pytest.mark.asyncio
async def test_desktop_bounds_events_emit_diagnostics_only_in_detailed_mode() -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.overlay.target = "desktop"
    controller.settings.overlay.desktop_flet.locked = False
    _overlay_owner(controller).active_target = "desktop"
    controller._runtime_logging = RuntimeLoggingSpy(detailed_enabled=True)
    payload: dict[object, object] = {
        "event": "window_bounds_changed",
        "source": "user",
        "persist": True,
        "x": 111,
        "y": 222,
        "width": 1152,
        "height": 288,
    }

    try:
        await controller._get_desktop_overlay_application_owner().handle_bounds_changed(payload)

        messages = [message for _level, message in controller._runtime_logging.detailed_messages]
        assert any(
            message.startswith("[DesktopOverlay][Bounds] received")
            and "source=user" in message
            and "persist=True" in message
            and "interaction_mode=edit" in message
            and "x=111" in message
            and "y=222" in message
            and "width=1152" in message
            and "height=288" in message
            for message in messages
        )
        assert any(
            message.startswith("[DesktopOverlay][Bounds] scheduled_persist") for message in messages
        )
    finally:
        await controller._get_desktop_overlay_application_owner().bounds_owner.cancel()

    basic_controller = _make_controller(app=SimpleNamespace())
    basic_controller.settings = AppSettings()
    basic_controller.settings.overlay.target = "desktop"
    _overlay_owner(basic_controller).active_target = "desktop"
    basic_controller._runtime_logging = RuntimeLoggingSpy(detailed_enabled=False)

    try:
        await basic_controller._get_desktop_overlay_application_owner().handle_bounds_changed(
            payload
        )
    finally:
        await basic_controller._get_desktop_overlay_application_owner().bounds_owner.cancel()

    assert basic_controller._runtime_logging.detailed_messages == []


@pytest.mark.asyncio
async def test_desktop_apply_settings_broadcasts_visual_config_for_background_alpha_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.ui.overlay_enabled = True
    controller.settings.overlay.target = "desktop"
    controller.settings.overlay.desktop_flet.visual.background_alpha = 0.5
    _overlay_owner(controller).active_target = "desktop"
    bridge = FakeOverlayBridge(session_token="desktop")
    _attach_overlay_bridge(controller, bridge)

    updated = copy.deepcopy(controller.settings)
    updated.overlay.desktop_flet.visual.background_alpha = 0.7

    await controller.apply_settings(updated)

    assert bridge.desktop_runtime_control_payloads == [
        {
            "command": "apply_visual_config",
            "text_scale": 1.0,
            "background_alpha": 0.7,
            "outline_width": None,
        }
    ]


@pytest.mark.asyncio
async def test_desktop_apply_settings_preserves_runtime_lock_without_persisting_saved_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serialized_desktop: list[dict[str, object]] = []

    def record_saved_settings(_path, settings) -> None:
        serialized_desktop.append(to_dict(settings)["overlay"]["desktop_flet"])

    _patch_settings_save(monkeypatch, record_saved_settings)
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.ui.overlay_enabled = True
    controller.settings.overlay.target = "desktop"
    controller.settings.overlay.desktop_flet.locked = False
    _overlay_owner(controller).active_target = "desktop"
    _overlay_owner(controller).state = "connected"
    bridge = FakeOverlayBridge(session_token="desktop")
    _attach_overlay_bridge(controller, bridge)
    await controller.set_desktop_overlay_captions_locked(True)
    serialized_desktop.clear()
    bridge.desktop_runtime_control_payloads.clear()

    updated = copy.deepcopy(controller.settings)
    updated.overlay.desktop_flet.visual.background_alpha = 0.7

    await controller.apply_settings(updated)

    assert controller.desktop_overlay_captions_locked is True
    assert controller.settings.overlay.desktop_flet.locked is False
    assert serialized_desktop == [
        {
            "size_preset": "medium",
            "position": {"x": None, "y": None},
            "visual": {"background_alpha": 0.7},
        }
    ]
    assert bridge.desktop_runtime_control_payloads == [
        {
            "command": "apply_visual_config",
            "text_scale": 1.0,
            "background_alpha": 0.7,
            "outline_width": None,
        }
    ]


@pytest.mark.asyncio
async def test_desktop_interaction_mode_controls_are_desktop_only_and_update_locked_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    save_calls: list[bool] = []

    def fake_save_settings(self: SettingsOwner, **_kwargs: object) -> bool:
        assert self.current is not None
        save_calls.append(self.current.overlay.desktop_flet.locked)
        return True

    monkeypatch.setattr(SettingsOwner, "save_current", fake_save_settings)

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.overlay.target = "desktop"
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)
    manager = FakeOverlayProcessManager.instances[0]
    bridge = FakeOverlayBridge.instances[0]
    manager.complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")

    renderer_events = manager.renderer_events
    assert isinstance(renderer_events, asyncio.Queue)
    await renderer_events.put(
        {
            "type": "overlay_event",
            "payload": {"event": "interaction_mode_changed", "mode": "pass_through"},
        }
    )
    await _wait_until(lambda: controller.desktop_overlay_captions_locked)

    await controller.set_desktop_overlay_captions_locked(False)

    assert controller.desktop_overlay_captions_locked is False
    assert controller.settings.overlay.desktop_flet.locked is False
    assert save_calls == []
    assert bridge.desktop_runtime_control_payloads[-1] == {
        "command": "set_interaction_mode",
        "mode": "edit",
    }

    steam_controller = _make_controller(app=SimpleNamespace())
    steam_controller.settings = AppSettings()
    steam_controller.settings.overlay.target = "steamvr"
    _overlay_owner(steam_controller).active_target = "steamvr"
    steam_bridge = FakeOverlayBridge(session_token="steamvr")
    _attach_overlay_bridge(steam_controller, steam_bridge)

    await steam_controller.set_desktop_overlay_captions_locked(True)

    assert steam_controller.desktop_overlay_captions_locked is False
    assert steam_bridge.desktop_runtime_control_payloads == []

    await controller.set_overlay_enabled(False)


@pytest.mark.asyncio
async def test_desktop_lock_request_is_ignored_without_active_desktop_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved_locked: list[bool] = []

    def fake_save_settings(self: SettingsOwner, **_kwargs: object) -> bool:
        assert self.current is not None
        saved_locked.append(self.current.overlay.desktop_flet.locked)
        return True

    monkeypatch.setattr(SettingsOwner, "save_current", fake_save_settings)

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.overlay.target = "desktop"

    await controller.set_desktop_overlay_captions_locked(True)

    assert controller.desktop_overlay_captions_locked is False
    assert controller._get_desktop_overlay_application_owner().interaction_mode == "edit"
    assert controller.settings.overlay.desktop_flet.locked is False
    assert saved_locked == []


@pytest.mark.asyncio
async def test_desktop_lock_request_is_ignored_until_desktop_renderer_connected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved_locked: list[bool] = []

    def fake_save_settings(self: SettingsOwner, **_kwargs: object) -> bool:
        assert self.current is not None
        saved_locked.append(self.current.overlay.desktop_flet.locked)
        return True

    monkeypatch.setattr(SettingsOwner, "save_current", fake_save_settings)

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.overlay.target = "desktop"
    _overlay_owner(controller).active_target = "desktop"
    _attach_overlay_bridge(controller, FakeOverlayBridge(session_token="desktop"))
    _overlay_owner(controller).state = "starting"

    await controller.set_desktop_overlay_captions_locked(True)

    assert controller.desktop_overlay_captions_locked is False
    assert controller._get_desktop_overlay_application_owner().interaction_mode == "edit"
    assert controller.settings.overlay.desktop_flet.locked is False
    assert _overlay_runtime(controller).bridge.desktop_runtime_control_payloads == []
    assert saved_locked == []


@pytest.mark.asyncio
async def test_overlay_target_routing_apply_settings_stops_before_switching_running_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        lambda self, *, enabled: asyncio.sleep(0),
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.overlay.target = "steamvr"
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)
    manager = FakeOverlayProcessManager.instances[0]
    manager.complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")

    updated = copy.deepcopy(controller.settings)
    updated.overlay.target = "desktop"
    updated.ui.overlay_enabled = True

    await controller.apply_settings(updated)

    assert controller.settings.overlay.target == "desktop"
    assert controller.settings.ui.overlay_enabled is False
    assert _overlay_owner(controller).snapshot.state == "off"
    assert manager.stop_calls == 1
    assert len(FakeOverlayProcessManager.instances) == 1


@pytest.mark.asyncio
async def test_overlay_target_routing_apply_settings_stops_after_in_place_target_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        lambda self, *, enabled: asyncio.sleep(0),
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.overlay.target = "steamvr"
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)
    manager = FakeOverlayProcessManager.instances[0]
    manager.complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")

    shared_settings = controller.settings
    shared_settings.overlay.target = "desktop"
    shared_settings.ui.overlay_enabled = True

    await controller.apply_settings(shared_settings)

    assert controller.settings.overlay.target == "desktop"
    assert controller.settings.ui.overlay_enabled is False
    assert _overlay_owner(controller).snapshot.state == "off"
    assert manager.stop_calls == 1
    assert len(FakeOverlayProcessManager.instances) == 1


@pytest.mark.asyncio
async def test_overlay_toggle_does_not_persist_transient_button_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_calls: list[str] = []
    controller = _make_controller(app=SimpleNamespace(refresh_overlay_peer_contract=lambda: None))
    controller.settings = AppSettings()
    controller.hub = DummyHub()

    async def fake_begin_overlay_start(self: GuiController) -> None:
        _ = self

    async def fake_shutdown_overlay_runtime(
        self: GuiController, *, preserve_failure_reason: bool
    ) -> None:
        _ = (self, preserve_failure_reason)

    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: save_calls.append("save") or True,
    )
    monkeypatch.setattr(OverlayApplicationOwner, "begin_start", fake_begin_overlay_start)
    monkeypatch.setattr(OverlayApplicationOwner, "shutdown", fake_shutdown_overlay_runtime)

    await controller.set_overlay_enabled(True)
    await controller.set_overlay_enabled(False)

    assert save_calls == []
    assert controller.settings.ui.overlay_enabled is False


@pytest.mark.asyncio
async def test_peer_translation_toggle_does_not_persist_transient_button_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_calls: list[str] = []
    controller = _make_controller(app=SimpleNamespace(refresh_overlay_peer_contract=lambda: None))
    controller.settings = AppSettings()
    controller.settings.ui.peer_translation_eula_accepted = True
    controller.hub = DummyHub(llm=object(), stt=object(), peer_stt=object())
    _overlay_owner(controller).state = "connected"

    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: save_calls.append("save") or True,
    )
    monkeypatch.setattr(
        GuiController,
        "_refresh_overlay_runtime_dependencies",
        lambda self: asyncio.sleep(0),
    )

    await controller.set_peer_translation_enabled(True)
    await controller.set_peer_translation_enabled(False)

    assert save_calls == []
    assert controller.settings.ui.peer_translation_enabled is False


@pytest.mark.asyncio
async def test_overlay_start_keeps_compatibility_refresh_until_vr_capability_is_confirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)

    assert _overlay_runtime(controller).presenter is not None
    assert _overlay_runtime(controller).presenter.peer_presentation_refresh_burst is True
    assert _overlay_runtime(controller).presenter.self_presentation_refresh_burst is True
    FakeOverlayProcessManager.instances[0].complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")
    await controller.set_overlay_enabled(False)


@pytest.mark.asyncio
async def test_desktop_overlay_start_disables_peer_presentation_refresh_for_new_presenter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.overlay.target = OVERLAY_TARGET_DESKTOP
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)

    assert _overlay_owner(controller).runtime is not None
    presenter = _overlay_owner(controller).runtime.presenter
    assert isinstance(presenter, OverlayPresenter)
    assert presenter.peer_presentation_refresh_burst is False
    assert presenter.self_presentation_refresh_burst is False
    FakeOverlayProcessManager.instances[0].complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")
    await controller.set_overlay_enabled(False)


@pytest.mark.asyncio
async def test_overlay_start_restores_compatibility_refresh_for_existing_presenter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )

    controller = GuiController(
        page=SimpleNamespace(),
        app=_presentation(SimpleNamespace()),
        config_path=Path("settings.json"),
    )
    controller.settings = AppSettings()
    controller.hub = DummyHub()
    _attach_overlay_presenter(
        controller,
        OverlayPresenter(
            calibration=controller.overlay_calibration.copy(),
            clock=controller.clock,
            peer_presentation_refresh_burst=False,
            self_presentation_refresh_burst=False,
        ),
    )

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)

    assert _overlay_runtime(controller).presenter.peer_presentation_refresh_burst is True
    assert _overlay_runtime(controller).presenter.self_presentation_refresh_burst is True
    FakeOverlayProcessManager.instances[0].complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")
    await controller.set_overlay_enabled(False)


@pytest.mark.asyncio
async def test_preserved_presenter_restart_renegotiates_one_retry_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)
    first_manager = FakeOverlayProcessManager.instances[0]
    first_manager.complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")
    presenter = _overlay_runtime(controller).presenter
    assert isinstance(presenter, OverlayPresenter)
    adapter = OverlayEventAdapter(clock=controller.clock)
    target_id = uuid4()
    target_key = ("peer", target_id)
    target_identity = f"peer:{target_id}"
    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=target_id,
                channel="peer",
                text="durable restart source",
                is_final=True,
                created_at=controller.clock.now(),
            ),
            source_language="en",
            target_language="ko",
        )
    )
    await presenter.emit(
        adapter.translation_final(
            utterance_id=target_id,
            channel="peer",
            text="durable restart target",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
        )
    )
    assert presenter._presentation_state.peer_presentation_refresh_target_key == target_key
    assert presenter._peer_presentation_refresh_burst_task is not None
    await first_manager.confirm_native_retry_ownership()
    assert presenter.native_retry_trigger_emission is True
    assert presenter.peer_presentation_refresh_burst is False
    assert presenter.self_presentation_refresh_burst is False
    first_native_snapshot = presenter.snapshot()
    assert first_native_snapshot.native_fresh_render_generations.peer is not None
    assert first_native_snapshot.native_fresh_render_targets.peer == target_identity
    assert presenter._peer_presentation_refresh_burst_task is None

    _overlay_owner(controller).state = "failed"
    await _overlay_owner(controller).begin_start()
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 2)
    second_manager = FakeOverlayProcessManager.instances[1]
    restarted = _overlay_runtime(controller).presenter
    assert restarted is presenter
    assert presenter.native_retry_trigger_emission is False
    assert presenter.peer_presentation_refresh_burst is True
    assert presenter.self_presentation_refresh_burst is True
    fallback_snapshot = presenter.snapshot()
    assert fallback_snapshot.native_fresh_render_generations is None
    assert fallback_snapshot.native_fresh_render_targets is None
    assert presenter._presentation_state.peer_presentation_refresh_target_key == target_key
    assert presenter._peer_presentation_refresh_burst_task is not None
    second_manager.complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")
    await second_manager.confirm_native_retry_ownership()
    assert presenter.native_retry_trigger_emission is True
    assert presenter.peer_presentation_refresh_burst is False
    assert presenter.self_presentation_refresh_burst is False
    second_native_snapshot = presenter.snapshot()
    assert second_native_snapshot.native_fresh_render_generations.peer is not None
    assert second_native_snapshot.native_fresh_render_targets.peer == target_identity
    assert presenter._presentation_state.peer_presentation_refresh_target_key is None
    assert presenter._peer_presentation_refresh_burst_task is None

    _overlay_owner(controller).state = "failed"
    await _overlay_owner(controller).begin_start()
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 3)
    old_manager = FakeOverlayProcessManager.instances[2]
    assert _overlay_runtime(controller).presenter is presenter
    assert presenter.native_retry_trigger_emission is False
    assert presenter.peer_presentation_refresh_burst is True
    assert presenter.self_presentation_refresh_burst is True
    old_fallback_snapshot = presenter.snapshot()
    assert old_fallback_snapshot.native_fresh_render_generations is None
    assert old_fallback_snapshot.native_fresh_render_targets is None
    assert presenter._presentation_state.peer_presentation_refresh_target_key == target_key
    assert presenter._peer_presentation_refresh_burst_task is not None
    old_manager.complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")
    await first_manager.confirm_native_retry_ownership()
    assert presenter.native_retry_trigger_emission is False
    assert presenter.peer_presentation_refresh_burst is True
    assert presenter.self_presentation_refresh_burst is True
    stale_snapshot = presenter.snapshot()
    assert stale_snapshot.native_fresh_render_generations is None
    assert stale_snapshot.native_fresh_render_targets is None
    assert presenter._presentation_state.peer_presentation_refresh_target_key == target_key
    assert presenter._peer_presentation_refresh_burst_task is not None
    await controller.set_overlay_enabled(False)


@pytest.mark.asyncio
async def test_desktop_overlay_start_disables_existing_peer_presentation_refresh_presenter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )

    controller = GuiController(
        page=SimpleNamespace(),
        app=_presentation(SimpleNamespace()),
        config_path=Path("settings.json"),
    )
    controller.settings = AppSettings()
    controller.settings.overlay.target = OVERLAY_TARGET_DESKTOP
    controller.hub = DummyHub()
    _attach_overlay_presenter(
        controller,
        OverlayPresenter(
            calibration=controller.overlay_calibration.copy(),
            clock=controller.clock,
            peer_presentation_refresh_burst=True,
            self_presentation_refresh_burst=True,
        ),
    )

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)

    assert _overlay_owner(controller).runtime is not None
    presenter = _overlay_owner(controller).runtime.presenter
    assert isinstance(presenter, OverlayPresenter)
    assert presenter.peer_presentation_refresh_burst is False
    assert presenter.self_presentation_refresh_burst is False
    FakeOverlayProcessManager.instances[0].complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")
    await controller.set_overlay_enabled(False)


@pytest.mark.asyncio
async def test_overlay_start_syncs_bridge_after_preserved_presenter_cleans_refresh_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )

    bridge_start_released_burst = False
    clock = FakeClock(_now=10.0)
    sleep_events: list[asyncio.Event] = []

    async def fake_sleep(delay: float) -> None:
        release = asyncio.Event()
        sleep_events.append(release)
        await release.wait()
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=fake_sleep,
        peer_presentation_refresh_burst=True,
    )
    adapter = OverlayEventAdapter(clock=clock)
    peer_turn_id = uuid4()
    transcript = Transcript(
        utterance_id=peer_turn_id,
        channel="peer",
        text="peer source preserved across restart",
        is_final=True,
        created_at=10.0,
    )
    await presenter.emit(
        adapter.transcript_final(
            transcript,
            source_language="en",
            target_language="ko",
        )
    )
    await presenter.emit(
        adapter.translation_final(
            utterance_id=peer_turn_id,
            channel="peer",
            text="재시작 중 보존된 번역",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=10.1,
        )
    )
    await asyncio.sleep(0)
    sleep_events[-1].set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await _wait_until(lambda: len(sleep_events) >= 2)

    stale_snapshot = presenter.snapshot()
    assert stale_snapshot.blocks[0].session_scope == "peer_presentation_refresh=1"

    class CleaningDuringStartOverlayBridge(FakeOverlayBridge):
        instances: list["CleaningDuringStartOverlayBridge"] = []

        async def start(self) -> None:
            nonlocal bridge_start_released_burst
            await super().start()
            for _ in range(25):
                if presenter._peer_presentation_refresh_burst_task is None:
                    break
                assert sleep_events, "refresh burst should be waiting before bridge attach"
                sleep_events[-1].set()
                await asyncio.sleep(0)
                await asyncio.sleep(0)
            bridge_start_released_burst = True
            assert presenter._peer_presentation_refresh_burst_task is None
            assert presenter.snapshot().blocks[0].session_scope is None

    class ImmediateConnectedOverlayProcessManager(FakeOverlayProcessManager):
        instances: list["ImmediateConnectedOverlayProcessManager"] = []

        async def start(self) -> None:
            self.state = "connected"
            self.failure_reason = None

    monkeypatch.setattr(
        overlay_generation_start_module,
        "OverlayBridge",
        CleaningDuringStartOverlayBridge,
    )
    monkeypatch.setattr(
        overlay_generation_start_module,
        "OverlayProcessManager",
        ImmediateConnectedOverlayProcessManager,
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.overlay.target = "desktop"
    controller.hub = DummyHub()
    runtime = _overlay_owner(controller).new_runtime()
    runtime.adopt_presenter(presenter)

    await _overlay_owner(controller).run_start(runtime)

    bridge = CleaningDuringStartOverlayBridge.instances[0]
    assert bridge_start_released_burst is True
    assert bridge.initial_snapshot is not stale_snapshot
    assert bridge.initial_snapshot.blocks[0].session_scope is None
    assert presenter.snapshot().blocks[0].session_scope is None
    assert bridge.current_snapshot == presenter.snapshot()
    assert bridge.snapshots == []

    await _overlay_owner(controller).teardown(preserve_presenter_state=False)


@pytest.mark.asyncio
async def test_desktop_overlay_start_cleans_preserved_self_refresh_marker_before_initial_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )

    clock = FakeClock(_now=10.0)
    sleep_events: list[asyncio.Event] = []

    async def fake_sleep(delay: float) -> None:
        release = asyncio.Event()
        sleep_events.append(release)
        await release.wait()
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=fake_sleep,
        peer_presentation_refresh_burst=False,
        self_presentation_refresh_burst=True,
    )
    adapter = OverlayEventAdapter(clock=clock)
    self_turn_id = uuid4()

    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=self_turn_id,
                channel="self",
                text="self source preserved across desktop restart",
                is_final=True,
                created_at=10.0,
            ),
            source_language="ko",
            target_language="en",
        )
    )
    await asyncio.sleep(0)
    assert sleep_events
    sleep_events[-1].set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    stale_snapshot = presenter.snapshot()
    assert stale_snapshot.blocks[0].session_scope == "self_presentation_refresh=1"

    class ImmediateConnectedOverlayProcessManager(FakeOverlayProcessManager):
        instances: list["ImmediateConnectedOverlayProcessManager"] = []

        async def start(self) -> None:
            self.state = "connected"
            self.failure_reason = None

    monkeypatch.setattr(
        overlay_generation_start_module,
        "OverlayProcessManager",
        ImmediateConnectedOverlayProcessManager,
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.overlay.target = OVERLAY_TARGET_DESKTOP
    controller.hub = DummyHub()
    runtime = _overlay_owner(controller).new_runtime()
    runtime.adopt_presenter(presenter)

    try:
        await _overlay_owner(controller).run_start(runtime)

        bridge = FakeOverlayBridge.instances[0]
        assert presenter.self_presentation_refresh_burst is False
        assert bridge.initial_snapshot.blocks[0].session_scope is None
        assert presenter.snapshot().blocks[0].session_scope is None
        assert bridge.current_snapshot == presenter.snapshot()
    finally:
        await _overlay_owner(controller).teardown(preserve_presenter_state=False)


@pytest.mark.asyncio
async def test_successful_overlay_start_refreshes_consumers_after_peer_runtime_becomes_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )

    contracts = []
    app = SimpleNamespace(
        view_dashboard=SimpleNamespace(
            set_overlay_peer_contract=contracts.append,
        )
    )
    controller = _make_controller(app=app)

    def on_overlay_state_changed(*, state: str, failure_reason: str | None = None) -> None:
        app.overlay_state = state
        app.overlay_failure_reason = failure_reason

    app.on_overlay_state_changed = on_overlay_state_changed
    controller._ui_event_bridge = SimpleNamespace(
        report_overlay_state=lambda state, failure_reason=None: on_overlay_state_changed(
            state=state,
            failure_reason=failure_reason,
        )
    )
    controller.settings = AppSettings()
    controller.settings.ui.peer_translation_enabled = True
    controller.settings.ui.peer_translation_eula_accepted = True
    controller.hub = DummyHub(peer_stt=None)

    async def fake_refresh_peer_stt_runtime(self: GuiController) -> None:
        self.hub.peer_stt = object()

    monkeypatch.setattr(GuiController, "_refresh_peer_stt_runtime", fake_refresh_peer_stt_runtime)

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)

    manager = FakeOverlayProcessManager.instances[0]
    manager.complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")

    assert len(contracts) >= 2
    assert any(contract.peer.warning_reason == "runtime_unavailable" for contract in contracts)
    assert contracts[-1].peer.state == "on"
    assert contracts[-1].peer.helper_text == ""


@pytest.mark.asyncio
async def test_overlay_toggle_off_sends_shutdown_event_before_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.ui.overlay_enabled = True
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)

    manager = FakeOverlayProcessManager.instances[0]
    bridge = FakeOverlayBridge.instances[0]
    manager.complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")

    presenter = _overlay_runtime(controller).presenter
    assert presenter is not None
    await presenter.emit(
        SelfTranscriptFinal(
            event_id="self-final",
            seq=1,
            utterance_id=uuid4(),
            channel="self",
            created_at=10.0,
            text="discard me",
            source_language="ko",
            target_language="en",
            is_final=True,
        )
    )

    await controller.set_overlay_enabled(False)

    assert bridge.shutdown_calls == 1
    assert bridge.snapshots[-1].blocks == []
    assert manager.stop_calls == 1


@pytest.mark.asyncio
async def test_begin_overlay_start_uses_empty_runtime_without_owned_presenter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from puripuly_heart.core.runtime.overlay import OverlayRuntimeHandle

    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )

    class ImmediateConnectedOverlayProcessManager(FakeOverlayProcessManager):
        instances: list["ImmediateConnectedOverlayProcessManager"] = []

        async def start(self) -> None:
            self.state = "connected"
            self.failure_reason = None

    monkeypatch.setattr(
        overlay_generation_start_module,
        "OverlayProcessManager",
        ImmediateConnectedOverlayProcessManager,
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.hub = DummyHub()
    _overlay_owner(controller).state = "failed"

    _overlay_owner(controller).runtime = OverlayRuntimeHandle(shutdown_grace_s=0)

    try:
        await _overlay_owner(controller).begin_start()
        await _wait_until(lambda: len(FakeOverlayBridge.instances) == 1)
        await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")

        runtime = _overlay_owner(controller).runtime
        assert runtime is not None
        assert controller.hub.overlay_sink is runtime.presenter
        assert FakeOverlayBridge.instances[0].initial_snapshot.blocks == []
    finally:
        await _overlay_owner(controller).teardown(preserve_presenter_state=False)


@pytest.mark.asyncio
async def test_overlay_start_uses_presenter_owned_by_runtime_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from puripuly_heart.core.runtime.overlay import OverlayRuntimeHandle

    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )

    class ImmediateConnectedOverlayProcessManager(FakeOverlayProcessManager):
        instances: list["ImmediateConnectedOverlayProcessManager"] = []

        async def start(self) -> None:
            self.state = "connected"
            self.failure_reason = None

    monkeypatch.setattr(
        overlay_generation_start_module,
        "OverlayProcessManager",
        ImmediateConnectedOverlayProcessManager,
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.hub = DummyHub()

    stale_presenter = OverlayPresenter(
        calibration=controller.overlay_calibration.copy(),
        clock=controller.clock,
    )
    await stale_presenter.emit(
        SelfTranscriptFinal(
            event_id="stale-self-final",
            seq=1,
            utterance_id=uuid4(),
            channel="self",
            created_at=10.0,
            text="stale mirror should not seed new runtime",
            source_language="ko",
            target_language="en",
            is_final=True,
        )
    )
    runtime = OverlayRuntimeHandle(shutdown_grace_s=0)
    runtime.attach_presenter(stale_presenter)
    _overlay_owner(controller).runtime = runtime

    try:
        await _overlay_owner(controller).run_start(runtime)

        assert runtime.presenter is stale_presenter
        assert controller.hub.overlay_sink is runtime.presenter
        assert FakeOverlayBridge.instances[0].initial_snapshot.blocks != []
    finally:
        await _overlay_owner(controller).teardown(preserve_presenter_state=False)


@pytest.mark.asyncio
async def test_overlay_restart_reuses_presenter_scene_for_new_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)
    FakeOverlayProcessManager.instances[0].complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")

    presenter = _overlay_runtime(controller).presenter
    assert presenter is not None

    utterance_id = uuid4()
    await presenter.emit(
        SelfTranscriptFinal(
            event_id="self-final",
            seq=1,
            utterance_id=utterance_id,
            channel="self",
            created_at=10.0,
            text="persist me",
            source_language="ko",
            target_language="en",
            is_final=True,
        )
    )
    saved_snapshot = presenter.snapshot()

    await _overlay_owner(controller).teardown(preserve_presenter_state=True)

    assert _overlay_runtime(controller).presenter is presenter
    assert controller.hub.overlay_sink is None

    _overlay_owner(controller).state = "failed"
    await _overlay_owner(controller).begin_start()
    await _wait_until(lambda: len(FakeOverlayBridge.instances) == 2)

    assert FakeOverlayBridge.instances[1].initial_snapshot == saved_snapshot
    assert _overlay_runtime(controller).presenter is presenter


@pytest.mark.asyncio
async def test_preserved_overlay_presenter_detaches_from_hub_ingress_until_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)
    FakeOverlayProcessManager.instances[0].complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")

    presenter = _overlay_runtime(controller).presenter
    assert presenter is not None

    await presenter.emit(
        SelfTranscriptFinal(
            event_id="self-final",
            seq=1,
            utterance_id=uuid4(),
            channel="self",
            created_at=10.0,
            text="preserve without stale ingress",
            source_language="ko",
            target_language="en",
            is_final=True,
        )
    )
    saved_snapshot = presenter.snapshot()

    await _overlay_owner(controller).teardown(preserve_presenter_state=True)

    assert _overlay_runtime(controller).presenter is presenter
    assert controller.hub.overlay_sink is None

    _overlay_owner(controller).state = "failed"
    await _overlay_owner(controller).begin_start()
    await _wait_until(lambda: len(FakeOverlayBridge.instances) == 2)

    assert _overlay_runtime(controller).presenter is presenter
    assert FakeOverlayBridge.instances[1].initial_snapshot == saved_snapshot


@pytest.mark.asyncio
async def test_overlay_restart_detaches_preserved_presenter_from_old_runtime_before_adoption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)
    FakeOverlayProcessManager.instances[0].complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")

    presenter = _overlay_runtime(controller).presenter
    assert presenter is not None
    await presenter.emit(
        SelfTranscriptFinal(
            event_id="self-final",
            seq=1,
            utterance_id=uuid4(),
            channel="self",
            created_at=10.0,
            text="preserve through runtime owner",
            source_language="ko",
            target_language="en",
            is_final=True,
        )
    )
    saved_snapshot = presenter.snapshot()

    await _overlay_owner(controller).teardown(preserve_presenter_state=True)
    old_runtime = _overlay_owner(controller).runtime
    assert old_runtime is not None
    assert old_runtime.presenter is presenter

    _overlay_owner(controller).state = "failed"

    try:
        await _overlay_owner(controller).begin_start()
        await _wait_until(lambda: len(FakeOverlayBridge.instances) == 2)

        new_runtime = _overlay_owner(controller).runtime
        assert new_runtime is not None
        assert new_runtime is not old_runtime
        assert old_runtime.presenter is None
        assert new_runtime.presenter is presenter
        assert FakeOverlayBridge.instances[1].initial_snapshot == saved_snapshot
    finally:
        if len(FakeOverlayProcessManager.instances) >= 2:
            FakeOverlayProcessManager.instances[1].complete_startup()
            with contextlib.suppress(AssertionError):
                await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")
        await _overlay_owner(controller).teardown(preserve_presenter_state=False)


@pytest.mark.asyncio
async def test_overlay_restart_applies_current_preferences_before_bridge_initial_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)
    FakeOverlayProcessManager.instances[0].complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")

    presenter = _overlay_runtime(controller).presenter
    assert presenter is not None
    await presenter.emit(
        SelfTranscriptFinal(
            event_id="self-final",
            seq=1,
            utterance_id=uuid4(),
            channel="self",
            created_at=10.0,
            text="current preferences must seed restart bridge",
            source_language="ko",
            target_language="en",
            is_final=True,
        )
    )
    assert FakeOverlayBridge.instances[0].snapshots[-1].blocks[0].secondary_enabled is True

    await _overlay_owner(controller).teardown(preserve_presenter_state=True)

    controller.settings.overlay.show_translation = False
    controller.settings.overlay.show_peer_original = False
    controller.settings.overlay.calibration = OverlayCalibration(distance=1.7, offset_x=0.4)
    controller._get_overlay_calibration_application_owner().replace_current(
        controller.settings.overlay.calibration.copy()
    )
    _overlay_owner(controller).state = "failed"

    try:
        await _overlay_owner(controller).begin_start()
        await _wait_until(lambda: len(FakeOverlayBridge.instances) == 2)

        restarted_bridge = FakeOverlayBridge.instances[1]
        assert restarted_bridge.initial_snapshot.blocks[0].secondary_enabled is False
        assert restarted_bridge.initial_snapshot.calibration.distance == 1.7
        assert restarted_bridge.initial_snapshot.calibration.offset_x == 0.4
    finally:
        if len(FakeOverlayProcessManager.instances) >= 2:
            FakeOverlayProcessManager.instances[1].complete_startup()
            with contextlib.suppress(AssertionError):
                await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")
        await _overlay_owner(controller).teardown(preserve_presenter_state=False)


@pytest.mark.asyncio
async def test_explicit_overlay_disable_resets_presenter_scene_for_next_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)
    FakeOverlayProcessManager.instances[0].complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")

    presenter = _overlay_runtime(controller).presenter
    assert presenter is not None
    await presenter.emit(
        SelfTranscriptFinal(
            event_id="self-final",
            seq=1,
            utterance_id=uuid4(),
            channel="self",
            created_at=10.0,
            text="discard me",
            source_language="ko",
            target_language="en",
            is_final=True,
        )
    )

    await controller.set_overlay_enabled(False)

    assert _overlay_owner(controller).runtime is None
    assert FakeOverlayBridge.instances[0].snapshots[-1].blocks == []

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayBridge.instances) == 2)

    assert FakeOverlayBridge.instances[1].initial_snapshot.blocks == []


@pytest.mark.asyncio
async def test_refresh_overlay_runtime_dependencies_does_not_clear_overlay_scene(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.hub = DummyHub(peer_stt=object())

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)
    FakeOverlayProcessManager.instances[0].complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")

    presenter = _overlay_runtime(controller).presenter
    bridge = FakeOverlayBridge.instances[0]
    assert presenter is not None

    await presenter.emit(
        SelfTranscriptFinal(
            event_id="self-final",
            seq=1,
            utterance_id=uuid4(),
            channel="self",
            created_at=10.0,
            text="stay visible",
            source_language="ko",
            target_language="en",
            is_final=True,
        )
    )
    saved_snapshot = bridge.snapshots[-1]
    controller._last_peer_stt_runtime_signature = _peer_runtime_signature(
        controller, controller.settings
    )

    await controller._refresh_overlay_runtime_dependencies()

    assert bridge.snapshots[-1] == saved_snapshot


@pytest.mark.asyncio
async def test_explicit_overlay_off_clears_saved_peer_translation_toggle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.ui.overlay_enabled = True
    controller.settings.ui.peer_translation_enabled = True

    await controller.set_overlay_enabled(False)

    assert controller.settings.ui.overlay_enabled is False
    assert controller.settings.ui.peer_translation_enabled is False


def test_effective_integrated_context_falls_back_until_peer_translation_is_effective() -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.ui.integrated_context_enabled = True
    controller.hub = DummyHub(peer_stt=object())

    assert controller._effective_integrated_context_enabled_for(controller.settings) is False

    _overlay_owner(controller).state = "connected"
    controller.settings.ui.peer_translation_enabled = True
    controller.settings.ui.peer_translation_eula_accepted = True

    assert controller._effective_integrated_context_enabled_for(controller.settings) is True

    controller.settings.ui.peer_translation_enabled = False

    assert controller._effective_integrated_context_enabled_for(controller.settings) is False


@pytest.mark.asyncio
async def test_overlay_start_failure_keeps_saved_preferences_but_effective_state_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    monkeypatch.setattr(
        GuiController, "_replace_runtime_stt_provider", lambda self: asyncio.sleep(0)
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.ui.peer_translation_enabled = True
    controller.settings.ui.peer_translation_eula_accepted = True
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)

    manager = FakeOverlayProcessManager.instances[0]
    manager.complete_startup(failure_reason="renderer_init_failed")
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "failed")

    assert controller.settings.ui.overlay_enabled is True
    assert _overlay_owner(controller).snapshot.failure_reason == "renderer_init_failed"
    presentation = controller.overlay_peer_presentation_state()
    assert presentation is not None
    assert presentation.peer_effective_enabled is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_reason",
    [
        "stale_overlay_build",
        "vendored_openvr_dll_missing",
        "packaged_openvr_dll_missing",
        "openvr_dll_hash_mismatch",
        "hmd_not_found",
    ],
)
async def test_overlay_start_failure_preserves_specific_preflight_reason(
    monkeypatch: pytest.MonkeyPatch,
    failure_reason: str,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)

    manager = FakeOverlayProcessManager.instances[0]
    manager.complete_startup(failure_reason=failure_reason)
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "failed")

    assert controller.settings.ui.overlay_enabled is True
    assert _overlay_owner(controller).snapshot.failure_reason == failure_reason


@pytest.mark.asyncio
async def test_overlay_runtime_disconnect_keeps_saved_preferences_without_auto_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    monkeypatch.setattr(
        GuiController, "_replace_runtime_stt_provider", lambda self: asyncio.sleep(0)
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.ui.peer_translation_enabled = True
    controller.settings.ui.peer_translation_eula_accepted = True
    controller.hub = DummyHub(peer_stt=object())

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)
    manager = FakeOverlayProcessManager.instances[0]
    manager.complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")
    assert controller.hub.peer_translation_enabled is True

    manager.trigger_runtime_failure("runtime_disconnected")
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "failed")

    assert controller.settings.ui.overlay_enabled is True
    assert controller.settings.ui.peer_translation_enabled is True
    assert _overlay_owner(controller).snapshot.failure_reason == "runtime_disconnected"
    presentation = controller.overlay_peer_presentation_state()
    assert presentation is not None
    assert presentation.peer_effective_enabled is False
    assert controller.hub.peer_translation_enabled is False
    assert _overlay_owner(controller).snapshot.auto_restart_scheduled is False


@pytest.mark.asyncio
async def test_overlay_runtime_crash_keeps_saved_preferences_without_auto_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    monkeypatch.setattr(
        GuiController, "_replace_runtime_stt_provider", lambda self: asyncio.sleep(0)
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.ui.peer_translation_enabled = True
    controller.settings.ui.peer_translation_eula_accepted = True
    controller.hub = DummyHub(peer_stt=object())

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)
    manager = FakeOverlayProcessManager.instances[0]
    manager.complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")
    assert controller.hub.peer_translation_enabled is True

    manager.trigger_runtime_failure("runtime_crashed")
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "failed")

    assert controller.settings.ui.overlay_enabled is True
    assert controller.settings.ui.peer_translation_enabled is True
    assert _overlay_owner(controller).snapshot.failure_reason == "runtime_crashed"
    assert controller.hub.peer_translation_enabled is False
    assert _overlay_owner(controller).snapshot.auto_restart_scheduled is False


def test_overlay_runtime_crash_logs_state_transition() -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller._runtime_logging = RuntimeLoggingSpy()
    _overlay_owner(controller).state = "connected"
    _attach_overlay_manager(controller, SimpleNamespace(state="failed"))
    _attach_overlay_presenter(controller, object())
    _attach_overlay_bridge(controller, object())

    _overlay_owner(controller).on_runtime_crashed()

    assert _overlay_owner(controller).snapshot.state == "failed"
    assert controller._runtime_logging.basic_messages == [
        (
            logging.INFO,
            "[Overlay] State transition: connected -> failed failure_reason=runtime_crashed",
        )
    ]
    assert controller._runtime_logging.detailed_messages == [
        (
            logging.INFO,
            "[Overlay] State detail: presenter_attached=True bridge_attached=True manager_state=failed",
        )
    ]


@pytest.mark.asyncio
async def test_overlay_successful_recovery_clears_previous_failure_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    monkeypatch.setattr(
        GuiController, "_replace_runtime_stt_provider", lambda self: asyncio.sleep(0)
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 1)
    FakeOverlayProcessManager.instances[0].complete_startup(failure_reason="bridge_auth_failed")
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "failed")

    assert _overlay_owner(controller).snapshot.failure_reason == "bridge_auth_failed"

    await controller.set_overlay_enabled(False)
    assert _overlay_owner(controller).snapshot.state == "off"
    assert _overlay_owner(controller).snapshot.failure_reason == "bridge_auth_failed"

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayProcessManager.instances) == 2)
    FakeOverlayProcessManager.instances[1].complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")

    assert _overlay_owner(controller).snapshot.failure_reason is None


@pytest.mark.asyncio
async def test_stop_terminally_closes_vrc_receiver_runtime_before_hub_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []
    controller = _make_controller(app=SimpleNamespace())

    async def fake_set_stt_enabled(self, enabled: bool) -> None:
        _ = self
        events.append(("stt", enabled))

    class FakeReceiverRuntime:
        async def stop(self, *, strict_runtime_errors: bool = False) -> None:
            events.append(("receiver_stop", strict_runtime_errors))

        async def close(self) -> None:
            events.append(("receiver_close", None))
            controller.receiver = None

    class FakeHub:
        async def stop(self) -> None:
            events.append(("hub_stop", None))

    class FakeSender:
        def close(self) -> None:
            events.append(("sender_close", None))

    monkeypatch.setattr(GuiController, "set_stt_enabled", fake_set_stt_enabled)
    controller._get_vrc_mic_sync_owner().runtime = FakeReceiverRuntime()
    controller.receiver = object()
    controller.hub = FakeHub()
    controller.sender = FakeSender()
    controller._bridge_task = asyncio.create_task(asyncio.sleep(3600))

    await controller.stop()

    assert events[:3] == [("stt", False), ("receiver_close", None), ("hub_stop", None)]
    assert ("receiver_stop", False) not in events
    assert controller.hub is None
    assert controller.sender is None
    assert controller._bridge_task is None


@pytest.mark.asyncio
async def test_stop_aggregates_vrc_receiver_close_failure_and_still_stops_hub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    controller = _make_controller(app=SimpleNamespace())
    receiver = object()

    class FailingReceiverRuntime:
        async def stop(self, *, strict_runtime_errors: bool = False) -> None:
            _ = strict_runtime_errors
            events.append("receiver_stop")

        async def close(self) -> None:
            events.append("receiver_close")
            raise RuntimeError("receiver close failed")

    class FakeHub:
        async def stop(self) -> None:
            events.append("hub_stop")

    class FakeSender:
        def close(self) -> None:
            events.append("sender_close")

    async def fake_set_stt_enabled(self, enabled: bool) -> None:
        _ = (self, enabled)
        events.append("stt_off")

    async def fake_shutdown_overlay(self, *, preserve_failure_reason: bool) -> None:
        _ = (self, preserve_failure_reason)
        events.append("overlay_shutdown")

    owner = controller._get_vrc_mic_sync_owner()
    owner.runtime = FailingReceiverRuntime()
    controller.receiver = receiver
    controller.hub = FakeHub()
    controller.sender = FakeSender()

    monkeypatch.setattr(GuiController, "set_stt_enabled", fake_set_stt_enabled)
    monkeypatch.setattr(OverlayApplicationOwner, "shutdown", fake_shutdown_overlay)

    with pytest.raises(RuntimeError, match="receiver close failed"):
        await controller.stop()

    assert events[:4] == ["stt_off", "receiver_close", "overlay_shutdown", "hub_stop"]
    assert events[-1] == "sender_close"
    assert "receiver_stop" not in events
    assert owner.runtime is not None
    assert controller.receiver is receiver


@pytest.mark.asyncio
async def test_stop_closes_peer_runtime_without_replacing_self_stt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.hub = DummyHub(llm=object(), stt=object(), peer_stt=object())
    controller._peer_runtime = DummyPeerRuntime()

    monkeypatch.setattr(GuiController, "set_stt_enabled", lambda self, value: asyncio.sleep(0))
    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        lambda self, enabled: asyncio.sleep(0),
    )
    monkeypatch.setattr(
        OverlayApplicationOwner,
        "shutdown",
        lambda self, preserve_failure_reason: asyncio.sleep(0),
    )

    await controller.stop()

    assert controller._peer_runtime is None
    assert controller.hub is None


@pytest.mark.asyncio
async def test_stop_preserves_peer_runtime_when_close_fails_and_stops_hub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    hub = DummyHub(llm=object(), stt=object(), peer_stt=object())

    class FailingPeerRuntime(DummyPeerRuntime):
        async def close(self) -> None:
            raise RuntimeError("peer runtime close failed")

    peer_runtime = FailingPeerRuntime()
    controller.hub = hub
    controller._peer_runtime = peer_runtime

    monkeypatch.setattr(GuiController, "set_stt_enabled", lambda self, value: asyncio.sleep(0))
    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        lambda self, enabled: asyncio.sleep(0),
    )
    monkeypatch.setattr(
        OverlayApplicationOwner,
        "shutdown",
        lambda self, preserve_failure_reason: asyncio.sleep(0),
    )

    with pytest.raises(RuntimeError, match="peer runtime close failed"):
        await controller.stop()

    assert controller._peer_runtime is peer_runtime
    assert hub.stop_calls == 1
    assert controller.hub is None


@pytest.mark.asyncio
async def test_stop_preserves_hub_when_hub_stop_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    hub = DummyHub(llm=object(), stt=object(), peer_stt=object())

    async def failing_stop() -> None:
        hub.stop_calls += 1
        raise RuntimeError("hub stop failed")

    hub.stop = failing_stop  # type: ignore[method-assign]
    controller.hub = hub

    monkeypatch.setattr(GuiController, "set_stt_enabled", lambda self, value: asyncio.sleep(0))
    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        lambda self, enabled: asyncio.sleep(0),
    )
    monkeypatch.setattr(
        OverlayApplicationOwner,
        "shutdown",
        lambda self, preserve_failure_reason: asyncio.sleep(0),
    )

    with pytest.raises(RuntimeError, match="hub stop failed"):
        await controller.stop()

    assert controller.hub is hub
    assert hub.stop_calls == 1


@pytest.mark.asyncio
async def test_stop_closes_runtime_logging_service(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = _make_controller(app=SimpleNamespace())
    events: list[str] = []

    class FakeRuntimeLogging:
        def close_after_producers_stop(self, *, cleanup_failures=()) -> None:
            events.append(
                "runtime_logging_summary:"
                + ",".join(type(failure).__name__ for failure in cleanup_failures)
            )
            events.append("runtime_logging_close")

    monkeypatch.setattr(GuiController, "set_stt_enabled", lambda self, value: asyncio.sleep(0))
    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        lambda self, enabled: asyncio.sleep(0),
    )
    monkeypatch.setattr(
        OverlayApplicationOwner,
        "shutdown",
        lambda self, preserve_failure_reason: asyncio.sleep(0),
    )

    controller._runtime_logging = FakeRuntimeLogging()

    await controller.stop()

    assert events == ["runtime_logging_summary:", "runtime_logging_close"]
    assert controller._runtime_logging is not None


@pytest.mark.asyncio
async def test_stop_emits_shutdown_summary_after_hub_failure_before_logging_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    events: list[str] = []
    hub = DummyHub(llm=object(), stt=object(), peer_stt=object())

    async def failing_stop() -> None:
        hub.stop_calls += 1
        events.append("hub_stop")
        raise RuntimeError("hub stop failed with raw detail")

    class FakeRuntimeLogging:
        def close_after_producers_stop(self, *, cleanup_failures=()) -> None:
            events.append(
                "runtime_logging_summary:"
                + ",".join(type(failure).__name__ for failure in cleanup_failures)
            )
            events.append("runtime_logging_close")

    hub.stop = failing_stop  # type: ignore[method-assign]
    controller.hub = hub
    controller._runtime_logging = FakeRuntimeLogging()

    monkeypatch.setattr(GuiController, "set_stt_enabled", lambda self, value: asyncio.sleep(0))
    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        lambda self, enabled: asyncio.sleep(0),
    )
    monkeypatch.setattr(
        OverlayApplicationOwner,
        "shutdown",
        lambda self, preserve_failure_reason: asyncio.sleep(0),
    )

    with pytest.raises(RuntimeError, match="hub stop failed"):
        await controller.stop()

    assert events == [
        "hub_stop",
        "runtime_logging_summary:RuntimeError",
        "runtime_logging_close",
    ]
    assert controller._runtime_logging is not None


@pytest.mark.asyncio
async def test_log_basic_after_stop_uses_closed_logging_owner_without_recreation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    runtime_logging = RuntimeLoggingSpy()
    controller._runtime_logging = runtime_logging
    new_service_creations: list[str] = []

    def create_new_runtime_logging(*_args, **_kwargs):
        new_service_creations.append("created")
        return RuntimeLoggingSpy()

    monkeypatch.setattr(
        controller_module,
        "SessionRuntimeLoggingService",
        create_new_runtime_logging,
    )
    monkeypatch.setattr(GuiController, "set_stt_enabled", lambda self, value: asyncio.sleep(0))
    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        lambda self, enabled: asyncio.sleep(0),
    )
    monkeypatch.setattr(
        OverlayApplicationOwner,
        "shutdown",
        lambda self, preserve_failure_reason: asyncio.sleep(0),
    )

    await controller.stop()
    stopped_runtime_logging = controller._runtime_logging
    controller.log_basic("late after stop")

    assert controller._runtime_logging is stopped_runtime_logging
    assert runtime_logging.basic_messages[-1] == (logging.INFO, "late after stop")
    assert new_service_creations == []


@pytest.mark.asyncio
async def test_runtime_logging_close_failure_is_aggregated_not_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    hub = DummyHub(llm=object(), stt=object(), peer_stt=object())
    events: list[str] = []

    async def failing_stop() -> None:
        hub.stop_calls += 1
        events.append("hub_stop")
        raise RuntimeError("hub stop failed")

    class FakeRuntimeLogging:
        def close_after_producers_stop(self, *, cleanup_failures=()) -> None:
            events.append(
                "runtime_logging_summary:"
                + ",".join(type(failure).__name__ for failure in cleanup_failures)
            )
            events.append("runtime_logging_close")
            raise OSError("runtime logging close failed")

    hub.stop = failing_stop  # type: ignore[method-assign]
    controller.hub = hub
    controller._runtime_logging = FakeRuntimeLogging()

    monkeypatch.setattr(GuiController, "set_stt_enabled", lambda self, value: asyncio.sleep(0))
    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        lambda self, enabled: asyncio.sleep(0),
    )
    monkeypatch.setattr(
        OverlayApplicationOwner,
        "shutdown",
        lambda self, preserve_failure_reason: asyncio.sleep(0),
    )

    with pytest.raises(ExceptionGroup) as exc_info:
        await controller.stop()

    assert [type(failure).__name__ for failure in exc_info.value.exceptions] == [
        "RuntimeError",
        "OSError",
    ]
    assert events == [
        "hub_stop",
        "runtime_logging_summary:RuntimeError",
        "runtime_logging_close",
    ]


@pytest.mark.asyncio
async def test_stop_aggregates_oauth_runtime_close_failure_and_continues_later_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    events: list[str] = []
    oauth_failure = RuntimeError("oauth cleanup sentinel")
    hub_failure = RuntimeError("hub cleanup sentinel")
    logging_failure = OSError("runtime logging cleanup sentinel")

    class FailingOAuthRuntime:
        async def close(self) -> None:
            events.append("oauth_close")
            raise oauth_failure

    class FakeRuntimeLogging:
        def close_after_producers_stop(self, *, cleanup_failures=()) -> None:
            events.append(
                "runtime_logging_summary:"
                + ",".join(type(failure).__name__ for failure in cleanup_failures)
            )
            events.append("runtime_logging_close")
            raise logging_failure

    hub = DummyHub(llm=object(), stt=object(), peer_stt=object())

    async def failing_stop() -> None:
        hub.stop_calls += 1
        events.append("hub_stop")
        raise hub_failure

    async def fake_set_stt_enabled(self, value: bool) -> None:
        _ = (self, value)
        events.append("stt_off")

    async def fake_shutdown_overlay(self, *, preserve_failure_reason: bool) -> None:
        _ = (self, preserve_failure_reason)
        events.append("overlay_shutdown")

    hub.stop = failing_stop  # type: ignore[method-assign]
    controller.hub = hub
    controller._get_openrouter_pkce_flow_owner().runtime = FailingOAuthRuntime()  # type: ignore[assignment]
    controller._runtime_logging = FakeRuntimeLogging()

    monkeypatch.setattr(GuiController, "set_stt_enabled", fake_set_stt_enabled)
    monkeypatch.setattr(OverlayApplicationOwner, "shutdown", fake_shutdown_overlay)

    with pytest.raises(ExceptionGroup) as exc_info:
        await controller.stop()

    assert list(exc_info.value.exceptions) == [
        oauth_failure,
        hub_failure,
        logging_failure,
    ]
    assert events == [
        "oauth_close",
        "stt_off",
        "overlay_shutdown",
        "hub_stop",
        "runtime_logging_summary:RuntimeError,RuntimeError",
        "runtime_logging_close",
    ]
    assert controller.hub is hub
    assert controller._runtime_logging is not None


@pytest.mark.asyncio
async def test_stop_closes_app_owned_oauth_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class FakeApp:
        async def close_oauth_runtime(self) -> None:
            events.append("app_oauth_close")

    controller = _make_controller(app=FakeApp())

    monkeypatch.setattr(GuiController, "set_stt_enabled", lambda self, value: asyncio.sleep(0))
    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        lambda self, enabled: asyncio.sleep(0),
    )
    monkeypatch.setattr(
        OverlayApplicationOwner,
        "shutdown",
        lambda self, preserve_failure_reason: asyncio.sleep(0),
    )

    await controller.stop()

    assert events == ["app_oauth_close"]


def test_log_error_fallback_does_not_append_duplicate_ui_line(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    logs = DummyLogsView()
    controller = _make_controller(app=SimpleNamespace(view_logs=logs))
    controller._runtime_logging = RuntimeLoggingSpy(basic_error=RuntimeError("boom"))

    with caplog.at_level(logging.ERROR, logger=controller_module.logger.name):
        controller._log_error("shared failure")

    assert logs.logs == []
    assert any("shared failure" in message for message in caplog.messages)


def test_overlay_state_transition_routes_snapshot_details_to_detailed_log() -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller._runtime_logging = RuntimeLoggingSpy()
    _overlay_owner(controller).failure_reason = "runtime_crashed"
    _attach_overlay_presenter(controller, object())
    _attach_overlay_bridge(controller, object())
    _attach_overlay_manager(controller, SimpleNamespace(state="failed"))

    controller._get_overlay_application_owner()._log_state_transition(
        "connected",
        "failed",
    )

    assert controller._runtime_logging.basic_messages == [
        (
            logging.INFO,
            "[Overlay] State transition: connected -> failed failure_reason=runtime_crashed",
        )
    ]
    assert controller._runtime_logging.detailed_messages == [
        (
            logging.INFO,
            "[Overlay] State detail: presenter_attached=True bridge_attached=True manager_state=failed",
        )
    ]


@pytest.mark.asyncio
async def test_apply_settings_updates_vrc_gate_and_reconfigures_receiver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    settings = AppSettings()
    controller.settings = settings
    controller.hub = DummyHub()
    controller.hub.source_language = settings.languages.source_language
    controller.hub.target_language = settings.languages.target_language
    controller.hub.system_prompt = settings.system_prompt
    controller.hub.low_latency_mode = settings.stt.low_latency_mode
    controller.hub.low_latency_merge_gap_ms = settings.stt.low_latency_merge_gap_ms
    controller.hub.low_latency_spec_retry_max = settings.stt.low_latency_spec_retry_max
    controller._last_stt_runtime_signature = build_self_stt_runtime_signature(settings)

    gate = DummyGate()
    configure_calls: list[bool] = []
    _patch_settings_save(monkeypatch, lambda *_args, **_kwargs: None)

    async def fake_configure_vrc_mic_receiver(self, *, enabled: bool) -> None:
        _ = self
        configure_calls.append(enabled)

    controller.vrc_mic_audio_gate = gate
    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        fake_configure_vrc_mic_receiver,
    )

    settings.osc.vrc_mic_intercept = True
    await controller.apply_settings(settings)
    settings.osc.vrc_mic_intercept = False
    await controller.apply_settings(settings)

    assert gate.enabled_calls == [True, False]
    assert configure_calls == [True, False]


@pytest.mark.asyncio
async def test_init_pipeline_initializes_vrc_state_and_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.osc.vrc_mic_intercept = True
    controller.receiver = object()
    configure_calls: list[bool] = []

    _patch_init_pipeline_dependencies(monkeypatch)

    async def fake_configure_vrc_mic_receiver(self, *, enabled: bool) -> None:
        _ = self
        configure_calls.append(enabled)

    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        fake_configure_vrc_mic_receiver,
    )

    await controller._init_pipeline()

    assert isinstance(controller.vrc_mic_state, VrcMicState)
    assert isinstance(controller.vrc_mic_audio_gate, VrcMicAudioGate)
    assert controller.vrc_mic_audio_gate.state is controller.vrc_mic_state
    assert controller.vrc_mic_audio_gate.enabled is True
    assert controller.vrc_mic_audio_gate.receiver_active is True
    assert controller.vrc_mic_audio_gate._sync_deadline is not None
    assert configure_calls == [True]


@pytest.mark.asyncio
async def test_init_pipeline_reuses_existing_gate_and_updates_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.osc.vrc_mic_intercept = True
    controller.receiver = object()
    original_state = VrcMicState(muted=False)
    gate = VrcMicAudioGate(state=original_state, enabled=False)

    _patch_init_pipeline_dependencies(monkeypatch)

    async def fake_configure_vrc_mic_receiver(self, *, enabled: bool) -> None:
        _ = self
        _ = enabled

    controller.vrc_mic_audio_gate = gate
    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        fake_configure_vrc_mic_receiver,
    )

    await controller._init_pipeline()

    assert controller.vrc_mic_audio_gate is gate
    assert controller.vrc_mic_state is not None
    assert gate.state is controller.vrc_mic_state
    assert gate.state is not original_state
    assert gate.enabled is True
    assert gate.receiver_active is True
    assert gate._sync_deadline is not None


@pytest.mark.asyncio
async def test_init_pipeline_configures_receiver_after_pipeline_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.osc.vrc_mic_intercept = True
    created = _patch_init_pipeline_dependencies(monkeypatch)
    snapshots: list[tuple[bool, bool, bool, bool]] = []

    async def fake_configure_vrc_mic_receiver(self, *, enabled: bool) -> None:
        _ = self
        snapshots.append(
            (
                controller.sender is created["sender"],
                controller.osc is created["osc"],
                controller.hub is created["hub"],
                enabled,
            )
        )

    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        fake_configure_vrc_mic_receiver,
    )

    await controller._init_pipeline()

    assert snapshots == [(True, True, True, True)]


@pytest.mark.asyncio
async def test_init_pipeline_passes_runtime_logging_to_smart_osc_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    created = _patch_init_pipeline_dependencies(monkeypatch)

    await controller._init_pipeline()

    assert created["osc_kwargs"]["runtime_logging"] is controller.runtime_logging


def _self_mic_decision(
    *,
    device_idx: int | None,
    preferred_channels: int,
    status: str = "ok",
    name: str | None = "Compat Mic",
) -> SelfMicCaptureChannelDecision:
    return SelfMicCaptureChannelDecision(
        device_idx=device_idx,
        internal_channels=1,
        preferred_capture_channels=preferred_channels,
        metadata=SoundDeviceInputMetadata(
            device_idx=device_idx,
            name=name,
            max_input_channels=preferred_channels,
            default_samplerate=48000.0,
            metadata_status=status,
        ),
    )


def _create_self_capture_source_via_adapter_for_test(controller: GuiController) -> object:
    assert controller.settings is not None
    source = SelfCaptureSourceAdapter(
        normalize_host_api=normalize_input_host_api,
        resolve_device=audio_source_module.resolve_sounddevice_input_device,
        channel_decision=controller_module.determine_self_mic_capture_channels,
        source_factory=controller_module.SoundDeviceAudioSource,
        log_detailed=controller.log_detailed,
        wrap_source=lambda raw_source: controller._wrap_diagnostic_audio_source(
            raw_source,
            channel_label="self",
        ),
    )(
        build_self_capture_session_config(controller.settings),
    )
    controller._audio_source = source
    return source


class MicTestSelfCaptureOwner:
    def __init__(self, *, source: object, task: asyncio.Task[None]) -> None:
        self.source = source
        self.cleanup_source = None
        self.loop_task = task
        self.vad = object()
        self.last_cleanup_exception = None
        self.generation = 0

    async def apply_intent(self, config, *, enabled: bool, **kwargs):
        _ = config, kwargs
        assert enabled is False
        self.generation += 1
        if self.loop_task is not None:
            self.loop_task.cancel()
            await asyncio.gather(self.loop_task, return_exceptions=True)
            self.loop_task = None
        source = self.source if self.source is not None else self.cleanup_source
        self.source = None
        self.vad = None
        failure = None
        if source is not None:
            try:
                await getattr(source, "close")()
            except Exception as exc:
                failure = exc
                self.cleanup_source = source
                self.last_cleanup_exception = exc
            else:
                self.cleanup_source = None
                self.last_cleanup_exception = None
        return SelfCaptureSessionSnapshot(
            state=(
                SelfCaptureSessionState.FAULTED
                if failure is not None
                else SelfCaptureSessionState.STOPPED
            ),
            provider_status=SelfCaptureProviderStatus.DETACHED,
            desired_active=False,
            effective_active=False,
            generation=self.generation,
            provider_id=None,
            runtime_signature=None,
            failure_reason=(
                SelfCaptureFailureReason.CLEANUP_FAILED if failure is not None else None
            ),
            admission_reason=None,
            has_source=False,
            has_vad=False,
            has_loop_task=False,
            cleanup_debt=1 if failure is not None else 0,
            closed=False,
        )


def _mic_test_route_observation(
    *,
    should_attempt_open: bool = True,
    saved_host_api: str = WINDOWS_MME_HOST_API,
    actual_host_api: str = WINDOWS_MME_HOST_API,
    requested_device: str = "마이크",
    hostapi_index: int | None = 2,
    resolved_device_idx: int | None = 12,
    resolved_device_name: str | None = "마이크",
    resolution_exception_class: str | None = None,
    resolution_exception_message: str | None = None,
    wasapi_auto_convert: bool = False,
    wasapi_exclusive: bool = False,
) -> MicrophoneTestRouteObservation:
    return MicrophoneTestRouteObservation(
        saved_host_api=saved_host_api,
        actual_host_api=actual_host_api,
        requested_device=requested_device,
        hostapi_index=hostapi_index,
        resolved_device_idx=resolved_device_idx,
        resolved_device_name=resolved_device_name,
        resolution_exception_class=resolution_exception_class,
        resolution_exception_message=resolution_exception_message,
        should_attempt_open=should_attempt_open,
        wasapi_auto_convert=wasapi_auto_convert,
        wasapi_exclusive=wasapi_exclusive,
    )


def _mic_test_basic_messages(controller: GuiController) -> list[str]:
    runtime_logging = controller._runtime_logging
    assert runtime_logging is not None
    return [message for _level, message in runtime_logging.basic_messages]


class _CallbackMicrophoneTestCapturePort:
    def __init__(
        self,
        controller: GuiController,
        callback: Callable[..., Awaitable[None]],
    ) -> None:
        self.controller = controller
        self.callback = callback

    async def capture(
        self,
        request: MicrophoneTestCaptureRequest,
        *,
        runtime: MicrophoneTestRuntimePort,
    ) -> None:
        _ = runtime
        await self.callback(
            self.controller,
            generation=request.generation,
            meter_callback=request.meter_callback,
            level_log_interval_s=request.level_log_interval_s,
        )


def _patch_microphone_test_capture(
    monkeypatch: pytest.MonkeyPatch,
    controller: GuiController,
    callback: Callable[..., Awaitable[None]],
) -> None:
    capture_port = _CallbackMicrophoneTestCapturePort(controller, callback)
    owner = controller._microphone_test_owner
    if owner is not None:
        owner.capture_port = capture_port
    monkeypatch.setattr(
        GuiController,
        "_build_microphone_test_capture_adapter",
        lambda _controller: capture_port,
    )


async def _capture_via_microphone_test_port(
    controller: GuiController,
    *,
    generation: int | None = None,
    meter_callback: Callable[[float], object] | None = None,
    level_log_interval_s: float = 1.0,
) -> None:
    capture_port = controller._build_microphone_test_capture_adapter()
    request = controller._microphone_test_capture_request(
        generation,
        meter_callback,
        level_log_interval_s,
    )
    await capture_port.capture(
        request,
        runtime=controller._get_microphone_test_owner().runtime,
    )


def _assert_mic_test_event_and_field_names_have_no_verdict_labels(messages: list[str]) -> None:
    banned = {"success", "failure", "failed", "usable", "near_silence", "good", "bad"}
    for message in messages:
        if not message.startswith("[MicTest] "):
            continue
        event_match = re.match(r"\[MicTest\] (?P<event>\w+)", message)
        assert event_match is not None
        assert event_match.group("event") not in banned
        field_names = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)=", message)
        assert not (set(field_names) & banned)


@pytest.mark.asyncio
async def test_microphone_test_capture_port_stream_open_exception_logs_raw_message_only_as_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.audio.input_host_api = WINDOWS_MME_HOST_API
    controller.settings.audio.input_device = "마이크"
    controller._runtime_logging = RuntimeLoggingSpy()
    raw_message = "bad failure usable near_silence 마이크"
    meter_values: list[float] = []

    def fake_source(*_args, **_kwargs):
        raise RuntimeError(raw_message)

    monkeypatch.setattr(
        controller_module,
        "observe_microphone_test_route",
        lambda **kwargs: _mic_test_route_observation(),
    )
    monkeypatch.setattr(
        controller_module,
        "determine_self_mic_capture_channels",
        lambda *, device_idx, internal_channels: _self_mic_decision(
            device_idx=device_idx,
            preferred_channels=1,
            name="마이크",
        ),
    )
    monkeypatch.setattr(controller_module, "SoundDeviceAudioSource", fake_source)

    await _capture_via_microphone_test_port(
        controller,
        meter_callback=meter_values.append,
    )

    messages = _mic_test_basic_messages(controller)
    assert controller._get_microphone_test_owner().meter_level == 0.0
    assert meter_values == [0.0, 0.0]
    assert any(
        message.startswith("[MicTest] open ")
        and "attempted=True" in message
        and "opened=False" in message
        and "exception_class='RuntimeError'" in message
        and f"exception_message={raw_message!r}" in message
        for message in messages
    )
    assert any(
        message.startswith("[MicTest] end ")
        and "opened=False" in message
        and "exception_class='RuntimeError'" in message
        and f"exception_message={raw_message!r}" in message
        for message in messages
    )
    _assert_mic_test_event_and_field_names_have_no_verdict_labels(messages)


@pytest.mark.asyncio
async def test_microphone_test_capture_port_silent_frames_and_throttles_periodic_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.clock = FakeClock()
    controller._runtime_logging = RuntimeLoggingSpy()
    meter_values: list[float] = []

    class FakeSource:
        actual_sample_rate_hz = 48000
        requested_channels = 1
        opened_channels = 1
        frame_channels = 1
        queue_drop_count = 0
        callback_status_count = 0

        async def frames(self):
            for _ in range(3):
                controller.clock.advance(0.4)
                yield AudioFrameF32(
                    samples=np.zeros((480,), dtype=np.float32),
                    sample_rate_hz=48000,
                    channels=1,
                )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        controller_module,
        "observe_microphone_test_route",
        lambda **kwargs: _mic_test_route_observation(),
    )
    monkeypatch.setattr(
        controller_module,
        "determine_self_mic_capture_channels",
        lambda *, device_idx, internal_channels: _self_mic_decision(
            device_idx=device_idx,
            preferred_channels=1,
            name="마이크",
        ),
    )
    monkeypatch.setattr(controller_module, "SoundDeviceAudioSource", lambda *a, **k: FakeSource())

    await _capture_via_microphone_test_port(
        controller,
        meter_callback=meter_values.append,
    )

    messages = _mic_test_basic_messages(controller)
    level_messages = [message for message in messages if message.startswith("[MicTest] level ")]
    assert len(level_messages) == 1
    assert "rms_db=-120.0" in level_messages[0]
    assert "peak_db=-120.0" in level_messages[0]
    assert "zero_ratio=1.000" in level_messages[0]
    assert "frames=3" in level_messages[0]
    assert controller._get_microphone_test_owner().meter_level == 0.0
    assert meter_values == [0.0, 0.0, 0.0, 0.0, 0.0]
    _assert_mic_test_event_and_field_names_have_no_verdict_labels(messages)


@pytest.mark.asyncio
async def test_microphone_test_capture_port_cancellation_closes_source_and_logs_zero_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller._runtime_logging = RuntimeLoggingSpy()
    frames_started = asyncio.Event()
    close_calls: list[str] = []

    class FakeSource:
        actual_sample_rate_hz = 48000
        requested_channels = 1
        opened_channels = 1
        frame_channels = 1
        queue_drop_count = 0
        callback_status_count = 0

        async def frames(self):
            frames_started.set()
            await asyncio.sleep(3600)
            yield AudioFrameF32(
                samples=np.ones((480,), dtype=np.float32),
                sample_rate_hz=48000,
                channels=1,
            )

        async def close(self) -> None:
            close_calls.append("closed")

    monkeypatch.setattr(
        controller_module,
        "observe_microphone_test_route",
        lambda **kwargs: _mic_test_route_observation(),
    )
    monkeypatch.setattr(
        controller_module,
        "determine_self_mic_capture_channels",
        lambda *, device_idx, internal_channels: _self_mic_decision(
            device_idx=device_idx,
            preferred_channels=1,
            name="마이크",
        ),
    )
    monkeypatch.setattr(controller_module, "SoundDeviceAudioSource", lambda *a, **k: FakeSource())

    task = asyncio.create_task(
        _capture_via_microphone_test_port(
            controller,
            level_log_interval_s=0.01,
        )
    )
    await frames_started.wait()
    await asyncio.sleep(0.03)
    task.cancel()
    results = await asyncio.gather(task, return_exceptions=True)

    messages = _mic_test_basic_messages(controller)
    assert len(results) == 1
    assert isinstance(results[0], asyncio.CancelledError)
    assert close_calls == ["closed"]
    assert any(
        message.startswith("[MicTest] level ")
        and "frames=0" in message
        and "rms_db=-120.0" in message
        for message in messages
    )
    assert any(
        message.startswith("[MicTest] end ")
        and "opened=True" in message
        and "frames_total=0" in message
        and "exception_class='CancelledError'" in message
        for message in messages
    )
    assert controller._get_microphone_test_owner().meter_level == 0.0
    _assert_mic_test_event_and_field_names_have_no_verdict_labels(messages)


@pytest.mark.asyncio
async def test_microphone_test_capture_port_cancellation_after_nonzero_frame_clears_meter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller._runtime_logging = RuntimeLoggingSpy()
    meter_values: list[float] = []
    positive_meter_seen = asyncio.Event()
    close_calls: list[str] = []
    positive_seen = False
    final_clear_after_end: list[bool] = []

    class FakeSource:
        actual_sample_rate_hz = 48000
        requested_channels = 1
        opened_channels = 1
        frame_channels = 1
        queue_drop_count = 0
        callback_status_count = 0

        async def frames(self):
            yield AudioFrameF32(
                samples=np.ones((480,), dtype=np.float32) * np.float32(0.25),
                sample_rate_hz=48000,
                channels=1,
            )
            await asyncio.sleep(3600)

        async def close(self) -> None:
            close_calls.append("closed")

    def record_meter(value: float) -> None:
        nonlocal positive_seen
        if value > 0.0:
            positive_seen = True
            positive_meter_seen.set()
        elif positive_seen:
            final_clear_after_end.append(
                any(
                    message.startswith("[MicTest] end ")
                    for message in _mic_test_basic_messages(controller)
                )
            )
        meter_values.append(value)

    monkeypatch.setattr(
        controller_module,
        "observe_microphone_test_route",
        lambda **kwargs: _mic_test_route_observation(),
    )
    monkeypatch.setattr(
        controller_module,
        "determine_self_mic_capture_channels",
        lambda *, device_idx, internal_channels: _self_mic_decision(
            device_idx=device_idx,
            preferred_channels=1,
            name="마이크",
        ),
    )
    monkeypatch.setattr(controller_module, "SoundDeviceAudioSource", lambda *a, **k: FakeSource())

    task = asyncio.create_task(
        _capture_via_microphone_test_port(
            controller,
            meter_callback=record_meter,
            level_log_interval_s=0.0,
        )
    )
    await positive_meter_seen.wait()
    task.cancel()
    results = await asyncio.gather(task, return_exceptions=True)

    messages = _mic_test_basic_messages(controller)
    assert len(results) == 1
    assert isinstance(results[0], asyncio.CancelledError)
    assert close_calls == ["closed"]
    assert meter_values[0] == 0.0
    assert any(value > 0.0 for value in meter_values)
    assert meter_values[-1] == 0.0
    assert controller._get_microphone_test_owner().meter_level == 0.0
    assert final_clear_after_end == [True]
    assert any(
        message.startswith("[MicTest] end ")
        and "opened=True" in message
        and "frames_total=1" in message
        and "exception_class='CancelledError'" in message
        for message in messages
    )
    _assert_mic_test_event_and_field_names_have_no_verdict_labels(messages)


@pytest.mark.asyncio
async def test_start_microphone_test_clears_pending_self_stt_desire_before_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller._runtime_logging = RuntimeLoggingSpy()
    controller._local_stt_pending_enable_after_install = True
    capture_pending_state: list[bool] = []

    async def fake_capture(self, **_kwargs) -> None:
        capture_pending_state.append(self._local_stt_pending_enable_after_install)

    _patch_microphone_test_capture(monkeypatch, controller, fake_capture)

    started = await controller.start_microphone_test()
    await _wait_until(lambda: _microphone_test_task(controller) is None)

    messages = _mic_test_basic_messages(controller)
    assert started is True
    assert capture_pending_state == [False]
    assert any(
        message.startswith("[MicTest] stt_auto_off ")
        and "requested=True" in message
        and "completed=True" in message
        for message in messages
    )


@pytest.mark.asyncio
async def test_controller_stop_closes_mic_test_runtime_and_continues_after_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller._runtime_logging = RuntimeLoggingSpy()
    runtime = controller._get_microphone_test_owner().runtime
    generation = runtime.begin_direct_capture()
    events: list[str] = []

    class FailingSource:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("mic source close failed")

    source = FailingSource()
    assert runtime.attach_source(source, generation=generation) is True

    async def fake_set_stt_enabled(self, enabled: bool) -> None:
        _ = self, enabled
        events.append("stt-off")

    async def fake_close_vrc_mic_receiver_runtime_for_release(
        self,
        failures: list[Exception],
    ) -> None:
        _ = self, failures
        events.append("vrc-off")

    async def fake_shutdown_overlay_runtime(self, *, preserve_failure_reason: bool) -> None:
        _ = self, preserve_failure_reason
        events.append("overlay-shutdown")

    monkeypatch.setattr(GuiController, "set_stt_enabled", fake_set_stt_enabled)
    monkeypatch.setattr(
        GuiController,
        "_close_vrc_mic_receiver_runtime_for_release",
        fake_close_vrc_mic_receiver_runtime_for_release,
    )
    monkeypatch.setattr(OverlayApplicationOwner, "shutdown", fake_shutdown_overlay_runtime)

    with pytest.raises(RuntimeError, match="mic source close failed"):
        await controller.stop()

    assert events == ["stt-off", "vrc-off", "overlay-shutdown"]
    assert runtime.is_closed is True
    assert runtime.source is source
    assert source.close_calls == 1
    assert await controller.start_microphone_test() is False


@pytest.mark.asyncio
async def test_direct_microphone_capture_rejects_overlap_without_invalidating_active_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller._runtime_logging = RuntimeLoggingSpy()
    runtime = controller._get_microphone_test_owner().runtime
    session_started = asyncio.Event()

    async def active_session(generation: int) -> None:
        session_started.set()
        assert runtime.is_current_generation(generation)
        await asyncio.sleep(3600)

    runtime.start(active_session)
    await session_started.wait()
    active_generation = runtime.generation

    monkeypatch.setattr(
        controller_module,
        "observe_microphone_test_route",
        lambda **kwargs: _mic_test_route_observation(should_attempt_open=False),
    )

    with pytest.raises(RuntimeError, match="active capture"):
        await _capture_via_microphone_test_port(controller)

    assert runtime.generation == active_generation
    assert runtime.is_current_generation(active_generation) is True
    assert controller.microphone_test_active is True

    await controller.stop_microphone_test()


@pytest.mark.asyncio
async def test_direct_microphone_capture_releases_direct_generation_when_initial_meter_cancels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller._runtime_logging = RuntimeLoggingSpy()
    runtime = controller._get_microphone_test_owner().runtime
    route_observed = False

    async def cancelled_meter_callback(_level: float) -> None:
        raise asyncio.CancelledError

    def observe_route(**_kwargs):
        nonlocal route_observed
        route_observed = True
        return _mic_test_route_observation(should_attempt_open=False)

    monkeypatch.setattr(controller_module, "observe_microphone_test_route", observe_route)

    with pytest.raises(asyncio.CancelledError):
        await _capture_via_microphone_test_port(
            controller,
            meter_callback=cancelled_meter_callback,
        )

    assert route_observed is False
    assert runtime.has_active_direct_capture is False
    assert runtime.source is None
    assert runtime.pending_frame_task is None

    async def fake_capture(self, **_kwargs) -> None:
        _ = self

    _patch_microphone_test_capture(monkeypatch, controller, fake_capture)
    assert await controller.start_microphone_test() is True
    await _wait_until(lambda: _microphone_test_task(controller) is None)


@pytest.mark.asyncio
async def test_direct_microphone_capture_releases_direct_generation_when_route_observation_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller._runtime_logging = RuntimeLoggingSpy()
    runtime = controller._get_microphone_test_owner().runtime

    def observe_route(**_kwargs):
        raise RuntimeError("route observation failed")

    monkeypatch.setattr(controller_module, "observe_microphone_test_route", observe_route)

    with pytest.raises(RuntimeError, match="route observation failed"):
        await _capture_via_microphone_test_port(controller)

    assert runtime.has_active_direct_capture is False
    assert runtime.source is None
    assert runtime.pending_frame_task is None

    async def fake_capture(self, **_kwargs) -> None:
        _ = self

    _patch_microphone_test_capture(monkeypatch, controller, fake_capture)
    assert await controller.start_microphone_test() is True
    await _wait_until(lambda: _microphone_test_task(controller) is None)


@pytest.mark.asyncio
async def test_direct_microphone_capture_source_close_failure_is_observable_and_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller._runtime_logging = RuntimeLoggingSpy()

    class FailsOnceSource:
        actual_sample_rate_hz = 48000
        requested_channels = 1
        opened_channels = 1
        frame_channels = 1
        queue_drop_count = 0
        callback_status_count = 0

        def __init__(self) -> None:
            self.close_calls = 0

        async def frames(self):
            if False:
                yield AudioFrameF32(
                    samples=np.zeros((480,), dtype=np.float32),
                    sample_rate_hz=48000,
                    channels=1,
                )

        async def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("mic source close failed")

    source = FailsOnceSource()

    monkeypatch.setattr(
        controller_module,
        "observe_microphone_test_route",
        lambda **kwargs: _mic_test_route_observation(),
    )
    monkeypatch.setattr(
        controller_module,
        "determine_self_mic_capture_channels",
        lambda *, device_idx, internal_channels: _self_mic_decision(
            device_idx=device_idx,
            preferred_channels=1,
        ),
    )
    monkeypatch.setattr(controller_module, "SoundDeviceAudioSource", lambda *a, **k: source)

    with pytest.raises(RuntimeError, match="mic source close failed"):
        await _capture_via_microphone_test_port(controller)

    runtime = controller._get_microphone_test_owner().runtime_if_created
    assert runtime is not None
    assert runtime.source is source
    assert source.close_calls == 1
    assert controller._get_microphone_test_owner().meter_level == 0.0

    await runtime.stop()
    assert runtime.source is None
    assert source.close_calls == 2


@pytest.mark.asyncio
async def test_audio_settings_change_stops_active_microphone_test_and_next_start_uses_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = GuiController(
        page=SimpleNamespace(),
        app=_presentation(SimpleNamespace()),
        config_path=tmp_path / "settings.json",
    )
    controller.settings = AppSettings()
    controller.settings.audio.input_device = "Old Mic"
    controller._runtime_logging = RuntimeLoggingSpy()
    capture_devices: list[str] = []
    capture_cancelled: list[str] = []
    capture_started = [asyncio.Event(), asyncio.Event()]

    async def fake_capture(self, **_kwargs) -> None:
        device_name = self.settings.audio.input_device
        capture_devices.append(device_name)
        capture_started[len(capture_devices) - 1].set()
        try:
            await asyncio.sleep(3600)
        finally:
            capture_cancelled.append(device_name)

    _patch_microphone_test_capture(monkeypatch, controller, fake_capture)

    assert await controller.start_microphone_test() is True
    await capture_started[0].wait()

    updated = copy.deepcopy(controller.settings)
    updated.audio.input_device = "New Mic"
    await controller.apply_settings(updated)

    assert _microphone_test_task(controller) is None
    assert capture_devices == ["Old Mic"]
    assert capture_cancelled == ["Old Mic"]

    assert await controller.start_microphone_test() is True
    await capture_started[1].wait()
    await controller.stop_microphone_test()

    assert capture_devices == ["Old Mic", "New Mic"]


@pytest.mark.asyncio
async def test_audio_settings_change_stops_active_microphone_test_after_in_place_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = GuiController(
        page=SimpleNamespace(),
        app=_presentation(SimpleNamespace()),
        config_path=tmp_path / "settings.json",
    )
    controller.settings = AppSettings()
    controller.settings.audio.input_device = "Old Mic"
    controller._runtime_logging = RuntimeLoggingSpy()
    capture_started = asyncio.Event()
    capture_cancelled: list[str] = []

    async def fake_capture(self, **_kwargs) -> None:
        device_name = self.settings.audio.input_device
        capture_started.set()
        try:
            await asyncio.sleep(3600)
        finally:
            capture_cancelled.append(device_name)

    _patch_microphone_test_capture(monkeypatch, controller, fake_capture)

    assert await controller.start_microphone_test() is True
    await capture_started.wait()

    controller.settings.audio.input_device = "New Mic"
    try:
        await controller.apply_settings(controller.settings)

        assert _microphone_test_task(controller) is None
        assert capture_cancelled == ["Old Mic"]
    finally:
        await controller.stop_microphone_test()


@pytest.mark.asyncio
async def test_controller_stop_cancels_active_microphone_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller._runtime_logging = RuntimeLoggingSpy()
    capture_started = asyncio.Event()
    capture_cancelled: list[str] = []

    async def fake_capture(self, **_kwargs) -> None:
        _ = self
        capture_started.set()
        try:
            await asyncio.sleep(3600)
        finally:
            capture_cancelled.append("cancelled")

    _patch_microphone_test_capture(monkeypatch, controller, fake_capture)

    assert await controller.start_microphone_test() is True
    await capture_started.wait()
    await controller.stop()

    assert capture_cancelled == ["cancelled"]
    assert _microphone_test_task(controller) is None


def test_self_capture_source_adapter_normalizes_wasapi_compatibility_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.audio.input_host_api = WINDOWS_WASAPI_COMPATIBILITY_HOST_API
    controller.settings.audio.input_device = "Compat Mic"
    controller.hub = DummyHub()
    controller._runtime_logging = RuntimeLoggingSpy()
    resolve_calls: list[dict[str, object]] = []
    source_calls: list[dict[str, object]] = []

    class FakeSource:
        async def close(self) -> None:
            return None

    def fake_resolve(*, host_api: str, device: str) -> int:
        resolve_calls.append({"host_api": host_api, "device": device})
        return 7

    def fake_source(*_args, **kwargs) -> FakeSource:
        source_calls.append(dict(kwargs))
        return FakeSource()

    monkeypatch.setattr(audio_source_module, "resolve_sounddevice_input_device", fake_resolve)
    monkeypatch.setattr(
        controller_module,
        "determine_self_mic_capture_channels",
        lambda *, device_idx, internal_channels: _self_mic_decision(
            device_idx=device_idx,
            preferred_channels=1,
        ),
    )
    monkeypatch.setattr(controller_module, "SoundDeviceAudioSource", fake_source)

    _create_self_capture_source_via_adapter_for_test(controller)

    assert resolve_calls == [{"host_api": WINDOWS_WASAPI_HOST_API, "device": "Compat Mic"}]
    assert source_calls[0]["device"] == 7
    assert source_calls[0].get("wasapi_auto_convert") is True
    assert source_calls[0].get("wasapi_exclusive") is False
    assert source_calls[0]["channels"] == 1


def test_self_capture_vad_adapter_wires_self_diagnostics() -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.audio.input_host_api = WINDOWS_WASAPI_COMPATIBILITY_HOST_API
    controller.settings.audio.input_device = "Compat Mic"
    controller.hub = DummyHub()
    controller._runtime_logging = RuntimeLoggingSpy()
    vad_calls: list[dict[str, object]] = []

    def fake_vad_gating(*_args, **kwargs):
        vad_calls.append(dict(kwargs))
        return SimpleNamespace()

    adapter = SelfCaptureVadAdapter(
        model_path_resolver=lambda: Path("vad.onnx"),
        engine_factory=lambda *args, **kwargs: object(),
        gating_factory=fake_vad_gating,
        log_detailed=controller.log_detailed,
        diagnostics_enabled=controller._detailed_audio_diag_enabled,
    )
    controller._vad = adapter(build_self_capture_session_config(controller.settings))

    assert vad_calls[0].get("max_segment_ms") is None
    assert vad_calls[0]["diagnostic_label"] == "self"
    diagnostics_enabled = vad_calls[0]["diagnostics_enabled"]
    assert callable(diagnostics_enabled)
    assert diagnostics_enabled() is True
    diagnostic_callback = vad_calls[0]["diagnostic_event_callback"]
    assert callable(diagnostic_callback)

    diagnostic_callback("[AudioDiag][VAD][self] probe")

    assert (
        logging.INFO,
        "[AudioDiag][VAD][self] probe",
    ) in controller._runtime_logging.detailed_messages
    controller._runtime_logging.set_mode("basic")
    assert diagnostics_enabled() is False


def test_self_capture_source_adapter_requests_two_channel_capture_from_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.audio.input_host_api = WINDOWS_WASAPI_COMPATIBILITY_HOST_API
    controller.settings.audio.input_device = "마이크"
    controller.hub = DummyHub()
    controller._runtime_logging = RuntimeLoggingSpy()
    source_calls: list[dict[str, object]] = []

    class FakeSource:
        actual_sample_rate_hz = 48000
        requested_channels = 2
        opened_channels = 2
        frame_channels = 2

        async def close(self) -> None:
            return None

    def fake_source(*_args, **kwargs) -> FakeSource:
        source_calls.append(dict(kwargs))
        return FakeSource()

    monkeypatch.setattr(
        audio_source_module,
        "resolve_sounddevice_input_device",
        lambda **kwargs: 7,
    )
    monkeypatch.setattr(
        controller_module,
        "determine_self_mic_capture_channels",
        lambda *, device_idx, internal_channels: _self_mic_decision(
            device_idx=device_idx,
            preferred_channels=2,
            name="마이크",
        ),
    )
    monkeypatch.setattr(controller_module, "SoundDeviceAudioSource", fake_source)

    _create_self_capture_source_via_adapter_for_test(controller)

    detailed_logs = [message for _level, message in controller._runtime_logging.detailed_messages]
    basic_logs = [message for _level, message in controller._runtime_logging.basic_messages]

    assert source_calls[0]["device"] == 7
    assert source_calls[0]["channels"] == 2
    assert source_calls[0].get("wasapi_auto_convert") is True
    assert source_calls[0].get("wasapi_exclusive") is False
    assert basic_logs == []
    assert any("Microphone capture format" in item for item in detailed_logs)
    assert any("requested_channels=2" in item for item in detailed_logs)
    assert any("opened_channels=2" in item for item in detailed_logs)
    assert any("frame_channels=2" in item for item in detailed_logs)
    assert any("frame_channels_source='opened_fallback'" in item for item in detailed_logs)
    assert any("metadata_device_name='마이크'" in item for item in detailed_logs)


def test_self_capture_source_adapter_retries_same_device_with_mono_after_two_channel_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.audio.input_host_api = WINDOWS_WASAPI_COMPATIBILITY_HOST_API
    controller.settings.audio.input_device = "Compat Mic"
    controller.hub = DummyHub()
    controller._runtime_logging = RuntimeLoggingSpy()
    resolve_calls: list[dict[str, object]] = []
    source_calls: list[dict[str, object]] = []

    class FakeSource:
        actual_sample_rate_hz = 48000
        requested_channels = 1
        opened_channels = 1
        frame_channels = 1

        async def close(self) -> None:
            return None

    def fake_resolve(*, host_api: str, device: str) -> int:
        resolve_calls.append({"host_api": host_api, "device": device})
        return 7

    def fake_source(*_args, **kwargs) -> FakeSource:
        source_calls.append(dict(kwargs))
        if kwargs["channels"] == 2:
            raise RuntimeError("2ch rejected")
        return FakeSource()

    monkeypatch.setattr(audio_source_module, "resolve_sounddevice_input_device", fake_resolve)
    monkeypatch.setattr(
        controller_module,
        "determine_self_mic_capture_channels",
        lambda *, device_idx, internal_channels: _self_mic_decision(
            device_idx=device_idx,
            preferred_channels=2,
        ),
    )
    monkeypatch.setattr(controller_module, "SoundDeviceAudioSource", fake_source)

    _create_self_capture_source_via_adapter_for_test(controller)

    detailed_logs = [message for _level, message in controller._runtime_logging.detailed_messages]
    basic_logs = [message for _level, message in controller._runtime_logging.basic_messages]

    assert resolve_calls == [{"host_api": WINDOWS_WASAPI_HOST_API, "device": "Compat Mic"}]
    assert [(call["device"], call["channels"]) for call in source_calls] == [(7, 2), (7, 1)]
    assert source_calls[0].get("wasapi_auto_convert") is True
    assert source_calls[1].get("wasapi_auto_convert") is True
    assert basic_logs == []
    assert any("will_retry_mono=True" in item for item in detailed_logs)
    assert any("primary_mono_retry" in item for item in detailed_logs)
    assert any("requested_channels=1" in item for item in detailed_logs)
    assert any("opened_channels=1" in item for item in detailed_logs)


def test_self_capture_source_adapter_recomputes_channels_for_name_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.audio.input_host_api = WINDOWS_WASAPI_COMPATIBILITY_HOST_API
    controller.settings.audio.input_device = "Compat Mic"
    controller.hub = DummyHub()
    controller._runtime_logging = RuntimeLoggingSpy()
    source_calls: list[dict[str, object]] = []

    class FakeSource:
        actual_sample_rate_hz = 48000
        requested_channels = 2
        opened_channels = 2
        frame_channels = 2

        async def close(self) -> None:
            return None

    def fake_resolve(*, host_api: str, device: str) -> int:
        if host_api == WINDOWS_WASAPI_HOST_API:
            return 7
        return 8

    def fake_decision(
        *, device_idx: int | None, internal_channels: int
    ) -> SelfMicCaptureChannelDecision:
        if device_idx == 7:
            return _self_mic_decision(device_idx=7, preferred_channels=1)
        return _self_mic_decision(device_idx=device_idx, preferred_channels=2)

    def fake_source(*_args, **kwargs) -> FakeSource:
        source_calls.append(dict(kwargs))
        if kwargs["device"] == 7:
            raise RuntimeError("primary failed")
        return FakeSource()

    monkeypatch.setattr(audio_source_module, "resolve_sounddevice_input_device", fake_resolve)
    monkeypatch.setattr(
        controller_module,
        "determine_self_mic_capture_channels",
        fake_decision,
    )
    monkeypatch.setattr(controller_module, "SoundDeviceAudioSource", fake_source)

    _create_self_capture_source_via_adapter_for_test(controller)

    assert [(call["device"], call["channels"]) for call in source_calls] == [(7, 1), (8, 2)]
    assert source_calls[1].get("wasapi_auto_convert") is False
    assert source_calls[1].get("wasapi_exclusive") is False


def test_self_capture_source_adapter_uses_default_metadata_for_system_default_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.audio.input_host_api = WINDOWS_WASAPI_COMPATIBILITY_HOST_API
    controller.settings.audio.input_device = ""
    controller.hub = DummyHub()
    controller._runtime_logging = RuntimeLoggingSpy()
    source_calls: list[dict[str, object]] = []

    class FakeSource:
        actual_sample_rate_hz = 48000
        requested_channels = 2
        opened_channels = 2
        frame_channels = 2

        async def close(self) -> None:
            return None

    def fake_source(*_args, **kwargs) -> FakeSource:
        source_calls.append(dict(kwargs))
        if kwargs["device"] == 7:
            raise RuntimeError("primary failed")
        return FakeSource()

    def fake_decision(
        *, device_idx: int | None, internal_channels: int
    ) -> SelfMicCaptureChannelDecision:
        if device_idx is None:
            return _self_mic_decision(
                device_idx=None,
                preferred_channels=2,
                status="default_resolved",
                name="Default Mic",
            )
        return _self_mic_decision(device_idx=device_idx, preferred_channels=1)

    monkeypatch.setattr(
        audio_source_module,
        "resolve_sounddevice_input_device",
        lambda **kwargs: 7,
    )
    monkeypatch.setattr(
        controller_module,
        "determine_self_mic_capture_channels",
        fake_decision,
    )
    monkeypatch.setattr(controller_module, "SoundDeviceAudioSource", fake_source)

    _create_self_capture_source_via_adapter_for_test(controller)

    assert [(call["device"], call["channels"]) for call in source_calls] == [(7, 1), (None, 2)]
    assert source_calls[1].get("wasapi_auto_convert") is False
    assert source_calls[1].get("wasapi_exclusive") is False


def test_self_capture_source_adapter_suppresses_format_diagnostics_in_basic_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.audio.input_host_api = WINDOWS_WASAPI_COMPATIBILITY_HOST_API
    controller.settings.audio.input_device = "마이크"
    controller.hub = DummyHub()
    controller._runtime_logging = RuntimeLoggingSpy(detailed_enabled=False)
    source_calls: list[dict[str, object]] = []

    class FakeSource:
        actual_sample_rate_hz = 48000
        requested_channels = 2
        opened_channels = 2
        frame_channels = 2

        async def close(self) -> None:
            return None

    def fake_source(*_args, **kwargs) -> FakeSource:
        source_calls.append(dict(kwargs))
        return FakeSource()

    monkeypatch.setattr(
        audio_source_module,
        "resolve_sounddevice_input_device",
        lambda **kwargs: 7,
    )
    monkeypatch.setattr(
        controller_module,
        "determine_self_mic_capture_channels",
        lambda *, device_idx, internal_channels: _self_mic_decision(
            device_idx=device_idx,
            preferred_channels=2,
            name="마이크",
        ),
    )
    monkeypatch.setattr(controller_module, "SoundDeviceAudioSource", fake_source)

    _create_self_capture_source_via_adapter_for_test(controller)

    all_messages = [
        message
        for _level, message in (
            controller._runtime_logging.basic_messages
            + controller._runtime_logging.detailed_messages
        )
    ]
    assert source_calls[0]["channels"] == 2
    assert not any("Microphone capture format" in message for message in all_messages)


def test_runtime_logging_writes_non_ascii_detailed_messages_to_utf8_file(tmp_path):
    log_file = tmp_path / "runtime.log"
    stream_handler = logging.NullHandler()
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    root_logger = logging.getLogger(f"test-root-{uuid4()}")
    session_logger = logging.getLogger(f"test-session-{uuid4()}")
    service = SessionRuntimeLoggingService(
        root_logger=root_logger,
        session_logger=session_logger,
        sinks=RuntimeLoggingSinks(
            stream_handler=stream_handler,
            file_handler=file_handler,
            log_file=log_file,
        ),
    )

    try:
        service.set_mode(SessionLoggingMode.DETAILED)
        assert (
            service.emit_detailed("[STT] Microphone capture format: metadata_device_name='마이크'")
            is True
        )
        file_handler.flush()
        assert "마이크" in log_file.read_text(encoding="utf-8")
    finally:
        for logger_obj in (root_logger, session_logger):
            for handler in list(logger_obj.handlers):
                logger_obj.removeHandler(handler)
        file_handler.close()
        stream_handler.close()


def test_controller_runtime_logging_uses_injected_main_sinks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    log_file = tmp_path / "runtime.log"
    sinks = RuntimeLoggingSinks(
        stream_handler=logging.NullHandler(),
        file_handler=logging.NullHandler(),
        log_file=log_file,
    )
    session_log_file = log_file

    class FakeSessionRuntimeLoggingService:
        mode = SessionLoggingMode.BASIC
        log_file = session_log_file

        def set_mode(self, mode) -> None:
            self.mode = SessionLoggingMode(mode)

        def attach_realtime_sink(self, sink) -> None:
            _ = sink

        def detach_realtime_sink(self) -> None:
            return None

        def emit_basic(self, message, *, level=logging.INFO) -> None:
            _ = message, level

        def emit_detailed(self, message, *, level=logging.INFO) -> bool:
            _ = message, level
            return False

        def emit_detailed_lazy(self, build_message, *, level=logging.INFO) -> bool:
            _ = build_message, level
            return False

        def emit_persisted(self, message, *, level=logging.INFO) -> None:
            _ = message, level

        def close(self) -> None:
            return None

    def create_session(**kwargs):
        captured.update(kwargs)
        return FakeSessionRuntimeLoggingService()

    monkeypatch.setattr(
        controller_module,
        "SessionRuntimeLoggingService",
        create_session,
    )
    controller = GuiController(
        page=SimpleNamespace(),
        app=_presentation(SimpleNamespace()),
        config_path=tmp_path / "settings.json",
        runtime_logging_sinks=sinks,
    )

    assert controller.runtime_logging.log_file == log_file
    assert captured["sinks"] is sinks


def test_self_capture_source_adapter_omits_wasapi_flags_from_name_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.audio.input_host_api = WINDOWS_WASAPI_COMPATIBILITY_HOST_API
    controller.settings.audio.input_device = "Compat Mic"
    controller.hub = DummyHub()
    controller._runtime_logging = RuntimeLoggingSpy()
    resolve_calls: list[dict[str, object]] = []
    source_calls: list[dict[str, object]] = []

    class FakeSource:
        async def close(self) -> None:
            return None

    def fake_resolve(*, host_api: str, device: str) -> int:
        resolve_calls.append({"host_api": host_api, "device": device})
        if host_api == WINDOWS_WASAPI_HOST_API:
            return 7
        if host_api == "":
            return 8
        return 99

    def fake_source(*_args, **kwargs) -> FakeSource:
        source_calls.append(dict(kwargs))
        if len(source_calls) == 1:
            raise RuntimeError("first open failed")
        return FakeSource()

    monkeypatch.setattr(audio_source_module, "resolve_sounddevice_input_device", fake_resolve)
    monkeypatch.setattr(
        controller_module,
        "determine_self_mic_capture_channels",
        lambda *, device_idx, internal_channels: _self_mic_decision(
            device_idx=device_idx,
            preferred_channels=1,
        ),
    )
    monkeypatch.setattr(controller_module, "SoundDeviceAudioSource", fake_source)

    _create_self_capture_source_via_adapter_for_test(controller)

    assert resolve_calls == [
        {"host_api": WINDOWS_WASAPI_HOST_API, "device": "Compat Mic"},
        {"host_api": "", "device": "Compat Mic"},
    ]
    assert source_calls[0].get("wasapi_auto_convert") is True
    assert source_calls[1]["device"] == 8
    assert source_calls[1].get("wasapi_auto_convert") is False
    assert source_calls[1].get("wasapi_exclusive") is False
    assert [call["channels"] for call in source_calls] == [1, 1]


def test_self_capture_source_adapter_retries_same_device_without_wasapi_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.audio.input_host_api = WINDOWS_WASAPI_COMPATIBILITY_HOST_API
    controller.settings.audio.input_device = "Compat Mic"
    controller.hub = DummyHub()
    controller._runtime_logging = RuntimeLoggingSpy()
    resolve_calls: list[dict[str, object]] = []
    source_calls: list[dict[str, object]] = []

    class FakeSource:
        async def close(self) -> None:
            return None

    def fake_resolve(*, host_api: str, device: str) -> int:
        resolve_calls.append({"host_api": host_api, "device": device})
        return 7

    def fake_source(*_args, **kwargs) -> FakeSource:
        source_calls.append(dict(kwargs))
        if len(source_calls) == 1:
            raise RuntimeError("first open failed")
        return FakeSource()

    monkeypatch.setattr(audio_source_module, "resolve_sounddevice_input_device", fake_resolve)
    monkeypatch.setattr(
        controller_module,
        "determine_self_mic_capture_channels",
        lambda *, device_idx, internal_channels: _self_mic_decision(
            device_idx=device_idx,
            preferred_channels=1,
        ),
    )
    monkeypatch.setattr(controller_module, "SoundDeviceAudioSource", fake_source)

    _create_self_capture_source_via_adapter_for_test(controller)

    assert resolve_calls == [
        {"host_api": WINDOWS_WASAPI_HOST_API, "device": "Compat Mic"},
        {"host_api": "", "device": "Compat Mic"},
    ]
    assert len(source_calls) == 2
    assert source_calls[0]["device"] == 7
    assert source_calls[0].get("wasapi_auto_convert") is True
    assert source_calls[0].get("wasapi_exclusive") is False
    assert source_calls[1]["device"] == 7
    assert source_calls[1].get("wasapi_auto_convert") is False
    assert source_calls[1].get("wasapi_exclusive") is False
    assert [call["channels"] for call in source_calls] == [1, 1]


def test_self_capture_source_adapter_omits_wasapi_flags_from_system_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.audio.input_host_api = WINDOWS_WASAPI_COMPATIBILITY_HOST_API
    controller.settings.audio.input_device = ""
    controller.hub = DummyHub()
    controller._runtime_logging = RuntimeLoggingSpy()
    resolve_calls: list[dict[str, object]] = []
    source_calls: list[dict[str, object]] = []

    class FakeSource:
        async def close(self) -> None:
            return None

    def fake_resolve(*, host_api: str, device: str) -> int:
        resolve_calls.append({"host_api": host_api, "device": device})
        if host_api == WINDOWS_WASAPI_HOST_API:
            return 7
        return 99

    def fake_source(*_args, **kwargs) -> FakeSource:
        source_calls.append(dict(kwargs))
        if len(source_calls) == 1:
            raise RuntimeError("first open failed")
        return FakeSource()

    monkeypatch.setattr(audio_source_module, "resolve_sounddevice_input_device", fake_resolve)
    monkeypatch.setattr(
        controller_module,
        "determine_self_mic_capture_channels",
        lambda *, device_idx, internal_channels: _self_mic_decision(
            device_idx=device_idx,
            preferred_channels=1,
        ),
    )
    monkeypatch.setattr(controller_module, "SoundDeviceAudioSource", fake_source)

    _create_self_capture_source_via_adapter_for_test(controller)

    assert resolve_calls == [{"host_api": WINDOWS_WASAPI_HOST_API, "device": ""}]
    assert source_calls[0].get("wasapi_auto_convert") is True
    assert source_calls[1]["device"] is None
    assert source_calls[1].get("wasapi_auto_convert") is False
    assert source_calls[1].get("wasapi_exclusive") is False
    assert [call["channels"] for call in source_calls] == [1, 1]


@pytest.mark.asyncio
async def test_controller_stop_closes_vrc_mic_receiver_before_hub_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.hub = DummyHub()
    order: list[str] = []

    class FakeReceiverRuntime:
        async def stop(self, *, strict_runtime_errors: bool = False) -> None:
            _ = strict_runtime_errors
            order.append("receiver_stop")

        async def close(self) -> None:
            order.append("receiver_close")
            controller.receiver = None

    async def fake_set_stt_enabled(self, enabled: bool) -> None:
        _ = (self, enabled)
        order.append("stt")

    async def fake_shutdown_overlay(self, *, preserve_failure_reason: bool) -> None:
        _ = (self, preserve_failure_reason)
        order.append("overlay")

    async def fake_stop_hub_for_release(self, failures: list[Exception]) -> None:
        _ = failures
        order.append("hub")
        self.hub = None

    async def fake_noop(self, *args, **kwargs) -> None:  # noqa: ANN001, ANN002, ANN003
        _ = (self, args, kwargs)

    controller._get_vrc_mic_sync_owner().runtime = FakeReceiverRuntime()
    controller.receiver = object()

    monkeypatch.setattr(GuiController, "set_stt_enabled", fake_set_stt_enabled)
    monkeypatch.setattr(GuiController, "_close_app_oauth_runtime_for_release", fake_noop)
    monkeypatch.setattr(GuiController, "_close_oauth_runtime", fake_noop)
    monkeypatch.setattr(GuiController, "_close_local_asr_provisioning", fake_noop)
    monkeypatch.setattr(GuiController, "_close_microphone_test_runtime_for_release", fake_noop)
    monkeypatch.setattr(OverlayApplicationOwner, "shutdown", fake_shutdown_overlay)
    monkeypatch.setattr(GuiController, "_close_peer_runtime_for_release", fake_noop)
    monkeypatch.setattr(GuiController, "_stop_hub_for_release", fake_stop_hub_for_release)
    monkeypatch.setattr(GuiController, "_replace_managed_openrouter_release_service", fake_noop)
    monkeypatch.setattr(
        GuiController, "_close_app_github_star_prompt_runtime_for_release", fake_noop
    )

    await controller.stop()

    assert order == ["stt", "receiver_close", "overlay", "hub"]


@pytest.mark.asyncio
async def test_controller_stop_uses_bounded_prompt_runtime_close_and_still_stops_hub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    runtime = GithubStarPromptRuntime(cancel_timeout_s=0.01)
    started = asyncio.Event()
    release = asyncio.Event()
    events: list[str] = []
    hub_stopped = asyncio.Event()

    async def suppress_cancellation() -> bool:
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                events.append("prompt_cancelled")
        return True

    task = runtime.start_translation_success_observation(suppress_cancellation())
    await started.wait()
    controller._get_github_star_prompt_owner().runtime = runtime

    class FakeHub:
        async def stop(self) -> None:
            events.append("hub_stop")
            hub_stopped.set()

    async def fake_set_stt_enabled(self, enabled: bool) -> None:
        _ = self
        events.append(f"stt:{enabled}")

    async def fake_shutdown_overlay(self, *, preserve_failure_reason: bool) -> None:
        _ = (self, preserve_failure_reason)
        events.append("overlay_shutdown")

    async def fake_noop(self, *args, **kwargs) -> None:  # noqa: ANN001, ANN002, ANN003
        _ = (self, args, kwargs)

    controller.hub = FakeHub()
    monkeypatch.setattr(GuiController, "set_stt_enabled", fake_set_stt_enabled)
    monkeypatch.setattr(GuiController, "_close_app_oauth_runtime_for_release", fake_noop)
    monkeypatch.setattr(GuiController, "_close_oauth_runtime", fake_noop)
    monkeypatch.setattr(GuiController, "_close_local_asr_provisioning", fake_noop)
    monkeypatch.setattr(GuiController, "_close_microphone_test_runtime_for_release", fake_noop)
    monkeypatch.setattr(OverlayApplicationOwner, "shutdown", fake_shutdown_overlay)
    monkeypatch.setattr(GuiController, "_close_peer_runtime_for_release", fake_noop)
    monkeypatch.setattr(GuiController, "_replace_managed_openrouter_release_service", fake_noop)

    stop_task = asyncio.create_task(controller.stop())
    try:
        await asyncio.wait_for(hub_stopped.wait(), timeout=0.2)

        with pytest.raises(TimeoutError, match="translation_success"):
            await asyncio.wait_for(stop_task, timeout=0.2)

        assert events[:3] == ["prompt_cancelled", "stt:False", "overlay_shutdown"]
        assert events[-1] == "hub_stop"
        assert runtime.translation_success_task is task
    finally:
        release.set()
        await asyncio.wait_for(task, timeout=0.2)
        if not stop_task.done():
            with contextlib.suppress(Exception):
                await asyncio.wait_for(stop_task, timeout=0.2)


@pytest.mark.asyncio
async def test_schedule_github_star_prompt_translation_success_uses_runtime_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.translation.connection = TranslationConnection.OPENROUTER
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = settings
    observed = asyncio.Event()

    async def persist_success() -> bool:
        observed.set()
        return True

    owner = controller._get_github_star_prompt_owner()
    monkeypatch.setattr(
        type(owner),
        "persist_translation_success_observed",
        lambda _self: persist_success(),
    )

    assert controller.schedule_github_star_prompt_translation_success_observed() is True
    runtime = owner.runtime
    assert runtime is not None
    assert runtime.translation_success_task is not None

    await observed.wait()
    await owner.drain_translation_success_observation()

    assert runtime.translation_success_task is None


@pytest.mark.asyncio
async def test_start_initializes_dashboard_and_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.ui.overlay_enabled = False
    settings.provider.llm = LLMProviderName.QWEN
    settings.provider.stt = STTProviderName.QWEN_ASR
    settings.qwen.region = QwenRegion.SINGAPORE
    settings.api_key_verified.alibaba_singapore = True

    dash = DummyDashboard()
    logs = DummyLogsView()
    locale_calls: list[str] = []
    sync_calls: list[str] = []
    bridge_events: list[object] = []
    gpu_discovery_requests: list[str] = []
    hub = DummyHub(llm=object(), stt=object())

    class FakeBridge:
        def __init__(self, **kwargs) -> None:
            assert "app" not in kwargs
            assert {
                "event_queue",
                "runtime_logging",
                "dashboard_destination",
                "history_destination",
                "conversation_destination",
                "get_language_codes",
                "is_translation_enabled",
                "get_stt_state",
                "clear_managed_auth_pending",
                "show_snackbar",
                "on_github_star_translation_success",
                "on_overlay_state_changed",
            }.issubset(kwargs)
            bridge_events.append(("init", kwargs["event_queue"], kwargs.get("runtime_logging")))
            self.started = asyncio.Event()
            self.closed = asyncio.Event()

        async def run(self) -> None:
            bridge_events.append("run")
            self.started.set()
            await self.closed.wait()

        async def wait_started(self) -> None:
            await self.started.wait()

        def report_overlay_state(
            self,
            _state: str,
            *,
            failure_reason: str | None = None,
        ) -> None:
            _ = failure_reason

        def close(self) -> None:
            self.closed.set()

    async def fake_init_pipeline(self) -> None:
        self.hub = hub

    monkeypatch.setattr(GuiController, "_load_or_init_settings", lambda self, path: settings)
    monkeypatch.setattr(
        GuiController,
        "_sync_ui_from_settings",
        lambda self: sync_calls.append("synced"),
    )
    monkeypatch.setattr(GuiController, "_init_pipeline", fake_init_pipeline)
    monkeypatch.setattr(
        presentation_adapter_module,
        "set_ui_locale",
        lambda locale: locale_calls.append(locale),
    )
    monkeypatch.setattr(presentation_adapter_module, "UIEventBridge", FakeBridge)
    app = SimpleNamespace(
        view_dashboard=dash,
        view_logs=logs,
        apply_locale=lambda: locale_calls.append("apply"),
        _on_gpu_discovery_requested=lambda: gpu_discovery_requests.append("discover"),
    )
    controller = _make_controller(app=app)

    assert callable(getattr(controller, "set_runtime_logging_mode", None))
    controller.set_runtime_logging_mode("detailed")

    await controller.start()
    await asyncio.sleep(0)

    assert controller.settings is settings
    assert getattr(controller, "runtime_logging", None) is not None
    assert getattr(controller, "runtime_logging_mode", None) == "basic"
    assert sync_calls == ["synced"]
    assert locale_calls == [settings.ui.locale, "apply"]
    assert logs.attach_calls == 0
    assert dash.stt_needs_key is False
    assert dash.translation_needs_key is False
    assert dash.stt_enabled is False
    assert dash.translation_enabled is False
    assert hub.translation_enabled is False
    assert hub.start_calls == [True]
    assert gpu_discovery_requests == []
    assert bridge_events[0] == ("init", hub.ui_events, controller.runtime_logging)
    assert "run" in bridge_events


@pytest.mark.asyncio
async def test_controller_start_failure_runs_best_effort_cleanup_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    failure = RuntimeError("pipeline failed")
    cleanup_calls: list[str] = []

    async def fail_start(self: GuiController) -> None:
        raise failure

    async def stop(self: GuiController) -> None:
        cleanup_calls.append("stop")

    monkeypatch.setattr(GuiController, "_start_impl", fail_start)
    monkeypatch.setattr(GuiController, "stop", stop)

    with pytest.raises(RuntimeError) as exc_info:
        await controller.start()

    assert exc_info.value is failure
    assert cleanup_calls == ["stop"]


@pytest.mark.asyncio
async def test_start_does_not_auto_restore_transient_overlay_or_peer_toggles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.ui.overlay_enabled = True
    settings.ui.peer_translation_enabled = True
    settings.ui.peer_translation_eula_accepted = True
    settings.provider.peer_stt = STTProviderName.LOCAL_QWEN_GPU

    dash = DummyDashboard()
    logs = DummyLogsView()
    hub = DummyHub(llm=object(), stt=object(), peer_stt=object())
    overlay_calls: list[bool] = []
    gpu_discovery_requests: list[str] = []

    async def fake_init_pipeline(self) -> None:
        self.hub = hub

    async def fake_set_overlay_enabled(self: GuiController, enabled: bool) -> None:
        _ = self
        overlay_calls.append(enabled)

    class FakeBridge:
        def __init__(self, **kwargs) -> None:
            _ = kwargs

        async def run(self) -> None:
            return None

    monkeypatch.setattr(GuiController, "_load_or_init_settings", lambda self, path: settings)
    monkeypatch.setattr(GuiController, "_sync_ui_from_settings", lambda self: None)
    monkeypatch.setattr(GuiController, "_init_pipeline", fake_init_pipeline)
    monkeypatch.setattr(GuiController, "set_overlay_enabled", fake_set_overlay_enabled)
    monkeypatch.setattr(presentation_adapter_module, "set_ui_locale", lambda _locale: None)
    monkeypatch.setattr(presentation_adapter_module, "UIEventBridge", FakeBridge)

    controller = _make_controller(
        app=SimpleNamespace(
            view_dashboard=dash,
            view_logs=logs,
            _on_gpu_discovery_requested=lambda: gpu_discovery_requests.append("discover"),
        )
    )

    await controller.start()
    await asyncio.sleep(0)

    assert overlay_calls == []
    assert hub.peer_translation_enabled is False
    assert gpu_discovery_requests == []


@pytest.mark.asyncio
async def test_set_runtime_logging_mode_emits_audio_snapshot_once_on_basic_to_detailed(
    monkeypatch,
) -> None:
    class FakePage:
        def __init__(self) -> None:
            self.tasks: list[object] = []

        def run_task(self, coro_fn) -> None:
            self.tasks.append(coro_fn)

    sounddevice_lines = ["[AudioDiag][Snapshot][SoundDevice] one"]
    loopback_lines = ["[AudioDiag][Snapshot][Loopback] one"]
    monkeypatch.setattr(
        "puripuly_heart.core.audio.diagnostics.collect_sounddevice_snapshot_lines",
        lambda: sounddevice_lines,
    )
    monkeypatch.setattr(
        "puripuly_heart.core.audio.diagnostics.collect_pyaudiowpatch_snapshot_lines",
        lambda: loopback_lines,
    )

    runtime = RuntimeLoggingSpy(detailed_enabled=False)
    page = FakePage()
    controller = GuiController(
        page=page,
        app=_presentation(SimpleNamespace(), page=page),
        config_path=Path("settings.json"),
    )
    controller._runtime_logging = runtime

    controller.set_runtime_logging_mode("detailed")
    controller.set_runtime_logging_mode("detailed")
    assert len(page.tasks) == 1
    await page.tasks[0]()

    messages = [message for _level, message in runtime.detailed_messages]
    assert messages.count("[AudioDiag][Snapshot][SoundDevice] one") == 1
    assert messages.count("[AudioDiag][Snapshot][Loopback] one") == 1


@pytest.mark.asyncio
async def test_set_runtime_logging_mode_audio_snapshot_run_task_failure_falls_back_to_loop(
    monkeypatch,
) -> None:
    class FailingPage:
        def run_task(self, coro_fn) -> None:
            _ = coro_fn
            raise RuntimeError("run_task rejected")

    sounddevice_lines = ["[AudioDiag][Snapshot][SoundDevice] fallback"]
    loopback_lines = ["[AudioDiag][Snapshot][Loopback] fallback"]
    monkeypatch.setattr(
        "puripuly_heart.core.audio.diagnostics.collect_sounddevice_snapshot_lines",
        lambda: sounddevice_lines,
    )
    monkeypatch.setattr(
        "puripuly_heart.core.audio.diagnostics.collect_pyaudiowpatch_snapshot_lines",
        lambda: loopback_lines,
    )

    runtime = RuntimeLoggingSpy(detailed_enabled=False)
    page = FailingPage()
    controller = GuiController(
        page=page,
        app=_presentation(SimpleNamespace(), page=page),
        config_path=Path("settings.json"),
    )
    controller._runtime_logging = runtime

    controller.set_runtime_logging_mode("detailed")
    await _wait_until(
        lambda: any(
            message == "[AudioDiag][Snapshot][Loopback] fallback"
            for _level, message in runtime.detailed_messages
        ),
        attempts=50,
        delay_s=0.01,
    )

    messages = [message for _level, message in runtime.detailed_messages]
    assert "[AudioDiag][Snapshot][SoundDevice] fallback" in messages
    assert "[AudioDiag][Snapshot][Loopback] fallback" in messages


@pytest.mark.asyncio
async def test_set_runtime_logging_mode_updates_overlay_runtime_contract() -> None:
    class FakePage:
        def __init__(self) -> None:
            self.tasks: list[object] = []

        def run_task(self, coro_fn) -> None:
            self.tasks.append(coro_fn)

    class OverlayManagerSpy:
        def __init__(self) -> None:
            self.modes: list[str] = []

        def set_logging_mode(self, mode: str) -> None:
            self.modes.append(mode)

    page = FakePage()
    controller = GuiController(
        page=page,
        app=_presentation(SimpleNamespace(), page=page),
        config_path=Path("settings.json"),
    )
    controller._runtime_logging = RuntimeLoggingSpy(detailed_enabled=True)
    _attach_overlay_bridge(controller, FakeOverlayBridge(session_token="token"))
    manager = OverlayManagerSpy()
    _attach_overlay_manager(controller, manager)

    controller.set_runtime_logging_mode("detailed")

    assert controller.runtime_logging_mode == "detailed"
    assert manager.modes == ["detailed"]
    assert len(page.tasks) == 1

    await page.tasks[0]()

    assert _overlay_runtime(controller).bridge.runtime_control_messages == ["detailed"]


@pytest.mark.asyncio
async def test_start_keeps_managed_openrouter_dashboard_toggle_available_without_local_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.ui.overlay_enabled = False
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.api_key_verified.openrouter = False

    dash = DummyDashboard()
    logs = DummyLogsView()
    settings_view = DummySettingsView()
    hub = DummyHub(llm=object(), stt=object())

    async def fake_init_pipeline(self) -> None:
        self.hub = hub

    monkeypatch.setattr(GuiController, "_load_or_init_settings", lambda self, path: settings)
    monkeypatch.setattr(GuiController, "_sync_ui_from_settings", lambda self: None)
    monkeypatch.setattr(GuiController, "_init_pipeline", fake_init_pipeline)
    monkeypatch.setattr(presentation_adapter_module, "set_ui_locale", lambda _locale: None)
    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecrets({}),
    )

    async def fail_fetch_key_metadata(_api_key: str):
        raise AssertionError("fetch_key_metadata should not run without a managed key")

    monkeypatch.setattr(
        OpenRouterLLMProvider,
        "fetch_key_metadata",
        staticmethod(fail_fetch_key_metadata),
    )

    class FakeBridge:
        def __init__(self, **kwargs) -> None:
            _ = kwargs

        async def run(self) -> None:
            return None

    monkeypatch.setattr(presentation_adapter_module, "UIEventBridge", FakeBridge)

    controller = _make_controller(
        app=SimpleNamespace(view_dashboard=dash, view_logs=logs, view_settings=settings_view)
    )

    await controller.start()
    await asyncio.sleep(0)

    assert dash.translation_needs_key is False
    assert dash.translation_enabled is False
    assert settings_view.managed_trial_usage_state is None

    assert await controller.refresh_openrouter_usage_after_launch() is False
    assert settings_view.managed_trial_usage_state == {
        "visible": True,
        "remaining_percent": None,
    }
    assert dash.managed_trial_calls == []


@pytest.mark.asyncio
async def test_exhausted_managed_start_does_not_auto_show_founder_letter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.ui.overlay_enabled = False
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    settings.managed_identity.active_managed_credential_ref = "hash_123"

    shown: list[str] = []
    dash = DummyDashboard()
    logs = DummyLogsView()
    settings_view = DummySettingsView()
    hub = DummyHub(llm=object(), stt=object())

    async def fake_init_pipeline(self) -> None:
        self.hub = hub

    monkeypatch.setattr(GuiController, "_load_or_init_settings", lambda self, path: settings)
    monkeypatch.setattr(GuiController, "_sync_ui_from_settings", lambda self: None)
    monkeypatch.setattr(GuiController, "_init_pipeline", fake_init_pipeline)
    monkeypatch.setattr(presentation_adapter_module, "set_ui_locale", lambda _locale: None)
    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecrets({"openrouter_managed_api_key": "managed-key"}),
    )

    async def fake_fetch_key_metadata(_api_key: str):
        return OpenRouterKeyMetadata(
            limit_usd=0.07,
            remaining_usd=0.0007,
            usage_usd=0.0693,
        )

    async def fake_verify_api_key(_api_key: str) -> bool:
        return True

    monkeypatch.setattr(
        OpenRouterLLMProvider,
        "fetch_key_metadata",
        staticmethod(fake_fetch_key_metadata),
    )
    monkeypatch.setattr(
        OpenRouterLLMProvider,
        "verify_api_key",
        staticmethod(fake_verify_api_key),
    )

    class FakeBridge:
        def __init__(self, **kwargs) -> None:
            _ = kwargs

        async def run(self) -> None:
            return None

    monkeypatch.setattr(presentation_adapter_module, "UIEventBridge", FakeBridge)

    controller = _make_controller(
        app=SimpleNamespace(
            view_dashboard=dash,
            view_logs=logs,
            view_settings=settings_view,
            show_founder_letter_dialog=lambda: shown.append("shown"),
        )
    )

    await controller.start()
    await asyncio.sleep(0)
    assert shown == []


class CapturingManagedKeySettingsView(DummySettingsView):
    def __init__(self) -> None:
        super().__init__()
        self.managed_key_state_calls: list[dict[str, object]] = []

    def set_managed_key_state(
        self,
        *,
        visible: bool,
        remaining_percent: int | None = None,
        referral_id: str | None = None,
        pass_status: object | None = None,
    ) -> None:
        self.managed_key_state_calls.append(
            {
                "visible": visible,
                "remaining_percent": remaining_percent,
                "referral_id": referral_id,
                "pass_status": pass_status,
            }
        )


class ManagedStatusRefreshService:
    def __init__(self, result: ManagedOpenRouterStatusRefreshResult) -> None:
        self.result = result
        self.calls = 0

    async def refresh_managed_status(self) -> ManagedOpenRouterStatusRefreshResult:
        self.calls += 1
        return self.result


def _install_managed_usage_metadata_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummySecretsForTrial:
        def get(self, key: str) -> str | None:
            if key == "openrouter_managed_api_key":
                return "managed-key"
            return None

    async def fake_fetch_key_metadata(_api_key: str):
        return OpenRouterKeyMetadata(
            limit_usd=0.07,
            remaining_usd=0.05,
            usage_usd=0.02,
        )

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecretsForTrial(),
    )
    monkeypatch.setattr(
        OpenRouterLLMProvider,
        "fetch_key_metadata",
        staticmethod(fake_fetch_key_metadata),
    )


def _make_managed_usage_controller(
    monkeypatch: pytest.MonkeyPatch,
    *,
    settings_view: CapturingManagedKeySettingsView,
    status_service: ManagedStatusRefreshService,
) -> GuiController:
    _install_managed_usage_metadata_stubs(monkeypatch)
    controller = _make_controller(
        app=SimpleNamespace(view_dashboard=DummyDashboard(), view_settings=settings_view)
    )
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.settings.managed_identity.referral_id = "7KQ9M2"
    controller.hub = DummyHub(llm=object())
    controller._managed_openrouter_release_service = status_service  # noqa: SLF001
    return controller


def test_status_refresh_managed_key_setter_type_error_is_not_masked() -> None:
    class RaisingManagedKeySettingsView(DummySettingsView):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def set_managed_key_state(
            self,
            *,
            visible: bool,
            remaining_percent: int | None = None,
            referral_id: str | None = None,
            pass_status: object | None = None,
        ) -> None:
            _ = visible, remaining_percent, referral_id, pass_status
            self.calls += 1
            raise TypeError("pass_status setter internals failed")

    settings_view = RaisingManagedKeySettingsView()
    controller = _make_controller(
        app=SimpleNamespace(view_dashboard=DummyDashboard(), view_settings=settings_view)
    )
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.settings.managed_identity.referral_id = "7KQ9M2"

    with pytest.raises(TypeError, match="pass_status setter internals failed"):
        controller._get_managed_usage_owner().set_view_state(
            visible=True,
            remaining_percent=71,
            referral_id="7KQ9M2",
        )

    assert settings_view.calls == 1


@pytest.mark.asyncio
async def test_refresh_managed_trial_usage_state_uses_settings_view_live_openrouter_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dash = DummyDashboard()
    settings_view = DummySettingsView()
    controller = _make_controller(
        app=SimpleNamespace(view_dashboard=dash, view_settings=settings_view)
    )
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.hub = DummyHub(llm=object())

    class DummySecretsForTrial:
        def get(self, key: str) -> str | None:
            if key == "openrouter_managed_api_key":
                return "managed-key"
            return None

    async def fake_fetch_key_metadata(_api_key: str):
        return OpenRouterKeyMetadata(
            limit_usd=0.07,
            remaining_usd=0.05,
            usage_usd=0.02,
        )

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecretsForTrial(),
    )
    monkeypatch.setattr(
        OpenRouterLLMProvider,
        "fetch_key_metadata",
        staticmethod(fake_fetch_key_metadata),
    )

    await controller._get_managed_usage_owner().refresh(auto_show_founder_letter=True)

    assert settings_view.managed_trial_usage_state == {
        "visible": True,
        "remaining_percent": 71,
    }
    assert dash.managed_trial_calls == []


@pytest.mark.asyncio
async def test_refresh_managed_trial_usage_state_exposes_refreshed_referral_id_to_managed_key_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dash = DummyDashboard()

    class ManagedKeySettingsView(DummySettingsView):
        def __init__(self) -> None:
            super().__init__()
            self.managed_key_state_calls: list[dict[str, object]] = []

        def set_managed_key_state(
            self,
            *,
            visible: bool,
            remaining_percent: int | None = None,
            referral_id: str | None = None,
            pass_status: object | None = None,
        ) -> None:
            self.managed_key_state_calls.append(
                {
                    "visible": visible,
                    "remaining_percent": remaining_percent,
                    "referral_id": referral_id,
                    "pass_status": pass_status,
                }
            )

    class FakeStatusRefreshService:
        def __init__(self) -> None:
            self.calls = 0

        async def refresh_owned_referral_id_from_status(self) -> str | None:
            self.calls += 1
            return "7KQ9M2"

    settings_view = ManagedKeySettingsView()
    controller = _make_controller(
        app=SimpleNamespace(view_dashboard=dash, view_settings=settings_view)
    )
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.hub = DummyHub(llm=object())
    status_service = FakeStatusRefreshService()
    controller._managed_openrouter_release_service = status_service  # noqa: SLF001

    class DummySecretsForTrial:
        def get(self, key: str) -> str | None:
            if key == "openrouter_managed_api_key":
                return "managed-key"
            return None

    async def fake_fetch_key_metadata(_api_key: str):
        return OpenRouterKeyMetadata(
            limit_usd=0.07,
            remaining_usd=0.05,
            usage_usd=0.02,
        )

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecretsForTrial(),
    )
    monkeypatch.setattr(
        OpenRouterLLMProvider,
        "fetch_key_metadata",
        staticmethod(fake_fetch_key_metadata),
    )

    await controller._get_managed_usage_owner().refresh(auto_show_founder_letter=True)

    await _wait_until(lambda: status_service.calls == 1)
    assert status_service.calls == 1
    await _wait_until(
        lambda: bool(settings_view.managed_key_state_calls)
        and settings_view.managed_key_state_calls[-1]["referral_id"] == "7KQ9M2"
    )
    assert settings_view.managed_key_state_calls[-1] == {
        "visible": True,
        "remaining_percent": 71,
        "referral_id": "7KQ9M2",
        "pass_status": None,
    }


@pytest.mark.asyncio
async def test_refresh_managed_trial_usage_state_preserves_known_referral_id_when_status_omits_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dash = DummyDashboard()

    class ManagedKeySettingsView(DummySettingsView):
        def __init__(self) -> None:
            super().__init__()
            self.managed_key_state_calls: list[dict[str, object]] = []

        def set_managed_key_state(
            self,
            *,
            visible: bool,
            remaining_percent: int | None = None,
            referral_id: str | None = None,
            pass_status: object | None = None,
        ) -> None:
            self.managed_key_state_calls.append(
                {
                    "visible": visible,
                    "remaining_percent": remaining_percent,
                    "referral_id": referral_id,
                    "pass_status": pass_status,
                }
            )

    class OldBrokerStatusRefreshService:
        def __init__(self) -> None:
            self.calls = 0

        async def refresh_owned_referral_id_from_status(self) -> str | None:
            self.calls += 1
            return None

    settings_view = ManagedKeySettingsView()
    controller = _make_controller(
        app=SimpleNamespace(view_dashboard=dash, view_settings=settings_view)
    )
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.settings.managed_identity.referral_id = "7KQ9M2"
    controller.hub = DummyHub(llm=object())
    status_service = OldBrokerStatusRefreshService()
    controller._managed_openrouter_release_service = status_service  # noqa: SLF001

    class DummySecretsForTrial:
        def get(self, key: str) -> str | None:
            if key == "openrouter_managed_api_key":
                return "managed-key"
            return None

    async def fake_fetch_key_metadata(_api_key: str):
        return OpenRouterKeyMetadata(
            limit_usd=0.07,
            remaining_usd=0.05,
            usage_usd=0.02,
        )

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecretsForTrial(),
    )
    monkeypatch.setattr(
        OpenRouterLLMProvider,
        "fetch_key_metadata",
        staticmethod(fake_fetch_key_metadata),
    )

    await controller._get_managed_usage_owner().refresh(auto_show_founder_letter=True)

    await _wait_until(lambda: status_service.calls == 1)
    assert status_service.calls == 1
    assert settings_view.managed_key_state_calls
    assert all(call["referral_id"] == "7KQ9M2" for call in settings_view.managed_key_state_calls)
    assert settings_view.managed_key_state_calls[-1] == {
        "visible": True,
        "remaining_percent": 71,
        "referral_id": "7KQ9M2",
        "pass_status": None,
    }


@pytest.mark.asyncio
async def test_refresh_managed_trial_usage_state_hides_referral_card_when_openrouter_byok_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dash = DummyDashboard()

    class ManagedKeySettingsView(DummySettingsView):
        def __init__(self) -> None:
            super().__init__()
            self.managed_key_state_calls: list[dict[str, object]] = []

        def set_managed_key_state(
            self,
            *,
            visible: bool,
            remaining_percent: int | None = None,
            referral_id: str | None = None,
            pass_status: object | None = None,
        ) -> None:
            self.managed_key_state_calls.append(
                {
                    "visible": visible,
                    "remaining_percent": remaining_percent,
                    "referral_id": referral_id,
                    "pass_status": pass_status,
                }
            )

    settings_view = ManagedKeySettingsView()
    controller = _make_controller(
        app=SimpleNamespace(view_dashboard=dash, view_settings=settings_view)
    )
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.translation.connection = TranslationConnection.OPENROUTER
    controller.settings.translation.connection_history[TranslationModel.GEMMA4.value] = (
        TranslationConnection.OPENROUTER
    )
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    controller.settings.managed_identity.referral_id = "7KQ9M2"
    controller.hub = DummyHub(llm=object())

    await controller._get_managed_usage_owner().refresh(auto_show_founder_letter=True)

    assert settings_view.managed_key_state_calls == [
        {
            "visible": False,
            "remaining_percent": None,
            "referral_id": "7KQ9M2",
            "pass_status": None,
        }
    ]


@pytest.mark.asyncio
async def test_refresh_managed_trial_usage_state_hides_card_when_connection_is_openrouter_even_if_source_is_managed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dash = DummyDashboard()

    class ManagedKeySettingsView(DummySettingsView):
        def __init__(self) -> None:
            super().__init__()
            self.managed_key_state_calls: list[dict[str, object]] = []

        def set_managed_key_state(
            self,
            *,
            visible: bool,
            remaining_percent: int | None = None,
            referral_id: str | None = None,
            pass_status: object | None = None,
        ) -> None:
            self.managed_key_state_calls.append(
                {
                    "visible": visible,
                    "remaining_percent": remaining_percent,
                    "referral_id": referral_id,
                    "pass_status": pass_status,
                }
            )

    class EmptySecrets:
        def get(self, _key: str) -> str | None:
            return None

    settings_view = ManagedKeySettingsView()
    controller = _make_controller(
        app=SimpleNamespace(view_dashboard=dash, view_settings=settings_view)
    )
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.translation.connection = TranslationConnection.OPENROUTER
    controller.settings.translation.connection_history[TranslationModel.GEMMA4.value] = (
        TranslationConnection.OPENROUTER
    )
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.settings.managed_identity.referral_id = "7KQ9M2"
    controller.hub = DummyHub(llm=object())
    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: EmptySecrets(),
    )

    await controller._get_managed_usage_owner().refresh(auto_show_founder_letter=True)

    assert settings_view.managed_key_state_calls == [
        {
            "visible": False,
            "remaining_percent": None,
            "referral_id": "7KQ9M2",
            "pass_status": None,
        }
    ]


@pytest.mark.asyncio
async def test_status_refresh_background_view_update_error_is_logged_not_left_on_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dash = DummyDashboard()
    log_messages: list[tuple[str, int]] = []

    class FailingManagedKeySettingsView(DummySettingsView):
        def set_managed_key_state(
            self,
            *,
            visible: bool,
            remaining_percent: int | None = None,
            referral_id: str | None = None,
            pass_status: object | None = None,
        ) -> None:
            _ = pass_status
            if referral_id == "7KQ9M2":
                raise RuntimeError("managed key repaint failed")
            super().set_managed_trial_usage_state(
                visible=visible,
                remaining_percent=remaining_percent,
            )

    class FakeStatusRefreshService:
        async def refresh_owned_referral_id_from_status(self) -> str | None:
            return "7KQ9M2"

    def fake_log_basic(
        _self: GuiController,
        message: str,
        *,
        level: int = logging.INFO,
    ) -> None:
        log_messages.append((message, level))

    settings_view = FailingManagedKeySettingsView()
    controller = _make_controller(
        app=SimpleNamespace(view_dashboard=dash, view_settings=settings_view)
    )
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.hub = DummyHub(llm=object())
    controller._managed_openrouter_release_service = FakeStatusRefreshService()  # noqa: SLF001

    monkeypatch.setattr(GuiController, "log_basic", fake_log_basic)
    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: DummySecrets({"openrouter_managed_api_key": "managed-key"}),
    )

    async def fake_fetch_key_metadata(_api_key: str):
        return OpenRouterKeyMetadata(
            limit_usd=0.07,
            remaining_usd=0.05,
            usage_usd=0.02,
        )

    monkeypatch.setattr(
        OpenRouterLLMProvider,
        "fetch_key_metadata",
        staticmethod(fake_fetch_key_metadata),
    )

    await controller._get_managed_usage_owner().refresh(auto_show_founder_letter=True)
    await _wait_until(lambda: bool(log_messages))
    managed_usage_owner = controller._managed_usage_owner
    assert managed_usage_owner is not None
    owner = managed_usage_owner.refresh_owner
    await _wait_until(lambda: not owner.active_task_names)

    assert owner.active_task_names == ()
    assert any(
        "Referral ID status refresh failed" in message
        and "managed key repaint failed" in message
        and level == logging.WARNING
        for message, level in log_messages
    )


@pytest.mark.asyncio
async def test_managed_usage_owner_founder_route_does_not_wait_for_slow_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shown: list[str] = []
    dash = DummyDashboard()
    settings_view = DummySettingsView()
    controller = _make_controller(
        app=SimpleNamespace(
            view_dashboard=dash,
            view_settings=settings_view,
            show_founder_letter_dialog=lambda: shown.append("shown"),
        )
    )
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.settings.managed_identity.active_managed_credential_ref = "hash_123"
    controller.hub = DummyHub(llm=object())

    class SlowStatusRefreshService:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.finished = asyncio.Event()

        async def refresh_owned_referral_id_from_status(self) -> str | None:
            self.started.set()
            await self.release.wait()
            self.finished.set()
            return None

    status_service = SlowStatusRefreshService()
    controller._managed_openrouter_release_service = status_service  # noqa: SLF001

    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_a, **_k: DummySecrets({"openrouter_managed_api_key": "managed-key"}),
    )

    async def fake_fetch_key_metadata(_api_key: str):
        return OpenRouterKeyMetadata(
            limit_usd=0.07,
            remaining_usd=0.0007,
            usage_usd=0.0693,
        )

    monkeypatch.setattr(
        OpenRouterLLMProvider,
        "fetch_key_metadata",
        staticmethod(fake_fetch_key_metadata),
    )

    route_task = asyncio.create_task(
        controller._get_managed_usage_owner().should_route_to_founder_letter()
    )
    try:
        await asyncio.wait_for(status_service.started.wait(), timeout=1.0)

        assert route_task.done()
        assert route_task.result() is True
        assert shown == ["shown"]
    finally:
        status_service.release.set()
        await asyncio.wait_for(status_service.finished.wait(), timeout=1.0)
        if not route_task.done():
            route_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await route_task


@pytest.mark.asyncio
async def test_refresh_managed_trial_usage_state_auto_shows_founder_letter_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shown: list[str] = []
    dash = DummyDashboard()
    settings_view = DummySettingsView()
    controller = _make_controller(
        app=SimpleNamespace(
            view_dashboard=dash,
            view_settings=settings_view,
            show_founder_letter_dialog=lambda: shown.append("shown"),
        )
    )
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.hub = DummyHub(llm=object())
    controller.settings.managed_identity.active_managed_credential_ref = "hash_123"
    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_a, **_k: DummySecrets({"openrouter_managed_api_key": "managed-key"}),
    )

    async def fake_fetch_key_metadata(_api_key: str):
        return OpenRouterKeyMetadata(
            limit_usd=0.07,
            remaining_usd=0.0007,
            usage_usd=0.0693,
        )

    monkeypatch.setattr(
        OpenRouterLLMProvider,
        "fetch_key_metadata",
        staticmethod(fake_fetch_key_metadata),
    )

    await controller._get_managed_usage_owner().refresh(auto_show_founder_letter=True)
    await controller._get_managed_usage_owner().refresh(auto_show_founder_letter=True)

    assert shown == ["shown"]


@pytest.mark.asyncio
async def test_set_translation_enabled_reopens_founder_letter_on_exhausted_managed_trans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shown: list[str] = []
    controller = _make_controller(
        app=SimpleNamespace(
            view_dashboard=DummyDashboard(),
            view_settings=DummySettingsView(),
            show_founder_letter_dialog=lambda: shown.append("shown"),
        )
    )
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.hub = DummyHub(llm=object())
    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_a, **_k: DummySecrets({"openrouter_managed_api_key": "managed-key"}),
    )

    async def fake_fetch_key_metadata(_api_key: str):
        return OpenRouterKeyMetadata(
            limit_usd=0.07,
            remaining_usd=0.0007,
            usage_usd=0.0693,
        )

    monkeypatch.setattr(
        OpenRouterLLMProvider,
        "fetch_key_metadata",
        staticmethod(fake_fetch_key_metadata),
    )

    await controller.set_translation_enabled(True)

    assert shown == ["shown"]
    assert controller.hub.translation_enabled is False


@pytest.mark.asyncio
async def test_set_translation_enabled_does_not_route_stale_exhausted_metadata_across_entitlements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shown: list[str] = []
    controller = _make_controller(
        app=SimpleNamespace(
            view_dashboard=DummyDashboard(),
            view_settings=DummySettingsView(),
            show_founder_letter_dialog=lambda: shown.append("shown"),
        )
    )
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.hub = DummyHub(llm=object())
    controller.settings.managed_identity.active_managed_credential_ref = "hash_old"
    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_a, **_k: DummySecrets({"openrouter_managed_api_key": "managed-key"}),
    )

    metadata_calls = 0

    async def fake_fetch_key_metadata(_api_key: str):
        nonlocal metadata_calls
        metadata_calls += 1
        if metadata_calls == 1:
            return OpenRouterKeyMetadata(
                limit_usd=0.07,
                remaining_usd=0.0007,
                usage_usd=0.0693,
            )
        raise RuntimeError("metadata boom")

    monkeypatch.setattr(
        OpenRouterLLMProvider,
        "fetch_key_metadata",
        staticmethod(fake_fetch_key_metadata),
    )

    await controller._get_managed_usage_owner().refresh(auto_show_founder_letter=True)
    assert shown == ["shown"]

    shown.clear()
    controller.settings.managed_identity.active_managed_credential_ref = "hash_new"

    class DummyService:
        def __init__(self) -> None:
            self.calls = 0

        async def prepare_for_translation(self):
            self.calls += 1
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.READY,
                message_key="managed_release.ready",
                api_key="managed-key",
                local_key_available=True,
            )

    service = DummyService()
    controller._managed_openrouter_release_service = service
    monkeypatch.setattr(ManagedUsageOwner, "schedule_usage_refresh", lambda self: None)

    await controller.set_translation_enabled(True)

    assert shown == []
    assert service.calls == 1
    assert controller.hub.translation_enabled is True


@pytest.mark.asyncio
async def test_set_stt_enabled_marks_promo_and_runs_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller._runtime_logging = RuntimeLoggingSpy()
    controller.settings = AppSettings()
    controller.settings.provider.stt = STTProviderName.DEEPGRAM
    controller.hub = DummyHub()
    switch_calls: list[bool] = []

    async def fake_ensure_stt_switch(self) -> None:
        switch_calls.append(self._stt_desired)

    monkeypatch.setattr(GuiController, "_ensure_stt_switch", fake_ensure_stt_switch)

    await controller.set_stt_enabled(True)

    assert controller._stt_desired is True
    assert controller.hub.promo_calls == 1
    assert switch_calls == [True]
    assert controller._runtime_logging.basic_messages == [
        (logging.INFO, "[STT] Toggle request: enabled=True"),
        (logging.INFO, "[STT] Enabled with provider: deepgram"),
    ]
    assert controller._runtime_logging.detailed_messages == [
        (
            logging.INFO,
            "[STT] Toggle detail: desired_before=False overlay_state=off",
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider",
    [STTProviderName.LOCAL_QWEN, STTProviderName.LOCAL_QWEN_GPU],
)
async def test_local_stt_button_stays_starting_until_activation_completes(
    monkeypatch: pytest.MonkeyPatch,
    provider: STTProviderName,
) -> None:
    dashboard = DummyDashboard()
    controller = _make_controller(app=SimpleNamespace(view_dashboard=dashboard))
    controller.settings = AppSettings()
    controller.settings.provider.stt = provider
    controller.hub = DummyHub(stt=object())
    activation_started = asyncio.Event()
    finish_activation = asyncio.Event()

    async def validate_gpu(_self: GuiController) -> bool:
        return True

    async def activate(_self: GuiController) -> None:
        activation_started.set()
        await finish_activation.wait()
        controller._mic_task = SimpleNamespace()

    monkeypatch.setattr(GuiController, "_validate_gpu_activation", validate_gpu)
    monkeypatch.setattr(GuiController, "_ensure_stt_switch", activate)

    task = asyncio.create_task(controller.set_stt_enabled(True))
    await activation_started.wait()

    assert dashboard.stt_starting is True
    assert dashboard.stt_enabled is None

    finish_activation.set()
    await task

    assert dashboard.stt_starting is False
    assert dashboard.stt_enabled is True


@pytest.mark.asyncio
async def test_ensure_stt_switch_delegates_to_owner_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    run_calls: list[str] = []

    async def fake_run_stt_switch(self) -> None:
        _ = self
        run_calls.append("run")

    monkeypatch.setattr(GuiController, "_run_stt_switch", fake_run_stt_switch)

    await controller._ensure_stt_switch()

    assert run_calls == ["run"]


@pytest.mark.asyncio
async def test_run_stt_switch_stop_path_drains_self_ingress_without_touching_peer() -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller._stt_desired = False
    apply_calls: list[dict[str, object]] = []
    peer_calls: list[str] = []

    class FakePeerStt:
        async def close(self) -> None:
            peer_calls.append("close")

    class FakeOwner:
        loop_task = None
        source = None
        cleanup_source = None
        vad = None
        last_cleanup_exception = None

        async def apply_intent(self, config, **kwargs):
            _ = config
            apply_calls.append(kwargs)
            return SelfCaptureSessionSnapshot(
                state=SelfCaptureSessionState.STOPPED,
                provider_status=SelfCaptureProviderStatus.DETACHED,
                desired_active=False,
                effective_active=False,
                generation=1,
                provider_id=None,
                runtime_signature=None,
                failure_reason=None,
                admission_reason=None,
                has_source=False,
                has_vad=False,
                has_loop_task=False,
                cleanup_debt=0,
                closed=False,
            )

    controller.hub = DummyHub(peer_stt=FakePeerStt())
    controller._self_capture_owner = FakeOwner()

    await controller._run_stt_switch()

    assert apply_calls == [
        {
            "enabled": False,
            "restart": False,
            "force_immediate": False,
            "explicit_toggle_off": True,
        }
    ]
    assert controller.hub.drain_self_stt_calls == []
    assert peer_calls == []


@pytest.mark.asyncio
async def test_run_stt_switch_warns_when_hub_missing() -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller._runtime_logging = RuntimeLoggingSpy()
    controller._stt_desired = True
    controller.hub = None

    await controller._run_stt_switch()

    assert controller._runtime_logging.detailed_messages == [
        (logging.WARNING, "[STT] Enable requested before hub is ready")
    ]


@pytest.mark.asyncio
async def test_run_stt_switch_restart_path_drains_and_warms_through_hub_owner() -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller._stt_desired = True
    controller._stt_restart_requested = True
    apply_calls: list[dict[str, object]] = []
    peer_calls: list[str] = []

    class FakePeerStt:
        async def close(self) -> None:
            peer_calls.append("close")

        async def warmup(self) -> None:
            peer_calls.append("warmup")

    class FakeOwner:
        loop_task = None
        source = None
        cleanup_source = None
        vad = None
        last_cleanup_exception = None

        async def apply_intent(self, config, **kwargs):
            _ = config
            apply_calls.append(kwargs)
            return SelfCaptureSessionSnapshot(
                state=SelfCaptureSessionState.RUNNING,
                provider_status=SelfCaptureProviderStatus.READY,
                desired_active=True,
                effective_active=True,
                generation=1,
                provider_id="deepgram",
                runtime_signature=("runtime",),
                failure_reason=None,
                admission_reason=None,
                has_source=True,
                has_vad=True,
                has_loop_task=True,
                cleanup_debt=0,
                closed=False,
            )

    controller.hub = DummyHub(peer_stt=FakePeerStt())
    controller._self_capture_owner = FakeOwner()

    await controller._run_stt_switch()

    assert apply_calls == [
        {
            "enabled": True,
            "restart": True,
            "force_immediate": False,
            "explicit_toggle_off": False,
        }
    ]
    assert controller.hub.drain_self_stt_calls == []
    assert controller.hub.warmup_stt_calls == []
    assert peer_calls == []


@pytest.mark.asyncio
async def test_submit_text_returns_without_hub_and_logs_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    errors: list[str] = []

    await controller.submit_text("hello")

    class FailingHub:
        async def submit_text(self, text: str, *, source: str) -> None:
            _ = (text, source)
            raise RuntimeError("submit boom")

    monkeypatch.setattr(GuiController, "_log_error", lambda self, message: errors.append(message))
    controller.hub = FailingHub()

    await controller.submit_text("hello")

    assert errors == ["Submit failed: submit boom"]


@pytest.mark.asyncio
async def test_apply_settings_replaces_stt_provider_when_source_language_changes_and_applies_locale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.languages.source_language = "ko"
    settings.ui.locale = "ja"
    controller = _make_controller(app=SimpleNamespace(apply_locale=lambda: None))
    controller.settings = settings
    controller.hub = DummyHub()
    controller.hub.source_language = "en"
    saved: list[str] = []
    replace_calls: list[str] = []
    locale_calls: list[str] = []

    async def fake_replace_runtime_stt_provider(self) -> None:
        _ = self
        replace_calls.append("replace")

    monkeypatch.setattr(
        presentation_adapter_module,
        "set_ui_locale",
        lambda locale: locale_calls.append(locale),
    )
    monkeypatch.setattr(presentation_adapter_module, "get_ui_locale", lambda: "en")
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: saved.append("saved") or True,
    )
    monkeypatch.setattr(
        GuiController, "_replace_runtime_stt_provider", fake_replace_runtime_stt_provider
    )
    controller._last_stt_runtime_signature = ("old",)

    await controller.apply_settings(settings)

    assert saved == ["saved"]
    assert replace_calls == ["replace"]
    assert locale_calls == ["ja"]


@pytest.mark.asyncio
async def test_apply_settings_source_language_change_reloads_settings_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.languages.source_language = "ko"
    settings.stt.custom_terms = {"ko": ["Puripuly"], "en": ["Avatar"]}
    settings_view = DummySettingsView()
    controller = _make_controller(
        app=SimpleNamespace(view_settings=settings_view, apply_locale=lambda: None)
    )
    controller.settings = settings
    controller._get_overlay_calibration_application_owner().replace_current(
        settings.overlay.calibration.copy()
    )
    controller.hub = DummyHub()
    controller.hub.source_language = "en"
    replace_calls: list[str] = []

    async def fake_replace_runtime_stt_provider(self) -> None:
        _ = self
        replace_calls.append("replace")

    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    monkeypatch.setattr(
        GuiController, "_replace_runtime_stt_provider", fake_replace_runtime_stt_provider
    )
    controller._last_stt_runtime_signature = ("old",)

    await controller.apply_settings(settings)

    assert replace_calls == ["replace"]
    assert settings_view.calls == [(settings, Path("settings.json"), True)]


@pytest.mark.asyncio
async def test_apply_settings_reloads_settings_view_for_target_only_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.languages.source_language = "en"
    settings.languages.target_language = "ja"
    settings_view = DummySettingsView()
    controller = _make_controller(
        app=SimpleNamespace(view_settings=settings_view, apply_locale=lambda: None)
    )
    controller.settings = settings
    controller._get_overlay_calibration_application_owner().replace_current(
        settings.overlay.calibration.copy()
    )
    controller.hub = DummyHub()
    controller.hub.source_language = "en"
    controller.hub.target_language = "ko"

    async def fake_refresh_peer_stt_runtime(self) -> None:
        _ = self

    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    monkeypatch.setattr(GuiController, "_refresh_peer_stt_runtime", fake_refresh_peer_stt_runtime)
    controller._last_self_stt_runtime_signature = build_self_stt_runtime_signature(settings)
    controller._last_peer_stt_runtime_signature = _peer_runtime_signature(controller, settings)

    await controller.apply_settings(settings)

    assert settings_view.calls == [(settings, Path("settings.json"), True)]


@pytest.mark.asyncio
async def test_apply_settings_target_only_change_clears_self_language_runtime_state_without_restarting_stt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = settings
    controller.hub = DummyHub()
    controller.hub.source_language = settings.languages.source_language
    controller.hub.target_language = settings.languages.target_language
    controller._last_self_stt_runtime_signature = build_self_stt_runtime_signature(settings)
    controller._last_peer_stt_runtime_signature = _peer_runtime_signature(controller, settings)
    controller._last_peer_translation_enabled = settings.ui.peer_translation_enabled
    controller._last_vrc_mic_sync_enabled = settings.osc.vrc_mic_intercept

    replace_calls: list[str] = []
    refresh_peer_calls: list[str] = []

    updated = copy.deepcopy(settings)
    updated.languages.target_language = "ja"

    async def fake_replace_runtime_stt_provider(self) -> None:
        _ = self
        replace_calls.append("replace")

    async def fake_refresh_peer_stt_runtime(self) -> None:
        _ = self
        refresh_peer_calls.append("peer")

    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    monkeypatch.setattr(
        GuiController,
        "_clear_local_stt_pending_enable_if_provider_switched_away",
        lambda self: None,
    )
    monkeypatch.setattr(
        GuiController, "_replace_runtime_stt_provider", fake_replace_runtime_stt_provider
    )
    monkeypatch.setattr(GuiController, "_refresh_peer_stt_runtime", fake_refresh_peer_stt_runtime)

    await controller.apply_settings(updated)

    assert controller.hub.clear_language_runtime_state_calls == ["self"]
    assert replace_calls == []
    assert refresh_peer_calls == []
    assert controller.hub.target_language == "ja"


@pytest.mark.asyncio
async def test_apply_settings_self_target_change_clears_peer_runtime_when_peer_target_follows_self(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.languages.peer_target_language = ""
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = settings
    controller.hub = DummyHub()
    controller.hub.source_language = settings.languages.source_language
    controller.hub.target_language = settings.languages.target_language
    controller.hub.peer_source_language = settings.languages.peer_source_language
    controller.hub.peer_target_language = settings.languages.peer_target_language
    controller._last_self_stt_runtime_signature = build_self_stt_runtime_signature(settings)
    controller._last_peer_stt_runtime_signature = _peer_runtime_signature(controller, settings)
    controller._last_peer_translation_enabled = settings.ui.peer_translation_enabled
    controller._last_vrc_mic_sync_enabled = settings.osc.vrc_mic_intercept

    refresh_peer_calls: list[str] = []

    updated = copy.deepcopy(settings)
    updated.languages.target_language = "ja"

    async def fake_replace_runtime_stt_provider(self) -> None:
        raise AssertionError("self STT runtime should not restart for target-only change")

    async def fake_refresh_peer_stt_runtime(self) -> None:
        _ = self
        refresh_peer_calls.append("peer")

    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    monkeypatch.setattr(
        GuiController,
        "_clear_local_stt_pending_enable_if_provider_switched_away",
        lambda self: None,
    )
    monkeypatch.setattr(
        GuiController, "_replace_runtime_stt_provider", fake_replace_runtime_stt_provider
    )
    monkeypatch.setattr(GuiController, "_refresh_peer_stt_runtime", fake_refresh_peer_stt_runtime)

    await controller.apply_settings(updated)

    assert controller.hub.clear_language_runtime_state_calls == ["self", "peer"]
    assert refresh_peer_calls == []


@pytest.mark.asyncio
async def test_apply_settings_self_source_change_clears_peer_runtime_when_peer_source_follows_self(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.languages.peer_source_language = ""
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = settings
    controller.hub = DummyHub()
    controller.hub.source_language = settings.languages.source_language
    controller.hub.target_language = settings.languages.target_language
    controller.hub.peer_source_language = settings.languages.peer_source_language
    controller.hub.peer_target_language = settings.languages.peer_target_language
    controller._last_self_stt_runtime_signature = build_self_stt_runtime_signature(settings)
    controller._last_peer_stt_runtime_signature = _peer_runtime_signature(controller, settings)
    controller._last_peer_translation_enabled = settings.ui.peer_translation_enabled
    controller._last_vrc_mic_sync_enabled = settings.osc.vrc_mic_intercept

    replace_calls: list[str] = []
    refresh_peer_calls: list[str] = []

    updated = copy.deepcopy(settings)
    updated.languages.source_language = "ja"

    async def fake_replace_runtime_stt_provider(self) -> None:
        _ = self
        replace_calls.append("replace")

    async def fake_refresh_peer_stt_runtime(self) -> None:
        _ = self
        refresh_peer_calls.append("peer")

    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    monkeypatch.setattr(
        GuiController,
        "_clear_local_stt_pending_enable_if_provider_switched_away",
        lambda self: None,
    )
    monkeypatch.setattr(
        GuiController, "_replace_runtime_stt_provider", fake_replace_runtime_stt_provider
    )
    monkeypatch.setattr(GuiController, "_refresh_peer_stt_runtime", fake_refresh_peer_stt_runtime)

    await controller.apply_settings(updated)

    assert controller.hub.clear_language_runtime_state_calls == ["self", "peer"]
    assert replace_calls == ["replace"]
    assert refresh_peer_calls == ["peer"]


@pytest.mark.asyncio
async def test_apply_settings_logs_and_continues_when_language_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.languages.peer_target_language = "fr"
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = settings
    controller.hub = DummyHub()
    controller.hub.source_language = settings.languages.source_language
    controller.hub.target_language = settings.languages.target_language
    controller.hub.peer_source_language = settings.languages.peer_source_language
    controller.hub.peer_target_language = settings.languages.peer_target_language
    controller.hub.clear_language_runtime_state_errors["self"] = RuntimeError("cleanup boom")
    controller._last_self_stt_runtime_signature = build_self_stt_runtime_signature(settings)
    controller._last_peer_stt_runtime_signature = _peer_runtime_signature(controller, settings)
    controller._last_peer_translation_enabled = settings.ui.peer_translation_enabled
    controller._last_vrc_mic_sync_enabled = settings.osc.vrc_mic_intercept

    errors: list[str] = []

    updated = copy.deepcopy(settings)
    updated.languages.target_language = "ja"

    async def fake_replace_runtime_stt_provider(self) -> None:
        raise AssertionError("self STT runtime should not restart for target-only change")

    async def fake_refresh_peer_stt_runtime(self) -> None:
        raise AssertionError("peer runtime should not refresh for explicit peer target")

    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    monkeypatch.setattr(
        GuiController,
        "_clear_local_stt_pending_enable_if_provider_switched_away",
        lambda self: None,
    )
    monkeypatch.setattr(
        GuiController, "_replace_runtime_stt_provider", fake_replace_runtime_stt_provider
    )
    monkeypatch.setattr(GuiController, "_refresh_peer_stt_runtime", fake_refresh_peer_stt_runtime)
    monkeypatch.setattr(GuiController, "_log_error", lambda self, message: errors.append(message))

    await controller.apply_settings(updated)

    assert controller.hub.clear_language_runtime_state_calls == ["self"]
    assert controller.hub.target_language == "ja"
    assert _settings_result(controller) is not None
    assert (
        _settings_result(controller).status
        == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
    )
    assert not any("cleanup boom" in message for message in errors)


@pytest.mark.asyncio
async def test_order22_language_runtime_clear_failure_degrades_without_raw_log_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller._runtime_logging = RuntimeLoggingSpy()
    controller.settings = AppSettings()
    controller.hub = DummyHub(stt=object())
    controller.hub.source_language = controller.settings.languages.source_language
    controller.hub.target_language = controller.settings.languages.target_language
    controller._last_self_stt_runtime_signature = build_self_stt_runtime_signature(
        controller.settings
    )
    controller._last_peer_stt_runtime_signature = _peer_runtime_signature(
        controller, controller.settings
    )
    controller._last_peer_translation_enabled = controller.settings.ui.peer_translation_enabled
    controller._last_vrc_mic_sync_enabled = controller.settings.osc.vrc_mic_intercept
    raw_failure_text = "language cleanup failed secret-token-must-not-leak"
    controller.hub.clear_language_runtime_state_errors["self"] = RuntimeError(raw_failure_text)
    pending = copy.deepcopy(controller.settings)
    pending.languages.target_language = "ja"

    _patch_settings_save(monkeypatch, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(GuiController, "_sync_clipboard_watcher", lambda self: asyncio.sleep(0))
    monkeypatch.setattr(
        GuiController,
        "_clear_local_stt_pending_enable_if_provider_switched_away",
        lambda self: None,
    )

    await controller.apply_settings(pending)

    result = _settings_result(controller)
    assert result is not None
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
    assert result.diagnostics == messages.ErrorDiagnostics(
        component="gui_controller",
        operation="apply_stt_language_audio_runtime",
        code="stt_language_audio_runtime_apply_exception",
        category=messages.DIAGNOSTIC_CATEGORY_LIFECYCLE,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"surface": "stt_language_audio"},
    )
    logged_text = "\n".join(
        message
        for _level, message in (
            controller._runtime_logging.basic_messages
            + controller._runtime_logging.detailed_messages
        )
    )
    assert raw_failure_text not in repr(result)
    assert raw_failure_text not in logged_text
    assert "language runtime state" in logged_text


@pytest.mark.asyncio
async def test_order24_apply_settings_routes_ui_prompt_clipboard_state_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(apply_locale=lambda: None))
    controller.settings = AppSettings()
    controller.settings_mutation_service = RecordingSettingsMutationService()
    direct_saves: list[str] = []

    async def noop_sync_clipboard(_self: GuiController) -> None:
        return None

    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: direct_saves.append("save") or True,
    )
    monkeypatch.setattr(GuiController, "_sync_clipboard_watcher", noop_sync_clipboard)

    updated = copy.deepcopy(controller.settings)
    updated.secrets.backend = SecretsBackend.ENCRYPTED_FILE
    updated.secrets.encrypted_file_path = "secure-secrets.json"
    updated.ui.locale = "ja"
    updated.ui.peer_translation_eula_accepted = True
    updated.ui.integrated_context_enabled = False
    updated.ui.integrated_context_bootstrapped = True
    updated.ui.clipboard_auto_translate_enabled = True
    updated.ui.github_star_prompt_clicked = True
    updated.ui.github_star_prompt_last_shown_at = "2026-06-08T00:00:00Z"
    updated.ui.github_star_prompt_show_count = 2
    updated.ui.github_star_prompt_translation_success_observed = True
    updated.ui.github_star_prompt_eligible_launch_count = 3
    updated.system_prompt = "custom translation style"

    await controller.apply_settings(updated)

    service = controller.settings_mutation_service
    assert service is not None
    assert len(service.requests) == 1
    request = service.requests[0]
    assert request.reason == settings_mutation.SETTINGS_MUTATION_SURFACE_UI_PROMPT_CLIPBOARD_STATE
    assert dict(request.values) == {
        "secrets.backend": SecretsBackend.ENCRYPTED_FILE,
        "secrets.encrypted_file_path": "secure-secrets.json",
        "ui.locale": "ja",
        "ui.peer_translation_eula_accepted": True,
        "ui.integrated_context_bootstrapped": True,
        "ui.clipboard_auto_translate_enabled": True,
        "ui.github_star_prompt_clicked": True,
        "ui.github_star_prompt_last_shown_at": "2026-06-08T00:00:00Z",
        "ui.github_star_prompt_show_count": 2,
        "ui.github_star_prompt_translation_success_observed": True,
        "ui.github_star_prompt_eligible_launch_count": 3,
        "system_prompt": "custom translation style",
    }
    assert "ui.overlay_enabled" not in request.values
    assert "ui.peer_translation_enabled" not in request.values
    assert controller.settings is not None
    assert controller.settings.ui.locale == "ja"
    assert controller.settings.system_prompt == "custom translation style"
    assert direct_saves == []


@pytest.mark.asyncio
async def test_order24_locale_runtime_failure_degrades_without_raw_exception_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_failure_text = "locale failed secret-token-must-not-leak"
    applied_locales: list[str] = []
    saved_settings: list[AppSettings] = []

    class LocaleApp:
        def apply_locale(self) -> None:
            raise RuntimeError(raw_failure_text)

    controller = _make_controller(app=LocaleApp())
    controller._runtime_logging = RuntimeLoggingSpy()
    controller.settings = AppSettings()
    pending = copy.deepcopy(controller.settings)
    pending.ui.locale = "ja"

    def record_saved_settings(_path, incoming: AppSettings) -> None:
        saved_settings.append(copy.deepcopy(incoming))

    _patch_settings_save(monkeypatch, record_saved_settings)
    monkeypatch.setattr(
        presentation_adapter_module,
        "set_ui_locale",
        lambda locale: applied_locales.append(locale),
    )
    monkeypatch.setattr(GuiController, "_sync_clipboard_watcher", lambda self: asyncio.sleep(0))

    await controller.apply_settings(pending)

    result = _settings_result(controller)
    assert result is not None
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
    assert result.diagnostics == messages.ErrorDiagnostics(
        component="gui_controller",
        operation="apply_ui_prompt_clipboard_state_runtime",
        code="ui_prompt_clipboard_state_runtime_apply_exception",
        category=messages.DIAGNOSTIC_CATEGORY_LIFECYCLE,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"surface": "ui_prompt_clipboard_state"},
    )
    assert [settings.ui.locale for settings in saved_settings] == ["ja"]
    assert applied_locales == ["ja"]
    assert controller.settings is not None
    assert controller.settings.ui.locale == "ja"
    logged_text = "\n".join(
        message
        for _level, message in (
            controller._runtime_logging.basic_messages
            + controller._runtime_logging.detailed_messages
        )
    )
    assert raw_failure_text not in repr(result)
    assert raw_failure_text not in logged_text


@pytest.mark.asyncio
async def test_order24_clipboard_start_failure_degrades_without_raw_exception_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_failure_text = "clipboard start failed secret-token-must-not-leak"
    saved_settings: list[AppSettings] = []

    class FailingStartClipboardWatcher:
        def start(self) -> None:
            raise RuntimeError(raw_failure_text)

        def stop(self) -> None:
            return None

    controller = _make_controller(app=SimpleNamespace(apply_locale=lambda: None))
    controller._runtime_logging = RuntimeLoggingSpy()
    controller.settings = AppSettings()
    pending = copy.deepcopy(controller.settings)
    pending.ui.clipboard_auto_translate_enabled = True

    def record_saved_settings(_path, incoming: AppSettings) -> None:
        saved_settings.append(copy.deepcopy(incoming))

    _patch_settings_save(monkeypatch, record_saved_settings)
    monkeypatch.setattr(controller_module.sys, "platform", "win32")
    monkeypatch.setattr(
        controller_module,
        "create_clipboard_watcher",
        lambda _on_text: FailingStartClipboardWatcher(),
    )

    await controller.apply_settings(pending)

    result = _settings_result(controller)
    assert result is not None
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
    assert result.diagnostics == messages.ErrorDiagnostics(
        component="gui_controller",
        operation="apply_ui_prompt_clipboard_state_runtime",
        code="ui_prompt_clipboard_state_runtime_apply_exception",
        category=messages.DIAGNOSTIC_CATEGORY_LIFECYCLE,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"surface": "ui_prompt_clipboard_state"},
    )
    assert [settings.ui.clipboard_auto_translate_enabled for settings in saved_settings] == [True]
    assert controller.settings is not None
    assert controller.settings.ui.clipboard_auto_translate_enabled is True
    runtime = controller._get_clipboard_auto_translation_owner().runtime
    assert runtime is not None
    assert runtime.watcher is None
    logged_text = "\n".join(
        message
        for _level, message in (
            controller._runtime_logging.basic_messages
            + controller._runtime_logging.detailed_messages
        )
    )
    assert raw_failure_text not in repr(result)
    assert raw_failure_text not in logged_text


@pytest.mark.asyncio
async def test_order24_clipboard_stop_failure_degrades_without_raw_exception_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_failure_text = "clipboard stop failed secret-token-must-not-leak"
    saved_settings: list[AppSettings] = []

    class FailingStopClipboardWatcher:
        def start(self) -> None:
            return None

        def stop(self) -> None:
            raise RuntimeError(raw_failure_text)

    controller = _make_controller(app=SimpleNamespace(apply_locale=lambda: None))
    controller._runtime_logging = RuntimeLoggingSpy()
    controller.settings = AppSettings()
    controller.settings.ui.clipboard_auto_translate_enabled = True
    monkeypatch.setattr(controller_module.sys, "platform", "win32")
    monkeypatch.setattr(
        controller_module,
        "create_clipboard_watcher",
        lambda _on_text: FailingStopClipboardWatcher(),
    )
    await controller._sync_clipboard_watcher()
    pending = copy.deepcopy(controller.settings)
    pending.ui.clipboard_auto_translate_enabled = False

    def record_saved_settings(_path, incoming: AppSettings) -> None:
        saved_settings.append(copy.deepcopy(incoming))

    _patch_settings_save(monkeypatch, record_saved_settings)

    await controller.apply_settings(pending)

    result = _settings_result(controller)
    assert result is not None
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
    assert result.diagnostics == messages.ErrorDiagnostics(
        component="gui_controller",
        operation="apply_ui_prompt_clipboard_state_runtime",
        code="ui_prompt_clipboard_state_runtime_apply_exception",
        category=messages.DIAGNOSTIC_CATEGORY_LIFECYCLE,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"surface": "ui_prompt_clipboard_state"},
    )
    assert [settings.ui.clipboard_auto_translate_enabled for settings in saved_settings] == [False]
    assert controller.settings is not None
    assert controller.settings.ui.clipboard_auto_translate_enabled is False
    runtime = controller._get_clipboard_auto_translation_owner().runtime
    assert runtime is not None
    assert runtime.watcher is None
    logged_text = "\n".join(
        message
        for _level, message in (
            controller._runtime_logging.basic_messages
            + controller._runtime_logging.detailed_messages
        )
    )
    assert raw_failure_text not in repr(result)
    assert raw_failure_text not in logged_text


@pytest.mark.asyncio
async def test_order24_runtime_only_overlay_and_peer_toggles_are_not_service_routed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings_mutation_service = RecordingSettingsMutationService()
    direct_saves: list[str] = []

    async def noop_sync_clipboard(_self: GuiController) -> None:
        return None

    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: direct_saves.append("save") or True,
    )
    monkeypatch.setattr(GuiController, "_sync_clipboard_watcher", noop_sync_clipboard)
    monkeypatch.setattr(
        GuiController, "set_overlay_enabled", lambda self, enabled: asyncio.sleep(0)
    )

    updated = copy.deepcopy(controller.settings)
    updated.ui.overlay_enabled = True
    updated.ui.peer_translation_enabled = True

    await controller.apply_settings(updated)

    service = controller.settings_mutation_service
    assert service is not None
    assert service.requests == []
    assert direct_saves == ["save"]
    assert controller.settings is updated


@pytest.mark.asyncio
async def test_github_star_prompt_click_persistence_routes_through_order24_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings_mutation_service = RecordingSettingsMutationService()

    def fail_direct_save(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("github star prompt state should use order24 service")

    _patch_settings_save(monkeypatch, fail_direct_save)

    assert await controller.persist_github_star_prompt_clicked() is True

    service = controller.settings_mutation_service
    assert service is not None
    assert len(service.requests) == 1
    assert service.requests[0].reason == (
        settings_mutation.SETTINGS_MUTATION_SURFACE_UI_PROMPT_CLIPBOARD_STATE
    )
    assert service.requests[0].values == {"ui.github_star_prompt_clicked": True}
    assert controller.settings.ui.github_star_prompt_clicked is True


@pytest.mark.asyncio
async def test_first_live_settings_view_order24_mutation_uses_synced_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings_view = DummySettingsView()
    controller = _make_controller(app=SimpleNamespace(view_settings=settings_view))
    controller.settings = settings
    controller.settings_mutation_service = RecordingSettingsMutationService()
    direct_saves: list[str] = []

    async def noop_sync_clipboard(_self: GuiController) -> None:
        return None

    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: direct_saves.append("save") or True,
    )
    monkeypatch.setattr(GuiController, "_sync_clipboard_watcher", noop_sync_clipboard)

    controller._sync_ui_from_settings()
    settings.ui.clipboard_auto_translate_enabled = True
    pending = copy.deepcopy(settings)

    await controller.apply_settings(pending)

    service = controller.settings_mutation_service
    assert service is not None
    assert len(service.requests) == 1
    assert service.requests[0].reason == (
        settings_mutation.SETTINGS_MUTATION_SURFACE_UI_PROMPT_CLIPBOARD_STATE
    )
    assert service.requests[0].values == {"ui.clipboard_auto_translate_enabled": True}
    assert direct_saves == []


@pytest.mark.asyncio
async def test_apply_settings_reload_updates_overlay_calibration_baseline_without_clobbering_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.overlay.calibration.distance = 0.9
    settings_view = DummySettingsView()
    controller = _make_controller(
        app=SimpleNamespace(view_settings=settings_view, apply_locale=lambda: None)
    )
    controller.settings = settings
    controller._get_overlay_calibration_application_owner().replace_current(
        settings.overlay.calibration.copy()
    )
    controller.hub = DummyHub()
    controller.hub.source_language = settings.languages.source_language
    controller.hub.target_language = settings.languages.target_language

    async def fake_replace_runtime_stt_provider(self) -> None:
        _ = self

    async def fake_refresh_peer_stt_runtime(self) -> None:
        _ = self

    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    monkeypatch.setattr(
        GuiController, "_replace_runtime_stt_provider", fake_replace_runtime_stt_provider
    )
    monkeypatch.setattr(GuiController, "_refresh_peer_stt_runtime", fake_refresh_peer_stt_runtime)

    controller.begin_overlay_calibration()
    controller.set_overlay_calibration_field("distance", 1.2)

    updated = AppSettings()
    updated.languages.source_language = "ja"
    updated.overlay.calibration.distance = 0.8

    await controller.apply_settings(updated)

    assert settings_view.calls == [(updated, Path("settings.json"), True)]
    assert controller.overlay_calibration.distance == 0.8
    assert controller.begin_overlay_calibration().distance == 1.2

    canceled = controller.cancel_overlay_calibration()

    assert canceled.distance == 0.8


@pytest.mark.asyncio
async def test_apply_settings_restarts_stt_and_reports_locale_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.QWEN
    settings.stt.low_latency_mode = True
    settings.ui.locale = "ko"
    settings.osc.vrc_mic_intercept = True

    errors: list[str] = []
    rebuild_llm_calls: list[str] = []
    receiver_calls: list[bool] = []
    switch_calls: list[str] = []
    locale_calls: list[str] = []

    raw_failure_text = "locale boom"
    app = SimpleNamespace(
        apply_locale=lambda: (_ for _ in ()).throw(RuntimeError(raw_failure_text))
    )
    controller = _make_controller(app=app)
    controller.settings = settings
    controller.hub = DummyHub()
    controller.hub.source_language = settings.languages.source_language
    controller.hub.target_language = settings.languages.target_language
    controller.hub.system_prompt = settings.system_prompt
    controller.hub.low_latency_mode = False
    controller.hub.low_latency_merge_gap_ms = settings.stt.low_latency_merge_gap_ms
    controller.hub.low_latency_spec_retry_max = settings.stt.low_latency_spec_retry_max
    controller.hub.hangover_s = 1.1
    controller._last_stt_runtime_signature = ("old",)
    controller._mic_task = object()
    controller._stt_desired = True

    async def fake_rebuild_llm_provider(self) -> None:
        rebuild_llm_calls.append("rebuild_llm")

    async def fake_configure_vrc_mic_receiver(self, *, enabled: bool) -> None:
        receiver_calls.append(enabled)

    async def fake_rebuild_stt_provider(self) -> None:
        _ = self
        switch_calls.append("rebuild_stt")

    async def fake_ensure_stt_switch(self) -> None:
        _ = self
        switch_calls.append("switch")

    async def fake_replace_runtime_stt_provider(
        self,
        *,
        smooth_local: bool = False,
    ) -> None:
        _ = self, smooth_local
        switch_calls.append("replace_stt")

    monkeypatch.setattr(
        presentation_adapter_module,
        "set_ui_locale",
        lambda locale: locale_calls.append(locale),
    )
    monkeypatch.setattr(presentation_adapter_module, "get_ui_locale", lambda: "en")
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    monkeypatch.setattr(
        provider_runtime_apply_module.LlmProviderRebuildOwner, "rebuild", fake_rebuild_llm_provider
    )
    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        fake_configure_vrc_mic_receiver,
    )
    monkeypatch.setattr(GuiController, "_rebuild_stt_provider", fake_rebuild_stt_provider)
    monkeypatch.setattr(GuiController, "_ensure_stt_switch", fake_ensure_stt_switch)
    monkeypatch.setattr(
        GuiController,
        "_replace_runtime_stt_provider",
        fake_replace_runtime_stt_provider,
    )
    monkeypatch.setattr(GuiController, "_log_error", lambda self, message: errors.append(message))

    await controller.apply_settings(settings)

    assert rebuild_llm_calls == []
    assert receiver_calls == [True]
    assert controller._stt_restart_requested is False
    assert switch_calls == ["replace_stt"]
    assert locale_calls == ["ko"]
    assert controller.hub.low_latency_mode is True
    assert "Failed to apply locale" in errors
    assert raw_failure_text not in "\n".join(errors)


@pytest.mark.asyncio
async def test_apply_settings_rebuilds_stt_provider_when_runtime_changes_while_stt_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.DEEPGRAM

    dash = DummyDashboard()
    controller = _make_controller(app=SimpleNamespace(view_dashboard=dash))
    controller.settings = settings

    prepared_configs: list[object] = []

    class FakeOwner:
        loop_task = None
        source = None
        cleanup_source = None
        vad = None
        last_cleanup_exception = None

        async def prepare_provider(self, config):
            prepared_configs.append(config)
            return SelfCaptureSessionSnapshot(
                state=SelfCaptureSessionState.STOPPED,
                provider_status=SelfCaptureProviderStatus.READY,
                desired_active=False,
                effective_active=False,
                generation=1,
                provider_id=config.provider_id,
                runtime_signature=config.runtime_signature,
                failure_reason=None,
                admission_reason=None,
                has_source=False,
                has_vad=False,
                has_loop_task=False,
                cleanup_debt=0,
                closed=False,
            )

    controller.hub = DummyHub(stt=object())
    controller.hub.source_language = settings.languages.source_language
    controller.hub.target_language = settings.languages.target_language
    controller.hub.system_prompt = settings.system_prompt
    controller.hub.low_latency_mode = settings.stt.low_latency_mode
    controller.hub.low_latency_merge_gap_ms = settings.stt.low_latency_merge_gap_ms
    controller.hub.low_latency_spec_retry_max = settings.stt.low_latency_spec_retry_max
    controller.hub.hangover_s = 1.1
    controller._last_stt_runtime_signature = build_self_stt_runtime_signature(settings)
    controller._stt_desired = False
    controller._mic_task = None
    controller._self_capture_owner = FakeOwner()

    settings.stt.custom_vocabulary_enabled = True
    settings.stt.custom_terms = {"ko": ["Puripuly"]}

    async def fake_configure_vrc_mic_receiver(self, *, enabled: bool) -> None:
        _ = (self, enabled)

    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        fake_configure_vrc_mic_receiver,
    )
    await controller.apply_settings(settings)

    request = controller._self_capture_provider_request(prepared_configs[-1], False)
    assert request.config.source_language == "ko"
    assert request.config.custom_vocabulary_enabled is True
    assert request.config.custom_terms == {"ko": ("Puripuly",)}
    assert controller.hub.replace_stt_request_calls == []
    assert dash.stt_needs_key is False


@pytest.mark.asyncio
async def test_apply_settings_replaces_running_stt_provider_for_custom_vocabulary_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.DEEPGRAM

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = settings
    controller.hub = DummyHub(stt=object())
    controller.hub.source_language = settings.languages.source_language
    controller.hub.target_language = settings.languages.target_language
    controller.hub.system_prompt = settings.system_prompt
    controller.hub.low_latency_mode = settings.stt.low_latency_mode
    controller.hub.low_latency_merge_gap_ms = settings.stt.low_latency_merge_gap_ms
    controller.hub.low_latency_spec_retry_max = settings.stt.low_latency_spec_retry_max
    controller.hub.hangover_s = 1.1
    controller._last_stt_runtime_signature = build_self_stt_runtime_signature(settings)
    controller._stt_desired = True

    settings.stt.custom_vocabulary_enabled = True
    settings.stt.custom_terms = {"ko": ["Puripuly", "VRChat"]}

    apply_calls: list[dict[str, object]] = []

    class FakeOwner:
        loop_task = object()
        source = object()
        cleanup_source = None
        vad = object()
        last_cleanup_exception = None

        async def apply_intent(self, config, **kwargs):
            apply_calls.append(kwargs)
            return SelfCaptureSessionSnapshot(
                state=SelfCaptureSessionState.RUNNING,
                provider_status=SelfCaptureProviderStatus.READY,
                desired_active=True,
                effective_active=True,
                generation=1,
                provider_id=config.provider_id,
                runtime_signature=config.runtime_signature,
                failure_reason=None,
                admission_reason=None,
                has_source=True,
                has_vad=True,
                has_loop_task=True,
                cleanup_debt=0,
                closed=False,
            )

    async def fake_configure_vrc_mic_receiver(self, *, enabled: bool) -> None:
        _ = (self, enabled)

    controller._self_capture_owner = FakeOwner()
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        fake_configure_vrc_mic_receiver,
    )

    await controller.apply_settings(settings)

    assert apply_calls == [
        {
            "enabled": True,
            "restart": False,
            "explicit_toggle_off": False,
        }
    ]
    assert controller._stt_restart_requested is False


@pytest.mark.asyncio
async def test_apply_settings_does_not_restart_stt_for_qwen_custom_vocabulary_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.QWEN_ASR

    replace_calls: list[str] = []

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = settings
    controller.hub = DummyHub()
    controller.hub.source_language = settings.languages.source_language
    controller.hub.target_language = settings.languages.target_language
    controller.hub.system_prompt = settings.system_prompt
    controller.hub.low_latency_mode = settings.stt.low_latency_mode
    controller.hub.low_latency_merge_gap_ms = settings.stt.low_latency_merge_gap_ms
    controller.hub.low_latency_spec_retry_max = settings.stt.low_latency_spec_retry_max
    controller.hub.hangover_s = 1.1
    controller._last_stt_runtime_signature = build_self_stt_runtime_signature(settings)
    controller._stt_desired = True
    controller._mic_task = object()

    settings.stt.custom_vocabulary_enabled = True
    settings.stt.custom_terms = {"ko": ["Puripuly", "VRChat"]}

    async def fake_configure_vrc_mic_receiver(self, *, enabled: bool) -> None:
        _ = (self, enabled)

    async def fake_replace_runtime_stt_provider(self) -> None:
        _ = self
        replace_calls.append("replace")

    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        fake_configure_vrc_mic_receiver,
    )
    monkeypatch.setattr(
        GuiController, "_replace_runtime_stt_provider", fake_replace_runtime_stt_provider
    )

    await controller.apply_settings(settings)

    assert controller._stt_restart_requested is False
    assert replace_calls == []


@pytest.mark.asyncio
async def test_apply_settings_restarts_stt_for_local_qwen_custom_vocabulary_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.LOCAL_QWEN

    replace_calls: list[str] = []

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = settings
    controller.hub = DummyHub()
    controller.hub.source_language = settings.languages.source_language
    controller.hub.target_language = settings.languages.target_language
    controller.hub.system_prompt = settings.system_prompt
    controller.hub.low_latency_mode = settings.stt.low_latency_mode
    controller.hub.low_latency_merge_gap_ms = settings.stt.low_latency_merge_gap_ms
    controller.hub.low_latency_spec_retry_max = settings.stt.low_latency_spec_retry_max
    controller.hub.hangover_s = 1.1
    controller._last_stt_runtime_signature = build_self_stt_runtime_signature(settings)
    controller._stt_desired = True
    controller._mic_task = object()

    settings.stt.custom_vocabulary_enabled = True
    settings.stt.custom_terms = {"ko": ["Puripuly", "VRChat"]}

    async def fake_configure_vrc_mic_receiver(self, *, enabled: bool) -> None:
        _ = (self, enabled)

    async def fake_replace_runtime_stt_provider(self) -> None:
        _ = self
        replace_calls.append("replace")

    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        fake_configure_vrc_mic_receiver,
    )
    monkeypatch.setattr(
        GuiController, "_replace_runtime_stt_provider", fake_replace_runtime_stt_provider
    )

    await controller.apply_settings(settings)

    assert controller._stt_restart_requested is False
    assert replace_calls == ["replace"]


@pytest.mark.asyncio
async def test_apply_settings_skips_vrc_sync_when_setting_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.QWEN_ASR

    receiver_calls: list[bool] = []

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = settings
    controller.hub = SimpleNamespace(
        source_language=settings.languages.source_language,
        target_language=settings.languages.target_language,
        system_prompt=settings.system_prompt,
        low_latency_mode=settings.stt.low_latency_mode,
        low_latency_merge_gap_ms=settings.stt.low_latency_merge_gap_ms,
        low_latency_spec_retry_max=settings.stt.low_latency_spec_retry_max,
        hangover_s=1.1,
        peer_stt=None,
    )
    controller._last_stt_runtime_signature = build_self_stt_runtime_signature(settings)
    controller._last_vrc_mic_sync_enabled = settings.osc.vrc_mic_intercept

    settings.stt.custom_vocabulary_enabled = True
    settings.stt.custom_terms = {"ko": ["Puripuly"]}

    async def fake_configure_vrc_mic_receiver(self, *, enabled: bool) -> None:
        _ = self
        receiver_calls.append(enabled)

    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )
    monkeypatch.setattr(
        GuiController,
        "_configure_vrc_mic_receiver",
        fake_configure_vrc_mic_receiver,
    )

    with caplog.at_level(logging.INFO, logger=controller_module.logger.name):
        await controller.apply_settings(settings)

    assert receiver_calls == []
    assert all("VRC mic sync enabled" not in record.message for record in caplog.records)


@pytest.mark.parametrize(
    ("provider", "result", "expected"),
    [
        ("deepgram", True, (True, "Verification successful")),
        ("deepgram", False, (False, "Verification failed (check logs/console for details)")),
        ("openrouter", True, (True, "Verification successful")),
        ("soniox", True, (True, "Verification successful")),
    ],
)
@pytest.mark.asyncio
async def test_verify_api_key_success_and_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    result: bool,
    expected: tuple[bool, str],
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()

    async def fake_verify(_key: str) -> bool:
        return result

    monkeypatch.setattr(DeepgramRealtimeSTTBackend, "verify_api_key", staticmethod(fake_verify))
    monkeypatch.setattr(OpenRouterLLMProvider, "verify_api_key", staticmethod(fake_verify))
    monkeypatch.setattr(SonioxRealtimeSTTBackend, "verify_api_key", staticmethod(fake_verify))

    outcome = await controller.verify_api_key(provider, "secret")

    assert outcome == expected


@pytest.mark.asyncio
async def test_verify_api_key_routes_alibaba_singapore_to_qwen_owner() -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    calls: list[tuple[str, str, str | None, bool]] = []

    async def fake_verify(
        key: str,
        *,
        base_url: str,
        model: str | None,
        low_latency: bool,
    ) -> bool:
        calls.append((key, base_url, model, low_latency))
        return True

    controller.provider_verifier = SimpleNamespace(
        verify_qwen_llm_api_key=fake_verify,
    )

    outcome = await controller.verify_api_key("alibaba_singapore", "secret")

    assert outcome == (True, "Verification successful")
    assert calls == [
        (
            "secret",
            "https://dashscope-intl.aliyuncs.com/api/v1",
            controller.settings.qwen.llm_model.value,
            True,
        )
    ]


@pytest.mark.asyncio
async def test_create_openrouter_pkce_client_uses_openrouter_documented_localhost_port() -> None:
    controller = _make_controller(
        app=SimpleNamespace(view_dashboard=DummyDashboard(), view_settings=DummySettingsView())
    )

    client = controller._create_openrouter_pkce_client()
    session = client.build_session()

    assert client.callback_origin == "http://localhost:3000"
    assert "callback_url=http%3A%2F%2Flocalhost%3A3000%2Fcallback" in session.authorization_url


@pytest.mark.asyncio
async def test_connect_openrouter_via_pkce_rejects_unverified_exchanged_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(
        app=SimpleNamespace(view_dashboard=DummyDashboard(), view_settings=DummySettingsView())
    )
    controller.settings = AppSettings()
    previous_settings = copy.deepcopy(controller.settings)
    target_settings = copy.deepcopy(controller.settings)
    target_settings.provider.llm = LLMProviderName.OPENROUTER
    target_settings.openrouter.selection_alias = OpenRouterSelectionAlias.GEMMA4_BYOK
    target_settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    target_settings.openrouter.llm_model = OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
    store = DummySecrets({"openrouter_api_key": "legacy-key"})

    class DummyPKCEClient:
        async def run_desktop_flow(self) -> OpenRouterPKCEExchangeResult:
            return OpenRouterPKCEExchangeResult(api_key="sk-or-v1-user", user_id="user_123")

    monkeypatch.setattr(
        GuiController,
        "_create_openrouter_pkce_client",
        lambda self: DummyPKCEClient(),
    )
    monkeypatch.setattr(controller_module, "create_secret_store", lambda *_a, **_k: store)
    verify_calls: list[str] = []

    async def fake_verify_openrouter_api_key(api_key: str) -> bool:
        verify_calls.append(api_key)
        return False

    monkeypatch.setattr(
        OpenRouterLLMProvider,
        "verify_api_key",
        fake_verify_openrouter_api_key,
    )
    applied: list[AppSettings] = []

    async def fake_apply_providers(
        self,
        settings: AppSettings | None = None,
        *,
        force_rebuild_llm: bool = False,
    ) -> None:
        _ = self
        _ = force_rebuild_llm
        assert settings is not None
        applied.append(copy.deepcopy(settings))

    monkeypatch.setattr(GuiController, "apply_providers", fake_apply_providers)

    ok = await controller.connect_openrouter_via_pkce(
        target_settings=target_settings,
        launch_source="settings",
    )

    assert ok is False
    assert verify_calls == ["sk-or-v1-user"]
    assert applied == []
    assert controller.settings == previous_settings
    assert store.get("openrouter_api_key") == "legacy-key"
    assert store.set_calls == []
    assert store.delete_calls == []


@pytest.mark.asyncio
async def test_connect_openrouter_via_pkce_rebuilds_llm_when_signature_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dashboard = DummyDashboard()
    dashboard.translation_needs_key = True
    controller = _make_controller(
        app=SimpleNamespace(view_dashboard=dashboard, view_settings=DummySettingsView())
    )
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selection_alias = OpenRouterSelectionAlias.GEMMA4_BYOK
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    controller.settings.openrouter.llm_model = OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
    controller.hub = DummyHub(llm=None)
    controller._sync_signature_caches(controller.settings)
    target_settings = copy.deepcopy(controller.settings)
    store = DummySecrets({})

    class DummyPKCEClient:
        async def run_desktop_flow(self) -> OpenRouterPKCEExchangeResult:
            return OpenRouterPKCEExchangeResult(api_key="sk-or-v1-user", user_id="user_123")

    monkeypatch.setattr(
        GuiController,
        "_create_openrouter_pkce_client",
        lambda self: DummyPKCEClient(),
    )
    monkeypatch.setattr(controller_module, "create_secret_store", lambda *_a, **_k: store)

    class DummyManagedReleaseService:
        async def close(self) -> None:
            return None

    def fake_create_managed_release_service(self, *, secrets):
        _ = (self, secrets)
        return DummyManagedReleaseService()

    monkeypatch.setattr(
        GuiController,
        "_create_managed_openrouter_release_service",
        fake_create_managed_release_service,
    )
    _patch_settings_save(monkeypatch, lambda *_args, **_kwargs: None)

    async def fake_verify_openrouter_api_key(_api_key: str) -> bool:
        return True

    monkeypatch.setattr(
        OpenRouterLLMProvider,
        "verify_api_key",
        fake_verify_openrouter_api_key,
    )
    created_llm: list[str] = []

    def fake_create_llm_provider(*_args, **_kwargs):
        created_llm.append(store.get("openrouter_api_key") or "")
        return "rebuilt-llm"

    monkeypatch.setattr(controller_module, "create_llm_provider", fake_create_llm_provider)

    async def fake_refresh_managed_trial_usage_state_best_effort(self) -> None:
        _ = self

    monkeypatch.setattr(
        GuiController,
        "_refresh_managed_trial_usage_state_best_effort",
        fake_refresh_managed_trial_usage_state_best_effort,
    )

    ok = await controller.connect_openrouter_via_pkce(
        target_settings=target_settings,
        launch_source="settings",
    )

    assert ok is True
    assert created_llm == ["sk-or-v1-user"]
    assert controller.hub.llm == "rebuilt-llm"
    assert controller.settings.api_key_verified.openrouter is True
    assert dashboard.translation_needs_key is False


def test_reopen_openrouter_pkce_authorization_url_delegates_to_active_client() -> None:
    reopen_calls: list[str] = []
    controller = _make_controller(
        app=SimpleNamespace(view_dashboard=DummyDashboard(), view_settings=DummySettingsView())
    )
    controller._get_openrouter_pkce_flow_owner().active_client = SimpleNamespace(
        reopen_authorization_url=lambda: reopen_calls.append("reopen") or True
    )

    assert controller.reopen_openrouter_pkce_authorization_url() is True
    assert reopen_calls == ["reopen"]


@pytest.mark.asyncio
async def test_connect_openrouter_via_pkce_leaves_settings_unchanged_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(
        app=SimpleNamespace(view_dashboard=DummyDashboard(), view_settings=DummySettingsView())
    )
    controller.settings = AppSettings()
    controller.settings.openrouter.selection_alias = OpenRouterSelectionAlias.GEMMA4_MANAGED
    target_settings = copy.deepcopy(controller.settings)
    target_settings.openrouter.selection_alias = OpenRouterSelectionAlias.GEMMA4_BYOK
    target_settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    target_settings.openrouter.llm_model = OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
    store = DummySecrets({})

    class DummyPKCEClient:
        async def run_desktop_flow(self) -> OpenRouterPKCEExchangeResult:
            raise RuntimeError("browser failed")

    monkeypatch.setattr(
        GuiController,
        "_create_openrouter_pkce_client",
        lambda self: DummyPKCEClient(),
    )
    monkeypatch.setattr(controller_module, "create_secret_store", lambda *_a, **_k: store)

    ok = await controller.connect_openrouter_via_pkce(
        target_settings=target_settings,
        launch_source="settings",
    )

    assert ok is False
    assert controller.settings.openrouter.selection_alias == OpenRouterSelectionAlias.GEMMA4_MANAGED
    assert store.set_calls == []


@pytest.mark.asyncio
async def test_connect_openrouter_via_pkce_reopens_letter_context_on_letter_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shown: list[str] = []
    controller = _make_controller(
        app=SimpleNamespace(
            view_dashboard=DummyDashboard(),
            view_settings=DummySettingsView(),
            show_founder_letter_dialog=lambda: shown.append("shown"),
        )
    )
    controller.settings = AppSettings()
    target_settings = copy.deepcopy(controller.settings)
    target_settings.openrouter.selection_alias = OpenRouterSelectionAlias.GEMMA4_BYOK
    target_settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    target_settings.openrouter.llm_model = OpenRouterLLMModel.GEMMA_4_26B_A4B_IT

    class DummyPKCEClient:
        async def run_desktop_flow(self) -> OpenRouterPKCEExchangeResult:
            raise RuntimeError("browser failed")

    monkeypatch.setattr(
        GuiController,
        "_create_openrouter_pkce_client",
        lambda self: DummyPKCEClient(),
    )

    ok = await controller.connect_openrouter_via_pkce(
        target_settings=target_settings,
        launch_source="letter",
    )

    assert ok is False
    assert shown == ["shown"]


@pytest.mark.asyncio
async def test_connect_openrouter_via_pkce_returns_degraded_on_runtime_apply_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(
        app=SimpleNamespace(view_dashboard=DummyDashboard(), view_settings=DummySettingsView())
    )
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selection_alias = OpenRouterSelectionAlias.GEMMA4_MANAGED
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    target_settings = copy.deepcopy(controller.settings)
    target_settings.provider.llm = LLMProviderName.OPENROUTER
    target_settings.openrouter.selection_alias = OpenRouterSelectionAlias.GEMMA4_BYOK
    target_settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    target_settings.openrouter.llm_model = OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
    store = DummySecrets({"openrouter_api_key": "legacy-key"})

    class DummyPKCEClient:
        async def run_desktop_flow(self) -> OpenRouterPKCEExchangeResult:
            return OpenRouterPKCEExchangeResult(api_key="sk-or-v1-user", user_id="user_123")

    monkeypatch.setattr(
        GuiController,
        "_create_openrouter_pkce_client",
        lambda self: DummyPKCEClient(),
    )
    monkeypatch.setattr(controller_module, "create_secret_store", lambda *_a, **_k: store)
    _patch_settings_save(monkeypatch, lambda *_a, **_k: None)

    async def fake_verify_openrouter_api_key(_api_key: str) -> bool:
        return True

    monkeypatch.setattr(
        OpenRouterLLMProvider,
        "verify_api_key",
        fake_verify_openrouter_api_key,
    )

    async def fake_apply_provider_runtime_plan(
        _owner,
        settings: AppSettings,
        plan: object,
    ) -> None:
        _ = plan
        controller.settings = copy.deepcopy(settings)
        raise RuntimeError("apply failed after mutation")

    monkeypatch.setattr(
        provider_runtime_apply_module.ProviderRuntimeOwner,
        "apply",
        fake_apply_provider_runtime_plan,
    )

    ok = await controller.connect_openrouter_via_pkce(
        target_settings=target_settings,
        launch_source="settings",
    )

    assert ok is True
    assert store.get("openrouter_api_key") == "sk-or-v1-user"
    assert store.set_calls == [("openrouter_api_key", "sk-or-v1-user")]
    assert store.delete_calls == []
    assert controller.settings.api_key_verified.openrouter is True
    assert (
        _settings_result(controller) is not None
        and _settings_result(controller).status
        == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
    )


@pytest.mark.asyncio
async def test_connect_openrouter_via_pkce_restores_secret_on_settings_commit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(
        app=SimpleNamespace(view_dashboard=DummyDashboard(), view_settings=DummySettingsView())
    )
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selection_alias = OpenRouterSelectionAlias.GEMMA4_MANAGED
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    target_settings = copy.deepcopy(controller.settings)
    target_settings.openrouter.selection_alias = OpenRouterSelectionAlias.GEMMA4_BYOK
    target_settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    target_settings.openrouter.llm_model = OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
    store = DummySecrets({"openrouter_api_key": "legacy-key"})

    class DummyPKCEClient:
        async def run_desktop_flow(self) -> OpenRouterPKCEExchangeResult:
            return OpenRouterPKCEExchangeResult(api_key="sk-or-v1-user", user_id="user_123")

    monkeypatch.setattr(
        GuiController,
        "_create_openrouter_pkce_client",
        lambda self: DummyPKCEClient(),
    )
    monkeypatch.setattr(controller_module, "create_secret_store", lambda *_a, **_k: store)

    def raise_on_save(*_a, **_k):
        raise RuntimeError("disk write failed")

    _patch_settings_save(monkeypatch, raise_on_save)

    async def fake_verify_openrouter_api_key(_api_key: str) -> bool:
        return True

    monkeypatch.setattr(
        OpenRouterLLMProvider,
        "verify_api_key",
        fake_verify_openrouter_api_key,
    )

    ok = await controller.connect_openrouter_via_pkce(
        target_settings=target_settings,
        launch_source="settings",
    )

    assert ok is False
    assert store.get("openrouter_api_key") == "legacy-key"
    assert store.set_calls == [
        ("openrouter_api_key", "sk-or-v1-user"),
        ("openrouter_api_key", "legacy-key"),
    ]
    assert store.delete_calls == []
    assert (
        _settings_result(controller) is not None
        and _settings_result(controller).status
        == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED_SECRET_RESTORED
    )


def test_merge_settings_tab_apply_with_current_languages_preserves_all_language_fields() -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    controller.settings.openrouter.selection_alias = OpenRouterSelectionAlias.GEMMA4_BYOK
    controller.settings.translation.fallback = TranslationFallbackSettings(
        enabled=True,
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.OPENROUTER,
    )
    controller.settings.languages.source_language = "fr"
    controller.settings.languages.target_language = "de"
    controller.settings.languages.peer_source_language = "ja"
    controller.settings.languages.peer_target_language = "it"
    controller.settings.languages.recent_source_languages = ["fr", "ko"]
    controller.settings.languages.recent_target_languages = ["de", "en"]
    controller.settings.stt.low_latency_mode = True
    controller.settings.stt.low_latency_vad_hangover_ms = 650
    controller.settings.desktop_audio.vad_hangover_ms = 950
    controller.hub = DummyHub()
    controller.hub.source_language = "es"
    controller.hub.target_language = "pt"
    controller.hub.peer_source_language = "zh-CN"
    controller.hub.peer_target_language = "nl"

    pending = AppSettings()
    pending.languages.source_language = "ko"
    pending.languages.target_language = "en"
    pending.languages.peer_source_language = ""
    pending.languages.peer_target_language = "ja"
    pending.provider.stt = STTProviderName.SONIOX
    pending.provider.peer_stt = STTProviderName.SONIOX
    pending.provider.llm = LLMProviderName.OPENROUTER
    pending.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    pending.openrouter.selection_alias = OpenRouterSelectionAlias.QWEN35_FLASH_MANAGED
    pending.translation.fallback = TranslationFallbackSettings(
        enabled=True,
        model=TranslationModel.GEMMA4_31B_CEREBRAS,
        connection=TranslationConnection.OFFICIAL_BYOK,
    )
    pending.openrouter.routing_mode = OpenRouterRoutingMode.LATENCY
    pending.qwen.llm_model = QwenLLMModel.QWEN_35_FLASH
    pending.qwen.region = QwenRegion.SINGAPORE
    pending.managed_identity.verified_hardware_hash = "pending-hash"
    pending.managed_identity.verified_hardware_hash_salt_version = 7
    pending.system_prompt = "draft prompt"
    pending.system_prompts = {"openrouter": "draft prompt"}

    merged = controller.merge_settings_tab_apply_with_current_languages(pending)

    assert merged is not controller.settings
    assert merged is not pending
    assert merged.languages.source_language == "es"
    assert merged.languages.target_language == "pt"
    assert merged.languages.peer_source_language == "zh-CN"
    assert merged.languages.peer_target_language == "nl"
    assert merged.languages.recent_source_languages == ["fr", "ko"]
    assert merged.languages.recent_target_languages == ["de", "en"]
    assert merged.provider.stt == STTProviderName.SONIOX
    assert merged.provider.peer_stt == STTProviderName.SONIOX
    assert merged.provider.llm == LLMProviderName.OPENROUTER
    assert merged.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert merged.openrouter.selection_alias == OpenRouterSelectionAlias.QWEN35_FLASH_MANAGED
    assert merged.translation.fallback == TranslationFallbackSettings(
        enabled=True,
        model=TranslationModel.GEMMA4_31B_CEREBRAS,
        connection=TranslationConnection.OFFICIAL_BYOK,
    )
    assert merged.openrouter.routing_mode == OpenRouterRoutingMode.LATENCY
    assert merged.qwen.llm_model == QwenLLMModel.QWEN_35_FLASH
    assert merged.qwen.region == QwenRegion.SINGAPORE
    assert merged.managed_identity.verified_hardware_hash == "pending-hash"
    assert merged.managed_identity.verified_hardware_hash_salt_version == 7
    assert merged.system_prompt == "draft prompt"
    assert merged.system_prompts == {}


def test_peer_auto_mode_survives_soniox_to_qwen_gpu_provider_switch() -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.provider.peer_stt = STTProviderName.SONIOX
    controller.settings.languages.peer_source_mode = "auto"
    controller.settings.languages.peer_source_language = "ja"
    pending = copy.deepcopy(controller.settings)
    pending.provider.peer_stt = STTProviderName.LOCAL_QWEN_GPU

    merged = controller.merge_settings_tab_apply_with_current_languages(pending)

    assert merged.provider.peer_stt == STTProviderName.LOCAL_QWEN_GPU
    assert merged.languages.peer_source_mode == "auto"
    assert merged.languages.peer_source_language == "ja"


def test_peer_auto_mode_falls_back_to_manual_without_replacing_saved_language() -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.provider.peer_stt = STTProviderName.SONIOX
    controller.settings.languages.peer_source_mode = "auto"
    controller.settings.languages.peer_source_language = "ja"
    pending = copy.deepcopy(controller.settings)
    pending.provider.peer_stt = STTProviderName.DEEPGRAM

    merged = controller.merge_settings_tab_apply_with_current_languages(pending)

    assert merged.provider.peer_stt == STTProviderName.DEEPGRAM
    assert merged.languages.peer_source_mode == "manual"
    assert merged.languages.peer_source_language == "ja"


@pytest.mark.asyncio
async def test_order21_validation_failure_does_not_leak_rejected_canonical_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = AppSettings()
    baseline.llm.concurrency_limit = 2
    controller, path = _controller_with_persisted_settings(tmp_path, baseline)
    pending = copy.deepcopy(controller.settings)
    pending.llm.concurrency_limit = 3

    class RejectingValidator:
        async def validate(self, request):
            _ = request
            return settings_mutation.SettingsMutationValidationResult(
                succeeded=False,
                message=None,
                diagnostics=None,
            )

    monkeypatch.setattr(
        provider_settings_module,
        "settings_path_mutation_validator_for_command",
        lambda _command: RejectingValidator(),
    )

    await controller.apply_providers(pending)

    assert _settings_result(controller) is not None
    assert _settings_result(controller).status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED
    assert controller.settings.llm.concurrency_limit == 2
    assert controller.vnext_settings is not None
    assert controller.vnext_settings.intent.translation.concurrency_limit == 2

    controller.settings.ui.locale = "ja"
    assert controller._get_settings_owner().save_current() is True
    persisted = (
        canonical_persistence_adapter_module.SettingsVNextCanonicalPersistenceAdapter().load_active(
            path
        )
    )
    assert persisted.canonical_settings.intent.translation.concurrency_limit == 2
    assert persisted.canonical_settings.intent.ui.locale == "ja"


@pytest.mark.asyncio
async def test_order21_plan_failure_does_not_leak_rejected_canonical_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = AppSettings()
    baseline.llm.concurrency_limit = 2
    controller, path = _controller_with_persisted_settings(tmp_path, baseline)
    pending = copy.deepcopy(controller.settings)
    pending.llm.concurrency_limit = 3

    def fail_plan(*_args, **_kwargs):
        raise RuntimeError("injected provider plan failure")

    monkeypatch.setattr(
        provider_runtime_apply_module.ProviderRuntimeOwner,
        "build_plan",
        fail_plan,
    )

    with pytest.raises(RuntimeError, match="injected provider plan failure"):
        await controller.apply_providers(pending)

    assert controller.settings.llm.concurrency_limit == 2
    assert controller.vnext_settings is not None
    assert controller.vnext_settings.intent.translation.concurrency_limit == 2

    controller.settings.ui.locale = "ko"
    assert controller._get_settings_owner().save_current() is True
    persisted = (
        canonical_persistence_adapter_module.SettingsVNextCanonicalPersistenceAdapter().load_active(
            path
        )
    )
    assert persisted.canonical_settings.intent.translation.concurrency_limit == 2
    assert persisted.canonical_settings.intent.ui.locale == "ko"


@pytest.mark.asyncio
async def test_managed_auth_repository_persists_pending_delivery_ack_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    committed = AppSettings()
    saved: list[AppSettings] = []

    def record_saved(_path: Path, settings: AppSettings) -> None:
        saved.append(copy.deepcopy(settings))

    _patch_settings_save(monkeypatch, record_saved)
    repository = controller._get_settings_owner().create_legacy_patch_repository(
        committed_settings=committed,
        surface="managed_connection_auth",
        save_failure_sink=controller._log_error,
    )

    result = await repository.save(
        SettingsCommitRequest(
            values={
                "state": {
                    "managed_connection": {
                        "pending_delivery_ack_source": "discord",
                        "pending_delivery_ack_delivery_id": "delivery-1",
                        "pending_delivery_ack_managed_credential_ref": "managed-ref-1",
                        "pending_delivery_ack_expires_at": "2026-07-07T00:15:00.000Z",
                    }
                }
            },
            expected_revision=None,
            reason="managed_connection_auth",
        )
    )

    assert result.succeeded is True
    assert saved[0].managed_identity.pending_delivery_ack_source == "discord"
    assert saved[0].managed_identity.pending_delivery_ack_delivery_id == "delivery-1"
    assert saved[0].managed_identity.pending_delivery_ack_managed_credential_ref == "managed-ref-1"
    assert "delivery_ack_token" not in repr(result.snapshot.values if result.snapshot else {})


@pytest.mark.asyncio
async def test_apply_providers_failed_signature_retries_same_settings_without_raw_exception_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller._runtime_logging = RuntimeLoggingSpy()
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.GEMINI
    controller.hub = DummyHub(llm=object())
    controller._last_self_stt_provider_signature = build_self_stt_provider_signature(
        controller.settings
    )
    controller._last_peer_stt_provider_signature = _peer_provider_signature(
        controller, controller.settings
    )
    controller._last_llm_provider_signature = controller._build_llm_provider_signature(
        controller.settings
    )
    pending = copy.deepcopy(controller.settings)
    pending.provider.llm = LLMProviderName.OPENROUTER
    pending.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    pending.openrouter.selection_alias = OpenRouterSelectionAlias.GEMMA4_BYOK
    pending_signature = controller._build_llm_provider_signature(pending)
    recovered_llm = object()
    create_attempts: list[LLMProviderName] = []
    saved_settings: list[AppSettings] = []
    raw_exception_text = "provider unavailable secret-token-must-not-leak"

    def record_saved_settings(_path, settings) -> None:
        saved_settings.append(copy.deepcopy(settings))

    def fail_then_recover_llm_provider(settings, **_kwargs) -> object:
        create_attempts.append(settings.provider.llm)
        if len(create_attempts) == 1:
            raise RuntimeError(raw_exception_text)
        return recovered_llm

    _patch_settings_save(monkeypatch, record_saved_settings)
    monkeypatch.setattr(
        controller_module, "create_secret_store", lambda *_args, **_kwargs: DummySecrets({})
    )
    monkeypatch.setattr(controller_module, "create_llm_provider", fail_then_recover_llm_provider)

    await controller.apply_providers(pending)

    first_result = _settings_result(controller)
    first_signature_after_failure = controller._last_llm_provider_signature
    first_basic_logs = "\n".join(
        message for _level, message in controller._runtime_logging.basic_messages
    )
    assert first_result is not None
    assert (
        first_result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
    )
    assert saved_settings[0].provider.llm == LLMProviderName.OPENROUTER
    assert controller.settings.provider.llm == LLMProviderName.OPENROUTER
    assert controller.hub.llm is None
    assert raw_exception_text not in first_basic_logs
    assert first_signature_after_failure != pending_signature

    await controller.apply_providers(pending)

    assert create_attempts == [LLMProviderName.OPENROUTER, LLMProviderName.OPENROUTER]
    assert controller.hub.llm is recovered_llm
    assert controller._last_llm_provider_signature == pending_signature


@pytest.mark.asyncio
async def test_apply_providers_force_rebuild_failed_signature_uses_miss_sentinel_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.GEMINI
    controller.hub = DummyHub(llm=object())
    controller._last_self_stt_provider_signature = build_self_stt_provider_signature(
        controller.settings
    )
    controller._last_peer_stt_provider_signature = _peer_provider_signature(
        controller, controller.settings
    )
    target_signature = controller._build_llm_provider_signature(controller.settings)
    controller._last_llm_provider_signature = target_signature
    recovered_llm = object()
    create_attempts: list[LLMProviderName] = []

    def fail_then_recover_llm_provider(settings, **_kwargs) -> object:
        create_attempts.append(settings.provider.llm)
        if len(create_attempts) == 1:
            raise RuntimeError("provider unavailable")
        return recovered_llm

    _patch_settings_save(monkeypatch, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        controller_module, "create_secret_store", lambda *_args, **_kwargs: DummySecrets({})
    )
    monkeypatch.setattr(controller_module, "create_llm_provider", fail_then_recover_llm_provider)

    await controller.apply_providers(force_rebuild_llm=True)

    assert create_attempts == [LLMProviderName.GEMINI]
    assert controller.hub.llm is None
    assert controller._last_llm_provider_signature == ()

    await controller.apply_providers()

    assert create_attempts == [LLMProviderName.GEMINI, LLMProviderName.GEMINI]
    assert controller.hub.llm is recovered_llm
    assert controller._last_llm_provider_signature == target_signature


@pytest.mark.asyncio
async def test_apply_providers_broker_base_url_rebuilds_managed_broker_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.settings.openrouter.selection_alias = OpenRouterSelectionAlias.GEMMA4_MANAGED
    controller.settings.openrouter.broker_base_url = "https://old-broker.example.test/"
    controller.hub = DummyHub()
    old_service = DummyManagedReleaseService(
        ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            message_key="managed_release.ready",
        )
    )
    controller._managed_openrouter_release_service = old_service
    controller._last_self_stt_provider_signature = build_self_stt_provider_signature(
        controller.settings
    )
    controller._last_peer_stt_provider_signature = _peer_provider_signature(
        controller, controller.settings
    )
    controller._last_llm_provider_signature = controller._build_llm_provider_signature(
        controller.settings
    )
    pending = copy.deepcopy(controller.settings)
    pending.openrouter.broker_base_url = "https://new-broker.example.test/"
    captured_services: list[object | None] = []

    def capture_llm_provider(*_args, managed_release_service=None, **_kwargs) -> object:
        captured_services.append(managed_release_service)
        return object()

    _patch_settings_save(monkeypatch, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        controller_module, "create_secret_store", lambda *_args, **_kwargs: DummySecrets({})
    )
    monkeypatch.setattr(controller_module, "create_llm_provider", capture_llm_provider)

    await controller.apply_providers(pending)

    assert old_service.close_calls == 1
    assert len(captured_services) == 1
    service = captured_services[0]
    assert isinstance(service, ManagedOpenRouterReleaseService)
    assert isinstance(service.client, HttpManagedOpenRouterBrokerClient)
    assert service.client.base_url == "https://new-broker.example.test"


def test_create_managed_openrouter_release_service_skips_managed_fallback_branch() -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.GEMINI
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    controller.settings.openrouter.selection_alias = OpenRouterSelectionAlias.GEMMA4_BYOK
    controller.settings.translation.fallback = TranslationFallbackSettings(
        enabled=True,
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.MANAGED_CHINA,
    )

    service = controller._create_managed_openrouter_release_service(secrets=DummySecrets({}))

    assert service is None


@pytest.mark.asyncio
async def test_apply_providers_managed_identity_rebuilds_service_with_pending_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.settings.openrouter.selection_alias = OpenRouterSelectionAlias.GEMMA4_MANAGED
    controller.settings.managed_identity.verified_hardware_hash = "old-hardware-hash"
    controller.settings.managed_identity.verified_hardware_hash_salt_version = 1
    controller.hub = DummyHub()
    controller._last_self_stt_provider_signature = build_self_stt_provider_signature(
        controller.settings
    )
    controller._last_peer_stt_provider_signature = _peer_provider_signature(
        controller, controller.settings
    )
    controller._last_llm_provider_signature = controller._build_llm_provider_signature(
        controller.settings
    )
    pending = copy.deepcopy(controller.settings)
    pending.managed_identity.verified_hardware_hash = "pending-hardware-hash"
    pending.managed_identity.verified_hardware_hash_salt_version = 9
    captured_services: list[object | None] = []

    def capture_llm_provider(*_args, managed_release_service=None, **_kwargs) -> object:
        captured_services.append(managed_release_service)
        return object()

    _patch_settings_save(monkeypatch, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        controller_module, "create_secret_store", lambda *_args, **_kwargs: DummySecrets({})
    )
    monkeypatch.setattr(controller_module, "create_llm_provider", capture_llm_provider)

    await controller.apply_providers(pending)

    assert len(captured_services) == 1
    service = captured_services[0]
    assert isinstance(service, ManagedOpenRouterReleaseService)
    assert service.managed_state.verified_hardware_hash == "pending-hardware-hash"
    assert service.managed_state.verified_hardware_hash_salt_version == 9
    assert controller.settings.managed_identity.verified_hardware_hash == "pending-hardware-hash"


@pytest.mark.asyncio
async def test_apply_providers_mixed_degraded_default_service_persists_full_provider_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.translation.fallback = TranslationFallbackSettings(enabled=False)
    controller.settings.system_prompt = "base prompt"
    controller.hub = DummyHub()
    controller._last_self_stt_provider_signature = build_self_stt_provider_signature(
        controller.settings
    )
    controller._last_peer_stt_provider_signature = _peer_provider_signature(
        controller, controller.settings
    )
    controller._last_llm_provider_signature = controller._build_llm_provider_signature(
        controller.settings
    )
    pending = copy.deepcopy(controller.settings)
    pending.translation.fallback = TranslationFallbackSettings(
        enabled=True,
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.OPENROUTER,
    )
    pending.system_prompt = "draft prompt"
    saved_settings: list[AppSettings] = []
    rebuild_prompts: list[str] = []

    def record_saved_settings(_path, settings) -> None:
        saved_settings.append(copy.deepcopy(settings))

    async def fail_first_rebuild_llm_provider(_owner) -> None:
        assert controller.settings is not None
        rebuild_prompts.append(controller.settings.system_prompt)
        if len(rebuild_prompts) == 1:
            raise RuntimeError("first runtime apply failed")

    _patch_settings_save(monkeypatch, record_saved_settings)
    monkeypatch.setattr(
        provider_runtime_apply_module.LlmProviderRebuildOwner,
        "rebuild",
        fail_first_rebuild_llm_provider,
    )

    await controller.apply_providers(pending)

    assert _settings_result(controller) is not None
    assert (
        _settings_result(controller).status
        == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
    )
    assert [settings.system_prompt for settings in saved_settings] == [
        "base prompt",
        "draft prompt",
    ]
    assert controller.settings.system_prompt == "draft prompt"
    assert controller.hub.system_prompt == "draft prompt"
    assert rebuild_prompts == ["base prompt"]


@pytest.mark.asyncio
async def test_order22_apply_settings_routes_stt_language_audio_patch_through_default_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.QWEN
    controller.settings.stt.low_latency_mode = False
    controller.settings.languages.peer_source_language = "en"
    controller.settings.languages.peer_target_language = "ko"
    controller.hub = DummyHub(stt=object(), peer_stt=object())
    controller.hub.source_language = controller.settings.languages.source_language
    controller.hub.target_language = controller.settings.languages.target_language
    controller.hub.peer_source_language = controller.settings.languages.peer_source_language
    controller.hub.peer_target_language = controller.settings.languages.peer_target_language
    controller.hub.low_latency_mode = controller.settings.stt.low_latency_mode
    controller._peer_runtime = DummyPeerRuntime()
    controller._last_self_stt_runtime_signature = build_self_stt_runtime_signature(
        controller.settings
    )
    controller._last_peer_stt_runtime_signature = _peer_runtime_signature(
        controller, controller.settings
    )
    controller._last_microphone_test_audio_settings_signature = (
        controller._microphone_test_audio_settings_signature(controller.settings)
    )
    controller._last_peer_translation_enabled = controller.settings.ui.peer_translation_enabled
    controller._last_vrc_mic_sync_enabled = controller.settings.osc.vrc_mic_intercept

    pending = copy.deepcopy(controller.settings)
    pending.provider.stt = STTProviderName.SONIOX
    pending.languages.source_language = "ja"
    pending.languages.peer_source_language = ""
    pending.audio.input_device = "Headset Mic"
    pending.desktop_audio.vad_hangover_ms = 950
    pending.stt.low_latency_mode = True
    pending.soniox_stt.trailing_silence_ms = 175
    requests: list[settings_mutation.SettingsMutationRequest] = []
    saved_settings: list[AppSettings] = []
    calls: list[str] = []

    original_mutate = settings_mutation.SettingsMutationService.mutate

    async def capture_mutate(self, request):
        requests.append(request)
        return await original_mutate(self, request)

    def record_saved_settings(_path, settings) -> None:
        saved_settings.append(copy.deepcopy(settings))

    async def fake_stop_microphone_test_for_audio_settings_change(self) -> None:
        calls.append("mic_stop")

    async def fake_rebuild_llm_provider(self) -> None:
        calls.append("llm")

    async def fake_refresh_peer_stt_runtime(self) -> None:
        calls.append("peer")

    async def fake_replace_runtime_stt_provider(self) -> None:
        calls.append("replace")

    monkeypatch.setattr(settings_mutation.SettingsMutationService, "mutate", capture_mutate)
    _patch_settings_save(monkeypatch, record_saved_settings)
    monkeypatch.setattr(GuiController, "_sync_clipboard_watcher", lambda self: asyncio.sleep(0))
    monkeypatch.setattr(
        GuiController,
        "_clear_local_stt_pending_enable_if_provider_switched_away",
        lambda self: None,
    )
    monkeypatch.setattr(
        GuiController,
        "stop_microphone_test_for_audio_settings_change",
        fake_stop_microphone_test_for_audio_settings_change,
    )
    monkeypatch.setattr(
        provider_runtime_apply_module.LlmProviderRebuildOwner, "rebuild", fake_rebuild_llm_provider
    )
    monkeypatch.setattr(GuiController, "_refresh_peer_stt_runtime", fake_refresh_peer_stt_runtime)
    monkeypatch.setattr(
        GuiController, "_replace_runtime_stt_provider", fake_replace_runtime_stt_provider
    )

    await controller.apply_settings(pending)

    assert len(requests) == 1
    request = requests[0]
    assert request.reason == settings_mutation.SETTINGS_MUTATION_SURFACE_STT_LANGUAGE_AUDIO
    assert request.values == {
        "provider.stt": STTProviderName.SONIOX,
        "languages.source_language": "ja",
        "languages.peer_source_language": "",
        "audio.input_device": "Headset Mic",
        "desktop_audio.vad_hangover_ms": 950,
        "soniox_stt.trailing_silence_ms": 175,
    }
    assert "translation.model" not in request.values
    assert "overlay.target" not in request.values
    assert len(saved_settings) == 1
    assert saved_settings[0].provider.stt == STTProviderName.SONIOX
    assert saved_settings[0].languages.source_language == "ja"
    assert saved_settings[0].audio.input_device == "Headset Mic"
    assert controller.hub.clear_language_runtime_state_calls == ["self", "peer"]
    assert calls == ["mic_stop", "peer", "replace"]
    assert controller.hub.source_language == "ja"
    assert controller.hub.low_latency_mode is True


@pytest.mark.asyncio
async def test_order22_apply_settings_runtime_failure_degrades_without_rollback_or_raw_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    controller.hub = DummyHub(stt=object())
    controller.hub.source_language = controller.settings.languages.source_language
    controller.hub.target_language = controller.settings.languages.target_language
    controller._last_self_stt_runtime_signature = build_self_stt_runtime_signature(
        controller.settings
    )
    controller._last_peer_stt_runtime_signature = _peer_runtime_signature(
        controller, controller.settings
    )
    controller._last_peer_translation_enabled = controller.settings.ui.peer_translation_enabled
    controller._last_vrc_mic_sync_enabled = controller.settings.osc.vrc_mic_intercept
    pending = copy.deepcopy(controller.settings)
    pending.provider.stt = STTProviderName.SONIOX
    pending.languages.source_language = "ja"
    saved_settings: list[AppSettings] = []
    raw_failure_text = "stt runtime failed secret-token-must-not-leak"

    def record_saved_settings(_path, settings) -> None:
        saved_settings.append(copy.deepcopy(settings))

    async def fail_replace_runtime_stt_provider(self) -> None:
        raise RuntimeError(raw_failure_text)

    _patch_settings_save(monkeypatch, record_saved_settings)
    monkeypatch.setattr(GuiController, "_sync_clipboard_watcher", lambda self: asyncio.sleep(0))
    monkeypatch.setattr(
        GuiController,
        "_clear_local_stt_pending_enable_if_provider_switched_away",
        lambda self: None,
    )
    monkeypatch.setattr(
        GuiController, "_replace_runtime_stt_provider", fail_replace_runtime_stt_provider
    )

    await controller.apply_settings(pending)

    result = _settings_result(controller)
    assert result is not None
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
    assert result.message == messages.UserMessageRef(
        key="settings.mutation.runtime_apply_failed",
        params={"phase": "runtime_apply"},
        severity=messages.SEVERITY_WARNING,
    )
    assert result.diagnostics == messages.ErrorDiagnostics(
        component="gui_controller",
        operation="apply_stt_language_audio_runtime",
        code="stt_language_audio_runtime_apply_exception",
        category=messages.DIAGNOSTIC_CATEGORY_LIFECYCLE,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"surface": "stt_language_audio"},
    )
    assert raw_failure_text not in repr(result)
    assert len(saved_settings) == 1
    assert saved_settings[0].provider.stt == STTProviderName.SONIOX
    assert controller.settings.provider.stt == STTProviderName.SONIOX


@pytest.mark.asyncio
async def test_order22_save_failure_surface_is_stt_language_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller._runtime_logging = RuntimeLoggingSpy()
    controller.settings = AppSettings()
    pending = copy.deepcopy(controller.settings)
    pending.audio.input_device = "Headset Mic"
    raw_failure_text = "save failed secret-token-must-not-leak"

    def fail_save_settings(_path, _settings) -> None:
        raise RuntimeError(raw_failure_text)

    _patch_settings_save(monkeypatch, fail_save_settings)

    await controller.apply_settings(pending)

    result = _settings_result(controller)
    assert result is not None
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED
    assert result.diagnostics == messages.ErrorDiagnostics(
        component="settings_repository",
        operation="save",
        code="settings_save_failed",
        category=messages.DIAGNOSTIC_CATEGORY_TRANSACTION,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"surface": "stt_language_audio"},
    )
    assert raw_failure_text not in repr(result)
    assert raw_failure_text not in repr(controller._runtime_logging.basic_messages)


@pytest.mark.asyncio
async def test_order22_save_failure_does_not_leak_rejected_canonical_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = AppSettings()
    baseline.provider.stt = STTProviderName.DEEPGRAM
    controller, path = _controller_with_persisted_settings(tmp_path, baseline)
    pending = copy.deepcopy(controller.settings)
    pending.provider.stt = STTProviderName.SONIOX
    adapter_type = canonical_persistence_adapter_module.SettingsVNextCanonicalPersistenceAdapter
    original_persist = adapter_type.persist
    persist_calls = 0

    def fail_once(self, incoming_path, canonical) -> None:
        nonlocal persist_calls
        persist_calls += 1
        if persist_calls == 1:
            raise OSError("injected order22 save failure")
        original_persist(self, incoming_path, canonical)

    monkeypatch.setattr(adapter_type, "persist", fail_once)

    await controller.apply_settings(pending)

    assert _settings_result(controller) is not None
    assert _settings_result(controller).status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED
    assert controller.settings.provider.stt == STTProviderName.DEEPGRAM
    assert controller.vnext_settings is not None
    assert controller.vnext_settings.intent.stt.provider == STTProviderName.DEEPGRAM.value

    controller.settings.ui.locale = "ja"
    assert controller._get_settings_owner().save_current() is True
    persisted = adapter_type().load_active(path)
    assert persisted.canonical_settings.intent.stt.provider == STTProviderName.DEEPGRAM.value
    assert persisted.canonical_settings.intent.ui.locale == "ja"


@pytest.mark.asyncio
async def test_order22_live_settings_view_alias_save_failure_restores_legacy_and_canonical_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = AppSettings()
    baseline.provider.stt = STTProviderName.DEEPGRAM
    baseline.provider.peer_stt = STTProviderName.DEEPGRAM
    baseline.audio.input_device = "Built-in Mic"
    controller, path = _controller_with_persisted_settings(tmp_path, baseline)
    settings_view = DummySettingsView()
    controller.app = _presentation(
        SimpleNamespace(view_dashboard=DummyDashboard(), view_settings=settings_view)
    )
    controller._runtime_logging = RuntimeLoggingSpy()
    controller._sync_ui_from_settings()

    assert settings_view.calls[-1][0] is controller.settings

    controller.settings.audio.input_device = "Headset Mic"
    pending = copy.deepcopy(controller.settings)
    adapter_type = canonical_persistence_adapter_module.SettingsVNextCanonicalPersistenceAdapter
    original_persist = adapter_type.persist
    persist_calls = 0
    raw_failure_text = "live alias save failed secret-token-must-not-leak"

    def fail_once(self, incoming_path, canonical) -> None:
        nonlocal persist_calls
        persist_calls += 1
        if persist_calls == 1:
            raise OSError(raw_failure_text)
        original_persist(self, incoming_path, canonical)

    monkeypatch.setattr(adapter_type, "persist", fail_once)

    await controller.apply_settings(pending)

    result = _settings_result(controller)
    assert result is not None
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED
    assert controller.settings.audio.input_device == "Built-in Mic"
    assert controller.vnext_settings is not None
    assert controller.vnext_settings.intent.audio.input_device == "Built-in Mic"
    assert raw_failure_text not in repr(result)
    assert raw_failure_text not in repr(controller._runtime_logging.basic_messages)

    controller.settings.ui.locale = "ja"
    assert controller._get_settings_owner().save_current() is True
    persisted = adapter_type().load_active(path)
    assert persisted.canonical_settings.intent.audio.input_device == "Built-in Mic"
    assert persisted.canonical_settings.intent.ui.locale == "ja"


@pytest.mark.asyncio
async def test_order22_apply_settings_self_stt_provider_specific_change_restarts_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    controller.settings.provider.stt = STTProviderName.DEEPGRAM
    controller.settings.deepgram_stt.model = "nova-3"
    controller.hub = DummyHub(stt=object())
    controller.hub.source_language = controller.settings.languages.source_language
    controller.hub.target_language = controller.settings.languages.target_language
    controller._last_self_stt_runtime_signature = build_self_stt_runtime_signature(
        controller.settings
    )
    controller._last_peer_stt_runtime_signature = _peer_runtime_signature(
        controller, controller.settings
    )
    controller._last_peer_translation_enabled = controller.settings.ui.peer_translation_enabled
    controller._last_vrc_mic_sync_enabled = controller.settings.osc.vrc_mic_intercept
    pending = copy.deepcopy(controller.settings)
    pending.deepgram_stt.model = "nova-2"
    requests: list[settings_mutation.SettingsMutationRequest] = []
    replace_calls: list[str] = []

    original_mutate = settings_mutation.SettingsMutationService.mutate

    async def capture_mutate(self, request):
        requests.append(request)
        return await original_mutate(self, request)

    async def fake_replace_runtime_stt_provider(self) -> None:
        replace_calls.append("replace")

    monkeypatch.setattr(settings_mutation.SettingsMutationService, "mutate", capture_mutate)
    _patch_settings_save(monkeypatch, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(GuiController, "_sync_clipboard_watcher", lambda self: asyncio.sleep(0))
    monkeypatch.setattr(
        GuiController,
        "_clear_local_stt_pending_enable_if_provider_switched_away",
        lambda self: None,
    )
    monkeypatch.setattr(
        GuiController, "_replace_runtime_stt_provider", fake_replace_runtime_stt_provider
    )

    await controller.apply_settings(pending)

    assert len(requests) == 1
    assert requests[0].values == {"deepgram_stt.model": "nova-2"}
    assert replace_calls == ["replace"]


@pytest.mark.asyncio
async def test_order22_apply_settings_mixed_draft_applies_audio_runtime_and_preserves_out_of_scope_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    controller.settings.audio.input_device = "Base Mic"
    controller.settings.overlay.show_translation = True
    controller.settings.system_prompt = "base prompt"
    controller.hub = DummyHub(stt=object())
    controller.hub.source_language = controller.settings.languages.source_language
    controller.hub.target_language = controller.settings.languages.target_language
    controller.hub.system_prompt = controller.settings.system_prompt
    controller._last_self_stt_runtime_signature = build_self_stt_runtime_signature(
        controller.settings
    )
    controller._last_peer_stt_runtime_signature = _peer_runtime_signature(
        controller, controller.settings
    )
    controller._last_peer_translation_enabled = controller.settings.ui.peer_translation_enabled
    controller._last_vrc_mic_sync_enabled = controller.settings.osc.vrc_mic_intercept
    pending = copy.deepcopy(controller.settings)
    pending.audio.input_device = "Desk Mic"
    pending.overlay.show_translation = False
    pending.osc.chatbox_include_source = True
    pending.system_prompt = "draft prompt"
    requests: list[settings_mutation.SettingsMutationRequest] = []
    saved_settings: list[AppSettings] = []
    calls: list[str] = []

    original_mutate = settings_mutation.SettingsMutationService.mutate

    async def capture_mutate(self, request):
        requests.append(request)
        return await original_mutate(self, request)

    def record_saved_settings(_path, settings) -> None:
        saved_settings.append(copy.deepcopy(settings))

    async def fake_stop_microphone_test_for_audio_settings_change(self) -> None:
        calls.append("mic_stop")

    monkeypatch.setattr(settings_mutation.SettingsMutationService, "mutate", capture_mutate)
    _patch_settings_save(monkeypatch, record_saved_settings)
    monkeypatch.setattr(GuiController, "_sync_clipboard_watcher", lambda self: asyncio.sleep(0))
    monkeypatch.setattr(
        GuiController,
        "_clear_local_stt_pending_enable_if_provider_switched_away",
        lambda self: None,
    )
    monkeypatch.setattr(GuiController, "_refresh_peer_stt_runtime", lambda self: asyncio.sleep(0))
    monkeypatch.setattr(
        GuiController, "_replace_runtime_stt_provider", lambda self: asyncio.sleep(0)
    )
    monkeypatch.setattr(
        GuiController,
        "stop_microphone_test_for_audio_settings_change",
        fake_stop_microphone_test_for_audio_settings_change,
    )

    await controller.apply_settings(pending)

    assert len(requests) == 3
    assert requests[0].values == {"audio.input_device": "Desk Mic"}
    assert requests[1].values == {
        "overlay.show_translation": False,
        "osc.chatbox_include_source": True,
    }
    assert (
        requests[2].reason == settings_mutation.SETTINGS_MUTATION_SURFACE_UI_PROMPT_CLIPBOARD_STATE
    )
    assert requests[2].values == {"system_prompt": "draft prompt"}
    assert "overlay.show_translation" not in requests[0].values
    assert "osc.chatbox_include_source" not in requests[0].values
    assert "system_prompt" not in requests[0].values
    assert [settings.audio.input_device for settings in saved_settings] == [
        "Desk Mic",
        "Desk Mic",
        "Desk Mic",
    ]
    assert saved_settings[0].overlay.show_translation is True
    assert saved_settings[0].system_prompt == "base prompt"
    assert saved_settings[1].overlay.show_translation is False
    assert saved_settings[1].osc.chatbox_include_source is True
    assert saved_settings[1].system_prompt == "base prompt"
    assert saved_settings[2].overlay.show_translation is False
    assert saved_settings[2].osc.chatbox_include_source is True
    assert saved_settings[2].system_prompt == "draft prompt"
    assert calls == ["mic_stop"]
    assert controller.settings.overlay.show_translation is False
    assert controller.settings.osc.chatbox_include_source is True
    assert controller.settings.system_prompt == "draft prompt"


@pytest.mark.asyncio
async def test_mixed_order22_order23_order24_fallback_save_failure_restores_committed_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(apply_locale=lambda: None))
    controller._runtime_logging = RuntimeLoggingSpy()
    controller.settings = AppSettings()
    controller.settings.llm.concurrency_limit = 2
    controller.settings.overlay.calibration = OverlayCalibration(distance=0.8, offset_x=0.2)
    controller._get_overlay_calibration_application_owner().replace_current(
        controller.settings.overlay.calibration.copy()
    )
    controller.settings.overlay.show_translation = True
    controller.settings.system_prompt = "base prompt"
    controller.hub = DummyHub(stt=object())
    controller.hub.source_language = controller.settings.languages.source_language
    controller.hub.target_language = controller.settings.languages.target_language
    controller.hub.system_prompt = controller.settings.system_prompt
    controller._last_self_stt_runtime_signature = build_self_stt_runtime_signature(
        controller.settings
    )
    controller._last_peer_stt_runtime_signature = _peer_runtime_signature(
        controller, controller.settings
    )
    controller._last_peer_translation_enabled = controller.settings.ui.peer_translation_enabled
    controller._last_vrc_mic_sync_enabled = controller.settings.osc.vrc_mic_intercept
    pending = copy.deepcopy(controller.settings)
    pending.languages.source_language = "ja"
    pending.overlay.calibration = OverlayCalibration(distance=1.6, offset_x=0.7)
    pending.overlay.show_translation = False
    pending.system_prompt = "draft prompt"
    pending.llm.concurrency_limit = 3
    saved_settings: list[AppSettings] = []
    raw_failure_text = "mixed full draft save failed secret-token-must-not-leak"

    def fail_uncommitted_full_draft_save(_path, incoming: AppSettings) -> None:
        saved_settings.append(copy.deepcopy(incoming))
        if incoming.llm.concurrency_limit == pending.llm.concurrency_limit:
            raise RuntimeError(raw_failure_text)

    _patch_settings_save(monkeypatch, fail_uncommitted_full_draft_save)
    monkeypatch.setattr(GuiController, "_sync_clipboard_watcher", lambda self: asyncio.sleep(0))
    monkeypatch.setattr(
        GuiController,
        "_clear_local_stt_pending_enable_if_provider_switched_away",
        lambda self: None,
    )
    monkeypatch.setattr(GuiController, "_refresh_peer_stt_runtime", lambda self: asyncio.sleep(0))
    monkeypatch.setattr(
        GuiController, "_replace_runtime_stt_provider", lambda self: asyncio.sleep(0)
    )

    await controller.apply_settings(pending)

    result = _settings_result(controller)
    assert result is not None
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
    assert result.diagnostics == messages.ErrorDiagnostics(
        component="gui_controller",
        operation="apply_order22_order23_order24_full_draft_save",
        code="settings_save_failed",
        category=messages.DIAGNOSTIC_CATEGORY_TRANSACTION,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"surface": "ui_prompt_clipboard_state"},
    )
    assert [settings.llm.concurrency_limit for settings in saved_settings] == [2, 2, 2, 3]
    assert controller.settings is not None
    assert controller.settings.languages.source_language == "ja"
    assert controller.hub.source_language == "ja"
    assert controller.settings.overlay.show_translation is False
    assert controller.settings.overlay.calibration.distance == 1.6
    assert controller.overlay_calibration.distance == 1.6
    assert controller.settings.system_prompt == "draft prompt"
    assert controller.hub.system_prompt == "draft prompt"
    assert controller.settings.llm.concurrency_limit == 2
    logged_text = "\n".join(
        message
        for _level, message in (
            controller._runtime_logging.basic_messages
            + controller._runtime_logging.detailed_messages
        )
    )
    assert raw_failure_text not in repr(result)
    assert raw_failure_text not in logged_text


@pytest.mark.asyncio
async def test_order22_apply_settings_mixed_full_draft_save_failure_degrades_and_restores_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller._runtime_logging = RuntimeLoggingSpy()
    controller.settings = AppSettings()
    controller.settings.audio.input_device = "Base Mic"
    controller.settings.overlay.calibration = OverlayCalibration(distance=0.8, offset_x=0.2)
    controller._get_overlay_calibration_application_owner().replace_current(
        controller.settings.overlay.calibration.copy()
    )
    controller.settings.overlay.show_translation = True
    controller.settings.system_prompt = "base prompt"
    controller.hub = DummyHub(stt=object())
    controller.hub.source_language = controller.settings.languages.source_language
    controller.hub.target_language = controller.settings.languages.target_language
    controller.hub.system_prompt = controller.settings.system_prompt
    controller._last_self_stt_runtime_signature = build_self_stt_runtime_signature(
        controller.settings
    )
    controller._last_peer_stt_runtime_signature = _peer_runtime_signature(
        controller, controller.settings
    )
    controller._last_peer_translation_enabled = controller.settings.ui.peer_translation_enabled
    controller._last_vrc_mic_sync_enabled = controller.settings.osc.vrc_mic_intercept
    pending = copy.deepcopy(controller.settings)
    pending.audio.input_device = "Desk Mic"
    pending.languages.source_language = "ja"
    pending.overlay.calibration = OverlayCalibration(distance=1.6, offset_x=0.7)
    pending.overlay.show_translation = False
    pending.system_prompt = "draft prompt"
    save_attempts: list[AppSettings] = []
    raw_failure_text = "full draft save failed secret-token-must-not-leak"

    def fail_third_save(_path, incoming: AppSettings) -> None:
        save_attempts.append(copy.deepcopy(incoming))
        if len(save_attempts) == 3:
            raise RuntimeError(raw_failure_text)

    _patch_settings_save(monkeypatch, fail_third_save)
    monkeypatch.setattr(GuiController, "_sync_clipboard_watcher", lambda self: asyncio.sleep(0))
    monkeypatch.setattr(
        GuiController,
        "_clear_local_stt_pending_enable_if_provider_switched_away",
        lambda self: None,
    )
    monkeypatch.setattr(GuiController, "_refresh_peer_stt_runtime", lambda self: asyncio.sleep(0))
    monkeypatch.setattr(
        GuiController, "_replace_runtime_stt_provider", lambda self: asyncio.sleep(0)
    )

    await controller.apply_settings(pending)

    result = _settings_result(controller)
    assert result is not None
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED
    assert result.diagnostics == messages.ErrorDiagnostics(
        component="settings_repository",
        operation="save",
        code="settings_save_failed",
        category=messages.DIAGNOSTIC_CATEGORY_TRANSACTION,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"surface": "ui_prompt_clipboard_state"},
    )
    assert [settings.audio.input_device for settings in save_attempts] == [
        "Desk Mic",
        "Desk Mic",
        "Desk Mic",
    ]
    assert [settings.languages.source_language for settings in save_attempts] == [
        "ja",
        "ja",
        "ja",
    ]
    assert [settings.overlay.calibration.distance for settings in save_attempts] == [
        0.8,
        1.6,
        1.6,
    ]
    assert save_attempts[0].overlay.show_translation is True
    assert save_attempts[0].system_prompt == "base prompt"
    assert save_attempts[1].overlay.show_translation is False
    assert save_attempts[1].system_prompt == "base prompt"
    assert save_attempts[2].overlay.show_translation is False
    assert save_attempts[2].system_prompt == "draft prompt"
    assert controller.settings.audio.input_device == "Desk Mic"
    assert controller.settings.languages.source_language == "ja"
    assert controller.hub.source_language == "ja"
    assert controller.settings.overlay.show_translation is False
    assert controller.settings.overlay.calibration.distance == 1.6
    assert controller.overlay_calibration.distance == 1.6
    assert controller.settings.system_prompt == "base prompt"
    assert controller.hub.system_prompt == "base prompt"
    assert raw_failure_text not in repr(result)
    assert raw_failure_text not in repr(controller._runtime_logging.basic_messages)


@pytest.mark.asyncio
async def test_order22_mixed_provider_draft_rebuilds_self_stt_from_pre_mutation_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    controller.settings.provider.stt = STTProviderName.DEEPGRAM
    controller.settings.system_prompt = "base prompt"
    controller.hub = DummyHub()
    controller.hub.system_prompt = controller.settings.system_prompt
    controller._stt_desired = False
    pending = copy.deepcopy(controller.settings)
    pending.provider.stt = STTProviderName.SONIOX
    pending.system_prompt = "draft prompt"
    requests: list[settings_mutation.SettingsMutationRequest] = []
    saved_settings: list[AppSettings] = []
    calls: list[str] = []

    original_mutate = settings_mutation.SettingsMutationService.mutate

    async def capture_mutate(self, request):
        requests.append(request)
        return await original_mutate(self, request)

    def record_saved_settings(_path, settings) -> None:
        saved_settings.append(copy.deepcopy(settings))

    async def fake_rebuild_stt_provider(self) -> None:
        calls.append("rebuild_stt")

    async def fake_replace_runtime_stt_provider(self) -> None:
        calls.append("replace")

    async def fake_refresh_peer_stt_runtime(self) -> None:
        calls.append("peer")

    async def fake_rebuild_llm_provider(self) -> None:
        calls.append("llm")

    monkeypatch.setattr(settings_mutation.SettingsMutationService, "mutate", capture_mutate)
    _patch_settings_save(monkeypatch, record_saved_settings)
    monkeypatch.setattr(GuiController, "_rebuild_stt_provider", fake_rebuild_stt_provider)
    monkeypatch.setattr(
        GuiController, "_replace_runtime_stt_provider", fake_replace_runtime_stt_provider
    )
    monkeypatch.setattr(GuiController, "_refresh_peer_stt_runtime", fake_refresh_peer_stt_runtime)
    monkeypatch.setattr(
        provider_runtime_apply_module.LlmProviderRebuildOwner, "rebuild", fake_rebuild_llm_provider
    )

    await controller.apply_providers(pending)

    assert len(requests) == 2
    assert requests[0].reason == settings_mutation.SETTINGS_MUTATION_SURFACE_STT_LANGUAGE_AUDIO
    assert requests[0].values == {"provider.stt": STTProviderName.SONIOX}
    assert "system_prompt" not in requests[0].values
    assert (
        requests[1].reason == settings_mutation.SETTINGS_MUTATION_SURFACE_UI_PROMPT_CLIPBOARD_STATE
    )
    assert requests[1].values == {"system_prompt": "draft prompt"}
    assert [settings.provider.stt for settings in saved_settings] == [
        STTProviderName.SONIOX,
        STTProviderName.SONIOX,
    ]
    assert saved_settings[0].system_prompt == "base prompt"
    assert saved_settings[1].system_prompt == "draft prompt"
    assert controller.settings.provider.stt == STTProviderName.SONIOX
    assert controller.settings.system_prompt == "draft prompt"
    assert controller.hub.system_prompt == "draft prompt"
    assert calls == ["rebuild_stt"]


@pytest.mark.asyncio
async def test_order21_provider_only_mixed_full_draft_save_failure_restores_committed_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller._runtime_logging = RuntimeLoggingSpy()
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.settings.llm.concurrency_limit = 2
    controller.hub = DummyHub(llm=object(), stt=object())
    controller._sync_signature_caches(controller.settings)
    pending = copy.deepcopy(controller.settings)
    pending.llm.concurrency_limit = 3
    pending.managed_identity.verified_hardware_hash = "pending-hardware-hash"
    pending.managed_identity.verified_hardware_hash_salt_version = 9
    requests: list[settings_mutation.SettingsMutationRequest] = []
    saved_settings: list[AppSettings] = []
    raw_failure_text = "order21 full draft save failed secret-token-must-not-leak"

    original_mutate = settings_mutation.SettingsMutationService.mutate

    async def capture_mutate(self, request):
        requests.append(request)
        return await original_mutate(self, request)

    def fail_uncommitted_full_draft_save(_path, incoming: AppSettings) -> None:
        saved_settings.append(copy.deepcopy(incoming))
        if (
            incoming.managed_identity.verified_hardware_hash
            == pending.managed_identity.verified_hardware_hash
        ):
            raise RuntimeError(raw_failure_text)

    async def fake_rebuild_llm_provider(self) -> None:
        assert self.hub is not None
        self.hub.llm = object()

    monkeypatch.setattr(settings_mutation.SettingsMutationService, "mutate", capture_mutate)
    _patch_settings_save(monkeypatch, fail_uncommitted_full_draft_save)
    monkeypatch.setattr(
        provider_runtime_apply_module.LlmProviderRebuildOwner, "rebuild", fake_rebuild_llm_provider
    )

    await controller.apply_providers(pending)

    assert len(requests) == 1
    assert requests[0].reason == settings_mutation.SETTINGS_MUTATION_SURFACE_TRANSLATION_PROVIDER
    assert requests[0].values == {"llm.concurrency_limit": 3}
    result = _settings_result(controller)
    assert result is not None
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
    assert result.diagnostics == messages.ErrorDiagnostics(
        component="gui_controller",
        operation="apply_translation_provider_full_draft_save",
        code="settings_save_failed",
        category=messages.DIAGNOSTIC_CATEGORY_TRANSACTION,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"surface": "translation_provider"},
    )
    assert [settings.managed_identity.verified_hardware_hash for settings in saved_settings] == [
        None,
        "pending-hardware-hash",
    ]
    assert controller.settings is not None
    assert controller.settings.llm.concurrency_limit == 3
    assert controller.settings.provider.llm == LLMProviderName.OPENROUTER
    assert controller.settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert controller.settings.managed_identity.verified_hardware_hash is None
    assert controller.settings.managed_identity.verified_hardware_hash_salt_version is None
    committed_signature = controller._build_llm_provider_signature(controller.settings)
    pending_signature = controller._build_llm_provider_signature(pending)
    assert controller._last_llm_provider_signature == committed_signature
    assert controller._last_llm_provider_signature != pending_signature
    logged_text = "\n".join(
        message
        for _level, message in (
            controller._runtime_logging.basic_messages
            + controller._runtime_logging.detailed_messages
        )
    )
    assert raw_failure_text not in repr(result)
    assert raw_failure_text not in logged_text


@pytest.mark.asyncio
async def test_order21_provider_only_mixed_full_draft_save_failure_preserves_retry_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller._runtime_logging = RuntimeLoggingSpy()
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.settings.llm.concurrency_limit = 2
    controller.hub = DummyHub(llm=object(), stt=object())
    controller._sync_signature_caches(controller.settings)
    pending = copy.deepcopy(controller.settings)
    pending.llm.concurrency_limit = 3
    pending.managed_identity.verified_hardware_hash = "pending-hardware-hash"
    pending.managed_identity.verified_hardware_hash_salt_version = 9
    requests: list[settings_mutation.SettingsMutationRequest] = []
    saved_settings: list[AppSettings] = []
    raw_failure_text = "order21 full draft save failed secret-token-must-not-leak"

    original_mutate = settings_mutation.SettingsMutationService.mutate

    async def capture_mutate(self, request):
        requests.append(request)
        return await original_mutate(self, request)

    def fail_uncommitted_full_draft_save(_path, incoming: AppSettings) -> None:
        saved_settings.append(copy.deepcopy(incoming))
        if (
            incoming.managed_identity.verified_hardware_hash
            == pending.managed_identity.verified_hardware_hash
        ):
            raise RuntimeError(raw_failure_text)

    async def unavailable_rebuild_llm_provider(_owner) -> None:
        assert controller.hub is not None
        controller.hub.llm = None

    monkeypatch.setattr(settings_mutation.SettingsMutationService, "mutate", capture_mutate)
    _patch_settings_save(monkeypatch, fail_uncommitted_full_draft_save)
    monkeypatch.setattr(
        provider_runtime_apply_module.LlmProviderRebuildOwner,
        "rebuild",
        unavailable_rebuild_llm_provider,
    )

    await controller.apply_providers(pending)

    assert len(requests) == 1
    assert requests[0].reason == settings_mutation.SETTINGS_MUTATION_SURFACE_TRANSLATION_PROVIDER
    assert requests[0].values == {"llm.concurrency_limit": 3}
    result = _settings_result(controller)
    assert result is not None
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
    assert result.diagnostics == messages.ErrorDiagnostics(
        component="gui_controller",
        operation="apply_translation_provider_full_draft_save",
        code="settings_save_failed",
        category=messages.DIAGNOSTIC_CATEGORY_TRANSACTION,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"surface": "translation_provider"},
    )
    assert [settings.managed_identity.verified_hardware_hash for settings in saved_settings] == [
        None,
        "pending-hardware-hash",
    ]
    assert controller.settings is not None
    assert controller.settings.llm.concurrency_limit == 3
    assert controller.settings.provider.llm == LLMProviderName.OPENROUTER
    assert controller.settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert controller.settings.managed_identity.verified_hardware_hash is None
    assert controller.settings.managed_identity.verified_hardware_hash_salt_version is None
    committed_signature = controller._build_llm_provider_signature(controller.settings)
    pending_signature = controller._build_llm_provider_signature(pending)
    assert controller._last_llm_provider_signature == ()
    assert controller._last_llm_provider_signature != committed_signature
    assert controller._last_llm_provider_signature != pending_signature
    logged_text = "\n".join(
        message
        for _level, message in (
            controller._runtime_logging.basic_messages
            + controller._runtime_logging.detailed_messages
        )
    )
    assert raw_failure_text not in repr(result)
    assert raw_failure_text not in logged_text


@pytest.mark.asyncio
async def test_order21_provider_only_mixed_full_draft_runtime_unavailable_degrades_without_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.settings.llm.concurrency_limit = 2
    controller.hub = DummyHub(llm=object(), stt=object())
    controller._sync_signature_caches(controller.settings)
    pending = copy.deepcopy(controller.settings)
    pending.llm.concurrency_limit = 3
    pending.managed_identity.verified_hardware_hash = "pending-hardware-hash"
    pending.managed_identity.verified_hardware_hash_salt_version = 9
    saved_settings: list[AppSettings] = []

    def record_saved_settings(_path, incoming: AppSettings) -> None:
        saved_settings.append(copy.deepcopy(incoming))

    async def conditionally_unavailable_rebuild_llm_provider(_owner) -> None:
        assert controller.settings is not None
        assert controller.hub is not None
        if (
            controller.settings.managed_identity.verified_hardware_hash
            == pending.managed_identity.verified_hardware_hash
        ):
            controller.hub.llm = None
        else:
            controller.hub.llm = object()

    _patch_settings_save(monkeypatch, record_saved_settings)
    monkeypatch.setattr(
        provider_runtime_apply_module.LlmProviderRebuildOwner,
        "rebuild",
        conditionally_unavailable_rebuild_llm_provider,
    )

    await controller.apply_providers(pending)

    result = _settings_result(controller)
    assert result is not None
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
    assert result.diagnostics == messages.ErrorDiagnostics(
        component="gui_controller",
        operation="apply_translation_provider_runtime",
        code="provider_runtime_apply_unavailable",
        category=messages.DIAGNOSTIC_CATEGORY_LIFECYCLE,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"surface": "translation_provider"},
    )
    assert [settings.managed_identity.verified_hardware_hash for settings in saved_settings] == [
        None,
        "pending-hardware-hash",
    ]
    assert controller.settings is not None
    assert controller.settings.llm.concurrency_limit == 3
    assert controller.settings.managed_identity.verified_hardware_hash == "pending-hardware-hash"
    assert controller.settings.managed_identity.verified_hardware_hash_salt_version == 9
    assert controller.hub.llm is None


@pytest.mark.asyncio
async def test_provider_order21_order24_fallback_save_failure_restores_committed_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(apply_locale=lambda: None))
    controller._runtime_logging = RuntimeLoggingSpy()
    controller.settings = AppSettings()
    controller.settings.llm.concurrency_limit = 2
    controller.settings.system_prompt = "base prompt"
    controller.hub = DummyHub(stt=object(), llm=object())
    controller.hub.system_prompt = controller.settings.system_prompt
    controller._stt_desired = False
    pending = copy.deepcopy(controller.settings)
    pending.llm.concurrency_limit = 3
    pending.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    pending.managed_identity.verified_hardware_hash = "pending-hardware-hash"
    pending.managed_identity.verified_hardware_hash_salt_version = 9
    pending.system_prompt = "provider prompt"
    saved_settings: list[AppSettings] = []
    raw_failure_text = "provider mixed full draft save failed secret-token-must-not-leak"

    def fail_uncommitted_full_draft_save(_path, incoming: AppSettings) -> None:
        saved_settings.append(copy.deepcopy(incoming))
        if (
            incoming.managed_identity.verified_hardware_hash
            == pending.managed_identity.verified_hardware_hash
        ):
            raise RuntimeError(raw_failure_text)

    async def fake_rebuild_llm_provider(self) -> None:
        assert self.hub is not None
        self.hub.llm = object()

    _patch_settings_save(monkeypatch, fail_uncommitted_full_draft_save)
    monkeypatch.setattr(GuiController, "_sync_clipboard_watcher", lambda self: asyncio.sleep(0))
    monkeypatch.setattr(
        provider_runtime_apply_module.LlmProviderRebuildOwner, "rebuild", fake_rebuild_llm_provider
    )

    await controller.apply_providers(pending)

    result = _settings_result(controller)
    assert result is not None
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
    assert result.diagnostics == messages.ErrorDiagnostics(
        component="gui_controller",
        operation="apply_order21_order22_order24_provider_full_draft_save",
        code="settings_save_failed",
        category=messages.DIAGNOSTIC_CATEGORY_TRANSACTION,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"surface": "translation_provider"},
    )
    assert [settings.managed_identity.verified_hardware_hash for settings in saved_settings] == [
        None,
        None,
        "pending-hardware-hash",
    ]
    assert controller.settings is not None
    assert controller.settings.llm.concurrency_limit == 3
    assert controller.settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert controller.settings.system_prompt == "provider prompt"
    assert controller.hub.system_prompt == "provider prompt"
    assert controller.settings.managed_identity.verified_hardware_hash is None
    assert controller.settings.managed_identity.verified_hardware_hash_salt_version is None
    logged_text = "\n".join(
        message
        for _level, message in (
            controller._runtime_logging.basic_messages
            + controller._runtime_logging.detailed_messages
        )
    )
    assert raw_failure_text not in repr(result)
    assert raw_failure_text not in logged_text


@pytest.mark.asyncio
async def test_order22_provider_mixed_fallback_failure_degrades_without_raw_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    controller.settings.provider.stt = STTProviderName.DEEPGRAM
    controller.settings.system_prompt = "base prompt"
    controller.hub = DummyHub(stt=object())
    controller._stt_desired = False
    pending = copy.deepcopy(controller.settings)
    pending.provider.stt = STTProviderName.SONIOX
    pending.system_prompt = "draft prompt"
    raw_failure_text = "stt rebuild failed secret-token-must-not-leak"
    saved_settings: list[AppSettings] = []

    def record_saved_settings(_path, settings) -> None:
        saved_settings.append(copy.deepcopy(settings))

    async def fail_rebuild_stt_provider(self) -> None:
        raise RuntimeError(raw_failure_text)

    _patch_settings_save(monkeypatch, record_saved_settings)
    monkeypatch.setattr(GuiController, "_rebuild_stt_provider", fail_rebuild_stt_provider)

    await controller.apply_providers(pending)

    result = _settings_result(controller)
    assert result is not None
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
    assert result.message == messages.UserMessageRef(
        key="settings.mutation.runtime_apply_failed",
        params={"phase": "runtime_apply"},
        severity=messages.SEVERITY_WARNING,
    )
    assert result.diagnostics == messages.ErrorDiagnostics(
        component="gui_controller",
        operation="apply_stt_language_audio_provider_runtime",
        code="provider_runtime_apply_exception",
        category=messages.DIAGNOSTIC_CATEGORY_LIFECYCLE,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"surface": "stt_language_audio"},
    )
    assert raw_failure_text not in repr(result)
    assert [settings.system_prompt for settings in saved_settings] == [
        "base prompt",
        "draft prompt",
    ]
    assert controller.settings.provider.stt == STTProviderName.SONIOX
    assert controller.settings.system_prompt == "draft prompt"


@pytest.mark.asyncio
async def test_order22_provider_mixed_full_draft_save_failure_degrades_and_restores_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller._runtime_logging = RuntimeLoggingSpy()
    controller.settings = AppSettings()
    controller.settings.provider.stt = STTProviderName.DEEPGRAM
    controller.settings.system_prompt = "base prompt"
    controller.hub = DummyHub(stt=object())
    controller.hub.system_prompt = controller.settings.system_prompt
    controller._stt_desired = False
    pending = copy.deepcopy(controller.settings)
    pending.provider.stt = STTProviderName.SONIOX
    pending.system_prompt = "draft prompt"
    save_attempts: list[AppSettings] = []
    runtime_calls: list[str] = []
    raw_failure_text = "provider full draft save failed secret-token-must-not-leak"

    def fail_second_save(_path, incoming: AppSettings) -> None:
        save_attempts.append(copy.deepcopy(incoming))
        if len(save_attempts) == 2:
            raise RuntimeError(raw_failure_text)

    async def record_rebuild_stt_provider(self) -> None:
        _ = self
        runtime_calls.append("rebuild_stt")

    _patch_settings_save(monkeypatch, fail_second_save)
    monkeypatch.setattr(GuiController, "_rebuild_stt_provider", record_rebuild_stt_provider)

    applied = await controller.apply_providers(pending)

    result = _settings_result(controller)
    assert applied is False
    assert result is not None
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED
    assert result.diagnostics == messages.ErrorDiagnostics(
        component="settings_repository",
        operation="save",
        code="settings_save_failed",
        category=messages.DIAGNOSTIC_CATEGORY_TRANSACTION,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"surface": "ui_prompt_clipboard_state"},
    )
    assert [settings.provider.stt for settings in save_attempts] == [
        STTProviderName.SONIOX,
        STTProviderName.SONIOX,
    ]
    assert save_attempts[0].system_prompt == "base prompt"
    assert save_attempts[1].system_prompt == "draft prompt"
    assert controller.settings.provider.stt == STTProviderName.SONIOX
    assert controller.settings.system_prompt == "base prompt"
    assert controller.hub.system_prompt == "base prompt"
    assert runtime_calls == ["rebuild_stt"]
    assert raw_failure_text not in repr(result)
    assert raw_failure_text not in repr(controller._runtime_logging.basic_messages)


@pytest.mark.asyncio
async def test_order22_mixed_settings_direct_fallback_degrades_when_stt_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    controller.settings.provider.stt = STTProviderName.DEEPGRAM
    controller.settings.overlay.show_translation = True
    controller.hub = DummyHub(stt=object())
    controller.hub.source_language = controller.settings.languages.source_language
    controller.hub.target_language = controller.settings.languages.target_language
    controller._stt_desired = True
    controller._last_self_stt_runtime_signature = build_self_stt_runtime_signature(
        controller.settings
    )
    controller._last_peer_stt_runtime_signature = _peer_runtime_signature(
        controller, controller.settings
    )
    controller._last_peer_translation_enabled = controller.settings.ui.peer_translation_enabled
    controller._last_vrc_mic_sync_enabled = controller.settings.osc.vrc_mic_intercept
    pending = copy.deepcopy(controller.settings)
    pending.provider.stt = STTProviderName.SONIOX
    pending.overlay.show_translation = False

    async def unavailable_replace_runtime_stt_provider(self) -> None:
        assert self.hub is not None
        self.hub.stt = None

    _patch_settings_save(monkeypatch, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(GuiController, "_sync_clipboard_watcher", lambda self: asyncio.sleep(0))
    monkeypatch.setattr(
        GuiController,
        "_clear_local_stt_pending_enable_if_provider_switched_away",
        lambda self: None,
    )
    monkeypatch.setattr(
        GuiController, "_replace_runtime_stt_provider", unavailable_replace_runtime_stt_provider
    )

    await controller.apply_settings(pending)

    result = _settings_result(controller)
    assert result is not None
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
    assert result.diagnostics == messages.ErrorDiagnostics(
        component="gui_controller",
        operation="apply_stt_language_audio_runtime",
        code="stt_language_audio_runtime_unavailable",
        category=messages.DIAGNOSTIC_CATEGORY_LIFECYCLE,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"surface": "stt_language_audio"},
    )
    assert controller.hub.stt is None


@pytest.mark.asyncio
async def test_order22_qwen_historical_low_latency_change_does_not_rebuild_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.QWEN
    controller.settings.stt.low_latency_mode = False
    controller.hub = DummyHub(llm=object(), stt=object())
    controller.hub.source_language = controller.settings.languages.source_language
    controller.hub.target_language = controller.settings.languages.target_language
    controller.hub.low_latency_mode = controller.settings.stt.low_latency_mode
    controller._last_self_stt_runtime_signature = build_self_stt_runtime_signature(
        controller.settings
    )
    controller._last_peer_stt_runtime_signature = _peer_runtime_signature(
        controller, controller.settings
    )
    controller._last_peer_translation_enabled = controller.settings.ui.peer_translation_enabled
    controller._last_vrc_mic_sync_enabled = controller.settings.osc.vrc_mic_intercept
    pending = copy.deepcopy(controller.settings)
    pending.stt.low_latency_mode = True

    rebuild_calls: list[bool] = []

    async def unavailable_rebuild_llm_provider(self) -> None:
        assert self.hub is not None
        rebuild_calls.append(self.hub.low_latency_mode)

    _patch_settings_save(monkeypatch, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(GuiController, "_sync_clipboard_watcher", lambda self: asyncio.sleep(0))
    monkeypatch.setattr(
        GuiController,
        "_clear_local_stt_pending_enable_if_provider_switched_away",
        lambda self: None,
    )
    monkeypatch.setattr(
        provider_runtime_apply_module.LlmProviderRebuildOwner,
        "rebuild",
        unavailable_rebuild_llm_provider,
    )

    await controller.apply_settings(pending)

    assert rebuild_calls == []
    assert controller.hub.llm is not None
    assert controller.hub.low_latency_mode is True
    assert controller.settings.stt.low_latency_mode is True


@pytest.mark.asyncio
async def test_order22_qwen_historical_false_cannot_restore_non_fast_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.QWEN
    controller.settings.stt.low_latency_mode = False
    controller.hub = DummyHub(llm=object(), stt=object())
    controller.hub.source_language = controller.settings.languages.source_language
    controller.hub.target_language = controller.settings.languages.target_language
    controller.hub.low_latency_mode = controller.settings.stt.low_latency_mode
    controller._last_self_stt_runtime_signature = build_self_stt_runtime_signature(
        controller.settings
    )
    controller._last_peer_stt_runtime_signature = _peer_runtime_signature(
        controller, controller.settings
    )
    controller._last_peer_translation_enabled = controller.settings.ui.peer_translation_enabled
    controller._last_vrc_mic_sync_enabled = controller.settings.osc.vrc_mic_intercept
    pending = copy.deepcopy(controller.settings)
    pending.stt.low_latency_mode = False
    pending.stt.low_latency_merge_gap_ms += 1
    rebuild_markers: list[bool] = []

    async def fail_then_recover_rebuild_llm_provider(self) -> None:
        assert self.hub is not None
        rebuild_markers.append(self.hub.low_latency_mode)

    _patch_settings_save(monkeypatch, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(GuiController, "_sync_clipboard_watcher", lambda self: asyncio.sleep(0))
    monkeypatch.setattr(
        GuiController,
        "_clear_local_stt_pending_enable_if_provider_switched_away",
        lambda self: None,
    )
    monkeypatch.setattr(
        provider_runtime_apply_module.LlmProviderRebuildOwner,
        "rebuild",
        fail_then_recover_rebuild_llm_provider,
    )

    await controller.apply_settings(pending)

    first_result = _settings_result(controller)
    assert first_result is not None
    assert (
        first_result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    )
    assert rebuild_markers == []
    assert controller.hub.llm is not None
    assert controller.hub.low_latency_mode is True

    await controller.apply_settings(copy.deepcopy(controller.settings))

    assert rebuild_markers == []
    assert controller.hub.low_latency_mode is True


@pytest.mark.asyncio
async def test_order23_apply_settings_routes_overlay_osc_output_patch_through_settings_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    controller.settings.ui.overlay_enabled = False
    controller.settings.ui.peer_translation_enabled = False
    service = RecordingSettingsMutationService()
    controller.settings_mutation_service = service

    pending = copy.deepcopy(controller.settings)
    pending.overlay.target = OVERLAY_TARGET_DESKTOP
    pending.overlay.show_translation = False
    pending.overlay.show_peer_original = False
    pending.overlay.calibration = OverlayCalibration(distance=1.7, offset_x=0.4)
    pending.overlay.desktop_flet.size_preset = "large"
    pending.overlay.desktop_flet.position.x = 12
    pending.overlay.desktop_flet.position.y = 34
    pending.overlay.desktop_flet.visual.background_alpha = 0.42
    pending.osc.host = "192.0.2.44"
    pending.osc.port = 9001
    pending.osc.chatbox_address = "/chatbox/custom"
    pending.osc.chatbox_send = False
    pending.osc.chatbox_clear = True
    pending.osc.chatbox_max_chars = 120
    pending.osc.vrc_mic_intercept = True
    pending.osc.chatbox_include_source = True
    pending.ui.overlay_enabled = True
    pending.ui.peer_translation_enabled = True

    def fail_direct_save(*_args, **_kwargs) -> None:
        raise AssertionError("direct save should not persist routed order23 settings")

    _patch_settings_save(monkeypatch, fail_direct_save)

    await controller.apply_settings(pending)

    assert len(service.requests) == 1
    request = service.requests[0]
    assert request.reason == settings_mutation.SETTINGS_MUTATION_SURFACE_OVERLAY_OSC_OUTPUT
    assert request.values == {
        "overlay.target": OVERLAY_TARGET_DESKTOP,
        "overlay.show_translation": False,
        "overlay.show_peer_original": False,
        "overlay.calibration.offset_x": 0.4,
        "overlay.calibration.distance": 1.7,
        "overlay.desktop_flet.size_preset": "large",
        "overlay.desktop_flet.position.x": 12,
        "overlay.desktop_flet.position.y": 34,
        "overlay.desktop_flet.visual.background_alpha": 0.42,
        "osc.host": "192.0.2.44",
        "osc.port": 9001,
        "osc.chatbox_address": "/chatbox/custom",
        "osc.chatbox_send": False,
        "osc.chatbox_clear": True,
        "osc.chatbox_max_chars": 120,
        "osc.vrc_mic_intercept": True,
        "osc.chatbox_include_source": True,
    }
    assert "ui.overlay_enabled" not in request.values
    assert "ui.peer_translation_enabled" not in request.values
    assert "active_chatbox_channel" not in request.values
    assert controller.settings.overlay.target == OVERLAY_TARGET_DESKTOP
    assert controller.settings.osc.chatbox_include_source is True
    assert controller.settings.ui.overlay_enabled is False
    assert controller.settings.ui.peer_translation_enabled is False


@pytest.mark.asyncio
async def test_order23_apply_settings_restore_baseline_on_live_settings_view_save_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller._runtime_logging = RuntimeLoggingSpy()
    controller.settings = AppSettings()
    controller.settings.overlay.show_translation = True
    controller.settings.osc.chatbox_include_source = False
    controller._settings_projection().remember_order23(controller.settings)
    raw_failure_text = "order23 save failed secret-token-must-not-leak"

    # SettingsView mutates the loaded AppSettings object in place before emitting a copy.
    controller.settings.overlay.show_translation = False
    controller.settings.osc.chatbox_include_source = True
    pending = copy.deepcopy(controller.settings)

    def fail_save_settings(_path, _settings) -> None:
        raise RuntimeError(raw_failure_text)

    _patch_settings_save(monkeypatch, fail_save_settings)

    await controller.apply_settings(pending)

    result = _settings_result(controller)
    assert result is not None
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED
    assert result.diagnostics == messages.ErrorDiagnostics(
        component="settings_repository",
        operation="save",
        code="settings_save_failed",
        category=messages.DIAGNOSTIC_CATEGORY_TRANSACTION,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"surface": "overlay_osc_output"},
    )
    assert controller.settings.overlay.show_translation is True
    assert controller.settings.osc.chatbox_include_source is False
    assert raw_failure_text not in repr(result)
    assert raw_failure_text not in repr(controller._runtime_logging.basic_messages)


@pytest.mark.asyncio
async def test_order23_first_live_settings_view_mutation_after_sync_uses_baseline_and_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_view = DummySettingsView()
    controller = _make_controller(app=SimpleNamespace(view_settings=settings_view))
    controller._runtime_logging = RuntimeLoggingSpy()
    controller.settings = AppSettings()
    controller.settings.overlay.show_translation = True
    controller.settings.osc.chatbox_include_source = False
    raw_failure_text = "order23 synced baseline save failed secret-token-must-not-leak"

    controller._sync_ui_from_settings()

    assert settings_view.calls

    # SettingsView mutates the loaded AppSettings object in place before emitting a copy.
    controller.settings.overlay.show_translation = False
    controller.settings.osc.chatbox_include_source = True
    pending = copy.deepcopy(controller.settings)

    def fail_save_settings(_path, _settings) -> None:
        raise RuntimeError(raw_failure_text)

    _patch_settings_save(monkeypatch, fail_save_settings)

    await controller.apply_settings(pending)

    result = _settings_result(controller)
    assert result is not None
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED
    assert result.diagnostics == messages.ErrorDiagnostics(
        component="settings_repository",
        operation="save",
        code="settings_save_failed",
        category=messages.DIAGNOSTIC_CATEGORY_TRANSACTION,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"surface": "overlay_osc_output"},
    )
    assert controller.settings.overlay.show_translation is True
    assert controller.settings.osc.chatbox_include_source is False
    assert raw_failure_text not in repr(result)
    assert raw_failure_text not in repr(controller._runtime_logging.basic_messages)


@pytest.mark.asyncio
async def test_order23_apply_settings_runtime_failure_degrades_without_rollback_or_raw_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingOverlayPresenter:
        async def update_display_preferences(self, **_kwargs) -> None:
            raise RuntimeError("overlay runtime failed secret-token-must-not-leak")

    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    controller.hub = DummyHub()
    _overlay_owner(controller).state = "connected"
    _attach_overlay_presenter(controller, FailingOverlayPresenter())
    pending = copy.deepcopy(controller.settings)
    pending.overlay.show_translation = False
    saved_settings: list[AppSettings] = []

    def record_saved_settings(_path, settings) -> None:
        saved_settings.append(copy.deepcopy(settings))

    _patch_settings_save(monkeypatch, record_saved_settings)

    await controller.apply_settings(pending)

    result = _settings_result(controller)
    assert result is not None
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
    assert result.message == messages.UserMessageRef(
        key="settings.mutation.runtime_apply_failed",
        params={"phase": "runtime_apply"},
        severity=messages.SEVERITY_WARNING,
    )
    assert result.diagnostics == messages.ErrorDiagnostics(
        component="gui_controller",
        operation="apply_overlay_osc_output_runtime",
        code="overlay_osc_output_runtime_apply_exception",
        category=messages.DIAGNOSTIC_CATEGORY_LIFECYCLE,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"surface": "overlay_osc_output"},
    )
    assert "secret-token-must-not-leak" not in repr(result)
    assert len(saved_settings) == 1
    assert saved_settings[0].overlay.show_translation is False
    assert controller.settings.overlay.show_translation is False


@pytest.mark.asyncio
async def test_order23_runtime_only_overlay_active_flags_are_not_routed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    service = RecordingSettingsMutationService()
    controller.settings_mutation_service = service
    begin_calls: list[bool] = []
    saved_settings: list[AppSettings] = []

    async def fake_begin_overlay_start(self) -> None:
        begin_calls.append(True)
        self.state = "starting"

    def record_saved_settings(_path, settings) -> None:
        saved_settings.append(copy.deepcopy(settings))

    _patch_settings_save(monkeypatch, record_saved_settings)
    monkeypatch.setattr(
        overlay_application_module.OverlayApplicationOwner,
        "begin_start",
        fake_begin_overlay_start,
    )

    pending = copy.deepcopy(controller.settings)
    pending.ui.overlay_enabled = True
    pending.ui.peer_translation_enabled = True

    await controller.apply_settings(pending)

    assert service.requests == []
    assert begin_calls == [True]
    assert controller.settings.ui.overlay_enabled is True
    assert controller.settings.ui.peer_translation_enabled is True
    assert saved_settings and to_dict(saved_settings[0])["ui"].get("overlay_enabled") is None


@pytest.mark.asyncio
async def test_apply_providers_preserves_current_languages_while_applying_provider_and_prompt_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    controller.settings.openrouter.selection_alias = OpenRouterSelectionAlias.GEMMA4_BYOK
    controller.settings.translation.fallback = TranslationFallbackSettings(
        enabled=True,
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.OPENROUTER,
    )
    controller.settings.languages.source_language = "fr"
    controller.settings.languages.target_language = "de"
    controller.settings.languages.peer_source_language = "ja"
    controller.settings.languages.peer_target_language = "it"
    controller.settings.languages.recent_source_languages = ["fr", "ko"]
    controller.settings.languages.recent_target_languages = ["de", "en"]
    controller.settings.stt.low_latency_mode = True
    controller.settings.stt.low_latency_vad_hangover_ms = 650
    controller.settings.desktop_audio.vad_hangover_ms = 950
    controller.hub = DummyHub()
    controller.hub.source_language = "es"
    controller.hub.target_language = "pt"
    controller.hub.peer_source_language = "zh-CN"
    controller.hub.peer_target_language = "nl"
    controller._stt_desired = False
    controller._last_self_stt_provider_signature = build_self_stt_provider_signature(
        controller.settings
    )
    controller._last_peer_stt_provider_signature = _peer_provider_signature(
        controller, controller.settings
    )
    controller._last_llm_provider_signature = controller._build_llm_provider_signature(
        controller.settings
    )
    calls: list[str] = []

    pending = AppSettings()
    pending.languages.source_language = "ko"
    pending.languages.target_language = "en"
    pending.languages.peer_source_language = ""
    pending.languages.peer_target_language = "ja"
    pending.provider.stt = STTProviderName.SONIOX
    pending.provider.peer_stt = STTProviderName.SONIOX
    pending.provider.llm = LLMProviderName.OPENROUTER
    pending.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    pending.openrouter.selection_alias = OpenRouterSelectionAlias.QWEN35_FLASH_MANAGED
    pending.translation.fallback = TranslationFallbackSettings(
        enabled=True,
        model=TranslationModel.GEMMA4,
        connection=TranslationConnection.OPENROUTER,
    )
    pending.openrouter.routing_mode = OpenRouterRoutingMode.LATENCY
    pending.system_prompt = "draft prompt"
    pending.system_prompts = {"openrouter": "draft prompt"}

    _patch_settings_save(monkeypatch, lambda *_args, **_kwargs: None)

    async def fake_rebuild_stt_provider(self) -> None:
        calls.append("rebuild_stt")

    async def fake_refresh_peer_stt_runtime(self) -> None:
        calls.append("peer")

    async def fake_rebuild_llm_provider(self) -> None:
        calls.append("llm")

    monkeypatch.setattr(GuiController, "_rebuild_stt_provider", fake_rebuild_stt_provider)
    monkeypatch.setattr(GuiController, "_refresh_peer_stt_runtime", fake_refresh_peer_stt_runtime)
    monkeypatch.setattr(
        provider_runtime_apply_module.LlmProviderRebuildOwner, "rebuild", fake_rebuild_llm_provider
    )

    await controller.apply_providers(pending)

    assert controller.settings.languages.source_language == "es"
    assert controller.settings.languages.target_language == "pt"
    assert controller.settings.languages.peer_source_language == "zh-CN"
    assert controller.settings.languages.peer_target_language == "nl"
    assert controller.settings.languages.recent_source_languages == ["fr", "ko"]
    assert controller.settings.languages.recent_target_languages == ["de", "en"]
    assert controller.hub.source_language == "es"
    assert controller.hub.target_language == "pt"
    assert controller.hub.peer_source_language == "zh-CN"
    assert controller.hub.peer_target_language == "nl"
    assert controller.settings.provider.stt == STTProviderName.SONIOX
    assert controller.settings.provider.peer_stt == STTProviderName.SONIOX
    assert controller.settings.provider.llm == LLMProviderName.OPENROUTER
    assert controller.settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert (
        controller.settings.openrouter.selection_alias
        == OpenRouterSelectionAlias.QWEN35_FLASH_MANAGED
    )
    assert controller.settings.translation.fallback == pending.translation.fallback
    assert controller.settings.openrouter.routing_mode == OpenRouterRoutingMode.LATENCY
    assert controller.settings.system_prompt == "draft prompt"
    assert controller.settings.system_prompts == {}
    assert controller.hub.hangover_s == 0.65
    assert controller.hub.peer_hangover_s == 0.95
    assert calls == ["llm", "peer", "rebuild_stt"]


@pytest.mark.asyncio
async def test_on_dashboard_language_change_routes_self_and_peer_updates_through_shared_controller_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    controller.settings.languages.peer_source_language = "zh-CN"
    controller.settings.languages.peer_target_language = "ja"
    captured: list[AppSettings] = []

    async def fake_apply_settings(
        self: SettingsApplicationOwner,
        settings: AppSettings,
    ) -> bool:
        captured.append(settings)
        return True

    monkeypatch.setattr(SettingsApplicationOwner, "apply", fake_apply_settings)

    await controller.on_dashboard_language_change(
        _language_selection_change(
            source_code="fr",
            target_code="de",
            peer_source_code="",
            peer_target_code="it",
            recent_source_codes=("fr",),
            recent_target_codes=("de",),
        )
    )

    assert controller.settings.languages.source_language == "ko"
    assert controller.settings.languages.target_language == "en"
    assert controller.settings.languages.peer_source_language == "zh-CN"
    assert controller.settings.languages.peer_target_language == "ja"
    assert len(captured) == 1
    assert captured[0].languages.source_language == "fr"
    assert captured[0].languages.target_language == "de"
    assert captured[0].languages.peer_source_language == ""
    assert captured[0].languages.peer_target_language == "it"
    assert captured[0].languages.recent_source_languages == ["fr"]
    assert captured[0].languages.recent_target_languages == ["de"]


@pytest.mark.asyncio
async def test_dashboard_language_change_commits_languages_and_recents_in_one_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    requests: list[settings_mutation.SettingsMutationRequest] = []
    saves: list[AppSettings] = []
    original_mutate = settings_mutation.SettingsMutationService.mutate

    async def capture_mutate(self, request):
        requests.append(request)
        return await original_mutate(self, request)

    monkeypatch.setattr(settings_mutation.SettingsMutationService, "mutate", capture_mutate)
    _patch_settings_save(
        monkeypatch,
        lambda _path, settings: saves.append(copy.deepcopy(settings)),
    )

    await controller.on_dashboard_language_change(
        _language_selection_change(
            source_code="ja",
            target_code="fr",
            peer_source_code="en",
            peer_target_code="ko",
            recent_source_codes=("ja", "ko"),
            recent_target_codes=("fr", "en"),
        )
    )

    assert len(requests) == 1
    assert requests[0].reason == settings_mutation.SETTINGS_MUTATION_SURFACE_STT_LANGUAGE_AUDIO
    assert requests[0].values["languages.source_language"] == "ja"
    assert requests[0].values["languages.target_language"] == "fr"
    assert list(requests[0].values["languages.recent_source_languages"]) == ["ja", "ko"]
    assert list(requests[0].values["languages.recent_target_languages"]) == ["fr", "en"]
    assert len(saves) == 1
    assert saves[0].languages.source_language == "ja"
    assert saves[0].languages.target_language == "fr"
    assert saves[0].languages.recent_source_languages[:2] == ["ja", "ko"]
    assert saves[0].languages.recent_target_languages[:2] == ["fr", "en"]


@pytest.mark.asyncio
async def test_on_dashboard_language_change_preserves_explicit_peer_override_when_self_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    controller.settings.languages.peer_source_language = "ja"
    controller.settings.languages.peer_target_language = "fr"
    captured: list[AppSettings] = []

    async def fake_apply_settings(
        self: SettingsApplicationOwner,
        settings: AppSettings,
    ) -> bool:
        captured.append(settings)
        return True

    monkeypatch.setattr(SettingsApplicationOwner, "apply", fake_apply_settings)

    await controller.on_dashboard_language_change(
        _language_selection_change(
            source_code="ja",
            target_code="en",
            peer_source_code="ja",
            peer_target_code="fr",
        )
    )

    assert len(captured) == 1
    assert captured[0].languages.source_language == "ja"
    assert captured[0].languages.target_language == "en"
    assert captured[0].languages.peer_source_language == "ja"
    assert captured[0].languages.peer_target_language == "fr"


@pytest.mark.asyncio
async def test_on_dashboard_language_change_persists_explicit_automatic_peer_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    controller.settings.languages.peer_source_language = "ja"
    _patch_settings_save(monkeypatch, lambda *_args, **_kwargs: None)

    await controller.on_dashboard_language_change(
        _language_selection_change(
            source_code="ko",
            target_code="en",
            peer_source_code="ja",
            peer_target_code="fr",
            peer_source_mode="auto",
        )
    )

    assert controller.settings.languages.peer_source_language == "ja"
    assert controller.settings.languages.peer_source_mode == "auto"
    assert controller.vnext_settings is not None
    assert controller.vnext_settings.intent.languages.peer_source_mode == "auto"


@pytest.mark.asyncio
async def test_dashboard_peer_language_change_refreshes_peer_translation_pipeline_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_view = DummySettingsView()
    controller = _make_controller(
        app=SimpleNamespace(
            view_dashboard=DummyDashboard(),
            view_settings=settings_view,
        )
    )
    controller.settings = AppSettings()
    controller.hub = DummyHub()
    controller._last_self_stt_runtime_signature = build_self_stt_runtime_signature(
        controller.settings
    )
    controller._last_peer_stt_runtime_signature = _peer_runtime_signature(
        controller, controller.settings
    )
    controller._last_peer_translation_enabled = controller.settings.ui.peer_translation_enabled
    refreshed: list[str] = []

    _patch_settings_save(monkeypatch, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        GuiController,
        "_clear_local_stt_pending_enable_if_provider_switched_away",
        lambda self: None,
    )

    async def fake_refresh_peer_stt_runtime(self) -> None:
        refreshed.append("peer")

    async def fake_replace_runtime_stt_provider(self) -> None:
        raise AssertionError("self STT runtime should not restart for peer-only change")

    monkeypatch.setattr(GuiController, "_refresh_peer_stt_runtime", fake_refresh_peer_stt_runtime)
    monkeypatch.setattr(
        GuiController, "_replace_runtime_stt_provider", fake_replace_runtime_stt_provider
    )

    await controller.on_dashboard_language_change(
        _language_selection_change(
            source_code="ko",
            target_code="en",
            peer_source_code="ja",
            peer_target_code="fr",
        )
    )

    assert refreshed == ["peer"]
    assert controller.hub.peer_source_language == "ja"
    assert controller.hub.peer_target_language == "fr"
    assert len(settings_view.calls) == 1
    reloaded_settings, _config_path, preserve_custom_vocab_draft = settings_view.calls[0]
    assert reloaded_settings.languages.peer_source_language == "ja"
    assert reloaded_settings.languages.peer_target_language == "fr"
    assert preserve_custom_vocab_draft is True


@pytest.mark.asyncio
async def test_apply_providers_republishes_overlay_peer_contract_after_peer_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contracts = []
    app = SimpleNamespace(
        view_dashboard=SimpleNamespace(
            set_overlay_peer_contract=contracts.append,
        )
    )
    controller = _make_controller(app=app)
    controller.settings = AppSettings()
    controller.settings.ui.overlay_enabled = True
    controller.settings.ui.peer_translation_enabled = True
    controller.settings.ui.peer_translation_eula_accepted = True
    controller.hub = DummyHub(peer_stt=None)
    _overlay_owner(controller).state = "connected"
    updated = AppSettings()
    updated.ui.overlay_enabled = True
    updated.ui.peer_translation_enabled = True
    updated.ui.peer_translation_eula_accepted = True
    updated.provider.peer_stt = STTProviderName.SONIOX

    _patch_settings_save(monkeypatch, lambda *_args, **_kwargs: None)

    async def fake_refresh_peer_stt_runtime(self) -> None:
        assert self.hub is not None
        self.hub.peer_stt = object()

    monkeypatch.setattr(GuiController, "_refresh_peer_stt_runtime", fake_refresh_peer_stt_runtime)

    await controller.apply_providers(updated)

    assert contracts
    assert contracts[-1].peer.state == "on"
    assert contracts[-1].peer.warning_reason is None


@pytest.mark.asyncio
async def test_apply_providers_clears_local_qwen_pending_enable_after_switch_away(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    controller.settings.provider.stt = STTProviderName.LOCAL_QWEN
    controller.hub = DummyHub()
    controller._local_stt_pending_enable_after_install = True

    updated = AppSettings()
    updated.provider.stt = STTProviderName.DEEPGRAM

    _patch_settings_save(monkeypatch, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        provider_runtime_apply_module.LlmProviderRebuildOwner,
        "rebuild",
        lambda self: asyncio.sleep(0),
    )
    monkeypatch.setattr(GuiController, "_refresh_peer_stt_runtime", lambda self: asyncio.sleep(0))
    monkeypatch.setattr(GuiController, "_rebuild_stt_provider", lambda self: asyncio.sleep(0))

    await controller.apply_providers(updated)

    assert controller._local_stt_pending_enable_after_install is False


@pytest.mark.asyncio
async def test_apply_providers_switch_to_managed_blocks_concurrent_toggle_from_using_old_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dash = DummyDashboard()
    controller = _make_controller(app=SimpleNamespace(view_dashboard=dash))
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK

    close_started = asyncio.Event()
    release_close = asyncio.Event()

    class SlowClosingLlm:
        async def close(self) -> None:
            close_started.set()
            await release_close.wait()

    controller.hub = DummyHub(llm=SlowClosingLlm())

    updated = AppSettings()
    updated.provider.llm = LLMProviderName.OPENROUTER
    updated.openrouter.selected_source = OpenRouterCredentialSource.MANAGED

    _patch_settings_save(monkeypatch, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(controller_module, "create_secret_store", lambda *_a, **_k: object())
    monkeypatch.setattr(controller_module, "create_llm_provider", lambda *_a, **_k: object())

    apply_task = asyncio.create_task(controller.apply_providers(updated))
    await close_started.wait()

    await controller.set_translation_enabled(True)

    assert controller.hub.translation_enabled is False
    assert controller.hub.clear_context_calls == 0
    assert dash.translation_enabled is False

    release_close.set()
    await apply_task


@pytest.mark.asyncio
async def test_apply_providers_splits_qwen_region_refresh_by_active_consumers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _make_controller(app=SimpleNamespace(view_dashboard=DummyDashboard()))
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.QWEN
    controller.settings.provider.stt = STTProviderName.QWEN_ASR
    controller.settings.provider.peer_stt = STTProviderName.QWEN_ASR
    controller.hub = DummyHub()
    controller._stt_desired = True
    calls: list[str] = []

    updated = AppSettings()
    updated.provider.llm = LLMProviderName.QWEN
    updated.provider.stt = STTProviderName.QWEN_ASR
    updated.provider.peer_stt = STTProviderName.QWEN_ASR
    updated.qwen.region = QwenRegion.SINGAPORE

    _patch_settings_save(monkeypatch, lambda *_args, **_kwargs: None)

    async def fake_rebuild_llm_provider(self) -> None:
        calls.append("llm")

    async def fake_refresh_peer_stt_runtime(self) -> None:
        calls.append("peer")

    async def fake_replace_runtime_stt_provider(self) -> None:
        calls.append("replace")

    monkeypatch.setattr(
        provider_runtime_apply_module.LlmProviderRebuildOwner, "rebuild", fake_rebuild_llm_provider
    )
    monkeypatch.setattr(GuiController, "_refresh_peer_stt_runtime", fake_refresh_peer_stt_runtime)
    monkeypatch.setattr(
        GuiController, "_replace_runtime_stt_provider", fake_replace_runtime_stt_provider
    )
    await controller.apply_providers(updated)

    assert calls.count("llm") == 1
    assert calls.count("peer") == 1
    assert calls.count("replace") == 1
    assert not any(call.startswith("pipeline:") for call in calls)


def test_load_or_init_settings_loads_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.ui.locale = "ja"
    save_settings(path, settings)
    controller = _make_controller(app=SimpleNamespace())
    controller.config_path = path

    loaded = controller._load_or_init_settings(path)

    assert loaded.ui.locale == "ja"
    assert controller.vnext_settings is not None


def test_load_or_init_settings_creates_default_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested" / "settings.json"
    controller = _make_controller(app=SimpleNamespace())
    controller.config_path = path

    loaded = controller._load_or_init_settings(path)
    shared_prompt = load_prompt_for_provider("gemini")

    assert isinstance(loaded, AppSettings)
    assert loaded.ui.overlay_enabled is False
    assert loaded.system_prompt == shared_prompt
    assert loaded.system_prompts == {}
    assert path.parent.exists() is True
    reloaded = controller._get_settings_owner().persistence.load_active(path)
    assert reloaded.compatibility_settings.ui.overlay_enabled is False
    assert reloaded.compatibility_settings.system_prompt == shared_prompt
    assert reloaded.compatibility_settings.system_prompts == {}


def test_create_managed_openrouter_release_service_uses_http_broker_client_and_raw_fingerprint_provider() -> (
    None
):
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.settings.openrouter.broker_base_url = "https://broker.example.test/"

    service = controller._create_managed_openrouter_release_service(secrets=DummySecrets({}))

    assert isinstance(service, ManagedOpenRouterReleaseService)
    assert isinstance(service.client, HttpManagedOpenRouterBrokerClient)
    assert service.client.base_url == "https://broker.example.test"
    assert (
        service.raw_hardware_fingerprint_provider is controller_module.get_raw_hardware_fingerprint
    )


def test_create_managed_openrouter_release_service_degrades_to_unavailable_client_for_invalid_broker_base_url() -> (
    None
):
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.provider.llm = LLMProviderName.OPENROUTER
    controller.settings.openrouter.selected_source = OpenRouterCredentialSource.MANAGED
    controller.settings.openrouter.broker_base_url = "https://broker.example.test/prefix"

    service = controller._create_managed_openrouter_release_service(secrets=DummySecrets({}))

    assert isinstance(service, ManagedOpenRouterReleaseService)
    assert isinstance(service.client, UnavailableManagedOpenRouterReleaseClient)


@pytest.mark.asyncio
async def test_rebuild_stt_provider_logs_only_failure_when_owned_replacement_fails() -> None:
    dash = DummyDashboard()
    controller = _make_controller(app=SimpleNamespace(view_dashboard=dash))
    controller._runtime_logging = RuntimeLoggingSpy()
    controller.settings = AppSettings()
    controller.settings.provider.stt = STTProviderName.DEEPGRAM
    previous_stt = object()
    controller.hub = DummyHub(stt=previous_stt)

    async def failed_replacement(request: object, *, start: bool):
        _ = request, start
        return SimpleNamespace(status="failed")

    controller.hub.replace_stt_provider_request = failed_replacement

    await controller._rebuild_stt_provider()

    assert controller.hub.stt is previous_stt
    assert dash.stt_needs_key is True
    assert dash.stt_enabled is False
    assert controller._runtime_logging.basic_messages == [
        (logging.ERROR, "STT backend not available")
    ]


@pytest.mark.asyncio
async def test_stop_closes_managed_openrouter_release_service() -> None:
    controller = _make_controller(app=SimpleNamespace())
    service = DummyManagedReleaseService(
        ManagedOpenRouterReleaseResult(
            behavior=ManagedOpenRouterReleaseBehavior.READY,
            message_key="managed_release.ready",
        )
    )
    controller._managed_openrouter_release_service = service

    await controller.stop()

    assert service.close_calls == 1
    assert controller._managed_openrouter_release_service is None


def test_overlay_calibration_controls_follow_apply_cancel_contract() -> None:
    controller = _make_controller(app=SimpleNamespace())

    controller.begin_overlay_calibration()
    controller.set_overlay_calibration_field("distance", 1.2)
    controller.cancel_overlay_calibration()

    assert controller.overlay_calibration.distance != 1.2

    controller.begin_overlay_calibration()
    controller.set_overlay_calibration_field("distance", 1.2)
    controller.apply_overlay_calibration()

    assert controller.overlay_calibration.distance == 1.2


def test_apply_overlay_calibration_uses_page_run_task_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePage:
        def __init__(self) -> None:
            self.tasks: list[object] = []

        def run_task(self, coro_fn) -> None:
            self.tasks.append(coro_fn)

    def fail_direct_save(*_args, **_kwargs) -> None:
        raise AssertionError("overlay calibration must not use direct settings save")

    _patch_settings_save(monkeypatch, fail_direct_save)

    page = FakePage()
    controller = GuiController(
        page=page,
        app=_presentation(SimpleNamespace(), page=page),
        config_path=Path("settings.json"),
    )
    controller.settings = AppSettings()
    _overlay_owner(controller).state = "connected"
    service = RecordingSettingsMutationService()
    controller.settings_mutation_service = service
    bridge = FakeOverlayBridge(session_token="token")
    _attach_overlay_bridge(controller, bridge)
    _attach_overlay_presenter(
        controller,
        OverlayPresenter(
            bridge=bridge,
            calibration=controller.overlay_calibration.copy(),
            clock=controller.clock,
        ),
    )

    controller.begin_overlay_calibration()
    controller.set_overlay_calibration_field("offset_x", 0.25)
    applied = controller.apply_overlay_calibration()

    assert applied.offset_x == 0.25
    assert controller.overlay_calibration.offset_x == 0.25
    assert controller.settings.overlay.calibration.offset_x == 0.0
    assert len(page.tasks) == 2

    for task in list(page.tasks):
        asyncio.run(task())

    assert len(service.requests) == 1
    assert (
        service.requests[0].reason == settings_mutation.SETTINGS_MUTATION_SURFACE_OVERLAY_OSC_OUTPUT
    )
    assert service.requests[0].values == {"overlay.calibration.offset_x": 0.25}
    assert controller.settings.overlay.calibration.offset_x == 0.25
    runtime = _overlay_owner(controller).runtime
    assert runtime is not None
    assert runtime.bridge.snapshots[-1].calibration.offset_x == 0.25


def test_apply_overlay_calibration_without_page_run_task_skips_persistence_and_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_direct_save(*_args, **_kwargs) -> None:
        raise AssertionError("overlay calibration must not use direct settings save")

    _patch_settings_save(monkeypatch, fail_direct_save)

    controller = _make_controller(app=SimpleNamespace())
    controller._runtime_logging = RuntimeLoggingSpy(detailed_enabled=True)
    controller.settings = AppSettings()
    service = RecordingSettingsMutationService()
    controller.settings_mutation_service = service

    controller.begin_overlay_calibration()
    controller.set_overlay_calibration_field("distance", 1.2)
    applied = controller.apply_overlay_calibration()

    assert applied.distance == 1.2
    assert controller.overlay_calibration.distance == 1.2
    assert controller.settings.overlay.calibration.distance == 1.1
    assert service.requests == []
    messages = [message for _level, message in controller._runtime_logging.detailed_messages]
    assert any(
        "[Overlay] Calibration persistence skipped reason=page_run_task_unavailable" in message
        for message in messages
    )


def test_schedule_overlay_calibration_emit_preserves_traceback_in_detailed_log() -> None:
    class FailingPage:
        def run_task(self, coro_fn) -> None:
            _ = coro_fn
            raise RuntimeError("boom")

    page = FailingPage()
    controller = GuiController(
        page=page,
        app=_presentation(SimpleNamespace(), page=page),
        config_path=Path("settings.json"),
    )
    controller._runtime_logging = RuntimeLoggingSpy()
    _overlay_owner(controller).state = "connected"
    _attach_overlay_presenter(controller, object())

    controller._get_overlay_calibration_application_owner().schedule_emit()

    assert controller._runtime_logging.basic_messages == []
    assert len(controller._runtime_logging.detailed_messages) == 1
    level, message = controller._runtime_logging.detailed_messages[0]
    assert level == logging.WARNING
    assert "[Overlay] Failed to schedule calibration update via page.run_task" in message
    assert "Traceback (most recent call last):" in message
    assert "RuntimeError: boom" in message


@pytest.mark.asyncio
async def test_apply_overlay_calibration_persists_settings_and_emits_overlay_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePage:
        def __init__(self) -> None:
            self.tasks: list[object] = []

        def run_task(self, coro_fn) -> None:
            self.tasks.append(coro_fn)

    def fail_direct_save(*_args, **_kwargs) -> None:
        raise AssertionError("overlay calibration must not use direct settings save")

    _patch_settings_save(monkeypatch, fail_direct_save)

    page = FakePage()
    controller = GuiController(
        page=page,
        app=_presentation(SimpleNamespace(), page=page),
        config_path=Path("settings.json"),
    )
    controller.settings = AppSettings()
    _overlay_owner(controller).state = "connected"
    service = RecordingSettingsMutationService()
    controller.settings_mutation_service = service
    bridge = FakeOverlayBridge(session_token="token")
    _attach_overlay_bridge(controller, bridge)
    _attach_overlay_presenter(
        controller,
        OverlayPresenter(
            bridge=bridge,
            calibration=controller.overlay_calibration.copy(),
            clock=controller.clock,
        ),
    )

    controller.begin_overlay_calibration()
    controller.set_overlay_calibration_field("distance", 1.2)
    controller.apply_overlay_calibration()

    assert len(page.tasks) == 2

    for task in list(page.tasks):
        await task()

    assert controller.settings.overlay.calibration.distance == 1.2
    assert len(service.requests) == 1
    assert service.requests[0].values == {"overlay.calibration.distance": 1.2}
    runtime = _overlay_owner(controller).runtime
    assert runtime is not None
    assert runtime.bridge.snapshots[-1].calibration.distance == 1.2


@pytest.mark.asyncio
async def test_apply_settings_updates_overlay_presenter_display_preferences() -> None:
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.hub = DummyHub()
    _overlay_owner(controller).state = "connected"
    bridge = FakeOverlayBridge(session_token="token")
    _attach_overlay_bridge(controller, bridge)
    _attach_overlay_presenter(
        controller,
        OverlayPresenter(
            bridge=bridge,
            calibration=controller.overlay_calibration.copy(),
            clock=controller.clock,
            peer_presentation_refresh_burst=True,
        ),
    )

    updated = AppSettings()
    updated.overlay.show_translation = False
    updated.overlay.show_peer_original = False

    await controller.apply_settings(updated)

    runtime = _overlay_owner(controller).runtime
    assert runtime is not None
    presenter = runtime.presenter
    assert presenter.show_translation is False
    assert presenter.show_peer_original is False
    # The product refresh burst is runtime-only, not a settings knob.
    assert presenter.peer_presentation_refresh_burst is True


@pytest.mark.asyncio
async def test_apply_settings_pushes_updated_overlay_snapshot_to_bridge_and_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.ui.overlay_enabled = True
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayBridge.instances) == 1)
    FakeOverlayProcessManager.instances[0].complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")

    presenter = _overlay_runtime(controller).presenter
    assert presenter is not None

    utterance_id = uuid4()
    await presenter.emit(
        SelfTranscriptFinal(
            event_id="self-final",
            seq=1,
            utterance_id=utterance_id,
            channel="self",
            created_at=10.0,
            text="persist me",
            source_language="ko",
            target_language="en",
            is_final=True,
        )
    )

    initial_bridge = FakeOverlayBridge.instances[0]
    assert initial_bridge.snapshots[-1].blocks[0].secondary_enabled is True

    updated = AppSettings()
    updated.ui.overlay_enabled = True
    updated.overlay.show_translation = False
    updated.overlay.show_peer_original = False

    await controller.apply_settings(updated)

    assert initial_bridge.snapshots[-1].blocks[0].secondary_enabled is False

    await _overlay_owner(controller).teardown(preserve_presenter_state=True)
    _overlay_owner(controller).state = "failed"
    await _overlay_owner(controller).begin_start()
    await _wait_until(lambda: len(FakeOverlayBridge.instances) == 2)

    restarted_bridge = FakeOverlayBridge.instances[1]
    assert restarted_bridge.initial_snapshot.blocks[0].secondary_enabled is False


@pytest.mark.asyncio
async def test_self_local_qwen_rebuilds_after_idle_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warmup_calls = 0
    available = False

    class Hub:
        local_asr_provider_runtime = SimpleNamespace(
            snapshot=SimpleNamespace(
                channel_for=lambda channel: SimpleNamespace(
                    phase="ready",
                    model_id=LOCAL_STT_MODEL_ID,
                )
            )
        )

        def has_stt_provider(self, channel: str) -> bool:
            assert channel == "self"
            return available

        async def warmup_stt_channel(self, channel: str) -> None:
            nonlocal warmup_calls
            assert channel == "self"
            warmup_calls += 1

    hub = Hub()
    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.provider.stt = STTProviderName.LOCAL_QWEN
    controller.hub = hub

    async def rebuild(_self) -> None:
        nonlocal available
        available = True

    monkeypatch.setattr(GuiController, "_current_local_stt_runtime_status", lambda _self: "ready")
    monkeypatch.setattr(GuiController, "_rebuild_stt_provider", rebuild)

    assert await controller._ensure_local_stt_ready() is True
    assert warmup_calls == 1


@pytest.mark.asyncio
async def test_apply_settings_pushes_peer_overlay_snapshot_preferences_to_bridge_and_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_overlay_runtime(monkeypatch)
    monkeypatch.setattr(
        SettingsOwner,
        "save_current",
        lambda self, **_kwargs: True,
    )

    controller = _make_controller(app=SimpleNamespace())
    controller.settings = AppSettings()
    controller.settings.ui.overlay_enabled = True
    controller.hub = DummyHub()

    await controller.set_overlay_enabled(True)
    await _wait_until(lambda: len(FakeOverlayBridge.instances) == 1)
    FakeOverlayProcessManager.instances[0].complete_startup()
    await _wait_until(lambda: _overlay_owner(controller).snapshot.state == "connected")

    presenter = _overlay_runtime(controller).presenter
    assert presenter is not None

    utterance_id = uuid4()
    await presenter.emit(
        PeerTranscriptFinal(
            event_id="peer-final",
            seq=1,
            utterance_id=utterance_id,
            channel="peer",
            created_at=10.0,
            text="peer original",
            source_language="en",
            target_language="ko",
            is_final=True,
        )
    )
    await presenter.emit(
        TranslationFinal(
            event_id="peer-translation",
            seq=2,
            utterance_id=utterance_id,
            channel="peer",
            created_at=10.1,
            text="상대 번역",
            source_language="en",
            target_language="ko",
            is_final=True,
            applied_context_mode=None,
        )
    )

    initial_bridge = FakeOverlayBridge.instances[0]
    assert initial_bridge.snapshots[-1].blocks[0].secondary_enabled is True

    updated = AppSettings()
    updated.ui.overlay_enabled = True
    updated.overlay.show_translation = True
    updated.overlay.show_peer_original = False

    await controller.apply_settings(updated)

    assert initial_bridge.snapshots[-1].blocks[0].secondary_enabled is False

    await _overlay_owner(controller).teardown(preserve_presenter_state=True)
    _overlay_owner(controller).state = "failed"
    await _overlay_owner(controller).begin_start()
    await _wait_until(lambda: len(FakeOverlayBridge.instances) == 2)

    restarted_bridge = FakeOverlayBridge.instances[1]
    assert restarted_bridge.initial_snapshot.blocks[0].secondary_enabled is False
