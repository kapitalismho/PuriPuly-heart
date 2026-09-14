from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal, cast

from puripuly_heart.app.ports.translation_diagnostics_runtime import (
    TranslationOverlayDiagnosticsPort,
)
from puripuly_heart.app.ports.translation_output_projection import (
    TranslationOutputProjectionPort,
)
from puripuly_heart.app.ports.ui_models import OverlayPeerPresentationState
from puripuly_heart.app.services.peer_application import PeerApplicationSnapshot
from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.config.resolved import (
    OVERLAY_TARGET_DESKTOP,
    OVERLAY_TARGET_STEAMVR,
    ResolvedOverlayConfig,
)
from puripuly_heart.core.clock import Clock
from puripuly_heart.core.overlay.bridge import OverlayBridge
from puripuly_heart.core.overlay.diagnostics import OverlayDiagnosticsRecorder
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.process import (
    DefaultOverlayProcessRunner,
    DesktopFletOverlayRunner,
    OverlayProcessManager,
    OverlayProcessRunner,
)
from puripuly_heart.core.runtime.overlay import OverlayRuntimeHandle
from puripuly_heart.core.runtime.overlay_session_fallback import (
    OverlaySessionFallbackOwner,
)

from .overlay_generation_start import (
    OverlayGenerationStartDiagnostic,
    OverlayGenerationStartEffects,
    OverlayGenerationStartOwner,
    OverlayGenerationStartRequest,
)
from .overlay_session_transition import (
    OverlaySessionShutdownExecution,
    OverlaySessionStartExecution,
    OverlaySessionStartStatus,
    OverlaySessionTransitionDiagnostic,
    OverlaySessionTransitionOwner,
)

OVERLAY_STARTUP_TIMEOUT_MS = 15000
OVERLAY_SHUTDOWN_GRACE_S = 3.0
OVERLAY_TERMINAL_RESTART_MAX = 3
OVERLAY_TERMINAL_RESTART_BACKOFF_S = 0.05
OVERLAY_TERMINAL_RESTART_WINDOW_S = 60.0
OVERLAY_STEAMVR_FALLBACK_POLICY: Literal["retry_every_enable"] = "retry_every_enable"
OVERLAY_FAILURE_REASONS = frozenset(
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
        "render_failed",
        "openvr_failed",
        "gpu_readiness_late",
        "gpu_readiness_cancelled",
        "gpu_query_failed",
        "gpu_stalled",
        "runtime_disconnected",
        "window_configuration_failed",
        "native_acceptance_timeout",
        "native_owner_unresponsive",
        "unsupported_binary",
        "termination_unconfirmed",
        "window_reveal_lost",
        "window_visibility_unstable",
        "window_identity_failed",
        "window_observation_failed",
        "window_bounds_failed",
        "window_native_ready_failed",
        "runtime_control_invalid",
        "runtime_crashed",
        "shutdown_not_acknowledged",
        "runtime_exit_nonzero",
        "shutdown_forced",
        "shutdown_cleanup_failed",
        "unknown",
    }
)
DESKTOP_STARTUP_RECOVERABLE_REASONS = frozenset(
    {
        "window_reveal_lost",
        "window_visibility_unstable",
        "window_observation_failed",
        "window_bounds_failed",
        "window_native_ready_failed",
        "window_identity_failed",
    }
)
DESKTOP_STARTUP_RECOVERY_MAX_ATTEMPTS = 1


@dataclass(frozen=True, slots=True)
class OverlayApplicationSnapshot:
    state: str
    failure_reason: str | None
    auto_restart_scheduled: bool
    active_target: str | None
    fallback_active: bool
    fallback_policy: Literal["retry_every_enable"]
    recovery_active: bool = False
    recovery_reason: str | None = None


@dataclass(frozen=True, slots=True)
class OverlayApplicationState:
    settings_available: bool
    overlay_intent_enabled: bool
    configured_target: str
    locale: str


OverlayStateProvider = Callable[[], OverlayApplicationState]
OverlayConfigProvider = Callable[[], ResolvedOverlayConfig]
OverlayIntentSink = Callable[[bool], None]
OverlayOutputProvider = Callable[[], TranslationOutputProjectionPort | None]
OverlayDiagnosticsProvider = Callable[
    [],
    TranslationOverlayDiagnosticsPort | None,
]
OverlayPeerSnapshotProvider = Callable[[], PeerApplicationSnapshot]
OverlayEffect = Callable[[], None]
OverlayAsyncEffect = Callable[[], Awaitable[None]]
OverlayPresentationSink = Callable[[OverlayPeerPresentationState | None], None]
OverlayStateSink = Callable[[str, str | None], None]
OverlayFallbackNoticeSink = Callable[[bool], None]
OverlayDetailedLogSink = Callable[[str, int, Exception | None], object]
OverlayBasicLogSink = Callable[[str, int], object]
OverlayCalibrationProvider = Callable[[], OverlayCalibration]
OverlayValueProvider = Callable[[], str]
OverlayDesktopControlsFactory = Callable[[object], list[dict[str, object]]]
OverlayInteractionModeSink = Callable[[str | None], None]
OverlayBoundsControlSink = Callable[[dict[str, object]], None]
OverlayRendererEventConsumer = Callable[
    [asyncio.Queue[dict[str, object]], str],
    Awaitable[None],
]
OverlayTranslationEnabledProvider = Callable[[], bool]


@dataclass(slots=True)
class OverlayApplicationOwner:
    state_provider: OverlayStateProvider = field(repr=False)
    config_provider: OverlayConfigProvider = field(repr=False)
    overlay_intent_sink: OverlayIntentSink = field(repr=False)
    output_provider: OverlayOutputProvider = field(repr=False)
    diagnostics_provider: OverlayDiagnosticsProvider = field(repr=False)
    peer_snapshot_provider: OverlayPeerSnapshotProvider = field(repr=False)
    sync_peer_effective: OverlayEffect = field(repr=False)
    cancel_peer_activation: OverlayEffect = field(repr=False)
    refresh_peer_dependencies: OverlayAsyncEffect = field(repr=False)
    presentation_sink: OverlayPresentationSink = field(repr=False)
    state_sink: OverlayStateSink = field(repr=False)
    fallback_notice_sink: OverlayFallbackNoticeSink = field(repr=False)
    cancel_bounds_persistence: OverlayAsyncEffect = field(repr=False)
    clear_bounds_suppressed: OverlayEffect = field(repr=False)
    calibration_provider: OverlayCalibrationProvider = field(repr=False)
    logging_mode_provider: OverlayValueProvider = field(repr=False)
    log_dir_provider: OverlayValueProvider = field(repr=False)
    desktop_controls_factory: OverlayDesktopControlsFactory = field(repr=False)
    interaction_mode_sink: OverlayInteractionModeSink = field(repr=False)
    bounds_control_sink: OverlayBoundsControlSink = field(repr=False)
    renderer_event_consumer: OverlayRendererEventConsumer = field(repr=False)
    edit_interaction_mode: str
    clock: Clock
    log_basic: OverlayBasicLogSink = field(repr=False)
    log_detailed: OverlayDetailedLogSink = field(repr=False)
    translation_enabled_provider: OverlayTranslationEnabledProvider = field(repr=False)
    _runtime: OverlayRuntimeHandle | None = field(init=False, default=None, repr=False)
    _state: str = field(init=False, default="off", repr=False)
    _failure_reason: str | None = field(init=False, default=None, repr=False)
    _auto_restart_scheduled: bool = field(init=False, default=False, repr=False)
    _terminal_restart_attempts: int = field(init=False, default=0, repr=False)
    _recovering_from_crash: bool = field(init=False, default=False, repr=False)
    _recovery_episode_started_at: float | None = field(init=False, default=None, repr=False)
    _recovery_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _shutting_down: bool = field(init=False, default=False, repr=False)
    _active_target: str | None = field(init=False, default=None, repr=False)
    _ingress_stopped: bool = field(init=False, default=False, repr=False)
    _translation_sync_generation: int = field(init=False, default=0, repr=False)
    _desktop_startup_recovery_attempted: bool = field(init=False, default=False, repr=False)
    _startup_recovery: dict[str, object] | None = field(init=False, default=None, repr=False)
    _last_startup_recovery: dict[str, object] | None = field(init=False, default=None, repr=False)
    _startup_recovery_generation: int = field(init=False, default=0, repr=False)
    _startup_recovery_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _retired_runtimes: dict[OverlayRuntimeHandle, asyncio.Task[bool]] = field(
        init=False, default_factory=dict, repr=False
    )
    _transition_owner: OverlaySessionTransitionOwner = field(init=False, repr=False)
    _generation_owner: OverlayGenerationStartOwner = field(init=False, repr=False)
    _fallback_owner: OverlaySessionFallbackOwner = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._transition_owner = OverlaySessionTransitionOwner(
            diagnostic_sink=self._on_transition_diagnostic,
        )
        self._generation_owner = OverlayGenerationStartOwner(
            diagnostic_sink=self._on_generation_diagnostic,
        )
        self._fallback_owner = OverlaySessionFallbackOwner(
            can_start=self._can_start_fallback,
            start_overlay=self._begin_fallback_start,
            publish_notice=self.fallback_notice_sink,
            diagnostics_sink=self._on_fallback_diagnostic,
        )

    @property
    def runtime(self) -> OverlayRuntimeHandle | None:
        return self._runtime

    @runtime.setter
    def runtime(self, value: OverlayRuntimeHandle | None) -> None:
        self._runtime = value

    @property
    def state(self) -> str:
        return self._state

    @state.setter
    def state(self, value: str) -> None:
        self._state = value

    @property
    def failure_reason(self) -> str | None:
        return self._failure_reason

    @failure_reason.setter
    def failure_reason(self, value: str | None) -> None:
        self._failure_reason = value

    @property
    def auto_restart_scheduled(self) -> bool:
        return self._auto_restart_scheduled

    @auto_restart_scheduled.setter
    def auto_restart_scheduled(self, value: bool) -> None:
        self._auto_restart_scheduled = bool(value)

    @property
    def active_target(self) -> str | None:
        return self._active_target

    @active_target.setter
    def active_target(self, value: str | None) -> None:
        self._active_target = value

    @property
    def transition_owner(self) -> OverlaySessionTransitionOwner:
        return self._transition_owner

    @property
    def generation_owner(self) -> OverlayGenerationStartOwner:
        return self._generation_owner

    @property
    def fallback_owner(self) -> OverlaySessionFallbackOwner:
        return self._fallback_owner

    @property
    def snapshot(self) -> OverlayApplicationSnapshot:
        recovery = self._startup_recovery
        return OverlayApplicationSnapshot(
            state=self._state,
            failure_reason=self._failure_reason,
            auto_restart_scheduled=self._auto_restart_scheduled,
            active_target=self._active_target,
            fallback_active=self._fallback_owner.active,
            fallback_policy=OVERLAY_STEAMVR_FALLBACK_POLICY,
            recovery_active=recovery is not None,
            recovery_reason=(
                str(recovery.get("failure_reason"))
                if recovery is not None and recovery.get("failure_reason") is not None
                else None
            ),
        )

    @property
    def startup_recovery(self) -> dict[str, object] | None:
        recovery = self._startup_recovery
        return dict(recovery) if recovery is not None else None

    @property
    def last_startup_recovery(self) -> dict[str, object] | None:
        recovery = self._last_startup_recovery
        return dict(recovery) if recovery is not None else None

    @staticmethod
    def normalized_target(value: object) -> str:
        if value == OVERLAY_TARGET_DESKTOP:
            return OVERLAY_TARGET_DESKTOP
        return OVERLAY_TARGET_STEAMVR

    def target_for_state(self, state: OverlayApplicationState | None = None) -> str:
        resolved = state or self.state_provider()
        if not resolved.settings_available:
            return OVERLAY_TARGET_STEAMVR
        return self.normalized_target(resolved.configured_target)

    def effective_target_for_start(self) -> str:
        if self._fallback_owner.active:
            return OVERLAY_TARGET_DESKTOP
        return self.target_for_state()

    def clear_fallback(self) -> None:
        self._fallback_owner.clear()

    def publish_fallback(self, active: bool) -> None:
        self._fallback_owner.publish(active)

    def should_fallback(self, reason: str) -> bool:
        state = self.state_provider()
        return self._fallback_owner.should_fallback(
            reason=reason,
            active_target=self._active_target,
            configured_enabled=bool(state.settings_available and state.overlay_intent_enabled),
            configured_target=self.target_for_state(state),
            desktop_target=OVERLAY_TARGET_DESKTOP,
            steamvr_target=OVERLAY_TARGET_STEAMVR,
        )

    def presentation_state(self) -> OverlayPeerPresentationState | None:
        state = self.state_provider()
        if not state.settings_available:
            return None
        peer = self.peer_snapshot_provider()
        return OverlayPeerPresentationState(
            overlay_intent_enabled=state.overlay_intent_enabled,
            overlay_state=self._state,
            overlay_failure_reason=self._failure_reason,
            peer_intent_enabled=peer.intent_enabled,
            peer_effective_enabled=peer.effective_enabled,
            peer_warning_reason=peer.process_warning_reason,
            peer_activation_starting=peer.activation_starting or peer.model_loading,
            desktop_first_visible=self._current_desktop_first_visible(),
        )

    def _current_desktop_first_visible(self) -> bool:
        runtime = self._runtime
        if runtime is None:
            return False
        if self._active_target != OVERLAY_TARGET_DESKTOP:
            return False
        manager = runtime.process_manager
        if manager is None:
            return False
        return bool(getattr(manager, "desktop_first_visible", False))

    def on_desktop_first_visible(
        self,
        runtime: OverlayRuntimeHandle,
        overlay_instance_id: str | None,
    ) -> None:
        try:
            if self._runtime is not runtime:
                return
            if not self.runtime_is_current(runtime, overlay_instance_id=overlay_instance_id):
                return
            if self._state not in {"starting", "recovering"}:
                return
            if self._active_target != OVERLAY_TARGET_DESKTOP:
                return
        except Exception:
            return
        self.publish_presentation()

    def publish_presentation(self) -> None:
        with contextlib.suppress(Exception):
            self.presentation_sink(self.presentation_state())

    async def set_enabled(self, enabled: bool) -> None:
        state = self.state_provider()
        if not state.settings_available or self._ingress_stopped:
            return
        runtime = self._runtime
        self.log_basic(
            "[Overlay] Toggle: "
            f"enabled={enabled} state={self._state} "
            f"target={self._active_target or 'none'} "
            f"overlay_instance_id={runtime.overlay_instance_id if runtime is not None else 'none'}",
            logging.INFO,
        )
        self.overlay_intent_sink(bool(enabled))
        if not enabled:
            self.clear_fallback()
            self._cancel_startup_recovery()
            self.publish_presentation()
            await self.shutdown(preserve_failure_reason=True)
            return
        try:
            status = await self.begin_start()
        except Exception:
            self.publish_presentation()
            raise
        if status != "started":
            self.publish_presentation()

    def new_runtime(self) -> OverlayRuntimeHandle:
        runtime = OverlayRuntimeHandle(shutdown_grace_s=OVERLAY_SHUTDOWN_GRACE_S)
        self._runtime = runtime
        return runtime

    def ensure_runtime(self) -> OverlayRuntimeHandle:
        runtime = self._runtime
        if runtime is None:
            runtime = self.new_runtime()
        return runtime

    def runtime_is_current(
        self,
        runtime: OverlayRuntimeHandle,
        *,
        overlay_instance_id: str | None = None,
    ) -> bool:
        if self._runtime is not runtime:
            return False
        if overlay_instance_id is None:
            return True
        return runtime.is_current_instance_id(overlay_instance_id)

    @staticmethod
    def runtime_has_resources(runtime: OverlayRuntimeHandle | None) -> bool:
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

    def runtime_is_active(self) -> bool:
        runtime = self._runtime
        start_task = runtime.start_task if runtime is not None else None
        return bool(
            self._state in {"starting", "recovering", "connected"}
            or (runtime is not None and runtime.bridge is not None)
            or (runtime is not None and runtime.process_manager is not None)
            or (start_task is not None and not start_task.done())
        )

    def current_presenter(self) -> OverlayPresenter | None:
        runtime = self._runtime
        if runtime is None or self._state not in {"starting", "recovering", "connected"}:
            return None
        return cast(OverlayPresenter | None, runtime.current_presenter_for_ingress())

    def notify_translation_runtime_state_changed(self) -> None:
        if self._ingress_stopped:
            return
        runtime = self._runtime
        if runtime is None:
            return
        presenter = self.current_presenter()
        if presenter is None:
            return
        enabled = bool(self.translation_enabled_provider())
        self._translation_sync_generation += 1
        generation = self._translation_sync_generation
        overlay_instance_id = runtime.overlay_instance_id
        runtime.create_child_task(
            self._apply_translation_enabled_to_presenter(
                generation=generation,
                runtime=runtime,
                presenter=presenter,
                overlay_instance_id=overlay_instance_id,
                enabled=enabled,
            ),
            task_name="overlay-translation-sync",
        )

    async def _apply_translation_enabled_to_presenter(
        self,
        *,
        generation: int,
        runtime: OverlayRuntimeHandle,
        presenter: OverlayPresenter,
        overlay_instance_id: str | None,
        enabled: bool,
    ) -> None:
        if generation != self._translation_sync_generation:
            return
        if self._ingress_stopped:
            return
        if self._runtime is not runtime:
            return
        if not self.runtime_is_current(runtime, overlay_instance_id=overlay_instance_id):
            return
        if self.current_presenter() is not presenter:
            return
        await presenter.update_translation_enabled(enabled)

    def current_bridge(self) -> OverlayBridge | None:
        runtime = self._runtime
        if runtime is None:
            return None
        return cast(OverlayBridge | None, runtime.current_bridge_for_runtime_command())

    def previous_target_for_apply(self) -> str:
        if self.runtime_is_active() and self._active_target is not None:
            return self._active_target
        return self.target_for_state()

    async def replace_output_sink(
        self,
        overlay_sink: object | None,
        *,
        expected_current: object | None = None,
        require_match: bool = False,
    ) -> bool:
        output = self.output_provider()
        if output is None:
            return False
        return await output.replace_overlay_sink(
            cast(OverlayPresenter | None, overlay_sink),
            expected_current=cast(OverlayPresenter | None, expected_current),
            require_match=require_match,
        )

    async def detach_output_sink(self, expected_current: object | None) -> bool:
        return await self.replace_output_sink(
            None,
            expected_current=expected_current,
            require_match=True,
        )

    async def reset_output_preview(self) -> None:
        output = self.output_provider()
        if output is not None:
            await output.reset_overlay_preview()

    async def close_stale_start(self, runtime: OverlayRuntimeHandle) -> None:
        presenter = runtime.presenter
        diagnostics = runtime.diagnostics
        try:
            await runtime.close(
                preserve_presenter_state=True,
                overlay_sink_detach=self.detach_output_sink,
                preview_reset=self.reset_output_preview,
                diagnostics_detach=self.detach_translation_diagnostics,
                emit_shutdown=False,
            )
        except Exception as exc:
            self.log_detailed(
                "[Overlay] Stale overlay start cleanup reported failure",
                logging.WARNING,
                exc,
            )
        output = self.output_provider()
        if output is not None and output.overlay_sink is presenter:
            try:
                await self.replace_output_sink(
                    None,
                    expected_current=presenter,
                    require_match=True,
                )
            except Exception as exc:
                message = "[Overlay] Stale output ingress detach reported failure"
                detailed_emitted = self.log_detailed(message, logging.WARNING, exc)
                if not detailed_emitted:
                    self.log_basic(message, logging.WARNING)
        try:
            self.detach_translation_diagnostics(diagnostics)
        except Exception as exc:
            message = "[Overlay] Stale diagnostics detach reported failure"
            detailed_emitted = self.log_detailed(message, logging.WARNING, exc)
            if not detailed_emitted:
                self.log_basic(message, logging.WARNING)

    async def begin_start(self) -> OverlaySessionStartStatus | None:
        if self._ingress_stopped:
            return None
        self._cancel_startup_recovery()
        await self._drain_startup_recovery_task()
        return await self._transition_owner.begin_start(self._start_execution)

    async def _begin_fallback_start(self) -> None:
        generation = self._fallback_owner.generation
        reason = self._fallback_owner.reason
        try:
            status = await self._transition_owner.begin_start(
                lambda: self._start_execution(replace_starting=True)
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._complete_fallback_failure(reason, generation=generation)
            raise
        if status == "started":
            return
        if status == "already_active" and self._state in {"starting", "recovering", "connected"}:
            return
        await self._complete_fallback_failure(reason, generation=generation)

    def _cancel_startup_recovery(self) -> None:
        self._startup_recovery_generation += 1
        self._startup_recovery = None
        task = self._startup_recovery_task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    async def _drain_startup_recovery_task(self) -> None:
        task = self._startup_recovery_task
        if task is None or task is asyncio.current_task():
            return
        await asyncio.gather(task, return_exceptions=True)
        if self._startup_recovery_task is task:
            self._startup_recovery_task = None

    def _start_execution(
        self,
        *,
        replace_starting: bool = False,
    ) -> OverlaySessionStartExecution:
        return OverlaySessionStartExecution(
            state=self._state,
            previous_runtime=self._runtime,
            teardown=lambda: self.teardown(preserve_presenter_state=True),
            create_runtime=self.new_runtime,
            resolve_target=self.effective_target_for_start,
            on_starting=self._mark_starting,
            run_start=self.run_start,
            replace_starting=replace_starting,
            retire_previous=(
                self._retire_fallback_runtime
                if replace_starting
                and self._fallback_owner.active
                and self._active_target == OVERLAY_TARGET_STEAMVR
                and self._runtime is not None
                else None
            ),
        )

    async def _retire_fallback_runtime(self) -> object | None:
        runtime = self._runtime
        if runtime is None:
            return None
        await self.cancel_bounds_persistence()
        presenter = await runtime.retire_presentation(
            overlay_sink_detach=self.detach_output_sink,
            diagnostics_detach=self.detach_translation_diagnostics,
        )
        cleanup = self._close_retired_runtime(runtime)
        try:
            task = asyncio.create_task(
                cleanup,
                name=f"OverlayApplicationOwner:retired-{runtime.overlay_instance_id}",
            )
        except BaseException:
            cleanup.close()
            runtime.attach_presenter(presenter)
            raise
        self._retired_runtimes[runtime] = task
        task.add_done_callback(lambda completed: self._retired_runtime_closed(runtime, completed))
        self._runtime = None
        self._active_target = None
        self.clear_bounds_suppressed()
        return presenter

    async def _close_retired_runtime(self, runtime: OverlayRuntimeHandle) -> bool:
        try:
            await runtime.close(preserve_presenter_state=False, emit_shutdown=False)
            if runtime.has_resources():
                raise RuntimeError("retired overlay still owns resources")
        except Exception as exc:
            message = (
                "[Overlay] Retired VR cleanup failed: "
                f"overlay_instance_id={runtime.overlay_instance_id}"
            )
            if not self.log_detailed(message, logging.WARNING, exc):
                self.log_basic(message, logging.WARNING)
            return False
        return True

    def _retired_runtime_closed(
        self, runtime: OverlayRuntimeHandle, task: asyncio.Task[bool]
    ) -> None:
        if not task.cancelled() and task.exception() is None and task.result():
            if self._retired_runtimes.get(runtime) is task:
                self._retired_runtimes.pop(runtime)

    async def _teardown_all_runtimes(self) -> bool:
        succeeded = await self.teardown(preserve_presenter_state=False, emit_shutdown=True)
        for runtime, task in tuple(self._retired_runtimes.items()):
            try:
                closed = await asyncio.shield(task)
            except Exception:
                closed = False
            if not closed:
                closed = await self._close_retired_runtime(runtime)
            if closed:
                self._retired_runtimes.pop(runtime, None)
            else:
                succeeded = False
        return succeeded

    def _mark_starting(self, runtime: OverlayRuntimeHandle, target: str) -> None:
        if self._runtime is not runtime:
            raise RuntimeError("overlay start transition runtime is not current")
        self._active_target = target
        if self._startup_recovery is not None:
            return
        if not self._recovering_from_crash:
            self._auto_restart_scheduled = False
            self._terminal_restart_attempts = 0
            self._recovery_episode_started_at = None
        self._desktop_startup_recovery_attempted = False
        if self._state != "starting":
            self._transition_state("starting")
            self._notify_state()

    async def run_start(self, runtime: OverlayRuntimeHandle | None = None) -> None:
        if runtime is None:
            runtime = self._runtime or self.new_runtime()
        if not self.state_provider().settings_available or self.output_provider() is None:
            self._active_target = None
            if self.runtime_is_current(runtime):
                self.on_start_failed("unknown")
            return
        await self._generation_owner.start(
            runtime,
            self._generation_request,
            self._generation_effects(),
        )

    def _generation_request(self) -> OverlayGenerationStartRequest:
        state = self.state_provider()
        if not state.settings_available:
            raise RuntimeError("overlay start requires settings")
        config = self.config_provider()
        target = self._active_target or self.normalized_target(config.target)
        return OverlayGenerationStartRequest(
            config=config,
            target=target,
            clock=self.clock,
            startup_timeout_ms=OVERLAY_STARTUP_TIMEOUT_MS,
            fallback_reason=self._fallback_owner.reason if self._fallback_owner.active else None,
            translation_enabled=bool(self.translation_enabled_provider()),
        )

    def record_lifecycle_trace(self, event: str, **fields: object) -> None:
        runtime = self._runtime
        manager = runtime.process_manager if runtime is not None else None
        record_trace = getattr(manager, "record_lifecycle_trace", None)
        if callable(record_trace):
            record_trace("peer_application", event, **fields)

    def _generation_effects(self) -> OverlayGenerationStartEffects:
        return OverlayGenerationStartEffects(
            log_runtime=lambda message, **_kwargs: self.log_detailed(
                message,
                logging.INFO,
                None,
            ),
            log_failure=lambda message, level, exception: self.log_detailed(
                message,
                level,
                exception,
            ),
            is_current=lambda runtime, instance_id: self.runtime_is_current(
                runtime,
                overlay_instance_id=instance_id,
            ),
            close_stale=self.close_stale_start,
            replace_sink=self.replace_output_sink,
            set_diagnostics=self.attach_translation_diagnostics,
            set_target=self._set_active_target,
            calibration_snapshot=self.calibration_provider,
            logging_mode=self.logging_mode_provider,
            locale=self._locale,
            log_dir=self.log_dir_provider,
            build_desktop_controls=self.desktop_controls_factory,
            set_interaction_mode=self.interaction_mode_sink,
            track_bounds_control=self.bounds_control_sink,
            process_runner=self.process_runner,
            run_renderer_events=self.renderer_event_consumer,
            handle_failure=self.handle_start_failure,
            mark_connected=self.mark_connected,
            refresh_dependencies=self.refresh_peer_dependencies,
            watch_runtime=lambda manager, monitor, runtime, instance_id: self.watch_runtime(
                manager,
                monitor,
                runtime=runtime,
                overlay_instance_id=instance_id,
            ),
            notify_first_visible=self.on_desktop_first_visible,
        )

    def _should_restart_after_terminal_failure(self, manager: OverlayProcessManager) -> bool:
        if manager.restart_refill_ready:
            self._terminal_restart_attempts = 0
            self._recovery_episode_started_at = None
            manager.restart_refill_ready = False
        if not manager.restart_scheduled or manager.failure_reason == "termination_unconfirmed":
            return False
        if self._ingress_stopped:
            return False
        state = self.state_provider()
        if not state.settings_available or not state.overlay_intent_enabled:
            return False
        if self._terminal_restart_attempts >= OVERLAY_TERMINAL_RESTART_MAX:
            return False
        started_at = self._recovery_episode_started_at
        return (
            started_at is None or self.clock.now() - started_at < OVERLAY_TERMINAL_RESTART_WINDOW_S
        )

    async def _restart_after_terminal_failure(
        self,
        *,
        failure_reason: str | None,
        runtime: OverlayRuntimeHandle,
    ) -> None:
        if self._recovery_episode_started_at is None:
            self._recovery_episode_started_at = self.clock.now()
        self._terminal_restart_attempts += 1
        self.log_basic(
            f"[Overlay] Recovery requested: attempt={self._terminal_restart_attempts} "
            f"failure_reason={failure_reason}",
            logging.INFO,
        )
        self._recovering_from_crash = True
        self._auto_restart_scheduled = True
        self._failure_reason = None
        if self._state != "starting":
            self._transition_state("starting")
            self._notify_state()
        await asyncio.sleep(OVERLAY_TERMINAL_RESTART_BACKOFF_S * self._terminal_restart_attempts)
        if not self.runtime_is_current(runtime) or self._shutting_down:
            return
        episode_started_at = self._recovery_episode_started_at
        if (
            episode_started_at is not None
            and self.clock.now() - episode_started_at >= OVERLAY_TERMINAL_RESTART_WINDOW_S
        ):
            await self._fail_terminal_restart(failure_reason)
            return
        if self._ingress_stopped or not self.state_provider().overlay_intent_enabled:
            self._recovering_from_crash = False
            self._auto_restart_scheduled = False
            await self.teardown(preserve_presenter_state=True)
            return
        try:
            status = await self._transition_owner.begin_start(
                lambda: self._start_execution(replace_starting=True)
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._fail_terminal_restart(failure_reason)
            return
        if status == "started":
            return
        if status == "already_active" and self._state == "connected":
            return
        await self._fail_terminal_restart(failure_reason)

    async def _fail_terminal_restart(self, failure_reason: str | None) -> None:
        self.on_start_failed(failure_reason)
        await self.teardown(preserve_presenter_state=True)
        await self.refresh_peer_dependencies()

    def attach_translation_diagnostics(self, diagnostics: object) -> None:
        owner = self.diagnostics_provider()
        if owner is not None:
            owner.replace_overlay_diagnostics(cast(OverlayDiagnosticsRecorder, diagnostics))

    def detach_translation_diagnostics(self, expected_current: object | None) -> bool:
        owner = self.diagnostics_provider()
        if owner is None:
            return False
        return owner.replace_overlay_diagnostics(
            None,
            expected_current=cast(
                OverlayDiagnosticsRecorder | None,
                expected_current,
            ),
            require_match=True,
        )

    def _set_active_target(self, target: str) -> None:
        self._active_target = target

    def _locale(self) -> str:
        state = self.state_provider()
        if not state.settings_available:
            raise RuntimeError("overlay locale requires settings")
        return state.locale

    @staticmethod
    def process_runner(
        target: str,
        task_factory: object | None,
    ) -> OverlayProcessRunner:
        runner_cls = (
            DesktopFletOverlayRunner
            if target == OVERLAY_TARGET_DESKTOP
            else DefaultOverlayProcessRunner
        )
        try:
            return runner_cls(task_factory=task_factory)
        except TypeError:
            runner = runner_cls()
            with contextlib.suppress(Exception):
                setattr(runner, "task_factory", task_factory)
            return runner

    async def watch_runtime(
        self,
        manager: OverlayProcessManager,
        monitor_task: asyncio.Task[None],
        *,
        runtime: OverlayRuntimeHandle | None = None,
        overlay_instance_id: str | None = None,
    ) -> None:
        runtime = runtime or self._runtime
        try:
            await monitor_task
            if runtime is not None and not self.runtime_is_current(
                runtime,
                overlay_instance_id=overlay_instance_id,
            ):
                return
            if runtime is None or runtime.process_manager is not manager:
                return
            if manager.state != "failed":
                return
            if self._ingress_stopped or self._shutting_down:
                return
            if self._recovery_task is not None and not self._recovery_task.done():
                return
            self._recovery_task = asyncio.create_task(
                self._handle_runtime_failure(manager, runtime, runtime.monitor_task),
                name="overlay-application-recovery",
            )
            self._recovery_task.add_done_callback(self._clear_recovery_task)
        except asyncio.CancelledError:
            raise

    def _clear_recovery_task(self, task: asyncio.Task[None]) -> None:
        if self._recovery_task is task:
            self._recovery_task = None
        if not task.cancelled() and (error := task.exception()) is not None:
            self.log_basic(
                f"[Overlay] Recovery task failed: exception_type={type(error).__name__}",
                logging.ERROR,
            )

    async def _handle_runtime_failure(
        self,
        manager: OverlayProcessManager,
        runtime: OverlayRuntimeHandle,
        watcher: asyncio.Task[object] | None,
    ) -> None:
        if watcher is not None:
            await asyncio.shield(watcher)
        if (
            self._ingress_stopped
            or self._shutting_down
            or not self.runtime_is_current(runtime)
            or runtime.process_manager is not manager
        ):
            return
        if self._should_restart_after_terminal_failure(manager):
            await self._restart_after_terminal_failure(
                failure_reason=manager.failure_reason,
                runtime=runtime,
            )
            return
        await self._fail_terminal_restart(manager.failure_reason)

    async def _cancel_recovery(self) -> None:
        task = self._recovery_task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if self._recovery_task is task:
                self._recovery_task = None

    def _desktop_startup_recovery_candidate(
        self,
        failure_reason: str | None,
    ) -> tuple[object | None, dict[str, object] | None, str | None, str | None]:
        if self._ingress_stopped:
            return None, None, None, None
        if self._state not in {"starting", "recovering"}:
            return None, None, None, None
        if self._desktop_startup_recovery_attempted or self._startup_recovery is not None:
            return None, None, None, None
        try:
            state = self.state_provider()
        except Exception:
            return None, None, None, None
        if not state.settings_available or not state.overlay_intent_enabled:
            return None, None, None, None
        if self._active_target != OVERLAY_TARGET_DESKTOP:
            return None, None, None, None
        runtime = self._runtime
        if runtime is None:
            return None, None, None, None
        instance_id = runtime.overlay_instance_id
        if instance_id is None or not self.runtime_is_current(
            runtime, overlay_instance_id=instance_id
        ):
            return None, None, None, None
        manager = runtime.process_manager
        if manager is None or getattr(manager, "state", None) != "failed":
            return None, None, None, None
        reason = self.normalize_failure_reason(getattr(manager, "failure_reason", failure_reason))
        if reason not in DESKTOP_STARTUP_RECOVERABLE_REASONS:
            return None, None, None, None
        if not bool(getattr(manager, "startup_recovery_eligible", False)):
            return None, None, None, None
        raw_evidence = getattr(manager, "startup_failure_evidence", None)
        if raw_evidence is None:
            evidence: dict[str, object] = {}
        elif isinstance(raw_evidence, dict):
            evidence = dict(raw_evidence)
        else:
            return None, None, None, None
        return manager, evidence, reason, instance_id

    async def handle_start_failure(self, failure_reason: str | None) -> None:
        candidate = self._desktop_startup_recovery_candidate(failure_reason)
        if candidate[0] is not None:
            await self._run_desktop_startup_recovery(
                manager=candidate[0],
                evidence=candidate[1] or {},
                reason=candidate[2] or self.normalize_failure_reason(failure_reason),
                failed_instance_id=candidate[3],
            )
            return
        reason = self.normalize_failure_reason(failure_reason)
        if self.should_fallback(reason):
            self.log_basic(
                "[Overlay] Session fallback to desktop: "
                f"policy={OVERLAY_STEAMVR_FALLBACK_POLICY} reason={reason}",
                logging.INFO,
            )
            self._fallback_owner.activate(reason)
            self._failure_reason = None
            self.publish_fallback(True)
            if not self._fallback_owner.schedule():
                await self._complete_fallback_failure(reason)
            return
        if self._startup_recovery is not None:
            await self._finish_startup_recovery_terminal(
                self.normalize_failure_reason(failure_reason)
            )
            return
        self.on_start_failed(failure_reason)
        await self.teardown(preserve_presenter_state=True)
        await self.refresh_peer_dependencies()

    async def _run_desktop_startup_recovery(
        self,
        *,
        manager: object,
        evidence: dict[str, object],
        reason: str,
        failed_instance_id: str | None,
    ) -> None:
        recovery_generation = self._startup_recovery_generation
        self._desktop_startup_recovery_attempted = True
        self._startup_recovery = {
            "failure_reason": reason,
            "failed_overlay_instance_id": failed_instance_id,
            "evidence": dict(evidence),
            "attempt": DESKTOP_STARTUP_RECOVERY_MAX_ATTEMPTS,
            "generation": recovery_generation,
            "replacement_started": False,
        }
        self.log_detailed(
            "[Overlay] Desktop startup recovery eligible: "
            f"failure_reason={reason} "
            f"failed_overlay_instance_id={failed_instance_id} "
            f"fallback_active={self._fallback_owner.active}",
            logging.WARNING,
            None,
        )
        self._transition_state("recovering")
        self._notify_state()
        previous = self._runtime
        held_manager = manager
        teardown_succeeded = await self.teardown(preserve_presenter_state=True)
        cleanup_complete = bool(getattr(held_manager, "desktop_cleanup_complete", False))
        if (
            not teardown_succeeded
            or previous is None
            or not previous.is_closed
            or not previous.transfer_reap_complete()
            or not cleanup_complete
        ):
            self.log_detailed(
                "[Overlay] Desktop startup recovery teardown uncertain; terminal",
                logging.WARNING,
                None,
            )
            await self._finish_startup_recovery_terminal(reason)
            return
        if (
            recovery_generation != self._startup_recovery_generation
            or self._ingress_stopped
            or self._state != "recovering"
        ):
            return
        previous_task = self._startup_recovery_task
        if (
            previous_task is not None
            and not previous_task.done()
            and previous_task is not asyncio.current_task()
        ):
            previous_task.cancel()
            await asyncio.gather(previous_task, return_exceptions=True)
            if self._startup_recovery_task is previous_task:
                self._startup_recovery_task = None
        self._startup_recovery_task = asyncio.create_task(
            self._start_recovery_replacement(
                previous=previous,
                reason=reason,
                recovery_generation=recovery_generation,
            ),
            name="OverlayApplicationOwner:startup-recovery-replacement",
        )

    async def _start_recovery_replacement(
        self,
        *,
        previous: OverlayRuntimeHandle,
        reason: str,
        recovery_generation: int,
    ) -> None:
        await asyncio.sleep(0)
        try:
            if (
                recovery_generation != self._startup_recovery_generation
                or self._ingress_stopped
                or self._state != "recovering"
                or self._startup_recovery is None
            ):
                return
            try:
                state = self.state_provider()
            except Exception:
                await self._finish_startup_recovery_terminal(reason)
                return
            if not state.settings_available or not state.overlay_intent_enabled:
                self._startup_recovery = None
                return
            try:
                status = await self._transition_owner.begin_start(
                    lambda: self._start_execution(replace_starting=True)
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                await self._finish_startup_recovery_terminal(reason)
                return
            if status == "started":
                recovery = self._startup_recovery
                if recovery is not None:
                    recovery["replacement_started"] = True
                if not self._recovery_replacement_still_valid(recovery_generation):
                    await self.teardown(preserve_presenter_state=False)
                return
            if status == "already_active" and self._state in {
                "recovering",
                "starting",
                "connected",
            }:
                return
            await self._finish_startup_recovery_terminal(reason)
        finally:
            if self._startup_recovery_task is asyncio.current_task():
                self._startup_recovery_task = None

    def _recovery_replacement_still_valid(self, recovery_generation: int) -> bool:
        if recovery_generation != self._startup_recovery_generation:
            return False
        if self._ingress_stopped or self._state != "recovering":
            return False
        if self._startup_recovery is None:
            return False
        try:
            state = self.state_provider()
        except Exception:
            return False
        return bool(state.settings_available and state.overlay_intent_enabled)

    async def _finish_startup_recovery_terminal(self, failure_reason: str | None) -> None:
        recovery = self._startup_recovery
        reason = self.normalize_failure_reason(failure_reason)
        try:
            if recovery is not None:
                replacement_id = None
                if recovery.get("replacement_started"):
                    replacement_id = (
                        self._runtime.overlay_instance_id if self._runtime is not None else None
                    )
                self._last_startup_recovery = {
                    "failure_reason": recovery.get("failure_reason"),
                    "failed_overlay_instance_id": recovery.get("failed_overlay_instance_id"),
                    "replacement_overlay_instance_id": replacement_id,
                    "evidence": (
                        dict(recovery.get("evidence", {}))
                        if isinstance(recovery.get("evidence"), dict)
                        else recovery.get("evidence")
                    ),
                    "outcome": "failed",
                    "terminal_reason": reason,
                }
        except Exception:
            pass
        self._startup_recovery = None
        self.on_start_failed(reason)
        await self.teardown(preserve_presenter_state=True)
        try:
            await self.refresh_peer_dependencies()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log_detailed(
                "[Overlay] Peer dependency refresh failed after desktop recovery terminal",
                logging.WARNING,
                exc,
            )

    async def _complete_fallback_failure(
        self,
        failure_reason: str | None,
        *,
        generation: int | None = None,
    ) -> None:
        if generation is not None and not self._fallback_owner.is_current(generation):
            return
        reason = self.normalize_failure_reason(failure_reason or self._fallback_owner.reason)
        self._fallback_owner.clear()
        await self.teardown(preserve_presenter_state=True)
        self.on_start_failed(reason)
        try:
            await self.refresh_peer_dependencies()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log_detailed(
                "[Overlay] Peer dependency refresh failed after terminal fallback",
                logging.WARNING,
                exc,
            )

    def on_start_failed(self, failure_reason: str | None) -> None:
        self._failure_reason = self.normalize_failure_reason(failure_reason)
        self._auto_restart_scheduled = False
        self._recovering_from_crash = False
        self._startup_recovery = None
        self._transition_state("failed")
        self._notify_state()

    def on_runtime_disconnected(self) -> None:
        self.on_start_failed("runtime_disconnected")

    def on_runtime_crashed(self) -> None:
        self.on_start_failed("runtime_crashed")

    async def shutdown(self, *, preserve_failure_reason: bool) -> None:
        runtime = self._runtime
        self.log_basic(
            "[Overlay] Shutdown: "
            f"requested=True preserve_failure_reason={preserve_failure_reason} "
            f"state={self._state} target={self._active_target or 'none'} "
            f"overlay_instance_id={runtime.overlay_instance_id if runtime is not None else 'none'}",
            logging.INFO,
        )
        self._shutting_down = True
        try:
            await self._cancel_recovery()
            await self._transition_owner.shutdown(
                lambda: self._shutdown_execution(
                    preserve_failure_reason=preserve_failure_reason,
                )
            )
            await self._drain_startup_recovery_task()
        finally:
            self._shutting_down = False

    def _shutdown_execution(
        self,
        *,
        preserve_failure_reason: bool,
    ) -> OverlaySessionShutdownExecution:
        return OverlaySessionShutdownExecution(
            state=self._state,
            has_resources=(
                self.runtime_has_resources(self._runtime) or bool(self._retired_runtimes)
            ),
            teardown=self._teardown_all_runtimes,
            has_resources_after_teardown=lambda: (
                self.runtime_has_resources(self._runtime) or bool(self._retired_runtimes)
            ),
            on_stopping=self._mark_stopping,
            on_failed=lambda: self._complete_shutdown_failure(
                preserve_failure_reason=preserve_failure_reason,
            ),
            on_stopped=lambda: self._complete_shutdown(
                preserve_failure_reason=preserve_failure_reason,
            ),
        )

    def _mark_stopping(self) -> None:
        self._auto_restart_scheduled = False
        self._cancel_startup_recovery()
        self._transition_state("stopping")
        self._notify_state()

    async def _complete_shutdown_failure(
        self,
        *,
        preserve_failure_reason: bool,
    ) -> None:
        if not preserve_failure_reason or self._failure_reason is None:
            self._failure_reason = self.normalize_failure_reason(None)
        self._transition_state("failed")
        await self.refresh_peer_dependencies()
        self._notify_state()

    async def _complete_shutdown(
        self,
        *,
        preserve_failure_reason: bool,
    ) -> None:
        if not preserve_failure_reason:
            self._failure_reason = None
        self._transition_state("off")
        await self.refresh_peer_dependencies()
        self._notify_state()

    async def teardown(
        self,
        *,
        preserve_presenter_state: bool,
        emit_shutdown: bool = False,
    ) -> bool:
        existing = self._runtime
        if existing is None:
            await self.cancel_bounds_persistence()
            self._active_target = None
            self.clear_bounds_suppressed()
            if not preserve_presenter_state:
                self.interaction_mode_sink(self.edit_interaction_mode)
            return True
        if existing.is_closed and (
            preserve_presenter_state or not self.runtime_has_resources(existing)
        ):
            await self.cancel_bounds_persistence()
            if not self.runtime_has_resources(existing):
                self._runtime = None
            self._active_target = None
            self.clear_bounds_suppressed()
            if not preserve_presenter_state:
                self.interaction_mode_sink(self.edit_interaction_mode)
            return True
        runtime = existing
        await self.cancel_bounds_persistence()
        close_succeeded = True
        try:
            await runtime.close(
                preserve_presenter_state=preserve_presenter_state,
                overlay_sink_detach=self.detach_output_sink,
                preview_reset=self.reset_output_preview,
                diagnostics_detach=self.detach_translation_diagnostics,
                emit_shutdown=emit_shutdown,
            )
        except Exception as exc:
            close_succeeded = False
            message = "[Overlay] Overlay runtime close reported cleanup failure"
            detailed_emitted = self.log_detailed(message, logging.WARNING, exc)
            if not detailed_emitted:
                self.log_basic(message, logging.WARNING)
        if close_succeeded and not self.runtime_has_resources(runtime):
            self._runtime = None
        self._active_target = None
        self.clear_bounds_suppressed()
        if not preserve_presenter_state:
            self.interaction_mode_sink(self.edit_interaction_mode)
        return close_succeeded

    def mark_connected(self) -> None:
        recovery = self._startup_recovery
        if recovery is not None:
            try:
                failed_id = recovery.get("failed_overlay_instance_id")
                replacement_id = (
                    self._runtime.overlay_instance_id if self._runtime is not None else None
                )
                self._last_startup_recovery = {
                    "failure_reason": recovery.get("failure_reason"),
                    "failed_overlay_instance_id": failed_id,
                    "replacement_overlay_instance_id": replacement_id,
                    "evidence": (
                        dict(recovery.get("evidence", {}))
                        if isinstance(recovery.get("evidence"), dict)
                        else recovery.get("evidence")
                    ),
                    "outcome": "connected",
                }
                self.log_detailed(
                    "[Overlay] Desktop startup recovery connected: "
                    f"failure_reason={recovery.get('failure_reason')} "
                    f"failed_overlay_instance_id={failed_id} "
                    f"replacement_overlay_instance_id={replacement_id}",
                    logging.INFO,
                    None,
                )
            except Exception:
                pass
            self._startup_recovery = None
        self._failure_reason = None
        self._auto_restart_scheduled = False
        self._recovering_from_crash = False
        self._transition_state("connected")
        self._notify_state()

    @staticmethod
    def normalize_failure_reason(failure_reason: str | None) -> str:
        if isinstance(failure_reason, str) and failure_reason in OVERLAY_FAILURE_REASONS:
            return failure_reason
        return "unknown"

    def _transition_state(
        self,
        next_state: str,
        *,
        preserve_peer_activation: bool = False,
    ) -> None:
        previous = self._state
        self._state = next_state
        self._log_state_transition(previous, next_state)
        self.sync_peer_effective()
        if (
            next_state not in {"starting", "recovering", "connected"}
            and not preserve_peer_activation
        ):
            self.cancel_peer_activation()

    def _notify_state(self) -> None:
        self.state_sink(self._state, self._failure_reason)
        self.publish_presentation()

    def _log_state_transition(self, previous: str, next_state: str) -> None:
        runtime = self._runtime
        manager = runtime.process_manager if runtime is not None else None
        message = (
            f"[Overlay] State: previous={previous} current={next_state} "
            f"target={self._active_target or 'none'} "
            f"overlay_instance_id={runtime.overlay_instance_id if runtime is not None else 'none'} "
            f"manager_state={manager.state if manager is not None else 'none'}"
        )
        if self._failure_reason is not None:
            message = f"{message} failure_reason={self._failure_reason}"
        self.log_basic(message, logging.INFO)

    def _on_generation_diagnostic(
        self,
        diagnostic: OverlayGenerationStartDiagnostic,
    ) -> None:
        fields = [
            f"outcome={diagnostic.outcome}",
            f"target={diagnostic.target or 'unknown'}",
            f"overlay_instance_id={diagnostic.overlay_instance_id or 'unknown'}",
        ]
        if diagnostic.failure_type is not None:
            fields.append(f"failure_type={diagnostic.failure_type}")
        self.log_detailed(
            f"[Overlay] generation_start {' '.join(fields)}",
            logging.WARNING if diagnostic.outcome == "failed" else logging.INFO,
            None,
        )

    def _on_transition_diagnostic(
        self,
        diagnostic: OverlaySessionTransitionDiagnostic,
    ) -> None:
        fields = [
            f"operation={diagnostic.operation}",
            f"outcome={diagnostic.outcome}",
        ]
        if diagnostic.failure_type is not None:
            fields.append(f"failure_type={diagnostic.failure_type}")
        if diagnostic.stage is not None:
            fields.append(f"stage={diagnostic.stage}")
        if diagnostic.outcome in {"failed", "teardown_failed"}:
            self.log_basic(
                f"[Overlay] session_transition {' '.join(fields)}",
                logging.WARNING,
            )
            return
        self.log_detailed(
            f"[Overlay] session_transition {' '.join(fields)}",
            (
                logging.WARNING
                if diagnostic.outcome in {"failed", "teardown_failed"}
                else logging.INFO
            ),
            None,
        )

    def _on_fallback_diagnostic(
        self,
        event: str,
        _metadata: object,
        exception: Exception | None,
    ) -> None:
        self.log_detailed(
            f"[Overlay] Session desktop fallback failed: event={event}",
            logging.WARNING,
            exception,
        )

    def _can_start_fallback(self) -> bool:
        state = self.state_provider()
        return bool(
            not self._ingress_stopped
            and state.settings_available
            and state.overlay_intent_enabled
            and self._state == "starting"
        )

    def stop_ingress(self) -> None:
        self._ingress_stopped = True
        self._cancel_startup_recovery()
        self._fallback_owner.stop_ingress()

    async def close(self) -> None:
        self.stop_ingress()
        await self.shutdown(preserve_failure_reason=True)
        self.clear_fallback()
        await self._fallback_owner.close()
        await self._drain_startup_recovery_task()


__all__ = [
    "DESKTOP_STARTUP_RECOVERABLE_REASONS",
    "DESKTOP_STARTUP_RECOVERY_MAX_ATTEMPTS",
    "OVERLAY_FAILURE_REASONS",
    "OVERLAY_SHUTDOWN_GRACE_S",
    "OVERLAY_STEAMVR_FALLBACK_POLICY",
    "OVERLAY_STARTUP_TIMEOUT_MS",
    "OverlayApplicationState",
    "OverlayApplicationOwner",
    "OverlayApplicationSnapshot",
]
