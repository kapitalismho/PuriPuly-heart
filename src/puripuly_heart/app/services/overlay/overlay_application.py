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
    OverlaySessionTransitionDiagnostic,
    OverlaySessionTransitionOwner,
)

OVERLAY_STARTUP_TIMEOUT_MS = 15000
OVERLAY_SHUTDOWN_GRACE_S = 0.05
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
        "runtime_disconnected",
        "window_configuration_failed",
        "runtime_control_invalid",
        "runtime_crashed",
        "unknown",
    }
)


@dataclass(frozen=True, slots=True)
class OverlayApplicationSnapshot:
    state: str
    failure_reason: str | None
    auto_restart_scheduled: bool
    active_target: str | None
    fallback_active: bool
    fallback_policy: Literal["retry_every_enable"]


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


@dataclass(slots=True)
class OverlayApplicationOwner:
    state_provider: OverlayStateProvider = field(repr=False)
    config_provider: OverlayConfigProvider = field(repr=False)
    overlay_intent_sink: OverlayIntentSink = field(repr=False)
    output_provider: OverlayOutputProvider = field(repr=False)
    diagnostics_provider: OverlayDiagnosticsProvider = field(repr=False)
    peer_snapshot_provider: OverlayPeerSnapshotProvider = field(repr=False)
    disable_peer_intent: OverlayEffect = field(repr=False)
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
    _runtime: OverlayRuntimeHandle | None = field(init=False, default=None, repr=False)
    _state: str = field(init=False, default="off", repr=False)
    _failure_reason: str | None = field(init=False, default=None, repr=False)
    _auto_restart_scheduled: bool = field(init=False, default=False, repr=False)
    _active_target: str | None = field(init=False, default=None, repr=False)
    _ingress_stopped: bool = field(init=False, default=False, repr=False)
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
        return OverlayApplicationSnapshot(
            state=self._state,
            failure_reason=self._failure_reason,
            auto_restart_scheduled=self._auto_restart_scheduled,
            active_target=self._active_target,
            fallback_active=self._fallback_owner.active,
            fallback_policy=OVERLAY_STEAMVR_FALLBACK_POLICY,
        )

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
        )

    def publish_presentation(self) -> None:
        with contextlib.suppress(Exception):
            self.presentation_sink(self.presentation_state())

    async def set_enabled(self, enabled: bool) -> None:
        state = self.state_provider()
        if not state.settings_available or self._ingress_stopped:
            return
        runtime = self._runtime
        self.log_basic(f"[Overlay] Toggle request: enabled={enabled}", logging.INFO)
        self.log_detailed(
            "[Overlay] Toggle detail: "
            f"current_state={self._state} "
            f"has_bridge={runtime is not None and runtime.bridge is not None} "
            f"has_manager={runtime is not None and runtime.process_manager is not None}",
            logging.INFO,
            None,
        )
        self.overlay_intent_sink(bool(enabled))
        if not enabled:
            self.disable_peer_intent()
            self.clear_fallback()
        self.publish_presentation()
        if enabled:
            await self.begin_start()
            return
        await self.shutdown(preserve_failure_reason=True)

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
            self._state in {"starting", "connected"}
            or (runtime is not None and runtime.bridge is not None)
            or (runtime is not None and runtime.process_manager is not None)
            or (start_task is not None and not start_task.done())
        )

    def current_presenter(self) -> OverlayPresenter | None:
        runtime = self._runtime
        if runtime is None or self._state not in {"starting", "connected"}:
            return None
        return cast(OverlayPresenter | None, runtime.current_presenter_for_ingress())

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

    async def begin_start(self) -> None:
        if self._ingress_stopped:
            return
        await self._transition_owner.begin_start(self._start_execution)

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
        if status == "already_active" and self._state in {"starting", "connected"}:
            return
        await self._complete_fallback_failure(reason, generation=generation)

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
        )

    def _mark_starting(self, runtime: OverlayRuntimeHandle, target: str) -> None:
        if self._runtime is not runtime:
            raise RuntimeError("overlay start transition runtime is not current")
        self._active_target = target
        self._auto_restart_scheduled = False
        if self._state != "starting":
            self._transition_state("starting")
            self._notify_state()

    async def _apply_retry_ownership(
        self,
        runtime: OverlayRuntimeHandle,
        presenter: OverlayPresenter,
        manager: OverlayProcessManager,
        *,
        confirmed: bool,
    ) -> None:
        if not self.runtime_is_current(runtime) or runtime.process_manager is not manager:
            return
        await presenter.update_native_retry_ownership(confirmed)

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
            apply_retry_ownership=lambda runtime, presenter, manager, confirmed: (
                self._apply_retry_ownership(
                    runtime,
                    presenter,
                    manager,
                    confirmed=confirmed,
                )
            ),
            handle_failure=self.handle_start_failure,
            mark_connected=self.mark_connected,
            refresh_dependencies=self.refresh_peer_dependencies,
            watch_runtime=lambda manager, monitor, runtime, instance_id: self.watch_runtime(
                manager,
                monitor,
                runtime=runtime,
                overlay_instance_id=instance_id,
            ),
        )

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
            self.on_start_failed(manager.failure_reason)
            await self.teardown(preserve_presenter_state=True)
            await self.refresh_peer_dependencies()
        except asyncio.CancelledError:
            raise

    async def handle_start_failure(self, failure_reason: str | None) -> None:
        reason = self.normalize_failure_reason(failure_reason)
        if self.should_fallback(reason):
            self.log_basic(
                "[Overlay] Session fallback to desktop: "
                f"policy={OVERLAY_STEAMVR_FALLBACK_POLICY} reason={reason}",
                logging.INFO,
            )
            self._fallback_owner.activate(reason)
            teardown_succeeded = await self.teardown(preserve_presenter_state=True)
            if not teardown_succeeded and self.runtime_has_resources(self._runtime):
                await self._complete_fallback_failure(reason)
                return
            self._failure_reason = None
            try:
                await self.refresh_peer_dependencies()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.log_detailed(
                    "[Overlay] Peer dependency refresh failed during desktop fallback",
                    logging.WARNING,
                    exc,
                )
                await self._complete_fallback_failure(reason)
                return
            self.publish_fallback(True)
            if not self._fallback_owner.schedule():
                await self._complete_fallback_failure(reason)
            return
        self.on_start_failed(failure_reason)
        await self.teardown(preserve_presenter_state=True)
        await self.refresh_peer_dependencies()

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
        self._transition_state("failed")
        self._notify_state()

    def on_runtime_disconnected(self) -> None:
        self.on_start_failed("runtime_disconnected")

    def on_runtime_crashed(self) -> None:
        self.on_start_failed("runtime_crashed")

    async def shutdown(self, *, preserve_failure_reason: bool) -> None:
        self.log_basic("[Overlay] Shutdown requested", logging.INFO)
        runtime = self._runtime
        self.log_detailed(
            "[Overlay] Shutdown detail: "
            f"preserve_failure_reason={preserve_failure_reason} "
            f"state={self._state} "
            f"has_bridge={runtime is not None and runtime.bridge is not None} "
            f"has_manager={runtime is not None and runtime.process_manager is not None} "
            f"presenter_attached={runtime is not None and runtime.presenter is not None}",
            logging.INFO,
            None,
        )
        await self._transition_owner.shutdown(
            lambda: self._shutdown_execution(
                preserve_failure_reason=preserve_failure_reason,
            )
        )

    def _shutdown_execution(
        self,
        *,
        preserve_failure_reason: bool,
    ) -> OverlaySessionShutdownExecution:
        return OverlaySessionShutdownExecution(
            state=self._state,
            has_resources=self.runtime_has_resources(self._runtime),
            teardown=lambda: self.teardown(
                preserve_presenter_state=False,
                emit_shutdown=True,
            ),
            has_resources_after_teardown=lambda: self.runtime_has_resources(self._runtime),
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
        runtime = self.ensure_runtime()
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
        self._failure_reason = None
        self._auto_restart_scheduled = False
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
        if next_state not in {"starting", "connected"} and not preserve_peer_activation:
            self.cancel_peer_activation()

    def _notify_state(self) -> None:
        self.state_sink(self._state, self._failure_reason)
        self.publish_presentation()

    def _log_state_transition(self, previous: str, next_state: str) -> None:
        runtime = self._runtime
        manager = runtime.process_manager if runtime is not None else None
        message = f"[Overlay] State transition: {previous} -> {next_state}"
        if self._failure_reason is not None:
            message = f"{message} failure_reason={self._failure_reason}"
        self.log_basic(message, logging.INFO)
        self.log_detailed(
            "[Overlay] State detail: "
            f"presenter_attached={runtime is not None and runtime.presenter is not None} "
            f"bridge_attached={runtime is not None and runtime.bridge is not None} "
            f"manager_state={manager.state if manager is not None else None}",
            logging.INFO,
            None,
        )

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
        self._fallback_owner.stop_ingress()

    async def close(self) -> None:
        self.stop_ingress()
        await self.shutdown(preserve_failure_reason=True)
        self.clear_fallback()
        await self._fallback_owner.close()


__all__ = [
    "OVERLAY_FAILURE_REASONS",
    "OVERLAY_SHUTDOWN_GRACE_S",
    "OVERLAY_STEAMVR_FALLBACK_POLICY",
    "OVERLAY_STARTUP_TIMEOUT_MS",
    "OverlayApplicationState",
    "OverlayApplicationOwner",
    "OverlayApplicationSnapshot",
]
