from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Literal, cast

from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.config.resolved import OVERLAY_TARGET_DESKTOP, ResolvedOverlayConfig
from puripuly_heart.core.clock import Clock
from puripuly_heart.core.overlay.bridge import OverlayBridge
from puripuly_heart.core.overlay.diagnostics import OverlayDiagnosticsRecorder
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.process import (
    OverlayProcessManager,
    OverlayProcessRunner,
)
from puripuly_heart.core.runtime.overlay import OverlayRuntimeHandle

OverlayGenerationStartStatus = Literal[
    "connected",
    "detached",
    "failed",
    "stale",
]
OverlayGenerationRequestFactory = Callable[[], "OverlayGenerationStartRequest"]
OverlayGenerationRuntimeLogger = Callable[..., object]
OverlayGenerationFailureLogger = Callable[[str, int, Exception], None]
OverlayGenerationIsCurrent = Callable[[OverlayRuntimeHandle, str | None], bool]
OverlayGenerationCloseStale = Callable[[OverlayRuntimeHandle], Coroutine[object, object, None]]
OverlayGenerationReplaceSink = Callable[[OverlayPresenter], Coroutine[object, object, None]]
OverlayGenerationSetDiagnostics = Callable[[OverlayDiagnosticsRecorder], None]
OverlayGenerationSetTarget = Callable[[str], None]
OverlayGenerationCalibrationSnapshot = Callable[[], OverlayCalibration]
OverlayGenerationLoggingMode = Callable[[], str]
OverlayGenerationLocale = Callable[[], str]
OverlayGenerationLogDir = Callable[[], str]
OverlayGenerationDesktopControls = Callable[
    [ResolvedOverlayConfig],
    list[dict[str, object]],
]
OverlayGenerationSetInteractionMode = Callable[[str | None], None]
OverlayGenerationTrackBounds = Callable[[dict[str, object]], None]
OverlayGenerationProcessRunnerFactory = Callable[
    [str, object | None],
    OverlayProcessRunner,
]
OverlayGenerationRendererEvents = Callable[
    [asyncio.Queue[dict[str, object]], str],
    Coroutine[object, object, None],
]
OverlayGenerationRetryOwnership = Callable[
    [OverlayRuntimeHandle, OverlayPresenter, OverlayProcessManager, bool],
    Coroutine[object, object, None],
]
OverlayGenerationFailureHandler = Callable[[str | None], Coroutine[object, object, None]]
OverlayGenerationConnectedHandler = Callable[[], None]
OverlayGenerationRefresh = Callable[[], Coroutine[object, object, None]]
OverlayGenerationMonitor = Callable[
    [OverlayProcessManager, asyncio.Task[None], OverlayRuntimeHandle, str],
    Coroutine[object, object, None],
]


@dataclass(frozen=True, slots=True)
class OverlayGenerationStartRequest:
    config: ResolvedOverlayConfig
    target: str
    clock: Clock
    startup_timeout_ms: int
    fallback_reason: str | None = None
    recovering_from_crash: bool = False

    @property
    def desktop(self) -> bool:
        return self.target == OVERLAY_TARGET_DESKTOP


@dataclass(frozen=True, slots=True)
class OverlayGenerationStartEffects:
    log_runtime: OverlayGenerationRuntimeLogger
    log_failure: OverlayGenerationFailureLogger
    is_current: OverlayGenerationIsCurrent
    close_stale: OverlayGenerationCloseStale
    replace_sink: OverlayGenerationReplaceSink
    set_diagnostics: OverlayGenerationSetDiagnostics
    set_target: OverlayGenerationSetTarget
    calibration_snapshot: OverlayGenerationCalibrationSnapshot
    logging_mode: OverlayGenerationLoggingMode
    locale: OverlayGenerationLocale
    log_dir: OverlayGenerationLogDir
    build_desktop_controls: OverlayGenerationDesktopControls
    set_interaction_mode: OverlayGenerationSetInteractionMode
    track_bounds_control: OverlayGenerationTrackBounds
    process_runner: OverlayGenerationProcessRunnerFactory
    run_renderer_events: OverlayGenerationRendererEvents
    apply_retry_ownership: OverlayGenerationRetryOwnership
    handle_failure: OverlayGenerationFailureHandler
    mark_connected: OverlayGenerationConnectedHandler
    refresh_dependencies: OverlayGenerationRefresh
    watch_runtime: OverlayGenerationMonitor


@dataclass(frozen=True, slots=True)
class OverlayGenerationStartDiagnostic:
    outcome: Literal["cancelled", "connected", "detached", "failed", "stale"]
    target: str | None
    overlay_instance_id: str | None
    failure_type: str | None = None


OverlayGenerationStartDiagnosticSink = Callable[[OverlayGenerationStartDiagnostic], None]


@dataclass(slots=True)
class OverlayGenerationStartOwner:
    diagnostic_sink: OverlayGenerationStartDiagnosticSink | None = field(
        default=None,
        repr=False,
    )
    instance_token_factory: Callable[[], str] = field(
        default=lambda: secrets.token_hex(8),
        repr=False,
    )
    session_token_factory: Callable[[], str] = field(
        default=lambda: secrets.token_urlsafe(16),
        repr=False,
    )

    @property
    def owner_name(self) -> str:
        return "OverlayGenerationStartOwner"

    async def start(
        self,
        runtime: OverlayRuntimeHandle,
        request_factory: OverlayGenerationRequestFactory,
        effects: OverlayGenerationStartEffects,
    ) -> OverlayGenerationStartStatus:
        request: OverlayGenerationStartRequest | None = None
        overlay_instance_id: str | None = None
        try:
            presenter = cast(OverlayPresenter | None, runtime.presenter)
            overlay_instance_id = f"overlay-{self.instance_token_factory()}"
            runtime.set_overlay_instance_id(overlay_instance_id)
            diagnostics = OverlayDiagnosticsRecorder(
                overlay_instance_id=overlay_instance_id,
                logging_mode=effects.logging_mode(),
            )
            runtime.attach_diagnostics(diagnostics)
            request = request_factory()
            effects.set_target(request.target)
            peer_refresh_burst = not request.desktop
            self_refresh_burst = not request.desktop
            effects.log_runtime(
                "[Overlay][Start] "
                f"target={request.target} "
                f"overlay_instance_id={overlay_instance_id} "
                f"logging_mode={effects.logging_mode()} "
                f"peer_presentation_refresh_burst={peer_refresh_burst} "
                f"self_presentation_refresh_burst={self_refresh_burst}"
            )
            if presenter is None:
                presenter = OverlayPresenter(
                    calibration=effects.calibration_snapshot(),
                    clock=request.clock,
                    diagnostics=diagnostics,
                    runtime_log_detailed=effects.log_runtime,
                    show_translation=request.config.show_translation,
                    show_peer_original=request.config.show_peer_original,
                    task_factory=runtime.create_child_task,
                    peer_presentation_refresh_burst=peer_refresh_burst,
                    self_presentation_refresh_burst=self_refresh_burst,
                )
            else:
                presenter.runtime_log_detailed = effects.log_runtime
            presenter = cast(OverlayPresenter, runtime.adopt_presenter(presenter))
            presenter.runtime_log_detailed = effects.log_runtime
            if not request.desktop:
                if request.recovering_from_crash:
                    await presenter.discard_epoch_retry_intent()
                else:
                    await presenter.update_native_retry_ownership(False)
            await presenter.update_calibration(effects.calibration_snapshot())
            await presenter.update_display_preferences(
                show_translation=request.config.show_translation,
                show_peer_original=request.config.show_peer_original,
            )
            await presenter.update_peer_presentation_refresh_burst(peer_refresh_burst)
            await presenter.update_self_presentation_refresh_burst(self_refresh_burst)
            bridge = OverlayBridge(
                session_token=self.session_token_factory(),
                initial_snapshot=presenter.snapshot(),
                overlay_instance_id=overlay_instance_id,
                diagnostics=diagnostics,
                runtime_logging_mode=effects.logging_mode(),
                desktop_runtime_controls_enabled=request.desktop,
                task_factory=runtime.create_child_task,
            )
            if request.desktop:
                initial_controls = effects.build_desktop_controls(request.config)
                effects.set_interaction_mode(cast(str | None, initial_controls[-1].get("mode")))
                for payload in initial_controls:
                    effects.track_bounds_control(payload)
                bridge.set_initial_desktop_runtime_controls(initial_controls)
            runtime.attach_bridge(bridge)
            await bridge.start()
            if not effects.is_current(runtime, overlay_instance_id):
                await effects.close_stale(runtime)
                self._emit("stale", request, overlay_instance_id)
                return "stale"
            current_presenter = cast(
                OverlayPresenter | None,
                runtime.current_presenter_for_ingress(),
            )
            if current_presenter is not presenter:
                await effects.close_stale(runtime)
                self._emit("stale", request, overlay_instance_id)
                return "stale"
            presenter = current_presenter
            presenter.attach_bridge(bridge)
            latest_snapshot = presenter.snapshot()
            if bridge.snapshot() != latest_snapshot:
                await bridge.replace_snapshot(latest_snapshot)
            runtime.attach_diagnostics(diagnostics)
            await effects.replace_sink(presenter)
            effects.set_diagnostics(diagnostics)
            renderer_events: asyncio.Queue[dict[str, object]] | None = None
            if request.desktop:
                renderer_events = asyncio.Queue(maxsize=64)
                runtime.attach_renderer_events(renderer_events)
                runtime.create_renderer_event_task(
                    effects.run_renderer_events(
                        renderer_events,
                        overlay_instance_id,
                    )
                )
            else:
                runtime.attach_renderer_events(None)
            manager = OverlayProcessManager(
                process_runner=effects.process_runner(
                    request.target,
                    runtime.create_child_task,
                ),
                bridge_url=bridge.url,
                bridge_messages=bridge.messages,
                session_token=bridge.session_token,
                locale=effects.locale(),
                log_dir=effects.log_dir(),
                startup_timeout_ms=request.startup_timeout_ms,
                renderer_events=renderer_events,
                overlay_instance_id=overlay_instance_id,
                logging_mode=effects.logging_mode(),
                diagnostics=diagnostics,
                task_factory=runtime.create_child_task,
                selected_target=request.target,
                fallback_reason=request.fallback_reason,
                geometry_authority="flet" if request.desktop else "native",
                graceful_shutdown_request=(bridge.broadcast_shutdown if request.desktop else None),
                retry_ownership_changed=(
                    None
                    if request.desktop
                    else lambda confirmed: effects.apply_retry_ownership(
                        runtime,
                        presenter,
                        manager,
                        confirmed,
                    )
                ),
            )
            runtime.attach_process_manager(manager)
            await manager.start()
            if not effects.is_current(runtime, overlay_instance_id):
                await effects.close_stale(runtime)
                self._emit("stale", request, overlay_instance_id)
                return "stale"
            if runtime.process_manager is not manager:
                self._emit("detached", request, overlay_instance_id)
                return "detached"
            if manager.state != "connected":
                await effects.handle_failure(manager.failure_reason)
                self._emit("failed", request, overlay_instance_id)
                return "failed"
            effects.mark_connected()
            await effects.refresh_dependencies()
            monitor_task = getattr(manager, "_monitor_task", None)
            if monitor_task is not None:
                runtime.create_monitor_task(
                    effects.watch_runtime(
                        manager,
                        monitor_task,
                        runtime,
                        overlay_instance_id,
                    )
                )
            self._emit("connected", request, overlay_instance_id)
            return "connected"
        except asyncio.CancelledError:
            self._emit("cancelled", request, overlay_instance_id)
            raise
        except Exception as exc:
            if not effects.is_current(runtime, overlay_instance_id):
                effects.log_failure(
                    "[Overlay] Ignoring stale overlay runtime start failure",
                    logging.WARNING,
                    exc,
                )
                await effects.close_stale(runtime)
                self._emit("stale", request, overlay_instance_id, failure=exc)
                return "stale"
            effects.log_failure(
                "[Overlay] Failed to start overlay runtime",
                logging.ERROR,
                exc,
            )
            await effects.handle_failure("unknown")
            self._emit("failed", request, overlay_instance_id, failure=exc)
            return "failed"

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "resource_owner": "OverlayRuntimeHandle",
            "operation_policy": (
                "assemble one presenter, bridge, renderer and process generation in order"
            ),
            "stale_policy": "close stale generations before returning",
            "cancellation_policy": "propagate cancellation to OverlayRuntimeHandle",
            "failure_policy": "contain ordinary startup failure through the session adapter",
        }

    def _emit(
        self,
        outcome: Literal["cancelled", "connected", "detached", "failed", "stale"],
        request: OverlayGenerationStartRequest | None,
        overlay_instance_id: str | None,
        *,
        failure: Exception | None = None,
    ) -> None:
        if self.diagnostic_sink is None:
            return
        with contextlib.suppress(Exception):
            self.diagnostic_sink(
                OverlayGenerationStartDiagnostic(
                    outcome=outcome,
                    target=request.target if request is not None else None,
                    overlay_instance_id=overlay_instance_id,
                    failure_type=type(failure).__name__ if failure is not None else None,
                )
            )


__all__ = [
    "OverlayGenerationCalibrationSnapshot",
    "OverlayGenerationCloseStale",
    "OverlayGenerationConnectedHandler",
    "OverlayGenerationDesktopControls",
    "OverlayGenerationFailureHandler",
    "OverlayGenerationFailureLogger",
    "OverlayGenerationIsCurrent",
    "OverlayGenerationLocale",
    "OverlayGenerationLogDir",
    "OverlayGenerationLoggingMode",
    "OverlayGenerationMonitor",
    "OverlayGenerationProcessRunnerFactory",
    "OverlayGenerationRefresh",
    "OverlayGenerationRendererEvents",
    "OverlayGenerationReplaceSink",
    "OverlayGenerationRequestFactory",
    "OverlayGenerationRetryOwnership",
    "OverlayGenerationRuntimeLogger",
    "OverlayGenerationSetDiagnostics",
    "OverlayGenerationSetInteractionMode",
    "OverlayGenerationSetTarget",
    "OverlayGenerationStartDiagnostic",
    "OverlayGenerationStartDiagnosticSink",
    "OverlayGenerationStartEffects",
    "OverlayGenerationStartOwner",
    "OverlayGenerationStartRequest",
    "OverlayGenerationStartStatus",
    "OverlayGenerationTrackBounds",
]
