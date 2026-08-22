from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import pytest
from puripuly_heart.app.services.overlay_generation_start import (
    OverlayGenerationStartDiagnostic,
    OverlayGenerationStartEffects,
    OverlayGenerationStartOwner,
    OverlayGenerationStartRequest,
)

from puripuly_heart.app.services import (
    overlay_generation_start as overlay_generation_start_module,
)
from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.config.resolved import ResolvedOverlayConfig
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.runtime.overlay import OverlayRuntimeHandle


class FakePresenter:
    instances: list["FakePresenter"] = []
    events: list[str] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self.runtime_log_detailed = kwargs["runtime_log_detailed"]
        self.diagnostics = kwargs["diagnostics"]
        self.task_factory = kwargs["task_factory"]
        self.bridge: object | None = None
        self.snapshot_value = ("snapshot", len(self.instances))
        self.instances.append(self)
        self.events.append("presenter:create")

    async def update_native_retry_ownership(self, confirmed: bool) -> None:
        self.events.append(f"presenter:native_retry:{confirmed}")

    async def discard_epoch_retry_intent(self) -> None:
        self.events.append("presenter:discard_epoch")
        if isinstance(self.snapshot_value, dict):
            self.snapshot_value = {
                key: value
                for key, value in self.snapshot_value.items()
                if key != "native_fresh_render_generations"
            }

    async def update_calibration(self, calibration: OverlayCalibration) -> None:
        self.events.append(f"presenter:calibration:{calibration.distance}")

    async def update_display_preferences(
        self,
        *,
        show_translation: bool,
        show_peer_original: bool,
    ) -> None:
        self.events.append(f"presenter:preferences:{show_translation}:{show_peer_original}")

    async def update_peer_presentation_refresh_burst(self, enabled: bool) -> None:
        self.events.append(f"presenter:peer_refresh:{enabled}")

    async def update_self_presentation_refresh_burst(self, enabled: bool) -> None:
        self.events.append(f"presenter:self_refresh:{enabled}")

    def snapshot(self) -> object:
        return self.snapshot_value

    def attach_bridge(self, bridge: object) -> None:
        self.bridge = bridge
        self.events.append("presenter:attach_bridge")

    def detach_bridge(self) -> None:
        self.bridge = None


class FakeBridge:
    instances: list["FakeBridge"] = []
    events: list[str] = []
    start_failure: BaseException | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self.session_token = kwargs["session_token"]
        self.messages: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self.url = "ws://overlay.test"
        self.current_snapshot = kwargs["initial_snapshot"]
        self.initial_controls: list[dict[str, object]] = []
        self.instances.append(self)
        self.events.append("bridge:create")

    async def start(self) -> None:
        self.events.append("bridge:start")
        if self.start_failure is not None:
            raise self.start_failure

    def snapshot(self) -> object:
        return self.current_snapshot

    async def replace_snapshot(self, snapshot: object) -> None:
        self.current_snapshot = snapshot
        self.events.append("bridge:replace_snapshot")

    def set_initial_desktop_runtime_controls(
        self,
        controls: list[dict[str, object]],
    ) -> None:
        self.initial_controls = [dict(control) for control in controls]
        self.events.append("bridge:desktop_controls")

    async def broadcast_shutdown(self) -> None:
        self.events.append("bridge:shutdown")


class FakeProcessManager:
    instances: list["FakeProcessManager"] = []
    events: list[str] = []
    start_state = "connected"
    failure_reason: str | None = None
    after_start: Any = None
    monitor_enabled = True

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self.state = "off"
        self.failure_reason = None
        self._monitor_task: asyncio.Task[None] | None = None
        self.instances.append(self)
        self.events.append("manager:create")

    async def start(self) -> None:
        self.events.append("manager:start")
        self.state = self.start_state
        self.failure_reason = type(self).failure_reason
        if self.monitor_enabled:
            self._monitor_task = asyncio.create_task(asyncio.sleep(0))
        after_start = type(self).after_start
        if after_start is not None:
            result = after_start(self)
            if asyncio.iscoroutine(result):
                await result


@dataclass
class StartHarness:
    events: list[str] = field(default_factory=list)
    current_results: list[bool] = field(default_factory=list)
    closed: list[OverlayRuntimeHandle] = field(default_factory=list)
    diagnostics: list[object] = field(default_factory=list)
    owner_diagnostics: list[OverlayGenerationStartDiagnostic] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    failure_reasons: list[str | None] = field(default_factory=list)
    failure_logs: list[tuple[str, int, Exception]] = field(default_factory=list)
    retry_calls: list[bool] = field(default_factory=list)
    renderer_calls: list[str] = field(default_factory=list)
    watch_calls: list[str] = field(default_factory=list)
    connected: int = 0
    refreshes: int = 0

    def __post_init__(self) -> None:
        FakePresenter.events = self.events
        FakeBridge.events = self.events
        FakeProcessManager.events = self.events

    def request(
        self,
        *,
        desktop: bool,
        recovering_from_crash: bool = False,
    ) -> OverlayGenerationStartRequest:
        target = "desktop" if desktop else "steamvr"
        return OverlayGenerationStartRequest(
            config=ResolvedOverlayConfig(
                enabled=True,
                target=target,
                show_translation=False,
                show_peer_original=True,
                calibration={},
                desktop_overlay_options={},
            ),
            target=target,
            recovering_from_crash=recovering_from_crash,
            clock=FakeClock(),
            startup_timeout_ms=3210,
        )

    def effects(self) -> OverlayGenerationStartEffects:
        return OverlayGenerationStartEffects(
            log_runtime=lambda message: self.events.append(f"log:{message}"),
            log_failure=self._log_failure,
            is_current=self._is_current,
            close_stale=self._close_stale,
            replace_sink=self._replace_sink,
            set_diagnostics=self.diagnostics.append,
            set_target=self.targets.append,
            calibration_snapshot=lambda: OverlayCalibration(distance=1.4),
            logging_mode=lambda: "detailed",
            locale=lambda: "ko",
            log_dir=lambda: "C:\\overlay-log",
            build_desktop_controls=lambda _config: [
                {"command": "apply_window_bounds", "x": 1},
                {"command": "set_interaction_mode", "mode": "locked"},
            ],
            set_interaction_mode=lambda mode: self.events.append(f"interaction:{mode}"),
            track_bounds_control=lambda payload: self.events.append(
                f"control:{payload['command']}"
            ),
            process_runner=lambda target, _task_factory: ("runner", target),
            run_renderer_events=self._run_renderer_events,
            apply_retry_ownership=self._apply_retry_ownership,
            handle_failure=self._handle_failure,
            mark_connected=self._mark_connected,
            refresh_dependencies=self._refresh_dependencies,
            watch_runtime=self._watch_runtime,
        )

    def _log_failure(self, message: str, level: int, exception: Exception) -> None:
        self.failure_logs.append((message, level, exception))

    def _is_current(
        self,
        _runtime: OverlayRuntimeHandle,
        _overlay_instance_id: str | None,
    ) -> bool:
        if self.current_results:
            return self.current_results.pop(0)
        return True

    async def _close_stale(self, runtime: OverlayRuntimeHandle) -> None:
        self.closed.append(runtime)
        self.events.append("close:stale")

    async def _replace_sink(self, _presenter: object) -> None:
        self.events.append("sink:replace")

    async def _run_renderer_events(
        self,
        _queue: asyncio.Queue[dict[str, object]],
        overlay_instance_id: str,
    ) -> None:
        self.renderer_calls.append(overlay_instance_id)

    async def _apply_retry_ownership(
        self,
        _runtime: OverlayRuntimeHandle,
        _presenter: object,
        _manager: object,
        confirmed: bool,
    ) -> None:
        self.retry_calls.append(confirmed)

    async def _handle_failure(self, failure_reason: str | None) -> None:
        self.failure_reasons.append(failure_reason)

    def _mark_connected(self) -> None:
        self.connected += 1
        self.events.append("connected")

    async def _refresh_dependencies(self) -> None:
        self.refreshes += 1
        self.events.append("refresh")

    async def _watch_runtime(
        self,
        _manager: object,
        monitor_task: asyncio.Task[None],
        _runtime: OverlayRuntimeHandle,
        overlay_instance_id: str,
    ) -> None:
        self.watch_calls.append(overlay_instance_id)
        await monitor_task


@pytest.fixture(autouse=True)
def _patch_generation_components(monkeypatch: pytest.MonkeyPatch) -> None:
    FakePresenter.instances = []
    FakePresenter.events = []
    FakeBridge.instances = []
    FakeBridge.events = FakePresenter.events
    FakeBridge.start_failure = None
    FakeProcessManager.instances = []
    FakeProcessManager.events = FakePresenter.events
    FakeProcessManager.start_state = "connected"
    FakeProcessManager.failure_reason = None
    FakeProcessManager.after_start = None
    FakeProcessManager.monitor_enabled = True
    monkeypatch.setattr(overlay_generation_start_module, "OverlayPresenter", FakePresenter)
    monkeypatch.setattr(overlay_generation_start_module, "OverlayBridge", FakeBridge)
    monkeypatch.setattr(
        overlay_generation_start_module,
        "OverlayProcessManager",
        FakeProcessManager,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("desktop", [True, False], ids=["desktop", "native"])
async def test_owner_assembles_connected_generation_and_hands_off_monitor(
    desktop: bool,
) -> None:
    harness = StartHarness()
    runtime = OverlayRuntimeHandle(shutdown_grace_s=0)
    owner = OverlayGenerationStartOwner(
        diagnostic_sink=harness.owner_diagnostics.append,
        instance_token_factory=lambda: "instance",
        session_token_factory=lambda: "session",
    )

    status = await owner.start(
        runtime,
        lambda: harness.request(desktop=desktop),
        harness.effects(),
    )
    await asyncio.sleep(0)

    assert status == "connected"
    assert runtime.overlay_instance_id == "overlay-instance"
    assert runtime.presenter is FakePresenter.instances[0]
    assert runtime.bridge is FakeBridge.instances[0]
    assert runtime.process_manager is FakeProcessManager.instances[0]
    assert runtime.diagnostics is harness.diagnostics[0]
    assert harness.targets == ["desktop" if desktop else "steamvr"]
    assert harness.connected == 1
    assert harness.refreshes == 1
    assert harness.watch_calls == ["overlay-instance"]
    assert harness.owner_diagnostics[-1].outcome == "connected"
    manager = FakeProcessManager.instances[0]
    assert manager.kwargs["startup_timeout_ms"] == 3210
    assert manager.kwargs["locale"] == "ko"
    assert manager.kwargs["logging_mode"] == "detailed"
    assert manager.kwargs["selected_target"] == ("desktop" if desktop else "steamvr")
    assert manager.kwargs["fallback_reason"] is None
    assert manager.kwargs["geometry_authority"] == ("flet" if desktop else "native")
    assert FakePresenter.events.index("bridge:start") < FakePresenter.events.index("sink:replace")
    assert FakePresenter.events.index("sink:replace") < FakePresenter.events.index("manager:create")
    assert FakePresenter.events.index("manager:start") < FakePresenter.events.index("connected")

    if desktop:
        assert runtime.renderer_events is not None
        assert harness.renderer_calls == ["overlay-instance"]
        assert manager.kwargs["retry_ownership_changed"] is None
        assert callable(manager.kwargs["graceful_shutdown_request"])
        assert FakeBridge.instances[0].initial_controls[-1]["mode"] == "locked"
        assert "presenter:native_retry:False" not in FakePresenter.events
    else:
        assert runtime.renderer_events is None
        assert harness.renderer_calls == []
        assert manager.kwargs["graceful_shutdown_request"] is None
        retry = manager.kwargs["retry_ownership_changed"]
        assert retry is not None
        await retry(True)
        assert harness.retry_calls == [True]
        assert "presenter:native_retry:False" in FakePresenter.events


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("current_results", "expected_manager_count"),
    [([False], 0), ([True, False], 1)],
    ids=["after_bridge", "after_manager"],
)
async def test_owner_closes_stale_generation_at_each_external_start_boundary(
    current_results: list[bool],
    expected_manager_count: int,
) -> None:
    harness = StartHarness(current_results=list(current_results))
    runtime = OverlayRuntimeHandle(shutdown_grace_s=0)
    owner = OverlayGenerationStartOwner(instance_token_factory=lambda: "stale")

    status = await owner.start(
        runtime,
        lambda: harness.request(desktop=False),
        harness.effects(),
    )

    assert status == "stale"
    assert harness.closed == [runtime]
    assert len(FakeProcessManager.instances) == expected_manager_count
    assert harness.connected == 0
    assert harness.failure_reasons == []


@pytest.mark.asyncio
async def test_owner_returns_detached_when_runtime_replaces_process_manager() -> None:
    harness = StartHarness()
    runtime = OverlayRuntimeHandle(shutdown_grace_s=0)
    detached_manager = object()
    FakeProcessManager.after_start = lambda _manager: runtime.attach_process_manager(
        detached_manager
    )

    status = await OverlayGenerationStartOwner().start(
        runtime,
        lambda: harness.request(desktop=False),
        harness.effects(),
    )

    assert status == "detached"
    assert runtime.process_manager is detached_manager
    assert harness.connected == 0
    assert harness.failure_reasons == []


@pytest.mark.asyncio
async def test_owner_routes_process_failure_reason_without_marking_connected() -> None:
    harness = StartHarness()
    runtime = OverlayRuntimeHandle(shutdown_grace_s=0)
    FakeProcessManager.start_state = "failed"
    FakeProcessManager.failure_reason = "steamvr_not_running"
    FakeProcessManager.monitor_enabled = False

    status = await OverlayGenerationStartOwner().start(
        runtime,
        lambda: harness.request(desktop=False),
        harness.effects(),
    )

    assert status == "failed"
    assert harness.failure_reasons == ["steamvr_not_running"]
    assert harness.connected == 0
    assert harness.refreshes == 0


@pytest.mark.asyncio
async def test_owner_contains_current_exception_with_traceback_boundary() -> None:
    harness = StartHarness()
    runtime = OverlayRuntimeHandle(shutdown_grace_s=0)
    failure = RuntimeError("bridge failed")
    FakeBridge.start_failure = failure

    status = await OverlayGenerationStartOwner().start(
        runtime,
        lambda: harness.request(desktop=True),
        harness.effects(),
    )

    assert status == "failed"
    assert harness.failure_reasons == ["unknown"]
    assert harness.failure_logs == [
        ("[Overlay] Failed to start overlay runtime", logging.ERROR, failure)
    ]


@pytest.mark.asyncio
async def test_owner_contains_stale_exception_without_failing_current_session() -> None:
    harness = StartHarness(current_results=[False])
    runtime = OverlayRuntimeHandle(shutdown_grace_s=0)
    failure = RuntimeError("stale bridge failed")
    FakeBridge.start_failure = failure

    status = await OverlayGenerationStartOwner().start(
        runtime,
        lambda: harness.request(desktop=True),
        harness.effects(),
    )

    assert status == "stale"
    assert harness.closed == [runtime]
    assert harness.failure_reasons == []
    assert harness.failure_logs == [
        (
            "[Overlay] Ignoring stale overlay runtime start failure",
            logging.WARNING,
            failure,
        )
    ]


@pytest.mark.asyncio
async def test_owner_propagates_cancellation_after_attaching_generation_resources() -> None:
    harness = StartHarness()
    runtime = OverlayRuntimeHandle(shutdown_grace_s=0)
    FakeBridge.start_failure = asyncio.CancelledError()
    owner = OverlayGenerationStartOwner(
        diagnostic_sink=harness.owner_diagnostics.append,
        instance_token_factory=lambda: "cancelled",
    )

    with pytest.raises(asyncio.CancelledError):
        await owner.start(
            runtime,
            lambda: harness.request(desktop=True),
            harness.effects(),
        )

    assert runtime.overlay_instance_id == "overlay-cancelled"
    assert runtime.presenter is FakePresenter.instances[0]
    assert runtime.bridge is FakeBridge.instances[0]
    assert harness.failure_reasons == []
    assert harness.owner_diagnostics[-1].outcome == "cancelled"


@pytest.mark.asyncio
async def test_owner_crash_recovery_discards_old_epoch_snapshot_before_new_process() -> None:
    harness = StartHarness()
    runtime = OverlayRuntimeHandle(shutdown_grace_s=0)
    presenter = FakePresenter(
        runtime_log_detailed=lambda message, *, level=logging.INFO: False,
        diagnostics=None,
        task_factory=runtime.create_child_task,
    )
    presenter.snapshot_value = {
        "native_fresh_render_generations": {"self": 4},
        "text": "keep this caption",
    }
    runtime.adopt_presenter(presenter)
    owner = OverlayGenerationStartOwner(
        diagnostic_sink=harness.owner_diagnostics.append,
        instance_token_factory=lambda: "recovered",
        session_token_factory=lambda: "new-session",
    )

    status = await owner.start(
        runtime,
        lambda: harness.request(desktop=False, recovering_from_crash=True),
        harness.effects(),
    )
    await asyncio.sleep(0)

    assert status == "connected"
    assert "presenter:discard_epoch" in FakePresenter.events
    assert "presenter:native_retry:False" not in FakePresenter.events
    assert FakeBridge.instances[0].session_token == "new-session"
    assert FakeBridge.instances[0].current_snapshot == {"text": "keep this caption"}
    assert runtime.overlay_instance_id == "overlay-recovered"


def test_owner_declares_generation_assembly_without_absorbing_resource_teardown() -> None:
    assert OverlayGenerationStartOwner().lifecycle_owner_snapshot() == {
        "owner": "OverlayGenerationStartOwner",
        "resource_owner": "OverlayRuntimeHandle",
        "operation_policy": (
            "assemble one presenter, bridge, renderer and process generation in order"
        ),
        "stale_policy": "close stale generations before returning",
        "cancellation_policy": "propagate cancellation to OverlayRuntimeHandle",
        "failure_policy": "contain ordinary startup failure through the session adapter",
    }
