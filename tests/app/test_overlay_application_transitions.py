from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest
from puripuly_heart.app.services.overlay_application import (
    OVERLAY_STARTUP_TIMEOUT_MS,
    OverlayApplicationOwner,
    OverlayApplicationState,
)

from puripuly_heart.app.ports.ui_models import OverlayPeerPresentationState
from puripuly_heart.app.services.overlay.overlay_session_transition import (
    OverlaySessionTransitionDiagnostic,
)
from puripuly_heart.app.services.peer_application import (
    PeerApplicationOwner,
    PeerApplicationSnapshot,
    PeerApplicationState,
)
from puripuly_heart.app.wiring import build_peer_capture_session_config
from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.config.resolved import ResolvedOverlayConfig
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.sink import OverlayEventAdapter
from puripuly_heart.core.peer_capture import (
    PeerCaptureProviderStatus,
    PeerCaptureSessionSnapshot,
    PeerCaptureSessionState,
)
from puripuly_heart.domain.models import Transcript
from puripuly_heart.ui.overlay_peer_contract import (
    build_overlay_peer_consumer_contract_from_state,
)


async def _noop_async() -> None:
    return None


async def _noop_renderer(queue, overlay_instance_id: str) -> None:
    _ = queue, overlay_instance_id


class Recorder:
    def __init__(self) -> None:
        self.cancel_peer_activation_calls = 0
        self.sync_peer_effective_calls = 0
        self.states: list[tuple[str, str | None]] = []
        self.peer_activation_starting = True
        self.peer_effective_enabled = False
        self.peer_surface_states: list[str] = []
        self.logs: list[str] = []

    def cancel_peer_activation(self) -> None:
        self.cancel_peer_activation_calls += 1
        self.peer_activation_starting = False

    def sync_peer_effective(self) -> None:
        self.sync_peer_effective_calls += 1
        if self.peer_effective_enabled:
            self.peer_activation_starting = False

    def state_sink(self, state: str, failure_reason: str | None) -> None:
        self.states.append((state, failure_reason))

    def peer_snapshot(self) -> PeerApplicationSnapshot:
        return PeerApplicationSnapshot(
            intent_enabled=True,
            activation_requested=True,
            effective_enabled=self.peer_effective_enabled,
            desired_active=True,
            activation_generation=1,
            activation_starting=self.peer_activation_starting,
            model_loading=False,
            process_warning_reason=None,
            runtime_signature=None,
            provider_signature=None,
        )

    def presentation_sink(self, state: OverlayPeerPresentationState | None) -> None:
        if state is None:
            return
        contract = build_overlay_peer_consumer_contract_from_state(state)
        self.peer_surface_states.append(contract.peer.state)

    def log_basic(self, message: str, _level: int) -> None:
        self.logs.append(message)


class CaptureRuntime:
    def __init__(self) -> None:
        self.current_signature: object | None = None
        self.prepare_calls = 0
        self.policy_calls: list[tuple[bool, str]] = []
        self.close_calls = 0
        self._config = None
        self._generation = 0
        self._state = PeerCaptureSessionState.STOPPED
        self._provider_status = PeerCaptureProviderStatus.DETACHED
        self._target_status = None
        self._desired_active = False
        self._effective_active = False
        self._failure_reason = None
        self._has_source = False
        self._has_vad = False
        self._has_loop_task = False
        self._closed = False

    @property
    def snapshot(self) -> PeerCaptureSessionSnapshot:
        config = self._config
        return PeerCaptureSessionSnapshot(
            state=self._state,
            provider_status=self._provider_status,
            target_status=self._target_status,
            desired_active=self._desired_active,
            effective_active=self._effective_active,
            generation=self._generation,
            provider_id=config.provider_id if config is not None else None,
            runtime_signature=(config.runtime_signature if config is not None else None),
            capture_target=config.capture_target if config is not None else None,
            resolved_target=None,
            language=config.delivery_language if config is not None else None,
            failure_reason=self._failure_reason,
            admission_reason=None,
            target_reason=None,
            retry_available=False,
            has_source=self._has_source,
            has_vad=self._has_vad,
            has_loop_task=self._has_loop_task,
            cleanup_debt=0,
            closed=self._closed,
        )

    async def prepare_provider(self, config):
        self.prepare_calls += 1
        self._config = config
        self.current_signature = config.runtime_signature
        self._generation += 1
        self._state = PeerCaptureSessionState.STOPPED
        self._provider_status = PeerCaptureProviderStatus.READY
        self._desired_active = False
        self._effective_active = False
        self._failure_reason = None
        self._has_source = False
        self._has_vad = False
        self._has_loop_task = False
        return self.snapshot

    def activate_capture(self) -> None:
        if self._config is None:
            raise AssertionError("capture must be prepared before activation")
        self._generation += 1
        self._state = PeerCaptureSessionState.RUNNING
        self._provider_status = PeerCaptureProviderStatus.READY
        self._desired_active = True
        self._effective_active = True
        self._failure_reason = None
        self._has_source = True
        self._has_vad = True
        self._has_loop_task = True

    async def apply_policy(
        self,
        *,
        config,
        desired_active: bool,
        stop_mode: str = "retain",
    ) -> None:
        self._config = config
        self.current_signature = config.runtime_signature
        self.policy_calls.append((desired_active, stop_mode))
        if desired_active and not self._desired_active:
            self._generation += 1
        self._desired_active = desired_active
        self._effective_active = desired_active
        self._state = (
            PeerCaptureSessionState.RUNNING if desired_active else PeerCaptureSessionState.STOPPED
        )
        self._provider_status = PeerCaptureProviderStatus.READY
        self._failure_reason = None
        self._has_source = desired_active
        self._has_vad = desired_active
        self._has_loop_task = desired_active

    async def close(self) -> None:
        self.close_calls += 1
        self._closed = True
        self._desired_active = False
        self._effective_active = False
        self._state = PeerCaptureSessionState.STOPPED
        self._has_source = False
        self._has_vad = False
        self._has_loop_task = False


class PeerOverlayHarness:
    def __init__(self) -> None:
        self.overlay_state = "starting"
        self.peer_intent_enabled = False
        self.states: list[tuple[str, str | None]] = []
        self.peer_surface_states: list[str] = []
        self.fallback_notices: list[bool] = []
        self.logs: list[str] = []
        self.capture_runtime = CaptureRuntime()
        self.refresh_error: Exception | None = None
        self.peer = PeerApplicationOwner(
            state_provider=self.peer_state,
            config_factory=lambda: build_peer_capture_session_config(AppSettingsVNext()),
            peer_intent_sink=lambda enabled: setattr(
                self,
                "peer_intent_enabled",
                enabled,
            ),
            overlay_intent_sink=lambda _enabled: None,
            persist_manual_fallback=lambda: True,
            ensure_local_ready=self.ensure_local_ready,
            clear_cpu_pending=lambda: None,
            clear_gpu_pending=lambda: None,
            clear_switched_pending=lambda: None,
            sync_local_notice=lambda: None,
            presentation_changed=lambda: None,
            begin_overlay_start=_noop_async,
            effective_sink=lambda _peer, _context: None,
            disclosure_sink=lambda: None,
            superseded_sink=lambda: None,
            log_basic=lambda _message: None,
            log_diagnostic=lambda _message: None,
            log_failure=lambda _message: None,
        )
        self.peer.bind_runtime(self.capture_runtime)  # type: ignore[arg-type]
        self.overlay = OverlayApplicationOwner(
            state_provider=lambda: OverlayApplicationState(
                settings_available=True,
                overlay_intent_enabled=True,
                configured_target="steamvr",
                locale="en",
            ),
            config_provider=lambda: cast(ResolvedOverlayConfig, object()),
            overlay_intent_sink=lambda _enabled: None,
            output_provider=lambda: None,
            diagnostics_provider=lambda: None,
            peer_snapshot_provider=self.peer.snapshot,
            sync_peer_effective=self.peer.sync_effective_flags,
            cancel_peer_activation=self.peer.cancel_activation_starting,
            refresh_peer_dependencies=self.refresh_peer,
            presentation_sink=self.presentation_sink,
            state_sink=self.state_sink,
            fallback_notice_sink=self.fallback_notices.append,
            cancel_bounds_persistence=_noop_async,
            clear_bounds_suppressed=lambda: None,
            calibration_provider=lambda: cast(OverlayCalibration, object()),
            log_dir_provider=lambda: "",
            desktop_controls_factory=lambda _config: [],
            interaction_mode_sink=lambda _mode: None,
            bounds_control_sink=lambda _control: None,
            renderer_event_consumer=_noop_renderer,
            edit_interaction_mode="edit",
            clock=FakeClock(_now=0.0),
            log_basic=lambda message, _level: self.logs.append(message),
            log_diagnostic=lambda _message, _level, _exception: False,
            translation_enabled_provider=lambda: True,
        )
        self.overlay.state = "starting"
        self.overlay.active_target = "steamvr"

    def peer_state(self) -> PeerApplicationState:
        overlay_state = self.overlay.state if hasattr(self, "overlay") else self.overlay_state
        return PeerApplicationState(
            settings_available=True,
            peer_intent_enabled=self.peer_intent_enabled,
            eula_accepted=True,
            overlay_intent_enabled=True,
            peer_provider_id="local_cpu_auto",
            runtime_available=True,
            peer_provider_available=True,
            overlay_state=overlay_state,
        )

    async def ensure_local_ready(self, _generation: int) -> bool:
        return True

    async def refresh_peer(self) -> None:
        if self.refresh_error is not None:
            raise self.refresh_error
        await self.peer.refresh_dependencies()

    def state_sink(self, state: str, failure_reason: str | None) -> None:
        self.overlay_state = state
        self.states.append((state, failure_reason))

    def presentation_sink(self, state: OverlayPeerPresentationState | None) -> None:
        if state is None:
            return
        contract = build_overlay_peer_consumer_contract_from_state(state)
        self.peer_surface_states.append(contract.peer.state)

    async def activate_peer(self) -> None:
        await self.peer.set_enabled(True)
        self.overlay._notify_state()

    def assert_terminal_fallback_failure(
        self,
        *,
        expected_notices: list[bool] | None = None,
        expected_surfaces: list[str] | None = None,
    ) -> None:
        assert self.overlay.state == "failed"
        assert self.overlay.failure_reason == "steamvr_not_running"
        assert self.overlay.snapshot.fallback_active is False
        assert self.peer.snapshot().activation_starting is False
        assert self.peer_surface_states == (
            ["starting", "warning"] if expected_surfaces is None else expected_surfaces
        )
        assert self.fallback_notices == (
            [True, False] if expected_notices is None else expected_notices
        )


def test_overlay_shutdown_failure_reasons_remain_distinct_for_application_consumers() -> None:
    reasons = {
        "shutdown_not_acknowledged",
        "runtime_exit_nonzero",
        "shutdown_forced",
        "shutdown_cleanup_failed",
        "render_failed",
        "openvr_failed",
    }

    assert {
        OverlayApplicationOwner.normalize_failure_reason(reason) for reason in reasons
    } == reasons


def test_overlay_recovery_failure_is_visible_in_basic_logs() -> None:
    recorder = Recorder()
    owner = make_owner(recorder)
    owner._on_transition_diagnostic(
        OverlaySessionTransitionDiagnostic(
            operation="start",
            outcome="failed",
            failure_type="RuntimeError",
            stage="detach_presenter",
        )
    )
    assert len(recorder.logs) == 1
    assert "outcome=failed" in recorder.logs[0]
    assert "stage=detach_presenter" in recorder.logs[0]
    assert "failure_type=RuntimeError" in recorder.logs[0]


class FixedStartTransition:
    def __init__(self, status: str) -> None:
        self.status = status

    async def begin_start(self, _execution_factory) -> str:
        return self.status


class SuccessfulStartTransition:
    async def begin_start(self, execution_factory) -> str:
        execution = execution_factory()
        if not await execution.teardown():
            return "teardown_failed"
        runtime = execution.create_runtime()
        execution.on_starting(runtime, execution.resolve_target())
        return "started"


class RaisingStartTransition:
    async def begin_start(self, _execution_factory) -> str:
        raise RuntimeError("fallback start failed")


class RecordingStartTransition:
    def __init__(self) -> None:
        self.calls = 0
        self.execution_state: str | None = None

    async def begin_start(self, execution_factory) -> str:
        self.calls += 1
        execution = execution_factory()
        self.execution_state = execution.state
        return "started"


def make_owner(recorder: Recorder) -> OverlayApplicationOwner:
    return OverlayApplicationOwner(
        state_provider=lambda: OverlayApplicationState(
            settings_available=True,
            overlay_intent_enabled=True,
            configured_target="steamvr",
            locale="en",
        ),
        config_provider=lambda: cast(ResolvedOverlayConfig, object()),
        overlay_intent_sink=lambda _enabled: None,
        output_provider=lambda: None,
        diagnostics_provider=lambda: None,
        peer_snapshot_provider=recorder.peer_snapshot,
        sync_peer_effective=recorder.sync_peer_effective,
        cancel_peer_activation=recorder.cancel_peer_activation,
        refresh_peer_dependencies=_noop_async,
        presentation_sink=recorder.presentation_sink,
        state_sink=recorder.state_sink,
        fallback_notice_sink=lambda _active: None,
        cancel_bounds_persistence=_noop_async,
        clear_bounds_suppressed=lambda: None,
        calibration_provider=lambda: cast(OverlayCalibration, object()),
        log_dir_provider=lambda: "",
        desktop_controls_factory=lambda _config: [],
        interaction_mode_sink=lambda _mode: None,
        bounds_control_sink=lambda _control: None,
        renderer_event_consumer=_noop_renderer,
        edit_interaction_mode="edit",
        clock=FakeClock(_now=0.0),
        log_basic=recorder.log_basic,
        log_diagnostic=lambda _message, _level, _exception: False,
        translation_enabled_provider=lambda: True,
    )


def test_connect_transition_keeps_peer_activation_starting_alive() -> None:
    recorder = Recorder()
    owner = make_owner(recorder)

    owner.mark_connected()

    assert owner.state == "connected"
    assert recorder.cancel_peer_activation_calls == 0
    assert recorder.sync_peer_effective_calls == 1
    assert recorder.states == [("connected", None)]


def test_failure_transition_cancels_peer_activation_starting() -> None:
    for failure_reason in (
        "startup_timeout",
        "gpu_readiness_late",
        "gpu_readiness_cancelled",
        "gpu_query_failed",
        "gpu_stalled",
    ):
        recorder = Recorder()
        owner = make_owner(recorder)

        owner.on_start_failed(failure_reason)

        assert owner.state == "failed"
        assert recorder.cancel_peer_activation_calls == 1
        assert recorder.states == [("failed", failure_reason)]


def test_disconnect_transitions_cancel_peer_activation_starting() -> None:
    recorder = Recorder()
    owner = make_owner(recorder)

    owner.on_runtime_disconnected()
    owner.on_runtime_crashed()

    assert recorder.cancel_peer_activation_calls == 2


async def test_internal_steamvr_fallback_keeps_one_visible_peer_activation() -> None:
    recorder = Recorder()
    owner = make_owner(recorder)
    owner.state = "starting"
    owner.active_target = "steamvr"
    owner._notify_state()

    async def start_desktop() -> None:
        runtime = owner.new_runtime()
        owner._mark_starting(runtime, owner.effective_target_for_start())

    owner.fallback_owner.start_overlay = start_desktop

    await owner.handle_start_failure("steamvr_not_running")
    fallback_task = owner.fallback_owner.task
    assert fallback_task is not None
    await fallback_task

    recorder.peer_effective_enabled = True
    owner.mark_connected()

    assert recorder.cancel_peer_activation_calls == 0
    assert recorder.states == [("starting", None), ("connected", None)]
    assert recorder.peer_surface_states == ["starting", "on"]
    assert owner.snapshot.fallback_active is True
    assert owner.active_target == "desktop"
    assert any(
        "policy=retry_every_enable reason=steamvr_not_running" in message
        for message in recorder.logs
    )


async def test_retry_every_enable_policy_retries_configured_steamvr_after_disable() -> None:
    recorder = Recorder()
    owner = make_owner(recorder)
    owner.fallback_owner.activate("steamvr_not_running")
    owner.active_target = "desktop"

    assert owner.snapshot.fallback_policy == "retry_every_enable"
    assert owner.effective_target_for_start() == "desktop"
    request = owner._generation_request()
    assert request.target == "desktop"
    assert request.fallback_reason == "steamvr_not_running"
    assert request.startup_timeout_ms == OVERLAY_STARTUP_TIMEOUT_MS

    await owner.set_enabled(False)

    assert owner.snapshot.fallback_active is False
    assert owner.effective_target_for_start() == "steamvr"


async def test_caption_disable_keeps_listen_intent_and_capture_demand() -> None:
    harness = PeerOverlayHarness()
    await harness.activate_peer()
    harness.capture_runtime.activate_capture()
    harness.peer.sync_effective_flags()

    await harness.overlay.set_enabled(False)

    assert harness.overlay.state == "off"
    assert harness.peer_intent_enabled is True
    assert harness.peer.snapshot().activation_requested is True
    assert harness.capture_runtime.prepare_calls == 1
    assert harness.capture_runtime.policy_calls
    assert all(desired for desired, _stop_mode in harness.capture_runtime.policy_calls)
    assert harness.capture_runtime.close_calls == 0
    assert harness.peer.snapshot().desired_active is True
    harness.overlay._transition_owner = cast(object, SuccessfulStartTransition())
    await harness.overlay.set_enabled(True)

    assert harness.overlay.state == "starting"
    assert harness.capture_runtime.prepare_calls == 1
    assert all(desired for desired, _stop_mode in harness.capture_runtime.policy_calls)
    assert harness.capture_runtime.close_calls == 0


async def test_overlay_startup_timeout_is_shared_for_desktop_and_steamvr() -> None:
    recorder = Recorder()
    owner = make_owner(recorder)

    owner.active_target = "steamvr"
    steamvr_request = owner._generation_request()
    owner.active_target = "desktop"
    desktop_request = owner._generation_request()

    assert steamvr_request.startup_timeout_ms == OVERLAY_STARTUP_TIMEOUT_MS
    assert desktop_request.startup_timeout_ms == OVERLAY_STARTUP_TIMEOUT_MS
    assert OVERLAY_STARTUP_TIMEOUT_MS == 15000


async def test_fallback_task_creation_failure_terminates_real_peer_activation() -> None:
    harness = PeerOverlayHarness()
    await harness.activate_peer()

    def fail_task_creation(_coroutine, _name):
        raise RuntimeError("task creation failed")

    harness.overlay.fallback_owner.task_factory = fail_task_creation

    await harness.overlay.handle_start_failure("steamvr_not_running")

    harness.assert_terminal_fallback_failure()
    assert harness.overlay.fallback_owner.task is None


async def test_fallback_starts_before_peer_refresh_failure_without_hiding_peer_state() -> None:
    harness = PeerOverlayHarness()
    await harness.activate_peer()
    harness.refresh_error = RuntimeError("peer refresh failed")
    harness.overlay._transition_owner = cast(object, SuccessfulStartTransition())

    await harness.overlay.handle_start_failure("steamvr_not_running")
    fallback_task = harness.overlay.fallback_owner.task
    assert fallback_task is not None
    await fallback_task

    assert harness.overlay.state == "starting"
    assert harness.overlay.snapshot.fallback_active is True
    assert harness.peer.snapshot().activation_starting is True


async def test_successful_fallback_keeps_real_peer_starting_until_capture_effective() -> None:
    harness = PeerOverlayHarness()
    await harness.activate_peer()
    harness.overlay._transition_owner = cast(object, SuccessfulStartTransition())

    await harness.overlay.handle_start_failure("steamvr_not_running")
    fallback_task = harness.overlay.fallback_owner.task
    assert fallback_task is not None
    await fallback_task

    assert harness.peer.snapshot().activation_starting is True
    assert harness.states == [("starting", None)]
    assert harness.peer_surface_states == ["starting"]

    harness.peer.bind_runtime(
        cast(
            object,
            SimpleNamespace(snapshot=SimpleNamespace(effective_active=True)),
        )
    )
    harness.overlay.mark_connected()

    assert harness.peer.snapshot().activation_starting is False
    assert harness.states == [("starting", None), ("connected", None)]
    assert harness.peer_surface_states == ["starting", "on"]


async def test_fallback_start_exception_terminates_real_peer_activation() -> None:
    harness = PeerOverlayHarness()
    await harness.activate_peer()
    harness.overlay._transition_owner = cast(object, RaisingStartTransition())

    await harness.overlay.handle_start_failure("steamvr_not_running")
    fallback_task = harness.overlay.fallback_owner.task
    assert fallback_task is not None
    await fallback_task

    harness.assert_terminal_fallback_failure()
    assert harness.overlay.fallback_owner.task is None


async def test_fallback_teardown_failed_terminates_real_peer_activation() -> None:
    harness = PeerOverlayHarness()
    await harness.activate_peer()
    harness.overlay._transition_owner = cast(
        object,
        FixedStartTransition("teardown_failed"),
    )

    await harness.overlay.handle_start_failure("steamvr_not_running")
    fallback_task = harness.overlay.fallback_owner.task
    assert fallback_task is not None
    await fallback_task

    harness.assert_terminal_fallback_failure()
    assert harness.overlay.fallback_owner.task is None


async def test_watch_runtime_restarts_connected_crash_and_keeps_peer_activation() -> None:
    recorder = Recorder()
    owner = make_owner(recorder)
    runtime = owner.new_runtime()
    manager = SimpleNamespace(
        state="failed",
        restart_scheduled=True,
        failure_reason="runtime_crashed",
        restart_refill_ready=False,
    )
    runtime.attach_process_manager(manager)
    owner.state = "connected"
    owner._transition_owner = cast(object, FixedStartTransition("started"))
    monitor = asyncio.get_running_loop().create_future()
    monitor.set_result(None)

    await owner.watch_runtime(manager, monitor, runtime=runtime)
    await owner._recovery_task

    assert owner.auto_restart_scheduled is True
    assert owner.state == "starting"
    assert owner.failure_reason is None
    assert recorder.cancel_peer_activation_calls == 0


async def test_watch_runtime_does_not_restart_when_shutdown_was_not_scheduled() -> None:
    recorder = Recorder()
    owner = make_owner(recorder)
    runtime = owner.new_runtime()
    manager = SimpleNamespace(
        state="failed",
        restart_scheduled=False,
        failure_reason="runtime_crashed",
        restart_refill_ready=False,
    )
    runtime.attach_process_manager(manager)
    owner.state = "connected"
    monitor = asyncio.get_running_loop().create_future()
    monitor.set_result(None)

    await owner.watch_runtime(manager, monitor, runtime=runtime)
    await owner._recovery_task

    assert owner.auto_restart_scheduled is False
    assert owner.state == "failed"
    assert owner.failure_reason == "runtime_crashed"


async def test_watch_runtime_restart_teardown_failure_fails_instead_of_staying_starting() -> None:
    recorder = Recorder()
    owner = make_owner(recorder)
    runtime = owner.new_runtime()
    manager = SimpleNamespace(
        state="failed",
        restart_scheduled=True,
        failure_reason="runtime_crashed",
        restart_refill_ready=False,
    )
    runtime.attach_process_manager(manager)
    owner.state = "connected"
    owner._transition_owner = cast(object, FixedStartTransition("teardown_failed"))
    monitor = asyncio.get_running_loop().create_future()
    monitor.set_result(None)

    await owner.watch_runtime(manager, monitor, runtime=runtime)
    await owner._recovery_task

    assert owner.auto_restart_scheduled is False
    assert owner.state == "failed"
    assert owner.failure_reason == "runtime_crashed"

    follow_up = RecordingStartTransition()
    owner._transition_owner = cast(object, follow_up)
    await owner.begin_start()
    assert follow_up.calls == 1
    assert follow_up.execution_state == "failed"


async def test_terminal_restart_budget_rejects_cap_plus_one_until_qualified_progress() -> None:
    owner = make_owner(Recorder())
    manager = SimpleNamespace(
        restart_scheduled=True,
        restart_refill_ready=False,
        failure_reason="runtime_crashed",
    )

    for expected_attempt in range(3):
        assert owner._should_restart_after_terminal_failure(manager)
        owner._terminal_restart_attempts += 1
        assert owner._terminal_restart_attempts == expected_attempt + 1
    assert not owner._should_restart_after_terminal_failure(manager)

    manager.restart_refill_ready = True
    assert owner._should_restart_after_terminal_failure(manager)
    assert owner._terminal_restart_attempts == 0
    assert manager.restart_refill_ready is False


@pytest.mark.parametrize("failure_reason", ["render_failed", "native_owner_unresponsive"])
@pytest.mark.parametrize("elapsed", [1.0, 31.0])
async def test_owned_monitor_recovers_with_presenter_and_original_expiration(
    monkeypatch, failure_reason: str, elapsed: float
) -> None:
    recorder = Recorder()
    owner = make_owner(recorder)
    runtime = owner.new_runtime()
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=owner.clock)
    runtime.adopt_presenter(presenter)
    await presenter.emit(
        OverlayEventAdapter(clock=owner.clock).transcript_final(
            Transcript(
                utterance_id=uuid4(),
                channel="self",
                text="preserved caption",
                is_final=True,
                created_at=0.0,
            ),
            source_language="en",
            target_language="ko",
        )
    )
    manager = SimpleNamespace(
        state="failed",
        restart_scheduled=True,
        failure_reason=failure_reason,
        restart_refill_ready=False,
    )
    runtime.attach_process_manager(manager)
    owner.state = "connected"
    connected = asyncio.Event()

    async def start_replacement(self, replacement) -> None:
        assert runtime.is_closed
        assert runtime.monitor_task is None
        assert runtime.child_task_names == ()
        assert replacement is not runtime
        assert replacement.presenter is presenter
        owner.clock.advance(elapsed)
        await presenter.begin_native_retry_epoch(enabled=True)
        self.mark_connected()
        connected.set()

    monkeypatch.setattr(OverlayApplicationOwner, "run_start", start_replacement)
    monitor = asyncio.create_task(_noop_async())
    watcher = runtime.create_monitor_task(owner.watch_runtime(manager, monitor, runtime=runtime))
    await asyncio.wait_for(connected.wait(), 2.0)
    assert watcher.done()
    assert owner.state == "connected"
    assert owner.failure_reason is None
    assert bool(presenter.snapshot().blocks) is (elapsed < 8.0)
    assert recorder.cancel_peer_activation_calls == 0
    assert not any("stage=detach_presenter" in message for message in recorder.logs)
    await owner.close()


@pytest.mark.parametrize("phase", ["backoff", "teardown"])
async def test_off_drains_recovery_without_late_respawn(monkeypatch, phase: str) -> None:
    owner = make_owner(Recorder())
    runtime = owner.new_runtime()
    runtime.adopt_presenter(OverlayPresenter(calibration=OverlayCalibration(), clock=owner.clock))
    reached = asyncio.Event()
    stop_calls = 0
    starts = []

    async def stop() -> None:
        nonlocal stop_calls
        stop_calls += 1
        if phase == "teardown" and stop_calls == 1:
            reached.set()
            await asyncio.Event().wait()

    async def unexpected_start(self, replacement) -> None:
        starts.append(replacement)

    def state_changed(state, _reason) -> None:
        if phase == "backoff" and state == "starting":
            reached.set()

    owner.state_sink = state_changed
    monkeypatch.setattr(OverlayApplicationOwner, "run_start", unexpected_start)
    manager = SimpleNamespace(
        state="failed",
        restart_scheduled=True,
        failure_reason="render_failed",
        restart_refill_ready=False,
        stop=stop,
    )
    runtime.attach_process_manager(manager)
    owner.state = "connected"
    monitor = asyncio.create_task(_noop_async())
    watcher = runtime.create_monitor_task(owner.watch_runtime(manager, monitor, runtime=runtime))
    await asyncio.wait_for(reached.wait(), 2.0)
    recovery = owner._recovery_task
    assert recovery is not None
    await asyncio.wait_for(owner.set_enabled(False), 2.0)
    assert owner.state == "off"
    assert owner._recovery_task is None
    assert recovery.done()
    assert watcher.done()
    assert runtime.is_closed
    assert runtime.child_task_names == ()
    assert starts == []


async def test_stale_recovery_cannot_replace_new_generation(monkeypatch) -> None:
    owner = make_owner(Recorder())
    old_runtime = owner.new_runtime()
    reached = asyncio.Event()
    owner.state_sink = lambda state, _reason: reached.set() if state == "starting" else None
    manager = SimpleNamespace(
        state="failed",
        restart_scheduled=True,
        failure_reason="render_failed",
        restart_refill_ready=False,
    )
    old_runtime.attach_process_manager(manager)
    owner.state = "connected"
    transition = RecordingStartTransition()
    owner._transition_owner = cast(object, transition)
    monitor = asyncio.create_task(_noop_async())
    old_runtime.create_monitor_task(owner.watch_runtime(manager, monitor, runtime=old_runtime))
    await asyncio.wait_for(reached.wait(), 2.0)
    recovery = owner._recovery_task
    replacement = owner.new_runtime()
    owner.mark_connected()
    await asyncio.wait_for(recovery, 2.0)
    assert owner.runtime is replacement
    assert owner.state == "connected"
    assert transition.calls == 0
    await old_runtime.close(preserve_presenter_state=False)


@pytest.mark.parametrize("cleanup_action", ["finish", "off", "fail"])
async def test_desktop_fallback_starts_while_retired_vr_cleanup_is_blocked(
    monkeypatch, cleanup_action: str
) -> None:
    owner = make_owner(Recorder())
    old_runtime = owner.new_runtime()
    old_runtime.set_overlay_instance_id("failed-vr")
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=owner.clock)
    old_runtime.adopt_presenter(presenter)
    adapter = OverlayEventAdapter(clock=owner.clock)
    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=uuid4(),
                channel="self",
                text="caption survives fallback",
                is_final=True,
                created_at=0.0,
            ),
            source_language="en",
            target_language="ko",
        )
    )
    output = SimpleNamespace(overlay_sink=presenter)

    async def replace_sink(sink, *, expected_current=None, require_match=False):
        if require_match and output.overlay_sink is not expected_current:
            return False
        output.overlay_sink = sink
        return True

    output.replace_overlay_sink = replace_sink
    output.reset_overlay_preview = _noop_async
    owner.output_provider = lambda: output
    cleanup_entered = asyncio.Event()
    release_cleanup = asyncio.Event()
    fail_cleanup = cleanup_action == "fail"

    async def stop_vr() -> None:
        nonlocal fail_cleanup
        cleanup_entered.set()
        await release_cleanup.wait()
        if fail_cleanup:
            fail_cleanup = False
            raise RuntimeError("retired process cleanup failed")

    old_runtime.attach_process_manager(SimpleNamespace(stop=stop_vr))
    owner.state = "starting"
    owner.active_target = "steamvr"
    connected = asyncio.Event()

    async def start_desktop(self, runtime) -> None:
        assert self.active_target == "desktop"
        await presenter.begin_native_retry_epoch(enabled=False)
        await replace_sink(runtime.presenter)
        self.mark_connected()
        connected.set()

    monkeypatch.setattr(OverlayApplicationOwner, "run_start", start_desktop)
    stop_task = None
    try:
        await asyncio.wait_for(owner.handle_start_failure("steamvr_not_running"), timeout=1.0)
        await asyncio.wait_for(connected.wait(), timeout=1.0)
        await asyncio.wait_for(cleanup_entered.wait(), timeout=1.0)
        assert not release_cleanup.is_set()
        assert owner.state == "connected"
        assert output.overlay_sink is presenter
        assert presenter.snapshot().blocks[0].primary_text == "caption survives fallback"
        assert old_runtime.current_presenter_for_ingress() is None
        assert not old_runtime.is_current_instance_id("failed-vr")
        reapers = tuple(owner._retired_runtimes.values())
        assert all(not task.done() for task in reapers)
        if cleanup_action == "off":
            stop_task = asyncio.create_task(owner.set_enabled(False))
            await asyncio.sleep(0)
            assert not stop_task.done()
        release_cleanup.set()
        await asyncio.wait_for(asyncio.gather(*reapers), timeout=1.0)
        if stop_task is not None:
            await asyncio.wait_for(stop_task, timeout=1.0)
            assert owner.state == "off"
            assert output.overlay_sink is None
        else:
            assert owner.state == "connected"
            assert output.overlay_sink is presenter
            assert presenter.snapshot().blocks[0].primary_text == "caption survives fallback"
        if cleanup_action == "fail":
            assert owner._retired_runtimes
            await owner.set_enabled(False)
            assert owner.state == "off"
        assert not owner._retired_runtimes
        assert not old_runtime.has_resources()
    finally:
        release_cleanup.set()
        if stop_task is not None:
            await stop_task
        await owner.close()


async def test_off_during_fallback_retirement_never_starts_desktop(monkeypatch) -> None:
    owner = make_owner(Recorder())
    runtime = owner.new_runtime()
    runtime.set_overlay_instance_id("retiring-vr")
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=owner.clock)
    runtime.adopt_presenter(presenter)
    output = SimpleNamespace(overlay_sink=presenter)
    detach_entered = asyncio.Event()
    detach_cancelled = asyncio.Event()
    block_detach = True

    async def replace_sink(sink, *, expected_current=None, require_match=False):
        nonlocal block_detach
        if block_detach:
            block_detach = False
            detach_entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                detach_cancelled.set()
        if require_match and output.overlay_sink is not expected_current:
            return False
        output.overlay_sink = sink
        return True

    output.replace_overlay_sink = replace_sink
    output.reset_overlay_preview = _noop_async
    owner.output_provider = lambda: output
    owner.state = "starting"
    owner.active_target = "steamvr"
    starts = []

    async def start_desktop(self, replacement) -> None:
        starts.append(replacement)

    monkeypatch.setattr(OverlayApplicationOwner, "run_start", start_desktop)
    try:
        await asyncio.wait_for(owner.handle_start_failure("steamvr_not_running"), timeout=1.0)
        await asyncio.wait_for(detach_entered.wait(), timeout=1.0)
        await asyncio.wait_for(owner.set_enabled(False), timeout=1.0)
        assert detach_cancelled.is_set()
        assert owner.state == "off"
        assert output.overlay_sink is None
        assert not runtime.has_resources()
        assert not starts
        assert not owner._retired_runtimes
    finally:
        await owner.close()
