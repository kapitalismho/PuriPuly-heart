from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from puripuly_heart.app.services.overlay_application import (
    OVERLAY_STARTUP_TIMEOUT_MS,
    OverlayApplicationOwner,
    OverlayApplicationState,
)

from puripuly_heart.app.ports.ui_models import OverlayPeerPresentationState
from puripuly_heart.app.services.peer_application import (
    PeerApplicationOwner,
    PeerApplicationSnapshot,
    PeerApplicationState,
)
from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.config.resolved import ResolvedOverlayConfig
from puripuly_heart.core.clock import FakeClock
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


class PeerOverlayHarness:
    def __init__(self) -> None:
        self.overlay_state = "starting"
        self.peer_intent_enabled = False
        self.states: list[tuple[str, str | None]] = []
        self.peer_surface_states: list[str] = []
        self.fallback_notices: list[bool] = []
        self.logs: list[str] = []
        self.refresh_error: Exception | None = None
        self.peer = PeerApplicationOwner(
            state_provider=self.peer_state,
            config_factory=lambda: cast(object, object()),
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
            log_detailed=lambda _message: None,
            log_failure=lambda _message: None,
        )
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
            disable_peer_intent=self.peer.disable_for_overlay,
            sync_peer_effective=self.peer.sync_effective_flags,
            cancel_peer_activation=self.peer.cancel_activation_starting,
            refresh_peer_dependencies=self.refresh_peer,
            presentation_sink=self.presentation_sink,
            state_sink=self.state_sink,
            fallback_notice_sink=self.fallback_notices.append,
            cancel_bounds_persistence=_noop_async,
            clear_bounds_suppressed=lambda: None,
            calibration_provider=lambda: cast(OverlayCalibration, object()),
            logging_mode_provider=lambda: "basic",
            log_dir_provider=lambda: "",
            desktop_controls_factory=lambda _config: [],
            interaction_mode_sink=lambda _mode: None,
            bounds_control_sink=lambda _control: None,
            renderer_event_consumer=_noop_renderer,
            edit_interaction_mode="edit",
            clock=FakeClock(_now=0.0),
            log_basic=lambda message, _level: self.logs.append(message),
            log_detailed=lambda _message, _level, _exception: False,
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
            overlay_command_available=True,
        )

    async def ensure_local_ready(self, _generation: int) -> bool:
        return True

    async def refresh_peer(self) -> None:
        if self.refresh_error is not None:
            raise self.refresh_error
        self.peer.sync_effective_flags()

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
        disable_peer_intent=lambda: None,
        sync_peer_effective=recorder.sync_peer_effective,
        cancel_peer_activation=recorder.cancel_peer_activation,
        refresh_peer_dependencies=_noop_async,
        presentation_sink=recorder.presentation_sink,
        state_sink=recorder.state_sink,
        fallback_notice_sink=lambda _active: None,
        cancel_bounds_persistence=_noop_async,
        clear_bounds_suppressed=lambda: None,
        calibration_provider=lambda: cast(OverlayCalibration, object()),
        logging_mode_provider=lambda: "basic",
        log_dir_provider=lambda: "",
        desktop_controls_factory=lambda _config: [],
        interaction_mode_sink=lambda _mode: None,
        bounds_control_sink=lambda _control: None,
        renderer_event_consumer=_noop_renderer,
        edit_interaction_mode="edit",
        clock=FakeClock(_now=0.0),
        log_basic=recorder.log_basic,
        log_detailed=lambda _message, _level, _exception: False,
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
    recorder = Recorder()
    owner = make_owner(recorder)

    owner.on_start_failed("startup_timeout")

    assert owner.state == "failed"
    assert recorder.cancel_peer_activation_calls == 1
    assert recorder.states == [("failed", "startup_timeout")]


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


async def test_fallback_refresh_failure_keeps_one_actionable_terminal_reason() -> None:
    harness = PeerOverlayHarness()
    await harness.activate_peer()
    harness.refresh_error = RuntimeError("peer refresh failed")

    await harness.overlay.handle_start_failure("steamvr_not_running")

    harness.assert_terminal_fallback_failure(
        expected_notices=[],
        expected_surfaces=["starting", "warning"],
    )
    assert harness.states == [
        ("starting", None),
        ("failed", "steamvr_not_running"),
    ]


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
