from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from puripuly_heart.app.ports.ui_models import OverlayPeerPresentationState
from puripuly_heart.app.services.overlay.overlay_application import (
    OverlayApplicationOwner,
    OverlayApplicationState,
)
from puripuly_heart.app.services.overlay.overlay_generation_start import (
    OverlayGenerationStartOwner,
)
from puripuly_heart.app.services.peer_application import PeerApplicationSnapshot
from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.config.resolved import ResolvedOverlayConfig
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.runtime.overlay import OverlayRuntimeHandle
from puripuly_heart.ui.overlay_peer_contract import (
    build_overlay_peer_consumer_contract_from_state,
)


async def _noop_async() -> None:
    return None


async def _noop_renderer(queue: Any, overlay_instance_id: str) -> None:
    _ = queue, overlay_instance_id


class RecoveryManager:
    def __init__(
        self,
        *,
        state: str = "failed",
        failure_reason: str | None = "window_reveal_lost",
        eligible: bool = True,
        evidence: dict[str, object] | None = None,
        instance_id: str = "overlay-A",
        stop_error: BaseException | None = None,
        cleanup_complete: bool = True,
        first_visible: bool = False,
    ) -> None:
        self.state = state
        self.failure_reason = failure_reason
        self.startup_recovery_eligible = eligible
        self.startup_failure_evidence = evidence
        self.overlay_instance_id = instance_id
        self.desktop_cleanup_complete = cleanup_complete
        self.desktop_first_visible = first_visible
        self.stop_calls = 0
        self._stop_error = stop_error
        self._monitor_task: asyncio.Task[None] | None = None

    def mark_shutdown_requested(self) -> None:
        return None

    async def stop(self) -> None:
        self.stop_calls += 1
        if self._stop_error is not None:
            raise self._stop_error


def _evidence(
    reason: str = "window_reveal_lost",
    *,
    desktop_target: bool = True,
) -> dict[str, object]:
    return {
        "failure_reason": reason,
        "startup_phase": "bounds_confirmed",
        "canonical_bounds": (593, 864, 1344, 336),
        "observed_bounds": (593, 864, 1344, 336),
        "bounds_drift": False,
        "title_confirmed": True,
        "visible_confirmed": False,
        "bounds_confirmed": True,
        "win32_error": None,
        "port_reason": "visible_bounds_not_retained",
        "hwnd": 12345,
        "owner_pid": 10572,
        "hwnd_owner_pid": 10572,
        "pid_file_pid": 10572,
        "endpoint_identity": "page",
        "endpoint_matches": True,
        "desktop_target": desktop_target,
        "target": "desktop",
    }


class RecoveryPresenter:
    def __init__(self) -> None:
        self.detach_bridge_calls = 0

    def detach_bridge(self) -> None:
        self.detach_bridge_calls += 1


class RecoveryBridge:
    def __init__(self) -> None:
        self.stop_calls = 0

    async def stop(self) -> None:
        self.stop_calls += 1


class Harness:
    def __init__(self) -> None:
        self.states: list[tuple[str, str | None]] = []
        self.presentations: list[OverlayPeerPresentationState | None] = []
        self.refresh_calls = 0
        self.refresh_error: BaseException | None = None
        self.transition_calls = 0
        self.next_instance = "overlay-B"
        self.connect_new_runtime = False
        self.intent_enabled = True
        self.configured_target = "desktop"
        self.owner = OverlayApplicationOwner(
            state_provider=lambda: OverlayApplicationState(
                settings_available=True,
                overlay_intent_enabled=self.intent_enabled,
                configured_target=self.configured_target,
                locale="en",
            ),
            config_provider=lambda: cast(ResolvedOverlayConfig, object()),
            overlay_intent_sink=lambda _enabled: None,
            output_provider=lambda: None,
            diagnostics_provider=lambda: None,
            peer_snapshot_provider=lambda: PeerApplicationSnapshot(
                intent_enabled=False,
                activation_requested=False,
                effective_enabled=False,
                desired_active=False,
                activation_generation=0,
                activation_starting=False,
                model_loading=False,
                process_warning_reason=None,
                runtime_signature=None,
                provider_signature=None,
            ),
            disable_peer_intent=lambda: None,
            sync_peer_effective=lambda: None,
            cancel_peer_activation=lambda: None,
            refresh_peer_dependencies=self.refresh_peer,
            presentation_sink=self.presentations.append,
            state_sink=lambda state, reason: self.states.append((state, reason)),
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
            log_basic=lambda _message, _level: None,
            log_detailed=lambda _message, _level, _exception: False,
            translation_enabled_provider=lambda: True,
        )
        self.owner.state = "starting"
        self.owner.active_target = "desktop"

    async def refresh_peer(self) -> None:
        self.refresh_calls += 1
        if self.refresh_error is not None:
            raise self.refresh_error

    def attach_failed_manager(
        self,
        manager: RecoveryManager,
        instance_id: str,
    ) -> OverlayRuntimeHandle:
        runtime = OverlayRuntimeHandle(shutdown_grace_s=0)
        runtime.set_overlay_instance_id(instance_id)
        runtime.attach_process_manager(cast(Any, manager))
        self.owner.runtime = runtime
        return runtime

    async def attach_failed_generation(
        self,
        manager: RecoveryManager,
        instance_id: str,
    ) -> tuple[OverlayRuntimeHandle, RecoveryPresenter, RecoveryBridge]:
        runtime = self.attach_failed_manager(manager, instance_id)
        presenter = RecoveryPresenter()
        runtime.attach_presenter(presenter)
        runtime.attach_diagnostics(object())
        bridge = RecoveryBridge()
        runtime.attach_bridge(cast(Any, bridge))
        runtime.attach_renderer_events(asyncio.Queue())

        async def _idle() -> None:
            await asyncio.sleep(30)

        runtime.create_child_task(_idle(), task_name="idle-probe")
        return runtime, presenter, bridge

    async def await_recovery_replacement(self) -> None:
        task = self.owner._startup_recovery_task
        if task is not None:
            await task

    def install_transition_stub(self) -> None:
        harness = self

        class StubTransition:
            async def begin_start(self, execution_factory: Any) -> str:
                harness.transition_calls += 1
                execution = execution_factory()
                teardown_ok = await execution.teardown()
                if not teardown_ok:
                    return "teardown_failed"
                runtime = execution.create_runtime()
                runtime.set_overlay_instance_id(harness.next_instance)
                execution.on_starting(runtime, execution.resolve_target())
                if harness.connect_new_runtime:
                    harness.owner.mark_connected()
                return "started"

            async def shutdown(self, execution_factory: Any) -> str:
                execution = execution_factory()
                execution.on_stopping()
                teardown_ok = await execution.teardown()
                if not teardown_ok and execution.has_resources_after_teardown():
                    await execution.on_failed()
                    return "failed"
                await execution.on_stopped()
                return "stopped"

        self.owner._transition_owner = cast(Any, StubTransition())


async def test_eligible_recovery_reaps_a_before_distinct_b_and_connects() -> None:
    harness = Harness()
    harness.install_transition_stub()
    evidence = _evidence("window_reveal_lost")
    manager_a = RecoveryManager(evidence=evidence, instance_id="overlay-A")
    runtime_a, presenter_a, bridge_a = await harness.attach_failed_generation(
        manager_a, "overlay-A"
    )

    await harness.owner.handle_start_failure("window_reveal_lost")

    assert harness.owner.state == "recovering"
    assert harness.owner.snapshot.recovery_active is True
    assert harness.owner.snapshot.recovery_reason == "window_reveal_lost"
    assert ("recovering", None) in harness.states
    assert runtime_a.is_closed
    assert runtime_a.transfer_reap_complete()
    assert manager_a.stop_calls == 1
    assert bridge_a.stop_calls == 1
    assert runtime_a.bridge is None
    assert runtime_a.renderer_events is None
    assert runtime_a.presenter is presenter_a
    await harness.await_recovery_replacement()
    assert harness.transition_calls == 1
    runtime_b = harness.owner.runtime
    assert runtime_b is not None and runtime_b is not runtime_a
    assert runtime_b.overlay_instance_id == "overlay-B"
    assert harness.owner.startup_recovery is not None
    assert harness.owner.startup_recovery["failed_overlay_instance_id"] == "overlay-A"

    contract = build_overlay_peer_consumer_contract_from_state(
        cast(OverlayPeerPresentationState, harness.owner.presentation_state())
    )
    assert contract.overlay.state == "on"

    harness.owner.mark_connected()

    assert harness.owner.state == "connected"
    assert harness.owner.snapshot.recovery_active is False
    last = harness.owner.last_startup_recovery
    assert last is not None
    assert last["failed_overlay_instance_id"] == "overlay-A"
    assert last["replacement_overlay_instance_id"] == "overlay-B"
    assert last["failure_reason"] == "window_reveal_lost"
    assert last["outcome"] == "connected"


async def test_terminal_repeat_after_replacement_failure() -> None:
    harness = Harness()
    harness.install_transition_stub()
    manager_a = RecoveryManager(evidence=_evidence("window_reveal_lost"), instance_id="overlay-A")
    harness.attach_failed_manager(manager_a, "overlay-A")

    await harness.owner.handle_start_failure("window_reveal_lost")
    assert harness.owner.state == "recovering"
    await harness.await_recovery_replacement()
    assert harness.transition_calls == 1

    manager_b = RecoveryManager(
        evidence=_evidence("window_visibility_unstable"),
        failure_reason="window_visibility_unstable",
        instance_id="overlay-B",
    )
    runtime_b = harness.owner.runtime
    assert runtime_b is not None
    assert runtime_b.overlay_instance_id == "overlay-B"
    runtime_b.attach_process_manager(cast(Any, manager_b))

    await harness.owner.handle_start_failure("window_visibility_unstable")

    assert harness.owner.state == "failed"
    assert harness.owner.failure_reason == "window_visibility_unstable"
    assert harness.transition_calls == 1
    last = harness.owner.last_startup_recovery
    assert last is not None
    assert last["failed_overlay_instance_id"] == "overlay-A"
    assert last["terminal_reason"] == "window_visibility_unstable"


async def test_recovery_never_reprobes_steamvr_and_preserves_fallback() -> None:
    harness = Harness()
    harness.install_transition_stub()
    harness.owner.fallback_owner.activate("steamvr_not_running")
    harness.owner.publish_fallback(True)
    manager_a = RecoveryManager(evidence=_evidence(), instance_id="overlay-A")
    harness.attach_failed_manager(manager_a, "overlay-A")

    await harness.owner.handle_start_failure("window_reveal_lost")

    assert harness.owner.state == "recovering"
    await harness.await_recovery_replacement()
    assert harness.owner.snapshot.fallback_active is True
    assert harness.owner.fallback_owner.task is None
    assert harness.owner.effective_target_for_start() == "desktop"


@pytest.mark.parametrize(
    "reason",
    [
        "window_configuration_failed",
        "unknown",
    ],
)
async def test_non_recoverable_reasons_go_terminal_without_replacement(
    reason: str,
) -> None:
    harness = Harness()
    harness.install_transition_stub()
    manager = RecoveryManager(
        failure_reason=reason,
        eligible=False,
        evidence=_evidence(reason) if reason.startswith("window_") else None,
        instance_id="overlay-A",
    )
    harness.attach_failed_manager(manager, "overlay-A")

    await harness.owner.handle_start_failure(reason)

    assert harness.owner.state == "failed"
    assert harness.owner.failure_reason == reason
    assert harness.transition_calls == 0
    assert harness.owner.snapshot.recovery_active is False


async def test_eligible_reason_without_sibling_approval_goes_terminal() -> None:
    harness = Harness()
    harness.install_transition_stub()
    manager = RecoveryManager(
        failure_reason="window_reveal_lost",
        eligible=False,
        evidence=_evidence("window_reveal_lost"),
        instance_id="overlay-A",
    )
    harness.attach_failed_manager(manager, "overlay-A")

    await harness.owner.handle_start_failure("window_reveal_lost")

    assert harness.owner.state == "failed"
    assert harness.transition_calls == 0


async def test_imperfect_evidence_still_recovers_when_sibling_approves() -> None:
    harness = Harness()
    harness.install_transition_stub()
    bad = _evidence("window_reveal_lost")
    bad["desktop_target"] = False
    bad["title_confirmed"] = False
    bad.pop("hwnd", None)
    manager = RecoveryManager(
        failure_reason="window_reveal_lost",
        eligible=True,
        evidence=bad,
        instance_id="overlay-A",
    )
    harness.attach_failed_manager(manager, "overlay-A")

    await harness.owner.handle_start_failure("window_reveal_lost")

    assert harness.owner.state == "recovering"
    await harness.await_recovery_replacement()
    assert harness.transition_calls == 1


async def test_missing_cleanup_proof_goes_terminal_without_replacement() -> None:
    harness = Harness()
    harness.install_transition_stub()
    manager = RecoveryManager(
        evidence=_evidence(),
        instance_id="overlay-A",
        cleanup_complete=False,
    )
    harness.attach_failed_manager(manager, "overlay-A")

    await harness.owner.handle_start_failure("window_reveal_lost")

    assert harness.owner.state == "failed"
    assert harness.owner.failure_reason == "window_reveal_lost"
    assert harness.transition_calls == 0
    assert harness.owner.snapshot.recovery_active is False


async def test_uncertain_teardown_blocks_replacement() -> None:
    harness = Harness()
    harness.install_transition_stub()
    manager = RecoveryManager(
        evidence=_evidence(),
        instance_id="overlay-A",
        stop_error=RuntimeError("cannot reap child"),
    )
    harness.attach_failed_manager(manager, "overlay-A")

    await harness.owner.handle_start_failure("window_reveal_lost")

    assert harness.owner.state == "failed"
    assert harness.owner.failure_reason == "window_reveal_lost"
    assert harness.transition_calls == 0


async def test_toggle_off_during_recovery_drains_without_leaks() -> None:
    harness = Harness()
    harness.install_transition_stub()
    manager = RecoveryManager(evidence=_evidence(), instance_id="overlay-A")
    harness.attach_failed_manager(manager, "overlay-A")

    await harness.owner.handle_start_failure("window_reveal_lost")
    assert harness.owner.state == "recovering"

    await harness.owner.set_enabled(False)

    assert harness.owner.startup_recovery is None
    assert harness.owner.snapshot.recovery_active is False
    assert harness.owner.state == "off"
    assert not OverlayApplicationOwner.runtime_has_resources(harness.owner.runtime)


async def test_stale_generation_events_do_not_mutate_connected_replacement() -> None:
    harness = Harness()
    harness.install_transition_stub()
    harness.connect_new_runtime = True
    manager_a = RecoveryManager(evidence=_evidence(), instance_id="overlay-A")
    runtime_a = harness.attach_failed_manager(manager_a, "overlay-A")

    await harness.owner.handle_start_failure("window_reveal_lost")
    assert harness.owner.state == "recovering"
    await harness.await_recovery_replacement()
    assert harness.owner.state == "connected"

    assert harness.owner.runtime_is_current(cast(OverlayRuntimeHandle, runtime_a)) is False
    stale_states = list(harness.states)
    assert harness.owner.state == "connected"
    assert harness.states == stale_states


async def test_recovery_adopts_presenter_through_real_transition(
    monkeypatch: Any,
) -> None:
    harness = Harness()
    manager_a = RecoveryManager(evidence=_evidence(), instance_id="overlay-A")
    runtime_a, presenter_a, bridge_a = await harness.attach_failed_generation(
        manager_a, "overlay-A"
    )
    seen_presenters: list[object | None] = []

    async def fake_run_start(self: OverlayApplicationOwner, runtime: Any = None) -> None:
        if runtime is None:
            runtime = self._runtime or self.new_runtime()
        runtime.set_overlay_instance_id("overlay-B")
        seen_presenters.append(runtime.presenter)
        self.mark_connected()

    monkeypatch.setattr(OverlayApplicationOwner, "run_start", fake_run_start)

    await harness.owner.handle_start_failure("window_reveal_lost")
    assert harness.owner.state == "recovering"
    recovery_task = harness.owner._startup_recovery_task
    assert recovery_task is not None
    await recovery_task
    for _ in range(100):
        if harness.owner.state == "connected":
            break
        await asyncio.sleep(0)

    assert harness.owner.state == "connected"
    assert runtime_a.is_closed
    assert runtime_a.presenter is None
    assert runtime_a.process_manager is None
    assert runtime_a.bridge is None
    assert bridge_a.stop_calls == 1
    runtime_b = harness.owner.runtime
    assert runtime_b is not None and runtime_b is not runtime_a
    assert runtime_b.overlay_instance_id == "overlay-B"
    assert runtime_b.presenter is presenter_a
    assert seen_presenters == [presenter_a]
    last = harness.owner.last_startup_recovery
    assert last is not None
    assert last["failed_overlay_instance_id"] == "overlay-A"
    assert last["replacement_overlay_instance_id"] == "overlay-B"
    assert last["outcome"] == "connected"


async def test_rapid_off_on_during_recovery_starts_fresh(monkeypatch: Any) -> None:
    harness = Harness()
    manager_a = RecoveryManager(evidence=_evidence(), instance_id="overlay-A")
    await harness.attach_failed_generation(manager_a, "overlay-A")
    created_ids: list[str | None] = []

    async def fake_run_start(self: OverlayApplicationOwner, runtime: Any = None) -> None:
        if runtime is None:
            runtime = self._runtime or self.new_runtime()
        created_ids.append("overlay-C")
        runtime.set_overlay_instance_id("overlay-C")
        self.mark_connected()

    monkeypatch.setattr(OverlayApplicationOwner, "run_start", fake_run_start)

    await harness.owner.handle_start_failure("window_reveal_lost")
    assert harness.owner.state == "recovering"

    await harness.owner.set_enabled(False)

    assert harness.owner.startup_recovery is None
    assert harness.owner.state == "off"
    assert harness.owner.runtime is None
    assert created_ids == []

    await harness.owner.set_enabled(True)
    for _ in range(100):
        if harness.owner.state == "connected":
            break
        await asyncio.sleep(0)

    assert harness.owner.state == "connected"
    assert harness.owner.startup_recovery is None
    assert harness.owner._desktop_startup_recovery_attempted is False
    assert created_ids == ["overlay-C"]


async def test_fallback_schedules_without_prestart_peer_refresh() -> None:
    harness = Harness()
    harness.configured_target = "steamvr"
    harness.owner.state = "starting"
    harness.owner.active_target = "steamvr"
    harness.refresh_error = RuntimeError("peer down")

    class OkTransition:
        async def begin_start(self, execution_factory: Any) -> str:
            execution = execution_factory()
            if not await execution.teardown():
                return "teardown_failed"
            runtime = execution.create_runtime()
            execution.on_starting(runtime, execution.resolve_target())
            return "started"

    harness.owner._transition_owner = cast(Any, OkTransition())
    await harness.owner.handle_start_failure("steamvr_not_running")

    assert harness.refresh_calls == 0
    assert harness.owner.snapshot.fallback_active is True
    task = harness.owner.fallback_owner.task
    assert task is not None
    await task


async def test_fallback_postconnect_refresh_failure_keeps_desktop_connected(
    monkeypatch: Any,
) -> None:
    import puripuly_heart.app.services.overlay.overlay_generation_start as gen_start
    import tests.app.test_overlay_generation_start_owner as gen_module

    gen_module.FakePresenter.instances = []
    gen_module.FakePresenter.events = []
    gen_module.FakeBridge.instances = []
    gen_module.FakeBridge.events = gen_module.FakePresenter.events
    gen_module.FakeBridge.start_failure = None
    gen_module.FakeProcessManager.instances = []
    gen_module.FakeProcessManager.events = gen_module.FakePresenter.events
    gen_module.FakeProcessManager.start_state = "connected"
    gen_module.FakeProcessManager.failure_reason = None
    gen_module.FakeProcessManager.after_start = None
    gen_module.FakeProcessManager.monitor_enabled = True
    monkeypatch.setattr(gen_start, "OverlayPresenter", gen_module.FakePresenter)
    monkeypatch.setattr(gen_start, "OverlayBridge", gen_module.FakeBridge)
    monkeypatch.setattr(gen_start, "OverlayProcessManager", gen_module.FakeProcessManager)

    owner = OverlayGenerationStartOwner()
    runtime = OverlayRuntimeHandle(shutdown_grace_s=0)
    logged: list[str] = []
    watched: list[str] = []

    async def failing_refresh() -> None:
        raise RuntimeError("peer refresh down")

    harness = gen_module.StartHarness()
    effects = harness.effects()
    object.__setattr__(effects, "refresh_dependencies", failing_refresh)
    object.__setattr__(
        effects,
        "log_failure",
        lambda message, level, exc: logged.append(message),
    )

    async def watch(manager: Any, monitor: asyncio.Task[None], rt: Any, instance_id: str) -> None:
        watched.append(instance_id)
        await monitor

    object.__setattr__(effects, "watch_runtime", watch)
    request = harness.request(desktop=True)
    object.__setattr__(request, "fallback_reason", "steamvr_not_running")

    status = await owner.start(runtime, lambda: request, effects)
    assert status == "connected"
    assert runtime.monitor_task is not None
    await runtime.monitor_task
    assert watched != []
    assert any("Peer dependency refresh failed" in message for message in logged)
    await runtime.close(
        preserve_presenter_state=False,
        overlay_sink_detach=None,
        preview_reset=None,
        diagnostics_detach=None,
        emit_shutdown=False,
    )


async def test_direct_postconnect_refresh_failure_preserves_terminal_semantics(
    monkeypatch: Any,
) -> None:
    import puripuly_heart.app.services.overlay.overlay_generation_start as gen_start
    import tests.app.test_overlay_generation_start_owner as gen_module

    gen_module.FakePresenter.instances = []
    gen_module.FakePresenter.events = []
    gen_module.FakeBridge.instances = []
    gen_module.FakeBridge.events = gen_module.FakePresenter.events
    gen_module.FakeBridge.start_failure = None
    gen_module.FakeProcessManager.instances = []
    gen_module.FakeProcessManager.events = gen_module.FakePresenter.events
    gen_module.FakeProcessManager.start_state = "connected"
    gen_module.FakeProcessManager.failure_reason = None
    gen_module.FakeProcessManager.after_start = None
    gen_module.FakeProcessManager.monitor_enabled = True
    monkeypatch.setattr(gen_start, "OverlayPresenter", gen_module.FakePresenter)
    monkeypatch.setattr(gen_start, "OverlayBridge", gen_module.FakeBridge)
    monkeypatch.setattr(gen_start, "OverlayProcessManager", gen_module.FakeProcessManager)

    owner = OverlayGenerationStartOwner()
    runtime = OverlayRuntimeHandle(shutdown_grace_s=0)

    async def failing_refresh() -> None:
        raise RuntimeError("peer refresh down")

    harness = gen_module.StartHarness()
    effects = harness.effects()
    object.__setattr__(effects, "refresh_dependencies", failing_refresh)
    request = harness.request(desktop=True)
    object.__setattr__(request, "fallback_reason", None)

    status = await owner.start(runtime, lambda: request, effects)

    assert status == "failed"
    await runtime.close(
        preserve_presenter_state=False,
        overlay_sink_detach=None,
        preview_reset=None,
        diagnostics_detach=None,
        emit_shutdown=False,
    )


@pytest.mark.parametrize(
    "reason",
    [
        "window_reveal_lost",
        "window_visibility_unstable",
        "window_observation_failed",
        "window_bounds_failed",
        "window_native_ready_failed",
        "window_identity_failed",
    ],
)
async def test_broadened_window_failures_recover_with_proven_cleanup(
    reason: str,
) -> None:
    harness = Harness()
    harness.install_transition_stub()
    manager = RecoveryManager(
        failure_reason=reason,
        eligible=True,
        evidence=_evidence(reason),
        instance_id="overlay-A",
        cleanup_complete=True,
    )
    harness.attach_failed_manager(manager, "overlay-A")

    await harness.owner.handle_start_failure(reason)

    assert harness.owner.state == "recovering"
    await harness.await_recovery_replacement()
    assert harness.transition_calls == 1
    assert harness.owner.startup_recovery is not None
    assert harness.owner.startup_recovery["failure_reason"] == reason
    assert harness.owner.startup_recovery["failed_overlay_instance_id"] == "overlay-A"


@pytest.mark.parametrize("evidence", [None, {}, {"failure_reason": "window_reveal_lost"}])
async def test_sparse_evidence_still_recovers_when_sibling_approves(
    evidence: dict[str, object] | None,
) -> None:
    harness = Harness()
    harness.install_transition_stub()
    manager = RecoveryManager(
        failure_reason="window_reveal_lost",
        eligible=True,
        evidence=evidence,
        instance_id="overlay-A",
        cleanup_complete=True,
    )
    harness.attach_failed_manager(manager, "overlay-A")

    await harness.owner.handle_start_failure("window_reveal_lost")

    assert harness.owner.state == "recovering"
    await harness.await_recovery_replacement()
    assert harness.transition_calls == 1


async def test_native_ready_timeout_recovers_without_forensics() -> None:
    harness = Harness()
    harness.install_transition_stub()
    manager = RecoveryManager(
        failure_reason="window_native_ready_failed",
        eligible=True,
        evidence={
            "failure_reason": "window_native_ready_failed",
            "port_reason": "native_ready_timeout",
        },
        instance_id="overlay-A",
        cleanup_complete=True,
    )
    harness.attach_failed_manager(manager, "overlay-A")

    await harness.owner.handle_start_failure("window_native_ready_failed")

    assert harness.owner.state == "recovering"
    await harness.await_recovery_replacement()
    assert harness.transition_calls == 1


async def test_first_visible_stops_spinner_before_connected() -> None:
    harness = Harness()
    manager = RecoveryManager(
        state="starting",
        failure_reason=None,
        eligible=False,
        evidence=None,
        instance_id="overlay-A",
        cleanup_complete=False,
        first_visible=True,
    )
    harness.attach_failed_manager(manager, "overlay-A")
    harness.owner.state = "starting"
    harness.owner.active_target = "desktop"

    state = harness.owner.presentation_state()
    assert state is not None
    assert state.desktop_first_visible is True

    contract = build_overlay_peer_consumer_contract_from_state(state)
    assert contract.overlay.state == "on"
    assert contract.overlay.effective_enabled is False
    assert contract.desktop_first_visible is True

    from puripuly_heart.ui.dashboard.capture import capture_presentation_from_contract

    presentation = capture_presentation_from_contract(contract)
    assert presentation.overlay.enabled is True
    assert presentation.overlay.starting is False


async def test_starting_without_first_visible_keeps_spinner() -> None:
    harness = Harness()
    manager = RecoveryManager(
        state="starting",
        failure_reason=None,
        eligible=False,
        evidence=None,
        instance_id="overlay-A",
        cleanup_complete=False,
        first_visible=False,
    )
    harness.attach_failed_manager(manager, "overlay-A")
    harness.owner.state = "starting"
    harness.owner.active_target = "desktop"

    state = harness.owner.presentation_state()
    assert state is not None
    assert state.desktop_first_visible is False

    contract = build_overlay_peer_consumer_contract_from_state(state)

    from puripuly_heart.ui.dashboard.capture import capture_presentation_from_contract

    presentation = capture_presentation_from_contract(contract)
    assert presentation.overlay.starting is True


async def test_recovery_resets_first_visible_for_replacement_spinner() -> None:
    harness = Harness()
    harness.install_transition_stub()
    manager_a = RecoveryManager(
        evidence=_evidence(),
        instance_id="overlay-A",
        cleanup_complete=True,
        first_visible=True,
    )
    harness.attach_failed_manager(manager_a, "overlay-A")

    await harness.owner.handle_start_failure("window_reveal_lost")
    assert harness.owner.state == "recovering"
    await harness.await_recovery_replacement()

    runtime_b = harness.owner.runtime
    assert runtime_b is not None
    assert runtime_b.overlay_instance_id == "overlay-B"
    assert runtime_b.process_manager is None

    state = harness.owner.presentation_state()
    assert state is not None
    assert state.desktop_first_visible is False

    from puripuly_heart.ui.dashboard.capture import capture_presentation_from_contract

    contract = build_overlay_peer_consumer_contract_from_state(state)
    presentation = capture_presentation_from_contract(contract)
    assert presentation.overlay.starting is True


async def test_stale_first_visible_callback_is_ignored() -> None:
    harness = Harness()
    harness.install_transition_stub()
    manager_a = RecoveryManager(evidence=_evidence(), instance_id="overlay-A")
    runtime_a = harness.attach_failed_manager(manager_a, "overlay-A")

    await harness.owner.handle_start_failure("window_reveal_lost")
    assert harness.owner.state == "recovering"
    await harness.await_recovery_replacement()
    runtime_b = harness.owner.runtime
    assert runtime_b is not None

    presentations_before = list(harness.presentations)
    harness.owner.on_desktop_first_visible(runtime_a, "overlay-A")
    assert harness.presentations == presentations_before


async def test_duplicate_first_visible_callbacks_publish_idempotently() -> None:
    harness = Harness()
    manager = RecoveryManager(
        state="starting",
        failure_reason=None,
        eligible=False,
        evidence=None,
        instance_id="overlay-A",
        cleanup_complete=False,
        first_visible=True,
    )
    runtime = harness.attach_failed_manager(manager, "overlay-A")
    harness.owner.state = "starting"
    harness.owner.active_target = "desktop"

    harness.owner.on_desktop_first_visible(runtime, "overlay-A")
    first_count = len(harness.presentations)
    harness.owner.on_desktop_first_visible(runtime, "overlay-A")
    assert len(harness.presentations) == first_count + 1
    assert harness.presentations[-1] is not None
    assert harness.presentations[-1].desktop_first_visible is True  # type: ignore[union-attr]


async def test_shutdown_waits_for_canceled_recovery_cleanup() -> None:
    harness = Harness()
    entered = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_finished = asyncio.Event()

    class BlockingTransition:
        async def begin_start(self, execution_factory: Any) -> str:
            entered.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cleanup_started.set()
                await asyncio.sleep(0.2)
                cleanup_finished.set()
                raise
            return "started"

        async def shutdown(self, execution_factory: Any) -> str:
            execution = execution_factory()
            execution.on_stopping()
            teardown_ok = await execution.teardown()
            if not teardown_ok and execution.has_resources_after_teardown():
                await execution.on_failed()
                return "failed"
            await execution.on_stopped()
            return "stopped"

    harness.owner._transition_owner = cast(Any, BlockingTransition())
    manager = RecoveryManager(evidence=_evidence(), instance_id="overlay-A")
    harness.attach_failed_manager(manager, "overlay-A")

    await harness.owner.handle_start_failure("window_reveal_lost")
    assert harness.owner.state == "recovering"
    await asyncio.wait_for(entered.wait(), timeout=5)
    recovery_task = harness.owner._startup_recovery_task
    assert recovery_task is not None

    shutdown_task = asyncio.create_task(harness.owner.shutdown(preserve_failure_reason=True))
    await asyncio.wait_for(cleanup_started.wait(), timeout=5)
    assert not shutdown_task.done()
    await asyncio.wait_for(asyncio.shield(shutdown_task), timeout=5)
    assert cleanup_finished.is_set()
    assert harness.owner.state == "off"
    assert harness.owner._startup_recovery_task is None
    assert recovery_task.done()

    harness.install_transition_stub()
    status = await asyncio.wait_for(harness.owner.begin_start(), timeout=5)
    assert status == "started"
    await harness.owner.close()
    await harness.owner.shutdown(preserve_failure_reason=True)


async def test_close_drains_owned_recovery_failure_without_propagating() -> None:
    harness = Harness()
    harness.install_transition_stub()

    async def failing() -> None:
        await asyncio.sleep(0)
        raise RuntimeError("recovery boom")

    task = asyncio.create_task(failing())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert task.done()
    harness.owner._startup_recovery_task = task
    await harness.owner.close()
    assert harness.owner._startup_recovery_task is None
    assert task.done()
    assert isinstance(task.exception(), RuntimeError)
    await harness.owner.close()
    await harness.owner.shutdown(preserve_failure_reason=True)
