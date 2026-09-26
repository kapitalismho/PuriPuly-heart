from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

ft = pytest.importorskip("flet")

from puripuly_heart.app.ports.ui_models import OverlayPeerPresentationState
from puripuly_heart.app.services.overlay.overlay_application import (
    OverlayApplicationOwner,
    OverlayApplicationState,
)
from puripuly_heart.app.services.overlay.overlay_session_transition import (
    OverlaySessionTransitionOwner,
)
from puripuly_heart.app.services.peer_application import PeerApplicationSnapshot
from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.config.resolved import ResolvedOverlayConfig
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.ui.dashboard.capture import (
    DashboardCaptureControls,
    capture_presentation_from_contract,
)
from puripuly_heart.ui.overlay_peer_contract import (
    build_overlay_peer_consumer_contract_from_state,
)
from puripuly_heart.ui.theme import COLOR_PRIMARY, COLOR_SURFACE, COLOR_WARNING


async def _noop_async() -> None:
    return None


async def _noop_renderer(queue: Any, overlay_instance_id: str) -> None:
    _ = queue, overlay_instance_id


def _peer_snapshot() -> PeerApplicationSnapshot:
    return PeerApplicationSnapshot(
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
    )


class PendingStartTransition:
    def __init__(self) -> None:
        self.begin_calls = 0

    async def begin_start(self, execution_factory: Any) -> str:
        self.begin_calls += 1
        execution = execution_factory()
        teardown_ok = await execution.teardown()
        if not teardown_ok:
            return "teardown_failed"
        runtime = execution.create_runtime()
        execution.on_starting(runtime, execution.resolve_target())
        return "started"

    async def shutdown(self, execution_factory: Any) -> str:
        return await OverlaySessionTransitionOwner().shutdown(execution_factory)


class BlockedStartTransition(PendingStartTransition):
    async def begin_start(self, execution_factory: Any) -> str:
        execution = execution_factory()
        await execution.teardown()
        return "teardown_failed"


class RaisingStartTransition(PendingStartTransition):
    async def begin_start(self, execution_factory: Any) -> str:
        raise RuntimeError("start transition failed")


class Harness:
    def __init__(self) -> None:
        self.intent_enabled = False
        self.configured_target = "desktop"
        self.states: list[tuple[str, str | None]] = []
        self.presentations: list[OverlayPeerPresentationState | None] = []
        self.owner = OverlayApplicationOwner(
            state_provider=lambda: OverlayApplicationState(
                settings_available=True,
                overlay_intent_enabled=self.intent_enabled,
                configured_target=self.configured_target,
                locale="en",
            ),
            config_provider=lambda: cast(ResolvedOverlayConfig, object()),
            overlay_intent_sink=lambda enabled: setattr(self, "intent_enabled", bool(enabled)),
            output_provider=lambda: None,
            diagnostics_provider=lambda: None,
            peer_snapshot_provider=_peer_snapshot,
            sync_peer_effective=lambda: None,
            cancel_peer_activation=lambda: None,
            refresh_peer_dependencies=_noop_async,
            presentation_sink=self.presentations.append,
            state_sink=lambda state, reason: self.states.append((state, reason)),
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
            log_basic=lambda _message, _level: None,
            log_diagnostic=lambda _message, _level, _exception: False,
            translation_enabled_provider=lambda: True,
        )
        self.controls = DashboardCaptureControls(
            on_self_capture_click=lambda: None,
            on_peer_capture_click=lambda: None,
            on_overlay_click=lambda: None,
        )

    def draw_new_presentations(self, start_index: int) -> list[SimpleNamespace]:
        draws: list[SimpleNamespace] = []
        for presented in self.presentations[start_index:]:
            if presented is None:
                continue
            contract = build_overlay_peer_consumer_contract_from_state(presented)
            projection = capture_presentation_from_contract(contract)
            self.controls.apply_presentation(projection)
            button = self.controls.overlay_control()
            draws.append(
                SimpleNamespace(
                    warning=projection.overlay.warning,
                    starting=projection.overlay.starting,
                    enabled=projection.overlay.enabled,
                    bgcolor=button.bgcolor,
                    spinner_visible=button._progress_control.visible,
                    icon_visible=button._icon_control.visible,
                )
            )
        return draws


async def test_enable_from_off_first_draws_starting_spinner_without_warning() -> None:
    harness = Harness()
    harness.owner._transition_owner = cast(Any, PendingStartTransition())
    assert harness.owner.state == "off"

    marked = len(harness.presentations)
    await harness.owner.set_enabled(True)
    draws = harness.draw_new_presentations(marked)

    assert harness.intent_enabled is True
    assert harness.owner.state == "starting"
    first = draws[0]
    assert first.warning is False
    assert first.starting is True
    assert first.enabled is True
    assert first.bgcolor == COLOR_SURFACE
    assert first.bgcolor != COLOR_WARNING
    assert first.spinner_visible is True
    assert first.icon_visible is False

    marked = len(harness.presentations)
    harness.owner.mark_connected()
    draws = harness.draw_new_presentations(marked)

    final = draws[0]
    assert final.warning is False
    assert final.starting is False
    assert final.enabled is True
    assert final.bgcolor == COLOR_PRIMARY
    assert final.spinner_visible is False
    assert final.icon_visible is True


async def test_blocked_start_stays_warning_actionable() -> None:
    harness = Harness()
    harness.owner._transition_owner = cast(Any, BlockedStartTransition())

    marked = len(harness.presentations)
    await harness.owner.set_enabled(True)
    draws = harness.draw_new_presentations(marked)

    assert harness.intent_enabled is True
    only = draws[-1]
    assert only.warning is True
    assert only.starting is False
    assert only.bgcolor == COLOR_WARNING
    assert only.spinner_visible is False


async def test_start_exception_still_publishes_intent_truth() -> None:
    harness = Harness()
    harness.owner._transition_owner = cast(Any, RaisingStartTransition())

    marked = len(harness.presentations)
    with pytest.raises(RuntimeError, match="start transition failed"):
        await harness.owner.set_enabled(True)
    draws = harness.draw_new_presentations(marked)

    assert harness.intent_enabled is True
    assert draws[-1].warning is True
    assert draws[-1].bgcolor == COLOR_WARNING


async def test_reopen_from_failure_draws_spinner_before_any_warning() -> None:
    harness = Harness()
    harness.owner._transition_owner = cast(Any, PendingStartTransition())
    harness.intent_enabled = True
    harness.owner.on_start_failed("startup_timeout")
    assert harness.owner.state == "failed"
    base = len(harness.presentations)

    await harness.owner.set_enabled(True)
    draws = harness.draw_new_presentations(base)

    assert harness.owner.state == "starting"
    assert draws[0].warning is False
    assert draws[0].starting is True
    assert draws[0].bgcolor == COLOR_SURFACE
    assert draws[0].spinner_visible is True


async def test_disable_while_starting_clears_spinner_immediately() -> None:
    harness = Harness()
    harness.owner._transition_owner = cast(Any, PendingStartTransition())

    await harness.owner.set_enabled(True)
    assert harness.owner.state == "starting"

    marked = len(harness.presentations)
    await harness.owner.set_enabled(False)
    draws = harness.draw_new_presentations(marked)

    assert harness.intent_enabled is False
    assert harness.owner.state == "off"
    assert draws[0].starting is False
    assert draws[0].warning is False
    assert draws[0].spinner_visible is False
    assert draws[0].bgcolor != COLOR_WARNING
    assert all(draw.warning is False for draw in draws)
    assert all(draw.starting is False for draw in draws)


async def test_redundant_enable_while_connected_keeps_connected_without_warning() -> None:
    harness = Harness()
    harness.intent_enabled = True
    harness.owner.state = "connected"

    marked = len(harness.presentations)
    await harness.owner.set_enabled(True)
    draws = harness.draw_new_presentations(marked)
    await asyncio.sleep(0)

    assert harness.owner.state == "connected"
    assert draws[0].warning is False
    assert draws[0].enabled is True
    assert draws[0].bgcolor == COLOR_PRIMARY


@pytest.mark.parametrize("latest_enabled", [False, True])
async def test_toggle_during_desktop_shutdown_updates_button_before_cleanup_and_honors_latest_intent(
    monkeypatch: pytest.MonkeyPatch, latest_enabled: bool
) -> None:
    harness = Harness()
    owner = harness.owner
    harness.intent_enabled = True
    owner.state = "connected"
    owner.active_target = "desktop"
    previous = owner.new_runtime()
    stop_entered = asyncio.Event()
    release_stop = asyncio.Event()
    intent_received = asyncio.Event()
    started = asyncio.Event()
    starts = []

    async def stop_previous() -> None:
        stop_entered.set()
        await release_stop.wait()

    def accept_intent(enabled: bool) -> None:
        harness.intent_enabled = enabled
        intent_received.set()

    async def start_replacement(self, runtime) -> None:
        assert release_stop.is_set()
        starts.append(runtime)
        self.mark_connected()
        started.set()

    previous.attach_process_manager(
        SimpleNamespace(stop=stop_previous, desktop_first_visible=True, state="connected")
    )
    owner.overlay_intent_sink = accept_intent
    monkeypatch.setattr(OverlayApplicationOwner, "run_start", start_replacement)
    tasks = [asyncio.create_task(owner.set_enabled(False))]
    try:
        await asyncio.wait_for(stop_entered.wait(), timeout=1)
        marked = len(harness.presentations)
        intent_received.clear()
        tasks.append(asyncio.create_task(owner.set_enabled(True)))
        await asyncio.wait_for(intent_received.wait(), timeout=1)
        waiting = harness.draw_new_presentations(marked)[-1]
        assert waiting.spinner_visible is True
        assert waiting.warning is False
        assert owner.state == "stopping"
        assert not starts
        if not latest_enabled:
            marked = len(harness.presentations)
            intent_received.clear()
            tasks.append(asyncio.create_task(owner.set_enabled(False)))
            await asyncio.wait_for(intent_received.wait(), timeout=1)
            off = harness.draw_new_presentations(marked)[-1]
            assert off.spinner_visible is False
            assert off.enabled is False
        release_stop.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=1)
        if latest_enabled:
            await asyncio.wait_for(started.wait(), timeout=1)
            assert owner.state == "connected"
            assert owner.runtime is starts[0]
            assert owner.runtime is not previous
        else:
            assert not starts
            assert owner.runtime is None
            assert owner.state == "off"
    finally:
        release_stop.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await owner.close()


async def test_retired_vr_cleanup_does_not_block_toggle_but_is_owned_until_app_close() -> None:
    harness = Harness()
    owner = harness.owner
    owner._transition_owner = cast(Any, PendingStartTransition())
    retired = owner.new_runtime()
    owner.runtime = None
    entered = asyncio.Event()
    release = asyncio.Event()

    async def stop_retired() -> None:
        entered.set()
        await release.wait()

    retired.attach_process_manager(SimpleNamespace(stop=stop_retired))
    reaper = asyncio.create_task(owner._close_retired_runtime(retired))
    owner._retired_runtimes[retired] = reaper
    close_task = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        await asyncio.wait_for(owner.set_enabled(True), timeout=1)
        await asyncio.wait_for(owner.set_enabled(False), timeout=1)
        await asyncio.wait_for(owner.set_enabled(True), timeout=1)
        replacement = owner.runtime
        assert replacement is not None
        owner.mark_connected()
        assert owner.state == "connected"
        assert not reaper.done()
        close_task = asyncio.create_task(owner.close())
        await asyncio.sleep(0)
        assert not close_task.done()
        assert retired.has_resources()
        release.set()
        await asyncio.wait_for(close_task, timeout=1)
        assert not owner._retired_runtimes
        assert not retired.has_resources()
        assert not replacement.has_resources()
        assert owner.state == "off"
    finally:
        release.set()
        if close_task is not None:
            await close_task
        await owner.close()
