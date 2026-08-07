from __future__ import annotations

from typing import cast

from puripuly_heart.app.services.overlay_application import (
    OverlayApplicationOwner,
    OverlayApplicationState,
)

from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.config.resolved import ResolvedOverlayConfig
from puripuly_heart.core.clock import FakeClock


async def _noop_async() -> None:
    return None


async def _noop_renderer(queue, overlay_instance_id: str) -> None:
    _ = queue, overlay_instance_id


class Recorder:
    def __init__(self) -> None:
        self.cancel_peer_activation_calls = 0
        self.sync_peer_effective_calls = 0
        self.states: list[tuple[str, str | None]] = []

    def cancel_peer_activation(self) -> None:
        self.cancel_peer_activation_calls += 1

    def sync_peer_effective(self) -> None:
        self.sync_peer_effective_calls += 1

    def state_sink(self, state: str, failure_reason: str | None) -> None:
        self.states.append((state, failure_reason))


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
        peer_snapshot_provider=lambda: cast(object, object()),
        disable_peer_intent=lambda: None,
        sync_peer_effective=recorder.sync_peer_effective,
        cancel_peer_activation=recorder.cancel_peer_activation,
        refresh_peer_dependencies=_noop_async,
        presentation_sink=lambda _state: None,
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
        log_basic=lambda _message, _level: None,
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
