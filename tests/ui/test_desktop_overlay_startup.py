from __future__ import annotations

import pytest

from puripuly_heart.ui.desktop_overlay_startup import (
    DesktopOverlayStartupCoordinator,
    DesktopOverlayStartupPhase,
)


def test_startup_coordinator_accepts_only_the_required_forward_sequence() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    coordinator = DesktopOverlayStartupCoordinator(
        7,
        trace_sink=lambda event, fields: events.append((event, fields)),
    )

    for phase in (
        DesktopOverlayStartupPhase.PAGE_CONFIGURED,
        DesktopOverlayStartupPhase.NATIVE_READY,
        DesktopOverlayStartupPhase.BOUNDS_CONFIRMED,
        DesktopOverlayStartupPhase.VISIBLE_CONFIRMED,
        DesktopOverlayStartupPhase.READY,
    ):
        coordinator.advance(phase)

    assert coordinator.ready is True
    assert [event for event, _fields in events] == [phase.value for phase in coordinator._SEQUENCE]
    assert all(fields["generation"] == 7 for _event, fields in events)
    assert all(isinstance(fields["monotonic_ms"], float) for _event, fields in events)


@pytest.mark.parametrize(
    "phase",
    [
        DesktopOverlayStartupPhase.LAUNCHED,
        DesktopOverlayStartupPhase.NATIVE_READY,
        DesktopOverlayStartupPhase.READY,
    ],
)
def test_startup_coordinator_rejects_illegal_transition(
    phase: DesktopOverlayStartupPhase,
) -> None:
    coordinator = DesktopOverlayStartupCoordinator(1)

    with pytest.raises(RuntimeError, match="illegal desktop overlay startup transition"):
        coordinator.advance(phase)


def test_startup_coordinator_rejects_stale_generation_and_retired_work() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    coordinator = DesktopOverlayStartupCoordinator(
        3,
        trace_sink=lambda event, fields: events.append((event, fields)),
    )

    assert coordinator.accepts(3) is True
    assert coordinator.accepts(2) is False
    coordinator.reject("bounds_callback", 2)
    coordinator.retire()

    assert coordinator.accepts(3) is False
    with pytest.raises(RuntimeError, match="retired"):
        coordinator.advance(DesktopOverlayStartupPhase.PAGE_CONFIGURED)
    assert events[-1][1]["accepted"] is False
    assert events[-1][1]["event_generation"] == 2


def test_startup_coordinator_attaches_confirmation_evidence_to_phase_trace() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    coordinator = DesktopOverlayStartupCoordinator(
        4,
        trace_sink=lambda event, fields: events.append((event, fields)),
    )
    coordinator.advance(DesktopOverlayStartupPhase.PAGE_CONFIGURED)
    coordinator.advance(DesktopOverlayStartupPhase.NATIVE_READY)
    coordinator.advance(
        DesktopOverlayStartupPhase.BOUNDS_CONFIRMED,
        canonical_bounds={"x": 10, "y": 20, "width": 800, "height": 240},
        observed_bounds=(10, 20, 800, 240),
    )

    event, fields = events[-1]
    assert event == "bounds_confirmed"
    assert fields["canonical_bounds"] == {"x": 10, "y": 20, "width": 800, "height": 240}
    assert fields["observed_bounds"] == (10, 20, 800, 240)
