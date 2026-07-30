from __future__ import annotations

import pytest

from puripuly_heart.ui.dashboard import capture as dashboard_capture_module
from puripuly_heart.ui.dashboard.capture import (
    CAPTURE_PRESENTATION_IDLE,
    DashboardCaptureControls,
    capture_presentation_from_contract,
)
from puripuly_heart.ui.dashboard.contract import DashboardSurfaceSlots
from puripuly_heart.ui.overlay_peer_contract import (
    OverlayPeerConsumerContract,
    OverlayPeerToggleContract,
)


class FakePowerButton:
    def __init__(self, *, label, icon, on_click, icon_size, label_size) -> None:
        self.label = label
        self.icon = icon
        self.on_click = on_click
        self.icon_size = icon_size
        self.label_size = label_size
        self.states: list[tuple[bool, bool, bool]] = []

    def set_state(self, enabled, *, needs_key=False, is_starting=False) -> None:
        self.states.append((bool(enabled), bool(needs_key), bool(is_starting)))

    def set_label(self, label) -> None:
        self.label = label


@pytest.fixture()
def controls(monkeypatch: pytest.MonkeyPatch) -> DashboardCaptureControls:
    monkeypatch.setattr(dashboard_capture_module, "PowerButton", FakePowerButton)
    monkeypatch.setattr(dashboard_capture_module, "t", lambda key, **_kwargs: f"i18n:{key}")
    return DashboardCaptureControls(
        on_self_capture_click=lambda: None,
        on_peer_capture_click=lambda: None,
        on_overlay_click=lambda: None,
    )


def test_capture_controls_expose_the_dashboard_slot_contract(
    controls: DashboardCaptureControls,
) -> None:
    slots = DashboardSurfaceSlots.from_capture_provider(
        controls,
        translation=object(),
        display=object(),
        language=object(),
    )

    assert slots.self_capture is controls.self_capture_control()
    assert slots.peer_capture is controls.peer_capture_control()
    assert slots.overlay is controls.overlay_control()


def test_capture_controls_render_every_channel_state(controls: DashboardCaptureControls) -> None:
    controls.apply_self_capture_state(enabled=False, starting=True, warning=False)
    controls.apply_self_capture_state(enabled=True, starting=False, warning=False)
    controls.apply_self_capture_state(enabled=False, starting=False, warning=True)
    controls.apply_peer_capture_state(enabled=False, starting=True)
    controls.apply_peer_capture_state(enabled=True)
    controls.apply_peer_capture_state(enabled=False, warning=True)
    controls.apply_overlay_state(enabled=True)
    controls.apply_overlay_state(enabled=False, warning=True)

    assert controls.self_capture_control().states == [
        (False, False, True),
        (True, False, False),
        (False, True, False),
    ]
    assert controls.peer_capture_control().states == [
        (False, False, True),
        (True, False, False),
        (False, True, False),
    ]
    assert controls.overlay_control().states == [(True, False, False), (False, True, False)]


def test_capture_controls_localize_labels_at_the_ui_boundary(
    controls: DashboardCaptureControls,
) -> None:
    assert controls.self_capture_control().label == "i18n:dashboard.stt_label"
    assert controls.peer_capture_control().label == "i18n:dashboard.peer_label"
    assert controls.overlay_control().label == "i18n:dashboard.overlay_label"

    controls.self_capture_control().label = "stale"
    controls.apply_locale()

    assert controls.self_capture_control().label == "i18n:dashboard.stt_label"


def test_capture_controls_forward_clicks_to_their_own_intents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dashboard_capture_module, "PowerButton", FakePowerButton)
    clicks: list[str] = []
    controls = DashboardCaptureControls(
        on_self_capture_click=lambda: clicks.append("self"),
        on_peer_capture_click=lambda: clicks.append("peer"),
        on_overlay_click=lambda: clicks.append("overlay"),
    )

    controls.self_capture_control().on_click()
    controls.peer_capture_control().on_click()
    controls.overlay_control().on_click()

    assert clicks == ["self", "peer", "overlay"]


def _contract(
    *,
    peer_state: str = "off",
    peer_reason: str | None = None,
    peer_helper: str = "",
    overlay_state: str = "off",
) -> OverlayPeerConsumerContract:
    return OverlayPeerConsumerContract(
        peer=OverlayPeerToggleContract(
            intent_enabled=peer_state in {"on", "starting"},
            effective_enabled=peer_state == "on",
            action_enabled=True,
            state=peer_state,
            status_text="",
            helper_text=peer_helper,
            warning_reason=peer_reason,
        ),
        overlay=OverlayPeerToggleContract(
            intent_enabled=overlay_state in {"on", "starting"},
            effective_enabled=overlay_state == "on",
            action_enabled=True,
            state=overlay_state,
            status_text="",
        ),
    )


def test_capture_presentation_is_idle_without_a_contract() -> None:
    assert capture_presentation_from_contract(None) is CAPTURE_PRESENTATION_IDLE


@pytest.mark.parametrize(
    ("peer_state", "expected"),
    [
        ("off", (False, False, False)),
        ("starting", (False, True, False)),
        ("on", (True, False, False)),
        ("warning", (False, False, True)),
    ],
)
def test_capture_presentation_projects_every_peer_state(peer_state, expected) -> None:
    presentation = capture_presentation_from_contract(_contract(peer_state=peer_state))

    assert (
        presentation.peer.enabled,
        presentation.peer.starting,
        presentation.peer.warning,
    ) == expected


def test_capture_presentation_reports_process_capture_warning_only_with_helper_text() -> None:
    active = capture_presentation_from_contract(
        _contract(
            peer_state="warning",
            peer_reason="process_capture_unavailable",
            peer_helper="pick another window",
        )
    )
    assert active.process_capture_warning_active is True
    assert active.process_capture_warning_reason == "process_capture_unavailable"
    assert active.process_capture_warning_text == "pick another window"

    without_text = capture_presentation_from_contract(
        _contract(peer_state="warning", peer_reason="process_capture_unavailable")
    )
    assert without_text.process_capture_warning_active is False
    assert without_text.process_capture_warning_text == ""

    other_reason = capture_presentation_from_contract(
        _contract(peer_state="warning", peer_reason="missing_api_key", peer_helper="add a key")
    )
    assert other_reason.process_capture_warning_active is False


def test_capture_controls_apply_projected_presentation(
    controls: DashboardCaptureControls,
) -> None:
    controls.apply_presentation(
        capture_presentation_from_contract(_contract(peer_state="on", overlay_state="warning"))
    )

    assert controls.peer_capture_control().states == [(True, False, False)]
    assert controls.overlay_control().states == [(False, True, False)]
