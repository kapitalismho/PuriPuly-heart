from __future__ import annotations

from types import SimpleNamespace

import pytest

ft = pytest.importorskip("flet")

from puripuly_heart.ui.components.power_button import PowerButton
from puripuly_heart.ui.theme import (
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SURFACE,
    COLOR_TRANS_TONAL,
    COLOR_WARNING,
)

HOVER_ENTER = SimpleNamespace(data="true")
HOVER_LEAVE = SimpleNamespace(data="false")


def _button(clicked: dict | None = None) -> PowerButton:
    counter = clicked if clicked is not None else {"count": 0}
    return PowerButton(
        label="STT", icon="MIC", on_click=lambda: counter.__setitem__("count", counter["count"] + 1)
    )


def test_power_button_set_state_transitions_and_renders_icon_and_label_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clicked = {"count": 0}
    btn = _button(clicked)

    column = btn.content.content
    assert column.controls == [btn._icon_slot, btn._label_control]
    assert not hasattr(btn, "_status_control")
    assert not hasattr(btn, "_helper_control")

    btn.set_state(False, needs_key=False, status_text="Off")
    assert btn.bgcolor == COLOR_TRANS_TONAL
    assert btn._icon_control.color == COLOR_SECONDARY
    assert btn._label_control.color == COLOR_SECONDARY
    assert btn._icon_control.visible is True
    assert btn._progress_control.visible is False
    assert btn.border is None

    btn.set_state(False, is_starting=True, status_text="Starting")
    assert btn.bgcolor == COLOR_SURFACE
    assert btn.border is None
    assert btn._icon_control.visible is False
    assert btn._progress_control.visible is True
    assert btn._progress_control.color == COLOR_PRIMARY
    assert btn._label_control.color == COLOR_PRIMARY

    btn.set_state(True, needs_key=False, status_text="On", helper_text="Ready now")
    assert btn.bgcolor == COLOR_PRIMARY
    assert btn._icon_control.color == ft.Colors.WHITE
    assert btn._label_control.color == ft.Colors.WHITE
    assert btn._icon_control.visible is True
    assert btn._progress_control.visible is False

    btn.set_state(False, needs_key=True, status_text="Needs key", helper_text="Enter API key")
    assert btn.bgcolor == COLOR_WARNING
    assert btn._icon_control.color == btn._label_control.color == ft.Colors.WHITE

    btn.set_label("NEW")
    assert btn._label_control.value == "NEW"


def test_power_button_hover_off_state_scales_only() -> None:
    btn = _button()
    btn.set_state(False)
    btn._on_hover(HOVER_ENTER)
    assert btn.scale == 1.02
    assert btn.bgcolor == COLOR_TRANS_TONAL
    assert btn._icon_control.color == COLOR_SECONDARY
    assert btn._label_control.color == COLOR_SECONDARY
    btn._on_hover(HOVER_LEAVE)
    assert btn.scale == 1.0


def test_power_button_hover_on_state_scales_only() -> None:
    btn = _button()
    btn.set_state(True)
    btn._on_hover(HOVER_ENTER)
    assert btn.scale == 1.02
    assert btn.bgcolor == COLOR_PRIMARY
    assert btn._icon_control.color == ft.Colors.WHITE
    assert btn._label_control.color == ft.Colors.WHITE
    btn._on_hover(HOVER_LEAVE)
    assert btn.scale == 1.0


def test_power_button_hover_warning_state_scales_only() -> None:
    btn = _button()
    btn.set_state(False, needs_key=True)
    btn._on_hover(HOVER_ENTER)
    assert btn.scale == 1.02
    assert btn.bgcolor == COLOR_WARNING
    assert btn._icon_control.color == ft.Colors.WHITE
    assert btn._label_control.color == ft.Colors.WHITE
    btn._on_hover(HOVER_LEAVE)
    assert btn.scale == 1.0


def test_power_button_hover_starting_state_keeps_surface() -> None:
    btn = _button()
    btn.set_state(False, is_starting=True)
    btn._on_hover(HOVER_ENTER)
    assert btn.scale == 1.0
    assert btn.bgcolor == COLOR_SURFACE
    assert btn._icon_control.color == COLOR_SECONDARY
    assert btn._label_control.color == COLOR_PRIMARY


def test_power_button_state_change_keeps_scale_while_hovering() -> None:
    btn = _button()
    btn.set_state(False)
    btn._on_hover(HOVER_ENTER)
    assert btn.scale == 1.02
    btn.set_state(True)
    assert btn.bgcolor == COLOR_PRIMARY
    assert btn.scale == 1.02
    btn._on_hover(HOVER_LEAVE)
    assert btn.scale == 1.0
