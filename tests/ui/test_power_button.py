from __future__ import annotations

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


def test_power_button_set_state_transitions_and_renders_icon_and_label_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clicked = {"count": 0}
    btn = PowerButton(
        label="STT", icon="MIC", on_click=lambda: clicked.__setitem__("count", clicked["count"] + 1)
    )

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
