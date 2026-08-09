from __future__ import annotations

from types import SimpleNamespace

import flet as ft
import pytest

from puripuly_heart.ui.components.settings.osc_connection_modal import (
    _PORT_BOTTOM_SPACER_HEIGHT,
    _PORT_FIELD_HEIGHT,
    _PORT_LABEL_HEIGHT,
    OscConnectionModal,
)
from puripuly_heart.ui.theme import COLOR_BACKGROUND, COLOR_NEUTRAL_DARK, COLOR_PRIMARY


class DummyPage:
    def __init__(self) -> None:
        self.dialogs: list[object] = []

    def show_dialog(self, dialog: object) -> None:
        self.dialogs.append(dialog)

    def pop_dialog(self) -> object | None:
        return self.dialogs.pop() if self.dialogs else None


def test_automatic_mode_displays_effective_ports_and_read_only_fields() -> None:
    page = DummyPage()
    selected: list[tuple[str, int, int]] = []
    modal = OscConnectionModal(
        page,
        lambda mode, send_port, receive_port: selected.append((mode, send_port, receive_port)),
        effective_ports_provider=lambda: (9100, 49152),
    )

    modal.open("automatic", 9000, 9001)

    assert modal.send_port_field is not None
    assert modal.receive_port_field is not None
    assert modal.send_port_field.value == "9100"
    assert modal.receive_port_field.value == "49152"
    assert modal.send_port_field.read_only is True
    assert modal.receive_port_field.read_only is True
    assert selected == []


def test_manual_mode_accepts_ports_and_off_mode_preserves_saved_ports() -> None:
    page = DummyPage()
    selected: list[tuple[str, int, int]] = []
    modal = OscConnectionModal(page, lambda *values: selected.append(values))
    modal.open("manual", 9010, 9011)

    assert modal.send_port_field is not None
    assert modal.receive_port_field is not None
    assert modal.send_port_field.read_only is False
    assert modal.receive_port_field.read_only is False
    assert modal.send_port_field.on_change is not None
    assert modal.receive_port_field.on_change is not None
    assert modal.send_port_field.color == COLOR_PRIMARY
    assert modal.receive_port_field.color == COLOR_PRIMARY
    modal.send_port_field.value = "9012"
    modal.receive_port_field.value = "9013"
    modal._commit()

    assert selected == [("manual", 9012, 9013)]
    assert page.dialogs

    modal._select_mode("off")
    assert selected == [("manual", 9012, 9013), ("off", 9012, 9013)]
    assert page.dialogs == []
    assert modal.send_port_field.value == "9012"
    assert modal.receive_port_field.value == "9013"
    assert modal.send_port_field.read_only is True
    assert modal.receive_port_field.read_only is True
    assert modal.send_port_field.color == COLOR_NEUTRAL_DARK
    assert modal.receive_port_field.color == COLOR_NEUTRAL_DARK


def test_manual_mode_rejects_invalid_ports_without_closing() -> None:
    page = DummyPage()
    selected: list[tuple[str, int, int]] = []
    modal = OscConnectionModal(page, lambda *values: selected.append(values))
    modal.open("manual", 9000, 9001)

    assert modal.send_port_field is not None
    assert modal.receive_port_field is not None
    modal.send_port_field.value = "0"
    modal.receive_port_field.value = "9001"
    modal._commit()

    assert selected == []
    assert page.dialogs
    assert modal.send_port_field.error_text
    assert modal.receive_port_field.error_text


def test_mode_change_repaints_a_mounted_dialog() -> None:
    page = DummyPage()
    modal = OscConnectionModal(
        page,
        lambda *_values: None,
        effective_ports_provider=lambda: (9100, 49152),
    )
    modal.open("automatic", 9000, 9001)
    updates: list[str] = []
    modal._dialog = SimpleNamespace(
        page=object(),
        update=lambda: updates.append("dialog"),
    )

    assert modal.selected_mode == "automatic"
    assert modal.mode_cards["automatic"].bgcolor == COLOR_PRIMARY
    modal._select_mode("manual")

    assert modal.send_port_field is not None
    assert modal.receive_port_field is not None
    assert modal.selected_mode == "manual"
    assert modal.send_port_field.read_only is False
    assert modal.receive_port_field.read_only is False
    assert modal.mode_cards["manual"].bgcolor == COLOR_PRIMARY
    assert modal.mode_cards["automatic"].bgcolor == COLOR_BACKGROUND
    assert updates == ["dialog"]


def test_modal_uses_two_columns_with_ports_left_and_mode_cards_right() -> None:
    page = DummyPage()
    modal = OscConnectionModal(page, lambda *_values: None)
    modal.open("automatic", 9000, 9001)

    assert modal.dialog is not None
    content = modal.dialog.content.content
    assert isinstance(content, ft.Column)
    assert content.controls[0].value == "OSC Connection"

    columns = content.controls[2]
    assert isinstance(columns, ft.Row)
    assert len(columns.controls) == 2
    left, right = columns.controls
    assert isinstance(left, ft.Column)
    assert isinstance(right, ft.Column)
    assert left.controls[0].value == "Ports"
    assert left.controls[1].content.value == "OSC Send"
    assert left.controls[1].height == _PORT_LABEL_HEIGHT
    assert left.controls[2] is modal.send_port_field
    assert isinstance(left.controls[3], ft.Container)
    assert left.controls[3].height == _PORT_BOTTOM_SPACER_HEIGHT
    assert left.controls[4].content.value == "OSC Receive"
    assert left.controls[4].height == _PORT_LABEL_HEIGHT
    assert left.controls[5] is modal.receive_port_field
    assert modal.send_port_field.height == _PORT_FIELD_HEIGHT
    assert modal.receive_port_field.height == _PORT_FIELD_HEIGHT
    assert right.controls[0].value == "Connection Method"
    mode_cards = right.controls[1:]
    assert len(mode_cards) == 3
    assert not any(isinstance(control, ft.TextField) for control in mode_cards)
    assert {card.bgcolor for card in mode_cards} == {
        COLOR_PRIMARY,
        COLOR_BACKGROUND,
    }


def test_commit_skips_redundant_duplicates() -> None:
    page = DummyPage()
    selected: list[tuple[str, int, int]] = []
    modal = OscConnectionModal(page, lambda *values: selected.append(values))
    modal.open("manual", 9010, 9011)

    modal._commit()
    modal._commit()

    assert selected == [("manual", 9010, 9011)]


def test_dismiss_commits_latest_manual_port_values() -> None:
    page = DummyPage()
    selected: list[tuple[str, int, int]] = []
    modal = OscConnectionModal(page, lambda *values: selected.append(values))
    modal.open("manual", 9000, 9001)

    assert modal.dialog is not None
    assert modal.send_port_field is not None
    assert modal.receive_port_field is not None
    modal.send_port_field.value = "9020"
    modal.receive_port_field.value = "9021"
    modal.dialog.on_dismiss(None)

    assert selected == [("manual", 9020, 9021)]


@pytest.mark.parametrize("locale", ["en", "ko", "ja", "ru", "zh-CN"])
def test_osc_connection_localization_keys_exist(locale: str) -> None:
    from puripuly_heart.ui.i18n import _load_bundle

    bundle = _load_bundle(locale)
    for key in (
        "settings.osc.connection.title",
        "settings.osc.ports",
        "settings.osc.mode",
        "settings.osc.mode.automatic",
        "settings.osc.mode.manual",
        "settings.osc.mode.off",
        "settings.osc.send_port",
        "settings.osc.receive_port",
    ):
        assert bundle[key]
