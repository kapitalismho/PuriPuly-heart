from __future__ import annotations

from types import SimpleNamespace

import flet as ft
import pytest

from puripuly_heart.ui.components.settings.osc_connection_modal import OscConnectionModal


class DummyPage:
    def __init__(self) -> None:
        self.dialogs: list[object] = []

    def show_dialog(self, dialog: object) -> None:
        self.dialogs.append(dialog)

    def pop_dialog(self) -> object | None:
        return self.dialogs.pop() if self.dialogs else None


def test_automatic_mode_displays_effective_ports_and_disables_fields() -> None:
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
    assert modal.send_port_field.disabled is True
    assert modal.receive_port_field.disabled is True
    assert selected == []


def test_manual_mode_accepts_ports_and_off_preserves_saved_ports() -> None:
    page = DummyPage()
    selected: list[tuple[str, int, int]] = []
    modal = OscConnectionModal(page, lambda *values: selected.append(values))
    modal.open("manual", 9010, 9011)

    assert modal.send_port_field is not None
    assert modal.receive_port_field is not None
    assert modal.send_port_field.disabled is False
    assert modal.receive_port_field.disabled is False
    modal.send_port_field.value = "9012"
    modal.receive_port_field.value = "9013"
    modal._apply(None)

    assert selected == [("manual", 9012, 9013)]
    assert page.dialogs == []

    modal.open("off", 9012, 9013)
    assert modal.send_port_field is not None
    assert modal.receive_port_field is not None
    assert modal.send_port_field.value == "9012"
    assert modal.receive_port_field.value == "9013"
    assert modal.send_port_field.disabled is True
    assert modal.receive_port_field.disabled is True


def test_manual_mode_rejects_invalid_ports_without_closing() -> None:
    page = DummyPage()
    selected: list[tuple[str, int, int]] = []
    modal = OscConnectionModal(page, lambda *values: selected.append(values))
    modal.open("manual", 9000, 9001)

    assert modal.send_port_field is not None
    assert modal.receive_port_field is not None
    modal.send_port_field.value = "0"
    modal.receive_port_field.value = "9001"
    modal._apply(None)

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

    assert modal.mode_group is not None
    modal.mode_group.value = "manual"
    modal._on_mode_change(None)

    assert modal.send_port_field is not None
    assert modal.receive_port_field is not None
    assert modal.send_port_field.disabled is False
    assert modal.receive_port_field.disabled is False
    assert updates == ["dialog"]


def test_modal_has_two_columns_with_ports_only_on_the_left() -> None:
    page = DummyPage()
    modal = OscConnectionModal(page, lambda *_values: None)
    modal.open("automatic", 9000, 9001)

    assert modal.dialog is not None
    content = modal.dialog.content.content
    assert isinstance(content, ft.Column)
    columns = content.controls[1]
    assert isinstance(columns, ft.Row)
    assert len(columns.controls) == 2
    left, right = columns.controls
    assert isinstance(left, ft.Column)
    assert isinstance(right, ft.Column)
    assert left.controls[0].value == "Ports"
    assert right.controls[0].value == "Connection"
    assert {field.label for field in (modal.send_port_field, modal.receive_port_field)} == {
        "OSC Send Port",
        "OSC Receive Port",
    }
    assert not any(isinstance(control, ft.TextField) for control in right.controls)


@pytest.mark.parametrize("locale", ["en", "ko", "ja", "ru", "zh-CN"])
def test_osc_connection_localization_keys_exist(locale: str) -> None:
    from puripuly_heart.ui.i18n import _load_bundle

    bundle = _load_bundle(locale)
    for key in (
        "settings.vrchat_osc",
        "settings.osc.connection.title",
        "settings.osc.ports",
        "settings.osc.connection",
        "settings.osc.mode.automatic",
        "settings.osc.mode.manual",
        "settings.osc.mode.off",
        "settings.osc.send_port",
        "settings.osc.receive_port",
    ):
        assert bundle[key]
