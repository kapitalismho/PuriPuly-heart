from __future__ import annotations

from dataclasses import dataclass, field

import pytest


@dataclass
class DummyPage:
    opened: list[object] = field(default_factory=list)
    closed: list[object] = field(default_factory=list)

    def show_dialog(self, dialog: object) -> None:
        self.opened.append(dialog)

    def pop_dialog(self) -> object | None:
        if not self.opened:
            return None
        dialog = self.opened[-1]
        self.closed.append(dialog)
        return dialog


class DialogTrackingPage:
    def __init__(self) -> None:
        self.dialog: object | None = None
        self.opened: list[object] = []
        self.closed: list[object] = []

    def show_dialog(self, dialog: object) -> None:
        self.dialog = dialog
        self.opened.append(dialog)

    def pop_dialog(self) -> object | None:
        dialog = self.dialog
        if dialog is None:
            return None
        self.closed.append(dialog)
        self.dialog = None
        return dialog


def attach_dummy_page(
    monkeypatch: pytest.MonkeyPatch,
    control: object,
    page: object | None = None,
) -> object:
    attached_page = object() if page is None else page
    control_type = type(control)
    isolated_control_type = type(
        f"_AttachedDummyPage{control_type.__name__}",
        (control_type,),
        {"__module__": control_type.__module__},
    )
    monkeypatch.setattr(control, "__class__", isolated_control_type)
    monkeypatch.setattr(
        type(control),
        "page",
        property(lambda _self: attached_page),
    )
    return attached_page
