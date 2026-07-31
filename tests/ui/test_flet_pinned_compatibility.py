from __future__ import annotations

import re
import sys
import tomllib
from importlib.metadata import version
from types import SimpleNamespace

import pytest

pytest.importorskip("flet")

import flet as ft

from puripuly_heart.ui.components.settings import api_key_field as api_key_field_module
from puripuly_heart.ui.views import settings as settings_view
from tests.helpers.flet_page import DummyPage, attach_dummy_page
from tests.helpers.paths import REPO_ROOT as ROOT

FLET_VERSION = "0.86.1"


def test_flet_runtime_and_lock_use_one_exact_0861_protocol() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    build_dependencies = pyproject["project"]["optional-dependencies"]["build"]
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked_versions = {
        package["name"]: package["version"]
        for package in lock["package"]
        if package["name"] in {"flet", "flet-desktop", "flet-cli"}
    }

    assert f"flet=={FLET_VERSION}" in dependencies
    assert f"flet-desktop=={FLET_VERSION}" in dependencies
    assert f"flet-cli=={FLET_VERSION}" in build_dependencies
    assert locked_versions == {
        "flet": FLET_VERSION,
        "flet-cli": FLET_VERSION,
        "flet-desktop": FLET_VERSION,
    }
    assert version("flet") == FLET_VERSION
    assert version("flet-desktop") == FLET_VERSION


def test_gate_a_windows_runtime_uses_python_312() -> None:
    if sys.platform != "win32":
        pytest.skip("Gate A runtime version is verified on Windows")
    assert sys.version_info[:2] == (3, 12)


def test_ui_uses_flet_0861_dialog_api() -> None:
    assert hasattr(ft.Page, "show_dialog")
    assert hasattr(ft.Page, "pop_dialog")
    assert not hasattr(ft.Page, "open")
    assert not hasattr(ft.Page, "close")

    removed_api = re.compile(
        r"(?:\bpage|self\.page|self\._page)\.(?:open|close)\(|"
        r"getattr\((?:self\.)?_?page,\s*[\"'](?:open|close)[\"']"
    )
    violations = []
    for path in sorted((ROOT / "src" / "puripuly_heart" / "ui").rglob("*.py")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if removed_api.search(line):
                violations.append(f"{path.relative_to(ROOT)}:{line_number}:{line.strip()}")

    assert violations == []


def test_api_key_field_uses_flet_086_icon_api(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeIcon:
        def __init__(self, *, icon, color, size, tooltip):
            self.icon = icon
            self.color = color
            self.size = size
            self.tooltip = tooltip
            self.page = None

        def update(self) -> None:
            return None

    class FakeIconButton:
        def __init__(self, **kwargs):
            self.icon = kwargs.get("icon")

    class FakeTextField:
        def __init__(self, **kwargs):
            self.value = ""
            self.password = kwargs.get("password", False)
            self.page = None

        def update(self) -> None:
            return None

    class FakeRow:
        def __init__(self, *, controls, vertical_alignment):
            self.controls = controls
            self.vertical_alignment = vertical_alignment

    monkeypatch.setattr(api_key_field_module.ft, "Icon", FakeIcon)
    monkeypatch.setattr(api_key_field_module.ft, "IconButton", FakeIconButton)
    monkeypatch.setattr(api_key_field_module.ft, "TextField", FakeTextField)
    monkeypatch.setattr(api_key_field_module.ft, "Row", FakeRow)

    field = api_key_field_module.ApiKeyField(
        "settings.deepgram_api_key",
        "deepgram_api_key",
        "deepgram",
    )
    field._set_status("success")

    assert field._status_icon.icon == api_key_field_module.icons.CHECK_CIRCLE_ROUNDED


def test_make_text_button_uses_flet_086_content_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class FakeTextButton:
        def __init__(self, *, content, **kwargs):
            seen["content"] = content
            seen["kwargs"] = kwargs
            self.content = content

    monkeypatch.setattr(settings_view.ft, "TextButton", FakeTextButton)

    button = settings_view._make_text_button("Gemma 4", style="style")

    assert seen["content"] == "Gemma 4"
    assert seen["kwargs"] == {"style": "style"}
    assert button.content == "Gemma 4"


def test_set_text_button_label_uses_flet_086_content_property() -> None:
    class FakeButton:
        __slots__ = ("content",)

        def __init__(self) -> None:
            self.content = ""

    button = FakeButton()

    settings_view._set_text_button_label(button, "Managed")

    assert button.content == "Managed"


def test_make_overlay_anchor_dropdown_uses_flet_086_on_select(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class FakeOption:
        def __init__(self, *, key, text):
            self.key = key
            self.text = text

    class FakeDropdown:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(settings_view.ft.dropdown, "Option", FakeOption)
    monkeypatch.setattr(settings_view.ft, "Dropdown", FakeDropdown)

    on_change = SimpleNamespace()
    settings_view._make_overlay_anchor_dropdown("center", on_change)

    assert seen["value"] == "center"
    assert seen["on_select"] is on_change
    assert "on_change" not in seen
    assert len(seen["options"]) == len(settings_view.OVERLAY_CALIBRATION_ANCHORS)


class _RaisingPageControl:
    @property
    def page(self):
        raise RuntimeError("not attached")


def test_attach_dummy_page_replaces_unreadable_page_property(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _RaisingPageControl()

    page = attach_dummy_page(monkeypatch, control)

    assert control.page is page
    assert bool(control.page) is True


def test_attach_dummy_page_only_changes_target_control_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _RaisingPageControl()
    other_control = _RaisingPageControl()

    page = attach_dummy_page(monkeypatch, control)

    assert control.page is page
    with pytest.raises(RuntimeError, match="not attached"):
        _ = other_control.page


def test_attach_dummy_page_uses_explicit_dummy_page(monkeypatch: pytest.MonkeyPatch) -> None:
    control = _RaisingPageControl()
    page = DummyPage()

    returned = attach_dummy_page(monkeypatch, control, page)
    returned.show_dialog("dialog")
    returned.pop_dialog()

    assert returned is page
    assert control.page is page
    assert page.opened == ["dialog"]
    assert page.closed == ["dialog"]
