from __future__ import annotations

import ast
import pathlib

import flet as ft

from puripuly_heart.ui.shell import contract as shell_contract
from puripuly_heart.ui.shell import renderer as shell_renderer
from puripuly_heart.ui.shell.contract import AppShellSlots
from puripuly_heart.ui.shell.renderer import (
    APP_SHELL_LAYOUT_SPACING,
    APP_SHELL_ROOT_PADDING,
    compose_app_shell,
)
from tests.helpers.paths import SOURCE_ROOT

FORBIDDEN_IMPORT_PREFIXES = (
    "puripuly_heart.core",
    "puripuly_heart.runtime",
    "puripuly_heart.ui.views",
    "puripuly_heart.app.services",
    "puripuly_heart.app.wiring",
    "puripuly_heart.config",
)

SHELL_CONTRACT_MODULES = (shell_contract, shell_renderer)


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def _slots(*, debug_panel: ft.Control | None = None) -> AppShellSlots:
    return AppShellSlots(
        title_bar=ft.Text("title-bar"),
        content=ft.Text("content"),
        bottom_nav=ft.Text("bottom-nav"),
        content_padding=24,
        debug_panel=debug_panel,
    )


def test_shell_contract_and_renderer_stay_above_backend_and_views() -> None:
    for module in SHELL_CONTRACT_MODULES:
        path = pathlib.Path(module.__file__)
        for imported in _imported_modules(path):
            assert not imported.startswith(
                FORBIDDEN_IMPORT_PREFIXES
            ), f"{path.name} must not import {imported}"


def test_compose_app_shell_keeps_the_accepted_ordering_and_geometry() -> None:
    slots = _slots()
    regions = compose_app_shell(slots)

    assert regions.layout.controls == [slots.title_bar, regions.content_area, slots.bottom_nav]
    assert regions.layout.spacing == APP_SHELL_LAYOUT_SPACING
    assert regions.layout.expand is True
    assert regions.content_area.content is slots.content
    assert regions.content_area.padding == slots.content_padding
    assert regions.content_area.expand is True
    assert regions.root.content is regions.layout
    assert regions.root.padding == APP_SHELL_ROOT_PADDING
    assert regions.root.expand is True
    assert regions.debug_stack is None


def test_compose_app_shell_mounts_the_debug_panel_only_when_supplied() -> None:
    panel = ft.Text("debug-panel")
    regions = compose_app_shell(_slots(debug_panel=panel))

    assert regions.debug_stack is not None
    assert regions.debug_stack.controls[1] is panel
    assert regions.debug_stack.fit == ft.StackFit.EXPAND
    assert regions.root.content is regions.debug_stack

    root_content = regions.debug_stack.controls[0]
    assert isinstance(root_content, ft.Container)
    assert root_content.content is regions.layout


def test_translator_app_shell_is_composed_by_the_renderer() -> None:
    source = (SOURCE_ROOT / "ui" / "app.py").read_text(encoding="utf-8")
    assert source.count("compose_app_shell(") == 1
    assert "AppShellSlots(" in source

    tree = ast.parse(source)
    build_layout = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_build_layout"
    )
    composed = {
        node.func.attr
        for node in ast.walk(build_layout)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "ft"
    }
    assert composed == set(), f"_build_layout must not compose Flet controls inline: {composed}"
