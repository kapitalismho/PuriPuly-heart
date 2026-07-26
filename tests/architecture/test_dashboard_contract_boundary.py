from __future__ import annotations

import ast
import pathlib

import puripuly_heart
from puripuly_heart.ui.dashboard import contract as dashboard_contract
from puripuly_heart.ui.dashboard import renderer as dashboard_renderer

SOURCE_ROOT = pathlib.Path(puripuly_heart.__file__).resolve().parent
FORBIDDEN_IMPORT_PREFIXES = (
    "puripuly_heart.core",
    "puripuly_heart.runtime",
    "puripuly_heart.ui.controller",
    "puripuly_heart.app.services",
)


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def test_dashboard_contract_and_renderer_stay_above_backend_owners() -> None:
    for module in (dashboard_contract, dashboard_renderer):
        path = pathlib.Path(module.__file__)
        for imported in _imported_modules(path):
            assert not imported.startswith(
                FORBIDDEN_IMPORT_PREFIXES
            ), f"{path.name} must not import backend implementation: {imported}"


def test_dashboard_view_implements_the_explicit_contract() -> None:
    from puripuly_heart.ui.views.dashboard import DashboardView

    for method in (
        "bind_dashboard_intents",
        "self_capture_control",
        "peer_capture_control",
        "overlay_control",
    ):
        assert callable(getattr(DashboardView, method)), method


def test_translator_app_wires_dashboard_intents_through_one_path() -> None:
    app_source = (SOURCE_ROOT / "ui" / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(app_source)
    direct_assignments = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr.startswith("on_")
                and isinstance(target.value, ast.Attribute)
                and target.value.attr == "view_dashboard"
            ):
                direct_assignments.append(target.attr)
    assert direct_assignments == []
    assert "bind_dashboard_intents(" in app_source
