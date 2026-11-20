from __future__ import annotations

import ast
import inspect

from puripuly_heart.app.ports.ui_application import UiApplicationPort
from puripuly_heart.app.ports.ui_presentation import UiPresentationPort
from puripuly_heart.app.services.ui_application import (
    UiApplicationBoundary,
)
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter
from tests.helpers.ast_sources import imported_modules as _imports
from tests.helpers.paths import REPO_ROOT

APP_PATH = REPO_ROOT / "src" / "puripuly_heart" / "ui" / "app.py"
APPLICATION_RUNTIME_PATH = (
    REPO_ROOT / "src" / "puripuly_heart" / "composition" / "application_runtime.py"
)


def _contract_members(contract: type[object]) -> set[str]:
    return {
        name
        for name, member in contract.__dict__.items()
        if not name.startswith("_") and (inspect.isfunction(member) or isinstance(member, property))
    }


def _assert_contract_signatures(contract: type[object], implementation: type[object]) -> None:
    for name, contract_member in contract.__dict__.items():
        if name.startswith("_"):
            continue
        implementation_member = inspect.getattr_static(implementation, name)
        if isinstance(contract_member, property):
            assert isinstance(implementation_member, property)
            assert inspect.signature(contract_member.fget) == inspect.signature(
                implementation_member.fget
            )
        elif inspect.isfunction(contract_member):
            assert inspect.signature(contract_member) == inspect.signature(implementation_member)


def test_ui_application_contract_covers_every_translator_app_boundary_access() -> None:
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    accessed = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "self"
        and node.value.attr == "application"
    }

    contract = _contract_members(UiApplicationPort)
    implementation = _contract_members(UiApplicationBoundary)

    assert accessed <= contract
    assert contract <= implementation
    assert "__getattr__" not in UiApplicationBoundary.__dict__
    assert not hasattr(UiApplicationBoundary, "backend")


def test_ui_boundary_implementations_match_every_declared_contract_signature() -> None:
    _assert_contract_signatures(UiApplicationPort, UiApplicationBoundary)
    _assert_contract_signatures(UiPresentationPort, FletUiPresentationAdapter)


def test_translator_app_does_not_bypass_the_application_port() -> None:
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    direct_controller_accesses: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name in {"__init__", "application"}:
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Attribute)
                and isinstance(child.value, ast.Name)
                and child.value.id == "self"
                and child.attr == "controller"
                and isinstance(child.ctx, ast.Load)
            ):
                direct_controller_accesses.append((node.name, child.lineno))

    source = APP_PATH.read_text(encoding="utf-8")
    assert direct_controller_accesses == []
    assert "getattr(self.application" not in source


def test_translator_app_imports_only_the_approved_backend_boundary_surface() -> None:
    imports = _imports(APP_PATH)
    forbidden_prefixes = (
        "puripuly_heart.app.wiring",
        "puripuly_heart.config.settings",
        "puripuly_heart.core.orchestrator",
        "puripuly_heart.core.runtime",
        "puripuly_heart.core.runtime_logging",
        "puripuly_heart.core.storage",
        "puripuly_heart.providers",
    )

    assert not any(module.startswith(forbidden_prefixes) for module in imports)
    assert "puripuly_heart.app.ports.ui_application" in imports
    assert "puripuly_heart.app.services.ui_application" not in imports


def test_application_composition_presentation_access_is_explicit_and_adapter_is_closed() -> None:
    tree = ast.parse(APPLICATION_RUNTIME_PATH.read_text(encoding="utf-8"))
    accessed: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "presentation"
        ):
            accessed.add(node.attr)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "presentation"
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            accessed.add(node.args[1].value)

    assert accessed <= _contract_members(UiPresentationPort)
    assert _contract_members(UiPresentationPort) <= _contract_members(FletUiPresentationAdapter)
    assert "__getattr__" not in FletUiPresentationAdapter.__dict__
    assert not hasattr(FletUiPresentationAdapter, "app")


def test_application_runtime_has_no_flet_or_ui_module_dependencies() -> None:
    imports = _imports(APPLICATION_RUNTIME_PATH)

    assert "flet" not in imports
    assert not any(module.startswith("puripuly_heart.ui") for module in imports)
