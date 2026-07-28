from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = ROOT / "src" / "puripuly_heart" / "ui" / "controller.py"
OWNER_PATH = (
    ROOT / "src" / "puripuly_heart" / "app" / "services" / "provider_secret_change_serialization.py"
)


def _controller_method_source(method_name: str) -> str:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    controller = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GuiController"
    )
    method = next(
        node
        for node in controller.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == method_name
    )
    return ast.get_source_segment(source, method) or ""


def test_controller_provider_secret_change_delegates_serialization_owner() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")
    method = _controller_method_source("persist_provider_secret_change")

    assert "_provider_secret_change_lock" not in source
    assert "_get_provider_secret_change_serialization_owner().run(" in method
    assert "asyncio.Lock()" not in method


def test_provider_secret_change_serialization_owner_has_no_ui_or_controller_dependency() -> None:
    source = OWNER_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in source
    assert "GuiController" not in source
    assert "asyncio.Lock" in source
    assert "await operation()" in source
