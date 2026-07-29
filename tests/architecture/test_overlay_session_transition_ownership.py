from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = ROOT / "src" / "puripuly_heart" / "ui" / "controller.py"
OWNER_PATH = ROOT / "src" / "puripuly_heart" / "app" / "services" / "overlay_session_transition.py"
APPLICATION_OWNER_PATH = (
    ROOT / "src" / "puripuly_heart" / "app" / "services" / "overlay_application.py"
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


def test_controller_delegates_overlay_start_and_shutdown_transitions() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")
    application = APPLICATION_OWNER_PATH.read_text(encoding="utf-8")
    start = _controller_method_source("_begin_overlay_start")
    shutdown = _controller_method_source("_shutdown_overlay_runtime")

    assert "_get_overlay_application_owner().begin_start()" in start
    assert "_get_overlay_application_owner().shutdown(" in shutdown
    assert "self._transition_owner.begin_start(" in application
    assert "self._transition_owner.shutdown(" in application
    assert "_overlay_lock" not in source
    assert "async with self._overlay_lock" not in source


def test_transition_owner_preserves_overlay_runtime_handle_resource_ownership() -> None:
    source = OWNER_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in source
    assert "GuiController" not in source
    assert "OverlayBridge" not in source
    assert "OverlayProcessManager" not in source
    assert "OverlayPresenter" not in source
    assert "runtime.create_start_task(" in source
    assert "execution.teardown()" in source
    assert "asyncio.Lock" in source
