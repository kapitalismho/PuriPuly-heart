from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = ROOT / "src" / "puripuly_heart" / "ui" / "controller.py"
ADAPTER_PATH = ROOT / "src" / "puripuly_heart" / "app" / "adapters" / "self_capture_source.py"
OWNER_PATH = ROOT / "src" / "puripuly_heart" / "core" / "runtime" / "self_capture.py"


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


def test_controller_composes_self_capture_source_adapter_without_source_algorithm() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")
    composition = _controller_method_source("_get_self_capture_owner")

    assert "source_factory=create_self_capture_source_adapter(" in composition
    assert "_create_self_capture_source" not in source
    assert "All microphone attempts failed" not in source
    assert "name_fallback" not in source
    assert "system_default" not in source


def test_self_capture_source_adapter_has_no_ui_settings_or_lifecycle_ownership() -> None:
    source = ADAPTER_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in source
    assert "GuiController" not in source
    assert "AppSettings" not in source
    assert "asyncio" not in source
    assert "async def close(" not in source
    assert "SelfCaptureSessionConfig" in source
    assert "name_fallback" in source
    assert "system_default" in source
    assert "will_retry_mono=True" in source


def test_self_capture_session_owner_remains_source_lifecycle_owner() -> None:
    source = OWNER_PATH.read_text(encoding="utf-8")

    assert "class SelfCaptureSessionOwner" in source
    assert '"_source"' in source
    assert '"_retired_sources"' in source
    assert "await self._close_source" in source
    assert "source_factory: SelfCaptureSourceFactory" in source
