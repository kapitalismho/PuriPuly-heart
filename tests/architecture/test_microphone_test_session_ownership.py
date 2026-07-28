from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = ROOT / "src" / "puripuly_heart" / "ui" / "controller.py"
OWNER_PATH = ROOT / "src" / "puripuly_heart" / "app" / "services" / "microphone_test.py"


def _method_source(path: Path, class_name: str, method_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    owner = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    )
    return ast.get_source_segment(source, method) or ""


def test_controller_microphone_test_start_and_stop_are_owner_delegates() -> None:
    start = _method_source(CONTROLLER_PATH, "GuiController", "start_microphone_test")
    stop = _method_source(CONTROLLER_PATH, "GuiController", "stop_microphone_test")

    assert "MicrophoneTestSessionRequest(" in start
    assert "_get_microphone_test_owner().start(" in start
    assert "runtime.start(" not in start
    assert "_get_microphone_test_lifecycle_lock" not in start
    assert "owner.stop()" in stop
    assert "runtime.stop()" not in stop


def test_controller_microphone_test_state_is_backed_by_owner_properties() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")

    assert "_microphone_test_owner: MicrophoneTestSessionOwner | None" in source
    assert "_microphone_test_lifecycle_lock: asyncio.Lock | None" not in source
    assert "_microphone_test_meter_level: float = field(" not in source
    assert "_microphone_test_runtime: MicTestRuntime | None = field(" not in source
    assert 'owner_name="MicrophoneTestSessionOwner"' in source


def test_microphone_test_session_owner_has_no_ui_or_controller_dependency() -> None:
    source = OWNER_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in source
    assert "GuiController" not in source
    assert "MicTestRuntime" in source
    assert "MicrophoneTestSessionRequest" in source
    assert "drop meter updates from stale runtime generations" in source
