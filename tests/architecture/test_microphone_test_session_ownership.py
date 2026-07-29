from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = ROOT / "src" / "puripuly_heart" / "composition" / "application_runtime.py"
RUNTIME_PATH = ROOT / "src" / "puripuly_heart" / "app" / "wiring_microphone_test.py"


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


def test_microphone_test_runtime_owns_start_and_stop() -> None:
    start = _method_source(RUNTIME_PATH, "MicrophoneTestRuntime", "start")
    stop = _method_source(RUNTIME_PATH, "MicrophoneTestRuntime", "stop")

    assert "return await self.owner().start(" in start
    assert "MicrophoneTestSessionRequest(" in start
    assert "await self._owner.stop()" in stop


def test_controller_has_no_duplicate_microphone_test_state_or_capture_algorithms() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")

    assert "_microphone_test_lifecycle_lock: asyncio.Lock | None" not in source
    assert "_microphone_test_meter_level: float = field(" not in source
    assert "_microphone_test_runtime: MicTestRuntime | None = field(" not in source
    assert "def _microphone_test_runtime(" not in source
    assert "@_microphone_test_runtime.setter" not in source
    assert "def _prepare_microphone_test_capture(" not in source
    assert "def _self_stt_active_or_desired_for_microphone_test(" not in source
    assert "def _log_microphone_test_stt_auto_off(" not in source


def test_production_microphone_test_session_does_not_reenter_controller_capture() -> None:
    factory = _method_source(RUNTIME_PATH, "MicrophoneTestRuntime", "owner")

    assert "capture_port=self.capture_port()" in factory
    assert "capture_request_factory=self.capture_request" in factory
    assert "run_microphone_test_capture(" not in factory
    assert "puripuly_heart.ui" not in RUNTIME_PATH.read_text(encoding="utf-8")


def test_controller_direct_microphone_test_capture_compatibility_method_is_removed() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")

    assert "def run_microphone_test_capture(" not in source
