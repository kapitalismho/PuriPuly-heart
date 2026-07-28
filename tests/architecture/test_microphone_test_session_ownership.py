from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = ROOT / "src" / "puripuly_heart" / "ui" / "controller.py"
OWNER_PATH = ROOT / "src" / "puripuly_heart" / "app" / "services" / "microphone_test.py"
ADAPTER_PATH = ROOT / "src" / "puripuly_heart" / "app" / "adapters" / "microphone_test_capture.py"


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
    assert "def _prepare_microphone_test_capture(" not in source
    assert "def _self_stt_active_or_desired_for_microphone_test(" not in source
    assert "def _log_microphone_test_stt_auto_off(" not in source
    assert 'owner_name="MicrophoneTestSessionOwner"' in source


def test_microphone_test_session_owner_has_no_ui_or_controller_dependency() -> None:
    source = OWNER_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in source
    assert "GuiController" not in source
    assert "MicTestRuntime" in source
    assert "MicrophoneTestCapturePort" in source
    assert "await self.capture_port.capture(" in source
    assert "MicrophoneTestSessionRequest" in source
    assert "MicrophoneTestSelfCaptureState" in source
    assert "await self.disable_self_capture()" in source
    assert "self microphone source still open after STT auto-off" in source
    assert "drop meter updates from stale runtime generations" in source


def test_production_microphone_test_session_does_not_reenter_controller_capture() -> None:
    factory = _method_source(
        CONTROLLER_PATH,
        "GuiController",
        "_get_microphone_test_owner",
    )

    assert "capture_port=self._build_microphone_test_capture_adapter()" in factory
    assert "capture_request_factory=self._microphone_test_capture_request" in factory
    assert "run_microphone_test_capture(" not in factory


def test_controller_direct_microphone_test_capture_compatibility_method_is_removed() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")

    assert "def run_microphone_test_capture(" not in source


def test_microphone_test_capture_adapter_has_no_ui_or_controller_dependency() -> None:
    source = ADAPTER_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in source
    assert "GuiController" not in source
    assert "MicrophoneTestCaptureRequest" in source
    assert "MicrophoneTestRuntimePort" in source
    assert "source_factory" in source
