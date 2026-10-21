from __future__ import annotations

from tests.helpers.ast_sources import method_source as _method_source
from tests.helpers.paths import REPO_ROOT as ROOT

CONTROLLER_PATH = ROOT / "src" / "puripuly_heart" / "composition" / "application_runtime.py"
RUNTIME_PATH = ROOT / "src" / "puripuly_heart" / "app" / "wiring" / "wiring_audio_test.py"


def test_audio_test_runtime_delegates_start_and_stop_to_the_session_owner() -> None:
    start = _method_source(RUNTIME_PATH, "AudioTestRuntime", "start")
    stop = _method_source(RUNTIME_PATH, "AudioTestRuntime", "stop")

    assert "return await self.owner().start(" in start
    assert "await self._owner.stop()" in stop


def test_controller_has_no_duplicate_audio_test_state_or_capture_algorithms() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")

    assert "_microphone_test_lifecycle_lock" not in source
    assert "_microphone_test_meter_level" not in source
    assert "AudioTestCaptureRuntime" not in source
    assert "def _prepare_microphone_test_capture(" not in source
    assert "def run_microphone_test_capture(" not in source


def test_production_audio_test_session_uses_wiring_capture_and_no_ui() -> None:
    factory = _method_source(RUNTIME_PATH, "AudioTestRuntime", "owner")

    assert "capture_port=create_audio_test_capture_adapter(" in factory
    assert "plan_factory=self.plan" in factory
    assert "puripuly_heart.ui" not in RUNTIME_PATH.read_text(encoding="utf-8")
