from __future__ import annotations

from tests.helpers.paths import REPO_ROOT as ROOT

CONTROLLER_PATH = ROOT / "src" / "puripuly_heart" / "composition" / "application_runtime.py"
CAPTURE_WIRING_PATH = (
    ROOT / "src" / "puripuly_heart" / "app" / "wiring" / "wiring_capture_runtime.py"
)
ADAPTER_PATH = (
    ROOT / "src" / "puripuly_heart" / "app" / "adapters" / "self_capture" / "self_capture_source.py"
)
VAD_ADAPTER_PATH = (
    ROOT / "src" / "puripuly_heart" / "app" / "adapters" / "self_capture" / "self_capture_vad.py"
)
AUDIO_LOOP_ADAPTER_PATH = (
    ROOT
    / "src"
    / "puripuly_heart"
    / "app"
    / "adapters"
    / "self_capture"
    / "self_capture_audio_loop.py"
)


def test_capture_wiring_composes_self_source_adapter_without_controller_algorithm() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")
    composition = CAPTURE_WIRING_PATH.read_text(encoding="utf-8")

    assert "source_factory=create_self_capture_source_adapter(" in composition
    assert "source_factory=create_self_capture_source_adapter(" not in source
    assert "_create_self_capture_source" not in source
    assert "All microphone attempts failed" not in source
    assert "name_fallback" not in source
    assert "system_default" not in source


def test_self_capture_source_adapter_has_no_ui_settings_or_lifecycle_ownership() -> None:
    source = ADAPTER_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in source
    assert "AppSettings" not in source
    assert "asyncio" not in source
    assert "async def close(" not in source
    assert "SelfCaptureSessionConfig" in source
    assert "name_fallback" in source
    assert "system_default" in source
    assert "will_retry_mono=True" in source


def test_capture_wiring_composes_self_vad_adapter_without_controller_algorithm() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")
    composition = CAPTURE_WIRING_PATH.read_text(encoding="utf-8")

    assert "vad_factory=create_self_capture_vad_adapter(" in composition
    assert "vad_factory=create_self_capture_vad_adapter(" not in source
    assert "_create_self_capture_vad" not in source
    assert "SileroVadOnnx" not in source
    assert "VadGating" not in source


def test_self_vad_adapter_has_no_ui_peer_vad_or_session_lifecycle_ownership() -> None:
    source = VAD_ADAPTER_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in source
    assert "AppSettings" not in source
    assert "PeerCaptureSessionConfig" not in source
    assert "run_audio_vad_loop" not in source
    assert "asyncio" not in source
    assert "async def close(" not in source
    assert "SelfCaptureSessionConfig" in source


def test_capture_wiring_composes_self_audio_loop_without_controller_algorithm() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")
    composition = CAPTURE_WIRING_PATH.read_text(encoding="utf-8")

    assert "run_audio_loop=create_self_capture_audio_loop_adapter(" in composition
    assert "audio_gate_provider=lambda: audio_gate" in composition
    assert "run_audio_loop=create_self_capture_audio_loop_adapter(" not in source
    assert "_run_self_capture_audio_loop" not in source


def test_self_audio_loop_adapter_has_no_ui_task_or_gate_lifecycle_ownership() -> None:
    source = AUDIO_LOOP_ADAPTER_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in source
    assert "AppSettings" not in source
    assert "VrcMicSyncOwner" not in source
    assert "asyncio.create_task" not in source
    assert "async def close(" not in source
    assert 'channel_label="self"' in source
    assert "audio_gate=self.audio_gate_provider()" in source
