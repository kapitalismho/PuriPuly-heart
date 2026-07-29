from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = ROOT / "src" / "puripuly_heart" / "ui" / "controller.py"
OWNER_PATH = ROOT / "src" / "puripuly_heart" / "app" / "services" / "overlay_generation_start.py"
APPLICATION_OWNER_PATH = (
    ROOT / "src" / "puripuly_heart" / "app" / "services" / "overlay_application.py"
)
RUNTIME_PATH = ROOT / "src" / "puripuly_heart" / "core" / "runtime" / "overlay.py"


def test_overlay_application_owns_generation_start_without_controller_wrapper() -> None:
    controller = CONTROLLER_PATH.read_text(encoding="utf-8")
    application = APPLICATION_OWNER_PATH.read_text(encoding="utf-8")

    assert "async def _run_overlay_start(" not in controller
    assert "async def run_start(" in application
    assert "self._generation_owner.start(" in application
    assert "self._generation_request" in application
    assert "self._generation_effects()" in application
    for constructor in (
        "OverlayDiagnosticsRecorder(",
        "OverlayPresenter(",
        "OverlayBridge(",
        "asyncio.Queue(",
        "OverlayProcessManager(",
    ):
        assert constructor not in controller


def test_generation_owner_has_no_ui_session_transition_or_fallback_dependency() -> None:
    source = OWNER_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in source
    assert "GuiController" not in source
    assert "OverlaySessionTransitionOwner" not in source
    assert "OverlaySessionFallbackOwner" not in source
    assert "OverlayDiagnosticsRecorder(" in source
    assert "OverlayPresenter(" in source
    assert "OverlayBridge(" in source
    assert "OverlayProcessManager(" in source
    assert "runtime.create_monitor_task(" in source


def test_generation_resource_and_cross_generation_owners_remain_separate() -> None:
    controller = CONTROLLER_PATH.read_text(encoding="utf-8")
    application = APPLICATION_OWNER_PATH.read_text(encoding="utf-8")
    owner = OWNER_PATH.read_text(encoding="utf-8")
    runtime = RUNTIME_PATH.read_text(encoding="utf-8")

    assert "OverlaySessionTransitionOwner" not in controller
    assert "OverlaySessionFallbackOwner" not in controller
    assert "OverlaySessionTransitionOwner" in application
    assert "OverlaySessionFallbackOwner" in application
    assert "OverlayRuntimeHandle" in owner
    assert "async def close(" not in owner
    assert "class OverlayRuntimeHandle" in runtime
    for resource in (
        "_presenter",
        "_bridge",
        "_process_manager",
        "_monitor_task",
        "_renderer_event_task",
    ):
        assert resource in runtime
