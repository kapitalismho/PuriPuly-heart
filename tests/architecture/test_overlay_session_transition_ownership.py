from __future__ import annotations

from tests.helpers.paths import REPO_ROOT as ROOT

CONTROLLER_PATH = ROOT / "src" / "puripuly_heart" / "composition" / "application_runtime.py"
OWNER_PATH = (
    ROOT
    / "src"
    / "puripuly_heart"
    / "app"
    / "services"
    / "overlay"
    / "overlay_session_transition.py"
)
APPLICATION_OWNER_PATH = (
    ROOT / "src" / "puripuly_heart" / "app" / "services" / "overlay" / "overlay_application.py"
)


def test_overlay_application_owns_start_and_shutdown_transitions() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")
    application = APPLICATION_OWNER_PATH.read_text(encoding="utf-8")

    assert "async def _begin_overlay_start(" not in source
    assert "async def _shutdown_overlay_runtime(" not in source
    assert "async def begin_start(" in application
    assert "async def shutdown(" in application
    assert "self._transition_owner.begin_start(" in application
    assert "self._transition_owner.shutdown(" in application
    assert "_overlay_lock" not in source
    assert "async with self._overlay_lock" not in source


def test_transition_owner_preserves_overlay_runtime_handle_resource_ownership() -> None:
    source = OWNER_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in source
    assert "OverlayBridge" not in source
    assert "OverlayProcessManager" not in source
    assert "OverlayPresenter" not in source
    assert "runtime.create_start_task(" in source
    assert "execution.teardown()" in source
    assert "asyncio.Lock" in source
