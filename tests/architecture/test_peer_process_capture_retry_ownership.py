from tests.helpers.ast_sources import method_source as _method_source
from tests.helpers.paths import REPO_ROOT

UI_RUNTIME_PATH = REPO_ROOT / "src" / "puripuly_heart" / "app" / "adapters" / "ui_runtime.py"
OWNER_PATH = REPO_ROOT / "src" / "puripuly_heart" / "app" / "services" / "peer_application.py"
DRIVER_PATH = (
    REPO_ROOT / "src" / "puripuly_heart" / "release_evidence" / "windows_process_isolation.py"
)


def test_ui_retry_is_only_an_owner_delegate() -> None:
    method = _method_source(
        UI_RUNTIME_PATH,
        "UiPeerCaptureRuntimeAdapter",
        "retry_peer_process_capture",
    )

    assert "self.peer.owner.retry_process_capture()" in method
    assert "_peer_process_warning_reason" not in method
    assert "_build_peer_runtime_config" not in method


def test_retry_owner_and_evidence_driver_stay_outside_ui_implementation() -> None:
    owner_source = OWNER_PATH.read_text(encoding="utf-8")
    driver_source = DRIVER_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in owner_source
    assert "PeerApplicationOwner.retry_process_capture" in driver_source
