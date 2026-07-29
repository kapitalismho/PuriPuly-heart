import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = REPO_ROOT / "src" / "puripuly_heart" / "ui" / "controller.py"
OWNER_PATH = REPO_ROOT / "src" / "puripuly_heart" / "app" / "services" / "peer_application.py"
DRIVER_PATH = (
    REPO_ROOT / "src" / "puripuly_heart" / "release_evidence" / "windows_process_isolation.py"
)


def _method_source(path: Path, class_name: str, method_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    )
    return ast.get_source_segment(source, method)


def test_controller_retry_is_only_an_owner_delegate() -> None:
    method = _method_source(CONTROLLER_PATH, "GuiController", "retry_peer_process_capture")

    assert "_get_peer_application_runtime().owner.retry_process_capture()" in method
    assert "_peer_process_warning_reason" not in method
    assert "_build_peer_runtime_config" not in method


def test_retry_owner_and_evidence_driver_stay_outside_ui_implementation() -> None:
    owner_source = OWNER_PATH.read_text(encoding="utf-8")
    driver_source = DRIVER_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in owner_source
    assert "puripuly_heart.ui.controller" not in driver_source
    assert "PeerApplicationOwner.retry_process_capture" in driver_source
