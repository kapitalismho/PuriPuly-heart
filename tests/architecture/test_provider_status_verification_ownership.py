from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = ROOT / "src" / "puripuly_heart" / "ui" / "controller.py"
OWNER_PATH = (
    ROOT / "src" / "puripuly_heart" / "app" / "services" / "provider_status_verification.py"
)


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


def test_controller_schedules_status_request_and_result_delivery_through_owner() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")
    method = _method_source(
        CONTROLLER_PATH,
        "GuiController",
        "_schedule_provider_status_verification",
    )

    assert "def _verify_and_update_status(" not in source
    assert "request_factory=self._build_provider_status_verification_request" in method
    assert "_get_provider_status_verification_owner().schedule(" in method
    assert "result_handler=self._apply_provider_status_verification_result" in method
    assert "_verify_and_update_status" not in method


def test_provider_status_owner_has_no_ui_or_controller_dependency() -> None:
    source = OWNER_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in source
    assert "GuiController" not in source
    assert "ConfiguredProviderStatusVerificationRequest" in source
    assert "ConfiguredProviderStatusVerificationResult" in source
    assert "ProviderVerifierPort" in source
