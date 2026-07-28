from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = ROOT / "src" / "puripuly_heart" / "ui" / "controller.py"
OWNER_PATH = ROOT / "src" / "puripuly_heart" / "app" / "services" / "gpu_provider_recovery.py"


def _controller_method_source(method_name: str) -> str:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    controller = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GuiController"
    )
    method = next(
        node
        for node in controller.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == method_name
    )
    return ast.get_source_segment(source, method) or ""


def test_controller_delegates_manual_and_settings_gpu_recovery_to_owner() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")
    manual = _controller_method_source("retry_gpu_activation")
    settings = _controller_method_source("_apply_gpu_runtime_owner_recovery")

    assert "_get_gpu_provider_recovery_owner().recover(" in manual
    assert "_get_gpu_provider_recovery_owner().recover(" in settings
    for retired_name in (
        "_gpu_provider_recovery_lock",
        "_get_gpu_provider_recovery_lock",
        "_apply_gpu_runtime_owner_recovery_locked",
        "_execute_gpu_provider_recovery_retry",
        "_build_gpu_recovery_request",
        "_abort_provider_recoveries",
        "_resume_gpu_provider_consumers",
    ):
        assert retired_name not in source


def test_gpu_provider_recovery_owner_has_no_ui_controller_or_settings_dependency() -> None:
    source = OWNER_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in source
    assert "GuiController" not in source
    assert "AppSettings" not in source
    assert "asyncio.Lock" in source
    assert "runtime.recover_gpu(" in source
    assert "await item.plan.adopt(" in source
    assert "self._abort_channels(prepared)" in source
