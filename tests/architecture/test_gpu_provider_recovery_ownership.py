from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = ROOT / "src" / "puripuly_heart" / "ui" / "controller.py"
PROVIDER_RUNTIME_PATH = ROOT / "src" / "puripuly_heart" / "app" / "wiring_provider_runtime.py"


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


def test_provider_runtime_owns_gpu_recovery_and_controller_has_no_algorithm() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")
    manual = _controller_method_source("retry_gpu_activation")
    provider_runtime = PROVIDER_RUNTIME_PATH.read_text(encoding="utf-8")

    assert "_get_gpu_provider_recovery_owner().recover(" in manual
    assert "_gpu_provider_recovery_request(" in manual
    assert "recover_gpu=effects.gpu_recovery" in provider_runtime
    for retired_name in (
        "_gpu_provider_recovery_lock",
        "_get_gpu_provider_recovery_lock",
        "_apply_gpu_runtime_owner_recovery_locked",
        "_execute_gpu_provider_recovery_retry",
        "_build_gpu_recovery_request",
        "_abort_provider_recoveries",
        "_resume_gpu_provider_consumers",
        "_gpu_provider_recovery_execution",
        "_complete_gpu_provider_recovery",
        "_desired_gpu_channels",
        "_gpu_provider_recovery_channel_plans",
    ):
        assert retired_name not in source
