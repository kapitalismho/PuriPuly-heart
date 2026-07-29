from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = ROOT / "src" / "puripuly_heart" / "ui" / "controller.py"


def _method_source(path: Path, method_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"method not found: {method_name}")


def test_controller_vrc_mic_stop_compatibility_helper_is_absent() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")

    assert "def _stop_vrc_mic_receiver(" not in source


def test_controller_vrc_mic_configure_remains_owner_delegate() -> None:
    method = _method_source(CONTROLLER_PATH, "_configure_vrc_mic_receiver")

    assert "_get_vrc_mic_sync_owner().configure(enabled=enabled)" in method
