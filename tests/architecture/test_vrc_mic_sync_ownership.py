from __future__ import annotations

import ast
from pathlib import Path

from tests.helpers.paths import REPO_ROOT as ROOT

CONTROLLER_PATH = ROOT / "src" / "puripuly_heart" / "composition" / "application_runtime.py"


def _method_source(path: Path, method_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"method not found: {method_name}")


def test_application_has_no_vrc_mic_stop_compatibility_helper() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")

    assert "def _stop_vrc_mic_receiver(" not in source


def test_application_vrc_mic_configure_remains_owner_delegate() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")

    assert source.count("compose_vrc_mic_sync(") == 1
    assert (
        "configure_vrc_mic=lambda *, enabled: "
        "(require_vrc_mic_sync().configure(enabled=enabled))"
    ) in source
