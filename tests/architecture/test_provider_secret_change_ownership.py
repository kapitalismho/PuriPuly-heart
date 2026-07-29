from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UI_RUNTIME_PATH = ROOT / "src" / "puripuly_heart" / "app" / "adapters" / "ui_runtime.py"


def _adapter_method_source(method_name: str) -> str:
    source = UI_RUNTIME_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    adapter = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "UiProviderRuntimeAdapter"
    )
    method = next(
        node
        for node in adapter.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == method_name
    )
    return ast.get_source_segment(source, method) or ""


def test_ui_delegates_complete_provider_secret_transaction_owner() -> None:
    source = UI_RUNTIME_PATH.read_text(encoding="utf-8")
    method = _adapter_method_source("persist_provider_secret_change")

    assert "_provider_secret_change_serialization_owner" not in source
    assert "_persist_provider_secret_change_serialized" not in source
    assert "self.provider_settings.change_secret(key, value)" in method
    assert "_provider_secret_change_execution" not in source
    assert "_apply_provider_secret_change_result" not in source
    assert "LifecycleScope" not in source
    assert "start_lifecycle_task" not in source
    assert "asyncio.shield" not in source
