import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = REPO_ROOT / "src" / "puripuly_heart" / "ui" / "controller.py"


def _method_source(class_name: str, method_name: str) -> str:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")
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


def test_controller_manual_typing_owner_is_only_factory_composition() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")
    method = _method_source("GuiController", "_get_manual_typing_owner")

    assert "def _begin_manual_submit_typing(" not in source
    assert "def _manual_typing_idle_task(" not in source
    assert "create_manual_typing_owner(" in method
    assert "ManualTypingOwner(" not in method
    assert 'getattr(hub, "set_self_chatbox_typing_reason", None)' in method
    assert 'getattr(hub, "clear_self_chatbox_typing_reasons", None)' in method
    assert 'getattr(runtime, "translation_tasks", None)' in method
    assert "log_detailed=lambda message: self.log_detailed(message)" in method
    assert "log_error=lambda message: self._log_error(message)" in method
    assert "idle_timeout_seconds=MANUAL_INPUT_TYPING_IDLE_TIMEOUT_S" in method
    assert "submit_timeout_seconds=MANUAL_SUBMIT_TYPING_TIMEOUT_S" in method
    assert "self.osc" not in method


def test_controller_manual_submit_preserves_self_source() -> None:
    method = _method_source("GuiController", "submit_text")

    assert 'hub.submit_text(text, source="You")' in method
    assert "self.osc" not in method
