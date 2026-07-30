import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSITION_PATH = REPO_ROOT / "src" / "puripuly_heart" / "composition" / "application_runtime.py"
UI_RUNTIME_PATH = REPO_ROOT / "src" / "puripuly_heart" / "app" / "adapters" / "ui_runtime.py"


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


def test_application_manual_typing_owner_is_only_factory_composition() -> None:
    source = COMPOSITION_PATH.read_text(encoding="utf-8")

    assert "def _begin_manual_submit_typing(" not in source
    assert "def _manual_typing_idle_task(" not in source
    assert source.count("create_manual_typing_owner(") == 1
    assert "return pipeline.translation_output_projection" in source
    assert 'getattr(hub, "set_self_chatbox_typing_reason", None)' not in source
    assert '"clear_self_chatbox_typing_reasons",' not in source
    assert 'getattr(runtime, "translation_tasks", None)' in source
    assert "idle_timeout_seconds=MANUAL_INPUT_TYPING_IDLE_TIMEOUT_S" in source
    assert "submit_timeout_seconds=MANUAL_SUBMIT_TYPING_TIMEOUT_S" in source


def test_ui_manual_submit_preserves_self_source() -> None:
    method = _method_source(UI_RUNTIME_PATH, "UiInputRuntimeAdapter", "submit_text")

    assert 'hub.submit_text(text, source="You")' in method
