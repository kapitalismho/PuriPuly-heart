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


def test_controller_presence_owner_is_only_factory_composition() -> None:
    method = _method_source("GuiController", "_get_vrchat_osc_presence_owner")

    assert "create_vrchat_osc_presence_probe_owner(" in method
    assert "VrchatOscPresenceProbeOwner(" not in method
    assert "presence_provider=lambda: self.vrchat_osc_presence" in method
    assert "port_provider=self._vrchat_osc_probe_port" in method
    assert "publish_notice=self.app.set_dashboard_vrchat_osc_notice" in method
