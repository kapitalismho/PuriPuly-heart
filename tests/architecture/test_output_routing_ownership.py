from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src" / "puripuly_heart"


def _repo_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def test_output_runtime_is_the_only_production_output_owner_construction() -> None:
    runtime_constructions: list[tuple[str, int]] = []
    router_constructions: list[tuple[str, int]] = []
    for source_file in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _call_name(node)
            if call_name == "OutputRuntime":
                runtime_constructions.append((_repo_path(source_file), node.lineno))
            elif call_name == "OutputRouter":
                router_constructions.append((_repo_path(source_file), node.lineno))

    assert [path for path, _line in runtime_constructions] == [
        "src/puripuly_heart/core/orchestrator/hub.py"
    ]
    assert router_constructions == []


def test_hub_delegates_output_side_effects_to_output_runtime() -> None:
    hub_source = (SOURCE_ROOT / "core" / "orchestrator" / "hub.py").read_text(encoding="utf-8")
    controller_source = (SOURCE_ROOT / "ui" / "controller.py").read_text(encoding="utf-8")
    output_source = (SOURCE_ROOT / "core" / "runtime" / "output.py").read_text(encoding="utf-8")

    assert "self.overlay_sink.emit(" not in hub_source
    for side_effect in (
        "enqueue",
        "send_immediate",
        "send_typing",
        "set_typing_reason",
        "clear_typing_reasons",
        "process_due",
        "drop_pending",
    ):
        assert f"self.osc.{side_effect}(" not in hub_source
        assert f"self.osc.{side_effect}(" not in controller_source
    assert "self.output_runtime.publish_overlay_event(" in hub_source
    assert "self.output_runtime.publish_chatbox(" in hub_source
    assert "self.output_runtime.publish_system_disclosure_chatbox(" in hub_source
    assert "self.output_runtime.publish_system_immediate_chatbox(" in hub_source
    assert "self.output_runtime.set_self_chatbox_typing_reason(" in hub_source
    assert "self.output_runtime.clear_self_chatbox_typing_reasons(" in hub_source
    assert "self.overlay_sink.emit(" in output_source
    assert "self.chatbox.enqueue(" in output_source
    assert "self.chatbox.send_immediate(" in output_source
    assert "self.chatbox.set_typing_reason(" in output_source
    assert "self.chatbox.clear_typing_reasons(" in output_source


def test_flet_composition_uses_owner_without_importing_output_implementation() -> None:
    controller_source = (SOURCE_ROOT / "ui" / "controller.py").read_text(encoding="utf-8")
    imported_modules: set[str] = set()
    for source_file in sorted((SOURCE_ROOT / "ui").rglob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.add(node.module)

    assert "ClientHub(" in controller_source
    assert "hub.output_runtime.start_ui_event_bridge(" in controller_source
    assert "puripuly_heart.core.runtime.output" not in imported_modules
    assert "puripuly_heart.core.output.router" not in imported_modules
