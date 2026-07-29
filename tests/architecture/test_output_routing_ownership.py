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
    composition_source = (SOURCE_ROOT / "composition" / "application_runtime.py").read_text(
        encoding="utf-8"
    )
    overlay_source = (SOURCE_ROOT / "app" / "services" / "overlay_application.py").read_text(
        encoding="utf-8"
    )
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
        assert f"self.osc.{side_effect}(" not in composition_source
    assert "self.output_runtime.publish_overlay_event(" in hub_source
    assert "self.output_runtime.replace_overlay_sink(" in hub_source
    assert "self.output_runtime.publish_chatbox(" in hub_source
    assert "self.output_runtime.publish_system_disclosure_chatbox(" in hub_source
    assert "self.output_runtime.publish_system_immediate_chatbox(" in hub_source
    assert "self.output_runtime.set_self_chatbox_typing_reason(" in hub_source
    assert "self.output_runtime.clear_self_chatbox_typing_reasons(" in hub_source
    assert "overlay_sink.emit(" in output_source
    assert "self.chatbox.enqueue(" in output_source
    assert "self.chatbox.send_immediate(" in output_source
    assert "self.chatbox.set_typing_reason(" in output_source
    assert "self.chatbox.clear_typing_reasons(" in output_source
    assert "await self.replace_hub_sink(" in overlay_source


def test_flet_composition_uses_owner_without_importing_output_implementation() -> None:
    composition_source = (SOURCE_ROOT / "composition" / "application_runtime.py").read_text(
        encoding="utf-8"
    )
    ui_composition_source = (SOURCE_ROOT / "composition" / "ui_application.py").read_text(
        encoding="utf-8"
    )
    pipeline_source = (SOURCE_ROOT / "app" / "wiring_runtime_pipeline.py").read_text(
        encoding="utf-8"
    )
    imported_modules: set[str] = set()
    for source_file in sorted((SOURCE_ROOT / "ui").rglob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.add(node.module)

    assert "_init_pipeline" not in composition_source
    for retired_assembly in (
        "_get_managed_account_components",
        "_get_provider_application_owner",
        "_get_provider_runtime_components",
        "_get_runtime_pipeline_launcher",
        "_get_self_capture_owner",
        "_get_capture_owner_factory",
    ):
        assert retired_assembly not in composition_source
    for extracted_construction in (
        "compose_managed_account(",
        "compose_provider_runtime(",
        "CaptureOwnerFactory(",
        "RuntimePipelineLauncher(",
    ):
        assert composition_source.count(extracted_construction) == 1
    assert "compose_runtime_pipeline(" in pipeline_source
    assert "ClientHub(" in pipeline_source
    assert "compose_application_runtime(" in ui_composition_source
    assert "compose_managed_account(" in composition_source
    assert "compose_provider_runtime(" in composition_source
    assert "CaptureOwnerFactory(" in composition_source
    assert "RuntimePipelineLauncher(" in composition_source
    assert "RuntimeCompositionComponents(" in composition_source
    assert "hub.output_runtime.start_ui_event_bridge(" in composition_source
    assert "puripuly_heart.core.runtime.output" not in imported_modules
    assert "puripuly_heart.core.output.router" not in imported_modules
