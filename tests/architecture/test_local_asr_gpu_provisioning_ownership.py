from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src" / "puripuly_heart"
COMPOSITION_PATH = SOURCE_ROOT / "composition" / "application_runtime.py"
UI_RUNTIME_PATH = SOURCE_ROOT / "app" / "adapters" / "ui_runtime.py"
INTERACTION_PATH = SOURCE_ROOT / "app" / "services" / "gpu_runtime_interaction.py"
COMPOSITION_PATH = SOURCE_ROOT / "app" / "wiring_composition.py"


def _method_source(path: Path, method_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"method not found: {method_name}")


def test_gpu_provisioning_owner_is_constructed_only_by_gpu_interaction_owner() -> None:
    constructions: list[str] = []
    for source_file in sorted(SOURCE_ROOT.rglob("*.py")):
        source = source_file.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "LocalASRGpuProvisioningOwner"
            ):
                constructions.append(source_file.relative_to(ROOT).as_posix())

    assert constructions == ["src/puripuly_heart/app/services/gpu_runtime_interaction.py"]


def test_ui_gpu_provisioning_command_delegates_to_composed_interaction_owner() -> None:
    selected = _method_source(UI_RUNTIME_PATH, "install_selected_gpu_model_if_needed")
    composition = COMPOSITION_PATH.read_text(encoding="utf-8")

    assert "self.gpu.install_selected_model_if_needed()" in selected
    assert "LocalASRInstallRequest" not in selected
    assert composition.count("create_gpu_runtime_interaction_owner(") == 1
    assert "LocalASRGpuProvisioningOwner(" not in composition
    assert "LocalASRGpuProvisioningOwner(" in INTERACTION_PATH.read_text(encoding="utf-8")
