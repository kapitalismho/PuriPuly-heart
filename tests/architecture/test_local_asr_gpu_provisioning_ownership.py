from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src" / "puripuly_heart"
CONTROLLER_PATH = SOURCE_ROOT / "ui" / "controller.py"
OWNER_PATH = SOURCE_ROOT / "app" / "services" / "local_asr_gpu_provisioning.py"
INTERACTION_PATH = SOURCE_ROOT / "app" / "services" / "gpu_runtime_interaction.py"
COMPOSITION_PATH = SOURCE_ROOT / "app" / "wiring_composition.py"


def _method_source(path: Path, method_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"method not found: {method_name}")


def test_gpu_provisioning_owner_is_ui_and_settings_neutral() -> None:
    source = OWNER_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in source
    assert "puripuly_heart.config.settings" not in source
    assert "AppSettings" not in source
    assert "GuiController" not in source
    assert "flet" not in source


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


def test_controller_gpu_provisioning_commands_are_compatibility_delegates() -> None:
    selected = _method_source(CONTROLLER_PATH, "install_selected_gpu_model_if_needed")
    repair = _method_source(CONTROLLER_PATH, "install_or_repair_gpu_model")
    getter = _method_source(CONTROLLER_PATH, "_get_gpu_runtime_interaction_owner")

    assert "install_selected_model_if_needed()" in selected
    assert "_get_gpu_runtime_interaction_owner()" in selected
    assert "LocalASRInstallRequest" not in selected
    assert ".install_or_repair(" in repair
    assert "_get_gpu_runtime_interaction_owner()" in repair
    assert "LocalASRInstallRequest" not in repair
    assert "create_gpu_runtime_interaction_owner(" in getter
    assert "LocalASRGpuProvisioningOwner(" not in getter
    assert "LocalASRGpuProvisioningOwner(" in INTERACTION_PATH.read_text(encoding="utf-8")
