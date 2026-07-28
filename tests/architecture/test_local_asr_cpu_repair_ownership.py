from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src" / "puripuly_heart"
CONTROLLER_PATH = SOURCE_ROOT / "ui" / "controller.py"
OWNER_PATH = SOURCE_ROOT / "app" / "services" / "local_asr_cpu_repair.py"


def _method_source(path: Path, method_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"method not found: {method_name}")


def test_cpu_repair_owner_is_ui_and_settings_neutral() -> None:
    source = OWNER_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in source
    assert "puripuly_heart.config.settings" not in source
    assert "AppSettings" not in source
    assert "GuiController" not in source
    assert "flet" not in source


def test_cpu_repair_owner_is_constructed_only_by_application_composition() -> None:
    constructions: list[str] = []
    for source_file in sorted(SOURCE_ROOT.rglob("*.py")):
        source = source_file.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "LocalASRCpuRepairOwner"
            ):
                constructions.append(source_file.relative_to(ROOT).as_posix())

    assert constructions == ["src/puripuly_heart/app/wiring_composition.py"]


def test_controller_cpu_repair_commands_are_compatibility_delegates() -> None:
    request = _method_source(CONTROLLER_PATH, "_request_unavailable_local_asr_repair")
    clear = _method_source(
        CONTROLLER_PATH,
        "_clear_local_stt_pending_enable_if_provider_switched_away",
    )
    getter = _method_source(CONTROLLER_PATH, "_get_local_asr_cpu_repair_owner")

    assert ".request_repair(" in request
    assert "LocalASRCpuRepairRequest(" in request
    assert "LocalASRInstallRequest" not in request
    assert ".clear_if_provider_switched_away()" in clear
    assert "create_local_asr_cpu_repair_owner(" in getter
    assert "LocalASRCpuRepairOwner(" not in getter


def test_controller_has_no_cpu_repair_state_fields_or_result_algorithm() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    controller = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GuiController"
    )
    annotated_fields = {
        node.target.id
        for node in controller.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }

    assert "_local_stt_pending_enable_after_install" not in annotated_fields
    assert "_local_stt_pending_enable_generation" not in annotated_fields
    assert "_local_stt_pending_peer_enable_after_install" not in annotated_fields
    assert "_handle_local_stt_install_result" not in source
    assert "LocalASRInstallRequest" not in source
    assert "LocalASRInstallResult" not in source
    assert "_self_pending" in OWNER_PATH.read_text(encoding="utf-8")
    assert "_peer_pending" in OWNER_PATH.read_text(encoding="utf-8")
