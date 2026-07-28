from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src" / "puripuly_heart"
CONTROLLER_PATH = SOURCE_ROOT / "ui" / "controller.py"
OWNER_PATH = SOURCE_ROOT / "app" / "services" / "local_asr_diagnostics.py"
COMPOSITION_PATH = SOURCE_ROOT / "app" / "wiring_composition.py"
BASIC_LOGGING_TEST_PATH = ROOT / "tests" / "ui" / "test_asr_basic_logging.py"


def test_local_asr_diagnostics_owner_is_ui_and_settings_neutral() -> None:
    source = OWNER_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in source
    assert "puripuly_heart.config.settings" not in source
    assert "AppSettings" not in source
    assert "GuiController" not in source
    assert "flet" not in source


def test_local_asr_diagnostics_owner_is_constructed_only_by_application_composition() -> None:
    constructions: list[str] = []
    for source_file in sorted(SOURCE_ROOT.rglob("*.py")):
        source = source_file.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "LocalASRDiagnosticsOwner"
            ):
                constructions.append(source_file.relative_to(ROOT).as_posix())

    assert constructions == ["src/puripuly_heart/app/wiring_composition.py"]


def test_controller_composes_local_asr_diagnostics_without_rendering_algorithms() -> None:
    controller = CONTROLLER_PATH.read_text(encoding="utf-8")
    composition = COMPOSITION_PATH.read_text(encoding="utf-8")

    assert "def _on_local_asr_provider_runtime_diagnostic(" not in controller
    assert "def _local_asr_transition_diagnostic(" not in controller
    assert "def _log_local_asr_load_result(" not in controller
    assert "create_local_asr_diagnostics_owner(" in controller
    assert ".provider_runtime_diagnostic" in controller
    assert controller.count(".transition_diagnostic") == 1
    assert "return LocalASRDiagnosticsOwner(" in composition


def test_basic_logging_contract_targets_owner_without_controller_fixture() -> None:
    source = BASIC_LOGGING_TEST_PATH.read_text(encoding="utf-8")

    assert "LocalASRDiagnosticsOwner" in source
    assert "GuiController" not in source
    assert "puripuly_heart.ui.controller" not in source
