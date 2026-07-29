import ast
from pathlib import Path

from puripuly_heart.app.ports.ui_presentation import UiPresentationPort
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER_PATH = (
    REPO_ROOT
    / "src"
    / "puripuly_heart"
    / "release_evidence"
    / "local_asr_production_composition.py"
)
PORT_PATH = (
    REPO_ROOT / "src" / "puripuly_heart" / "app" / "ports" / "local_asr_production_evidence.py"
)
COMPOSITION_PATH = (
    REPO_ROOT / "src" / "puripuly_heart" / "composition" / "local_asr_production_evidence.py"
)
UI_COMPOSITION_PATH = REPO_ROOT / "src" / "puripuly_heart" / "composition" / "ui_application.py"


def test_local_asr_evidence_driver_has_no_controller_dependency() -> None:
    source = DRIVER_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui.controller" not in source
    assert "GuiController" not in source
    for private_name in (
        "_load_or_init_settings",
        "_init_pipeline",
        "_self_stt_provider_request",
        "_build_peer_runtime_config",
        "_peer_stt_provider_request",
    ):
        assert f".{private_name}(" not in source


def test_evidence_contract_is_ui_neutral_and_composition_is_page_free() -> None:
    port_source = PORT_PATH.read_text(encoding="utf-8")
    composition_source = COMPOSITION_PATH.read_text(encoding="utf-8")
    ui_composition_source = UI_COMPOSITION_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in port_source
    assert "compose_gui_controller(" in composition_source
    assert "page=None" in ui_composition_source
    imported_modules = {
        node.module
        for node in ast.walk(ast.parse(composition_source))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(ast.parse(composition_source))
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert "flet" not in imported_modules
    assert "_ControllerBackedLocalASRProductionEvidence" in composition_source
    required_presentation_members = {
        name for name in UiPresentationPort.__dict__ if not name.startswith("_")
    }
    assert required_presentation_members <= set(dir(FletUiPresentationAdapter))
