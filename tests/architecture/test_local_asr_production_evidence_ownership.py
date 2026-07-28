from pathlib import Path

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

    assert "puripuly_heart.ui" not in port_source
    assert "page=None" in composition_source
    assert "flet" not in composition_source.casefold()
    assert "_ControllerBackedLocalASRProductionEvidence" in composition_source
