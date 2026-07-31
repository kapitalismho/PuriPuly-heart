from __future__ import annotations

from tests.helpers.paths import REPO_ROOT as ROOT
from tests.helpers.paths import SOURCE_ROOT

OWNER_COMPOSITION_PATH = SOURCE_ROOT / "app" / "wiring" / "wiring_composition.py"
APPLICATION_COMPOSITION_PATH = SOURCE_ROOT / "composition" / "application_runtime.py"
BASIC_LOGGING_TEST_PATH = ROOT / "tests" / "ui" / "test_asr_basic_logging.py"


def test_application_composes_local_asr_diagnostics_without_rendering_algorithms() -> None:
    application_composition = APPLICATION_COMPOSITION_PATH.read_text(encoding="utf-8")
    composition = OWNER_COMPOSITION_PATH.read_text(encoding="utf-8")

    assert "def _on_local_asr_provider_runtime_diagnostic(" not in application_composition
    assert "def _local_asr_transition_diagnostic(" not in application_composition
    assert "def _log_local_asr_load_result(" not in application_composition
    assert application_composition.count("create_local_asr_diagnostics_owner(") == 1
    assert application_composition.count(".provider_runtime_diagnostic") == 1
    assert application_composition.count(".transition_diagnostic") == 1
    assert "return LocalASRDiagnosticsOwner(" in composition


def test_basic_logging_contract_targets_owner_without_controller_fixture() -> None:
    source = BASIC_LOGGING_TEST_PATH.read_text(encoding="utf-8")

    assert "LocalASRDiagnosticsOwner" in source
