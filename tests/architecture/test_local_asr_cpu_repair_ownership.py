from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src" / "puripuly_heart"
COMPOSITION_PATH = SOURCE_ROOT / "composition" / "application_runtime.py"
OWNER_PATH = SOURCE_ROOT / "app" / "services" / "local_asr" / "local_asr_cpu_repair.py"
WIRING_PATH = SOURCE_ROOT / "app" / "wiring_local_asr_application.py"


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


def test_local_asr_application_wiring_owns_cpu_repair_composition() -> None:
    composition_source = COMPOSITION_PATH.read_text(encoding="utf-8")
    wiring_source = WIRING_PATH.read_text(encoding="utf-8")

    assert "_request_unavailable_local_asr_repair" not in composition_source
    assert "_clear_local_stt_pending_enable_if_provider_switched_away" not in composition_source
    assert "_get_local_asr_cpu_repair_owner" not in composition_source
    assert "create_local_asr_cpu_repair_owner(" in wiring_source
    assert "LocalASRCpuRepairRequest(" in wiring_source
    assert "LocalASRInstallRequest" not in wiring_source


def test_composition_has_no_cpu_repair_state_fields_or_result_algorithm() -> None:
    source = COMPOSITION_PATH.read_text(encoding="utf-8")

    assert "_local_stt_pending_enable_after_install" not in source
    assert "_local_stt_pending_enable_generation" not in source
    assert "_local_stt_pending_peer_enable_after_install" not in source
    assert "_handle_local_stt_install_result" not in source
    assert "LocalASRInstallRequest" not in source
    assert "LocalASRInstallResult" not in source
    assert "_self_pending" in OWNER_PATH.read_text(encoding="utf-8")
    assert "_peer_pending" in OWNER_PATH.read_text(encoding="utf-8")
