from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src" / "puripuly_heart"


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def test_production_constructs_one_local_asr_provisioning_owner_in_composition() -> None:
    constructions: list[str] = []
    for source_file in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node) == "LocalASRProvisioningOwner":
                constructions.append(source_file.relative_to(REPO_ROOT).as_posix())

    assert constructions == ["src/puripuly_heart/app/wiring.py"]


def test_controller_delegates_provisioning_without_asset_or_task_ownership() -> None:
    source = (SOURCE_ROOT / "ui" / "controller.py").read_text(encoding="utf-8")
    cpu_repair = (SOURCE_ROOT / "app" / "services" / "local_asr_cpu_repair.py").read_text(
        encoding="utf-8"
    )
    gpu_provisioning = (
        SOURCE_ROOT / "app" / "services" / "local_asr_gpu_provisioning.py"
    ).read_text(encoding="utf-8")
    gpu_interaction = (SOURCE_ROOT / "app" / "services" / "gpu_runtime_interaction.py").read_text(
        encoding="utf-8"
    )
    readiness = (SOURCE_ROOT / "app" / "services" / "local_asr_readiness.py").read_text(
        encoding="utf-8"
    )
    selection = (SOURCE_ROOT / "app" / "services" / "local_asr_selection.py").read_text(
        encoding="utf-8"
    )
    application_wiring = (SOURCE_ROOT / "app" / "wiring_local_asr_application.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)

    assert "create_local_asr_provisioning_owner(" in source
    assert "create_local_asr_cpu_repair_owner(" not in source
    assert "create_local_asr_cpu_repair_owner(" in application_wiring
    assert "create_gpu_runtime_interaction_owner(" in source
    assert "create_local_asr_readiness_owner(" not in source
    assert "create_local_asr_readiness_owner(" in application_wiring
    assert ".inspect_cpu(" in source
    assert ".inspect_gpu(" in source
    assert ".inspect_gpu(" in gpu_provisioning
    assert ".inspect_gpu_readiness(" in gpu_interaction
    assert ".start_install(" not in source
    assert ".start_install(" in cpu_repair
    assert ".start_install(" in gpu_provisioning
    assert "result_handler=" in cpu_repair
    assert ".report_model_validation_failure(" not in source
    assert ".report_model_validation_failure(" in readiness
    assert ".ensure_self_ready(" not in source
    assert ".ensure_self_ready(" in application_wiring
    assert ".ensure_peer_ready(" in source
    assert ".ensure_peer_ready(" in application_wiring
    assert "_probe_self_local_stt_runtime_load" not in source
    assert "LocalCPUAutoUnavailableError" not in source
    assert "LocalSTTModelMissingError" not in source
    assert "_required_local_stt_model_ids_for_provider" not in source
    assert "_local_stt_runtime_status_for_provider" not in source
    assert "def required_local_asr_model_ids(" in selection
    assert "def local_asr_status_for_provider(" in selection
    assert "required_local_asr_model_ids(" in readiness
    assert "local_asr_status_for_provider(" in readiness
    assert ".close()" in source
    assert "puripuly_heart.core.local_stt_runtime_installer" not in imports
    assert "puripuly_heart.core.local_stt_huggingface_xet_adapter" not in imports
    assert "puripuly_heart.core.runtime.local_stt_download" not in imports
    assert "inspect_local_stt_install_state" not in source
    assert "inspect_local_cpu_model_installs" not in source
    assert "inspect_required_cpu_model_installs" not in source
    assert "inspect_local_gpu_install" not in source
    assert "load_local_gpu_asset_manifest" not in source
    assert "ensure_local_stt_installed" not in source
    assert "cleanup_local_stt_install_residue" not in source
    assert "_local_stt_download_runtime" not in source
    assert "_gpu_install_runtime" not in source
    assert "_peer_local_stt_probe_task" not in source
    assert "_run_peer_local_stt_runtime_probe" not in source
    assert "_probe_peer_local_stt_runtime_load" not in source
    assert "_refresh_local_stt_runtime_state" not in source
    assert "_record_strict_local_stt_ready" not in source
    assert "_cancel_local_stt_download" not in source
    assert "_start_local_stt_download" not in source
    assert "_handle_local_stt_unavailable" not in source
    assert "_ui_background_scope" not in source


def test_provider_wiring_consumes_only_installed_path_contract() -> None:
    stt_wiring = (SOURCE_ROOT / "app" / "wiring_stt_factory.py").read_text(encoding="utf-8")
    provisioning = (SOURCE_ROOT / "core" / "runtime" / "local_asr_provisioning.py").read_text(
        encoding="utf-8"
    )

    assert "LocalASRProvisioningOwner" not in stt_wiring
    assert "create_stt_backend" not in provisioning
    assert "ManagedSTTProvider" not in provisioning
    assert "ClientHub" not in provisioning
    assert "AppSettings" not in provisioning
    assert "flet" not in provisioning
