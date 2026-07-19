from __future__ import annotations

import ast
from pathlib import Path

from puripuly_heart.app.ports import gpu_worker as compatibility_gpu_worker
from puripuly_heart.core import gpu_worker as owning_gpu_worker

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src" / "puripuly_heart"


def _imported_modules(source_file: Path) -> set[str]:
    tree = ast.parse(source_file.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def _call_names(source_file: Path) -> list[str]:
    tree = ast.parse(source_file.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    return names


def _defined_function_names(source_file: Path) -> set[str]:
    tree = ast.parse(source_file.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_gpu_worker_contract_has_core_owner_and_app_compatibility_reexport() -> None:
    exported = (
        "GpuWorkerActivation",
        "GpuWorkerClientPort",
        "GpuWorkerClosedError",
        "GpuWorkerDevice",
        "GpuWorkerError",
        "GpuWorkerEvent",
        "GpuWorkerMode",
        "GpuWorkerProcessFactoryPort",
        "GpuWorkerRequestError",
        "GpuWorkerTranscription",
    )

    for name in exported:
        assert getattr(compatibility_gpu_worker, name) is getattr(owning_gpu_worker, name)

    gpu_runtime_imports = _imported_modules(SOURCE_ROOT / "core" / "runtime" / "gpu_asr.py")
    assert "puripuly_heart.core.gpu_worker" in gpu_runtime_imports
    assert "puripuly_heart.app.ports.gpu_worker" not in gpu_runtime_imports


def test_provider_runtime_contract_and_owner_do_not_import_ui_hub_or_app_layers() -> None:
    contract_file = SOURCE_ROOT / "core" / "local_asr_provider_runtime.py"
    owner_file = SOURCE_ROOT / "core" / "runtime" / "local_asr_provider_runtime.py"

    for source_file in (contract_file, owner_file):
        imports = _imported_modules(source_file)
        assert not any(module.startswith("puripuly_heart.ui") for module in imports)
        assert not any(module.startswith("puripuly_heart.app") for module in imports)
        assert "puripuly_heart.core.orchestrator.hub" not in imports

    owner_calls = _call_names(owner_file)
    assert owner_calls.count("ProviderRuntimeHandle") == 2
    assert "ClientHub" not in owner_calls


def test_contract_owner_consumes_provisioning_only_through_public_port() -> None:
    owner_file = SOURCE_ROOT / "core" / "runtime" / "local_asr_provider_runtime.py"
    source = owner_file.read_text(encoding="utf-8")
    imports = _imported_modules(owner_file)

    assert "puripuly_heart.core.local_asr_provisioning" in imports
    assert "LocalASRProvisioningPort" in source
    assert ".inspect_gpu(" in source
    assert "LocalASRProvisioningOwner" not in source
    assert "_models" not in source
    assert "_activities" not in source
    assert "inspect_local_gpu_install" not in source


def test_controller_has_no_provider_construction_or_gpu_runtime_lifecycle_path() -> None:
    source = (SOURCE_ROOT / "ui" / "controller.py").read_text(encoding="utf-8")

    for retired_name in (
        "_gpu_asr_runtime",
        "_gpu_discovery_task",
        "_get_gpu_asr_runtime",
        "_gpu_runtime_for_provider",
        "_create_self_stt_provider_for_settings",
        "_create_peer_stt_provider_from_runtime_config",
        "_create_shared_gpu_asr_runtime",
        "SharedGpuASRRuntime",
        "create_stt_backend",
        "create_peer_stt_backend_from_resolved_config",
    ):
        assert retired_name not in source
    assert "ManagedSTTProvider(" not in source


def test_hub_has_one_local_asr_owner_port_and_no_concrete_stt_handle_lifecycle() -> None:
    hub_file = SOURCE_ROOT / "core" / "orchestrator" / "hub.py"
    source = hub_file.read_text(encoding="utf-8")
    function_names = _defined_function_names(hub_file)

    assert source.count("_local_asr_provider_runtime") > 0
    for retired_name in (
        "_self_stt_provider_runtime",
        "_peer_stt_provider_runtime",
        "replace_stt_provider",
        "replace_peer_stt_provider",
        "handoff_stt_provider",
        "handoff_peer_stt_provider",
        "cancel_stt_provider_handoff",
        "cancel_peer_stt_provider_handoff",
    ):
        assert retired_name not in function_names


def test_peer_runtime_retains_capture_policy_without_concrete_provider_lifecycle() -> None:
    source = (SOURCE_ROOT / "core" / "runtime" / "peer_channel.py").read_text(encoding="utf-8")

    for retired_name in (
        "_stt_factory",
        "_retained_stt",
        "_retired_peer_providers",
        "ManagedSTTProvider",
        "ProviderRuntimeHandle",
        "create_peer_stt_backend_from_resolved_config",
    ):
        assert retired_name not in source
    assert "provider_request_factory" in source
    assert "apply_policy" in source
