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
    contract_file = SOURCE_ROOT / "core" / "local_asr" / "local_asr_provider_runtime.py"
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


def test_application_composition_has_no_provider_runtime_lifecycle_algorithm() -> None:
    source = (SOURCE_ROOT / "composition" / "application_runtime.py").read_text(encoding="utf-8")

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
        "_gpu_reconfiguration_lock",
        "_gpu_manual_retry_channels",
        "_retry_gpu_activation_locked",
        "_restore_gpu_channels_after_manual_retry",
        "_apply_coordinated_gpu_restart",
        "_quiesce_shared_gpu_consumers",
    ):
        assert retired_name not in source
    assert "ManagedSTTProvider(" not in source


def test_peer_owner_has_one_local_asr_port_and_no_concrete_stt_lifecycle() -> None:
    owner_file = SOURCE_ROOT / "core" / "orchestrator" / "peer_translation_channel.py"
    source = owner_file.read_text(encoding="utf-8")
    function_names = _defined_function_names(owner_file)

    assert source.count("local_asr_runtime") > 0
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


def test_prebuilt_compatibility_factory_delegates_to_canonical_owner() -> None:
    compatibility_file = SOURCE_ROOT / "core" / "runtime" / "prebuilt_local_asr_provider_runtime.py"
    source = compatibility_file.read_text(encoding="utf-8")
    calls = _call_names(compatibility_file)

    assert "class PrebuiltLocalASRProviderRuntime:" not in source
    assert "ProviderRuntimeHandle" not in source
    assert calls.count("LocalASRProviderRuntimeOwner") == 1
    assert "prebuilt_providers=" in source


def test_shipped_source_constructs_shared_gpu_runtime_only_in_canonical_factory() -> None:
    constructor_calls = {}
    for source_file in SOURCE_ROOT.rglob("*.py"):
        count = _call_names(source_file).count("SharedGpuASRRuntime")
        if count:
            constructor_calls[source_file.relative_to(SOURCE_ROOT).as_posix()] = count

    assert constructor_calls == {"app/wiring_local_asr_provider_runtime.py": 1}
    wiring_source = (SOURCE_ROOT / "app" / "wiring.py").read_text(encoding="utf-8")
    assert "_create_shared_gpu_asr_runtime" not in wiring_source


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
