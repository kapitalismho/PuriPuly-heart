from __future__ import annotations

import ast
from pathlib import Path

from tests.helpers.ast_sources import imported_modules as _imported_modules
from tests.helpers.paths import SOURCE_ROOT


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


def test_native_asr_contract_and_consumers_do_not_depend_on_app_adapters() -> None:
    for relative_path in (
        "core/asr_worker.py",
        "core/runtime/gpu_asr.py",
        "providers/stt/local_cpp.py",
    ):
        imports = _imported_modules(SOURCE_ROOT / relative_path)
        assert not any(module.startswith("puripuly_heart.app") for module in imports)
        assert not any(module.startswith("puripuly_heart.ui") for module in imports)


def test_provider_runtime_contract_and_owner_do_not_import_ui_or_app_layers() -> None:
    contract_file = SOURCE_ROOT / "core" / "local_asr" / "local_asr_provider_runtime.py"
    owner_file = SOURCE_ROOT / "core" / "runtime" / "local_asr_provider_runtime.py"

    for source_file in (contract_file, owner_file):
        imports = _imported_modules(source_file)
        assert not any(module.startswith("puripuly_heart.ui") for module in imports)
        assert not any(module.startswith("puripuly_heart.app") for module in imports)


def test_shipped_source_constructs_shared_gpu_runtime_only_in_canonical_factory() -> None:
    constructor_calls = {}
    for source_file in SOURCE_ROOT.rglob("*.py"):
        count = _call_names(source_file).count("SharedGpuASRRuntime")
        if count:
            constructor_calls[source_file.relative_to(SOURCE_ROOT).as_posix()] = count

    assert constructor_calls == {"app/wiring/wiring_local_asr_provider_runtime.py": 1}
