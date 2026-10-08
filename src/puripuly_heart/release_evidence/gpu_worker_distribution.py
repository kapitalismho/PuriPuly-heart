from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

GPU_WORKER_INVENTORY = "gpu-worker-runtime.json"
_REQUIRED_DLLS = {
    "transcribe.dll",
    "ggml.dll",
    "ggml-base.dll",
    "ggml-vulkan.dll",
    "ggml-cpu-x64.dll",
}


def validate_gpu_worker_runtime(root: Path) -> dict[str, Any]:
    inventory_path = root / GPU_WORKER_INVENTORY
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if (
        inventory.get("schema_version") != 1
        or inventory.get("cpu_selection") != "upstream-cpuid"
        or not isinstance(inventory.get("build_id"), str)
        or len(inventory["build_id"]) != 32
        or any(character not in "0123456789abcdef" for character in inventory["build_id"])
    ):
        raise ValueError("unsupported GPU worker runtime inventory")
    files = inventory.get("files")
    if not isinstance(files, list):
        raise ValueError("GPU worker runtime inventory lacks files")
    expected = {}
    for entry in files:
        name, digest = entry.get("path"), entry.get("sha256")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or "/" in name
            or "\\" in name
            or name.casefold() in expected
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not (name == "PuriPulyHeartGpuWorker.exe" or name.lower().endswith(".dll"))
        ):
            raise ValueError("invalid GPU worker runtime inventory entry")
        expected[name.casefold()] = digest
    required = {"puripulyheartgpuworker.exe", *_REQUIRED_DLLS}
    missing = required - expected.keys()
    if missing:
        raise ValueError(f"GPU worker runtime inventory lacks required files: {sorted(missing)}")
    cpu_modules = sorted(
        name for name in expected if name.startswith("ggml-cpu-") and name.endswith(".dll")
    )
    if len(cpu_modules) < 2:
        raise ValueError("GPU worker runtime lacks optimized CPU modules")
    actual = {
        path.name.casefold(): path
        for path in root.iterdir()
        if path.is_file()
        and (
            path.name.casefold() == "puripulyheartgpuworker.exe"
            or path.name.casefold() == "transcribe.dll"
            or path.name.casefold().startswith("ggml")
            and path.suffix.casefold() == ".dll"
            or path.name.casefold() in expected
        )
    }
    if actual.keys() != expected.keys():
        raise ValueError(
            f"GPU worker runtime inventory mismatch: missing={sorted(expected.keys() - actual.keys())}, "
            f"unexpected={sorted(actual.keys() - expected.keys())}"
        )
    for name, digest in expected.items():
        if hashlib.sha256(actual[name].read_bytes()).hexdigest() != digest:
            raise ValueError(f"GPU worker runtime hash mismatch: {name}")
    return {
        "build_id": inventory["build_id"],
        "cpu_selection": inventory["cpu_selection"],
        "cpu_modules": cpu_modules,
        "files": dict(sorted(expected.items())),
    }


def collect_gpu_worker_runtime_binaries(root: Path) -> list[tuple[str, str]]:
    validate_gpu_worker_runtime(root)
    inventory = json.loads((root / GPU_WORKER_INVENTORY).read_text(encoding="utf-8"))
    return [(str(root / entry["path"]), ".") for entry in inventory["files"]]
