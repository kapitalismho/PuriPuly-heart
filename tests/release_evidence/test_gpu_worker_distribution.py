from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from puripuly_heart.release_evidence.gpu_worker_distribution import (
    GPU_WORKER_INVENTORY,
    collect_gpu_worker_runtime_binaries,
    validate_gpu_worker_runtime,
)


@pytest.fixture
def gpu_payload(tmp_path: Path) -> Path:
    names = (
        "PuriPulyHeartGpuWorker.exe",
        "transcribe.dll",
        "ggml.dll",
        "ggml-base.dll",
        "ggml-vulkan.dll",
        "ggml-cpu-x64.dll",
        "ggml-cpu-haswell.dll",
        "ggml-cpu-icelake.dll",
    )
    files = []
    for name in names:
        data = name.encode()
        (tmp_path / name).write_bytes(data)
        files.append({"path": name, "sha256": hashlib.sha256(data).hexdigest()})
    (tmp_path / GPU_WORKER_INVENTORY).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "build_id": "a" * 32,
                "cpu_selection": "upstream-cpuid",
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_complete_payload_is_collected_beside_worker(gpu_payload: Path) -> None:
    report = validate_gpu_worker_runtime(gpu_payload)
    assert report["cpu_modules"] == [
        "ggml-cpu-haswell.dll",
        "ggml-cpu-icelake.dll",
        "ggml-cpu-x64.dll",
    ]
    binaries = collect_gpu_worker_runtime_binaries(gpu_payload)
    assert {Path(path).name for path, _ in binaries} == {
        path.name for path in gpu_payload.iterdir() if path.suffix in {".dll", ".exe"}
    }
    assert {destination for _, destination in binaries} == {"."}


@pytest.mark.parametrize(
    "name",
    [
        "PuriPulyHeartGpuWorker.exe",
        "transcribe.dll",
        "ggml.dll",
        "ggml-base.dll",
        "ggml-vulkan.dll",
        "ggml-cpu-x64.dll",
        "ggml-cpu-icelake.dll",
    ],
)
def test_omitted_current_build_runtime_is_rejected(gpu_payload: Path, name: str) -> None:
    (gpu_payload / name).unlink()
    with pytest.raises(ValueError, match="inventory mismatch"):
        collect_gpu_worker_runtime_binaries(gpu_payload)


@pytest.mark.parametrize("name", ["transcribe.dll", "ggml-vulkan.dll", "ggml-cpu-x64.dll"])
def test_required_family_cannot_be_removed_from_inventory(gpu_payload: Path, name: str) -> None:
    path = gpu_payload / GPU_WORKER_INVENTORY
    record = json.loads(path.read_text())
    record["files"] = [entry for entry in record["files"] if entry["path"] != name]
    path.write_text(json.dumps(record))
    (gpu_payload / name).unlink()
    with pytest.raises(ValueError, match="lacks required files"):
        validate_gpu_worker_runtime(gpu_payload)


@pytest.mark.parametrize("name", ["PuriPulyHeartGpuWorker.exe", "ggml-cpu-haswell.dll"])
def test_mixed_build_payload_is_rejected(gpu_payload: Path, name: str) -> None:
    (gpu_payload / name).write_bytes(b"different build")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_gpu_worker_runtime(gpu_payload)


def test_stale_unlisted_variant_is_rejected(gpu_payload: Path) -> None:
    (gpu_payload / "ggml-cpu-skylakex.dll").write_bytes(b"stale build")
    with pytest.raises(ValueError, match="unexpected=.*skylakex"):
        validate_gpu_worker_runtime(gpu_payload)


def test_missing_inventory_is_rejected(gpu_payload: Path) -> None:
    (gpu_payload / GPU_WORKER_INVENTORY).unlink()
    with pytest.raises(FileNotFoundError):
        collect_gpu_worker_runtime_binaries(gpu_payload)
