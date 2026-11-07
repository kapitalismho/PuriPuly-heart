from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from puripuly_heart.release_evidence.asr_worker_distribution import (
    ASR_WORKER_INVENTORY,
    validate_asr_worker_runtime,
)


@pytest.fixture
def gpu_payload(tmp_path: Path) -> Path:
    names = (
        "PuriPulyHeartASRWorker.exe",
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
    (tmp_path / ASR_WORKER_INVENTORY).write_text(
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


def test_complete_payload_preserves_current_cpu_family(gpu_payload: Path) -> None:
    report = validate_asr_worker_runtime(gpu_payload)
    assert report["cpu_modules"] == [
        "ggml-cpu-haswell.dll",
        "ggml-cpu-icelake.dll",
        "ggml-cpu-x64.dll",
    ]


@pytest.mark.parametrize(
    "name",
    [
        "PuriPulyHeartASRWorker.exe",
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
        validate_asr_worker_runtime(gpu_payload)


@pytest.mark.parametrize("name", ["transcribe.dll", "ggml-vulkan.dll", "ggml-cpu-x64.dll"])
def test_required_family_cannot_be_removed_from_inventory(gpu_payload: Path, name: str) -> None:
    path = gpu_payload / ASR_WORKER_INVENTORY
    record = json.loads(path.read_text())
    record["files"] = [entry for entry in record["files"] if entry["path"] != name]
    path.write_text(json.dumps(record))
    (gpu_payload / name).unlink()
    with pytest.raises(ValueError, match="lacks required files"):
        validate_asr_worker_runtime(gpu_payload)


@pytest.mark.parametrize("name", ["PuriPulyHeartASRWorker.exe", "ggml-cpu-haswell.dll"])
def test_mixed_build_payload_is_rejected(gpu_payload: Path, name: str) -> None:
    (gpu_payload / name).write_bytes(b"different build")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_asr_worker_runtime(gpu_payload)


def test_stale_unlisted_variant_is_rejected(gpu_payload: Path) -> None:
    (gpu_payload / "ggml-cpu-skylakex.dll").write_bytes(b"stale build")
    with pytest.raises(ValueError, match="unexpected=.*skylakex"):
        validate_asr_worker_runtime(gpu_payload)


def test_missing_inventory_is_rejected(gpu_payload: Path) -> None:
    (gpu_payload / ASR_WORKER_INVENTORY).unlink()
    with pytest.raises(FileNotFoundError):
        validate_asr_worker_runtime(gpu_payload)
