from __future__ import annotations

import json
import struct
import threading
from pathlib import Path

import numpy as np
import pytest

from puripuly_heart.app.adapters.nemotron.artifacts import (
    _import_local_artifact,
    convert_artifact,
    inspect_artifact,
    sha256_file,
    validate_artifact_provenance,
    validate_runtime,
)


def gguf(
    path: Path,
    *,
    version="v3",
    speakers=8,
    file_type=0,
    weight_type=0,
    aux_type=None,
    head_channels=8,
):
    def string(value):
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded

    metadata = {
        "general.architecture": "sortformer",
        "general.file_type": file_type,
        "sortformer.version": version,
        "sortformer.num_speakers": speakers,
        "sortformer.preprocessor.sample_rate": 16000,
        "sortformer.high_resolution": True,
        "sortformer.output_subsampling_factor": 1,
    }
    tensors = [
        ("encoder_proj.weight", np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32), weight_type),
        (
            "head.single_hidden_to_spks.weight",
            np.full((head_channels, 2), 0.1, dtype=np.float32),
            0,
        ),
        ("head.single_hidden_to_spks.bias", np.full(head_channels, 0.1, dtype=np.float32), 0),
        (
            "subpixel_upsample.weight",
            np.array([[[0.1, 0.2]]], dtype=np.float32),
            weight_type if aux_type is None else aux_type,
        ),
    ]
    header = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(metadata)))
    for key, value in metadata.items():
        if isinstance(value, str):
            kind, data = 8, string(value)
        elif isinstance(value, bool):
            kind, data = 7, struct.pack("<?", value)
        else:
            kind, data = 4, struct.pack("<I", value)
        header += string(key) + struct.pack("<I", kind) + data
    payload = bytearray()
    for name, tensor, kind in tensors:
        header += string(name) + struct.pack("<I", tensor.ndim)
        header += struct.pack("<" + "Q" * tensor.ndim, *tensor.shape[::-1])
        header += struct.pack("<IQ", kind, len(payload))
        payload += tensor.astype(np.float32 if kind == 0 else np.float16).tobytes()
        payload += bytes((-len(payload)) % 32)
    header += bytes((-len(header)) % 32)
    path.write_bytes(header + payload)
    return tensors


def test_exact_official_dtype_dispatch_and_checksums(tmp_path):
    source, output = tmp_path / "source.gguf", tmp_path / "model.gguf"
    originals = gguf(source)
    manifest = convert_artifact(source, output)
    assert manifest["sha256"] == sha256_file(output)
    assert manifest["tensor_formats"] == {"F16": 2, "F32": 2}
    assert manifest["conversion"]["source_sha256"] == sha256_file(source)
    assert manifest["conversion"]["original_checkpoint_lineage"] == "unverified"
    assert (
        json.loads(output.with_name("model.manifest.json").read_text())["sha256"]
        == manifest["sha256"]
    )
    from puripuly_heart.app.adapters.nemotron.artifacts import _parse

    _, tensors, _, _, start, _ = _parse(output)
    raw = output.read_bytes()
    for (name, dimensions, kind, offset, size), (_, original, _) in zip(tensors, originals):
        expected = original.astype(np.float16 if kind == 1 else np.float32)
        observed = np.frombuffer(
            raw[start + offset : start + offset + size], dtype="<f2" if kind == 1 else "<f4"
        )
        assert np.array_equal(observed, expected.ravel())
    assert inspect_artifact(output)["num_speakers"] == 8


@pytest.mark.parametrize(
    "changes",
    [
        {"version": "v2"},
        {"speakers": 4},
        {"file_type": 0},
        {"file_type": 1, "weight_type": 0},
        {"file_type": 1, "weight_type": 1, "head_channels": 4},
    ],
)
def test_reject_wrong_artifact(tmp_path, changes):
    path = tmp_path / "model.gguf"
    gguf(path, **changes)
    with pytest.raises(ValueError):
        inspect_artifact(path)


def test_cancellation_never_promotes_artifact(tmp_path):
    source, output = tmp_path / "source.gguf", tmp_path / "model.gguf"
    gguf(source)
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(InterruptedError):
        convert_artifact(source, output, cancel_event=cancelled)
    assert not output.exists()
    assert not output.with_suffix(".gguf.converting").exists()
    assert not output.with_name("model.manifest.json").exists()


def test_runtime_requires_exact_inventory_and_hashes(tmp_path):
    from puripuly_heart.app.adapters.nemotron.artifacts import SOURCE_REVISION

    names = [
        "nemo_speech_asr_c.dll",
        "nemo_speech_asr.dll",
        "ggml-vulkan.dll",
        "ggml.dll",
        "ggml-base.dll",
        "ggml-cpu.dll",
    ]
    (tmp_path / "bin").mkdir()
    files = {}
    for name in names:
        path = tmp_path / "bin" / name
        path.write_bytes(name.encode())
        files["bin/" + name] = sha256_file(path)
    pins = json.loads(
        Path("src/puripuly_heart/data/models/nemotron-runtime.sources.json").read_text()
    )
    manifest = {
        "schema_version": 1,
        "abi": "puripuly-nemotron-1",
        "source": {"revision": SOURCE_REVISION},
        "dependencies": {
            name: pins["archives"][name]["revision"] for name in ("llama.cpp", "vcpkg")
        },
        "patches": pins["patches"],
        "policy": {
            "backend": "Vulkan",
            "neural_cpu_fallback": False,
            "precision": "F16",
            "schedule_guard": True,
            "physical_gpu_only": True,
        },
        "files": files,
    }
    (tmp_path / "runtime-manifest.json").write_text(json.dumps(manifest))
    assert validate_runtime(tmp_path)["abi"] == "puripuly-nemotron-1"
    del manifest["policy"]["physical_gpu_only"]
    (tmp_path / "runtime-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="strict Vulkan policy"):
        validate_runtime(tmp_path)
    manifest["policy"]["physical_gpu_only"] = True
    (tmp_path / "runtime-manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "bin/ggml.dll").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_runtime(tmp_path)
    (tmp_path / "extra.dll").write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory mismatch"):
        validate_runtime(tmp_path)


def test_permitted_auxiliary_f32_is_not_recast_on_approved_import(tmp_path, monkeypatch):
    from puripuly_heart.app.adapters.nemotron import artifacts

    source, destination = tmp_path / "source.gguf", tmp_path / "imported/model.gguf"
    gguf(source, file_type=1, weight_type=1, aux_type=0)
    actual = inspect_artifact(source)
    assert actual["tensor_formats"] == {"F16": 1, "F32": 3}
    monkeypatch.setattr(artifacts, "PRECONVERTED_SHA256", actual["sha256"])
    monkeypatch.setattr(artifacts, "PRECONVERTED_SIZE", actual["size_bytes"])
    result = _import_local_artifact(source, destination, threading.Event())
    assert destination.read_bytes() == source.read_bytes()
    assert validate_artifact_provenance(destination)["sha256"] == result["sha256"]
    assert result["provenance"]["limitation"]


def test_unverified_cached_conversion_cannot_enable_product(tmp_path):
    source, destination = tmp_path / "source.gguf", tmp_path / "model.gguf"
    gguf(source)
    convert_artifact(source, destination)
    with pytest.raises(ValueError, match="probe-only"):
        validate_artifact_provenance(destination)
    with pytest.raises(ValueError, match="probe-only"):
        _import_local_artifact(destination, tmp_path / "imported/model.gguf", threading.Event())


def test_f16_import_cancellation_never_promotes(tmp_path, monkeypatch):
    from puripuly_heart.app.adapters.nemotron import artifacts

    source, destination = tmp_path / "source.gguf", tmp_path / "imported/model.gguf"
    gguf(source, file_type=1, weight_type=1)
    monkeypatch.setattr(artifacts, "PRECONVERTED_SHA256", sha256_file(source))
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(InterruptedError):
        _import_local_artifact(source, destination, cancelled)
    assert not destination.exists()
    assert not destination.with_suffix(".gguf.importing").exists()
