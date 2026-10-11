from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import struct
import threading
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

SOURCE_REVISION = "b809bbb467fb2da2be0042fff5ef1f503828c9cf"
MODEL_REVISION = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"
CHECKPOINT_SHA256 = "867c53f552998f772e5b5e5c082962ae85ee7ca5669c2bc17d7f615133d4e96d"
PRECONVERTED_REPOSITORY = "Bonenk/Nemotron-3-Diarization-GGUF"
PRECONVERTED_REVISION = "25cf0d6b0c91b679376df0f6ef3087b4fd96af1f"
PRECONVERTED_FILENAME = "Nemotron-3-Diarization.f16.gguf"
PRECONVERTED_SHA256 = "9b277e1dcec60b2c771a1dcb45dc8e36f7285ff7c1619d880d3926a373036b38"
PRECONVERTED_SIZE = 201832928
LINEAR_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"^encoder\.layers\.\d+\.self_attn\.linear_(q|k|v|pos|out)\.weight$",
        r"^encoder\.layers\.\d+\.feed_forward[12]\.linear[12]\.weight$",
        r"^encoder\.pre_encode\.out\.weight$",
        r"^encoder\.layers\.\d+\.conv\.pointwise_conv[12]\.weight$",
        r"^encoder_proj\.weight$",
        r"^encoder\.pre_encode\.proj\.weight$",
        r"^encoder\.layers\.\d+\.attn\.(w_qkv|out_proj)\.weight$",
        r"^encoder\.layers\.\d+\.ffn\.net\.(0|3)\.weight$",
        r"^transformer\.layers\.\d+\.(first_sub_layer\.(query_net|key_net|value_net|out_projection)|second_sub_layer\.dense_(in|out))\.weight$",
        r"^(encoder\.layers\.\d+\.conv\.depthwise_conv|encoder\.pre_encode\.conv\.\d+|subpixel_upsample)\.weight$",
    )
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_exact(stream, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ValueError("truncated GGUF")
    return value


def _string(stream) -> str:
    size = struct.unpack("<Q", _read_exact(stream, 8))[0]
    if size > 1024 * 1024:
        raise ValueError("oversized GGUF string")
    return _read_exact(stream, size).decode("utf-8")


def _value(stream, kind: int):
    formats = {
        0: "B",
        1: "b",
        2: "H",
        3: "h",
        4: "I",
        5: "i",
        6: "f",
        7: "?",
        10: "Q",
        11: "q",
        12: "d",
    }
    if kind == 8:
        return _string(stream)
    if kind == 9:
        child, count = struct.unpack("<IQ", _read_exact(stream, 12))
        if count > 100000 or child == 9:
            raise ValueError("oversized/nested GGUF metadata array")
        return tuple(_value(stream, child) for _ in range(count))
    if kind not in formats:
        raise ValueError("unsupported GGUF metadata type")
    fmt = "<" + formats[kind]
    return struct.unpack(fmt, _read_exact(stream, struct.calcsize(fmt)))[0]


def _parse(path: Path):
    with path.open("rb") as stream:
        magic, version, n_tensors, n_fields = struct.unpack("<4sIQQ", _read_exact(stream, 24))
        if magic != b"GGUF" or version != 3 or not 1 <= n_tensors <= 4096 or n_fields > 4096:
            raise ValueError("unsupported GGUF header")
        fields = {}
        file_type_position = None
        for _ in range(n_fields):
            key = _string(stream)
            kind = struct.unpack("<I", _read_exact(stream, 4))[0]
            if key in fields:
                raise ValueError("duplicate GGUF metadata")
            if key == "general.file_type":
                if kind != 4:
                    raise ValueError("invalid file type metadata")
                file_type_position = stream.tell()
            fields[key] = _value(stream, kind)
        metadata_end = stream.tell()
        tensors = []
        names = set()
        for _ in range(n_tensors):
            name = _string(stream)
            ndim = struct.unpack("<I", _read_exact(stream, 4))[0]
            if name in names or not 1 <= ndim <= 4:
                raise ValueError("invalid GGUF tensor name/dimensions")
            names.add(name)
            dimensions = struct.unpack("<" + "Q" * ndim, _read_exact(stream, 8 * ndim))
            kind, offset = struct.unpack("<IQ", _read_exact(stream, 12))
            if kind not in (0, 1) or any(d == 0 for d in dimensions):
                raise ValueError("only nonquantized F32/F16 tensors are permitted")
            size = math.prod(dimensions) * (4 if kind == 0 else 2)
            tensors.append((name, dimensions, kind, offset, size))
        alignment = fields.get("general.alignment", 32)
        if (
            not isinstance(alignment, int)
            or alignment < 1
            or alignment > 4096
            or alignment & (alignment - 1)
        ):
            raise ValueError("invalid GGUF alignment")
        data_start = (stream.tell() + alignment - 1) // alignment * alignment
        total = path.stat().st_size
        ranges = sorted((offset, offset + size) for _, _, _, offset, size in tensors)
        if any(data_start + end > total for _, end in ranges) or any(
            a[1] > b[0] for a, b in zip(ranges, ranges[1:])
        ):
            raise ValueError("invalid GGUF tensor storage")
        if (
            fields.get("general.architecture") != "sortformer"
            or fields.get("sortformer.version") != "v3"
        ):
            raise ValueError("requires Nemotron V3 Sortformer")
        if (
            fields.get("sortformer.num_speakers") != 8
            or fields.get("sortformer.preprocessor.sample_rate") != 16000
        ):
            raise ValueError("requires eight-channel 16kHz artifact")
        if (
            fields.get("sortformer.high_resolution") is not True
            or fields.get("sortformer.output_subsampling_factor") != 1
        ):
            raise ValueError("requires V3 10ms output cadence")
        shapes = {name: dimensions for name, dimensions, *_ in tensors}
        head = shapes.get("head.single_hidden_to_spks.weight", ())
        if len(head) != 2 or head[1] != 8 or shapes.get("head.single_hidden_to_spks.bias") != (8,):
            raise ValueError("requires actual eight-channel output head tensors")
        if not any(p.match(name) for name, *_ in tensors for p in LINEAR_PATTERNS):
            raise ValueError("artifact has no neural weights")
        return fields, tensors, metadata_end, file_type_position, data_start, alignment


def inspect_artifact(path: Path) -> dict:
    path = Path(path)
    fields, tensors, *_ = _parse(path)
    if fields.get("general.file_type") != 1:
        raise ValueError("requires F16 deployment artifact")
    for name, _, kind, _, _ in tensors:
        auxiliary_projection = (
            name == "encoder.pre_encode.proj.weight" or name == "subpixel_upsample.weight"
        )
        expected = 1 if any(p.match(name) for p in LINEAR_PATTERNS) else 0
        if not auxiliary_projection and kind != expected:
            raise ValueError(f"unexpected tensor precision: {name}")
    return {
        "sha256": sha256_file(path),
        "version": "v3",
        "num_speakers": 8,
        "sample_rate": 16000,
        "file_type": 1,
        "tensor_formats": dict(Counter("F32" if t[2] == 0 else "F16" for t in tensors)),
        "size_bytes": path.stat().st_size,
        "tensor_count": len(tensors),
    }


def verified_preconverted_manifest():
    from puripuly_heart.core.local_asr.local_stt_assets import LocalSTTAssetManifest
    from puripuly_heart.runtime_layout import current_runtime_layout

    path = current_runtime_layout().package_resource(
        "data", "models", "nemotron-3-diarization-f16-vulkan.manifest.json"
    )
    return LocalSTTAssetManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def adopt_preconverted_artifact(path: Path) -> dict:
    path = Path(path)
    metadata = inspect_artifact(path)
    if metadata["sha256"] != PRECONVERTED_SHA256 or metadata["size_bytes"] != PRECONVERTED_SIZE:
        raise ValueError("preconverted artifact differs from pinned Bonenk F16")
    metadata.update(
        {
            "schema_version": 1,
            "source": {
                "repository": PRECONVERTED_REPOSITORY,
                "revision": PRECONVERTED_REVISION,
                "filename": PRECONVERTED_FILENAME,
                "url": f"https://huggingface.co/{PRECONVERTED_REPOSITORY}/resolve/{PRECONVERTED_REVISION}/{PRECONVERTED_FILENAME}",
                "license": "OpenMDW-1.1",
                "base_model": "nvidia/Nemotron-3-Diarization",
            },
            "provenance": {
                "publisher_statement": "conversion of NVIDIA Nemotron-3-Diarization into NeMo Sortformer V3 layout",
                "verified": "pinned publisher artifact SHA256/geometry/precision; loader and GPU qualification recorded separately",
                "limitation": "publisher conversion recipe and original NVIDIA checkpoint checksum lineage are not independently reproduced",
            },
        }
    )
    path.with_name("model.manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def validate_artifact_provenance(path: Path, metadata: dict | None = None) -> dict:
    path = Path(path)
    metadata = inspect_artifact(path) if metadata is None else metadata
    installed = json.loads(path.with_name("model.manifest.json").read_text(encoding="utf-8"))
    if installed.get("sha256") != metadata["sha256"]:
        raise ValueError("model provenance hash mismatch")
    source = installed.get("source", {})
    community = (
        metadata["sha256"] == PRECONVERTED_SHA256
        and source.get("repository") == PRECONVERTED_REPOSITORY
        and source.get("revision") == PRECONVERTED_REVISION
    )
    official = (
        source.get("checkpoint_sha256") == CHECKPOINT_SHA256
        and source.get("revision") == MODEL_REVISION
        and installed.get("conversion", {}).get("runtime_revision") == SOURCE_REVISION
        and installed.get("conversion", {}).get("original_checkpoint_lineage")
        == "verified exact pinned checkpoint"
    )
    if not community and not official:
        raise ValueError(
            "artifact has no approved pinned model source provenance; cached conversion is probe-only"
        )
    return installed


def _import_local_artifact(source: Path, destination: Path, cancel_event: threading.Event) -> dict:
    fields, *_ = _parse(source)
    if fields.get("general.file_type") != 1:
        return convert_artifact(source, destination, cancel_event=cancel_event)
    metadata = inspect_artifact(source)
    if metadata["sha256"] == PRECONVERTED_SHA256:
        provenance = None
    else:
        provenance = validate_artifact_provenance(source, metadata)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".importing")
    try:
        with source.open("rb") as src, temporary.open("wb") as dst:
            while block := src.read(1048576):
                if cancel_event.is_set():
                    raise InterruptedError("artifact import cancelled")
                dst.write(block)
        if cancel_event.is_set():
            raise InterruptedError("artifact import cancelled")
        if sha256_file(temporary) != metadata["sha256"]:
            raise ValueError("artifact changed during import")
        temporary.replace(destination)
        if provenance is None:
            return adopt_preconverted_artifact(destination)
        destination.with_name("model.manifest.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        return provenance
    finally:
        temporary.unlink(missing_ok=True)


def convert_artifact(
    source_path: Path, output_path: Path, *, cancel_event: threading.Event | None = None
) -> dict:
    source_path, output_path = Path(source_path), Path(output_path)
    if source_path.resolve() == output_path.resolve():
        raise ValueError("conversion destination must differ from source")
    fields, tensors, metadata_end, type_position, data_start, alignment = _parse(source_path)
    if fields.get("general.file_type") not in (0, 1) or type_position is None:
        raise ValueError("requires nonquantized GGUF source")
    source_hash = sha256_file(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(output_path.suffix + ".converting")
    offsets, cursor = [], 0
    for name, dimensions, _, _, _ in tensors:
        target = 1 if any(p.match(name) for p in LINEAR_PATTERNS) else 0
        offsets.append((cursor, target))
        cursor += math.prod(dimensions) * (2 if target == 1 else 4)
        cursor = (cursor + alignment - 1) // alignment * alignment
    try:
        with source_path.open("rb") as src, temp.open("wb") as out:
            metadata = bytearray(_read_exact(src, metadata_end))
            struct.pack_into("<I", metadata, type_position, 1)
            out.write(metadata)
            for (name, dimensions, _, _, _), (offset, target) in zip(tensors, offsets):
                encoded = name.encode("utf-8")
                out.write(struct.pack("<Q", len(encoded)) + encoded)
                out.write(struct.pack("<I", len(dimensions)))
                out.write(struct.pack("<" + "Q" * len(dimensions), *dimensions))
                out.write(struct.pack("<IQ", target, offset))
            dest_data = (out.tell() + alignment - 1) // alignment * alignment
            out.write(bytes(dest_data - out.tell()))
            for (name, dimensions, kind, offset, size), (target_offset, target) in zip(
                tensors, offsets
            ):
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("artifact conversion cancelled")
                out.write(bytes(dest_data + target_offset - out.tell()))
                src.seek(data_start + offset)
                remaining = math.prod(dimensions)
                while remaining:
                    if cancel_event is not None and cancel_event.is_set():
                        raise InterruptedError("artifact conversion cancelled")
                    count = min(remaining, 262144)
                    values = np.frombuffer(
                        _read_exact(src, count * (4 if kind == 0 else 2)),
                        dtype="<f4" if kind == 0 else "<f2",
                    )
                    if not np.isfinite(values).all():
                        raise ValueError(f"nonfinite tensor: {name}")
                    cast = values.astype("<f2" if target == 1 else "<f4", copy=False)
                    if not np.isfinite(cast).all():
                        raise ValueError(f"tensor overflows FP16: {name}")
                    out.write(cast.tobytes())
                    remaining -= count
        metadata = inspect_artifact(temp)
        metadata.update(
            {
                "schema_version": 1,
                "conversion": {
                    "recipe": "pinned-diarization-pick-dtype-gguf-recast",
                    "runtime_revision": SOURCE_REVISION,
                    "tool_sha256": sha256_file(Path(__file__)),
                    "numpy_version": np.__version__,
                    "source_sha256": source_hash,
                    "source_file_type": fields["general.file_type"],
                    "original_checkpoint_lineage": "unverified",
                },
                "official_checkpoint": {
                    "revision": MODEL_REVISION,
                    "sha256": CHECKPOINT_SHA256,
                    "url": f"https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/{MODEL_REVISION}/Nemotron-3-Diarization.nemo",
                    "license": "OpenMDW-1.1",
                    "source_relationship": "not established by cached GGUF conversion",
                },
            }
        )
        temp.replace(output_path)
        output_path.with_name("model.manifest.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        return metadata
    finally:
        temp.unlink(missing_ok=True)


async def acquire_artifact(
    destination: Path,
    *,
    download_consent: bool,
    cancel_event: threading.Event,
    on_status: Callable[[str], None] | None = None,
    source_path: Path | None = None,
) -> dict:
    if source_path is None:
        raise RuntimeError(
            "explicit local source_path required; product owns consent-gated pinned download"
        )
    if on_status is not None:
        on_status("importing_local_artifact")
    task = asyncio.create_task(
        asyncio.to_thread(
            _import_local_artifact, Path(source_path), Path(destination), cancel_event
        )
    )
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        cancel_event.set()
        await asyncio.gather(task, return_exceptions=True)
        raise


def validate_runtime(runtime_dir: Path) -> dict:
    runtime_dir = Path(runtime_dir)
    manifest = json.loads((runtime_dir / "runtime-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("abi") != "puripuly-nemotron-1":
        raise ValueError("unsupported Nemotron runtime manifest")
    if manifest.get("source", {}).get("revision") != SOURCE_REVISION:
        raise ValueError("unexpected Nemotron source revision")
    from puripuly_heart.runtime_layout import current_runtime_layout

    source_path = current_runtime_layout().package_resource(
        "data", "models", "nemotron-runtime.sources.json"
    )
    pins = json.loads(source_path.read_text(encoding="utf-8"))
    dependencies = {name: pins["archives"][name]["revision"] for name in ("llama.cpp", "vcpkg")}
    if manifest.get("dependencies") != dependencies or manifest.get("patches") != pins["patches"]:
        raise ValueError("runtime dependency/patch provenance mismatch")
    if manifest.get("policy") != {
        "backend": "Vulkan",
        "neural_cpu_fallback": False,
        "precision": "F16",
        "schedule_guard": True,
        "physical_gpu_only": True,
    }:
        raise ValueError("runtime lacks strict Vulkan policy")
    files = manifest.get("files", {})
    required = {
        "bin/nemo_speech_asr_c.dll",
        "bin/nemo_speech_asr.dll",
        "bin/ggml-vulkan.dll",
        "bin/ggml.dll",
        "bin/ggml-base.dll",
        "bin/ggml-cpu.dll",
    }
    if not required.issubset(files):
        raise ValueError("runtime inventory incomplete")
    actual = {
        p.relative_to(runtime_dir).as_posix()
        for p in runtime_dir.rglob("*")
        if p.is_file() and p.name != "runtime-manifest.json"
    }
    if actual != set(files):
        raise ValueError("runtime inventory mismatch")
    for name, digest in files.items():
        path = runtime_dir / name
        if (
            path.resolve().is_relative_to(runtime_dir.resolve()) is False
            or sha256_file(path) != digest
        ):
            raise ValueError(f"runtime hash mismatch: {name}")
    return manifest
