from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

from puripuly_heart.release_evidence.native_installer_cleanup import normalize_windows_relative

MANIFEST = "runtime-manifest.json"
SOURCE_REVISION = "b809bbb467fb2da2be0042fff5ef1f503828c9cf"


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _relative(value: str) -> PurePosixPath:
    normalized = normalize_windows_relative(value)
    if "\\" in value:
        raise ValueError(f"unsafe Nemotron artifact path: {value!r}")
    return PurePosixPath(normalized)


def validate_runtime(root: Path, *, required: bool = False) -> dict[str, Any]:
    if not root.exists():
        if required:
            raise FileNotFoundError(f"required Nemotron runtime is missing: {root}")
        return {"bundled": False}
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Nemotron runtime must be an isolated directory")
    manifest_path = root / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != 1 or manifest.get("abi") != "puripuly-nemotron-1":
        raise ValueError("incompatible Nemotron runtime ABI/schema")
    if manifest.get("source") != {
        "repository": "NVIDIA/NeMo-Speech.cpp",
        "revision": SOURCE_REVISION,
    }:
        raise ValueError("Nemotron runtime source revision is not pinned")
    policy = manifest.get("policy", {})
    for key, expected in {
        "backend": "Vulkan",
        "neural_cpu_fallback": False,
        "precision": "F16",
        "schedule_guard": True,
        "physical_gpu_only": True,
    }.items():
        if policy.get(key) != expected or type(policy.get(key)) is not type(expected):
            raise ValueError(f"incompatible Nemotron runtime policy: {key}")
    for field, width in (("dependencies", 40), ("patches", 64)):
        values = manifest.get(field)
        if not isinstance(values, dict) or not values:
            raise ValueError(f"Nemotron runtime lacks {field} provenance")
        for name, digest in values.items():
            _relative(name)
            if not isinstance(digest, str) or not re.fullmatch(f"[0-9a-f]{{{width}}}", digest):
                raise ValueError(f"invalid Nemotron {field} digest: {name}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Nemotron runtime lacks exact-build file inventory")
    expected_files = {}
    for name, digest in files.items():
        relative = _relative(name)
        if (
            name == MANIFEST
            or name.casefold() in expected_files
            or not isinstance(digest, str)
            or not re.fullmatch("[0-9a-f]{64}", digest)
        ):
            raise ValueError("invalid Nemotron runtime inventory entry")
        if relative.suffix.lower() in {".gguf", ".nemo", ".safetensors", ".pt", ".onnx"}:
            raise ValueError("model weights must not be bundled with Nemotron runtime")
        if relative.suffix.lower() in {".dll", ".exe", ".pyd"} and relative.parent != PurePosixPath(
            "bin"
        ):
            raise ValueError("Nemotron native binaries must remain in private bin directory")
        expected_files[name.casefold()] = (name, digest)
    required_files = {
        *(
            f"bin/{name}.dll"
            for name in (
                "nemo_speech_asr_c",
                "nemo_speech_asr",
                "ggml",
                "ggml-base",
                "ggml-cpu",
                "ggml-vulkan",
            )
        ),
        *(
            f"share/licenses/nemo-speech/{name}"
            for name in (
                "license",
                "notice",
                "third_party_notices.md",
                "third_party/ggml/license",
            )
        ),
        *(
            f"share/licenses/nemo-speech/third_party/{name}/copyright"
            for name in ("abseil", "protobuf", "sentencepiece", "utf8-range")
        ),
    }
    missing = required_files - expected_files.keys()
    if missing:
        raise ValueError(f"Nemotron runtime lacks required files: {sorted(missing)}")
    if manifest["dependencies"].keys() != {"llama.cpp", "vcpkg"}:
        raise ValueError("Nemotron runtime lacks pinned dependency revisions")
    actual = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Nemotron artifact cannot contain symlinks: {path}")
        if path.is_file() and path != manifest_path:
            name = path.relative_to(root).as_posix().casefold()
            if name in actual:
                raise ValueError("case-ambiguous Nemotron artifact")
            actual[name] = path
    if actual.keys() != expected_files.keys():
        raise ValueError(
            f"Nemotron inventory mismatch: missing={sorted(expected_files.keys() - actual.keys())}, "
            f"unexpected={sorted(actual.keys() - expected_files.keys())}"
        )
    for name, (_, digest) in expected_files.items():
        if _digest(actual[name]) != digest:
            raise ValueError(f"Nemotron runtime hash mismatch: {name}")
    return {
        "bundled": True,
        "manifest_sha256": _digest(manifest_path),
        "source": manifest["source"],
        "dependencies": manifest["dependencies"],
        "patches": manifest["patches"],
        "policy": policy,
        "files": files,
    }


def stage_runtime(source: Path, destination: Path) -> dict[str, Any]:
    evidence = validate_runtime(source, required=True)
    if destination.exists():
        raise FileExistsError(
            f"refusing to merge Nemotron runtime into existing directory: {destination}"
        )
    shutil.copytree(source, destination)
    if validate_runtime(destination, required=True) != evidence:
        raise ValueError("Nemotron runtime changed while staging")
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate or stage isolated, model-free Nemotron runtime"
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--required", action="store_true")
    parser.add_argument("--stage-to", type=Path)
    parser.add_argument("--expected-runtime", type=Path)
    args = parser.parse_args(argv)
    result = (
        stage_runtime(args.root, args.stage_to)
        if args.stage_to
        else validate_runtime(args.root, required=args.required)
    )
    if args.expected_runtime:
        expected = validate_runtime(args.expected_runtime, required=True)
        if result != expected:
            raise ValueError("packaged Nemotron runtime differs from explicitly selected build")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
