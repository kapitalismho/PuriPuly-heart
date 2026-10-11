from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(cache: Path, repo: Path) -> tuple[Path, Path]:
    manifest = json.loads(
        (repo / "src/puripuly_heart/data/models/nemotron-runtime.sources.json").read_text()
    )
    trees = {}
    cache.mkdir(parents=True, exist_ok=True)
    for name, spec in manifest["archives"].items():
        archive = cache / (name.replace(".", "-") + ".zip")
        if not archive.exists():
            with (
                urllib.request.urlopen(spec["url"], timeout=120) as response,
                archive.open("wb") as target,
            ):
                shutil.copyfileobj(response, target, 1048576)
        if sha256(archive) != spec["sha256"]:
            raise ValueError(f"source archive hash mismatch: {name}")
        tree_root = cache / (name.replace(".", "-") + "-code")
        if not tree_root.exists():
            tree_root.mkdir()
            with zipfile.ZipFile(archive) as zipped:
                for entry in zipped.infolist():
                    relative = Path(*Path(entry.filename).parts[1:])
                    target = tree_root / relative
                    if not target.resolve().is_relative_to(tree_root.resolve()):
                        raise ValueError("source archive path traversal")
                    if entry.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with zipped.open(entry) as src, target.open("wb") as out:
                            shutil.copyfileobj(src, out, 1048576)
        trees[name] = tree_root
    source = trees["nemo"]
    if not (source / "llama.cpp/ggml/CMakeLists.txt").exists():
        shutil.copytree(trees["llama.cpp"], source / "llama.cpp", dirs_exist_ok=True)
    patches = json.loads((repo / "native/nemotron/patches.json").read_text())
    for name, patch in patches["files"].items():
        target = source / name
        digest = sha256(target)
        if digest == patch["after_sha256"]:
            continue
        if digest != patch["before_sha256"]:
            raise ValueError(f"unexpected patch source: {name}")
        lines = target.read_bytes().decode("utf-8").splitlines(keepends=True)
        for change in reversed(patch["edits"]):
            start = change["start"]
            old = change["old"].splitlines(keepends=True)
            if lines[start : start + len(old)] != old:
                raise ValueError(f"patch context mismatch: {name}")
            lines[start : start + len(old)] = change["new"].splitlines(keepends=True)
        target.write_bytes("".join(lines).encode("utf-8"))
        if sha256(target) != patch["after_sha256"]:
            raise ValueError(f"patch output mismatch: {name}")
    return source, trees["vcpkg"]


def record(output: Path, repo: Path, build: Path) -> None:
    sources = json.loads(
        (repo / "src/puripuly_heart/data/models/nemotron-runtime.sources.json").read_text()
    )
    files = {
        p.relative_to(output).as_posix(): sha256(p)
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "runtime-manifest.json"
    }
    manifest = {
        "schema_version": 1,
        "abi": "puripuly-nemotron-1",
        "source": {
            "repository": "NVIDIA/NeMo-Speech.cpp",
            "revision": sources["archives"]["nemo"]["revision"],
        },
        "dependencies": {
            name: sources["archives"][name]["revision"] for name in ("llama.cpp", "vcpkg")
        },
        "archives": sources["archives"],
        "patches": {"native/nemotron/patches.json": sha256(repo / "native/nemotron/patches.json")},
        "policy": {
            "backend": "Vulkan",
            "neural_cpu_fallback": False,
            "precision": "F16",
            "schedule_guard": True,
            "physical_gpu_only": True,
        },
        "build": {
            "generator": "Ninja",
            "configuration": "Release",
            "nemo_ggml_patch_set": False,
            "ggml_physical_device_classification_patch": True,
            "cmake_cache_sha256": sha256(build / "CMakeCache.txt"),
            "neural_schedule": "selected Vulkan GPU checked before cached/direct and scheduler compute; CPU frontend permitted",
            "accumulation": "backend-selected Vulkan shader accumulation; F16 weights, F32 auxiliary weights/activations; no all-F16 claim",
        },
        "files": files,
    }
    (output / "runtime-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("prepare", "record"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--build", type=Path)
    args = parser.parse_args()
    if args.operation == "prepare":
        src, vcpkg = prepare(args.cache, args.repo)
        print(json.dumps({"source": str(src), "vcpkg": str(vcpkg)}))
    else:
        record(args.output, args.repo, args.build)
