from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def extended_path(path):
    absolute = os.path.abspath(path)
    if os.name != "nt" or absolute.startswith("\\\\?\\"):
        return absolute
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


def copy_payload_file(source, destination):
    return shutil.copy2(extended_path(source), extended_path(destination))


def package_copy_ignore(source_packages):
    cache_filter = shutil.ignore_patterns("__pycache__", "*.pyc")
    root_filter = shutil.ignore_patterns(
        "*.pth",
        "*.egg-link",
        "cmake",
        "clang",
        "ruff",
        "pytest*",
        "_pytest",
        "black*",
        "blib2to3",
        "coverage*",
        "virtualenv*",
        "libclang*",
        "pip*",
    )

    def ignore(directory, names):
        ignored = cache_filter(directory, names)
        if Path(directory) == source_packages:
            ignored.update(root_filter(directory, names))
        return ignored

    return ignore


def validate_beta_inputs(repo):
    from puripuly_heart.release_evidence.gpu_worker_distribution import validate_gpu_worker_runtime

    inventory = validate_gpu_worker_runtime(repo / "build" / "gpu_worker")
    if not (repo / "build" / "overlay" / "PuriPulyHeartOverlay.exe").is_file():
        raise SystemExit("Beta requires a compiled overlay")
    from puripuly_heart.core.overlay.openvr_vendor import validate_vendored_openvr_bundle

    validate_vendored_openvr_bundle(repo / "third_party" / "openvr")
    return inventory


def stage_beta_openvr_runtime(repo, output):
    from puripuly_heart.core.overlay.openvr_vendor import (
        validate_openvr_runtime_dll,
        validate_vendored_openvr_bundle,
    )

    bundle = validate_vendored_openvr_bundle(repo / "third_party" / "openvr")
    destination = output / "app" / "build" / "overlay" / "openvr_api.dll"
    shutil.copy2(bundle.dll_path, destination)
    validate_openvr_runtime_dll(destination)
    return bundle.dll_sha256


def stage_beta_vc_runtime(output, source):
    required = ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")
    if source is None or not all((source / name).is_file() for name in required):
        raise SystemExit("Beta requires the x64 Visual C++ redistributable CRT directory")
    destinations = [
        output / "runtime",
        output / "flet" / "flet",
        output / "app" / "build" / "gpu_worker",
        output / "app" / "build" / "overlay",
    ]
    destinations.extend((output / "app" / "build" / "llama.cpp").glob("*/cpu"))
    destinations.extend((output / "app" / "build" / "llama.cpp").glob("*/vulkan"))
    hashes = {}
    for dll in source.glob("*.dll"):
        with dll.open("rb") as stream:
            hashes[dll.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        for destination in destinations:
            if not destination.is_dir():
                raise SystemExit("Missing Beta runtime directory")
            shutil.copy2(dll, destination / dll.name)
    return hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beta", action="store_true")
    parser.add_argument("--vc-runtime", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output.exists():
        raise SystemExit("Output directory already exists; choose a fresh versioned directory")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True
    ).strip():
        raise SystemExit("Commit tracked application changes before building an identified preview")
    flet_archive = repo / "build" / "flet" / "flet-windows.zip"
    with flet_archive.open("rb") as reader:
        if (
            hashlib.file_digest(reader, "sha256").hexdigest()
            != "758f21506fbb9ad180bd93c7460a2ca55630401c6026a9bc9e2273444014491d"
        ):
            raise SystemExit("Pinned Flet archive checksum mismatch")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=repo, text=True
    ).strip()
    gpu_inventory = None
    if args.beta:
        gpu_inventory = validate_beta_inputs(repo)
        if args.vc_runtime is None:
            raise SystemExit("Beta requires --vc-runtime")
    output.mkdir(parents=True)
    shutil.copy2(
        repo / "scripts" / "preview" / ("README-BETA.txt" if args.beta else "README.md"),
        output / ("README-BETA.txt" if args.beta else "README.md"),
    )
    shutil.copy2(repo / "docs" / "desktop-feature-pr-notes.md", output / "FEATURES.md")
    runtime = output / "runtime"
    shutil.copytree(
        Path(sys.base_prefix),
        runtime,
        copy_function=copy_payload_file,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "test",
            "tests",
            "ensurepip",
            "tkinter",
            "tcl",
            "idlelib",
            "pip*",
            "Scripts",
        ),
    )
    source_packages = repo / ".venv" / "Lib" / "site-packages"
    shutil.copytree(
        source_packages,
        runtime / "Lib" / "site-packages",
        dirs_exist_ok=True,
        copy_function=copy_payload_file,
        ignore=package_copy_ignore(source_packages),
    )
    app = output / "app"
    app.mkdir()
    shutil.copytree(repo / "prompts", app / "prompts", copy_function=copy_payload_file)
    shutil.copytree(
        repo / "src",
        app / "src",
        copy_function=copy_payload_file,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
    )
    for name in ("pyproject.toml", "LICENSE", "uv.lock"):
        shutil.copy2(repo / name, app / name)
    shutil.copy2(repo / "scripts" / "preview" / "entry.py", app / "preview_entry.py")
    shutil.copytree(
        repo / "build" / "llama.cpp", app / "build" / "llama.cpp", copy_function=copy_payload_file
    )
    shutil.copytree(
        repo / "third_party" / "llama.cpp",
        app / "third_party" / "llama.cpp",
        copy_function=copy_payload_file,
    )
    if args.beta:
        for component in ("gpu_worker", "overlay"):
            shutil.copytree(
                repo / "build" / component,
                app / "build" / component,
                copy_function=copy_payload_file,
            )
        openvr_sha256 = stage_beta_openvr_runtime(repo, output)
        subprocess.run(
            ["git", "archive", "--format=zip", f"--output={output / 'SOURCE.zip'}", revision],
            cwd=repo,
            check=True,
        )
    for name in ("openvr", "noto-sans-cjk", "soxr"):
        source = repo / "third_party" / name
        if source.is_dir():
            shutil.copytree(source, app / "third_party" / name, copy_function=copy_payload_file)
    shutil.unpack_archive(flet_archive, output / "flet")
    vc_runtime = stage_beta_vc_runtime(output, args.vc_runtime) if args.beta else None
    compiler = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")
    smoke_environment = dict(os.environ)
    smoke_environment["PYTHONPATH"] = str(app / "src")
    smoke_environment.pop("PURIPULY_HEART_PROMPTS_DIR", None)
    subprocess.run(
        [
            str(compiler),
            "/nologo",
            "/target:winexe",
            "/reference:System.Windows.Forms.dll",
            *(["/define:BETA_DISTRIBUTION"] if args.beta else []),
            f"/out:{output / 'PuriPulyHeartPreview.exe'}",
            str(repo / "scripts" / "preview" / "Launcher.cs"),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(runtime / "python.exe"),
            "-c",
            "from flet.app import run; import numpy, sounddevice, soxr, huggingface_hub; from puripuly_heart.config.prompts import load_prompt; assert load_prompt('translation_prompt'); from puripuly_heart.ui.i18n import available_locales; assert 'zh-CN' in available_locales(); print('Bundled runtime GUI dependency/resources smoke: PASS')",
        ],
        cwd=app,
        env=smoke_environment,
        check=True,
    )
    (output / "BUILD.json").write_text(
        json.dumps(
            dict(
                branch=branch,
                commit=revision,
                qq_baseline="677a9ee837b7e19abfa758998e86f8bce6a28b3a",
                upstream_baseline="75c4485a6787d504f5335d1602973f63b4d0c234",
                version="2.8.2-beta" if args.beta else "2.8.2-preview",
                packaging="portable bundled CPython + locked Flet desktop (not native release)",
                python=sys.version,
                isolated_keyring=(
                    "puripuly-heart.test.desktop-beta"
                    if args.beta
                    else "puripuly-heart.test.desktop-preview"
                ),
                data_directory="%LOCALAPPDATA%/PuriPulyHeartBeta" if args.beta else "preview-data",
                gpu_worker=gpu_inventory,
                vc_runtime=vc_runtime,
                openvr_sha256=openvr_sha256 if args.beta else None,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
