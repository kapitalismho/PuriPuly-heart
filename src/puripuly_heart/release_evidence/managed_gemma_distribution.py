from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Sequence

from puripuly_heart.core.local_translation.runtime_profile import (
    LLAMA_CPP_BUILD,
    LLAMA_CPP_COMMIT,
    LLAMA_CPP_CPU_ARCHIVE,
    LLAMA_CPP_CPU_ARCHIVE_SHA256,
    LLAMA_CPP_CPU_ARCHIVE_SIZE,
    LLAMA_CPP_RUNTIME_DIRNAME,
    LLAMA_CPP_VULKAN_ARCHIVE,
    LLAMA_CPP_VULKAN_ARCHIVE_SHA256,
    LLAMA_CPP_VULKAN_ARCHIVE_SIZE,
)

MANIFEST_SCHEMA = "puripuly-heart/llama-cpp-runtime-distribution/v1"
RELEASE_URL = f"https://github.com/ggml-org/llama.cpp/releases/tag/{LLAMA_CPP_BUILD}"
DOWNLOAD_BASE_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_BUILD}"
PACKAGED_RUNTIME_RELATIVE_DIR = Path("_runtime") / LLAMA_CPP_RUNTIME_DIRNAME
PROVENANCE_RELATIVE_DIR = Path("third_party") / "llama.cpp"
LICENSE_SHA256 = "94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d"
LICENSE_SIZE = 1078
README_SHA256 = "330ed9a36deb19c7bc8cef37b0d471e9fa73b597b6687d3d9a6131fe2c4acf01"
README_SIZE = 495
KNOWN_MODEL_FILENAMES = {
    "gemma-4-e4b-it-qat-ud-q4_k_xl.gguf",
    "mtp-gemma-4-e4b-it.gguf",
    "gemma-4-12b-it-qat-ud-q4_k_xl.gguf",
}
INSTALLER_FORBIDDEN_IDENTITIES = KNOWN_MODEL_FILENAMES | {
    "unsloth/gemma-4-e4b-it-qat-gguf",
    "unsloth/gemma-4-12b-it-qat-gguf",
}
COMMON_REQUIRED_FILES = {
    "ggml-base.dll",
    "ggml-cpu-x64.dll",
    "ggml.dll",
    "libomp140.x86_64.dll",
    "llama-common.dll",
    "llama-server-impl.dll",
    "llama-server.exe",
    "llama.dll",
    "mtmd.dll",
}
FIXED_RUNTIME_FILES = COMMON_REQUIRED_FILES - {"ggml-cpu-x64.dll"}


@dataclass(frozen=True, slots=True)
class ArchiveContract:
    backend: str
    filename: str
    size: int
    sha256: str
    required_files: frozenset[str]

    @property
    def url(self) -> str:
        return f"{DOWNLOAD_BASE_URL}/{self.filename}"


ARCHIVES = (
    ArchiveContract(
        backend="cpu",
        filename=LLAMA_CPP_CPU_ARCHIVE,
        size=LLAMA_CPP_CPU_ARCHIVE_SIZE,
        sha256=LLAMA_CPP_CPU_ARCHIVE_SHA256,
        required_files=frozenset(COMMON_REQUIRED_FILES),
    ),
    ArchiveContract(
        backend="vulkan",
        filename=LLAMA_CPP_VULKAN_ARCHIVE,
        size=LLAMA_CPP_VULKAN_ARCHIVE_SIZE,
        sha256=LLAMA_CPP_VULKAN_ARCHIVE_SHA256,
        required_files=frozenset(COMMON_REQUIRED_FILES | {"ggml-vulkan.dll"}),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, int | str]:
    return {"size": path.stat().st_size, "sha256": _sha256(path)}


def _validate_identity(path: Path, *, size: int, sha256: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} not found: {path}")
    actual_size = path.stat().st_size
    if actual_size != size:
        raise RuntimeError(f"{label} size mismatch: expected {size}, found {actual_size}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != sha256:
        raise RuntimeError(f"{label} SHA-256 mismatch: expected {sha256}, found {actual_sha256}")


def _download(contract: ArchiveContract, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            _validate_identity(
                destination,
                size=contract.size,
                sha256=contract.sha256,
                label=contract.filename,
            )
            return
        except RuntimeError:
            destination.unlink()
    partial = destination.with_name(f"{destination.name}.partial")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(contract.url, headers={"User-Agent": "PuriPulyHeart-build"})
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        _validate_identity(
            partial,
            size=contract.size,
            sha256=contract.sha256,
            label=contract.filename,
        )
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def _is_forbidden_model_name(name: str) -> bool:
    lowered = name.casefold()
    return lowered.endswith(".gguf") or "mmproj" in lowered or lowered in KNOWN_MODEL_FILENAMES


def _selected_runtime_name(name: str) -> bool:
    lowered = name.casefold()
    return (
        lowered in FIXED_RUNTIME_FILES
        or (lowered.startswith("ggml-cpu-") and lowered.endswith(".dll"))
        or lowered == "ggml-vulkan.dll"
    )


def _read_verified_archive(archive_path: Path, contract: ArchiveContract) -> bytes:
    try:
        payload = archive_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"{contract.filename} not found: {archive_path}") from exc
    if len(payload) != contract.size:
        raise RuntimeError(
            f"{contract.filename} size mismatch: expected {contract.size}, found {len(payload)}"
        )
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != contract.sha256:
        raise RuntimeError(
            f"{contract.filename} SHA-256 mismatch: expected {contract.sha256}, "
            f"found {actual_sha256}"
        )
    return payload


def _extract_runtime_archive(
    archive_source: Path | BinaryIO,
    destination: Path,
    contract: ArchiveContract,
) -> list[dict[str, int | str]]:
    destination.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, int | str]] = []
    selected_names: set[str] = set()
    with zipfile.ZipFile(archive_source) as archive:
        for entry in archive.infolist():
            normalized = entry.filename.replace("\\", "/")
            path = PurePosixPath(normalized)
            if entry.is_dir():
                continue
            if path.is_absolute() or len(path.parts) != 1 or path.name in {"", ".", ".."}:
                raise RuntimeError(f"unsafe llama.cpp archive member: {entry.filename}")
            if _is_forbidden_model_name(path.name):
                raise RuntimeError(
                    f"llama.cpp runtime archive contains model artifact: {path.name}"
                )
            if not _selected_runtime_name(path.name):
                continue
            lowered = path.name.casefold()
            if lowered in selected_names:
                raise RuntimeError(f"duplicate llama.cpp runtime member: {path.name}")
            selected_names.add(lowered)
            target = destination / path.name
            with archive.open(entry) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            identity = _file_identity(target)
            records.append({"path": path.name, **identity})
    missing = contract.required_files - selected_names
    if missing:
        raise RuntimeError(
            f"{contract.backend} llama.cpp runtime archive is missing required files: "
            f"{sorted(missing)}"
        )
    return sorted(records, key=lambda item: str(item["path"]).casefold())


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid llama.cpp runtime manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid llama.cpp runtime manifest object: {path}")
    return payload


def _expected_archive_map() -> dict[str, ArchiveContract]:
    return {contract.backend: contract for contract in ARCHIVES}


def _validate_manifest_identity(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError("llama.cpp runtime manifest schema mismatch")
    release = manifest.get("release")
    expected_release = {
        "build": LLAMA_CPP_BUILD,
        "commit": LLAMA_CPP_COMMIT,
        "url": RELEASE_URL,
        "runtime_dirname": LLAMA_CPP_RUNTIME_DIRNAME,
        "packaged_relative_dir": PACKAGED_RUNTIME_RELATIVE_DIR.as_posix(),
    }
    if release != expected_release:
        raise RuntimeError("llama.cpp runtime manifest release identity mismatch")
    archives = manifest.get("archives")
    if not isinstance(archives, dict) or set(archives) != set(_expected_archive_map()):
        raise RuntimeError("llama.cpp runtime manifest backend set mismatch")
    normalized: dict[str, dict[str, object]] = {}
    for backend, contract in _expected_archive_map().items():
        entry = archives.get(backend)
        if not isinstance(entry, dict):
            raise RuntimeError(f"invalid {backend} llama.cpp runtime manifest entry")
        expected_identity = {
            "filename": contract.filename,
            "size": contract.size,
            "sha256": contract.sha256,
            "url": contract.url,
        }
        if {key: entry.get(key) for key in expected_identity} != expected_identity:
            raise RuntimeError(f"{backend} llama.cpp archive identity mismatch")
        files = entry.get("files")
        if not isinstance(files, list):
            raise RuntimeError(f"invalid {backend} llama.cpp runtime file inventory")
        normalized[backend] = entry
    return normalized


def _validate_runtime_tree(manifest_path: Path, runtime_root: Path) -> dict[str, object]:
    manifest = _read_manifest(manifest_path)
    archives = _validate_manifest_identity(manifest)
    expected_root_names = {"manifest.json", *archives.keys()}
    actual_root_names = {path.name for path in runtime_root.iterdir()}
    if actual_root_names != expected_root_names:
        raise RuntimeError(
            "llama.cpp runtime root inventory mismatch: "
            f"expected {sorted(expected_root_names)}, found {sorted(actual_root_names)}"
        )
    for backend, entry in archives.items():
        backend_root = runtime_root / backend
        if not backend_root.is_dir() or backend_root.is_symlink():
            raise RuntimeError(f"packaged llama.cpp backend directory not found: {backend_root}")
        records = entry["files"]
        record_names: set[str] = set()
        for raw_record in records:
            if not isinstance(raw_record, dict):
                raise RuntimeError(f"invalid {backend} llama.cpp runtime file record")
            name = raw_record.get("path")
            size = raw_record.get("size")
            sha256 = raw_record.get("sha256")
            if (
                not isinstance(name, str)
                or PurePosixPath(name).name != name
                or "/" in name
                or "\\" in name
            ):
                raise RuntimeError(f"invalid {backend} llama.cpp runtime path: {name}")
            lowered = name.casefold()
            if lowered in record_names or not _selected_runtime_name(name):
                raise RuntimeError(f"invalid {backend} llama.cpp runtime inventory member: {name}")
            if _is_forbidden_model_name(name):
                raise RuntimeError(f"model artifact found in llama.cpp runtime inventory: {name}")
            if not isinstance(size, int) or not isinstance(sha256, str):
                raise RuntimeError(f"invalid {backend} llama.cpp runtime identity: {name}")
            record_names.add(lowered)
            _validate_identity(
                backend_root / name,
                size=size,
                sha256=sha256,
                label=f"{backend} llama.cpp runtime file {name}",
            )
        required = _expected_archive_map()[backend].required_files
        if not required.issubset(record_names):
            raise RuntimeError(
                f"{backend} llama.cpp runtime inventory is missing required files: "
                f"{sorted(required - record_names)}"
            )
        actual_entries = list(backend_root.iterdir())
        if any(not path.is_file() or path.is_symlink() for path in actual_entries):
            raise RuntimeError(f"{backend} llama.cpp runtime directory contains nested content")
        actual_names = {path.name.casefold() for path in actual_entries}
        if actual_names != record_names:
            raise RuntimeError(f"{backend} llama.cpp runtime directory inventory mismatch")
    return manifest


def _fixed_provenance() -> dict[str, dict[str, int | str]]:
    return {
        "LICENSE": {"size": LICENSE_SIZE, "sha256": LICENSE_SHA256},
        "README.md": {"size": README_SIZE, "sha256": README_SHA256},
    }


def _validate_packaged_provenance(manifest: dict[str, object], provenance_root: Path) -> None:
    expected = _fixed_provenance()
    if manifest.get("provenance") != expected:
        raise RuntimeError("llama.cpp package manifest provenance identity mismatch")
    actual_entries = list(provenance_root.iterdir()) if provenance_root.is_dir() else []
    if {path.name for path in actual_entries} != set(expected) or any(
        not path.is_file() or path.is_symlink() for path in actual_entries
    ):
        raise RuntimeError("llama.cpp packaged provenance inventory mismatch")
    for name, identity in expected.items():
        _validate_identity(
            provenance_root / name,
            size=int(identity["size"]),
            sha256=str(identity["sha256"]),
            label=f"packaged llama.cpp provenance {name}",
        )


def _validate_provenance(repo_root: Path) -> dict[str, dict[str, int | str]]:
    provenance_root = repo_root / PROVENANCE_RELATIVE_DIR
    license_path = provenance_root / "LICENSE"
    readme_path = provenance_root / "README.md"
    _validate_identity(
        license_path,
        size=LICENSE_SIZE,
        sha256=LICENSE_SHA256,
        label="llama.cpp license",
    )
    _validate_identity(
        readme_path,
        size=README_SIZE,
        sha256=README_SHA256,
        label="llama.cpp provenance README",
    )
    readme = readme_path.read_text(encoding="utf-8")
    for expected in (LLAMA_CPP_BUILD, LLAMA_CPP_COMMIT, RELEASE_URL):
        if expected not in readme:
            raise RuntimeError(f"llama.cpp provenance README is missing identity: {expected}")
    return _fixed_provenance()


def prepare_runtime(repo_root: Path, cache_dir: Path, output_root: Path) -> dict[str, object]:
    repo_root = repo_root.resolve()
    cache_dir = cache_dir.resolve()
    output_root = output_root.resolve()
    provenance = _validate_provenance(repo_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists():
        try:
            existing = _validate_runtime_tree(output_root / "manifest.json", output_root)
            if existing.get("provenance") != provenance:
                raise RuntimeError("llama.cpp prepared provenance identity mismatch")
        except RuntimeError:
            pass
        else:
            return existing
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent)
    ).resolve()
    try:
        archive_manifest: dict[str, object] = {}
        for contract in ARCHIVES:
            archive_path = cache_dir / contract.filename
            _download(contract, archive_path)
            archive_payload = _read_verified_archive(archive_path, contract)
            records = _extract_runtime_archive(
                io.BytesIO(archive_payload),
                temporary_root / contract.backend,
                contract,
            )
            archive_manifest[contract.backend] = {
                "filename": contract.filename,
                "size": contract.size,
                "sha256": contract.sha256,
                "url": contract.url,
                "files": records,
            }
        manifest: dict[str, object] = {
            "schema": MANIFEST_SCHEMA,
            "release": {
                "build": LLAMA_CPP_BUILD,
                "commit": LLAMA_CPP_COMMIT,
                "url": RELEASE_URL,
                "runtime_dirname": LLAMA_CPP_RUNTIME_DIRNAME,
                "packaged_relative_dir": PACKAGED_RUNTIME_RELATIVE_DIR.as_posix(),
            },
            "archives": archive_manifest,
            "provenance": provenance,
        }
        manifest_path = temporary_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _validate_runtime_tree(manifest_path, temporary_root)
        if output_root.exists():
            shutil.rmtree(output_root)
        os.replace(temporary_root, output_root)
        return _read_manifest(output_root / "manifest.json")
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def pyinstaller_data_entries(repo_root: Path) -> list[tuple[str, str]]:
    repo_root = repo_root.resolve()
    runtime_root = repo_root / "build" / "llama.cpp" / LLAMA_CPP_RUNTIME_DIRNAME
    manifest_path = runtime_root / "manifest.json"
    manifest = _validate_runtime_tree(manifest_path, runtime_root)
    provenance = _validate_provenance(repo_root)
    entries: list[tuple[str, str]] = []
    archives = manifest["archives"]
    for backend, entry in archives.items():
        destination = PACKAGED_RUNTIME_RELATIVE_DIR / backend
        for record in entry["files"]:
            entries.append((str(runtime_root / backend / record["path"]), destination.as_posix()))
    entries.append((str(manifest_path), PACKAGED_RUNTIME_RELATIVE_DIR.as_posix()))
    for name in provenance:
        entries.append(
            (str(repo_root / PROVENANCE_RELATIVE_DIR / name), PROVENANCE_RELATIVE_DIR.as_posix())
        )
    return entries


def validate_runtime_root(runtime_root: Path) -> dict[str, object]:
    runtime_root = runtime_root.resolve()
    return _validate_runtime_tree(runtime_root / "manifest.json", runtime_root)


def normalize_pyinstaller_binaries(binaries: list[tuple[str, str, str]], repo_root: Path) -> None:
    prepared_root = (
        repo_root.resolve() / "build" / "llama.cpp" / LLAMA_CPP_RUNTIME_DIRNAME
    ).resolve()
    packaged_prefix = f"{PACKAGED_RUNTIME_RELATIVE_DIR.as_posix().casefold()}/"
    retained = []
    for binary in binaries:
        destination, source, _typecode = binary
        normalized_destination = destination.replace("\\", "/").casefold()
        try:
            Path(source).resolve().relative_to(prepared_root)
        except ValueError:
            retained.append(binary)
            continue
        if normalized_destination.startswith(packaged_prefix):
            retained.append(binary)
    binaries[:] = retained


def verify_package(package_root: Path, *, launch: bool = False) -> dict[str, object]:
    package_root = package_root.resolve()
    runtime_root = package_root / PACKAGED_RUNTIME_RELATIVE_DIR
    manifest = _validate_runtime_tree(runtime_root / "manifest.json", runtime_root)
    provenance_root = package_root / PROVENANCE_RELATIVE_DIR
    _validate_packaged_provenance(manifest, provenance_root)
    forbidden = [
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file() and _is_forbidden_model_name(path.name)
    ]
    if forbidden:
        raise RuntimeError(f"managed Gemma model artifacts found in package: {sorted(forbidden)}")
    runtime_hashes = {
        raw_record["sha256"]
        for archive in manifest["archives"].values()
        for raw_record in archive["files"]
    }
    duplicate_runtime_files = []
    for path in package_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            path.relative_to(runtime_root)
        except ValueError:
            if _sha256(path) in runtime_hashes:
                duplicate_runtime_files.append(path.relative_to(package_root).as_posix())
    if duplicate_runtime_files:
        raise RuntimeError(
            "managed llama.cpp runtime files escaped their owned package directory: "
            f"{sorted(duplicate_runtime_files)}"
        )
    launched_backends: list[str] = []
    if launch:
        for backend in sorted(manifest["archives"]):
            executable = runtime_root / backend / "llama-server.exe"
            completed = subprocess.run(
                [str(executable), "--version"],
                cwd=executable.parent,
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            output = f"{completed.stdout}\n{completed.stderr}"
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{backend} packaged llama.cpp server failed with exit code "
                    f"{completed.returncode}: {output.strip()}"
                )
            if f"build {LLAMA_CPP_BUILD.removeprefix('b')}" not in output:
                raise RuntimeError(f"{backend} packaged llama.cpp server build identity mismatch")
            if f"commit {LLAMA_CPP_COMMIT[:9]}" not in output:
                raise RuntimeError(f"{backend} packaged llama.cpp server commit identity mismatch")
            launched_backends.append(backend)
    return {
        "schema": MANIFEST_SCHEMA,
        "package_root": str(package_root),
        "build": LLAMA_CPP_BUILD,
        "commit": LLAMA_CPP_COMMIT,
        "backends": sorted(manifest["archives"]),
        "launched_backends": launched_backends,
        "model_artifacts": [],
    }


def verify_installer(installer_path: Path) -> dict[str, object]:
    installer_path = installer_path.resolve()
    try:
        installer = installer_path.read_text(encoding="utf-8").casefold()
    except OSError as exc:
        raise RuntimeError(f"installer source not found: {installer_path}") from exc
    present = sorted(
        identity for identity in INSTALLER_FORBIDDEN_IDENTITIES if identity in installer
    )
    if present:
        raise RuntimeError(
            f"installer source contains managed Gemma download identities: {present}"
        )
    if ".gguf" in installer or "mmproj" in installer:
        raise RuntimeError("installer source contains model artifact identities")
    run_section = re.search(r"(?ms)^\[run\]\s*(.*?)(?=^\[|\Z)", installer)
    if run_section is not None and "{#myappexename}" in run_section.group(1):
        raise RuntimeError("installer source launches the application postinstall")
    packaged_tree_source = 'source: "{#mypackagedappdir}\\*"'
    if packaged_tree_source not in installer:
        raise RuntimeError(
            "installer source does not install the verified packaged application tree"
        )
    return {
        "schema": MANIFEST_SCHEMA,
        "installer_path": str(installer_path),
        "managed_gemma_install_download": False,
        "application_postinstall_launch": False,
    }


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--repo-root", type=Path, default=_default_repo_root())
    prepare.add_argument("--cache-dir", type=Path)
    prepare.add_argument("--output-root", type=Path)
    verify = subparsers.add_parser("verify-package")
    verify.add_argument("package_root", type=Path)
    verify.add_argument("--launch", action="store_true")
    verify_installer_parser = subparsers.add_parser("verify-installer")
    verify_installer_parser.add_argument("installer_path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.command == "prepare":
        repo_root = arguments.repo_root.resolve()
        cache_dir = arguments.cache_dir or repo_root / "build" / "download-cache" / "llama.cpp"
        output_root = (
            arguments.output_root or repo_root / "build" / "llama.cpp" / LLAMA_CPP_RUNTIME_DIRNAME
        )
        result = prepare_runtime(repo_root, cache_dir, output_root)
    elif arguments.command == "verify-package":
        result = verify_package(arguments.package_root, launch=arguments.launch)
    else:
        result = verify_installer(arguments.installer_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
