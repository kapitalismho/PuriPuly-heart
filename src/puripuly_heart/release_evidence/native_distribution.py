from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import io
import json
import marshal
import os
import py_compile
import shutil
import sys
import tomllib
import types
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from packaging.requirements import Requirement

from puripuly_heart.release_evidence.gpu_worker_distribution import validate_gpu_worker_runtime
from puripuly_heart.release_evidence.native_pe import stage_vc_runtime, validate_pe_dependencies

_LAYOUT_SCHEMA = "puripuly-heart/native-artifact-layout/v1"
_MANIFEST_SCHEMA = "puripuly-heart/native-artifact-manifest/v1"
_NATIVE_MARKER_ENVIRONMENT = {
    "implementation_name": "cpython",
    "implementation_version": "3.14.7",
    "os_name": "nt",
    "platform_machine": "AMD64",
    "platform_python_implementation": "CPython",
    "platform_system": "Windows",
    "python_full_version": "3.14.7",
    "python_version": "3.14",
    "sys_platform": "win32",
    "extra": "",
}
_FORBIDDEN_DISTRIBUTIONS = frozenset({"flet-desktop", "flet-cli", "pyinstaller"})
_FORBIDDEN_PTH = frozenset({"a1_coverage.pth", "distutils-precedence.pth"})
_SOUNDDEVICE_RUNTIME_ROOT = PurePosixPath("_sounddevice_data/portaudio-binaries")
_SOUNDDEVICE_STANDARD_DLL = _SOUNDDEVICE_RUNTIME_ROOT / "libportaudio64bit.dll"
_SOUNDDEVICE_ASIO_DLL = _SOUNDDEVICE_RUNTIME_ROOT / "libportaudio64bit-asio.dll"


def _canonical_name(value: str) -> str:
    return value.lower().replace("_", "-").replace(".", "-")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or str(path) != value
        or "\\" in value
        or ":" in value
        or "\0" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"native artifact path must be a normalized relative path: {value!r}")
    return path


@dataclass(frozen=True)
class NativeArtifactLayout:
    source: Path
    values: dict[str, str]

    @classmethod
    def load(cls, source: Path) -> "NativeArtifactLayout":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.pop("schema", None) != _LAYOUT_SCHEMA:
            raise ValueError(f"unexpected native artifact layout schema in {source}")
        expected = {
            "host_executable",
            "console_executable",
            "python_executable",
            "application_root",
            "python_archive",
            "dependency_root",
            "stdlib_root",
            "stdlib_archive",
            "extension_dll_root",
            "native_runtime_root",
            "soxr_runtime_root",
            "soxr_compliance_root",
            "llama_runtime_root",
            "local_qwen_runtime_root",
            "overlay_executable",
            "gpu_worker_executable",
            "openvr_dll",
            "artifact_manifest",
        }
        if set(payload) != expected:
            missing = sorted(expected - set(payload))
            extra = sorted(set(payload) - expected)
            raise ValueError(
                f"native artifact layout keys differ: missing={missing}, extra={extra}"
            )
        values: dict[str, str] = {}
        for key, value in payload.items():
            if not isinstance(value, str):
                raise TypeError(f"native artifact path {key!r} must be a string")
            if value != ".":
                _safe_relative(value)
            values[key] = value
        if values["host_executable"] != "PuriPulyHeart.exe":
            raise ValueError("native host executable must be PuriPulyHeart.exe")
        if values["console_executable"] != "puripuly.exe":
            raise ValueError("native console executable must be puripuly.exe")
        if values["python_executable"] != "python.exe":
            raise ValueError("native Python executable must be python.exe")
        if values["stdlib_archive"] != "python314.zip":
            raise ValueError("native standard library archive must be python314.zip")
        if values["python_archive"] != str(
            PurePosixPath(values["application_root"]) / "python.zip"
        ):
            raise ValueError("native Python archive must be application_root/python.zip")
        code_roots = [
            PurePosixPath(values[key])
            for key in ("application_root", "dependency_root", "stdlib_root")
        ]
        if any(root == PurePosixPath(".") for root in code_roots) or any(
            left == right or left in right.parents or right in left.parents
            for index, left in enumerate(code_roots)
            for right in code_roots[index + 1 :]
        ):
            raise ValueError(
                "native Python code roots must be distinct non-overlapping directories"
            )
        return cls(source.resolve(), values)

    def resolve(self, root: Path, key: str) -> Path:
        value = self.values[key]
        return (
            root.resolve() if value == "." else root.resolve().joinpath(*PurePosixPath(value).parts)
        )

    def render_cpp_header(self) -> str:
        names = {
            "host_executable": "kHostExecutable",
            "python_executable": "kPythonExecutable",
            "application_root": "kApplicationRoot",
            "python_archive": "kPythonArchive",
            "dependency_root": "kDependencyRoot",
            "stdlib_archive": "kStdlibArchive",
            "extension_dll_root": "kExtensionDllRoot",
        }
        lines = ["#pragma once", "", "namespace puripuly_layout {"]
        for key, cpp_name in names.items():
            value = self.values[key].replace("/", "\\\\")
            lines.append(f'inline constexpr wchar_t {cpp_name}[] = L"{value}";')
        lines.extend(["}", ""])
        return "\n".join(lines)


def verify_upstream_inputs(spec_path: Path, input_root: Path) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema") != "puripuly-heart/native-upstream-inputs/v1":
        raise ValueError("unexpected upstream input schema")
    verified: dict[str, Any] = {}
    for filename, expected in spec["inputs"].items():
        matches = [path for path in input_root.rglob(filename) if path.is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected exactly one pinned input named {filename!r} below {input_root}, got {matches}"
            )
        path = matches[0]
        size = path.stat().st_size
        digest = _sha256(path)
        if size != expected["bytes"] or digest != expected["sha256"]:
            raise ValueError(f"pinned input mismatch: {path}")
        verified[filename] = {
            "path": str(path.resolve()),
            "bytes": size,
            "sha256": digest,
        }
    return {"versions": spec["versions"], "inputs": verified}


def filter_requirements(source: Path, destination: Path) -> list[str]:
    excluded = {"flet-desktop", "soxr"}
    lines = source.read_text(encoding="utf-8").splitlines()
    preamble: list[str] = []
    requirements: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        starts_requirement = bool(
            line and not line[0].isspace() and not line.startswith(("#", "--")) and "==" in line
        )
        if starts_requirement:
            current = [line]
            requirements.append(current)
        elif current is None:
            preamble.append(line)
        else:
            current.append(line)
    kept = list(preamble)
    removed: list[str] = []
    for block in requirements:
        requirement = Requirement(block[0].rstrip().removesuffix("\\").strip())
        name = _canonical_name(requirement.name)
        if name in excluded:
            removed.append(name)
            continue
        if requirement.marker is not None:
            if not requirement.marker.evaluate(_NATIVE_MARKER_ENVIRONMENT):
                continue
            continuation = " \\" if block[0].rstrip().endswith("\\") else ""
            block[0] = str(requirement).split(";", 1)[0].rstrip() + continuation
        kept.extend(block)
    if sorted(removed) != sorted(excluded):
        raise ValueError(f"requirements did not contain exactly the native exclusions: {removed}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(kept).rstrip() + "\n", encoding="utf-8", newline="\n")
    return removed


def stage_product_metadata(site_packages: Path, pyproject_path: Path) -> dict[str, str]:
    project = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["project"]
    name = _canonical_name(project["name"])
    version = project["version"]
    if name != "puripuly-heart":
        raise ValueError(f"unexpected native product distribution name: {name}")
    destination = site_packages / f"puripuly_heart-{version}.dist-info"
    if destination.exists():
        raise FileExistsError(f"native product metadata already exists: {destination}")
    destination.mkdir(parents=True)
    members = {
        "METADATA": (
            "Metadata-Version: 2.4\n"
            f"Name: {project['name']}\n"
            f"Version: {version}\n"
            f"Summary: {project.get('description', '')}\n"
        ).encode(),
        "WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: puripuly-heart-native-staging\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ).encode(),
        "INSTALLER": b"native-experimental\n",
    }
    for filename, data in members.items():
        (destination / filename).write_bytes(data)
    record_name = f"{destination.name}/RECORD"
    rows = [
        [f"{destination.name}/{filename}", _record_digest(data), str(len(data))]
        for filename, data in sorted(members.items())
    ]
    rows.append([record_name, "", ""])
    with (destination / "RECORD").open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    return {"name": project["name"], "version": version, "metadata": destination.name}


def _record_digest(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode(
        "ascii"
    )


def finalize_soxr_wheel(wheel_path: Path, dll_path: Path, destination: Path) -> dict[str, str]:
    dll_data = dll_path.read_bytes()
    with zipfile.ZipFile(wheel_path) as source:
        names = source.namelist()
        record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
        if len(record_names) != 1:
            raise ValueError("soxr wheel must contain exactly one RECORD")
        record_name = record_names[0]
        members = {
            name: source.read(name)
            for name in names
            if not name.endswith("/") and name != record_name and name != "soxr/soxr.dll"
        }
    if not any(name.startswith("soxr/") and name.lower().endswith(".pyd") for name in members):
        raise ValueError("soxr wheel does not contain its extension module")
    members["soxr/soxr.dll"] = dll_data
    rows = [[name, _record_digest(data), str(len(data))] for name, data in sorted(members.items())]
    rows.append([record_name, "", ""])
    record_buffer = []
    for row in rows:
        output = __import__("io").StringIO()
        csv.writer(output, lineterminator="\n").writerow(row)
        record_buffer.append(output.getvalue())
    members[record_name] = "".join(record_buffer).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as target:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            target.writestr(info, data)
    os.replace(temporary, destination)
    verify_wheel_record(destination)
    return {
        "wheel_sha256": _sha256(destination),
        "dll_sha256": hashlib.sha256(dll_data).hexdigest(),
    }


def verify_wheel_record(wheel_path: Path) -> None:
    with zipfile.ZipFile(wheel_path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
        if len(record_names) != 1:
            raise ValueError("wheel must contain exactly one RECORD")
        record_name = record_names[0]
        rows = {
            row[0]: row[1:]
            for row in csv.reader(archive.read(record_name).decode("utf-8").splitlines())
        }
        if set(rows) != set(names):
            raise ValueError("wheel RECORD ownership does not match archive members")
        for name in names:
            digest, size = rows[name]
            if name == record_name:
                if digest or size:
                    raise ValueError("wheel RECORD must leave its own digest and size empty")
                continue
            data = archive.read(name)
            if digest != _record_digest(data) or size != str(len(data)):
                raise ValueError(f"wheel RECORD mismatch: {name}")
        soxr_runtime = {name for name in names if name.startswith("soxr/")}
        if not any(name.lower().endswith(".pyd") for name in soxr_runtime):
            raise ValueError("wheel lacks the soxr extension module")
        if "soxr/soxr.dll" not in soxr_runtime:
            raise ValueError("wheel lacks soxr/soxr.dll")


def verify_installed_soxr_record(site_packages: Path) -> dict[str, str]:
    records = list(site_packages.glob("soxr-*.dist-info/RECORD"))
    if len(records) != 1:
        raise ValueError(f"installed soxr must contain exactly one RECORD: {records}")
    record = records[0]
    rows = {
        row[0].replace("\\", "/"): row[1:]
        for row in csv.reader(record.read_text(encoding="utf-8").splitlines())
    }
    runtime_members = sorted(
        name
        for name in rows
        if name == "soxr/soxr.dll" or (name.startswith("soxr/") and name.lower().endswith(".pyd"))
    )
    if len(runtime_members) != 2 or "soxr/soxr.dll" not in runtime_members:
        raise ValueError(
            f"installed soxr RECORD does not own one extension and DLL: {runtime_members}"
        )
    identities: dict[str, str] = {}
    for name in runtime_members:
        path = site_packages.joinpath(*PurePosixPath(name).parts)
        data = path.read_bytes()
        digest, size = rows[name]
        if digest != _record_digest(data) or size != str(len(data)):
            raise ValueError(f"installed soxr RECORD mismatch: {name}")
        identities[name] = hashlib.sha256(data).hexdigest()
    return identities


def verify_sounddevice_portaudio_runtime(site_packages: Path) -> dict[str, str]:
    standard = site_packages.joinpath(*_SOUNDDEVICE_STANDARD_DLL.parts)
    asio = site_packages.joinpath(*_SOUNDDEVICE_ASIO_DLL.parts)
    if not standard.is_file():
        raise ValueError("native artifact lacks the standard sounddevice PortAudio runtime")
    if asio.exists():
        raise ValueError("native artifact includes the unsupported sounddevice ASIO runtime")
    return {str(_SOUNDDEVICE_STANDARD_DLL): _sha256(standard)}


def stage_sounddevice_portaudio_runtime(site_packages: Path) -> dict[str, dict[str, str]]:
    standard = site_packages.joinpath(*_SOUNDDEVICE_STANDARD_DLL.parts)
    asio = site_packages.joinpath(*_SOUNDDEVICE_ASIO_DLL.parts)
    if not standard.is_file():
        raise ValueError("native artifact lacks the standard sounddevice PortAudio runtime")
    excluded: dict[str, str] = {}
    if asio.exists():
        excluded[str(_SOUNDDEVICE_ASIO_DLL)] = _sha256(asio)
        asio.unlink()
    return {
        "retained": verify_sounddevice_portaudio_runtime(site_packages),
        "excluded": excluded,
    }


def _distribution_versions(site_packages: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions(path=[str(site_packages)]):
        name = distribution.metadata.get("Name")
        if not name:
            raise ValueError(f"distribution has no Name metadata: {distribution._path}")
        canonical = _canonical_name(name)
        if canonical in versions:
            raise ValueError(f"duplicate distribution metadata: {canonical}")
        versions[canonical] = distribution.version
    return versions


def validate_dependencies(site_packages: Path, requirements_path: Path) -> dict[str, Any]:
    requirements: dict[str, Requirement] = {}
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        if not line or line[0].isspace() or line.startswith(("#", "--")):
            continue
        requirement = Requirement(line.rstrip().removesuffix("\\").strip())
        name = _canonical_name(requirement.name)
        if name == "flet-desktop":
            continue
        if requirement.marker is None or requirement.marker.evaluate(_NATIVE_MARKER_ENVIRONMENT):
            requirements[name] = requirement
    versions = _distribution_versions(site_packages)
    names = set(versions)
    forbidden = sorted(names & _FORBIDDEN_DISTRIBUTIONS)
    if forbidden:
        raise ValueError(f"native dependency closure includes forbidden distributions: {forbidden}")
    missing = sorted(requirements.keys() - names)
    unexpected = sorted(names - requirements.keys() - {"puripuly-heart"})
    mismatched = {
        name: {"expected": str(requirement.specifier), "actual": versions[name]}
        for name, requirement in requirements.items()
        if name in versions and not requirement.specifier.contains(versions[name], prereleases=True)
    }
    if missing or unexpected or mismatched:
        raise ValueError(
            f"native dependency closure mismatch: missing={missing}, "
            f"unexpected={unexpected}, versions={mismatched}"
        )
    return {
        "distribution_count": len(versions),
        "distributions": sorted(versions),
        "versions": versions,
        "requirements_sha256": _sha256(requirements_path),
        "marker_environment": _NATIVE_MARKER_ENVIRONMENT,
    }


def validate_target(
    target_root: Path, layout_path: Path, requirements_path: Path, vc_runtime_path: Path
) -> dict[str, Any]:
    layout = NativeArtifactLayout.load(layout_path)
    required_paths = {
        key: layout.resolve(target_root, key)
        for key in (
            "host_executable",
            "console_executable",
            "python_executable",
            "application_root",
            "python_archive",
            "dependency_root",
            "stdlib_archive",
            "extension_dll_root",
            "overlay_executable",
            "gpu_worker_executable",
            "openvr_dll",
        )
    }
    missing = sorted(str(path) for path in required_paths.values() if not path.exists())
    if missing:
        raise FileNotFoundError(f"native artifact is incomplete: {missing}")
    gpu_worker_runtime = validate_gpu_worker_runtime(required_paths["gpu_worker_executable"].parent)
    python_archive = _validate_python_archive(target_root, layout)
    site_packages = required_paths["dependency_root"]
    dependencies = validate_dependencies(site_packages, requirements_path)
    product_metadata = list(site_packages.glob("puripuly_heart-*.dist-info"))
    if len(product_metadata) != 1 or (product_metadata[0] / "direct_url.json").exists():
        raise ValueError(
            "native artifact must contain one non-editable product distribution metadata directory"
        )
    bad_pth = sorted(
        path.name for path in site_packages.glob("*.pth") if path.name.lower() in _FORBIDDEN_PTH
    )
    if bad_pth:
        raise ValueError(f"native dependency closure contains build-only path hooks: {bad_pth}")
    soxr_runtime = verify_installed_soxr_record(site_packages)
    portaudio_runtime = verify_sounddevice_portaudio_runtime(site_packages)
    python_digest = _sha256(required_paths["python_executable"])
    if python_digest != "4942b86a6597e5aee0128daa00050ed79bc21f6e709a78eb19cbfeb0c2f39ac9":
        raise ValueError("native python.exe is not the pinned official CPython executable")
    python_copies = sorted(path for path in target_root.rglob("python.exe") if path.is_file())
    if python_copies != [required_paths["python_executable"]]:
        raise ValueError(f"native artifact must contain one python.exe: {python_copies}")
    pe_dependencies = validate_pe_dependencies(target_root, vc_runtime_path)
    return {
        **dependencies,
        "python_executable_sha256": python_digest,
        "soxr_runtime": soxr_runtime,
        "python_archive": python_archive,
        "portaudio_runtime": portaudio_runtime,
        "gpu_worker_runtime": gpu_worker_runtime,
        "pe_dependencies": pe_dependencies,
        "paths": {
            key: str(value.relative_to(target_root)) for key, value in required_paths.items()
        },
    }


def _runtime_files(root: Path) -> list[Path]:
    if root.is_symlink():
        raise ValueError(f"Python archive inputs cannot be symlinks: {root}")
    files = []
    seen = set()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Python archive inputs cannot be symlinks: {path}")
        relative = path.relative_to(root).as_posix()
        _safe_relative(relative)
        if relative.casefold() in seen:
            raise ValueError(f"ambiguous Python input path: {relative}")
        seen.add(relative.casefold())
        if path.is_file():
            files.append(path)
    return files


def _validate_bytecode(data: bytes, name: str) -> str:
    if len(data) < 16 or data[:4] != importlib.util.MAGIC_NUMBER:
        raise ValueError(f"invalid Python bytecode header: {name}")
    flags = int.from_bytes(data[4:8], "little")
    modes = {0: "timestamp", 1: "unchecked-hash", 3: "checked-hash"}
    if flags not in modes:
        raise ValueError(f"invalid Python bytecode flags: {name}")
    stream = io.BytesIO(data)
    stream.seek(16)
    try:
        code = marshal.load(stream)
    except (EOFError, ValueError, TypeError) as exc:
        raise ValueError(f"malformed bytecode: {name}") from exc
    if not isinstance(code, types.CodeType) or stream.tell() != len(data):
        raise ValueError(f"bytecode does not contain exactly one module: {name}")
    return modes[flags]


def _validate_compiled_member(data: bytes, source: bytes, name: str) -> None:
    if _validate_bytecode(data, name) != "unchecked-hash" or data[
        8:16
    ] != importlib.util.source_hash(source):
        raise ValueError(f"invalid unchecked-hash bytecode: {name}")


def compile_runtime(target_root: Path, layout_path: Path) -> dict[str, Any]:
    if sys.flags.optimize != 0:
        raise RuntimeError(
            "native runtime bytecode must be compiled by an optimization-0 interpreter"
        )
    target_root = target_root.resolve()
    layout = NativeArtifactLayout.load(layout_path)
    application_root = layout.resolve(target_root, "application_root")
    stdlib_root = layout.resolve(target_root, "stdlib_root")
    files = [
        path
        for root in (application_root, layout.resolve(target_root, "dependency_root"), stdlib_root)
        for path in _runtime_files(root)
    ]
    failures: list[str] = []
    outputs: list[dict[str, str]] = []
    for source in sorted(path for path in files if path.suffix == ".py"):
        if "__pycache__" in source.parts:
            raise ValueError(f"packed package contains source inside a cache directory: {source}")
        destination = (
            application_root / "product_bootstrap.pyc"
            if source == application_root / "product_bootstrap.py"
            else Path(importlib.util.cache_from_source(str(source), optimization=""))
        )
        try:
            py_compile.compile(
                str(source),
                cfile=str(destination),
                dfile=source.relative_to(target_root).as_posix(),
                doraise=True,
                optimize=0,
                invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
            )
        except py_compile.PyCompileError as exc:
            failures.append(str(exc))
            continue
        outputs.append(
            {
                "path": destination.relative_to(target_root).as_posix(),
                "source": source.relative_to(target_root).as_posix(),
                "sha256": _sha256(destination),
                "source_sha256": _sha256(source),
                "optimization": "0",
                "invalidation_mode": "unchecked-hash",
                "provenance": "build-derived",
            }
        )
    if failures:
        raise RuntimeError("\n".join(failures))
    upstream = []
    for path in files:
        if not path.is_relative_to(stdlib_root) or path.suffix != ".pyc":
            continue
        if "__pycache__" in path.relative_to(stdlib_root).parts:
            try:
                source = Path(importlib.util.source_from_cache(str(path)))
            except ValueError as exc:
                raise ValueError(f"unsupported standard library cache: {path}") from exc
            if not source.is_file():
                raise ValueError(f"sourceless standard library cache is not importable: {path}")
            continue
        if path.with_suffix(".py").is_file():
            continue
        upstream.append(
            {
                "path": path.relative_to(target_root).as_posix(),
                "input_path": path.relative_to(target_root).as_posix(),
                "sha256": _sha256(path),
                "optimization": "upstream-unspecified",
                "invalidation_mode": _validate_bytecode(path.read_bytes(), str(path)),
                "provenance": "trusted-upstream-sourceless",
            }
        )
    return {
        "python": sys.version,
        "python_optimize": sys.flags.optimize,
        "bytecode": outputs,
        "stdlib_bytecode": upstream,
    }


def _dependency_modules(root: Path) -> dict[str, str]:
    files = [path for path in _runtime_files(root) if path.relative_to(root).parts[0] != "flet"]
    paths = {path.relative_to(root).as_posix() for path in files}

    def native_module(parent: PurePosixPath, stem: str) -> bool:
        return any(
            (parent / (stem + suffix)).as_posix() in paths
            for suffix in importlib.machinery.EXTENSION_SUFFIXES
        )

    def package(parent: PurePosixPath, stem: str) -> bool:
        directory = parent / stem
        return (directory / "__init__.py").as_posix() in paths or native_module(
            directory, "__init__"
        )

    modules = {}
    folded_names = set()
    for path in files:
        if path.suffix != ".py" or "__pycache__" in path.parts:
            continue
        relative = PurePosixPath(path.relative_to(root).as_posix())
        parts = relative.with_suffix("").parts
        is_package = parts[-1] == "__init__" and len(parts) > 1
        module_parts = parts[:-1] if is_package else parts
        if any(not part or "." in part for part in module_parts):
            continue
        parent = PurePosixPath(".")
        hidden = False
        for part in module_parts[:-1]:
            if not package(parent, part) and (
                (parent / (part + ".py")).as_posix() in paths or native_module(parent, part)
            ):
                hidden = True
                break
            parent /= part
        if hidden or native_module(relative.parent, relative.stem):
            continue
        if not is_package and package(relative.parent, relative.stem):
            continue
        fullname = ".".join(module_parts)
        if fullname.casefold() in folded_names:
            raise ValueError(f"ambiguous dependency module: {fullname}")
        folded_names.add(fullname.casefold())
        modules[fullname] = relative.as_posix()
    return dict(sorted(modules.items()))


def _dependency_index(data: bytes) -> dict[str, str]:
    stream = io.BytesIO(data)
    try:
        payload = marshal.load(stream)
    except (EOFError, ValueError, TypeError) as exc:
        raise ValueError("malformed dependency index") from exc
    if (
        stream.tell() != len(data)
        or not isinstance(payload, dict)
        or set(payload) != {"version", "modules"}
        or type(payload["version"]) is not int
        or payload["version"] != 1
        or not isinstance(payload["modules"], dict)
    ):
        raise ValueError("invalid dependency index schema")
    modules = payload["modules"]
    seen_names = set()
    seen_sources = set()
    for fullname, source in modules.items():
        if not isinstance(fullname, str) or not fullname or not isinstance(source, str):
            raise ValueError("invalid dependency index module")
        path = _safe_relative(source)
        parts = path.with_suffix("").parts
        expected = ".".join(parts[:-1] if parts[-1] == "__init__" and len(parts) > 1 else parts)
        if (
            path.suffix != ".py"
            or fullname != expected
            or "__pycache__" in path.parts
            or any(not part or "." in part for part in parts)
        ):
            raise ValueError(f"invalid dependency index source: {source}")
        if fullname.casefold() in seen_names or source.casefold() in seen_sources:
            raise ValueError(f"ambiguous dependency index entry: {fullname}")
        seen_names.add(fullname.casefold())
        seen_sources.add(source.casefold())
    return modules


def _archive_members(
    archive_path: Path, *, stdlib: bool = False, dependency_root: Path | None = None
) -> set[str]:
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        normalized = [info.filename.rstrip("/").casefold() for info in infos]
        if len(normalized) != len(set(normalized)):
            raise ValueError("Python archive contains duplicate or ambiguous members")
        members = {info.filename for info in infos if not info.is_dir()}
        folded_files = {name.casefold() for name in members}
        for info in infos:
            name = info.filename
            path = _safe_relative(name.rstrip("/"))
            if "__pycache__" in path.parts:
                raise ValueError(f"Python archive contains cache paths: {name}")
            if info.compress_type != zipfile.ZIP_DEFLATED:
                raise ValueError(f"Python archive member is not deflated: {name}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Python archive contains a symlink: {name}")
            if any(parent.as_posix().casefold() in folded_files for parent in path.parents):
                raise ValueError(f"Python archive contains a file/directory collision: {name}")
            if info.is_dir():
                continue
            if path.suffix.lower() in {".pyd", ".dll", ".so", ".dylib", ".exe"}:
                raise ValueError(f"Python archive cannot contain native modules: {name}")
            if path.suffix not in {".py", ".pyc"}:
                with archive.open(info) as stream:
                    signature = stream.read(4)
                if signature[:2] == b"MZ" or signature in {
                    b"\x7fELF",
                    b"\xfe\xed\xfa\xce",
                    b"\xce\xfa\xed\xfe",
                    b"\xfe\xed\xfa\xcf",
                    b"\xcf\xfa\xed\xfe",
                }:
                    raise ValueError(f"Python archive cannot contain native binaries: {name}")
            if not stdlib and (
                name.startswith("product_bootstrap.")
                or name.startswith("prompts/")
                or (name.startswith("puripuly_heart/data/") and path.suffix not in {".py", ".pyc"})
            ):
                raise ValueError(f"Python archive contains filesystem-only content: {name}")
            if name.startswith("_native_dependencies/"):
                if stdlib or path.suffix != ".pyc" or dependency_root is None:
                    raise ValueError(f"invalid dependency archive member: {name}")
                source = dependency_root.joinpath(*path.parts[1:]).with_suffix(".py")
                if not source.is_file():
                    raise ValueError(f"dependency archive lacks retained source: {name}")
                _validate_compiled_member(archive.read(info), source.read_bytes(), name)
            elif path.suffix == ".py":
                if path.with_suffix(".pyc").as_posix() not in members:
                    raise ValueError(f"Python archive source lacks bytecode: {name}")
            elif path.suffix == ".pyc":
                source_name = path.with_suffix(".py").as_posix()
                data = archive.read(info)
                if source_name in members:
                    _validate_compiled_member(data, archive.read(source_name), name)
                elif stdlib:
                    _validate_bytecode(data, name)
                else:
                    raise ValueError(f"Python archive bytecode lacks diagnostic source: {name}")
        required = (
            {"encodings/__init__.pyc"}
            if stdlib
            else {
                "puripuly_heart/__init__.pyc",
                "flet/__init__.pyc",
                "flet/controls/material/icons.json",
                "_native_dependencies.index",
            }
        )
        if not required <= members:
            raise ValueError(f"Python archive lacks required members: {sorted(required - members)}")
        if not stdlib:
            modules = _dependency_index(archive.read("_native_dependencies.index"))
            if dependency_root is None or modules != _dependency_modules(dependency_root):
                raise ValueError("dependency index differs from filesystem import precedence")
            for source in modules.values():
                member = "_native_dependencies/" + str(PurePosixPath(source).with_suffix(".pyc"))
                if member not in members:
                    raise ValueError(f"dependency index lacks archived bytecode: {member}")
    return members


def _write_archive(
    destination: Path,
    members: dict[str, Path | bytes],
    records: dict[str, dict[str, str]],
    target_root: Path,
) -> dict[str, Any]:
    directories = {
        parent.as_posix() + "/"
        for name in members
        for parent in PurePosixPath(name).parents
        if parent != PurePosixPath(".")
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(set(members) | directories):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.compress_level = 9
            info.external_attr = (0o40755 if name.endswith("/") else 0o100644) << 16
            value = members.get(name, b"")
            digest = hashlib.sha256()
            with archive.open(info, "w") as output:
                if isinstance(value, Path):
                    with value.open("rb") as source:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            digest.update(block)
                            output.write(block)
                else:
                    output.write(value)
                    digest.update(value)
            if name in records and digest.hexdigest() != records[name]["sha256"]:
                raise ValueError(f"Python archive input changed while packing: {name}")
    return {
        "path": destination.relative_to(target_root).as_posix(),
        "sha256": _sha256(destination),
        "members": [
            (
                {
                    "path": name,
                    "sha256": hashlib.sha256(b"").hexdigest(),
                    "provenance": "build-derived-directory",
                }
                if name in directories
                else records[name]
            )
            for name in sorted(set(members) | directories)
        ],
    }


def bundle_runtime(target_root: Path, layout_path: Path, bytecode_path: Path) -> dict[str, Any]:
    target_root = target_root.resolve()
    layout = NativeArtifactLayout.load(layout_path)
    application_root = layout.resolve(target_root, "application_root")
    dependency_root = layout.resolve(target_root, "dependency_root")
    stdlib_root = layout.resolve(target_root, "stdlib_root")
    flet_root = dependency_root / "flet"
    destination = layout.resolve(target_root, "python_archive")
    stdlib_destination = layout.resolve(target_root, "stdlib_archive")
    evidence = json.loads(bytecode_path.read_text(encoding="utf-8"))
    if (
        evidence.get("python_optimize") != 0
        or "archive" in evidence
        or "stdlib_archive" in evidence
    ):
        raise ValueError("bundle-runtime requires optimization-0 prepackaging compile evidence")
    for required in (
        application_root / "puripuly_heart" / "__init__.py",
        application_root / "product_bootstrap.py",
        flet_root / "__init__.py",
        flet_root / "controls" / "material" / "icons.json",
    ):
        if not required.is_file():
            raise FileNotFoundError(f"missing Python archive input: {required}")
    if not stdlib_root.is_dir():
        raise FileNotFoundError(f"missing standard library input: {stdlib_root}")
    for path in (destination, stdlib_destination):
        if path.exists():
            raise FileExistsError(f"Python archive already exists: {path}")
    input_roots = (application_root, dependency_root, stdlib_root)
    files = {root: _runtime_files(root) for root in input_roots}
    all_files = {path for paths in files.values() for path in paths}
    compiled = {}
    for entry in evidence["bytecode"]:
        source = str(_safe_relative(entry["source"]))
        if source in compiled:
            raise ValueError(f"duplicate compile evidence: {source}")
        source_path = target_root.joinpath(*PurePosixPath(source).parts)
        path = target_root.joinpath(*_safe_relative(entry["path"]).parts)
        expected = (
            application_root / "product_bootstrap.pyc"
            if source_path == application_root / "product_bootstrap.py"
            else Path(importlib.util.cache_from_source(str(source_path), optimization=""))
        )
        if source_path not in all_files or source_path.suffix != ".py":
            raise ValueError(f"compile evidence does not own a runtime source: {source}")
        if path != expected:
            raise ValueError(f"unexpected compiled input path: {source}")
        if not path.is_file() or not source_path.is_file():
            raise FileNotFoundError(f"missing compiled input: {source}")
        if (
            _sha256(path) != entry["sha256"]
            or _sha256(source_path) != entry.get("source_sha256")
            or entry.get("optimization") != "0"
            or entry.get("invalidation_mode") != "unchecked-hash"
            or entry.get("provenance") != "build-derived"
        ):
            raise ValueError(f"compile evidence mismatch: {source}")
        _validate_compiled_member(path.read_bytes(), source_path.read_bytes(), entry["path"])
        compiled[source] = entry
    if compiled.keys() != {
        path.relative_to(target_root).as_posix() for path in all_files if path.suffix == ".py"
    }:
        raise ValueError("compile evidence does not own every runtime source")
    upstream = {}
    for entry in evidence.get("stdlib_bytecode", []):
        path = target_root.joinpath(*_safe_relative(entry["path"]).parts)
        if (
            path not in all_files
            or not path.is_relative_to(stdlib_root)
            or path.suffix != ".pyc"
            or "__pycache__" in path.parts
            or path.with_suffix(".py").exists()
            or entry["path"] in upstream
            or "source" in entry
            or "source_sha256" in entry
            or entry.get("input_path") != entry["path"]
            or entry.get("provenance") != "trusted-upstream-sourceless"
            or entry.get("optimization") != "upstream-unspecified"
            or _sha256(path) != entry.get("sha256")
            or _validate_bytecode(path.read_bytes(), entry["path"])
            != entry.get("invalidation_mode")
        ):
            raise ValueError(f"upstream bytecode evidence mismatch: {entry['path']}")
        upstream[entry["path"]] = entry
    members: dict[str, Path | bytes] = {}
    records: dict[str, dict[str, str]] = {}
    stdlib_members: dict[str, Path | bytes] = {}
    stdlib_records: dict[str, dict[str, str]] = {}
    folded_members = {False: set(), True: set()}
    cleanup = []
    archived_sources = set()

    def add(
        name: str, value: Path | bytes, record: dict[str, str], *, stdlib: bool = False
    ) -> None:
        destination_members = stdlib_members if stdlib else members
        destination_records = stdlib_records if stdlib else records
        _safe_relative(name)
        if name.casefold() in folded_members[stdlib]:
            raise ValueError(f"Python archive member collision: {name}")
        folded_members[stdlib].add(name.casefold())
        destination_members[name] = value
        destination_records[name] = {**record, "path": name}

    for root in input_roots:
        for path in files[root]:
            relative = path.relative_to(root)
            name = relative.as_posix()
            source = path.relative_to(target_root).as_posix()
            suffix = path.suffix.lower()
            stdlib = root == stdlib_root
            flet = root == dependency_root and relative.parts[0] == "flet"
            filesystem_resource = root == application_root and (
                relative.parts[0] == "prompts" or relative.parts[:2] == ("puripuly_heart", "data")
            )
            if suffix in {".pyd", ".dll", ".so", ".dylib", ".exe"} and (
                stdlib
                or flet
                or (root == application_root and not (suffix == ".dll" and filesystem_resource))
            ):
                raise ValueError(f"packed package contains unsupported native module: {path}")
            if root == application_root and (
                name == "_native_dependencies.index" or relative.parts[0] == "_native_dependencies"
            ):
                raise ValueError(f"reserved Python archive input: {name}")
            if suffix == ".pyc":
                if stdlib and source in upstream:
                    add(name, path, upstream[source], stdlib=True)
                else:
                    try:
                        original = (
                            Path(importlib.util.source_from_cache(str(path)))
                            if "__pycache__" in relative.parts
                            else path.with_suffix(".py")
                        )
                    except ValueError as exc:
                        raise ValueError(f"unsupported bytecode cache: {path}") from exc
                    if not original.is_file() or (
                        root == application_root
                        and "__pycache__" not in relative.parts
                        and path != application_root / "product_bootstrap.pyc"
                    ):
                        raise ValueError(
                            f"packed package contains unsupported sourceless module: {path}"
                        )
                cleanup.append(path)
                continue
            if "__pycache__" in relative.parts:
                if suffix == ".py":
                    raise ValueError(
                        f"packed package contains source inside a cache directory: {path}"
                    )
                continue
            if suffix == ".py":
                entry = compiled[source]
                expected = target_root.joinpath(*PurePosixPath(entry["path"]).parts)
                if path == application_root / "product_bootstrap.py":
                    cleanup.append(path)
                    continue
                member = str(PurePosixPath(name).with_suffix(".pyc"))
                if root == dependency_root and not flet:
                    add("_native_dependencies/" + member, expected, entry)
                    archived_sources.add(source)
                    continue
                add(member, expected, {**entry, "source_member": name}, stdlib=stdlib)
                archived_sources.add(source)
                cleanup.append(path)
            elif root == application_root or (root == dependency_root and not flet):
                continue
            add(
                name,
                path,
                {
                    "source": source,
                    "sha256": _sha256(path),
                    "provenance": "build-input",
                },
                stdlib=stdlib,
            )
            if stdlib:
                cleanup.append(path)
    index = marshal.dumps({"version": 1, "modules": _dependency_modules(dependency_root)})
    add(
        "_native_dependencies.index",
        index,
        {
            "sha256": hashlib.sha256(index).hexdigest(),
            "provenance": "build-derived-index",
        },
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    stdlib_temporary = stdlib_destination.with_suffix(stdlib_destination.suffix + ".tmp")
    try:
        archive_record = _write_archive(temporary, members, records, target_root)
        stdlib_record = _write_archive(
            stdlib_temporary, stdlib_members, stdlib_records, target_root
        )
        if _archive_members(temporary, dependency_root=dependency_root) != members.keys():
            raise ValueError("Python archive differs from staged inputs")
        if _archive_members(stdlib_temporary, stdlib=True) != stdlib_members.keys():
            raise ValueError("standard library archive differs from staged inputs")
        archive_record["path"] = destination.relative_to(target_root).as_posix()
        stdlib_record["path"] = stdlib_destination.relative_to(target_root).as_posix()
        os.replace(stdlib_temporary, stdlib_destination)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
        stdlib_temporary.unlink(missing_ok=True)
    bootstrap = application_root / "product_bootstrap.pyc"
    for path in set(cleanup):
        if path != bootstrap:
            path.unlink()
    shutil.rmtree(flet_root)
    for root in input_roots:
        for directory in sorted(root.rglob("__pycache__"), reverse=True):
            shutil.rmtree(directory)
    return {
        **evidence,
        "bytecode": [
            entry for entry in evidence["bytecode"] if entry["source"] not in archived_sources
        ],
        "stdlib_bytecode": [],
        "archive": archive_record,
        "stdlib_archive": stdlib_record,
    }


def _validate_python_archive(target_root: Path, layout: NativeArtifactLayout) -> dict[str, Any]:
    application_root = layout.resolve(target_root, "application_root")
    dependency_root = layout.resolve(target_root, "dependency_root")
    stdlib_root = layout.resolve(target_root, "stdlib_root")
    archive_path = layout.resolve(target_root, "python_archive")
    stdlib_path = layout.resolve(target_root, "stdlib_archive")
    members = _archive_members(archive_path, dependency_root=dependency_root)
    stdlib_members = _archive_members(stdlib_path, stdlib=True)
    loose = [
        path
        for root in (application_root, dependency_root, stdlib_root)
        for path in root.rglob("*")
        if path.is_file()
        and (
            (path.suffix.lower() == ".pyc" and path != application_root / "product_bootstrap.pyc")
            or (path.suffix.lower() == ".py" and root != dependency_root)
            or root == stdlib_root
        )
    ]
    if loose or (dependency_root / "flet").exists():
        raise ValueError(f"native artifact contains parallel loose archived code: {loose}")
    if not (application_root / "product_bootstrap.pyc").is_file():
        raise FileNotFoundError("native artifact lacks standalone product_bootstrap.pyc")
    return {
        "path": archive_path.relative_to(target_root.resolve()).as_posix(),
        "sha256": _sha256(archive_path),
        "member_count": len(members),
        "stdlib_archive": {
            "path": stdlib_path.relative_to(target_root.resolve()).as_posix(),
            "sha256": _sha256(stdlib_path),
            "member_count": len(stdlib_members),
        },
    }


def _validate_bundle_evidence(
    target_root: Path, layout: NativeArtifactLayout, evidence: dict[str, Any]
) -> None:
    target_root = target_root.resolve()
    if evidence.get("python_optimize") != 0 or evidence.get("stdlib_bytecode") != []:
        raise ValueError(
            "Python archive evidence must use optimization 0 and own deployed stdlib bytecode"
        )
    deployed = _validate_python_archive(target_root, layout)
    for key, deployed_record, root in (
        ("archive", deployed, layout.resolve(target_root, "application_root")),
        ("stdlib_archive", deployed["stdlib_archive"], layout.resolve(target_root, "stdlib_root")),
    ):
        record = evidence.get(key)
        if not isinstance(record, dict) or any(
            record.get(field) != deployed_record[field] for field in ("path", "sha256")
        ):
            raise ValueError(f"Python archive evidence does not match deployed archive: {key}")
        with zipfile.ZipFile(target_root / record["path"]) as archive:
            members = set(archive.namelist())
            records = record["members"]
            if len(records) != len(members) or {entry["path"] for entry in records} != members:
                raise ValueError("Python archive evidence does not own every member")
            for entry in records:
                name = entry["path"]
                data = archive.read(name) if name.endswith(".pyc") else None
                digest = hashlib.sha256(data) if data is not None else hashlib.sha256()
                if data is None:
                    with archive.open(name) as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                if entry.get("sha256") != digest.hexdigest():
                    raise ValueError(f"Python archive member evidence mismatch: {name}")
                if name.endswith("/"):
                    if entry.get("provenance") != "build-derived-directory" or "source" in entry:
                        raise ValueError(f"Python archive directory evidence mismatch: {name}")
                    continue
                if name == "_native_dependencies.index":
                    if entry.get("provenance") != "build-derived-index":
                        raise ValueError("dependency index provenance evidence mismatch")
                    continue
                if not name.endswith(".pyc"):
                    expected_input = (root / name).relative_to(target_root).as_posix()
                    if key == "archive" and name.startswith("flet/"):
                        expected_input = (
                            (layout.resolve(target_root, "dependency_root") / name)
                            .relative_to(target_root)
                            .as_posix()
                        )
                    if (
                        entry.get("provenance") != "build-input"
                        or entry.get("source") != expected_input
                    ):
                        raise ValueError(f"Python archive input evidence mismatch: {name}")
                    continue
                dependency = name.startswith("_native_dependencies/")
                source_name = str(PurePosixPath(name).with_suffix(".py"))
                if dependency:
                    relative = PurePosixPath(source_name).relative_to("_native_dependencies")
                    source = layout.resolve(target_root, "dependency_root").joinpath(
                        *relative.parts
                    )
                    source_data = source.read_bytes()
                    expected_source = source.relative_to(target_root).as_posix()
                    if "source_member" in entry:
                        raise ValueError(
                            f"dependency source evidence must own retained source: {name}"
                        )
                else:
                    source_data = archive.read(source_name) if source_name in members else None
                    expected_source = (root / source_name).relative_to(target_root).as_posix()
                    if key == "archive" and name.startswith("flet/"):
                        expected_source = (
                            (layout.resolve(target_root, "dependency_root") / source_name)
                            .relative_to(target_root)
                            .as_posix()
                        )
                if source_data is None:
                    if (
                        key != "stdlib_archive"
                        or entry.get("provenance") != "trusted-upstream-sourceless"
                        or entry.get("optimization") != "upstream-unspecified"
                        or entry.get("invalidation_mode") != _validate_bytecode(data, name)
                        or entry.get("input_path")
                        != (root / name).relative_to(target_root).as_posix()
                        or "source" in entry
                        or "source_sha256" in entry
                        or "source_member" in entry
                    ):
                        raise ValueError(f"upstream bytecode evidence mismatch: {name}")
                else:
                    if (
                        entry.get("source") != expected_source
                        or (not dependency and entry.get("source_member") != source_name)
                        or entry.get("source_sha256") != hashlib.sha256(source_data).hexdigest()
                        or entry.get("optimization") != "0"
                        or entry.get("invalidation_mode") != "unchecked-hash"
                        or entry.get("provenance") != "build-derived"
                    ):
                        raise ValueError(f"Python archive bytecode evidence mismatch: {name}")
                    _validate_compiled_member(data, source_data, name)
    bootstrap = layout.resolve(target_root, "application_root") / "product_bootstrap.pyc"
    loose = evidence["bytecode"]
    if len(loose) != 1 or loose[0]["path"] != bootstrap.relative_to(target_root).as_posix():
        raise ValueError("loose bytecode evidence must own only standalone product_bootstrap.pyc")
    entry = loose[0]
    if (
        _sha256(bootstrap) != entry.get("sha256")
        or entry.get("source") != bootstrap.with_suffix(".py").relative_to(target_root).as_posix()
        or not isinstance(entry.get("source_sha256"), str)
        or len(entry["source_sha256"]) != 64
        or entry.get("optimization") != "0"
        or entry.get("invalidation_mode") != "unchecked-hash"
        or entry.get("provenance") != "build-derived"
        or _validate_bytecode(bootstrap.read_bytes(), entry["path"]) != "unchecked-hash"
    ):
        raise ValueError("loose bytecode compile evidence mismatch")


def validate_compliance(target_root: Path, repo_root: Path, soxr_manifest: Path) -> dict[str, Any]:
    from puripuly_heart.release_evidence.release_identity import (
        verify_packaged_license_payloads,
        verify_soxr_packaging,
    )

    return {
        "licenses": verify_packaged_license_payloads(
            target_root,
            application_root="app",
            dependency_root="site-packages",
        ),
        "soxr": verify_soxr_packaging(target_root, repo_root, soxr_manifest),
    }


def render_template(
    template_root: Path,
    overlay_root: Path,
    layout_path: Path,
    python_bootstrap_path: Path,
) -> dict[str, str]:
    layout = NativeArtifactLayout.load(layout_path)
    runner_root = template_root / "{{cookiecutter.out_dir}}" / "windows" / "runner"
    shutil.copy2(overlay_root / "windows" / "runner" / "main.cpp", runner_root / "main.cpp")
    (runner_root / "native_layout.generated.h").write_text(
        layout.render_cpp_header(), encoding="utf-8", newline="\n"
    )
    lib_root = template_root / "{{cookiecutter.out_dir}}" / "lib"
    bootstrap = python_bootstrap_path.read_text(encoding="utf-8")
    dart_bootstrap = "const pythonScript = r'''\n" + bootstrap + "\n''';\n"
    (lib_root / "python.dart").write_text(dart_bootstrap, encoding="utf-8", newline="\n")

    runtime_path = lib_root / "native_runtime.dart"
    runtime = runtime_path.read_text(encoding="utf-8").replace(
        "import 'dart:io';",
        "import 'dart:ffi';\nimport 'dart:io';",
        1,
    )
    start = runtime.index("Future<String?> runPython({")
    end = runtime.index("\n}\n", start) + 3
    replacement = """typedef _SetEnvironmentVariableWNative = Int32 Function(
  Pointer<Uint16>,
  Pointer<Uint16>,
);
typedef _SetEnvironmentVariableWDart = int Function(
  Pointer<Uint16>,
  Pointer<Uint16>,
);
typedef _MallocNative = Pointer<Void> Function(IntPtr);
typedef _MallocDart = Pointer<Void> Function(int);
typedef _FreeNative = Void Function(Pointer<Void>);
typedef _FreeDart = void Function(Pointer<Void>);

void _clearNativeProductArgumentsTransport() {
  const name = "PURIPULY_HEART_NATIVE_ARGV_JSON";
  var runtime = DynamicLibrary.open("ucrtbase.dll");
  var malloc = runtime.lookupFunction<_MallocNative, _MallocDart>("malloc");
  var free = runtime.lookupFunction<_FreeNative, _FreeDart>("free");
  var allocation = malloc((name.length + 2) * 2);
  if (allocation.address == 0) {
    throw StateError("Could not allocate native argv transport cleanup buffer");
  }
  try {
    var nameBuffer = allocation.cast<Uint16>();
    var units = nameBuffer.asTypedList(name.length + 2);
    units.setRange(0, name.length, name.codeUnits);
    units[name.length] = 0;
    units[name.length + 1] = 0;
    var setEnvironmentVariable = DynamicLibrary.open("kernel32.dll")
        .lookupFunction<
          _SetEnvironmentVariableWNative,
          _SetEnvironmentVariableWDart
        >("SetEnvironmentVariableW");
    if (setEnvironmentVariable(nameBuffer, nullptr) == 0) {
      throw StateError("Could not clear native product argv transport");
    }
    var emptyBuffer = nameBuffer.elementAt(name.length + 1);
    var clearRuntimeEnvironment =
        runtime.lookupFunction<
          _SetEnvironmentVariableWNative,
          _SetEnvironmentVariableWDart
        >("_wputenv_s");
    if (clearRuntimeEnvironment(nameBuffer, emptyBuffer) != 0) {
      throw StateError("Could not clear native product argv runtime cache");
    }
  } finally {
    free(allocation);
  }
}

const int errorExitCode = 255;

Future<String?> runPython({
  required String moduleName,
  required String appDir,
  required String outLogFilename,
  required Map<String, String> environmentVariables,
  required List<String> args,
}) async {
  var encodedProductArgs =
      environmentVariables.remove("PURIPULY_HEART_NATIVE_ARGV_JSON");
  if (encodedProductArgs == null) {
    throw StateError("Native product argv transport is missing");
  }
  var decodedProductArgs = jsonDecode(encodedProductArgs);
  if (decodedProductArgs is! List ||
      decodedProductArgs.any((value) => value is! String)) {
    throw const FormatException("Invalid native product argv payload");
  }
  var productArgs = decodedProductArgs.cast<String>();
  _clearNativeProductArgumentsTransport();
  var script = pythonScript
      .replaceAll('{module_name}', jsonEncode(moduleName))
      .replaceAll('{argv}', jsonEncode(productArgs))
      .replaceAll('{host_executable}', jsonEncode(Platform.resolvedExecutable))
      .replaceAll('{error_exit_code}', errorExitCode.toString());
  var completer = Completer<String?>();
  var exitBytes = <int>[];
  const maximumExitPayloadBytes = 131072;
  StreamSubscription<Uint8List>? exitSub;
  var finishing = false;

  Future<void> finish() async {
    if (completer.isCompleted || finishing) {
      return;
    }
    finishing = true;
    await exitSub?.cancel();
    try {
      var payload = jsonDecode(utf8.decode(exitBytes)) as Map<String, dynamic>;
      var exitCode = payload["code"] as int;
      var error = payload["error"] as String? ?? "";
      if (exitCode == errorExitCode) {
        completer.complete(error);
      } else {
        DartBridge.instance.hardExit(exitCode);
        exit(exitCode);
      }
    } catch (error) {
      completer.complete("Embedded Python returned an invalid exit payload: $error");
    }
  }

  exitSub = _exitBridge!.messages.listen(
    (data) {
      exitBytes.addAll(data);
      if (exitBytes.length > maximumExitPayloadBytes) {
        exitBytes.removeRange(0, exitBytes.length - maximumExitPayloadBytes);
      }
      finish();
    },
    onError: (error) async {
      await exitSub?.cancel();
      if (!completer.isCompleted) {
        completer.complete("Embedded Python exit bridge failed: $error");
      }
    },
    onDone: finish,
    cancelOnError: false,
  );

  SeriousPython.runProgram(
    path.join(appDir, "$moduleName.pyc"),
    script: script,
    modulePaths: [
      path.joinAll([File(Platform.resolvedExecutable).parent.path, __PYTHON_ARCHIVE__]),
      path.joinAll([File(Platform.resolvedExecutable).parent.path, __STDLIB_ARCHIVE__]),
    ],
    environmentVariables: environmentVariables,
  );
  return completer.future;
}
""".replace(
        "__PYTHON_ARCHIVE__",
        ", ".join(
            json.dumps(part) for part in PurePosixPath(layout.values["python_archive"]).parts
        ),
    ).replace("__STDLIB_ARCHIVE__", json.dumps(layout.values["stdlib_archive"]))
    runtime_path.write_text(
        runtime[:start] + replacement + runtime[end:], encoding="utf-8", newline="\n"
    )

    main_path = lib_root / "main.dart"
    main = main_path.read_text(encoding="utf-8")
    old_assets = '    assetsDir = path.join(appDir, "assets");'
    new_assets = '    assetsDir = path.join(appDir, "puripuly_heart", "data");'
    if main.count(old_assets) != 1:
        raise ValueError("pinned Flet main.dart asset root changed")
    main = main.replace(old_assets, new_assets)

    old_storage = """    var appDataPath = path.join(
        (await path_provider.getApplicationSupportDirectory()).path, "data");
    if (!await Directory(appDataPath).exists()) {
      await Directory(appDataPath).create(recursive: true);
    }
    Directory.current = appDataPath;

    // FLET_APP_STORAGE_CACHE — regenerable; the OS may purge it.
    var appCachePath = (await path_provider.getApplicationCacheDirectory()).path;
    // FLET_APP_STORAGE_TEMP — volatile OS temp; may vanish between launches.
    var appTempPath = (await path_provider.getTemporaryDirectory()).path;

    environmentVariables.putIfAbsent("FLET_APP_STORAGE_DATA", () => appDataPath);
    environmentVariables.putIfAbsent(
        "FLET_APP_STORAGE_CACHE", () => appCachePath);
    environmentVariables.putIfAbsent("FLET_APP_STORAGE_TEMP", () => appTempPath);
"""
    new_storage = """    var appDataPath = environmentVariables["FLET_APP_STORAGE_DATA"] ??
        path.join(
            (await path_provider.getApplicationSupportDirectory()).path, "data");
    // FLET_APP_STORAGE_CACHE — regenerable; the OS may purge it.
    var appCachePath = environmentVariables["FLET_APP_STORAGE_CACHE"] ??
        (await path_provider.getApplicationCacheDirectory()).path;
    // FLET_APP_STORAGE_TEMP — volatile OS temp; may vanish between launches.
    var appTempPath = environmentVariables["FLET_APP_STORAGE_TEMP"] ??
        (await path_provider.getTemporaryDirectory()).path;

    appDataPath = appDataPath.isEmpty
        ? appDataPath
        : Directory(appDataPath).absolute.path;
    appCachePath = appCachePath.isEmpty
        ? appCachePath
        : Directory(appCachePath).absolute.path;
    appTempPath = appTempPath.isEmpty
        ? appTempPath
        : Directory(appTempPath).absolute.path;

    for (var directory in [appDataPath, appCachePath, appTempPath]) {
      var selectedDirectory = Directory(directory);
      if (!await selectedDirectory.exists()) {
        await selectedDirectory.create(recursive: true);
      }
    }
    Directory.current = appDataPath;

    environmentVariables["FLET_APP_STORAGE_DATA"] = appDataPath;
    environmentVariables["FLET_APP_STORAGE_CACHE"] = appCachePath;
    environmentVariables["FLET_APP_STORAGE_TEMP"] = appTempPath;
"""
    if main.count(old_storage) != 1:
        raise ValueError("pinned Flet main.dart storage block changed")
    main = main.replace(old_storage, new_storage)

    old_console = """    outLogFilename = path.join(appCachePath, "console.log");
    environmentVariables.putIfAbsent("FLET_APP_CONSOLE", () => outLogFilename);

"""
    if main.count(old_console) != 1:
        raise ValueError("pinned Flet main.dart console block changed")
    main_path.write_text(main.replace(old_console, ""), encoding="utf-8", newline="\n")
    return {
        "layout_sha256": _sha256(layout_path),
        "python_bootstrap_sha256": _sha256(python_bootstrap_path),
        "runner_sha256": _sha256(runner_root / "main.cpp"),
    }


def create_manifest(
    target_root: Path,
    layout_path: Path,
    provenance_path: Path,
    bytecode_path: Path,
    destination: Path,
) -> dict[str, Any]:
    layout = NativeArtifactLayout.load(layout_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    bytecode = json.loads(bytecode_path.read_text(encoding="utf-8"))
    _validate_bundle_evidence(target_root, layout, bytecode)
    destination = destination.resolve()
    inventory = []
    for path in sorted(target_root.rglob("*")):
        if path.is_file() and path.resolve() != destination:
            inventory.append(
                {
                    "path": path.relative_to(target_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    manifest = {
        "schema": _MANIFEST_SCHEMA,
        "layout_schema": _LAYOUT_SCHEMA,
        "layout_sha256": _sha256(layout.source),
        "provenance": provenance,
        "bytecode": bytecode,
        "inventory": inventory,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _write_json(path: Path | None, value: Any) -> None:
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(text, end="")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-inputs")
    verify.add_argument("--spec", type=Path, required=True)
    verify.add_argument("--input-root", type=Path, required=True)
    verify.add_argument("--output", type=Path)
    filter_parser = commands.add_parser("filter-requirements")
    filter_parser.add_argument("source", type=Path)
    filter_parser.add_argument("destination", type=Path)
    metadata = commands.add_parser("stage-product-metadata")
    metadata.add_argument("--site-packages", type=Path, required=True)
    metadata.add_argument("--pyproject", type=Path, required=True)
    metadata.add_argument("--output", type=Path)
    sounddevice = commands.add_parser("stage-sounddevice-runtime")
    sounddevice.add_argument("--site-packages", type=Path, required=True)
    sounddevice.add_argument("--output", type=Path)
    vc_runtime = commands.add_parser("stage-vc-runtime")
    vc_runtime.add_argument("--target-root", type=Path, required=True)
    vc_runtime.add_argument("--cmake-build-dir", type=Path, required=True)
    vc_runtime.add_argument("--output", type=Path, required=True)
    wheel = commands.add_parser("finalize-soxr-wheel")
    wheel.add_argument("--wheel", type=Path, required=True)
    wheel.add_argument("--dll", type=Path, required=True)
    wheel.add_argument("--output", type=Path, required=True)
    render = commands.add_parser("render-template")
    render.add_argument("--template-root", type=Path, required=True)
    render.add_argument("--overlay-root", type=Path, required=True)
    render.add_argument("--layout", type=Path, required=True)
    render.add_argument("--python-bootstrap", type=Path, required=True)
    render.add_argument("--output", type=Path)
    compile_parser = commands.add_parser("compile-runtime")
    compile_parser.add_argument("--target-root", type=Path, required=True)
    compile_parser.add_argument("--layout", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    bundle = commands.add_parser("bundle-runtime")
    bundle.add_argument("--target-root", type=Path, required=True)
    bundle.add_argument("--layout", type=Path, required=True)
    bundle.add_argument("--bytecode", type=Path, required=True)
    bundle.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate-target")
    validate.add_argument("--target-root", type=Path, required=True)
    validate.add_argument("--layout", type=Path, required=True)
    validate.add_argument("--requirements", type=Path, required=True)
    validate.add_argument("--vc-runtime", type=Path, required=True)
    validate.add_argument("--output", type=Path)
    compliance = commands.add_parser("validate-compliance")
    compliance.add_argument("--target-root", type=Path, required=True)
    compliance.add_argument("--repo-root", type=Path, required=True)
    compliance.add_argument("--soxr-manifest", type=Path, required=True)
    compliance.add_argument("--output", type=Path)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--target-root", type=Path, required=True)
    manifest.add_argument("--layout", type=Path, required=True)
    manifest.add_argument("--provenance", type=Path, required=True)
    manifest.add_argument("--bytecode", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify-inputs":
        _write_json(args.output, verify_upstream_inputs(args.spec, args.input_root))
    elif args.command == "filter-requirements":
        filter_requirements(args.source, args.destination)
    elif args.command == "stage-product-metadata":
        _write_json(
            args.output,
            stage_product_metadata(args.site_packages, args.pyproject),
        )
    elif args.command == "stage-sounddevice-runtime":
        _write_json(
            args.output,
            stage_sounddevice_portaudio_runtime(args.site_packages),
        )
    elif args.command == "stage-vc-runtime":
        _write_json(args.output, stage_vc_runtime(args.target_root, args.cmake_build_dir))
    elif args.command == "finalize-soxr-wheel":
        _write_json(None, finalize_soxr_wheel(args.wheel, args.dll, args.output))
    elif args.command == "render-template":
        _write_json(
            args.output,
            render_template(
                args.template_root, args.overlay_root, args.layout, args.python_bootstrap
            ),
        )
    elif args.command == "compile-runtime":
        _write_json(args.output, compile_runtime(args.target_root, args.layout))
    elif args.command == "bundle-runtime":
        _write_json(args.output, bundle_runtime(args.target_root, args.layout, args.bytecode))
    elif args.command == "validate-target":
        _write_json(
            args.output,
            validate_target(args.target_root, args.layout, args.requirements, args.vc_runtime),
        )
    elif args.command == "validate-compliance":
        _write_json(
            args.output,
            validate_compliance(args.target_root, args.repo_root, args.soxr_manifest),
        )
    elif args.command == "manifest":
        create_manifest(args.target_root, args.layout, args.provenance, args.bytecode, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
