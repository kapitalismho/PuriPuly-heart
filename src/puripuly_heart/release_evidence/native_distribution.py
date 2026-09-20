from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import py_compile
import shutil
import sys
import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_LAYOUT_SCHEMA = "puripuly-heart/native-artifact-layout/v1"
_MANIFEST_SCHEMA = "puripuly-heart/native-artifact-manifest/v1"
_REQUIRED_DISTRIBUTIONS = frozenset(
    {
        "filelock",
        "pywin32-ctypes",
        "proc-tap",
        "psutil",
        "puripuly-heart",
        "repath",
        "six",
        "typing-extensions",
    }
)
_FORBIDDEN_DISTRIBUTIONS = frozenset({"flet-desktop", "flet-cli", "pyinstaller"})
_FORCED_WINDOWS_REQUIREMENTS = frozenset({"proc-tap", "psutil"})
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
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
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
            "python_executable",
            "application_root",
            "dependency_root",
            "stdlib_root",
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
        if values["python_executable"] != "python.exe":
            raise ValueError("native Python executable must be python.exe")
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
            "dependency_root": "kDependencyRoot",
            "stdlib_root": "kStdlibRoot",
            "extension_dll_root": "kExtensionDllRoot",
        }
        lines = ["#pragma once", "", "namespace puripuly_layout {"]
        for key, cpp_name in names.items():
            value = self.values[key].replace("/", "\\")
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
        name = _canonical_name(block[0].split("==", 1)[0].strip())
        if name in excluded:
            removed.append(name)
            continue
        if name in _FORCED_WINDOWS_REQUIREMENTS and " ; " in block[0]:
            continuation = " \\" if block[0].rstrip().endswith("\\") else ""
            block[0] = block[0].split(" ; ", 1)[0].rstrip() + continuation
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


def _distribution_names(site_packages: Path) -> set[str]:
    names: set[str] = set()
    for distribution in importlib.metadata.distributions(path=[str(site_packages)]):
        name = distribution.metadata.get("Name")
        if not name:
            raise ValueError(f"distribution has no Name metadata: {distribution._path}")
        canonical = _canonical_name(name)
        if canonical in names:
            raise ValueError(f"duplicate distribution metadata: {canonical}")
        names.add(canonical)
    return names


def validate_target(
    target_root: Path, layout_path: Path, expected_count: int = 81
) -> dict[str, Any]:
    layout = NativeArtifactLayout.load(layout_path)
    required_paths = {
        key: layout.resolve(target_root, key)
        for key in (
            "host_executable",
            "python_executable",
            "application_root",
            "dependency_root",
            "stdlib_root",
            "extension_dll_root",
            "overlay_executable",
            "gpu_worker_executable",
            "openvr_dll",
        )
    }
    missing = sorted(str(path) for path in required_paths.values() if not path.exists())
    if missing:
        raise FileNotFoundError(f"native artifact is incomplete: {missing}")
    site_packages = required_paths["dependency_root"]
    names = _distribution_names(site_packages)
    if len(names) != expected_count:
        raise ValueError(
            f"native dependency closure must contain {expected_count} distributions, got {len(names)}"
        )
    forbidden = sorted(names & _FORBIDDEN_DISTRIBUTIONS)
    if forbidden:
        raise ValueError(f"native dependency closure includes forbidden distributions: {forbidden}")
    absent = sorted(_REQUIRED_DISTRIBUTIONS - names)
    if absent:
        raise ValueError(
            f"native dependency closure is missing required transitive distributions: {absent}"
        )
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
    return {
        "distribution_count": len(names),
        "distributions": sorted(names),
        "python_executable_sha256": python_digest,
        "soxr_runtime": soxr_runtime,
        "portaudio_runtime": portaudio_runtime,
        "paths": {
            key: str(value.relative_to(target_root)) for key, value in required_paths.items()
        },
    }


def compile_application(application_root: Path) -> dict[str, Any]:
    if sys.flags.optimize != 0:
        raise RuntimeError(
            "native application bytecode must be compiled by an optimization-0 interpreter"
        )
    sources = sorted(application_root.rglob("*.py"))
    failures: list[str] = []
    outputs: list[dict[str, str]] = []
    for source in sources:
        destination = (
            application_root / "product_bootstrap.pyc"
            if source == application_root / "product_bootstrap.py"
            else Path(importlib.util.cache_from_source(str(source), optimization=""))
        )
        try:
            py_compile.compile(
                str(source),
                cfile=str(destination),
                dfile=source.relative_to(application_root.parent).as_posix(),
                doraise=True,
                optimize=0,
                invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH,
            )
        except py_compile.PyCompileError as exc:
            failures.append(str(exc))
            continue
        outputs.append(
            {
                "path": destination.relative_to(application_root.parent).as_posix(),
                "source": source.relative_to(application_root.parent).as_posix(),
                "sha256": _sha256(destination),
                "optimization": "0",
                "invalidation_mode": "checked-hash",
                "provenance": "build-derived",
            }
        )
    if failures:
        raise RuntimeError("\n".join(failures))
    return {"python": sys.version, "python_optimize": sys.flags.optimize, "bytecode": outputs}


def validate_compliance(target_root: Path, repo_root: Path, soxr_manifest: Path) -> dict[str, Any]:
    from puripuly_heart.release_evidence.release_identity import (
        PACKAGED_LICENSE_PATHS,
        verify_packaged_license_payloads,
        verify_soxr_packaging,
    )

    return {
        "licenses": verify_packaged_license_payloads(
            target_root,
            application_root="app",
            dependency_root="site-packages",
            license_paths=tuple(
                path for path in PACKAGED_LICENSE_PATHS if not path.startswith("scipy/")
            ),
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
    environmentVariables: environmentVariables,
  );
  return completer.future;
}
"""
    runtime_path.write_text(
        runtime[:start] + replacement + runtime[end:], encoding="utf-8", newline="\n"
    )

    main_path = lib_root / "main.dart"
    main = main_path.read_text(encoding="utf-8")
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
    compile_parser = commands.add_parser("compile-app")
    compile_parser.add_argument("--application-root", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate-target")
    validate.add_argument("--target-root", type=Path, required=True)
    validate.add_argument("--layout", type=Path, required=True)
    validate.add_argument("--expected-count", type=int, default=81)
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
    elif args.command == "finalize-soxr-wheel":
        _write_json(None, finalize_soxr_wheel(args.wheel, args.dll, args.output))
    elif args.command == "render-template":
        _write_json(
            args.output,
            render_template(
                args.template_root, args.overlay_root, args.layout, args.python_bootstrap
            ),
        )
    elif args.command == "compile-app":
        _write_json(args.output, compile_application(args.application_root))
    elif args.command == "validate-target":
        _write_json(
            args.output, validate_target(args.target_root, args.layout, args.expected_count)
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
