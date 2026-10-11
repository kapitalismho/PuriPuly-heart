from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_OS_DLLS = frozenset(
    "advapi32.dll avrt.dll bcrypt.dll bcryptprimitives.dll cabinet.dll cfgmgr32.dll "
    "combase.dll comctl32.dll comdlg32.dll crypt32.dll cryptbase.dll cryptsp.dll "
    "d2d1.dll d3d9.dll d3d11.dll d3d12.dll dbghelp.dll dnsapi.dll dwmapi.dll "
    "dwrite.dll dxgi.dll dxva2.dll gdi32.dll gdiplus.dll hid.dll imm32.dll "
    "iphlpapi.dll kernel32.dll kernelbase.dll mf.dll mfplat.dll mfreadwrite.dll "
    "mpr.dll msvcrt.dll mswsock.dll netapi32.dll normaliz.dll ntdll.dll ole32.dll "
    "oleacc.dll oleaut32.dll opengl32.dll pdh.dll powrprof.dll propsys.dll psapi.dll "
    "rpcrt4.dll secur32.dll setupapi.dll shcore.dll shell32.dll shlwapi.dll "
    "ucrtbase.dll uiautomationcore.dll urlmon.dll user32.dll userenv.dll usp10.dll "
    "uxtheme.dll version.dll winhttp.dll wininet.dll winmm.dll winspool.drv "
    "wintrust.dll wldap32.dll ws2_32.dll wtsapi32.dll".split()
)
_GPU_DRIVER_DLLS = frozenset({"vulkan-1.dll", "nvcuda.dll", "nvapi64.dll", "opencl.dll"})
_VC_NAME = re.compile(r"(?:msvcp|msvcr|vcruntime|vccorlib|concrt|vcomp|vcamp)\d+(?:_\w+)?d?\.dll")
_DEBUG_DLL = re.compile(r"(?:msvcp|msvcr|vcruntime|vccorlib|concrt|vcomp|vcamp)\d+(?:_\w+)?d\.dll")
_RUNTIME_SCHEMA = "puripuly-heart/native-vc-runtime/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PeImage:
    path: Path
    imports: tuple[tuple[str, str], ...]
    version: str
    company: str
    debug: bool


def read_pe_image(path: Path) -> PeImage:
    try:
        import pefile
    except ImportError as exc:
        raise RuntimeError(
            "pefile from the existing Windows build environment is required"
        ) from exc
    try:
        pe = pefile.PE(str(path), fast_load=True)
    except pefile.PEFormatError as exc:
        raise ValueError(f"invalid shipped PE image: {path}: {exc}") from exc
    try:
        if pe.FILE_HEADER.Machine != 0x8664:
            raise ValueError(f"native artifact requires x64 PE images: {path}")
        directories = (
            ("IMAGE_DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_IMPORT", "import"),
            ("IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT", "delay_import"),
        )
        pe.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY[key] for key, _, _ in directories]
            + [pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"]]
        )
        imports = []
        for key, attribute, kind in directories:
            index = pefile.DIRECTORY_ENTRY[key]
            if pe.OPTIONAL_HEADER.DATA_DIRECTORY[index].VirtualAddress and not hasattr(
                pe, attribute
            ):
                raise ValueError(f"unreadable {kind} table: {path}")
            for entry in getattr(pe, attribute, []):
                name = entry.dll.decode("ascii").lower()
                if not name or "/" in name or "\\" in name or ":" in name:
                    raise ValueError(f"invalid imported DLL name in {path}: {name!r}")
                imports.append((kind, name))
        fixed = getattr(pe, "VS_FIXEDFILEINFO", [])
        version = ""
        debug = False
        if fixed:
            info = fixed[0]
            version = ".".join(
                str(value)
                for value in (
                    info.FileVersionMS >> 16,
                    info.FileVersionMS & 0xFFFF,
                    info.FileVersionLS >> 16,
                    info.FileVersionLS & 0xFFFF,
                )
            )
            debug = bool(info.FileFlags & info.FileFlagsMask & 1)
        company = ""
        for group in getattr(pe, "FileInfo", []) or []:
            for entry in group:
                for table in getattr(entry, "StringTable", []):
                    value = table.entries.get(b"CompanyName", b"")
                    if value:
                        company = value.decode("utf-8").rstrip("\x00")
        return PeImage(path, tuple(sorted(set(imports))), version, company, debug)
    finally:
        pe.close()


def _images(target_root: Path) -> list[PeImage]:
    images = []
    for path in sorted(target_root.rglob("*")):
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            signature = stream.read(2)
        if signature == b"MZ" or path.suffix.lower() in {".dll", ".exe", ".pyd"}:
            if not path.resolve().is_relative_to(target_root.resolve()):
                raise ValueError(f"PE image escapes native artifact: {path}")
            image = read_pe_image(path)
            images.append(image)
    if not images:
        raise ValueError("native artifact contains no PE images")
    return images


def resolve_msvc_redist(cmake_build_dir: Path) -> dict[str, str]:
    records = list((cmake_build_dir / "CMakeFiles").glob("*/CMakeCXXCompiler.cmake"))
    if len(records) != 1:
        raise ValueError(f"expected one selected CMake C++ compiler record: {records}")
    record = records[0].read_text(encoding="utf-8")
    compiler_match = re.search(r'^set\(CMAKE_CXX_COMPILER "([^"]+)"\)$', record, re.MULTILINE)
    if not compiler_match or 'set(CMAKE_CXX_COMPILER_ID "MSVC")' not in record:
        raise ValueError("native CRT staging requires the selected MSVC compiler")
    compiler = Path(compiler_match[1])
    match = re.fullmatch(
        r"(.+)/VC/Tools/MSVC/(\d+\.\d+\.\d+)/bin/Hostx64/x64/cl\.exe",
        compiler.as_posix(),
        re.IGNORECASE,
    )
    if not match or not compiler.is_file():
        raise ValueError(f"selected compiler is not an installed x64 MSVC toolchain: {compiler}")
    installation = Path(match[1])
    tool_version = match[2]
    version_file = installation / "VC/Auxiliary/Build/Microsoft.VCRedistVersion.default.txt"
    redist_version = version_file.read_text(encoding="utf-8-sig").strip()
    if (
        not re.fullmatch(r"\d+\.\d+\.\d+", redist_version)
        or redist_version.split(".")[:2] != tool_version.split(".")[:2]
    ):
        raise ValueError(
            f"official default redist {redist_version!r} does not match selected MSVC {tool_version}"
        )
    redist = installation / "VC/Redist/MSVC" / redist_version / "x64"
    if not redist.is_dir():
        raise FileNotFoundError(f"official x64 release redist is missing: {redist}")
    return {
        "compiler": str(compiler.resolve()),
        "compiler_record": str(records[0].resolve()),
        "toolchain_version": tool_version,
        "visual_studio_installation": str(installation.resolve()),
        "redist_version_file": str(version_file.resolve()),
        "redist_directory_version": redist_version,
        "redist_root": str(redist.resolve()),
    }


def stage_vc_runtime(target_root: Path, cmake_build_dir: Path) -> dict[str, Any]:
    toolchain = resolve_msvc_redist(cmake_build_dir)
    redist = Path(toolchain["redist_root"])
    crt_directories = sorted(redist.glob("Microsoft.VC*.CRT"))
    if len(crt_directories) != 1:
        raise ValueError(f"expected one official release CRT set: {crt_directories}")
    sources = {path.name.lower(): path for path in crt_directories[0].glob("*.dll")}
    if "msvcp140_1.dll" not in sources:
        raise FileNotFoundError("official CRT set lacks required msvcp140_1.dll")
    excluded = []
    for path in sorted((target_root / "DLLs").glob("*_d.*")):
        if path.suffix.lower() not in {".pyd", ".dll"}:
            continue
        release = path.with_name(path.stem.removesuffix("_d") + path.suffix)
        if not release.is_file():
            raise ValueError(
                f"cannot exclude debug interpreter module without its release pair: {path}"
            )
        excluded.append({"path": path.relative_to(target_root).as_posix(), "sha256": _sha256(path)})
        path.unlink()
    images = _images(target_root)
    needed = {name for image in images for _, name in image.imports if _VC_NAME.fullmatch(name)}
    for name in sorted(needed - sources.keys()):
        matches = list(redist.glob(f"Microsoft.VC*/{name}"))
        if len(matches) != 1:
            raise FileNotFoundError(f"selected official x64 redist cannot supply {name}")
        sources[name] = matches[0]
    if any(not path.resolve().is_relative_to(redist.resolve()) for path in sources.values()):
        raise ValueError("VC runtime source escapes the selected official redist directory")
    source_images = {name: read_pe_image(path) for name, path in sources.items()}
    versions = {image.version for image in source_images.values()}
    if len(versions) != 1 or not next(iter(versions)).startswith(
        ".".join(toolchain["toolchain_version"].split(".")[:2]) + "."
    ):
        raise ValueError(f"official VC runtime set has mismatched file versions: {versions}")
    for name, image in source_images.items():
        if image.company != "Microsoft Corporation" or image.debug or _DEBUG_DLL.fullmatch(name):
            raise ValueError(f"not an official Microsoft release VC runtime: {image.path}")
    modules = []
    for name, source in sorted(sources.items()):
        destinations = {target_root / name}
        destinations.update(image.path for image in images if image.path.name.lower() == name)
        digest = _sha256(source)
        for destination in sorted(destinations):
            if destination.is_relative_to(target_root / "nemotron"):
                if not destination.is_file() or _sha256(destination) != digest:
                    raise ValueError(
                        f"Nemotron CRT differs from selected official release runtime: {destination}"
                    )
            else:
                shutil.copy2(source, destination)
            modules.append(
                {
                    "path": destination.relative_to(target_root).as_posix(),
                    "source": str(source.resolve()),
                    "version": source_images[name].version,
                    "sha256": digest,
                }
            )
    return {
        "schema": _RUNTIME_SCHEMA,
        "toolchain": toolchain,
        "file_version": next(iter(versions)),
        "modules": modules,
        "excluded_debug_interpreter_modules": excluded,
    }


def _wheel_library_directory(image: Path, target_root: Path) -> Path | None:
    relative = image.relative_to(target_root)
    if len(relative.parts) < 3 or relative.parts[0] != "site-packages":
        return None
    package = relative.parts[1]
    init = target_root / "site-packages" / package / "__init__.py"
    libraries = target_root / "site-packages" / f"{package}.libs"
    if not init.is_file() or not libraries.is_dir():
        return None
    tree = ast.parse(init.read_text(encoding="utf-8-sig"), filename=str(init))
    nodes = list(ast.walk(tree))
    expected = ast.parse(
        "os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "
        + repr(f"{package}.libs")
        + "))",
        mode="eval",
    ).body
    variables = set()
    for node in nodes:
        if isinstance(node, ast.NamedExpr):
            targets, value = [node.target], node.value
        elif isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        else:
            continue
        if ast.dump(value) == ast.dump(expected):
            variables.update(target.id for target in targets if isinstance(target, ast.Name))
    registers_directory = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
        and node.func.attr == "add_dll_directory"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id in variables
        for node in nodes
    )
    return libraries if registers_directory else None


def validate_pe_dependencies(target_root: Path, vc_runtime_path: Path) -> dict[str, Any]:
    runtime = json.loads(vc_runtime_path.read_text(encoding="utf-8"))
    if runtime.get("schema") != _RUNTIME_SCHEMA or not runtime.get("modules"):
        raise ValueError("missing official VC runtime staging evidence")
    staged = {}
    for module in runtime["modules"]:
        relative = Path(module["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("VC runtime evidence contains an unsafe artifact path")
        path = target_root / relative
        if not path.is_file() or _sha256(path) != module["sha256"]:
            raise ValueError(f"staged VC runtime is missing or changed: {module['path']}")
        staged[path] = module
    images = _images(target_root)
    by_directory: dict[Path, dict[str, Path]] = {}
    for image in images:
        entries = by_directory.setdefault(image.path.parent, {})
        name = image.path.name.lower()
        if name in entries:
            raise ValueError(f"case-ambiguous native DLL: {image.path}")
        entries[name] = image.path
        if _VC_NAME.fullmatch(name) and image.path not in staged:
            raise ValueError(f"VC runtime has no matched official staging origin: {image.path}")
    evidence = []
    missing = []
    wheel_directories: dict[tuple[str, ...], Path | None] = {}
    for image in images:
        search = [
            image.path.parent,
            target_root,
            target_root / "DLLs",
            target_root / "site-packages",
        ]
        package_key = image.path.relative_to(target_root).parts[:2]
        if package_key not in wheel_directories:
            wheel_directories[package_key] = _wheel_library_directory(image.path, target_root)
        wheel_libraries = wheel_directories[package_key]
        if wheel_libraries is not None:
            search.append(wheel_libraries)
        # The diarizer's ggml build must never resolve against the ASR DLLs.
        nemotron_image = image.path.is_relative_to(target_root / "nemotron")
        imports = []
        for kind, name in image.imports:
            if _DEBUG_DLL.fullmatch(name) or name in {"ucrtbased.dll", "python314_d.dll"}:
                raise ValueError(f"debug native dependency {name} imported by {image.path}")
            if name in _OS_DLLS or name.startswith(("api-ms-win-", "ext-ms-win-")):
                resolution = {"external": "windows_os"}
            elif name in _GPU_DRIVER_DLLS:
                resolution = {"external": "gpu_driver"}
            else:
                resolved = next(
                    (
                        by_directory[directory][name]
                        for directory in search
                        if (
                            not nemotron_image
                            or directory == image.path.parent
                            or _VC_NAME.fullmatch(name)
                        )
                        and name in by_directory.get(directory, {})
                    ),
                    None,
                )
                if resolved is None:
                    missing.append(
                        f"{image.path.relative_to(target_root).as_posix()} [{kind}] -> {name}"
                    )
                    continue
                resolution = {"path": resolved.relative_to(target_root).as_posix()}
                if _VC_NAME.fullmatch(name):
                    resolution["source"] = staged[resolved]["source"]
            imports.append({"kind": kind, "name": name, **resolution})
        evidence.append(
            {
                "path": image.path.relative_to(target_root).as_posix(),
                "sha256": _sha256(image.path),
                "debug_resource_flag": image.debug,
                "search_directories": [
                    path.relative_to(target_root).as_posix() for path in dict.fromkeys(search)
                ],
                "imports": imports,
            }
        )
    if missing:
        raise FileNotFoundError(
            "native PE dependency closure is incomplete:\n" + "\n".join(missing)
        )
    return {"parser": "pefile", "images": evidence, "vc_runtime": runtime}
