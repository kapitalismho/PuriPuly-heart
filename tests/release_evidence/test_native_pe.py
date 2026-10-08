from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from puripuly_heart.release_evidence.native_pe import (
    read_pe_image,
    resolve_msvc_redist,
    stage_vc_runtime,
    validate_pe_dependencies,
)

pytest.importorskip("pefile")


def _version_node(key: str, value: bytes, children: bytes = b"", *, text: bool = False) -> bytes:
    payload = bytearray(struct.pack("<HHH", 0, len(value) // 2 if text else len(value), int(text)))
    payload.extend((key + "\0").encode("utf-16le"))
    payload.extend(b"\0" * (-len(payload) % 4))
    payload.extend(value)
    if children:
        payload.extend(b"\0" * (-len(payload) % 4))
        payload.extend(children)
    struct.pack_into("<H", payload, 0, len(payload))
    return bytes(payload)


def _pe(
    path: Path,
    imports: tuple[str, ...] = (),
    delay_imports: tuple[str, ...] = (),
    *,
    version: str = "14.44.35211.0",
    company: str = "Microsoft Corporation",
    debug: bool = False,
    machine: int = 0x8664,
) -> Path:
    payload = bytearray(0x2200)
    payload[:2] = b"MZ"
    struct.pack_into("<I", payload, 0x3C, 0x80)
    payload[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", payload, 0x84, machine, 1, 0, 0, 0, 240, 0x2022)
    optional = 0x98
    struct.pack_into("<H", payload, optional, 0x20B)
    struct.pack_into("<Q", payload, optional + 24, 0x180000000)
    struct.pack_into("<II", payload, optional + 32, 0x1000, 0x200)
    struct.pack_into("<II", payload, optional + 56, 0x3000, 0x200)
    struct.pack_into("<H", payload, optional + 68, 3)
    struct.pack_into("<I", payload, optional + 108, 16)
    struct.pack_into(
        "<8sIIIIIIHHI",
        payload,
        optional + 240,
        b".rdata\0\0",
        0x2000,
        0x1000,
        0x2000,
        0x200,
        0,
        0,
        0,
        0,
        0x40000040,
    )
    cursor = 0x200

    def allocate(data: bytes) -> tuple[int, int]:
        nonlocal cursor
        cursor = (cursor + 7) & ~7
        offset = cursor
        payload[offset : offset + len(data)] = data
        cursor += len(data)
        return offset, offset - 0x200 + 0x1000

    for names, directory, width in ((imports, 1, 20), (delay_imports, 13, 32)):
        if not names:
            continue
        offset, rva = allocate(bytes(width * (len(names) + 1)))
        struct.pack_into(
            "<II", payload, optional + 112 + directory * 8, rva, width * (len(names) + 1)
        )
        for index, name in enumerate(names):
            _, name_rva = allocate(name.encode("ascii") + b"\0")
            _, thunk_rva = allocate(struct.pack("<QQ", 0x8000000000000001, 0))
            if directory == 1:
                struct.pack_into(
                    "<IIIII", payload, offset + index * width, thunk_rva, 0, 0, name_rva, thunk_rva
                )
            else:
                struct.pack_into(
                    "<IIIIIIII",
                    payload,
                    offset + index * width,
                    1,
                    name_rva,
                    0,
                    thunk_rva,
                    thunk_rva,
                    0,
                    0,
                    0,
                )
    major, minor, build, revision = (int(part) for part in version.split("."))
    fixed = struct.pack(
        "<13I",
        0xFEEF04BD,
        0x10000,
        major << 16 | minor,
        build << 16 | revision,
        major << 16 | minor,
        build << 16 | revision,
        0x3F,
        int(debug),
        0x40004,
        2,
        0,
        0,
        0,
    )
    company_node = _version_node("CompanyName", (company + "\0").encode("utf-16le"), text=True)
    table = _version_node("040904b0", b"", company_node, text=True)
    strings = _version_node("StringFileInfo", b"", table, text=True)
    version_info = _version_node("VS_VERSION_INFO", fixed, strings)
    resource_offset, resource_rva = allocate(bytes(88))
    _, value_rva = allocate(version_info)
    for index in range(3):
        struct.pack_into("<IIHHHH", payload, resource_offset + index * 24, 0, 0, 0, 0, 0, 1)
    struct.pack_into("<II", payload, resource_offset + 16, 16, 0x80000000 | 24)
    struct.pack_into("<II", payload, resource_offset + 40, 1, 0x80000000 | 48)
    struct.pack_into("<II", payload, resource_offset + 64, 1033, 72)
    struct.pack_into("<IIII", payload, resource_offset + 72, value_rva, len(version_info), 0, 0)
    struct.pack_into("<II", payload, optional + 112 + 2 * 8, resource_rva, cursor - resource_offset)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _toolchain(tmp_path: Path) -> tuple[Path, Path]:
    installation = tmp_path / "Visual Studio/BuildTools"
    compiler = installation / "VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe"
    compiler.parent.mkdir(parents=True)
    compiler.write_bytes(b"compiler identity")
    defaults = installation / "VC/Auxiliary/Build/Microsoft.VCRedistVersion.default.txt"
    defaults.parent.mkdir(parents=True)
    defaults.write_text("14.44.35112\n", encoding="utf-8")
    build = tmp_path / "console-build"
    record = build / "CMakeFiles/3.31.10/CMakeCXXCompiler.cmake"
    record.parent.mkdir(parents=True)
    record.write_text(
        f'set(CMAKE_CXX_COMPILER "{compiler.as_posix()}")\n' 'set(CMAKE_CXX_COMPILER_ID "MSVC")\n',
        encoding="utf-8",
    )
    crt = installation / "VC/Redist/MSVC/14.44.35112/x64/Microsoft.VC143.CRT"
    _pe(crt / "vcruntime140.dll", ("KERNEL32.dll", "api-ms-win-crt-runtime-l1-1-0.dll"))
    _pe(crt / "msvcp140.dll", ("vcruntime140.dll",))
    _pe(crt / "msvcp140_1.dll", ("vcruntime140.dll",))
    return build, crt


def _staged(tmp_path: Path) -> tuple[Path, Path, dict]:
    build, _ = _toolchain(tmp_path)
    root = tmp_path / "artifact"
    _pe(root / "PuriPulyHeart.exe", ("python314.dll",))
    _pe(root / "python314.dll", ("KERNEL32.dll",))
    _pe(
        root / "site-packages/onnxruntime/capi/onnxruntime_pybind11_state.pyd",
        ("python314.dll", "MSVCP140_1.dll"),
        ("vulkan-1.dll",),
    )
    runtime = stage_vc_runtime(root, build)
    evidence = tmp_path / "vc-runtime.json"
    evidence.write_text(json.dumps(runtime), encoding="utf-8")
    return root, evidence, runtime


def test_real_import_and_delay_tables_and_version_resources(tmp_path: Path) -> None:
    path = _pe(tmp_path / "module.pyd", ("PRIVATE.dll",), ("DELAY.dll",))
    image = read_pe_image(path)
    assert image.imports == (("delay_import", "delay.dll"), ("import", "private.dll"))
    assert image.version == "14.44.35211.0"
    assert image.company == "Microsoft Corporation"
    assert not image.debug
    assert read_pe_image(_pe(tmp_path / "debug.dll", debug=True)).debug


def test_stages_matched_official_crt_and_records_resolved_onnx_origin(tmp_path: Path) -> None:
    root, evidence, runtime = _staged(tmp_path)
    modules = {item["path"]: item for item in runtime["modules"]}
    dll = root / "msvcp140_1.dll"
    assert dll.is_file()
    assert modules[dll.name]["sha256"] == hashlib.sha256(dll.read_bytes()).hexdigest()
    assert Path(modules[dll.name]["source"]).parent.name == "Microsoft.VC143.CRT"
    assert {item["version"] for item in modules.values()} == {"14.44.35211.0"}
    assert runtime["toolchain"]["toolchain_version"] == "14.44.35207"
    assert runtime["toolchain"]["redist_directory_version"] == "14.44.35112"
    report = validate_pe_dependencies(root, evidence)
    onnx = next(
        item for item in report["images"] if item["path"].endswith("onnxruntime_pybind11_state.pyd")
    )
    imported = next(item for item in onnx["imports"] if item["name"] == "msvcp140_1.dll")
    assert imported["path"] == "msvcp140_1.dll"
    assert imported["source"] == modules["msvcp140_1.dll"]["source"]
    assert {item["external"] for item in onnx["imports"] if "external" in item} == {"gpu_driver"}


@pytest.mark.parametrize("kind", ["import", "delay_import"])
def test_required_private_dll_removal_fails_even_if_developer_path_has_a_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    root, evidence, _ = _staged(tmp_path)
    private = _pe(root / "site-packages/example/private.dll")
    module = root / "site-packages/example/module.pyd"
    _pe(
        module,
        ("private.dll",) if kind == "import" else (),
        ("private.dll",) if kind == "delay_import" else (),
    )
    validate_pe_dependencies(root, evidence)
    developer = tmp_path / "developer/System32"
    private_copy = _pe(developer / "private.dll")
    monkeypatch.setenv("PATH", str(developer))
    private.unlink()
    assert private_copy.is_file()
    with pytest.raises(FileNotFoundError, match=f"\\[{kind}\\] -> private.dll"):
        validate_pe_dependencies(root, evidence)


def test_only_registered_same_package_wheel_libraries_resolve(tmp_path: Path) -> None:
    root, evidence, _ = _staged(tmp_path)
    package = root / "site-packages/scipy"
    _pe(package / "special/module.pyd", ("blas.dll",))
    _pe(root / "site-packages/scipy.libs/blas.dll")
    init = package / "__init__.py"
    init.write_text("import os\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="blas.dll"):
        validate_pe_dependencies(root, evidence)
    init.write_text(
        "import os\n"
        "libs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scipy.libs'))\n"
        "os.add_dll_directory(libs_dir)\n",
        encoding="utf-8",
    )
    validate_pe_dependencies(root, evidence)
    init.write_text(
        "import os\nlibs_dir = 'unrelated.libs'\nos.add_dll_directory(libs_dir)\n", encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="blas.dll"):
        validate_pe_dependencies(root, evidence)


def test_missing_or_modified_staged_crt_is_not_an_os_dependency(tmp_path: Path) -> None:
    root, evidence, _ = _staged(tmp_path)
    (root / "msvcp140_1.dll").unlink()
    with pytest.raises(ValueError, match="missing or changed: msvcp140_1.dll"):
        validate_pe_dependencies(root, evidence)


@pytest.mark.parametrize("invalid", ["wrong-version", "debug", "wrong-vendor", "x86"])
def test_official_crt_source_must_be_matching_x64_release_set(tmp_path: Path, invalid: str) -> None:
    build, crt = _toolchain(tmp_path)
    options = {
        "wrong-version": {"version": "14.43.12345.0"},
        "debug": {"debug": True},
        "wrong-vendor": {"company": "Untrusted"},
        "x86": {"machine": 0x14C},
    }[invalid]
    _pe(crt / "msvcp140_1.dll", **options)
    root = tmp_path / "artifact"
    _pe(root / "host.exe", ("msvcp140_1.dll",))
    with pytest.raises(ValueError):
        stage_vc_runtime(root, build)
    assert not (root / "msvcp140_1.dll").exists()


def test_redist_selection_never_uses_unmatched_default_or_debug_nonredist(tmp_path: Path) -> None:
    build, crt = _toolchain(tmp_path)
    defaults = crt.parents[4] / "Auxiliary/Build/Microsoft.VCRedistVersion.default.txt"
    defaults.write_text("14.43.12345\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match selected MSVC"):
        resolve_msvc_redist(build)
    defaults.write_text("14.44.35112\n", encoding="utf-8")
    (crt / "msvcp140_1.dll").unlink()
    _pe(
        crt.parent.parent / "debug_nonredist/x64/Microsoft.VC143.DebugCRT/msvcp140_1.dll",
        debug=True,
    )
    root = tmp_path / "artifact"
    _pe(root / "host.exe")
    with pytest.raises(FileNotFoundError, match="lacks required msvcp140_1.dll"):
        stage_vc_runtime(root, build)


def test_debug_interpreter_pairs_are_excluded_and_debug_crt_dependencies_fail(
    tmp_path: Path,
) -> None:
    build, _ = _toolchain(tmp_path)
    root = tmp_path / "artifact"
    _pe(root / "host.exe")
    release = _pe(root / "DLLs/_ssl.pyd")
    debug = _pe(root / "DLLs/_ssl_d.pyd", ("python314_d.dll", "ucrtbased.dll"), debug=True)
    report = stage_vc_runtime(root, build)
    assert release.is_file() and not debug.exists()
    assert report["excluded_debug_interpreter_modules"][0]["path"] == "DLLs/_ssl_d.pyd"
    evidence = tmp_path / "vc-runtime.json"
    evidence.write_text(json.dumps(report), encoding="utf-8")
    upstream = _pe(root / "site-packages/custom/upstream.dll", debug=True)
    validated = validate_pe_dependencies(root, evidence)
    assert next(
        image
        for image in validated["images"]
        if image["path"] == upstream.relative_to(root).as_posix()
    )["debug_resource_flag"]
    _pe(root / "site-packages/custom/debug.pyd", ("ucrtbased.dll",))
    with pytest.raises(ValueError, match="debug native dependency"):
        validate_pe_dependencies(root, evidence)


def test_unpaired_debug_interpreter_module_is_not_silently_discarded(tmp_path: Path) -> None:
    build, _ = _toolchain(tmp_path)
    root = tmp_path / "artifact"
    debug = _pe(root / "DLLs/_ssl_d.pyd", debug=True)
    with pytest.raises(ValueError, match="without its release pair"):
        stage_vc_runtime(root, build)
    assert debug.is_file()


@pytest.mark.parametrize(
    "module", ["transcribe.dll", "ggml-vulkan.dll", "ggml-cpu-x64.dll", "ggml-cpu-icelake.dll"]
)
def test_dynamic_gpu_modules_cannot_escape_native_import_closure(
    tmp_path: Path, module: str
) -> None:
    root, evidence, _ = _staged(tmp_path)
    _pe(root / module, ("builder-only.dll",))
    with pytest.raises(FileNotFoundError, match=module):
        validate_pe_dependencies(root, evidence)


def test_dynamic_cpu_module_must_match_x64_payload(tmp_path: Path) -> None:
    root, evidence, _ = _staged(tmp_path)
    _pe(root / "ggml-cpu-x64.dll", machine=0x14C)
    with pytest.raises(ValueError, match="x64"):
        validate_pe_dependencies(root, evidence)
