from __future__ import annotations

import os
import subprocess
import tomllib
from pathlib import Path

PRODUCT_NAME = "PuriPuly <3"
COMPANY_NAME = "salee"
MAIN_FILE_DESCRIPTION = "PuriPuly <3"
SMOKE_FILE_DESCRIPTION = "PuriPuly <3 Process Capture Smoke"
OVERLAY_FILE_DESCRIPTION = "PuriPuly <3 Overlay"
GPU_WORKER_FILE_DESCRIPTION = "PuriPuly <3 GPU Worker"
SETUP_FILE_DESCRIPTION = "PuriPuly <3 Setup"

_TRANSLATION_ID = "040904B0"


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def read_project_version(repo_root: Path | None = None) -> str:
    root = Path(repo_root).resolve() if repo_root is not None else repo_root_from_here()
    pyproject_path = root / "pyproject.toml"
    try:
        payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"project version source not found: {pyproject_path}") from exc
    try:
        version = payload["project"]["version"]
    except KeyError as exc:
        raise RuntimeError(f"project version missing in {pyproject_path}") from exc
    if not isinstance(version, str) or not version.strip():
        raise RuntimeError(f"project version is blank in {pyproject_path}")
    return version.strip()


def version_tuple(version: str) -> tuple[int, int, int, int]:
    parts = version.strip().split(".")
    if not 1 <= len(parts) <= 4:
        raise ValueError(f"version must have 1-4 numeric parts: {version!r}")
    numbers: list[int] = []
    for part in parts:
        if not part.isdigit():
            raise ValueError(f"version part is not numeric: {part!r} in {version!r}")
        value = int(part)
        if not 0 <= value <= 65535:
            raise ValueError(f"version part out of range 0-65535: {part!r} in {version!r}")
        numbers.append(value)
    while len(numbers) < 4:
        numbers.append(0)
    return (numbers[0], numbers[1], numbers[2], numbers[3])


def file_description_for_executable(executable_name: str) -> str:
    lowered = executable_name.lower()
    if lowered.startswith("puripulyheartoverlay"):
        return OVERLAY_FILE_DESCRIPTION
    if lowered.startswith("puripulyheartgpuworker"):
        return GPU_WORKER_FILE_DESCRIPTION
    if lowered.startswith("puripulyheartprocesscapturesmoke"):
        return SMOKE_FILE_DESCRIPTION
    return MAIN_FILE_DESCRIPTION


def render_pyinstaller_version_info(
    *,
    version: str,
    product_name: str = PRODUCT_NAME,
    file_description: str,
    internal_name: str,
    original_filename: str,
    company_name: str = COMPANY_NAME,
) -> str:
    if product_name != PRODUCT_NAME:
        raise ValueError(f"ProductName must be exactly {PRODUCT_NAME!r}, got {product_name!r}")
    numbers = version_tuple(version)
    text = version.strip()
    lines = [
        "# UTF-8",
        "#",
        "# Windows version resource for PuriPuly <3.",
        "# Generated from pyproject.toml; do not edit by hand.",
        "VSVersionInfo(",
        "  ffi=FixedFileInfo(",
        f"    filevers={numbers!r},",
        f"    prodvers={numbers!r},",
        "    mask=0x3f,",
        "    flags=0x0,",
        "    OS=0x40004,",
        "    fileType=0x1,",
        "    subtype=0x0,",
        "    date=(0, 0)",
        "  ),",
        "  kids=[",
        "    StringFileInfo(",
        "      [",
        "      StringTable(",
        f"        { _TRANSLATION_ID!r},",
        "        [",
        f"          StringStruct('CompanyName', {company_name!r}),",
        f"          StringStruct('FileDescription', {file_description!r}),",
        f"          StringStruct('FileVersion', {text!r}),",
        f"          StringStruct('InternalName', {internal_name!r}),",
        f"          StringStruct('OriginalFilename', {original_filename!r}),",
        f"          StringStruct('ProductName', {product_name!r}),",
        f"          StringStruct('ProductVersion', {text!r})",
        "        ])",
        "      ]),",
        "    VarFileInfo([VarStruct('Translation', [1033, 1200])])",
        "  ]",
        ")",
        "",
    ]
    return "\n".join(lines)


def ensure_pyinstaller_version_file(
    *,
    repo_root: Path,
    executable_name: str,
    output_path: Path | None = None,
) -> Path:
    version = read_project_version(repo_root)
    file_description = file_description_for_executable(executable_name)
    internal_name = executable_name
    original_filename = (
        executable_name if executable_name.lower().endswith(".exe") else executable_name + ".exe"
    )
    text = render_pyinstaller_version_info(
        version=version,
        product_name=PRODUCT_NAME,
        file_description=file_description,
        internal_name=internal_name,
        original_filename=original_filename,
    )
    destination = (
        Path(output_path)
        if output_path is not None
        else Path(repo_root) / "build" / (f"version_info_{executable_name}.txt")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return destination


def _pe_metadata_via_pefile(exe_path: Path) -> dict[str, str]:
    import pefile

    pe = pefile.PE(str(exe_path), fast_load=True)
    try:
        pe.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"]]
        )
        result: dict[str, str] = {}
        if hasattr(pe, "FileInfo"):
            for file_info in pe.FileInfo or []:
                for entry in file_info:
                    if hasattr(entry, "StringTable"):
                        for table in entry.StringTable:
                            for string_entry in table.entries.values():
                                try:
                                    key = string_entry.key.decode("utf-16le", errors="strict")
                                    value = string_entry.value.decode(
                                        "utf-16le", errors="strict"
                                    ).rstrip("\x00")
                                except UnicodeDecodeError:
                                    continue
                                result[key] = value
        filevers = ""
        prodvers = ""
        try:
            fixed = pe.VS_FIXEDFILEINFO[0]
            filevers = (
                f"{fixed.FileVersionMS >> 16}.{fixed.FileVersionMS & 0xFFFF}."
                f"{fixed.FileVersionLS >> 16}.{fixed.FileVersionLS & 0xFFFF}"
            )
            prodvers = (
                f"{fixed.ProductVersionMS >> 16}.{fixed.ProductVersionMS & 0xFFFF}."
                f"{fixed.ProductVersionLS >> 16}.{fixed.ProductVersionLS & 0xFFFF}"
            )
        except (AttributeError, IndexError):
            pass
        result["__FileVersionBinary"] = filevers
        result["__ProductVersionBinary"] = prodvers
        return result
    finally:
        pe.close()


def _pe_metadata_via_powershell(exe_path: Path) -> dict[str, str]:
    script = (
        "$v = (Get-Item -LiteralPath $env:PURIPULY_PE_PATH).VersionInfo; "
        "@($v.ProductName, $v.ProductVersion, $v.FileVersion, $v.FileDescription, "
        '$v.CompanyName, $v.InternalName, $v.OriginalFilename) -join "`n"'
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env={"PURIPULY_PE_PATH": str(exe_path), "PATH": os.environ.get("PATH", "")},
    )
    rows = completed.stdout.splitlines()
    while len(rows) < 7:
        rows.append("")
    return {
        "ProductName": rows[0].strip(),
        "ProductVersion": rows[1].strip(),
        "FileVersion": rows[2].strip(),
        "FileDescription": rows[3].strip(),
        "CompanyName": rows[4].strip(),
        "InternalName": rows[5].strip(),
        "OriginalFilename": rows[6].strip(),
    }


def read_pe_product_metadata(exe_path: Path) -> dict[str, str]:
    exe_path = Path(exe_path).resolve()
    if not exe_path.is_file():
        raise RuntimeError(f"PE file not found: {exe_path}")
    try:
        return _pe_metadata_via_pefile(exe_path)
    except ImportError:
        pass
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"pefile probe failed for {exe_path}: {exc}") from exc
    return _pe_metadata_via_powershell(exe_path)


def verify_pe_product_metadata(
    exe_path: Path,
    *,
    expected_version: str,
    expected_product: str = PRODUCT_NAME,
) -> dict[str, str]:
    metadata = read_pe_product_metadata(exe_path)
    product = metadata.get("ProductName", "")
    if product != expected_product:
        raise RuntimeError(
            f"ProductName mismatch in {exe_path}: expected {expected_product!r}, found {product!r}"
        )
    expected_tuple = version_tuple(expected_version)
    expected_binary = (
        f"{expected_tuple[0]}.{expected_tuple[1]}.{expected_tuple[2]}.{expected_tuple[3]}"
    )
    binary_product = metadata.get("__ProductVersionBinary", "")
    if binary_product:
        if binary_product != expected_binary:
            raise RuntimeError(
                f"binary ProductVersion mismatch in {exe_path}: "
                f"expected {expected_binary}, found {binary_product}"
            )
    else:
        text_product = metadata.get("ProductVersion", "")
        normalized = text_product.strip()
        if normalized not in {expected_version.strip(), expected_binary}:
            raise RuntimeError(
                f"ProductVersion mismatch in {exe_path}: "
                f"expected {expected_version!r} ({expected_binary}), found {text_product!r}"
            )
    return metadata
