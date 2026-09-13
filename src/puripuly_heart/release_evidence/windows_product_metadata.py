from __future__ import annotations

import tomllib
from pathlib import Path

PRODUCT_NAME = "PuriPuly <3"
COMPANY_NAME = "salee"
MAIN_FILE_DESCRIPTION = "PuriPuly <3"
SMOKE_FILE_DESCRIPTION = "PuriPuly <3 Process Capture Smoke"
OVERLAY_FILE_DESCRIPTION = "PuriPuly <3 Overlay"
GPU_WORKER_FILE_DESCRIPTION = "PuriPuly <3 GPU Worker"

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
