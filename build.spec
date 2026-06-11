# ruff: noqa: F821
# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for PuriPuly <3.

Direct Windows PyInstaller packaging (executable-only / manual installer packaging):
    This direct path is not the release-complete compliance-packaging path and requires the staged overlay executable at build/overlay/PuriPulyHeartOverlay.exe plus the vendored OpenVR bundle under third_party/openvr/ (enforced below).
    pwsh -File scripts/ci/prepare-soxr-release-inputs.ps1
    pyinstaller build.spec
    ISCC installer.iss

Full release-complete compliance packaging requires scripts/ci/prepare-soxr-release-inputs.ps1 before scripts/ci/build-release-artifacts.ps1:
    pwsh -File scripts/ci/prepare-soxr-release-inputs.ps1
    pwsh -File scripts/ci/build-release-artifacts.ps1 -AppVersion <version> -InnoSetupVersion <version>

Output:
    dist/PuriPulyHeart/  (folder with all files)
"""

import json
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# Add src to path for imports
src_path = Path("src").resolve()
sys.path.insert(0, str(src_path))

overlay_staged_path = Path("build").resolve() / "overlay" / "PuriPulyHeartOverlay.exe"
if not overlay_staged_path.exists():
    raise SystemExit(
        "Staged overlay executable not found at "
        f"{overlay_staged_path}. Build and stage the Rust overlay before PyInstaller packaging."
    )

from puripuly_heart.core.local_qwen_runtime import LOCAL_QWEN_PACKAGED_RUNTIME_RELATIVE_DIR
from puripuly_heart.core.overlay.openvr_vendor import collect_vendored_openvr_runtime_binaries

block_cipher = None
SOXR_RELEASE_INPUTS_MANIFEST_PATH = Path("build/soxr-release-inputs/manifest.json").resolve()
SOXR_PACKAGED_RUNTIME_RELATIVE_DIR = Path("soxr")
NOTO_CJK_SOURCE_FONT_PATH = src_path / "puripuly_heart" / "data" / "fonts" / "NotoSansCJK-Medium.ttc"
NOTO_CJK_PROVENANCE_DIR = Path("third_party/noto-sans-cjk").resolve()
NOTO_CJK_PACKAGED_PROVENANCE_RELATIVE_DIR = Path("third_party/noto-sans-cjk")

if not NOTO_CJK_SOURCE_FONT_PATH.is_file():
    raise SystemExit(f"Noto Sans CJK Medium TTC not found: {NOTO_CJK_SOURCE_FONT_PATH}")


def get_prepared_soxr_runtime_paths() -> tuple[Path, Path]:
    if not SOXR_RELEASE_INPUTS_MANIFEST_PATH.is_file():
        raise SystemExit(
            "Staged soxr release inputs manifest not found at "
            f"{SOXR_RELEASE_INPUTS_MANIFEST_PATH}. "
            "Run scripts/ci/prepare-soxr-release-inputs.ps1 before PyInstaller packaging."
        )

    manifest = json.loads(SOXR_RELEASE_INPUTS_MANIFEST_PATH.read_text(encoding="utf-8-sig"))
    runtime_manifest = manifest["runtime"]
    packaged_relative_dir = Path(runtime_manifest["packaged_relative_dir"])
    if packaged_relative_dir.as_posix() != SOXR_PACKAGED_RUNTIME_RELATIVE_DIR.as_posix():
        raise SystemExit(
            "Prepared soxr runtime packaged layout mismatch: expected "
            f"{SOXR_PACKAGED_RUNTIME_RELATIVE_DIR.as_posix()}, got "
            f"{packaged_relative_dir.as_posix()}"
        )

    extension_path = Path(runtime_manifest["extension_path"]).resolve()
    sibling_dll_path = Path(runtime_manifest["dll_path"]).resolve()
    expected_runtime_names = {"soxr_ext.pyd", "soxr.dll"}
    actual_runtime_names = {extension_path.name.lower(), sibling_dll_path.name.lower()}
    if actual_runtime_names != expected_runtime_names:
        raise SystemExit(
            "Prepared soxr runtime inputs must contain exactly soxr_ext.pyd and soxr.dll; "
            f"got {sorted(actual_runtime_names)}"
        )

    for runtime_path in (extension_path, sibling_dll_path):
        if not runtime_path.is_file():
            raise SystemExit(f"Prepared soxr runtime file not found: {runtime_path}")

    return extension_path, sibling_dll_path


def collect_staged_soxr_runtime_binaries() -> list[tuple[str, str]]:
    extension_path, sibling_dll_path = get_prepared_soxr_runtime_paths()

    return [
        (str(extension_path), SOXR_PACKAGED_RUNTIME_RELATIVE_DIR.as_posix()),
        (str(sibling_dll_path), SOXR_PACKAGED_RUNTIME_RELATIVE_DIR.as_posix()),
    ]


def normalize_soxr_runtime_binaries(binaries):
    binaries[:] = [
        binary
        for binary in binaries
        if not _is_root_level_auto_collected_soxr_dll(binary)
    ]


def _is_root_level_auto_collected_soxr_dll(binary) -> bool:
    destination_name, _source_path, _typecode = binary
    normalized_destination_name = destination_name.replace("\\", "/")
    return normalized_destination_name == "soxr.dll"

# Collect data files
datas = [
    # Project license text for packaged/installed distributions
    ("LICENSE", "."),
    # VAD model and data files
    (str(src_path / "puripuly_heart" / "data"), "puripuly_heart/data"),
    # Prompt templates
    ("prompts", "prompts"),
    # Native VR Subtitle Overlay distribution provenance.
    # The TTC itself is included by the packaged puripuly_heart/data tree above.
    (str(NOTO_CJK_PROVENANCE_DIR / "OFL.txt"), NOTO_CJK_PACKAGED_PROVENANCE_RELATIVE_DIR.as_posix()),
    (str(NOTO_CJK_PROVENANCE_DIR / "README.md"), NOTO_CJK_PACKAGED_PROVENANCE_RELATIVE_DIR.as_posix()),
    (str(NOTO_CJK_PROVENANCE_DIR / "SHA256SUMS.txt"), NOTO_CJK_PACKAGED_PROVENANCE_RELATIVE_DIR.as_posix()),
] + collect_data_files("flet_desktop")

runtime_binaries = collect_dynamic_libs(
    "onnxruntime", destdir=LOCAL_QWEN_PACKAGED_RUNTIME_RELATIVE_DIR.as_posix()
)
runtime_binaries += collect_staged_soxr_runtime_binaries()
runtime_binaries += collect_vendored_openvr_runtime_binaries()

# Hidden imports for dynamic imports
hiddenimports = [
    "puripuly_heart.providers.stt.deepgram",
    "puripuly_heart.providers.stt.qwen_asr",
    "puripuly_heart.providers.stt.sixtydb",
    "puripuly_heart.providers.stt.soniox",
    "puripuly_heart.providers.llm.gemini",
    "puripuly_heart.providers.llm.qwen",
    "puripuly_heart.providers.llm.qwen_async",
    "google.genai",
    "dashscope",
    "deepgram",
    "websockets",
    "flet",
    "flet_desktop",
    "httpx",
    "keyring.backends.Windows",
    "onnxruntime",
    # NumPy's C-extension is required before the packaged CLI can even boot.
    "numpy._core._multiarray_umath",
    "soxr",
    "sounddevice",
]

a = Analysis(
    [str(src_path / "puripuly_heart" / "main.py")],
    pathex=[str(src_path)],
    binaries=runtime_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "soxr.soxr_ext",
        "tkinter",
        "unittest",
        "pydoc",
        "doctest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

normalize_soxr_runtime_binaries(a.binaries)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PuriPulyHeart",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Windowed application (no terminal)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    contents_directory=".",
    icon=str(src_path / "puripuly_heart" / "data" / "icons" / "icon.ico"),
    version_info=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="PuriPulyHeart",
)
