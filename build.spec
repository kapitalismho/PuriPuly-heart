# ruff: noqa: F821
# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for PuriPuly <3.

Direct Windows PyInstaller packaging (executable-only / manual installer packaging):
    This direct path is not the release-complete compliance-packaging path and requires the staged overlay executable at build/overlay/PuriPulyHeartOverlay.exe plus the vendored OpenVR bundle under third_party/openvr/ (enforced below).
    pwsh -File scripts/ci/prepare-soxr-release-inputs.ps1
    pwsh -File scripts/ci/prepare-flet-runtime.ps1
    python -m puripuly_heart.release_evidence.managed_gemma_distribution prepare
    pyinstaller build.spec
    ISCC installer.iss

Full release-complete compliance packaging requires scripts/ci/prepare-soxr-release-inputs.ps1 before scripts/ci/build-release-artifacts.ps1:
    pwsh -File scripts/ci/prepare-soxr-release-inputs.ps1
    pwsh -File scripts/ci/prepare-flet-runtime.ps1
    pwsh -File scripts/ci/build-release-artifacts.ps1 -AppVersion <version> -InnoSetupVersion <version>

Output:
    dist/PuriPulyHeart/  (folder with all files)
"""

import hashlib
import json
import os
import sys
from importlib import import_module
from pathlib import Path

import flet_cli.__pyinstaller.config as flet_pyinstaller_hook_config
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    get_module_file_attribute,
)

# Add src to path for imports
src_path = Path("src").resolve()
sys.path.insert(0, str(src_path))
from puripuly_heart._compat import moved_module_alias_targets

for moved_module_alias_parent in (
    "puripuly_heart.app",
    "puripuly_heart.app.adapters",
    "puripuly_heart.app.services",
    "puripuly_heart.core",
):
    import_module(moved_module_alias_parent)

moved_module_hiddenimports = list(moved_module_alias_targets())
release_smoke = os.environ.get("PURIPULY_HEART_RELEASE_PROCESS_CAPTURE_SMOKE") == "1"
entry_script = (
    Path("scripts/release/process-capture-runtime-smoke.py").resolve()
    if release_smoke
    else src_path / "puripuly_heart" / "main.py"
)
executable_name = "PuriPulyHeartProcessCaptureSmoke" if release_smoke else "PuriPulyHeart"

overlay_staged_path = Path("build").resolve() / "overlay" / "PuriPulyHeartOverlay.exe"
if not overlay_staged_path.exists():
    raise SystemExit(
        "Staged overlay executable not found at "
        f"{overlay_staged_path}. Build and stage the Rust overlay before PyInstaller packaging."
    )

gpu_worker_staged_path = (
    Path("build").resolve() / "gpu_worker" / "PuriPulyHeartGpuWorker.exe"
)
if not gpu_worker_staged_path.exists():
    raise SystemExit(
        "Staged GPU worker executable not found at "
        f"{gpu_worker_staged_path}. Build and stage the Rust GPU worker before PyInstaller packaging."
    )

from puripuly_heart.core.local_qwen_runtime import LOCAL_QWEN_PACKAGED_RUNTIME_RELATIVE_DIR

from puripuly_heart.core.overlay.openvr_vendor import collect_vendored_openvr_runtime_binaries
from puripuly_heart.release_evidence.managed_gemma_distribution import (
    normalize_pyinstaller_binaries,
    pyinstaller_data_entries,
)

block_cipher = None
SOXR_RELEASE_INPUTS_MANIFEST_PATH = Path("build/soxr-release-inputs/manifest.json").resolve()
SOXR_PACKAGED_RUNTIME_RELATIVE_DIR = Path("soxr")
FLET_WINDOWS_RUNTIME_ARCHIVE_PATH = Path("build/flet/flet-windows.zip").resolve()
FLET_WINDOWS_RUNTIME_SHA256 = "2cf0865b31bd0e394a24a6c2d270e084cf9dad9c711e0b5d0cf9fa9bfac31e14"
NOTO_CJK_SOURCE_FONT_PATH = src_path / "puripuly_heart" / "data" / "fonts" / "NotoSansCJK-Medium.ttc"
NOTO_CJK_PROVENANCE_DIR = Path("third_party/noto-sans-cjk").resolve()
NOTO_CJK_PACKAGED_PROVENANCE_RELATIVE_DIR = Path("third_party/noto-sans-cjk")
HTTP_EXTENSION_EXAMPLES_SOURCE_DIR = Path("examples/http_extensions").resolve()
HTTP_EXTENSION_EXAMPLES_PACKAGED_DIR = Path("examples/http_extensions")
managed_gemma_runtime_datas = [] if release_smoke else pyinstaller_data_entries(Path.cwd())

if not NOTO_CJK_SOURCE_FONT_PATH.is_file():
    raise SystemExit(f"Noto Sans CJK Medium TTC not found: {NOTO_CJK_SOURCE_FONT_PATH}")

if not HTTP_EXTENSION_EXAMPLES_SOURCE_DIR.is_dir():
    raise SystemExit(
        "HTTP extension examples directory not found: "
        f"{HTTP_EXTENSION_EXAMPLES_SOURCE_DIR}"
    )

if not FLET_WINDOWS_RUNTIME_ARCHIVE_PATH.is_file():
    raise SystemExit(
        "Pinned Flet Windows runtime archive not found at "
        f"{FLET_WINDOWS_RUNTIME_ARCHIVE_PATH}. "
        "Run scripts/ci/prepare-flet-runtime.ps1 before PyInstaller packaging."
    )
flet_windows_runtime_sha256 = hashlib.sha256(
    FLET_WINDOWS_RUNTIME_ARCHIVE_PATH.read_bytes()
).hexdigest()
if flet_windows_runtime_sha256 != FLET_WINDOWS_RUNTIME_SHA256:
    raise SystemExit(
        "Pinned Flet Windows runtime checksum mismatch: expected "
        f"{FLET_WINDOWS_RUNTIME_SHA256}, found {flet_windows_runtime_sha256}"
    )
flet_pyinstaller_hook_config.temp_bin_dir = str(FLET_WINDOWS_RUNTIME_ARCHIVE_PATH.parent)


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
    (
        str(HTTP_EXTENSION_EXAMPLES_SOURCE_DIR),
        HTTP_EXTENSION_EXAMPLES_PACKAGED_DIR.as_posix(),
    ),
    # Native VR Subtitle Overlay distribution provenance.
    # The TTC itself is included by the packaged puripuly_heart/data tree above.
    (str(NOTO_CJK_PROVENANCE_DIR / "OFL.txt"), NOTO_CJK_PACKAGED_PROVENANCE_RELATIVE_DIR.as_posix()),
    (str(NOTO_CJK_PROVENANCE_DIR / "README.md"), NOTO_CJK_PACKAGED_PROVENANCE_RELATIVE_DIR.as_posix()),
    (str(NOTO_CJK_PROVENANCE_DIR / "SHA256SUMS.txt"), NOTO_CJK_PACKAGED_PROVENANCE_RELATIVE_DIR.as_posix()),
] + collect_data_files("flet_desktop") + collect_data_files("huggingface_hub") + managed_gemma_runtime_datas

runtime_binaries = collect_dynamic_libs(
    "onnxruntime", destdir=LOCAL_QWEN_PACKAGED_RUNTIME_RELATIVE_DIR.as_posix()
)
runtime_binaries += collect_dynamic_libs("sherpa_onnx", destdir="sherpa_onnx/lib")
proctap_native_extension = Path(get_module_file_attribute("proctap._native")).resolve()
if not proctap_native_extension.is_file() or not proctap_native_extension.name.lower().startswith("_native"):
    raise SystemExit("Pinned ProcTap package did not provide a packageable _native extension")
proctap_runtime_binaries = [(str(proctap_native_extension), "proctap")]
proctap_runtime_binaries += collect_dynamic_libs("proctap", destdir="proctap")
runtime_binaries += proctap_runtime_binaries
runtime_binaries += collect_staged_soxr_runtime_binaries()
runtime_binaries += collect_vendored_openvr_runtime_binaries()
runtime_binaries += [(str(gpu_worker_staged_path), ".")]
hf_xet_native_extension = Path(get_module_file_attribute("hf_xet.hf_xet")).resolve()
if not hf_xet_native_extension.is_file() or hf_xet_native_extension.name.lower() != "hf_xet.pyd":
    raise SystemExit("Pinned hf_xet package did not provide the Windows hf_xet.pyd extension")
runtime_binaries += [(str(hf_xet_native_extension), "hf_xet")]

# Hidden imports for dynamic imports
hiddenimports = [
    "puripuly_heart.providers.stt.deepgram",
    "puripuly_heart.providers.stt.qwen_asr",
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
    "cryptography",
    "cryptography.fernet",
    "cryptography.hazmat.primitives.asymmetric.ed25519",
    "cryptography.hazmat.primitives.kdf.scrypt",
    "httpx",
    "hf_xet",
    "hf_xet.hf_xet",
    "sherpa_onnx",
    "sherpa_onnx.lib._sherpa_onnx",
    "sherpa_onnx.offline_recognizer",
    "keyring.backends.Windows",
    "onnxruntime",
    # NumPy's C-extension is required before the packaged CLI can even boot.
    "numpy._core._multiarray_umath",
    "soxr",
    "sounddevice",
    "puripuly_heart.core.local_asr",
    "puripuly_heart.core.local_asr.local_qwen_runtime",
    "puripuly_heart.config.process_capture_platform",
    "puripuly_heart.core.audio.process_source",
] + moved_module_hiddenimports + collect_submodules("proctap") + collect_submodules("huggingface_hub")

required_proctap_hiddenimports = {"proctap", "proctap._native", "proctap.backends.windows"}
if not required_proctap_hiddenimports.issubset(set(hiddenimports)):
    raise SystemExit("Required ProcTap hidden imports were not collected")

required_huggingface_hiddenimports = {"huggingface_hub", "hf_xet", "hf_xet.hf_xet"}
if not required_huggingface_hiddenimports.issubset(set(hiddenimports)):
    raise SystemExit("Required Hugging Face/Xet hidden imports were not collected")

a = Analysis(
    [str(entry_script)],
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
    noarchive=release_smoke,
)

normalize_soxr_runtime_binaries(a.binaries)
normalize_pyinstaller_binaries(a.binaries, Path.cwd())

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=executable_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=release_smoke,
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
    name=executable_name,
)
