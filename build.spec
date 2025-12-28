# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for PuriPuly <3.

Build command:
    pyinstaller build.spec

Output:
    dist/PuriPulyHeart/  (folder with all files)
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path("src").resolve()
sys.path.insert(0, str(src_path))

from puripuly_heart import __version__

block_cipher = None

# Collect data files
datas = [
    # VAD model and data files
    (str(src_path / "puripuly_heart" / "data"), "puripuly_heart/data"),
    # Prompt templates
    ("prompts", "prompts"),
]

# Hidden imports for dynamic imports
hiddenimports = [
    "puripuly_heart.providers.stt.deepgram",
    "puripuly_heart.providers.stt.qwen_asr",
    "puripuly_heart.providers.stt.soniox",
    "puripuly_heart.providers.llm.gemini",
    "puripuly_heart.providers.llm.qwen",
    "google.genai",
    "dashscope",
    "deepgram",
    "websockets",
    "flet",
    "httpx",
    "keyring.backends.Windows",
    "onnxruntime",
    "sounddevice",
    "numpy",
]

a = Analysis(
    [str(src_path / "puripuly_heart" / "main.py")],
    pathex=[str(src_path)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
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
    icon=None,  # TODO: Add icon path if available
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
