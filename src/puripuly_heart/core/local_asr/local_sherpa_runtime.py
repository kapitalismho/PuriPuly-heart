from __future__ import annotations

import ctypes
import logging
import os
import sys
from pathlib import Path

from puripuly_heart.runtime_layout import current_runtime_layout

logger = logging.getLogger(__name__)

LOCAL_SHERPA_PACKAGED_RUNTIME_RELATIVE_DIR = Path("_runtime") / "local_qwen"
REQUIRED_LOCAL_SHERPA_RUNTIME_DLLS = (
    "onnxruntime.dll",
    "onnxruntime_providers_shared.dll",
)


class LocalSherpaRuntimeBootstrapError(RuntimeError):
    """Raised when the local sherpa Windows runtime cannot be prepared."""


_REGISTERED_LOCAL_SHERPA_RUNTIME_DIR: Path | None = None
_REGISTERED_LOCAL_SHERPA_RUNTIME_HANDLE: object | None = None
_REGISTERED_LOCAL_SHERPA_RUNTIME_LIBRARY: object | None = None


def resolve_local_sherpa_runtime_dir() -> Path:
    layout = current_runtime_layout()
    if sys.platform == "win32" and layout.host_kind != "source":
        return layout.native(*LOCAL_SHERPA_PACKAGED_RUNTIME_RELATIVE_DIR.parts)

    import onnxruntime

    return Path(onnxruntime.__file__).resolve().parent / "capi"


def _validate_local_sherpa_runtime_dir(runtime_dir: Path) -> None:
    if not runtime_dir.exists():
        raise LocalSherpaRuntimeBootstrapError(
            f"local sherpa runtime directory does not exist: {runtime_dir}"
        )
    if not runtime_dir.is_dir():
        raise LocalSherpaRuntimeBootstrapError(
            f"local sherpa runtime path is not a directory: {runtime_dir}"
        )

    missing = [
        dll_name
        for dll_name in REQUIRED_LOCAL_SHERPA_RUNTIME_DLLS
        if not (runtime_dir / dll_name).is_file()
    ]
    if missing:
        raise LocalSherpaRuntimeBootstrapError(
            f"missing required local sherpa runtime DLLs in {runtime_dir}: {', '.join(missing)}"
        )


def _path_contains_entry(path_value: str, entry: str) -> bool:
    normalized_entry = os.path.normcase(os.path.normpath(entry))
    return any(
        os.path.normcase(os.path.normpath(candidate)) == normalized_entry
        for candidate in path_value.split(os.pathsep)
        if candidate
    )


def ensure_local_sherpa_windows_runtime() -> Path:
    global _REGISTERED_LOCAL_SHERPA_RUNTIME_DIR
    global _REGISTERED_LOCAL_SHERPA_RUNTIME_HANDLE
    global _REGISTERED_LOCAL_SHERPA_RUNTIME_LIBRARY

    if sys.platform != "win32":
        return Path()

    runtime_dir = resolve_local_sherpa_runtime_dir().resolve()
    _validate_local_sherpa_runtime_dir(runtime_dir)

    if _REGISTERED_LOCAL_SHERPA_RUNTIME_DIR != runtime_dir:
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if add_dll_directory is None:
            raise LocalSherpaRuntimeBootstrapError("os.add_dll_directory is unavailable")
        _REGISTERED_LOCAL_SHERPA_RUNTIME_HANDLE = add_dll_directory(str(runtime_dir))
        win_dll = getattr(ctypes, "WinDLL", None)
        if win_dll is None:
            _REGISTERED_LOCAL_SHERPA_RUNTIME_HANDLE = None
            raise LocalSherpaRuntimeBootstrapError("ctypes.WinDLL is unavailable")
        try:
            _REGISTERED_LOCAL_SHERPA_RUNTIME_LIBRARY = win_dll(str(runtime_dir / "onnxruntime.dll"))
        except OSError as exc:
            _REGISTERED_LOCAL_SHERPA_RUNTIME_HANDLE = None
            raise LocalSherpaRuntimeBootstrapError(
                f"failed to load local sherpa runtime DLL: {runtime_dir / 'onnxruntime.dll'}"
            ) from exc
        _REGISTERED_LOCAL_SHERPA_RUNTIME_DIR = runtime_dir

    current_path = os.environ.get("PATH", "")
    runtime_dir_str = str(runtime_dir)
    if not _path_contains_entry(current_path, runtime_dir_str):
        os.environ["PATH"] = (
            f"{runtime_dir_str}{os.pathsep}{current_path}" if current_path else runtime_dir_str
        )

    return runtime_dir
