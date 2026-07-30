from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

LOCAL_QWEN_PACKAGED_RUNTIME_RELATIVE_DIR = Path("_runtime") / "local_qwen"
REQUIRED_LOCAL_QWEN_RUNTIME_DLLS = (
    "onnxruntime.dll",
    "onnxruntime_providers_shared.dll",
)


class LocalQwenRuntimeBootstrapError(RuntimeError):
    """Raised when the local Qwen Windows runtime cannot be prepared."""


_REGISTERED_LOCAL_QWEN_RUNTIME_DIR: Path | None = None
_REGISTERED_LOCAL_QWEN_RUNTIME_HANDLE: object | None = None


def resolve_local_qwen_runtime_dir() -> Path:
    if sys.platform == "win32" and getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve().parent
        packaged_runtime_dir = executable_dir / LOCAL_QWEN_PACKAGED_RUNTIME_RELATIVE_DIR
        internal_packaged_runtime_dir = (
            executable_dir / "_internal" / LOCAL_QWEN_PACKAGED_RUNTIME_RELATIVE_DIR
        )
        if internal_packaged_runtime_dir.is_dir() and not packaged_runtime_dir.is_dir():
            return internal_packaged_runtime_dir
        return packaged_runtime_dir

    import onnxruntime

    return Path(onnxruntime.__file__).resolve().parent / "capi"


def _validate_local_qwen_runtime_dir(runtime_dir: Path) -> None:
    if not runtime_dir.exists():
        raise LocalQwenRuntimeBootstrapError(
            f"local qwen runtime directory does not exist: {runtime_dir}"
        )
    if not runtime_dir.is_dir():
        raise LocalQwenRuntimeBootstrapError(
            f"local qwen runtime path is not a directory: {runtime_dir}"
        )

    missing = [
        dll_name
        for dll_name in REQUIRED_LOCAL_QWEN_RUNTIME_DLLS
        if not (runtime_dir / dll_name).is_file()
    ]
    if missing:
        raise LocalQwenRuntimeBootstrapError(
            f"missing required local qwen runtime DLLs in {runtime_dir}: {', '.join(missing)}"
        )


def _path_contains_entry(path_value: str, entry: str) -> bool:
    normalized_entry = os.path.normcase(os.path.normpath(entry))
    return any(
        os.path.normcase(os.path.normpath(candidate)) == normalized_entry
        for candidate in path_value.split(os.pathsep)
        if candidate
    )


def ensure_local_qwen_windows_runtime() -> Path:
    global _REGISTERED_LOCAL_QWEN_RUNTIME_DIR
    global _REGISTERED_LOCAL_QWEN_RUNTIME_HANDLE

    if sys.platform != "win32":
        return Path()

    runtime_dir = resolve_local_qwen_runtime_dir().resolve()
    _validate_local_qwen_runtime_dir(runtime_dir)

    if _REGISTERED_LOCAL_QWEN_RUNTIME_DIR != runtime_dir:
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if add_dll_directory is None:
            raise LocalQwenRuntimeBootstrapError("os.add_dll_directory is unavailable")
        _REGISTERED_LOCAL_QWEN_RUNTIME_HANDLE = add_dll_directory(str(runtime_dir))
        _REGISTERED_LOCAL_QWEN_RUNTIME_DIR = runtime_dir

    current_path = os.environ.get("PATH", "")
    runtime_dir_str = str(runtime_dir)
    if not _path_contains_entry(current_path, runtime_dir_str):
        os.environ["PATH"] = (
            f"{runtime_dir_str}{os.pathsep}{current_path}" if current_path else runtime_dir_str
        )

    logger.info("[STT][local_qwen] Using Windows runtime DLL directory: %s", runtime_dir)
    return runtime_dir
