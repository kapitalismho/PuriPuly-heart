from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

from puripuly_heart.core.soxr_runtime import (
    SOXR_EXTENSION_MODULE_NAME,
    SOXR_SIBLING_DLL_NAME,
    SoxrRuntimeAvailabilityError,
    SoxrRuntimePaths,
    resolve_soxr_runtime_paths,
    validate_soxr_runtime_paths,
)

SOXR_SMOKE_INPUT_RATE_HZ = 48000
SOXR_SMOKE_OUTPUT_RATE_HZ = 16000
SOXR_SMOKE_FRAME_COUNT = 480
SOXR_RUNTIME_REPORT_PATH_ENV_VAR = "PURIPULY_HEART_SOXR_RUNTIME_REPORT_PATH"
WINDOWS_MAX_MODULE_PATH_LENGTH = 32768


def _print_error(message: str) -> None:
    print(f"Error: {message}", flush=True)


def _path_to_string(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path)


def _resolve_imported_soxr_extension_path() -> Path | None:
    try:
        soxr_extension_module = importlib.import_module(SOXR_EXTENSION_MODULE_NAME)
    except Exception:
        return None

    module_file = getattr(soxr_extension_module, "__file__", None)
    if not module_file:
        return None
    return Path(module_file)


def _resolve_loaded_soxr_dll_path() -> Path | None:
    if sys.platform != "win32":
        return None

    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_module_handle = kernel32.GetModuleHandleW
        get_module_handle.argtypes = [wintypes.LPCWSTR]
        get_module_handle.restype = wintypes.HMODULE
        get_module_filename = kernel32.GetModuleFileNameW
        get_module_filename.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
        get_module_filename.restype = wintypes.DWORD

        module_handle = get_module_handle(SOXR_SIBLING_DLL_NAME)
        if not module_handle:
            return None

        buffer_size = 260
        while buffer_size <= WINDOWS_MAX_MODULE_PATH_LENGTH:
            path_buffer = ctypes.create_unicode_buffer(buffer_size)
            path_length = get_module_filename(module_handle, path_buffer, buffer_size)
            if path_length == 0:
                return None
            if path_length < buffer_size - 1:
                return Path(path_buffer.value).resolve()
            buffer_size *= 2
    except Exception:
        return None

    return None


def _write_soxr_runtime_report(runtime_paths: SoxrRuntimePaths) -> None:
    report_path_value = os.environ.get(SOXR_RUNTIME_REPORT_PATH_ENV_VAR)
    if report_path_value is None or not report_path_value.strip():
        return

    report_path = Path(report_path_value)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "expected_extension_path": str(runtime_paths.extension_path),
        "expected_runtime_dir": str(runtime_paths.runtime_dir),
        "expected_sibling_dll_path": str(runtime_paths.sibling_dll_path),
        "imported_extension_path": _path_to_string(_resolve_imported_soxr_extension_path()),
        "loaded_sibling_dll_path": _path_to_string(_resolve_loaded_soxr_dll_path()),
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_soxr_runtime_check() -> int:
    if sys.platform != "win32":
        _print_error("soxr-runtime-check is only supported on Windows")
        return 2

    try:
        runtime_paths = validate_soxr_runtime_paths(resolve_soxr_runtime_paths())
    except SoxrRuntimeAvailabilityError as exc:
        _print_error(f"failed to verify packaged soxr runtime: {exc}")
        return 2

    try:
        import numpy as np

        soxr = importlib.import_module("soxr")
        stream = soxr.ResampleStream(
            SOXR_SMOKE_INPUT_RATE_HZ,
            SOXR_SMOKE_OUTPUT_RATE_HZ,
            1,
            dtype="float32",
        )
        output = stream.resample_chunk(
            np.zeros(SOXR_SMOKE_FRAME_COUNT, dtype=np.float32),
            last=True,
        )
        if len(output) == 0:
            raise RuntimeError("smoke resample returned no output")
    except Exception as exc:
        _print_error(f"failed to import or smoke-test soxr: {exc}")
        return 2

    print(f"soxr_extension_path={runtime_paths.extension_path}", flush=True)
    print(f"soxr_runtime_dir={runtime_paths.runtime_dir}", flush=True)
    print(f"soxr_sibling_dll={runtime_paths.sibling_dll_path}", flush=True)

    try:
        _write_soxr_runtime_report(runtime_paths)
    except OSError as exc:
        _print_error(f"failed to write soxr runtime report: {exc}")
        return 2

    return 0
