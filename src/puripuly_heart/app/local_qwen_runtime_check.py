from __future__ import annotations

import importlib
import sys

from puripuly_heart.core.local_asr import local_qwen_runtime


def _print_error(message: str) -> None:
    print(f"Error: {message}", flush=True)


def run_local_qwen_runtime_check() -> int:
    if sys.platform != "win32":
        _print_error("local-qwen-runtime-check is only supported on Windows")
        return 2

    try:
        runtime_dir = local_qwen_runtime.ensure_local_qwen_windows_runtime()
    except local_qwen_runtime.LocalQwenRuntimeBootstrapError as exc:
        _print_error(f"failed to verify Local Qwen Windows runtime DLL directory: {exc}")
        return 2

    try:
        importlib.import_module("sherpa_onnx")
        importlib.import_module("sherpa_onnx.offline_recognizer")
    except (ImportError, OSError) as exc:
        _print_error(f"failed to import sherpa_onnx: {exc}")
        return 2

    print(f"local_qwen_runtime_dir={runtime_dir}", flush=True)
    return 0
