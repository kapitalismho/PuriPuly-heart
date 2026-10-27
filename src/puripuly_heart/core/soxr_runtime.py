from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

SOXR_EXTENSION_MODULE_NAME = "soxr.soxr_ext"
SOXR_SIBLING_DLL_NAME = "soxr.dll"


class SoxrRuntimeAvailabilityError(RuntimeError):
    """Raised when the packaged soxr runtime contract is unavailable."""


@dataclass(frozen=True, slots=True)
class SoxrRuntimePaths:
    extension_path: Path
    runtime_dir: Path
    sibling_dll_path: Path


def resolve_soxr_runtime_paths() -> SoxrRuntimePaths:
    try:
        spec = importlib.util.find_spec(SOXR_EXTENSION_MODULE_NAME)
    except ImportError as exc:
        raise SoxrRuntimeAvailabilityError(
            f"failed to resolve soxr extension module spec: {exc}"
        ) from exc

    if spec is None or not spec.origin:
        raise SoxrRuntimeAvailabilityError("soxr extension module spec is unavailable")

    extension_path = Path(spec.origin).resolve()
    runtime_dir = extension_path.parent.resolve()
    sibling_dll_path = (runtime_dir / SOXR_SIBLING_DLL_NAME).resolve()
    return SoxrRuntimePaths(
        extension_path=extension_path,
        runtime_dir=runtime_dir,
        sibling_dll_path=sibling_dll_path,
    )


def validate_soxr_runtime_paths(runtime_paths: SoxrRuntimePaths) -> SoxrRuntimePaths:
    if not runtime_paths.runtime_dir.exists():
        raise SoxrRuntimeAvailabilityError(
            f"soxr runtime directory does not exist: {runtime_paths.runtime_dir}"
        )
    if not runtime_paths.runtime_dir.is_dir():
        raise SoxrRuntimeAvailabilityError(
            f"soxr runtime path is not a directory: {runtime_paths.runtime_dir}"
        )
    if not runtime_paths.extension_path.is_file():
        raise SoxrRuntimeAvailabilityError(
            f"soxr extension file does not exist: {runtime_paths.extension_path}"
        )
    if not runtime_paths.sibling_dll_path.is_file():
        raise SoxrRuntimeAvailabilityError(
            f"missing required soxr sibling DLL: {runtime_paths.sibling_dll_path}"
        )
    return runtime_paths
