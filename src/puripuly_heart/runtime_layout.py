from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from puripuly_heart.config.paths import user_config_dir

RuntimeHostKind = Literal["source", "pyinstaller", "native"]


@dataclass(frozen=True, slots=True)
class RuntimeLayout:
    host_kind: RuntimeHostKind
    app_resource_root: Path
    native_runtime_root: Path
    host_executable: Path
    python_executable: Path
    user_data_root: Path
    model_cache_root: Path
    log_root: Path

    def resource(self, *parts: str) -> Path:
        return self.app_resource_root.joinpath(*parts)

    def native(self, *parts: str) -> Path:
        return self.native_runtime_root.joinpath(*parts)


_NATIVE_RESOURCE_ROOT_ENV = "PURIPULY_HEART_NATIVE_RESOURCE_ROOT"
_NATIVE_RUNTIME_ROOT_ENV = "PURIPULY_HEART_NATIVE_RUNTIME_ROOT"
_NATIVE_HOST_EXECUTABLE_ENV = "PURIPULY_HEART_NATIVE_HOST_EXECUTABLE"
_NATIVE_PYTHON_EXECUTABLE_ENV = "PURIPULY_HEART_NATIVE_PYTHON_EXECUTABLE"


def current_runtime_layout() -> RuntimeLayout:
    user_root = user_config_dir().resolve()
    native_resource_root = os.getenv(_NATIVE_RESOURCE_ROOT_ENV)
    if native_resource_root:
        resource_root = Path(native_resource_root).resolve()
        runtime_root = Path(os.getenv(_NATIVE_RUNTIME_ROOT_ENV, native_resource_root)).resolve()
        host_executable = Path(
            os.getenv(_NATIVE_HOST_EXECUTABLE_ENV, sys.executable)
        ).resolve()
        python_executable = Path(
            os.getenv(_NATIVE_PYTHON_EXECUTABLE_ENV, sys.executable)
        ).resolve()
        host_kind: RuntimeHostKind = "native"
    elif bool(getattr(sys, "frozen", False)):
        host_executable = Path(sys.executable).resolve()
        resource_root = Path(getattr(sys, "_MEIPASS", host_executable.parent)).resolve()
        runtime_root = resource_root
        python_executable = host_executable
        host_kind = "pyinstaller"
    else:
        resource_root = Path(__file__).resolve().parents[2]
        runtime_root = resource_root / "build"
        host_executable = Path(sys.executable).resolve()
        python_executable = host_executable
        host_kind = "source"

    return RuntimeLayout(
        host_kind=host_kind,
        app_resource_root=resource_root,
        native_runtime_root=runtime_root,
        host_executable=host_executable,
        python_executable=python_executable,
        user_data_root=user_root,
        model_cache_root=user_root / "models",
        log_root=user_root / "logs",
    )
