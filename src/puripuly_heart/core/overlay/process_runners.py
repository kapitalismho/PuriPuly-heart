from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from puripuly_heart.runtime_layout import current_runtime_layout

from . import openvr_vendor
from .manifest import OverlayLaunchManifest
from .process_adapter import OverlayManagedProcess, _AsyncioOverlayProcess

logger = logging.getLogger("puripuly_heart.core.overlay.process")

OVERLAY_EXECUTABLE_NAME = "PuriPulyHeartOverlay.exe"
OPENVR_RUNTIME_DLL_NAME = "openvr_api.dll"
QUIET_TAIL_PROFILE_ENV = "PURIPULY_OVERLAY_QUIET_TAIL_PROFILE"
HANDOFF_EXPERIMENT_ENV = "PURIPULY_OVERLAY_HANDOFF_EXPERIMENT"
HANDOFF_EXPERIMENT_OFF = "off"
HANDOFF_EXPERIMENT_CACHED_FRAME_REHANDOFF = "cached_frame_rehandoff"
_HANDOFF_EXPERIMENT_VALUES = frozenset(
    {HANDOFF_EXPERIMENT_OFF, HANDOFF_EXPERIMENT_CACHED_FRAME_REHANDOFF}
)


def normalize_handoff_experiment(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("overlay handoff experiment must be a string")
    normalized = value.strip().lower()
    if normalized not in _HANDOFF_EXPERIMENT_VALUES:
        raise ValueError("overlay handoff experiment must be off or cached_frame_rehandoff")
    return normalized


class OverlayPreparationError(Exception):
    def __init__(self, failure_reason: str, message: str | None = None) -> None:
        super().__init__(message or failure_reason)
        self.failure_reason = failure_reason


class OverlayProcessRunner(Protocol):
    def configure_runtime(
        self,
        *,
        quiet_tail_profile: str,
        handoff_experiment: str,
    ) -> None: ...

    def prepare(self, manifest: OverlayLaunchManifest) -> Path: ...
    async def spawn(
        self,
        executable_path: Path,
        manifest_path: Path,
    ) -> OverlayManagedProcess: ...


@dataclass(slots=True)
class DefaultOverlayProcessRunner:
    executable_path: Path | None = None
    task_factory: Any | None = None
    quiet_tail_profile: str = "p05"
    handoff_experiment: str = HANDOFF_EXPERIMENT_OFF

    def configure_runtime(
        self,
        *,
        quiet_tail_profile: str,
        handoff_experiment: str,
    ) -> None:
        self.quiet_tail_profile = quiet_tail_profile
        self.handoff_experiment = normalize_handoff_experiment(handoff_experiment)

    def prepare(self, manifest: OverlayLaunchManifest) -> Path:
        _ = manifest
        if self.executable_path is not None:
            path = self.executable_path
        else:
            path = self._resolve_default_executable()
        if not path.exists():
            raise FileNotFoundError(path)
        stale_source = self._newer_local_dev_overlay_source(path)
        if stale_source is not None:
            raise OverlayPreparationError(
                "stale_overlay_build",
                f"staged overlay executable is older than overlay source: {stale_source}",
            )
        if path.name == OVERLAY_EXECUTABLE_NAME:
            self.ensure_bundled_openvr_runtime_dll(path)
        return path

    async def spawn(
        self,
        executable_path: Path,
        manifest_path: Path,
    ) -> OverlayManagedProcess:
        command: tuple[str, ...]
        if executable_path.suffix.lower() == ".py":
            command = (str(sys.executable), str(executable_path), "--config", str(manifest_path))
        else:
            command = (str(executable_path), "--config", str(manifest_path))
        child_env = os.environ.copy()
        child_env[QUIET_TAIL_PROFILE_ENV] = self.quiet_tail_profile
        child_env[HANDOFF_EXPERIMENT_ENV] = normalize_handoff_experiment(self.handoff_experiment)
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=child_env,
        )
        return _AsyncioOverlayProcess(process=process, task_factory=self.task_factory)

    @classmethod
    def default_executable_candidates(
        cls,
        *,
        sys_executable: Path | None = None,
        repo_root: Path | None = None,
    ) -> tuple[Path, Path]:
        if sys_executable is None and repo_root is None:
            layout = current_runtime_layout()
            if layout.host_kind == "source":
                executable = layout.native("overlay", OVERLAY_EXECUTABLE_NAME)
            else:
                executable = layout.native(OVERLAY_EXECUTABLE_NAME)
            return executable, executable
        executable = (sys_executable or Path(sys.executable)).resolve()
        root = repo_root or Path(__file__).resolve().parents[4]
        return executable.with_name(OVERLAY_EXECUTABLE_NAME), root / "build" / "overlay" / (
            OVERLAY_EXECUTABLE_NAME
        )

    @classmethod
    def resolve_default_executable(
        cls,
        *,
        sys_executable: Path | None = None,
        repo_root: Path | None = None,
    ) -> Path:
        packaged_sibling, staged = cls.default_executable_candidates(
            sys_executable=sys_executable,
            repo_root=repo_root,
        )
        if packaged_sibling.exists() and staged.exists():
            if staged.stat().st_mtime > packaged_sibling.stat().st_mtime:
                return staged
            return packaged_sibling
        if packaged_sibling.exists():
            return packaged_sibling
        if staged.exists():
            return staged
        return packaged_sibling

    def _resolve_default_executable(self) -> Path:
        return self.resolve_default_executable()

    @classmethod
    def _newer_local_dev_overlay_source(cls, executable_path: Path) -> Path | None:
        repo_root = cls._local_dev_repo_root_for_staged_executable(executable_path)
        if repo_root is None:
            return None

        executable_mtime = executable_path.stat().st_mtime
        for source_path in cls._local_dev_overlay_source_paths(repo_root):
            if source_path.stat().st_mtime > executable_mtime:
                return source_path
        return None

    @classmethod
    def _local_dev_repo_root_for_staged_executable(cls, executable_path: Path) -> Path | None:
        if executable_path.name != OVERLAY_EXECUTABLE_NAME:
            return None
        if executable_path.parent.name != "overlay":
            return None
        build_dir = executable_path.parent.parent
        if build_dir.name != "build":
            return None

        repo_root = build_dir.parent
        source_root = repo_root / "native" / "overlay" / "src"
        if not source_root.exists():
            return None
        return repo_root

    @classmethod
    def _local_dev_overlay_source_paths(cls, repo_root: Path) -> tuple[Path, ...]:
        overlay_root = repo_root / "native" / "overlay"
        source_paths: list[Path] = []
        for relative_path in ("Cargo.toml", "Cargo.lock", "build.rs"):
            candidate = overlay_root / relative_path
            if candidate.exists():
                source_paths.append(candidate)

        source_root = overlay_root / "src"
        if source_root.exists():
            source_paths.extend(
                sorted(path for path in source_root.rglob("*.rs") if path.is_file())
            )
        return tuple(source_paths)

    @classmethod
    def bundled_openvr_runtime_dll_path(cls, executable_path: Path) -> Path:
        return executable_path.with_name(OPENVR_RUNTIME_DLL_NAME)

    @classmethod
    def ensure_bundled_openvr_runtime_dll(
        cls,
        executable_path: Path,
    ) -> Path:
        bundled_path = cls.bundled_openvr_runtime_dll_path(executable_path)
        if cls._local_dev_repo_root_for_staged_executable(executable_path) is not None:
            try:
                vendored_bundle = openvr_vendor.validate_vendored_openvr_bundle()
            except (FileNotFoundError, ValueError) as error:
                raise OverlayPreparationError("vendored_openvr_dll_missing", str(error)) from error
            return cls._refresh_staged_openvr_runtime_dll(bundled_path, vendored_bundle)
        return cls._validate_packaged_openvr_runtime_dll(bundled_path)

    @classmethod
    def _refresh_staged_openvr_runtime_dll(
        cls,
        bundled_path: Path,
        vendored_bundle: openvr_vendor.VendoredOpenVrBundle,
    ) -> Path:
        if cls._staged_openvr_runtime_dll_needs_refresh(bundled_path, vendored_bundle):
            bundled_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(vendored_bundle.dll_path, bundled_path)
        return openvr_vendor.validate_openvr_runtime_dll(
            bundled_path,
            expected_sha256=vendored_bundle.dll_sha256,
        )

    @classmethod
    def _staged_openvr_runtime_dll_needs_refresh(
        cls,
        bundled_path: Path,
        vendored_bundle: openvr_vendor.VendoredOpenVrBundle,
    ) -> bool:
        if not bundled_path.is_file():
            return True

        try:
            openvr_vendor.validate_openvr_runtime_dll(
                bundled_path,
                expected_sha256=vendored_bundle.dll_sha256,
            )
        except ValueError:
            return True
        return False

    @classmethod
    def _validate_packaged_openvr_runtime_dll(
        cls,
        bundled_path: Path,
    ) -> Path:
        if not bundled_path.is_file():
            raise OverlayPreparationError(
                "packaged_openvr_dll_missing",
                f"Packaged OpenVR runtime DLL not found: {bundled_path}",
            )

        try:
            return openvr_vendor.validate_openvr_runtime_dll(bundled_path)
        except FileNotFoundError as error:
            raise OverlayPreparationError("packaged_openvr_dll_missing", str(error)) from error
        except ValueError as error:
            raise OverlayPreparationError("openvr_dll_hash_mismatch", str(error)) from error


@dataclass(slots=True)
class DesktopFletOverlayRunner:
    frozen: bool | None = None
    python_executable: Path | None = None
    app_executable: Path | None = None
    module_name: str = "puripuly_heart.ui.desktop_overlay"
    task_factory: Any | None = None

    def configure_runtime(
        self,
        *,
        quiet_tail_profile: str,
        handoff_experiment: str,
    ) -> None:
        _ = quiet_tail_profile
        if normalize_handoff_experiment(handoff_experiment) != HANDOFF_EXPERIMENT_OFF:
            raise ValueError("desktop overlay does not support handoff experiments")

    def prepare(self, manifest: OverlayLaunchManifest) -> Path:
        _ = manifest
        return self._launcher_executable()

    def build_command(
        self,
        manifest_path: Path,
        *,
        executable_path: Path | None = None,
    ) -> tuple[str, ...]:
        launcher = executable_path or self._launcher_executable()
        layout = current_runtime_layout()
        if self.frozen is None and self.python_executable is None and self.app_executable is None:
            if layout.host_kind == "source":
                return (
                    str(layout.python_executable),
                    "-m",
                    self.module_name,
                    "--config",
                    str(manifest_path),
                )
            native_prefix = ("--headless",) if layout.host_kind == "native" else ()
            return (
                str(layout.host_executable),
                *native_prefix,
                "run-desktop-overlay",
                "--config",
                str(manifest_path),
            )
        if self._is_frozen():
            return (str(launcher), "run-desktop-overlay", "--config", str(manifest_path))
        return (str(launcher), "-m", self.module_name, "--config", str(manifest_path))

    async def spawn(
        self,
        executable_path: Path,
        manifest_path: Path,
    ) -> OverlayManagedProcess:
        kwargs: dict[str, object] = {}
        if os.name == "nt":
            kwargs["creationflags"] = (
                subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
            )
        process = await asyncio.create_subprocess_exec(
            *self.build_command(manifest_path, executable_path=executable_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **kwargs,
        )
        return _AsyncioOverlayProcess(process=process, task_factory=self.task_factory)

    def _is_frozen(self) -> bool:
        if self.frozen is not None:
            return self.frozen
        return current_runtime_layout().host_kind != "source"

    def _launcher_executable(self) -> Path:
        layout = current_runtime_layout()
        if self._is_frozen():
            return self.app_executable or layout.host_executable
        return self.python_executable or layout.python_executable
