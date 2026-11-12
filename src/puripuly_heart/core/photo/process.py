from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

from puripuly_heart.core.lifecycle import LifecycleScope
from puripuly_heart.core.overlay.manifest import OverlayLaunchManifest
from puripuly_heart.core.overlay.process_adapter import (
    OverlayManagedProcess,
    _AsyncioOverlayProcess,
)
from puripuly_heart.core.overlay.process_runners import DefaultOverlayProcessRunner
from puripuly_heart.core.owned_thread import run_owned_thread_call

from .bridge import PhotoBridge


class PhotoProcessRunner:
    def __init__(self, executable_path: Path | None = None) -> None:
        self.executable_path = executable_path

    def prepare(self, manifest: dict) -> Path:
        return DefaultOverlayProcessRunner(executable_path=self.executable_path).prepare(
            cast(OverlayLaunchManifest, manifest)
        )

    async def spawn(
        self, executable: Path, manifest_path: Path, *, task_factory: Callable
    ) -> OverlayManagedProcess:
        kwargs: dict = {}
        if os.name == "nt":
            kwargs["creationflags"] = (
                subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
            )
        process = await asyncio.create_subprocess_exec(
            str(executable),
            "--role",
            "photo",
            "--config",
            str(manifest_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **kwargs,
        )
        return _AsyncioOverlayProcess(process=process, task_factory=task_factory)


class PhotoProcessSupervisor:
    def __init__(
        self,
        *,
        bridge: PhotoBridge,
        renderer_changed: Callable[[int], None],
        diagnostic: Callable[[str, dict], None],
        runner: PhotoProcessRunner | None = None,
        backend: str = "openvr",
        interaction: Callable[[], dict | None] = lambda: None,
        startup_timeout: float = 10,
        shutdown_timeout: float = 3,
        restart_limit: int = 2,
        temporary_root: Path | None = None,
    ) -> None:
        if backend not in ("openvr", "probe"):
            raise ValueError("unsupported photo backend")
        self.bridge = bridge
        self._renderer_changed = renderer_changed
        self._diagnostic = diagnostic
        self._runner = runner or PhotoProcessRunner()
        self._interaction = interaction
        self.backend = backend
        self._startup_timeout = startup_timeout
        self._shutdown_timeout = shutdown_timeout
        self._restart_limit = restart_limit
        self._temporary_root = temporary_root
        self._scope = LifecycleScope("PhotoProcessSupervisor")
        self._stop = asyncio.Event()
        self._restart = asyncio.Event()
        self._generation_stopped = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.process: OverlayManagedProcess | None = None
        self.generation = 0
        self.instance_id: str | None = None
        self.failure_code: str | None = None
        self.last_exit_code: int | None = None
        self.last_shutdown_acknowledged = False
        self._reader_sequence = 0

    @property
    def pid(self) -> int | None:
        return getattr(self.process, "pid", None)

    async def start(self) -> None:
        if self._stop.is_set() or self._task is not None and not self._task.done():
            return
        await self.bridge.start()
        self._task = self._scope.create_task(self._run(), name="supervise")

    def _reader_task(self, coroutine, *, task_name: str):
        self._reader_sequence += 1
        return self._scope.create_task(coroutine, name=f"{task_name}-{self._reader_sequence}")

    async def abort_preparation(self, generation: int) -> None:
        if generation != self.generation or self.process is None:
            self.bridge.renderer_stopped(generation)
            return
        self._restart.set()
        await self._generation_stopped.wait()

    async def _run(self) -> None:
        for recovery in range(self._restart_limit + 1):
            if self._stop.is_set():
                return
            self.generation += 1
            self._restart.clear()
            self._generation_stopped.clear()
            self._renderer_changed(self.generation)
            manifest = await self.bridge.activate(self.generation, backend=self.backend)
            self.instance_id = manifest["instance_id"]
            interaction = await run_owned_thread_call(self._interaction)
            if interaction is not None:
                manifest["interaction"] = interaction
            directory = tempfile.TemporaryDirectory(
                prefix="puripuly-photo-child-", dir=self._temporary_root
            )
            waiters: list[asyncio.Task] = []
            process: OverlayManagedProcess | None = None
            try:
                manifest_path = Path(directory.name) / "launch.json"
                await run_owned_thread_call(lambda: self._write_manifest(manifest_path, manifest))
                executable = await run_owned_thread_call(lambda: self._runner.prepare(manifest))
                if self._stop.is_set():
                    return
                process = await self._runner.spawn(
                    executable, manifest_path, task_factory=self._reader_task
                )
                self.process = process
                self._diagnostic(
                    "process_started", {"generation": self.generation, "pid": self.pid}
                )
                exit_wait = self._scope.create_task(
                    process.wait_for_exit(), name=f"exit-{self.generation}"
                )
                ready_wait = self._scope.create_task(
                    self.bridge.ready.wait(), name=f"ready-{self.generation}"
                )
                stop_wait = self._scope.create_task(
                    self._stop.wait(), name=f"stop-{self.generation}"
                )
                restart_wait = self._scope.create_task(
                    self._restart.wait(), name=f"restart-{self.generation}"
                )
                waiters.extend((exit_wait, ready_wait, stop_wait, restart_wait))
                done, _ = await asyncio.wait(
                    waiters, timeout=self._startup_timeout, return_when=asyncio.FIRST_COMPLETED
                )
                if stop_wait in done:
                    return
                if ready_wait not in done or exit_wait in done:
                    self.failure_code = "photo_child_start_failed"
                else:
                    self.failure_code = None
                    disconnect_wait = self._scope.create_task(
                        self.bridge.disconnected.wait(), name=f"disconnect-{self.generation}"
                    )
                    waiters.append(disconnect_wait)
                    await asyncio.wait(
                        (exit_wait, stop_wait, disconnect_wait, restart_wait),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if self._stop.is_set():
                        return
                    self.failure_code = (
                        "photo_child_disconnected"
                        if disconnect_wait.done()
                        else "photo_child_exited"
                    )
                self._diagnostic(
                    "process_failed",
                    {"generation": self.generation, "failure_code": self.failure_code},
                )
            except Exception:
                self.failure_code = "photo_child_unavailable"
                self._diagnostic(
                    "process_failed",
                    {"generation": self.generation, "failure_code": self.failure_code},
                )
            finally:
                try:
                    if process is not None:
                        await self._stop_child(process)
                    self.bridge.renderer_stopped(self.generation)
                    self._generation_stopped.set()
                finally:
                    for task in waiters:
                        task.cancel()
                    if waiters:
                        await asyncio.gather(*waiters, return_exceptions=True)
                    await run_owned_thread_call(directory.cleanup)
            if recovery < self._restart_limit and not self._stop.is_set():
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=0.25 * (recovery + 1))
                except TimeoutError:
                    pass

    @staticmethod
    def _write_manifest(path: Path, manifest: dict) -> None:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, separators=(",", ":"))
        if os.name != "nt":
            path.chmod(0o600)

    async def _stop_child(self, process: OverlayManagedProcess) -> None:
        self.last_shutdown_acknowledged = False
        if getattr(process, "returncode", None) is None:
            self.bridge.send({"type": "shutdown"})
            try:
                self.last_exit_code = await asyncio.wait_for(
                    process.wait_for_exit(), timeout=self._shutdown_timeout
                )
            except TimeoutError:
                await process.terminate()
                self.last_exit_code = await process.wait_for_exit()
        else:
            self.last_exit_code = await process.wait_for_exit()
        self.last_shutdown_acknowledged = self.bridge.shutdown_complete.is_set()
        await process.finish_readers()
        if self.process is process:
            self.process = None

    async def close(self) -> None:
        self._stop.set()
        if self._task is not None:
            await asyncio.shield(self._task)
        await self._scope.close()
        await self.bridge.close()
