from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from puripuly_heart.core.overlay.protocol import OverlayPresentationSnapshot
from puripuly_heart.ui.desktop_overlay import FletDesktopRendererWindow
from puripuly_heart.ui.flet_desktop_runtime import (
    FletDesktopViewProcessOwner,
    _WindowsKillOnCloseProcessJob,
)


class FakeProcess:
    def __init__(
        self,
        *,
        pid: int = 4321,
        terminate_exits: bool = True,
        kill_exits: bool = True,
        terminate_raises: bool = False,
    ) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.terminate_exits = terminate_exits
        self.kill_exits = kill_exits
        self.terminate_raises = terminate_raises
        self.terminate_calls = 0
        self.kill_calls = 0
        self._exited = asyncio.Event()

    async def wait(self) -> int:
        await self._exited.wait()
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self.terminate_raises:
            raise OSError("terminate failed")
        if self.terminate_exits:
            self.exit(15)

    def kill(self) -> None:
        self.kill_calls += 1
        if self.kill_exits:
            self.exit(9)

    def exit(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self._exited.set()


class FakeProcessJob:
    def __init__(self) -> None:
        self.assigned_pids: list[int] = []
        self.close_calls = 0

    def assign(self, pid: int) -> bool:
        self.assigned_pids.append(pid)
        return True

    def close(self) -> None:
        self.close_calls += 1


def _pid_file(tmp_path: Path) -> Path:
    path = tmp_path / "flet-view.pid"
    path.write_text("4321", encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_owner_normal_close_waits_for_exit_before_removing_pid_file(
    tmp_path: Path,
) -> None:
    process = FakeProcess()
    pid_file = _pid_file(tmp_path)
    process_job = FakeProcessJob()
    owner = FletDesktopViewProcessOwner(
        graceful_timeout_s=0.05,
        process_job=process_job,
    )
    assert await owner.attach(process, str(pid_file), endpoint_identity="local:1") is True

    async def graceful_close() -> None:
        assert pid_file.exists()
        process.exit()

    await owner.close(graceful_close)

    assert process.terminate_calls == 0
    assert process.kill_calls == 0
    assert not pid_file.exists()
    assert owner.process_info is None
    assert process_job.assigned_pids == [4321]
    assert process_job.close_calls == 1


@pytest.mark.asyncio
async def test_owner_escalates_from_graceful_close_to_terminate(tmp_path: Path) -> None:
    process = FakeProcess()
    pid_file = _pid_file(tmp_path)
    events: list[str] = []
    owner = FletDesktopViewProcessOwner(
        graceful_timeout_s=0.05,
        terminate_timeout_s=0.1,
        trace_sink=lambda event, _fields: events.append(event),
    )
    await owner.attach(process, str(pid_file))

    await owner.close(lambda: None)

    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert not pid_file.exists()
    assert events == [
        "process_started",
        "job_assignment_failed",
        "stop_requested",
        "graceful_close_requested",
        "terminate_requested",
        "process_exited",
        "pid_file_removed",
    ]


@pytest.mark.asyncio
async def test_owner_escalates_from_failed_terminate_to_kill(tmp_path: Path) -> None:
    process = FakeProcess(terminate_exits=False)
    pid_file = _pid_file(tmp_path)
    owner = FletDesktopViewProcessOwner(
        graceful_timeout_s=0.01,
        terminate_timeout_s=0.01,
    )
    await owner.attach(process, str(pid_file))

    await owner.close(lambda: None)

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.returncode == 9
    assert not pid_file.exists()


@pytest.mark.asyncio
async def test_owner_escalates_from_terminate_exception_to_kill(tmp_path: Path) -> None:
    process = FakeProcess(terminate_raises=True)
    pid_file = _pid_file(tmp_path)
    owner = FletDesktopViewProcessOwner(
        graceful_timeout_s=0.01,
        terminate_timeout_s=0.01,
    )
    await owner.attach(process, str(pid_file))

    await owner.close(lambda: None)

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.returncode == 9
    assert not pid_file.exists()


@pytest.mark.asyncio
async def test_owner_repeated_close_is_idempotent(tmp_path: Path) -> None:
    process = FakeProcess()
    pid_file = _pid_file(tmp_path)
    graceful_calls = 0
    owner = FletDesktopViewProcessOwner(graceful_timeout_s=0.01)
    await owner.attach(process, str(pid_file))

    def graceful_close() -> None:
        nonlocal graceful_calls
        graceful_calls += 1

    await owner.close(graceful_close)
    await owner.close(graceful_close)

    assert graceful_calls == 1
    assert process.terminate_calls == 1
    assert process.kill_calls == 0


@pytest.mark.asyncio
async def test_owner_finishes_cleanup_when_close_caller_is_cancelled(tmp_path: Path) -> None:
    process = FakeProcess()
    pid_file = _pid_file(tmp_path)
    owner = FletDesktopViewProcessOwner(
        graceful_timeout_s=0.02,
        terminate_timeout_s=0.01,
    )
    await owner.attach(process, str(pid_file))

    close_task = asyncio.create_task(owner.close(lambda: None))
    await asyncio.sleep(0)
    close_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await close_task

    assert process.terminate_calls == 1
    assert not pid_file.exists()
    assert owner.process_info is None


@pytest.mark.asyncio
async def test_owner_reaps_process_attached_after_close_started(tmp_path: Path) -> None:
    owner = FletDesktopViewProcessOwner(terminate_timeout_s=0.1)
    await owner.close(lambda: None)
    process = FakeProcess()
    pid_file = _pid_file(tmp_path)

    accepted = await owner.attach(process, str(pid_file))

    assert accepted is False
    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert not pid_file.exists()


@pytest.mark.asyncio
async def test_renderer_startup_timeout_reaps_process_before_page_creation(
    tmp_path: Path,
) -> None:
    process = FakeProcess()
    pid_file = _pid_file(tmp_path)
    owner = FletDesktopViewProcessOwner(
        graceful_timeout_s=0.01,
        terminate_timeout_s=0.01,
    )
    release_runner = asyncio.Event()

    async def app_runner(_target: object) -> None:
        await owner.attach(process, str(pid_file))
        await release_runner.wait()

    window = FletDesktopRendererWindow(
        app_runner=app_runner,
        startup_timeout_s=0.01,
        view_process_owner=owner,
    )

    with pytest.raises(RuntimeError, match="page was not created"):
        await window.start(OverlayPresentationSnapshot())
    await window.close()

    assert process.terminate_calls == 1
    assert not pid_file.exists()


@pytest.mark.asyncio
async def test_renderer_startup_exception_after_spawn_reaps_process(tmp_path: Path) -> None:
    process = FakeProcess()
    pid_file = _pid_file(tmp_path)
    owner = FletDesktopViewProcessOwner(
        graceful_timeout_s=0.01,
        terminate_timeout_s=0.01,
    )

    async def app_runner(_target: object) -> None:
        await owner.attach(process, str(pid_file))
        raise RuntimeError("spawned runner failed")

    window = FletDesktopRendererWindow(
        app_runner=app_runner,
        startup_timeout_s=0.1,
        view_process_owner=owner,
    )

    with pytest.raises(RuntimeError, match="spawned runner failed"):
        await window.start(OverlayPresentationSnapshot())
    await window.close()

    assert process.terminate_calls == 1
    assert not pid_file.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
@pytest.mark.asyncio
async def test_windows_kill_on_close_job_reaps_assigned_real_process() -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    job = _WindowsKillOnCloseProcessJob()
    try:
        assert process.pid is not None
        assert job.assign(process.pid) is True

        job.close()

        await asyncio.wait_for(process.wait(), timeout=5.0)
        assert process.returncode is not None
    finally:
        job.close()
        if process.returncode is None:
            process.kill()
            await process.wait()


@pytest.mark.skipif(os.name != "nt", reason="Windows process lifecycle evidence")
@pytest.mark.asyncio
async def test_owner_reaps_real_process_and_pid_file_across_ten_cycles(
    tmp_path: Path,
) -> None:
    for cycle in range(10):
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        pid_file = tmp_path / f"flet-view-{cycle}.pid"
        pid_file.write_text(str(process.pid), encoding="utf-8")
        owner = FletDesktopViewProcessOwner(
            graceful_timeout_s=0.01,
            terminate_timeout_s=1.0,
        )
        await owner.attach(process, str(pid_file), endpoint_identity=f"local:{cycle}")

        await owner.close(lambda: None)

        assert process.returncode is not None
        assert not pid_file.exists()
        assert owner.process_info is None
