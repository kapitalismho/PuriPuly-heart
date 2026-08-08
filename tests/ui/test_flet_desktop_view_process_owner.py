from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import Callable
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


class CancelledWaitAfterExitProcess(FakeProcess):
    async def wait(self) -> int:
        await self._exited.wait()
        raise asyncio.CancelledError


class UnconfirmedExitProcess(FakeProcess):
    async def wait(self) -> int:
        raise OSError("wait failed")


class FakeProcessJob:
    def __init__(self) -> None:
        self.assigned_pids: list[int] = []
        self.close_calls = 0

    def assign(self, pid: int) -> bool:
        self.assigned_pids.append(pid)
        return True

    def close(self) -> None:
        self.close_calls += 1


class FakeCloseRequester:
    def __init__(
        self,
        *,
        requested: bool = False,
        on_request: Callable[[], None] | None = None,
        failure_reason: str | None = "not_requested",
    ) -> None:
        self.requested = requested
        self.on_request = on_request
        self.failure_reason = failure_reason
        self.requested_pids: list[int] = []

    def request_close(self, pid: int) -> bool:
        self.requested_pids.append(pid)
        if self.on_request is not None:
            self.on_request()
        return self.requested


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
        close_requester=FakeCloseRequester(requested=True, on_request=process.exit),
    )
    assert await owner.attach(process, str(pid_file), endpoint_identity="local:1") is True

    await owner.close()

    assert process.terminate_calls == 0
    assert process.kill_calls == 0
    assert not pid_file.exists()
    assert owner.process_info is None
    assert process_job.assigned_pids == [4321]
    assert process_job.close_calls == 1


@pytest.mark.asyncio
async def test_owner_skips_pid_close_request_when_process_already_exited(tmp_path: Path) -> None:
    process = FakeProcess()
    process.exit()
    pid_file = _pid_file(tmp_path)
    close_requester = FakeCloseRequester(requested=True)
    owner = FletDesktopViewProcessOwner(
        close_requester=close_requester,
        process_job=FakeProcessJob(),
    )
    await owner.attach(process, str(pid_file))

    await owner.close()

    assert close_requester.requested_pids == []
    assert process.terminate_calls == 0
    assert process.kill_calls == 0
    assert not pid_file.exists()
    assert owner.process_info is None


@pytest.mark.asyncio
async def test_owner_recovers_when_process_wait_is_cancelled_after_exit(tmp_path: Path) -> None:
    process = CancelledWaitAfterExitProcess()
    pid_file = _pid_file(tmp_path)
    events: list[str] = []
    owner = FletDesktopViewProcessOwner(
        graceful_timeout_s=0.05,
        trace_sink=lambda event, _fields: events.append(event),
        process_job=FakeProcessJob(),
        close_requester=FakeCloseRequester(
            requested=True,
            on_request=lambda: asyncio.get_running_loop().call_soon(process.exit),
        ),
    )
    await owner.attach(process, str(pid_file))

    await owner.close()

    assert process.terminate_calls == 0
    assert process.kill_calls == 0
    assert not pid_file.exists()
    assert owner.process_info is None
    assert events[-3:] == ["process_wait_failed", "process_exited", "pid_file_removed"]


@pytest.mark.asyncio
async def test_owner_synchronous_close_admission_leaves_no_detached_task(tmp_path: Path) -> None:
    process = FakeProcess()
    pid_file = _pid_file(tmp_path)
    events: list[str] = []
    tasks_before = asyncio.all_tasks()
    owner = FletDesktopViewProcessOwner(
        graceful_timeout_s=0.01,
        terminate_timeout_s=0.05,
        trace_sink=lambda event, _fields: events.append(event),
        process_job=FakeProcessJob(),
        close_requester=FakeCloseRequester(),
    )
    await owner.attach(process, str(pid_file))

    await owner.close()

    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert not pid_file.exists()
    assert owner.process_info is None
    assert asyncio.all_tasks() == tasks_before
    assert events[-4:] == [
        "graceful_close_timeout",
        "terminate_requested",
        "process_exited",
        "pid_file_removed",
    ]


@pytest.mark.asyncio
async def test_owner_does_not_create_task_for_failed_graceful_close_request(
    tmp_path: Path,
) -> None:
    process = FakeProcess()
    pid_file = _pid_file(tmp_path)
    events: list[str] = []
    owner = FletDesktopViewProcessOwner(
        graceful_timeout_s=0.01,
        terminate_timeout_s=0.05,
        trace_sink=lambda event, _fields: events.append(event),
        process_job=FakeProcessJob(),
        close_requester=FakeCloseRequester(),
    )
    await owner.attach(process, str(pid_file))

    await asyncio.wait_for(owner.close(), timeout=0.2)

    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert not pid_file.exists()
    assert owner.process_info is None
    assert events[-6:] == [
        "graceful_close_requested",
        "graceful_close_failed",
        "graceful_close_timeout",
        "terminate_requested",
        "process_exited",
        "pid_file_removed",
    ]


@pytest.mark.asyncio
async def test_owner_trace_reports_renderer_pid_as_flet_view_parent(tmp_path: Path) -> None:
    process = FakeProcess()
    pid_file = _pid_file(tmp_path)
    events: list[tuple[str, dict[str, object]]] = []
    owner = FletDesktopViewProcessOwner(
        graceful_timeout_s=0.05,
        trace_sink=lambda event, fields: events.append((event, dict(fields))),
        process_job=FakeProcessJob(),
        close_requester=FakeCloseRequester(requested=True, on_request=process.exit),
    )
    await owner.attach(process, str(pid_file))

    await owner.close()

    assert events
    assert all(fields["parent_pid"] == os.getpid() for _event, fields in events)
    started = next(fields for event, fields in events if event == "process_started")
    assert started["pid"] == 4321


@pytest.mark.asyncio
async def test_owner_escalates_from_graceful_close_to_terminate(tmp_path: Path) -> None:
    process = FakeProcess()
    pid_file = _pid_file(tmp_path)
    events: list[str] = []
    owner = FletDesktopViewProcessOwner(
        graceful_timeout_s=0.05,
        terminate_timeout_s=0.1,
        trace_sink=lambda event, _fields: events.append(event),
        close_requester=FakeCloseRequester(),
    )
    await owner.attach(process, str(pid_file))

    await owner.close()

    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert not pid_file.exists()
    assert events == [
        "process_started",
        "job_assignment_failed",
        "stop_requested",
        "graceful_close_requested",
        "graceful_close_failed",
        "graceful_close_timeout",
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
        close_requester=FakeCloseRequester(),
    )
    await owner.attach(process, str(pid_file))

    await owner.close()

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
        close_requester=FakeCloseRequester(),
    )
    await owner.attach(process, str(pid_file))

    await owner.close()

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.returncode == 9
    assert not pid_file.exists()


@pytest.mark.asyncio
async def test_owner_repeated_close_is_idempotent(tmp_path: Path) -> None:
    process = FakeProcess()
    pid_file = _pid_file(tmp_path)
    close_requester = FakeCloseRequester()
    owner = FletDesktopViewProcessOwner(
        graceful_timeout_s=0.01,
        close_requester=close_requester,
    )
    await owner.attach(process, str(pid_file))

    await owner.close()
    await owner.close()

    assert close_requester.requested_pids == [4321]
    assert process.terminate_calls == 1
    assert process.kill_calls == 0


@pytest.mark.asyncio
async def test_owner_finishes_cleanup_when_close_caller_is_cancelled(tmp_path: Path) -> None:
    process = FakeProcess()
    pid_file = _pid_file(tmp_path)
    owner = FletDesktopViewProcessOwner(
        graceful_timeout_s=0.02,
        terminate_timeout_s=0.01,
        close_requester=FakeCloseRequester(),
    )
    await owner.attach(process, str(pid_file))

    close_task = asyncio.create_task(owner.close())
    await asyncio.sleep(0)
    close_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await close_task

    assert process.terminate_calls == 1
    assert not pid_file.exists()
    assert owner.process_info is None


@pytest.mark.asyncio
async def test_owner_retains_pid_file_and_process_when_exit_is_unconfirmed(
    tmp_path: Path,
) -> None:
    process = UnconfirmedExitProcess(terminate_exits=False, kill_exits=False)
    pid_file = _pid_file(tmp_path)
    events: list[str] = []
    owner = FletDesktopViewProcessOwner(
        graceful_timeout_s=0.01,
        terminate_timeout_s=0.01,
        trace_sink=lambda event, _fields: events.append(event),
        process_job=FakeProcessJob(),
        close_requester=FakeCloseRequester(),
    )
    await owner.attach(process, str(pid_file))

    with pytest.raises(RuntimeError, match="did not exit"):
        await owner.close()

    assert pid_file.exists()
    assert owner.process_info == (4321, str(pid_file))
    assert "pid_file_removed" not in events
    assert events[-1] == "cleanup_incomplete"


@pytest.mark.asyncio
async def test_owner_reports_unlink_failure_and_retries_retained_pid_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = FakeProcess()
    pid_file = _pid_file(tmp_path)
    events: list[str] = []
    original_unlink = Path.unlink
    unlink_calls = 0

    def fail_once(path: Path, *args: object, **kwargs: object) -> None:
        nonlocal unlink_calls
        if path == pid_file:
            unlink_calls += 1
            if unlink_calls == 1:
                raise PermissionError("pid file locked")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_once)
    close_requester = FakeCloseRequester(requested=True, on_request=process.exit)
    owner = FletDesktopViewProcessOwner(
        graceful_timeout_s=0.01,
        trace_sink=lambda event, _fields: events.append(event),
        process_job=FakeProcessJob(),
        close_requester=close_requester,
    )
    await owner.attach(process, str(pid_file))

    with pytest.raises(RuntimeError, match="PID file cleanup failed"):
        await owner.close()

    assert pid_file.exists()
    assert owner.process_info == (4321, str(pid_file))
    assert "pid_file_remove_failed" in events
    assert "pid_file_removed" not in events

    await owner.close()

    assert close_requester.requested_pids == [4321]
    assert not pid_file.exists()
    assert owner.process_info is None
    assert events[-1] == "pid_file_removed"


@pytest.mark.asyncio
async def test_owner_reaps_process_attached_after_close_started(tmp_path: Path) -> None:
    owner = FletDesktopViewProcessOwner(
        terminate_timeout_s=0.1,
        close_requester=FakeCloseRequester(),
    )
    await owner.close()
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
        close_requester=FakeCloseRequester(),
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
        close_requester=FakeCloseRequester(),
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
            close_requester=FakeCloseRequester(),
        )
        await owner.attach(process, str(pid_file), endpoint_identity=f"local:{cycle}")

        await owner.close()

        assert process.returncode is not None
        assert not pid_file.exists()
        assert owner.process_info is None
