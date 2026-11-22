from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading

import pytest

from puripuly_heart.config.process_capture_resolution import ResolvedProcessCaptureIdentity
from puripuly_heart.config.settings_vnext.schema import ProcessCaptureTargetIntent
from puripuly_heart.core.audio.process_identity import (
    SystemProcessIdentityWatcher,
    WindowsCurrentUserProcessSnapshots,
)
from puripuly_heart.core.audio.windows_process_table import (
    list_windows_processes,
    snapshot_windows_processes,
)

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows process table")

psutil = pytest.importorskip("psutil")


@pytest.fixture
def child_process():
    with subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1); sys.exit(259)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) as child:
        try:
            yield child
        finally:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)


def _identity(pid: int, instance_id: str) -> ResolvedProcessCaptureIdentity:
    return ResolvedProcessCaptureIdentity(
        pid=pid,
        target=ProcessCaptureTargetIntent.generic_executable(r"C:\Apps\Game\Game.exe"),
        instance_id=instance_id,
    )


def _restrict_process_open(monkeypatch, pid: int, *, allow_query: bool = True) -> None:
    import _winapi

    open_process = _winapi.OpenProcess

    def check_access(access: int, inherit: bool, target_pid: int) -> int:
        if target_pid == pid and not (allow_query and access == 0x1000):
            raise ctypes.WinError(5)
        return open_process(access, inherit, target_pid)

    monkeypatch.setattr(_winapi, "OpenProcess", check_access)


def test_process_list_reports_child_name_and_parent(child_process) -> None:
    expected_name = psutil.Process(child_process.pid).name()

    entries = {entry.pid: entry for entry in list_windows_processes()}

    entry = entries[child_process.pid]
    assert entry.parent_pid == os.getpid()
    assert entry.image_name.casefold() == expected_name.casefold()


def test_snapshot_reports_child_identity_like_psutil(child_process) -> None:
    process = psutil.Process(child_process.pid)

    records = {record.pid: record for record in snapshot_windows_processes()}

    record = records[child_process.pid]
    assert record.parent_pid == os.getpid()
    assert record.is_current_user is True
    assert record.executable_path is not None
    assert record.executable_path.casefold() == process.exe().casefold()
    assert f"{record.pid}:{record.create_time}" == f"{process.pid}:{process.create_time()}"


@pytest.mark.parametrize("query_only", [False, True], ids=["native-wait", "query-only"])
def test_snapshot_identity_is_accepted_by_process_identity_watcher(
    child_process, monkeypatch, query_only
) -> None:
    snapshot = next(
        item
        for item in WindowsCurrentUserProcessSnapshots().snapshots()
        if item.pid == child_process.pid
    )
    assert snapshot.instance_id is not None
    identity = _identity(snapshot.pid, snapshot.instance_id)
    if query_only:
        _restrict_process_open(monkeypatch, child_process.pid)

    callbacks = []
    watch = SystemProcessIdentityWatcher().watch(
        identity, on_terminal=lambda reason, cause: callbacks.append((reason, cause))
    )
    try:
        assert watch.identity_verified is True
        assert watch.watch_thread_alive is True
    finally:
        watch.close()
    assert watch.watch_thread_alive is False
    assert child_process.poll() is None
    assert callbacks == []


def test_query_only_process_identity_watch_reports_exit_code_259(child_process, monkeypatch) -> None:
    pid = child_process.pid
    identity = _identity(pid, f"{pid}:{psutil.Process(pid).create_time()}")
    _restrict_process_open(monkeypatch, pid)
    callbacks = []
    notified = threading.Event()

    def on_terminal(reason, cause) -> None:
        callbacks.append((reason, cause))
        notified.set()

    watch = SystemProcessIdentityWatcher().watch(identity, on_terminal)
    try:
        assert watch.identity_verified is True
        assert child_process.stdin is not None
        child_process.stdin.write(b"x")
        child_process.stdin.flush()
        assert child_process.wait(timeout=5) == 259
        assert notified.wait(timeout=5)
    finally:
        watch.close()
    assert callbacks == [("target_exited", None)]
    assert watch.watch_thread_alive is False


@pytest.mark.parametrize(
    ("allow_query", "reason"),
    [(True, "target_identity_mismatch"), (False, "identity_unavailable")],
    ids=["stale-identity", "query-denied"],
)
def test_query_only_process_identity_watch_rejects_unverified_identity(
    child_process, monkeypatch, allow_query, reason
) -> None:
    pid = child_process.pid
    created = 0.0 if allow_query else psutil.Process(pid).create_time()
    identity = _identity(pid, f"{pid}:{created}")
    _restrict_process_open(monkeypatch, pid, allow_query=allow_query)
    callbacks = []
    watch = SystemProcessIdentityWatcher().watch(
        identity, on_terminal=lambda terminal, cause: callbacks.append((terminal, cause))
    )
    try:
        assert watch.identity_verified is False
        assert watch.terminal_reason == reason
        assert watch.watch_thread_alive is False
        if allow_query:
            assert watch.failure_diagnostics is None
        else:
            assert watch.failure_diagnostics is not None
            assert watch.failure_diagnostics.code == "capture.peer.identity_unavailable"
            assert watch.failure_diagnostics.fields["winerror"] == 5
    finally:
        watch.close()
    assert child_process.poll() is None
    assert callbacks == []


def test_process_identity_watcher_rejects_already_exited_child(child_process) -> None:
    pid = child_process.pid
    identity = _identity(pid, f"{pid}:{psutil.Process(pid).create_time()}")
    child_process.terminate()
    child_process.wait(timeout=5)
    callbacks = []
    watch = SystemProcessIdentityWatcher().watch(
        identity, on_terminal=lambda reason, _cause: callbacks.append(reason)
    )
    try:
        assert watch.identity_verified is False
        assert watch.terminal_reason == "target_exited"
        assert watch.failure_diagnostics is None
    finally:
        watch.close()
    assert not callbacks


def test_snapshot_paths_and_ownership_match_psutil_for_live_processes() -> None:
    current_username = psutil.Process().username()
    expected = {
        process.info["pid"]: process.info
        for process in psutil.process_iter(["pid", "exe", "username"])
    }

    records = snapshot_windows_processes()

    compared = 0
    for record in records:
        info = expected.get(record.pid)
        if info is None:
            continue
        expected_path = info.get("exe")
        if expected_path:
            if record.executable_path is None and not psutil.pid_exists(record.pid):
                continue
            assert record.executable_path is not None, info
            assert record.executable_path.casefold() == expected_path.casefold()
            compared += 1
        if info.get("username") == current_username:
            assert record.is_current_user is True, info
    assert compared > 0
