from __future__ import annotations

import os
import subprocess
import sys

import pytest

from puripuly_heart.core.audio.process_identity import (
    PsutilProcessIdentityWatcher,
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
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        yield child
    finally:
        child.terminate()
        child.wait(timeout=5)


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


def test_snapshot_identity_is_accepted_by_process_identity_watcher(child_process) -> None:
    from puripuly_heart.config.process_capture_resolution import (
        ResolvedProcessCaptureIdentity,
    )
    from puripuly_heart.config.settings_vnext.schema import ProcessCaptureTargetIntent

    snapshot = next(
        item
        for item in WindowsCurrentUserProcessSnapshots().snapshots()
        if item.pid == child_process.pid
    )
    assert snapshot.instance_id is not None
    identity = ResolvedProcessCaptureIdentity(
        pid=snapshot.pid,
        target=ProcessCaptureTargetIntent.generic_executable(r"C:\Apps\Game\Game.exe"),
        instance_id=snapshot.instance_id,
    )

    watch = PsutilProcessIdentityWatcher().watch(identity, on_terminal=lambda: None)
    try:
        assert watch.identity_verified is True
    finally:
        watch.close()


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
