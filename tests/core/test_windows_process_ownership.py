from __future__ import annotations

import asyncio
import ctypes
import os
import subprocess
import sys
import textwrap
from ctypes import wintypes
from pathlib import Path

import pytest

from puripuly_heart.core import windows_process_ownership


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
@pytest.mark.skipif(os.getenv("INTEGRATION") != "1", reason="requires real processes")
@pytest.mark.parametrize("abrupt", [False, True], ids=["normal-exit", "parent-loss"])
@pytest.mark.asyncio
async def test_process_lifetime_job_reaps_descendants_without_changing_root_exit(
    abrupt: bool,
) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    source_root = str(Path(windows_process_ownership.__file__).resolve().parents[2])
    program = textwrap.dedent(f"""
        import subprocess
        import sys
        sys.path.insert(0, {source_root!r})
        from puripuly_heart.core.windows_process_ownership import retain_current_process_job
        retain_current_process_job()
        child = subprocess.Popen(
            [sys.executable, "-I", "-S", "-B", "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        print(child.pid, flush=True)
        sys.stdin.buffer.read(1)
        raise SystemExit(7)
        """)
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-c",
        program,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    child_handle = None
    try:
        child_pid = int(await asyncio.wait_for(process.stdout.readline(), timeout=5.0))
        child_handle = kernel32.OpenProcess(0x00100000 | 0x1000 | 0x0001, False, child_pid)
        assert child_handle
        assert kernel32.WaitForSingleObject(child_handle, 100) == 258
        if abrupt:
            process.kill()
        else:
            process.stdin.write(b"\n")
            await process.stdin.drain()
        assert await asyncio.wait_for(process.wait(), timeout=5.0) == (1 if abrupt else 7)
        assert kernel32.WaitForSingleObject(child_handle, 5000) == 0
    finally:
        if process.returncode is None:
            process.kill()
            await asyncio.wait_for(process.wait(), timeout=5.0)
        if child_handle:
            try:
                if kernel32.WaitForSingleObject(child_handle, 0) != 0:
                    assert kernel32.TerminateProcess(child_handle, 1)
                    assert kernel32.WaitForSingleObject(child_handle, 5000) == 0
            finally:
                kernel32.CloseHandle(child_handle)
