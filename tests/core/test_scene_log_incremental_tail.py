from __future__ import annotations

import asyncio
import builtins
import time
from pathlib import Path

import pytest

from puripuly_heart.config.process_capture_resolution import ProcessSnapshot
from puripuly_heart.core.vrchat_scene_service import VrchatSceneService
from puripuly_heart.core.vrchat_scene_tailer import (
    VrchatSceneLogTailer,
    VrchatSceneLogTruncated,
)

_LOG_NAME = "output_log_2026-09-07_01-48-37.txt"
_USER_A = "usr_11111111-1111-4111-8111-111111111111"


def _line(payload: str) -> str:
    return f"2026.09.07 01:48:58 Debug      -  {payload}"


def _roster_lines() -> list[str]:
    return [
        _line("[Behaviour] Entering Room: Testville"),
        _line("[Behaviour] Joining wrld_12345678-1234-4234-8234-123456789012:12345"),
        _line(f"[Behaviour] OnPlayerJoined Hal ({_USER_A})"),
        _line('[Behaviour] Initialized PlayerAPI "Hal" is local'),
    ]


def _noise(count: int, *, start: int = 0) -> str:
    return "".join(
        _line(f"[UdonBehaviour] ambient tick {index}") + "\n"
        for index in range(start, start + count)
    )


def test_plain_append_is_read_incrementally(tmp_path: Path) -> None:
    path = tmp_path / _LOG_NAME
    path.write_text("first\nsecond\n", encoding="utf-8")
    tailer = VrchatSceneLogTailer(tmp_path)

    assert tailer.begin_replay(path) == ["first", "second"]

    for index in range(3):
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(f"appended-{index}\n")
        assert tailer.read_new_lines() == [f"appended-{index}"]


def test_same_name_replacement_is_still_detected(tmp_path: Path) -> None:
    path = tmp_path / _LOG_NAME
    path.write_text("first\nsecond\nthird\n", encoding="utf-8")
    tailer = VrchatSceneLogTailer(tmp_path)
    assert tailer.begin_replay(path) == ["first", "second", "third"]

    path.write_text("replacement\n", encoding="utf-8")
    with pytest.raises(VrchatSceneLogTruncated):
        tailer.read_new_lines()
    assert tailer.begin_replay(path) == ["replacement"]
    with open(path, "a", encoding="utf-8") as stream:
        stream.write("appended\n")
    assert tailer.read_new_lines() == ["appended"]


async def test_growing_log_is_tailed_without_full_replays(tmp_path: Path) -> None:
    create_time = time.time() - 5
    log_name = time.strftime("output_log_%Y-%m-%d_%H-%M-%S.txt", time.localtime(create_time))
    path = tmp_path / log_name
    path.write_text("\n".join(_roster_lines()) + "\n", encoding="utf-8")

    read_bytes = 0
    real_open = builtins.open

    def counting_open(file, mode="r", *args, **kwargs):
        nonlocal read_bytes
        if "r" in mode and str(file) == str(path):
            stream = real_open(file, mode, *args, **kwargs)

            class _Counting:
                def __enter__(self):
                    return self

                def __exit__(self, *exc_info):
                    stream.close()
                    return False

                def seek(self, offset):
                    return stream.seek(offset)

                def read(self, size=-1):
                    nonlocal read_bytes
                    chunk = stream.read(size)
                    read_bytes += len(chunk)
                    return chunk

            return _Counting()
        return real_open(file, mode, *args, **kwargs)

    import puripuly_heart.core.vrchat_scene_tailer as tailer_module

    class FakeSnapshots:
        def __init__(self) -> None:
            self.items = (
                ProcessSnapshot(
                    pid=4242,
                    parent_pid=None,
                    is_current_user=True,
                    executable_path="C:\\VRChat\\VRChat.exe",
                    instance_id=f"4242:{create_time}",
                ),
            )

        def snapshots(self):
            return self.items

    class FakeWatchHandle:
        def close(self) -> None:
            return None

    class FakeWatcherFactory:
        def watch(self, identity, on_terminal):
            return FakeWatchHandle()

    snapshots = FakeSnapshots()
    service = VrchatSceneService(
        snapshots=snapshots,
        watcher_factory=FakeWatcherFactory(),
        log_directory=tmp_path,
        poll_interval_s=0.02,
        quiescence_s=0.05,
    )

    tailer_module.open = counting_open  # type: ignore[attr-defined]
    try:
        await service.start()
        for _ in range(200):
            if service.snapshot().status == "ready":
                break
            await asyncio.sleep(0.02)
        assert service.snapshot().status == "ready"
        ready_at = read_bytes

        for _ in range(20):
            with open(path, "a", encoding="utf-8") as stream:
                stream.write(_noise(30))
            await asyncio.sleep(0.05)

        assert service.snapshot().status == "ready"
        assert service.snapshot().participant_count == 1
        consumed = read_bytes - ready_at
        grown = path.stat().st_size - len("\n".join(_roster_lines())) - 1
        assert (
            consumed <= grown * 2
        ), f"tailed {consumed} bytes for a log that grew by {grown} bytes"
    finally:
        del tailer_module.open
        await service.close()
