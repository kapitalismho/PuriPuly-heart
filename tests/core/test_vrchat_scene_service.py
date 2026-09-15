from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from puripuly_heart.config.process_capture_resolution import ProcessSnapshot
from puripuly_heart.core.vrchat_scene_service import VrchatSceneService

_USER_A = "usr_11111111-1111-4111-8111-111111111111"
_USER_B = "usr_22222222-2222-4222-8222-222222222222"
_USER_C = "usr_33333333-3333-4333-8333-333333333333"


def _line(payload: str) -> str:
    return f"2026.09.07 01:48:58 Debug      -  [Behaviour] {payload}"


def _join(name: str, user_id: str) -> str:
    return _line(f"OnPlayerJoined {name} ({user_id})")


def _leave(name: str, user_id: str) -> str:
    return _line(f"OnPlayerLeft {name} ({user_id})")


def _local_init(name: str) -> str:
    return _line(f'Initialized PlayerAPI "{name}" is local')


def _remote_init(name: str) -> str:
    return _line(f'Initialized PlayerAPI "{name}" is remote')


def _roster_block(names: tuple[str, ...], ids: tuple[str, ...], local: str) -> list[str]:
    lines = [
        _line("Entering Room: Testville"),
        _line("Joining wrld_12345678-1234-4234-8234-123456789012:12345"),
    ]
    for name, user_id in zip(names, ids):
        lines.append(_join(name, user_id))
    for name in names:
        if name == local:
            lines.append(_local_init(name))
        else:
            lines.append(_remote_init(name))
    return lines


def _log_name(create_time: float) -> str:
    return time.strftime("output_log_%Y-%m-%d_%H-%M-%S.txt", time.localtime(create_time))


class FakeSnapshots:
    def __init__(self) -> None:
        self.items: tuple[ProcessSnapshot, ...] = ()

    def snapshots(self) -> tuple[ProcessSnapshot, ...]:
        return self.items


class FakeWatchHandle:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeWatcherFactory:
    def __init__(self, snapshots: FakeSnapshots) -> None:
        self.snapshots = snapshots
        self.terminals: list[Callable[[], None]] = []
        self.handles: list[FakeWatchHandle] = []

    def watch(self, identity: object, on_terminal: Callable[[], None]) -> FakeWatchHandle:
        handle = FakeWatchHandle()
        self.handles.append(handle)

        def terminal() -> None:
            self.snapshots.items = ()
            on_terminal()

        self.terminals.append(terminal)
        return handle

    def latest_terminal(self) -> Callable[[], None]:
        return self.terminals[-1]


def _snapshot(pid: int, executable_path: str, create_time: float) -> ProcessSnapshot:
    return ProcessSnapshot(
        pid=pid,
        parent_pid=None,
        is_current_user=True,
        executable_path=executable_path,
        instance_id=f"{pid}:{create_time}",
    )


def _service(
    snapshots: FakeSnapshots,
    watcher: FakeWatcherFactory,
    log_directory: Path,
) -> VrchatSceneService:
    return VrchatSceneService(
        snapshots=snapshots,
        watcher_factory=watcher,  # type: ignore[arg-type]
        log_directory=log_directory,
        poll_interval_s=0.02,
        quiescence_s=0.05,
    )


async def _wait_for_status(service: VrchatSceneService, status: str) -> None:
    for _ in range(200):
        if service.snapshot().status == status:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"scene never reached {status}: {service.snapshot()}")


async def test_no_process_means_unavailable(tmp_path: Path) -> None:
    snapshots = FakeSnapshots()
    service = _service(snapshots, FakeWatcherFactory(snapshots), tmp_path)
    await service.start()
    try:
        await asyncio.sleep(0.1)
        snapshot = service.snapshot()
        assert snapshot.status == "unavailable"
        assert snapshot.participant_count is None
    finally:
        await service.close()


async def test_process_start_after_app_start_reconstructs_roster(tmp_path: Path) -> None:
    snapshots = FakeSnapshots()
    watcher = FakeWatcherFactory(snapshots)
    service = _service(snapshots, watcher, tmp_path)
    await service.start()
    try:
        await asyncio.sleep(0.08)
        assert service.snapshot().status == "unavailable"

        create_time = time.time() - 5
        (tmp_path / _log_name(create_time)).write_text(
            "\n".join(_roster_block(("Self", "Peer"), (_USER_A, _USER_B), "Self")) + "\n",
            encoding="utf-8",
        )
        snapshots.items = (_snapshot(4242, "C:\\VRChat\\VRChat.exe", create_time),)

        await _wait_for_status(service, "ready")
        assert service.snapshot().participant_count == 2
    finally:
        await service.close()


async def test_late_startup_inside_populated_instance_replays_roster(tmp_path: Path) -> None:
    create_time = time.time() - 5
    (tmp_path / _log_name(create_time)).write_text(
        "\n".join(_roster_block(("Self", "Peer", "Third"), (_USER_A, _USER_B, _USER_C), "Self"))
        + "\n",
        encoding="utf-8",
    )
    snapshots = FakeSnapshots()
    snapshots.items = (_snapshot(4242, "C:\\VRChat\\VRChat.exe", create_time),)
    service = _service(snapshots, FakeWatcherFactory(snapshots), tmp_path)
    await service.start()
    try:
        await _wait_for_status(service, "ready")
        assert service.snapshot().participant_count == 3
    finally:
        await service.close()


async def test_live_join_and_leave_update_ready_count(tmp_path: Path) -> None:
    create_time = time.time() - 5
    log_path = tmp_path / _log_name(create_time)
    log_path.write_text(
        "\n".join(_roster_block(("Self", "Peer"), (_USER_A, _USER_B), "Self")) + "\n",
        encoding="utf-8",
    )
    snapshots = FakeSnapshots()
    snapshots.items = (_snapshot(4242, "C:\\VRChat\\VRChat.exe", create_time),)
    service = _service(snapshots, FakeWatcherFactory(snapshots), tmp_path)
    await service.start()
    try:
        await _wait_for_status(service, "ready")

        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write(_join("Third", _USER_C) + "\n")
            stream.write(_remote_init("Third") + "\n")
        for _ in range(200):
            if service.snapshot().participant_count == 3:
                break
            await asyncio.sleep(0.02)
        assert service.snapshot().participant_count == 3

        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write(_leave("Peer", _USER_B) + "\n")
        for _ in range(200):
            if service.snapshot().participant_count == 2:
                break
            await asyncio.sleep(0.02)
        assert service.snapshot().participant_count == 2
    finally:
        await service.close()


async def test_instance_transition_clears_old_members(tmp_path: Path) -> None:
    create_time = time.time() - 5
    log_path = tmp_path / _log_name(create_time)
    log_path.write_text(
        "\n".join(_roster_block(("Self", "Peer"), (_USER_A, _USER_B), "Self")) + "\n",
        encoding="utf-8",
    )
    snapshots = FakeSnapshots()
    snapshots.items = (_snapshot(4242, "C:\\VRChat\\VRChat.exe", create_time),)
    service = _service(snapshots, FakeWatcherFactory(snapshots), tmp_path)
    await service.start()
    try:
        await _wait_for_status(service, "ready")

        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write(_line("OnLeftRoom") + "\n")
        await _wait_for_status(service, "unavailable")

        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write("\n".join(_roster_block(("Self",), (_USER_A,), "Self")) + "\n")
        await _wait_for_status(service, "ready")
        assert service.snapshot().participant_count == 1
    finally:
        await service.close()


async def test_unknown_leave_degrades_withheld_count(tmp_path: Path) -> None:
    create_time = time.time() - 5
    log_path = tmp_path / _log_name(create_time)
    log_path.write_text(
        "\n".join(_roster_block(("Self", "Peer"), (_USER_A, _USER_B), "Self")) + "\n",
        encoding="utf-8",
    )
    snapshots = FakeSnapshots()
    snapshots.items = (_snapshot(4242, "C:\\VRChat\\VRChat.exe", create_time),)
    service = _service(snapshots, FakeWatcherFactory(snapshots), tmp_path)
    await service.start()
    try:
        await _wait_for_status(service, "ready")

        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write(_leave("Ghost", "usr_99999999-9999-4999-8999-999999999999") + "\n")
        await _wait_for_status(service, "degraded")
        assert service.snapshot().participant_count is None
    finally:
        await service.close()


async def test_process_exit_clears_snapshot_immediately(tmp_path: Path) -> None:
    create_time = time.time() - 5
    (tmp_path / _log_name(create_time)).write_text(
        "\n".join(_roster_block(("Self", "Peer"), (_USER_A, _USER_B), "Self")) + "\n",
        encoding="utf-8",
    )
    snapshots = FakeSnapshots()
    snapshots.items = (_snapshot(4242, "C:\\VRChat\\VRChat.exe", create_time),)
    watcher = FakeWatcherFactory(snapshots)
    service = _service(snapshots, watcher, tmp_path)
    await service.start()
    try:
        await _wait_for_status(service, "ready")

        watcher.latest_terminal()()
        await _wait_for_status(service, "unavailable")
        assert service.snapshot().participant_count is None
    finally:
        await service.close()


async def test_partial_appended_line_produces_no_phantom_event(tmp_path: Path) -> None:
    create_time = time.time() - 5
    log_path = tmp_path / _log_name(create_time)
    log_path.write_text(
        "\n".join(_roster_block(("Self",), (_USER_A,), "Self")) + "\n",
        encoding="utf-8",
    )
    snapshots = FakeSnapshots()
    snapshots.items = (_snapshot(4242, "C:\\VRChat\\VRChat.exe", create_time),)
    service = _service(snapshots, FakeWatcherFactory(snapshots), tmp_path)
    await service.start()
    try:
        await _wait_for_status(service, "ready")

        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write("2026.09.07 01:49:00 Debug      -  [Behaviour] OnPlayerJoined Parti")
            stream.flush()
        await asyncio.sleep(0.15)
        assert service.snapshot().participant_count == 1

        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write(f"alPeer ({_USER_B})\n")
            stream.write(_remote_init("PartialPeer") + "\n")
        for _ in range(200):
            if service.snapshot().participant_count == 2:
                break
            await asyncio.sleep(0.02)
        assert service.snapshot().participant_count == 2
    finally:
        await service.close()


async def test_log_rotation_rebuilds_new_roster(tmp_path: Path) -> None:
    first_start = time.time() - 300
    (tmp_path / _log_name(first_start)).write_text(
        "\n".join(_roster_block(("Self", "Peer"), (_USER_A, _USER_B), "Self")) + "\n",
        encoding="utf-8",
    )
    snapshots = FakeSnapshots()
    snapshots.items = (_snapshot(4242, "C:\\VRChat\\VRChat.exe", first_start),)
    service = _service(snapshots, FakeWatcherFactory(snapshots), tmp_path)
    await service.start()
    try:
        await _wait_for_status(service, "ready")

        second_start = time.time() - 5
        (tmp_path / _log_name(second_start)).write_text(
            "\n".join(_roster_block(("Self",), (_USER_A,), "Self")) + "\n",
            encoding="utf-8",
        )
        snapshots.items = (_snapshot(6100, "C:\\VRChat\\VRChat.exe", second_start),)

        for _ in range(300):
            snapshot = service.snapshot()
            if snapshot.status == "ready" and snapshot.participant_count == 1:
                break
            await asyncio.sleep(0.02)
        assert service.snapshot().participant_count == 1
    finally:
        await service.close()


async def test_truncation_never_exposes_stale_ready_count(tmp_path: Path) -> None:
    create_time = time.time() - 5
    log_path = tmp_path / _log_name(create_time)
    log_path.write_text(
        "\n".join(_roster_block(("Self", "Peer"), (_USER_A, _USER_B), "Self")) + "\n",
        encoding="utf-8",
    )
    snapshots = FakeSnapshots()
    snapshots.items = (_snapshot(4242, "C:\\VRChat\\VRChat.exe", create_time),)
    service = _service(snapshots, FakeWatcherFactory(snapshots), tmp_path)
    await service.start()
    try:
        await _wait_for_status(service, "ready")

        log_path.write_text(
            "\n".join(_roster_block(("Self",), (_USER_A,), "Self")) + "\n",
            encoding="utf-8",
        )
        for _ in range(300):
            snapshot = service.snapshot()
            if snapshot.status != "ready" or snapshot.participant_count == 1:
                break
            await asyncio.sleep(0.02)
        snapshot = service.snapshot()
        assert snapshot.participant_count in (None, 1)
        if snapshot.status == "ready":
            assert snapshot.participant_count == 1
    finally:
        await service.close()


async def test_deleted_log_degrades_instead_of_serving_stale_ready(tmp_path: Path) -> None:
    create_time = time.time() - 5
    log_path = tmp_path / _log_name(create_time)
    log_path.write_text(
        "\n".join(_roster_block(("Self", "Peer"), (_USER_A, _USER_B), "Self")) + "\n",
        encoding="utf-8",
    )
    snapshots = FakeSnapshots()
    snapshots.items = (_snapshot(4242, "C:\\VRChat\\VRChat.exe", create_time),)
    service = _service(snapshots, FakeWatcherFactory(snapshots), tmp_path)
    await service.start()
    try:
        await _wait_for_status(service, "ready")
        log_path.unlink()
        await _wait_for_status(service, "degraded")
        assert service.snapshot().participant_count is None
    finally:
        await service.close()


async def test_malformed_log_withholds_count_and_hides_payload(tmp_path: Path) -> None:
    create_time = time.time() - 5
    (tmp_path / _log_name(create_time)).write_text(
        "\n".join(
            [
                _line("Entering Room: Testville"),
                _line("Joining wrld_12345678-1234-4234-8234-123456789012:1"),
                _line("OnPlayerJoined NamelessWanderer"),
                _line("[UdonBehaviour] secret world payload https://example.invalid/x"),
                _line("Joining or Creating Room: Testville"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    snapshots = FakeSnapshots()
    snapshots.items = (_snapshot(4242, "C:\\VRChat\\VRChat.exe", create_time),)
    service = _service(snapshots, FakeWatcherFactory(snapshots), tmp_path)
    await service.start()
    try:
        await asyncio.sleep(0.3)
        snapshot = service.snapshot()
        assert snapshot.participant_count is None
        assert snapshot.status in ("syncing", "degraded")
        assert "NamelessWanderer" not in repr(snapshot)
        assert "example.invalid" not in repr(snapshot)
        assert "NamelessWanderer" not in repr(service.lifecycle_owner_snapshot())
    finally:
        await service.close()


async def test_non_ready_statuses_never_carry_count(tmp_path: Path) -> None:
    snapshots = FakeSnapshots()
    service = _service(snapshots, FakeWatcherFactory(snapshots), tmp_path)
    await service.start()
    try:
        await asyncio.sleep(0.1)
        assert service.snapshot().participant_count is None
    finally:
        await service.close()


async def test_close_clears_state_and_releases_tasks(tmp_path: Path) -> None:
    create_time = time.time() - 5
    (tmp_path / _log_name(create_time)).write_text(
        "\n".join(_roster_block(("Self",), (_USER_A,), "Self")) + "\n",
        encoding="utf-8",
    )
    snapshots = FakeSnapshots()
    snapshots.items = (_snapshot(4242, "C:\\VRChat\\VRChat.exe", create_time),)
    watcher = FakeWatcherFactory(snapshots)
    service = _service(snapshots, watcher, tmp_path)
    await service.start()
    await _wait_for_status(service, "ready")

    await service.close()

    assert service.snapshot().status == "unavailable"
    assert service.snapshot().participant_count is None
    assert service._scope.active_task_names == ()
    assert all(handle.closed for handle in watcher.handles)

    await service.start()
    try:
        await _wait_for_status(service, "ready")
        assert service.snapshot().participant_count == 1
    finally:
        await service.close()


async def test_generation_rejects_terminal_from_retired_lifecycle(tmp_path: Path) -> None:
    create_time = time.time() - 5
    (tmp_path / _log_name(create_time)).write_text(
        "\n".join(_roster_block(("Self",), (_USER_A,), "Self")) + "\n",
        encoding="utf-8",
    )
    snapshots = FakeSnapshots()
    snapshots.items = (_snapshot(4242, "C:\\VRChat\\VRChat.exe", create_time),)
    watcher = FakeWatcherFactory(snapshots)
    service = _service(snapshots, watcher, tmp_path)
    await service.start()
    try:
        await _wait_for_status(service, "ready")
        stale_terminal = watcher.terminals[0]

        snapshots.items = ()
        await _wait_for_status(service, "unavailable")

        stale_terminal()
        await asyncio.sleep(0.1)
        assert service.snapshot().status == "unavailable"
    finally:
        await service.close()


def test_snapshot_hides_ids_and_names() -> None:
    snapshots = FakeSnapshots()
    service = _service(snapshots, FakeWatcherFactory(snapshots), Path("."))

    assert _USER_A not in repr(service.snapshot())
    assert _USER_A not in repr(service.lifecycle_owner_snapshot())
    assert service.lifecycle_owner_snapshot()["owner"] == "VrchatSceneService"


@pytest.mark.parametrize("executable", ["C:\\Apps\\Other.exe", "", None])
async def test_non_vrchat_processes_are_ignored(tmp_path: Path, executable: object) -> None:
    snapshots = FakeSnapshots()
    snapshots.items = (
        ProcessSnapshot(
            pid=4242,
            parent_pid=None,
            is_current_user=True,
            executable_path=executable,  # type: ignore[arg-type]
            instance_id="4242:1234567890.0",
        ),
    )
    service = _service(snapshots, FakeWatcherFactory(snapshots), tmp_path)
    await service.start()
    try:
        await asyncio.sleep(0.1)
        assert service.snapshot().status == "unavailable"
    finally:
        await service.close()


async def test_restart_rejects_stale_previous_log_until_delayed_fresh_log(
    tmp_path: Path,
) -> None:
    old_start = time.time() - 900
    old_path = tmp_path / _log_name(old_start)
    old_path.write_text(
        "\n".join(_roster_block(("OldSelf", "OldPeer"), (_USER_A, _USER_B), "OldSelf")) + "\n",
        encoding="utf-8",
    )
    exited = time.time() - 60
    os.utime(old_path, (exited, exited))
    snapshots = FakeSnapshots()
    watcher = FakeWatcherFactory(snapshots)
    service = _service(snapshots, watcher, tmp_path)
    await service.start()
    try:
        create_time = time.time() - 5
        snapshots.items = (_snapshot(4242, "C:\\VRChat\\VRChat.exe", create_time),)

        await asyncio.sleep(0.25)
        assert service.snapshot().status in ("unavailable", "syncing")
        assert service.snapshot().participant_count is None

        (tmp_path / _log_name(create_time)).write_text(
            "\n".join(_roster_block(("NewSelf",), (_USER_C,), "NewSelf")) + "\n",
            encoding="utf-8",
        )
        await _wait_for_status(service, "ready")
        assert service.snapshot().participant_count == 1
    finally:
        await service.close()


async def test_restart_rejects_prior_log_with_recent_exit_flush(tmp_path: Path) -> None:
    old_start = time.time() - 900
    old_path = tmp_path / _log_name(old_start)
    old_path.write_text(
        "\n".join(_roster_block(("OldSelf", "OldPeer"), (_USER_A, _USER_B), "OldSelf")) + "\n",
        encoding="utf-8",
    )
    snapshots = FakeSnapshots()
    watcher = FakeWatcherFactory(snapshots)
    service = _service(snapshots, watcher, tmp_path)
    await service.start()
    try:
        create_time = time.time() - 5
        snapshots.items = (_snapshot(4242, "C:\\VRChat\\VRChat.exe", create_time),)
        await asyncio.sleep(0.15)
        with open(old_path, "a", encoding="utf-8") as stream:
            stream.write(
                _line("OnPlayerJoined LateFlush (usr_99999999-9999-4999-8999-999999999999)")
            )
            stream.write("\n")
        await asyncio.sleep(0.25)
        assert service.snapshot().status in ("unavailable", "syncing")
        assert service.snapshot().participant_count is None

        (tmp_path / _log_name(create_time)).write_text(
            "\n".join(_roster_block(("NewSelf",), (_USER_C,), "NewSelf")) + "\n",
            encoding="utf-8",
        )
        await _wait_for_status(service, "ready")
        assert service.snapshot().participant_count == 1
    finally:
        await service.close()


async def test_rotation_substantially_after_process_start_rebuilds(tmp_path: Path) -> None:
    create_time = time.time() - 5
    first_path = tmp_path / _log_name(create_time)
    first_path.write_text(
        "\n".join(_roster_block(("Self", "Peer"), (_USER_A, _USER_B), "Self")) + "\n",
        encoding="utf-8",
    )
    snapshots = FakeSnapshots()
    snapshots.items = (_snapshot(4242, "C:\\VRChat\\VRChat.exe", create_time),)
    service = _service(snapshots, FakeWatcherFactory(snapshots), tmp_path)
    await service.start()
    try:
        await _wait_for_status(service, "ready")
        assert service.snapshot().participant_count == 2

        rotated_name = time.strftime("output_log_%Y-%m-%d_%H-%M-%S.txt", time.localtime())
        (tmp_path / rotated_name).write_text(
            "\n".join(_roster_block(("Self",), (_USER_A,), "Self")) + "\n",
            encoding="utf-8",
        )
        for _ in range(300):
            snapshot = service.snapshot()
            if snapshot.status == "ready" and snapshot.participant_count == 1:
                break
            await asyncio.sleep(0.02)
        assert service.snapshot().participant_count == 1
    finally:
        await service.close()


async def test_late_app_start_long_running_process_selects_active_log(
    tmp_path: Path,
) -> None:
    create_time = time.time() - 7200
    (tmp_path / _log_name(create_time)).write_text(
        "\n".join(_roster_block(("Self", "Peer"), (_USER_A, _USER_B), "Self")) + "\n",
        encoding="utf-8",
    )
    snapshots = FakeSnapshots()
    snapshots.items = (_snapshot(4242, "C:\\VRChat\\VRChat.exe", create_time),)
    service = _service(snapshots, FakeWatcherFactory(snapshots), tmp_path)
    await service.start()
    try:
        await _wait_for_status(service, "ready")
        assert service.snapshot().participant_count == 2
    finally:
        await service.close()


async def test_same_size_replacement_rebuilds_without_stale_ready(tmp_path: Path) -> None:
    create_time = time.time() - 5
    log_path = tmp_path / _log_name(create_time)
    log_path.write_text(
        "\n".join(_roster_block(("Self", "Peer"), (_USER_A, _USER_B), "Self")) + "\n",
        encoding="utf-8",
    )
    snapshots = FakeSnapshots()
    snapshots.items = (_snapshot(4242, "C:\\VRChat\\VRChat.exe", create_time),)
    service = _service(snapshots, FakeWatcherFactory(snapshots), tmp_path)
    await service.start()
    try:
        await _wait_for_status(service, "ready")
        assert service.snapshot().participant_count == 2

        replacement = "\n".join(_roster_block(("Self",), (_USER_A,), "Self")) + "\n"
        padding = log_path.stat().st_size - len(replacement.encode("utf-8")) - 1
        assert padding > 0
        replacement += "x" * padding + "\n"
        staging = tmp_path / "output_log_staging.txt"
        staging.write_bytes(replacement.encode("utf-8"))
        assert staging.stat().st_size == log_path.stat().st_size
        os.replace(staging, log_path)

        stale_ready_seen = False
        for _ in range(300):
            snapshot = service.snapshot()
            if snapshot.status != "ready" or snapshot.participant_count != 2:
                break
            await asyncio.sleep(0.02)
        for _ in range(300):
            snapshot = service.snapshot()
            if snapshot.status == "ready" and snapshot.participant_count == 2:
                stale_ready_seen = True
            if snapshot.status == "ready" and snapshot.participant_count == 1:
                break
            await asyncio.sleep(0.02)
        assert stale_ready_seen is False
        assert service.snapshot().participant_count == 1
        assert set(service._tracker._members) == {_USER_A}
    finally:
        await service.close()


async def test_unrelated_log_traffic_does_not_block_ready(tmp_path: Path) -> None:
    create_time = time.time() - 5
    log_path = tmp_path / _log_name(create_time)
    log_path.write_text(
        "\n".join(_roster_block(("Self",), (_USER_A,), "Self")) + "\n",
        encoding="utf-8",
    )
    snapshots = FakeSnapshots()
    snapshots.items = (_snapshot(4242, "C:\\VRChat\\VRChat.exe", create_time),)
    service = _service(snapshots, FakeWatcherFactory(snapshots), tmp_path)
    await service.start()
    try:
        noise = "\n".join(
            f"2026.09.07 01:49:{second:02d} Debug      -  [UdonBehaviour] ambient tick {index}"
            for index, second in enumerate(range(30, 60))
        )
        for _ in range(100):
            if service.snapshot().status == "ready":
                break
            with open(log_path, "a", encoding="utf-8") as stream:
                stream.write(noise + "\n")
            await asyncio.sleep(0.03)
        assert service.snapshot().status == "ready"
        assert service.snapshot().participant_count == 1
    finally:
        await service.close()


async def test_lifecycle_end_clears_buffered_raw_payload(tmp_path: Path) -> None:
    create_time = time.time() - 5
    log_path = tmp_path / _log_name(create_time)
    log_path.write_text(
        "\n".join(_roster_block(("Self",), (_USER_A,), "Self")) + "\n",
        encoding="utf-8",
    )
    snapshots = FakeSnapshots()
    snapshots.items = (_snapshot(4242, "C:\\VRChat\\VRChat.exe", create_time),)
    watcher = FakeWatcherFactory(snapshots)
    service = _service(snapshots, watcher, tmp_path)
    await service.start()
    try:
        await _wait_for_status(service, "ready")

        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write("2026.09.07 01:49:00 Debug      -  [Behaviour] OnPlayerJoined Hal")
            stream.flush()
        await asyncio.sleep(0.15)
        assert service._tailer._pending_text != ""

        watcher.latest_terminal()()
        await _wait_for_status(service, "unavailable")
        assert service._tailer._pending_text == ""
        assert service._tailer.active_path is None
    finally:
        await service.close()
        assert service._tailer._pending_text == ""
        assert service._tailer.active_path is None


class CountingSnapshots(FakeSnapshots):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def snapshots(self) -> tuple[ProcessSnapshot, ...]:
        self.calls += 1
        return self.items


class PinnedWatchHandle:
    def __init__(self, *, identity_verified: bool = True, alive: bool = True) -> None:
        self.identity_verified = identity_verified
        self._alive = alive
        self.closed = False

    @property
    def watch_thread_alive(self) -> bool:
        return self._alive

    def kill_thread(self) -> None:
        self._alive = False

    def close(self) -> None:
        self.closed = True


class PinnedWatcherFactory:
    def __init__(
        self,
        *,
        identity_verified: bool = True,
        alive: bool = True,
        raise_on_watch: bool = False,
    ) -> None:
        self.identity_verified = identity_verified
        self.alive = alive
        self.raise_on_watch = raise_on_watch
        self.terminals: list[Callable[[], None]] = []
        self.handles: list[PinnedWatchHandle] = []

    def watch(self, identity: object, on_terminal: Callable[[], None]) -> PinnedWatchHandle:
        if self.raise_on_watch:
            raise RuntimeError("watch registration failed")
        handle = PinnedWatchHandle(identity_verified=self.identity_verified, alive=self.alive)
        self.handles.append(handle)
        self.terminals.append(on_terminal)
        return handle


def _pinned_service(
    snapshots: CountingSnapshots,
    watcher: PinnedWatcherFactory,
    log_directory: Path,
) -> VrchatSceneService:
    return VrchatSceneService(
        snapshots=snapshots,
        watcher_factory=watcher,  # type: ignore[arg-type]
        log_directory=log_directory,
        poll_interval_s=0.02,
        quiescence_s=0.05,
    )


async def _ready_pinned(
    tmp_path: Path,
    names: tuple[str, ...],
    ids: tuple[str, ...],
    watcher: PinnedWatcherFactory,
) -> tuple[CountingSnapshots, VrchatSceneService, float]:
    create_time = time.time() - 5
    (tmp_path / _log_name(create_time)).write_text(
        "\n".join(_roster_block(names, ids, names[0])) + "\n",
        encoding="utf-8",
    )
    snapshots = CountingSnapshots()
    snapshots.items = (_snapshot(4242, "C:\\VRChat\\VRChat.exe", create_time),)
    service = _pinned_service(snapshots, watcher, tmp_path)
    await service.start()
    await _wait_for_status(service, "ready")
    return snapshots, service, create_time


async def test_pinned_healthy_watch_skips_enumeration_while_logs_update(
    tmp_path: Path,
) -> None:
    snapshots, service, _ = await _ready_pinned(
        tmp_path, ("Self",), (_USER_A,), PinnedWatcherFactory()
    )
    try:
        assert service.snapshot().participant_count == 1
        pinned_instance = service._instance_id
        snapshots.calls = 0
        log_path = next(tmp_path.iterdir())
        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write(_join("Peer", _USER_B) + "\n")
            stream.write(_remote_init("Peer") + "\n")
        for _ in range(10):
            await service._poll_once(service._generation)
        assert snapshots.calls == 0
        assert service._instance_id == pinned_instance
        await asyncio.sleep(0.15)
        for _ in range(200):
            if service.snapshot().participant_count == 2:
                break
            await service._poll_once(service._generation)
        assert service.snapshot().participant_count == 2
    finally:
        await service.close()


async def test_pinned_service_ignores_newer_second_process(tmp_path: Path) -> None:
    snapshots, service, create_time = await _ready_pinned(
        tmp_path, ("Self",), (_USER_A,), PinnedWatcherFactory()
    )
    try:
        newer_start = time.time() - 1
        snapshots.items = (
            _snapshot(4242, "C:\\VRChat\\VRChat.exe", create_time),
            _snapshot(6100, "C:\\VRChat\\VRChat.exe", newer_start),
        )
        snapshots.calls = 0
        for _ in range(10):
            await service._poll_once(service._generation)
        await asyncio.sleep(0.15)
        assert snapshots.calls == 0
        assert service._instance_id == f"4242:{create_time}"
        assert service.snapshot().participant_count == 1
    finally:
        await service.close()


async def test_terminal_resumes_discovery_and_reused_pid_restart_rebuilds(
    tmp_path: Path,
) -> None:
    watcher = PinnedWatcherFactory()
    snapshots, service, _ = await _ready_pinned(
        tmp_path, ("Self", "Peer"), (_USER_A, _USER_B), watcher
    )
    try:
        assert service.snapshot().participant_count == 2
        snapshots.calls = 0
        for _ in range(5):
            await service._poll_once(service._generation)
        snapshots.items = ()
        watcher.terminals[-1]()
        await _wait_for_status(service, "unavailable")
        snapshots.calls = 0
        new_start = time.time()
        (tmp_path / _log_name(new_start)).write_text(
            "\n".join(_roster_block(("Self",), (_USER_A,), "Self")) + "\n",
            encoding="utf-8",
        )
        snapshots.items = (_snapshot(4242, "C:\\VRChat\\VRChat.exe", new_start),)
        await _wait_for_status(service, "ready")
        assert snapshots.calls > 0
        assert service._instance_id == f"4242:{new_start}"
        assert service.snapshot().participant_count == 1
    finally:
        await service.close()


async def test_invalid_watch_falls_back_to_enumeration(tmp_path: Path) -> None:
    snapshots, service, _ = await _ready_pinned(
        tmp_path, ("Self",), (_USER_A,), PinnedWatcherFactory(identity_verified=False)
    )
    try:
        snapshots.calls = 0
        for _ in range(5):
            await service._poll_once(service._generation)
        assert snapshots.calls >= 5
        assert service.snapshot().status == "ready"
        snapshots.items = ()
        await _wait_for_status(service, "unavailable")
        assert service.snapshot().participant_count is None
    finally:
        await service.close()


async def test_watch_registration_exception_falls_back_to_enumeration(
    tmp_path: Path,
) -> None:
    snapshots, service, _ = await _ready_pinned(
        tmp_path, ("Self",), (_USER_A,), PinnedWatcherFactory(raise_on_watch=True)
    )
    try:
        assert service._watch is None
        snapshots.calls = 0
        for _ in range(5):
            await service._poll_once(service._generation)
        assert snapshots.calls >= 5
        assert service.snapshot().status == "ready"
        snapshots.items = ()
        await _wait_for_status(service, "unavailable")
        assert service.snapshot().participant_count is None
    finally:
        await service.close()


async def test_dead_watch_thread_resumes_discovery(tmp_path: Path) -> None:
    watcher = PinnedWatcherFactory()
    snapshots, service, _ = await _ready_pinned(tmp_path, ("Self",), (_USER_A,), watcher)
    try:
        snapshots.calls = 0
        for _ in range(5):
            await service._poll_once(service._generation)
        assert snapshots.calls == 0
        watcher.handles[0].kill_thread()
        for _ in range(5):
            await service._poll_once(service._generation)
        assert snapshots.calls >= 5
        snapshots.items = ()
        await _wait_for_status(service, "unavailable")
        assert service.snapshot().participant_count is None
    finally:
        await service.close()


async def test_first_replay_ingests_roster_before_next_poll(tmp_path: Path) -> None:
    create_time = time.time() - 5
    (tmp_path / _log_name(create_time)).write_text(
        "\n".join(_roster_block(("Self", "Peer"), (_USER_A, _USER_B), "Self")) + "\n",
        encoding="utf-8",
    )
    snapshots = CountingSnapshots()
    snapshots.items = (_snapshot(4242, "C:\\VRChat\\VRChat.exe", create_time),)
    service = _pinned_service(snapshots, PinnedWatcherFactory(), tmp_path)
    service._started = True
    try:
        await service._poll_once(service._generation)
        assert snapshots.calls == 1
        assert service._log_candidate is not None
        assert service.snapshot().status == "syncing"
        assert set(service._tracker._members) == {_USER_A, _USER_B}
    finally:
        await service.close()


_SCENE_LOGGER = "puripuly_heart.core.vrchat_scene_service"


async def test_lifecycle_logs_selection_and_ready_without_identities(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    create_time = time.time() - 5
    file_name = _log_name(create_time)
    (tmp_path / file_name).write_text(
        "\n".join(_roster_block(("Self", "Peer"), (_USER_A, _USER_B), "Self")) + "\n",
        encoding="utf-8",
    )
    snapshots = FakeSnapshots()
    snapshots.items = (_snapshot(4242, "C:\\VRChat\\VRChat.exe", create_time),)
    service = _service(snapshots, FakeWatcherFactory(snapshots), tmp_path)
    with caplog.at_level(logging.INFO, logger=_SCENE_LOGGER):
        await service.start()
        try:
            await _wait_for_status(service, "ready")
        finally:
            await service.close()

    assert "[VrchatScene] lifecycle begin pid=4242" in caplog.messages
    assert f"[VrchatScene] log selected name={file_name} replayed=6 parsed=5" in caplog.messages
    assert "[VrchatScene] status syncing -> ready people=2" in caplog.messages
    assert _USER_A not in caplog.text
    assert _USER_B not in caplog.text
    assert "VRChat.exe" not in caplog.text


async def test_ready_transition_logs_once_across_polls(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    create_time = time.time() - 5
    (tmp_path / _log_name(create_time)).write_text(
        "\n".join(_roster_block(("Self", "Peer"), (_USER_A, _USER_B), "Self")) + "\n",
        encoding="utf-8",
    )
    snapshots = FakeSnapshots()
    snapshots.items = (_snapshot(4242, "C:\\VRChat\\VRChat.exe", create_time),)
    service = _service(snapshots, FakeWatcherFactory(snapshots), tmp_path)
    with caplog.at_level(logging.INFO, logger=_SCENE_LOGGER):
        await service.start()
        try:
            await _wait_for_status(service, "ready")
            for _ in range(5):
                await service._poll_once(service._generation)
        finally:
            await service.close()

    ready_logs = [message for message in caplog.messages if "-> ready people=" in message]
    assert ready_logs == ["[VrchatScene] status syncing -> ready people=2"]


async def test_deleted_log_warns_with_safe_reason_without_file_name(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    create_time = time.time() - 5
    file_name = _log_name(create_time)
    log_path = tmp_path / file_name
    log_path.write_text(
        "\n".join(_roster_block(("Self", "Peer"), (_USER_A, _USER_B), "Self")) + "\n",
        encoding="utf-8",
    )
    snapshots = FakeSnapshots()
    snapshots.items = (_snapshot(4242, "C:\\VRChat\\VRChat.exe", create_time),)
    service = _service(snapshots, FakeWatcherFactory(snapshots), tmp_path)
    with caplog.at_level(logging.INFO, logger=_SCENE_LOGGER):
        await service.start()
        try:
            await _wait_for_status(service, "ready")
            caplog.clear()
            log_path.unlink()
            await _wait_for_status(service, "degraded")
        finally:
            await service.close()

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].getMessage() == (
        "[VRChat Scene] Scene context became unavailable · Cause FileNotFoundError"
    )
    assert file_name not in warnings[0].getMessage()
    assert _USER_A not in caplog.text
    assert _USER_B not in caplog.text
