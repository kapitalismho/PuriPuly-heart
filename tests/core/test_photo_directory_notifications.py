from __future__ import annotations

import asyncio
import os
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from puripuly_heart.core.photo.capture import PhotoCaptureWatcher
from puripuly_heart.core.photo.directory_notifications import (
    DirectoryChanges,
    WindowsDirectoryNotifications,
    _DirectoryWatch,
)

windows = pytest.mark.skipif(os.name != "nt", reason="Requires Windows directory notifications")


async def test_unsupported_platform_fails_without_polling(monkeypatch, tmp_path):
    import puripuly_heart.core.photo.directory_notifications as module

    monkeypatch.setattr(module, "os", SimpleNamespace(name="unsupported"))
    source = WindowsDirectoryNotifications()
    with pytest.raises(RuntimeError, match="require Windows"):
        await source.start((tmp_path,))
    await source.close()


async def test_notification_buffer_coalesces_paths_and_bounds_scheduled_wakeups(tmp_path):
    source = WindowsDirectoryNotifications()
    callbacks = []
    source._active = True
    source._loop = SimpleNamespace(call_soon_threadsafe=callbacks.append)
    path = tmp_path / "new.png"
    for _ in range(5000):
        source._publish(DirectoryChanges(frozenset({path}), frozenset({path})))
    assert len(callbacks) == 1
    changes = await source.wait_changes()
    assert changes.paths == frozenset({path})
    assert changes.created == frozenset({path})
    assert not changes.rescan
    await source.close()


async def test_notification_overflow_exposes_resynchronization_and_discards_growth(tmp_path):
    source = WindowsDirectoryNotifications()
    source._active = True
    paths = frozenset(tmp_path / f"{index}.png" for index in range(5000))
    source._publish(DirectoryChanges(paths, paths))
    source._publish(DirectoryChanges(paths, paths))
    assert not source._paths
    assert not source._created
    changes = await source.wait_changes()
    assert changes.rescan
    assert not changes.paths
    await source.close()


async def test_buffer_failure_is_propagated_without_repeated_recovery(tmp_path):
    source = WindowsDirectoryNotifications()
    source._active = True
    error = OSError("notification read failed")
    source._publish(failure=error)
    with pytest.raises(OSError, match="notification read failed") as raised:
        await source.wait_changes()
    assert raised.value is error
    await source.close()


def test_native_record_parser_filters_unrelated_ancestor_contents(tmp_path):
    watch = object.__new__(_DirectoryWatch)
    watch.root = tmp_path / "photos"

    def record(name):
        encoded = name.encode("utf-16-le")
        return struct.pack("<III", 0, 1, len(encoded)) + encoded

    assert not watch._changes(tmp_path, record("unrelated.png")).paths
    changes = watch._changes(tmp_path, record("photos"))
    assert changes.paths == frozenset({watch.root})
    assert changes.created == changes.paths
    with pytest.raises(RuntimeError, match="Truncated"):
        watch._changes(tmp_path, b"short")


@windows
async def test_native_new_month_and_first_png_arrive_without_tree_polling(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    old = photos / "old.png"
    old.write_bytes(b"old")
    watcher = PhotoCaptureWatcher(photos, tmp_path / "logs")
    await watcher.start()
    try:
        old.write_bytes(b"edited startup photo")
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(watcher.wait_capture(), 0.05)
        month = photos / "2026-11"
        month.mkdir()
        new = month / "first.png"
        new.write_bytes(b"")
        event = await asyncio.wait_for(watcher.wait_capture(), 3)
        assert event.path == new.resolve()
        assert event.file_id == (new.stat().st_dev, new.stat().st_ino)
        new.write_bytes(b"completed")
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(watcher.wait_capture(), 0.05)
    finally:
        await watcher.close()


@windows
async def test_native_missing_roots_custom_log_path_and_root_replacement(tmp_path):
    photos = tmp_path / "missing" / "photos"
    logs = tmp_path / "missing-logs" / "logs"
    watcher = PhotoCaptureWatcher(photos, logs)
    await watcher.start()
    try:
        photos.mkdir(parents=True)
        first = photos / "first.png"
        first.write_bytes(b"first")
        assert (await asyncio.wait_for(watcher.wait_capture(), 3)).path == first.resolve()
        photos.rename(photos.with_name("previous"))
        photos.mkdir()
        replaced = photos / "replacement.png"
        replaced.write_bytes(b"replacement")
        assert (await asyncio.wait_for(watcher.wait_capture(), 3)).path == replaced.resolve()
        logs.mkdir(parents=True)
        custom = tmp_path / "custom" / "late.png"
        log = logs / "output_log_current.txt"
        log.write_text(f"[VRC Camera] Took screenshot to: {custom}\n", encoding="utf-8")
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(watcher.wait_capture(), 0.05)
        custom.parent.mkdir()
        custom.write_bytes(b"late custom file")
        assert (await asyncio.wait_for(watcher.wait_capture(), 3)).path == custom.resolve()
    finally:
        await watcher.close()


@windows
async def test_native_watch_is_registered_before_baseline(tmp_path, monkeypatch):
    photos = tmp_path / "photos"
    photos.mkdir()
    watcher = PhotoCaptureWatcher(photos, tmp_path / "logs")
    scan = watcher._files
    new = photos / "during-baseline.png"
    created = False

    def baseline(root):
        nonlocal created
        if not created:
            created = True
            new.write_bytes(b"during baseline")
        return scan(root)

    monkeypatch.setattr(watcher, "_files", baseline)
    try:
        await watcher.start()
        assert (await asyncio.wait_for(watcher.wait_capture(), 3)).path == new.resolve()
    finally:
        await watcher.close()


@windows
async def test_native_close_settles_threads_and_restart_does_not_replay(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    source = WindowsDirectoryNotifications()
    watcher = PhotoCaptureWatcher(photos, tmp_path / "logs", notification_source=source)
    await watcher.start()
    watches = tuple(source._watches)
    await watcher.close()
    assert all(not watch._thread.is_alive() for watch in watches)
    assert all(watch._stop_handle is None for watch in watches)
    old = photos / "disabled.png"
    old.write_bytes(b"disabled")
    await watcher.start()
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(watcher.wait_capture(), 0.05)
        new = photos / "enabled.png"
        new.write_bytes(b"enabled")
        assert (await asyncio.wait_for(watcher.wait_capture(), 3)).path == new.resolve()
    finally:
        await watcher.close()


@windows
async def test_native_missing_root_filters_unrelated_ancestor_notifications(tmp_path):
    source = WindowsDirectoryNotifications()
    root = tmp_path / "not-created" / "photos"
    await source.start((root,))
    try:
        unrelated = tmp_path / "unrelated"
        unrelated.mkdir()
        (unrelated / "other.png").write_bytes(b"unrelated")
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(source.wait_changes(), 0.05)
        root.mkdir(parents=True)
        (root / "first.png").write_bytes(b"first")
        changes = await asyncio.wait_for(source.wait_changes(), 3)
        assert any(path == root or path in root.parents or root in path.parents for path in changes.paths)
    finally:
        await source.close()


def test_missing_root_uses_nearest_existing_ancestor_without_enumeration(tmp_path, monkeypatch):
    watch = object.__new__(_DirectoryWatch)
    watch.root = tmp_path / "missing" / "nested" / "photos"
    opened = []
    watch._kernel = SimpleNamespace(
        CreateFileW=lambda path, *args: opened.append(path) or 123
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("Enumerating ancestor contents")

    monkeypatch.setattr(os, "scandir", unexpected)
    anchor, identity, handle = watch._open()
    assert anchor == tmp_path
    assert opened == [str(tmp_path)]
    assert identity == (tmp_path.stat().st_dev, tmp_path.stat().st_ino)
    assert handle == 123


def test_ancestor_permission_failure_is_not_silently_skipped(tmp_path, monkeypatch):
    watch = object.__new__(_DirectoryWatch)
    watch.root = tmp_path / "private" / "photos"
    original = Path.stat

    def status(path, *args, **kwargs):
        if path == watch.root.parent:
            raise PermissionError("directory denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", status)
    with pytest.raises(PermissionError, match="directory denied"):
        watch._open()
