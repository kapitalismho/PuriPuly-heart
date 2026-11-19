from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from puripuly_heart.core.photo.capture import PhotoCaptureWatcher
from puripuly_heart.core.photo.directory_notifications import DirectoryChanges


class Notifications:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.waiting = asyncio.Event()
        self.started = False
        self.starts = 0
        self.closes = 0
        self.loop = None

    async def start(self, roots):
        self.loop = asyncio.get_running_loop()
        self.roots = roots
        self.started = True
        self.starts += 1

    def emit(self, *paths, created=(), rescan=False):
        changes = DirectoryChanges(frozenset(paths), frozenset(created), rescan)
        self.loop.call_soon_threadsafe(self.queue.put_nowait, changes)

    def fail(self, error):
        self.queue.put_nowait(error)

    async def wait_changes(self):
        self.waiting.set()
        changes = await self.queue.get()
        if isinstance(changes, Exception):
            raise changes
        return changes

    async def close(self):
        self.started = False
        self.closes += 1
        while not self.queue.empty():
            self.queue.get_nowait()


@asynccontextmanager
async def watching(photos, logs):
    source = Notifications()
    watcher = PhotoCaptureWatcher(photos, logs, notification_source=source)
    await watcher.start()
    try:
        yield watcher, source
    finally:
        await watcher.close()


async def capture(watcher):
    return await asyncio.wait_for(watcher.wait_capture(), 2)


async def quiet(watcher):
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(watcher.wait_capture(), 0.03)


async def test_existing_photos_are_not_startup_offers_and_incomplete_file_appears_once(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    old = photos / "old.png"
    old.write_bytes(b"old")
    async with watching(photos, tmp_path / "logs") as (watcher, source):
        await quiet(watcher)
        old.write_bytes(b"edited existing photo")
        source.emit(old)
        await quiet(watcher)
        month = photos / "2026-10"
        month.mkdir()
        new = month / "new.png"
        new.write_bytes(b"")
        source.emit(month, created=(month,))
        event = await capture(watcher)
        assert event.path == new.resolve()
        assert event.file_size == 0
        assert event.file_id == (new.stat().st_dev, new.stat().st_ino)
        new.write_bytes(b"still writing")
        source.emit(new)
        await quiet(watcher)


async def test_only_newest_first_appearance_is_selected_and_replacement_is_new_identity(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    async with watching(photos, tmp_path / "logs") as (watcher, source):
        first = photos / "a.png"
        second = photos / "b.png"
        first.write_bytes(b"one")
        second.write_bytes(b"two")
        source.emit(first, second, created=(first, second))
        assert (await capture(watcher)).path == second.resolve()
        await quiet(watcher)
        replacement = photos / "replacement.tmp"
        replacement.write_bytes(b"new identity")
        replacement.replace(second)
        source.emit(second, created=(second,))
        event = await capture(watcher)
        assert event.path == second.resolve()
        assert event.file_size == len(b"new identity")
        source.emit(first, second)
        await quiet(watcher)


async def test_camera_log_is_secondary_and_does_not_replay_startup_or_duplicate_files(tmp_path):
    photos = tmp_path / "photos"
    logs = tmp_path / "logs"
    photos.mkdir()
    logs.mkdir()
    old = tmp_path / "custom-old.png"
    old.write_bytes(b"old")
    log = logs / "output_log_2026-10-31_00-00-00.txt"
    log.write_text(f"[VRC Camera] Took screenshot to: {old}\n", encoding="utf-8")
    async with watching(photos, logs) as (watcher, source):
        await quiet(watcher)
        new = tmp_path / "custom-new.png"
        new.write_bytes(b"partial")
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f'[VRC Camera] Took screenshot to: "{new}"')
        source.emit(log)
        await quiet(watcher)
        with log.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        source.emit(log)
        assert (await capture(watcher)).path == new.resolve()
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"[VRC Camera] Took screenshot to: {new}\n")
        source.emit(log)
        await quiet(watcher)


async def test_missing_roots_are_safe_and_new_root_is_observed(tmp_path):
    photos = tmp_path / "not-created" / "nested"
    logs = tmp_path / "missing-logs"
    async with watching(photos, logs) as (watcher, source):
        await quiet(watcher)
        photos.mkdir(parents=True)
        path = photos / "new.png"
        path.write_bytes(b"new")
        source.emit(photos.parent, created=(photos.parent,))
        assert (await capture(watcher)).path == path.resolve()
        logs.mkdir()
        custom = tmp_path / "custom.png"
        custom.write_bytes(b"custom")
        log = logs / "output_log_new.txt"
        log.write_text(f"[VRC Camera] Took screenshot to: {custom}\n", encoding="utf-8")
        source.emit(logs, created=(logs,))
        assert (await capture(watcher)).path == custom.resolve()


async def test_creation_during_baseline_is_not_suppressed(tmp_path, monkeypatch):
    photos = tmp_path / "photos"
    photos.mkdir()
    old = photos / "old.png"
    old.write_bytes(b"old")
    source = Notifications()
    watcher = PhotoCaptureWatcher(photos, tmp_path / "logs", notification_source=source)
    original = watcher._files
    new = photos / "during-baseline.png"

    def baseline(root):
        assert source.started
        new.write_bytes(b"new")
        source.emit(new, created=(new,))
        return original(root)

    monkeypatch.setattr(watcher, "_files", baseline)
    try:
        await watcher.start()
        assert (await capture(watcher)).path == new.resolve()
        await quiet(watcher)
    finally:
        await watcher.close()


async def test_idle_watcher_does_not_enumerate_or_stat_files(tmp_path, monkeypatch):
    photos = tmp_path / "photos"
    photos.mkdir()
    for index in range(1200):
        (photos / f"{index}.png").write_bytes(b"old")
    async with watching(photos, tmp_path / "logs") as (watcher, source):
        await source.waiting.wait()

        def unexpected(*args, **kwargs):
            raise AssertionError("Filesystem access while idle")

        with monkeypatch.context() as patch:
            patch.setattr(Path, "stat", unexpected)
            patch.setattr(Path, "glob", unexpected)
            patch.setattr(os, "walk", unexpected)
            await quiet(watcher)
        new = photos / "new.png"
        new.write_bytes(b"new")
        source.emit(new, created=(new,))
        assert (await capture(watcher)).path == new.resolve()


async def test_file_notifications_do_not_scan_photo_or_log_trees(tmp_path, monkeypatch):
    photos = tmp_path / "photos"
    photos.mkdir()
    async with watching(photos, tmp_path / "logs") as (watcher, source):

        def unexpected(*args):
            raise AssertionError("Whole-tree scan for a file notification")

        monkeypatch.setattr(watcher, "_files", unexpected)
        monkeypatch.setattr(watcher, "_scan_logs", unexpected)
        new = photos / "new.png"
        new.write_bytes(b"new")
        source.emit(new, created=(new,))
        assert (await capture(watcher)).path == new.resolve()


async def test_log_supplied_missing_custom_path_is_retried_without_another_event(tmp_path):
    photos = tmp_path / "photos"
    logs = tmp_path / "logs"
    photos.mkdir()
    logs.mkdir()
    log = logs / "output_log_current.txt"
    log.write_text("startup\n", encoding="utf-8")
    async with watching(photos, logs) as (watcher, source):
        new = tmp_path / "custom" / "late.png"
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"[VRC Camera] Took screenshot to: {new}\n")
        source.emit(log)
        await quiet(watcher)
        new.parent.mkdir()
        new.write_bytes(b"late")
        assert (await capture(watcher)).path == new.resolve()
        await quiet(watcher)


async def test_log_chunks_are_drained_without_another_notification(tmp_path):
    photos = tmp_path / "photos"
    logs = tmp_path / "logs"
    photos.mkdir()
    logs.mkdir()
    log = logs / "output_log_current.txt"
    log.write_text("startup\n", encoding="utf-8")
    async with watching(photos, logs) as (watcher, source):
        new = tmp_path / "custom.png"
        new.write_bytes(b"new")
        with log.open("a", encoding="utf-8") as stream:
            stream.write("unrelated\n" * 16000)
            stream.write(f"[VRC Camera] Took screenshot to: {new}\n")
        source.emit(log)
        assert (await capture(watcher)).path == new.resolve()


@pytest.mark.parametrize("startup_log", ["current", "older"])
async def test_log_rotation_truncation_and_return_to_older_log_do_not_replay(tmp_path, startup_log):
    photos = tmp_path / "photos"
    logs = tmp_path / "logs"
    photos.mkdir()
    logs.mkdir()
    old = tmp_path / "old.png"
    old.write_bytes(b"old")
    older = logs / "output_log_older.txt"
    current = logs / "output_log_current.txt"
    older.write_text(f"[VRC Camera] Took screenshot to: {old}\n", encoding="utf-8")
    current.write_text("startup" * 200, encoding="utf-8")
    clock = current.stat().st_mtime_ns

    def touch(path, offset_seconds):
        stamp = clock + offset_seconds * 1_000_000_000
        os.utime(path, ns=(stamp, stamp))

    touch(older, 1 if startup_log == "older" else -1)
    async with watching(photos, logs) as (watcher, source):
        new = tmp_path / "new.png"
        new.write_bytes(b"new")
        current.write_text(f"[VRC Camera] Took screenshot to: {new}\n", encoding="utf-8")
        touch(current, 2)
        source.emit(current)
        assert (await capture(watcher)).path == new.resolve()
        rotated = tmp_path / "rotated.png"
        rotated.write_bytes(b"rotated")
        with older.open("a", encoding="utf-8") as log:
            log.write(f"[VRC Camera] Took screenshot to: {rotated}\n")
        touch(older, 3)
        source.emit(older)
        assert (await capture(watcher)).path == rotated.resolve()

        resumed = tmp_path / "resumed.png"
        resumed.write_bytes(b"resumed")
        with current.open("a", encoding="utf-8") as log:
            log.write(f"[VRC Camera] Took screenshot to: {resumed}\n")
            log.write("unrelated\n" * 400)
        touch(current, 4)
        source.emit(current)
        assert (await capture(watcher)).path == resumed.resolve()

        current.unlink()
        source.emit(current)
        await quiet(watcher)
        replacement = tmp_path / "replacement.png"
        replacement.write_bytes(b"replacement")
        current.write_text(f"[VRC Camera] Took screenshot to: {replacement}\n", encoding="utf-8")
        touch(current, 5)
        source.emit(current, created=(current,))
        assert (await capture(watcher)).path == replacement.resolve()


@pytest.mark.parametrize("transition", ["truncated", "replaced"])
async def test_rotation_does_not_offer_startup_or_replacement_log_content(tmp_path, transition):
    photos = tmp_path / "photos"
    logs = tmp_path / "logs"
    photos.mkdir()
    logs.mkdir()
    retiring = logs / "output_log_a.txt"
    successor = logs / "output_log_b.txt"
    startup = tmp_path / "startup.png"
    startup.write_bytes(b"startup")
    startup_line = f"[VRC Camera] Took screenshot to: {startup}\n"
    retiring.write_text(startup_line + "unrelated\n" * 100, encoding="utf-8")

    async with watching(photos, logs) as (watcher, source):
        if transition == "truncated":
            retiring.write_text(startup_line, encoding="utf-8")
        else:
            other = tmp_path / "other.png"
            other.write_bytes(b"belongs to a different log")
            replacement = tmp_path / "replacement.txt"
            replacement.write_text(f"[VRC Camera] Took screenshot to: {other}\n", encoding="utf-8")
            replacement.replace(retiring)
        successor.write_text("session started\n", encoding="utf-8")
        stamp = retiring.stat().st_mtime_ns + 1_000_000_000
        os.utime(successor, ns=(stamp, stamp))
        source.emit(retiring, successor, created=(successor,))
        await quiet(watcher)

        fresh = tmp_path / "fresh.png"
        fresh.write_bytes(b"fresh")
        with successor.open("a", encoding="utf-8") as log:
            log.write(f"[VRC Camera] Took screenshot to: {fresh}\n")
        source.emit(successor)
        assert (await capture(watcher)).path == fresh.resolve()


async def test_overflow_resynchronizes_without_replaying_seen_photos(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    (photos / "old.png").write_bytes(b"old")
    async with watching(photos, tmp_path / "logs") as (watcher, source):
        new = photos / "new.png"
        new.write_bytes(b"new")
        source.emit(rescan=True)
        assert (await capture(watcher)).path == new.resolve()
        source.emit(rescan=True)
        await quiet(watcher)


async def test_waiter_cancellation_does_not_stop_watching_and_restart_rebaselines(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    source = Notifications()
    watcher = PhotoCaptureWatcher(photos, tmp_path / "logs", notification_source=source)
    try:
        await watcher.start()
        await watcher.start()
        assert source.starts == 1
        await quiet(watcher)
        new = photos / "new.png"
        new.write_bytes(b"new")
        source.emit(new, created=(new,))
        assert (await capture(watcher)).path == new.resolve()
        await watcher.close()
        assert source.closes == 1
        new.write_bytes(b"written while disabled")
        await watcher.start()
        assert source.starts == 2
        await quiet(watcher)
        next_photo = photos / "next.png"
        next_photo.write_bytes(b"next")
        source.emit(next_photo, created=(next_photo,))
        assert (await capture(watcher)).path == next_photo.resolve()
    finally:
        await watcher.close()


async def test_notification_failure_reaches_waiter_and_close_settles_worker(tmp_path):
    source = Notifications()
    watcher = PhotoCaptureWatcher(
        tmp_path / "photos", tmp_path / "logs", notification_source=source
    )
    await watcher.start()
    failure = OSError("native notification failed")
    source.fail(failure)
    try:
        with pytest.raises(OSError, match="native notification failed") as error:
            await capture(watcher)
        assert error.value is failure
    finally:
        await watcher.close()
    assert not source.started
    assert watcher._task is None


async def test_log_candidate_expiry_stops_scoped_retries(tmp_path):
    photos = tmp_path / "photos"
    logs = tmp_path / "logs"
    photos.mkdir()
    logs.mkdir()
    async with watching(photos, logs) as (watcher, source):
        missing = tmp_path / "never.png"
        log = logs / "output_log_new.txt"
        log.write_text(f"[VRC Camera] Took screenshot to: {missing}\n", encoding="utf-8")
        source.emit(log, created=(log,))
        await quiet(watcher)
        assert missing in watcher._pending_paths
        watcher._pending_paths[missing] -= 11
        source.emit()
        await quiet(watcher)
        assert not watcher._pending_paths
        missing.write_bytes(b"too late")
        await quiet(watcher)


async def test_source_timeout_failure_is_not_hidden_by_missing_candidate_retry(tmp_path):
    photos = tmp_path / "photos"
    logs = tmp_path / "logs"
    photos.mkdir()
    logs.mkdir()
    async with watching(photos, logs) as (watcher, source):
        missing = tmp_path / "never.png"
        log = logs / "output_log_current.txt"
        log.write_text(f"[VRC Camera] Took screenshot to: {missing}\n", encoding="utf-8")
        source.emit(log, created=(log,))
        await quiet(watcher)
        source.fail(TimeoutError("native request timed out"))
        with pytest.raises(TimeoutError, match="native request timed out"):
            await capture(watcher)


async def test_photo_and_camera_log_notifications_share_identity_suppression(tmp_path):
    photos = tmp_path / "photos"
    logs = tmp_path / "logs"
    photos.mkdir()
    logs.mkdir()
    async with watching(photos, logs) as (watcher, source):
        new = photos / "new.png"
        new.write_bytes(b"new")
        source.emit(new, created=(new,))
        assert (await capture(watcher)).path == new.resolve()
        log = logs / "output_log_current.txt"
        log.write_text(f"[VRC Camera] Took screenshot to: {new}\n", encoding="utf-8")
        source.emit(log, created=(log,))
        await quiet(watcher)


async def test_close_wakes_blocked_capture_waiter(tmp_path):
    source = Notifications()
    watcher = PhotoCaptureWatcher(
        tmp_path / "photos", tmp_path / "logs", notification_source=source
    )
    await watcher.start()
    waiter = asyncio.create_task(watcher.wait_capture())
    await source.waiting.wait()
    await asyncio.sleep(0)
    await watcher.close()
    with pytest.raises(RuntimeError, match="closed"):
        await waiter
