from __future__ import annotations

import asyncio

from puripuly_heart.core.photo.capture import PhotoCaptureWatcher


async def test_existing_photos_are_not_startup_offers_and_incomplete_file_appears_once(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    old = photos / "old.png"
    old.write_bytes(b"old")
    watcher = PhotoCaptureWatcher(photos, tmp_path / "logs")
    await watcher.start()
    assert await watcher.poll() is None
    month = photos / "2026-10"
    month.mkdir()
    new = month / "new.png"
    new.write_bytes(b"")
    event = await watcher.poll()
    assert event.path == new.resolve()
    assert event.file_size == 0
    assert event.file_id == (new.stat().st_dev, new.stat().st_ino)
    new.write_bytes(b"still writing")
    assert await watcher.poll() is None


async def test_only_newest_first_appearance_is_selected_and_replacement_is_new_identity(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    watcher = PhotoCaptureWatcher(photos, tmp_path / "logs")
    await watcher.start()
    first = photos / "a.png"
    second = photos / "b.png"
    first.write_bytes(b"one")
    await asyncio.sleep(0.01)
    second.write_bytes(b"two")
    assert (await watcher.poll()).path == second.resolve()
    assert await watcher.poll() is None
    replacement = photos / "replacement.tmp"
    replacement.write_bytes(b"new identity")
    replacement.replace(second)
    event = await watcher.poll()
    assert event.path == second.resolve()
    assert event.file_size == len(b"new identity")


async def test_camera_log_is_secondary_and_does_not_replay_startup_or_duplicate_files(tmp_path):
    photos = tmp_path / "photos"
    logs = tmp_path / "logs"
    photos.mkdir()
    logs.mkdir()
    old = tmp_path / "custom-old.png"
    old.write_bytes(b"old")
    log = logs / "output_log_2026-10-31_00-00-00.txt"
    log.write_text(f"[VRC Camera] Took screenshot to: {old}\n", encoding="utf-8")
    watcher = PhotoCaptureWatcher(photos, logs)
    await watcher.start()
    assert await watcher.poll() is None
    new = tmp_path / "custom-new.png"
    new.write_bytes(b"partial")
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"[VRC Camera] Took screenshot to: {new}")
    assert await watcher.poll() is None
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    assert (await watcher.poll()).path == new.resolve()
    assert await watcher.poll() is None
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"[VRC Camera] Took screenshot to: {new}\n")
    assert await watcher.poll() is None


async def test_missing_roots_are_safe_and_new_root_is_observed(tmp_path):
    photos = tmp_path / "not-created"
    watcher = PhotoCaptureWatcher(photos, tmp_path / "missing-logs")
    await watcher.start()
    assert await watcher.poll() is None
    photos.mkdir()
    path = photos / "new.png"
    path.write_bytes(b"new")
    assert (await watcher.poll()).path == path.resolve()
