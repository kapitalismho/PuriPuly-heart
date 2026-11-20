from __future__ import annotations

import asyncio
import os
import time
from unittest.mock import AsyncMock

import pytest

from puripuly_heart.app.services import photo_application
from puripuly_heart.app.services.photo_application import PhotoApplicationOwner
from puripuly_heart.core.photo.capture import PhotoCapture, PhotoCaptureWatcher
from puripuly_heart.core.photo.directory_notifications import WindowsDirectoryNotifications
from puripuly_heart.core.photo.image import PreparedPhoto
from puripuly_heart.core.photo.lifecycle import PhotoRoute
from puripuly_heart.core.photo.protocol import (
    PhotoRequestSent,
    PhotoStreamCompleted,
    PhotoTextDelta,
)
from tests.core.test_photo_capture import Notifications
from tests.core.test_photo_native_process import png


async def test_native_prepare_timings_keep_identity_and_ignore_invalid_values(
    tmp_path, monkeypatch
):
    async def no_route():
        return None

    observations = []
    owner = PhotoApplicationOwner(
        resolve_route=no_route,
        target_language=lambda: "ko",
        diagnostic=lambda stage, fields: observations.append((stage, fields)),
        watcher=PhotoCaptureWatcher(tmp_path / "photos", tmp_path / "logs"),
        temporary_root=tmp_path,
    )
    prepared = object()
    reader = AsyncMock(return_value=prepared)
    monkeypatch.setattr(photo_application, "load_prepared_photo", reader)
    owner.bridge.prepare = AsyncMock(
        return_value={
            "decode_ms": 3.25,
            "resize_ms": 7.5,
            "jpeg_ms": 1.125,
            "source_path": "must-not-enter-diagnostics",
            "source": "private text",
        }
    )
    photo_id = "fce34b2a-a0b3-48aa-952e-de4b05002061"
    try:
        assert (
            await owner._prepare(
                photo_id, 1, tmp_path / "source.png", tmp_path, expected_file_id=(1, 2)
            )
            is prepared
        )
        assert observations == [
            ("decode", {"photo_id": photo_id, "attempt_id": 1, "duration_ms": 3.25}),
            ("resize", {"photo_id": photo_id, "attempt_id": 1, "duration_ms": 7.5}),
            ("jpeg", {"photo_id": photo_id, "attempt_id": 1, "duration_ms": 1.125}),
        ]
        observations.clear()
        owner.bridge.prepare.return_value = {
            "decode_ms": float("nan"),
            "resize_ms": -1.0,
            "jpeg_ms": True,
        }
        await owner._prepare(
            photo_id, 1, tmp_path / "source.png", tmp_path, expected_file_id=(1, 2)
        )
        assert observations == []
        assert reader.await_count == 2
    finally:
        await owner.close()


def test_interaction_notes_reach_photo_diagnostics_without_changing_state(tmp_path):
    observations = []

    async def no_route():
        return None

    owner = PhotoApplicationOwner(
        resolve_route=no_route,
        target_language=lambda: "ko",
        diagnostic=lambda stage, fields: observations.append((stage, fields)),
        watcher=PhotoCaptureWatcher(tmp_path / "photos", tmp_path / "logs"),
        temporary_root=tmp_path,
    )
    before = owner.lifecycle.snapshot
    note = {
        "type": "photo_interaction",
        "interaction": "icon_near_miss",
        "photo_id": "p",
        "attempt_id": 0,
        "facing_deg": 50.0,
        "forward_deg": 41.5,
    }
    assert owner._control(note, 1) is None
    assert observations == [
        ("interaction", {key: value for key, value in note.items() if key != "type"})
    ]
    assert owner.lifecycle.snapshot == before


async def eventually(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


class Watcher:
    def __init__(self):
        self.active = False
        self.starts = 0
        self.closes = 0
        self.waits = 0
        self.queue = asyncio.Queue()
        self.start_entered = asyncio.Event()
        self.start_gate = None

    async def start(self):
        self.starts += 1
        self.active = True
        self.start_entered.set()
        if self.start_gate is not None:
            await self.start_gate.wait()

    async def wait_capture(self):
        self.waits += 1
        value = await self.queue.get()
        if isinstance(value, Exception):
            raise value
        return value

    async def close(self):
        self.closes += 1
        self.active = False
        while not self.queue.empty():
            self.queue.get_nowait()

    def emit(self, value):
        if self.active:
            self.queue.put_nowait(value)


class Connection:
    google = False
    model = "gpt-6-luna"
    connect_ms = 0

    async def preopen(self):
        pass

    async def stream(self, image, target_language):
        yield PhotoRequestSent(time.monotonic())
        yield PhotoTextDelta(
            '{"blocks":[{"bbox":[0,0,100,100],"source":"source","translation":"translated"}]}'
        )
        yield PhotoStreamCompleted({}, self.model)

    async def close(self):
        pass


def make_owner(tmp_path, watcher, *, enabled=lambda: True, discard=None):
    observations = []
    resolve = AsyncMock(return_value=PhotoRoute("luna", lambda model: Connection()))
    owner = PhotoApplicationOwner(
        resolve_route=resolve,
        target_language=lambda: "ko",
        diagnostic=lambda stage, fields: observations.append((stage, fields)),
        watcher=watcher,
        temporary_root=tmp_path,
        enabled=enabled,
        discard_translated=discard,
    )
    owner.supervisor.start = AsyncMock()
    owner.supervisor.close = AsyncMock()

    async def prepare(photo_id, attempt_id, source, directory, *, expected_file_id):
        snapshot = directory / "source.png"
        snapshot.write_bytes(b"snapshot")
        return PreparedPhoto("jpeg", snapshot, 100, 100, 1, 0, 0, 0)

    owner.lifecycle._prepare = prepare
    return owner, resolve, observations


@pytest.mark.parametrize(
    "delivery",
    [
        "coalesced",
        "rescan",
        "separate",
        "two-photos-retiring",
        "two-photos-successor",
        "late-retiring",
        "photo-folder",
        pytest.param(
            "native",
            marks=pytest.mark.skipif(os.name != "nt", reason="Requires Windows notifications"),
        ),
    ],
)
async def test_camera_log_rotation_offers_only_the_newest_fresh_photo(
    tmp_path, monkeypatch, delivery
):
    photos = tmp_path / "photos"
    logs = tmp_path / "logs"
    photos.mkdir()
    logs.mkdir()
    retiring = logs / "output_log_a.txt"
    successor = logs / "output_log_b.txt"
    startup = tmp_path / "startup.png"
    image = png(32, 32)

    def set_later_mtime(path, earlier):
        newer = earlier.stat().st_mtime_ns + 1_000_000_000
        os.utime(path, ns=(newer, newer))

    startup.write_bytes(image)
    retiring.write_text(f"[VRC Camera] Took screenshot to: {startup}\n", encoding="utf-8")
    if delivery == "native":
        source = WindowsDirectoryNotifications()
        drain = asyncio.Event()
        wait_changes = source.wait_changes

        async def wait_for_batch():
            await drain.wait()
            return await wait_changes()

        def batch_ready():
            with source._lock:
                return retiring in source._paths and successor in source._paths

        monkeypatch.setattr(source, "wait_changes", wait_for_batch)
    else:
        source = Notifications()
    watcher = PhotoCaptureWatcher(photos, logs, notification_source=source)
    owner, _, observations = make_owner(tmp_path, watcher)
    offers = []
    duplicate = asyncio.Event()

    def record_offer():
        offers.append(owner.lifecycle.snapshot["photo"])
        if len(offers) > 1:
            duplicate.set()

    owner.supervisor.start.side_effect = record_offer
    await owner.start()
    try:
        fresh = tmp_path / "fresh.png"
        competing = (photos if delivery == "photo-folder" else tmp_path) / "competing.png"
        if delivery == "two-photos-retiring":
            competing.write_bytes(image)
        if delivery == "late-retiring":
            staged = tmp_path / "staged.png"
            staged.write_bytes(image)
            await asyncio.sleep(0.02)
        else:
            fresh.write_bytes(image)
        if delivery == "two-photos-retiring":
            set_later_mtime(fresh, competing)
        with retiring.open("a", encoding="utf-8") as log:
            log.write(f"[VRC Camera] Took screenshot to: {fresh}\n")
        if delivery == "separate":
            source.emit(retiring)
            await eventually(lambda: offers)

        successor_contents = "session started\n"
        successor_wins = delivery in {"two-photos-successor", "late-retiring", "photo-folder"}
        if successor_wins:
            competing.write_bytes(image)
            set_later_mtime(competing, staged if delivery == "late-retiring" else fresh)
        if delivery.startswith("two-photos") or delivery == "late-retiring":
            successor_contents += f"[VRC Camera] Took screenshot to: {competing}\n"
        successor.write_text(successor_contents, encoding="utf-8")
        set_later_mtime(successor, retiring)
        if delivery == "native":
            await eventually(batch_ready)
            drain.set()
        elif delivery == "rescan":
            source.emit(rescan=True)
        elif delivery == "separate":
            source.emit(successor, created=(successor,))
        elif delivery == "photo-folder":
            source.emit(retiring, successor, competing, created=(successor, competing))
        else:
            source.emit(retiring, successor, created=(successor,))

        await eventually(lambda: offers)
        expected = competing if successor_wins else fresh
        assert offers[0]["path"] == str(expected.resolve())
        assert offers[0]["offer_open"]
        assert owner.lifecycle.request_starts == 0

        if delivery != "native":
            if delivery == "late-retiring":
                staged.rename(fresh)
            with retiring.open("a", encoding="utf-8") as log:
                log.write(f"[VRC Camera] Took screenshot to: {expected}\n")
            set_later_mtime(retiring, successor)
            source.waiting.clear()
            source.emit(retiring, successor)
            await eventually(source.waiting.is_set)
            source.waiting.clear()
            source.emit(rescan=True)
            await eventually(source.waiting.is_set)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(duplicate.wait(), 0.1)
        assert not any(stage == "capture_unavailable" for stage, _ in observations)
    finally:
        await owner.close()


async def test_delayed_retiring_photo_cannot_overwrite_newest_queued_offer(tmp_path):
    photos = tmp_path / "photos"
    logs = tmp_path / "logs"
    photos.mkdir()
    logs.mkdir()
    retiring = logs / "output_log_a.txt"
    successor = logs / "output_log_b.txt"
    retiring.write_text("session started\n", encoding="utf-8")
    source = Notifications()
    watcher = PhotoCaptureWatcher(photos, logs, notification_source=source)
    owner, resolve, _ = make_owner(tmp_path, watcher)
    resolving = asyncio.Event()
    resume = asyncio.Event()
    offers = []
    owner.supervisor.start.side_effect = lambda: offers.append(owner.lifecycle.snapshot["photo"])

    async def resolve_when_resumed():
        resolving.set()
        await resume.wait()
        return resolve.return_value

    await owner.start()
    try:
        resolve.side_effect = resolve_when_resumed
        image = png(32, 32)
        previous = photos / "previous.png"
        previous.write_bytes(image)
        source.emit(previous, created=(previous,))
        await eventually(resolving.is_set)

        staged = tmp_path / "staged.png"
        delayed = tmp_path / "delayed.png"
        newest = tmp_path / "newest.png"
        staged.write_bytes(image)
        with retiring.open("a", encoding="utf-8") as log:
            log.write(f"[VRC Camera] Took screenshot to: {delayed}\n")
        await asyncio.sleep(0.02)
        newest.write_bytes(image)
        successor.write_text(f"[VRC Camera] Took screenshot to: {newest}\n", encoding="utf-8")
        stamp = retiring.stat().st_mtime_ns + 1_000_000_000
        os.utime(successor, ns=(stamp, stamp))
        source.waiting.clear()
        source.emit(retiring, successor, created=(successor,))
        await eventually(source.waiting.is_set)
        assert delayed in watcher._pending_paths

        staged.rename(delayed)
        await eventually(lambda: not watcher._pending_paths)
        source.waiting.clear()
        source.emit()
        await eventually(source.waiting.is_set)
        resume.set()

        await eventually(lambda: len(offers) >= 2)
        assert [offer["path"] for offer in offers] == [str(previous), str(newest)]
        assert offers[-1]["offer_open"]
    finally:
        await owner.close()


def captured(tmp_path, name="capture.png", *, observed=None):
    return PhotoCapture(
        tmp_path / name,
        7,
        (1, 2),
        1_234_000_000,
        time.monotonic() if observed is None else observed,
    )


async def test_start_waits_for_registration_and_concurrent_starts_share_it(tmp_path):
    watcher = Watcher()
    watcher.start_gate = asyncio.Event()
    owner, _, _ = make_owner(tmp_path, watcher)
    first = asyncio.create_task(owner.start())
    second = asyncio.create_task(owner.start())
    try:
        await watcher.start_entered.wait()
        assert not first.done() and not second.done()
        watcher.start_gate.set()
        await asyncio.gather(first, second)
        assert watcher.active and watcher.starts == 1
    finally:
        await owner.close()
    assert not watcher.active
    assert owner._capture_task is None
    assert owner._scope.active_task_names == ()


async def test_close_during_startup_settles_start_and_watcher(tmp_path):
    watcher = Watcher()
    watcher.start_gate = asyncio.Event()
    owner, _, _ = make_owner(tmp_path, watcher)
    starting = asyncio.create_task(owner.start())
    await watcher.start_entered.wait()
    await owner.close()
    await starting
    assert not watcher.active
    assert owner.lifecycle.snapshot["photo"] is None
    assert owner._scope.active_task_names == ()


async def test_disabled_owner_never_observes_captures_and_resume_skips_disabled_era(tmp_path):
    state = {"enabled": False}
    watcher = Watcher()
    owner, resolve, _ = make_owner(tmp_path, watcher, enabled=lambda: state["enabled"])
    try:
        await owner.start()
        assert watcher.starts == watcher.waits == resolve.await_count == 0
        watcher.emit(captured(tmp_path, "disabled.png"))
        state["enabled"] = True
        owner.notify_policy_changed()
        await owner.wait_policy_applied()
        assert watcher.active and watcher.starts == 1
        assert owner.lifecycle.snapshot["photo"] is None
        watcher.emit(captured(tmp_path, "enabled.png"))
        await eventually(lambda: owner.lifecycle.snapshot["photo"] is not None)
        assert owner.lifecycle.snapshot["photo"]["path"].endswith("enabled.png")
        state["enabled"] = False
        owner.notify_policy_changed()
        assert (
            owner._control(
                {
                    "type": "photo_pull",
                    "photo_id": owner.lifecycle.snapshot["photo"]["photo_id"],
                    "event_id": "disabled-pull",
                },
                0,
            )
            is None
        )
        assert not owner.lifecycle.snapshot["photo"]["visible"]
        await owner.wait_policy_applied()
        assert not watcher.active
        assert owner.lifecycle.snapshot["photo"] is None
        watcher.emit(captured(tmp_path, "also-disabled.png"))
        state["enabled"] = True
        owner.notify_policy_changed()
        await owner.wait_policy_applied()
        assert watcher.starts == 2
        assert owner.lifecycle.snapshot["photo"] is None
    finally:
        await owner.close()


async def test_rapid_off_on_discards_queued_old_generation_without_native_start(tmp_path):
    state = {"enabled": True}
    watcher = Watcher()
    owner, _, _ = make_owner(tmp_path, watcher, enabled=lambda: state["enabled"])
    try:
        await owner.start()
        watcher.emit(captured(tmp_path, "stale.png"))
        state["enabled"] = False
        owner.notify_policy_changed()
        state["enabled"] = True
        owner.notify_policy_changed()
        await owner.wait_policy_applied()
        assert watcher.active and watcher.starts == 2
        assert owner.lifecycle.snapshot["photo"] is None
        owner.supervisor.start.assert_not_awaited()
        watcher.emit(captured(tmp_path, "fresh.png"))
        await eventually(lambda: owner.supervisor.start.await_count == 1)
        assert owner.lifecycle.snapshot["photo"]["path"].endswith("fresh.png")
    finally:
        await owner.close()


async def test_unrelated_policy_commit_preserves_pending_capture(tmp_path):
    watcher = Watcher()
    owner, _, _ = make_owner(tmp_path, watcher)
    try:
        await owner.start()
        watcher.emit(captured(tmp_path))
        owner.notify_policy_changed()
        await owner.wait_policy_applied()
        await eventually(lambda: owner.supervisor.start.await_count == 1)
        assert watcher.starts == 1
        assert owner.lifecycle.snapshot["photo"] is not None
    finally:
        await owner.close()


async def test_idle_and_lifecycle_output_do_not_repeat_policy_or_request_work(tmp_path):
    watcher = Watcher()
    owner, resolve, _ = make_owner(tmp_path, watcher)
    try:
        await owner.start()
        initial_resolutions = resolve.await_count
        await asyncio.sleep(0.3)
        assert owner.lifecycle.request_starts == 0
        owner.supervisor.start.assert_not_awaited()
        assert resolve.await_count == initial_resolutions
        watcher.emit(captured(tmp_path))
        await eventually(lambda: owner.supervisor.start.await_count == 1)
        photo_id = owner.lifecycle.snapshot["photo"]["photo_id"]
        assert owner.lifecycle.pull(photo_id, "pull")
        await eventually(lambda: owner.lifecycle.snapshot["photo"]["phase"] == "completed")
        resolutions = resolve.await_count
        starts = owner.lifecycle.request_starts
        native_starts = owner.supervisor.start.await_count
        owner.lifecycle.hidden(photo_id)
        assert owner.lifecycle.pull(photo_id, "reopen")
        await asyncio.sleep(0.3)
        assert owner.lifecycle.request_starts == starts
        assert owner.supervisor.start.await_count == native_starts
        assert resolve.await_count == resolutions
    finally:
        await owner.close()


async def test_watcher_failure_waits_for_explicit_policy_recovery(tmp_path):
    watcher = Watcher()
    owner, _, observations = make_owner(tmp_path, watcher)
    try:
        await owner.start()
        watcher.emit(OSError("watch failed"))
        await eventually(lambda: not watcher.active)
        assert watcher.starts == 1
        assert observations == [
            ("capture_unavailable", {"failure_code": "photo_capture_unavailable"})
        ]
        owner.notify_policy_changed()
        await owner.wait_policy_applied()
        assert watcher.active and watcher.starts == 2
        watcher.emit(captured(tmp_path))
        await eventually(lambda: owner.supervisor.start.await_count == 1)
    finally:
        await owner.close()


async def test_stale_capture_does_not_start_native_supervisor(tmp_path):
    watcher = Watcher()
    owner, resolve, _ = make_owner(tmp_path, watcher)
    try:
        await owner.start()
        watcher.emit(captured(tmp_path, observed=time.monotonic() - 11))
        await eventually(lambda: watcher.waits >= 2)
        assert owner.lifecycle.snapshot["photo"] is None
        assert resolve.await_count == 1
        owner.supervisor.start.assert_not_awaited()
    finally:
        await owner.close()


async def test_offer_expires_and_retention_cleans_up_without_policy_notifications(tmp_path):
    watcher = Watcher()
    discarded = AsyncMock()
    owner, _, _ = make_owner(tmp_path, watcher, discard=discarded)
    try:
        await owner.start()
        watcher.emit(captured(tmp_path, observed=time.monotonic() - 9.9))
        await eventually(
            lambda: owner.lifecycle.snapshot["photo"] is not None
            and owner.lifecycle.snapshot["photo"]["phase"] == "expired"
        )
        watcher.emit(captured(tmp_path, "translated.png"))
        await eventually(
            lambda: owner.lifecycle.snapshot["photo"]["path"].endswith("translated.png")
        )
        photo_id = owner.lifecycle.snapshot["photo"]["photo_id"]
        assert owner.lifecycle.pull(photo_id, "translate")
        await eventually(lambda: owner.lifecycle.snapshot["photo"]["phase"] == "completed")
        retained = owner.lifecycle._current
        owner.lifecycle._clock = lambda: retained.capture.observed_monotonic + 599.9
        owner.lifecycle.hidden(photo_id)
        await eventually(lambda: not owner.lifecycle._changed_event.is_set())
        owner.lifecycle._clock = lambda: retained.capture.observed_monotonic + 600
        await eventually(lambda: owner.lifecycle.snapshot["photo"] is None)
        await eventually(lambda: discarded.await_count == 1)
        assert discarded.await_args.args[0].file_id == (1, 2)
        assert list(tmp_path.iterdir()) == []
    finally:
        await owner.close()
    assert discarded.await_count == 1


async def test_policy_acknowledgment_reports_activation_failure_and_recovers(tmp_path):
    watcher = Watcher()
    owner, _, observations = make_owner(tmp_path, watcher)
    original_start = watcher.start
    watcher.start = AsyncMock(side_effect=OSError("registration failed"))
    try:
        await owner.wait_policy_applied()
        watcher.start.assert_not_awaited()
        await owner.start()
        with pytest.raises(OSError, match="registration failed"):
            await owner.wait_policy_applied()
        assert watcher.start.await_count == 1
        assert observations == [
            ("capture_unavailable", {"failure_code": "photo_capture_unavailable"})
        ]
        watcher.start = original_start
        await asyncio.to_thread(owner.notify_policy_changed)
        await owner.wait_policy_applied()
        assert watcher.active
    finally:
        await owner.close()
    await owner.wait_policy_applied()


async def test_disable_during_capture_resolution_cannot_publish_or_start_native(tmp_path):
    state = {"enabled": True}
    watcher = Watcher()
    owner, resolve, _ = make_owner(tmp_path, watcher, enabled=lambda: state["enabled"])
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def slow_route():
        entered.set()
        await finish.wait()
        return resolve.return_value

    try:
        await owner.start()
        owner._resolve_route = slow_route
        watcher.emit(captured(tmp_path))
        await entered.wait()
        state["enabled"] = False
        owner.notify_policy_changed()
        finish.set()
        await owner.wait_policy_applied()
        assert owner.lifecycle.snapshot["photo"] is None
        assert not watcher.active
        owner.supervisor.start.assert_not_awaited()
    finally:
        finish.set()
        await owner.close()


async def test_unrelated_commit_during_route_resolution_revalidates_without_losing_photo(tmp_path):
    watcher = Watcher()
    owner, resolve, _ = make_owner(tmp_path, watcher)
    entered = asyncio.Event()
    finish = asyncio.Event()
    calls = 0

    async def slow_route():
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await finish.wait()
        return resolve.return_value

    try:
        await owner.start()
        owner._resolve_route = slow_route
        watcher.emit(captured(tmp_path))
        await entered.wait()
        owner.notify_policy_changed()
        finish.set()
        await owner.wait_policy_applied()
        await eventually(lambda: owner.supervisor.start.await_count == 1)
        assert owner.lifecycle.snapshot["photo"] is not None
        assert watcher.starts == 1
        assert calls >= 2
    finally:
        finish.set()
        await owner.close()


async def test_close_before_control_task_starts_settles_start_waiter(tmp_path):
    watcher = Watcher()
    owner, _, _ = make_owner(tmp_path, watcher)
    starting = asyncio.create_task(owner.start())
    await asyncio.sleep(0)
    await owner.close()
    await starting
    assert not watcher.active
    assert owner._capture_task is None
    assert owner._scope.active_task_names == ()


async def test_terminal_control_failure_settles_policy_waiter_with_original_error(tmp_path):
    watcher = Watcher()
    owner, _, _ = make_owner(tmp_path, watcher)
    failure = OSError("retired cleanup failed")
    try:
        await owner.start()
        owner.lifecycle.refresh = AsyncMock(side_effect=failure)
        owner.notify_policy_changed()
        async with asyncio.timeout(2):
            with pytest.raises(OSError, match="retired cleanup failed"):
                await owner.wait_policy_applied()
        assert not watcher.active
        assert owner.lifecycle.snapshot["photo"] is None
    finally:
        with pytest.raises(OSError, match="retired cleanup failed"):
            await owner.close()
    owner.supervisor.close.assert_awaited_once()
    assert owner._scope.active_task_names == ()
