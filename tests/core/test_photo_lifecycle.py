from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from puripuly_heart.core.photo.capture import PhotoCapture
from puripuly_heart.core.photo.image import PreparedPhoto
from puripuly_heart.core.photo.lifecycle import PhotoLifecycle, PhotoRoute
from puripuly_heart.core.photo.protocol import (
    PhotoProviderError,
    PhotoRequestSent,
    PhotoStreamCompleted,
    PhotoTextDelta,
)


async def eventually(predicate) -> None:
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


class Connection:
    def __init__(self, model, script, *, google=False, preopen_error=None, clock=time.monotonic):
        self.model = model
        self.google = google
        self.script = script
        self.preopen_error = preopen_error
        self.opened = False
        self.closed = False
        self.starts = 0
        self.connect_ms = 1.0
        self.clock = clock

    async def preopen(self):
        self.opened = True
        if self.preopen_error:
            raise self.preopen_error

    async def stream(self, image, target_language):
        self.starts += 1
        yield PhotoRequestSent(self.clock())
        for event in self.script:
            if isinstance(event, asyncio.Event):
                await event.wait()
            elif isinstance(event, Exception):
                raise event
            else:
                yield event

    async def close(self):
        self.closed = True


def block(source="source", translation="translated", *, google=False):
    key = "box_2d" if google else "bbox"
    return PhotoTextDelta(
        f'{{"blocks":[{{"{key}":[0,0,100,100],"source":"{source}","translation":"{translation}"}}]}}'
    )


def capture(path, clock=0):
    return PhotoCapture(path, 1, (1, 2), 1234, clock)


class Harness:
    def __init__(self, tmp_path, scripts, *, google=False, prepare=None):
        self.time = 0.0
        self.connections = []
        self.snapshots = []
        self.measurements = []
        self.prepares = []
        self.scripts = list(scripts)
        self.google = google
        self.route = PhotoRoute("google" if google else "luna", self.connect)

        async def resolve():
            return self.route

        async def prepare_image(photo_id, attempt_id, source, directory, *, expected_file_id):
            self.prepares.append((photo_id, attempt_id, source, expected_file_id))
            if prepare is not None:
                return await prepare(
                    photo_id, attempt_id, source, directory, expected_file_id=expected_file_id
                )
            snapshot = directory / "source.png"
            snapshot.write_bytes(b"immutable")
            return PreparedPhoto("jpeg", snapshot, 1920, 1080, 2 / 3, 0, 0, 4.5)

        self.owner = PhotoLifecycle(
            resolve_route=resolve,
            target_language=lambda: "ko",
            publish=self.snapshots.append,
            diagnostic=lambda stage, fields: self.measurements.append((stage, fields)),
            prepare=prepare_image,
            clock=lambda: self.time,
            temporary_root=tmp_path,
        )

    def connect(self, model):
        connection = Connection(
            model or ("gemini-3.5-flash-lite" if self.google else "gpt-6-luna"),
            self.scripts.pop(0),
            google=self.google,
            clock=lambda: self.time,
        )
        self.connections.append(connection)
        return connection

    @property
    def photo(self):
        return self.owner.snapshot["photo"]

    async def offer(self):
        await self.owner.captured(capture(Path("capture.png"), self.time))
        await eventually(lambda: self.connections and self.connections[-1].opened)
        return self.photo["photo_id"]


async def test_offer_preopens_without_preparation_or_send_then_reopen_reuses_result(tmp_path):
    h = Harness(tmp_path, [[block(), PhotoStreamCompleted({}, "gpt-6-luna")]])
    try:
        photo_id = await h.offer()
        assert h.prepares == []
        assert h.photo["expected_file_id"] == ["1", "2"]
        assert h.photo["attempt_id"] == 0
        assert h.photo["image"] is None
        assert h.photo["preparations"] == []
        assert h.photo["blocks"] == []
        assert h.owner.request_starts == 0
        assert h.owner.pull(photo_id, "pull-1")
        await eventually(lambda: h.photo["phase"] == "completed")
        assert h.owner.request_starts == 1
        assert len(h.prepares) == 1
        assert h.prepares[0] == (photo_id, 1, Path(h.photo["path"]), (1, 2))
        assert h.photo["blocks"][0]["bbox"] == [0.0, 0.0, 150.0, 150.0]
        assert h.photo["preparations"][0]["source_line_count"] == 1
        h.owner.hidden(photo_id)
        assert not h.photo["visible"]
        h.time = 100
        assert h.owner.pull(photo_id, "pull-2")
        assert h.photo["visible"]
        assert not h.owner.pull(photo_id, "pull-2")
        assert h.owner.request_starts == 1
        assert h.connections[0].closed
        stages = [stage for stage, _ in h.measurements]
        assert {"connection", "preparation", "ttft", "first_block", "total"} <= set(stages)
    finally:
        await h.owner.close()
    assert list(tmp_path.iterdir()) == []


async def test_capture_identity_is_immutable_through_prepare_reopen_and_replacement(tmp_path):
    h = Harness(tmp_path, [
        [block(), PhotoStreamCompleted({}, "gpt-6-luna")],
        [block(), PhotoStreamCompleted({}, "gpt-6-luna")],
    ])
    first = PhotoCapture(tmp_path / "first.png", 17, (9, 2**100), 1234, 0)
    second = PhotoCapture(tmp_path / "second.png", 23, (10, 2**101), 1235, 0)
    try:
        assert await h.owner.captured(first)
        await eventually(lambda: h.connections and h.connections[0].opened)
        before_pull = h.photo
        first_id = before_pull["photo_id"]
        assert before_pull["path"] == str(first.path)
        assert before_pull["expected_file_id"] == ["9", str(2**100)]
        assert before_pull["attempt_id"] == 0
        assert before_pull["image"] is None
        assert before_pull["preparations"] == before_pull["blocks"] == []
        assert not before_pull["visible"]
        assert h.prepares == []
        assert h.owner.request_starts == h.connections[0].starts == 0
        before_pull["expected_file_id"][0] = "mutated-consumer-copy"
        assert h.photo["expected_file_id"] == ["9", str(2**100)]
        assert h.owner.pull(first_id, "authorize")
        await eventually(lambda: h.photo["phase"] == "completed")
        assert h.prepares == [(first_id, 1, first.path, first.file_id)]
        h.owner.hidden(first_id)
        assert h.owner.pull(first_id, "reopen")
        assert len(h.prepares) == h.owner.request_starts == 1
        assert h.photo["path"] == str(first.path)
        assert h.photo["expected_file_id"] == ["9", str(2**100)]
        assert await h.owner.captured(second)
        second_id = h.photo["photo_id"]
        assert second_id != first_id
        assert h.photo["path"] == str(second.path)
        assert h.photo["expected_file_id"] == ["10", str(2**101)]
        assert h.photo["attempt_id"] == 0 and h.photo["image"] is None
        assert not h.owner.pull(first_id, "stale")
        assert h.owner.pull(second_id, "authorize-second")
        await eventually(lambda: h.photo["phase"] == "completed")
        assert h.prepares[-1] == (second_id, 1, second.path, second.file_id)
    finally:
        await h.owner.close()


async def test_offer_expiry_and_started_request_retention_are_distinct(tmp_path):
    gate = asyncio.Event()
    h = Harness(tmp_path, [[gate, block(), PhotoStreamCompleted({}, "gpt-6-luna")], []])
    try:
        photo_id = await h.offer()
        h.time = 9.9
        assert h.owner.pull(photo_id, "valid")
        await eventually(lambda: h.owner.request_starts == 1)
        h.time = 11
        await h.owner.refresh(h.route)
        assert h.photo["phase"] == "requesting"
        gate.set()
        await eventually(lambda: h.photo["phase"] == "completed")
        h.time = 599
        await h.owner.refresh(h.route)
        assert h.photo["phase"] == "completed"
        h.time = 600
        await h.owner.refresh(h.route)
        assert h.photo is None
        h.time = 700
        new_id = await h.offer()
        h.time = 710
        assert not h.owner.pull(new_id, "late")
        await h.owner.refresh(h.route)
        await eventually(lambda: h.connections[-1].closed)
        assert h.photo["phase"] == "expired"
        assert h.prepares and len(h.prepares) == 1
    finally:
        await h.owner.close()


async def test_google_quota_fallback_consumes_second_send_and_does_not_retry_again(tmp_path):
    h = Harness(
        tmp_path,
        [
            [PhotoProviderError("photo_quota_exhausted", quota=True)],
            [PhotoProviderError("photo_transport_failed")],
        ],
        google=True,
    )
    try:
        photo_id = await h.offer()
        h.owner.pull(photo_id, "pull")
        await eventually(lambda: h.photo["phase"] == "failed")
        assert [c.model for c in h.connections] == [
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
        ]
        assert h.owner.request_starts == 2
        assert h.photo["attempt_id"] == 2
        assert len(h.prepares) == 1
        assert h.photo["expected_file_id"] == ["1", "2"]
        assert all(
            snapshot["photo"]["expected_file_id"] == ["1", "2"]
            for snapshot in h.snapshots if snapshot["photo"] is not None
        )
        assert h.prepares[0][3] == (1, 2)
    finally:
        await h.owner.close()


async def test_partial_failure_clears_attempt_and_does_not_switch_model(tmp_path):
    second = asyncio.Event()
    first_text = block("old", "old translated", google=True)
    h = Harness(
        tmp_path,
        [
            [first_text, PhotoProviderError("photo_quota_exhausted", quota=True)],
            [
                second,
                block("new", "new translated", google=True),
                PhotoStreamCompleted({}, "gemini-3.5-flash-lite"),
            ],
        ],
        google=True,
    )
    try:
        photo_id = await h.offer()
        h.owner.pull(photo_id, "pull")
        await eventually(lambda: h.photo["attempt_id"] == 2)
        assert h.photo["blocks"] == []
        assert h.photo["preparations"] == []
        assert [c.model for c in h.connections] == ["gemini-3.5-flash-lite"] * 2
        assert any(
            s["photo"] and s["photo"]["attempt_id"] == 1 and s["photo"]["blocks"]
            for s in h.snapshots
        )
        second.set()
        await eventually(lambda: h.photo["phase"] == "completed")
        assert [b["source"] for b in h.photo["blocks"]] == ["new"]
    finally:
        await h.owner.close()


@pytest.mark.parametrize("code", ["photo_authentication_failed", "photo_usage_limit"])
async def test_nonretryable_failures_immediately_fail_with_safe_diagnostics(tmp_path, code):
    h = Harness(tmp_path, [[PhotoProviderError(code, retryable=False)]])
    try:
        photo_id = await h.offer()
        h.owner.pull(photo_id, "pull")
        await eventually(lambda: h.photo["phase"] == "failed")
        assert h.owner.request_starts == 1
        assert h.photo["failure_code"] == code
        assert all(
            "capture.png" not in str(item) and "jpeg" not in str(item) for item in h.measurements
        )
    finally:
        await h.owner.close()


async def test_haptic_requires_draw_and_is_not_replayed_after_reopen_or_recovery(tmp_path):
    h = Harness(
        tmp_path,
        [
            [block(), PhotoStreamCompleted({}, "gpt-6-luna")],
            [block(), PhotoStreamCompleted({}, "gpt-6-luna")],
        ],
    )
    try:
        h.owner.renderer_changed(1)
        photo_id = await h.offer()
        h.owner.pull(photo_id, "pull")
        assert h.owner.first_block_drawn(photo_id, 1, 0, 1) is None
        await eventually(lambda: h.photo["phase"] == "completed")
        assert h.owner.first_block_drawn(photo_id, 1, 0, 2) is None
        assert h.owner.first_block_drawn(photo_id, 1, 1, 1) is None
        assert h.owner.first_block_drawn(photo_id, 1, 0, 1)["type"] == "photo_haptic"
        h.owner.hidden(photo_id)
        h.owner.pull(photo_id, "reopen")
        assert h.owner.first_block_drawn(photo_id, 1, 0, 1) is None
        h.owner.renderer_changed(2)
        assert h.owner.first_block_drawn(photo_id, 1, 0, 2) is None
        new_id = await h.offer()
        h.owner.pull(new_id, "pull-new")
        await eventually(lambda: h.photo["phase"] == "completed")
        h.owner.renderer_changed(3)
        assert h.owner.first_block_drawn(new_id, 1, 0, 3) is None
        assert h.owner.request_starts == 1
    finally:
        await h.owner.close()


async def test_replaced_preparation_drains_and_cannot_publish_stale_image(tmp_path):
    started = asyncio.Event()
    finish = asyncio.Event()
    drained = asyncio.Event()

    async def prepare(photo_id, attempt_id, source, directory, **kwargs):
        started.set()
        try:
            await finish.wait()
        except asyncio.CancelledError:
            await finish.wait()
        snapshot = directory / "source.png"
        snapshot.write_bytes(b"old")
        drained.set()
        return PreparedPhoto("jpeg", snapshot, 1280, 720, 1, 0, 0, 1)

    h = Harness(tmp_path, [[], []], prepare=prepare)
    try:
        old_id = await h.offer()
        h.owner.pull(old_id, "old-pull")
        await started.wait()
        new_id = await h.offer()
        assert old_id != new_id
        assert not h.owner.pull(old_id, "stale")
        finish.set()
        await drained.wait()
        await eventually(lambda: h.connections[0].closed)
        await h.owner.refresh(h.route)
        assert h.photo["photo_id"] == new_id
        assert h.photo["image"] is None
        assert h.owner.request_starts == 0
    finally:
        finish.set()
        await h.owner.close()
    assert list(tmp_path.iterdir()) == []


async def test_route_revocation_suppresses_offer_and_cancels_active_request(tmp_path):
    gate = asyncio.Event()
    h = Harness(tmp_path, [[gate]])
    try:
        photo_id = await h.offer()
        h.owner.pull(photo_id, "pull")
        await eventually(lambda: h.owner.request_starts == 1)
        h.route = None
        await h.owner.refresh(None)
        assert h.photo is None
        await eventually(lambda: h.connections[0].closed)
        assert not await h.owner.captured(capture(Path("unsupported.png")))
        assert h.photo is None
    finally:
        await h.owner.close()


async def test_every_block_accumulates_and_stage_timing_starts_at_send_not_pull(tmp_path):
    bbox = asyncio.Event()
    source = asyncio.Event()
    complete = asyncio.Event()
    h = Harness(
        tmp_path,
        [
            [
                bbox,
                PhotoTextDelta('{"blocks":[{"bbox":[0,0,100,100],'),
                source,
                PhotoTextDelta('"source":"one\\ntwo",'),
                complete,
                PhotoTextDelta(
                    '"translation":"translated"},{"bbox":[100,100,200,200],"source":"second","translation":"second translated"}]}'
                ),
                PhotoStreamCompleted({}, "gpt-6-luna"),
            ]
        ],
    )
    try:
        photo_id = await h.offer()
        h.time = 8
        h.owner.pull(photo_id, "pull")
        await eventually(lambda: h.owner.request_starts == 1)
        h.time = 11
        bbox.set()
        await eventually(lambda: len(h.photo["preparations"]) == 1)
        assert h.photo["preparations"][0]["source_line_count"] is None
        assert h.photo["blocks"] == []
        source.set()
        await eventually(lambda: h.photo["preparations"][0]["source_line_count"] == 2)
        assert h.photo["blocks"] == []
        h.time = 12
        complete.set()
        await eventually(lambda: h.photo["phase"] == "completed")
        assert [b["block_id"] for b in h.photo["blocks"]] == [0, 1]
        assert (
            next(fields["duration_ms"] for stage, fields in h.measurements if stage == "ttft")
            == 3000
        )
        assert (
            next(
                fields["duration_ms"] for stage, fields in h.measurements if stage == "first_block"
            )
            == 4000
        )
    finally:
        await h.owner.close()


async def test_preopen_failure_is_bounded_without_spending_send_budget(tmp_path):
    h = Harness(tmp_path, [[], []])
    original = h.route.connect

    def connect(model):
        connection = original(model)
        connection.preopen_error = PhotoProviderError("photo_transport_failed")
        return connection

    h.route = PhotoRoute("luna", connect)
    try:
        await h.offer()
        await eventually(lambda: h.photo["phase"] == "failed")
        assert len(h.connections) == 2
        assert h.owner.request_starts == 0
        assert h.prepares == []
        assert all(connection.closed for connection in h.connections)
    finally:
        await h.owner.close()


async def test_new_attempt_can_grant_first_draw_again_but_old_attempt_cannot(tmp_path):
    fail = asyncio.Event()
    retry = asyncio.Event()
    h = Harness(
        tmp_path,
        [
            [block(), fail, PhotoProviderError("photo_transport_failed")],
            [retry, block(), PhotoStreamCompleted({}, "gpt-6-luna")],
        ],
    )
    try:
        h.owner.renderer_changed(1)
        photo_id = await h.offer()
        h.owner.pull(photo_id, "pull")
        await eventually(lambda: bool(h.photo["blocks"]))
        assert h.owner.first_block_drawn(photo_id, 1, 0, 1) is not None
        fail.set()
        await eventually(lambda: h.photo["attempt_id"] == 2)
        assert not h.photo["first_block_drawn"]
        assert h.owner.first_block_drawn(photo_id, 1, 0, 1) is None
        retry.set()
        await eventually(lambda: h.photo["phase"] == "completed")
        assert h.owner.first_block_drawn(photo_id, 2, 0, 1) is not None
        assert h.owner.first_block_drawn(photo_id, 2, 0, 1) is None
    finally:
        await h.owner.close()
