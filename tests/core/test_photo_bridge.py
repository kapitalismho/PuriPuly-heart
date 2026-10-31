from __future__ import annotations

import asyncio
import json

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from puripuly_heart.core.photo.bridge import PhotoBridge
from puripuly_heart.core.photo.protocol import PhotoProviderError


def auth(manifest):
    return {
        key: manifest[key]
        for key in ("protocol_version", "role", "session_token", "instance_id", "generation")
    } | {"type": "photo_auth"}


async def receive(connection, kind):
    async with asyncio.timeout(2):
        while True:
            value = json.loads(await connection.recv())
            if value["type"] == kind:
                return value


async def test_wrong_identity_unknown_control_and_old_generation_are_rejected():
    controls = []
    bridge = PhotoBridge(control=lambda value, generation: controls.append((value, generation)))
    await bridge.start()
    first = await bridge.activate(1, backend="probe")
    try:
        for mutation in (
            {"session_token": "wrong"},
            {"role": "caption"},
            {"generation": 2},
            {"protocol_version": True},
        ):
            async with connect(first["bridge_url"]) as connection:
                await connection.send(json.dumps(auth(first) | mutation))
                with pytest.raises(ConnectionClosed):
                    await connection.recv()
        async with connect(first["bridge_url"]) as connection:
            await connection.send(json.dumps(auth(first)))
            assert (await receive(connection, "photo_snapshot"))["photo"] is None
            await connection.send(
                json.dumps({"type": "photo_pull", "photo_id": "current", "event_id": "one"})
            )
            async with asyncio.timeout(1):
                while not controls:
                    await asyncio.sleep(0.001)
            assert controls == [
                ({"type": "photo_pull", "photo_id": "current", "event_id": "one"}, 1)
            ]
            await connection.send(json.dumps({"type": "unknown"}))
            with pytest.raises(ConnectionClosed):
                await connection.recv()
        second = await bridge.activate(2, backend="probe")
        assert first["session_token"] != second["session_token"]
        assert first["instance_id"] != second["instance_id"]
        async with connect(second["bridge_url"]) as connection:
            await connection.send(json.dumps(auth(first)))
            with pytest.raises(ConnectionClosed):
                await connection.recv()
    finally:
        await bridge.close()


async def test_full_snapshot_coalescing_retains_every_independent_block_and_gates_probe():
    bridge = PhotoBridge(control=lambda value, generation: None)
    await bridge.start()
    manifest = await bridge.activate(1, backend="openvr")
    try:
        async with connect(manifest["bridge_url"]) as connection:
            await connection.send(json.dumps(auth(manifest)))
            await receive(connection, "photo_snapshot")
            await connection.send(json.dumps({"type": "photo_ready"}))
            await asyncio.wait_for(bridge.ready.wait(), 1)
            assert not bridge.send(
                {"type": "photo_probe_pull", "photo_id": "one", "event_id": "probe"}
            )
            blocks = []
            for revision in range(1, 41):
                blocks.append({"block_id": revision - 1})
                bridge.publish(
                    {
                        "type": "photo_snapshot",
                        "revision": revision,
                        "photo": {"blocks": list(blocks)},
                    }
                )
            snapshot = await receive(connection, "photo_snapshot")
            assert snapshot["revision"] == 40
            assert [b["block_id"] for b in snapshot["photo"]["blocks"]] == list(range(40))
    finally:
        await bridge.close()


async def test_preparation_response_is_fenced_and_cancelled_work_is_drained(tmp_path):
    bridge = PhotoBridge(control=lambda value, generation: None)
    await bridge.start()
    manifest = await bridge.activate(1, backend="probe")
    try:
        async with connect(manifest["bridge_url"]) as connection:
            await connection.send(json.dumps(auth(manifest)))
            await receive(connection, "photo_snapshot")
            task = asyncio.create_task(
                bridge.prepare(
                    "current", 1, tmp_path / "capture.png", tmp_path, expected_file_id=(9, 2**100)
                )
            )
            request = await receive(connection, "photo_prepare")
            assert request["expected_file_id"] == ["9", str(2**100)]
            response = {
                "type": "photo_prepared",
                "photo_id": "stale",
                "attempt_id": 1,
                "source_path": str(tmp_path / "source.png"),
                "jpeg_path": str(tmp_path / "send.jpg"),
                "width": 1920,
                "height": 1080,
                "scale": 2 / 3,
                "offset_x": 0,
                "offset_y": 0,
                "preparation_ms": 4.0,
            }
            await connection.send(json.dumps(response))
            await asyncio.sleep(0.01)
            assert not task.done()
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            response["photo_id"] = "current"
            await connection.send(json.dumps(response))
            with pytest.raises(asyncio.CancelledError):
                await task
    finally:
        await bridge.close()


async def test_process_stop_fails_pending_preparation_without_reissuing(tmp_path):
    bridge = PhotoBridge(control=lambda value, generation: None)
    await bridge.start()
    manifest = await bridge.activate(1, backend="probe")
    try:
        async with connect(manifest["bridge_url"]) as connection:
            await connection.send(json.dumps(auth(manifest)))
            await receive(connection, "photo_snapshot")
            task = asyncio.create_task(
                bridge.prepare(
                    "current", 1, tmp_path / "capture.png", tmp_path, expected_file_id=(1, 2)
                )
            )
            await receive(connection, "photo_prepare")
            bridge.renderer_stopped(1)
            with pytest.raises(PhotoProviderError, match="photo_preparation_failed"):
                await task
            assert bridge._preparations == {}
    finally:
        await bridge.close()


@pytest.mark.parametrize(
    "mutation",
    [
        {"stage": "unknown"},
        {"duration_ms": -1},
        {"duration_ms": float("nan")},
        {"duration_ms": float("inf")},
        {"duration_ms": 10**400},
        {"duration_ms": True},
        {"block_id": True},
        {"attempt_id": True},
        {"source": "private content must not cross this event"},
        {"source_path": "private-path"},
    ],
)
async def test_image_timing_rejects_nonfinite_unknown_or_content_fields(mutation):
    bridge = PhotoBridge(control=lambda value, generation: None)
    await bridge.start()
    manifest = await bridge.activate(1, backend="probe")
    try:
        async with connect(manifest["bridge_url"]) as connection:
            await connection.send(json.dumps(auth(manifest)))
            await receive(connection, "photo_snapshot")
            value = {
                "type": "photo_image_timing",
                "photo_id": "current",
                "attempt_id": 1,
                "block_id": 0,
                "stage": "mask",
                "duration_ms": 2.5,
            }
            await connection.send(json.dumps(value | mutation))
            with pytest.raises(ConnectionClosed):
                await connection.recv()
    finally:
        await bridge.close()


async def test_image_timing_preserves_hidden_work_and_current_progress_after_stale_rejection():
    controls = []
    bridge = PhotoBridge(control=lambda value, generation: controls.append((value, generation)))
    await bridge.start()
    manifest = await bridge.activate(1, backend="probe")
    photo = {
        "photo_id": "current",
        "attempt_id": 1,
        "phase": "requesting",
        "blocks": [{"block_id": 1}],
        "preparations": [{"block_id": 0, "source": "SOURCE", "source_line_count": 1}],
    }
    bridge.publish({"type": "photo_snapshot", "revision": 1, "photo": photo})
    value = {
        "type": "photo_image_timing",
        "photo_id": "current",
        "attempt_id": 1,
        "block_id": 0,
        "stage": "mask",
        "duration_ms": 2.5,
    }
    try:
        async with connect(manifest["bridge_url"]) as connection:
            await connection.send(json.dumps(auth(manifest)))
            await receive(connection, "photo_snapshot")
            for stage in ("mask", "erase"):
                await connection.send(json.dumps(value | {"stage": stage}))
            for stage in ("mask", "erase", "typeset", "roi_upload", "compose_gpu"):
                await connection.send(json.dumps(value | {"block_id": 1, "stage": stage}))
            await connection.send(json.dumps(value | {"stage": "erase", "duration_ms": 0.5}))
            for mutation in (
                {"photo_id": "retired"},
                {"attempt_id": 2},
                {"block_id": 99},
                {"stage": "typeset"},
            ):
                await connection.send(json.dumps(value | mutation))
            await connection.send(
                json.dumps(value | {"block_id": 1, "stage": "typeset", "duration_ms": 9})
            )
            async with asyncio.timeout(1):
                while not controls or controls[-1][0]["duration_ms"] != 9:
                    await asyncio.sleep(0)
            assert len(controls) == 9
            assert [event["duration_ms"] for event, _ in controls if event["block_id"] == 0] == [
                2.5,
                2.5,
                0.5,
            ]
            assert {generation for _, generation in controls} == {1}
            expired = photo | {"phase": "expired"}
            bridge.publish({"type": "photo_snapshot", "revision": 2, "photo": expired})
            assert (await receive(connection, "photo_snapshot"))["revision"] == 2
            bridge.ready.clear()
            await connection.send(json.dumps(value))
            await connection.send(json.dumps({"type": "photo_ready"}))
            await asyncio.wait_for(bridge.ready.wait(), 1)
            assert len(controls) == 9
            retried = photo | {
                "attempt_id": 2,
                "blocks": [{"block_id": 8}],
                "preparations": [],
            }
            bridge.publish({"type": "photo_snapshot", "revision": 3, "photo": retried})
            assert (await receive(connection, "photo_snapshot"))["revision"] == 3
            await connection.send(
                json.dumps(value | {"attempt_id": 2, "block_id": 8, "stage": "typeset"})
            )
            async with asyncio.timeout(1):
                while len(controls) < 10:
                    await asyncio.sleep(0)
            assert controls[-1][0]["attempt_id"] == 2
            assert controls[-1][0]["block_id"] == 8
    finally:
        await bridge.close()
