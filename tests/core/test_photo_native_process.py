from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import struct
import time
import zlib
from pathlib import Path

import pytest

from puripuly_heart.app.services.photo_application import PhotoApplicationOwner
from puripuly_heart.core.photo.capture import PhotoCaptureWatcher
from puripuly_heart.core.photo.lifecycle import PhotoRoute
from puripuly_heart.core.photo.process import PhotoProcessRunner
from puripuly_heart.core.photo.protocol import (
    PhotoRequestSent,
    PhotoStreamCompleted,
    PhotoTextDelta,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("PHOTO_NATIVE_EXECUTABLE"),
    reason="explicit native photo executable required",
)


def png(width, height):
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    data = (b"\x00" + b"\x40\x80\xc0" * width) * height
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(data))
        + chunk(b"IEND", b"")
    )


async def until(predicate):
    async with asyncio.timeout(20):
        while not predicate():
            await asyncio.sleep(0.01)


class ControlledConnection:
    model = "gpt-6-luna"
    google = False
    connect_ms = 0.0

    def __init__(self, records):
        self.records = records

    async def preopen(self):
        self.records["preopened"] += 1

    async def stream(self, image, target_language):
        assert base64.b64decode(image.jpeg_base64).startswith(b"\xff\xd8")
        assert image.source_path.read_bytes().startswith(b"\x89PNG")
        assert target_language == "ko"
        self.records["starts"] += 1
        self.records["prepared"] = image
        yield PhotoRequestSent(time.monotonic())
        yield PhotoTextDelta(
            '{"blocks":[{"bbox":[0,0,100,100],"source":"one","translation":"translated one"},'
        )
        await asyncio.sleep(0.03)
        yield PhotoTextDelta(
            '{"bbox":[100,100,200,200],"source":"two","translation":"translated two"}]}'
        )
        yield PhotoStreamCompleted({}, self.model)

    async def close(self):
        self.records["closed"] += 1


async def test_actual_native_pull_preparation_cumulative_reopen_restart_and_shutdown(tmp_path):
    executable = Path(os.environ["PHOTO_NATIVE_EXECUTABLE"]).resolve(strict=True)
    executable_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()
    photos = tmp_path / "photos"
    photos.mkdir()
    temporary = tmp_path / "owned"
    temporary.mkdir()
    (photos / "old.png").write_bytes(png(1280, 720))
    records = {"preopened": 0, "starts": 0, "closed": 0}
    route = PhotoRoute("controlled", lambda model: ControlledConnection(records))

    async def resolve():
        return route

    diagnostics = []
    owner = PhotoApplicationOwner(
        resolve_route=resolve,
        target_language=lambda: "ko",
        diagnostic=lambda stage, fields: diagnostics.append((stage, fields)),
        watcher=PhotoCaptureWatcher(photos, tmp_path / "isolated-logs"),
        runner=PhotoProcessRunner(executable),
        backend="probe",
        temporary_root=temporary,
        poll_interval=0.01,
    )
    try:
        await owner.start()
        assert owner.lifecycle.snapshot["photo"] is None
        source = photos / "new.png"
        source.write_bytes(png(1280, 720))
        await until(
            lambda: owner.lifecycle.snapshot["photo"] is not None and owner.bridge.ready.is_set()
        )
        photo_id = owner.lifecycle.snapshot["photo"]["photo_id"]
        captured = owner.lifecycle.snapshot["photo"]
        source_stat = source.stat()
        expected_file_id = [str(source_stat.st_dev), str(source_stat.st_ino)]
        assert captured["path"] == str(source)
        assert captured["expected_file_id"] == expected_file_id
        assert captured["attempt_id"] == 0
        assert captured["preparations"] == captured["blocks"] == []
        assert owner.bridge._preparations == {}
        assert owner.lifecycle.request_starts == 0
        assert not any(temporary.rglob("*.jpg"))
        assert records["starts"] == 0
        assert owner.lifecycle.snapshot["photo"]["image"] is None
        assert owner.bridge.send(
            {"type": "photo_probe_pull", "photo_id": photo_id, "event_id": "first"}
        )
        await until(lambda: owner.lifecycle.snapshot["photo"]["phase"] in ("completed", "failed"))
        assert owner.lifecycle.snapshot["photo"]["phase"] == "completed", diagnostics
        await until(
            lambda: any(
                r["photo_id"] == photo_id and r["block_count"] == 2 for r in owner.bridge.receipts
            )
        )
        assert records["starts"] == 1
        assert records["prepared"].source_width == 1280
        assert [b["source"] for b in owner.lifecycle.snapshot["photo"]["blocks"]] == ["one", "two"]
        assert not owner.lifecycle.snapshot["photo"]["first_block_drawn"]
        native_timings = {
            stage: fields["duration_ms"]
            for stage, fields in diagnostics
            if stage in ("decode", "resize", "jpeg")
        }
        assert set(native_timings) == {"decode", "resize", "jpeg"}
        assert all(value >= 0 for value in native_timings.values())
        assert owner.bridge.send({"type": "photo_probe_hide", "photo_id": photo_id})
        await until(lambda: not owner.lifecycle.snapshot["photo"]["visible"])
        assert owner.bridge.send(
            {"type": "photo_probe_pull", "photo_id": photo_id, "event_id": "reopen"}
        )
        await until(lambda: owner.lifecycle.snapshot["photo"]["visible"])
        assert records["starts"] == 1
        old_pid = owner.supervisor.pid
        old_instance = owner.supervisor.instance_id
        old_generation = owner.supervisor.generation
        await owner.supervisor.process.terminate()
        await until(
            lambda: owner.supervisor.generation > old_generation and owner.bridge.ready.is_set()
        )
        await until(
            lambda: any(
                r["generation"] == owner.supervisor.generation
                and r["photo_id"] == photo_id
                and r["block_count"] == 2
                for r in owner.bridge.receipts
            )
        )
        assert owner.supervisor.pid != old_pid
        assert owner.supervisor.instance_id != old_instance
        assert records["starts"] == 1
        assert owner.lifecycle.snapshot["photo"]["path"] == str(source)
        assert owner.lifecycle.snapshot["photo"]["expected_file_id"] == expected_file_id
        assert not owner.lifecycle.snapshot["photo"]["first_block_drawn"]
        print(
            {
                "executable": executable.name,
                "old_pid": old_pid,
                "new_pid": owner.supervisor.pid,
                "generations": [old_generation, owner.supervisor.generation],
                "provider_starts": records["starts"],
                "native_receipt_blocks": 2,
                "draw_observed": False,
                "backend": "probe",
                "executable_sha256": executable_sha256,
                "native_stage_ms": native_timings,
            }
        )
    finally:
        await owner.close()
    assert owner.supervisor.process is None
    assert owner.supervisor.last_exit_code == 0
    assert owner.supervisor.last_shutdown_acknowledged
    assert list(temporary.iterdir()) == []


async def test_actual_native_death_during_prepare_fails_without_resuming_job(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    owned = tmp_path / "owned"
    owned.mkdir()
    records = {"preopened": 0, "starts": 0, "closed": 0}
    route = PhotoRoute("controlled", lambda model: ControlledConnection(records))

    async def resolve():
        return route

    owner = PhotoApplicationOwner(
        resolve_route=resolve,
        target_language=lambda: "ko",
        diagnostic=lambda stage, fields: None,
        watcher=PhotoCaptureWatcher(photos, tmp_path / "isolated-logs"),
        runner=PhotoProcessRunner(Path(os.environ["PHOTO_NATIVE_EXECUTABLE"])),
        backend="probe",
        temporary_root=owned,
        poll_interval=0.01,
    )
    try:
        await owner.start()
        (photos / "incomplete.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        await until(
            lambda: owner.lifecycle.snapshot["photo"] is not None and owner.bridge.ready.is_set()
        )
        photo_id = owner.lifecycle.snapshot["photo"]["photo_id"]
        owner.bridge.send(
            {"type": "photo_probe_pull", "photo_id": photo_id, "event_id": "prepare-pull"}
        )
        await until(lambda: bool(owner.bridge._preparations))
        old_generation = owner.supervisor.generation
        await owner.supervisor.process.terminate()
        await until(lambda: owner.lifecycle.snapshot["photo"]["phase"] == "failed")
        await until(
            lambda: owner.supervisor.generation > old_generation and owner.bridge.ready.is_set()
        )
        assert owner.lifecycle.snapshot["photo"]["failure_code"] == "photo_preparation_failed"
        assert owner.lifecycle.snapshot["photo"]["image"] is None
        assert owner.bridge._preparations == {}
        assert records["starts"] == 0
        assert owner.lifecycle.request_starts == 0
        print(
            {
                "preparation_process_loss": "explicit failure",
                "generation_after_recovery": owner.supervisor.generation,
                "provider_starts": 0,
                "job_resumed": False,
            }
        )
    finally:
        await owner.close()
    assert list(owned.iterdir()) == []


def jpeg_size(data: bytes) -> tuple[int, int]:
    index = 2
    while index < len(data):
        marker, length = data[index + 1], struct.unpack(">H", data[index + 2 : index + 4])[0]
        if marker in (0xC0, 0xC2):
            height, width = struct.unpack(">HH", data[index + 5 : index + 9])
            return width, height
        index += 2 + length
    raise AssertionError("JPEG frame header missing")


async def test_actual_native_portrait_photo_is_sent_upright_in_portrait_canvas(tmp_path):
    executable = Path(os.environ["PHOTO_NATIVE_EXECUTABLE"]).resolve(strict=True)
    photos = tmp_path / "photos"
    photos.mkdir()
    temporary = tmp_path / "owned"
    temporary.mkdir()
    records = {"preopened": 0, "starts": 0, "closed": 0}
    route = PhotoRoute("controlled", lambda model: ControlledConnection(records))
    calibration = tmp_path / "photo-wrist-calibration.json"
    calibration.write_text(
        '{"transform":[[1,0,0,0.05],[0,1,0,0.03],[0,0,1,0.06]],"reference":{"model":"synthetic",'
        '"serial":"synthetic","handgrip_transform":[[1,0,0,0],[0,1,0,0],[0,0,1,0]]}}',
        encoding="utf-8",
    )

    async def resolve():
        return route

    owner = PhotoApplicationOwner(
        resolve_route=resolve,
        target_language=lambda: "ko",
        diagnostic=lambda stage, fields: None,
        wrist_calibration_path=calibration,
        watcher=PhotoCaptureWatcher(photos, tmp_path / "isolated-logs"),
        runner=PhotoProcessRunner(executable),
        backend="probe",
        temporary_root=temporary,
        poll_interval=0.01,
    )
    try:
        await owner.start()
        (photos / "portrait.png").write_bytes(png(540, 960))
        await until(
            lambda: owner.lifecycle.snapshot["photo"] is not None and owner.bridge.ready.is_set()
        )
        photo_id = owner.lifecycle.snapshot["photo"]["photo_id"]
        assert owner.bridge.send(
            {"type": "photo_probe_pull", "photo_id": photo_id, "event_id": "portrait"}
        )
        await until(lambda: owner.lifecycle.snapshot["photo"]["phase"] in ("completed", "failed"))
        assert owner.lifecycle.snapshot["photo"]["phase"] == "completed"
        prepared = records["prepared"]
        assert prepared.send_size == (720, 1280)
        assert jpeg_size(base64.b64decode(prepared.jpeg_base64)) == (720, 1280)
        assert (prepared.offset_x, prepared.offset_y) == (0, 0)
        boxes = [b["bbox"] for b in owner.lifecycle.snapshot["photo"]["blocks"]]
        assert boxes == [[0.0, 0.0, 75.0, 75.0], [75.0, 75.0, 150.0, 150.0]]
        assert owner._interaction()["wrist_calibration"]["reference"]["model"] == "synthetic"
    finally:
        await owner.close()
    assert owner.supervisor.last_exit_code == 0
