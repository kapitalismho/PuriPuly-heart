from __future__ import annotations

import asyncio
import importlib.util
import struct
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/probes/x3_photo_preview.py"
spec = importlib.util.spec_from_file_location("x3_photo_preview", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def make_png_header(path, width=1920, height=1080):
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + struct.pack(">II", width, height) + bytes(9)
    )
    return path


@pytest.fixture
def preview(tmp_path):
    published = []
    events = []
    state = probe.PreviewState(
        make_png_header(tmp_path / "selected.png"),
        1920,
        1080,
        published.append,
        lambda kind, **fields: events.append((kind, fields)),
    )
    state.snapshot()
    return state, published, events


def pull(state, now, event_id="pull", photo_id=None, generation=1):
    return state.control(
        {
            "type": "photo_pull",
            "photo_id": photo_id or state.photo["photo_id"],
            "event_id": event_id,
        },
        generation,
        now,
    )


def test_no_payload_before_pull_and_ten_second_offer_expiry(preview):
    state, snapshots, events = preview
    assert snapshots == [{"type": "photo_snapshot", "revision": 0, "photo": None}]
    state.offer(100, 123000)
    offer = snapshots[-1]["photo"]
    assert (offer["attempt_id"], offer["image"], offer["offer_open"]) == (0, None, True)
    state.tick(109.999)
    assert state.photo["offer_open"]
    pull(state, 110)
    assert not state.photo["offer_open"]
    assert state.photo["attempt_id"] == 0
    assert state.photo["image"] is None
    assert events[-1] == ("control_rejected", {"control": "photo_pull", "reason": "offer_closed"})


def test_pull_before_deadline_opens_native_base_image_only(preview):
    state, snapshots, events = preview
    state.offer(100, 123000)
    assert pull(state, 109.999) is None
    photo = snapshots[-1]["photo"]
    assert photo["phase"] == "completed"
    assert photo["attempt_id"] == 1
    assert photo["image"] == {"path": str(state.image), "width": 1920, "height": 1080}
    assert photo["visible"] and not photo["offer_open"]
    assert photo["blocks"] == photo["preparations"] == []
    assert photo["first_block_drawn"] is False
    assert snapshots[-2]["photo"]["image"] is None


def test_latest_only_replacement_rejects_stale_pull_and_hide(preview):
    state, snapshots, events = preview
    state.offer(100, 123000)
    old_id = state.photo["photo_id"]
    pull(state, 101)
    state.offer(102, 125000)
    current = dict(state.photo)
    pull(state, 103, "late", old_id)
    state.control({"type": "photo_hidden", "photo_id": old_id}, 1, 103)
    pull(state, 103, "old-generation", generation=2)
    assert state.photo == current
    assert state.photo["photo_id"] != old_id
    assert state.photo["attempt_id"] == 0 and state.photo["image"] is None
    pull(state, 104)
    assert state.photo["attempt_id"] == 1
    assert state.photo["visible"]


def test_grab_pull_then_hidden_stays_hidden_until_retained_reopen(preview):
    state, snapshots, events = preview
    state.offer(100, 123000)
    pull(state, 101)
    state.control({"type": "photo_hidden", "photo_id": state.photo["photo_id"]}, 1, 101)
    hidden_revision = state.revision
    state.tick(112)
    assert not state.photo["visible"]
    assert snapshots[-1]["photo"]["visible"] is False
    assert state.revision == hidden_revision
    pull(state, 120, "reopen")
    assert state.photo["visible"] and state.photo["attempt_id"] == 1
    assert state.photo["first_block_drawn"] is False
    assert state.photo["blocks"] == state.photo["preparations"] == []
    assert events[-1][0] == "preview_reopened"
    state.control(
        {
            "type": "photo_first_block_drawn",
            "photo_id": state.photo["photo_id"],
            "attempt_id": 1,
            "block_id": 0,
        },
        1,
        121,
    )
    assert not state.photo["first_block_drawn"]
    assert events[-1][1]["reason"] == "preview_has_no_translation_blocks"


def test_retention_boundary_rejects_reopen_and_duplicate_pull(preview):
    state, snapshots, events = preview
    state.offer(100, 123000)
    pull(state, 101)
    revision = state.revision
    pull(state, 102)
    assert state.revision == revision
    pull(state, 699.999, "last-reopen")
    assert state.photo["visible"]
    pull(state, 700, "expired")
    assert state.photo["phase"] == "expired"
    assert not state.photo["visible"] and not state.photo["offer_open"]
    state.tick(701)
    assert events[-1][1]["reason"] == "not_current"


def test_software_pull_never_admitted_for_openvr(tmp_path):
    with pytest.raises(SystemExit) as result:
        probe.main(
            [
                "--image",
                str(tmp_path / "unused.png"),
                "--executable",
                "unused.exe",
                "--software-pull-after",
                "0",
            ]
        )
    assert result.value.code == 2


def test_png_dimensions_use_header_not_pixel_decoder(tmp_path):
    path = make_png_header(tmp_path / "reference.png", 2560, 1440)
    assert probe.png_dimensions(path) == (2560, 1440)
    path.write_bytes(b"not a png")
    with pytest.raises(ValueError):
        probe.png_dimensions(path)


@pytest.mark.asyncio
async def test_cleanup_sends_shutdown_drains_and_preserves_nonzero_exit():
    bridge = SimpleNamespace(
        send=Mock(return_value=True), close=AsyncMock(), shutdown_complete=asyncio.Event()
    )
    evidence = SimpleNamespace(summary={}, event=Mock())
    process = SimpleNamespace(returncode=None, terminate=Mock(), kill=Mock())

    async def wait():
        process.returncode = 0
        bridge.shutdown_complete.set()
        return 0

    process.wait = wait
    drained = asyncio.Event()

    async def reader():
        drained.set()

    errors = await probe.stop_child(process, bridge, [asyncio.create_task(reader())], evidence)
    assert errors == []
    assert drained.is_set()
    bridge.send.assert_called_once_with({"type": "shutdown"})
    bridge.close.assert_awaited_once()
    assert evidence.summary == {"native_exit_code": 0, "shutdown_acknowledged": True}
    process.returncode = 7
    assert await probe.stop_child(process, bridge, [], evidence) == ["native_exit_nonzero"]
    assert evidence.summary["native_exit_code"] == 7


@pytest.mark.asyncio
async def test_cleanup_reports_failure_and_still_closes_bridge():
    bridge = SimpleNamespace(
        send=Mock(side_effect=OSError), close=AsyncMock(), shutdown_complete=asyncio.Event()
    )
    evidence = SimpleNamespace(summary={}, event=Mock())
    process = SimpleNamespace(returncode=None)

    async def wait():
        process.returncode = 0
        return 0

    process.wait = wait
    assert await probe.stop_child(process, bridge, [], evidence) == ["shutdown_send_OSError"]
    bridge.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_native_terminal_error_is_not_readiness_or_completion():
    stream = asyncio.StreamReader()
    stream.feed_data(
        b'{"type":"photo_runtime_error","code":"graphics_failed","private":"discard"}\n'
    )
    stream.feed_eof()
    evidence = SimpleNamespace(event=Mock())
    signals = {"decoded": False, "composed": False, "error": None}
    await probe.drain(stream, evidence, signals)
    assert signals == {"decoded": False, "composed": False, "error": "graphics_failed"}
    evidence.event.assert_called_once_with(
        "native", type="photo_runtime_error", code="graphics_failed"
    )


@pytest.mark.asyncio
async def test_runtime_absence_does_not_start_native_or_steamvr(tmp_path):
    image = make_png_header(tmp_path / "selected.png")
    executable = tmp_path / "trusted.exe"
    executable.write_bytes(b"isolated fixture")
    args = probe.parser().parse_args(["--image", str(image), "--executable", str(executable)])
    evidence = SimpleNamespace(summary={}, event=Mock())
    with (
        patch.object(probe, "runtime_running", return_value=False),
        patch.object(probe.asyncio, "create_subprocess_exec", new_callable=AsyncMock) as spawn,
    ):
        assert await probe.run(args, evidence) == 2
    assert evidence.summary["status"] == "runtime_not_running"
    assert evidence.summary["executable_sha256"]
    spawn.assert_not_awaited()


@pytest.mark.asyncio
async def test_startup_process_exit_is_terminal_and_manifest_is_removed(tmp_path):
    image = make_png_header(tmp_path / "selected.png")
    executable = tmp_path / "trusted.exe"
    executable.write_bytes(b"isolated fixture")
    args = probe.parser().parse_args(
        ["--image", str(image), "--executable", str(executable), "--backend", "offscreen"]
    )
    evidence = SimpleNamespace(path=tmp_path, summary={}, event=Mock())
    streams = [asyncio.StreamReader(), asyncio.StreamReader()]
    for stream in streams:
        stream.feed_eof()
    process = SimpleNamespace(
        returncode=7, pid=123, wait=AsyncMock(return_value=7), stdout=streams[0], stderr=streams[1]
    )
    with patch.object(
        probe.asyncio, "create_subprocess_exec", new_callable=AsyncMock, return_value=process
    ):
        assert await probe.run(args, evidence) == 2
    assert evidence.summary["status"] == "cleanup_failed"
    assert evidence.summary["primary_status"] == "native_start_failed"
    assert evidence.summary["native_exit_code"] == 7
    assert evidence.summary["cleanup_errors"] == ["native_exit_nonzero"]
    assert not (tmp_path / "launch.json").exists()


@pytest.mark.parametrize(
    "output, expected",
    [
        (b"\xc1\xa4\xba\xb8: no matching process\r\n", False),
        (b'"vrserver.exe","123","Console","1","1,000 K"\r\n', True),
    ],
)
def test_runtime_preflight_handles_localized_tasklist(output, expected):
    with (
        patch.object(probe.os, "name", "nt"),
        patch.object(
            probe.subprocess, "run", return_value=SimpleNamespace(stdout=output)
        ) as tasklist,
    ):
        assert probe.runtime_running() is expected
    assert tasklist.call_args.args[0][0] == "tasklist"


@pytest.mark.asyncio
async def test_cleanup_escalates_unresponsive_child_and_drains():
    bridge = SimpleNamespace(
        send=Mock(return_value=True), close=AsyncMock(), shutdown_complete=asyncio.Event()
    )
    evidence = SimpleNamespace(summary={}, event=Mock())
    process = SimpleNamespace(returncode=None, terminate=Mock(), kill=Mock())
    waits = 0

    async def wait():
        nonlocal waits
        waits += 1
        if waits == 1:
            raise TimeoutError
        process.returncode = 1
        return 1

    process.wait = wait
    errors = await probe.stop_child(process, bridge, [], evidence)
    assert errors == ["native_exit_nonzero"]
    process.terminate.assert_called_once()
    process.kill.assert_not_called()
    assert not evidence.summary["shutdown_acknowledged"]
    bridge.close.assert_awaited_once()
