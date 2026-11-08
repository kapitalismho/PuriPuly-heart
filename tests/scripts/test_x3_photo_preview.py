from __future__ import annotations

import asyncio
import importlib.util
import json
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


def test_tuning_schedule_preserves_full_windows_fresh_ids_and_released_boundaries(preview):
    state, snapshots, events = preview
    commands = []
    barrier = probe.Quiescence(lambda command: commands.append(command) or True, Mock(), 1)
    schedule = probe.TuningSchedule(state, barrier, 100, 60, 2, Mock())
    assert not schedule.tick(101.99, 0)
    assert state.photo is None
    schedule.tick(102, 102000)
    first = state.photo["photo_id"]
    schedule.tick(112, 112000)
    assert not state.photo["offer_open"]
    schedule.tick(116.99, 0)
    assert commands == []
    schedule.tick(117, 117000)
    request = commands[-1]["request_id"]
    state.control({"type": "photo_hidden", "photo_id": first}, 1, 118)
    schedule.tick(130, 130000)
    assert state.photo["photo_id"] == first
    barrier.control({"request_id": "wrong"}, 1)
    barrier.control({"request_id": request}, 2)
    schedule.tick(131, 131000)
    assert state.photo["photo_id"] == first
    barrier.control({"request_id": request}, 1)
    schedule.tick(132, 132000)
    second = state.photo["photo_id"]
    assert second != first
    assert state.photo["shutter_unix_ms"] == 132000
    assert state.photo["offer_open"]
    schedule.tick(147, 147000)
    barrier.control({"request_id": commands[-1]["request_id"]}, 1)
    schedule.tick(150.001, 0)
    assert state.photo["photo_id"] == second
    assert schedule.tick(160, 160000)
    assert schedule.tick(190, 190000)
    assert state.photo["photo_id"] == second
    offers = [snapshot["photo"] for snapshot in snapshots if snapshot["photo"] and snapshot["photo"]["offer_open"]]
    assert {photo["photo_id"] for photo in offers} == {first, second}


@pytest.mark.parametrize("first_poll", [102.001, 120.0])
def test_late_first_tuning_poll_guarantees_full_opportunity_before_boundary(preview, first_poll):
    state, snapshots, events = preview
    commands = []
    barrier = probe.Quiescence(lambda command: commands.append(command) or True, Mock(), 1)
    schedule = probe.TuningSchedule(state, barrier, 100, 12, 2, Mock())
    assert not schedule.tick(first_poll, int(first_poll * 1000))
    assert state.photo is not None
    photo_id = state.photo["photo_id"]
    assert state.photo["offer_open"]
    assert state.shutter == first_poll
    assert schedule.end >= first_poll + 10
    assert commands == []
    assert not schedule.tick(first_poll + 9.999, 0)
    assert state.photo["offer_open"]
    assert commands == []
    assert not schedule.tick(first_poll + 10, 0)
    assert not state.photo["offer_open"]
    assert state.photo["photo_id"] == photo_id
    assert len(commands) == 1
    assert commands[0]["type"] == "photo_debug_quiesce"
    barrier.control({"request_id": commands[0]["request_id"]}, 1)
    assert schedule.tick(first_poll + 11, 0)


@pytest.mark.asyncio
async def test_tuning_cleanup_requires_shutdown_ack_even_with_zero_exit():
    bridge = SimpleNamespace(
        send=Mock(return_value=True), close=AsyncMock(), shutdown_complete=asyncio.Event()
    )
    process = SimpleNamespace(returncode=0)
    evidence = SimpleNamespace(summary={}, event=Mock())
    assert await probe.stop_child(process, bridge, [], evidence, require_ack=True) == [
        "shutdown_ack_missing"
    ]


@pytest.fixture
def tuning_launch(tmp_path, monkeypatch):
    from puripuly_heart.core.photo.bridge import PhotoBridge

    clock = SimpleNamespace(now=0.0)
    harness = SimpleNamespace(
        manifests=[], processes=[], bridges=[], events=[], active=None, pending=[],
        ack_delay=0, cleanup_delay=0, fail_at=None, cancel_at=None, shutdown_ack=True,
    )
    real_sleep = asyncio.sleep

    class Bridge(PhotoBridge):
        async def start(self):
            pass

        async def close(self):
            await sleep(harness.cleanup_delay)
            assert harness.active.returncode is not None
            harness.events.append(("closed", clock.now))
            harness.active = None

        def send(self, command):
            harness.events.append((command["type"], clock.now))
            if command["type"] == "photo_debug_quiesce":
                harness.pending.append((clock.now + harness.ack_delay, self, command))
            elif command["type"] == "shutdown":
                if harness.shutdown_ack:
                    self.shutdown_complete.set()
                harness.active.returncode = 0
                harness.active.exited.set()
            elif command["type"] == "photo_probe_pull":
                self._control({**command, "type": "photo_pull"}, self._generation)
            elif command["type"] == "photo_probe_hide":
                self._control({**command, "type": "photo_hidden"}, self._generation)
            elif command["type"] == "photo_debug_export":
                export_dir = Path(harness.manifests[-1]["image"]["export_dir"])
                make_png_header(export_dir / f"{command['label']}.png")
                (export_dir / f"{command['label']}.json").write_text(
                    json.dumps({"blocks": [], "original_uploads": 1}), encoding="utf-8"
                )
            return True

    def new_bridge(**kwargs):
        bridge = Bridge(**kwargs)
        harness.bridges.append(bridge)
        return bridge

    async def sleep(seconds):
        clock.now += seconds or 0.01
        if harness.cancel_at is not None and clock.now >= harness.cancel_at:
            harness.cancel_at = None
            raise asyncio.CancelledError
        if harness.fail_at is not None and clock.now >= harness.fail_at:
            harness.fail_at = None
            harness.active.returncode = 7
            harness.active.exited.set()
        for due, bridge, command in tuple(harness.pending):
            if clock.now >= due:
                harness.pending.remove((due, bridge, command))
                bridge._control(
                    {"type": "photo_debug_quiescent", "request_id": command["request_id"]},
                    bridge._generation,
                )
                harness.events.append(("quiescent", clock.now))
        await real_sleep(0)

    async def spawn(*command, **kwargs):
        assert harness.active is None
        manifest = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        harness.manifests.append(manifest)
        harness.events.append(("spawn", clock.now))
        process = SimpleNamespace(
            pid=100 + len(harness.manifests), returncode=None, exited=asyncio.Event(),
            stdout=asyncio.StreamReader(), stderr=asyncio.StreamReader(),
        )
        process.stdout.feed_data(
            b'{"stage":"display_source_ready"}\n'
            b'{"stage":"composed","original_uploads":1}\n'
        )
        process.stdout.feed_eof()
        process.stderr.feed_eof()

        async def wait():
            await process.exited.wait()
            return process.returncode

        process.wait = wait
        harness.active = process
        harness.processes.append(process)
        harness.bridges[-1].ready.set()
        return process

    image = make_png_header(tmp_path / "selected.png")
    executable = tmp_path / "trusted.exe"
    executable.write_bytes(b"synthetic executable, never run")
    harness.args = probe.parser().parse_args(
        ["--image", str(image), "--executable", str(executable), "--backend", "offscreen",
         "--tune-wrist-and-grab", "--seconds", "12", "--offer-delay", "1"]
    )
    monkeypatch.setattr(
        "puripuly_heart.core.photo.bridge.PhotoBridge", new_bridge
    )
    monkeypatch.setattr(probe, "time", SimpleNamespace(monotonic=lambda: clock.now, time=lambda: clock.now))
    monkeypatch.setattr(
        probe, "asyncio",
        SimpleNamespace(**{name: getattr(asyncio, name) for name in dir(asyncio) if name != "sleep"}, sleep=sleep),
    )
    monkeypatch.setattr(probe.asyncio, "create_subprocess_exec", spawn)
    harness.evidence = probe.Evidence(tmp_path / "evidence")
    yield harness
    harness.evidence.finish()


@pytest.mark.asyncio
async def test_one_launch_runs_two_fixed_factor_phases_and_waits_for_boundary_and_cleanup(tuning_launch):
    harness = tuning_launch
    harness.ack_delay = 4
    harness.cleanup_delay = 2
    assert await probe.run_tuning(harness.args, harness.evidence) == 0
    manifests = harness.manifests
    assert len(manifests) == 6
    assert [item["generation"] for item in manifests] == list(range(1, 7))
    assert len({item["instance_id"] for item in manifests}) == 6
    assert len({item["session_token"] for item in manifests}) == 6
    interaction = [item["interaction"] for item in manifests]
    assert [item["wrist_handgrip_point_m"] for item in interaction[:3]] == [
        [0, 0.04373328, 0.01863404], [0, 0.063, 0.024], [0, 0.08226672, 0.02936596]
    ]
    assert [item["grab_m"] for item in interaction[:3]] == [0.08] * 3
    assert [item["grab_m"] for item in interaction[3:]] == [0.08, 0.12, 0.15]
    assert all(item["wrist_handgrip_point_m"] == [0, 0.063, 0.024] for item in interaction[3:])
    assert [item["debug_grab_volume"] for item in interaction] == [False] * 3 + [True] * 3
    assert all(item["image"]["debug_controls"] for item in manifests)
    assert all(
        item["debug_case_label"].isascii()
        and len(item["debug_case_label"]) <= 128
        and len(item["debug_case_label"].splitlines()) <= 2
        for item in interaction
    )
    assert [kind for kind, _ in harness.events] == [
        kind for _ in range(6)
        for kind in ("spawn", "photo_debug_quiesce", "quiescent", "shutdown", "closed")
    ]
    assert all(case["elapsed_s"] >= 18 for case in harness.evidence.summary["cases"])
    assert harness.evidence.summary["status"] == "experiment_completed"
    assert not list(harness.evidence.path.rglob("launch.json"))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["native", "missing_ack", "cancel"])
async def test_tuning_failure_or_interruption_never_starts_next_child(tuning_launch, failure):
    harness = tuning_launch
    if failure == "native":
        harness.fail_at = 3
    elif failure == "missing_ack":
        harness.shutdown_ack = False
    else:
        harness.cancel_at = 3
    if failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await probe.run_tuning(harness.args, harness.evidence)
        assert harness.evidence.summary["status"] == "interrupted"
    else:
        assert await probe.run_tuning(harness.args, harness.evidence) == 2
        assert harness.evidence.summary["status"] == "experiment_aborted"
    assert len(harness.manifests) == 1
    assert harness.active is None
    assert not list(harness.evidence.path.rglob("launch.json"))


def test_tuning_rejects_live_watch_directory_before_launch(tmp_path):
    with pytest.raises(SystemExit) as result:
        probe.main([
            "--image", "unused.png", "--executable", "unused.exe",
            "--tune-wrist-and-grab", "--watch-dir", str(tmp_path),
        ])
    assert result.value.code == 2


@pytest.mark.asyncio
async def test_tuning_repeats_synthetic_offers_with_full_windows_in_real_runner(tuning_launch):
    harness = tuning_launch
    harness.args.seconds = 60
    harness.args.offer_delay = 2
    assert await probe.run_tuning(harness.args, harness.evidence) == 0
    for case in harness.evidence.summary["cases"]:
        records = [
            json.loads(line)
            for line in (harness.evidence.path / case["evidence"] / "events.jsonl")
            .read_text(encoding="utf-8").splitlines()
        ]
        offers = [record for record in records if record["event"] == "synthetic_offer"]
        assert len(offers) == 4
        assert len({offer["photo_id"] for offer in offers}) == 4
        assert all(offer["target_remaining_s"] >= 10 for offer in offers)
        assert all(
            later["elapsed_s"] - earlier["elapsed_s"] >= 15
            for earlier, later in zip(offers, offers[1:])
        )
        acknowledgements = [
            record for record in records if record["event"] == "boundary_acknowledged"
        ]
        assert len(acknowledgements) == 4
        assert all(
            ack["elapsed_s"] <= offer["elapsed_s"]
            for ack, offer in zip(acknowledgements, offers[1:])
        )


@pytest.mark.asyncio
async def test_offscreen_tuning_smoke_finishes_early_but_waits_for_quiescence(tuning_launch):
    harness = tuning_launch
    harness.args.seconds = 60
    harness.args.software_pull_after = 0.2
    harness.ack_delay = 2
    assert await probe.run_tuning(harness.args, harness.evidence) == 0
    assert len(harness.manifests) == 6
    assert harness.evidence.summary["offscreen_verification"] is True
    for case in harness.evidence.summary["cases"]:
        assert case["status"] == "offscreen_preview_exported"
        assert case["elapsed_s"] < 60
        path = harness.evidence.path / case["evidence"]
        assert (path / "preview.png").is_file()
        assert (path / "reopened.png").is_file()
    kinds = [kind for kind, _ in harness.events]
    assert kinds.count("photo_probe_pull") == 12
    assert kinds.count("photo_debug_export") == 12
    assert kinds.count("quiescent") == 6
    assert all(
        kinds[index - 1] == "quiescent"
        for index, kind in enumerate(kinds) if kind == "shutdown"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_failure_during_unacknowledged_boundary_aborts_without_next_child(tuning_launch, cancel):
    harness = tuning_launch
    harness.ack_delay = 60
    if cancel:
        harness.cancel_at = 14
        with pytest.raises(asyncio.CancelledError):
            await probe.run_tuning(harness.args, harness.evidence)
    else:
        harness.fail_at = 14
        assert await probe.run_tuning(harness.args, harness.evidence) == 2
    assert len(harness.manifests) == 1
    kinds = [kind for kind, _ in harness.events]
    assert kinds.count("photo_debug_quiesce") == 1
    assert "quiescent" not in kinds
    assert harness.active is None
