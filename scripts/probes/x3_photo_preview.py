from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4


class Quiescence:
    def __init__(self, send, event, generation):
        self.send = send
        self.event = event
        self.generation = generation
        self.request_id = None
        self.acknowledged = asyncio.Event()

    def request(self, reason):
        if self.request_id is not None:
            return
        self.request_id = str(uuid4())
        self.acknowledged.clear()
        if not self.send({"type": "photo_debug_quiesce", "request_id": self.request_id}):
            raise RuntimeError("quiescence request rejected")
        self.event("boundary_requested", request_id=self.request_id, reason=reason)

    def control(self, value, generation):
        if (
            generation != self.generation
            or self.request_id is None
            or value["request_id"] != self.request_id
        ):
            self.event("boundary_ack_rejected", request_id=value["request_id"], generation=generation)
            return
        if not self.acknowledged.is_set():
            self.acknowledged.set()
            self.event("boundary_acknowledged", request_id=self.request_id, generation=generation)

    def reset(self):
        self.request_id = None
        self.acknowledged.clear()


class TuningSchedule:
    def __init__(self, state, barrier, started, seconds, delay, event, *, repeat=True):
        self.state = state
        self.barrier = barrier
        self.end = started + seconds
        self.first = started + delay
        self.event = event
        self.repeat = repeat
        self.offered_at = None
        self.early_finish = False

    def finish_early(self):
        self.early_finish = True

    def tick(self, now, wall_ms):
        self.state.tick(now)
        if self.early_finish or now >= self.end:
            self.barrier.request("case_end")
            return self.barrier.acknowledged.is_set()
        if now + 10 > self.end:
            return False
        if self.barrier.request_id is not None:
            if not self.barrier.acknowledged.is_set():
                return False
            self.barrier.reset()
        elif self.offered_at is not None:
            if not self.repeat or now - self.offered_at < 15:
                return False
            self.barrier.request("next_offer")
            return False
        elif now < self.first:
            return False
        self.state.offer(now, wall_ms)
        self.offered_at = now
        self.event(
            "synthetic_offer",
            photo_id=self.state.photo["photo_id"],
            offer_window_s=10,
            retention_s=600,
            target_remaining_s=self.end - now,
        )
        return False


def tuning_cases():
    points = (
        ("A", "-2 cm", [0, 0.04373328, 0.01863404]),
        ("B", "current", [0, 0.063, 0.024]),
        ("C", "+2 cm", [0, 0.08226672, 0.02936596]),
    )
    cases = []
    for index, (name, direction, point) in enumerate(points, 1):
        cases.append(
            {
                "phase": "position",
                "case": name,
                "interaction": {
                    "wrist_handgrip_point_m": point,
                    "grab_m": 0.08,
                    "debug_grab_volume": False,
                    "debug_case_label": (
                        f"POSITION {index}/3: {name} ({direction})\nRadius fixed: 8 cm"
                    ),
                },
            }
        )
    for index, radius in enumerate((0.08, 0.12, 0.15), 1):
        cases.append(
            {
                "phase": "radius",
                "case": f"{round(radius * 100):02d}cm",
                "interaction": {
                    "wrist_handgrip_point_m": points[1][2],
                    "grab_m": radius,
                    "debug_grab_volume": True,
                    "debug_case_label": (
                        f"RADIUS {index}/3: {round(radius * 100)} cm\nPosition fixed: B"
                    ),
                },
            }
        )
    return cases


async def run_tuning(args, evidence):
    cases = tuning_cases()
    evidence.summary.update(
        mode="wrist_and_grab_two_phases",
        target_seconds_per_case=args.seconds,
        offscreen_verification=args.software_pull_after is not None,
        cases=[],
    )
    evidence.event(
        "experiment_plan",
        phases=[
            "Position: A handward heuristic, B current, C forearmward heuristic; radius 0.08m; volume OFF",
            "Radius: 0.08m, 0.12m, 0.15m; position B; wireframe volume ON",
        ],
        heuristic_not_anatomical=True,
        target_seconds_per_case=args.seconds,
        release_wait_may_extend=True,
    )
    try:
        for generation, case in enumerate(cases, 1):
            child = Evidence(
                evidence.path / f"{generation:02d}-{case['phase']}-{case['case']}",
                experiment_phase=case["phase"],
                case=case["case"],
            )
            try:
                result = await run(args, child, case=case, generation=generation)
            except asyncio.CancelledError:
                child.summary["status"] = "interrupted"
                raise
            except Exception as exc:
                child.summary["status"] = "launcher_error"
                child.event("launcher_error", error_type=type(exc).__name__)
                result = 2
            finally:
                child.finish()
                evidence.summary["cases"].append(
                    {"evidence": child.path.name, **child.summary}
                )
            clean = (
                result == 0
                and child.summary.get("native_exit_code") == 0
                and child.summary.get("shutdown_acknowledged") is True
                and child.summary.get("cleanup_errors") == []
                and child.summary.get("native_error") is None
            )
            evidence.event("case_outcome", phase=case["phase"], case=case["case"], clean=clean)
            if not clean:
                evidence.summary["status"] = "experiment_aborted"
                return 2
        evidence.summary["status"] = "experiment_completed"
        return 0
    except asyncio.CancelledError:
        evidence.summary["status"] = "interrupted"
        evidence.event("experiment_interrupted")
        raise


class PreviewState:
    def __init__(self, image: Path, width: int, height: int, publish, event, generation=1):
        self.image = image
        self.width = width
        self.height = height
        self.publish = publish
        self.event = event
        self.revision = 0
        self.photo = None
        self.shutter = 0.0
        self.pulls = set()
        self.generation = generation

    def snapshot(self):
        self.publish(
            {
                "type": "photo_snapshot",
                "revision": self.revision,
                "photo": None if self.photo is None else dict(self.photo),
            }
        )

    def changed(self, kind):
        self.revision += 1
        self.snapshot()
        self.event(
            kind,
            photo_id=self.photo["photo_id"],
            revision=self.revision,
            attempt_id=self.photo["attempt_id"],
            phase=self.photo["phase"],
            visible=self.photo["visible"],
        )

    def offer(self, now, wall_ms):
        self.shutter = now
        self.pulls.clear()
        self.photo = {
            "photo_id": str(uuid4()),
            "path": str(self.image),
            "file_size": self.image.stat().st_size,
            "shutter_unix_ms": wall_ms,
            "attempt_id": 0,
            "phase": "offer",
            "offer_open": True,
            "visible": False,
            "image": None,
            "preparations": [],
            "blocks": [],
            "failure_code": None,
            "first_block_drawn": False,
        }
        self.changed("preview_offer")
        self.tick(now)

    def tick(self, now):
        photo = self.photo
        if photo is None or photo["phase"] == "expired":
            return
        if now - self.shutter >= 600:
            photo.update(phase="expired", offer_open=False, visible=False)
            self.changed("retention_expired")
        elif now - self.shutter >= 10 and photo["offer_open"]:
            photo["offer_open"] = False
            self.changed("offer_closed")

    def control(self, value, generation, now):
        self.tick(now)
        kind = value["type"]
        photo = self.photo
        if (
            generation != self.generation
            or photo is None
            or value.get("photo_id") != photo["photo_id"]
            or photo["phase"] == "expired"
        ):
            self.event("control_rejected", control=kind, reason="not_current")
            return
        if kind == "photo_pull":
            if value["event_id"] in self.pulls:
                self.event("control_rejected", control=kind, reason="duplicate_pull")
                return
            if photo["attempt_id"] == 0 and not photo["offer_open"]:
                self.event("control_rejected", control=kind, reason="offer_closed")
                return
            self.pulls.add(value["event_id"])
            reopened = photo["attempt_id"] == 1
            photo.update(
                attempt_id=1,
                phase="completed",
                offer_open=False,
                visible=True,
                image={"path": str(self.image), "width": self.width, "height": self.height},
            )
            self.changed("preview_reopened" if reopened else "preview_opened")
        elif kind == "photo_hidden" and photo["visible"]:
            photo["visible"] = False
            self.changed("preview_hidden")
        else:
            self.event("control_rejected", control=kind, reason="preview_has_no_translation_blocks")


def png_dimensions(path):
    with path.open("rb") as stream:
        header = stream.read(33)
    if len(header) != 33 or header[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR":
        raise ValueError("image must have a PNG IHDR header")
    width, height = struct.unpack(">II", header[16:24])
    if not width or not height or max(width, height) > 0x7FFFFFFF:
        raise ValueError("invalid PNG dimensions")
    return width, height


class Evidence:
    def __init__(self, path=None, **tags):
        self.path = Path(tempfile.mkdtemp(prefix="puripuly-x3-preview-")) if path is None else path
        self.path.mkdir(parents=True, exist_ok=True)
        self.tags = tags
        self.stream = (self.path / "events.jsonl").open("x", encoding="utf-8")
        self.started = time.monotonic()
        self.summary = {
            "status": "starting",
            "preview_only": True,
            "provider_requests": 0,
            "translation_blocks": 0,
            "physical_observations": "unperformed",
        }
        self.summary.update(tags)

    def event(self, kind, **fields):
        record = {
            "event": kind,
            "elapsed_s": time.monotonic() - self.started,
            **self.tags,
            **fields,
        }
        self.stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.stream.flush()
        print(json.dumps(record, allow_nan=False), flush=True)

    def finish(self):
        self.summary["elapsed_s"] = time.monotonic() - self.started
        (self.path / "summary.json").write_text(
            json.dumps(self.summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        self.stream.close()


def runtime_running():
    if os.name != "nt":
        return False
    result = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq vrserver.exe", "/FO", "CSV", "/NH"],
        capture_output=True,
        check=True,
        timeout=5,
    )
    return any(
        row and row[0].lower() == "vrserver.exe"
        for row in csv.reader(result.stdout.decode("ascii", errors="replace").splitlines())
    )


async def drain(stream, evidence, signals):
    while raw := await stream.readline():
        try:
            value = json.loads(raw)
        except ValueError, UnicodeDecodeError:
            evidence.event("native_non_json_output", bytes=len(raw))
            continue
        if not isinstance(value, dict):
            continue
        fields = {
            key: value[key]
            for key in (
                "type",
                "stage",
                "code",
                "pid",
                "backend",
                "photo_id",
                "attempt_id",
                "block_count",
                "original_uploads",
                "submitted",
                "recovered",
                "elapsed_ms",
            )
            if key in value
        }
        evidence.event("native", **fields)
        if value.get("type") in ("photo_runtime_error", "photo_image_failed"):
            signals["error"] = value.get("code", "native_error")
        if value.get("stage") == "display_source_ready":
            signals["decoded"] = True
        if value.get("stage") == "composed" and value.get("original_uploads", 0) > 0:
            signals["composed"] = True


async def stop_child(process, bridge, readers, evidence, *, require_ack=False):
    errors = []
    if process is not None:
        try:
            if process.returncode is None:
                try:
                    bridge.send({"type": "shutdown"})
                except Exception as exc:
                    errors.append("shutdown_send_" + type(exc).__name__)
                try:
                    await asyncio.wait_for(process.wait(), 3)
                except TimeoutError:
                    if require_ack:
                        errors.append("shutdown_timeout")
                    evidence.event("shutdown_escalated", action="terminate")
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), 2)
                    except TimeoutError:
                        evidence.event("shutdown_escalated", action="kill")
                        process.kill()
                        await asyncio.wait_for(process.wait(), 2)
            evidence.summary["native_exit_code"] = process.returncode
            evidence.summary["shutdown_acknowledged"] = bridge.shutdown_complete.is_set()
            if process.returncode != 0:
                errors.append("native_exit_nonzero")
        except Exception as exc:
            errors.append("child_cleanup_" + type(exc).__name__)
            if process.returncode is None:
                try:
                    process.kill()
                    await asyncio.wait_for(process.wait(), 2)
                except Exception as cleanup_exc:
                    errors.append("child_kill_" + type(cleanup_exc).__name__)
    if readers:
        try:
            results = await asyncio.wait_for(asyncio.gather(*readers, return_exceptions=True), 2)
            errors.extend(
                "reader_" + type(item).__name__
                for item in results
                if isinstance(item, BaseException)
            )
        except TimeoutError:
            errors.append("reader_drain_timeout")
    if process is not None and require_ack and not bridge.shutdown_complete.is_set():
        if process.returncode == 0:
            try:
                await asyncio.wait_for(bridge.shutdown_complete.wait(), 1)
            except TimeoutError:
                pass
        if not bridge.shutdown_complete.is_set():
            errors.append("shutdown_ack_missing")
    evidence.summary["shutdown_acknowledged"] = bridge.shutdown_complete.is_set()
    try:
        await asyncio.wait_for(bridge.close(), 3)
    except Exception as exc:
        errors.append("bridge_cleanup_" + type(exc).__name__)
    return errors


async def run(args, evidence, *, case=None, generation=1):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from puripuly_heart.core.photo.bridge import PhotoBridge
    from puripuly_heart.core.photo.capture import PhotoCaptureWatcher

    image = args.image.expanduser().resolve(strict=True)
    executable = args.executable.expanduser().resolve(strict=True)
    width, height = png_dimensions(image)
    control_mode = "probe" if args.backend == "offscreen" else "openvr"
    with executable.open("rb") as stream:
        executable_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    evidence.summary.update(
        requested_native_backend=args.backend,
        bridge_control_mode=control_mode,
        executable=str(executable),
        executable_sha256=executable_hash,
        image_dimensions=[width, height],
        max_size=[2560, 1440] if args.reference_1440 else [1920, 1080],
        source_script=str(Path(__file__).resolve()),
        python_executable=sys.executable,
        offer_source=(
            "new_file_signal_same_selected_asset" if args.watch_dir else "delayed_synthetic"
        ),
        bridge_source=str(Path(sys.modules[PhotoBridge.__module__].__file__).resolve()),
    )
    evidence.event(
        "configuration",
        requested_native_backend=args.backend,
        bridge_control_mode=control_mode,
        image_dimensions=[width, height],
        max_size=evidence.summary["max_size"],
    )
    if args.backend == "openvr" and not await asyncio.to_thread(runtime_running):
        evidence.summary["status"] = "runtime_not_running"
        evidence.event("runtime_not_running", native_started=False, steamvr_launch_attempted=False)
        return 2
    barrier = None

    def control(value, owner):
        if value["type"] == "photo_debug_quiescent":
            if barrier is not None:
                barrier.control(value, owner)
            return
        state.control(value, owner, time.monotonic())

    bridge = PhotoBridge(control=control)
    state = PreviewState(image, width, height, bridge.publish, evidence.event, generation)
    if case is not None:
        barrier = Quiescence(bridge.send, evidence.event, generation)
    state.snapshot()
    process = None
    readers = []
    ready_task = None
    exit_task = None
    manifest_path = evidence.path / "launch.json"
    signals = {"decoded": False, "composed": False, "error": None}
    result = 1
    try:
        await bridge.start()
        manifest = await bridge.activate(generation, backend=control_mode)
        manifest["backend"] = args.backend
        manifest["image"] = {
            "max_size": evidence.summary["max_size"],
            "debug_controls": True,
            "export_dir": str(evidence.path),
        }
        if case is not None:
            manifest["interaction"] = case["interaction"]
            evidence.summary["requested_interaction"] = case["interaction"]
            evidence.event(
                "case_parameters",
                interaction=case["interaction"],
                configuration_source="immutable_launch_manifest",
                measured_world_pose=False,
                generation=generation,
                instance_id=manifest["instance_id"],
                target_seconds=args.seconds,
                offscreen_verification=args.software_pull_after is not None,
            )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        if os.name != "nt":
            manifest_path.chmod(0o600)
        watcher = None
        if args.watch_dir:
            watcher = PhotoCaptureWatcher(
                args.watch_dir.expanduser().resolve(strict=True),
                evidence.path / "empty-vrchat-logs",
            )
            await watcher.start()
            evidence.event(
                "watch_baselined", old_files_ignored=True, capture_is_not_translated=True
            )
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        process = await asyncio.create_subprocess_exec(
            str(executable),
            "--role",
            "photo",
            "--config",
            str(manifest_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **kwargs,
        )
        evidence.event("native_started", pid=process.pid)
        readers = [
            asyncio.create_task(drain(stream, evidence, signals))
            for stream in (process.stdout, process.stderr)
        ]
        exit_task = asyncio.create_task(process.wait())
        ready_task = asyncio.create_task(bridge.ready.wait())
        done, _ = await asyncio.wait(
            (exit_task, ready_task),
            timeout=args.startup_timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if exit_task in done or ready_task not in done:
            evidence.summary["status"] = "native_start_failed"
            return 2
        evidence.event("native_ready", handshake=True, preview_decoded=False)
        started = time.monotonic()
        schedule = (
            TuningSchedule(
                state, barrier, started, args.seconds, args.offer_delay, evidence.event,
                repeat=args.software_pull_after is None,
            )
            if case is not None else None
        )
        offered_at = None
        smoke_stage = 0
        exported_at = None
        last_receipt = -1
        while schedule is not None or time.monotonic() - started < args.seconds:
            now = time.monotonic()
            if exit_task.done() or bridge.disconnected.is_set() or signals["error"]:
                evidence.summary["status"] = "native_terminal_error"
                return 2
            if schedule is not None:
                if schedule.tick(now, int(time.time() * 1000)):
                    smoke_complete = args.software_pull_after is None or schedule.early_finish
                    evidence.summary["status"] = (
                        "offscreen_preview_exported" if schedule.early_finish
                        else "duration_completed" if smoke_complete else "smoke_incomplete"
                    )
                    result = 0 if smoke_complete else 2
                    break
                offered_at = schedule.offered_at
            if watcher:
                capture = await watcher.poll()
                if capture is not None:
                    state.offer(capture.observed_monotonic, capture.shutter_unix_ms)
                    offered_at = now
                    evidence.event(
                        "new_file_signal", selected_asset_unchanged=True, captured_pixels_read=False
                    )
            elif schedule is None and offered_at is None and now - started >= args.offer_delay:
                state.offer(now, int(time.time() * 1000))
                offered_at = now
            state.tick(now)
            for receipt in tuple(bridge.receipts):
                if receipt["revision"] > last_receipt:
                    evidence.event(
                        "state_received",
                        **{
                            key: receipt[key]
                            for key in ("photo_id", "attempt_id", "revision", "block_count")
                        },
                    )
                    last_receipt = receipt["revision"]
            if args.software_pull_after is not None and offered_at is not None:
                photo = state.photo
                if smoke_stage == 0 and now - offered_at >= args.software_pull_after:
                    if not bridge.send(
                        {
                            "type": "photo_probe_pull",
                            "photo_id": photo["photo_id"],
                            "event_id": str(uuid4()),
                        }
                    ):
                        raise RuntimeError("software pull rejected")
                    evidence.event("offscreen_software_pull", physical_gesture=False)
                    smoke_stage = 1
                elif (
                    smoke_stage in (1, 4)
                    and photo["visible"]
                    and signals["decoded"]
                    and signals["composed"]
                ):
                    label = "preview" if smoke_stage == 1 else "reopened"
                    if not bridge.send(
                        {
                            "type": "photo_debug_export",
                            "photo_id": photo["photo_id"],
                            "label": label,
                        }
                    ):
                        raise RuntimeError("export rejected")
                    smoke_stage += 1
                    exported_at = now
                elif smoke_stage in (2, 5):
                    label = "preview" if smoke_stage == 2 else "reopened"
                    png = evidence.path / f"{label}.png"
                    metadata = evidence.path / f"{label}.json"
                    if png.exists() and metadata.exists():
                        dimensions = png_dimensions(png)
                        data = json.loads(metadata.read_text(encoding="utf-8"))
                        scale = min(
                            evidence.summary["max_size"][0] / width,
                            evidence.summary["max_size"][1] / height,
                        )
                        expected = [max(1, round(width * scale)), max(1, round(height * scale))]
                        if (
                            list(dimensions) != expected
                            or data["blocks"]
                            or data["original_uploads"] != 1
                        ):
                            raise RuntimeError("preview export contract mismatch")
                        evidence.event(
                            "offscreen_exported",
                            label=label,
                            dimensions=dimensions,
                            translation_blocks=0,
                            original_uploads=data["original_uploads"],
                            physical_display=False,
                        )
                        if smoke_stage == 5:
                            if schedule is None:
                                evidence.summary["status"] = "offscreen_preview_exported"
                                result = 0
                                break
                            if not bridge.send(
                                {"type": "photo_probe_hide", "photo_id": photo["photo_id"]}
                            ):
                                raise RuntimeError("final hide rejected")
                            schedule.finish_early()
                            smoke_stage = 6
                        else:
                            if not bridge.send(
                                {"type": "photo_probe_hide", "photo_id": photo["photo_id"]}
                            ):
                                raise RuntimeError("hide rejected")
                            smoke_stage = 3
                    elif now - exported_at > 5:
                        raise RuntimeError("preview export timeout")
                elif smoke_stage == 3 and not photo["visible"]:
                    if not bridge.send(
                        {
                            "type": "photo_probe_pull",
                            "photo_id": photo["photo_id"],
                            "event_id": str(uuid4()),
                        }
                    ):
                        raise RuntimeError("reopen rejected")
                    smoke_stage = 4
            await asyncio.sleep(0.05)
        else:
            evidence.summary["status"] = (
                "smoke_incomplete" if args.software_pull_after is not None else "duration_completed"
            )
            result = 2 if args.software_pull_after is not None else 0
    except asyncio.CancelledError:
        evidence.summary["status"] = "interrupted"
        raise
    except Exception as exc:
        evidence.summary["status"] = "launcher_error"
        evidence.event("launcher_error", error_type=type(exc).__name__)
    finally:
        errors = await stop_child(
            process, bridge, readers, evidence, require_ack=case is not None
        )
        for task in (ready_task, exit_task):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(task for task in (ready_task, exit_task) if task is not None), return_exceptions=True
        )
        manifest_path.unlink(missing_ok=True)
        evidence.summary["cleanup_errors"] = errors
        evidence.event(
            "case_cleanup" if case is not None else "cleanup",
            errors=errors,
            native_exit_code=evidence.summary.get("native_exit_code"),
            shutdown_acknowledged=evidence.summary.get("shutdown_acknowledged", False),
        )
        if errors:
            evidence.event("cleanup_failed", errors=errors)
            evidence.summary["primary_status"] = evidence.summary["status"]
            evidence.summary["status"] = "cleanup_failed"
            result = 2
        evidence.summary.update(
            source_decoded=signals["decoded"],
            composed=signals["composed"],
            native_error=signals["error"],
        )
    return result


def parser():
    command = argparse.ArgumentParser(
        description="Temporary NO-provider X3 native base-image preview. No translation or fake blocks. "
        "OpenVR shares VRChat input; start SteamVR yourself before running. "
        "A separate --image invocation is needed to view an original asset."
    )
    command.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Explicit recorded PNG asset; never decoded in Python",
    )
    command.add_argument(
        "--executable",
        type=Path,
        required=True,
        help="Trusted built overlay executable beside its DLL/fonts",
    )
    command.add_argument("--backend", choices=("openvr", "offscreen"), default="openvr")
    command.add_argument(
        "--reference-1440",
        action="store_true",
        help="Use 2560x1440 bounds instead of native 1920x1080 default",
    )
    command.add_argument(
        "--watch-dir",
        type=Path,
        help="Baseline old files; new PNG signals offer the SAME --image, never translate the capture",
    )
    command.add_argument(
        "--offer-delay",
        type=float,
        default=2,
        help="Seconds after native ready for the first synthetic offer without --watch-dir",
    )
    command.add_argument(
        "--seconds", type=float, default=60,
        help="Duration after native ready; tuning target PER CASE (default 60s), extended for release",
    )
    command.add_argument("--startup-timeout", type=float, default=10)
    command.add_argument(
        "--tune-wrist-and-grab",
        action="store_true",
        help="Six serial cases: positions A/B/C at 8cm, then radii 8/12/15cm at B; --seconds per case",
    )
    command.add_argument(
        "--software-pull-after",
        type=float,
        help="OFFSCREEN ONLY: seconds after synthetic offer; export, hide, reopen, export, then exit",
    )
    return command


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    for name in ("offer_delay", "seconds", "startup_timeout", "software_pull_after"):
        value = getattr(args, name)
        if value is not None and (
            not math.isfinite(value)
            or value < 0
            or name in ("seconds", "startup_timeout")
            and value == 0
        ):
            command.error(
                f"--{name.replace('_', '-')} must be finite and positive (delays may be zero)"
            )
    if args.software_pull_after is not None and (
        args.backend != "offscreen" or args.watch_dir or args.software_pull_after >= 10
    ):
        command.error(
            "--software-pull-after requires --backend offscreen, no --watch-dir, and delay < 10s"
        )
    if args.tune_wrist_and_grab and args.watch_dir:
        command.error("--tune-wrist-and-grab uses synthetic --image offers; no --watch-dir")
    if args.tune_wrist_and_grab and args.seconds < args.offer_delay + 10:
        command.error("tuning --seconds must fit --offer-delay plus the full 10s offer window")
    evidence = Evidence()
    print(f"Local evidence: {evidence.path}", flush=True)
    print(
        "NO provider/model/API: selected PNG is only a native base image; no translated-block haptic. "
        "New watched captures only signal this SAME asset. Physical gestures/haptics/display are unobserved.",
        flush=True,
    )
    try:
        result = asyncio.run(
            run_tuning(args, evidence) if args.tune_wrist_and_grab else run(args, evidence)
        )
    except KeyboardInterrupt:
        result = 130
    except Exception as exc:
        evidence.summary["status"] = "launcher_error"
        evidence.event("launcher_error", error_type=type(exc).__name__)
        result = 2
    finally:
        evidence.finish()
    return result


if __name__ == "__main__":
    sys.exit(main())
