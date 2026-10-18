"""Bounded #216 real desktop renderer observation; only task-local evidence writes.

No application composition, settings, audio capture, cloud/model/VR APIs, or fake
translation. Replays only authorized recorded public-source text to the existing
OverlayEventAdapter -> OverlayPresenter -> actual DesktopOverlayRenderer.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ctypes
import dataclasses
import hashlib
import json
import os
import secrets
import struct
import subprocess
import sys
import time
import traceback
import zlib
from ctypes import wintypes
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def backdrop_main(title):
    # This separate, owned process never enumerates/captures any other application.
    import threading
    import tkinter as tk

    root = tk.Tk()
    root.title(title)
    root.overrideredirect(True)
    root.geometry("1344x336+32+32")
    root.attributes("-topmost", True)
    canvas = tk.Canvas(root, width=1344, height=336, highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    for row in range(4):
        for column in range(8):
            canvas.create_rectangle(
                column * 168,
                row * 84,
                (column + 1) * 168,
                (row + 1) * 84,
                outline="",
                fill="#135BFF" if (row + column) % 2 == 0 else "#E000A8",
            )
    root.update()
    print(json.dumps({"pid": os.getpid(), "title": title}), flush=True)
    stopping = threading.Event()
    threading.Thread(target=lambda: (sys.stdin.readline(), stopping.set()), daemon=True).start()

    def check_stop():
        if stopping.is_set():
            root.destroy()
        else:
            root.after(50, check_stop)

    root.after(50, check_stop)
    root.mainloop()


class OwnedClientCapture:
    """Real screen pixels, bounded to an owned overlay over an opaque owned backdrop.

    Before any pixel read, verifies both HWND PIDs and client rectangles and rejects
    any intersecting visible foreign HWND above the opaque backdrop. No full-screen
    screenshot is taken, even temporarily; only the exact owned client crop is read.
    """

    def __init__(self):
        self.u = ctypes.WinDLL("user32", use_last_error=True)
        self.g = ctypes.WinDLL("gdi32", use_last_error=True)
        self.callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        self.u.EnumWindows.argtypes = [self.callback_type, wintypes.LPARAM]
        self.u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        self.u.IsWindowVisible.argtypes = [wintypes.HWND]
        self.u.IsIconic.argtypes = [wintypes.HWND]
        self.u.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.u.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.u.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        self.u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        self.u.SetWindowPos.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        self.u.GetDC.argtypes = [wintypes.HWND]
        self.u.GetDC.restype = wintypes.HDC
        self.u.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
        self.g.CreateCompatibleDC.argtypes = [wintypes.HDC]
        self.g.CreateCompatibleDC.restype = wintypes.HDC
        self.g.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
        self.g.CreateCompatibleBitmap.restype = wintypes.HBITMAP
        self.g.SelectObject.argtypes = [wintypes.HDC, wintypes.HANDLE]
        self.g.SelectObject.restype = wintypes.HANDLE
        self.g.BitBlt.argtypes = [
            wintypes.HDC,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.HDC,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.DWORD,
        ]
        self.g.GetDIBits.argtypes = [
            wintypes.HDC,
            wintypes.HBITMAP,
            wintypes.UINT,
            wintypes.UINT,
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.UINT,
        ]
        self.g.DeleteObject.argtypes = [wintypes.HANDLE]
        self.g.DeleteDC.argtypes = [wintypes.HDC]

    def pid(self, hwnd):
        value = wintypes.DWORD()
        self.u.GetWindowThreadProcessId(hwnd, ctypes.byref(value))
        return value.value

    def windows(self):
        result = []
        callback = self.callback_type(lambda hwnd, _: result.append(int(hwnd)) or True)
        if not self.u.EnumWindows(callback, 0):
            raise ctypes.WinError(ctypes.get_last_error())
        return result

    def owned_window(self, pid, title=None):
        candidates = []
        for hwnd in self.windows():
            if self.pid(hwnd) != pid or not self.u.IsWindowVisible(hwnd):
                continue
            if title is not None:
                text = ctypes.create_unicode_buffer(512)
                self.u.GetWindowTextW(hwnd, text, 512)
                if text.value != title:
                    continue
            candidates.append(hwnd)
        if len(candidates) != 1:
            raise RuntimeError(f"owned HWND unresolved for PID {pid}: {len(candidates)} candidates")
        return candidates[0]

    def rect(self, hwnd, client=False):
        value = wintypes.RECT()
        function = self.u.GetClientRect if client else self.u.GetWindowRect
        if not function(hwnd, ctypes.byref(value)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not client:
            return (value.left, value.top, value.right, value.bottom)
        point = wintypes.POINT(value.left, value.top)
        if not self.u.ClientToScreen(hwnd, ctypes.byref(point)):
            raise ctypes.WinError(ctypes.get_last_error())
        return (
            point.x,
            point.y,
            point.x + value.right - value.left,
            point.y + value.bottom - value.top,
        )

    def capture(self, overlay_pid, backdrop_pid, backdrop_title, path):
        overlay = self.owned_window(overlay_pid)
        backdrop = self.owned_window(backdrop_pid, backdrop_title)
        # Reassert only the two owned fixture HWNDs, without activating, moving,
        # hiding, or otherwise interacting with a foreign application window.
        for hwnd in (backdrop, overlay):
            if not self.u.SetWindowPos(hwnd, wintypes.HWND(-1), 0, 0, 0, 0, 0x0013):
                raise ctypes.WinError(ctypes.get_last_error())
        ctypes.WinDLL("dwmapi").DwmFlush()
        crop = self.rect(overlay, client=True)
        background = self.rect(backdrop, client=True)
        x, y, right, bottom = crop
        if not (
            background[0] <= x < right <= background[2]
            and background[1] <= y < bottom <= background[3]
        ):
            raise RuntimeError(
                "owned opaque backdrop does not cover overlay client; no pixels captured"
            )
        stack = self.windows()
        if stack.index(overlay) >= stack.index(backdrop):
            raise RuntimeError("owned overlay is not above owned backdrop; no pixels captured")
        checked = 0
        for hwnd in stack[: stack.index(backdrop)]:
            if self.pid(hwnd) in (overlay_pid, backdrop_pid):
                continue
            if not self.u.IsWindowVisible(hwnd) or self.u.IsIconic(hwnd):
                continue
            left2, top2, right2, bottom2 = self.rect(hwnd)
            checked += 1
            if max(x, left2) < min(right, right2) and max(y, top2) < min(bottom, bottom2):
                raise RuntimeError(
                    "visible foreign HWND intersects owned crop above backdrop; "
                    f"rect={(left2, top2, right2, bottom2)}; no pixels captured"
                )
        width, height = right - x, bottom - y
        dc = self.u.GetDC(None)
        memory = self.g.CreateCompatibleDC(dc)
        bitmap = self.g.CreateCompatibleBitmap(dc, width, height)
        previous = self.g.SelectObject(memory, bitmap)
        try:
            if not self.g.BitBlt(memory, 0, 0, width, height, dc, x, y, 0x40CC0020):
                raise ctypes.WinError(ctypes.get_last_error())
            self.g.SelectObject(memory, previous)
            info = (
                struct.pack(
                    "<IiiHHIIiiII", 40, width, -height, 1, 32, 0, width * height * 4, 0, 0, 0, 0
                )
                + b"\0" * 4
            )
            pixels = ctypes.create_string_buffer(width * height * 4)
            info_buffer = ctypes.create_string_buffer(info)
            if self.g.GetDIBits(memory, bitmap, 0, height, pixels, info_buffer, 0) != height:
                raise ctypes.WinError(ctypes.get_last_error())
            raw = pixels.raw
            rgb = bytearray(width * height * 3)
            rgb[0::3], rgb[1::3], rgb[2::3] = raw[2::4], raw[1::4], raw[0::4]
            rows = b"".join(
                b"\0" + rgb[row * width * 3 : (row + 1) * width * 3] for row in range(height)
            )

            def chunk(kind, data):
                return (
                    struct.pack(">I", len(data))
                    + kind
                    + data
                    + struct.pack(">I", zlib.crc32(kind + data))
                )

            png = (
                b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(rows))
                + chunk(b"IEND", b"")
            )
            path.write_bytes(png)
            colors = {
                name: sum(
                    1
                    for offset in range(0, len(rgb), 3)
                    if tuple(rgb[offset : offset + 3]) == value
                )
                for name, value in {
                    "gold": (255, 215, 0),
                    "cyan": (64, 219, 255),
                    "gray": (180, 180, 180),
                }.items()
            }
            return {
                "method": "Win32 BitBlt exact owned client crop over opaque owned checkerboard",
                "overlay_hwnd": overlay,
                "overlay_pid": overlay_pid,
                "backdrop_hwnd": backdrop,
                "backdrop_pid": backdrop_pid,
                "crop_screen_xyxy": crop,
                "backdrop_client_xyxy": background,
                "foreign_visible_rectangles_checked_without_capture": checked,
                "foreign_intersection": False,
                "width": width,
                "height": height,
                "exact_color_pixel_counts": colors,
                "sha256": hashlib.sha256(png).hexdigest(),
                "path": path.relative_to(ROOT).as_posix(),
            }
        finally:
            self.g.SelectObject(memory, previous)
            self.g.DeleteObject(bitmap)
            self.g.DeleteDC(memory)
            self.u.ReleaseDC(None, dc)


def read_sources(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    case = next(item for item in value["cases"] if item["case"] == "distinct_repeated_A_B_A")
    if case["status"] != "passed":
        raise RuntimeError("Recorded source case did not complete")
    return [
        {
            "admitted_at": item["admitted_at"],
            "source": {
                key: item["source"][key]
                for key in (
                    "source_id",
                    "revision",
                    "channel",
                    "order",
                    "raw_text",
                    "language",
                    "text_range",
                    "speaker",
                    "display_group_id",
                )
            },
        }
        for item in case["sources"]
    ]


async def run(args, result):
    # Deliberately direct isolated startup: puripuly_heart.main/settings/app composition
    # is never invoked. Renderer stays default FletDesktopRendererWindow, persistence-free.
    from puripuly_heart.config.overlay_calibration import OverlayCalibration
    from puripuly_heart.core.overlay.manifest import OVERLAY_CONTRACT_VERSION, OverlayLaunchManifest
    from puripuly_heart.core.overlay.presenter import OverlayPresenter
    from puripuly_heart.core.overlay.protocol import OverlayPresentationSnapshot
    from puripuly_heart.core.overlay.sink import OverlayEventAdapter
    from puripuly_heart.core.speaker_identity import PeerSpeakerIdentityAllocator
    from puripuly_heart.domain.models import FinalSpeakerRun, Transcript
    from puripuly_heart.ui.desktop_overlay import (
        DesktopOverlayRenderer,
        DiagnosticLocalRendererPort,
    )
    from puripuly_heart.ui.desktop_overlay_repro import (
        LocalAuthenticatedRawIngress,
        _NeverExitParentMonitor,
    )
    from puripuly_heart.ui.desktop_overlay_surface.renderer import build_desktop_caption_plan

    ingress = LocalAuthenticatedRawIngress(
        secrets.token_urlsafe(32),
        OverlayPresentationSnapshot(revision=0, blocks=[]),
        "issue216-presentation",
    )
    port = DiagnosticLocalRendererPort(acknowledgement_timeout_s=10)
    ready = asyncio.Event()
    committed = {}
    record_events = []
    renderer = None
    renderer_task = None
    consumer_task = None
    backdrop = None
    backdrop_pid = None
    presenter = None
    capture = OwnedClientCapture()
    title = "issue216-owned-checkerboard-" + secrets.token_hex(8)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    result["frames"] = []
    result["renderer_events"] = record_events
    result["cleanup"] = {}
    relevant = (
        "src/puripuly_heart/ui/desktop_overlay.py",
        "src/puripuly_heart/ui/desktop_overlay_repro.py",
        "src/puripuly_heart/ui/desktop_overlay_surface/renderer.py",
        "src/puripuly_heart/ui/desktop_overlay_surface/contract.py",
        "src/puripuly_heart/core/overlay/presenter.py",
        "src/puripuly_heart/core/overlay/state.py",
        "src/puripuly_heart/core/overlay/sink.py",
        "src/puripuly_heart/core/speaker_identity.py",
    )
    result["presentation_source_sha256"] = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in relevant
    }

    async def lifecycle(event):
        if event.get("type") == "overlay_ready":
            ready.set()
        if event.get("type") in ("startup_error", "runtime_error"):
            result.setdefault("lifecycle_errors", []).append(event)

    async def consume():
        while True:
            envelope = await port.next_event()
            record = dict(envelope.record)
            record_events.append(record)
            revision = record["renderer_revision"]
            if envelope.acknowledgement is not None:
                port.acknowledge_render_commit(revision)
            if record["event_type"] == "render_commit_acknowledgement":
                committed.setdefault(revision, asyncio.Event()).set()

    class PresentationIngress:
        revision = 0
        latest = None

        async def replace_snapshot(self, snapshot, *, block_expirations=None):
            self.revision += 1
            self.latest = dataclasses.replace(snapshot, revision=self.revision)
            await ingress.write_batch([self.latest])
            await asyncio.wait_for(committed.setdefault(self.revision, asyncio.Event()).wait(), 10)

        async def broadcast_shutdown(self):
            await ingress.close()

    transport = PresentationIngress()
    try:
        backdrop = subprocess.Popen(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--backdrop", title],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        header = await asyncio.wait_for(asyncio.to_thread(backdrop.stdout.readline), 10)
        backdrop_pid = json.loads(header)["pid"]
        result["backdrop_identity"] = {
            "launcher_pid": backdrop.pid,
            "window_process_pid": backdrop_pid,
            "title": title,
        }
        capture.owned_window(backdrop_pid, title)
        url = await ingress.start()
        manifest = OverlayLaunchManifest(
            contract_version=OVERLAY_CONTRACT_VERSION,
            app_version="issue216-presentation",
            overlay_instance_id="issue216-presentation",
            bridge_url=url,
            session_token=ingress.token,
            parent_pid=os.getpid(),
            startup_deadline_ms=10000,
            log_dir="diagnostic",
            log_level="WARNING",
            locale="en",
        )
        result["ingress"] = {
            "host": "127.0.0.1",
            "ephemeral_port": int(url.rsplit(":", 1)[1]),
            "authenticated": True,
            "token_retained": False,
        }
        renderer = DesktopOverlayRenderer(
            manifest,
            lifecycle_sink=type("Lifecycle", (), {"emit": staticmethod(lifecycle)})(),
            parent_monitor=_NeverExitParentMonitor(),
            diagnostic_port=port,
        )
        consumer_task = asyncio.create_task(consume())
        renderer_task = asyncio.create_task(renderer.run())
        await ingress.wait_connected()
        await asyncio.wait_for(ready.wait(), 15)
        await ingress.send_runtime_control(
            {"command": "apply_window_bounds", "x": 32, "y": 32, "width": 1344, "height": 336}
        )
        await ingress.send_runtime_control(
            {"command": "set_interaction_mode", "mode": "pass_through"}
        )
        await asyncio.sleep(0.2)
        result["actual_renderer"] = type(renderer).__name__
        result["actual_window"] = type(renderer.window).__name__
        overlay_pid = renderer.window._owner_process_pid() or renderer.window._flet_process_pid
        if not overlay_pid or overlay_pid in (os.getpid(), backdrop_pid):
            raise RuntimeError("Default Flet view ownership was not independently identified")
        result["owned_processes"] = {
            "driver": os.getpid(),
            "renderer": overlay_pid,
            "backdrop": backdrop_pid,
        }
        for kind, path, labels in (
            ("baseline", args.baseline, ("baseline_plain", "baseline_terminal")),
            (
                "fixed",
                args.source,
                (
                    "fixed_A",
                    "fixed_gray_crossing",
                    "fixed_B",
                    "fixed_mixed_return_A",
                    "fixed_terminal_A",
                ),
            ),
        ):
            sources = read_sources(path)
            if len(sources) != len(labels):
                raise RuntimeError(f"Unexpected recorded {kind} source count: {len(sources)}")
            result[kind + "_source_groups"] = sources
            allocator = PeerSpeakerIdentityAllocator()
            adapter = OverlayEventAdapter()
            presenter = OverlayPresenter(
                calibration=OverlayCalibration(), bridge=transport, translation_enabled=False
            )
            for sequence, (item, label) in enumerate(zip(sources, labels), 1):
                source = item["source"]
                run_record = source["speaker"]
                run_value = None if run_record is None else FinalSpeakerRun(**run_record)
                # Mirrors translation_turn child Transcript construction. Serialized origins/
                # settings are not rehydrated: they do not affect this presentation seam.
                runs = (
                    (run_value,)
                    if run_value is not None
                    and (run_value.session_scope or run_value.attribution_state != "unavailable")
                    else ()
                )
                transcript = Transcript(
                    utterance_id=UUID(source["source_id"]),
                    text=source["raw_text"],
                    is_final=True,
                    created_at=time.monotonic(),
                    channel="peer",
                    final_speaker_runs=runs,
                    publication_generation=source["order"][0],
                    source_order=source["order"][1],
                    source_text_range=tuple(source["text_range"]),
                    source_text_revision=source["revision"],
                )
                assignment = allocator.observe(transcript, child_sequence=sequence)
                event = adapter.transcript_final(
                    transcript,
                    source_language=source["language"] or "en",
                    target_language="",
                    speaker_assignment=assignment,
                )
                receipt = await presenter.emit_peer_when_admissible(event)
                close_receipt = await presenter.emit(
                    adapter.utterance_closed(
                        utterance_id=transcript.utterance_id, channel="peer", is_final=True
                    )
                )
                await asyncio.sleep(args.dwell_ms / 1000)
                if renderer_task.done():
                    raise RuntimeError(f"Renderer exited early: {renderer_task.result()}")
                snapshot = transport.latest
                if snapshot is None:
                    raise RuntimeError("No actual presenter snapshot")
                image_path = output / (label + ".png")
                captured = capture.capture(overlay_pid, backdrop_pid, title, image_path)
                plan = build_desktop_caption_plan(snapshot, window_width=1344, window_height=336)
                result["frames"].append(
                    {
                        "label": label,
                        "source_id": source["source_id"],
                        "source_raw_text": source["raw_text"],
                        "recorded_speaker": run_record,
                        "assignment": dataclasses.asdict(assignment),
                        "application_receipt": dataclasses.asdict(receipt),
                        "close_receipt": dataclasses.asdict(close_receipt),
                        "presenter_snapshot": snapshot.to_dict(),
                        "existing_caption_plan_lines": [
                            dataclasses.asdict(line) for line in plan.lines
                        ],
                        "capture": captured,
                    }
                )
                write_json(output / "result.json", result)
            await presenter.close()
            presenter = None
            # Clears only this isolated renderer when switching baseline -> candidate.
            await transport.replace_snapshot(OverlayPresentationSnapshot(revision=0, blocks=[]))
        result["outcome"] = "captured_pending_visual_observation"
    except Exception as exc:
        result["outcome"] = "failed"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        if presenter is not None:
            with contextlib.suppress(Exception):
                await presenter.close()
        await ingress.close()
        result["cleanup"]["ingress_closed"] = True
        if renderer is not None:
            with contextlib.suppress(Exception):
                await renderer.shutdown()
            result["cleanup"]["renderer_shutdown"] = renderer.is_shutdown
        if renderer_task is not None:
            try:
                result["cleanup"]["renderer_exit_code"] = await asyncio.wait_for(renderer_task, 10)
            except Exception as exc:
                result["cleanup"]["renderer_exit_error"] = str(exc)
        if consumer_task is not None:
            consumer_task.cancel()
            await asyncio.gather(consumer_task, return_exceptions=True)
        if backdrop is not None:
            with contextlib.suppress(OSError):
                backdrop.stdin.write("close\n")
                backdrop.stdin.flush()
            try:
                await asyncio.to_thread(backdrop.wait, 5)
            except subprocess.TimeoutExpired:
                backdrop.kill()
                await asyncio.to_thread(backdrop.wait, 5)
            result["cleanup"]["owned_backdrop_stopped"] = backdrop.poll() is not None
        write_json(output / "result.json", result)
    return 0 if result["outcome"] == "captured_pending_visual_observation" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backdrop")
    parser.add_argument("--source", type=Path, default=ROOT / "native/nemotron/issue216_final.json")
    parser.add_argument(
        "--baseline", type=Path, default=ROOT / "native/nemotron/issue216_baseline.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "native/nemotron/issue216_presentation_images"
    )
    parser.add_argument("--dwell-ms", type=int, default=650)
    args = parser.parse_args()
    if os.name != "nt":
        raise RuntimeError("Requires local authorized Windows GUI session")
    ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    if args.backdrop:
        backdrop_main(args.backdrop)
        return 0
    args.output = args.output.resolve()
    args.source = args.source.resolve()
    args.baseline = args.baseline.resolve()
    if args.output.parent != ROOT / "native/nemotron" or not args.output.name.startswith(
        "issue216_presentation"
    ):
        raise ValueError(
            "Output must stay in assigned native/nemotron/issue216_presentation* surfaces"
        )
    if not 250 <= args.dwell_ms <= 3000:
        raise ValueError("Bounded dwell must be 250..3000 ms")
    result = {
        "schema": "issue216-actual-desktop-presentation-1",
        "command": [
            sys.executable,
            "-B",
            "native/nemotron/issue216_presentation.py",
            *sys.argv[1:],
        ],
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_report": {
            "path": args.source.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        },
        "baseline_report": {
            "path": args.baseline.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        },
        "safety": {
            "ordinary_app_composition": False,
            "settings_read_write": False,
            "audio_capture": False,
            "cloud_model_calls": False,
            "VR_interaction": False,
            "only_spawned_processes_stopped": True,
            "whole_desktop_capture": False,
        },
        "seam": "Recorded SourceChild fields -> Transcript (existing translation_turn child pattern) -> PeerSpeakerIdentityAllocator -> OverlayEventAdapter.transcript_final/utterance_closed -> OverlayPresenter.emit_peer_when_admissible -> LocalAuthenticatedRawIngress -> default DesktopOverlayRenderer/FletDesktopRendererWindow",
        "limits": [
            "Renderer-only Windows desktop pixels, not HMD/photon evidence",
            "No translation execution or fabricated translated text",
            "Source semantic/ASR ambiguity remains; acknowledgements alone prove no quality",
            "Serialized origins/settings retained in source report, not rehydrated; no source partition/ownership revalidation",
            "Existing default two rows, pacing, TTL unchanged; display_group_id recorded, no #217 aggregation",
            "Replayed sanitized source groups, not original live-time end-to-end GUI timing",
        ],
    }
    return asyncio.run(run(args, result))


if __name__ == "__main__":
    raise SystemExit(main())
