from __future__ import annotations

import argparse
import asyncio
import ctypes
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import time
import traceback
import zipfile
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import issue218_experiment as local
from issue218_policy_visibility import GuardedDesktop

from puripuly_heart.core.overlay.protocol import OverlayPresentationSnapshot


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    text = json.dumps(local.serial(value), indent=2, ensure_ascii=False, default=str)
    for home in (str(Path.home()), str(Path.home()).replace("\\", "/")):
        text = text.replace(home.replace("\\", "\\\\"), "<USER_HOME>").replace(home, "<USER_HOME>")
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text + "\n")


def facts(desktop):
    window = desktop.renderer.window
    capture = desktop.capture
    overlay = capture.owned_window(desktop.pid)
    backdrop = capture.owned_window(desktop.backdrop_pid, desktop.title)
    get_style = capture.u.GetWindowLongPtrW
    get_style.argtypes = [wintypes.HWND, ctypes.c_int]
    get_style.restype = ctypes.c_ssize_t
    color = wintypes.DWORD()
    alpha = ctypes.c_ubyte()
    flags = wintypes.DWORD()
    layered = capture.u.GetLayeredWindowAttributes
    layered.argtypes = [
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.POINTER(wintypes.DWORD),
    ]
    layered.restype = wintypes.BOOL
    layered_ok = bool(
        layered(overlay, ctypes.byref(color), ctypes.byref(alpha), ctypes.byref(flags))
    )
    cloak = wintypes.DWORD()
    dwm = ctypes.WinDLL("dwmapi")
    cloak_result = dwm.DwmGetWindowAttribute(
        wintypes.HWND(overlay), 14, ctypes.byref(cloak), ctypes.sizeof(cloak)
    )
    page = window._page
    model = window._retained_caption_surface
    stack = capture.windows()
    return {
        "at": time.monotonic(),
        "overlay_hwnd": overlay,
        "overlay_pid": capture.pid(overlay),
        "backdrop_hwnd": backdrop,
        "backdrop_pid": capture.pid(backdrop),
        "overlay_visible": bool(capture.u.IsWindowVisible(overlay)),
        "overlay_iconic": bool(capture.u.IsIconic(overlay)),
        "overlay_client_rect": capture.rect(overlay, client=True),
        "backdrop_client_rect": capture.rect(backdrop, client=True),
        "overlay_above_backdrop": stack.index(overlay) < stack.index(backdrop),
        "overlay_style": get_style(overlay, -16),
        "overlay_extended_style": get_style(overlay, -20),
        "layered_attributes": {
            "available": layered_ok,
            "color_key": color.value if layered_ok else None,
            "alpha": alpha.value if layered_ok else None,
            "flags": flags.value if layered_ok else None,
        },
        "cloaked": cloak.value if cloak_result == 0 else None,
        "Flet_page_attached": page is not None,
        "Flet_page_root_count": len(page.controls) if page is not None else None,
        "Flet_window": (
            {
                k: getattr(page.window, k, None)
                for k in (
                    "visible",
                    "opacity",
                    "width",
                    "height",
                    "left",
                    "top",
                    "ignore_mouse_events",
                )
            }
            if page is not None
            else None
        ),
        "renderer_visual_state": window.renderer_visual_state(),
        "retained_surface": (
            {
                "root_visible": model.root.visible,
                "caption_surface_visible": model.caption_surface.visible,
                "slot_column_visible": model.slot_column.visible,
                "slot_visible": [item.visible for item in model.slot_containers],
                "primary_text": [
                    {"value": text.value, "color": text.color} for text in model.primary_texts
                ],
                "secondary_text": [
                    {"value": text.value, "color": text.color} for text in model.secondary_texts
                ],
            }
            if model is not None
            else None
        ),
    }


class ProbeDesktop(GuardedDesktop):
    async def readback(self, snapshot, ack_at):
        before = facts(self)
        await super().readback(snapshot, ack_at)
        record = next(r for r in self.readbacks if r["revision"] == snapshot.revision)
        record["window_facts_at_dispatch"] = before
        record["window_facts_after_readback"] = facts(self)

    async def delayed_picture(self, revision, delay):
        await asyncio.sleep(delay)
        record = {
            "revision": revision,
            "delay_after_ACK_s": delay,
            "at": time.monotonic(),
            "current_revision": self.revision,
            "window_facts": facts(self),
        }
        record["capture"] = await asyncio.to_thread(
            self.capture.capture,
            self.pid,
            self.backdrop_pid,
            self.title,
            self.output / f"late_{revision}_{int(delay * 1000)}.png",
        )
        record["completed_at"] = time.monotonic()
        return record


def presentation_sources(report):
    result = {}
    for relative, historical in report["sources_sha256"].items():
        if relative.startswith(
            (
                "src/puripuly_heart/ui/desktop_overlay",
                "src/puripuly_heart/ui/desktop_window_zorder",
                "src/puripuly_heart/core/overlay/",
            )
        ):
            result[relative] = {
                "historical_sha256": historical,
                "current_sha256": digest(ROOT / relative),
            }
    return result


async def stage(args, report, label, records):
    output = args.output / label
    output.mkdir(parents=True, exist_ok=True)
    desktop = ProbeDesktop(output)
    observed = {
        "label": label,
        "submissions": [],
        "delayed_readbacks": [],
        "errors": [],
        "status": "running",
    }
    report["stages"].append(observed)
    late_tasks = []
    try:
        await desktop.start()
        observed["startup_window_facts"] = facts(desktop)
        if args.repair_verify and label == "Deepgram_original_submissions":
            hwnd = desktop.capture.owned_window(desktop.pid)
            if desktop.capture.pid(hwnd) != desktop.pid:
                raise RuntimeError("Refusing nonowned minimize")
            desktop.capture.u.ShowWindow(hwnd, 6)
            await asyncio.sleep(0.1)
            observed["controlled_owned_minimize"] = facts(desktop)
        origin = records[0]["record"]["submitted_at"]
        started = time.monotonic()
        observed["started_at"] = started
        for index, item in enumerate(records):
            saved = item["record"]
            due = started + saved["submitted_at"] - origin
            await asyncio.sleep(max(0, due - time.monotonic()))
            snapshot = OverlayPresentationSnapshot.from_dict(saved["snapshot"])
            desktop.revision = snapshot.revision - 1
            before = len(desktop.snapshots)
            await desktop.replace_snapshot(snapshot)
            actual = desktop.snapshots[before]
            exact = actual["snapshot"] == saved["snapshot"]
            if not exact:
                raise ValueError("Recorded snapshot changed during renderer-only submission")
            observed["submissions"].append(
                {
                    "case": item["case"],
                    "mode": item["mode"],
                    "profile": item["profile"],
                    "repetition": item["repetition"],
                    "historical_submitted_at": saved["submitted_at"],
                    "historical_dispatch_ACK_at": saved.get("software_commit_observed_at"),
                    "scheduled_at": due,
                    "actual_submitted_at": actual["submitted_at"],
                    "actual_dispatch_ACK_at": actual["software_commit_observed_at"],
                    "exact_recorded_snapshot": exact,
                    "snapshot_sha256": hashlib.sha256(
                        json.dumps(saved["snapshot"], sort_keys=True, ensure_ascii=False).encode()
                    ).hexdigest(),
                    "snapshot": saved["snapshot"],
                    "window_facts_after_dispatch": facts(desktop),
                }
            )
            if snapshot.blocks and index == 0 and not args.repair_verify:
                for delay in (0.25, 1.0):
                    late_tasks.append(
                        asyncio.create_task(desktop.delayed_picture(snapshot.revision, delay))
                    )
            if args.repair_verify and desktop.readback_tasks:
                await asyncio.wait_for(asyncio.gather(*desktop.readback_tasks), 3)
        if late_tasks:
            observed["delayed_readbacks"] = await asyncio.gather(*late_tasks)
        if desktop.readback_tasks:
            await asyncio.gather(*desktop.readback_tasks)
        observed["status"] = "render_only_replay_executed"
    except Exception as error:
        observed["status"] = "failed"
        observed["error"] = str(error)
        observed["traceback"] = traceback.format_exc()
    finally:
        await desktop.close()
        observed["renderer_events"] = desktop.events
        observed["renderer_snapshots"] = desktop.snapshots
        observed["readbacks"] = desktop.readbacks
        observed["errors"] += desktop.errors
        observed["nonempty_snapshots"] = sum(
            bool(s["snapshot"]["blocks"]) for s in observed["submissions"]
        )
        observed["positive_caption_pixel_readbacks"] = sum(
            sum(r.get("capture", {}).get("exact_color_pixel_counts", {}).values()) > 0
            for r in observed["readbacks"]
        )
        observed["blank_caption_pixel_readbacks"] = sum(
            bool(r.get("capture"))
            and not sum(r["capture"].get("exact_color_pixel_counts", {}).values())
            for r in observed["readbacks"]
        )
        observed["visibility_guard_observations"] = desktop.capture.observations
        observed["valid_owned_caption_readbacks"] = sum(
            r.get("readback_valid") is True for r in observed["readbacks"]
        )
        observed["invalid_visibility_readbacks"] = [
            r["revision"] for r in observed["readbacks"] if r.get("readback_valid") is not True
        ]
        if observed["status"] == "render_only_replay_executed":
            observed["surface_status"] = (
                "all_recorded_captions_visibly_verified"
                if observed["valid_owned_caption_readbacks"] == observed["nonempty_snapshots"]
                else "visibility_or_caption_failure"
            )
        save(args.output / "trace.json", report)
        print(
            json.dumps(
                {
                    "stage": label,
                    "status": observed["status"],
                    "snapshots": len(observed["submissions"]),
                    "positive_caption_pixel_readbacks": observed[
                        "positive_caption_pixel_readbacks"
                    ],
                    "blank_caption_pixel_readbacks": observed["blank_caption_pixel_readbacks"],
                    "error": observed.get("error"),
                }
            ),
            flush=True,
        )


async def run(args):
    for name in tuple(os.environ):
        if re.search(
            r"(?:API_KEY|TOKEN|SECRET|CREDENTIAL|AUTHORIZATION)", name, re.I
        ) or name.startswith(("GOOGLE_", "GEMINI_")):
            os.environ.pop(name, None)
    gpu = json.loads(args.gpu_trace.read_bytes())
    recorded = json.loads(args.recorded_trace.read_bytes())
    gpu_case = next(
        c
        for c in gpu["cases"]
        if c["case"] == "distinct_repeated_A_B_A"
        and c["mode"] == "new"
        and c["profile"] == "1.04"
        and c["repetition"] == 1
    )
    controls = [
        {
            "case": gpu_case["case"],
            "mode": gpu_case["mode"],
            "profile": gpu_case["profile"],
            "repetition": 1,
            "record": s,
        }
        for s in gpu_case["renderer_snapshots"]
    ]
    replay = [
        {
            "case": c["case"],
            "mode": c["mode"],
            "profile": c["profile"],
            "repetition": c["repetition"],
            "record": s,
        }
        for c in recorded["cases"]
        for s in c["renderer_snapshots"]
    ]
    if args.repair_verify:
        replay = [
            r
            for r in replay
            if r["case"] == "linguistic_correction"
            and r["mode"] == "new"
            and r["profile"] == "1.04"
        ]
    for rows in (controls, replay):
        revisions = [r["record"]["snapshot"]["revision"] for r in rows]
        if revisions != sorted(set(revisions)):
            raise ValueError("Recorded renderer event order is not strictly increasing")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "issue218-renderer-only-observed-output-1",
        "command": sys.argv,
        "driver_sha256": digest(Path(__file__)),
        "historical_product_candidate": recorded["candidate"],
        "current_product_state": "Uncommitted F1/F2 core changes reported by Director; renderer-only data bypasses ASR/translation/source policy. Exact presentation file hashes recorded.",
        "original_traces": {
            "gpu": {"path": str(args.gpu_trace), "sha256": digest(args.gpu_trace)},
            "fixed_input_replay": {
                "path": str(args.recorded_trace),
                "sha256": digest(args.recorded_trace),
            },
        },
        "presentation_sources": presentation_sources(recorded),
        "helpers_sha256": {
            p: digest(ROOT / p)
            for p in (
                "native/nemotron/issue218_experiment.py",
                "native/nemotron/issue216_presentation.py",
            )
        },
        "dependencies": {"flet": importlib.metadata.version("flet")},
        "privacy": {
            "ASR_calls": 0,
            "Gemma_calls": 0,
            "provider_network_calls": 0,
            "credential_reads": False,
            "ambient_capture": False,
            "VRChat_SteamVR": False,
        },
        "contract": "Renderer-only replay of actual previously observed source/translation snapshots. Preserve recorded text/style/identity/scope/generation/revision/retirement-frontier/event order; rebase dispatch host origin only. Recorded subset omits unretained redundant clear submissions; revision gaps stay intact. No fresh ASR/translation or E2E/photon latency qualification.",
        "stages": [],
        "status": "running",
    }
    report["visibility_helper_sha256"] = digest(
        ROOT / "native/nemotron/issue218_policy_visibility.py"
    )
    report["visibility_guard_contract"] = (
        "At most2owned-only show/restore/current-desktop moves per capture; at most1same-revision owner-bounds reapply+recapture. Explicit invalid visibility on exhaustion. No global desktop switches/settings or unrelated-window mutation. Both overlay/backdrop uncloaked before+aftercapture plus actualrequestedpalettepixels/stablerevision required for validreadback."
    )
    report["repair_verify"] = args.repair_verify
    report["repair_verify_scope"] = (
        "Working4-captioncontrol plus sole invalid minimized numeric29; previously unaffected22/23guardedreadbacks reused. Renderer-only diagnostic readback barrier preserves eventorder but does not qualify old/fresh E2E latency."
        if args.repair_verify
        else "All23recordedDeepgramcaptions plus working4-captioncontrol at rebased original submission intervals."
    )
    if args.numeric_only:
        report["repair_verify_scope"] = (
            "Sole minimized numeric29 after evidenced owned backdrop/layout correction; unaffected22Deepgram and working4control guardedreadbacks reused. Same recorded immutable text/palette/scope/eventorder. Renderer-only, no new E2Elatency."
        )
    if not all(
        v["historical_sha256"] == v["current_sha256"]
        for v in report["presentation_sources"].values()
    ):
        raise ValueError("Presentation sources changed; needs Director applicability disposition")
    source_paths = sorted(
        set(report["presentation_sources"])
        | {
            "native/nemotron/issue218_policy_renderer_probe.py",
            "native/nemotron/issue218_policy_visibility.py",
            "native/nemotron/issue218_experiment.py",
            "native/nemotron/issue216_presentation.py",
        }
    )
    with zipfile.ZipFile(
        args.output / "execution_sources.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for relative in source_paths:
            archive.write(ROOT / relative, relative)
    report["execution_sources_zip_sha256"] = digest(args.output / "execution_sources.zip")
    try:
        async with asyncio.timeout(180):
            if not args.numeric_only:
                await stage(args, report, "GPU_working_control", controls)
            await stage(args, report, "Deepgram_original_submissions", replay)
        report["status"] = (
            "renderer_only_probe_surface_verified"
            if all(
                s.get("surface_status") == "all_recorded_captions_visibly_verified"
                for s in report["stages"]
            )
            else "renderer_only_probe_partial"
        )
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
    save(args.output / "trace.json", report)
    return 0 if report["status"] == "renderer_only_probe_surface_verified" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu-trace",
        type=Path,
        default=ROOT / "native/nemotron/issue218_policy_gpu_valid/trace.json",
    )
    parser.add_argument(
        "--recorded-trace",
        type=Path,
        default=ROOT / "native/nemotron/issue218_policy_deepgram_final/trace.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repair-verify", action="store_true")
    parser.add_argument("--numeric-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.numeric_only and not args.repair_verify:
        raise ValueError("Numeric-only requires explicit renderer repair verification")
    if args.output.parent not in (
        ROOT / "native/nemotron",
        ROOT / "build",
    ) or not args.output.name.startswith("issue218_policy"):
        raise ValueError("Probe output must remain assigned issue218_policy* surfaces")
    raise SystemExit(asyncio.run(run(args)))
