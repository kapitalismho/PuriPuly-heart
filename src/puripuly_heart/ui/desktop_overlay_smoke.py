from __future__ import annotations

import argparse
import asyncio
import ctypes
import hashlib
import json
import os
import secrets
import subprocess
import threading
import time
from collections.abc import Callable
from ctypes import wintypes
from pathlib import Path
from typing import Any

from puripuly_heart.core.overlay.manifest import OVERLAY_CONTRACT_VERSION, OverlayLaunchManifest
from puripuly_heart.core.overlay.protocol import (
    OverlayPresentationBlock,
    OverlayPresentationSnapshot,
)
from puripuly_heart.ui.desktop_overlay import DesktopOverlayRenderer, FletDesktopRendererWindow
from puripuly_heart.ui.desktop_overlay_repro import LocalAuthenticatedRawIngress
from puripuly_heart.ui.desktop_window_zorder import (
    WindowsWindowZOrderPort,
    WindowZOrderWatch,
    _CtypesWin32WindowApi,
)


class _ParentMonitor:
    async def wait_for_parent_exit(self, stop_event: asyncio.Event) -> None:
        await stop_event.wait()


class _ObservedWindowApi(_CtypesWin32WindowApi):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[int, int]] = []
        self.reassertions = 0
        self.order_queries = 0

    def watch_z_order(self, callback: Callable[[int, int], None]) -> WindowZOrderWatch:
        def observe(event: int, hwnd: int) -> None:
            self.events.append((event, hwnd))
            callback(event, hwnd)

        return super().watch_z_order(observe)

    def topmost_peers_ahead(self, hwnd: int) -> tuple[int, ...]:
        self.order_queries += 1
        return super().topmost_peers_ahead(hwnd)

    def set_topmost_no_activate(self, hwnd: int) -> tuple[bool, int | None]:
        self.reassertions += 1
        return super().set_topmost_no_activate(hwnd)


class _Desktop:
    def __init__(self) -> None:
        import tkinter as tk

        self.api = _CtypesWin32WindowApi()
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        self.user32.GetAncestor.restype = wintypes.HWND
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
        self.user32.SetCursorPos.restype = wintypes.BOOL
        self.user32.mouse_event.argtypes = [
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_size_t,
        ]
        self.root = tk.Tk()
        self.root.title("PuriPuly synthetic desktop smoke")
        self.root.overrideredirect(True)
        self.root.geometry("1344x480+0+0")
        self.canvas = tk.Canvas(self.root, bg="#18354b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_rectangle(0, 0, 1344, 32, fill="#205060", outline="")
        self.canvas.create_text(
            16, 16, anchor="w", text="CONTROLLED SYNTHETIC PEER", fill="#63df96"
        )
        self.clicks = 0
        self.root.bind("<Button-1>", self._clicked)
        self.root.update()
        self.hwnd = int(self.user32.GetAncestor(self.root.winfo_id(), 2))
        self.original_cursor = wintypes.POINT()
        self.user32.GetCursorPos(ctypes.byref(self.original_cursor))

    def _clicked(self, _event: object) -> None:
        self.clicks += 1

    def foreground(self) -> int:
        return int(self.user32.GetForegroundWindow() or 0)

    async def wait(self, seconds: float = 0.3) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.root.update()
            await asyncio.sleep(0.01)

    async def click(self, x: int, y: int) -> None:
        self.user32.SetCursorPos(x, y)
        self.user32.mouse_event(0x0002, 0, 0, 0, 0)
        self.user32.mouse_event(0x0004, 0, 0, 0, 0)
        await self.wait()

    async def drag(self, x: int, y: int, dx: int, dy: int) -> None:
        self.user32.SetCursorPos(x, y)
        self.user32.mouse_event(0x0002, 0, 0, 0, 0)
        try:
            await self.wait(0.1)
            for step in range(1, 11):
                self.user32.SetCursorPos(x + dx * step // 10, y + dy * step // 10)
                await self.wait(0.025)
        finally:
            self.user32.mouse_event(0x0004, 0, 0, 0, 0)
        await self.wait()

    def ahead(self, first: int, second: int) -> bool:
        windows: list[int] = []
        callback = self.api._enum_proc_type(lambda hwnd, _: windows.append(int(hwnd)) or True)
        self.api._user32.EnumWindows(callback, 0)
        return windows.index(first) < windows.index(second)

    def capture(self, hwnd: int, path: Path) -> int:
        from PIL import ImageGrab

        bounds = self.api.window_bounds(hwnd)
        peer_bounds = self.api.window_bounds(self.hwnd)
        if bounds is None or peer_bounds is None:
            raise RuntimeError("capture_window_missing")
        x, y, width, height = bounds
        px, py, pw, ph = peer_bounds
        if not (px <= x and py <= y and x + width <= px + pw and y + height <= py + ph):
            raise RuntimeError("capture_would_include_uncontrolled_desktop")
        image = ImageGrab.grab(bbox=(x, y, x + width, y + height), all_screens=True)
        image.save(path)
        return sum(
            r > 210 and g > 210 and b > 210 for r, g, b in image.convert("RGB").get_flattened_data()
        )

    def close(self) -> None:
        self.user32.mouse_event(0x0004, 0, 0, 0, 0)
        self.user32.SetCursorPos(self.original_cursor.x, self.original_cursor.y)
        self.root.destroy()


def _source_identity() -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    sources = {}
    for name in ("desktop_overlay.py", "desktop_window_zorder.py", "desktop_overlay_smoke.py"):
        path = Path(__file__).with_name(name)
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"revision": revision, "sources_sha256": sources}


async def _capture_caption_frames(
    desktop: _Desktop, hwnd: int, output_dir: Path, name: str
) -> dict[str, object]:
    pixels = desktop.capture(hwnd, output_dir / f"{name}.png")
    observation: dict[str, object] = {"initial_pixels": pixels, "final_pixels": pixels}
    if pixels <= 100:
        await desktop.wait(0.3)
        observation["final_pixels"] = desktop.capture(hwnd, output_dir / f"{name}-subsequent.png")
        observation["subsequent_frame_after_s"] = 0.3
    return observation


async def run_smoke(output_dir: Path) -> int:
    result: dict[str, Any] = {
        "scenario": "later-visible-topmost-peer",
        "source": _source_identity(),
        "checks": {},
        "cleanup": {},
        "limits": "Synthetic desktop only; capture pixel counts require visual inspection. No VRChat/GPU compatibility claim.",
    }
    checks = result["checks"]
    cleanup = result["cleanup"]
    desktop: _Desktop | None = None
    ingress: LocalAuthenticatedRawIngress | None = None
    renderer: DesktopOverlayRenderer | None = None
    renderer_task: asyncio.Task[int] | None = None
    started = time.monotonic()
    try:
        desktop = _Desktop()
        snapshot = OverlayPresentationSnapshot(
            revision=1,
            blocks=[
                OverlayPresentationBlock(
                    id="fixed",
                    occupant_key="synthetic:fixed",
                    appearance_seq=1,
                    channel="self",
                    block_variant="finalized",
                    primary_text="SYNTHETIC CAPTION 291",
                    secondary_text="",
                    secondary_enabled=False,
                )
            ],
        )
        instance = "desktop-overlay-smoke-" + secrets.token_hex(4)
        token = secrets.token_urlsafe(32)
        ingress = LocalAuthenticatedRawIngress(token, snapshot, instance)
        bridge_url = await ingress.start()
        ready = asyncio.Event()
        events: list[dict[str, object]] = []

        class Lifecycle:
            async def emit(self, event: dict[str, object]) -> None:
                events.append(event)
                if event.get("type") == "overlay_ready":
                    ready.set()

        native_api = _ObservedWindowApi()
        lifecycle = Lifecycle()
        window = FletDesktopRendererWindow(
            window_z_order_port=WindowsWindowZOrderPort(api=native_api),
            event_sink=lifecycle.emit,
            locale="en",
            overlay_instance_id=instance,
        )
        renderer = DesktopOverlayRenderer(
            OverlayLaunchManifest(
                contract_version=OVERLAY_CONTRACT_VERSION,
                app_version="desktop-overlay-smoke",
                overlay_instance_id=instance,
                bridge_url=bridge_url,
                session_token=token,
                parent_pid=os.getpid(),
                startup_deadline_ms=5000,
                log_dir=str(output_dir),
                log_level="WARNING",
                locale="en",
            ),
            window=window,
            lifecycle_sink=lifecycle,
            parent_monitor=_ParentMonitor(),
        )
        renderer_task = asyncio.create_task(renderer.run())
        deadline = time.monotonic() + 8
        while not ready.is_set():
            if renderer_task.done():
                raise RuntimeError(f"renderer_exited_before_ready:{renderer_task.result()}")
            if time.monotonic() >= deadline:
                raise RuntimeError("renderer_ready_timeout")
            await desktop.wait(0.02)
        pid = renderer.window._bound_owner_pid
        windows = desktop.api.top_level_windows_for_process(pid).windows
        if len(windows) != 1:
            raise RuntimeError("renderer_window_identity_ambiguous")
        hwnd = windows[0]
        result["renderer_pid"] = pid
        result["overlay_hwnd"] = hwnd
        result["peer_hwnd"] = desktop.hwnd

        async def mode(value: str) -> None:
            await ingress.send_runtime_control({"command": "set_interaction_mode", "mode": value})
            target = "locked" if value == "pass_through" else "edit"
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                logical = renderer.window.renderer_visual_state().get("interaction_mode")
                transparent = bool(desktop.api.extended_style(hwnd) & 0x20)
                if logical == target and transparent == (target == "locked"):
                    await desktop.wait(0.3)
                    return
                await desktop.wait(0.02)
            raise RuntimeError("interaction_mode_timeout")

        await mode("pass_through")
        await desktop.click(20, 16)
        checks["controlled_peer_foreground"] = desktop.foreground() == desktop.hwnd
        result["bounds_before_transition"] = desktop.api.window_bounds(hwnd)
        checks["initial_caption_pixels"] = desktop.capture(hwnd, output_dir / "before-peer.png")
        peer_before = desktop.api.window_bounds(desktop.hwnd)
        foreground_before = desktop.foreground()
        checks["peer_visible_before_transition"] = desktop.api.is_window_visible(desktop.hwnd)
        checks["peer_ordinary_before_transition"] = not bool(
            desktop.api.extended_style(desktop.hwnd) & 8
        )
        native_api.events.clear()
        reassertions_before = native_api.reassertions
        queries_before = native_api.order_queries
        transition_start = time.monotonic()
        applied, error = desktop.api.set_topmost_no_activate(desktop.hwnd)
        if not applied:
            raise RuntimeError(f"peer_transition_failed:{error}")
        checks["transition_puts_peer_ahead"] = desktop.ahead(desktop.hwnd, hwnd)
        result["occluded_caption_pixels_before_recovery"] = desktop.capture(
            hwnd, output_dir / "peer-raised-before-recovery.png"
        )
        await desktop.wait(0.7)
        result["transition_observed_after_s"] = time.monotonic() - transition_start
        result["bounds_after_transition"] = desktop.api.window_bounds(hwnd)
        checks["caption_pixels_after_transition"] = desktop.capture(
            hwnd, output_dir / "after-peer.png"
        )
        checks["overlay_ahead_of_peer"] = desktop.ahead(hwnd, desktop.hwnd)
        result["topmost_peers_ahead_after_recovery"] = list(desktop.api.topmost_peers_ahead(hwnd))
        result["transition_native_events"] = [
            {
                "event": hex(event),
                "source": (
                    "renderer"
                    if event_hwnd == hwnd
                    else "controlled_peer" if event_hwnd == desktop.hwnd else "other"
                ),
            }
            for event, event_hwnd in list(native_api.events)
        ]
        result["transition_reassertions"] = native_api.reassertions - reassertions_before
        result["transition_order_queries"] = native_api.order_queries - queries_before
        settled_reassertions = native_api.reassertions
        await desktop.wait(0.2)
        checks["recovery_settles_without_feedback"] = (
            native_api.reassertions == settled_reassertions
        )
        checks["placement_stable"] = (
            result["bounds_before_transition"] == result["bounds_after_transition"]
        )
        checks["peer_placement_stable"] = peer_before == desktop.api.window_bounds(desktop.hwnd)
        checks["recovery_preserves_foreground"] = (
            desktop.foreground() == foreground_before == desktop.hwnd
        )
        checks["topmost_retained"] = bool(desktop.api.extended_style(hwnd) & 8)

        bounds = desktop.api.window_bounds(hwnd)
        x, y, width, height = bounds
        before_clicks = desktop.clicks
        await desktop.click(x + width // 2, y + height // 2)
        checks["locked_click_reaches_peer"] = desktop.clicks == before_clicks + 1
        await mode("edit")
        before_move = desktop.api.window_bounds(hwnd)
        before_clicks = desktop.clicks
        await desktop.drag(x + width // 2, y + height // 2, 0, 60)
        after_move = desktop.api.window_bounds(hwnd)
        result["bounds_after_real_drag"] = after_move
        checks["unlocked_drag_moves_overlay"] = (
            abs(after_move[1] - before_move[1] - 60) <= 3
            and abs(after_move[0] - before_move[0]) <= 3
            and after_move[2:] == before_move[2:]
            and desktop.clicks == before_clicks
        )
        unlocked_caption = await _capture_caption_frames(
            desktop, hwnd, output_dir, "unlocked-moved"
        )
        result["unlocked_caption_observation"] = unlocked_caption
        result["unlocked_visual_state"] = renderer.window.renderer_visual_state()
        checks["unlocked_caption_pixels"] = unlocked_caption["final_pixels"]
        await mode("pass_through")
        await desktop.click(20, 16)
        moved = desktop.api.window_bounds(hwnd)
        checks["relock_preserves_moved_placement"] = moved == after_move
        before_clicks = desktop.clicks
        await desktop.click(moved[0] + moved[2] // 2, moved[1] + moved[3] // 2)
        checks["relocked_click_reaches_peer"] = desktop.clicks == before_clicks + 1
        relocked_caption = await _capture_caption_frames(desktop, hwnd, output_dir, "relocked")
        result["relocked_caption_observation"] = relocked_caption
        result["relocked_visual_state"] = renderer.window.renderer_visual_state()
        checks["relocked_caption_pixels"] = relocked_caption["final_pixels"]
        result["lifecycle_types"] = [event.get("type") for event in events]
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if ingress is not None:
            try:
                await ingress.close()
                cleanup["ingress_closed"] = True
            except Exception as exc:
                cleanup["ingress_error"] = type(exc).__name__
        if renderer is not None:
            try:
                await renderer.shutdown()
                cleanup["renderer_shutdown"] = renderer.is_shutdown
            except Exception as exc:
                cleanup["renderer_error"] = type(exc).__name__
        if renderer_task is not None:
            try:
                cleanup["renderer_exit_code"] = await asyncio.wait_for(renderer_task, 5)
            except BaseException as exc:
                cleanup["renderer_task_error"] = type(exc).__name__
        if desktop is not None:
            try:
                desktop.close()
                cleanup["controlled_window_closed"] = not desktop.api.is_window(desktop.hwnd)
            except Exception as exc:
                cleanup["desktop_error"] = type(exc).__name__
        cleanup["native_watch_threads_drained"] = not any(
            thread.name == "desktop-overlay-zorder" for thread in threading.enumerate()
        )
    result["duration_s"] = time.monotonic() - started
    result["passed"] = (
        "error" not in result
        and all(value if isinstance(value, bool) else value > 100 for value in checks.values())
        and cleanup.get("ingress_closed") is True
        and cleanup.get("renderer_shutdown") is True
        and cleanup.get("renderer_exit_code") == 0
        and cleanup.get("controlled_window_closed") is True
        and cleanup["native_watch_threads_drained"]
    )
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "result": str(output_dir / "result.json")}))
    return 0 if result["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Short real Windows desktop overlay smoke; requires Pillow and an interactive desktop."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    if os.name != "nt":
        parser.error("interactive Windows is required")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    if any(arguments.output_dir.iterdir()):
        parser.error("output directory must be empty")
    return asyncio.run(run_smoke(arguments.output_dir))


if __name__ == "__main__":
    raise SystemExit(main())
