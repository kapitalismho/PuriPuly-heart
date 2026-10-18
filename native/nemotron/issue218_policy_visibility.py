from __future__ import annotations

import asyncio
import ctypes
import threading
import time
from ctypes import wintypes
from uuid import UUID

import issue218_experiment as local
from issue216_presentation import OwnedClientCapture


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def parse(cls, value):
        return cls.from_buffer_copy(UUID(value).bytes_le)


class VisibilityUnavailable(RuntimeError):
    pass


class OwnedVisibleCapture(OwnedClientCapture):
    def __init__(self):
        super().__init__()
        self.observations = []
        self.lock = threading.Lock()
        self.owned_hwnds = {}
        self.dwm = ctypes.WinDLL("dwmapi")
        self.dwm.DwmGetWindowAttribute.argtypes = [
            wintypes.HWND,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        self.dwm.DwmGetWindowAttribute.restype = ctypes.c_long
        self.u.GetForegroundWindow.restype = wintypes.HWND
        self.u.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        self.u.ShowWindow.restype = wintypes.BOOL

    def owned_window(self, pid, title=None):
        key = (pid, title)
        hwnd = self.owned_hwnds.get(key)
        if hwnd is not None and self.pid(hwnd) == pid:
            return hwnd
        hwnd = super().owned_window(pid, title)
        self.owned_hwnds[key] = hwnd
        return hwnd

    def window_state(self, hwnd, expected_pid):
        if self.pid(hwnd) != expected_pid:
            raise VisibilityUnavailable("Owned HWND identity changed")
        cloak = wintypes.DWORD()
        result = self.dwm.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloak), ctypes.sizeof(cloak))
        return {
            "hwnd": hwnd,
            "pid": expected_pid,
            "visible": bool(self.u.IsWindowVisible(hwnd)),
            "iconic": bool(self.u.IsIconic(hwnd)),
            "cloaked": cloak.value if result == 0 else None,
            "cloak_hresult": result,
        }

    @staticmethod
    def eligible(states):
        return all(s["visible"] and not s["iconic"] and s["cloaked"] == 0 for s in states)

    def move_owned_to_current(self, hwnd, expected_pid):
        if self.pid(hwnd) != expected_pid:
            raise VisibilityUnavailable("Refusing desktop move for nonowned HWND")
        ole = ctypes.WinDLL("ole32")
        ole.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        ole.CoInitializeEx.restype = ctypes.c_long
        initialized = ole.CoInitializeEx(None, 2)
        if initialized < 0 and initialized != -2147417850:
            return {"available": False, "initialize_hresult": initialized}
        pointer = ctypes.c_void_p()
        clsid = GUID.parse("aa509086-5ca9-4c25-8f95-589d3c07b48a")
        iid = GUID.parse("a5cd92ff-29be-454c-8d04-d82879fb3f1b")
        ole.CoCreateInstance.argtypes = [
            ctypes.POINTER(GUID),
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.POINTER(GUID),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        ole.CoCreateInstance.restype = ctypes.c_long
        created = ole.CoCreateInstance(
            ctypes.byref(clsid), None, 1, ctypes.byref(iid), ctypes.byref(pointer)
        )
        outcome = {
            "available": created == 0,
            "create_hresult": created,
            "owned_hwnd": hwnd,
            "owned_pid": expected_pid,
            "scope": "Only owned HWND moved; foreground HWND used read-only for active desktop GUID, never activated/moved/captured",
        }
        try:
            if created != 0:
                return outcome
            table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            is_current = ctypes.WINFUNCTYPE(
                ctypes.c_long, ctypes.c_void_p, wintypes.HWND, ctypes.POINTER(wintypes.BOOL)
            )(table[3])
            get_id = ctypes.WINFUNCTYPE(
                ctypes.c_long, ctypes.c_void_p, wintypes.HWND, ctypes.POINTER(GUID)
            )(table[4])
            move = ctypes.WINFUNCTYPE(
                ctypes.c_long, ctypes.c_void_p, wintypes.HWND, ctypes.POINTER(GUID)
            )(table[5])
            foreground = self.u.GetForegroundWindow()
            current = wintypes.BOOL()
            checked = is_current(pointer, foreground, ctypes.byref(current)) if foreground else -1
            outcome["foreground_is_current_hresult"] = checked
            outcome["foreground_is_current"] = bool(current.value)
            if checked != 0 or not current.value:
                return outcome
            owned_current = wintypes.BOOL()
            outcome["owned_is_current_hresult"] = is_current(
                pointer, hwnd, ctypes.byref(owned_current)
            )
            outcome["owned_is_current"] = bool(owned_current.value)
            if outcome["owned_is_current_hresult"] == 0 and owned_current.value:
                outcome["move_skipped"] = (
                    "Owned HWND already on active desktop; cloak2 alone does not establish foreign virtual desktop"
                )
                return outcome
            target = GUID()
            identified = get_id(pointer, foreground, ctypes.byref(target))
            outcome["desktop_identifier_hresult"] = identified
            if identified != 0:
                return outcome
            if self.pid(hwnd) != expected_pid:
                raise VisibilityUnavailable("Owned HWND changed before desktop move")
            outcome["target_desktop_guid"] = str(UUID(bytes_le=bytes(target)))
            outcome["move_hresult"] = move(pointer, hwnd, ctypes.byref(target))
            return outcome
        except OSError as error:
            outcome["API_error"] = str(error)
            return outcome
        finally:
            if pointer.value:
                table = ctypes.cast(
                    pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
                ).contents
                ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(table[2])(pointer)
            if initialized in (0, 1):
                ole.CoUninitialize()

    def capture(self, overlay_pid, backdrop_pid, backdrop_title, path):
        with self.lock:
            overlay = self.owned_window(overlay_pid)
            backdrop = self.owned_window(backdrop_pid, backdrop_title)
            owned = ((overlay, overlay_pid), (backdrop, backdrop_pid))
            observation = {
                "started_at": time.monotonic(),
                "path": str(path),
                "before": [self.window_state(hwnd, pid) for hwnd, pid in owned],
                "recovery_attempts": [],
            }
            self.observations.append(observation)
            states = observation["before"]
            for attempt in range(2):
                if self.eligible(states):
                    break
                actions = []
                for state, (hwnd, pid) in zip(states, owned):
                    if self.pid(hwnd) != pid:
                        raise VisibilityUnavailable("Owned HWND changed during recovery")
                    action = {"hwnd": hwnd, "pid": pid}
                    if not state["visible"] or state["iconic"] or state["cloaked"]:
                        self.u.ShowWindow(hwnd, 9 if state["iconic"] else 4)
                        action["show_owned_window"] = True
                    if state["cloaked"] is not None and state["cloaked"] & 2:
                        action["move_to_current_desktop"] = self.move_owned_to_current(hwnd, pid)
                    actions.append(action)
                time.sleep(0.1)
                states = [self.window_state(hwnd, pid) for hwnd, pid in owned]
                observation["recovery_attempts"].append(
                    {"attempt": attempt + 1, "actions": actions, "after": states}
                )
            if not self.eligible(states):
                observation["status"] = "invalid_visibility_no_capture"
                observation["before_capture"] = states
                raise VisibilityUnavailable(
                    "Owned overlay/backdrop not visibly composed: " + str(states)
                )
            observation["before_capture"] = states
            try:
                captured = super().capture(overlay_pid, backdrop_pid, backdrop_title, path)
            except Exception as error:
                observation["status"] = "invalid_owned_capture"
                observation["capture_error"] = str(error)
                observation["completed_at"] = time.monotonic()
                raise
            after = [self.window_state(hwnd, pid) for hwnd, pid in owned]
            observation["after_capture"] = after
            observation["completed_at"] = time.monotonic()
            if not self.eligible(after):
                observation["status"] = "invalid_visibility_changed_during_capture"
                observation["invalid_capture"] = captured
                raise VisibilityUnavailable("Owned visibility changed during pixel readback")
            observation["status"] = "uncloaked_owned_pixel_capture"
            captured["visibility_valid"] = True
            captured["visibility_before_capture"] = states
            captured["visibility_after_capture"] = after
            captured["visibility_recovery_attempts"] = observation["recovery_attempts"]
            return captured


class GuardedDesktop(local.Desktop):
    def __init__(self, output):
        super().__init__(output)
        self.capture = OwnedVisibleCapture()

    async def readback(self, snapshot, ack_at):
        await super().readback(snapshot, ack_at)
        record = next(r for r in self.readbacks if r["revision"] == snapshot.revision)
        captured = record.get("capture", {})
        pixels = captured.get("exact_color_pixel_counts", {})
        expected = {b.speaker_style for b in snapshot.blocks if b.speaker_style is not None}
        record["readback_valid"] = (
            captured.get("visibility_valid") is True
            and sum(pixels.values()) > 0
            and all(pixels.get(style, 0) > 0 for style in expected)
            and record["revision"]
            == record["current_revision_at_readback_start"]
            == record["current_revision_at_readback_end"]
        )
        visual = self.renderer.window.renderer_visual_state()
        owned = [
            (self.capture.owned_window(self.pid), self.pid),
            (self.capture.owned_window(self.backdrop_pid, self.title), self.backdrop_pid),
        ]
        states = [self.capture.window_state(hwnd, pid) for hwnd, pid in owned]
        overlay_rect = self.capture.rect(owned[0][0], client=True)
        backdrop_rect = self.capture.rect(owned[1][0], client=True)
        left, top, right, bottom = backdrop_rect
        geometry_changed = overlay_rect != backdrop_rect or (
            visual["window_width"],
            visual["window_height"],
        ) != (right - left, bottom - top)
        if (
            not record["readback_valid"]
            and self.capture.eligible(states)
            and self.revision == snapshot.revision
            and geometry_changed
        ):
            record["initial_capture"] = captured
            record["initial_capture_error"] = record.get("capture_error")
            record["initial_readback_valid"] = False
            record["initial_readback_completed_at"] = record["readback_completed_at"]
            control = {
                "command": "apply_window_bounds",
                "x": left,
                "y": top,
                "width": right - left,
                "height": bottom - top,
            }
            record["owned_layout_recovery"] = {
                "before_visual_state": visual,
                "owned_overlay_rect": overlay_rect,
                "owned_backdrop_rect": backdrop_rect,
                "control": control,
                "started_at": time.monotonic(),
                "scope": "One current-owner bounds reapply to owned opaque backdrop after native restore; same immutable snapshot/text/palette/revision, no unrelated HWND or OS desktop mutation",
            }
            await self.renderer.window.dispatch_runtime_control(control)
            await asyncio.sleep(0.06)
            if self.revision == snapshot.revision:
                try:
                    recovered = await asyncio.to_thread(
                        self.capture.capture,
                        self.pid,
                        self.backdrop_pid,
                        self.title,
                        self.output / f"guard_recovered_{snapshot.revision}.png",
                    )
                    record["capture"] = recovered
                    pixels = recovered.get("exact_color_pixel_counts", {})
                    record["readback_completed_at"] = time.monotonic()
                    record["current_revision_at_readback_end"] = self.revision
                    record["readback_valid"] = (
                        recovered.get("visibility_valid") is True
                        and sum(pixels.values()) > 0
                        and all(pixels.get(style, 0) > 0 for style in expected)
                        and self.revision == snapshot.revision
                    )
                    if record["readback_valid"]:
                        record.pop("capture_error", None)
                except Exception as error:
                    record["capture_error"] = str(error)
            record["owned_layout_recovery"][
                "after_visual_state"
            ] = self.renderer.window.renderer_visual_state()
            record["owned_layout_recovery"]["completed_at"] = time.monotonic()
        record["readback_eligibility"] = (
            "Uncloaked/nonminimized/visible owned overlay and backdrop before+after actual pixels; all requested palette styles observed; stable requested revision. ACK alone is not validity."
        )
        if not record["readback_valid"]:
            record["invalid_readback_reason"] = record.get(
                "capture_error",
                "No requested caption pixels or unstable revision; not valid readback",
            )
