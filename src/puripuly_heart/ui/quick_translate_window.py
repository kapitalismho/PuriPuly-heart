from __future__ import annotations

import asyncio
import ctypes
import json
import os
import subprocess
import sys
from uuid import uuid4

import flet as ft

from puripuly_heart.app.services.quick_chat import chat_text
from puripuly_heart.app.services.quick_translate import FRAME_PREFIX, MAX_FRAME_BYTES
from puripuly_heart.runtime_layout import current_runtime_layout
from puripuly_heart.ui.flet_runtime import is_control_mounted
from puripuly_heart.ui.fonts import font_for_language, register_ui_fonts
from puripuly_heart.ui.i18n import get_locale, set_locale, t
from puripuly_heart.ui.quick_window_layout import (
    DEFAULT_QUICK_HEIGHT,
    DEFAULT_QUICK_WIDTH,
    clamp_window_bounds,
)
from puripuly_heart.ui.theme import (
    COLOR_BACKGROUND,
    COLOR_DIVIDER,
    COLOR_NEUTRAL_DARK,
    get_app_theme,
)


def quick_child_environment(base):
    environment = dict(base)
    for name in tuple(environment):
        if (
            name.startswith("FLET_DART_BRIDGE")
            or name in {"FLET_FORCE_WEB_SERVER", "FLET_DISPLAY_URL_PREFIX", "FLET_SERVER_PORT"}
            or name.endswith(("_API_KEY", "_TOKEN", "_PSK"))
        ):
            environment.pop(name, None)
    environment["FLET_SERVER_IP"] = "127.0.0.1"
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


async def spawn_quick_window(locale: str):
    layout = current_runtime_layout()
    executable = layout.host_executable
    if layout.host_kind == "source" and executable.name.lower() == "pythonw.exe":
        executable = executable.with_name("python.exe")
    command = [str(executable)]
    if layout.host_kind == "source":
        command.extend(["-m", "puripuly_heart.main"])
    elif layout.host_kind == "native":
        command.append("--headless")
    command.extend(["run-quick-translate", "--locale", locale])
    return await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        env=quick_child_environment(os.environ),
        limit=MAX_FRAME_BYTES,
    )


async def stop_quick_window(process):
    if process.returncode is not None:
        return
    if os.name == "nt":
        killer = await asyncio.create_subprocess_exec(
            "taskkill",
            "/PID",
            str(process.pid),
            "/T",
            "/F",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        await killer.wait()
    else:
        process.kill()
    await process.wait()


def primary_work_area():
    if os.name != "nt":
        return 0, 0, 1920, 1080
    from ctypes import wintypes

    rect = wintypes.RECT()
    if not ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(rect), 0):
        return 0, 0, 1280, 720
    scale = ctypes.windll.user32.GetDpiForSystem() / 96.0
    return tuple(round(value / scale) for value in (rect.left, rect.top, rect.right, rect.bottom))


class QuickWindow:
    def __init__(self, page):
        self.page = page
        self.pending = None
        self.draft_revision = 0
        self.history = {}
        self.order = []
        self.records = {}
        self.context = dict(translation_enabled=True, peer_translation_enabled=True)
        self.visibility_revision = 0
        self.input = ft.TextField(
            color=COLOR_NEUTRAL_DARK,
            expand=True,
            label=t("quick.input"),
            multiline=True,
            shift_enter=True,
            min_lines=2,
            max_lines=4,
            max_length=8000,
            on_submit=self.send,
            on_change=self.draft_changed,
        )
        self.status = ft.Text(t("quick.hint"), color=COLOR_NEUTRAL_DARK, expand=True)
        self.results = ft.Column(
            expand=True,
            spacing=12,
            scroll=ft.ScrollMode.AUTO,
            auto_scroll=True,
            auto_scroll_animation=0,
            on_scroll=self.history_scrolled,
        )
        self.pin = ft.Switch(
            label=t("quick.pin"),
            value=True,
            on_change=self.set_pin,
            label_text_style=ft.TextStyle(color=COLOR_NEUTRAL_DARK),
        )
        self.mode = ft.Dropdown(
            label=t("quick.peer_display"),
            value="both",
            options=[
                ft.DropdownOption(key=key, text=t("quick.mode." + key))
                for key in ("both", "translation", "original")
            ],
            on_select=self.set_mode,
        )
        self.languages = ft.Text(color=COLOR_NEUTRAL_DARK, size=12)
        self.opacity = ft.Slider(
            min=40,
            max=100,
            divisions=60,
            value=100,
            label="{value}%",
            expand=True,
            on_change=self.set_opacity,
            on_change_end=lambda _: self.save_preferences(),
        )
        self.button = ft.Button(t("quick.send"), on_click=self.send)
        page.title = t("quick.title")
        page.theme_mode = ft.ThemeMode.LIGHT
        page.theme = get_app_theme(font_family=font_for_language(get_locale()))
        register_ui_fonts(page)
        page.bgcolor = COLOR_BACKGROUND
        page.window.bgcolor = COLOR_BACKGROUND
        page.window.opacity = 1
        page.window.width = DEFAULT_QUICK_WIDTH
        page.window.height = DEFAULT_QUICK_HEIGHT
        page.window.min_width = 360
        page.window.min_height = 300
        page.window.prevent_close = True
        page.window.on_event = self.window_event
        page.add(
            ft.Container(
                expand=True,
                padding=16,
                border=ft.Border.all(1, COLOR_DIVIDER),
                content=ft.Column(
                    [
                        ft.Row([self.pin, ft.TextButton(t("quick.hide"), on_click=self.hide)]),
                        self.mode,
                        ft.Row(
                            [ft.Text(t("quick.opacity"), color=COLOR_NEUTRAL_DARK), self.opacity]
                        ),
                        self.languages,
                        self.results,
                        ft.Row([self.input]),
                        ft.Row([self.button, self.status]),
                    ],
                    expand=True,
                ),
            )
        )
        page.run_task(self.read_parent)

    def emit(self, message):
        line = FRAME_PREFIX + json.dumps(message, ensure_ascii=False) + "\n"
        sys.stdout.buffer.write(line.encode("utf-8"))
        sys.stdout.buffer.flush()

    async def send(self, _event=None):
        text = (self.input.value or "").strip()
        if self.pending is not None or not text:
            return
        self.results.auto_scroll = True
        identity = str(uuid4())
        self.pending = (identity, text, self.draft_revision)
        self.button.disabled = True
        self.status.value = t("quick.pending")
        try:
            self.emit(dict(type="submit", id=identity, text=text))
        except Exception:
            self.status.value = t("quick.error.window_connection_failed")
            self.button.disabled = False
            self.pending = None
        self.page.update()

        await self.scroll_latest()

    async def scroll_latest(self):
        if not is_control_mounted(self.results):
            return
        try:
            self.results.auto_scroll = False
            self.page.update()
            await asyncio.sleep(0.05)
            await asyncio.wait_for(self.results.scroll_to(offset=-1, duration=0), 1)
            self.results.auto_scroll = True
            self.page.update()
        except Exception:
            self.emit(dict(type="failure", code="scroll_failed"))
            self.status.value = t("quick.error.scroll_failed")
            self.page.update()

    async def copy_row(self, row):
        try:
            await ft.Clipboard().set(row.value)
        except Exception:
            self.emit(dict(type="failure"))
            self.status.value = t("quick.error.window_connection_failed")
            self.page.update()

    async def read_parent(self):
        try:
            while True:
                line = await asyncio.to_thread(sys.stdin.buffer.readline, MAX_FRAME_BYTES + 1)
                if not line:
                    await self.page.window.destroy()
                    return
                if len(line) > MAX_FRAME_BYTES:
                    raise ValueError("Window message too large")
                message = json.loads(line)
                if message["type"] == "exit":
                    await self.page.window.destroy()
                    return
                if message["type"] in {"show", "hide"}:
                    revision = message.get("visibility_revision", 0)
                    if not isinstance(revision, int) or revision < self.visibility_revision:
                        continue
                    self.visibility_revision = revision
                if message["type"] == "show":
                    values = message["preferences"]
                    for name, value in clamp_window_bounds(
                        values, work_area=primary_work_area()
                    ).items():
                        setattr(self.page.window, name, value)
                    self.pin.value = bool(values.get("always_on_top", True))
                    self.page.window.always_on_top = self.pin.value
                    self.page.window.visible = True
                    self.page.window.minimized = False
                    self.accept_context(values)
                elif message["type"] == "hide":
                    await self.hide()
                    continue
                elif message["type"] == "result":
                    self.accept_result(message)
                elif message["type"] == "chat":
                    self.accept_chat(message["record"])
                elif message["type"] == "context":
                    self.accept_context(message["values"])
                self.page.update()
                if message["type"] == "result" and not message.get("code"):
                    await self.scroll_latest()
                if message["type"] == "show":
                    self.emit_visibility(True)
        except Exception:
            self.emit(dict(type="failure"))
            self.status.value = t("quick.error.window_connection_failed")
            self.page.update()

    def accept_result(self, message):
        identity = message["id"]
        if not self.pending or self.pending[0] != identity:
            return
        original = self.pending[1]
        code = message.get("code")
        if code:
            canonical = message.get("utterance_id")
            if canonical:
                self.accept_chat(
                    dict(
                        id="self:" + canonical,
                        role="self",
                        original=original,
                        translation="",
                        state="failed",
                    )
                )
            self.status.value = (
                t("quick.error." + code, default=t("quick.error.translation_failed"))
                + " "
                + t("quick.retry")
            )
        else:
            self.results.auto_scroll = True
            self.accept_chat(
                dict(
                    id="self:" + message.get("utterance_id", identity),
                    role="self",
                    original=original,
                    translation=message["translation"],
                    state="complete" if message["translation"] else "original",
                )
            )
            if (
                self.pending
                and self.pending[0] == identity
                and self.draft_revision == self.pending[2]
                and (self.input.value or "").strip() == self.pending[1]
            ):
                self.input.value = ""
            self.status.value = t("quick.complete")
        if self.pending and self.pending[0] == identity:
            self.pending = None
            self.button.disabled = False

    def draft_changed(self, _event):
        self.draft_revision += 1

    def history_scrolled(self, event):
        if event.event_type == ft.ScrollType.USER:
            self.results.auto_scroll = (
                event.direction == ft.ScrollDirection.IDLE
                and event.max_scroll_extent - event.pixels < 24
            )

    def accept_chat(self, record):
        identity = record["id"]
        if identity not in self.records:
            self.order.append(identity)
        previous = self.records.get(identity, {})
        self.records[identity] = {**previous, **record}
        if previous.get("state") == "failed" and record.get("state") == "pending":
            self.records[identity]["state"] = "failed"
        self.order.sort(key=lambda key: self.records[key].get("sequence", float("inf")))
        while len(self.order) > 100:
            removed = self.order.pop(0)
            self.records.pop(removed, None)
        self.render_chat()

    def render_chat(self):
        controls = []
        self.history.clear()
        for identity in self.order:
            record = self.records[identity]
            peer = record["role"] == "peer"
            enabled = self.context.get("translation_enabled", False)
            text = chat_text(record, peer_mode=self.mode.value, translation_enabled=enabled)
            if record["state"] == "failed":
                text += "\n" + t("quick.error.translation_failed")
            elif record["state"] == "pending" and enabled:
                text += "\n" + t("quick.pending")
            row = ft.Text(text, selectable=True, color=COLOR_NEUTRAL_DARK)
            self.history[identity] = (record["original"], row)
            controls.append(
                ft.Container(
                    bgcolor="#FFF2E1" if peer else "#E8F2FF",
                    padding=12,
                    border_radius=10,
                    content=ft.Column(
                        [
                            ft.Row(
                                [
                                    ft.Text(
                                        t("quick.role.peer" if peer else "quick.role.self"),
                                        color=COLOR_NEUTRAL_DARK,
                                        weight=ft.FontWeight.BOLD,
                                        expand=True,
                                    ),
                                    ft.TextButton(
                                        t("quick.copy"),
                                        on_click=lambda _, item=row: self.page.run_task(
                                            self.copy_row, item
                                        ),
                                    ),
                                ]
                            ),
                            row,
                        ],
                        spacing=4,
                    ),
                )
            )
        self.results.controls = controls

    def accept_context(self, values):
        self.context.update(values)
        self.opacity.value = max(40, min(100, int(values.get("opacity_percent", 100))))
        self.page.window.opacity = self.opacity.value / 100
        mode = values.get("peer_display_mode", self.mode.value)
        self.mode.value = mode if mode in {"both", "translation", "original"} else "both"
        self.languages.value = (
            t(
                "quick.languages",
                source=values.get("source_language", ""),
                target=values.get("target_language", ""),
            )
            + " · "
            + t(
                "quick.translation_on"
                if values.get("translation_enabled")
                else "quick.translation_off"
            )
        )
        self.render_chat()

    async def set_mode(self, _event):
        self.render_chat()
        self.page.update()
        self.save_preferences()

    async def set_opacity(self, _event):
        self.page.window.opacity = max(40, min(100, self.opacity.value)) / 100
        self.page.update()

    async def set_pin(self, _event):
        self.page.window.always_on_top = bool(self.pin.value)
        self.page.update()
        self.save_preferences()

    def save_preferences(self):
        window = self.page.window
        self.emit(
            dict(
                type="preferences",
                values=dict(
                    always_on_top=bool(self.pin.value),
                    width=round(window.width or DEFAULT_QUICK_WIDTH),
                    height=round(window.height or DEFAULT_QUICK_HEIGHT),
                    left=round(window.left or 0),
                    top=round(window.top or 0),
                    peer_display_mode=self.mode.value,
                    opacity_percent=round(self.opacity.value),
                ),
            )
        )

    def emit_visibility(self, visible):
        self.emit(
            dict(
                type="visibility",
                visible=bool(visible),
                visibility_revision=self.visibility_revision,
            )
        )

    async def hide(self, _event=None):
        self.save_preferences()
        self.page.window.visible = False
        self.page.update()
        self.emit_visibility(False)

    async def window_event(self, event):
        if event.type == ft.WindowEventType.CLOSE:
            await self.hide()


def run_quick_window(locale):
    set_locale(locale)
    ft.run(lambda page: QuickWindow(page), host="127.0.0.1")
