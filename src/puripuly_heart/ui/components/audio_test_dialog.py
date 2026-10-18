from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence

import flet as ft
import flet.canvas as cv

from puripuly_heart.config.resolved import VAD_ONSET_MAX, VAD_ONSET_MIN
from puripuly_heart.ui.flet_runtime import update_control_if_mounted
from puripuly_heart.ui.i18n import t
from puripuly_heart.ui.theme import (
    COLOR_BACKGROUND,
    COLOR_NEUTRAL_DARK,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SURFACE,
    COLOR_WARNING,
)

AUDIO_TEST_CHANNELS = ("self", "peer")
HISTORY_SAMPLES = 160
SAMPLE_INTERVAL_S = 0.032
RENDER_INTERVAL_S = 1 / 12
SILENCE_FILL_AFTER_S = 0.3
_GRAPH_WIDTH = 360
_GRAPH_HEIGHT = 300
_CHANNEL_TITLE_KEYS = {
    "self": "audio_test.channel.self",
    "peer": "audio_test.channel.peer",
}
_FAILURE_KEYS = {
    "input_route_unavailable": "audio_test.error.input_route_unavailable",
    "no_audio_frames": "audio_test.error.no_audio_frames",
    "target_unavailable": "audio_test.error.target_unavailable",
    "vad_unavailable": "audio_test.error.vad_unavailable",
}

AudioTestRunTask = Callable[[Callable[[], Awaitable[None]]], object]


class _ChannelPanel:
    def __init__(
        self,
        channel: str,
        *,
        on_threshold_change: Callable[[str, float], None],
    ) -> None:
        self.channel = channel
        self._on_threshold_change = on_threshold_change
        self.history: deque[tuple[float, float]] = deque(
            [(0.0, 0.0)] * HISTORY_SAMPLES,
            maxlen=HISTORY_SAMPLES,
        )
        self.last_sample_at: float | None = None
        self.threshold = 0.5
        self.dirty = True
        self.device_text = ft.Text("", size=14, color=COLOR_NEUTRAL_DARK, max_lines=1)
        self.warning_text = ft.Text("", size=13, color=COLOR_WARNING, visible=False)
        self.status_dot = ft.Container(
            width=12,
            height=12,
            border_radius=6,
            bgcolor=COLOR_NEUTRAL_DARK,
        )
        self.status_text = ft.Text(
            t("audio_test.state.pending"),
            size=16,
            weight=ft.FontWeight.BOLD,
            color=COLOR_NEUTRAL_DARK,
            expand=True,
            no_wrap=False,
        )
        self.canvas = cv.Canvas(
            shapes=[],
            width=_GRAPH_WIDTH,
            height=_GRAPH_HEIGHT,
        )
        self.slider = ft.Slider(
            min=VAD_ONSET_MIN,
            max=VAD_ONSET_MAX,
            divisions=18,
            value=self.threshold,
            label=f"{self.threshold:.2f}",
            active_color=COLOR_PRIMARY,
            expand=True,
            on_change=self._handle_slider_change,
            on_change_end=self._handle_slider_change_end,
        )
        self.threshold_text = ft.Text(
            f"{self.threshold:.2f}",
            size=14,
            weight=ft.FontWeight.BOLD,
            color=COLOR_NEUTRAL_DARK,
            width=40,
            text_align=ft.TextAlign.RIGHT,
        )
        self.control = self._build()

    def _build(self) -> ft.Column:
        graph = ft.Container(
            content=self.canvas,
            bgcolor=COLOR_BACKGROUND,
            border_radius=16,
            padding=ft.Padding.all(16),
            alignment=ft.Alignment.CENTER,
        )
        return ft.Column(
            controls=[
                ft.Text(
                    t(_CHANNEL_TITLE_KEYS[self.channel]),
                    size=18,
                    weight=ft.FontWeight.BOLD,
                    color=COLOR_SECONDARY,
                ),
                self.device_text,
                graph,
                ft.Row(
                    controls=[self.status_dot, self.status_text],
                    spacing=10,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Row(
                    controls=[
                        ft.Text(
                            t("audio_test.threshold"),
                            size=14,
                            weight=ft.FontWeight.BOLD,
                            color=COLOR_NEUTRAL_DARK,
                        ),
                        self.slider,
                        self.threshold_text,
                    ],
                    spacing=8,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                self.warning_text,
            ],
            spacing=12,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

    def set_threshold(self, value: float) -> None:
        self.threshold = float(value)
        self.slider.value = self.threshold
        self.slider.label = f"{self.threshold:.2f}"
        self.threshold_text.value = f"{self.threshold:.2f}"
        self.dirty = True

    def add_samples(self, samples: Sequence[object], now: float) -> None:
        for sample in samples:
            probability = float(getattr(sample, "probability", 0.0))
            level = float(getattr(sample, "level", 0.0))
            self.history.append((_clamp(probability), _clamp(level)))
        if samples:
            self.last_sample_at = now
            self.dirty = True

    def fill_silence(self, now: float) -> None:
        if self.last_sample_at is None or now - self.last_sample_at < SILENCE_FILL_AFTER_S:
            return
        missing = int((now - self.last_sample_at) / SAMPLE_INTERVAL_S)
        for _ in range(min(missing, HISTORY_SAMPLES)):
            self.history.append((0.0, 0.0))
        self.last_sample_at += missing * SAMPLE_INTERVAL_S
        self.dirty = True

    def apply_snapshot(self, snapshot: Mapping[str, object]) -> None:
        state = str(snapshot.get("state") or "off")
        device = snapshot.get("device")
        self.device_text.value = _device_label(device if isinstance(device, Mapping) else {})
        fallback = isinstance(device, Mapping) and bool(device.get("default_fallback"))
        self.warning_text.value = t("audio_test.device.default_fallback") if fallback else ""
        self.warning_text.visible = fallback
        if state == "failed":
            reason = str(snapshot.get("failure_reason") or "")
            key = (
                f"audio_test.error.{self.channel}_source_open_failed"
                if reason == "source_open_failed"
                else _FAILURE_KEYS.get(reason, "audio_test.error.generic")
            )
            self.status_text.value = t(key)
            self.status_text.color = COLOR_NEUTRAL_DARK
            self.status_dot.bgcolor = COLOR_WARNING
        elif state == "ready" and snapshot.get("speech_detected"):
            self.status_text.value = t("audio_test.state.speech")
            self.status_text.color = COLOR_PRIMARY
            self.status_dot.bgcolor = COLOR_PRIMARY
        elif state == "ready":
            self.status_text.value = t("audio_test.state.listening")
            self.status_text.color = COLOR_NEUTRAL_DARK
            self.status_dot.bgcolor = COLOR_NEUTRAL_DARK
        elif state in {"pending", "stopping"}:
            self.status_text.value = t("audio_test.state.pending")
            self.status_text.color = COLOR_NEUTRAL_DARK
            self.status_dot.bgcolor = COLOR_NEUTRAL_DARK
        else:
            self.status_text.value = t("audio_test.state.stopped")
            self.status_text.color = COLOR_NEUTRAL_DARK
            self.status_dot.bgcolor = COLOR_NEUTRAL_DARK

    def render(self) -> None:
        if not self.dirty:
            return
        self.dirty = False
        self.canvas.shapes = _graph_shapes(tuple(self.history), self.threshold)

    def _handle_slider_change(self, event) -> None:  # noqa: ANN001
        self.set_threshold(float(event.control.value))
        self.render()
        update_control_if_mounted(self.control)

    def _handle_slider_change_end(self, event) -> None:  # noqa: ANN001
        self.set_threshold(float(event.control.value))
        self._on_threshold_change(self.channel, self.threshold)


class AudioTestDialog:
    """Two-column live VAD view for the microphone and loopback inputs."""

    def __init__(
        self,
        page: ft.Page,
        *,
        on_close: Callable[[], None],
        on_threshold_change: Callable[[str, float], None],
        snapshot_provider: Callable[[], Mapping[str, object]],
        run_task: AudioTestRunTask,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._page = page
        self._on_close = on_close
        self._snapshot_provider = snapshot_provider
        self._run_task = run_task
        self._clock = clock
        self._panels = {
            channel: _ChannelPanel(channel, on_threshold_change=on_threshold_change)
            for channel in AUDIO_TEST_CHANNELS
        }
        self._notice_text = ft.Text(
            t("audio_test.capture_paused"),
            size=14,
            color=COLOR_NEUTRAL_DARK,
            visible=False,
        )
        self._content: ft.Container | None = None
        self._dialog: ft.AlertDialog | None = None
        self._is_open = False
        self._close_notified = False
        self._render_task: object | None = None
        self._start_failed = False

    @property
    def dialog(self) -> ft.AlertDialog | None:
        return self._dialog

    @property
    def is_open(self) -> bool:
        return self._is_open

    def panel(self, channel: str) -> _ChannelPanel:
        return self._panels[channel]

    def open(self, thresholds: Mapping[str, float] | None = None) -> None:
        if self._is_open:
            return
        self._start_failed = False
        for channel, panel in self._panels.items():
            panel.history.extend([(0.0, 0.0)] * HISTORY_SAMPLES)
            panel.last_sample_at = None
            if thresholds is not None and channel in thresholds:
                panel.set_threshold(thresholds[channel])
            panel.apply_snapshot({"state": "pending"})
            panel.dirty = True
            panel.render()
        self._notice_text.visible = False
        self._close_notified = False
        self._dialog = self._build_dialog()
        self._is_open = True
        self._page.show_dialog(self._dialog)
        self._render_task = self._run_task(self._render_loop)

    def close(self, *, notify: bool = False) -> None:
        dialog = self._dialog
        if dialog is None:
            self._close_notified = True
            return
        was_open = self._is_open
        self._is_open = False
        if notify:
            self._notify_close_once()
        else:
            self._close_notified = True
        if was_open:
            self._page.pop_dialog()

    def add_samples(self, channel: str, samples: Sequence[object]) -> None:
        panel = self._panels.get(channel)
        if panel is not None and self._is_open:
            panel.add_samples(samples, self._clock())

    def refresh(self) -> None:
        snapshot = self._snapshot_provider()
        channels = snapshot.get("channels")
        channels = channels if isinstance(channels, Mapping) else {}
        if any(
            isinstance(item, Mapping) and item.get("state") in {"pending", "ready", "failed"}
            for item in channels.values()
        ):
            self._start_failed = False
        now = self._clock()
        for channel, panel in self._panels.items():
            channel_snapshot = channels.get(channel)
            panel.apply_snapshot(
                {"state": "failed"}
                if self._start_failed
                else channel_snapshot if isinstance(channel_snapshot, Mapping) else {}
            )
            if isinstance(channel_snapshot, Mapping) and channel_snapshot.get("state") == "ready":
                panel.fill_silence(now)
            panel.render()
        restore = snapshot.get("restore_capture")
        self._notice_text.visible = bool(restore)
        if self._content is not None:
            update_control_if_mounted(self._content)

    def show_failure(self) -> None:
        self._start_failed = True
        for panel in self._panels.values():
            panel.apply_snapshot({"state": "failed"})
        if self._content is not None:
            update_control_if_mounted(self._content)

    async def _render_loop(self) -> None:
        while self._is_open:
            self.refresh()
            await asyncio.sleep(RENDER_INTERVAL_S)

    def _build_dialog(self) -> ft.AlertDialog:
        columns = ft.Row(
            controls=[panel.control for panel in self._panels.values()],
            spacing=32,
            vertical_alignment=ft.CrossAxisAlignment.START,
            expand=True,
        )
        self._content = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text(
                        t("settings.audio_test"),
                        size=24,
                        weight=ft.FontWeight.BOLD,
                        color=COLOR_SECONDARY,
                    ),
                    ft.Container(height=16),
                    columns,
                    self._notice_text,
                ],
                spacing=20,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            ),
            width=880,
            height=700,
            padding=ft.Padding.symmetric(horizontal=32, vertical=32),
            bgcolor=COLOR_SURFACE,
            border_radius=28,
        )
        return ft.AlertDialog(
            modal=False,
            content=self._content,
            content_padding=0,
            bgcolor=ft.Colors.TRANSPARENT,
            semantics_label=t("settings.audio_test"),
            on_dismiss=self._handle_dismiss,
        )

    def _handle_dismiss(self, _event) -> None:  # noqa: ANN001
        if not self._is_open:
            return
        self._is_open = False
        self._notify_close_once()

    def _notify_close_once(self) -> None:
        if self._close_notified:
            return
        self._close_notified = True
        self._on_close()


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _device_label(device: Mapping[str, object]) -> str:
    name = str(device.get("name") or "").strip()
    if not name:
        return t("audio_test.device.unknown")
    if device.get("kind") == "process":
        return t("audio_test.device.process", name=name)
    host_api = str(device.get("host_api") or "").strip()
    return f"{name} · {host_api}" if host_api else name


def _graph_shapes(history: Sequence[tuple[float, float]], threshold: float) -> list[cv.Shape]:
    count = len(history)
    step = _GRAPH_WIDTH / max(1, count - 1)

    def y(value: float) -> float:
        return _GRAPH_HEIGHT - value * _GRAPH_HEIGHT

    level_elements: list[cv.Path.PathElement] = [cv.Path.MoveTo(0, _GRAPH_HEIGHT)]
    probability_elements: list[cv.Path.PathElement] = []
    for index, (probability, level) in enumerate(history):
        x = index * step
        level_elements.append(cv.Path.LineTo(x, y(level)))
        if index == 0:
            probability_elements.append(cv.Path.MoveTo(x, y(probability)))
        else:
            probability_elements.append(cv.Path.LineTo(x, y(probability)))
    level_elements.append(cv.Path.LineTo(_GRAPH_WIDTH, _GRAPH_HEIGHT))
    level_elements.append(cv.Path.Close())
    threshold_y = y(_clamp(threshold))
    return [
        cv.Path(
            level_elements,
            paint=ft.Paint(
                color=ft.Colors.with_opacity(0.18, COLOR_NEUTRAL_DARK),
                style=ft.PaintingStyle.FILL,
            ),
        ),
        cv.Line(
            0,
            threshold_y,
            _GRAPH_WIDTH,
            threshold_y,
            paint=ft.Paint(
                color=COLOR_SECONDARY,
                stroke_width=2,
                stroke_dash_pattern=[8, 6],
            ),
        ),
        cv.Path(
            probability_elements,
            paint=ft.Paint(
                color=COLOR_PRIMARY,
                stroke_width=3,
                style=ft.PaintingStyle.STROKE,
            ),
        ),
    ]


__all__ = ["AudioTestDialog"]
