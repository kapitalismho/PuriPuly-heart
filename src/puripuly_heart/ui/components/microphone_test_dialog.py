from __future__ import annotations

from collections.abc import Callable

import flet as ft

from puripuly_heart.ui.flet_runtime import is_control_mounted
from puripuly_heart.ui.fonts import font_for_language
from puripuly_heart.ui.i18n import get_locale, t
from puripuly_heart.ui.theme import (
    COLOR_NEUTRAL_DARK,
    COLOR_PRIMARY,
    COLOR_SURFACE,
)

_DIALOG_WIDTH = 450
_DIALOG_HEIGHT = 500
_CONTENT_SIZE = 386
_PERCENT_TEXT_SIZE = 96
_FAILURE_TEXT_SIZE = 24
_HINT_TEXT_SIZE = 28


def _clamp_level(value: float) -> float:
    level = max(0.0, min(1.0, float(value)))
    if level <= 1e-6:
        return 0.0
    return level


def _level_percent(value: float) -> int:
    return int(round(_clamp_level(value) * 100))


class MicrophoneTestDialog:
    """Minimal microphone-test modal with a large live percentage readout."""

    def __init__(
        self,
        page: ft.Page,
        *,
        on_close: Callable[[], None],
        on_record: Callable[[], None] | None = None,
        on_stop: Callable[[], None] | None = None,
        on_play: Callable[[bool], None] | None = None,
        on_stop_playback: Callable[[], None] | None = None,
    ) -> None:
        self._page = page
        self._on_close = on_close
        self._on_record = on_record
        self._on_stop = on_stop
        self._on_play = on_play
        self._on_stop_playback = on_stop_playback
        self._record_status = ft.Text(
            t("audio_test.hint"), selectable=True, size=14, color=COLOR_NEUTRAL_DARK
        )
        self._play_raw = ft.Button(
            t("audio_test.play_raw"), on_click=lambda _: on_play(False), disabled=True
        )
        self._play_processed = ft.Button(
            t("audio_test.play_processed"), on_click=lambda _: on_play(True), disabled=True
        )
        self._dialog: ft.AlertDialog | None = None
        self._level_text: ft.Text | None = None
        self._hint_text: ft.Text | None = None
        self._level = 0.0
        self._failed = False
        self._failure_message = None
        self._is_open = False
        self._close_notified = False

    @property
    def dialog(self) -> ft.AlertDialog | None:
        return self._dialog

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self) -> None:
        if self._is_open:
            return
        self._close_notified = False
        self._dialog = self._build_dialog()
        self._is_open = True
        self._page.show_dialog(self._dialog)

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

    def reset(self) -> None:
        self._level = 0.0
        self._failed = False
        self._failure_message = None
        self._sync_text()

    def set_level(self, value: float) -> None:
        self._level = _clamp_level(value)
        self._failed = False
        self._sync_text()

    def show_failure(self, code: str | None = None) -> None:
        self._level = 0.0
        self._failed = True
        self._failure_message = (
            t("audio_test.error." + code, default=t("audio_test.error.capture_failed"))
            if code
            else None
        )
        self._sync_text()

    def set_snapshot(self, snapshot: dict) -> None:
        device = snapshot.get("actual_device", {})
        name = device.get("name") or t("audio_test.unknown_device")
        api = device.get("host_api") or "?"
        state = (
            "playing"
            if snapshot.get("playing")
            else "recorded" if snapshot.get("recording_complete") else "recording"
        )
        self._record_status.value = f"{t('audio_test.state.' + state)} · {snapshot.get('recorded_seconds', 0):.1f}s\n{name} · {api}\n{snapshot.get('sample_rate') or '?'} Hz · {snapshot.get('channels') or '?'} ch\n{t('audio_test.hint')}"
        failure = snapshot.get("failure_reason")
        if not failure and (snapshot.get("playing") or snapshot.get("recording_complete")):
            self._failed = False
            self._failure_message = None
            self._sync_text()
        if failure:
            self._record_status.value += "\n" + t(
                "audio_test.error." + failure, default=t("audio_test.error.capture_failed")
            )
        if (
            snapshot.get("recording_complete")
            and snapshot.get("has_recording")
            and snapshot.get("peak", 0) <= 1e-6
        ):
            self._record_status.value += "\n" + t("audio_test.silence")
        can_play = bool(
            snapshot.get("has_recording")
            and snapshot.get("recording_complete")
            and not snapshot.get("playing")
        )
        self._play_raw.disabled = self._play_processed.disabled = not can_play
        if self.is_open and is_control_mounted(self._record_status):
            self._page.update()

    def _build_dialog(self) -> ft.AlertDialog:
        self._level_text = ft.Text(
            self._text_value(),
            size=self._text_size(),
            weight=ft.FontWeight.BOLD,
            color=self._text_color(),
            text_align=ft.TextAlign.CENTER,
            font_family=font_for_language(get_locale()),
            semantics_label=t("settings.microphone_test.level_label"),
        )
        self._hint_text = ft.Text(
            t("settings.microphone_test.host_api_hint"),
            size=_HINT_TEXT_SIZE,
            color=COLOR_NEUTRAL_DARK,
            text_align=ft.TextAlign.CENTER,
            font_family=font_for_language(get_locale()),
        )

        modal_content = ft.Container(
            width=_DIALOG_WIDTH,
            height=_DIALOG_HEIGHT,
            padding=28,
            bgcolor=COLOR_SURFACE,
            border_radius=30,
            alignment=ft.Alignment.CENTER,
            content=ft.Column(
                controls=[
                    ft.Container(
                        width=_CONTENT_SIZE,
                        content=self._level_text,
                        alignment=ft.Alignment.CENTER,
                        bgcolor=ft.Colors.TRANSPARENT,
                        expand=True,
                    ),
                    self._hint_text,
                ],
                spacing=12,
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                tight=False,
            ),
        )
        if self._on_record is not None:
            self._hint_text.size = 14
            modal_content.content.controls.extend(
                [
                    self._record_status,
                    ft.Row(
                        [
                            ft.Button(t("audio_test.record"), on_click=lambda _: self._on_record()),
                            ft.Button(t("audio_test.stop"), on_click=lambda _: self._on_stop()),
                        ],
                        wrap=True,
                    ),
                    ft.Row([self._play_raw, self._play_processed], wrap=True),
                    ft.Row(
                        [
                            ft.Button(
                                t("audio_test.stop_playback"),
                                on_click=lambda _: self._on_stop_playback(),
                            ),
                            ft.TextButton(
                                t("audio_test.close"), on_click=lambda _: self.close(notify=True)
                            ),
                        ],
                        wrap=True,
                    ),
                ]
            )
            modal_content.content.controls[0].expand = False
            modal_content.content.controls[0].height = 110
            modal_content.height = 640

        return ft.AlertDialog(
            modal=False,
            content=modal_content,
            content_padding=0,
            bgcolor=ft.Colors.TRANSPARENT,
            semantics_label=t("settings.microphone_test"),
            on_dismiss=self._handle_dismiss,
        )

    def _text_value(self) -> str:
        if self._failed:
            return self._failure_message or t("settings.microphone_test.start_failed")
        return f"{_level_percent(self._level)}%"

    def _text_size(self) -> int:
        return _FAILURE_TEXT_SIZE if self._failed else _PERCENT_TEXT_SIZE

    def _text_color(self) -> str:
        return COLOR_NEUTRAL_DARK if self._failed else COLOR_PRIMARY

    def _sync_text(self) -> None:
        if self._level_text is None:
            return
        self._level_text.value = self._text_value()
        self._level_text.size = self._text_size()
        self._level_text.color = self._text_color()
        if not is_control_mounted(self._level_text):
            return
        try:
            self._level_text.update()
        except AssertionError as exc:
            if "Control must be added" not in str(exc):
                raise

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
