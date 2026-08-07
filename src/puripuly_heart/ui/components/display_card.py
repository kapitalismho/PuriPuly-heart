import logging
import time
from typing import Callable

import flet as ft

from puripuly_heart.ui.flet_runtime import (
    is_control_mounted,
    run_control_method,
    update_control_if_mounted,
)
from puripuly_heart.ui.i18n import t
from puripuly_heart.ui.theme import (
    COLOR_NEUTRAL,
    COLOR_NEUTRAL_DARK,
    COLOR_SECONDARY,
    COLOR_SURFACE,
    COLOR_WARNING,
    get_card_shadow,
)

# CJK (Chinese, Japanese, Korean) characters start at this Unicode point.
_CJK_START = 0x3000


def _weighted_len(text: str) -> int:
    """Calculate weighted length for CJK-aware font sizing."""
    return sum(2 if ord(char) >= _CJK_START else 1 for char in text)


def _display_size_for_length(length: int) -> int:
    if length <= 12:
        return 48
    if length <= 20:
        return 40
    if length <= 32:
        return 34
    if length <= 44:
        return 28
    return 24


def _status_label(status: str) -> str:
    if status == "connecting":
        return t("display.connecting")
    if status == "connected":
        return t("display.connected")
    if status == "stopping":
        return t("display.stopping")
    return t("display.disconnected")


def _apply_debug_prefix(text: str, debug_prefix: str | None) -> str:
    prefix = (debug_prefix or "").strip()
    if not prefix or not text:
        return text
    return f"{prefix} {text}"


class DisplayCard(ft.Container):
    """Multi-purpose display card with input field and decorative gradient."""

    def __init__(
        self,
        on_submit: Callable[[str], None],
        on_input_focus_change: Callable[[bool], None] | None = None,
        on_input_activity: Callable[[bool], None] | None = None,
    ):
        self._on_submit = on_submit
        self._on_input_focus_change = on_input_focus_change
        self._on_input_activity = on_input_activity
        self._status = "disconnected"
        self._showing_status = True
        self._primary_value = _status_label(self._status)
        self._secondary_value: str | None = None
        self._primary_font_family: str | None = None
        self._secondary_font_family: str | None = None
        self._debug_prefix: str | None = None
        self._notice_value: str | None = None
        self._notice_tone: str | None = None
        self._notice_action: Callable[[], None] | None = None
        self._notice_yields_to_content = False
        self.input_is_focused = False

        self._display_primary = ft.Text(
            self._primary_value,
            size=48,
            weight=ft.FontWeight.BOLD,
            color=COLOR_NEUTRAL_DARK,
            selectable=True,
            no_wrap=False,
            max_lines=2,
            overflow=ft.TextOverflow.ELLIPSIS,
        )

        self._display_secondary = ft.Text(
            "",
            size=48,
            weight=ft.FontWeight.BOLD,
            color=COLOR_NEUTRAL_DARK,
            selectable=True,
            no_wrap=False,
            max_lines=2,
            overflow=ft.TextOverflow.ELLIPSIS,
            visible=False,
        )

        self._notice_text = ft.Text(
            "",
            size=13,
            weight=ft.FontWeight.W_600,
            color=ft.Colors.WHITE,
        )
        self._notice_chip = ft.Container(
            content=self._notice_text,
            visible=False,
            bgcolor=COLOR_WARNING,
            border_radius=999,
            padding=ft.Padding.symmetric(horizontal=10, vertical=6),
        )
        self._notice_action_button = ft.TextButton(
            content="",
            visible=False,
            height=36,
            on_click=lambda _event: self._run_notice_action(),
        )

        self._input_field = ft.TextField(
            hint_text=t("display.input_hint"),
            border=ft.InputBorder.NONE,
            text_size=20,
            color=COLOR_NEUTRAL_DARK,
            hint_style=ft.TextStyle(color=COLOR_SECONDARY, italic=True),
            expand=True,
            on_submit=self._handle_submit,
            on_change=self._handle_input_change,
            on_focus=self._handle_input_focus,
            on_blur=self._handle_input_blur,
        )

        display_region = ft.Container(
            content=ft.Row(
                [
                    ft.Container(
                        content=ft.Column(
                            [
                                self._display_primary,
                                self._display_secondary,
                            ],
                            spacing=4,
                            tight=True,
                        ),
                        expand=True,
                    ),
                    self._notice_action_button,
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.START,
            ),
            alignment=ft.Alignment.TOP_LEFT,
            padding=ft.Padding.only(left=8),
            expand=True,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        )
        input_footer = ft.Column(
            [
                ft.Container(
                    content=ft.Divider(height=1, color=ft.Colors.with_opacity(0.2, COLOR_NEUTRAL)),
                    padding=ft.Padding.only(bottom=4),
                ),
                ft.Row(
                    [
                        ft.Container(
                            content=ft.Text("•", size=36, color="#FFADAC"),
                            padding=ft.Padding.only(right=8),
                        ),
                        self._input_field,
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            ],
            spacing=0,
            tight=True,
        )

        main_content = ft.Column(
            [display_region, input_footer],
            expand=True,
            alignment=ft.MainAxisAlignment.START,
            spacing=8,
        )

        # The content container handles the internal padding (32)
        content_container = ft.Container(content=main_content, expand=True, padding=32)

        super().__init__(
            content=content_container,
            bgcolor=COLOR_SURFACE,
            border_radius=16,
            border=ft.Border.all(1, ft.Colors.with_opacity(0.4, ft.Colors.WHITE)),
            expand=True,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
            shadow=get_card_shadow(),
        )

    def _handle_submit(self, e):
        text = e.control.value.strip()
        if text:
            self._emit_input_activity(False)
            self._on_submit(text)
            e.control.value = ""
            e.control.update()
            run_control_method(e.control, "focus")

    def _handle_input_change(self, e) -> None:
        self._emit_input_activity(bool((e.control.value or "").strip()))

    def _handle_input_focus(self, _e) -> None:
        self._set_input_focus(True)

    def _handle_input_blur(self, _e) -> None:
        self._set_input_focus(False)
        self._emit_input_activity(False)

    def _set_input_focus(self, focused: bool) -> None:
        self.input_is_focused = bool(focused)
        if self._on_input_focus_change is not None:
            self._on_input_focus_change(self.input_is_focused)

    def _emit_input_activity(self, has_text: bool) -> None:
        if self._on_input_activity is not None:
            self._on_input_activity(bool(has_text))

    def focus_input(self) -> None:
        run_control_method(self._input_field, "focus")

    def set_display(
        self,
        text: str,
        is_error: bool = False,
        font_family: str | None = None,
        *,
        runtime_log_detailed: Callable[..., bool | None] | None = None,
        update_id: str | None = None,
        origin_wall_clock_ms: int | None = None,
        utterance_id: object | None = None,
        channel: str | None = None,
        source_text_len: int | None = None,
        transcript_kind: str | None = None,
        should_log: bool = False,
        debug_prefix: str | None = None,
    ):
        """Update the primary display text and clear any secondary text."""
        self._showing_status = False
        self._primary_value = text
        self._primary_font_family = font_family
        self._secondary_value = None
        self._secondary_font_family = None
        self._debug_prefix = debug_prefix
        measure = should_log and runtime_log_detailed is not None
        primary_update_issued, _, flet_update_elapsed_us = self._sync_display(
            is_error=is_error, measure_flet_update=measure
        )
        if should_log:
            self._emit_dashboard_primary_applied(
                runtime_log_detailed=runtime_log_detailed,
                update_id=update_id,
                origin_wall_clock_ms=origin_wall_clock_ms,
                utterance_id=utterance_id,
                channel=channel,
                source_text_len=source_text_len,
                transcript_kind=transcript_kind,
                primary_update_issued=primary_update_issued,
                flet_update_elapsed_us=flet_update_elapsed_us,
            )

    def set_display_translation(
        self,
        text: str | None,
        font_family: str | None = None,
        *,
        runtime_log_detailed: Callable[..., bool | None] | None = None,
        update_id: str | None = None,
        origin_wall_clock_ms: int | None = None,
        utterance_id: object | None = None,
        channel: str | None = None,
        session_scope: str | None = None,
        source_text_hash: str | None = None,
        source_text_len: int | None = None,
        logical_turn_key: str | None = None,
        debug_prefix: str | None = None,
    ) -> None:
        """Update the secondary display text and emit a post-update visual commit marker."""
        self._showing_status = False
        self._secondary_value = text or None
        self._secondary_font_family = font_family if text else None
        self._debug_prefix = debug_prefix
        measure = runtime_log_detailed is not None
        (
            primary_update_issued,
            secondary_update_issued,
            flet_update_elapsed_us,
        ) = self._sync_display(measure_flet_update=measure)
        self._emit_dashboard_translation_visual_commit(
            runtime_log_detailed=runtime_log_detailed,
            update_id=update_id,
            origin_wall_clock_ms=origin_wall_clock_ms,
            utterance_id=utterance_id,
            channel=channel,
            session_scope=session_scope,
            source_text_hash=source_text_hash,
            source_text_len=source_text_len,
            logical_turn_key=logical_turn_key,
            primary_update_issued=primary_update_issued,
            secondary_update_issued=secondary_update_issued,
            flet_update_elapsed_us=flet_update_elapsed_us,
        )

    def set_status(self, status: str, font_family: str | None = None):
        """Update connection status display."""
        self._status = status
        self._showing_status = True
        self._primary_value = _status_label(status)
        self._primary_font_family = font_family
        self._secondary_value = None
        self._secondary_font_family = None
        self._debug_prefix = None
        self._sync_display()

    def set_notice(
        self,
        text: str | None,
        tone: str | None = None,
        *,
        action_label: str | None = None,
        on_action: Callable[[], None] | None = None,
        yields_to_content: bool = False,
    ) -> None:
        self._notice_value = text or None
        self._notice_tone = tone if self._notice_value else None
        self._notice_yields_to_content = bool(yields_to_content) if self._notice_value else False
        self._notice_action = on_action if self._notice_value and action_label else None
        self._notice_action_button.content = action_label or ""
        self._notice_action_button.visible = self._notice_action is not None
        self._notice_text.value = ""
        self._notice_chip.visible = False
        self._sync_display()
        update_control_if_mounted(self._notice_action_button)

    def _run_notice_action(self) -> None:
        action = self._notice_action
        if action is not None:
            action()

    def clear_input(self):
        """Clear the input field."""
        self._input_field.value = ""
        self._input_field.update()

    def set_input_font(self, font_family: str | None) -> None:
        # Force strict system fallback if None (to break theme inheritance)
        final_font = font_family if font_family else ""
        self._input_field.text_style = ft.TextStyle(font_family=final_font)
        # Hint style is now managed separately by apply_locale (using UI font)
        update_control_if_mounted(self._input_field)

    def apply_locale(
        self,
        *,
        display_font_family: str | None = None,
        input_font_family: str | None = None,
    ) -> None:
        self._input_field.hint_text = t("display.input_hint")

        # Explicitly set hint font to UI font (Display font)
        self._input_field.hint_style = ft.TextStyle(
            color=COLOR_SECONDARY,
            italic=True,
            font_family=display_font_family,
        )

        if input_font_family is not None:
            self.set_input_font(input_font_family)
        else:
            update_control_if_mounted(self._input_field)
        if self._showing_status:
            self._primary_value = _status_label(self._status)
            self._primary_font_family = display_font_family
            self._secondary_value = None
            self._secondary_font_family = None
            self._sync_display()

    def _emit_dashboard_translation_visual_commit(
        self,
        *,
        runtime_log_detailed: Callable[..., bool | None] | None,
        update_id: str | None,
        origin_wall_clock_ms: int | None,
        utterance_id: object | None,
        channel: str | None,
        session_scope: str | None,
        source_text_hash: str | None,
        source_text_len: int | None,
        logical_turn_key: str | None,
        primary_update_issued: bool,
        secondary_update_issued: bool,
        flet_update_elapsed_us: int | None = None,
    ) -> None:
        if runtime_log_detailed is None or update_id is None:
            return
        if not (primary_update_issued or secondary_update_issued):
            return
        if not self._display_secondary.visible or not self._display_secondary.value:
            return

        elapsed_ms = None
        if origin_wall_clock_ms is not None:
            elapsed_ms = max(0, int(time.time() * 1000) - origin_wall_clock_ms)

        parts = [
            "[Detailed][DisplayCard] dashboard_translation_visual_commit",
            f"utterance_id={utterance_id}",
            f"channel={channel}",
            f"update_id={update_id}",
            f"origin_wall_clock_ms={origin_wall_clock_ms}",
            f"session_scope={session_scope}",
            f"source_text_hash={source_text_hash}",
            f"source_text_len={source_text_len}",
            f"logical_turn_key={logical_turn_key}",
            f"primary_text_len={len(self._display_primary.value or '')}",
            f"secondary_text_len={len(self._display_secondary.value or '')}",
            f"secondary_visible={self._display_secondary.visible}",
            f"showing_status={self._showing_status}",
            f"primary_update_issued={primary_update_issued}",
            f"secondary_update_issued={secondary_update_issued}",
        ]
        if elapsed_ms is not None:
            parts.append(f"elapsed_ms={elapsed_ms}")
        if flet_update_elapsed_us is not None:
            parts.append(f"flet_update_elapsed_us={flet_update_elapsed_us}")

        try:
            runtime_log_detailed(" ".join(parts), level=logging.INFO)
        except Exception:
            return

    def _emit_dashboard_primary_applied(
        self,
        *,
        runtime_log_detailed: Callable[..., bool | None] | None,
        update_id: str | None,
        origin_wall_clock_ms: int | None,
        utterance_id: object | None,
        channel: str | None,
        source_text_len: int | None,
        transcript_kind: str | None,
        primary_update_issued: bool,
        flet_update_elapsed_us: int | None = None,
    ) -> None:
        if runtime_log_detailed is None:
            return
        if not primary_update_issued:
            return

        elapsed_ms = None
        if origin_wall_clock_ms is not None:
            elapsed_ms = max(0, int(time.time() * 1000) - origin_wall_clock_ms)

        parts = [
            "[Detailed][DisplayCard] dashboard_primary_applied",
            f"utterance_id={utterance_id}",
            f"channel={channel}",
            f"update_id={update_id if update_id is not None else 'none'}",
            f"origin_wall_clock_ms={origin_wall_clock_ms}",
            f"transcript_kind={transcript_kind}",
            f"source_text_len={source_text_len}",
            f"primary_text_len={len(self._display_primary.value or '')}",
            f"showing_status={self._showing_status}",
            f"primary_update_issued={primary_update_issued}",
        ]
        if elapsed_ms is not None:
            parts.append(f"elapsed_ms={elapsed_ms}")
        if flet_update_elapsed_us is not None:
            parts.append(f"flet_update_elapsed_us={flet_update_elapsed_us}")

        try:
            runtime_log_detailed(" ".join(parts), level=logging.INFO)
        except Exception:
            return

    def _sync_display(
        self, *, is_error: bool = False, measure_flet_update: bool = False
    ) -> tuple[bool, bool, int | None]:
        notice_blocks = self._notice_value is not None and (
            not self._notice_yields_to_content or self._showing_status
        )
        notice_text = self._notice_value if notice_blocks else None
        primary_text = (
            notice_text
            if notice_text is not None
            else _apply_debug_prefix(self._primary_value or "", self._debug_prefix)
        )
        secondary_text = (
            ""
            if notice_text is not None
            else _apply_debug_prefix(self._secondary_value or "", self._debug_prefix)
        )
        max_len = max(_weighted_len(primary_text), _weighted_len(secondary_text))
        new_size = _display_size_for_length(max_len)

        text_color = COLOR_NEUTRAL_DARK

        self._display_primary.value = primary_text
        self._display_primary.size = new_size
        self._display_primary.color = text_color
        self._display_primary.font_family = self._primary_font_family

        self._display_secondary.value = secondary_text
        self._display_secondary.visible = bool(self._secondary_value)
        self._display_secondary.size = new_size
        self._display_secondary.color = text_color
        self._display_secondary.font_family = self._secondary_font_family

        primary_update_issued = is_control_mounted(self._display_primary)
        secondary_update_issued = is_control_mounted(self._display_secondary)

        flet_update_elapsed_us: int | None = None
        start_ns = time.perf_counter_ns() if measure_flet_update else 0

        if primary_update_issued:
            self._display_primary.update()
        if secondary_update_issued:
            self._display_secondary.update()

        if measure_flet_update:
            flet_update_elapsed_us = max(0, (time.perf_counter_ns() - start_ns) // 1000)

        return primary_update_issued, secondary_update_issued, flet_update_elapsed_us
