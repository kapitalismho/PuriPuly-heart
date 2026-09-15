from typing import Callable

import flet as ft

from puripuly_heart.ui.flet_runtime import (
    is_control_mounted,
    run_control_method,
    update_control_if_mounted,
)
from puripuly_heart.ui.i18n import t
from puripuly_heart.ui.theme import (
    COLOR_DISPLAY_SOURCE,
    COLOR_NEUTRAL_DARK,
    COLOR_SECONDARY,
    COLOR_SURFACE,
)

# CJK (Chinese, Japanese, Korean) characters start at this Unicode point.
_CJK_START = 0x3000

DISPLAY_SOURCE_COLOR = COLOR_DISPLAY_SOURCE
DISPLAY_MESSAGE_COLOR = COLOR_NEUTRAL_DARK
DISPLAY_SOURCE_WEIGHT = ft.FontWeight.W_500
DISPLAY_MESSAGE_WEIGHT = ft.FontWeight.BOLD

DISPLAY_CARD_PADDING = 32

DISPLAY_TEXT_WIDTH = 526
DISPLAY_TEXT_HEIGHT = 210
DISPLAY_LINE_HEIGHT_RATIO = 1.4
DISPLAY_WEIGHTED_UNIT_WIDTH_RATIO = 0.55
DISPLAY_SIZE_CANDIDATES: tuple[int, ...] = (48, 44, 40, 36, 32, 28, 24, 20, 16)


def _weighted_len(text: str) -> int:
    """Calculate weighted length for CJK-aware font sizing."""
    return sum(2 if ord(char) >= _CJK_START else 1 for char in text)


def _lines_for_size(size: int) -> int:
    return max(1, int(DISPLAY_TEXT_HEIGHT // (size * DISPLAY_LINE_HEIGHT_RATIO)))


def _capacity_for_size(size: int) -> int:
    units_per_line = DISPLAY_TEXT_WIDTH / (size * DISPLAY_WEIGHTED_UNIT_WIDTH_RATIO)
    return int(_lines_for_size(size) * units_per_line)


def _display_layout_for_length(length: int) -> tuple[int, int]:
    for size in DISPLAY_SIZE_CANDIDATES:
        if length <= _capacity_for_size(size):
            return size, _lines_for_size(size)
    smallest = DISPLAY_SIZE_CANDIDATES[-1]
    return smallest, _lines_for_size(smallest)


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
    """Single-state display card with input field.

    The card shows exactly one thing at a time. A turn moves through the source
    text and then the translation, replacing the same slot rather than stacking
    two lines. Colour distinguishes the two.
    """

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
        self._source_value = _status_label(self._status)
        self._translation_value: str | None = None
        self._source_font_family: str | None = None
        self._translation_font_family: str | None = None
        self._debug_prefix: str | None = None
        self._notice_value: str | None = None
        self._notice_action: Callable[[], None] | None = None
        self._notice_yields_to_content = False
        self._turn_size_cap: int | None = None
        self._source_as_message = False
        self._translation_is_visible = False
        self.input_is_focused = False

        initial_size, initial_max_lines = _display_layout_for_length(
            _weighted_len(self._source_value)
        )
        self._display_text = ft.Text(
            self._source_value,
            size=initial_size,
            weight=DISPLAY_MESSAGE_WEIGHT,
            color=DISPLAY_MESSAGE_COLOR,
            selectable=True,
            no_wrap=False,
            max_lines=initial_max_lines,
            overflow=ft.TextOverflow.ELLIPSIS,
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
                    ft.Container(content=self._display_text, expand=True),
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
                    content=ft.Divider(
                        height=1, color=ft.Colors.with_opacity(0.2, COLOR_SECONDARY)
                    ),
                    padding=ft.Padding.only(bottom=4),
                ),
                ft.Row(
                    [self._input_field],
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

        content_container = ft.Container(
            content=main_content, expand=True, padding=DISPLAY_CARD_PADDING
        )

        super().__init__(
            content=content_container,
            bgcolor=COLOR_SURFACE,
            border_radius=16,
            expand=True,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
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
        debug_prefix: str | None = None,
        as_translation: bool = False,
    ):
        """Show source text and start a new turn, dropping any prior translation."""
        _ = is_error
        self._showing_status = False
        self._source_value = text
        self._source_font_family = font_family
        self._source_as_message = bool(as_translation)
        self._translation_value = None
        self._translation_font_family = None
        self._debug_prefix = debug_prefix
        self._turn_size_cap = _display_layout_for_length(
            _weighted_len(_apply_debug_prefix(text or "", debug_prefix))
        )[0]
        self._sync_display()

    def set_display_translation(
        self,
        text: str | None,
        font_family: str | None = None,
        *,
        debug_prefix: str | None = None,
    ) -> None:
        """Replace the visible slot with the translation."""
        self._showing_status = False
        self._source_as_message = False
        self._translation_value = text or None
        self._translation_font_family = font_family if text else None
        self._debug_prefix = debug_prefix
        self._sync_display()

    def set_status(self, status: str, font_family: str | None = None):
        """Update connection status display."""
        self._status = status
        self._showing_status = True
        self._source_as_message = False
        self._source_value = _status_label(status)
        self._source_font_family = font_family
        self._translation_value = None
        self._translation_font_family = None
        self._debug_prefix = None
        self._turn_size_cap = None
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
        _ = tone
        self._notice_value = text or None
        self._notice_yields_to_content = bool(yields_to_content) if self._notice_value else False
        self._notice_action = on_action if self._notice_value and action_label else None
        self._notice_action_button.content = action_label or ""
        self._notice_action_button.visible = self._notice_action is not None
        self._sync_display()
        update_control_if_mounted(self._notice_action_button)

    def _run_notice_action(self) -> None:
        action = self._notice_action
        if action is not None:
            action()

    def clear_input(self):
        """Clear the input field."""
        self._input_field.value = ""
        update_control_if_mounted(self._input_field)

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
            self._source_value = _status_label(self._status)
            self._source_font_family = display_font_family
            self._translation_value = None
            self._translation_font_family = None
            self._sync_display()

            return

    def _sync_display(self) -> tuple[bool, None]:
        notice_blocks = self._notice_value is not None and (
            not self._notice_yields_to_content or self._showing_status
        )
        if notice_blocks:
            visible_text = self._notice_value or ""
            font_family = self._source_font_family
            showing_translation = False
            showing_source = False
        elif self._translation_value:
            visible_text = _apply_debug_prefix(self._translation_value, self._debug_prefix)
            font_family = self._translation_font_family
            showing_translation = True
            showing_source = False
        else:
            visible_text = _apply_debug_prefix(self._source_value or "", self._debug_prefix)
            font_family = self._source_font_family
            showing_translation = False
            showing_source = not self._showing_status

        self._translation_is_visible = showing_translation
        new_size, new_max_lines = _display_layout_for_length(_weighted_len(visible_text))
        if showing_translation and self._turn_size_cap is not None:
            new_size = min(self._turn_size_cap, new_size)
            new_max_lines = _lines_for_size(new_size)

        self._display_text.value = visible_text
        self._display_text.size = new_size
        self._display_text.max_lines = new_max_lines
        show_as_message = showing_translation or not showing_source or self._source_as_message
        self._display_text.color = (
            DISPLAY_MESSAGE_COLOR if show_as_message else DISPLAY_SOURCE_COLOR
        )
        self._display_text.weight = (
            DISPLAY_MESSAGE_WEIGHT if show_as_message else DISPLAY_SOURCE_WEIGHT
        )
        self._display_text.font_family = font_family

        display_update_issued = is_control_mounted(self._display_text)

        if display_update_issued:
            self._display_text.update()

        return display_update_issued, None
