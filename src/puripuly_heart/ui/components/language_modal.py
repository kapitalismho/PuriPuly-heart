"""Language selection modal component.

Provides a reusable modal dialog for selecting source or target language
with recent language chips and a scrollable full list.
"""

import logging
from typing import Callable, Sequence

import flet as ft

from puripuly_heart.app.language_selection import RECENT_LANGUAGE_QUEUE_LIMIT
from puripuly_heart.ui.flet_runtime import is_hover_active
from puripuly_heart.ui.i18n import language_name, t
from puripuly_heart.ui.theme import (
    COLOR_BACKGROUND,
    COLOR_NEUTRAL_DARK,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SURFACE,
)

logger = logging.getLogger(__name__)

RECENT_CHIP_WIDTH = 170
RECENT_CHIP_SPACING = 10
RECENT_CHIP_RADIUS = 16
RECENT_CHIP_TEXT_SIZE = 22
_FORMER_RECENT_ROWS = 2
_FORMER_RECENT_TEXT_SIZE = 16
_FORMER_RECENT_VERTICAL_PADDING = 16
_FORMER_RECENT_RUN_SPACING = 10
_FORMER_RECENT_CHIP_HEIGHT = _FORMER_RECENT_TEXT_SIZE + (2 * _FORMER_RECENT_VERTICAL_PADDING)
RECENT_CHIP_HEIGHT = (
    _FORMER_RECENT_ROWS * _FORMER_RECENT_CHIP_HEIGHT
    + (_FORMER_RECENT_ROWS - 1) * _FORMER_RECENT_RUN_SPACING
)


def build_recent_language_chip(
    label: str,
    *,
    on_click: Callable,
    on_hover: Callable | None = None,
    selected: bool = False,
    data: str | None = None,
) -> ft.Container:
    """One recent-language chip. Three of these fill the old six-chip block."""
    return ft.Container(
        data=data,
        content=ft.Text(
            label,
            size=RECENT_CHIP_TEXT_SIZE,
            weight=ft.FontWeight.BOLD,
            color=ft.Colors.WHITE if selected else COLOR_NEUTRAL_DARK,
            text_align=ft.TextAlign.CENTER,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
        ),
        bgcolor=COLOR_PRIMARY if selected else COLOR_BACKGROUND,
        border_radius=RECENT_CHIP_RADIUS,
        padding=ft.Padding.symmetric(horizontal=8, vertical=16),
        alignment=ft.Alignment.CENTER,
        on_click=on_click,
        on_hover=on_hover,
        animate=ft.Animation(200, ft.AnimationCurve.EASE_OUT),
        width=RECENT_CHIP_WIDTH,
        height=RECENT_CHIP_HEIGHT,
    )


def apply_recent_language_chip_hover(event: ft.ControlEvent) -> None:
    """Match the language-modal hover: primary text, leaving a selected chip alone."""
    container = event.control
    text_control = container.content
    if getattr(text_control, "color", None) == ft.Colors.WHITE:
        return
    text_control.color = COLOR_PRIMARY if is_hover_active(event) else COLOR_NEUTRAL_DARK
    container.update()


class LanguageModal:
    """Modal dialog for language selection.

    Features:
    - Recent languages as chips, newest first, up to three
    - Full language list with current selection highlighted
    - Closes on selection or outside click
    """

    def __init__(
        self,
        page: ft.Page,
        languages: Sequence[tuple[str, str]],
        on_select: Callable[[str], None],
        label_for_code: Callable[[str], str] | None = None,
        description_for_code: Callable[[str], str] | None = None,
        disabled_codes: set[str] | None = None,
    ):
        """Initialize language modal.

        Args:
            page: Flet page for dialog management.
            languages: List of (code, name) tuples.
            on_select: Callback when a language is selected (receives code).
            label_for_code: Optional callable to resolve display label from code.
            description_for_code: Optional callable to resolve description text
                from code. When a non-empty string is returned, it is rendered
                below the language label in the list item.
            disabled_codes: Optional set of codes that should be rendered as
                disabled (not clickable, muted style).
        """
        self._page = page
        self._languages = languages
        self._on_select = on_select
        self._label_for_code = label_for_code or language_name
        self._description_for_code = description_for_code
        self._disabled_codes = disabled_codes or set()
        self._dialog: ft.AlertDialog | None = None

    def open(
        self,
        current: str,
        recent: list[str],
    ) -> None:
        """Open the language selection dialog.

        Args:
            current: Currently selected language code.
            recent: Recently selected language codes, newest first. The queue
                holds three codes, and the dialog shows them in that order.
        """
        recent_grid = self._build_recent_grid(recent, current)

        # Build language list (1-column grid style)
        lang_list = self._build_language_list(current)

        # Content column
        content_controls: list[ft.Control] = []

        # Header for Recent
        content_controls.append(
            ft.Text(
                t("language_modal.recent"),
                size=18,
                weight=ft.FontWeight.BOLD,
                color=COLOR_SECONDARY,
            )
        )

        # Recent Chips (Cards)
        if recent_grid:
            content_controls.append(recent_grid)
            content_controls.append(ft.Container(height=12))  # Spacer

        # Header for All Languages
        content_controls.append(
            ft.Text(
                t("language_modal.all_languages"),
                size=18,
                weight=ft.FontWeight.BOLD,
                color=COLOR_SECONDARY,
            )
        )

        content_controls.append(lang_list)

        modal_content = ft.Container(
            content=ft.Column(
                content_controls,
                spacing=8,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,  # Stretch width
            ),
            width=600,  # Larger width
            height=700,
            padding=ft.Padding.symmetric(horizontal=32, vertical=32),
            bgcolor=COLOR_SURFACE,
            border_radius=28,
        )

        # Transparent AlertDialog, content handles the background
        self._dialog = ft.AlertDialog(
            modal=False,
            content=modal_content,
            content_padding=0,
            bgcolor=ft.Colors.TRANSPARENT,
        )

        self._page.show_dialog(self._dialog)

    def _build_recent_grid(self, recent: list[str], current: str) -> ft.Control | None:
        """Build the recent chips in stored order, up to the queue limit.

        Three chips fill the block previously used by six chips in two rows.
        """
        visible = recent[:RECENT_LANGUAGE_QUEUE_LIMIT]
        if not visible:
            return None

        items = []
        for lang_code in visible:
            chip = build_recent_language_chip(
                self._label_for_code(lang_code),
                on_click=lambda e, code=lang_code: self._select(code),
                on_hover=self._on_chip_hover,
                selected=lang_code == current,
            )
            items.append(chip)

        return ft.Row(
            controls=items,
            wrap=False,
            spacing=RECENT_CHIP_SPACING,
            alignment=ft.MainAxisAlignment.START,
        )

    def _build_language_list(self, current: str) -> ft.ListView:
        """Build scrollable list of all languages (Bento Card Style)."""
        items = []
        for code, _name in self._languages:
            is_disabled = code in self._disabled_codes
            is_selected = code == current and not is_disabled

            # Colors
            if is_disabled:
                bg_color = COLOR_BACKGROUND
                text_color = ft.Colors.with_opacity(0.35, COLOR_NEUTRAL_DARK)
                desc_color = ft.Colors.with_opacity(0.35, COLOR_NEUTRAL_DARK)
                border = None
            else:
                bg_color = COLOR_PRIMARY if is_selected else COLOR_BACKGROUND
                text_color = ft.Colors.WHITE if is_selected else COLOR_NEUTRAL_DARK
                desc_color = (
                    ft.Colors.with_opacity(0.8, ft.Colors.WHITE)
                    if is_selected
                    else COLOR_NEUTRAL_DARK
                )
                border = None

            font_weight = ft.FontWeight.BOLD

            description = (
                self._description_for_code(code) if self._description_for_code is not None else ""
            )
            if description:
                content: ft.Control = ft.Column(
                    controls=[
                        ft.Text(
                            self._label_for_code(code),
                            size=20,
                            color=text_color,
                            weight=font_weight,
                            text_align=ft.TextAlign.CENTER,
                        ),
                        ft.Text(
                            description,
                            size=16,
                            color=desc_color,
                            text_align=ft.TextAlign.CENTER,
                        ),
                    ],
                    spacing=8,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                )
            else:
                content = ft.Text(
                    self._label_for_code(code),
                    size=20,
                    color=text_color,
                    weight=font_weight,
                    text_align=ft.TextAlign.CENTER,
                )

            item = ft.Container(
                content=content,
                bgcolor=bg_color,
                border_radius=16,
                border=border,
                padding=ft.Padding.all(24),
                alignment=ft.Alignment.CENTER,
                on_click=None if is_disabled else lambda e, selected=code: self._select(selected),
                on_hover=None if is_disabled else self._on_item_hover,
                animate=ft.Animation(150, ft.AnimationCurve.EASE_OUT),
                height=110,
            )
            items.append(item)

        return ft.ListView(
            controls=items,
            expand=True,
            spacing=16,
            padding=ft.Padding.only(right=8, bottom=12),
        )

    def _on_chip_hover(self, event: ft.ControlEvent) -> None:
        """Handle hover effect on chips."""
        apply_recent_language_chip_hover(event)

    def _on_item_hover(self, e: ft.ControlEvent) -> None:
        """Handle hover effect on list cards."""
        container = e.control
        content = container.content

        is_hovering = is_hover_active(e)
        # If text is white, it's selected. Don't hover.
        if isinstance(content, ft.Column) and content.controls:
            text_control = content.controls[0]
            desc_control = content.controls[1] if len(content.controls) > 1 else None
        else:
            text_control = content
            desc_control = None

        is_selected = getattr(text_control, "color", None) == ft.Colors.WHITE

        if not is_selected:
            if is_hovering:
                text_control.color = COLOR_PRIMARY
                if desc_control:
                    desc_control.color = COLOR_PRIMARY
            else:
                text_control.color = COLOR_NEUTRAL_DARK
                if desc_control:
                    desc_control.color = COLOR_NEUTRAL_DARK

            container.update()

    def _select(self, name: str) -> None:
        """Handle language selection."""
        logger.info("[LanguageModal] Selection requested: %s", name)
        if self._dialog:
            self._page.pop_dialog()
        self._on_select(name)
