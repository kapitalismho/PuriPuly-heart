"""Settings selection modal component.

Provides a reusable modal dialog for selecting settings options
with optional descriptions for each option.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import flet as ft

from puripuly_heart.ui.components.glow import create_glow_stack
from puripuly_heart.ui.theme import (
    COLOR_BACKGROUND,
    COLOR_NEUTRAL,
    COLOR_NEUTRAL_DARK,
    COLOR_ON_BACKGROUND,
    COLOR_PRIMARY,
    COLOR_SURFACE,
    get_card_shadow,
)


@dataclass
class OptionItem:
    """Option item for settings modal."""

    value: str
    label: str
    description: str = ""
    disabled: bool = False
    section: str = ""


class SettingsModal:
    """Modal dialog for settings selection.

    Features:
    - Scrollable option list with current selection highlighted
    - Optional descriptions for each option
    - Closes on selection or outside click
    """

    def __init__(
        self,
        page: ft.Page,
        title: str,
        options: Sequence[OptionItem],
        on_select: Callable[[str], None],
        *,
        show_description: bool = False,
    ):
        """Initialize settings modal.

        Args:
            page: Flet page for dialog management.
            title: Modal title text.
            options: List of OptionItem objects.
            on_select: Callback when an option is selected (receives value).
            show_description: Whether to show descriptions for options.
        """
        self._page = page
        self._title = title
        self._options = options
        self._on_select = on_select
        self._show_description = show_description
        self._dialog: ft.AlertDialog | None = None

    def open(self, current: str) -> None:
        """Open the settings selection dialog.

        Args:
            current: Currently selected option value.
        """
        # Build option list
        option_list = self._build_option_list(current)

        # Content column
        content_controls: list[ft.Control] = [
            ft.Text(
                self._title,
                size=24,
                weight=ft.FontWeight.BOLD,
                color=COLOR_NEUTRAL,
            ),
            ft.Container(height=16),
            option_list,
        ]

        # Modal content
        modal_content = ft.Container(
            content=ft.Column(
                content_controls,
                spacing=8,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            ),
            width=600,
            height=700,
            padding=ft.padding.symmetric(horizontal=32, vertical=32),
            bgcolor=COLOR_SURFACE,
            border_radius=28,
            shadow=get_card_shadow(),
        )

        # Transparent AlertDialog
        self._dialog = ft.AlertDialog(
            modal=False,
            content=create_glow_stack(modal_content),
            content_padding=0,
            bgcolor=ft.Colors.TRANSPARENT,
            surface_tint_color=ft.Colors.TRANSPARENT,
        )

        self._page.open(self._dialog)

    def _build_option_list(self, current: str) -> ft.ListView:
        """Build scrollable list of options."""
        items = []
        previous_section: str | None = None
        is_first_section = True
        for option in self._options:
            if option.section and option.section != previous_section:
                items.append(self._build_section_header(option.section, is_first_section))
                previous_section = option.section
                is_first_section = False
            is_selected = option.value == current and not option.disabled

            # Colors
            if option.disabled:
                bg_color = COLOR_SURFACE
                text_color = COLOR_NEUTRAL
                desc_color = COLOR_ON_BACKGROUND
                border = ft.border.all(1, ft.Colors.with_opacity(0.2, COLOR_PRIMARY))
            else:
                bg_color = COLOR_PRIMARY if is_selected else COLOR_BACKGROUND
                text_color = ft.Colors.WHITE if is_selected else COLOR_ON_BACKGROUND
                desc_color = (
                    ft.Colors.with_opacity(0.8, ft.Colors.WHITE)
                    if is_selected
                    else COLOR_NEUTRAL_DARK
                )
                border = None

            # Shadow for depth
            shadow = (
                ft.BoxShadow(
                    blur_radius=2,
                    color=ft.Colors.with_opacity(0.05, ft.Colors.BLACK),
                    offset=ft.Offset(0, 1),
                )
                if not is_selected
                else None
            )

            # Build content based on show_description
            if self._show_description and option.description:
                content = ft.Column(
                    controls=[
                        ft.Text(
                            option.label,
                            size=20,
                            color=text_color,
                            weight=ft.FontWeight.BOLD,
                            text_align=ft.TextAlign.CENTER,
                        ),
                        ft.Text(
                            option.description,
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
                    option.label,
                    size=20,
                    color=text_color,
                    weight=ft.FontWeight.BOLD,
                    text_align=ft.TextAlign.CENTER,
                )

            item = ft.Container(
                content=content,
                bgcolor=bg_color,
                border_radius=16,
                border=border,
                padding=ft.padding.all(24),
                alignment=ft.alignment.center,
                on_click=None if option.disabled else lambda e, val=option.value: self._select(val),
                on_hover=None if option.disabled else self._on_item_hover,
                animate=ft.Animation(150, ft.AnimationCurve.EASE_OUT),
                shadow=shadow,
                height=110,
            )
            items.append(item)

        return ft.ListView(
            controls=items,
            expand=True,
            spacing=12,
            padding=ft.padding.only(right=8, bottom=12),
        )

    def _build_section_header(self, label: str, is_first: bool) -> ft.Control:
        """Build a section header label."""
        controls: list[ft.Control] = []
        if not is_first:
            controls.append(ft.Container(height=8))
        controls.append(
            ft.Text(
                label,
                size=18,
                weight=ft.FontWeight.BOLD,
                color=COLOR_NEUTRAL,
            )
        )
        return ft.Container(
            content=ft.Column(controls, spacing=0),
            padding=ft.padding.symmetric(horizontal=4),
        )

    def _on_item_hover(self, e: ft.ControlEvent) -> None:
        """Handle hover effect on option cards."""
        container = e.control
        content = container.content

        is_hovering = e.data == "true"

        # Get text control (could be Text or Column with Text)
        if isinstance(content, ft.Text):
            text_control = content
            desc_control = None
        elif isinstance(content, ft.Column) and content.controls:
            text_control = content.controls[0]
            desc_control = content.controls[1] if len(content.controls) > 1 else None
        else:
            return

        # If text is white, it's selected. Don't hover.
        is_selected = text_control.color == ft.Colors.WHITE

        if not is_selected:
            if is_hovering:
                text_control.color = COLOR_PRIMARY
                if desc_control:
                    desc_control.color = COLOR_PRIMARY
            else:
                text_control.color = COLOR_ON_BACKGROUND
                if desc_control:
                    desc_control.color = COLOR_NEUTRAL_DARK

            container.update()

    def _select(self, value: str) -> None:
        """Handle option selection."""
        if self._dialog:
            self._page.close(self._dialog)
        self._on_select(value)
