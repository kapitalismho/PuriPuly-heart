"""Settings selection modal component.

Provides a reusable modal dialog for selecting settings options
with optional descriptions for each option.
"""

from __future__ import annotations

from typing import Callable, Sequence

import flet as ft

from puripuly_heart.app.ports.ui_models import OptionItem
from puripuly_heart.ui.flet_runtime import is_hover_active
from puripuly_heart.ui.theme import (
    COLOR_BACKGROUND,
    COLOR_NEUTRAL_DARK,
    COLOR_ON_BACKGROUND,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SURFACE,
)


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
        two_column: bool = False,
        left_column_sections: int = 1,
    ):
        """Initialize settings modal.

        Args:
            page: Flet page for dialog management.
            title: Modal title text.
            options: List of OptionItem objects.
            on_select: Callback when an option is selected (receives value).
            show_description: Whether to show descriptions for options.
            two_column: Whether to force a 2-column layout. When False, the
                modal always renders as a single column regardless of how
                many sections the options carry.
            left_column_sections: Number of sections placed in the left column
                of a two-column layout. Only used when two_column is True.
        """
        self._page = page
        self._title = title
        self._options = options
        self._on_select = on_select
        self._show_description = show_description
        self._two_column = two_column
        self._left_column_sections = left_column_sections
        self._dialog: ft.AlertDialog | None = None
        self._option_list: ft.ListView | ft.Row | None = None
        self._section_lists: list[tuple[ft.ListView, list[str]]] | None = None
        self._current: str = ""
        self._loading_section: str = ""
        self._partition_left_sections = left_column_sections

    def open(self, current: str, *, loading_section: str = "") -> None:
        """Open the settings selection dialog.

        Args:
            current: Currently selected option value.
            loading_section: Section label to show as loading placeholder.
                When set, a spinner is shown for that section instead of options.
        """
        self._current = current
        self._loading_section = loading_section
        sections = self._collect_sections()
        is_two_column = self._two_column
        option_list = self._build_option_list(current, sections, is_two_column)

        content_controls: list[ft.Control] = [
            ft.Text(
                self._title,
                size=24,
                weight=ft.FontWeight.BOLD,
                color=COLOR_SECONDARY,
            ),
            ft.Container(height=16),
            option_list,
        ]

        modal_content = ft.Container(
            content=ft.Column(
                content_controls,
                spacing=8,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            ),
            width=880 if is_two_column else 600,
            height=700,
            padding=ft.Padding.symmetric(horizontal=32, vertical=32),
            bgcolor=COLOR_SURFACE,
            border_radius=28,
        )

        # Transparent AlertDialog
        self._dialog = ft.AlertDialog(
            modal=False,
            content=modal_content,
            content_padding=0,
            bgcolor=ft.Colors.TRANSPARENT,
        )

        self._page.show_dialog(self._dialog)

    def replace_options(self, options: Sequence[OptionItem]) -> None:
        """Replace the option list after the modal is open.

        Args:
            options: New list of OptionItem objects.
        """
        self._options = options
        self._loading_section = ""
        if self._section_lists is not None:
            for list_view, assigned_sections in self._section_lists:
                list_view.controls = self._build_column_items(self._current, assigned_sections)
                try:
                    list_view.update()
                except Exception:
                    pass
        elif self._option_list is not None:
            self._option_list.controls = self._build_option_items(self._current)
            try:
                self._option_list.update()
            except Exception:
                pass

    def _collect_sections(self) -> list[str]:
        """Collect distinct section labels in order of appearance."""
        seen: list[str] = []
        for option in self._options:
            if option.section and option.section not in seen:
                seen.append(option.section)
        return seen

    def _build_option_list(
        self,
        current: str,
        sections: list[str] | None = None,
        is_two_column: bool | None = None,
    ) -> ft.Control:
        """Build scrollable list of options."""
        if sections is None:
            sections = self._collect_sections()
        if is_two_column is None:
            is_two_column = self._two_column
        if is_two_column:
            return self._build_two_column_list(current, sections)
        return self._build_single_column_list(current)

    def _build_single_column_list(self, current: str) -> ft.ListView:
        items = self._build_option_items(current)
        self._option_list = ft.ListView(
            controls=items,
            expand=True,
            spacing=12,
            padding=ft.Padding.only(right=8, bottom=12),
        )
        self._section_lists = None
        return self._option_list

    def _build_two_column_list(self, current: str, sections: list[str]) -> ft.Row:
        columns: list[ft.Control] = []
        self._section_lists = []
        for assigned_sections in self._partition_sections(sections, 2):
            list_view = ft.ListView(
                controls=self._build_column_items(current, assigned_sections),
                expand=True,
                spacing=12,
                padding=ft.Padding.only(right=8, bottom=12),
            )
            self._section_lists.append((list_view, assigned_sections))
            columns.append(ft.Container(content=list_view, width=400))
        self._option_list = ft.Row(
            controls=columns,
            spacing=16,
            expand=True,
        )
        return self._option_list

    def _partition_sections(self, sections: list[str], n_columns: int) -> list[list[str]]:
        """Distribute sections across a fixed number of columns.

        The first column receives ``left_column_sections`` leading groups;
        the remaining sections are balanced across the other columns.
        """
        n = len(sections)
        if n == 0:
            return [[] for _ in range(n_columns)]
        if n_columns <= 1:
            return [list(sections)]
        left_count = self._partition_left_sections
        left_count = max(1, min(left_count, n))
        left = list(sections[:left_count])
        rest = sections[left_count:]
        remaining = n_columns - 1
        if not rest:
            return [left, []]
        base = len(rest) // remaining
        extra = len(rest) % remaining
        cols: list[list[str]] = [left]
        idx = 0
        for i in range(remaining):
            size = base + (1 if i < extra else 0)
            cols.append(list(rest[idx : idx + size]))
            idx += size
        return cols

    def _build_column_items(
        self,
        current: str,
        assigned_sections: list[str],
    ) -> list[ft.Control]:
        """Build items for a single column, including headers for each assigned section."""
        items: list[ft.Control] = []
        is_first = True
        for section in assigned_sections:
            items.append(self._build_section_header(section, is_first=is_first))
            if self._loading_section and section == self._loading_section:
                items.append(self._build_loading_placeholder())
            else:
                for option in self._options:
                    if option.section != section:
                        continue
                    items.append(self._build_option_card(option, current))
            is_first = False
        return items

    def _build_option_items(self, current: str) -> list[ft.Control]:
        """Build list of option item controls (1-column mode)."""
        items: list[ft.Control] = []
        previous_section: str | None = None
        is_first_section = True
        for option in self._options:
            if option.section and option.section != previous_section:
                items.append(self._build_section_header(option.section, is_first_section))
                previous_section = option.section
                is_first_section = False
                if self._loading_section and option.section == self._loading_section:
                    items.append(self._build_loading_placeholder())
                    continue
            if self._loading_section and option.section == self._loading_section:
                continue
            items.append(self._build_option_card(option, current))
        return items

    def _build_option_card(self, option: OptionItem, current: str) -> ft.Control:
        """Build a single option card."""
        is_selected = option.value == current and not option.disabled

        if option.disabled:
            bg_color = COLOR_BACKGROUND
            text_color = ft.Colors.with_opacity(0.35, COLOR_NEUTRAL_DARK)
            desc_color = ft.Colors.with_opacity(0.35, COLOR_NEUTRAL_DARK)
            border = None
        else:
            bg_color = COLOR_PRIMARY if is_selected else COLOR_BACKGROUND
            text_color = ft.Colors.WHITE if is_selected else COLOR_ON_BACKGROUND
            desc_color = (
                ft.Colors.with_opacity(0.8, ft.Colors.WHITE) if is_selected else COLOR_NEUTRAL_DARK
            )
            border = None

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

        return ft.Container(
            content=content,
            bgcolor=bg_color,
            border_radius=16,
            border=border,
            padding=ft.Padding.all(24),
            alignment=ft.Alignment.CENTER,
            on_click=None if option.disabled else lambda e, val=option.value: self._select(val),
            on_hover=None if option.disabled else self._on_item_hover,
            animate=ft.Animation(150, ft.AnimationCurve.EASE_OUT),
            height=110,
        )

    def _build_loading_placeholder(self) -> ft.Control:
        """Build a loading placeholder with a spinner."""
        from puripuly_heart.ui.components.loading_spinner import create_section_spinner

        return ft.Container(
            content=ft.Row(
                controls=[
                    create_section_spinner(size=32, stroke_width=3),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            bgcolor=COLOR_BACKGROUND,
            border_radius=16,
            padding=ft.Padding.all(24),
            height=110,
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
                color=COLOR_SECONDARY,
            )
        )
        return ft.Container(
            content=ft.Column(controls, spacing=0),
            padding=ft.Padding.symmetric(horizontal=4),
        )

    def _on_item_hover(self, e: ft.ControlEvent) -> None:
        """Handle hover effect on option cards."""
        container = e.control
        content = container.content

        is_hovering = is_hover_active(e)

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
            self._page.pop_dialog()
        self._on_select(value)
