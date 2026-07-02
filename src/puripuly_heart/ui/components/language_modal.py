"""Language selection modal component.

Provides a reusable modal dialog for selecting source or target language
with recent language chips and a scrollable full list.
"""

import logging
from typing import Callable, Sequence

import flet as ft

from puripuly_heart.ui.components.glow import create_glow_stack
from puripuly_heart.ui.i18n import language_name, t
from puripuly_heart.ui.theme import (
    COLOR_BACKGROUND,
    COLOR_NEUTRAL,
    COLOR_NEUTRAL_DARK,
    COLOR_PRIMARY,
    COLOR_SURFACE,
    get_card_shadow,
)

logger = logging.getLogger(__name__)


class LanguageModal:
    """Modal dialog for language selection.

    Features:
    - Recent languages as chips (up to 6)
    - Full language list with current selection highlighted
    - Closes on selection or outside click
    """

    def __init__(
        self,
        page: ft.Page,
        languages: Sequence[tuple[str, str]],
        on_select: Callable[[str], None],
    ):
        """Initialize language modal.

        Args:
            page: Flet page for dialog management.
            languages: List of (code, name) tuples.
            on_select: Callback when a language is selected (receives code).
        """
        self._page = page
        self._languages = languages
        self._on_select = on_select
        self._dialog: ft.AlertDialog | None = None

    def open(
        self,
        current: str,
        recent: list[str],
    ) -> None:
        """Open the language selection dialog.

        Args:
            current: Currently selected language code.
            recent: List of recently selected language codes (up to 3).
        """
        # Build recent grid (up to 6 items)
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
                color=COLOR_NEUTRAL,
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
                color=COLOR_NEUTRAL,
            )
        )

        content_controls.append(lang_list)

        # Apply glow effect and shadows
        modal_content = ft.Container(
            content=ft.Column(
                content_controls,
                spacing=8,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,  # Stretch width
            ),
            width=600,  # Larger width
            height=700,  # Larger height
            padding=ft.padding.symmetric(horizontal=32, vertical=32),
            bgcolor=COLOR_SURFACE,
            border_radius=28,
            shadow=get_card_shadow(),
        )

        # Transparent AlertDialog, content handles the background/shadow/glow
        self._dialog = ft.AlertDialog(
            modal=False,
            content=create_glow_stack(modal_content),
            content_padding=0,
            bgcolor=ft.Colors.TRANSPARENT,
            surface_tint_color=ft.Colors.TRANSPARENT,
        )

        self._page.open(self._dialog)

    def _build_recent_grid(self, recent: list[str], current: str) -> ft.Control | None:
        """Build grid controls for recent languages (up to 6)."""
        if not recent:
            return None

        items = []
        for lang_code in recent[:6]:
            is_current = lang_code == current
            # Bento Card Style
            bg_color = COLOR_PRIMARY if is_current else COLOR_BACKGROUND
            text_color = ft.Colors.WHITE if is_current else COLOR_NEUTRAL_DARK
            font_weight = ft.FontWeight.BOLD

            shadow = (
                None
                if is_current
                else ft.BoxShadow(
                    blur_radius=5,
                    color=ft.Colors.with_opacity(0.05, ft.Colors.BLACK),
                    offset=ft.Offset(0, 2),
                )
            )

            chip = ft.Container(
                content=ft.Text(
                    language_name(lang_code),
                    size=16,  # Larger text for VR
                    weight=font_weight,
                    color=text_color,
                    text_align=ft.TextAlign.CENTER,
                    max_lines=1,
                    overflow=ft.TextOverflow.ELLIPSIS,
                ),
                bgcolor=bg_color,
                border_radius=16,
                padding=ft.padding.symmetric(horizontal=8, vertical=16),  # Taller padding
                alignment=ft.alignment.center,
                on_click=lambda e, code=lang_code: self._select(code),
                on_hover=self._on_chip_hover,
                animate=ft.Animation(200, ft.AnimationCurve.EASE_OUT),
                shadow=shadow,
                width=170,  # Fixed width for 3-column layout (536px available / 3 ~= 178px)
            )
            items.append(chip)

        return ft.Row(
            controls=items,
            wrap=True,
            spacing=10,
            run_spacing=10,
            alignment=ft.MainAxisAlignment.START,
        )

    def _build_language_list(self, current: str) -> ft.ListView:
        """Build scrollable list of all languages (Bento Card Style)."""
        items = []
        for code, _name in self._languages:
            is_selected = code == current

            # Bento Card Style
            bg_color = COLOR_PRIMARY if is_selected else COLOR_BACKGROUND
            text_color = ft.Colors.WHITE if is_selected else COLOR_NEUTRAL_DARK
            font_weight = ft.FontWeight.BOLD

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

            item = ft.Container(
                content=ft.Text(
                    language_name(code),
                    size=20,  # Much Larger text for VR
                    color=text_color,
                    weight=font_weight,
                    text_align=ft.TextAlign.CENTER,
                ),
                bgcolor=bg_color,
                border_radius=16,
                padding=ft.padding.all(24),  # Much Larger padding (approx 120-130px total height)
                alignment=ft.alignment.center,
                on_click=lambda e, selected=code: self._select(selected),
                on_hover=self._on_item_hover,
                animate=ft.Animation(150, ft.AnimationCurve.EASE_OUT),
                shadow=shadow,
                height=110,  # Explicit height to ensure 4 items fit (600px content area / 4 = 150px)
            )
            items.append(item)

        return ft.ListView(
            controls=items,
            expand=True,
            spacing=16,  # Increased spacing
            padding=ft.padding.only(right=8, bottom=12),
        )

    def _on_chip_hover(self, e: ft.ControlEvent) -> None:
        """Handle hover effect on chips."""
        container = e.control
        text_control = container.content

        # If selected (primary bg), usually we don't need hover effect on text
        # But we must ensure specific check.
        # Here we assume selected items provide WHITE text initially.
        # So if text is WHITE, it is selected.
        if text_control.color == ft.Colors.WHITE:
            return

        if e.data == "true":
            text_control.color = COLOR_PRIMARY
            # container.bgcolor = ft.Colors.with_opacity(0.05, COLOR_PRIMARY) # Removed per user request
        else:
            text_control.color = COLOR_NEUTRAL_DARK
            # container.bgcolor = COLOR_BACKGROUND # Keep original

        container.update()

    def _on_item_hover(self, e: ft.ControlEvent) -> None:
        """Handle hover effect on list cards."""
        container = e.control
        text_control = container.content

        is_hovering = e.data == "true"
        # If text is white, it's selected. Don't hover.
        is_selected = text_control.color == ft.Colors.WHITE

        if not is_selected:
            if is_hovering:
                text_control.color = COLOR_PRIMARY
                # container.bgcolor = ft.Colors.with_opacity(0.05, COLOR_PRIMARY) # Optional tint
            else:
                text_control.color = COLOR_NEUTRAL_DARK
                # container.bgcolor = COLOR_BACKGROUND

            container.update()

    def _select(self, name: str) -> None:
        """Handle language selection."""
        logger.info("[LanguageModal] Selection requested: %s", name)
        if self._dialog:
            self._page.close(self._dialog)
        self._on_select(name)
