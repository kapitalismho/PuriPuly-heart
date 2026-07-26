from __future__ import annotations

from typing import Callable

import flet as ft

from puripuly_heart.config.prompts import load_prompt_for_provider
from puripuly_heart.ui.flet_runtime import FILL_PARENT_WIDTH, update_control_if_mounted
from puripuly_heart.ui.theme import COLOR_DIVIDER, COLOR_NEUTRAL_DARK, COLOR_PRIMARY


class PromptEditor(ft.Column):
    """System prompt editor component."""

    def __init__(
        self,
        on_change: Callable[[str], None] | None = None,
        on_commit: Callable[[str], None] | None = None,
    ):
        self._on_change = on_change
        self._on_commit = on_commit
        self._current_provider = "gemini"

        self._text_field = ft.TextField(
            multiline=True,
            min_lines=5,
            on_change=self._handle_change,
            on_blur=self._handle_blur,
            width=FILL_PARENT_WIDTH,
            border_radius=12,
            border_color=COLOR_DIVIDER,
            focused_border_color=COLOR_PRIMARY,
            text_size=16,
            color=COLOR_NEUTRAL_DARK,
        )

        super().__init__(
            controls=[self._text_field],
            spacing=12,
        )

    @property
    def value(self) -> str:
        """Get current prompt value."""
        return self._text_field.value or ""

    @value.setter
    def value(self, val: str) -> None:
        """Set prompt value."""
        self._text_field.value = val
        update_control_if_mounted(self._text_field)

    def set_provider(self, provider_name: str) -> None:
        """Update the current provider."""
        self._current_provider = provider_name

    def load_default_prompt(self, *, emit_change: bool = True) -> None:
        """Load default prompt for current provider."""
        self.value = load_prompt_for_provider(self._current_provider)
        if emit_change:
            self._emit_change()

    def load_default_if_empty(self) -> None:
        """Load default prompt only if current value is empty."""
        if not self.value.strip():
            self.load_default_prompt(emit_change=False)

    def commit(self) -> None:
        if self._on_commit:
            self._on_commit(self.value)

    def _handle_change(self, e) -> None:
        self._emit_change()

    def _handle_blur(self, e) -> None:
        self.commit()

    def _emit_change(self) -> None:
        if self._on_change:
            self._on_change(self.value)

    def apply_locale(self) -> None:
        """Update labels when locale changes."""
        update_control_if_mounted(self)
