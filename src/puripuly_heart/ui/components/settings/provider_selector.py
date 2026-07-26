"""Provider selector dropdown component."""

from __future__ import annotations

from enum import Enum
from typing import Callable, TypeVar

import flet as ft

from puripuly_heart.ui.flet_runtime import update_control_if_mounted
from puripuly_heart.ui.i18n import provider_label, t
from puripuly_heart.ui.theme import COLOR_NEUTRAL_DARK

T = TypeVar("T", bound=Enum)


class ProviderSelector(ft.Dropdown):
    """Dropdown for selecting providers from an Enum."""

    def __init__(
        self,
        label_key: str,
        enum_class: type[T],
        on_change: Callable[[T], None] | None = None,
    ):
        self._label_key = label_key
        self._enum_class = enum_class
        self._on_change_callback = on_change
        self._value_map: dict[str, T] = {}

        options = self._build_options()

        super().__init__(
            label=t(label_key),
            options=options,
            on_select=self._handle_change,
            border_radius=12,
            text_size=28,
            label_style=ft.TextStyle(size=20, weight=ft.FontWeight.BOLD),
            color=COLOR_NEUTRAL_DARK,
        )

    def _build_options(self) -> list[ft.dropdown.Option]:
        """Build dropdown options from enum."""
        options = []
        self._value_map.clear()

        for member in self._enum_class:
            # Use provider_label for display, enum.value as key
            display = provider_label(member.value)
            self._value_map[member.value] = member
            options.append(ft.dropdown.Option(key=member.value, text=display))

        return options

    @property
    def selected_provider(self) -> T | None:
        """Get the currently selected provider enum."""
        if self.value:
            return self._value_map.get(self.value)
        return None

    @selected_provider.setter
    def selected_provider(self, provider: T) -> None:
        """Set the selected provider."""
        self.value = provider.value
        update_control_if_mounted(self)

    def _handle_change(self, e) -> None:
        if self._on_change_callback and self.value:
            provider = self._value_map.get(self.value)
            if provider:
                self._on_change_callback(provider)

    def apply_locale(self) -> None:
        """Update labels when locale changes."""
        self.label = t(self._label_key)
        self.options = self._build_options()
        update_control_if_mounted(self)
