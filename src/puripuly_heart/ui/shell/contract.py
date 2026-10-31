from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import flet as ft


class AppShellNavigationSink(Protocol):
    def set_selected_index(self, index: int) -> None: ...


@dataclass(frozen=True, slots=True)
class AppShellSlots:
    title_bar: ft.Control
    content: ft.Control
    bottom_nav: ft.Control
    content_padding: int
    debug_panel: ft.Control | None = None


@dataclass(frozen=True, slots=True)
class AppShellRegions:
    root: ft.Container
    layout: ft.Column
    content_area: ft.Container
    debug_stack: ft.Stack | None


__all__ = [
    "AppShellNavigationSink",
    "AppShellRegions",
    "AppShellSlots",
]
