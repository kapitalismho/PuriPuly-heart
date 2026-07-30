from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import flet as ft


class AboutStateSink(Protocol):
    def apply_locale(self) -> None: ...


@dataclass(frozen=True, slots=True)
class AboutSurfaceSlots:
    app_name_card: ft.Control
    version_card: ft.Control
    credits_card: ft.Control
    inspired_by_card: ft.Control
    special_thanks_card: ft.Control
    licenses_card: ft.Control


@dataclass(frozen=True, slots=True)
class AboutSurfaceRegions:
    controls: tuple[ft.Control, ...]
    header_row: ft.Container
    top_row: ft.Container


__all__ = [
    "AboutStateSink",
    "AboutSurfaceRegions",
    "AboutSurfaceSlots",
]
