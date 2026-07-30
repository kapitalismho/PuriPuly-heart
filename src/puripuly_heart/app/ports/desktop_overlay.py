from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

SettingsT = TypeVar("SettingsT")
DesktopBounds = dict[str, int | float]
DesktopRuntimeControl = dict[str, object]
DesktopWorkArea = tuple[int | float, int | float, int | float, int | float]


@dataclass(frozen=True, slots=True)
class DesktopOverlayPolicy:
    minimum_width: int
    minimum_height: int
    default_text_scale: float
    default_background_alpha: float
    default_size_preset: str
    size_presets: Mapping[str, tuple[int, int]]


class DesktopWorkAreaPort(Protocol):
    def primary_work_area(self) -> DesktopWorkArea | None: ...


class DesktopOverlayRuntimeEffectsPort(Protocol, Generic[SettingsT]):
    def prepare_settings_update(
        self,
        previous_settings: SettingsT | None,
        next_settings: SettingsT,
    ) -> tuple[DesktopRuntimeControl, ...]: ...

    async def prepare_persistence(
        self,
        previous_settings: SettingsT,
        next_settings: SettingsT,
    ) -> None: ...

    def sync_from_settings(self, settings: SettingsT) -> None: ...

    async def apply_controls(
        self,
        controls: tuple[DesktopRuntimeControl, ...],
    ) -> None: ...


__all__ = [
    "DesktopBounds",
    "DesktopOverlayPolicy",
    "DesktopOverlayRuntimeEffectsPort",
    "DesktopRuntimeControl",
    "DesktopWorkArea",
    "DesktopWorkAreaPort",
]
