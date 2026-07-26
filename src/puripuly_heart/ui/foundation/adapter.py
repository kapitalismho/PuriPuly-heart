from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from puripuly_heart.app.ports.ui_application import UiApplicationPort
from puripuly_heart.app.ports.ui_presentation import UiPresentationPort


@dataclass(frozen=True, slots=True)
class FoundationApplicationSnapshot:
    config_path: Path
    runtime_logging_mode: str
    translation_enabled: bool
    debug_preview_enabled: bool


@dataclass(slots=True)
class FletFoundationAdapter:
    _application: UiApplicationPort
    _presentation: UiPresentationPort

    def snapshot(self) -> FoundationApplicationSnapshot:
        state = self._application.state()
        return FoundationApplicationSnapshot(
            config_path=state.config_path,
            runtime_logging_mode=state.runtime_logging_mode,
            translation_enabled=state.translation_enabled,
            debug_preview_enabled=self.debug_preview_enabled,
        )

    @property
    def debug_preview_enabled(self) -> bool:
        return self._presentation.debug_ui_preview

    def apply_locale(self) -> None:
        self._presentation.apply_locale()


__all__ = ["FletFoundationAdapter", "FoundationApplicationSnapshot"]
