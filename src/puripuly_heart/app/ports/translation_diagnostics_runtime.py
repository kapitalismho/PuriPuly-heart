from __future__ import annotations

from typing import Protocol

from puripuly_heart.core.overlay.diagnostics import OverlayDiagnosticsRecorder


class TranslationOverlayDiagnosticsPort(Protocol):
    def replace_overlay_diagnostics(
        self,
        diagnostics: OverlayDiagnosticsRecorder | None,
        *,
        expected_current: OverlayDiagnosticsRecorder | None = None,
        require_match: bool = False,
    ) -> bool: ...


__all__ = ["TranslationOverlayDiagnosticsPort"]
