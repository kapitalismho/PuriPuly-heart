from __future__ import annotations

from typing import Protocol

from puripuly_heart.core.overlay.sink import OverlaySink


class TranslationOutputProjectionPort(Protocol):
    @property
    def overlay_sink(self) -> OverlaySink | None: ...

    async def replace_overlay_sink(
        self,
        overlay_sink: OverlaySink | None,
        *,
        expected_current: OverlaySink | None = None,
        require_match: bool = False,
    ) -> bool: ...

    async def reset_overlay_preview(self) -> None: ...


__all__ = ["TranslationOutputProjectionPort"]
