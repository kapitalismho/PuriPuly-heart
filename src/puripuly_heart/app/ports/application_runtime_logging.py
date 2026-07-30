from __future__ import annotations

import logging
from typing import Protocol


class ApplicationRuntimeLoggingPort(Protocol):
    @property
    def mode(self) -> str: ...

    def emit_basic(self, message: str, *, level: int = logging.INFO) -> None: ...

    def emit_detailed(
        self,
        message: str,
        *,
        level: int = logging.INFO,
        exception: BaseException | None = None,
    ) -> bool: ...
