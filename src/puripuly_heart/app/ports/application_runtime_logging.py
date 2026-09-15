from __future__ import annotations

import logging
from typing import Protocol


class ApplicationRuntimeLoggingPort(Protocol):
    def emit_basic(self, message: str, *, level: int = logging.INFO) -> None: ...

    def emit_diagnostic(
        self,
        message: str,
        *,
        level: int = logging.INFO,
        exception: BaseException | None = None,
    ) -> bool: ...
