from __future__ import annotations

import logging
from typing import Protocol

from puripuly_heart.app.ports.application_startup import ApplicationStartupDiagnostic
from puripuly_heart.core.messages import ErrorDiagnostics


class ApplicationRuntimeLoggingPort(Protocol):
    def bind_instance(self, instance_id: str) -> None: ...

    def emit_failure(
        self,
        diagnostics: ErrorDiagnostics,
        *,
        severity: str = "error",
        disposition: str = "failed",
        stage: str | None = None,
        exception: BaseException | None = None,
    ) -> dict: ...

    def diagnostic_history(self, arguments: dict) -> dict: ...

    def diagnostic_health(self) -> dict: ...

    def emit_basic(self, message: str, *, level: int = logging.INFO) -> None: ...

    def emit_diagnostic(
        self,
        message: str,
        *,
        level: int = logging.INFO,
        exception: BaseException | None = None,
    ) -> bool: ...

    def emit_startup_diagnostic(self, diagnostic: ApplicationStartupDiagnostic) -> None: ...
