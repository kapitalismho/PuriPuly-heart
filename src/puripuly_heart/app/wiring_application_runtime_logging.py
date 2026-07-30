from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from puripuly_heart.app.ports.ui_presentation import UiPresentationPort
from puripuly_heart.app.services.application_runtime_logging import (
    ApplicationRuntimeLoggingOwner,
)
from puripuly_heart.core.runtime.logging import RuntimeLoggingService
from puripuly_heart.core.runtime_logging import (
    RealtimeLogHandler,
    RuntimeLoggingSinks,
    SessionRuntimeLoggingService,
)


def compose_application_runtime_logging(
    *,
    presentation: UiPresentationPort,
    sinks: RuntimeLoggingSinks | None,
    overlay_logging_mode_update: Callable[[], Awaitable[None]],
    overlay_logging_mode_update_available: Callable[[], bool],
) -> ApplicationRuntimeLoggingOwner:
    fallback_logger = logging.getLogger("puripuly_heart.runtime")
    return ApplicationRuntimeLoggingOwner(
        presentation=presentation,
        service_factory=lambda: RuntimeLoggingService(
            session_factory=lambda: SessionRuntimeLoggingService(
                sinks=sinks,
                ui_handler_factory=RealtimeLogHandler,
            ),
            fallback_logger=fallback_logger,
        ),
        fallback_logger=fallback_logger,
        overlay_logging_mode_update=overlay_logging_mode_update,
        overlay_logging_mode_update_available=overlay_logging_mode_update_available,
    )
