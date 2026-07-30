from __future__ import annotations

from puripuly_heart.app.ports.application_startup import (
    ApplicationStartupEventsPort,
    ApplicationStartupPresentationPort,
    ApplicationStartupRuntimePort,
    ApplicationStartupSettingsPort,
)


class ApplicationStartupOwner:
    def __init__(
        self,
        *,
        settings: ApplicationStartupSettingsPort,
        presentation: ApplicationStartupPresentationPort,
        runtime: ApplicationStartupRuntimePort,
        events: ApplicationStartupEventsPort,
    ) -> None:
        self._settings = settings
        self._presentation = presentation
        self._runtime = runtime
        self._events = events

    async def start(self) -> None:
        state = await self._settings.prepare_startup_settings()
        self._presentation.apply_startup_presentation(state)
        await self._runtime.launch_startup_runtime(state)
        await self._events.start_application_events()
