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
        self.phase = "startup_prepare"

    async def start(self) -> None:
        self.phase = "startup_prepare"
        state = await self._settings.prepare_startup_settings()
        self.phase = "startup_presentation"
        self._presentation.apply_startup_presentation(state)
        self.phase = "startup_runtime"
        await self._runtime.launch_startup_runtime(state)
        self.phase = "startup_events"
        await self._events.start_application_events()
