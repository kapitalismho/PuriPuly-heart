from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext


@dataclass(frozen=True, slots=True)
class ApplicationStartupState:
    settings: AppSettingsVNext
    fallback_channels: tuple[str, ...]
    installation_fallback: bool


class ApplicationStartupSettingsPort(Protocol):
    async def prepare_startup_settings(self) -> ApplicationStartupState: ...


class ApplicationStartupPresentationPort(Protocol):
    def apply_startup_presentation(self, state: ApplicationStartupState) -> None: ...


class ApplicationStartupRuntimePort(Protocol):
    async def launch_startup_runtime(self, state: ApplicationStartupState) -> None: ...


class ApplicationStartupEventsPort(Protocol):
    async def start_application_events(self) -> None: ...
