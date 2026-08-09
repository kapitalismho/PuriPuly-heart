from __future__ import annotations

from collections.abc import Mapping

from puripuly_heart.app.ports.oscquery import (
    OscQueryAdvertisement,
    OscQueryServiceInfo,
    OscQueryServicePort,
    OscQueryServicesChanged,
)


class NoopOscQueryService(OscQueryServicePort):
    def __init__(self) -> None:
        self.advertisement: OscQueryAdvertisement | None = None
        self.started = False

    async def start(self, services_changed: OscQueryServicesChanged | None = None) -> None:
        _ = services_changed
        self.started = True

    async def stop(self) -> None:
        self.started = False
        self.advertisement = None

    async def discover_vrchat(self) -> OscQueryServiceInfo | None:
        return None

    async def advertise_receiver(self, advertisement: OscQueryAdvertisement) -> None:
        self.advertisement = advertisement

    async def unadvertise_receiver(self) -> None:
        self.advertisement = None

    async def query_avatar(self, service: OscQueryServiceInfo) -> Mapping[str, object]:
        _ = service
        return {}


__all__ = ["NoopOscQueryService"]
