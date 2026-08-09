from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class OscQueryServiceInfo:
    service_id: str
    host: str
    query_port: int | None = None
    osc_send_port: int | None = None
    osc_receive_port: int | None = None
    is_vrchat: bool = False


@dataclass(frozen=True, slots=True)
class OscQueryAdvertisement:
    host: str
    port: int
    parameters: Mapping[str, str]
    query_port: int | None = None
    service_name: str = "PuriPuly Heart"


OscQueryServicesChanged = Callable[[Sequence[OscQueryServiceInfo]], Awaitable[None] | None]


class OscQueryServicePort(Protocol):
    async def start(self, services_changed: OscQueryServicesChanged | None = None) -> None: ...

    async def stop(self) -> None: ...

    async def discover_vrchat(self) -> OscQueryServiceInfo | None: ...

    async def advertise_receiver(
        self,
        advertisement: OscQueryAdvertisement,
    ) -> None: ...

    async def unadvertise_receiver(self) -> None: ...

    async def query_avatar(self, service: OscQueryServiceInfo) -> Mapping[str, object]: ...


__all__ = [
    "OscQueryAdvertisement",
    "OscQueryServiceInfo",
    "OscQueryServicePort",
    "OscQueryServicesChanged",
]
