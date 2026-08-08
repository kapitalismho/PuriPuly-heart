from __future__ import annotations

import pytest

from puripuly_heart.app.ports.oscquery import (
    OscQueryAdvertisement,
    OscQueryServiceInfo,
)
from puripuly_heart.core.runtime.oscquery import OscQueryRuntime


class FakeService:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.advertisements: list[OscQueryAdvertisement] = []
        self.info = OscQueryServiceInfo(
            service_id="vrchat",
            host="127.0.0.1",
            osc_send_port=9010,
            is_vrchat=True,
        )

    async def start(self, _callback=None) -> None:
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1

    async def discover_vrchat(self) -> OscQueryServiceInfo:
        return self.info

    async def advertise_receiver(self, advertisement: OscQueryAdvertisement) -> None:
        self.advertisements.append(advertisement)

    async def unadvertise_receiver(self) -> None:
        self.advertisements.clear()

    async def query_avatar(self, _service: OscQueryServiceInfo) -> dict[str, object]:
        return {"parameters": {"PuriPuly_Talk": {"TYPE": "T"}}}


class StartupRaceService(FakeService):
    def __init__(self) -> None:
        super().__init__()
        self._services_changed = None
        self._discoveries = 0

    async def start(self, services_changed=None) -> None:
        self.started += 1
        self._services_changed = services_changed

    async def discover_vrchat(self) -> OscQueryServiceInfo:
        self._discoveries += 1
        result = self.info
        if self._discoveries == 1:
            self.info = OscQueryServiceInfo(
                service_id="vrchat-updated",
                host="127.0.0.1",
                osc_send_port=9999,
                is_vrchat=True,
            )
            if self._services_changed is not None:
                await self._services_changed(())
        assert result is not None
        return result


class FailingStartService(FakeService):
    async def start(self, _callback=None) -> None:
        self.started += 1
        raise RuntimeError("service start failed")


@pytest.mark.asyncio
async def test_oscquery_runtime_uses_dynamic_receive_port_and_discovered_send_port() -> None:
    service = FakeService()
    events: list[tuple[str, object]] = []
    running = False

    async def start_receiver() -> None:
        nonlocal running
        running = True

    async def stop_receiver() -> None:
        nonlocal running
        running = False

    runtime = OscQueryRuntime(
        service=service,
        receiver_start=start_receiver,
        receiver_stop=stop_receiver,
        receiver_effective_port=lambda: 49152,
        sender_destination_changed=lambda host, port: events.append((host, port)),
        snapshot_publisher=lambda reason: events.append(("snapshot", reason)),
    )

    await runtime.start("automatic")

    assert running is True
    assert runtime.effective_receive_port == 49152
    assert runtime.effective_send_port == 9010
    assert service.advertisements[0].port == 49152
    assert ("127.0.0.1", 9010) in events

    await runtime.stop()
    assert running is False
    assert service.stopped == 1


@pytest.mark.asyncio
async def test_oscquery_runtime_stops_service_when_start_fails() -> None:
    service = FailingStartService()
    runtime = OscQueryRuntime(
        service=service,
        receiver_start=lambda: _noop(),
        receiver_stop=lambda: _noop(),
        receiver_effective_port=lambda: 49152,
    )

    with pytest.raises(RuntimeError, match="service start failed"):
        await runtime.start("automatic")

    assert service.started == 1
    assert service.stopped == 1
    assert runtime.started is False


@pytest.mark.asyncio
async def test_oscquery_start_applies_discovery_events_during_initialization() -> None:
    service = StartupRaceService()
    events: list[tuple[str, object]] = []
    runtime = OscQueryRuntime(
        service=service,
        receiver_start=lambda: _noop(),
        receiver_stop=lambda: _noop(),
        receiver_effective_port=lambda: 49152,
        sender_destination_changed=lambda host, port: events.append((host, port)),
    )

    await runtime.start("automatic")

    assert runtime.effective_send_port == 9999
    assert ("127.0.0.1", 9999) in events
    await runtime.stop()


async def _noop() -> None:
    return None
