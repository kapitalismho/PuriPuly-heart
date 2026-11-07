from __future__ import annotations

import asyncio
import json
import socket
from types import SimpleNamespace

import pytest

from puripuly_heart.app.ports.oscquery import (
    OscQueryAdvertisement,
    OscQueryServiceInfo,
)
from puripuly_heart.core.osc.oscquery import OSCJSON_SERVICE_TYPE, ZeroconfOscQueryService
from puripuly_heart.core.osc.udp_sender import VrchatOscUdpSender
from puripuly_heart.core.runtime.oscquery import OscQueryRuntime


class FakeService:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.advertisements: list[OscQueryAdvertisement] = []
        self.info: OscQueryServiceInfo | None = OscQueryServiceInfo(
            service_id="vrchat",
            host="127.0.0.1",
            osc_send_port=9010,
            is_vrchat=True,
        )

    async def start(self, _callback=None) -> None:
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1

    async def discover_vrchat(self) -> OscQueryServiceInfo | None:
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
        snapshot_publisher=lambda reason, _generation: events.append(("snapshot", reason)),
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
async def test_failed_automatic_start_can_restart_cleanly_with_current_discovery() -> None:
    class FailOnceDiscovery(FakeService):
        fail = True

        async def discover_vrchat(self) -> OscQueryServiceInfo | None:
            if self.fail:
                self.fail = False
                raise OSError("controlled discovery failure")
            return await super().discover_vrchat()

    service = FailOnceDiscovery()
    destinations = []
    snapshots = []
    runtime = OscQueryRuntime(
        service=service,
        receiver_start=lambda: _noop(),
        receiver_stop=lambda: _noop(),
        receiver_effective_port=lambda: 49152,
        sender_destination_changed=lambda host, port: destinations.append((host, port)),
        snapshot_publisher=lambda reason, generation: snapshots.append((reason, generation)),
    )
    try:
        with pytest.raises(OSError, match="controlled discovery failure"):
            await runtime.start("automatic", snapshot_generation=1)
        assert runtime.started is False
        assert runtime.service_info is None
        assert runtime.effective_send_port is None
        assert runtime.effective_receive_port is None
        assert runtime.advertised_port is None
        assert service.stopped == 1
        assert snapshots == []
        assert destinations == []
        await runtime.start("automatic", snapshot_generation=2)
        assert runtime.started is True
        assert runtime.service_info == service.info
        assert service.started == 2
        assert destinations == [("127.0.0.1", 9010)]
        assert snapshots == [("start", 2)]
        assert service.advertisements == [
            OscQueryAdvertisement(
                host="127.0.0.1",
                port=49152,
                parameters={"/avatar": "PuriPuly_*"},
            )
        ]
    finally:
        await runtime.stop()


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


@pytest.mark.asyncio
async def test_oscquery_monitor_detects_disappearance_and_rediscovery_without_callback() -> None:
    service = FakeService()
    events: list[tuple[str, object]] = []
    runtime = OscQueryRuntime(
        service=service,
        receiver_start=lambda: _noop(),
        receiver_stop=lambda: _noop(),
        receiver_effective_port=lambda: 49152,
        sender_destination_changed=lambda host, port: events.append((host, port)),
        snapshot_publisher=lambda reason, _generation: events.append(("snapshot", reason)),
        discovery_poll_interval_seconds=0.01,
    )
    await runtime.start("automatic", manual_send_port=9123)
    await asyncio.sleep(0.03)

    assert events.count(("snapshot", "discovery")) == 0

    service.info = None
    await _wait_until(lambda: runtime.service_info is None)

    assert runtime.effective_send_port == 9000
    assert runtime.avatar_tree is None

    service.info = OscQueryServiceInfo(
        service_id="vrchat-restarted",
        host="127.0.0.1",
        osc_send_port=9999,
        is_vrchat=True,
    )
    await _wait_until(lambda: runtime.effective_send_port == 9999)

    assert ("127.0.0.1", 9999) in events
    assert events.count(("snapshot", "discovery")) == 1
    await runtime.stop()
    assert runtime._monitor_task is None


async def _noop() -> None:
    return None


async def _wait_until(predicate, timeout: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition did not become true")


@pytest.mark.asyncio
async def test_real_host_info_transient_recovers_through_existing_monitor_without_new_mdns() -> (
    None
):
    class LocalDiscovery(ZeroconfOscQueryService):
        async def start(self, _services_changed=None) -> None:
            self.started = True

        async def advertise_receiver(self, advertisement) -> None:
            self.advertisement = advertisement

    failures = 0
    requests = 0
    destinations: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
    target = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target.bind(("127.0.0.1", 0))
    target_port = target.getsockname()[1]
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver_port = receiver.getsockname()[1]
    sender = VrchatOscUdpSender()

    async def serve(reader, writer):
        nonlocal failures, requests
        try:
            request = await reader.readuntil(b"\r\n\r\n")
            failed = False
            if b"?HOST_INFO" in request:
                requests += 1
                if failures:
                    failures -= 1
                    failed = True
                payload = {"OSC_IP": "127.0.0.1", "OSC_PORT": target_port}
            else:
                payload = {}
            body = json.dumps(payload).encode()
            status = b"503 Service Unavailable" if failed else b"200 OK"
            writer.write(
                b"HTTP/1.1 " + status + b"\r\nContent-Type: application/json\r\n"
                b"Connection: close\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\n\r\n"
                + body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    service = LocalDiscovery()
    service_name = f"VRChat-local.{OSCJSON_SERVICE_TYPE}"
    service._queue_service_update(
        OSCJSON_SERVICE_TYPE,
        service_name,
        SimpleNamespace(
            parsed_addresses=lambda: ["127.0.0.1"],
            port=server.sockets[0].getsockname()[1],
            properties={b"NAME": b"VRChat-local"},
        ),
    )

    def apply_destination(host, port):
        sender.set_destination(host, port)
        destinations.put_nowait((host, port))

    runtime = OscQueryRuntime(
        service=service,
        receiver_start=_noop,
        receiver_stop=_noop,
        receiver_effective_port=lambda: receiver_port,
        sender_destination_changed=apply_destination,
        discovery_poll_interval_seconds=0.1,
    )
    try:
        await runtime.start("automatic")
        assert await asyncio.wait_for(destinations.get(), timeout=5) == ("127.0.0.1", target_port)
        assert sender.port == target_port
        failures = 1
        assert await asyncio.wait_for(destinations.get(), timeout=5) == ("127.0.0.1", 9000)
        assert await asyncio.wait_for(destinations.get(), timeout=5) == ("127.0.0.1", target_port)
        assert requests >= 3
        assert runtime.service_info is not None
        assert runtime.effective_receive_port == receiver_port
        assert receiver.fileno() >= 0
        service._queue_service_update(OSCJSON_SERVICE_TYPE, service_name, None)
        assert await asyncio.wait_for(destinations.get(), timeout=5) == ("127.0.0.1", 9000)
        removed_requests = requests
        await asyncio.sleep(0.25)
        assert requests == removed_requests
        assert runtime.service_info is None
        assert sender.port == 9000
    finally:
        await runtime.stop()
        server.close()
        await server.wait_closed()
        receiver.close()
        target.close()
        sender.close()


@pytest.mark.asyncio
async def test_explicit_forced_refresh_reapplies_same_target_without_control_snapshot() -> None:
    service = FakeService()
    destinations = []
    snapshots = []
    runtime = OscQueryRuntime(
        service=service,
        receiver_start=_noop,
        receiver_stop=_noop,
        receiver_effective_port=lambda: 49152,
        sender_destination_changed=lambda host, port: destinations.append((host, port)),
        snapshot_publisher=lambda reason, generation: snapshots.append((reason, generation)),
    )
    try:
        await runtime.start("automatic")
        before = list(snapshots)
        assert await runtime.refresh(publish_snapshot=False, force_requery=True) is True
        assert destinations == [("127.0.0.1", 9010), ("127.0.0.1", 9010)]
        assert snapshots == before
        assert service.started == 1
        assert len(service.advertisements) == 1
    finally:
        await runtime.stop()
