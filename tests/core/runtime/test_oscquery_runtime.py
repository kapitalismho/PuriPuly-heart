from __future__ import annotations

import asyncio
import json
import socket
from dataclasses import replace
from types import SimpleNamespace

import pytest

from puripuly_heart.app.ports.oscquery import (
    OscQueryAdvertisement,
    OscQueryServiceInfo,
    OscQueryServicesChanged,
)
from puripuly_heart.core.diagnostics import DIAGNOSTIC_RECORD_ATTRIBUTE
from puripuly_heart.core.osc.oscquery import (
    OSC_SERVICE_TYPE,
    OSCJSON_SERVICE_TYPE,
    ZeroconfOscQueryService,
)
from puripuly_heart.core.osc.udp_sender import VrchatOscUdpSender
from puripuly_heart.core.runtime.oscquery import OscQueryRuntime


class FakeService:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.advertisements: list[OscQueryAdvertisement] = []
        self.discoveries = 0
        self.services_changed: OscQueryServicesChanged | None = None
        self.info: OscQueryServiceInfo | None = OscQueryServiceInfo(
            service_id="vrchat",
            host="127.0.0.1",
            osc_send_port=9010,
            is_vrchat=True,
        )

    async def start(self, services_changed: OscQueryServicesChanged | None = None) -> None:
        self.started += 1
        self.services_changed = services_changed

    async def stop(self) -> None:
        self.stopped += 1

    async def discover_vrchat(
        self, *, preferred_service_id: str | None = None
    ) -> OscQueryServiceInfo | None:
        self.discoveries += 1
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

    async def discover_vrchat(
        self, *, preferred_service_id: str | None = None
    ) -> OscQueryServiceInfo:
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
                self._services_changed(())
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

        async def discover_vrchat(
            self, *, preferred_service_id: str | None = None
        ) -> OscQueryServiceInfo | None:
            if self.fail:
                self.fail = False
                raise OSError("controlled discovery failure")
            return await super().discover_vrchat(preferred_service_id=preferred_service_id)

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
async def test_healthy_probe_detects_silent_disappearance_then_waits_for_rediscovery() -> None:
    service = FakeService()
    events: list[tuple[str, object]] = []
    runtime = OscQueryRuntime(
        service=service,
        receiver_start=lambda: _noop(),
        receiver_stop=lambda: _noop(),
        receiver_effective_port=lambda: 49152,
        sender_destination_changed=lambda host, port: events.append((host, port)),
        snapshot_publisher=lambda reason, _generation: events.append(("snapshot", reason)),
        healthy_probe_interval_seconds=0.1,
    )
    await runtime.start("automatic", manual_send_port=9123)
    await asyncio.sleep(0.03)

    assert events.count(("snapshot", "discovery")) == 0

    service.info = None
    await _wait_until(lambda: runtime.service_info is None)

    assert runtime.effective_send_port == 9000
    assert runtime.avatar_tree is None
    discoveries = service.discoveries
    await asyncio.sleep(0.2)
    assert service.discoveries == discoveries

    service.info = OscQueryServiceInfo(
        service_id="vrchat-restarted",
        host="127.0.0.1",
        osc_send_port=9999,
        is_vrchat=True,
    )
    assert service.services_changed is not None
    service.services_changed((service.info,))
    await _wait_until(lambda: runtime.effective_send_port == 9999)

    assert ("127.0.0.1", 9999) in events
    assert events.count(("snapshot", "discovery")) == 1
    await runtime.stop()


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


class LocalDiscovery(ZeroconfOscQueryService):
    async def start(self, services_changed=None) -> None:
        self.started = True
        self._loop = asyncio.get_running_loop()
        self._services_changed = services_changed

    async def advertise_receiver(self, advertisement) -> None:
        self.advertisement = advertisement


@pytest.mark.asyncio
async def test_real_host_info_transient_recovers_through_existing_monitor_without_new_mdns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger="puripuly_heart.core.runtime.oscquery")

    failures = 0
    portless_replies = 0
    recoveries_before_request: list[int] = []
    requests: list[float] = []
    destinations: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
    target = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target.bind(("127.0.0.1", 0))
    target_port = target.getsockname()[1]
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver_port = receiver.getsockname()[1]
    sender = VrchatOscUdpSender()

    async def serve(reader, writer):
        nonlocal failures, portless_replies
        try:
            request = await reader.readuntil(b"\r\n\r\n")
            failed = False
            if b"?HOST_INFO" in request:
                requests.append(asyncio.get_running_loop().time())
                recoveries_before_request.append(
                    caplog.messages.count("[OSCQuery] Discovery recovered")
                )
                if failures:
                    failures -= 1
                    failed = True
                payload = {"OSC_IP": "127.0.0.1", "OSC_PORT": target_port}
                if not failed and portless_replies:
                    portless_replies -= 1
                    payload = {}
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
        healthy_probe_interval_seconds=0.15,
        discovery_retry_delays_seconds=(0.04, 0.08, 0.12),
    )
    try:
        await runtime.start("automatic")
        assert await asyncio.wait_for(destinations.get(), timeout=5) == ("127.0.0.1", target_port)
        assert sender.port == target_port
        failures = 4
        portless_replies = 1
        first_failure = len(requests)
        assert await asyncio.wait_for(destinations.get(), timeout=5) == ("127.0.0.1", 9000)
        assert await asyncio.wait_for(destinations.get(), timeout=5) == ("127.0.0.1", 9000)
        assert await asyncio.wait_for(destinations.get(), timeout=5) == ("127.0.0.1", target_port)
        attempt_times = requests[first_failure:]
        assert len(attempt_times) == 6
        assert recoveries_before_request[first_failure:] == [0] * 6
        for earlier, later, delay in zip(
            attempt_times, attempt_times[1:], (0.04, 0.08, 0.12, 0.12, 0.12)
        ):
            assert later - earlier >= delay
        assert caplog.messages.count("[Failure] structured_failure") == 1
        assert caplog.messages.count("[OSCQuery] Discovery recovered") == 1
        assert runtime.service_info is not None
        assert runtime.effective_receive_port == receiver_port
        assert receiver.fileno() >= 0
        failures = 1
        assert await asyncio.wait_for(destinations.get(), timeout=5) == ("127.0.0.1", 9000)
        assert caplog.messages.count("[Failure] structured_failure") == 2
        assert await asyncio.wait_for(destinations.get(), timeout=5) == ("127.0.0.1", target_port)
        service._queue_service_update(OSCJSON_SERVICE_TYPE, service_name, None)
        assert await asyncio.wait_for(destinations.get(), timeout=5) == ("127.0.0.1", 9000)
        removed_requests = len(requests)
        await asyncio.sleep(0.25)
        assert len(requests) == removed_requests
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
async def test_mixed_outage_survives_mdns_replacement_and_recovers_once(caplog) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    replaced = asyncio.Event()
    alternate_port = None

    class Discovery(LocalDiscovery):
        async def _resolve_host_info(self, service):
            if service.service_id == "VRChat-B":
                return replace(service, osc_send_port=alternate_port)
            if service.query_port == 19011:
                entered.set()
                await release.wait()
            elif service.query_port == 19012:
                replaced.set()
            raise OSError("synthetic HOST_INFO failure")

    logger_name = "puripuly_heart.core.runtime.oscquery"
    caplog.set_level("INFO", logger=logger_name)
    service = Discovery()
    runtime = OscQueryRuntime(
        service,
        _noop,
        _noop,
        lambda: 49152,
        discovery_retry_delays_seconds=(3600,),
    )

    def update(name, port):
        record = (
            SimpleNamespace(parsed_addresses=lambda: ["127.0.0.1"], port=port, properties={})
            if port is not None
            else None
        )
        service._queue_service_update(
            OSCJSON_SERVICE_TYPE, f"{name}.{OSCJSON_SERVICE_TYPE}", record
        )

    def discovery_logs():
        return [
            (record.levelname, getattr(record, DIAGNOSTIC_RECORD_ATTRIBUTE, (None,) * 3)[2])
            for record in caplog.records
            if record.name == logger_name
        ]

    try:
        update("VRChat-A", 19010)
        update("VRChat-B", 19020)
        await runtime.start("automatic")
        assert runtime.effective_send_port == 9000
        assert service.discovery_observation == "unavailable"
        assert discovery_logs() == [("WARNING", "degraded")]

        update("VRChat-B", None)
        update("VRChat-A", 19011)
        await asyncio.wait_for(entered.wait(), 2)
        update("VRChat-A", 19012)
        release.set()
        await asyncio.wait_for(replaced.wait(), 2)
        assert service.discovery_observation == "unavailable"
        assert discovery_logs() == [("WARNING", "degraded")]

        alternate_port = 19021
        update("VRChat-B", 19020)
        assert await runtime.refresh(
            publish_snapshot=False, force_requery=True, report_discovery_failure=True
        )
        assert runtime.effective_send_port == alternate_port
        assert discovery_logs() == [("WARNING", "degraded"), ("INFO", "recovered")]
    finally:
        release.set()
        await runtime.stop()


@pytest.mark.asyncio
async def test_mdns_burst_probes_once_and_removal_cancels_failed_discovery_retry(caplog) -> None:
    requests = []
    preferred_portless = False

    class Discovery(LocalDiscovery):
        async def _resolve_host_info(self, service):
            requests.append(service.service_id)
            if service.service_id == "VRChat":
                if preferred_portless:
                    return replace(service, osc_send_port=19002)
                raise OSError("synthetic HOST_INFO failure")
            if preferred_portless:
                return replace(service, osc_send_port=None)
            return service

    caplog.set_level("INFO", logger="puripuly_heart.core.runtime.oscquery")
    service = Discovery()
    runtime = OscQueryRuntime(
        service,
        _noop,
        _noop,
        lambda: 49152,
        healthy_probe_interval_seconds=0.1,
        discovery_retry_delays_seconds=(0.6,),
    )

    def update(name, service_type, port):
        info = (
            SimpleNamespace(parsed_addresses=lambda: ["127.0.0.1"], port=port, properties={})
            if port is not None
            else None
        )
        service._queue_service_update(service_type, f"{name}.{service_type}", info)

    try:
        await runtime.start("automatic")
        assert service.discovery_observation == "absent"
        update("VRChat", OSCJSON_SERVICE_TYPE, 19010)
        update("VRChat", OSC_SERVICE_TYPE, 19000)
        await _wait_until(lambda: service.discovery_observation == "unavailable")
        update("VRChat", OSCJSON_SERVICE_TYPE, 19010)
        update("VRChat", OSC_SERVICE_TYPE, 19000)
        await asyncio.sleep(0.15)
        assert requests == ["VRChat"]
        assert runtime.effective_send_port == 9000

        assert not await runtime.refresh(publish_snapshot=False, report_discovery_failure=True)
        assert requests == ["VRChat", "VRChat"]
        assert caplog.messages.count("[Failure] structured_failure") == 2

        update("VRChat", OSCJSON_SERVICE_TYPE, None)
        update("VRChat", OSC_SERVICE_TYPE, None)
        await _wait_until(lambda: service.discovery_observation == "absent")
        await asyncio.sleep(0.7)
        assert requests == ["VRChat", "VRChat"]
        assert service.discovery_failure is None
        assert "[OSCQuery] Discovery recovered" not in caplog.messages

        update("VRChat", OSCJSON_SERVICE_TYPE, 19010)
        update("VRChat-good", OSC_SERVICE_TYPE, 19001)
        await _wait_until(lambda: runtime.effective_send_port == 19001)
        assert requests[2:4] == ["VRChat", "VRChat-good"]
        await _wait_until(lambda: len(requests) >= 5)
        assert requests[4] == "VRChat-good"
        attempts = len(requests)
        preferred_portless = True
        await _wait_until(lambda: runtime.effective_send_port == 19002)
        assert requests[attempts : attempts + 2] == ["VRChat-good", "VRChat"]
        assert caplog.messages.count("[Failure] structured_failure") == 2
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_startup_mdns_burst_and_later_destination_metadata_change() -> None:
    requests: list[int | None] = []
    destinations: list[tuple[str, int]] = []
    destination_host = "127.0.0.1"

    class Discovery(LocalDiscovery):
        async def advertise_receiver(self, advertisement) -> None:
            await super().advertise_receiver(advertisement)
            update(OSCJSON_SERVICE_TYPE, 19010)
            asyncio.get_running_loop().call_soon(update, OSC_SERVICE_TYPE, 19000)

        async def _resolve_host_info(self, service):
            requests.append(service.osc_send_port)
            return replace(service, host=destination_host)

    service = Discovery()

    def update(service_type, port):
        record = SimpleNamespace(
            parsed_addresses=lambda: ["127.0.0.1"],
            port=port,
            properties={b"OSC_IP": destination_host.encode("ascii")},
        )
        service._queue_service_update(service_type, f"VRChat.{service_type}", record)

    runtime = OscQueryRuntime(
        service,
        _noop,
        _noop,
        lambda: 49152,
        sender_destination_changed=lambda host, port: destinations.append((host, port)),
    )
    try:
        await runtime.start("automatic")
        assert requests == [19000]
        assert destinations[-1] == ("127.0.0.1", 19000)
        destination_host = "127.0.0.2"
        update(OSCJSON_SERVICE_TYPE, 19010)
        await _wait_until(lambda: destinations[-1] == ("127.0.0.2", 19000))
        assert requests == [19000, 19000]
    finally:
        await runtime.stop()


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
