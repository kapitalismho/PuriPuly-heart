from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from puripuly_heart.core.osc.oscquery import (
    OSC_SERVICE_TYPE,
    OSCJSON_SERVICE_TYPE,
    ZeroconfOscQueryService,
)

VRCHAT = "VRChat-4242"
MDNS_SEND_PORT = 19000
_TARGET_PORT = 9123


class FakeZeroconf:
    def __init__(self, *, query_port: int) -> None:
        self.query_port = query_port
        self.unresolvable: set[str] = set()

    def get_service_info(self, service_type: str, _name: str, *, timeout: int) -> object | None:
        _ = timeout
        if service_type in self.unresolvable:
            return None
        if service_type == OSCJSON_SERVICE_TYPE:
            return SimpleNamespace(
                parsed_addresses=lambda: ["127.0.0.1"],
                port=self.query_port,
                properties={b"NAME": VRCHAT.encode()},
            )
        return SimpleNamespace(
            parsed_addresses=lambda: ["127.0.0.1"],
            port=MDNS_SEND_PORT,
            properties={
                b"NAME": VRCHAT.encode(),
                b"OSC_IP": b"127.0.0.1",
                b"OSC_PORT": str(MDNS_SEND_PORT).encode(),
            },
        )


def _state_change(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


async def _flush() -> None:
    for _ in range(4):
        await asyncio.sleep(0)


async def _start_host_info_server(host_info_port: int) -> asyncio.Server:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readline()
        payload = json.dumps({"OSC_IP": "127.0.0.1", "OSC_PORT": host_info_port}).encode()
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode("ascii")
            + payload
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    return await asyncio.start_server(handle, "127.0.0.1", 0)


async def _register(
    service: ZeroconfOscQueryService,
    zeroconf: FakeZeroconf,
) -> None:
    service._service_state_changed(
        zeroconf,
        OSCJSON_SERVICE_TYPE,
        f"{VRCHAT}.{OSCJSON_SERVICE_TYPE}",
        _state_change("Registered"),
    )
    service._service_state_changed(
        zeroconf,
        OSC_SERVICE_TYPE,
        f"{VRCHAT}.{OSC_SERVICE_TYPE}",
        _state_change("Registered"),
    )
    await _flush()


@pytest.mark.asyncio
async def test_transient_mdns_timeout_keeps_the_known_vrchat_candidate() -> None:
    server = await _start_host_info_server(_TARGET_PORT)
    query_port = int(server.sockets[0].getsockname()[1])
    zeroconf = FakeZeroconf(query_port=query_port)
    service = ZeroconfOscQueryService()
    service._loop = asyncio.get_running_loop()
    try:
        await _register(service, zeroconf)
        registered = service._services[VRCHAT]
        assert registered.query_port == query_port
        assert registered.osc_send_port == MDNS_SEND_PORT

        zeroconf.unresolvable = {OSCJSON_SERVICE_TYPE, OSC_SERVICE_TYPE}
        service._service_state_changed(
            zeroconf,
            OSCJSON_SERVICE_TYPE,
            f"{VRCHAT}.{OSCJSON_SERVICE_TYPE}",
            _state_change("Changed"),
        )
        service._service_state_changed(
            zeroconf,
            OSC_SERVICE_TYPE,
            f"{VRCHAT}.{OSC_SERVICE_TYPE}",
            _state_change("Changed"),
        )
        await _flush()

        assert service._services[VRCHAT] == registered
        discovered = await service.discover_vrchat()
        assert discovered is not None
        assert discovered.osc_send_port == _TARGET_PORT
    finally:
        server.close()
        await server.wait_closed()
        await service.stop()


@pytest.mark.asyncio
async def test_transient_oscquery_timeout_keeps_the_query_port_for_host_info_resolution() -> None:
    server = await _start_host_info_server(_TARGET_PORT)
    query_port = int(server.sockets[0].getsockname()[1])
    zeroconf = FakeZeroconf(query_port=query_port)
    service = ZeroconfOscQueryService()
    service._loop = asyncio.get_running_loop()
    try:
        await _register(service, zeroconf)

        zeroconf.unresolvable = {OSCJSON_SERVICE_TYPE}
        service._service_state_changed(
            zeroconf,
            OSCJSON_SERVICE_TYPE,
            f"{VRCHAT}.{OSCJSON_SERVICE_TYPE}",
            _state_change("Changed"),
        )
        await _flush()

        retained = service._services[VRCHAT]
        assert retained.query_port == query_port
        discovered = await service.discover_vrchat()
        assert discovered is not None
        assert discovered.osc_send_port == _TARGET_PORT
    finally:
        server.close()
        await server.wait_closed()
        await service.stop()


@pytest.mark.asyncio
async def test_actual_mdns_removal_still_clears_the_candidate() -> None:
    server = await _start_host_info_server(_TARGET_PORT)
    zeroconf = FakeZeroconf(query_port=int(server.sockets[0].getsockname()[1]))
    service = ZeroconfOscQueryService()
    service._loop = asyncio.get_running_loop()
    try:
        await _register(service, zeroconf)

        service._service_state_changed(
            zeroconf,
            OSCJSON_SERVICE_TYPE,
            f"{VRCHAT}.{OSCJSON_SERVICE_TYPE}",
            _state_change("Removed"),
        )
        await _flush()
        partial = service._services[VRCHAT]
        assert partial.query_port is None
        assert partial.osc_send_port == MDNS_SEND_PORT

        service._service_state_changed(
            zeroconf,
            OSC_SERVICE_TYPE,
            f"{VRCHAT}.{OSC_SERVICE_TYPE}",
            _state_change("Removed"),
        )
        await _flush()
        assert VRCHAT not in service._services
        assert await service.discover_vrchat() is None
    finally:
        server.close()
        await server.wait_closed()
        await service.stop()
