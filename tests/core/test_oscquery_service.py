from __future__ import annotations

import asyncio
import json
import logging
import os

import pytest
import zeroconf

from puripuly_heart.app.ports.oscquery import OscQueryAdvertisement, OscQueryServiceInfo
from puripuly_heart.core.osc.oscquery import (
    OSC_SERVICE_TYPE,
    OSCQUERY_ADVERTISEMENT_TTL_SECONDS,
    ZeroconfOscQueryService,
    mdns_instance_name,
)


@pytest.mark.asyncio
async def test_zeroconf_service_advertises_and_serves_the_puripuly_tree() -> None:
    service = ZeroconfOscQueryService()
    await service.start()
    try:
        await service.advertise_receiver(
            OscQueryAdvertisement(
                host="127.0.0.1",
                port=49152,
                parameters={"/avatar": "PuriPuly_*"},
            )
        )
        assert service.advertisement is not None
        assert service.advertisement.query_port is not None

        response = await _request(service.advertisement.query_port, "/")
        assert response.split(b"\r\n", 1)[0] == b"HTTP/1.1 200 OK"
        payload = json.loads(response.split(b"\r\n\r\n", 1)[1])
        parameters = payload["CONTENTS"]["avatar"]["CONTENTS"]["parameters"]["CONTENTS"]
        assert len(parameters) == 16
        assert parameters["PuriPuly_Talk"]["TYPE"] == "T"
        assert parameters["PuriPuly_SelfASR"]["TYPE"] == "i"
        assert parameters["MuteSelf"] == {
            "FULL_PATH": "/avatar/parameters/MuteSelf",
            "ACCESS": 2,
            "TYPE": "T",
        }

        avatar_response = await _request(service.advertisement.query_port, "/avatar")
        avatar_payload = json.loads(avatar_response.split(b"\r\n\r\n", 1)[1])
        assert avatar_response.split(b"\r\n", 1)[0] == b"HTTP/1.1 200 OK"
        assert avatar_payload["FULL_PATH"] == "/avatar"

        parameter_response = await _request(
            service.advertisement.query_port,
            "/avatar/parameters/PuriPuly_Talk",
        )
        parameter_payload = json.loads(parameter_response.split(b"\r\n\r\n", 1)[1])
        assert parameter_response.split(b"\r\n", 1)[0] == b"HTTP/1.1 200 OK"
        assert parameter_payload["FULL_PATH"] == "/avatar/parameters/PuriPuly_Talk"

        access_response = await _request(
            service.advertisement.query_port,
            "/avatar/parameters/PuriPuly_Talk?ACCESS",
        )
        assert access_response.split(b"\r\n", 1)[0] == b"HTTP/1.1 200 OK"
        assert json.loads(access_response.split(b"\r\n\r\n", 1)[1]) == {"ACCESS": 3}

        value_response = await _request(
            service.advertisement.query_port,
            "/avatar/parameters/PuriPuly_Talk?VALUE",
        )
        assert value_response.split(b"\r\n", 1)[0] == b"HTTP/1.1 200 OK"
        assert json.loads(value_response.split(b"\r\n\r\n", 1)[1]) == {"VALUE": [False]}

        mute_self_response = await _request(
            service.advertisement.query_port,
            "/avatar/parameters/MuteSelf",
        )
        mute_self_payload = json.loads(mute_self_response.split(b"\r\n\r\n", 1)[1])
        assert mute_self_payload["ACCESS"] == 2
        assert mute_self_payload["TYPE"] == "T"

        avatar_change_response = await _request(
            service.advertisement.query_port,
            "/avatar/change",
        )
        avatar_change_payload = json.loads(avatar_change_response.split(b"\r\n\r\n", 1)[1])
        assert avatar_change_payload["ACCESS"] == 2

        avatar_change_access_response = await _request(
            service.advertisement.query_port,
            "/avatar/change?ACCESS",
        )
        assert avatar_change_access_response.split(b"\r\n", 1)[0] == b"HTTP/1.1 200 OK"
        assert json.loads(avatar_change_access_response.split(b"\r\n\r\n", 1)[1]) == {"ACCESS": 2}

        avatar_change_value_response = await _request(
            service.advertisement.query_port,
            "/avatar/change?VALUE",
        )
        assert avatar_change_value_response.split(b"\r\n", 1)[0] == (b"HTTP/1.1 204 No Content")
        assert avatar_change_value_response.endswith(b"\r\n\r\n")

        host_info_response = await _request(service.advertisement.query_port, "/?HOST_INFO")
        host_info_payload = json.loads(host_info_response.split(b"\r\n\r\n", 1)[1])
        assert host_info_response.split(b"\r\n", 1)[0] == b"HTTP/1.1 200 OK"
        assert host_info_payload["NAME"] == "PuriPuly Heart"
        assert host_info_payload["OSC_IP"] == "127.0.0.1"
        assert host_info_payload["OSC_PORT"] == 49152

        missing_response = await _request(service.advertisement.query_port, "/unknown")
        assert missing_response.split(b"\r\n", 1)[0] == b"HTTP/1.1 404 Not Found"

        invalid_query_response = await _request(
            service.advertisement.query_port,
            "/avatar/parameters/PuriPuly_Talk?ACCESS&VALUE",
        )
        assert invalid_query_response.split(b"\r\n", 1)[0] == b"HTTP/1.1 400 Bad Request"

        duplicate_query_targets = (
            "/avatar/parameters/PuriPuly_Talk?ACCESS&ACCESS",
            "/avatar/parameters/PuriPuly_Talk?VALUE&VALUE",
            "/?HOST_INFO&HOST_INFO",
        )
        for target in duplicate_query_targets:
            duplicate_query_response = await _request(
                service.advertisement.query_port,
                target,
            )
            assert duplicate_query_response.split(b"\r\n", 1)[0] == (b"HTTP/1.1 400 Bad Request")
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_zeroconf_start_cleans_up_partial_browser_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeZeroconf:
        instances: list[FakeZeroconf] = []

        def __init__(self) -> None:
            self.closed = False
            self.__class__.instances.append(self)

        def close(self) -> None:
            self.closed = True

    class FakeBrowser:
        instances: list[FakeBrowser] = []

        def __init__(self, _zeroconf: object, service_type: str, *, handlers: object) -> None:
            self.cancelled = False
            self.__class__.instances.append(self)
            if service_type == OSC_SERVICE_TYPE:
                raise RuntimeError("second browser failed")

        def cancel(self) -> None:
            self.cancelled = True

    monkeypatch.setattr(zeroconf, "Zeroconf", FakeZeroconf)
    monkeypatch.setattr(zeroconf, "ServiceBrowser", FakeBrowser)
    service = ZeroconfOscQueryService()

    with pytest.raises(RuntimeError, match="second browser failed"):
        await service.start()

    assert service.started is False
    assert service._zeroconf is None
    assert service._browsers == []
    assert service._loop is None
    assert FakeZeroconf.instances[0].closed is True


def test_mdns_instance_name_stays_stable_for_a_process() -> None:
    first = mdns_instance_name("PuriPuly Heart")
    second = mdns_instance_name("PuriPuly Heart", pid=os.getpid())

    assert first == second
    assert str(os.getpid()) in first
    assert first.startswith("PuriPuly-Heart-")


@pytest.mark.asyncio
async def test_advertise_registers_unique_pid_name_with_short_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered: list[object] = []

    class FakeZeroconf:
        def __init__(self) -> None:
            self.loop = None

        def register_service(self, info: object, *, allow_name_change: bool = False) -> None:
            registered.append((info, allow_name_change))

        def unregister_service(self, info: object) -> None:
            return None

        def close(self) -> None:
            return None

    class FakeBrowser:
        def __init__(self, _zeroconf: object, _service_type: str, *, handlers: object) -> None:
            _ = handlers

        def cancel(self) -> None:
            return None

    monkeypatch.setattr(zeroconf, "Zeroconf", FakeZeroconf)
    monkeypatch.setattr(zeroconf, "ServiceBrowser", FakeBrowser)
    service = ZeroconfOscQueryService()
    await service.start()
    try:
        await service.advertise_receiver(
            OscQueryAdvertisement(
                host="127.0.0.1",
                port=49152,
                parameters={"/avatar": "PuriPuly_*"},
            )
        )
    finally:
        await service.stop()

    assert len(registered) == 2
    instance = mdns_instance_name("PuriPuly Heart")
    for info, allow_name_change in registered:
        assert allow_name_change is True
        assert info.name.startswith(f"{instance}.")
        assert info.host_ttl == OSCQUERY_ADVERTISEMENT_TTL_SECONDS
        assert info.other_ttl == OSCQUERY_ADVERTISEMENT_TTL_SECONDS


@pytest.mark.asyncio
async def test_dead_puripuly_advertisement_is_forgotten() -> None:
    service = ZeroconfOscQueryService()
    service._services["PuriPuly-Heart-1"] = OscQueryServiceInfo(
        service_id="PuriPuly-Heart-1",
        host="127.0.0.1",
        query_port=1,
    )

    await service._forget_unreachable_puripuly_services()

    assert "PuriPuly-Heart-1" not in service._services


@pytest.mark.asyncio
async def test_partial_register_unregisters_successful_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unregistered: list[object] = []

    class FakeZeroconf:
        def __init__(self) -> None:
            self.loop = None

        def register_service(self, info: object, *, allow_name_change: bool = False) -> None:
            _ = allow_name_change
            if str(info.type).startswith("_osc._udp"):
                raise RuntimeError("second register failed")

        def unregister_service(self, info: object) -> None:
            unregistered.append(info)

        def close(self) -> None:
            return None

    class FakeBrowser:
        def __init__(self, _zeroconf: object, _service_type: str, *, handlers: object) -> None:
            _ = handlers

        def cancel(self) -> None:
            return None

    monkeypatch.setattr(zeroconf, "Zeroconf", FakeZeroconf)
    monkeypatch.setattr(zeroconf, "ServiceBrowser", FakeBrowser)
    service = ZeroconfOscQueryService()
    await service.start()
    with pytest.raises(RuntimeError, match="second register failed"):
        await service.advertise_receiver(
            OscQueryAdvertisement(
                host="127.0.0.1",
                port=49152,
                parameters={"/avatar": "PuriPuly_*"},
            )
        )

    assert len(unregistered) == 1
    assert str(unregistered[0].type).startswith("_oscjson._tcp")
    assert service._registered_infos == []
    await service.stop()


@pytest.mark.asyncio
async def test_discovery_resolves_the_osc_destination_from_host_info(
    caplog: pytest.LogCaptureFixture,
) -> None:
    requests: list[str] = []

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        request = await reader.readline()
        requests.append(request.decode("ascii").split()[1])
        payload = json.dumps(
            {
                "NAME": "VRChat-Client",
                "OSC_IP": "127.0.0.1",
                "OSC_PORT": 9123,
            }
        ).encode("utf-8")
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode("ascii")
            + payload
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", 0)
    query_port = int(server.sockets[0].getsockname()[1])
    service = ZeroconfOscQueryService()
    service._services["VRChat-Client"] = OscQueryServiceInfo(
        service_id="VRChat-Client",
        host="127.0.0.1",
        query_port=query_port,
        is_vrchat=True,
    )
    caplog.set_level(logging.DEBUG, logger="httpx")
    caplog.set_level(logging.DEBUG, logger="httpcore")
    caplog.clear()
    try:
        discovered = await service.discover_vrchat()
        assert discovered is not None
        avatar = await service.query_avatar(discovered)
    finally:
        server.close()
        await server.wait_closed()

    assert discovered is not None
    assert discovered.osc_send_port == 9123
    assert avatar["NAME"] == "VRChat-Client"
    assert requests == ["/?HOST_INFO", "/avatar"]
    logging.getLogger("httpx").info("unrelated HTTP request remains visible")
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == "httpx" or record.name.startswith("httpcore.")
    ] == ["unrelated HTTP request remains visible"]


@pytest.mark.asyncio
async def test_discovery_rejects_a_stale_oscquery_service() -> None:
    server = await asyncio.start_server(lambda _reader, _writer: None, "127.0.0.1", 0)
    query_port = int(server.sockets[0].getsockname()[1])
    server.close()
    await server.wait_closed()
    service = ZeroconfOscQueryService()
    service._services["VRChat-Client"] = OscQueryServiceInfo(
        service_id="VRChat-Client",
        host="127.0.0.1",
        query_port=query_port,
        osc_send_port=9000,
        is_vrchat=True,
    )

    assert await service.discover_vrchat() is None


async def _request(port: int | None, target: str) -> bytes:
    assert port is not None
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(f"GET {target} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".encode())
    await writer.drain()
    response = await reader.read()
    writer.close()
    await writer.wait_closed()
    return response
