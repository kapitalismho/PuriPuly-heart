from __future__ import annotations

import asyncio
import json

import pytest
import zeroconf

from puripuly_heart.app.ports.oscquery import OscQueryAdvertisement
from puripuly_heart.core.osc.oscquery import (
    OSC_SERVICE_TYPE,
    ZeroconfOscQueryService,
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
        assert len(parameters) == 15
        assert parameters["PuriPuly_Talk"]["TYPE"] == "T"
        assert parameters["PuriPuly_SelfASR"]["TYPE"] == "i"

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
    assert FakeBrowser.instances[0].cancelled is True


async def _request(port: int | None, target: str) -> bytes:
    assert port is not None
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(f"GET {target} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".encode())
    await writer.drain()
    response = await reader.read()
    writer.close()
    await writer.wait_closed()
    return response
