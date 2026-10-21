from __future__ import annotations

import asyncio
import threading

import pytest

from puripuly_heart.core import updater


@pytest.fixture(autouse=True)
def isolated_transport_preparation(monkeypatch):
    monkeypatch.setattr(updater.external_network, "select_tls", object)
    monkeypatch.setattr(updater.external_network, "ProxyPolicy", object)


class DummyResponse:
    def __init__(self, status_code: int, data: dict) -> None:
        self.status_code = status_code
        self._data = data

    def json(self) -> dict:
        return self._data


class DummyClient:
    def __init__(self, response: DummyResponse) -> None:
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _tb):
        return None

    async def get(self, *_args, **_kwargs):
        return self._response


def test_parse_version_handles_prefix_and_prerelease():
    assert updater._parse_version("v0.1.0-beta") == (0, 1, 0)


def test_is_newer_compares_versions():
    assert updater._is_newer("0.2.0", "0.1.9") is True
    assert updater._is_newer("0.1.0", "0.1.0") is False


@pytest.mark.asyncio
async def test_check_for_update_returns_info(monkeypatch):
    data = {
        "tag_name": "v1.2.3",
        "assets": [{"name": "app.exe", "browser_download_url": "https://example/app.exe"}],
        "body": "release notes",
    }
    response = DummyResponse(200, data)

    monkeypatch.setattr(updater, "__version__", "0.1.0")
    monkeypatch.setattr(
        updater.network_clients, "external_async_client", lambda **_kwargs: DummyClient(response)
    )

    info = await updater.check_for_update()

    assert info is not None
    assert info.version == "1.2.3"
    assert info.download_url == "https://example/app.exe"
    assert info.release_notes == "release notes"


@pytest.mark.asyncio
async def test_check_for_update_returns_none_on_error(monkeypatch):
    response = DummyResponse(500, {})
    monkeypatch.setattr(
        updater.network_clients, "external_async_client", lambda **_kwargs: DummyClient(response)
    )

    assert await updater.check_for_update() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_update_transport_preparation_keeps_loop_responsive_and_drains_on_cancel(
    monkeypatch, cancel
):
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    entered = asyncio.Event()
    release = threading.Event()
    preparation_threads = []
    client_events = []
    tls = object()
    policy = object()

    def prepare_tls():
        preparation_threads.append(threading.get_ident())
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(3)
        return tls

    class Client(DummyClient):
        async def __aexit__(self, *_exc):
            client_events.append(("closed", threading.get_ident()))

        async def get(self, *_args, **_kwargs):
            client_events.append(("request", threading.get_ident()))
            return self._response

    def create_client(**kwargs):
        assert kwargs == {"tls": tls, "policy": policy, "timeout": 5.0}
        client_events.append(("created", threading.get_ident()))
        return Client(DummyResponse(200, {"tag_name": "v999.0.0"}))

    monkeypatch.setattr(updater.external_network, "select_tls", prepare_tls)
    monkeypatch.setattr(updater.external_network, "ProxyPolicy", lambda: policy)
    monkeypatch.setattr(updater.network_clients, "external_async_client", create_client)
    task = asyncio.create_task(updater.check_for_update())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.sleep(0)
        assert not task.done()
        assert preparation_threads != [loop_thread]
        if cancel:
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        release.set()
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await task
            assert client_events == []
        else:
            info = await task
            assert info is not None
            assert info.version == "999.0.0"
            assert client_events == [
                ("created", loop_thread),
                ("request", loop_thread),
                ("closed", loop_thread),
            ]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_update_transport_preparation_failure_is_best_effort(monkeypatch):
    def fail():
        raise OSError("synthetic TLS preparation failure")

    monkeypatch.setattr(updater.external_network, "select_tls", fail)
    assert await updater.check_for_update() is None
