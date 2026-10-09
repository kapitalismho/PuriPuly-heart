from __future__ import annotations

import pytest

from puripuly_heart.core import updater


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
    monkeypatch.setattr(updater.network_clients, "external_async_client", lambda timeout=5.0: DummyClient(response))

    info = await updater.check_for_update()

    assert info is not None
    assert info.version == "1.2.3"
    assert info.download_url == "https://example/app.exe"
    assert info.release_notes == "release notes"


@pytest.mark.asyncio
async def test_check_for_update_returns_none_on_error(monkeypatch):
    response = DummyResponse(500, {})
    monkeypatch.setattr(updater.network_clients, "external_async_client", lambda timeout=5.0: DummyClient(response))

    assert await updater.check_for_update() is None
