from __future__ import annotations

import httpx
import pytest

from puripuly_heart.app.adapters import photo_provider
from puripuly_heart.app.adapters.photo_provider import PhotoProviderAdapter


class _Tokens:
    def __init__(self) -> None:
        self.invalidated: list[str] = []

    async def get_token(self) -> str:
        return "synthetic-token"

    def invalidate(self, token: str) -> None:
        self.invalidated.append(token)


def _adapter(monkeypatch, handler, tokens=None) -> PhotoProviderAdapter:
    monkeypatch.setattr(
        photo_provider.network_clients,
        "external_async_client",
        lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return PhotoProviderAdapter(
        session=None,
        secret_store=lambda: {},
        token_owner=tokens or _Tokens(),
        relay_endpoint="https://relay.test/v1/photo",
    )


async def test_allowance_reads_remaining_photos_from_the_relay(monkeypatch):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"allowance": 100, "remaining": 72, "reset_after_s": 60})

    assert await _adapter(monkeypatch, handler).allowance() == (72, 100)
    assert str(requests[0].url) == "https://relay.test/v1/photo/allowance"
    assert requests[0].headers["authorization"] == "Bearer synthetic-token"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"allowance": 100, "remaining": 101}),
        httpx.Response(200, json={"allowance": 0, "remaining": 0}),
        httpx.Response(200, json={"allowance": "100", "remaining": 1}),
        httpx.Response(200, json=[1, 2]),
        httpx.Response(503, json={"error": {"code": "service_shutdown"}}),
    ],
)
async def test_allowance_rejects_unusable_relay_answers(monkeypatch, response):
    assert await _adapter(monkeypatch, lambda request: response).allowance() is None


async def test_allowance_invalidates_a_rejected_token(monkeypatch):
    tokens = _Tokens()
    adapter = _adapter(monkeypatch, lambda request: httpx.Response(401), tokens)

    assert await adapter.allowance() is None
    assert tokens.invalidated == ["synthetic-token"]
