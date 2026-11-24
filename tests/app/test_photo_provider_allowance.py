from __future__ import annotations

import ssl

import httpx
import pytest

from puripuly_heart.app.adapters import photo_provider
from puripuly_heart.app.adapters.photo_provider import PhotoProviderAdapter
from puripuly_heart.core.network_clients import first_party_user_agent


class _Tokens:
    def __init__(self) -> None:
        self.invalidated: list[str] = []

    async def get_token(self) -> str:
        return "replacement-token" if self.invalidated else "synthetic-token"

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
    )


async def test_allowance_reads_remaining_photos_from_the_relay(monkeypatch):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"allowance": 100, "remaining": 72, "reset_after_s": 60})

    assert await _adapter(monkeypatch, handler).allowance() == (72, 100)
    assert str(requests[0].url) == "https://puripuly-relay.salee.sarl/v1/photo/allowance"
    assert requests[0].headers["authorization"] == "Bearer synthetic-token"
    assert requests[0].headers["user-agent"] == first_party_user_agent()


async def test_cached_allowance_reuses_todays_relay_answer_without_a_request(monkeypatch):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"allowance": 50, "remaining": 0})

    adapter = _adapter(monkeypatch, handler)
    assert adapter.cached_allowance() is None
    assert await adapter.allowance() == (0, 50)
    assert adapter.cached_allowance() == (0, 50)
    assert len(requests) == 1


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


async def test_allowance_recovers_with_fresh_token_only_after_application_rejection(monkeypatch):
    tokens = _Tokens()
    requests = []

    def handler(request):
        requests.append(request.headers["authorization"])
        if len(requests) == 1:
            return httpx.Response(
                401,
                json={
                    "error": {
                        "code": "service_authentication_failed",
                        "message": "The assistant service could not authenticate.",
                    }
                },
            )
        return httpx.Response(200, json={"allowance": 100, "remaining": 99})

    adapter = _adapter(monkeypatch, handler, tokens)

    assert await adapter.allowance() is None
    assert tokens.invalidated == ["synthetic-token"]
    assert requests == ["Bearer synthetic-token"]
    assert await adapter.allowance() == (99, 100)
    assert requests == ["Bearer synthetic-token", "Bearer replacement-token"]


@pytest.mark.parametrize(
    ("status", "kwargs", "shape"),
    [
        (401, {"text": "<html>private-body</html>"}, "html"),
        (
            401,
            {
                "headers": {"CF-Mitigated": "challenge"},
                "json": {"error": {"code": "service_authentication_failed"}},
            },
            "edge_challenge",
        ),
        (
            403,
            {
                "headers": {"CF-Ray": "1234567890abcdef-ICN"},
                "json": {"error": {"code": "service_authentication_failed"}},
            },
            "public_error_envelope",
        ),
        (401, {"text": "private-body"}, "unknown"),
    ],
)
async def test_allowance_does_not_invalidate_on_unverified_rejection(
    monkeypatch, caplog, status, kwargs, shape
):
    tokens = _Tokens()
    adapter = _adapter(monkeypatch, lambda _: httpx.Response(status, **kwargs), tokens)
    assert await adapter.allowance() is None
    assert not tokens.invalidated
    assert f"'response_shape': '{shape}'" in caplog.text
    assert "'http_status': " + str(status) in caplog.text
    assert "private-body" not in caplog.text
    assert "synthetic-token" not in caplog.text


@pytest.mark.parametrize("kind", ["proxy", "tls"])
async def test_allowance_logs_only_safe_transport_evidence(monkeypatch, caplog, kind):
    def fail(request):
        failure = (
            httpx.ProxyError("private-proxy-credentials", request=request)
            if kind == "proxy"
            else ssl.SSLCertVerificationError("private-certificate")
        )
        error = httpx.ConnectError("private-network-error", request=request)
        error.connection_diagnostics = {
            "tls_source": "windows",
            "proxy_source": "system",
            "stage": "request_send",
            "proxy_url": "private-proxy-url",
        }
        raise error from failure

    tokens = _Tokens()
    assert await _adapter(monkeypatch, fail, tokens).allowance() is None
    assert not tokens.invalidated
    assert f"'network_error_kind': '{kind}'" in caplog.text
    assert "'tls_source': 'windows'" in caplog.text
    assert "'proxy_source': 'system'" in caplog.text
    assert "private-" not in caplog.text
