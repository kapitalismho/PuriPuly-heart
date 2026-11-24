from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from puripuly_heart.app.adapters.photo_access import PhotoManagedCredentialOwner
from puripuly_heart.config.service_endpoints import (
    DEFAULT_OPENROUTER_BROKER_BASE_URL,
    LEGACY_OPENROUTER_BROKER_BASE_URL,
)
from puripuly_heart.core.network_clients import first_party_user_agent
from puripuly_heart.core.photo.protocol import PhotoProviderError
from puripuly_heart.core.storage.secrets import InMemorySecretStore


class Token:
    def __init__(self):
        self.requests = 0
        self.invalidated = []

    async def get_token(self):
        self.requests += 1
        return "replacement-token" if self.invalidated else "synthetic-token"

    def invalidate(self, token):
        self.invalidated.append(token)


def credential():
    return {
        "key": "synthetic-restricted-photo-key",
        "operation_id": "synthetic-operation",
        "managed_credential_ref": "synthetic-key-hash",
        "delivery_id": "synthetic-delivery",
        "delivery_ack_token": "synthetic-ack",
        "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        "model": "openai/gpt-6-luna",
        "acknowledged": False,
    }


def owner(tokens, store, client, identity="synthetic-installation", base_url="https://broker.example"):
    async def installation():
        return identity

    return PhotoManagedCredentialOwner(
        tokens, installation, lambda: base_url, lambda: store, lambda: client
    )


async def test_lost_activation_recovers_saved_key_after_origin_cutover_without_reissuance():
    store = InMemorySecretStore()
    tokens = Token()
    actions = []
    requests = []

    def broker(request):
        assert request.url.path == "/v1/photo/access"
        assert request.headers["user-agent"] == first_party_user_agent()
        requests.append(request)
        body = json.loads(request.content)
        actions.append(body["action"])
        if body["action"] == "get":
            return httpx.Response(200, json=credential())
        saved = json.loads(next(iter(store._items.values())))
        assert saved["key"] == credential()["key"]
        assert saved["acknowledged"] is False
        if actions.count("acknowledge") == 1:
            raise httpx.ReadTimeout("Lost successful activation response", request=request)
        return httpx.Response(
            200, json={"acknowledged": True, "operation_id": "synthetic-operation"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(broker)) as client:
        with pytest.raises(PhotoProviderError):
            await owner(
                tokens, store, client, base_url=LEGACY_OPENROUTER_BROKER_BASE_URL
            ).get_credential()
        restarted = owner(tokens, store, client, base_url=DEFAULT_OPENROUTER_BROKER_BASE_URL)
        assert await restarted.get_credential() == credential()["key"]
        assert actions == ["get", "acknowledge", "acknowledge"]
        assert [str(request.url) for request in requests] == [
            LEGACY_OPENROUTER_BROKER_BASE_URL + "/v1/photo/access",
            LEGACY_OPENROUTER_BROKER_BASE_URL + "/v1/photo/access",
            "https://puripuly-broker.salee.sarl/v1/photo/access",
        ]
        calls = tokens.requests
        assert await restarted.get_credential() == credential()["key"]
        assert (
            await owner(
                tokens, store, client, base_url=DEFAULT_OPENROUTER_BROKER_BASE_URL
            ).get_credential()
            == credential()["key"]
        )
        assert tokens.requests == calls
        assert actions == ["get", "acknowledge", "acknowledge"]


async def test_secret_persistence_failure_never_activates_delivered_key():
    class UnwritableStore(InMemorySecretStore):
        def set(self, key, value):
            raise OSError("Isolated secure storage failure")

    actions = []

    def broker(request):
        actions.append(json.loads(request.content)["action"])
        return httpx.Response(200, json=credential())

    async with httpx.AsyncClient(transport=httpx.MockTransport(broker)) as client:
        with pytest.raises(OSError):
            await owner(Token(), UnwritableStore(), client).get_credential()
    assert actions == ["get"]


async def test_cached_access_is_bound_to_installation_and_revocation_requires_upstream_confirmation():
    store = InMemorySecretStore()
    tokens = Token()
    actions = []
    confirmed = False

    def broker(request):
        body = json.loads(request.content)
        actions.append(body["action"])
        if body["action"] == "get":
            return httpx.Response(200, json=credential())
        if body["action"] == "revoke":
            return httpx.Response(200 if confirmed else 202, json={"revoked": confirmed})
        return httpx.Response(
            200, json={"acknowledged": True, "operation_id": "synthetic-operation"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(broker)) as client:
        first = owner(tokens, store, client)
        await first.get_credential()
        await owner(tokens, store, client, identity="other-installation").get_credential()
        assert actions == ["get", "acknowledge", "get", "acknowledge"]
        assert await first.revoke() is False
        assert len(store._items) == 2
        calls = list(actions)
        with pytest.raises(PhotoProviderError):
            await first.get_credential()
        assert actions == calls
        confirmed = True
        assert await first.revoke() is True
        assert len(store._items) == 1


@pytest.mark.parametrize(
    ("status", "kwargs", "code", "invalidate"),
    [
        (401, {"json": {"error": {"code": "invalid_token"}}}, "photo_authentication_failed", True),
        (401, {"text": "<html>private-response</html>"}, "photo_stream_invalid", False),
        (
            401,
            {
                "headers": {"CF-Mitigated": "challenge"},
                "json": {"error": {"code": "invalid_token"}},
            },
            "photo_stream_invalid",
            False,
        ),
        (
            403,
            {"headers": {"CF-Ray": "1234567890abcdef-ICN"}},
            "photo_request_rejected",
            False,
        ),
        (401, {"json": {"error": {"code": "unknown_auth_code"}}}, "photo_request_rejected", False),
        (429, {"json": {"error": {"code": "rate_limited"}}}, "photo_quota_exhausted", False),
    ],
)
async def test_photo_broker_ack_recovery_never_reissues_credentials(
    caplog, status, kwargs, code, invalidate
):
    tokens = Token()
    store = InMemorySecretStore()
    requests = []

    def broker(request):
        action = json.loads(request.content)["action"]
        requests.append((action, request.headers["authorization"]))
        if action == "get":
            return httpx.Response(200, json=credential())
        if len(requests) == 2:
            return httpx.Response(status, **kwargs)
        return httpx.Response(200, json={"acknowledged": True, "operation_id": "synthetic-operation"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(broker)) as client:
        with pytest.raises(PhotoProviderError) as failure:
            await owner(tokens, store, client).get_credential()
        assert failure.value.code == code
        assert tokens.invalidated == (["synthetic-token"] if invalidate else [])
        assert len(requests) == 2
        assert len(store._items) == 1
        assert json.loads(next(iter(store._items.values())))["acknowledged"] is False
        assert await owner(tokens, store, client).get_credential() == credential()["key"]
        assert requests == [
            ("get", "Bearer synthetic-token"),
            ("acknowledge", "Bearer synthetic-token"),
            ("acknowledge", "Bearer replacement-token" if invalidate else "Bearer synthetic-token"),
        ]
    assert "private-response" not in caplog.text
    assert "synthetic-token" not in caplog.text
