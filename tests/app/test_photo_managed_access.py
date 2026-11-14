from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from puripuly_heart.app.adapters.photo_access import PhotoManagedCredentialOwner
from puripuly_heart.core.photo.protocol import PhotoProviderError
from puripuly_heart.core.storage.secrets import InMemorySecretStore


class Token:
    def __init__(self):
        self.requests = 0
        self.invalidated = []

    async def get_token(self):
        self.requests += 1
        return "synthetic-token"

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


def owner(tokens, store, client, identity="synthetic-installation"):
    async def installation():
        return identity

    return PhotoManagedCredentialOwner(
        tokens, installation, lambda: "https://broker.example", lambda: store, lambda: client
    )


async def test_lost_activation_response_recovers_saved_key_without_reissuance_or_per_photo_calls():
    store = InMemorySecretStore()
    tokens = Token()
    actions = []

    def broker(request):
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
            await owner(tokens, store, client).get_credential()
        restarted = owner(tokens, store, client)
        assert await restarted.get_credential() == credential()["key"]
        assert actions == ["get", "acknowledge", "acknowledge"]
        calls = tokens.requests
        assert await restarted.get_credential() == credential()["key"]
        assert await owner(tokens, store, client).get_credential() == credential()["key"]
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
