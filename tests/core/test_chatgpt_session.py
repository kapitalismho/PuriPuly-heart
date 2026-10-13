from __future__ import annotations

import asyncio
import json
from urllib.parse import parse_qs

import httpx
import pytest

from puripuly_heart.core.chatgpt.oauth import (
    CHATGPT_PLAN_SCOPE,
    CHATGPT_REVOKE_URL,
    CHATGPT_TOKEN_URL,
    ChatGptIdentity,
    ChatGptReauthRequired,
    ChatGptTokenSet,
)
from puripuly_heart.core.chatgpt.session import (
    CHATGPT_ACCOUNT_SECRET,
    CHATGPT_CLIENT_ID_SECRET,
    CHATGPT_HOST_ID_SECRET,
    CHATGPT_REFRESH_TOKEN_SECRET,
    ChatGptSession,
)
from puripuly_heart.core.storage.secrets import InMemorySecretStore

WINDOWS_CREDENTIAL_MANAGER_MAX_CHARS = 1280
REAL_ACCESS_TOKEN_LENGTH = 2105


class _TokenServer:
    def __init__(self) -> None:
        self.refresh_calls = 0
        self.revoke_calls = 0
        self.refresh_status = 200
        self.refresh_error = "invalid_grant"
        self.revoke_status = 200

    def handler(self, request: httpx.Request) -> httpx.Response:
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        if str(request.url) == CHATGPT_TOKEN_URL:
            assert form["grant_type"] == "refresh_token"
            assert form["client_id"] == "oaiapp_client"
            self.refresh_calls += 1
            if self.refresh_status != 200:
                return httpx.Response(self.refresh_status, json={"error": self.refresh_error})
            return httpx.Response(
                200,
                json={
                    "access_token": f"a{self.refresh_calls}" + "x" * REAL_ACCESS_TOKEN_LENGTH,
                    "refresh_token": f"r{self.refresh_calls}",
                    "expires_in": 3600,
                    "scope": f"openid offline_access {CHATGPT_PLAN_SCOPE}",
                },
            )
        if str(request.url) == CHATGPT_REVOKE_URL:
            assert form["token_type_hint"] == "refresh_token"
            self.revoke_calls += 1
            return httpx.Response(self.revoke_status)
        return httpx.Response(404)


def _session(store: InMemorySecretStore, server: _TokenServer, now: list[float]) -> ChatGptSession:
    return ChatGptSession(
        secret_store=lambda: store,
        http_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(server.handler)),
        clock=lambda: now[0],
    )


def _signed_in_store() -> InMemorySecretStore:
    store = InMemorySecretStore()
    store.set(CHATGPT_CLIENT_ID_SECRET, "oaiapp_client")
    store.set(CHATGPT_REFRESH_TOKEN_SECRET, "r0")
    store.set(CHATGPT_ACCOUNT_SECRET, json.dumps({"subject": "s", "email": "u@example.com"}))
    return store


async def test_access_token_refreshes_once_and_persists_rotated_refresh_token() -> None:
    store = _signed_in_store()
    server = _TokenServer()
    now = [1000.0]
    session = _session(store, server, now)

    tokens = await asyncio.gather(*(session.access_token() for _ in range(5)))

    assert len(set(tokens)) == 1
    assert server.refresh_calls == 1
    assert store.get(CHATGPT_REFRESH_TOKEN_SECRET) == "r1"
    generation = session.token_generation
    now[0] += 3600 - 200
    await session.access_token()
    assert server.refresh_calls == 2
    assert session.token_generation == generation + 1
    await session.close()


async def test_stored_values_fit_windows_credential_manager_even_with_long_access_tokens() -> None:
    store = _signed_in_store()
    session = _session(store, _TokenServer(), [1000.0])

    token = await session.access_token()
    session.host_id()

    assert len(token) > WINDOWS_CREDENTIAL_MANAGER_MAX_CHARS
    stored = [
        store.get(key)
        for key in (
            CHATGPT_REFRESH_TOKEN_SECRET,
            CHATGPT_CLIENT_ID_SECRET,
            CHATGPT_HOST_ID_SECRET,
            CHATGPT_ACCOUNT_SECRET,
        )
    ]
    assert all(value is not None for value in stored)
    assert all(len(value) <= WINDOWS_CREDENTIAL_MANAGER_MAX_CHARS for value in stored)
    assert token not in stored
    await session.close()


async def test_terminal_refresh_failure_signs_out_but_keeps_registration() -> None:
    store = _signed_in_store()
    server = _TokenServer()
    server.refresh_status = 400
    session = _session(store, server, [1000.0])

    with pytest.raises(ChatGptReauthRequired):
        await session.access_token()

    assert session.status().signed_in is False
    assert store.get(CHATGPT_CLIENT_ID_SECRET) == "oaiapp_client"
    assert session.login_hint() == "u@example.com"
    await session.close()


async def test_signed_out_session_requires_reauthorization() -> None:
    session = _session(InMemorySecretStore(), _TokenServer(), [1000.0])
    with pytest.raises(ChatGptReauthRequired):
        await session.access_token()


async def test_sign_in_then_sign_out_revokes_and_keeps_client_and_host() -> None:
    store = InMemorySecretStore()
    server = _TokenServer()
    session = _session(store, server, [1000.0])
    host_id = session.host_id()
    session.store_sign_in(
        client_id="oaiapp_client",
        tokens=ChatGptTokenSet(
            access_token="a",
            refresh_token="r-signed-in",
            expires_at=5000.0,
            scopes=(CHATGPT_PLAN_SCOPE,),
        ),
        identity=ChatGptIdentity(subject="s", email="u@example.com"),
    )

    assert session.status().signed_in is True
    assert session.status().email == "u@example.com"
    assert await session.access_token() == "a"
    assert server.refresh_calls == 0

    result = await session.sign_out()
    assert result.remote_revoked is True
    assert result.local_credentials_removed is True
    assert server.revoke_calls == 1
    assert session.status().signed_in is False
    assert store.get(CHATGPT_CLIENT_ID_SECRET) == "oaiapp_client"
    assert session.host_id() == host_id
    await session.close()


async def test_sign_out_reports_unconfirmed_remote_revocation() -> None:
    store = _signed_in_store()
    server = _TokenServer()
    server.revoke_status = 500
    session = _session(store, server, [1000.0])

    result = await session.sign_out()
    assert result.remote_revoked is False
    assert result.local_credentials_removed is True
    assert result.diagnostics.status_code == 500
    assert result.diagnostics.operation == "remote_revoke"
    assert store.get(CHATGPT_REFRESH_TOKEN_SECRET) is None
    await session.close()
