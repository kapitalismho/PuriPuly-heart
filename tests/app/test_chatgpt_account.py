from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from puripuly_heart.app.services.chatgpt_account import ChatGptAccountOwner
from puripuly_heart.core.chatgpt.loopback import ChatGptOAuthCallback
from puripuly_heart.core.chatgpt.oauth import (
    CHATGPT_ISSUER,
    CHATGPT_JWKS_URL,
    CHATGPT_PLAN_SCOPE,
    CHATGPT_TOKEN_URL,
    ChatGptIdentity,
    ChatGptTokenSet,
)
from puripuly_heart.core.chatgpt.session import (
    CHATGPT_CLIENT_ID_SECRET,
    CHATGPT_HOST_ID_SECRET,
    CHATGPT_REFRESH_TOKEN_SECRET,
    ChatGptSession,
)
from puripuly_heart.core.storage.secrets import (
    EncryptedFileSecretStore,
    InMemorySecretStore,
    SecretStore,
)

_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _jwk() -> dict[str, str]:
    numbers = _KEY.public_key().public_numbers()
    return {
        "kty": "RSA",
        "kid": "k1",
        "alg": "RS256",
        "n": _b64url(numbers.n.to_bytes((numbers.n.bit_length() + 7) // 8, "big")),
        "e": _b64url(numbers.e.to_bytes((numbers.e.bit_length() + 7) // 8, "big")),
    }


def _id_token(*, client_id: str, nonce: str) -> str:
    header = _b64url(json.dumps({"alg": "RS256", "kid": "k1"}).encode())
    payload = _b64url(
        json.dumps(
            {
                "iss": CHATGPT_ISSUER,
                "aud": client_id,
                "sub": "user-sub",
                "email": "user@example.com",
                "nonce": nonce,
                "exp": int(time.time()) + 3600,
            }
        ).encode()
    )
    signature = _KEY.sign(f"{header}.{payload}".encode(), padding.PKCS1v15(), hashes.SHA256())
    return f"{header}.{payload}.{_b64url(signature)}"


class _Flow:
    def __init__(
        self,
        *,
        issued_client_id: str | None = "oaiapp_new",
        scope: str = f"openid email offline_access {CHATGPT_PLAN_SCOPE}",
        error: str | None = None,
        tamper_state: bool = False,
    ) -> None:
        self.issued_client_id = issued_client_id
        self.scope = scope
        self.error = error
        self.tamper_state = tamper_state
        self.authorization_params: dict[str, str] = {}
        self.exchanged: dict[str, str] = {}

    def listener_factory(self, *, locale: str | None = None) -> _Listener:
        return _Listener(self)

    def open_browser(self, url: str) -> None:
        self.authorization_params = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}

    def handler(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) == CHATGPT_JWKS_URL:
            return httpx.Response(200, json={"keys": [_jwk()]})
        if str(request.url) == CHATGPT_TOKEN_URL:
            self.exchanged = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
            client_id = self.exchanged["client_id"]
            return httpx.Response(
                200,
                json={
                    "access_token": "access",
                    "refresh_token": "refresh",
                    "id_token": _id_token(
                        client_id=client_id, nonce=self.authorization_params["nonce"]
                    ),
                    "expires_in": 3600,
                    "scope": self.scope,
                },
            )
        return httpx.Response(404)


class _Listener:
    redirect_uri = "http://127.0.0.1:1455/auth/callback"

    def __init__(self, flow: _Flow) -> None:
        self.flow = flow
        self.closed = False

    def wait(self, timeout: float | None = None) -> ChatGptOAuthCallback:
        params = self.flow.authorization_params
        return ChatGptOAuthCallback(
            state="tampered" if self.flow.tamper_state else params["state"],
            code=None if self.flow.error else "auth-code",
            client_id=self.flow.issued_client_id,
            scope=self.flow.scope,
            error=self.flow.error,
        )

    def close(self) -> None:
        self.closed = True


def _owner(store: SecretStore, flow: _Flow) -> ChatGptAccountOwner:
    session = ChatGptSession(
        secret_store=lambda: store,
        http_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(flow.handler)),
    )
    return ChatGptAccountOwner(
        session=session,
        browser_opener=flow.open_browser,
        listener_factory=flow.listener_factory,
    )


async def test_first_sign_in_registers_client_and_stores_only_small_credentials() -> None:
    store = InMemorySecretStore()
    flow = _Flow()
    owner = _owner(store, flow)
    urls: list[str] = []

    result = await owner.connect(authorization_url_sink=urls.append)

    assert result.succeeded is True
    assert result.first_sign_in is True
    assert result.email == "user@example.com"
    assert len(urls) == 1
    assert flow.authorization_params["client_id"] == "dynamic_agent_client"
    assert flow.exchanged["client_id"] == "oaiapp_new"
    assert flow.exchanged["redirect_uri"] == _Listener.redirect_uri
    assert store.get(CHATGPT_CLIENT_ID_SECRET) == "oaiapp_new"
    assert store.get(CHATGPT_REFRESH_TOKEN_SECRET) == "refresh"
    snapshot = owner.snapshot()
    assert snapshot.signed_in is True
    assert snapshot.email == "user@example.com"
    assert snapshot.in_progress is False
    assert await owner.session.access_token() == "access"
    await owner.close()


async def test_returning_sign_in_reuses_saved_client_and_is_not_first() -> None:
    store = InMemorySecretStore()
    store.set(CHATGPT_CLIENT_ID_SECRET, "oaiapp_saved")
    flow = _Flow(issued_client_id=None)
    owner = _owner(store, flow)

    result = await owner.connect()

    assert result.succeeded is True
    assert result.first_sign_in is False
    assert flow.authorization_params["client_id"] == "oaiapp_saved"
    assert "agent_name_hint" not in flow.authorization_params
    assert flow.exchanged["client_id"] == "oaiapp_saved"
    await owner.close()


async def test_sign_in_rejects_state_mismatch_without_storing_credentials() -> None:
    store = InMemorySecretStore()
    owner = _owner(store, _Flow(tamper_state=True))

    result = await owner.connect()

    assert result.succeeded is False
    assert result.failure_code == "state_mismatch"
    assert store.get(CHATGPT_REFRESH_TOKEN_SECRET) is None
    await owner.close()


async def test_declined_consent_reports_access_denied() -> None:
    owner = _owner(InMemorySecretStore(), _Flow(error="access_denied"))
    result = await owner.connect()
    assert (result.succeeded, result.failure_code) == (False, "access_denied")
    await owner.close()


async def test_sign_in_without_plan_scope_is_not_stored() -> None:
    store = InMemorySecretStore()
    owner = _owner(store, _Flow(scope="openid email offline_access"))

    result = await owner.connect()

    assert (result.succeeded, result.failure_code) == (False, "plan_scope_missing")
    assert store.get(CHATGPT_REFRESH_TOKEN_SECRET) is None
    await owner.close()


async def test_reauthorization_rejects_callback_for_a_different_client() -> None:
    store = InMemorySecretStore()
    store.set(CHATGPT_CLIENT_ID_SECRET, "oaiapp_saved")
    owner = _owner(store, _Flow(issued_client_id="oaiapp_other"))

    result = await owner.connect()

    assert (result.succeeded, result.failure_code) == (False, "client_mismatch")
    assert store.get(CHATGPT_CLIENT_ID_SECRET) == "oaiapp_saved"
    await owner.close()


async def test_logout_allows_subsequent_authorized_sign_in_and_refresh(tmp_path: Path) -> None:
    path = tmp_path / "secrets.json"
    store = EncryptedFileSecretStore(path, passphrase="synthetic")
    refreshes = []

    class ReturningFlow(_Flow):
        def handler(self, request: httpx.Request) -> httpx.Response:
            form = parse_qs(request.content.decode())
            if form.get("grant_type") == ["refresh_token"]:
                refreshes.append(form["refresh_token"][0])
                return httpx.Response(
                    200,
                    json={
                        "access_token": "rotated-access",
                        "refresh_token": "rotated-refresh",
                        "expires_in": 3600,
                        "scope": CHATGPT_PLAN_SCOPE,
                    },
                )
            return super().handler(request)

    flow = ReturningFlow(issued_client_id=None)
    owner = _owner(store, flow)
    host = owner.session.host_id()
    store.set("unrelated", "keep")
    owner.session.store_sign_in(
        client_id="oaiapp_saved",
        tokens=ChatGptTokenSet(
            access_token="old-access",
            refresh_token="old-refresh",
            expires_at=0,
            scopes=(CHATGPT_PLAN_SCOPE,),
        ),
        identity=ChatGptIdentity(subject="old-user", email="old@example.invalid"),
    )
    try:
        logout = await owner.sign_out()
        assert logout.local_credentials_removed is True
        assert logout.remote_revoked is False
        assert logout.diagnostics.code == "chatgpt_remote_revocation_unconfirmed"
        fresh = EncryptedFileSecretStore(path, passphrase="synthetic")
        assert fresh.get(CHATGPT_REFRESH_TOKEN_SECRET) is None
        assert owner.snapshot().signed_in is False
        signed_in = await owner.connect()
        assert signed_in.succeeded is True
        assert signed_in.first_sign_in is False
        assert flow.authorization_params["client_id"] == "oaiapp_saved"
        assert await owner.session.access_token() == "access"
        owner.session.invalidate_access_token("access")
        assert await owner.session.access_token() == "rotated-access"
        assert refreshes == ["refresh"]
        assert owner.snapshot().signed_in is True
        assert fresh.get(CHATGPT_REFRESH_TOKEN_SECRET) == "rotated-refresh"
        assert fresh.get(CHATGPT_CLIENT_ID_SECRET) == "oaiapp_saved"
        assert fresh.get(CHATGPT_HOST_ID_SECRET) == host
        assert fresh.get("unrelated") == "keep"
    finally:
        await owner.close()


class _BlockedExchangeFlow(_Flow):
    def __init__(self) -> None:
        super().__init__()
        self.exchange_started = asyncio.Event()
        self.release_exchange = asyncio.Event()
        self.listener_count = 0

    def listener_factory(self, *, locale: str | None = None) -> _Listener:
        self.listener_count += 1
        return super().listener_factory(locale=locale)

    async def handler(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) == CHATGPT_TOKEN_URL:
            form = parse_qs(request.content.decode())
            if form.get("grant_type") == ["authorization_code"]:
                self.exchange_started.set()
                await self.release_exchange.wait()
        return super().handler(request)


@pytest.mark.asyncio
async def test_cancel_after_callback_prevents_late_credential_commit() -> None:
    store = InMemorySecretStore()
    flow = _BlockedExchangeFlow()
    owner = _owner(store, flow)
    connecting = asyncio.create_task(owner.connect())
    try:
        await asyncio.wait_for(flow.exchange_started.wait(), timeout=2)
        assert owner.in_progress
        owner.cancel()
        result = await asyncio.wait_for(connecting, timeout=2)
        assert result.succeeded is False
        assert result.failure_code == "cancelled"
        assert store.get(CHATGPT_REFRESH_TOKEN_SECRET) is None
        assert not flow.release_exchange.is_set()
    finally:
        flow.release_exchange.set()
        await asyncio.gather(connecting, return_exceptions=True)
        await owner.close()


@pytest.mark.asyncio
async def test_second_connect_after_callback_cannot_compete_with_active_exchange() -> None:
    store = InMemorySecretStore()
    flow = _BlockedExchangeFlow()
    owner = _owner(store, flow)
    connecting = asyncio.create_task(owner.connect())
    try:
        await asyncio.wait_for(flow.exchange_started.wait(), timeout=2)
        duplicate = await asyncio.wait_for(owner.connect(open_browser=False), timeout=2)
        assert duplicate.succeeded is False
        assert duplicate.failure_code == "in_progress"
        assert flow.listener_count == 1
        owner.cancel()
        result = await asyncio.wait_for(connecting, timeout=2)
        assert result.failure_code == "cancelled"
        assert store.get(CHATGPT_REFRESH_TOKEN_SECRET) is None
    finally:
        flow.release_exchange.set()
        await asyncio.gather(connecting, return_exceptions=True)
        await owner.close()


@pytest.mark.asyncio
async def test_cancel_after_credential_commit_started_reports_settled_outcome() -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingStore(InMemorySecretStore):
        def set(self, key: str, value: str) -> None:
            if key == CHATGPT_REFRESH_TOKEN_SECRET:
                entered.set()
                assert release.wait(2)
            super().set(key, value)

    store = BlockingStore()
    owner = _owner(store, _Flow())
    connecting = asyncio.create_task(owner.connect())
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        owner.cancel()
        release.set()
        result = await asyncio.wait_for(connecting, timeout=2)
        assert result.succeeded is True
        assert store.get(CHATGPT_REFRESH_TOKEN_SECRET) == "refresh"
    finally:
        release.set()
        await asyncio.gather(connecting, return_exceptions=True)
        await owner.close()


@pytest.mark.asyncio
async def test_close_after_callback_cancels_and_drains_active_exchange() -> None:
    store = InMemorySecretStore()
    flow = _BlockedExchangeFlow()
    owner = _owner(store, flow)
    connecting = asyncio.create_task(owner.connect())
    await asyncio.wait_for(flow.exchange_started.wait(), timeout=2)
    await asyncio.wait_for(owner.close(), timeout=2)
    result = await asyncio.wait_for(connecting, timeout=2)
    assert result.succeeded is False
    assert result.failure_code == "cancelled"
    assert store.get(CHATGPT_REFRESH_TOKEN_SECRET) is None
    assert not flow.release_exchange.is_set()
