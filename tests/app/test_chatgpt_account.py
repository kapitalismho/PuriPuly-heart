from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from dataclasses import replace
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
from puripuly_heart.core.diagnostics import serialize_diagnostics
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


class _RetryFlow(_Flow):
    def __init__(self, issued_client_id: str | None, *errors: str) -> None:
        super().__init__(issued_client_id=issued_client_id)
        self.errors = errors
        self.authorizations: list[dict[str, str]] = []
        self.token_requests: list[dict[str, str]] = []

    def open_browser(self, url: str) -> None:
        super().open_browser(url)
        self.authorizations.append(self.authorization_params)

    def listener_factory(self, *, locale: str | None = None) -> _RetryListener:
        return _RetryListener(self, len(self.authorizations) + 1)

    def handler(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) == CHATGPT_TOKEN_URL:
            form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
            self.token_requests.append(form)
            index = len(self.token_requests) - 1
            if index < len(self.errors):
                return httpx.Response(
                    400,
                    json={"error": self.errors[index], "error_description": "private-error-detail"},
                )
        return super().handler(request)


class _RetryListener(_Listener):
    def __init__(self, flow: _RetryFlow, attempt: int) -> None:
        super().__init__(flow)
        self.attempt = attempt
        self.redirect_uri = f"http://127.0.0.1:{1454 + attempt}/auth/callback"

    def wait(self, timeout: float | None = None) -> ChatGptOAuthCallback:
        return replace(
            super().wait(timeout),
            code=f"private-code-{self.attempt}",
            client_id=self.flow.issued_client_id if self.attempt == 1 else None,
        )


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


@pytest.mark.parametrize("returning", [False, True])
async def test_invalid_grant_starts_one_fresh_authorization_with_the_issued_client(
    returning: bool,
) -> None:
    store = InMemorySecretStore()
    client = "oaiapp_private"
    if returning:
        store.set(CHATGPT_CLIENT_ID_SECRET, client)
    flow = _RetryFlow(None if returning else client, "invalid_grant")
    owner = _owner(store, flow)

    def observe_authorization(_url: str) -> None:
        assert store.get(CHATGPT_CLIENT_ID_SECRET) == (client if returning else None)
        assert owner.snapshot().signed_in is False

    try:
        result = await owner.connect(authorization_url_sink=observe_authorization)
        first, second = flow.authorizations
        assert first["client_id"] == (client if returning else "dynamic_agent_client")
        assert second["client_id"] == client
        assert first["ext_agent_host_id"] == second["ext_agent_host_id"]
        assert all(first[key] != second[key] for key in ("state", "nonce", "code_challenge"))
        before, after = flow.token_requests
        assert before["code"] != after["code"]
        for authorization, form in zip(flow.authorizations, flow.token_requests, strict=True):
            assert form["client_id"] == client
            assert form["redirect_uri"] == authorization["redirect_uri"]
            assert form["resource"] == authorization["resource"]
            assert (
                _b64url(hashlib.sha256(form["code_verifier"].encode()).digest())
                == authorization["code_challenge"]
            )
        assert (result.succeeded, result.first_sign_in) == (True, not returning)
        assert (await owner.session.access_token(), store.get(CHATGPT_CLIENT_ID_SECRET)) == (
            "access",
            client,
        )

        (recovery,) = result.secondary_diagnostics
        assert (recovery.operation, recovery.code, recovery.status_code) == (
            "token_exchange",
            "chatgpt_invalid_grant",
            400,
        )
        fields = serialize_diagnostics(recovery)["fields"]
        assert fields["authorization_attempt"] == 1
        assert fields["client_id_reused"] is returning
        assert fields["loopback_port"] == 1455
        assert fields["retry_outcome"] == "succeeded"
        exported = json.dumps(serialize_diagnostics(recovery))
        for private in (client, before["code"], before["code_verifier"], "private-error-detail"):
            assert private not in exported
    finally:
        await owner.close()


@pytest.mark.parametrize("errors", [("invalid_grant", "invalid_grant"), ("invalid_client",)])
async def test_terminal_code_exchange_error_keeps_the_existing_connection(errors: tuple) -> None:
    store = InMemorySecretStore()
    flow = _RetryFlow(None, *errors)
    owner = _owner(store, flow)
    client = "oaiapp_existing"
    owner.session.store_sign_in(
        client_id=client,
        tokens=ChatGptTokenSet(
            access_token="existing-access",
            refresh_token="existing-refresh",
            expires_at=time.time() + 3600,
            scopes=(CHATGPT_PLAN_SCOPE,),
        ),
        identity=ChatGptIdentity(subject="existing-subject", email="existing@example.invalid"),
    )
    try:
        result = await owner.connect()
        assert (result.succeeded, result.failure_code) == (False, errors[-1])
        assert [p["client_id"] for p in flow.authorizations] == [client] * len(errors)
        assert len(flow.token_requests) == len(errors)
        assert await owner.session.access_token() == "existing-access"
        assert store.get(CHATGPT_REFRESH_TOKEN_SECRET) == "existing-refresh"
        assert store.get(CHATGPT_CLIENT_ID_SECRET) == client
        assert owner.snapshot().email == "existing@example.invalid"
        fields = serialize_diagnostics(result.diagnostics)["fields"]
        assert fields["authorization_attempt"] == len(errors)
        assert fields["client_id_reused"] is True
        assert fields["loopback_port"] == 1454 + len(errors)
        if len(errors) == 2:
            (recovery,) = result.secondary_diagnostics
            assert serialize_diagnostics(recovery)["fields"]["retry_outcome"] == "failed"
        else:
            assert result.secondary_diagnostics == ()
    finally:
        await owner.close()


async def test_cancelling_during_retry_exchange_does_not_activate_credentials() -> None:
    store = InMemorySecretStore()
    flow = _RetryFlow("oaiapp_uncommitted", "invalid_grant")
    owner = _owner(store, flow)
    exchanging = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == CHATGPT_TOKEN_URL and len(flow.token_requests) == 1:
            exchanging.set()
            await release.wait()
        return flow.handler(request)

    owner.session.http_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connect = asyncio.create_task(owner.connect())
    try:
        await asyncio.wait_for(exchanging.wait(), timeout=5)
        duplicate = await owner.connect(open_browser=False)
        assert duplicate.failure_code == "in_progress"
        owner.cancel()
        release.set()
        result = await asyncio.wait_for(connect, timeout=5)
        assert (result.succeeded, result.failure_code) == (False, "cancelled")
        assert owner.snapshot().in_progress is False
        assert store.get(CHATGPT_CLIENT_ID_SECRET) is None
        assert store.get(CHATGPT_REFRESH_TOKEN_SECRET) is None
        (recovery,) = result.secondary_diagnostics
        assert serialize_diagnostics(recovery)["fields"]["retry_outcome"] == "cancelled"
    finally:
        release.set()
        await asyncio.gather(connect, return_exceptions=True)
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
