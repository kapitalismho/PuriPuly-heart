from __future__ import annotations

import asyncio
import json
from pathlib import Path
from threading import Event
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import httpx
import pytest

from puripuly_heart.app.services.chatgpt_account import ChatGptAccountOwner
from puripuly_heart.core.chatgpt.loopback import ChatGptOAuthCallback
from puripuly_heart.core.chatgpt.oauth import (
    CHATGPT_PLAN_SCOPE,
    CHATGPT_REVOKE_URL,
    CHATGPT_TOKEN_URL,
    ChatGptAuthError,
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
from puripuly_heart.core.diagnostics import DIAGNOSTIC_RECORD_ATTRIBUTE, normalize_diagnostic
from puripuly_heart.core.storage.secrets import EncryptedFileSecretStore, InMemorySecretStore

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


async def test_refresh_rotation_preserves_concurrent_managed_delivery(tmp_path: Path) -> None:
    path = tmp_path / "secrets.json"
    retained = EncryptedFileSecretStore(path, passphrase="synthetic")
    retained.set(CHATGPT_CLIENT_ID_SECRET, "oaiapp_client")
    retained.set(CHATGPT_REFRESH_TOKEN_SECRET, "r0")
    retained.set("unrelated", "keep")
    entered = asyncio.Event()
    release = asyncio.Event()
    server = _TokenServer()

    async def handler(request: httpx.Request) -> httpx.Response:
        entered.set()
        await release.wait()
        return server.handler(request)

    session = ChatGptSession(
        secret_store=lambda: retained,
        http_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), trust_env=False
        ),
        clock=lambda: 1000.0,
    )
    refresh = asyncio.create_task(session.access_token())
    managed_values = {
        "openrouter_managed_qq_api_key": "managed-new",
        "openrouter_managed_qq_delivery_ack_token": "ack-new",
        "openrouter_managed_operation_resume_token": "resume-new",
    }
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        managed = EncryptedFileSecretStore(path, passphrase="synthetic")
        for key, value in managed_values.items():
            managed.set(key, value)
        release.set()
        assert await refresh == "a1" + "x" * REAL_ACCESS_TOKEN_LENGTH
        fresh = EncryptedFileSecretStore(path, passphrase="synthetic")
        assert fresh.get(CHATGPT_REFRESH_TOKEN_SECRET) == "r1"
        assert fresh.get("unrelated") == "keep"
        assert {key: fresh.get(key) for key in managed_values} == managed_values
        assert {key: retained.get(key) for key in managed_values} == managed_values
    finally:
        release.set()
        await asyncio.gather(refresh, return_exceptions=True)
        await session.close()


def _authorize(session: ChatGptSession) -> None:
    session.store_sign_in(
        client_id="oaiapp_client",
        tokens=ChatGptTokenSet(
            access_token="authorized-access",
            refresh_token="authorized-refresh",
            expires_at=5000,
            scopes=(CHATGPT_PLAN_SCOPE,),
        ),
        identity=ChatGptIdentity(subject="new-user", email="new@example.invalid"),
    )


@pytest.mark.parametrize("rejected", [False, True])
async def test_retired_refresh_cannot_overwrite_or_delete_new_sign_in(rejected: bool) -> None:
    store = _signed_in_store()
    store.set("unrelated", "keep")
    server = _TokenServer()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == CHATGPT_TOKEN_URL and not entered.is_set():
            entered.set()
            await release.wait()
            if rejected:
                return httpx.Response(400, json={"error": "invalid_grant"})
        return server.handler(request)

    session = ChatGptSession(
        secret_store=lambda: store,
        http_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), trust_env=False
        ),
        clock=lambda: 1000,
    )
    refresh = asyncio.create_task(session.access_token())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert (await session.sign_out()).local_credentials_removed is True
        _authorize(session)
        release.set()
        with pytest.raises(ChatGptReauthRequired):
            await asyncio.wait_for(refresh, 5)
        assert store.get(CHATGPT_REFRESH_TOKEN_SECRET) == "authorized-refresh"
        assert await session.access_token() == "authorized-access"
        assert session.status().email == "new@example.invalid"
        session.invalidate_access_token("authorized-access")
        tokens = await asyncio.wait_for(
            asyncio.gather(*(session.access_token() for _ in range(5))), 5
        )
        assert len(set(tokens)) == 1
        assert server.refresh_calls == (1 if rejected else 2)
        assert store.get("unrelated") == "keep"
        assert session.status().signed_in is True
    finally:
        release.set()
        await asyncio.gather(refresh, return_exceptions=True)
        await session.close()


@pytest.mark.parametrize("revoke_status", [200, 500])
async def test_sign_in_supersedes_logout_before_removal(revoke_status: int) -> None:
    store = _signed_in_store()
    server = _TokenServer()
    server.revoke_status = revoke_status
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == CHATGPT_REVOKE_URL:
            entered.set()
            await release.wait()
        return server.handler(request)

    session = ChatGptSession(
        secret_store=lambda: store,
        http_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), trust_env=False
        ),
        clock=lambda: 1000,
    )
    logout = asyncio.create_task(session.sign_out())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(ChatGptReauthRequired) as error:
            await session.access_token()
        assert error.value.code == "signed_out"
        assert server.refresh_calls == 0
        _authorize(session)
        release.set()
        result = await asyncio.wait_for(logout, 5)
        assert result.local_credentials_removed is False
        assert result.remote_revoked is (revoke_status == 200)
        diagnostics = (result.diagnostics, *result.secondary_diagnostics)
        superseded = next(d for d in diagnostics if d.code == "chatgpt_logout_superseded")
        assert superseded.category == "lifecycle"
        assert superseded.fields["local_credentials_removed"] is False
        records = [
            normalize_diagnostic(
                diagnostic,
                instance_id=str(uuid4()),
                severity="warning",
                disposition="degraded",
            )
            for diagnostic in diagnostics
        ]
        for diagnostic, record in zip(diagnostics, records, strict=True):
            assert record["diagnostic_id"] == diagnostic.diagnostic_id
            assert record["cause_id"] == diagnostic.cause_id
            assert record["code"] == diagnostic.code
            assert record["component"] == "chatgpt_session"
            assert record["operation"] == diagnostic.operation
            assert record["stage"] == diagnostic.fields["phase"]
            assert record["category"] == diagnostic.category
            assert record["status_code"] == diagnostic.status_code
            assert record["fields"] == diagnostic.fields
            assert record["detail"] == {
                "state": "available",
                "rejected_fields": 0,
                "truncated": False,
            }
        if revoke_status == 500:
            assert result.diagnostics.code == "chatgpt_remote_revocation_unconfirmed"
            assert superseded.cause_id == result.diagnostics.diagnostic_id
            assert len(records) == 2
            assert records[0]["status_code"] == 500
            assert records[0]["fields"] == {
                "phase": "remote_revoke",
                "remote_revoked": False,
                "local_credentials_removed": False,
            }
            assert records[1]["cause_id"] == records[0]["diagnostic_id"]
            assert records[1]["fields"] == {
                "phase": "local_credential_delete",
                "local_credentials_removed": False,
            }
        else:
            assert len(records) == 1
            assert records[0]["cause_id"] is None
            assert records[0]["fields"] == {
                "phase": "local_credential_delete",
                "local_credentials_removed": False,
                "remote_revoked": True,
            }
        assert store.get(CHATGPT_REFRESH_TOKEN_SECRET) == "authorized-refresh"
        assert await session.access_token() == "authorized-access"
        session.invalidate_access_token("authorized-access")
        assert await asyncio.wait_for(session.access_token(), 5) == (
            "a1" + "x" * REAL_ACCESS_TOKEN_LENGTH
        )
        assert server.refresh_calls == 1
        assert session.status().signed_in is True
        assert session.status().email == "new@example.invalid"
        assert store.get(CHATGPT_REFRESH_TOKEN_SECRET) == "r1"
    finally:
        release.set()
        await asyncio.gather(logout, return_exceptions=True)
        await session.close()


@pytest.mark.parametrize("revoke_status", [200, 500])
async def test_failed_sign_in_rollback_does_not_supersede_logout(
    tmp_path: Path, revoke_status: int
) -> None:
    path = tmp_path / "secrets.json"

    class FailingStore(EncryptedFileSecretStore):
        fail_sign_in = True

        def set(self, key: str, value: str) -> None:
            super().set(key, value)
            if (
                self.fail_sign_in
                and key == CHATGPT_REFRESH_TOKEN_SECRET
                and value == "authorized-refresh"
            ):
                self.fail_sign_in = False
                raise OSError("synthetic partial sign-in write failure")

    store = FailingStore(path, passphrase="synthetic")
    before = {
        CHATGPT_CLIENT_ID_SECRET: "oaiapp_client",
        CHATGPT_REFRESH_TOKEN_SECRET: "r0",
        CHATGPT_ACCOUNT_SECRET: json.dumps({"subject": "s", "email": "u@example.com"}),
    }
    for key, value in before.items():
        store.set(key, value)
    store.set(CHATGPT_HOST_ID_SECRET, "host")
    store.set("unrelated", "keep")
    entered = asyncio.Event()
    release = asyncio.Event()
    server = _TokenServer()
    server.revoke_status = revoke_status
    authorization_params: dict[str, list[str]] = {}

    def capture_authorization(url: str) -> None:
        authorization_params.update(parse_qs(urlsplit(url).query))

    class Listener:
        redirect_uri = "http://127.0.0.1:1455/auth/callback"
        closed = False

        def wait(self, timeout: float | None = None) -> ChatGptOAuthCallback:
            assert not self.closed
            return ChatGptOAuthCallback(
                state=authorization_params["state"][0],
                code="synthetic-code",
                client_id=None,
                scope=CHATGPT_PLAN_SCOPE,
                error=None,
            )

        def close(self) -> None:
            self.closed = True

    class Verifier:
        async def verify(self, token: str, *, client_id: str, nonce: str) -> ChatGptIdentity:
            return ChatGptIdentity(subject="new-user", email="new@example.invalid")

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == CHATGPT_REVOKE_URL:
            entered.set()
            await release.wait()
        elif str(request.url) == CHATGPT_TOKEN_URL:
            form = parse_qs(request.content.decode())
            if form["grant_type"] == ["authorization_code"]:
                return httpx.Response(
                    200,
                    json={
                        "access_token": "authorized-access",
                        "refresh_token": "authorized-refresh",
                        "id_token": "synthetic-id-token",
                        "expires_in": 3600,
                        "scope": CHATGPT_PLAN_SCOPE,
                    },
                )
        return server.handler(request)

    session = ChatGptSession(
        secret_store=lambda: store,
        http_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), trust_env=False
        ),
        clock=lambda: 1000,
    )
    account = ChatGptAccountOwner(
        session=session,
        listener_factory=lambda **kwargs: Listener(),
        verifier_factory=lambda http: Verifier(),
    )
    logout = asyncio.create_task(account.sign_out())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        failed = await asyncio.wait_for(
            account.connect(open_browser=False, authorization_url_sink=capture_authorization), 5
        )
        assert failed.succeeded is False
        assert failed.failure_code == "credential_store_failed"
        assert failed.diagnostics.code == "chatgpt_credential_store_failed"
        assert failed.diagnostics.operation == "credential_store"
        assert failed.diagnostics.category == "transaction"
        assert failed.diagnostics.fields["exception_class"] == "OSError"
        assert failed.secondary_diagnostics == ()
        fresh = EncryptedFileSecretStore(path, passphrase="synthetic")
        assert {key: fresh.get(key) for key in before} == before
        assert account.snapshot().signed_in is True
        assert not logout.done()

        release.set()
        result = await asyncio.wait_for(logout, 5)
        assert result.local_credentials_removed is True
        assert result.remote_revoked is (revoke_status == 200)
        assert result.secondary_diagnostics == ()
        if revoke_status == 500:
            assert result.diagnostics.code == "chatgpt_remote_revocation_unconfirmed"
            assert result.diagnostics.status_code == 500
            assert result.diagnostics.fields["remote_revoked"] is False
            assert result.diagnostics.fields["local_credentials_removed"] is True
        else:
            assert result.diagnostics is None
        assert server.revoke_calls == 1
        assert server.refresh_calls == 0
        assert fresh.get(CHATGPT_REFRESH_TOKEN_SECRET) is None
        assert account.snapshot().signed_in is False
        assert {key: fresh.get(key) for key in before if key != CHATGPT_REFRESH_TOKEN_SECRET} == {
            key: value for key, value in before.items() if key != CHATGPT_REFRESH_TOKEN_SECRET
        }

        authorized = await asyncio.wait_for(
            account.connect(open_browser=False, authorization_url_sink=capture_authorization), 5
        )
        assert authorized.succeeded is True
        assert await session.access_token() == "authorized-access"
        session.invalidate_access_token("authorized-access")
        tokens = await asyncio.wait_for(
            asyncio.gather(*(session.access_token() for _ in range(5))), 5
        )
        assert set(tokens) == {"a1" + "x" * REAL_ACCESS_TOKEN_LENGTH}
        assert server.refresh_calls == 1
        assert fresh.get(CHATGPT_REFRESH_TOKEN_SECRET) == "r1"
        assert fresh.get(CHATGPT_CLIENT_ID_SECRET) == "oaiapp_client"
        assert fresh.get(CHATGPT_HOST_ID_SECRET) == "host"
        assert fresh.get("unrelated") == "keep"
        assert account.snapshot().signed_in is True
        assert account.snapshot().email == "new@example.invalid"
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(logout, return_exceptions=True), 5)
        await account.close()


@pytest.mark.parametrize("mutation", ["rotation", "rejection", "logout"])
async def test_cancelled_credential_mutation_drains_and_reports_storage_failure(
    mutation: str, caplog: pytest.LogCaptureFixture
) -> None:
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    finished = asyncio.Event()
    release = Event()

    def fail() -> None:
        loop.call_soon_threadsafe(entered.set)
        try:
            if not release.wait(10):
                raise TimeoutError("Mutation gate was not released")
            raise OSError("synthetic-private-secret")
        finally:
            loop.call_soon_threadsafe(finished.set)

    class FailingStore(InMemorySecretStore):
        def set(self, key: str, value: str) -> None:
            if key == CHATGPT_REFRESH_TOKEN_SECRET and value == "r1":
                fail()
            super().set(key, value)

        def delete(self, key: str) -> None:
            if key == CHATGPT_REFRESH_TOKEN_SECRET:
                fail()
            super().delete(key)

    store = FailingStore()
    store.set(CHATGPT_CLIENT_ID_SECRET, "oaiapp_client")
    store.set(CHATGPT_REFRESH_TOKEN_SECRET, "r0")
    server = _TokenServer()
    if mutation == "rejection":
        server.refresh_status = 400
    if mutation == "logout":
        server.revoke_status = 500
    session = _session(store, server, [1000])
    task = asyncio.create_task(
        session.sign_out() if mutation == "logout" else session.access_token()
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), 0.1)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        await asyncio.wait_for(finished.wait(), 5)
        diagnostics = [
            getattr(record, DIAGNOSTIC_RECORD_ATTRIBUTE)[0]
            for record in caplog.records
            if hasattr(record, DIAGNOSTIC_RECORD_ATTRIBUTE)
        ]
        failed = next(d for d in diagnostics if d.fields.get("exception_class") == "OSError")
        assert failed.operation == (
            "refresh_store" if mutation == "rotation" else "local_credential_delete"
        )
        if mutation != "rotation":
            primary = next(d for d in diagnostics if d.diagnostic_id == failed.cause_id)
            assert primary.code == (
                "chatgpt_invalid_grant"
                if mutation == "rejection"
                else "chatgpt_remote_revocation_unconfirmed"
            )
            assert failed.fields["local_credentials_removed"] is False
        assert "synthetic-private-secret" not in repr(diagnostics)
        assert store.get(CHATGPT_REFRESH_TOKEN_SECRET) == "r0"
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 5)
        await session.close()


@pytest.mark.parametrize("mutation", ["rotation", "rejection", "logout"])
async def test_uncancelled_credential_storage_failures_remain_observable(mutation: str) -> None:
    class FailingStore(InMemorySecretStore):
        def set(self, key: str, value: str) -> None:
            if key == CHATGPT_REFRESH_TOKEN_SECRET and value == "r1":
                raise OSError("synthetic write failure")
            super().set(key, value)

        def delete(self, key: str) -> None:
            raise OSError("synthetic cleanup failure")

    store = FailingStore()
    store.set(CHATGPT_CLIENT_ID_SECRET, "oaiapp_client")
    store.set(CHATGPT_REFRESH_TOKEN_SECRET, "r0")
    server = _TokenServer()
    session = _session(store, server, [1000])
    try:
        if mutation == "logout":
            server.revoke_status = 500
            result = await session.sign_out()
            assert result.local_credentials_removed is False
            assert result.remote_revoked is False
            assert result.diagnostics.code == "chatgpt_remote_revocation_unconfirmed"
            failed = result.secondary_diagnostics[0]
            assert failed.cause_id == result.diagnostics.diagnostic_id
        elif mutation == "rejection":
            server.refresh_status = 400
            with pytest.raises(ChatGptReauthRequired) as error:
                await session.access_token()
            assert error.value.diagnostics.fields["local_credentials_removed"] is False
            failed = error.value.secondary_diagnostics[0]
            assert failed.cause_id == error.value.diagnostics.diagnostic_id
        else:
            with pytest.raises(ChatGptAuthError) as error:
                await session.access_token()
            failed = error.value.diagnostics
        assert failed.fields["exception_class"] == "OSError"
        assert failed.operation == (
            "refresh_store" if mutation == "rotation" else "local_credential_delete"
        )
        assert store.get(CHATGPT_REFRESH_TOKEN_SECRET) == "r0"
    finally:
        await session.close()


@pytest.mark.parametrize("rejected", [False, True])
async def test_started_refresh_mutation_settles_before_new_sign_in(
    tmp_path: Path, rejected: bool
) -> None:
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    finished = asyncio.Event()
    release = Event()
    path = tmp_path / "secrets.json"

    def mutate(operation) -> None:
        loop.call_soon_threadsafe(entered.set)
        try:
            if not release.wait(10):
                raise TimeoutError("Mutation gate was not released")
            operation()
        finally:
            loop.call_soon_threadsafe(finished.set)

    class GatedStore(EncryptedFileSecretStore):
        def set(self, key: str, value: str) -> None:
            if key == CHATGPT_REFRESH_TOKEN_SECRET and value == "r1":
                mutate(lambda: super(GatedStore, self).set(key, value))
            else:
                super().set(key, value)

        def delete(self, key: str) -> None:
            mutate(lambda: super(GatedStore, self).delete(key))

    store = GatedStore(path, passphrase="synthetic")
    store.set(CHATGPT_CLIENT_ID_SECRET, "oaiapp_client")
    store.set(CHATGPT_REFRESH_TOKEN_SECRET, "r0")
    store.set(CHATGPT_HOST_ID_SECRET, "host")
    store.set("unrelated", "keep")
    server = _TokenServer()
    if rejected:
        server.refresh_status = 400
    session = ChatGptSession(
        secret_store=lambda: store,
        http_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(server.handler), trust_env=False
        ),
        clock=lambda: 1000,
    )
    refresh = asyncio.create_task(session.access_token())
    sign_in = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        refresh.cancel()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(refresh), 0.1)
        sign_in = asyncio.create_task(asyncio.to_thread(_authorize, session))
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(sign_in), 0.1)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(refresh, 5)
        await asyncio.wait_for(sign_in, 5)
        await asyncio.wait_for(finished.wait(), 5)
        fresh = EncryptedFileSecretStore(path, passphrase="synthetic")
        assert fresh.get(CHATGPT_REFRESH_TOKEN_SECRET) == "authorized-refresh"
        assert session.status().signed_in is True
        assert await session.access_token() == "authorized-access"
        server.refresh_status = 200
        session.invalidate_access_token("authorized-access")
        await asyncio.wait_for(session.access_token(), 5)
        assert fresh.get(CHATGPT_REFRESH_TOKEN_SECRET) == "r2"
        assert fresh.get(CHATGPT_CLIENT_ID_SECRET) == "oaiapp_client"
        assert fresh.get(CHATGPT_HOST_ID_SECRET) == "host"
        assert fresh.get("unrelated") == "keep"
    finally:
        release.set()
        await asyncio.wait_for(
            asyncio.gather(
                refresh, *([sign_in] if sign_in is not None else []), return_exceptions=True
            ),
            5,
        )
        await session.close()


async def test_refresh_transport_evidence_survives_auth_wrapper_without_credentials():
    failure = httpx.ProxyError("synthetic-private-url")
    failure.connection_diagnostics = {
        "transport": "https",
        "proxy_source": "env",
        "stage": "request_send",
        "dispatch_certainty": "possible",
    }

    def handler(_request):
        raise failure

    store = _signed_in_store()
    session = ChatGptSession(
        secret_store=lambda: store,
        http_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    try:
        with pytest.raises(ChatGptAuthError) as raised:
            await session.access_token()
        diagnostics = raised.value.diagnostics
        assert diagnostics.code == "chatgpt_refresh_network_error"
        assert diagnostics.category == "network"
        assert diagnostics.fields["proxy_source"] == "env"
        assert diagnostics.fields["dispatch_certainty"] == "possible"
        assert session.status().signed_in is True
        assert "synthetic-private" not in str(diagnostics)
        assert "u@example.com" not in str(diagnostics)
    finally:
        await session.close()


async def test_refresh_invalid_json_retains_status_and_shape_without_response_body():
    store = _signed_in_store()
    session = ChatGptSession(
        secret_store=lambda: store,
        http_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, text="synthetic-private-body")
            )
        ),
    )
    try:
        with pytest.raises(ChatGptAuthError) as raised:
            await session.access_token()
        assert raised.value.diagnostics.status_code == 200
        assert raised.value.diagnostics.category == "invalid_response"
        assert raised.value.diagnostics.fields["response_shape"] == "non_json"
        assert store.get(CHATGPT_REFRESH_TOKEN_SECRET) == "r0"
        assert "synthetic-private-body" not in str(raised.value.diagnostics)
    finally:
        await session.close()


async def test_sign_out_timeout_preserves_remote_failure_and_local_cleanup():
    def handler(_request):
        raise TimeoutError("synthetic-private-transport")

    store = _signed_in_store()
    session = ChatGptSession(
        secret_store=lambda: store,
        http_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    try:
        result = await session.sign_out()
        assert result.remote_revoked is False
        assert result.local_credentials_removed is True
        assert result.diagnostics.category == "timeout"
        assert result.diagnostics.fields["exception_class"] == "TimeoutError"
        assert store.get(CHATGPT_REFRESH_TOKEN_SECRET) is None
        assert "synthetic-private" not in str(result)
    finally:
        await session.close()


async def test_refresh_malformed_object_is_invalid_response_without_retiring_credentials():
    store = _signed_in_store()
    session = ChatGptSession(
        secret_store=lambda: store,
        http_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={}))
        ),
    )
    try:
        with pytest.raises(ChatGptAuthError) as raised:
            await session.access_token()
        assert raised.value.diagnostics.category == "invalid_response"
        assert raised.value.diagnostics.status_code == 200
        assert raised.value.diagnostics.fields["response_shape"] == "malformed_success"
        assert store.get(CHATGPT_REFRESH_TOKEN_SECRET) == "r0"
        assert session.status().signed_in is True
    finally:
        await session.close()


async def test_refresh_without_rotated_refresh_token_keeps_existing_authority():
    store = _signed_in_store()
    session = ChatGptSession(
        secret_store=lambda: store,
        http_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200, json={"access_token": "synthetic-access", "scope": CHATGPT_PLAN_SCOPE}
                )
            )
        ),
    )
    try:
        assert await session.access_token() == "synthetic-access"
        assert store.get(CHATGPT_REFRESH_TOKEN_SECRET) == "r0"
    finally:
        await session.close()
