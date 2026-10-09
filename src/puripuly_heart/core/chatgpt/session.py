from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final, Protocol

import httpx

from puripuly_heart.core import network_clients
from puripuly_heart.core.chatgpt.oauth import (
    ChatGptAuthError,
    ChatGptIdentity,
    ChatGptReauthRequired,
    ChatGptTokenSet,
    new_host_id,
    refresh_access_token,
    revoke_refresh_token,
    token_set_from_response,
)
from puripuly_heart.core.storage.secrets import SecretStore

CHATGPT_REFRESH_TOKEN_SECRET: Final = "chatgpt_refresh_token"
CHATGPT_CLIENT_ID_SECRET: Final = "chatgpt_client_id"
CHATGPT_HOST_ID_SECRET: Final = "chatgpt_host_id"
CHATGPT_ACCOUNT_SECRET: Final = "chatgpt_account"
CHATGPT_SECRET_KEYS: Final[tuple[str, ...]] = (
    CHATGPT_REFRESH_TOKEN_SECRET,
    CHATGPT_CLIENT_ID_SECRET,
    CHATGPT_HOST_ID_SECRET,
    CHATGPT_ACCOUNT_SECRET,
)
_REFRESH_MARGIN_S: Final = 300.0


class ChatGptAccessTokenPort(Protocol):
    @property
    def token_generation(self) -> int: ...

    async def access_token(self) -> str: ...

    def invalidate_access_token(self, token: str) -> None: ...


@dataclass(frozen=True, slots=True)
class ChatGptSessionStatus:
    signed_in: bool
    email: str | None


@dataclass(slots=True)
class ChatGptSession:
    secret_store: Callable[[], SecretStore]
    http_factory: Callable[[], httpx.AsyncClient] = lambda: network_clients.external_async_client(timeout=20.0)
    clock: Callable[[], float] = time.time
    refresh_margin_s: float = _REFRESH_MARGIN_S
    _tokens: ChatGptTokenSet | None = field(init=False, default=None, repr=False)
    _lock: asyncio.Lock | None = field(init=False, default=None, repr=False)
    _http: httpx.AsyncClient | None = field(init=False, default=None, repr=False)
    _generation: int = field(init=False, default=0, repr=False)

    @property
    def token_generation(self) -> int:
        return self._generation

    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = self.http_factory()
        return self._http

    def status(self) -> ChatGptSessionStatus:
        store = self.secret_store()
        signed_in = bool(store.get(CHATGPT_REFRESH_TOKEN_SECRET)) and bool(
            store.get(CHATGPT_CLIENT_ID_SECRET)
        )
        return ChatGptSessionStatus(signed_in=signed_in, email=self._stored_email(store))

    def client_id(self) -> str | None:
        return self.secret_store().get(CHATGPT_CLIENT_ID_SECRET) or None

    def login_hint(self) -> str | None:
        return self._stored_email(self.secret_store())

    def host_id(self) -> str:
        store = self.secret_store()
        existing = store.get(CHATGPT_HOST_ID_SECRET)
        if existing:
            return existing
        value = new_host_id()
        store.set(CHATGPT_HOST_ID_SECRET, value)
        return value

    def store_sign_in(
        self,
        *,
        client_id: str,
        tokens: ChatGptTokenSet,
        identity: ChatGptIdentity,
    ) -> None:
        store = self.secret_store()
        store.set(CHATGPT_CLIENT_ID_SECRET, client_id)
        store.set(CHATGPT_REFRESH_TOKEN_SECRET, tokens.refresh_token)
        store.set(
            CHATGPT_ACCOUNT_SECRET,
            json.dumps({"subject": identity.subject, "email": identity.email}),
        )
        self._tokens = tokens
        self._generation += 1

    async def access_token(self) -> str:
        tokens = self._tokens
        if tokens is not None and tokens.expires_at - self.refresh_margin_s > self.clock():
            return tokens.access_token
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            tokens = self._tokens
            if tokens is not None and tokens.expires_at - self.refresh_margin_s > self.clock():
                return tokens.access_token
            return (await self._refresh()).access_token

    def invalidate_access_token(self, token: str) -> None:
        tokens = self._tokens
        if tokens is not None and tokens.access_token == token:
            self._tokens = None

    async def sign_out(self) -> bool:
        store = self.secret_store()
        client_id = store.get(CHATGPT_CLIENT_ID_SECRET)
        refresh_token = store.get(CHATGPT_REFRESH_TOKEN_SECRET)
        revoked = False
        if client_id and refresh_token:
            try:
                revoked = await revoke_refresh_token(
                    self.http(), client_id=client_id, refresh_token=refresh_token
                )
            except httpx.HTTPError:
                revoked = False
        store.delete(CHATGPT_REFRESH_TOKEN_SECRET)
        self._tokens = None
        self._generation += 1
        return revoked

    async def close(self) -> None:
        http = self._http
        self._http = None
        if http is not None:
            await http.aclose()

    async def _refresh(self) -> ChatGptTokenSet:
        store = self.secret_store()
        client_id = store.get(CHATGPT_CLIENT_ID_SECRET)
        refresh_token = store.get(CHATGPT_REFRESH_TOKEN_SECRET)
        if not client_id or not refresh_token:
            raise ChatGptReauthRequired("signed_out")
        try:
            payload = await refresh_access_token(
                self.http(), client_id=client_id, refresh_token=refresh_token
            )
        except ChatGptReauthRequired:
            await asyncio.to_thread(store.delete, CHATGPT_REFRESH_TOKEN_SECRET)
            self._tokens = None
            self._generation += 1
            raise
        except httpx.HTTPError as exc:
            raise ChatGptAuthError("refresh_network_error") from exc
        tokens = token_set_from_response(
            payload, previous_refresh_token=refresh_token, now=self.clock()
        )
        if not tokens.plan_usage_granted:
            raise ChatGptReauthRequired("plan_scope_missing")
        if tokens.refresh_token != refresh_token:
            await asyncio.to_thread(store.set, CHATGPT_REFRESH_TOKEN_SECRET, tokens.refresh_token)
        self._tokens = tokens
        self._generation += 1
        return tokens

    @staticmethod
    def _stored_email(store: SecretStore) -> str | None:
        raw = store.get(CHATGPT_ACCOUNT_SECRET)
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except ValueError:
            return None
        email = data.get("email") if isinstance(data, dict) else None
        return email if isinstance(email, str) and email else None


__all__ = [
    "CHATGPT_ACCOUNT_SECRET",
    "CHATGPT_CLIENT_ID_SECRET",
    "CHATGPT_HOST_ID_SECRET",
    "CHATGPT_REFRESH_TOKEN_SECRET",
    "CHATGPT_SECRET_KEYS",
    "ChatGptAccessTokenPort",
    "ChatGptSession",
    "ChatGptSessionStatus",
]
