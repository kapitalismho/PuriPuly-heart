from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
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
from puripuly_heart.core.diagnostics import emit_failure, safe_exception_fields
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.storage.secrets import SecretStore

logger = logging.getLogger(__name__)


def _session_diagnostics(
    code: str,
    phase: str,
    exception: BaseException | None = None,
    *,
    status_code: int | None = None,
    **fields: bool,
) -> ErrorDiagnostics:
    owning = getattr(exception, "diagnostics", None)
    if isinstance(owning, ErrorDiagnostics):
        return replace(owning, fields={**owning.fields, **fields})
    return ErrorDiagnostics(
        component="chatgpt_session",
        operation=phase,
        code=code if code.startswith("chatgpt_") else f"chatgpt_{code}",
        category=(
            "network"
            if code in {"refresh_network_error", "chatgpt_refresh_network_error"}
            else (
                "auth"
                if isinstance(exception, ChatGptAuthError)
                else (
                    "timeout"
                    if isinstance(exception, (TimeoutError, httpx.TimeoutException))
                    else (
                        "network"
                        if isinstance(exception, httpx.HTTPError)
                        else "transaction" if isinstance(exception, OSError) else "unknown"
                    )
                )
            )
        ),
        visibility="basic",
        content_policy="metadata_only",
        status_code=status_code,
        retry_after_ms=None,
        fields={
            "phase": phase,
            **fields,
            **(safe_exception_fields(exception) if exception is not None else {}),
        },
    )


@dataclass(frozen=True, slots=True)
class ChatGptSessionSignOutResult:
    remote_revoked: bool
    local_credentials_removed: bool
    diagnostics: ErrorDiagnostics | None = None
    secondary_diagnostics: tuple[ErrorDiagnostics, ...] = ()
    secondary_omitted_count: int = 0

    def __post_init__(self) -> None:
        excess = max(0, len(self.secondary_diagnostics) - 4)
        if excess:
            object.__setattr__(self, "secondary_diagnostics", self.secondary_diagnostics[:4])
            object.__setattr__(
                self, "secondary_omitted_count", self.secondary_omitted_count + excess
            )


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
    signed_in: bool | None
    email: str | None
    diagnostics: ErrorDiagnostics | None = None


@dataclass(slots=True)
class ChatGptSession:
    secret_store: Callable[[], SecretStore]
    http_factory: Callable[[], httpx.AsyncClient] = lambda: network_clients.external_async_client(
        timeout=20.0
    )
    clock: Callable[[], float] = time.time
    refresh_margin_s: float = _REFRESH_MARGIN_S
    _tokens: ChatGptTokenSet | None = field(init=False, default=None, repr=False)
    _lock: asyncio.Lock | None = field(init=False, default=None, repr=False)
    _http: httpx.AsyncClient | None = field(init=False, default=None, repr=False)
    _generation: int = field(init=False, default=0, repr=False)
    _refresh_phase: str = field(init=False, default="secret_read", repr=False)

    @property
    def token_generation(self) -> int:
        return self._generation

    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = self.http_factory()
        return self._http

    def status(self) -> ChatGptSessionStatus:
        try:
            store = self.secret_store()
            signed_in = bool(store.get(CHATGPT_REFRESH_TOKEN_SECRET)) and bool(
                store.get(CHATGPT_CLIENT_ID_SECRET)
            )
            return ChatGptSessionStatus(signed_in=signed_in, email=self._stored_email(store))
        except Exception as exc:
            diagnostics = _session_diagnostics(
                "chatgpt_secret_read_failed",
                "secret_observation",
                exc,
                observation_succeeded=False,
            )
            emit_failure(logger, diagnostics, severity="warning", disposition="degraded")
            return ChatGptSessionStatus(None, None, diagnostics)

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
        before = {
            key: store.get(key)
            for key in (
                CHATGPT_CLIENT_ID_SECRET,
                CHATGPT_REFRESH_TOKEN_SECRET,
                CHATGPT_ACCOUNT_SECRET,
            )
        }
        try:
            store.set(CHATGPT_CLIENT_ID_SECRET, client_id)
            store.set(CHATGPT_REFRESH_TOKEN_SECRET, tokens.refresh_token)
            store.set(
                CHATGPT_ACCOUNT_SECRET,
                json.dumps({"subject": identity.subject, "email": identity.email}),
            )
        except Exception as exc:
            primary = _session_diagnostics(
                "chatgpt_credential_store_failed", "credential_store", exc
            )
            secondary = []
            for key, value in before.items():
                try:
                    if value is None:
                        store.delete(key)
                    else:
                        store.set(key, value)
                except Exception as restore_exc:
                    secondary.append(
                        replace(
                            _session_diagnostics(
                                "chatgpt_credential_restore_failed", "restore_secret", restore_exc
                            ),
                            cause_id=primary.diagnostic_id,
                        )
                    )
            raise ChatGptAuthError(
                "credential_store_failed",
                diagnostics=primary,
                secondary_diagnostics=tuple(secondary),
            ) from exc
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

    async def sign_out(self) -> ChatGptSessionSignOutResult:
        primary = None
        secondary: list[ErrorDiagnostics] = []
        revoked = False
        removed = False
        client_id = None
        refresh_token = None
        store = None
        try:
            store = self.secret_store()
            client_id = store.get(CHATGPT_CLIENT_ID_SECRET)
            refresh_token = store.get(CHATGPT_REFRESH_TOKEN_SECRET)
        except Exception as exc:
            primary = _session_diagnostics("chatgpt_secret_read_failed", "secret_read", exc)
            emit_failure(logger, primary)
            if store is None:
                self._tokens = None
                self._generation += 1
                return ChatGptSessionSignOutResult(False, False, primary)
        if client_id and refresh_token:
            try:
                result = await revoke_refresh_token(
                    self.http(), client_id=client_id, refresh_token=refresh_token
                )
                revoked = result.revoked
                if not revoked:
                    primary = _session_diagnostics(
                        "chatgpt_remote_revocation_unconfirmed",
                        "remote_revoke",
                        status_code=result.status_code,
                        remote_revoked=False,
                    )
            except Exception as exc:
                primary = _session_diagnostics(
                    "chatgpt_remote_revocation_unconfirmed",
                    "remote_revoke",
                    exc,
                    remote_revoked=False,
                )
        try:
            await asyncio.to_thread(store.delete, CHATGPT_REFRESH_TOKEN_SECRET)
            removed = True
        except Exception as exc:
            cause = _session_diagnostics(
                "chatgpt_local_credential_delete_failed",
                "local_credential_delete",
                exc,
                local_credentials_removed=False,
            )
            if primary is None:
                primary = cause
            else:
                secondary.append(replace(cause, cause_id=primary.diagnostic_id))
        self._tokens = None
        self._generation += 1
        if primary is not None:
            primary = replace(
                primary,
                fields={
                    **primary.fields,
                    "local_credentials_removed": removed,
                    "remote_revoked": revoked,
                },
            )
            emit_failure(logger, primary, severity="warning", disposition="degraded")
        for cause in secondary:
            emit_failure(logger, cause)
        return ChatGptSessionSignOutResult(revoked, removed, primary, tuple(secondary))

    async def close(self) -> None:
        http = self._http
        self._http = None
        if http is not None:
            await http.aclose()

    async def _refresh(self) -> ChatGptTokenSet:
        self._refresh_phase = "secret_read"
        try:
            return await self._refresh_owned()
        except ChatGptAuthError as exc:
            diagnostics = exc.diagnostics or _session_diagnostics(
                exc.code, self._refresh_phase, exc, status_code=exc.status_code
            )
            exc.diagnostics = diagnostics
            emit_failure(logger, diagnostics)
            for cause in exc.secondary_diagnostics:
                emit_failure(logger, cause)
            raise
        except Exception as exc:
            diagnostics = _session_diagnostics("chatgpt_refresh_failed", self._refresh_phase, exc)
            emit_failure(logger, diagnostics)
            raise ChatGptAuthError("refresh_failed", diagnostics=diagnostics) from exc

    async def _refresh_owned(self) -> ChatGptTokenSet:
        store = self.secret_store()
        client_id = store.get(CHATGPT_CLIENT_ID_SECRET)
        refresh_token = store.get(CHATGPT_REFRESH_TOKEN_SECRET)
        if not client_id or not refresh_token:
            raise ChatGptReauthRequired("signed_out")
        self._refresh_phase = "refresh_request"
        try:
            payload = await refresh_access_token(
                self.http(), client_id=client_id, refresh_token=refresh_token
            )
        except ChatGptReauthRequired as exc:
            primary = _session_diagnostics(
                exc.code,
                "refresh_rejected",
                exc,
                status_code=exc.status_code,
                local_credentials_removed=False,
            )
            self._tokens = None
            self._generation += 1
            try:
                await asyncio.to_thread(store.delete, CHATGPT_REFRESH_TOKEN_SECRET)
            except Exception as delete_exc:
                exc.secondary_diagnostics = (
                    replace(
                        _session_diagnostics(
                            "chatgpt_local_credential_delete_failed",
                            "local_credential_delete",
                            delete_exc,
                            local_credentials_removed=False,
                        ),
                        cause_id=primary.diagnostic_id,
                    ),
                )
            else:
                primary = replace(
                    primary, fields={**primary.fields, "local_credentials_removed": True}
                )
            exc.diagnostics = primary
            raise
        except httpx.HTTPError as exc:
            raise ChatGptAuthError("refresh_network_error") from exc
        self._refresh_phase = "refresh_response"
        tokens = token_set_from_response(
            payload, previous_refresh_token=refresh_token, now=self.clock()
        )
        if not tokens.plan_usage_granted:
            raise ChatGptReauthRequired("plan_scope_missing")
        self._refresh_phase = "refresh_store"
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
