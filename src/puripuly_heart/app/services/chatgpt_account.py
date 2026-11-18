from __future__ import annotations

import asyncio
import logging
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from urllib.parse import urlsplit

import httpx

from puripuly_heart.app.ports.chatgpt_account import (
    ChatGptAccountSnapshot,
    ChatGptConnectResult,
    ChatGptSignOutResult,
)
from puripuly_heart.core.chatgpt.loopback import (
    ChatGptOAuthCallback,
    ChatGptOAuthLoopbackClosedError,
    ChatGptOAuthLoopbackListener,
    bind_chatgpt_loopback,
)
from puripuly_heart.core.chatgpt.oauth import (
    CHATGPT_DYNAMIC_CLIENT_ID,
    ChatGptAuthError,
    ChatGptAuthorizationRequest,
    ChatGptIdTokenVerifier,
    build_authorization_request,
    exchange_authorization_code,
    token_set_from_response,
)
from puripuly_heart.core.chatgpt.session import ChatGptSession
from puripuly_heart.core.diagnostics import (
    emit_failure,
    exception_diagnostics,
    safe_exception_fields,
)
from puripuly_heart.core.messages import ErrorDiagnostics

logger = logging.getLogger(__name__)

CHATGPT_SIGN_IN_TIMEOUT_S = 600.0


@dataclass(slots=True)
class ChatGptAccountOwner:
    session: ChatGptSession
    locale_provider: Callable[[], str | None] = lambda: None
    browser_opener: Callable[[str], object] = webbrowser.open
    listener_factory: Callable[..., ChatGptOAuthLoopbackListener] = bind_chatgpt_loopback
    verifier_factory: Callable[[httpx.AsyncClient], ChatGptIdTokenVerifier] = ChatGptIdTokenVerifier
    timeout_s: float = CHATGPT_SIGN_IN_TIMEOUT_S
    _sign_in_active: bool = field(init=False, default=False, repr=False)
    _cancel_requested: bool = field(init=False, default=False, repr=False)
    _listener: ChatGptOAuthLoopbackListener | None = field(init=False, default=None, repr=False)
    _authorization_url: str | None = field(init=False, default=None, repr=False)
    _phase: str = field(init=False, default="sign_in", repr=False)
    _cleanup_diagnostics: tuple[ErrorDiagnostics, ...] = field(init=False, default=(), repr=False)
    _exchange_fields: dict[str, object] = field(init=False, default_factory=dict, repr=False)
    _retry_diagnostic: ErrorDiagnostics | None = field(init=False, default=None, repr=False)
    _pending_listeners: dict[int, ChatGptOAuthLoopbackListener] = field(
        init=False, default_factory=dict, repr=False
    )

    @property
    def in_progress(self) -> bool:
        return self._sign_in_active

    def snapshot(self) -> ChatGptAccountSnapshot:
        status = self.session.status()
        return ChatGptAccountSnapshot(
            signed_in=status.signed_in,
            email=status.email,
            in_progress=self.in_progress,
            diagnostics=status.diagnostics,
        )

    def reopen_authorization_url(self) -> bool:
        url = self._authorization_url
        if url is None:
            return False
        self.browser_opener(url)
        return True

    async def connect(
        self,
        *,
        open_browser: bool = True,
        authorization_url_sink: Callable[[str], None] | None = None,
    ) -> ChatGptConnectResult:
        if self.in_progress:
            secondary = ()
            if open_browser:
                try:
                    self.reopen_authorization_url()
                except Exception as exc:
                    diagnostic = ErrorDiagnostics(
                        component="chatgpt_account",
                        operation="browser_open",
                        code="chatgpt_browser_open_failed",
                        category="lifecycle",
                        visibility="basic",
                        content_policy="metadata_only",
                        status_code=None,
                        retry_after_ms=None,
                        fields={"phase": "browser_open", **safe_exception_fields(exc)},
                    )
                    emit_failure(logger, diagnostic, severity="warning", disposition="degraded")
                    secondary = (diagnostic,)
            return ChatGptConnectResult(
                succeeded=False, failure_code="in_progress", secondary_diagnostics=secondary
            )
        self._sign_in_active = True
        self._cancel_requested = False
        self._phase = "secret_read"
        self._cleanup_diagnostics = ()
        self._retry_diagnostic = None
        try:
            result = await self._connect(
                open_browser=open_browser,
                authorization_url_sink=authorization_url_sink,
            )
            secondary = self._secondary_diagnostics("succeeded")
            for cause in secondary:
                emit_failure(logger, cause, severity="warning", disposition="degraded")
            return replace(result, secondary_diagnostics=secondary) if secondary else result
        except Exception as exc:
            if isinstance(exc, ChatGptOAuthLoopbackClosedError) or (
                self._cancel_requested and self._phase != "credential_store"
            ):
                secondary = self._secondary_diagnostics("cancelled")
                for cause in secondary:
                    emit_failure(logger, cause, severity="warning", disposition="degraded")
                return ChatGptConnectResult(
                    succeeded=False, failure_code="cancelled", secondary_diagnostics=secondary
                )
            if isinstance(exc, ChatGptAuthError):
                code = exc.code
            elif isinstance(exc, TimeoutError):
                code = "timeout"
            elif isinstance(exc, OSError) and self._phase == "loopback_bind":
                code = "loopback_unavailable"
            elif isinstance(exc, httpx.HTTPError):
                code = "network_error"
            else:
                code = "failed"
            primary = self._failure_diagnostic(exc, code)
            secondary = (
                *(exc.secondary_diagnostics if isinstance(exc, ChatGptAuthError) else ()),
                *(
                    replace(cause, cause_id=primary.diagnostic_id)
                    for cause in self._secondary_diagnostics("failed")
                ),
            )
            emit_failure(logger, primary, stage=self._phase)
            for cause in secondary:
                emit_failure(logger, cause, severity="warning", disposition="degraded")
            return ChatGptConnectResult(
                succeeded=False,
                failure_code=code,
                diagnostics=primary,
                secondary_diagnostics=secondary,
                secondary_omitted_count=(
                    exc.secondary_omitted_count if isinstance(exc, ChatGptAuthError) else 0
                ),
            )
        finally:
            self._sign_in_active = False

    def _secondary_diagnostics(self, retry_outcome: str) -> tuple[ErrorDiagnostics, ...]:
        diagnostic = self._retry_diagnostic
        if diagnostic is None:
            return self._cleanup_diagnostics
        return (
            replace(diagnostic, fields={**diagnostic.fields, "retry_outcome": retry_outcome}),
            *self._cleanup_diagnostics,
        )

    def _failure_diagnostic(self, exc: Exception, code: str) -> ErrorDiagnostics:
        diagnostic = exc.diagnostics if isinstance(exc, ChatGptAuthError) else None
        fields = self._exchange_fields if self._phase == "token_exchange" else {}
        if diagnostic is None:
            diagnostic = exception_diagnostics(
                exc,
                component="chatgpt_account",
                operation=self._phase,
                code=f"chatgpt_{code}",
                category=(
                    "auth"
                    if isinstance(exc, ChatGptAuthError)
                    else "lifecycle" if isinstance(exc, (OSError, httpx.HTTPError)) else "unknown"
                ),
                fields={"phase": self._phase, **fields},
            )
        elif fields:
            diagnostic = replace(diagnostic, fields={**diagnostic.fields, **fields})
        if isinstance(exc, ChatGptAuthError) and exc.status_code is not None:
            diagnostic = replace(diagnostic, status_code=exc.status_code)
        return diagnostic

    async def sign_out(self) -> ChatGptSignOutResult:
        result = await self.session.sign_out()
        return ChatGptSignOutResult(
            remote_revoked=result.remote_revoked,
            local_credentials_removed=result.local_credentials_removed,
            diagnostics=result.diagnostics,
            secondary_diagnostics=result.secondary_diagnostics,
            secondary_omitted_count=result.secondary_omitted_count,
        )

    def cancel(self) -> None:
        if self.in_progress:
            self._cancel_requested = True
        listener = self._listener
        if listener is not None:
            listener.close()

    async def close(self) -> None:
        self._cancel_requested = True
        failures: list[Exception] = []
        listeners = dict(self._pending_listeners)
        if self._listener is not None:
            listeners[id(self._listener)] = self._listener
        for identity, listener in listeners.items():
            try:
                listener.close()
            except Exception as exc:
                self._pending_listeners[identity] = listener
                diagnostic = ErrorDiagnostics(
                    component="chatgpt_account",
                    operation="watch_close",
                    code="chatgpt_loopback_close_failed",
                    category="lifecycle",
                    visibility="basic",
                    content_policy="metadata_only",
                    status_code=None,
                    retry_after_ms=None,
                    fields={
                        "phase": "watch_close",
                        "cleanup_confirmed": False,
                        **safe_exception_fields(exc),
                    },
                )
                emit_failure(logger, diagnostic, disposition="failed")
                failure = ChatGptAuthError("loopback_close_failed", diagnostics=diagnostic)
                failure.__cause__ = exc
                failures.append(failure)
            else:
                self._pending_listeners.pop(identity, None)
                if listener is self._listener:
                    self._listener = None
                    self._authorization_url = None
        try:
            await self.session.close()
        except Exception as exc:
            failures.append(exc)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ExceptionGroup("ChatGPT account shutdown failed", failures)

    def _check_cancelled(self) -> None:
        if self._cancel_requested:
            raise ChatGptOAuthLoopbackClosedError("ChatGPT sign-in was cancelled")

    async def _connect(
        self,
        *,
        open_browser: bool,
        authorization_url_sink: Callable[[str], None] | None,
    ) -> ChatGptConnectResult:
        saved_client_id = self.session.client_id()
        client_id = saved_client_id
        login_hint = self.session.login_hint() if saved_client_id else None
        host_id = self.session.host_id()
        deadline = asyncio.get_running_loop().time() + self.timeout_s
        attempt = 1
        while True:
            self._check_cancelled()
            request, callback = await self._authorize(
                client_id=client_id,
                host_id=host_id,
                login_hint=login_hint,
                deadline=deadline,
                open_browser=open_browser,
                authorization_url_sink=authorization_url_sink,
            )
            self._check_cancelled()
            if callback.state != request.state:
                raise ChatGptAuthError("state_mismatch")
            if callback.error is not None:
                raise ChatGptAuthError(
                    "access_denied" if callback.error == "access_denied" else "authorization_failed"
                )
            if callback.code is None:
                raise ChatGptAuthError("authorization_failed")
            if client_id is None:
                client_id = callback.client_id
                if not client_id or client_id == CHATGPT_DYNAMIC_CLIENT_ID:
                    raise ChatGptAuthError("registration_incomplete")
            elif callback.client_id and callback.client_id != client_id:
                raise ChatGptAuthError("client_mismatch")

            http = self.session.http()
            self._phase = "token_exchange"
            self._exchange_fields = {
                "client_id_reused": request.client_id is not None,
                "loopback_port": urlsplit(request.redirect_uri).port,
                "authorization_attempt": attempt,
            }
            try:
                payload = await exchange_authorization_code(
                    http,
                    client_id=client_id,
                    code=callback.code,
                    code_verifier=request.code_verifier,
                    redirect_uri=request.redirect_uri,
                )
            except ChatGptAuthError as exc:
                if exc.code != "invalid_grant" or attempt == 2:
                    raise
                self._retry_diagnostic = self._failure_diagnostic(exc, exc.code)
                attempt += 1
            else:
                break

        self._check_cancelled()
        tokens = token_set_from_response(payload)
        if tokens.id_token is None:
            raise ChatGptAuthError("invalid_id_token")
        self._phase = "identity_verification"
        identity = await self.verifier_factory(http).verify(
            tokens.id_token, client_id=client_id, nonce=request.nonce
        )
        self._check_cancelled()
        if not tokens.plan_usage_granted:
            raise ChatGptAuthError("plan_scope_missing")
        self._phase = "credential_store"
        await asyncio.to_thread(
            self.session.store_sign_in,
            client_id=client_id,
            tokens=tokens,
            identity=identity,
        )
        return ChatGptConnectResult(
            succeeded=True,
            first_sign_in=saved_client_id is None,
            email=identity.email,
        )

    async def _authorize(
        self,
        *,
        client_id: str | None,
        host_id: str,
        login_hint: str | None,
        deadline: float,
        open_browser: bool,
        authorization_url_sink: Callable[[str], None] | None,
    ) -> tuple[ChatGptAuthorizationRequest, ChatGptOAuthCallback]:
        clock = asyncio.get_running_loop().time
        self._phase = "authorization_wait"
        if deadline <= clock():
            raise TimeoutError
        self._phase = "loopback_bind"
        listener = self.listener_factory(locale=self.locale_provider())
        self._listener = listener
        try:
            self._phase = "authorization_request"
            request = build_authorization_request(
                redirect_uri=listener.redirect_uri,
                host_id=host_id,
                client_id=client_id,
                login_hint=login_hint,
            )
            self._authorization_url = request.url
            if authorization_url_sink is not None:
                self._phase = "authorization_notification"
                authorization_url_sink(request.url)
            self._check_cancelled()
            if open_browser:
                self._phase = "browser_open"
                self.browser_opener(request.url)
            self._phase = "authorization_wait"
            callback = await asyncio.to_thread(listener.wait, max(0.0, deadline - clock()))
        finally:
            self._listener = None
            self._authorization_url = None
            try:
                listener.close()
            except Exception as exc:
                self._pending_listeners[id(listener)] = listener
                self._cleanup_diagnostics = (
                    *self._cleanup_diagnostics,
                    ErrorDiagnostics(
                        component="chatgpt_account",
                        operation="watch_close",
                        code="chatgpt_loopback_close_failed",
                        category="lifecycle",
                        visibility="basic",
                        content_policy="metadata_only",
                        status_code=None,
                        retry_after_ms=None,
                        fields={
                            "phase": "watch_close",
                            "cleanup_confirmed": False,
                            **safe_exception_fields(exc),
                        },
                    ),
                )
        return request, callback


__all__ = [
    "CHATGPT_SIGN_IN_TIMEOUT_S",
    "ChatGptAccountOwner",
]
