from __future__ import annotations

import asyncio
import logging
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import httpx

from puripuly_heart.app.ports.chatgpt_account import (
    ChatGptAccountSnapshot,
    ChatGptConnectResult,
    ChatGptSignOutResult,
)
from puripuly_heart.core.chatgpt.loopback import (
    ChatGptOAuthLoopbackClosedError,
    ChatGptOAuthLoopbackListener,
    bind_chatgpt_loopback,
)
from puripuly_heart.core.chatgpt.oauth import (
    ChatGptAuthError,
    ChatGptIdTokenVerifier,
    build_authorization_request,
    exchange_authorization_code,
    token_set_from_response,
)
from puripuly_heart.core.chatgpt.session import ChatGptSession
from puripuly_heart.core.diagnostics import emit_failure, safe_exception_fields
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
    _listener: ChatGptOAuthLoopbackListener | None = field(init=False, default=None, repr=False)
    _authorization_url: str | None = field(init=False, default=None, repr=False)
    _phase: str = field(init=False, default="sign_in", repr=False)
    _cleanup_diagnostics: tuple[ErrorDiagnostics, ...] = field(init=False, default=(), repr=False)
    _pending_listeners: dict[int, ChatGptOAuthLoopbackListener] = field(
        init=False, default_factory=dict, repr=False
    )

    @property
    def in_progress(self) -> bool:
        return self._listener is not None

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
        if self._listener is not None:
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
        self._phase = "secret_read"
        self._cleanup_diagnostics = ()
        try:
            result = await self._connect(
                open_browser=open_browser,
                authorization_url_sink=authorization_url_sink,
            )
            for cause in result.secondary_diagnostics:
                emit_failure(logger, cause, severity="warning", disposition="degraded")
            return result
        except ChatGptOAuthLoopbackClosedError:
            for cause in self._cleanup_diagnostics:
                emit_failure(logger, cause, severity="warning", disposition="degraded")
            return ChatGptConnectResult(
                succeeded=False,
                failure_code="cancelled",
                secondary_diagnostics=self._cleanup_diagnostics,
            )
        except Exception as exc:
            code = (
                exc.code
                if isinstance(exc, ChatGptAuthError)
                else (
                    "timeout"
                    if isinstance(exc, TimeoutError)
                    else (
                        "loopback_unavailable"
                        if isinstance(exc, OSError) and self._phase == "loopback_bind"
                        else "network_error" if isinstance(exc, httpx.HTTPError) else "failed"
                    )
                )
            )
            primary = (
                exc.diagnostics if isinstance(exc, ChatGptAuthError) else None
            ) or ErrorDiagnostics(
                component="chatgpt_account",
                operation=self._phase,
                code=f"chatgpt_{code}",
                category=(
                    "auth"
                    if isinstance(exc, ChatGptAuthError)
                    else (
                        "timeout"
                        if isinstance(exc, TimeoutError)
                        else (
                            "network"
                            if isinstance(exc, httpx.HTTPError)
                            else "lifecycle" if isinstance(exc, OSError) else "unknown"
                        )
                    )
                ),
                visibility="basic",
                content_policy="metadata_only",
                status_code=exc.status_code if isinstance(exc, ChatGptAuthError) else None,
                retry_after_ms=None,
                fields={"phase": self._phase, **safe_exception_fields(exc)},
            )
            secondary = (
                *(exc.secondary_diagnostics if isinstance(exc, ChatGptAuthError) else ()),
                *(
                    replace(cause, cause_id=primary.diagnostic_id)
                    for cause in self._cleanup_diagnostics
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
        listener = self._listener
        if listener is not None:
            listener.close()

    async def close(self) -> None:
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

    async def _connect(
        self,
        *,
        open_browser: bool,
        authorization_url_sink: Callable[[str], None] | None,
    ) -> ChatGptConnectResult:
        saved_client_id = self.session.client_id()
        self._phase = "loopback_bind"
        listener = self.listener_factory(locale=self.locale_provider())
        self._listener = listener
        try:
            self._phase = "authorization_request"
            request = build_authorization_request(
                redirect_uri=listener.redirect_uri,
                host_id=self.session.host_id(),
                client_id=saved_client_id,
                login_hint=self.session.login_hint() if saved_client_id else None,
            )
            self._authorization_url = request.url
            if authorization_url_sink is not None:
                self._phase = "authorization_notification"
                authorization_url_sink(request.url)
            if open_browser:
                self._phase = "browser_open"
                self.browser_opener(request.url)
            self._phase = "authorization_wait"
            callback = await asyncio.to_thread(listener.wait, self.timeout_s)
        finally:
            self._listener = None
            self._authorization_url = None
            try:
                listener.close()
            except Exception as exc:
                self._pending_listeners[id(listener)] = listener
                self._cleanup_diagnostics = (
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

        if callback.state != request.state:
            raise ChatGptAuthError("state_mismatch")
        if callback.error is not None:
            raise ChatGptAuthError(
                "access_denied" if callback.error == "access_denied" else "authorization_failed"
            )
        if callback.code is None:
            raise ChatGptAuthError("authorization_failed")
        if saved_client_id is None:
            client_id = callback.client_id
            if not client_id:
                raise ChatGptAuthError("registration_incomplete")
        else:
            if callback.client_id and callback.client_id != saved_client_id:
                raise ChatGptAuthError("client_mismatch")
            client_id = saved_client_id

        http = self.session.http()
        self._phase = "token_exchange"
        payload = await exchange_authorization_code(
            http,
            client_id=client_id,
            code=callback.code,
            code_verifier=request.code_verifier,
            redirect_uri=request.redirect_uri,
        )
        tokens = token_set_from_response(payload)
        if tokens.id_token is None:
            raise ChatGptAuthError("invalid_id_token")
        self._phase = "identity_verification"
        identity = await self.verifier_factory(http).verify(
            tokens.id_token, client_id=client_id, nonce=request.nonce
        )
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
            secondary_diagnostics=self._cleanup_diagnostics,
        )


__all__ = [
    "CHATGPT_SIGN_IN_TIMEOUT_S",
    "ChatGptAccountOwner",
]
