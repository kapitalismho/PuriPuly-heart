from __future__ import annotations

import re
import ssl
from typing import Literal

import httpx

from puripuly_heart.core.messages import DiagnosticCategory
from puripuly_heart.core.network_diagnostics import (
    classify_transport_error,
    safe_transport_fields,
    transport_exception_chain,
)

OpenRouterAuthStage = Literal[
    "listener_start",
    "browser_launch",
    "callback_wait",
    "code_exchange",
    "key_verification",
    "settings_commit",
    "flow",
]
OpenRouterAuthReason = Literal[
    "credential_rejected",
    "access_denied",
    "http",
    "rate_limit",
    "service_unavailable",
    "tls",
    "proxy",
    "network",
    "timeout",
    "callback_timeout",
    "setup",
    "invalid_response",
    "unknown",
]
_CATEGORIES: dict[OpenRouterAuthReason, DiagnosticCategory] = {
    "credential_rejected": "auth",
    "access_denied": "invalid_response",
    "http": "invalid_response",
    "rate_limit": "rate_limit",
    "service_unavailable": "service_unavailable",
    "tls": "network",
    "proxy": "network",
    "network": "network",
    "timeout": "timeout",
    "callback_timeout": "timeout",
    "setup": "lifecycle",
    "invalid_response": "invalid_response",
    "unknown": "unknown",
}


class OpenRouterAuthenticationError(RuntimeError):
    diagnostic_provider = "openrouter"

    def __init__(
        self,
        *,
        stage: OpenRouterAuthStage,
        reason: OpenRouterAuthReason,
        status_code: int | None = None,
        exception_type: str = "UnknownError",
    ) -> None:
        self.stage = stage
        self.reason = reason
        self.status_code = status_code
        self.exception_type = (
            exception_type
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", exception_type)
            else "UnknownError"
        )
        self.diagnostic_category = _CATEGORIES[reason]
        self.message_key = f"error.openrouter_auth.{reason}"
        super().__init__(self.message_key)
        self.diagnostic_transport_fields = safe_transport_fields(self)

    @classmethod
    def from_status(
        cls, status: int, *, stage: OpenRouterAuthStage
    ) -> OpenRouterAuthenticationError:
        reason: OpenRouterAuthReason
        if status == 401:
            reason = "credential_rejected"
        elif status == 403:
            reason = "access_denied"
        elif status == 429:
            reason = "rate_limit"
        elif status in (408, 504):
            reason = "timeout"
        elif 500 <= status <= 599:
            reason = "service_unavailable"
        else:
            reason = "http"
        return cls(stage=stage, reason=reason, status_code=status, exception_type="HTTPStatusError")

    @classmethod
    def from_exception(
        cls, exc: Exception, *, stage: OpenRouterAuthStage
    ) -> OpenRouterAuthenticationError:
        if isinstance(exc, cls):
            return exc
        if isinstance(exc, httpx.HTTPStatusError):
            failure = cls.from_status(exc.response.status_code, stage=stage)
            failure.diagnostic_transport_fields = safe_transport_fields(exc)
            return failure
        transport = classify_transport_error(exc)
        reason: OpenRouterAuthReason = transport or "unknown"
        classified = next(
            (
                item
                for item in transport_exception_chain(exc)
                if isinstance(item, (ssl.SSLError, httpx.ProxyError))
            ),
            exc,
        )
        if transport == "timeout":
            reason = "callback_timeout" if stage == "callback_wait" else "timeout"
        elif transport not in ("tls", "proxy"):
            if stage in ("listener_start", "browser_launch", "settings_commit"):
                reason = "setup"
            elif isinstance(
                exc, (ImportError, ValueError, FileNotFoundError, PermissionError, httpx.InvalidURL)
            ):
                reason = "setup"
        failure = cls(stage=stage, reason=reason, exception_type=type(classified).__name__)
        failure.diagnostic_transport_fields = safe_transport_fields(exc)
        return failure
