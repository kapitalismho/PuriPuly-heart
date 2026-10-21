from __future__ import annotations

from dataclasses import replace

from websockets.exceptions import (
    ConnectionClosed,
    InvalidProxy,
    InvalidStatus,
    InvalidURI,
    ProxyError,
    WebSocketException,
)

from puripuly_heart.core.diagnostics import sanitize_diagnostics
from puripuly_heart.core.messages import DiagnosticCategory, ErrorDiagnostics
from puripuly_heart.core.network_diagnostics import (
    classify_transport_error,
    transport_exception_chain,
)
from puripuly_heart.core.stt.backend import STTProviderTurnIdentity
from puripuly_heart.core.stt.diagnostics import recognition_failure_diagnostics

_NATIVE_FAILURES: dict[str, tuple[DiagnosticCategory, bool]] = {
    "CLIENT_ERROR": ("invalid_response", False),
    "InvalidApiKey": ("auth", False),
    "AccessDenied": ("auth", False),
    "AccessDenied.Unpurchased": ("auth", False),
    "InvalidParameter": ("invalid_response", False),
    "DataInspectionFailed": ("invalid_response", False),
    "ModelNotFound": ("invalid_response", False),
    "Throttling": ("rate_limit", True),
    "Throttling.RateQuota": ("rate_limit", True),
    "Throttling.BurstRate": ("rate_limit", True),
    "Throttling.AllocationQuota": ("quota", False),
    "Throttling.ServiceOverloaded": ("service_unavailable", True),
    "Throttling.ResourceExhausted": ("service_unavailable", True),
    "InternalError": ("service_unavailable", True),
    "InternalError.Timeout": ("timeout", True),
    "SystemError": ("service_unavailable", True),
    "RequestTimeOut": ("timeout", True),
    "ServiceUnavailable": ("service_unavailable", True),
    "ModelUnavailable": ("service_unavailable", True),
}
_RETRYABLE_CLOSE_CODES = frozenset({1000, 1001, 1006, 1011, 1012, 1013})


def _close_code(frame: object) -> int | None:
    code = getattr(frame, "code", None)
    return int(code) if isinstance(code, int) and not isinstance(code, bool) else None


def _retryable_websocket_close(exc: ConnectionClosed) -> bool:
    return all(
        frame is None or _close_code(frame) in _RETRYABLE_CLOSE_CODES
        for frame in (exc.rcvd, exc.sent)
    )


def annotate_qwen_failure(
    exc: Exception,
    *,
    channel: str,
    operation: str,
    reason: str,
    identity: STTProviderTurnIdentity | None = None,
    native_code: str | None = None,
    idle_ms: int | None = None,
) -> Exception:
    chain = tuple(transport_exception_chain(exc))
    for cause in chain if native_code is None else (exc,):
        preserved = getattr(cause, "failure_diagnostics", None)
        if isinstance(preserved, ErrorDiagnostics):
            exc.failure_diagnostics = sanitize_diagnostics(preserved)
            retryable = getattr(cause, "readiness_retryable", None)
            if type(retryable) is bool:
                exc.readiness_retryable = retryable
            return exc

    diagnostics = recognition_failure_diagnostics(
        exc,
        provider="qwen_audio",
        channel=channel,
        operation=operation,
        reason=reason,
        identity=identity,
    )
    transport = classify_transport_error(exc)
    configuration_failure = any(
        isinstance(
            cause, (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError)
        )
        for cause in chain
    )
    fields = dict(diagnostics.fields)
    category = diagnostics.category
    kind = (
        "configuration"
        if configuration_failure
        else transport if transport in {"tls", "proxy"} else "transport" if transport else "unknown"
    )
    retryable: bool | None = None
    websocket_errors = tuple(cause for cause in chain if isinstance(cause, WebSocketException))
    if native_code is not None:
        known_code = (
            native_code
            if type(native_code) is str and native_code in _NATIVE_FAILURES
            else "unknown"
        )
        fields["qwen_native_code"] = known_code
        kind = "native"
        category, retryable = _NATIVE_FAILURES.get(known_code, ("unknown", False))
    else:
        close = next(
            (cause for cause in websocket_errors if isinstance(cause, ConnectionClosed)), None
        )
        if close is not None:
            received_code = _close_code(close.rcvd)
            fields["ws_close_code"] = received_code if received_code is not None else 1006
        if any(isinstance(cause, (InvalidProxy, ProxyError)) for cause in websocket_errors):
            kind, category, retryable = "proxy", "network", False
        elif any(isinstance(cause, InvalidURI) for cause in websocket_errors):
            kind, category, retryable = "configuration", "invalid_response", False
        elif any(
            not isinstance(cause, (ConnectionClosed, InvalidStatus)) for cause in websocket_errors
        ):
            kind, category, retryable = "websocket_protocol", "invalid_response", False
        elif close is not None:
            code = fields["ws_close_code"]
            retryable = all(
                _retryable_websocket_close(cause)
                for cause in websocket_errors
                if isinstance(cause, ConnectionClosed)
            )
            kind = "websocket_close"
            category = (
                "invalid_response"
                if not retryable
                else "service_unavailable" if code in {1011, 1012, 1013} else "network"
            )
        elif websocket_errors:
            kind = "websocket_handshake"
            retryable = category in {"timeout", "service_unavailable", "rate_limit"}

    if retryable is not None:
        permanent_websocket_failure = any(
            (
                not _retryable_websocket_close(cause)
                if isinstance(cause, ConnectionClosed)
                else not isinstance(cause, InvalidStatus)
            )
            for cause in websocket_errors
        )
        exc.readiness_retryable = (
            retryable
            and not configuration_failure
            and transport not in {"tls", "proxy"}
            and diagnostics.category not in {"auth", "quota", "invalid_response"}
            and not permanent_websocket_failure
        )
    fields["qwen_failure_kind"] = kind
    if type(idle_ms) is int and 0 <= idle_ms < 2**64:
        fields["idle_ms"] = idle_ms
    exc.failure_diagnostics = sanitize_diagnostics(
        replace(diagnostics, code=f"stt.{category}", category=category, fields=fields)
    )
    return exc
