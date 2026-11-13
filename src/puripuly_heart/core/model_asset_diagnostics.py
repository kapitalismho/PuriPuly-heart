from __future__ import annotations

import errno
from collections.abc import Iterator, Mapping
from dataclasses import replace

import httpx

from puripuly_heart.core.diagnostics import (
    DIAGNOSTIC_CAUSE_MAX_COUNT,
    DIAGNOSTIC_CAUSE_MAX_DEPTH,
    exception_diagnostics,
    sanitize_diagnostic_fields,
    sanitize_diagnostics,
)
from puripuly_heart.core.error_messages import provider_failure_report
from puripuly_heart.core.messages import ErrorDiagnostics


def _asset_exception_chain(error: BaseException) -> Iterator[BaseException]:
    current: BaseException | None = error
    seen: set[int] = set()
    for _ in range(DIAGNOSTIC_CAUSE_MAX_DEPTH + 1):
        if current is None or id(current) in seen:
            return
        seen.add(id(current))
        yield current
        current = current.__cause__


def model_asset_failure_code(error: BaseException) -> str:
    for item in _asset_exception_chain(error):
        if isinstance(item, httpx.TimeoutException):
            return "download_timeout"
        if isinstance(item, httpx.ProxyError):
            return "proxy_failed"
        if isinstance(item, httpx.ConnectError):
            return "connection_failed"
        if isinstance(item, httpx.HTTPStatusError):
            return "source_unavailable"
        if isinstance(item, OSError) and item.errno == errno.ENOSPC:
            return "disk_full"
        fields, _ = sanitize_diagnostic_fields(
            {"asset_failure_code": getattr(item, "failure_code", "unknown")}
        )
        code = fields.get("asset_failure_code")
        if code is not None and code != "unknown":
            return str(code)
    return "unknown"


def model_asset_failure_diagnostics(
    error: BaseException,
    *,
    component: str = "model_assets",
    operation: str = "gemma_install",
    code: str = "model_assets.install_failed",
    fields: Mapping[str, object] | None = None,
    _depth: int = 0,
) -> ErrorDiagnostics:
    preserved = getattr(error, "failure_diagnostics", None)
    if isinstance(preserved, ErrorDiagnostics):
        return preserved
    if not isinstance(error, BaseExceptionGroup):
        for linked in _asset_exception_chain(error):
            preserved = getattr(linked, "failure_diagnostics", None)
            if isinstance(preserved, ErrorDiagnostics):
                return preserved
    if isinstance(error, BaseExceptionGroup):
        native = exception_diagnostics(
            error, component=component, operation=operation, code=code, fields=fields
        )
        native = sanitize_diagnostics(replace(native, status_code=None, category="lifecycle"))
        if _depth >= DIAGNOSTIC_CAUSE_MAX_DEPTH:
            return native
        return sanitize_diagnostics(
            replace(
                native,
                causes=tuple(
                    model_asset_failure_diagnostics(
                        child,
                        component=component,
                        operation=operation,
                        code=code,
                        _depth=_depth + 1,
                    )
                    for child in error.exceptions[:DIAGNOSTIC_CAUSE_MAX_COUNT]
                ),
                causes_truncated=(
                    native.causes_truncated or len(error.exceptions) > DIAGNOSTIC_CAUSE_MAX_COUNT
                ),
            )
        )
    native = (
        provider_failure_report(error, provider="unknown", operation=operation).diagnostics
        if (
            isinstance(error, httpx.HTTPError)
            or getattr(error, "failure_code", None) is not None
            or error.__cause__ is not None
        )
        else exception_diagnostics(
            error,
            component=component,
            operation=operation,
            code=code,
            category="unknown",
            fields=fields,
        )
    )
    technical: dict[str, object] = {}
    for item in _asset_exception_chain(error):
        for attribute, key in (
            ("worker_exit_code", "exit_code"),
            ("os_error_code", "errno"),
        ):
            value = getattr(item, attribute, None)
            if key not in technical and type(value) is int:
                technical[key] = value
        cause_type, _ = sanitize_diagnostic_fields(
            {"cause_class_1": getattr(item, "cause_type", None)}
        )
        if (
            native.fields.get("cause_class_1") in {None, "unclassified"}
            and "cause_class_1" not in technical
        ):
            technical.update(cause_type)
    failure_code = model_asset_failure_code(error)
    category = native.category
    if category == "unknown":
        if failure_code == "download_timeout":
            category = "timeout"
        elif failure_code in {"proxy_failed", "connection_failed"}:
            category = "network"
        elif failure_code in {"checksum_failed", "file_size_mismatch", "range_invalid"}:
            category = "invalid_response"
    return sanitize_diagnostics(
        replace(
            native,
            component=component,
            operation=operation,
            code=code,
            category=category,
            visibility="diagnostic_only",
            fields={
                **native.fields,
                **technical,
                **(fields or {}),
                "asset_failure_code": failure_code,
            },
        )
    )
