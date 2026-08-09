from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import Final

from puripuly_heart.core.diagnostic_validation import (
    BROKER_RAW_MESSAGE_KEYS,
    BROKER_RAW_MESSAGE_REDACTION_MARKER,
    DIAGNOSTIC_REDACTION_MARKER,
    DIAGNOSTIC_SINK_BASIC_LOGS,
    DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED,
    LOCAL_LLM_EXTRA_BODY_REDACTION_MARKER,
    PROVIDER_RESPONSE_BODY_KEYS,
    PROVIDER_RESPONSE_BODY_REDACTION_MARKER,
    DiagnosticSink,
    validate_diagnostics_for_sink,
)
from puripuly_heart.core.messages import (
    CONTENT_POLICY_METADATA_ONLY,
    DIAGNOSTIC_CATEGORIES,
    DIAGNOSTIC_CATEGORY_AUTH,
    DIAGNOSTIC_CATEGORY_INVALID_RESPONSE,
    DIAGNOSTIC_CATEGORY_LIFECYCLE,
    DIAGNOSTIC_CATEGORY_NETWORK,
    DIAGNOSTIC_CATEGORY_QUOTA,
    DIAGNOSTIC_CATEGORY_RATE_LIMIT,
    DIAGNOSTIC_CATEGORY_SERVICE_UNAVAILABLE,
    DIAGNOSTIC_CATEGORY_TIMEOUT,
    DIAGNOSTIC_CATEGORY_UNKNOWN,
    DIAGNOSTIC_VISIBILITY_BASIC,
    SEVERITY_ERROR,
    DiagnosticCategory,
    DiagnosticFieldValue,
    ErrorDiagnostics,
    UserErrorReport,
    UserMessageRef,
)


def _legacy_raw_key_assignment_re(keys: frozenset[str]) -> re.Pattern[str]:
    alternatives = sorted(
        (_legacy_raw_key_pattern(key) for key in keys),
        key=len,
        reverse=True,
    )
    return re.compile(r"(?is)\b(?:" + "|".join(alternatives) + r")\b[\"']?\s*[=:]\s*")


def _legacy_raw_key_pattern(key: str) -> str:
    return r"[_ -]?".join(re.escape(part) for part in key.split("_"))


_STATUS_CODE_RE: Final = re.compile(r"(?i)\bstatus(?:_code)?\s*[=:]\s*(\d{3})\b")
_LEGACY_RAW_ERROR_TEXT_MAX_LENGTH: Final = 512
_PROVIDER_RESPONSE_TEXT_RE: Final = _legacy_raw_key_assignment_re(PROVIDER_RESPONSE_BODY_KEYS)
_BROKER_RAW_TEXT_RE: Final = _legacy_raw_key_assignment_re(BROKER_RAW_MESSAGE_KEYS)
_LOCAL_LLM_EXTRA_BODY_TEXT_RE: Final = re.compile(
    r"(?is)\b(?:local[_ -]?(?:llm|openai)[_ -]?extra[_ -]?body)\b\s*[=:]\s*"
)
_STACK_TRACE_BLOCK_RE: Final = re.compile(r"(?is)\n?Traceback \(most recent call last\):.*")
_STACK_FRAME_LINE_RE: Final = re.compile(r'(?im)^\s*File "[^"]+", line \d+.*$')
_SECRET_ASSIGNMENT_RE: Final = re.compile(
    r"(?i)\b(api[_-]?key|authorization|password|private[_-]?key|secret|session[_-]?token|token)\b[\"']?\s*[:=]\s*[\"']?(?:Bearer\s+[A-Za-z0-9._~+\-/]{8,}|[^\s\"',;}]+)"
)
_BEARER_SECRET_RE: Final = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+\-/]{8,}")
_OPENAI_STYLE_SECRET_RE: Final = re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9._-]{8,}\b")
_PRIVATE_KEY_BLOCK_RE: Final = re.compile(
    r"(?is)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----"
)


def provider_failure_report(
    exc: BaseException | None,
    *,
    provider: str,
    operation: str,
) -> UserErrorReport:
    return _failure_report(
        exc,
        surface="provider",
        message_key="provider.failure",
        component="provider.llm",
        provider=provider,
        operation=operation,
        extra_fields={},
    )


def stt_failure_report(
    exc: BaseException | None,
    *,
    provider: str,
    operation: str,
    channel: str,
    attempts: int | None = None,
) -> UserErrorReport:
    extra_fields: dict[str, DiagnosticFieldValue] = {"channel": _safe_label(channel, "unknown")}
    if attempts is not None:
        extra_fields["attempts"] = attempts
    return _failure_report(
        exc,
        surface="stt",
        message_key="stt.failure",
        component="provider.stt",
        provider=provider,
        operation=operation,
        extra_fields=extra_fields,
    )


def format_error_report_for_log(
    report: UserErrorReport,
    *,
    sink: DiagnosticSink = DIAGNOSTIC_SINK_BASIC_LOGS,
) -> str:
    validation = validate_diagnostics_for_sink(report.diagnostics, sink)
    if validation.status != DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED or validation.diagnostics is None:
        return "category=unknown code=diagnostics.rejected diagnostic_status=rejected"

    diagnostics = validation.diagnostics
    parts = [
        f"category={diagnostics.category}",
        f"code={diagnostics.code or 'unknown'}",
    ]
    if diagnostics.status_code is not None:
        parts.append(f"status={diagnostics.status_code}")
    for key in (
        "managed_operation",
        "managed_code",
        "managed_error_class",
        "managed_subcode",
    ):
        value = diagnostics.fields.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    if diagnostics.retry_after_ms is not None:
        parts.append(f"retry_after_ms={diagnostics.retry_after_ms}")
    return " ".join(parts)


def sanitize_legacy_raw_user_visible_error_text(value: object) -> str | None:
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return None

    text = _redact_raw_assignment_values(
        text,
        _PROVIDER_RESPONSE_TEXT_RE,
        PROVIDER_RESPONSE_BODY_REDACTION_MARKER,
    )
    text = _redact_raw_assignment_values(
        text,
        _BROKER_RAW_TEXT_RE,
        BROKER_RAW_MESSAGE_REDACTION_MARKER,
    )
    text = _redact_raw_assignment_values(
        text,
        _LOCAL_LLM_EXTRA_BODY_TEXT_RE,
        LOCAL_LLM_EXTRA_BODY_REDACTION_MARKER,
    )
    text = _STACK_TRACE_BLOCK_RE.sub(DIAGNOSTIC_REDACTION_MARKER, text)
    text = _STACK_FRAME_LINE_RE.sub(DIAGNOSTIC_REDACTION_MARKER, text)
    text = _PRIVATE_KEY_BLOCK_RE.sub(DIAGNOSTIC_REDACTION_MARKER, text)
    text = _SECRET_ASSIGNMENT_RE.sub(_redact_secret_assignment, text)
    text = _BEARER_SECRET_RE.sub(f"Bearer {DIAGNOSTIC_REDACTION_MARKER}", text)
    text = _OPENAI_STYLE_SECRET_RE.sub(DIAGNOSTIC_REDACTION_MARKER, text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    if len(text) > _LEGACY_RAW_ERROR_TEXT_MAX_LENGTH:
        return text[: _LEGACY_RAW_ERROR_TEXT_MAX_LENGTH - 3].rstrip() + "..."
    return text


def _redact_secret_assignment(match: re.Match[str]) -> str:
    return f"{match.group(1)}={DIAGNOSTIC_REDACTION_MARKER}"


def _redact_raw_assignment_values(
    text: str,
    pattern: re.Pattern[str],
    marker: str,
) -> str:
    redacted_parts: list[str] = []
    cursor = 0
    for match in pattern.finditer(text):
        if match.start() < cursor:
            continue
        redacted_parts.append(text[cursor : match.start()])
        redacted_parts.append(marker)
        cursor = _raw_assignment_value_end(text, match.end())
    redacted_parts.append(text[cursor:])
    return "".join(redacted_parts)


def _raw_assignment_value_end(text: str, start: int) -> int:
    if start >= len(text):
        return start
    char = text[start]
    if char in ("{", "["):
        return _balanced_raw_value_end(text, start)
    if char in ("'", '"'):
        return _quoted_raw_value_end(text, start)

    end = start
    while end < len(text) and text[end] not in "\n;":
        end += 1
    return end


def _balanced_raw_value_end(text: str, start: int) -> int:
    closing_for_open = {"{": "}", "[": "]"}
    expected_closers = [closing_for_open[text[start]]]
    index = start + 1
    while index < len(text):
        char = text[index]
        if char in ("'", '"'):
            index = _quoted_raw_value_end(text, index)
            continue
        if char in closing_for_open:
            expected_closers.append(closing_for_open[char])
        elif char == expected_closers[-1]:
            expected_closers.pop()
            if not expected_closers:
                return index + 1
        index += 1
    return len(text)


def _quoted_raw_value_end(text: str, start: int) -> int:
    quote = text[start]
    index = start + 1
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == quote:
            return index + 1
        index += 1
    return len(text)


def _failure_report(
    exc: BaseException | None,
    *,
    surface: str,
    message_key: str,
    component: str,
    provider: str,
    operation: str,
    extra_fields: Mapping[str, DiagnosticFieldValue],
) -> UserErrorReport:
    provider_label = _diagnostic_provider(exc, provider)
    operation_label = _safe_label(operation, "unknown")
    status_code = _status_code(exc)
    category = _classify_failure(exc, status_code=status_code)
    fields: dict[str, DiagnosticFieldValue] = dict(extra_fields)
    fields["exception_type"] = type(exc).__name__ if exc is not None else "UnknownError"
    fields["provider"] = provider_label
    fields.update(_managed_diagnostic_fields(exc))
    return UserErrorReport(
        message=UserMessageRef(
            key=message_key,
            params={
                "category": category,
                "operation": operation_label,
                "provider": provider_label,
            },
            severity=SEVERITY_ERROR,
        ),
        diagnostics=ErrorDiagnostics(
            component=component,
            operation=operation_label,
            code=f"{surface}.{category}",
            category=category,
            visibility=DIAGNOSTIC_VISIBILITY_BASIC,
            content_policy=CONTENT_POLICY_METADATA_ONLY,
            status_code=status_code,
            retry_after_ms=_retry_after_ms(exc),
            fields=fields,
        ),
    )


def _classify_failure(
    exc: BaseException | None,
    *,
    status_code: int | None,
) -> DiagnosticCategory:
    text = _exception_chain_text(exc)
    if status_code in (401, 403):
        return DIAGNOSTIC_CATEGORY_AUTH
    if status_code == 408:
        return DIAGNOSTIC_CATEGORY_TIMEOUT
    if status_code == 429:
        if "quota" in text:
            return DIAGNOSTIC_CATEGORY_QUOTA
        return DIAGNOSTIC_CATEGORY_RATE_LIMIT
    if status_code in (500, 502, 503, 504):
        return DIAGNOSTIC_CATEGORY_SERVICE_UNAVAILABLE
    if status_code is not None and 400 <= status_code < 500:
        return DIAGNOSTIC_CATEGORY_INVALID_RESPONSE

    for item in _exception_chain(exc):
        explicit_category = getattr(item, "diagnostic_category", None)
        if explicit_category in DIAGNOSTIC_CATEGORIES:
            return explicit_category

    if _is_timeout_exception(exc) or "timeout" in text or "timed out" in text:
        return DIAGNOSTIC_CATEGORY_TIMEOUT
    if isinstance(exc, ConnectionError | OSError) or any(
        marker in text for marker in ("connection", "network", "socket", "dns")
    ):
        return DIAGNOSTIC_CATEGORY_NETWORK
    if any(marker in text for marker in ("unauthorized", "forbidden", "api key", "auth")):
        return DIAGNOSTIC_CATEGORY_AUTH
    if "quota" in text:
        return DIAGNOSTIC_CATEGORY_QUOTA
    if any(marker in text for marker in ("rate limit", "rate_limit", "too many requests")):
        return DIAGNOSTIC_CATEGORY_RATE_LIMIT
    if any(marker in text for marker in ("unavailable", "overloaded", "bad gateway")):
        return DIAGNOSTIC_CATEGORY_SERVICE_UNAVAILABLE
    if any(
        marker in text
        for marker in (
            "invalid json",
            "not valid json",
            "malformed",
            "did not contain",
            "empty message content",
            "truncated",
        )
    ):
        return DIAGNOSTIC_CATEGORY_INVALID_RESPONSE
    if any(marker in text for marker in ("not configured", "closed", "cancelled")):
        return DIAGNOSTIC_CATEGORY_LIFECYCLE
    return DIAGNOSTIC_CATEGORY_UNKNOWN


def _diagnostic_provider(exc: BaseException | None, fallback: str) -> str:
    for item in _exception_chain(exc):
        provider = getattr(item, "diagnostic_provider", None)
        if isinstance(provider, str) and provider.strip():
            return _safe_label(provider, "unknown")
    return _safe_label(fallback, "unknown")


def _status_code(exc: BaseException | None) -> int | None:
    for item in _exception_chain(exc):
        status = getattr(item, "status_code", None)
        if isinstance(status, int):
            return status
        response = getattr(item, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status
        match = _STATUS_CODE_RE.search(str(item))
        if match:
            return int(match.group(1))
    return None


def _retry_after_ms(exc: BaseException | None) -> int | None:
    for item in _exception_chain(exc):
        retry_after = getattr(item, "retry_after_ms", None)
        if isinstance(retry_after, int):
            return retry_after
        diagnostics = getattr(item, "diagnostics", None)
        diagnostic_retry_after = getattr(diagnostics, "retry_after_ms", None)
        if isinstance(diagnostic_retry_after, int):
            return diagnostic_retry_after
    return None


def _managed_diagnostic_fields(exc: BaseException | None) -> dict[str, DiagnosticFieldValue]:
    fields: dict[str, DiagnosticFieldValue] = {}
    for item in _exception_chain(exc):
        diagnostics = getattr(item, "diagnostics", None)
        if diagnostics is None:
            continue
        _add_optional_field(fields, "managed_operation", getattr(diagnostics, "operation", None))
        _add_optional_field(fields, "managed_code", getattr(diagnostics, "code", None))
        _add_optional_field(
            fields,
            "managed_error_class",
            getattr(diagnostics, "error_class", None),
        )
        _add_optional_field(fields, "managed_subcode", getattr(diagnostics, "subcode", None))
        break
    return fields


def _add_optional_field(
    fields: dict[str, DiagnosticFieldValue],
    key: str,
    value: object,
) -> None:
    if isinstance(value, str) and value.strip():
        fields[key] = _safe_label(value, "unknown")


def _exception_chain_text(exc: BaseException | None) -> str:
    return " ".join(_exception_text_for_classification(item) for item in _exception_chain(exc))


def _exception_text_for_classification(exc: BaseException) -> str:
    if hasattr(exc, "message_key") and hasattr(exc, "diagnostics"):
        diagnostics = getattr(exc, "diagnostics", None)
        parts = [type(exc).__name__, str(getattr(exc, "message_key", ""))]
        for attr in ("operation", "code", "error_class", "subcode"):
            value = getattr(diagnostics, attr, None)
            if isinstance(value, str):
                parts.append(value)
        return " ".join(parts).lower()
    return str(exc).lower()


def _exception_chain(exc: BaseException | None) -> Iterator[BaseException]:
    seen: set[int] = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        cause = current.__cause__
        current = cause if cause is not None else current.__context__


def _is_timeout_exception(exc: BaseException | None) -> bool:
    return any("timeout" in type(item).__name__.lower() for item in _exception_chain(exc))


def _safe_label(value: object, default: str) -> str:
    text = str(value or "").strip()
    return text[:64] if text else default


__all__ = [
    "format_error_report_for_log",
    "provider_failure_report",
    "sanitize_legacy_raw_user_visible_error_text",
    "stt_failure_report",
]
