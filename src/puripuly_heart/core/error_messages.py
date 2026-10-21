from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import replace
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
from puripuly_heart.core.diagnostics import (
    safe_exception_fields,
    sanitize_diagnostic_fields,
    sanitize_diagnostics,
)
from puripuly_heart.core.llm.fallback_racing import LLMProviderRaceError
from puripuly_heart.core.messages import (
    CONTENT_POLICY_METADATA_ONLY,
    DIAGNOSTIC_CATEGORIES,
    DIAGNOSTIC_CATEGORY_AUTH,
    DIAGNOSTIC_CATEGORY_INVALID_RESPONSE,
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
from puripuly_heart.core.network_diagnostics import (
    TRANSPORT_FIELD_KEYS,
    classify_transport_error,
    safe_transport_fields,
    transport_exception_chain,
)
from puripuly_heart.core.openrouter.authentication import (
    OpenRouterAuthenticationError,
    OpenRouterAuthStage,
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
    exc: BaseException | ErrorDiagnostics | None,
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


def openrouter_auth_failure_report(
    exc: Exception,
    *,
    stage: OpenRouterAuthStage,
) -> UserErrorReport:
    failure = OpenRouterAuthenticationError.from_exception(exc, stage=stage)
    return UserErrorReport(
        message=UserMessageRef(key=failure.message_key, params={}, severity=SEVERITY_ERROR),
        diagnostics=ErrorDiagnostics(
            component="provider.openrouter.auth",
            operation=failure.stage,
            code=f"openrouter.auth.{failure.reason}",
            category=failure.diagnostic_category,
            visibility=DIAGNOSTIC_VISIBILITY_BASIC,
            content_policy=CONTENT_POLICY_METADATA_ONLY,
            status_code=failure.status_code,
            retry_after_ms=None,
            fields={
                "provider": "openrouter",
                **safe_exception_fields(failure),
                **safe_transport_fields(failure),
            },
        ),
    )


def stt_failure_report(
    exc: BaseException | ErrorDiagnostics | None,
    *,
    provider: str,
    operation: str,
    channel: str,
    attempts: int | None = None,
) -> UserErrorReport:
    preserved = getattr(exc, "failure_diagnostics", None)
    if isinstance(preserved, ErrorDiagnostics):
        exc = preserved
    if isinstance(exc, ErrorDiagnostics):
        exc = sanitize_diagnostics(exc)
        return UserErrorReport(
            message=UserMessageRef(
                key="stt.failure",
                params={
                    "category": exc.category,
                    "operation": exc.operation or "unknown",
                    "provider": _diagnostic_provider(
                        None, str(exc.fields.get("provider", provider))
                    ),
                },
                severity=SEVERITY_ERROR,
            ),
            diagnostics=exc,
        )
    extra_fields: dict[str, DiagnosticFieldValue] = {"channel": _safe_label(channel, "unknown")}
    if attempts is not None:
        extra_fields["attempts"] = attempts
    report = _failure_report(
        exc,
        surface="stt",
        message_key="stt.failure",
        component="provider.stt",
        provider=provider,
        operation=operation,
        extra_fields=extra_fields,
    )
    return replace(
        report,
        diagnostics=replace(
            report.diagnostics, channel=channel if channel in {"self", "peer"} else None
        ),
    )


def format_error_report_for_log(
    report: UserErrorReport,
    *,
    sink: DiagnosticSink = DIAGNOSTIC_SINK_BASIC_LOGS,
) -> str:
    diagnostics = sanitize_diagnostics(report.diagnostics)
    validation = validate_diagnostics_for_sink(diagnostics, sink)
    if validation.status != DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED or validation.diagnostics is None:
        return (
            f"category={diagnostics.category} code={diagnostics.code} "
            f"component={diagnostics.component} operation={diagnostics.operation or 'unknown'} "
            "diagnostic_status=rejected"
        )
    diagnostics = validation.diagnostics
    parts = [
        f"category={diagnostics.category}",
        f"code={diagnostics.code or 'unknown'}",
        f"component={diagnostics.component}",
        f"operation={diagnostics.operation or 'unknown'}",
    ]
    if diagnostics.status_code is not None:
        parts.append(f"status={diagnostics.status_code}")
    for key in (
        "provider",
        "channel",
        "exception_type",
        "managed_operation",
        "managed_code",
        "managed_error_class",
        "managed_subcode",
        *TRANSPORT_FIELD_KEYS,
    ):
        value = diagnostics.fields.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    if diagnostics.retry_after_ms is not None:
        parts.append(f"retry_after_ms={diagnostics.retry_after_ms}")
    if diagnostics.rejected_fields:
        parts.append(f"diagnostic_status=rejected rejected_fields={diagnostics.rejected_fields}")
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
    exc: BaseException | ErrorDiagnostics | None,
    *,
    surface: str,
    message_key: str,
    component: str,
    provider: str,
    operation: str,
    extra_fields: Mapping[str, DiagnosticFieldValue],
) -> UserErrorReport:
    preserved = (
        exc if isinstance(exc, ErrorDiagnostics) else getattr(exc, "failure_diagnostics", None)
    )
    if preserved is None:
        preserved = getattr(exc, "diagnostics", None)
    if isinstance(preserved, ErrorDiagnostics):
        if isinstance(exc, BaseException):
            status = preserved.status_code or _status_code(exc)
            category = (
                _classify_failure(exc, status_code=status)
                if preserved.category == DIAGNOSTIC_CATEGORY_UNKNOWN
                else preserved.category
            )
            preserved = replace(
                preserved,
                status_code=status,
                category=category,
                code=(
                    f"{surface}.{category}"
                    if preserved.code == f"{surface}.unknown"
                    and category != DIAGNOSTIC_CATEGORY_UNKNOWN
                    else preserved.code
                ),
                fields={
                    **preserved.fields,
                    **safe_exception_fields(exc),
                    **safe_transport_fields(exc),
                },
            )
        preserved = sanitize_diagnostics(preserved)
        provider_label = _diagnostic_provider(None, str(preserved.fields.get("provider", provider)))
        if preserved.status_code == 402 and provider_label == "openrouter":
            message_key = _openrouter_402_message_key(preserved.fields)
        if provider_label == "chatgpt" and isinstance(exc, BaseException):
            message_key = _chatgpt_message_key(exc, preserved.status_code) or message_key
        return UserErrorReport(
            message=UserMessageRef(
                key=message_key,
                params={
                    "category": preserved.category,
                    "operation": preserved.operation or "unknown",
                    "provider": _diagnostic_provider(
                        None, str(preserved.fields.get("provider", provider))
                    ),
                },
                severity=SEVERITY_ERROR,
            ),
            diagnostics=preserved,
        )
    operation_label = _safe_label(operation, "unknown")
    race = next(
        (item for item in _exception_chain(exc) if isinstance(item, LLMProviderRaceError)),
        None,
    )
    if race is not None:
        branch_reports = tuple(
            provider_failure_report(error, provider=provider, operation=operation)
            for error in race.errors
        )
        matching = bool(branch_reports) and all(
            report.diagnostics.category == branch_reports[0].diagnostics.category
            and report.diagnostics.status_code == branch_reports[0].diagnostics.status_code
            and report.message.params["provider"] == branch_reports[0].message.params["provider"]
            for report in branch_reports
        )
        unanimous_402 = (
            matching
            and branch_reports[0].diagnostics.status_code == 402
            and branch_reports[0].message.params["provider"] == "openrouter"
        )
        provider_label = (
            str(branch_reports[0].message.params["provider"])
            if matching
            else _safe_label(provider, "unknown")
        )
        status_code = branch_reports[0].diagnostics.status_code if matching else None
        category = (
            branch_reports[0].diagnostics.category if matching else DIAGNOSTIC_CATEGORY_UNKNOWN
        )
        message_key = (
            branch_reports[0].message.key
            if unanimous_402
            and all(
                report.message.key == branch_reports[0].message.key for report in branch_reports
            )
            else "provider.openrouter.payment_required" if unanimous_402 else "provider.failure"
        )
        if (
            matching
            and branch_reports[0].message.key.startswith("provider.chatgpt.")
            and all(
                report.message.key == branch_reports[0].message.key for report in branch_reports
            )
        ):
            message_key = branch_reports[0].message.key
        retry_values = {report.diagnostics.retry_after_ms for report in branch_reports}
        retry_after_ms = retry_values.pop() if matching and len(retry_values) == 1 else None
        shared_fields = {}
        if unanimous_402:
            for key in ("limit_source", "limit_reason"):
                values = {report.diagnostics.fields.get(key) for report in branch_reports}
                if len(values) == 1:
                    value = values.pop()
                    if value is not None:
                        shared_fields[key] = value
    else:
        provider_label = _diagnostic_provider(exc, provider)
        status_code = _status_code(exc)
        category = _classify_failure(exc, status_code=status_code)
        retry_after_ms = _retry_after_ms(exc)
        shared_fields = _openrouter_limit_fields(exc) if status_code == 402 else {}
        if status_code == 402 and provider_label == "openrouter":
            message_key = _openrouter_402_message_key(shared_fields)
        if provider_label == "chatgpt":
            message_key = _chatgpt_message_key(exc, status_code) or message_key
    fields: dict[str, DiagnosticFieldValue] = dict(extra_fields)
    fields["exception_class"] = (
        str(safe_exception_fields(exc)["exception_class"]) if exc is not None else "unclassified"
    )
    fields["provider"] = provider_label
    if exc is not None:
        fields.update(safe_exception_fields(exc))
        fields.update(safe_transport_fields(exc))
    fields.update(_managed_diagnostic_fields(exc))
    fields.update(shared_fields)
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
        diagnostics=sanitize_diagnostics(
            ErrorDiagnostics(
                component=component,
                operation=operation_label,
                code=f"{surface}.{category}",
                category=category,
                visibility=DIAGNOSTIC_VISIBILITY_BASIC,
                content_policy=CONTENT_POLICY_METADATA_ONLY,
                status_code=status_code,
                retry_after_ms=retry_after_ms,
                fields=fields,
            )
        ),
    )


def _openrouter_limit_fields(exc: BaseException | None) -> dict[str, DiagnosticFieldValue]:
    for item in _exception_chain(exc):
        if getattr(item, "diagnostic_provider", None) != "openrouter":
            continue
        fields: dict[str, DiagnosticFieldValue] = {}
        source = getattr(item, "limit_source", None)
        reason = getattr(item, "limit_reason", None)
        if source in ("openrouter_credits", "openrouter_key_limit", "openrouter_in_flight_budget"):
            fields["limit_source"] = source
        if reason in ("in_flight_budget_exhausted", "weight_exceeds_budget"):
            fields["limit_reason"] = reason
        return fields
    return {}


def _chatgpt_message_key(exc: BaseException | None, status_code: int | None) -> str | None:
    has_subscription_code = False
    for item in _exception_chain(exc):
        if getattr(item, "chatgpt_reauth_required", False):
            return "provider.chatgpt.reauth_required"
        code = getattr(item, "subscription_code", None)
        if code:
            has_subscription_code = True
        if code == "subscription_sharing_usage_limit_exceeded":
            return "provider.chatgpt.usage_limit"
        if code in ("subscription_sharing_user_not_eligible", "chatpass_v2_scope_not_authorized"):
            return "provider.chatgpt.not_eligible"
    if status_code == 401:
        return "provider.chatgpt.reauth_required"
    if not has_subscription_code:
        if status_code == 403:
            return "provider.chatgpt.not_eligible"
        if status_code == 429:
            return "provider.chatgpt.usage_limit"
    return None


def _openrouter_402_message_key(fields: Mapping[str, DiagnosticFieldValue]) -> str:
    source = fields.get("limit_source")
    reason = fields.get("limit_reason")
    if reason == "weight_exceeds_budget":
        return "provider.openrouter.payment_required"
    if reason == "in_flight_budget_exhausted":
        if source not in (None, "openrouter_in_flight_budget"):
            return "provider.openrouter.payment_required"
        return "provider.openrouter.temporary_limit"
    if source == "openrouter_in_flight_budget":
        return "provider.openrouter.temporary_limit"
    if source == "openrouter_key_limit":
        return "provider.openrouter.key_limit"
    if source == "openrouter_credits":
        return "provider.openrouter.insufficient_credits"
    return "provider.openrouter.payment_required"


def _classify_failure(
    exc: BaseException | None,
    *,
    status_code: int | None,
) -> DiagnosticCategory:
    explicit_categories = tuple(
        category
        for item in _exception_chain(exc)
        if isinstance(category := getattr(item, "diagnostic_category", None), str)
        and category in DIAGNOSTIC_CATEGORIES
        and category != DIAGNOSTIC_CATEGORY_UNKNOWN
    )
    if status_code in (401, 403):
        return DIAGNOSTIC_CATEGORY_AUTH
    if status_code == 408:
        return DIAGNOSTIC_CATEGORY_TIMEOUT
    if status_code == 429:
        if DIAGNOSTIC_CATEGORY_QUOTA in explicit_categories:
            return DIAGNOSTIC_CATEGORY_QUOTA
        return DIAGNOSTIC_CATEGORY_RATE_LIMIT
    if status_code in (500, 502, 503, 504):
        return DIAGNOSTIC_CATEGORY_SERVICE_UNAVAILABLE
    if status_code == 402:
        return DIAGNOSTIC_CATEGORY_QUOTA
    if status_code is not None and 400 <= status_code < 500:
        return DIAGNOSTIC_CATEGORY_INVALID_RESPONSE

    if explicit_categories:
        return explicit_categories[0]

    transport = classify_transport_error(exc) if exc is not None else None
    if transport == "timeout":
        return DIAGNOSTIC_CATEGORY_TIMEOUT
    if transport is not None:
        return DIAGNOSTIC_CATEGORY_NETWORK
    return DIAGNOSTIC_CATEGORY_UNKNOWN


def _diagnostic_provider(exc: BaseException | None, fallback: str) -> str:
    for item in _exception_chain(exc):
        provider = getattr(item, "diagnostic_provider", None)
        if isinstance(provider, str) and provider.strip():
            fields, _rejected = sanitize_diagnostic_fields({"provider": provider})
            if "provider" in fields:
                return str(fields["provider"])
    fields, _rejected = sanitize_diagnostic_fields({"provider": fallback})
    return str(fields.get("provider", "unknown"))


def _status_code(exc: BaseException | None) -> int | None:
    for item in _exception_chain(exc):
        status = getattr(item, "status_code", None)
        if type(status) is int and 100 <= status <= 599:
            return status
        native_status = getattr(item, "http_status", None)
        if type(native_status) is int and 100 <= native_status <= 599:
            return native_status
        response = getattr(item, "response", None)
        response_status = getattr(response, "status_code", None)
        if type(response_status) is int and 100 <= response_status <= 599:
            return response_status
        if type(item).__module__.startswith(("google.genai.errors", "google.api_core.exceptions")):
            sdk_status = getattr(item, "code", None)
            if type(sdk_status) is int and 100 <= sdk_status <= 599:
                return sdk_status
    return None


def _retry_after_ms(exc: BaseException | None) -> int | None:
    for item in _exception_chain(exc):
        retry_after = getattr(item, "retry_after_ms", None)
        if type(retry_after) is int and 0 <= retry_after < 2**31:
            return retry_after
        diagnostics = getattr(item, "diagnostics", None)
        diagnostic_retry_after = getattr(diagnostics, "retry_after_ms", None)
        if type(diagnostic_retry_after) is int and 0 <= diagnostic_retry_after < 2**31:
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


def _exception_chain(exc: BaseException | None) -> Iterator[BaseException]:
    yield from transport_exception_chain(exc)


def _safe_label(value: object, default: str) -> str:
    text = str(value or "").strip()
    return text[:64] if text else default


__all__ = [
    "format_error_report_for_log",
    "openrouter_auth_failure_report",
    "provider_failure_report",
    "sanitize_legacy_raw_user_visible_error_text",
    "stt_failure_report",
]
